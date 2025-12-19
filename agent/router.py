"""
Agent 路由与编排层（核心逻辑）

本文件负责把用户输入（文本/图片）转成“该调用哪些能力”的决策，并编排执行顺序：

1) 文本输入：使用“混合路由”
   - 首选：主 LLM 基于 few-shot 样例输出结构化 JSON（是否追问、是否需要 RAG、检索 query）
   - 兜底：关键词启发式（FOLLOWUP_MARKERS / NEED_IMAGE_MARKERS / FACTUAL_MARKERS）

2) 图片输入：采用“图像优先”的策略
   - 先用视觉工具（ClassifierTool）做初步识别，输出候选/特征/置信度/解释
   - 若置信度不足：用 RAG 根据候选+视觉描述检索资料，再交给主 LLM 输出“最终判定 JSON”

注意：
- 这里的“路由器”不是 LangChain/AutoTool 那种完全动态工具选择（虽然也能扩展到那一步），
  当前目标是：在可控、可解释的基础上引入 LLM 的语义判断能力。
- 本文件只负责“选择/编排”和“把结构化结果合成回复”，具体工具实现分别位于：
  - agent/tools/classifier.py（视觉模型识别）
  - agent/rag.py（检索引擎）
  - agent/llm.py（Ollama LLM 封装）
"""

import logging
import json
import asyncio
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

from agent.core import get_agent
from agent.rag import get_rag_engine

logger = logging.getLogger(__name__)


#
# 规则兜底：关键词启发式（在 LLM 路由失败/不可用时保证系统仍能工作）
#
# - FOLLOWUP_MARKERS：追问/指代词，常见于“刚才那只…它…”这种依赖上下文的问题
# - NEED_IMAGE_MARKERS：用户在“让系统识别/分类”的明确意图，但当前请求可能没带图
#
FOLLOWUP_MARKERS = ("刚才", "这只", "它", "上一只", "那个", "刚刚", "刚刚那只", "刚刚那個")
NEED_IMAGE_MARKERS = ("这是什么", "识别", "分类", "这只是什么", "这是什么鸟", "帮我识别", "帮我分类")

FACTUAL_MARKERS = (
    "是什么", "有哪些", "特征", "习性", "分布", "吃什么", "饲养", "寿命", "性格", "价格", "注意", "怎么养", "喂", "能不能", "区别"
)

# few-shot 路由 prompt：让主 LLM 用“样例 + 当前输入”输出路由 JSON
# 约束：必须只输出 JSON（便于 _safe_json_extract 解析）
ROUTE_FEWSHOT = """
你是一个路由器，任务是：根据用户输入判断是否为“追问上一只识别对象”，以及是否需要调用知识检索（RAG）。
只输出 JSON，不要输出其它文字。

输出 JSON 结构：
{
  "is_followup": true/false,
  "use_rag": true/false,
  "search_query": "用于检索的简短查询（可为空）"
}

样例1：
last_species=玄凤鹦鹉
用户：刚才那只吃什么？
输出：{"is_followup": true, "use_rag": true, "search_query": "玄凤鹦鹉 吃什么 饮食"}

样例2：
last_species=蓝黄金刚鹦鹉
用户：它适合新手养吗？
输出：{"is_followup": true, "use_rag": true, "search_query": "蓝黄金刚鹦鹉 新手 饲养 注意事项"}

样例3：
last_species=null
用户：你好
输出：{"is_followup": false, "use_rag": false, "search_query": ""}

样例4：
last_species=null
用户：玄凤鹦鹉的特征是什么？
输出：{"is_followup": false, "use_rag": true, "search_query": "玄凤鹦鹉 特征 外观 习性"}
""".strip()


def detect_needs_image(text: str) -> bool:
    """兜底规则：判断文本是否明确表达“我要识别/分类”，但当前没有图片。"""
    t = (text or "").strip()
    if not t:
        return False
    return any(m in t for m in NEED_IMAGE_MARKERS)


def detect_followup(text: str) -> bool:
    """兜底规则：判断文本是否包含典型追问/指代词（是否依赖上下文）。"""
    t = (text or "").strip()
    if not t:
        return False
    return any(m in t for m in FOLLOWUP_MARKERS)


def extract_species_from_analyze(analysis: Any) -> Optional[str]:
    """
    辅助函数：从 analyze 结果中提取 top1 物种名。
    当前 analyze 结果类型可能来自 pydantic 模型，也可能是 dict（取决于调用方）。
    """
    try:
        return analysis.classification.top_candidates[0].name
    except Exception:
        return None


def _confidence_is_low(confidence_level: Optional[str], top_score: Optional[float]) -> bool:
    """
    决策：识图置信度是否“低到需要检索增强”。

    设计原则：
    - 如果视觉模型明确给出 Low：无条件触发（避免误判）
    - Medium：根据分值再做一次阈值判断（保守触发）
    - High：不触发（节省成本/降低无关检索噪声）
    - 若缺失：退回到 top_score 阈值判断

    参数：
    - confidence_level：视觉模型输出的置信度档位（High/Medium/Low）
    - top_score：top1 的 score（0~1）
    """
    if confidence_level:
        c = confidence_level.strip().lower()
        if c == "low":
            return True
        if c == "medium":
            # medium can still benefit from retrieval, but keep it conservative
            return (top_score or 0) < 0.65
        if c == "high":
            return False
    # Fallback to numeric score
    if top_score is None:
        return True
    return top_score < 0.75


def _safe_json_extract(text: str) -> Optional[Dict[str, Any]]:
    """
    尝试从 LLM 输出中提取 JSON 对象。

    说明：
    - 我们要求路由/终判 prompt “只输出 JSON”，但模型仍可能夹杂多余文本。
    - 这里采用简单策略：截取第一个 '{' 到最后一个 '}' 之间的子串再 json.loads。
    - 如果解析失败，调用方应走兜底逻辑（heuristics / 默认值）。
    """
    if not text:
        return None
    s = text.strip()
    start = s.find("{")
    end = s.rfind("}") + 1
    if start >= 0 and end > start:
        s = s[start:end]
    try:
        return json.loads(s)
    except Exception:
        return None


class AgentRouter:
    """
    Agent 路由器（对外提供 run_analyze / run_ask，供 app.py 调用）。

    目前的策略（混合路由）：
    - 图片：始终走图像优先流程（run_analyze），并在低置信度时引入 RAG + 主 LLM 终判
    - 文本：使用 decide_text_routing（LLM few-shot + 关键词兜底）决定是否使用 RAG

    注意：
    - “追问是否绑定 last_species”属于上层（app.py 的 /agent/message）在管理会话时做的事；
      这里的 decide_text_routing 会输出 is_followup，但上层是否有可绑定的 last_species 决定最终 hint。
    """

    def __init__(self):
        # AgentService：持有 main_llm + classifier_tool（核心依赖）
        self.agent = get_agent()

    async def decide_text_routing(self, question: str, last_species: Optional[str] = None) -> Dict[str, Any]:
        """
        Hybrid routing for text:
        - Prefer LLM semantic judgement with few-shot examples
        - Fallback to keyword heuristics on failure
        Returns: {is_followup, use_rag, search_query}

        返回值语义：
        - is_followup：是否为追问（语义判断）
        - use_rag：是否需要 RAG 才能可靠回答（语义判断 + 启发式兜底）
        - search_query：用于 RAG 的检索 query（可为空；为空时调用方应 fallback）
        """
        q = (question or "").strip()
        # Heuristic defaults
        is_followup = detect_followup(q)
        use_rag = any(m in q for m in FACTUAL_MARKERS)
        search_query = f"关于{last_species}：{q}" if last_species else q

        if not q:
            return {"is_followup": False, "use_rag": False, "search_query": ""}

        prompt = (
            ROUTE_FEWSHOT
            + "\n\n"
            + f"last_species={last_species or 'null'}\n"
            + f"用户：{q}\n"
            + "输出："
        )

        try:
            raw = await asyncio.to_thread(self.agent.main_llm._call, prompt)
            parsed = _safe_json_extract(raw)
            if parsed and isinstance(parsed, dict):
                if parsed.get("is_followup") is not None:
                    is_followup = bool(parsed.get("is_followup"))
                if parsed.get("use_rag") is not None:
                    use_rag = bool(parsed.get("use_rag"))
                if parsed.get("search_query") is not None:
                    sq = str(parsed.get("search_query")).strip()
                    if sq:
                        search_query = sq
        except Exception as e:
            logger.info(f"Text routing LLM failed, fallback to heuristics: {e}")

        return {"is_followup": is_followup, "use_rag": use_rag, "search_query": search_query}

    async def run_analyze(self, image_path: Path) -> Tuple[str, Dict[str, Any]]:
        """
        Image-first agent behavior:
        1) Classify image (vision tool)
        2) Decide whether to use RAG (based on confidence)
        3) If low confidence: retrieve by visual features + candidates, then let main LLM decide

        返回：
        - reply：面向用户的自然语言回复（已经融合了“终判结果/建议补充信息”）
        - artifacts：结构化中间产物，便于前端展示与调试
        """
        # 1) 视觉模型初步识别（在单独线程里跑，避免阻塞 event loop）
        classification = await asyncio.to_thread(self.agent.classifier_tool.run, image_path)
        classification_dict = classification.model_dump() if hasattr(classification, "model_dump") else {}

        top1 = (classification.top_candidates[0] if classification.top_candidates else None)
        top_name = getattr(top1, "name", None) or "未知物种"
        top_prob = getattr(top1, "probability", None)
        top_score = getattr(top1, "score", None)
        conf_level = getattr(classification, "confidence_level", None)
        visual_desc = getattr(classification, "visual_features_description", None) or ""
        explain = getattr(classification, "explanation", None) or ""

        # 2) 根据置信度决定是否“检索增强”
        use_rag = _confidence_is_low(conf_level, top_score)

        rag_hits: List[Dict[str, Any]] = []
        decision: Dict[str, Any] = {"final_name": top_name, "reasoning": explain, "used_rag": False}

        if use_rag:
            # 3) 检索阶段：根据候选与视觉描述构造 query，取回资料片段
            rag = get_rag_engine()
            # Query candidates + visual features
            cand_names = [c.name for c in (classification.top_candidates or [])[:3] if getattr(c, "name", None)]
            queries = []
            if cand_names:
                queries.append(f"{' / '.join(cand_names)} 这些鹦鹉的区别与识别要点是什么？")
            if visual_desc:
                queries.append(f"根据描述：{visual_desc}。这更可能是哪种鹦鹉？请给出判断依据。")
            if top_name:
                queries.append(f"{top_name} 的典型外观特征是什么？")

            # Collect hits (dedup by (source, content prefix))
            seen = set()
            for q in queries[:3]:
                for h in rag.retrieve(q, top_k=2):
                    key = (h.get("source"), (h.get("content") or "")[:80])
                    if key in seen:
                        continue
                    seen.add(key)
                    rag_hits.append(h)

            context = "\n\n".join([f"[{h.get('source','unknown')}] {h.get('content','')}".strip() for h in rag_hits if h.get("content")])

            # 4) 终判阶段：将候选 + 视觉描述 + 检索资料交给主 LLM，要求输出 JSON
            #    终判结果被解析后写入 decision 字段（final_name/reasoning/need_more_info/followup）
            decide_prompt = (
                "你是鸟类识别专家。我们对一张鹦鹉图片进行了初步分类，但置信度不高。\n"
                "请在候选物种中做最终判定，或输出“无法确定”。\n"
                "要求：只输出 JSON。\n"
                "JSON 结构：\n"
                "{\n"
                '  "final_name": "从候选中选择一个名称，或无法确定",\n'
                '  "reasoning": "简要说明依据（结合视觉描述与资料）",\n'
                '  "need_more_info": true/false,\n'
                '  "followup": "如果无法确定，告诉用户应补充什么角度的照片/信息"\n'
                "}\n\n"
                f"候选：{[{'name': c.name, 'score': c.score} for c in (classification.top_candidates or [])[:3]]}\n"
                f"视觉描述：{visual_desc}\n"
                f"初步解释：{explain}\n\n"
            )
            if context:
                decide_prompt += f"资料：\n{context}\n\n"
            decide_prompt += "输出："

            raw = ""
            try:
                raw = await asyncio.to_thread(self.agent.main_llm._call, decide_prompt)
            except Exception as e:
                logger.warning(f"Decision LLM failed: {e}")

            parsed = _safe_json_extract(raw)
            if parsed and isinstance(parsed, dict):
                decision = {
                    "final_name": parsed.get("final_name") or top_name,
                    "reasoning": parsed.get("reasoning") or explain,
                    "need_more_info": bool(parsed.get("need_more_info")) if parsed.get("need_more_info") is not None else False,
                    "followup": parsed.get("followup") or "",
                    "used_rag": True,
                }
            else:
                # JSON 解析失败：保留“已使用 RAG”的事实，但最终判定退回 top1
                decision = {"final_name": top_name, "reasoning": explain, "used_rag": True}

        # 5) 输出合成：将终判结果转成用户可读文本
        final_name = decision.get("final_name") or top_name
        reply = f"识别结果：{final_name}"
        if top_prob is not None:
            reply += f"（{top_prob}%）"
        if decision.get("reasoning"):
            reply += f"\n判定依据：{decision['reasoning']}"
        if decision.get("need_more_info") and decision.get("followup"):
            reply += f"\n\n为进一步确认：{decision['followup']}"

        artifacts: Dict[str, Any] = {
            # classification：来自视觉工具的结构化结果（候选/特征/解释/置信度）
            "classification": classification_dict,
            # rag_hits：仅在低置信度触发检索时非空
            "rag_hits": rag_hits,
            # decision：主 LLM 终判 JSON（或兜底）
            "decision": decision,
        }
        return reply, artifacts

    async def run_ask(self, question: str, top_k: int = 3, species_hint: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        文本问答入口（用于 /ask 与 /agent/message 的 text-only 分支）。

        关键点：
        - 是否启用 RAG 由 decide_text_routing 决定（LLM few-shot + 兜底）
        - 若启用 RAG：将命中片段拼接成 context，再让主 LLM 回答
        - 若不启用 RAG：直接让主 LLM 回答（减少检索噪声与成本）
        """
        q = (question or "").strip()
        if not q:
            return "请先输入问题。", {"hits": []}

        # Hybrid: let LLM decide use_rag + query, fallback to heuristics
        route = await self.decide_text_routing(q, last_species=species_hint)
        use_rag = bool(route.get("use_rag"))
        search_query = str(route.get("search_query") or (f"关于{species_hint}：{q}" if species_hint else q)).strip()

        hits: List[Dict[str, Any]] = []
        context = ""
        if use_rag:
            rag = get_rag_engine()
            hits = rag.retrieve(search_query, top_k=top_k)
            context = "\n\n".join(
                [f"[{h.get('source','unknown')}] {h.get('content','')}".strip() for h in hits if h.get("content")]
            )

        # Use Agent main LLM to answer with citations context
        prompt = (
            "你是 Parrot Set 的鹦鹉知识助手。请用中文回答用户问题。\n"
            "规则：\n"
            "1) 优先基于“资料”作答，不要编造。\n"
            "2) 如果资料不足，请明确说明资料不足，并给出下一步建议（例如上传鹦鹉图片或补充知识库文档）。\n"
            "3) 回答尽量简洁、可执行。\n\n"
        )
        if context:
            prompt += f"资料：\n{context}\n\n"
        prompt += f"问题：{q}\n\n回答："

        answer_text = ""
        try:
            # main_llm call can block; keep it in thread via caller if needed
            answer_text = await __import__("asyncio").to_thread(self.agent.main_llm._call, prompt)
            answer_text = (answer_text or "").strip()
        except Exception as e:
            logger.warning(f"LLM answer failed, fallback to retrieval-only: {e}")

        if not answer_text:
            if hits:
                bullets = "\n".join([f"- {h.get('content','').strip()}" for h in hits if h.get("content")])
                answer_text = f"我在知识库中找到了以下相关内容（供参考）：\n{bullets}".strip()
            else:
                answer_text = "我暂时无法从知识库中检索到直接相关的内容。你可以上传鹦鹉图片让我先识别物种，或补充知识库文档后再问。"

        # artifacts 中返回：是否使用检索 + 最终 query + hits（便于前端展示与调试）
        artifacts = {"hits": hits, "rag_used": use_rag, "rag_query": search_query}
        return answer_text, artifacts

    @staticmethod
    def summarize_artifacts_for_debug(artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract high-signal debug info without returning huge payloads.
        """
        info: Dict[str, Any] = {}
        try:
            top1 = (artifacts.get("classification") or {}).get("top_candidates", [{}])[0]
            info["top1"] = {
                "name": top1.get("name"),
                "probability": top1.get("probability"),
                "score": top1.get("score"),
            }
        except Exception:
            pass
        try:
            hits = artifacts.get("knowledge_hits") or {}
            info["kb_hit_sources"] = {k: (v or {}).get("source") for k, v in hits.items()}
        except Exception:
            pass
        return info



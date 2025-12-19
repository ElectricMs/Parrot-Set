"""
AgentService（工具与模型的初始化容器）

本模块提供一个“可复用的服务对象”，负责：
- 读取 agent 配置（agent/config.py）
- 初始化主 LLM（文本模型，用于路由、问答、终判）
- 初始化视觉分类工具（agent/tools/classifier.py）

设计说明：
- 该类本身提供 `classify()` 与 `analyze()` 两个入口：
  - classify：仅做视觉识别（并返回解释/置信度/视觉特征）
  - analyze：旧版流水线：classify + search_knowledge（RAG 命中）
- 当前更通用的编排逻辑在 `agent/router.py`（/agent/message 的图片/文本路由、低置信度检索增强等）。
"""

import logging
import asyncio
from pathlib import Path
from typing import Dict, Any

from .llm import get_llm_instance
from .config import load_agent_config
from .tools.classifier import ClassifierTool
from .tools.search import search_knowledge
from .models import ClassificationResult, AnalyzeResult

logger = logging.getLogger(__name__)

class AgentService:
    def __init__(self):
        # 1) 加载配置：来自项目根目录 config.json 的 agent_settings（有默认值与向后兼容）
        self.config = load_agent_config()
        
        main_model = self.config.get("main_model", "qwen2.5:7b-instruct")
        main_temp = float(self.config.get("main_temperature", 0.5))
        
        logger.info(f"Initializing AgentService with main model: {main_model}")
        
        # 2) 初始化主 LLM（通常为纯文本模型）：用于问答、路由、终判等
        self.main_llm = get_llm_instance(main_model, main_temp)
        
        # 3) 初始化工具：ClassifierTool（多模态模型，负责识图/特征抽取/解释/置信度）
        cls_conf = self.config.get("tools", {}).get("classifier", {})
        self.classifier_tool = ClassifierTool(cls_conf)
        
    async def classify(self, image_path: Path) -> ClassificationResult:
        """
        仅做图片分类（视觉模型输出 + 数据库对齐）。
        说明：ClassifierTool 是同步实现，这里用 asyncio.to_thread 放到线程池，避免阻塞事件循环。
        """
        return await asyncio.to_thread(self.classifier_tool.run, image_path)
        
    async def analyze(self, image_path: Path) -> AnalyzeResult:
        """
        旧版分析流水线：Classify -> Search（RAG）。
        注意：
        - 该接口保留用于兼容旧前端/脚本。
        - 更“Agent 化”的流程（低置信度触发 RAG + 主 LLM 终判）在 agent/router.py。
        """
        # 1. Classify (Get candidates, features, and explanation)
        logger.info("Agent Step 1: Classify & Analyze Image")
        classification = await self.classify(image_path)
        
        # 2. Search Knowledge
        logger.info("Agent Step 2: Search Knowledge")
        hits = search_knowledge(classification.top_candidates)
        
        # Explanation is already in classification result
        explanation = classification.explanation or "无法生成解释"
        
        return AnalyzeResult(
            classification=classification,
            knowledge_hits=hits,
            explanation=explanation
        )

# Singleton instance management
_agent_instance = None

def get_agent() -> AgentService:
    """
    获取 AgentService 单例。
    说明：FastAPI 作为长驻进程时，单例可以避免重复加载模型/初始化工具。
    """
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = AgentService()
    return _agent_instance

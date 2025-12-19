import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from ..models import ClassificationResult, TopCandidate, VisualFeatures
from ..llm import get_llm_instance
from parrot_db import get_database

logger = logging.getLogger(__name__)

class ClassifierTool:
    def __init__(self, config: Dict[str, Any]):
        # 视觉模型名称（Ollama 中的模型 tag），需要支持 images 输入
        self.model_name = config.get("model_name", "qwen3-vl:2b-instruct-q4_K_M")
        # temperature 越低越“稳定/保守”，适合结构化 JSON 输出
        self.temperature = config.get("temperature", 0.1)
        self.llm = get_llm_instance(self.model_name, self.temperature)
        logger.info(f"ClassifierTool initialized with model: {self.model_name}")

    def run(self, image_path: Path) -> ClassificationResult:
        """
        Classify parrot image using multimodal LLM.
        Returns classification, visual description, explanation and confidence.

        输出约定：
        - 通过 prompt 强制模型只输出 JSON
        - JSON 字段会被解析并映射到 Pydantic 模型（ClassificationResult/TopCandidate/VisualFeatures）

        稳健性：
        - 如果模型输出不是合法 JSON，会返回一个 candidates 为空的 ClassificationResult，并保留 raw_text 以便排错。
        """
        prompt = """
分析这张鹦鹉图片，输出 JSON 格式结果：

{
  "top_candidates": [
    {"name": "物种名(中文)", "score": 概率(0-1)}
  ],
  "visual_features": {
    "description": "对鸟类外观的自然语言描述（颜色、体型、喙、羽毛特征等）",
    "colors": ["主要颜色"],
    "crown": "冠羽特征",
    "beak": "喙特征",
    "patterns": ["斑纹特征"]
  },
  "explanation": "简短解释为什么判定为该物种（依据视觉特征）",
  "confidence_level": "High/Medium/Low"
}

要求：
1. top_candidates 至少包含1个候选。
2. description 是一段连贯的中文描述。
3. confidence_level 必须是 High, Medium 或 Low。
4. 只返回 JSON，不要其他文字。
""".strip()

        logger.info(f"Using LLM to classify image: {image_path}")
        raw_text = self.llm._call(prompt, images=[str(image_path)])
        
        # Parse JSON
        try:
            clean_text = raw_text.strip()
            start = clean_text.find("{")
            end = clean_text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(clean_text[start:end])
            else:
                data = json.loads(clean_text)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e}")
            # Try to recover partial data if possible or return empty
            return ClassificationResult(
                top_candidates=[],
                visual_features=None,
                visual_features_description=None,
                raw_text=raw_text
            )

        # Convert to models and enrich with DB info
        # 说明：这里会尝试用 parrot_db 对齐物种名，补全英文名/学名/科目/链接等信息。
        db = get_database()
        candidates = []
        for c in data.get("top_candidates", []):
            name = c.get("name", "").strip()
            if not name: continue
            
            record = db.find_by_name(name)
            
            db_info = None
            if record:
                db_info = {
                    "chinese_name": record['chinese_name'],
                    "english_name": record['english_name'],
                    "scientific_name": record['scientific_name'],
                    "order": record['order'],
                    "family": record['family'],
                    "link": record['link'],
                }
                
            candidates.append(TopCandidate(
                name=name,
                score=float(c.get("score", 0)),
                # probability 为展示字段（百分比），便于 UI 直接显示
                probability=round(float(c.get("score", 0)) * 100, 2),
                exists_in_db=record is not None,
                db_info=db_info
            ))
            
        vf = data.get("visual_features") or {}
        visual = VisualFeatures(
            description=vf.get("description"),
            colors=vf.get("colors"),
            crown=vf.get("crown"),
            beak=vf.get("beak"),
            patterns=vf.get("patterns"),
            notes=vf.get("notes"),
        )
        
        # Fallback for confidence level if not provided
        # 如果模型未返回 confidence_level，则用 top1 score 推导一个档位
        confidence = data.get("confidence_level")
        if not confidence and candidates:
            score = candidates[0].score
            if score >= 0.8: confidence = "High"
            elif score >= 0.5: confidence = "Medium"
            else: confidence = "Low"
            
        return ClassificationResult(
            top_candidates=candidates,
            visual_features=visual,
            visual_features_description=visual.to_description() if visual else None,
            explanation=data.get("explanation"),
            confidence_level=confidence,
            raw_text=raw_text.strip()
        )

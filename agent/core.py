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
        # Load configuration
        self.config = load_agent_config()
        
        main_model = self.config.get("main_model", "qwen2.5:7b-instruct")
        main_temp = float(self.config.get("main_temperature", 0.5))
        
        logger.info(f"Initializing AgentService with main model: {main_model}")
        
        # 1. Initialize Main LLM (Pure text)
        self.main_llm = get_llm_instance(main_model, main_temp)
        
        # 2. Initialize Tools
        # Classifier (Vision Tool - Now handles classification + basic explanation)
        cls_conf = self.config.get("tools", {}).get("classifier", {})
        self.classifier_tool = ClassifierTool(cls_conf)
        
    async def classify(self, image_path: Path) -> ClassificationResult:
        """
        Run classification tool (includes visual description and explanation).
        """
        # Run in thread pool
        return await asyncio.to_thread(self.classifier_tool.run, image_path)
        
    async def analyze(self, image_path: Path) -> AnalyzeResult:
        """
        Run full analysis pipeline: Classify -> Search.
        (Explanation is now part of Classify)
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
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = AgentService()
    return _agent_instance

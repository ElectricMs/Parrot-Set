from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class TopCandidate(BaseModel):
    """Candidate species information"""
    name: str
    score: float
    probability: float
    exists_in_db: bool = False
    db_info: Optional[Dict[str, Any]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Blue-and-yellow Macaw",
                "score": 0.87,
                "probability": 87.0,
                "exists_in_db": True,
                "db_info": {}
            }
        }

class VisualFeatures(BaseModel):
    """Visual features extracted from image"""
    description: Optional[str] = None # 模型生成的自然语言描述
    colors: Optional[List[str]] = None
    crown: Optional[str] = None
    beak: Optional[str] = None
    patterns: Optional[List[str]] = None
    notes: Optional[str] = None
    
    def to_description(self) -> str:
        # 优先使用模型生成的描述
        if self.description:
            return self.description
            
        parts = []
        if self.colors: parts.append(f"颜色: {', '.join(self.colors)}")
        if self.crown: parts.append(f"冠羽: {self.crown}")
        if self.beak: parts.append(f"喙: {self.beak}")
        if self.patterns: parts.append(f"斑纹: {', '.join(self.patterns)}")
        if self.notes: parts.append(f"其他: {self.notes}")
        return " | ".join(parts) if parts else "未检测到显著特征"

class ClassificationResult(BaseModel):
    """Image classification result with explanation"""
    top_candidates: List[TopCandidate]
    visual_features: Optional[VisualFeatures] = None
    visual_features_description: Optional[str] = None
    explanation: Optional[str] = None # 预测解释
    confidence_level: Optional[str] = None # 置信度评级 (高/中/低)
    raw_text: str
    
class AnalyzeResult(BaseModel):
    """Full analysis result (Legacy support)"""
    classification: ClassificationResult
    knowledge_hits: Dict[str, Any]
    explanation: str

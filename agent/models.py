from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class TopCandidate(BaseModel):
    """
    候选物种信息（由视觉模型输出 + 数据库对齐后形成）。

    字段说明：
    - name: 物种中文名（尽量与 parrot_db 的中文名匹配）
    - score: 模型给出的概率（0~1）
    - probability: UI 展示用百分比（score*100）
    - exists_in_db/db_info: 若在 parrot_db 匹配成功，会补全结构化信息
    """
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
    """
    视觉特征（由视觉模型从图片中抽取）。

    优先策略：
    - 如果模型给了 description（自然语言描述），to_description() 会直接返回 description；
    - 否则将 colors/crown/beak/patterns 组合成简短描述。
    """
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
    """
    图片分类结果（ClassifierTool 输出）。

    常用字段：
    - top_candidates: Top-K 候选
    - visual_features_description: 用于 UI 展示的“特征摘要”
    - explanation: 判定依据（简短）
    - confidence_level: High/Medium/Low（档位置信度，供路由层做“是否触发检索增强”判断）
    - raw_text: 模型原始输出（用于排错；可能包含非 JSON 内容）
    """
    top_candidates: List[TopCandidate]
    visual_features: Optional[VisualFeatures] = None
    visual_features_description: Optional[str] = None
    explanation: Optional[str] = None # 预测解释
    confidence_level: Optional[str] = None # 置信度评级 (高/中/低)
    raw_text: str
    
class AnalyzeResult(BaseModel):
    """
    旧版分析结果（Legacy support）。

    - 由 agent/core.py::analyze 生成：classification + knowledge_hits + explanation
    - 新版 /agent/message 的图片增强流程会返回不同的 artifacts 结构（见 agent/router.py）。
    """
    classification: ClassificationResult
    knowledge_hits: Dict[str, Any]
    explanation: str

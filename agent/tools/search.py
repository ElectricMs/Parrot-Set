from typing import List, Dict, Any
from ..models import TopCandidate

# Hardcoded knowledge base (migrated from config)
PARROT_KB = {
    "蓝黄金刚鹦鹉": {
        "features": ["体羽蓝色+黄色胸腹", "粗壮黑喙，脸部白色裸区", "尾羽长，热带雨林分布"],
        "notes": "学名 Ara ararauna，常见于南美热带雨林。"
    },
    "玄凤鹦鹉": {
        "features": ["头顶竖冠羽", "脸颊橙色斑，体羽灰/黄色", "中小型体型，常见宠物鸟"],
        "notes": "学名 Nymphicus hollandicus，澳洲原生。"
    },
    "虎皮鹦鹉": {
        "features": ["体型小，额部平滑无冠羽", "颈部黑色斑点，翅膀有波浪纹", "常见绿色/蓝色羽色"],
        "notes": "学名 Melopsittacus undulatus，常见宠物鸟。"
    }
}

def search_knowledge(candidates: List[TopCandidate]) -> Dict[str, Any]:
    """
    Search for knowledge about candidate species.
    Currently uses hardcoded dictionary.
    """
    hits = {}
    for cand in candidates:
        entry = {}
        if cand.name in PARROT_KB:
            entry.update(PARROT_KB[cand.name])
        
        if entry:
            hits[cand.name] = entry
    return hits


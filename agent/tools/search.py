from typing import List, Dict, Any
import logging
from ..models import TopCandidate
from ..rag import get_rag_engine

logger = logging.getLogger(__name__)

def search_knowledge(candidates: List[TopCandidate]) -> Dict[str, Any]:
    """
    Search for knowledge about candidate species using RAG.

    注意：
    - 这是旧版 analyze 流水线使用的检索封装（agent/core.py::analyze 调用）。
    - 新版 /agent/message 的图片低置信度增强检索在 agent/router.py::run_analyze 中实现。
    - 这里的策略较简单：每个候选取 top_k=1，并做一个相似度阈值过滤（>0.4）。

    返回结构：
    {
      "物种名": { "source": "...", "content": "...", "score": 0.52 },
      ...
    }
    """
    rag = get_rag_engine()
    hits = {}
    
    for cand in candidates:
        logger.info(f"Searching knowledge for: {cand.name}")
        
        # 构造查询：直接用物种名，或者构造一个问题
        query = f"{cand.name}的特征和习性是什么？"
        
        results = rag.retrieve(query, top_k=1)
        
        if results:
            # 取最匹配的一条
            best_match = results[0]
            # 只有当相似度足够高时才采纳 (例如 > 0.4)
            if best_match['score'] > 0.4:
                source_text = best_match['source']
                # 如果有切片信息，添加到来源中
                if 'chunk_info' in best_match:
                    info = best_match['chunk_info']
                    source_text += f" (片段 {info['chunk_index'] + 1}/{info['total_chunks']})"
                
                hits[cand.name] = {
                    "source": source_text,
                    "content": best_match['content'],
                    "score": float(best_match['score']) # 转换为 float 以便 JSON 序列化
                }
            else:
                logger.info(f"Low confidence match for {cand.name}: {best_match['score']}")
    
    return hits



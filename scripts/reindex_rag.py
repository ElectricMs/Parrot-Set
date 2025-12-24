"""
手动触发 RAG 向量化/索引同步（本地脚本）。

默认行为：增量同步（只处理新增/变更/删除的文档）。
可选：--full 全量重建索引（更慢，但最干净）。

用法：
  python scripts/reindex_rag.py
  python scripts/reindex_rag.py --full
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.rag import get_rag_engine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="Force full rebuild instead of incremental sync")
    args = ap.parse_args()

    rag = get_rag_engine()
    result = rag.sync_index(force_full_rebuild=args.full)
    print("[reindex] mode =", result.get("mode"))
    print("[reindex] done")


if __name__ == "__main__":
    main()




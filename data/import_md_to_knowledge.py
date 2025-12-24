"""
将 data/ 下抓取/生成的 Markdown 文档批量导入到 knowledge/ 目录，供 RAG 索引。

为什么需要这个脚本？
- 当前 RAGEngine 默认只索引项目根目录的 knowledge/（见 agent/rag.py: knowledge_base_path="knowledge"）
- 你抓取的 huaNiao8 文档在 data/huaniao8_psittaciformes_md/，需要拷贝/同步到 knowledge/

用法示例：
  # 把 huaNiao8 的 5 个 part*.md 复制到 knowledge/（默认前缀 huaniao8_）
  python data/import_md_to_knowledge.py --src data/huaniao8_psittaciformes_md --dest knowledge

  # 覆盖同名文件（危险：确认你想覆盖）
  python data/import_md_to_knowledge.py --src data/huaniao8_psittaciformes_md --dest knowledge --overwrite
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Import markdown files into knowledge/ for RAG indexing.")
    ap.add_argument("--src", type=str, default="data/huaniao8_psittaciformes_md", help="Source directory containing .md files")
    ap.add_argument("--dest", type=str, default="knowledge", help="Destination knowledge directory")
    ap.add_argument("--prefix", type=str, default="huaniao8_", help="Prefix added to imported filenames to avoid collisions")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files in destination")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dest = Path(args.dest).resolve()

    if not src.exists() or not src.is_dir():
        raise SystemExit(f"Source directory not found: {src}")
    dest.mkdir(parents=True, exist_ok=True)

    md_files = sorted([p for p in src.glob("*.md") if p.is_file()])
    if not md_files:
        raise SystemExit(f"No .md files found in: {src}")

    copied = 0
    skipped = 0
    for p in md_files:
        out_name = f"{args.prefix}{p.name}" if args.prefix else p.name
        out_path = dest / out_name
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        shutil.copy2(p, out_path)
        copied += 1

    print(f"[import] src={src}")
    print(f"[import] dest={dest}")
    print(f"[import] copied={copied} skipped={skipped} overwrite={args.overwrite}")
    print("[import] done")


if __name__ == "__main__":
    main()

















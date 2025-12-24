"""
RAG 粗排+精排（reranker）效果对照测试脚本

对同一批 query 同时执行：
1) baseline：纯向量检索（等价于旧版 retrieve：n_results=top_k）
2) rerank：向量粗排 Top-10 + bge-reranker 精排后取 Top-K（当前 agent/rag.py::RAGEngine.retrieve）

输出：
- 统计信息（两种方式的耗时、命中启发式等）
- 每条 query 的 baseline/rerank Top-K 结果（包含 vector_score / rerank_score / score）

用法：
  python scripts/test_rag_rerank_effect.py --output rag_rerank_test_results.json
  python scripts/test_rag_rerank_effect.py --csv unitTest/parrots_complete.csv --num 100 --top-k 3
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.rag import get_rag_engine, get_detailed_instruct  # noqa: E402


def load_parrot_names(csv_path: Path) -> List[str]:
    names: List[str] = []
    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            if not fieldnames:
                print("Error: CSV file has no headers", file=sys.stderr)
                return []

            name_key = None
            possible_keys = ["名称 (Name)", "名称", "Name", fieldnames[0]]
            for key in possible_keys:
                if key in fieldnames:
                    name_key = key
                    break
            if not name_key:
                name_key = fieldnames[0]
                print(f"Warning: Using first column '{name_key}' as name field", file=sys.stderr)

            for row in reader:
                name_field = (row.get(name_key, "") or "").strip()
                if not name_field:
                    continue
                chinese_name = name_field.split(" / ")[0].strip()
                if chinese_name:
                    names.append(chinese_name)
    except Exception as e:
        print(f"Error reading CSV file: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
    return names


def _baseline_vector_retrieve(rag, query: str, top_k: int) -> List[Dict[str, Any]]:
    """
    纯向量检索 baseline（不走 rag.retrieve，以避免 rerank/粗排逻辑影响对照）。
    """
    if rag.collection.count() == 0:
        rag.build_index()
        if rag.collection.count() == 0:
            return []

    query_text = get_detailed_instruct(rag.task_description, query)
    results = rag.collection.query(query_texts=[query_text], n_results=int(top_k or 3))

    parsed: List[Dict[str, Any]] = []
    if results.get("ids") and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            similarity = 1 - distance
            metadata = results["metadatas"][0][i]
            content = results["documents"][0][i]
            parsed.append(
                {
                    "content": content,
                    "score": float(similarity),  # baseline 的 score 就是 vector_score
                    "vector_score": float(similarity),
                    "source": metadata.get("source", "unknown"),
                    "chunk_info": metadata,
                }
            )
    return parsed


def _format_hits(hits: List[Dict[str, Any]], preview_len: int = 200) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for h in hits:
        content = str(h.get("content") or "")
        out.append(
            {
                "score": round(float(h.get("score", 0.0) or 0.0), 6),
                "vector_score": round(float(h.get("vector_score", 0.0) or 0.0), 6)
                if h.get("vector_score") is not None
                else None,
                "rerank_score": round(float(h.get("rerank_score", 0.0) or 0.0), 6)
                if h.get("rerank_score") is not None
                else None,
                "source": h.get("source"),
                "content_preview": (content[:preview_len] + "...") if len(content) > preview_len else content,
                "chunk_info": h.get("chunk_info", {}) or {},
            }
        )
    return out


def _contains_query(hits: List[Dict[str, Any]], query: str) -> bool:
    q = (query or "").strip()
    if not q:
        return False
    for h in hits:
        if q in (h.get("content") or ""):
            return True
    return False


def run_test(
    rag,
    queries: List[str],
    top_k: int,
    warmup: bool = True,
) -> Tuple[List[Dict[str, Any]], List[float], List[float]]:
    """
    返回：(results, baseline_times_ms, rerank_times_ms)
    """
    if warmup and queries:
        print("Warming up (baseline vector query)...")
        try:
            _baseline_vector_retrieve(rag, "鹦鹉", top_k=top_k)
        except Exception as e:
            print(f"Warning: baseline warmup failed: {e}", file=sys.stderr)

        print("Warming up (rerank query)...")
        try:
            rag.retrieve("鹦鹉", top_k=top_k)
        except Exception as e:
            print(f"Warning: rerank warmup failed: {e}", file=sys.stderr)

    baseline_times: List[float] = []
    rerank_times: List[float] = []
    results: List[Dict[str, Any]] = []

    for idx, q in enumerate(queries, 1):
        print(f"[{idx}/{len(queries)}] Querying: {q}")

        # baseline
        t0 = time.perf_counter()
        baseline_hits = _baseline_vector_retrieve(rag, q, top_k=top_k)
        t1 = time.perf_counter()
        baseline_ms = (t1 - t0) * 1000.0
        baseline_times.append(baseline_ms)

        # rerank
        t2 = time.perf_counter()
        rerank_hits = rag.retrieve(q, top_k=top_k)
        t3 = time.perf_counter()
        rerank_ms = (t3 - t2) * 1000.0
        rerank_times.append(rerank_ms)

        results.append(
            {
                "query": q,
                "query_index": idx,
                "baseline": {
                    "query_time_ms": round(baseline_ms, 2),
                    "hit_count": len(baseline_hits),
                    "contains_query": _contains_query(baseline_hits, q),
                    "hits": _format_hits(baseline_hits),
                },
                "rerank": {
                    "query_time_ms": round(rerank_ms, 2),
                    "hit_count": len(rerank_hits),
                    "contains_query": _contains_query(rerank_hits, q),
                    "hits": _format_hits(rerank_hits),
                },
            }
        )

    return results, baseline_times, rerank_times


def _safe_avg(xs: List[float]) -> float:
    return round(sum(xs) / len(xs), 2) if xs else 0.0


def main():
    ap = argparse.ArgumentParser(description="Compare baseline vector retrieval vs rerank retrieval")
    ap.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "unitTest" / "parrots_complete.csv",
        help="Path to CSV file containing parrot names",
    )
    ap.add_argument("--num", type=int, default=100, help="Number of random samples to test (default: 100)")
    ap.add_argument("--top-k", type=int, default=3, help="Top-K results per query (default: 3)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    ap.add_argument("--output", type=Path, default=None, help="Output JSON file path")
    ap.add_argument("--summary-only", action="store_true", help="Only output statistics")
    ap.add_argument("--no-warmup", action="store_true", help="Disable warmup queries")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")

    print(f"Loading parrot names from: {args.csv}")
    parrot_names = load_parrot_names(args.csv)
    print(f"Loaded {len(parrot_names)} parrot names")

    if len(parrot_names) <= 0:
        print("No parrot names loaded.", file=sys.stderr)
        sys.exit(1)

    if len(parrot_names) < args.num:
        print(f"Warning: Only {len(parrot_names)} names available, using all of them.", file=sys.stderr)
        sampled = parrot_names
    else:
        sampled = random.sample(parrot_names, args.num)

    print("\nInitializing RAG engine...")
    init_start = time.perf_counter()
    rag = get_rag_engine()
    init_ms = (time.perf_counter() - init_start) * 1000.0
    print(f"RAG engine initialized in {init_ms:.2f}ms")

    results, baseline_times, rerank_times = run_test(
        rag,
        sampled,
        top_k=int(args.top_k),
        warmup=not args.no_warmup,
    )

    overheads = [max(0.0, r - b) for b, r in zip(baseline_times, rerank_times)]
    baseline_contains = sum(1 for r in results if r["baseline"]["contains_query"])
    rerank_contains = sum(1 for r in results if r["rerank"]["contains_query"])

    stats: Dict[str, Any] = {
        "total_queries": len(results),
        "top_k": int(args.top_k),
        "rag_init_time_ms": round(init_ms, 2),
        "baseline_avg_query_time_ms": _safe_avg(baseline_times),
        "rerank_avg_query_time_ms": _safe_avg(rerank_times),
        "rerank_avg_overhead_ms": _safe_avg(overheads),
        # 一个很粗的启发式：Top-K 的任一片段是否包含 query 字符串
        "baseline_contains_query_rate": round(baseline_contains / len(results) * 100, 2) if results else 0.0,
        "rerank_contains_query_rate": round(rerank_contains / len(results) * 100, 2) if results else 0.0,
        "delta_contains_query_rate_pp": round(
            (rerank_contains - baseline_contains) / len(results) * 100, 2
        )
        if results
        else 0.0,
    }

    out = {"statistics": stats, "results": [] if args.summary_only else results}

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {args.output}")
    else:
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        for k, v in stats.items():
            print(f"{k}: {v}")
        print("=" * 60)


if __name__ == "__main__":
    main()




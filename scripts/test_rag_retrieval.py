"""
RAG 检索效果测试脚本

从 parrots_complete.csv 中随机抽取100种鹦鹉名称进行查询，
返回每种鹦鹉的Top3检索结果和相似度分数。

用法：
  python scripts/test_rag_retrieval.py
  python scripts/test_rag_retrieval.py --csv unitTest/parrots_complete.csv --num 100
  python scripts/test_rag_retrieval.py --output results.json

    # 基本用法（随机抽取100种，显示统计信息）
    python scripts/test_rag_retrieval.py

    # 指定CSV文件和样本数量
    python scripts/test_rag_retrieval.py --csv unitTest/parrots_complete.csv --num 100

    # 保存完整结果到JSON文件
    python scripts/test_rag_retrieval.py --output rag_test_results.json

    # 只显示统计摘要，不显示详细结果
    python scripts/test_rag_retrieval.py --summary-only

    # 设置随机种子以便复现
    python scripts/test_rag_retrieval.py --seed 42

    # 自定义Top-K结果数
    python scripts/test_rag_retrieval.py --top-k 5
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.rag import get_rag_engine


def load_parrot_names(csv_path: Path) -> List[str]:
    """
    从CSV文件中加载鹦鹉名称列表。
    
    返回：鹦鹉名称列表（中文名称部分，去除英文和学名）
    """
    names = []
    try:
        # 使用 utf-8-sig 自动处理BOM标记
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            # 获取第一列的列名（处理可能的BOM问题）
            fieldnames = reader.fieldnames
            if not fieldnames:
                print("Error: CSV file has no headers", file=sys.stderr)
                return []
            
            # 第一列通常是名称列，尝试多种可能的列名
            name_key = None
            possible_keys = ['名称 (Name)', '名称', 'Name', fieldnames[0]]
            for key in possible_keys:
                if key in fieldnames:
                    name_key = key
                    break
            
            if not name_key:
                # 如果都找不到，使用第一列
                name_key = fieldnames[0]
                print(f"Warning: Using first column '{name_key}' as name field", file=sys.stderr)
            
            for row in reader:
                # 获取名称字段
                name_field = row.get(name_key, '').strip()
                if name_field:
                    # 提取中文名称（第一个斜杠之前的部分）
                    chinese_name = name_field.split(' / ')[0].strip()
                    if chinese_name:
                        names.append(chinese_name)
    except Exception as e:
        print(f"Error reading CSV file: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    return names


def test_retrieval(rag, parrot_names: List[str], num_samples: int = 100, top_k: int = 3) -> Tuple[List[Dict[str, Any]], List[float]]:
    """
    对随机抽取的鹦鹉名称进行RAG检索测试。
    
    参数：
    - rag: 已初始化的RAG引擎实例
    - parrot_names: 所有鹦鹉名称列表
    - num_samples: 随机抽取的数量
    - top_k: 每个查询返回的Top结果数
    
    返回：(测试结果列表, 查询时间列表)
    """
    # 随机抽取
    if len(parrot_names) < num_samples:
        print(f"Warning: Only {len(parrot_names)} names available, using all of them.", file=sys.stderr)
        sampled_names = parrot_names
    else:
        sampled_names = random.sample(parrot_names, num_samples)
    
    print(f"Testing {len(sampled_names)} parrot names...")
    
    # 预热：执行一次查询以确保模型完全加载
    print("Warming up RAG engine (first query may be slower)...")
    warmup_start = time.perf_counter()
    try:
        # 使用一个简单的固定查询词进行预热，避免影响后续测试样本
        rag.retrieve("鹦鹉", top_k=top_k)
        warmup_end = time.perf_counter()
        warmup_time_ms = (warmup_end - warmup_start) * 1000
        print(f"Warmup completed in {warmup_time_ms:.2f}ms")
    except Exception as e:
        print(f"Warning: Warmup query failed: {e}", file=sys.stderr)
    
    results = []
    query_times = []
    
    for idx, name in enumerate(sampled_names, 1):
        print(f"[{idx}/{len(sampled_names)}] Querying: {name}")
        
        try:
            # 记录查询开始时间
            start_time = time.perf_counter()
            
            # 执行检索
            hits = rag.retrieve(name, top_k=top_k)
            
            # 记录查询结束时间
            end_time = time.perf_counter()
            query_time_ms = (end_time - start_time) * 1000  # 转换为毫秒
            query_times.append(query_time_ms)
            
            # 格式化结果
            result = {
                "query": name,
                "query_index": idx,
                "query_time_ms": round(query_time_ms, 2),
                "hits": []
            }
            
            for hit in hits:
                result["hits"].append({
                    "score": round(hit["score"], 4),
                    "source": hit["source"],
                    "content_preview": hit["content"][:200] + "..." if len(hit["content"]) > 200 else hit["content"],
                    "chunk_info": hit.get("chunk_info", {})
                })
            
            print(f"  Query completed in {query_time_ms:.2f}ms")
            results.append(result)
            
        except Exception as e:
            print(f"  Error querying '{name}': {e}", file=sys.stderr)
            results.append({
                "query": name,
                "query_index": idx,
                "error": str(e),
                "query_time_ms": None,
                "hits": []
            })
    
    return results, query_times


def calculate_statistics(results: List[Dict[str, Any]], query_times: List[float], trim_outliers: int = 1) -> Dict[str, Any]:
    """
    计算检索效果统计信息。
    
    参数：
    - results: 查询结果列表
    - query_times: 每次查询的时间（毫秒）列表
    - trim_outliers: 去除最高和最低的样本数量（用于去除异常值）
    """
    total_queries = len(results)
    successful_queries = sum(1 for r in results if "error" not in r and len(r.get("hits", [])) > 0)
    failed_queries = total_queries - successful_queries
    
    scores = []
    for r in results:
        if "error" not in r:
            for hit in r.get("hits", []):
                scores.append(hit.get("score", 0))
    
    # 计算查询时间统计
    valid_times = [t for t in query_times if t is not None]
    
    # 原始统计（包含所有样本）
    avg_query_time = round(sum(valid_times) / len(valid_times), 2) if valid_times else 0
    max_query_time = round(max(valid_times), 2) if valid_times else 0
    min_query_time = round(min(valid_times), 2) if valid_times else 0
    total_query_time = round(sum(valid_times), 2) if valid_times else 0
    
    # 去除异常值后的统计（trimmed statistics）
    trimmed_times = valid_times.copy()
    if len(trimmed_times) > trim_outliers * 2:
        trimmed_times.sort()
        trimmed_times = trimmed_times[trim_outliers:-trim_outliers] if trim_outliers > 0 else trimmed_times
    
    avg_query_time_trimmed = round(sum(trimmed_times) / len(trimmed_times), 2) if trimmed_times else 0
    median_query_time = round(sorted(valid_times)[len(valid_times) // 2], 2) if valid_times else 0
    
    stats = {
        "total_queries": total_queries,
        "successful_queries": successful_queries,
        "failed_queries": failed_queries,
        "success_rate": round(successful_queries / total_queries * 100, 2) if total_queries > 0 else 0,
        "total_hits": len(scores),
        "avg_score": round(sum(scores) / len(scores), 4) if scores else 0,
        "max_score": round(max(scores), 4) if scores else 0,
        "min_score": round(min(scores), 4) if scores else 0,
        "avg_query_time_ms": avg_query_time,
        "avg_query_time_trimmed_ms": avg_query_time_trimmed,
        "median_query_time_ms": median_query_time,
        "max_query_time_ms": max_query_time,
        "min_query_time_ms": min_query_time,
        "total_query_time_ms": total_query_time,
        "trimmed_samples": trim_outliers,
    }
    
    return stats


def main():
    ap = argparse.ArgumentParser(description="Test RAG retrieval performance on parrot names")
    ap.add_argument("--csv", type=Path, default=ROOT / "unitTest" / "parrots_complete.csv",
                    help="Path to CSV file containing parrot names")
    ap.add_argument("--num", type=int, default=100,
                    help="Number of random samples to test (default: 100)")
    ap.add_argument("--top-k", type=int, default=3,
                    help="Number of top results to retrieve per query (default: 3)")
    ap.add_argument("--seed", type=int, default=None,
                    help="Random seed for reproducibility")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output JSON file path (default: print to stdout)")
    ap.add_argument("--summary-only", action="store_true",
                    help="Only print summary statistics, not detailed results")
    ap.add_argument("--trim-outliers", type=int, default=1,
                    help="Number of highest and lowest samples to trim when calculating average query time (default: 1)")
    
    args = ap.parse_args()
    
    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")
    
    # 加载鹦鹉名称
    print(f"Loading parrot names from: {args.csv}")
    parrot_names = load_parrot_names(args.csv)
    print(f"Loaded {len(parrot_names)} parrot names")
    
    # 初始化RAG引擎（在查询前初始化）
    print("\nInitializing RAG engine...")
    init_start = time.perf_counter()
    rag = get_rag_engine()
    init_end = time.perf_counter()
    init_time_ms = (init_end - init_start) * 1000
    print(f"RAG engine initialized in {init_time_ms:.2f}ms")
    
    # 执行测试
    results, query_times = test_retrieval(rag, parrot_names, num_samples=args.num, top_k=args.top_k)
    
    # 计算统计信息
    stats = calculate_statistics(results, query_times, trim_outliers=args.trim_outliers)
    stats["rag_init_time_ms"] = round(init_time_ms, 2)
    
    # 准备输出
    output_data = {
        "statistics": stats,
        "results": results if not args.summary_only else []
    }
    
    # 输出结果
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {args.output}")
    else:
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        print(f"Total queries: {stats['total_queries']}")
        print(f"Successful queries: {stats['successful_queries']}")
        print(f"Failed queries: {stats['failed_queries']}")
        print(f"Success rate: {stats['success_rate']}%")
        print(f"Total hits: {stats['total_hits']}")
        print(f"Average score: {stats['avg_score']}")
        print(f"Max score: {stats['max_score']}")
        print(f"Min score: {stats['min_score']}")
        print(f"\nRAG initialization time: {stats['rag_init_time_ms']}ms")
        print(f"Average query time (all samples): {stats['avg_query_time_ms']}ms")
        print(f"Average query time (trimmed {stats['trimmed_samples']} outliers): {stats['avg_query_time_trimmed_ms']}ms")
        print(f"Median query time: {stats['median_query_time_ms']}ms")
        print(f"Max query time: {stats['max_query_time_ms']}ms")
        print(f"Min query time: {stats['min_query_time_ms']}ms")
        print(f"Total query time: {stats['total_query_time_ms']}ms")
        print("="*60)
        
        if not args.summary_only:
            print("\nDETAILED RESULTS")
            print("="*60)
            for result in results[:10]:  # 只显示前10个结果作为示例
                print(f"\nQuery {result['query_index']}: {result['query']}")
                if "error" in result:
                    print(f"  Error: {result['error']}")
                else:
                    query_time = result.get('query_time_ms', 'N/A')
                    print(f"  Query time: {query_time}ms")
                    for i, hit in enumerate(result['hits'], 1):
                        print(f"  Top {i}: Score={hit['score']:.4f}, Source={hit['source']}")
                        print(f"    Preview: {hit['content_preview']}")
            if len(results) > 10:
                print(f"\n... (showing first 10 of {len(results)} results)")
                print("Use --output to save all results to a file")


if __name__ == "__main__":
    main()


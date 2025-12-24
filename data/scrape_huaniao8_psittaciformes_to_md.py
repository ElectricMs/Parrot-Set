"""
爬取 huaNiao8.com 的“鹦形目（Psittaciformes）”条目详情，并输出 Markdown。

目标页面（列表）：
- https://www.huaniao8.com/psittaciformes
分页形态（常见）：
- https://www.huaniao8.com/psittaciformes/page/2

详情页示例：
- https://www.huaniao8.com/niao/85099.html

输出：
- data/huaniao8_psittaciformes_md/huaniao8_psittaciformes_part01.md 等（按批次合并成“几个 md 文件”）

特点：
- 支持缓存 HTML 到 data/huaniao8_cache/，便于调试与减轻网站压力
- 支持断点续跑（state.json）
- 解析失败会保留 raw 片段，便于定位页面结构差异

免责声明：
- huaNiao8 页面内容可能受版权保护。本脚本默认只抽取结构化字段与少量简介片段，并附上来源链接。
  如需大规模转载请遵守目标站点版权声明与相关法律法规。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup


BASE = "https://www.huaniao8.com"
LIST_URL = f"{BASE}/psittaciformes"


def get_headers() -> Dict[str, str]:
    # 适度伪装 UA，避免被当成机器人；不要过度并发/频率过高
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.6",
    }


def clean_url(url: str) -> str:
    if not url:
        return ""
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if url.startswith("/"):
        return BASE + url
    return BASE + "/" + url


def sleep_polite(base: float = 0.35, jitter: float = 0.25):
    time.sleep(base + random.random() * jitter)


def fetch_text(url: str, timeout: int = 15) -> str:
    r = requests.get(url, headers=get_headers(), timeout=timeout)
    r.raise_for_status()
    # requests 会根据响应头猜编码；这里强制让它自己检测并保持 text
    return r.text


def list_parrots_from_site(max_pages: Optional[int] = None) -> List[Dict[str, str]]:
    """
    从鹦形目列表页抓取所有条目链接。
    返回：[{title, detail_url, image_url}]
    """
    results: List[Dict[str, str]] = []
    page = 1
    while True:
        if max_pages is not None and page > max_pages:
            break
        url = LIST_URL if page == 1 else f"{LIST_URL}/page/{page}"
        print(f"[list] page={page} url={url}")
        try:
            html = fetch_text(url)
        except requests.HTTPError as e:
            # 常见：404 表示没有更多页
            if getattr(e.response, "status_code", None) == 404:
                print("[list] reached end (404).")
                break
            raise
        soup = BeautifulSoup(html, "html.parser")
        articles = soup.find_all("article", class_="post")
        if not articles:
            print("[list] no articles found, stop.")
            break

        for article in articles:
            h2 = article.find("h2", class_="entry-title")
            if not h2:
                continue
            a = h2.find("a")
            if not a:
                continue
            title = a.get_text(strip=True)
            detail_url = clean_url(a.get("href") or "")

            img_url = ""
            img = article.find("img")
            if img:
                img_url = img.get("data-src") or img.get("data-original") or img.get("src") or ""
                img_url = clean_url(img_url) if img_url.startswith("/") else img_url

            results.append({"title": title, "detail_url": detail_url, "image_url": img_url})

        page += 1
        sleep_polite()

    # 去重（按 detail_url）
    dedup: Dict[str, Dict[str, str]] = {}
    for it in results:
        if it["detail_url"]:
            dedup[it["detail_url"]] = it
    out = list(dedup.values())
    print(f"[list] total={len(out)}")
    return out


def list_parrots_from_csv(csv_path: Path) -> List[Dict[str, str]]:
    """
    复用已有列表（减少对网站列表页压力）。
    读取 data/parrots_list.csv（字段：名称, 详情链接, 图片链接）。
    """
    items: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = (row.get("名称") or "").strip()
            detail = (row.get("详情链接") or "").strip()
            image = (row.get("图片链接") or "").strip()
            if detail:
                items.append({"title": title, "detail_url": detail, "image_url": image})
    # 去重
    dedup: Dict[str, Dict[str, str]] = {it["detail_url"]: it for it in items if it.get("detail_url")}
    out = list(dedup.values())
    print(f"[seed] loaded from csv: {csv_path} total={len(out)}")
    return out


@dataclass
class ParrotDetail:
    title: str
    detail_url: str
    image_url: str = ""
    chinese_name: str = ""
    english_name: str = ""
    scientific_name: str = ""
    taxonomy_order: str = ""
    taxonomy_family: str = ""
    distribution: str = ""
    summary: str = ""
    body_md: str = ""
    raw_taxonomy_text: str = ""


def _extract_label(text: str, label: str) -> str:
    # label like "英文名：" "学名：" "地理分布："
    m = re.search(re.escape(label) + r"\s*([^\n]+)", text)
    return m.group(1).strip() if m else ""


def parse_detail_page(html: str, url: str, fallback_title: str = "") -> ParrotDetail:
    soup = BeautifulSoup(html, "html.parser")

    # Title: h1 is usually "中文 / English / Scientific"
    h1 = soup.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else fallback_title

    # Get page text for regex parsing
    page_text = soup.get_text("\n", strip=True)

    english = _extract_label(page_text, "英文名：")
    scientific = _extract_label(page_text, "学名：")
    dist = _extract_label(page_text, "地理分布：")

    # Chinese name: often bold or first segment of title
    chinese = ""
    if title:
        # e.g. "琉璃金刚鹦鹉 / Blue-and-yellow Macaw / Ara ararauna"
        parts = [p.strip() for p in title.split("/") if p.strip()]
        if parts:
            chinese = parts[0]

    # Taxonomy: page often has a paragraph like:
    # "纲目科属：鹦形目 / Psittaciformes 鹦鹉科 / ... / Psittacidae"
    raw_tax = ""
    tax_order = ""
    tax_family = ""
    for p in soup.find_all("p"):
        t = p.get_text(" ", strip=True)
        if "纲目科属" in t or "Psittaciformes" in t or "鹦形目" in t:
            raw_tax = t.replace("\xa0", " ").strip()
            break
    if raw_tax:
        clean = raw_tax.replace("纲目科属：", "").strip()
        # try regex: order part ends with "目 / Latin"
        m = re.search(r"(.*?目\s*/\s*[A-Za-z]+)\s*(.*?科.*)", clean)
        if m:
            tax_order = m.group(1).strip()
            tax_family = m.group(2).strip()
        else:
            tax_order = clean

    def extract_body_md(max_chars: Optional[int] = None) -> str:
        """
        提取详情页“正文主内容”，并转成 Markdown。

        目标站点详情页通常在：
        - div.entry-content.u-text-format

        过滤规则（尽量保证“更详细”但去掉无关/声明/导航）：
        - 跳过包含“声明/版权声明”的 note 区域
        - 跳过仅包含“英文名/学名/纲目科属”的字段段落（这些已结构化提取）
        - 保留描述性段落与“地理分布”等内容
        """
        entry = soup.select_one("div.entry-content")
        if not entry:
            return ""

        md_lines: List[str] = []
        seen: set[str] = set()

        # 遍历 entry-content 内的段落/列表
        for node in entry.descendants:
            if getattr(node, "name", None) in ("script", "style"):
                continue
            # 停止/跳过声明区域
            if getattr(node, "name", None) == "div":
                cls = " ".join(node.get("class", []) or [])
                txt = node.get_text(" ", strip=True)
                if "声明" in txt or "版权声明" in txt:
                    # 直接 break：声明通常在正文末尾
                    break
                if "post-note" in cls:
                    # warning/info note，通常不算正文信息
                    continue

            if getattr(node, "name", None) == "p":
                # 图片段落
                img = node.find("img")
                if img and img.get("src"):
                    src = img.get("src")
                    alt = (img.get("alt") or "").strip()
                    md_lines.append(f"![{alt}]({src})" if alt else f"![]({src})")
                    md_lines.append("")
                    continue

                # 详情页的第一段通常是“加粗中文名 + 英文名/学名”字段块（已结构化提取），直接跳过
                if node.find("b") is not None:
                    continue

                t = node.get_text("\n", strip=True).replace("\xa0", " ").strip()
                if not t:
                    continue

                # 跳过字段型段落（已结构化提取）
                if t.startswith("纲目科属") or "纲目科属" in t:
                    continue

                # 去重（防止重复抓到同一段）
                key = t[:120]
                if key in seen:
                    continue
                seen.add(key)

                md_lines.append(t)
                md_lines.append("")

        body = "\n".join(md_lines).strip()
        if max_chars and len(body) > max_chars:
            body = body[:max_chars].rstrip() + "\n\n（正文过长，已截断）"
        return body

    body_md = extract_body_md()

    # Summary：默认取正文前 400 字作为摘要（若无正文则留空）
    summary = ""
    if body_md:
        summary = body_md.replace("\n", " ").strip()[:400]

    return ParrotDetail(
        title=title or fallback_title,
        detail_url=url,
        chinese_name=chinese,
        english_name=english,
        scientific_name=scientific,
        taxonomy_order=tax_order,
        taxonomy_family=tax_family,
        distribution=dist,
        summary=summary,
        body_md=body_md,
        raw_taxonomy_text=raw_tax,
    )


def load_cache(cache_dir: Path, key: str) -> Optional[str]:
    p = cache_dir / f"{key}.html"
    if p.exists():
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            return p.read_text(encoding="utf-8", errors="ignore")
    return None


def save_cache(cache_dir: Path, key: str, html: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"{key}.html"
    p.write_text(html, encoding="utf-8")


def url_to_key(url: str) -> str:
    # Use last path segment as key, e.g. /niao/85099.html -> niao_85099
    m = re.search(r"/([^/]+?)/(\d+)\.html", url)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return re.sub(r"[^a-zA-Z0-9]+", "_", url).strip("_")[:80]


def fetch_one_detail(item: Dict[str, str], cache_dir: Path, refresh: bool, polite_sleep: bool) -> ParrotDetail:
    url = item["detail_url"]
    key = url_to_key(url)
    html = None if refresh else load_cache(cache_dir, key)
    if html is None:
        html = fetch_text(url)
        save_cache(cache_dir, key, html)
        if polite_sleep:
            sleep_polite(0.25, 0.35)
    detail = parse_detail_page(html, url, fallback_title=item.get("title", ""))
    detail.image_url = item.get("image_url", "") or ""
    # ensure title exists
    if not detail.title:
        detail.title = item.get("title", "")
    return detail


def write_markdown_batches(details: List[ParrotDetail], out_dir: Path, per_file: int = 80) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    # 去重：以 detail_url 为 key（refresh/断点合并时避免重复）
    dedup: Dict[str, ParrotDetail] = {}
    for d in details:
        if d.detail_url:
            dedup[d.detail_url] = d
    details = list(dedup.values())

    total = len(details)
    if total == 0:
        return paths

    # deterministic order
    details_sorted = sorted(details, key=lambda d: (d.chinese_name or d.title or "", d.detail_url))

    parts = (total + per_file - 1) // per_file
    for idx in range(parts):
        chunk = details_sorted[idx * per_file : (idx + 1) * per_file]
        part_path = out_dir / f"huaniao8_psittaciformes_part{idx+1:02d}.md"
        paths.append(part_path)
        lines: List[str] = []
        lines.append("# huaNiao8 鹦形目（Psittaciformes）详情摘录")
        lines.append("")
        lines.append(f"- 来源：`{LIST_URL}`（详情页逐条抓取）")
        lines.append("- 说明：此文件为自动抓取后的结构化摘录，包含来源链接，仅用于项目内部测试/检索评估。")
        lines.append(f"- 条目范围：第 {idx+1} / {parts} 份（本文件 {len(chunk)} 条，总计 {total} 条）")
        lines.append("")
        lines.append("---")
        lines.append("")

        for d in chunk:
            title = d.title or d.chinese_name or d.detail_url
            lines.append(f"## {title}")
            lines.append("")
            lines.append(f"- **来源链接**：`{d.detail_url}`")
            if d.image_url:
                lines.append(f"- **图片**：`{d.image_url}`")
            if d.chinese_name:
                lines.append(f"- **中文名**：{d.chinese_name}")
            if d.english_name:
                lines.append(f"- **英文名**：{d.english_name}")
            if d.scientific_name:
                lines.append(f"- **学名**：{d.scientific_name}")
            if d.taxonomy_order:
                lines.append(f"- **目**：{d.taxonomy_order}")
            if d.taxonomy_family:
                lines.append(f"- **科**：{d.taxonomy_family}")
            if d.distribution:
                lines.append(f"- **地理分布**：{d.distribution}")
            lines.append("")

            if d.summary:
                lines.append("### 简介（摘录）")
                lines.append("")
                lines.append(d.summary)
                lines.append("")

            if d.body_md:
                lines.append("### 正文")
                lines.append("")
                lines.append(d.body_md.strip())
                lines.append("")

            lines.append("---")
            lines.append("")

        part_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    return paths


def load_state(state_path: Path) -> Dict[str, Dict]:
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(state_path: Path, state: Dict[str, Dict]) -> None:
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Scrape huaNiao8 psittaciformes details to Markdown batches.")
    ap.add_argument("--use-csv-seed", action="store_true", help="Use data/parrots_list.csv as seed list (recommended).")
    ap.add_argument("--max-pages", type=int, default=None, help="Max list pages to crawl (only when not using csv seed).")
    ap.add_argument("--workers", type=int, default=6, help="Thread workers for detail pages.")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of details to fetch (for quick test).")
    ap.add_argument("--refresh", action="store_true", help="Ignore cache and refetch detail pages.")
    ap.add_argument("--no-politeness-sleep", action="store_true", help="Disable random sleep between requests (not recommended).")
    ap.add_argument("--per-file", type=int, default=80, help="How many birds per output md file.")
    args = ap.parse_args()

    data_dir = Path(__file__).resolve().parent
    cache_dir = data_dir / "huaniao8_cache"
    out_dir = data_dir / "huaniao8_psittaciformes_md"
    state_path = data_dir / "huaniao8_state.json"

    # 1) Seed list
    if args.use_csv_seed:
        seed_csv = data_dir / "parrots_list.csv"
        if not seed_csv.exists():
            raise FileNotFoundError(f"CSV seed not found: {seed_csv}")
        items = list_parrots_from_csv(seed_csv)
    else:
        items = list_parrots_from_site(max_pages=args.max_pages)

    if args.limit is not None:
        items = items[: args.limit]

    # 2) Resume state
    state = load_state(state_path)
    done_urls = set(state.keys())

    # 如果 --refresh：即使已做过也重新抓取并覆盖 state（用于升级解析规则/补充字段）
    if args.refresh:
        todo = [it for it in items if it.get("detail_url")]
    else:
        todo = [it for it in items if it.get("detail_url") and it["detail_url"] not in done_urls]
    print(f"[detail] total={len(items)} already_done={len(done_urls)} todo={len(todo)}")

    details: List[ParrotDetail] = []
    # 非 refresh 模式下，先加载已完成的详情（避免重复抓取，输出也包含已完成的）
    if not args.refresh:
        for url, saved in state.items():
            try:
                d = ParrotDetail(
                    title=saved.get("title", ""),
                    detail_url=url,
                    image_url=saved.get("image_url", ""),
                    chinese_name=saved.get("chinese_name", ""),
                    english_name=saved.get("english_name", ""),
                    scientific_name=saved.get("scientific_name", ""),
                    taxonomy_order=saved.get("taxonomy_order", ""),
                    taxonomy_family=saved.get("taxonomy_family", ""),
                    distribution=saved.get("distribution", ""),
                    summary=saved.get("summary", ""),
                    body_md=saved.get("body_md", ""),
                    raw_taxonomy_text=saved.get("raw_taxonomy_text", ""),
                )
                details.append(d)
            except Exception:
                continue

    polite = not args.no_politeness_sleep

    # 3) Fetch details concurrently (polite: workers not too high)
    if todo:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futures = {
                ex.submit(fetch_one_detail, it, cache_dir, args.refresh, polite): it for it in todo
            }
            ok = 0
            fail = 0
            for fut in as_completed(futures):
                it = futures[fut]
                url = it.get("detail_url", "")
                try:
                    d = fut.result()
                    details.append(d)
                    state[url] = {
                        "title": d.title,
                        "detail_url": d.detail_url,
                        "image_url": d.image_url,
                        "chinese_name": d.chinese_name,
                        "english_name": d.english_name,
                        "scientific_name": d.scientific_name,
                        "taxonomy_order": d.taxonomy_order,
                        "taxonomy_family": d.taxonomy_family,
                        "distribution": d.distribution,
                        "summary": d.summary,
                        "body_md": d.body_md,
                        "raw_taxonomy_text": d.raw_taxonomy_text,
                    }
                    ok += 1
                    if ok % 20 == 0:
                        print(f"[detail] progress ok={ok} fail={fail} (state saved)")
                        save_state(state_path, state)
                except Exception as e:
                    fail += 1
                    print(f"[detail] FAIL url={url} err={e}")
                    # keep a minimal record to avoid infinite retries
                    if url:
                        state[url] = {"detail_url": url, "error": str(e), "title": it.get("title", "")}
                # incremental save
                if (ok + fail) % 50 == 0:
                    save_state(state_path, state)
        save_state(state_path, state)

    # 4) Write markdown batches
    paths = write_markdown_batches(details, out_dir=out_dir, per_file=args.per_file)
    print("[write] files:")
    for p in paths:
        print(" -", p)
    print("[done]")


if __name__ == "__main__":
    main()



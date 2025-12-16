import requests
from bs4 import BeautifulSoup
import csv
import time
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_headers():
    return {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

def clean_url(url, base_domain="https://www.huaniao8.com"):
    if not url:
        return ""
    if url.startswith("http"):
        return url
    if url.startswith("/"):
        return base_domain + url
    return base_domain + "/" + url

def fetch_taxonomy(detail_url):
    """
    Fetches the detail page and extracts Order and Family information.
    Returns (Order, Family) tuple.
    """
    if not detail_url:
        return ("未知", "未知")
        
    try:
        # Retry logic could be added here
        response = requests.get(detail_url, headers=get_headers(), timeout=10)
        if response.status_code != 200:
            return ("请求失败", "请求失败")
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Strategy 1: Find text containing 'Psittaciformes' or '鹦形目'
        # The structure observed: "纲目科属：鹦形目 / Psittaciformes凤头鹦鹉科 / Cockatoos / Cacatuidae"
        # Using a broad search to find the container
        target_text = ""
        
        # Try finding the specific paragraph often found in these pages
        p_tags = soup.find_all('p')
        for p in p_tags:
            text = p.get_text(strip=True)
            if '鹦形目' in text or 'Psittaciformes' in text:
                target_text = text
                break
        
        if not target_text:
            return ("未找到分类信息", "未找到分类信息")
            
        # Clean the text: remove "纲目科属：" prefix if present
        clean_text = target_text.replace("纲目科属：", "").strip()
        
        # Use regex to split Order and Family
        # Pattern: [Order Part] followed by [Family Part]
        # Order part usually ends with Latin name (English chars)
        # Family part starts with Chinese
        
        # Regex: (Anything ending in "目" + / + Latin)(Anything ending in "科" + ...)
        # Note: The text might be concatenated like "Psittaciformes凤头鹦鹉科"
        
        match = re.search(r'(.*?目\s*/\s*[a-zA-Z]+)\s*(.*?科.*)', clean_text)
        if match:
            order_info = match.group(1).strip()
            family_info = match.group(2).strip()
            return (order_info, family_info)
            
        # Fallback: Just return the whole text if regex fails but we found the paragraph
        return (clean_text, "")
        
    except Exception as e:
        return (f"Error: {str(e)}", "")

def process_parrot(parrot_basic_info):
    """
    Worker function to process a single parrot entry.
    parrot_basic_info: [name, detail_url, img_url]
    Returns: [name, detail_url, img_url, order, family]
    """
    name, detail_url, img_url = parrot_basic_info
    order, family = fetch_taxonomy(detail_url)
    return [name, detail_url, img_url, order, family]

def scrape_parrots():
    base_url = "https://www.huaniao8.com/psittaciformes"
    page = 1
    basic_parrots_list = []
    
    headers = get_headers()
    
    print("第一阶段：爬取所有鹦鹉的基本列表...")
    
    while True:
        url = base_url if page == 1 else f"{base_url}/page/{page}"
        print(f"正在扫描第 {page} 页: {url}")
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 404:
                print("已扫描完所有列表页面。")
                break
                
            if response.status_code != 200:
                print(f"列表页请求失败，状态码: {response.status_code}")
                break
            
            soup = BeautifulSoup(response.text, 'html.parser')
            articles = soup.find_all('article', class_='post')
            
            if not articles:
                print("当前页面无数据，结束扫描。")
                break
                
            for article in articles:
                try:
                    title_elem = article.find('h2', class_='entry-title')
                    if not title_elem: continue
                        
                    link_elem = title_elem.find('a')
                    name = link_elem.get_text(strip=True)
                    raw_link = link_elem.get('href')
                    detail_url = clean_url(raw_link) # Ensure absolute URL
                    
                    img_elem = article.find('img')
                    img_url = "无图片"
                    if img_elem:
                        img_url = img_elem.get('data-src') or img_elem.get('data-original') or img_elem.get('src')
                    
                    basic_parrots_list.append([name, detail_url, img_url])
                    
                except Exception as e:
                    print(f"解析列表项出错: {e}")
            
            time.sleep(0.5)
            page += 1
            
        except Exception as e:
            print(f"网络错误: {e}")
            break
            
    print(f"\n共找到 {len(basic_parrots_list)} 种鹦鹉。")
    print("第二阶段：并发获取每种鹦鹉的分类信息（可能需要几分钟）...")
    
    final_results = []
    # Use ThreadPoolExecutor for concurrent requests
    # Adjust max_workers based on network/server tolerance. 10 is usually safe for light scraping.
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks
        future_to_parrot = {executor.submit(process_parrot, p): p for p in basic_parrots_list}
        
        completed_count = 0
        total_count = len(basic_parrots_list)
        
        for future in as_completed(future_to_parrot):
            try:
                result = future.result()
                final_results.append(result)
                completed_count += 1
                if completed_count % 10 == 0:
                    print(f"进度: {completed_count}/{total_count}")
            except Exception as e:
                print(f"处理任务出错: {e}")
    
    # Sort by name or original order (preserving original order is harder with as_completed unless we map back)
    # Let's just sort by name for now, or we can use map to preserve order but that blocks.
    # To preserve order, we can re-sort based on the index in basic_parrots_list, but that's minor.
    
    print("\n保存结果...")
    output_file = 'parrots_list_detailed.csv'
    try:
        with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['名称', '详情链接', '图片链接', '目 (Order)', '科 (Family)'])
            writer.writerows(final_results)
        print(f"完成！数据已保存至: {os.path.abspath(output_file)}")
        
    except Exception as e:
        print(f"保存文件失败: {e}")

if __name__ == "__main__":
    scrape_parrots()


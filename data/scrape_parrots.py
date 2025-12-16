import requests
from bs4 import BeautifulSoup
import csv
import time
import os

def scrape_parrots():
    # 基础URL
    base_url = "https://www.huaniao8.com/psittaciformes"
    page = 1
    all_parrots = []
    
    # 伪装User-Agent
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    print("开始爬取鹦鹉种类信息...")
    
    while True:
        # 构造每一页的URL
        url = base_url if page == 1 else f"{base_url}/page/{page}"
        print(f"正在处理第 {page} 页: {url}")
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            
            # 如果是404，说明页码超出了，结束循环
            if response.status_code == 404:
                print("已到达最后一页 (404)。")
                break
                
            if response.status_code != 200:
                print(f"请求失败，状态码: {response.status_code}")
                break
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 查找所有的文章容器
            # 根据之前的分析，每个条目是一个 article 标签，类名为 post
            articles = soup.find_all('article', class_='post')
            
            if not articles:
                print("当前页面未找到任何数据，停止爬取。")
                break
                
            for article in articles:
                try:
                    # 提取标题和链接
                    title_elem = article.find('h2', class_='entry-title')
                    if not title_elem:
                        continue
                        
                    link_elem = title_elem.find('a')
                    name = link_elem.get_text(strip=True)
                    detail_url = link_elem.get('href')
                    
                    # 提取图片链接
                    # 图片通常有懒加载，优先获取 data-src
                    img_elem = article.find('img')
                    img_url = "无图片"
                    if img_elem:
                        # 尝试获取真实的图片地址，防止获取到base64占位符
                        img_url = img_elem.get('data-src') or img_elem.get('data-original') or img_elem.get('src')
                    
                    all_parrots.append([name, detail_url, img_url])
                    
                except Exception as e:
                    print(f"解析条目时出错: {e}")
                    continue
            
            # 简单的防爬限制规避
            time.sleep(1)
            page += 1
            
        except Exception as e:
            print(f"发生网络错误: {e}")
            break
            
    # 保存结果到CSV文件
    output_file = 'parrots_list.csv'
    try:
        # 使用 utf-8-sig 以便 Excel 正确打开中文
        with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['名称', '详情链接', '图片链接'])
            writer.writerows(all_parrots)
        print(f"\n爬取结束！共获取 {len(all_parrots)} 种鹦鹉信息。")
        print(f"数据已保存至: {os.path.abspath(output_file)}")
        
    except Exception as e:
        print(f"保存文件时出错: {e}")

if __name__ == "__main__":
    scrape_parrots()


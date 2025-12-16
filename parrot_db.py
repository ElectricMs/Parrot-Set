"""
鹦鹉种类数据库查询模块

基于 parrots_list_detailed.csv 文件提供种类查询和验证功能。
"""

import csv
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re

logger = logging.getLogger(__name__)

# CSV文件路径
CSV_FILE_PATH = Path("data/parrots_list_detailed.csv")


class ParrotDatabase:
    """
    鹦鹉种类数据库
    
    从CSV文件加载数据，提供快速查询和验证功能。
    """
    
    def __init__(self, csv_path: Path = CSV_FILE_PATH):
        """
        初始化数据库
        
        Args:
            csv_path: CSV文件路径
        """
        self.csv_path = csv_path
        self.parrots: List[Dict[str, str]] = []
        self.name_index: Dict[str, Dict[str, str]] = {}  # 中文名 -> 完整记录
        self.english_index: Dict[str, Dict[str, str]] = {}  # 英文名 -> 完整记录
        self.scientific_index: Dict[str, Dict[str, str]] = {}  # 学名 -> 完整记录
        self._load_data()
    
    def _load_data(self):
        """加载CSV数据并建立索引"""
        if not self.csv_path.exists():
            logger.warning(f"CSV文件不存在: {self.csv_path}")
            return
        
        try:
            with open(self.csv_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # 清理数据
                    name_field = row.get('名称', '').strip()
                    if not name_field:
                        continue
                    
                    # 解析名称字段：格式为 "中文名 / English Name / Scientific Name"
                    parts = [p.strip() for p in name_field.split('/')]
                    chinese_name = parts[0] if len(parts) > 0 else ""
                    english_name = parts[1] if len(parts) > 1 else ""
                    scientific_name = parts[2] if len(parts) > 2 else ""
                    
                    # 构建记录
                    record = {
                        'chinese_name': chinese_name,
                        'english_name': english_name,
                        'scientific_name': scientific_name,
                        'full_name': name_field,
                        'link': row.get('详情链接', '').strip(),
                        'image': row.get('图片链接', '').strip(),
                        'order': row.get('目 (Order)', '').strip(),
                        'family': row.get('科 (Family)', '').strip(),
                    }
                    
                    self.parrots.append(record)
                    
                    # 建立索引
                    if chinese_name:
                        self.name_index[chinese_name] = record
                    if english_name:
                        self.english_index[english_name.lower()] = record
                    if scientific_name:
                        self.scientific_index[scientific_name.lower()] = record
            
            logger.info(f"成功加载 {len(self.parrots)} 条鹦鹉记录")
        except Exception as e:
            logger.error(f"加载CSV文件失败: {e}", exc_info=True)
    
    def find_by_name(self, name: str) -> Optional[Dict[str, str]]:
        """
        根据名称查找鹦鹉记录
        
        支持中文名、英文名、学名的精确匹配。
        
        Args:
            name: 鹦鹉名称（中文、英文或学名）
        
        Returns:
            找到的记录字典，如果未找到返回None
        """
        name = name.strip()
        if not name:
            return None
        
        # 尝试中文名匹配
        if name in self.name_index:
            return self.name_index[name]
        
        # 尝试英文名匹配（不区分大小写）
        name_lower = name.lower()
        if name_lower in self.english_index:
            return self.english_index[name_lower]
        
        # 尝试学名匹配（不区分大小写）
        if name_lower in self.scientific_index:
            return self.scientific_index[name_lower]
        
        return None
    
    def fuzzy_search(self, query: str, limit: int = 10) -> List[Dict[str, str]]:
        """
        模糊搜索鹦鹉名称
        
        支持部分匹配中文名、英文名、学名。
        
        Args:
            query: 搜索关键词
            limit: 返回结果数量限制
        
        Returns:
            匹配的记录列表
        """
        query = query.strip().lower()
        if not query:
            return []
        
        results = []
        seen = set()  # 避免重复
        
        # 搜索中文名
        for chinese_name, record in self.name_index.items():
            if query in chinese_name.lower():
                if chinese_name not in seen:
                    results.append(record)
                    seen.add(chinese_name)
        
        # 搜索英文名
        for english_name, record in self.english_index.items():
            if query in english_name:
                chinese = record['chinese_name']
                if chinese not in seen:
                    results.append(record)
                    seen.add(chinese)
        
        # 搜索学名
        for scientific_name, record in self.scientific_index.items():
            if query in scientific_name:
                chinese = record['chinese_name']
                if chinese not in seen:
                    results.append(record)
                    seen.add(chinese)
        
        return results[:limit]
    
    def exists(self, name: str) -> bool:
        """
        检查某个鹦鹉种类是否存在
        
        Args:
            name: 鹦鹉名称
        
        Returns:
            如果存在返回True，否则返回False
        """
        return self.find_by_name(name) is not None
    
    def validate_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """
        验证候选种类列表，标记哪些存在于数据库中
        
        Args:
            candidates: 候选列表，每个元素包含 'name' 字段
        
        Returns:
            验证后的列表，每个元素添加 'exists_in_db' 和 'db_info' 字段
        """
        validated = []
        for cand in candidates:
            name = cand.get('name', '')
            record = self.find_by_name(name)
            
            validated_cand = cand.copy()
            validated_cand['exists_in_db'] = record is not None
            
            if record:
                validated_cand['db_info'] = {
                    'chinese_name': record['chinese_name'],
                    'english_name': record['english_name'],
                    'scientific_name': record['scientific_name'],
                    'order': record['order'],
                    'family': record['family'],
                    'link': record['link'],
                }
            else:
                validated_cand['db_info'] = None
            
            validated.append(validated_cand)
        
        return validated
    
    def get_all_species(self) -> List[str]:
        """
        获取所有鹦鹉的中文名称列表
        
        Returns:
            中文名称列表
        """
        return list(self.name_index.keys())


# 全局数据库实例
_db_instance: Optional[ParrotDatabase] = None


def get_database() -> ParrotDatabase:
    """
    获取全局数据库实例（单例模式）
    
    Returns:
        ParrotDatabase实例
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = ParrotDatabase()
    return _db_instance


def check_species_exists(name: str) -> Tuple[bool, Optional[Dict[str, str]]]:
    """
    检查鹦鹉种类是否存在（便捷函数）
    
    Args:
        name: 鹦鹉名称
    
    Returns:
        (是否存在, 记录信息) 元组
    """
    db = get_database()
    record = db.find_by_name(name)
    exists = record is not None
    return exists, record


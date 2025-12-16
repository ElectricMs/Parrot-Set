# 鹦鹉种类数据库查询功能说明

## 功能概述

已为 Parrot Set 项目添加了基于 `data/parrots_list_detailed.csv` 的鹦鹉种类查询和验证功能。

## 主要功能

### 1. 自动验证分类结果

在 `/classify` 和 `/analyze` 接口返回的分类结果中，每个候选种类都会自动验证是否存在于数据库中：

- `exists_in_db`: 布尔值，表示该种类是否存在于数据库
- `db_info`: 如果存在，包含详细信息（中文名、英文名、学名、目、科、链接等）

### 2. 新增API端点

#### `/check_species` - 检查种类是否存在

**请求方式**: GET

**参数**:
- `name` (必需): 鹦鹉名称（支持中文名、英文名、学名）

**返回示例**:
```json
{
  "exists": true,
  "info": {
    "chinese_name": "琉璃金刚鹦鹉",
    "english_name": "Blue-and-yellow Macaw",
    "scientific_name": "Ara ararauna",
    "order": "鹦形目 / Psittaciformes",
    "family": "鹦鹉科 / African and New World Parrots / Psittacidae",
    "link": "https://www.huaniao8.com/niao/85099.html",
    "image": "https://..."
  },
  "suggestions": []
}
```

**使用示例**:
```bash
# 检查中文名
curl "http://localhost:8000/check_species?name=琉璃金刚鹦鹉"

# 检查英文名
curl "http://localhost:8000/check_species?name=Blue-and-yellow Macaw"

# 检查学名
curl "http://localhost:8000/check_species?name=Ara ararauna"
```

#### `/search_species` - 模糊搜索种类

**请求方式**: GET

**参数**:
- `query` (必需): 搜索关键词
- `limit` (可选): 返回结果数量限制，默认10

**返回示例**:
```json
{
  "count": 5,
  "results": [
    {
      "chinese_name": "大绿金刚鹦鹉",
      "english_name": "Great Green Macaw",
      "scientific_name": "Ara ambiguus",
      "order": "鹦形目 / Psittaciformes",
      "family": "鹦鹉科 / African and New World Parrots / Psittacidae",
      "link": "https://..."
    },
    ...
  ]
}
```

**使用示例**:
```bash
# 搜索包含"金刚"的种类
curl "http://localhost:8000/search_species?query=金刚&limit=5"

# 搜索英文名
curl "http://localhost:8000/search_species?query=Macaw"
```

## 代码结构

### 新增文件

- `parrot_db.py`: 数据库查询模块
  - `ParrotDatabase` 类：管理CSV数据加载和查询
  - `get_database()`: 获取全局数据库实例（单例模式）
  - `check_species_exists()`: 便捷查询函数

### 修改文件

- `app.py`: 
  - 更新 `TopCandidate` 模型，添加 `exists_in_db` 和 `db_info` 字段
  - 在 `llm_classify_image()` 函数中集成数据库验证
  - 新增 `/check_species` 和 `/search_species` API端点

## 使用场景

1. **分类结果验证**: 当模型识别出鹦鹉种类后，自动检查该种类是否在已知数据库中
2. **知识库查询**: 通过API查询特定鹦鹉的详细信息（目、科、学名等）
3. **模糊搜索**: 用户输入部分名称时，提供匹配建议
4. **数据质量检查**: 验证模型输出的种类名称是否有效

## 数据库信息

- **数据源**: `data/parrots_list_detailed.csv`
- **记录数**: 398条鹦鹉种类
- **字段**: 名称（中/英/学名）、详情链接、图片链接、目、科

## 注意事项

1. CSV文件使用 UTF-8-BOM 编码，确保中文正确显示
2. 名称匹配支持：
   - 中文名精确匹配
   - 英文名不区分大小写匹配
   - 学名不区分大小写匹配
3. 模糊搜索支持部分匹配，返回最相关的结果


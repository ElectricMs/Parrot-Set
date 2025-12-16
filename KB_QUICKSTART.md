# 知识库系统快速开始

## 快速测试

### 1. 启动服务

```bash
python app.py
```

服务将在 `http://localhost:8000` 启动。

### 2. 上传测试文档

使用项目根目录下的示例文件：

```bash
# 上传单个文档
curl -X POST "http://localhost:8000/kb/upload" \
  -F "file=@sample_parrot_info.txt"

# 或使用Python
import requests

with open("sample_parrot_info.txt", "rb") as f:
    response = requests.post(
        "http://localhost:8000/kb/upload",
        files={"file": f}
    )
    print(response.json())
```

### 3. 搜索测试

```bash
# 搜索知识库
curl "http://localhost:8000/kb/search?query=蓝黄金刚鹦鹉&k=3"

# 或使用Python
import requests

response = requests.get(
    "http://localhost:8000/kb/search",
    params={"query": "蓝黄金刚鹦鹉", "k": 3}
)
print(response.json())
```

### 4. 查看统计

```bash
curl "http://localhost:8000/kb/stats"
```

## 集成到分类流程

知识库已自动集成到 `/analyze` 接口：

1. 上传图片进行分类
2. 系统会自动从向量数据库检索相关信息
3. 检索结果会合并到解释生成中

**示例**:

```python
import requests

# 1. 先上传一些知识文档
with open("parrot_knowledge.pdf", "rb") as f:
    requests.post("http://localhost:8000/kb/upload", files={"file": f})

# 2. 上传图片进行分析
with open("parrot.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/analyze",
        files={"image": f}
    )
    result = response.json()
    
    # 结果中会包含向量数据库检索的信息
    print(result["explanation"])
```

## 常用操作

### 批量上传文档

```python
import requests
from pathlib import Path

files = []
for file_path in Path("./documents").glob("*.pdf"):
    files.append(("files", open(file_path, "rb")))

response = requests.post(
    "http://localhost:8000/kb/upload_batch",
    files=files
)
print(response.json())
```

### 处理整个目录

```bash
curl -X POST "http://localhost:8000/kb/add_directory?directory_path=./documents&recursive=true"
```

### 查看所有文档

```bash
curl "http://localhost:8000/kb/documents"
```

## 文件格式要求

### PDF
- 必须包含可提取的文本（非纯图片扫描）
- 建议使用文本型PDF

### TXT
- 必须使用UTF-8编码
- 支持中文、英文等多语言

### Markdown
- 标准Markdown格式
- 支持代码块、表格等

### Word (.docx)
- Microsoft Word 2007+格式
- 不支持旧版.doc格式

### Excel
- 支持.xlsx和.xls格式
- 每个工作表会单独处理

### CSV
- 使用UTF-8编码
- 第一行为列名

## 性能优化建议

1. **文档大小**: 单个文档建议<10MB
2. **批量处理**: 使用批量上传而非逐个上传
3. **定期清理**: 删除不需要的文档释放空间
4. **索引优化**: 文档结构清晰有助于检索效果

## 下一步

- 查看完整文档: `data/KB_COMPLETE_GUIDE.md`
- 查看API文档: 访问 `http://localhost:8000/docs`
- 查看使用示例: `data/KB_USAGE.md`

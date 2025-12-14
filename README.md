# 🦜 Parrot Set 鹦鹉集

一个基于 AI 的鹦鹉识别与知识查询系统，使用 Qwen3-VL 多模态模型进行图像识别，结合 RAG 技术提供详细的物种特征解释。

## ✨ 功能特性

- **🖼️ 图像识别分类**：上传鹦鹉照片，自动识别品种并给出 top3 候选结果
- **📊 概率分析**：提供每个候选品种的置信度分数和概率百分比
- **🔍 视觉特征提取**：自动提取照片中的颜色、冠羽、喙、斑纹等关键特征
- **📚 知识库检索**：基于 RAG 技术检索相关鹦鹉品种的详细特征信息
- **💡 智能解释**：结合视觉特征和知识库，生成详细的分类依据和解释

## 🛠️ 技术栈

- **后端框架**：FastAPI
- **AI 模型**：Qwen3-VL-2B-Thinking（通过 Ollama 部署）
- **多模态处理**：LangChain + Ollama API
- **知识检索**：RAG（当前为本地字典，可扩展为向量数据库）
- **Python 版本**：3.8+

## 📦 安装步骤

### 1. 克隆项目

```bash
git clone <repository-url>
cd "Parrot Set"
```

### 2. 创建虚拟环境

```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

如果没有 `requirements.txt`，手动安装：

```bash
pip install fastapi uvicorn python-multipart requests langchain langchain-core
```

### 4. 安装并配置 Ollama

#### 下载 Ollama

- **Windows**: 从 [Ollama 官网](https://ollama.com/download) 下载安装包
- **Linux/Mac**: 
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

#### 下载模型

```bash
# 拉取 Qwen3-VL-2B-Thinking 模型
ollama pull qwen3-vl:2b-thinking-q4_K_M
```

或者使用本地 GGUF 文件（参考 `Qwen3-VL-2B-Thinking-GGUF/` 目录）：

```bash
cd Qwen3-VL-2B-Thinking-GGUF
ollama create qwen3-vl-2b-local -f Modelfile
```

### 5. 验证安装

```bash
# 检查 Ollama 服务
ollama list

# 测试模型
ollama run qwen3-vl:2b-thinking-q4_K_M "你好"
```

## 🚀 使用方法

### 启动后端服务

```bash
# 方式1：使用 uvicorn 命令
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# 方式2：直接运行
python app.py
```

服务启动后，访问：
- **API 文档**：http://localhost:8000/docs
- **健康检查**：http://localhost:8000/health

### 使用前端界面（批量上传和自动分类）

#### 方式 1：直接打开（推荐）

1. 确保后端服务已启动
2. 用浏览器打开 `frontend/index.html` 文件
3. 上传图片，点击"开始识别"
4. 系统会自动识别并保存到指定文件夹

#### 方式 2：使用本地服务器

```bash
cd frontend
python -m http.server 8080
```

然后访问：http://localhost:8080

**前端功能**：
- ✅ 批量上传图片（支持拖拽）
- ✅ 实时显示识别进度
- ✅ 展示识别结果（top3 候选 + 概率 + 视觉特征）
- ✅ 自动保存分类结果到文件夹
- ✅ 统计信息（总数、成功、失败、已保存）

### API 接口

#### 1. 健康检查

```http
GET /health
```

**响应示例**：
```json
{
  "status": "ok",
  "ollama_available": true,
  "model_loaded": true,
  "model": "qwen3-vl:2b-thinking-q4_K_M",
  "message": "服务正常"
}
```

#### 2. 图像分类

```http
POST /classify
Content-Type: multipart/form-data
```

**请求**：
- `image`: 图片文件（支持 JPG、PNG 等格式）

**响应示例**：
```json
{
  "top_candidates": [
    {
      "name": "蓝黄金刚鹦鹉",
      "score": 0.87,
      "probability": 87.0
    },
    {
      "name": "玄凤鹦鹉",
      "score": 0.1,
      "probability": 10.0
    },
    {
      "name": "虎皮鹦鹉",
      "score": 0.03,
      "probability": 3.0
    }
  ],
  "visual_features": {
    "colors": ["蓝色", "黄色"],
    "crown": "无冠羽",
    "beak": "粗壮黑色",
    "patterns": [],
    "notes": "大型鹦鹉"
  },
  "visual_features_description": "颜色: 蓝色, 黄色 | 冠羽: 无冠羽 | 喙: 粗壮黑色 | 其他: 大型鹦鹉",
  "raw_text": "..."
}
```

#### 3. 完整分析

```http
POST /analyze
Content-Type: multipart/form-data
```

**请求**：
- `image`: 图片文件

**响应示例**：
```json
{
  "classification": {
    "top_candidates": [...],
    "visual_features": {...},
    "visual_features_description": "...",
    "raw_text": "..."
  },
  "knowledge_hits": {
    "蓝黄金刚鹦鹉": {
      "features": [
        "体羽蓝色+黄色胸腹",
        "粗壮黑喙，脸部白色裸区",
        "尾羽长，热带雨林分布"
      ],
      "notes": "学名 Ara ararauna，常见于南美热带雨林。"
    }
  },
  "explanation": "根据视觉特征和知识库对比，该鹦鹉最可能是蓝黄金刚鹦鹉..."
}
```

#### 4. 保存分类结果

```http
POST /save_classified
Content-Type: multipart/form-data
```

**请求**：
- `image`: 图片文件
- `species`: 物种名称（用于创建文件夹）
- `output_path`: 输出路径（可选，默认 `./dataset`）

**响应示例**：
```json
{
  "success": true,
  "file_path": "./dataset/蓝黄金刚鹦鹉/image_20241209_120000.jpg",
  "folder_path": "./dataset/蓝黄金刚鹦鹉",
  "species": "蓝黄金刚鹦鹉",
  "message": "图片已保存到: ./dataset/蓝黄金刚鹦鹉/image_20241209_120000.jpg"
}
```

### 使用 Swagger UI 测试

1. 访问 http://localhost:8000/docs
2. 点击接口展开详情
3. 点击 "Try it out"
4. 上传图片文件
5. 点击 "Execute" 查看结果

## ⚙️ 配置说明

### 模型配置

在 `app.py` 中修改：

```python
MODEL_NAME = "qwen3-vl:2b-thinking-q4_K_M"  # Ollama 模型名称
LLM_TEMPERATURE = 0.3  # 模型温度参数（0-1，越低越确定）
```

### 超时设置

- **API 超时**：600 秒（10 分钟）
- **连接超时**：30 秒
- **读取超时**：600 秒

可在 `langchain_test.py` 和 `app.py` 中调整。

### 知识库配置

当前使用本地字典 `PARROT_KB`，可扩展为：

- **Chroma**：向量数据库
- **FAISS**：Facebook AI 相似性搜索
- **Milvus**：开源向量数据库

## 📁 项目结构

```
Parrot Set/
├── app.py                          # FastAPI 主应用
├── langchain_test.py               # Ollama LLM 包装器
├── qwen_vl_demo.py                 # Qwen3-VL 测试脚本
├── frontend/                        # 前端文件目录
│   ├── index.html                  # 前端主页面
│   ├── style.css                   # 样式文件
│   ├── app.js                      # 前端逻辑
│   └── README.md                   # 前端使用说明
├── Qwen3-VL-2B-Thinking-GGUF/      # 模型文件目录
│   ├── Modelfile                   # Ollama 模型配置
│   └── *.gguf                      # 模型权重文件
├── samples/                        # 测试图片目录
├── dataset/                        # 分类结果保存目录（自动创建）
│   ├── 蓝黄金刚鹦鹉/
│   ├── 玄凤鹦鹉/
│   └── ...
├── README.md                       # 本文件
└── Parrot Set 鹦鹉集.md            # 项目设计文档
```

## 🔧 常见问题

### 1. Ollama 服务连接失败

**错误**：`ConnectionRefusedError` 或 `10061`

**解决方案**：
```bash
# 检查 Ollama 是否运行
ollama list

# 如果未运行，启动服务
ollama serve

# Windows 用户：确保 Ollama 应用已启动
```

### 2. 模型响应超时

**原因**：
- 图片太大
- 模型首次加载需要时间
- CPU 性能不足

**解决方案**：
- 压缩图片到 2MB 以下
- 使用 GPU 加速（如果有 NVIDIA 显卡）
- 增加超时时间（修改 `app.py` 中的 `timeout` 参数）

### 3. JSON 解析失败

**原因**：模型返回的文本格式不符合预期

**解决方案**：
- 查看 `raw_text` 字段获取原始输出
- 调整 prompt 要求模型只返回 JSON
- 检查模型是否正确加载

### 4. 模型未找到

**错误**：`model not found`

**解决方案**：
```bash
# 拉取模型
ollama pull qwen3-vl:2b-thinking-q4_K_M

# 或使用本地模型
cd Qwen3-VL-2B-Thinking-GGUF
ollama create qwen3-vl-2b-local -f Modelfile
```

## 🚧 开发计划

### MVP 已完成 ✅
- [x] FastAPI 后端框架
- [x] 图像分类接口
- [x] 视觉特征提取
- [x] RAG 知识检索
- [x] 二次解释生成

### 待开发功能 🔄
- [x] 前端界面（HTML/CSS/JS）✅
- [x] 图片自动分类存储 ✅
- [x] 批量处理功能 ✅
- [ ] 向量数据库集成（Chroma/FAISS）
- [ ] 更多鹦鹉品种知识库
- [ ] 用户认证系统
- [ ] 历史记录查询
- [ ] 图片预览和编辑功能

## 📝 开发说明

### 添加新的鹦鹉品种

在 `app.py` 的 `PARROT_KB` 字典中添加：

```python
PARROT_KB = {
    "品种名称": {
        "features": [
            "特征1",
            "特征2",
            "特征3"
        ],
        "notes": "备注信息"
    }
}
```

### 扩展为向量数据库

1. 安装依赖：
```bash
pip install chromadb
# 或
pip install faiss-cpu
```

2. 修改 `rag_lookup` 函数，使用向量检索替代字典查找

## 📄 许可证

本项目采用 Apache 2.0 许可证。

## 🙏 致谢

- [Qwen3-VL](https://github.com/QwenLM/Qwen-VL) - 多模态模型
- [Ollama](https://ollama.com/) - 本地模型部署
- [FastAPI](https://fastapi.tiangolo.com/) - Web 框架
- [LangChain](https://www.langchain.com/) - LLM 应用框架

## 📮 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。

---

**注意**：本项目为 MVP 原型，生产环境使用前请进行充分测试和优化。




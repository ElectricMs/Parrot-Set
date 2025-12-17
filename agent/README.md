# Parrot Set Agent 模块

该模块包含了 Parrot Set 应用程序的核心智能逻辑。它采用模块化设计，由一个主 LLM（通常为纯文本模型）进行协调，并调用专门的工具（如视觉分类器）来完成多模态任务。

## 目录结构

```text
agent/
├── __init__.py         # 模块导出
├── config.py           # 配置加载与管理
├── core.py             # AgentService: 主入口点，负责编排工具和 LLM
├── llm.py              # OllamaLLM 包装器和工厂函数
├── models.py           # 用于 API 响应的 Pydantic 数据模型
└── tools/              # 具体功能工具
    ├── __init__.py
    ├── classifier.py   # ClassifierTool: 图像分类工具 (独立 Vision LLM)
    ├── explainer.py    # ExplainerTool: 解释生成工具 (支持使用主 LLM)
    └── search.py       # SearchTool: 知识检索工具
```

## 核心组件

### 1. Agent 服务 (`core.py`)
`AgentService` 类是系统的大脑。它根据配置初始化主模型和各个工具，并提供统一的调用接口。

**主要功能：**
- **主模型 (Main LLM)**: 通常配置为纯文本大模型 (如 Qwen2.5)，用于逻辑推理、规划和生成解释。
- **工具链管理**: 自动加载并初始化 `ClassifierTool` 和 `ExplainerTool`。
- **任务执行**:
    - `classify(image_path)`: 调用分类工具处理图像。
    - `analyze(image_path)`: 执行完整流程：分类 -> 检索 -> 解释。

### 2. 配置管理 (`config.py`)
支持通过项目根目录的 `config.json` 进行精细化配置。你可以为不同的任务指定不同的模型和温度参数。

**配置示例 (`config.json`):**
```json
{
  "agent_settings": {
    "default_model": "qwen2.5:7b-instruct",
    "default_temperature": 0.5,
    "tools": {
      "classifier": {
        "model_name": "qwen3-vl:2b-instruct-q4_K_M", // 必须是视觉模型
        "temperature": 0.1
      },
      "explainer": {
        "model_name": null, // null 表示复用主模型
        "temperature": 0.7
      }
    }
  }
}
```

### 3. LLM 包装器 (`llm.py`)
封装了与本地 **Ollama** API 的交互，支持文本生成和多模态输入。

## 工具集 (`agent/tools/`)

### 分类器 (ClassifierTool) - `classifier.py`
专用的视觉分类工具。
- **模型**: 独立配置，通常使用 Qwen3-VL 等多模态模型。
- **功能**: 识别图像中的鹦鹉品种，并结合本地数据库 (`parrot_db`) 输出结构化数据。

### 搜索器 (SearchTool) - `search.py`
知识检索工具。
- **功能**: 根据分类结果检索相关知识（目前使用硬编码字典，可扩展为向量库）。

### 解释器 (ExplainerTool) - `explainer.py`
解释生成工具。
- **模型**: 可以配置为使用专用的视觉模型，也可以复用 Agent 的主模型（纯文本）。
- **逻辑**: 如果使用主模型，它将基于分类器提取的文本特征（颜色、喙形等）和检索到的知识生成解释。

## 使用示例

```python
import asyncio
from agent.core import get_agent

async def main():
    # 获取 Agent 单例 (自动读取配置文件)
    agent = get_agent()
    
    image_path = "path/to/parrot.jpg"
    
    # 1. 仅分类
    print("正在分类...")
    result = await agent.classify(image_path)
    print(f"预测结果: {result.top_candidates[0].name}")
    
    # 2. 完整分析
    print("\n正在分析...")
    analysis = await agent.analyze(image_path)
    print(f"专家解释: {analysis.explanation}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 扩展 Agent

添加新工具 (例如 "栖息地查询工具")：

1.  在 `agent/tools/` 中创建一个新文件 (例如 `habitat.py`)。
2.  定义一个工具类 (例如 `HabitatTool`)，在 `__init__` 中接收配置。
3.  在 `agent/config.py` 的默认配置中添加该工具的配置项。
4.  在 `agent/core.py` 的 `AgentService` 中初始化该工具。

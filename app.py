"""
Parrot Set 鹦鹉集 - FastAPI 后端服务

这是一个基于 AI 的鹦鹉识别与知识查询系统，使用 Qwen3-VL 多模态模型进行图像识别，
结合 RAG 技术提供详细的物种特征解释。

主要功能：
    - /health              : 健康检查，验证 Ollama 服务状态
    - /classify            : 图像分类接口，返回 top3 候选和视觉特征
    - /analyze             : 完整分析接口，包含分类 + RAG 检索 + 二次解释

技术架构：
    1. 使用 FastAPI 作为 Web 框架
    2. 通过 Ollama 调用本地部署的 Qwen3-VL 多模态模型
    3. 使用 LangChain 封装模型调用
    4. 实现简单的 RAG 检索（可扩展为向量数据库）

依赖安装：
    pip install fastapi uvicorn python-multipart requests langchain langchain-core

模型要求：
    - 需要先安装并启动 Ollama 服务
    - 下载模型：ollama pull qwen3-vl:2b-instruct-q4_K_M
    - 或使用本地 GGUF 文件创建模型

作者：Parrot Set Team
版本：0.1.0 (MVP)
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_test import OllamaLLM  # 复用已有包装器
from parrot_db import get_database  # 导入数据库查询模块

# ========== 日志配置 ==========
# 配置日志系统，记录 INFO 级别及以上的日志信息
# 日志会输出到控制台，包括请求处理、模型调用、错误信息等
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== 配置文件加载 ==========
CONFIG_FILE = Path("config.json")

def load_config() -> Dict[str, Any]:
    """
    加载配置文件
    如果文件不存在，使用默认配置并创建文件
    """
    default_config = {
        "model_settings": {
            "model_name": "qwen3-vl:2b-instruct-q4_K_M",
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 1024
        },
        "api_settings": {
            "timeout": 600,
            "host": "0.0.0.0",
            "port": 8000
        },
        "parrot_knowledge_base": {
            "蓝黄金刚鹦鹉": {
                "features": ["体羽蓝色+黄色胸腹", "粗壮黑喙，脸部白色裸区", "尾羽长，热带雨林分布"],
                "notes": "学名 Ara ararauna，常见于南美热带雨林。"
            },
            "玄凤鹦鹉": {
                "features": ["头顶竖冠羽", "脸颊橙色斑，体羽灰/黄色", "中小型体型，常见宠物鸟"],
                "notes": "学名 Nymphicus hollandicus，澳洲原生。"
            },
            "虎皮鹦鹉": {
                "features": ["体型小，额部平滑无冠羽", "颈部黑色斑点，翅膀有波浪纹", "常见绿色/蓝色羽色"],
                "notes": "学名 Melopsittacus undulatus，常见宠物鸟。"
            }
        }
    }
    
    if not CONFIG_FILE.exists():
        logger.info("配置文件不存在，创建默认配置")
        save_config(default_config)
        return default_config
        
    try:
        config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        # 合并默认配置，确保新字段存在
        # 这里只做简单的第一层合并，实际使用可能需要递归合并
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}，使用默认配置")
        return default_config

def save_config(config: Dict[str, Any]):
    """保存配置到文件"""
    try:
        CONFIG_FILE.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.error(f"保存配置文件失败: {e}")

# 加载配置
APP_CONFIG = load_config()

# ========== FastAPI 应用初始化 ==========
# 创建 FastAPI 应用实例，设置标题和版本号
# title 会显示在 Swagger UI 文档中
app = FastAPI(title="Parrot Set MVP", version="0.1.0")

# ========== CORS 配置 ==========
# 配置跨域资源共享，允许前端从不同端口访问
# 这对于本地开发很重要（前端可能在 8080，后端在 8000）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源（生产环境应限制为具体域名）
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有请求头
)

# 注册知识库路由
# app.include_router(kb_router)  # 已移除本地知识库功能

# ---- 配置 ----
# 从配置文件读取模型配置
MODEL_NAME = APP_CONFIG["model_settings"].get("model_name", "qwen3-vl:2b-instruct-q4_K_M")
LLM_TEMPERATURE = float(APP_CONFIG["model_settings"].get("temperature", 0.3))

# 覆盖环境变量配置（如果存在）
if os.getenv("PARROT_MODEL_NAME"):
    MODEL_NAME = os.getenv("PARROT_MODEL_NAME")
if os.getenv("PARROT_TEMPERATURE"):
    LLM_TEMPERATURE = float(os.getenv("PARROT_TEMPERATURE"))

# ========== 知识库配置 ==========
# 从配置文件读取知识库
PARROT_KB = APP_CONFIG.get("parrot_knowledge_base", {})


# ========== Pydantic 数据模型 ==========
# 使用 Pydantic 定义 API 的请求/响应数据结构
# 优点：自动验证数据类型、生成 API 文档、序列化/反序列化

class TopCandidate(BaseModel):
    """
    候选品种信息
    
    属性：
        name: 鹦鹉品种的中文名称
        score: 置信度分数（0-1 之间，1 表示完全确定）
        probability: 概率百分比（0-100，便于前端展示）
        exists_in_db: 该种类是否存在于数据库中
        db_info: 数据库中的详细信息（如果存在）
    
    示例：
        TopCandidate(name="蓝黄金刚鹦鹉", score=0.87, probability=87.0, exists_in_db=True)
    """
    name: str  # 品种名称
    score: float  # 0-1 之间的置信度分数
    probability: float  # 百分比形式的概率（0-100），由 score * 100 计算得出
    exists_in_db: bool = False  # 是否存在于数据库中
    db_info: Optional[Dict[str, Any]] = None  # 数据库中的详细信息（目、科、学名等）
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "蓝黄金刚鹦鹉",
                "score": 0.87,
                "probability": 87.0,
                "exists_in_db": True,
                "db_info": {
                    "chinese_name": "琉璃金刚鹦鹉",
                    "english_name": "Blue-and-yellow Macaw",
                    "scientific_name": "Ara ararauna",
                    "order": "鹦形目 / Psittaciformes",
                    "family": "鹦鹉科 / African and New World Parrots / Psittacidae"
                }
            }
        }


class VisualFeatures(BaseModel):
    """
    视觉特征提取结果
    
    从图像中提取的关键视觉特征，用于后续的匹配和解释。
    所有字段都是可选的，因为不同图片可能检测到不同的特征。
    
    属性：
        colors: 主要颜色列表，如 ["蓝色", "黄色"]
        crown: 冠羽特征，如 "竖冠羽"、"无冠羽"、"平冠羽"
        beak: 喙的特征描述，如 "粗壮黑色"、"细长灰色"
        patterns: 斑纹或特殊图案列表，如 ["脸颊橙色斑", "颈部黑色斑点"]
        notes: 其他显著特征，如 "大型鹦鹉"、"尾羽长"
    """
    colors: Optional[List[str]] = None  # 主要颜色列表
    crown: Optional[str] = None  # 冠羽特征
    beak: Optional[str] = None  # 喙的特征
    patterns: Optional[List[str]] = None  # 斑纹/图案列表
    notes: Optional[str] = None  # 其他特征备注
    
    def to_description(self) -> str:
        """
        将视觉特征转换为可读的描述文本
        
        将结构化的特征数据转换为人类可读的字符串格式，
        用于前端展示或日志输出。
        
        返回格式示例：
            "颜色: 蓝色, 黄色 | 冠羽: 无冠羽 | 喙: 粗壮黑色 | 斑纹: 无"
        
        Returns:
            str: 格式化的特征描述字符串，如果没有特征则返回 "未检测到显著特征"
        """
        parts = []
        if self.colors:
            parts.append(f"颜色: {', '.join(self.colors)}")
        if self.crown:
            parts.append(f"冠羽: {self.crown}")
        if self.beak:
            parts.append(f"喙: {self.beak}")
        if self.patterns:
            parts.append(f"斑纹: {', '.join(self.patterns)}")
        if self.notes:
            parts.append(f"其他: {self.notes}")
        return " | ".join(parts) if parts else "未检测到显著特征"


class ClassificationResult(BaseModel):
    """
    图像分类结果
    
    包含模型对图像的识别结果和提取的视觉特征。
    这是 /classify 接口的返回数据结构。
    
    属性：
        top_candidates: top3 候选品种列表，按置信度从高到低排序
        visual_features: 结构化的视觉特征对象（可选）
        visual_features_description: 格式化的特征描述字符串（可选，便于前端展示）
        raw_text: 模型返回的原始文本（用于调试和错误排查）
    """
    top_candidates: List[TopCandidate]  # top3 候选品种，按置信度排序
    visual_features: Optional[VisualFeatures] = None  # 结构化的视觉特征
    visual_features_description: Optional[str] = None  # 格式化的特征描述（由 to_description() 生成）
    raw_text: str  # 模型原始输出，用于调试
    
    class Config:
        json_schema_extra = {
            "example": {
                "top_candidates": [
                    {"name": "蓝黄金刚鹦鹉", "score": 0.87, "probability": 87.0},
                    {"name": "玄凤鹦鹉", "score": 0.1, "probability": 10.0},
                    {"name": "虎皮鹦鹉", "score": 0.03, "probability": 3.0}
                ],
                "visual_features_description": "颜色: 蓝色, 黄色 | 冠羽: 无冠羽 | 喙: 粗壮黑色 | 斑纹: 无"
            }
        }


class AnalyzeResult(BaseModel):
    """
    完整分析结果
    
    包含分类结果、知识库检索结果和生成的解释文本。
    这是 /analyze 接口的返回数据结构。
    
    属性：
        classification: 图像分类结果（包含 top3 候选和视觉特征）
        knowledge_hits: RAG 检索到的知识库片段，key 为品种名称
        explanation: 模型生成的详细解释文本，说明分类依据和特征匹配情况
    """
    classification: ClassificationResult  # 分类结果
    knowledge_hits: Dict[str, Any]  # 知识库检索命中结果，格式：{品种名: {features: [...], notes: "..."}}
    explanation: str  # 生成的解释文本


# ========== 工具函数 ==========

def save_upload_temp(upload: UploadFile) -> Path:
    """
    保存上传的文件到临时目录
    
    将 FastAPI 接收到的上传文件保存到本地临时文件，
    供后续的模型处理使用。文件会在处理完成后自动删除。
    
    Args:
        upload: FastAPI 的 UploadFile 对象，包含上传的文件数据
    
    Returns:
        Path: 临时文件的路径对象
    
    Raises:
        HTTPException: 如果文件为空，返回 400 错误
    
    注意：
        - 临时文件保存在项目根目录，文件名格式：tmp_upload.{扩展名}
        - 调用方需要负责在完成后删除临时文件
    """
    # 读取上传文件的数据
    data = upload.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="空文件")
    
    # 获取文件扩展名，如果没有则默认为 .jpg
    suffix = Path(upload.filename or "").suffix or ".jpg"
    
    # 创建临时文件路径
    tmp_path = Path("tmp_upload" + suffix)
    
    # 将数据写入临时文件
    tmp_path.write_bytes(data)
    
    return tmp_path


def save_classified_image(
    upload: UploadFile,
    species_name: str,
    output_path: str = "./dataset"
) -> Dict[str, Any]:
    """
    保存分类后的图片到指定文件夹
    
    根据识别出的鹦鹉品种，将图片保存到对应的文件夹中。
    如果文件夹不存在，会自动创建。
    
    工作流程：
        1. 验证物种名称和输出路径
        2. 清理物种名称（移除特殊字符，确保文件夹名合法）
        3. 创建目标文件夹（如果不存在）
        4. 生成唯一文件名（避免覆盖）
        5. 保存图片文件
    
    Args:
        upload: FastAPI 的 UploadFile 对象，包含要保存的图片
        species_name: 识别出的鹦鹉品种名称（用于创建文件夹）
        output_path: 输出根目录路径，默认为 "./dataset"
                    - 相对路径：相对于项目根目录
                    - 绝对路径：如 "E:\\Project\\Parrot Set\\dataset"
    
    Returns:
        Dict[str, Any]: 包含保存信息的字典
            - success: 是否成功
            - file_path: 保存的文件路径
            - folder_path: 目标文件夹路径
            - species: 物种名称
    
    Raises:
        HTTPException:
            - 400: 参数无效（空文件、无效路径等）
            - 500: 文件保存失败
    
    示例：
        save_classified_image(upload, "蓝黄金刚鹦鹉", "./dataset")
        # 结果：图片保存到 ./dataset/蓝黄金刚鹦鹉/xxx.jpg
    """
    import shutil
    from datetime import datetime
    
    # ========== 参数验证 ==========
    if not species_name or not species_name.strip():
        raise HTTPException(status_code=400, detail="物种名称不能为空")
    
    # 读取文件数据
    upload.file.seek(0)  # 重置文件指针（可能已被读取过）
    data = upload.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="空文件")
    
    # ========== 路径处理 ==========
    # 清理物种名称：移除特殊字符，确保文件夹名合法
    # Windows 不允许的字符：< > : " / \ | ? *
    clean_species = "".join(
        c for c in species_name.strip() 
        if c not in '<>:"/\\|?*'
    ).strip()
    
    if not clean_species:
        clean_species = "未知品种"
    
    # 构建目标文件夹路径
    output_dir = Path(output_path).expanduser().resolve()  # 支持 ~ 和相对路径
    species_dir = output_dir / clean_species
    
    # 创建文件夹（如果不存在）
    try:
        species_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"创建/确认文件夹: {species_dir}")
    except Exception as e:
        logger.error(f"创建文件夹失败: {e}")
        raise HTTPException(status_code=500, detail=f"无法创建文件夹: {str(e)}")
    
    # ========== 文件保存 ==========
    # 生成唯一文件名：原文件名 + 时间戳（避免覆盖）
    original_name = upload.filename or "image"
    file_ext = Path(original_name).suffix or ".jpg"
    file_stem = Path(original_name).stem
    
    # 如果文件名已存在，添加时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{file_stem}_{timestamp}{file_ext}"
    target_path = species_dir / file_name
    
    # 如果仍然存在，添加随机数
    counter = 1
    while target_path.exists():
        file_name = f"{file_stem}_{timestamp}_{counter}{file_ext}"
        target_path = species_dir / file_name
        counter += 1
    
    # 保存文件
    try:
        target_path.write_bytes(data)
        logger.info(f"图片已保存: {target_path}")
    except Exception as e:
        logger.error(f"保存文件失败: {e}")
        raise HTTPException(status_code=500, detail=f"保存文件失败: {str(e)}")
    
    return {
        "success": True,
        "file_path": str(target_path),
        "folder_path": str(species_dir),
        "species": clean_species,
        "message": f"图片已保存到: {target_path}"
    }


def ensure_llm(model_name: str = MODEL_NAME, temperature: float = LLM_TEMPERATURE) -> OllamaLLM:
    """
    创建并返回 Ollama LLM 实例
    
    这是一个工厂函数，用于统一创建 LLM 实例。
    可以在这里添加连接池、缓存等优化逻辑。
    
    Args:
        model_name: Ollama 模型名称，默认使用全局配置 MODEL_NAME
        temperature: 模型温度参数（0-1），默认使用全局配置 LLM_TEMPERATURE
                    - 0.0-0.3: 更确定、更保守的输出
                    - 0.4-0.7: 平衡的创造性和准确性
                    - 0.8-1.0: 更随机、更有创造性的输出
    
    Returns:
        OllamaLLM: 配置好的 LLM 实例，可以调用 _call 方法进行推理
    """
    return OllamaLLM(model_name=model_name, temperature=temperature)


def llm_classify_image(image_path: Path, llm: OllamaLLM) -> ClassificationResult:
    """
    调用多模态模型进行图像分类和特征提取
    
    这是核心的分类函数，执行以下步骤：
    1. 构建 prompt，要求模型输出结构化的 JSON
    2. 调用模型进行推理（传入图片路径和 prompt）
    3. 解析模型返回的 JSON 数据
    4. 转换为 ClassificationResult 对象
    
    Args:
        image_path: 图片文件的路径（Path 对象）
        llm: OllamaLLM 实例，用于调用模型
    
    Returns:
        ClassificationResult: 包含 top3 候选、视觉特征和原始文本的分类结果
    
    Raises:
        HTTPException: 
            - 503: Ollama 服务连接失败
            - 504: 模型响应超时
            - 500: 其他模型调用错误
    
    注意：
        - 如果 JSON 解析失败，会返回空的结构但保留原始文本（便于调试）
        - 模型可能返回包含额外文字的 JSON，代码会自动提取 JSON 部分
    """
    # 构建 prompt：要求模型输出结构化的 JSON 格式
    # 明确指定输出格式，提高解析成功率
    prompt = """
识别这张鹦鹉图片，输出 JSON：

{
  "top_candidates": [
    {"name": "物种名", "score": 概率分数},
    {"name": "物种名", "score": 概率分数},
    {"name": "物种名", "score": 概率分数}
  ],
  "visual_features": {
    "colors": ["主要颜色"],
    "crown": "冠羽/无冠羽",
    "beak": "喙特征",
    "patterns": ["斑纹"],
    "notes": "其他特征"
  }
}

只返回 JSON，不要其他文字。
"""
    full_prompt = prompt.strip()
    
    # ========== 调用模型进行推理 ==========
    try:
        logger.info(f"开始调用模型识别图片: {image_path}")
        # 调用 LLM，传入 prompt 和图片路径
        # images 参数接受文件路径列表，模型会读取并处理图片
        raw_text = llm._call(full_prompt, images=[str(image_path)])
        logger.info(f"模型返回文本长度: {len(raw_text)}")
    # ========== 错误处理 ==========
    except ConnectionError as e:
        # 处理连接错误：Ollama 服务未运行或无法连接
        error_msg = str(e)
        # Windows 错误码 10061 表示连接被拒绝
        if "10061" in error_msg or "积极拒绝" in error_msg or "refused" in error_msg.lower():
            logger.error("Ollama 服务未运行")
            raise HTTPException(
                status_code=503,  # Service Unavailable
                detail="Ollama 服务未运行。请先启动 Ollama 服务：\n1. 检查 Ollama 是否已安装\n2. 运行 'ollama serve' 或启动 Ollama 应用\n3. 确认服务运行在 http://127.0.0.1:11434"
            )
        raise HTTPException(status_code=503, detail=f"无法连接到 Ollama 服务: {str(e)}")
    except Exception as e:
        # 处理其他异常：超时、模型错误等
        logger.error(f"模型调用失败: {e}", exc_info=True)
        error_msg = str(e)
        if "timeout" in error_msg.lower() or "超时" in error_msg:
            raise HTTPException(status_code=504, detail=f"模型响应超时: {str(e)}")  # Gateway Timeout
        raise HTTPException(status_code=500, detail=f"模型调用失败: {str(e)}")  # Internal Server Error

    # ========== JSON 解析 ==========
    # 模型可能返回纯 JSON，也可能返回包含额外文字的文本
    # 需要智能提取 JSON 部分
    try:
        raw_text_clean = raw_text.strip()
        
        # 策略1：尝试找到第一个 { 和最后一个 }，提取中间的 JSON
        # 这样可以处理模型返回 "这是结果：{...}" 的情况
        start_idx = raw_text_clean.find("{")
        end_idx = raw_text_clean.rfind("}") + 1
        
        if start_idx >= 0 and end_idx > start_idx:
            # 提取 JSON 字符串
            json_str = raw_text_clean[start_idx:end_idx]
            data = json.loads(json_str)
        else:
            # 策略2：如果找不到大括号，尝试直接解析整个文本
            data = json.loads(raw_text_clean)
    except json.JSONDecodeError as e:
        # JSON 解析失败：可能是模型返回了非 JSON 格式的文本
        # 返回空结构但保留原始文本，便于调试和查看模型实际输出
        logger.warning(f"JSON 解析失败，返回原始文本: {e}")
        return ClassificationResult(
            top_candidates=[],
            visual_features=None,
            visual_features_description=None,
            raw_text=raw_text.strip(),
        )

    # ========== 数据转换 ==========
    # 将解析后的字典数据转换为 Pydantic 模型对象
    
    # 获取数据库实例用于验证
    db = get_database()
    
    # 转换 top_candidates：列表推导式创建 TopCandidate 对象，并验证是否存在于数据库
    candidates = []
    for c in data.get("top_candidates", []):  # 从数据中获取候选列表，默认为空列表
        name = c.get("name", "").strip()
        if not name:
            continue
        
        # 查询数据库
        record = db.find_by_name(name)
        exists_in_db = record is not None
        
        # 构建数据库信息
        db_info = None
        if record:
            db_info = {
                "chinese_name": record['chinese_name'],
                "english_name": record['english_name'],
                "scientific_name": record['scientific_name'],
                "order": record['order'],
                "family": record['family'],
                "link": record['link'],
            }
        
        candidates.append(
        TopCandidate(
                name=name,  # 品种名称
            score=float(c.get("score", 0)),  # 置信度分数，转换为 float
                probability=round(float(c.get("score", 0)) * 100, 2),  # 转换为百分比，保留2位小数
                exists_in_db=exists_in_db,  # 是否存在于数据库
                db_info=db_info  # 数据库信息
        )
        )
    
    # 转换 visual_features：创建 VisualFeatures 对象
    vf = data.get("visual_features") or {}  # 获取视觉特征字典，如果不存在则使用空字典
    visual = VisualFeatures(
        colors=vf.get("colors"),  # 颜色列表
        crown=vf.get("crown"),  # 冠羽特征
        beak=vf.get("beak"),  # 喙特征
        patterns=vf.get("patterns"),  # 斑纹列表
        notes=vf.get("notes"),  # 其他备注
    )
    
    # 生成格式化的特征描述字符串（用于前端展示）
    visual_description = visual.to_description() if visual else None
    
    return ClassificationResult(
        top_candidates=candidates,
        visual_features=visual,
        visual_features_description=visual_description,
        raw_text=raw_text.strip(),
    )


def rag_lookup(candidates: List[TopCandidate]) -> Dict[str, Any]:
    """
    RAG 检索：仅使用硬编码知识库
    
    1. 根据候选品种名称查找硬编码知识库 (PARROT_KB)
    
    Args:
        candidates: 候选品种列表（通常来自分类结果）
    
    Returns:
        Dict[str, Any]: 检索到的知识库片段
                       key: 品种名称
                       value: 包含 features, notes 的字典
    """
    hits = {}  # 存储检索到的知识库条目
    
    # 遍历所有候选品种
    for cand in candidates:
        entry = {}
        
        # 1. 查找硬编码知识库
        if cand.name in PARROT_KB:
            entry.update(PARROT_KB[cand.name])
            
        if entry:
            hits[cand.name] = entry
    
    return hits


def llm_explain(
    classification: ClassificationResult,
    knowledge_hits: Dict[str, Any],
    image_path: Path,
    llm: OllamaLLM,
) -> str:
    """
    生成分类解释：结合视觉特征和知识库生成详细说明
    
    这是二次推理函数，在初步分类的基础上，结合知识库信息，
    生成更详细、更有说服力的分类解释。
    
    工作流程：
        1. 格式化分类结果和知识库信息
        2. 构建包含所有信息的 prompt
        3. 调用模型生成解释文本
        4. 返回生成的解释
    
    Args:
        classification: 初步分类结果，包含 top3 候选和视觉特征
        knowledge_hits: RAG 检索到的知识库片段
        image_path: 图片路径（用于二次推理时再次查看图片）
        llm: OllamaLLM 实例
    
    Returns:
        str: 生成的解释文本，说明分类依据、特征匹配情况等
    
    Raises:
        HTTPException:
            - 503: Ollama 服务连接失败
            - 504: 模型响应超时
            - 500: 其他错误
    """
    # ========== 格式化输入数据 ==========
    
    # 格式化 top3 候选：转换为易读的列表格式
    top_str = "\n".join(
        [f"- {c.name} (score {c.score})" for c in classification.top_candidates]
    )
    
    # 格式化知识库信息：将字典转换为字符串
    kb_parts = []
    for k, v in knowledge_hits.items():
        # 处理硬编码特征
        if "features" in v:
            kb_parts.append(f"{k} 特征: {', '.join(v['features'])}")
        if "notes" in v:
            kb_parts.append(f"{k} 备注: {v['notes']}")
            
        # 处理向量检索结果
        if "vector_docs" in v:
            for doc in v["vector_docs"]:
                # 截断过长的内容
                content = doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
                kb_parts.append(f"{k} 参考资料 ({doc['source']}): {content}")
                
    kb_str = "\n".join(kb_parts) or "无命中"
    
    # 格式化视觉特征：转换为 JSON 字符串
    vf = classification.visual_features
    vf_str = vf.json() if vf else "null"  # 如果有视觉特征则转为 JSON，否则为 "null"

    # ========== 构建解释 prompt ==========
    # 要求模型作为鸟类学专家，结合所有信息生成简洁的解释
    prompt = f"""
你是鸟类学专家。根据以下信息简洁解释图像分类：

初步判定：
{top_str}

视觉特征：
{vf_str}

知识库：
{kb_str}

请用3-5句话说明：
1) 最可能物种及依据（引用视觉特征与知识库）
2) 主要不确定性
3) 与第二候选的差异

保持简洁，控制在200字以内。
"""
    try:
        logger.info("开始生成解释...")
        result = llm._call(prompt.strip(), images=[str(image_path)])
        logger.info(f"解释生成完成，长度: {len(result)}")
        return result
    except ConnectionError as e:
        error_msg = str(e)
        if "10061" in error_msg or "积极拒绝" in error_msg or "refused" in error_msg.lower():
            logger.error("Ollama 服务未运行")
            raise HTTPException(
                status_code=503,
                detail="Ollama 服务未运行。请先启动 Ollama 服务。"
            )
        raise HTTPException(status_code=503, detail=f"无法连接到 Ollama 服务: {str(e)}")
    except Exception as e:
        logger.error(f"解释生成失败: {e}", exc_info=True)
        error_msg = str(e)
        if "timeout" in error_msg.lower() or "超时" in error_msg:
            raise HTTPException(status_code=504, detail=f"模型响应超时: {str(e)}")
        raise HTTPException(status_code=500, detail=f"解释生成失败: {str(e)}")


# ========== API 路由 ==========

@app.get("/health")
async def health():
    """
    健康检查接口
    
    检查服务状态和 Ollama 服务的可用性。
    可以用于监控、负载均衡器健康检查等场景。
    
    返回信息：
        - status: 服务状态（"ok" 表示正常）
        - ollama_available: Ollama 服务是否可用
        - model: 当前配置的模型名称
    
    Returns:
        Dict: 包含服务状态信息的字典
    
    使用场景：
        - 启动时检查依赖服务
        - 监控系统定期检查
        - 前端显示服务状态
    """
    try:
        import requests
        # 尝试访问 Ollama 的 API，检查服务是否运行
        # /api/tags 是 Ollama 的模型列表接口，响应快且不需要认证
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        ollama_ok = response.status_code == 200  # 200 表示服务正常
    except Exception as e:
        # 连接失败或超时，说明 Ollama 服务不可用
        logger.warning(f"Ollama 服务检查失败: {e}")
        ollama_ok = False
    
    return {
        "status": "ok",  # 本服务正常
        "ollama_available": ollama_ok,  # Ollama 服务状态
        "model": MODEL_NAME  # 当前使用的模型
    }


@app.post("/classify", response_model=ClassificationResult)
async def classify(image: UploadFile = File(...)):
    """
    图像分类接口：初步判定鹦鹉品种
    
    这是核心的分类接口，接收用户上传的鹦鹉图片，调用多模态模型进行识别，
    返回 top3 候选品种及其置信度、视觉特征等信息。
    
    工作流程：
        1. 检查 Ollama 服务是否可用（前置检查，快速失败）
        2. 保存上传的图片到临时文件
        3. 调用模型进行图像分类和特征提取（异步执行，避免阻塞）
        4. 解析模型返回的 JSON 数据
        5. 清理临时文件
        6. 返回结构化的分类结果
    
    Args:
        image: FastAPI 的 UploadFile 对象，包含用户上传的图片文件
              - 支持常见图片格式：JPG、PNG、GIF 等
              - 建议图片大小 < 10MB，以获得更好的性能
    
    Returns:
        ClassificationResult: 包含以下信息：
            - top_candidates: top3 候选品种列表，每个包含 name、score、probability
            - visual_features: 提取的视觉特征（颜色、冠羽、喙、斑纹等）
            - visual_features_description: 格式化的特征描述字符串
            - raw_text: 模型原始输出（用于调试）
    
    Raises:
        HTTPException:
            - 503: Ollama 服务未运行或无法连接
            - 504: 请求超时（模型响应时间超过 600 秒）
            - 500: 其他内部错误（模型调用失败、JSON 解析失败等）
    
    性能说明：
        - 模型推理时间：通常 10-60 秒（取决于图片大小和硬件性能）
        - 超时设置：600 秒（10 分钟），多模态模型处理图片需要较长时间
        - 异步处理：使用 asyncio.to_thread 在线程池中执行同步调用，不阻塞事件循环
    
    使用示例：
        ```python
        import requests
        
        with open("parrot.jpg", "rb") as f:
            response = requests.post(
                "http://localhost:8000/classify",
                files={"image": f}
            )
        result = response.json()
        print(result["top_candidates"][0]["name"])  # 输出最可能的品种
        ```
    """
    # ========== 前置检查：Ollama 服务可用性 ==========
    # 在开始处理前先检查 Ollama 服务，避免浪费资源处理无效请求
    # 使用短超时（2秒）快速失败，提供更好的用户体验
    import requests
    try:
        requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
    except requests.exceptions.ConnectionError:
        # 连接失败，说明 Ollama 服务未运行
        raise HTTPException(
            status_code=503,  # Service Unavailable
            detail="Ollama 服务未运行。请先启动 Ollama 服务。"
        )
    
    # ========== 文件处理 ==========
    tmp_path = None  # 临时文件路径，用于 finally 块清理
    try:
        logger.info(f"收到分类请求，文件名: {image.filename}")
        
        # 保存上传文件到临时目录
        # 注意：临时文件会在 finally 块中自动删除
        tmp_path = save_upload_temp(image)
        logger.info(f"文件已保存到: {tmp_path}")
        
        # 创建 LLM 实例
        # 使用工厂函数确保配置一致性
        llm = ensure_llm()
        
        # ========== 异步模型调用 ==========
        # 关键：使用 asyncio.to_thread 将同步的模型调用放到线程池执行
        # 原因：
        #   1. llm._call 是同步函数，会阻塞当前线程
        #   2. FastAPI 是异步框架，阻塞会严重影响并发性能
        #   3. 在线程池中执行可以避免阻塞事件循环，保持服务响应性
        #
        # asyncio.wait_for 设置超时保护：
        #   - 如果模型响应超过 600 秒，自动取消并抛出 TimeoutError
        #   - 防止长时间等待导致资源浪费
        result = await asyncio.wait_for(
            asyncio.to_thread(llm_classify_image, tmp_path, llm),
            timeout=600.0  # 10 分钟超时
        )
        logger.info("分类完成")
        return result
        
    # ========== 异常处理 ==========
    except asyncio.TimeoutError:
        # 超时异常：模型响应时间过长
        logger.error("分类请求超时（600秒）")
        raise HTTPException(
            status_code=504,  # Gateway Timeout
            detail="请求超时，模型响应时间过长。请检查 Ollama 服务是否正常运行，或尝试使用更小的图片。"
        )
    except HTTPException:
        # 重新抛出 HTTPException，保持错误信息不变
        # 这些异常已经在内部函数中处理过，包含详细的错误信息
        raise
    except Exception as e:
        # 捕获所有其他异常，记录详细日志并返回通用错误信息
        # exc_info=True 会记录完整的堆栈跟踪，便于调试
        logger.error(f"分类失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,  # Internal Server Error
            detail=f"分类失败: {str(e)}"
        )
    finally:
        # ========== 资源清理 ==========
        # 无论成功还是失败，都要删除临时文件
        # 避免磁盘空间浪费和潜在的安全问题
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)  # missing_ok=True 避免文件不存在时报错
            logger.info(f"临时文件已删除: {tmp_path}")


@app.post("/save_classified")
async def save_classified(
    image: UploadFile = File(...),
    species: str = Form(...),
    output_path: str = Form("./dataset")
):
    """
    保存分类后的图片到指定文件夹
    
    根据识别出的鹦鹉品种，将图片保存到对应的文件夹中。
    这个接口通常由前端在识别完成后调用。
    
    Args:
        image: 要保存的图片文件
        species: 识别出的鹦鹉品种名称（用于创建文件夹）
        output_path: 输出根目录路径，默认为 "./dataset"
    
    Returns:
        Dict: 包含保存信息的字典
    
    使用示例：
        ```python
        import requests
        
        with open("parrot.jpg", "rb") as f:
            response = requests.post(
                "http://localhost:8000/save_classified",
                files={"image": f},
                data={
                    "species": "蓝黄金刚鹦鹉",
                    "output_path": "./dataset"
                }
            )
        ```
    """
    # species 参数已通过 Form(...) 设置为必填，无需再次检查
    try:
        result = save_classified_image(image, species, output_path)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"保存分类图片失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"保存失败: {str(e)}")


@app.post("/analyze", response_model=AnalyzeResult)
async def analyze(image: UploadFile = File(...)):
    """
    完整分析接口：图像分类 + RAG 检索 + 二次解释生成
    
    这是功能最完整的接口，执行三步流程：
        1. 图像分类：识别鹦鹉品种并提取视觉特征
        2. RAG 检索：根据分类结果从知识库检索相关品种信息
        3. 解释生成：结合视觉特征和知识库，生成详细的分类解释
    
    与 /classify 的区别：
        - /classify: 只执行第一步，返回分类结果和视觉特征
        - /analyze: 执行完整流程，额外包含知识库检索和解释生成
        - /analyze 需要调用模型两次（分类 + 解释），耗时更长但信息更丰富
    
    工作流程详解：
        ┌─────────────────┐
        │  1. 图像分类     │ → 调用模型识别图片，得到 top3 候选和视觉特征
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │  2. RAG 检索     │ → 根据候选品种名称查找知识库（同步，很快）
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │  3. 生成解释     │ → 调用模型生成详细解释（结合分类结果和知识库）
        └─────────────────┘
    
    Args:
        image: FastAPI 的 UploadFile 对象，包含用户上传的图片文件
    
    Returns:
        AnalyzeResult: 包含以下信息：
            - classification: 分类结果（与 /classify 返回相同）
            - knowledge_hits: RAG 检索到的知识库片段
            - explanation: 模型生成的详细解释文本
    
    Raises:
        HTTPException:
            - 503: Ollama 服务未运行
            - 504: 请求超时（任何一步超过 600 秒）
            - 500: 其他内部错误
    
    性能说明：
        - 总耗时：通常 20-120 秒（两次模型调用）
        - 第一步（分类）：10-60 秒
        - 第二步（RAG）：< 1 秒（本地字典查找）
        - 第三步（解释）：10-60 秒
        - 超时设置：每步 600 秒
    
    使用场景：
        - 需要详细解释和依据的场景
        - 教育、科普应用
        - 需要对比多个候选品种的场景
    
    使用示例：
        ```python
        import requests
        
        with open("parrot.jpg", "rb") as f:
            response = requests.post(
                "http://localhost:8000/analyze",
                files={"image": f}
            )
        result = response.json()
        print(result["explanation"])  # 输出详细解释
        ```
    """
    # ========== 前置检查：Ollama 服务可用性 ==========
    # 与 /classify 相同的前置检查逻辑
    import requests
    try:
        requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Ollama 服务未运行。请先启动 Ollama 服务。"
        )
    
    # ========== 文件处理 ==========
    tmp_path = None
    try:
        logger.info(f"收到分析请求，文件名: {image.filename}")
        tmp_path = save_upload_temp(image)
        logger.info(f"文件已保存到: {tmp_path}")
        
        llm = ensure_llm()
        
        # ========== 第一步：图像分类 ==========
        # 调用模型识别图片，获取 top3 候选和视觉特征
        # 这一步与 /classify 接口的逻辑完全相同
        logger.info("开始第一步：图像分类...")
        classification = await asyncio.wait_for(
            asyncio.to_thread(llm_classify_image, tmp_path, llm),
            timeout=600.0  # 10 分钟超时
        )
        # 记录分类结果，便于调试和监控
        top1_name = classification.top_candidates[0].name if classification.top_candidates else 'N/A'
        logger.info(f"分类完成，top1: {top1_name}")
        
        # ========== 第二步：RAG 检索 ==========
        # 根据分类结果中的候选品种名称，从知识库中检索相关信息
        # 注意：这是同步操作，但速度很快（字典查找），不需要异步处理
        # 如果未来扩展为向量数据库，可能需要异步处理
        hits = rag_lookup(classification.top_candidates)
        logger.info(f"RAG 检索完成，命中: {list(hits.keys())}")
        
        # ========== 第三步：生成解释 ==========
        # 结合分类结果和知识库信息，调用模型生成详细的解释文本
        # 这一步会再次查看图片，结合所有信息进行推理
        logger.info("开始第三步：生成解释...")
        explanation = await asyncio.wait_for(
            asyncio.to_thread(llm_explain, classification, hits, tmp_path, llm),
            timeout=600.0  # 10 分钟超时
        )
        logger.info("分析完成")
        
        # ========== 组装返回结果 ==========
        return AnalyzeResult(
            classification=classification,  # 第一步的分类结果
            knowledge_hits=hits,  # 第二步的检索结果
            explanation=explanation.strip(),  # 第三步生成的解释（去除首尾空白）
        )
        
    # ========== 异常处理 ==========
    # 异常处理逻辑与 /classify 相同
    except asyncio.TimeoutError:
        logger.error("分析请求超时（600秒）")
        raise HTTPException(
            status_code=504,
            detail="请求超时，模型响应时间过长。请检查 Ollama 服务是否正常运行，或尝试使用更小的图片。"
        )
    except HTTPException:
        raise  # 重新抛出 HTTPException
    except Exception as e:
        logger.error(f"分析失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")
    finally:
        # ========== 资源清理 ==========
        # 确保临时文件被删除
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
            logger.info(f"临时文件已删除: {tmp_path}")


@app.get("/check_species")
async def check_species(name: str):
    """
    检查鹦鹉种类是否存在
    
    根据提供的名称（支持中文名、英文名、学名）查询数据库中是否存在该种类。
    
    Args:
        name: 鹦鹉名称（中文、英文或学名）
    
    Returns:
        Dict: 包含以下信息：
            - exists: 是否存在
            - info: 详细信息（如果存在）
            - suggestions: 模糊匹配建议（如果不存在）
    
    示例：
        GET /check_species?name=蓝黄金刚鹦鹉
        GET /check_species?name=Blue-and-yellow Macaw
        GET /check_species?name=Ara ararauna
    """
    db = get_database()
    
    # 精确匹配
    record = db.find_by_name(name)
    if record:
        return {
            "exists": True,
            "info": {
                "chinese_name": record['chinese_name'],
                "english_name": record['english_name'],
                "scientific_name": record['scientific_name'],
                "order": record['order'],
                "family": record['family'],
                "link": record['link'],
                "image": record['image'],
            },
            "suggestions": []
        }
    
    # 如果精确匹配失败，提供模糊搜索建议
    suggestions = db.fuzzy_search(name, limit=5)
    suggestion_list = [
        {
            "chinese_name": s['chinese_name'],
            "english_name": s['english_name'],
            "scientific_name": s['scientific_name'],
        }
        for s in suggestions
    ]
    
    return {
        "exists": False,
        "info": None,
        "suggestions": suggestion_list
    }


@app.get("/search_species")
async def search_species(query: str, limit: int = 10):
    """
    模糊搜索鹦鹉种类
    
    根据关键词搜索匹配的鹦鹉种类（支持中文、英文、学名部分匹配）。
    
    Args:
        query: 搜索关键词
        limit: 返回结果数量限制（默认10）
    
    Returns:
        Dict: 包含匹配结果列表
    
    示例：
        GET /search_species?query=金刚鹦鹉
        GET /search_species?query=Macaw&limit=5
    """
    db = get_database()
    results = db.fuzzy_search(query, limit=limit)
    
    return {
        "count": len(results),
        "results": [
            {
                "chinese_name": r['chinese_name'],
                "english_name": r['english_name'],
                "scientific_name": r['scientific_name'],
                "order": r['order'],
                "family": r['family'],
                "link": r['link'],
            }
            for r in results
        ]
    }


@app.get("/stats/species")
async def get_species_stats(output_path: str = "./dataset"):
    """
    获取品种统计信息
    
    返回所有已知品种（基于知识库）及其收集状态（基于文件系统）。
    用于前端展示分类树。
    """
    stats = []
    
    # 1. 获取所有已知品种（从知识库）
    known_species = list(PARROT_KB.keys())
    
    # 2. 扫描数据集目录，获取已收集的品种和数量
    output_dir = Path(output_path).expanduser().resolve()
    collected_data = {}
    
    if output_dir.exists():
        for item in output_dir.iterdir():
            if item.is_dir():
                # 统计该目录下的图片数量
                count = sum(1 for f in item.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp'])
                if count > 0:
                    collected_data[item.name] = count
    
    # 3. 合并数据
    # 先添加知识库中的品种
    for species in known_species:
        count = collected_data.get(species, 0)
        stats.append({
            "name": species,
            "count": count,
            "collected": count > 0,
            "source": "knowledge_base"
        })
    
    # 再添加知识库中没有但文件夹中存在的品种（可能是未知品种或新发现）
    for species, count in collected_data.items():
        if species not in known_species:
            stats.append({
                "name": species,
                "count": count,
                "collected": True,
                "source": "dataset"
            })
    
    # 按是否收集和名称排序
    stats.sort(key=lambda x: (not x["collected"], x["name"]))
    
    return {
        "total_species": len(stats),
        "collected_species": len([s for s in stats if s["collected"]]),
        "species_list": stats
    }


# ========== 主程序入口 ==========
# 当直接运行此文件时（python app.py），启动 FastAPI 开发服务器
if __name__ == "__main__":
    import uvicorn
    
    # 启动 Uvicorn 服务器
    # 参数说明：
    #   - "app:app": 模块路径，app 是模块名，第二个 app 是 FastAPI 实例名
    #   - host="0.0.0.0": 监听所有网络接口，允许外部访问
    #                     如果只想本地访问，使用 "127.0.0.1"
    #   - port=8000: 服务端口号
    #   - reload=True: 开发模式，代码修改后自动重启（生产环境应设为 False）
    #
    # 生产环境建议使用：
    #   uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
    #   其中 --workers 4 表示启动 4 个工作进程，提高并发性能
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

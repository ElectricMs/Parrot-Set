"""
RAG（Retrieval-Augmented Generation）检索引擎

本模块提供一个基于 ChromaDB 的本地向量检索引擎，用于：
- 文本问答时检索 knowledge/ 下的文档片段（agent/router.py::run_ask）
- 低置信度识图时，基于“视觉特征描述/候选物种”检索资料（agent/router.py::run_analyze）
- 旧版 analyze 流水线的检索（agent/tools/search.py）

核心组件：
- QwenEmbeddingFunction：将文本转 embedding（HuggingFace transformers）
- RAGEngine：负责加载知识库文件、切片、构建/复用索引、以及 query 检索

索引持久化：
- 默认存放在 data/chroma_db（包含 chroma.sqlite3 等文件）

注意：
- 第一次检索如果 collection 为空，会触发 build_index（可能比较慢）
- 文档变更检测通过文件内容 md5 + kb_hash 实现；未变更时复用索引
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Any
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import pickle
import json
import hashlib
import logging
import shutil
import pypdf
import docx
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

# 配置日志
logger = logging.getLogger(__name__)

def read_pdf(file_path: Path) -> str:
    """读取 PDF 文件内容"""
    try:
        reader = pypdf.PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Error reading PDF {file_path}: {e}")
        return ""

def read_docx(file_path: Path) -> str:
    """读取 Docx 文件内容"""
    try:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        logger.error(f"Error reading Docx {file_path}: {e}")
        return ""

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """提取最后一个有效token的表示"""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(task_description: str, query: str) -> str:
    """格式化查询指令"""
    return f'Instruct: {task_description}\nQuery:{query}'

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """
    简单的文本切片函数
    """
    if not text:
        return []
        
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        
        # 简单的切分逻辑
        chunk = text[start:end]
        chunks.append(chunk)
        
        if end == text_len:
            break
            
        start += chunk_size - overlap
        
    return chunks

class QwenEmbeddingFunction(EmbeddingFunction):
    """
    适配 ChromaDB 的 Qwen Embedding 函数
    """
    def __init__(self, model_name: str = 'Qwen/Qwen3-Embedding-0.6B', device: str = None):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.task_description = 'Given a web search query, retrieve relevant passages that answer the query'

    def _load_model(self):
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, 
                    padding_side='left',
                    trust_remote_code=True
                )
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                ).to(self.device)
                self.model.eval()
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise

    def __call__(self, input: Documents) -> Embeddings:
        """
        ChromaDB 回调：输入一组文档字符串，输出对应 embedding 列表。

        说明：
        - 采用 last_token_pool + L2 normalize，返回 cosine space 适配向量检索。
        - 当前 max_length=8192（对长文档/切片有帮助，但也会增加计算成本）。
        """
        self._load_model()
        
        # ChromaDB 传入的是字符串列表
        batch_dict = self.tokenizer(
            input,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        )
        batch_dict.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = last_token_pool(
                outputs.last_hidden_state, 
                batch_dict['attention_mask']
            )
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
        return embeddings.cpu().numpy().tolist()

class RAGEngine:
    """
    RAG 引擎核心类 (基于 ChromaDB)
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(RAGEngine, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        knowledge_base_path: Union[str, Path] = "knowledge",
        cache_dir: Union[str, Path] = "data/chroma_db", 
        embedding_model_name: str = 'Qwen/Qwen3-Embedding-0.6B',
        device: str = None
    ):
        if hasattr(self, 'initialized') and self.initialized:
            return

        self.knowledge_base_path = Path(knowledge_base_path).resolve()
        self.cache_dir = Path(cache_dir).resolve()
        self.embedding_model_name = embedding_model_name
        self.device = device
        
        # 确保目录存在
        self.knowledge_base_path.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化 ChromaDB（持久化 client）
        logger.info(f"Initializing ChromaDB at {self.cache_dir}")
        self.client = chromadb.PersistentClient(path=str(self.cache_dir))
        
        # 初始化 Embedding Function
        self.embedding_fn = QwenEmbeddingFunction(
            model_name=embedding_model_name,
            device=device
        )
        
        # 获取或创建 Collection
        # - name 固定为 parrot_knowledge
        # - embedding_function 为 QwenEmbeddingFunction
        # - metadata 指定 hnsw:space=cosine
        self.collection = self.client.get_or_create_collection(
            name="parrot_knowledge",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"} # 使用余弦相似度
        )
        
        # 用于检测变更的元数据文件
        self.metadata_file = self.cache_dir / 'kb_metadata.json'
        
        self.task_description = 'Given a web search query, retrieve relevant passages that answer the query'
        self.initialized = True
        logger.info(f"RAGEngine initialized.")

    def load_knowledge_base(self) -> Tuple[List[str], List[str], Dict]:
        """
        加载 knowledge_base_path 下的知识库文档。

        支持扩展名：
        - .md / .txt / .pdf / .docx
        会跳过 README.md。

        返回：
        - documents: 文档全文字符串列表
        - doc_paths: 对应的文件名（用于 metadata/source）
        - metadata: 文件 hash 信息，用于检测变更
        """
        documents = []
        doc_paths = []
        metadata = {
            "files": {}
        }
        
        extensions = ['*.md', '*.txt', '*.pdf', '*.docx']
        files = []
        for ext in extensions:
            files.extend(self.knowledge_base_path.glob(ext))
        files = sorted(list(set(files)))
        
        for file_path in files:
            if file_path.name == "README.md":
                continue
            try:
                content = ""
                if file_path.suffix.lower() == '.pdf':
                    content = read_pdf(file_path)
                elif file_path.suffix.lower() == '.docx':
                    content = read_docx(file_path)
                else:
                    content = file_path.read_text(encoding='utf-8')
                
                if not content.strip():
                    continue
                    
                documents.append(content)
                doc_paths.append(file_path.name)
                
                # 计算哈希
                raw_bytes = file_path.read_bytes()
                file_hash = hashlib.md5(raw_bytes).hexdigest()
                
                metadata["files"][file_path.name] = {
                    "hash": file_hash
                }
            except Exception as e:
                logger.error(f"Failed to read {file_path}: {e}")
        
        return documents, doc_paths, metadata

    def _compute_kb_hash(self, metadata: Dict) -> str:
        """计算整个知识库的哈希"""
        return hashlib.md5(json.dumps(metadata, sort_keys=True).encode()).hexdigest()

    def build_index(self, force_rebuild: bool = False):
        """
        构建索引：
        1. 检查知识库是否有变更
        2. 如果有变更或强制重建，则重新处理
        """
        # 加载当前知识库
        documents, doc_paths, current_metadata = self.load_knowledge_base()
        current_hash = self._compute_kb_hash(current_metadata)
        
        # 检查缓存状态
        cached_hash = ""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    cached_data = json.load(f)
                    cached_hash = cached_data.get('kb_hash', "")
            except:
                pass
        
        # 如果知识库未变更且 collection 非空，直接复用旧索引
        if not force_rebuild and current_hash == cached_hash and self.collection.count() > 0:
            logger.info("Knowledge base unchanged, using existing index.")
            return

        logger.info("Rebuilding index...")
        
        try:
            self.client.delete_collection("parrot_knowledge")
            self.collection = self.client.get_or_create_collection(
                name="parrot_knowledge",
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            logger.warning(f"Error resetting collection: {e}")

        if not documents:
            logger.warning("No documents to index.")
            self._save_metadata(current_hash)
            return

        ids = []
        embeddings_texts = []
        metadatas = []
        
        # 切片并写入向量库：按 chunk_text 简单切片，chunk_id=filename_index
        for doc_content, file_name in zip(documents, doc_paths):
            chunks = chunk_text(doc_content)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{file_name}_{i}"
                ids.append(chunk_id)
                embeddings_texts.append(chunk)
                metadatas.append({
                    "source": file_name,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                })
        
        if ids:
            batch_size = 100
            total = len(ids)
            for i in range(0, total, batch_size):
                end = min(i + batch_size, total)
                self.collection.add(
                    documents=embeddings_texts[i:end],
                    metadatas=metadatas[i:end],
                    ids=ids[i:end]
                )
                logger.info(f"Indexed chunks {i} to {end} / {total}")
        
        self._save_metadata(current_hash)
        logger.info("Index rebuild complete.")

    def _save_metadata(self, kb_hash: str):
        with open(self.metadata_file, 'w') as f:
            json.dump({'kb_hash': kb_hash}, f)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        检索接口：给定 query，返回 top_k 个最相关片段。

        返回结构（列表）：
        [
          {
            "content": "...",         # 命中文本片段
            "score": 0.62,            # 相似度（1-distance）
            "source": "xxx.md",       # 来源文件名
            "chunk_info": {...}       # chunk_index/total_chunks 等元数据
          },
          ...
        ]

        注意：
        - 当 collection 为空时会自动 build_index（第一次检索可能较慢）
        - query 会被包上 task_description 指令（get_detailed_instruct）
        """
        if self.collection.count() == 0:
            self.build_index()
            if self.collection.count() == 0:
                return []

        # 构造带指令的查询
        query_text = get_detailed_instruct(self.task_description, query)
        
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k
        )
        
        parsed_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                distance = results['distances'][0][i]
                similarity = 1 - distance
                
                metadata = results['metadatas'][0][i]
                content = results['documents'][0][i]
                
                parsed_results.append({
                    "content": content,
                    "score": similarity,
                    "source": metadata.get("source", "unknown"),
                    "chunk_info": metadata
                })
        
        return parsed_results

    def add_document(self, filename: str, content: str) -> bool:
        try:
            file_path = self.knowledge_base_path / filename
            file_path.write_text(content, encoding='utf-8')
            self.build_index(force_rebuild=True)
            return True
        except Exception:
            return False

    def delete_document(self, filename: str) -> bool:
        try:
            file_path = self.knowledge_base_path / filename
            if file_path.exists():
                file_path.unlink()
                self.build_index(force_rebuild=True)
                return True
            return False
        except Exception:
            return False

    def list_documents(self) -> List[Dict]:
        """列出所有文档"""
        docs = []
        if self.knowledge_base_path.exists():
            for f in self.knowledge_base_path.iterdir():
                if f.is_file() and f.suffix.lower() in ['.md', '.txt', '.pdf', '.docx'] and f.name != 'README.md':
                    stat = f.stat()
                    docs.append({
                        "name": f.name,
                        "size": stat.st_size,
                        "mtime": stat.st_mtime
                    })
        return docs

def get_rag_engine() -> RAGEngine:
    return RAGEngine()

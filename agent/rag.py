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

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Any, TYPE_CHECKING, Callable
import pickle
import json
import hashlib
import logging
import time
import shutil
import pypdf
import docx
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

if TYPE_CHECKING:  # pragma: no cover
    from torch import Tensor  # type: ignore[import-not-found]

# 配置日志
logger = logging.getLogger(__name__)

def _lazy_import_torch():
    """
    延迟导入 torch。
    目的：避免 Windows 上 torch DLL 失效（WinError 1114）导致服务在 import 阶段直接崩溃。
    """
    try:
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore
        return torch, F
    except Exception as e:
        raise RuntimeError(
            "torch 加载失败（常见为 WinError 1114 / DLL 初始化失败）。\n"
            "这会影响向量化/Embedding/RAG 索引构建，但服务应当仍可启动。\n"
            "建议排查：\n"
            "1) 安装/修复 Microsoft Visual C++ 2015-2022 Redistributable（x64）\n"
            "2) 重新安装与当前环境匹配的 torch（CPU 版或正确 CUDA 版）\n"
            "3) 确认 conda/pip 环境中没有残留冲突的 torch\n"
            f"原始错误：{repr(e)}"
        ) from e

def _lazy_import_transformers():
    """延迟导入 transformers（Embedding 需要）。"""
    try:
        from transformers import AutoTokenizer, AutoModel  # type: ignore
        return AutoTokenizer, AutoModel
    except Exception as e:
        raise RuntimeError(
            "transformers 加载失败（用于 embedding）。请检查依赖安装是否完整。\n"
            f"原始错误：{repr(e)}"
        ) from e


def _lazy_import_sentence_transformers():
    """延迟导入 sentence-transformers（Rerank 需要）。"""
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
        return CrossEncoder
    except Exception as e:
        raise RuntimeError(
            "sentence-transformers 加载失败（用于 rerank）。请先安装依赖：pip install sentence-transformers\n"
            f"原始错误：{repr(e)}"
        ) from e

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
        torch, _ = _lazy_import_torch()
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(task_description: str, query: str) -> str:
    """格式化查询指令"""
    return f'Instruct: {task_description}\nQuery:{query}'

def _normalize_text(text: str) -> str:
    """
    轻量归一化：统一换行、去掉过多空白。
    不做激进清洗，避免破坏 markdown/标题结构。
    """
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    # collapse trailing spaces
    t = "\n".join([ln.rstrip() for ln in t.split("\n")])
    # collapse multiple blank lines
    while "\n\n\n" in t:
        t = t.replace("\n\n\n", "\n\n")
    return t.strip()


def _semantic_units(text: str) -> List[str]:
    """
    语义切分的“最小单元”：
    - 以 markdown 标题行（# / ## / ### ...）为强边界
    - 其次按空行分段落

    返回一个 units 列表（每个 unit 是一段可读文本）。
    """
    t = _normalize_text(text)
    if not t:
        return []

    lines = t.split("\n")
    units: List[str] = []
    buf: List[str] = []

    def flush():
        nonlocal buf
        s = "\n".join(buf).strip()
        if s:
            units.append(s)
        buf = []

    for ln in lines:
        # markdown 标题：独立作为边界（先 flush 再写入）
        if ln.lstrip().startswith("#"):
            flush()
            buf.append(ln)
            flush()
            continue

        if ln.strip() == "":
            # 段落边界
            flush()
            continue

        buf.append(ln)

    flush()
    return units


def _fixed_chunks(text: str, chunk_size_chars: int, overlap_chars: int) -> List[str]:
    """
    固定长度切片（字符窗口滑动）。
    - chunk_size_chars: 每块最大字符数
    - overlap_chars: 相邻块重叠字符数
    """
    t = _normalize_text(text)
    if not t:
        return []
        
    if chunk_size_chars <= 0:
        return [t]
    overlap_chars = max(0, min(overlap_chars, chunk_size_chars - 1)) if chunk_size_chars > 1 else 0

    chunks: List[str] = []
    start = 0
    n = len(t)
    step = max(1, chunk_size_chars - overlap_chars)
    while start < n:
        end = min(start + chunk_size_chars, n)
        chunks.append(t[start:end])
        if end == n:
            break
        start += step
    return chunks


def _semantic_chunks(text: str, chunk_size_chars: int, overlap_chars: int) -> List[str]:
    """
    语义切片：先按标题/段落拆成 units，再将多个 units 合并到 chunk_size_chars 附近。
    overlap 用“前一块末尾 overlap_chars 字符”作为下一块前缀（跨块保留上下文）。
    """
    units = _semantic_units(text)
    if not units:
        return []

    chunk_size_chars = max(1, chunk_size_chars)
    overlap_chars = max(0, overlap_chars)

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    def flush_with_overlap():
        nonlocal current, current_len
        if not current:
            return ""
        chunk = "\n\n".join(current).strip()
        if chunk:
            chunks.append(chunk)
        tail = chunk[-overlap_chars:] if (overlap_chars > 0 and chunk) else ""
        current = []
        current_len = 0
        return tail

    overlap_prefix = ""
    for unit in units:
        unit_len = len(unit)

        # unit 本身过大：先把已有 current flush，再对 unit 做 fixed chunk
        if unit_len > chunk_size_chars:
            if current:
                overlap_prefix = flush_with_overlap()
            big_parts = _fixed_chunks(unit, chunk_size_chars, overlap_chars)
            # big_parts 自带 overlap，不再额外前缀
            chunks.extend(big_parts)
            overlap_prefix = big_parts[-1][-overlap_chars:] if overlap_chars and big_parts else ""
            continue

        # 尝试把 unit 放入 current
        add_sep = 2 if current else 0  # "\n\n"
        if current_len + add_sep + unit_len <= chunk_size_chars:
            current.append(unit)
            current_len += add_sep + unit_len
            continue

        # current 满了：flush，并把 overlap_prefix 加到下一块的开头（作为一个小 unit）
        overlap_prefix = flush_with_overlap()
        if overlap_prefix:
            current.append(overlap_prefix)
            current_len = len(overlap_prefix)
        current.append(unit)
        current_len += (2 if current_len > 0 else 0) + unit_len

    flush_with_overlap()
    return chunks


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 100,
    strategy: str = "fixed",
) -> List[str]:
    """
    文本切片入口（可选策略）。

    参数：
    - chunk_size: 每块目标长度（字符数）
    - overlap: 相邻块重叠长度（字符数）
    - strategy:
      - \"fixed\": 固定长度滑窗（适合任意纯文本）
      - \"semantic\": 先按标题/段落切 unit，再合并到目标长度（更适合 markdown/结构化文档）

    返回：
    - chunks: 字符串列表（每个元素为一个可检索的片段）
    """
    if not text:
        return []
    s = (strategy or "fixed").strip().lower()
    if s == "semantic":
        return _semantic_chunks(text, chunk_size_chars=chunk_size, overlap_chars=overlap)
    return _fixed_chunks(text, chunk_size_chars=chunk_size, overlap_chars=overlap)

class QwenEmbeddingFunction(EmbeddingFunction):
    """
    适配 ChromaDB 的 Qwen Embedding 函数
    """
    def __init__(self, model_name: str = 'Qwen/Qwen3-Embedding-0.6B', device: str = None):
        self.model_name = model_name
        # 不在 __init__ 里导入 torch（避免 Windows torch DLL 失效导致服务启动崩溃）
        # device=None 表示稍后在 _load_model 时根据 torch.cuda.is_available() 决定
        self.device = device
        self.tokenizer = None
        self.model = None
        self.task_description = 'Given a web search query, retrieve relevant passages that answer the query'

    def _load_model(self):
        if self.model is None:
            torch, _ = _lazy_import_torch()
            AutoTokenizer, AutoModel = _lazy_import_transformers()

            if not self.device:
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        try:
            torch, F = _lazy_import_torch()
        except Exception as e:
            error_msg = f"导入 torch 失败：{str(e)}。请检查 torch 环境是否正确安装。"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
        
        # 轻量埋点：每次 embedding 打印 chunk 数、token 总量、耗时
        t0 = time.perf_counter()
        chunk_count = len(input) if input else 0

        try:
            self._load_model()
        except Exception as e:
            error_msg = f"加载 embedding 模型失败：{str(e)}。请检查 torch/transformers 环境是否正确安装。"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
        
        try:
            # ChromaDB 传入的是字符串列表
            batch_dict = self.tokenizer(
                input,
                padding=True,
                truncation=True,
                max_length=8192,
                return_tensors="pt",
            )

            try:
                # attention_mask: 1 表示有效 token，sum 即总 token 数（本地 tokenizer 口径）
                total_tokens = int(batch_dict["attention_mask"].sum().item())
            except Exception:
                total_tokens = -1

            batch_dict.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**batch_dict)
                embeddings = last_token_pool(
                    outputs.last_hidden_state, 
                    batch_dict['attention_mask']
                )
                embeddings = F.normalize(embeddings, p=2, dim=1)
                
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            msg = f"[embedding] chunks={chunk_count} total_tokens={total_tokens} time_ms={elapsed_ms:.2f}"
            # 控制台输出（stdout）：方便直接观察每次 embedding 的成本/耗时
            print(msg, flush=True)
            logger.info(msg)
            return embeddings.cpu().numpy().tolist()
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            msg = f"[embedding] chunks={chunk_count} total_tokens=? time_ms={elapsed_ms:.2f} (failed)"
            print(msg, flush=True)
            logger.info(msg)
            error_msg = f"向量化计算失败：{str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e


class CrossEncoderReranker:
    """
    基于 sentence-transformers CrossEncoder 的 reranker。

    说明：
    - 输出为模型 raw score（不保证在 0~1），仅用于相对排序。
    - 采用 lazy load，避免启动时强依赖 sentence-transformers/torch。
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = "cpu"):
        self.model_name = model_name
        self.device = device or "cpu"
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        CrossEncoder = _lazy_import_sentence_transformers()
        logger.info(f"Loading reranker model: {self.model_name} on {self.device}")
        # CrossEncoder 会自行处理 tokenizer/model 加载
        self._model = CrossEncoder(self.model_name, device=self.device)

    def score(self, query: str, docs: List[str], batch_size: int = 32) -> List[float]:
        """
        对 (query, doc) pairs 进行打分，返回与 docs 等长的分数列表。
        """
        if not docs:
            return []
        self._load()
        assert self._model is not None
        pairs = [(query, d) for d in docs]
        scores = self._model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        # numpy array / list -> list[float]
        return [float(x) for x in scores]

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
        device: str = None,
        enable_rerank: bool = True,
        rerank_model_name: str = "BAAI/bge-reranker-base",
        rerank_device: str = "cpu",
        rerank_coarse_k: int = 10,
    ):
        if hasattr(self, 'initialized') and self.initialized:
            return

        self.knowledge_base_path = Path(knowledge_base_path).resolve()
        self.cache_dir = Path(cache_dir).resolve()
        self.embedding_model_name = embedding_model_name
        self.device = device
        self.enable_rerank = bool(enable_rerank)
        self.rerank_model_name = rerank_model_name
        self.rerank_device = rerank_device
        self.rerank_coarse_k = int(rerank_coarse_k) if rerank_coarse_k else 10
        self._reranker: Optional[CrossEncoderReranker] = None
        
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

    def _get_reranker(self) -> Optional[CrossEncoderReranker]:
        if not self.enable_rerank:
            return None
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(
                model_name=self.rerank_model_name,
                device=self.rerank_device,
            )
        return self._reranker

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

    def _load_cached_metadata(self) -> Dict[str, Any]:
        """
        读取本地索引元数据（用于增量同步）。

        兼容历史格式：
        - 旧格式：{"kb_hash": "..."}
        - 新格式：{"kb_hash": "...", "files": {"xxx.md": {"hash": "..."}, ...}}
        """
        if not self.metadata_file.exists():
            return {"kb_hash": "", "files": {}}
        try:
            with open(self.metadata_file, "r") as f:
                data = json.load(f) or {}
        except Exception:
            return {"kb_hash": "", "files": {}}

        # normalize
        if "files" not in data or not isinstance(data.get("files"), dict):
            data["files"] = {}
        if "kb_hash" not in data:
            data["kb_hash"] = ""
        return data

    def _save_metadata(self, kb_hash: str, files_metadata: Optional[Dict[str, Any]] = None):
        """
        保存索引元数据（用于变更检测/增量同步）。
        """
        payload = {"kb_hash": kb_hash, "files": files_metadata or {}}
        with open(self.metadata_file, "w") as f:
            json.dump(payload, f, ensure_ascii=False)

    def build_index(self, force_rebuild: bool = False, on_progress: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        构建索引：
        1. 检查知识库是否有变更
        2. 如果有变更或强制重建，则重新处理
        """
        # 加载当前知识库
        documents, doc_paths, current_metadata = self.load_knowledge_base()
        current_hash = self._compute_kb_hash(current_metadata)
        
        # 检查缓存状态
        cached_data = self._load_cached_metadata()
        cached_hash = cached_data.get("kb_hash", "")
        
        # 如果知识库未变更且 collection 非空，直接复用旧索引
        if not force_rebuild and current_hash == cached_hash and self.collection.count() > 0:
            logger.info("Knowledge base unchanged, using existing index.")
            return

        logger.info("Rebuilding index...")
        if on_progress:
            on_progress({"event": "status", "data": {"text": "开始全量重建索引…"}})
        
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
            self._save_metadata(current_hash, current_metadata.get("files"))
            return

        ids = []
        embeddings_texts = []
        metadatas = []
        
        # 切片并写入向量库：支持 fixed/semantic 两种策略（chunk_id=filename_index）
        # 当前默认策略为 semantic（更适合 markdown/段落结构的知识库）
        total_files = len(doc_paths)
        for file_idx, (doc_content, file_name) in enumerate(zip(documents, doc_paths), start=1):
            chunks = chunk_text(doc_content, chunk_size=500, overlap=80, strategy="semantic")
            if on_progress:
                on_progress({"event": "progress", "data": {"phase": "build_index", "file": file_name, "file_index": file_idx, "total_files": total_files, "chunk_total": len(chunks), "chunk_done": 0}})
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
                try:
                    self.collection.add(
                        documents=embeddings_texts[i:end],
                        metadatas=metadatas[i:end],
                        ids=ids[i:end]
                    )
                    logger.info(f"Indexed chunks {i} to {end} / {total}")
                    if on_progress:
                        on_progress({"event": "progress", "data": {"phase": "build_index", "chunk_done_total": end, "chunk_total_total": total}})
                except Exception as e:
                    error_msg = f"向量化失败（chunks {i}-{end}）：{str(e)}"
                    logger.error(error_msg, exc_info=True)
                    if on_progress:
                        on_progress({"event": "error", "data": {"text": error_msg, "phase": "build_index", "chunk_range": f"{i}-{end}"}})
                    raise  # 重新抛出异常，让上层知道失败
        
        self._save_metadata(current_hash, current_metadata.get("files"))
        logger.info("Index rebuild complete.")
        if on_progress:
            on_progress({"event": "status", "data": {"text": "全量重建完成。"}})

    def _delete_by_source(self, source_filename: str) -> None:
        """
        从向量库中删除某个文件对应的全部 chunks。
        """
        try:
            # Preferred: delete by metadata filter
            self.collection.delete(where={"source": source_filename})
            return
        except Exception:
            pass
        try:
            res = self.collection.get(where={"source": source_filename})
            ids = res.get("ids") if isinstance(res, dict) else None
            if ids:
                self.collection.delete(ids=ids)
        except Exception:
            pass

    def _index_one_file(self, filename: str, content: str, on_progress: Optional[Callable[[Dict[str, Any]], None]] = None) -> int:
        """
        将单个文件内容切片并写入向量库，返回写入 chunk 数。
        """
        chunks = chunk_text(content, chunk_size=500, overlap=80, strategy="semantic")
        if not chunks:
            return 0
        total_chunks = len(chunks)
        if on_progress:
            on_progress({"event": "progress", "data": {"phase": "index_file", "file": filename, "chunk_total": total_chunks, "chunk_done": 0}})

        # 分批写入：便于提供进度提示
        batch_size = 64
        done = 0
        for start in range(0, total_chunks, batch_size):
            end = min(start + batch_size, total_chunks)
            ids = [f"{filename}_{i}" for i in range(start, end)]
            metadatas = [{"source": filename, "chunk_index": i, "total_chunks": total_chunks} for i in range(start, end)]
            try:
                self.collection.add(documents=chunks[start:end], metadatas=metadatas, ids=ids)
                done = end
                if on_progress:
                    on_progress({"event": "progress", "data": {"phase": "index_file", "file": filename, "chunk_total": total_chunks, "chunk_done": done}})
            except Exception as e:
                error_msg = f"向量化文件 {filename} 失败（chunks {start}-{end}）：{str(e)}"
                logger.error(error_msg, exc_info=True)
                if on_progress:
                    on_progress({"event": "error", "data": {"text": error_msg, "phase": "index_file", "file": filename, "chunk_range": f"{start}-{end}"}})
                raise  # 重新抛出异常，让上层知道失败

        return total_chunks

    def sync_index(self, force_full_rebuild: bool = False, on_progress: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        """
        增量同步索引（推荐用于“批量增删后点击一次向量化”）。

        行为：
        - force_full_rebuild=True：等价于 build_index(force_rebuild=True)
        - 否则：
          - 新增文件：只向量化新增文件
          - 修改文件：删除旧 chunks 后重新向量化该文件
          - 删除文件：删除该文件对应的 chunks

        返回：统计信息（added/updated/removed/unchanged 等）。
        """
        try:
            if force_full_rebuild:
                self.build_index(force_rebuild=True, on_progress=on_progress)
                return {"mode": "full_rebuild"}
        except Exception as e:
            error_msg = f"全量重建索引失败：{str(e)}"
            logger.error(error_msg, exc_info=True)
            if on_progress:
                on_progress({"event": "error", "data": {"text": error_msg}})
            raise

        # 当前知识库状态
        documents, doc_paths, current_metadata = self.load_knowledge_base()
        current_files = current_metadata.get("files", {}) if isinstance(current_metadata, dict) else {}
        current_hash = self._compute_kb_hash(current_metadata)

        cached = self._load_cached_metadata()
        cached_files = cached.get("files", {}) if isinstance(cached.get("files"), dict) else {}

        current_set = set(current_files.keys())
        cached_set = set(cached_files.keys())
        added = sorted(list(current_set - cached_set))
        removed = sorted(list(cached_set - current_set))
        modified = sorted([fn for fn in (current_set & cached_set) if current_files.get(fn, {}).get("hash") != cached_files.get(fn, {}).get("hash")])
        unchanged = sorted(list((current_set & cached_set) - set(modified)))

        total_to_index = len(added) + len(modified)
        if on_progress:
            on_progress({"event": "status", "data": {"text": f"开始增量同步：新增{len(added)}、更新{len(modified)}、移除{len(removed)}（待向量化文件 {total_to_index} 个）"}})

        # Ensure collection is ready; if empty and we have files, do a lightweight add-all
        try:
            count = self.collection.count()
        except Exception:
            count = 0

        # Map filename -> content for current documents
        file_to_content = {fn: doc for fn, doc in zip(doc_paths, documents)}

        # Delete removed + modified
        for fn in removed:
            if on_progress:
                on_progress({"event": "status", "data": {"text": f"删除旧向量：{fn}"}})
            self._delete_by_source(fn)
        for fn in modified:
            if on_progress:
                on_progress({"event": "status", "data": {"text": f"删除旧向量：{fn}（将重新向量化）"}})
            self._delete_by_source(fn)

        # Add new + modified
        indexed_added = {}
        indexed_modified = {}
        file_no = 0
        for fn in added:
            file_no += 1
            if on_progress:
                on_progress({"event": "progress", "data": {"phase": "file", "action": "add", "file": fn, "file_index": file_no, "total_files": total_to_index}})
            try:
                content = file_to_content.get(fn, "")
                n = self._index_one_file(fn, content, on_progress=on_progress)
                indexed_added[fn] = n
            except Exception as e:
                error_msg = f"向量化新增文件 {fn} 失败：{str(e)}"
                logger.error(error_msg, exc_info=True)
                if on_progress:
                    on_progress({"event": "error", "data": {"text": error_msg, "file": fn}})
                raise  # 重新抛出异常，让上层知道失败
        for fn in modified:
            file_no += 1
            if on_progress:
                on_progress({"event": "progress", "data": {"phase": "file", "action": "update", "file": fn, "file_index": file_no, "total_files": total_to_index}})
            try:
                content = file_to_content.get(fn, "")
                n = self._index_one_file(fn, content, on_progress=on_progress)
                indexed_modified[fn] = n
            except Exception as e:
                error_msg = f"向量化更新文件 {fn} 失败：{str(e)}"
                logger.error(error_msg, exc_info=True)
                if on_progress:
                    on_progress({"event": "error", "data": {"text": error_msg, "file": fn}})
                raise  # 重新抛出异常，让上层知道失败

        # If collection was empty and we had no cached metadata, but current has files, we should ensure all indexed
        # (e.g., first time using sync_index but metadata existed without files list)
        if count == 0 and (not cached_files) and current_set and not (added or modified):
            # Index all current files once
            indexed_all = {}
            for fn in doc_paths:
                if on_progress:
                    # 这里不精确区分新增/更新，只用于展示“正在补齐首次索引”
                    on_progress({"event": "progress", "data": {"phase": "file", "action": "add", "file": fn}})
                indexed_all[fn] = self._index_one_file(fn, file_to_content.get(fn, ""), on_progress=on_progress)
            indexed_added.update(indexed_all)
            added = sorted(list(set(added) | set(indexed_all.keys())))

        # Save metadata (now in new format)
        self._save_metadata(current_hash, current_files)

        if on_progress:
            on_progress({"event": "status", "data": {"text": "索引同步完成。"}})

        return {
            "mode": "incremental",
            "kb_hash": current_hash,
            "added": added,
            "modified": modified,
            "removed": removed,
            "unchanged_count": len(unchanged),
            "indexed_added_chunks": indexed_added,
            "indexed_modified_chunks": indexed_modified,
        }

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
        
        # Two-stage: coarse vector retrieval then optional rerank
        coarse_k = max(int(top_k or 3), int(self.rerank_coarse_k or 10))
        results = self.collection.query(
            query_texts=[query_text],
            n_results=coarse_k
        )
        
        parsed_results: List[Dict[str, Any]] = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                distance = results['distances'][0][i]
                similarity = 1 - distance
                
                metadata = results['metadatas'][0][i]
                content = results['documents'][0][i]
                
                parsed_results.append({
                    "content": content,
                    # 向量粗排分数（cosine space: 1-distance）
                    "vector_score": similarity,
                    "source": metadata.get("source", "unknown"),
                    "chunk_info": metadata
                })

        if not parsed_results:
            return []

        # 精排：用 CrossEncoder 对 coarse hits 重打分并截断到 top_k
        reranker = self._get_reranker()
        if reranker is None:
            # 兼容旧语义：没有 rerank 时，把 score 设为 vector_score
            for h in parsed_results:
                h["score"] = float(h.get("vector_score", 0.0))
            return parsed_results[: int(top_k or 3)]

        try:
            t_r = time.perf_counter()
            docs = [str(h.get("content") or "") for h in parsed_results]
            rerank_scores = reranker.score(query=query, docs=docs)
            elapsed_ms = (time.perf_counter() - t_r) * 1000.0

            for h, s in zip(parsed_results, rerank_scores):
                h["rerank_score"] = float(s)
                # 对外统一用 score 作为“最终排序分数”
                h["score"] = float(s)

            parsed_results.sort(key=lambda x: float(x.get("score", float("-inf"))), reverse=True)
            logger.info(f"[rerank] pairs={len(docs)} time_ms={elapsed_ms:.2f} model={self.rerank_model_name} device={self.rerank_device}")
            return parsed_results[: int(top_k or 3)]
        except Exception as e:
            # 失败降级：回退向量分数排序
            logger.warning(f"[rerank] failed, fallback to vector score: {e}")
            for h in parsed_results:
                h["score"] = float(h.get("vector_score", 0.0))
            parsed_results.sort(key=lambda x: float(x.get("vector_score", float("-inf"))), reverse=True)
            return parsed_results[: int(top_k or 3)]

    def add_document(self, filename: str, content: str) -> bool:
        try:
            file_path = self.knowledge_base_path / filename
            file_path.write_text(content, encoding='utf-8')
            return True
        except Exception:
            return False

    def delete_document(self, filename: str) -> bool:
        try:
            file_path = self.knowledge_base_path / filename
            if file_path.exists():
                file_path.unlink()
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

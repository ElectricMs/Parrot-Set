import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
)
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
KB_DIR = Path("knowledge_base")
VECTOR_DB_DIR = Path("vector_db")
UPLOAD_DIR = Path("uploads")

# Ensure directories exist
KB_DIR.mkdir(exist_ok=True)
VECTOR_DB_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

class LocalKnowledgeBase:
    """
    Local Knowledge Base System
    
    Handles file processing, embedding generation, and vector storage using ChromaDB.
    Supports PDF, TXT, MD, DOCX, CSV formats.
    """
    
    def __init__(self, collection_name: str = "parrot_kb"):
        """
        Initialize the knowledge base.
        
        Args:
            collection_name: Name of the ChromaDB collection
        """
        self._lock = threading.Lock()  # 添加线程锁，防止ChromaDB并发写入冲突
        
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2", # Lightweight and fast model
            model_kwargs={'device': 'cpu'},
            encode_kwargs={
                'batch_size': 32,  # 批量处理嵌入，加快速度
                'normalize_embeddings': False  # 移除不支持的参数
            }
        )
        
        self.vector_db = Chroma(
            persist_directory=str(VECTOR_DB_DIR),
            embedding_function=self.embedding_model,
            collection_name=collection_name
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # 增大chunk大小，减少chunk数量，加快处理速度
            chunk_overlap=200,
            length_function=len,
        )

    def _get_loader(self, file_path: Path):
        """
        Factory method to get appropriate loader based on file extension.
        
        支持的文件格式：
        - .txt: 纯文本文件
        - .pdf: PDF文档
        - .docx: Word文档
        - .md: Markdown文件
        - .csv: CSV表格
        
        注意：Excel文件 (.xlsx, .xls) 在 add_document 中直接处理
        """
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
            
        ext = file_path.suffix.lower()
        if ext == ".txt":
            return TextLoader(str(file_path), encoding="utf-8")
        elif ext == ".pdf":
            try:
                import pypdf
            except ImportError:
                raise ImportError("处理PDF需要安装pypdf库: pip install pypdf")
            return PyPDFLoader(str(file_path))
        elif ext == ".docx":
            try:
                import docx2txt
            except ImportError:
                raise ImportError("处理Word文档需要安装docx2txt库: pip install docx2txt")
            return Docx2txtLoader(str(file_path))
        elif ext == ".md":
            try:
                import unstructured
            except ImportError:
                raise ImportError("处理Markdown需要安装unstructured库: pip install unstructured")
            return UnstructuredMarkdownLoader(str(file_path))
        elif ext == ".csv":
            return CSVLoader(str(file_path), encoding="utf-8")
        else:
            raise ValueError(f"不支持的文件格式: {ext}。支持格式: .txt, .pdf, .docx, .md, .csv, .xlsx, .xls")
    
    def _process_sheet(self, file_path: Path, sheet_name: str) -> Optional[Document]:
        """
        处理单个Excel工作表（用于并行处理）
        
        Args:
            file_path: Excel文件路径
            sheet_name: 工作表名称
            
        Returns:
            Document对象或None
        """
        try:
            # 使用read_excel读取
            df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=None)
            
            if df.empty:
                logger.warning(f"工作表 {sheet_name} 为空，跳过")
                return None
            
            # 优化：使用to_csv而不是逐行遍历，更快
            # 对于大文件，限制显示的行数（减少以加快处理速度）
            max_rows_display = 5000  # 最多显示5000行（减少以加快处理）
            
            text_content = f"工作表: {sheet_name}\n"
            text_content += f"总行数: {len(df)}\n"
            text_content += f"总列数: {len(df.columns)}\n\n"
            
            # 列名信息
            text_content += "列名: " + " | ".join(df.columns.astype(str)) + "\n\n"
            
            # 如果行数太多，只处理前N行和统计信息
            if len(df) > max_rows_display:
                logger.info(f"工作表 {sheet_name} 较大（{len(df)}行），仅处理前{max_rows_display}行")
                text_content += f"注意：由于文件较大，仅显示前{max_rows_display}行数据\n\n"
                df_sample = df.head(max_rows_display)
                text_content += df_sample.to_string(index=False, max_rows=max_rows_display)
                
                # 添加统计信息
                text_content += "\n\n数据统计信息:\n"
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    text_content += df[numeric_cols].describe().to_string()
            else:
                # 小文件，直接转换
                text_content += df.to_string(index=False)
            
            # 添加数据类型信息
            text_content += "\n\n列数据类型:\n"
            for col in df.columns:
                text_content += f"{col}: {df[col].dtype}\n"
            
            doc = Document(
                page_content=text_content,
                metadata={
                    "source": file_path.name,
                    "sheet": sheet_name,
                    "total_rows": len(df),
                    "total_cols": len(df.columns)
                }
            )
            logger.info(f"工作表 {sheet_name} 处理完成，生成 {len(text_content)} 字符")
            return doc
        except Exception as e:
            logger.error(f"处理工作表 {sheet_name} 失败: {e}", exc_info=True)
            return None
    
    def _load_excel(self, file_path: Path) -> List[Document]:
        """
        加载Excel文件并转换为Document列表
        
        多线程优化版本：并行处理多个工作表
        
        Args:
            file_path: Excel文件路径
            
        Returns:
            Document列表
        """
        try:
            logger.info(f"开始加载Excel文件: {file_path}")
            # 读取Excel文件的所有工作表
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            
            if len(sheet_names) == 0:
                logger.warning("Excel文件没有工作表")
                return []
            
            logger.info(f"发现 {len(sheet_names)} 个工作表，开始并行处理...")
            
            documents = []
            
            # 如果只有一个工作表，直接处理（避免线程开销）
            if len(sheet_names) == 1:
                doc = self._process_sheet(file_path, sheet_names[0])
                if doc:
                    documents.append(doc)
            else:
                # 多个工作表，使用线程池并行处理
                max_workers = min(len(sheet_names), 4)  # 最多4个线程，避免过多线程
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # 提交所有任务
                    future_to_sheet = {
                        executor.submit(self._process_sheet, file_path, sheet_name): sheet_name
                        for sheet_name in sheet_names
                    }
                    
                    # 收集结果
                    for future in as_completed(future_to_sheet):
                        sheet_name = future_to_sheet[future]
                        try:
                            doc = future.result()
                            if doc:
                                documents.append(doc)
                        except Exception as e:
                            logger.error(f"处理工作表 {sheet_name} 时出错: {e}", exc_info=True)
            
            logger.info(f"Excel文件处理完成，共 {len(documents)} 个文档")
            return documents
        except Exception as e:
            logger.error(f"加载Excel文件失败: {e}", exc_info=True)
            raise ValueError(f"无法读取Excel文件: {str(e)}")

    def add_document(self, file_path: Path, move_to_kb: bool = True) -> Dict[str, Any]:
        """
        Process a document and add it to the vector database.
        
        Args:
            file_path: Path to the document file
            move_to_kb: Whether to move file to knowledge_base directory after processing
            
        Returns:
            Dict containing processing results:
                - chunks_added: Number of chunks added
                - total_chars: Total characters processed
                - file_size: File size in bytes
                - status: Processing status
        """
        try:
            logger.info(f"Processing document: {file_path}")
            
            # 获取文件大小
            file_size = file_path.stat().st_size if file_path.exists() else 0
            
            # 1. Load document
            ext = file_path.suffix.lower()
            logger.info(f"文件扩展名: {ext}, 文件路径: {file_path}")
            
            try:
            if ext in [".xlsx", ".xls"]:
                # Excel文件需要特殊处理
                    logger.info("使用Excel加载器")
                documents = self._load_excel(file_path)
            else:
                    logger.info(f"使用标准加载器: {ext}")
                loader = self._get_loader(file_path)
                documents = loader.load()
            
                logger.info(f"文档加载完成，共 {len(documents) if documents else 0} 个文档")
            except Exception as load_error:
                logger.error(f"加载文档失败: {load_error}", exc_info=True)
                raise ValueError(f"无法加载文档: {str(load_error)}")
            
            if not documents or len(documents) == 0:
                logger.warning(f"No content extracted from {file_path}")
                return {
                    "chunks_added": 0,
                    "total_chars": 0,
                    "file_size": file_size,
                    "status": "warning",
                    "message": "未提取到内容"
                }
            
            # 检查文档内容
            empty_docs = [i for i, doc in enumerate(documents) if not doc.page_content or len(doc.page_content.strip()) == 0]
            if empty_docs:
                logger.warning(f"发现 {len(empty_docs)} 个空文档，索引: {empty_docs}")
                documents = [doc for doc in documents if doc.page_content and len(doc.page_content.strip()) > 0]
            
            if not documents or len(documents) == 0:
                logger.warning(f"所有文档都为空")
                return {
                    "chunks_added": 0,
                    "total_chars": 0,
                    "file_size": file_size,
                    "status": "warning",
                    "message": "所有文档内容都为空"
                }
            
            # 计算总字符数
            total_chars = sum(len(doc.page_content) for doc in documents)
            logger.info(f"文档总字符数: {total_chars}")
                
            # 2. Split text
            logger.info("开始分块处理...")
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"分块完成，共 {len(chunks)} 个片段")
            
            # 3. Add to vector store
            # Persist original filename and additional metadata
            logger.info("准备添加到向量数据库...")
            logger.info(f"准备处理 {len(chunks)} 个chunks")
            
            # 确保所有chunks都有正确的metadata
            for i, chunk in enumerate(chunks):
                if not chunk.metadata:
                    chunk.metadata = {}
                chunk.metadata["source"] = file_path.name
                chunk.metadata["chunk_index"] = i
                chunk.metadata["file_size"] = file_size
                # 确保page_content不为空
                if not chunk.page_content or len(chunk.page_content.strip()) == 0:
                    logger.warning(f"Chunk {i} 内容为空，跳过")
                    continue
            
            # 过滤掉空内容的chunks
            valid_chunks = [chunk for chunk in chunks if chunk.page_content and len(chunk.page_content.strip()) > 0]
            if len(valid_chunks) == 0:
                logger.warning("所有chunks都为空，无法添加到向量数据库")
                return {
                    "chunks_added": 0,
                    "total_chars": total_chars,
                    "file_size": file_size,
                    "status": "warning",
                    "message": "所有文本片段都为空"
                }
            
            logger.info(f"有效chunks数量: {len(valid_chunks)}/{len(chunks)}")
            
            # 批量添加，避免逐个添加导致的性能问题
            # ChromaDB的add_documents已经支持批量添加，但我们可以分批处理大文件
            # 增大批次大小以减少数据库写入次数
            batch_size = 200  # 每批处理200个chunks（增大批次以加快速度）
            if len(valid_chunks) > batch_size:
                logger.info(f"文件较大，分批添加（每批{batch_size}个）...")
                for i in range(0, len(valid_chunks), batch_size):
                    batch = valid_chunks[i:i + batch_size]
                    try:
                        # ChromaDB会自动持久化（因为设置了persist_directory）
                        # 使用锁保护写入操作
                        with self._lock:
                            ids = self.vector_db.add_documents(batch)
                        
                        progress = min(i + batch_size, len(valid_chunks))
                        logger.info(f"已添加 {progress}/{len(valid_chunks)} 个片段 ({progress*100//len(valid_chunks)}%), IDs数量: {len(ids) if ids else 0}")
                    except Exception as batch_error:
                        logger.error(f"批量添加失败（批次 {i//batch_size + 1}）: {batch_error}", exc_info=True)
                        raise
            else:
                try:
                    # ChromaDB会自动持久化（因为设置了persist_directory）
                    # 使用锁保护写入操作
                    with self._lock:
                        ids = self.vector_db.add_documents(valid_chunks)
                    logger.info(f"添加文档成功，返回IDs数量: {len(ids) if ids else 0}")
                except Exception as add_error:
                    logger.error(f"添加文档到向量数据库失败: {add_error}", exc_info=True)
                    raise
            
            logger.info(f"成功添加 {len(valid_chunks)} 个片段到向量数据库")
            
            # 4. Move file to knowledge_base directory if requested
            if move_to_kb and file_path.parent != KB_DIR:
                target_path = KB_DIR / file_path.name
                # 如果目标文件已存在，添加序号
                counter = 1
                while target_path.exists():
                    stem = file_path.stem
                    suffix = file_path.suffix
                    target_path = KB_DIR / f"{stem}_{counter}{suffix}"
                    counter += 1
                shutil.move(str(file_path), str(target_path))
                logger.info(f"Moved file to: {target_path}")
            
            return {
                "chunks_added": len(chunks),
                "total_chars": total_chars,
                "file_size": file_size,
                "status": "success",
                "message": f"成功处理文档，生成 {len(chunks)} 个片段"
            }
            
        except Exception as e:
            logger.error(f"Error adding document {file_path}: {e}", exc_info=True)
            raise

    def query(self, query_text: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Search the knowledge base.
        
        Args:
            query_text: The search query
            k: Number of results to return
            
        Returns:
            List of results with content and metadata
        """
        results = self.vector_db.similarity_search_with_score(query_text, k=k)
        
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score) # Chroma returns distance score (lower is better)
            })
            
        return formatted_results

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents in the knowledge base with metadata.
        
        Returns:
            List of document info dictionaries containing:
                - filename: Document filename
                - file_size: File size in bytes
                - chunks_count: Number of chunks (if available)
        """
        documents_info = []
        
        # Get unique sources from vector store
        try:
            collection = self.vector_db._collection
            count = collection.count()
            
            if count > 0:
                # Get all metadata to find unique sources
                results = collection.get(include=["metadatas"])
                sources_info = {}
                
                for metadata in results.get("metadatas", []):
                    source = metadata.get("source", "unknown")
                    if source not in sources_info:
                        sources_info[source] = {
                            "filename": source,
                            "file_size": metadata.get("file_size", 0),
                            "chunks_count": 0
                        }
                    sources_info[source]["chunks_count"] += 1
                
                documents_info = list(sources_info.values())
        except Exception as e:
            logger.warning(f"无法从向量数据库获取文档列表: {e}")
            # Fallback: List files in KB_DIR
            for f in KB_DIR.iterdir():
                if f.is_file():
                    documents_info.append({
                        "filename": f.name,
                        "file_size": f.stat().st_size,
                        "chunks_count": None
                    })
        
        return documents_info
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Dictionary with statistics:
                - total_documents: Number of documents
                - total_chunks: Total number of chunks
                - total_size: Total size of documents in bytes
        """
        try:
            collection = self.vector_db._collection
            total_chunks = collection.count()
            
            documents = self.list_documents()
            total_documents = len(documents)
            total_size = sum(doc.get("file_size", 0) for doc in documents)
            
            return {
                "total_documents": total_documents,
                "total_chunks": total_chunks,
                "total_size": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2)
            }
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}", exc_info=True)
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "total_size": 0,
                "total_size_mb": 0
            }

    def clear_database(self, clear_files: bool = False):
        """
        Clear the vector database.
        
        Args:
            clear_files: Whether to also delete files in KB_DIR
        """
        try:
            with self._lock:
                self.vector_db.delete_collection()
                self.vector_db = Chroma(
                    persist_directory=str(VECTOR_DB_DIR),
                    embedding_function=self.embedding_model,
                    collection_name="parrot_kb"
                )
            logger.info("Vector database cleared")
            
            if clear_files:
                # Clear files in KB_DIR
                for f in KB_DIR.iterdir():
                    if f.is_file():
                        f.unlink()
                logger.info("Knowledge base files cleared")
        except Exception as e:
            logger.error(f"清空数据库失败: {e}", exc_info=True)
            raise
    
    def delete_document(self, filename: str) -> bool:
        """
        Delete a specific document from the vector database.
        
        Args:
            filename: Name of the document to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._lock:
                collection = self.vector_db._collection
                
                # Get all document IDs with matching source
                results = collection.get(include=["metadatas", "ids"])
                ids_to_delete = []
                
                for i, metadata in enumerate(results.get("metadatas", [])):
                    if metadata.get("source") == filename:
                        ids_to_delete.append(results["ids"][i])
                
                if ids_to_delete:
                    collection.delete(ids=ids_to_delete)
                    logger.info(f"Deleted {len(ids_to_delete)} chunks for document: {filename}")
                else:
                    logger.warning(f"Document not found in vector db: {filename}")
                    # Even if not found in DB, try to delete file below
            
            # Delete file (outside lock as it's FS operation)
            file_path = KB_DIR / filename
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted file: {file_path}")
                return True
            
            return len(ids_to_delete) > 0
                
        except Exception as e:
            logger.error(f"删除文档失败: {e}", exc_info=True)
            return False
    
    def add_directory(self, directory_path: Path, recursive: bool = True) -> Dict[str, Any]:
        """
        批量处理目录中的所有支持格式文件
        
        Args:
            directory_path: 目录路径
            recursive: 是否递归处理子目录
            
        Returns:
            处理结果统计
        """
        if not directory_path.exists() or not directory_path.is_dir():
            raise ValueError(f"目录不存在或不是目录: {directory_path}")
        
        supported_exts = {'.pdf', '.txt', '.md', '.docx', '.csv', '.xlsx', '.xls'}
        results = {
            "processed": [],
            "failed": [],
            "total_files": 0,
            "success_count": 0,
            "failed_count": 0
        }
        
        # 收集所有文件
        pattern = "**/*" if recursive else "*"
        for file_path in directory_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in supported_exts:
                results["total_files"] += 1
                try:
                    result = self.add_document(file_path, move_to_kb=True)
                    results["processed"].append({
                        "filename": file_path.name,
                        "status": "success",
                        **result
                    })
                    results["success_count"] += 1
                except Exception as e:
                    results["failed"].append({
                        "filename": file_path.name,
                        "status": "failed",
                        "error": str(e)
                    })
                    results["failed_count"] += 1
                    logger.error(f"处理文件失败 {file_path.name}: {e}")
        
        return results
    
    def update_document(self, file_path: Path) -> Dict[str, Any]:
        """
        更新文档：先删除旧版本，再添加新版本
        
        Args:
            file_path: 文档路径
            
        Returns:
            处理结果
        """
        filename = file_path.name
        
        # 先删除旧版本
        self.delete_document(filename)
        
        # 再添加新版本
        return self.add_document(file_path, move_to_kb=True)

# Singleton instance
_kb_instance = None

def get_knowledge_base() -> LocalKnowledgeBase:
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = LocalKnowledgeBase()
    return _kb_instance


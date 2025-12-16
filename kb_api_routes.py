"""
知识库API路由

提供文档上传、查询、管理等功能的API接口。
"""

from fastapi import UploadFile, File, HTTPException, APIRouter, Query
from pathlib import Path
import logging
import asyncio
import shutil
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import the knowledge base logic
from knowledge_base import get_knowledge_base

# Setup logging
logger = logging.getLogger(__name__)

# 创建路由器
kb_router = APIRouter(prefix="/kb", tags=["Knowledge Base"])


@kb_router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    上传文档到知识库
    
    支持格式：PDF, TXT, MD, DOCX, CSV, XLSX, XLS
    上传后会自动进行分块、生成向量并存入 ChromaDB。
    
    Args:
        file: 上传的文件
        
    Returns:
        Dict: 处理结果，包含：
            - filename: 文件名
            - chunks_added: 生成的片段数量
            - total_chars: 总字符数
            - file_size: 文件大小（字节）
            - status: 处理状态
    """
    if not file:
        raise HTTPException(status_code=400, detail="未上传文件")
        
    filename = file.filename
    # 文件格式检查
    allowed_exts = {'.pdf', '.txt', '.md', '.docx', '.csv', '.xlsx', '.xls'}
    ext = Path(filename).suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400, 
            detail=f"不支持的文件格式: {ext}。支持格式: {', '.join(allowed_exts)}"
        )
        
    try:
        # 保存到 uploads 目录
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        upload_path = upload_dir / filename
        
        # 如果文件已存在，添加序号
        counter = 1
        while upload_path.exists():
            stem = Path(filename).stem
            suffix = Path(filename).suffix
            upload_path = upload_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        
        content = await file.read()
        with open(upload_path, "wb") as f:
            f.write(content)
            
        # 处理并添加到知识库
        kb = get_knowledge_base()
        # 在线程池中运行，避免阻塞
        result = await asyncio.to_thread(kb.add_document, upload_path, move_to_kb=True)
        
        return {
            "filename": upload_path.name,
            **result
        }
        
    except ValueError as e:
        # 文件格式错误
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"文档上传处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@kb_router.get("/documents")
async def list_documents():
    """
    列出知识库中已索引的文档
    
    Returns:
        Dict: 包含文档列表和统计信息
    """
    try:
        kb = get_knowledge_base()
        documents = kb.list_documents()
        
        return {
            "count": len(documents),
            "documents": documents
        }
    except Exception as e:
        logger.error(f"获取文档列表失败: {e}", exc_info=True)
        return {"count": 0, "documents": [], "error": str(e)}


@kb_router.get("/stats")
async def get_stats():
    """
    获取知识库统计信息
    
    Returns:
        Dict: 统计信息，包含：
            - total_documents: 文档总数
            - total_chunks: 片段总数
            - total_size: 总大小（字节）
            - total_size_mb: 总大小（MB）
    """
    try:
        kb = get_knowledge_base()
        stats = kb.get_stats()
        return stats
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@kb_router.get("/search")
async def search_knowledge_base(
    query: str = Query(..., description="搜索查询文本"),
    k: int = Query(3, ge=1, le=20, description="返回结果数量")
):
    """
    在知识库中搜索相关内容
    
    Args:
        query: 搜索查询文本
        k: 返回结果数量（1-20）
        
    Returns:
        Dict: 搜索结果列表，每个结果包含：
            - content: 内容文本
            - metadata: 元数据（来源文件等）
            - score: 相似度分数（越低越相似）
    """
    try:
        kb = get_knowledge_base()
        results = kb.query(query, k=k)
        
        return {
            "query": query,
            "count": len(results),
            "results": results
        }
    except Exception as e:
        logger.error(f"搜索失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@kb_router.delete("/documents/{filename}")
async def delete_document(filename: str):
    """
    删除知识库中的指定文档
    
    Args:
        filename: 要删除的文档文件名
        
    Returns:
        Dict: 删除结果
    """
    try:
        kb = get_knowledge_base()
        success = await asyncio.to_thread(kb.delete_document, filename)
        
        if success:
            return {
                "status": "success",
                "message": f"文档 '{filename}' 已删除"
            }
        else:
            raise HTTPException(status_code=404, detail=f"文档 '{filename}' 不存在")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除文档失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@kb_router.post("/upload_batch")
async def upload_batch_documents(files: List[UploadFile] = File(...)):
    """
    批量上传多个文档到知识库
    
    支持格式：PDF, TXT, MD, DOCX, CSV, XLSX, XLS
    
    Args:
        files: 上传的文件列表
        
    Returns:
        Dict: 批量处理结果
    """
    if not files:
        raise HTTPException(status_code=400, detail="未上传文件")
    
    logger.info(f"开始批量上传 {len(files)} 个文件")
    
    allowed_exts = {'.pdf', '.txt', '.md', '.docx', '.csv', '.xlsx', '.xls'}
    results = {
        "total": len(files),
        "success": [],
        "failed": []
    }
    
    kb = get_knowledge_base()
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    
    # 先保存所有文件
    file_paths = []
    for idx, file in enumerate(files, 1):
        filename = file.filename or f"file_{idx}"
        ext = Path(filename).suffix.lower()
        
        logger.info(f"保存文件 {idx}/{len(files)}: {filename}")
        
        if ext not in allowed_exts:
            error_msg = f"不支持的文件格式: {ext}"
            logger.warning(f"{filename}: {error_msg}")
            results["failed"].append({
                "filename": filename,
                "error": error_msg
            })
            continue
        
        try:
            # 保存文件
            upload_path = upload_dir / filename
            counter = 1
            while upload_path.exists():
                stem = Path(filename).stem
                suffix = Path(filename).suffix
                upload_path = upload_dir / f"{stem}_{counter}{suffix}"
                counter += 1
            
            content = await file.read()
            file_size = len(content)
            logger.info(f"文件大小: {file_size} 字节 ({file_size / 1024 / 1024:.2f} MB)")
            
            with open(upload_path, "wb") as f:
                f.write(content)
            
            file_paths.append((upload_path, filename))
        except Exception as e:
            error_msg = str(e)
            logger.error(f"保存文件失败 {filename}: {error_msg}", exc_info=True)
            results["failed"].append({
                "filename": filename,
                "error": f"保存文件失败: {error_msg}"
            })
    
    # 并行处理文件（如果只有一个文件，直接处理；多个文件则并行）
    if len(file_paths) == 0:
        logger.warning("没有文件需要处理")
    elif len(file_paths) == 1:
        # 单个文件，直接处理
        upload_path, filename = file_paths[0]
        try:
            logger.info(f"开始处理文档: {filename}")
            result = await asyncio.to_thread(kb.add_document, upload_path, move_to_kb=True)
            logger.info(f"文档处理完成: {filename}, 片段数: {result.get('chunks_added', 0)}")
            results["success"].append({
                "filename": filename,
                **result
            })
        except Exception as e:
            error_msg = str(e)
            logger.error(f"处理文件失败 {filename}: {error_msg}", exc_info=True)
            results["failed"].append({
                "filename": filename,
                "error": error_msg
            })
    else:
        # 多个文件，并行处理
        # 注意：由于ChromaDB (SQLite) 不支持高并发写入，我们将max_workers设置为1，
        # 或者虽然使用多线程但实际上由于KnowledgeBase中的锁，写入是串行的。
        # 保持较低的并发数以避免超时和资源争用。
        logger.info(f"开始处理 {len(file_paths)} 个文件...")
        max_workers = 1  # 串行处理以确保稳定性
        
        async def process_single_file(upload_path: Path, filename: str):
            """处理单个文件的异步包装"""
            try:
                logger.info(f"开始处理文档: {filename}")
                result = await asyncio.to_thread(kb.add_document, upload_path, move_to_kb=True)
                logger.info(f"文档处理完成: {filename}, 片段数: {result.get('chunks_added', 0)}")
                return {
                    "success": True,
                    "filename": filename,
                    "result": result
                }
            except Exception as e:
                error_msg = str(e)
                logger.error(f"处理文件失败 {filename}: {error_msg}", exc_info=True)
                return {
                    "success": False,
                    "filename": filename,
                    "error": error_msg
                }
        
        # 创建任务列表
        tasks = [process_single_file(upload_path, filename) for upload_path, filename in file_paths]
        
        # 使用asyncio.gather并行执行，但限制并发数
        # 由于asyncio.gather会同时执行所有任务，我们需要分批处理
        batch_size = max_workers
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch)
            
            for batch_result in batch_results:
                if batch_result["success"]:
                    results["success"].append({
                        "filename": batch_result["filename"],
                        **batch_result["result"]
                    })
                else:
                    results["failed"].append({
                        "filename": batch_result["filename"],
                        "error": batch_result["error"]
                    })
    
    logger.info(f"批量上传完成: 成功 {len(results['success'])} 个, 失败 {len(results['failed'])} 个")
    
    return {
        **results,
        "success_count": len(results["success"]),
        "failed_count": len(results["failed"])
    }


@kb_router.post("/add_directory")
async def add_directory_to_kb(
    directory_path: str = Query(..., description="要处理的目录路径（相对于项目根目录）"),
    recursive: bool = Query(True, description="是否递归处理子目录")
):
    """
    批量处理目录中的所有支持格式文件
    
    Args:
        directory_path: 目录路径（相对路径）
        recursive: 是否递归处理子目录
        
    Returns:
        Dict: 处理结果统计
    """
    try:
        dir_path = Path(directory_path).expanduser().resolve()
        
        if not dir_path.exists():
            raise HTTPException(status_code=404, detail=f"目录不存在: {directory_path}")
        
        if not dir_path.is_dir():
            raise HTTPException(status_code=400, detail=f"不是目录: {directory_path}")
        
        kb = get_knowledge_base()
        result = await asyncio.to_thread(kb.add_directory, dir_path, recursive=recursive)
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理目录失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@kb_router.delete("/clear")
async def clear_knowledge_base(clear_files: bool = Query(False, description="是否同时删除文件")):
    """
    清空知识库（慎用）
    
    Args:
        clear_files: 是否同时删除 knowledge_base 目录中的文件
        
    Returns:
        Dict: 清空结果
    """
    try:
        kb = get_knowledge_base()
        await asyncio.to_thread(kb.clear_database, clear_files=clear_files)
        
        # 如果清空文件，也清理 uploads 目录
        if clear_files:
            upload_dir = Path("uploads")
            if upload_dir.exists():
                shutil.rmtree(upload_dir)
                upload_dir.mkdir()
            
        return {
            "status": "success",
            "message": "知识库已清空" + ("（包括文件）" if clear_files else "")
        }
    except Exception as e:
        logger.error(f"清空知识库失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@kb_router.get("/config")
async def get_kb_config():
    """
    获取知识库配置信息
    
    Returns:
        Dict: 配置信息，包含知识库路径等
    """
    try:
        from knowledge_base import KB_DIR, VECTOR_DB_DIR, UPLOAD_DIR
        
        return {
            "kb_dir": str(KB_DIR.resolve()),
            "vector_db_dir": str(VECTOR_DB_DIR.resolve()),
            "upload_dir": str(UPLOAD_DIR.resolve()),
            "kb_dir_exists": KB_DIR.exists(),
            "vector_db_dir_exists": VECTOR_DB_DIR.exists(),
        }
    except Exception as e:
        logger.error(f"获取配置失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

"""
Parrot Set - FastAPI Backend
Refactored to use modular Agent architecture.
"""
import os
import json
import logging
import asyncio
import mimetypes
import io
import platform
import subprocess
import time
import threading
import queue
from pathlib import Path
from typing import Dict, Any, List, Optional

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pydantic import BaseModel, Field
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from PIL import Image

# Import from new modules
from agent.core import get_agent
from agent.models import ClassificationResult, AnalyzeResult
from agent.rag import get_rag_engine
from agent.router import AgentRouter, detect_followup, detect_needs_image
from agent.memory import SessionMemoryStore
from parrot_db import get_database
from utils import save_upload_temp, save_classified_image

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
CONFIG_FILE = Path("config.json")

def load_config() -> Dict[str, Any]:
    default = {
        "model_settings": {
            "model_name": "qwen3-vl:2b-instruct-q4_K_M",
            "temperature": 0.3
        },
        "api_settings": {"host": "0.0.0.0", "port": 8000}
    }
    if not CONFIG_FILE.exists():
        return default
    try:
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except:
        return default

APP_CONFIG = load_config()
MODEL_NAME = APP_CONFIG["model_settings"].get("model_name", "qwen3-vl:2b-instruct-q4_K_M")
LLM_TEMPERATURE = float(APP_CONFIG["model_settings"].get("temperature", 0.3))

if os.getenv("PARROT_MODEL_NAME"):
    MODEL_NAME = os.getenv("PARROT_MODEL_NAME")

# App Init
app = FastAPI(title="Parrot Set MVP", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure dataset directory exists and mount it (still useful for default path)
dataset_path = Path("dataset")
dataset_path.mkdir(exist_ok=True)
app.mount("/dataset", StaticFiles(directory="dataset"), name="dataset")

# Agent Instance
agent = get_agent()
router = AgentRouter()
session_store = SessionMemoryStore(ttl_seconds=3600, max_turns=10)

# RAG Engine: 启动时初始化，提前暴露 torch 环境问题
_rag_engine_initialized = False
_rag_engine_error = None
try:
    rag_engine = get_rag_engine()
    # 尝试预加载 embedding 模型（触发 torch 导入和模型加载）
    # 使用一个空列表测试，不会真正向量化，只是触发模型加载
    logger.info("Pre-loading RAG embedding model...")
    try:
        # 触发 _load_model，但不实际向量化
        if hasattr(rag_engine.embedding_fn, '_load_model'):
            rag_engine.embedding_fn._load_model()
        logger.info("RAG embedding model pre-loaded successfully.")
        _rag_engine_initialized = True
    except Exception as e:
        _rag_engine_error = str(e)
        logger.error(f"Failed to pre-load RAG embedding model: {e}", exc_info=True)
        logger.warning("RAG service will be unavailable. Vectorization will fail until torch environment is fixed.")
        # 确保标志为 False，即使 RAGEngine 对象已创建
        _rag_engine_initialized = False
except Exception as e:
    _rag_engine_error = str(e)
    logger.error(f"Failed to initialize RAG engine: {e}", exc_info=True)
    logger.warning("RAG service will be unavailable.")
    _rag_engine_initialized = False

def _json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return json.dumps(obj)

# Routes

@app.get("/health")
async def health():
    import requests
    try:
        requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        ollama_ok = True
    except:
        ollama_ok = False
    return {
        "status": "ok",
        "ollama_available": ollama_ok,
        "model": MODEL_NAME,
        "rag_available": _rag_engine_initialized,
        "rag_error": _rag_engine_error if not _rag_engine_initialized else None
    }

@app.post("/classify", response_model=ClassificationResult)
async def classify(image: UploadFile = File(...)):
    tmp_path = None
    try:
        tmp_path = save_upload_temp(image)
        logger.info(f"Classifying: {image.filename}")
        
        # Call Agent
        result = await asyncio.wait_for(
            agent.classify(tmp_path),
            timeout=600.0
        )
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timed out")
    except Exception as e:
        logger.error(f"Classification failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

@app.post("/analyze", response_model=AnalyzeResult)
async def analyze(image: UploadFile = File(...)):
    tmp_path = None
    try:
        tmp_path = save_upload_temp(image)
        logger.info(f"Analyzing: {image.filename}")
        
        # Call Agent
        result = await asyncio.wait_for(
            agent.analyze(tmp_path),
            timeout=600.0
        )
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timed out")
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

# ---------- Agent Q&A (text only) ----------
class AskRequest(BaseModel):
    question: str = Field(..., description="用户问题（中文）")
    top_k: int = Field(3, ge=1, le=10, description="RAG 返回片段数")
    species_hint: Optional[str] = Field(None, description="可选：物种提示（用于“刚才那只...”）")

class AskResponse(BaseModel):
    answer: str
    hits: List[Dict[str, Any]] = []

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """
    Text-only Q&A endpoint.
    Uses RAG to retrieve relevant knowledge, then (if Ollama available) uses Agent main LLM to answer.
    """
    answer, artifacts = await router.run_ask(req.question, top_k=req.top_k, species_hint=req.species_hint)
    return AskResponse(answer=answer, hits=artifacts.get("hits", []))

# ---------- RAG Debug / Benchmark Endpoint ----------
class RagRetrieveRequest(BaseModel):
    query: str = Field(..., description="检索 query（建议中文自然语言）")
    top_k: int = Field(3, ge=1, le=20, description="返回片段数")
    force_rebuild: bool = Field(False, description="是否强制重建索引（用于测试构建耗时）")

class RagRetrieveResponse(BaseModel):
    query: str
    top_k: int
    elapsed_ms: int
    index_rebuilt: bool
    collection_count: int
    hits: List[Dict[str, Any]]

@app.post("/rag/retrieve", response_model=RagRetrieveResponse)
async def rag_retrieve(req: RagRetrieveRequest):
    """
    直接调用 RAG 检索，用于测试“检索质量/速度”。\n
    返回：命中片段（content/score/source/chunk_info）+ 耗时信息。\n
    说明：\n
    - 若 force_rebuild=true，会先强制 build_index，再执行 retrieve。\n
    - retrieve/build_index 可能较耗时（embedding 模型加载/索引构建），这里用 to_thread 避免阻塞事件循环。\n
    """
    q = (req.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="query is required")

    rag = get_rag_engine()
    index_rebuilt = False
    t0 = time.monotonic()

    if req.force_rebuild:
        index_rebuilt = True
        await asyncio.to_thread(rag.build_index, True)

    hits = await asyncio.to_thread(rag.retrieve, q, req.top_k)
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    # collection_count 可能触发内部 build_index（retrieve 会在 count==0 时构建）
    try:
        collection_count = int(rag.collection.count())
    except Exception:
        collection_count = -1

    return RagRetrieveResponse(
        query=q,
        top_k=req.top_k,
        elapsed_ms=elapsed_ms,
        index_rebuilt=index_rebuilt,
        collection_count=collection_count,
        hits=hits,
    )

# ---------- Unified Agent Router Endpoint ----------
class AgentMessageResponse(BaseModel):
    session_id: str
    mode: str  # analyze / ask / prompt
    reply: str
    artifacts: Dict[str, Any] = {}
    debug: Optional[Dict[str, Any]] = None

def _sse(event: str, data: Any) -> str:
    """
    SSE 格式化：每条事件以空行分隔。
    data 统一序列化为 JSON 字符串（前端可 JSON.parse）。
    """
    return f"event: {event}\ndata: {_json_dumps(data)}\n\n"

@app.post("/agent/message", response_model=AgentMessageResponse)
async def agent_message(
    session_id: Optional[str] = Form(None),
    message: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
):
    logger.info(f"/agent/message session_id={session_id!r} has_image={image is not None} msg_len={len((message or '').strip())}")
    st = session_store.get_or_create(session_id)
    user_text = (message or "").strip()

    # Persist user text (image is represented as a placeholder)
    if image is not None:
        session_store.append(st, "user", "[image]")
    if user_text:
        session_store.append(st, "user", user_text)

    used_species_hint = None
    mode = "ask"
    artifacts: Dict[str, Any] = {}

    # Routing rules
    if image is not None:
        mode = "analyze"
        tmp_path = None
        try:
            tmp_path = save_upload_temp(image)
            reply, artifacts = await router.run_analyze(tmp_path)
            species = None
            try:
                species = artifacts.get("classification", {}).get("top_candidates", [{}])[0].get("name")
            except Exception:
                species = None
            session_store.set_last_species(st, species)
            session_store.set_last_analyze_artifacts(st, artifacts)
        except Exception as e:
            logger.error(f"/agent/message analyze failed: {e}", exc_info=True)
            reply = f"图片分析失败：{str(e)}"
            artifacts = {"error": str(e)}
        finally:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
    else:
        # Text only
        if not user_text:
            mode = "prompt"
            reply = "你可以直接提问，或上传一张鹦鹉图片让我进行识别。"
        else:
            if detect_needs_image(user_text):
                mode = "prompt"
                reply = "要进行识别/分类，请先上传一张清晰的鹦鹉图片（正面/侧面都可以）。"
            else:
                mode = "ask"
                # Hybrid follow-up detection: LLM semantic router + keyword fallback
                if st.last_species:
                    route = await router.decide_text_routing(user_text, last_species=st.last_species)
                    if route.get("is_followup"):
                        used_species_hint = st.last_species
                else:
                    # no session species available; keep behavior deterministic
                    used_species_hint = None

                reply, artifacts = await router.run_ask(user_text, top_k=3, species_hint=used_species_hint)

    session_store.append(st, "agent", reply)

    debug = {
        "used_species_hint": used_species_hint,
        "last_species": st.last_species,
        "session_count": session_store.size(),
        "history_len": len(st.history),
    }
    if mode == "analyze" and artifacts:
        debug["analyze_summary"] = router.summarize_artifacts_for_debug(artifacts)
    if mode == "ask" and artifacts.get("hits") is not None:
        try:
            debug["hit_sources"] = [h.get("source") for h in artifacts.get("hits", [])]
        except Exception:
            pass
    return AgentMessageResponse(
        session_id=st.session_id,
        mode=mode,
        reply=reply,
        artifacts=artifacts,
        debug=debug
    )

@app.post("/agent/message/stream")
async def agent_message_stream(
    session_id: Optional[str] = Form(None),
    message: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
):
    """
    SSE 流式版本：逐步输出状态与 token。

    事件类型：
    - status: {"text": "..."}
    - tool_start/tool_end: {"tool_name": "...", ...}
    - token: {"delta": "..."}
    - done: {session_id, mode, reply, artifacts, debug}
    """

    async def gen():
        start_ts = time.monotonic()
        yield _sse("status", {"text": "已收到请求，准备处理中…"})

        # 流式逻辑：
        # - 文本：路由(LLM) -> (可选)RAG -> 主 LLM token 流
        # - 图片：ClassifierTool -> (低置信度)RAG -> 终判 LLM token 流（JSON）-> 汇总回复
        st = session_store.get_or_create(session_id)
        user_text = (message or "").strip()

        if image is not None:
            session_store.append(st, "user", "[image]")
        if user_text:
            session_store.append(st, "user", user_text)

        used_species_hint = None
        mode = "ask"
        artifacts: Dict[str, Any] = {}
        reply = ""

        try:
            if image is not None:
                mode = "analyze"
                tmp_path = None
                try:
                    tmp_path = save_upload_temp(image)
                    async for ev in router.stream_analyze_events(tmp_path):
                        if ev.get("event") == "final":
                            reply = ev["data"]["reply"]
                            artifacts = ev["data"]["artifacts"]
                        else:
                            yield _sse(ev.get("event", "status"), ev.get("data", {}))

                    # 保存会话信息（用于追问绑定）
                    species = None
                    try:
                        species = artifacts.get("classification", {}).get("top_candidates", [{}])[0].get("name")
                    except Exception:
                        species = None
                    session_store.set_last_species(st, species)
                    session_store.set_last_analyze_artifacts(st, artifacts)
                finally:
                    if tmp_path and tmp_path.exists():
                        tmp_path.unlink(missing_ok=True)
            else:
                if not user_text:
                    mode = "prompt"
                    reply = "你可以直接提问，或上传一张鹦鹉图片让我进行识别。"
                elif detect_needs_image(user_text):
                    mode = "prompt"
                    reply = "要进行识别/分类，请先上传一张清晰的鹦鹉图片（正面/侧面都可以）。"
                else:
                    mode = "ask"
                    # 追问绑定与是否使用 RAG 的决策在 router.stream_ask_events 内完成（LLM few-shot + 兜底）
                    async for ev in router.stream_ask_events(user_text, top_k=3, last_species=st.last_species):
                        if ev.get("event") == "final":
                            reply = ev["data"]["reply"]
                            artifacts = ev["data"]["artifacts"]
                            used_species_hint = artifacts.get("used_species_hint") if isinstance(artifacts, dict) else None
                        else:
                            yield _sse(ev.get("event", "status"), ev.get("data", {}))

        except Exception as e:
            logger.error(f"/agent/message/stream failed: {e}", exc_info=True)
            mode = "prompt"
            reply = f"处理失败：{str(e)}"
            artifacts = {"error": str(e)}

        # 写入会话历史
        session_store.append(st, "agent", reply)

        debug = {
            "used_species_hint": used_species_hint,
            "last_species": st.last_species,
            "session_count": session_store.size(),
            "history_len": len(st.history),
            "elapsed_ms": int((time.monotonic() - start_ts) * 1000),
        }
        if mode == "ask":
            try:
                debug["hit_sources"] = [h.get("source") for h in artifacts.get("hits", [])] if isinstance(artifacts, dict) else []
            except Exception:
                pass

        yield _sse(
            "done",
            {
                "session_id": st.session_id,
                "mode": mode,
                "reply": reply,
                "artifacts": artifacts,
                "debug": debug,
            },
        )

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        # For some proxies (nginx) - harmless otherwise
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

# Knowledge Base Routes
@app.post("/knowledge/upload")
async def upload_knowledge(file: UploadFile = File(...)):
    """Upload a document to the knowledge base."""
    # 扩展支持的文件类型
    allowed_extensions = ('.md', '.txt', '.pdf', '.docx')
    if not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(status_code=400, detail=f"Only {allowed_extensions} files are supported")
    
    try:
        rag = get_rag_engine()
        save_path = rag.knowledge_base_path / file.filename
        
        # 直接保存文件（支持二进制文件）
        content = await file.read()
        save_path.write_bytes(content)
        
        logger.info(f"Saved uploaded file to {save_path}")
        return {"status": "success", "message": f"Uploaded {file.filename}. 请点击“重建索引/向量化”按钮使索引生效。"}
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge/list")
async def list_knowledge():
    """List all documents in the knowledge base."""
    rag = get_rag_engine()
    return {"documents": rag.list_documents()}

@app.delete("/knowledge/{filename}")
async def delete_knowledge(filename: str):
    """Delete a document from the knowledge base."""
    rag = get_rag_engine()
    success = rag.delete_document(filename)
    if success:
        return {"status": "success", "message": f"Deleted {filename}. 请点击“重建索引/向量化”按钮使索引生效。"}
    else:
        raise HTTPException(status_code=404, detail="File not found or delete failed")

@app.post("/knowledge/reindex")
async def reindex_knowledge(force_full: bool = False):
    """
    向量化/索引同步入口（手动触发）。

    默认：增量同步（只向量化新增/变更文档，并删除已移除文档的向量）。
    force_full=true：全量重建索引（更慢，但最干净）。
    """
    if not _rag_engine_initialized:
        error_msg = f"RAG 服务未初始化。原因：{_rag_engine_error or '未知错误'}"
        raise HTTPException(
            status_code=503,
            detail=error_msg + " 请检查 torch/transformers 环境是否正确安装。"
        )
    
    try:
        rag = get_rag_engine()
        result = await asyncio.to_thread(rag.sync_index, force_full)
        return {"status": "success", "mode": result.get("mode"), "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/knowledge/reindex/stream")
async def reindex_knowledge_stream(force_full: bool = False):
    """
    SSE 流式重建索引：后端会持续推送进度，前端可实时显示“向量化到哪一步了”。

    事件类型：
    - status: {"text": "..."}
    - progress: {"phase": "...", ...}
    - done: {"status":"success", "result": {...}}
    - error: {"status":"error", "detail":"..."}
    """
    # 检查 RAG 是否已初始化
    if not _rag_engine_initialized:
        error_msg = f"RAG 服务未初始化。原因：{_rag_engine_error or '未知错误'}"
        logger.error(error_msg)
        
        async def gen_error():
            yield _sse("status", {"text": "RAG 服务未就绪"})
            yield _sse("error", {
                "status": "error",
                "detail": error_msg,
                "suggestion": "请检查 torch/transformers 环境是否正确安装。如果是在 Windows 上遇到 DLL 错误，请检查 VC++ 运行库和 torch 版本兼容性。"
            })
        
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(gen_error(), media_type="text/event-stream", headers=headers)
    
    rag = get_rag_engine()

    q: "queue.Queue[Any]" = queue.Queue()
    sentinel = object()

    def on_progress(evt: Dict[str, Any]):
        # evt 约定为 {"event": "...", "data": {...}}
        q.put(evt)

    def worker():
        try:
            result = rag.sync_index(force_full_rebuild=force_full, on_progress=on_progress)
            q.put({"event": "done", "data": {"status": "success", "result": result}})
        except Exception as e:
            q.put({"event": "error", "data": {"status": "error", "detail": str(e)}})
        finally:
            q.put(sentinel)

    threading.Thread(target=worker, daemon=True).start()

    async def gen():
        start = time.monotonic()
        try:
            yield _sse("status", {"text": "已开始向量化/同步索引…"})
            # 心跳：避免长时间无输出时，前端看起来“卡住”
            while True:
                try:
                    item = await asyncio.to_thread(q.get, True, 1.0)  # block=True, timeout=1s
                except queue.Empty:
                    elapsed = int((time.monotonic() - start) * 1000)
                    yield _sse("status", {"text": f"向量化中…（已运行 {elapsed}ms）"})
                    continue
                if item is sentinel:
                    break
                if isinstance(item, dict) and item.get("event"):
                    yield _sse(item["event"], item.get("data", {}))
                    # 如果收到 error 事件，立即结束流
                    if item.get("event") == "error":
                        break
                else:
                    yield _sse("status", {"text": str(item)})
        except Exception as e:
            logger.error(f"SSE generator error: {e}", exc_info=True)
            yield _sse("error", {"status": "error", "detail": f"流式输出异常：{str(e)}"})

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

@app.post("/save_classified")
async def save_classified(
    image: UploadFile = File(...),
    species: str = Form(...),
    output_path: str = Form("./dataset")
):
    try:
        return save_classified_image(image, species, output_path)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/check_species")
async def check_species(name: str):
    db = get_database()
    record = db.find_by_name(name)
    if record:
        return {
            "exists": True,
            "info": {k: v for k, v in record.items() if k != 'name_variants'},
            "suggestions": []
        }
    
    suggestions = db.fuzzy_search(name, limit=5)
    return {
        "exists": False,
        "info": None,
        "suggestions": [
            {k: s[k] for k in ['chinese_name', 'english_name', 'scientific_name']}
            for s in suggestions
        ]
    }

@app.get("/search_species")
async def search_species(query: str, limit: int = 10):
    db = get_database()
    results = db.fuzzy_search(query, limit=limit)
    return {"count": len(results), "results": results}

@app.get("/stats/species")
async def get_species_stats(output_path: str = "./dataset"):
    logger.info(f"Received stats request for path: {output_path}")
    try:
        # 常用鹦鹉列表（作为基础展示）
        known_species = [
            "蓝黄金刚鹦鹉", "五彩金刚鹦鹉", "绯红金刚鹦鹉", "紫蓝金刚鹦鹉",
            "灰鹦鹉", "折衷鹦鹉", "葵花凤头鹦鹉", "小葵花凤头鹦鹉", "玄凤鹦鹉",
            "虎皮鹦鹉", "牡丹鹦鹉", "和尚鹦鹉", "吸蜜鹦鹉",
            "凯克鹦鹉", "亚马逊鹦鹉", "金太阳鹦鹉", "白腹凯克鹦鹉", "黑头凯克鹦鹉"
        ]
        
        stats = []
        output_dir = Path(output_path).expanduser().resolve()
        collected_data = {}
        
        if output_dir.exists():
            for item in output_dir.iterdir():
                if item.is_dir():
                    count = sum(1 for f in item.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp'])
                    if count > 0:
                        collected_data[item.name] = count
        
        for species in known_species:
            count = collected_data.get(species, 0)
            stats.append({
                "name": species,
                "count": count,
                "collected": count > 0,
                "source": "knowledge_base"
            })
        
        for species, count in collected_data.items():
            if species not in known_species:
                stats.append({
                    "name": species,
                    "count": count,
                    "collected": True,
                    "source": "dataset"
                })
        
        stats.sort(key=lambda x: (not x["collected"], x["name"]))
        return {
            "total_species": len(stats),
            "collected_species": len([s for s in stats if s["collected"]]),
            "species_list": stats
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/file_view")
def view_file(
    path: str = Query(..., description="Full path to the image file"),
    thumbnail: bool = False,
    width: int = 300
):
    """
    Proxy endpoint to view local image files with thumbnail support.
    """
    file_path = Path(path).resolve()
    
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
        
    # Security check
    if file_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp']:
        raise HTTPException(status_code=400, detail="Only image files are allowed")

    # Cache Control Header (1 hour)
    headers = {"Cache-Control": "public, max-age=3600"}

    if thumbnail:
        try:
            with Image.open(file_path) as img:
                # Convert to RGB if necessary (e.g. for PNG with transparency)
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                
                # Calculate height to preserve aspect ratio
                aspect_ratio = img.height / img.width
                new_height = int(width * aspect_ratio)
                
                img.thumbnail((width, new_height), Image.Resampling.LANCZOS)
                
                img_io = io.BytesIO()
                img.save(img_io, 'JPEG', quality=85)
                img_io.seek(0)
                
                return StreamingResponse(
                    img_io, 
                    media_type="image/jpeg",
                    headers=headers
                )
        except Exception as e:
            logger.error(f"Error creating thumbnail for {file_path}: {e}")
            # Fallback to full image if thumbnail creation fails
            pass
        
    # Full image
    media_type, _ = mimetypes.guess_type(file_path)
    if not media_type:
        media_type = "application/octet-stream"
        
    return StreamingResponse(
        open(file_path, "rb"), 
        media_type=media_type,
        headers=headers
    )

@app.get("/collection/{species_name}")
async def get_collection_images(
    species_name: str, 
    output_path: str = Query("./dataset", description="Root path where images are saved")
):
    """
    Get list of collected images for a specific species from the custom output path.
    """
    root_path = Path(output_path).expanduser().resolve()
    species_dir = root_path / species_name
    
    if not species_dir.exists() or not species_dir.is_dir():
        return {"images": []}
        
    images = []
    for f in species_dir.iterdir():
        if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
            # Return path as parameter
            proxy_url = f"/file_view?path={f.resolve()}"
            images.append(proxy_url)
            
    # Sort by modification time (newest first)
    try:
        images.sort(key=lambda x: Path(x.split("path=")[1]).stat().st_mtime, reverse=True)
    except Exception as e:
        logger.warning(f"Error sorting images: {e}")
            
    return {"images": images}

@app.post("/open_folder")
async def open_folder(path: str = Form(...)):
    """Open the folder in the system's file explorer."""
    folder_path = Path(path).resolve()
    
    # Try to resolve to the parent directory if it's a file or doesn't exist directly but parent does
    if not folder_path.exists() or not folder_path.is_dir():
        if folder_path.parent.exists() and folder_path.parent.is_dir():
            folder_path = folder_path.parent
        elif not folder_path.exists():
             raise HTTPException(status_code=404, detail="Folder not found")
            
    try:
        if platform.system() == "Windows":
            os.startfile(folder_path)
        elif platform.system() == "Darwin":  # macOS
            subprocess.Popen(["open", str(folder_path)])
        else:  # Linux
            subprocess.Popen(["xdg-open", str(folder_path)])
        return {"status": "success", "message": f"Opened {folder_path}"}
    except Exception as e:
        logger.error(f"Failed to open folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # 默认关闭 reload：向量化/索引会频繁写 data/ 与 knowledge/，在 --reload 模式下会触发热重载，
    # 导致 SSE 连接中断、前端看不到进度。需要开发热重载时：设置 PARROT_RELOAD=1
    reload = os.getenv("PARROT_RELOAD", "0") == "1"
    # 注意：传入 "app:app" 会再次 import 本文件（第一次是 __main__，第二次是 module app），
    # 导致顶层初始化逻辑（包括 RAG 预加载）重复执行，日志重复且在某些环境下可能引发意外退出。
    # - 非 reload：直接传入 app 对象，避免二次 import。
    # - 需要 reload：必须使用 import string（uvicorn 限制）。
    if reload:
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000)

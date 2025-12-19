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
    return {"status": "ok", "ollama_available": ollama_ok, "model": MODEL_NAME}

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

# ---------- Unified Agent Router Endpoint ----------
class AgentMessageResponse(BaseModel):
    session_id: str
    mode: str  # analyze / ask / prompt
    reply: str
    artifacts: Dict[str, Any] = {}
    debug: Optional[Dict[str, Any]] = None

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
        
        # 触发索引重建
        rag.build_index(force_rebuild=True)
        
        return {"status": "success", "message": f"Uploaded {file.filename}"}
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
        return {"status": "success", "message": f"Deleted {filename}"}
    else:
        raise HTTPException(status_code=404, detail="File not found or delete failed")

@app.post("/knowledge/reindex")
async def reindex_knowledge():
    """Force rebuild the knowledge index."""
    try:
        rag = get_rag_engine()
        rag.build_index(force_rebuild=True)
        return {"status": "success", "message": "Index rebuild triggered"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

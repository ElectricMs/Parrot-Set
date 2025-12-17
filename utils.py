import shutil
from pathlib import Path
from typing import Dict, Any
from fastapi import UploadFile, HTTPException
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def save_upload_temp(upload: UploadFile) -> Path:
    """Save uploaded file to temporary path."""
    data = upload.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    
    suffix = Path(upload.filename or "").suffix or ".jpg"
    tmp_path = Path("tmp_upload" + suffix)
    tmp_path.write_bytes(data)
    return tmp_path

def save_classified_image(
    upload: UploadFile,
    species_name: str,
    output_path: str = "./dataset"
) -> Dict[str, Any]:
    """Save classified image to organized folder structure."""
    if not species_name or not species_name.strip():
        raise HTTPException(status_code=400, detail="Species name required")
    
    upload.file.seek(0)
    data = upload.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    
    clean_species = "".join(
        c for c in species_name.strip() 
        if c not in '<>:"/\\|?*'
    ).strip() or "Unknown"
    
    output_dir = Path(output_path).expanduser().resolve()
    species_dir = output_dir / clean_species
    
    try:
        species_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create directory: {e}")
        raise HTTPException(status_code=500, detail=f"Directory creation failed: {e}")
    
    original_name = upload.filename or "image"
    file_ext = Path(original_name).suffix or ".jpg"
    file_stem = Path(original_name).stem
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{file_stem}_{timestamp}{file_ext}"
    target_path = species_dir / file_name
    
    counter = 1
    while target_path.exists():
        file_name = f"{file_stem}_{timestamp}_{counter}{file_ext}"
        target_path = species_dir / file_name
        counter += 1
    
    try:
        target_path.write_bytes(data)
        logger.info(f"Image saved: {target_path}")
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        raise HTTPException(status_code=500, detail=f"File save failed: {e}")
    
    return {
        "success": True,
        "file_path": str(target_path),
        "folder_path": str(species_dir),
        "species": clean_species,
        "message": f"Saved to: {target_path}"
    }


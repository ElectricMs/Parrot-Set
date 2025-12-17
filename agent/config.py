import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

CONFIG_FILE = Path("config.json")

# Default configuration
DEFAULT_AGENT_CONFIG = {
    "main_model": "qwen2.5:7b-instruct", # Pure LLM for reasoning/explanation
    "main_temperature": 0.5,
    "tools": {
        "classifier": {
            "model_name": "qwen3-vl:2b-instruct-q4_K_M", # Vision model
            "temperature": 0.1
        },
        "analyzer": { # Analyzer pipeline configuration
            "use_main_model_for_explanation": True # If true, use main_model for explanation
        }
    }
}

def load_agent_config() -> Dict[str, Any]:
    """
    Load agent configuration from config.json.
    """
    config = DEFAULT_AGENT_CONFIG.copy()
    
    if not CONFIG_FILE.exists():
        return config

    try:
        file_content = CONFIG_FILE.read_text(encoding="utf-8")
        full_config = json.loads(file_content)
        
        user_agent_config = full_config.get("agent_settings", {})
        
        # Merge logic
        if user_agent_config:
            # Update top level
            for k, v in user_agent_config.items():
                if k != "tools" and v is not None:
                    config[k] = v
            
            # Update tools
            if "tools" in user_agent_config:
                for tool_name, tool_conf in user_agent_config["tools"].items():
                    if tool_name in config["tools"]:
                        config["tools"][tool_name].update(tool_conf)
                    else:
                        config["tools"][tool_name] = tool_conf
                        
        return config
        
    except Exception as e:
        logger.error(f"Failed to load agent config: {e}, using defaults")
        return config

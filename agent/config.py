import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

CONFIG_FILE = Path("config.json")

# Default configuration
DEFAULT_AGENT_CONFIG = {
    # main_model：主 LLM（通常为纯文本模型），用于路由、问答、终判等
    "main_model": "qwen2.5:7b-instruct",
    "main_temperature": 0.5,
    "tools": {
        "classifier": {
            # model_name：多模态视觉模型（必须支持图片输入）
            "model_name": "qwen3-vl:2b-instruct-q4_K_M",
            "temperature": 0.1
        },
        "analyzer": {  # 旧版 analyze 流水线的配置占位（当前解释主要由 ClassifierTool 生成）
            "use_main_model_for_explanation": True
        }
    }
}

def load_agent_config() -> Dict[str, Any]:
    """
    读取 Agent 配置：从项目根目录 config.json 的 agent_settings 合并到默认配置。

    合并策略：
    - 顶层字段（main_model/main_temperature）若用户配置不为 None，则覆盖默认值
    - tools 下按工具名 merge（支持新增工具配置）

    向后兼容：
    - 早期配置可能使用 default_model/default_temperature，运行时会映射到 main_model/main_temperature
    """
    config = DEFAULT_AGENT_CONFIG.copy()
    
    if not CONFIG_FILE.exists():
        return config

    try:
        file_content = CONFIG_FILE.read_text(encoding="utf-8")
        full_config = json.loads(file_content)
        
        user_agent_config = full_config.get("agent_settings", {})

        # Backward compatibility:
        # Older configs used `default_model/default_temperature` but runtime expects
        # `main_model/main_temperature`.
        if isinstance(user_agent_config, dict):
            if user_agent_config.get("main_model") is None and user_agent_config.get("default_model") is not None:
                user_agent_config["main_model"] = user_agent_config.get("default_model")
            if user_agent_config.get("main_temperature") is None and user_agent_config.get("default_temperature") is not None:
                user_agent_config["main_temperature"] = user_agent_config.get("default_temperature")
        
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

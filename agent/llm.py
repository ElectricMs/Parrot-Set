"""
LLM Wrapper for Ollama
"""
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from typing import Optional, List, Any
import requests
import base64
from pathlib import Path

class OllamaLLM(LLM):
    """
    LangChain wrapper for Ollama local model.
    Supports text generation and multimodal inputs (images).
    """
    
    model_name: str = "qwen3-vl-2b"
    api_url: str = "http://127.0.0.1:11434"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    
    def __init__(
        self,
        model_name: str = "qwen3-vl-2b",
        api_url: str = "http://127.0.0.1:11434",
        temperature: float = 0.7,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.api_url = api_url.rstrip("/")
        self.temperature = temperature
    
    @property
    def _llm_type(self) -> str:
        return "ollama"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        images: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Call Ollama API.
        """
        content = prompt
        
        # Handle images
        image_data_list = []
        if images:
            for img_path in images:
                img_path_obj = Path(img_path)
                if not img_path_obj.exists():
                    raise FileNotFoundError(f"Image not found: {img_path}")
                
                with img_path_obj.open("rb") as f:
                    img_data = base64.b64encode(f.read()).decode("utf-8")
                    image_data_list.append(img_data)
        
        # Build payload
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
            }
        }
        
        if image_data_list:
            payload["messages"][0]["images"] = image_data_list
        
        if self.max_tokens:
            payload["options"]["num_predict"] = self.max_tokens
        
        if stop:
            payload["options"]["stop"] = stop
        
        # Send request
        try:
            response = requests.post(
                f"{self.api_url}/api/chat",
                json=payload,
                timeout=(30, 600)  # Connection timeout 30s, Read timeout 600s
            )
            
            if response.status_code != 200:
                raise RuntimeError(
                    f"Ollama API failed (Status {response.status_code}): {response.text}"
                )
            
            response.raise_for_status()
            data = response.json()
            
            if "message" in data and "content" in data["message"]:
                return data["message"]["content"]
            elif "response" in data:
                return data["response"]
            else:
                return str(data)
                
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                raise RuntimeError(f"Ollama API failed: {e}, Detail: {e.response.text}")
            raise RuntimeError(f"Ollama API failed: {e}")

def get_llm_instance(model_name: str, temperature: float = 0.7) -> OllamaLLM:
    """Factory function to create LLM instance"""
    return OllamaLLM(model_name=model_name, temperature=temperature)


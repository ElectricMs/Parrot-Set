"""
LLM Wrapper for Ollama

本模块提供一个最小可用的 Ollama /api/chat 封装（不依赖 LangChain）。

关键点：
- 支持纯文本与图片输入（images 参数）
- 图片会被读取并 base64 编码后放入 payload.messages[0].images
- 使用 /api/chat（非 stream）一次性返回

注意：
- 该实现的 _call 是同步网络请求；上层调用时通常用 asyncio.to_thread 包一层，避免阻塞事件循环。
- 这里故意不引入 langchain/transformers/torch 等重依赖，避免在 Windows 环境出现 torch DLL 失败时导致服务无法启动。
"""
from typing import Optional, List, Any, Iterator
import requests
import base64
import json
from pathlib import Path

class OllamaLLM:
    """
    Minimal wrapper for Ollama local model.
    Supports text generation and multimodal inputs (images).
    """
    
    def __init__(
        self,
        model_name: str = "qwen3-vl-2b",
        api_url: str = "http://127.0.0.1:11434",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        # kwargs kept for backward compatibility (call sites may pass extra args)
        self.model_name = model_name
        self.api_url = api_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        images: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """
        调用 Ollama API（/api/chat）。

        参数：
        - prompt：用户输入（字符串）
        - images：图片文件路径列表（可选）。注意：这里传的是路径，函数内部会读取并 base64 编码。
        - stop/max_tokens/temperature：会转成 Ollama 的 options 字段

        返回：
        - 模型返回的 content 字符串（不保证是 JSON；JSON 由上层工具负责约束与解析）
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
        
        # Build payload 请求体
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

    def stream_chat(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """
        流式调用 Ollama（/api/chat, stream=true）。

        返回一个同步迭代器：每次 yield 一个增量文本片段（delta）。
        上层通常会在单独线程中迭代，并把 delta 通过 SSE/WebSocket 推送到前端。

        说明：
        - Ollama 的 stream 响应通常是 NDJSON（每行一个 JSON 对象）
        - 每个对象可能包含 message.content（增量），done=true 表示结束
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
            "messages": [{"role": "user", "content": content}],
            "stream": True,
            "options": {"temperature": self.temperature},
        }
        if image_data_list: # 如果有图片数据，则将其添加到用户消息中
            payload["messages"][0]["images"] = image_data_list
        if self.max_tokens: # 如果设置了最大令牌数，则将其添加到选项中
            payload["options"]["num_predict"] = self.max_tokens
        if stop: # 如果有停止标记，则将其添加到选项中
            payload["options"]["stop"] = stop

        try:
            response = requests.post(
                f"{self.api_url}/api/chat",
                json=payload,
                stream=True,
                timeout=(30, 600),
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Ollama API failed (Status {response.status_code}): {response.text}"
                )
            response.raise_for_status()

            for line in response.iter_lines(decode_unicode=True): # 逐行读取响应内容，解码为 Unicode 字符串
                if not line:
                    continue
                try:
                    data = json.loads(line) # 尝试将每一行解析为 JSON 对象
                except Exception:
                    continue

                # 提取和返回文本增量    
                # Typical streaming shape:
                # {"message":{"role":"assistant","content":"..."},"done":false,...}
                delta = "" # 初始化一个空字符串，用于存储文本增量
                if isinstance(data, dict):
                    msg = data.get("message")
                    if isinstance(msg, dict):
                        delta = msg.get("content") or ""
                    # Some variants use "response"
                    if not delta and data.get("response"):
                        delta = data.get("response") or ""
                    if delta:
                        yield delta
                    if data.get("done") is True:
                        break
        except requests.exceptions.RequestException as e:
            if hasattr(e, "response") and e.response is not None:
                raise RuntimeError(f"Ollama API failed: {e}, Detail: {e.response.text}")
            raise RuntimeError(f"Ollama API failed: {e}")

def get_llm_instance(model_name: str, temperature: float = 0.7) -> OllamaLLM:
    """
    工厂函数：创建一个 OllamaLLM 实例。
    说明：目前只暴露 model_name/temperature 两个关键参数，其余参数可按需扩展。
    """
    return OllamaLLM(model_name=model_name, temperature=temperature)


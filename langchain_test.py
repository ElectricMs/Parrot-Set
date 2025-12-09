"""
LangChain 包装器：将 Ollama 本地模型（支持多模态）接入 LangChain。

功能：
- 支持文本对话
- 支持图片输入（多模态，如 Qwen3-VL）
- 使用 Ollama 的 /api/chat 端点（现代 API）
"""

from langchain_core.language_models.llms import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from typing import Optional, List, Dict, Any
import requests
import base64
from pathlib import Path


class OllamaLLM(LLM):
    """
    LangChain 包装器：封装 Ollama 本地模型调用。
    
    支持：
    - 文本生成
    - 多模态（图片 + 文本）
    """
    
    model_name: str = "qwen3-vl-2b"
    api_url: str = "http://127.0.0.1:11434"  # Ollama 默认端口
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
        调用 Ollama API 生成文本。
        
        Args:
            prompt: 文本提示
            stop: 停止词列表
            run_manager: 回调管理器
            images: 图片路径列表（支持多模态）
            **kwargs: 其他参数
        
        Returns:
            生成的文本
        """
        # 构建消息内容
        # Ollama API: content 是字符串，图片通过单独的 images 参数传递
        content = prompt
        
        # 处理图片（Ollama 需要 base64 编码的图片数组）
        image_data_list = []
        if images:
            for img_path in images:
                img_path_obj = Path(img_path)
                if not img_path_obj.exists():
                    raise FileNotFoundError(f"图片不存在: {img_path}")
                
                # 读取图片并转 base64
                with img_path_obj.open("rb") as f:
                    img_data = base64.b64encode(f.read()).decode("utf-8")
                    image_data_list.append(img_data)
        
        # 构建请求 payload（使用 /api/chat 端点）
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
        
        # 如果有图片，添加到 payload
        if image_data_list:
            payload["messages"][0]["images"] = image_data_list
        
        if self.max_tokens:
            payload["options"]["num_predict"] = self.max_tokens
        
        if stop:
            payload["options"]["stop"] = stop
        
        # 发送请求
        try:
            response = requests.post(
                f"{self.api_url}/api/chat",
                json=payload,
                timeout=300  # 5 分钟超时
            )
            
            # 如果出错，打印详细错误信息
            if response.status_code != 200:
                error_detail = response.text
                raise RuntimeError(
                    f"Ollama API 调用失败 (状态码 {response.status_code}): {error_detail}\n"
                    f"请求 payload: {payload}"
                )
            
            response.raise_for_status()
            data = response.json()
            
            # 提取生成的文本
            if "message" in data and "content" in data["message"]:
                return data["message"]["content"]
            elif "response" in data:
                return data["response"]
            else:
                return str(data)
                
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                error_detail = e.response.text
                raise RuntimeError(
                    f"Ollama API 调用失败: {e}\n错误详情: {error_detail}\n"
                    f"请求 payload: {payload}"
                )
            raise RuntimeError(f"Ollama API 调用失败: {e}")


# ========== 使用示例 ==========

if __name__ == "__main__":
    # 示例 1: 纯文本对话
    print("=== 示例 1: 纯文本对话 ===")
    llm = OllamaLLM(model_name="qwen3-vl:2b-thinking-q4_K_M")
    try:
        # LangChain LLM 使用 invoke() 方法
        output = llm.invoke("请用一句话描述自由女神像")
        print(output)
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    print()
    
    # 示例 2: 多模态（图片 + 文本）
    print("=== 示例 2: 多模态识别鹦鹉 ===")
    llm_vision = OllamaLLM(model_name="qwen3-vl:2b-thinking-q4_K_M", temperature=0.7)
    
    # 假设有鹦鹉图片
    image_path = "samples/1.JPG"
    if Path(image_path).exists():
        try:
            # 对于多模态，需要通过 _call 方法直接调用
            output = llm_vision._call(
                "请识别这只鹦鹉的品种，并给出依据。",
                images=[image_path]
            )
            print(output)
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"图片不存在: {image_path}")

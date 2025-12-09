"""
Qwen3-VL-2B-Thinking GGUF 本地多模态推理示例（llama-cpp-python）。

用法（CPU 示例）：
    python qwen_vl_demo.py ^
        --model "Qwen3VL-2B-Thinking-Q4_K_M.gguf" ^
        --mmproj "mmproj-Qwen3VL-2B-Thinking-Q8_0.gguf" ^
        --image samples/1.jpg ^
        --prompt "请识别这只鹦鹉的品种，并给出依据。"

如有 GPU（CUDA 版 llama-cpp-python），可加：
    --gpu-layers 20
"""

import argparse
import base64
from pathlib import Path
from typing import Optional

from llama_cpp import Llama
from PIL import Image  # noqa: F401  # pillow is required by llama_cpp for vision


def encode_image_to_b64(image_path: Path) -> str:
    """读取图片并转 base64，供 <image:...> 占位使用。"""
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_prompt(image_b64: str, user_prompt: str) -> str:
    """拼接 llava 风格的多模态 prompt，Qwen3-VL 使用 <image:...> 标签。"""
    return f"<image:{image_b64}>\n{user_prompt.strip()}\n"


def run_inference(
    model_path: Path,
    mmproj_path: Optional[Path],
    image_path: Path,
    prompt: str,
    n_threads: int,
    n_ctx: int,
    gpu_layers: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    # 初始化模型：
    # - model_path 指向主 GGUF（已量化）
    # - mmproj_path 有些仓库会单独提供视觉投影；若 GGUF 已内置，可不传
    # - chat_format="llava-v1" 让 llama-cpp 按图文格式解析 <image:...> 标签
    llm = Llama(
        model_path=str(model_path),
        mmproj=str(mmproj_path) if mmproj_path else None,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=gpu_layers,  # 0 表示纯 CPU
        chat_format="llava-v1",
    )

    image_b64 = encode_image_to_b64(image_path)
    full_prompt = build_prompt(image_b64, prompt)

    result = llm(
        full_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return result["choices"][0]["text"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 llama-cpp-python 运行 Qwen3-VL-2B-Thinking GGUF（视觉）。"
    )
    parser.add_argument("--model", required=True, type=Path, help="主 GGUF 权重路径。")
    parser.add_argument(
        "--mmproj",
        type=Path,
        default=None,
        help="可选，视觉投影 GGUF（若模型未内置视觉头则需要）。",
    )
    parser.add_argument("--image", required=True, type=Path, help="待识别图片路径。")
    parser.add_argument(
        "--prompt",
        required=True,
        help="用户问题/指令，例如：请识别并解释这张鹦鹉照片。",
    )
    parser.add_argument("--threads", type=int, default=8, help="CPU 线程数。")
    parser.add_argument("--ctx", type=int, default=4096, help="上下文长度。")
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=0,
        help="下 offload 到 GPU 的层数（0 表示纯 CPU）。",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度。")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p 采样。")
    parser.add_argument("--max-tokens", type=int, default=512, help="生成最大长度。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for p in [args.model, args.image] + ([args.mmproj] if args.mmproj else []):
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

    output = run_inference(
        model_path=args.model,
        mmproj_path=args.mmproj,
        image_path=args.image,
        prompt=args.prompt,
        n_threads=args.threads,
        n_ctx=args.ctx,
        gpu_layers=args.gpu_layers,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    print("\n=== Model Output ===\n")
    print(output.strip())


if __name__ == "__main__":
    main()


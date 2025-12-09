# !pip install llama-cpp-python

from llama_cpp import Llama

llm = Llama.from_pretrained(
	repo_id="Qwen/Qwen3-VL-2B-Thinking-GGUF",
	filename="Qwen3VL-2B-Thinking-Q4_K_M.gguf",
)

llm.create_chat_completion(
	messages = [
		{
			"role": "user",
			"content": [
				{
					"type": "text",
					"text": "Describe this image in one sentence."
				},
				{
					"type": "image_url",
					"image_url": {
						"url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
					}
				}
			]
		}
	]
)
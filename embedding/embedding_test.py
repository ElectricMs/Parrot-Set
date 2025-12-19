# Requires transformers>=4.51.0

import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    """
    从模型的隐藏状态中提取最后一个有效token的表示作为文本嵌入向量
    
    参数:
        last_hidden_states: 模型输出的所有token的隐藏状态 [batch_size, seq_len, hidden_size]
        attention_mask: 注意力掩码，标记哪些位置是真实token（1）哪些是填充（0）
    
    返回:
        每个文本的嵌入向量 [batch_size, hidden_size]
    """
    # 检查是否是左填充（left padding）：如果所有序列的最后一个位置都是1，说明是左填充
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        # 左填充时，最后一个token就是序列的最后一个位置
        return last_hidden_states[:, -1]
    else:
        # 右填充时，需要找到每个序列的最后一个有效token的位置
        # attention_mask.sum(dim=1) - 1 得到每个序列最后一个有效token的索引
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        # 从每个序列中提取对应位置的隐藏状态
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    """
    为查询添加任务描述指令，形成完整的指令格式
    
    参数:
        task_description: 任务描述，说明这个查询的用途
        query: 实际的查询问题
    
    返回:
        格式化后的完整指令字符串
    """
    return f'Instruct: {task_description}\nQuery:{query}'

# 定义任务描述：这是一个网页搜索查询任务，需要检索相关文档
# 每个查询都需要带有一个一句话的指令来描述任务
task = 'Given a web search query, retrieve relevant passages that answer the query'

# 准备两个查询示例，每个查询都包含任务指令和具体问题
queries = [
    get_detailed_instruct(task, 'What is the capital of China?'),  # 查询1：中国的首都是什么？
    get_detailed_instruct(task, 'Explain gravity')                 # 查询2：解释重力
]

# 准备两个文档，用于检索匹配（文档不需要添加指令）
documents = [
    "The capital of China is Beijing.",  # 文档1：关于中国首都的答案
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."  # 文档2：关于重力的解释
]

# 将所有文本合并：前2个是查询，后2个是文档
input_texts = queries + documents

# 加载分词器和模型：使用 Qwen3-Embedding-0.6B 嵌入模型
# padding_side='left' 表示在左侧填充（对于生成式模型通常用左填充）
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B', padding_side='left')
model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B')

# 推荐使用 flash_attention_2 以获得更好的加速和内存节省（需要GPU和相应依赖）
# model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B', attn_implementation="flash_attention_2", torch_dtype=torch.float16).cuda()

max_length = 8192  # 最大序列长度

# 对输入文本进行分词和编码
# padding=True: 自动填充到批次中最长序列的长度
# truncation=True: 如果超过max_length则截断
# return_tensors="pt": 返回PyTorch张量格式
batch_dict = tokenizer(
    input_texts,
    padding=True,
    truncation=True,
    max_length=max_length,
    return_tensors="pt",
)

# 将输入数据移动到模型所在的设备（CPU或GPU）
batch_dict.to(model.device)

# 将分词后的输入传入模型，获取隐藏状态
outputs = model(**batch_dict)

# 从模型的隐藏状态中提取每个文本的嵌入向量（使用最后一个有效token）
embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

# 对嵌入向量进行L2归一化（单位化），使得向量长度为1
# 归一化后的向量点积就等于余弦相似度
embeddings = F.normalize(embeddings, p=2, dim=1)

# 计算查询和文档之间的相似度分数（余弦相似度）
# embeddings[:2] 是前2个查询的嵌入向量
# embeddings[2:] 是后2个文档的嵌入向量
# @ 表示矩阵乘法，.T 表示转置
# 结果是一个2x2矩阵：
#   scores[0][0]: 查询1与文档1的相似度
#   scores[0][1]: 查询1与文档2的相似度
#   scores[1][0]: 查询2与文档1的相似度
#   scores[1][1]: 查询2与文档2的相似度
scores = (embeddings[:2] @ embeddings[2:].T)

# 打印相似度分数矩阵
# 运行结果示例：
# [[0.764556884765625, 0.14142510294914246], [0.13549764454364777, 0.5999547839164734]]
# 
# 结果解读：
# - 第一行 [0.764, 0.141]：查询1（中国首都）与文档1（北京）相似度0.764（高），与文档2（重力）相似度0.141（低）
# - 第二行 [0.135, 0.599]：查询2（重力）与文档1（北京）相似度0.135（低），与文档2（重力）相似度0.599（较高）
# 
# 可以看到，相关的查询-文档对相似度较高，不相关的相似度较低，说明模型能够有效区分文本语义
print(scores.tolist())

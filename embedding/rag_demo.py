"""
简易 RAG (Retrieval-Augmented Generation) 流程演示
结合嵌入检索和 Qwen3-VL 模型回答问题
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import sys
import pickle
import json
import hashlib

# 添加项目根目录到路径，以便导入 agent.llm
sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.llm import get_llm_instance


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """
    从模型的隐藏状态中提取最后一个有效token的表示作为文本嵌入向量
    
    参数:
        last_hidden_states: 模型输出的所有token的隐藏状态 [batch_size, seq_len, hidden_size]
        attention_mask: 注意力掩码，标记哪些位置是真实token（1）哪些是填充（0）
    
    返回:
        每个文本的嵌入向量 [batch_size, hidden_size]
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
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


class SimpleRAG:
    """
    简易 RAG 系统
    使用嵌入模型进行文档检索，使用 Qwen3-VL 模型生成答案
    """
    
    def __init__(
        self,
        embedding_model_name: str = 'Qwen/Qwen3-Embedding-0.6B',
        llm_model_name: str = 'qwen3-vl:2b-instruct-q4_K_M',
        llm_temperature: float = 0.7,
        knowledge_base_path: str = None,
        task_description: str = 'Given a web search query, retrieve relevant passages that answer the query'
    ):
        """
        初始化 RAG 系统
        
        参数:
            embedding_model_name: 嵌入模型名称
            llm_model_name: 生成模型名称（用于回答问题）
            llm_temperature: LLM 温度参数
            knowledge_base_path: 知识库目录路径（如果为 None，则自动查找）
            task_description: 检索任务描述
        """
        self.task_description = task_description
        
        # 自动查找知识库路径
        if knowledge_base_path is None:
            # 从当前文件位置向上查找 knowledge 目录
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent
            knowledge_base_path = project_root / 'knowledge'
        
        self.knowledge_base_path = Path(knowledge_base_path).resolve()
        
        print("正在加载嵌入模型...")
        # 加载嵌入模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            embedding_model_name, 
            padding_side='left'
        )
        self.embedding_model = AutoModel.from_pretrained(embedding_model_name)
        self.embedding_model.eval()  # 设置为评估模式
        
        print("正在加载生成模型...")
        # 加载 LLM 模型（用于生成答案）
        self.llm = get_llm_instance(llm_model_name, llm_temperature)
        
        # 存储文档和嵌入向量
        self.documents: List[str] = []
        self.document_embeddings: Tensor = None
        self.max_length = 8192
        
        # 向量缓存目录（存储在 embedding 目录下的 cache 子目录）
        self.cache_dir = Path(__file__).parent / 'cache'
        self.cache_dir.mkdir(exist_ok=True)
        
        # 缓存文件路径
        self.embeddings_cache_file = self.cache_dir / 'document_embeddings.pt'
        self.documents_cache_file = self.cache_dir / 'documents.pkl'
        self.metadata_cache_file = self.cache_dir / 'metadata.json'
        
    def load_knowledge_base(self) -> Tuple[List[str], Dict]:
        """
        从知识库目录加载所有文档，并返回元数据
        
        返回:
            (文档列表, 元数据字典)
        """
        documents = []
        metadata = {
            "knowledge_base_path": str(self.knowledge_base_path),
            "files": {}
        }
        
        if not self.knowledge_base_path.exists():
            print(f"警告: 知识库目录不存在: {self.knowledge_base_path}")
            return documents, metadata
        
        # 读取所有 .md 文件
        for md_file in sorted(self.knowledge_base_path.glob("*.md")):
            if md_file.name == "README.md":
                continue
            try:
                content = md_file.read_text(encoding='utf-8')
                documents.append(content)
                
                # 记录文件元数据（修改时间和文件大小，用于检查是否需要重新索引）
                stat = md_file.stat()
                metadata["files"][md_file.name] = {
                    "mtime": stat.st_mtime,
                    "size": stat.st_size,
                    "hash": hashlib.md5(content.encode('utf-8')).hexdigest()
                }
                print(f"已加载文档: {md_file.name}")
            except Exception as e:
                print(f"加载文档失败 {md_file.name}: {e}")
        
        return documents, metadata
    
    def _compute_knowledge_base_hash(self, metadata: Dict) -> str:
        """
        计算知识库的哈希值，用于判断是否需要重新索引
        
        参数:
            metadata: 知识库元数据
        
        返回:
            哈希字符串
        """
        # 基于文件列表和每个文件的哈希值计算总体哈希
        file_info = sorted([
            (name, info["hash"])
            for name, info in metadata["files"].items()
        ])
        combined = json.dumps(file_info, sort_keys=True)
        return hashlib.md5(combined.encode('utf-8')).hexdigest()
    
    def save_index(self, documents: List[str], embeddings: Tensor, metadata: Dict):
        """
        保存索引（文档和向量）到缓存文件
        
        参数:
            documents: 文档列表
            embeddings: 嵌入向量张量
            metadata: 元数据字典
        """
        try:
            print("正在保存索引到缓存...")
            
            # 保存向量（使用 torch.save）
            torch.save(embeddings, self.embeddings_cache_file)
            
            # 保存文档列表（使用 pickle）
            with open(self.documents_cache_file, 'wb') as f:
                pickle.dump(documents, f)
            
            # 保存元数据（包含知识库哈希）
            metadata["knowledge_base_hash"] = self._compute_knowledge_base_hash(metadata)
            with open(self.metadata_cache_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"索引已保存到: {self.cache_dir}")
        except Exception as e:
            print(f"保存索引失败: {e}")
            import traceback
            traceback.print_exc()
    
    def load_index(self) -> Optional[Tuple[List[str], Tensor, Dict]]:
        """
        从缓存文件加载索引
        
        返回:
            (文档列表, 嵌入向量, 元数据) 如果加载成功，否则返回 None
        """
        try:
            # 检查所有缓存文件是否存在
            if not all([
                self.embeddings_cache_file.exists(),
                self.documents_cache_file.exists(),
                self.metadata_cache_file.exists()
            ]):
                return None
            
            print("正在从缓存加载索引...")
            
            # 加载向量
            embeddings = torch.load(self.embeddings_cache_file)
            
            # 加载文档列表
            with open(self.documents_cache_file, 'rb') as f:
                documents = pickle.load(f)
            
            # 加载元数据
            with open(self.metadata_cache_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            print(f"成功从缓存加载 {len(documents)} 个文档的索引")
            return documents, embeddings, metadata
            
        except Exception as e:
            print(f"加载缓存失败: {e}")
            return None
    
    def is_index_valid(self, cached_metadata: Dict) -> bool:
        """
        检查缓存的索引是否仍然有效（知识库是否有更新）
        
        参数:
            cached_metadata: 缓存的元数据
        
        返回:
            True 如果索引仍然有效，False 如果需要重新构建
        """
        # 检查知识库路径是否匹配
        if cached_metadata.get("knowledge_base_path") != str(self.knowledge_base_path):
            print("知识库路径已更改，需要重新构建索引")
            return False
        
        # 加载当前知识库的元数据
        _, current_metadata = self.load_knowledge_base()
        
        # 计算当前知识库的哈希
        current_hash = self._compute_knowledge_base_hash(current_metadata)
        cached_hash = cached_metadata.get("knowledge_base_hash")
        
        if current_hash != cached_hash:
            print("知识库内容已更新，需要重新构建索引")
            return False
        
        print("缓存索引仍然有效")
        return True
    
    def encode_texts(self, texts: List[str]) -> Tensor:
        """
        将文本列表编码为嵌入向量
        
        参数:
            texts: 文本列表
        
        返回:
            嵌入向量张量 [batch_size, hidden_size]
        """
        # 分词和编码
        batch_dict = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch_dict.to(self.embedding_model.device)
        
        # 获取嵌入向量
        with torch.no_grad():
            outputs = self.embedding_model(**batch_dict)
            embeddings = last_token_pool(
                outputs.last_hidden_state, 
                batch_dict['attention_mask']
            )
            # L2 归一化
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def build_index(self, force_rebuild: bool = False):
        """
        构建文档索引：加载知识库并生成嵌入向量
        如果缓存存在且有效，则直接加载缓存，否则重新构建
        
        参数:
            force_rebuild: 如果为 True，强制重新构建索引（忽略缓存）
        """
        print("\n正在构建文档索引...")
        
        # 如果不需要强制重建，尝试加载缓存
        if not force_rebuild:
            cached_data = self.load_index()
            if cached_data is not None:
                documents, embeddings, metadata = cached_data
                # 检查索引是否仍然有效
                if self.is_index_valid(metadata):
                    self.documents = documents
                    self.document_embeddings = embeddings
                    print(f"使用缓存索引，共 {len(self.documents)} 个文档\n")
                    return
                else:
                    print("缓存已过期，将重新构建索引...")
        
        # 重新构建索引
        self.documents, metadata = self.load_knowledge_base()
        
        if not self.documents:
            print("警告: 没有加载到任何文档！")
            return
        
        print(f"正在为 {len(self.documents)} 个文档生成嵌入向量...")
        self.document_embeddings = self.encode_texts(self.documents)
        
        # 保存索引到缓存
        self.save_index(self.documents, self.document_embeddings, metadata)
        
        print(f"索引构建完成！共 {len(self.documents)} 个文档\n")
    
    def retrieve(self, query: str, top_k: int = 2) -> List[Tuple[str, float]]:
        """
        检索与查询最相关的文档
        
        参数:
            query: 查询文本
            top_k: 返回前 k 个最相关的文档
        
        返回:
            [(文档内容, 相似度分数), ...] 列表，按相似度降序排列
        """
        if self.document_embeddings is None or len(self.documents) == 0:
            raise ValueError("请先调用 build_index() 构建索引！")
        
        # 为查询生成嵌入向量
        query_with_instruct = get_detailed_instruct(self.task_description, query)
        query_embedding = self.encode_texts([query_with_instruct])
        
        # 计算查询与所有文档的相似度（余弦相似度）
        # query_embedding: [1, hidden_size]
        # document_embeddings: [num_docs, hidden_size]
        # scores: [1, num_docs]
        scores = query_embedding @ self.document_embeddings.T
        
        # 获取 top_k 个最相关的文档
        top_scores, top_indices = torch.topk(scores[0], k=min(top_k, len(self.documents)))
        
        # 返回文档内容和相似度分数
        results = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            results.append((self.documents[idx], score))
        
        return results
    
    def generate_answer(self, query: str, retrieved_docs: List[Tuple[str, float]]) -> str:
        """
        基于检索到的文档生成答案
        
        参数:
            query: 用户查询
            retrieved_docs: 检索到的文档列表 [(文档内容, 相似度分数), ...]
        
        返回:
            生成的答案
        """
        # 构建提示词
        context = "\n\n".join([
            f"文档 {i+1} (相似度: {score:.3f}):\n{doc}"
            for i, (doc, score) in enumerate(retrieved_docs)
        ])
        
        prompt = f"""基于以下检索到的文档，回答用户的问题。

检索到的相关文档：
{context}

用户问题：{query}

请基于上述文档内容，用中文详细回答用户的问题。如果文档中没有相关信息，请说明无法从知识库中找到答案。
"""
        
        print("\n正在生成答案...")
        answer = self.llm._call(prompt)
        return answer
    
    def query(self, question: str, top_k: int = 2) -> Dict:
        """
        完整的 RAG 查询流程：检索 + 生成
        
        参数:
            question: 用户问题
            top_k: 检索前 k 个文档
        
        返回:
            包含检索结果和生成答案的字典
        """
        print(f"\n用户问题: {question}")
        print("-" * 50)
        
        # 1. 检索相关文档
        print("正在检索相关文档...")
        retrieved_docs = self.retrieve(question, top_k=top_k)
        
        print(f"\n检索到 {len(retrieved_docs)} 个相关文档:")
        for i, (doc, score) in enumerate(retrieved_docs):
            # 显示文档的前100个字符
            preview = doc[:100].replace('\n', ' ') + "..."
            print(f"  [{i+1}] 相似度: {score:.3f} - {preview}")
        
        # 2. 生成答案
        answer = self.generate_answer(question, retrieved_docs)
        
        return {
            "question": question,
            "retrieved_documents": [
                {"content": doc, "similarity": score}
                for doc, score in retrieved_docs
            ],
            "answer": answer
        }


def main():
    """
    主函数：演示 RAG 流程
    
    命令行参数:
        --rebuild: 强制重新构建索引（忽略缓存）
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='简易 RAG 系统演示')
    parser.add_argument(
        '--rebuild',
        action='store_true',
        help='强制重新构建索引（忽略缓存）'
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("简易 RAG 系统演示")
    print("=" * 60)
    
    # 初始化 RAG 系统（knowledge_base_path 会自动查找，也可以手动指定）
    rag = SimpleRAG(
        llm_model_name='qwen3-vl:2b-instruct-q4_K_M',
        llm_temperature=0.7
    )
    
    # 构建索引（如果指定了 --rebuild，则强制重建）
    rag.build_index(force_rebuild=args.rebuild)
    
    # 示例查询
    questions = [
        "虎皮鹦鹉有什么特征？",
        "哪些鹦鹉有冠羽？",
        "蓝黄金刚鹦鹉的学名是什么？",
    ]
    
    # 逐个查询
    for question in questions:
        result = rag.query(question, top_k=2)
        
        print("\n" + "=" * 60)
        print("生成的答案:")
        print("=" * 60)
        print(result["answer"])
        print("\n")
    
    # 交互式查询
    print("\n" + "=" * 60)
    print("进入交互模式（输入 'quit' 退出）")
    print("=" * 60)
    
    while True:
        try:
            question = input("\n请输入您的问题: ").strip()
            if question.lower() in ['quit', 'exit', '退出']:
                break
            if not question:
                continue
            
            result = rag.query(question, top_k=2)
            print("\n" + "=" * 60)
            print("生成的答案:")
            print("=" * 60)
            print(result["answer"])
            
        except KeyboardInterrupt:
            print("\n\n退出程序")
            break
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()


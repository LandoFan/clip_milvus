"""
混合检索：结合向量检索和关键词检索
"""
import re
from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import Counter, defaultdict


class BM25:
    """BM25关键词检索算法"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        初始化BM25
        
        Args:
            k1: 词频饱和度参数
            b: 长度归一化参数
        """
        self.k1 = k1
        self.b = b
        self.documents = []
        self.frequencies = []
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.idf = {}
        self.vocab = set()
    
    def fit(self, documents: List[str]):
        """
        训练BM25模型
        
        Args:
            documents: 文档列表
        """
        self.documents = documents
        self.frequencies = []
        self.doc_lengths = []
        
        # 处理文档
        for doc in documents:
            words = self._tokenize(doc)
            self.vocab.update(words)
            freq = Counter(words)
            self.frequencies.append(freq)
            self.doc_lengths.append(len(words))
        
        # 计算平均文档长度
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        
        # 计算IDF
        n_docs = len(documents)
        for word in self.vocab:
            df = sum(1 for freq in self.frequencies if word in freq)
            self.idf[word] = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
    
    def _tokenize(self, text: str) -> List[str]:
        """简单的分词"""
        # 移除标点，转换为小写，分词
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        return words
    
    def score(self, query: str, doc_idx: int) -> float:
        """
        计算查询和文档的相关度分数
        
        Args:
            query: 查询文本
            doc_idx: 文档索引
            
        Returns:
            BM25分数
        """
        if doc_idx >= len(self.frequencies):
            return 0.0
        
        query_terms = self._tokenize(query)
        doc_freq = self.frequencies[doc_idx]
        doc_length = self.doc_lengths[doc_idx]
        
        score = 0.0
        for term in query_terms:
            if term not in self.idf:
                continue
            
            tf = doc_freq.get(term, 0)
            if tf == 0:
                continue
            
            # BM25公式
            idf = self.idf[term]
            numerator = idf * tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            score += numerator / denominator
        
        return score
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        搜索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回前k个结果
            
        Returns:
            [(文档索引, 分数)] 列表
        """
        scores = []
        for i in range(len(self.documents)):
            score = self.score(query, i)
            if score > 0:
                scores.append((i, score))
        
        # 按分数排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]


class HybridRetriever:
    """混合检索器：结合向量检索和关键词检索"""
    
    def __init__(self,
                 alpha: float = 0.7,
                 bm25_k1: float = 1.5,
                 bm25_b: float = 0.75):
        """
        初始化混合检索器
        
        Args:
            alpha: 向量检索权重 (0-1)，alpha越大，向量检索权重越高
            bm25_k1: BM25参数k1
            bm25_b: BM25参数b
        """
        self.alpha = alpha
        self.beta = 1.0 - alpha  # 关键词检索权重
        self.bm25 = BM25(k1=bm25_k1, b=bm25_b)
        self.documents = []
        self.doc_indices = []  # 文档在原始数据中的索引映射
    
    def index_documents(self, documents: List[str], indices: Optional[List[int]] = None):
        """
        索引文档
        
        Args:
            documents: 文档文本列表
            indices: 文档在原始数据中的索引（可选）
        """
        self.documents = documents
        self.doc_indices = indices if indices is not None else list(range(len(documents)))
        self.bm25.fit(documents)
    
    def hybrid_search(self,
                     query_text: str,
                     vector_scores: List[Tuple[int, float]],
                     top_k: int = 10,
                     normalize: bool = True) -> List[Tuple[int, float]]:
        """
        混合检索
        
        Args:
            query_text: 查询文本
            vector_scores: 向量检索结果 [(doc_idx, score), ...]
            top_k: 返回前k个结果
            normalize: 是否归一化分数
            
        Returns:
            [(doc_idx, final_score), ...] 列表
        """
        # 向量检索分数
        vector_score_map = {idx: score for idx, score in vector_scores}
        
        # 关键词检索分数
        bm25_scores = self.bm25.search(query_text, top_k=len(self.documents))
        bm25_score_map = {idx: score for idx, score in bm25_scores}
        
        # 获取所有候选文档索引
        all_indices = set(vector_score_map.keys()) | set(bm25_score_map.keys())
        
        # 归一化分数
        if normalize:
            # 归一化向量分数 (转换为相似度，距离越小分数越高)
            if vector_scores:
                max_vector_score = max(score for _, score in vector_scores)
                min_vector_score = min(score for _, score in vector_scores)
                vector_range = max_vector_score - min_vector_score if max_vector_score != min_vector_score else 1.0
                
                for idx in all_indices:
                    if idx in vector_score_map:
                        # 如果是距离，转换为相似度（距离越小，相似度越高）
                        normalized = 1.0 - (vector_score_map[idx] - min_vector_score) / vector_range
                        vector_score_map[idx] = max(0.0, normalized)
            
            # 归一化BM25分数
            if bm25_scores:
                max_bm25_score = max(score for _, score in bm25_scores)
                min_bm25_score = min(score for _, score in bm25_scores)
                bm25_range = max_bm25_score - min_bm25_score if max_bm25_score != min_bm25_score else 1.0
                
                for idx in all_indices:
                    if idx in bm25_score_map:
                        normalized = (bm25_score_map[idx] - min_bm25_score) / bm25_range
                        bm25_score_map[idx] = normalized
        
        # 混合分数
        hybrid_scores = []
        for idx in all_indices:
            vector_score = vector_score_map.get(idx, 0.0)
            bm25_score = bm25_score_map.get(idx, 0.0)
            
            # 加权组合
            final_score = self.alpha * vector_score + self.beta * bm25_score
            hybrid_scores.append((idx, final_score))
        
        # 排序并返回top_k
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        return hybrid_scores[:top_k]
    
    def search(self,
              query_text: str,
              vector_scores: List[Tuple[int, float]],
              top_k: int = 10) -> List[Tuple[int, float]]:
        """
        执行混合检索（简化接口）
        
        Args:
            query_text: 查询文本
            vector_scores: 向量检索结果
            top_k: 返回结果数量
            
        Returns:
            [(doc_idx, score), ...]
        """
        return self.hybrid_search(query_text, vector_scores, top_k)


class HierarchicalHybridRetriever:
    """层次化混合检索器：支持父子关系的混合检索"""
    
    def __init__(self,
                 alpha: float = 0.7,
                 parent_boost: float = 1.2,
                 sibling_boost: float = 1.1):
        """
        初始化层次化混合检索器
        
        Args:
            alpha: 向量检索权重
            parent_boost: 父节点分数提升倍数
            sibling_boost: 兄弟节点分数提升倍数
        """
        self.alpha = alpha
        self.parent_boost = parent_boost
        self.sibling_boost = sibling_boost
        self.hybrid_retriever = HybridRetriever(alpha=alpha)
        self.chunk_tree = {}
        self.chunk_to_doc_idx = {}  # chunk索引到文档索引的映射
    
    def index_chunks(self, chunks, chunk_tree: Dict):
        """
        索引文档块
        
        Args:
            chunks: 块列表
            chunk_tree: 块树结构
        """
        self.chunk_tree = chunk_tree
        documents = [chunk.content for chunk in chunks]
        indices = [chunk.index for chunk in chunks]
        
        # 建立映射
        self.chunk_to_doc_idx = {chunk.index: i for i, chunk in enumerate(chunks)}
        
        self.hybrid_retriever.index_documents(documents, indices)
    
    def search_with_context(self,
                           query_text: str,
                           vector_scores: List[Tuple[int, float]],
                           top_k: int = 10) -> List[Tuple[int, float, Dict]]:
        """
        带上下文的混合检索
        
        Args:
            query_text: 查询文本
            vector_scores: 向量检索结果 [(chunk_idx, score), ...]
            top_k: 返回结果数量
            
        Returns:
            [(chunk_idx, score, context_info), ...]
            其中context_info包含父节点、子节点等信息
        """
        # 执行混合检索
        hybrid_results = self.hybrid_retriever.search(query_text, vector_scores, top_k=top_k * 2)
        
        # 应用层次化提升
        enhanced_results = []
        seen_chunks = set()
        
        for chunk_idx, score in hybrid_results:
            if chunk_idx in seen_chunks:
                continue
            
            context_info = {
                'parent_id': None,
                'children_ids': [],
                'sibling_ids': [],
                'level': 0
            }
            
            # 获取块信息
            chunk = self.chunk_tree.get(chunk_idx)
            if chunk:
                context_info['parent_id'] = chunk.parent_id
                context_info['children_ids'] = chunk.children_ids
                context_info['level'] = chunk.level
                
                # 获取兄弟节点
                if chunk.parent_id:
                    parent = self.chunk_tree.get(chunk.parent_id)
                    if parent:
                        context_info['sibling_ids'] = [cid for cid in parent.children_ids if cid != chunk_idx]
                
                # 如果父节点也被检索到，提升分数
                if chunk.parent_id and chunk.parent_id in [idx for idx, _ in hybrid_results]:
                    score *= self.parent_boost
                
                # 提升子节点分数（如果匹配到父节点）
                enhanced_results.append((chunk_idx, score, context_info))
                seen_chunks.add(chunk_idx)
        
        # 重新排序
        enhanced_results.sort(key=lambda x: x[1], reverse=True)
        
        return enhanced_results[:top_k]


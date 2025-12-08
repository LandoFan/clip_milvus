"""
层次化存储：支持父子关系的存储和检索
"""
from typing import List, Dict, Optional, Tuple
import numpy as np

from milvus.milvus_store import MilvusStore
from milvus.hybrid_search import HybridRetriever, HierarchicalHybridRetriever
from file_parsers.hierarchical_parser import HierarchicalContent, Chunk


class HierarchicalMilvusStore(MilvusStore):
    """支持层次化结构的Milvus存储"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk_id_to_milvus_id = {}  # 块索引到Milvus ID的映射
        self.milvus_id_to_chunk_id = {}  # Milvus ID到块索引的映射
        self.chunk_contents = {}  # 存储块内容用于混合检索
        self.hybrid_retriever = None
    
    def insert_hierarchical_chunks(self,
                                   hierarchical_content: HierarchicalContent,
                                   embeddings: np.ndarray,
                                   file_path: str,
                                   file_type: str):
        """
        插入层次化块
        
        Args:
            hierarchical_content: 层次化内容对象
            embeddings: 块向量数组
            file_path: 文件路径
            file_type: 文件类型
        """
        if len(hierarchical_content.chunks) != len(embeddings):
            raise ValueError(f"Chunks and embeddings length mismatch: {len(hierarchical_content.chunks)} vs {len(embeddings)}")
        
        data = []
        from datetime import datetime
        import json
        current_time = datetime.now().isoformat()
        
        # 存储块内容用于混合检索
        for chunk in hierarchical_content.chunks:
            self.chunk_contents[chunk.index] = chunk.content
        
        # 准备插入数据
        for chunk, embedding in zip(hierarchical_content.chunks, embeddings):
            meta_dict = chunk.metadata.copy()
            meta_dict.update({
                'chunk_index': chunk.index,
                'children_ids': chunk.children_ids,
            })
            
            data.append({
                "content_type": "text",
                "content": chunk.content[:65535],
                "embedding": embedding.tolist(),
                "file_path": file_path[:1024],
                "file_type": file_type,
                "chunk_index": chunk.index,
                "parent_id": chunk.parent_id if chunk.parent_id is not None else -1,
                "chunk_type": chunk.chunk_type.value,
                "level": chunk.level,
                "metadata": json.dumps(meta_dict),
                "created_at": current_time
            })
        
        # 插入数据
        insert_result = self.collection.insert(data)
        self.collection.flush()
        
        # 建立映射关系
        for i, chunk in enumerate(hierarchical_content.chunks):
            milvus_id = insert_result.primary_keys[i]
            self.chunk_id_to_milvus_id[chunk.index] = milvus_id
            self.milvus_id_to_chunk_id[milvus_id] = chunk.index
        
        # 更新混合检索器（增量添加，而不是替换）
        documents = [chunk.content for chunk in hierarchical_content.chunks]
        indices = [chunk.index for chunk in hierarchical_content.chunks]
        
        if self.hybrid_retriever is None:
            # 首次创建混合检索器
            self.hybrid_retriever = HybridRetriever(alpha=0.7)
            # 需要从Milvus加载所有现有文档来构建完整的索引
            self._rebuild_hybrid_retriever()
        else:
            # 增量添加新文档
            existing_docs = self.hybrid_retriever.documents
            existing_indices = self.hybrid_retriever.doc_indices
            
            # 合并新文档
            all_documents = existing_docs + documents
            all_indices = existing_indices + indices
            
            # 重新索引
            self.hybrid_retriever.index_documents(all_documents, all_indices)
        
        print(f"✓ Inserted {len(hierarchical_content.chunks)} hierarchical chunks")
    
    def hybrid_search(self,
                     query_text: str,
                     query_vector: np.ndarray,
                     content_type: Optional[str] = None,
                     limit: int = 10,
                     alpha: float = 0.7,
                     expr: Optional[str] = None) -> List[Dict]:
        """
        混合检索：结合向量检索和关键词检索
        
        Args:
            query_text: 查询文本
            query_vector: 查询向量
            content_type: 内容类型筛选
            limit: 返回结果数量
            alpha: 向量检索权重 (0-1)
            expr: 额外的过滤表达式
            
        Returns:
            搜索结果列表
        """
        # 确保集合已加载
        if not self.collection.has_index():
            raise RuntimeError("Collection does not have an index. Please create index first.")
        
        self.collection.load()
        
        # 1. 向量检索
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        
        filter_expr = None
        if content_type:
            filter_expr = f'content_type == "{content_type}"'
        
        if expr:
            filter_expr = f"{filter_expr} && ({expr})" if filter_expr else expr
        
        # 向量搜索，返回更多结果用于混合检索
        vector_results = self.collection.search(
            data=[query_vector.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=limit * 3,  # 获取更多候选结果
            expr=filter_expr,
            output_fields=["chunk_index", "content", "parent_id", "chunk_type", "level", "metadata"]
        )
        
        # 准备向量检索结果
        vector_scores = []
        for hits in vector_results:
            for hit in hits:
                chunk_idx = hit.entity.get('chunk_index')
                if chunk_idx is not None:
                    vector_scores.append((int(chunk_idx), float(hit.distance)))
        
        # 2. 混合检索
        if self.hybrid_retriever and len(vector_scores) > 0:
            # 确保混合检索器已初始化（从Milvus重建索引）
            if not self.hybrid_retriever.documents:
                self._rebuild_hybrid_retriever()
            
            # 使用混合检索器
            self.hybrid_retriever.alpha = alpha
            hybrid_results = self.hybrid_retriever.search(query_text, vector_scores, top_k=limit)
            
            # 创建chunk_index到hit的映射
            chunk_idx_to_hit = {}
            for hits in vector_results:
                for hit in hits:
                    chunk_idx = hit.entity.get('chunk_index')
                    if chunk_idx is not None:
                        chunk_idx_to_hit[int(chunk_idx)] = hit
            
            # 获取详细结果
            detailed_results = []
            for chunk_idx, score in hybrid_results:
                # 从向量检索结果中获取详细信息
                hit_info = chunk_idx_to_hit.get(chunk_idx)
                if not hit_info:
                    continue
                
                import json
                detailed_results.append({
                    'id': hit_info.id,
                    'distance': score,  # 使用混合检索分数
                    'content_type': 'text',
                    'content': hit_info.entity.get('content'),
                    'chunk_index': chunk_idx,
                    'parent_id': hit_info.entity.get('parent_id'),
                    'chunk_type': hit_info.entity.get('chunk_type'),
                    'level': hit_info.entity.get('level'),
                    'file_path': hit_info.entity.get('file_path', ''),
                    'file_type': hit_info.entity.get('file_type', ''),
                    'metadata': json.loads(hit_info.entity.get('metadata', '{}'))
                })
            
            return detailed_results
        else:
            # 回退到纯向量检索
            return self._format_vector_results(vector_results, limit)
    
    def hierarchical_search(self,
                           query_text: str,
                           query_vector: np.ndarray,
                           content_type: Optional[str] = None,
                           limit: int = 10,
                           alpha: float = 0.7,
                           include_children: bool = True,
                           include_parent: bool = True) -> List[Dict]:
        """
        层次化混合检索：考虑父子关系
        
        Args:
            query_text: 查询文本
            query_vector: 查询向量
            content_type: 内容类型
            limit: 返回结果数量
            alpha: 向量检索权重
            include_children: 是否包含子节点
            include_parent: 是否包含父节点
            
        Returns:
            搜索结果列表（包含上下文信息）
        """
        # 先执行混合检索
        results = self.hybrid_search(
            query_text=query_text,
            query_vector=query_vector,
            content_type=content_type,
            limit=limit * 2,  # 获取更多结果用于扩展
            alpha=alpha
        )
        
        # 扩展结果：添加父节点和子节点
        expanded_results = []
        seen_ids = set()
        
        for result in results:
            chunk_idx = result.get('chunk_index')
            if chunk_idx in seen_ids:
                continue
            
            # 添加当前结果
            expanded_results.append(result)
            seen_ids.add(chunk_idx)
            
            # 添加父节点
            if include_parent:
                parent_id = result.get('parent_id')
                if parent_id and parent_id != -1:
                    parent_result = self._get_chunk_by_index(parent_id)
                    if parent_result and parent_id not in seen_ids:
                        expanded_results.append(parent_result)
                        seen_ids.add(parent_id)
            
            # 添加子节点
            if include_children:
                children_ids = result.get('metadata', {}).get('children_ids', [])
                for child_id in children_ids:
                    if child_id not in seen_ids:
                        child_result = self._get_chunk_by_index(child_id)
                        if child_result:
                            expanded_results.append(child_result)
                            seen_ids.add(child_id)
        
        # 重新排序（按相关度）
        expanded_results.sort(key=lambda x: x.get('distance', float('inf')))
        
        return expanded_results[:limit]
    
    def _get_chunk_by_index(self, chunk_index: int) -> Optional[Dict]:
        """根据块索引获取块信息"""
        # 这里需要从Milvus中查询
        # 简化实现：从已有的映射中查找
        milvus_id = self.chunk_id_to_milvus_id.get(chunk_index)
        if milvus_id is None:
            return None
        
        # 实际应该从Milvus查询，这里简化处理
        return None
    
    def _rebuild_hybrid_retriever(self):
        """从Milvus重建混合检索器索引"""
        try:
            if not self.collection.has_index():
                # 如果集合还没有索引，创建空的检索器
                if self.hybrid_retriever is None:
                    self.hybrid_retriever = HybridRetriever(alpha=0.7)
                    self.hybrid_retriever.index_documents([], [])
                return
            
            self.collection.load()
            
            # 查询所有文本块（Milvus limit 最大 16384）
            results = self.collection.query(
                expr="content_type == 'text'",
                output_fields=["chunk_index", "content"],
                limit=16384
            )
            
            if not results:
                # 如果没有现有数据，创建空的检索器
                if self.hybrid_retriever is None:
                    self.hybrid_retriever = HybridRetriever(alpha=0.7)
                self.hybrid_retriever.index_documents([], [])
                return
            
            # 重建索引
            documents = [r['content'] for r in results]
            indices = [r['chunk_index'] for r in results]
            
            if self.hybrid_retriever is None:
                self.hybrid_retriever = HybridRetriever(alpha=0.7)
            
            self.hybrid_retriever.index_documents(documents, indices)
            print(f"✓ 重建混合检索索引，共 {len(documents)} 个文档块")
        
        except Exception as e:
            print(f"Warning: 重建混合检索索引失败: {e}")
            # 创建空的检索器作为后备
            if self.hybrid_retriever is None:
                self.hybrid_retriever = HybridRetriever(alpha=0.7)
                self.hybrid_retriever.index_documents([], [])
    
    def _format_vector_results(self, vector_results, limit: int) -> List[Dict]:
        """格式化向量检索结果"""
        import json
        formatted_results = []
        for hits in vector_results:
            for hit in hits:
                formatted_results.append({
                    'id': hit.id,
                    'distance': hit.distance,
                    'content_type': 'text',
                    'content': hit.entity.get('content'),
                    'chunk_index': hit.entity.get('chunk_index'),
                    'parent_id': hit.entity.get('parent_id'),
                    'chunk_type': hit.entity.get('chunk_type'),
                    'level': hit.entity.get('level'),
                    'metadata': json.loads(hit.entity.get('metadata', '{}'))
                })
                if len(formatted_results) >= limit:
                    break
            if len(formatted_results) >= limit:
                break
        return formatted_results


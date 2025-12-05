"""
Milvus知识库API：统一的接口封装，便于调用
"""
import os
import sys
from typing import List, Dict, Optional, Union
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# 确保父目录在导入路径中
_parent_dir = Path(__file__).parent.parent.absolute()
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from file_parsers.hierarchical_parser import (
    HierarchicalWordParser,
    HierarchicalMarkdownParser,
    HierarchicalContent
)
from clip.vectorizer import CLIPVectorizer
from milvus.hierarchical_store import HierarchicalMilvusStore


class KnowledgeBase:
    """
    Milvus知识库统一API
    
    提供便捷的接口用于：
    - 添加文档到知识库
    - 查询知识库
    - 管理知识库
    """
    
    def __init__(self,
                 clip_server: str = None,
                 milvus_host: str = None,
                 milvus_port: int = None,
                 collection_name: str = None,
                 max_chunk_size: int = 500,
                 auto_reconnect: bool = True):
        """
        初始化知识库
        
        Args:
            clip_server: CLIP服务器地址，如 'grpc://0.0.0.0:51000'
            milvus_host: Milvus服务器地址
            milvus_port: Milvus服务器端口
            collection_name: 集合名称
            max_chunk_size: 最大块大小（字符数）
            auto_reconnect: 是否自动重连
        """
        # 从环境变量获取默认值
        self.clip_server = clip_server or os.getenv('CLIP_SERVER', 'grpc://0.0.0.0:51000')
        self.milvus_host = milvus_host or os.getenv('MILVUS_HOST', 'localhost')
        self.milvus_port = milvus_port or int(os.getenv('MILVUS_PORT', '19530'))
        self.collection_name = collection_name or os.getenv('MILVUS_COLLECTION', 'clip_documents_hierarchical')
        self.max_chunk_size = max_chunk_size
        self.auto_reconnect = auto_reconnect
        
        # 初始化组件
        print("正在初始化知识库...")
        self._init_components()
        print("✓ 知识库初始化完成")
    
    def _init_components(self):
        """初始化所有组件"""
        # 初始化CLIP向量化器
        self.vectorizer = CLIPVectorizer(server_url=self.clip_server)
        self.embedding_dim = self.vectorizer.get_embedding_dimension()
        
        # 初始化Milvus存储
        self.store = HierarchicalMilvusStore(
            host=self.milvus_host,
            port=self.milvus_port,
            collection_name=self.collection_name,
            embedding_dim=self.embedding_dim,
            drop_existing=False
        )
        
        # 初始化解析器（延迟创建）
        self._word_parser = None
        self._md_parser = None
    
    def add_document(self,
                    file_path: Union[str, Path],
                    file_type: Optional[str] = None) -> Dict:
        """
        添加文档到知识库
        
        Args:
            file_path: 文件路径
            file_type: 文件类型 ('word' 或 'markdown')，如果为None则自动检测
            
        Returns:
            处理结果字典，包含：
            - success: 是否成功
            - chunks_count: 块数量
            - file_path: 文件路径
            - message: 消息
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {
                'success': False,
                'message': f'文件不存在: {file_path}'
            }
        
        # 自动检测文件类型
        if file_type is None:
            ext = file_path.suffix.lower()
            if ext == '.docx':
                file_type = 'word'
            elif ext in ['.md', '.markdown']:
                file_type = 'markdown'
            else:
                return {
                    'success': False,
                    'message': f'不支持的文件类型: {ext}'
                }
        
        try:
            # 解析文件
            if file_type == 'word':
                if self._word_parser is None:
                    self._word_parser = HierarchicalWordParser(max_chunk_size=self.max_chunk_size)
                hierarchical_content = self._word_parser.parse(str(file_path))
            
            elif file_type == 'markdown':
                if self._md_parser is None:
                    self._md_parser = HierarchicalMarkdownParser(max_chunk_size=self.max_chunk_size)
                hierarchical_content = self._md_parser.parse(str(file_path))
            
            else:
                return {
                    'success': False,
                    'message': f'不支持的文件类型: {file_type}'
                }
            
            # 向量化
            texts = [chunk.content for chunk in hierarchical_content.chunks]
            if not texts:
                return {
                    'success': False,
                    'message': '文件中没有提取到内容'
                }
            
            embeddings = self.vectorizer.encode_texts(texts, show_progress=False)
            
            # 存储到Milvus
            self.store.insert_hierarchical_chunks(
                hierarchical_content=hierarchical_content,
                embeddings=embeddings,
                file_path=str(file_path),
                file_type=file_type
            )
            
            return {
                'success': True,
                'chunks_count': len(hierarchical_content.chunks),
                'file_path': str(file_path),
                'file_type': file_type,
                'message': f'成功添加 {len(hierarchical_content.chunks)} 个块'
            }
        
        except Exception as e:
            return {
                'success': False,
                'message': f'处理文件时出错: {str(e)}',
                'error': str(e)
            }
    
    def add_documents(self,
                     file_paths: List[Union[str, Path]],
                     show_progress: bool = True) -> List[Dict]:
        """
        批量添加文档
        
        Args:
            file_paths: 文件路径列表
            show_progress: 是否显示进度
            
        Returns:
            处理结果列表
        """
        from tqdm import tqdm
        
        results = []
        iterator = tqdm(file_paths, desc="处理文档") if show_progress else file_paths
        
        for file_path in iterator:
            result = self.add_document(file_path)
            results.append(result)
        
        return results
    
    def query(self,
             query_text: str,
             top_k: int = 10,
             alpha: float = 0.7,
             hierarchical: bool = True,
             include_children: bool = True,
             include_parent: bool = True,
             content_type: Optional[str] = None,
             filter_expr: Optional[str] = None) -> List[Dict]:
        """
        查询知识库
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数量
            alpha: 混合检索权重（0-1），alpha越高向量检索权重越高
            hierarchical: 是否使用层次化检索
            include_children: 是否包含子块
            include_parent: 是否包含父块
            content_type: 内容类型筛选 ('text' 或 'image')
            filter_expr: 额外的过滤表达式（Milvus表达式）
            
        Returns:
            搜索结果列表，每个结果包含：
            - content: 内容文本
            - distance: 相似度分数（越小越好）
            - chunk_type: 块类型
            - level: 层级
            - parent_id: 父块ID
            - file_path: 文件路径
            - metadata: 元数据
        """
        try:
            # 向量化查询文本
            query_vector = self.vectorizer.encode_texts([query_text], show_progress=False)[0]
            
            # 执行检索
            if hierarchical:
                results = self.store.hierarchical_search(
                    query_text=query_text,
                    query_vector=query_vector,
                    content_type=content_type,
                    limit=top_k,
                    alpha=alpha,
                    include_children=include_children,
                    include_parent=include_parent
                )
            else:
                results = self.store.hybrid_search(
                    query_text=query_text,
                    query_vector=query_vector,
                    content_type=content_type,
                    limit=top_k,
                    alpha=alpha,
                    expr=filter_expr
                )
            
            # 格式化结果
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'content': result.get('content', ''),
                    'distance': result.get('distance', float('inf')),
                    'chunk_type': result.get('chunk_type', 'unknown'),
                    'level': result.get('level', 0),
                    'parent_id': result.get('parent_id'),
                    'file_path': result.get('file_path', ''),
                    'chunk_index': result.get('chunk_index'),
                    'metadata': result.get('metadata', {})
                })
            
            return formatted_results
        
        except Exception as e:
            print(f"查询时出错: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def query_batch(self,
                   queries: List[str],
                   top_k: int = 10,
                   alpha: float = 0.7,
                   hierarchical: bool = True) -> List[List[Dict]]:
        """
        批量查询
        
        Args:
            queries: 查询文本列表
            top_k: 每个查询返回结果数量
            alpha: 混合检索权重
            hierarchical: 是否使用层次化检索
            
        Returns:
            每个查询的结果列表
        """
        results = []
        for query in queries:
            result = self.query(
                query_text=query,
                top_k=top_k,
                alpha=alpha,
                hierarchical=hierarchical
            )
            results.append(result)
        return results
    
    def get_context(self, chunk_index: int, include_siblings: bool = False) -> Dict:
        """
        获取块的上下文信息（父块、子块、兄弟块）
        
        Args:
            chunk_index: 块索引
            include_siblings: 是否包含兄弟块
            
        Returns:
            上下文信息字典
        """
        # 从Milvus查询块信息
        # 这里简化实现，实际应该从数据库查询
        return {
            'chunk_index': chunk_index,
            'parent': None,
            'children': [],
            'siblings': []
        }
    
    def get_stats(self) -> Dict:
        """
        获取知识库统计信息
        
        Returns:
            统计信息字典
        """
        stats = self.store.get_stats()
        stats.update({
            'clip_server': self.clip_server,
            'milvus_host': self.milvus_host,
            'milvus_port': self.milvus_port,
            'collection_name': self.collection_name
        })
        return stats
    
    def delete_document(self, file_path: str) -> Dict:
        """
        从知识库删除文档的所有块
        
        Args:
            file_path: 文件路径
            
        Returns:
            删除结果
        """
        try:
            # 构建删除表达式
            expr = f'file_path == "{file_path}"'
            
            # Milvus的删除操作
            # 注意：Milvus的删除需要先查询ID，然后删除
            self.store.collection.load()
            
            # 查询要删除的ID
            results = self.store.collection.query(
                expr=expr,
                output_fields=["id"]
            )
            
            if not results:
                return {
                    'success': False,
                    'message': '未找到该文件的数据'
                }
            
            ids_to_delete = [r['id'] for r in results]
            
            # 执行删除
            self.store.collection.delete(expr=expr)
            self.store.collection.flush()
            
            return {
                'success': True,
                'deleted_count': len(ids_to_delete),
                'message': f'成功删除 {len(ids_to_delete)} 条记录'
            }
        
        except Exception as e:
            return {
                'success': False,
                'message': f'删除时出错: {str(e)}',
                'error': str(e)
            }
    
    def list_documents(self) -> List[str]:
        """
        列出知识库中的所有文档路径
        
        Returns:
            文档路径列表
        """
        try:
            self.store.collection.load()
            
            # 查询所有唯一的文件路径
            results = self.store.collection.query(
                expr="",  # 查询所有
                output_fields=["file_path"],
                limit=10000  # 设置一个合理的上限
            )
            
            unique_paths = list(set(r['file_path'] for r in results))
            return unique_paths
        
        except Exception as e:
            print(f"列出文档时出错: {e}")
            return []
    
    def rebuild_hybrid_index(self):
        """
        重建混合检索索引（BM25）
        
        注意：这会重新索引所有文档，可能需要一些时间
        """
        try:
            self.store.collection.load()
            
            # 查询所有块
            results = self.store.collection.query(
                expr="content_type == 'text'",
                output_fields=["chunk_index", "content"],
                limit=100000
            )
            
            if not results:
                print("没有找到文档，无法重建索引")
                return
            
            # 重建BM25索引
            documents = [r['content'] for r in results]
            indices = [r['chunk_index'] for r in results]
            
            from hybrid_search import HybridRetriever
            self.store.hybrid_retriever = HybridRetriever(alpha=0.7)
            self.store.hybrid_retriever.index_documents(documents, indices)
            
            print(f"✓ 成功重建混合检索索引，共 {len(documents)} 个文档")
        
        except Exception as e:
            print(f"重建索引时出错: {e}")
            import traceback
            traceback.print_exc()


# 便捷函数
def create_knowledge_base(
    clip_server: str = None,
    milvus_host: str = None,
    milvus_port: int = None,
    collection_name: str = None,
    **kwargs
) -> KnowledgeBase:
    """
    创建知识库实例的便捷函数
    
    Args:
        clip_server: CLIP服务器地址
        milvus_host: Milvus服务器地址
        milvus_port: Milvus服务器端口
        collection_name: 集合名称
        **kwargs: 其他参数
        
    Returns:
        KnowledgeBase实例
    """
    return KnowledgeBase(
        clip_server=clip_server,
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        collection_name=collection_name,
        **kwargs
    )


"""
Milvus数据库存储：将向量存储到Milvus
"""
from typing import List, Dict, Optional
import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from datetime import datetime
import json


class MilvusStore:
    """Milvus向量数据库存储"""
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 19530,
                 collection_name: str = "clip_documents",
                 embedding_dim: int = 512,
                 drop_existing: bool = False):
        """
        初始化Milvus存储
        
        Args:
            host: Milvus服务器地址
            port: Milvus服务器端口
            collection_name: 集合名称
            embedding_dim: 向量维度
            drop_existing: 如果集合已存在，是否删除重建
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        
        # 连接到Milvus
        connections.connect(
            alias="default",
            host=host,
            port=port
        )
        print(f"✓ Connected to Milvus: {host}:{port}")
        
        # 创建或获取集合
        self._setup_collection(drop_existing)
    
    def _setup_collection(self, drop_existing: bool = False):
        """设置Milvus集合"""
        
        # 检查集合是否存在
        if utility.has_collection(self.collection_name):
            if drop_existing:
                utility.drop_collection(self.collection_name)
                print(f"✓ Dropped existing collection: {self.collection_name}")
            else:
                self.collection = Collection(self.collection_name)
                print(f"✓ Using existing collection: {self.collection_name}")
                return
        
        # 定义字段
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="content_type", dtype=DataType.VARCHAR, max_length=20),  # 'text' or 'image'
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),  # 原始内容
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="file_type", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),  # 文本块索引或图像索引
            FieldSchema(name="parent_id", dtype=DataType.INT64, default_value=-1),  # 父块ID（用于层次化结构，-1表示无父节点）
            FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=20),  # 块类型（document/section/paragraph等）
            FieldSchema(name="level", dtype=DataType.INT64),  # 层级深度
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),  # JSON格式的额外元数据
            FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=50),
        ]
        
        # 创建集合schema
        schema = CollectionSchema(
            fields=fields,
            description="CLIP embeddings for documents (text and images)"
        )
        
        # 创建集合
        self.collection = Collection(
            name=self.collection_name,
            schema=schema
        )
        
        # 创建索引
        index_params = {
            "metric_type": "L2",  # 使用L2距离
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        
        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        
        print(f"✓ Created collection: {self.collection_name}")
        print(f"✓ Created index on embedding field")
    
    def insert_texts(self,
                     texts: List[str],
                     embeddings: np.ndarray,
                     file_path: str,
                     file_type: str,
                     metadata: Optional[Dict] = None):
        """
        插入文本向量
        
        Args:
            texts: 文本列表
            embeddings: 文本向量数组
            file_path: 文件路径
            file_type: 文件类型
            metadata: 额外元数据
        """
        if len(texts) != len(embeddings):
            raise ValueError(f"Texts and embeddings length mismatch: {len(texts)} vs {len(embeddings)}")
        
        data = []
        current_time = datetime.now().isoformat()
        
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            meta_dict = metadata or {}
            meta_dict['chunk_index'] = i
            
            data.append({
                "content_type": "text",
                "content": text[:65535],  # 截断过长的文本
                "embedding": embedding.tolist(),
                "file_path": file_path[:1024],
                "file_type": file_type,
                "chunk_index": i,
                "parent_id": meta_dict.get('parent_id', -1),
                "chunk_type": meta_dict.get('chunk_type', 'paragraph'),
                "level": meta_dict.get('level', 0),
                "metadata": json.dumps(meta_dict),
                "created_at": current_time
            })
        
        # 插入数据
        self.collection.insert(data)
        self.collection.flush()
        
        print(f"✓ Inserted {len(texts)} text embeddings")
    
    def insert_images(self,
                      image_paths: List[str],
                      embeddings: np.ndarray,
                      file_path: str,
                      file_type: str,
                      metadata: Optional[Dict] = None):
        """
        插入图像向量
        
        Args:
            image_paths: 图像路径列表
            embeddings: 图像向量数组
            file_path: 源文件路径
            file_type: 文件类型
            metadata: 额外元数据
        """
        if len(image_paths) != len(embeddings):
            raise ValueError(f"Image paths and embeddings length mismatch: {len(image_paths)} vs {len(embeddings)}")
        
        data = []
        current_time = datetime.now().isoformat()
        
        for i, (img_path, embedding) in enumerate(zip(image_paths, embeddings)):
            meta_dict = metadata or {}
            meta_dict['image_index'] = i
            meta_dict['image_path'] = img_path
            
            data.append({
                "content_type": "image",
                "content": img_path,  # 存储图像路径
                "embedding": embedding.tolist(),
                "file_path": file_path[:1024],
                "file_type": file_type,
                "chunk_index": i,
                "metadata": json.dumps(meta_dict),
                "created_at": current_time
            })
        
        # 插入数据
        self.collection.insert(data)
        self.collection.flush()
        
        print(f"✓ Inserted {len(image_paths)} image embeddings")
    
    def search(self,
               query_vectors: np.ndarray,
               content_type: Optional[str] = None,
               limit: int = 10,
               expr: Optional[str] = None) -> List[Dict]:
        """
        搜索相似向量
        
        Args:
            query_vectors: 查询向量，可以是单个向量或向量数组
            content_type: 筛选内容类型 ('text' 或 'image')
            limit: 返回结果数量
            expr: 额外的过滤表达式
            
        Returns:
            搜索结果列表
        """
        # 确保集合已加载
        if not self.collection.has_index():
            raise RuntimeError("Collection does not have an index. Please create index first.")
        
        self.collection.load()
        
        # 构建过滤表达式
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        
        filter_expr = None
        if content_type:
            filter_expr = f'content_type == "{content_type}"'
        
        if expr:
            filter_expr = f"{filter_expr} && ({expr})" if filter_expr else expr
        
        # 搜索
        results = self.collection.search(
            data=query_vectors.tolist() if isinstance(query_vectors, np.ndarray) else [query_vectors.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            expr=filter_expr,
            output_fields=["content_type", "content", "file_path", "file_type", "metadata"]
        )
        
        # 格式化结果
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    'id': hit.id,
                    'distance': hit.distance,
                    'content_type': hit.entity.get('content_type'),
                    'content': hit.entity.get('content'),
                    'file_path': hit.entity.get('file_path'),
                    'file_type': hit.entity.get('file_type'),
                    'metadata': json.loads(hit.entity.get('metadata', '{}'))
                })
        
        return formatted_results
    
    def get_stats(self) -> Dict:
        """获取集合统计信息"""
        stats = {
            'collection_name': self.collection_name,
            'num_entities': self.collection.num_entities,
            'has_index': self.collection.has_index()
        }
        
        return stats


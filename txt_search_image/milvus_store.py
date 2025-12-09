"""
Milvus 向量数据库存储模块
用于存储和检索图像向量
"""
import os
from typing import List, Dict, Optional, Any
import numpy as np

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)


class MilvusStore:
    """
    Milvus 向量数据库操作类
    
    使用示例:
    ```python
    store = MilvusStore(collection_name='image_collection', dim=512)
    
    # 插入向量
    store.insert(
        vectors=embeddings,
        image_paths=['path1.jpg', 'path2.jpg'],
        descriptions=['cat', 'dog']
    )
    
    # 搜索
    results = store.search(query_vector, top_k=5)
    ```
    """
    
    def __init__(
        self,
        collection_name: str = 'clip_images',
        dim: int = 512,
        host: str = 'localhost',
        port: str = '19530',
        index_type: str = 'IVF_FLAT',
        metric_type: str = 'IP',  # IP: 内积 (用于归一化向量相当于余弦相似度)
        nlist: int = 1024,
    ):
        """
        初始化 Milvus 存储
        
        Args:
            collection_name: 集合名称
            dim: 向量维度
            host: Milvus 服务器地址
            port: Milvus 服务器端口
            index_type: 索引类型，可选 'IVF_FLAT', 'IVF_SQ8', 'HNSW' 等
            metric_type: 距离度量类型，'IP' (内积) 或 'L2' (欧氏距离)
            nlist: IVF 索引的聚类数
        """
        self.collection_name = collection_name
        self.dim = dim
        self.host = host
        self.port = port
        self.index_type = index_type
        self.metric_type = metric_type
        self.nlist = nlist
        
        self._collection: Optional[Collection] = None
        self._connect()
    
    def _connect(self):
        """连接到 Milvus 服务器"""
        connections.connect(
            alias='default',
            host=self.host,
            port=self.port,
        )
        print(f"已连接到 Milvus 服务器: {self.host}:{self.port}")
    
    def _create_collection(self):
        """创建集合"""
        # 定义字段
        fields = [
            FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name='image_path', dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name='description', dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=self.dim),
        ]
        
        # 创建 Schema
        schema = CollectionSchema(
            fields=fields,
            description='CLIP image embeddings collection'
        )
        
        # 创建集合
        self._collection = Collection(
            name=self.collection_name,
            schema=schema,
        )
        
        print(f"已创建集合: {self.collection_name}")
        
        # 创建索引
        self._create_index()
    
    def _create_index(self):
        """创建向量索引"""
        if self._collection is None:
            return
        
        index_params = {
            'index_type': self.index_type,
            'metric_type': self.metric_type,
            'params': {'nlist': self.nlist}
        }
        
        if self.index_type == 'HNSW':
            index_params['params'] = {'M': 16, 'efConstruction': 256}
        
        self._collection.create_index(
            field_name='embedding',
            index_params=index_params
        )
        
        print(f"已创建索引: {self.index_type}")
    
    def get_or_create_collection(self) -> Collection:
        """获取或创建集合"""
        if utility.has_collection(self.collection_name):
            self._collection = Collection(self.collection_name)
            print(f"已加载现有集合: {self.collection_name}")
        else:
            self._create_collection()
        
        return self._collection
    
    def drop_collection(self):
        """删除集合"""
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            self._collection = None
            print(f"已删除集合: {self.collection_name}")
    
    def insert(
        self,
        vectors: np.ndarray,
        image_paths: List[str],
        descriptions: Optional[List[str]] = None,
    ) -> List[int]:
        """
        插入向量数据
        
        Args:
            vectors: 向量数组，形状为 (N, dim)
            image_paths: 图像路径列表
            descriptions: 图像描述列表，可选
            
        Returns:
            插入数据的主键 ID 列表
        """
        if self._collection is None:
            self.get_or_create_collection()
        
        if descriptions is None:
            descriptions = [''] * len(image_paths)
        
        assert len(vectors) == len(image_paths) == len(descriptions), \
            "向量、路径和描述的数量必须相同"
        
        # 准备数据
        data = [
            image_paths,
            descriptions,
            vectors.tolist(),
        ]
        
        # 插入数据
        result = self._collection.insert(data)
        
        # 刷新以确保数据持久化
        self._collection.flush()
        
        print(f"已插入 {len(vectors)} 条数据")
        return result.primary_keys
    
    def search(
        self,
        query_vectors: np.ndarray,
        top_k: int = 10,
        nprobe: int = 16,
        output_fields: Optional[List[str]] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        搜索相似向量
        
        Args:
            query_vectors: 查询向量，形状为 (N, dim) 或 (dim,)
            top_k: 返回结果数量
            nprobe: 搜索时探测的聚类数
            output_fields: 需要返回的字段列表
            
        Returns:
            搜索结果列表，每个元素是一个查询的结果列表
        """
        if self._collection is None:
            self.get_or_create_collection()
        
        # 加载集合到内存
        self._collection.load()
        
        # 确保是 2D 数组
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)
        
        if output_fields is None:
            output_fields = ['image_path', 'description']
        
        search_params = {
            'metric_type': self.metric_type,
            'params': {'nprobe': nprobe}
        }
        
        if self.index_type == 'HNSW':
            search_params['params'] = {'ef': 64}
        
        # 执行搜索
        results = self._collection.search(
            data=query_vectors.tolist(),
            anns_field='embedding',
            param=search_params,
            limit=top_k,
            output_fields=output_fields,
        )
        
        # 格式化结果
        formatted_results = []
        for hits in results:
            query_results = []
            for hit in hits:
                result = {
                    'id': hit.id,
                    'distance': hit.distance,
                    'score': hit.distance,  # IP 距离即为相似度分数
                }
                for field in output_fields:
                    result[field] = hit.entity.get(field)
                query_results.append(result)
            formatted_results.append(query_results)
        
        return formatted_results
    
    def count(self) -> int:
        """获取集合中的数据数量"""
        if self._collection is None:
            self.get_or_create_collection()
        return self._collection.num_entities
    
    def release(self):
        """释放集合内存"""
        if self._collection is not None:
            self._collection.release()
    
    def close(self):
        """关闭连接"""
        self.release()
        connections.disconnect('default')
        print("已断开 Milvus 连接")


if __name__ == '__main__':
    # 测试代码
    import numpy as np
    
    store = MilvusStore(collection_name='test_collection', dim=512)
    
    # 测试创建集合
    store.get_or_create_collection()
    
    # 测试插入
    test_vectors = np.random.randn(5, 512).astype(np.float32)
    test_vectors = test_vectors / np.linalg.norm(test_vectors, axis=-1, keepdims=True)
    
    ids = store.insert(
        vectors=test_vectors,
        image_paths=[f'image_{i}.jpg' for i in range(5)],
        descriptions=[f'description_{i}' for i in range(5)]
    )
    print(f"插入的 ID: {ids}")
    
    # 测试搜索
    query = test_vectors[0:1]
    results = store.search(query, top_k=3)
    print(f"搜索结果: {results}")
    
    # 清理
    store.drop_collection()
    store.close()


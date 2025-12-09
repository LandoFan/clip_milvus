"""
文本搜图主模块
提供完整的图像索引和文本搜索功能

支持两种编码模式:
1. service 模式: 连接 clip-as-service 服务器 (推荐生产环境使用)
2. local 模式: 本地直接加载模型 (适合单机开发测试)
"""
import os
import glob
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np
from PIL import Image

from milvus_store import MilvusStore


@dataclass
class SearchResult:
    """搜索结果数据类"""
    image_path: str
    description: str
    score: float
    rank: int


class ImageSearchEngine:
    """
    文本搜图引擎
    
    功能:
    1. 将图像集编码为向量并存入 Milvus 数据库
    2. 输入文本，搜索并返回相似的图像
    
    使用示例:
    ```python
    # 方式一: 使用 clip-as-service 服务 (推荐)
    engine = ImageSearchEngine(
        mode='service',
        clip_server='grpc://localhost:51000'
    )
    
    # 方式二: 本地加载模型
    engine = ImageSearchEngine(mode='local')
    
    # 索引图像目录
    engine.index_directory('path/to/images')
    
    # 文本搜图
    results = engine.search('a cute cat', top_k=5)
    ```
    """
    
    def __init__(
        self,
        collection_name: str = 'clip_images',
        mode: str = 'service',
        # clip-as-service 参数
        clip_server: str = 'grpc://localhost:51000',
        # 本地模式参数
        model_name: str = 'ViT-B-32::openai',
        device: Optional[str] = None,
        # Milvus 参数
        milvus_host: str = 'localhost',
        milvus_port: str = '19530',
    ):
        """
        初始化搜图引擎
        
        Args:
            collection_name: Milvus 集合名称
            mode: 编码模式，'service' 使用 clip-as-service，'local' 本地加载模型
            clip_server: clip-as-service 服务器地址 (mode='service' 时使用)
            model_name: CLIP 模型名称 (mode='local' 时使用)
            device: 运行设备 (mode='local' 时使用)
            milvus_host: Milvus 服务器地址
            milvus_port: Milvus 服务器端口
        """
        print("=" * 50)
        print("初始化文本搜图引擎")
        print("=" * 50)
        
        self._mode = mode
        self._encoder = None
        self._clip_client = None
        
        if mode == 'service':
            # 使用 clip-as-service 客户端
            self._init_service_mode(clip_server)
        elif mode == 'local':
            # 本地加载模型
            self._init_local_mode(model_name, device)
        else:
            raise ValueError(f"不支持的模式: {mode}，请使用 'service' 或 'local'")
        
        # 获取向量维度
        self._dim = self._get_embedding_dim()
        print(f"向量维度: {self._dim}")
        
        # 初始化 Milvus 存储
        self.store = MilvusStore(
            collection_name=collection_name,
            dim=self._dim,
            host=milvus_host,
            port=milvus_port,
        )
        self.store.get_or_create_collection()
        
        print("=" * 50)
        print("引擎初始化完成!")
        print("=" * 50)
    
    def _init_service_mode(self, clip_server: str):
        """初始化 clip-as-service 客户端模式"""
        from clip_client import Client
        
        print(f"连接 clip-as-service 服务器: {clip_server}")
        self._clip_client = Client(clip_server)
        print("已连接到 clip-as-service 服务器")
    
    def _init_local_mode(self, model_name: str, device: Optional[str]):
        """初始化本地模型模式"""
        from clip_encoder import CLIPEncoder
        
        self._encoder = CLIPEncoder(model_name=model_name, device=device)
    
    def _get_embedding_dim(self) -> int:
        """获取嵌入向量维度"""
        if self._mode == 'service':
            # 编码一个测试文本来获取维度
            embeddings = self._clip_client.encode(['test'])
            return embeddings.shape[-1]
        else:
            return self._encoder.embedding_dim
    
    def _encode_images(self, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
        """编码图像列表"""
        if self._mode == 'service':
            # 使用 clip-as-service 客户端
            embeddings = self._clip_client.encode(
                image_paths,
                batch_size=batch_size,
                show_progress=True
            )
            # 归一化
            embeddings = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)
            return embeddings.astype(np.float32)
        else:
            return self._encoder.encode_images(image_paths, batch_size=batch_size, normalize=True)
    
    def _encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """编码文本列表"""
        if self._mode == 'service':
            embeddings = self._clip_client.encode(
                texts,
                batch_size=batch_size,
            )
            embeddings = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)
            return embeddings.astype(np.float32)
        else:
            return self._encoder.encode_texts(texts, batch_size=batch_size, normalize=True)
    
    def _get_image_files(
        self,
        directory: str,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'),
        recursive: bool = True,
    ) -> List[str]:
        """获取目录下所有图像文件"""
        image_files = []
        
        if recursive:
            for ext in extensions:
                pattern = os.path.join(directory, '**', f'*{ext}')
                image_files.extend(glob.glob(pattern, recursive=True))
                pattern = os.path.join(directory, '**', f'*{ext.upper()}')
                image_files.extend(glob.glob(pattern, recursive=True))
        else:
            for ext in extensions:
                pattern = os.path.join(directory, f'*{ext}')
                image_files.extend(glob.glob(pattern))
                pattern = os.path.join(directory, f'*{ext.upper()}')
                image_files.extend(glob.glob(pattern))
        
        return sorted(set(image_files))
    
    def index_images(
        self,
        image_paths: List[str],
        descriptions: Optional[Dict[str, str]] = None,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> int:
        """
        索引图像列表
        
        Args:
            image_paths: 图像路径列表
            descriptions: 图像描述字典
            batch_size: 批处理大小
            show_progress: 是否显示进度
            
        Returns:
            成功索引的图像数量
        """
        if descriptions is None:
            descriptions = {}
        
        total = len(image_paths)
        indexed = 0
        
        print(f"\n开始索引 {total} 张图像...")
        
        for i in range(0, total, batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_descriptions = []
            valid_paths = []
            
            # 验证图像
            for path in batch_paths:
                try:
                    img = Image.open(path)
                    img.verify()
                    valid_paths.append(path)
                    desc = descriptions.get(path, os.path.basename(path))
                    batch_descriptions.append(desc)
                except Exception as e:
                    print(f"警告: 跳过无效图像 {path}: {e}")
                    continue
            
            if not valid_paths:
                continue
            
            # 编码图像
            embeddings = self._encode_images(valid_paths, batch_size=batch_size)
            
            # 存入数据库
            self.store.insert(
                vectors=embeddings,
                image_paths=valid_paths,
                descriptions=batch_descriptions,
            )
            
            indexed += len(valid_paths)
            
            if show_progress:
                progress = min(i + batch_size, total)
                print(f"进度: {progress}/{total} ({progress/total*100:.1f}%)")
        
        print(f"\n索引完成! 成功索引 {indexed} 张图像")
        return indexed
    
    def index_directory(
        self,
        directory: str,
        descriptions: Optional[Dict[str, str]] = None,
        recursive: bool = True,
        batch_size: int = 32,
    ) -> int:
        """索引目录下的所有图像"""
        print(f"\n扫描目录: {directory}")
        image_files = self._get_image_files(directory, recursive=recursive)
        print(f"发现 {len(image_files)} 张图像")
        
        if not image_files:
            print("未找到图像文件!")
            return 0
        
        return self.index_images(
            image_paths=image_files,
            descriptions=descriptions,
            batch_size=batch_size,
        )
    
    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """文本搜图"""
        print(f"\n搜索: \"{query}\"")
        
        query_embedding = self._encode_texts([query])[0]
        
        results = self.store.search(
            query_vectors=query_embedding,
            top_k=top_k,
        )
        
        search_results = []
        for rank, hit in enumerate(results[0], 1):
            result = SearchResult(
                image_path=hit['image_path'],
                description=hit['description'],
                score=hit['score'],
                rank=rank,
            )
            search_results.append(result)
        
        return search_results
    
    def search_batch(
        self,
        queries: List[str],
        top_k: int = 10,
    ) -> List[List[SearchResult]]:
        """批量文本搜图"""
        query_embeddings = self._encode_texts(queries)
        
        all_results = self.store.search(
            query_vectors=query_embeddings,
            top_k=top_k,
        )
        
        batch_results = []
        for query_results in all_results:
            search_results = []
            for rank, hit in enumerate(query_results, 1):
                result = SearchResult(
                    image_path=hit['image_path'],
                    description=hit['description'],
                    score=hit['score'],
                    rank=rank,
                )
                search_results.append(result)
            batch_results.append(search_results)
        
        return batch_results
    
    def get_stats(self) -> Dict:
        """获取引擎统计信息"""
        return {
            'collection_name': self.store.collection_name,
            'total_images': self.store.count(),
            'embedding_dim': self._dim,
            'mode': self._mode,
        }
    
    def clear(self):
        """清空所有数据"""
        self.store.drop_collection()
        self.store.get_or_create_collection()
        print("已清空所有数据")
    
    def close(self):
        """关闭引擎"""
        self.store.close()


def print_results(results: List[SearchResult], show_path: bool = True):
    """打印搜索结果"""
    print("\n" + "=" * 60)
    print("搜索结果:")
    print("=" * 60)
    
    for result in results:
        print(f"\n[{result.rank}] 相似度分数: {result.score:.4f}")
        if show_path:
            print(f"    图像路径: {result.image_path}")
        print(f"    描述: {result.description}")
    
    print("\n" + "=" * 60)


def create_engine(
    mode: str = 'service',
    collection_name: str = 'clip_images',
    **kwargs
) -> ImageSearchEngine:
    """创建搜图引擎的便捷函数"""
    return ImageSearchEngine(
        mode=mode,
        collection_name=collection_name,
        **kwargs
    )


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='CLIP 文本搜图工具')
    parser.add_argument('--mode', choices=['index', 'search', 'demo'], default='demo',
                        help='运行模式')
    parser.add_argument('--encode-mode', choices=['service', 'local'], default='service',
                        help='编码模式: service=连接clip-as-service, local=本地加载模型')
    parser.add_argument('--clip-server', type=str, default='grpc://localhost:51000',
                        help='clip-as-service 服务器地址')
    parser.add_argument('--image-dir', type=str, help='图像目录')
    parser.add_argument('--query', type=str, help='搜索文本')
    parser.add_argument('--top-k', type=int, default=5, help='返回结果数量')
    parser.add_argument('--collection', type=str, default='clip_images', help='集合名称')
    parser.add_argument('--model', type=str, default='ViT-B-32::openai', help='CLIP 模型名称')
    parser.add_argument('--milvus-host', type=str, default='localhost', help='Milvus 地址')
    parser.add_argument('--milvus-port', type=str, default='19530', help='Milvus 端口')
    
    args = parser.parse_args()
    
    # 创建引擎
    engine = create_engine(
        mode=args.encode_mode,
        collection_name=args.collection,
        clip_server=args.clip_server,
        model_name=args.model,
        milvus_host=args.milvus_host,
        milvus_port=args.milvus_port,
    )
    
    try:
        if args.mode == 'index':
            if not args.image_dir:
                print("错误: index 模式需要指定 --image-dir")
            else:
                engine.index_directory(args.image_dir)
        
        elif args.mode == 'search':
            if not args.query:
                print("错误: search 模式需要指定 --query")
            else:
                results = engine.search(args.query, top_k=args.top_k)
                print_results(results)
        
        elif args.mode == 'demo':
            stats = engine.get_stats()
            print(f"\n当前模式: {stats['mode']}")
            print(f"集合: {stats['collection_name']}")
            print(f"已索引图像: {stats['total_images']}")
            
            if stats['total_images'] > 0:
                print("\n输入搜索文本 (输入 'quit' 退出):")
                while True:
                    query = input("\n> ").strip()
                    if query.lower() == 'quit':
                        break
                    if query:
                        results = engine.search(query, top_k=args.top_k)
                        print_results(results)
    
    finally:
        engine.close()

"""
CLIP向量化服务：使用CLIP-as-service对文本和图像进行编码
"""
import io
from typing import List, Union, Dict
import numpy as np
from clip_client import Client
from tqdm import tqdm


class CLIPVectorizer:
    """使用CLIP-as-service进行向量化"""
    
    def __init__(self, server_url: str = "grpc://0.0.0.0:51000"):
        """
        初始化CLIP向量化器
        
        Args:
            server_url: CLIP服务器地址，格式如 'grpc://0.0.0.0:51000'
        """
        self.client = Client(server_url)
        print(f"✓ Connected to CLIP server: {server_url}")
    
    def encode_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        对文本列表进行向量化
        
        Args:
            texts: 文本列表
            show_progress: 是否显示进度条
            
        Returns:
            向量数组，shape: (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        # 使用clip_client进行编码
        result = self.client.encode(
            texts,
            show_progress=show_progress
        )
        
        # 转换为numpy数组
        if hasattr(result, 'embeddings'):
            return np.array(result.embeddings)
        elif isinstance(result, np.ndarray):
            return result
        else:
            # 如果是DocumentArray
            embeddings = []
            for doc in result:
                if hasattr(doc, 'embedding'):
                    embeddings.append(doc.embedding)
                elif hasattr(doc, 'tensor'):
                    embeddings.append(doc.tensor)
            return np.array(embeddings)
    
    def encode_images(self, images: List[Union[str, bytes, Dict]], show_progress: bool = True) -> np.ndarray:
        """
        对图像列表进行向量化
        
        Args:
            images: 图像列表，可以是：
                - 图像路径字符串列表
                - 图像二进制数据列表 (bytes)
                - 包含 'binary' 或 'path' 键的字典列表
            show_progress: 是否显示进度条
            
        Returns:
            向量数组，shape: (n_images, embedding_dim)
        """
        if not images:
            return np.array([])
        
        # 预处理图像输入
        image_inputs = []
        for img in images:
            if isinstance(img, dict):
                # 如果是字典，优先使用binary，否则使用path
                if 'binary' in img:
                    # 转换为base64 data URI
                    import base64
                    img_format = img.get('format', 'PNG')
                    img_data = base64.b64encode(img['binary']).decode()
                    image_inputs.append(f"data:image/{img_format.lower()};base64,{img_data}")
                elif 'path' in img:
                    image_inputs.append(img['path'])
                else:
                    raise ValueError("Dictionary must contain 'binary' or 'path' key")
            elif isinstance(img, bytes):
                # 二进制数据转换为base64
                import base64
                img_data = base64.b64encode(img).decode()
                image_inputs.append(f"data:image/png;base64,{img_data}")
            else:
                # 字符串路径
                image_inputs.append(img)
        
        # 使用clip_client进行编码
        result = self.client.encode(
            image_inputs,
            show_progress=show_progress
        )
        
        # 转换为numpy数组
        if hasattr(result, 'embeddings'):
            return np.array(result.embeddings)
        elif isinstance(result, np.ndarray):
            return result
        else:
            # 如果是DocumentArray
            embeddings = []
            for doc in result:
                if hasattr(doc, 'embedding'):
                    embeddings.append(doc.embedding)
                elif hasattr(doc, 'tensor'):
                    embeddings.append(doc.tensor)
            return np.array(embeddings)
    
    def encode_mixed(self, texts: List[str], images: List[Union[str, bytes, Dict]], 
                     show_progress: bool = True) -> tuple:
        """
        同时对文本和图像进行向量化
        
        Args:
            texts: 文本列表
            images: 图像列表
            show_progress: 是否显示进度条
            
        Returns:
            (text_vectors, image_vectors) 元组
        """
        text_vectors = self.encode_texts(texts, show_progress=show_progress) if texts else np.array([])
        image_vectors = self.encode_images(images, show_progress=show_progress) if images else np.array([])
        
        return text_vectors, image_vectors
    
    def get_embedding_dimension(self) -> int:
        """
        获取嵌入向量的维度
        
        Returns:
            向量维度
        """
        # 使用一个简单的测试文本来获取维度
        test_vector = self.encode_texts(["test"])
        return test_vector.shape[1] if len(test_vector) > 0 else 512


"""
CLIP 编码器模块
封装 CLIP 模型的加载和图像/文本编码功能
"""
import os
import io
from typing import List, Union, Optional
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from clip_server.model.clip_model import CLIPModel
from clip_server.model.tokenization import Tokenizer


def _convert_image_to_rgb(image):
    return image.convert('RGB')


def _transform(n_px: int):
    """创建图像预处理 pipeline"""
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ])


class CLIPEncoder:
    """
    CLIP 编码器，用于将图像和文本编码为向量
    
    使用示例:
    ```python
    encoder = CLIPEncoder()
    
    # 编码图像
    image_embeddings = encoder.encode_images(['path/to/image1.jpg', 'path/to/image2.png'])
    
    # 编码文本
    text_embeddings = encoder.encode_texts(['a cat', 'a dog'])
    ```
    """
    
    def __init__(
        self,
        model_name: str = 'ViT-B-32::openai',
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        初始化 CLIP 编码器
        
        Args:
            model_name: CLIP 模型名称，默认为 'ViT-B-32::openai'
                       可选: 'ViT-B-16::openai', 'ViT-L-14::openai' 等
            device: 运行设备，'cpu' 或 'cuda'，默认自动检测
            dtype: 数据类型，默认 CPU 使用 float32，CUDA 使用 float16
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._device = device
        
        if dtype is None:
            dtype = torch.float32 if device == 'cpu' else torch.float16
        self._dtype = dtype
        
        print(f"正在加载 CLIP 模型: {model_name} (设备: {device})")
        self._model = CLIPModel(model_name, device=device, jit=False, dtype=dtype)
        self._tokenizer = Tokenizer(model_name)
        
        # 获取图像大小并创建预处理 pipeline
        self._image_size = self._model.image_size
        self._image_transform = _transform(self._image_size)
        
        print(f"模型加载完成! 图像大小: {self._image_size}x{self._image_size}")
    
    @property
    def embedding_dim(self) -> int:
        """获取嵌入向量的维度"""
        # 通过编码一个空白图像来获取维度
        dummy_image = Image.new('RGB', (self._image_size, self._image_size))
        dummy_tensor = self._image_transform(dummy_image).unsqueeze(0).to(self._device).type(self._dtype)
        with torch.no_grad():
            embedding = self._model.encode_image(dummy_tensor)
        return embedding.shape[-1]
    
    def _load_image(self, image_path: str) -> Image.Image:
        """加载图像文件"""
        return Image.open(image_path)
    
    def _preprocess_images(self, images: List[Union[str, Image.Image]]) -> torch.Tensor:
        """预处理图像列表"""
        tensors = []
        for img in images:
            if isinstance(img, str):
                img = self._load_image(img)
            tensor = self._image_transform(img)
            tensors.append(tensor)
        
        batch = torch.stack(tensors).to(self._device).type(self._dtype)
        return batch
    
    def _preprocess_texts(self, texts: List[str]) -> torch.Tensor:
        """预处理文本列表"""
        result = self._tokenizer(texts)
        input_ids = result['input_ids'].to(self._device)
        return input_ids
    
    def encode_images(
        self,
        images: List[Union[str, Image.Image]],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        编码图像列表为向量
        
        Args:
            images: 图像路径列表或 PIL.Image 对象列表
            batch_size: 批处理大小
            normalize: 是否对向量进行 L2 归一化
            
        Returns:
            np.ndarray: 形状为 (N, D) 的向量数组，N 为图像数量，D 为向量维度
        """
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                batch_tensor = self._preprocess_images(batch_images)
                
                embeddings = self._model.encode_image(batch_tensor)
                embeddings = embeddings.cpu().numpy().astype(np.float32)
                
                if normalize:
                    embeddings = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)
                
                all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)
    
    def encode_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        编码文本列表为向量
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            normalize: 是否对向量进行 L2 归一化
            
        Returns:
            np.ndarray: 形状为 (N, D) 的向量数组，N 为文本数量，D 为向量维度
        """
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                input_ids = self._preprocess_texts(batch_texts)
                
                embeddings = self._model.encode_text(input_ids)
                embeddings = embeddings.cpu().numpy().astype(np.float32)
                
                if normalize:
                    embeddings = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)
                
                all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)
    
    def encode_single_image(self, image: Union[str, Image.Image], normalize: bool = True) -> np.ndarray:
        """编码单张图像"""
        return self.encode_images([image], normalize=normalize)[0]
    
    def encode_single_text(self, text: str, normalize: bool = True) -> np.ndarray:
        """编码单条文本"""
        return self.encode_texts([text], normalize=normalize)[0]


if __name__ == '__main__':
    # 测试代码
    encoder = CLIPEncoder()
    print(f"向量维度: {encoder.embedding_dim}")
    
    # 测试文本编码
    texts = ["a photo of a cat", "a photo of a dog"]
    text_embeddings = encoder.encode_texts(texts)
    print(f"文本向量形状: {text_embeddings.shape}")


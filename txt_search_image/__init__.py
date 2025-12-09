"""
CLIP 文本搜图模块

提供基于 CLIP 模型和 Milvus 向量数据库的文本搜图功能。

支持两种编码模式:
- service 模式: 连接 clip-as-service 服务器 (推荐生产环境)
- local 模式: 本地直接加载模型 (适合开发测试)

主要组件:
- MilvusStore: Milvus 向量存储，管理向量的存储和检索
- ImageSearchEngine: 文本搜图引擎，整合编码和检索功能
- CLIPEncoder: 本地 CLIP 编码器 (local 模式使用)

使用示例:
```python
from txt_search_image import ImageSearchEngine, print_results

# 方式一: 使用 clip-as-service 服务 (推荐)
engine = ImageSearchEngine(
    mode='service',
    clip_server='grpc://localhost:51000'
)

# 方式二: 本地加载模型
engine = ImageSearchEngine(mode='local')

# 索引图像
engine.index_directory('/path/to/images')

# 搜索
results = engine.search('a cute cat', top_k=5)
print_results(results)
```
"""

from .milvus_store import MilvusStore
from .image_search import (
    ImageSearchEngine,
    SearchResult,
    print_results,
    create_engine,
)

# CLIPEncoder 仅在 local 模式下需要
def get_clip_encoder():
    """获取本地 CLIP 编码器 (仅 local 模式需要)"""
    from .clip_encoder import CLIPEncoder
    return CLIPEncoder

__all__ = [
    'MilvusStore',
    'ImageSearchEngine',
    'SearchResult',
    'print_results',
    'create_engine',
    'get_clip_encoder',
]

__version__ = '1.0.0'


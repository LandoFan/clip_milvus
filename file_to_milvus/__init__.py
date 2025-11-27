"""
Word和Markdown文件向量化存储系统
"""

__version__ = "1.0.0"

from .file_parser import FileParserFactory, WordParser, MarkdownParser
from .vectorizer import CLIPVectorizer
from .milvus_store import MilvusStore

__all__ = [
    'FileParserFactory',
    'WordParser',
    'MarkdownParser',
    'CLIPVectorizer',
    'MilvusStore',
]


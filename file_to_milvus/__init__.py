"""
Word和Markdown文件向量化存储系统
"""

__version__ = "1.0.0"

from .file_parsers.file_parser import FileParserFactory, WordParser, MarkdownParser
from .clip.vectorizer import CLIPVectorizer
from .milvus.milvus_store import MilvusStore

__all__ = [
    'FileParserFactory',
    'WordParser',
    'MarkdownParser',
    'CLIPVectorizer',
    'MilvusStore',
]


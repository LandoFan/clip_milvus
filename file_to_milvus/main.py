"""
主程序：将Word和Markdown文件向量化并存储到Milvus
"""
import os
import sys
from pathlib import Path
from typing import List, Optional
import argparse
from tqdm import tqdm
from dotenv import load_dotenv

from file_parsers.file_parser import FileParserFactory, ExtractedContent
from clip.vectorizer import CLIPVectorizer
from milvus.milvus_store import MilvusStore


# 加载环境变量
load_dotenv()


def process_file(file_path: str,
                 parser_factory: FileParserFactory,
                 vectorizer: CLIPVectorizer,
                 milvus_store: MilvusStore,
                 batch_size: int = 10):
    """
    处理单个文件
    
    Args:
        file_path: 文件路径
        parser_factory: 文件解析器工厂
        vectorizer: CLIP向量化器
        milvus_store: Milvus存储
        batch_size: 批处理大小
    """
    print(f"\n处理文件: {file_path}")
    print("-" * 60)
    
    try:
        # 解析文件
        parser = parser_factory.get_parser(file_path)
        extracted = parser.parse(file_path)
        
        print(f"  提取到 {len(extracted.text_chunks)} 个文本块")
        print(f"  提取到 {len(extracted.images)} 个图像")
        
        # 向量化文本
        if extracted.text_chunks:
            print("  向量化文本...")
            text_embeddings = vectorizer.encode_texts(
                extracted.text_chunks,
                show_progress=False
            )
            
            # 存储文本向量
            milvus_store.insert_texts(
                texts=extracted.text_chunks,
                embeddings=text_embeddings,
                file_path=file_path,
                file_type=extracted.metadata['file_type'],
                metadata=extracted.metadata
            )
        
        # 向量化图像
        if extracted.images:
            print("  向量化图像...")
            image_embeddings = vectorizer.encode_images(
                extracted.images,
                show_progress=False
            )
            
            # 提取图像路径
            image_paths = [img['path'] for img in extracted.images]
            
            # 存储图像向量
            milvus_store.insert_images(
                image_paths=image_paths,
                embeddings=image_embeddings,
                file_path=file_path,
                file_type=extracted.metadata['file_type'],
                metadata=extracted.metadata
            )
        
        print(f"  ✓ 文件处理完成")
        
    except Exception as e:
        print(f"  ✗ 处理文件时出错: {e}")
        import traceback
        traceback.print_exc()


def process_directory(directory: str,
                     parser_factory: FileParserFactory,
                     vectorizer: CLIPVectorizer,
                     milvus_store: MilvusStore,
                     file_extensions: List[str] = ['.docx', '.md', '.markdown'],
                     recursive: bool = True):
    """
    处理目录中的所有文件
    
    Args:
        directory: 目录路径
        parser_factory: 文件解析器工厂
        vectorizer: CLIP向量化器
        milvus_store: Milvus存储
        file_extensions: 支持的文件扩展名列表
        recursive: 是否递归处理子目录
    """
    directory_path = Path(directory)
    
    if not directory_path.exists():
        print(f"错误: 目录不存在: {directory}")
        return
    
    # 收集所有文件
    files = []
    if recursive:
        for ext in file_extensions:
            files.extend(directory_path.rglob(f"*{ext}"))
    else:
        for ext in file_extensions:
            files.extend(directory_path.glob(f"*{ext}"))
    
    if not files:
        print(f"在目录 {directory} 中未找到支持的文件")
        return
    
    print(f"\n找到 {len(files)} 个文件需要处理")
    print("=" * 60)
    
    # 处理每个文件
    for file_path in tqdm(files, desc="处理文件"):
        process_file(
            str(file_path),
            parser_factory,
            vectorizer,
            milvus_store
        )
    
    print(f"\n✓ 所有文件处理完成")


def search(query_text: str,
          vectorizer: CLIPVectorizer,
          milvus_store: MilvusStore,
          content_type: Optional[str] = None,
          limit: int = 10):
    """
    搜索功能
    
    Args:
        query_text: 查询文本
        vectorizer: CLIP向量化器
        milvus_store: Milvus存储
        content_type: 内容类型筛选 ('text' 或 'image')
        limit: 返回结果数量
    """
    print(f"\n搜索: {query_text}")
    print("=" * 60)
    
    # 向量化查询文本
    query_vector = vectorizer.encode_texts([query_text], show_progress=False)
    
    # 在Milvus中搜索
    results = milvus_store.search(
        query_vectors=query_vector,
        content_type=content_type,
        limit=limit
    )
    
    # 显示结果
    print(f"\n找到 {len(results)} 个结果:\n")
    for i, result in enumerate(results, 1):
        print(f"结果 {i}:")
        print(f"  相似度: {result['distance']:.4f}")
        print(f"  类型: {result['content_type']}")
        print(f"  文件: {result['file_path']}")
        if result['content_type'] == 'text':
            content = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
            print(f"  内容: {content}")
        else:
            print(f"  图像路径: {result['content']}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="将Word和Markdown文件向量化并存储到Milvus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理单个文件
  python main.py --file document.docx
  
  # 处理目录中的所有文件
  python main.py --dir ./documents
  
  # 搜索
  python main.py --search "查询文本"
  
  # 指定CLIP服务器地址
  python main.py --file document.docx --clip-server grpc://localhost:51000
        """
    )
    
    # 输入选项
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--file', type=str, help='处理单个文件')
    input_group.add_argument('--dir', type=str, help='处理目录中的所有文件')
    input_group.add_argument('--search', type=str, help='搜索模式')
    
    # 配置选项
    parser.add_argument('--clip-server', type=str,
                       default=os.getenv('CLIP_SERVER', 'grpc://0.0.0.0:51000'),
                       help='CLIP服务器地址 (默认: grpc://0.0.0.0:51000)')
    
    parser.add_argument('--milvus-host', type=str,
                       default=os.getenv('MILVUS_HOST', 'localhost'),
                       help='Milvus服务器地址 (默认: localhost)')
    
    parser.add_argument('--milvus-port', type=int,
                       default=int(os.getenv('MILVUS_PORT', '19530')),
                       help='Milvus服务器端口 (默认: 19530)')
    
    parser.add_argument('--collection', type=str,
                       default=os.getenv('MILVUS_COLLECTION', 'clip_documents'),
                       help='Milvus集合名称 (默认: clip_documents)')
    
    parser.add_argument('--embedding-dim', type=int,
                       default=int(os.getenv('EMBEDDING_DIM', '512')),
                       help='向量维度 (默认: 512)')
    
    parser.add_argument('--drop-collection', action='store_true',
                       help='删除已存在的集合并重建')
    
    parser.add_argument('--content-type', type=str, choices=['text', 'image'],
                       help='搜索时筛选内容类型')
    
    parser.add_argument('--limit', type=int, default=10,
                       help='搜索结果数量 (默认: 10)')
    
    parser.add_argument('--no-recursive', action='store_true',
                       help='不递归处理子目录')
    
    args = parser.parse_args()
    
    # 初始化组件
    print("初始化组件...")
    print("=" * 60)
    
    try:
        # 初始化CLIP向量化器
        vectorizer = CLIPVectorizer(server_url=args.clip_server)
        embedding_dim = vectorizer.get_embedding_dimension()
        print(f"✓ 向量维度: {embedding_dim}")
        
        # 初始化Milvus存储
        milvus_store = MilvusStore(
            host=args.milvus_host,
            port=args.milvus_port,
            collection_name=args.collection,
            embedding_dim=embedding_dim,
            drop_existing=args.drop_collection
        )
        
        # 获取统计信息
        stats = milvus_store.get_stats()
        print(f"✓ Milvus集合: {stats['collection_name']}")
        print(f"✓ 现有实体数: {stats['num_entities']}")
        
        # 初始化文件解析器工厂
        parser_factory = FileParserFactory()
        
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 执行操作
    if args.search:
        # 搜索模式
        search(
            query_text=args.search,
            vectorizer=vectorizer,
            milvus_store=milvus_store,
            content_type=args.content_type,
            limit=args.limit
        )
    
    elif args.file:
        # 处理单个文件
        process_file(
            file_path=args.file,
            parser_factory=parser_factory,
            vectorizer=vectorizer,
            milvus_store=milvus_store
        )
        
        # 显示统计信息
        stats = milvus_store.get_stats()
        print(f"\n✓ 完成! 当前集合实体数: {stats['num_entities']}")
    
    elif args.dir:
        # 处理目录
        process_directory(
            directory=args.dir,
            parser_factory=parser_factory,
            vectorizer=vectorizer,
            milvus_store=milvus_store,
            recursive=not args.no_recursive
        )
        
        # 显示统计信息
        stats = milvus_store.get_stats()
        print(f"\n✓ 完成! 当前集合实体数: {stats['num_entities']}")


if __name__ == '__main__':
    main()


"""
主程序（层次化版本）：使用父子分段和混合检索
"""
import os
import sys
from pathlib import Path
from typing import List, Optional
import argparse
from tqdm import tqdm
from dotenv import load_dotenv

from parser.hierarchical_parser import (
    HierarchicalWordParser,
    HierarchicalMarkdownParser,
    HierarchicalContent
)
from clip.vectorizer import CLIPVectorizer
from milvus.hierarchical_store import HierarchicalMilvusStore

# 加载环境变量
load_dotenv()


def process_file_hierarchical(file_path: str,
                              vectorizer: CLIPVectorizer,
                              milvus_store: HierarchicalMilvusStore,
                              max_chunk_size: int = 500):
    """
    使用层次化方式处理单个文件
    
    Args:
        file_path: 文件路径
        vectorizer: CLIP向量化器
        milvus_store: 层次化Milvus存储
        max_chunk_size: 最大块大小
    """
    print(f"\n处理文件: {file_path}")
    print("-" * 60)
    
    try:
        # 根据文件类型选择解析器
        ext = Path(file_path).suffix.lower()
        
        if ext == '.docx':
            parser = HierarchicalWordParser(max_chunk_size=max_chunk_size)
        elif ext in ['.md', '.markdown']:
            parser = HierarchicalMarkdownParser(max_chunk_size=max_chunk_size)
        else:
            print(f"  ✗ 不支持的文件类型: {ext}")
            return
        
        # 解析文件
        hierarchical_content = parser.parse(file_path)
        
        print(f"  提取到 {len(hierarchical_content.chunks)} 个块")
        print(f"  最大层级: {hierarchical_content.metadata['max_level']}")
        
        # 提取文本内容
        texts = [chunk.content for chunk in hierarchical_content.chunks]
        
        # 向量化
        print("  向量化文本块...")
        embeddings = vectorizer.encode_texts(texts, show_progress=False)
        
        # 存储到Milvus
        milvus_store.insert_hierarchical_chunks(
            hierarchical_content=hierarchical_content,
            embeddings=embeddings,
            file_path=file_path,
            file_type=hierarchical_content.metadata['file_type']
        )
        
        print(f"  ✓ 文件处理完成")
        
    except Exception as e:
        print(f"  ✗ 处理文件时出错: {e}")
        import traceback
        traceback.print_exc()


def process_directory_hierarchical(directory: str,
                                  vectorizer: CLIPVectorizer,
                                  milvus_store: HierarchicalMilvusStore,
                                  file_extensions: List[str] = ['.docx', '.md', '.markdown'],
                                  recursive: bool = True,
                                  max_chunk_size: int = 500):
    """
    批量处理目录
    
    Args:
        directory: 目录路径
        vectorizer: CLIP向量化器
        milvus_store: Milvus存储
        file_extensions: 支持的文件扩展名
        recursive: 是否递归
        max_chunk_size: 最大块大小
    """
    directory_path = Path(directory)
    
    if not directory_path.exists():
        print(f"错误: 目录不存在: {directory}")
        return
    
    # 收集文件
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
        process_file_hierarchical(
            str(file_path),
            vectorizer,
            milvus_store,
            max_chunk_size
        )
    
    print(f"\n✓ 所有文件处理完成")


def hybrid_search(query_text: str,
                 vectorizer: CLIPVectorizer,
                 milvus_store: HierarchicalMilvusStore,
                 content_type: Optional[str] = None,
                 limit: int = 10,
                 alpha: float = 0.7,
                 hierarchical: bool = True):
    """
    混合检索
    
    Args:
        query_text: 查询文本
        vectorizer: CLIP向量化器
        milvus_store: Milvus存储
        content_type: 内容类型
        limit: 返回结果数量
        alpha: 向量检索权重
        hierarchical: 是否使用层次化检索
    """
    print(f"\n搜索: {query_text}")
    print(f"混合检索权重 - 向量: {alpha:.1%}, 关键词: {1-alpha:.1%}")
    print("=" * 60)
    
    # 向量化查询文本
    query_vector = vectorizer.encode_texts([query_text], show_progress=False)[0]
    
    # 执行检索
    if hierarchical:
        results = milvus_store.hierarchical_search(
            query_text=query_text,
            query_vector=query_vector,
            content_type=content_type,
            limit=limit,
            alpha=alpha,
            include_children=True,
            include_parent=True
        )
    else:
        results = milvus_store.hybrid_search(
            query_text=query_text,
            query_vector=query_vector,
            content_type=content_type,
            limit=limit,
            alpha=alpha
        )
    
    # 显示结果
    print(f"\n找到 {len(results)} 个结果:\n")
    for i, result in enumerate(results, 1):
        print(f"结果 {i}:")
        print(f"  相似度分数: {result['distance']:.4f}")
        print(f"  块类型: {result.get('chunk_type', 'unknown')}")
        print(f"  层级: {result.get('level', 0)}")
        print(f"  父块ID: {result.get('parent_id', 'N/A')}")
        print(f"  文件: {result.get('file_path', 'N/A')}")
        
        content = result.get('content', '')
        if content:
            content_preview = content[:200] + "..." if len(content) > 200 else content
            print(f"  内容: {content_preview}")
        
        # 显示上下文信息
        metadata = result.get('metadata', {})
        if metadata.get('children_ids'):
            print(f"  子块数: {len(metadata['children_ids'])}")
        
        print()


def main():
    parser = argparse.ArgumentParser(
        description="层次化文件向量化和混合检索系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理单个文件（层次化分段）
  python main_hierarchical.py --file document.docx
  
  # 处理目录
  python main_hierarchical.py --dir ./documents
  
  # 混合检索
  python main_hierarchical.py --search "查询文本"
  
  # 调整混合检索权重（alpha越高，向量检索权重越高）
  python main_hierarchical.py --search "查询" --alpha 0.8
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
                       help='CLIP服务器地址')
    
    parser.add_argument('--milvus-host', type=str,
                       default=os.getenv('MILVUS_HOST', 'localhost'),
                       help='Milvus服务器地址')
    
    parser.add_argument('--milvus-port', type=int,
                       default=int(os.getenv('MILVUS_PORT', '19530')),
                       help='Milvus服务器端口')
    
    parser.add_argument('--collection', type=str,
                       default=os.getenv('MILVUS_COLLECTION', 'clip_documents_hierarchical'),
                       help='Milvus集合名称')
    
    parser.add_argument('--embedding-dim', type=int,
                       default=int(os.getenv('EMBEDDING_DIM', '512')),
                       help='向量维度')
    
    parser.add_argument('--drop-collection', action='store_true',
                       help='删除已存在的集合并重建')
    
    parser.add_argument('--content-type', type=str, choices=['text', 'image'],
                       help='搜索时筛选内容类型')
    
    parser.add_argument('--limit', type=int, default=10,
                       help='搜索结果数量')
    
    parser.add_argument('--no-recursive', action='store_true',
                       help='不递归处理子目录')
    
    parser.add_argument('--max-chunk-size', type=int, default=500,
                       help='最大块大小（字符数）')
    
    parser.add_argument('--alpha', type=float, default=0.7,
                       help='混合检索中向量检索的权重 (0-1, 默认0.7)')
    
    parser.add_argument('--no-hierarchical', action='store_true',
                       help='不使用层次化检索（仅混合检索）')
    
    args = parser.parse_args()
    
    # 初始化组件
    print("初始化组件...")
    print("=" * 60)
    
    try:
        # 初始化CLIP向量化器
        vectorizer = CLIPVectorizer(server_url=args.clip_server)
        embedding_dim = vectorizer.get_embedding_dimension()
        print(f"✓ 向量维度: {embedding_dim}")
        
        # 初始化层次化Milvus存储
        milvus_store = HierarchicalMilvusStore(
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
        
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 执行操作
    if args.search:
        # 搜索模式
        hybrid_search(
            query_text=args.search,
            vectorizer=vectorizer,
            milvus_store=milvus_store,
            content_type=args.content_type,
            limit=args.limit,
            alpha=args.alpha,
            hierarchical=not args.no_hierarchical
        )
    
    elif args.file:
        # 处理单个文件
        process_file_hierarchical(
            file_path=args.file,
            vectorizer=vectorizer,
            milvus_store=milvus_store,
            max_chunk_size=args.max_chunk_size
        )
        
        # 显示统计信息
        stats = milvus_store.get_stats()
        print(f"\n✓ 完成! 当前集合实体数: {stats['num_entities']}")
    
    elif args.dir:
        # 处理目录
        process_directory_hierarchical(
            directory=args.dir,
            vectorizer=vectorizer,
            milvus_store=milvus_store,
            recursive=not args.no_recursive,
            max_chunk_size=args.max_chunk_size
        )
        
        # 显示统计信息
        stats = milvus_store.get_stats()
        print(f"\n✓ 完成! 当前集合实体数: {stats['num_entities']}")


if __name__ == '__main__':
    main()


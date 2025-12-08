"""
知识库API使用示例

使用方法：
    python kb_api_example.py --file your_document.docx --query "搜索内容"
    
使用前请确保：
1. CLIP服务器已启动: python -m clip_server
2. Milvus已启动
"""
import argparse
from pathlib import Path

from milvus.knowledge_base import KnowledgeBase, create_knowledge_base


def example_basic_usage(file_path: str, query_text: str = "CLIP"):
    """
    基本使用示例
    
    Args:
        file_path: 文档文件路径 (.docx 或 .md)
        query_text: 查询文本
    """
    print("=" * 60)
    print("知识库示例")
    print("=" * 60)
    
    # 创建知识库实例
    kb = KnowledgeBase(
        clip_server="grpc://0.0.0.0:51000",
        milvus_host="localhost",
        milvus_port=19530,
        collection_name="my_knowledge_base"
    )
    
    # 添加文档
    print(f"\n📄 添加文档: {file_path}")
    result = kb.add_document(file_path)
    print(f"添加结果: {result}")
    
    if result['success']:
        # 查询
        print(f"\n🔍 查询: '{query_text}'")
        results = kb.query(query_text, top_k=5, hierarchical=False, alpha=1.0)
        
        print(f"\n找到 {len(results)} 个结果:")
        for i, r in enumerate(results, 1):
            content = r['content'][:150] + "..." if len(r['content']) > 150 else r['content']
            similarity = 1 / (1 + r['distance'])
            print(f"\n[{i}] 相似度: {similarity:.4f}")
            print(f"    文件: {r['file_path']}")
            print(f"    内容: {content}")
    
    # 统计
    print("\n📊 知识库统计:")
    stats = kb.get_stats()
    print(f"  集合名称: {stats['collection_name']}")
    print(f"  实体数量: {stats['num_entities']}")


def example_add_and_search():
    """添加文档并搜索的完整示例"""
    print("\n" + "=" * 60)
    print("示例2: 添加文档并搜索")
    print("=" * 60)
    
    kb = KnowledgeBase(collection_name="test_kb")
    
    # 获取用户输入
    file_path = input("请输入要添加的文件路径: ").strip()
    
    if not file_path or not Path(file_path).exists():
        print(f"❌ 文件不存在: {file_path}")
        return
    
    # 添加文档
    print(f"\n正在添加文档...")
    result = kb.add_document(file_path)
    
    if result['success']:
        print(f"✓ 成功添加 {result['chunks_count']} 个文本块")
        
        # 交互式搜索
        print("\n现在可以搜索了 (输入 'q' 退出):")
        while True:
            query = input("\n🔍 查询 > ").strip()
            if query.lower() in ['q', 'quit', 'exit']:
                break
            if not query:
                continue
            
            results = kb.query(query, top_k=3, hierarchical=False, alpha=1.0)
            
            if results:
                for i, r in enumerate(results, 1):
                    content = r['content'][:200] + "..." if len(r['content']) > 200 else r['content']
                    similarity = 1 / (1 + r['distance'])
                    print(f"\n[{i}] 相似度: {similarity:.4f}")
                    print(f"    {content}")
            else:
                print("未找到相关结果")
    else:
        print(f"❌ 添加失败: {result['message']}")


def example_batch_operations():
    """批量操作示例"""
    print("\n" + "=" * 60)
    print("示例3: 批量操作")
    print("=" * 60)
    
    kb = KnowledgeBase()
    
    # 查找当前目录下所有文档
    doc_files = list(Path(".").glob("*.docx")) + list(Path(".").glob("*.md"))
    
    if not doc_files:
        print("当前目录没有找到文档文件")
        return
    
    print(f"找到 {len(doc_files)} 个文档:")
    for f in doc_files:
        print(f"  - {f}")
    
    # 批量添加
    results = kb.add_documents([str(f) for f in doc_files], show_progress=True)
    
    for result in results:
        if result['success']:
            print(f"✓ {result['file_path']}: {result['chunks_count']} 个块")
        else:
            print(f"✗ {result.get('file_path', 'unknown')}: {result['message']}")


def example_pure_vector_search():
    """纯向量搜索（效果更好的方式）"""
    print("\n" + "=" * 60)
    print("示例4: 纯向量搜索")
    print("=" * 60)
    
    kb = KnowledgeBase(collection_name="my_knowledge_base")
    
    query_text = "深度学习模型"
    print(f"查询: {query_text}")
    
    # 使用纯向量搜索 (alpha=1.0 表示只用向量，不用BM25)
    results = kb.query(
        query_text=query_text,
        top_k=5,
        hierarchical=False,  # 不使用层次化
        alpha=1.0  # 纯向量搜索
    )
    
    print(f"\n找到 {len(results)} 个结果:")
    for i, r in enumerate(results, 1):
        content = r['content'][:150] + "..." if len(r['content']) > 150 else r['content']
        similarity = 1 / (1 + r['distance'])
        print(f"\n[{i}] 相似度: {similarity:.4f}")
        print(f"    {content}")


def example_knowledge_base_management():
    """知识库管理示例"""
    print("\n" + "=" * 60)
    print("示例5: 知识库管理")
    print("=" * 60)
    
    kb = KnowledgeBase()
    
    # 获取统计信息
    stats = kb.get_stats()
    print(f"\n知识库统计:")
    print(f"  集合名称: {stats['collection_name']}")
    print(f"  实体数量: {stats['num_entities']}")
    print(f"  CLIP服务器: {stats['clip_server']}")
    
    # 列出所有文档
    documents = kb.list_documents()
    print(f"\n知识库中的文档 ({len(documents)} 个):")
    for doc in documents[:10]:  # 只显示前10个
        print(f"  - {doc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="知识库API示例")
    parser.add_argument("--file", type=str, required=True, help="文档文件路径 (.docx 或 .md)")
    parser.add_argument("--query", type=str, default="CLIP", help="查询文本 (默认: CLIP)")
    
    args = parser.parse_args()
    
    try:
        example_basic_usage(args.file, args.query)
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()

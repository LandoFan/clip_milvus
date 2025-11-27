"""
知识库API使用示例
"""
from milvus.knowledge_base import KnowledgeBase, create_knowledge_base


def example_basic_usage():
    """基本使用示例"""
    print("=" * 60)
    print("示例1: 基本使用")
    print("=" * 60)
    
    # 创建知识库实例
    kb = KnowledgeBase(
        clip_server="grpc://0.0.0.0:51000",
        milvus_host="localhost",
        milvus_port=19530,
        collection_name="my_knowledge_base"
    )
    
    # 添加文档
    result = kb.add_document("document.docx")
    print(f"添加文档结果: {result}")
    
    # 查询
    results = kb.query("机器学习算法", top_k=5)
    print(f"\n查询结果数量: {len(results)}")
    for i, result in enumerate(results, 1):
        print(f"\n结果 {i}:")
        print(f"  内容: {result['content'][:100]}...")
        print(f"  分数: {result['distance']:.4f}")
        print(f"  文件: {result['file_path']}")


def example_batch_operations():
    """批量操作示例"""
    print("\n" + "=" * 60)
    print("示例2: 批量操作")
    print("=" * 60)
    
    kb = KnowledgeBase()
    
    # 批量添加文档
    file_paths = ["doc1.docx", "doc2.md", "doc3.docx"]
    results = kb.add_documents(file_paths, show_progress=True)
    
    for result in results:
        if result['success']:
            print(f"✓ {result['file_path']}: {result['chunks_count']} 个块")
        else:
            print(f"✗ {result.get('file_path', 'unknown')}: {result['message']}")


def example_advanced_query():
    """高级查询示例"""
    print("\n" + "=" * 60)
    print("示例3: 高级查询")
    print("=" * 60)
    
    kb = KnowledgeBase()
    
    # 查询1: 层次化检索（包含上下文）
    print("\n查询1: 层次化检索")
    results1 = kb.query(
        query_text="深度学习",
        top_k=5,
        hierarchical=True,
        include_children=True,
        include_parent=True
    )
    print(f"找到 {len(results1)} 个结果")
    
    # 查询2: 纯混合检索（不包含层次结构）
    print("\n查询2: 纯混合检索")
    results2 = kb.query(
        query_text="深度学习",
        top_k=5,
        hierarchical=False,
        alpha=0.8  # 更多向量检索
    )
    print(f"找到 {len(results2)} 个结果")
    
    # 查询3: 带过滤条件
    print("\n查询3: 带文件类型过滤")
    results3 = kb.query(
        query_text="算法",
        top_k=5,
        filter_expr='file_type == "word"'
    )
    print(f"找到 {len(results3)} 个结果")


def example_knowledge_base_management():
    """知识库管理示例"""
    print("\n" + "=" * 60)
    print("示例4: 知识库管理")
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
    for doc in documents[:5]:  # 只显示前5个
        print(f"  - {doc}")
    
    # 删除文档
    if documents:
        result = kb.delete_document(documents[0])
        print(f"\n删除文档结果: {result}")


def example_integration():
    """集成到其他项目的示例"""
    print("\n" + "=" * 60)
    print("示例5: 集成示例")
    print("=" * 60)
    
    # 使用便捷函数创建
    kb = create_knowledge_base(
        collection_name="production_kb"
    )
    
    # 在实际应用中使用
    def answer_question(question: str) -> str:
        """基于知识库回答问题"""
        results = kb.query(question, top_k=3, hierarchical=True)
        
        if not results:
            return "抱歉，在知识库中没有找到相关信息。"
        
        # 组合最相关的结果作为答案
        answer_parts = []
        for result in results:
            if result['distance'] < 0.5:  # 相似度阈值
                answer_parts.append(result['content'])
        
        if answer_parts:
            return "\n\n".join(answer_parts[:2])  # 最多使用前2个结果
        else:
            return "找到了一些相关信息，但相关性不高。"
    
    # 使用示例
    question = "什么是机器学习？"
    answer = answer_question(question)
    print(f"\n问题: {question}")
    print(f"答案: {answer[:200]}...")


if __name__ == "__main__":
    print("知识库API使用示例\n")
    print("注意: 这些示例需要先启动CLIP服务器和Milvus数据库")
    print("取消注释下面的代码行以运行示例\n")
    
    # 取消注释以运行示例
    # example_basic_usage()
    # example_batch_operations()
    # example_advanced_query()
    # example_knowledge_base_management()
    # example_integration()


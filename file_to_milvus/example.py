"""
使用示例：演示如何使用各个组件
"""
from parser.file_parser import FileParserFactory
from clip.vectorizer import CLIPVectorizer
from milvus.milvus_store import MilvusStore


def example_parse_file():
    """示例：解析文件"""
    print("=" * 60)
    print("示例1: 解析文件")
    print("=" * 60)
    
    # 创建解析器
    factory = FileParserFactory()
    
    # 解析Word文档
    parser = factory.get_parser("example.docx")
    extracted = parser.parse("example.docx")
    
    print(f"提取到 {len(extracted.text_chunks)} 个文本块")
    print(f"提取到 {len(extracted.images)} 个图像")
    
    # 显示第一个文本块
    if extracted.text_chunks:
        print(f"\n第一个文本块:\n{extracted.text_chunks[0][:200]}...")


def example_vectorize():
    """示例：向量化"""
    print("\n" + "=" * 60)
    print("示例2: 向量化")
    print("=" * 60)
    
    # 连接到CLIP服务器
    vectorizer = CLIPVectorizer(server_url="grpc://0.0.0.0:51000")
    
    # 向量化文本
    texts = ["这是第一个文本", "这是第二个文本"]
    embeddings = vectorizer.encode_texts(texts)
    
    print(f"文本数量: {len(texts)}")
    print(f"向量维度: {embeddings.shape}")
    print(f"向量形状: {embeddings.shape}")


def example_store():
    """示例：存储到Milvus"""
    print("\n" + "=" * 60)
    print("示例3: 存储到Milvus")
    print("=" * 60)
    
    # 连接到Milvus
    store = MilvusStore(
        host="localhost",
        port=19530,
        collection_name="example_collection",
        embedding_dim=512
    )
    
    # 插入示例数据
    texts = ["示例文本1", "示例文本2"]
    embeddings = [[0.1] * 512, [0.2] * 512]  # 示例向量
    
    import numpy as np
    store.insert_texts(
        texts=texts,
        embeddings=np.array(embeddings),
        file_path="example.docx",
        file_type="word"
    )
    
    print("✓ 数据已存储")


def example_search():
    """示例：搜索"""
    print("\n" + "=" * 60)
    print("示例4: 搜索")
    print("=" * 60)
    
    # 初始化组件
    vectorizer = CLIPVectorizer(server_url="grpc://0.0.0.0:51000")
    store = MilvusStore(
        host="localhost",
        port=19530,
        collection_name="example_collection",
        embedding_dim=512
    )
    
    # 搜索
    query_text = "搜索关键词"
    query_vector = vectorizer.encode_texts([query_text])
    
    results = store.search(
        query_vectors=query_vector,
        limit=5
    )
    
    print(f"找到 {len(results)} 个结果")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['content'][:100]}...")


def example_full_workflow():
    """完整工作流程示例"""
    print("\n" + "=" * 60)
    print("完整工作流程示例")
    print("=" * 60)
    
    # 1. 解析文件
    factory = FileParserFactory()
    parser = factory.get_parser("example.docx")
    extracted = parser.parse("example.docx")
    
    print(f"提取到 {len(extracted.text_chunks)} 个文本块")
    print(f"提取到 {len(extracted.images)} 个图像")
    
    # 2. 向量化
    vectorizer = CLIPVectorizer(server_url="grpc://0.0.0.0:51000")
    
    # 3. 存储
    store = MilvusStore(
        host="localhost",
        port=19530,
        collection_name="example_collection",
        embedding_dim=512
    )
    
    # 处理文本
    if extracted.text_chunks:
        text_embeddings = vectorizer.encode_texts(extracted.text_chunks)
        store.insert_texts(
            texts=extracted.text_chunks,
            embeddings=text_embeddings,
            file_path="example.docx",
            file_type="word",
            metadata=extracted.metadata
        )
        print(f"✓ 已存储 {len(extracted.text_chunks)} 个文本块")
    
    # 处理图像
    if extracted.images:
        from PIL import Image
        import io
        
        # 将图像二进制数据转换为PIL Image对象
        pil_images = []
        image_paths = []
        for img_info in extracted.images:
            try:
                img = Image.open(io.BytesIO(img_info['binary']))
                # 转换为RGB模式（CLIP需要）
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                pil_images.append(img)
                image_paths.append(img_info['path'])
            except Exception as e:
                print(f"Warning: 无法加载图像 {img_info['path']}: {e}")
        
        if pil_images:
            # 向量化图像
            image_embeddings = vectorizer.encode_images(pil_images)
            
            # 存储图像向量
            store.insert_images(
                image_paths=image_paths,
                embeddings=image_embeddings,
                file_path="example.docx",
                file_type="word",
                metadata=extracted.metadata
            )
            print(f"✓ 已存储 {len(pil_images)} 个图像")
    
    print("✓ 完整流程执行成功")


if __name__ == "__main__":
    print("这是一个使用示例文件")
    print("请根据实际情况修改文件路径和配置")
    print("\n可用的示例函数:")
    print("  - example_parse_file()")
    print("  - example_vectorize()")
    print("  - example_store()")
    print("  - example_search()")
    print("  - example_full_workflow()")
    
    # 取消注释以运行示例
    # example_parse_file()
    # example_vectorize()
    # example_store()
    # example_search()
    # example_full_workflow()


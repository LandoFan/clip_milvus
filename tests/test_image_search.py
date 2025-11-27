"""
测试图像搜索：演示如何查询嵌入的图片
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'file_to_milvus'))

from parser.file_parser import FileParserFactory
from clip.vectorizer import CLIPVectorizer
from milvus.milvus_store import MilvusStore
from PIL import Image
import io


def setup_test_data(docx_path: str):
    """准备测试数据：解析Word文档并存储到Milvus"""
    print("=" * 60)
    print("1. 准备测试数据")
    print("=" * 60)
    
    # 解析文档
    factory = FileParserFactory()
    parser = factory.get_parser(docx_path)
    extracted = parser.parse(docx_path)
    
    print(f"提取到 {len(extracted.text_chunks)} 个文本块")
    print(f"提取到 {len(extracted.images)} 个图像")
    
    if not extracted.images:
        print("❌ 文档中没有图像，请确保Word文档包含图片")
        return None
    
    # 初始化组件
    vectorizer = CLIPVectorizer(server_url="grpc://0.0.0.0:51000")
    store = MilvusStore(
        host="localhost",
        port=19530,
        collection_name="image_search_test",
        embedding_dim=512,
        drop_existing=True  # 重新创建集合
    )
    
    # 存储文本
    if extracted.text_chunks:
        text_embeddings = vectorizer.encode_texts(extracted.text_chunks)
        store.insert_texts(
            texts=extracted.text_chunks,
            embeddings=text_embeddings,
            file_path=docx_path,
            file_type="word"
        )
    
    # 存储图像
    pil_images = []
    image_paths = []
    for img_info in extracted.images:
        try:
            img = Image.open(io.BytesIO(img_info['binary']))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            pil_images.append(img)
            image_paths.append(img_info['path'])
            print(f"  - 加载图像: {img_info['path']}")
        except Exception as e:
            print(f"  - 警告: 无法加载图像 {img_info['path']}: {e}")
    
    if pil_images:
        image_embeddings = vectorizer.encode_images(pil_images)
        store.insert_images(
            image_paths=image_paths,
            embeddings=image_embeddings,
            file_path=docx_path,
            file_type="word"
        )
    
    return store, vectorizer


def test_search_by_text(store: MilvusStore, vectorizer: CLIPVectorizer, query: str):
    """方法1：用文本描述搜索图片"""
    print("\n" + "=" * 60)
    print(f"2. 用文本搜索图片: '{query}'")
    print("=" * 60)
    
    # 将文本转为向量
    query_vector = vectorizer.encode_texts([query])
    
    # 只搜索图片类型
    results = store.search(
        query_vectors=query_vector,
        content_type="image",  # 只搜索图片
        limit=5
    )
    
    print(f"找到 {len(results)} 个结果:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. 图片路径: {result['content']}")
        print(f"     距离: {result['distance']:.4f}")
        print(f"     来源文件: {result['file_path']}")
    
    return results


def test_search_by_image(store: MilvusStore, vectorizer: CLIPVectorizer, image_path: str):
    """方法2：用图片搜索相似图片"""
    print("\n" + "=" * 60)
    print(f"3. 用图片搜索相似图片: '{image_path}'")
    print("=" * 60)
    
    # 加载查询图片
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # 将图片转为向量
    query_vector = vectorizer.encode_images([img])
    
    # 搜索相似图片
    results = store.search(
        query_vectors=query_vector,
        content_type="image",
        limit=5
    )
    
    print(f"找到 {len(results)} 个相似图片:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. 图片路径: {result['content']}")
        print(f"     距离: {result['distance']:.4f}")
    
    return results


def test_search_all(store: MilvusStore, vectorizer: CLIPVectorizer, query: str):
    """方法3：用文本同时搜索文本和图片"""
    print("\n" + "=" * 60)
    print(f"4. 用文本搜索所有内容(文本+图片): '{query}'")
    print("=" * 60)
    
    query_vector = vectorizer.encode_texts([query])
    
    # 不指定content_type，搜索所有内容
    results = store.search(
        query_vectors=query_vector,
        limit=10
    )
    
    print(f"找到 {len(results)} 个结果:")
    for i, result in enumerate(results, 1):
        content_type = result['content_type']
        if content_type == 'image':
            print(f"  {i}. [图片] {result['content']}")
        else:
            print(f"  {i}. [文本] {result['content'][:80]}...")
        print(f"     距离: {result['distance']:.4f}")
    
    return results


if __name__ == "__main__":
    # 修改为你的Word文档路径
    DOCX_PATH = "example.docx"
    
    # 1. 准备数据
    result = setup_test_data(DOCX_PATH)
    if result is None:
        exit(1)
    
    store, vectorizer = result
    
    # 2. 用文本描述搜索图片
    # 修改为与你图片相关的描述
    test_search_by_text(store, vectorizer, "图表")
    test_search_by_text(store, vectorizer, "流程图")
    test_search_by_text(store, vectorizer, "示意图")
    
    # 3. 用文本搜索所有内容
    test_search_all(store, vectorizer, "系统架构")
    
    # 4. 如果有本地图片，可以用图片搜索相似图片
    # test_search_by_image(store, vectorizer, "query_image.png")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


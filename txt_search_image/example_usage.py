"""
CLIP 文本搜图使用示例

这个示例展示如何:
1. 索引图像到 Milvus 数据库
2. 使用文本搜索相似图像
3. 展示搜索结果

使用前请确保:
1. 已安装依赖: pip install -r requirements.txt
2. Milvus 服务已启动
"""
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

from image_search import ImageSearchEngine, print_results


def example_basic_usage():
    """基本使用示例"""
    print("\n" + "=" * 60)
    print("示例 1: 基本使用")
    print("=" * 60)
    
    # 创建搜图引擎
    engine = ImageSearchEngine(
        collection_name='demo_collection',
        model_name='ViT-B-32::openai',
    )
    
    # 获取测试图像 (使用项目中的测试图像)
    test_images_dir = os.path.join(
        os.path.dirname(__file__),
        '..', 'clip-as-service', 'tests', 'img'
    )
    
    if os.path.exists(test_images_dir):
        print(f"\n使用测试图像目录: {test_images_dir}")
        
        # 索引图像
        engine.index_directory(test_images_dir)
        
        # 搜索示例
        queries = [
            "a photo of a person",
            "outdoor scene",
            "colorful image",
        ]
        
        for query in queries:
            results = engine.search(query, top_k=3)
            print_results(results)
    else:
        print(f"测试图像目录不存在: {test_images_dir}")
        print("请指定您自己的图像目录")
    
    # 查看统计
    stats = engine.get_stats()
    print(f"\n统计信息: {stats}")
    
    # 清理 (可选 - 删除演示集合)
    # engine.clear()
    
    engine.close()


def example_with_descriptions():
    """带描述的索引示例"""
    print("\n" + "=" * 60)
    print("示例 2: 带自定义描述的索引")
    print("=" * 60)
    
    engine = ImageSearchEngine(collection_name='demo_with_desc')
    
    # 假设有以下图像和描述
    # 这里只是演示，实际使用时替换为真实路径
    demo_images = {
        'cat.jpg': '一只橘色的猫咪在阳光下睡觉',
        'dog.jpg': '一只金毛犬在公园里奔跑',
        'beach.jpg': '美丽的海滩日落风景',
    }
    
    # 检查哪些文件存在
    existing_images = []
    descriptions = {}
    
    for img_path, desc in demo_images.items():
        if os.path.exists(img_path):
            existing_images.append(img_path)
            descriptions[img_path] = desc
    
    if existing_images:
        engine.index_images(existing_images, descriptions=descriptions)
        
        # 搜索
        results = engine.search('猫咪', top_k=3)
        print_results(results)
    else:
        print("演示图像不存在，请替换为实际图像路径")
    
    engine.close()


def example_interactive():
    """交互式搜索示例"""
    print("\n" + "=" * 60)
    print("示例 3: 交互式搜索")
    print("=" * 60)
    
    engine = ImageSearchEngine(collection_name='demo_collection')
    
    stats = engine.get_stats()
    print(f"\n当前已索引 {stats['total_images']} 张图像")
    
    if stats['total_images'] == 0:
        print("数据库为空，请先运行基本使用示例或索引您的图像")
        engine.close()
        return
    
    print("\n开始交互式搜索 (输入 'quit' 退出):")
    
    while True:
        try:
            query = input("\n请输入搜索文本: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if query.lower() == 'quit':
            break
        
        if not query:
            continue
        
        results = engine.search(query, top_k=5)
        print_results(results)
    
    print("\n退出交互式搜索")
    engine.close()


def example_batch_search():
    """批量搜索示例"""
    print("\n" + "=" * 60)
    print("示例 4: 批量搜索")
    print("=" * 60)
    
    engine = ImageSearchEngine(collection_name='demo_collection')
    
    stats = engine.get_stats()
    if stats['total_images'] == 0:
        print("数据库为空，请先索引图像")
        engine.close()
        return
    
    # 批量搜索多个查询
    queries = [
        "人物照片",
        "自然风景",
        "室内场景",
    ]
    
    print(f"\n批量搜索 {len(queries)} 个查询...")
    
    batch_results = engine.search_batch(queries, top_k=3)
    
    for query, results in zip(queries, batch_results):
        print(f"\n查询: \"{query}\"")
        print("-" * 40)
        for result in results:
            print(f"  [{result.rank}] {result.image_path} (分数: {result.score:.4f})")
    
    engine.close()


def main():
    """运行所有示例"""
    print("=" * 60)
    print("CLIP 文本搜图示例程序")
    print("=" * 60)
    
    print("\n选择要运行的示例:")
    print("1. 基本使用 - 索引测试图像并搜索")
    print("2. 带描述索引 - 使用自定义描述索引")
    print("3. 交互式搜索 - 命令行交互搜索")
    print("4. 批量搜索 - 一次搜索多个查询")
    print("5. 运行所有示例")
    print("q. 退出")
    
    try:
        choice = input("\n请选择 (1-5 或 q): ").strip()
    except (EOFError, KeyboardInterrupt):
        return
    
    if choice == '1':
        example_basic_usage()
    elif choice == '2':
        example_with_descriptions()
    elif choice == '3':
        example_interactive()
    elif choice == '4':
        example_batch_search()
    elif choice == '5':
        example_basic_usage()
        example_batch_search()
        example_interactive()
    elif choice.lower() == 'q':
        print("退出")
    else:
        print("无效选择")


if __name__ == '__main__':
    main()


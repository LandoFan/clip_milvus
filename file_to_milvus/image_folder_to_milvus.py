"""
若把图像放到word文档内：
# 5. 处理文档
python main.py --dir ./your_documents_folder

# 6. 搜索图像
python main.py --search "你想找的图像描述" --content-type image

若只在文件夹内：
图片文件夹处理工具：将文件夹中的图片上传到Milvus并支持文本检索

使用方法：
    # 上传图片文件夹
    python image_folder_to_milvus.py --upload ./your_image_folder
    
    # 搜索图片
    python image_folder_to_milvus.py --search "一只可爱的猫"
    
    # 搜索并显示图片
    python image_folder_to_milvus.py --search "蓝天白云" --show
"""
import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
from dotenv import load_dotenv
import numpy as np

from clip.vectorizer import CLIPVectorizer
from milvus.milvus_store import MilvusStore

# 加载环境变量
load_dotenv()

# 支持的图片格式
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}


def collect_images(directory: str, recursive: bool = True) -> List[Path]:
    """
    收集目录中的所有图片文件
    
    Args:
        directory: 目录路径
        recursive: 是否递归搜索子目录
        
    Returns:
        图片文件路径列表
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")
    
    images = []
    if recursive:
        for ext in SUPPORTED_FORMATS:
            images.extend(dir_path.rglob(f"*{ext}"))
            images.extend(dir_path.rglob(f"*{ext.upper()}"))
    else:
        for ext in SUPPORTED_FORMATS:
            images.extend(dir_path.glob(f"*{ext}"))
            images.extend(dir_path.glob(f"*{ext.upper()}"))
    
    # 去重
    images = list(set(images))
    return sorted(images)


def upload_images(image_dir: str,
                  vectorizer: CLIPVectorizer,
                  milvus_store: MilvusStore,
                  batch_size: int = 32,
                  recursive: bool = True):
    """
    将图片文件夹上传到Milvus
    
    Args:
        image_dir: 图片目录
        vectorizer: CLIP向量化器
        milvus_store: Milvus存储
        batch_size: 批处理大小
        recursive: 是否递归搜索
    """
    print(f"\n📁 扫描目录: {image_dir}")
    images = collect_images(image_dir, recursive)
    
    if not images:
        print(f"⚠️ 未找到支持的图片文件 (支持格式: {', '.join(SUPPORTED_FORMATS)})")
        return
    
    print(f"✓ 找到 {len(images)} 张图片")
    print("=" * 60)
    
    # 分批处理
    for i in tqdm(range(0, len(images), batch_size), desc="上传图片"):
        batch_paths = images[i:i + batch_size]
        
        # 将图片路径转为字符串
        str_paths = [str(p) for p in batch_paths]
        
        try:
            # 向量化图片
            embeddings = vectorizer.encode_images(str_paths, show_progress=False)
            
            # 存储到Milvus
            milvus_store.insert_images(
                image_paths=str_paths,
                embeddings=embeddings,
                file_path=image_dir,
                file_type="image_folder",
                metadata={"source_dir": image_dir}
            )
        except Exception as e:
            print(f"\n⚠️ 批次处理失败: {e}")
            # 尝试逐个处理
            for path in str_paths:
                try:
                    emb = vectorizer.encode_images([path], show_progress=False)
                    milvus_store.insert_images(
                        image_paths=[path],
                        embeddings=emb,
                        file_path=image_dir,
                        file_type="image_folder",
                        metadata={"source_dir": image_dir}
                    )
                except Exception as e2:
                    print(f"  ✗ 跳过图片 {path}: {e2}")
    
    print(f"\n✓ 图片上传完成!")


def search_images(query_text: str,
                  vectorizer: CLIPVectorizer,
                  milvus_store: MilvusStore,
                  limit: int = 10,
                  show_images: bool = True) -> List[dict]:
    """
    用文本搜索图片
    
    Args:
        query_text: 查询文本（描述你想找的图片）
        vectorizer: CLIP向量化器
        milvus_store: Milvus存储
        limit: 返回结果数量
        show_images: 是否显示图片
        
    Returns:
        搜索结果列表
    """
    print(f"\n🔍 搜索: {query_text}")
    print("=" * 60)
    
    # 将文本转为向量
    query_vector = vectorizer.encode_texts([query_text], show_progress=False)
    
    # 在Milvus中搜索（只搜索图片类型）
    results = milvus_store.search(
        query_vectors=query_vector,
        content_type="image",
        limit=limit
    )
    
    if not results:
        print("未找到匹配的图片")
        return []
    
    print(f"\n找到 {len(results)} 张相关图片:\n")
    
    for i, result in enumerate(results, 1):
        image_path = result['content']
        distance = result['distance']
        # L2距离越小越相似，转换为相似度分数
        similarity = 1 / (1 + distance)
        
        print(f"[{i}] 相似度: {similarity:.4f}")
        print(f"    路径: {image_path}")
        
        if show_images:
            try:
                from PIL import Image
                import matplotlib.pyplot as plt
                
                if Path(image_path).exists():
                    img = Image.open(image_path)
                    plt.figure(figsize=(8, 8))
                    plt.imshow(img)
                    plt.title(f"#{i} 相似度: {similarity:.4f}\n{Path(image_path).name}")
                    plt.axis('off')
                    plt.show()
            except ImportError:
                print("    (需要安装 matplotlib 才能显示图片: pip install matplotlib)")
            except Exception as e:
                print(f"    ⚠️ 无法显示图片: {e}")
        print()
    
    return results


def interactive_search(vectorizer: CLIPVectorizer,
                       milvus_store: MilvusStore,
                       show_images: bool = False):
    """
    交互式搜索模式
    """
    print("\n" + "=" * 60)
    print("🎯 交互式图片搜索")
    print("输入描述来搜索图片，输入 'q' 或 'quit' 退出")
    print("=" * 60)
    
    while True:
        try:
            query = input("\n🔍 请输入搜索描述 > ").strip()
            if query.lower() in ['q', 'quit', 'exit', '退出']:
                print("再见！")
                break
            if not query:
                continue
            
            search_images(query, vectorizer, milvus_store, limit=5, show_images=show_images)
            
        except KeyboardInterrupt:
            print("\n再见！")
            break


def main():
    parser = argparse.ArgumentParser(
        description="图片文件夹 -> Milvus 向量检索工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 上传图片文件夹到Milvus
  python image_folder_to_milvus.py --upload ./my_photos
  
  # 搜索图片
  python image_folder_to_milvus.py --search "蓝天白云下的山"
  
  # 搜索并显示图片
  python image_folder_to_milvus.py --search "可爱的小猫" --show
  
  # 交互式搜索模式
  python image_folder_to_milvus.py --interactive
  
  # 指定返回数量
  python image_folder_to_milvus.py --search "红色的花" --limit 20
        """
    )
    
    # 操作模式
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--upload', type=str, help='上传图片文件夹路径')
    mode_group.add_argument('--search', type=str, help='搜索图片（输入文本描述）')
    mode_group.add_argument('--interactive', action='store_true', help='交互式搜索模式')
    
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
                        default=os.getenv('MILVUS_COLLECTION', 'clip_images'),
                        help='Milvus集合名称 (默认: clip_images)')
    
    parser.add_argument('--limit', type=int, default=10,
                        help='搜索返回数量 (默认: 10)')
    
    parser.add_argument('--show', action='store_true',
                        help='显示搜索到的图片')
    
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批处理大小 (默认: 32)')
    
    parser.add_argument('--no-recursive', action='store_true',
                        help='不递归搜索子目录')
    
    parser.add_argument('--drop-collection', action='store_true',
                        help='删除已存在的集合并重建')
    
    args = parser.parse_args()
    
    # 初始化组件
    print("🚀 初始化组件...")
    print("=" * 60)
    
    try:
        # 初始化CLIP向量化器
        vectorizer = CLIPVectorizer(server_url=args.clip_server)
        embedding_dim = vectorizer.get_embedding_dimension()
        print(f"✓ CLIP服务器连接成功，向量维度: {embedding_dim}")
        
        # 初始化Milvus存储
        milvus_store = MilvusStore(
            host=args.milvus_host,
            port=args.milvus_port,
            collection_name=args.collection,
            embedding_dim=embedding_dim,
            drop_existing=args.drop_collection
        )
        
        stats = milvus_store.get_stats()
        print(f"✓ Milvus集合: {stats['collection_name']}")
        print(f"✓ 现有实体数: {stats['num_entities']}")
        
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 执行操作
    if args.upload:
        upload_images(
            image_dir=args.upload,
            vectorizer=vectorizer,
            milvus_store=milvus_store,
            batch_size=args.batch_size,
            recursive=not args.no_recursive
        )
        stats = milvus_store.get_stats()
        print(f"\n✓ 完成! 当前集合实体数: {stats['num_entities']}")
    
    elif args.search:
        search_images(
            query_text=args.search,
            vectorizer=vectorizer,
            milvus_store=milvus_store,
            limit=args.limit,
            show_images=args.show
        )
    
    elif args.interactive:
        interactive_search(
            vectorizer=vectorizer,
            milvus_store=milvus_store,
            show_images=args.show
        )


if __name__ == '__main__':
    main()


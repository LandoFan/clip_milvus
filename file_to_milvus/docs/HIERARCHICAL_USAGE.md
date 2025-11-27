# 层次化分段和混合检索使用指南

## 概述

本系统实现了两种重要的改进：

1. **父子分段（Hierarchical Chunking）**: 保持文档的层次结构，建立父子关系
2. **混合检索（Hybrid Search）**: 结合向量检索和关键词检索，提高检索准确性

## 父子分段

### 概念

父子分段保持文档的层次结构，例如：

```
文档 (Document)
  └── 章节1 (Section)
      ├── 小节1.1 (Subsection)
      │   └── 段落1 (Paragraph)
      │   └── 段落2 (Paragraph)
      └── 小节1.2 (Subsection)
          └── 段落3 (Paragraph)
```

### 优势

- **保持上下文**: 每个块知道它的父块和子块
- **更好的检索**: 可以检索相关块及其上下文
- **结构理解**: 理解文档的层次结构

### 块类型

- `document`: 文档级别
- `section`: 章节级别
- `subsection`: 小节级别
- `paragraph`: 段落级别
- `sentence`: 句子级别（可选）

## 混合检索

### 概念

混合检索结合两种检索方式：

1. **向量检索（Embedding Search）**: 基于语义相似度
2. **关键词检索（BM25）**: 基于关键词匹配

### 优势

- **更好的准确性**: 结合语义和关键词匹配
- **灵活性**: 可以调整两种检索的权重
- **互补性**: 向量检索捕获语义，关键词检索捕获精确匹配

### 参数说明

- `alpha`: 向量检索权重 (0-1)
  - `alpha=1.0`: 纯向量检索
  - `alpha=0.0`: 纯关键词检索
  - `alpha=0.7`: 推荐值，向量70%，关键词30%

## 使用方法

### 1. 处理文件（使用层次化分段）

```bash
# 处理单个文件
python main_hierarchical.py --file document.docx

# 指定最大块大小
python main_hierarchical.py --file document.docx --max-chunk-size 300

# 批量处理
python main_hierarchical.py --dir ./documents
```

### 2. 混合检索

```bash
# 基本搜索
python main_hierarchical.py --search "查询关键词"

# 调整混合检索权重（向量70%，关键词30%）
python main_hierarchical.py --search "查询" --alpha 0.7

# 更多向量检索（80%向量，20%关键词）
python main_hierarchical.py --search "查询" --alpha 0.8

# 更多关键词检索（50%向量，50%关键词）
python main_hierarchical.py --search "查询" --alpha 0.5

# 不使用层次化检索（仅混合检索）
python main_hierarchical.py --search "查询" --no-hierarchical
```

### 3. 层次化检索

层次化检索会自动：
- 如果找到匹配的子块，也返回父块（上下文）
- 如果找到匹配的父块，也返回子块（详细内容）

```bash
# 使用层次化检索（默认）
python main_hierarchical.py --search "查询" --alpha 0.7

# 关闭层次化检索
python main_hierarchical.py --search "查询" --no-hierarchical
```

## 代码示例

### Python代码使用

```python
from hierarchical_parser import HierarchicalWordParser
from vectorizer import CLIPVectorizer
from hierarchical_store import HierarchicalMilvusStore

# 1. 初始化
vectorizer = CLIPVectorizer(server_url="grpc://0.0.0.0:51000")
milvus_store = HierarchicalMilvusStore(
    host="localhost",
    port=19530,
    collection_name="my_collection",
    embedding_dim=512
)

# 2. 解析文件（层次化）
parser = HierarchicalWordParser(max_chunk_size=500)
hierarchical_content = parser.parse("document.docx")

# 3. 向量化
texts = [chunk.content for chunk in hierarchical_content.chunks]
embeddings = vectorizer.encode_texts(texts)

# 4. 存储
milvus_store.insert_hierarchical_chunks(
    hierarchical_content=hierarchical_content,
    embeddings=embeddings,
    file_path="document.docx",
    file_type="word"
)

# 5. 混合检索
query_text = "查询关键词"
query_vector = vectorizer.encode_texts([query_text])[0]

results = milvus_store.hybrid_search(
    query_text=query_text,
    query_vector=query_vector,
    limit=10,
    alpha=0.7  # 70%向量，30%关键词
)

# 6. 层次化混合检索
results = milvus_store.hierarchical_search(
    query_text=query_text,
    query_vector=query_vector,
    limit=10,
    alpha=0.7,
    include_children=True,  # 包含子块
    include_parent=True     # 包含父块
)
```

## 检索结果说明

检索结果包含以下字段：

```python
{
    'id': 123,                    # Milvus ID
    'distance': 0.234,            # 相似度分数（越小越好）
    'content': '文本内容...',      # 块内容
    'chunk_index': 5,             # 块索引
    'parent_id': 2,               # 父块ID
    'chunk_type': 'paragraph',    # 块类型
    'level': 2,                   # 层级深度
    'metadata': {                 # 元数据
        'children_ids': [6, 7],   # 子块IDs
        ...
    }
}
```

## 性能调优

### 块大小

- **小块（200-300字符）**: 更精确的匹配，但可能丢失上下文
- **大块（500-800字符）**: 更多上下文，但可能包含无关内容
- **推荐**: 300-500字符

### 混合检索权重

- **向量为主（alpha=0.7-0.8）**: 适合语义搜索
- **关键词为主（alpha=0.3-0.5）**: 适合精确匹配
- **平衡（alpha=0.5-0.7）**: 适合一般搜索

### 层次化检索

- **include_parent=True**: 有助于理解上下文
- **include_children=True**: 有助于获取详细信息
- 两者结合使用效果最好

## 与普通版本的对比

| 特性 | 普通版本 | 层次化版本 |
|------|---------|-----------|
| 分段方式 | 简单分块 | 父子关系分段 |
| 检索方式 | 向量检索 | 混合检索 |
| 上下文理解 | 无 | 支持父子关系 |
| 检索准确性 | 中等 | 更高 |
| 处理速度 | 快 | 稍慢（但更准确） |

## 注意事项

1. **存储空间**: 层次化存储需要更多空间（存储父子关系）
2. **检索时间**: 混合检索比纯向量检索稍慢
3. **alpha参数**: 根据你的数据调整，建议从0.7开始
4. **块大小**: 根据文档类型调整，技术文档可以稍大，对话内容可以稍小

## 故障排除

### 检索结果不准确

- 尝试调整`alpha`参数
- 检查块大小是否合适
- 确认CLIP模型适合你的语言

### 层次关系不完整

- 检查文档是否有清晰的标题结构
- 调整`max_chunk_size`参数
- 查看解析的块结构是否正确

## 下一步

- 尝试不同的`alpha`值找到最佳平衡
- 根据文档类型调整块大小
- 使用层次化检索获取更完整的上下文


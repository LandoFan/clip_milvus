# 完整使用指南

## 📚 目录

1. [系统概述](#系统概述)
2. [快速开始](#快速开始)
3. [核心功能使用](#核心功能使用)
4. [API参考](#api参考)
5. [常见场景](#常见场景)
6. [故障排除](#故障排除)

---

## 系统概述

本系统提供了完整的文档向量化和检索解决方案：

- ✅ Word/Markdown文件解析
- ✅ CLIP向量化（文本和图像）
- ✅ 层次化分段（父子关系）
- ✅ 混合检索（向量+关键词）
- ✅ Milvus向量数据库存储
- ✅ 统一的Python API

---

## 快速开始

### 1. 安装依赖

```bash
cd file_to_milvus
pip install -r requirements.txt
```

### 2. 启动服务

**启动CLIP服务器**（新终端窗口）:
```bash
python -m clip_server
```

**启动Milvus数据库**（Docker）:
```bash
docker run -d --name milvus-standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  -v $(pwd)/milvus_data:/var/lib/milvus \
  milvusdb/milvus:latest
```

### 3. 使用知识库

```python
from knowledge_base import KnowledgeBase

# 创建知识库
kb = KnowledgeBase()

# 添加文档
kb.add_document("document.docx")

# 查询
results = kb.query("查询关键词", top_k=10)
```

---

## 核心功能使用

### 1. 添加文档

#### 单个文档
```python
from knowledge_base import KnowledgeBase

kb = KnowledgeBase()

# 自动检测文件类型
result = kb.add_document("report.docx")
print(result)  # {'success': True, 'chunks_count': 15, ...}

# 指定文件类型
result = kb.add_document("guide.md", file_type="markdown")
```

#### 批量添加
```python
# 批量添加多个文档
file_paths = ["doc1.docx", "doc2.md", "doc3.docx"]
results = kb.add_documents(file_paths, show_progress=True)

for result in results:
    if result['success']:
        print(f"✓ {result['file_path']}: {result['chunks_count']} 块")
    else:
        print(f"✗ {result['file_path']}: {result['message']}")
```

### 2. 查询知识库

#### 基本查询
```python
# 简单查询
results = kb.query("机器学习算法", top_k=5)

for result in results:
    print(f"内容: {result['content'][:100]}...")
    print(f"相似度: {result['distance']:.4f}")
    print(f"文件: {result['file_path']}")
    print()
```

#### 高级查询

**调整混合检索权重**:
```python
# 更多向量检索（语义相似）
results = kb.query("深度学习", alpha=0.8, top_k=5)

# 更多关键词检索（精确匹配）
results = kb.query("深度学习", alpha=0.3, top_k=5)

# 平衡（默认）
results = kb.query("深度学习", alpha=0.7, top_k=5)
```

**层次化检索（包含上下文）**:
```python
# 自动包含父块和子块
results = kb.query(
    "查询关键词",
    top_k=5,
    hierarchical=True,      # 使用层次化检索
    include_parent=True,    # 包含父块
    include_children=True   # 包含子块
)
```

**过滤查询**:
```python
# 只查询Word文档
results = kb.query(
    "查询",
    filter_expr='file_type == "word"'
)

# 只查询特定文件
results = kb.query(
    "查询",
    filter_expr='file_path == "document.docx"'
)
```

#### 批量查询
```python
queries = ["查询1", "查询2", "查询3"]
all_results = kb.query_batch(queries, top_k=5)

for i, results in enumerate(all_results):
    print(f"查询 {i+1}: 找到 {len(results)} 个结果")
```

### 3. 文档管理

#### 列出所有文档
```python
documents = kb.list_documents()
print(f"知识库中共有 {len(documents)} 个文档:")
for doc in documents:
    print(f"  - {doc}")
```

#### 删除文档
```python
result = kb.delete_document("document.docx")
if result['success']:
    print(f"删除了 {result['deleted_count']} 条记录")
```

#### 查看统计信息
```python
stats = kb.get_stats()
print(f"集合名称: {stats['collection_name']}")
print(f"实体数量: {stats['num_entities']}")
print(f"CLIP服务器: {stats['clip_server']}")
print(f"Milvus地址: {stats['milvus_host']}:{stats['milvus_port']}")
```

### 4. 重建索引

如果混合检索结果不准确，可以重建索引：

```python
kb.rebuild_hybrid_index()
```

---

## API参考

### KnowledgeBase类

#### 初始化

```python
kb = KnowledgeBase(
    clip_server="grpc://0.0.0.0:51000",    # CLIP服务器地址
    milvus_host="localhost",                # Milvus主机
    milvus_port=19530,                      # Milvus端口
    collection_name="my_kb",                # 集合名称
    max_chunk_size=500,                     # 最大块大小
    auto_reconnect=True                     # 自动重连
)
```

#### 主要方法

##### `add_document(file_path, file_type=None)`

添加文档到知识库。

**参数**:
- `file_path`: 文件路径（str或Path）
- `file_type`: 文件类型（'word'或'markdown'），None则自动检测

**返回**:
```python
{
    'success': True,
    'chunks_count': 15,
    'file_path': 'document.docx',
    'file_type': 'word',
    'message': '成功添加 15 个块'
}
```

##### `query(query_text, top_k=10, alpha=0.7, hierarchical=True, ...)`

查询知识库。

**参数**:
- `query_text`: 查询文本
- `top_k`: 返回结果数量
- `alpha`: 混合检索权重（0-1）
- `hierarchical`: 是否使用层次化检索
- `include_children`: 是否包含子块
- `include_parent`: 是否包含父块
- `content_type`: 内容类型筛选
- `filter_expr`: Milvus过滤表达式

**返回**: 结果列表
```python
[
    {
        'content': '文本内容...',
        'distance': 0.234,
        'chunk_type': 'paragraph',
        'level': 2,
        'parent_id': 5,
        'file_path': 'document.docx',
        'chunk_index': 10,
        'metadata': {...}
    },
    ...
]
```

---

## 常见场景

### 场景1: 构建知识库

```python
from knowledge_base import KnowledgeBase

kb = KnowledgeBase(collection_name="my_knowledge_base")

# 添加所有文档
import os
doc_dir = "./documents"
for filename in os.listdir(doc_dir):
    if filename.endswith(('.docx', '.md')):
        file_path = os.path.join(doc_dir, filename)
        result = kb.add_document(file_path)
        print(f"{filename}: {result['message']}")
```

### 场景2: RAG系统集成

```python
from knowledge_base import KnowledgeBase

class RAGSystem:
    def __init__(self):
        self.kb = KnowledgeBase()
    
    def retrieve_context(self, question: str, top_k: int = 3):
        """检索相关上下文"""
        results = self.kb.query(
            question,
            top_k=top_k,
            hierarchical=True,
            alpha=0.7
        )
        
        # 组合上下文
        context = "\n\n".join([
            f"[来源: {r['file_path']}]\n{r['content']}"
            for r in results
        ])
        
        return context
    
    def answer(self, question: str):
        """回答问题"""
        context = self.retrieve_context(question)
        
        # 使用LLM生成答案
        prompt = f"""基于以下上下文回答问题：

{context}

问题：{question}

答案："""
        
        # answer = llm.generate(prompt)
        return prompt  # 示例
```

### 场景3: 文档检索服务

```python
from knowledge_base import KnowledgeBase
from flask import Flask, jsonify, request

app = Flask(__name__)
kb = KnowledgeBase()

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', '')
    top_k = data.get('top_k', 10)
    
    results = kb.query(query, top_k=top_k)
    
    return jsonify({
        'results': [
            {
                'content': r['content'],
                'score': r['distance'],
                'source': r['file_path']
            }
            for r in results
        ]
    })

if __name__ == '__main__':
    app.run(port=5000)
```

---

## 故障排除

### 问题1: CLIP服务器连接失败

**错误**: `Connection refused` 或 `无法连接到CLIP服务器`

**解决**:
1. 检查CLIP服务器是否运行: `ps aux | grep clip_server`
2. 检查端口是否正确（默认51000）
3. 检查防火墙设置
4. 尝试使用完整地址: `grpc://localhost:51000`

### 问题2: Milvus连接失败

**错误**: `Failed to connect to Milvus`

**解决**:
1. 检查Milvus容器: `docker ps | grep milvus`
2. 检查端口19530是否可访问
3. 查看Milvus日志: `docker logs milvus-standalone`
4. 确认Milvus已完全启动

### 问题3: 混合检索结果不准确

**问题**: 查询结果相关性不高

**解决**:
1. 调整alpha参数
   ```python
   # 尝试不同的权重
   results = kb.query("查询", alpha=0.5)  # 更平衡
   results = kb.query("查询", alpha=0.8)  # 更语义
   ```

2. 重建混合检索索引
   ```python
   kb.rebuild_hybrid_index()
   ```

3. 检查块大小是否合适
   - 技术文档: 500-800字符
   - 一般文档: 300-500字符

### 问题4: 内存不足

**错误**: `Out of memory` 或 `MemoryError`

**解决**:
1. 减少批处理大小
2. 减少max_chunk_size
3. 分批处理文档
4. 使用更大的机器

### 问题5: 文件解析失败

**错误**: `无法解析文件` 或 `提取不到内容`

**解决**:
1. 检查文件格式是否正确
2. Word文档必须是`.docx`格式（不是`.doc`）
3. 检查文件是否损坏
4. 查看错误日志了解详情

---

## 最佳实践

### 1. 块大小选择

- **技术文档**: 500-800字符（保持完整性）
- **一般文档**: 300-500字符（平衡）
- **对话内容**: 200-300字符（更精确）

### 2. 混合检索权重

- **语义搜索为主**: alpha=0.7-0.8
- **精确匹配为主**: alpha=0.3-0.5
- **平衡**: alpha=0.5-0.7（推荐）

### 3. 层次化检索

- **需要上下文**: 使用`hierarchical=True`
- **仅精确匹配**: 使用`hierarchical=False`

### 4. 性能优化

- 批量添加文档（使用`add_documents()`）
- 合理设置top_k（不要太大）
- 使用过滤器减少搜索空间

---

## 完整示例

```python
from knowledge_base import KnowledgeBase

# 1. 初始化
kb = KnowledgeBase(
    clip_server="grpc://0.0.0.0:51000",
    milvus_host="localhost",
    collection_name="production_kb"
)

# 2. 添加文档
kb.add_document("技术文档.docx")
kb.add_document("产品说明.md")

# 3. 查询
results = kb.query(
    "如何使用API",
    top_k=5,
    alpha=0.7,
    hierarchical=True
)

# 4. 处理结果
for result in results:
    print(f"文件: {result['file_path']}")
    print(f"内容: {result['content']}")
    print(f"相似度: {result['distance']:.4f}")
    print(f"层级: {result['level']}")
    print("-" * 60)

# 5. 查看统计
stats = kb.get_stats()
print(f"知识库中共有 {stats['num_entities']} 个文档块")
```

---

## 更多资源

- 📖 完整文档: `README.md`
- 🚀 快速开始: `QUICKSTART.md`
- 🏗️ 架构说明: `ARCHITECTURE.md`
- 📝 API示例: `kb_api_example.py`


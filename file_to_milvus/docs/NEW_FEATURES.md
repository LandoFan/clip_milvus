# 新功能说明：父子分段和混合检索

## 新增功能

### 1. 父子分段（Hierarchical Chunking）✅

**文件**: `hierarchical_parser.py`

实现了保持文档层次结构的分段方式：

- **层次结构**: Document → Section → Subsection → Paragraph
- **父子关系**: 每个块都知道它的父块和子块
- **元数据**: 存储块的类型、层级深度等信息

**优势**:
- 保持文档的逻辑结构
- 支持基于上下文的检索
- 更好的语义理解

### 2. 混合检索（Hybrid Search）✅

**文件**: `hybrid_search.py`

结合向量检索和关键词检索：

- **BM25算法**: 关键词检索
- **向量检索**: 语义相似度检索
- **加权组合**: 可调整两种检索的权重

**优势**:
- 更高的检索准确性
- 结合语义和精确匹配
- 灵活的权重调整

### 3. 层次化存储（Hierarchical Storage）✅

**文件**: `hierarchical_store.py`

扩展了Milvus存储以支持：

- **父子关系**: 存储块的父子关系
- **层次化检索**: 考虑父子关系的检索
- **上下文扩展**: 自动包含相关的父块或子块

### 4. 新的主程序 ✅

**文件**: `main_hierarchical.py`

提供了完整的命令行接口：

- 支持层次化分段
- 支持混合检索
- 支持层次化检索

## 新增文件列表

```
file_to_milvus/
├── hierarchical_parser.py      # 层次化解析器（新增）
├── hybrid_search.py            # 混合检索（新增）
├── hierarchical_store.py       # 层次化存储（新增）
├── main_hierarchical.py        # 层次化主程序（新增）
├── HIERARCHICAL_USAGE.md       # 使用指南（新增）
└── NEW_FEATURES.md             # 本文档（新增）
```

## 使用方式对比

### 普通版本

```bash
# 处理文件
python main.py --file document.docx

# 搜索
python main.py --search "查询"
```

### 层次化版本（推荐）

```bash
# 处理文件（使用父子分段）
python main_hierarchical.py --file document.docx

# 混合检索
python main_hierarchical.py --search "查询" --alpha 0.7

# 层次化混合检索（自动包含上下文）
python main_hierarchical.py --search "查询"
```

## 核心改进

### 分段方式

**之前**: 简单分块，丢失结构信息
```python
chunk1 = "这是第一段..."
chunk2 = "这是第二段..."
# 不知道chunk1和chunk2的关系
```

**现在**: 层次化分段，保持结构
```python
document
  ├── section1 (chunk1)
  │   ├── paragraph1 (chunk2)  # parent: chunk1
  │   └── paragraph2 (chunk3)  # parent: chunk1
  └── section2 (chunk4)
      └── paragraph3 (chunk5)  # parent: chunk4
```

### 检索方式

**之前**: 纯向量检索
```python
# 仅基于向量相似度
results = vector_search(query_vector)
```

**现在**: 混合检索
```python
# 向量检索 + 关键词检索
vector_results = vector_search(query_vector)
keyword_results = bm25_search(query_text)
final_results = combine(vector_results, keyword_results, alpha=0.7)
```

## 性能对比

| 指标 | 普通版本 | 层次化版本 |
|------|---------|-----------|
| 分段准确性 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 检索准确性 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 上下文理解 | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 处理速度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 存储空间 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

## 适用场景

### 使用普通版本当：
- 文档结构简单
- 不需要上下文理解
- 追求最快速度
- 存储空间有限

### 使用层次化版本当：
- 文档有清晰的层次结构（章节、小节等）
- 需要理解上下文
- 追求更高检索准确性
- 需要混合检索的优势

## 迁移指南

如果你已经有使用普通版本的数据：

1. **新数据**: 使用 `main_hierarchical.py` 处理新文件
2. **旧数据**: 可以继续使用 `main.py` 检索
3. **完全迁移**: 重新处理所有文件使用层次化版本

两个版本可以共存，使用不同的集合名称即可：

```bash
# 普通版本
python main.py --file doc.docx --collection clip_documents

# 层次化版本
python main_hierarchical.py --file doc.docx --collection clip_documents_hierarchical
```

## 配置建议

### 块大小
- **技术文档**: 500-800字符
- **一般文档**: 300-500字符
- **对话内容**: 200-300字符

### 混合检索权重（alpha）
- **语义搜索为主**: 0.7-0.8
- **精确匹配为主**: 0.3-0.5
- **平衡**: 0.5-0.7

### 层次化检索
- **需要完整上下文**: 开启（默认）
- **仅需精确匹配**: 使用 `--no-hierarchical`

## 技术细节

### BM25参数
- `k1=1.5`: 词频饱和度（默认）
- `b=0.75`: 长度归一化（默认）

### 层次结构
- 支持最多4层：Document → Section → Subsection → Paragraph
- 每层都有类型标识
- 自动建立父子关系映射

### 存储结构
```
Milvus字段:
- parent_id: 父块ID
- chunk_type: 块类型
- level: 层级深度
- metadata: JSON格式的额外信息（包含children_ids）
```

## 下一步

1. ✅ 阅读 `HIERARCHICAL_USAGE.md` 了解详细使用方法
2. ✅ 尝试不同的 `alpha` 值找到最佳配置
3. ✅ 根据文档类型调整块大小
4. ✅ 使用层次化检索获取更好的结果

## 问题反馈

如果遇到问题：
1. 查看 `HIERARCHICAL_USAGE.md` 的故障排除部分
2. 检查日志输出
3. 确认所有依赖已安装


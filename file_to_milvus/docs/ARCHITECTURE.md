# 系统架构说明

## 概述

本系统将Word和Markdown文档中的文本和图像进行向量化，并存储到Milvus向量数据库中，支持语义搜索功能。

## 系统架构图

```
┌─────────────────┐
│   Word/MD文件    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  File Parser    │ ← 提取文本和图像
│ (file_parser.py)│
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐  ┌───────┐
│ 文本块 │  │ 图像  │
└───┬───┘  └───┬───┘
    │          │
    └────┬─────┘
         │
         ▼
┌─────────────────┐
│  CLIP Vectorizer│ ← 向量化
│ (vectorizer.py) │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐  ┌───────┐
│文本向量│  │图像向量│
└───┬───┘  └───┬───┘
    │          │
    └────┬─────┘
         │
         ▼
┌─────────────────┐
│  Milvus Store   │ ← 存储到数据库
│ (milvus_store.py)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Milvus Database│
└─────────────────┘
```

## 组件说明

### 1. File Parser (`file_parser.py`)

**职责**: 解析Word和Markdown文件，提取文本和图像

**主要类**:
- `WordParser`: 解析.docx文件
- `MarkdownParser`: 解析.md/.markdown文件
- `FileParserFactory`: 根据文件类型创建相应的解析器

**输出**: `ExtractedContent` 对象，包含：
- `text_chunks`: 文本块列表
- `images`: 图像列表（包含二进制数据和路径）
- `metadata`: 文件元数据

### 2. CLIP Vectorizer (`vectorizer.py`)

**职责**: 使用CLIP-as-service对文本和图像进行向量化

**主要类**:
- `CLIPVectorizer`: CLIP向量化器

**功能**:
- 连接CLIP服务器
- 对文本列表进行编码
- 对图像列表进行编码
- 返回numpy数组格式的向量

**依赖**: 
- CLIP服务器必须运行
- clip-client库

### 3. Milvus Store (`milvus_store.py`)

**职责**: 管理Milvus数据库的存储和检索

**主要类**:
- `MilvusStore`: Milvus存储管理器

**功能**:
- 创建集合（Collection）
- 插入文本向量
- 插入图像向量
- 向量相似度搜索
- 获取统计信息

**数据模型**:
```
Collection Schema:
├── id (INT64, Primary Key, Auto ID)
├── content_type (VARCHAR) - 'text' or 'image'
├── content (VARCHAR) - 原始内容
├── embedding (FLOAT_VECTOR) - 向量嵌入
├── file_path (VARCHAR) - 源文件路径
├── file_type (VARCHAR) - 'word' or 'markdown'
├── chunk_index (INT64) - 块索引
├── metadata (VARCHAR) - JSON格式元数据
└── created_at (VARCHAR) - 创建时间
```

### 4. Main Program (`main.py`)

**职责**: 整合所有组件，提供命令行接口

**主要功能**:
- 处理单个文件
- 批量处理目录
- 搜索功能
- 配置管理

## 数据流程

### 处理流程

1. **文件输入** → 用户提供Word或Markdown文件
2. **文件解析** → 提取文本块和图像
3. **向量化** → 使用CLIP模型生成向量
4. **存储** → 保存到Milvus数据库

### 搜索流程

1. **查询输入** → 用户输入搜索文本
2. **查询向量化** → 将查询文本转换为向量
3. **相似度搜索** → 在Milvus中查找相似向量
4. **结果返回** → 返回最相似的内容

## 技术栈

### 文件解析
- `python-docx`: Word文档解析
- `markdown`: Markdown解析
- `beautifulsoup4`: HTML解析
- `Pillow`: 图像处理

### 向量化
- `clip-client`: CLIP-as-service客户端
- `numpy`: 向量操作

### 数据库
- `pymilvus`: Milvus Python客户端

### 其他
- `tqdm`: 进度条
- `python-dotenv`: 环境变量管理

## 性能考虑

### 批量处理
- 支持批量向量化，提高效率
- 文本和图像分别批处理

### 内存管理
- 大文件分块处理
- 图像数据及时释放

### 数据库优化
- 使用索引加速搜索
- 批量插入数据

## 扩展性

### 添加新文件格式
1. 在 `file_parser.py` 中创建新解析器类
2. 实现 `parse()` 方法
3. 在 `FileParserFactory` 中注册

### 更换向量模型
- 修改CLIP服务器配置
- 调整向量维度（自动检测）

### 更换数据库
- 实现新的存储类
- 保持相同接口

## 安全考虑

- 文件路径验证
- 输入内容大小限制
- 数据库连接安全
- 错误处理机制

## 故障恢复

- 文件处理失败不影响其他文件
- 数据库操作事务性
- 详细的错误日志

## 监控和日志

- 处理进度显示
- 错误信息输出
- 统计信息展示

## 未来改进方向

- [ ] 支持更多文件格式（PDF, HTML等）
- [ ] 增量更新功能
- [ ] 分布式处理支持
- [ ] Web界面
- [ ] 更多搜索选项（日期、文件类型过滤等）
- [ ] 向量缓存机制
- [ ] 异步处理支持


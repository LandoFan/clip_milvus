# Word和Markdown文件向量化存储系统

这是一个完整的解决方案，用于将Word (.docx) 和Markdown (.md) 文件中的文本和图像提取出来，使用CLIP-as-service进行向量化，然后存储到Milvus向量数据库中。

## 功能特性

- ✅ 支持Word文档 (.docx) 解析
- ✅ 支持Markdown文档 (.md, .markdown) 解析
- ✅ 自动提取文档中的文本和图像
- ✅ 使用CLIP模型对文本和图像进行向量化
- ✅ 存储到Milvus向量数据库
- ✅ 支持语义搜索
- ✅ 支持批量处理
- ✅ 支持递归处理目录

## 系统要求

1. **CLIP-as-service服务器** - 需要先启动CLIP服务器
2. **Milvus数据库** - 需要安装并运行Milvus
3. **Python 3.7+**

## 安装步骤

### 1. 安装依赖

```bash
cd file_to_milvus
pip install -r requirements.txt
```

### 2. 启动CLIP服务器

在另一个终端窗口中，启动CLIP服务器：

```bash
# 安装clip-server (如果还没安装)
pip install clip-server

# 启动服务器
python -m clip_server
```

默认端口是 `51000`。

### 3. 启动Milvus数据库

确保Milvus数据库正在运行。可以使用Docker快速启动：

```bash
docker run -d \
  --name milvus-standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  -v $(pwd)/milvus_data:/var/lib/milvus \
  milvusdb/milvus:latest
```

或参考 [Milvus官方文档](https://milvus.io/docs/install_standalone-docker.md) 进行安装。

### 4. 配置环境变量 (可选)

复制 `.env.example` 为 `.env` 并根据需要修改：

```bash
cp .env.example .env
```

编辑 `.env` 文件：
```env
CLIP_SERVER=grpc://0.0.0.0:51000
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=clip_documents
```

## 使用方法

### 处理单个文件

```bash
# 处理Word文档
python main.py --file document.docx

# 处理Markdown文档
python main.py --file README.md
```

### 批量处理目录

```bash
# 处理目录中的所有文件（递归子目录）
python main.py --dir ./documents

# 只处理当前目录，不递归子目录
python main.py --dir ./documents --no-recursive
```

### 搜索

```bash
# 搜索文本内容
python main.py --search "查询关键词"

# 只搜索文本类型
python main.py --search "查询关键词" --content-type text

# 只搜索图像类型
python main.py --search "查询关键词" --content-type image

# 指定返回结果数量
python main.py --search "查询关键词" --limit 20
```

### 高级选项

```bash
# 指定CLIP服务器地址
python main.py --file document.docx --clip-server grpc://192.168.1.100:51000

# 指定Milvus配置
python main.py --file document.docx --milvus-host 192.168.1.101 --milvus-port 19530

# 删除并重建集合
python main.py --dir ./documents --drop-collection
```

## 项目结构

```
file_to_milvus/
├── file_parser.py      # 文件解析器（Word和Markdown）
├── vectorizer.py       # CLIP向量化服务
├── milvus_store.py     # Milvus数据库存储
├── main.py             # 主程序
├── requirements.txt    # 依赖包
├── .env.example        # 环境变量示例
└── README.md           # 本文档
```

## 工作原理

1. **文件解析**: 
   - `file_parser.py` 解析Word或Markdown文件
   - 提取文本块和图像

2. **向量化**:
   - `vectorizer.py` 使用CLIP-as-service客户端
   - 对文本和图像分别进行向量化编码

3. **存储**:
   - `milvus_store.py` 将向量存储到Milvus
   - 同时保存元数据（文件路径、类型、内容等）

4. **搜索**:
   - 将查询文本向量化
   - 在Milvus中进行相似度搜索

## 数据格式

存储到Milvus的数据包含以下字段：

- `id`: 主键（自动生成）
- `content_type`: 内容类型 ('text' 或 'image')
- `content`: 原始内容（文本或图像路径）
- `embedding`: 向量嵌入（512维或根据模型）
- `file_path`: 源文件路径
- `file_type`: 文件类型 ('word' 或 'markdown')
- `chunk_index`: 文本块或图像索引
- `metadata`: JSON格式的额外元数据
- `created_at`: 创建时间

## 示例

### 示例1: 处理文档目录

```bash
python main.py --dir ./my_documents
```

输出：
```
初始化组件...
============================================================
✓ Connected to CLIP server: grpc://0.0.0.0:51000
✓ 向量维度: 512
✓ Connected to Milvus: localhost:19530
✓ Using existing collection: clip_documents
✓ Milvus集合: clip_documents
✓ 现有实体数: 150

找到 5 个文件需要处理
============================================================
处理文件: ./my_documents/doc1.docx
------------------------------------------------------------
  提取到 12 个文本块
  提取到 3 个图像
  向量化文本...
✓ Inserted 12 text embeddings
  向量化图像...
✓ Inserted 3 image embeddings
  ✓ 文件处理完成

✓ 完成! 当前集合实体数: 165
```

### 示例2: 搜索文档

```bash
python main.py --search "机器学习算法"
```

输出：
```
搜索: 机器学习算法
============================================================

找到 5 个结果:

结果 1:
  相似度: 0.2341
  类型: text
  文件: ./my_documents/ml_guide.docx
  内容: 本文介绍了常用的机器学习算法，包括监督学习、无监督学习和强化学习...

结果 2:
  相似度: 0.2567
  类型: text
  文件: ./my_documents/ai_basics.md
  内容: 机器学习是人工智能的核心领域之一...
```

## 注意事项

1. **CLIP服务器**: 确保CLIP服务器正在运行，否则无法进行向量化
2. **Milvus连接**: 确保Milvus数据库可访问
3. **内存使用**: 处理大量文件时注意内存使用
4. **图像路径**: Markdown中的图像路径需要是相对路径或绝对路径
5. **文件大小**: 文本内容会被截断到65535字符

## 故障排除

### CLIP服务器连接失败

```
✓ Connected to CLIP server: grpc://0.0.0.0:51000
```

如果连接失败，检查：
- CLIP服务器是否正在运行
- 服务器地址和端口是否正确
- 防火墙设置

### Milvus连接失败

如果无法连接到Milvus：
- 检查Milvus是否正在运行: `docker ps`
- 检查端口是否正确
- 检查网络连接

### 模型未找到

如果遇到模型相关错误：
- 确保CLIP服务器已正确启动
- 检查CLIP模型是否已下载

## 开发

### 添加新的文件格式支持

1. 在 `file_parser.py` 中创建新的解析器类
2. 实现 `parse()` 方法返回 `ExtractedContent`
3. 在 `FileParserFactory` 中注册新格式

### 修改向量维度

如果使用不同的CLIP模型，向量维度可能不同。系统会自动检测，但也可以在 `.env` 中手动设置。

## 许可证

本项目遵循 Apache 2.0 许可证。

## 贡献

欢迎提交问题和拉取请求！


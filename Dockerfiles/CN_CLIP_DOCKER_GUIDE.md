# Chinese-CLIP Docker 部署指南

本指南介绍如何基于已有的 `clip-server-gpu` 镜像部署支持 Chinese-CLIP 的服务。

## 前提条件

- 已部署 `clip-server-gpu` 镜像
- 已下载 `clip-cn-vit-l-14.pt` 模型文件
- Linux 环境，已安装 Docker 和 NVIDIA Container Toolkit

## 1. 准备模型文件

创建模型目录并放置模型文件：

```bash
# 创建模型目录
mkdir -p /opt/models/cn_clip

# 将模型文件复制到目录
cp clip-cn-vit-l-14.pt /opt/models/cn_clip/
```

## 2. 构建 Docker 镜像

```bash
cd /path/to/clip-as-service-main

# 构建 cnclip-server 镜像
docker build -f Dockerfiles/cnclip-server.Dockerfile -t cnclip-server:latest .
```

> **注意**：如果你的基础镜像名称不是 `clip-server-gpu:latest`，请先修改 `cnclip-server.Dockerfile` 中的 `FROM` 行。

## 3. 配置 docker-compose

编辑 `Dockerfiles/docker-compose.cnclip.yml`，修改模型挂载路径：

```yaml
volumes:
  - /opt/models/cn_clip:/models/cn_clip:ro  # 改为你的实际路径
```

## 4. 启动服务

```bash
# 使用 docker-compose 启动
docker-compose -f Dockerfiles/docker-compose.cnclip.yml up -d

# 查看日志
docker logs -f cnclip-server
```

## 5. 验证服务

### Python 客户端测试

```python
from clip_client import Client

# 连接服务
client = Client("grpc://localhost:51000")

# 测试中文文本编码
texts = ["一只猫在草地上", "美丽的风景"]
text_embeddings = client.encode(texts)
print(f"文本向量维度: {text_embeddings.shape}")

# 测试图像编码
from docarray import Document
doc = Document(uri="test.jpg")
image_embedding = client.encode([doc])
print(f"图像向量维度: {image_embedding.shape}")
```

## 6. 常用命令

```bash
# 停止服务
docker-compose -f Dockerfiles/docker-compose.cnclip.yml down

# 重启服务
docker-compose -f Dockerfiles/docker-compose.cnclip.yml restart

# 查看服务状态
docker ps | grep cnclip
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `cnclip-server.Dockerfile` | 基于 clip-server-gpu 构建的 CN-CLIP 镜像 |
| `docker-compose.cnclip.yml` | Docker Compose 配置文件 |

## 故障排除

### 模型加载失败

确认模型文件路径正确：

```bash
docker exec cnclip-server ls -la /models/cn_clip/
```

### GPU 不可用

检查 NVIDIA 驱动和容器工具：

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.6.0-base-ubuntu20.04 nvidia-smi
```

### 端口冲突

修改 `docker-compose.cnclip.yml` 中的端口映射：

```yaml
ports:
  - "51001:51000"  # 改为其他端口
```


# CLIP 服务配置指南

本文档介绍如何通过配置文件启动不同的 CLIP 模型服务。

## 快速开始

```powershell
# 启动默认模型 (ViT-B-32)
python -m clip_server

# 启动指定配置
python -m clip_server <配置文件路径>
```

## 配置文件结构

配置文件使用 YAML 格式，基本结构如下：

```yaml
jtype: Flow
version: '1'
with:
  port: 51000              # 服务端口
  protocol: grpc           # 协议: grpc / http / websocket
executors:
  - name: clip_encoder
    uses:
      jtype: CLIPEncoder
      with:
        name: '模型名称'    # CLIP 模型
        device: 'cuda'     # 设备: cuda / cpu
    replicas: 1            # 副本数
```

## 预置配置文件

### 1. 默认配置 (英文 CLIP)

文件: `clip-as-service/server/clip_server/torch-flow.yml`

```yaml
jtype: Flow
version: '1'
with:
  port: 51000
  protocol: grpc
executors:
  - name: clip_encoder
    uses:
      jtype: CLIPEncoder
      with:
        name: 'ViT-B-32::openai'
        device: 'cuda'
    replicas: 1
```

启动命令:
```powershell
cd clip-as-service/server
python -m clip_server
# 或
python -m clip_server clip_server/torch-flow.yml
```

### 2. 中文 CLIP (CN-CLIP)

文件: `txt_search_image/cnclip-flow.yml`

```yaml
jtype: Flow
version: '1'
with:
  port: 51000
  protocol: grpc
executors:
  - name: clip_encoder
    uses:
      jtype: CLIPEncoder
      with:
        name: 'CN-CLIP/ViT-L-14'
        device: 'cuda'
    replicas: 1
```

启动命令:
```powershell
python -m clip_server txt_search_image/cnclip-flow.yml
```

**前置条件:**
1. 安装 cn_clip: `pip install cn_clip`
2. 模型文件放到 `~/.cache/clip/ViT-L-14.pt`

## 自定义配置示例

### 高精度模型配置

```yaml
# high-accuracy-flow.yml
jtype: Flow
version: '1'
with:
  port: 51000
  protocol: grpc
executors:
  - name: clip_encoder
    uses:
      jtype: CLIPEncoder
      with:
        name: 'ViT-L-14::openai'    # 更大的模型
        device: 'cuda'
        dtype: 'fp16'               # 半精度节省显存
    replicas: 1
```

### CPU 配置 (无 GPU)

```yaml
# cpu-flow.yml
jtype: Flow
version: '1'
with:
  port: 51000
  protocol: grpc
executors:
  - name: clip_encoder
    uses:
      jtype: CLIPEncoder
      with:
        name: 'ViT-B-32::openai'
        device: 'cpu'               # 使用 CPU
        dtype: 'fp32'               # CPU 建议用 fp32
    replicas: 1
```

### HTTP 协议配置

```yaml
# http-flow.yml
jtype: Flow
version: '1'
with:
  port: 8080                        # HTTP 端口
  protocol: http                    # 使用 HTTP 协议
  cors: true                        # 允许跨域
executors:
  - name: clip_encoder
    uses:
      jtype: CLIPEncoder
      with:
        name: 'ViT-B-32::openai'
        device: 'cuda'
    replicas: 1
```

客户端连接:
```python
engine = ImageSearchEngine(
    mode='service',
    clip_server='http://localhost:8080'  # 使用 HTTP
)
```

### 多副本负载均衡

```yaml
# multi-replica-flow.yml
jtype: Flow
version: '1'
with:
  port: 51000
  protocol: grpc
executors:
  - name: clip_encoder
    uses:
      jtype: CLIPEncoder
      with:
        name: 'ViT-B-32::openai'
        device: 'cuda'
    replicas: 2                     # 2 个副本并行处理
    polling: 'round_robin'          # 轮询策略
```

## 可用模型列表

### OpenAI CLIP 模型

| 模型名称 | 向量维度 | 说明 |
|---------|---------|------|
| `ViT-B-32::openai` | 512 | 默认，速度快 |
| `ViT-B-16::openai` | 512 | 精度较高 |
| `ViT-L-14::openai` | 768 | 高精度 |
| `ViT-L-14-336::openai` | 768 | 最高精度，336px |
| `RN50::openai` | 1024 | ResNet 版本 |
| `RN101::openai` | 512 | ResNet 101 |

### LAION 训练模型

| 模型名称 | 向量维度 | 说明 |
|---------|---------|------|
| `ViT-B-32::laion2b_e16` | 512 | LAION-2B 数据集 |
| `ViT-L-14::laion2b-s32b-b82k` | 768 | 大规模训练 |
| `ViT-H-14::laion2b-s32b-b79k` | 1024 | 超大模型 |

### 中文 CLIP 模型 (需要 cn_clip)

| 模型名称 | 向量维度 | 说明 |
|---------|---------|------|
| `CN-CLIP/ViT-B-16` | 512 | 中文基础版 |
| `CN-CLIP/ViT-L-14` | 768 | 中文大模型 |
| `CN-CLIP/ViT-L-14-336` | 768 | 中文高分辨率 |
| `CN-CLIP/ViT-H-14` | 1024 | 中文超大模型 |

## 配置参数说明

### Flow 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `port` | 服务端口 | 51000 |
| `protocol` | 协议类型 | grpc |
| `cors` | 是否允许跨域 (HTTP) | false |

### CLIPEncoder 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `name` | 模型名称 | ViT-B-32::openai |
| `device` | 运行设备 | cuda (如有) |
| `dtype` | 数据类型 | fp16 (GPU) / fp32 (CPU) |
| `jit` | 是否使用 JIT | false |
| `num_worker_preprocess` | 预处理线程数 | 4 |
| `minibatch_size` | 小批量大小 | 32 |

### Executor 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `replicas` | 副本数量 | 1 |
| `polling` | 负载策略 | round_robin |

## 使用流程

```
1. 选择/创建配置文件
        │
        ▼
2. 启动 clip-server
   python -m clip_server <config.yml>
        │
        ▼
3. 启动搜图应用 (无需指定模型)
   engine = ImageSearchEngine(
       mode='service',
       clip_server='grpc://localhost:51000'
   )
        │
        ▼
4. 开始使用
   engine.index_directory('/images')
   engine.search('搜索文本')
```

## 常见问题

### Q: 如何切换模型？
A: 修改配置文件中的 `name` 参数，重启服务即可。搜图应用代码不需要改动。

### Q: 显存不足怎么办？
A: 
1. 使用更小的模型 (如 ViT-B-32)
2. 设置 `dtype: 'fp16'` 使用半精度
3. 减小 `minibatch_size`

### Q: 如何提高吞吐量？
A:
1. 增加 `replicas` 数量
2. 增大 `minibatch_size`
3. 使用更快的 GPU

### Q: CPU 模式太慢？
A: CPU 模式建议使用较小的模型，并增加 `num_worker_preprocess` 参数。


# 基于已有的 clip-server-gpu 镜像
FROM clip-server-gpu:latest

USER root

# 安装 Chinese-CLIP 依赖
RUN pip install --no-cache-dir cn_clip ftfy

# 创建模型目录
RUN mkdir -p /models/cn_clip && \
    chown -R cas:cas /models

USER cas

# 默认使用 CN-CLIP 模型
ENV CN_CLIP_MODEL_PATH=/models/cn_clip/clip-cn-vit-l-14.pt

ENTRYPOINT ["python", "-m", "clip_server"]


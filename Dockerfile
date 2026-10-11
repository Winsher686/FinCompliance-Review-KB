# syntax=docker/dockerfile:1.7
# ============================================================
# Graph RAG 电商合规审查系统 · 运行镜像
#
# 基础镜像选型说明（不是随手挑的）：
#   1. 需要 **GPU**：BGE-M3 Embedding / BGE-Reranker 在 GPU 上才有可用吞吐，CPU 能跑但慢一个量级；
#   2. 选 **CUDA 11.8 runtime** 而不是 12.x：`knowledge/requirements.txt` 锁的是
#      `torch==2.7.1+cu118`，镜像内的 CUDA runtime 版本必须与 wheel 的编译版本对齐，
#      否则会出现 libcudart / libcublas 版本冲突（典型现象：import torch 直接段错误）；
#   3. 用 **runtime** 变体而不是 devel：运行期不需要 nvcc，镜像小几个 GB；
#   4. 无 GPU 的机器也能跑（torch 自动回落 CPU），只是慢——所以 compose 里 GPU 是**可选叠加**。
#
# 构建：
#   docker build -t ecommerce-compliance-kb:latest .
# 运行：
#   docker run --rm -p 8002:8002 --env-file knowledge/.env ecommerce-compliance-kb:latest
# ============================================================
ARG CUDA_VERSION=11.8.0
ARG UBUNTU_VERSION=22.04

# ------------------------------------------------------------
# 阶段 1：依赖构建
# 单独一层装依赖，让「只改业务代码」的重建命中缓存，
# 不必每次重装 10GB 级的 torch / sentence-transformers。
# ------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv git \
    && rm -rf /var/lib/apt/lists/*

# 依赖装在独立 venv，便于整目录复制到运行镜像，避免把 apt 缓存一起带走
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY knowledge/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r /tmp/requirements.txt

# ------------------------------------------------------------
# 阶段 2：运行镜像
# ------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:${PATH}" \
    TZ=Asia/Shanghai

# 运行期系统依赖：
#   libgl1 / libglib2.0-0  → opencv、图像处理相关（PDF 转图、切片配图）
#   poppler-utils          → PDF 文本/结构解析
#   fonts-noto-cjk         → 中文渲染（否则图表里的中文变方块）
#   curl                   → 健康检查
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 libgl1 libglib2.0-0 poppler-utils fonts-noto-cjk curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=appuser:appuser . /app

# 运行期可写目录（compose 会挂卷持久化；属主交给非 root 用户）
RUN mkdir -p /app/knowledge/processor/import_process/temp_data \
             /app/knowledge/data/observability \
    && chown -R appuser:appuser /app/knowledge

# 不以 root 运行：容器被攻破时的爆炸半径更小
USER appuser

EXPOSE 8000 8001 8002

# 默认探针指向审查服务；compose 里每个服务会覆盖成自己的端口
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:8002/health" || exit 1

# 默认起审查服务；导入/查询服务在 compose 里覆盖 command
CMD ["uvicorn", "knowledge.processor.review_process.api.review_router:create_app", \
     "--factory", "--host", "0.0.0.0", "--port", "8002"]

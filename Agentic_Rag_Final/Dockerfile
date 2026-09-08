FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# ------------------------------------------------------------
# System packages
# ------------------------------------------------------------

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Python
# ------------------------------------------------------------

RUN python3 -m pip install --upgrade pip

# ------------------------------------------------------------
# Project dependencies
# ------------------------------------------------------------

COPY requirements.txt .

# ------------------------------------------------------------
# GPU-enabled PyTorch matching CUDA 12.8
# Install this FIRST so other packages use this version
# ------------------------------------------------------------

RUN pip install \
    --default-timeout=120 \
    --retries=20 \
    --resume-retries=20 \
    torch==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu128

# ------------------------------------------------------------
# Remaining project dependencies
# ------------------------------------------------------------

RUN pip install \
    --default-timeout=120 \
    --retries=20 \
    -r requirements.txt

# ------------------------------------------------------------
# Copy project
# ------------------------------------------------------------

COPY . .

# ------------------------------------------------------------
# Hugging Face / Python settings
# ------------------------------------------------------------

ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface

# ------------------------------------------------------------
# Ports
# ------------------------------------------------------------

EXPOSE 8000
EXPOSE 8501

# ------------------------------------------------------------
# Backend
# ------------------------------------------------------------

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
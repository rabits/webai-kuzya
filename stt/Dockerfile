# Whisper STT service for DGX Spark (aarch64, CUDA 13).
# torch must come from the cu130 index, not default PyPI.
# openai-whisper is used instead of faster-whisper: CTranslate2 has no
# CUDA aarch64 wheels, so Speaches/faster-whisper would silently fall back to CPU.

FROM nvidia/cuda:13.3.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev build-essential pkg-config \
        ffmpeg libsndfile1 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt \
    && /opt/venv/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130/ \
    && /opt/venv/bin/pip install 'numpy>=1.26,<2' \
    && /opt/venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

COPY main.py entrypoint.sh ./
RUN chmod +x /app/entrypoint.sh

EXPOSE 8001
ENTRYPOINT ["/app/entrypoint.sh"]

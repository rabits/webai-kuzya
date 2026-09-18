# F5-TTS. Platform torch comes from compose build args, not PyPI.
# NVIDIA: docker-compose.nvidia.yml (cu130). AMD: docker-compose.amd.yml (ROCm).

ARG BASE_IMAGE=nvidia/cuda:13.3.1-cudnn-runtime-ubuntu24.04
FROM ${BASE_IMAGE}

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130/
ARG PIP_EXTRA_FLAGS=""
ARG TORCH_PACKAGES="torch torchaudio"
ARG GPU_FLAVOR=nvidia

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev build-essential pkg-config \
        ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./

# f5-tts pulls torch from PyPI (CUDA). Install the platform wheel first, then
# re-assert it after requirements so pip cannot keep a +cu wheel as "satisfied".
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && if [ -n "${TORCH_INDEX_URL}" ]; then \
         echo "torch from ${TORCH_INDEX_URL} (${GPU_FLAVOR})"; \
         /opt/venv/bin/pip install ${PIP_EXTRA_FLAGS} ${TORCH_PACKAGES} \
           --index-url "${TORCH_INDEX_URL}"; \
       fi \
    && /opt/venv/bin/pip install -r requirements.txt \
    && if [ -n "${TORCH_INDEX_URL}" ]; then \
         /opt/venv/bin/pip install ${PIP_EXTRA_FLAGS} --force-reinstall --no-deps \
           ${TORCH_PACKAGES} --index-url "${TORCH_INDEX_URL}"; \
       fi \
    && /opt/venv/bin/pip uninstall -y torchcodec || true \
    && /opt/venv/bin/pip install 'numpy>=1.26,<2' \
    && /opt/venv/bin/python -c "\
import sys, torch; v=torch.__version__; print('torch', v, 'hip', getattr(torch.version,'hip',None)); \
flavor='${GPU_FLAVOR}'; \
sys.exit('expected ROCm torch, got '+v) if flavor=='rocm' and 'rocm' not in v.lower() else None; \
sys.exit('expected CUDA torch, got '+v) if flavor=='nvidia' and 'rocm' in v.lower() else None"

COPY main.py entrypoint.sh ./
RUN chmod +x /app/entrypoint.sh

EXPOSE 8002
ENTRYPOINT ["/app/entrypoint.sh"]

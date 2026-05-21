FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /workspace/project

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    unzip \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/local/bin/python

RUN python -m pip install --upgrade pip setuptools wheel

RUN python -m pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt /workspace/project/requirements.txt

RUN python -m pip install --no-cache-dir -r /workspace/project/requirements.txt

COPY Q2fix180.py /workspace/project/Q2fix180.py

EXPOSE 8888

CMD ["bash", "-lc", "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.token=${JUPYTER_TOKEN:-gcube} --ServerApp.password='' --notebook-dir=/workspace"]
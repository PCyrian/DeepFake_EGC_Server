# Use the CUDA 11.8.0 devel image with Ubuntu 20.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    zlib1g-dev \
    libssl-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libsqlite3-dev \
    libreadline-dev \
    libffi-dev \
    curl \
    libbz2-dev \
    ffmpeg \
    git \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://www.python.org/ftp/python/3.10.11/Python-3.10.11.tgz \
    && tar -xvzf Python-3.10.11.tgz \
    && cd Python-3.10.11 \
    && ./configure --enable-optimizations \
    && make -j$(nproc) \
    && make altinstall \
    && ln -s /usr/local/bin/python3.10 /usr/local/bin/python3 \
    && ln -s /usr/local/bin/pip3.10 /usr/local/bin/pip3

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN python3 -m pip install pip==23.3.1
COPY requirements.txt /app/requirements.txt

RUN pip3 install -r /app/requirements.txt

WORKDIR /app

COPY app.py /app/app.py
COPY tos_acceptation.py /app/tos_acceptation.py

EXPOSE 7860

RUN yes | python3 tos_acceptation.py

CMD ["python3", "app.py"]

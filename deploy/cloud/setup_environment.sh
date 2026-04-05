#!/bin/bash
#============================================
# LAVAD 阿里云 GPU 环境配置脚本
# 适用于 Ubuntu 20.04 + NVIDIA GPU
#============================================

set -e

# 配置清华镜像源
echo "=== 配置 apt 镜像源 ==="
sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak
sudo sed -i 's/cn.archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list
sudo sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list
sudo apt-get update

# 安装基础依赖
echo "=== 安装系统依赖 ==="
sudo apt-get install -y \
    wget \
    vim \
    git \
    curl \
    unzip \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    zlib1g-dev \
    libgomp1 \
    bzip2 \
    lzma

# 安装 Miniconda (使用清华源)
echo "=== 安装 Miniconda ==="
cd /tmp
wget -q https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p /opt/miniconda
rm miniconda.sh

# 配置 conda 镜像
export PATH=/opt/miniconda/bin:$PATH
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --set show_channel_urls yes

# 创建 conda 环境 (Python 3.10 + CUDA 11.7)
echo "=== 创建 conda 环境 lavad ==="
conda create -n lavad python=3.10 -y
conda activate lavad

# 安装 PyTorch (CUDA 12.1)
echo "=== 安装 PyTorch ==="
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    -f https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cu121.html \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装项目核心依赖
echo "=== 安装项目依赖 ==="
cd /home/lavad

pip install timm==0.6.7 ftfy regex einops fvcore \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install eva-decord==0.6.1 iopath \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install fairscale fire sentencepiece transformers==4.31.0 accelerate \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install scikit-learn faiss-gpu opencv-python matplotlib numpy \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 pytorchvideo
pip install pytorchvideo@git+https://github.com/facebookresearch/pytorchvideo.git@28fe037d212663c6a24f373b94cc5d478c8c1a1d \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 Web 服务依赖
pip install fastapi uvicorn python-multipart aiofiles \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install streamlit requests \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "=== 环境配置完成 ==="
echo "请执行以下命令激活环境: source /opt/miniconda/bin/activate && conda activate lavad"</
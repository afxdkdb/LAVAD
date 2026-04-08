# LAVAD 阿里云 GPU 部署完整指南

## 目录

1. [阿里云 GPU 服务器配置要求](#1-阿里云-gpu-服务器配置要求)
2. [环境配置步骤](#2-环境配置步骤)
3. [下载预训练模型](#3-下载预训练模型)
4. [部署 Web 服务](#4-部署-web-服务)
5. [本地访问](#5-本地访问)
6. [常见问题](#6-常见问题)

---

## 1. 阿里云 GPU 服务器配置要求

### 推荐配置

| 配置项 | 推荐规格 |
|--------|----------|
| **GPU** | NVIDIA A10 / A100 / V100（推荐 A10，24GB 显存） |
| **内存** | ≥ 64GB |
| **系统盘** | ≥ 100GB SSD |
| **数据盘** | ≥ 500GB（存放模型和视频数据） |
| **操作系统** | Ubuntu 20.04 LTS |
| **CUDA** | CUDA 11.7 |
| **Python** | Python 3.10 |

### 购买阿里云 GPU 服务器

1. 登录阿里云官网：https://www.aliyun.com
2. 进入 **ECS GPU 云服务器** 产品页面
3. 选择以下配置：
   - 实例类型：`ecs.gn7-c12g1.3xlarge`（A10 GPU）或 `ecs.gn7i-c16g1.4xlarge`（A10）
   - 操作系统：**Ubuntu 20.04 LTS 64位**
   - 存储：高性能 SSD 云盘
4. 配置安全组，开放端口：**8000**（API）、**8501**（Web）

---

## 2. 环境配置步骤

### 2.1 连接云服务器

```bash
ssh root@<你的服务器IP>
```

### 2.2 运行环境配置脚本

将项目上传到云服务器：

```bash
# 在本地执行（需要先配置 ssh 密钥）
scp -r ./lavad root@<服务器IP>:/home/lavad/

# 或者在云服务器上 clone
git clone https://github.com/lucazanella/lavad.git /home/lavad/
```

### 2.3 执行环境安装脚本

```bash
# 登录云服务器
ssh root@<服务器IP>

# 进入项目目录
cd /home/lavad

# 添加执行权限
chmod +x deploy/cloud/setup_environment.sh

# 运行环境配置脚本
bash deploy/cloud/setup_environment.sh
```

脚本将自动完成以下操作：
- 配置清华镜像源
- 安装系统依赖
- 安装 Miniconda
- 配置 conda 镜像
- 创建 Python 3.10 环境
- 安装 PyTorch 1.13.0 + CUDA 11.7
- 安装项目依赖

### 2.4 手动验证环境

```bash
# 激活环境
source /opt/miniconda/bin/activate
conda activate lavad

# 验证 Python
python --version  # 应该是 3.10.x

# 验证 PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# 验证关键依赖
python -c "import transformers; import cv2; import faiss; print('Dependencies OK')"
```

---

## 3. 下载预训练模型

### 3.1 BLIP-2 和 ImageBind

这两个模型会在首次使用时自动下载，但建议预先下载：

```bash
conda activate lavad

# 预先下载 BLIP-2
python -c "from transformers import Blip2Processor, Blip2ForConditionalGeneration; \
    processor = Blip2Processor.from_pretrained('Salesforce/blip2-opt-6.7b-coco'); \
    model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-6.7b-coco', torch_dtype=torch.float16)"

# 预先下载 ImageBind
python -c "from libs.ImageBind.imagebind.models.imagebind_model import imagebind_huge; \
    model = imagebind_huge(pretrained=True)"
```

### 3.2 LLaMA 2 模型下载（必须手动）

按照 `deploy/cloud/download_llama2_cn.md` 中的方法下载：

#### 方法 A：通过 Hugging Face 镜像（推荐）

```bash
conda activate lavad

# 设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 安装 huggingface_hub
pip install huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple

# 创建目录
mkdir -p /home/lavad/libs/llama

# 下载模型（约 26GB，需要 Hugging Face token）
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='meta-llama/Llama-2-13b-chat',
    local_dir='/home/lavad/libs/llama/llama-2-13b-chat',
    local_dir_use_symlinks=False
)
"

# 下载 tokenizer
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='meta-llama/Llama-2-13b-chat',
    filename='tokenizer.model',
    local_dir='/home/lavad/libs/llama'
)
"
```

#### 方法 B：通过魔搭社区

```bash
conda activate lavad
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple

python -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download(
    model_id='LLM-Research/llama-2-13b-chat',
    cache_dir='/home/lavad/libs/llama',
    revision='master'
)
"
```

### 3.3 验证模型下载

```bash
ls -la /home/lavad/libs/llama/llama-2-13b-chat/
# 应该包含 consolidated.*.pth 和 params.json

ls -la /home/lavad/libs/llama/tokenizer.model
```

---

## 4. 部署 Web 服务

### 4.1 配置安全组

在阿里云控制台 → ECS → 安全组，添加规则：

| 方向 | 协议 | 端口 | 来源 |
|------|------|------|------|
| 入方向 | TCP | 8000 | 0.0.0.0/0 |
| 入方向 | TCP | 8501 | 0.0.0.0/0 |

### 4.2 启动服务

```bash
# 登录云服务器
ssh root@<服务器IP>

# 进入项目目录
cd /home/lavad

# 激活环境
source /opt/miniconda/bin/activate
conda activate lavad

# 添加执行权限
chmod +x deploy/cloud/start_server.sh

# 启动服务
bash deploy/cloud/start_server.sh
```

### 4.3 检查服务状态

```bash
# 查看运行中的进程
ps aux | grep -E "uvicorn|streamlit"

# 查看日志
tail -f /tmp/lavad_backend.log
tail -f /tmp/lavad_frontend.log

# 测试 API
curl http://localhost:8000/health
```

预期输出：
```json
{"status":"healthy","device":"cuda:0","cuda_available":true}
```

---

## 5. 本地访问

### 5.1 访问 Web 界面

在浏览器中打开：
```
http://<云服务器公网IP>:8501
```

### 5.2 配置本地连接

如果云服务器 IP 发生变化，修改 Streamlit 前端配置：

在 **本地浏览器** 访问时，输入云服务器的公网 IP。

### 5.3 上传视频测试

1. 准备一个测试视频（MP4 或 AVI 格式）
2. 点击 "选择视频文件" 上传
3. 点击 "开始检测"
4. 等待处理完成（可能需要几分钟）
5. 查看异常分数曲线和可视化结果

---

## 6. 常见问题

### Q1: CUDA out of memory

**问题**：GPU 显存不足

**解决方案**：
```bash
# 减小批处理大小
export CUDA_VISIBLE_DEVICES=0

# 或使用更小的模型（如 llama-2-7b-chat）
```

### Q2: 模型下载失败

**问题**：网络原因导致 Hugging Face 下载失败

**解决方案**：
```bash
# 使用镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或使用 ModelScope
pip install modelscope
```

### Q3: 端口无法访问

**解决方案**：
1. 检查阿里云安全组是否开放端口
2. 检查云服务器防火墙
```bash
# 开放端口
sudo ufw allow 8000
sudo ufw allow 8501
```

### Q4: 视频上传超时

**问题**：视频文件太大

**解决方案**：
- 前端默认超时 300 秒
- 建议视频时长控制在 2 分钟以内
- 视频文件大小控制在 100MB 以内

### Q5: 后端服务启动失败

**排查步骤**：
```bash
# 查看错误日志
cat /tmp/lavad_backend.log

# 检查端口占用
netstat -tlnp | grep 8000

# 手动启动查看错误
source /opt/miniconda/bin/activate
conda activate lavad
python -m uvicorn deploy.service.backend.main:app --host 0.0.0.0 --port 8000
```

---

## 服务架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      本地浏览器                              │
│              http://<公网IP>:8501                            │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    阿里云 GPU 服务器                          │
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │  Streamlit 前端  │    │      FastAPI 后端               │ │
│  │   Port: 8501    │◄──►│        Port: 8000                │ │
│  └─────────────────┘    └──────────┬──────────────────────┘ │
│                                    │                         │
│                                    ▼                         │
│                     ┌──────────────────────────┐            │
│                     │     LAVAD Pipeline      │            │
│                     │  - BLIP-2 Caption       │            │
│                     │  - ImageBind Index       │            │
│                     │  - LLaMA 2 Scoring       │            │
│                     └──────────────────────────┘            │
│                                    │                         │
│                                    ▼                         │
│                     ┌──────────────────────────┐            │
│                     │   NVIDIA GPU (A10)      │            │
│                     └──────────────────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

---

## 快速命令汇总

```bash
# 连接服务器
ssh root@<服务器IP>

# 激活环境
source /opt/miniconda/bin/activate && conda activate lavad

# 查看服务状态
ps aux | grep -E "uvicorn|streamlit"

# 查看日志
tail -f /tmp/lavad_backend.log
tail -f /tmp/lavad_frontend.log

# 重启服务
pkill -f uvicorn
pkill -f streamlit
bash /home/lavad/deploy/cloud/start_server.sh

# 测试 API
curl http://localhost:8000/health

# 停止服务
pkill -f uvicorn
pkill -f streamlit
```
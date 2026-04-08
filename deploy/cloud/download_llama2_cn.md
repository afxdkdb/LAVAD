# LLaMA 2 模型下载指南（国内镜像）

## 方法一：通过 Hugging Face 镜像站下载（推荐）

LLaMA 2 模型托管在 Hugging Face，由于网络原因，国内用户可以通过以下方式下载：

### 1. 设置 Hugging Face 镜像

```bash
# 设置 Hugging Face 镜像站
export HF_ENDPOINT=https://hf-mirror.com

# 或者在 ~/.bashrc 中永久设置
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

### 2. 下载模型

```bash
# 安装 huggingface_hub
pip install huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple

# 登录 Hugging Face（需要 access token）
huggingface-cli login

# 下载 llama-2-13b-chat
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='meta-llama/Llama-2-13b-chat',
    local_dir='/home/lavad/libs/llama/llama-2-13b-chat',
    local_dir_use_symlinks=False
)
"
```

### 3. 下载 Tokenizer

```bash
# 下载 tokenizer.model
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='meta-llama/Llama-2-13b-chat',
    filename='tokenizer.model',
    local_dir='/home/lavad/libs/llama'
)
"
```

---

## 方法二：直接从 Meta 官网申请下载（需科学上网）

### 1. 访问 Meta LLaMA 官网
访问 https://llama.meta.com/llama-downloads/

填写表格申请下载权限。

### 2. 批准后获取下载链接
Meta 会通过邮件发送下载链接。

### 3. 使用国内服务器代理下载
在云服务器上使用 wget/curl 下载：

```bash
# 替换 <YOUR_URL> 为邮件中的下载链接
wget -c <YOUR_URL> -O llama-2-13b-chat.tar.gz
```

### 4. 解压模型

```bash
tar -xzf llama-2-13b-chat.tar.gz
mv llama-2-13b-chat /home/lavad/libs/llama/
rm llama-2-13b-chat.tar.gz
```

---

## 方法三：通过魔搭社区（ModelScope）下载

ModelScope 提供了 LLaMA 2 的中文镜像，适合国内用户。

### 1. 安装 modelscope

```bash
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 下载模型

```python
from modelscope.hub.snapshot_download import snapshot_download

# 下载 llama-2-13b-chat
snapshot_download(
    model_id='LLM-Research/llama-2-13b-chat',
    cache_dir='/home/lavad/libs/llama',
    revision='master'
)
```

---

## 下载后目录结构

```
libs/
└── llama/
    ├── llama-2-13b-chat/
    │   ├── consolidated.00.pth
    │   ├── consolidated.01.pth
    │   ├── params.json
    │   └── iteration_00001/
    └── tokenizer.model
```

---

## 验证模型是否下载成功

```python
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

model_path = "/home/lavad/libs/llama/llama-2-13b-chat"
tokenizer_path = "/home/lavad/libs/llama/tokenizer.model"

tokenizer = LlamaTokenizer(tokenizer_path)
model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)

print("LLaMA 2 模型加载成功!")
```

---

## 常见问题

### Q: Hugging Face 需要 access token 怎么办？
A: 注册 Hugging Face 账号，在 https://huggingface.co/settings/tokens 创建 access token。

### Q: 模型文件太大，下载中断怎么办？
A: 使用 wget -c 断点续传：
```bash
wget -c <URL> -O llama-2-13b-chat.tar.gz
```

### Q: 磁盘空间不足？
A: LLaMA 2 13B 模型约需 26GB 磁盘空间。确保云服务器有足够存储。

---

## 模型版本说明

| 模型 | 参数量 | 磁盘占用 | 最低显存 |
|------|--------|----------|----------|
| llama-2-7b-chat | 7B | ~13GB | ~16GB |
| llama-2-13b-chat | 13B | ~26GB | ~32GB |
| llama-2-70b-chat | 70B | ~140GB | ~160GB |

推荐使用 **llama-2-13b-chat**，在性能和硬件要求之间有较好平衡。
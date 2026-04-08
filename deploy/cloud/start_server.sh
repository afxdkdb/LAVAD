#!/bin/bash
#============================================
# 阿里云 GPU 服务器部署启动脚本
#============================================

set -e

# 激活 conda 环境
export PATH=/opt/miniconda/bin:$PATH
source /opt/miniconda/etc/profile.d/conda.sh
conda activate lavad

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/home/lavad:/home/lavad/libs/ImageBind:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=YOUR_HF_TOKEN

# 项目目录
PROJECT_DIR="/home/lavad"
cd $PROJECT_DIR

# 创建必要的目录
mkdir -p /tmp/lavad_uploads
mkdir -p /tmp/lavad_outputs

# 启动后端服务 (FastAPI)
echo "=== 启动后端服务 ==="
nohup /opt/miniconda/envs/lavad/bin/python -m uvicorn deploy.service.backend.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    > /tmp/lavad_backend.log 2>&1 &

BACKEND_PID=$!
echo "后端服务 PID: $BACKEND_PID"

# 等待后端启动
sleep 5

# 检查后端是否启动成功
if ps -p $BACKEND_PID > /dev/null; then
    echo "✅ 后端服务启动成功 (http://0.0.0.0:8000)"
else
    echo "❌ 后端服务启动失败，请查看日志: /tmp/lavad_backend.log"
    exit 1
fi

# 启动前端服务 (Streamlit)
echo "=== 启动前端服务 ==="
export API_URL="http://localhost:8000"

nohup /opt/miniconda/envs/lavad/bin/streamlit run deploy/service/frontend/app.py \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.headless true \
    > /tmp/lavad_frontend.log 2>&1 &

FRONTEND_PID=$!
echo "前端服务 PID: $FRONTEND_PID"

# 等待前端启动
sleep 5

# 检查前端是否启动成功
if ps -p $FRONTEND_PID > /dev/null; then
    echo "✅ 前端服务启动成功 (http://0.0.0.0:8501)"
else
    echo "❌ 前端服务启动失败，请查看日志: /tmp/lavad_frontend.log"
    exit 1
fi

echo ""
echo "=========================================="
echo "🎉 LAVAD 服务部署成功!"
echo "=========================================="
echo "后端 API: http://<云服务器IP>:8000"
echo "前端界面: http://<云服务器IP>:8501"
echo ""
echo "查看日志:"
echo "  后端日志: tail -f /tmp/lavad_backend.log"
echo "  前端日志: tail -f /tmp/lavad_frontend.log"
echo ""
echo "停止服务:"
echo "  kill $BACKEND_PID $FRONTEND_PID"
echo "=========================================="
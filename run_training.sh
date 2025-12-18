#!/bin/bash
# R-Zero 训练启动脚本 - 自动设置环境变量

# 使用 GPU 0,1 进行训练
export CUDA_VISIBLE_DEVICES=0,1

# 设置环境变量
export STORAGE_PATH=/data/user5/R-Zero
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACENAME="YOUR_HUGGINGFACE_TOKEN_HERE"

# 验证环境变量
echo "=== 环境变量检查 ==="
echo "STORAGE_PATH: $STORAGE_PATH"
echo "HF_ENDPOINT: $HF_ENDPOINT"
echo ""

# 检查并创建必要目录
mkdir -p $STORAGE_PATH/models
mkdir -p $STORAGE_PATH/generated_question
mkdir -p $STORAGE_PATH/temp_results
mkdir -p $STORAGE_PATH/evaluation

# 进入项目目录
cd /data/user5/R-Zero

# 运行训练
echo "=== 开始训练 ==="
bash scripts/main.sh "${1:-/data/user5/R-Zero/OctoThinker-3B-Hybrid-Base}" "${2:-questioner_v1}"


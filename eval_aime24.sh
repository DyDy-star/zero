#!/bin/bash
# AIME 2024 测评脚本 - 使用 questioner_v1 模型 (Base Challenger)

# ⚠️ 强制只使用 GPU 0
export CUDA_VISIBLE_DEVICES=2

# 设置环境变量
export STORAGE_PATH=/data/user5/R-Zero
export HUGGINGFACENAME="YOUR_HUGGINGFACE_TOKEN_HERE"
export HF_ENDPOINT=https://hf-mirror.com

# 禁用 vLLM 编译缓存
export VLLM_DISABLE_COMPILE_CACHE=1

# 模型路径
MODEL_PATH="/data/user5/R-Zero/OctoThinker-3B-Hybrid-Base"

# 检查模型是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型路径不存在: $MODEL_PATH"
    exit 1
fi

echo "================================"
echo "开始 AIME 2024 测评"
echo "模型: $MODEL_PATH"
echo "数据集: aime2024"
echo "使用 GPU: $CUDA_VISIBLE_DEVICES"
echo "================================"

# 创建评估结果目录
mkdir -p ${STORAGE_PATH}/evaluation

# 设置 Python 路径，确保可以导入 evaluation 模块
export PYTHONPATH="${STORAGE_PATH}:${PYTHONPATH}"

# 切换到项目根目录
cd ${STORAGE_PATH}

# 运行 AIME 2024 测评
python evaluation/generate.py \
    --model "$MODEL_PATH" \
    --dataset "aime2024"

echo "================================"
echo "AIME 2024 测评完成！"
echo "结果保存在: ${STORAGE_PATH}/evaluation/$(basename $(dirname $(dirname $(dirname $MODEL_PATH))))/results_aime2024.json"
echo "================================"


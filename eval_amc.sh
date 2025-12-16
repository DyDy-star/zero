#!/bin/bash
# AMC 测评脚本 - 使用 OctoThinker-3B-Hybrid-Base 模型

# ⚠️ 强制只使用 GPU 0
export CUDA_VISIBLE_DEVICES=0

# 设置环境变量
export STORAGE_PATH=/data/user5/R-Zero
export HUGGINGFACENAME="YOUR_HF_TOKEN_HERE"
export HF_ENDPOINT=https://hf-mirror.com

# 禁用 vLLM 编译缓存
export VLLM_DISABLE_COMPILE_CACHE=1

# 模型路径
MODEL_PATH="/data/user5/R-Zero/models/octo_3b_questioner_v1/global_step_5/actor/huggingface"

# 检查模型是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型路径不存在: $MODEL_PATH"
fi

echo "================================"
echo "开始 AMC 测评"
echo "模型: $MODEL_PATH"
echo "数据集: amc"
echo "使用 GPU: $CUDA_VISIBLE_DEVICES"
echo "================================"

# 创建评估结果目录
mkdir -p ${STORAGE_PATH}/evaluation

# 设置 Python 路径，确保可以导入 evaluation 模块
export PYTHONPATH="${STORAGE_PATH}:${PYTHONPATH}"

# 切换到项目根目录
cd ${STORAGE_PATH}

# 运行 AMC 测评
python evaluation/generate.py \
    --model "$MODEL_PATH" \
    --dataset "amc"

echo "================================"
echo "AMC 测评完成！"
echo "结果保存在: ${STORAGE_PATH}/evaluation/$(basename $MODEL_PATH | sed 's/\//_/g')/results_amc.json"
echo "================================"


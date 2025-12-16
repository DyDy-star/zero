#!/bin/bash
#
# 从 Questioner v2 开始继续训练（Solver v1 已完成）
# 训练流程：questioner_v2 -> solver_v2 -> questioner_v3 -> solver_v3
#

set -e  # 遇到错误立即退出

echo "=========================================="
echo "从 Questioner v2 开始继续训练"
echo "=========================================="
echo ""

# 设置环境变量
export STORAGE_PATH=/data/user5/R-Zero
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACENAME=123YYY123
export CUDA_VISIBLE_DEVICES=4,5,6,7
export WANDB_BASE_URL=https://api.bandw.top
export PYTHONPATH=/data/user5/R-Zero:$PYTHONPATH

echo "✓ STORAGE_PATH=$STORAGE_PATH"
echo "✓ HF_ENDPOINT=$HF_ENDPOINT"
echo "✓ HUGGINGFACENAME=$HUGGINGFACENAME"
echo "✓ CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "✓ WANDB_BASE_URL=$WANDB_BASE_URL"
echo "✓ PYTHONPATH=$PYTHONPATH"
echo ""

# 设置模型前缀
Base_model=/data/user5/R-Zero/OctoThinker-3B-Hybrid-Base
Model_abbr=octo_3b

# 验证必需的模型是否存在
if [ ! -d "${STORAGE_PATH}/models/${Model_abbr}_solver_v1/global_step_15/actor/huggingface" ]; then
    echo "❌ 错误: Solver v1 模型不存在，请先训练 Solver v1"
    exit 1
fi

if [ ! -d "${STORAGE_PATH}/models/${Model_abbr}_questioner_v1/global_step_5/actor/huggingface" ]; then
    echo "❌ 错误: Questioner v1 模型不存在，请先训练 Questioner v1"
    exit 1
fi

echo "✓ Solver v1 模型已存在"
echo "✓ Questioner v1 模型已存在"
echo ""

# ================================
# 第2轮：训练 Questioner v2
# ================================
echo "=========================================="
echo "开始训练 Questioner v2"
echo "=========================================="
bash scripts/questioner_train_penalty.sh \
    ${STORAGE_PATH}/models/${Model_abbr}_solver_v1/global_step_15/actor/huggingface \
    ${STORAGE_PATH}/models/${Model_abbr}_questioner_v1/global_step_5/actor/huggingface \
    ${Model_abbr}_questioner_v2

if [ $? -ne 0 ]; then
    echo "❌ Questioner v2 训练失败"
    exit 1
fi

echo "✓ Questioner v2 训练完成"
echo ""

# ================================
# 第2轮：训练 Solver v2
# ================================
echo "=========================================="
echo "开始训练 Solver v2"
echo "=========================================="
bash scripts/solver_train.sh \
    ${STORAGE_PATH}/models/${Model_abbr}_solver_v1/global_step_15/actor/huggingface \
    ${STORAGE_PATH}/models/${Model_abbr}_questioner_v2/global_step_5/actor/huggingface \
    ${Model_abbr}_solver_v2

if [ $? -ne 0 ]; then
    echo "❌ Solver v2 训练失败"
    exit 1
fi

# 验证 solver_v2 模型是否生成
if [ ! -d "${STORAGE_PATH}/models/${Model_abbr}_solver_v2/global_step_15/actor/huggingface" ]; then
    echo "❌ 错误: Solver v2 模型未生成"
    exit 1
fi

echo "✓ Solver v2 训练完成"
echo ""

# ================================
# 第3轮：训练 Questioner v3
# ================================
echo "=========================================="
echo "开始训练 Questioner v3"
echo "=========================================="
bash scripts/questioner_train_penalty.sh \
    ${STORAGE_PATH}/models/${Model_abbr}_solver_v2/global_step_15/actor/huggingface \
    ${STORAGE_PATH}/models/${Model_abbr}_questioner_v2/global_step_5/actor/huggingface \
    ${Model_abbr}_questioner_v3

if [ $? -ne 0 ]; then
    echo "❌ Questioner v3 训练失败"
    exit 1
fi

echo "✓ Questioner v3 训练完成"
echo ""

# ================================
# 第3轮：训练 Solver v3
# ================================
echo "=========================================="
echo "开始训练 Solver v3"
echo "=========================================="
bash scripts/solver_train.sh \
    ${STORAGE_PATH}/models/${Model_abbr}_solver_v2/global_step_15/actor/huggingface \
    ${STORAGE_PATH}/models/${Model_abbr}_questioner_v3/global_step_5/actor/huggingface \
    ${Model_abbr}_solver_v3

if [ $? -ne 0 ]; then
    echo "❌ Solver v3 训练失败"
    exit 1
fi

# 验证 solver_v3 模型是否生成
if [ ! -d "${STORAGE_PATH}/models/${Model_abbr}_solver_v3/global_step_15/actor/huggingface" ]; then
    echo "❌ 错误: Solver v3 模型未生成"
    exit 1
fi

echo "✓ Solver v3 训练完成"
echo ""

# ================================
# 最终评估已禁用（可在训练完成后手动运行）
# echo "=========================================="
# echo "开始最终评估"
# echo "=========================================="
# bash evaluation/evaluate_4gpu.bash ${STORAGE_PATH}/models/${Model_abbr}_solver_v3/global_step_15/actor/huggingface

echo ""
echo "=========================================="
echo "🎉 完整的训练全部完成！"
echo "=========================================="
echo ""
echo "已完成的模型："
echo "  ✓ Questioner v1: ${STORAGE_PATH}/models/${Model_abbr}_questioner_v1/global_step_5/actor/huggingface"
echo "  ✓ Solver v1:     ${STORAGE_PATH}/models/${Model_abbr}_solver_v1/global_step_15/actor/huggingface"
echo "  ✓ Questioner v2: ${STORAGE_PATH}/models/${Model_abbr}_questioner_v2/global_step_5/actor/huggingface"
echo "  ✓ Solver v2:     ${STORAGE_PATH}/models/${Model_abbr}_solver_v2/global_step_15/actor/huggingface"
echo "  ✓ Questioner v3: ${STORAGE_PATH}/models/${Model_abbr}_questioner_v3/global_step_5/actor/huggingface"
echo "  ✓ Solver v3:     ${STORAGE_PATH}/models/${Model_abbr}_solver_v3/global_step_15/actor/huggingface"
echo ""
echo "最终模型位置："
echo "  Questioner: ${STORAGE_PATH}/models/${Model_abbr}_questioner_v3/global_step_5/actor/huggingface"
echo "  Solver:     ${STORAGE_PATH}/models/${Model_abbr}_solver_v3/global_step_15/actor/huggingface"
echo ""
echo "如需评估，运行以下命令："
echo "  bash evaluation/evaluate_4gpu.bash ${STORAGE_PATH}/models/${Model_abbr}_solver_v3/global_step_15/actor/huggingface"
echo ""



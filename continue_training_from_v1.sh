#!/bin/bash
#
# 从 questioner_v1 继续训练到第3轮（完整的3轮迭代）
# 自动训练：solver_v1 -> questioner_v2 -> solver_v2 -> questioner_v3 -> solver_v3
#

echo "=========================================="
echo "R-Zero 继续训练：第1-3轮（完整训练）"
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
echo ""

# 设置模型前缀
Base_model=/data/user5/R-Zero/OctoThinker-3B-Hybrid-Base
Model_abbr=octo_3b

# ================================
# 第1轮：训练 Solver v1
# ================================
echo "=========================================="
echo "开始训练 Solver v1"
echo "=========================================="
bash scripts/solver_train.sh \
    $Base_model \
    ${STORAGE_PATH}/models/${Model_abbr}_questioner_v1/global_step_5/actor/huggingface \
    ${Model_abbr}_solver_v1

if [ $? -ne 0 ]; then
    echo "❌ Solver v1 训练失败"
    exit 1
fi

echo "✓ Solver v1 训练完成"
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

echo "✓ Solver v3 训练完成"
echo ""

# ================================
# 最终评估
# ================================
echo "=========================================="
echo "开始最终评估"
echo "=========================================="
bash evaluation/evaluate_4gpu.bash ${STORAGE_PATH}/models/${Model_abbr}_solver_v3/global_step_15/actor/huggingface

echo ""
echo "=========================================="
echo "🎉 完整的3轮迭代训练全部完成！"
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
echo "训练已完成！可以查看评估结果。"
echo ""


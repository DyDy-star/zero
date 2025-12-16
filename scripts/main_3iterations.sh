#!/bin/bash
#
# R-Zero 3轮迭代训练脚本
# 符合论文标准实验设置
#

Base_model=$1
Model_abbr=$2
echo "Model_abbr: $Model_abbr"
echo "训练配置: 3轮迭代 (v1, v2, v3)"
echo ""

# 第1轮迭代：从base model初始化
echo "=========================================="
echo "开始第1轮迭代 (v1)"
echo "=========================================="
bash scripts/questioner_train_penalty.sh $Base_model $Base_model ${Model_abbr}_questioner_v1
bash scripts/solver_train.sh $Base_model ${STORAGE_PATH}/models/${Model_abbr}_questioner_v1/global_step_5/actor/huggingface ${Model_abbr}_solver_v1
echo "✓ 第1轮迭代完成"
echo ""

# 第2-3轮迭代：co-evolution
for i in {2..3}; do
    prev=$((i-1))
    
    echo "=========================================="
    echo "开始第${i}轮迭代 (v${i})"
    echo "=========================================="
    
    # Train Questioner (Challenger)
    echo "训练 Questioner v${i}..."
    bash scripts/questioner_train_penalty.sh \
        ${STORAGE_PATH}/models/${Model_abbr}_solver_v${prev}/global_step_15/actor/huggingface \
        ${STORAGE_PATH}/models/${Model_abbr}_questioner_v${prev}/global_step_5/actor/huggingface \
        ${Model_abbr}_questioner_v${i}

    # Train Solver
    echo "训练 Solver v${i}..."
    bash scripts/solver_train.sh \
        ${STORAGE_PATH}/models/${Model_abbr}_solver_v${prev}/global_step_15/actor/huggingface \
        ${STORAGE_PATH}/models/${Model_abbr}_questioner_v${i}/global_step_5/actor/huggingface \
        ${Model_abbr}_solver_v${i}
    
    echo "✓ 第${i}轮迭代完成"
    echo ""
done

# 最终评估
echo "=========================================="
echo "开始最终评估"
echo "=========================================="
bash evaluation/evaluate.bash ${STORAGE_PATH}/models/${Model_abbr}_solver_v3/global_step_15/actor/huggingface

echo ""
echo "=========================================="
echo "3轮迭代训练全部完成！"
echo "=========================================="
echo "最终模型位置:"
echo "  Questioner: ${STORAGE_PATH}/models/${Model_abbr}_questioner_v3/global_step_5/actor/huggingface"
echo "  Solver:     ${STORAGE_PATH}/models/${Model_abbr}_solver_v3/global_step_15/actor/huggingface"
echo ""

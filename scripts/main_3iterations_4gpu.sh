#!/bin/bash
#
# R-Zero 3轮迭代训练脚本（4-GPU版本：仅使用GPU 0,1,2,3）
# 符合论文标准实验设置
# 训练完成后自动上传到 HuggingFace Hub
#

Base_model=$1
Model_abbr=$2

# ================================
# 环境配置
# ================================
echo "=========================================="
echo "R-Zero 3轮迭代训练（参数共享版本）"
echo "=========================================="

# 设置 HuggingFace 镜像（解决网络访问问题）
export HF_ENDPOINT=https://hf-mirror.com
echo "✓ HuggingFace 镜像: $HF_ENDPOINT"

# 设置 HuggingFace 用户名
export HUGGINGFACENAME="123YYY123"
echo "✓ HuggingFace 用户名: $HUGGINGFACENAME"

# 配置 wandb 中国镜像
export WANDB_BASE_URL=https://api.bandw.top
echo "✓ WandB 镜像: $WANDB_BASE_URL"

# 限制只使用 GPU 4,5,6,7（物理GPU编号）
export CUDA_VISIBLE_DEVICES=4,5,6,7
echo "✓ 使用 GPU: $CUDA_VISIBLE_DEVICES（物理编号，内部将映射为0,1,2,3）"

# 验证必要的环境变量
if [ -z "$STORAGE_PATH" ]; then
    echo "❌ 错误: STORAGE_PATH 未设置"
    echo "请先运行: export STORAGE_PATH=/data/user5/R-Zero"
    exit 1
fi

echo "✓ 存储路径: $STORAGE_PATH"
echo "✓ 模型缩写: $Model_abbr"
echo "✓ 基础模型: $Base_model"
echo ""

# ================================
# 第1轮迭代：从base model初始化（参数共享）
# ================================
echo "=========================================="
echo "开始第1轮迭代 (v1) - 参数共享"
echo "=========================================="

# 阶段1: 训练 Questioner（提问能力）
echo "【1/2】训练提问能力..."
bash scripts/questioner_train_penalty.sh $Base_model $Base_model ${Model_abbr}_questioner_v1

# 阶段2: 从 Questioner checkpoint 继续训练 Solver（解答能力）
# 关键：使用同一个模型基础，实现参数共享
echo "【2/2】基于提问模型继续训练解答能力（参数共享）..."
bash scripts/solver_train.sh \
    ${STORAGE_PATH}/models/${Model_abbr}_questioner_v1/global_step_5/actor/huggingface \
    ${STORAGE_PATH}/models/${Model_abbr}_questioner_v1/global_step_5/actor/huggingface \
    ${Model_abbr}_unified_v1

echo "✓ 第1轮迭代完成 - 统一模型: ${STORAGE_PATH}/models/${Model_abbr}_unified_v1/global_step_15/actor/huggingface"
echo ""

# ================================
# 第2-3轮迭代：co-evolution（参数共享）
# ================================
for i in {2..3}; do
    prev=$((i-1))
    
    echo "=========================================="
    echo "开始第${i}轮迭代 (v${i}) - 参数共享"
    echo "=========================================="
    
    # 阶段1: 从上一轮的统一模型训练 Questioner（提问能力）
    echo "【1/2】基于上一轮统一模型训练提问能力..."
    bash scripts/questioner_train_penalty.sh \
        ${STORAGE_PATH}/models/${Model_abbr}_unified_v${prev}/global_step_15/actor/huggingface \
        ${STORAGE_PATH}/models/${Model_abbr}_unified_v${prev}/global_step_15/actor/huggingface \
        ${Model_abbr}_questioner_v${i}

    # 阶段2: 从本轮 Questioner checkpoint 继续训练 Solver（解答能力）
    # 关键：实现参数共享
    echo "【2/2】基于提问模型继续训练解答能力（参数共享）..."
    bash scripts/solver_train.sh \
        ${STORAGE_PATH}/models/${Model_abbr}_questioner_v${i}/global_step_5/actor/huggingface \
        ${STORAGE_PATH}/models/${Model_abbr}_questioner_v${i}/global_step_5/actor/huggingface \
        ${Model_abbr}_unified_v${i}
    
    echo "✓ 第${i}轮迭代完成 - 统一模型: ${STORAGE_PATH}/models/${Model_abbr}_unified_v${i}/global_step_15/actor/huggingface"
    echo ""
done

# ================================
# 最终评估
# ================================
echo "=========================================="
echo "开始最终评估（使用统一模型）"
echo "=========================================="
bash evaluation/evaluate_4gpu.bash ${STORAGE_PATH}/models/${Model_abbr}_unified_v3/global_step_15/actor/huggingface

echo ""
echo "=========================================="
echo "3轮迭代训练全部完成！（参数共享版本）"
echo "=========================================="
echo "最终统一模型位置:"
echo "  ${STORAGE_PATH}/models/${Model_abbr}_unified_v3/global_step_15/actor/huggingface"
echo ""
echo "该统一模型同时具备："
echo "  ✓ 提问能力 (Questioner) - 使用 questioner.jinja 格式"
echo "  ✓ 解答能力 (Solver) - 使用 solver.jinja 格式"
echo ""

# ================================
# 上传到 HuggingFace Hub
# ================================
echo "=========================================="
echo "开始上传项目到 HuggingFace Hub"
echo "=========================================="

# 创建上传脚本
cat > /tmp/upload_to_hf.py << 'PYEOF'
from huggingface_hub import login, upload_folder
import os
import sys

print("正在登录 HuggingFace...")
try:
    # 尝试从 tokens.json 读取 token
    import json
    token_file = os.path.join(os.getcwd(), 'tokens.json')
    if os.path.exists(token_file):
        with open(token_file, 'r') as f:
            tokens = json.load(f)
            if 'huggingface' in tokens:
                login(token=tokens['huggingface'])
                print("✓ 使用 tokens.json 中的凭证登录成功")
            else:
                print("⚠️ tokens.json 中未找到 huggingface token")
                login()  # 尝试使用已保存的凭证
    else:
        print("⚠️ tokens.json 不存在，尝试使用已保存的凭证...")
        login()  # 尝试使用已保存的凭证
except Exception as e:
    print(f"❌ 登录失败: {e}")
    print("请先运行: huggingface-cli login")
    sys.exit(1)

print("\n正在上传项目...")
try:
    upload_folder(
        folder_path=".",
        repo_id="123YYY123/R-zero",
        repo_type="dataset",
        ignore_patterns=[
            ".git/*",
            ".gitignore",
            "__pycache__/*",
            "*.pyc",
            ".ipynb_checkpoints/*",
            "wandb/run-*/",  # 只上传最终的wandb结果
            ".cursor/*",
            "*.log",
            "*.tmp"
        ]
    )
    print("✓ 项目上传成功！")
    print("✓ 访问: https://huggingface.co/datasets/123YYY123/R-zero")
except Exception as e:
    print(f"❌ 上传失败: {e}")
    print("\n可以稍后手动上传:")
    print("  python -c \"from huggingface_hub import login, upload_folder; login(); upload_folder(folder_path='.', repo_id='123YYY123/R-zero', repo_type='dataset')\"")
    sys.exit(1)
PYEOF

# 执行上传
cd ${STORAGE_PATH}
python /tmp/upload_to_hf.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓✓✓ 全部完成！"
    echo "=========================================="
    echo "训练结果已上传到 HuggingFace Hub"
    echo "访问: https://huggingface.co/datasets/123YYY123/R-zero"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "⚠️  训练完成但上传失败"
    echo "=========================================="
    echo "请检查 HuggingFace 登录状态并手动上传"
    echo ""
fi

# 清理临时文件
rm -f /tmp/upload_to_hf.py

echo "脚本执行完毕"
echo "训练日志保存在: ${STORAGE_PATH}/wandb/"
echo ""


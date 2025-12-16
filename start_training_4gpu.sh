#!/bin/bash
#
# 快速启动脚本 - 4GPU训练
#

echo "========================================"
echo "R-Zero 4-GPU训练快速启动"
echo "========================================"
echo ""

# 检查环境
if [ "$CONDA_DEFAULT_ENV" != "zero" ]; then
    echo "⚠️  当前不在 zero 环境中"
    echo "正在激活 zero 环境..."
    source activate zero 2>/dev/null || conda activate zero
    if [ $? -ne 0 ]; then
        echo "❌ 无法激活 zero 环境"
        echo "请手动运行: conda activate zero"
        exit 1
    fi
fi

echo "✓ 环境: $CONDA_DEFAULT_ENV"
echo ""

# 设置环境变量
export STORAGE_PATH=/data/user5/R-Zero
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACENAME=123YYY123
export CUDA_VISIBLE_DEVICES=4,5,6,7
export WANDB_BASE_URL=https://api.bandw.top

echo "✓ STORAGE_PATH=$STORAGE_PATH"
echo "✓ HF_ENDPOINT=$HF_ENDPOINT"
echo "✓ HUGGINGFACENAME=$HUGGINGFACENAME"
echo "✓ CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo ""

# 检查 HuggingFace 登录状态
echo "检查 HuggingFace 登录状态..."
huggingface-cli whoami > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "⚠️  未登录 HuggingFace"
    echo ""
    echo "请选择登录方式:"
    echo "  1. 交互式登录: huggingface-cli login"
    echo "  2. 使用tokens.json（如果已配置）"
    echo ""
    read -p "是否现在登录？(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        huggingface-cli login
    else
        echo "⚠️  警告: 未登录可能导致上传失败"
        echo "可以稍后手动上传"
    fi
else
    HF_USER=$(huggingface-cli whoami 2>/dev/null | head -1)
    echo "✓ 已登录 HuggingFace: $HF_USER"
fi
echo ""

# 确认开始训练
echo "========================================"
echo "准备开始训练"
echo "========================================"
echo ""
echo "训练配置:"
echo "  - 基础模型: /data/user5/R-Zero/OctoThinker-3B-Hybrid-Base"
echo "  - 模型前缀: octo_3b"
echo "  - GPU: 4, 5, 6, 7"
echo "  - 轮次: 3轮迭代"
echo "  - 上传到: 123YYY123/R-zero"
echo ""
echo "预计时间: 数小时到数天（取决于数据集大小）"
echo ""

read -p "确认开始训练？(y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

echo ""
echo "========================================"
echo "开始训练..."
echo "========================================"
echo ""

# 启动训练
cd /data/user5/R-Zero
bash scripts/main_3iterations_4gpu.sh \
  /data/user5/R-Zero/OctoThinker-3B-Hybrid-Base \
  octo_3b

echo ""
echo "========================================"
echo "训练脚本执行完毕"
echo "========================================"


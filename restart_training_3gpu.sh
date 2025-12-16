# 启动新训练
echo "1. 启动新训练（使用GPU 4,5,6,7）..."
# 强制设置GPU
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4,5,6,7

# 设置变量
export STORAGE_PATH=/data/user5/R-Zero
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACENAME=123YYY123

cd /data/user5/R-Zero

# 显示配置
echo '=========================================='
echo '训练配置'
echo '=========================================='
echo 'CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES
echo 'CUDA_DEVICE_ORDER='$CUDA_DEVICE_ORDER
python -c 'import torch; print(f\"可见GPU数量: {torch.cuda.device_count()}\")'
echo ''
echo '开始时间: '$(date)
echo ''

# 启动训练
bash scripts/main_3iterations_4gpu.sh /data/user5/R-Zero/OctoThinker-3B-Hybrid-Base octo_3b

echo ''
echo '结束时间: '$(date)
echo ''
echo '按Enter退出'
read
"

sleep 3
echo "✓ 训练已启动"
echo ""

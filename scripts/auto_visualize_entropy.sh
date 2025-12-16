#!/bin/bash
# 自动查找和可视化策略熵
# 
# 用法:
#   bash scripts/auto_visualize_entropy.sh
#   bash scripts/auto_visualize_entropy.sh /path/to/models/dir
#   bash scripts/auto_visualize_entropy.sh /path/to/models/dir output.png

# 设置默认值
MODELS_DIR=${1:-${STORAGE_PATH}/models}
OUTPUT_FILE=${2:-entropy_comparison.png}

echo "搜索熵历史文件在: $MODELS_DIR"

# 查找所有 entropy_history JSON 文件
ENTROPY_FILES=($(find "$MODELS_DIR" -name "entropy_history_*.json" -type f 2>/dev/null))

if [ ${#ENTROPY_FILES[@]} -eq 0 ]; then
    echo "错误: 没有找到任何熵历史文件"
    echo "请确保训练已经运行并生成了熵历史文件"
    exit 1
fi

echo "找到 ${#ENTROPY_FILES[@]} 个熵历史文件:"
for file in "${ENTROPY_FILES[@]}"; do
    echo "  - $file"
done

# 提取标签（从文件名中）
LABELS=()
for file in "${ENTROPY_FILES[@]}"; do
    # 提取 entropy_history_<name>.json 中的 <name>
    basename=$(basename "$file")
    label=$(echo "$basename" | sed 's/entropy_history_//;s/\.json$//')
    LABELS+=("$label")
done

echo "标签: ${LABELS[@]}"

# 构建可视化命令
if [ ${#ENTROPY_FILES[@]} -eq 1 ]; then
    # 单文件模式
    echo "生成单个熵曲线图..."
    python scripts/visualize_entropy.py \
        --entropy_file "${ENTROPY_FILES[0]}" \
        --output "$OUTPUT_FILE"
else
    # 多文件模式
    echo "生成熵对比图..."
    python scripts/visualize_entropy.py \
        --entropy_files "${ENTROPY_FILES[@]}" \
        --labels "${LABELS[@]}" \
        --output "$OUTPUT_FILE" \
        --with_stats
fi

echo "完成! 图像已保存到: $OUTPUT_FILE"


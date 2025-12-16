#!/bin/bash
# HuggingFace 镜像配置脚本

# 设置 HuggingFace 镜像源（适用于中国大陆）
export HF_ENDPOINT=https://hf-mirror.com

# 或者使用 ModelScope 镜像
# export HF_ENDPOINT=https://www.modelscope.cn/api/v1/models

echo "✅ HuggingFace 镜像已设置为: $HF_ENDPOINT"
echo "现在可以运行您的训练脚本了"


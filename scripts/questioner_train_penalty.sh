#!/bin/bash
#
# Questioner 训练脚本（支持恢复训练）
#
# 用法：
#   1. 新训练：
#      bash scripts/questioner_train_penalty.sh <solver_model_path> <questioner_model_path> <save_path>
#
#   2. 恢复训练：
#      bash scripts/questioner_train_penalty.sh <solver_model_path> <questioner_model_path> <save_path> <wandb_run_id> <checkpoint_path>
#
# 示例：
#   # 新训练
#   bash scripts/questioner_train_penalty.sh Qwen/Qwen3-4B-Base Qwen/Qwen3-4B-Base qwen3-4b_questioner_v1
#
#   # 恢复训练
#   bash scripts/questioner_train_penalty.sh Qwen/Qwen3-4B-Base Qwen/Qwen3-4B-Base qwen3-4b_questioner_v1 pz7bgq08 /root/autodl-tmp/R-Zero/models/qwen3-4b_questioner_v1/global_step_4
#

solver_model_path=$1
questioner_model_path=$2
save_path=$3
wandb_run_id=$4  # 可选：用于恢复训练的 wandb run id
resume_checkpoint=$5  # 可选：用于恢复训练的 checkpoint 路径

# 使用 vLLM v0 引擎（更稳定，内存管理更宽松）
export VLLM_USE_V1=0

# 配置 wandb 中国镜像
export WANDB_BASE_URL=https://api.bandw.top

echo "save_path: $save_path"
# 生成唯一 RUN_ID
RUN_ID=$(date +%s%N)
export RUN_ID

echo "RUN_ID=$RUN_ID"

# 设置 WANDB_RUN_ID（如果提供）
if [ ! -z "$wandb_run_id" ]; then
    export WANDB_RUN_ID=$wandb_run_id
    export WANDB_RESUME="allow"
    echo "恢复 Wandb Run ID: $WANDB_RUN_ID"
fi

# 启动 vllm 服务（记录 PID）
bash vllm_service_init/start.sh $solver_model_path $RUN_ID
echo "vLLM services started with RUN_ID=$RUN_ID"

# 开始训练 Questioner
echo "Start training questioner: $questioner_model_path -> $save_path"

# 构建训练命令
# 注意：不覆盖CUDA_VISIBLE_DEVICES，使用父脚本的GPU设置
# 父脚本设置了CUDA_VISIBLE_DEVICES=4,5,6,7，这里看到的GPU 0,1对应物理GPU 4,5
train_cmd="python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=4096 \
    worker.actor.model.model_path=$questioner_model_path \
    worker.actor.model.tokenizer_path=$questioner_model_path \
    trainer.experiment_name=$save_path \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/$save_path \
    trainer.total_epochs=5 \
    worker.reward.reward_function=./examples/reward_function/caller_penalty.py:compute_score \
    trainer.val_freq=-1 \
    trainer.n_gpus_per_node=2 \
    data.format_prompt=./examples/format_prompt/questioner.jinja \
    worker.rollout.n=4 \
    worker.actor.global_batch_size=128 \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.actor.offload.offload_params=true \
    worker.actor.offload.offload_optimizer=true \
    trainer.max_steps=5 \
    trainer.save_freq=1 \
    trainer.save_limit=1"

# 如果提供了 checkpoint 路径，添加恢复参数
if [ ! -z "$resume_checkpoint" ]; then
    train_cmd="$train_cmd trainer.load_checkpoint_path=$resume_checkpoint"
    echo "从 checkpoint 恢复训练: $resume_checkpoint"
fi

# 执行训练命令
eval $train_cmd

sleep 5

# 自动找到最新的 checkpoint 并合并模型
echo "查找最新的 checkpoint..."
latest_checkpoint=$(ls -d ${STORAGE_PATH}/models/$save_path/global_step_* 2>/dev/null | sort -V | tail -1)

if [ -z "$latest_checkpoint" ]; then
    echo "警告: 没有找到任何 checkpoint，跳过模型合并"
else
    checkpoint_name=$(basename $latest_checkpoint)
    echo "找到最新的 checkpoint: $checkpoint_name"
    
    if [ -d "$latest_checkpoint/actor" ]; then
        echo "开始合并模型..."
        python scripts/model_merger.py --local_dir $latest_checkpoint/actor
        
        if [ $? -eq 0 ]; then
            echo "模型合并成功: $latest_checkpoint/actor/huggingface"
        else
            echo "警告: 模型合并失败"
        fi
    else
        echo "警告: 找不到 actor 目录: $latest_checkpoint/actor"
    fi
fi

sleep 10

# 更安全的清理方式：只杀死当前 RUN_ID 相关的进程
echo "清理当前训练的 vLLM 服务进程..."
pkill -f "start_vllm_server.py --port 5000"
pkill -f "start_vllm_server.py --port 5001"

# 等待进程完全结束
sleep 2

echo "questioner training finished"

# 自动备份已禁用
# 如需备份，请手动运行:
# bash scripts/backup_training_data.sh model <model_name> <checkpoint>

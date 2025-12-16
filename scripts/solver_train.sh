#!/bin/bash
set -e  # 任何命令失败立即退出

solver_model_path=$1
questioner_model_path=$2
experiment_name=$3

# 配置 wandb 中国镜像
export WANDB_BASE_URL=https://api.bandw.top

# 设置 Python 路径
export PYTHONPATH=/data/user5/R-Zero:$PYTHONPATH

echo $STORAGE_PATH

echo "start train solver $experiment_name $solver_model_path $questioner_model_path" 

export VLLM_DISABLE_COMPILE_CACHE=1
echo 'start generate question'
bash question_generate/question_generate.bash $questioner_model_path 8000 $experiment_name
echo 'start evaluate generated question'
bash question_evaluate/evaluate.sh $solver_model_path $experiment_name
echo 'start upload'
# 论文参数: 保留 score 在 0.25-0.75 之间的问题 (δ=0.25, 对应 3-7 个答案匹配多数投票)
python question_evaluate/upload.py --repo_name ${experiment_name} --max_score 0.75 --min_score 0.25 --experiment_name ${experiment_name}
echo 'start train'

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=4096 \
    worker.actor.model.model_path=$solver_model_path \
    trainer.experiment_name=${experiment_name} \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/${experiment_name}/ \
    data.train_files=${HUGGINGFACENAME}/${experiment_name}@train \
    trainer.total_epochs=15 \
    trainer.max_steps=15 \
    data.format_prompt=./examples/format_prompt/solver.jinja \
    worker.reward.reward_function=./examples/reward_function/math.py:compute_score \
    trainer.val_freq=4 \
    trainer.save_freq=1 \
    trainer.save_limit=1 \
    trainer.n_gpus_per_node=4 \
    worker.rollout.tensor_parallel_size=4 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \

echo "merging model"
python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/${experiment_name}/global_step_15/actor

sleep 10

# 验证模型是否成功生成
if [ ! -d "${STORAGE_PATH}/models/${experiment_name}/global_step_15/actor/huggingface" ]; then
    echo "❌ 错误: 模型合并失败或模型文件不存在"
    exit 1
fi

echo "✓ Solver 模型已成功生成: ${STORAGE_PATH}/models/${experiment_name}/global_step_15/actor/huggingface"
echo "solver training finished"

# 自动备份已禁用
# 如需备份，请手动运行:
# bash scripts/backup_training_data.sh model ${experiment_name} 15
# bash scripts/backup_training_data.sh questions ${experiment_name}

# 评估已禁用（可在所有训练完成后手动运行）
# bash evaluation/evaluate.bash ${STORAGE_PATH}/models/${experiment_name}/global_step_15/actor/huggingface

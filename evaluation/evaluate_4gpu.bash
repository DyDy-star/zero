#!/bin/bash
export VLLM_DISABLE_COMPILE_CACHE=1

MODEL_NAMES=(
  "/data/user5/R-Zero/models/octo_3b_questioner_v1/global_step_5/actor/huggingface"
  "/data/user5/R-Zero/models/octo_3b_unified_v1/global_step_15/actor/huggingface"
  "/data/user5/R-Zero/models/octo_3b_questioner_v2/global_step_5/actor/huggingface"
  "/data/user5/R-Zero/models/octo_3b_unified_v2/global_step_15/actor/huggingface"
  "/data/user5/R-Zero/models/octo_3b_questioner_v3/global_step_5/actor/huggingface"
  "/data/user5/R-Zero/models/octo_3b_unified_v3/global_step_15/actor/huggingface"
)

TASKS=(
  "math"
  "gsm8k" 
  "amc"
)

GPU_QUEUE=(3)
echo "Available GPUs: ${GPU_QUEUE[@]}"

declare -A pids

start_job() {
  local gpu_id="$1"
  local model="$2"
  local task="$3"

  echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Start task [${task}] with model [${model}] on GPU [${gpu_id}] ..."

  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  python evaluation/generate.py --model "${model}" --dataset "${task}" &

  pids["${gpu_id}"]=$!
}

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    echo "==> Processing model: ${MODEL_NAME}"
    TASK_INDEX=0
    NUM_TASKS=${#TASKS[@]}

    while :; do
        while [ ${#GPU_QUEUE[@]} -gt 0 ] && [ ${TASK_INDEX} -lt ${NUM_TASKS} ]; do
            gpu_id="${GPU_QUEUE[0]}"
            GPU_QUEUE=("${GPU_QUEUE[@]:1}")

            task="${TASKS[${TASK_INDEX}]}"
            ((TASK_INDEX++))

            start_job "$gpu_id" "$MODEL_NAME" "$task"
        done

        if [ ${TASK_INDEX} -ge ${NUM_TASKS} ] && [ ${#pids[@]} -eq 0 ]; then
            break
        fi

        for gpu_id in "${!pids[@]}"; do
            pid="${pids[$gpu_id]}"
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] GPU [${gpu_id}] job finished with PID [${pid}]."
                unset pids["$gpu_id"]
                GPU_QUEUE+=("$gpu_id")
            fi
        done

        sleep 1
    done
    
    # 每个模型评估完成后进行结果检查
    echo "==> Running results_recheck for model: ${MODEL_NAME}"
    python evaluation/results_recheck.py --model_name "${MODEL_NAME}"
done

# 额外评估任务已禁用（仅使用GPU 0,1）
# CUDA_VISIBLE_DEVICES=4 python evaluation/eval_supergpqa.py --model_path $model_name &
# PID1=$!
# CUDA_VISIBLE_DEVICES=5 python evaluation/eval_bbeh.py --model_path $model_name &
# PID2=$!
# CUDA_VISIBLE_DEVICES=6 python evaluation/eval_mmlupro.py --model_path $model_name &
# PID3=$!

# wait $PID1 $PID2 $PID3

echo "==> All tasks have finished!"


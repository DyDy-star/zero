#!/bin/bash

model_name=$1
save_name=$2

pids=()

# 使用GPU 4,5,6,7 并行评估（4倍速度）
# 使用物理GPU 4,5,6,7（父脚本设置了CUDA_VISIBLE_DEVICES=4,5,6,7，这里直接用物理编号）
# 论文参数: m = 10 个答案采样
echo "启动4GPU并行评估 - 使用 GPU 4, 5, 6, 7"
CUDA_VISIBLE_DEVICES=4 python question_evaluate/evaluate.py --model $model_name --suffix 0 --save_name $save_name --num_samples 10 &
pids[0]=$!
CUDA_VISIBLE_DEVICES=5 python question_evaluate/evaluate.py --model $model_name --suffix 1 --save_name $save_name --num_samples 10 &
pids[1]=$!
CUDA_VISIBLE_DEVICES=6 python question_evaluate/evaluate.py --model $model_name --suffix 2 --save_name $save_name --num_samples 10 &
pids[2]=$!
CUDA_VISIBLE_DEVICES=7 python question_evaluate/evaluate.py --model $model_name --suffix 3 --save_name $save_name --num_samples 10 &
pids[3]=$!

wait ${pids[0]}
echo "✓ Task 0 (GPU 4) finished."
wait ${pids[1]}
echo "✓ Task 1 (GPU 5) finished."
wait ${pids[2]}
echo "✓ Task 2 (GPU 6) finished."
wait ${pids[3]}
echo "✓ Task 3 (GPU 7) finished."
echo "✓ 所有评估任务完成！"

# load the model name from the command line
model_name=$1
num_samples=$2
save_name=$3
export VLLM_DISABLE_COMPILE_CACHE=1

# 使用物理GPU 4,5,6,7 并行生成（4倍速度）
# 每个GPU生成1/4的问题数
quarter_samples=$((num_samples / 4))
echo "启动4GPU并行问题生成 - 每个GPU: $quarter_samples 个问题"
echo "GPU 4: $quarter_samples, GPU 5: $quarter_samples, GPU 6: $quarter_samples, GPU 7: $quarter_samples"

pids=()
CUDA_VISIBLE_DEVICES=4 python question_generate/question_generate.py --model $model_name --suffix 0 --num_samples $quarter_samples --save_name $save_name &
pids[0]=$!
CUDA_VISIBLE_DEVICES=5 python question_generate/question_generate.py --model $model_name --suffix 1 --num_samples $quarter_samples --save_name $save_name &
pids[1]=$!
CUDA_VISIBLE_DEVICES=6 python question_generate/question_generate.py --model $model_name --suffix 2 --num_samples $quarter_samples --save_name $save_name &
pids[2]=$!
CUDA_VISIBLE_DEVICES=7 python question_generate/question_generate.py --model $model_name --suffix 3 --num_samples $quarter_samples --save_name $save_name &
pids[3]=$!

wait ${pids[0]}
echo "✓ Task 0 (GPU 4) finished."
wait ${pids[1]}
echo "✓ Task 1 (GPU 5) finished."
wait ${pids[2]}
echo "✓ Task 2 (GPU 6) finished."
wait ${pids[3]}
echo "✓ Task 3 (GPU 7) finished."
echo "✓ 所有生成任务完成！总计: $num_samples 个问题"

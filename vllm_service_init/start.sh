model_path=$1
run_id=$2
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=0

# 父脚本设置了CUDA_VISIBLE_DEVICES=4,5,6,7，内部映射为0,1,2,3
# questioner训练使用逻辑GPU 0,1（物理4,5）
# vLLM服务需要使用物理GPU 6,7

# 临时保存父进程的 CUDA_VISIBLE_DEVICES
PARENT_CUDA_DEVICES=$CUDA_VISIBLE_DEVICES

# vLLM 服务直接使用物理 GPU 6,7
CUDA_VISIBLE_DEVICES=6 python vllm_service_init/start_vllm_server.py --port 5000 --model_path $model_path --gpu_mem_util 0.9 &
CUDA_VISIBLE_DEVICES=7 python vllm_service_init/start_vllm_server.py --port 5001 --model_path $model_path --gpu_mem_util 0.9 &

# 恢复父进程的设置
export CUDA_VISIBLE_DEVICES=$PARENT_CUDA_DEVICES
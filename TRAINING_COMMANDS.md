# R-Zero 训练命令速查表

## 🚀 快速启动命令

### 方法1：使用一键启动脚本（推荐）

```bash
# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export STORAGE_PATH=/data/user5/R-Zero
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACENAME="YOUR_HF_TOKEN_HERE"

# 切换到项目目录
cd /data/user5/R-Zero

# 启动训练（使用默认模型）
bash start_r_zero_training.sh

# 或者指定base model和模型名称
bash start_r_zero_training.sh /data/user5/R-Zero/OctoThinker-3B-Hybrid-Base octo_3b
```

### 方法2：直接调用main.sh

```bash
# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export STORAGE_PATH=/data/user5/R-Zero
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACENAME="YOUR_HF_TOKEN_HERE"
export VLLM_USE_V1=0
export VLLM_DISABLE_COMPILE_CACHE=1

# 切换到项目目录
cd /data/user5/R-Zero

# 开始训练
bash scripts/main.sh <base_model_path> <model_abbr>

# 示例
bash scripts/main.sh /data/user5/R-Zero/OctoThinker-3B-Hybrid-Base octo_3b
```

---

## 📋 完整命令（复制粘贴版）

### 使用OctoThinker-3B-Hybrid-Base训练

```bash
# 一次性复制所有命令
export CUDA_VISIBLE_DEVICES=0,1,2,3 && \
export STORAGE_PATH=/data/user5/R-Zero && \
export HF_ENDPOINT=https://hf-mirror.com && \
export HUGGINGFACENAME="YOUR_HF_TOKEN_HERE" && \
export VLLM_USE_V1=0 && \
export VLLM_DISABLE_COMPILE_CACHE=1 && \
cd /data/user5/R-Zero && \
bash scripts/main.sh /data/user5/R-Zero/OctoThinker-3B-Hybrid-Base octo_3b
```

### 使用自定义模型训练

```bash
# 修改BASE_MODEL为你的模型路径
BASE_MODEL="/path/to/your/model"
MODEL_NAME="your_model_name"

export CUDA_VISIBLE_DEVICES=0,1,2,3 && \
export STORAGE_PATH=/data/user5/R-Zero && \
export HF_ENDPOINT=https://hf-mirror.com && \
export HUGGINGFACENAME="YOUR_HF_TOKEN_HERE" && \
export VLLM_USE_V1=0 && \
export VLLM_DISABLE_COMPILE_CACHE=1 && \
cd /data/user5/R-Zero && \
bash scripts/main.sh "$BASE_MODEL" "$MODEL_NAME"
```

---

## 🔧 训练中的监控命令

### 监控GPU使用

```bash
# 实时监控GPU状态
watch -n 1 nvidia-smi

# 预期看到：
# GPU 0-1: 高负载（训练进程）
# GPU 2-3: 中等负载（vLLM推理服务）
```

### 查看训练进度

```bash
# 查看wandb日志
# 访问 https://wandb.ai 查看训练曲线

# 查看本地日志
tail -f /data/user5/R-Zero/wandb/latest-run/logs/debug.log

# 检查模型保存
ls -lh /data/user5/R-Zero/models/
```

### 检查vLLM服务状态

```bash
# 检查vLLM进程
ps aux | grep vllm_server

# 测试vLLM服务
curl http://0.0.0.0:5000/hello?name=test
curl http://0.0.0.0:5001/hello?name=test
```

---

## 📊 训练流程说明

### 完整5轮迭代流程

```
第1轮 (v1):
  1. Base Model → Train Questioner_v1 (5 steps)
  2. Questioner_v1 → Generate questions → Train Solver_v1 (15 steps)

第2轮 (v2):
  1. Solver_v1 + Questioner_v1 → Train Questioner_v2 (5 steps)
  2. Solver_v1 + Questioner_v2 → Generate questions → Train Solver_v2 (15 steps)

第3轮 (v3):
  1. Solver_v2 + Questioner_v2 → Train Questioner_v3 (5 steps)
  2. Solver_v2 + Questioner_v3 → Generate questions → Train Solver_v3 (15 steps)

第4轮 (v4):
  1. Solver_v3 + Questioner_v3 → Train Questioner_v4 (5 steps)
  2. Solver_v3 + Questioner_v4 → Generate questions → Train Solver_v4 (15 steps)

第5轮 (v5):
  1. Solver_v4 + Questioner_v4 → Train Questioner_v5 (5 steps)
  2. Solver_v4 + Questioner_v5 → Generate questions → Train Solver_v5 (15 steps)

最终评估:
  Evaluate Solver_v5 on benchmarks
```

### 训练时间估算

| 阶段 | 步数 | 预计时间（4-GPU） | 说明 |
|------|------|------------------|------|
| Questioner训练 | 5 steps | ~30-60分钟 | 取决于模型大小 |
| Question生成 | 1000问题 | ~20-40分钟 | vLLM推理 |
| Question评估 | 1000问题 | ~20-40分钟 | vLLM推理 |
| Solver训练 | 15 steps | ~1-2小时 | 取决于模型大小 |
| **单轮总计** | - | **~2-4小时** | - |
| **5轮总计** | - | **~10-20小时** | - |

---

## 🛠️ 故障排查

### 1. GPU显存不足 (OOM)

```bash
# 方案1: 降低batch size
# 编辑 scripts/questioner_train_penalty.sh
worker.actor.global_batch_size=64  # 从128降到64

# 方案2: 启用更多CPU offload
# 编辑 examples/config.yaml
worker.actor.fsdp.enable_cpu_offload: true
```

### 2. vLLM服务启动失败

```bash
# 检查GPU是否被占用
nvidia-smi

# 手动清理进程
pkill -9 python

# 重新启动
bash vllm_service_init/start.sh <model_path> <run_id>
```

### 3. 训练中断需要恢复

```bash
# Questioner恢复训练
bash scripts/questioner_train_penalty.sh \
    <solver_model> <questioner_model> <save_path> \
    <wandb_run_id> <checkpoint_path>

# 示例
bash scripts/questioner_train_penalty.sh \
    /data/user5/R-Zero/OctoThinker-3B-Hybrid-Base \
    /data/user5/R-Zero/OctoThinker-3B-Hybrid-Base \
    octo_3b_questioner_v1 \
    pz7bgq08 \
    /data/user5/R-Zero/models/octo_3b_questioner_v1/global_step_3
```

### 4. 清理所有进程重新开始

```bash
# 停止所有训练相关进程
pkill -9 python

# 确认GPU已释放
nvidia-smi

# 等待几秒后重新启动
sleep 5
bash start_r_zero_training.sh
```

---

## 📈 训练后评估

### 评估最终模型

```bash
# 自动评估（main.sh末尾会自动运行）
bash evaluation/evaluate.bash /data/user5/R-Zero/models/octo_3b_solver_v5/global_step_15/actor/huggingface

# 或者手动评估特定benchmark
bash eval_math.sh
bash eval_gsm8k.sh
bash eval_amc.sh
```

### 查看评估结果

```bash
# 结果保存位置
ls -lh /data/user5/R-Zero/evaluation/

# 查看具体结果
cat /data/user5/R-Zero/evaluation/results_math.json
cat /data/user5/R-Zero/evaluation/results_gsm8k.json
```

---

## 🎯 验证配置

### 验证训练配置是否正确

```bash
# 运行验证脚本
bash compare_configs.sh

# 查看详细验证报告
cat REWARD_AND_TRAINING_VERIFICATION.md
cat PAPER_CONFIG_COMPARISON.md
```

### 检查关键配置

```bash
# 检查GPU配置
echo $CUDA_VISIBLE_DEVICES  # 应该输出: 0,1,2,3

# 检查questioner配置
grep -E "max_steps|global_batch_size|rollout.n" scripts/questioner_train_penalty.sh

# 检查solver配置
grep "max_steps" scripts/solver_train.sh

# 检查迭代次数
grep "for i in" scripts/main.sh
```

---

## 📚 相关文档

- **配置验证报告**: `REWARD_AND_TRAINING_VERIFICATION.md`
- **论文配置对比**: `PAPER_CONFIG_COMPARISON.md`
- **GPU配置说明**: `GPU_CONFIG_4GPU.md`
- **原始文件对比**: `DIFF_WITH_ORIGINAL.md`
- **配置验证**: `CONFIG_VERIFICATION.md`

---

**最后更新**: 2025-12-05  
**版本**: 4-GPU配置（使用GPU 0-3）  
**状态**: ✅ 所有配置已验证，完全符合论文要求


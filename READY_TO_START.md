# ✅ 准备就绪！可以开始训练了

## 🎯 已完成的配置

### 1. ✅ HuggingFace 用户名设置
- **用户名**: `123YYY123`
- **上传仓库**: `123YYY123/R-zero` (dataset类型)
- **位置**: 已集成到 `scripts/main_3iterations_4gpu.sh`

### 2. ✅ GPU 限制
- **限制为**: GPU 0, 1, 2, 3（共4个）
- **不会使用**: GPU 4, 5, 6, 7
- **位置**: 
  - `scripts/main_3iterations_4gpu.sh` (主脚本)
  - `evaluation/evaluate_4gpu.bash` (评估脚本)

### 3. ✅ 自动上传功能
- **时机**: 3轮训练和评估全部完成后
- **方式**: 使用 huggingface_hub 库
- **认证**: 优先使用 tokens.json，其次使用已保存的凭证
- **目标**: https://huggingface.co/datasets/123YYY123/R-zero

### 4. ✅ 修复的报错
- ✅ HFValidationError (本地路径问题) - 已修复
- ✅ ModuleNotFoundError (evaluation模块) - 已修复
- ✅ 所有评估脚本已更新

---

## 🚀 三种启动方式

### 方式1: 快速启动（推荐）⭐

```bash
cd /data/user5/R-Zero
bash start_training_4gpu.sh
```

这个脚本会：
- 自动激活 zero 环境
- 检查 HuggingFace 登录状态
- 显示配置信息
- 请求确认后开始训练

### 方式2: 在screen中运行（推荐用于长时间训练）⭐

```bash
# 1. 创建screen会话
screen -S r_zero_4gpu

# 2. 运行快速启动脚本
cd /data/user5/R-Zero
bash start_training_4gpu.sh

# 3. 训练开始后，退出screen（训练会继续）
# 按 Ctrl+A, 然后按 D

# 4. 稍后重新连接查看进度
screen -r r_zero_4gpu
```

### 方式3: 直接运行主脚本

```bash
# 设置环境
conda activate zero
export STORAGE_PATH=/data/user5/R-Zero

# 运行训练
cd /data/user5/R-Zero
bash scripts/main_3iterations_4gpu.sh \
  /data/user5/R-Zero/OctoThinker-3B-Hybrid-Base \
  octo_3b
```

---

## 📁 创建的新文件

| 文件 | 说明 | 权限 |
|------|------|------|
| `scripts/main_3iterations_4gpu.sh` | 主训练脚本（4-GPU + 自动上传） | ✅ 可执行 |
| `evaluation/evaluate_4gpu.bash` | 评估脚本（4-GPU限制） | ✅ 可执行 |
| `start_training_4gpu.sh` | 快速启动脚本 | ✅ 可执行 |
| `START_4GPU_TRAINING.md` | 详细使用文档 | 📄 文档 |
| `READY_TO_START.md` | 本文件（总结） | 📄 文档 |

---

## 🔍 训练前检查清单

在开始训练前，请确认：

- [ ] **环境**: 已激活 zero 环境
  ```bash
  conda activate zero
  ```

- [ ] **HuggingFace**: 已登录
  ```bash
  huggingface-cli login
  # 或确保 tokens.json 包含有效token
  ```

- [ ] **GPU**: GPU 0-3 可用
  ```bash
  nvidia-smi
  ```

- [ ] **磁盘**: 有足够空间（约40GB）
  ```bash
  df -h /data/user5
  ```

- [ ] **模型**: 基础模型存在
  ```bash
  ls /data/user5/R-Zero/OctoThinker-3B-Hybrid-Base
  ```

---

## 📊 训练流程预览

完整流程需要数小时到数天：

```
1. 环境配置 (1分钟)
   ├─ 设置 HUGGINGFACENAME=123YYY123
   ├─ 限制 GPU=0,1,2,3
   └─ 验证环境变量

2. 第1轮迭代 (数小时)
   ├─ Questioner v1 训练
   └─ Solver v1 训练

3. 第2轮迭代 (数小时)
   ├─ Questioner v2 训练
   └─ Solver v2 训练

4. 第3轮迭代 (数小时)
   ├─ Questioner v3 训练
   └─ Solver v3 训练

5. 最终评估 (数小时)
   └─ 在4个GPU上评估7个数据集

6. 自动上传 (数分钟到数小时)
   └─ 上传到 123YYY123/R-zero
```

---

## 🎯 预期结果

训练成功完成后，您将获得：

### 本地模型（6个）
```
/data/user5/R-Zero/models/
├── octo_3b_questioner_v1/global_step_5/actor/huggingface/
├── octo_3b_solver_v1/global_step_15/actor/huggingface/
├── octo_3b_questioner_v2/global_step_5/actor/huggingface/
├── octo_3b_solver_v2/global_step_15/actor/huggingface/
├── octo_3b_questioner_v3/global_step_5/actor/huggingface/
└── octo_3b_solver_v3/global_step_15/actor/huggingface/
```

### HuggingFace 仓库
- **地址**: https://huggingface.co/datasets/123YYY123/R-zero
- **内容**: 完整项目（包括模型、脚本、配置、结果）
- **类型**: dataset

### 评估结果
- `final_results.jsonl` - 所有数据集的评估结果
- wandb 日志 - 完整的训练记录

---

## 🔧 训练监控

### 实时查看进度

```bash
# GPU 使用情况
watch -n 1 nvidia-smi

# 训练日志
tail -f /data/user5/R-Zero/wandb/latest-run/logs/debug.log

# 已生成的模型
ls -lh /data/user5/R-Zero/models/
```

### 在screen中查看

```bash
# 连接到screen会话
screen -r r_zero_4gpu

# 查看正在运行的进程
ps aux | grep python | grep -E "verl|vllm|question"

# 退出screen（不中断训练）
# 按 Ctrl+A, 然后按 D
```

---

## ⚠️ 重要提示

### GPU 验证
训练期间，运行 `nvidia-smi` 应该：
- ✅ GPU 0-3: 有进程，高使用率
- ✅ GPU 4-7: **无进程，空闲**

如果看到 GPU 4-7 也在使用，说明配置未生效！

### 上传验证
训练完成后：
1. 访问 https://huggingface.co/datasets/123YYY123/R-zero
2. 检查文件是否完整
3. 确认模型大小正确

### 中断恢复
如果训练中断：
- Questioner 训练支持恢复（需要 wandb_run_id 和 checkpoint_path）
- Solver 训练需要重新开始该轮
- 建议在 screen 中运行以防止意外中断

---

## 📝 快速命令参考

```bash
# 启动训练（快速方式）
cd /data/user5/R-Zero
bash start_training_4gpu.sh

# 在screen中启动
screen -S r_zero_4gpu
bash start_training_4gpu.sh
# Ctrl+A, D 退出

# 重连screen
screen -r r_zero_4gpu

# 查看GPU
nvidia-smi

# 查看模型
ls -lh /data/user5/R-Zero/models/

# 查看日志
tail -f /data/user5/R-Zero/wandb/latest-run/logs/debug.log

# 手动上传（如果自动上传失败）
cd /data/user5/R-Zero
python -c "from huggingface_hub import login, upload_folder; login(); upload_folder(folder_path='.', repo_id='123YYY123/R-zero', repo_type='dataset')"
```

---

## 🎉 就这些！

**一切已准备就绪！** 

现在您可以：

### 选项A: 立即开始训练
```bash
cd /data/user5/R-Zero
bash start_training_4gpu.sh
```

### 选项B: 在screen中运行（推荐）
```bash
screen -S r_zero_4gpu
cd /data/user5/R-Zero
bash start_training_4gpu.sh
```

**祝训练顺利！** 🚀

---

## 📚 相关文档

- `START_4GPU_TRAINING.md` - 详细的训练指南
- `HUGGINGFACENAME_GUIDE.md` - HuggingFace 配置说明
- `SCREEN_ERROR_FIXED.md` - 已修复的错误列表
- `PRE_TRAINING_CHECK.md` - 训练前检查报告

有任何问题，查看这些文档或检查 wandb 日志！


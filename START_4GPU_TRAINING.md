# 🚀 4-GPU训练启动指南

## ✅ 已配置内容

### 1. HuggingFace 用户名
- ✅ 已设置: `HUGGINGFACENAME="123YYY123"`
- ✅ 上传目标: `123YYY123/R-zero` (dataset类型)

### 2. GPU 限制
- ✅ 已限制为: **GPU 0, 1, 2, 3** (共4个)
- ✅ 不会使用 GPU 4, 5, 6, 7

### 3. 自动上传
- ✅ 训练完成后自动上传整个项目到 HuggingFace Hub
- ✅ 仓库类型: dataset
- ✅ 访问地址: https://huggingface.co/datasets/123YYY123/R-zero

---

## 📋 使用前准备

### 步骤1: 登录 HuggingFace

```bash
# 方式1: 交互式登录（推荐）
huggingface-cli login

# 方式2: 使用tokens.json中的token（如果已配置）
# 脚本会自动检测并使用
```

### 步骤2: 验证tokens.json

```bash
cat /data/user5/R-Zero/tokens.json
```

应该包含：
```json
{
  "huggingface": "hf_your_token_here"
}
```

### 步骤3: 激活环境

```bash
conda activate zero
```

---

## 🚀 启动训练

### 方式1: 在screen中运行（推荐）

```bash
# 1. 创建新的screen会话
screen -S r_zero_training

# 2. 设置环境变量
export STORAGE_PATH=/data/user5/R-Zero

# 3. 启动训练
cd /data/user5/R-Zero
bash scripts/main_3iterations_4gpu.sh \
  /data/user5/R-Zero/OctoThinker-3B-Hybrid-Base \
  octo_3b

# 4. 退出screen (训练会继续运行)
# 按 Ctrl+A, 然后按 D
```

### 方式2: 直接运行

```bash
# 设置环境变量
export STORAGE_PATH=/data/user5/R-Zero

# 启动训练
cd /data/user5/R-Zero
bash scripts/main_3iterations_4gpu.sh \
  /data/user5/R-Zero/OctoThinker-3B-Hybrid-Base \
  octo_3b
```

---

## 📊 训练流程

脚本将自动执行以下步骤：

### 1️⃣ 环境配置
- ✅ 设置 HUGGINGFACENAME=123YYY123
- ✅ 限制 GPU 为 0,1,2,3
- ✅ 验证环境变量

### 2️⃣ 第1轮迭代
- 训练 questioner_v1
- 训练 solver_v1

### 3️⃣ 第2轮迭代
- 训练 questioner_v2
- 训练 solver_v2

### 4️⃣ 第3轮迭代
- 训练 questioner_v3
- 训练 solver_v3

### 5️⃣ 最终评估
- 在4个GPU上并行评估7个数据集
- 运行额外的评估任务

### 6️⃣ 自动上传
- 登录 HuggingFace (使用tokens.json)
- 上传整个项目到 123YYY123/R-zero
- 排除临时文件和缓存

---

## 📁 生成的模型

训练完成后将生成6个模型：

```
models/
├── octo_3b_questioner_v1/global_step_5/actor/huggingface/
├── octo_3b_solver_v1/global_step_15/actor/huggingface/
├── octo_3b_questioner_v2/global_step_5/actor/huggingface/
├── octo_3b_solver_v2/global_step_15/actor/huggingface/
├── octo_3b_questioner_v3/global_step_5/actor/huggingface/
└── octo_3b_solver_v3/global_step_15/actor/huggingface/
```

---

## 🔍 监控训练

### 查看实时日志

```bash
# 如果在screen中运行
screen -r r_zero_training

# 查看wandb日志
tail -f /data/user5/R-Zero/wandb/latest-run/logs/debug.log

# 查看GPU使用情况
watch -n 1 nvidia-smi
```

### 检查训练进度

```bash
# 查看已生成的模型
ls -lh /data/user5/R-Zero/models/

# 查看wandb运行记录
ls -lht /data/user5/R-Zero/wandb/ | head -10
```

---

## ⚠️ GPU使用验证

训练过程中，各阶段的GPU使用情况：

| 阶段 | GPU使用 | 说明 |
|------|---------|------|
| Questioner训练 | 0, 1 | 2个GPU并行 |
| 问题生成 | 2 | 单GPU |
| 问题评估 | 2 | 单GPU |
| Solver训练 | 0, 1, 2, 3 | 4个GPU并行 |
| 最终评估 | 0, 1, 2, 3 | 4个GPU并行 |

**验证方法**:
```bash
# 运行训练时执行
nvidia-smi

# 应该只看到 GPU 0, 1, 2, 3 有进程
# GPU 4, 5, 6, 7 应该是空闲的
```

---

## 🌐 上传后的结果

训练完成后，项目会自动上传到：
- **仓库**: https://huggingface.co/datasets/123YYY123/R-zero
- **类型**: dataset
- **可见性**: public (默认)

### 上传的内容包括:
- ✅ 所有训练好的模型
- ✅ 训练脚本
- ✅ 配置文件
- ✅ 评估结果
- ❌ 不包括：.git, __pycache__, wandb临时文件, .cursor等

### 手动上传（如果自动上传失败）

```bash
cd /data/user5/R-Zero

python << 'EOF'
from huggingface_hub import login, upload_folder

# 登录
login()

# 上传
upload_folder(
    folder_path=".",
    repo_id="123YYY123/R-zero",
    repo_type="dataset"
)
EOF
```

---

## 🆘 常见问题

### Q1: 如何确认GPU限制生效？
```bash
# 在另一个终端运行
watch -n 1 nvidia-smi

# 应该只看到GPU 0-3有活动
```

### Q2: 上传失败怎么办？
```bash
# 检查登录状态
huggingface-cli whoami

# 重新登录
huggingface-cli login

# 手动上传（见上面的手动上传部分）
```

### Q3: 训练中断了怎么办？
```bash
# Questioner训练支持恢复
bash scripts/questioner_train_penalty.sh \
  <solver_path> \
  <questioner_path> \
  <save_path> \
  <wandb_run_id> \
  <checkpoint_path>
```

### Q4: 如何只运行某一轮？
```bash
# 第1轮
bash scripts/questioner_train_penalty.sh \
  /data/user5/R-Zero/OctoThinker-3B-Hybrid-Base \
  /data/user5/R-Zero/OctoThinker-3B-Hybrid-Base \
  octo_3b_questioner_v1

bash scripts/solver_train.sh \
  /data/user5/R-Zero/OctoThinker-3B-Hybrid-Base \
  ${STORAGE_PATH}/models/octo_3b_questioner_v1/global_step_5/actor/huggingface \
  octo_3b_solver_v1
```

---

## ✅ 检查清单

训练前确认：
- [ ] 已激活 zero 环境
- [ ] 已设置 STORAGE_PATH
- [ ] 已登录 HuggingFace
- [ ] tokens.json 包含有效的 token
- [ ] GPU 0-3 可用
- [ ] 有足够的磁盘空间（约40GB）

---

## 📞 需要帮助？

如有问题，检查：
1. wandb日志: `/data/user5/R-Zero/wandb/latest-run/logs/debug.log`
2. 脚本输出
3. GPU状态: `nvidia-smi`
4. HuggingFace状态: `huggingface-cli whoami`

---

**准备好了？立即开始训练！** 🚀

```bash
export STORAGE_PATH=/data/user5/R-Zero
cd /data/user5/R-Zero
bash scripts/main_3iterations_4gpu.sh \
  /data/user5/R-Zero/OctoThinker-3B-Hybrid-Base \
  octo_3b
```


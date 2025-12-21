# Question 熵日志保存指南

## 新的文件命名格式

### 格式说明

文件名格式：`question_entropy_{version}_step{step}.json`

示例：
```
/data/user5/R-Zero/question_entropy_logs/
├── question_entropy_v1_step1.json
├── question_entropy_v1_step2.json
├── question_entropy_v1_step3.json
├── question_entropy_v1_step4.json
├── question_entropy_v1_step5.json
├── question_entropy_v2_step1.json
├── question_entropy_v2_step2.json
└── ...
```

### 版本（Version）vs 步骤（Step）

- **Version (v1, v2, v3...)**: 对应不同的训练版本，如 questionerv1, questionerv2
- **Step (1, 2, 3...)**: 同一版本内的训练步骤，每次调用奖励函数时递增

## 设置方法

### 1. 设置环境变量

在训练脚本中设置 `MODEL_VERSION`：

```bash
# 对于 questionerv1 训练
export MODEL_VERSION=v1
bash continue_from_questionerv2.sh

# 对于 questionerv2 训练
export MODEL_VERSION=v2
bash continue_from_questionerv2.sh
```

### 2. 在训练脚本中设置

编辑 `continue_from_questionerv2.sh`，添加：

```bash
# 设置模型版本
export MODEL_VERSION=v1  # 或 v2, v3...

# 其他环境变量...
export STORAGE_PATH=/data/user5/R-Zero
```

## JSON 文件格式

### 不再包含时间戳

```json
{
  "metadata": {
    "version": "v1",
    "step": 3,
    "total_samples": 32,
    "valid_samples": 30
  },
  "questions": [
    {
      "sample_id": 0,
      "question_text": "如何求解这个方程？",
      "token_entropies": [2.34, 1.89, 2.45, 2.12],
      "mean_entropy": 2.20,
      "token_count": 4,
      "score": 0.85,
      "penalty": 0.15
    }
  ]
}
```

## 训练输出

### 新的输出格式

```
============================================================
Question 熵统计 (标签内部的文本，已strip):
  有效样本数: 30
  平均熵: 2.3456
  熵标准差: 0.1234
  最小熵: 1.8900
  最大熵: 2.7800
  Question Token 数量统计:
    平均: 42.5
    最小: 18
    最大: 65
============================================================

✓ Question 熵信息已保存到: /data/user5/R-Zero/question_entropy_logs/question_entropy_v1_step3.json
  - 版本: v1
  - Step: 3
  - 有效样本数: 30/32
```

## 分析方法

### 分析单个文件

```bash
python scripts/analyze_question_entropy.py --file question_entropy_logs/question_entropy_v1_step3.json
```

### 分析所有步骤

```bash
# 分析所有版本的所有步骤
python scripts/analyze_question_entropy.py --dir question_entropy_logs/

# 只分析 v1 版本的所有步骤
python scripts/analyze_question_entropy.py --dir question_entropy_logs/ --version v1

# 只分析 v2 版本的所有步骤
python scripts/analyze_question_entropy.py --dir question_entropy_logs/ --version v2
```

### 分析输出示例

```
找到 5 个熵日志文件
✓ 已保存: entropy_over_training.png
✓ 已保存: metrics_over_training.png
✓ 已保存: entropy_distribution_comparison.png
✓ 已保存: entropy_vs_quality.png
✓ 已保存: summary.json

============================================================
分析完成！所有图表和报告已保存到: question_entropy_logs/analysis/
============================================================

训练步骤范围: v1_step1 - v1_step5 (共 5 步)

熵变化:
  初始平均熵 (v1_step1): 2.8456
  最终平均熵 (v1_step5): 2.1234
  变化: -0.7222 (-25.4%)

质量变化:
  初始 Score (v1_step1): 0.7523
  最终 Score (v1_step5): 0.8932
  变化: +0.1409 (+18.7%)
```

## 完整训练示例

### questionerv1 训练（5个steps）

```bash
# 设置版本
export MODEL_VERSION=v1
export STORAGE_PATH=/data/user5/R-Zero

# 运行训练
bash continue_from_questionerv2.sh

# 训练过程中会自动保存：
# - question_entropy_v1_step1.json  (第1次调用奖励函数)
# - question_entropy_v1_step2.json  (第2次调用奖励函数)
# - question_entropy_v1_step3.json  (第3次调用奖励函数)
# - question_entropy_v1_step4.json  (第4次调用奖励函数)
# - question_entropy_v1_step5.json  (第5次调用奖励函数)
```

### questionerv2 训练

```bash
# 切换到 v2
export MODEL_VERSION=v2

# 运行训练
bash continue_from_questionerv2.sh

# 会自动保存：
# - question_entropy_v2_step1.json
# - question_entropy_v2_step2.json
# - ...
```

### 对比不同版本

```bash
# 分析 v1 版本
python scripts/analyze_question_entropy.py --dir question_entropy_logs/ --version v1 --output analysis_v1/

# 分析 v2 版本
python scripts/analyze_question_entropy.py --dir question_entropy_logs/ --version v2 --output analysis_v2/

# 现在可以对比两个版本的训练曲线
```

## 注意事项

### 1. Step 计数器

- Step 计数器在每次调用 `compute_score()` 时自动递增
- 如果程序重启，Step 会从已有文件的最大值+1 继续
- 同一个 version 下的 step 是连续的

### 2. 手动管理版本

如果需要从头开始一个新版本：

```bash
# 方法1: 更改 MODEL_VERSION
export MODEL_VERSION=v3

# 方法2: 删除旧的日志文件（谨慎！）
rm question_entropy_logs/question_entropy_v1_*.json

# 方法3: 移动到备份目录
mkdir -p question_entropy_logs/backup_v1/
mv question_entropy_logs/question_entropy_v1_*.json question_entropy_logs/backup_v1/
```

### 3. 目录结构建议

```
/data/user5/R-Zero/
├── question_entropy_logs/
│   ├── question_entropy_v1_step1.json
│   ├── question_entropy_v1_step2.json
│   ├── ...
│   ├── question_entropy_v2_step1.json
│   ├── question_entropy_v2_step2.json
│   ├── ...
│   ├── analysis_v1/          # v1 分析结果
│   │   ├── entropy_over_training.png
│   │   └── summary.json
│   └── analysis_v2/          # v2 分析结果
│       ├── entropy_over_training.png
│       └── summary.json
└── models/
    ├── octo_3b_questioner_v1_no_format/
    └── octo_3b_questioner_v2/
```

## Python 脚本使用

### 读取特定版本的所有步骤

```python
import json
import glob

# 读取 v1 的所有步骤
v1_files = sorted(glob.glob('/data/user5/R-Zero/question_entropy_logs/question_entropy_v1_step*.json'))

for file in v1_files:
    with open(file, 'r') as f:
        data = json.load(f)
    print(f"{data['metadata']['version']}_step{data['metadata']['step']}: 平均熵 = {np.mean([q['mean_entropy'] for q in data['questions']]):.4f}")
```

### 对比两个版本

```python
import json
import glob
import numpy as np

def get_version_entropies(version):
    files = sorted(glob.glob(f'/data/user5/R-Zero/question_entropy_logs/question_entropy_{version}_step*.json'))
    entropies = []
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
        mean_entropies = [q['mean_entropy'] for q in data['questions']]
        entropies.append(np.mean(mean_entropies))
    return entropies

v1_entropies = get_version_entropies('v1')
v2_entropies = get_version_entropies('v2')

print(f"v1 最终熵: {v1_entropies[-1]:.4f}")
print(f"v2 最终熵: {v2_entropies[-1]:.4f}")
print(f"改进: {(v2_entropies[-1] - v1_entropies[-1]):.4f}")
```

## 常见问题

### Q: 如何知道当前是哪个 step？

查看最新的文件：
```bash
ls -lt question_entropy_logs/question_entropy_v1_*.json | head -1
```

### Q: 可以跨版本合并分析吗？

可以，只需不指定 `--version` 参数：
```bash
python scripts/analyze_question_entropy.py --dir question_entropy_logs/
```

### Q: 如何清理旧数据？

```bash
# 备份
mkdir -p backups/
mv question_entropy_logs/ backups/entropy_logs_$(date +%Y%m%d)/

# 重新开始
mkdir question_entropy_logs/
```










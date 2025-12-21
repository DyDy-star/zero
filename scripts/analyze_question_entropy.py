#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析 Question 熵日志的脚本

用法:
    # 分析单个文件
    python scripts/analyze_question_entropy.py --file question_entropy_logs/question_entropy_round_0001_*.json
    
    # 分析所有轮次
    python scripts/analyze_question_entropy.py --dir question_entropy_logs/
    
    # 生成详细报告
    python scripts/analyze_question_entropy.py --dir question_entropy_logs/ --output report.html
"""

import json
import glob
import os
import argparse
import numpy as np
from typing import List, Dict
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from collections import defaultdict


def load_entropy_file(filepath: str) -> Dict:
    """加载单个熵日志文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_single_file(filepath: str):
    """分析单个文件并打印统计信息"""
    data = load_entropy_file(filepath)
    metadata = data['metadata']
    questions = data['questions']
    
    print(f"\n{'='*60}")
    print(f"文件: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    version = metadata.get('version', f"v{metadata.get('training_round', 1)}")
    step = metadata.get('step', 1)
    print(f"版本: {version}")
    print(f"Step: {step}")
    print(f"总样本数: {metadata['total_samples']}")
    print(f"有效样本数: {metadata['valid_samples']}")
    
    if questions:
        mean_entropies = [q['mean_entropy'] for q in questions]
        token_counts = [q['token_count'] for q in questions]
        scores = [q['score'] for q in questions]
        penalties = [q['penalty'] for q in questions]
        
        print(f"\n熵统计:")
        print(f"  平均熵: {np.mean(mean_entropies):.4f} ± {np.std(mean_entropies):.4f}")
        print(f"  熵范围: [{np.min(mean_entropies):.4f}, {np.max(mean_entropies):.4f}]")
        print(f"  中位数: {np.median(mean_entropies):.4f}")
        
        print(f"\nToken 统计:")
        print(f"  平均 token 数: {np.mean(token_counts):.1f} ± {np.std(token_counts):.1f}")
        print(f"  Token 范围: [{int(np.min(token_counts))}, {int(np.max(token_counts))}]")
        
        print(f"\n质量统计:")
        print(f"  平均 score: {np.mean(scores):.4f}")
        print(f"  平均 penalty: {np.mean(penalties):.4f}")
        
        # 关联分析
        corr_entropy_score = np.corrcoef(mean_entropies, scores)[0, 1]
        corr_entropy_penalty = np.corrcoef(mean_entropies, penalties)[0, 1]
        print(f"\n相关性分析:")
        print(f"  熵 vs Score: {corr_entropy_score:.4f}")
        print(f"  熵 vs Penalty: {corr_entropy_penalty:.4f}")
        
        # 显示几个示例
        print(f"\n示例 Questions (前3个):")
        for i, q in enumerate(questions[:3]):
            print(f"\n  [{i+1}] {q['question_text'][:50]}...")
            print(f"      熵: {q['mean_entropy']:.4f}, Tokens: {q['token_count']}, Score: {q['score']:.4f}")


def analyze_all_files(directory: str, output_dir: str = None, version: str = None):
    """分析目录中的所有文件并生成可视化
    
    Args:
        directory: 日志文件目录
        output_dir: 输出目录（可选）
        version: 只分析特定版本，如 "v1"（可选）
    """
    if version:
        # Only analyze files for specific version
        pattern = os.path.join(directory, f'question_entropy_{version}_step*.json')
    else:
        # Analyze all files
        pattern = os.path.join(directory, 'question_entropy_v*_step*.json')
    
    files = glob.glob(pattern)
    
    # Sort by version and step
    def get_version_step(filepath):
        try:
            basename = os.path.basename(filepath)
            # Extract v1_step3 -> ("v1", 3)
            parts = basename.replace('question_entropy_', '').replace('.json', '').split('_step')
            version_str = parts[0]  # "v1"
            step_num = int(parts[1]) if len(parts) > 1 else 0
            # Convert v1 -> 1 for sorting
            version_num = int(version_str.replace('v', ''))
            return (version_num, step_num)
        except:
            return (0, 0)
    
    files = sorted(files, key=get_version_step)
    
    if not files:
        print(f"错误: 在 {directory} 中没有找到熵日志文件")
        return
    
    print(f"\n找到 {len(files)} 个熵日志文件")
    
    # 收集数据
    labels = []  # 存储标签，如 "v1_step1", "v1_step2"
    avg_entropies = []
    std_entropies = []
    min_entropies = []
    max_entropies = []
    avg_token_counts = []
    avg_scores = []
    avg_penalties = []
    
    for filepath in files:
        data = load_entropy_file(filepath)
        metadata = data['metadata']
        questions = data['questions']
        
        if not questions:
            continue
        
        mean_entropies = [q['mean_entropy'] for q in questions]
        token_counts = [q['token_count'] for q in questions]
        scores = [q['score'] for q in questions]
        penalties = [q['penalty'] for q in questions]
        
        # Create label like "v1_step3"
        version = metadata.get('version', f"v{metadata.get('training_round', 1)}")
        step = metadata.get('step', metadata.get('training_round', 1))
        label = f"{version}_step{step}"
        labels.append(label)
        avg_entropies.append(np.mean(mean_entropies))
        std_entropies.append(np.std(mean_entropies))
        min_entropies.append(np.min(mean_entropies))
        max_entropies.append(np.max(mean_entropies))
        avg_token_counts.append(np.mean(token_counts))
        avg_scores.append(np.mean(scores))
        avg_penalties.append(np.mean(penalties))
    
    # 创建输出目录
    if output_dir is None:
        output_dir = os.path.join(directory, 'analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘图 1: 熵随训练变化
    x_positions = list(range(len(labels)))
    
    plt.figure(figsize=(14, 6))
    plt.plot(x_positions, avg_entropies, marker='o', linewidth=2, label='Mean Entropy')
    plt.fill_between(x_positions, 
                      np.array(avg_entropies) - np.array(std_entropies),
                      np.array(avg_entropies) + np.array(std_entropies),
                      alpha=0.3, label='± Std Dev')
    plt.plot(x_positions, min_entropies, '--', alpha=0.5, label='Min Entropy')
    plt.plot(x_positions, max_entropies, '--', alpha=0.5, label='Max Entropy')
    plt.xlabel('Training Progress', fontsize=12)
    plt.ylabel('Entropy', fontsize=12)
    plt.title('Question Entropy over Training', fontsize=14)
    plt.xticks(x_positions, labels, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'entropy_over_training.png'), dpi=150)
    plt.close()
    print(f"✓ 已保存: entropy_over_training.png")
    
    # 绘图 2: 多指标对比
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 2.1 平均熵
    axes[0, 0].plot(x_positions, avg_entropies, marker='o', color='blue')
    axes[0, 0].set_xlabel('Training Progress')
    axes[0, 0].set_ylabel('Mean Entropy')
    axes[0, 0].set_title('Mean Entropy')
    axes[0, 0].set_xticks(x_positions)
    axes[0, 0].set_xticklabels(labels, rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2.2 平均 token 数
    axes[0, 1].plot(x_positions, avg_token_counts, marker='s', color='green')
    axes[0, 1].set_xlabel('Training Progress')
    axes[0, 1].set_ylabel('Avg Token Count')
    axes[0, 1].set_title('Average Token Count')
    axes[0, 1].set_xticks(x_positions)
    axes[0, 1].set_xticklabels(labels, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 2.3 平均分数
    axes[1, 0].plot(x_positions, avg_scores, marker='^', color='orange')
    axes[1, 0].set_xlabel('Training Progress')
    axes[1, 0].set_ylabel('Avg Score')
    axes[1, 0].set_title('Average Answer Quality Score')
    axes[1, 0].set_xticks(x_positions)
    axes[1, 0].set_xticklabels(labels, rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 2.4 平均 penalty
    axes[1, 1].plot(x_positions, avg_penalties, marker='d', color='red')
    axes[1, 1].set_xlabel('Training Progress')
    axes[1, 1].set_ylabel('Avg Penalty')
    axes[1, 1].set_title('Average Clustering Penalty')
    axes[1, 1].set_xticks(x_positions)
    axes[1, 1].set_xticklabels(labels, rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_over_training.png'), dpi=150)
    plt.close()
    print(f"✓ 已保存: metrics_over_training.png")
    
    # 绘图 3: 熵分布直方图（第一轮 vs 最后一轮）
    if len(files) >= 2:
        first_data = load_entropy_file(files[0])
        last_data = load_entropy_file(files[-1])
        
        first_entropies = [q['mean_entropy'] for q in first_data['questions']]
        last_entropies = [q['mean_entropy'] for q in last_data['questions']]
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(first_entropies, bins=20, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('Mean Entropy')
        plt.ylabel('Frequency')
        first_version = first_data['metadata'].get('version', f"v{first_data['metadata'].get('training_round', 1)}")
        first_step = first_data['metadata'].get('step', first_data['metadata'].get('training_round', 1))
        plt.title(f"{first_version}_step{first_step} (First)")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(last_entropies, bins=20, alpha=0.7, color='green', edgecolor='black')
        plt.xlabel('Mean Entropy')
        plt.ylabel('Frequency')
        last_version = last_data['metadata'].get('version', f"v{last_data['metadata'].get('training_round', 1)}")
        last_step = last_data['metadata'].get('step', last_data['metadata'].get('training_round', 1))
        plt.title(f"{last_version}_step{last_step} (Last)")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'entropy_distribution_comparison.png'), dpi=150)
        plt.close()
        print(f"✓ 已保存: entropy_distribution_comparison.png")
    
    # 绘图 4: 熵 vs 质量指标散点图（使用最新一轮数据）
    latest_data = load_entropy_file(files[-1])
    latest_questions = latest_data['questions']
    
    if latest_questions:
        entropies = [q['mean_entropy'] for q in latest_questions]
        scores = [q['score'] for q in latest_questions]
        penalties = [q['penalty'] for q in latest_questions]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        latest_version = latest_data['metadata'].get('version', f"v{latest_data['metadata'].get('training_round', 1)}")
        latest_step = latest_data['metadata'].get('step', latest_data['metadata'].get('training_round', 1))
        latest_label = f"{latest_version}_step{latest_step}"
        
        axes[0].scatter(entropies, scores, alpha=0.6, s=50)
        axes[0].set_xlabel('Mean Entropy')
        axes[0].set_ylabel('Score (Answer Accuracy)')
        axes[0].set_title(f'Entropy vs Quality ({latest_label})')
        axes[0].grid(True, alpha=0.3)
        
        # 添加趋势线
        z = np.polyfit(entropies, scores, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(entropies), max(entropies), 100)
        axes[0].plot(x_trend, p(x_trend), "r--", alpha=0.8, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
        axes[0].legend()
        
        axes[1].scatter(entropies, penalties, alpha=0.6, s=50, color='orange')
        axes[1].set_xlabel('Mean Entropy')
        axes[1].set_ylabel('Penalty (Clustering)')
        axes[1].set_title(f'Entropy vs Diversity ({latest_label})')
        axes[1].grid(True, alpha=0.3)
        
        # 添加趋势线
        z = np.polyfit(entropies, penalties, 1)
        p = np.poly1d(z)
        axes[1].plot(x_trend, p(x_trend), "r--", alpha=0.8, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'entropy_vs_quality.png'), dpi=150)
        plt.close()
        print(f"✓ 已保存: entropy_vs_quality.png")
    
    # 生成汇总统计报告
    summary = {
        'total_steps': len(labels),
        'labels': labels,
        'avg_entropies': avg_entropies,
        'std_entropies': std_entropies,
        'avg_token_counts': avg_token_counts,
        'avg_scores': avg_scores,
        'avg_penalties': avg_penalties
    }
    
    summary_file = os.path.join(output_dir, 'summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✓ 已保存: summary.json")
    
    print(f"\n{'='*60}")
    print(f"分析完成！所有图表和报告已保存到: {output_dir}")
    print(f"{'='*60}")
    
    # 打印汇总统计
    print(f"\n训练步骤范围: {labels[0]} - {labels[-1]} (共 {len(labels)} 步)")
    print(f"\n熵变化:")
    print(f"  初始平均熵 ({labels[0]}): {avg_entropies[0]:.4f}")
    print(f"  最终平均熵 ({labels[-1]}): {avg_entropies[-1]:.4f}")
    print(f"  变化: {avg_entropies[-1] - avg_entropies[0]:.4f} ({(avg_entropies[-1]/avg_entropies[0]-1)*100:+.1f}%)")
    
    print(f"\n质量变化:")
    print(f"  初始 Score ({labels[0]}): {avg_scores[0]:.4f}")
    print(f"  最终 Score ({labels[-1]}): {avg_scores[-1]:.4f}")
    print(f"  变化: {avg_scores[-1] - avg_scores[0]:.4f} ({(avg_scores[-1]/avg_scores[0]-1)*100:+.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='分析 Question 熵日志')
    parser.add_argument('--file', type=str, help='单个日志文件路径')
    parser.add_argument('--dir', type=str, help='日志文件目录')
    parser.add_argument('--output', type=str, help='输出目录（可选）')
    parser.add_argument('--version', type=str, help='只分析特定版本，如 v1, v2（可选）')
    
    args = parser.parse_args()
    
    if args.file:
        analyze_single_file(args.file)
    elif args.dir:
        analyze_all_files(args.dir, args.output, args.version)
    else:
        parser.print_help()
        print("\n错误: 必须指定 --file 或 --dir 参数")


if __name__ == '__main__':
    main()


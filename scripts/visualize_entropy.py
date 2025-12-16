#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略熵可视化脚本

用法:
    python scripts/visualize_entropy.py --entropy_file path/to/entropy_history.json --output path/to/output.png
    
    或指定多个文件（例如同时可视化questioner和solver）:
    python scripts/visualize_entropy.py --entropy_files path/to/questioner_entropy.json path/to/solver_entropy.json --labels Questioner Solver --output path/to/output.png
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端


def load_entropy_history(file_path):
    """加载熵历史数据"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def plot_entropy_single(entropy_data, output_path, title="Policy Entropy Over Training Steps"):
    """绘制单个熵曲线"""
    steps = [item['step'] for item in entropy_data]
    entropies = [item['entropy'] for item in entropy_data]
    
    plt.figure(figsize=(12, 6))
    plt.plot(steps, entropies, linewidth=2, marker='o', markersize=4)
    plt.xlabel('Training Step', fontsize=14)
    plt.ylabel('Policy Entropy', fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图像
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Entropy plot saved to: {output_path}")
    plt.close()


def plot_entropy_multiple(entropy_data_list, labels, output_path, title="Policy Entropy Comparison"):
    """绘制多个熵曲线对比"""
    plt.figure(figsize=(14, 7))
    
    colors = plt.cm.tab10(range(len(entropy_data_list)))
    
    for i, (entropy_data, label) in enumerate(zip(entropy_data_list, labels)):
        steps = [item['step'] for item in entropy_data]
        entropies = [item['entropy'] for item in entropy_data]
        plt.plot(steps, entropies, linewidth=2, marker='o', markersize=3, 
                label=label, color=colors[i], alpha=0.8)
    
    plt.xlabel('Training Step', fontsize=14)
    plt.ylabel('Policy Entropy', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图像
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Entropy comparison plot saved to: {output_path}")
    plt.close()


def plot_entropy_stats(entropy_data_list, labels, output_path):
    """绘制熵统计信息（平均值、最大值、最小值）"""
    import numpy as np
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (entropy_data, label) in enumerate(zip(entropy_data_list, labels)):
        entropies = [item['entropy'] for item in entropy_data]
        
        # 平均熵
        axes[0].bar(i, np.mean(entropies), label=label, alpha=0.7)
        # 最大熵
        axes[1].bar(i, np.max(entropies), label=label, alpha=0.7)
        # 最小熵
        axes[2].bar(i, np.min(entropies), label=label, alpha=0.7)
    
    axes[0].set_title('Average Entropy', fontsize=14)
    axes[0].set_ylabel('Entropy', fontsize=12)
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].set_title('Maximum Entropy', fontsize=14)
    axes[1].set_ylabel('Entropy', fontsize=12)
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    axes[2].set_title('Minimum Entropy', fontsize=14)
    axes[2].set_ylabel('Entropy', fontsize=12)
    axes[2].set_xticks(range(len(labels)))
    axes[2].set_xticklabels(labels, rotation=45)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    stats_output = output_path.replace('.png', '_stats.png')
    plt.savefig(stats_output, dpi=300, bbox_inches='tight')
    print(f"Entropy statistics plot saved to: {stats_output}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize policy entropy over training steps')
    parser.add_argument('--entropy_file', type=str, help='Path to single entropy history JSON file')
    parser.add_argument('--entropy_files', nargs='+', help='Paths to multiple entropy history JSON files')
    parser.add_argument('--labels', nargs='+', help='Labels for multiple entropy files')
    parser.add_argument('--output', type=str, default='entropy_plot.png', help='Output plot file path')
    parser.add_argument('--title', type=str, help='Custom plot title')
    parser.add_argument('--with_stats', action='store_true', help='Also generate statistics plot')
    
    args = parser.parse_args()
    
    # 单文件模式
    if args.entropy_file:
        entropy_data = load_entropy_history(args.entropy_file)
        title = args.title or "Policy Entropy Over Training Steps"
        plot_entropy_single(entropy_data, args.output, title)
    
    # 多文件模式
    elif args.entropy_files:
        if not args.labels or len(args.labels) != len(args.entropy_files):
            print("Warning: Number of labels must match number of files. Using default labels.")
            args.labels = [f"Model {i+1}" for i in range(len(args.entropy_files))]
        
        entropy_data_list = [load_entropy_history(f) for f in args.entropy_files]
        title = args.title or "Policy Entropy Comparison"
        plot_entropy_multiple(entropy_data_list, args.labels, args.output, title)
        
        if args.with_stats:
            plot_entropy_stats(entropy_data_list, args.labels, args.output)
    
    else:
        print("Error: Please provide either --entropy_file or --entropy_files")
        parser.print_help()


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析高熵问题中高熵token与低熵token的比例关系
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_all_token_entropies(base_path):
    """加载所有版本的token熵数据"""
    all_data = {'v1': {'questions': [], 'all_token_entropies': []}, 
                'v2': {'questions': [], 'all_token_entropies': []}, 
                'v3': {'questions': [], 'all_token_entropies': []}}
    
    for version in ['v1', 'v2', 'v3']:
        print(f"\n加载 {version} 数据...")
        
        for step in range(1, 6):
            file_path = Path(base_path) / f"question_entropy_{version}_step{step}.json"
            print(f"  - Step {step}...")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for question in data['questions']:
                if question['token_entropies']:
                    # 收集所有token熵值
                    all_data[version]['all_token_entropies'].extend(question['token_entropies'])
                    
                    # 保存问题信息
                    avg_entropy = np.mean(question['token_entropies'])
                    all_data[version]['questions'].append({
                        'sample_id': question['sample_id'],
                        'step': step,
                        'question_text': question['question_text'],
                        'token_entropies': question['token_entropies'],
                        'avg_entropy': avg_entropy
                    })
        
        print(f"  总问题数: {len(all_data[version]['questions'])}")
        print(f"  总token数: {len(all_data[version]['all_token_entropies'])}")
    
    return all_data

def calculate_thresholds(all_data):
    """计算每个版本的阈值"""
    thresholds = {}
    
    print(f"\n{'='*60}")
    print("计算阈值")
    print(f"{'='*60}")
    
    for version in ['v1', 'v2', 'v3']:
        # Token熵的80%分位数（高熵token定义）
        token_entropies = all_data[version]['all_token_entropies']
        token_threshold = np.percentile(token_entropies, 80)
        
        # 问题平均熵的80%分位数（高熵问题定义）
        question_avg_entropies = [q['avg_entropy'] for q in all_data[version]['questions']]
        question_threshold = np.percentile(question_avg_entropies, 80)
        
        thresholds[version] = {
            'token_entropy_threshold': token_threshold,
            'question_entropy_threshold': question_threshold
        }
        
        print(f"\n{version.upper()}:")
        print(f"  Token熵80%分位数: {token_threshold:.4f}")
        print(f"  问题平均熵80%分位数: {question_threshold:.4f}")
        print(f"  Token总数: {len(token_entropies)}")
        print(f"  问题总数: {len(question_avg_entropies)}")
    
    return thresholds

def analyze_token_ratio_in_questions(all_data, thresholds):
    """分析每个问题中高熵token和低熵token的比例"""
    results = {}
    
    print(f"\n{'='*60}")
    print("分析高熵问题中的token比例")
    print(f"{'='*60}")
    
    for version in ['v1', 'v2', 'v3']:
        print(f"\n处理 {version.upper()}...")
        
        token_threshold = thresholds[version]['token_entropy_threshold']
        question_threshold = thresholds[version]['question_entropy_threshold']
        
        # 筛选高熵问题
        high_entropy_questions = [
            q for q in all_data[version]['questions'] 
            if q['avg_entropy'] >= question_threshold
        ]
        
        # 分析每个高熵问题中的token比例
        question_analyses = []
        high_token_ratios = []
        low_token_ratios = []
        high_token_counts = []
        low_token_counts = []
        total_token_counts = []
        
        for q in high_entropy_questions:
            token_entropies = q['token_entropies']
            total_tokens = len(token_entropies)
            
            # 统计高熵和低熵token数量
            high_entropy_tokens = sum(1 for e in token_entropies if e >= token_threshold)
            low_entropy_tokens = total_tokens - high_entropy_tokens
            
            # 计算比例
            high_ratio = high_entropy_tokens / total_tokens if total_tokens > 0 else 0
            low_ratio = low_entropy_tokens / total_tokens if total_tokens > 0 else 0
            
            high_token_ratios.append(high_ratio)
            low_token_ratios.append(low_ratio)
            high_token_counts.append(high_entropy_tokens)
            low_token_counts.append(low_entropy_tokens)
            total_token_counts.append(total_tokens)
            
            question_analyses.append({
                'sample_id': q['sample_id'],
                'step': q['step'],
                'avg_entropy': q['avg_entropy'],
                'total_tokens': total_tokens,
                'high_entropy_tokens': high_entropy_tokens,
                'low_entropy_tokens': low_entropy_tokens,
                'high_token_ratio': high_ratio,
                'low_token_ratio': low_ratio,
                'question_preview': q['question_text'][:200]
            })
        
        # 统计信息
        print(f"  高熵问题数: {len(high_entropy_questions)}")
        print(f"  平均高熵token比例: {np.mean(high_token_ratios):.4f} ({np.mean(high_token_ratios)*100:.2f}%)")
        print(f"  平均低熵token比例: {np.mean(low_token_ratios):.4f} ({np.mean(low_token_ratios)*100:.2f}%)")
        print(f"  高熵token比例中位数: {np.median(high_token_ratios):.4f}")
        print(f"  高熵token比例标准差: {np.std(high_token_ratios):.4f}")
        
        # 按高熵token比例分组统计
        ratio_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ratio_dist = np.histogram(high_token_ratios, bins=ratio_bins)[0]
        
        print(f"\n  高熵token比例分布:")
        for i in range(len(ratio_bins)-1):
            count = ratio_dist[i]
            percentage = count / len(high_token_ratios) * 100 if len(high_token_ratios) > 0 else 0
            print(f"    {ratio_bins[i]:.1f}-{ratio_bins[i+1]:.1f}: {count} ({percentage:.1f}%)")
        
        results[version] = {
            'high_entropy_question_count': len(high_entropy_questions),
            'question_analyses': question_analyses,
            'statistics': {
                'mean_high_token_ratio': float(np.mean(high_token_ratios)),
                'median_high_token_ratio': float(np.median(high_token_ratios)),
                'std_high_token_ratio': float(np.std(high_token_ratios)),
                'mean_low_token_ratio': float(np.mean(low_token_ratios)),
                'mean_high_token_count': float(np.mean(high_token_counts)),
                'mean_low_token_count': float(np.mean(low_token_counts)),
                'mean_total_tokens': float(np.mean(total_token_counts))
            },
            'ratio_distribution': {
                'bins': ratio_bins,
                'counts': ratio_dist.tolist()
            },
            'all_high_ratios': high_token_ratios,
            'all_low_ratios': low_token_ratios,
            'all_high_counts': high_token_counts,
            'all_low_counts': low_token_counts,
            'all_total_counts': total_token_counts
        }
    
    return results

def find_patterns(results):
    """寻找高熵token比例的规律"""
    print(f"\n{'='*60}")
    print("探索规律")
    print(f"{'='*60}")
    
    patterns = {}
    
    for version in ['v1', 'v2', 'v3']:
        print(f"\n{version.upper()} 规律分析:")
        
        analyses = results[version]['question_analyses']
        
        # 1. 按问题平均熵分组，看高熵token比例的变化
        entropy_groups = {
            'very_high': [a for a in analyses if a['avg_entropy'] >= np.percentile([a['avg_entropy'] for a in analyses], 90)],
            'high': [a for a in analyses if np.percentile([a['avg_entropy'] for a in analyses], 70) <= a['avg_entropy'] < np.percentile([a['avg_entropy'] for a in analyses], 90)],
            'medium': [a for a in analyses if a['avg_entropy'] < np.percentile([a['avg_entropy'] for a in analyses], 70)]
        }
        
        print(f"\n  规律1: 问题平均熵与高熵token比例的关系")
        for group_name, group_data in entropy_groups.items():
            if group_data:
                avg_ratio = np.mean([a['high_token_ratio'] for a in group_data])
                avg_entropy = np.mean([a['avg_entropy'] for a in group_data])
                print(f"    {group_name}: 平均熵={avg_entropy:.4f}, 高熵token比例={avg_ratio:.4f} ({avg_ratio*100:.2f}%)")
        
        # 2. 按问题长度（token数）分组
        length_groups = {
            'short': [a for a in analyses if a['total_tokens'] < np.percentile([a['total_tokens'] for a in analyses], 33)],
            'medium': [a for a in analyses if np.percentile([a['total_tokens'] for a in analyses], 33) <= a['total_tokens'] < np.percentile([a['total_tokens'] for a in analyses], 67)],
            'long': [a for a in analyses if a['total_tokens'] >= np.percentile([a['total_tokens'] for a in analyses], 67)]
        }
        
        print(f"\n  规律2: 问题长度与高熵token比例的关系")
        for group_name, group_data in length_groups.items():
            if group_data:
                avg_ratio = np.mean([a['high_token_ratio'] for a in group_data])
                avg_length = np.mean([a['total_tokens'] for a in group_data])
                print(f"    {group_name}: 平均长度={avg_length:.1f} tokens, 高熵token比例={avg_ratio:.4f} ({avg_ratio*100:.2f}%)")
        
        # 3. 相关性分析
        avg_entropies = [a['avg_entropy'] for a in analyses]
        high_ratios = [a['high_token_ratio'] for a in analyses]
        token_counts = [a['total_tokens'] for a in analyses]
        
        corr_entropy_ratio = np.corrcoef(avg_entropies, high_ratios)[0, 1]
        corr_length_ratio = np.corrcoef(token_counts, high_ratios)[0, 1]
        
        print(f"\n  规律3: 相关性分析")
        print(f"    问题平均熵 vs 高熵token比例: r={corr_entropy_ratio:.4f}")
        print(f"    问题长度 vs 高熵token比例: r={corr_length_ratio:.4f}")
        
        patterns[version] = {
            'entropy_groups': {k: {
                'count': len(v),
                'avg_high_token_ratio': float(np.mean([a['high_token_ratio'] for a in v])) if v else 0,
                'avg_question_entropy': float(np.mean([a['avg_entropy'] for a in v])) if v else 0
            } for k, v in entropy_groups.items()},
            'length_groups': {k: {
                'count': len(v),
                'avg_high_token_ratio': float(np.mean([a['high_token_ratio'] for a in v])) if v else 0,
                'avg_length': float(np.mean([a['total_tokens'] for a in v])) if v else 0
            } for k, v in length_groups.items()},
            'correlations': {
                'entropy_vs_ratio': float(corr_entropy_ratio),
                'length_vs_ratio': float(corr_length_ratio)
            }
        }
    
    return patterns

def create_visualizations(results, patterns, thresholds):
    """创建可视化图表"""
    output_dir = Path('/data/user5/R-Zero/question_entropy_analysis_results')
    
    # 图1: 高熵token比例分布（三个版本对比）
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for idx, version in enumerate(['v1', 'v2', 'v3']):
        # 第一行：分布直方图
        ax1 = axes[0, idx]
        high_ratios = results[version]['all_high_ratios']
        
        ax1.hist(high_ratios, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
        ax1.axvline(np.mean(high_ratios), color='red', linestyle='--', linewidth=2,
                   label=f'平均值: {np.mean(high_ratios):.3f}')
        ax1.axvline(np.median(high_ratios), color='orange', linestyle='--', linewidth=2,
                   label=f'中位数: {np.median(high_ratios):.3f}')
        ax1.set_xlabel('高熵Token比例', fontsize=12)
        ax1.set_ylabel('问题数量', fontsize=12)
        ax1.set_title(f'{version.upper()} - 高熵Token比例分布', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 第二行：箱线图
        ax2 = axes[1, idx]
        bp = ax2.boxplot([high_ratios], widths=0.6, patch_artist=True,
                         labels=[version.upper()])
        bp['boxes'][0].set_facecolor('lightblue')
        ax2.set_ylabel('高熵Token比例', fontsize=12)
        ax2.set_title(f'{version.upper()} - 箱线图', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3, axis='y')
        
        # 添加统计信息文本
        stats_text = f"均值: {np.mean(high_ratios):.3f}\n中位数: {np.median(high_ratios):.3f}\n标准差: {np.std(high_ratios):.3f}"
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'token_ratio_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\n已保存: token_ratio_distribution.png")
    plt.close()
    
    # 图2: 问题平均熵 vs 高熵token比例（散点图）
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, version in enumerate(['v1', 'v2', 'v3']):
        ax = axes[idx]
        analyses = results[version]['question_analyses']
        
        avg_entropies = [a['avg_entropy'] for a in analyses]
        high_ratios = [a['high_token_ratio'] for a in analyses]
        
        # 散点图
        scatter = ax.scatter(avg_entropies, high_ratios, alpha=0.5, s=30)
        
        # 添加趋势线
        z = np.polyfit(avg_entropies, high_ratios, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(avg_entropies), max(avg_entropies), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f'趋势线: y={z[0]:.3f}x+{z[1]:.3f}')
        
        # 相关系数
        corr = patterns[version]['correlations']['entropy_vs_ratio']
        ax.text(0.05, 0.95, f'相关系数: {corr:.4f}', transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
               fontsize=11, fontweight='bold')
        
        ax.set_xlabel('问题平均熵', fontsize=12)
        ax.set_ylabel('高熵Token比例', fontsize=12)
        ax.set_title(f'{version.upper()} - 熵与Token比例关系', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'entropy_vs_token_ratio.png', dpi=300, bbox_inches='tight')
    print(f"已保存: entropy_vs_token_ratio.png")
    plt.close()
    
    # 图3: 问题长度 vs 高熵token比例
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, version in enumerate(['v1', 'v2', 'v3']):
        ax = axes[idx]
        analyses = results[version]['question_analyses']
        
        token_counts = [a['total_tokens'] for a in analyses]
        high_ratios = [a['high_token_ratio'] for a in analyses]
        
        # 散点图
        scatter = ax.scatter(token_counts, high_ratios, alpha=0.5, s=30, c=token_counts, cmap='viridis')
        
        # 相关系数
        corr = patterns[version]['correlations']['length_vs_ratio']
        ax.text(0.05, 0.95, f'相关系数: {corr:.4f}', transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
               fontsize=11, fontweight='bold')
        
        ax.set_xlabel('问题长度 (Token数)', fontsize=12)
        ax.set_ylabel('高熵Token比例', fontsize=12)
        ax.set_title(f'{version.upper()} - 长度与Token比例关系', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Token数', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'length_vs_token_ratio.png', dpi=300, bbox_inches='tight')
    print(f"已保存: length_vs_token_ratio.png")
    plt.close()
    
    # 图4: 跨版本对比
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 4.1: 平均高熵token比例对比
    ax1 = axes[0, 0]
    versions = ['V1', 'V2', 'V3']
    mean_ratios = [results[v]['statistics']['mean_high_token_ratio'] for v in ['v1', 'v2', 'v3']]
    bars = ax1.bar(versions, mean_ratios, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
    ax1.set_ylabel('平均高熵Token比例', fontsize=12)
    ax1.set_title('跨版本：平均高熵Token比例', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, ratio in zip(bars, mean_ratios):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{ratio:.3f}\n({ratio*100:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 4.2: 高熵token比例的标准差对比
    ax2 = axes[0, 1]
    std_ratios = [results[v]['statistics']['std_high_token_ratio'] for v in ['v1', 'v2', 'v3']]
    bars = ax2.bar(versions, std_ratios, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
    ax2.set_ylabel('标准差', fontsize=12)
    ax2.set_title('跨版本：高熵Token比例标准差', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    
    for bar, std in zip(bars, std_ratios):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{std:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 4.3: 相关系数对比
    ax3 = axes[1, 0]
    entropy_corrs = [patterns[v]['correlations']['entropy_vs_ratio'] for v in ['v1', 'v2', 'v3']]
    length_corrs = [patterns[v]['correlations']['length_vs_ratio'] for v in ['v1', 'v2', 'v3']]
    
    x = np.arange(len(versions))
    width = 0.35
    bars1 = ax3.bar(x - width/2, entropy_corrs, width, label='熵相关', alpha=0.8)
    bars2 = ax3.bar(x + width/2, length_corrs, width, label='长度相关', alpha=0.8)
    
    ax3.set_ylabel('相关系数', fontsize=12)
    ax3.set_title('跨版本：相关性对比', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(versions)
    ax3.legend()
    ax3.grid(alpha=0.3, axis='y')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 4.4: 箱线图总对比
    ax4 = axes[1, 1]
    all_ratios = [results[v]['all_high_ratios'] for v in ['v1', 'v2', 'v3']]
    bp = ax4.boxplot(all_ratios, labels=versions, patch_artist=True)
    
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax4.set_ylabel('高熵Token比例', fontsize=12)
    ax4.set_title('跨版本：高熵Token比例分布对比', fontsize=14, fontweight='bold')
    ax4.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cross_version_comparison.png', dpi=300, bbox_inches='tight')
    print(f"已保存: cross_version_comparison.png")
    plt.close()

def save_results(results, patterns, thresholds):
    """保存分析结果"""
    output_dir = Path('/data/user5/R-Zero/question_entropy_analysis_results')
    
    # 保存详细结果
    output_data = {
        'thresholds': thresholds,
        'results': {
            version: {
                'statistics': results[version]['statistics'],
                'ratio_distribution': results[version]['ratio_distribution'],
                'sample_questions': results[version]['question_analyses'][:100]  # 保存前100个示例
            }
            for version in ['v1', 'v2', 'v3']
        },
        'patterns': patterns
    }
    
    with open(output_dir / 'token_ratio_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n已保存详细数据: token_ratio_analysis.json")
    
    # 生成文本报告
    report_file = output_dir / 'token_ratio_analysis_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("高熵问题中Token比例分析报告\n")
        f.write("="*80 + "\n\n")
        
        f.write("一、阈值定义\n")
        f.write("-"*80 + "\n\n")
        for version in ['v1', 'v2', 'v3']:
            f.write(f"{version.upper()}:\n")
            f.write(f"  高熵Token阈值（80%分位数）: {thresholds[version]['token_entropy_threshold']:.4f}\n")
            f.write(f"  高熵问题阈值（80%分位数）: {thresholds[version]['question_entropy_threshold']:.4f}\n\n")
        
        f.write("\n二、统计摘要\n")
        f.write("-"*80 + "\n\n")
        for version in ['v1', 'v2', 'v3']:
            stats = results[version]['statistics']
            f.write(f"{version.upper()}:\n")
            f.write(f"  高熵问题数: {results[version]['high_entropy_question_count']}\n")
            f.write(f"  平均高熵Token比例: {stats['mean_high_token_ratio']:.4f} ({stats['mean_high_token_ratio']*100:.2f}%)\n")
            f.write(f"  中位数高熵Token比例: {stats['median_high_token_ratio']:.4f}\n")
            f.write(f"  标准差: {stats['std_high_token_ratio']:.4f}\n")
            f.write(f"  平均问题长度: {stats['mean_total_tokens']:.1f} tokens\n")
            f.write(f"  平均高熵Token数: {stats['mean_high_token_count']:.1f}\n")
            f.write(f"  平均低熵Token数: {stats['mean_low_token_count']:.1f}\n\n")
        
        f.write("\n三、发现的规律\n")
        f.write("-"*80 + "\n\n")
        for version in ['v1', 'v2', 'v3']:
            f.write(f"{version.upper()}:\n\n")
            
            f.write("  规律1: 问题平均熵与高熵Token比例的关系\n")
            for group_name, group_data in patterns[version]['entropy_groups'].items():
                f.write(f"    {group_name}: 平均熵={group_data['avg_question_entropy']:.4f}, ")
                f.write(f"高熵Token比例={group_data['avg_high_token_ratio']:.4f} ({group_data['avg_high_token_ratio']*100:.2f}%)\n")
            
            f.write("\n  规律2: 问题长度与高熵Token比例的关系\n")
            for group_name, group_data in patterns[version]['length_groups'].items():
                f.write(f"    {group_name}: 平均长度={group_data['avg_length']:.1f} tokens, ")
                f.write(f"高熵Token比例={group_data['avg_high_token_ratio']:.4f} ({group_data['avg_high_token_ratio']*100:.2f}%)\n")
            
            f.write("\n  规律3: 相关性\n")
            f.write(f"    问题平均熵 vs 高熵Token比例: r={patterns[version]['correlations']['entropy_vs_ratio']:.4f}\n")
            f.write(f"    问题长度 vs 高熵Token比例: r={patterns[version]['correlations']['length_vs_ratio']:.4f}\n\n")
        
        f.write("\n四、示例问题\n")
        f.write("-"*80 + "\n\n")
        for version in ['v1', 'v2', 'v3']:
            f.write(f"{version.upper()} - 高熵Token比例最高的5个问题:\n\n")
            
            # 按高熵token比例排序
            sorted_questions = sorted(results[version]['question_analyses'], 
                                    key=lambda x: x['high_token_ratio'], reverse=True)[:5]
            
            for idx, q in enumerate(sorted_questions, 1):
                f.write(f"  示例 {idx}:\n")
                f.write(f"    Sample ID: {q['sample_id']}, Step: {q['step']}\n")
                f.write(f"    问题平均熵: {q['avg_entropy']:.4f}\n")
                f.write(f"    总Token数: {q['total_tokens']}\n")
                f.write(f"    高熵Token数: {q['high_entropy_tokens']} ({q['high_token_ratio']*100:.1f}%)\n")
                f.write(f"    低熵Token数: {q['low_entropy_tokens']} ({q['low_token_ratio']*100:.1f}%)\n")
                f.write(f"    问题预览: {q['question_preview']}\n\n")
    
    print(f"已保存文本报告: token_ratio_analysis_report.txt")

def main():
    base_path = '/data/user5/R-Zero/question_entropy_logs'
    
    print("="*60)
    print("高熵问题中Token比例分析")
    print("="*60)
    
    # 1. 加载数据
    all_data = load_all_token_entropies(base_path)
    
    # 2. 计算阈值
    thresholds = calculate_thresholds(all_data)
    
    # 3. 分析token比例
    results = analyze_token_ratio_in_questions(all_data, thresholds)
    
    # 4. 寻找规律
    patterns = find_patterns(results)
    
    # 5. 创建可视化
    print(f"\n{'='*60}")
    print("生成可视化图表...")
    print(f"{'='*60}")
    create_visualizations(results, patterns, thresholds)
    
    # 6. 保存结果
    print(f"\n{'='*60}")
    print("保存分析结果...")
    print(f"{'='*60}")
    save_results(results, patterns, thresholds)
    
    print(f"\n{'='*60}")
    print("分析完成！")
    print(f"{'='*60}")
    print(f"\n所有结果已保存到: /data/user5/R-Zero/question_entropy_analysis_results/")

if __name__ == '__main__':
    main()


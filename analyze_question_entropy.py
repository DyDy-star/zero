#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析问题熵数据，识别和分析高熵问题
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from collections import defaultdict

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_all_data(base_path):
    """加载所有版本的数据"""
    all_data = {'v1': [], 'v2': [], 'v3': []}
    
    for version in ['v1', 'v2', 'v3']:
        for step in range(1, 6):
            file_path = Path(base_path) / f"question_entropy_{version}_step{step}.json"
            print(f"正在加载: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 计算每个问题的平均熵
            for question in data['questions']:
                if question['token_entropies']:
                    avg_entropy = np.mean(question['token_entropies'])
                    all_data[version].append({
                        'sample_id': question['sample_id'],
                        'step': step,
                        'question_text': question['question_text'],
                        'avg_entropy': avg_entropy,
                        'token_entropies': question['token_entropies']
                    })
    
    return all_data

def is_format_error(question_text):
    """判断是否为格式错误或乱码"""
    # 检查常见的格式错误特征
    
    # 1. 过多的特殊字符或符号（超过30%）
    special_chars = len(re.findall(r'[^a-zA-Z0-9\s\u4e00-\u9fff\(\)\[\]\{\}\.,;:\-\+\=\\\/_]', question_text))
    if len(question_text) > 0 and special_chars / len(question_text) > 0.3:
        return True, "过多特殊字符"
    
    # 2. 包含大量HTML标签
    if question_text.count('<') > 5 and question_text.count('>') > 5:
        return True, "包含HTML标签"
    
    # 3. 重复字符过多
    if re.search(r'(.)\1{10,}', question_text):
        return True, "重复字符过多"
    
    # 4. 包含明显的错误格式标记
    error_patterns = [
        r'\\u[0-9a-fA-F]{4}',  # Unicode转义
        r'&[a-z]+;',  # HTML实体
        r'\[object\s+Object\]',  # JavaScript对象
        r'undefined',
        r'NaN',
        r'null{5,}',
    ]
    for pattern in error_patterns:
        if re.search(pattern, question_text):
            return True, f"包含错误格式: {pattern}"
    
    # 5. 文本过短（少于20字符）或过长（超过5000字符）
    if len(question_text) < 20:
        return True, "文本过短"
    if len(question_text) > 5000:
        return True, "文本过长"
    
    # 6. 包含大量换行或空白
    if question_text.count('\n') > 50:
        return True, "过多换行符"
    
    # 7. 检查是否包含明显的元数据或指令文本
    meta_patterns = [
        r'Question:.*Base Concept',
        r'Formatting and Output',
        r'Mechanics:',
        r'Expression of Final Answer',
        r'This process allows',
        r'Crafting a challenging problem',
    ]
    for pattern in meta_patterns:
        if re.search(pattern, question_text, re.IGNORECASE):
            return True, "包含元指令文本"
    
    return False, None

def analyze_high_entropy_reasons(question_text, token_entropies):
    """分析高熵问题的原因"""
    reasons = []
    
    # 1. 计算熵的变异系数
    entropy_std = np.std(token_entropies)
    entropy_mean = np.mean(token_entropies)
    if entropy_mean > 0:
        cv = entropy_std / entropy_mean
        if cv > 0.5:
            reasons.append(f"熵值波动大(变异系数:{cv:.2f})")
    
    # 2. 高熵token比例
    high_entropy_tokens = sum(1 for e in token_entropies if e > 3.0)
    high_entropy_ratio = high_entropy_tokens / len(token_entropies) if len(token_entropies) > 0 else 0
    if high_entropy_ratio > 0.2:
        reasons.append(f"高熵token比例高({high_entropy_ratio*100:.1f}%)")
    
    # 3. 问题复杂度
    text_length = len(question_text)
    if text_length > 1000:
        reasons.append(f"文本长度大({text_length}字符)")
    
    # 4. 数学符号密度
    math_symbols = len(re.findall(r'\\[a-zA-Z]+|[\$\^\{\}\_]', question_text))
    if math_symbols > 20:
        reasons.append(f"数学符号密集({math_symbols}个)")
    
    # 5. 问题类型识别
    if re.search(r'combination|permutation|probability', question_text, re.IGNORECASE):
        reasons.append("组合/概率问题")
    if re.search(r'geometry|triangle|circle|polygon', question_text, re.IGNORECASE):
        reasons.append("几何问题")
    if re.search(r'sequence|series|recursive', question_text, re.IGNORECASE):
        reasons.append("序列/递归问题")
    if re.search(r'optimization|maximize|minimize', question_text, re.IGNORECASE):
        reasons.append("优化问题")
    if re.search(r'proof|prove|demonstrate', question_text, re.IGNORECASE):
        reasons.append("证明问题")
    
    # 6. 多步骤问题
    if question_text.count('?') > 1 or re.search(r'step\s+\d+|first.*then.*finally', question_text, re.IGNORECASE):
        reasons.append("多步骤问题")
    
    # 7. 抽象概念
    abstract_keywords = ['concept', 'theory', 'general', 'arbitrary', 'any', 'all']
    abstract_count = sum(1 for keyword in abstract_keywords if keyword in question_text.lower())
    if abstract_count >= 3:
        reasons.append(f"抽象概念多({abstract_count}个)")
    
    return reasons

def analyze_versions(all_data):
    """分析每个版本的问题熵分布"""
    results = {}
    
    for version in ['v1', 'v2', 'v3']:
        print(f"\n{'='*60}")
        print(f"分析版本: {version}")
        print(f"{'='*60}")
        
        questions = all_data[version]
        entropies = [q['avg_entropy'] for q in questions]
        
        # 计算统计信息
        percentile_80 = np.percentile(entropies, 80)
        mean_entropy = np.mean(entropies)
        median_entropy = np.median(entropies)
        std_entropy = np.std(entropies)
        
        print(f"问题总数: {len(questions)}")
        print(f"平均熵: {mean_entropy:.4f}")
        print(f"中位数熵: {median_entropy:.4f}")
        print(f"标准差: {std_entropy:.4f}")
        print(f"80%分位数: {percentile_80:.4f}")
        
        # 识别高熵问题
        high_entropy_questions = [q for q in questions if q['avg_entropy'] >= percentile_80]
        low_entropy_questions = [q for q in questions if q['avg_entropy'] < percentile_80]
        
        print(f"高熵问题数: {len(high_entropy_questions)} ({len(high_entropy_questions)/len(questions)*100:.1f}%)")
        print(f"低熵问题数: {len(low_entropy_questions)} ({len(low_entropy_questions)/len(questions)*100:.1f}%)")
        
        # 过滤格式错误
        valid_high_entropy = []
        format_errors = []
        
        for q in high_entropy_questions:
            is_error, error_reason = is_format_error(q['question_text'])
            if is_error:
                format_errors.append({
                    'question': q,
                    'reason': error_reason
                })
            else:
                valid_high_entropy.append(q)
        
        print(f"\n格式错误的高熵问题: {len(format_errors)} ({len(format_errors)/len(high_entropy_questions)*100:.1f}%)")
        print(f"有效的高熵问题: {len(valid_high_entropy)} ({len(valid_high_entropy)/len(high_entropy_questions)*100:.1f}%)")
        
        # 分析有效高熵问题的原因
        print(f"\n分析有效高熵问题的原因...")
        reason_counter = defaultdict(int)
        analyzed_questions = []
        
        for q in valid_high_entropy:
            reasons = analyze_high_entropy_reasons(q['question_text'], q['token_entropies'])
            for reason in reasons:
                reason_counter[reason] += 1
            
            analyzed_questions.append({
                'sample_id': q['sample_id'],
                'step': q['step'],
                'avg_entropy': q['avg_entropy'],
                'question_text': q['question_text'][:500],  # 只保存前500字符
                'full_question': q['question_text'],
                'reasons': reasons,
                'max_entropy': max(q['token_entropies']),
                'min_entropy': min(q['token_entropies']),
                'entropy_std': np.std(q['token_entropies'])
            })
        
        # 按熵值排序
        analyzed_questions.sort(key=lambda x: x['avg_entropy'], reverse=True)
        
        print(f"\n高熵原因统计 (Top 10):")
        for reason, count in sorted(reason_counter.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {reason}: {count} ({count/len(valid_high_entropy)*100:.1f}%)")
        
        results[version] = {
            'total_questions': len(questions),
            'mean_entropy': mean_entropy,
            'median_entropy': median_entropy,
            'std_entropy': std_entropy,
            'percentile_80': percentile_80,
            'high_entropy_count': len(high_entropy_questions),
            'format_error_count': len(format_errors),
            'valid_high_entropy_count': len(valid_high_entropy),
            'reason_stats': dict(reason_counter),
            'format_errors': format_errors[:10],  # 只保存前10个格式错误示例
            'analyzed_questions': analyzed_questions,
            'all_entropies': entropies
        }
    
    return results

def create_visualizations(results):
    """创建可视化图表"""
    output_dir = Path('/data/user5/R-Zero/question_entropy_analysis_results')
    output_dir.mkdir(exist_ok=True)
    
    # 图1: 各版本熵分布对比
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, version in enumerate(['v1', 'v2', 'v3']):
        ax = axes[idx]
        entropies = results[version]['all_entropies']
        percentile_80 = results[version]['percentile_80']
        
        ax.hist(entropies, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(percentile_80, color='red', linestyle='--', linewidth=2, 
                   label=f'80%分位数: {percentile_80:.2f}')
        ax.set_xlabel('平均熵值', fontsize=12)
        ax.set_ylabel('问题数量', fontsize=12)
        ax.set_title(f'{version.upper()} 问题熵分布', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'question_entropy_distribution.png', dpi=300, bbox_inches='tight')
    print(f"已保存: question_entropy_distribution.png")
    plt.close()
    
    # 图2: 高熵问题比例对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    versions = ['v1', 'v2', 'v3']
    high_entropy_counts = [results[v]['high_entropy_count'] for v in versions]
    format_error_counts = [results[v]['format_error_count'] for v in versions]
    valid_high_entropy_counts = [results[v]['valid_high_entropy_count'] for v in versions]
    
    x = np.arange(len(versions))
    width = 0.25
    
    ax1.bar(x - width, high_entropy_counts, width, label='总高熵问题', alpha=0.8)
    ax1.bar(x, format_error_counts, width, label='格式错误', alpha=0.8)
    ax1.bar(x + width, valid_high_entropy_counts, width, label='有效高熵问题', alpha=0.8)
    
    ax1.set_xlabel('版本', fontsize=12)
    ax1.set_ylabel('问题数量', fontsize=12)
    ax1.set_title('高熵问题数量对比', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([v.upper() for v in versions])
    ax1.legend()
    ax1.grid(alpha=0.3, axis='y')
    
    # 比例图
    format_error_ratios = [results[v]['format_error_count'] / results[v]['high_entropy_count'] * 100 
                          for v in versions]
    valid_ratios = [results[v]['valid_high_entropy_count'] / results[v]['high_entropy_count'] * 100 
                   for v in versions]
    
    ax2.bar(x, format_error_ratios, width*2, label='格式错误比例', alpha=0.8, color='orange')
    ax2.bar(x, valid_ratios, width*2, bottom=format_error_ratios, label='有效问题比例', alpha=0.8, color='green')
    
    ax2.set_xlabel('版本', fontsize=12)
    ax2.set_ylabel('比例 (%)', fontsize=12)
    ax2.set_title('高熵问题组成比例', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([v.upper() for v in versions])
    ax2.legend()
    ax2.grid(alpha=0.3, axis='y')
    ax2.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'high_entropy_comparison.png', dpi=300, bbox_inches='tight')
    print(f"已保存: high_entropy_comparison.png")
    plt.close()
    
    # 图3: 高熵原因分析
    fig, axes = plt.subplots(3, 1, figsize=(14, 15))
    
    for idx, version in enumerate(['v1', 'v2', 'v3']):
        ax = axes[idx]
        reason_stats = results[version]['reason_stats']
        
        if reason_stats:
            # 取前10个原因
            sorted_reasons = sorted(reason_stats.items(), key=lambda x: x[1], reverse=True)[:10]
            reasons, counts = zip(*sorted_reasons)
            
            y_pos = np.arange(len(reasons))
            ax.barh(y_pos, counts, alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(reasons, fontsize=10)
            ax.set_xlabel('问题数量', fontsize=12)
            ax.set_title(f'{version.upper()} 高熵原因分析 (Top 10)', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3, axis='x')
            
            # 添加数值标签
            for i, count in enumerate(counts):
                ax.text(count, i, f' {count}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'high_entropy_reasons.png', dpi=300, bbox_inches='tight')
    print(f"已保存: high_entropy_reasons.png")
    plt.close()

def save_detailed_results(results):
    """保存详细分析结果"""
    output_dir = Path('/data/user5/R-Zero/question_entropy_analysis_results')
    output_dir.mkdir(exist_ok=True)
    
    for version in ['v1', 'v2', 'v3']:
        # 保存有效高熵问题
        output_file = output_dir / f'high_entropy_questions_{version}.json'
        
        # 准备输出数据
        output_data = {
            'metadata': {
                'version': version,
                'total_questions': results[version]['total_questions'],
                'mean_entropy': float(results[version]['mean_entropy']),
                'median_entropy': float(results[version]['median_entropy']),
                'std_entropy': float(results[version]['std_entropy']),
                'percentile_80': float(results[version]['percentile_80']),
                'high_entropy_count': results[version]['high_entropy_count'],
                'format_error_count': results[version]['format_error_count'],
                'valid_high_entropy_count': results[version]['valid_high_entropy_count']
            },
            'reason_statistics': results[version]['reason_stats'],
            'format_error_examples': [
                {
                    'sample_id': e['question']['sample_id'],
                    'step': e['question']['step'],
                    'avg_entropy': float(e['question']['avg_entropy']),
                    'error_reason': e['reason'],
                    'question_preview': e['question']['question_text'][:200]
                }
                for e in results[version]['format_errors']
            ],
            'valid_high_entropy_questions': [
                {
                    'sample_id': q['sample_id'],
                    'step': q['step'],
                    'avg_entropy': float(q['avg_entropy']),
                    'max_entropy': float(q['max_entropy']),
                    'min_entropy': float(q['min_entropy']),
                    'entropy_std': float(q['entropy_std']),
                    'reasons': q['reasons'],
                    'question_text': q['full_question']
                }
                for q in results[version]['analyzed_questions']
            ]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"已保存 {version} 详细结果到: {output_file}")

def generate_summary_report(results):
    """生成综合分析报告"""
    output_dir = Path('/data/user5/R-Zero/question_entropy_analysis_results')
    report_file = output_dir / 'analysis_summary_report.txt'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("问题熵分析综合报告\n")
        f.write("="*80 + "\n\n")
        
        for version in ['v1', 'v2', 'v3']:
            f.write(f"\n{'='*80}\n")
            f.write(f"版本: {version.upper()}\n")
            f.write(f"{'='*80}\n\n")
            
            r = results[version]
            
            f.write(f"1. 基本统计\n")
            f.write(f"   - 问题总数: {r['total_questions']}\n")
            f.write(f"   - 平均熵: {r['mean_entropy']:.4f}\n")
            f.write(f"   - 中位数熵: {r['median_entropy']:.4f}\n")
            f.write(f"   - 标准差: {r['std_entropy']:.4f}\n")
            f.write(f"   - 80%分位数阈值: {r['percentile_80']:.4f}\n\n")
            
            f.write(f"2. 高熵问题分布\n")
            f.write(f"   - 高熵问题总数: {r['high_entropy_count']} ({r['high_entropy_count']/r['total_questions']*100:.1f}%)\n")
            f.write(f"   - 格式错误数: {r['format_error_count']} ({r['format_error_count']/r['high_entropy_count']*100:.1f}%)\n")
            f.write(f"   - 有效高熵问题: {r['valid_high_entropy_count']} ({r['valid_high_entropy_count']/r['high_entropy_count']*100:.1f}%)\n\n")
            
            f.write(f"3. 高熵原因统计 (Top 10)\n")
            sorted_reasons = sorted(r['reason_stats'].items(), key=lambda x: x[1], reverse=True)[:10]
            for idx, (reason, count) in enumerate(sorted_reasons, 1):
                percentage = count / r['valid_high_entropy_count'] * 100 if r['valid_high_entropy_count'] > 0 else 0
                f.write(f"   {idx}. {reason}: {count} ({percentage:.1f}%)\n")
            
            f.write(f"\n4. 示例高熵问题 (Top 5)\n")
            for idx, q in enumerate(r['analyzed_questions'][:5], 1):
                f.write(f"\n   示例 {idx}:\n")
                f.write(f"   - Sample ID: {q['sample_id']}, Step: {q['step']}\n")
                f.write(f"   - 平均熵: {q['avg_entropy']:.4f}\n")
                f.write(f"   - 熵范围: [{q['min_entropy']:.2f}, {q['max_entropy']:.2f}]\n")
                f.write(f"   - 原因: {', '.join(q['reasons'])}\n")
                f.write(f"   - 问题内容: {q['question_text'][:300]}...\n")
        
        f.write(f"\n{'='*80}\n")
        f.write(f"跨版本对比分析\n")
        f.write(f"{'='*80}\n\n")
        
        f.write(f"1. 平均熵变化趋势\n")
        for version in ['v1', 'v2', 'v3']:
            f.write(f"   {version.upper()}: {results[version]['mean_entropy']:.4f}\n")
        
        f.write(f"\n2. 有效高熵问题比例变化\n")
        for version in ['v1', 'v2', 'v3']:
            ratio = results[version]['valid_high_entropy_count'] / results[version]['total_questions'] * 100
            f.write(f"   {version.upper()}: {ratio:.2f}%\n")
        
        f.write(f"\n3. 格式错误率变化\n")
        for version in ['v1', 'v2', 'v3']:
            if results[version]['high_entropy_count'] > 0:
                ratio = results[version]['format_error_count'] / results[version]['high_entropy_count'] * 100
                f.write(f"   {version.upper()}: {ratio:.2f}%\n")
    
    print(f"\n已生成综合分析报告: {report_file}")

def main():
    base_path = '/data/user5/R-Zero/question_entropy_logs'
    
    print("开始加载数据...")
    all_data = load_all_data(base_path)
    
    print("\n开始分析...")
    results = analyze_versions(all_data)
    
    print("\n创建可视化图表...")
    create_visualizations(results)
    
    print("\n保存详细结果...")
    save_detailed_results(results)
    
    print("\n生成综合报告...")
    generate_summary_report(results)
    
    print("\n分析完成!")
    print(f"结果保存在: /data/user5/R-Zero/question_entropy_analysis_results/")

if __name__ == '__main__':
    main()


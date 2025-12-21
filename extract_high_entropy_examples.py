#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取高熵问题的具体示例用于分析
"""

import json
import numpy as np
from pathlib import Path

def is_format_error(question_text):
    """判断是否为格式错误或乱码"""
    import re
    
    # 1. 过多的特殊字符或符号（超过30%）
    special_chars = len(re.findall(r'[^a-zA-Z0-9\s\u4e00-\u9fff\(\)\[\]\{\}\.,;:\-\+\=\\\/_]', question_text))
    if len(question_text) > 0 and special_chars / len(question_text) > 0.3:
        return True
    
    # 2. 包含大量HTML标签
    if question_text.count('<') > 5 and question_text.count('>') > 5:
        return True
    
    # 3. 重复字符过多
    if re.search(r'(.)\1{10,}', question_text):
        return True
    
    # 4. 文本过短（少于20字符）
    if len(question_text) < 20:
        return True
    
    # 5. 包含明显的元数据或指令文本
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
            return True
    
    return False

def load_and_extract_examples():
    """加载数据并提取示例"""
    base_path = Path('/data/user5/R-Zero/question_entropy_logs')
    output_path = Path('/data/user5/R-Zero/question_entropy_analysis_results')
    
    all_examples = {'v1': [], 'v2': [], 'v3': []}
    
    for version in ['v1', 'v2', 'v3']:
        print(f"\n处理版本 {version}...")
        
        questions = []
        for step in range(1, 6):
            file_path = base_path / f"question_entropy_{version}_step{step}.json"
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for q in data['questions']:
                if q['token_entropies']:
                    questions.append({
                        'sample_id': q['sample_id'],
                        'step': step,
                        'question_text': q['question_text'],
                        'avg_entropy': np.mean(q['token_entropies']),
                        'max_entropy': max(q['token_entropies']),
                        'min_entropy': min(q['token_entropies']),
                        'entropy_std': np.std(q['token_entropies']),
                        'token_count': len(q['token_entropies'])
                    })
        
        # 计算80%分位数
        entropies = [q['avg_entropy'] for q in questions]
        threshold = np.percentile(entropies, 80)
        
        print(f"80%分位数阈值: {threshold:.4f}")
        
        # 筛选高熵且非格式错误的问题
        high_entropy_valid = []
        for q in questions:
            if q['avg_entropy'] >= threshold and not is_format_error(q['question_text']):
                high_entropy_valid.append(q)
        
        # 按不同特征分类提取示例
        print(f"有效高熵问题数: {len(high_entropy_valid)}")
        
        # 1. 证明问题示例（包含proof/prove关键词）
        import re
        proof_questions = [q for q in high_entropy_valid 
                          if re.search(r'proof|prove|demonstrate|show that', q['question_text'], re.IGNORECASE)]
        
        # 2. 几何问题示例
        geometry_questions = [q for q in high_entropy_valid 
                             if re.search(r'triangle|circle|polygon|angle|geometry|coordinate', q['question_text'], re.IGNORECASE)]
        
        # 3. 序列递归问题示例
        sequence_questions = [q for q in high_entropy_valid 
                             if re.search(r'sequence|recursive|series|term', q['question_text'], re.IGNORECASE)]
        
        # 4. 组合概率问题示例
        combo_questions = [q for q in high_entropy_valid 
                          if re.search(r'combination|permutation|probability|choose', q['question_text'], re.IGNORECASE)]
        
        # 5. 最高熵的问题（排除格式错误）
        top_entropy = sorted(high_entropy_valid, key=lambda x: x['avg_entropy'], reverse=True)[:10]
        
        print(f"  证明问题: {len(proof_questions)}")
        print(f"  几何问题: {len(geometry_questions)}")
        print(f"  序列问题: {len(sequence_questions)}")
        print(f"  组合问题: {len(combo_questions)}")
        
        all_examples[version] = {
            'threshold': float(threshold),
            'total_high_entropy': len(high_entropy_valid),
            'proof_examples': proof_questions[:5] if proof_questions else [],
            'geometry_examples': geometry_questions[:5] if geometry_questions else [],
            'sequence_examples': sequence_questions[:5] if sequence_questions else [],
            'combo_examples': combo_questions[:5] if combo_questions else [],
            'top_entropy_examples': top_entropy[:10]
        }
    
    # 保存到文件
    output_file = output_path / 'high_entropy_question_examples.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_examples, f, ensure_ascii=False, indent=2)
    
    print(f"\n示例已保存到: {output_file}")
    
    # 生成可读的文本报告
    report_file = output_path / 'high_entropy_examples_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("高熵问题示例分析报告\n")
        f.write("="*100 + "\n\n")
        
        for version in ['v1', 'v2', 'v3']:
            f.write(f"\n{'='*100}\n")
            f.write(f"版本: {version.upper()}\n")
            f.write(f"{'='*100}\n\n")
            f.write(f"80%分位数阈值: {all_examples[version]['threshold']:.4f}\n")
            f.write(f"有效高熵问题总数: {all_examples[version]['total_high_entropy']}\n\n")
            
            # 证明问题示例
            if all_examples[version]['proof_examples']:
                f.write(f"\n{'-'*100}\n")
                f.write(f"【证明问题示例】\n")
                f.write(f"{'-'*100}\n\n")
                for idx, q in enumerate(all_examples[version]['proof_examples'], 1):
                    f.write(f"示例 {idx}:\n")
                    f.write(f"Sample ID: {q['sample_id']}, Step: {q['step']}\n")
                    f.write(f"平均熵: {q['avg_entropy']:.4f}, 熵范围: [{q['min_entropy']:.2f}, {q['max_entropy']:.2f}], 标准差: {q['entropy_std']:.4f}\n")
                    f.write(f"Token数: {q['token_count']}\n")
                    f.write(f"问题内容:\n{q['question_text'][:800]}\n")
                    if len(q['question_text']) > 800:
                        f.write(f"... (内容过长，已截断)\n")
                    f.write(f"\n")
            
            # 几何问题示例
            if all_examples[version]['geometry_examples']:
                f.write(f"\n{'-'*100}\n")
                f.write(f"【几何问题示例】\n")
                f.write(f"{'-'*100}\n\n")
                for idx, q in enumerate(all_examples[version]['geometry_examples'], 1):
                    f.write(f"示例 {idx}:\n")
                    f.write(f"Sample ID: {q['sample_id']}, Step: {q['step']}\n")
                    f.write(f"平均熵: {q['avg_entropy']:.4f}, 熵范围: [{q['min_entropy']:.2f}, {q['max_entropy']:.2f}]\n")
                    f.write(f"问题内容:\n{q['question_text'][:800]}\n")
                    if len(q['question_text']) > 800:
                        f.write(f"... (内容过长，已截断)\n")
                    f.write(f"\n")
            
            # 序列问题示例
            if all_examples[version]['sequence_examples']:
                f.write(f"\n{'-'*100}\n")
                f.write(f"【序列/递归问题示例】\n")
                f.write(f"{'-'*100}\n\n")
                for idx, q in enumerate(all_examples[version]['sequence_examples'], 1):
                    f.write(f"示例 {idx}:\n")
                    f.write(f"Sample ID: {q['sample_id']}, Step: {q['step']}\n")
                    f.write(f"平均熵: {q['avg_entropy']:.4f}\n")
                    f.write(f"问题内容:\n{q['question_text'][:800]}\n")
                    if len(q['question_text']) > 800:
                        f.write(f"... (内容过长，已截断)\n")
                    f.write(f"\n")
            
            # 最高熵问题
            f.write(f"\n{'-'*100}\n")
            f.write(f"【最高熵问题（Top 5）】\n")
            f.write(f"{'-'*100}\n\n")
            for idx, q in enumerate(all_examples[version]['top_entropy_examples'][:5], 1):
                f.write(f"Top {idx}:\n")
                f.write(f"Sample ID: {q['sample_id']}, Step: {q['step']}\n")
                f.write(f"平均熵: {q['avg_entropy']:.4f}, 最大熵: {q['max_entropy']:.2f}\n")
                f.write(f"Token数: {q['token_count']}\n")
                f.write(f"问题内容:\n{q['question_text'][:1000]}\n")
                if len(q['question_text']) > 1000:
                    f.write(f"... (内容过长，已截断)\n")
                f.write(f"\n{'-'*80}\n\n")
    
    print(f"文本报告已保存到: {report_file}")

if __name__ == '__main__':
    load_and_extract_examples()


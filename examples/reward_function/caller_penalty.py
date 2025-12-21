# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import regex as re
from typing import Dict, List
import json
from mathruler.grader import extract_boxed_content, grade_answer
import os
import time
import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import glob
STORAGE_PATH = os.getenv("STORAGE_PATH", "/data/user5/R-Zero")

# 获取当前模型版本（由训练脚本自动设置，从 save_path 中提取）
# 例如：questioner_v1 -> v1, unified_v2 -> v2
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
print(f"{'='*60}")
print(f"当前训练版本: {MODEL_VERSION} (由训练脚本自动设置)")
print(f"{'='*60}")
def _bleu_distance_matrix(sentences):
    n = len(sentences)
    dist = np.zeros((n, n))
    smoother = SmoothingFunction().method1
    for i in range(n):
        for j in range(i, n):
            if i == j:
                score = 1.0
            else:
                ref = [sentences[j].split()]
                hyp = sentences[i].split()
                score = sentence_bleu(ref, hyp, smoothing_function=smoother)
            dist[i, j] = dist[j, i] = 1 - score
    return dist

def cluster_share_per_problem(
        problems,
        distance_threshold: float = 0.5,
        linkage: str = "average"):
    if not problems:
        return []
    print('start clustering')
    start_time = time.time()
    dist_mat = _bleu_distance_matrix(problems)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage=linkage
    )
    labels = clustering.fit_predict(dist_mat)
    print(f'end clustering, time: {time.time() - start_time}')
    total = len(problems)
    cluster_size = Counter(labels)
    cluster_ratio = {lab: sz / total for lab, sz in cluster_size.items()}

    # 打印聚类统计信息
    print(f'=' * 60)
    print(f'聚类统计信息:')
    print(f'  B (总问题数): {total}')
    print(f'  聚类数量: {len(cluster_size)}')
    print(f'  各聚类大小: {dict(cluster_size)}')
    print(f'  聚类比例 |C_k|/B:')
    for lab, ratio in sorted(cluster_ratio.items(), key=lambda x: x[1], reverse=True):
        print(f'    聚类 {lab}: {cluster_size[lab]}/{total} = {ratio:.4f}')
    print(f'=' * 60)

    # 保存聚类统计信息到 JSON
    clustering_stats = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "B_total_questions": total,
        "num_clusters": len(cluster_size),
        "cluster_sizes": {int(lab): int(sz) for lab, sz in cluster_size.items()}
    }
    
    # 生成文件名（带时间戳）
    clustering_log_dir = f"{STORAGE_PATH}/clustering_logs"
    os.makedirs(clustering_log_dir, exist_ok=True)
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    log_filename = f"{clustering_log_dir}/clustering_stats_{timestamp_str}.json"
    
    with open(log_filename, 'w', encoding='utf-8') as f:
        json.dump(clustering_stats, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 聚类统计已保存到: {log_filename}")

    proportions = [cluster_ratio[lab] for lab in labels]
    return proportions

def generate_temp_filename(prefix="temp", suffix=".json"):
    timestamp = int(time.time() * 1000) 
    rand_part = random.randint(0, 99999)
    return f"{STORAGE_PATH}/temp_results/{prefix}_{timestamp}_{rand_part}{suffix}"
def split_list(lst, n=4):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

os.environ["NO_PROXY"] = "0.0.0.0,127.0.0.1"

def fetch(index,i):
    # 使用端口 5000 和 5001（两个 vLLM 服务）
    response = requests.get(f"http://0.0.0.0:{5000+index}/hello?name={i}")
    print(response)
    return True

def generate_results(data):
    # 使用 2 个 vLLM 服务并行处理
    num_workers = 2
    datas = split_list(data, num_workers)
    random_names = [generate_temp_filename(prefix=f"temp_{i}", suffix=".json") for i in range(num_workers)]
    for i in range(num_workers):
        with open(random_names[i],'w') as f:
            json.dump(datas[i],f,indent=4)

    final_results = []
    # 并行执行，充分利用 2 个 vLLM 服务
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(fetch, i, random_names[i]) for i in range(num_workers)]
        for future in as_completed(futures):
            print(future.result())

    for i in range(num_workers):
        with open(random_names[i].replace('.json','_results.json'),'r') as f:
            final_results.extend(json.load(f))
    for i in range(num_workers):
        os.remove(random_names[i].replace('.json','_results.json'))
    return final_results

def format_reward(predict: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, predict)
    return 1.0 if format_match else 0.0


def accuracy_reward(predict: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(predicts: List[str], ground_truths: List[str], format_weight: float = 0.1, file_path: str = "", entropy_info: List[Dict] = None) -> List[Dict[str, float]]:
    print(f'\n{"="*60}')
    print(f'Questioner 训练 - 奖励函数调用')
    print(f'输入样本数: {len(predicts)}')
    print(f'{"="*60}\n')
    
    # Helper function to extract question token entropies
    def extract_question_entropy(response_text: str, token_entropies: List[float], response_ids: List[int], tokenizer) -> Dict:
        """
        Extract entropy for the stripped question tokens (内部内容，不含标签).
        
        关键：question 是 <question>...</question> 标签内部的内容，且经过 strip() 处理
        注意：如果有多个 <question> 标签，取最后一个（与原始代码一致）
        """
        # Find ALL questions with tags (same as original code line 160)
        questions_matches = re.findall(r"<question>(.*?)</question>", response_text, re.DOTALL)
        if not questions_matches:
            return {'mean_entropy': 0.0, 'token_count': 0, 'token_entropies': []}
        
        # Extract the LAST question's inner content (same as line 164: questions[-1].strip())
        question_inner = questions_matches[-1].strip()  # ← 取最后一个，与 questions[-1].strip() 完全一致
        
        if not question_inner:
            return {'mean_entropy': 0.0, 'token_count': 0, 'token_entropies': []}
        
        # Strategy: tokenize the question_inner directly and find it in response_ids
        question_tokens = tokenizer.encode(question_inner, add_special_tokens=False)
        question_token_count = len(question_tokens)
        
        if question_token_count == 0:
            return {'mean_entropy': 0.0, 'token_count': 0, 'token_entropies': []}
        
        # Find the question tokens in response_ids
        # Search for the subsequence (find the LAST occurrence to match questions[-1])
        found_start_idx = -1
        for start_idx in range(len(response_ids) - question_token_count + 1):
            if response_ids[start_idx:start_idx + question_token_count] == question_tokens:
                found_start_idx = start_idx  # Keep updating to get the LAST match
        
        if found_start_idx == -1:
            # Fallback: find the LAST <question> tag position using re.finditer
            all_matches = list(re.finditer(r"<question>(.*?)</question>", response_text, re.DOTALL))
            if all_matches:
                last_match = all_matches[-1]  # Get the last match
                
                # Tokenize text before the LAST question tag
                pre_question_text = response_text[:last_match.start()]
                pre_tokens = tokenizer.encode(pre_question_text, add_special_tokens=False)
                
                # Estimate: skip opening tag tokens
                opening_tag_tokens = tokenizer.encode("<question>", add_special_tokens=False)
                found_start_idx = len(pre_tokens) + len(opening_tag_tokens)
                
                # Also need to skip any leading whitespace tokens
                inner_start_text = last_match.group(1)  # Before strip
                if inner_start_text != question_inner:
                    # There was whitespace, try to account for it
                    leading_ws = len(inner_start_text) - len(inner_start_text.lstrip())
                    if leading_ws > 0:
                        ws_text = inner_start_text[:leading_ws]
                        ws_tokens = tokenizer.encode(ws_text, add_special_tokens=False)
                        found_start_idx += len(ws_tokens)
        
        end_idx = found_start_idx + question_token_count
        
        # Extract entropies for question tokens
        if found_start_idx >= 0 and end_idx <= len(token_entropies):
            question_token_entropies = token_entropies[found_start_idx:end_idx]
            if question_token_entropies:
                return {
                    'mean_entropy': np.mean(question_token_entropies),
                    'token_count': len(question_token_entropies),
                    'token_entropies': question_token_entropies
                }
        
        return {'mean_entropy': 0.0, 'token_count': 0, 'token_entropies': []}
    
    results = []
    question_entropy_stats = []  # Store entropy stats for each sample
    
    with open('test.json','w') as f:
        json.dump(predicts,f,indent=4)
    
    for i in range(len(predicts)):
        questions = re.findall(r"<question>(.*?)</question>", predicts[i], re.DOTALL)
        answers = extract_boxed_content(predicts[i])
        
        # Extract question entropy if available
        entropy_stat = None
        if entropy_info and i < len(entropy_info):
            entropy_stat = extract_question_entropy(
                entropy_info[i]['response_text'],
                entropy_info[i]['token_entropies'],
                entropy_info[i]['response_ids'],
                entropy_info[i]['tokenizer']
            )
            question_entropy_stats.append(entropy_stat)
        
        if questions and answers:
            try:
                question = questions[-1].strip()  # 这里的处理与上面 extract_question_entropy 中的 question_inner 一致
                answer = answers[-1].strip()
                results.append({"question": question, "answer": answer})
            except:
                results.append({"question": "", "answer": ""})
        else:
            results.append({"question": "", "answer": ""})

    final_results = generate_results(results)
    penalty = cluster_share_per_problem([result['question'] for result in final_results], distance_threshold=0.5)
    # print(penalty)
    assert len(penalty) == len(final_results)
    
    # Print entropy statistics for stripped question text
    if question_entropy_stats:
        valid_entropies = [stat['mean_entropy'] for stat in question_entropy_stats if stat and stat['mean_entropy'] > 0]
        if valid_entropies:
            print(f'\n{"="*60}')
            print(f'Question 熵统计 (标签内部的文本，已strip):')
            print(f'  有效样本数: {len(valid_entropies)}')
            print(f'  平均熵: {np.mean(valid_entropies):.4f}')
            print(f'  熵标准差: {np.std(valid_entropies):.4f}')
            print(f'  最小熵: {np.min(valid_entropies):.4f}')
            print(f'  最大熵: {np.max(valid_entropies):.4f}')
            
            # Token count statistics
            token_counts = [stat['token_count'] for stat in question_entropy_stats if stat and stat['token_count'] > 0]
            if token_counts:
                print(f'  Question Token 数量统计:')
                print(f'    平均: {np.mean(token_counts):.1f}')
                print(f'    最小: {np.min(token_counts)}')
                print(f'    最大: {np.max(token_counts)}')
            print(f'{"="*60}\n')
    
    scores = []
    for i in range(len(final_results)):
        final_score = max(0, ((1 - 2 * abs(final_results[i]["score"] - 0.5)) if final_results[i]['question'] else -1) - penalty[i])
        score_dict = {
            "overall": final_score,
            "format": 1 if final_results[i]['question'] else 0,
            "accuracy": penalty[i]
        }
        
        # Add question entropy if available (for the stripped inner text)
        if question_entropy_stats and i < len(question_entropy_stats):
            entropy_stat = question_entropy_stats[i]
            if entropy_stat and entropy_stat['mean_entropy'] > 0:
                score_dict['question_entropy'] = entropy_stat['mean_entropy']
                score_dict['question_token_count'] = entropy_stat['token_count']
        
        scores.append(score_dict)
    
    # Save question entropy information to JSON file
    if question_entropy_stats:
        # Determine next step number for current version by checking existing files
        entropy_log_dir = f"{STORAGE_PATH}/question_entropy_logs"
        os.makedirs(entropy_log_dir, exist_ok=True)
        
        # Find existing files for current version (e.g., question_entropy_v1_step*.json)
        version_pattern = f"{entropy_log_dir}/question_entropy_{MODEL_VERSION}_step*.json"
        existing_files = glob.glob(version_pattern)
        
        if existing_files:
            # Extract step numbers from existing files
            steps = []
            for f in existing_files:
                try:
                    # Extract step number from filename like "question_entropy_v1_step3.json"
                    basename = os.path.basename(f)
                    # Remove prefix and suffix to get step number
                    step_str = basename.replace(f'question_entropy_{MODEL_VERSION}_step', '').replace('.json', '')
                    steps.append(int(step_str))
                except:
                    pass
            next_step = max(steps) + 1 if steps else 1
        else:
            next_step = 1
        
        # Prepare data to save
        entropy_data = {
            "metadata": {
                "version": MODEL_VERSION,
                "step": next_step,
                "total_samples": len(question_entropy_stats),
                "valid_samples": sum(1 for stat in question_entropy_stats if stat and stat['mean_entropy'] > 0)
            },
            "questions": []
        }
        
        for i in range(len(question_entropy_stats)):
            entropy_stat = question_entropy_stats[i]
            if entropy_stat and entropy_stat['mean_entropy'] > 0:
                question_data = {
                    "sample_id": i,
                    "question_text": final_results[i]['question'] if i < len(final_results) else "",
                    "token_entropies": entropy_stat['token_entropies'],
                    "mean_entropy": entropy_stat['mean_entropy'],
                    "token_count": entropy_stat['token_count'],
                    "score": final_results[i].get('score', 0.0) if i < len(final_results) else 0.0,
                    "penalty": penalty[i] if i < len(penalty) else 0.0
                }
                entropy_data["questions"].append(question_data)
        
        # Save to file with version and step format (v1_step1, v1_step2, ...)
        log_filename = f"{entropy_log_dir}/question_entropy_{MODEL_VERSION}_step{next_step}.json"
        
        with open(log_filename, 'w', encoding='utf-8') as f:
            json.dump(entropy_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Question 熵信息已保存到: {log_filename}")
        print(f"  - 版本: {MODEL_VERSION}")
        print(f"  - Step: {next_step}")
        print(f"  - 有效样本数: {entropy_data['metadata']['valid_samples']}/{entropy_data['metadata']['total_samples']}\n")
    
    return scores









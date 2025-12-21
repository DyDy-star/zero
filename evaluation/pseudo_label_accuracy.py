import json
import os
import requests
from tqdm import tqdm
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 设置在GPU2和GPU3上运行
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# 获取环境变量
STORAGE_PATH = os.getenv("STORAGE_PATH")
if not STORAGE_PATH:
    STORAGE_PATH = "/data/user5/R-Zero"

# SiliconFlow API 配置
API_KEY = "sk-randqfjczhdfrfaynhxqtbzwttcshputggfwenofpitwumja"
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL_NAME = "deepseek-ai/DeepSeek-V3.2-Exp"

# 请求头
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def get_ground_truth_answer(question):
    """使用 DeepSeek API 生成问题的真实答案"""
    try:
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a helpful math problem solver. Please solve the given math problem and provide only the final answer in a concise format. If the problem is a proof problem, provide the key result to be proved. If you cannot solve it, return 'Cannot solve'."},
                {"role": "user", "content": f"Please solve this math problem and provide only the final answer:\n\n{question}"}
            ],
            "temperature": 0.1,
            "stream": False
        }
        
        response = requests.post(API_URL, json=payload, headers=HEADERS)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"API Error: {e}")
        return "Error"

def compare_answers(pseudo_label, ground_truth):
    """使用 DeepSeek API 比较伪标签和真实答案是否一致"""
    # 如果伪标签为空或None，直接返回False
    if not pseudo_label or pseudo_label == "None" or pseudo_label.strip() == "":
        return False
    
    # 如果真实答案出错，跳过
    if ground_truth == "Error" or ground_truth == "Cannot solve":
        return None
    
    try:
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a math answer checker. Compare two answers and determine if they are essentially the same mathematically."},
                {"role": "user", "content": f"Are these two answers essentially the same?\n\nAnswer 1 (pseudo-label): {pseudo_label}\n\nAnswer 2 (ground truth): {ground_truth}\n\nPlease respond with ONLY 'Yes' or 'No'."}
            ],
            "temperature": 0.1,
            "stream": False
        }
        
        response = requests.post(API_URL, json=payload, headers=HEADERS)
        response.raise_for_status()
        result = response.json()
        response_text = result['choices'][0]['message']['content'].strip().lower()
        return "yes" in response_text
    except Exception as e:
        print(f"API Error in comparison: {e}")
        return None

def load_version_data(version, max_score=0.75, min_score=0.25):
    """加载指定版本的所有数据并过滤"""
    all_data = []
    version_name = f"octo_3b_solver_{version}"
    
    for i in range(4):  # v1, v2, v3 都有 0-3 四个文件
        file_path = f'{STORAGE_PATH}/generated_question/{version_name}_{i}_results.json'
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            continue
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                all_data.extend(data)
            print(f"成功加载: {file_path}, 数据条数: {len(data)}")
        except Exception as e:
            print(f"加载文件失败 {file_path}: {e}")
            continue
    
    # 过滤数据：score在min_score和max_score之间，且answer不为空
    filtered_data = [
        item for item in all_data 
        if min_score <= item.get('score', 0) <= max_score 
        and item.get('answer') 
        and item.get('answer') != 'None' 
        and item.get('answer').strip() != ''
    ]
    
    print(f"\n版本 {version}:")
    print(f"  总数据条数: {len(all_data)}")
    print(f"  过滤后训练集条数: {len(filtered_data)}")
    
    return filtered_data

def process_single_item(item):
    """处理单个数据项（用于并行处理）"""
    question = item['question']
    pseudo_label = item['answer']
    
    # 获取真实答案
    ground_truth = get_ground_truth_answer(question)
    
    # 比较答案
    is_correct = compare_answers(pseudo_label, ground_truth)
    
    # 保存结果
    result_item = {
        'question': question,
        'pseudo_label': pseudo_label,
        'ground_truth': ground_truth,
        'is_correct': is_correct,
        'score': item.get('score', 0)
    }
    
    return result_item

def evaluate_pseudo_labels(version, sample_size=None, max_score=0.75, min_score=0.25, num_workers=8):
    """评估指定版本的伪标签准确率（支持多线程并行）"""
    print(f"\n{'='*60}")
    print(f"开始评估版本 {version} 的伪标签准确率 (使用 {num_workers} 个并行线程)")
    print(f"{'='*60}")
    
    # 加载数据
    data = load_version_data(version, max_score, min_score)
    
    if not data:
        print(f"版本 {version} 没有找到符合条件的数据")
        return None
    
    # 如果指定了采样大小，随机采样
    if sample_size and sample_size < len(data):
        import random
        random.seed(42)
        data = random.sample(data, sample_size)
        print(f"随机采样 {sample_size} 条数据进行评估")
    
    # 保存中间结果的文件
    results_file = f'{STORAGE_PATH}/evaluation/pseudo_label_check_{version}_results.json'
    
    # 检查是否已有部分结果
    processed_questions = set()
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                existing_results = json.load(f)
                processed_questions = {item['question'] for item in existing_results}
                print(f"已找到 {len(processed_questions)} 条已处理的数据，将继续处理...")
        except:
            existing_results = []
    else:
        existing_results = []
    
    # 过滤出未处理的数据
    unprocessed_data = [item for item in data if item['question'] not in processed_questions]
    print(f"待处理数据: {len(unprocessed_data)} 条")
    
    if not unprocessed_data:
        print("所有数据已处理完成！")
        # 直接统计已有结果
        results = existing_results
    else:
        # 使用线程池并行处理
        results = existing_results.copy()
        results_lock = threading.Lock()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有任务
            future_to_item = {executor.submit(process_single_item, item): item for item in unprocessed_data}
            
            # 使用tqdm显示进度
            for future in tqdm(as_completed(future_to_item), total=len(unprocessed_data), desc=f"处理版本 {version}"):
                try:
                    result_item = future.result()
                    
                    # 线程安全地添加结果
                    with results_lock:
                        results.append(result_item)
                        
                        # 每处理10条就保存一次
                        if len(results) % 10 == 0:
                            with open(results_file, 'w', encoding='utf-8') as f:
                                json.dump(results, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"处理失败: {e}")
        
        # 最终保存
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 统计结果
    correct_count = sum(1 for r in results if r.get('is_correct') == True)
    skipped_count = sum(1 for r in results if r.get('is_correct') is None)
    total_count = len(results) - skipped_count
    
    # 计算准确率
    if total_count > 0:
        accuracy = correct_count / total_count * 100
    else:
        accuracy = 0
    
    result_summary = {
        'version': version,
        'total_samples': len(data),
        'processed_count': total_count,
        'correct_count': correct_count,
        'skipped_count': skipped_count,
        'accuracy': accuracy
    }
    
    print(f"\n版本 {version} 评估结果:")
    print(f"  训练集总数: {len(data)}")
    print(f"  成功处理: {total_count}")
    print(f"  正确数量: {correct_count}")
    print(f"  跳过数量: {skipped_count}")
    print(f"  伪标签准确率: {accuracy:.2f}%")
    
    return result_summary

def main():
    parser = argparse.ArgumentParser(description='评估伪标签准确率（支持多线程加速）')
    parser.add_argument('--versions', type=str, default='v1,v2,v3', 
                        help='要评估的版本，用逗号分隔，例如: v1,v2,v3')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='每个版本采样的数据量，默认使用全部数据')
    parser.add_argument('--max_score', type=float, default=0.75,
                        help='训练集过滤的最大分数阈值')
    parser.add_argument('--min_score', type=float, default=0.25,
                        help='训练集过滤的最小分数阈值')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='并行处理的线程数，默认16（建议8-32之间）')
    args = parser.parse_args()
    
    versions = args.versions.split(',')
    
    all_results = []
    
    for version in versions:
        version = version.strip()
        result = evaluate_pseudo_labels(
            version, 
            sample_size=args.sample_size,
            max_score=args.max_score,
            min_score=args.min_score,
            num_workers=args.num_workers
        )
        if result:
            all_results.append(result)
    
    # 保存汇总结果
    summary_file = f'{STORAGE_PATH}/evaluation/pseudo_label_accuracy_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print("所有版本评估完成！")
    print(f"{'='*60}")
    print("\n汇总结果:")
    for result in all_results:
        print(f"\n版本 {result['version']}:")
        print(f"  训练集总数: {result['total_samples']}")
        print(f"  成功处理: {result['processed_count']}")
        print(f"  正确数量: {result['correct_count']}")
        print(f"  伪标签准确率: {result['accuracy']:.2f}%")
    
    print(f"\n详细结果已保存至: {summary_file}")

if __name__ == "__main__":
    main()



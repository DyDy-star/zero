import json
from mathruler.grader import extract_boxed_content, grade_answer
import requests
from tqdm import tqdm
import random
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
args = parser.parse_args()

STORAGE_PATH = os.getenv("STORAGE_PATH")

# SiliconFlow API 配置
API_KEY = "sk-ztultkmdsppcbrjwkjvvpbgpbeyfknxolqxtuapfkabapmne"
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL_NAME = "deepseek-ai/DeepSeek-V3.2-Exp"

# 请求头
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}



def process_example(answer, response):
    """使用 DeepSeek API 检查答案是否正确"""
    try:
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a math answer checker."},
                {"role": "user", "content": f"Hi, there is a answer: {answer}\n\n, and the ground truth answer is: {response}\n\n, please check whether the answer is correct or not, and return the **only** Yes or No."}
            ],
            "temperature": 0.1,
            "stream": False
        }
        
        response_obj = requests.post(API_URL, json=payload, headers=HEADERS)
        response_obj.raise_for_status()
        result = response_obj.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        print(f"API Error: {e}")
        return "No"
new_results = []
for model_name in [args.model_name]:
    for dataset in [
    "math",
    "gsm8k", 
    "amc",
    ]:
        result_file = f'{STORAGE_PATH}/evaluation/{model_name.replace("/","_")}/results_{dataset}.json'
        if not os.path.exists(result_file):
            print(f"Skipping {dataset} - file not found: {result_file}")
            continue
        with open(result_file, 'r') as f:
            results = json.load(f)

        for i in tqdm(range(len(results)-1)):
                if results[i]['score'] < 0.5:
                    gpt_check = process_example(results[i]['answer'],results[i]['response'])
                    if "yes" in gpt_check.lower():
                        results[i]['score']=1
        new_results.append({
            'model': model_name,
            'dataset': dataset,
            'score': round(sum([result['score'] for result in results[:-1]])/len(results[:-1])*100, 2)
        })
        print(new_results)
        with open(f'final_results_unified（deepseek-reasoner）.jsonl', 'a') as f:
            json.dump({
                'model': model_name,
                'dataset': dataset,
                'score': round(sum([result['score'] for result in results[:-1]])/len(results[:-1])*100, 2)
            }, f)
            f.write('\n')






import json
from mathruler.grader import extract_boxed_content, grade_answer
from openai import OpenAI
from tqdm import tqdm
import random
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
args = parser.parse_args()

STORAGE_PATH = os.getenv("STORAGE_PATH")

# API 配置 - DeepSeek API
API_KEY = "sk-7742d7286f5c408ab4e05c8a317f4836"
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)



def process_example(answer, response):
    """使用 DeepSeek API 检查答案是否正确"""
    try:
        # 使用 OpenAI SDK 调用 DeepSeek API
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a math answer checker."},
                {"role": "user", "content": f"Hi, there is a answer: {answer}\n\n, and the ground truth answer is: {response}\n\n, please check whether the answer is correct or not, and return the **only** Yes or No."}
            ],
            temperature=0.1,
            stream=False
        )
        return completion.choices[0].message.content
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
        with open(f'final_results（deepseek-reasoner）.jsonl', 'a') as f:
            json.dump({
                'model': model_name,
                'dataset': dataset,
                'score': round(sum([result['score'] for result in results[:-1]])/len(results[:-1])*100, 2)
            }, f)
            f.write('\n')






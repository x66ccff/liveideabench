import numpy as np
import pandas as pd
import random
import re
import os
import json
import argparse
import requests
from config import CRITIC_MODELS, IDEA_MODELS
from utils.LLM import create_llm
from utils.utils import stable_hash, run_with_timeout
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

try:
    from FlagEmbedding import BGEM3FlagModel
    bge_model = None  # Will be loaded on demand
except ImportError:
    print("Warning: FlagEmbedding not installed. BGE embedding method will not be available.")

# For MXBai embedding model
try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim
    mxbai_model = None  # Will be loaded on demand
except ImportError:
    print("Warning: sentence-transformers not installed. MXBai embedding method will not be available.")

def calculate_jaccard_similarity(idea_A, idea_B):
    # Tokenize and create sets
    set_A = set(word_tokenize(idea_A.lower()))
    set_B = set(word_tokenize(idea_B.lower()))
    
    # Calculate intersection and union
    intersection = set_A.intersection(set_B)
    union = set_A.union(set_B)
    
    # Jaccard similarity formula
    jaccard_similarity = len(intersection) / len(union) if len(union) > 0 else 0
    
    # Convert to dissimilarity score on the 0-10 scale (0=identical, 10=completely different)
    score = (1 - jaccard_similarity) * 10
    
    return score

def calculate_tfidf_similarity(idea_A, idea_B):
    # Use TfidfVectorizer for tokenization and TF-IDF calculation
    vectorizer = TfidfVectorizer(tokenizer=word_tokenize, lowercase=True)
    
    # Input texts
    ideas = [idea_A, idea_B]
    
    # Calculate TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(ideas)
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    # Convert to dissimilarity score on the 0-10 scale
    score = (1 - cosine_sim) * 10
    
    return score

def calculate_embedding_similarity(idea_A, idea_B):
    global bge_model
    
    # Load model on first use
    if bge_model is None:
        bge_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    
    # Encode the ideas
    embedding_A = bge_model.encode([idea_A], batch_size=1, max_length=8192)['dense_vecs']
    embedding_B = bge_model.encode([idea_B], batch_size=1, max_length=8192)['dense_vecs']
    
    # Calculate similarity
    similarity = (embedding_A @ embedding_B.T)[0][0]
    
    # Convert to dissimilarity score on the 0-10 scale
    score = float((1 - similarity) * 10)
    
    return score

def calculate_embedding_mxbai(idea_A, idea_B):
    global mxbai_model
    
    # Load model on first use
    if mxbai_model is None:
        # Set dimensions to 512 as in the example
        dimensions = 512
        mxbai_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", truncate_dim=dimensions)
    
    # Encode the ideas
    embedding_A = mxbai_model.encode(idea_A)
    embedding_B = mxbai_model.encode(idea_B)
    
    # Calculate similarity using cos_sim function
    similarity = cos_sim(embedding_A.reshape(1, -1), embedding_B.reshape(1, -1))[0][0]
    
    # Convert to dissimilarity score on the 0-10 scale (0=identical, 10=completely different)
    score = float((1 - similarity) * 10)
    
    return score

def query_local_model(prompt, model="olmo2:7b"):
    """Query the local Ollama model running on port 11434"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            print(f"Error from Ollama API: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return None

def main(max_num_eval=None, method="llm"):
    
    # 设置固定的随机种子确保可重复性
    random.seed(42)  # 可以使用任意固定数字作为种子
    np.random.seed(42)  # 确保numpy的随机函数也是确定性的
    # 加载和预处理数据
    df = pd.read_parquet('./data/data.parquet')

    # 筛选idea长度
    df['idea_length_in_char'] = df['idea'].apply(lambda x: len(str(x)))
    df = df[(df['idea_length_in_char'] >= 100) & (df['idea_length_in_char'] < 2500)]

    # 筛选词数
    df['idea_length_in_words'] = df['idea'].apply(lambda x: len(str(x).split()))
    df = df[df['idea_length_in_words'] < 200]

    # 提取唯一的关键词和模型名称
    all_keywords = df['keywords'].unique().tolist()
    all_idea_models = df['idea_model'].unique().tolist()

    # 创建或加载JSON文件
    # json_file_path = './data/idea_hash.json'
    json_file_path = f'./data/idea_hash_{method}.json'
    if not os.path.exists(json_file_path):
        with open(json_file_path, 'w') as json_file:
            json.dump([], json_file)
    
    # 读取现有的评价数据
    try:
        with open(json_file_path, 'r') as json_file:
            ideas_data = json.load(json_file)
    except Exception as e:
        print(f"读取JSON文件时出错: {e}")
        ideas_data = []
    
    # 获取已评估的数量
    num_evaluated = len(ideas_data)
    print(f"当前已评估 {num_evaluated} 对想法")
    
    # 如果已达到目标数量，则直接返回
    if max_num_eval is not None and num_evaluated >= max_num_eval:
        print(f"已达到目标评估数量: {max_num_eval}")
        return
    
    # 创建所有可能的关键词和模型组合
    keyword_model_pairs = [(keyword, idea_model) for keyword in all_keywords for idea_model in all_idea_models]
    
    # 已经评估过的idea哈希
    existing_hashes = set()
    for item in ideas_data:
        existing_hashes.add(item.get('hash_A', ''))
        existing_hashes.add(item.get('hash_B', ''))
    
    # 随机打乱关键词和模型组合
    random.shuffle(keyword_model_pairs)
    
    # 如果使用LLM或SLM方法，则读取提示模板
    critic_prompt = None
    if method == "llm" or method == "slm":
        with open('utils/prompts.json', 'r', encoding='utf-8') as f:
            prompts = json.load(f)
            critic_prompt = prompts["fluency_critic_prompt"]["description"]
    
    # 主处理逻辑
    for keyword, idea_model in keyword_model_pairs:
        # 检查是否达到目标数量
        if max_num_eval is not None and len(ideas_data) >= max_num_eval:
            print(f"已达到目标评估数量: {max_num_eval}")
            break
            
        # 筛选特定关键词和模型的数据
        df_view = df[(df['keywords'] == keyword) & (df['idea_model'] == idea_model)]
        
        if len(df_view) < 2:
            continue
            
        # 提取唯一的想法列表
        unique_ideas_list = df_view['idea'].unique().tolist()
        
        if len(unique_ideas_list) <= 1:
            print('只有一个或没有想法，跳过')
            continue
        
        # 随机选择两个想法进行比较
        random.shuffle(unique_ideas_list)
        idea_A = unique_ideas_list[0]
        idea_B = unique_ideas_list[1]
        
        # 计算哈希值
        idea_hash_A = stable_hash(idea_A)
        idea_hash_B = stable_hash(idea_B)
        
        # 检查是否已经评价过这对想法
        if idea_hash_A in existing_hashes or idea_hash_B in existing_hashes:
            print('想法已存在，跳过', idea_A[:10], '...')
            continue
        
        # 根据选择的方法计算相似性分数
        success = False
        reasoning = None
        critic_model = None
        
        if method == "llm":
            # 原有的LLM评价逻辑
            critic_model = random.choice(CRITIC_MODELS)
            filled_prompt = critic_prompt.replace("{{keyword}}", keyword).replace("{{A}}", idea_A).replace("{{B}}", idea_B)
            critic_llm = create_llm("critic", critic_model)

            def do(idea, prompt_this):
                critique = critic_llm.critique_idea(idea, prompt=prompt_this)
                return critique
            
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries and not success:
                try:
                    critique = run_with_timeout(do, timeout=5, idea=idea_A, prompt_this=filled_prompt)
                    answers = ['A','B','C','D']
                    
                    # 处理推理部分
                    if isinstance(critique, tuple):
                        reasoning = critique[1]
                        critique = critique[0]
                    
                    # 清理结果，只保留字母
                    cleaned_critique = re.sub(r'[^a-zA-Z]', '', critique)
                    
                    if any(answer in cleaned_critique for answer in answers):
                        # 根据回答计算相似性分数
                        score_mapping = {
                            'A': 10,      # 完全不同
                            'B': 6.6667,  # 类似但不等价
                            'C': 3.3333,  # 非常相似
                            'D': 0        # 完全相同
                        }
                        score = score_mapping[cleaned_critique[0]]
                        success = True
                    else:
                        print(f"未能获取有效答案，重试 {retry_count+1}/{max_retries}")
                        retry_count += 1
                except Exception as e:
                    print(f"错误: {e}, 重试 {retry_count+1}/{max_retries}")
                    retry_count += 1
            
            if not success:
                print(f"三次尝试后仍然失败，跳过当前比较")
                continue
        
        elif method == "slm":
            # SLM (Local Ollama model) 评价逻辑
            critic_model = "olmo2:7b"
            filled_prompt = critic_prompt.replace("{{keyword}}", keyword).replace("{{A}}", idea_A).replace("{{B}}", idea_B)
            
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries and not success:
                try:
                    critique = query_local_model(filled_prompt, model=critic_model)
                    if critique:
                        answers = ['A','B','C','D']
                        
                        # 使用正则表达式查找答案和解释
                        answer_match = re.search(r'Answer: ([A-D])', critique)
                        reasoning_match = re.search(r'Reasoning: (.*?)($|Answer:)', critique, re.DOTALL)
                        
                        if answer_match:
                            answer = answer_match.group(1)
                            if reasoning_match:
                                reasoning = reasoning_match.group(1).strip()
                                
                            # 根据回答计算相似性分数
                            score_mapping = {
                                'A': 10,      # 完全不同
                                'B': 6.6667,  # 类似但不等价
                                'C': 3.3333,  # 非常相似
                                'D': 0        # 完全相同
                            }
                            
                            if answer in score_mapping:
                                score = score_mapping[answer]
                                success = True
                            else:
                                print(f"未识别的答案: {answer}, 重试 {retry_count+1}/{max_retries}")
                                retry_count += 1
                        else:
                            # 如果没有找到明确的答案格式，尝试直接找字母
                            cleaned_critique = re.sub(r'[^a-zA-Z]', '', critique)
                            for ans in answers:
                                if ans in cleaned_critique:
                                    score_mapping = {
                                        'A': 10,      # 完全不同
                                        'B': 6.6667,  # 类似但不等价
                                        'C': 3.3333,  # 非常相似
                                        'D': 0        # 完全相同
                                    }
                                    score = score_mapping[ans]
                                    success = True
                                    break
                            
                            if not success:
                                print(f"未能获取有效答案，重试 {retry_count+1}/{max_retries}")
                                retry_count += 1
                    else:
                        print(f"未获取到Ollama响应，重试 {retry_count+1}/{max_retries}")
                        retry_count += 1
                except Exception as e:
                    print(f"错误: {e}, 重试 {retry_count+1}/{max_retries}")
                    retry_count += 1
            
            if not success:
                print(f"三次尝试后仍然失败，跳过当前比较")
                continue
        
        elif method == "jaccard":
            try:
                score = calculate_jaccard_similarity(idea_A, idea_B)
                success = True
                reasoning = f"Jaccard similarity calculated between the two ideas"
            except Exception as e:
                print(f"Jaccard计算错误: {e}")
                continue
        
        elif method == "tfidf":
            try:
                score = calculate_tfidf_similarity(idea_A, idea_B)
                success = True
                reasoning = f"TF-IDF similarity calculated between the two ideas"
            except Exception as e:
                print(f"TF-IDF计算错误: {e}")
                continue
        
        elif method == "embedding":
            try:
                score = calculate_embedding_similarity(idea_A, idea_B)
                success = True
                reasoning = f"BGE-M3 embedding similarity calculated between the two ideas"
            except Exception as e:
                print(f"Embedding计算错误: {e}")
                continue
                
        elif method == "embedding_mxbai":
            try:
                score = calculate_embedding_mxbai(idea_A, idea_B)
                success = True
                reasoning = f"MXBai embedding similarity calculated between the two ideas"
            except Exception as e:
                print(f"MXBai Embedding计算错误: {e}")
                continue
        
        else:
            print(f"不支持的方法: {method}")
            continue

        # 输出结果
        print(f"Score: {score}, Hash A: {idea_hash_A}, Hash B: {idea_hash_B}")

        # 创建新的数据项
        new_idea_data = {
            "hash_A": idea_hash_A,
            "hash_B": idea_hash_B,
            "score": score,
            "method": method
        }
        
        # 添加LLM/SLM相关信息
        if method == "llm" or method == "slm":
            new_idea_data["fluency_critic_model"] = critic_model
        
        if reasoning:
            new_idea_data["reasoning"] = reasoning
        
        # 添加并保存数据
        ideas_data.append(new_idea_data)
        existing_hashes.add(idea_hash_A)
        existing_hashes.add(idea_hash_B)
        
        with open(json_file_path, 'w') as json_file:
            json.dump(ideas_data, json_file, indent=2)
        
        print(f"当前已评估 {len(ideas_data)} 对想法")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate ideas for fluency.')
    parser.add_argument('--max_num_eval', type=int, help='Maximum number of evaluations to perform')
    parser.add_argument('--method', type=str, choices=['llm', 'slm', 'jaccard', 'tfidf', 'embedding', 'embedding_mxbai'], 
                       default='llm', help='Method to calculate similarity (default: llm, slm for local Ollama model)')
    args = parser.parse_args()
    
    main(args.max_num_eval, args.method)
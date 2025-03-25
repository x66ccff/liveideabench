#!/usr/bin/env python3
"""
LiveIdeaBench 主程序

用于运行科学想法生成和评价的基准测试
"""

import os
import random
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import pandas as pd
from openpyxl import load_workbook

# 自定义模块
from config import config
from utils.LLM import create_llm, parse_critique, is_response_rejected
from utils.database import save_result, check_duplicate_entries, close_all_connections
from config import CRITIC_MODELS, IDEA_MODELS

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'./logs/run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 确保目录存在
os.makedirs('./logs', exist_ok=True)
os.makedirs('./data', exist_ok=True)

    


def load_keywords(file_path: str = './keywords_data/keywordsEverywhere20241216.xlsx') -> List[str]:
    """加载关键词数据

    Args:
        file_path: Excel文件路径

    Returns:
        关键词列表
    """
    try:
        wb = load_workbook(file_path)
        sheet = wb['Sheet1']
        
        data = sheet.values
        columns = next(data)
        df = pd.DataFrame(data, columns=columns)
        df = df.drop(0).reset_index(drop=True)
        df.columns = ['Index', 'Keyword', 'Search_Volume', 'CPC', 'Competition', 'Trend']
        
        # 清理数据
        df = df.replace('None', pd.NA)
        df = df.drop(columns=['Index', 'Search_Volume', 'CPC', 'Competition', 'Trend'])
        df = df.astype(str)
        
        return df['Keyword'].tolist()
    except Exception as e:
        logger.error(f"加载关键词文件失败: {str(e)}")
        raise


def load_prompts(file_path: str = './utils/prompts.json') -> Dict[str, Dict[str, str]]:
    """加载提示模板

    Args:
        file_path: JSON提示文件路径

    Returns:
        提示模板字典
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        logger.error(f"加载提示文件失败: {str(e)}")
        raise


def clean_text(text: str) -> str:
    """清理文本，移除不必要的空白字符

    Args:
        text: 原始文本

    Returns:
        清理后的文本
    """
    # 移除换行符、回车符和制表符
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    # 将多个空格替换为单个空格
    return ' '.join(text.split())


def run_evaluation(keyword: str, idea_model: str, critic_models: List[str], 
                  prompts: Dict[str, Dict[str, str]], provider: Optional[str] = None) -> None:
    """运行单次评价

    Args:
        keyword: 关键词
        idea_model: 想法模型名称
        critic_models: 评价模型列表
        prompts: 提示模板
        provider: 可选的模型提供商
    """
    # 准备idea prompt
    idea_prompt = prompts['idea_prompt']['description'].replace('{{keywords}}', str(keyword))
    idea_fallback_prompt = prompts['idea_prompt']['fallback_description'].replace('{{keywords}}', str(keyword))
    
    # 创建idea LLM并生成想法
    logger.info(f"使用模型 {idea_model} 生成关于 '{keyword}' 的想法")
    

    idea_llm = create_llm("idea", idea_model, provider)
    generation_result = idea_llm.generate_idea(idea_prompt, fallback_prompt=idea_fallback_prompt)
    
    idea = generation_result["idea"]
    full_response = generation_result["full_response"]
    first_was_rejected = generation_result["first_was_rejected"]
    first_reject_response = generation_result.get("first_reject_response")
    used_fallback = generation_result.get("used_fallback", False)
    
    if first_was_rejected:
        logger.warning(f"模型 {idea_model} 拒绝了请求" + (" (使用回退提示后)" if used_fallback else ""))
    
    if len(idea) < 2:
        raise ValueError(f"生成的想法长度过短: {len(idea)} 字符")
    
    logger.info(f"想法生成完成，长度: {len(idea)} 字符")
    
    # 对每个评委模型进行评价
    for critic_model in critic_models:
        logger.info(f"使用评委 {critic_model} 评价想法")
        
        critic_prompt = prompts['critic_prompt']['description'].replace('{{keywords}}', str(keyword))
        
        critic_llm = create_llm("critic", critic_model, provider)
        critique = critic_llm.critique_idea(idea, critic_prompt=critic_prompt)
        
        # 处理critique可能是元组的情况（当模型返回reasoning字段时）
        critique_reasoning = None
        if isinstance(critique, tuple) and len(critique) == 2:
            critique_reasoning = critique[1]  # 保存模型的推理过程
            critique = critique[0]  # 提取第一个元素作为批评内容
        
        # 添加重试逻辑，最多尝试3次
        max_retries = 3
        retry_count = 0
        parsed_result = None
        error_msg = None
        
        while retry_count < max_retries:
            parsed_result = parse_critique(critique)
            
            # 检查解析结果是否有效
            if parsed_result.get('is_valid', False) and parsed_result.get('scores'):
                break  # 解析成功，跳出循环
            
            retry_count += 1
            logger.warning(f"评价解析失败 (尝试 {retry_count}/{max_retries}): 无法获取有效评分")
            
            if retry_count < max_retries:
                # 重新请求评价
                logger.info(f"重新请求评价...")
                critique = critic_llm.critique_idea(idea, critic_prompt=critic_prompt)
                
                # 处理新的critique可能是元组的情况
                if isinstance(critique, tuple) and len(critique) == 2:
                    critique_reasoning = critique[1]
                    critique = critique[0]
        
        # 如果所有尝试都失败，则记录错误信息
        if retry_count == max_retries and not (parsed_result.get('is_valid', False) and parsed_result.get('scores')):
            error_msg = f"评价解析失败: 3次尝试后仍无法获取有效评分"
            logger.error(error_msg)
        
        # 保存结果
        result_data = {
            'timestamp': datetime.now().isoformat(),
            'keywords': str(keyword),
            'idea_model': idea_model,
            'critic_model': critic_model,
            'idea': clean_text(idea),
            'raw_critique': clean_text(critique),
            'parsed_score': parsed_result.get('scores'),
            'parsed_feedback': parsed_result.get('reasoning'),
            'critique_reasoning': clean_text(critique_reasoning) if critique_reasoning else None,
            'error': error_msg,
            'full_response': clean_text(full_response),
            'first_was_rejected': first_was_rejected,
            'first_reject_response': first_reject_response,
            'used_fallback': used_fallback
        }
        
        # 只有在评价有效或明确记录了错误的情况下才保存结果
        save_result(result_data)
        
        if error_msg:
            raise ValueError(error_msg)
        else:
            logger.info(f"评价结果已保存")
            


def continue_from_last_run() -> None:
    """从上次运行继续，找出评价次数最少的组合并运行"""
    from utils.database import query_results
    
    logger.info("继续上次运行，查找待补充的模型-关键词组合...")
    
    # 获取所有结果的统计
    results = query_results()
    
    # 如果没有任何结果，无法继续
    if not results:
        logger.info("数据库中没有任何记录，无法继续上次运行")
        return
    
    # 统计每个关键词-模型组合的评价次数
    combo_counts = {}
    used_critics = {}
    
    for result in results:
        key = (result['keywords'], result['idea_model'])
        
        if key not in combo_counts:
            combo_counts[key] = 0
            used_critics[key] = set()
            
        combo_counts[key] += 1
        used_critics[key].add(result['critic_model'])
    
    # 找出评价次数最少的组合
    min_count = min(combo_counts.values())
    min_combos = [key for key, count in combo_counts.items() if count == min_count]
    
    if not min_combos:
        logger.info("所有组合已达到相同的评价次数，无需补充")
        return
    
    # 随机选择一个组合
    selected_combo = random.choice(min_combos)
    keyword, idea_model = selected_combo
    
    logger.info(f"选择补充: 关键词 '{keyword}', 模型 '{idea_model}', 当前评价次数: {min_count}")
    
    # 选择未使用过的评委
    available_critics = [m for m in CRITIC_MODELS if m != idea_model and m not in used_critics[selected_combo]]
    
    if not available_critics:
        logger.info(f"所有评委已用于此组合，跳过")
        return
    
    # 随机选择一个未使用过的评委
    critic_model = random.choice(available_critics)
    
    # 运行评价
    prompts = load_prompts()
    run_evaluation(keyword, idea_model, [critic_model], prompts)


def main() -> None:
    """主函数，处理命令行参数并运行评价"""
    parser = argparse.ArgumentParser(description="LiveIdeaBench - LLM科学想法评价基准")
    parser.add_argument('--idea_model', type=str, help='想法生成模型名称')
    parser.add_argument('--start_from_last_run', action='store_true', help='从上次运行继续')
    parser.add_argument('--provider', type=str, choices=['openrouter', 'gemini', 'stepfun', 'ollama'],
                       help='模型提供商，默认根据模型名称自动推断')
    parser.add_argument('--keyword', nargs='+', help='指定关键字进行想法生成，可以是单个关键字或多个关键字列表，不指定则使用所有关键字')
    
    args = parser.parse_args()
    
    # 设置默认提供商
    if args.provider:
        config.set_default_provider(args.provider)
        logger.info(f"设置默认提供商: {args.provider}")
    
    if args.start_from_last_run:
        logger.info("从上次运行继续...")
        continue_from_last_run()
        return
    
    if not args.idea_model:
        logger.error("必须提供想法生成模型名称")
        return
    
    if args.idea_model not in IDEA_MODELS:
        logger.error(f"不支持的模型: {args.idea_model}")
        return
    
    # 加载关键词和提示
    all_keywords = load_keywords()
    prompts = load_prompts()
    
    # 处理用户指定的关键词
    if args.keyword:
        specified_keywords = args.keyword
        logger.info(f"使用用户指定的关键词: {', '.join(specified_keywords)}")
    else:
        specified_keywords = all_keywords
        logger.info("使用所有关键词")
    
    while True:
        # 随机选择一个关键词（从指定的关键词列表中）
        keyword = random.choice(specified_keywords)
        
        # 检查是否已有足够数量的评价
        if check_duplicate_entries(keyword, args.idea_model):
            logger.info(f"跳过 '{keyword}' 和 '{args.idea_model}'，已有足够评价")
            if len(specified_keywords) == 1:
                logger.info("用户仅指定了一个关键词，且已有足够评价，终止程序")
                break
            continue
        
        # 从评委模型中随机选择3个，但排除idea模型本身
        available_critics = [m for m in CRITIC_MODELS if m != args.idea_model]
        num_critics = 3
        critic_models = random.sample(available_critics, min(num_critics, len(available_critics)))
        
        logger.info(f"选择关键词: '{keyword}'")
        logger.info(f"选择评委: {', '.join(critic_models)}")
        
        # 运行评价
        run_evaluation(keyword, args.idea_model, critic_models, prompts, args.provider)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("用户中断了程序")
    # except Exception as e:
    #     logger.exception(f"程序发生意外错误: {str(e)}")
    finally:
        # 确保关闭所有数据库连接
        close_all_connections()

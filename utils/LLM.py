"""
LLM交互和处理模块

负责与各种LLM API的通信，处理请求和响应
"""

import os
import json
import re
import logging
import time
import random
from typing import Dict, List, Optional, Union, Tuple, Any

from openai import OpenAI
import google.generativeai as genai
import random

from config import config

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 拒绝识别关键词
REJECTION_PHRASES = [
    "I'm sorry", "I am sorry", "I apologize", "As an AI", "As a language model",
    "As an assistant", "I cannot", "I can't", "I am unable to", "I'm unable to", 
    "I am not able to", "I'm not able to"
]

# OpenRouter模型名称到AIGCBest模型名称的映射字典
OPENROUTER_TO_AIGCBEST_MODEL_MAPPING = {
    # 这里将由用户填入具体的模型映射
    "google/gemini-2.0-flash-thinking-exp:free": "gemini-2.0-flash-thinking-exp",
    "google/gemini-2.0-pro-exp-02-05:free": "gemini-2.0-pro-exp-02-05",
    "google/gemini-2.0-flash-exp:free":"gemini-2.0-flash-exp",
    "google/gemini-pro-1.5":"gemini-1.5-pro-002",
    "google/gemini-2.0-flash-lite-001":"gemini-2.0-flash-lite-preview-02-05",
}


class BaseLLM:
    """基础LLM类，处理与LLM API的通信"""
    
    def __init__(self, model_name: str, provider: Optional[str] = None):
        """初始化LLM

        Args:
            model_name: 模型名称
            provider: 模型提供商，如果为None，将根据model_name自动推断
        """
        self.model_name = model_name
        self.provider = provider if provider else config.get_provider_for_model(model_name)
        self._setup_client()
        
    def _setup_client(self) -> None:
        """根据提供商设置客户端"""
        if self.provider == "gemini":
            # 设置Gemini API
            genai.configure(api_key=config.get_api_key("gemini"))
            
        elif self.provider == "ollama":
            # 设置Ollama API
            self.client = OpenAI(
                base_url='http://localhost:11434/v1',
                api_key="ollama"
            )
            
        elif self.provider == "stepfun":
            # 设置StepFun API
            api_key = config.get_api_key("stepfun")
            if not api_key:
                raise ValueError("未找到StepFun API密钥")
            self.client = OpenAI(
                base_url="https://api.stepfun.com/v1",
                api_key=api_key
            )
            
        elif self.provider == "aigcbest":
            # 设置AIGCBest API
            api_key = config.get_api_key("aigcbest")
            if not api_key:
                raise ValueError("未找到AIGCBest API密钥")
            self.client = OpenAI(
                base_url="https://api2.aigcbest.top/v1",
                api_key=api_key
            )
            
        else:
            # 默认为OpenRouter
            api_key = config.get_api_key("openrouter")
            if not api_key:
                raise ValueError("未找到OpenRouter API密钥")
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key
            )
    
    def completion(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """执行非流式LLM完成请求

        Args:
            prompt: 用户提示
            system_prompt: 可选的系统提示

        Returns:
            LLM的响应文本
        """
        if self.provider == "gemini":
            return self._gemini_completion(prompt, system_prompt)
        else:
            return self._openai_compatible_completion(prompt, system_prompt)
    
    def _gemini_completion(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """执行Gemini API的完成请求

        Args:
            prompt: 用户提示
            system_prompt: 可选的系统提示

        Returns:
            Gemini模型的响应文本
        """

        model = genai.GenerativeModel(model_name=self.model_name)
        
        # 构建提示
        if system_prompt:
            prompt = f"{system_prompt}\n\n{prompt}"
        
        response = model.generate_content(prompt)
        return response.text

    
    def _openai_compatible_completion(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """执行与OpenAI兼容API的完成请求

        Args:
            prompt: 用户提示
            system_prompt: 可选的系统提示

        Returns:
            模型的响应文本
        """
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        # 如果是AIGCBest提供商，需要将模型名称转换为AIGCBest支持的名称
        model_name = self.model_name
        if self.provider == "aigcbest" and model_name in OPENROUTER_TO_AIGCBEST_MODEL_MAPPING:
            model_name = OPENROUTER_TO_AIGCBEST_MODEL_MAPPING[model_name]
            logger.info(f"AIGCBest API: 将模型名称从 {self.model_name} 转换为 {model_name}")
        
        # 实现指数退避重试
        max_retries = 10  # 最大重试次数
        max_wait_time = 300  # 最大等待时间（5分钟 = 300秒）
        base_wait_time = 1  # 初始等待时间（秒）
        
        retry_count = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    stream=False
                )
                
                print('-'*40)
                from pprint import pprint
                pprint(response)
                print('-'*40)
                
                # 检查是否是500错误
                if hasattr(response, 'error') and isinstance(response.error, dict) and response.error.get('code') == 500:
                    error_msg = response.error.get('message', '内部服务器错误')
                    logger.warning(f"遇到500错误: {error_msg}，准备重试 ({retry_count + 1}/{max_retries})")
                    raise Exception(f"内部服务器错误: {error_msg}")
                
                # 检查响应是否有效
                if response and hasattr(response, 'choices') and response.choices and len(response.choices) > 0:
                    # 检查是否有reasoning字段
                    if hasattr(response.choices[0].message, 'reasoning') and response.choices[0].message.reasoning:
                        return (response.choices[0].message.content, response.choices[0].message.reasoning)
                    else:
                        return response.choices[0].message.content
                else:
                    # 检查是否是500错误对象
                    if response is None or (hasattr(response, 'id') and response.id is None and hasattr(response, 'error') and 
                       isinstance(response.error, dict) and response.error.get('code') == 500):
                        error_msg = response.error.get('message', '内部服务器错误') if hasattr(response, 'error') else "内部服务器错误"
                        logger.warning(f"遇到500错误: {error_msg}，准备重试 ({retry_count + 1}/{max_retries})")
                        raise Exception(f"内部服务器错误: {error_msg}")
                    
                    # 返回错误消息作为响应，防止None值返回
                    error_msg = "API响应无效或不完整"
                    logger.error(f"API响应错误: {error_msg}")
                    print("response", response)
                    
                    from pprint import pprint
                    pprint(response)
                    
                    raise ValueError(error_msg)
            
            except Exception as e:
                # 检查是否是内部服务器错误
                error_str = str(e).lower()
                is_server_error = "500" in error_str or "内部服务器错误" in error_str or "internal server error" in error_str
                
                # 如果不是服务器错误或已达到最大重试次数，则抛出异常
                if not is_server_error or retry_count >= max_retries:
                    raise
                
                # 计算等待时间（指数退避 + 随机抖动）
                wait_time = min(base_wait_time * (2 ** retry_count) + random.uniform(0, 1), max_wait_time)
                
                # 如果等待时间超过最大等待时间，则抛出异常
                if wait_time >= max_wait_time:
                    logger.error(f"达到最大等待时间 ({max_wait_time}秒)，停止重试。")
                    raise ValueError(f"多次重试后仍然失败: {e}")
                
                logger.warning(f"重试 {retry_count + 1}/{max_retries}，等待 {wait_time:.2f} 秒...")
                time.sleep(wait_time)
                retry_count += 1
                continue
            



def extract_final_idea(text: str) -> str:
    """从带有Final Idea标记的文本中提取最终想法

    Args:
        text: 输入文本

    Returns:
        提取的最终想法或原始文本
    """
    # 定义所有可能的标记，从最严格到最宽松
    markers = [
        "**Final Idea:**",
        "**Final Idea**:",
        "**final idea:**",
        "**final idea**:",
        "final idea"
    ]
    
    # 对输入文本进行规范化处理，去除首尾空白
    text = text.strip()
    
    # 依次尝试每个标记
    for marker in markers:
        # 对当前标记进行不区分大小写的搜索
        marker_lower = marker.lower()
        text_lower = text.lower()
        
        if marker_lower in text_lower:
            # 找到标记在原文中的实际位置
            start_idx = text_lower.index(marker_lower)
            # 从原文中提取标记之后的内容
            final_idea = text[start_idx + len(marker):].strip()
            return final_idea
            
    # 如果所有标记都没找到，返回完整文本
    return text


def is_response_rejected(text: str) -> bool:
    """检查响应是否包含拒绝短语

    Args:
        text: 要检查的文本

    Returns:
        如果文本包含拒绝短语，返回True；否则返回False
    """
    return any(phrase.lower() in text.lower() for phrase in REJECTION_PHRASES)


class IdeaLLM(BaseLLM):
    """处理科学想法生成的专用LLM类"""
    
    def generate_idea(self, prompt: str, fallback_prompt: Optional[str] = None) -> Dict[str, str]:
        """生成科学想法

        Args:
            prompt: 主要提示
            fallback_prompt: 可选的回退提示，用于模型拒绝时

        Returns:
            包含想法和完整响应的字典
        """
        if "qwq-32b-preview" in self.model_name.lower():
            prompt = prompt + "\n\nYou MUST give your answer after **Final Idea:**"
            if fallback_prompt:
                fallback_prompt = fallback_prompt + "\n\nYou MUST give your answer after **Final Idea:**"
        
        # 首先尝试使用主要提示
        response = self.completion(prompt)
        
        # 处理响应格式
        if isinstance(response, tuple):
            idea = response[0]
            full_response = response[1]
            
            print('='*40)
            print('idea len: ',len(idea))
            print('full_response len: ',len(full_response))
            print('='*40)
            
            if len(idea) < 10:
                raise ValueError("生成的想法太短，可能是模型拒绝了请求")
        else:
            # 对于特殊格式的模型，提取最终想法
            if "qwq-32b-preview" in self.model_name.lower():
                idea = extract_final_idea(response)
                full_response = response
            else:
                idea = response
                full_response = response
        
        # 检查初始响应是否被拒绝
        first_was_rejected = is_response_rejected(idea)
        first_reject_response = full_response if first_was_rejected else None
        
        # 如果被拒绝且提供了回退提示，则重试
        if first_was_rejected and fallback_prompt:
            logger.info(f"模型 {self.model_name} 拒绝了请求，尝试使用回退提示")
            
            # 使用回退提示重试
            fallback_response = self.completion(fallback_prompt)
            
            # 处理回退响应
            if isinstance(fallback_response, tuple):
                fallback_idea = fallback_response[0]
                fallback_full = fallback_response[1]
            else:
                # 对于特殊格式的模型，提取最终想法
                if "qwq-32b-preview" in self.model_name.lower():
                    fallback_idea = extract_final_idea(fallback_response)
                    fallback_full = fallback_response
                else:
                    fallback_idea = fallback_response
                    fallback_full = fallback_response
            
            # 返回回退结果
            return {
                "idea": fallback_idea,
                "full_response": fallback_full,
                "first_was_rejected": True,
                "first_reject_response": first_reject_response,
                "used_fallback": True
            }
        
        # 返回原始结果
        return {
            "idea": idea,
            "full_response": full_response,
            "first_was_rejected": first_was_rejected,
            "first_reject_response": first_reject_response,
            "used_fallback": False
        }


class CriticLLM(BaseLLM):
    """处理科学想法评价的专用LLM类"""
    
    def critique_idea(self, idea: str, critic_prompt: Optional[str] = None, prompt: Optional[str] = None) -> str:
        """评价科学想法

        Args:
            idea: 要评价的想法
            critic_prompt: 可选的系统提示
            prompt: 可选的完整提示，如果提供，将忽略critic_prompt

        Returns:
            评价结果
        """
        if prompt is None:
            prompt = "Please evaluate the following scientific idea:\n\n" + idea
            raw_response = self.completion(prompt, system_prompt=critic_prompt)
        else:
            raw_response = self.completion(prompt)
            
        return raw_response

def get_values_from_dict(text):
    """
    从文本中提取JSON格式的评分并返回originality, feasibility, clarity的值列表
    
    参数:
        text (str): 包含评分的文本
    
    返回:
        list: [originality, feasibility, clarity]的值列表
    """
    # 尝试方法1: 提取```json和```之间的内容
    json_pattern1 = r'```json(.*?)```'
    match = re.search(json_pattern1, text, re.DOTALL)
    
    if not match:
        # 尝试方法2: 提取```和```之间的内容
        json_pattern2 = r'```(.*?)```'
        match = re.search(json_pattern2, text, re.DOTALL)
    
    if match:
        # 提取JSON字符串
        json_str = match.group(1).strip()
        
        try:
            # 解析JSON
            rating_dict = json.loads(json_str)
            
            # 按指定顺序返回评分值
            return [
                rating_dict.get("originality", None),
                rating_dict.get("feasibility", None),
                rating_dict.get("clarity", None)
            ]
        except json.JSONDecodeError:
            # JSON解析失败，继续尝试下一种方法
            pass
    
    # 尝试方法3: 直接匹配花括号{}中的内容
    braces_pattern = r'\{(.*?)\}'
    match = re.search(braces_pattern, text, re.DOTALL)
    
    if match:
        json_str = '{' + match.group(1) + '}'
        
        try:
            # 解析JSON
            rating_dict = json.loads(json_str)
            
            # 按指定顺序返回评分值
            return [
                rating_dict.get("originality", None),
                rating_dict.get("feasibility", None),
                rating_dict.get("clarity", None)
            ]
        except json.JSONDecodeError:
            # JSON解析失败，继续尝试下一种方法
            pass
    
    # 尝试方法4: 直接匹配关键词后面的整数
    results = []
    for key in ["originality", "feasibility", "clarity"]:
        pattern = r'"{0}"\s*:\s*(\d+)'.format(key)
        match = re.search(pattern, text)
        if match:
            results.append(int(match.group(1)))
        else:
            # 尝试没有引号的版本
            pattern = r'{0}\s*:\s*(\d+)'.format(key)
            match = re.search(pattern, text)
            if match:
                results.append(int(match.group(1)))
            else:
                results.append(None)
    
    # 如果找到了任何评分，返回结果
    if any(result is not None for result in results):
        return results
    
    # 所有方法都失败
    return [None, None, None]



def parse_critique(raw_text: str) -> Dict[str, Union[str, Dict]]:
    """解析评价文本，提取推理和分数

    Args:
        raw_text: 原始评价文本

    Returns:
        包含解析后的推理和分数的字典
    """
    try:
        # 使用get_values_from_dict函数解析JSON格式评分
        scores_values = get_values_from_dict(raw_text)
        
        # 检查评分解析是否成功（任何一个维度为None都视为失败）
        is_valid = all(score is not None for score in scores_values)
        
        result: Dict[str, Union[str, Dict]] = {
            "is_valid": is_valid
        }
        
        # 如果评分有效，将其添加到结果中
        if is_valid:
            result["scores"] = {
                "originality": scores_values[0],
                "feasibility": scores_values[1],
                "clarity": scores_values[2]
            }

        return result

    except Exception as e:
        logger.error(f"解析评价时出现意外错误: {str(e)}")
        return {"is_valid": False}


def create_llm(llm_type: str, model_name: str, provider: Optional[str] = None) -> Union[IdeaLLM, CriticLLM]:
    """创建LLM实例的工厂函数

    Args:
        llm_type: LLM类型，"idea"或"critic"
        model_name: 模型名称
        provider: 可选的提供商名称

    Returns:
        创建的LLM实例

    Raises:
        ValueError: 如果llm_type不受支持
    """
    if llm_type.lower() == "idea":
        return IdeaLLM(model_name, provider)
    elif llm_type.lower() == "critic":
        return CriticLLM(model_name, provider)
    else:
        raise ValueError(f"不支持的LLM类型: {llm_type}")

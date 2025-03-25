"""
配置文件处理模块

包含API密钥管理、通用配置项和环境变量处理逻辑
"""

import os
from typing import Dict, List, Optional

# 定义支持的模型提供商
PROVIDER_NAMES = ["openrouter", "ollama", "stepfun", "gemini", "aigcbest", "default"]

# 定义评委模型（较强的模型）
CRITIC_MODELS = [
    "anthropic/claude-3.7-sonnet:thinking",
    "openai/o3-mini-high",
    "openai/gpt-4.5-preview", 
    # "openai/o1",           # 因为openai超过2模型而被排除
    # "anthropic/claude-3.7", # 因为和 thinking 共享一样的基模被排除
    "qwen/qwq-32b",
    "deepseek/deepseek-r1",
    "google/gemini-2.0-flash-thinking-exp:free",
    "google/gemini-2.0-pro-exp-02-05:free",
    "qwen/qwen-max",
    "deepseek/deepseek-chat", # (v3)
    # "google/gemini-2.0-flash-exp:free", # 因为google超过2模型被排除
    "anthropic/claude-3.5-sonnet"
]

IDEA_MODELS = [
    "google/gemini-2.0-flash-exp:free",
    "anthropic/claude-3.7-sonnet",
    "openai/o1",
    "openai/o3-mini",
    "openai/o1-mini",
    "step-2-16k-202411",
    "openai/gpt-4o-2024-11-20",
    "deepseek/deepseek-r1-distill-llama-70b",
    "google/gemini-pro-1.5",
    "x-ai/grok-2-1212",
    
    "google/gemini-2.0-flash-lite-001",
    "sammcj/qwen2.5-dracarys2-72b:Q4_K_M",
    "meta-llama/llama-3.1-405b-instruct",
    "qwen/qwen-2.5-72b-instruct",
    "openai/gpt-4-turbo",
    "meta-llama/llama-3.3-70b-instruct",
    "anthropic/claude-3-opus",
    "mistralai/mistral-large-2411",
    "qwen/qwen-2.5-coder-32b-instruct",
    "deepseek/deepseek-r1-distill-qwen-32b",
    
    "meta-llama/llama-3.1-70b-instruct",
    "amazon/nova-pro-v1",
    "anthropic/claude-3.5-haiku-20241022",
    "mistralai/mistral-small-24b-instruct-2501",
    "microsoft/phi-4",
    "openai/gpt-4o-mini",
    "qwen/qwq-32b-preview",
    "amazon/nova-lite-v1",
    "qwen/qwen-2.5-7b-instruct",
    "mistralai/mistral-small", # (v2409)
    
    "google/gemma-2-27b-it",
    
] + CRITIC_MODELS

class Config:
    """配置类，负责加载和管理API密钥及其他配置项"""

    def __init__(self):
        # 从环境变量或apikey文件加载密钥
        self.api_keys = self._load_api_keys()
        
        # 默认值
        self.default_provider = "openrouter"
        
        # 模型名称到提供商的映射
        self.model_provider_mapping = {
            # Ollama模型映射
            "dracarys": "ollama",
            "6cf/": "ollama",
            # Gemini模型默认使用aigcbest
            "gemini": "aigcbest",
            "step-2-16k-202411": "stepfun"
            # 其他映射可在运行时动态添加
        }

    def _load_api_keys(self) -> Dict[str, str]:
        """从apikey文件和环境变量中加载API密钥"""
        # 初始化默认值
        keys = {
            "openrouter": None,
            "stepfun": None,
            "gemini": [],
            "aigcbest": None,
        }
        
        # 从环境变量加载
        if os.getenv("OPENROUTER_API_KEY"):
            keys["openrouter"] = os.getenv("OPENROUTER_API_KEY")
        if os.getenv("STEP_API_KEY"):
            keys["stepfun"] = os.getenv("STEP_API_KEY")
        if os.getenv("GEMINI_API_KEYS"):
            # 假设环境变量是逗号分隔的
            keys["gemini"] = os.getenv("GEMINI_API_KEYS").split(",")
        if os.getenv("AIGCBEST_API_KEY"):
            keys["aigcbest"] = os.getenv("AIGCBEST_API_KEY")
            
        # 从apikey文件加载
        try:
            with open("apikey", "r") as f:
                content = f.read().strip()
                if not keys["openrouter"]:
                    keys["openrouter"] = content
        except (FileNotFoundError, IOError):
            pass
            
        # 确保至少有一个有效的API密钥
        if not keys["openrouter"]:
            raise ValueError("未找到有效的OpenRouter API密钥，请在apikey文件或环境变量中提供")
            
        return keys
        
    def get_api_key(self, provider: str) -> str:
        """获取指定提供商的API密钥"""
        if provider == "gemini":
            # 对于Gemini，随机选择一个密钥
            import random
            if not self.api_keys["gemini"]:
                raise ValueError("未找到有效的Gemini API密钥")
            return random.choice(self.api_keys["gemini"])
        return self.api_keys.get(provider)
        
    def get_provider_for_model(self, model_name: str) -> str:
        """根据模型名称确定应使用的提供商"""
        # 检查是否有直接匹配
        for prefix, provider in self.model_provider_mapping.items():
            if prefix in model_name.lower():
                return provider
                
        # 没有匹配项，返回默认提供商
        return self.default_provider
        
    def set_default_provider(self, provider: str) -> None:
        """设置默认提供商"""
        if provider not in PROVIDER_NAMES:
            raise ValueError(f"不支持的提供商: {provider}。支持的提供商有: {', '.join(PROVIDER_NAMES)}")
        self.default_provider = provider


# 创建全局配置实例
config = Config()

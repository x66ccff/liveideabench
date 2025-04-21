"""
Configuration File Processing Module

Includes API key management, general configuration items, and environment variable processing logic
"""

import os
from typing import Dict, List, Optional

# Define the supported model providers
PROVIDER_NAMES = ["openrouter", "ollama", "stepfun", "gemini", "aigcbest", "default"]

# Define the critic models (stronger models)
CRITIC_MODELS = [
    "anthropic/claude-3.7-sonnet:thinking",
    "openai/o3-mini-high",
    "openai/gpt-4.5-preview", 
    # "openai/o1",           # Excluded due to openai exceeding 2 models
    # "anthropic/claude-3.7", # Excluded as it shares the same base model as 'thinking'
    "qwen/qwq-32b",
    "deepseek/deepseek-r1",
    "google/gemini-2.0-flash-thinking-exp:free",
    "google/gemini-2.0-pro-exp-02-05:free",
    "qwen/qwen-max",
    "deepseek/deepseek-chat", # (v3)
    # "google/gemini-2.0-flash-exp:free", # Excluded due to google exceeding 2 models
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
    """Configuration class responsible for loading and managing API keys and other configuration items"""

    def __init__(self):
        # Load keys from environment variables or apikey file
        self.api_keys = self._load_api_keys()
        
        # Default values
        self.default_provider = "openrouter"
        
        # Mapping from model names to providers
        self.model_provider_mapping = {
            # Mapping for Ollama models
            "dracarys": "ollama",
            "6cf/": "ollama",
            # Gemini models use aigcbest by default
            "gemini": "aigcbest",
            "step-2-16k-202411": "stepfun"
            # Other mappings can be dynamically added at runtime
        }

    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from apikey file and environment variables"""
        # Initialize default values
        keys = {
            "openrouter": None,
            "stepfun": None,
            "gemini": [],
            "aigcbest": None,
        }
        
        # Load from environment variables
        if os.getenv("OPENROUTER_API_KEY"):
            keys["openrouter"] = os.getenv("OPENROUTER_API_KEY")
        if os.getenv("STEP_API_KEY"):
            keys["stepfun"] = os.getenv("STEP_API_KEY")
        if os.getenv("GEMINI_API_KEYS"):
            # Assume the environment variable is comma-separated
            keys["gemini"] = os.getenv("GEMINI_API_KEYS").split(",")
        if os.getenv("AIGCBEST_API_KEY"):
            keys["aigcbest"] = os.getenv("AIGCBEST_API_KEY")
            
        # Load from apikey file
        try:
            with open("apikey", "r") as f:
                content = f.read().strip()
                if not keys["openrouter"]:
                    keys["openrouter"] = content
        except (FileNotFoundError, IOError):
            pass
            
        # Ensure at least one valid API key is found
        if not keys["openrouter"]:
            raise ValueError("No valid OpenRouter API key found, please provide one in the apikey file or environment variable")
            
        return keys
        
    def get_api_key(self, provider: str) -> str:
        """Get the API key for the specified provider"""
        if provider == "gemini":
            # For Gemini, randomly select one of the keys
            import random
            if not self.api_keys["gemini"]:
                raise ValueError("No valid Gemini API keys found")
            return random.choice(self.api_keys["gemini"])
        return self.api_keys.get(provider)
        
    def get_provider_for_model(self, model_name: str) -> str:
        """Determine the provider to use based on the model name"""
        # Check for direct matches
        for prefix, provider in self.model_provider_mapping.items():
            if prefix in model_name.lower():
                return provider
                
        # No match found, return the default provider
        return self.default_provider
        
    def set_default_provider(self, provider: str) -> None:
        """Set the default provider"""
        if provider not in PROVIDER_NAMES:
            raise ValueError(f"Unsupported provider: {provider}. Supported providers are: {', '.join(PROVIDER_NAMES)}")
        self.default_provider = provider


# Create a global config instance
config = Config()
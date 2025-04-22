"""
LLM Interaction and Processing Module

Responsible for communication with various LLM APIs, handling requests and responses
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Rejection phrases
REJECTION_PHRASES = [
    "I'm sorry", "I am sorry", "I apologize", "As an AI", "As a language model",
    "As an assistant", "I cannot", "I can't", "I am unable to", "I'm unable to",
    "I am not able to", "I'm not able to"
]

# Mapping dictionary from OpenRouter model names to AIGCBest model names
OPENROUTER_TO_AIGCBEST_MODEL_MAPPING = {
    # Specific model mappings will be filled in by the user here
    "google/gemini-2.0-flash-thinking-exp:free": "gemini-2.0-flash-thinking-exp",
    "google/gemini-2.0-pro-exp-02-05:free": "gemini-2.0-pro-exp-02-05",
    "google/gemini-2.0-flash-exp:free":"gemini-2.0-flash-exp",
    "google/gemini-pro-1.5":"gemini-1.5-pro-002",
    "google/gemini-2.0-flash-lite-001":"gemini-2.0-flash-lite-preview-02-05",
}


class BaseLLM:
    """Base LLM class, handling communication with LLM APIs"""

    def __init__(self, model_name: str, provider: Optional[str] = None):
        """Initialize LLM

        Args:
            model_name: Model name
            provider: Model provider, if None, it will be automatically inferred from model_name
        """
        self.model_name = model_name
        self.provider = provider if provider else config.get_provider_for_model(model_name)
        self._setup_client()

    def _setup_client(self) -> None:
        """Set up the client based on the provider"""
        if self.provider == "gemini":
            # Set up Gemini API
            genai.configure(api_key=config.get_api_key("gemini"))

        elif self.provider == "ollama":
            # Set up Ollama API
            self.client = OpenAI(
                base_url='http://localhost:11434/v1',
                api_key="ollama"
            )

        elif self.provider == "stepfun":
            # Set up StepFun API
            api_key = config.get_api_key("stepfun")
            if not api_key:
                raise ValueError("StepFun API key not found")
            self.client = OpenAI(
                base_url="https://api.stepfun.com/v1",
                api_key=api_key
            )

        elif self.provider == "aigcbest":
            # Set up AIGCBest API
            api_key = config.get_api_key("aigcbest")
            if not api_key:
                raise ValueError("AIGCBest API key not found")
            self.client = OpenAI(
                base_url="https://api2.aigcbest.top/v1",
                api_key=api_key
            )

        else:
            # Default to OpenRouter
            api_key = config.get_api_key("openrouter")
            if not api_key:
                raise ValueError("OpenRouter API key not found")
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key
            )

    def completion(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Execute non-streaming LLM completion request

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            LLM's response text
        """
        if self.provider == "gemini":
            return self._gemini_completion(prompt, system_prompt)
        else:
            return self._openai_compatible_completion(prompt, system_prompt)

    def _gemini_completion(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Execute Gemini API completion request

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            Gemini model's response text
        """

        model = genai.GenerativeModel(model_name=self.model_name)

        # Build the prompt
        if system_prompt:
            prompt = f"{system_prompt}\n\n{prompt}"

        response = model.generate_content(prompt)
        return response.text


    def _openai_compatible_completion(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Execute completion request for OpenAI-compatible API

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            Model's response text
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

        # If the provider is AIGCBest, the model name needs to be converted to an AIGCBest supported name
        model_name = self.model_name
        if self.provider == "aigcbest" and model_name in OPENROUTER_TO_AIGCBEST_MODEL_MAPPING:
            model_name = OPENROUTER_TO_AIGCBEST_MODEL_MAPPING[model_name]
            logger.info(f"AIGCBest API: Converting model name from {self.model_name} to {model_name}")

        # Implement exponential backoff retry
        max_retries = 10  # Maximum number of retries
        max_wait_time = 300  # Maximum wait time (5 minutes = 300 seconds)
        base_wait_time = 1  # Initial wait time (seconds)

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

                # Check if it is a 500 error
                if hasattr(response, 'error') and isinstance(response.error, dict) and response.error.get('code') == 500:
                    error_msg = response.error.get('message', 'Internal Server Error')
                    logger.warning(f"Encountered 500 error: {error_msg}, preparing to retry ({retry_count + 1}/{max_retries})")
                    raise Exception(f"Internal Server Error: {error_msg}")

                # Check if the response is valid
                if response and hasattr(response, 'choices') and response.choices and len(response.choices) > 0:
                    # Check if there is a reasoning field
                    if hasattr(response.choices[0].message, 'reasoning') and response.choices[0].message.reasoning:
                        return (response.choices[0].message.content, response.choices[0].message.reasoning)
                    else:
                        return response.choices[0].message.content
                else:
                    # Check if it is a 500 error object
                    if response is None or (hasattr(response, 'id') and response.id is None and hasattr(response, 'error') and
                       isinstance(response.error, dict) and response.error.get('code') == 500):
                        error_msg = response.error.get('message', 'Internal Server Error') if hasattr(response, 'error') else "Internal Server Error"
                        logger.warning(f"Encountered 500 error: {error_msg}, preparing to retry ({retry_count + 1}/{max_retries})")
                        raise Exception(f"Internal Server Error: {error_msg}")

                    # Return the error message as the response to prevent returning None
                    error_msg = "API response is invalid or incomplete"
                    logger.error(f"API response error: {error_msg}")
                    print("response", response)

                    from pprint import pprint
                    pprint(response)

                    raise ValueError(error_msg)

            except Exception as e:
                # Check if it is an internal server error
                error_str = str(e).lower()
                is_server_error = "500" in error_str or "internal server error" in error_str

                # If it's not a server error or the maximum retry count has been reached, raise the exception
                if not is_server_error or retry_count >= max_retries:
                    raise

                # Calculate wait time (exponential backoff + random jitter)
                wait_time = min(base_wait_time * (2 ** retry_count) + random.uniform(0, 1), max_wait_time)

                # If the wait time exceeds the maximum wait time, raise an exception
                if wait_time >= max_wait_time:
                    logger.error(f"Reached maximum wait time ({max_wait_time} seconds), stopping retries.")
                    raise ValueError(f"Failed after multiple retries: {e}")

                logger.warning(f"Retrying {retry_count + 1}/{max_retries}, waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                retry_count += 1
                continue




def extract_final_idea(text: str) -> str:
    """Extract the final idea from text marked with "Final Idea"

    Args:
        text: Input text

    Returns:
        The extracted final idea or the original text
    """
    # Define all possible markers, from most strict to least strict
    markers = [
        "**Final Idea:**",
        "**Final Idea**:",
        "**final idea:**",
        "**final idea**:",
        "final idea"
    ]

    # Normalize the input text, remove leading/trailing whitespace
    text = text.strip()

    # Try each marker sequentially
    for marker in markers:
        # Perform a case-insensitive search for the current marker
        marker_lower = marker.lower()
        text_lower = text.lower()

        if marker_lower in text_lower:
            # Find the actual position of the marker in the original text
            start_idx = text_lower.index(marker_lower)
            # Extract the content after the marker from the original text
            final_idea = text[start_idx + len(marker):].strip()
            return final_idea

    # If no markers are found, return the full text
    return text


def is_response_rejected(text: str) -> bool:
    """Check if the response contains rejection phrases

    Args:
        text: The text to check

    Returns:
        Returns True if the text contains rejection phrases; otherwise returns False
    """
    return any(phrase.lower() in text.lower() for phrase in REJECTION_PHRASES)


class IdeaLLM(BaseLLM):
    """Specialized LLM class for handling scientific idea generation"""

    def generate_idea(self, prompt: str, fallback_prompt: Optional[str] = None) -> Dict[str, str]:
        """Generate scientific idea

        Args:
            prompt: Main prompt
            fallback_prompt: Optional fallback prompt, used when the model rejects

        Returns:
            Dictionary containing the idea and the full response
        """
        if "qwq-32b-preview" in self.model_name.lower():
            prompt = prompt + "\n\nYou MUST give your answer after **Final Idea:**"
            if fallback_prompt:
                fallback_prompt = fallback_prompt + "\n\nYou MUST give your answer after **Final Idea:**"

        # First, try using the main prompt
        response = self.completion(prompt)

        # Handle response format
        if isinstance(response, tuple):
            idea = response[0]
            full_response = response[1]

            print('='*40)
            print('idea len: ',len(idea))
            print('full_response len: ',len(full_response))
            print('='*40)

            if len(idea) < 10:
                raise ValueError("Generated idea is too short, the model might have rejected the request")
        else:
            # For models with special formatting, extract the final idea
            if "qwq-32b-preview" in self.model_name.lower():
                idea = extract_final_idea(response)
                full_response = response
            else:
                idea = response
                full_response = response

        # Check if the initial response was rejected
        first_was_rejected = is_response_rejected(idea)
        first_reject_response = full_response if first_was_rejected else None

        # If rejected and a fallback prompt is provided, retry
        if first_was_rejected and fallback_prompt:
            logger.info(f"Model {self.model_name} rejected the request, trying fallback prompt")

            # Retry using the fallback prompt
            fallback_response = self.completion(fallback_prompt)

            # Handle fallback response
            if isinstance(fallback_response, tuple):
                fallback_idea = fallback_response[0]
                fallback_full = fallback_response[1]
            else:
                # For models with special formatting, extract the final idea
                if "qwq-32b-preview" in self.model_name.lower():
                    fallback_idea = extract_final_idea(fallback_response)
                    fallback_full = fallback_response
                else:
                    fallback_idea = fallback_response
                    fallback_full = fallback_response

            # Return fallback result
            return {
                "idea": fallback_idea,
                "full_response": fallback_full,
                "first_was_rejected": True,
                "first_reject_response": first_reject_response,
                "used_fallback": True
            }

        # Return original result
        return {
            "idea": idea,
            "full_response": full_response,
            "first_was_rejected": first_was_rejected,
            "first_reject_response": first_reject_response,
            "used_fallback": False
        }


class CriticLLM(BaseLLM):
    """Specialized LLM class for handling scientific idea critique"""

    def critique_idea(self, idea: str, critic_prompt: Optional[str] = None, prompt: Optional[str] = None) -> str:
        """Critique scientific idea

        Args:
            idea: The idea to critique
            critic_prompt: Optional system prompt
            prompt: Optional full prompt, if provided, critic_prompt will be ignored

        Returns:
            Critique result
        """
        if prompt is None:
            prompt = "Please evaluate the following scientific idea:\n\n" + idea
            raw_response = self.completion(prompt, system_prompt=critic_prompt)
        else:
            raw_response = self.completion(prompt)

        return raw_response

def get_values_from_dict(text):
    """
    Extract JSON formatted scores from text and return a list of values for originality, feasibility, clarity

    Args:
        text (str): Text containing the scores

    Returns:
        list: List of values [originality, feasibility, clarity]
    """
    # Try method 1: Extract content between ```json and ```
    json_pattern1 = r'```json(.*?)```'
    match = re.search(json_pattern1, text, re.DOTALL)

    if not match:
        # Try method 2: Extract content between ``` and ```
        json_pattern2 = r'```(.*?)```'
        match = re.search(json_pattern2, text, re.DOTALL)

    if match:
        # Extract JSON string
        json_str = match.group(1).strip()

        try:
            # Parse JSON
            rating_dict = json.loads(json_str)

            # Return score values in the specified order
            return [
                rating_dict.get("originality", None),
                rating_dict.get("feasibility", None),
                rating_dict.get("clarity", None)
            ]
        except json.JSONDecodeError:
            # JSON parsing failed, continue to the next method
            pass

    # Try method 3: Directly match content within curly braces {}
    braces_pattern = r'\{(.*?)\}'
    match = re.search(braces_pattern, text, re.DOTALL)

    if match:
        json_str = '{' + match.group(1) + '}'

        try:
            # Parse JSON
            rating_dict = json.loads(json_str)

            # Return score values in the specified order
            return [
                rating_dict.get("originality", None),
                rating_dict.get("feasibility", None),
                rating_dict.get("clarity", None)
            ]
        except json.JSONDecodeError:
            # JSON parsing failed, continue to the next method
            pass

    # Try method 4: Directly match integers after keywords
    results = []
    for key in ["originality", "feasibility", "clarity"]:
        pattern = r'"{0}"\s*:\s*(\d+)'.format(key)
        match = re.search(pattern, text)
        if match:
            results.append(int(match.group(1)))
        else:
            # Try the version without quotes
            pattern = r'{0}\s*:\s*(\d+)'.format(key)
            match = re.search(pattern, text)
            if match:
                results.append(int(match.group(1)))
            else:
                results.append(None)

    # If any scores were found, return the results
    if any(result is not None for result in results):
        return results

    # All methods failed
    return [None, None, None]



def parse_critique(raw_text: str) -> Dict[str, Union[str, Dict]]:
    """Parse critique text, extract reasoning and scores

    Args:
        raw_text: Raw critique text

    Returns:
        Dictionary containing the parsed reasoning and scores
    """
    try:
        # Use the get_values_from_dict function to parse JSON formatted scores
        scores_values = get_values_from_dict(raw_text)

        # Check if score parsing was successful (any dimension being None is considered a failure)
        is_valid = all(score is not None for score in scores_values)

        result: Dict[str, Union[str, Dict]] = {
            "is_valid": is_valid
        }

        # If the scores are valid, add them to the result
        if is_valid:
            result["scores"] = {
                "originality": scores_values[0],
                "feasibility": scores_values[1],
                "clarity": scores_values[2]
            }

        return result

    except Exception as e:
        logger.error(f"Unexpected error occurred while parsing critique: {str(e)}")
        return {"is_valid": False}


def create_llm(llm_type: str, model_name: str, provider: Optional[str] = None) -> Union[IdeaLLM, CriticLLM]:
    """Factory function to create LLM instances

    Args:
        llm_type: LLM type, "idea" or "critic"
        model_name: Model name
        provider: Optional provider name

    Returns:
        The created LLM instance

    Raises:
        ValueError: If llm_type is unsupported
    """
    if llm_type.lower() == "idea":
        return IdeaLLM(model_name, provider)
    elif llm_type.lower() == "critic":
        return CriticLLM(model_name, provider)
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}")
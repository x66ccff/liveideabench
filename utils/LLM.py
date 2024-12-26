import os
import json
from typing import Dict, Optional, Union
import re
import random
from openai import OpenAI
import google.generativeai as genai

class BaseLLM:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        self.STEP_API_KEY = os.getenv("STEP_API_KEY")
        self.GEMINI_API_KEYS = [
            ""
        ]

        random.shuffle(self.GEMINI_API_KEYS)
        
        if not self.OPENROUTER_API_KEY:
            self.OPENROUTER_API_KEY = ""
        if not self.STEP_API_KEY:
            self.STEP_API_KEY = ""

        if "gemini-2.0-flash-exp" in model_name.lower():
            self.use_gemini = True
            self.current_gemini_key_index = 0
            self._setup_gemini()
        else:
            self.use_gemini = False
            if "step" in model_name.lower():
                self.base_url = "https://api.stepfun.com/v1"
                self.api_key = self.STEP_API_KEY
            elif "dracarys" in model_name.lower():
                self.base_url = 'http://localhost:11434/v1'
                self.api_key = "ollama"
            else:
                self.base_url = "https://openrouter.ai/api/v1"
                self.api_key = self.OPENROUTER_API_KEY
            
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )

    def _setup_gemini(self):
        try:
            api_key = self.GEMINI_API_KEYS[self.current_gemini_key_index]
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel("gemini-2.0-flash-exp")
            return
        except Exception as e:
            print(f"Failed to setup Gemini with key index {self.current_gemini_key_index}: {str(e)}")
            self.current_gemini_key_index += 1
            if self.current_gemini_key_index < len(self.GEMINI_API_KEYS):
                return self._setup_gemini()
            else:
                print("All Gemini API keys failed, falling back to OpenRouter")
                self.use_gemini = False
                self.base_url = "https://openrouter.ai/api/v1"
                self.api_key = self.OPENROUTER_API_KEY
                self.client = OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key
                )

    def _stream_completion(self, prompt: str, system_prompt: str = None) -> str:
        if hasattr(self, 'api_key') and self.api_key == "ollama":
            client = self.client
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

            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
            all_text = response.choices[0].message.content
            return all_text

        else:
            if self.use_gemini:
                try:
                    messages = []
                    if system_prompt:
                        messages.append({'role': 'system', 'parts': [system_prompt]})
                    messages.append({'role': 'user', 'parts': [prompt]})
                    
                    response = self.gemini_model.generate_content(messages)
                    return response.text
                except Exception as e:
                    print(f"Gemini API error: {str(e)}")
                    self.current_gemini_key_index += 1
                    if self.current_gemini_key_index < len(self.GEMINI_API_KEYS):
                        self._setup_gemini()
                        return self._stream_completion(prompt, system_prompt)
                    else:
                        print("All Gemini API keys failed, falling back to OpenRouter")
                        self.use_gemini = False
                        self.base_url = "https://openrouter.ai/api/v1"
                        self.api_key = self.OPENROUTER_API_KEY
                        self.client = OpenAI(
                            base_url=self.base_url,
                            api_key=self.api_key
                        )
            
            try:
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
                
                stream = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    stream=True
                )
                
                all_text = ""
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    content = chunk.choices[0].delta.content or ""
                    all_text += content
                    
                return all_text
                
            except Exception as e:
                raise Exception(f"Error in streaming completion: {str(e)}")



def extract_final_idea(text):
    markers = [
        "**Final Idea:**",
        "**Final Idea**:",
        "**final idea:**",
        "**final idea**:",
        "final idea"
    ]
    
    text = text.strip()
    
    for marker in markers:
        marker_lower = marker.lower()
        text_lower = text.lower()
        
        if marker_lower in text_lower:
            start_idx = text_lower.index(marker_lower)
            final_idea = text[start_idx + len(marker):].strip()
            return final_idea
            
    return text


class IdeaLLM(BaseLLM):
    def generate_idea(self, idea_prompt: str) -> Union[str, Dict[str, str]]:
        if "qwq" in self.model_name.lower():
            prompt = idea_prompt + "\nYou MUST give your answer after **Final Idea:**"
            full_response = self._stream_completion(prompt)
            
            return {
                "idea": extract_final_idea(full_response),
                "full_response": full_response
            }
        else:
            idea = self._stream_completion(idea_prompt)
            return {
                "idea": idea,
                "full_response": idea
            }

class CriticLLM(BaseLLM):
    def critique_idea(self, idea: str, critic_prompt=None, prompt=None) -> Dict[str, Union[str, Dict]]:
        if prompt is None:
            prompt = "Please evaluate the following scientific idea and give your scores directly:\n\n" + idea
            raw_response = self._stream_completion(system_prompt=critic_prompt, prompt=prompt)
        else:
            critic_prompt = None
            raw_response = self._stream_completion(system_prompt=critic_prompt, prompt=prompt)
        return raw_response
    
def create_llm(llm_type: str, model_name: str) -> Union[IdeaLLM, CriticLLM]:
    """Factory function to create LLM instances"""
    if llm_type.lower() == "idea":
        return IdeaLLM(model_name)
    elif llm_type.lower() == "critic":
        return CriticLLM(model_name)
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")
    

def parse_critique(raw_text: str) -> Dict[str, Union[str, Dict]]:
    """Parses critique text containing REASONING and SCORES.

    Args:
        raw_text: The raw text containing the critique.

    Returns:
        A dictionary containing the parsed REASONING and SCORES, or an empty dictionary if parsing fails.
        Handles potential errors like invalid JSON or missing sections.
    """
    try:
        # Use regex to find the REASONING and SCORES blocks. This is more robust to variations in formatting.
        reasoning_match = re.search(r"REASONING\s*=\s*({.+?})", raw_text, re.DOTALL)
        scores_match = re.search(r"SCORES\s*=\s*({.+?})", raw_text, re.DOTALL)

        result: Dict[str, Union[str, Dict]] = {}

        if reasoning_match:
            reasoning_str = reasoning_match.group(1)
            try:
                # Attempt to parse the reasoning string as JSON.
                reasoning = json.loads(reasoning_str)
                result["reasoning"] = reasoning
            except json.JSONDecodeError as e:
                print(f"Error decoding REASONING JSON: {e}")
                # Attempt to clean up the JSON string before parsing, handling common errors.
                cleaned_reasoning_str = reasoning_str.replace("'", "\"").replace('\n', '') # replace single quotes with double quotes and remove newlines
                try:
                    reasoning = json.loads(cleaned_reasoning_str)
                    result["reasoning"] = reasoning
                except json.JSONDecodeError as e2:
                    print(f"Error decoding cleaned REASONING JSON: {e2}")
                    result["reasoning"] = "Error parsing REASONING"

        if scores_match:
            scores_str = scores_match.group(1)
            try:
                scores = json.loads(scores_str)
                result["scores"] = scores
            except json.JSONDecodeError as e:
                print(f"Error decoding SCORES JSON: {e}")
                cleaned_scores_str = scores_str.replace("'", "\"").replace('\n', '')
                try:
                    scores = json.loads(cleaned_scores_str)
                    result["scores"] = scores
                except json.JSONDecodeError as e2:
                    print(f"Error decoding cleaned SCORES JSON: {e2}")
                    result["scores"] = "Error parsing SCORES"

        return result

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {}
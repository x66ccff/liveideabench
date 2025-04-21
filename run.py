#!/usr/bin/env python3
"""
LiveIdeaBench Main Program

Used for running benchmark tests for scientific idea generation and evaluation
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

# Custom modules
from config import config
from utils.LLM import create_llm, parse_critique, is_response_rejected
from utils.database import save_result, check_duplicate_entries, close_all_connections
from config import CRITIC_MODELS, IDEA_MODELS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'./logs/run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure directories exist
os.makedirs('./logs', exist_ok=True)
os.makedirs('./data', exist_ok=True)

    


def load_keywords(file_path: str = './keywords_data/keywordsEverywhere20241216.xlsx') -> List[str]:
    """Load keyword data

    Args:
        file_path: Path to the Excel file

    Returns:
        List of keywords
    """
    try:
        wb = load_workbook(file_path)
        sheet = wb['Sheet1']
        
        data = sheet.values
        columns = next(data)
        df = pd.DataFrame(data, columns=columns)
        df = df.drop(0).reset_index(drop=True)
        df.columns = ['Index', 'Keyword', 'Search_Volume', 'CPC', 'Competition', 'Trend']
        
        # Clean data
        df = df.replace('None', pd.NA)
        df = df.drop(columns=['Index', 'Search_Volume', 'CPC', 'Competition', 'Trend'])
        df = df.astype(str)
        
        return df['Keyword'].tolist()
    except Exception as e:
        logger.error(f"Failed to load keyword file: {str(e)}")
        raise


def load_prompts(file_path: str = './utils/prompts.json') -> Dict[str, Dict[str, str]]:
    """Load prompt templates

    Args:
        file_path: Path to the JSON prompt file

    Returns:
        Dictionary of prompt templates
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        logger.error(f"Failed to load prompt file: {str(e)}")
        raise


def clean_text(text: str) -> str:
    """Clean text by removing unnecessary whitespace

    Args:
        text: Raw text

    Returns:
        Cleaned text
    """
    # Remove newlines, carriage returns, and tabs
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    # Replace multiple spaces with a single space
    return ' '.join(text.split())


def run_evaluation(keyword: str, idea_model: str, critic_models: List[str], 
                  prompts: Dict[str, Dict[str, str]], provider: Optional[str] = None) -> None:
    """Run a single evaluation

    Args:
        keyword: Keyword
        idea_model: Idea model name
        critic_models: List of critic model names
        prompts: Prompt templates
        provider: Optional model provider
    """
    # Prepare idea prompt
    idea_prompt = prompts['idea_prompt']['description'].replace('{{keywords}}', str(keyword))
    idea_fallback_prompt = prompts['idea_prompt']['fallback_description'].replace('{{keywords}}', str(keyword))
    
    # Create idea LLM and generate idea
    logger.info(f"Using model {idea_model} to generate idea for '{keyword}'")
    

    idea_llm = create_llm("idea", idea_model, provider)
    generation_result = idea_llm.generate_idea(idea_prompt, fallback_prompt=idea_fallback_prompt)
    
    idea = generation_result["idea"]
    full_response = generation_result["full_response"]
    first_was_rejected = generation_result["first_was_rejected"]
    first_reject_response = generation_result.get("first_reject_response")
    used_fallback = generation_result.get("used_fallback", False)
    
    if first_was_rejected:
        logger.warning(f"Model {idea_model} rejected the request" + (" (used fallback prompt)" if used_fallback else ""))
    
    if len(idea) < 2:
        raise ValueError(f"Generated idea is too short: {len(idea)} characters")
    
    logger.info(f"Idea generation complete, length: {len(idea)} characters")
    
    # Evaluate the idea with each critic model
    for critic_model in critic_models:
        logger.info(f"Using critic {critic_model} to evaluate the idea")
        
        critic_prompt = prompts['critic_prompt']['description'].replace('{{keywords}}', str(keyword))
        
        critic_llm = create_llm("critic", critic_model, provider)
        critique = critic_llm.critique_idea(idea, critic_prompt=critic_prompt)
        
        # Handle the case where critique may be a tuple (when the model returns a reasoning field)
        critique_reasoning = None
        if isinstance(critique, tuple) and len(critique) == 2:
            critique_reasoning = critique[1]  # Save the model's reasoning process
            critique = critique[0]  # Extract the first element as the critique content
        
        # Add retry logic, up to 3 attempts
        max_retries = 3
        retry_count = 0
        parsed_result = None
        error_msg = None
        
        while retry_count < max_retries:
            parsed_result = parse_critique(critique)
            
            # Check if the parsing result is valid
            if parsed_result.get('is_valid', False) and parsed_result.get('scores'):
                break  # Parsing successful, exit the loop
            
            retry_count += 1
            logger.warning(f"Critique parsing failed (attempt {retry_count}/{max_retries}): Unable to get valid scores")
            
            if retry_count < max_retries:
                # Retry the evaluation
                logger.info(f"Retrying evaluation...")
                critique = critic_llm.critique_idea(idea, critic_prompt=critic_prompt)
                
                # Handle the case where the new critique may be a tuple
                if isinstance(critique, tuple) and len(critique) == 2:
                    critique_reasoning = critique[1]
                    critique = critique[0]
        
        # If all attempts failed, record the error message
        if retry_count == max_retries and not (parsed_result.get('is_valid', False) and parsed_result.get('scores')):
            error_msg = f"Critique parsing failed: Unable to get valid scores after 3 attempts"
            logger.error(error_msg)
        
        # Save the result
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
        
        # Save the result only if the evaluation is valid or there is a clear error record
        save_result(result_data)
        
        if error_msg:
            raise ValueError(error_msg)
        else:
            logger.info(f"Evaluation result saved")
            


def continue_from_last_run() -> None:
    """Continue from the last run, find the combination with the least number of evaluations and run it"""
    from utils.database import query_results
    
    logger.info("Continuing from the last run, finding combinations that need more evaluations...")
    
    # Get the statistics of all the results
    results = query_results()
    
    # If there are no results, cannot continue
    if not results:
        logger.info("No records in the database, cannot continue from the last run")
        return
    
    # Count the number of evaluations for each keyword-model combination
    combo_counts = {}
    used_critics = {}
    
    for result in results:
        key = (result['keywords'], result['idea_model'])
        
        if key not in combo_counts:
            combo_counts[key] = 0
            used_critics[key] = set()
            
        combo_counts[key] += 1
        used_critics[key].add(result['critic_model'])
    
    # Find the combinations with the least number of evaluations
    min_count = min(combo_counts.values())
    min_combos = [key for key, count in combo_counts.items() if count == min_count]
    
    if not min_combos:
        logger.info("All combinations have the same number of evaluations, no need to add more")
        return
    
    # Randomly select one combination
    selected_combo = random.choice(min_combos)
    keyword, idea_model = selected_combo
    
    logger.info(f"Selecting to add more evaluations: Keyword '{keyword}', Model '{idea_model}', Current count: {min_count}")
    
    # Select a critic model that has not been used for this combination
    available_critics = [m for m in CRITIC_MODELS if m != idea_model and m not in used_critics[selected_combo]]
    
    if not available_critics:
        logger.info(f"All critics have been used for this combination, skipping")
        return
    
    # Randomly select an unused critic model
    critic_model = random.choice(available_critics)
    
    # Run the evaluation
    prompts = load_prompts()
    run_evaluation(keyword, idea_model, [critic_model], prompts)


def main() -> None:
    """Main function, handle command-line arguments and run the evaluations"""
    parser = argparse.ArgumentParser(description="LiveIdeaBench - LLM Scientific Idea Evaluation Benchmark")
    parser.add_argument('--idea_model', type=str, help='Idea generation model name')
    parser.add_argument('--start_from_last_run', action='store_true', help='Continue from the last run')
    parser.add_argument('--provider', type=str, choices=['openrouter', 'gemini', 'stepfun', 'ollama'],
                       help='Model provider, default is inferred from the model name')
    parser.add_argument('--keyword', nargs='+', help='Specify keywords for idea generation, can be a single keyword or a list of keywords; if not specified, all keywords will be used')
    
    args = parser.parse_args()
    
    # Set the default provider
    if args.provider:
        config.set_default_provider(args.provider)
        logger.info(f"Set default provider: {args.provider}")
    
    if args.start_from_last_run:
        logger.info("Continuing from the last run...")
        continue_from_last_run()
        return
    
    if not args.idea_model:
        logger.error("Must provide the idea generation model name")
        return
    
    if args.idea_model not in IDEA_MODELS:
        logger.error(f"Unsupported model: {args.idea_model}")
        return
    
    # Load keywords and prompts
    all_keywords = load_keywords()
    prompts = load_prompts()
    
    # Handle user-specified keywords
    if args.keyword:
        specified_keywords = args.keyword
        logger.info(f"Using user-specified keywords: {', '.join(specified_keywords)}")
    else:
        specified_keywords = all_keywords
        logger.info("Using all keywords")
    
    while True:
        # Randomly select a keyword (from the specified list)
        keyword = random.choice(specified_keywords)
        
        # Check if there are enough evaluations
        if check_duplicate_entries(keyword, args.idea_model):
            logger.info(f"Skipping '{keyword}' and '{args.idea_model}', enough evaluations available")
            if len(specified_keywords) == 1:
                logger.info("Only one keyword was specified, and it has enough evaluations, terminating the program")
                break
            continue
        
        # Randomly select 3 critic models, excluding the idea model itself
        available_critics = [m for m in CRITIC_MODELS if m != args.idea_model]
        num_critics = 3
        critic_models = random.sample(available_critics, min(num_critics, len(available_critics)))
        
        logger.info(f"Selected keyword: '{keyword}'")
        logger.info(f"Selected critics: {', '.join(critic_models)}")
        
        # Run the evaluation
        run_evaluation(keyword, args.idea_model, critic_models, prompts, args.provider)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Program interrupted by the user")
    # except Exception as e:
    #     logger.exception(f"Unexpected error occurred: {str(e)}")
    finally:
        # Ensure all database connections are closed
        close_all_connections()
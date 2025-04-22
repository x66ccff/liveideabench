"""
Database Management Module

Uses SQLite for data storage, supporting concurrent access from multiple processes.
"""

import os
import sqlite3
import json
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database file path
DB_PATH = './data/ideabench.db'

# Ensure the database directory exists
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# Thread lock for protecting the connection pool
_lock = threading.Lock()

# Connection pool - one connection per thread
_connection_pool = {}


def get_connection() -> sqlite3.Connection:
    """Get the database connection for the current thread"""
    thread_id = threading.get_ident()
    
    with _lock:
        if thread_id not in _connection_pool:
            conn = sqlite3.connect(DB_PATH, timeout=30.0)
            conn.row_factory = sqlite3.Row  # Allows row objects to be accessed by column name
            _connection_pool[thread_id] = conn
            return conn
        return _connection_pool[thread_id]


def close_all_connections() -> None:
    """Close all database connections"""
    with _lock:
        for conn in _connection_pool.values():
            conn.close()
        _connection_pool.clear()


def init_database() -> None:
    """Initialize the database structure"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Create the results table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        keywords TEXT NOT NULL,
        idea_model TEXT NOT NULL,
        critic_model TEXT NOT NULL,
        idea TEXT NOT NULL,
        raw_critique TEXT NOT NULL,
        parsed_scores TEXT,          -- Scores stored in JSON format
        parsed_reasoning TEXT,       -- Analysis stored in JSON format
        critique_reasoning TEXT,     -- Reasoning process of the critic model
        error TEXT,                  -- Potential error messages
        full_response TEXT NOT NULL, -- Full response
        first_was_rejected INTEGER DEFAULT 0, -- Flag indicating if the model rejected the request initially
        first_reject_response TEXT   -- Stores the reason for the initial rejection
    )
    ''')
    
    # Create indexes to speed up queries
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_keyword_model ON results (keywords, idea_model)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON results (timestamp)')
    
    conn.commit()


def save_result(result_data: Dict[str, Any]) -> int:
    """Save an evaluation result to the database

    Args:
        result_data: Dictionary containing the evaluation results

    Returns:
        The ID of the newly inserted record
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Extract and process data
    timestamp = datetime.now().isoformat()
    keywords = result_data.get('keywords', '')
    idea_model = result_data.get('idea_model', '')
    critic_model = result_data.get('critic_model', '')
    idea = result_data.get('idea', '')
    raw_critique = result_data.get('raw_critique', '')
    full_response = result_data.get('full_response', '')
    error = result_data.get('error')
    
    # Handle parsing results
    parsed_scores = None
    parsed_reasoning = None
    if 'parsed_score' in result_data and result_data['parsed_score']:
        parsed_scores = json.dumps(result_data['parsed_score'])
    if 'parsed_feedback' in result_data and result_data['parsed_feedback']:
        parsed_reasoning = json.dumps(result_data['parsed_feedback'])
    
    # Get the model's reasoning process
    critique_reasoning = result_data.get('critique_reasoning')
    
    # Get rejection status
    first_was_rejected = result_data.get('first_was_rejected', 0)
    if isinstance(first_was_rejected, bool):
        first_was_rejected = 1 if first_was_rejected else 0
    first_reject_response = result_data.get('first_reject_response')
    
    try:
        cursor.execute('''
        INSERT INTO results 
        (timestamp, keywords, idea_model, critic_model, idea, raw_critique, 
         parsed_scores, parsed_reasoning, critique_reasoning, error, full_response, first_was_rejected, first_reject_response)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp, keywords, idea_model, critic_model, idea, raw_critique,
            parsed_scores, parsed_reasoning, critique_reasoning, error, full_response, first_was_rejected, first_reject_response
        ))
        
        conn.commit()
        return cursor.lastrowid
    except sqlite3.Error as e:
        logger.error(f"Database insertion error: {str(e)}")
        conn.rollback()
        raise


def check_duplicate_entries(keyword: str, idea_model: str, limit: int = 6) -> bool:
    """Check if a sufficient number of records exist for the same keyword and model combination

    Args:
        keyword: The keyword
        idea_model: The idea model name
        limit: The maximum record count limit

    Returns:
        True if the record count meets or exceeds the limit, False otherwise
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT COUNT(*) as count FROM results 
    WHERE keywords = ? AND idea_model = ?
    ''', (keyword, idea_model))
    
    result = cursor.fetchone()
    count = result['count'] if result else 0
    
    return count >= limit


def query_results(filters: Optional[Dict[str, Any]] = None, 
                 limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Query results based on filter conditions

    Args:
        filters: Dictionary of filter conditions
        limit: Maximum number of records to return

    Returns:
        List of results
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    query = "SELECT * FROM results"
    params = []
    
    if filters:
        conditions = []
        for key, value in filters.items():
            if key in ['keywords', 'idea_model', 'critic_model', 'first_was_rejected']:
                conditions.append(f"{key} = ?")
                params.append(value)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
    
    query += " ORDER BY timestamp DESC"
    
    if limit:
        query += f" LIMIT {limit}"
    
    cursor.execute(query, params)
    
    results = []
    for row in cursor.fetchall():
        result_dict = dict(row)
        
        # Parse JSON fields
        if result_dict.get('parsed_scores'):
            try:
                result_dict['parsed_scores'] = json.loads(result_dict['parsed_scores'])
            except json.JSONDecodeError:
                pass
                
        if result_dict.get('parsed_reasoning'):
            try:
                result_dict['parsed_reasoning'] = json.loads(result_dict['parsed_reasoning'])
            except json.JSONDecodeError:
                pass
                
        results.append(result_dict)
    
    return results


def export_to_csv(output_path: str) -> None:
    """Export the database to a CSV file

    Args:
        output_path: The output path for the CSV file
    """
    import pandas as pd
    
    conn = get_connection()
    
    # Read all results
    df = pd.read_sql_query("SELECT * FROM results", conn)
    
    # Process JSON fields
    for json_col in ['parsed_scores', 'parsed_reasoning']:
        if json_col in df.columns:
            df[json_col] = df[json_col].apply(
                lambda x: json.loads(x) if x and isinstance(x, str) else x
            )
    
    # Export to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Successfully exported data to {output_path}")


def check_and_add_column() -> None:
    """Check and add new columns to the existing table"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get the current table structure
    cursor.execute("PRAGMA table_info(results)")
    columns = [column[1] for column in cursor.fetchall()]
    
    # Check if the critique_reasoning column exists
    if 'critique_reasoning' not in columns:
        logger.info("Adding critique_reasoning column to results table")
        cursor.execute("ALTER TABLE results ADD COLUMN critique_reasoning TEXT")
        conn.commit()


# Initialize the database
init_database()

# Check and update table structure
check_and_add_column()
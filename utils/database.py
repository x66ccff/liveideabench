"""
数据库管理模块

使用SQLite处理数据存储，支持多进程并发访问
"""

import os
import sqlite3
import json
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 数据库文件路径
DB_PATH = './data/ideabench.db'

# 确保数据库目录存在
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# 线程锁，用于保护连接池
_lock = threading.Lock()

# 连接池 - 每个线程一个连接
_connection_pool = {}


def get_connection() -> sqlite3.Connection:
    """获取当前线程的数据库连接"""
    thread_id = threading.get_ident()
    
    with _lock:
        if thread_id not in _connection_pool:
            conn = sqlite3.connect(DB_PATH, timeout=30.0)
            conn.row_factory = sqlite3.Row  # 使行对象可以通过列名访问
            _connection_pool[thread_id] = conn
            return conn
        return _connection_pool[thread_id]


def close_all_connections() -> None:
    """关闭所有数据库连接"""
    with _lock:
        for conn in _connection_pool.values():
            conn.close()
        _connection_pool.clear()


def init_database() -> None:
    """初始化数据库结构"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # 创建结果表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        keywords TEXT NOT NULL,
        idea_model TEXT NOT NULL,
        critic_model TEXT NOT NULL,
        idea TEXT NOT NULL,
        raw_critique TEXT NOT NULL,
        parsed_scores TEXT,          -- JSON格式存储的分数
        parsed_reasoning TEXT,       -- JSON格式存储的分析
        critique_reasoning TEXT,     -- 批评模型的推理过程
        error TEXT,                  -- 可能的错误信息
        full_response TEXT NOT NULL, -- 完整响应
        first_was_rejected INTEGER DEFAULT 0, -- 标记模型是否首次拒绝请求
        first_reject_response TEXT   -- 保存模型首次拒绝的原因
    )
    ''')
    
    # 创建索引以加速查询
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_keyword_model ON results (keywords, idea_model)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON results (timestamp)')
    
    conn.commit()


def save_result(result_data: Dict[str, Any]) -> int:
    """保存一条评测结果到数据库

    Args:
        result_data: 包含评测结果的字典

    Returns:
        新插入记录的ID
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # 提取并处理数据
    timestamp = datetime.now().isoformat()
    keywords = result_data.get('keywords', '')
    idea_model = result_data.get('idea_model', '')
    critic_model = result_data.get('critic_model', '')
    idea = result_data.get('idea', '')
    raw_critique = result_data.get('raw_critique', '')
    full_response = result_data.get('full_response', '')
    error = result_data.get('error')
    
    # 处理解析结果
    parsed_scores = None
    parsed_reasoning = None
    if 'parsed_score' in result_data and result_data['parsed_score']:
        parsed_scores = json.dumps(result_data['parsed_score'])
    if 'parsed_feedback' in result_data and result_data['parsed_feedback']:
        parsed_reasoning = json.dumps(result_data['parsed_feedback'])
    
    # 获取模型的推理过程
    critique_reasoning = result_data.get('critique_reasoning')
    
    # 获取拒绝状态
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
        logger.error(f"数据库插入错误: {str(e)}")
        conn.rollback()
        raise


def check_duplicate_entries(keyword: str, idea_model: str, limit: int = 6) -> bool:
    """检查是否存在足够数量的相同关键词和模型组合的记录

    Args:
        keyword: 关键词
        idea_model: 想法模型名称
        limit: 最大记录数限制

    Returns:
        如果记录数量达到或超过限制，返回True；否则返回False
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
    """根据过滤条件查询结果

    Args:
        filters: 过滤条件字典
        limit: 最大返回记录数

    Returns:
        结果列表
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
        
        # 解析JSON字段
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
    """将数据库导出为CSV文件

    Args:
        output_path: CSV文件的输出路径
    """
    import pandas as pd
    
    conn = get_connection()
    
    # 读取所有结果
    df = pd.read_sql_query("SELECT * FROM results", conn)
    
    # 处理JSON字段
    for json_col in ['parsed_scores', 'parsed_reasoning']:
        if json_col in df.columns:
            df[json_col] = df[json_col].apply(
                lambda x: json.loads(x) if x and isinstance(x, str) else x
            )
    
    # 导出为CSV
    df.to_csv(output_path, index=False)
    logger.info(f"成功导出数据至 {output_path}")


def check_and_add_column() -> None:
    """检查并添加新字段到现有表中"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # 获取表的当前结构
    cursor.execute("PRAGMA table_info(results)")
    columns = [column[1] for column in cursor.fetchall()]
    
    # 检查critique_reasoning列是否存在
    if 'critique_reasoning' not in columns:
        logger.info("添加critique_reasoning列到results表")
        cursor.execute("ALTER TABLE results ADD COLUMN critique_reasoning TEXT")
        conn.commit()


# 初始化数据库
init_database()

# 检查并更新表结构
check_and_add_column()

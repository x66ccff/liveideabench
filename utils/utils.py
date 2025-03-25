"""
通用工具函数模块

包含哈希生成、超时处理等通用工具函数
"""

import hashlib
import signal

# 生成稳定的哈希值，确保相同的文本总是产生相同的哈希值
def stable_hash(text):
    """
    为输入文本生成稳定的MD5哈希值
    
    Args:
        text: 需要哈希的文本
        
    Returns:
        文本的MD5哈希值（十六进制字符串）
    """
    return hashlib.md5(str(text).encode('utf-8')).hexdigest()

# 超时处理相关函数
def timeout_handler(signum, frame):
    """当函数执行超时时触发的处理器"""
    raise TimeoutError("Function call timed out")

def run_with_timeout(func, timeout, *args, **kwargs):
    """
    在指定时间内运行函数，如果超时则抛出异常
    
    Args:
        func: 要执行的函数
        timeout: 超时时间（秒）
        *args, **kwargs: 传递给func的参数
        
    Returns:
        函数执行的结果
        
    Raises:
        TimeoutError: 如果函数执行超过指定时间
    """
    # 设置信号处理器
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    try:
        result = func(*args, **kwargs)
    finally:
        # 取消警报
        signal.alarm(0)
    return result

def append_content_new(txt_path, content_new):
    """
    将新内容追加到指定的文本文件中
    
    Args:
        txt_path: 目标文件路径
        content_new: 要追加的内容
    """
    with open(txt_path, "a", encoding='utf-8') as f:
        f.write(content_new)

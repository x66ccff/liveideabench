"""
General utility function module

Contains general utility functions such as hash generation and timeout handling
"""

import hashlib
import signal

# Generate a stable hash value, ensuring the same text always produces the same hash value
def stable_hash(text):
    """
    Generate a stable MD5 hash value for the input text
    
    Args:
        text: The text to be hashed
        
    Returns:
        The MD5 hash value of the text (hexadecimal string)
    """
    return hashlib.md5(str(text).encode('utf-8')).hexdigest()

# Timeout handling related functions
def timeout_handler(signum, frame):
    """Handler triggered when function execution times out"""
    raise TimeoutError("Function call timed out")

def run_with_timeout(func, timeout, *args, **kwargs):
    """
    Run a function within a specified time, raising an exception if it times out
    
    Args:
        func: The function to execute
        timeout: Timeout duration (seconds)
        *args, **kwargs: Arguments to pass to func
        
    Returns:
        The result of the function execution
        
    Raises:
        TimeoutError: If the function execution exceeds the specified time
    """
    # Set the signal handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    try:
        result = func(*args, **kwargs)
    finally:
        # Cancel the alarm
        signal.alarm(0)
    return result

def append_content_new(txt_path, content_new):
    """
    Append new content to the specified text file
    
    Args:
        txt_path: Target file path
        content_new: The content to append
    """
    with open(txt_path, "a", encoding='utf-8') as f:
        f.write(content_new)
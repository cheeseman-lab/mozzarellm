import logging
import os
from datetime import datetime

def setup_logger(log_filename, log_dir="logs"):
    """
    Set up and configure a logger with both file and console handlers.
    
    Args:
        log_filename: Base name for the log file
        log_dir: Directory to store log files
        
    Returns:
        logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_log_path = os.path.join(log_dir, f"{log_filename}_{timestamp}.log")
    
    # Create logger
    logger = logging.getLogger(log_filename)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers if any
    if logger.handlers:
        logger.handlers.clear()
    
    # Create file handler
    file_handler = logging.FileHandler(full_log_path)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_format = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_format)
    console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
    logger.addHandler(console_handler)
    
    return logger


def get_model_logger(model_name, start_idx, end_idx, log_dir="logs"):
    """
    Get a logger specifically for tracking model runs.
    
    Args:
        model_name: Name of the model being used
        start_idx: Start index of the data being processed
        end_idx: End index of the data being processed
        log_dir: Directory to store log files
        
    Returns:
        logger: Configured logger instance
    """
    # Clean up model name for filename
    clean_model_name = model_name.replace(':', '_').replace('/', '_').replace('-', '_')
    log_filename = f"{clean_model_name}_{start_idx}_{end_idx}"
    
    return setup_logger(log_filename, log_dir)


def log_api_usage(logger, model, tokens=None, cost=None):
    """
    Log API usage including tokens and cost if available.
    
    Args:
        logger: Logger instance
        model: Model name
        tokens: Number of tokens used (optional)
        cost: Cost of API call (optional)
    """
    message = f"API call to {model}"
    if tokens is not None:
        message += f", {tokens} tokens"
    if cost is not None:
        message += f", ${cost:.6f}"
    
    logger.info(message)
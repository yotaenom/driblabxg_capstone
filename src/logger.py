import logging
import yaml
import os
from datetime import datetime
from typing import Optional


def setup_logger(name: str = "driblab_xg", 
                log_level: str = "INFO",
                log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up a centralized logger for the pipeline.
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_pipeline_step(logger: logging.Logger, step: str, status: str = "STARTED"):
    """
    Log pipeline step execution.
    
    Args:
        logger: Logger instance
        step: Step name
        status: Step status (STARTED, COMPLETED, FAILED)
    """
    logger.info(f"Pipeline Step: {step} - {status}")


def log_metrics(logger: logging.Logger, metrics: dict):
    """
    Log performance metrics.
    
    Args:
        logger: Logger instance
        metrics: Dictionary of metrics to log
    """
    for metric_name, metric_value in metrics.items():
        logger.info(f"Metric - {metric_name}: {metric_value}")


def log_data_info(logger: logging.Logger, data_type: str, count: int, shape: Optional[tuple] = None):
    """
    Log data loading information.
    
    Args:
        logger: Logger instance
        data_type: Type of data (shots, tracking, mappings)
        count: Number of records
        shape: Optional shape tuple for DataFrames
    """
    if shape:
        logger.info(f"Loaded {data_type}: {count} records, shape: {shape}")
    else:
        logger.info(f"Loaded {data_type}: {count} records")


if __name__ == "__main__":
    # Test logger setup
    logger = setup_logger()
    logger.info("Logger test successful") 
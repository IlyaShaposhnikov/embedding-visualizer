"""Logging configuration module"""

import logging
import sys
from pathlib import Path


def setup_logging(
    verbose: bool = False,
    log_file: Path = None,
    console_level: int = logging.INFO
) -> None:
    """
    Setup application.

    Args:
        verbose: If True, sets console level to DEBUG, otherwise INFO
        log_file: Optional path to write logs to file
        console_level: Logging level for console output
    """
    # Determine console logging level
    console_log_level = logging.DEBUG if verbose else console_level

    # Clear any existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set root logger level to lowest level (DEBUG) to allow filtering
    root_logger.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Optional file handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Suppress overly verbose loggers from external libraries
    logging.getLogger('matplotlib').setLevel
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

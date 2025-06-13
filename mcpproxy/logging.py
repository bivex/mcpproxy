import logging
import os
import sys
from datetime import datetime
from pathlib import Path


def configure_logging(level: str = "INFO", log_file: str | None = None) -> logging.Logger:
    """Configure logging with datetime formatting.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for file logging
    """

    # Custom formatter with datetime
    class DateTimeFormatter(logging.Formatter):
        def format(self, record):
            # Add current datetime to the record
            record.datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return super().format(record)

    # Create logger
    logger = logging.getLogger("mcpproxy")
    logger.setLevel(getattr(logging, level.upper()))

    # Clear any existing handlers to ensure clean configuration
    # This prevents issues with duplicate or improperly configured handlers
    logger.handlers.clear()

    # Create console handler - use stderr to avoid conflicts with MCP stdio protocol
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(getattr(logging, level.upper()))

    # Create formatter
    formatter = DateTimeFormatter(
        fmt="%(datetime)s [%(levelname)s] %(name)s: %(message)s"
    )
    console_handler.setFormatter(formatter)

    # Add console handler to logger
    logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        try:
            # Ensure log directory exists
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, level.upper()))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            # Log the error to stderr but don't fail
            print(f"Warning: Could not create log file {log_file}: {e}", file=sys.stderr)

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    return logger


def get_logger(name: str = "mcpproxy") -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)

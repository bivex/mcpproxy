import logging
import sys
from datetime import datetime


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configure logging with datetime formatting."""

    # Custom formatter with datetime
    class DateTimeFormatter(logging.Formatter):
        def format(self, record):
            # Add current datetime to the record
            record.datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return super().format(record)

    # Create logger
    logger = logging.getLogger("mcpproxy")
    logger.setLevel(getattr(logging, level.upper()))

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))

    # Create formatter
    formatter = DateTimeFormatter(
        fmt="%(datetime)s [%(levelname)s] %(name)s: %(message)s"
    )
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)

    return logger


def get_logger(name: str = "mcpproxy") -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)

"""Handles file operations for configuration, including reading, writing, and creating sample configs."""

import json
from pathlib import Path
from typing import Any

from mcpproxy.logging import get_logger


class ConfigFileHandler:
    """Manages reading, writing, and creating configuration files."""

    def __init__(self, config_path: Path):
        self.config_path = config_path

    def file_exists(self) -> bool:
        """Check if the configuration file exists."""
        return self.config_path.exists()

    def read_config_file(self) -> dict[str, Any]:
        """Read configuration data from the JSON file."""
        with open(self.config_path) as f:
            return json.load(f)

    def create_sample_config(self) -> None:
        """Create a sample configuration file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        sample_config = {
            "mcpServers": {
                "core-docs": {"url": "http://localhost:8000/sse"},
            },
            "tool_name_limit": 60,
            "tools_limit": 15,
        }

        with open(self.config_path, "w") as f:
            json.dump(sample_config, f, indent=2)

        logger = get_logger()
        logger.info(f"Sample configuration created at {self.config_path}")
        logger.info("Set environment variables:")
        logger.info("  MCPPROXY_EMBEDDER=BM25|HF|OPENAI")
        logger.info("  MCPPROXY_HF_MODEL=sentence-transformers/all-MiniLM-L6-v2")
        logger.info("  MCPPROXY_TOOL_NAME_LIMIT=60  # Maximum tool name length")
        logger.info("  MCPPROXY_TOOLS_LIMIT=15  # Maximum number of tools in active pool") 

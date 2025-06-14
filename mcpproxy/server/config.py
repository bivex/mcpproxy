"""Configuration loader for Smart MCP Proxy."""

import json
import os
from pathlib import Path

from ..logging import get_logger
from ..models.schemas import EmbedderType, ProxyConfig, ServerConfig


class ConfigLoader:
    """Configuration loader supporting Cursor IDE style JSON config."""

    def __init__(self, config_path: str = "mcp_config.json"):
        self.config_path = Path(config_path)

    def load_config(self) -> ProxyConfig:
        """Load configuration from JSON file and environment variables."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path) as f:
            config_data = json.load(f)

        # Parse server configurations
        mcp_servers = {}
        for name, server_data in config_data.get("mcpServers", {}).items():
            mcp_servers[name] = ServerConfig(**server_data)

        # Get embedder configuration from environment
        embedder_type = EmbedderType(os.getenv("MCPPROXY_EMBEDDER", "BM25"))
        hf_model = os.getenv("MCPPROXY_HF_MODEL")
        top_k = int(os.getenv("MCPPROXY_TOP_K", "5"))
        tool_name_limit = int(os.getenv("MCPPROXY_TOOL_NAME_LIMIT", "60"))

        return ProxyConfig(
            mcp_servers=mcp_servers,
            embedder=embedder_type,
            hf_model=hf_model,
            top_k=top_k,
            tool_name_limit=tool_name_limit,
        )

    def resolve_env_vars(self, text: str) -> str:
        """Resolve environment variables in text like ${VAR_NAME}."""
        import re

        def replace_var(match):
            var_name = match.group(1)
            return os.getenv(var_name, match.group(0))

        return re.sub(r"\$\{([^}]+)\}", replace_var, text)

    def create_sample_config(self, output_path: str = "mcp_config.json") -> None:
        """Create a sample configuration file."""
        sample_config = {
            "mcpServers": {
                "core-docs": {"url": "http://localhost:8000/sse"},
            }
        }

        with open(output_path, "w") as f:
            json.dump(sample_config, f, indent=2)

        logger = get_logger()
        logger.info(f"Sample configuration created at {output_path}")
        logger.info("Set environment variables:")
        logger.info("  MCPPROXY_EMBEDDER=BM25|HF|OPENAI")
        logger.info("  MCPPROXY_HF_MODEL=sentence-transformers/all-MiniLM-L6-v2")
        logger.info("  MCPPROXY_TOOL_NAME_LIMIT=60  # Maximum tool name length")

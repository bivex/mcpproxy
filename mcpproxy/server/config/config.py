"""Configuration loader for Smart MCP Proxy."""

import json
import os
from pathlib import Path

from mcpproxy.logging import get_logger
from mcpproxy.models.schemas import EmbedderType, ProxyConfig, ServerConfig
from mcpproxy.utils.dependency_management.dependencies import resolve_env_vars
from .config_file_handler import ConfigFileHandler

TOP_K_DEFAULT = 5  # Default number of top results for search


class ConfigLoader:
    """Configuration loader supporting Cursor IDE style JSON config."""

    def __init__(self, config_path: str | None = None):
        if config_path is None:
            self.config_path = Path.home() / ".cursor" / "mcp_proxy.json"
        else:
            self.config_path = Path(config_path)
        self.file_handler = ConfigFileHandler(self.config_path)

    def load_config(self) -> ProxyConfig:
        """Load configuration from JSON file and environment variables."""
        if not self.file_handler.file_exists():
            logger = get_logger()
            logger.warning(f"Configuration file not found: {self.config_path}. Returning default configuration.")
            return ProxyConfig(mcp_servers={}, embedder=EmbedderType.BM25, hf_model=None, top_k=TOP_K_DEFAULT, tool_name_limit=60)

        config_data = self.file_handler.read_config_file()

        # Parse server configurations
        mcp_servers = {}
        for name, server_data in config_data.get("mcpServers", {}).items():
            mcp_servers[name] = ServerConfig(**server_data)

        # Get embedder configuration from environment
        embedder_type = EmbedderType(os.getenv("MCPPROXY_EMBEDDER", "BM25"))
        hf_model = os.getenv("MCPPROXY_HF_MODEL")
        top_k = int(os.getenv("MCPPROXY_TOP_K", str(TOP_K_DEFAULT)))
        tool_name_limit = int(os.getenv("MCPPROXY_TOOL_NAME_LIMIT", "60"))

        return ProxyConfig(
            mcp_servers=mcp_servers,
            embedder=embedder_type,
            hf_model=hf_model,
            top_k=top_k,
            tool_name_limit=tool_name_limit,
        )

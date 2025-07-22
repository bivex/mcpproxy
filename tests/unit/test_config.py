"""Tests for configuration loading functionality."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from mcpproxy.models.schemas import EmbedderType, ProxyConfig, ServerConfig
from mcpproxy.server.config.config import ConfigLoader
from mcpproxy.server.config.config_file_handler import ConfigFileHandler
from mcpproxy.models.constants import DEFAULT_TOOL_NAME_LIMIT, DEFAULT_TOOLS_LIMIT, DEFAULT_SEARCH_RESULT_LIMIT


class TestConfigLoader:
    """Test cases for ConfigLoader."""

    def create_sample_config_file(self, config_data: dict) -> str:
        """Create a temporary config file with given data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            return f.name

    def test_config_default_tool_name_limit(self):
        """Test loading config with default tool name limit."""
        config_data = {
            "mcpServers": {
                "test-server": {
                    "url": "http://localhost:8000"
                }
            }
        }
        
        config_file = self.create_sample_config_file(config_data)
        
        try:
            with patch.dict(os.environ, {}, clear=True):
                # Clear environment to test defaults
                loader = ConfigLoader(config_file)
                config = loader.load_config()
                
                assert config.tool_name_limit == DEFAULT_TOOL_NAME_LIMIT  # Default value
                assert config.embedder == EmbedderType.BM25  # Default embedder
                assert config.top_k == DEFAULT_SEARCH_RESULT_LIMIT  # Default top_k
                assert config.tools_limit == DEFAULT_TOOLS_LIMIT # Default tools_limit
        finally:
            os.unlink(config_file)

    def test_config_custom_tool_name_limit(self):
        """Test loading config with custom tool name limit from environment."""
        config_data = {
            "mcpServers": {
                "test-server": {
                    "url": "http://localhost:8000"
                }
            }
        }
        
        config_file = self.create_sample_config_file(config_data)
        
        try:
            CUSTOM_TOOL_NAME_LIMIT = 40
            with patch.dict(os.environ, {"MCPPROXY_TOOL_NAME_LIMIT": str(CUSTOM_TOOL_NAME_LIMIT)}, clear=True):
                loader = ConfigLoader(config_file)
                config = loader.load_config()
                
                assert config.tool_name_limit == CUSTOM_TOOL_NAME_LIMIT
        finally:
            os.unlink(config_file)

    def test_load_config_all_env_vars(self):
        """Test loading config with all environment variables set."""
        config_data = {
            "mcpServers": {
                "test-server": {
                    "url": "http://localhost:8000"
                },
                "command-server": {
                    "command": "test-command",
                    "args": ["arg1", "arg2"]
                }
            }
        }
        
        config_file = self.create_sample_config_file(config_data)
        
        try:
            env_vars = {
                "MCPPROXY_EMBEDDER": "HF",
                "MCPPROXY_HF_MODEL": "custom-model",
                "MCPPROXY_TOP_K": "10",
                "MCPPROXY_TOOL_NAME_LIMIT": "80"
            }
            CUSTOM_TOP_K = 10
            CUSTOM_TOOL_NAME_LIMIT_ENV = 80
            with patch.dict(os.environ, {
                "MCPPROXY_EMBEDDER": "HF",
                "MCPPROXY_HF_MODEL": "custom-model",
                "MCPPROXY_TOP_K": str(CUSTOM_TOP_K),
                "MCPPROXY_TOOL_NAME_LIMIT": str(CUSTOM_TOOL_NAME_LIMIT_ENV)
            }, clear=True):
                loader = ConfigLoader(config_file)
                config = loader.load_config()
                
                assert config.embedder == EmbedderType.HF
                assert config.hf_model == "custom-model"
                assert config.top_k == CUSTOM_TOP_K
                assert config.tool_name_limit == CUSTOM_TOOL_NAME_LIMIT_ENV
                assert len(config.mcp_servers) == 2
        finally:
            os.unlink(config_file)

    def test_config_invalid_tool_name_limit(self):
        """Test behavior with invalid tool name limit value."""
        config_data = {
            "mcpServers": {
                "test-server": {
                    "url": "http://localhost:8000"
                }
            }
        }
        
        config_file = self.create_sample_config_file(config_data)
        
        try:
            with patch.dict(os.environ, {"MCPPROXY_TOOL_NAME_LIMIT": "not_a_number"}, clear=True):
                loader = ConfigLoader(config_file)
                
                # Should raise ValueError when trying to convert invalid string to int
                with pytest.raises(ValueError):
                    loader.load_config()
        finally:
            os.unlink(config_file)

    def test_config_zero_tool_name_limit(self):
        """Test behavior with zero tool name limit."""
        config_data = {
            "mcpServers": {
                "test-server": {
                    "url": "http://localhost:8000"
                }
            }
        }
        
        config_file = self.create_sample_config_file(config_data)
        
        try:
            with patch.dict(os.environ, {"MCPPROXY_TOOL_NAME_LIMIT": "0"}, clear=True):
                loader = ConfigLoader(config_file)
                config = loader.load_config()
                
                assert config.tool_name_limit == 0
        finally:
            os.unlink(config_file)

    def test_config_large_tool_name_limit(self):
        """Test behavior with very large tool name limit."""
        config_data = {
            "mcpServers": {
                "test-server": {
                    "url": "http://localhost:8000"
                }
            }
        }
        
        config_file = self.create_sample_config_file(config_data)
        
        try:
            with patch.dict(os.environ, {"MCPPROXY_TOOL_NAME_LIMIT": "1000"}, clear=True):
                loader = ConfigLoader(config_file)
                config = loader.load_config()
                
                assert config.tool_name_limit == 1000
        finally:
            os.unlink(config_file)

    def test_load_config_missing_file(self, tmp_path):
        """Test loading config when file does not exist, should return default config."""
        non_existent_config_file = tmp_path / "nonexistent_config.json"
        
        loader = ConfigLoader(str(non_existent_config_file))
        config = loader.load_config()
        
        # Assert that a default ProxyConfig is returned
        assert isinstance(config, ProxyConfig)
        assert config.mcp_servers == {}
        assert config.embedder == EmbedderType.BM25
        assert config.top_k == DEFAULT_SEARCH_RESULT_LIMIT
        assert config.tool_name_limit == DEFAULT_TOOL_NAME_LIMIT
        assert config.tools_limit == DEFAULT_TOOLS_LIMIT # Ensure tools_limit is also default

    def test_config_includes_tool_name_limit_docs(self, tmp_path):
        """Test that create_sample_config includes tool_name_limit documentation."""
        config_path = tmp_path / "temp_mcp_proxy.json"
        handler = ConfigFileHandler(config_path)
        handler.create_sample_config()

        with open(config_path) as f:
            content = json.load(f)
            assert content["tool_name_limit"] == DEFAULT_TOOL_NAME_LIMIT
            assert content["tools_limit"] == DEFAULT_TOOLS_LIMIT

    def test_proxy_config_defaults(self):
        """Test that ProxyConfig model has correct default values."""
        config = ProxyConfig(mcp_servers={})
        
        assert config.embedder == EmbedderType.BM25
        assert config.hf_model is None
        assert config.top_k == DEFAULT_SEARCH_RESULT_LIMIT
        assert config.tool_name_limit == DEFAULT_TOOL_NAME_LIMIT

    def test_config_custom_values(self):
        """Test ProxyConfig model with custom values."""
        config = ProxyConfig(
            mcp_servers={"test": ServerConfig(url="http://test")},
            embedder=EmbedderType.HF,
            hf_model="custom-model",
            top_k=15,
            tool_name_limit=120,
            tools_limit=30
        )
        
        assert config.embedder == EmbedderType.HF
        assert config.hf_model == "custom-model"
        assert config.top_k == 15
        assert config.tool_name_limit == 120
        assert config.tools_limit == 30 

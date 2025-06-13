"""Tests for configuration loading functionality."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from mcpproxy.models.schemas import EmbedderType, ProxyConfig, ServerConfig
from mcpproxy.server.config import ConfigLoader


class TestConfigLoader:
    """Test cases for ConfigLoader."""

    def create_sample_config_file(self, config_data: dict) -> str:
        """Create a temporary config file with given data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            return f.name

    def test_load_config_with_default_tool_name_limit(self):
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
                
                assert config.tool_name_limit == 60  # Default value
                assert config.embedder == EmbedderType.BM25  # Default embedder
                assert config.top_k == 5  # Default top_k
        finally:
            os.unlink(config_file)

    def test_load_config_with_custom_tool_name_limit(self):
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
            with patch.dict(os.environ, {"MCPPROXY_TOOL_NAME_LIMIT": "40"}, clear=True):
                loader = ConfigLoader(config_file)
                config = loader.load_config()
                
                assert config.tool_name_limit == 40
        finally:
            os.unlink(config_file)

    def test_load_config_with_all_environment_variables(self):
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
            
            with patch.dict(os.environ, env_vars, clear=True):
                loader = ConfigLoader(config_file)
                config = loader.load_config()
                
                assert config.embedder == EmbedderType.HF
                assert config.hf_model == "custom-model"
                assert config.top_k == 10
                assert config.tool_name_limit == 80
                assert len(config.mcp_servers) == 2
        finally:
            os.unlink(config_file)

    def test_load_config_invalid_tool_name_limit(self):
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

    def test_load_config_zero_tool_name_limit(self):
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

    def test_load_config_very_large_tool_name_limit(self):
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

    def test_load_config_missing_file(self):
        """Test behavior when config file doesn't exist."""
        loader = ConfigLoader("nonexistent_config.json")
        
        with pytest.raises(FileNotFoundError):
            loader.load_config()

    def test_create_sample_config_includes_tool_name_limit_docs(self, caplog):
        """Test that creating sample config includes documentation about MCPPROXY_TOOL_NAME_LIMIT."""
        import io
        import sys
        from unittest.mock import patch
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"
            
            # Capture stdout since the logger might output to stdout
            captured_output = io.StringIO()
            
            with patch('sys.stdout', captured_output):
                with caplog.at_level('INFO'):
                    loader = ConfigLoader()
                    loader.create_sample_config(str(config_path))
            
            # Check that the file was created
            assert config_path.exists()
            
            # Check both captured logs and stdout for the tool name limit documentation
            log_output = caplog.text
            stdout_output = captured_output.getvalue()
            combined_output = log_output + stdout_output
            
            assert "MCPPROXY_TOOL_NAME_LIMIT=60" in combined_output

    def test_proxy_config_model_defaults(self):
        """Test that ProxyConfig model has correct default values."""
        config = ProxyConfig(mcp_servers={})
        
        assert config.embedder == EmbedderType.BM25
        assert config.hf_model is None
        assert config.top_k == 5
        assert config.tool_name_limit == 60

    def test_proxy_config_model_custom_values(self):
        """Test ProxyConfig model with custom values."""
        config = ProxyConfig(
            mcp_servers={"test": ServerConfig(url="http://test")},
            embedder=EmbedderType.HF,
            hf_model="custom-model",
            top_k=15,
            tool_name_limit=120
        )
        
        assert config.embedder == EmbedderType.HF
        assert config.hf_model == "custom-model"
        assert config.top_k == 15
        assert config.tool_name_limit == 120 
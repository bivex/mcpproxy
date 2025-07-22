"""Tests for Smart MCP Proxy server functionality."""

import pytest
from unittest.mock import Mock, patch

from mcpproxy.models.schemas import ProxyConfig, ServerConfig, EmbedderType
from mcpproxy.server.mcp_server import SmartMCPProxyServer
from mcpproxy.server.config.config import ConfigLoader
from mcpproxy.utils.name_sanitization.name_sanitizer import sanitize_tool_name # Corrected path


class TestSmartMCPProxyServer:
    """Test cases for SmartMCPProxyServer."""

    def create_test_config(self, tool_name_limit: int = 60, tools_limit: int = 15) -> ProxyConfig:
        """Create a test configuration with specified tool name limit."""
        return ProxyConfig(
            mcp_servers={},
            embedder=EmbedderType.BM25,
            tool_name_limit=tool_name_limit,
            tools_limit=tools_limit,
        )

    def create_mock_server(self, tool_name_limit: int = 60, tools_limit: int = 15) -> SmartMCPProxyServer:
        """Create a mock server instance with test configuration."""
        with patch("mcpproxy.server.mcp_server.ConfigLoader") as mock_loader:
            mock_loader.return_value.load_config.return_value = self.create_test_config(
                tool_name_limit,
                tools_limit,
            )
            server = SmartMCPProxyServer("test_config.json")
            return server


class TestToolNameSanitizer:
    """Test cases for tool name sanitization functionality."""

    @pytest.mark.parametrize(
        "test_case",
        [
            {"server_name": "test_server", "tool_name": "test_tool", "tool_name_limit": 60, "expected_start": "test_server_test_tool", "expected_len": 60, "expected_no_trailing_underscore": True, "expected_contains": []},  # Basic
            {"server_name": "my-server.com", "tool_name": "get/data:v1", "tool_name_limit": 60, "expected_start": "my_server_com_get_data_v1", "expected_len": 60, "expected_no_trailing_underscore": True, "expected_contains": []},  # Special chars
            {"server_name": "MyServer", "tool_name": "GetData", "tool_name_limit": 60, "expected_start": "myserver_getdata", "expected_len": 60, "expected_no_trailing_underscore": True, "expected_contains": []},  # Uppercase
            {"server_name": "very_long_server_name_that_goes_on_and_on", "tool_name": "extremely_long_tool_name_that_exceeds_reasonable_limits", "tool_name_limit": 60, "expected_start": "very_long_server_name_that_goes_on_and_on_", "expected_len": 60, "expected_no_trailing_underscore": True, "expected_contains": []}, # Default limit
            {"server_name": "long_server_name", "tool_name": "long_tool_name", "tool_name_limit": 30, "expected_start": "long_server_name_long_tool", "expected_len": 30, "expected_no_trailing_underscore": True, "expected_contains": []},  # Custom limit 30
            {"server_name": "server_name_that_would_exceed_default_60_limit", "tool_name": "tool_name_that_would_also_exceed_60_char_limit", "tool_name_limit": 100, "expected_start": "server_name_that_would_exceed_default_60_limit_tool_name_that_would_also_exceed_60_char_limit", "expected_len": 100, "expected_no_trailing_underscore": True, "expected_contains": []}, # Custom limit 100
            {"server_name": "server", "tool_name": "tool", "tool_name_limit": 10, "expected_start": "server_tool", "expected_len": 10, "expected_no_trailing_underscore": True, "expected_contains": []}, # Original test for truncation, expect full name and correct length
            {"server_name": "server", "tool_name": "tool", "tool_name_limit": 10, "expected_start": "server_too", "expected_len": 10, "expected_no_trailing_underscore": True, "expected_contains": []}, # Very short limit, expected: "server_too"
            {"server_name": "test_server", "tool_name": "test_tool", "tool_name_limit": 60, "expected_start": "test_server_test_tool", "expected_len": 60, "expected_no_trailing_underscore": True, "expected_contains": []}, # Test normal case
            {"server_name": "test_server", "tool_name": "", "tool_name_limit": 60, "expected_start": "test_server_tool", "expected_len": 60, "expected_no_trailing_underscore": True, "expected_contains": []}, # Empty tool name
            {"server_name": "", "tool_name": "test_tool", "tool_name_limit": 60, "expected_start": "server_test_tool", "expected_len": 60, "expected_no_trailing_underscore": True, "expected_contains": []}, # Empty server name
            {"server_name": "", "tool_name": "", "tool_name_limit": 60, "expected_start": "server_tool", "expected_len": 60, "expected_no_trailing_underscore": True, "expected_contains": []}, # Both empty
            {"server_name": "test__server", "tool_name": "test___tool", "tool_name_limit": 60, "expected_start": "test_server_test_tool", "expected_len": 60, "expected_no_trailing_underscore": True, "expected_contains": []}, # Consecutive underscores
            {"server_name": "myserv", "tool_name": "very_long_tool_name_here", "tool_name_limit": 20, "expected_start": "myserv_very_long", "expected_len": 17, "expected_no_trailing_underscore": True, "expected_contains": []}, # Server part + truncated tool name, expected: "myserv_very_long"
        ],
    )
    def test_sanitize_name_scenarios(self, test_case):
        """Test various scenarios for tool name sanitization."""
        server_name = test_case["server_name"]
        tool_name = test_case["tool_name"]
        tool_name_limit = test_case["tool_name_limit"]
        expected_start = test_case["expected_start"]
        expected_len = test_case["expected_len"]
        expected_no_trailing_underscore = test_case["expected_no_trailing_underscore"]
        expected_contains = test_case["expected_contains"]

        result = sanitize_tool_name(server_name, tool_name, tool_name_limit=tool_name_limit)

        assert len(result) <= expected_len
        assert result.startswith(expected_start) or any(s in result for s in expected_contains)

        if expected_no_trailing_underscore:
            assert not result.endswith("_")

    def test_sanitize_name_starts_with_letter(self):
        """Test that sanitized names start with letter or underscore."""
        result = sanitize_tool_name("123server", "456tool")
        
        assert result.startswith(("tool_", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "_"))

    def test_sanitize_name_regex_compliance(self):
        """Test that sanitized names conform to a simple regex (alphanumeric, underscore)."""
        # Test names that should be fully sanitized
        names_to_test = [
            ("my-server.com", "get/data:v1"),
            ("Server Name With Spaces", "Tool.Name-With.Dots"),
            ("123prefix", "tool_name"),
            ("!@#$", "%^&*()"),
        ]

        for s_name, t_name in names_to_test:
            result = sanitize_tool_name(s_name, t_name)
            # Check if it contains only alphanumeric or underscores
            assert all(c.isalnum() or c == '_' for c in result)
            assert not ('-' in result or '.' in result or '/' in result or ':' in result)
            assert result.lower() == result # Check for lowercase conversion 

"""Tests for Smart MCP Proxy server functionality."""

import pytest
from unittest.mock import Mock, patch

from mcpproxy.models.schemas import ProxyConfig, ServerConfig, EmbedderType
from mcpproxy.server.mcp_server import SmartMCPProxyServer


class TestSmartMCPProxyServer:
    """Test cases for SmartMCPProxyServer."""

    def create_test_config(self, tool_name_limit: int = 60) -> ProxyConfig:
        """Create a test configuration with specified tool name limit."""
        return ProxyConfig(
            mcp_servers={},
            embedder=EmbedderType.BM25,
            tool_name_limit=tool_name_limit,
        )

    def create_mock_server(self, tool_name_limit: int = 60) -> SmartMCPProxyServer:
        """Create a mock server instance with test configuration."""
        with patch("mcpproxy.server.mcp_server.ConfigLoader") as mock_loader:
            mock_loader.return_value.load_config.return_value = self.create_test_config(
                tool_name_limit
            )
            server = SmartMCPProxyServer("test_config.json")
            return server


class TestToolNameSanitization(TestSmartMCPProxyServer):
    """Test cases for tool name sanitization functionality."""

    @pytest.mark.parametrize(
        "server_name, tool_name, tool_name_limit, expected_start, expected_len, expected_no_trailing_underscore, expected_contains",
        [
            ("test_server", "test_tool", 60, "test_server_test_tool", 60, True, []),  # Basic
            ("my-server.com", "get/data:v1", 60, "my_server_com_get_data_v1", 60, True, []),  # Special chars
            ("MyServer", "GetData", 60, "myserver_getdata", 60, True, []),  # Uppercase
            ("very_long_server_name_that_goes_on_and_on", "extremely_long_tool_name_that_exceeds_reasonable_limits", 60, "very_long_server_name_that_goes_on_and_on_", 60, True, []), # Default limit
            ("long_server_name", "long_tool_name", 30, "long_server_name_long_tool", 30, True, []),  # Custom limit 30
            ("server_name_that_would_exceed_default_60_limit", "tool_name_that_would_also_exceed_60_char_limit", 100, "server_name_that_would_exceed_default_60_limit_tool_name_that_would_also_exceed_60_char_limit", 100, True, []), # Custom limit 100
            ("server", "tool", 10, "server_tool", 10, True, []),  # Very short limit
            ("", "test_tool", 60, "server_test_tool", 60, True, []),  # Empty server name
            ("test_server", "", 60, "test_server_tool", 60, True, []),  # Empty tool name
            ("", "", 60, "server_tool", 60, True, []),  # Both empty
            ("test__server", "test___tool", 60, "test_server_test_tool", 60, True, []),  # Consecutive underscores
            ("myserv", "very_long_tool_name_here", 20, "myserv_very_long_", 20, True, ["myserv_"]),  # Preserve prefix
        ],
    )
    def test_sanitize_tool_name_scenarios(self,
        server_name: str,
        tool_name: str,
        tool_name_limit: int,
        expected_start: str,
        expected_len: int,
        expected_no_trailing_underscore: bool,
        expected_contains: list[str],
    ):
        """Test various scenarios for tool name sanitization."""
        server = self.create_mock_server(tool_name_limit=tool_name_limit)
        result = server._sanitize_tool_name(server_name, tool_name)

        assert len(result) <= expected_len
        assert result.startswith(expected_start) or any(s in result for s in expected_contains)

        if expected_no_trailing_underscore:
            assert not result.endswith("_")

    def test_sanitize_tool_name_starts_with_letter_requirement(self):
        """Test that sanitized names start with letter or underscore."""
        server = self.create_mock_server()
        result = server._sanitize_tool_name("123server", "456tool")
        
        assert result.startswith(("tool_", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "_"))

    def test_sanitize_tool_name_regex_compliance(self):
        """Test that sanitized names conform to a simple regex (alphanumeric, underscore)."""
        server = self.create_mock_server()
        # Test names that should be fully sanitized
        names_to_test = [
            ("my-server.com", "get/data:v1"),
            ("Server Name With Spaces", "Tool.Name-With.Dots"),
            ("123prefix", "tool_name"),
            ("!@#$", "%^&*()"),
        ]

        for s_name, t_name in names_to_test:
            result = server._sanitize_tool_name(s_name, t_name)
            # Check if it contains only alphanumeric or underscores
            assert all(c.isalnum() or c == '_' for c in result)
            assert not ('-' in result or '.' in result or '/' in result or ':' in result)
            assert result.lower() == result # Check for lowercase conversion 

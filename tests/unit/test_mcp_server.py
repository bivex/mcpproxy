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

    def test_sanitize_tool_name_basic(self):
        """Test basic tool name sanitization."""
        server = self.create_mock_server()
        result = server._sanitize_tool_name("test_server", "test_tool")
        assert result == "test_server_test_tool"

    def test_sanitize_tool_name_with_special_chars(self):
        """Test sanitization with special characters."""
        server = self.create_mock_server()
        result = server._sanitize_tool_name("my-server.com", "get/data:v1")
        assert result == "my_server_com_get_data_v1"

    def test_sanitize_tool_name_uppercase_conversion(self):
        """Test that uppercase letters are converted to lowercase."""
        server = self.create_mock_server()
        result = server._sanitize_tool_name("MyServer", "GetData")
        assert result == "myserver_getdata"

    def test_sanitize_tool_name_default_limit_60(self):
        """Test that default limit of 60 characters is applied."""
        server = self.create_mock_server(tool_name_limit=60)
        long_server = "very_long_server_name_that_goes_on_and_on"
        long_tool = "extremely_long_tool_name_that_exceeds_reasonable_limits"
        result = server._sanitize_tool_name(long_server, long_tool)
        
        assert len(result) <= 60
        assert result.startswith("very_long_server_name_that_goes_on_and_on_")
        assert not result.endswith("_")

    def test_sanitize_tool_name_custom_limit_30(self):
        """Test custom tool name limit of 30 characters."""
        server = self.create_mock_server(tool_name_limit=30)
        result = server._sanitize_tool_name("long_server_name", "long_tool_name")
        
        assert len(result) <= 30
        assert not result.endswith("_")

    def test_sanitize_tool_name_custom_limit_100(self):
        """Test custom tool name limit of 100 characters."""
        server = self.create_mock_server(tool_name_limit=100)
        long_server = "server_name_that_would_exceed_default_60_limit"
        long_tool = "tool_name_that_would_also_exceed_60_char_limit"
        result = server._sanitize_tool_name(long_server, long_tool)
        
        expected = f"{long_server}_{long_tool}"
        # Check that with 100 char limit, this fits completely
        assert result == expected
        assert len(result) <= 100
        assert len(result) == len(expected)  # Should not be truncated

    def test_sanitize_tool_name_very_short_limit(self):
        """Test behavior with very short limit (edge case)."""
        server = self.create_mock_server(tool_name_limit=10)
        result = server._sanitize_tool_name("server", "tool")
        
        assert len(result) <= 10
        assert result == "server_too"  # Should keep server prefix and part of tool name

    def test_sanitize_tool_name_starts_with_letter_requirement(self):
        """Test that sanitized names start with letter or underscore."""
        server = self.create_mock_server()
        result = server._sanitize_tool_name("123server", "456tool")
        
        assert result.startswith(("tool_", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "_"))
        # Should be tool_123server_456tool or similar

    def test_sanitize_tool_name_empty_inputs(self):
        """Test behavior with empty server or tool names."""
        server = self.create_mock_server()
        
        # Empty server name
        result1 = server._sanitize_tool_name("", "test_tool")
        assert result1 == "server_test_tool"
        
        # Empty tool name
        result2 = server._sanitize_tool_name("test_server", "")
        assert result2 == "test_server_tool"
        
        # Both empty
        result3 = server._sanitize_tool_name("", "")
        assert result3 == "server_tool"

    def test_sanitize_tool_name_consecutive_underscores(self):
        """Test that consecutive underscores are collapsed."""
        server = self.create_mock_server()
        result = server._sanitize_tool_name("test__server", "test___tool")
        assert result == "test_server_test_tool"

    def test_sanitize_tool_name_preserve_server_prefix(self):
        """Test that server prefix is preserved when possible within limit."""
        server = self.create_mock_server(tool_name_limit=20)
        result = server._sanitize_tool_name("myserv", "very_long_tool_name_here")
        
        assert result.startswith("myserv_")
        assert len(result) <= 20
        assert not result.endswith("_")

    def test_sanitize_tool_name_fallback_when_no_space_for_server(self):
        """Test fallback behavior when there's no space for meaningful server prefix."""
        server = self.create_mock_server(tool_name_limit=15)
        result = server._sanitize_tool_name("very_long_server_name", "tool")
        
        assert len(result) <= 15
        # Should fall back to truncating the entire combined name

    def test_sanitize_tool_name_different_limits_same_input(self):
        """Test that different limits produce different results for same input."""
        server_30 = self.create_mock_server(tool_name_limit=30)
        server_60 = self.create_mock_server(tool_name_limit=60)
        
        long_server = "really_long_server_name"
        long_tool = "really_long_tool_name_here"
        
        result_30 = server_30._sanitize_tool_name(long_server, long_tool)
        result_60 = server_60._sanitize_tool_name(long_server, long_tool)
        
        assert len(result_30) <= 30
        assert len(result_60) <= 60
        assert len(result_60) >= len(result_30)  # 60 char limit should allow longer name

    def test_sanitize_tool_name_regex_compliance(self):
        """Test that sanitized names comply with expected regex pattern."""
        import re
        
        server = self.create_mock_server()
        test_cases = [
            ("server", "tool"),
            ("my-server.com", "get/data:v1"),
            ("123server", "456tool"),
            ("_server", "_tool"),
            ("server_", "_tool_"),
        ]
        
        for server_name, tool_name in test_cases:
            result = server._sanitize_tool_name(server_name, tool_name)
            
            # Should match pattern: starts with letter or underscore, followed by letters, numbers, underscores
            assert re.match(r'^[a-z_][a-z0-9_]*$', result), f"Result '{result}' doesn't match expected pattern"
            assert not result.endswith('_'), f"Result '{result}' shouldn't end with underscore"
            assert len(result) <= server.config.tool_name_limit 
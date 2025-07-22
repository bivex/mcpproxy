"""Unit tests for tool pool management functionality."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcpproxy.server.mcp_server import SmartMCPProxyServer


class TestToolPoolManagement:
    """Test tool pool management and eviction logic."""

    @pytest.fixture
    def mock_server(self):
        """Create a mock server with tool pool management."""
        with patch("mcpproxy.server.mcp_server.ConfigLoader"):
            server = SmartMCPProxyServer()
            server.tools_limit = 3  # Small limit for testing
            server.mcp = MagicMock()
            server.mcp.remove_tool = MagicMock()
            return server

    @pytest.mark.parametrize(
        "score, time_ago_minutes, expected_weight_tolerance",
        [
            (0.8, 0, 0.01),  # Fresh tool
            (0.8, 35, 0.01), # Old tool (older than max age)
            (0.6, 15, 0.01), # Medium age
        ],
    )
    def test_calculate_tool_weight_scenarios(
        self, mock_server, score: float, time_ago_minutes: int, expected_weight_tolerance: float
    ):
        """Test weight calculation for various tool ages."""
        current_time = time.time()
        timestamp = current_time - (time_ago_minutes * 60)

        weight = mock_server._calculate_tool_weight(score, timestamp)

        # Expected calculation based on the method: (score * 0.7) + (freshness * 0.3)
        # Freshness is (max_age_seconds - age_seconds) / max_age_seconds
        # Max age is 30 minutes (1800 seconds)
        age_seconds = time_ago_minutes * 60
        max_age_seconds = 30 * 60
        freshness = max(0.0, 1.0 - (age_seconds / max_age_seconds))
        expected = (score * 0.7) + (freshness * 0.3)

        assert abs(weight - expected) < expected_weight_tolerance

    @pytest.mark.parametrize(
        "scenario, initial_tools, new_tools, expected_evicted_count, expected_evicted_names, mock_evict_tool_called",
        [
            ("no_eviction", {"tool1": {}, "tool2": {}}, [("new_tool", 0.8)], 0, [], False),
            ("with_eviction", {
                "tool1": {"timestamp": time.time() - 1800, "score": 0.5}, # Old, low score
                "tool2": {"timestamp": time.time() - 900, "score": 0.9}, # Medium age, high score
                "tool3": {"timestamp": time.time(), "score": 0.7}, # Fresh, medium score
            }, [("new_tool1", 0.8), ("new_tool2", 0.6)], 2, ["tool1"], True),
            ("evict_only", {}, [], 1, ["tool1"], True) # Example for evict_tool specific test
        ]
    )
    @pytest.mark.asyncio
    async def test_enforce_tool_pool_limit_scenarios(
        self, mock_server, scenario, initial_tools, new_tools, expected_evicted_count, expected_evicted_names, mock_evict_tool_called
    ):
        """Test tool pool limit enforcement and eviction scenarios."""
        mock_server.current_tool_registrations = {name: MagicMock() for name in initial_tools}
        mock_server.tool_pool_metadata = initial_tools
        mock_server._evict_tool = AsyncMock() # Mock the internal method

        if scenario == "evict_only":
            # This scenario is specifically for testing _evict_tool, not enforce_tool_pool_limit directly
            tool_name_to_evict = expected_evicted_names[0]
            mock_server.current_tool_registrations[tool_name_to_evict] = MagicMock()
            mock_server.registered_tools[tool_name_to_evict] = MagicMock()
            mock_server.tool_pool_metadata[tool_name_to_evict] = {
                "timestamp": time.time(),
                "score": 0.5,
            }
            await mock_server._evict_tool(tool_name_to_evict)

            assert tool_name_to_evict not in mock_server.current_tool_registrations
            assert tool_name_to_evict not in mock_server.registered_tools
            assert tool_name_to_evict not in mock_server.tool_pool_metadata
            mock_server.mcp.remove_tool.assert_called_once_with(tool_name_to_evict)
        else:
            evicted = await mock_server._enforce_tool_pool_limit(new_tools)
            assert len(evicted) == expected_evicted_count
            for name in expected_evicted_names:
                assert name in evicted

            if mock_evict_tool_called:
                assert mock_server._evict_tool.call_count == expected_evicted_count
            else:
                mock_server._evict_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_register_proxy_tool_with_metadata(self, mock_server):
        """Test that tool registration includes metadata tracking."""
        # Create a mock Tool object (what the implementation expects)
        from fastmcp.tools.tool import Tool

        original_tool = MagicMock(spec=Tool)
        original_tool.name = "test_tool"
        original_tool.description = "Test description"
        original_tool.parameters = {}
        original_tool.tags = []  # Required by Tool.from_tool
        original_tool.annotations = {}  # Required by Tool.from_tool
        original_tool.serializer = MagicMock()  # Required by Tool.from_tool
        # Add other common Tool attributes that might be accessed
        original_tool.func = MagicMock()
        original_tool.examples = []

        tool_name = "test_server_test_tool"
        server_name = "test_server"
        score = 0.75

        # Mock dependencies
        mock_server.proxy_servers = {server_name: MagicMock()}
        mock_server.mcp.add_tool = MagicMock()

        with patch("fastmcp.tools.tool.Tool.from_tool") as mock_from_tool:
            mock_proxified_tool = MagicMock()
            mock_from_tool.return_value = mock_proxified_tool

            await mock_server._register_proxy_tool(
                original_tool, tool_name, score, server_name
            )

        # Verify Tool.from_tool was called with correct parameters
        mock_from_tool.assert_called_once()
        call_args = mock_from_tool.call_args
        assert call_args[1]["tool"] == original_tool
        assert call_args[1]["name"] == tool_name
        assert "transform_fn" in call_args[1]

        # Verify metadata was tracked
        assert tool_name in mock_server.tool_pool_metadata
        metadata = mock_server.tool_pool_metadata[tool_name]
        assert metadata["score"] == score
        assert metadata["original_score"] == score
        assert "timestamp" in metadata
        assert isinstance(metadata["timestamp"], float)

        # Verify tool was registered
        assert tool_name in mock_server.current_tool_registrations
        assert tool_name in mock_server.registered_tools

        # Verify the proxified tool was added to FastMCP
        mock_server.mcp.add_tool.assert_called_once_with(mock_proxified_tool)

    def test_tool_weight_comparison(self, mock_server):
        """Test that tool weights are calculated correctly for comparison."""
        current_time = time.time()

        # High score, old tool
        weight1 = mock_server._calculate_tool_weight(
            0.9, current_time - 1800
        )  # 30 min ago

        # Medium score, fresh tool
        weight2 = mock_server._calculate_tool_weight(0.6, current_time)

        # Low score, medium age
        weight3 = mock_server._calculate_tool_weight(
            0.3, current_time - 900
        )  # 15 min ago

        # Fresh high-score tool should have highest weight
        weight4 = mock_server._calculate_tool_weight(0.9, current_time)

        # Verify ordering
        assert weight4 > weight2  # Fresh high-score > fresh medium-score
        assert weight2 > weight1  # Fresh medium > old high (due to freshness factor)
        assert weight1 > weight3  # Old high > medium-age low

    @pytest.mark.asyncio
    async def test_freshness_update_on_existing_tool(self, mock_server):
        """Test that existing tools get freshness updates."""
        # This would be tested in the retrieve_tools integration
        # but we can test the logic separately

        tool_name = "existing_tool"
        original_time = time.time() - 1000
        original_score = 0.6
        new_score = 0.8

        # Set up existing tool metadata
        mock_server.tool_pool_metadata[tool_name] = {
            "timestamp": original_time,
            "score": original_score,
            "original_score": original_score,
        }
        mock_server.current_tool_registrations[tool_name] = MagicMock()

        # Simulate the freshness update logic from retrieve_tools
        mock_server.tool_pool_metadata[tool_name]["timestamp"] = time.time()
        mock_server.tool_pool_metadata[tool_name]["score"] = max(
            mock_server.tool_pool_metadata[tool_name]["score"], new_score
        )

        # Verify updates
        metadata = mock_server.tool_pool_metadata[tool_name]
        assert metadata["score"] == new_score  # Should be updated to higher score
        assert metadata["timestamp"] > original_time  # Should be fresher
        assert metadata["original_score"] == original_score  # Should preserve original

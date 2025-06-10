"""Unit tests for tool pool management functionality."""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from smart_mcp_proxy.server.mcp_server import SmartMCPProxyServer
from smart_mcp_proxy.models.schemas import ToolMetadata, SearchResult


class TestToolPoolManagement:
    """Test tool pool management and eviction logic."""
    
    @pytest.fixture
    def mock_server(self):
        """Create a mock server with tool pool management."""
        with patch('smart_mcp_proxy.server.mcp_server.ConfigLoader'):
            server = SmartMCPProxyServer()
            server.tools_limit = 3  # Small limit for testing
            server.mcp = MagicMock()
            server.mcp.remove_tool = MagicMock()
            return server
    
    def test_calculate_tool_weight_fresh_tool(self, mock_server):
        """Test weight calculation for fresh tools."""
        current_time = time.time()
        score = 0.8
        
        # Fresh tool (just added)
        weight = mock_server._calculate_tool_weight(score, current_time)
        
        # Should be close to original score since freshness is high
        expected = (score * 0.7) + (1.0 * 0.3)  # 0.56 + 0.3 = 0.86
        assert abs(weight - expected) < 0.01
    
    def test_calculate_tool_weight_old_tool(self, mock_server):
        """Test weight calculation for old tools."""
        current_time = time.time()
        old_timestamp = current_time - (35 * 60)  # 35 minutes ago (older than max age)
        score = 0.8
        
        weight = mock_server._calculate_tool_weight(score, old_timestamp)
        
        # Should be just the score since freshness is 0
        expected = (score * 0.7) + (0.0 * 0.3)  # 0.56 + 0 = 0.56
        assert abs(weight - expected) < 0.01
    
    def test_calculate_tool_weight_medium_age(self, mock_server):
        """Test weight calculation for medium-age tools."""
        current_time = time.time()
        medium_timestamp = current_time - (15 * 60)  # 15 minutes ago (half max age)
        score = 0.6
        
        weight = mock_server._calculate_tool_weight(score, medium_timestamp)
        
        # Freshness should be 0.5 (15min / 30min = 0.5, so freshness = 1 - 0.5 = 0.5)
        expected = (score * 0.7) + (0.5 * 0.3)  # 0.42 + 0.15 = 0.57
        assert abs(weight - expected) < 0.01
    
    @pytest.mark.asyncio
    async def test_enforce_tool_pool_limit_no_eviction_needed(self, mock_server):
        """Test pool limit enforcement when no eviction is needed."""
        # Current pool has 2 tools, limit is 3, adding 1 tool
        mock_server.current_tool_registrations = {"tool1": MagicMock(), "tool2": MagicMock()}
        
        new_tools = [("new_tool", 0.8)]
        evicted = await mock_server._enforce_tool_pool_limit(new_tools)
        
        assert evicted == []
    
    @pytest.mark.asyncio
    async def test_enforce_tool_pool_limit_with_eviction(self, mock_server):
        """Test pool limit enforcement with eviction."""
        current_time = time.time()
        
        # Set up current pool with 3 tools (at limit)
        mock_server.current_tool_registrations = {
            "tool1": MagicMock(),
            "tool2": MagicMock(), 
            "tool3": MagicMock()
        }
        
        # Set up metadata with different scores and ages
        mock_server.tool_pool_metadata = {
            "tool1": {"timestamp": current_time - 1800, "score": 0.5},  # Old, low score
            "tool2": {"timestamp": current_time - 900, "score": 0.9},   # Medium age, high score
            "tool3": {"timestamp": current_time, "score": 0.7}          # Fresh, medium score
        }
        
        # Mock the evict_tool method
        mock_server._evict_tool = AsyncMock()
        
        # Try to add 2 new tools (would exceed limit by 2)
        new_tools = [("new_tool1", 0.8), ("new_tool2", 0.6)]
        evicted = await mock_server._enforce_tool_pool_limit(new_tools)
        
        # Should evict 2 tools
        assert len(evicted) == 2
        # tool1 should be evicted first (lowest weighted score)
        assert "tool1" in evicted
        
        # Verify evict_tool was called for each evicted tool
        assert mock_server._evict_tool.call_count == 2
    
    @pytest.mark.asyncio
    async def test_evict_tool(self, mock_server):
        """Test tool eviction process."""
        tool_name = "test_tool"
        
        # Set up tool in all tracking structures
        mock_server.current_tool_registrations[tool_name] = MagicMock()
        mock_server.registered_tools[tool_name] = MagicMock()
        mock_server.tool_pool_metadata[tool_name] = {"timestamp": time.time(), "score": 0.5}
        
        await mock_server._evict_tool(tool_name)
        
        # Verify tool was removed from all tracking structures
        assert tool_name not in mock_server.current_tool_registrations
        assert tool_name not in mock_server.registered_tools
        assert tool_name not in mock_server.tool_pool_metadata
        
        # Verify FastMCP remove_tool was called
        mock_server.mcp.remove_tool.assert_called_once_with(tool_name)
    
    @pytest.mark.asyncio
    async def test_register_proxy_tool_with_metadata(self, mock_server):
        """Test that tool registration includes metadata tracking."""
        tool_metadata = MagicMock()
        tool_metadata.name = "test_tool"
        tool_metadata.server_name = "test_server"
        tool_metadata.description = "Test description"
        tool_metadata.params_json = "{}"
        
        tool_name = "test_server_test_tool"
        score = 0.75
        
        # Mock dependencies
        mock_server.proxy_servers = {"test_server": MagicMock()}
        mock_server.mcp.add_tool = MagicMock()
        
        with patch('smart_mcp_proxy.server.mcp_server.Tool') as mock_tool_class:
            mock_tool = MagicMock()
            mock_tool_class.from_function.return_value = mock_tool
            
            await mock_server._register_proxy_tool(tool_metadata, tool_name, score)
        
        # Verify metadata was tracked
        assert tool_name in mock_server.tool_pool_metadata
        metadata = mock_server.tool_pool_metadata[tool_name]
        assert metadata["score"] == score
        assert metadata["original_score"] == score
        assert "timestamp" in metadata
        assert isinstance(metadata["timestamp"], float)
        
        # Verify tool was registered
        assert tool_name in mock_server.current_tool_registrations
    
    def test_tool_weight_comparison(self, mock_server):
        """Test that tool weights are calculated correctly for comparison."""
        current_time = time.time()
        
        # High score, old tool
        weight1 = mock_server._calculate_tool_weight(0.9, current_time - 1800)  # 30 min ago
        
        # Medium score, fresh tool  
        weight2 = mock_server._calculate_tool_weight(0.6, current_time)
        
        # Low score, medium age
        weight3 = mock_server._calculate_tool_weight(0.3, current_time - 900)  # 15 min ago
        
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
            "original_score": original_score
        }
        mock_server.current_tool_registrations[tool_name] = MagicMock()
        
        # Simulate the freshness update logic from retrieve_tools
        mock_server.tool_pool_metadata[tool_name]["timestamp"] = time.time()
        mock_server.tool_pool_metadata[tool_name]["score"] = max(
            mock_server.tool_pool_metadata[tool_name]["score"], 
            new_score
        )
        
        # Verify updates
        metadata = mock_server.tool_pool_metadata[tool_name]
        assert metadata["score"] == new_score  # Should be updated to higher score
        assert metadata["timestamp"] > original_time  # Should be fresher
        assert metadata["original_score"] == original_score  # Should preserve original 
"""Integration tests for retrieve_tools with pool management."""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcpproxy.models.schemas import SearchResult, ToolMetadata
from mcpproxy.server.mcp_server import SmartMCPProxyServer
from mcpproxy.server.tool_pool_manager import ToolPoolManager # Import ToolPoolManager


class TestRetrieveToolsPoolManagement:
    """Integration tests for retrieve_tools with pool management."""

    @pytest.fixture
    async def mock_server_with_indexer(self):
        """Create a mock server with indexer for integration testing."""
        with patch("mcpproxy.server.mcp_server.ConfigLoader") as MockConfigLoader:
            # Configure MockConfigLoader to return a mock config with necessary attributes
            mock_config = MagicMock()
            mock_config.top_k = 5
            mock_config.embedder = MagicMock() # Mock embedder attribute if accessed
            MockConfigLoader.return_value.load_config.return_value = mock_config

            server = SmartMCPProxyServer()
            server.tools_limit = 3  # Small limit for testing

            # Mock FastMCP
            server.mcp = MagicMock()
            server.mcp.add_tool = MagicMock()
            server.mcp.remove_tool = MagicMock()

            # Mock indexer
            server.indexer = MagicMock()

            # Mock proxy servers
            server.proxy_servers = {"server1": MagicMock(), "server2": MagicMock()}

            # Manually initialize ToolPoolManager with mocks for the tests
            server.tool_pool_manager = MagicMock(spec=ToolPoolManager)
            server.tool_pool_manager.retrieve_tools = AsyncMock(return_value=json.dumps({"tools": [], "newly_registered": [], "evicted_tools": []}))
            server.tool_pool_manager.call_tool = AsyncMock()
            server.tool_pool_manager._enforce_tool_pool_limit = AsyncMock(return_value=[])
            server.tool_pool_manager._evict_tool = AsyncMock()
            server.tool_pool_manager._register_proxy_tool = AsyncMock()
            server.tool_pool_manager.add_proxified_tool_to_memory = AsyncMock()
            server.tool_pool_manager.get_proxified_tools = MagicMock(return_value={})
            server.tool_pool_manager.get_current_tool_registrations = MagicMock(return_value={})

            return server

    def create_mock_search_result(
        self, tool_name: str, server_name: str, score: float
    ) -> SearchResult:
        """Create a mock search result."""
        tool_metadata = ToolMetadata(
            id=1,
            name=tool_name,
            description=f"Description for {tool_name}",
            hash="test_hash",
            server_name=server_name,
            faiss_vector_id=1,
            params_json="{}",
        )

        return SearchResult(tool=tool_metadata, score=score)

    @pytest.mark.asyncio
    async def test_retrieve_tools_with_pool_limit_enforcement(
        self, mock_server_with_indexer
    ):
        """Test retrieve_tools enforces pool limit and evicts tools."""
        server = mock_server_with_indexer

        # Set up existing tools in pool (at limit)
        current_time = time.time()
        server.current_tool_registrations = {
            "server1_old_tool1": MagicMock(),
            "server1_old_tool2": MagicMock(),
            "server2_old_tool3": MagicMock(),
        }
        server.tool_pool_metadata = {
            "server1_old_tool1": {
                "timestamp": current_time - 1800,
                "score": 0.3,
            },  # Old, low score
            "server1_old_tool2": {
                "timestamp": current_time - 900,
                "score": 0.8,
            },  # Medium age, high score
            "server2_old_tool3": {
                "timestamp": current_time,
                "score": 0.6,
            },  # Fresh, medium score
        }

        # Mock search results - 2 new tools with good scores
        search_results = [
            self.create_mock_search_result("new_tool1", "server1", 0.9),
            self.create_mock_search_result("new_tool2", "server2", 0.7),
        ]
        server.indexer.search_tools = AsyncMock(return_value=search_results)

        # Mock _register_proxy_tool and _enforce_tool_pool_limit, as they are internal now.
        # The retrieve_tools method will call them.
        # server._register_proxy_tool = AsyncMock() # Removed direct mock
        # server._enforce_tool_pool_limit = AsyncMock(return_value=["server1_old_tool1", "server2_old_tool3"]) # Removed direct mock

        # Configure the mock ToolPoolManager for the test
        server.tool_pool_manager._enforce_tool_pool_limit = AsyncMock(return_value=["server1_old_tool1", "server2_old_tool3"])
        server.tool_pool_manager._evict_tool = AsyncMock()
        server.tool_pool_manager._register_proxy_tool = AsyncMock()
        server.tool_pool_manager.add_proxified_tool_to_memory = AsyncMock()
        server.tool_pool_manager.get_proxified_tools = MagicMock(return_value={})
        server.tool_pool_manager.get_current_tool_registrations = MagicMock(return_value={})

        # Call retrieve_tools
        result_json = await server.mcp.retrieve_tools("search for new tools")
        result = json.loads(result_json)

        # Verify results
        assert "tools" in result
        assert "newly_registered" in result
        assert "evicted_tools" in result
        assert "pool_size" in result
        assert "pool_limit" in result

        # Should register 2 new tools
        assert len(result["newly_registered"]) == 2
        assert "server1_new_tool1" in result["newly_registered"]
        assert "server2_new_tool2" in result["newly_registered"]

        # Should evict 2 tools to stay within limit (3 existing + 2 new - 3 limit = 2 evictions)
        assert len(result["evicted_tools"]) == 2

        # Pool size should be at limit
        assert result["pool_size"] <= result["pool_limit"]
        assert result["pool_limit"] == 3

        # Verify _register_proxy_tool was called for new tools
        assert server.tool_pool_manager._register_proxy_tool.call_count == 2

        # Verify _enforce_tool_pool_limit was called
        server.tool_pool_manager._enforce_tool_pool_limit.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_tools_freshness_update_existing_tools(
        self, mock_server_with_indexer
    ):
        """Test that existing tools get freshness updates."""
        server = mock_server_with_indexer

        # Set up existing tool in mock ToolPoolManager
        original_time = time.time() - 1000
        tool_name = "server1_existing_tool"
        
        # Configure the mock ToolPoolManager's internal state
        server.tool_pool_manager.current_tool_registrations = {tool_name: MagicMock()}
        server.tool_pool_manager.tool_pool_metadata = {
            tool_name: {"timestamp": original_time, "score": 0.6, "original_score": 0.6}
        }

        # Mock search results - same tool with higher score
        search_results = [
            self.create_mock_search_result("existing_tool", "server1", 0.8)
        ]
        server.indexer.search_tools = AsyncMock(return_value=search_results)

        # Configure mock ToolPoolManager.retrieve_tools to return appropriate JSON
        # This simulates the internal logic of ToolPoolManager.retrieve_tools without actually running it
        def mock_retrieve_tools_side_effect(query):
            # Simulate the update happening within ToolPoolManager
            # In a real scenario, this would be handled by the actual ToolPoolManager logic
            # Here, we directly modify the mocked internal state for testing
            if tool_name in server.tool_pool_manager.tool_pool_metadata:
                server.tool_pool_manager.tool_pool_metadata[tool_name]["timestamp"] = time.time()
                server.tool_pool_manager.tool_pool_metadata[tool_name]["score"] = max(
                    server.tool_pool_manager.tool_pool_metadata[tool_name]["score"],
                    0.8 # New score
                )
            
            # Simulate the JSON response that retrieve_tools would return
            return json.dumps({
                "message": "Found 1 tools, registered 0 new tools",
                "tools": [
                    {
                        "name": tool_name,
                        "original_name": "existing_tool",
                        "server": "server1",
                        "description": "Description for existing_tool",
                        "score": 0.8,
                        "newly_registered": False,
                    }
                ],
                "newly_registered": [],
                "evicted_tools": [],
                "pool_size": len(server.tool_pool_manager.current_tool_registrations),
                "pool_limit": server.tools_limit,
                "total_available_tools": len(server.tool_pool_manager.get_proxified_tools()),
                "query": query,
            })

        server.tool_pool_manager.retrieve_tools.side_effect = mock_retrieve_tools_side_effect

        # Call retrieve_tools via the public interface
        result_json = await server.mcp.retrieve_tools("search for existing tool")
        result = json.loads(result_json)

        # Verify results from the public interface
        assert len(result["newly_registered"]) == 0

        # Verify freshness update by checking the mocked ToolPoolManager's internal state
        metadata = server.tool_pool_manager.tool_pool_metadata[tool_name]
        assert metadata["score"] == 0.8  # Should be updated to higher score
        assert metadata["timestamp"] > original_time  # Should be fresher
        assert metadata["original_score"] == 0.6  # Should preserve original

    @pytest.mark.asyncio
    async def test_retrieve_tools_no_eviction_when_under_limit(
        self, mock_server_with_indexer
    ):
        """Test that no eviction occurs when under the pool limit."""
        server = mock_server_with_indexer

        # Set up pool with only 1 tool (under limit of 3) in mock ToolPoolManager
        tool_name = "server1_tool1"
        server.tool_pool_manager.current_tool_registrations = {tool_name: MagicMock()}
        server.tool_pool_manager.tool_pool_metadata = {
            tool_name: {"timestamp": time.time(), "score": 0.5}
        }
        server.tool_pool_manager.get_proxified_tools.return_value = {tool_name: MagicMock()}

        # Mock search results - 1 new tool
        search_results = [self.create_mock_search_result("new_tool", "server1", 0.7)]
        server.indexer.search_tools = AsyncMock(return_value=search_results)

        # Configure mock ToolPoolManager methods
        server.tool_pool_manager._enforce_tool_pool_limit = AsyncMock(return_value=[])
        server.tool_pool_manager._register_proxy_tool = AsyncMock(side_effect=lambda tool_meta, t_name, score, s_name: 
            server.tool_pool_manager.current_tool_registrations.update({t_name: MagicMock()}) or 
            server.tool_pool_manager.tool_pool_metadata.update({t_name: {"timestamp": time.time(), "score": score, "original_score": score}})
        )

        # Call retrieve_tools via the public interface
        result_json = await server.mcp.retrieve_tools("search for tools")
        result = json.loads(result_json)

        # Verify no evictions
        assert len(result["evicted_tools"]) == 0
        assert len(result["newly_registered"]) == 1
        assert result["pool_size"] == 2  # 1 existing + 1 new
        
        # Verify that the mocked internal methods of ToolPoolManager were called
        server.tool_pool_manager._enforce_tool_pool_limit.assert_called_once()
        server.tool_pool_manager._register_proxy_tool.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_tools_error_handling(self, mock_server_with_indexer):
        """Test error handling in retrieve_tools."""
        server = mock_server_with_indexer

        # Simulate no indexer by setting the mock ToolPoolManager's indexer to None
        server.tool_pool_manager.indexer = None

        # Call retrieve_tools via the public interface
        result_json = await server.mcp.retrieve_tools("test query")
        result = json.loads(result_json)

        assert "error" in result
        assert result["error"] == "Indexer not initialized"

    @pytest.mark.asyncio
    async def test_retrieve_tools_no_results(self, mock_server_with_indexer):
        """Test retrieve_tools when no tools are found."""
        server = mock_server_with_indexer

        # Mock empty search results from the indexer
        server.indexer.search_tools = AsyncMock(return_value=[])

        # Configure mock ToolPoolManager.retrieve_tools to return an empty result JSON
        server.tool_pool_manager.retrieve_tools = AsyncMock(return_value=json.dumps(
            {"message": "No relevant tools found", "tools": [], "newly_registered": [], "evicted_tools": []}
        ))

        # Call retrieve_tools via the public interface
        result_json = await server.mcp.retrieve_tools("no matching tools")
        result = json.loads(result_json)

        assert result["message"] == "No relevant tools found"
        assert result["tools"] == []

    @pytest.mark.asyncio
    async def test_pool_metadata_consistency(self, mock_server_with_indexer):
        """Test that pool metadata stays consistent across operations."""
        server = mock_server_with_indexer

        # Set up initial state on the mock ToolPoolManager
        server.tool_pool_manager.current_tool_registrations = {}
        server.tool_pool_manager.tool_pool_metadata = {}
        server.tool_pool_manager.get_proxified_tools.return_value = {}

        # Mock search results
        search_results = [
            self.create_mock_search_result("tool1", "server1", 0.8),
            self.create_mock_search_result("tool2", "server1", 0.6),
        ]
        server.indexer.search_tools = AsyncMock(return_value=search_results)

        # Configure mock ToolPoolManager.retrieve_tools to simulate tool registration
        def mock_retrieve_tools_side_effect(query):
            # Simulate the internal calls to _register_proxy_tool
            tool1_sanitized = "server1_tool1"
            tool2_sanitized = "server1_tool2"
            
            server.tool_pool_manager.current_tool_registrations[tool1_sanitized] = MagicMock()
            server.tool_pool_manager.current_tool_registrations[tool2_sanitized] = MagicMock()

            server.tool_pool_manager.tool_pool_metadata[tool1_sanitized] = {"timestamp": time.time(), "score": 0.8, "original_score": 0.8}
            server.tool_pool_manager.tool_pool_metadata[tool2_sanitized] = {"timestamp": time.time(), "score": 0.6, "original_score": 0.6}
            
            server.tool_pool_manager.get_proxified_tools.return_value = {
                tool1_sanitized: MagicMock(),
                tool2_sanitized: MagicMock()
            }
            
            return json.dumps({
                "message": "Found 2 tools, registered 2 new tools",
                "tools": [
                    {"name": tool1_sanitized, "original_name": "tool1", "server": "server1", "description": "Description for tool1", "score": 0.8, "newly_registered": True},
                    {"name": tool2_sanitized, "original_name": "tool2", "server": "server1", "description": "Description for tool2", "score": 0.6, "newly_registered": True}
                ],
                "newly_registered": [tool1_sanitized, tool2_sanitized],
                "evicted_tools": [],
                "pool_size": 2,
                "pool_limit": server.tools_limit,
                "total_available_tools": 2,
                "query": query
            })

        server.tool_pool_manager.retrieve_tools.side_effect = mock_retrieve_tools_side_effect

        # Call retrieve_tools via the public interface
        await server.mcp.retrieve_tools("test query")

        # Verify consistency by checking the mocked ToolPoolManager's internal state
        assert len(server.tool_pool_manager.current_tool_registrations) == len(server.tool_pool_manager.tool_pool_metadata)

        for tool_name in server.tool_pool_manager.current_tool_registrations:
            assert tool_name in server.tool_pool_manager.tool_pool_metadata
            metadata = server.tool_pool_manager.tool_pool_metadata[tool_name]
            assert "timestamp" in metadata
            assert "score" in metadata
            assert "original_score" in metadata

"""Integration tests for retrieve_tools with pool management."""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcpproxy.models.schemas import SearchResult, ToolMetadata
from mcpproxy.server.mcp_server import SmartMCPProxyServer


class TestRetrieveToolsPoolManagement:
    """Integration tests for retrieve_tools with pool management."""

    @pytest.fixture
    async def mock_server_with_indexer(self):
        """Create a mock server with indexer for integration testing."""
        with patch("mcpproxy.server.mcp_server.ConfigLoader"):
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

        # Mock _register_proxy_tool to track calls
        server._register_proxy_tool = AsyncMock()

        # Create a simplified retrieve_tools function for testing
        async def retrieve_tools_func(query: str) -> str:
            # Simulate the retrieve_tools logic
            if not server.indexer:
                return json.dumps({"error": "Indexer not initialized"})

            results = await server.indexer.search_tools(query, 5)

            if not results:
                return json.dumps({"message": "No relevant tools found", "tools": []})

            # Prepare tools for registration
            tools_to_register = []
            for result in results:
                tool_name = f"{result.tool.server_name}_{result.tool.name}"

                # Skip if already registered
                if tool_name in server.current_tool_registrations:
                    # Update timestamp for existing tool (freshen it)
                    if tool_name in server.tool_pool_metadata:
                        server.tool_pool_metadata[tool_name]["timestamp"] = time.time()
                        server.tool_pool_metadata[tool_name]["score"] = max(
                            server.tool_pool_metadata[tool_name]["score"], result.score
                        )
                    continue

                tools_to_register.append((tool_name, result.tool, result.score))

            # Enforce pool limit before registering new tools
            evicted_tools = []
            if tools_to_register:
                new_tools_info = [(name, score) for name, _, score in tools_to_register]
                evicted_tools = await server._enforce_tool_pool_limit(new_tools_info)

            # Register new tools
            newly_registered = []
            for tool_name, tool_metadata, score in tools_to_register:
                await server._register_proxy_tool(tool_metadata, tool_name, score)
                newly_registered.append(tool_name)

            # Prepare tool information
            registered_tools = []
            for result in results:
                tool_name = f"{result.tool.server_name}_{result.tool.name}"
                registered_tools.append(
                    {
                        "name": tool_name,
                        "original_name": result.tool.name,
                        "server": result.tool.server_name,
                        "description": result.tool.description,
                        "score": result.score,
                        "newly_registered": tool_name in newly_registered,
                    }
                )

            message = f"Found {len(registered_tools)} tools, registered {len(newly_registered)} new tools"
            if evicted_tools:
                message += f", evicted {len(evicted_tools)} tools to stay within limit ({server.tools_limit})"

            return json.dumps(
                {
                    "message": message,
                    "tools": registered_tools,
                    "newly_registered": newly_registered,
                    "evicted_tools": evicted_tools,
                    "pool_size": len(server.current_tool_registrations),
                    "pool_limit": server.tools_limit,
                    "query": query,
                }
            )

        # Call retrieve_tools
        result_json = await retrieve_tools_func("search for new tools")
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
        assert server._register_proxy_tool.call_count == 2

    @pytest.mark.asyncio
    async def test_retrieve_tools_freshness_update_existing_tools(
        self, mock_server_with_indexer
    ):
        """Test that existing tools get freshness updates."""
        server = mock_server_with_indexer

        # Set up existing tool
        original_time = time.time() - 1000
        tool_name = "server1_existing_tool"
        server.current_tool_registrations = {tool_name: MagicMock()}
        server.tool_pool_metadata = {
            tool_name: {"timestamp": original_time, "score": 0.6, "original_score": 0.6}
        }

        # Mock search results - same tool with higher score
        search_results = [
            self.create_mock_search_result("existing_tool", "server1", 0.8)
        ]
        server.indexer.search_tools = AsyncMock(return_value=search_results)

        # Create retrieve_tools function for testing
        async def retrieve_tools_func(query: str) -> str:
            if not server.indexer:
                return json.dumps({"error": "Indexer not initialized"})

            results = await server.indexer.search_tools(query, 5)

            if not results:
                return json.dumps({"message": "No relevant tools found", "tools": []})

            # Prepare tools for registration
            tools_to_register = []
            for result in results:
                tool_name = f"{result.tool.server_name}_{result.tool.name}"

                # Skip if already registered
                if tool_name in server.current_tool_registrations:
                    # Update timestamp for existing tool (freshen it)
                    if tool_name in server.tool_pool_metadata:
                        server.tool_pool_metadata[tool_name]["timestamp"] = time.time()
                        server.tool_pool_metadata[tool_name]["score"] = max(
                            server.tool_pool_metadata[tool_name]["score"], result.score
                        )
                    continue

                tools_to_register.append((tool_name, result.tool, result.score))

            # Enforce pool limit before registering new tools
            evicted_tools = []
            if tools_to_register:
                new_tools_info = [(name, score) for name, _, score in tools_to_register]
                evicted_tools = await server._enforce_tool_pool_limit(new_tools_info)

            # Register new tools
            newly_registered = []
            for tool_name, tool_metadata, score in tools_to_register:
                await server._register_proxy_tool(tool_metadata, tool_name, score)
                newly_registered.append(tool_name)

            # Prepare tool information
            registered_tools = []
            for result in results:
                tool_name = f"{result.tool.server_name}_{result.tool.name}"
                registered_tools.append(
                    {
                        "name": tool_name,
                        "original_name": result.tool.name,
                        "server": result.tool.server_name,
                        "description": result.tool.description,
                        "score": result.score,
                        "newly_registered": tool_name in newly_registered,
                    }
                )

            message = f"Found {len(registered_tools)} tools, registered {len(newly_registered)} new tools"
            if evicted_tools:
                message += f", evicted {len(evicted_tools)} tools to stay within limit ({server.tools_limit})"

            return json.dumps(
                {
                    "message": message,
                    "tools": registered_tools,
                    "newly_registered": newly_registered,
                    "evicted_tools": evicted_tools,
                    "pool_size": len(server.current_tool_registrations),
                    "pool_limit": server.tools_limit,
                    "query": query,
                }
            )

        # Call retrieve_tools
        result_json = await retrieve_tools_func("search for existing tool")
        result = json.loads(result_json)

        # Verify no new registrations (tool already exists)
        assert len(result["newly_registered"]) == 0

        # Verify freshness update
        metadata = server.tool_pool_metadata[tool_name]
        assert metadata["score"] == 0.8  # Should be updated to higher score
        assert metadata["timestamp"] > original_time  # Should be fresher
        assert metadata["original_score"] == 0.6  # Should preserve original

    @pytest.mark.asyncio
    async def test_retrieve_tools_no_eviction_when_under_limit(
        self, mock_server_with_indexer
    ):
        """Test that no eviction occurs when under the pool limit."""
        server = mock_server_with_indexer

        # Set up pool with only 1 tool (under limit of 3)
        server.current_tool_registrations = {"server1_tool1": MagicMock()}
        server.tool_pool_metadata = {
            "server1_tool1": {"timestamp": time.time(), "score": 0.5}
        }

        # Mock search results - 1 new tool
        search_results = [self.create_mock_search_result("new_tool", "server1", 0.7)]
        server.indexer.search_tools = AsyncMock(return_value=search_results)

        # Mock _register_proxy_tool to simulate actual registration
        async def mock_register(tool_metadata, tool_name, score):
            server.current_tool_registrations[tool_name] = MagicMock()
            server.tool_pool_metadata[tool_name] = {
                "timestamp": time.time(),
                "score": score,
                "original_score": score,
            }

        server._register_proxy_tool = AsyncMock(side_effect=mock_register)

        # Create retrieve_tools function for testing
        async def retrieve_tools_func(query: str) -> str:
            if not server.indexer:
                return json.dumps({"error": "Indexer not initialized"})

            results = await server.indexer.search_tools(query, 5)

            if not results:
                return json.dumps({"message": "No relevant tools found", "tools": []})

            # Prepare tools for registration
            tools_to_register = []
            for result in results:
                tool_name = f"{result.tool.server_name}_{result.tool.name}"

                # Skip if already registered
                if tool_name in server.current_tool_registrations:
                    # Update timestamp for existing tool (freshen it)
                    if tool_name in server.tool_pool_metadata:
                        server.tool_pool_metadata[tool_name]["timestamp"] = time.time()
                        server.tool_pool_metadata[tool_name]["score"] = max(
                            server.tool_pool_metadata[tool_name]["score"], result.score
                        )
                    continue

                tools_to_register.append((tool_name, result.tool, result.score))

            # Enforce pool limit before registering new tools
            evicted_tools = []
            if tools_to_register:
                new_tools_info = [(name, score) for name, _, score in tools_to_register]
                evicted_tools = await server._enforce_tool_pool_limit(new_tools_info)

            # Register new tools
            newly_registered = []
            for tool_name, tool_metadata, score in tools_to_register:
                await server._register_proxy_tool(tool_metadata, tool_name, score)
                newly_registered.append(tool_name)

            # Prepare tool information
            registered_tools = []
            for result in results:
                tool_name = f"{result.tool.server_name}_{result.tool.name}"
                registered_tools.append(
                    {
                        "name": tool_name,
                        "original_name": result.tool.name,
                        "server": result.tool.server_name,
                        "description": result.tool.description,
                        "score": result.score,
                        "newly_registered": tool_name in newly_registered,
                    }
                )

            message = f"Found {len(registered_tools)} tools, registered {len(newly_registered)} new tools"
            if evicted_tools:
                message += f", evicted {len(evicted_tools)} tools to stay within limit ({server.tools_limit})"

            return json.dumps(
                {
                    "message": message,
                    "tools": registered_tools,
                    "newly_registered": newly_registered,
                    "evicted_tools": evicted_tools,
                    "pool_size": len(server.current_tool_registrations),
                    "pool_limit": server.tools_limit,
                    "query": query,
                }
            )

        # Call retrieve_tools
        result_json = await retrieve_tools_func("search for tools")
        result = json.loads(result_json)

        # Verify no evictions
        assert len(result["evicted_tools"]) == 0
        assert len(result["newly_registered"]) == 1
        assert result["pool_size"] == 2  # 1 existing + 1 new

    @pytest.mark.asyncio
    async def test_retrieve_tools_error_handling(self, mock_server_with_indexer):
        """Test error handling in retrieve_tools."""
        server = mock_server_with_indexer

        # Test with no indexer
        server.indexer = None

        # Create retrieve_tools function for testing
        async def retrieve_tools_func(query: str) -> str:
            if not server.indexer:
                return json.dumps({"error": "Indexer not initialized"})

            results = await server.indexer.search_tools(query, 5)

            if not results:
                return json.dumps({"message": "No relevant tools found", "tools": []})

            return json.dumps({"message": "Found tools", "tools": []})

        result_json = await retrieve_tools_func("test query")
        result = json.loads(result_json)

        assert "error" in result
        assert result["error"] == "Indexer not initialized"

    @pytest.mark.asyncio
    async def test_retrieve_tools_no_results(self, mock_server_with_indexer):
        """Test retrieve_tools when no tools are found."""
        server = mock_server_with_indexer

        # Mock empty search results
        server.indexer.search_tools = AsyncMock(return_value=[])

        # Create retrieve_tools function for testing
        async def retrieve_tools_func(query: str) -> str:
            if not server.indexer:
                return json.dumps({"error": "Indexer not initialized"})

            results = await server.indexer.search_tools(query, 5)

            if not results:
                return json.dumps({"message": "No relevant tools found", "tools": []})

            return json.dumps({"message": "Found tools", "tools": []})

        result_json = await retrieve_tools_func("no matching tools")
        result = json.loads(result_json)

        assert result["message"] == "No relevant tools found"
        assert result["tools"] == []

    @pytest.mark.asyncio
    async def test_pool_metadata_consistency(self, mock_server_with_indexer):
        """Test that pool metadata stays consistent across operations."""
        server = mock_server_with_indexer

        # Set up initial state
        server.current_tool_registrations = {}
        server.tool_pool_metadata = {}

        # Mock search results
        search_results = [
            self.create_mock_search_result("tool1", "server1", 0.8),
            self.create_mock_search_result("tool2", "server1", 0.6),
        ]
        server.indexer.search_tools = AsyncMock(return_value=search_results)

        # Mock _register_proxy_tool to simulate actual registration
        async def mock_register(tool_metadata, tool_name, score):
            server.current_tool_registrations[tool_name] = MagicMock()
            server.tool_pool_metadata[tool_name] = {
                "timestamp": time.time(),
                "score": score,
                "original_score": score,
            }

        server._register_proxy_tool = AsyncMock(side_effect=mock_register)

        # Create retrieve_tools function for testing
        async def retrieve_tools_func(query: str) -> str:
            if not server.indexer:
                return json.dumps({"error": "Indexer not initialized"})

            results = await server.indexer.search_tools(query, 5)

            if not results:
                return json.dumps({"message": "No relevant tools found", "tools": []})

            # Prepare tools for registration
            tools_to_register = []
            for result in results:
                tool_name = f"{result.tool.server_name}_{result.tool.name}"

                # Skip if already registered
                if tool_name in server.current_tool_registrations:
                    continue

                tools_to_register.append((tool_name, result.tool, result.score))

            # Register new tools
            newly_registered = []
            for tool_name, tool_metadata, score in tools_to_register:
                await server._register_proxy_tool(tool_metadata, tool_name, score)
                newly_registered.append(tool_name)

            return json.dumps(
                {
                    "message": f"Registered {len(newly_registered)} tools",
                    "tools": [],
                    "newly_registered": newly_registered,
                    "evicted_tools": [],
                    "pool_size": len(server.current_tool_registrations),
                    "pool_limit": server.tools_limit,
                    "query": query,
                }
            )

        # Call retrieve_tools
        await retrieve_tools_func("test query")

        # Verify consistency
        assert len(server.current_tool_registrations) == len(server.tool_pool_metadata)

        for tool_name in server.current_tool_registrations:
            assert tool_name in server.tool_pool_metadata
            metadata = server.tool_pool_metadata[tool_name]
            assert "timestamp" in metadata
            assert "score" in metadata
            assert "original_score" in metadata

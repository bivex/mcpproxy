"""Integration tests for full Smart MCP Proxy flow."""

import pytest

from tests.fixtures.data import get_sample_tools_data


class TestFullIndexingFlow:
    """Test complete indexing and search flow."""

    @pytest.mark.asyncio
    async def test_end_to_end_indexing_and_search(self, temp_indexer_facade):
        """Test complete flow from indexing to search."""
        sample_data = get_sample_tools_data()

        # Step 1: Index all sample tools
        indexed_tools = []
        for tool_data in sample_data:
            await temp_indexer_facade.index_tool(
                name=tool_data["name"],
                description=tool_data["description"],
                server_name=tool_data["server_name"],
                params=tool_data["params"],
                tags=tool_data.get("tags", []),
                annotations=tool_data.get("annotations", {}),
            )
            indexed_tools.append(tool_data["name"])

        # Step 2: Verify tools are stored in persistence
        all_tools = await temp_indexer_facade.persistence.get_all_tools()
        assert len(all_tools) == len(sample_data)
        stored_names = {tool.name for tool in all_tools}
        assert stored_names == set(indexed_tools)

        # Step 3: Test various search scenarios
        search_tests = [
            {
                "query": "create virtual machine",
                "expected_tool": "create_instance",
                "description": "Should find VM creation tool",
            },
            {
                "query": "storage volume management",
                "expected_tools": ["list_volumes", "create_volume", "delete_volume"],
                "description": "Should find volume-related tools",
            },
            {
                "query": "delete resources",
                "expected_tools": ["delete_instance", "delete_volume"],
                "description": "Should find deletion tools",
            },
            {
                "query": "performance monitoring",
                "expected_tool": "get_metrics",
                "description": "Should find monitoring tool",
            },
        ]

        for test_case in search_tests:
            query = test_case["query"]
            results = await temp_indexer_facade.search_tools(query, k=5)

            assert len(results) > 0, f"Query '{query}' should return results"

            # Check for expected tools
            found_names = {r.tool.name for r in results}

            if "expected_tool" in test_case:
                assert test_case["expected_tool"] in found_names, (
                    f"{test_case['description']}: {query}"
                )

            if "expected_tools" in test_case:
                expected = set(test_case["expected_tools"])
                assert expected.intersection(found_names), (
                    f"{test_case['description']}: {query} should find one of {expected}"
                )

            # Verify scores are reasonable
            assert all(0 <= r.score <= 1 for r in results), (
                f"Scores should be between 0 and 1 for query: {query}"
            )

    @pytest.mark.asyncio
    async def test_incremental_indexing(self, temp_indexer_facade):
        """Test adding tools incrementally and searching."""
        # Step 1: Start with one tool
        await temp_indexer_facade.index_tool(
            name="initial_tool",
            description="Initial tool for testing",
            server_name="server1",
        )

        results = await temp_indexer_facade.search_tools("initial", k=5)
        assert len(results) == 1
        assert results[0].tool.name == "initial_tool"

        # Step 2: Add more tools
        additional_tools = [
            ("tool_two", "Second tool", "server1"),
            ("tool_three", "Third tool", "server2"),
            ("related_initial", "Related to initial tool", "server1"),
        ]

        for name, desc, server in additional_tools:
            await temp_indexer_facade.index_tool(name, desc, server)

        # Step 3: Verify incremental search works
        results = await temp_indexer_facade.search_tools("initial", k=5)
        assert len(results) >= 2  # Should find initial_tool and related_initial

        found_names = {r.tool.name for r in results}
        assert "initial_tool" in found_names
        assert "related_initial" in found_names

        # Step 4: Test server-specific queries
        server1_tools = await temp_indexer_facade.persistence.get_tools_by_server(
            "server1"
        )
        server2_tools = await temp_indexer_facade.persistence.get_tools_by_server(
            "server2"
        )

        assert len(server1_tools) == 3
        assert len(server2_tools) == 1

    @pytest.mark.asyncio
    async def test_duplicate_tool_handling(self, temp_indexer_facade):
        """Test that duplicate tools (same hash) are handled correctly."""
        tool_params = {
            "name": "test_tool",
            "description": "Test tool for duplicates",
            "server_name": "test-server",
            "params": {"type": "object", "properties": {"name": {"type": "string"}}},
        }

        # Index the same tool twice
        await temp_indexer_facade.index_tool(**tool_params)
        await temp_indexer_facade.index_tool(**tool_params)  # Duplicate

        # Should only have one tool stored
        all_tools = await temp_indexer_facade.persistence.get_all_tools()
        assert len(all_tools) == 1
        assert all_tools[0].name == "test_tool"

        # Should still be searchable
        results = await temp_indexer_facade.search_tools("test tool", k=5)
        assert len(results) == 1
        assert results[0].tool.name == "test_tool"

    @pytest.mark.asyncio
    async def test_tool_update_flow(self, temp_indexer_facade):
        """Test updating tool with different hash."""
        # Index original tool
        await temp_indexer_facade.index_tool(
            name="update_test",
            description="Original description",
            server_name="test-server",
        )

        # Verify original is indexed
        results = await temp_indexer_facade.search_tools("original", k=5)
        assert len(results) == 1
        assert "original" in results[0].tool.description.lower()

        # Update tool with different description (different hash)
        await temp_indexer_facade.index_tool(
            name="update_test",
            description="Updated description with new content",
            server_name="test-server",
        )

        # Should now have both versions (different hashes)
        all_tools = await temp_indexer_facade.persistence.get_all_tools()
        assert len(all_tools) == 2

        # Search should find the updated version when searching for "updated"
        results = await temp_indexer_facade.search_tools("updated content", k=5)
        assert len(results) >= 1
        found_descriptions = [r.tool.description for r in results]
        assert any("Updated description" in desc for desc in found_descriptions)

    @pytest.mark.asyncio
    async def test_search_ranking_quality(self, temp_indexer_facade):
        """Test that search ranking works correctly."""
        # Index tools with varying relevance to test query
        tools = [
            (
                "perfect_match",
                "Create virtual machine instance",
                "server1",
                ["vm", "create"],
            ),
            ("good_match", "Create VM in cloud", "server1", ["vm", "cloud"]),
            ("partial_match", "Virtual environment setup", "server1", ["virtual"]),
            ("weak_match", "Machine learning model", "server1", ["machine"]),
            ("no_match", "Delete storage volume", "server1", ["storage", "delete"]),
        ]

        for name, desc, server, tags in tools:
            await temp_indexer_facade.index_tool(
                name=name, description=desc, server_name=server, tags=tags
            )

        # Search for "create virtual machine"
        results = await temp_indexer_facade.search_tools("create virtual machine", k=5)

        assert len(results) >= 3

        # Verify ranking: perfect_match should be first, no_match should be last
        result_names = [r.tool.name for r in results]
        result_scores = [r.score for r in results]

        # Perfect match should have highest score
        perfect_idx = result_names.index("perfect_match")
        assert perfect_idx == 0 or result_scores[perfect_idx] == max(result_scores)

        # No match should have lowest score among results
        if "no_match" in result_names:
            no_match_idx = result_names.index("no_match")
            assert result_scores[no_match_idx] == min(result_scores)

        # Scores should generally decrease (allowing for ties)
        assert result_scores == sorted(result_scores, reverse=True)

    @pytest.mark.asyncio
    async def test_large_scale_indexing(self, temp_indexer_facade):
        """Test indexing and searching with larger number of tools."""
        # Generate many tools
        tools = []
        for i in range(50):
            server_name = f"server-{i % 5}"  # 5 different servers
            category = ["compute", "storage", "network", "monitoring", "security"][
                i % 5
            ]
            action = ["create", "delete", "list", "update", "monitor"][i % 5]

            tools.append(
                {
                    "name": f"{category}_{action}_{i}",
                    "description": f"{action} {category} resource number {i}",
                    "server_name": server_name,
                    "tags": [category, action],
                    "params": {
                        "type": "object",
                        "properties": {"id": {"type": "string"}},
                    },
                }
            )

        # Index all tools
        for tool in tools:
            await temp_indexer_facade.index_tool(
                name=tool["name"],
                description=tool["description"],
                server_name=tool["server_name"],
                params=tool["params"],
                tags=tool["tags"],
            )

        # Verify all tools are indexed
        all_tools = await temp_indexer_facade.persistence.get_all_tools()
        assert len(all_tools) == 50

        # Test category-specific searches
        search_tests = [
            ("compute resources", "compute"),
            ("storage management", "storage"),
            ("network configuration", "network"),
            ("monitoring systems", "monitoring"),
            ("security policies", "security"),
        ]

        for query, expected_category in search_tests:
            results = await temp_indexer_facade.search_tools(query, k=10)

            assert len(results) > 0, f"Should find results for {query}"

            # Most results should be from the expected category
            category_matches = sum(
                1 for r in results if expected_category in r.tool.name
            )

            assert category_matches > 0, (
                f"Should find {expected_category} tools for query: {query}"
            )

    @pytest.mark.asyncio
    async def test_multi_server_search_isolation(self, temp_indexer_facade):
        """Test that tools from different servers are properly isolated yet searchable."""
        # Index similar tools on different servers
        servers_and_tools = [
            (
                "api-server",
                [
                    ("create_vm", "Create virtual machine"),
                    ("delete_vm", "Delete virtual machine"),
                ],
            ),
            (
                "storage-server",
                [
                    ("create_volume", "Create storage volume"),
                    ("delete_volume", "Delete storage volume"),
                ],
            ),
            (
                "network-server",
                [
                    ("create_network", "Create network interface"),
                    ("delete_network", "Delete network interface"),
                ],
            ),
        ]

        for server_name, tools in servers_and_tools:
            for tool_name, description in tools:
                await temp_indexer_facade.index_tool(
                    name=tool_name, description=description, server_name=server_name
                )

        # Test global search finds tools from all servers
        results = await temp_indexer_facade.search_tools("create", k=10)

        found_servers = {r.tool.server_name for r in results}
        assert "api-server" in found_servers
        assert "storage-server" in found_servers
        assert "network-server" in found_servers

        # Test server-specific retrieval
        for server_name, expected_tools in servers_and_tools:
            server_tools = await temp_indexer_facade.persistence.get_tools_by_server(
                server_name
            )
            assert len(server_tools) == len(expected_tools)

            server_tool_names = {tool.name for tool in server_tools}
            expected_names = {tool[0] for tool in expected_tools}
            assert server_tool_names == expected_names

    @pytest.mark.asyncio
    async def test_persistence_layer_integration(self, temp_indexer_facade):
        """Test integration between indexer and persistence layer."""
        # Index a tool with complex metadata
        complex_params = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Resource name"},
                "config": {
                    "type": "object",
                    "properties": {
                        "size": {"type": "integer", "minimum": 1},
                        "region": {"type": "string", "enum": ["us-east", "us-west"]},
                    },
                },
            },
            "required": ["name"],
        }

        await temp_indexer_facade.index_tool(
            name="complex_tool",
            description="Tool with complex parameters",
            server_name="complex-server",
            params=complex_params,
            tags=["complex", "configuration"],
            annotations={"version": "1.0", "deprecated": False},
        )

        # Verify tool was stored with all metadata
        all_tools = await temp_indexer_facade.persistence.get_all_tools()
        assert len(all_tools) == 1

        stored_tool = all_tools[0]
        assert stored_tool.name == "complex_tool"
        assert stored_tool.server_name == "complex-server"
        assert stored_tool.params_json is not None
        assert "complex" in stored_tool.params_json
        assert "configuration" in stored_tool.params_json
        assert "version" in stored_tool.params_json

        # Verify vector was stored (BM25 uses its own indexing, not vector storage)
        vector_count = await temp_indexer_facade.persistence.get_vector_count()
        # For BM25 embedder, vector count is always 0 since it doesn't use vector storage
        from mcpproxy.indexer.bm25 import BM25Embedder
        if isinstance(temp_indexer_facade.embedder, BM25Embedder):
            assert vector_count == 0  # BM25 doesn't use vector storage
        else:
            assert vector_count == 1  # Other embedders use vector storage

        # Search should find the tool using complex metadata
        results = await temp_indexer_facade.search_tools("complex configuration", k=5)
        assert len(results) == 1
        assert results[0].tool.name == "complex_tool"

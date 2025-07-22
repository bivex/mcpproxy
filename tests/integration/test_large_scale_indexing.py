"""Integration tests for large scale indexing."""

import pytest


class TestLargeScaleIndexing:
    """Test indexing and searching with larger number of tools."""

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

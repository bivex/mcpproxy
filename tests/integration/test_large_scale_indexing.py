"""Integration tests for large scale indexing."""

import pytest
from mcpproxy.models.schemas import ToolData


class TestLargeScaleIndexing:
    """Test indexing and searching with larger number of tools."""

    @pytest.mark.asyncio
    async def test_large_scale_indexing(self, temp_indexer_facade):
        """Test indexing and searching with larger number of tools."""
        # Generate many tools
        NUM_CATEGORIES = 5
        tools = []
        for i in range(50):
            server_name = f"server-{i % NUM_CATEGORIES}"  # 5 different servers
            category = ["compute", "storage", "network", "monitoring", "security"][
                i % NUM_CATEGORIES
            ]
            action = ["create", "delete", "list", "update", "monitor"][i % NUM_CATEGORIES]

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
        for tool_dict in tools:
            tool_data = ToolData(
                name=tool_dict["name"],
                description=tool_dict["description"],
                server_name=tool_dict["server_name"],
                params=tool_dict["params"],
                tags=tool_dict["tags"],
            )
            await temp_indexer_facade.index_tool(tool_data)

        # Verify all tools are indexed
        TOTAL_TOOLS_COUNT = 50
        all_tools = await temp_indexer_facade.persistence.get_all_tools()
        assert len(all_tools) == TOTAL_TOOLS_COUNT

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

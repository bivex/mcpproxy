"""Integration tests for end-to-end indexing and search flow."""

import pytest

from tests.fixtures.data import get_sample_tools_data
from mcpproxy.models.schemas import ToolData


class TestEndToEndFlow:
    """Test complete indexing and search flow."""

    @pytest.mark.asyncio
    async def test_e2e_indexing_and_search(self, temp_indexer_facade):
        """Test complete flow from indexing to search."""
        sample_data = get_sample_tools_data()

        # Step 1: Index all sample tools
        indexed_tools = []
        for tool_data_dict in sample_data:
            tool_data = ToolData(
                name=tool_data_dict["name"],
                description=tool_data_dict["description"],
                server_name=tool_data_dict["server_name"],
                params=tool_data_dict["params"],
                tags=tool_data_dict.get("tags", []),
                annotations=tool_data_dict.get("annotations", {}),
            )
            await temp_indexer_facade.index_tool(tool_data)
            indexed_tools.append(tool_data_dict["name"])

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

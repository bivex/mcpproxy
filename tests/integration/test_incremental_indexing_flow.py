"""Integration tests for incremental indexing flow."""

import pytest


class TestIncrementalIndexingFlow:
    """Test incremental indexing flow."""

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

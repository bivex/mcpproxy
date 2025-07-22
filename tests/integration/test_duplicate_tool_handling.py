"""Integration tests for duplicate tool handling."""

import pytest


class TestDuplicateToolHandling:
    """Test that duplicate tools (same hash) are handled correctly."""

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

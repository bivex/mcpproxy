"""Integration tests for tool update flow."""

import pytest


class TestToolUpdateFlow:
    """Test updating tool with different hash."""

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

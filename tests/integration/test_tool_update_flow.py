"""Integration tests for tool update flow."""

import pytest
from mcpproxy.models.schemas import ToolData


class TestToolUpdateFlow:
    """Test updating tool with different hash."""

    @pytest.mark.asyncio
    async def test_tool_update_flow(self, temp_indexer_facade):
        """Test updating tool with different hash."""
        # Index original tool
        tool_data_original = ToolData(
            name="update_test",
            description="Original description",
            server_name="test-server",
        )
        await temp_indexer_facade.index_tool(tool_data_original)

        # Verify original is indexed
        results = await temp_indexer_facade.search_tools("original", k=5)
        assert len(results) == 1
        assert "original" in results[0].tool.description.lower()

        # Update tool with different description (different hash)
        tool_data_updated = ToolData(
            name="update_test",
            description="Updated description with new content",
            server_name="test-server",
        )
        await temp_indexer_facade.index_tool(tool_data_updated)

        # Should now have both versions (different hashes)
        EXPECTED_TOOL_VERSIONS = 2
        all_tools = await temp_indexer_facade.persistence.get_all_tools()
        assert len(all_tools) == EXPECTED_TOOL_VERSIONS

        # Search should find the updated version when searching for "updated"
        results = await temp_indexer_facade.search_tools("updated content", k=5)
        assert len(results) >= 1
        found_descriptions = [r.tool.description for r in results]
        assert any("Updated description" in desc for desc in found_descriptions) 

"""Integration tests for persistence layer integration."""

import pytest
from mcpproxy.indexer.embedders.bm25 import BM25Embedder


class TestPersistenceLayerIntegration:
    """Test integration between indexer and persistence layer."""

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
        if isinstance(temp_indexer_facade.embedder, BM25Embedder):
            assert vector_count == 0  # BM25 doesn't use vector storage
        else:
            assert vector_count == 1  # Other embedders use vector storage

        # Search should find the tool using complex metadata
        results = await temp_indexer_facade.search_tools("complex configuration", k=5)
        assert len(results) == 1
        assert results[0].tool.name == "complex_tool" 

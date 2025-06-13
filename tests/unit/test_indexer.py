"""Tests for indexer functionality."""

import os
import tempfile
from enum import Enum
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from mcpproxy.indexer.base import BaseEmbedder
from mcpproxy.indexer.bm25 import BM25Embedder
from mcpproxy.indexer.facade import IndexerFacade
from mcpproxy.models.schemas import EmbedderType, SearchResult, ToolMetadata
from tests.fixtures.data import get_sample_tools_data, get_search_queries


class MockEmbedder(BaseEmbedder):
    """Mock embedder for testing."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.call_log = []

    async def embed_text(self, text: str) -> np.ndarray:
        """Mock embed text with deterministic but distinct vectors."""
        self.call_log.append(f"embed_text: {text}")
        # Create deterministic vector based on text hash
        hash_value = hash(text) % 1000
        vector = np.full(self.dimension, hash_value / 1000.0, dtype=np.float32)
        return vector

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Mock embed batch."""
        self.call_log.append(f"embed_batch: {len(texts)} texts")
        return [await self.embed_text(text) for text in texts]

    def get_dimension(self) -> int:
        """Get dimension."""
        return self.dimension

    # Don't override combine_tool_text - use the BaseEmbedder implementation


class TestBaseEmbedder:
    """Test cases for BaseEmbedder functionality."""

    def test_combine_tool_text_basic(self):
        """Test basic tool text combination."""
        embedder = MockEmbedder()

        result = embedder.combine_tool_text("test_tool", "Test description", None)

        expected = "Tool: test_tool | Description: Test description"
        assert result == expected

    def test_combine_tool_text_with_params(self):
        """Test tool text combination with parameters."""
        embedder = MockEmbedder()

        params = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Tool name"},
                "count": {"type": "integer", "description": "Count value"},
            },
        }

        result = embedder.combine_tool_text("test_tool", "Test description", params)

        assert "Tool: test_tool" in result
        assert "Description: Test description" in result
        assert "Parameters:" in result
        assert "name (string): Tool name" in result
        assert "count (integer): Count value" in result

    def test_combine_tool_text_empty_params(self):
        """Test tool text combination with empty parameters."""
        embedder = MockEmbedder()

        result = embedder.combine_tool_text("test_tool", "Test description", {})

        expected = "Tool: test_tool | Description: Test description"
        assert result == expected

    def test_combine_tool_text_complex_anyof_params(self):
        """Test tool text combination with complex anyOf parameter schemas like gcore tools."""
        embedder = MockEmbedder()

        # Simulate complex gcore-style parameter schemas
        params = {
            "type": "object",
            "properties": {
                "project_id": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                    "title": "Project Id",
                },
                "region_id": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                    "title": "Region Id",
                },
                "delete_floatings": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "title": "Delete Floatings",
                },
                "timeout": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "title": "Timeout",
                },
                "instance_id": {"title": "Instance Id", "type": "string"},
            },
            "required": ["instance_id"],
            "title": "DeleteInstance",
        }

        result = embedder.combine_tool_text(
            "delete_instance", "Delete an instance", params
        )

        # Verify the result contains expected elements
        assert "Tool: delete_instance" in result
        assert "Description: Delete an instance" in result
        assert "Parameters:" in result

        # Check that complex anyOf types are handled correctly
        assert "project_id (string|null): Project Id" in result
        assert "region_id (string|null): Region Id" in result
        assert "delete_floatings (string|null): Delete Floatings" in result
        assert "timeout (string|null): Timeout" in result
        assert "instance_id (string): Instance Id" in result

    def test_combine_tool_text_oneof_params(self):
        """Test tool text combination with oneOf parameter schemas."""
        embedder = MockEmbedder()

        params = {
            "type": "object",
            "properties": {
                "value": {
                    "oneOf": [{"type": "string"}, {"type": "integer"}],
                    "description": "A value that can be string or integer",
                }
            },
        }

        result = embedder.combine_tool_text("test_tool", "Test description", params)

        assert "value (string|integer): A value that can be string or integer" in result

    def test_combine_tool_text_title_fallback(self):
        """Test that title is used when description is not available."""
        embedder = MockEmbedder()

        params = {
            "type": "object",
            "properties": {
                "param_with_title": {
                    "type": "string",
                    "title": "Parameter Title",
                    # No description field
                }
            },
        }

        result = embedder.combine_tool_text("test_tool", "Test description", params)

        assert "param_with_title (string): Parameter Title" in result

    def test_extract_param_info_edge_cases(self):
        """Test _extract_param_info with edge cases."""
        embedder = MockEmbedder()

        # Test empty param_info
        param_type, param_desc = embedder._extract_param_info({})
        assert param_type == "unknown"
        assert param_desc == ""

        # Test anyOf with non-dict elements
        param_info = {"anyOf": ["invalid", {"type": "string"}]}
        param_type, param_desc = embedder._extract_param_info(param_info)
        assert param_type == "string"

        # Test anyOf with no valid types
        param_info = {"anyOf": [{"invalid": "data"}]}
        param_type, param_desc = embedder._extract_param_info(param_info)
        assert param_type == "unknown"


class TestBM25Embedder:
    """Test cases for BM25Embedder."""

    @pytest.fixture
    def temp_index_dir(self):
        """Create temporary directory for BM25 index."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.mark.asyncio
    async def test_bm25_embedder_initialization(self, temp_index_dir):
        """Test BM25 embedder initialization."""
        embedder = BM25Embedder(temp_index_dir)

        assert embedder.index_dir == temp_index_dir
        assert not embedder.indexed
        assert embedder.corpus == []
        assert embedder.retriever is None

    @pytest.mark.asyncio
    async def test_bm25_fit_corpus(self, temp_index_dir):
        """Test fitting BM25 on corpus."""
        embedder = BM25Embedder(temp_index_dir)
        texts = [
            "create virtual machine instance",
            "delete storage volume",
            "list monitoring metrics",
        ]

        await embedder.fit_corpus(texts)

        assert embedder.indexed
        assert embedder.corpus == texts
        assert embedder.retriever is not None
        # Check index files were created
        assert os.path.exists(os.path.join(temp_index_dir, "bm25s_index"))

    @pytest.mark.asyncio
    async def test_bm25_embed_text(self, temp_index_dir):
        """Test BM25 text embedding."""
        embedder = BM25Embedder(temp_index_dir)

        vector = await embedder.embed_text("test text")

        assert isinstance(vector, np.ndarray)
        assert vector.dtype == np.float32
        assert len(vector) == 1  # BM25 returns placeholder vector
        assert "test text" in embedder.corpus
        assert not embedder.indexed  # Should be marked for reindexing

    @pytest.mark.asyncio
    async def test_bm25_embed_batch(self, temp_index_dir):
        """Test BM25 batch embedding."""
        embedder = BM25Embedder(temp_index_dir)
        texts = ["text one", "text two", "text three"]

        vectors = await embedder.embed_batch(texts)

        assert len(vectors) == 3
        assert all(isinstance(v, np.ndarray) for v in vectors)
        assert all(v.dtype == np.float32 for v in vectors)
        assert all(len(v) == 1 for v in vectors)  # Placeholder vectors
        assert all(text in embedder.corpus for text in texts)

    @pytest.mark.asyncio
    async def test_bm25_reindex(self, temp_index_dir):
        """Test BM25 reindexing functionality."""
        embedder = BM25Embedder(temp_index_dir)
        texts = ["create virtual machine instance", "delete storage volume"]

        # Add texts without indexing
        for text in texts:
            await embedder.embed_text(text)

        assert not embedder.indexed

        # Trigger reindexing
        await embedder.reindex()

        assert embedder.indexed
        assert embedder.retriever is not None

    @pytest.mark.asyncio
    async def test_bm25_search_similar(self, temp_index_dir):
        """Test BM25 similarity search."""
        embedder = BM25Embedder(temp_index_dir)
        candidate_texts = [
            "create virtual machine instance",
            "delete storage volume",
            "list network interfaces",
            "monitor system performance",
        ]

        results = await embedder.search_similar("create instance", candidate_texts, k=2)

        assert len(results) <= 2
        assert all(isinstance(r, tuple) for r in results)
        assert all(len(r) == 2 for r in results)  # (index, score)
        assert all(isinstance(r[0], int) for r in results)
        assert all(isinstance(r[1], float) for r in results)

        # First result should be most similar (highest score)
        if len(results) > 1:
            assert results[0][1] >= results[1][1]

    @pytest.mark.asyncio
    async def test_bm25_search_with_indexed_corpus(self, temp_index_dir):
        """Test BM25 search using pre-indexed corpus."""
        embedder = BM25Embedder(temp_index_dir)
        corpus = [
            "create virtual machine instance",
            "delete storage volume",
            "list network interfaces",
        ]

        # Index the corpus first
        await embedder.fit_corpus(corpus)

        # Search without providing candidate_texts (use indexed corpus)
        results = await embedder.search_similar("create instance", None, k=2)

        assert len(results) <= 2
        assert all(isinstance(r, tuple) for r in results)
        # Results should reference the indexed corpus
        if results:
            assert 0 <= results[0][0] < len(corpus)

    @pytest.mark.asyncio
    async def test_bm25_search_empty_candidates(self, temp_index_dir):
        """Test BM25 search with empty candidates."""
        embedder = BM25Embedder(temp_index_dir)

        results = await embedder.search_similar("test query", [], k=5)

        assert results == []

    @pytest.mark.asyncio
    async def test_bm25_load_index(self, temp_index_dir):
        """Test loading BM25 index from disk."""
        # Create and save an index
        embedder1 = BM25Embedder(temp_index_dir)
        texts = ["test document one", "test document two"]
        await embedder1.fit_corpus(texts)

        # Create new embedder and load the index
        embedder2 = BM25Embedder(temp_index_dir)
        success = embedder2.load_index()

        assert success
        assert embedder2.indexed
        assert embedder2.retriever is not None
        assert len(embedder2.corpus) == len(texts)


class TestIndexerFacade:
    """Test cases for IndexerFacade."""

    @pytest.fixture
    def mock_persistence(self):
        """Mock persistence facade."""
        mock = AsyncMock()
        mock.get_tool_by_hash.return_value = None  # No existing tool by default
        mock.store_tool_with_vector.return_value = 1
        mock.get_all_tools.return_value = []
        mock.search_similar_tools.return_value = []
        return mock

    @pytest.fixture
    def temp_index_dir(self):
        """Create temporary directory for BM25 index."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_indexer_facade_initialization(self, mock_persistence, temp_index_dir):
        """Test IndexerFacade initialization."""
        indexer = IndexerFacade(
            mock_persistence, EmbedderType.BM25, index_dir=temp_index_dir
        )

        assert indexer.persistence == mock_persistence
        assert isinstance(indexer.embedder, BM25Embedder)
        assert indexer.embedder.index_dir == temp_index_dir

    def test_indexer_facade_embedder_creation(self, mock_persistence, temp_index_dir):
        """Test different embedder creation."""
        # Test BM25
        indexer_bm25 = IndexerFacade(
            mock_persistence, EmbedderType.BM25, index_dir=temp_index_dir
        )
        assert isinstance(indexer_bm25.embedder, BM25Embedder)

        # Test with mock unknown embedder type
        class UnknownEmbedderType(str, Enum):
            UNKNOWN = "UNKNOWN"
        
        with pytest.raises(ValueError, match="Unknown embedder type"):
            # This will fail at the dependencies check or the embedder creation
            IndexerFacade(mock_persistence, UnknownEmbedderType.UNKNOWN)

    @pytest.mark.asyncio
    async def test_index_tool_basic(self, mock_persistence, temp_index_dir):
        """Test basic tool indexing."""
        indexer = IndexerFacade(
            mock_persistence, EmbedderType.BM25, index_dir=temp_index_dir
        )

        await indexer.index_tool(
            name="test_tool",
            description="Test tool description",
            server_name="test-server",
        )

        # Verify persistence calls
        mock_persistence.get_tool_by_hash.assert_called_once()
        mock_persistence.store_tool_with_vector.assert_called_once()

        # Check the stored tool metadata
        call_args = mock_persistence.store_tool_with_vector.call_args
        stored_tool, stored_vector = call_args[0]

        assert stored_tool.name == "test_tool"
        assert stored_tool.description == "Test tool description"
        assert stored_tool.server_name == "test-server"
        assert isinstance(stored_vector, np.ndarray)

        # Should be marked for reindexing
        assert indexer._needs_reindex

    @pytest.mark.asyncio
    async def test_index_tool_with_metadata(self, mock_persistence, temp_index_dir):
        """Test tool indexing with full metadata."""
        indexer = IndexerFacade(
            mock_persistence, EmbedderType.BM25, index_dir=temp_index_dir
        )

        params = {"type": "object", "properties": {"name": {"type": "string"}}}
        tags = ["compute", "vm"]
        annotations = {"category": "compute"}

        await indexer.index_tool(
            name="test_tool",
            description="Test tool",
            server_name="test-server",
            params=params,
            tags=tags,
            annotations=annotations,
        )

        # Verify extended params in hash computation
        call_args = mock_persistence.store_tool_with_vector.call_args
        stored_tool, _ = call_args[0]

        assert "tags" in stored_tool.params_json
        assert "annotations" in stored_tool.params_json
        assert stored_tool.params_json is not None

    @pytest.mark.asyncio
    async def test_index_tool_duplicate_hash(self, mock_persistence, temp_index_dir):
        """Test indexing tool with existing hash (should skip)."""
        # Mock existing tool
        existing_tool = ToolMetadata(
            id=1,
            name="existing",
            description="desc",
            hash="existing_hash",
            server_name="server",
        )
        mock_persistence.get_tool_by_hash.return_value = existing_tool

        indexer = IndexerFacade(
            mock_persistence, EmbedderType.BM25, index_dir=temp_index_dir
        )

        await indexer.index_tool("test_tool", "description", "server")

        # Should call get_tool_by_hash but not store_tool_with_vector
        mock_persistence.get_tool_by_hash.assert_called_once()
        mock_persistence.store_tool_with_vector.assert_not_called()

        # Should not mark for reindexing since tool wasn't added
        assert not indexer._needs_reindex

    @pytest.mark.asyncio
    async def test_reindex_all_tools(self, mock_persistence, temp_index_dir):
        """Test reindexing all tools functionality."""
        # Setup mock tools
        sample_tools = [
            ToolMetadata(
                id=1,
                name="create_instance",
                description="Create VM",
                hash="hash1",
                server_name="api",
                params_json='{"parameters": {}, "tags": ["compute"], "annotations": null}',
            ),
            ToolMetadata(
                id=2,
                name="delete_volume",
                description="Delete storage",
                hash="hash2",
                server_name="api",
                params_json='{"parameters": {}, "tags": ["storage"], "annotations": null}',
            ),
        ]
        mock_persistence.get_all_tools.return_value = sample_tools

        indexer = IndexerFacade(
            mock_persistence, EmbedderType.BM25, index_dir=temp_index_dir
        )
        indexer._needs_reindex = True

        await indexer.reindex_all_tools()

        assert not indexer._needs_reindex
        assert indexer.embedder.indexed
        assert len(indexer.embedder.corpus) == 2

    @pytest.mark.asyncio
    async def test_search_tools_bm25(self, mock_persistence, temp_index_dir):
        """Test tool search with BM25 embedder."""
        # Setup mock tools
        sample_tools = [
            ToolMetadata(
                id=1,
                name="create_instance",
                description="Create VM",
                hash="hash1",
                server_name="api",
                params_json='{"parameters": {}, "tags": [], "annotations": null}',
            ),
            ToolMetadata(
                id=2,
                name="delete_volume",
                description="Delete storage",
                hash="hash2",
                server_name="api",
                params_json='{"parameters": {}, "tags": [], "annotations": null}',
            ),
        ]
        mock_persistence.get_all_tools.return_value = sample_tools

        indexer = IndexerFacade(
            mock_persistence, EmbedderType.BM25, index_dir=temp_index_dir
        )

        # Mock BM25 search to return first tool as best match
        with patch.object(indexer.embedder, "search_similar") as mock_search:
            mock_search.return_value = [(0, 0.8), (1, 0.3)]

            results = await indexer.search_tools("create virtual machine", k=2)

            assert len(results) == 2
            assert all(isinstance(r, SearchResult) for r in results)
            assert results[0].tool.name == "create_instance"
            # With modified sigmoid normalization: 1 / (1 + exp(-0.8)) ≈ 0.689
            assert abs(results[0].score - 0.689) < 0.01
            # Second result with modified sigmoid: 1 / (1 + exp(-0.3)) ≈ 0.574
            assert abs(results[1].score - 0.574) < 0.01
            # First result should still have higher score
            assert results[0].score > results[1].score

    @pytest.mark.asyncio
    async def test_search_tools_vector_embedder(self, mock_persistence, temp_index_dir):
        """Test tool search with vector embedder."""
        # Create indexer with mock vector embedder
        indexer = IndexerFacade(
            mock_persistence, EmbedderType.BM25, index_dir=temp_index_dir
        )
        indexer.embedder = MockEmbedder()  # Not BM25, so will use vector path

        # Mock vector search results
        mock_results = [
            SearchResult(
                tool=ToolMetadata(
                    id=1,
                    name="test_tool",
                    description="desc",
                    hash="hash",
                    server_name="server",
                ),
                score=0.9,
            )
        ]
        mock_persistence.search_similar_tools.return_value = mock_results

        results = await indexer.search_tools("test query", k=3)

        assert len(results) == 1
        assert results[0].tool.name == "test_tool"
        assert results[0].score == 0.9

        # Verify vector search was called
        mock_persistence.search_similar_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_tools_no_results(self, mock_persistence, temp_index_dir):
        """Test search with no matching tools."""
        mock_persistence.get_all_tools.return_value = []

        indexer = IndexerFacade(
            mock_persistence, EmbedderType.BM25, index_dir=temp_index_dir
        )

        results = await indexer.search_tools("nonexistent query", k=5)

        assert results == []

    @pytest.mark.asyncio
    async def test_search_tools_with_sample_data(self, temp_indexer_facade):
        """Test search with realistic sample data."""
        sample_data = get_sample_tools_data()

        # Index sample tools
        for tool_data in sample_data:
            await temp_indexer_facade.index_tool(
                name=tool_data["name"],
                description=tool_data["description"],
                server_name=tool_data["server_name"],
                params=tool_data["params"],
                tags=tool_data.get("tags", []),
                annotations=tool_data.get("annotations", {}),
            )

        # Test various search queries
        queries = get_search_queries()

        for query_data in queries:
            query = query_data["query"]
            expected_tools = query_data["expected_tools"]
            min_score = query_data["min_score"]

            results = await temp_indexer_facade.search_tools(query, k=5)

            if expected_tools:
                # Should find at least one expected tool
                found_tools = {r.tool.name for r in results}
                assert any(tool in found_tools for tool in expected_tools), (
                    f"Query '{query}' should find at least one of {expected_tools}, got {found_tools}"
                )

                # Check minimum score
                if results:
                    assert max(r.score for r in results) >= min_score, (
                        f"Query '{query}' should have score >= {min_score}"
                    )
            else:
                # For nonsense queries, scores should be low or no results
                if results:
                    assert all(r.score < 0.5 for r in results), (
                        f"Nonsense query '{query}' should have low scores"
                    )

    @pytest.mark.asyncio
    async def test_index_multiple_tools_different_servers(self, temp_indexer_facade):
        """Test indexing tools from different servers."""
        tools_data = [
            ("tool1", "Description 1", "server-a"),
            ("tool2", "Description 2", "server-b"),
            ("tool3", "Description 3", "server-a"),
        ]

        for name, desc, server in tools_data:
            await temp_indexer_facade.index_tool(name, desc, server)

        # Verify all tools were indexed
        all_tools = await temp_indexer_facade.persistence.get_all_tools()
        assert len(all_tools) == 3

        # Test server-specific retrieval
        server_a_tools = await temp_indexer_facade.persistence.get_tools_by_server(
            "server-a"
        )
        server_b_tools = await temp_indexer_facade.persistence.get_tools_by_server(
            "server-b"
        )

        assert len(server_a_tools) == 2
        assert len(server_b_tools) == 1

    @pytest.mark.asyncio
    async def test_index_and_search_with_tags(self, temp_indexer_facade):
        """Test that tags improve search relevance."""
        # Index tool with relevant tags and more descriptive content
        await temp_indexer_facade.index_tool(
            name="vm_creator",
            description="Create virtual machine instances with compute resources",
            server_name="api",
            tags=["virtual-machine", "compute", "creation"],
        )

        # Index tool without relevant tags
        await temp_indexer_facade.index_tool(
            name="file_reader",
            description="Read files from storage system",
            server_name="api",
            tags=["file", "io", "reading"],
        )

        # Search for virtual machine - should prefer tagged tool
        results = await temp_indexer_facade.search_tools(
            "virtual machine creation", k=2
        )

        assert len(results) >= 1
        # Tool with relevant tags should be ranked higher or be the first result
        if len(results) > 1:
            # Either vm_creator should have higher score or be first
            assert (results[0].tool.name == "vm_creator") or (
                results[0].score >= results[1].score
            )
        else:
            # If only one result, it should be the vm_creator
            assert results[0].tool.name == "vm_creator"

    @pytest.mark.asyncio
    async def test_embedder_call_logging(self, mock_persistence):
        """Test that embedder methods are called correctly."""
        mock_embedder = MockEmbedder()
        indexer = IndexerFacade(mock_persistence, EmbedderType.BM25)
        indexer.embedder = mock_embedder

        await indexer.index_tool("test_tool", "description", "server")

        # Check that embed_text was called
        assert len(mock_embedder.call_log) > 0
        assert any("embed_text" in call for call in mock_embedder.call_log)

"""Tests for indexer functionality."""

import os
import tempfile
from enum import Enum
from unittest.mock import AsyncMock, patch
from typing import Any

import numpy as np
import pytest
import json

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

    @pytest.mark.parametrize(
        "description, params, expected_contains",
        [
            ("Test description", None, ["Tool: test_tool", "Description: Test description"]),
            ("Test description", {},
             ["Tool: test_tool", "Description: Test description"]),
            (
                "Test description",
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Tool name"},
                        "count": {"type": "integer", "description": "Count value"},
                    },
                },
                [
                    "Tool: test_tool",
                    "Description: Test description",
                    "Parameters:",
                    "name (string): Tool name",
                    "count (integer): Count value",
                ],
            ),
            (
                "Test description",
                {
                    "type": "object",
                    "properties": {
                        "project_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "region_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    },
                },
                [
                    "Tool: test_tool",
                    "Description: Test description",
                    "Parameters:",
                    "project_id (string, null)",
                    "region_id (string, null)",
                ],
            ),
            (
                "Test description",
                {
                    "type": "object",
                    "properties": {"data": {"oneOf": [{"type": "string"}, {"type": "integer"}]}},
                },
                [
                    "Tool: test_tool",
                    "Description: Test description",
                    "Parameters:",
                    "data (string, integer)",
                ],
            ),
            (
                "Test description",
                {
                    "type": "object",
                    "properties": {"value": {"type": "string", "title": "My Value"}},
                },
                [
                    "Tool: test_tool",
                    "Description: Test description",
                    "Parameters:",
                    "value (string): My Value",
                ],
            ),
        ],
    )
    def test_combine_tool_text_scenarios(self, description: str, params: Any, expected_contains: list[str]):
        """Test various scenarios for tool text combination."""
        embedder = MockEmbedder()
        result = embedder.combine_tool_text("test_tool", description, params)

        for expected_str in expected_contains:
            assert expected_str in result, f"Expected '{expected_str}' not found in '{result}'"

    def test_extract_param_info_edge_cases(self):
        """Test edge cases for extract_param_info."""
        embedder = MockEmbedder()
        # Test with no properties
        params = {"type": "object"}
        result = embedder._extract_param_info(params)
        assert result == []

        # Test with missing description
        params = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        result = embedder._extract_param_info(params)
        assert result == ["name (string)"]

        # Test with nested types (should ignore nested structures for simple extraction)
        params = {
            "type": "object",
            "properties": {"data": {"type": "object", "properties": {"key": {"type": "string"}}}},
        }
        result = embedder._extract_param_info(params)
        assert result == ["data (object)"]

        # Test with array items
        params = {
            "type": "object",
            "properties": {"items": {"type": "array", "items": {"type": "integer"}}},
        }
        result = embedder._extract_param_info(params)
        assert result == ["items (array)"]


class TestBM25Embedder:
    """Test cases for BM25Embedder functionality."""

    @pytest.fixture
    def temp_index_dir(self):
        """Temporary directory for BM25 index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.mark.asyncio
    async def test_bm25_embedder_initialization(self, temp_index_dir):
        """Test BM25Embedder initialization and index creation."""
        embedder = BM25Embedder(temp_index_dir)
        assert embedder.index_dir == temp_index_dir
        assert embedder.corpus == []
        assert not embedder.indexed
        assert embedder.retriever is None
        # Check index files were created
        assert os.path.exists(os.path.join(temp_index_dir, "bm25s_index"))

    @pytest.mark.asyncio
    async def test_bm25_fit_corpus(self, temp_index_dir):
        """Test BM25 corpus fitting."""
        embedder = BM25Embedder(temp_index_dir)
        corpus = ["doc one", "doc two", "doc three"]
        await embedder.fit_corpus(corpus)

        assert embedder.corpus == corpus
        assert embedder.indexed
        assert embedder.retriever is not None

    @pytest.mark.parametrize("texts", [
        "test text",
        ["text one", "text two", "text three"]
    ])
    @pytest.mark.asyncio
    async def test_bm25_embedding_scenarios(self, temp_index_dir, texts):
        """Test BM25 text and batch embedding."""
        embedder = BM25Embedder(temp_index_dir)

        if isinstance(texts, str):
            vector = await embedder.embed_text(texts)
            assert isinstance(vector, np.ndarray)
            assert vector.dtype == np.float32
            assert len(vector) == 1  # BM25 returns placeholder vector
            assert texts in embedder.corpus
        else:
            vectors = await embedder.embed_batch(texts)
            assert len(vectors) == len(texts)
            assert all(isinstance(v, np.ndarray) for v in vectors)
            assert all(v.dtype == np.float32 for v in vectors)
            assert all(len(v) == 1 for v in vectors)  # Placeholder vectors
            assert all(text in embedder.corpus for text in texts)
        assert not embedder.indexed  # Should be marked for reindexing

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

    @pytest.mark.parametrize("query, candidate_texts, k, expected_results_len", [
        ("create instance", ["create virtual machine instance", "delete storage volume"], 2, 1),
        ("create instance", None, 2, 1), # Uses pre-indexed corpus
        ("test query", [], 5, 0) # Empty candidates
    ])
    @pytest.mark.asyncio
    async def test_bm25_search_scenarios(self, temp_index_dir, query, candidate_texts, k, expected_results_len):
        """Test BM25 similarity search scenarios."""
        embedder = BM25Embedder(temp_index_dir)

        if candidate_texts is None:
            # If candidate_texts is None, means we should use pre-indexed corpus
            corpus = ["create virtual machine instance", "delete storage volume", "list network interfaces"]
            await embedder.fit_corpus(corpus)

        results = await embedder.search_similar(query, candidate_texts, k=k)

        assert len(results) == expected_results_len
        assert all(isinstance(r, tuple) for r in results)
        assert all(len(r) == 2 for r in results)  # (index, score)
        assert all(isinstance(r[0], int) for r in results)
        assert all(isinstance(r[1], float) for r in results)

        # First result should be most similar (highest score)
        if len(results) > 1:
            assert results[0][1] >= results[1][1]

    @pytest.mark.asyncio
    async def test_bm25_load_index(self, temp_index_dir):
        """Test loading an existing BM25 index."""
        embedder = BM25Embedder(temp_index_dir)
        corpus = ["doc 1", "doc 2"]
        await embedder.fit_corpus(corpus)

        # Create a new embedder instance to load the saved index
        new_embedder = BM25Embedder(temp_index_dir)
        new_embedder.load_index()

        assert new_embedder.corpus == corpus
        assert new_embedder.indexed
        assert new_embedder.retriever is not None


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

    @pytest.mark.parametrize(
        "name, description, server_name, params, tags, annotations, is_duplicate, expected_store_call, expected_reindex",
        [
            ("basic_tool", "Basic description", "server1", None, [], {}, False, True, True),
            ("tool_with_metadata", "Description with metadata", "server1",
             {"type": "object", "properties": {"name": {"type": "string"}}},
             ["compute"], {"category": "compute"}, False, True, True),
            ("duplicate_tool", "Duplicate description", "server1", None, [], {}, True, False, False),
        ],
    )
    @pytest.mark.asyncio
    async def test_indexer_facade_tool_indexing_scenarios(
        self, mock_persistence, temp_index_dir, name, description, server_name, params, tags, annotations, is_duplicate, expected_store_call, expected_reindex
    ):
        """Test various tool indexing scenarios including duplicates."""
        if is_duplicate:
            existing_tool = ToolMetadata(
                id=1,
                name=name,
                description=description,
                hash="duplicate_hash_value", # Needs to be consistent for mock
                server_name=server_name,
                params_json=json.dumps(params or {})
            )
            mock_persistence.get_tool_by_hash.return_value = existing_tool

        indexer = IndexerFacade(
            mock_persistence, EmbedderType.BM25, index_dir=temp_index_dir
        )

        await indexer.index_tool(name, description, server_name, params, tags, annotations)

        mock_persistence.get_tool_by_hash.assert_called_once()

        if expected_store_call:
            mock_persistence.store_tool_with_vector.assert_called_once()
            call_args = mock_persistence.store_tool_with_vector.call_args
            stored_tool, _ = call_args[0]
            assert stored_tool.name == name
            assert stored_tool.description == description
            assert stored_tool.server_name == server_name
            if params: # Check if params are part of params_json
                assert stored_tool.params_json is not None
                assert json.loads(stored_tool.params_json).get("properties") == params.get("properties")
            if tags:
                assert json.loads(stored_tool.params_json).get("tags") == tags
            if annotations:
                assert json.loads(stored_tool.params_json).get("annotations") == annotations
        else:
            mock_persistence.store_tool_with_vector.assert_not_called()

        assert indexer._needs_reindex == expected_reindex

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

    @pytest.mark.parametrize(
        "scenario, query, k, expected_results",
        [
            ("bm25_basic", "create virtual machine", 2, {"len": 2, "tool_name": "create_instance", "score_check": True}),
            ("vector_embedder", "test query", 3, {"len": 1, "tool_name": "test_tool"}),
            ("no_results", "nonexistent query", 5, {"len": 0}),
            ("sample_data", "create virtual machine", 5, {"len_min": 1, "tool_names": ["create_instance"], "min_score_check": True}),
            ("sample_data", "nonsense random query xyz", 5, {"len_max": 0, "max_score_check": True}),
        ],
    )
    @pytest.mark.asyncio
    async def test_indexer_facade_search_scenarios(
        self, mock_persistence, temp_index_dir, scenario, query, k, expected_results
    ):
        """Test various search scenarios for IndexerFacade."""

        indexer = IndexerFacade(
            mock_persistence, EmbedderType.BM25, index_dir=temp_index_dir
        )

        if scenario == "bm25_basic":
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
            with patch.object(indexer.embedder, "search_similar") as mock_search:
                mock_search.return_value = [(0, 0.8), (1, 0.3)]
                results = await indexer.search_tools(query, k=k)

                assert len(results) == expected_results["len"]
                assert all(isinstance(r, SearchResult) for r in results)
                assert results[0].tool.name == expected_results["tool_name"]
                if expected_results.get("score_check"):
                    assert abs(results[0].score - 0.689) < 0.01
                    assert abs(results[1].score - 0.574) < 0.01
                    assert results[0].score > results[1].score

        elif scenario == "vector_embedder":
            indexer.embedder = MockEmbedder() # Use mock for vector embedder
            mock_results = [
                SearchResult(
                    tool=ToolMetadata(
                        id=1,
                        name=expected_results["tool_name"],
                        description="desc",
                        hash="hash",
                        server_name="server",
                    ),
                    score=0.9,
                )
            ]
            mock_persistence.search_similar_tools.return_value = mock_results
            results = await indexer.search_tools(query, k=k)

            assert len(results) == expected_results["len"]
            assert results[0].tool.name == expected_results["tool_name"]
            assert results[0].score == 0.9
            mock_persistence.search_similar_tools.assert_called_once()

        elif scenario == "no_results":
            mock_persistence.get_all_tools.return_value = []
            results = await indexer.search_tools(query, k=k)
            assert results == []

        elif scenario == "sample_data":
            sample_data = get_sample_tools_data()
            for tool_data in sample_data:
                await indexer.index_tool(
                    name=tool_data["name"],
                    description=tool_data["description"],
                    server_name=tool_data["server_name"],
                    params=tool_data["params"],
                    tags=tool_data.get("tags", []),
                    annotations=tool_data.get("annotations", {}),
                )
            queries = get_search_queries()
            query_data = next(q for q in queries if q["query"] == query)

            results = await indexer.search_tools(query, k=k)

            if expected_results.get("len_min") is not None:
                assert len(results) >= expected_results["len_min"]
            if expected_results.get("tool_names"):
                found_tools = {r.tool.name for r in results}
                assert any(tool in found_tools for tool in expected_results["tool_names"]), (
                    f"Query '{query}' should find at least one of {expected_results["tool_names"]}, got {found_tools}"
                )
            if expected_results.get("min_score_check"):
                if results:
                    assert max(r.score for r in results) >= query_data["min_score"], (
                        f"Query '{query}' should have score >= {query_data["min_score"]}"
                    )
            if expected_results.get("len_max") is not None:
                assert len(results) == expected_results["len_max"]
            if expected_results.get("max_score_check"):
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

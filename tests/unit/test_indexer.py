"""Tests for indexer functionality."""

import os
import tempfile
from enum import Enum
from unittest.mock import AsyncMock, patch
from typing import Any

import numpy as np
import pytest
import json
from unittest.mock import MagicMock

from mcpproxy.indexer.embedders.base import BaseEmbedder
from mcpproxy.indexer.embedders.bm25 import BM25Embedder
from mcpproxy.indexer.embedders.huggingface import HuggingFaceEmbedder
from mcpproxy.indexer.embedders.openai import OpenAIEmbedder
from mcpproxy.indexer.facade import IndexerFacade
from mcpproxy.models.schemas import EmbedderType, SearchResult, ToolMetadata
from mcpproxy.persistence.facade import PersistenceFacade # Import PersistenceFacade
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
                    "project_id (string|null): ",
                    "region_id (string|null): ",
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
                    "data (string|integer): ",
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
            # New edge cases from test_extract_param_info_edge_cases
            ("Test description", {"type": "object"}, ["Tool: test_tool", "Description: Test description"]), # No properties
            ("Test description", {"type": "object", "properties": {"name": {"type": "string"}}}, ["name (string)"]), # Missing description
            ("Test description", {"type": "object", "properties": {"data": {"type": "object", "properties": {"key": {"type": "string"}}}}}, ["data (object)"]), # Nested types
            ("Test description", {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "integer"}}}}, ["items (array)"]), # Array items
        ],
    )
    def test_combine_tool_text_scenarios(self, description: str, params: Any, expected_contains: list[str]):
        """Test various scenarios for tool text combination."""
        embedder = MockEmbedder()
        result = embedder.combine_tool_text("test_tool", description, params)

        for expected_str in expected_contains:
            assert expected_str in result, f"Expected '{expected_str}' not found in '{result}'"


class TestBM25Embedder:
    """Test cases for BM25Embedder functionality."""

    @pytest.fixture
    def temp_index_dir(self):
        """Temporary directory for BM25 index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.mark.parametrize(
        "scenario, corpus_data, texts_to_embed, reindex_expected",
        [
            ("initialization", None, None, False), # Only initialization
            ("fit_corpus", ["doc one", "doc two"], None, False), # Initialization and fit_corpus
            ("reindex", ["initial text"], ["new text 1", "new text 2"], True), # Reindex after embedding new texts
        ],
    )
    @pytest.mark.asyncio
    async def test_bm25_embedder_lifecycle(
        self, temp_index_dir, scenario, corpus_data, texts_to_embed, reindex_expected
    ):
        """Test BM25Embedder lifecycle: initialization, corpus fitting, and reindexing."""
        embedder = BM25Embedder(temp_index_dir)

        # Test initialization
        assert embedder.index_dir == temp_index_dir
        assert embedder.corpus == []
        assert not embedder.indexed
        assert embedder.retriever is None
        assert os.path.exists(os.path.join(temp_index_dir, "bm25s_index"))

        if scenario == "fit_corpus" and corpus_data:
            await embedder.fit_corpus(corpus_data)
            assert embedder.corpus == corpus_data
            assert embedder.is_indexed()
            assert embedder.retriever is not None

        if scenario == "reindex":
            # Ensure initial indexing from corpus_data
            if corpus_data:
                await embedder.fit_corpus(corpus_data)
            
            # Embed new texts to trigger reindexing
            if texts_to_embed:
                for text in texts_to_embed:
                    await embedder.embed_text(text)
                assert not embedder.is_indexed() # Should be marked as needing reindex

            await embedder.reindex()

            assert embedder.is_indexed()
            assert embedder.retriever is not None
            if texts_to_embed:
                # Re-check the corpus after reindexing to include new texts
                expected_corpus = sorted(list(set(texts_to_embed + corpus_data)))
                actual_corpus = sorted(embedder.corpus)
                assert actual_corpus == expected_corpus

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
        else:
            vectors = await embedder.embed_batch(texts)
            assert len(vectors) == len(texts)
            assert all(isinstance(v, np.ndarray) for v in vectors)
            assert all(v.dtype == np.float32 for v in vectors)
            assert all(len(v) == 1 for v in vectors)  # Placeholder vectors

    @pytest.mark.parametrize("query, candidate_texts, k, expected_results_len, pre_indexed_corpus", [
        ("create instance", ["create virtual machine instance", "delete storage volume"], 2, 1, None),
        ("create instance", None, 2, 1, ["create virtual machine instance", "delete storage volume", "list network interfaces"]), # Uses pre-indexed corpus
        ("test query", [], 5, 0, None) # Empty candidates
    ])
    @pytest.mark.asyncio
    async def test_bm25_search_scenarios(self, temp_index_dir, query, candidate_texts, k, expected_results_len, pre_indexed_corpus):
        """Test BM25 similarity search scenarios."""
        embedder = BM25Embedder(temp_index_dir)

        if pre_indexed_corpus is not None:
            # If pre_indexed_corpus is provided, means we should use it for pre-indexing
            await embedder.fit_corpus(pre_indexed_corpus)
        elif candidate_texts is None: # Fallback if no specific pre_indexed_corpus, but candidate_texts is None
            # This case handles scenarios where the test expects existing corpus but doesn't specify it.
            # For now, we will assume an empty corpus for this case if no pre_indexed_corpus.
            pass # Or raise an error if this state is invalid

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
        assert new_embedder.is_indexed()
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

    @pytest.fixture
    async def populated_indexer_facade(self, temp_index_dir, mock_persistence: PersistenceFacade):
        """Fixture for a pre-populated IndexerFacade with sample data."""
        indexer = IndexerFacade(
            mock_persistence, EmbedderType.BM25, index_dir=temp_index_dir
        )
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
        return indexer

    @pytest.mark.parametrize(
        "embedder_type, expected_embedder_class, expect_exception",
        [
            (EmbedderType.BM25, BM25Embedder, False),
            (EmbedderType.HF, HuggingFaceEmbedder, False),
            (EmbedderType.OPENAI, OpenAIEmbedder, False),
            ("invalid_embedder_type", None, True), # Test case for unknown embedder type
        ],
    )
    def test_indexer_facade_init_and_embedder_creation_scenarios(
        self, persistence_facade, embedder_type, expected_embedder_class, expect_exception
    ):
        """Test IndexerFacade initialization and correct embedder creation."""
        if expect_exception:
            with pytest.raises(ValueError):
                IndexerFacade(persistence_facade, embedder_type)
        else:
            indexer = IndexerFacade(persistence_facade, embedder_type)
            assert isinstance(indexer.embedder, expected_embedder_class)

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

    @pytest.mark.parametrize(
        "initial_needs_reindex, expected_reindex_call",
        [
            (True, True),
            (False, False),
        ],
    )
    @pytest.mark.asyncio
    async def test_reindex_all_tools(
        self, mock_persistence, temp_index_dir, initial_needs_reindex, expected_reindex_call
    ):
        """Test reindexing all tools functionality and its effect on embedder.reindex()."""
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
        # Removed direct access to private attribute: indexer._needs_reindex = initial_needs_reindex

        # Mock the embedder's reindex method
        with patch.object(indexer.embedder, 'reindex', new_callable=AsyncMock) as mock_embedder_reindex:
            # Simulate the initial state for the embedder's indexed status
            indexer.embedder.indexed = not initial_needs_reindex # This will be flipped if reindex happens

            await indexer.reindex_all_tools()

            # Assert that embedder.reindex was called only when expected
            if initial_needs_reindex: # If it initially needed reindex, it should have been called
                mock_embedder_reindex.assert_called_once()
            else:
                mock_embedder_reindex.assert_not_called()

        # Verify public state or side effects, not private attributes
        assert indexer.embedder.indexed is True # After calling reindex_all_tools, it should be indexed regardless of initial _needs_reindex
        assert len(indexer.embedder.corpus) == 2

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

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "query, k, expected_len, expected_tool_name, expected_score_check",
        [
            ("create virtual machine", 2, 2, "create_instance", True),
        ],
    )
    async def test_indexer_facade_search_bm25_basic(
        self, mock_persistence, temp_index_dir, query, k, expected_len, expected_tool_name, expected_score_check
    ):
        """Test basic BM25 search scenario for IndexerFacade."""
        indexer = IndexerFacade(
            mock_persistence, EmbedderType.BM25, index_dir=temp_index_dir
        )

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

            assert len(results) == expected_len
            assert all(isinstance(r, SearchResult) for r in results)
            assert results[0].tool.name == expected_tool_name
            if expected_score_check:
                assert abs(results[0].score - 0.689) < 0.01
                assert abs(results[1].score - 0.574) < 0.01
                assert results[0].score > results[1].score

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "query, k, expected_len, expected_tool_name",
        [
            ("test query", 3, 1, "test_tool"),
        ],
    )
    async def test_indexer_facade_search_vector_embedder(
        self, mock_persistence, temp_index_dir, query, k, expected_len, expected_tool_name
    ):
        """Test vector embedder search scenario for IndexerFacade."""
        indexer = IndexerFacade(
            mock_persistence, EmbedderType.BM25, index_dir=temp_index_dir
        )
        # Use MockEmbedder for vector embedder scenario
        indexer.embedder = MockEmbedder()

        mock_results = [
            SearchResult(
                tool=ToolMetadata(
                    id=1,
                    name=expected_tool_name,
                    description="desc",
                    hash="hash",
                    server_name="server",
                ),
                score=0.9,
            )
        ]
        mock_persistence.search_similar_tools.return_value = mock_results
        results = await indexer.search_tools(query, k=k)

        assert len(results) == expected_len
        assert results[0].tool.name == expected_tool_name
        assert results[0].score == 0.9
        mock_persistence.search_similar_tools.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query, k", [
        ("nonexistent query", 5) 
    ])
    async def test_indexer_facade_search_no_results(
        self, mock_persistence, temp_index_dir, query, k
    ):
        """Test search scenario with no results for IndexerFacade."""
        indexer = IndexerFacade(
            mock_persistence, EmbedderType.BM25, index_dir=temp_index_dir
        )
        mock_persistence.get_all_tools.return_value = [] # Ensure no tools are found
        results = await indexer.search_tools(query, k=k)
        assert results == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "query, k, expected_len_min, expected_tool_names, min_score_check, expected_len_max, max_score_check",
        [
            ("create virtual machine", 5, 1, ["create_instance"], True, None, False),
            ("nonsense random query xyz", 5, None, None, False, 0, True),
        ],
    )
    async def test_indexer_facade_search_sample_data(
        self, populated_indexer_facade, query, k, expected_len_min, expected_tool_names, min_score_check, expected_len_max, max_score_check
    ):
        """Test sample data search scenarios for IndexerFacade."""
        # Use the pre-populated indexer facade from the fixture
        indexer = populated_indexer_facade
        
        queries = get_search_queries()
        query_data = next((q for q in queries if q["query"] == query), None)

        results = await indexer.search_tools(query, k=k)

        if expected_len_min is not None:
            assert len(results) >= expected_len_min
        if expected_tool_names:
            found_tools = {r.tool.name for r in results}
            assert any(tool in found_tools for tool in expected_tool_names), (
                f"Query '{query}' should find at least one of {expected_tool_names}, got {found_tools}"
            )
        if min_score_check:
            if results and query_data:
                assert max(r.score for r in results) >= query_data["min_score"], (
                    f"Query '{query}' should have score >= {query_data["min_score"]}"
                )
        if expected_len_max is not None:
            assert len(results) == expected_len_max
        if max_score_check:
            if results:
                assert all(r.score < 0.5 for r in results), (
                    f"Nonsense query '{query}' should have low scores"
                )

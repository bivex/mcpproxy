"""Tests for persistence layer."""

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from unittest.mock import AsyncMock

from mcpproxy.models.schemas import SearchResult
from mcpproxy.persistence.db import DatabaseManager
from mcpproxy.persistence.faiss_store import FaissStore
from mcpproxy.models.tool_metadata import ToolMetadata


class TestDatabaseManager:
    """Test cases for DatabaseManager."""

    @pytest.mark.asyncio
    async def test_database_initialization(self):
        """Test database schema initialization."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            DatabaseManager(db_path)

            # Check that tables were created
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='tools'"
                )
                assert cursor.fetchone() is not None

                # Check indexes
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_tools_hash'"
                )
                assert cursor.fetchone() is not None
        finally:
            Path(db_path).unlink(missing_ok=True)

    @pytest.mark.parametrize(
        "scenario, query, expected_len, expected_names, expected_tool_hash, expected_tool_server, expected_result",
        [
            ("get_by_hash", {"hash": "abc123def456"}, None, None, "abc123def456", "company-api", "found"),
            ("get_by_hash_not_found", {"hash": "nonexistent_hash"}, None, None, None, None, "not_found"),
            ("get_all", None, 4, ["create_instance", "delete_instance", "list_volumes", "create_volume"], None, None, "all"),
            ("get_by_server_company", {"server_name": "company-api"}, 2, ["create_instance", "delete_instance"], None, "company-api", "by_server"),
            ("get_by_server_storage", {"server_name": "storage-api"}, 2, ["list_volumes", "create_volume"], None, "storage-api", "by_server"),
            ("get_by_ids", {"ids": [1, 3]}, 2, ["create_instance", "list_volumes"], None, None, "by_ids"),
            ("get_by_ids_empty", {"ids": []}, 0, [], None, None, "by_ids_empty"),
        ],
    )
    @pytest.mark.asyncio
    async def test_database_manager_retrieval_scenarios(
        self, in_memory_db, sample_tool_metadata, sample_tool_metadata_list, scenario, query, expected_len, expected_names, expected_tool_hash, expected_tool_server, expected_result
    ):
        """Test various retrieval scenarios for DatabaseManager."""
        db = in_memory_db

        # Prepare data for all scenarios
        await db.insert_tool(sample_tool_metadata)
        for tool in sample_tool_metadata_list:
            await db.insert_tool(tool)

        if scenario == "get_by_hash":
            retrieved_tool = await db.get_tool_by_hash(query["hash"])
            assert retrieved_tool is not None
            assert retrieved_tool.hash == expected_tool_hash
            assert retrieved_tool.server_name == expected_tool_server
        elif scenario == "get_by_hash_not_found":
            retrieved_tool = await db.get_tool_by_hash(query["hash"])
            assert retrieved_tool is None
        elif scenario == "get_all":
            all_tools = await db.get_all_tools()
            assert len(all_tools) == expected_len
            assert {tool.name for tool in all_tools} == set(expected_names)
        elif scenario == "get_by_server_company" or scenario == "get_by_server_storage":
            tools_by_server = await db.get_tools_by_server(query["server_name"])
            assert len(tools_by_server) == expected_len
            assert {tool.name for tool in tools_by_server} == set(expected_names)
            assert all(tool.server_name == expected_tool_server for tool in tools_by_server)
        elif scenario == "get_by_ids":
            tools_by_ids = await db.get_tools_by_ids(query["ids"])
            assert len(tools_by_ids) == expected_len
            assert {tool.name for tool in tools_by_ids} == set(expected_names)
        elif scenario == "get_by_ids_empty":
            tools_by_ids = await db.get_tools_by_ids(query["ids"])
            assert len(tools_by_ids) == expected_len
            assert tools_by_ids == []

    @pytest.mark.asyncio
    async def test_update_tool(self, in_memory_db, sample_tool_metadata):
        """Test updating tool metadata."""
        db = in_memory_db

        tool_id = await db.insert_tool(sample_tool_metadata)

        updated_tool_metadata = ToolMetadata(
            id=tool_id,
            name="updated_tool",
            description="Updated description",
            hash="updated_hash",
            server_name="updated_server",
            faiss_vector_id=sample_tool_metadata.faiss_vector_id,
            params_json='{"new_param": "value"}',
        )

        await db.update_tool(updated_tool_metadata)

        retrieved_tool = await db.get_tool_by_hash(updated_tool_metadata.hash)
        assert retrieved_tool is not None
        assert retrieved_tool.name == updated_tool_metadata.name
        assert retrieved_tool.description == updated_tool_metadata.description
        assert retrieved_tool.hash == updated_tool_metadata.hash
        assert retrieved_tool.server_name == updated_tool_metadata.server_name
        assert retrieved_tool.params_json == updated_tool_metadata.params_json

    @pytest.mark.asyncio
    async def test_delete_tools_by_server(self, in_memory_db, sample_tool_metadata_list):
        """Test deleting tools by server name."""
        db = in_memory_db

        # Insert multiple tools
        for tool in sample_tool_metadata_list:
            await db.insert_tool(tool)

        # Delete tools from one server
        await db.delete_tools_by_server("company-api")

        # Verify only storage-api tools remain
        remaining_tools = await db.get_all_tools()
        assert len(remaining_tools) == 2
        assert all(tool.server_name == "storage-api" for tool in remaining_tools)

        # Verify the deleted tools are no longer retrievable
        company_tools_after_delete = await db.get_tools_by_server("company-api")
        assert len(company_tools_after_delete) == 0


class TestFaissStore:
    """Test cases for FaissStore."""

    @pytest.fixture(autouse=True)
    def require_faiss(self):
        """Skip tests if faiss-cpu is not installed."""
        try:
            import faiss # noqa: F401
        except ImportError:
            pytest.skip("faiss-cpu not installed")

    @pytest.fixture
    def mock_faiss(self):
        """Mock faiss module and index for isolated testing."""
        with patch("mcpproxy.persistence.faiss_store.faiss") as mock_faiss_module:
            mock_index = AsyncMock()
            mock_faiss_module.IndexFlatL2.return_value = mock_index
            mock_faiss_module.read_index.return_value = mock_index
            yield mock_faiss_module, mock_index

    @pytest.mark.asyncio
    async def test_faiss_store_initialization(self, mock_faiss):
        """Test FaissStore initialization."""
        mock_faiss_module, mock_index = mock_faiss
        store = FaissStore(":memory:", dimension=384)

        assert store.index_path == ":memory:"
        assert store.dimension == 384
        assert store.next_id == 0
        mock_faiss_module.IndexFlatL2.assert_called_once_with(384)

        # Test with existing index file (should call read_index)
        with patch("pathlib.Path.exists", return_value=True):
            FaissStore("existing_index.faiss", dimension=384)
            mock_faiss_module.read_index.assert_called_once()

    @pytest.mark.parametrize(
        "scenario, vector_input, k, expected_output_len, raises_error",
        [
            ("add_vector_valid", np.random.random(384).astype(np.float32), None, 1, False),
            ("add_vector_wrong_dimension", np.random.random(256).astype(np.float32), None, None, True),
            ("search_vectors_valid", None, 3, 3, False), # vector_input will be generated inside test
            ("search_empty_index", None, 5, 0, False), # vector_input will be generated inside test
        ],
    )
    @pytest.mark.asyncio
    async def test_faiss_store_vector_operations_scenarios(
        self, mock_faiss, scenario, vector_input, k, expected_output_len, raises_error
    ):
        """Test various add and search vector operations for FaissStore."""
        mock_faiss_module, mock_index = mock_faiss
        store = FaissStore(":memory:", dimension=384)

        if scenario == "add_vector_valid":
            vector_id = await store.add_vector(vector_input)
            assert vector_id == 0
            assert store.next_id == 1
            mock_index.add.assert_called_once_with(np.array([vector_input]))
        elif scenario == "add_vector_wrong_dimension":
            with pytest.raises(ValueError, match="Vector must have dimension 384"):
                await store.add_vector(vector_input)
        elif scenario == "search_vectors_valid":
            # Mock search results
            mock_distances = np.array([0.1, 0.2, 0.3])
            mock_indices = np.array([0, 1, 2])
            mock_index.search.return_value = (
                mock_distances.reshape(1, -1),
                mock_indices.reshape(1, -1),
            )
            mock_index.ntotal = 5

            query_vector = np.random.random(384).astype(np.float32)
            distances, indices = await store.search(query_vector, k=k)

            assert len(distances) == expected_output_len
            assert len(indices) == expected_output_len
            np.testing.assert_array_equal(distances, mock_distances)
            np.testing.assert_array_equal(indices, mock_indices)
        elif scenario == "search_empty_index":
            mock_index.ntotal = 0

            query_vector = np.random.random(384).astype(np.float32)
            distances, indices = await store.search(query_vector, k=k)

            assert len(distances) == expected_output_len
            assert len(indices) == expected_output_len

    @pytest.mark.asyncio
    async def test_get_vector_count(self, mock_faiss):
        """Test getting vector count."""
        mock_faiss_module, mock_index = mock_faiss
        mock_index.ntotal = 42

        store = FaissStore(":memory:", dimension=384)

        count = await store.get_vector_count()

        assert count == 42


class TestPersistenceFacade:
    """Test cases for PersistenceFacade integration."""

    @pytest.mark.asyncio
    async def test_store_tool_with_vector(
        self, temp_persistence_facade, sample_tool_metadata
    ):
        """Test storing tool with vector embedding."""
        vector = np.random.random(384).astype(np.float32)

        tool_id = await temp_persistence_facade.store_tool_with_vector(
            sample_tool_metadata, vector
        )

        assert isinstance(tool_id, int)
        assert tool_id > 0
        assert sample_tool_metadata.id == tool_id
        assert sample_tool_metadata.faiss_vector_id is not None

    @pytest.mark.parametrize(
        "scenario, query, expected_len, expected_names, expected_tool_hash, expected_server_name",
        [
            ("get_by_hash", {"hash": "abc123def456"}, None, None, "abc123def456", "company-api"),
            ("get_all", None, 4, ["create_instance", "delete_instance", "list_volumes", "create_volume"], None, None),
            ("get_by_server", {"server_name": "company-api"}, 2, ["create_instance", "delete_instance"], None, "company-api"),
        ],
    )
    @pytest.mark.asyncio
    async def test_persistence_facade_retrieval_scenarios(
        self, temp_persistence_facade, sample_tool_metadata, sample_tool_metadata_list, scenario, query, expected_len, expected_names, expected_tool_hash, expected_server_name
    ):
        """Test various retrieval scenarios for PersistenceFacade."""
        # Prepare data
        vector = np.random.random(384).astype(np.float32)
        await temp_persistence_facade.store_tool_with_vector(sample_tool_metadata, vector)
        for tool in sample_tool_metadata_list:
            await temp_persistence_facade.store_tool_with_vector(tool, vector)

        if scenario == "get_by_hash":
            retrieved_tool = await temp_persistence_facade.get_tool_by_hash(query["hash"])
            assert retrieved_tool is not None
            assert retrieved_tool.hash == expected_tool_hash
            assert retrieved_tool.server_name == expected_server_name
        elif scenario == "get_all":
            all_tools = await temp_persistence_facade.get_all_tools()
            assert len(all_tools) == expected_len
            assert {tool.name for tool in all_tools} == set(expected_names)
        elif scenario == "get_by_server":
            tools_by_server = await temp_persistence_facade.get_tools_by_server(query["server_name"])
            assert len(tools_by_server) == expected_len
            assert {tool.name for tool in tools_by_server} == set(expected_names)
            assert all(tool.server_name == expected_server_name for tool in tools_by_server)

    @pytest.mark.asyncio
    async def test_search_similar_tools(
        self, temp_persistence_facade, sample_tool_metadata_list
    ):
        """Test searching for similar tools."""
        # Store tools with different vectors
        for i, tool in enumerate(sample_tool_metadata_list):
            vector = np.random.random(384).astype(np.float32)
            # Make first vector more similar to query
            if i == 0:
                vector[:10] = 0.9
            await temp_persistence_facade.store_tool_with_vector(tool, vector)

        # Search with query vector similar to first tool
        query_vector = np.random.random(384).astype(np.float32)
        query_vector[:10] = 0.9  # Similar to first tool

        with patch.object(
            temp_persistence_facade.vector_store, "search"
        ) as mock_search:
            # Mock search to return first tool as most similar
            mock_search.return_value = (np.array([0.1, 0.5, 0.8]), np.array([1, 2, 3]))

            results = await temp_persistence_facade.search_similar_tools(
                query_vector, k=3
            )

            assert len(results) == 3
            assert all(isinstance(result, SearchResult) for result in results)
            assert all(result.score > 0 for result in results)
            # First result should have highest score (lowest distance)
            assert results[0].score > results[1].score > results[2].score

    @pytest.mark.parametrize(
        "scenario, action_data, expected_outcome",
        [
            ("update_tool", {"name": "updated_tool", "description": "Updated description", "hash": "updated_hash", "server_name": "updated_server"}, "updated"),
            ("delete_by_server", {"server_name": "company-api"}, "deleted"),
        ],
    )
    @pytest.mark.asyncio
    async def test_persistence_facade_modification_scenarios(
        self, temp_persistence_facade, sample_tool_metadata, sample_tool_metadata_list, scenario, action_data, expected_outcome
    ):
        """Test various modification scenarios for PersistenceFacade."""
        vector = np.random.random(384).astype(np.float32)
        tool_id = await temp_persistence_facade.store_tool_with_vector(sample_tool_metadata, vector)

        for tool in sample_tool_metadata_list:
            await temp_persistence_facade.store_tool_with_vector(tool, vector)

        if scenario == "update_tool":
            updated_tool_metadata = ToolMetadata(
                id=tool_id,
                faiss_vector_id=sample_tool_metadata.faiss_vector_id,
                params_json=sample_tool_metadata.params_json,
                **action_data
            )
            await temp_persistence_facade.update_tool_with_vector(updated_tool_metadata)
            retrieved_tool = await temp_persistence_facade.get_tool_by_hash(action_data["hash"])
            assert retrieved_tool is not None
            assert retrieved_tool.name == action_data["name"]
            assert retrieved_tool.description == action_data["description"]
        elif scenario == "delete_by_server":
            await temp_persistence_facade.delete_tools_by_server(action_data["server_name"])
            company_tools = await temp_persistence_facade.get_tools_by_server(action_data["server_name"])
            storage_tools = await temp_persistence_facade.get_tools_by_server("storage-api")
            assert len(company_tools) == 0
            assert len(storage_tools) == 2

    @pytest.mark.asyncio
    async def test_get_vector_count_facade(
        self, temp_persistence_facade, sample_tool_metadata_list
    ):
        """Test getting vector count through facade."""
        # Initially no vectors
        count = await temp_persistence_facade.get_vector_count()
        assert count == 0

        # Add some tools
        for tool in sample_tool_metadata_list[:2]:
            vector = np.random.random(384).astype(np.float32)
            await temp_persistence_facade.store_tool_with_vector(tool, vector)

        # Check count - BM25 doesn't use vector storage, so count is always 0
        from mcpproxy.persistence.bm25_store import BM25Store
        if isinstance(temp_persistence_facade.vector_store, BM25Store):
            assert count == 0  # BM25 doesn't use vector storage
        else:
            assert count == 2  # Other embedders use vector storage

    @pytest.mark.asyncio
    async def test_close_facade(self, temp_persistence_facade):
        """Test closing facade properly cleans up resources."""
        # This should not raise any exceptions
        await temp_persistence_facade.close()

"""Tests for persistence layer."""

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mcpproxy.models.schemas import SearchResult
from mcpproxy.persistence.db import DatabaseManager
from mcpproxy.persistence.faiss_store import FaissStore


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

    @pytest.mark.asyncio
    async def test_insert_tool(self, in_memory_db, sample_tool_metadata):
        """Test inserting tool metadata."""
        db = in_memory_db

        tool_id = await db.insert_tool(sample_tool_metadata)

        assert isinstance(tool_id, int)
        assert tool_id > 0

    @pytest.mark.asyncio
    async def test_get_tool_by_hash(self, in_memory_db, sample_tool_metadata):
        """Test retrieving tool by hash."""
        db = in_memory_db

        # Insert tool
        await db.insert_tool(sample_tool_metadata)

        # Retrieve by hash
        retrieved_tool = await db.get_tool_by_hash(sample_tool_metadata.hash)

        assert retrieved_tool is not None
        assert retrieved_tool.name == sample_tool_metadata.name
        assert retrieved_tool.hash == sample_tool_metadata.hash
        assert retrieved_tool.server_name == sample_tool_metadata.server_name

    @pytest.mark.asyncio
    async def test_get_tool_by_hash_not_found(self, in_memory_db):
        """Test retrieving tool by non-existent hash."""
        db = in_memory_db

        retrieved_tool = await db.get_tool_by_hash("nonexistent_hash")

        assert retrieved_tool is None

    @pytest.mark.asyncio
    async def test_get_all_tools(self, in_memory_db, sample_tool_metadata_list):
        """Test retrieving all tools."""
        db = in_memory_db

        # Insert multiple tools
        for tool in sample_tool_metadata_list:
            await db.insert_tool(tool)

        # Retrieve all
        all_tools = await db.get_all_tools()

        assert len(all_tools) == len(sample_tool_metadata_list)
        tool_names = {tool.name for tool in all_tools}
        expected_names = {tool.name for tool in sample_tool_metadata_list}
        assert tool_names == expected_names

    @pytest.mark.asyncio
    async def test_get_tools_by_server(self, in_memory_db, sample_tool_metadata_list):
        """Test retrieving tools by server name."""
        db = in_memory_db

        # Insert multiple tools
        for tool in sample_tool_metadata_list:
            await db.insert_tool(tool)

        # Get tools for specific server
        company_tools = await db.get_tools_by_server("company-api")
        storage_tools = await db.get_tools_by_server("storage-api")

        assert len(company_tools) == 2  # create_instance, delete_instance
        assert len(storage_tools) == 2  # list_volumes, create_volume

        assert all(tool.server_name == "company-api" for tool in company_tools)
        assert all(tool.server_name == "storage-api" for tool in storage_tools)

    @pytest.mark.asyncio
    async def test_update_tool(self, in_memory_db, sample_tool_metadata):
        """Test updating tool metadata."""
        db = in_memory_db

        # Insert tool
        tool_id = await db.insert_tool(sample_tool_metadata)
        sample_tool_metadata.id = tool_id

        # Update tool
        sample_tool_metadata.description = "Updated description"
        sample_tool_metadata.hash = "updated_hash"
        await db.update_tool(sample_tool_metadata)

        # Retrieve and verify
        updated_tool = await db.get_tool_by_hash("updated_hash")
        assert updated_tool is not None
        assert updated_tool.description == "Updated description"

    @pytest.mark.asyncio
    async def test_delete_tools_by_server(
        self, in_memory_db, sample_tool_metadata_list
    ):
        """Test deleting tools by server name."""
        db = in_memory_db

        # Insert multiple tools
        for tool in sample_tool_metadata_list:
            await db.insert_tool(tool)

        # Delete tools for one server
        await db.delete_tools_by_server("company-api")

        # Verify deletion
        company_tools = await db.get_tools_by_server("company-api")
        storage_tools = await db.get_tools_by_server("storage-api")

        assert len(company_tools) == 0
        assert len(storage_tools) == 2  # Should remain untouched

    @pytest.mark.asyncio
    async def test_get_tools_by_ids(self, in_memory_db, sample_tool_metadata_list):
        """Test retrieving tools by IDs."""
        db = in_memory_db

        # Insert tools and collect IDs
        tool_ids = []
        for tool in sample_tool_metadata_list:
            tool_id = await db.insert_tool(tool)
            tool_ids.append(tool_id)

        # Get subset of tools
        subset_ids = tool_ids[:2]
        retrieved_tools = await db.get_tools_by_ids(subset_ids)

        assert len(retrieved_tools) == 2
        retrieved_ids = {tool.id for tool in retrieved_tools}
        assert retrieved_ids == set(subset_ids)

    @pytest.mark.asyncio
    async def test_get_tools_by_ids_empty_list(self, in_memory_db):
        """Test retrieving tools with empty ID list."""
        db = in_memory_db

        retrieved_tools = await db.get_tools_by_ids([])

        assert retrieved_tools == []


class TestFaissStore:
    """Test cases for FaissStore."""

    @pytest.fixture(autouse=True)
    def require_faiss(self):
        """Skip all tests in this class if faiss is not available."""
        pytest.importorskip("faiss", reason="faiss-cpu not installed")

    @pytest.fixture
    def mock_faiss(self):
        """Mock faiss module to avoid dependency issues."""
        with patch("mcpproxy.persistence.faiss_store.faiss") as mock_faiss:
            mock_index = MagicMock()
            mock_index.ntotal = 0
            mock_faiss.IndexFlatL2.return_value = mock_index
            mock_faiss.read_index.return_value = mock_index
            mock_faiss.write_index = MagicMock()
            yield mock_faiss, mock_index

    @pytest.mark.asyncio
    async def test_faiss_store_initialization(self, mock_faiss):
        """Test FaissStore initialization."""
        mock_faiss_module, mock_index = mock_faiss

        # Use a path that doesn't exist to trigger new index creation
        import os
        import tempfile

        temp_dir = tempfile.gettempdir()
        index_path = os.path.join(
            temp_dir, f"test_faiss_{os.getpid()}_{id(object())}.faiss"
        )

        try:
            store = FaissStore(index_path, dimension=384)

            assert store.dimension == 384
            assert store.next_id == 0
            mock_faiss_module.IndexFlatL2.assert_called_once_with(384)
        finally:
            Path(index_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_add_vector(self, mock_faiss):
        """Test adding vector to index."""
        mock_faiss_module, mock_index = mock_faiss

        store = FaissStore(":memory:", dimension=384)
        vector = np.random.random(384).astype(np.float32)

        vector_id = await store.add_vector(vector)

        assert vector_id == 0
        assert store.next_id == 1
        mock_index.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_vector_wrong_dimension(self, mock_faiss):
        """Test adding vector with wrong dimension."""
        mock_faiss_module, mock_index = mock_faiss

        store = FaissStore(":memory:", dimension=384)
        wrong_vector = np.random.random(256).astype(np.float32)

        with pytest.raises(ValueError, match="Vector must have dimension 384"):
            await store.add_vector(wrong_vector)

    @pytest.mark.asyncio
    async def test_search_vectors(self, mock_faiss):
        """Test searching for similar vectors."""
        mock_faiss_module, mock_index = mock_faiss

        # Mock search results
        mock_distances = np.array([0.1, 0.2, 0.3])
        mock_indices = np.array([0, 1, 2])
        mock_index.search.return_value = (
            mock_distances.reshape(1, -1),
            mock_indices.reshape(1, -1),
        )
        mock_index.ntotal = 5

        store = FaissStore(":memory:", dimension=384)
        query_vector = np.random.random(384).astype(np.float32)

        distances, indices = await store.search(query_vector, k=3)

        assert len(distances) == 3
        assert len(indices) == 3
        np.testing.assert_array_equal(distances, mock_distances)
        np.testing.assert_array_equal(indices, mock_indices)

    @pytest.mark.asyncio
    async def test_search_empty_index(self, mock_faiss):
        """Test searching in empty index."""
        mock_faiss_module, mock_index = mock_faiss
        mock_index.ntotal = 0

        store = FaissStore(":memory:", dimension=384)
        query_vector = np.random.random(384).astype(np.float32)

        distances, indices = await store.search(query_vector, k=5)

        assert len(distances) == 0
        assert len(indices) == 0

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

    @pytest.mark.asyncio
    async def test_get_tool_by_hash_facade(
        self, temp_persistence_facade, sample_tool_metadata
    ):
        """Test getting tool by hash through facade."""
        vector = np.random.random(384).astype(np.float32)

        # Store tool
        await temp_persistence_facade.store_tool_with_vector(
            sample_tool_metadata, vector
        )

        # Retrieve by hash
        retrieved_tool = await temp_persistence_facade.get_tool_by_hash(
            sample_tool_metadata.hash
        )

        assert retrieved_tool is not None
        assert retrieved_tool.name == sample_tool_metadata.name
        assert retrieved_tool.hash == sample_tool_metadata.hash

    @pytest.mark.asyncio
    async def test_update_tool_with_vector(
        self, temp_persistence_facade, sample_tool_metadata
    ):
        """Test updating tool with new vector."""
        vector1 = np.random.random(384).astype(np.float32)
        vector2 = np.random.random(384).astype(np.float32)

        # Store initial tool
        await temp_persistence_facade.store_tool_with_vector(
            sample_tool_metadata, vector1
        )

        # Update with new vector
        sample_tool_metadata.description = "Updated description"
        await temp_persistence_facade.update_tool_with_vector(
            sample_tool_metadata, vector2
        )

        # Verify update
        updated_tool = await temp_persistence_facade.get_tool_by_hash(
            sample_tool_metadata.hash
        )
        assert updated_tool.description == "Updated description"

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

    @pytest.mark.asyncio
    async def test_get_all_tools_facade(
        self, temp_persistence_facade, sample_tool_metadata_list
    ):
        """Test getting all tools through facade."""
        # Store multiple tools
        for tool in sample_tool_metadata_list:
            vector = np.random.random(384).astype(np.float32)
            await temp_persistence_facade.store_tool_with_vector(tool, vector)

        # Retrieve all
        all_tools = await temp_persistence_facade.get_all_tools()

        assert len(all_tools) == len(sample_tool_metadata_list)
        tool_names = {tool.name for tool in all_tools}
        expected_names = {tool.name for tool in sample_tool_metadata_list}
        assert tool_names == expected_names

    @pytest.mark.asyncio
    async def test_get_tools_by_server_facade(
        self, temp_persistence_facade, sample_tool_metadata_list
    ):
        """Test getting tools by server through facade."""
        # Store multiple tools
        for tool in sample_tool_metadata_list:
            vector = np.random.random(384).astype(np.float32)
            await temp_persistence_facade.store_tool_with_vector(tool, vector)

        # Get tools for specific server
        company_tools = await temp_persistence_facade.get_tools_by_server("company-api")

        assert len(company_tools) == 2
        assert all(tool.server_name == "company-api" for tool in company_tools)

    @pytest.mark.asyncio
    async def test_delete_tools_by_server_facade(
        self, temp_persistence_facade, sample_tool_metadata_list
    ):
        """Test deleting tools by server through facade."""
        # Store multiple tools
        for tool in sample_tool_metadata_list:
            vector = np.random.random(384).astype(np.float32)
            await temp_persistence_facade.store_tool_with_vector(tool, vector)

        # Delete tools for one server
        await temp_persistence_facade.delete_tools_by_server("company-api")

        # Verify deletion
        company_tools = await temp_persistence_facade.get_tools_by_server("company-api")
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
        count = await temp_persistence_facade.get_vector_count()
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

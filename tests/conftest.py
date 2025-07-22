"""Pytest configuration and common fixtures for Smart MCP Proxy tests."""

import asyncio
import tempfile
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
import pytest

from mcpproxy.indexer.facade import IndexerFacade
from mcpproxy.models.schemas import (
    EmbedderType,
    ProxyConfig,
    SearchResult,
    ServerConfig,
    ToolMetadata,
)
from mcpproxy.persistence.db import DatabaseManager
from mcpproxy.persistence.facade import PersistenceFacade
from .fixtures.data import sample_tool_metadata, sample_tool_metadata_list, sample_search_result, sample_proxy_config, sample_embeddings, sample_tool_params, sample_tool_tags, sample_tool_annotations


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def temp_db_path():
    """Temporary database path for testing."""
    # Generate a temporary file path but don't create the file
    import os

    temp_dir = tempfile.gettempdir()
    db_path = os.path.join(temp_dir, f"test_db_{os.getpid()}_{id(object())}.db")
    yield db_path
    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
async def temp_faiss_path():
    """Temporary Faiss index path for testing."""
    # Generate a temporary file path but don't create the file
    import os

    temp_dir = tempfile.gettempdir()
    faiss_path = os.path.join(
        temp_dir, f"test_faiss_{os.getpid()}_{id(object())}.faiss"
    )
    yield faiss_path
    # Cleanup
    Path(faiss_path).unlink(missing_ok=True)


@pytest.fixture
async def in_memory_db() -> AsyncGenerator[DatabaseManager, None]:
    """In-memory SQLite database for testing."""
    # Use :memory: for pure in-memory database
    db = DatabaseManager(":memory:")
    yield db
    # No cleanup needed for in-memory database


@pytest.fixture
async def temp_persistence_facade(
    temp_db_path, temp_faiss_path
) -> AsyncGenerator[PersistenceFacade, None]:
    """Temporary persistence facade with isolated storage."""
    facade = PersistenceFacade(
        db_path=temp_db_path, 
        index_path=temp_faiss_path, 
        vector_dimension=384,
        embedder_type=EmbedderType.BM25,
    )
    yield facade
    await facade.close()
    # Cleanup files
    Path(temp_db_path).unlink(missing_ok=True)
    Path(temp_faiss_path).unlink(missing_ok=True)


@pytest.fixture
async def temp_indexer_facade(
    temp_db_path, temp_faiss_path
) -> AsyncGenerator[IndexerFacade, None]:
    """Temporary indexer facade for testing."""

    # Create temporary directory for BM25 index
    with tempfile.TemporaryDirectory() as temp_bm25_dir:
        # Create persistence facade - BM25 doesn't need vector dimension since it doesn't use faiss
        persistence_facade = PersistenceFacade(
            db_path=temp_db_path,
            index_path=temp_faiss_path,
            vector_dimension=1,  # Placeholder dimension for BM25
            embedder_type=EmbedderType.BM25,
        )

        indexer = IndexerFacade(
            persistence=persistence_facade,
            embedder_type=EmbedderType.BM25,
            index_dir=temp_bm25_dir,
        )

        yield indexer

        await persistence_facade.close()
        # Cleanup files
        Path(temp_db_path).unlink(missing_ok=True)
        Path(temp_faiss_path).unlink(missing_ok=True)
        # temp_bm25_dir is automatically cleaned up by TemporaryDirectory

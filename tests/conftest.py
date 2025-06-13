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


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_tool_metadata() -> ToolMetadata:
    """Sample tool metadata for testing."""
    return ToolMetadata(
        id=1,
        name="create_instance",
        description="Create a new virtual machine instance",
        hash="abc123def456",
        server_name="company-api",
        faiss_vector_id=0,
        params_json='{"type": "object", "properties": {"name": {"type": "string"}, "flavor": {"type": "string"}}}',
    )


@pytest.fixture
def sample_tool_metadata_list() -> list[ToolMetadata]:
    """List of sample tool metadata for testing."""
    return [
        ToolMetadata(
            id=1,
            name="create_instance",
            description="Create a new virtual machine instance",
            hash="hash1",
            server_name="company-api",
            faiss_vector_id=0,
            params_json='{"type": "object", "properties": {"name": {"type": "string"}}}',
        ),
        ToolMetadata(
            id=2,
            name="delete_instance",
            description="Delete an existing virtual machine instance",
            hash="hash2",
            server_name="company-api",
            faiss_vector_id=1,
            params_json='{"type": "object", "properties": {"instance_id": {"type": "string"}}}',
        ),
        ToolMetadata(
            id=3,
            name="list_volumes",
            description="List all storage volumes",
            hash="hash3",
            server_name="storage-api",
            faiss_vector_id=2,
            params_json='{"type": "object", "properties": {"region": {"type": "string"}}}',
        ),
        ToolMetadata(
            id=4,
            name="create_volume",
            description="Create a new storage volume",
            hash="hash4",
            server_name="storage-api",
            faiss_vector_id=3,
            params_json='{"type": "object", "properties": {"size": {"type": "integer"}}}',
        ),
    ]


@pytest.fixture
def sample_search_result(sample_tool_metadata) -> SearchResult:
    """Sample search result for testing."""
    return SearchResult(tool=sample_tool_metadata, score=0.95)


@pytest.fixture
def sample_proxy_config() -> ProxyConfig:
    """Sample proxy configuration for testing."""
    return ProxyConfig(
        mcp_servers={
            "company-api": ServerConfig(url="http://localhost:8080/mcp"),
            "storage-api": ServerConfig(url="http://localhost:8081/mcp"),
            "local-tools": ServerConfig(
                command="python", args=["server.py"], env={"API_KEY": "test-key"}
            ),
        },
        embedder=EmbedderType.BM25,
        hf_model="sentence-transformers/all-MiniLM-L6-v2",
        top_k=5,
    )


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


@pytest.fixture
def sample_embeddings() -> list[np.ndarray]:
    """Sample embedding vectors for testing."""
    return [
        np.random.random(384).astype(np.float32),
        np.random.random(384).astype(np.float32),
        np.random.random(384).astype(np.float32),
        np.random.random(384).astype(np.float32),
    ]


@pytest.fixture
def sample_tool_params() -> dict:
    """Sample tool parameters schema."""
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Instance name"},
            "flavor": {"type": "string", "description": "Instance flavor/size"},
            "region": {
                "type": "string",
                "description": "Deployment region",
                "default": "us-east-1",
            },
        },
        "required": ["name", "flavor"],
    }


@pytest.fixture
def sample_tool_tags() -> list[str]:
    """Sample tool tags."""
    return ["compute", "vm", "creation", "cloud"]


@pytest.fixture
def sample_tool_annotations() -> dict:
    """Sample tool annotations."""
    return {
        "category": "compute",
        "cost": "medium",
        "permissions": ["instance:create"],
        "rate_limit": "10/minute",
    }

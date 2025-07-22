"""Sample data fixtures for testing."""

import pytest
from mcpproxy.models.schemas import EmbedderType, ProxyConfig, SearchResult, ServerConfig, ToolMetadata
import numpy as np
from typing import AsyncGenerator
from mcpproxy.persistence.db import DatabaseManager
from mcpproxy.persistence.facade import PersistenceFacade
import tempfile
from pathlib import Path
import os
import asyncio


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
def sample_embeddings() -> list[np.ndarray]:
    """Sample embeddings for testing."""
    return [
        np.array([0.1, 0.2, 0.3], dtype=np.float32),
        np.array([0.4, 0.5, 0.6], dtype=np.float32),
        np.array([0.7, 0.8, 0.9], dtype=np.float32),
    ]


@pytest.fixture
def sample_tool_params() -> dict:
    """Sample tool parameters for testing."""
    return {"name": "test_name", "value": 123}


@pytest.fixture
def sample_tool_tags() -> list[str]:
    """Sample tool tags for testing."""
    return ["tag1", "tag2"]


@pytest.fixture
def sample_tool_annotations() -> dict:
    """Sample tool annotations for testing."""
    return {"annotation_key": "annotation_value"}


from ..utils.sample_data_generators import get_sample_tools_data, get_sample_config_json, get_search_queries, get_sample_embeddings_data, get_hash_test_cases

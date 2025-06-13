"""Persistence layer facade for easy access to database and vector store."""

import os
import tempfile
from pathlib import Path

import numpy as np

from ..models.schemas import EmbedderType, SearchResult, ToolMetadata
from .db import DatabaseManager
from .bm25_store import BM25Store


def _get_data_directory() -> Path:
    """Get the data directory for storing database and index files.
    
    Checks in order:
    1. MCPPROXY_DATA_DIR environment variable
    2. ~/.mcpproxy/ (user home directory)
    3. /tmp/mcpproxy/ (fallback)
    
    Creates the directory if it doesn't exist.
    
    Returns:
        Path: The data directory path
        
    Raises:
        OSError: If unable to create or access the data directory
    """
    # Check environment variable first
    data_dir_env = os.getenv("MCPPROXY_DATA_DIR")
    if data_dir_env:
        data_dir = Path(data_dir_env).expanduser().resolve()
    else:
        # Try user home directory first
        try:
            home_dir = Path.home()
            data_dir = home_dir / ".mcpproxy"
        except (OSError, RuntimeError):
            # Fallback to temp directory if home is not accessible
            data_dir = Path(tempfile.gettempdir()) / "mcpproxy"
    
    # Ensure directory exists and is writable
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Test write permissions by creating a temporary file
        test_file = data_dir / ".test_write"
        test_file.touch()
        test_file.unlink()
        
        return data_dir
    except (OSError, PermissionError) as e:
        # If we can't create or write to the chosen directory, fallback to temp
        if data_dir_env:
            # User explicitly set a directory but it's not usable
            raise OSError(
                f"Cannot create or write to specified data directory: {data_dir}. "
                f"Please check permissions or set MCPPROXY_DATA_DIR to a writable location."
            ) from e
        
        # Try temp fallback
        try:
            temp_data_dir = Path(tempfile.gettempdir()) / "mcpproxy"
            temp_data_dir.mkdir(parents=True, exist_ok=True)
            
            # Test write permissions
            test_file = temp_data_dir / ".test_write"
            test_file.touch()
            test_file.unlink()
            
            return temp_data_dir
        except (OSError, PermissionError) as temp_e:
            raise OSError(
                f"Cannot create data directory. Tried: {data_dir} and {temp_data_dir}. "
                "Please set MCPPROXY_DATA_DIR environment variable to a writable location."
            ) from temp_e


class PersistenceFacade:
    """Unified facade for database and vector store operations."""

    def __init__(
        self,
        db_path: str | None = None,
        index_path: str | None = None,
        vector_dimension: int = 384,
        embedder_type: EmbedderType = EmbedderType.BM25,
    ):
        # Resolve data directory and file paths
        data_dir = _get_data_directory()
        
        # Use provided paths or defaults within data directory
        if db_path is None:
            db_path = str(data_dir / "proxy.db")
        elif not os.path.isabs(db_path):
            # Convert relative paths to be within data directory
            db_path = str(data_dir / db_path)
            
        if index_path is None:
            index_path = str(data_dir / "tools.faiss")
        elif not os.path.isabs(index_path):
            # Convert relative paths to be within data directory
            index_path = str(data_dir / index_path)
        
        self.db = DatabaseManager(db_path)
        
        # Use appropriate vector store based on embedder type
        if embedder_type == EmbedderType.BM25:
            self.vector_store = BM25Store(index_path, vector_dimension)
        else:
            # Only import faiss when needed for vector embedders
            from .faiss_store import FaissStore
            self.vector_store = FaissStore(index_path, vector_dimension)

    async def store_tool_with_vector(
        self, tool: ToolMetadata, vector: np.ndarray
    ) -> int:
        """Store tool metadata and its vector embedding."""
        # First add vector to get its ID
        vector_id = await self.vector_store.add_vector(vector)
        tool.faiss_vector_id = vector_id

        # Then store in database
        tool_id = await self.db.insert_tool(tool)
        tool.id = tool_id
        return tool_id

    async def update_tool_with_vector(
        self, tool: ToolMetadata, vector: np.ndarray
    ) -> None:
        """Update tool metadata and its vector embedding."""
        if tool.faiss_vector_id is not None:
            await self.vector_store.update_vector(tool.faiss_vector_id, vector)
        else:
            vector_id = await self.vector_store.add_vector(vector)
            tool.faiss_vector_id = vector_id

        await self.db.update_tool(tool)

    async def search_similar_tools(
        self, query_vector: np.ndarray, k: int = 5
    ) -> list[SearchResult]:
        """Search for similar tools using vector similarity."""
        distances, indices = await self.vector_store.search(query_vector, k)

        if len(indices) == 0:
            return []

        # Get tools by their database IDs
        # Note: This assumes vector_id maps to tool.id, which may need adjustment
        tools = await self.db.get_tools_by_ids(indices.tolist())

        results = []
        for i, tool in enumerate(tools):
            # Convert distance to similarity score (lower distance = higher similarity)
            score = 1.0 / (1.0 + float(distances[i]))
            results.append(SearchResult(tool=tool, score=score))

        return results

    async def get_tool_by_hash(self, hash: str) -> ToolMetadata | None:
        """Get tool by its hash."""
        return await self.db.get_tool_by_hash(hash)

    async def get_all_tools(self) -> list[ToolMetadata]:
        """Get all stored tools."""
        return await self.db.get_all_tools()

    async def get_tools_by_server(self, server_name: str) -> list[ToolMetadata]:
        """Get all tools for a specific server."""
        return await self.db.get_tools_by_server(server_name)

    async def delete_tools_by_server(self, server_name: str) -> None:
        """Delete all tools for a server."""
        await self.db.delete_tools_by_server(server_name)

    async def get_vector_count(self) -> int:
        """Get total number of vectors in the store."""
        return await self.vector_store.get_vector_count()

    async def reset_all_data(self) -> None:
        """Reset all data (database and vector store)."""
        await self.db.reset_database()
        await self.vector_store.reset()

    async def close(self) -> None:
        """Close database and vector store connections."""
        await self.vector_store.close()

"""Persistence layer facade for easy access to database and vector store."""

import numpy as np

from ..models.schemas import SearchResult, ToolMetadata
from .db import DatabaseManager
from .faiss_store import FaissStore


class PersistenceFacade:
    """Unified facade for database and vector store operations."""

    def __init__(
        self,
        db_path: str = "proxy.db",
        index_path: str = "tools.faiss",
        vector_dimension: int = 384,
    ):
        self.db = DatabaseManager(db_path)
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

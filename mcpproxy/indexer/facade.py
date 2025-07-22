"""Indexer and search facade."""

import os
from typing import Any
import json
import math

from mcpproxy.models.schemas import ToolMetadata, SearchResult, ToolData, EmbedderType
from mcpproxy.persistence.facade import PersistenceFacade
from mcpproxy.utils.dependency_management.dependencies import check_embedder_dependencies
from mcpproxy.utils.hashing.hashing import compute_tool_hash
from mcpproxy.utils.name_sanitization.name_sanitizer import sanitize_tool_name
from mcpproxy.indexer.embedders.base import BaseEmbedder
from mcpproxy.indexer.embedders.bm25 import BM25Embedder
from mcpproxy.indexer.embedders.openai import OpenAIEmbedder
from mcpproxy.indexer.embedders.huggingface import HuggingFaceEmbedder
from mcpproxy.models.constants import DEFAULT_SEARCH_RESULT_LIMIT, MIN_NORMALIZED_SCORE_THRESHOLD


class IndexerFacade:
    """Facade for indexing and searching tools."""

    def __init__(
        self,
        persistence: PersistenceFacade,
        embedder_type: EmbedderType = EmbedderType.BM25,
        hf_model: str | None = None,
        index_dir: str | None = None,
    ):
        self.persistence = persistence
        self.embedder_type = embedder_type
        self.embedder = self._create_embedder(embedder_type, hf_model, index_dir)
        self._needs_reindex = False  # Flag to indicate if re-indexing is needed

    def _create_embedder(
        self, embedder_type: EmbedderType, hf_model: str | None, index_dir: str | None
    ) -> BaseEmbedder:
        check_embedder_dependencies(embedder_type)
        if embedder_type == EmbedderType.BM25:
            return BM25Embedder(index_dir=index_dir)
        elif embedder_type == EmbedderType.HF:
            model_name = hf_model or "sentence-transformers/all-MiniLM-L6-v2"
            return HuggingFaceEmbedder(model_name)
        elif embedder_type == EmbedderType.OPENAI:
            return OpenAIEmbedder()
        else:
            raise ValueError(f"Unknown embedder type: {embedder_type}")

    async def index_tool(
        self, tool_data: ToolData
    ) -> None:
        """
        Indexes a single tool.

        If the tool already exists and has not changed, it skips re-indexing.
        Otherwise, it processes and stores the tool.
        """
        server_name = tool_data.server_name
        # Generate a hash for the current tool data to detect changes
        tool_hash = compute_tool_hash(tool_data)

        # Check if the tool exists and is unchanged
        if await self._is_tool_unchanged(tool_hash):
            return

        # Process and store the tool
        await self._process_and_store_tool(tool_data, tool_hash, server_name)

    async def index_tool_from_object(self, tool_obj: Any, server_name: str) -> None:
        """
        Extracts tool data from a tool object and indexes it.
        This method is used when new tools are discovered or updated.
        """
        tool_data = self._extract_tool_data_from_obj(tool_obj, server_name)
        await self.index_tool(tool_data)

    async def _is_tool_unchanged(self, tool_hash: str) -> bool:
        """Check if the tool exists and its hash is unchanged in the database."""
        return await self.persistence.get_tool_by_hash(tool_hash) is not None

    async def _process_and_store_tool(
        self, tool_data: ToolData, tool_hash: str, server_name: str
    ) -> None:
        """Processes tool data, stores it, and updates the index."""
        # Sanitize tool name for consistent indexing and searching
        tool_data.name = sanitize_tool_name(tool_data.name)

        # Index the tool content
        tool_content_to_index = f"{tool_data.name} {tool_data.description} {tool_data.args}"
        if isinstance(self.embedder, BM25Embedder):
            await self.embedder.add_texts_to_corpus(
                [tool_content_to_index], [tool_data.id]
            )
        else:
            vector = await self.embedder.embed_text(tool_content_to_index)
            tool_data.embedding = vector.tolist()  # Store as list for JSON serialization

        # Store minimal metadata in persistence layer (only for hash-based change detection)
        tool = ToolMetadata(
            name=tool_data.name,
            description=tool_data.description,
            args=self._serialize_params_to_json(tool_data.args),
            hash=tool_hash,
            id=tool_data.id,
            server_name=server_name,
            last_used_at=tool_data.last_used_at,
            embedding=tool_data.embedding,
        )
        await self.persistence.upsert_tool(tool)
        self._needs_reindex = True  # Mark for re-indexing for BM25 after updates

    def _serialize_params_to_json(self, params: dict[str, Any] | None) -> str | None:
        return json.dumps(params) if params is not None else None

    def _extract_tool_data_from_obj(self, tool_obj: Any, server_name: str) -> ToolData:
        # Placeholder for extracting ToolData from a tool object.
        # This will depend on the actual structure of tool_obj.
        return ToolData(
            id=tool_obj.id,
            name=tool_obj.name,
            description=tool_obj.description,
            args=tool_obj.args,
            server_name=server_name,
            last_used_at=tool_obj.last_used_at,
        )

    def _get_annotations_string(self, annotations: Any) -> str | None:
        # Placeholder for extracting annotations as a string.
        return str(annotations) if annotations else None

    async def reindex_all_tools(self) -> None:
        """Reindex all tools in the system. This is a costly operation."""
        self.embedder.clear_index()
        all_tools = await self.persistence.get_all_tools()
        if isinstance(self.embedder, BM25Embedder):
            corpus = [
                f"{tool.name} {tool.description} {tool.args}" for tool in all_tools
            ]
            ids = [tool.id for tool in all_tools]
            await self.embedder.add_texts_to_corpus(corpus, ids)
        # For vector embedders, embeddings are stored directly with the tool metadata,
        # so no separate re-indexing of the embedder is needed.
        self._needs_reindex = False

    async def reset_embedder_data(self) -> None:
        """Reset embedder-specific data."""
        if isinstance(self.embedder, BM25Embedder):
            await self.embedder.reset_stored_data()
        # For other embedders, no additional reset needed beyond persistence reset

    async def ensure_index_ready(self) -> None:
        """Ensure the index is ready for search (reindex if needed)."""
        if isinstance(self.embedder, BM25Embedder):
            # If we need reindexing or the embedder has no corpus
            if self._needs_reindex or not self.embedder.is_indexed():
                await self.reindex_all_tools()

    async def search_tools(self, query: str, k: int = DEFAULT_SEARCH_RESULT_LIMIT) -> list[SearchResult]:
        """Search for tools using the configured embedder."""
        # Ensure index is ready for BM25
        await self.ensure_index_ready()

        if isinstance(self.embedder, BM25Embedder):
            # For BM25, use direct search method
            return await self._search_with_bm25(query, k)
        else:
            # For vector embedders, use vector similarity
            return await self._search_with_vectors(query, k)

    async def _search_with_vectors(self, query: str, k: int) -> list[SearchResult]:
        """Search using vector similarity."""
        query_vector = await self.embedder.embed_text(query)
        return await self.persistence.search_similar_tools(query_vector, k)

    async def _search_with_bm25(self, query: str, k: int) -> list[SearchResult]:
        """Search using BM25 similarity."""
        # Get all tools and their texts
        all_tools = await self.persistence.get_all_tools()
        if not all_tools:
            return []

        # Use the BM25 embedder's search method with the indexed corpus
        results = await self.embedder.search_similar(query, None, k)

        if not results:
            return []

        # Extract scores for normalization
        scores = [score for _, score in results]

        # Use modified sigmoid normalization to map scores to [0,1] range
        # Handle zero scores (nonsense queries) specially
        normalized_scores = []
        for score in scores:
            if score <= 0.0:
                # Map zero/negative scores to very low values (< 0.5)
                normalized_score = MIN_NORMALIZED_SCORE_THRESHOLD
            else:
                # Apply sigmoid function: 1 / (1 + exp(-score))
                # This maps positive scores to (0.5, 1) range
                normalized_score = 1.0 / (1.0 + math.exp(-score))
            normalized_scores.append(normalized_score)

        # Convert to SearchResult objects with normalized scores
        search_results = []
        for i, (idx, _) in enumerate(results):
            if idx < len(all_tools):
                search_results.append(
                    SearchResult(tool=all_tools[idx], score=normalized_scores[i])
                )

        return search_results

    async def get_index_stats(self) -> dict[str, Any]:
        """Get statistics about the current index."""
        stats = {
            "embedder_type": self.embedder_type.value,
            "needs_reindex": self._needs_reindex,
        }

        if isinstance(self.embedder, BM25Embedder):
            self._update_bm25_stats(stats)
        else:
            self._update_vector_embedder_stats(stats)

        return stats

    def _update_bm25_stats(self, stats: dict[str, Any]) -> None:
        stats.update(
            {
                "corpus_size": len(self.embedder.corpus),
                "indexed": self.embedder.indexed,
                "index_dir": self.embedder.index_dir,
            }
        )

    def _update_vector_embedder_stats(self, stats: dict[str, Any]) -> None:
        stats.update(
            {
                "dimension": self.embedder.get_dimension(),
            }
        )

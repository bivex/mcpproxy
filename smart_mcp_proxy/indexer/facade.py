"""Indexer and search facade."""

import asyncio
import os
from typing import Any
import numpy as np

from .base import BaseEmbedder
from .bm25 import BM25Embedder
from .huggingface import HuggingFaceEmbedder
from .openai import OpenAIEmbedder
from ..models.schemas import EmbedderType, ToolMetadata, SearchResult
from ..persistence.facade import PersistenceFacade
from ..utils.hashing import compute_tool_hash


class IndexerFacade:
    """Facade for indexing and searching tools."""
    
    def __init__(self, persistence: PersistenceFacade, embedder_type: EmbedderType = EmbedderType.BM25, 
                 hf_model: str | None = None):
        self.persistence = persistence
        self.embedder = self._create_embedder(embedder_type, hf_model)
    
    def _create_embedder(self, embedder_type: EmbedderType, hf_model: str | None) -> BaseEmbedder:
        """Create embedder based on type."""
        if embedder_type == EmbedderType.BM25:
            return BM25Embedder()
        elif embedder_type == EmbedderType.HF:
            model_name = hf_model or "sentence-transformers/all-MiniLM-L6-v2"
            return HuggingFaceEmbedder(model_name)
        elif embedder_type == EmbedderType.OPENAI:
            return OpenAIEmbedder()
        else:
            raise ValueError(f"Unknown embedder type: {embedder_type}")
    
    async def index_tool(self, name: str, description: str, server_name: str, 
                        params: dict[str, Any] | None = None,
                        tags: list[str] | None = None,
                        annotations: Any | None = None) -> None:
        """Index a tool with its metadata using Tool class fields."""
        # Include tags and annotations in hash computation
        extended_params = {
            "parameters": params or {},
            "tags": tags or [],
            "annotations": annotations
        }
        tool_hash = compute_tool_hash(name, description, extended_params)
        
        # Check if tool already exists
        existing_tool = await self.persistence.get_tool_by_hash(tool_hash)
        if existing_tool:
            return  # Tool unchanged, no need to re-index
        
        # Create enhanced text for embedding (include tags)
        enhanced_text = self.embedder.combine_tool_text(name, description, params)
        if tags:
            enhanced_text += f" | Tags: {', '.join(tags)}"
        
        vector = await self.embedder.embed_text(enhanced_text)
        
        # Store in persistence layer with extended metadata
        tool = ToolMetadata(
            name=name,
            description=description,
            hash=tool_hash,
            server_name=server_name,
            params_json=str(extended_params) if extended_params else None
        )
        
        await self.persistence.store_tool_with_vector(tool, vector)
    
    async def search_tools(self, query: str, k: int = 5) -> list[SearchResult]:
        """Search for tools using the configured embedder."""
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
        
        # Create text representations
        texts = []
        for tool in all_tools:
            params = eval(tool.params_json) if tool.params_json else None
            text = self.embedder.combine_tool_text(tool.name, tool.description, params)
            texts.append(text)
        
        # Search with BM25
        results = await self.embedder.search_similar(query, texts, k)
        
        # Convert to SearchResult objects
        search_results = []
        for idx, score in results:
            if idx < len(all_tools):
                search_results.append(SearchResult(
                    tool=all_tools[idx],
                    score=score
                ))
        
        return search_results 
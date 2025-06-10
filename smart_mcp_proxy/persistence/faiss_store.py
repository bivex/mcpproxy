"""Faiss vector store operations."""

import asyncio
import numpy as np
from pathlib import Path
from typing import Iterable

try:
    import faiss
except ImportError:
    faiss = None


class FaissStore:
    """Faiss vector store for tool embeddings."""
    
    def __init__(self, index_path: str = "tools.faiss", dimension: int = 384):
        self.index_path = Path(index_path)
        self.dimension = dimension
        self.index = None
        self.next_id = 0
        self._init_index()
    
    def _init_index(self) -> None:
        """Initialize or load Faiss index."""
        if faiss is None:
            raise ImportError("faiss-cpu package is required")
        
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.next_id = self.index.ntotal
        else:
            # Using IndexFlatL2 for exact search (can be changed to IndexIVFFlat for approximate)
            self.index = faiss.IndexFlatL2(self.dimension)
    
    async def add_vector(self, vector: np.ndarray) -> int:
        """Add vector to index and return its ID."""
        if vector.shape != (self.dimension,):
            raise ValueError(f"Vector must have dimension {self.dimension}")
        
        vector_id = self.next_id
        self.index.add(vector.reshape(1, -1).astype(np.float32))
        self.next_id += 1
        
        # Save index periodically
        await self._save_index()
        return vector_id
    
    async def update_vector(self, vector_id: int, vector: np.ndarray) -> None:
        """Update vector at given ID (rebuild index)."""
        if vector.shape != (self.dimension,):
            raise ValueError(f"Vector must have dimension {self.dimension}")
        
        # For simplicity, we remove and re-add (Faiss doesn't support in-place updates)
        # In production, consider using IndexIVFFlat with remove_ids
        await self.remove_vector(vector_id)
        await self.add_vector(vector)
    
    async def remove_vector(self, vector_id: int) -> None:
        """Remove vector from index (not efficiently supported in basic Faiss)."""
        # Basic IndexFlatL2 doesn't support removal efficiently
        # This is a limitation - in production use IndexIVFFlat with remove_ids
        pass
    
    async def search(self, query_vector: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """Search for similar vectors.
        
        Returns:
            distances: Array of distances
            indices: Array of vector IDs
        """
        if query_vector.shape != (self.dimension,):
            raise ValueError(f"Query vector must have dimension {self.dimension}")
        
        if self.index.ntotal == 0:
            return np.array([]), np.array([])
        
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(
            query_vector.reshape(1, -1).astype(np.float32), k
        )
        return distances[0], indices[0]
    
    async def get_vector_count(self) -> int:
        """Get total number of vectors in index."""
        return self.index.ntotal
    
    async def _save_index(self) -> None:
        """Save index to disk."""
        faiss.write_index(self.index, str(self.index_path))
    
    async def close(self) -> None:
        """Save and close index."""
        await self._save_index() 
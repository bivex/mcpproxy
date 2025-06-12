"""HuggingFace transformer embedder implementation."""

import asyncio
from typing import Any
import numpy as np

from .base import BaseEmbedder


class HuggingFaceEmbedder(BaseEmbedder):
    """HuggingFace sentence transformer embedder."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print(f"\n❌ ERROR: HuggingFace embeddings requires sentence-transformers but it's not installed.")
            print(f"   To use HuggingFace embeddings, install with:")
            print(f"   pip install mcpproxy[huggingface]")
            print(f"   or pip install sentence-transformers")
            print()
            import sys
            sys.exit(1)
        
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    async def embed_text(self, text: str) -> np.ndarray:
        """Embed single text using sentence transformer."""
        # Run in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, self.model.encode, text
        )
        return embedding.astype(np.float32)
    
    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed batch of texts."""
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, self.model.encode, texts
        )
        return [emb.astype(np.float32) for emb in embeddings]
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension 
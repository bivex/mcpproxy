"""HuggingFace transformer embedder implementation."""

import asyncio
import sys

import numpy as np

from .base import BaseEmbedder


class HuggingFaceEmbedder(BaseEmbedder):
    """HuggingFace sentence transformer embedder."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print(
                "\n❌ ERROR: HuggingFace embeddings requires sentence-transformers but it's not installed.",
                file=sys.stderr
            )
            print("   To use HuggingFace embeddings, install with:", file=sys.stderr)
            print("   pip install mcpproxy[huggingface]", file=sys.stderr)
            print("   or pip install sentence-transformers", file=sys.stderr)
            print(file=sys.stderr)
            sys.exit(1)

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    async def embed_text(self, text: str) -> np.ndarray:
        """Embed single text using sentence transformer."""
        # Run in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, self.model.encode, text)
        return embedding.astype(np.float32)

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed batch of texts."""
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, self.model.encode, texts)
        return [emb.astype(np.float32) for emb in embeddings]

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension

"""OpenAI embedder implementation."""

import os
import sys

import numpy as np

from .base import BaseEmbedder
from mcpproxy.models.constants import OPENAI_EMBEDDING_DIMENSION


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embedder using text-embedding-ada-002."""

    def __init__(
        self, model: str = "text-embedding-ada-002", api_key: str | None = None
    ):
        try:
            import openai
        except ImportError:
            print(
                "\n❌ ERROR: OpenAI embeddings requires openai and tiktoken"
                " but they\'re not installed.",
                file=sys.stderr
            )
            print("   To use OpenAI embeddings, install with:", file=sys.stderr)
            print("   pip install mcpproxy[openai]", file=sys.stderr)
            print("   or pip install openai tiktoken", file=sys.stderr)
            print(file=sys.stderr)
            sys.exit(1)

        self.model = model
        self.client = openai.AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        # text-embedding-ada-002 has 1536 dimensions
        self.dimension = OPENAI_EMBEDDING_DIMENSION if model == "text-embedding-ada-002" else OPENAI_EMBEDDING_DIMENSION

    async def embed_text(self, text: str) -> np.ndarray:
        """Embed single text using OpenAI API."""
        response = await self.client.embeddings.create(model=self.model, input=text)
        embedding = response.data[0].embedding
        return np.array(embedding, dtype=np.float32)

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed batch of texts."""
        response = await self.client.embeddings.create(model=self.model, input=texts)
        embeddings = []
        for item in response.data:
            embeddings.append(np.array(item.embedding, dtype=np.float32))
        return embeddings

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension

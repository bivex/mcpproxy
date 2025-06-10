"""BM25 lexical search embedder implementation."""

import asyncio
from typing import Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseEmbedder


class BM25Embedder(BaseEmbedder):
    """BM25-based embedder using TF-IDF vectors."""
    
    def __init__(self, max_features: int = 10000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        self.fitted = False
        self.corpus = []
        self.dimension = max_features
    
    async def embed_text(self, text: str) -> np.ndarray:
        """Embed single text using TF-IDF."""
        if not self.fitted:
            # If not fitted, fit on just this text (not ideal but fallback)
            await self.fit_corpus([text])
        
        vector = self.vectorizer.transform([text])
        return vector.toarray()[0].astype(np.float32)
    
    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed batch of texts."""
        if not self.fitted:
            await self.fit_corpus(texts)
        
        vectors = self.vectorizer.transform(texts)
        return [vector.toarray()[0].astype(np.float32) for vector in vectors]
    
    async def fit_corpus(self, texts: list[str]) -> None:
        """Fit the vectorizer on a corpus of texts."""
        self.corpus = texts
        self.vectorizer.fit(texts)
        self.fitted = True
        # Update dimension to actual fitted size
        self.dimension = len(self.vectorizer.get_feature_names_out())
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension
    
    async def search_similar(self, query: str, candidate_texts: list[str], k: int = 5) -> list[tuple[int, float]]:
        """Search for similar texts using BM25/TF-IDF similarity.
        
        Args:
            query: Query text
            candidate_texts: Texts to search in
            k: Number of top results
            
        Returns:
            List of (index, score) tuples
        """
        if not candidate_texts:
            return []
        
        # Ensure we're fitted on the candidate texts
        all_texts = [query] + candidate_texts
        await self.fit_corpus(all_texts)
        
        # Transform query and candidates
        query_vector = self.vectorizer.transform([query])
        candidate_vectors = self.vectorizer.transform(candidate_texts)
        
        # Compute similarities
        similarities = cosine_similarity(query_vector, candidate_vectors)[0]
        
        # Get top k results
        top_indices = np.argsort(similarities)[::-1][:k]
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]
        
        return results 
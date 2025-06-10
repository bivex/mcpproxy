"""BM25 lexical search embedder implementation using bm25s library."""

import asyncio
import os
import tempfile
from typing import Any
import numpy as np
import bm25s

from .base import BaseEmbedder


class BM25Embedder(BaseEmbedder):
    """BM25-based embedder using bm25s library."""
    
    def __init__(self, index_dir: str | None = None):
        """Initialize BM25 embedder.
        
        Args:
            index_dir: Directory to save/load BM25 index. If None, uses temp directory.
        """
        self.index_dir = index_dir or tempfile.mkdtemp(prefix="bm25s_")
        self.retriever: bm25s.BM25 | None = None
        self.corpus: list[str] = []
        self.indexed = False
        
        # Ensure index directory exists
        os.makedirs(self.index_dir, exist_ok=True)
    
    async def embed_text(self, text: str) -> np.ndarray:
        """Embed single text using BM25.
        
        Note: For BM25, we don't create traditional embeddings.
        This method stores the text for later batch indexing.
        """
        if text not in self.corpus:
            self.corpus.append(text)
            self.indexed = False  # Mark as needing reindexing
        
        # Return a placeholder vector - BM25 doesn't use traditional embeddings
        return np.array([0.0], dtype=np.float32)
    
    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed batch of texts."""
        results = []
        for text in texts:
            await self.embed_text(text)
            results.append(np.array([0.0], dtype=np.float32))
        return results
    
    async def fit_corpus(self, texts: list[str]) -> None:
        """Fit the BM25 model on a corpus of texts."""
        self.corpus = texts
        await self._build_index()
    
    async def _build_index(self) -> None:
        """Build the BM25 index from the current corpus."""
        if not self.corpus:
            return
        
        # Tokenize the corpus
        corpus_tokens = bm25s.tokenize(self.corpus, stopwords="en")
        
        # Create and index the BM25 model
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens)
        
        # Save the index
        index_path = os.path.join(self.index_dir, "bm25s_index")
        self.retriever.save(index_path, corpus=self.corpus)
        
        self.indexed = True
    
    async def reindex(self) -> None:
        """Rebuild the entire BM25 index with current corpus."""
        await self._build_index()
    
    def get_dimension(self) -> int:
        """Get embedding dimension (not applicable for BM25)."""
        return 1  # Placeholder dimension
    
    def load_index(self) -> bool:
        """Load existing BM25 index from disk.
        
        Returns:
            True if index was loaded successfully, False otherwise.
        """
        index_path = os.path.join(self.index_dir, "bm25s_index")
        if os.path.exists(index_path):
            try:
                self.retriever = bm25s.BM25.load(index_path, load_corpus=True)
                # Extract corpus from loaded retriever if available
                if hasattr(self.retriever, 'corpus') and self.retriever.corpus is not None:
                    self.corpus = list(self.retriever.corpus)
                self.indexed = True
                return True
            except Exception:
                return False
        return False
    
    async def search_similar(self, query: str, candidate_texts: list[str] | None = None, k: int = 5) -> list[tuple[int, float]]:
        """Search for similar texts using BM25.
        
        Args:
            query: Query text
            candidate_texts: If provided, search within these texts. Otherwise use indexed corpus.
            k: Number of top results
            
        Returns:
            List of (index, score) tuples
        """
        # If candidate_texts provided, use them as search corpus
        if candidate_texts is not None:
            await self.fit_corpus(candidate_texts)
            search_corpus = candidate_texts
        else:
            # Ensure we have an index
            if not self.indexed and self.corpus:
                await self._build_index()
            search_corpus = self.corpus
        
        if not search_corpus or not self.retriever:
            return []
        
        # Tokenize query
        query_tokens = bm25s.tokenize([query], stopwords="en")
        
        # Retrieve results
        try:
            # Get document indices and scores
            results, scores = self.retriever.retrieve(query_tokens, k=min(k, len(search_corpus)))
            
            # Convert to list of (index, score) tuples
            if results.size > 0 and scores.size > 0:
                result_list = []
                for i in range(results.shape[1]):
                    doc_idx = int(results[0, i])
                    score = float(scores[0, i])
                    if doc_idx < len(search_corpus):  # Valid index
                        result_list.append((doc_idx, score))
                return result_list
        except Exception:
            pass
        
        return []
    
 
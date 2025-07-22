"""BM25 lexical search embedder implementation using bm25s library."""

import os
import tempfile

import bm25s
import numpy as np

from .base import BaseEmbedder

BM25_TOP_K = 5  # Default number of top results for BM25 search


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

        # Try to load existing index on initialization
        self._try_load_existing_index()

    def _try_load_existing_index(self) -> None:
        """Try to load existing index without throwing exceptions."""
        try:
            if self.load_index():
                # Successfully loaded existing index
                pass
        except Exception:
            # If loading fails, start fresh
            self.reset_index()

    def reset_index(self) -> None:
        """Reset the index to initial state."""
        self.retriever = None
        self.corpus = []
        self.indexed = False

    async def reset_stored_data(self) -> None:
        """Reset stored index data by removing files from disk."""
        import shutil

        self.reset_index()

        # Remove the entire index directory if it exists
        if os.path.exists(self.index_dir):
            try:
                # Remove all BM25 index files
                index_path = os.path.join(self.index_dir, "bm25s_index")
                if os.path.exists(index_path):
                    # bm25s saves multiple files, so remove the directory
                    if os.path.isdir(index_path):
                        shutil.rmtree(index_path)
                    else:
                        os.remove(index_path)

                # Also clean up any other index files that might exist
                for file in os.listdir(self.index_dir):
                    if file.startswith("bm25"):
                        file_path = os.path.join(self.index_dir, file)
                        if os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                        else:
                            os.remove(file_path)
            except Exception:
                # If cleanup fails, continue - the reset_index() call above is most important
                pass

    def get_corpus_size(self) -> int:
        """Get the current corpus size."""
        return len(self.corpus)

    def is_indexed(self) -> bool:
        """Check if the corpus is indexed."""
        return self.indexed and self.retriever is not None

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
        if not texts:
            self.reset_index()
            return

        self.corpus = texts
        await self._build_index()

    async def _build_index(self) -> None:
        """Build the BM25 index from the current corpus."""
        if not self.corpus:
            self.reset_index()
            return

        try:
            # Tokenize the corpus
            corpus_tokens = bm25s.tokenize(self.corpus, stopwords="en")

            # Create and index the BM25 model
            self.retriever = bm25s.BM25()
            self.retriever.index(corpus_tokens)

            # Save the index
            index_path = os.path.join(self.index_dir, "bm25s_index")
            self.retriever.save(index_path, corpus=self.corpus)

            self.indexed = True
        except Exception as e:
            # If indexing fails, reset to clean state
            self.reset_index()
            raise e

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
        if not os.path.exists(index_path):
            return False

        try:
            self.retriever = bm25s.BM25.load(index_path, load_corpus=True)

            # Extract corpus from loaded retriever if available
            if hasattr(self.retriever, "corpus") and self.retriever.corpus is not None:
                # Handle different corpus formats
                loaded_corpus = self.retriever.corpus
                if loaded_corpus:
                    # Check if corpus items are dictionaries or strings
                    if isinstance(loaded_corpus[0], dict):
                        # Extract text from dictionary format
                        self.corpus = [item["text"] for item in loaded_corpus]
                    else:
                        # Already in string format
                        self.corpus = list(loaded_corpus)
                    self.indexed = True
                    return True

            # If corpus is not available in retriever, reset
            self.reset_index()
            return False

        except Exception:
            self.reset_index()
            return False

    async def search_similar(
        self, query: str, candidate_texts: list[str] | None = None, k: int = BM25_TOP_K
    ) -> list[tuple[int, float]]:
        """Search for similar texts using BM25.

        Args:
            query: Query text
            candidate_texts: If provided, search within these texts. Otherwise use indexed corpus.
            k: Number of top results

        Returns:
            List of (index, score) tuples
        """
        if not query.strip():
            return []  # Empty query

        try:
            search_corpus, retriever = await self._get_search_context(candidate_texts)
            if not search_corpus or not retriever:
                return []

            results, scores = self._tokenize_and_retrieve(query, retriever, search_corpus, k)
            return self._format_search_results(results, scores, search_corpus)
        except Exception:
            # If search fails, return empty results
            return []

    async def _get_search_context(self, candidate_texts: list[str] | None):
        if candidate_texts is not None:
            if not candidate_texts:
                return None, None
            temp_embedder = BM25Embedder()
            await temp_embedder.fit_corpus(candidate_texts)
            return candidate_texts, temp_embedder.retriever
        else:
            if not self.corpus:
                return None, None
            if not self.is_indexed():
                await self._build_index()
            return self.corpus, self.retriever

    def _tokenize_and_retrieve(self, query: str, retriever, search_corpus: list[str], k: int):
        query_tokens = bm25s.tokenize([query], stopwords="en")
        results, scores = retriever.retrieve(query_tokens, k=min(k, len(search_corpus)))
        return results, scores

    def _format_search_results(self, results, scores, search_corpus: list[str]):
        result_list = []
        if results.size > 0 and scores.size > 0:
            if len(results) > 0 and isinstance(results[0], list):
                # results is a list of lists (new format)
                results_batch = results[0] if len(results) > 0 else []
                scores_batch = scores[0] if len(scores) > 0 else []

                for i, (result_item, score) in enumerate(zip(results_batch, scores_batch)):
                    doc_idx = result_item.get("id", i) if isinstance(result_item, dict) else int(result_item)
                    if 0 <= doc_idx < len(search_corpus):
                        result_list.append((doc_idx, float(score)))
            else:
                # Old format - results is ndarray with shape
                for i in range(results.shape[1]):
                    doc_idx = int(results[0, i])
                    if 0 <= doc_idx < len(search_corpus):
                        result_list.append((doc_idx, float(scores[0, i])))
        return result_list

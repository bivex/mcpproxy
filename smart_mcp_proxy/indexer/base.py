"""Base embedder interface for different embedding backends."""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class BaseEmbedder(ABC):
    """Abstract base class for embedders."""
    
    @abstractmethod
    async def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a batch of text strings.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of embeddings produced by this embedder."""
        pass
    
    def combine_tool_text(self, name: str, description: str, params: dict[str, Any] | None = None) -> str:
        """Combine tool metadata into a single text for embedding.
        
        Args:
            name: Tool name
            description: Tool description
            params: Tool parameters schema
            
        Returns:
            Combined text representation
        """
        text_parts = [f"Tool: {name}", f"Description: {description}"]
        
        if params:
            # Flatten parameters for text representation
            param_texts = []
            for param_name, param_info in params.get("properties", {}).items():
                param_type = param_info.get("type", "unknown")
                param_desc = param_info.get("description", "")
                param_texts.append(f"{param_name} ({param_type}): {param_desc}")
            
            if param_texts:
                text_parts.append(f"Parameters: {'; '.join(param_texts)}")
        
        return " | ".join(text_parts) 
"""Data models and schemas for Smart MCP Proxy."""

import json
from enum import Enum
from typing import Any, Optional
from dataclasses import dataclass

from pydantic import BaseModel


class EmbedderType(str, Enum):
    """Available embedder types."""

    BM25 = "BM25"
    HF = "HF"
    OPENAI = "OPENAI"


class ServerConfig(BaseModel):
    """Configuration for an MCP server."""

    url: str | None = None
    command: str | None = None
    args: list[str] | None = None
    env: dict[str, str] | None = None
    oauth: bool = False


class ProxyConfig(BaseModel):
    """Main proxy configuration."""

    mcp_servers: dict[str, ServerConfig]
    embedder: EmbedderType = EmbedderType.BM25
    hf_model: str | None = None
    top_k: int = 5
    tool_name_limit: int = 60
    tools_limit: int = 15


class ToolMetadata(BaseModel):
    """Tool metadata for indexing."""

    id: int | None = None
    name: str
    description: str
    hash: str
    server_name: str
    faiss_vector_id: int | None = None
    params_json: str | None = None


class SearchResult(BaseModel):
    """Search result with score."""

    tool: ToolMetadata
    score: float


class ToolRegistration(BaseModel):
    """Tool registration data."""

    name: str
    description: str
    input_schema: dict[str, Any]
    server_name: str


@dataclass
class ToolData:
    name: str
    description: str
    server_name: str
    params: Optional[dict[str, Any]] = None
    tags: Optional[list[str]] = None
    annotations: Any = None
    last_used_at: Optional[Any] = None
    embedding: Optional[list[float]] = None
    id: Optional[int] = None

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "server_name": self.server_name,
            "params": self.params,
            "tags": self.tags,
            "annotations": self.annotations
        }

@dataclass
class ToolPoolManagerDependencies:
    indexer: Any
    persistence: Any
    config: Any
    proxy_servers: Any

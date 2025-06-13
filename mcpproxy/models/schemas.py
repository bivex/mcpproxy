"""Data models and schemas for Smart MCP Proxy."""

from enum import Enum
from typing import Any

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

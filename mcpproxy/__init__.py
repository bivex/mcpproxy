"""Smart MCP Proxy - A federating gateway for MCP servers with intelligent tool discovery."""

from .indexer.facade import IndexerFacade
from .models.schemas import EmbedderType, ProxyConfig
from .persistence.facade import PersistenceFacade
from .server.config import ConfigLoader
from .server.mcp_server import SmartMCPProxyServer

__version__ = "0.2.1"
__all__ = [
    "SmartMCPProxyServer",
    "ConfigLoader",
    "PersistenceFacade",
    "IndexerFacade",
    "ProxyConfig",
    "EmbedderType",
]

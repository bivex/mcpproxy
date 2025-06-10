"""Smart MCP Proxy - A federating gateway for MCP servers with intelligent tool discovery."""

from .server.mcp_server import SmartMCPProxyServer
from .server.config import ConfigLoader
from .persistence.facade import PersistenceFacade
from .indexer.facade import IndexerFacade
from .models.schemas import ProxyConfig, EmbedderType

__version__ = "0.1.0"
__all__ = [
    "SmartMCPProxyServer",
    "ConfigLoader", 
    "PersistenceFacade",
    "IndexerFacade",
    "ProxyConfig",
    "EmbedderType",
]

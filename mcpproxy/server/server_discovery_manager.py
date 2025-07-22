import asyncio
import os
from typing import Any

from fastmcp import FastMCP # type: ignore[import-untyped]
from fastmcp.client import Client # type: ignore[import-untyped]
from fastmcp.tools.tool import Tool # type: ignore[import-not-found]

from ..indexer.facade import IndexerFacade
from ..logging import get_logger
from ..models.schemas import EmbedderType, ToolMetadata
from ..persistence.facade import PersistenceFacade
from ..utils.name_sanitizer import sanitize_tool_name
from .config.config import ConfigLoader, ProxyConfig # Import ConfigLoader and ProxyConfig
from .tool_pool_manager import ToolPoolManager # Import ToolPoolManager

logger = get_logger()


class ServerDiscoveryManager:
    """Manages the discovery, indexing, and cleanup of tools from upstream MCP servers."""

    def __init__(
        self,
        config: ProxyConfig, # Accept ProxyConfig directly
        indexer: IndexerFacade,
        persistence: PersistenceFacade,
        tool_pool_manager: ToolPoolManager,
    ):
        self.config = config # Store the entire config object
        self.indexer = indexer
        self.persistence = persistence
        self.tool_pool_manager = tool_pool_manager

        self.upstream_clients: dict[str, Client] = {}
        self.proxy_servers: dict[str, FastMCP] = {}

    async def create_upstream_clients_and_proxies(self) -> None:
        """Create upstream clients and proxy servers for all configured servers."""
        logger.info(
            f"Creating upstream clients for {len(self.config.mcp_servers)} servers..." # Use self.config.mcp_servers
        )

        for server_name, server_config in self.config.mcp_servers.items(): # Use self.config.mcp_servers
            try:
                logger.debug(
                    f"Creating client for {server_name}: url={server_config.url}, command={getattr(server_config, 'command', None)}"
                )

                if server_config.url:
                    logger.debug(
                        f"Creating URL-based client for {server_name} at {server_config.url}"
                    )
                    try:
                        client = Client(server_config.url)
                        logger.debug(f"URL client created for {server_name}")
                    except Exception as client_error:
                        logger.error(f"Failed to create URL client for {server_name}: {type(client_error).__name__}: {client_error}")
                        logger.error(f"URL client error details for {server_name}:", exc_info=True)
                        continue
                    
                    try:
                        proxy_server = FastMCP.as_proxy(
                            client, name=f"{server_name}_proxy"
                        )
                        logger.debug(f"URL proxy server created for {server_name}")
                    except Exception as proxy_error:
                        logger.error(f"Failed to create URL proxy server for {server_name}: {type(proxy_error).__name__}: {proxy_error}")
                        logger.error(f"URL proxy error details for {server_name}:", exc_info=True)
                        continue
                        
                elif server_config.command:
                    env_value = getattr(server_config, "env", {})
                    if env_value is None:
                        env_value = {}
                    
                    config_dict = {
                        "mcpServers": {
                            server_name: {
                                "command": server_config.command,
                                "args": getattr(server_config, "args", []),
                                "env": env_value,
                            }
                        }
                    }
                    logger.debug(
                        f"Creating command-based client for {server_name}: {config_dict}"
                    )
                    try:
                        client = Client(config_dict)
                        logger.debug(f"Command client created for {server_name}")
                    except Exception as client_error:
                        logger.error(f"Failed to create command client for {server_name}: {type(client_error).__name__}: {client_error}")
                        logger.error(f"Command client error details for {server_name}:", exc_info=True)
                        continue
                    
                    try:
                        proxy_server = FastMCP.as_proxy(
                            client, name=f"{server_name}_proxy"
                        )
                        logger.debug(f"Command proxy server created for {server_name}")
                    except Exception as proxy_error:
                        logger.error(f"Failed to create command proxy server for {server_name}: {type(proxy_error).__name__}: {proxy_error}")
                        logger.error(f"Command proxy error details for {server_name}:", exc_info=True)
                        continue
                else:
                    logger.warning(
                        f"Skipping {server_name}: no URL or command specified"
                    )
                    continue

                self.upstream_clients[server_name] = client
                self.proxy_servers[server_name] = proxy_server
                logger.info(
                    f"Created upstream client and proxy server for {server_name}"
                )

                logger.debug(
                    f"Proxy server for {server_name}: type={type(proxy_server)}, has_get_tools={hasattr(proxy_server, 'get_tools')}"
                )

            except Exception as e:
                logger.error(
                    f"Error creating upstream client/proxy for {server_name}: {type(e).__name__}: {e}"
                )
                logger.error(
                    f"Full exception details for {server_name}:", exc_info=True
                )

    async def discover_and_index_tools(self) -> None:
        """Discover tools from all configured servers and index them."""
        logger.info(f"Starting tool discovery for {len(self.proxy_servers)} servers...")

        try:
            current_servers = set(self.proxy_servers.keys())

            logger.debug("Cleaning up stale servers...")
            await self._cleanup_stale_servers(current_servers)
        except Exception as e:
            logger.error(f"Error during stale server cleanup: {type(e).__name__}: {e}")
            logger.error("Full stale server cleanup error details:", exc_info=True)
            raise

        current_tools = {}  # server_name -> set of tool names
        for server_name, proxy_server in self.proxy_servers.items():
            try:
                logger.debug(f"Starting tool discovery for server: {server_name}")
                tools = await self._discover_server_tools(server_name, proxy_server)
                current_tools[server_name] = set(tools.keys()) if tools else set()
                logger.debug(f"Completed tool discovery for server: {server_name}")
            except Exception as e:
                logger.error(
                    f"Error discovering tools from {server_name}: {type(e).__name__}: {e}"
                )
                logger.error(f"Exception details for {server_name}:", exc_info=True)
                current_tools[server_name] = set()  # Mark as having no tools

        try:
            logger.debug("Cleaning up stale tools...")
            await self._cleanup_stale_tools(current_tools)
        except Exception as e:
            logger.error(f"Error during stale tool cleanup: {type(e).__name__}: {e}")
            logger.error("Full stale tool cleanup error details:", exc_info=True)
            raise

    async def _cleanup_stale_servers(self, current_servers: set[str]) -> None:
        """Remove tools from servers that no longer exist in configuration."""
        if not self.persistence:
            return

        all_tools = await self.persistence.get_all_tools()

        db_servers = {tool.server_name for tool in all_tools}
        stale_servers = db_servers - current_servers

        if stale_servers:
            logger.info(
                f"Removing tools from {len(stale_servers)} stale servers: {stale_servers}"
            )
            for server_name in stale_servers:
                await self.persistence.delete_tools_by_server(server_name)
                logger.debug(f"Removed all tools from stale server: {server_name}")

    async def _cleanup_stale_tools(self, current_tools: dict[str, set[str]]) -> None:
        """Remove tools that no longer exist on their servers."""
        if not self.persistence:
            return

        removed_count = 0
        for server_name, tool_names in current_tools.items():
            db_tools = await self.persistence.get_tools_by_server(server_name)

            db_tool_names = {tool.name for tool in db_tools}
            stale_tool_names = db_tool_names - tool_names

            if stale_tool_names:
                logger.debug(
                    f"Server {server_name}: removing {len(stale_tool_names)} stale tools: {stale_tool_names}"
                )

                for tool in db_tools:
                    if tool.name in stale_tool_names:
                        await self._remove_tool_from_persistence(tool)
                        removed_count += 1

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} stale tools from database")

    async def _remove_tool_from_persistence(self, tool: ToolMetadata) -> None:
        """Remove a specific tool from persistence layer."""
        logger.debug(
            f"Would remove tool {tool.name} from {tool.server_name} (simplified cleanup)"
        )

    async def _discover_server_tools(
        self, server_name: str, proxy_server: FastMCP
    ) -> dict[str, str]:
        """Discover tools from a specific server using its proxy."""
        try:
            if not self.indexer:
                logger.warning(
                    f"Indexer not available, skipping discovery for {server_name}"
                )
                return {}

            logger.debug(f"Getting tools from proxy server for {server_name}...")

            if not hasattr(proxy_server, "get_tools"):
                logger.error(
                    f"Proxy server for {server_name} does not have get_tools method"
                )
                return {}

            try:
                logger.debug(f"Calling get_tools() on {server_name} proxy server...")
                tools = await proxy_server.get_tools()
                logger.debug(
                    f"get_tools() returned: {type(tools)} with {len(tools) if tools else 0} items"
                )
            except Exception as get_tools_error:
                logger.error(
                    f"Error calling get_tools on {server_name}: {type(get_tools_error).__name__}: {get_tools_error}"
                )
                raise

            if not tools:
                logger.warning(f"No tools returned from {server_name}")
                return {}

            indexed_count = 0
            tool_names = {}
            for tool_name, tool_obj in tools.items():
                try:
                    if self.tool_pool_manager: # Check if tool_pool_manager is initialized
                        sanitized_key = sanitize_tool_name(server_name, tool_name, self.config.tool_name_limit) # Use self.config.tool_name_limit
                        await self.tool_pool_manager.add_proxified_tool_to_memory(sanitized_key, tool_obj)

                    logger.debug(f"Indexing tool: {tool_name} from {server_name}")
                    await self.indexer.index_tool_from_object(tool_obj, server_name)
                    indexed_count += 1
                    tool_names[tool_name] = sanitized_key
                except Exception as tool_error:
                    error_msg = str(tool_error)
                    if (
                        "assert d == self.d" in error_msg
                        or "dimension" in error_msg.lower()
                    ):
                        logger.error(
                            f"Vector dimension mismatch for tool {tool_name}. This usually means:"
                        )
                        logger.error(
                            "1. Existing FAISS index has different dimension than current embedder"
                        )
                        logger.error(
                            "2. Solution: Set MCPPROXY_RESET_DATA=true environment variable to reset data"
                        )
                        logger.error("3. Or delete these data files manually:")
                        logger.error("   - tools.faiss (FAISS vector index)")
                        logger.error("   - proxy.db (SQLite database)")

                        if (
                            self.config.embedder == EmbedderType.BM25 # Use self.config.embedder
                            and self.indexer
                            and hasattr(self.indexer.embedder, "index_dir")
                        ):
                            bm25_dir = self.indexer.embedder.index_dir
                            logger.error(f"   - {bm25_dir}/ (BM25 index directory)")
                        else:
                            logger.error(
                                "   - BM25 index directory (if using BM25: usually a temp dir with bm25s_index/)"
                            )
                        logger.error(
                            "4. For BM25: the index dir is typically a temp directory starting with 'bm25s_'"
                        )
                    logger.error(
                        f"Error indexing tool {tool_name} from {server_name}: {type(tool_error).__name__}: {tool_error}"
                    )
                    logger.error(f"Full stack trace for {tool_name}:", exc_info=True)
                    logger.debug(f"Tool object details: {tool_obj}")

            logger.info(
                f"Successfully indexed {indexed_count}/{len(tools)} tools from {server_name}"
            )
            return tool_names

        except Exception as e:
            logger.error(
                f"Error discovering tools from {server_name}: {type(e).__name__}: {e}"
            )
            logger.error("Full exception details:", exc_info=True)
            return {} 

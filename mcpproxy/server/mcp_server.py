"""FastMCP server implementation for Smart MCP Proxy."""

import asyncio
import json
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import mcp.types as types  # type: ignore[import-untyped]
from fastmcp import FastMCP  # type: ignore[import-untyped]
from fastmcp.client import Client  # type: ignore[import-untyped]
from fastmcp.tools.tool import Tool  # type: ignore[import-not-found]
from mcp.server.lowlevel.server import NotificationOptions  # type: ignore[import-untyped]

from ..indexer.facade import IndexerFacade
from ..logging import configure_logging, get_logger
from ..models.schemas import (
    EmbedderType,
    ToolRegistration,
)
from ..persistence.facade import PersistenceFacade
from ..utils.name_sanitizer import sanitize_tool_name
from ..utils.tool_weight_calculator import calculate_tool_weight
from ..utils.output_truncator import truncate_output
from .config.config import ConfigLoader
from .tool_pool_manager import ToolPoolManager
from .server_discovery_manager import ServerDiscoveryManager

logger = get_logger()

# Removed patch for NotificationOptions.__init__
# original_init = NotificationOptions.__init__
# def patched_init(
#     self, prompts_changed=False, resources_changed=False, tools_changed=True
# ):
#     original_init(self, prompts_changed, resources_changed, tools_changed)
# NotificationOptions.__init__ = patched_init


class SmartMCPProxyServer:
    """Smart MCP Proxy server using FastMCP."""

    def __init__(self, config_path: str = "mcp_config.json"):
        # Check environment variable first, then use provided path, then default
        self.config_path = os.getenv("MCPPROXY_CONFIG_PATH", config_path)
        self.config_loader = ConfigLoader(self.config_path)
        self.config = self.config_loader.load_config()

        # Transport configuration from environment variables
        self.transport = os.getenv("MCPPROXY_TRANSPORT", "stdio")
        self.host = os.getenv("MCPPROXY_HOST", "127.0.0.1")
        self.port = int(os.getenv("MCPPROXY_PORT", "8000"))

        # Tool pool limit configuration
        self.tools_limit = int(os.getenv("MCPPROXY_TOOLS_LIMIT", "15"))

        # Output truncation configuration
        truncate_len = os.getenv("MCPPROXY_TRUNCATE_OUTPUT_LEN")
        self.truncate_output_len = int(truncate_len) if truncate_len else None

        # External command execution after tools list changes
        self.list_changed_exec_cmd = os.getenv("MCPPROXY_LIST_CHANGED_EXEC")

        # Routing type configuration
        self.routing_type = os.getenv("MCPPROXY_ROUTING_TYPE", "CALL_TOOL").upper()
        if self.routing_type not in ["DYNAMIC", "CALL_TOOL"]:
            raise ValueError(f"Invalid MCPPROXY_ROUTING_TYPE: {self.routing_type}. Must be 'DYNAMIC' or 'CALL_TOOL'")

        # Tool name limit configuration
        tool_name_limit_env = os.getenv("MCPPROXY_TOOL_NAME_LIMIT")
        if tool_name_limit_env is not None:
            self.tool_name_limit = int(tool_name_limit_env)
        elif hasattr(self.config, "tool_name_limit") and self.config.tool_name_limit:
            self.tool_name_limit = self.config.tool_name_limit
        else:
            self.tool_name_limit = 60  # Default value

        # Will be initialized in lifespan
        self.persistence: PersistenceFacade | None = None
        self.indexer: IndexerFacade | None = None
        self.tool_pool_manager: ToolPoolManager | None = None
        self.server_discovery_manager: ServerDiscoveryManager | None = None # Forward declaration

        # Track upstream clients and proxy servers
        self.upstream_clients: dict[str, Client] = {}
        self.proxy_servers: dict[str, FastMCP] = {}

        # Initialize FastMCP server with transport configuration
        if self.routing_type == "CALL_TOOL":
            instructions = """
            This server provides intelligent tool discovery and proxying for MCP servers.
            First, use 'retrieve_tools' to search and discover available tools from configured upstream servers.
            Then, use 'call_tool' with the tool name and arguments to execute the tool on the upstream server.
            Tools are not dynamically registered - use the call_tool interface instead.
            """
        else:  # DYNAMIC
            instructions = """
            This server provides intelligent tool discovery and proxying for MCP servers.
            Use 'retrieve_tools' to search and access tools from configured upstream servers.
            proxy tools are dynamically created and registered on the fly in accordance with the search results.
            Pass the original user query (if possible) to the 'retrieve_tools' tool to get the search results.
            """

        fastmcp_kwargs = {
            "name": "Smart MCP Proxy",
            "instructions": instructions,
            "lifespan": self._lifespan,
        }

        # Add host and port for non-stdio transports
        if self.transport != "stdio":
            fastmcp_kwargs["host"] = self.host
            fastmcp_kwargs["port"] = self.port

        self.mcp = FastMCP(**fastmcp_kwargs)

    def run(self) -> None:
        """Run the Smart MCP Proxy server with full initialization."""
        # Configure logging with debug level if requested
        log_level = os.getenv("MCPPROXY_LOG_LEVEL", "INFO")
        log_file = os.getenv("MCPPROXY_LOG_FILE")  # Optional file logging
        configure_logging(log_level, log_file)
        logger = get_logger()

        # Check for config file (already resolved in __init__)
        config_path = self.config_path

        if not Path(config_path).exists():
            logger.warning(f"Configuration file not found: {config_path}")
            logger.info("Creating sample configuration...")

            # Use the config_loader's file_handler to create sample config
            # config_loader = ConfigLoader() # No longer needed, use existing self.config_loader
            self.config_loader.file_handler.create_sample_config()

            logger.info(
                f"Please edit {config_path} and set required environment variables"
            )
            return

        try:
            logger.info(f"Starting Smart MCP Proxy on transport: {self.transport}")
            if self.transport != "stdio":
                logger.info(f"Server will be available at {self.host}:{self.port}")

            # Run the FastMCP app with configured transport
            if self.transport == "stdio":
                self.mcp.run()
            elif self.transport == "streamable-http":
                # For streamable-http transport, pass host and port
                self.mcp.run(
                    transport="streamable-http", host=self.host, port=self.port
                )
            elif self.transport == "sse":
                # For SSE transport (deprecated)
                self.mcp.run(transport="sse", host=self.host, port=self.port)
            else:
                # Fallback for any other transport
                self.mcp.run(transport=self.transport, host=self.host, port=self.port)

        except FileNotFoundError as e:
            logger.error(f"Configuration error: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error starting proxy: {e}")
            sys.exit(1)

    @asynccontextmanager
    async def _lifespan(self, app):
        """FastMCP lifespan context manager for resource management."""
        logger = get_logger()
        logger.info("Initializing Smart MCP Proxy resources...")

        try:
            await self._initialize_resources()
        except Exception as e:
            logger.error(f"Error during resource initialization: {type(e).__name__}: {e}")
            logger.error("Full initialization error details:", exc_info=True)
            raise  # Re-raise the exception

        try:
            yield  # Server is running
        finally:
            try:
                await self._cleanup_resources()
            except Exception as e:
                logger.error(f"Error during resource cleanup: {type(e).__name__}: {e}")
                logger.error("Full cleanup error details:", exc_info=True)

    async def _initialize_resources(self) -> None:
        """Core resource initialization logic."""
        logger = get_logger()

        try:
            # Check if we should reset data (useful when dimensions change)
            reset_data = os.getenv("MCPPROXY_RESET_DATA", "false").lower() == "true"

            # Determine vector dimension based on embedder type
            if self.config.embedder == EmbedderType.BM25:
                vector_dimension = 1  # BM25 uses placeholder vectors
            else:
                # For vector embedders, we'll set dimension after creating the embedder
                # Default to 384 for now, will be updated if needed
                vector_dimension = 384

            # Initialize persistence with appropriate dimension
            logger.debug(f"Initializing persistence with dimension: {vector_dimension}")
            self.persistence = PersistenceFacade(
                vector_dimension=vector_dimension, embedder_type=self.config.embedder
            )
            # Initialize PersistenceFacade asynchronously
            await self.persistence._ainit()

            # Reset data if requested
            if reset_data:
                logger.info("Resetting all data as requested...")
                await self.persistence.reset_all_data()

            # Initialize indexer
            logger.debug(f"Initializing indexer with embedder: {self.config.embedder}")
            self.indexer = IndexerFacade(
                self.persistence, self.config.embedder, self.config.hf_model
            )

            # Reset embedder data if requested (must be done after indexer creation)
            if reset_data:
                logger.info("Resetting embedder data...")
                await self.indexer.reset_embedder_data()

            # For non-BM25 embedders, update persistence with actual dimension
            if self.config.embedder != EmbedderType.BM25:
                actual_dimension = self.indexer.embedder.get_dimension()
                if actual_dimension != vector_dimension:
                    # Recreate persistence with correct dimension
                    logger.info(
                        f"Updating vector dimension from {vector_dimension} to {actual_dimension}"
                    )
                    await self.persistence.close()
                    self.persistence = PersistenceFacade(
                        vector_dimension=actual_dimension, embedder_type=self.config.embedder
                    )
                    # Update indexer to use new persistence
                    self.indexer.persistence = self.persistence

            # Initialize ToolPoolManager
            self.tool_pool_manager = ToolPoolManager(
                mcp_app=self.mcp,
                indexer=self.indexer,
                persistence=self.persistence,
                config=self.config, # Pass the config object directly
                proxy_servers=self.proxy_servers,
                truncate_output_fn=truncate_output,
                truncate_output_len=self.truncate_output_len,
            )
            # Now setup the tools managed by ToolPoolManager
            self._setup_tools() # Call _setup_tools here after tool_pool_manager is initialized

            # Initialize ServerDiscoveryManager after ToolPoolManager
            self.server_discovery_manager = ServerDiscoveryManager(
                config=self.config, # Pass the config object directly
                indexer=self.indexer,
                persistence=self.persistence,
                tool_pool_manager=self.tool_pool_manager,
            )

        except Exception as e:
            logger.error(f"Error during persistence/indexer initialization: {type(e).__name__}: {e}")
            logger.error("Full persistence/indexer error details:", exc_info=True)
            raise

        try:
            # Create upstream clients and proxy servers
            logger.debug("Creating upstream clients and proxy servers...")
            await self.server_discovery_manager.create_upstream_clients_and_proxies()
            self.upstream_clients = self.server_discovery_manager.upstream_clients # Update self's reference
            self.proxy_servers = self.server_discovery_manager.proxy_servers # Update self's reference
            logger.debug(
                f"Created {len(self.proxy_servers)} proxy servers and {len(self.upstream_clients)} upstream clients"
            )
        except Exception as e:
            logger.error(f"Error during upstream client/proxy creation: {type(e).__name__}: {e}")
            logger.error("Full upstream client/proxy error details:", exc_info=True)
            raise

        try:
            # Discover and index tools from upstream servers
            logger.debug("Discovering and indexing tools...")
            await self.server_discovery_manager.discover_and_index_tools()

        except Exception as e:
            logger.error(f"Error during tool discovery and indexing: {type(e).__name__}: {e}")
            logger.error("Full tool discovery error details:", exc_info=True)
            raise

        logger.info("Smart MCP Proxy resources initialized")

    async def _cleanup_resources(self) -> None:
        """Core resource cleanup logic."""
        logger = get_logger()
        logger.info("Shutting down Smart MCP Proxy resources...")

        # Close upstream clients
        for client in self.upstream_clients.values():
            try:
                if hasattr(client, "close"):
                    await client.close()
            except Exception as e:
                logger.warning(f"Error closing client: {e}")

        if self.persistence:
            await self.persistence.close()
        logger.info("Smart MCP Proxy resources cleaned up")

    def _setup_tools(self) -> None:
        """Setup core proxy tools."""
        if self.routing_type == "CALL_TOOL":
            @self.mcp.tool()
            async def call_tool(name: str, args: dict[str, Any]) -> str:
                if not self.tool_pool_manager:
                    return json.dumps({"error": "ToolPoolManager not initialized"})
                return await self.tool_pool_manager.call_tool(name, args)
        else: # DYNAMIC
            @self.mcp.tool()
            async def retrieve_tools(query: str) -> str:
                if not self.tool_pool_manager:
                    return json.dumps({"error": "ToolPoolManager not initialized"})

                # Get the result from ToolPoolManager first
                result_json_str = await self.tool_pool_manager.retrieve_tools(query)

                # Parse the result to check for newly_registered or evicted_tools
                result_data = json.loads(result_json_str)

                # Notify connected clients that the available tools list has changed so they can refresh
                try:
                    if result_data.get("newly_registered") or result_data.get("evicted_tools"):
                        # Standard notification (proper MCP way)
                        await self.mcp._mcp_server.request_context.session.send_notification(
                            types.ToolListChangedNotification(
                                method="notifications/tools/list_changed"
                            ),
                            related_request_id=self.mcp._mcp_server.request_context.request_id,
                        )
                        logger.debug(
                            f"Sent tools/list_changed notification {self.mcp._mcp_server.request_context.request_id}"
                        )

                        # Execute external command to trigger client refresh (workaround for clients
                        # that don't properly handle tools/list_changed notifications)
                        await self._execute_list_changed_command()

                except Exception as notify_err:
                    logger.warning(
                        f"Failed to emit tools/list_changed notification: {notify_err}"
                    )

                return result_json_str # Return the original result string

    async def _execute_list_changed_command(self) -> None:
        """Execute external command after tools list changes to trigger client refresh."""
        if not self.list_changed_exec_cmd:
            return

        logger = get_logger()
        try:
            # Execute command in background without blocking
            logger.debug(f"Executing list changed command: {self.list_changed_exec_cmd}")

            # Run command asynchronously
            process = await asyncio.create_subprocess_shell(
                self.list_changed_exec_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            # Wait for completion with timeout
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=60.0  # 60 second timeout
            )

            if process.returncode == 0:
                logger.debug("List changed command executed successfully")
            else:
                logger.warning(
                    f"List changed command failed with code {process.returncode}: "
                    f"stderr={stderr.decode().strip()}"
                )

        except asyncio.TimeoutError:
            logger.warning("List changed command timed out after 60 seconds")
        except Exception as e:
            logger.warning(f"Error executing list changed command: {e}")

    # Legacy methods for manual initialization (fallback)
    async def initialize_resources(self) -> None:
        """Manual initialization - fallback if lifespan doesn't work."""
        await self._initialize_resources()

    async def cleanup_resources(self) -> None:
        """Manual cleanup - fallback if lifespan doesn't work."""
        await self._cleanup_resources()

    def get_app(self):
        """Get the FastMCP application."""
        return self.mcp

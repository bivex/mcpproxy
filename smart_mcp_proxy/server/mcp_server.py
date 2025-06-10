"""FastMCP server implementation for Smart MCP Proxy."""

import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from fastmcp import FastMCP
from fastmcp.client import Client
from fastmcp.tools.tool import Tool  # type: ignore[import-not-found]

from ..models.schemas import ProxyConfig, ServerConfig, ToolRegistration, SearchResult, EmbedderType
from ..persistence.facade import PersistenceFacade
from ..indexer.facade import IndexerFacade
from ..logging import get_logger, configure_logging
from .config import ConfigLoader


class SmartMCPProxyServer:
    """Smart MCP Proxy server using FastMCP."""
    
    def __init__(self, config_path: str = "mcp_config.json"):
        self.config_path = config_path
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.load_config()
        
        # Transport configuration from environment variables
        self.transport = os.getenv("SP_TRANSPORT", "stdio")
        self.host = os.getenv("SP_HOST", "127.0.0.1")
        self.port = int(os.getenv("SP_PORT", "8000"))
        
        # Tool pool limit configuration
        self.tools_limit = int(os.getenv("SP_TOOLS_LIMIT", "15"))
        
        # Will be initialized in lifespan
        self.persistence: PersistenceFacade | None = None
        self.indexer: IndexerFacade | None = None
        
        # Track upstream clients and proxy servers
        self.upstream_clients: dict[str, Client] = {}
        self.proxy_servers: dict[str, FastMCP] = {}
        self.registered_tools: dict[str, ToolRegistration] = {}
        self.current_tool_registrations: dict[str, Any] = {}
        
        # Track tool pool with metadata for eviction
        self.tool_pool_metadata: dict[str, dict[str, Any]] = {}  # tool_name -> {timestamp, score, original_score}
        
        # Initialize FastMCP server with transport configuration
        fastmcp_kwargs = {
            "name": "Smart MCP Proxy",
            "instructions": """
            This server provides intelligent tool discovery and proxying for MCP servers.
            Use 'retrieve_tools' to search and access tools from configured upstream servers.
            proxy tools are dynamically created and registered on the fly in accordance with the search results.
            Pass the original user query (if possible) to the 'retrieve_tools' tool to get the search results.
            """,
            "lifespan": self._lifespan
        }
        
        # Add host and port for non-stdio transports
        if self.transport != "stdio":
            fastmcp_kwargs["host"] = self.host
            fastmcp_kwargs["port"] = self.port
        
        self.mcp = FastMCP(**fastmcp_kwargs)
        self._setup_tools()
    
    def run(self) -> None:
        """Run the Smart MCP Proxy server with full initialization."""
        # Configure logging
        configure_logging()
        logger = get_logger()
        
        # Check for config file
        config_path = os.getenv("MCP_CONFIG_PATH", self.config_path)
        
        if not Path(config_path).exists():
            logger.warning(f"Configuration file not found: {config_path}")
            logger.info("Creating sample configuration...")
            
            config_loader = ConfigLoader()
            config_loader.create_sample_config(config_path)
            
            logger.info(f"Please edit {config_path} and set required environment variables")
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
                self.mcp.run(transport="streamable-http", host=self.host, port=self.port)
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
        
        await self._initialize_resources()
        
        try:
            yield  # Server is running
        finally:
            await self._cleanup_resources()
    
    async def _initialize_resources(self) -> None:
        """Core resource initialization logic."""
        logger = get_logger()
        
        # Determine vector dimension based on embedder type
        if self.config.embedder == EmbedderType.BM25:
            vector_dimension = 1  # BM25 uses placeholder vectors
        else:
            # For vector embedders, we'll set dimension after creating the embedder
            # Default to 384 for now, will be updated if needed
            vector_dimension = 384
        
        # Initialize persistence with appropriate dimension
        self.persistence = PersistenceFacade(vector_dimension=vector_dimension)
        
        # Initialize indexer
        self.indexer = IndexerFacade(
            self.persistence, 
            self.config.embedder, 
            self.config.hf_model
        )
        
        # For non-BM25 embedders, update persistence with actual dimension
        if self.config.embedder != EmbedderType.BM25:
            actual_dimension = self.indexer.embedder.get_dimension()
            if actual_dimension != vector_dimension:
                # Recreate persistence with correct dimension
                await self.persistence.close()
                self.persistence = PersistenceFacade(vector_dimension=actual_dimension)
                # Update indexer to use new persistence
                self.indexer.persistence = self.persistence
        
        # Create upstream clients and proxy servers
        await self._create_upstream_clients_and_proxies()
        
        # Discover and index tools from upstream servers
        await self.discover_and_index_tools()
        
        logger.info("Smart MCP Proxy resources initialized")
    
    async def _cleanup_resources(self) -> None:
        """Core resource cleanup logic."""
        logger = get_logger()
        logger.info("Shutting down Smart MCP Proxy resources...")
        
        # Close upstream clients
        for client in self.upstream_clients.values():
            try:
                if hasattr(client, 'close'):
                    await client.close()
            except Exception as e:
                logger.warning(f"Error closing client: {e}")
        
        if self.persistence:
            await self.persistence.close()
        logger.info("Smart MCP Proxy resources cleaned up")
    
    def _setup_tools(self) -> None:
        """Setup core proxy tools."""
        
        @self.mcp.tool()
        async def retrieve_tools(query: str) -> str:
            """Search and retrieve tools based on query. Tools are dynamically created and registered on the fly in accordance with the search results.
            
            Args:
                query: Search query for finding relevant tools, pass the original user query (if possible) to the 'retrieve_tools' tool to get the search results.
                
            Returns:
                JSON string with discovered tools information
            """
            try:
                # Ensure indexer is initialized
                if not self.indexer:
                    return json.dumps({"error": "Indexer not initialized"})
                
                # Search for tools
                results = await self.indexer.search_tools(query, self.config.top_k)
                
                if not results:
                    return json.dumps({"message": "No relevant tools found", "tools": []})
                
                # Prepare tools for registration
                tools_to_register = []
                for result in results:
                    tool_name = f"{result.tool.server_name}_{result.tool.name}"
                    
                    # Skip if already registered
                    if tool_name in self.current_tool_registrations:
                        # Update timestamp for existing tool (freshen it)
                        if tool_name in self.tool_pool_metadata:
                            self.tool_pool_metadata[tool_name]["timestamp"] = time.time()
                            self.tool_pool_metadata[tool_name]["score"] = max(
                                self.tool_pool_metadata[tool_name]["score"], 
                                result.score
                            )
                        continue
                    
                    tools_to_register.append((tool_name, result.tool, result.score))
                
                # Enforce pool limit before registering new tools
                evicted_tools = []
                if tools_to_register:
                    new_tools_info = [(name, score) for name, _, score in tools_to_register]
                    evicted_tools = await self._enforce_tool_pool_limit(new_tools_info)
                
                # Register new tools
                newly_registered = []
                for tool_name, tool_metadata, score in tools_to_register:
                    await self._register_proxy_tool(tool_metadata, tool_name, score)
                    newly_registered.append(tool_name)
                
                # Prepare tool information
                registered_tools = []
                for result in results:
                    tool_name = f"{result.tool.server_name}_{result.tool.name}"
                    registered_tools.append({
                        "name": tool_name,
                        "original_name": result.tool.name,
                        "server": result.tool.server_name,
                        "description": result.tool.description,
                        "score": result.score,
                        "newly_registered": tool_name in newly_registered
                    })
                
                message = f"Found {len(registered_tools)} tools, registered {len(newly_registered)} new tools"
                if evicted_tools:
                    message += f", evicted {len(evicted_tools)} tools to stay within limit ({self.tools_limit})"
                
                return json.dumps({
                    "message": message,
                    "tools": registered_tools,
                    "newly_registered": newly_registered,
                    "evicted_tools": evicted_tools,
                    "pool_size": len(self.current_tool_registrations),
                    "pool_limit": self.tools_limit,
                    "query": query
                })
            
            except Exception as e:
                return json.dumps({"error": str(e)})
    
    def _calculate_tool_weight(self, score: float, added_timestamp: float) -> float:
        """Calculate weighted score for tool eviction based on score and freshness.
        
        Args:
            score: Original search score (0.0 to 1.0)
            added_timestamp: Timestamp when tool was added to pool
            
        Returns:
            Weighted score (higher is better, less likely to be evicted)
        """
        current_time = time.time()
        age_seconds = current_time - added_timestamp
        
        # Normalize age (0 = fresh, 1 = old)
        # Tools older than 30 minutes get maximum age penalty
        max_age_seconds = 30 * 60  # 30 minutes
        age_normalized = min(1.0, age_seconds / max_age_seconds)
        
        # Weighted formula: 70% score, 30% freshness
        score_weight = 0.7
        freshness_weight = 0.3
        freshness_score = 1.0 - age_normalized
        
        weighted_score = (score * score_weight) + (freshness_score * freshness_weight)
        return weighted_score
    
    async def _enforce_tool_pool_limit(self, new_tools: list[tuple[str, float]]) -> list[str]:
        """Enforce tool pool limit by evicting lowest-scoring tools.
        
        Args:
            new_tools: List of (tool_name, score) for tools to be added
            
        Returns:
            List of tool names that were evicted
        """
        current_pool_size = len(self.current_tool_registrations)
        new_tools_count = len(new_tools)
        total_after_addition = current_pool_size + new_tools_count
        
        if total_after_addition <= self.tools_limit:
            return []  # No eviction needed
        
        # Calculate how many tools to evict
        tools_to_evict_count = total_after_addition - self.tools_limit
        
        # Calculate weighted scores for all existing tools
        tool_weights = []
        for tool_name, metadata in self.tool_pool_metadata.items():
            if tool_name in self.current_tool_registrations:
                weight = self._calculate_tool_weight(metadata["score"], metadata["timestamp"])
                tool_weights.append((tool_name, weight))
        
        # Sort by weight (ascending - lowest weights first for eviction)
        tool_weights.sort(key=lambda x: x[1])
        
        # Evict the lowest scoring tools
        evicted_tools = []
        for i in range(min(tools_to_evict_count, len(tool_weights))):
            tool_name = tool_weights[i][0]
            await self._evict_tool(tool_name)
            evicted_tools.append(tool_name)
        
        return evicted_tools
    
    async def _evict_tool(self, tool_name: str) -> None:
        """Remove a tool from the active pool.
        
        Args:
            tool_name: Name of tool to evict
        """
        # Remove from FastMCP server
        if hasattr(self.mcp, 'remove_tool'):
            self.mcp.remove_tool(tool_name)
        
        # Clean up tracking
        self.current_tool_registrations.pop(tool_name, None)
        self.registered_tools.pop(tool_name, None)
        self.tool_pool_metadata.pop(tool_name, None)
        
        logger = get_logger()
        logger.debug(f"Evicted tool from pool: {tool_name}")
    
    async def _register_proxy_tool(self, tool_metadata: Any, tool_name: str, score: float = 0.0) -> None:
        """Register a single tool as a proxy that calls the upstream server."""
        
        original_tool_name = tool_metadata.name
        server_name = tool_metadata.server_name
        description = (tool_metadata.description or f"Proxy tool for {tool_metadata.name}") + f" (Original tool: {original_tool_name} from {server_name}. Pass arguments as a dictionary in the 'arguments' parameter.)"
        
        # Create a wrapper function that will be converted to a tool
        # Capture variables in closure by using them as default parameters
        async def proxy_wrapper(arguments: dict[str, Any] = None, 
                               _original_name: str = original_tool_name,
                               _server_name: str = server_name) -> str:
            """Dynamically created proxy tool."""
            try:
                # Get the proxy server for this tool's server
                proxy_server = self.proxy_servers.get(_server_name)
                if not proxy_server:
                    return f"Error: Server {_server_name} not available"
                
                # Use empty dict if no arguments provided
                tool_args = arguments or {}
                
                # Call the original tool through the proxy server
                result = await proxy_server._mcp_call_tool(_original_name, tool_args)
                
                # Extract text content from result
                if result and len(result) > 0:
                    content = result[0]
                    if hasattr(content, 'text'):
                        return content.text
                    elif isinstance(content, dict) and 'text' in content:
                        return content['text']
                
                return str(result)
            
            except Exception as e:
                return f"Error calling {_original_name}: {str(e)}"
        
        # Set function name and docstring
        proxy_wrapper.__name__ = tool_name
        proxy_wrapper.__doc__ = description
        
        # Create tool using Tool.from_function and add it to the server
        tool = Tool.from_function(proxy_wrapper, name=tool_name, description=description)
        self.mcp.add_tool(tool)
        
        # Track this registration
        self.current_tool_registrations[tool_name] = proxy_wrapper
        
        # Track tool pool metadata for eviction logic
        self.tool_pool_metadata[tool_name] = {
            "timestamp": time.time(),
            "score": score,
            "original_score": score
        }
        
        # Parse input schema if it's a string
        input_schema = {}
        if hasattr(tool_metadata, 'params_json'):
            if isinstance(tool_metadata.params_json, str):
                try:
                    input_schema = json.loads(tool_metadata.params_json)
                except json.JSONDecodeError:
                    input_schema = {}
            elif isinstance(tool_metadata.params_json, dict):
                input_schema = tool_metadata.params_json
        
        # Track in registered_tools
        self.registered_tools[tool_name] = ToolRegistration(
            name=tool_name,
            description=description,
            input_schema=input_schema,
            server_name=server_name
        )
        
        logger = get_logger()
        logger.debug(f"Registered proxy tool: {tool_name} (original: {original_tool_name}, score: {score:.3f})")
    
    async def _create_upstream_clients_and_proxies(self) -> None:
        """Create upstream clients and proxy servers for all configured servers."""
        for server_name, server_config in self.config.mcp_servers.items():
            try:
                if server_config.url:
                    # Create client for URL-based server
                    client = Client(server_config.url)
                    # Create proxy server using FastMCP.from_client
                    proxy_server = FastMCP.from_client(client, name=f"{server_name}_proxy")
                elif server_config.command:
                    # Create client for command-based server
                    config_dict = {
                        "mcpServers": {
                            server_name: {
                                "command": server_config.command,
                                "args": getattr(server_config, 'args', []),
                                "env": getattr(server_config, 'env', {})
                            }
                        }
                    }
                    client = Client(config_dict)
                    # Create proxy server using FastMCP.from_client
                    proxy_server = FastMCP.from_client(client, name=f"{server_name}_proxy")
                else:
                    continue
                
                self.upstream_clients[server_name] = client
                self.proxy_servers[server_name] = proxy_server
                logger = get_logger()
                logger.info(f"Created upstream client and proxy server for {server_name}")
            except Exception as e:
                logger = get_logger()
                logger.error(f"Error creating upstream client/proxy for {server_name}: {e}")
    
    async def discover_and_index_tools(self) -> None:
        """Discover tools from all configured servers and index them."""
        for server_name, proxy_server in self.proxy_servers.items():
            try:
                await self._discover_server_tools(server_name, proxy_server)
            except Exception as e:
                logger = get_logger()
                logger.error(f"Error discovering tools from {server_name}: {e}")
    
    async def _discover_server_tools(self, server_name: str, proxy_server: FastMCP) -> None:
        """Discover tools from a specific server using its proxy."""
        try:
            # Ensure indexer is available
            if not self.indexer:
                logger = get_logger()
                logger.warning(f"Indexer not available, skipping discovery for {server_name}")
                return
            
            # Get tools from the proxy server
            tools = await proxy_server.get_tools()
            
            # Index discovered tools
            for tool_name, tool_obj in tools.items():
                await self.indexer.index_tool(
                    name=tool_obj.name,
                    description=tool_obj.description or "",
                    server_name=server_name,
                    params=tool_obj.parameters,
                    tags=list(tool_obj.tags) if tool_obj.tags else [],
                    annotations=tool_obj.annotations or {}
                )
            
            logger = get_logger()
            logger.info(f"Indexed {len(tools)} tools from {server_name}")
        
        except Exception as e:
            logger = get_logger()
            logger.error(f"Error discovering tools from {server_name}: {e}")
    
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
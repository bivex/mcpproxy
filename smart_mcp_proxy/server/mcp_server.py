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
        
        # Store Tool objects from proxified servers in memory
        self.proxified_tools: dict[str, Tool] = {}  # tool_key -> Tool object
        
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
        # Configure logging with debug level if requested
        log_level = os.getenv("SP_LOG_LEVEL", "INFO")
        configure_logging(log_level)
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
        
        # Check if we should reset data (useful when dimensions change)
        reset_data = os.getenv("SP_RESET_DATA", "false").lower() == "true"
        
        # Determine vector dimension based on embedder type
        if self.config.embedder == EmbedderType.BM25:
            vector_dimension = 1  # BM25 uses placeholder vectors
        else:
            # For vector embedders, we'll set dimension after creating the embedder
            # Default to 384 for now, will be updated if needed
            vector_dimension = 384
        
        # Initialize persistence with appropriate dimension
        self.persistence = PersistenceFacade(vector_dimension=vector_dimension)
        
        # Reset data if requested
        if reset_data:
            logger.info("Resetting all data as requested...")
            await self.persistence.reset_all_data()
        
        # Initialize indexer
        self.indexer = IndexerFacade(
            self.persistence, 
            self.config.embedder, 
            self.config.hf_model
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
                logger.info(f"Updating vector dimension from {vector_dimension} to {actual_dimension}")
                await self.persistence.close()
                self.persistence = PersistenceFacade(vector_dimension=actual_dimension)
                # Update indexer to use new persistence
                self.indexer.persistence = self.persistence
        
        # Create upstream clients and proxy servers
        await self._create_upstream_clients_and_proxies()
        logger.debug(f"Created {len(self.proxy_servers)} proxy servers and {len(self.upstream_clients)} upstream clients")
        
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
                    
                    # Get the actual Tool object from memory
                    tool_key = f"{result.tool.server_name}_{result.tool.name}"
                    if tool_key in self.proxified_tools:
                        actual_tool = self.proxified_tools[tool_key]
                        tools_to_register.append((tool_name, actual_tool, result.score))
                    else:
                        logger = get_logger()
                        logger.warning(f"Tool {tool_key} not found in proxified_tools memory")
                
                # Enforce pool limit before registering new tools
                evicted_tools = []
                if tools_to_register:
                    new_tools_info = [(name, score) for name, _, score in tools_to_register]
                    evicted_tools = await self._enforce_tool_pool_limit(new_tools_info)
                
                # Register new tools using actual Tool objects
                newly_registered = []
                for tool_name, actual_tool, score in tools_to_register:
                    await self._register_proxy_tool(actual_tool, tool_name, score)
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
        """Register a single tool as a proxy that calls the upstream server transparently using Tool.from_tool."""
        from fastmcp.tools.tool import Tool

        # tool_metadata should be a Tool object from proxified servers
        if not isinstance(tool_metadata, Tool):
            logger = get_logger()
            logger.error(f"Expected Tool object, got {type(tool_metadata)}")
            return

        original_tool: Tool = tool_metadata
        # Extract server_name from tool_name (format: servername_toolname)
        if '_' in tool_name:
            server_name = tool_name.split('_', 1)[0]
        else:
            server_name = 'unknown'
        original_tool_name = original_tool.name

        # Define a transform_fn that forwards the call to the upstream server
        async def transform_fn(**kwargs):
            proxy_server = self.proxy_servers.get(server_name)
            if not proxy_server:
                return f"Error: Server {server_name} not available"
            # Forward the call to the upstream tool with original parameters
            result = await proxy_server._mcp_call_tool(original_tool_name, kwargs)
            if result and len(result) > 0:
                content = result[0]
                if hasattr(content, 'text'):
                    return content.text
                elif isinstance(content, dict) and 'text' in content:
                    return content['text']
            return str(result)

        # Create a proxified tool using Tool.from_tool
        proxified_tool = Tool.from_tool(
            tool=original_tool,
            transform_fn=transform_fn,
            name=tool_name  # Use unique name (e.g., servername_toolname)
        )
        self.mcp.add_tool(proxified_tool)

        # Track this registration
        self.current_tool_registrations[tool_name] = proxified_tool
        self.tool_pool_metadata[tool_name] = {
            "timestamp": time.time(),
            "score": score,
            "original_score": score
        }

        # Track in registered_tools for metadata
        input_schema = original_tool.parameters if hasattr(original_tool, 'parameters') else {}
        self.registered_tools[tool_name] = ToolRegistration(
            name=tool_name,
            description=original_tool.description or "",
            input_schema=input_schema,
            server_name=server_name
        )

        logger = get_logger()
        logger.debug(f"Registered proxy tool (from_tool): {tool_name} (original: {original_tool_name}, score: {score:.3f})")
    
    async def _create_upstream_clients_and_proxies(self) -> None:
        """Create upstream clients and proxy servers for all configured servers."""
        logger = get_logger()
        logger.info(f"Creating upstream clients for {len(self.config.mcp_servers)} servers...")
        
        for server_name, server_config in self.config.mcp_servers.items():
            try:
                logger.debug(f"Creating client for {server_name}: url={server_config.url}, command={getattr(server_config, 'command', None)}")
                
                if server_config.url:
                    # Create client for URL-based server
                    logger.debug(f"Creating URL-based client for {server_name} at {server_config.url}")
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
                    logger.debug(f"Creating command-based client for {server_name}: {config_dict}")
                    client = Client(config_dict)
                    # Create proxy server using FastMCP.from_client
                    proxy_server = FastMCP.from_client(client, name=f"{server_name}_proxy")
                else:
                    logger.warning(f"Skipping {server_name}: no URL or command specified")
                    continue
                
                self.upstream_clients[server_name] = client
                self.proxy_servers[server_name] = proxy_server
                logger.info(f"Created upstream client and proxy server for {server_name}")
                
                # Test if proxy server is properly initialized
                logger.debug(f"Proxy server for {server_name}: type={type(proxy_server)}, has_get_tools={hasattr(proxy_server, 'get_tools')}")
                
            except Exception as e:
                logger.error(f"Error creating upstream client/proxy for {server_name}: {type(e).__name__}: {e}")
                logger.error(f"Full exception details for {server_name}:", exc_info=True)
                logger.debug(f"Server config: {server_config}")
                # Continue with other servers even if one fails
    
    async def discover_and_index_tools(self) -> None:
        """Discover tools from all configured servers and index them."""
        logger = get_logger()
        logger.info(f"Starting tool discovery for {len(self.proxy_servers)} servers...")
        
        for server_name, proxy_server in self.proxy_servers.items():
            try:
                logger.debug(f"Starting tool discovery for server: {server_name}")
                await self._discover_server_tools(server_name, proxy_server)
                logger.debug(f"Completed tool discovery for server: {server_name}")
            except Exception as e:
                logger.error(f"Error discovering tools from {server_name}: {type(e).__name__}: {e}")
                logger.error(f"Exception details for {server_name}:", exc_info=True)
    
    async def _discover_server_tools(self, server_name: str, proxy_server: FastMCP) -> None:
        """Discover tools from a specific server using its proxy."""
        logger = get_logger()
        
        try:
            # Ensure indexer is available
            if not self.indexer:
                logger.warning(f"Indexer not available, skipping discovery for {server_name}")
                return
            
            logger.debug(f"Getting tools from proxy server for {server_name}...")
            
            # Check if proxy server is ready
            if not hasattr(proxy_server, 'get_tools'):
                logger.error(f"Proxy server for {server_name} does not have get_tools method")
                return
            
            # Get tools from the proxy server - these are actual Tool objects
            try:
                logger.debug(f"Calling get_tools() on {server_name} proxy server...")
                tools = await proxy_server.get_tools()
                logger.debug(f"get_tools() returned: {type(tools)} with {len(tools) if tools else 0} items")
            except Exception as get_tools_error:
                logger.error(f"Error calling get_tools on {server_name}: {type(get_tools_error).__name__}: {get_tools_error}")
                raise
            
            if not tools:
                logger.warning(f"No tools returned from {server_name}")
                return
            
            # Store Tool objects in memory and index them
            indexed_count = 0
            for tool_name, tool_obj in tools.items():
                try:
                    # Store Tool object in memory with server context
                    tool_key = f"{server_name}_{tool_name}"
                    self.proxified_tools[tool_key] = tool_obj
                    
                    logger.debug(f"Indexing tool: {tool_name} from {server_name}")
                    await self.indexer.index_tool_from_object(tool_obj, server_name)
                    indexed_count += 1
                except Exception as tool_error:
                    error_msg = str(tool_error)
                    if "assert d == self.d" in error_msg or "dimension" in error_msg.lower():
                        logger.error(f"Vector dimension mismatch for tool {tool_name}. This usually means:")
                        logger.error("1. Existing FAISS index has different dimension than current embedder")
                        logger.error("2. Solution: Set SP_RESET_DATA=true environment variable to reset data")
                        logger.error("3. Or delete these data files manually:")
                        logger.error("   - tools.faiss (FAISS vector index)")
                        logger.error("   - proxy.db (SQLite database)")
                        
                        # Add BM25-specific info if using BM25
                        if self.config.embedder == EmbedderType.BM25 and self.indexer and hasattr(self.indexer.embedder, 'index_dir'):
                            bm25_dir = self.indexer.embedder.index_dir
                            logger.error(f"   - {bm25_dir}/ (BM25 index directory)")
                        else:
                            logger.error("   - BM25 index directory (if using BM25: usually a temp dir with bm25s_index/)")
                        logger.error("4. For BM25: the index dir is typically a temp directory starting with 'bm25s_'")
                    logger.error(f"Error indexing tool {tool_name} from {server_name}: {type(tool_error).__name__}: {tool_error}")
                    logger.error(f"Full stack trace for {tool_name}:", exc_info=True)
                    logger.debug(f"Tool object details: {tool_obj}")
            
            logger.info(f"Successfully indexed {indexed_count}/{len(tools)} tools from {server_name}")
        
        except Exception as e:
            logger.error(f"Error discovering tools from {server_name}: {type(e).__name__}: {e}")
            logger.error(f"Full exception details:", exc_info=True)
            raise
    
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
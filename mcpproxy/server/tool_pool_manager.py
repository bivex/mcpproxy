import asyncio
import json
import time
from typing import Any, Callable

import mcp.types as types # type: ignore[import-untyped]
from fastmcp import FastMCP # type: ignore[import-untyped]
from fastmcp.client import Client # type: ignore[import-untyped]
from fastmcp.tools.tool import Tool # type: ignore[import-not-found]

from ..indexer.facade import IndexerFacade
from ..logging import get_logger
from ..models.schemas import ToolRegistration, ToolMetadata # Add ToolMetadata import
from ..persistence.facade import PersistenceFacade
from ..utils.name_sanitization.name_sanitizer import sanitize_tool_name
from ..utils.tool_scoring.tool_weight_calculator import calculate_tool_weight
from .config.config import ProxyConfig # Corrected import

logger = get_logger()


class ToolPoolManager:
    """Manages the pool of available tools, including discovery, registration, and eviction."""

    def __init__(
        self,
        mcp_app: FastMCP,
        indexer: IndexerFacade,
        persistence: PersistenceFacade,
        config: ProxyConfig,  # Change to accept ProxyConfig directly
        proxy_servers: dict[str, FastMCP],
        truncate_output_fn: Callable[[str], str],
        truncate_output_len: int | None,
    ):
        self.mcp = mcp_app
        self.indexer = indexer
        self.persistence = persistence
        self.config = config  # Store the entire config object
        self.proxy_servers = proxy_servers
        self._truncate_output = truncate_output_fn
        self.truncate_output_len = truncate_output_len

        self.registered_tools: dict[str, ToolRegistration] = {}
        self.current_tool_registrations: dict[str, Any] = {}
        self.proxified_tools: dict[str, Tool] = {}
        self.tool_pool_metadata: dict[
            str, dict[str, Any]
        ] = {}  # tool_name -> {timestamp, score, original_score}

    async def retrieve_tools(self, query: str) -> str:
        """Search and retrieve tools based on query. Tools are dynamically created and registered on the fly in accordance with the search results."""
        try:
            if not self.indexer:
                return json.dumps({"error": "Indexer not initialized"})

            results = await self.indexer.search_tools(query, self.config.top_k)

            if not results:
                return json.dumps({"message": "No relevant tools found", "tools": []})

            tools_to_register, newly_registered_tool_names = await self._process_search_results(results)

            evicted_tools = []
            if tools_to_register:
                new_tools_info = [
                    (name, score) for name, _, score in tools_to_register
                ]
                evicted_tools = await self._enforce_tool_pool_limit(new_tools_info)

            for tool_name, actual_tool, score in tools_to_register:
                original_server_name = self._get_original_server_name(tool_name, results)
                await self._register_proxy_tool(
                    actual_tool, tool_name, score, original_server_name
                )

            return self._build_response_payload(query, results, newly_registered_tool_names, evicted_tools)

        except Exception as e:
            logger.error(f"Error in retrieve_tools: {e}")
            return json.dumps({"error": str(e)})

    async def _process_search_results(self, results: list) -> tuple[list, list]:
        tools_to_register = []
        newly_registered_tool_names = []
        for result in results:
            tool_name = sanitize_tool_name(
                result.tool.server_name, result.tool.name, self.config.tool_name_limit
            )

            if tool_name in self.current_tool_registrations:
                self._update_tool_metadata(tool_name, result.score)
                continue

            if tool_name in self.proxified_tools:
                actual_tool = self.proxified_tools[tool_name]
                tools_to_register.append((tool_name, actual_tool, result.score))
                newly_registered_tool_names.append(tool_name)
            else:
                logger.warning(f"Tool {tool_name} not found in proxified_tools memory")
        return tools_to_register, newly_registered_tool_names

    def _update_tool_metadata(self, tool_name: str, score: float) -> None:
        if tool_name in self.tool_pool_metadata:
            self.tool_pool_metadata[tool_name]["timestamp"] = time.time()
            self.tool_pool_metadata[tool_name]["score"] = max(
                self.tool_pool_metadata[tool_name]["score"],
                score,
            )

    def _get_original_server_name(self, tool_name: str, results: list) -> str | None:
        for result in results:
            if (
                sanitize_tool_name(
                    result.tool.server_name,
                    result.tool.name,
                    self.config.tool_name_limit,
                )
                == tool_name
            ):
                return result.tool.server_name
        return None

    def _build_response_payload(self, query: str, results: list, newly_registered: list, evicted_tools: list) -> str:
        registered_tools = []
        for result in results:
            tool_name = sanitize_tool_name(
                result.tool.server_name, result.tool.name, self.config.tool_name_limit
            )
            registered_tools.append(
                {
                    "name": tool_name,
                    "original_name": result.tool.name,
                    "server": result.tool.server_name,
                    "description": result.tool.description,
                    "score": result.score,
                    "newly_registered": tool_name in newly_registered,
                }
            )

        message = f"Found {len(registered_tools)} tools, registered {len(newly_registered)} new tools"
        if evicted_tools:
            message += f", evicted {len(evicted_tools)} tools to stay within limit ({self.config.tools_limit})"

        return json.dumps(
            {
                "message": message,
                "tools": registered_tools,
                "newly_registered": newly_registered,
                "evicted_tools": evicted_tools,
                "pool_size": len(self.current_tool_registrations),
                "pool_limit": self.config.tools_limit,
                "total_available_tools": len(self.proxified_tools),
                "query": query,
            }
        )

    async def call_tool(self, name: str, args: dict[str, Any]) -> str:
        """Execute a tool on the upstream server using the call_tool interface."""
        try:
            if "_" not in name:
                return json.dumps({"error": f"Invalid tool name format: {name}. Expected format: servername_toolname"})
            
            if not self.indexer:
                return json.dumps({"error": "Indexer not initialized"})
            
            all_tools = await self.persistence.get_all_tools() if self.persistence else []
            
            matching_tool = None
            for tool_metadata in all_tools:
                sanitized_name = sanitize_tool_name(tool_metadata.server_name, tool_metadata.name, self.config.tool_name_limit) # Use self.config.tool_name_limit
                if sanitized_name == name:
                    matching_tool = tool_metadata
                    break
            
            if not matching_tool:
                return json.dumps({"error": f"Tool '{name}' not found. Use retrieve_tools first to discover available tools."})
            
            server_name = matching_tool.server_name
            proxy_server = self.proxy_servers.get(server_name)
            if not proxy_server:
                return json.dumps({"error": f"Server '{server_name}' not available"})
            
            original_tool_name = matching_tool.name
            logger.debug(f"Executing tool '{original_tool_name}' on server '{server_name}' with args: {args}")
            
            result = await proxy_server.call(original_tool_name, **args)
            
            output = ""
            if result and len(result) > 0:
                content = result[0]
                if hasattr(content, "text"):
                    output = content.text
                elif isinstance(content, dict) and "text" in content:
                    output = content["text"]
                else:
                    output = str(result)
            else:
                output = str(result)

            output = self._truncate_output(output, self.truncate_output_len)
            
            return output

        except Exception as e:
            logger.error(f"Error executing tool '{name}': {e}")
            return json.dumps({"error": f"Error executing tool '{name}': {str(e)}"})

    async def _enforce_tool_pool_limit(self, new_tools: list[tuple[str, float]]) -> list[str]:
        """Enforce tool pool limit by evicting lowest-scoring tools."""
        current_pool_size = len(self.current_tool_registrations)
        new_tools_count = len(new_tools)
        total_after_addition = current_pool_size + new_tools_count

        if total_after_addition <= self.config.tools_limit: # Use self.config.tools_limit
            return []

        tools_to_evict_count = total_after_addition - self.config.tools_limit # Use self.config.tools_limit

        tool_weights = []
        for tool_name, metadata in self.tool_pool_metadata.items():
            if tool_name in self.current_tool_registrations:
                weight = calculate_tool_weight(metadata["score"], metadata["timestamp"])
                tool_weights.append((tool_name, weight))

        tool_weights.sort(key=lambda x: x[1])

        evicted_tools = []
        for i in range(min(tools_to_evict_count, len(tool_weights))):
            tool_name = tool_weights[i][0]
            await self._evict_tool(tool_name)
            evicted_tools.append(tool_name)

        return evicted_tools

    async def _evict_tool(self, tool_name: str) -> None:
        """Remove a tool from the active pool."""
        if hasattr(self.mcp, "remove_tool"):
            self.mcp.remove_tool(tool_name)

        self.current_tool_registrations.pop(tool_name, None)
        self.registered_tools.pop(tool_name, None)
        self.tool_pool_metadata.pop(tool_name, None)

        logger.debug(f"Evicted tool from pool: {tool_name}")

    async def _register_proxy_tool(
        self,
        tool_metadata: Any,
        tool_name: str,
        score: float = 0.0,
        server_name: str | None = None,
    ) -> None:
        """Register a single tool as a proxy that calls the upstream server transparently using Tool.from_tool."""
        from fastmcp.tools.tool import Tool

        if not isinstance(tool_metadata, Tool):
            logger.error(f"Expected Tool object, got {type(tool_metadata)}")
            return

        original_tool: Tool = tool_metadata
        if server_name is None:
            if "_" in tool_name:
                server_name = tool_name.split("_", 1)[0]
            else:
                server_name = "unknown"
        original_tool_name = original_tool.name

        async def transform_fn(**kwargs):
            proxy_server = self.proxy_servers.get(server_name)
            if not proxy_server:
                return f"Error: Server {server_name} not available"
            
            result = await proxy_server.call(original_tool_name, **kwargs)
            output = ""
            if result and len(result) > 0:
                content = result[0]
                if hasattr(content, "text"):
                    output = content.text
                elif isinstance(content, dict) and "text" in content:
                    output = content["text"]
                else:
                    output = str(result)
            else:
                output = str(result)

            return self._truncate_output(output, self.truncate_output_len)

        proxified_tool = Tool.from_tool(
            tool=original_tool,
            transform_fn=transform_fn,
            name=tool_name,
        )
        self.mcp.add_tool(proxified_tool)

        self.current_tool_registrations[tool_name] = proxified_tool
        self.tool_pool_metadata[tool_name] = {
            "timestamp": time.time(),
            "score": score,
            "original_score": score,
        }

        input_schema = (
            original_tool.parameters if hasattr(original_tool, "parameters") else {}
        )
        self.registered_tools[tool_name] = ToolRegistration(
            name=tool_name,
            description=original_tool.description or "",
            input_schema=input_schema,
            server_name=server_name,
        )

        logger.debug(
            f"Registered proxy tool (from_tool): {tool_name} (original: {original_tool_name}, score: {score:.3f})"
        )

    async def add_proxified_tool_to_memory(self, sanitized_key: str, tool_obj: Tool) -> None:
        self.proxified_tools[sanitized_key] = tool_obj

    def get_proxified_tools(self) -> dict[str, Tool]:
        return self.proxified_tools

    def get_current_tool_registrations(self) -> dict[str, Any]:
        return self.current_tool_registrations 

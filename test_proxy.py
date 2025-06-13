#!/usr/bin/env python3
"""Test script for Smart MCP Proxy refactored implementation."""

import asyncio
import json
from pathlib import Path

from mcpproxy.server.mcp_server import SmartMCPProxyServer
from mcpproxy.logging import configure_logging, get_logger


async def test_proxy_initialization():
    """Test proxy server initialization with sample config."""
    
    # Create minimal test config
    config_data = {
        "mcpServers": {
            "test_server": {
                "url": "http://localhost:8080/mcp"
            }
        }
    }
    
    config_path = "test_config.json"
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    try:
        # Initialize proxy server
        proxy = SmartMCPProxyServer(config_path)
        
        logger = get_logger()
        logger.info("✓ Proxy server created successfully")
        logger.info(f"✓ FastMCP instance: {proxy.mcp}")
        logger.info(f"✓ Proxy servers dict: {proxy.proxy_servers}")
        logger.info(f"✓ Registered tools: {proxy.registered_tools}")
        
        # Test that the retrieve_tools function exists
        tools = await proxy.mcp.get_tools()
        logger.info(f"✓ Available tools: {list(tools.keys())}")
        
        if "retrieve_tools" in tools:
            logger.info("✓ retrieve_tools tool is properly registered")
        else:
            logger.warning("✗ retrieve_tools tool not found")
        
        # Clean up proxy
        await proxy.cleanup_resources()
        logger.info("✓ Proxy shut down successfully")
        
    finally:
        # Clean up test config
        Path(config_path).unlink(missing_ok=True)


async def test_proxy_server_creation():
    """Test proxy server creation without actual connections."""
    
    config_data = {
        "mcpServers": {
            "test_http": {
                "url": "http://example.com/mcp"
            },
            "test_command": {
                "command": "echo",
                "args": ["hello"],
                "env": {"TEST": "value"}
            }
        }
    }
    
    config_path = "test_config2.json" 
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    try:
        proxy = SmartMCPProxyServer(config_path)
        
        # Test proxy server creation logic (won't actually connect)
        logger = get_logger()
        logger.info("✓ Testing proxy server creation patterns")
        
        for server_name, server_config in proxy.config.mcp_servers.items():
            logger.info(f"  - Server: {server_name}")
            logger.info(f"    URL: {getattr(server_config, 'url', None)}")
            logger.info(f"    Command: {getattr(server_config, 'command', None)}")
        
        await proxy.cleanup_resources()
        
    finally:
        Path(config_path).unlink(missing_ok=True)


if __name__ == "__main__":
    configure_logging()
    logger = get_logger()
    
    logger.info("Testing Smart MCP Proxy refactored implementation...")
    
    asyncio.run(test_proxy_initialization())
    logger.info("")
    asyncio.run(test_proxy_server_creation())
    
    logger.info("All tests completed!") 
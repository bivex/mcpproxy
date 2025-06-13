#!/usr/bin/env python3
"""Entry point for mcpproxy package."""

from .server.mcp_server import SmartMCPProxyServer


def main():
    """Main function to start the Smart MCP Proxy."""
    proxy = SmartMCPProxyServer()
    proxy.run()


if __name__ == "__main__":
    main()

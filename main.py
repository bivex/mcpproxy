#!/usr/bin/env python3
"""Main entry point for Smart MCP Proxy."""

from mcpproxy import SmartMCPProxyServer


def main():
    """Main function to start the Smart MCP Proxy."""
    proxy = SmartMCPProxyServer()
    proxy.run()


if __name__ == "__main__":
    main() 
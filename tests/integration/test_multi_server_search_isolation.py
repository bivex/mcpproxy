"""Integration tests for multi-server search isolation."""

import pytest


class TestMultiServerSearchIsolation:
    """Test that tools from different servers are properly isolated yet searchable."""

    @pytest.mark.asyncio
    async def test_multi_server_search_isolation(self, temp_indexer_facade):
        """Test that tools from different servers are properly isolated yet searchable."""
        # Index similar tools on different servers
        servers_and_tools = [
            (
                "api-server",
                [
                    ("create_vm", "Create virtual machine"),
                    ("delete_vm", "Delete virtual machine"),
                ],
            ),
            (
                "storage-server",
                [
                    ("create_volume", "Create storage volume"),
                    ("delete_volume", "Delete storage volume"),
                ],
            ),
            (
                "network-server",
                [
                    ("create_network", "Create network interface"),
                    ("delete_network", "Delete network interface"),
                ],
            ),
        ]

        for server_name, tools in servers_and_tools:
            for tool_name, description in tools:
                await temp_indexer_facade.index_tool(
                    name=tool_name, description=description, server_name=server_name
                )

        # Test global search finds tools from all servers
        results = await temp_indexer_facade.search_tools("create", k=10)

        found_servers = {r.tool.server_name for r in results}
        assert "api-server" in found_servers
        assert "storage-server" in found_servers
        assert "network-server" in found_servers

        # Test server-specific retrieval
        for server_name, expected_tools in servers_and_tools:
            server_tools = await temp_indexer_facade.persistence.get_tools_by_server(
                server_name
            )
            assert len(server_tools) == len(expected_tools)

            server_tool_names = {tool.name for tool in server_tools}
            expected_names = {tool[0] for tool in expected_tools}
            assert server_tool_names == expected_names 

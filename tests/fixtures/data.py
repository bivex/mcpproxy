"""Sample data fixtures for testing."""

import json
from typing import Any


def get_sample_tools_data() -> list[dict[str, Any]]:
    """Get comprehensive sample tools data for testing."""
    return [
        {
            "name": "create_instance",
            "description": "Create a new virtual machine instance",
            "server_name": "company-api",
            "params": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Instance name"},
                    "flavor": {"type": "string", "description": "Instance flavor"},
                    "region": {"type": "string", "description": "Deployment region"},
                },
                "required": ["name", "flavor"],
            },
            "tags": ["compute", "vm", "creation"],
            "annotations": {"category": "compute", "cost": "medium"},
        },
        {
            "name": "delete_instance",
            "description": "Delete an existing virtual machine instance",
            "server_name": "company-api",
            "params": {
                "type": "object",
                "properties": {
                    "instance_id": {
                        "type": "string",
                        "description": "Instance ID to delete",
                    }
                },
                "required": ["instance_id"],
            },
            "tags": ["compute", "vm", "deletion"],
            "annotations": {"category": "compute", "cost": "low"},
        },
        {
            "name": "list_volumes",
            "description": "List all storage volumes in a region",
            "server_name": "storage-api",
            "params": {
                "type": "object",
                "properties": {
                    "region": {
                        "type": "string",
                        "description": "Region to list volumes from",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of volumes",
                    },
                },
            },
            "tags": ["storage", "volume", "list"],
            "annotations": {"category": "storage", "cost": "low"},
        },
        {
            "name": "create_volume",
            "description": "Create a new storage volume",
            "server_name": "storage-api",
            "params": {
                "type": "object",
                "properties": {
                    "size": {"type": "integer", "description": "Volume size in GB"},
                    "name": {"type": "string", "description": "Volume name"},
                    "region": {"type": "string", "description": "Deployment region"},
                },
                "required": ["size", "name"],
            },
            "tags": ["storage", "volume", "creation"],
            "annotations": {"category": "storage", "cost": "medium"},
        },
        {
            "name": "delete_volume",
            "description": "Delete a storage volume permanently",
            "server_name": "storage-api",
            "params": {
                "type": "object",
                "properties": {
                    "volume_id": {
                        "type": "string",
                        "description": "Volume ID to delete",
                    }
                },
                "required": ["volume_id"],
            },
            "tags": ["storage", "volume", "deletion"],
            "annotations": {"category": "storage", "cost": "low"},
        },
        {
            "name": "get_metrics",
            "description": "Retrieve system metrics and performance data",
            "server_name": "monitoring-api",
            "params": {
                "type": "object",
                "properties": {
                    "metric_type": {"type": "string", "description": "Type of metric"},
                    "start_time": {
                        "type": "string",
                        "description": "Start time for metrics",
                    },
                    "end_time": {
                        "type": "string",
                        "description": "End time for metrics",
                    },
                },
                "required": ["metric_type"],
            },
            "tags": ["monitoring", "metrics", "performance"],
            "annotations": {"category": "monitoring", "cost": "low"},
        },
    ]


def get_sample_config_json() -> str:
    """Get sample MCP configuration JSON."""
    config = {
        "mcpServers": {
            "company-api": {"url": "http://localhost:8080/mcp"},
            "storage-api": {"url": "http://localhost:8081/mcp", "oauth": True},
            "monitoring-api": {"url": "http://localhost:8082/mcp"},
            "local-tools": {
                "command": "python",
                "args": ["local_server.py"],
                "env": {"API_KEY": "${API_KEY}", "DEBUG": "true"},
            },
        }
    }
    return json.dumps(config, indent=2)


def get_search_queries() -> list[dict[str, Any]]:
    """Get sample search queries with expected results."""
    return [
        {
            "query": "create virtual machine",
            "expected_tools": ["create_instance"],
            "min_score": 0.1,
        },
        {
            "query": "delete resources",
            "expected_tools": ["delete_instance", "delete_volume"],
            "min_score": 0.05,
        },
        {
            "query": "storage volume management",
            "expected_tools": ["list_volumes", "create_volume", "delete_volume"],
            "min_score": 0.1,
        },
        {
            "query": "system performance monitoring",
            "expected_tools": ["get_metrics"],
            "min_score": 0.1,
        },
        {"query": "nonsense random query xyz", "expected_tools": [], "min_score": 0.0},
    ]


def get_sample_embeddings_data() -> list[tuple[str, list[float]]]:
    """Get sample text and embedding pairs for testing."""
    return [
        ("create virtual machine instance", [0.1, 0.2, 0.3] * 128),
        ("delete storage volume", [0.4, 0.5, 0.6] * 128),
        ("list monitoring metrics", [0.7, 0.8, 0.9] * 128),
        ("performance data retrieval", [0.2, 0.4, 0.6] * 128),
    ]


def get_hash_test_cases() -> list[dict[str, Any]]:
    """Get test cases for hash computation."""
    return [
        {
            "name": "create_instance",
            "description": "Create VM instance",
            "params": {"type": "object", "properties": {"name": {"type": "string"}}},
            "expected_hash_prefix": "a",  # Just check it's deterministic
        },
        {
            "name": "create_instance",
            "description": "Create VM instance",
            "params": {"type": "object", "properties": {"name": {"type": "string"}}},
            "expected_same_as_above": True,
        },
        {
            "name": "create_instance",
            "description": "Create VM instance DIFFERENT",  # Different description
            "params": {"type": "object", "properties": {"name": {"type": "string"}}},
            "expected_same_as_above": False,
        },
    ]

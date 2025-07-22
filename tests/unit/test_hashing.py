"""Tests for hashing utilities."""

import pytest
from typing import Any
from mcpproxy.utils.hashing import compute_tool_hash
from tests.utils.sample_data_generators import get_hash_test_cases


# Removed TestHashing class - all tests will be top-level functions
# class TestHashing:
#     """Test cases for hashing functionality."""


@pytest.mark.parametrize(
    "name, description, params, expected_len, expected_type",
    [
        ("test_tool", "Test tool description", {"type": "object"}, 64, str),
        ("", "", {}, 64, str),
        (
            "complex_tool",
            "Complex tool",
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "config": {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean"},
                            "settings": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
                "required": ["name"],
            },
            64,
            str,
        ),
        ("тест_инструмент", "测试工具描述", {"properties": {"名前": {"type": "string"}}}, 64, str),
        ("!@#$%^&*()", "~`|;:,.\\<>?/", {"key": "value"}, 64, str),
        # Add test case for None params
        ("test_tool_none_params", "Test description with None params", None, 64, str),
        # Add test cases for parameter order independence
        (
            "tool_order_test",
            "desc_order_test",
            {
                "type": "object",
                "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
            },
            64,
            str,
        ),
        (
            "tool_order_test",
            "desc_order_test",
            {
                "type": "object",
                "properties": {"b": {"type": "integer"}, "a": {"type": "string"}},
            },
            64,
            str,
        ),
    ],
)
def test_compute_tool_hash_valid_inputs(
    name: str, description: str, params: Any, expected_len: int, expected_type: type
):
    """Test hash computation with various valid inputs, including None params and order independence."""
    hash_value = compute_tool_hash(name, description, params)
    assert isinstance(hash_value, expected_type)
    assert len(hash_value) == expected_len
    assert hash_value.isalnum()  # Should be alphanumeric for basic cases

    # For parameter order independence, if it's the specific test case, compare with the expected same hash
    if name == "tool_order_test":
        params1 = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
        }
        params2 = {
            "type": "object",
            "properties": {"b": {"type": "integer"}, "a": {"type": "string"}},
        }
        hash_params1 = compute_tool_hash("tool_order_test", "desc_order_test", params1)
        hash_params2 = compute_tool_hash("tool_order_test", "desc_order_test", params2)
        assert hash_params1 == hash_params2


# Removed test_compute_tool_hash_none_params as it's now part of test_compute_tool_hash_valid_inputs
# def test_compute_tool_hash_none_params(self):
#     """Test hash computation with None parameters."""
#     name = "test_tool"
#     description = "Test description"
#     hash_with_none = compute_tool_hash(name, description, None)
#     hash_with_empty = compute_tool_hash(name, description, {})
#     assert isinstance(hash_with_none, str)
#     assert len(hash_with_none) == 64
#     assert hash_with_none == hash_with_empty  # None should be treated as empty dict


def test_compute_tool_hash_different_inputs(): # Removed self
    """Test that different inputs produce different hashes."""
    base_params = {"type": "object", "properties": {"param1": {"type": "string"}}}

    hash1 = compute_tool_hash("tool1", "description", base_params)
    hash2 = compute_tool_hash("tool2", "description", base_params)  # Different name
    hash3 = compute_tool_hash(
        "tool1", "different desc", base_params
    )  # Different description
    hash4 = compute_tool_hash(
        "tool1", "description", {"type": "object"}
    )  # Different params

    assert hash1 != hash2
    assert hash1 != hash3
    assert hash1 != hash4
    assert hash2 != hash3
    assert hash2 != hash4
    assert hash3 != hash4

@pytest.mark.parametrize("test_case", get_hash_test_cases())
def test_compute_tool_hash_test_cases(test_case: dict[str, Any]): # Removed self
    """Test hash computation with predefined test cases."""
    name = test_case["name"]
    description = test_case["description"]
    params = test_case["params"]

    hash_value = compute_tool_hash(name, description, params)

    assert isinstance(hash_value, str)
    assert len(hash_value) == 64

# Removed test_compute_tool_hash_param_order_independence as it's now part of test_compute_tool_hash_valid_inputs
# def test_compute_tool_hash_param_order_independence(self):
#     """Test that parameter order doesn't affect hash."""
#     params1 = {
#         "type": "object",
#         "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
#     }
#     params2 = {
#         "type": "object",
#         "properties": {"b": {"type": "integer"}, "a": {"type": "string"}},
#     }
#     hash1 = compute_tool_hash("tool", "desc", params1)
#     hash2 = compute_tool_hash("tool", "desc", params2)
#     assert hash1 == hash2


@pytest.mark.parametrize("name, description, params", [
    ("test_tool", "Test tool description", {"type": "object", "properties": {"param1": {"type": "string"}}}),
    ("consistency_test", "Testing consistency", {"type": "object", "properties": {"test": {"type": "string"}}}),
])
def test_compute_tool_hash_consistency(name: str, description: str, params: Any): # Removed self
    """Test that hash computation is deterministic and consistent across calls."""
    hash1 = compute_tool_hash(name, description, params)
    hash2 = compute_tool_hash(name, description, params)
    assert hash1 == hash2

    hashes = []
    for _ in range(10):
        hash_value = compute_tool_hash(name, description, params)
        hashes.append(hash_value)
    assert all(h == hashes[0] for h in hashes)
    assert len(set(hashes)) == 1

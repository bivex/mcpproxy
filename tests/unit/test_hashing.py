"""Tests for hashing utilities."""

from typing import Any

import pytest

from mcpproxy.utils.hashing import compute_tool_hash
from tests.fixtures.data import get_hash_test_cases


class TestHashing:
    """Test cases for hashing functionality."""

    def test_compute_tool_hash_basic(self):
        """Test basic hash computation."""
        name = "test_tool"
        description = "Test tool description"
        params = {"type": "object", "properties": {"param1": {"type": "string"}}}

        hash_value = compute_tool_hash(name, description, params)

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # SHA-256 produces 64-character hex string
        assert hash_value.isalnum()  # Should be alphanumeric

    def test_compute_tool_hash_deterministic(self):
        """Test that hash computation is deterministic."""
        name = "test_tool"
        description = "Test tool description"
        params = {"type": "object", "properties": {"param1": {"type": "string"}}}

        hash1 = compute_tool_hash(name, description, params)
        hash2 = compute_tool_hash(name, description, params)

        assert hash1 == hash2

    def test_compute_tool_hash_different_inputs(self):
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

    def test_compute_tool_hash_none_params(self):
        """Test hash computation with None parameters."""
        name = "test_tool"
        description = "Test description"

        hash_with_none = compute_tool_hash(name, description, None)
        hash_with_empty = compute_tool_hash(name, description, {})

        assert isinstance(hash_with_none, str)
        assert len(hash_with_none) == 64
        assert hash_with_none == hash_with_empty  # None should be treated as empty dict

    def test_compute_tool_hash_empty_strings(self):
        """Test hash computation with empty strings."""
        hash_value = compute_tool_hash("", "", {})

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_compute_tool_hash_complex_params(self):
        """Test hash computation with complex parameter structures."""
        complex_params = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Tool name"},
                "config": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "settings": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "required": ["name"],
        }

        hash_value = compute_tool_hash("complex_tool", "Complex tool", complex_params)

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_compute_tool_hash_unicode_characters(self):
        """Test hash computation with unicode characters."""
        name = "тест_инструмент"  # Russian
        description = "测试工具描述"  # Chinese
        params = {
            "type": "object",
            "properties": {"名前": {"type": "string"}},
        }  # Japanese

        hash_value = compute_tool_hash(name, description, params)

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_compute_tool_hash_param_order_independence(self):
        """Test that parameter order doesn't affect hash."""
        params1 = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
        }

        params2 = {
            "type": "object",
            "properties": {"b": {"type": "integer"}, "a": {"type": "string"}},
        }

        hash1 = compute_tool_hash("tool", "desc", params1)
        hash2 = compute_tool_hash("tool", "desc", params2)

        # Should be the same due to sort_keys=True in JSON serialization
        assert hash1 == hash2

    @pytest.mark.parametrize("test_case", get_hash_test_cases())
    def test_compute_tool_hash_test_cases(self, test_case: dict[str, Any]):
        """Test hash computation with predefined test cases."""
        name = test_case["name"]
        description = test_case["description"]
        params = test_case["params"]

        hash_value = compute_tool_hash(name, description, params)

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_compute_tool_hash_consistency_across_calls(self):
        """Test that multiple calls with same parameters produce same hash."""
        name = "consistency_test"
        description = "Testing consistency"
        params = {"type": "object", "properties": {"test": {"type": "string"}}}

        hashes = []
        for _ in range(10):
            hash_value = compute_tool_hash(name, description, params)
            hashes.append(hash_value)

        # All hashes should be identical
        assert all(h == hashes[0] for h in hashes)
        assert len(set(hashes)) == 1  # Only one unique hash

    def test_compute_tool_hash_special_characters(self):
        """Test hash computation with special characters in strings."""
        special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?/~`\"'\\"

        hash_value = compute_tool_hash(
            f"tool_{special_chars}",
            f"Description with {special_chars}",
            {"description": f"Parameter with {special_chars}"},
        )

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

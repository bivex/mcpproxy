"""SHA-256 hashing utilities for tool metadata."""

import hashlib
import json
from typing import Any


def compute_tool_hash(
    name: str, description: str, params: dict[str, Any] | None = None
) -> str:
    """Compute SHA-256 hash for tool metadata.

    Args:
        name: Tool name
        description: Tool description
        params: Tool parameters schema

    Returns:
        SHA-256 hash string
    """
    params_json = json.dumps(params or {}, sort_keys=True)
    content = f"{name}||{description}||{params_json}"
    return hashlib.sha256(content.encode()).hexdigest()

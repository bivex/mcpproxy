import re
import os
from mcpproxy.logging import get_logger

logger = get_logger()

def sanitize_tool_name(server_name: str, tool_name: str, tool_name_limit: int = 60) -> str:
    """Sanitize tool name to comply with Google Gemini API requirements."""

    combined = _clean_and_combine_names(server_name, tool_name)
    combined = _ensure_starts_with_valid_char(combined)
    combined = _truncate_name(combined, tool_name_limit)
    combined = _final_validation_and_fallback(combined, tool_name_limit)

    # Debug logging for validation
    if (
        len(combined) > tool_name_limit
        or not re.match(r"^[a-z_][a-z0-9_]*$", combined)
        or combined.endswith("_")
    ):
        logger.warning(
            f"Tool name may still be invalid: '{server_name}' + '{tool_name}' -> '{combined}' (len={len(combined)})"
        )

    return combined

def _clean_and_combine_names(server_name: str, tool_name: str) -> str:
    # First sanitize individual parts - be more aggressive for Google API
    # Convert to lowercase and replace anything that isn't alphanumeric with underscore
    server_clean = re.sub(r"[^a-z0-9_]", "_", server_name.lower())
    tool_clean = re.sub(r"[^a-z0-9_]", "_", tool_name.lower())

    # Remove consecutive underscores
    server_clean = re.sub(r"_+", "_", server_clean)
    tool_clean = re.sub(r"_+", "_", tool_clean)

    # Remove leading/trailing underscores from parts
    server_clean = server_clean.strip("_")
    tool_clean = tool_clean.strip("_")

    # If parts are empty after cleaning, use defaults
    if not server_clean:
        server_clean = "server"
    if not tool_clean:
        tool_clean = "tool"

    # Combine server and tool name
    return f"{server_clean}_{tool_clean}"

def _ensure_starts_with_valid_char(name: str) -> str:
    if not re.match(r"^[a-z_]", name):
        return f"tool_{name}"
    return name

def _truncate_name(name: str, max_length: int) -> str:
    if len(name) > max_length:
        # Try to keep server prefix if possible
        if "_" in name:
            parts = name.split("_", 1)
            server_part = parts[0]
            tool_part = parts[1]

            # Reserve space for server part + underscore
            available_space = max_length - len(server_part) - 1
            if available_space > 3:  # Keep at least 3 chars of tool name
                truncated = f"{server_part}_{tool_part[:available_space]}"
            else:
                # Not enough space for meaningful server prefix
                truncated = name[:max_length]
        else:
            truncated = name[:max_length]

        # Ensure doesn't end with underscore
        truncated = truncated.rstrip("_")

        # If we stripped all chars, add fallback
        if not truncated:
            truncated = "tool"

        return truncated
    return name

def _final_validation_and_fallback(name: str, max_length: int) -> str:
    # Final validation - ensure it still starts correctly and is valid
    if not re.match(r"^[a-z_]", name):
        name = f"tool_{name}"
        # Re-truncate if needed
        if len(name) > max_length:
            name = name[:max_length].rstrip("_")

    # Final fallback if somehow we ended up empty
    if not name:
        name = "tool"

    return name 

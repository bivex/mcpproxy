import re
import os
from mcpproxy.logging import get_logger

logger = get_logger()

def sanitize_tool_name(server_name: str, tool_name: str, tool_name_limit: int = 60) -> str:
    """Sanitize tool name to comply with Google Gemini API requirements.

    Google Gemini API Requirements (more strict than general MCP):
    - Must start with letter or underscore
    - Only lowercase letters (a-z), numbers (0-9), underscores (_)
    - Maximum length configurable via MCPPROXY_TOOL_NAME_LIMIT (default 60)
    - No dots or dashes (unlike general MCP spec)

    Args:
        server_name: Name of the server
        tool_name: Original tool name
        tool_name_limit: Maximum length for the tool name

    Returns:
        Sanitized tool name that complies with Google Gemini API
    """

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
    combined = f"{server_clean}_{tool_clean}"

    # Ensure starts with letter or underscore (Google requirement)
    if not re.match(r"^[a-z_]", combined):
        combined = f"tool_{combined}"

    # Truncate to configured limit if needed
    max_length = tool_name_limit
    if len(combined) > max_length:
        # Try to keep server prefix if possible
        if "_" in combined:
            parts = combined.split("_", 1)
            server_part = parts[0]
            tool_part = parts[1]

            # Reserve space for server part + underscore
            available_space = max_length - len(server_part) - 1
            if available_space > 3:  # Keep at least 3 chars of tool name
                truncated = f"{server_part}_{tool_part[:available_space]}"
            else:
                # Not enough space for meaningful server prefix
                truncated = combined[:max_length]
        else:
            truncated = combined[:max_length]

        # Ensure doesn't end with underscore
        truncated = truncated.rstrip("_")

        # If we stripped all chars, add fallback
        if not truncated:
            truncated = "tool"

        combined = truncated

    # Final validation - ensure it still starts correctly and is valid
    if not re.match(r"^[a-z_]", combined):
        combined = f"tool_{combined}"
        # Re-truncate if needed
        if len(combined) > max_length:
            combined = combined[:max_length].rstrip("_")

    # Final fallback if somehow we ended up empty
    if not combined:
        combined = "tool"

    # Debug logging for validation
    if (
        len(combined) > max_length
        or not re.match(r"^[a-z_][a-z0-9_]*$", combined)
        or combined.endswith("_")
    ):
        logger.warning(
            f"Tool name may still be invalid: '{server_name}' + '{tool_name}' -> '{combined}' (len={len(combined)})"
        )

    return combined 

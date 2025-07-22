def truncate_output(output: str, truncate_len: int | None) -> str:
    """Truncate output if it exceeds the configured length limit.

    Args:
        output: The original output string
        truncate_len: The maximum length for the output string. If None or 0, no truncation occurs.

    Returns:
        Truncated output with placeholder if needed, or original if within limit
    """
    if not truncate_len or len(output) <= truncate_len:
        return output

    # Calculate how many chars to show at start and end
    last_chars = 50
    first_chars = (
        truncate_len
        - last_chars
        - len(" <truncated by smart mcp proxy> ")
    )

    if first_chars <= 0:
        # If truncate length is too small, just show beginning
        return (
            output[:truncate_len] + " <truncated by smart mcp proxy>"
        )

    truncated = (
        output[:first_chars]
        + " <truncated by smart mcp proxy> "
        + output[-last_chars:]
    )

    return truncated 

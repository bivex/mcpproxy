import time
from mcpproxy.logging import get_logger

logger = get_logger()

def calculate_tool_weight(score: float, added_timestamp: float) -> float:
    """Calculate weighted score for tool eviction based on score and freshness.

    Args:
        score: Original search score (0.0 to 1.0)
        added_timestamp: Timestamp when tool was added to pool

    Returns:
        Weighted score (higher is better, less likely to be evicted)
    """
    current_time = time.time()
    age_seconds = current_time - added_timestamp

    # Normalize age (0 = fresh, 1 = old)
    # Tools older than 30 minutes get maximum age penalty
    max_age_seconds = 30 * 60  # 30 minutes
    age_normalized = min(1.0, age_seconds / max_age_seconds)

    # Weighted formula: 70% score, 30% freshness
    score_weight = 0.7
    freshness_weight = 0.3
    freshness_score = 1.0 - age_normalized

    weighted_score = (score * score_weight) + (freshness_score * freshness_weight)
    return weighted_score 

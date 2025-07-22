import time
from mcpproxy.logging import get_logger
from mcpproxy.models.constants import TOOL_SCORE_WEIGHT, TOOL_FRESHNESS_WEIGHT, MAX_TOOL_AGE_MINUTES, SECONDS_IN_MINUTE

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
    max_age_seconds = MAX_TOOL_AGE_MINUTES * SECONDS_IN_MINUTE
    age_normalized = min(1.0, age_seconds / max_age_seconds)

    # Weighted formula: 70% score, 30% freshness
    score_weight = TOOL_SCORE_WEIGHT
    freshness_weight = TOOL_FRESHNESS_WEIGHT
    freshness_score = 1.0 - age_normalized

    weighted_score = (score * score_weight) + (freshness_score * freshness_weight)
    return weighted_score 

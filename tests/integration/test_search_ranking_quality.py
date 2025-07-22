"""Integration tests for search ranking quality."""

import pytest
import numpy as np
from mcpproxy.models.schemas import ToolData


class TestSearchRankingQuality:
    """Test that search ranking works correctly."""

    @pytest.mark.asyncio
    async def test_search_ranking_quality(self, temp_indexer_facade):
        """Test that search ranking works correctly."""
        # Index tools with varying relevance to test query
        tools = [
            (
                "perfect_match",
                "Create virtual machine instance",
                "server1",
                ["vm", "create"],
            ),
            ("good_match", "Create VM in cloud", "server1", ["vm", "cloud"]),
            ("partial_match", "Virtual environment setup", "server1", ["virtual"]),
            ("weak_match", "Machine learning model", "server1", ["machine"]),
            ("no_match", "Delete storage volume", "server1", ["storage", "delete"]),
        ]

        for name, desc, server, tags in tools:
            tool_data = ToolData(
                name=name,
                description=desc,
                server_name=server,
                tags=tags
            )
            await temp_indexer_facade.index_tool(tool_data)

        # Search for "create virtual machine"
        results = await temp_indexer_facade.search_tools("create virtual machine", k=5)

        MIN_EXPECTED_RESULTS = 3
        assert len(results) >= MIN_EXPECTED_RESULTS

        # Verify ranking: perfect_match should be first, no_match should be last
        result_names = [r.tool.name for r in results]
        result_scores = [r.score for r in results]

        # Perfect match should have highest score
        perfect_idx = result_names.index("perfect_match")
        assert perfect_idx == 0 or result_scores[perfect_idx] == max(result_scores)

        # No match should have lowest score among results
        if "no_match" in result_names:
            no_match_idx = result_names.index("no_match")
            assert result_scores[no_match_idx] == min(result_scores)

        # Scores should generally decrease (allowing for ties)
        assert result_scores == sorted(result_scores, reverse=True) 

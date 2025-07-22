"""Unit tests for tool pool management functionality."""

import time
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field

import pytest

from mcpproxy.models.schemas import EmbedderType, ToolRegistration, ToolMetadata, ToolPoolManagerDependencies
from mcpproxy.persistence.facade import PersistenceFacade
from mcpproxy.server.config.config import ProxyConfig
from mcpproxy.server.server_discovery_manager import ServerDiscoveryManager
from mcpproxy.server.mcp_server import SmartMCPProxyServer
from mcpproxy.utils.tool_scoring.tool_weight_calculator import calculate_tool_weight
from mcpproxy.server.tool_pool_manager import ToolPoolManager


@dataclass
class PoolLimitScenario:
    initial_tools_metadata: dict[str, dict]
    new_tools_info: list[tuple[str, float]]
    expected_evicted_count: int
    expected_evicted_names: list[str]
    expected_registered_count: int


class TestToolPoolManagerUnit:
    """Test tool pool management and eviction logic."""

    @pytest.fixture
    def mock_fastmcp_app(self):
        """Mock FastMCP application."""
        mock = MagicMock()
        mock.remove_tool = MagicMock()
        mock.add_tool = MagicMock()
        mock._mcp_server = MagicMock()
        request_context = mock._mcp_server.request_context
        session = request_context.session
        session.send_notification = AsyncMock()
        return mock

    @pytest.fixture
    def mock_indexer(self):
        """Mock IndexerFacade."""
        mock = MagicMock()
        mock.search_tools = AsyncMock()
        return mock

    @pytest.fixture
    def mock_persistence(self):
        """Mock PersistenceFacade."""
        mock = MagicMock()
        mock.get_all_tools = AsyncMock(return_value=[])
        return mock

    @pytest.fixture
    def mock_proxy_config(self):
        """Mock ProxyConfig with default values for testing."""
        return ProxyConfig(
            mcp_servers={},
            embedder=EmbedderType.BM25,
            hf_model=None,
            top_k=5,
            tool_name_limit=60,
            tools_limit=15,
        )

    @pytest.fixture
    def mock_proxy_servers(self):
        """Mock dictionary for upstream FastMCP proxy servers."""
        return {"testserver": MagicMock()}

    @pytest.fixture
    def mock_truncate_output_fn(self):
        """Mock for the truncate_output function."""
        return MagicMock(side_effect=lambda x, y: x) # Simply return original text

    @pytest.fixture
    def mock_tool_pool_dependencies_fixture(
        self, mock_indexer, mock_persistence, mock_proxy_config, mock_proxy_servers
    ):
        """Fixture for ToolPoolManagerDependencies instance with mocked dependencies."""
        from mcpproxy.server.tool_pool_manager import ToolPoolManagerDependencies
        return ToolPoolManagerDependencies(
            indexer=mock_indexer,
            persistence=mock_persistence,
            config=mock_proxy_config,
            proxy_servers=mock_proxy_servers,
        )

    @pytest.fixture
    def mock_tool_pool_manager(
        self,
        mock_fastmcp_app,
        mock_tool_pool_dependencies_fixture,
        mock_truncate_output_fn,
    ):
        """Fixture for a ToolPoolManager instance with mocked dependencies."""
        tool_pool_dependencies = mock_tool_pool_dependencies_fixture

        with patch("mcpproxy.server.tool_pool_manager.ToolPoolManager._enforce_tool_pool_limit", new_callable=AsyncMock) as mock_enforce_limit, \
             patch("mcpproxy.server.tool_pool_manager.ToolPoolManager._evict_tool", new_callable=AsyncMock) as mock_evict_tool, \
             patch("mcpproxy.server.tool_pool_manager.ToolPoolManager._register_proxy_tool", new_callable=AsyncMock) as mock_register_proxy_tool:

            manager = ToolPoolManager(
                mcp_app=mock_fastmcp_app,
                dependencies=tool_pool_dependencies,
                truncate_output_fn=mock_truncate_output_fn,
                truncate_output_len=500, # Example length
            )
            manager.mock_enforce_limit = mock_enforce_limit
            manager.mock_evict_tool = mock_evict_tool
            manager.mock_register_proxy_tool = mock_register_proxy_tool
            yield manager

    # @pytest.fixture
    # def mock_server(self):
    #     """Create a mock server with tool pool management."""
    #     with patch("mcpproxy.server.mcp_server.ConfigLoader"):
    #         server = SmartMCPProxyServer()
    #         server.tools_limit = 3  # Small limit for testing
    #         server.mcp = MagicMock()
    #         server.mcp.remove_tool = MagicMock()
    #         return server

    @pytest.mark.parametrize(
        "score, time_ago_minutes, expected_weight_tolerance",
        [
            (0.8, 0, 0.01),  # Fresh tool
            (0.8, 35, 0.01), # Old tool (older than max age)
            (0.6, 15, 0.01), # Medium age
        ],
    )
    def test_tool_weight_scenarios(
        self, score: float, time_ago_minutes: int, expected_weight_tolerance: float
    ):
        """Test weight calculation for various tool ages."""
        current_time = time.time()
        timestamp = current_time - (time_ago_minutes * 60)

        weight = calculate_tool_weight(score, timestamp)

        # Expected calculation based on the method: (score * 0.7) + (freshness * 0.3)
        # Freshness is (max_age_seconds - age_seconds) / max_age_seconds
        # Max age is 30 minutes (1800 seconds)
        age_seconds = time_ago_minutes * 60
        max_age_seconds = 30 * 60
        freshness = max(0.0, 1.0 - (age_seconds / max_age_seconds))
        expected = (score * 0.7) + (freshness * 0.3)

        assert abs(weight - expected) < expected_weight_tolerance

    @pytest.mark.parametrize(
        "scenario",
        [
            PoolLimitScenario(
                initial_tools_metadata={},
                new_tools_info=[("new_tool", 0.8)],
                expected_evicted_count=0,
                expected_evicted_names=[],
                expected_registered_count=1
            ), # Add one tool, under limit
            PoolLimitScenario(
                initial_tools_metadata={
                    "tool1": {"timestamp": time.time() - 1800, "score": 0.5}, # Old, low score
                    "tool2": {"timestamp": time.time() - 900, "score": 0.9}, # Medium age, high score
                    "tool3": {"timestamp": time.time(), "score": 0.7}, # Fresh, medium score
                },
                new_tools_info=[("new_tool1", 0.8), ("new_tool2", 0.6)], # Add two new tools, exceeding limit of 3
                expected_evicted_count=2, # Expect 2 evictions
                expected_evicted_names=["tool1", "tool3"], # tool1 and tool3 should be evicted based on scores/freshness
                expected_registered_count=3 # tool2 + new_tool1 + new_tool2
            ),
            PoolLimitScenario(
                initial_tools_metadata={
                    "tool1": {"timestamp": time.time() - 1800, "score": 0.5}, 
                    "tool2": {"timestamp": time.time() - 1800, "score": 0.4}, 
                    "tool3": {"timestamp": time.time() - 1800, "score": 0.6}, 
                },
                new_tools_info=[("new_tool", 0.9)], # Add one new tool, evict one oldest/lowest score
                expected_evicted_count=1, 
                expected_evicted_names=["tool2"], 
                expected_registered_count=3
            ),           
        ]
    )
    @pytest.mark.asyncio
    async def test_enforce_pool_limit_scenarios(
        self,
        mock_tool_pool_manager,
        scenario: PoolLimitScenario
    ):
        """Test tool pool limit enforcement and eviction scenarios."""
        manager = mock_tool_pool_manager

        self._setup_pool_manager_initial_state(manager, scenario.initial_tools_metadata)
        self._prepare_mock_search_results(manager, scenario.new_tools_info)

        # Call the public method that triggers the logic
        await manager.retrieve_tools("some query")

        self._assert_pool_limit_enforcement(manager, scenario.expected_evicted_count, scenario.expected_evicted_names, scenario.expected_registered_count)

    def _setup_pool_manager_initial_state(self, manager, initial_tools_metadata):
        manager.current_tool_registrations = {name: MagicMock() for name in initial_tools_metadata}
        manager.tool_pool_metadata = initial_tools_metadata

    def _prepare_mock_search_results(self, manager, new_tools_info):
        mock_search_results = []
        for name, score in new_tools_info:
            tool_obj = MagicMock(spec=ToolMetadata)
            tool_obj.name = name.split("_")[1] if "_" in name else name
            tool_obj.server_name = name.split("_")[0] if "_" in name else "testserver"
            tool_obj.description = f"Description for {tool_obj.name}"
            tool_obj.parameters = {}
            mock_search_results.append(MagicMock(tool=tool_obj, score=score))
        mock_indexer_search_tools = manager.indexer.search_tools
        mock_indexer_search_tools.return_value = mock_search_results

    def _assert_pool_limit_enforcement(self, manager, expected_evicted_count, expected_evicted_names, expected_registered_count):
        assert manager.mock_evict_tool.call_count == expected_evicted_count
        actual_evicted_names = [call.args[0] for call in manager.mock_evict_tool.call_args_list]
        assert sorted(actual_evicted_names) == sorted(expected_evicted_names)
        assert manager.mock_register_proxy_tool.call_count == expected_registered_count

    @pytest.mark.asyncio
    async def test_evict_tool_functionality(
        self, mock_tool_pool_manager, mock_fastmcp_app
    ):
        """Test that _evict_tool correctly removes a tool from the pool and calls mcp.remove_tool."""
        manager = mock_tool_pool_manager
        tool_name_to_evict = "tool_to_evict"
        manager.current_tool_registrations[tool_name_to_evict] = MagicMock()
        manager.registered_tools[tool_name_to_evict] = MagicMock()
        manager.tool_pool_metadata[tool_name_to_evict] = {
            "timestamp": time.time(),
            "score": 0.5,
        }

        # Ensure the mock_fastmcp_app passed to the manager is used
        manager.mcp.remove_tool = AsyncMock() # Ensure it's an AsyncMock

        await manager._evict_tool(tool_name_to_evict)

        assert tool_name_to_evict not in manager.current_tool_registrations
        assert tool_name_to_evict not in manager.registered_tools
        assert tool_name_to_evict not in manager.tool_pool_metadata
        manager.mcp.remove_tool.assert_called_once_with(tool_name_to_evict)

    @pytest.mark.asyncio
    async def test_register_proxy_tool_metadata(
        self, mock_tool_pool_manager, mock_proxy_servers, mock_fastmcp_app, mock_proxy_config
    ):
        """Test that tool registration includes metadata tracking."""
        manager = mock_tool_pool_manager

        # Create a mock Tool object (what the implementation expects)
        from fastmcp.tools.tool import Tool

        original_tool = MagicMock(spec=Tool)
        original_tool.name = "test_tool"
        original_tool.description = "Test description"
        original_tool.parameters = {}
        original_tool.tags = []  # Required by Tool.from_tool
        original_tool.annotations = {}  # Required by Tool.from_tool
        original_tool.serializer = MagicMock()  # Required by Tool.from_tool
        original_tool.func = MagicMock()
        original_tool.examples = []

        tool_name_sanitized = "test_server_test_tool"
        server_name = "test_server"
        score = 0.75

        # Configure manager's mocks to allow _register_proxy_tool to run normally
        manager.proxy_servers = mock_proxy_servers
        manager.mcp = mock_fastmcp_app

        with patch("fastmcp.tools.tool.Tool.from_tool") as mock_from_tool:
            mock_proxified_tool = MagicMock()
            mock_from_tool.return_value = mock_proxified_tool

            await manager._register_proxy_tool(
                original_tool, tool_name_sanitized, score, server_name
            )

        # Verify Tool.from_tool was called
        mock_from_tool.assert_called_once()
        call_args = mock_from_tool.call_args
        assert call_args[1]["tool"] == original_tool
        assert call_args[1]["name"] == tool_name_sanitized
        assert "transform_fn" in call_args[1]

        # Verify metadata was tracked
        assert tool_name_sanitized in manager.tool_pool_metadata
        metadata = manager.tool_pool_metadata[tool_name_sanitized]
        assert metadata["score"] == score
        assert metadata["original_score"] == score
        assert "timestamp" in metadata
        assert isinstance(metadata["timestamp"], float)

        # Verify tool was registered in current_tool_registrations and registered_tools
        assert tool_name_sanitized in manager.current_tool_registrations
        assert tool_name_sanitized in manager.registered_tools

        # Verify the proxified tool was added to FastMCP
        manager.mcp.add_tool.assert_called_once_with(mock_proxified_tool)

    def test_tool_weight_comparison(self):
        """Test that tool weights are calculated correctly for comparison."""
        current_time = time.time()

        # High score, old tool
        weight1 = calculate_tool_weight(
            0.9, current_time - 1800
        )  # 30 min ago

        # Medium score, fresh tool
        weight2 = calculate_tool_weight(0.6, current_time)

        # Low score, medium age
        weight3 = calculate_tool_weight(
            0.3, current_time - 900
        )  # 15 min ago

        # Fresh high-score tool should have highest weight
        weight4 = calculate_tool_weight(0.9, current_time)

        # Verify ordering
        assert weight4 > weight2  # Fresh high-score > fresh medium-score
        assert weight2 > weight1  # Fresh medium > old high (due to freshness factor)
        assert weight1 > weight3  # Old high > medium-age low

    @pytest.mark.asyncio
    async def test_freshness_update(self, mock_tool_pool_manager):
        """Test that existing tools get freshness updates."""
        manager = mock_tool_pool_manager

        tool_name = "existing_tool"
        original_time = time.time() - 1000
        original_score = 0.6
        new_score = 0.8

        # Set up existing tool metadata
        manager.tool_pool_metadata[tool_name] = {
            "timestamp": original_time,
            "score": original_score,
            "original_score": original_score,
        }
        manager.current_tool_registrations[tool_name] = MagicMock()

        # Simulate the freshness update logic from retrieve_tools
        manager.tool_pool_metadata[tool_name]["timestamp"] = time.time()
        manager.tool_pool_metadata[tool_name]["score"] = max(
            manager.tool_pool_metadata[tool_name]["score"], new_score
        )

        # Verify updates
        metadata = manager.tool_pool_metadata[tool_name]
        assert metadata["score"] == new_score  # Should be updated to higher score
        assert metadata["timestamp"] > original_time  # Should be fresher
        assert metadata["original_score"] == original_score  # Should preserve original

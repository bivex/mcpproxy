import asyncio
import tempfile
from pathlib import Path
from typing import AsyncGenerator

import pytest

from mcpproxy.persistence.db import DatabaseManager


@pytest.fixture
async def temp_db_path():
    """Temporary database path for testing."""
    import os

    temp_dir = tempfile.gettempdir()
    db_path = os.path.join(temp_dir, f"test_db_{os.getpid()}_{id(object())}.db")
    yield db_path
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
async def temp_faiss_path():
    """Temporary Faiss index path for testing."""
    import os

    temp_dir = tempfile.gettempdir()
    faiss_path = os.path.join(
        temp_dir, f"test_faiss_{os.getpid()}_{id(object())}.faiss"
    )
    yield faiss_path
    Path(faiss_path).unlink(missing_ok=True)


@pytest.fixture
async def in_memory_db() -> AsyncGenerator[DatabaseManager, None]:
    """In-memory SQLite database for testing."""
    db = DatabaseManager(":memory:")
    yield db 

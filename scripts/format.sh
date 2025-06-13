#!/bin/bash
set -e

echo "🔧 Formatting Python code with ruff..."

# Format Python code using ruff (faster alternative to black)
uv run ruff format mcpproxy/ tests/

# Also run import sorting
uv run ruff check --select I --fix mcpproxy/ tests/

echo "✅ Code formatting completed!" 
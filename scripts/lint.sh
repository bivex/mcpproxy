#!/bin/bash
set -e

echo "🔍 Linting Python code..."

# Run ruff linting (combines flake8, isort, and many other linters)
echo "Running ruff checks..."
uv run ruff check mcpproxy/ tests/

# Run basic type checking with mypy (only syntax errors)
echo "Running mypy type checking (basic)..."
uv run mypy mcpproxy/ \
    --ignore-missing-imports \
    --allow-untyped-defs \
    --allow-incomplete-defs \
    --allow-untyped-calls \
    --no-warn-return-any \
    --warn-unreachable || echo "⚠️  mypy found type issues (non-blocking)"

echo "✅ Linting completed!" 
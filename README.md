# Smart MCP Proxy

[![PyPI version](https://badge.fury.io/py/smart-mcp-proxy.svg)](https://badge.fury.io/py/smart-mcp-proxy)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A federating gateway that sits between AI agents and multiple Model Context Protocol (MCP) servers, providing intelligent tool discovery and dynamic registration.

🌐 **Website**: [mcpproxy.app](https://mcpproxy.app)
📦 **PyPI**: [pypi.org/project/smart-mcp-proxy](https://pypi.org/project/smart-mcp-proxy/)
🔗 **GitHub**: [github.com/Dumbris/mcpproxy](https://github.com/Dumbris/mcpproxy)

## Features

- **Dynamic Tool Discovery**: Automatically discovers tools from multiple MCP servers
- **Intelligent Search**: Uses configurable embedding backends (BM25, HuggingFace, OpenAI) to find relevant tools
- **One-Click Tool Access**: Single `retrieve_tools` function that searches, registers, and exposes the top 5 most relevant tools
- **FastMCP Integration**: Built on FastMCP v2 for robust server runtime and client capabilities
- **Persistent Indexing**: SQLite + Faiss storage for fast tool lookup and change detection
- **MCP Spec Compliant**: Emits proper `notifications/tools/list_changed` events
- **Flexible Dependencies**: Optional dependencies for different backends to minimize install size

## Architecture

```
┌─────────────────┐    ┌─────────────────────────────────┐    ┌─────────────────┐
│   AI Agent      │───▶│     Smart MCP Proxy             │───▶│  MCP Servers    │
│                 │    │                                 │    │                 │
│ retrieve_tools()│    │ ┌─────────────┐ ┌─────────────┐ │    │ • company-prod    │
│                 │    │ │   Indexer   │ │ Persistence │ │    │ • company-docs    │
│ tool_1()        │◀───│ │   (BM25/    │ │  (SQLite +  │ │    │ • oauth-server  │
│ tool_2()        │    │ │ HF/OpenAI)  │ │   Faiss)    │ │    │ • ...           │
│ ...             │    │ └─────────────┘ └─────────────┘ │    │                 │
└─────────────────┘    └─────────────────────────────────┘    └─────────────────┘
```

## Quick Start

### 1. Installation

Choose your installation based on the embedding backend you want to use:

```bash
# Basic installation with BM25 (lexical search, no ML dependencies)
pip install smart-mcp-proxy

# Or with specific backends:
pip install smart-mcp-proxy[bm25]         # Explicit BM25 (same as basic)
pip install smart-mcp-proxy[huggingface]  # HuggingFace + vector search
pip install smart-mcp-proxy[openai]       # OpenAI embeddings + vector search
pip install smart-mcp-proxy[all]          # All backends available

# Development install
git clone https://github.com/Dumbris/mcpproxy.git
cd mcpproxy
pip install -e .[all]
```

The proxy will automatically check for required dependencies and provide helpful error messages if you try to use a backend without the required packages installed.

### 2. Configuration

Set environment variables:

```bash
export SP_EMBEDDER=BM25  # or HF, OPENAI
export SP_HF_MODEL=sentence-transformers/all-MiniLM-L6-v2  # if using HF
export OPENAI_API_KEY=your_key  # if using OpenAI
```

### 3. Create Configuration

Run the proxy to create a sample config:

```bash
python main.py
```

This creates `mcp_config.json`:

```json
{
  "mcpServers": {
    "company-mcp-server-http-prod": {
      "url": "http://localhost:8081/mcp"
    },
    "company-docs": {
      "url": "http://localhost:8000/sse"
    },
    "company-mcp-server-with-oauth": {
      "url": "http://localhost:8080/mcp",
      "oauth": true
    },
    "company-mcp-server-prod": {
      "command": "uvx",
      "args": [
        "--from",
        "mcp-company-python@git+https://github.com/algis-dumbris/mcp-company.git",
        "company-mcp-server"
      ],
      "env": {
        "COMPANY_TOKEN": "${COMPANY_TOKEN}",
        "PORT": "9090"
      }
    }
  }
}
```

### 4. Start the Proxy

```bash
# Using the installed script
smart-mcp-proxy

# Or directly with Python
python main.py
```

The proxy will:
1. Discover tools from all configured MCP servers
2. Index them using the chosen embedding backend
3. Start FastMCP server on `localhost:8000`

## Usage

### For AI Agents

Connect to the proxy as a standard MCP server. Use the `retrieve_tools` function:

```python
# Agent calls this to discover tools
result = await client.call_tool("retrieve_tools", {"query": "create cloud instance"})

# Proxy automatically registers relevant tools, now available:
instance = await client.call_tool("company-prod_create_instance", {
    "name": "my-instance",
    "flavor": "standard-2-4"
})
```

### Programmatic Usage

```python
from mcpproxy import SmartMCPProxyServer

proxy = SmartMCPProxyServer("config.json")
await proxy.start()

# Use the indexer directly
results = await proxy.indexer.search_tools("delete volume", k=3)
for result in results:
    print(f"{result.tool.name}: {result.score}")
```

## Project Structure

```
mcpproxy/
├── models/
│   └── schemas.py           # Pydantic models and schemas
├── persistence/
│   ├── db.py               # SQLite operations
│   ├── faiss_store.py      # Faiss vector storage
│   └── facade.py           # Unified persistence interface
├── indexer/
│   ├── base.py             # Base embedder interface
│   ├── bm25.py             # BM25 implementation
│   ├── huggingface.py      # HuggingFace embeddings
│   ├── openai.py           # OpenAI embeddings
│   └── facade.py           # Search and indexing interface
├── server/
│   ├── config.py           # Configuration management
│   └── mcp_server.py       # FastMCP server implementation
└── utils/
    └── hashing.py          # SHA-256 utilities for change detection
```

## Environment Variables

| Variable         | Values                        | Default | Description |
|------------------|-------------------------------|---------|-------------|
| `SP_EMBEDDER`    | `BM25`, `HF`, `OPENAI`        | `BM25`  | Embedding backend |
| `SP_HF_MODEL`    | HuggingFace model name        | `sentence-transformers/all-MiniLM-L6-v2` | HF model |
| `SP_TOP_K`       | Integer                       | `5`     | Number of tools to register |
| `OPENAI_API_KEY` | Your OpenAI API key           | -       | Required for OpenAI embedder |
| `MCP_CONFIG_PATH`| Path to config file           | `mcp_config.json` | Config file location |
| `PROXY_HOST`     | Host to bind                  | `localhost` | Server host |
| `PROXY_PORT`     | Port to bind                  | `8000`  | Server port |

## Development

### Adding New Embedders

1. Inherit from `BaseEmbedder`
2. Implement `embed_text`, `embed_batch`, `get_dimension`
3. Add to `IndexerFacade._create_embedder`

```python
class CustomEmbedder(BaseEmbedder):
    async def embed_text(self, text: str) -> np.ndarray:
        # Your implementation
        pass
```

### Adding New Server Types

Extend `SmartMCPProxyServer._discover_server_tools` to support new transport methods (WebSocket, etc.).

## Contributing

We welcome contributions! Please see our [GitHub repository](https://github.com/Dumbris/mcpproxy) for:

- 🐛 **Bug Reports**: [Submit an issue](https://github.com/Dumbris/mcpproxy/issues)
- 💡 **Feature Requests**: [Start a discussion](https://github.com/Dumbris/mcpproxy/discussions)
- 🔧 **Pull Requests**: Fork the repo and submit a PR

### Development Setup

```bash
git clone https://github.com/Dumbris/mcpproxy.git
cd mcpproxy
pip install -e .[dev,all]
pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

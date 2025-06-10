# Smart MCP Proxy

A federating gateway that sits between AI agents and multiple Model Context Protocol (MCP) servers, providing intelligent tool discovery and dynamic registration.

## Features

- **Dynamic Tool Discovery**: Automatically discovers tools from multiple MCP servers
- **Intelligent Search**: Uses configurable embedding backends (BM25, HuggingFace, OpenAI) to find relevant tools
- **One-Click Tool Access**: Single `retrieve_tools` function that searches, registers, and exposes the top 5 most relevant tools
- **FastMCP Integration**: Built on FastMCP v2 for robust server runtime and client capabilities
- **Persistent Indexing**: SQLite + Faiss storage for fast tool lookup and change detection
- **MCP Spec Compliant**: Emits proper `notifications/tools/list_changed` events

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

```bash
git clone <repository>
cd smart-mcp-proxy
pip install -e .
```

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
from smart_mcp_proxy import SmartMCPProxyServer

proxy = SmartMCPProxyServer("config.json")
await proxy.start()

# Use the indexer directly
results = await proxy.indexer.search_tools("delete volume", k=3)
for result in results:
    print(f"{result.tool.name}: {result.score}")
```

## Project Structure

```
smart_mcp_proxy/
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

1. Follow KISS and DRY principles
2. Use Python 3.11+ type annotations (plain types, not `typing.List`)
3. Add tests for new embedders and server types
4. Update documentation

## License

[License information]

# MCP Client Python Example

A clean, production-ready Python client for the Model Context Protocol (MCP) with a web-based chat interface.

## Features

- 🔗 Connect to any MCP server (HTTP/WebSocket/stdio)
- 🤖 OpenAI GPT integration for intelligent tool usage
- 💬 Real-time streaming chat interface
- 🛠️ Automatic tool discovery and execution
- 🎯 Production-ready with error handling

## Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd mcp-client-python-example
```

### 2. Install Dependencies

```bash
# API dependencies
cd api
pip install -r requirements.txt

# Frontend dependencies  
cd ../front
pip install -r requirements.txt
```

### 3. Configure Environment

Create `.env` file in the `api` directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
MCP_SERVER_ENDPOINT=http://your-mcp-server.com
```

### 4. Start Services

```bash
# Terminal 1: API Server
cd api
python main.py

# Terminal 2: Frontend
cd front
streamlit run main.py --server.port 8501
```

### 5. Access

- **Frontend**: http://localhost:8501
- **API**: http://localhost:8001

## Configuration

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | `sk-...` |
| `MCP_SERVER_ENDPOINT` | MCP server URL | `http://localhost:3000` |

### MCP Server Types

- **HTTP**: `http://your-server.com/mcp`
- **WebSocket**: `ws://your-server.com/mcp`  
- **Local**: `/path/to/server/main.py`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tools` | GET | List available tools |
| `/server/status` | GET | Connection status |
| `/query` | POST | Process query (sync) |
| `/query/stream` | POST | Process query (stream) |
| `/tool` | POST | Call specific tool |

## Project Structure

```
├── api/                 # FastAPI backend
│   ├── main.py         # API server
│   ├── mcp_client.py   # MCP client
│   ├── requirements.txt
│   └── utils/          # Utilities
├── front/              # Streamlit frontend
│   ├── main.py         # Frontend app
│   ├── chatbot.py      # Chat interface
│   ├── requirements.txt
│   └── utils/          # Utilities
└── README.md
```

## Usage

### Web Interface

1. Open http://localhost:8501
2. Type your query
3. AI will automatically use available tools

### API Usage

```python
import httpx

# Process query
response = httpx.post("http://localhost:8001/query", json={
    "query": "What's the weather in London?"
})
print(response.json())
```

## Development

### Adding Tools

Tools are automatically discovered from your MCP server - no client changes needed.

### Customization

- **Frontend**: Edit `front/chatbot.py`
- **API**: Edit `api/main.py`
- **MCP Client**: Edit `api/mcp_client.py`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Connection refused | Check MCP server is running |
| API key error | Verify `OPENAI_API_KEY` |
| Tool not found | Check MCP server implementation |

## License

MIT License

# Setup Guide

## Quick Setup

### 1. Install Dependencies

```bash
# API dependencies
cd api
pip install -r requirements.txt

# Frontend dependencies
cd ../front
pip install -r requirements.txt
```

### 2. Create Environment File

Create `api/.env` with:

```env
OPENAI_API_KEY=your_openai_api_key_here
MCP_SERVER_ENDPOINT=http://your-mcp-server.com
```

### 3. Start Services

**Option A: Use startup script**
```bash
./start.sh
```

**Option B: Manual start**
```bash
# Terminal 1: API
cd api
python main.py

# Terminal 2: Frontend
cd front
streamlit run main.py --server.port 8501
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Your OpenAI API key |
| `MCP_SERVER_ENDPOINT` | Yes | MCP server URL |

## MCP Server Examples

- **HTTP**: `http://localhost:3000`
- **WebSocket**: `ws://your-server.com/mcp`
- **Local**: `/path/to/server/main.py`

## Access Points

- **Frontend**: http://localhost:8501
- **API**: http://localhost:8001 
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any
from contextlib import asynccontextmanager
from mcp_client import MCPClient
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
import json
import asyncio
from utils.logger import logger

load_dotenv()

class Settings(BaseSettings):
    # Connect to the MCP server running on port 3000
    mcp_server_endpoint: str = "http://localhost:3000"
    
    # Alternative: Connect to a remote MCP server via URL
    # mcp_server_endpoint: str = "https://your-mcp-server.com"
    
    class Config:
        env_prefix = ""
        env_file = ".env"
        extra = "ignore"  # Allow extra fields from environment

settings = Settings()

# Log the configuration for debugging
logger.info(f"MCP Server endpoint: {settings.mcp_server_endpoint}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    client = MCPClient()
    try:
        # Wait for server to be ready with retries
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                connected = await client.connect_to_server(settings.mcp_server_endpoint)
                if connected:
                    app.state.client = client
                    break
                else:
                    raise Exception("Connection returned False")
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.info(f"Connection attempt {attempt + 1} failed, retrying in {retry_delay} seconds: {str(e)}")
                    await asyncio.sleep(retry_delay)
                else:
                    raise Exception(f"Failed to connect to server after {max_retries} attempts: {str(e)}")
        
        yield
    except Exception as e:
        logger.error(f"Lifespan error: {str(e)}")
        raise Exception(f"Failed to connect to server: {str(e)}")
    finally:
        # Shutdown
        await client.cleanup()

app = FastAPI(title="MCP Chatbot API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class QueryRequest(BaseModel):
    query: str



class ToolCall(BaseModel):
    name: str
    args: Dict[str, Any]

@app.get("/tools")
async def get_available_tools():
    """Get list of available tools"""
    try:
        tools = await app.state.client.get_mcp_tools()
        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
                for tool in tools
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/server/status")
async def get_server_status():
    """Get current MCP server connection status"""
    try:
        return {
            "connected": app.state.client.session is not None,
            "connection_type": app.state.client.connection_type,
            "endpoint": settings.mcp_server_endpoint,
            "tools_count": len(app.state.client.tools)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/query")
async def process_query(request: QueryRequest):
    """Process a query and return the response (non-streaming)"""
    try:
        messages = await app.state.client.process_query(request.query)
        return {"messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def serialize_message(message):
    """Serialize message for JSON response"""
    if isinstance(message, dict):
        # Simple serialization - handle any non-serializable objects
        try:
            json.dumps(message)
            return message
        except (TypeError, ValueError):
            # Convert to string if not serializable
            return {k: str(v) for k, v in message.items()}
    return message

@app.post("/query/stream")
async def process_query_stream(request: QueryRequest):
    """Process a query and stream responses via SSE"""
    async def generate():
        try:
            async for message in app.state.client.process_query_stream(request.query):
                serialized_message = serialize_message(message)
                yield f"data: {json.dumps(serialized_message)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'role': 'error', 'content': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.post("/tool")
async def call_tool(tool_call: ToolCall):
    """Call a specific tool"""
    try:
        result = await app.state.client.call_tool(tool_call.name, tool_call.args)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
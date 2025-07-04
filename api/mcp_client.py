from typing import Optional
from contextlib import AsyncExitStack
import traceback
from utils.logger import logger
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import json
import httpx
import asyncio
from openai import OpenAI


class MCPHttpClient:
    """Simple HTTP client for MCP servers"""
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self.session_id = None
        
    async def initialize(self):
        """Initialize the MCP session"""
        try:
            # Check if server is healthy
            response = await self.client.get(f"{self.server_url}/health")
            if response.status_code == 200:
                logger.info(f"Successfully connected to MCP server at {self.server_url}")
                return True
            else:
                logger.error(f"Server health check failed with status {response.status_code}")
                return False
            
        except Exception as e:
            logger.error(f"Failed to initialize HTTP client: {str(e)}")
            return False
    
    async def list_tools(self):
        """List available tools"""
        try:
            response = await self.client.get(f"{self.server_url}/tools")
            
            if response.status_code == 200:
                result = response.json()
                return result.get("tools", [])
            return []
            
        except Exception as e:
            logger.error(f"Failed to list tools: {str(e)}")
            return []
    
    async def call_tool(self, name: str, arguments: dict):
        """Call a tool"""
        try:
            request_data = {
                "name": name,
                "arguments": arguments
            }
            
            response = await self.client.post(
                f"{self.server_url}/tools/call",
                json=request_data
            )
            
            if response.status_code == 200:
                result = response.json()
                # The server returns the data directly, not wrapped in a "result" key
                return result
            return None
            
        except Exception as e:
            logger.error(f"Failed to call tool: {str(e)}")
            return None
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.http_client: Optional[MCPHttpClient] = None
        self.exit_stack = AsyncExitStack()
        self.llm = OpenAI()
        self.tools = []
        self.messages = []
        self.logger = logger
        self.connection_type = None  # 'stdio' or 'http'

    async def call_tool(self, tool_name: str, tool_args: dict):
        """Call a tool with the given name and arguments"""
        try:
            if self.connection_type == 'http':
                result = await self.http_client.call_tool(tool_name, tool_args)
            else:
                result = await self.session.call_tool(tool_name, tool_args)
            return result
        except Exception as e:
            self.logger.error(f"Failed to call tool: {str(e)}")
            raise Exception(f"Failed to call tool: {str(e)}")

    async def connect_to_server(self, server_endpoint: str):
        """Connect to an MCP server via stdio (local) or HTTP (remote)

        Args:
            server_endpoint: Either a local file path (.py/.js) or HTTP URL
        """
        try:
            self.logger.info(f"Attempting to connect to server endpoint: {server_endpoint}")
            
            # Check if it's a URL (remote server)
            if server_endpoint.startswith(('http://', 'https://', 'ws://', 'wss://')):
                return await self._connect_to_remote_server(server_endpoint)
            else:
                # Local stdio server
                return await self._connect_to_local_server(server_endpoint)
                
        except Exception as e:
            self.logger.error(f"Failed to connect to server: {str(e)}")
            self.logger.debug(f"Connection error details: {traceback.format_exc()}")
            raise Exception(f"Failed to connect to server: {str(e)}")

    async def _connect_to_remote_server(self, server_url: str):
        """Connect to a remote MCP server via HTTP/WebSocket"""
        try:
            self.logger.info(f"Attempting to connect to remote server: {server_url}")
            
            # Create HTTP client for remote MCP server
            self.http_client = MCPHttpClient(server_url)
            initialized = await self.http_client.initialize()
            if not initialized:
                raise Exception("Failed to initialize HTTP client")
            
            self.connection_type = 'http'
            
            # Get tools from remote server
            mcp_tools = await self.get_mcp_tools()
            if not mcp_tools:
                raise Exception("No tools available from server")
                
            self.tools = [
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "input_schema": tool["inputSchema"],
                }
                for tool in mcp_tools
            ]
            
            self.logger.info(
                f"Successfully connected to remote server. Available tools: {[tool['name'] for tool in self.tools]}"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to remote server: {str(e)}")
            raise

    async def _connect_to_local_server(self, server_script_path: str):
        """Connect to a local MCP server via stdio"""
        try:
            is_python = server_script_path.endswith(".py")
            is_js = server_script_path.endswith(".js")
            if not (is_python or is_js):
                raise ValueError("Server script must be a .py or .js file")

            self.logger.info(
                f"Attempting to connect to local server using script: {server_script_path}"
            )
            command = "python" if is_python else "node"
            server_params = StdioServerParameters(
                command=command, args=[server_script_path], env=None
            )

            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.write)
            )

            await self.session.initialize()
            self.connection_type = 'stdio'
            
            mcp_tools = await self.get_mcp_tools()
            self.tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
                for tool in mcp_tools
            ]
            self.logger.info(
                f"Successfully connected to local server. Available tools: {[tool['name'] for tool in self.tools]}"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to local server: {str(e)}")
            raise

    async def get_mcp_tools(self):
        try:
            self.logger.info("Requesting MCP tools from the server.")
            if self.connection_type == 'http':
                tools = await self.http_client.list_tools()
            else:
                response = await self.session.list_tools()
                tools = response.tools
            return tools
        except Exception as e:
            self.logger.error(f"Failed to get MCP tools: {str(e)}")
            self.logger.debug(f"Error details: {traceback.format_exc()}")
            raise Exception(f"Failed to get tools: {str(e)}")

    async def call_llm(self):
        """Call the LLM with the given query"""
        try:
            # Convert tools to OpenAI format
            openai_tools = []
            for tool in self.tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"]
                    }
                })
            
            return self.llm.chat.completions.create(
                model="gpt-4o",  # You can change this to gpt-3.5-turbo for cheaper option
                max_tokens=1000,
                messages=self.messages,
                tools=openai_tools,
                tool_choice="auto"
            )
        except Exception as e:
            self.logger.error(f"Failed to call LLM: {str(e)}")
            raise Exception(f"Failed to call LLM: {str(e)}")

    async def process_query(self, query: str):
        """Process a query using Claude and available tools, returning all messages at the end"""
        async for message in self.process_query_stream(query):
            pass  # Collect all messages
        return self.messages[-len(self.messages):]  # Return all messages

    async def process_query_stream(self, query: str):
        """Process a query using Claude and available tools, streaming messages as they come"""
        try:
            self.logger.info(
                f"Processing new query: {query[:100]}..."
            )  # Log first 100 chars of query

            # Add the initial user message
            user_message = {"role": "user", "content": query}
            self.messages.append(user_message)
            yield user_message

            while True:
                self.logger.debug("Calling OpenAI API")
                response = await self.call_llm()

                # Get the assistant message
                assistant_message = {
                    "role": "assistant",
                    "content": response.choices[0].message.content or "",
                }
                
                # Check if there are tool calls
                if response.choices[0].message.tool_calls:
                    # Add tool calls to the assistant message
                    assistant_message["tool_calls"] = []
                    for tool_call in response.choices[0].message.tool_calls:
                        assistant_message["tool_calls"].append({
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        })
                
                self.messages.append(assistant_message)
                yield assistant_message

                # If no tool calls, we're done
                if not response.choices[0].message.tool_calls:
                    break

                # Process tool calls
                for tool_call in response.choices[0].message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    tool_use_id = tool_call.id

                    self.logger.info(
                        f"Executing tool: {tool_name} with args: {tool_args}"
                    )
                    try:
                        result = await self.call_tool(tool_name, tool_args)
                        self.logger.info(f"Tool result: {result}")

                        # PATCH: Handle None or error result gracefully
                        if result is None:
                            error_msg = f"Tool '{tool_name}' returned no result."
                            self.logger.error(error_msg)
                            # Still provide a tool response message to satisfy OpenAI API
                            tool_result_message = {
                                "role": "tool",
                                "content": error_msg,
                                "tool_call_id": tool_use_id,
                                "name": tool_name,
                                "args": tool_args,
                            }
                            self.messages.append(tool_result_message)
                            yield tool_result_message
                            continue
                        if isinstance(result, dict) and "error" in result:
                            error_msg = f"Tool '{tool_name}' error: {result['error']}"
                            self.logger.error(error_msg)
                            # Still provide a tool response message to satisfy OpenAI API
                            tool_result_message = {
                                "role": "tool",
                                "content": error_msg,
                                "tool_call_id": tool_use_id,
                                "name": tool_name,
                                "args": tool_args,
                            }
                            self.messages.append(tool_result_message)
                            yield tool_result_message
                            continue

                        # Ensure the content is properly serializable
                        content = getattr(result, "content", result)
                        if hasattr(content, 'to_dict'):
                            content_data = content.to_dict()
                        elif hasattr(content, 'dict'):
                            content_data = content.dict()
                        elif hasattr(content, 'model_dump'):
                            content_data = content.model_dump()
                        elif hasattr(content, '__dict__'):
                            content_data = content.__dict__
                        else:
                            content_data = str(content)
                        
                        tool_result_message = {
                            "role": "tool",
                            "content": content_data,
                            "tool_call_id": tool_use_id,
                            "name": tool_name,
                            "args": tool_args,
                        }
                        self.messages.append(tool_result_message)
                        yield tool_result_message
                    except Exception as e:
                        error_msg = f"Tool execution failed: {str(e)}"
                        self.logger.error(error_msg)
                        # Still provide a tool response message to satisfy OpenAI API
                        tool_result_message = {
                            "role": "tool",
                            "content": error_msg,
                            "tool_call_id": tool_use_id,
                            "name": tool_name,
                            "args": tool_args,
                        }
                        self.messages.append(tool_result_message)
                        yield tool_result_message

        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            self.logger.debug(
                f"Query processing error details: {traceback.format_exc()}"
            )
            yield {"role": "error", "content": str(e)}



    async def cleanup(self):
        """Clean up resources"""
        try:
            self.logger.info("Cleaning up resources")
            
            # Close HTTP client if it exists
            if self.http_client:
                await self.http_client.close()
            
            # Close stdio client if it exists
            if self.session:
                await self.exit_stack.aclose()
                
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

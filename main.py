import asyncio
import json
import logging
from typing import Optional
from contextlib import AsyncExitStack
import sys
import traceback
from datetime import datetime

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from test_data import test_tool_result_content

from anthropic import Anthropic
from anthropic.types import Message
from dotenv import load_dotenv
import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("mcp_client.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("MCPClient")

load_dotenv()  # load environment variables from .env


class MCPClientError(Exception):
    """Base exception class for MCPClient errors"""

    pass


class ConnectionError(MCPClientError):
    """Raised when there are connection issues"""

    pass


class ToolExecutionError(MCPClientError):
    """Raised when there are issues executing tools"""

    pass


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.llm = Anthropic()
        self.tools = []
        self.messages = []
        self.logger = logging.getLogger("MCPClient")

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        try:
            is_python = server_script_path.endswith(".py")
            is_js = server_script_path.endswith(".js")
            if not (is_python or is_js):
                raise ValueError("Server script must be a .py or .js file")

            self.logger.info(
                f"Attempting to connect to server using script: {server_script_path}"
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
                f"Successfully connected to server. Available tools: {[tool['name'] for tool in self.tools]}"
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to server: {str(e)}")
            self.logger.debug(f"Connection error details: {traceback.format_exc()}")
            raise ConnectionError(f"Failed to connect to server: {str(e)}")

    async def get_mcp_tools(self):
        try:
            response = await self.session.list_tools()
            tools = response.tools
            return tools
        except Exception as e:
            self.logger.error(f"Failed to get MCP tools: {str(e)}")
            raise ToolExecutionError(f"Failed to get tools: {str(e)}")

    async def call_llm(self) -> Message:
        """Call the LLM with the given query"""
        try:
            return self.llm.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=self.messages,
                tools=self.tools,
            )
        except Exception as e:
            self.logger.error(f"Failed to call LLM: {str(e)}")
            raise ToolExecutionError(f"Failed to call LLM: {str(e)}")

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        try:
            self.logger.info(
                f"Processing new query: {query[:100]}..."
            )  # Log first 100 chars of query

            self.messages.append({"role": "user", "content": query})
            final_text = []

            while True:
                self.logger.debug("Calling Claude API")
                response = await self.call_llm()

                if response.content[0].type == "text" and len(response.content) == 1:
                    final_text.append(response.content[0].text)
                    self.messages.append(
                        {"role": "assistant", "content": response.content[0].text}
                    )
                    self.log_conversation()
                    break

                self.messages.append(
                    {"role": "assistant", "content": response.to_dict()["content"]}
                )
                self.log_conversation()

                for content in response.content:
                    if content.type == "text":
                        final_text.append(content.text)
                    elif content.type == "tool_use":
                        tool_name = content.name
                        tool_args = content.input
                        tool_use_id = content.id

                        self.logger.info(
                            f"Executing tool: {tool_name} with args: {tool_args}"
                        )
                        try:
                            # turn this one return a simple string
                            # result = await self.session.call_tool(tool_name, tool_args)
                            result = test_tool_result_content
                            self.messages.append(
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "tool_result",
                                            "tool_use_id": tool_use_id,
                                            "content": result,
                                        }
                                    ],
                                }
                            )
                        except Exception as e:
                            error_msg = f"Tool execution failed: {str(e)}"
                            self.logger.error(error_msg)
                            raise ToolExecutionError(error_msg)

                self.log_conversation()

            return "\n".join(final_text)
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            self.logger.debug(
                f"Query processing error details: {traceback.format_exc()}"
            )
            raise

    def log_conversation(self):
        """save conversation to a json file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"
            with open(filename, "w") as f:
                json.dump(self.messages, f, indent=2)
            self.logger.debug(f"Conversation logged to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to log conversation: {str(e)}")

    async def chat_loop(self):
        """Run an interactive chat loop"""
        self.logger.info("Starting chat loop")
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == "quit":
                    self.logger.info("User requested to quit")
                    break

                response = await self.process_query(query)
                print("\n" + response)
                self.log_conversation()

            except KeyboardInterrupt:
                self.logger.info("Received keyboard interrupt")
                print("\nShutting down gracefully...")
                break
            except Exception as e:
                self.logger.error(f"Error in chat loop: {str(e)}")
                self.logger.debug(f"Chat loop error details: {traceback.format_exc()}")
                print(f"\nError: {str(e)}")
                print("Type 'quit' to exit or try another query")

    async def cleanup(self):
        """Clean up resources"""
        try:
            self.logger.info("Cleaning up resources")
            await self.exit_stack.aclose()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")


async def main():

    @st.cache_resource
    def get_client():
        client = MCPClient()
        return client

    logger.info("Starting MCP Client application")
    st.set_page_config(page_title="MCP Client", page_icon=":shark:")
    if "server_connected" not in st.session_state:
        st.session_state["server_connected"] = False

    st.title("MCP Client")

    with st.sidebar:
        server_script_path = st.text_input("Server script path")
        if st.button("Connect"):
            client = get_client()
            server_connected = await client.connect_to_server(server_script_path)
            if server_connected:
                st.session_state["server_connected"] = True
                st.success("Connected to server")

    if st.session_state["server_connected"]:
        client = get_client()
        for message in client.messages:
            st.chat_message(message["role"]).markdown(message["content"])
        query = st.chat_input("Enter your query here")
        if query:
            response = await client.process_query(query)
            st.chat_message("assistant").markdown(response)

    if len(sys.argv) < 2:
        logger.error("No server script path provided")
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])

    except Exception as e:
        logger.critical(f"Critical error in main: {str(e)}")
        logger.debug(f"Main error details: {traceback.format_exc()}")
        sys.exit(1)
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

from logger import logger
from client import MCPClient
from anthropic.types import Message
import streamlit as st


class Chatbot:
    def __init__(self, client: MCPClient):
        self.client = client
        self.current_tool_call = {"name": None, "args": None}

    def display_message(self, message: Message):
        # display user message
        if message["role"] == "user" and type(message["content"]) == str:
            st.chat_message("user").markdown(message["content"])

        # display tool result
        if message["role"] == "user" and type(message["content"]) == list:
            for content in message["content"]:
                if content["type"] == "tool_result":
                    with st.chat_message("assistant"):
                        st.write(f"Called tool: {self.current_tool_call['name']}:")
                        st.json(
                            {
                                "name": self.current_tool_call["name"],
                                "args": self.current_tool_call["args"],
                                "content": content["content"],
                            },
                            expanded=False,
                        )

        # display ai message
        if message["role"] == "assistant" and type(message["content"]) == str:
            st.chat_message("assistant").markdown(message["content"])

        # store current ai tool use
        if message["role"] == "assistant" and type(message["content"]) == list:
            for content in message["content"]:
                # ai tool use
                if content["type"] == "tool_use":
                    self.current_tool_call = {
                        "name": content["name"],
                        "args": content["input"],
                    }

    async def render(self):
        st.title("MCP Client")

        with st.sidebar:
            server_script_path = st.text_input(
                "Server script path",
                value="/Users/alejandro/repos/code/mcp/documentation/main.py",
            )
            if st.button("Connect"):
                server_connected = await self.client.connect_to_server(
                    server_script_path
                )
                if server_connected:
                    st.session_state["server_connected"] = True
                    st.success("Connected to server")

        if st.session_state["server_connected"]:
            client = self.client
            # Display existing messages
            for message in client.messages:
                self.display_message(message)

            # Handle new query
            query = st.chat_input("Enter your query here")
            if query:
                async for message in client.process_query(query):
                    self.display_message(message)

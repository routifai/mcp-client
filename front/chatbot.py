import streamlit as st
import httpx
from typing import Dict, Any


class Chatbot:
    def __init__(self):
        self.api_url = "http://localhost:8000"
        self.current_tool_call = {"name": None, "args": None}
        self.messages = []

    def display_message(self, message: Dict[str, Any]):
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
            st.write("MCP Client")
        # Display existing messages
        for message in self.messages:
            self.display_message(message)

        # Handle new query
        query = st.chat_input("Enter your query here")
        if query:
            async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
                try:
                    response = await client.post(
                        f"{self.api_url}/query",
                        json={"query": query},
                        headers={"Content-Type": "application/json"},
                    )
                    if response.status_code == 200:
                        messages = response.json()["messages"]
                        st.session_state["messages"] = messages
                        for message in st.session_state["messages"]:
                            self.display_message(message)
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")

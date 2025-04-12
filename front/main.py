import asyncio
from utils.logger import logger
from api.client import MCPClient
from dotenv import load_dotenv
import streamlit as st
from front.chatbot import Chatbot

load_dotenv()  # load environment variables from .env


async def main():
    @st.cache_resource
    def get_client():
        client = MCPClient()
        return client
      
    if "server_connected" not in st.session_state:
        st.session_state["server_connected"] = False

    if "tools" not in st.session_state:
        st.session_state["tools"] = []

    st.set_page_config(page_title="MCP Client", page_icon=":shark:")

    chatbot = Chatbot(get_client())
    await chatbot.render()


if __name__ == "__main__":
    asyncio.run(main())

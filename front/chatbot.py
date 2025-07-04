import streamlit as st
import httpx
from typing import Dict, Any
import json
import asyncio


class Chatbot:
    def __init__(self, api_url: str):
        self.api_url = api_url
        self.current_tool_call = {"name": None, "args": None}
        self.messages = st.session_state["messages"]

    def display_message(self, message: Dict[str, Any]):
        # display user message
        if message["role"] == "user" and type(message["content"]) == str:
            st.chat_message("user").markdown(message["content"])

        # display tool result (OpenAI format)
        if message["role"] == "tool":
            self.display_tool_result(message)

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

    def display_tool_result(self, message: Dict[str, Any]):
        """Display tool result with rich annotations"""
        with st.chat_message("assistant"):
            # Use the name and args from the message itself, fallback to current_tool_call
            tool_name = message.get("name", self.current_tool_call.get("name", "Unknown"))
            tool_args = message.get("args", self.current_tool_call.get("args", {}))
            
            st.write(f"🔧 Tool result for: {tool_name}:")
            try:
                # Try to parse as JSON, fallback to string
                if isinstance(message["content"], str):
                    try:
                        content_data = json.loads(message["content"])
                    except json.JSONDecodeError:
                        content_data = message["content"]
                else:
                    content_data = message["content"]
                
                # Check if we have rich annotations
                if isinstance(content_data, dict) and "annotations" in content_data and content_data["annotations"]:
                    annotations = content_data["annotations"]
                    
                    # Display rich metadata
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Results", annotations.get("result_count", 0))
                    with col2:
                        st.metric("Status", annotations.get("status", "unknown"))
                    with col3:
                        st.metric("Type", annotations.get("query_type", "unknown"))
                    
                    # Display timestamp
                    if "timestamp" in annotations:
                        st.caption(f"🕒 Search performed at: {annotations['timestamp']}")
                    
                    # Display search engine info
                    if "search_engine" in annotations:
                        st.info(f"🔍 Search engine: {annotations['search_engine']}")
                    
                    # Display individual results if available
                    if "results" in annotations and annotations["results"]:
                        st.subheader("📊 Search Results Details")
                        for i, result in enumerate(annotations["results"][:3], 1):  # Show top 3
                            with st.expander(f"Result {i}: {result.get('title', 'No title')}"):
                                st.write(f"**URL:** {result.get('url', 'No URL')}")
                                st.write(f"**Domain:** {result.get('domain', result.get('section', 'Unknown'))}")
                                st.write(f"**Position:** {result.get('position', i)}")
                                st.write(f"**Snippet:** {result.get('snippet', 'No snippet')}")
                
                # Ensure we have valid data for display
                display_data = {
                    "name": tool_name if tool_name else "Unknown",
                    "args": tool_args if tool_args else {},
                    "content": content_data if content_data else "No content"
                }
                
                st.json(display_data, expanded=False)
            except Exception as e:
                st.error(f"Error displaying tool result: {str(e)}")
                st.text(f"Raw content: {message.get('content', 'No content')}")

    def display_streaming_response(self, response_text: str, tool_results: list, placeholder):
        """Display streaming response with tool results"""
        with placeholder.container():
            # Display assistant response
            if response_text:
                st.markdown(f"**Assistant:** {response_text}▌")
            
            # Display tool results
            for tool_result in tool_results:
                self.display_tool_result_inline(tool_result)

    def display_final_response(self, response_text: str, tool_results: list):
        """Display final response with tool results"""
        # Display assistant response
        if response_text:
            st.chat_message("assistant").markdown(response_text)
        
        # Don't display tool results again since they were already shown during streaming
        # The tool results are already visible from the streaming phase

    def display_tool_result_inline(self, message: Dict[str, Any]):
        """Display tool result inline without chat message wrapper"""
        # Use the name and args from the message itself, fallback to current_tool_call
        tool_name = message.get("name", self.current_tool_call.get("name", "Unknown"))
        tool_args = message.get("args", self.current_tool_call.get("args", {}))
        
        st.write(f"🔧 Tool result for: {tool_name}:")
        try:
            # Try to parse as JSON, fallback to string
            if isinstance(message["content"], str):
                try:
                    content_data = json.loads(message["content"])
                except json.JSONDecodeError:
                    content_data = message["content"]
            else:
                content_data = message["content"]
            
            # Check if we have rich annotations
            if isinstance(content_data, dict) and "annotations" in content_data and content_data["annotations"]:
                annotations = content_data["annotations"]
                
                # Display rich metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Results", annotations.get("result_count", 0))
                with col2:
                    st.metric("Status", annotations.get("status", "unknown"))
                with col3:
                    st.metric("Type", annotations.get("query_type", "unknown"))
                
                # Display timestamp
                if "timestamp" in annotations:
                    st.caption(f"🕒 Search performed at: {annotations['timestamp']}")
                
                # Display search engine info
                if "search_engine" in annotations:
                    st.info(f"🔍 Search engine: {annotations['search_engine']}")
                
                # Display individual results if available
                if "results" in annotations and annotations["results"]:
                    st.subheader("📊 Search Results Details")
                    for i, result in enumerate(annotations["results"][:3], 1):  # Show top 3
                        with st.expander(f"Result {i}: {result.get('title', 'No title')}"):
                            st.write(f"**URL:** {result.get('url', 'No URL')}")
                            st.write(f"**Domain:** {result.get('domain', result.get('section', 'Unknown'))}")
                            st.write(f"**Position:** {result.get('position', i)}")
                            st.write(f"**Snippet:** {result.get('snippet', 'No snippet')}")
            
            # Ensure we have valid data for display
            display_data = {
                "name": tool_name if tool_name else "Unknown",
                "args": tool_args if tool_args else {},
                "content": content_data if content_data else "No content"
            }
            
            st.json(display_data, expanded=False)
        except Exception as e:
            st.error(f"Error displaying tool result: {str(e)}")
            st.text(f"Raw content: {message.get('content', 'No content')}")

    async def get_tools(self):
        async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
            response = await client.get(
                f"{self.api_url}/tools",
                headers={"Content-Type": "application/json"},
            )
            return response.json()

    async def get_server_status(self):
        async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
            response = await client.get(
                f"{self.api_url}/server/status",
                headers={"Content-Type": "application/json"},
            )
            return response.json()

    async def render(self):
        st.title("MCP Client")

        with st.sidebar:
            st.subheader("Settings")
            st.write("API URL: ", self.api_url)
            
            # Show server status
            try:
                server_status = await self.get_server_status()
                st.subheader("🔗 MCP Server Status")
                
                if server_status.get("connected"):
                    st.success("✅ Connected")
                    st.write(f"**Type:** {server_status.get('connection_type', 'Unknown')}")
                    st.write(f"**Endpoint:** {server_status.get('endpoint', 'Unknown')}")
                    st.write(f"**Tools:** {server_status.get('tools_count', 0)} available")
                else:
                    st.error("❌ Disconnected")
                    st.write(f"**Endpoint:** {server_status.get('endpoint', 'Unknown')}")
            except Exception as e:
                st.error(f"❌ Server status error: {str(e)}")
            
            # Show available tools
            try:
                result = await self.get_tools()
                st.subheader("🛠️ Available Tools")
                st.write([tool["name"] for tool in result["tools"]])
            except Exception as e:
                st.error(f"❌ Tools error: {str(e)}")

        # Display existing messages
        for message in self.messages:
            self.display_message(message)

        # Handle new query
        query = st.chat_input("Enter your query here")
        if query:
            # Add user message immediately
            user_message = {"role": "user", "content": query}
            st.session_state["messages"].append(user_message)
            self.display_message(user_message)
            
            # Create a placeholder for streaming content
            message_placeholder = st.empty()
            full_response = ""
            tool_results = []
            
            # Stream the response
            async with httpx.AsyncClient(timeout=60.0, verify=False) as client:
                try:
                    async with client.stream(
                        "POST",
                        f"{self.api_url}/query/stream",
                        json={"query": query},
                        headers={"Content-Type": "application/json"},
                    ) as response:
                        if response.status_code == 200:
                            async for line in response.aiter_lines():
                                if line.startswith("data: "):
                                    try:
                                        data = json.loads(line[6:])  # Remove "data: " prefix
                                        
                                        if data["role"] == "assistant":
                                            if data["content"]:
                                                full_response += data["content"]
                                                # Display current response with tool results
                                                self.display_streaming_response(full_response, tool_results, message_placeholder)
                                        
                                        elif data["role"] == "tool":
                                            # Store tool result for display
                                            tool_results.append(data)
                                            # Display current response with updated tool results
                                            self.display_streaming_response(full_response, tool_results, message_placeholder)
                                        
                                        elif data["role"] == "error":
                                            st.error(f"Error: {data['content']}")
                                            break
                                            
                                    except json.JSONDecodeError:
                                        continue
                    
                    # Final display without cursor
                    self.display_final_response(full_response, tool_results)
                    
                    # Add final assistant message to session state
                    if full_response:
                        st.session_state["messages"].append({
                            "role": "assistant", 
                            "content": full_response
                        })
                        
                except Exception as e:
                    st.error(f"Frontend: Error processing query: {str(e)}")

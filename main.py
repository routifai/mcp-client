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

from anthropic import Anthropic
from dotenv import load_dotenv

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
                response = self.llm.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=self.messages,
                    tools=self.tools,
                )

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
                            result = await self.session.call_tool(tool_name, tool_args)
                            # final_text.append(result.content)
                            # print("tool call result", result)
                            # print("tool call result content", result.content)
                            # print("tool call result type", type(result.content))
                            test_content = """
                            This notebook covers how to get started with the Chroma vector store. Chroma is a AI-native open-source vector database focused on developer productivity and happiness. Chroma is licensed under Apache 2.0. View the full docs of Chroma at this page, and find the API reference for the LangChain integration at this page.
To access Chroma vector stores you'll need to install the langchain-chroma integration package.

pip install -qU "langchain-chroma>=0.1.2"

Credentials
You can use the Chroma vector store without any credentials, simply installing the package above is enough!

If you want to get best in-class automated tracing of your model calls you can also set your LangSmith API key by uncommenting below:

# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
# os.environ["LANGSMITH_TRACING"] = "true"


Initialization
Basic Initialization
Below is a basic initialization, including the use of a directory to save the data locally.

Select embeddings model:
pip install -qU langchain-openai

import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)


Initialization from client
You can also initialize from a Chroma client, which is particularly useful if you want easier access to the underlying database.

import chromadb

persistent_client = chromadb.PersistentClient()
collection = persistent_client.get_or_create_collection("collection_name")
collection.add(ids=["1", "2", "3"], documents=["a", "b", "c"])

vector_store_from_client = Chroma(
    client=persistent_client,
    collection_name="collection_name",
    embedding_function=embeddings,
)

Manage vector store
Once you have created your vector store, we can interact with it by adding and deleting different items.

Add items to vector store
We can add items to our vector store by using the add_documents function.

from uuid import uuid4

from langchain_core.documents import Document

document_1 = Document(
    page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},
    id=1,
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={"source": "news"},
    id=2,
)

document_3 = Document(
    page_content="Building an exciting new project with LangChain - come check it out!",
    metadata={"source": "tweet"},
    id=3,
)

document_4 = Document(
    page_content="Robbers broke into the city bank and stole $1 million in cash.",
    metadata={"source": "news"},
    id=4,
)

document_5 = Document(
    page_content="Wow! That was an amazing movie. I can't wait to see it again.",
    metadata={"source": "tweet"},
    id=5,
)

document_6 = Document(
    page_content="Is the new iPhone worth the price? Read this review to find out.",
    metadata={"source": "website"},
    id=6,
)

document_7 = Document(
    page_content="The top 10 soccer players in the world right now.",
    metadata={"source": "website"},
    id=7,
)

document_8 = Document(
    page_content="LangGraph is the best framework for building stateful, agentic applications!",
    metadata={"source": "tweet"},
    id=8,
)

document_9 = Document(
    page_content="The stock market is down 500 points today due to fears of a recession.",
    metadata={"source": "news"},
    id=9,
)

document_10 = Document(
    page_content="I have a bad feeling I am going to get deleted :(",
    metadata={"source": "tweet"},
    id=10,
)

documents = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5,
    document_6,
    document_7,
    document_8,
    document_9,
    document_10,
]
uuids = [str(uuid4()) for _ in range(len(documents))]

vector_store.add_documents(documents=documents, ids=uuids)


API Reference:Document
['f22ed484-6db3-4b76-adb1-18a777426cd6',
 'e0d5bab4-6453-4511-9a37-023d9d288faa',
 '877d76b8-3580-4d9e-a13f-eed0fa3d134a',
 '26eaccab-81ce-4c0a-8e76-bf542647df18',
 'bcaa8239-7986-4050-bf40-e14fb7dab997',
 'cdc44b38-a83f-4e49-b249-7765b334e09d',
 'a7a35354-2687-4bc2-8242-3849a4d18d34',
 '8780caf1-d946-4f27-a707-67d037e9e1d8',
 'dec6af2a-7326-408f-893d-7d7d717dfda9',
 '3b18e210-bb59-47a0-8e17-c8e51176ea5e']

Update items in vector store
Now that we have added documents to our vector store, we can update existing documents by using the update_documents function.

updated_document_1 = Document(
    page_content="I had chocolate chip pancakes and fried eggs for breakfast this morning.",
    metadata={"source": "tweet"},
    id=1,
)

updated_document_2 = Document(
    page_content="The weather forecast for tomorrow is sunny and warm, with a high of 82 degrees.",
    metadata={"source": "news"},
    id=2,
)

vector_store.update_document(document_id=uuids[0], document=updated_document_1)
# You can also update multiple documents at once
vector_store.update_documents(
    ids=uuids[:2], documents=[updated_document_1, updated_document_2]
)


Delete items from vector store
We can also delete items from our vector store as follows:

vector_store.delete(ids=uuids[-1])

Query vector store
Once your vector store has been created and the relevant documents have been added you will most likely wish to query it during the running of your chain or agent.

Query directly
Similarity search
Performing a simple similarity search can be done as follows:

results = vector_store.similarity_search(
    "LangChain provides abstractions to make working with LLMs easy",
    k=2,
    filter={"source": "tweet"},
)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")

* Building an exciting new project with LangChain - come check it out! [{'source': 'tweet'}]
* LangGraph is the best framework for building stateful, agentic applications! [{'source': 'tweet'}]


Similarity search with score
If you want to execute a similarity search and receive the corresponding scores you can run:

results = vector_store.similarity_search_with_score(
    "Will it be hot tomorrow?", k=1, filter={"source": "news"}
)
for res, score in results:
    print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")

* [SIM=1.726390] The stock market is down 500 points today due to fears of a recession. [{'source': 'news'}]


Search by vector
You can also search by vector:

results = vector_store.similarity_search_by_vector(
    embedding=embeddings.embed_query("I love green eggs and ham!"), k=1
)
for doc in results:
    print(f"* {doc.page_content} [{doc.metadata}]")

* I had chocalate chip pancakes and fried eggs for breakfast this morning. [{'source': 'tweet'}]


Other search methods
There are a variety of other search methods that are not covered in this notebook, such as MMR search or searching by vector. For a full list of the search abilities available for AstraDBVectorStore check out the API reference.

Query by turning into retriever
You can also transform the vector store into a retriever for easier usage in your chains. For more information on the different search types and kwargs you can pass, please visit the API reference here.

retriever = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 1, "fetch_k": 5}
)
retriever.invoke("Stealing from the bank is a crime", filter={"source": "news"})


[Document(metadata={'source': 'news'}, page_content='Robbers broke into the city bank and stole $1 million in cash.')]
                            """.strip()
                            self.messages.append(
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "tool_result",
                                            "tool_use_id": tool_use_id,
                                            "content": test_content,
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
    logger.info("Starting MCP Client application")

    if len(sys.argv) < 2:
        logger.error("No server script path provided")
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    except Exception as e:
        logger.critical(f"Critical error in main: {str(e)}")
        logger.debug(f"Main error details: {traceback.format_exc()}")
        sys.exit(1)
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

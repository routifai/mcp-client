import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("mcp_client.log"), logging.StreamHandler(sys.stdout)],
)

# Create logger instance
logger = logging.getLogger("MCPClient")

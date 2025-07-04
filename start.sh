#!/bin/bash

# MCP Client Startup Script
echo "🚀 Starting MCP Client..."

# Check if .env file exists
if [ ! -f "api/.env" ]; then
    echo "❌ Error: api/.env file not found!"
    echo "Please create api/.env with:"
    echo "OPENAI_API_KEY=your_openai_api_key_here"
    echo "MCP_SERVER_ENDPOINT=http://your-mcp-server.com"
    exit 1
fi

# Start API server
echo "📡 Starting API server..."
cd api
python main.py &
API_PID=$!
cd ..

# Wait a moment for API to start
sleep 3

# Start frontend
echo "🌐 Starting frontend..."
cd front
streamlit run main.py --server.port 8501 &
FRONTEND_PID=$!
cd ..

echo "✅ Services started!"
echo "🌐 Frontend: http://localhost:8501"
echo "📡 API: http://localhost:8001"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
trap "echo '🛑 Stopping services...'; kill $API_PID $FRONTEND_PID; exit" INT
wait 
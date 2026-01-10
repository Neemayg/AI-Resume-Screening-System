#!/bin/bash

# AI Resume Screening System - Backend Startup Script
# This script starts the FastAPI backend server

echo "🚀 Starting AI Resume Screening System Backend..."
echo ""

# Check if we're in the backend directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found!"
    echo "Please run this script from the backend directory:"
    echo "  cd backend"
    echo "  ./start_server.sh"
    exit 1
fi

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Install dependencies if needed
echo "📋 Checking dependencies..."
pip3 install -r requirements.txt --quiet

echo ""
echo "✅ Starting server on http://127.0.0.1:8000"
echo "   Press CTRL+C to stop"
echo ""

# Start the server with the CORRECT module name
python3 -m uvicorn main:app --reload --port 8000

#!/bin/bash

# AI Resume Screening System - Start Server Script

clear
echo "╔════════════════════════════════════════════════════════╗"
echo "║     AI Resume Screening System - Starting Server       ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if backend directory exists
if [ ! -d "backend" ]; then
    echo "❌ Error: backend directory not found!"
    echo "   Make sure you're running this from the project root."
    exit 1
fi

cd backend

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo ""
    echo "Please run setup first:"
    echo "  ./setup.sh"
    exit 1
fi

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found!"
    echo "   Make sure you're in the correct directory."
    exit 1
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

# Check if port 8000 is already in use
echo "🔍 Checking if port 8000 is available..."
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  Warning: Port 8000 is already in use!"
    echo ""
    read -p "   Kill existing process and continue? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "   → Killing process on port 8000..."
        lsof -ti:8000 | xargs kill -9 2>/dev/null
        sleep 1
        echo "   ✅ Port cleared"
    else
        echo "   Exiting..."
        exit 1
    fi
fi

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║              🚀 STARTING SERVER...                     ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo "📡 Backend API: http://127.0.0.1:8000"
echo "📖 API Docs:    http://127.0.0.1:8000/docs"
echo ""
echo "🌐 Open frontend/index.html in your browser"
echo ""
echo "Press CTRL+C to stop the server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Start the server with the CORRECT module name
uvicorn main:app --reload --port 8000

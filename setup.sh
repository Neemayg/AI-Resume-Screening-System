#!/bin/bash

# AI Resume Screening System - Complete Setup Script
# Works on Mac/Linux systems - Handles all edge cases

clear
echo "╔════════════════════════════════════════════════════════╗"
echo "║   AI Resume Screening System - Automatic Setup         ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "📁 Working directory: $SCRIPT_DIR"
cd "$SCRIPT_DIR"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python installation
echo "🔍 Step 1/5: Checking Python installation..."
if command_exists python3; then
    PYTHON_CMD="python3"
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    echo "   ✅ Found Python $PYTHON_VERSION"
elif command_exists python; then
    PYTHON_CMD="python"
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    echo "   ✅ Found Python $PYTHON_VERSION"
else
    echo "   ❌ Python is not installed!"
    echo ""
    echo "Please install Python 3.9 or higher:"
    echo "  Mac: brew install python3"
    echo "  Or download from: https://www.python.org/downloads/"
    exit 1
fi

# Check Python version (should be 3.9+)
PYTHON_MAJOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info[0])')
PYTHON_MINOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info[1])')

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    echo "   ⚠️  Warning: Python 3.9+ recommended (you have $PYTHON_VERSION)"
fi
echo ""

# Navigate to backend
echo "🔍 Step 2/5: Checking backend directory..."
if [ ! -d "backend" ]; then
    echo "   ❌ Backend directory not found!"
    echo "   Make sure you're running this from the project root."
    exit 1
fi
echo "   ✅ Backend directory found"
cd backend
echo ""

# Clean old virtual environment if exists
echo "🧹 Step 3/5: Setting up virtual environment..."
if [ -d "venv" ]; then
    echo "   ⚠️  Old virtual environment found, removing..."
    rm -rf venv
fi

$PYTHON_CMD -m venv venv
if [ $? -eq 0 ]; then
    echo "   ✅ Virtual environment created successfully"
else
    echo "   ❌ Failed to create virtual environment"
    exit 1
fi
echo ""

# Activate virtual environment
echo "🔌 Step 4/5: Activating virtual environment..."
source venv/bin/activate
if [ $? -eq 0 ]; then
    echo "   ✅ Virtual environment activated"
else
    echo "   ❌ Failed to activate virtual environment"
    exit 1
fi
echo ""

# Upgrade pip and install dependencies
echo "📚 Step 5/5: Installing dependencies..."
echo "   → Upgrading pip..."
pip install --upgrade pip --quiet

echo "   → Installing requirements (this may take a minute)..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "   ✅ All dependencies installed successfully"
else
    echo "   ❌ Failed to install dependencies"
    echo ""
    echo "Try manually:"
    echo "  cd backend"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi
echo ""

# Success message
echo "╔════════════════════════════════════════════════════════╗"
echo "║              ✅ SETUP COMPLETE! ✅                      ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo "🎯 Next Steps:"
echo "   1. Start the server:"
echo "      ./start.sh"
echo ""
echo "   2. Open frontend in browser:"
echo "      frontend/index.html"
echo ""
echo "   3. When your friend pushes new code:"
echo "      ./sync.sh"
echo ""
echo "📖 For troubleshooting, see: FRIEND_SETUP_GUIDE.md"
echo ""

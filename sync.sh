#!/bin/bash

# Sync Script - Pull latest code and update everything

clear
echo "╔════════════════════════════════════════════════════════╗"
echo "║        Syncing AI Resume Screening System              ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Error: Not a git repository!"
    echo "   Make sure you cloned the project properly."
    exit 1
fi

# Check internet connection
echo "🌐 Checking internet connection..."
if ! ping -c 1 github.com &> /dev/null; then
    echo "   ⚠️  Warning: Cannot reach GitHub"
    echo "   Check your internet connection"
    exit 1
fi
echo "   ✅ Connected"
echo ""

# Save any local changes
echo "💾 Saving any local changes..."
git stash push -m "Auto-stash before sync $(date)" 2>/dev/null
echo "   ✅ Local changes saved"
echo ""

# Pull latest changes
echo "📥 Pulling latest code from GitHub..."
git pull origin main

if [ $? -eq 0 ]; then
    echo "   ✅ Code updated successfully!"
else
    echo "   ❌ Failed to pull changes"
    echo ""
    echo "Try manually:"
    echo "  git pull origin main"
    exit 1
fi
echo ""

# Navigate to backend
cd backend

# Check if virtual environment exists
echo "🔍 Checking virtual environment..."
if [ ! -d "venv" ]; then
    echo "   ⚠️  Virtual environment not found, creating..."
    python3 -m venv venv
    if [ $? -eq 0 ]; then
        echo "   ✅ Virtual environment created"
    else
        echo "   ❌ Failed to create virtual environment"
        exit 1
    fi
else
    echo "   ✅ Virtual environment found"
fi
echo ""

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate
echo "   ✅ Activated"
echo ""

# Update dependencies
echo "📚 Updating dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --upgrade

if [ $? -eq 0 ]; then
    echo "   ✅ Dependencies updated"
else
    echo "   ⚠️  Some dependencies may have failed to update"
    echo "   The system should still work"
fi
echo ""

# Success message
echo "╔════════════════════════════════════════════════════════╗"
echo "║           ✅ SYNC COMPLETE! ✅                          ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo "🎯 Next step: Start the server"
echo "   ./start.sh"
echo ""
echo "📖 For help, see: FRIEND_SETUP_GUIDE.md"
echo ""

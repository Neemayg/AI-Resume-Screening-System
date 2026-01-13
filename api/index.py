"""
Vercel Serverless Function Entry Point
Wraps the FastAPI application for Vercel's Python runtime.
"""

import sys
from pathlib import Path

# Add backend directory to Python path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

# Import the FastAPI app from main.py
from main import app

# Vercel expects the app to be named 'app' or 'handler'
# FastAPI apps are ASGI-compatible and work directly with Vercel

"""
Vercel Serverless Function Entry Point
"""

import sys
import os
from pathlib import Path

# Add backend directory to Python path BEFORE other imports
backend_path = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_path))
os.chdir(backend_path)

# Load environment variables
from dotenv import load_dotenv
load_dotenv(backend_path / ".env")

# Now import the FastAPI app
from main import app

# Vercel will use this 'app' object

# AI Resume Screening System - Backend

A sophisticated AI-powered resume screening system that analyzes and ranks candidates based on job description matching using NLP techniques.

## Features

- **Document Parsing**: Extracts text from PDF and DOCX files
- **NLP Processing**: Tokenization, lemmatization, and stopword removal
- **TF-IDF Vectorization**: Converts text to numerical vectors
- **Cosine Similarity**: Computes semantic similarity between documents
- **Skill Matching**: Identifies and matches specific skills
- **Intelligent Ranking**: Ranks candidates with detailed scoring breakdown
- **RESTful API**: Clean, frontend-friendly JSON responses

## Tech Stack

- **Python 3.9+**
- **FastAPI** - Modern, fast web framework
- **scikit-learn** - TF-IDF and cosine similarity
- **NLTK** - Natural language processing
- **pdfplumber/PyPDF2** - PDF parsing
- **python-docx** - DOCX parsing

## Installation

### 1. Clone and Navigate

```bash
cd backend
```

### 2. Install Dependencies

```bash
pip3 install -r requirements.txt
```

## Running the Backend

### Quick Start (Recommended) ⚡

```bash
cd backend
chmod +x start_server.sh
./start_server.sh
```

### Manual Start

1. Navigate to backend directory:
```bash
cd backend
```

2. Install dependencies (if not already installed):
```bash
pip3 install -r requirements.txt
```

3. **⚠️ IMPORTANT - Use the correct command:**
```bash
python3 -m uvicorn main:app --reload --port 8000
```

**❌ Common Error:** Do NOT use `uvicorn app:app` - this will fail!  
**✅ Correct:** The FastAPI application is in `main.py`, so you must use `main:app`

The API will be available at `http://127.0.0.1:8000`  
API documentation will be at `http://127.0.0.1:8000/docs`

## Troubleshooting

### Error: "No module named uvicorn"
- Make sure you installed requirements: `pip3 install -r requirements.txt`
- Try using specific Python version: `python3.11 -m uvicorn main:app --reload --port 8000`

### Error: "Could not import module 'app'"
- You're using the wrong command!
- Use `main:app` instead of `app:app`
- The format is `filename:variable_name` and our FastAPI app is in `main.py`

## API Endpoints

- POST `/upload` - Upload and analyze resume
- POST `/analyze` - Analyze previously uploaded resume
- GET `/jobs` - Get all available job roles
- GET `/skills` - Get all skills in database
- GET `/candidates` - Get all analyzed candidates
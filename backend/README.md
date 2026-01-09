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
# AgriBot 🌱

AgriBot is an agriculture-focused RAG chatbot that answers questions using agricultural datasets.

## Features

- Agriculture-focused AI chatbot
- RAG-based semantic search
- DuckDB for structured agricultural data
- Chroma/vector search
- NVIDIA Nemotron for generation
- FastAPI backend
- HTML/CSS/JavaScript frontend

## Setup

### 1. Clone repository

git clone YOUR_REPOSITORY_URL
cd RAG

### 2. Create virtual environment

python -m venv venv

### 3. Activate

Windows:

venv\Scripts\activate

### 4. Install dependencies

pip install -r requirements.txt

### 5. Add NVIDIA API key

Create `.env`:

NVIDIA_API_KEY=your_api_key_here

### 6. Add datasets

Place the agricultural CSV files inside:

dataset/

### 7. Build database

python database_builder.py

### 8. Build vector database

python vector_builder.py

### 9. Start API

uvicorn api:app --reload --port 8000

### 10. Open frontend

Open index.html in your browser.
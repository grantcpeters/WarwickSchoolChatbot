# WarwickSchoolChatbot

An AI-powered chatbot for Warwick Prep School that uses the school website (warwickprep.com) as its knowledge base.

## Overview

This project crawls the school website (including nested pages and PDF documents), indexes the content into Azure AI Search with vector embeddings, and exposes a RAG-powered chat API backed by Azure OpenAI. A React web app and mobile-responsive frontend allow parents, students, and staff to ask questions in natural language.

## Architecture

```
warwickprep.com (website + PDFs)
        │
        ▼
┌───────────────────┐
│  Crawler / Ingestion │  src/crawler/
│  - Scrapy web spider  │
│  - PDF extractor      │
│  - Azure Blob Storage │
└───────────┬───────────┘
            │
            ▼
┌───────────────────┐
│  Indexer           │  src/indexer/
│  - Document Intelligence (PDF OCR) │
│  - Text chunking   │
│  - Azure OpenAI Embeddings         │
│  - Azure AI Search index           │
└───────────┬───────────┘
            │
            ▼
┌───────────────────┐
│  Chat API (FastAPI)│  src/api/
│  - RAG pipeline   │
│  - Azure OpenAI   │
│  - Conversation history │
└───────────┬───────────┘
            │
            ▼
┌───────────────────┐
│  Frontend (React) │  src/frontend/
│  - Chat UI        │
│  - Mobile responsive │
└───────────────────┘
```

## Project Structure

```
WarwickSchoolChatbot/
├── README.md                  # This file
├── .env.example               # Environment variable template
├── .gitignore
├── requirements.txt           # Python dependencies
├── data/
│   └── sample/                # Sample data for testing
├── docs/                      # Documentation and design notes
├── infra/                     # Azure infrastructure (Bicep)
├── models/                    # Model configs / prompts
├── notebooks/                 # Jupyter notebooks for experimentation
├── src/
│   ├── crawler/               # Web crawler and PDF downloader
│   ├── indexer/               # Chunking, embedding and indexing pipeline
│   ├── api/                   # FastAPI chat backend
│   ├── chatbot/               # RAG logic, prompt templates
│   └── frontend/              # React chat UI
└── tests/                     # Unit and integration tests
```

## Tech Stack

| Layer | Technology |
|---|---|
| Web Crawler | Python / Scrapy + BeautifulSoup |
| PDF Extraction | Azure AI Document Intelligence |
| Vector Store | Azure AI Search (vector + semantic) |
| Embeddings | Azure OpenAI `text-embedding-3-small` |
| LLM | Azure OpenAI `gpt-4o-mini` |
| Backend API | Python / FastAPI |
| Frontend | React + TypeScript |
| Storage | Azure Blob Storage |
| Infrastructure | Azure Bicep |

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 20+
- Azure CLI (`az login`)
- Azure subscription with the following resources provisioned (see `infra/`)

### 1. Clone and install

```bash
git clone https://github.com/grantcpeters/WarwickSchoolChatbot.git
cd WarwickSchoolChatbot
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your Azure resource details
```

### 3. Run the crawler

```bash
python src/crawler/run_crawler.py
```

### 4. Run the indexer

```bash
python src/indexer/run_indexer.py
```

### 5. Start the API

```bash
uvicorn src.api.main:app --reload
```

### 6. Start the frontend

```bash
cd src/frontend
npm install
npm start
```

## Environment Variables

See `.env.example` for all required variables.

## License

TBD

## Author

Grant Peters

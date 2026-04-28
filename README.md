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

## Cloud Deployment

The recommended first rollout is:

- One Azure App Service hosting the FastAPI backend and the built React frontend.
- Existing Azure OpenAI, Azure AI Search, Azure Blob Storage, and Azure Document Intelligence services.
- One scheduled GitHub Actions workflow that runs the crawler and indexer weekly on GitHub-hosted runners.

This keeps the chatbot fully cloud-hosted without requiring any local machine to stay online.

### Azure Hosting Model

- Users access one App Service URL.
- The React frontend is built during deployment and served by FastAPI in production.
- The weekly workflow runs `python src/crawler/run_crawler.py` and then `python src/indexer/run_indexer.py`.
- The live Warwick Prep website remains the canonical content source.

### GitHub Configuration

Add these repository secrets for Azure login:

- `AZURE_CLIENT_ID`
- `AZURE_TENANT_ID`
- `AZURE_SUBSCRIPTION_ID`

Add these repository variables for deployment and scheduled ingestion:

- `AZURE_RESOURCE_GROUP`
- `AZURE_WEBAPP_NAME`

## Local Development

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

The crawler keeps its state in Azure Blob Storage and is designed to be re-run on a schedule.
After the first crawl, subsequent runs reuse stored `ETag` or `Last-Modified` values, skip unchanged pages, upload only changed content, and remove blobs for pages that disappeared from the live site.

### 4. Run the indexer

```bash
python src/indexer/run_indexer.py
```

The indexer also runs incrementally. It reads crawler state, re-embeds only changed sources, and removes stale search documents when content is updated or removed.

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

## Operating Model

- The live Warwick Prep website is the canonical content source.
- Azure Blob Storage holds the latest crawled snapshot plus crawl and index state.
- Azure AI Search is the retrieval layer used by the chatbot at runtime.
- A practical cloud schedule is: run the crawler weekly in GitHub Actions, then run the indexer immediately after it in the same workflow.

This keeps chatbot responses fast and stable while still following changes made on the live website.

## License

TBD

## Author

Grant Peters

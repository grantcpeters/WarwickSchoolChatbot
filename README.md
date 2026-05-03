# WarwickSchoolChatbot

An AI-powered chatbot for Warwick Prep School that uses the school website (warwickprep.com) and weekly parent letters as its knowledge base.

## Overview

This project crawls the school website (including nested pages and PDF documents), ingests forwarded weekly parent letters from a monitored mailbox, indexes all content into Azure AI Search with vector embeddings, and exposes a RAG-powered chat API backed by Azure OpenAI. A React web app allows parents, students, and staff to ask questions in natural language.

## Architecture

```
warwickprep.com (website + PDFs)          Forwarded weekly parent letters
        │                                           │
        ▼                                           ▼
┌───────────────────┐              ┌─────────────────────────────┐
│  Crawler          │  src/crawler/ │  Letter Ingester            │
│  - BeautifulSoup  │              │  - Microsoft Graph REST API  │
│  - PDF extractor  │              │  - Outlook.com mailbox       │
│  - Azure Blob     │              │  - scripts/ingest_letters.py │
└───────────┬───────┘              └──────────────┬──────────────┘
            │                                     │
            └──────────────┬──────────────────────┘
                           ▼
┌──────────────────────────────────────────────────┐
│  Indexer                             src/indexer/ │
│  - Azure AI Document Intelligence (PDF OCR)       │
│  - Text chunking                                  │
│  - Azure OpenAI Embeddings                        │
│  - Azure AI Search index                          │
└───────────────────────────┬──────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────┐
│  Chat API (FastAPI)              src/api/         │
│  - RAG pipeline (hybrid search + re-ranking)      │
│  - Supplemental searches for staff/dates/menus    │
│  - Azure OpenAI (gpt-4o-mini)                     │
│  - Conversation history (last 10 turns)           │
└───────────────────────────┬──────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────┐
│  Frontend (React)            src/frontend/        │
│  - Chat UI, mobile-responsive                     │
│  - Served as static files by FastAPI              │
└──────────────────────────────────────────────────┘
```

## Project Structure

```
WarwickSchoolChatbot/
├── README.md
├── .env.example               # Environment variable template
├── .gitignore
├── requirements.txt
├── data/
│   └── sample/                # Sample data for testing
├── docs/
│   └── portal-usage-monitoring.md   # Azure Monitor workbook notes
├── infra/
│   └── main.bicep             # Azure infrastructure (Bicep)
├── scripts/
│   ├── ingest_letters.py      # Weekly letter ingester (Microsoft Graph)
│   ├── setup_letter_oauth.py  # One-time OAuth2 token setup
│   ├── eval_live_prompts.py   # Live prompt evaluation (Log Analytics)
│   ├── usage_report.py        # Usage report from Log Analytics
│   ├── deploy_workbook.py     # Deploy Azure Monitor workbook
│   └── check_menu.py          # Debug helper for menu retrieval
├── src/
│   ├── crawler/               # Web crawler and PDF downloader
│   ├── indexer/               # Chunking, embedding and indexing pipeline
│   ├── api/                   # FastAPI chat backend
│   ├── chatbot/               # RAG logic, prompt templates
│   │   └── rag_pipeline.py    # Hybrid search, re-ranking, keyword boosts
│   ├── frontend/              # React chat UI
│   └── shared/
│       ├── azure_credentials.py
│       └── blob_json_state.py
└── tests/                     # Unit and integration tests
```

## Tech Stack

| Layer            | Technology                                |
| ---------------- | ----------------------------------------- |
| Web Crawler      | Python / BeautifulSoup                    |
| PDF Extraction   | Azure AI Document Intelligence            |
| Vector Store     | Azure AI Search (hybrid vector + keyword) |
| Embeddings       | Azure OpenAI `text-embedding-3-small`     |
| LLM              | Azure OpenAI `gpt-4o-mini`                |
| Backend API      | Python / FastAPI                          |
| Frontend         | React + TypeScript                        |
| Storage          | Azure Blob Storage                        |
| Letter Ingestion | Microsoft Graph REST API (Outlook.com)    |
| Infrastructure   | Azure Bicep                               |
| CI/CD            | GitHub Actions                            |

## Cloud Deployment

- One Azure App Service hosting the FastAPI backend with the built React frontend served as static files.
- Existing Azure OpenAI, Azure AI Search, Azure Blob Storage, and Azure Document Intelligence services.
- Weekly GitHub Actions workflow that runs on Fridays at 17:00 UTC (end of school day, after letters are sent Thursday): ingests letters → crawls website → indexes content.

### GitHub Configuration

**Secrets** required:

| Secret                            | Description                                                     |
| --------------------------------- | --------------------------------------------------------------- |
| `AZURE_CLIENT_ID`                 | Service principal for Azure login (OIDC)                        |
| `AZURE_TENANT_ID`                 | Azure tenant                                                    |
| `AZURE_SUBSCRIPTION_ID`           | Azure subscription                                              |
| `AZURE_OPENAI_ENDPOINT`           | Azure OpenAI endpoint URL                                       |
| `AZURE_OPENAI_API_KEY`            | Azure OpenAI API key                                            |
| `AZURE_SEARCH_ENDPOINT`           | Azure AI Search endpoint URL                                    |
| `AZURE_SEARCH_API_KEY`            | Azure AI Search admin key                                       |
| `AZURE_DOC_INTELLIGENCE_ENDPOINT` | Document Intelligence endpoint                                  |
| `AZURE_DOC_INTELLIGENCE_KEY`      | Document Intelligence key                                       |
| `AZURE_STORAGE_CONNECTION_STRING` | Blob Storage connection string                                  |
| `LETTER_EMAIL`                    | Outlook.com address for forwarded letters                       |
| `LETTER_OAUTH_REFRESH_TOKEN`      | Graph API OAuth2 refresh token (set by `setup_letter_oauth.py`) |

**Variables** required:

| Variable                            | Description                               |
| ----------------------------------- | ----------------------------------------- |
| `AZURE_RESOURCE_GROUP`              | Resource group for App Service deployment |
| `AZURE_WEBAPP_NAME`                 | App Service name                          |
| `AZURE_OPENAI_CHAT_DEPLOYMENT`      | Chat model deployment name                |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Embedding model deployment name           |
| `AZURE_SEARCH_INDEX_NAME`           | Search index name                         |
| `AZURE_STORAGE_ACCOUNT_NAME`        | Storage account name                      |
| `AZURE_STORAGE_CONTAINER_RAW`       | Blob container for raw crawled HTML       |
| `AZURE_STORAGE_CONTAINER_PDF`       | Blob container for PDFs                   |
| `CRAWL_START_URL`                   | Root URL to begin crawl                   |
| `CRAWL_ALLOWED_DOMAINS`             | Comma-separated allowed domains           |

## Local Development

### Prerequisites

- Python 3.11+
- Node.js 20+
- Azure CLI (`az login`)
- Azure subscription with resources provisioned (see `infra/`)

### 1. Clone and install

```bash
git clone https://github.com/grantcpeters/WarwickSchoolChatbot.git
cd WarwickSchoolChatbot
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
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

The crawler keeps state in Azure Blob Storage and runs incrementally — subsequent runs skip unchanged pages and remove stale blobs.

### 4. Run the indexer

```bash
python src/indexer/run_indexer.py
```

The indexer also runs incrementally, re-embedding only changed sources and removing stale search documents.

### 5. (Optional) Ingest weekly letters

```bash
# One-time OAuth2 setup — follow the browser prompt
python scripts/setup_letter_oauth.py

# Then ingest any unprocessed letters
python scripts/ingest_letters.py
```

Set up a server-side forwarding rule in your email client: any email **From @warwickschools.co.uk** with **Subject containing "Weekly Letter"** → forward to the `LETTER_EMAIL` Outlook.com address.

### 6. Start the API

```bash
uvicorn src.api.main:app --reload
```

### 7. Start the frontend

```bash
cd src/frontend
npm install
npm start
```

## Useful Scripts

| Script                          | Purpose                                                                                      |
| ------------------------------- | -------------------------------------------------------------------------------------------- |
| `scripts/eval_live_prompts.py`  | Fetch real prompts from Log Analytics, run each through the pipeline, verdict PASS/WARN/FAIL |
| `scripts/usage_report.py`       | Human-readable usage summary (unique visitors, total hits, feedback, top prompts)            |
| `scripts/deploy_workbook.py`    | Deploy the Azure Monitor workbook for usage dashboards                                       |
| `scripts/check_menu.py`         | Debug helper — checks what the menu retrieval returns for a given date                       |
| `scripts/setup_letter_oauth.py` | One-time Microsoft Graph OAuth2 token setup                                                  |

Run eval against the last 48 hours of real prompts:

```bash
.venv\Scripts\python.exe scripts/eval_live_prompts.py --hours 48
```

## Operating Model

- The live Warwick Prep website is the primary canonical content source.
- Forwarded weekly parent letters add current school news and upcoming events that aren't on the website.
- Azure Blob Storage holds the latest crawled snapshot plus crawl/index state.
- Azure AI Search is the retrieval layer used by the chatbot at runtime.
- The GitHub Actions workflow runs every **Friday at 17:00 UTC**: letters → crawl → index.
- The RAG pipeline uses hybrid search (vector + keyword) with supplemental keyword-only searches to ensure key pages (staff list, term dates, lunch menus) are always retrieved for the relevant query types.

## License

TBD

## Author

Grant Peters

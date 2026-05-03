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

| Layer          | Technology                            |
| -------------- | ------------------------------------- |
| Web Crawler    | Python / Scrapy + BeautifulSoup       |
| PDF Extraction | Azure AI Document Intelligence        |
| Vector Store   | Azure AI Search (vector + semantic)   |
| Embeddings     | Azure OpenAI `text-embedding-3-small` |
| LLM            | Azure OpenAI `gpt-4o-mini`            |
| Backend API    | Python / FastAPI                      |
| Frontend       | React + TypeScript                    |
| Storage        | Azure Blob Storage                    |
| Infrastructure | Azure Bicep                           |

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

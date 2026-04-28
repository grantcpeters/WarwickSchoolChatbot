# Azure Deployment Plan

## 1. Overview
- Status: Ready for Validation
- Mode: MODIFY
- Goal: Move WarwickSchoolChatbot to a cloud-only Azure runtime so crawler, indexer, API, and frontend do not require local execution.

## 1.1 Recommendation
- Recommended recipe: Bicep + GitHub Actions
- Recommended hosting model: one Azure App Service for the user-facing web application, plus scheduled GitHub Actions workflows for crawler and indexer.
- Rationale: this is the simplest cloud-only option for a first rollout, keeps deployment easy to understand, avoids container orchestration, and still leaves a clean path to scale later.

## 2. Current Workspace
- Repository: WarwickSchoolChatbot
- Existing infrastructure: infra/main.bicep
- Existing app components: Python crawler, Python indexer, FastAPI API, React frontend

## 2.1 Current Gaps
- No App Service hosting resources exist yet for the frontend or API.
- The FastAPI backend does not yet serve the built frontend in production.
- The crawler and indexer are script entry points only and are not yet scheduled in the cloud.
- No GitHub Actions deployment assets existed yet for cloud deployment orchestration.

## 3. Proposed Azure Architecture
- One Azure App Service Plan and one Azure App Service named `web` running the FastAPI application and serving the built React frontend as static assets.
- One scheduled GitHub Actions workflow running `python src/crawler/run_crawler.py` weekly on GitHub-hosted runners.
- One scheduled GitHub Actions workflow step running `python src/indexer/run_indexer.py` immediately after the crawler in the same scheduled workflow.
- Existing Azure OpenAI, Azure AI Search, Azure Document Intelligence, and Azure Blob Storage remain the data and AI dependencies.
- Managed Identity should be preferred for storage access where practical; existing key-based fallbacks may remain for compatibility.

## 3.1 Content Flow
- The live Warwick Prep website remains the canonical source.
- The scheduled crawler job updates Blob Storage incrementally.
- The scheduled or manually chained indexer job updates Azure AI Search incrementally.
- The web Container App serves the frontend and API from Azure only.

## 3.2 Why This Option
- Simpler than Container Apps or Functions for a 5-user pilot.
- Easier to maintain than converting crawler and indexer into Azure Functions immediately.
- Keeps the public app surface to one Azure-hosted URL while moving scheduled ingestion fully off local machines.

## 4. Delivery Plan
- Update the FastAPI app to serve the built React frontend in production.
- Expand `infra/main.bicep` to provision Azure App Service Plan, Azure Web App, and supporting app settings while retaining the existing AI resources.
- Add deployment workflow assets for App Service deployment and scheduled crawl/index execution in GitHub Actions.
- Update documentation for cloud-only operations, including scheduled crawler/indexer behavior.

## 5. Validation Plan
- Validate Bicep syntax and deployment preview.
- Validate Python application import/build path for the App Service startup command.
- Validate frontend production build.
- Validate deployed web endpoint and scheduled GitHub workflow configuration.

## 6. Open Questions
- Azure subscription and preferred region
- Whether the weekly schedule should be Sunday night UTC by default.
- Whether to keep service API keys in configuration initially or move all possible services to managed identity immediately.

## 7. Validation Proof
- Not started

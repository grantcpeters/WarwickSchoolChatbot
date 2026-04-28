"""
Warwick Prep School — Indexer Pipeline
Reads raw HTML and PDF blobs from Azure Blob Storage,
extracts text (using Azure Document Intelligence for PDFs),
chunks it, generates embeddings, and upserts into Azure AI Search.
"""

import os
import hashlib
import logging
from typing import Iterator

from azure.storage.blob import BlobServiceClient
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticSearch,
    SemanticPrioritizedFields,
    SemanticField,
)
from bs4 import BeautifulSoup
from openai import AzureOpenAI
from dotenv import load_dotenv

from src.shared.azure_credentials import get_blob_service_client, get_service_credential
from src.shared.blob_json_state import load_json_state, save_json_state

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CONTAINER_RAW = os.getenv("AZURE_STORAGE_CONTAINER_RAW", "webcrawl-raw")
CONTAINER_PDF = os.getenv("AZURE_STORAGE_CONTAINER_PDF", "webcrawl-pdf")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "warwickprep-content")
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "64"))
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = 1536
CRAWLER_STATE_BLOB_NAME = "_crawler_state.json"
INDEX_STATE_BLOB_NAME = "_index_state.json"


def get_blob_client() -> BlobServiceClient:
    return get_blob_service_client()


def get_openai_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-01",
    )


def get_doc_intel_client() -> DocumentIntelligenceClient:
    return DocumentIntelligenceClient(
        endpoint=os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT"),
        credential=get_service_credential("AZURE_DOC_INTELLIGENCE_KEY"),
    )


def get_search_client() -> SearchClient:
    return SearchClient(
        endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        index_name=INDEX_NAME,
        credential=get_service_credential("AZURE_SEARCH_API_KEY"),
    )


def get_search_index_client() -> SearchIndexClient:
    return SearchIndexClient(
        endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        credential=get_service_credential("AZURE_SEARCH_API_KEY"),
    )


def ensure_search_index(index_client: SearchIndexClient) -> None:
    """Create the Azure AI Search index if it does not exist."""
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="source_url", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="source_type", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="chunk_index", type=SearchFieldDataType.Int32),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBEDDING_DIMENSIONS,
            vector_search_profile_name="hnsw-profile",
        ),
    ]
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
        profiles=[VectorSearchProfile(name="hnsw-profile", algorithm_configuration_name="hnsw")],
    )
    semantic_config = SemanticConfiguration(
        name="default",
        prioritized_fields=SemanticPrioritizedFields(
            content_fields=[SemanticField(field_name="content")]
        ),
    )
    index = SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
        semantic_search=SemanticSearch(configurations=[semantic_config]),
    )
    existing = [i.name for i in index_client.list_indexes()]
    if INDEX_NAME not in existing:
        index_client.create_index(index)
        log.info("Created search index: %s", INDEX_NAME)
    else:
        log.info("Search index already exists: %s", INDEX_NAME)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Simple token-aware chunker (word-based approximation)."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]


def embed(openai_client: AzureOpenAI, texts: list[str]) -> list[list[float]]:
    response = openai_client.embeddings.create(model=EMBEDDING_DEPLOYMENT, input=texts)
    return [item.embedding for item in response.data]


def extract_html_text(html_bytes: bytes) -> str:
    soup = BeautifulSoup(html_bytes, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


def extract_pdf_text(doc_intel: DocumentIntelligenceClient, pdf_bytes: bytes) -> str:
    poller = doc_intel.begin_analyze_document(
        "prebuilt-read",
        AnalyzeDocumentRequest(bytes_source=pdf_bytes),
    )
    result = poller.result()
    return "\n".join([p.content for p in result.paragraphs]) if result.paragraphs else ""


def iter_blobs(blob_client: BlobServiceClient, container: str) -> Iterator[tuple[str, bytes]]:
    cc = blob_client.get_container_client(container)
    for blob in cc.list_blobs():
        data = cc.download_blob(blob.name).readall()
        yield blob.name, data


def download_blob(blob_client: BlobServiceClient, container: str, blob_name: str) -> bytes:
    return blob_client.get_container_client(container).download_blob(blob_name).readall()


def make_doc_id(source: str, chunk_index: int) -> str:
    raw = f"{source}#{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


def delete_source_documents(search_client: SearchClient, source: str, chunk_count: int) -> None:
    if chunk_count <= 0:
        return

    search_client.delete_documents(
        documents=[{"id": make_doc_id(source, chunk_index)} for chunk_index in range(chunk_count)]
    )


def load_crawl_entries(blob_client: BlobServiceClient) -> list[dict]:
    state = load_json_state(blob_client, CONTAINER_RAW, CRAWLER_STATE_BLOB_NAME)
    pages = state.get("pages", {})
    if pages:
        return list(pages.values())

    entries: list[dict] = []
    for blob_name, _ in iter_blobs(blob_client, CONTAINER_RAW):
        if blob_name.startswith("_"):
            continue
        entries.append({"blob_name": blob_name, "source_type": "html", "url": blob_name})
    for blob_name, _ in iter_blobs(blob_client, CONTAINER_PDF):
        if blob_name.startswith("_"):
            continue
        entries.append({"blob_name": blob_name, "source_type": "pdf", "url": blob_name})
    return entries


def run_indexer() -> None:
    blob_client = get_blob_client()
    openai_client = get_openai_client()
    doc_intel_client = get_doc_intel_client()
    index_client = get_search_index_client()
    search_client = get_search_client()
    crawl_entries = load_crawl_entries(blob_client)
    previous_index_state = load_json_state(blob_client, CONTAINER_RAW, INDEX_STATE_BLOB_NAME).get("sources", {})
    next_index_state: dict[str, dict] = {}

    ensure_search_index(index_client)

    batch: list[dict] = []

    def flush_batch():
        if batch:
            search_client.upload_documents(batch)
            log.info("Indexed %d chunks", len(batch))
            batch.clear()

    for entry in crawl_entries:
        blob_name = entry["blob_name"]
        source_type = entry["source_type"]
        source_url = entry.get("url", blob_name)
        source_hash = entry.get("content_hash")
        previous_source_state = previous_index_state.get(blob_name)
        if previous_source_state and previous_source_state.get("content_hash") == source_hash:
            next_index_state[blob_name] = previous_source_state
            continue

        if previous_source_state:
            delete_source_documents(search_client, blob_name, previous_source_state.get("chunk_count", 0))

        log.info("Processing %s: %s", source_type.upper(), blob_name)
        container = CONTAINER_PDF if source_type == "pdf" else CONTAINER_RAW
        data = download_blob(blob_client, container, blob_name)
        text = extract_pdf_text(doc_intel_client, data) if source_type == "pdf" else extract_html_text(data)
        chunks = chunk_text(text)
        if not chunks:
            next_index_state[blob_name] = {
                "chunk_count": 0,
                "content_hash": source_hash,
                "source_type": source_type,
                "source_url": source_url,
            }
            continue

        vectors = embed(openai_client, chunks)
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            batch.append({
                "id": make_doc_id(blob_name, i),
                "content": chunk,
                "source_url": source_url,
                "source_type": source_type,
                "chunk_index": i,
                "content_vector": vector,
            })
        next_index_state[blob_name] = {
            "chunk_count": len(chunks),
            "content_hash": source_hash,
            "source_type": source_type,
            "source_url": source_url,
        }
        if len(batch) >= 100:
            flush_batch()

    removed_sources = set(previous_index_state) - {entry["blob_name"] for entry in crawl_entries}
    for removed_source in removed_sources:
        delete_source_documents(
            search_client,
            removed_source,
            previous_index_state[removed_source].get("chunk_count", 0),
        )

    flush_batch()
    save_json_state(
        blob_client,
        CONTAINER_RAW,
        INDEX_STATE_BLOB_NAME,
        {"sources": next_index_state},
    )
    log.info("Indexing complete.")


if __name__ == "__main__":
    run_indexer()

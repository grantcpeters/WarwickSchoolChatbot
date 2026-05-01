"""
Warwick Prep School — Indexer Pipeline
Reads raw HTML and PDF blobs from Azure Blob Storage,
extracts text (using Azure Document Intelligence for PDFs),
chunks it, generates embeddings, and upserts into Azure AI Search.
"""

import os
import re
import time
import hashlib
import logging
from datetime import date
from email.utils import parsedate_to_datetime
from typing import Iterator

from azure.core.exceptions import ResourceNotFoundError
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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

CONTAINER_RAW = os.getenv("AZURE_STORAGE_CONTAINER_RAW", "webcrawl-raw")
CONTAINER_PDF = os.getenv("AZURE_STORAGE_CONTAINER_PDF", "webcrawl-pdf")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "warwickprep-content")
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "64"))
EMBEDDING_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"
)
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
        max_retries=6,
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
        SimpleField(
            name="source_url", type=SearchFieldDataType.String, filterable=True
        ),
        SimpleField(
            name="source_type", type=SearchFieldDataType.String, filterable=True
        ),
        SimpleField(
            name="page_title", type=SearchFieldDataType.String, filterable=False
        ),
        SimpleField(name="chunk_index", type=SearchFieldDataType.Int32),
        SimpleField(
            name="last_modified",
            type=SearchFieldDataType.DateTimeOffset,
            filterable=True,
            sortable=True,
        ),
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
        profiles=[
            VectorSearchProfile(
                name="hnsw-profile", algorithm_configuration_name="hnsw"
            )
        ],
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
        # create_or_update_index safely adds new fields to an existing index.
        index_client.create_or_update_index(index)
        log.info("Updated search index schema: %s", INDEX_NAME)


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
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


# At 120K TPM: 16 chunks × ~680 tokens ≈ 10K tokens per batch.
# 6s delay → ~10 batches/min → ~100K TPM, safely under the 120K limit.
# The while-True retry in embed() handles any residual 429s without giving up.
EMBED_BATCH_SIZE = 16
EMBED_BATCH_DELAY = 6.0
CHECKPOINT_EVERY = 10  # save index state to blob after every N pages


def embed(openai_client: AzureOpenAI, texts: list[str]) -> list[list[float]]:
    """Embed texts in batches. Retries indefinitely on 429 with a 65-second back-off."""
    from openai import RateLimitError

    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        while True:
            try:
                response = openai_client.embeddings.create(
                    model=EMBEDDING_DEPLOYMENT, input=batch
                )
                all_embeddings.extend(item.embedding for item in response.data)
                break
            except RateLimitError:
                log.warning("429 rate limit — sleeping 65 s before retry")
                time.sleep(65)
        if i + EMBED_BATCH_SIZE < len(texts):
            time.sleep(EMBED_BATCH_DELAY)
    return all_embeddings


def _parse_last_modified(lm: str | None) -> str | None:
    """Parse an HTTP Last-Modified header (RFC 7231) to ISO 8601 for Azure Search."""
    if not lm:
        return None
    try:
        return parsedate_to_datetime(lm).isoformat()
    except Exception:
        return None


def extract_html_text(html_bytes: bytes) -> str:
    """Flat text extraction (kept for debugging / backward compat; indexer uses extract_html_chunks)."""
    soup = BeautifulSoup(html_bytes, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


# Heading levels used for breadcrumb context injection.
_HEADING_LEVELS: dict[str, int] = {"h1": 1, "h2": 2, "h3": 3, "h4": 4}


def _format_heading_prefix(stack: dict) -> str:
    """Build 'H1 text > H2 text > H3 text' from the current heading stack."""
    return " > ".join(stack[k] for k in sorted(stack))


# Matches academic-year date ranges like "2024/2025" or "2024/25".
_ACAD_YEAR_RE = re.compile(r"\b(20\d{2})[/\u2013-](20)?(\d{2})\b")


def _current_academic_year_start() -> int:
    """Return the start calendar year of the current academic year.

    Academic years run September–July, so before September we are still
    in the year that started the previous September.
    """
    today = date.today()
    return today.year if today.month >= 9 else today.year - 1


def _is_stale_section(prefix: str) -> bool:
    """Return True if a heading prefix explicitly names a past academic year.

    Examples (current academic year = 2025/26):
      "Fees for the Academic Year 2024/2025"  -> True  (stale)
      "Last Year's Fees"                       -> True  (stale)
      "Fees for the Academic Year 2025/2026"  -> False (current)
      "Our History"                            -> False (no year pattern)
    """
    lower = prefix.lower()
    if "last year" in lower or "previous year" in lower:
        return True
    current_start = _current_academic_year_start()
    for m in _ACAD_YEAR_RE.finditer(prefix):
        if int(m.group(1)) < current_start:
            return True
    return False


def extract_html_chunks(
    html_bytes: bytes,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Parse HTML and return chunks with heading breadcrumb prepended to each one.

    Each chunk is prefixed with its section context, e.g.:
      "[Main School Fees (Per term) > Fees for the Academic Year 2025/2026] £4,509 …"

    This ensures every retrieved chunk carries unambiguous section/year context so the
    LLM cannot confuse 2024/25 fee data with 2025/26 data, or last-year events with
    current ones.

    Tables are emitted as pipe-separated rows so fee/uniform/schedule tables are
    kept together as coherent blocks rather than being split mid-row.
    """
    soup = BeautifulSoup(html_bytes, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # ── Step 1: linear pass → (kind, data) items in document order ──────────
    # 'heading': (level: int, text: str)
    # 'text':    str
    items: list[tuple] = []
    _seen: set[int] = set()

    def _mark_subtree(node) -> None:
        _seen.add(id(node))
        if hasattr(node, "descendants"):
            for d in node.descendants:
                _seen.add(id(d))

    def _collect(node) -> None:
        if id(node) in _seen:
            return
        if not hasattr(node, "name") or node.name is None:
            return
        name = node.name.lower()

        if name in _HEADING_LEVELS:
            text = node.get_text(strip=True)
            if text:
                items.append(("heading", _HEADING_LEVELS[name], text))
            _mark_subtree(node)

        elif name == "table":
            # Emit the whole table as one text block with pipe-separated cells.
            rows = []
            for row in node.find_all("tr"):
                cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
                row_text = " | ".join(c for c in cells if c)
                if row_text:
                    rows.append(row_text)
            if rows:
                items.append(("text", " ".join(rows)))
            _mark_subtree(node)

        elif name in {"p", "li", "blockquote", "dd", "dt", "caption"}:
            # Emit leaf-level content blocks; recurse into any that contain headings.
            if not node.find(["h1", "h2", "h3", "h4"]):
                text = node.get_text(separator=" ", strip=True)
                if text:
                    items.append(("text", text))
                _mark_subtree(node)
            else:
                for child in node.children:
                    _collect(child)

        else:
            for child in node.children:
                _collect(child)

    body = soup.find("body") or soup
    for child in body.children:
        _collect(child)

    # ── Step 2: group items by current heading context ───────────────────────
    heading_stack: dict[int, str] = {}
    current_words: list[str] = []
    sections: list[tuple[str, str]] = []  # (prefix, text)

    for item in items:
        if item[0] == "heading":
            _, level, text = item
            if current_words:
                sections.append(
                    (_format_heading_prefix(heading_stack), " ".join(current_words))
                )
                current_words = []
            # Replace this level (and any deeper levels) with the new heading.
            for k in list(heading_stack.keys()):
                if k >= level:
                    del heading_stack[k]
            heading_stack[level] = text
        else:
            current_words.extend(item[1].split())

    if current_words:
        sections.append(
            (_format_heading_prefix(heading_stack), " ".join(current_words))
        )

    # ── Step 3: chunk each section, prepending its heading prefix ────────────
    # Sections whose heading explicitly names a past academic year are suppressed:
    # they contain stale fee/event data that would confuse the LLM.
    all_chunks: list[str] = []
    for prefix, text in sections:
        if _is_stale_section(prefix):
            log.debug("Suppressing stale section: %.80s", prefix)
            continue
        for sub in chunk_text(text, chunk_size, overlap):
            all_chunks.append(f"[{prefix}] {sub}" if prefix else sub)

    return [c for c in all_chunks if c.strip()]


# Suffixes stripped from <title> tags to get a clean page name.
_TITLE_STRIP = [
    " | Warwick Preparatory School",
    " - Warwick Preparatory School",
    " | Warwick Prep School",
    " - Warwick Prep School",
    " | Warwick Prep",
    " - Warwick Prep",
]


def extract_html_title(html_bytes: bytes) -> str:
    """Return the page title, stripping the site name suffix."""
    soup = BeautifulSoup(html_bytes, "lxml")
    tag = soup.find("title")
    if tag:
        title = tag.get_text(strip=True)
        for suffix in _TITLE_STRIP:
            if suffix.lower() in title.lower():
                title = title[: title.lower().index(suffix.lower())].strip()
                break
        if title:
            return title
    h1 = soup.find("h1")
    return h1.get_text(strip=True) if h1 else ""


def extract_pdf_text(doc_intel: DocumentIntelligenceClient, pdf_bytes: bytes) -> str:
    poller = doc_intel.begin_analyze_document(
        "prebuilt-read",
        AnalyzeDocumentRequest(bytes_source=pdf_bytes),
    )
    result = poller.result()
    return (
        "\n".join([p.content for p in result.paragraphs]) if result.paragraphs else ""
    )


def iter_blobs(
    blob_client: BlobServiceClient, container: str
) -> Iterator[tuple[str, bytes]]:
    cc = blob_client.get_container_client(container)
    for blob in cc.list_blobs():
        data = cc.download_blob(blob.name).readall()
        yield blob.name, data


def download_blob(
    blob_client: BlobServiceClient, container: str, blob_name: str
) -> bytes:
    return (
        blob_client.get_container_client(container).download_blob(blob_name).readall()
    )


def make_doc_id(source: str, chunk_index: int) -> str:
    raw = f"{source}#{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


def delete_source_documents(
    search_client: SearchClient, source: str, chunk_count: int
) -> None:
    if chunk_count <= 0:
        return

    search_client.delete_documents(
        documents=[
            {"id": make_doc_id(source, chunk_index)}
            for chunk_index in range(chunk_count)
        ]
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
        entries.append(
            {"blob_name": blob_name, "source_type": "html", "url": blob_name}
        )
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
    # Load existing index state — already-indexed pages are skipped (content-hash match).
    # This makes runs resumable: a failed run picks up where it left off.
    index_state: dict[str, dict] = load_json_state(
        blob_client, CONTAINER_RAW, INDEX_STATE_BLOB_NAME
    ).get("sources", {})

    ensure_search_index(index_client)

    batch: list[dict] = []
    pages_since_checkpoint = 0

    def flush_batch():
        if batch:
            search_client.upload_documents(batch)
            log.info("Indexed %d chunks", len(batch))
            batch.clear()

    def checkpoint():
        """Persist current index state so the next run can skip already-done pages."""
        flush_batch()
        save_json_state(
            blob_client, CONTAINER_RAW, INDEX_STATE_BLOB_NAME, {"sources": index_state}
        )
        log.info("Checkpoint saved (%d sources indexed so far)", len(index_state))

    for entry in crawl_entries:
        blob_name = entry["blob_name"]
        source_type = entry["source_type"]
        source_url = entry.get("url", blob_name)
        source_hash = entry.get("content_hash")
        existing = index_state.get(blob_name)
        if existing and existing.get("content_hash") == source_hash:
            # Already indexed and unchanged — skip.
            continue

        if existing:
            delete_source_documents(
                search_client, blob_name, existing.get("chunk_count", 0)
            )

        log.info("Processing %s: %s", source_type.upper(), blob_name)
        container = CONTAINER_PDF if source_type == "pdf" else CONTAINER_RAW
        try:
            data = download_blob(blob_client, container, blob_name)
        except ResourceNotFoundError:
            log.warning(
                "Blob not found in storage, skipping: %s/%s", container, blob_name
            )
            continue
        text = (
            extract_pdf_text(doc_intel_client, data) if source_type == "pdf" else None
        )
        page_title = extract_html_title(data) if source_type == "html" else ""
        chunks = chunk_text(text) if source_type == "pdf" else extract_html_chunks(data)
        if not chunks:
            index_state[blob_name] = {
                "chunk_count": 0,
                "content_hash": source_hash,
                "source_type": source_type,
                "source_url": source_url,
            }
            continue

        vectors = embed(openai_client, chunks)
        lm = _parse_last_modified(entry.get("last_modified"))
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            batch.append(
                {
                    "id": make_doc_id(blob_name, i),
                    "content": chunk,
                    "source_url": source_url,
                    "source_type": source_type,
                    "page_title": page_title,
                    "chunk_index": i,
                    "content_vector": vector,
                    "last_modified": lm,
                }
            )
        index_state[blob_name] = {
            "chunk_count": len(chunks),
            "content_hash": source_hash,
            "source_type": source_type,
            "source_url": source_url,
        }
        pages_since_checkpoint += 1
        if len(batch) >= 100:
            flush_batch()
        if pages_since_checkpoint >= CHECKPOINT_EVERY:
            checkpoint()
            pages_since_checkpoint = 0

    # Clean up pages that were removed from the crawl
    crawl_blob_names = {entry["blob_name"] for entry in crawl_entries}
    removed_sources = set(index_state) - crawl_blob_names
    for removed_source in removed_sources:
        delete_source_documents(
            search_client,
            removed_source,
            index_state[removed_source].get("chunk_count", 0),
        )
        del index_state[removed_source]

    # Final checkpoint
    checkpoint()
    log.info("Indexing complete. %d sources in index.", len(index_state))


if __name__ == "__main__":
    run_indexer()

"""
RAG pipeline — retrieves relevant chunks from Azure AI Search
and calls Azure OpenAI to generate a grounded response.
"""

import os
from datetime import date
from typing import AsyncIterator

from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv

from src.shared.azure_credentials import get_service_credential

load_dotenv()

INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "warwickprep-content")
TOP_K = int(os.getenv("RAG_TOP_K", "5"))
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

_SYSTEM_PROMPT_TEMPLATE = """\
You are a helpful assistant for Warwick Prep School.
Answer questions using only the information provided in the context below.
If the answer is not in the context, say you don't have that information and suggest \
the user contact the school directly at admissions@warwickprep.com or call 01926 491545.
Always be friendly, concise, and helpful.

Today's date is {today}.
If any information in the context refers to specific dates or events (such as open mornings, \
term dates, or school events) and those dates have already passed, clearly say that this \
information may be out of date and recommend checking warwickprep.com or contacting the \
school directly for current details.
"""


def _build_system_prompt() -> str:
    return _SYSTEM_PROMPT_TEMPLATE.format(today=date.today().strftime("%d %B %Y"))


def _get_openai() -> AsyncAzureOpenAI:
    return AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-01",
    )


def _get_search() -> SearchClient:
    return SearchClient(
        endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        index_name=INDEX_NAME,
        credential=get_service_credential("AZURE_SEARCH_API_KEY"),
    )


async def _embed(openai: AsyncAzureOpenAI, text: str) -> list[float]:
    resp = await openai.embeddings.create(model=EMBEDDING_DEPLOYMENT, input=text)
    return resp.data[0].embedding


async def retrieve(query: str) -> list[dict]:
    """Hybrid search: vector + keyword."""
    openai = _get_openai()
    vector = await _embed(openai, query)

    async with _get_search() as search_client:
        vector_query = VectorizedQuery(
            vector=vector,
            k_nearest_neighbors=TOP_K,
            fields="content_vector",
        )
        # Try selecting page_title (added in a later schema migration).
        # Fall back gracefully if the field doesn't exist yet in the index.
        select_fields = ["content", "source_url", "source_type", "page_title"]
        try:
            results = await search_client.search(
                search_text=query,
                vector_queries=[vector_query],
                top=TOP_K,
                select=select_fields,
            )
            return [
                {
                    "content": r["content"],
                    "source": r["source_url"],
                    "type": r["source_type"],
                    "title": r.get("page_title") or "",
                }
                async for r in results
            ]
        except Exception as e:
            if "page_title" not in str(e):
                raise
            # page_title field not yet in index schema — retry without it
            results = await search_client.search(
                search_text=query,
                vector_queries=[vector_query],
                top=TOP_K,
                select=["content", "source_url", "source_type"],
            )
            return [
                {
                    "content": r["content"],
                    "source": r["source_url"],
                    "type": r["source_type"],
                    "title": "",
                }
                async for r in results
            ]


async def chat(query: str, history: list[dict] | None = None) -> AsyncIterator[str]:
    """
    Stream a RAG response.
    history: list of {"role": "user"|"assistant", "content": "..."}
    """
    chunks = await retrieve(query)
    context = "\n\n---\n\n".join(c["content"] for c in chunks)

    # Deduplicate sources by URL, preserving the best title found.
    # Only include sources that look like real URLs (skip stale index docs with hash IDs).
    seen: dict[str, str] = {}
    for c in chunks:
        url = c["source"]
        if not url.startswith("http"):
            continue
        if url not in seen or (not seen[url] and c["title"]):
            seen[url] = c["title"]

    messages = [{"role": "system", "content": _build_system_prompt() + f"\n\nContext:\n{context}"}]
    if history:
        messages.extend(history[-10:])  # limit history to last 10 turns
    messages.append({"role": "user", "content": query})

    openai = _get_openai()
    stream = await openai.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=messages,
        stream=True,
        temperature=0.2,
        max_tokens=1024,
    )
    async for event in stream:
        delta = event.choices[0].delta.content if event.choices else None
        if delta:
            yield delta

    # Yield sources as a final metadata chunk — format: url:::title pairs joined by ,
    if seen:
        parts = ",".join(f"{url}:::{title}" for url, title in seen.items())
        yield f"\n\n__sources__:{parts}"

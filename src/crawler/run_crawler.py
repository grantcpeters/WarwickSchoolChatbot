"""
Warwick Prep School — Web Crawler
Crawls https://www.warwickprep.com/ recursively, downloads HTML pages and PDFs,
and stores them in Azure Blob Storage.

Uses asyncio + aiohttp with a bounded concurrency semaphore for speed.
"""

import json
import os
import re
import asyncio
import hashlib
import logging
from urllib.parse import urljoin, urlparse
from datetime import datetime, timezone

import aiohttp
from bs4 import BeautifulSoup
from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob.aio import BlobServiceClient as AsyncBlobServiceClient
from dotenv import load_dotenv

from src.shared.azure_credentials import get_blob_service_client
from src.shared.blob_json_state import load_json_state, save_json_state

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

START_URL = os.getenv("CRAWL_START_URL", "https://www.warwickprep.com/")
ALLOWED_DOMAIN = os.getenv("CRAWL_ALLOWED_DOMAINS", "warwickprep.com")
MAX_DEPTH = int(os.getenv("CRAWL_MAX_DEPTH", "5"))
CRAWL_DELAY = float(os.getenv("CRAWL_DELAY_SECONDS", "0.1"))
MAX_CONCURRENCY = int(os.getenv("CRAWL_MAX_CONCURRENCY", "10"))
CONTAINER_RAW = os.getenv("AZURE_STORAGE_CONTAINER_RAW", "webcrawl-raw")
CONTAINER_PDF = os.getenv("AZURE_STORAGE_CONTAINER_PDF", "webcrawl-pdf")
STATE_BLOB_NAME = "_crawler_state.json"

# URL path prefixes to skip — prevents historical/hidden pages from being indexed.
# Comma-separated list; matched against the URL path (lowercased, leading slash stripped).
# Default excludes 'hiddenarea/' (previous-year fee archives) and known stale event
# archive pages that would otherwise confuse open-morning / event queries.
_EXCLUDE_PATH_PREFIXES: list[str] = [
    p.strip().lower()
    for p in os.getenv(
        "CRAWL_EXCLUDE_PATHS",
        "hiddenarea,openmorningsept2024,openmorningold,apologies",
    ).split(",")
    if p.strip()
]


def _should_crawl(url: str) -> bool:
    """Return False if this URL's path starts with any excluded prefix."""
    path = urlparse(url).path.lower().lstrip("/")
    return not any(path.startswith(prefix) for prefix in _EXCLUDE_PATH_PREFIXES)


def _get_async_blob_client() -> AsyncBlobServiceClient:
    """Build an async BlobServiceClient mirroring the sync credential logic."""
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if conn_str:
        return AsyncBlobServiceClient.from_connection_string(conn_str)
    account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    if not account_name:
        raise ValueError(
            "Set AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_NAME."
        )
    from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential

    return AsyncBlobServiceClient(
        account_url=f"https://{account_name}.blob.core.windows.net",
        credential=AsyncDefaultAzureCredential(),
    )


def url_to_blob_name(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/") or "index"
    path = re.sub(r"[^a-zA-Z0-9/_.-]", "_", path)
    if parsed.query:
        q_hash = hashlib.md5(parsed.query.encode()).hexdigest()[:8]
        path = f"{path}_{q_hash}"
    return path


def hash_content(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_links(url: str, html_bytes: bytes) -> list[str]:
    soup = BeautifulSoup(html_bytes, "lxml")
    links: list[str] = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        absolute = urljoin(url, href)
        parsed = urlparse(absolute)
        if ALLOWED_DOMAIN in parsed.netloc:
            links.append(absolute)
    return sorted(set(links))


async def crawl_async() -> None:
    # Load previous state using sync client (only needed once at startup)
    sync_blob_client = get_blob_service_client()
    previous_state: dict = load_json_state(
        sync_blob_client, CONTAINER_RAW, STATE_BLOB_NAME
    ).get("pages", {})

    current_state: dict[str, dict] = {}
    new_count = 0
    updated_count = 0
    unchanged_count = 0
    # Protect shared state from concurrent modification
    state_lock = asyncio.Lock()

    visited: set[str] = set()
    visited_lock = asyncio.Lock()

    queue: asyncio.Queue[tuple[str, int]] = asyncio.Queue()
    await queue.put((START_URL, 0))
    # Track how many workers are active so we know when we're done
    in_flight = 0
    in_flight_lock = asyncio.Lock()

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def ensure_containers(client: AsyncBlobServiceClient) -> None:
        for name in (CONTAINER_RAW, CONTAINER_PDF):
            cc = client.get_container_client(name)
            if not await cc.exists():
                await cc.create_container()
                log.info("Created container: %s", name)

    async def upload_blob_async(
        client: AsyncBlobServiceClient,
        container: str,
        blob_name: str,
        data: bytes,
        content_type: str,
    ) -> None:
        cc = client.get_container_client(container)
        await cc.upload_blob(blob_name, data, overwrite=True, content_type=content_type)
        log.info("Uploaded %d bytes -> %s/%s", len(data), container, blob_name)

    async def blob_exists_async(
        client: AsyncBlobServiceClient, container: str, blob_name: str
    ) -> bool:
        return (
            await client.get_container_client(container)
            .get_blob_client(blob_name)
            .exists()
        )

    async def delete_blob_if_exists_async(
        client: AsyncBlobServiceClient, container: str, blob_name: str
    ) -> None:
        cc = client.get_container_client(container)
        try:
            await cc.delete_blob(blob_name)
            log.info("Deleted blob %s/%s", container, blob_name)
        except ResourceNotFoundError:
            pass

    async def process_url(
        session: aiohttp.ClientSession,
        client: AsyncBlobServiceClient,
        url: str,
        depth: int,
    ) -> None:
        nonlocal new_count, updated_count, unchanged_count

        previous_entry = previous_state.get(url, {})

        if not _should_crawl(url):
            log.debug("Skipping excluded path: %s", url)
            return

        headers: dict[str, str] = {"User-Agent": "WarwickSchoolChatbot-Crawler/1.0"}
        if previous_entry.get("etag"):
            headers["If-None-Match"] = previous_entry["etag"]
        elif previous_entry.get("last_modified"):
            headers["If-Modified-Since"] = previous_entry["last_modified"]

        try:
            log.info("[depth=%d] Fetching %s", depth, url)
            async with session.get(
                url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status == 304:
                    async with state_lock:
                        unchanged_count += 1
                        current_state[url] = {
                            **previous_entry,
                            "last_checked_at": datetime.now(timezone.utc).isoformat(),
                        }
                    for saved_link in previous_entry.get("links", []):
                        async with visited_lock:
                            if saved_link not in visited:
                                visited.add(saved_link)
                                await queue.put((saved_link, depth + 1))
                    return
                if resp.status >= 400:
                    log.warning("HTTP %d for %s", resp.status, url)
                    return
                data = await resp.read()
                content_type_header = resp.headers.get("content-type", "")
                etag = resp.headers.get("etag")
                last_modified = resp.headers.get("last-modified")
        except Exception as exc:
            log.warning("Failed to fetch %s: %s", url, exc)
            return

        is_pdf = (
            "pdf" in content_type_header
            or url.lower().endswith(".pdf")
            or "type=pdf" in url.lower()
            or data[:4] == b"%PDF"
        )
        source_type = "pdf" if is_pdf else "html"
        blob_name = url_to_blob_name(url) + (".pdf" if is_pdf else ".html")
        content_hash = hash_content(data)
        links = normalize_links(url, data) if not is_pdf else []

        current_entry = {
            "blob_name": blob_name,
            "content_hash": content_hash,
            "etag": etag,
            "last_checked_at": datetime.now(timezone.utc).isoformat(),
            "last_modified": last_modified,
            "links": links,
            "source_type": source_type,
            "url": url,
        }

        target_container = CONTAINER_PDF if is_pdf else CONTAINER_RAW
        blob_ok = await blob_exists_async(client, target_container, blob_name)
        if previous_entry.get("content_hash") == content_hash and blob_ok:
            async with state_lock:
                unchanged_count += 1
        else:
            if not blob_ok:
                log.warning(
                    "Blob missing from storage, re-uploading: %s/%s",
                    target_container,
                    blob_name,
                )
            await upload_blob_async(
                client,
                target_container,
                blob_name,
                data,
                "application/pdf" if is_pdf else "text/html",
            )
            async with state_lock:
                if url in previous_state:
                    updated_count += 1
                else:
                    new_count += 1

        async with state_lock:
            current_state[url] = current_entry

        # Enqueue new links (skip excluded paths)
        for absolute in links:
            if depth + 1 <= MAX_DEPTH and _should_crawl(absolute):
                async with visited_lock:
                    if absolute not in visited:
                        visited.add(absolute)
                        await queue.put((absolute, depth + 1))

        await asyncio.sleep(CRAWL_DELAY)

    async def worker(
        session: aiohttp.ClientSession, client: AsyncBlobServiceClient
    ) -> None:
        nonlocal in_flight
        while True:
            try:
                url, depth = queue.get_nowait()
            except asyncio.QueueEmpty:
                # Nothing queued right now; check if all work is done
                async with in_flight_lock:
                    if in_flight == 0:
                        return
                await asyncio.sleep(0.05)
                continue

            async with in_flight_lock:
                in_flight += 1
            try:
                async with semaphore:
                    await process_url(session, client, url, depth)
            finally:
                async with in_flight_lock:
                    in_flight -= 1
                queue.task_done()

    async with _get_async_blob_client() as async_client:
        await ensure_containers(async_client)

        # Mark start URL as visited
        visited.add(START_URL)

        connector = aiohttp.TCPConnector(limit=MAX_CONCURRENCY + 5)
        async with aiohttp.ClientSession(connector=connector) as session:
            workers = [
                asyncio.create_task(worker(session, async_client))
                for _ in range(MAX_CONCURRENCY)
            ]
            await asyncio.gather(*workers)

        # Clean up removed pages
        removed_urls = set(previous_state) - set(current_state)
        for removed_url in removed_urls:
            removed_entry = previous_state[removed_url]
            target_container = (
                CONTAINER_PDF
                if removed_entry.get("source_type") == "pdf"
                else CONTAINER_RAW
            )
            await delete_blob_if_exists_async(
                async_client, target_container, removed_entry["blob_name"]
            )
            log.info("Removed stale page: %s", removed_url)

    removed_count = len(removed_urls)

    save_json_state(
        sync_blob_client,
        CONTAINER_RAW,
        STATE_BLOB_NAME,
        {
            "pages": current_state,
            "stats": {
                "new": new_count,
                "updated": updated_count,
                "unchanged": unchanged_count,
                "removed": removed_count,
                "total": len(current_state),
            },
        },
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    stats_path = os.path.join(os.getcwd(), "crawl-stats.json")
    with open(stats_path, "w") as _f:
        json.dump(
            {
                "timestamp": timestamp,
                "max_depth": MAX_DEPTH,
                "total": len(current_state),
                "new": new_count,
                "updated": updated_count,
                "unchanged": unchanged_count,
                "removed": removed_count,
                "removed_urls": sorted(removed_urls),
            },
            _f,
            indent=2,
        )

    separator = "=" * 52
    log.info(separator)
    log.info("CRAWL COMPLETE — %s", timestamp)
    log.info("  Max depth configured : %d levels (0–%d)", MAX_DEPTH, MAX_DEPTH)
    log.info("  Pages active         : %d", len(current_state))
    log.info("  + New pages          : %d", new_count)
    log.info("  ~ Updated pages      : %d", updated_count)
    log.info("  = Unchanged pages    : %d", unchanged_count)
    log.info("  - Removed (stale)    : %d", removed_count)
    if removed_urls:
        for url in sorted(removed_urls):
            log.info("      - %s", url)
    log.info(separator)


def crawl() -> None:
    asyncio.run(crawl_async())


if __name__ == "__main__":
    crawl()

"""
Warwick Prep School — Web Crawler
Crawls https://www.warwickprep.com/ recursively, downloads HTML pages and PDFs,
and stores them in Azure Blob Storage.
"""

import os
import re
import hashlib
import logging
from urllib.parse import urljoin, urlparse
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup
from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

from src.shared.azure_credentials import get_blob_service_client
from src.shared.blob_json_state import load_json_state, save_json_state

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

START_URL = os.getenv("CRAWL_START_URL", "https://www.warwickprep.com/")
ALLOWED_DOMAIN = os.getenv("CRAWL_ALLOWED_DOMAINS", "warwickprep.com")
MAX_DEPTH = int(os.getenv("CRAWL_MAX_DEPTH", "5"))
CRAWL_DELAY = float(os.getenv("CRAWL_DELAY_SECONDS", "1"))
CONTAINER_RAW = os.getenv("AZURE_STORAGE_CONTAINER_RAW", "webcrawl-raw")
CONTAINER_PDF = os.getenv("AZURE_STORAGE_CONTAINER_PDF", "webcrawl-pdf")
STATE_BLOB_NAME = "_crawler_state.json"


def get_blob_client() -> BlobServiceClient:
    return get_blob_service_client()


def ensure_container(client: BlobServiceClient, name: str) -> None:
    cc = client.get_container_client(name)
    if not cc.exists():
        cc.create_container()
        log.info("Created container: %s", name)


def upload_blob(client: BlobServiceClient, container: str, blob_name: str, data: bytes, content_type: str) -> None:
    cc = client.get_container_client(container)
    cc.upload_blob(blob_name, data, overwrite=True, content_type=content_type)
    log.info("Uploaded %s -> %s/%s", len(data), container, blob_name)


def delete_blob_if_exists(client: BlobServiceClient, container: str, blob_name: str) -> None:
    cc = client.get_container_client(container)
    try:
        cc.delete_blob(blob_name)
        log.info("Deleted blob %s/%s", container, blob_name)
    except ResourceNotFoundError:
        return


def url_to_blob_name(url: str) -> str:
    """Convert a URL to a safe blob name."""
    parsed = urlparse(url)
    path = parsed.path.strip("/") or "index"
    # Replace unsafe chars
    path = re.sub(r"[^a-zA-Z0-9/_.-]", "_", path)
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


def crawl() -> None:
    session = requests.Session()
    session.headers["User-Agent"] = "WarwickSchoolChatbot-Crawler/1.0"

    visited: set[str] = set()
    queue: list[tuple[str, int]] = [(START_URL, 0)]

    blob_client = get_blob_client()
    ensure_container(blob_client, CONTAINER_RAW)
    ensure_container(blob_client, CONTAINER_PDF)
    previous_state = load_json_state(blob_client, CONTAINER_RAW, STATE_BLOB_NAME).get("pages", {})
    current_state: dict[str, dict] = {}
    changed_count = 0
    unchanged_count = 0

    import time

    while queue:
        url, depth = queue.pop(0)

        if url in visited or depth > MAX_DEPTH:
            continue
        visited.add(url)

        previous_entry = previous_state.get(url, {})
        headers: dict[str, str] = {}
        if previous_entry.get("etag"):
            headers["If-None-Match"] = previous_entry["etag"]
        elif previous_entry.get("last_modified"):
            headers["If-Modified-Since"] = previous_entry["last_modified"]

        try:
            log.info("[depth=%d] Fetching %s", depth, url)
            resp = session.get(url, headers=headers, timeout=15)
            if resp.status_code == 304:
                unchanged_count += 1
                current_state[url] = {
                    **previous_entry,
                    "last_checked_at": datetime.now(timezone.utc).isoformat(),
                }
                for saved_link in previous_entry.get("links", []):
                    if saved_link not in visited:
                        queue.append((saved_link, depth + 1))
                continue
            resp.raise_for_status()
        except Exception as exc:
            log.warning("Failed to fetch %s: %s", url, exc)
            continue

        content_type = resp.headers.get("content-type", "")
        is_pdf = "pdf" in content_type or url.lower().endswith(".pdf")
        source_type = "pdf" if is_pdf else "html"
        blob_name = url_to_blob_name(url) + (".pdf" if is_pdf else ".html")
        content_hash = hash_content(resp.content)
        links = normalize_links(url, resp.content) if not is_pdf else []
        current_entry = {
            "blob_name": blob_name,
            "content_hash": content_hash,
            "etag": resp.headers.get("etag"),
            "last_checked_at": datetime.now(timezone.utc).isoformat(),
            "last_modified": resp.headers.get("last-modified"),
            "links": links,
            "source_type": source_type,
            "url": url,
        }

        if previous_entry.get("content_hash") == content_hash:
            unchanged_count += 1
        else:
            changed_count += 1
            target_container = CONTAINER_PDF if is_pdf else CONTAINER_RAW
            upload_blob(
                blob_client,
                target_container,
                blob_name,
                resp.content,
                "application/pdf" if is_pdf else "text/html",
            )

        current_state[url] = current_entry
        for absolute in links:
            if absolute not in visited:
                queue.append((absolute, depth + 1))

        time.sleep(CRAWL_DELAY)

    removed_urls = set(previous_state) - set(current_state)
    for removed_url in removed_urls:
        removed_entry = previous_state[removed_url]
        target_container = CONTAINER_PDF if removed_entry.get("source_type") == "pdf" else CONTAINER_RAW
        delete_blob_if_exists(blob_client, target_container, removed_entry["blob_name"])

    save_json_state(
        blob_client,
        CONTAINER_RAW,
        STATE_BLOB_NAME,
        {
            "pages": current_state,
            "stats": {
                "changed": changed_count,
                "removed": len(removed_urls),
                "total": len(current_state),
                "unchanged": unchanged_count,
            },
        },
    )

    log.info(
        "Crawl complete. Active=%d changed=%d unchanged=%d removed=%d",
        len(current_state),
        changed_count,
        unchanged_count,
        len(removed_urls),
    )


if __name__ == "__main__":
    crawl()

"""
Warwick Prep School — Web Crawler
Crawls https://www.warwickprep.com/ recursively, downloads HTML pages and PDFs,
and stores them in Azure Blob Storage.
"""

import os
import re
import logging
from urllib.parse import urljoin, urlparse
from io import BytesIO

import requests
from bs4 import BeautifulSoup
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

START_URL = os.getenv("CRAWL_START_URL", "https://www.warwickprep.com/")
ALLOWED_DOMAIN = os.getenv("CRAWL_ALLOWED_DOMAINS", "warwickprep.com")
MAX_DEPTH = int(os.getenv("CRAWL_MAX_DEPTH", "5"))
CRAWL_DELAY = float(os.getenv("CRAWL_DELAY_SECONDS", "1"))
CONTAINER_RAW = os.getenv("AZURE_STORAGE_CONTAINER_RAW", "webcrawl-raw")
CONTAINER_PDF = os.getenv("AZURE_STORAGE_CONTAINER_PDF", "webcrawl-pdf")


def get_blob_client() -> BlobServiceClient:
    account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    credential = DefaultAzureCredential()
    return BlobServiceClient(
        account_url=f"https://{account_name}.blob.core.windows.net",
        credential=credential,
    )


def ensure_container(client: BlobServiceClient, name: str) -> None:
    cc = client.get_container_client(name)
    if not cc.exists():
        cc.create_container()
        log.info("Created container: %s", name)


def upload_blob(client: BlobServiceClient, container: str, blob_name: str, data: bytes, content_type: str) -> None:
    cc = client.get_container_client(container)
    cc.upload_blob(blob_name, data, overwrite=True, content_settings={"content_type": content_type})
    log.info("Uploaded %s -> %s/%s", len(data), container, blob_name)


def url_to_blob_name(url: str) -> str:
    """Convert a URL to a safe blob name."""
    parsed = urlparse(url)
    path = parsed.path.strip("/") or "index"
    # Replace unsafe chars
    path = re.sub(r"[^a-zA-Z0-9/_.-]", "_", path)
    return path


def crawl() -> None:
    session = requests.Session()
    session.headers["User-Agent"] = "WarwickSchoolChatbot-Crawler/1.0"

    visited: set[str] = set()
    queue: list[tuple[str, int]] = [(START_URL, 0)]

    blob_client = get_blob_client()
    ensure_container(blob_client, CONTAINER_RAW)
    ensure_container(blob_client, CONTAINER_PDF)

    import time

    while queue:
        url, depth = queue.pop(0)

        if url in visited or depth > MAX_DEPTH:
            continue
        visited.add(url)

        try:
            log.info("[depth=%d] Fetching %s", depth, url)
            resp = session.get(url, timeout=15)
            resp.raise_for_status()
        except Exception as exc:
            log.warning("Failed to fetch %s: %s", url, exc)
            continue

        content_type = resp.headers.get("content-type", "")
        blob_name = url_to_blob_name(url)

        if "pdf" in content_type or url.lower().endswith(".pdf"):
            upload_blob(blob_client, CONTAINER_PDF, blob_name + ".pdf", resp.content, "application/pdf")
        elif "html" in content_type:
            upload_blob(blob_client, CONTAINER_RAW, blob_name + ".html", resp.content, "text/html")

            # Parse links for further crawling
            soup = BeautifulSoup(resp.content, "lxml")
            for tag in soup.find_all("a", href=True):
                href = tag["href"].strip()
                absolute = urljoin(url, href)
                parsed = urlparse(absolute)
                if ALLOWED_DOMAIN in parsed.netloc and absolute not in visited:
                    queue.append((absolute, depth + 1))

        time.sleep(CRAWL_DELAY)

    log.info("Crawl complete. Pages visited: %d", len(visited))


if __name__ == "__main__":
    crawl()

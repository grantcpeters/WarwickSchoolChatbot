"""
Warwick Prep School — Weekly Letter Ingester

Polls an Outlook.com mailbox via Microsoft Graph REST API for forwarded
weekly letters, parses each one, and indexes each item as a chunk into
Azure AI Search.

Authentication uses OAuth2 (Microsoft Graph Mail.Read scope).
Run scripts/setup_letter_oauth.py once to obtain and store the
refresh token, then this script runs silently in GitHub Actions.

One-time setup
--------------
1. Create a free Outlook.com account, e.g. warwick-letters@outlook.com.
   No IMAP/POP3 settings required — Graph API works without them.

2. Run the one-time OAuth2 setup script:
     python scripts/setup_letter_oauth.py
   Follow the prompt (open a URL, enter a code, sign in).  The script
   saves LETTER_OAUTH_REFRESH_TOKEN to GitHub secrets and .env.

3. In your personal email client, create a server-side forwarding rule:
     Condition : From contains @warwickschools.co.uk
                 AND Subject contains "Weekly Letter"
     Action    : Forward to warwick-letters@outlook.com
   The rule runs on Microsoft's servers — no device needs to be open.

4. Ensure these are set in .env / GitHub secrets:
     LETTER_EMAIL=warwick-letters@outlook.com
     LETTER_OAUTH_REFRESH_TOKEN=<set by setup_letter_oauth.py>

The script is idempotent: it tracks indexed email Message-IDs in
Azure Blob Storage (_letter_index_state.json) and skips any letter
it has already processed.
"""

import email as email_lib
import logging
import os
import re
from datetime import datetime, timezone
from email import policy as email_policy

import msal
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from src.indexer.run_indexer import (
    embed,
    ensure_search_index,
    get_openai_client,
    get_search_client,
    get_search_index_client,
    make_doc_id,
)
from src.shared.azure_credentials import get_blob_service_client
from src.shared.blob_json_state import load_json_state, save_json_state

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────
LETTER_EMAIL = os.getenv("LETTER_EMAIL")
# OAuth2 credentials (set by scripts/setup_letter_oauth.py)
_OAUTH_CLIENT_ID = os.getenv(
    "LETTER_OAUTH_CLIENT_ID", "d983cb90-69c5-4d9c-b6b2-cd2a413c241a"
)
_OAUTH_AUTHORITY = "https://login.microsoftonline.com/consumers"
_OAUTH_SCOPES = ["https://graph.microsoft.com/Mail.ReadWrite"]
LETTER_OAUTH_REFRESH_TOKEN = os.getenv("LETTER_OAUTH_REFRESH_TOKEN")
_GRAPH_BASE = "https://graph.microsoft.com/v1.0"
# Only process emails whose subject contains this string (case-insensitive)
LETTER_SUBJECT_FILTER = os.getenv("LETTER_SUBJECT_FILTER", "Weekly Letter")
# Only process emails whose From address contains this domain (leave empty to skip check)
LETTER_FROM_FILTER = os.getenv("LETTER_FROM_FILTER", "")

LETTER_STATE_BLOB = "_letter_index_state.json"
CONTAINER_RAW = os.getenv("AZURE_STORAGE_CONTAINER_RAW", "webcrawl-raw")

# ── Parsing helpers ───────────────────────────────────────────

_MONTH_MAP = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

# Matches "Thursday 30 April 2026" style dates
_DATE_RE = re.compile(
    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+"
    r"(\d{1,2})\s+"
    r"(january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\s+"
    r"(\d{4})\b",
    re.IGNORECASE,
)

# Matches item lines: "Whole School: Some reminder text here." (inline format)
# Category label: starts with a letter, up to ~60 chars, may contain spaces/digits
_ITEM_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 &/,.()|\-]{0,60}):\s+(.+)$")

# Matches a label-only line: "Whole School:" (text follows on the next line)
_LABEL_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 &/,.()|\-]{0,60}):\s*$")

# Lines that signal the end of the letter body
_STOP_PHRASES = (
    "kind regards",
    "yours sincerely",
    "warwick prep",
    "warwick schools foundation",
)

# Lines to skip entirely (salutation / section headers)
_SKIP_LINES = {
    "dear parents",
    "dear parents,",
    "dear parent",
    "dear parent,",
    "reminders & information:",
    "reminders and information:",
    "reminders & information",
    "reminders and information",
}


def _extract_text(msg) -> str:
    """Return the best plain-text representation of an email message."""
    plain = None
    html = None

    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/plain" and plain is None:
                plain = part.get_content()
            elif ct == "text/html" and html is None:
                html = part.get_content()
    else:
        ct = msg.get_content_type()
        if ct == "text/plain":
            plain = msg.get_content()
        elif ct == "text/html":
            html = msg.get_content()

    if plain:
        return plain
    if html:
        soup = BeautifulSoup(html, "lxml")
        return soup.get_text(separator="\n", strip=True)
    return ""


def _parse_letter(text: str) -> dict | None:
    """Parse a weekly letter body into structured data.

    Returns a dict with keys:
        date        : datetime (UTC)
        date_iso    : str  e.g. "2026-04-30"
        year_groups : str  e.g. "NURSERY, RECEPTION, YEAR 1"
        items       : list of (category: str, text: str) tuples

    Returns None if the text cannot be identified as a weekly letter.
    """
    lines = [line.strip() for line in text.splitlines()]

    year_groups: str | None = None
    date: datetime | None = None
    items: list[tuple[str, str]] = []
    current_category: str | None = None
    current_parts: list[str] = []

    def _flush() -> None:
        if current_category and current_parts:
            items.append((current_category, " ".join(current_parts)))

    for line in lines:
        if not line:
            continue

        lower = line.lower()

        # Year-group header — skip ALL lines until we find this
        # (prevents forwarded email headers from triggering stop phrases)
        if year_groups is None:
            if re.match(
                r"^(NURSERY|RECEPTION|YEAR\s+\d|PRE.?PREP|PREP)", line, re.IGNORECASE
            ):
                year_groups = line
            continue  # ignore everything before the year-group header

        # Date line — skip until we find the date too
        if date is None:
            m = _DATE_RE.search(line)
            if m:
                date = datetime(
                    int(m.group(3)),
                    _MONTH_MAP[m.group(2).lower()],
                    int(m.group(1)),
                    tzinfo=timezone.utc,
                )
            continue  # ignore everything between year-group header and date

        # ── We are now in the letter body ────────────────────────

        # Stop at the sign-off
        if any(lower.startswith(p) for p in _STOP_PHRASES):
            _flush()
            current_category = None
            break

        # Skip salutation / section headers
        if lower in _SKIP_LINES:
            continue

        # Item line: "Category: text…" (inline) or "Category:" (text on next line)
        m = _ITEM_RE.match(line)
        if m:
            _flush()
            current_category = m.group(1).strip()
            current_parts = [m.group(2).strip()]
        else:
            m2 = _LABEL_RE.match(line)
            if m2 and m2.group(1).strip().lower() not in _SKIP_LINES:
                _flush()
                current_category = m2.group(1).strip()
                current_parts = []
            elif current_category:
                # Continuation of the previous item (wrapped line)
                current_parts.append(line)

    _flush()

    if not date or not items:
        log.debug("Letter parse failed — date=%s items=%d", date, len(items))
        return None

    return {
        "date": date,
        "date_iso": date.strftime("%Y-%m-%d"),
        "year_groups": year_groups or "Whole School",
        "items": items,
    }


def _make_source_url(date_iso: str, year_groups: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", year_groups.lower()).strip("-")
    return f"letter://warwickprep/{date_iso}/{slug}"


def _make_chunks(parsed: dict) -> list[str]:
    """One chunk per letter item, each prefixed with letter context."""
    prefix = f"[Weekly Letter - {parsed['date_iso']} - {parsed['year_groups']}]"
    return [f"{prefix} {category}: {text}" for category, text in parsed["items"]]


# ── Main entry point ──────────────────────────────────────────


def _get_access_token() -> str:
    """Use the stored refresh token to silently obtain a fresh OAuth2 access token."""
    msal_app = msal.PublicClientApplication(
        _OAUTH_CLIENT_ID, authority=_OAUTH_AUTHORITY
    )
    result = msal_app.acquire_token_by_refresh_token(
        LETTER_OAUTH_REFRESH_TOKEN, scopes=_OAUTH_SCOPES
    )
    if "access_token" not in result:
        raise RuntimeError(
            f"OAuth2 token refresh failed: {result.get('error_description', result)}"
        )
    return result["access_token"]


def _graph_get(path: str, token: str, params: dict | None = None) -> dict:
    """Make a GET request to Microsoft Graph and return the JSON response."""
    resp = requests.get(
        f"{_GRAPH_BASE}{path}",
        headers={"Authorization": f"Bearer {token}"},
        params=params,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _graph_patch(path: str, token: str, body: dict) -> None:
    """Make a PATCH request to Microsoft Graph (e.g. to mark message as read)."""
    resp = requests.patch(
        f"{_GRAPH_BASE}{path}",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=body,
        timeout=30,
    )
    resp.raise_for_status()


def _get_message_body(message_id: str, token: str) -> str:
    """Fetch the full MIME source of a message and extract text."""
    # Get the message with body content
    data = _graph_get(f"/me/messages/{message_id}", token, params={"$select": "body,bodyPreview"})
    body = data.get("body", {})
    content_type = body.get("contentType", "text")
    content = body.get("content", "")
    if content_type == "html":
        soup = BeautifulSoup(content, "lxml")
        return soup.get_text(separator="\n", strip=True)
    return content


def fetch_and_index_letters() -> int:
    """Fetch unread letters via Microsoft Graph, process them, return count indexed."""
    if not LETTER_EMAIL or not LETTER_OAUTH_REFRESH_TOKEN:
        raise ValueError(
            "LETTER_EMAIL and LETTER_OAUTH_REFRESH_TOKEN environment variables must be set. "
            "Run scripts/setup_letter_oauth.py first."
        )

    blob_client = get_blob_service_client()
    openai_client = get_openai_client()
    index_client = get_search_index_client()
    search_client = get_search_client()

    # Ensure the search index exists (safe to call repeatedly)
    ensure_search_index(index_client)

    # Load state: maps email Message-ID -> metadata about what was indexed
    state: dict = load_json_state(blob_client, CONTAINER_RAW, LETTER_STATE_BLOB).get(
        "letters", {}
    )

    indexed_count = 0
    access_token = _get_access_token()

    # Query Graph for unread messages; subject filtering happens in Python
    data = _graph_get(
        "/me/messages",
        access_token,
        params={
            "$filter": "isRead eq false",
            "$select": "id,subject,from,receivedDateTime,internetMessageId,body",
            "$top": "50",
            "$orderby": "receivedDateTime desc",
        },
    )
    messages = data.get("value", [])
    log.info("Found %d unread message(s) to check", len(messages))

    for msg_meta in messages:
        subject = msg_meta.get("subject", "")
        from_info = msg_meta.get("from", {}).get("emailAddress", {})
        from_addr = from_info.get("address", "")
        graph_id = msg_meta["id"]
        message_id = msg_meta.get("internetMessageId") or graph_id

        # Filter by subject keyword
        if LETTER_SUBJECT_FILTER.lower() not in subject.lower():
            log.debug("Skipping (subject mismatch): %s", subject)
            continue

        # Filter by sender domain
        if (
            LETTER_FROM_FILTER
            and LETTER_FROM_FILTER.lower() not in from_addr.lower()
        ):
            log.debug("Skipping (sender mismatch): %s", from_addr)
            continue

        # Skip letters already indexed
        if message_id in state:
            log.info("Already indexed, skipping: %s", subject)
            _graph_patch(f"/me/messages/{graph_id}", access_token, {"isRead": True})
            continue

        log.info("Processing: %s (from %s)", subject, from_addr)
        body_data = msg_meta.get("body", {})
        content_type = body_data.get("contentType", "text")
        content = body_data.get("content", "")
        if content_type == "html":
            soup = BeautifulSoup(content, "lxml")
            text = soup.get_text(separator="\n", strip=True)
        else:
            text = content
        parsed = _parse_letter(text)

        if not parsed:
            log.warning(
                "Could not parse letter body for: %s — marking read and skipping",
                subject,
            )
            _graph_patch(f"/me/messages/{graph_id}", access_token, {"isRead": True})
            continue

        source_url = _make_source_url(parsed["date_iso"], parsed["year_groups"])
        page_title = (
            f"Weekly Letter - {parsed['year_groups']} - {parsed['date_iso']}"
        )
        chunks = _make_chunks(parsed)
        log.info(
            "  Parsed %d item(s) from letter dated %s (%s)",
            len(chunks),
            parsed["date_iso"],
            parsed["year_groups"],
        )

        vectors = embed(openai_client, chunks)
        lm = parsed["date"].isoformat()

        docs = [
            {
                "id": make_doc_id(source_url, i),
                "content": chunk,
                "source_url": source_url,
                "source_type": "letter",
                "page_title": page_title,
                "chunk_index": i,
                "content_vector": vector,
                "last_modified": lm,
            }
            for i, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]
        search_client.upload_documents(docs)
        log.info("  Indexed %d chunk(s) for %s", len(docs), source_url)

        state[message_id] = {
            "source_url": source_url,
            "chunk_count": len(docs),
            "date_iso": parsed["date_iso"],
            "year_groups": parsed["year_groups"],
            "subject": subject,
        }

        # Mark as read so we never process it again
        _graph_patch(f"/me/messages/{graph_id}", access_token, {"isRead": True})
        indexed_count += 1

    save_json_state(blob_client, CONTAINER_RAW, LETTER_STATE_BLOB, {"letters": state})
    log.info("Letters indexed this run: %d", indexed_count)
    return indexed_count


if __name__ == "__main__":
    fetch_and_index_letters()

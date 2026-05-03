"""
Warwick Prep School — Weekly Letter Ingester

Polls an Outlook.com IMAP mailbox for forwarded weekly letters,
parses each one, and indexes each item as a chunk into Azure AI Search.

One-time setup
--------------
1. Create a free Outlook.com account, e.g. warwick-letters@outlook.com.
   Do NOT enable two-step verification on this account — it is a
   dedicated service mailbox, so use a strong unique password instead.

2. In the Outlook.com account go to:
   Settings → Mail → Sync email → and enable IMAP access.

3. In your personal email client, create a server-side forwarding rule:
     Condition : From contains @warwickschools.co.uk
                 AND Subject contains "Weekly Letter"
     Action    : Forward to warwick-letters@outlook.com
   The rule runs on Microsoft's servers so your device never needs to
   be open.

4. Add the following to your .env (and as GitHub Actions secrets):
     LETTER_EMAIL=warwick-letters@outlook.com
     LETTER_PASSWORD=<the Outlook.com account password>

The script is idempotent: it tracks indexed email Message-IDs in
Azure Blob Storage (_letter_index_state.json) and skips any letter
it has already processed.
"""

import email as email_lib
import logging
import imaplib
import os
import re
from datetime import datetime, timezone
from email import policy as email_policy

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
IMAP_SERVER = os.getenv("LETTER_IMAP_SERVER", "outlook.office365.com")
IMAP_PORT = int(os.getenv("LETTER_IMAP_PORT", "993"))
LETTER_EMAIL = os.getenv("LETTER_EMAIL")
LETTER_PASSWORD = os.getenv("LETTER_PASSWORD")
# Only process emails whose subject contains this string (case-insensitive)
LETTER_SUBJECT_FILTER = os.getenv("LETTER_SUBJECT_FILTER", "Weekly Letter")
# Only process emails whose From address contains this domain
LETTER_FROM_FILTER = os.getenv("LETTER_FROM_FILTER", "warwickschools.co.uk")

LETTER_STATE_BLOB = "_letter_index_state.json"
CONTAINER_RAW = os.getenv("AZURE_STORAGE_CONTAINER_RAW", "webcrawl-raw")

# ── Parsing helpers ───────────────────────────────────────────

_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
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

# Matches item lines: "Whole School: Some reminder text here."
# Category label: starts with a letter, up to ~60 chars, may contain spaces/digits
_ITEM_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 &/,.()\-]{0,60}):\s+(.+)$")

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

        # Stop at the sign-off
        if any(lower.startswith(p) for p in _STOP_PHRASES):
            _flush()
            current_category = None
            break

        # Year-group header — first non-empty line that names a year group
        if year_groups is None and re.match(
            r"^(NURSERY|RECEPTION|YEAR\s+\d|PRE.?PREP|PREP)", line, re.IGNORECASE
        ):
            year_groups = line
            continue

        # Date line
        if date is None:
            m = _DATE_RE.search(line)
            if m:
                date = datetime(
                    int(m.group(3)),
                    _MONTH_MAP[m.group(2).lower()],
                    int(m.group(1)),
                    tzinfo=timezone.utc,
                )
                continue

        # Skip salutation / section headers
        if lower in _SKIP_LINES:
            continue

        # Item line: "Category: text…"
        m = _ITEM_RE.match(line)
        if m:
            _flush()
            current_category = m.group(1).strip()
            current_parts = [m.group(2).strip()]
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
    prefix = (
        f"[Weekly Letter - {parsed['date_iso']} - {parsed['year_groups']}]"
    )
    return [f"{prefix} {category}: {text}" for category, text in parsed["items"]]


# ── Main entry point ──────────────────────────────────────────

def fetch_and_index_letters() -> int:
    """Connect to IMAP, process unread letters, return count of letters indexed."""
    if not LETTER_EMAIL or not LETTER_PASSWORD:
        raise ValueError(
            "LETTER_EMAIL and LETTER_PASSWORD environment variables must be set."
        )

    blob_client = get_blob_service_client()
    openai_client = get_openai_client()
    index_client = get_search_index_client()
    search_client = get_search_client()

    # Ensure the search index exists (safe to call repeatedly)
    ensure_search_index(index_client)

    # Load state: maps email Message-ID -> metadata about what was indexed
    state: dict = load_json_state(
        blob_client, CONTAINER_RAW, LETTER_STATE_BLOB
    ).get("letters", {})

    indexed_count = 0

    with imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT) as imap:
        imap.login(LETTER_EMAIL, LETTER_PASSWORD)
        imap.select("INBOX")

        # Fetch all unseen messages; subject/sender filtering happens in Python
        _, data = imap.search(None, "UNSEEN")
        uids = data[0].split() if data[0] else []
        log.info("Found %d unread message(s) to check", len(uids))

        for uid in uids:
            _, msg_data = imap.fetch(uid, "(RFC822)")
            raw = msg_data[0][1]
            msg = email_lib.message_from_bytes(raw, policy=email_policy.default)

            subject = msg.get("Subject", "")
            from_addr = msg.get("From", "")
            message_id = msg.get("Message-ID", uid.decode())

            # Filter by subject keyword
            if LETTER_SUBJECT_FILTER.lower() not in subject.lower():
                log.debug("Skipping (subject mismatch): %s", subject)
                continue

            # Filter by sender domain
            if LETTER_FROM_FILTER and LETTER_FROM_FILTER.lower() not in from_addr.lower():
                log.debug("Skipping (sender mismatch): %s", from_addr)
                continue

            # Skip letters already indexed
            if message_id in state:
                log.info("Already indexed, skipping: %s", subject)
                imap.store(uid, "+FLAGS", "\\Seen")
                continue

            log.info("Processing: %s (from %s)", subject, from_addr)
            text = _extract_text(msg)
            parsed = _parse_letter(text)

            if not parsed:
                log.warning("Could not parse letter body for: %s — marking read and skipping", subject)
                imap.store(uid, "+FLAGS", "\\Seen")
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
            imap.store(uid, "+FLAGS", "\\Seen")
            indexed_count += 1

    save_json_state(
        blob_client, CONTAINER_RAW, LETTER_STATE_BLOB, {"letters": state}
    )
    log.info("Letters indexed this run: %d", indexed_count)
    return indexed_count


if __name__ == "__main__":
    fetch_and_index_letters()

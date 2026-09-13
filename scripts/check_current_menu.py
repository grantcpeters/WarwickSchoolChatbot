"""Check Azure AI Search for a current or upcoming Warwick Prep menu PDF."""

import re
from datetime import date, timedelta
from urllib.parse import urlsplit

from src.indexer.run_indexer import get_search_client

_NUMERIC_DATE_RE = re.compile(
    r"\b([0-2]?\d|3[01])[.\-/](0?[1-9]|1[0-2])[.\-/](\d{4}|\d{2})\b"
)
_MONTHS = {
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
_TEXT_DATE_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+("
    + "|".join(_MONTHS)
    + r")\s+(\d{4})\b",
    re.IGNORECASE,
)


def extract_menu_dates(text: str) -> set[date]:
    """Extract numeric and long-form dates from menu content and URLs."""
    found: set[date] = set()
    for match in _NUMERIC_DATE_RE.finditer(text):
        year = int(match.group(3))
        if year < 100:
            year += 2000
        try:
            found.add(date(year, int(match.group(2)), int(match.group(1))))
        except ValueError:
            continue
    for match in _TEXT_DATE_RE.finditer(text):
        try:
            found.add(
                date(
                    int(match.group(3)),
                    _MONTHS[match.group(2).lower()],
                    int(match.group(1)),
                )
            )
        except ValueError:
            continue
    return found


def find_current_or_upcoming_menus(
    results: list[dict], today: date | None = None
) -> list[tuple[dict, list[date]]]:
    """Return menu PDFs dated from this week's Monday through three weeks ahead."""
    today = today or date.today()
    current_monday = today - timedelta(days=today.weekday())
    latest_allowed = current_monday + timedelta(days=21)
    matches: list[tuple[dict, list[date]]] = []
    seen_sources: set[str] = set()

    for result in results:
        source = result.get("source_url") or ""
        combined = " ".join(
            (source, result.get("page_title") or "", result.get("content") or "")
        )
        lower = combined.lower()
        source_lower = source.lower()
        is_pdf_url = urlsplit(source_lower).path.endswith(".pdf") or (
            "download.asp" in source_lower and "type=pdf" in source_lower
        )
        if source in seen_sources or not is_pdf_url:
            continue
        if not any(marker in lower for marker in ("menu", "week commencing", "option 1")):
            continue
        dates = sorted(extract_menu_dates(combined))
        relevant_dates = [
            menu_date
            for menu_date in dates
            if current_monday <= menu_date <= latest_allowed
        ]
        if relevant_dates:
            matches.append((result, relevant_dates))
            seen_sources.add(source)
    return matches


def main() -> int:
    client = get_search_client()
    results = list(
        client.search(
            search_text=(
                "weekly lunch menu monday tuesday wednesday thursday friday "
                "option 1 week commencing"
            ),
            top=50,
            select="content,source_url,source_type,page_title",
        )
    )
    matches = find_current_or_upcoming_menus(results)
    if not matches:
        print(
            "::warning::No current or upcoming menu PDF was found in Azure AI Search."
        )
        return 1

    for result, dates in matches:
        formatted_dates = ", ".join(menu_date.isoformat() for menu_date in dates)
        print(f"Menu indexed for {formatted_dates}: {result['source_url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

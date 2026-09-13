"""Tests for safe linked-PDF ingestion from weekly letters."""

from unittest.mock import MagicMock, patch

import pytest

from scripts import ingest_letters


def test_extract_linked_pdf_urls_accepts_school_pdf_and_deduplicates():
    html = """
    <a href="https://www.warwickprep.co.uk/files/menu.pdf">Lunch menu</a>
    <a href="https://www.warwickprep.co.uk/files/menu.pdf">Duplicate</a>
    <a href="https://example.com/private.pdf">External PDF</a>
    <a href="mailto:office@warwickprep.co.uk">Email</a>
    """

    assert ingest_letters._extract_linked_pdf_urls(html) == [
        (
            "Lunch menu",
            "https://www.warwickprep.co.uk/files/menu.pdf",
        )
    ]


def test_extract_linked_pdf_urls_accepts_legacy_pdf_query():
    html = """
    <a href="https://www.warwickprep.co.uk/download.asp?file=123&amp;type=pdf">
      Weekly menu
    </a>
    """

    assert ingest_letters._extract_linked_pdf_urls(html) == [
        (
            "Weekly menu",
            "https://www.warwickprep.co.uk/download.asp?file=123&type=pdf",
        )
    ]


def test_download_linked_pdf_rejects_redirect_outside_allowlist():
    response = MagicMock()
    response.url = "https://www.warwickprep.co.uk/files/menu.pdf"
    response.is_redirect = True
    response.is_permanent_redirect = False
    response.headers = {"location": "https://cdn.example.com/menu.pdf"}

    with patch("scripts.ingest_letters.requests.get", return_value=response):
        with pytest.raises(ValueError, match="outside the allowlist"):
            ingest_letters._download_linked_pdf(
                "https://www.warwickprep.co.uk/files/menu.pdf"
            )


def test_download_linked_pdf_enforces_streamed_size_limit(monkeypatch):
    monkeypatch.setattr(ingest_letters, "LETTER_LINK_MAX_BYTES", 5)
    response = MagicMock()
    response.url = "https://www.warwickprep.co.uk/files/menu.pdf"
    response.is_redirect = False
    response.is_permanent_redirect = False
    response.headers = {"content-type": "application/pdf"}
    response.iter_content.return_value = [b"%PDF", b"too-large"]

    with patch("scripts.ingest_letters.requests.get", return_value=response):
        with pytest.raises(ValueError, match="exceeds 5 bytes"):
            ingest_letters._download_linked_pdf(response.url)


def test_fetch_linked_pdf_documents_reports_failures_for_retry():
    html = """
    <a href="https://www.warwickprep.co.uk/files/menu.pdf">Lunch menu</a>
    """

    with patch(
        "scripts.ingest_letters._download_linked_pdf",
        side_effect=ingest_letters.requests.Timeout("temporary failure"),
    ):
        documents, failed_urls = ingest_letters._fetch_linked_pdf_documents(html)

    assert documents == []
    assert failed_urls == ["https://www.warwickprep.co.uk/files/menu.pdf"]

"""Tests for current-menu index monitoring."""

from datetime import date

from scripts.check_current_menu import extract_menu_dates, find_current_or_upcoming_menus


def test_extract_menu_dates_from_wordpress_filename_and_text():
    text = (
        "WPS-Lunch-Menu-Week-2-14.09.26.pdf "
        "Served on weeks commencing 21 September 2026"
    )

    assert extract_menu_dates(text) == {
        date(2026, 9, 14),
        date(2026, 9, 21),
    }


def test_find_current_or_upcoming_menu_accepts_current_week():
    result = {
        "source_url": (
            "https://www.warwickprep.co.uk/wp-content/uploads/"
            "WPS-Lunch-Menu-Week-2-14.09.26.pdf"
        ),
        "page_title": "",
        "content": "Monday Tuesday OPTION 1",
    }

    matches = find_current_or_upcoming_menus([result], today=date(2026, 9, 16))

    assert matches == [(result, [date(2026, 9, 14)])]


def test_find_current_or_upcoming_menu_accepts_legacy_pdf_url():
    result = {
        "source_url": (
            "https://www.warwickprep.co.uk/attachments/"
            "download.asp?file=123&type=pdf"
        ),
        "page_title": "Lunch menu",
        "content": "Served on weeks commencing 14 September 2026 OPTION 1",
    }

    matches = find_current_or_upcoming_menus([result], today=date(2026, 9, 16))

    assert matches == [(result, [date(2026, 9, 14)])]


def test_find_current_or_upcoming_menu_rejects_stale_or_unrelated_pdfs():
    stale_menu = {
        "source_url": "https://www.warwickprep.co.uk/menu-01.06.26.pdf",
        "page_title": "",
        "content": "Weekly menu OPTION 1",
    }
    inspection = {
        "source_url": "https://www.warwickprep.co.uk/inspection-14.09.26.pdf",
        "page_title": "Inspection",
        "content": "Inspection findings",
    }

    assert (
        find_current_or_upcoming_menus(
            [stale_menu, inspection], today=date(2026, 9, 16)
        )
        == []
    )

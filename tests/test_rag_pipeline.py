"""Tests for the RAG pipeline retrieval and chat logic."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

# ── Shared helpers ────────────────────────────────────────────


def make_search_result(
    content="Test content",
    source_url="https://www.warwickprep.com/about",
    source_type="html",
    page_title="About",
):
    return {
        "content": content,
        "source_url": source_url,
        "source_type": source_type,
        "page_title": page_title,
    }


class AsyncIteratorMock:
    """Async iterator that yields from a list."""

    def __init__(self, items):
        self._items = items

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for item in self._items:
            yield item


def make_openai_mock(embedding=None):
    mock = AsyncMock()
    mock.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=embedding or [0.1] * 1536)]
    )
    return mock


def make_search_mock(results):
    instance = AsyncMock()
    instance.__aenter__ = AsyncMock(return_value=instance)
    instance.__aexit__ = AsyncMock(return_value=False)
    instance.search.return_value = AsyncIteratorMock(results)
    return instance


# ── retrieve() ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retrieve_returns_results():
    with patch("src.chatbot.rag_pipeline._get_openai") as mock_openai_cls, patch(
        "src.chatbot.rag_pipeline._get_search"
    ) as mock_search_cls:

        mock_openai_cls.return_value = make_openai_mock()
        mock_search_cls.return_value = make_search_mock([make_search_result()])

        from src.chatbot.rag_pipeline import retrieve

        results = await retrieve("When was Warwick Prep School founded?")
        assert isinstance(results, list)
        assert len(results) == 1


@pytest.mark.asyncio
async def test_retrieve_maps_fields_correctly():
    with patch("src.chatbot.rag_pipeline._get_openai") as mock_openai_cls, patch(
        "src.chatbot.rag_pipeline._get_search"
    ) as mock_search_cls:

        mock_openai_cls.return_value = make_openai_mock()
        mock_search_cls.return_value = make_search_mock(
            [
                make_search_result(
                    content="School info",
                    source_url="https://warwickprep.com/info",
                    source_type="html",
                    page_title="Info Page",
                )
            ]
        )

        from src.chatbot.rag_pipeline import retrieve

        results = await retrieve("school info")
        assert results[0]["content"] == "School info"
        assert results[0]["source"] == "https://warwickprep.com/info"
        assert results[0]["type"] == "html"
        assert results[0]["title"] == "Info Page"


@pytest.mark.asyncio
async def test_retrieve_returns_empty_list_when_no_results():
    with patch("src.chatbot.rag_pipeline._get_openai") as mock_openai_cls, patch(
        "src.chatbot.rag_pipeline._get_search"
    ) as mock_search_cls:

        mock_openai_cls.return_value = make_openai_mock()
        mock_search_cls.return_value = make_search_mock([])

        from src.chatbot.rag_pipeline import retrieve

        results = await retrieve("something not in index")
        assert results == []


@pytest.mark.asyncio
async def test_retrieve_falls_back_when_page_title_missing_from_schema():
    with patch("src.chatbot.rag_pipeline._get_openai") as mock_openai_cls, patch(
        "src.chatbot.rag_pipeline._get_search"
    ) as mock_search_cls:

        mock_openai_cls.return_value = make_openai_mock()
        instance = AsyncMock()
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)

        call_count = 0

        async def side_effect_search(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Unknown field 'page_title' in select")
            return AsyncIteratorMock(
                [
                    {
                        "content": "fallback",
                        "source_url": "https://warwickprep.com",
                        "source_type": "html",
                    }
                ]
            )

        instance.search.side_effect = side_effect_search
        mock_search_cls.return_value = instance

        from src.chatbot.rag_pipeline import retrieve

        results = await retrieve("test query")
        assert call_count == 2
        assert results[0]["title"] == ""


@pytest.mark.asyncio
async def test_retrieve_reranks_admissions_before_news():
    """Canonical admissions pages should be sorted before news/blog posts."""
    news_chunk = make_search_result(
        content="Open morning Sept 2024",
        source_url="https://www.warwickprep.com/news-and-events/open-morning-sept2024",
    )
    admissions_chunk = make_search_result(
        content="Open morning March 2026",
        source_url="https://www.warwickprep.com/admissions/open-events",
    )

    with patch("src.chatbot.rag_pipeline._get_openai") as mock_openai_cls, patch(
        "src.chatbot.rag_pipeline._get_search"
    ) as mock_search_cls:

        mock_openai_cls.return_value = make_openai_mock()
        # Search returns news first, then admissions
        mock_search_cls.return_value = make_search_mock([news_chunk, admissions_chunk])

        from src.chatbot.rag_pipeline import retrieve

        results = await retrieve("when is the next open day")
        # After re-ranking, admissions page should come first
        assert "admissions" in results[0]["source"]
        assert "news" in results[1]["source"]


@pytest.mark.asyncio
async def test_retrieve_reranks_newer_news_before_older_news():
    """For event queries, chunks mentioning more recent dates should rank first within the news tier."""
    older_chunk = make_search_result(
        content="Open morning Saturday 21 September 2024, from 9:30am.",
        source_url="https://www.warwickprep.com/news-and-events/open-morning-sept2024",
    )
    newer_chunk = make_search_result(
        content="Open morning Saturday 21 March 2026, advance booking now closed.",
        source_url="https://www.warwickprep.com/news-and-events/open-morning-march2026",
    )

    with patch("src.chatbot.rag_pipeline._get_openai") as mock_openai_cls, patch(
        "src.chatbot.rag_pipeline._get_search"
    ) as mock_search_cls:

        mock_openai_cls.return_value = make_openai_mock()
        # Search returns older first
        mock_search_cls.return_value = make_search_mock([older_chunk, newer_chunk])

        from src.chatbot.rag_pipeline import retrieve

        # Must be an event query to trigger date-recency reranking
        results = await retrieve("when is the next open morning")
        # March 2026 is more recent than September 2024, should rank first within news tier
        assert "march2026" in results[0]["source"]
        assert "sept2024" in results[1]["source"]


@pytest.mark.asyncio
async def test_retrieve_surfaces_menu_pdf_for_lunch_query():
    """For lunch/menu queries, weekly menu PDFs from download.asp should be injected first."""
    general_chunk = make_search_result(
        content="Catering: all children have lunch at school. Menus are available online.",
        source_url="https://www.warwickprep.com/catering",
    )
    menu_pdf_chunk = make_search_result(
        content="WPS SUMMER TERM WEEK 3 2026 Monday Tuesday Wednesday Thursday Friday OPTION 1 BBQ Chicken",
        source_url="https://www.warwickprep.com/attachments/download.asp?file=979&type=pdf",
    )

    call_count = 0

    def search_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        search_text = kwargs.get("search_text", args[0] if args else "")
        if "monday tuesday wednesday" in search_text.lower():
            return AsyncIteratorMock([menu_pdf_chunk])
        return AsyncIteratorMock([general_chunk])

    instance = AsyncMock()
    instance.__aenter__ = AsyncMock(return_value=instance)
    instance.__aexit__ = AsyncMock(return_value=False)
    instance.search.side_effect = search_side_effect

    with patch("src.chatbot.rag_pipeline._get_openai") as mock_openai_cls, patch(
        "src.chatbot.rag_pipeline._get_search"
    ) as mock_search_cls:

        mock_openai_cls.return_value = make_openai_mock()
        mock_search_cls.return_value = instance

        from src.chatbot.rag_pipeline import retrieve

        results = await retrieve("what is on the lunch menu today")
        # Menu PDF should be injected at the front
        assert "download.asp" in results[0]["source"]
        assert call_count == 2  # main search + supplemental menu search


def test_most_recent_date_full_date():
    from src.chatbot.rag_pipeline import _most_recent_date
    from datetime import datetime

    text = "Open Morning on Saturday 21 March 2026, from 9.30am to 12.15pm."
    assert _most_recent_date(text) == datetime(2026, 3, 21).toordinal()


def test_most_recent_date_month_year_only():
    from src.chatbot.rag_pipeline import _most_recent_date
    from datetime import datetime

    text = "Our next open morning will be in September 2025."
    assert _most_recent_date(text) == datetime(2025, 9, 1).toordinal()


def test_most_recent_date_picks_latest():
    from src.chatbot.rag_pipeline import _most_recent_date
    from datetime import datetime

    text = "Past mornings: September 2024 and March 2026."
    assert _most_recent_date(text) == datetime(2026, 3, 1).toordinal()


def test_most_recent_date_returns_zero_when_no_dates():
    from src.chatbot.rag_pipeline import _most_recent_date

    assert _most_recent_date("No dates here at all.") == 0


# ── System prompt ─────────────────────────────────────────────


def test_system_prompt_contains_school_name():
    from src.chatbot.rag_pipeline import _build_system_prompt

    prompt = _build_system_prompt()
    assert "Warwick Prep School" in prompt


def test_system_prompt_contains_today_date():
    from datetime import date
    from src.chatbot.rag_pipeline import _build_system_prompt

    prompt = _build_system_prompt()
    assert date.today().strftime("%Y") in prompt


def test_system_prompt_contains_refusal_rules():
    from src.chatbot.rag_pipeline import _build_system_prompt

    prompt = _build_system_prompt()
    assert "STRICT RULES" in prompt
    assert "REFUSE" in prompt
    assert "code" in prompt.lower()


def test_system_prompt_allows_formatting():
    from src.chatbot.rag_pipeline import _build_system_prompt

    prompt = _build_system_prompt()
    # Must explicitly allow tables/formatting for school info
    assert "table" in prompt.lower() or "formatted" in prompt.lower()


def test_system_prompt_contains_contact_details():
    from src.chatbot.rag_pipeline import _build_system_prompt

    prompt = _build_system_prompt()
    assert "admissions@warwickprep.com" in prompt
    assert "01926 491545" in prompt


def test_system_prompt_warns_about_past_events():
    from src.chatbot.rag_pipeline import _build_system_prompt

    prompt = _build_system_prompt()
    assert "out of date" in prompt.lower() or "passed" in prompt.lower()


# ── chat() ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_chat_yields_tokens():
    with patch("src.chatbot.rag_pipeline.retrieve") as mock_retrieve, patch(
        "src.chatbot.rag_pipeline._get_openai"
    ) as mock_openai_cls:

        mock_retrieve.return_value = []

        event1 = MagicMock()
        event1.choices = [MagicMock()]
        event1.choices[0].delta.content = "Hello"
        event2 = MagicMock()
        event2.choices = [MagicMock()]
        event2.choices[0].delta.content = " World"
        event3 = MagicMock()
        event3.choices = []

        mock_openai = AsyncMock()
        mock_openai.chat.completions.create.return_value = AsyncIteratorMock(
            [event1, event2, event3]
        )
        mock_openai_cls.return_value = mock_openai

        from src.chatbot.rag_pipeline import chat

        tokens = [t async for t in chat("hi", [])]
        full = "".join(tokens)
        assert "Hello" in full
        assert "World" in full


@pytest.mark.asyncio
async def test_chat_appends_sources_when_present():
    with patch("src.chatbot.rag_pipeline.retrieve") as mock_retrieve, patch(
        "src.chatbot.rag_pipeline._get_openai"
    ) as mock_openai_cls:

        mock_retrieve.return_value = [
            {
                "content": "lunch info",
                "source": "https://warwickprep.com/catering",
                "type": "html",
                "title": "Catering",
            }
        ]

        event = MagicMock()
        event.choices = [MagicMock()]
        event.choices[0].delta.content = "We serve hot meals."
        mock_openai = AsyncMock()
        mock_openai.chat.completions.create.return_value = AsyncIteratorMock([event])
        mock_openai_cls.return_value = mock_openai

        from src.chatbot.rag_pipeline import chat

        tokens = [t async for t in chat("what is for lunch", [])]
        full = "".join(tokens)
        assert "__sources__:" in full
        assert "https://warwickprep.com/catering" in full
        assert "Catering" in full


@pytest.mark.asyncio
async def test_chat_filters_non_http_sources():
    with patch("src.chatbot.rag_pipeline.retrieve") as mock_retrieve, patch(
        "src.chatbot.rag_pipeline._get_openai"
    ) as mock_openai_cls:

        mock_retrieve.return_value = [
            {
                "content": "some content",
                "source": "abc123def456",
                "type": "html",
                "title": "",
            },
            {
                "content": "real content",
                "source": "https://warwickprep.com/page",
                "type": "html",
                "title": "Page",
            },
        ]

        event = MagicMock()
        event.choices = [MagicMock()]
        event.choices[0].delta.content = "answer"
        mock_openai = AsyncMock()
        mock_openai.chat.completions.create.return_value = AsyncIteratorMock([event])
        mock_openai_cls.return_value = mock_openai

        from src.chatbot.rag_pipeline import chat

        tokens = [t async for t in chat("question", [])]
        full = "".join(tokens)
        assert "abc123def456" not in full
        assert "https://warwickprep.com/page" in full


@pytest.mark.asyncio
async def test_chat_deduplicates_sources():
    with patch("src.chatbot.rag_pipeline.retrieve") as mock_retrieve, patch(
        "src.chatbot.rag_pipeline._get_openai"
    ) as mock_openai_cls:

        mock_retrieve.return_value = [
            {
                "content": "chunk 1",
                "source": "https://warwickprep.com/admissions",
                "type": "html",
                "title": "Admissions",
            },
            {
                "content": "chunk 2",
                "source": "https://warwickprep.com/admissions",
                "type": "html",
                "title": "Admissions",
            },
        ]

        event = MagicMock()
        event.choices = [MagicMock()]
        event.choices[0].delta.content = "info"
        mock_openai = AsyncMock()
        mock_openai.chat.completions.create.return_value = AsyncIteratorMock([event])
        mock_openai_cls.return_value = mock_openai

        from src.chatbot.rag_pipeline import chat

        tokens = [t async for t in chat("admissions", [])]
        sources_line = next((t for t in tokens if "__sources__:" in t), "")
        url = "https://warwickprep.com/admissions"
        assert sources_line.count(url) == 1


@pytest.mark.asyncio
async def test_chat_limits_history_to_ten_turns():
    captured_messages = []

    with patch("src.chatbot.rag_pipeline.retrieve") as mock_retrieve, patch(
        "src.chatbot.rag_pipeline._get_openai"
    ) as mock_openai_cls:

        mock_retrieve.return_value = []

        async def capture_create(**kwargs):
            captured_messages.extend(kwargs.get("messages", []))
            return AsyncIteratorMock([])

        mock_openai = AsyncMock()
        mock_openai.chat.completions.create.side_effect = capture_create
        mock_openai_cls.return_value = mock_openai

        history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(20)
        ]

        from src.chatbot.rag_pipeline import chat

        _ = [t async for t in chat("new question", history)]

        # messages = [system] + up to 10 history + [current user]
        history_in_messages = [
            m
            for m in captured_messages
            if m["role"] != "system" and m["content"] != "new question"
        ]
        assert len(history_in_messages) <= 10


@pytest.mark.asyncio
async def test_chat_no_sources_when_no_http_urls():
    with patch("src.chatbot.rag_pipeline.retrieve") as mock_retrieve, patch(
        "src.chatbot.rag_pipeline._get_openai"
    ) as mock_openai_cls:

        mock_retrieve.return_value = [
            {
                "content": "some content",
                "source": "blobhash123",
                "type": "html",
                "title": "",
            },
        ]

        event = MagicMock()
        event.choices = [MagicMock()]
        event.choices[0].delta.content = "answer"
        mock_openai = AsyncMock()
        mock_openai.chat.completions.create.return_value = AsyncIteratorMock([event])
        mock_openai_cls.return_value = mock_openai

        from src.chatbot.rag_pipeline import chat

        tokens = [t async for t in chat("question", [])]
        full = "".join(tokens)
        assert "__sources__:" not in full


@pytest.mark.asyncio
async def test_chat_filters_download_asp_sources():
    """download.asp PDF links should not appear as source chips."""
    with patch("src.chatbot.rag_pipeline.retrieve") as mock_retrieve, patch(
        "src.chatbot.rag_pipeline._get_openai"
    ) as mock_openai_cls:

        mock_retrieve.return_value = [
            {
                "content": "uniform info",
                "source": "https://www.warwickprep.com/attachments/download.asp?file=123&type=pdf",
                "type": "pdf",
                "title": "",
            },
            {
                "content": "more info",
                "source": "https://www.warwickprep.com/uniform",
                "type": "html",
                "title": "Uniform",
            },
        ]

        event = MagicMock()
        event.choices = [MagicMock()]
        event.choices[0].delta.content = "info"
        mock_openai = AsyncMock()
        mock_openai.chat.completions.create.return_value = AsyncIteratorMock([event])
        mock_openai_cls.return_value = mock_openai

        from src.chatbot.rag_pipeline import chat

        tokens = [t async for t in chat("uniform", [])]
        full = "".join(tokens)
        assert "download.asp" not in full
        assert "https://www.warwickprep.com/uniform" in full


@pytest.mark.asyncio
async def test_chat_filters_hex_hash_pdf_sources():
    """Hex-hash blob names used as source URLs should be filtered even if they start with http."""
    with patch("src.chatbot.rag_pipeline.retrieve") as mock_retrieve, patch(
        "src.chatbot.rag_pipeline._get_openai"
    ) as mock_openai_cls:

        mock_retrieve.return_value = [
            {
                "content": "pdf content",
                "source": "https://www.warwickprep.com/ED13E431B4F0ABE4E536FE4074E3ECD2.pdf",
                "type": "pdf",
                "title": "",
            },
            {
                "content": "page content",
                "source": "https://www.warwickprep.com/catering",
                "type": "html",
                "title": "Catering",
            },
        ]

        event = MagicMock()
        event.choices = [MagicMock()]
        event.choices[0].delta.content = "lunch"
        mock_openai = AsyncMock()
        mock_openai.chat.completions.create.return_value = AsyncIteratorMock([event])
        mock_openai_cls.return_value = mock_openai

        from src.chatbot.rag_pipeline import chat

        tokens = [t async for t in chat("lunch", [])]
        full = "".join(tokens)
        assert "ED13E431" not in full
        assert "https://www.warwickprep.com/catering" in full


@pytest.mark.asyncio
async def test_chat_filters_malformed_at_sign_urls():
    """URLs like http://admissions@warwickprep.com/... should not appear as source chips."""
    with patch("src.chatbot.rag_pipeline.retrieve") as mock_retrieve, patch(
        "src.chatbot.rag_pipeline._get_openai"
    ) as mock_openai_cls:

        mock_retrieve.return_value = [
            {
                "content": "open events info",
                "source": "http://admissions@warwickprep.com/admissions/open-events",
                "type": "html",
                "title": "Open Events",
            },
            {
                "content": "admissions info",
                "source": "https://www.warwickprep.com/admissions",
                "type": "html",
                "title": "Admissions",
            },
        ]

        event = MagicMock()
        event.choices = [MagicMock()]
        event.choices[0].delta.content = "info"
        mock_openai = AsyncMock()
        mock_openai.chat.completions.create.return_value = AsyncIteratorMock([event])
        mock_openai_cls.return_value = mock_openai

        from src.chatbot.rag_pipeline import chat

        tokens = [t async for t in chat("open day", [])]
        full = "".join(tokens)
        assert "admissions@warwickprep.com" not in full
        assert "https://www.warwickprep.com/admissions" in full


# ── Fees query — hiddenarea filtering ─────────────────────────


@pytest.mark.asyncio
async def test_fees_query_removes_hiddenarea_chunks():
    """For fees queries, chunks from hiddenarea/ (historical fee archives) must be excluded
    when current-year alternatives exist."""
    old_fees = make_search_result(
        content="Fees for the Academic Year 2024/2025 From 1 January 2025 Reception Years 1 and 2 £3,950",
        source_url="https://www.warwickprep.com/hiddenarea/fees-24-25",
        page_title="Fees 24 25",
    )
    current_fees = make_search_result(
        content="Fees for the Academic Year 2025/2026 Reception Years 1 & 2 Net Fee £4,509 VAT £902 Total £5,411",
        source_url="https://www.warwickprep.com/admissions/fees",
        page_title="Fees",
    )

    with patch("src.chatbot.rag_pipeline._get_openai") as mock_openai_cls, patch(
        "src.chatbot.rag_pipeline._get_search"
    ) as mock_search_cls:

        mock_openai_cls.return_value = make_openai_mock()
        # Search returns old fees first (higher score), then current fees
        mock_search_cls.return_value = make_search_mock([old_fees, current_fees])

        from src.chatbot.rag_pipeline import retrieve

        results = await retrieve("what are the fees for the 2025/2026 year")

        sources = [r["source"] for r in results]
        assert not any(
            "hiddenarea" in s for s in sources
        ), "hiddenarea chunks should be excluded for fees queries"
        assert any(
            "admissions/fees" in s for s in sources
        ), "current admissions/fees page should be included"


@pytest.mark.asyncio
async def test_fees_query_keeps_hiddenarea_if_no_alternatives():
    """If hiddenarea/ is the only available result, do not return an empty list."""
    only_old_fees = make_search_result(
        content="Fees for the Academic Year 2024/2025 Reception £3,950",
        source_url="https://www.warwickprep.com/hiddenarea/fees-24-25",
        page_title="Fees 24 25",
    )

    with patch("src.chatbot.rag_pipeline._get_openai") as mock_openai_cls, patch(
        "src.chatbot.rag_pipeline._get_search"
    ) as mock_search_cls:

        mock_openai_cls.return_value = make_openai_mock()
        mock_search_cls.return_value = make_search_mock([only_old_fees])

        from src.chatbot.rag_pipeline import retrieve

        results = await retrieve("what are the school fees")

        # Should fall back to returning the hiddenarea result rather than nothing
        assert len(results) == 1
        assert "hiddenarea" in results[0]["source"]


@pytest.mark.asyncio
async def test_non_fees_query_does_not_filter_hiddenarea():
    """The hiddenarea filter must only trigger for fees-related queries."""
    hidden_page = make_search_result(
        content="Welcome to Warwick Prep School",
        source_url="https://www.warwickprep.com/hiddenarea/some-other-page",
        page_title="Hidden Page",
    )

    with patch("src.chatbot.rag_pipeline._get_openai") as mock_openai_cls, patch(
        "src.chatbot.rag_pipeline._get_search"
    ) as mock_search_cls:

        mock_openai_cls.return_value = make_openai_mock()
        mock_search_cls.return_value = make_search_mock([hidden_page])

        from src.chatbot.rag_pipeline import retrieve

        results = await retrieve("tell me about the school")

        # For a non-fees query, the hiddenarea chunk should NOT be filtered
        assert len(results) == 1
        assert "hiddenarea" in results[0]["source"]


def test_system_prompt_contains_fees_guidance():
    """System prompt must instruct the LLM to use the most recent year's fee data."""
    from src.chatbot.rag_pipeline import _build_system_prompt

    prompt = _build_system_prompt()
    assert "FEES" in prompt
    assert "MOST RECENT" in prompt


# ── Crawler URL exclusion ─────────────────────────────────────


def test_should_crawl_excludes_hiddenarea():
    """URLs under hiddenarea/ must be excluded from crawling."""
    from src.crawler.run_crawler import _should_crawl

    assert not _should_crawl("https://www.warwickprep.com/hiddenarea/fees-24-25")
    assert not _should_crawl(
        "https://www.warwickprep.com/hiddenarea/fees-24-25/our-response-to-vat"
    )


def test_should_crawl_excludes_stale_event_pages():
    """Stale open-morning archive slugs must be excluded from crawling."""
    from src.crawler.run_crawler import _should_crawl

    assert not _should_crawl("https://www.warwickprep.com/openmorningsept2024")
    assert not _should_crawl("https://www.warwickprep.com/openmorningold")
    assert not _should_crawl("https://www.warwickprep.com/apologies")


def test_should_crawl_allows_normal_pages():
    """Non-excluded pages should pass the crawl filter."""
    from src.crawler.run_crawler import _should_crawl

    assert _should_crawl("https://www.warwickprep.com/admissions/fees")
    assert _should_crawl("https://www.warwickprep.com/about-us")
    assert _should_crawl("https://www.warwickprep.com/openmorning")
    assert _should_crawl("https://www.warwickprep.com/")


# ── School structure — staff-list retrieval ───────────────────
#
# Warwick Prep has three departments (from /about-us/structure-of-the-school):
#   1. The Squirrels Nursery  — boys & girls from 3+, 3 classes
#   2. Reception & Pre-Prep  — boys & girls from 4–7, 4 classes per year
#                              (Reception, Year 1, Year 2)
#   3. Prep department       — girls from 7–11, 2-3 classes per year
#                              (Years 3, 4, 5, 6)
#
# For any teacher/staff query the /information/information/staff-list page
# must surface at position 0 in the retrieved context.  The page is indexed
# as a single chunk and has no dates, so without the explicit supplemental
# search + post-sort pinning it loses to newsletter PDFs that contain the
# same year-group words at high keyword density.

_STAFF_LIST_URL = "https://www.warwickprep.com/information/information/staff-list"

# Realistic staff-list content mirroring the indexed chunk
_STAFF_LIST_CONTENT = (
    "Staff List Senior Leadership Team "
    "Mrs Hellen Dodsworth BA Hons (QTS) NPQH Headmistress "
    "Mrs Dee Alder BEd Hons NPQSL Deputy Head "
    "Mrs Julie Johnson BA Hons PGCE COGE Director of Studies Year 6 Form Teacher "
    "Mrs Gill Smeeton BSc Hons (QTS) Head of Pre-Prep Department "
    "Mrs Deborah Ward BA Hons PGCE NPQH Head of Prep Department "
    "Year 1 Form Teacher Mrs Julie Brotherhood Mrs Helen Comerford "
    "Mrs Hannah Dickson Mrs Jo Feary Mrs Samantha Jones "
    "Year 2 Form Teacher Mrs Jane Annett Mrs Emma Chan Mrs Angela Darby "
    "Mrs Jenny Watson Year 2 Co-ordinator Head of RE "
    "Reception Form Teacher Mrs Emma Burbidge Mrs Hannah Earl "
    "Miss Ellie Keen Mrs Helen Sykes Reception Co-ordinator "
    "Year 3 Form Teacher Mrs Laura Griggs Ms Victoria Sarson "
    "Year 4 Form Teacher Miss Caroline Cheetham Miss Angela Meloy Miss Nicola Murphy "
    "Year 5 Form Teacher Mrs Karen Charl Mr Richard Morris Mrs Wendy Stone Mrs Lucy Wilkinson "
    "Year 6 Form Teacher Mrs Caroline Murden Miss Sara Wilby STEAM Co-ordinator "
    "Head of Nursery Mrs Kate Smart BA Hons PGCE "
    "Miss Katie Clark BA Hons PGCE Director of Sport WPS "
    "Miss Alison Griggs BA Hons Director of Music"
)


def _make_staff_search_instance():
    """
    Returns a SearchClient mock whose .search() side-effect returns:
      - the staff-list chunk when the supplemental keyword search fires
        (detected by the unique substring "staff list form teacher")
      - a generic irrelevant newsletter PDF otherwise
    """
    staff_chunk = make_search_result(
        content=_STAFF_LIST_CONTENT,
        source_url=_STAFF_LIST_URL,
        page_title="Staff List",
    )
    irrelevant = make_search_result(
        content="Newsletter about school activities and events for parents.",
        source_url="https://www.warwickprep.com/attachments/download.asp?file=844&type=pdf",
        page_title="Newsletter",
    )

    def search_side_effect(*args, **kwargs):
        search_text = kwargs.get("search_text", args[0] if args else "")
        if "staff list form teacher" in search_text.lower():
            return AsyncIteratorMock([staff_chunk])
        return AsyncIteratorMock([irrelevant, irrelevant, irrelevant])

    instance = AsyncMock()
    instance.__aenter__ = AsyncMock(return_value=instance)
    instance.__aexit__ = AsyncMock(return_value=False)
    instance.search.side_effect = search_side_effect
    return instance


# -- Nursery (The Squirrels Nursery, boys & girls from 3+) -----

@pytest.mark.asyncio
@pytest.mark.parametrize("query", [
    "who teaches nursery",
    "who are the nursery teachers",
    "who is the head of nursery",
    "nursery teacher",
    "who are the squirrels nursery staff",
    "who teaches in the squirrels",
])
async def test_staff_list_pinned_for_nursery_queries(query):
    """
    Nursery queries must surface the staff-list page first.
    The Squirrels Nursery has three classes; the Head of Nursery is Mrs Kate Smart.
    """
    with patch("src.chatbot.rag_pipeline._get_openai") as mock_openai_cls, patch(
        "src.chatbot.rag_pipeline._get_search"
    ) as mock_search_cls:
        mock_openai_cls.return_value = make_openai_mock()
        mock_search_cls.return_value = _make_staff_search_instance()

        from src.chatbot.rag_pipeline import retrieve

        results = await retrieve(query)

        assert len(results) > 0, f"No results for query: {query!r}"
        assert results[0]["source"] == _STAFF_LIST_URL, (
            f"Staff list not at position 0 for nursery query {query!r}. "
            f"Got: {results[0]['source']}"
        )


# -- Reception & Pre-Prep (boys & girls from 4–7) --------------

@pytest.mark.asyncio
@pytest.mark.parametrize("query", [
    # Reception (4 classes, first year of Reception/Pre-Prep)
    "who are the reception teachers",
    "who is the reception form teacher",
    # Year 1 (Pre-Prep, boys and girls)
    "who are the year 1 teachers",
    "who are all the year 1 teachers",
    "year 1 form teacher",
    "who teaches year 1",
    # Year 2 (Pre-Prep, boys and girls)
    "who are the year 2 teachers",
    "year 2 form teacher",
    # Pre-Prep department as a whole
    "who teaches in pre-prep",
    "pre-prep teachers",
    "who are the pre-prep staff",
])
async def test_staff_list_pinned_for_pre_prep_queries(query):
    """
    Reception and Pre-Prep queries (Years 1-2, boys & girls, 4 classes per year)
    must surface the staff-list page first.
    """
    with patch("src.chatbot.rag_pipeline._get_openai") as mock_openai_cls, patch(
        "src.chatbot.rag_pipeline._get_search"
    ) as mock_search_cls:
        mock_openai_cls.return_value = make_openai_mock()
        mock_search_cls.return_value = _make_staff_search_instance()

        from src.chatbot.rag_pipeline import retrieve

        results = await retrieve(query)

        assert len(results) > 0, f"No results for query: {query!r}"
        assert results[0]["source"] == _STAFF_LIST_URL, (
            f"Staff list not at position 0 for Pre-Prep query {query!r}. "
            f"Got: {results[0]['source']}"
        )


# -- Prep department (girls from 7–11, Years 3–6) --------------

@pytest.mark.asyncio
@pytest.mark.parametrize("query", [
    # Year 3 (first Prep year, supported by full-time TA)
    "who are the year 3 teachers",
    "year 3 form teacher",
    # Year 4
    "who are the year 4 teachers",
    "year 4 form teacher",
    # Year 5 (Maths taught in ability groups from Y5)
    "who are the year 5 teachers",
    "year 5 form teacher",
    # Year 6 (oldest year group)
    "who are the year 6 teachers",
    "year 6 form teacher",
    # Prep department as a whole
    "who teaches in the prep department",
    "who are the prep teachers",
])
async def test_staff_list_pinned_for_prep_queries(query):
    """
    Prep department queries (Years 3-6, girls only, 2-3 classes per year) must
    surface the staff-list page first.
    """
    with patch("src.chatbot.rag_pipeline._get_openai") as mock_openai_cls, patch(
        "src.chatbot.rag_pipeline._get_search"
    ) as mock_search_cls:
        mock_openai_cls.return_value = make_openai_mock()
        mock_search_cls.return_value = _make_staff_search_instance()

        from src.chatbot.rag_pipeline import retrieve

        results = await retrieve(query)

        assert len(results) > 0, f"No results for query: {query!r}"
        assert results[0]["source"] == _STAFF_LIST_URL, (
            f"Staff list not at position 0 for Prep query {query!r}. "
            f"Got: {results[0]['source']}"
        )


# -- Senior Leadership Team queries ----------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("query", [
    # Headmistress — Mrs Hellen Dodsworth
    "who is the headmistress",
    "who is the head teacher",
    "who is the headteacher",
    # Deputy Head — Mrs Dee Alder
    "who is the deputy head",
    # Director of Studies — Mrs Julie Johnson
    "who is the director of studies",
    # Director of Sport — Miss Katie Clark
    "who is the director of sport",
    # Head of Pre-Prep — Mrs Gill Smeeton
    "who is the head of pre-prep",
    # Head of Prep — Mrs Deborah Ward
    "who is the head of prep",
    # Leadership team as a whole
    "who is in the senior leadership team",
    "who runs the school",
    "senior leadership team",
])
async def test_staff_list_pinned_for_leadership_queries(query):
    """
    Queries about leadership roles defined in the school structure must surface
    the staff-list page first (all five SLT members are listed there).
    """
    with patch("src.chatbot.rag_pipeline._get_openai") as mock_openai_cls, patch(
        "src.chatbot.rag_pipeline._get_search"
    ) as mock_search_cls:
        mock_openai_cls.return_value = make_openai_mock()
        mock_search_cls.return_value = _make_staff_search_instance()

        from src.chatbot.rag_pipeline import retrieve

        results = await retrieve(query)

        assert len(results) > 0, f"No results for query: {query!r}"
        assert results[0]["source"] == _STAFF_LIST_URL, (
            f"Staff list not at position 0 for leadership query {query!r}. "
            f"Got: {results[0]['source']}"
        )


# -- Whole-school staff queries --------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("query", [
    "list all teachers at warwick prep",
    "staff list",
    "who are all the staff",
    "list of teachers",
    "all members of staff",
    "form teacher",
    "who are the form teachers",
])
async def test_staff_list_pinned_for_whole_school_queries(query):
    """Whole-school staff queries must surface the staff-list page first."""
    with patch("src.chatbot.rag_pipeline._get_openai") as mock_openai_cls, patch(
        "src.chatbot.rag_pipeline._get_search"
    ) as mock_search_cls:
        mock_openai_cls.return_value = make_openai_mock()
        mock_search_cls.return_value = _make_staff_search_instance()

        from src.chatbot.rag_pipeline import retrieve

        results = await retrieve(query)

        assert len(results) > 0, f"No results for query: {query!r}"
        assert results[0]["source"] == _STAFF_LIST_URL, (
            f"Staff list not at position 0 for whole-school query {query!r}. "
            f"Got: {results[0]['source']}"
        )


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

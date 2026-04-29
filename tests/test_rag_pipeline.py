"""Tests for the RAG pipeline retrieval and chat logic."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


# ── Shared helpers ────────────────────────────────────────────

def make_search_result(content="Test content", source_url="https://www.warwickprep.com/about",
                       source_type="html", page_title="About"):
    return {"content": content, "source_url": source_url, "source_type": source_type, "page_title": page_title}


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
    with patch("src.chatbot.rag_pipeline._get_openai") as mock_openai_cls, \
         patch("src.chatbot.rag_pipeline._get_search") as mock_search_cls:

        mock_openai_cls.return_value = make_openai_mock()
        mock_search_cls.return_value = make_search_mock([make_search_result()])

        from src.chatbot.rag_pipeline import retrieve
        results = await retrieve("When was Warwick Prep School founded?")
        assert isinstance(results, list)
        assert len(results) == 1


@pytest.mark.asyncio
async def test_retrieve_maps_fields_correctly():
    with patch("src.chatbot.rag_pipeline._get_openai") as mock_openai_cls, \
         patch("src.chatbot.rag_pipeline._get_search") as mock_search_cls:

        mock_openai_cls.return_value = make_openai_mock()
        mock_search_cls.return_value = make_search_mock([
            make_search_result(content="School info", source_url="https://warwickprep.com/info",
                               source_type="html", page_title="Info Page")
        ])

        from src.chatbot.rag_pipeline import retrieve
        results = await retrieve("school info")
        assert results[0]["content"] == "School info"
        assert results[0]["source"] == "https://warwickprep.com/info"
        assert results[0]["type"] == "html"
        assert results[0]["title"] == "Info Page"


@pytest.mark.asyncio
async def test_retrieve_returns_empty_list_when_no_results():
    with patch("src.chatbot.rag_pipeline._get_openai") as mock_openai_cls, \
         patch("src.chatbot.rag_pipeline._get_search") as mock_search_cls:

        mock_openai_cls.return_value = make_openai_mock()
        mock_search_cls.return_value = make_search_mock([])

        from src.chatbot.rag_pipeline import retrieve
        results = await retrieve("something not in index")
        assert results == []


@pytest.mark.asyncio
async def test_retrieve_falls_back_when_page_title_missing_from_schema():
    with patch("src.chatbot.rag_pipeline._get_openai") as mock_openai_cls, \
         patch("src.chatbot.rag_pipeline._get_search") as mock_search_cls:

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
            return AsyncIteratorMock([
                {"content": "fallback", "source_url": "https://warwickprep.com", "source_type": "html"}
            ])

        instance.search.side_effect = side_effect_search
        mock_search_cls.return_value = instance

        from src.chatbot.rag_pipeline import retrieve
        results = await retrieve("test query")
        assert call_count == 2
        assert results[0]["title"] == ""


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
    with patch("src.chatbot.rag_pipeline.retrieve") as mock_retrieve, \
         patch("src.chatbot.rag_pipeline._get_openai") as mock_openai_cls:

        mock_retrieve.return_value = []

        event1 = MagicMock(); event1.choices = [MagicMock()]; event1.choices[0].delta.content = "Hello"
        event2 = MagicMock(); event2.choices = [MagicMock()]; event2.choices[0].delta.content = " World"
        event3 = MagicMock(); event3.choices = []

        mock_openai = AsyncMock()
        mock_openai.chat.completions.create.return_value = AsyncIteratorMock([event1, event2, event3])
        mock_openai_cls.return_value = mock_openai

        from src.chatbot.rag_pipeline import chat
        tokens = [t async for t in chat("hi", [])]
        full = "".join(tokens)
        assert "Hello" in full
        assert "World" in full


@pytest.mark.asyncio
async def test_chat_appends_sources_when_present():
    with patch("src.chatbot.rag_pipeline.retrieve") as mock_retrieve, \
         patch("src.chatbot.rag_pipeline._get_openai") as mock_openai_cls:

        mock_retrieve.return_value = [
            {"content": "lunch info", "source": "https://warwickprep.com/catering", "type": "html", "title": "Catering"}
        ]

        event = MagicMock(); event.choices = [MagicMock()]; event.choices[0].delta.content = "We serve hot meals."
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
    with patch("src.chatbot.rag_pipeline.retrieve") as mock_retrieve, \
         patch("src.chatbot.rag_pipeline._get_openai") as mock_openai_cls:

        mock_retrieve.return_value = [
            {"content": "some content", "source": "abc123def456", "type": "html", "title": ""},
            {"content": "real content", "source": "https://warwickprep.com/page", "type": "html", "title": "Page"},
        ]

        event = MagicMock(); event.choices = [MagicMock()]; event.choices[0].delta.content = "answer"
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
    with patch("src.chatbot.rag_pipeline.retrieve") as mock_retrieve, \
         patch("src.chatbot.rag_pipeline._get_openai") as mock_openai_cls:

        mock_retrieve.return_value = [
            {"content": "chunk 1", "source": "https://warwickprep.com/admissions", "type": "html", "title": "Admissions"},
            {"content": "chunk 2", "source": "https://warwickprep.com/admissions", "type": "html", "title": "Admissions"},
        ]

        event = MagicMock(); event.choices = [MagicMock()]; event.choices[0].delta.content = "info"
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

    with patch("src.chatbot.rag_pipeline.retrieve") as mock_retrieve, \
         patch("src.chatbot.rag_pipeline._get_openai") as mock_openai_cls:

        mock_retrieve.return_value = []

        async def capture_create(**kwargs):
            captured_messages.extend(kwargs.get("messages", []))
            return AsyncIteratorMock([])

        mock_openai = AsyncMock()
        mock_openai.chat.completions.create.side_effect = capture_create
        mock_openai_cls.return_value = mock_openai

        history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"} for i in range(20)]

        from src.chatbot.rag_pipeline import chat
        _ = [t async for t in chat("new question", history)]

        # messages = [system] + up to 10 history + [current user]
        history_in_messages = [m for m in captured_messages if m["role"] != "system" and m["content"] != "new question"]
        assert len(history_in_messages) <= 10


@pytest.mark.asyncio
async def test_chat_no_sources_when_no_http_urls():
    with patch("src.chatbot.rag_pipeline.retrieve") as mock_retrieve, \
         patch("src.chatbot.rag_pipeline._get_openai") as mock_openai_cls:

        mock_retrieve.return_value = [
            {"content": "some content", "source": "blobhash123", "type": "html", "title": ""},
        ]

        event = MagicMock(); event.choices = [MagicMock()]; event.choices[0].delta.content = "answer"
        mock_openai = AsyncMock()
        mock_openai.chat.completions.create.return_value = AsyncIteratorMock([event])
        mock_openai_cls.return_value = mock_openai

        from src.chatbot.rag_pipeline import chat
        tokens = [t async for t in chat("question", [])]
        full = "".join(tokens)
        assert "__sources__:" not in full


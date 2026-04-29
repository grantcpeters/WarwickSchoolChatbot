"""Tests for the RAG pipeline retrieval logic."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.mark.asyncio
async def test_retrieve_returns_results():
    mock_results = [
        {"content": "Warwick Prep School was founded in 1879.", "source_url": "about.html", "source_type": "html", "page_title": "About"}
    ]

    class MockResults:
        async def __aiter__(self):
            for r in mock_results:
                yield r

    with patch("src.chatbot.rag_pipeline._get_openai") as mock_openai_cls, \
         patch("src.chatbot.rag_pipeline._get_search") as mock_search_cls:

        mock_openai = AsyncMock()
        mock_openai.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1] * 1536)]
        )
        mock_openai_cls.return_value = mock_openai

        mock_search_instance = AsyncMock()
        mock_search_instance.__aenter__ = AsyncMock(return_value=mock_search_instance)
        mock_search_instance.__aexit__ = AsyncMock(return_value=False)
        mock_search_instance.search.return_value = MockResults()
        mock_search_cls.return_value = mock_search_instance

        from src.chatbot.rag_pipeline import retrieve
        results = await retrieve("When was Warwick Prep School founded?")
        assert isinstance(results, list)

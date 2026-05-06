"""Search client factory and mock behaviour — no network."""

from types import SimpleNamespace

from multi_agent_research_lab.services import search_client as sc
from multi_agent_research_lab.services.search_client import (
    MockSearchClient,
    TavilySearchClient,
    make_search_client,
)


def test_mock_returns_documents() -> None:
    client = MockSearchClient()
    docs = client.search("agents", max_results=3)
    assert 0 < len(docs) <= 3
    assert all(d.snippet for d in docs)


def test_factory_picks_tavily_when_key_set(monkeypatch) -> None:
    monkeypatch.setattr(
        sc, "get_settings", lambda: SimpleNamespace(tavily_api_key="test-key")
    )
    assert isinstance(make_search_client(), TavilySearchClient)


def test_factory_picks_mock_when_key_absent(monkeypatch) -> None:
    monkeypatch.setattr(
        sc, "get_settings", lambda: SimpleNamespace(tavily_api_key=None)
    )
    assert isinstance(make_search_client(), MockSearchClient)

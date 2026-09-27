from __future__ import annotations

from typing import Any

import httpx

from agentforge.config import get_settings
from agentforge.providers.fake import FixtureSearchProvider


def build_search_provider():
    settings = get_settings()
    if settings.search_provider == "tavily" and settings.tavily_api_key:
        return TavilySearchProvider(settings.tavily_api_key)
    return FixtureSearchProvider()


class TavilySearchProvider:
    def __init__(self, api_key: str):
        self.api_key = api_key

    async def search(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "max_results": limit,
                    "search_depth": "basic",
                },
            )
            response.raise_for_status()
            data = response.json()
        return [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
            }
            for item in data.get("results", [])
        ]

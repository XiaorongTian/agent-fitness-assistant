"""Web search tools for the health agent."""

import os
from typing import Any

from langchain.tools import tool

from common.tool_logging import log_tool_end, log_tool_start


SEARCH_TOOL_NAMES = {"web_search"}


def build_search_tools() -> list[Any]:
    """Build optional web search tools when Tavily is configured."""
    if not os.getenv("TAVILY_API_KEY"):
        return []

    try:
        from langchain_tavily import TavilySearch
    except ImportError as exc:
        raise RuntimeError("已配置 TAVILY_API_KEY，但未安装 langchain-tavily") from exc

    tavily_search = TavilySearch(
        max_results=int(os.getenv("TAVILY_MAX_RESULTS", "5")),
        topic="general",
        include_answer=True,
        include_raw_content=False,
        search_depth=os.getenv("TAVILY_SEARCH_DEPTH", "basic"),
    )

    @tool
    def web_search(query: str, time_range: str | None = None) -> dict[str, Any]:
        """使用 Tavily 搜索互联网实时信息。用户询问新闻、政策、研究、网页资料或模型未知的最新信息时使用。"""
        arguments = {"query": query, "time_range": time_range}
        log_tool_start("web_search", arguments)
        payload: dict[str, Any] = {"query": query}
        if time_range:
            payload["time_range"] = time_range
        result = tavily_search.invoke(payload)
        log_tool_end("web_search", result)
        return result

    return [web_search]

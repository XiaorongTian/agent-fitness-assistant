"""Agent 工具注册模块，统一暴露工具列表和业务工具名称。"""

from typing import Any

from tools.health_tools import HEALTH_TOOL_NAMES, build_health_tools
from tools.search_tools import SEARCH_TOOL_NAMES, build_search_tools


BUSINESS_TOOL_NAMES = HEALTH_TOOL_NAMES | SEARCH_TOOL_NAMES


def build_agent_tools(extra_tools: list[Any] | None = None) -> list[Any]:
    """构建 Agent 可调用的全部业务工具。"""
    return [*build_health_tools(), *build_search_tools(), *(extra_tools or [])]

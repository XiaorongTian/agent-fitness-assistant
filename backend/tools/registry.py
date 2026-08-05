"""Agent 工具注册模块，统一暴露工具列表和业务工具名称。"""

from typing import Any

from tools.diet_tools import DIET_TOOL_NAMES, build_diet_tools
from tools.exercise_task_tools import EXERCISE_TASK_TOOL_NAMES, build_exercise_task_tools
from tools.health_tools import HEALTH_TOOL_NAMES, build_health_tools
from tools.search_tools import SEARCH_TOOL_NAMES, build_search_tools


BUSINESS_TOOL_NAMES = (
    HEALTH_TOOL_NAMES | SEARCH_TOOL_NAMES | DIET_TOOL_NAMES | EXERCISE_TASK_TOOL_NAMES
)


def build_agent_tools(extra_tools: list[Any] | None = None) -> list[Any]:
    """构建 Agent 可调用的全部业务工具。"""
    return [
        *build_health_tools(),
        *build_diet_tools(),
        *build_exercise_task_tools(),
        *build_search_tools(),
        *(extra_tools or []),
    ]

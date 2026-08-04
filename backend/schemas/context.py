"""Agent 运行上下文，向动态提示词和工具传递用户信息。"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentContext:
    """单次 Agent 调用使用的用户、健康档案和位置（实际位置，位置来源）上下文。"""

    user_id: str
    profile: dict[str, Any] = field(default_factory=dict)
    activity_location: str | None = None
    location_source: str | None = None
    allow_md2card: bool = False

"""Agent 运行上下文，向动态提示词和工具传递用户信息。"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentContext:
    """单次 Agent 调用使用的用户标识和健康档案。"""

    user_id: str
    profile: dict[str, Any] = field(default_factory=dict)

"""MCP 工具调用的请求级观测上下文。"""

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MCPToolObservationContext:
    """单次对话内收集 MCP 工具调用指标，供日志和接口调用链复用。"""
    """用于保存当前请求这个mcp工具的观测信息"""
    trace_id: str
    user_id: str
    session_id: str
    allowed_mcp_servers: set[str] = field(default_factory=set)
    observations: list[dict[str, Any]] = field(default_factory=list)


mcp_observation_context: ContextVar[MCPToolObservationContext | None] = ContextVar(
    "mcp_observation_context", default=None
)

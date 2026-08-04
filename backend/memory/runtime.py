"""Agent 记忆运行时，管理 checkpointer、store 和标准 LangChain Agent。"""

import asyncio
import json
import os
from contextlib import AbstractAsyncContextManager
from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage

from agents.chat_agent import build_health_agent
from common.logger import logger
from common.mcp_observability import MCPToolObservationContext, mcp_observation_context
from schemas.chat import HealthAssistantOutput
from schemas.context import AgentContext
from tools.amap_mcp import load_amap_mcp_tools
from tools.md2card_api import load_md2card_api_tools
from tools.registry import BUSINESS_TOOL_NAMES

AGENT_RECURSION_LIMIT = int(os.getenv("AGENT_RECURSION_LIMIT", "10"))
PROFILE_NAMESPACE = "health_profile"
PROFILE_KEY = "current"
SESSION_NAMESPACE = "session_owner_v2"
SESSION_OWNER_KEY = "owner_user_id"
DEFAULT_ACTIVITY_LOCATION = os.getenv("DEFAULT_ACTIVITY_LOCATION", "北京来广营地铁站").strip()
TCM_CARD_KEYWORDS = ("中医", "中药", "经络", "穴位", "阴阳", "湿气", "气血", "脾胃", "体质", "节气养生")


def _parse_tool_output(content: Any) -> Any:
    """把工具消息内容尽量解析成 JSON；无法解析时保留原值。"""
    if not isinstance(content, str):
        return content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return content


def _is_tcm_card_request(message: str) -> bool:
    """仅对明确的中医养生咨询开放知识卡片 MCP。"""
    normalized_message = message.lower()
    return any(keyword in normalized_message for keyword in TCM_CARD_KEYWORDS)


def _extract_tool_calls(
    messages: list[Any],
    visible_tool_names: set[str],
    mcp_observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """从 Agent 消息中提取本轮业务工具调用及其返回结果。"""
    traces_by_id: dict[str, dict[str, Any]] = {}
    ordered_ids: list[str] = []

    for message in messages:
        for call in getattr(message, "tool_calls", []) or []:
            name = call.get("name")
            if name not in visible_tool_names:
                continue
            call_id = call.get("id")
            trace = {
                "name": name,
                "arguments": call.get("args") or {},
                "tool_call_id": call_id,
            }
            if call_id:
                traces_by_id[call_id] = trace
                ordered_ids.append(call_id)
            else:
                synthetic_id = f"{name}:{len(ordered_ids)}"
                traces_by_id[synthetic_id] = trace
                ordered_ids.append(synthetic_id)

        if getattr(message, "type", None) == "tool":
            call_id = getattr(message, "tool_call_id", None)
            if call_id in traces_by_id:
                traces_by_id[call_id]["output"] = _parse_tool_output(message.content)

    observations_by_name: dict[str, list[dict[str, Any]]] = {}
    for observation in mcp_observations:
        observations_by_name.setdefault(observation["name"], []).append(observation)

    traces = [traces_by_id[call_id] for call_id in ordered_ids]
    for trace in traces:
        matching_observations = observations_by_name.get(trace["name"], [])
        if matching_observations:
            trace.update(matching_observations.pop(0))
    return traces


def _current_turn_messages(messages: list[Any], current_message_id: str) -> list[Any]:
    """截取当前用户消息之后的消息，避免返回历史工具调用。"""
    for index, message in enumerate(messages):
        if getattr(message, "id", None) == current_message_id:
            return messages[index:]
    return messages


class ConversationRuntime:
    """管理 Agent、短期记忆 checkpointer 和长期记忆 store 的生命周期。"""

    def __init__(self) -> None:
        """创建空运行时；实际资源在 start 中按需初始化。"""
        self.checkpointer: Any | None = None
        self.store: Any | None = None
        self.agent: Any | None = None
        self.additional_tools: list[Any] = []
        self.visible_tool_names: set[str] = set(BUSINESS_TOOL_NAMES)
        self._checkpointer_context: AbstractAsyncContextManager[Any] | None = None
        self._store_context: AbstractAsyncContextManager[Any] | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """按环境变量初始化内存或 PostgreSQL 记忆后端，并构建 Agent。"""
        if self.agent is not None:
            return
        async with self._lock:
            if self.agent is not None:
                return
            backend = os.getenv("MEMORY_BACKEND", "memory").lower()
            if backend == "postgres":
                database_url = os.getenv("DATABASE_URL")
                if not database_url:
                    raise RuntimeError("MEMORY_BACKEND=postgres 时必须配置 DATABASE_URL")
                from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
                from langgraph.store.postgres.aio import AsyncPostgresStore

                self._checkpointer_context = AsyncPostgresSaver.from_conn_string(database_url)
                self._store_context = AsyncPostgresStore.from_conn_string(database_url)
                self.checkpointer = await self._checkpointer_context.__aenter__()
                self.store = await self._store_context.__aenter__()
                await self.checkpointer.setup()
                await self.store.setup()
            elif backend == "memory":
                from langgraph.checkpoint.memory import InMemorySaver
                from langgraph.store.memory import InMemoryStore

                self.checkpointer = InMemorySaver()
                self.store = InMemoryStore()
            else:
                raise RuntimeError("MEMORY_BACKEND 仅支持 memory 或 postgres")
            self.additional_tools = await load_amap_mcp_tools()
            self.additional_tools.extend(await load_md2card_api_tools())
            self.visible_tool_names = {
                *BUSINESS_TOOL_NAMES,
                *(tool.name for tool in self.additional_tools),
            }
            self.agent = build_health_agent(self.checkpointer, self.store, self.additional_tools)

    async def close(self) -> None:
        """关闭持久化连接并清空运行时资源。"""
        if self._store_context:
            await self._store_context.__aexit__(None, None, None)
        if self._checkpointer_context:
            await self._checkpointer_context.__aexit__(None, None, None)
        self.checkpointer = self.store = self.agent = None
        self.additional_tools = []
        self.visible_tool_names = set(BUSINESS_TOOL_NAMES)
        self._store_context = self._checkpointer_context = None

    async def get_profile_value(self, user_id: str) -> dict[str, Any]:
        """从长期记忆中读取用户健康档案原始值。"""
        await self.start()
        item = await self.store.aget(("user", user_id, PROFILE_NAMESPACE), PROFILE_KEY)
        return item.value if item else {}

    async def ensure_session_owner(self, user_id: str, session_id: str) -> None:
        """校验会话归属，防止其他用户复用同一个 session_id。"""
        await self.start()
        namespace = ("session", session_id, SESSION_NAMESPACE)
        item = await self.store.aget(namespace, SESSION_OWNER_KEY)
        owner_user_id = item.value.get("user_id") if item and isinstance(item.value, dict) else None
        if owner_user_id and owner_user_id != user_id:
            raise PermissionError("该会话不属于当前用户")
        if not item:
            await self.store.aput(namespace, SESSION_OWNER_KEY, {"user_id": user_id})

    async def invoke_chat(
        self,
        user_id: str,
        session_id: str,
        message: str,
        activity_location: str | None = None,
        trace_id: str | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """调用 Agent 完成一轮对话，并返回结构化结果和工具调用轨迹。"""
        try:
            logger.info("chat_stage=start_runtime user_id=%s session_id=%s", user_id, session_id)
            await self.start()
            logger.info("chat_stage=ensure_session_owner user_id=%s session_id=%s", user_id, session_id)
            await self.ensure_session_owner(user_id, session_id)
            logger.info("chat_stage=get_profile user_id=%s", user_id)
            profile = await self.get_profile_value(user_id)
            message_id = uuid4().hex
            logger.info("chat_stage=agent_invoke user_id=%s session_id=%s", user_id, session_id)
            resolved_location = activity_location or DEFAULT_ACTIVITY_LOCATION or None
            location_source = "user_consent" if activity_location else "mock_default"
            observation_context = MCPToolObservationContext(
                trace_id=trace_id or uuid4().hex,
                user_id=user_id,
                session_id=session_id,
                allowed_mcp_servers={"amap"},
            )
            context_token = mcp_observation_context.set(observation_context)
            try:
                agent_state = await self.agent.ainvoke(
                    {"messages": [HumanMessage(content=message, id=message_id)]},
                    {
                        "configurable": {"thread_id": session_id},
                        "recursion_limit": AGENT_RECURSION_LIMIT,
                    },
                    context=AgentContext(
                        user_id=user_id,
                        profile=profile,
                        activity_location=resolved_location,
                        location_source=location_source if resolved_location else None,
                        allow_md2card=_is_tcm_card_request(message),
                    ),
                )
            finally:
                mcp_observation_context.reset(context_token)
            logger.info("chat_stage=read_structured_response user_id=%s session_id=%s", user_id, session_id)
            result = agent_state.get("structured_response")
            if not result:
                raise RuntimeError("Agent 未返回结构化结果")
            if not isinstance(result, HealthAssistantOutput):
                result = HealthAssistantOutput.model_validate(result)
            messages = _current_turn_messages(agent_state.get("messages", []), message_id)
            return result.model_dump(mode="json"), _extract_tool_calls(
                messages,
                self.visible_tool_names,
                observation_context.observations,
            )
        except Exception:
            logger.exception("chat_stage=failed user_id=%s session_id=%s", user_id, session_id)
            raise


conversation_runtime = ConversationRuntime()

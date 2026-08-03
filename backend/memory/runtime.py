"""长期记忆的存储生命周期管控"""

import asyncio
import json
import os
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime

from agents.chat_agent import build_structured_model, get_chat_model
from prompts.chat import CHAT_SYSTEM_PROMPT
from schemas.chat import HealthAssistantOutput

MAX_RAW_MESSAGES = 12 #最大消息轮次
RECENT_MESSAGES_TO_KEEP = 6 #最近保留消息轮次
MAX_CONTEXT_CHARS = 12_000 #最多字符数量
PROFILE_NAMESPACE = "health_profile" #具体存储信息的命名空间
PROFILE_KEY = "current" #命名空间中最新的数据index


@dataclass
class GraphContext:
    user_id: str


class ConversationState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    running_summary: str
    owner_user_id: str
    last_result: dict[str, Any]


def _message_text(message: AIMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return str(message.content)


async def _initialize_session(
    state: ConversationState, runtime: Runtime[GraphContext]
) -> dict[str, str]:
    owner = state.get("owner_user_id")
    if owner and owner != runtime.context.user_id:
        raise PermissionError("该会话不属于当前用户")
    return {"owner_user_id": runtime.context.user_id}


async def _call_model(
    state: ConversationState, runtime: Runtime[GraphContext]
) -> dict[str, Any]:
    """把当前会话和用户健康档案送给 LLM，得到结构化的健康助手回答，并把回答保存成下一轮会话历史"""
    profile_item = await runtime.store.aget(
        ("user", runtime.context.user_id, PROFILE_NAMESPACE), PROFILE_KEY
    )
    profile = profile_item.value if profile_item else {}
    summary = state.get("running_summary", "")
    context = (
        "\n\n以下是可信的用户长期健康档案（可能为空）：\n"
        f"{json.dumps(profile, ensure_ascii=False)}\n"
        "仅在与当前问题有关时使用；档案中的限制优先于一般建议。"
    )
    if summary:
        context += f"\n\n以下是此前会话的压缩摘要：\n{summary}"

    model = build_structured_model()
    result = await model.ainvoke(
        [SystemMessage(content=f"{CHAT_SYSTEM_PROMPT}{context}"), *state["messages"]]
    )
    if not isinstance(result, HealthAssistantOutput):
        result = HealthAssistantOutput.model_validate(result)

    history_content = result.reply
    if result.actions:
        history_content += "\n行动建议：" + "；".join(
            f"{action.title}：{action.detail}" for action in result.actions
        )
    if result.safety_notice:
        history_content += f"\n安全提示：{result.safety_notice}"

    return {
        "messages": [AIMessage(content=history_content)],
        "last_result": result.model_dump(mode="json"),
    }


def _needs_summarization(
    state: ConversationState,
) -> Literal["summarize", "call_model"]:
    """检查当前会话是否需要压缩成摘要"""
    messages = state.get("messages", [])
    total_chars = sum(len(str(message.content)) for message in messages)
    if len(messages) > MAX_RAW_MESSAGES or total_chars > MAX_CONTEXT_CHARS:
        return "summarize"
    return "call_model"


async def _summarize(state: ConversationState) -> dict[str, Any]:
    """把当前会话RECENT_MESSAGES_TO_KEEP之前的内容压缩成摘要，并合并摘要"""
    messages = state["messages"]
    messages_to_summarize = messages[:-RECENT_MESSAGES_TO_KEEP]
    previous_summary = state.get("running_summary", "无")
    response = await get_chat_model().ainvoke(
        [
            SystemMessage(
                content=(
                    "请将以下健康助手会话压缩成中文工作摘要。保留用户目标、限制、"
                    "已确认事实、未完成事项和建议的关键上下文；不要编造信息；不超过 350 字。"
                )
            ),
            HumanMessage(
                content=(
                    f"已有摘要：\n{previous_summary}\n\n"
                    f"待压缩消息：\n"
                    + "\n".join(f"{message.type}: {_message_text(message)}" for message in messages_to_summarize)
                )
            ),
        ]
    )
    return {
        "running_summary": _message_text(response),
        "messages": [RemoveMessage(id=message.id) for message in messages_to_summarize],
    }


def build_conversation_graph(checkpointer: Any, store: Any) -> Any:
    """创建会话、对话产生信息、压缩信息"""
    builder = StateGraph(ConversationState, context_schema=GraphContext)
    builder.add_node("initialize_session", _initialize_session)
    builder.add_node("call_model", _call_model)
    builder.add_node("summarize", _summarize)
    builder.add_edge(START, "initialize_session")
    builder.add_conditional_edges("initialize_session", _needs_summarization)
    builder.add_edge("summarize", "call_model")
    builder.add_edge("call_model", END)
    return builder.compile(checkpointer=checkpointer, store=store)


class ConversationRuntime:
    """Owns persistence resources for the application lifespan."""

    def __init__(self) -> None:
        self.checkpointer: Any | None = None
        self.store: Any | None = None
        self.graph: Any | None = None
        self._checkpointer_context: AbstractAsyncContextManager[Any] | None = None
        self._store_context: AbstractAsyncContextManager[Any] | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if self.graph is not None:
            return
        async with self._lock:
            if self.graph is not None:
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
            self.graph = build_conversation_graph(self.checkpointer, self.store)

    async def close(self) -> None:
        if self._store_context:
            await self._store_context.__aexit__(None, None, None)
        if self._checkpointer_context:
            await self._checkpointer_context.__aexit__(None, None, None)
        self.checkpointer = self.store = self.graph = None
        self._store_context = self._checkpointer_context = None


conversation_runtime = ConversationRuntime()

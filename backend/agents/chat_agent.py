"""Health agent factory: model, tools, and structured final response."""

import json
import os
from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model

from prompts.chat import CHAT_SYSTEM_PROMPT
from schemas.chat import HealthAssistantOutput
from tools.health_tools import build_health_tools
from tools.search_tools import build_search_tools

def get_chat_model():
    """Build a tool-capable DashScope model, without hard-coding secrets."""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("未配置 DASHSCOPE_API_KEY，无法调用对话模型")

    return init_chat_model(
        model=os.getenv("CHAT_MODEL", "qqwen3.5-plus-2026-02-15"),
        model_provider = "openai", 
        api_key=api_key,
        base_url=os.getenv("DASHSCOPE_BASE_URL") or None,
        # ToolStrategy forces the final structured-output tool. DashScope thinking
        # mode rejects forced tool_choice, so this must stay disabled for this agent.
        extra_body={"enable_thinking": False},
    )


def build_health_agent(profile: dict[str, Any], summary: str = ""):
    """Create a small tool-calling agent for one user turn.

    ToolStrategy avoids provider-specific JSON response-format limitations while
    retaining the existing Pydantic contract for the final answer.
    """
    context = (
        "\n\n以下是可信的用户长期健康档案（可能为空）：\n"
        f"{json.dumps(profile, ensure_ascii=False)}\n"
        "仅在与当前问题有关时使用；档案中的限制优先于一般建议。"
    )
    if summary:
        context += f"\n\n以下是此前会话的压缩摘要：\n{summary}"
    tool_policy = """
        工具使用原则：
        - 需要实时信息、用户长期档案、天气或计划生成能力时，优先使用可用工具。
        - 每个工具的适用场景以工具自身描述为准；不要在未调用工具时编造实时数据或工具结果。
    """
    return create_agent(
        model=get_chat_model(),
        tools=[*build_health_tools(profile), *build_search_tools()],
        system_prompt=f"{CHAT_SYSTEM_PROMPT}{context}{tool_policy}",
        response_format=ToolStrategy(HealthAssistantOutput),
    )

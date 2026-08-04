"""健康 Agent 构建模块，配置模型、动态提示词、记忆和结构化输出。"""

import json
import os
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, SummarizationMiddleware, dynamic_prompt
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model

from prompts.chat import CHAT_SYSTEM_PROMPT
from schemas.chat import HealthAssistantOutput
from schemas.context import AgentContext
from tools.registry import build_agent_tools

SUMMARY_TRIGGER_TOKENS = int(os.getenv("SUMMARY_TRIGGER_TOKENS", "4000"))
SUMMARY_MESSAGES_TO_KEEP = int(os.getenv("SUMMARY_MESSAGES_TO_KEEP", "6"))

def get_chat_model():
    """根据环境变量构建 DashScope OpenAI 兼容对话模型。"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("未配置 DASHSCOPE_API_KEY，无法调用对话模型")

    return init_chat_model(
        model=os.getenv("CHAT_MODEL", "qqwen3.5-plus-2026-02-15"),
        model_provider = "openai", 
        api_key=api_key,
        base_url=os.getenv("DASHSCOPE_BASE_URL") or None,
        # 结构化输出会强制工具调用，Qwen 思考模式不兼容。
        extra_body={"enable_thinking": False},
    )


@dynamic_prompt
def health_system_prompt(request: ModelRequest) -> str:
    """根据运行时健康档案生成每次模型调用使用的系统提示词。"""
    context = request.runtime.context
    profile = context.profile if context else {}
    profile_context = (
        "\n\n以下是可信的用户长期健康档案（可能为空）：\n"
        f"{json.dumps(profile, ensure_ascii=False)}\n"
        "仅在与当前问题有关时使用；档案中的限制优先于一般建议。"
    )
    tool_policy = """

工具使用原则：
- 需要实时信息、用户长期档案、天气或计划生成能力时，优先使用可用工具。
- 每个工具的适用场景以工具自身描述为准；不要在未调用工具时编造实时数据或工具结果。
"""
    return f"{CHAT_SYSTEM_PROMPT}{profile_context}{tool_policy}"


def build_health_agent(checkpointer: Any, store: Any):
    """构建带短期记忆、长期存储、工具和结构化输出的健康 Agent。"""
    return create_agent(
        model=get_chat_model(),
        tools=build_agent_tools(),
        middleware=[
            health_system_prompt,
            SummarizationMiddleware(
                model=get_chat_model(),
                trigger={"tokens": SUMMARY_TRIGGER_TOKENS},
                keep=("messages", SUMMARY_MESSAGES_TO_KEEP),
            ),
        ],
        response_format=ToolStrategy(HealthAssistantOutput),
        context_schema=AgentContext,
        checkpointer=checkpointer,
        store=store,
    )

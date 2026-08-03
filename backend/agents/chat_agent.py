"""LangChain entry point for one structured health-chat response."""

import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model

from prompts.chat import CHAT_SYSTEM_PROMPT
from schemas.chat import ChatRequest, HealthAssistantOutput

def get_chat_model():
    """Build a model from environment configuration, without hard-coding secrets."""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("未配置 DASHSCOPE_API_KEY，无法调用对话模型")

    return init_chat_model(
        model=os.getenv("CHAT_MODEL", "qwen3.5-plus"),
        model_provider = "openai", 
        api_key=api_key,
        base_url=os.getenv("DASHSCOPE_BASE_URL") or None,
    )


def build_structured_model():
    return get_chat_model().with_structured_output(HealthAssistantOutput)

async def generate_chat_response(request: ChatRequest) -> HealthAssistantOutput:
    """Compatibility wrapper; conversations should use the LangGraph runtime."""
    result = await build_structured_model().ainvoke(
        [SystemMessage(content=CHAT_SYSTEM_PROMPT), HumanMessage(content=request.message)]
    )
    return result if isinstance(result, HealthAssistantOutput) else HealthAssistantOutput.model_validate(result)

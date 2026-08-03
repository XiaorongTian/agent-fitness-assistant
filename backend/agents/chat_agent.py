"""LangChain entry point for one structured health-chat response."""

import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model

from prompts.chat import CHAT_SYSTEM_PROMPT
from schemas.chat import ChatRequest, HealthAssistantOutput

# 创建模型
def _build_model() -> init_chat_model:
    """Build a model from environment configuration, without hard-coding secrets."""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("未配置 DASHSCOPE_API_KEY，无法调用对话模型")

    return init_chat_model(
        model=os.getenv("CHAT_MODEL", "qwen3.5-plus"),
        model_provider = "openai", 
        api_key=api_key,
        base_url=os.getenv("DASHSCOPE_BASE_URL") or None,
        # temperature=0.3,
    )

# 组装message
def _to_messages(request: ChatRequest) -> list[SystemMessage | HumanMessage | AIMessage]:
    """
        组装系统提示词+历史消息
    """
    messages: list[SystemMessage | HumanMessage | AIMessage] = [
        SystemMessage(content=CHAT_SYSTEM_PROMPT)
    ]
    for item in request.history:
        message_class = HumanMessage if item.role.value == "user" else AIMessage
        messages.append(message_class(content=item.content))
    messages.append(HumanMessage(content=request.message))
    return messages

# 直接调用模型，生成机构化输出
async def generate_chat_response(request: ChatRequest) -> HealthAssistantOutput:
    """Invoke the model and validate its answer against ``HealthAssistantOutput``."""
    structured_model = _build_model().with_structured_output(HealthAssistantOutput)
    print('============structured_model',structured_model)
    result = await structured_model.ainvoke(_to_messages(request))
    if not isinstance(result, HealthAssistantOutput):
        # Defensive validation for providers that return a dict despite the schema request.
        result = HealthAssistantOutput.model_validate(result)
    return result

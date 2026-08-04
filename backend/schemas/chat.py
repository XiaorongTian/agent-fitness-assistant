"""Schemas shared by the chat API and the structured LLM response."""

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ChatRole(str, Enum):
    """Roles accepted as previous conversation turns."""

    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    role: ChatRole
    content: str = Field(min_length=1, max_length=4_000)

    @field_validator("content")
    @classmethod
    def strip_content(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("消息内容不能为空")
        return value


class ChatRequest(BaseModel):
    """Input for one chat turn; conversation history is loaded server-side."""

    user_id: str = Field(min_length=1, max_length=128, description="Tenant-scoped user ID")
    message: str = Field(min_length=1, max_length=4_000, description="Current user message")
    image_url: Optional[str] = None
    session_id: str | None = Field(default=None, max_length=128)

    @field_validator("user_id", "message")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("字段不能为空")
        return value


class ChatIntent(str, Enum):
    # 用户意图识别类型枚举值 
    DIET = "diet"
    EXERCISE = "exercise"
    SEDENTARY = "sedentary"
    WELLNESS = "wellness"
    GENERAL = "general"
    HIGH_RISK = "high_risk"


class HealthAction(BaseModel):
    """模型给用户的建议输出"""

    title: str = Field(description="Short action title")
    detail: str = Field(description="Specific and executable instruction")
    priority: Literal["low", "medium", "high"] = "medium"


class HealthAssistantOutput(BaseModel):
    """模型输出的结构化结果，包含意图、回复、建议动作等信息"""

    intent: ChatIntent
    reply: str = Field(description="Concise Chinese response for the user")
    actions: list[HealthAction] = Field(default_factory=list, max_length=3)
    follow_up_question: str | None = Field(
        default=None, description="A single question only when more context is required"
    )
    safety_notice: str | None = Field(
        default=None, description="Safety or medical escalation notice when needed"
    )

    @field_validator("intent", mode="before")
    @classmethod
    def normalize_legacy_intent(cls, value: str) -> str:
        """兼容部分模型会输出的 normal；其语义等同于 general。"""
        if isinstance(value, str) and value.strip().lower() == "normal":
            return ChatIntent.GENERAL.value
        return value


class ToolCallTrace(BaseModel):
    """Visible trace information for one business tool call."""

    name: str = Field(description="Tool name")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Tool input arguments")
    tool_call_id: str | None = Field(default=None, description="Provider/LangChain tool call ID")
    output: Any | None = Field(default=None, description="Tool output, when available")


class ChatResponse(BaseModel):
    """输出显示到对话框的格式"""

    session_id: str | None = None
    result: HealthAssistantOutput
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    trace_id: str

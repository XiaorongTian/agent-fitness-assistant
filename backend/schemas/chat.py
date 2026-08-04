"""对话接口的数据模型，定义请求、响应和模型结构化输出。"""

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """用户发起单轮对话时提交的数据。"""

    user_id: str = Field(min_length=1, max_length=128, description="用户唯一标识")
    message: str = Field(min_length=1, max_length=4_000, description="本轮用户输入")
    image_url: Optional[str] = None
    session_id: str | None = Field(default=None, max_length=128)
    activity_location: str | None = Field(
        default=None,
        max_length=300,
        description="用户授权用于附近运动地点推荐的地址或地点名称",
    )
    location_consent: bool = Field(
        default=False,
        description="用户是否同意本轮将 activity_location 用于位置服务",
    )

    @field_validator("user_id", "message")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        """清理必填文本字段，并拒绝空字符串。"""
        value = value.strip()
        if not value:
            raise ValueError("字段不能为空")
        return value

    @field_validator("activity_location")
    @classmethod
    def strip_optional_location(cls, value: str | None) -> str | None:
        return value.strip() if value else None


class ChatIntent(str, Enum):
    """健康助手支持的用户意图分类。"""

    DIET = "diet"
    EXERCISE = "exercise"
    SEDENTARY = "sedentary"
    WELLNESS = "wellness"
    GENERAL = "general"
    HIGH_RISK = "high_risk"


class HealthAction(BaseModel):
    """面向用户的一条可执行建议。"""

    title: str = Field(description="简短动作标题")
    detail: str = Field(description="具体可执行说明")
    priority: Literal["low", "medium", "high"] = "medium"


class HealthAssistantOutput(BaseModel):
    """模型必须返回的健康助手结构化结果。"""

    intent: ChatIntent
    reply: str = Field(description="直接展示给用户的中文回复")
    actions: list[HealthAction] = Field(default_factory=list, max_length=3)
    follow_up_question: str | None = Field(
        default=None, description="信息不足时提出的一个追问"
    )
    safety_notice: str | None = Field(
        default=None, description="必要的安全或就医提醒"
    )

    @field_validator("intent", mode="before")
    @classmethod
    def normalize_legacy_intent(cls, value: str) -> str:
        """兼容 normal 意图。"""
        if isinstance(value, str) and value.strip().lower() == "normal":
            return ChatIntent.GENERAL.value
        return value


class ToolCallTrace(BaseModel):
    """对外返回的一次业务工具调用轨迹。"""

    name: str = Field(description="工具名称")
    arguments: dict[str, Any] = Field(default_factory=dict, description="工具入参")
    tool_call_id: str | None = Field(default=None, description="模型或 LangChain 生成的工具调用 ID")
    output: Any | None = Field(default=None, description="工具返回结果")
    mcp_server: str | None = Field(default=None, description="MCP 服务名；本地工具为空")
    duration_ms: int | None = Field(default=None, ge=0, description="工具调用耗时（毫秒）")
    status: Literal["succeeded", "failed"] | None = Field(
        default=None, description="工具调用状态"
    )
    error: str | None = Field(default=None, description="失败原因；成功时为空")


class ChatResponse(BaseModel):
    """对话接口返回给调用方的数据。"""

    session_id: str | None = None
    result: HealthAssistantOutput
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    trace_id: str

"""长期记忆的数据模型，定义用户健康档案和写入请求。"""

from datetime import datetime

from pydantic import BaseModel, Field


class HealthProfile(BaseModel):
    """经用户确认后可跨会话使用的健康档案。"""

    goal: str | None = Field(default=None, max_length=200)
    city: str | None = Field(default=None, max_length=100)
    food_restrictions: list[str] = Field(default_factory=list, max_length=20)
    exercise_limitations: list[str] = Field(default_factory=list, max_length=20)
    preferences: list[str] = Field(default_factory=list, max_length=20)
    updated_at: datetime | None = None
    source: str = "user_confirmed"


class SaveHealthProfileRequest(BaseModel):
    """保存健康档案的请求，必须带用户确认标记。"""

    user_id: str = Field(min_length=1, max_length=128)
    profile: HealthProfile
    confirmed: bool = Field(
        description="必须为 true，表示用户已确认该档案可写入长期记忆"
    )

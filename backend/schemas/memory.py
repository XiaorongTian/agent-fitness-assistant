"""长期记忆数据模型定义"""

from datetime import datetime

from pydantic import BaseModel, Field


class HealthProfile(BaseModel):
    """长期记忆接口定义的健康档案数据模型"""

    goal: str | None = Field(default=None, max_length=200)
    food_restrictions: list[str] = Field(default_factory=list, max_length=20)
    exercise_limitations: list[str] = Field(default_factory=list, max_length=20)
    preferences: list[str] = Field(default_factory=list, max_length=20)
    updated_at: datetime | None = None
    source: str = "user_confirmed"


class SaveHealthProfileRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    profile: HealthProfile
    confirmed: bool = Field(
        description="Must be true: the user has reviewed and confirmed this profile"
    )

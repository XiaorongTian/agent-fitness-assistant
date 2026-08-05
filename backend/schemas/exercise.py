"""运动任务、执行反馈与下一次自适应调整的数据模型。"""

from datetime import datetime, timezone
from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


class ExerciseIntensity(str, Enum):
    LOW = "low"
    MODERATE = "moderate"


class ExerciseTaskStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    TOO_HARD = "too_hard"
    DISLIKED = "disliked"


class BarrierReason(str, Enum):
    TIME = "time"
    FATIGUE = "fatigue"
    PAIN = "pain"
    WEATHER = "weather"
    LOCATION = "location"
    NOT_SURE_HOW = "not_sure_how"
    OTHER = "other"


class CreateExerciseTaskRequest(BaseModel):
    """用户主动请求创建一项当天运动任务。"""

    user_id: str = Field(min_length=1, max_length=128)
    available_minutes: int = Field(default=30, ge=10, le=120)
    city: str | None = Field(default=None, max_length=100)
    preferred_activity: str | None = Field(default=None, max_length=100)

    @field_validator("user_id")
    @classmethod
    def strip_user_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("user_id 不能为空")
        return value

    @field_validator("city", "preferred_activity")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None


class ExerciseTask(BaseModel):
    """用户显式创建且尚待执行或已反馈的运动任务。"""

    task_id: str = Field(default_factory=lambda: uuid4().hex)
    user_id: str
    title: str = Field(min_length=1, max_length=100)
    plan: str = Field(min_length=1, max_length=1_000)
    activity: str = Field(min_length=1, max_length=200)
    planned_minutes: int = Field(ge=10, le=120)
    intensity: ExerciseIntensity
    weather: dict = Field(default_factory=dict)
    limitations_applied: list[str] = Field(default_factory=list)
    adaptation_reason: str = Field(min_length=1, max_length=500)
    status: ExerciseTaskStatus = ExerciseTaskStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    feedback_at: datetime | None = None
    actual_minutes: int | None = Field(default=None, ge=0, le=240)
    fatigue_score: int | None = Field(default=None, ge=1, le=5)
    barrier_reason: BarrierReason | None = None
    feedback_note: str | None = Field(default=None, max_length=500)
    next_task_adjustment: str | None = Field(default=None, max_length=500)
    follow_up_question: str | None = Field(default=None, max_length=300)


class SubmitExerciseFeedbackRequest(BaseModel):
    """用户通过前端按钮提交任务执行结果；跳过时可后续补充阻碍原因。"""

    user_id: str = Field(min_length=1, max_length=128)
    status: Literal["completed", "skipped", "too_hard", "disliked"]
    actual_minutes: int | None = Field(default=None, ge=0, le=240)
    fatigue_score: int | None = Field(default=None, ge=1, le=5)
    barrier_reason: BarrierReason | None = None
    feedback_note: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def validate_feedback(self) -> "SubmitExerciseFeedbackRequest":
        if self.status == "completed" and self.actual_minutes is None:
            self.actual_minutes = 0
        return self

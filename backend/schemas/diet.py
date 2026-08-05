"""饮食记录的数据模型。所有营养数据均为估算区间，不作为医疗或处方依据。"""

from datetime import date, datetime, timezone
from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


class MealType(str, Enum):
    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    SNACK = "snack"
    UNKNOWN = "unknown"


class NutritionEstimate(BaseModel):
    """估算营养区间；单位分别为 kcal 与 g。"""

    calories_kcal_min: int = Field(ge=0, le=10_000)
    calories_kcal_max: int = Field(ge=0, le=10_000)
    protein_g_min: float = Field(ge=0, le=1_000)
    protein_g_max: float = Field(ge=0, le=1_000)
    carbohydrate_g_min: float = Field(ge=0, le=2_000)
    carbohydrate_g_max: float = Field(ge=0, le=2_000)
    fat_g_min: float = Field(ge=0, le=1_000)
    fat_g_max: float = Field(ge=0, le=1_000)

    @model_validator(mode="after")
    def validate_ranges(self) -> "NutritionEstimate":
        for lower, upper, label in (
            (self.calories_kcal_min, self.calories_kcal_max, "热量"),
            (self.protein_g_min, self.protein_g_max, "蛋白质"),
            (self.carbohydrate_g_min, self.carbohydrate_g_max, "碳水"),
            (self.fat_g_min, self.fat_g_max, "脂肪"),
        ):
            if lower > upper:
                raise ValueError(f"{label}最小值不能大于最大值")
        return self


class FoodItem(BaseModel):
    """一项食物及其份量、做法和营养估算。"""

    name: str = Field(min_length=1, max_length=100)
    amount_description: str = Field(min_length=1, max_length=100)
    cooking_method: str | None = Field(default=None, max_length=100)
    nutrition: NutritionEstimate


class FoodRecordAnalysis(BaseModel):
    """多模态模型生成的待确认饮食分析结果。"""

    meal_type: MealType = MealType.UNKNOWN
    foods: list[FoodItem] = Field(min_length=1, max_length=20)
    assumptions: list[str] = Field(default_factory=list, max_length=6)
    summary: str = Field(min_length=1, max_length=500)
    confidence: Literal["low", "medium", "high"] = "medium"


class CreateFoodRecordDraftRequest(BaseModel):
    """创建饮食草稿；至少提供文字描述或可访问的图片 URL 之一。"""

    user_id: str = Field(min_length=1, max_length=128)
    message: str | None = Field(default=None, max_length=2_000)
    image_url: str | None = Field(default=None, max_length=2_000)
    meal_type: MealType | None = None

    @field_validator("user_id")
    @classmethod
    def strip_user_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("user_id 不能为空")
        return value

    @field_validator("message", "image_url")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        return value.strip() if value else None

    @model_validator(mode="after")
    def validate_content(self) -> "CreateFoodRecordDraftRequest":
        if not self.message and not self.image_url:
            raise ValueError("至少提供 message 或 image_url 之一")
        return self


class FoodRecordDraft(FoodRecordAnalysis):
    """等待用户确认的饮食草稿，不参与统计。"""

    draft_id: str = Field(default_factory=lambda: uuid4().hex)
    user_id: str
    source: Literal["text", "image", "text_and_image"]
    image_url: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: Literal["pending_confirmation", "confirmed"] = "pending_confirmation"


class ConfirmFoodRecordRequest(BaseModel):
    """用户确认草稿时可修正餐次和食物项。"""

    user_id: str = Field(min_length=1, max_length=128)
    confirmed: bool
    meal_type: MealType | None = None
    foods: list[FoodItem] | None = Field(default=None, min_length=1, max_length=20)
    eaten_at: datetime | None = None


class FoodRecord(FoodRecordDraft):
    """用户确认后的饮食账本记录。"""

    record_id: str = Field(default_factory=lambda: uuid4().hex)
    eaten_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    confirmed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: Literal["confirmed"] = "confirmed"


class DailyNutritionSummary(BaseModel):
    """用户指定日期已确认饮食记录的汇总。"""

    user_id: str
    date: date
    records: list[FoodRecord] = Field(default_factory=list)
    total_nutrition: NutritionEstimate

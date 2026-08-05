"""中医科普 Agent 的受控输入输出模型。"""

from typing import Literal

from pydantic import BaseModel, Field


class TCMWellnessAction(BaseModel):
    """一条低风险、可立即执行的日常养生动作。"""

    title: str = Field(min_length=1, max_length=40)
    detail: str = Field(min_length=1, max_length=180)


class TCMWellnessSource(BaseModel):
    """检索知识片段的可追溯来源。"""

    source_id: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=120)


class TCMWellnessOutput(BaseModel):
    """中医科普子 Agent 返回给主 Agent 的结构化结果。"""

    scope: Literal["wellness_education", "medical_refusal"] = "wellness_education"
    title: str = Field(min_length=1, max_length=50)
    summary: str = Field(min_length=1, max_length=300)
    actions: list[TCMWellnessAction] = Field(default_factory=list, max_length=3)
    personalization_basis: list[str] = Field(default_factory=list, max_length=4)
    safety_notice: str = Field(min_length=1, max_length=240)
    sources: list[TCMWellnessSource] = Field(default_factory=list, max_length=3)

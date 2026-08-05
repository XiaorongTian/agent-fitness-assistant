"""中医 RAG 的受控查询重写：只改善召回表达，不生成健康建议。"""

from __future__ import annotations

import json
import logging
from typing import Any
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.chat_agent import get_chat_model

logger = logging.getLogger("fitness_assistant")
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


class TCMQueryRewrite(BaseModel):
    """供检索使用的单条独立查询。"""

    query: str = Field(min_length=2, max_length=220)


def _retrieval_context(user_context: dict[str, Any]) -> dict[str, Any]:
    weather = user_context.get("weather") or {}
    profile = user_context.get("profile") or {}
    return {
        "season": user_context.get("season"),
        "weather": {
            "condition": weather.get("condition"),
            "temperature_c": weather.get("temperature_c"),
            "apparent_temperature_c": weather.get("apparent_temperature_c"),
        },
        "goal": profile.get("goal"),
        "exercise_limitations": profile.get("exercise_limitations", []),
        "food_restrictions": profile.get("food_restrictions", []),
    }


async def rewrite_tcm_retrieval_query(question: str, user_context: dict[str, Any]) -> str:
    """结合可信上下文将口语问题改成可检索表达；失败时保留原问题。"""
    question = question.strip()
    if not question:
        return question
    prompt = (
        "用户原问题：\n"
        f"{question}\n\n"
        "可信上下文：\n"
        f"{json.dumps(_retrieval_context(user_context), ensure_ascii=False)}"
    )
    try:
        model = get_chat_model().with_structured_output(TCMQueryRewrite)
        result = await model.ainvoke(
            [
                SystemMessage(
                    content=(
                        "你是 RAG 查询重写器。只输出 JSON Schema 对应的 query 字段。"
                        "将用户口语问题改写成适合检索中医日常养生文档的一句中文查询。"
                        "只能补充提供的季节、天气和生活方式上下文；不得加入病名、体质、药方、剂量或任何答案。"
                    )
                ),
                HumanMessage(content=prompt),
            ]
        )
        rewrite = result.query if isinstance(result, TCMQueryRewrite) else TCMQueryRewrite.model_validate(result).query
        return rewrite.strip() or question
    except Exception as exc:
        logger.warning("tcm_rag_stage=query_rewrite_fallback error_type=%s", type(exc).__name__)
        return question

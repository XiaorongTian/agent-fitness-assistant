"""使用当前 DashScope 多模态模型生成待确认的饮食识别草稿。"""

import json
import os
from time import perf_counter

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from common.logger import logger
from schemas.diet import CreateFoodRecordDraftRequest, FoodRecordAnalysis


def _food_analysis_model():
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("未配置 DASHSCOPE_API_KEY，无法分析饮食记录")
    return init_chat_model(
        model=os.getenv("FOOD_ANALYSIS_MODEL", os.getenv("CHAT_MODEL", "qwen3.5-plus")),
        model_provider="openai",
        api_key=api_key,
        base_url=os.getenv("DASHSCOPE_BASE_URL") or None,
        extra_body={"enable_thinking": False},
    ).with_structured_output(FoodRecordAnalysis)


async def analyze_food_record(request: CreateFoodRecordDraftRequest) -> FoodRecordAnalysis:
    """分析文字和/或图片，但永远只返回待用户确认的估算结果。"""
    started_at = perf_counter()
    logger.info(
        "food_analysis_stage=start user_id=%s has_text=%s has_image=%s",
        request.user_id,
        bool(request.message),
        bool(request.image_url),
    )
    output_schema = json.dumps(FoodRecordAnalysis.model_json_schema(), ensure_ascii=False)
    instructions = (
        "你是饮食记录分析器。根据用户的文字和图片识别食物、估算份量及营养区间。"
        "不要把估算当作精确事实；油、糖、酱料、主食份量不明时扩大区间并写入 assumptions。"
        "每项食物都要有 nutrition；不得只返回整餐营养汇总。"
        "必须同时返回 meal_type、foods、assumptions、summary 四个字段，foods 至少包含一项，summary 是简短中文总结。"
        "nutrition 必须包含 calories_kcal_min、calories_kcal_max、protein_g_min、protein_g_max、"
        "carbohydrate_g_min、carbohydrate_g_max、fat_g_min、fat_g_max 八个数值字段。"
        "只输出完全符合以下 JSON Schema 的 JSON 对象，不要输出 Markdown 或其他文字："
        f"{output_schema}"
    )
    content: list[dict] = [{"type": "text", "text": f"{instructions}\n用户描述：{request.message or '未提供文字描述'}"}]
    if request.image_url:
        content.append({"type": "image_url", "image_url": {"url": request.image_url}})
    result = await _food_analysis_model().ainvoke([HumanMessage(content=content)])
    analysis = result if isinstance(result, FoodRecordAnalysis) else FoodRecordAnalysis.model_validate(result)
    if request.meal_type:
        analysis.meal_type = request.meal_type
    logger.info(
        "food_analysis_stage=end user_id=%s food_count=%s confidence=%s duration_ms=%s",
        request.user_id,
        len(analysis.foods),
        analysis.confidence,
        round((perf_counter() - started_at) * 1000),
    )
    return analysis

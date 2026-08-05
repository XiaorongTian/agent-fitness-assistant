"""饮食草稿和已确认饮食账本的持久化服务。"""

from datetime import date, datetime, timezone

from memory.runtime import conversation_runtime
from schemas.diet import (
    ConfirmFoodRecordRequest,
    FoodItem,
    FoodRecord,
    FoodRecordDraft,
    NutritionEstimate,
)

DIET_DRAFT_NAMESPACE = "diet_drafts_v1"
DIET_RECORD_NAMESPACE = "diet_records_v1"


def _draft_namespace(user_id: str) -> tuple[str, str, str]:
    return ("user", user_id, DIET_DRAFT_NAMESPACE)


def _record_namespace(user_id: str, recorded_date: date) -> tuple[str, str, str, str]:
    return ("user", user_id, DIET_RECORD_NAMESPACE, recorded_date.isoformat())


def _total_nutrition(foods: list[FoodItem]) -> NutritionEstimate:
    """聚合食物项营养区间，避免以单点数值伪装精确性。"""
    fields = (
        "calories_kcal_min", "calories_kcal_max", "protein_g_min", "protein_g_max",
        "carbohydrate_g_min", "carbohydrate_g_max", "fat_g_min", "fat_g_max",
    )
    values = {field: sum(getattr(food.nutrition, field) for food in foods) for field in fields}
    return NutritionEstimate(**values)


async def save_food_draft(draft: FoodRecordDraft) -> FoodRecordDraft:
    await conversation_runtime.start()
    await conversation_runtime.store.aput(
        _draft_namespace(draft.user_id), draft.draft_id, draft.model_dump(mode="json")
    )
    return draft


async def confirm_food_draft(draft_id: str, request: ConfirmFoodRecordRequest) -> FoodRecord:
    """仅允许草稿所属用户确认；确认后才写入按日组织的饮食账本。"""
    if not request.confirmed:
        raise ValueError("用户确认后才可写入饮食账本")
    await conversation_runtime.start()
    item = await conversation_runtime.store.aget(_draft_namespace(request.user_id), draft_id)
    if not item:
        raise LookupError("未找到该饮食草稿，或草稿不属于当前用户")
    draft = FoodRecordDraft.model_validate(item.value)
    if draft.status != "pending_confirmation":
        raise ValueError("该饮食草稿已经确认，不能重复入账")

    foods = request.foods or draft.foods
    eaten_at = request.eaten_at or datetime.now(timezone.utc)
    record_data = draft.model_dump(exclude={"status"})
    record_data.update(
        meal_type=request.meal_type or draft.meal_type,
        foods=[food.model_dump(mode="json") for food in foods],
        eaten_at=eaten_at,
    )
    record = FoodRecord(**record_data)
    await conversation_runtime.store.aput(
        _record_namespace(request.user_id, eaten_at.date()),
        record.record_id,
        record.model_dump(mode="json"),
    )
    draft.status = "confirmed"
    await conversation_runtime.store.aput(
        _draft_namespace(request.user_id), draft.draft_id, draft.model_dump(mode="json")
    )
    return record


async def get_daily_nutrition_records(user_id: str, recorded_date: date) -> tuple[list[FoodRecord], NutritionEstimate]:
    """读取某天全部已确认饮食记录，并计算合计营养区间。"""
    await conversation_runtime.start()
    items = await conversation_runtime.store.asearch(
        _record_namespace(user_id, recorded_date), limit=100
    )
    records = [FoodRecord.model_validate(item.value) for item in items]
    foods = [food for record in records for food in record.foods]
    return records, _total_nutrition(foods) if foods else NutritionEstimate(
        calories_kcal_min=0, calories_kcal_max=0,
        protein_g_min=0, protein_g_max=0,
        carbohydrate_g_min=0, carbohydrate_g_max=0,
        fat_g_min=0, fat_g_max=0,
    )

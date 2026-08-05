"""饮食记录 HTTP 接口：识别草稿、用户确认和当日汇总。"""

from datetime import date

from fastapi import APIRouter, HTTPException

from diet.analyzer import analyze_food_record
from memory.diet import confirm_food_draft, get_daily_nutrition_records, save_food_draft
from schemas.diet import (
    ConfirmFoodRecordRequest,
    CreateFoodRecordDraftRequest,
    DailyNutritionSummary,
    FoodRecord,
    FoodRecordDraft,
)

router = APIRouter()


@router.post("/diet/drafts", response_model=FoodRecordDraft)
async def create_food_draft(request: CreateFoodRecordDraftRequest) -> FoodRecordDraft:
    """从文本或图片创建待确认饮食草稿；草稿不参与营养统计。"""
    try:
        analysis = await analyze_food_record(request)
        source = "text_and_image" if request.message and request.image_url else "image" if request.image_url else "text"
        draft = FoodRecordDraft(
            **analysis.model_dump(),
            user_id=request.user_id,
            source=source,
            image_url=request.image_url,
        )
        return await save_food_draft(draft)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"饮食识别失败：{type(exc).__name__}: {exc}") from exc


@router.post("/diet/drafts/{draft_id}/confirm", response_model=FoodRecord)
async def confirm_food_record(draft_id: str, request: ConfirmFoodRecordRequest) -> FoodRecord:
    """确认或修正饮食草稿后写入用户饮食账本。"""
    try:
        return await confirm_food_draft(draft_id, request)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/diet/records/daily", response_model=DailyNutritionSummary)
async def get_daily_food_records(user_id: str, target_date: date | None = None) -> DailyNutritionSummary:
    """读取指定日期（默认当天）的已确认饮食记录和营养汇总。"""
    selected_date = target_date or date.today()
    records, total_nutrition = await get_daily_nutrition_records(user_id, selected_date)
    return DailyNutritionSummary(
        user_id=user_id,
        date=selected_date,
        records=records,
        total_nutrition=total_nutrition,
    )

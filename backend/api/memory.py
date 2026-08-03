"""Explicit, user-confirmed long-term memory endpoints."""

from fastapi import APIRouter, HTTPException

from memory.profile import delete_health_profile, get_health_profile, save_health_profile
from schemas.memory import HealthProfile, SaveHealthProfileRequest

router = APIRouter()


@router.get("/memory/profile", response_model=HealthProfile)
async def read_profile(user_id: str) -> HealthProfile:
    return await get_health_profile(user_id)


@router.put("/memory/profile", response_model=HealthProfile)
async def write_profile(request: SaveHealthProfileRequest) -> HealthProfile:
    if not request.confirmed:
        raise HTTPException(status_code=400, detail="用户确认后才可写入长期记忆")
    return await save_health_profile(request.user_id, request.profile)


@router.delete("/memory/profile")
async def remove_profile(user_id: str) -> dict[str, str]:
    await delete_health_profile(user_id)
    return {"message": "长期记忆已成功删除"}

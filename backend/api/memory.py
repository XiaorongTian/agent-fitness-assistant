"""长期记忆 HTTP 接口，负责健康档案的读取、写入和删除。"""

from fastapi import APIRouter, HTTPException

from memory.profile import delete_health_profile, get_health_profile, save_health_profile
from schemas.memory import HealthProfile, SaveHealthProfileRequest

router = APIRouter()


@router.get("/memory/profile", response_model=HealthProfile)
async def read_profile(user_id: str) -> HealthProfile:
    """读取指定用户已确认的健康档案。"""
    return await get_health_profile(user_id)


@router.put("/memory/profile", response_model=HealthProfile)
async def write_profile(request: SaveHealthProfileRequest) -> HealthProfile:
    """在用户确认后保存长期健康档案。"""
    if not request.confirmed:
        raise HTTPException(status_code=400, detail="用户确认后才可写入长期记忆")
    return await save_health_profile(request.user_id, request.profile)


@router.delete("/memory/profile")
async def remove_profile(user_id: str) -> dict[str, str]:
    """删除指定用户的长期健康档案。"""
    await delete_health_profile(user_id)
    return {"message": "长期记忆已成功删除"}

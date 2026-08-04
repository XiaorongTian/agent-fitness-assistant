"""长期健康档案读写模块，封装对 LangGraph store 的访问。"""

from datetime import datetime, timezone

from memory.runtime import PROFILE_KEY, PROFILE_NAMESPACE, conversation_runtime
from schemas.memory import HealthProfile


def _namespace(user_id: str) -> tuple[str, str, str]:
    """生成用户健康档案在 store 中的命名空间。"""
    return ("user", user_id, PROFILE_NAMESPACE)


async def get_health_profile(user_id: str) -> HealthProfile:
    """读取用户长期健康档案；不存在时返回空档案。"""
    await conversation_runtime.start()
    item = await conversation_runtime.store.aget(_namespace(user_id), PROFILE_KEY)
    return HealthProfile.model_validate(item.value) if item else HealthProfile()


async def save_health_profile(user_id: str, profile: HealthProfile) -> HealthProfile:
    """保存用户确认后的健康档案，并刷新更新时间。"""
    await conversation_runtime.start()
    profile.updated_at = datetime.now(timezone.utc)
    profile.source = "user_confirmed"
    await conversation_runtime.store.aput(
        _namespace(user_id), PROFILE_KEY, profile.model_dump(mode="json")
    )
    return profile


async def delete_health_profile(user_id: str) -> None:
    """删除用户长期健康档案，不影响会话历史。"""
    await conversation_runtime.start()
    await conversation_runtime.store.adelete(_namespace(user_id), PROFILE_KEY)

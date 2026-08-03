"""长期记忆的读写"""

from datetime import datetime, timezone

from memory.runtime import PROFILE_KEY, PROFILE_NAMESPACE, conversation_runtime
from schemas.memory import HealthProfile


def _namespace(user_id: str) -> tuple[str, str, str]:
    return ("user", user_id, PROFILE_NAMESPACE)

# 长期记忆-读
async def get_health_profile(user_id: str) -> HealthProfile:
    await conversation_runtime.start()
    item = await conversation_runtime.store.aget(_namespace(user_id), PROFILE_KEY)
    return HealthProfile.model_validate(item.value) if item else HealthProfile()

# 长期记忆-写
async def save_health_profile(user_id: str, profile: HealthProfile) -> HealthProfile:
    """Replace the profile only after the caller has obtained user confirmation."""
    await conversation_runtime.start()
    profile.updated_at = datetime.now(timezone.utc)
    profile.source = "user_confirmed"
    await conversation_runtime.store.aput(
        _namespace(user_id), PROFILE_KEY, profile.model_dump(mode="json")
    )
    return profile

# 长期记忆-删除
async def delete_health_profile(user_id: str) -> None:
    """Delete the user's cross-session health profile, not conversation history."""
    await conversation_runtime.start()
    await conversation_runtime.store.adelete(_namespace(user_id), PROFILE_KEY)

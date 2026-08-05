"""把用户明确请求的运动任务创建能力暴露给 Chat Agent。"""

from typing import Any

from langchain.tools import ToolRuntime, tool

from common.tool_logging import log_tool_end, log_tool_start
from schemas.context import AgentContext
from schemas.exercise import CreateExerciseTaskRequest

EXERCISE_TASK_TOOL_NAMES = {"create_user_requested_exercise_task"}


def build_exercise_task_tools() -> list[Any]:
    """仅在用户明确要创建/保存任务时由 Agent 调用的写入型工具。"""

    @tool
    async def create_user_requested_exercise_task(
        runtime: ToolRuntime[AgentContext],
        available_minutes: int = 30,
        city: str | None = None,
        preferred_activity: str | None = None,
    ) -> dict[str, Any]:
        """为当前用户创建一项可反馈的运动任务。

        仅当用户明确要求“创建/生成/保存一个运动任务或计划”时调用；
        只问建议、尚未决定执行时，应使用普通运动建议工具而不是创建任务。
        """
        user_id = runtime.context.user_id if runtime and runtime.context else None
        if not user_id:
            return {"error": "无法确定当前用户，不能创建运动任务。"}
        arguments = {
            "user_id": user_id,
            "available_minutes": available_minutes,
            "city": city,
            "preferred_activity": preferred_activity,
        }
        log_tool_start("create_user_requested_exercise_task", arguments)
        # 延迟导入以避免 Agent 初始化阶段的循环依赖。
        from exercise.service import create_user_requested_task

        task = await create_user_requested_task(CreateExerciseTaskRequest(**arguments))
        result = task.model_dump(mode="json")
        log_tool_end("create_user_requested_exercise_task", result)
        return result

    return [create_user_requested_exercise_task]

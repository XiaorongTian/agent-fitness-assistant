"""运动任务账本的持久化与历史读取。"""

from datetime import date, datetime, timedelta, timezone

from memory.runtime import conversation_runtime
from schemas.exercise import ExerciseTask, StartExerciseTaskRequest, SubmitExerciseFeedbackRequest

EXERCISE_TASK_NAMESPACE = "exercise_tasks_v1"


def _namespace(user_id: str, task_date: date) -> tuple[str, str, str, str]:
    return ("user", user_id, EXERCISE_TASK_NAMESPACE, task_date.isoformat())


async def save_exercise_task(task: ExerciseTask) -> ExerciseTask:
    await conversation_runtime.start()
    await conversation_runtime.store.aput(
        _namespace(task.user_id, task.created_at.date()),
        task.task_id,
        task.model_dump(mode="json"),
    )
    return task


async def get_exercise_task(user_id: str, task_id: str, days_back: int = 30) -> ExerciseTask | None:
    """在有限的最近日期窗口中查找任务，避免跨用户读取。"""
    await conversation_runtime.start()
    today = datetime.now(timezone.utc).date()
    for offset in range(days_back + 1):
        item = await conversation_runtime.store.aget(_namespace(user_id, today - timedelta(days=offset)), task_id)
        if item:
            return ExerciseTask.model_validate(item.value)
    return None


async def submit_exercise_feedback(task_id: str, request: SubmitExerciseFeedbackRequest) -> ExerciseTask:
    task = await get_exercise_task(request.user_id, task_id)
    if not task:
        raise LookupError("未找到该运动任务，或任务不属于当前用户")
    expected_status = "pending" if request.status == "skipped" else "in_progress"
    if task.status != expected_status:
        raise ValueError("请先按任务流程开始执行，或该任务已提交反馈")
    task.status = request.status
    task.actual_minutes = request.actual_minutes
    task.fatigue_score = request.fatigue_score
    task.barrier_reason = request.barrier_reason
    task.feedback_note = request.feedback_note.strip() if request.feedback_note else None
    task.feedback_at = datetime.now(timezone.utc)
    await save_exercise_task(task)
    return task


async def start_exercise_task(task_id: str, request: StartExerciseTaskRequest) -> ExerciseTask:
    """把待办运动任务标记为执行中；重复开始或已反馈任务会被拒绝。"""
    task = await get_exercise_task(request.user_id, task_id)
    if not task:
        raise LookupError("未找到该运动任务，或任务不属于当前用户")
    if task.status != "pending":
        raise ValueError("该运动任务不是待开始状态，不能重复开始")
    task.status = "in_progress"
    task.started_at = datetime.now(timezone.utc)
    await save_exercise_task(task)
    return task


async def get_recent_exercise_tasks(user_id: str, days: int = 7) -> list[ExerciseTask]:
    """读取最近若干天任务，供下一次计划的自适应规则使用。"""
    await conversation_runtime.start()
    today = datetime.now(timezone.utc).date()
    tasks: list[ExerciseTask] = []
    for offset in range(days):
        items = await conversation_runtime.store.asearch(
            _namespace(user_id, today - timedelta(days=offset)), limit=50
        )
        tasks.extend(ExerciseTask.model_validate(item.value) for item in items)
    return sorted(tasks, key=lambda task: task.created_at, reverse=True)

"""用户主动生成运动任务和提交执行反馈的 HTTP 接口。"""

from fastapi import APIRouter, HTTPException

from exercise.service import create_user_requested_task, describe_next_task_adjustment
from memory.exercise import (
    get_recent_exercise_tasks,
    save_exercise_task,
    start_exercise_task,
    submit_exercise_feedback,
)
from schemas.exercise import (
    CreateExerciseTaskRequest,
    ExerciseTask,
    StartExerciseTaskRequest,
    SubmitExerciseFeedbackRequest,
)

router = APIRouter()


@router.post("/exercise/tasks", response_model=ExerciseTask)
async def create_exercise_task(request: CreateExerciseTaskRequest) -> ExerciseTask:
    """由用户主动请求创建一项运动任务，不会发送定时推送。"""
    try:
        return await create_user_requested_task(request)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"运动任务生成失败：{type(exc).__name__}: {exc}") from exc


@router.post("/exercise/tasks/{task_id}/feedback", response_model=ExerciseTask)
async def submit_task_feedback(task_id: str, request: SubmitExerciseFeedbackRequest) -> ExerciseTask:
    """接收前端“完成/跳过/太难/不喜欢”按钮的执行反馈。"""
    try:
        task = await submit_exercise_feedback(task_id, request)
        task.next_task_adjustment = describe_next_task_adjustment(task)
        recent_tasks = await get_recent_exercise_tasks(request.user_id, days=7)
        skipped_count = sum(item.status == "skipped" for item in recent_tasks)
        if task.status == "skipped" and not task.barrier_reason and skipped_count >= 2:
            task.follow_up_question = "这次主要是时间不够、身体疲劳、天气/场地不便，还是动作不适合？"
        await save_exercise_task(task)
        return task
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/exercise/tasks/{task_id}/start", response_model=ExerciseTask)
async def start_task(task_id: str, request: StartExerciseTaskRequest) -> ExerciseTask:
    """用户确认开始执行任务后，将状态从 pending 切换为 in_progress。"""
    try:
        return await start_exercise_task(task_id, request)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

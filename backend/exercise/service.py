"""用户主动运动任务生成及基于历史反馈的可解释调整规则。"""

from dataclasses import dataclass

from memory.exercise import get_recent_exercise_tasks, save_exercise_task
from memory.runtime import conversation_runtime
from schemas.exercise import CreateExerciseTaskRequest, ExerciseIntensity, ExerciseTask
from tools.health_tools import (
    DEFAULT_ACTIVITY_LOCATION,
    _is_bad_outdoor_weather,
    _weather_query_from_location,
    fetch_today_weather,
)


@dataclass
class Adjustment:
    minutes: int
    intensity: ExerciseIntensity
    reason: str
    avoid_disliked_activity: bool = False


def _adjust_from_history(minutes: int, history: list[ExerciseTask]) -> Adjustment:
    """依据最近 7 天反馈调整下一次任务，规则优先于模型主观判断。"""
    feedback = [task for task in history if task.status != "pending"]
    if not feedback:
        return Adjustment(minutes, ExerciseIntensity.MODERATE, "暂无历史反馈，使用适中的起步强度。")

    recent = feedback[:3]
    too_hard_count = sum(task.status == "too_hard" for task in recent)
    skipped_count = sum(task.status == "skipped" for task in recent)
    disliked_count = sum(task.status == "disliked" for task in recent)
    fatigue_scores = [task.fatigue_score for task in recent if task.fatigue_score is not None]
    average_fatigue = sum(fatigue_scores) / len(fatigue_scores) if fatigue_scores else 0

    if too_hard_count or average_fatigue >= 4:
        return Adjustment(max(10, round(minutes * 0.7)), ExerciseIntensity.LOW, "近期反馈显示任务偏难或疲劳较高，已降低强度和时长。")
    if skipped_count >= 2:
        return Adjustment(max(10, round(minutes * 0.75)), ExerciseIntensity.LOW, "近期多次未执行，先缩短任务以降低开始门槛；完成后再逐步恢复。")
    if disliked_count:
        return Adjustment(minutes, ExerciseIntensity.LOW, "近期反馈不喜欢类似任务，本次优先更换运动类型。", True)

    completed = [task for task in feedback if task.status == "completed"]
    completion_rate = len(completed) / len(feedback)
    if len(feedback) >= 3 and completion_rate >= 0.75 and average_fatigue and average_fatigue <= 3:
        return Adjustment(min(90, round(minutes * 1.1)), ExerciseIntensity.MODERATE, "近期完成率较高且疲劳可接受，时长小幅增加约 10%。")
    return Adjustment(minutes, ExerciseIntensity.MODERATE, "根据近期执行情况维持当前难度。")


async def create_user_requested_task(request: CreateExerciseTaskRequest) -> ExerciseTask:
    """根据档案、天气和历史反馈创建一项用户主动请求的运动任务。"""
    profile = await conversation_runtime.get_profile_value(request.user_id)
    history = await get_recent_exercise_tasks(request.user_id)
    adjustment = _adjust_from_history(request.available_minutes, history)
    limitations = [str(value) for value in profile.get("exercise_limitations", [])]
    city = (request.city or str(profile.get("city") or "")).strip()
    location_source = "request_city" if request.city else "profile_city" if city else "default_activity_location"
    weather_location = city or _weather_query_from_location(DEFAULT_ACTIVITY_LOCATION)
    weather = fetch_today_weather(weather_location) if weather_location else {"error": "未设置城市或活动地点，未使用天气调整"}
    if weather_location:
        weather["location_source"] = location_source
        weather["weather_query"] = weather_location
    limitation_text = " ".join(limitations).lower()
    needs_low_impact = any(keyword in limitation_text for keyword in ("膝", "踝", "腰", "疼", "伤", "knee", "ankle", "pain"))
    indoor = _is_bad_outdoor_weather(weather) or needs_low_impact
    preferred = (request.preferred_activity or "").strip()
    if preferred and not adjustment.avoid_disliked_activity:
        activity = preferred
    elif indoor:
        activity = "室内快走/原地踏步、弹力带划船和温和拉伸"
    else:
        activity = "户外快走或轻松骑行，结束后进行温和拉伸"
    if needs_low_impact:
        adjustment.intensity = ExerciseIntensity.LOW
        adjustment.reason += " 已识别运动限制，优先低冲击动作。"
    task = ExerciseTask(
        user_id=request.user_id,
        title="今日主动运动任务",
        activity=activity,
        planned_minutes=adjustment.minutes,
        intensity=adjustment.intensity,
        plan=(
            f"{adjustment.minutes} 分钟：5 分钟热身；"
            f"{max(adjustment.minutes - 10, 5)} 分钟{activity}；5 分钟放松。"
        ),
        weather=weather,
        limitations_applied=limitations,
        adaptation_reason=adjustment.reason,
    )
    return await save_exercise_task(task)


def describe_next_task_adjustment(task: ExerciseTask) -> str | None:
    """反馈写入后给用户一个不评判的下一次调整预告。"""
    if task.status == "completed":
        return "已记录完成情况；下次会结合你的疲劳评分和近期完成率安排。"
    if task.status == "too_hard":
        return "下次将优先降低时长和冲击强度；如有疼痛，请停止相关动作并考虑咨询专业人士。"
    if task.status == "disliked":
        return "下次会优先更换运动类型，不重复推荐你不喜欢的活动。"
    return "已记录本次未完成；若近期再次跳过，系统会询问时间、疲劳或场地等阻碍因素，而不会简单归因于自律问题。"

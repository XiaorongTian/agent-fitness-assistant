"""主健康 Agent 调用中医科普子 Agent 的委派工具。"""

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from langchain.tools import ToolRuntime, tool

from common.tool_logging import log_tool_end, log_tool_start
from schemas.context import AgentContext

TCM_AGENT_TOOL_NAMES = {"consult_tcm_wellness_agent"}


def _season_label(now: datetime) -> str:
    """提供保守季节上下文；精确节气将在定时推送任务中单独接入。"""
    month = now.month
    if month in (3, 4, 5):
        return "春季"
    if month in (6, 7, 8):
        return "夏季"
    if month in (9, 10, 11):
        return "秋季"
    return "冬季"


def build_tcm_agent_tools() -> list[Any]:
    """构建主 Agent 委派中医科普子 Agent 的工具。"""

    @tool
    async def consult_tcm_wellness_agent(
        runtime: ToolRuntime[AgentContext],
        question: str,
    ) -> dict[str, Any]:
        """委派给中医科普子 Agent，回答节气养生、起居饮食调养等非诊断性问题。

        仅适用于中医日常科普、节气养生、作息调养、温和活动和饮食习惯问题。
        不适用于疾病诊断、症状判断、开方用药、针灸治疗或急症；这些情况应直接给出就医建议。
        """
        context = runtime.context if runtime and runtime.context else None
        user_id = context.user_id if context else None
        if not user_id:
            return {"error": "无法确定当前用户，不能调用中医科普 Agent。"}

        log_tool_start("consult_tcm_wellness_agent", {"user_id": user_id, "question_length": len(question)})
        try:
            from agents.tcm_agent import consult_tcm_wellness_agent as run_tcm_agent
            from memory.diet import get_daily_nutrition_records
            from memory.exercise import get_recent_exercise_tasks
            from tools.health_tools import fetch_today_weather, resolve_weather_location

            now = datetime.now(ZoneInfo("Asia/Shanghai"))
            profile = context.profile or {}
            city, location_source = resolve_weather_location(runtime)
            weather = fetch_today_weather(city) if city else {"error": "未设置城市或活动地点"}
            if city:
                weather["location_source"] = location_source
            _, nutrition = await get_daily_nutrition_records(user_id, now.date())
            recent_tasks = await get_recent_exercise_tasks(user_id, days=7)
            completed_count = sum(task.status == "completed" for task in recent_tasks)
            user_context = {
                "date": now.date().isoformat(),
                "season": _season_label(now),
                "solar_term": "未接入精确节气服务，不应声称具体节气",
                "profile": {
                    "goal": profile.get("goal"),
                    "food_restrictions": profile.get("food_restrictions", []),
                    "exercise_limitations": profile.get("exercise_limitations", []),
                    "preferences": profile.get("preferences", []),
                },
                "weather": weather,
                "today_nutrition": nutrition.model_dump(mode="json"),
                "recent_exercise": {
                    "task_count": len(recent_tasks),
                    "completed_count": completed_count,
                    "recent_fatigue_scores": [task.fatigue_score for task in recent_tasks if task.fatigue_score is not None][:3],
                },
            }
            result = await run_tcm_agent(question, user_context)
        except Exception as exc:
            result = {"error": f"中医科普 Agent 调用失败：{type(exc).__name__}"}
        log_tool_end("consult_tcm_wellness_agent", result)
        return result

    return [consult_tcm_wellness_agent]

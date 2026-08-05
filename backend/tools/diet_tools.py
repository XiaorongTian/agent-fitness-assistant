"""饮食账本只读工具，供健康 Agent 基于已确认记录给出建议。"""

from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from langchain.tools import ToolRuntime, tool

from common.tool_logging import log_tool_end, log_tool_start
from schemas.context import AgentContext

DIET_TOOL_NAMES = {"get_daily_diet_summary"}


def build_diet_tools() -> list[Any]:
    """构建饮食账本工具；工具只读已确认记录，绝不读取草稿。"""

    @tool
    async def get_daily_diet_summary(
        runtime: ToolRuntime[AgentContext],
        target_date: date | None = None,
    ) -> dict[str, Any]:
        """读取当前用户某天已确认的饮食记录与营养区间汇总。

        当用户询问今天已吃什么、蛋白质是否足够、当天剩余餐如何安排、
        热量或饮食复盘时必须调用。未确认草稿不会被返回或参与统计。
        """
        user_id = runtime.context.user_id if runtime and runtime.context else None
        if not user_id:
            return {"error": "无法确定当前用户，不能读取饮食账本。"}
        selected_date = target_date or datetime.now(ZoneInfo("Asia/Shanghai")).date()
        log_tool_start("get_daily_diet_summary", {"user_id": user_id, "date": selected_date.isoformat()})
        # 延迟导入以避免 memory.runtime 与工具注册阶段产生循环依赖。
        from memory.diet import get_daily_nutrition_records

        records, total_nutrition = await get_daily_nutrition_records(user_id, selected_date)
        result = {
            "date": selected_date.isoformat(),
            "record_count": len(records),
            "records": [record.model_dump(mode="json") for record in records],
            "total_nutrition": total_nutrition.model_dump(mode="json"),
            "note": "仅包含用户已确认的饮食记录；没有记录不等于用户没有进食。",
        }
        log_tool_end("get_daily_diet_summary", result)
        return result

    return [get_daily_diet_summary]

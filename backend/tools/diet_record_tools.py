"""把对话中的明确饮食陈述转换为待用户确认的饮食草稿。"""

from typing import Any

from langchain.tools import ToolRuntime, tool

from common.logger import logger
from common.tool_logging import log_tool_end, log_tool_start
from schemas.context import AgentContext
from schemas.diet import CreateFoodRecordDraftRequest

DIET_RECORD_TOOL_NAMES = {"create_food_record_draft"}


def build_diet_record_tools() -> list[Any]:
    """构建饮食草稿工具；仅创建草稿，绝不自动写入正式饮食账本。"""

    @tool
    async def create_food_record_draft(
        runtime: ToolRuntime[AgentContext],
        description: str,
        meal_type: str | None = None,
    ) -> dict[str, Any]:
        """将用户明确陈述的实际饮食转换为待确认草稿。

        仅在用户说自己已经吃了、喝了某些食物时调用，例如“早餐吃了两个鸡蛋”。
        不得用于泛泛询问“能不能吃”“应该吃什么”或尚未实际进食的计划。
        meal_type 只能使用 breakfast、lunch、dinner、snack 或 unknown；不确定时不传。
        返回草稿后，必须告知用户确认或修改后才会正式入账。
        """
        user_id = runtime.context.user_id if runtime and runtime.context else None
        if not user_id:
            return {"error": "无法确定当前用户，不能创建饮食草稿。"}
        image_url = runtime.context.food_image_url if runtime and runtime.context else None
        arguments = {
            "user_id": user_id,
            "description_length": len(description),
            "meal_type": meal_type,
            "has_image": bool(image_url),
        }
        log_tool_start("create_food_record_draft", arguments)
        try:
            # 延迟导入可避免 Agent 初始化阶段与运行时存储产生循环依赖。
            from diet.analyzer import analyze_food_record
            from memory.diet import save_food_draft
            from schemas.diet import FoodRecordDraft, MealType

            parsed_meal_type = MealType(meal_type) if meal_type else None
            analysis = await analyze_food_record(
                CreateFoodRecordDraftRequest(
                    user_id=user_id,
                    message=description,
                    image_url=image_url,
                    meal_type=parsed_meal_type,
                )
            )
            draft = FoodRecordDraft(
                **analysis.model_dump(),
                user_id=user_id,
                source="text_and_image" if description and image_url else "image" if image_url else "text",
                image_url=image_url,
            )
            result = (await save_food_draft(draft)).model_dump(mode="json")
        except ValueError as exc:
            result = {"error": f"饮食草稿参数无效：{exc}"}
        except Exception as exc:
            logger.exception("food_record_draft_stage=failed user_id=%s", user_id)
            result = {"error": f"饮食草稿创建失败：{type(exc).__name__}"}
        log_tool_end("create_food_record_draft", result)
        return result

    return [create_food_record_draft]

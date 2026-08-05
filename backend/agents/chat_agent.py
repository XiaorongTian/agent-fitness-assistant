"""健康 Agent 构建模块，配置模型、动态提示词、记忆和结构化输出。"""

import json
import os
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, SummarizationMiddleware, dynamic_prompt
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model

from prompts.chat import CHAT_SYSTEM_PROMPT
from schemas.chat import HealthAssistantOutput
from schemas.context import AgentContext
from tools.registry import build_agent_tools

SUMMARY_TRIGGER_TOKENS = int(os.getenv("SUMMARY_TRIGGER_TOKENS", "4000"))
SUMMARY_MESSAGES_TO_KEEP = int(os.getenv("SUMMARY_MESSAGES_TO_KEEP", "6"))

def get_chat_model():
    """根据环境变量构建 DashScope OpenAI 兼容对话模型。"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("未配置 DASHSCOPE_API_KEY，无法调用对话模型")

    return init_chat_model(
        model=os.getenv("CHAT_MODEL", "qqwen3.5-plus-2026-02-15"),
        model_provider = "openai", 
        api_key=api_key,
        base_url=os.getenv("DASHSCOPE_BASE_URL") or None,
        # 结构化输出会强制工具调用，Qwen 思考模式不兼容。
        extra_body={"enable_thinking": False},
    )


@dynamic_prompt
def health_system_prompt(request: ModelRequest) -> str:
    """根据运行时健康档案生成每次模型调用使用的系统提示词。"""
    context = request.runtime.context
    profile = context.profile if context else {}
    profile_context = (
        "\n\n以下是可信的用户长期健康档案（可能为空）：\n"
        f"{json.dumps(profile, ensure_ascii=False)}\n"
        "仅在与当前问题有关时使用；档案中的限制优先于一般建议。"
    )
    location_context = ""
    if context and context.activity_location:
        source_label = "用户本轮授权提供" if context.location_source == "user_consent" else "开发环境 mock"
        location_context = (
            "\n\n位置服务上下文：\n"
            f"- 当前活动地点：{context.activity_location}（来源：{source_label}）。\n"
            "- 仅当用户询问附近运动地点、户外运动或路线时，才使用高德 MCP 位置工具。\n"
            "- 若已调用高德工具并得到结果，直接基于结果生成最终建议，不要重复调用相同工具。\n"
            "- 推荐附近地点时说明距离、适合的运动类型和安全提醒；不得编造地点、距离或路线。\n"
        )
    card_context = ""
    if context and context.allow_md2card:
        card_context = (
            "\n\n中医知识卡片：\n"
            "- 当前问题允许使用 generate_tcm_knowledge_card。完成中医养生知识回答前，调用它将核心要点整理为简短 Markdown 卡片。\n"
            "- 卡片应包含标题、2-4 条生活方式要点和必要的就医/安全提示；使用 traditional-chinese 主题。\n"
            "- 卡片只用于辅助理解，不得将中医养生内容表述为诊断或处方。\n"
        )
    food_image_context = ""
    if context and context.food_image_url:
        food_image_context = (
            "\n\n饮食图片上下文：\n"
            "- 用户本轮上传了一张待识别的食物图片。必须调用 create_food_record_draft 创建待确认饮食记录草稿。\n"
            "- 不要自行猜测图片内容；该工具会把图片交给专用多模态饮食分析模型。\n"
            "- description 记录用户对图片的补充说明；未提供说明时使用“用户上传食物图片，请识别”。\n"
        )
    tool_policy = """

工具使用原则：
- 需要实时信息、用户长期档案、天气或计划生成能力时，优先使用可用工具。
- 用户询问当天已吃什么、热量/蛋白质/碳水/脂肪累计、剩余餐怎么安排、饮食复盘时，必须先调用 get_daily_diet_summary。
- get_daily_diet_summary 只包含已确认饮食记录；无记录时不得推断用户没有进食，应说明账本中尚无已确认记录。
- 用户明确陈述自己已经吃了或喝了某些食物时，必须调用 create_food_record_draft 创建待确认草稿；不要自动确认或写入正式账本。
- 创建草稿后，提醒用户在界面中确认或修改；“能不能吃/应该吃什么/计划吃什么”等问题不得创建草稿。
- 用户明确要求创建、保存或开始执行一项运动任务时，调用 create_user_requested_exercise_task；仅咨询建议时不要创建任务。
- 每个工具的适用场景以工具自身描述为准；不要在未调用工具时编造实时数据或工具结果。
"""
    return f"{CHAT_SYSTEM_PROMPT}{profile_context}{location_context}{card_context}{food_image_context}{tool_policy}"


def build_health_agent(checkpointer: Any, store: Any, extra_tools: list[Any] | None = None):
    """构建带短期记忆、长期存储、工具和结构化输出的健康 Agent。"""
    return create_agent(
        model=get_chat_model(),
        tools=build_agent_tools(extra_tools),
        middleware=[
            health_system_prompt,
            SummarizationMiddleware(
                model=get_chat_model(),
                trigger={"tokens": SUMMARY_TRIGGER_TOKENS},
                keep=("messages", SUMMARY_MESSAGES_TO_KEEP),
            ),
        ],
        response_format=ToolStrategy(HealthAssistantOutput),
        context_schema=AgentContext,
        checkpointer=checkpointer,
        store=store,
    )

"""受控中医科普子 Agent：独立模型调用、独立提示词与独立输出 Schema。"""

import json
from functools import lru_cache
from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import HumanMessage

from agents.chat_agent import get_chat_model
from prompts.tcm import TCM_WELLNESS_SYSTEM_PROMPT
from schemas.tcm import TCMWellnessOutput
from tcm.knowledge import KnowledgeChunk, retrieve_tcm_wellness_knowledge_for_agent


@lru_cache(maxsize=1)
def get_tcm_wellness_agent():
    """懒加载中医子 Agent，避免每次工具调用重复创建模型对象。"""
    return create_agent(
        model=get_chat_model(),
        tools=[],
        system_prompt=TCM_WELLNESS_SYSTEM_PROMPT,
        response_format=ToolStrategy(TCMWellnessOutput),
        name="tcm_wellness_agent",
    )


async def generate_tcm_wellness_answer(
    question: str,
    user_context: dict[str, Any],
    chunks: list[KnowledgeChunk],
) -> dict[str, Any]:
    """基于调用方提供的同一批 RAG 片段生成回答，供生产和评测共同复用。"""
    references = [
        {"source_id": chunk.source_id, "title": chunk.title, "content": chunk.content}
        for chunk in chunks
    ]
    prompt = (
        f"用户问题：{question}\n\n"
        f"用户与环境上下文：{json.dumps(user_context, ensure_ascii=False)}\n\n"
        f"可引用的受限知识片段：{json.dumps(references, ensure_ascii=False)}"
    )
    state = await get_tcm_wellness_agent().ainvoke(
        {"messages": [HumanMessage(content=prompt)]}
    )
    result = state.get("structured_response")
    if not result:
        raise RuntimeError("中医科普 Agent 未返回结构化结果")
    if not isinstance(result, TCMWellnessOutput):
        result = TCMWellnessOutput.model_validate(result)
    return result.model_dump(mode="json")


async def consult_tcm_wellness_agent(question: str, user_context: dict[str, Any]) -> dict[str, Any]:
    """生产入口：检索受限知识后调用子 Agent。"""
    chunks = await retrieve_tcm_wellness_knowledge_for_agent(question, user_context)
    return await generate_tcm_wellness_answer(question, user_context, chunks)

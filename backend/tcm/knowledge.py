"""中医子 Agent 的 PDF RAG 检索适配层。"""

from dataclasses import dataclass
from typing import Any

from rag.tcm_wellness import (
    context_limit,
    retrieve_tcm_wellness_knowledge as _retrieve_pdf_knowledge,
    retrieve_tcm_wellness_knowledge_with_trace,
)
from tcm.query_rewriter import rewrite_tcm_retrieval_query


@dataclass(frozen=True)
class KnowledgeChunk:
    """供 Agent 提示词使用的可追溯知识片段。"""

    source_id: str
    title: str
    content: str


def retrieve_tcm_wellness_knowledge(query: str, limit: int | None = None) -> list[KnowledgeChunk]:
    """从《了不起的中医养生妙招》第 8-126 页构建的 Chroma 库中检索。"""
    return [
        KnowledgeChunk(source_id=item.source_id, title=item.title, content=item.content)
        for item in _retrieve_pdf_knowledge(query, limit=limit or context_limit())
    ]


async def retrieve_tcm_wellness_knowledge_for_agent(
    question: str,
    user_context: dict[str, Any],
    limit: int | None = None,
) -> list[KnowledgeChunk]:
    """RAG Agent 专用链路：查询重写后执行双路混合召回与 rerank。"""
    rewritten_query = await rewrite_tcm_retrieval_query(question, user_context)
    result = retrieve_tcm_wellness_knowledge_with_trace(
        query=question,
        rewritten_query=rewritten_query,
        limit=limit or context_limit(),
    )
    return [
        KnowledgeChunk(source_id=item.source_id, title=item.title, content=item.content)
        for item in result.chunks
    ]

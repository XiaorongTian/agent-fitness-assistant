"""用于验证中医 PDF RAG 检索质量的只读接口。"""

from fastapi import APIRouter, HTTPException, Query

from rag.tcm_wellness import TCMRAGNotReadyError, retrieve_tcm_wellness_knowledge_with_trace
from tcm.query_rewriter import rewrite_tcm_retrieval_query

router = APIRouter()


@router.get("/rag/tcm/search")
async def search_tcm_knowledge(
    query: str = Query(min_length=2, max_length=300),
    limit: int = Query(default=3, ge=1, le=4),
    rewrite: bool = Query(default=True),
) -> dict:
    """查看“重写→混合召回→rerank”的中医 PDF 检索结果及诊断信息。"""
    try:
        rewritten_query = await rewrite_tcm_retrieval_query(query, {}) if rewrite else query
        retrieval = retrieve_tcm_wellness_knowledge_with_trace(
            query=query, rewritten_query=rewritten_query, limit=limit
        )
    except TCMRAGNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"中医 RAG 检索失败：{type(exc).__name__}") from exc
    return {
        "query": query,
        "rewritten_query": retrieval.trace.rewritten_query,
        "trace": {
            "vector_candidates": retrieval.trace.vector_candidates,
            "bm25_candidates": retrieval.trace.bm25_candidates,
            "fused_candidates": retrieval.trace.fused_candidates,
            "rerank_applied": retrieval.trace.rerank_applied,
            "elapsed_ms": retrieval.trace.elapsed_ms,
        },
        "results": [
            {
                "source_id": item.source_id,
                "title": item.title,
                "page_start": item.page_start,
                "page_end": item.page_end,
                "content": item.content,
            }
            for item in retrieval.chunks
        ],
    }

"""对话 HTTP 接口，负责接收用户问题并返回 Agent 结构化回答。"""

from common.logger import logger
from uuid import uuid4
from fastapi import APIRouter, HTTPException
from memory.runtime import conversation_runtime
from schemas.chat import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """发起或继续一次对话，并返回本轮工具调用轨迹。"""
    trace_id = uuid4().hex
    session_id = request.session_id or uuid4().hex
    try:
        await conversation_runtime.start()
        result, tool_calls = await conversation_runtime.invoke_chat(
            user_id=request.user_id,
            session_id=session_id,
            message=request.message,
        )
    except RuntimeError as exc:
        logger.warning("trace_id=%s model configuration error: %s", trace_id, exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except PermissionError as exc:
        logger.warning("trace_id=%s session access denied", trace_id)
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("trace_id=%s chat generation failed", trace_id)
        raise HTTPException(
            status_code=502,
            detail=f"对话生成失败：{type(exc).__name__}: {exc}",
        ) from exc

    return ChatResponse(
        session_id=session_id,
        result=result,
        tool_calls=tool_calls,
        trace_id=trace_id,
    )

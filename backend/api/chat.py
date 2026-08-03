"""Chat HTTP endpoint."""

from common.logger import logger
from uuid import uuid4
from fastapi import APIRouter, HTTPException
from agents.chat_agent import generate_chat_response
from schemas.chat import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    trace_id = uuid4().hex
    try:
        result = await generate_chat_response(request)
    except RuntimeError as exc:
        logger.warning("trace_id=%s model configuration error: %s", trace_id, exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("trace_id=%s chat generation failed", trace_id)
        raise HTTPException(status_code=502, detail="模型服务暂时不可用，请稍后重试") from exc

    return ChatResponse(session_id=request.session_id, result=result, trace_id=trace_id)

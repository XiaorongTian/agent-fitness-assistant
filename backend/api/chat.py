"""Chat HTTP endpoint."""

from common.logger import logger
from uuid import uuid4
from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage
from memory.runtime import GraphContext, conversation_runtime
from schemas.chat import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    trace_id = uuid4().hex
    session_id = request.session_id or uuid4().hex
    try:
        await conversation_runtime.start()
        state = await conversation_runtime.graph.ainvoke(
            {"messages": [HumanMessage(content=request.message)]},
            {"configurable": {"thread_id": session_id}},
            context=GraphContext(user_id=request.user_id),
        )
        result = state.get("last_result")
        if not result:
            raise RuntimeError("会话未返回模型结果")
    except RuntimeError as exc:
        logger.warning("trace_id=%s model configuration error: %s", trace_id, exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except PermissionError as exc:
        logger.warning("trace_id=%s session access denied", trace_id)
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("trace_id=%s chat generation failed", trace_id)
        raise HTTPException(status_code=502, detail="模型服务暂时不可用，请稍后重试") from exc

    return ChatResponse(
        session_id=session_id,
        result=result,
        trace_id=trace_id,
    )

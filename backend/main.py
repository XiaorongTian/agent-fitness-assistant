"""FastAPI 应用入口，负责初始化日志、生命周期和路由。"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import chat, diet, exercise, memory, oss, rag
from common.logger import setup_logging
from memory.runtime import conversation_runtime

setup_logging()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """应用启动时初始化 Agent 记忆资源，关闭时释放连接。"""
    await conversation_runtime.start()
    yield
    await conversation_runtime.close()


app = FastAPI(
    title="Fitness Assistant API",
    description="健康助手",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应收窄来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api", tags=["对话"])
app.include_router(memory.router, prefix="/api", tags=["记忆"])
app.include_router(diet.router, prefix="/api", tags=["饮食记录"])
app.include_router(exercise.router, prefix="/api", tags=["运动任务"])
app.include_router(oss.router, prefix="/api", tags=["申请上传签名url"])
app.include_router(rag.router, prefix="/api", tags=["中医知识库"])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)

# 启动入口

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import chat, memory, oss
from common.logger import setup_logging
from memory.runtime import conversation_runtime
import os

# 初始化日志配置
setup_logging()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await conversation_runtime.start()
    yield
    await conversation_runtime.close()


app = FastAPI(
    title="Fitness Assistant API",
    description="健康助手",
    version="0.1.0",
    lifespan=lifespan,
)

# 1. 配置跨域资源共享 (CORS)
# 插件开发中，由于请求来自浏览器扩展环境，必须正确配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境建议指定插件的 ID 或具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2.挂载路由
app.include_router(chat.router, prefix="/api", tags=["对话"])
app.include_router(memory.router, prefix="/api", tags=["记忆"])
app.include_router(oss.router, prefix="/api", tags=["申请上传签名url"])

# 3.挂载前端资源
# static_dir = os.path.join(os.path.dirname(__file__), "static")
# if os.path.exists(static_dir):
#     app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    # 启动命令：python -m app.main
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)

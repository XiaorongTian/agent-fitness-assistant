# Personal Fitness Assistant Backend

个人健康助手的 FastAPI 后端。目前包含结构化对话、服务端多轮上下文、会话摘要压缩，以及经用户确认的长期健康档案。

## 功能范围

- 基于 LangChain 的结构化健康对话输出。
- 基于 LangGraph 的多轮会话：以 `session_id` 恢复服务端上下文。
- 上下文压缩：超过 12 条消息或约 12,000 字符时，将早期内容压缩为会话摘要，保留最近 6 条消息。
- 长期档案：按 `user_id` 保存目标、忌口、运动限制和偏好；仅通过确认接口写入。
- OSS 图片上传预签名地址。

当前未包含前端、图片识别、RAG 和运动/饮食账本。

## 项目结构

```text
backend/
├── agents/       # 模型初始化与结构化输出
├── api/          # FastAPI 路由
├── memory/       # LangGraph 会话、摘要、长期档案
├── prompts/      # 系统 Prompt
├── schemas/      # 请求、响应和记忆数据模型
└── main.py       # 应用入口
```

## 前置条件

- Python 3.12+
- DashScope API Key（当前模型通过 OpenAI 兼容接口调用）
- Docker 启动方式还需要 Docker Compose v2+
- 本地 PostgreSQL 持久化模式需要 PostgreSQL 16+；Docker Compose 会自动提供。

## 环境变量

复制模板并填写模型配置：

```bash
cp .env.example .env
```

必填项：

```env
DASHSCOPE_API_KEY=你的密钥
DASHSCOPE_BASE_URL=你的 DashScope OpenAI 兼容接口地址
CHAT_MODEL=qwen3.5-plus
```

记忆存储配置：

```env
# 本地快速调试：进程重启后会话和档案均会丢失
MEMORY_BACKEND=memory

# PostgreSQL 模式需要的连接串；Docker Compose 会覆盖为容器内地址
DATABASE_URL=postgresql://user:password@localhost:5432/personal_fitness
```

`.env` 含密钥，不应提交到版本控制。

## 本地启动

创建虚拟环境并安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

使用内存记忆启动：

```bash
MEMORY_BACKEND=memory uvicorn main:app --app-dir backend --reload --port 8001
```

服务启动后访问：

- Swagger：<http://127.0.0.1:8001/docs>
- OpenAPI：<http://127.0.0.1:8001/openapi.json>

### 本地 PostgreSQL 模式

先确保 PostgreSQL 已运行，并将 `.env` 设置为：

```env
MEMORY_BACKEND=postgres
DATABASE_URL=postgresql://user:password@localhost:5432/personal_fitness
```

然后使用同一条 `uvicorn` 命令启动。首次启动会自动初始化 LangGraph 所需的持久化结构。

## Docker Compose 启动

Docker Compose 会启动 API 和 PostgreSQL，并强制 API 使用 PostgreSQL 持久化记忆。

```bash
cp .env.example .env
# 编辑 .env，至少填写 DASHSCOPE_API_KEY 和 DASHSCOPE_BASE_URL
docker compose up --build -d
```

查看状态和日志：

```bash
docker compose ps
docker compose logs -f api
```

服务地址为 <http://127.0.0.1:8001/docs>。PostgreSQL 数据存放在命名卷 `postgres_data`，执行以下命令只停止容器而保留记忆：

```bash
docker compose down
```

如需同时删除数据库和所有持久化记忆：

```bash
docker compose down -v
```

## API 快速测试

### 1. 发起或继续对话

首次请求不传 `session_id`，响应中会返回新会话 ID；后续请求携带该值即可恢复上下文。

```bash
curl -X POST http://127.0.0.1:8001/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": "demo-user",
    "message": "我今天久坐了 8 小时，下班只有 20 分钟，适合做什么运动？"
  }'
```

响应中的 `result` 固定包含 `intent`、`reply`、`actions`，并在需要时提供 `follow_up_question` 与 `safety_notice`。

### 2. 写入长期健康档案

`confirmed` 必须为 `true`，表示用户已确认这些信息可被跨会话使用。

```bash
curl -X PUT http://127.0.0.1:8001/api/memory/profile \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": "demo-user",
    "confirmed": true,
    "profile": {
      "goal": "减脂",
      "food_restrictions": ["不吃海鲜"],
      "exercise_limitations": ["膝盖旧伤，避免跑跳"],
      "preferences": ["工作日训练不超过 30 分钟"]
    }
  }'
```

读取档案：

```bash
curl 'http://127.0.0.1:8001/api/memory/profile?user_id=demo-user'
```

### 3. 删除长期记忆

此操作只删除长期健康档案，不删除会话消息和会话摘要。

```bash
curl -X DELETE 'http://127.0.0.1:8001/api/memory/profile?user_id=demo-user'
```

成功响应：

```json
{"message": "长期记忆已成功删除"}
```

## 记忆存储说明

| 数据 | 隔离键 | `memory` 模式 | `postgres` / Docker Compose 模式 |
|---|---|---|---|
| 会话消息与摘要 | `session_id` | 进程内存，重启丢失 | PostgreSQL，重启保留 |
| 健康档案 | `user_id` | 进程内存，重启丢失 | PostgreSQL，重启保留 |

同一会话只能由创建它的 `user_id` 继续使用。当前项目尚未接入身份认证，生产部署时应由认证后的用户身份生成 `user_id`，而不是直接信任客户端传值。

# Personal Fitness Assistant

当前第一阶段提供文本对话、Prompt 约束与 Pydantic 结构化输出。图片识别、记忆和工具调用会在后续闭环中接入。

## 启动

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# 在 .env 中填写 OPENAI_API_KEY；可选填写 OPENAI_BASE_URL 和 CHAT_MODEL
uvicorn main:app --app-dir backend --reload --port 8001
```

`OPENAI_BASE_URL` 可用于兼容 OpenAI API 的模型网关；未设置时使用默认 OpenAI 地址。

## 调用示例

```bash
curl -X POST http://127.0.0.1:8001/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": "demo-user",
    "session_id": "session-001",
    "message": "我今天久坐了 8 小时，下班只有 20 分钟，适合做什么运动？",
    "history": []
  }'
```

返回的 `result` 由模型按 `HealthAssistantOutput` 校验，固定包含：

- `intent`：饮食、运动、久坐、养生、一般问题或高风险。
- `reply`：面向用户的回答。
- `actions`：最多 3 条可执行行动。
- `follow_up_question` 与 `safety_notice`：仅在需要时出现。

Prompt 位于 `backend/prompts/chat.py`，输入/输出模型位于 `backend/schemas/chat.py`。

"""MD2Card HTTP 工具：把中医养生要点生成图片卡片。"""

import os
from typing import Any

import httpx
from langchain.tools import tool

from common.logger import logger
from common.tool_logging import log_tool_end, log_tool_start

MD2CARD_API_URL = "https://md2card.cn/api/generate"


async def load_md2card_api_tools() -> list[Any]:
    """按需注册 MD2Card HTTP 工具；未配置密钥时不影响其他能力。"""
    api_key = os.getenv("MD2CARD_API_KEY", "").strip()
    if not api_key:
        logger.info("md2card_http_stage=disabled reason=missing_api_key")
        return []

    @tool
    async def generate_tcm_knowledge_card(
        markdown: str,
        theme: str = "traditional-chinese",
    ) -> dict[str, Any]:
        """将中医养生知识要点生成图片卡片，仅在用户明确咨询中医养生时调用。

        markdown 必须是简短 Markdown，包含标题、2 至 4 条生活方式建议和安全提示。
        成功后返回 image_urls，前端可直接展示其中的图片。
        """
        content = markdown.strip()
        if not content:
            return {"error": "卡片内容不能为空。"}
        log_tool_start(
            "generate_tcm_knowledge_card",
            {"markdown_length": len(content), "theme": theme},
        )
        try:
            timeout = httpx.Timeout(60.0, connect=10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    os.getenv("MD2CARD_API_URL", MD2CARD_API_URL).strip() or MD2CARD_API_URL,
                    headers={"x-api-key": api_key},
                    json={
                        "markdown": content,
                        "theme": theme,
                        "width": 440,
                        "height": 586,
                    },
                )
            payload = response.json()
            if response.is_error:
                result = {
                    "error": "知识卡片服务暂时不可用。",
                    "status_code": response.status_code,
                }
                logger.warning(
                    "md2card_http_stage=generate_failed status_code=%s provider_error=%s",
                    response.status_code,
                    payload.get("message") if isinstance(payload, dict) else None,
                )
            else:
                images = payload.get("images", []) if isinstance(payload, dict) else []
                image_urls = [item.get("url") for item in images if isinstance(item, dict) and item.get("url")]
                result = (
                    {"image_urls": image_urls, "images": images}
                    if image_urls
                    else {"error": "知识卡片服务未返回图片。"}
                )
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("md2card_http_stage=generate_failed error=%s", type(exc).__name__)
            result = {"error": "知识卡片服务调用失败，请稍后重试。"}
        log_tool_end("generate_tcm_knowledge_card", result)
        return result

    logger.info("md2card_http_stage=tool_loaded tool_name=generate_tcm_knowledge_card")
    return [generate_tcm_knowledge_card]

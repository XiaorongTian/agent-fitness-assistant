"""高德地图 MCP Client：把远程 MCP 工具转换为 LangChain 工具。"""

import os
import json
from time import perf_counter
from typing import Any
from urllib.parse import urlencode

from common.logger import logger
from common.mcp_observability import mcp_observation_context
from langchain_core.tools import StructuredTool


def _amap_mcp_url() -> str | None:
    """构造高德 Streamable HTTP MCP 地址，避免在日志中暴露 Key。"""
    api_key = os.getenv("AMAP_MAPS_API_KEY", "").strip()
    if not api_key:
        return None
    base_url = os.getenv("AMAP_MCP_URL", "https://mcp.amap.com/mcp").strip()
    separator = "&" if "?" in base_url else "?"
    return f"{base_url}{separator}{urlencode({'key': api_key})}"


def _mcp_business_error(output: Any) -> str | None:
    """识别 MCP 以内容块返回的业务错误，避免将其误计为调用成功。"""
    blocks = output if isinstance(output, list) else [output]
    for block in blocks:
        text = block.get("text") if isinstance(block, dict) else None
        if not isinstance(text, str):
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("error"):
            return str(payload.get("message") or payload.get("error"))
    return None


async def load_amap_mcp_tools() -> list[Any]:
    """加载高德 MCP 工具；第三方服务不可用时降级为不注册位置工具。"""
    mcp_url = _amap_mcp_url()
    if not mcp_url:
        logger.info("amap_mcp_stage=disabled reason=missing_api_key")
        return []

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient(
            {
                "amap": {
                    "transport": "http",
                    "url": mcp_url,
                }
            }
        )
        raw_tools = await client.get_tools()
        tools = [wrap_mcp_tool(tool, "amap") for tool in raw_tools]
        logger.info(
            "amap_mcp_stage=tools_loaded tool_count=%s tool_names=%s",
            len(tools),
            [tool.name for tool in tools],
        )
        return tools
    except Exception:
        logger.exception("amap_mcp_stage=tools_load_failed")
        return []


def wrap_mcp_tool(source_tool: Any, server_name: str) -> StructuredTool:
    """保留 MCP 工具 Schema，并记录每次实际调用的参数、耗时和状态。"""

    async def invoke_with_observability(**arguments: Any) -> Any:
        context = mcp_observation_context.get()
        started_at = perf_counter()
        if context and server_name not in context.allowed_mcp_servers:
            logger.warning(
                "mcp_tool_call rejected server=%s name=%s trace_id=%s reason=policy_denied",
                server_name,
                source_tool.name,
                context.trace_id,
            )
            context.observations.append(
                {
                    "name": source_tool.name,
                    "mcp_server": server_name,
                    "duration_ms": 0,
                    "status": "failed",
                    "error": "当前问题不允许调用该 MCP 服务",
                }
            )
            return {"error": "当前问题不允许调用该工具。"}
        logger.info(
            "mcp_tool_call start server=%s name=%s trace_id=%s user_id=%s session_id=%s arguments=%s",
            server_name,
            source_tool.name,
            context.trace_id if context else None,
            context.user_id if context else None,
            context.session_id if context else None,
            arguments,
        )
        try:
            output = await source_tool.ainvoke(arguments)
        except Exception as exc:
            duration_ms = round((perf_counter() - started_at) * 1000)
            observation = {
                "name": source_tool.name,
                "mcp_server": server_name,
                "duration_ms": duration_ms,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            if context:
                context.observations.append(observation)
            logger.exception(
                "mcp_tool_call failed server=%s name=%s trace_id=%s duration_ms=%s",
                server_name,
                source_tool.name,
                context.trace_id if context else None,
                duration_ms,
            )
            return {"error": f"{server_name} MCP 服务调用失败，请稍后重试。"}

        duration_ms = round((perf_counter() - started_at) * 1000)
        business_error = _mcp_business_error(output)
        if business_error:
            if context:
                context.observations.append(
                    {
                        "name": source_tool.name,
                        "mcp_server": server_name,
                        "duration_ms": duration_ms,
                        "status": "failed",
                        "error": business_error,
                    }
                )
            logger.warning(
                "mcp_tool_call business_failed server=%s name=%s trace_id=%s duration_ms=%s error=%s",
                server_name,
                source_tool.name,
                context.trace_id if context else None,
                duration_ms,
                business_error,
            )
            return output
        if context:
            context.observations.append(
                {
                    "name": source_tool.name,
                    "mcp_server": server_name,
                    "duration_ms": duration_ms,
                    "status": "succeeded",
                    "error": None,
                }
            )
        logger.info(
            "mcp_tool_call end server=%s name=%s trace_id=%s duration_ms=%s status=succeeded",
            server_name,
            source_tool.name,
            context.trace_id if context else None,
            duration_ms,
        )
        return output

    return StructuredTool.from_function(
        coroutine=invoke_with_observability,
        name=source_tool.name,
        description=source_tool.description,
        args_schema=source_tool.args_schema,
        metadata={"mcp_server": server_name},
    )

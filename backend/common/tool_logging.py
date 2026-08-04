"""工具调用日志模块，统一记录工具开始和结束信息。"""

import json
from typing import Any

from common.logger import logger


def log_tool_start(name: str, arguments: dict[str, Any]) -> None:
    """记录工具名称和输入参数。"""
    logger.info("tool_call start name=%s arguments=%s", name, json.dumps(arguments, ensure_ascii=False))


def log_tool_end(name: str, output: Any) -> None:
    """记录工具名称和返回结果。"""
    logger.info("tool_call end name=%s output=%s", name, json.dumps(output, ensure_ascii=False))

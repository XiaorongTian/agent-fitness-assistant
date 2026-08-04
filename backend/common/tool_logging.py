"""Shared logging helpers for agent tool execution."""

import json
from typing import Any

from common.logger import logger


def log_tool_start(name: str, arguments: dict[str, Any]) -> None:
    logger.info("tool_call start name=%s arguments=%s", name, json.dumps(arguments, ensure_ascii=False))


def log_tool_end(name: str, output: Any) -> None:
    logger.info("tool_call end name=%s output=%s", name, json.dumps(output, ensure_ascii=False))

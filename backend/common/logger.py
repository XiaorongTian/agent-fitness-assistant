"""日志配置模块，提供应用统一使用的 logger。"""

import logging
import sys

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"


def setup_logging():
    """初始化控制台日志格式和级别。"""
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


logger = logging.getLogger("fitness_assistant")

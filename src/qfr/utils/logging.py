"""Project-wide logging built on loguru, configured from ``settings.log_level``.

Import the shared ``logger`` anywhere::

    from qfr.utils.logging import logger
    logger.info("hello")
"""

from __future__ import annotations

import sys

from loguru import logger

from qfr.utils.config import settings

_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
)


def configure_logging(level: str | None = None):
    """(Re)configure the single loguru sink. Safe to call repeatedly."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=(level or settings.log_level).upper(),
        format=_FORMAT,
        colorize=True,
        backtrace=False,
        diagnose=False,
    )
    return logger


configure_logging()

__all__ = ["logger", "configure_logging"]

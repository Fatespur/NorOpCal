"""Structured logging setup without import-time global side effects."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    """Format log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize a log record without private process or host fields.

        Args:
            record: Standard-library log record.

        Returns:
            A stable JSON string suitable for JSONL logs.
        """

        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def configure_logging(
    level: str = "INFO",
    *,
    log_file: Path | None = None,
    json_output: bool = False,
) -> logging.Logger:
    """Configure and return the root ShiftAD logger.

    Args:
        level: Standard uppercase logging level.
        log_file: Optional UTF-8 log destination.
        json_output: Emit JSONL instead of readable text.

    Returns:
        The configured ``shiftad`` logger.

    Raises:
        ValueError: If ``level`` is not a recognized logging level.
        OSError: If the log destination cannot be created.

    Assumptions:
        Calling this function replaces handlers on the ShiftAD logger only.
    """

    numeric_level = logging.getLevelNamesMapping().get(level)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unknown log level {level!r}; use a standard uppercase level.")

    logger = logging.getLogger("shiftad")
    logger.setLevel(numeric_level)
    logger.propagate = False
    logger.handlers.clear()

    formatter: logging.Formatter
    if json_output:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

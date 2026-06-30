from __future__ import annotations

import logging
from typing import Any

from ped_agent.utils.config import select


def configure_logging(config: Any | None = None) -> None:
    level_name = select(config, "app.log_level", "INFO") if config is not None else "INFO"
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


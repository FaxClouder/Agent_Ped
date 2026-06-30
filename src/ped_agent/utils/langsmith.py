from __future__ import annotations

import os
from typing import Any

from ped_agent.utils.config import select


def configure_langsmith(config: Any) -> None:
    if not select(config, "langsmith.enabled", False):
        os.environ.setdefault("LANGSMITH_TRACING", "false")
        return

    os.environ.setdefault("LANGSMITH_TRACING", "true")
    if api_key := select(config, "langsmith.api_key"):
        os.environ.setdefault("LANGSMITH_API_KEY", api_key)
    if project := select(config, "langsmith.project_name"):
        os.environ.setdefault("LANGSMITH_PROJECT", project)


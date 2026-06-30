from __future__ import annotations

import importlib.metadata
from typing import Any

from ped_agent.vision.interface import VisionBackend


class VisionRegistry:
    _backends: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(backend_class: type):
            cls._backends[name] = backend_class
            return backend_class

        return decorator

    @classmethod
    def discover(cls) -> None:
        try:
            entry_points = importlib.metadata.entry_points(group="ped_agent.vision")
        except Exception:
            return
        for entry_point in entry_points:
            if entry_point.name not in cls._backends:
                cls._backends[entry_point.name] = entry_point.load()

    @classmethod
    def get(cls, name: str, config: dict | None = None) -> VisionBackend:
        ensure_builtin_backends_registered()
        cls.discover()
        if name not in cls._backends:
            available = ", ".join(sorted(cls._backends)) or "none"
            raise ValueError(f"Unknown vision backend '{name}'. Available: {available}")
        return cls._backends[name](config or {})

    @classmethod
    def list_available(cls) -> list[str]:
        ensure_builtin_backends_registered()
        cls.discover()
        return sorted(cls._backends)


def ensure_builtin_backends_registered() -> None:
    from ped_agent.vision.plugins import yolo26_bytetrack, yolo26_deepsort  # noqa: F401

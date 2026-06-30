from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

try:  # pragma: no cover - exercised when optional dependency is installed.
    from omegaconf import DictConfig, OmegaConf
except Exception:  # pragma: no cover - fallback is covered by tests.
    DictConfig = Any  # type: ignore[misc, assignment]
    OmegaConf = None  # type: ignore[assignment]


CONFIG_FILES = (
    "default.yaml",
    "llm.yaml",
    "rag.yaml",
    "analysis.yaml",
    "vision.yaml",
    "langsmith.yaml",
)


class ConfigNode(dict):
    """Dict with attribute access used when OmegaConf is unavailable."""

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def copy(self) -> "ConfigNode":  # type: ignore[override]
        return ConfigNode({k: _wrap(v) for k, v in self.items()})


class ConfigValidationResult(BaseModel):
    valid: bool = True
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ConfigManager:
    """Load layered YAML configuration with environment and CLI overrides."""

    _instance: ConfigManager | None = None
    _config: DictConfig | ConfigNode | None = None

    def __new__(cls) -> ConfigManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(
        self,
        config_dir: str | Path = "config",
        env_file: str | Path = ".env",
        overrides: list[str] | None = None,
    ) -> DictConfig | ConfigNode:
        self._load_dotenv(env_file)
        config_path = Path(config_dir)

        if OmegaConf is not None:
            configs = [
                OmegaConf.load(path)
                for filename in CONFIG_FILES
                if (path := config_path / filename).exists()
            ]
            merged = OmegaConf.merge(*configs) if configs else OmegaConf.create({})
            if overrides:
                normalized = [_normalize_override(item) for item in overrides]
                merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(normalized))
            OmegaConf.resolve(merged)
            self._config = merged
            return merged

        merged_dict: dict[str, Any] = {}
        for filename in CONFIG_FILES:
            path = config_path / filename
            if path.exists():
                with path.open("r", encoding="utf-8") as handle:
                    payload = yaml.safe_load(handle) or {}
                merged_dict = _deep_merge(merged_dict, payload)

        for key, value in _parse_overrides(overrides or []).items():
            _set_dotted(merged_dict, key, value)

        resolved = _resolve_env(merged_dict)
        self._config = _wrap(resolved)
        return self._config

    @property
    def config(self) -> DictConfig | ConfigNode:
        if self._config is None:
            return self.load()
        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        return select(self.config, key, default)

    @staticmethod
    def _load_dotenv(env_file: str | Path) -> None:
        env_path = Path(env_file)
        if not env_path.exists():
            return
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_config() -> DictConfig | ConfigNode:
    return ConfigManager().config


def select(config: Any, key: str, default: Any = None) -> Any:
    if OmegaConf is not None and not isinstance(config, dict):
        value = OmegaConf.select(config, key, default=default)
        return default if value is None else value

    current = config
    for part in key.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current


def validate_config(config: Any, require_secrets: bool = False) -> ConfigValidationResult:
    result = ConfigValidationResult()

    required_paths = [
        "app.name",
        "agent.mode",
        "llm.default_provider",
        "rag.embedding.model",
        "analysis.default_metrics",
        "vision.enabled",
    ]
    for path in required_paths:
        if select(config, path) is None:
            result.errors.append(f"Missing required configuration: {path}")

    provider = select(config, "llm.default_provider")
    providers = select(config, "llm.providers", {})
    if provider and provider not in providers:
        result.errors.append(f"Default LLM provider '{provider}' is not defined.")

    for provider_name in {provider, select(config, "llm.fallback_provider")} - {None, "local"}:
        api_key = select(config, f"llm.providers.{provider_name}.api_key")
        if not api_key:
            message = f"LLM provider '{provider_name}' has no API key configured."
            if require_secrets:
                result.errors.append(message)
            else:
                result.warnings.append(message)

    if select(config, "langsmith.enabled", False) and not select(config, "langsmith.api_key"):
        message = "LangSmith is enabled but LANGSMITH_API_KEY is not configured."
        if require_secrets:
            result.errors.append(message)
        else:
            result.warnings.append(message)

    if select(config, "vision.enabled", False):
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            result.errors.append("Vision is enabled but ultralytics is not installed.")

    result.valid = not result.errors
    return result


def _normalize_override(item: str) -> str:
    return item[2:] if item.startswith("--") else item


def _parse_overrides(overrides: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in overrides:
        normalized = _normalize_override(item)
        if "=" not in normalized:
            continue
        key, value = normalized.split("=", 1)
        parsed[key] = yaml.safe_load(value)
    return parsed


def _set_dotted(target: dict[str, Any], key: str, value: Any) -> None:
    current = target
    parts = key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _deep_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


ENV_PATTERN = re.compile(r"\$\{oc\.env:([^,}]+),?([^}]*)\}")


def _resolve_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        env_name = match.group(1)
        default = match.group(2)
        if default == "null":
            default = ""
        return os.getenv(env_name, default)

    resolved = ENV_PATTERN.sub(replace, value)
    return None if resolved == "" and value.startswith("${oc.env:") else resolved


def _wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return ConfigNode({k: _wrap(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_wrap(item) for item in value]
    return value

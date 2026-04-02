from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


load_dotenv()


@dataclass(slots=True)
class Settings:
    llm_api_key: str | None = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    llm_base_url: str | None = (
        os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.deepseek.com/v1"
    )
    chat_model: str = (
        os.getenv("CHAT_MODEL") or os.getenv("DEEPSEEK_MODEL") or os.getenv("OPENAI_CHAT_MODEL") or "deepseek-chat"
    )
    embedding_api_key: str | None = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
    embedding_base_url: str | None = os.getenv("EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    embedding_model: str = os.getenv("EMBEDDING_MODEL") or os.getenv("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-large"
    vector_store_dir: Path = Path(os.getenv("VECTOR_STORE_DIR", ".vector_store"))
    knowledge_dir: Path = Path(os.getenv("KNOWLEDGE_DIR", "data/knowledge_base"))


def _maybe(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def get_chat_model_kwargs(settings: Settings) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"model": settings.chat_model, "temperature": 0.1}
    api_key = _maybe(settings.llm_api_key)
    base_url = _maybe(settings.llm_base_url)
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return kwargs


def get_embedding_kwargs(settings: Settings) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"model": settings.embedding_model}
    api_key = _maybe(settings.embedding_api_key)
    base_url = _maybe(settings.embedding_base_url)
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return kwargs


def get_settings() -> Settings:
    return Settings()

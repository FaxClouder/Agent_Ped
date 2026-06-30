from __future__ import annotations

from typing import Any

from ped_agent.utils.config import select


class MissingLLMDependencyError(RuntimeError):
    pass


class LLMFactory:
    """Build LangChain chat models from provider config."""

    def __init__(self, config: Any):
        self.config = config

    def build(self, provider: str | None = None):
        provider_name = provider or select(self.config, "llm.default_provider", "local")
        provider_config = select(self.config, f"llm.providers.{provider_name}")
        if provider_config is None:
            raise ValueError(f"Unknown LLM provider: {provider_name}")

        if provider_name == "claude":
            try:
                from langchain_anthropic import ChatAnthropic
            except ImportError as exc:
                raise MissingLLMDependencyError("Install langchain-anthropic.") from exc
            return ChatAnthropic(
                model=provider_config["model"],
                api_key=provider_config.get("api_key"),
                max_tokens=provider_config.get("max_tokens", 4096),
                temperature=provider_config.get("temperature", 0.1),
                timeout=provider_config.get("timeout", 60),
                max_retries=provider_config.get("max_retries", 3),
            )

        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise MissingLLMDependencyError("Install langchain-openai.") from exc

        kwargs = {
            "model": provider_config["model"],
            "api_key": provider_config.get("api_key"),
            "temperature": provider_config.get("temperature", 0.1),
            "timeout": provider_config.get("timeout", 60),
            "max_retries": provider_config.get("max_retries", 3),
        }
        if provider_config.get("base_url"):
            kwargs["base_url"] = provider_config["base_url"]
        if provider_config.get("max_tokens"):
            kwargs["max_tokens"] = provider_config["max_tokens"]
        return ChatOpenAI(**kwargs)


def build_chat_model(config: Any, provider: str | None = None):
    return LLMFactory(config).build(provider)


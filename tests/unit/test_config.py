from __future__ import annotations

from pathlib import Path

from ped_agent.utils.config import ConfigManager, select, validate_config


def test_config_loads_defaults_and_overrides():
    config = ConfigManager().load(
        config_dir=Path("config"),
        env_file=Path(".missing-env"),
        overrides=["--llm.default_provider=local", "app.log_level=DEBUG"],
    )

    assert select(config, "app.name") == "Ped-Agent"
    assert select(config, "llm.default_provider") == "local"
    assert select(config, "app.log_level") == "DEBUG"


def test_config_validation_allows_missing_keys_as_warnings():
    config = ConfigManager().load(config_dir=Path("config"), env_file=Path(".missing-env"))
    result = validate_config(config, require_secrets=False)

    assert result.valid
    assert any("LLM provider" in warning for warning in result.warnings)


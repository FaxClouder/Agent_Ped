from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from ped_agent.agent.graph import build_agent_graph
from ped_agent.utils.config import ConfigManager, validate_config
from ped_agent.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Ped-Agent scaffold.")
    parser.add_argument("query", nargs="?", help="Question or task to route through the agent.")
    parser.add_argument(
        "--config-dir",
        default="config",
        help="Directory containing Ped-Agent YAML configuration files.",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Environment file to load before configuration resolution.",
    )
    parser.add_argument(
        "--strict-secrets",
        action="store_true",
        help="Fail validation when configured providers are missing API keys.",
    )
    parser.add_argument("overrides", nargs="*", help="OmegaConf-style config overrides.")
    args, unknown = parser.parse_known_args(argv)
    args.overrides = [*args.overrides, *unknown]
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = ConfigManager().load(
        config_dir=args.config_dir,
        env_file=args.env_file,
        overrides=args.overrides,
    )
    configure_logging(config)

    validation = validate_config(config, require_secrets=args.strict_secrets)
    for warning in validation.warnings:
        logger.warning(warning)
    if not validation.valid:
        for error in validation.errors:
            logger.error(error)
        return 1

    if not args.query:
        logger.info("Ped-Agent scaffold is ready. Pass a query to exercise the routing graph.")
        return 0

    graph = build_agent_graph(config)
    result = graph.invoke({"query": args.query})
    print(result.get("answer") or result.get("result") or result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

from __future__ import annotations

import argparse

from ped_agent.agent.graph import build_agent_graph
from ped_agent.utils.config import ConfigManager


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test the Ped-Agent routing graph.")
    parser.add_argument("query")
    args = parser.parse_args()

    config = ConfigManager().load()
    graph = build_agent_graph(config)
    result = graph.invoke({"query": args.query})
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


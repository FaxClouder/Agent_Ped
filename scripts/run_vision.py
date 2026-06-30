from __future__ import annotations

import argparse

from ped_agent.utils.config import ConfigManager, select
from ped_agent.vision.registry import VisionRegistry


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a configured vision backend.")
    parser.add_argument("video_path")
    parser.add_argument("--backend", default=None)
    args = parser.parse_args()

    config = ConfigManager().load()
    backend_name = args.backend or select(config, "vision.backend", "yolo26_bytetrack")
    backend = VisionRegistry.get(backend_name, select(config, "vision", {}))
    result = backend.process_video(args.video_path)
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


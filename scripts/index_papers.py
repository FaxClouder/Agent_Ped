from __future__ import annotations

import argparse
from pathlib import Path

from ped_agent.knowledge.indexer import Indexer


def main() -> int:
    parser = argparse.ArgumentParser(description="Index local paper text files.")
    parser.add_argument("paths", nargs="*", help="Text files to index in the scaffold store.")
    args = parser.parse_args()

    indexer = Indexer()
    chunks = []
    for item in args.paths:
        chunks.extend(indexer.load_text_file(Path(item)))
    ids = indexer.index_chunks(chunks)
    print(f"Indexed {len(ids)} chunks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


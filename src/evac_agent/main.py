from __future__ import annotations

import argparse

from evac_agent.graph import build_graph
from evac_agent.rag import build_or_load_vectorstore


def main() -> None:
    parser = argparse.ArgumentParser(description="Evacuation QA agent with LangGraph and RAG.")
    parser.add_argument("question", nargs="*", help="User question for evacuation QA.")
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Force rebuild of the local vector store before answering.",
    )
    parser.add_argument(
        "--prepare-index",
        action="store_true",
        help="Only build or refresh the local vector store, then exit.",
    )
    args = parser.parse_args()

    if args.prepare_index:
        build_or_load_vectorstore(force_rebuild=args.rebuild_index)
        print("Knowledge index is ready.")
        return

    question = " ".join(args.question).strip()
    if not question:
        raise SystemExit('Usage: python -m evac_agent.main "你的疏散问题" [--rebuild-index]')

    app = build_graph()
    result = app.invoke({"question": question})
    print(result["final_answer"])


if __name__ == "__main__":
    main()

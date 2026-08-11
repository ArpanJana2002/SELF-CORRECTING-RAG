"""
CLI entry point for the self-correcting RAG agent.

Usage:
    # 1. Ingest documents into the vector store (run once, or whenever docs change)
    python main.py ingest --path ./data

    # 2. Ask a single question
    python main.py ask "What is corrective RAG?"

    # 3. Interactive chat loop
    python main.py chat
"""
import argparse
import sys

from src.graph import run
from src.ingestion import build_vectorstore


def cmd_ingest(args):
    build_vectorstore(source_path=args.path, urls=args.url, reset=args.reset)


def cmd_ask(args):
    final_state = run(args.question, verbose=not args.quiet)
    print("\n" + "=" * 70)
    print("ANSWER:")
    print(final_state["generation"])
    print("=" * 70)
    if final_state.get("used_web_search"):
        print("(note: this answer was supplemented with a live web search)")


def cmd_chat(args):
    print("Self-correcting RAG agent. Type 'exit' to quit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break
        final_state = run(question, verbose=not args.quiet)
        print(f"\nAgent: {final_state['generation']}\n")
        if final_state.get("used_web_search"):
            print("(supplemented with live web search)\n")


def main():
    parser = argparse.ArgumentParser(description="Self-correcting RAG agent (LangGraph)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Build the vector store from documents")
    p_ingest.add_argument("--path", type=str, default="./data")
    p_ingest.add_argument("--url", action="append", default=[])
    p_ingest.add_argument("--reset", action="store_true")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="Ask a single question")
    p_ask.add_argument("question", type=str)
    p_ask.add_argument("--quiet", action="store_true", help="Suppress step-by-step trace")
    p_ask.set_defaults(func=cmd_ask)

    p_chat = sub.add_parser("chat", help="Interactive chat loop")
    p_chat.add_argument("--quiet", action="store_true", help="Suppress step-by-step trace")
    p_chat.set_defaults(func=cmd_chat)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)

"""
Standard question validation script.

Runs 10 common user questions through the live RAG pipeline and prints:
  - Which chunks were retrieved (source URL + first 200 chars)
  - The LLM's full answer

Usage:
    .venv\Scripts\python.exe scripts/test_retrieval.py
    .venv\Scripts\python.exe scripts/test_retrieval.py --questions 1 3 5   (run subset)

Requires: AZURE_OPENAI_*, AZURE_SEARCH_* env vars (or .env file).
DO NOT deploy until human validation of these answers is complete.
"""

import asyncio
import argparse
import os
import sys
import textwrap

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

QUESTIONS = [
    "What are the fees for the 2025/2026 year?",
    "When is the next open morning?",
    "What are the term dates?",
    "Who is the headmistress?",
    "What clubs and activities are available?",
    "What is the school uniform?",
    "How do I apply to the school?",
    "What are the Nursery fees?",
    "What sports does the school offer?",
    "Is there an after-school care programme?",
]

WIDTH = 100
SEP = "─" * WIDTH


async def run_question(n: int, question: str) -> None:
    from src.chatbot.rag_pipeline import retrieve, chat

    print(f"\n{SEP}")
    print(f"Q{n}: {question}")
    print(SEP)

    # ── Retrieval ───────────────────────────────────────────────────────────
    chunks = await retrieve(question)
    print(f"\n  Retrieved {len(chunks)} chunks:")
    for i, c in enumerate(chunks, 1):
        snippet = c["content"][:180].replace("\n", " ")
        title = c.get("title") or ""
        print(f"  [{i}] {c['source']}")
        if title:
            print(f"       Title: {title}")
        print(f"       {snippet}…")

    # ── LLM answer ──────────────────────────────────────────────────────────
    print(f"\n  Answer:")
    answer_parts = []
    async for token in chat(question, history=[]):
        if token.startswith("__sources__:"):
            sources_json = token[len("__sources__:"):]
            print()
            print(f"  Sources: {sources_json}")
        else:
            answer_parts.append(token)
            print(token, end="", flush=True)
    print()  # newline after streamed answer


async def main(selected: list[int]) -> None:
    print(f"\n{'═' * WIDTH}")
    print("  ASK WARWICK — Standard Question Validation")
    print(f"  Running {len(selected)} question(s)")
    print(f"{'═' * WIDTH}")

    for n in selected:
        await run_question(n, QUESTIONS[n - 1])

    print(f"\n{SEP}")
    print("  All questions complete. Please review each answer above before approving deployment.")
    print(SEP)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions", "-q",
        nargs="*",
        type=int,
        default=list(range(1, len(QUESTIONS) + 1)),
        help="Which question numbers to run (1-based). Default: all.",
    )
    args = parser.parse_args()

    invalid = [q for q in args.questions if q < 1 or q > len(QUESTIONS)]
    if invalid:
        print(f"Invalid question numbers: {invalid}. Valid range: 1–{len(QUESTIONS)}.")
        sys.exit(1)

    asyncio.run(main(sorted(set(args.questions))))

"""
Live Prompt Evaluation Script
==============================
Fetches the unique prompts asked in the last 24 hours from Azure Log Analytics,
runs each one through the live RAG pipeline, and evaluates whether the response
is correct/appropriate.

Evaluation criteria
-------------------
OUT_OF_SCOPE  — questions not about Warwick Prep School (AI/LLM questions, opinions,
                personal queries). The chatbot MUST refuse these.
IN_SCOPE      — genuine school questions. The chatbot should return a grounded answer
                with supporting sources. FAIL if it returns an error or says it has no
                information when we'd expect it to.

Usage:
    .venv\\Scripts\\python.exe scripts/eval_live_prompts.py
    .venv\\Scripts\\python.exe scripts/eval_live_prompts.py --hours 48
"""

import asyncio
import argparse
import json
import os
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

WORKSPACE_ID = "ad672a5e-50bc-4694-9ad6-a39d549b24da"
ANALYTICS_URL = f"https://api.loganalytics.io/v1/workspaces/{WORKSPACE_ID}/query"

WIDTH = 100
SEP = "─" * WIDTH
DSEQ = "═" * WIDTH

# ── Classification ────────────────────────────────────────────────────────────
# Questions whose correct response is a REFUSAL (not about the school).
OUT_OF_SCOPE_PATTERNS = [
    "which model",
    "what model",
    "llm",
    "openai",
    "gpt",
    "when were you created",
    "when was i created",
    "when we're you created",
    "does warwick prep us ai",
    "does warwick preo uses ai",
    "is there a better school",
    "is there better school",
    "better school than",
]

# Phrases that indicate a proper refusal / scope-limit response.
REFUSAL_PHRASES = [
    "i can only help with questions about warwick prep school",
    "i'm only able to help with",
    "i only answer questions about",
    "outside my scope",
    "can only assist with",
    "general search engine",
    "not able to provide",
    "i cannot answer",
    "i can't answer",
    "i don't have information about that",
    "i don't have information about",
    "i can only help",
]

# Phrases indicating the bot has no information (failure for in-scope questions
# where we expect real content to be indexed).
NO_INFO_PHRASES = [
    "i don't have that information",
    "i don't have information",
    "no information available",
    "unable to find",
    "not found in",
    "sorry, an error occurred",
]

# For these in-scope questions we *expect* "contact school directly" is fine
# (e.g. staff list, specific policies not on the website).
EXPECTED_NO_INFO = [
    "who are all the year 1 teachers",
    "what's the school policy on social media",
    "what is the policy on authorised leave",
    # "NWS" is not a Warwick Prep entity — bot correctly says no info
    "nws students",
    # Warwick Prep is a prep school (ages 3–11, up to Year 6); Year 9 does not exist
    "year 9",
]


def _is_out_of_scope(question: str) -> bool:
    q = question.lower()
    return any(pat in q for pat in OUT_OF_SCOPE_PATTERNS)


def _check_refusal(answer: str) -> bool:
    a = answer.lower()
    return any(phrase in a for phrase in REFUSAL_PHRASES)


def _check_no_info(answer: str) -> bool:
    a = answer.lower()
    return any(phrase in a for phrase in NO_INFO_PHRASES)


def _expected_no_info(question: str) -> bool:
    q = question.lower()
    return any(pat in q for pat in EXPECTED_NO_INFO)


# ── Log Analytics helpers ─────────────────────────────────────────────────────


def _get_token() -> str:
    cmd = "az account get-access-token --resource https://api.loganalytics.io --query accessToken -o tsv"
    result = subprocess.run(cmd, capture_output=True, text=True, check=True, shell=True)
    return result.stdout.strip()


def _run_kql(query: str, token: str) -> list[dict]:
    body = json.dumps({"query": query}).encode()
    req = urllib.request.Request(
        ANALYTICS_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    table = data["tables"][0]
    cols = [c["name"] for c in table["columns"]]
    return [dict(zip(cols, row)) for row in table["rows"]]


def fetch_prompts(hours: int) -> list[str]:
    """Return deduplicated list of prompts from the last `hours` hours."""
    print(f"  Fetching token...", end="", flush=True)
    token = _get_token()
    print(" OK")
    print(
        f"  Querying Log Analytics for last {hours}h of prompts...", end="", flush=True
    )
    rows = _run_kql(
        f"""
AppServiceConsoleLogs
| where TimeGenerated > ago({hours}h)
| where ResultDescription has "PROMPT"
| extend question = coalesce(
    extract(@"msg='([^']+)'", 1, ResultDescription),
    extract(@'msg="([^"]+)"', 1, ResultDescription)
  )
| where isnotempty(question)
| summarize count() by question
| order by count_ desc
""",
        token,
    )
    print(f" OK — {len(rows)} unique prompts found")
    return [(r["question"], r["count_"]) for r in rows]


# ── Live pipeline runner ──────────────────────────────────────────────────────


async def run_prompt(question: str) -> tuple[str, list[dict]]:
    """Run `question` through the live RAG pipeline. Returns (full_answer, chunks)."""
    from src.chatbot.rag_pipeline import retrieve, chat

    chunks = await retrieve(question)
    answer_parts: list[str] = []
    async for token in chat(question, history=[]):
        if not token.startswith("__sources__:"):
            answer_parts.append(token)
    return "".join(answer_parts), chunks


# ── Evaluation ────────────────────────────────────────────────────────────────

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"  # answered but uncertain quality


def evaluate(question: str, answer: str, chunks: list[dict]) -> tuple[str, str]:
    """Return (verdict, reason)."""
    if _is_out_of_scope(question):
        if _check_refusal(answer):
            return PASS, "Out-of-scope question correctly refused"
        else:
            return (
                FAIL,
                "Out-of-scope question was NOT refused — bot answered when it should have declined",
            )

    # In-scope question
    if "sorry, an error occurred" in answer.lower():
        return FAIL, "Pipeline returned an error"

    no_info = _check_no_info(answer)
    if no_info:
        if _expected_no_info(question):
            return (
                WARN,
                "Bot said it has no info — this is expected for staff/policy data not published on the website",
            )
        if not chunks:
            return (
                WARN,
                "No chunks retrieved AND bot said it has no info — topic may not be indexed",
            )
        return (
            FAIL,
            "Bot said it has no information for an in-scope question where indexed content was retrieved",
        )

    if not chunks:
        return (
            WARN,
            "Answer provided but no supporting chunks retrieved (may be hallucination)",
        )

    return PASS, f"Answer provided with {len(chunks)} supporting source(s)"


# ── Main ─────────────────────────────────────────────────────────────────────


async def main(hours: int) -> None:
    print(DSEQ)
    print("  ASK WARWICK — Live Prompt Evaluation")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"  Generated : {now}  |  Window: last {hours}h")
    print(DSEQ)

    prompts_with_counts = fetch_prompts(hours)

    if not prompts_with_counts:
        print("\n  No prompts found in the specified window. Nothing to evaluate.")
        return

    results = []
    total = len(prompts_with_counts)

    for idx, (question, count) in enumerate(prompts_with_counts, 1):
        scope = "OUT-OF-SCOPE" if _is_out_of_scope(question) else "IN-SCOPE"
        print(f"\n{SEP}")
        print(f"  [{idx}/{total}]  ({scope})  Asked {count}x")
        print(f"  Q: {question}")
        print(SEP)

        try:
            answer, chunks = await run_prompt(question)
        except Exception as exc:
            answer = f"[ERROR: {exc}]"
            chunks = []

        verdict, reason = evaluate(question, answer, chunks)

        # Print answer (truncated for readability)
        answer_display = answer.strip()
        if len(answer_display) > 600:
            answer_display = answer_display[:600] + "\n  ... [truncated]"
        print(f"\n  Answer:\n  {answer_display.replace(chr(10), chr(10) + '  ')}")

        if chunks:
            print(f"\n  Sources retrieved ({len(chunks)}):")
            for c in chunks[:5]:
                print(f"    • {c['source']}")

        verdict_icon = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠"}.get(verdict, "?")
        print(f"\n  {verdict_icon} Verdict: {verdict}  —  {reason}")

        results.append(
            {
                "question": question,
                "count": count,
                "scope": scope,
                "answer": answer.strip(),
                "chunks": [c["source"] for c in chunks],
                "verdict": verdict,
                "reason": reason,
            }
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    passes = [r for r in results if r["verdict"] == PASS]
    fails = [r for r in results if r["verdict"] == FAIL]
    warns = [r for r in results if r["verdict"] == WARN]

    print(f"\n{DSEQ}")
    print("  EVALUATION SUMMARY")
    print(DSEQ)
    print(f"  Total unique prompts tested : {total}")
    print(f"  ✓ PASS  : {len(passes)}")
    print(f"  ⚠ WARN  : {len(warns)}")
    print(f"  ✗ FAIL  : {len(fails)}")
    print()

    if fails:
        print("  FAILURES:")
        for r in fails:
            print(f"    ✗ [{r['scope']}] {r['question']!r}")
            print(f"        → {r['reason']}")
    if warns:
        print()
        print("  WARNINGS:")
        for r in warns:
            print(f"    ⚠ [{r['scope']}] {r['question']!r}")
            print(f"        → {r['reason']}")

    print(DSEQ)

    # Save JSON results for downstream processing
    out_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"generated": now, "hours": hours, "results": results},
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n  Full results saved to: {out_path}")
    print(DSEQ)
    print()

    return fails, warns


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate live prompts against the RAG pipeline"
    )
    parser.add_argument(
        "--hours", type=int, default=24, help="Hours to look back (default: 24)"
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(args.hours))
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)

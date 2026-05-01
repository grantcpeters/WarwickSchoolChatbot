"""
Ask Warwick — Usage Report
Usage:  python scripts/usage_report.py [--hours 24]
"""

import argparse
import subprocess
import json
import sys
from datetime import datetime, timezone

WORKSPACE_ID = "ad672a5e-50bc-4694-9ad6-a39d549b24da"
ANALYTICS_URL = f"https://api.loganalytics.io/v1/workspaces/{WORKSPACE_ID}/query"


def get_token() -> str:
    cmd = "az account get-access-token --resource https://api.loganalytics.io --query accessToken -o tsv"
    result = subprocess.run(cmd, capture_output=True, text=True, check=True, shell=True)
    return result.stdout.strip()


def run_kql(query: str, token: str) -> list[dict]:
    import urllib.request

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


def hdr(title: str):
    w = 66
    print()
    print("=" * w)
    print(f"  {title}")
    print("=" * w)


def sect(title: str):
    print()
    print(f"  {title}")
    print("  " + "-" * len(title))


def main():
    parser = argparse.ArgumentParser(description="Ask Warwick usage report")
    parser.add_argument(
        "--hours", type=int, default=24, help="Hours to look back (default: 24)"
    )
    args = parser.parse_args()
    hours = args.hours

    hdr(f"Ask Warwick — Usage Report  (last {hours} hours)")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"  Generated : {now}")
    print("  Fetching token...", end="", flush=True)
    token = get_token()
    print(" OK")

    # ── 1. Unique visitors ────────────────────────────────────────────────────

    sect("1. Unique Visitors  (IPs that sent /chat messages)")
    rows = run_kql(
        f"""
AppServiceHTTPLogs
| where TimeGenerated > ago({hours}h)
| where CsUriStem == "/chat" and CsMethod == "POST"
| summarize messages=count() by CIp
| order by messages desc
""",
        token,
    )

    if not rows:
        print("    No /chat requests found.")
    else:
        print(f"    {'IP Address':<22} {'Messages':>8}")
        print(f"    {'----------':<22} {'--------':>8}")
        for r in rows:
            print(f"    {r['CIp']:<22} {r['messages']:>8}")
        total_msg = sum(r["messages"] for r in rows)
        print()
        print(f"    Total unique visitors : {len(rows)}")
        print(f"    Total messages sent   : {total_msg}")

    # ── 2. Total site hits ────────────────────────────────────────────────────

    sect("2. Total Site Hits")
    rows = run_kql(
        f"""
AppServiceHTTPLogs
| where TimeGenerated > ago({hours}h)
| summarize
    page_loads    = countif(CsUriStem == "/" and CsMethod == "GET"),
    chat_calls    = countif(CsUriStem == "/chat" and CsMethod == "POST"),
    feedback_hits = countif(CsUriStem == "/feedback" and CsMethod == "POST"),
    total         = count()
""",
        token,
    )

    if rows:
        h = rows[0]
        print(f"    Page loads  (GET /)    : {h['page_loads']}")
        print(f"    Chat messages (/chat)  : {h['chat_calls']}")
        print(f"    Feedback votes         : {h['feedback_hits']}")
        print(f"    Total HTTP requests    : {h['total']}")

    # ── 3. Feedback ───────────────────────────────────────────────────────────

    sect("3. Feedback — Thumbs Up / Down")
    rows = run_kql(
        f"""
AppServiceConsoleLogs
| where TimeGenerated > ago({hours}h)
| where ResultDescription has "FEEDBACK"
| extend rating   = extract("rating=(GOOD|BAD)", 1, ResultDescription)
| extend question = coalesce(extract(@"msg='([^']+)'", 1, ResultDescription), extract(@'msg="([^"]+)"', 1, ResultDescription))
| extend snippet  = coalesce(extract(@"response_snippet='([^']+)'", 1, ResultDescription), extract(@'response_snippet="([^"]+)"', 1, ResultDescription))
| project TimeGenerated, rating, question, snippet
| order by TimeGenerated desc
""",
        token,
    )

    if not rows:
        print("    No feedback recorded yet.")
    else:
        good = sum(1 for r in rows if r["rating"] == "GOOD")
        bad = sum(1 for r in rows if r["rating"] == "BAD")
        print(f"    Thumbs UP   : {good}")
        print(f"    Thumbs DOWN : {bad}")
        print()
        for r in rows:
            icon = "[+]" if r["rating"] == "GOOD" else "[-]"
            ts = r["TimeGenerated"][:16]
            q = r["question"] or "(unknown)"
            print(f"    {icon} {ts}  Q: {q}")
            if r.get("snippet"):
                snip = r["snippet"][:100]
                print(f"           A: {snip}...")

    # ── 4. Prompts ────────────────────────────────────────────────────────────

    sect("4. Prompts Asked  (most recent first)")
    rows = run_kql(
        f"""
AppServiceConsoleLogs
| where TimeGenerated > ago({hours}h)
| where ResultDescription has "PROMPT"
| extend turns    = toint(extract(@"history_turns=(\d+)", 1, ResultDescription))
| extend question = coalesce(extract(@"msg='([^']+)'", 1, ResultDescription), extract(@'msg="([^"]+)"', 1, ResultDescription))
| project TimeGenerated, turns, question
| order by TimeGenerated desc
""",
        token,
    )

    if not rows:
        print("    No prompts recorded yet.")
    else:
        print(f"    {'Time (UTC)':<16}  {'Turn':<4}  Question")
        print(f"    {'----------':<16}  {'----':<4}  --------")
        for r in rows:
            ts = r["TimeGenerated"][:16]
            q = r["question"] or ""
            q = q[:68] + "..." if len(q) > 68 else q
            print(f"    {ts:<16}  {str(r['turns']):<4}  {q}")

        print()
        print("    --- Top questions ---")
        from collections import Counter

        counts = Counter(r["question"] for r in rows if r["question"])
        for question, count in counts.most_common(10):
            print(f"    {count:3}x  {question}")

    print()
    print("=" * 66)
    print("  End of report")
    print("=" * 66)
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)

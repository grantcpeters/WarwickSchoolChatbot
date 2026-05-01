"""
Ask Warwick — Deploy Usage Workbook to Azure Monitor.

Creates (or updates) the workbook in the portal. Run once:
    python scripts/deploy_workbook.py

Then open in portal:
    Log Analytics workspace → Workbooks → Ask Warwick Usage Report
"""

import json
import os
import subprocess
import sys
import uuid
import tempfile

RESOURCE_GROUP = "warwickschoolchatbot-rg"
WORKSPACE_NAME = "warwickprep-5bwffhz4ojyvq-logs"
WORKBOOK_NAME = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"  # stable GUID for this workbook
WORKBOOK_DISPLAY = "Ask Warwick Usage Report"
LOCATION = "uksouth"
API_VERSION = "2022-04-01"


def az(cmd: str) -> str:
    result = subprocess.run(
        "az " + cmd, capture_output=True, text=True, shell=True, check=True
    )
    return result.stdout.strip()


def az_json(cmd: str) -> dict:
    return json.loads(az(cmd + " -o json"))


# ── KQL queries ────────────────────────────────────────────────────────────────

Q_VISITORS = """\
AppServiceHTTPLogs
| where CsUriStem == "/chat" and CsMethod == "POST"
| summarize Messages=count() by ['IP Address']=CIp
| order by Messages desc"""

Q_HITS = """\
AppServiceHTTPLogs
| summarize
    ['Page loads']    = countif(CsUriStem == "/" and CsMethod == "GET"),
    ['Chat messages'] = countif(CsUriStem == "/chat" and CsMethod == "POST"),
    ['Feedback votes']= countif(CsUriStem == "/feedback" and CsMethod == "POST"),
    ['Total requests']= count()"""

_Q_MSG = """coalesce(extract(@"msg='([^']+)'", 1, ResultDescription), extract(@'msg="([^"]+)"', 1, ResultDescription))"""
_Q_SNIP = """coalesce(extract(@"response_snippet='([^']+)'", 1, ResultDescription), extract(@'response_snippet="([^"]+)"', 1, ResultDescription))"""

Q_FEEDBACK = (
    "AppServiceConsoleLogs\n"
    '| where ResultDescription has "FEEDBACK"\n'
    '| extend Rating    = extract("rating=(GOOD|BAD)", 1, ResultDescription)\n'
    f"| extend Question  = {_Q_MSG}\n"
    f"| extend ['Answer preview'] = {_Q_SNIP}\n"
    "| project TimeGenerated, Rating, Question, ['Answer preview']\n"
    "| order by TimeGenerated desc"
)

Q_TOP = (
    "AppServiceConsoleLogs\n"
    '| where ResultDescription has "PROMPT"\n'
    f"| extend Question = {_Q_MSG}\n"
    "| where isnotempty(Question)\n"
    "| summarize Count=count() by Question\n"
    "| order by Count desc\n"
    "| take 20"
)

Q_ALL = (
    "AppServiceConsoleLogs\n"
    '| where ResultDescription has "PROMPT"\n'
    '| extend Turns    = toint(extract(@"history_turns=(\\d+)", 1, ResultDescription))\n'
    f"| extend Question = {_Q_MSG}\n"
    "| order by TimeGenerated desc\n"
    "| project ['Time (UTC)']=TimeGenerated, Turns, Question"
)


# ── Workbook item builders ────────────────────────────────────────────────────


def text_item(markdown: str) -> dict:
    return {
        "type": 1,
        "content": {"json": markdown},
        "name": f"text-{uuid.uuid4().hex[:8]}",
        "styleSettings": {},
    }


def query_item(name: str, query: str, visualization: str = "table") -> dict:
    return {
        "type": 3,
        "content": {
            "version": "KqlItem/1.0",
            "query": query,
            "size": 0,
            "queryType": 0,
            "resourceType": "microsoft.operationalinsights/workspaces",
            "visualization": visualization,
            "timeContext": {"durationMs": 0},
            "timeContextFromParameter": "TimeRange",
        },
        "name": name,
        "timeContext": {"durationMs": 0},
        "timeContextFromParameter": "TimeRange",
    }


def build_workbook(workspace_resource_id: str) -> dict:
    return {
        "version": "Notebook/1.0",
        "items": [
            # Time range parameter picker
            {
                "type": 9,
                "content": {
                    "version": "KqlParameterItem/1.0",
                    "parameters": [
                        {
                            "id": str(uuid.uuid4()),
                            "version": "KqlParameterItem/1.0",
                            "name": "TimeRange",
                            "label": "Time range",
                            "type": 4,
                            "value": {"durationMs": 86400000},
                            "typeSettings": {
                                "selectableValues": [
                                    {
                                        "durationMs": 3600000,
                                        "displayName": "Last 1 hour",
                                    },
                                    {
                                        "durationMs": 86400000,
                                        "displayName": "Last 24 hours",
                                    },
                                    {
                                        "durationMs": 172800000,
                                        "displayName": "Last 48 hours",
                                    },
                                    {
                                        "durationMs": 604800000,
                                        "displayName": "Last 7 days",
                                    },
                                    {
                                        "durationMs": 2592000000,
                                        "displayName": "Last 30 days",
                                    },
                                ]
                            },
                        }
                    ],
                    "style": "above",
                    "queryType": 0,
                    "resourceType": "microsoft.operationalinsights/workspaces",
                },
                "name": "parameters",
            },
            text_item(
                "## 1. Unique Visitors\nIPs that sent at least one chat message."
            ),
            query_item("visitors", Q_VISITORS, "table"),
            text_item("## 2. Total Site Hits"),
            query_item("hits", Q_HITS, "table"),
            text_item("## 3. Feedback — Thumbs Up / Down"),
            query_item("feedback", Q_FEEDBACK, "table"),
            text_item("## 4. Top Questions"),
            query_item("top-questions", Q_TOP, "barchart"),
            text_item("## 5. All Prompts  *(newest first)*"),
            query_item("all-prompts", Q_ALL, "table"),
        ],
        "fallbackResourceIds": [workspace_resource_id],
        "$schema": "https://github.com/Microsoft/Application-Insights-Workbooks/blob/master/schema/workbook.json",
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    print("Ask Warwick — Deploy Usage Workbook")
    print("=" * 42)

    print("Fetching subscription ID...", end=" ", flush=True)
    sub_id = az("account show --query id -o tsv")
    print(sub_id)

    workspace_resource_id = (
        f"/subscriptions/{sub_id}/resourceGroups/{RESOURCE_GROUP}"
        f"/providers/Microsoft.OperationalInsights/workspaces/{WORKSPACE_NAME}"
    )
    workbook_resource_id = (
        f"/subscriptions/{sub_id}/resourceGroups/{RESOURCE_GROUP}"
        f"/providers/microsoft.insights/workbooks/{WORKBOOK_NAME}"
    )

    print("Building workbook definition...", end=" ", flush=True)
    workbook_data = build_workbook(workspace_resource_id)
    serialized = json.dumps(workbook_data)
    print("OK")

    full_body = {
        "name": WORKBOOK_NAME,
        "type": "microsoft.insights/workbooks",
        "location": LOCATION,
        "kind": "shared",
        "properties": {
            "displayName": WORKBOOK_DISPLAY,
            "serializedData": serialized,
            "version": "1.0",
            "sourceId": workspace_resource_id,
            "category": "workbook",
        },
    }

    # Write body to temp file (avoids shell quoting issues with large JSON)
    tf = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(full_body, tf)
    tf.close()

    print("Deploying to Azure...", end=" ", flush=True)
    try:
        az(
            f"resource create"
            f' --id "{workbook_resource_id}"'
            f" --api-version {API_VERSION}"
            f" --is-full-object"
            f" --properties @{tf.name}"
        )
        print("OK")
    finally:
        os.unlink(tf.name)

    portal_url = f"https://portal.azure.com/#resource{workbook_resource_id}/workbook"
    print()
    print("Workbook deployed successfully!")
    print()
    print(f"  Open in portal:")
    print(f"  {portal_url}")
    print()
    print("  Or: Log Analytics workspace → Workbooks → Ask Warwick Usage Report")
    print()


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\nAzure CLI error:\n{e.stderr}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)

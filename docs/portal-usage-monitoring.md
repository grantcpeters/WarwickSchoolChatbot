# Usage Monitoring — Azure Portal Guide

This guide shows how to view site usage (visitors, hits, feedback, prompts) directly
in the Azure Portal using Log Analytics, without running any scripts.

---

## Prerequisites

- Sign in to [portal.azure.com](https://portal.azure.com)
- Navigate to the Log Analytics workspace:
  **Resource groups → warwickschoolchatbot-rg → warwickprep-5bwffhz4ojyvq-logs**

---

## Part 1 — Run the Four Queries

From inside the Log Analytics workspace, click **Logs** in the left sidebar.
Dismiss the query picker if it appears.

Paste each query below into the editor, set your time range (top-right dropdown,
e.g. "Last 24 hours"), then click **Run**.

---

### Query 1 — Unique Visitors

```kql
AppServiceHTTPLogs
| where CsUriStem == "/chat" and CsMethod == "POST"
| summarize messages=count() by CIp
| order by messages desc
```

**What you see:** One row per unique IP address with a count of messages sent.

---

### Query 2 — Total Site Hits

```kql
AppServiceHTTPLogs
| summarize
    page_loads    = countif(CsUriStem == "/" and CsMethod == "GET"),
    chat_messages = countif(CsUriStem == "/chat" and CsMethod == "POST"),
    feedback_hits = countif(CsUriStem == "/feedback" and CsMethod == "POST"),
    total         = count()
```

**What you see:** A single row with page loads, chat calls, feedback votes, and total requests.

---

### Query 3 — Feedback (Thumbs Up / Down)

```kql
AppServiceConsoleLogs
| where ResultDescription has "FEEDBACK"
| extend rating   = extract("rating=(GOOD|BAD)", 1, ResultDescription)
| extend question = coalesce(
    extract(@"msg='([^']+)'", 1, ResultDescription),
    extract(@'msg="([^"]+)"', 1, ResultDescription))
| extend snippet  = coalesce(
    extract(@"response_snippet='([^']+)'", 1, ResultDescription),
    extract(@'response_snippet="([^"]+)"', 1, ResultDescription))
| project TimeGenerated, rating, question, snippet
| order by TimeGenerated desc
```

**What you see:** Each thumbs up/down with the question that triggered it and the
first 120 characters of the AI's answer.

---

### Query 4 — Prompts Asked

```kql
AppServiceConsoleLogs
| where ResultDescription has "PROMPT"
| extend turns    = toint(extract(@"history_turns=(\d+)", 1, ResultDescription))
| extend question = coalesce(
    extract(@"msg='([^']+)'", 1, ResultDescription),
    extract(@'msg="([^"]+)"', 1, ResultDescription))
| project TimeGenerated, turns, question
| order by TimeGenerated desc
```

**What you see:** Every message sent to the chatbot, newest first, with how many
turns into the conversation it was.

---

### Bonus — Top Questions

```kql
AppServiceConsoleLogs
| where ResultDescription has "PROMPT"
| extend question = coalesce(
    extract(@"msg='([^']+)'", 1, ResultDescription),
    extract(@'msg="([^"]+)"', 1, ResultDescription))
| where isnotempty(question)
| summarize count=count() by question
| order by count desc
| take 20
```

---

## Part 2 — Save Queries as Favourites

So you can re-run them without pasting each time:

1. Run any of the queries above
2. Click **Save** (top toolbar) → **Save as query**
3. Fill in:
   - **Name** — e.g. `Ask Warwick — Unique Visitors`
   - **Category** — e.g. `Ask Warwick`
   - **Description** — optional
4. Click **Save**
5. Repeat for all four queries

To open saved queries later: click the **Queries** button (top-left of the Logs editor)
→ filter by your category name.

---

## Part 3 — Pin Results to a Dashboard

After running any query:

1. Click the **chart icon** (top-right of the results panel) to switch to a visual
   — Bar chart works well for visitor counts; Table for prompts
2. Click **Pin to dashboard** (pin icon, top-right)
3. Choose **Create new** → give the dashboard a name like `Ask Warwick Usage`
4. Click **Pin**

Repeat for each query. You'll build up a dashboard with all four panels.

To view the dashboard later: click the **≡ menu** (top-left) → **Dashboard** →
select `Ask Warwick Usage` from the dropdown.

---

## Part 4 — Build a Workbook (Optional, Nicer View)

Workbooks let you combine all four queries into one scrollable page with headings.

1. In the Log Analytics workspace, click **Workbooks** in the left sidebar
2. Click **+ New**
3. Click **Add** → **Add query**
4. Paste Query 1, set **Visualization** to `Grid`, click **Done editing**
5. Click **Add** → **Add text**, type `## Unique Visitors` as a heading
6. Repeat for Queries 2–4
7. Click **Save** → name it `Ask Warwick Usage Report`
   - Set **Resource group**: `warwickschoolchatbot-rg`
   - Set **Location**: same region as your workspace

To open later: **Workbooks** → click `Ask Warwick Usage Report`.

> **Tip:** At the top of the workbook, add a **Parameters** block with a time-range
> picker so you can switch between "last 24h / 7 days / 30 days" without editing queries.

---

## Quick Reference

| What           | Where in portal                         |
| -------------- | --------------------------------------- |
| Run ad-hoc KQL | Log Analytics workspace → **Logs**      |
| Saved queries  | Logs editor → **Queries** button        |
| Dashboard      | Portal menu → **Dashboard**             |
| Workbook       | Log Analytics workspace → **Workbooks** |
| Time range     | Top-right of every Logs/Workbook view   |

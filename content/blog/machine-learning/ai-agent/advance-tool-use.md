---
title: "Advanced Tool Use: How Claude Discovers, Learns, and Executes Tools Dynamically"
publishDate: "2026-02-24"
category: "machine-learning"
subcategory: "AI Agent"
tags:
  ["AI", "tool-use", "Claude", "Anthropic", "MCP", "AI Agents", "LLM", "API"]
date: "2026-02-24"
author: "Hiep Tran"
featured: true
image: "/imgs/blogs/advance-tool-use-20260225235101.png"
excerpt: "A deep dive into Anthropic's three advanced tool use features — Tool Search Tool, Programmatic Tool Calling, and Tool Use Examples — that enable Claude to work efficiently with hundreds of tools while saving tokens, reducing latency, and improving accuracy."
---

## Introduction

The future of AI agents depends on their ability to seamlessly work across hundreds or even thousands of tools. Imagine an IDE assistant that integrates git operations, file manipulation, package managers, testing frameworks, and deployment pipelines. Or an operations coordinator connecting Slack, GitHub, Google Drive, Jira, company databases, and dozens of MCP servers simultaneously.

The problem? Traditional tool calling approaches break down at scale. Tool definitions consume massive token budgets, intermediate results pollute the context window, and JSON schemas alone can't teach models the nuances of correct tool usage.

Anthropic introduced three powerful features on the Claude Developer Platform to solve these challenges:

1. **Tool Search Tool** — Discover tools on-demand instead of loading everything upfront
2. **Programmatic Tool Calling (PTC)** — Execute tools through code rather than individual API round-trips
3. **Tool Use Examples** — Teach Claude correct tool usage patterns through concrete examples

In this article, we'll explore each feature in depth, understand the problems they solve, and see how to implement them in practice.

## 1. Tool Search Tool

### The Problem: Context Window Overload

Think of Claude's context window like a desk. Every tool definition you send is a document sitting on that desk. With five tools, the desk is tidy. But in real-world agent setups with multiple MCP servers? Your desk looks like this:

| MCP Server | Number of Tools | Token Cost      |
| ---------- | --------------- | --------------- |
| GitHub     | 35 tools        | ~26K tokens     |
| Slack      | 11 tools        | ~21K tokens     |
| Sentry     | 5 tools         | ~3K tokens      |
| Grafana    | 5 tools         | ~3K tokens      |
| Splunk     | 2 tools         | ~2K tokens      |
| **Total**  | **58 tools**    | **~55K tokens** |

That's **55,000 tokens** consumed before the conversation even starts — and the user hasn't even said "hello" yet. Add more servers like Jira (~17K tokens alone) and you're quickly approaching 100K+ token overhead. At Anthropic, they've seen tool definitions consume **134K tokens** before optimization.

But token cost isn't the only issue. Imagine you have two tools named `notification-send-user` and `notification-send-channel`. When Claude sees 50+ tool definitions at once, it's like trying to find the right screwdriver in a toolbox with 200 tools — the chance of grabbing the wrong one goes up significantly. The most common failures are **wrong tool selection** and **incorrect parameters**.

### The Solution: On-Demand Tool Discovery

The idea is beautifully simple: instead of dumping every tool definition onto Claude's desk at the start, let Claude **search for tools when it needs them** — like having a well-organized tool shed instead of a cluttered workbench.

Here's the difference at a glance:

**Traditional approach (the cluttered desk):**

- All tool definitions loaded upfront (~72K tokens for 50+ MCP tools)
- Conversation history and system prompt compete for remaining space
- Total context consumption: ~77K tokens before any work begins

**With Tool Search Tool (the organized shed):**

- Only the Tool Search Tool itself is loaded upfront (~500 tokens)
- Tools are discovered on-demand as needed (3-5 relevant tools, ~3K tokens)
- Total context consumption: ~8.7K tokens — **preserving 95% of the context window**

That's an **85% reduction** in token usage while Claude retains access to the full tool library.

Internal testing showed dramatic accuracy improvements too:

| Model    | Without Tool Search | With Tool Search | Improvement |
| -------- | ------------------- | ---------------- | ----------- |
| Opus 4   | 49%                 | 74%              | +25pp       |
| Opus 4.5 | 79.5%               | 88.1%            | +8.6pp      |

### How It Works

Let's walk through this step by step with a real scenario.

**Imagine you're building a DevOps assistant** that connects to GitHub, Slack, Jira, Sentry, PagerDuty, and Google Drive. That's 80+ tools total.

**Step 1: Mark tools as deferred.** You provide all 80+ tool definitions to the API, but add `defer_loading: true` to each one. These tools are registered but **not** loaded into Claude's context initially.

**Step 2: Claude starts lean.** When a user starts a conversation, Claude only sees the Tool Search Tool itself (~500 tokens) plus any tools you explicitly keep loaded (`defer_loading: false`) — like your 3-5 most-used tools.

**Step 3: Claude searches when needed.** When the user says _"Create a PR for the bugfix branch and notify the team on Slack,"_ Claude searches for "github pull request" and "slack message." Only `github.createPullRequest` and `slack.sendMessage` get loaded into context — not the other 78 tools.

Here's a visual analogy:

```
❌ Traditional: Load ALL tools → [GitHub×35] [Slack×11] [Jira×15] [Sentry×5] ... = 77K tokens
✅ Tool Search: Load search tool → User asks about GitHub → Load github.createPR = 3.5K tokens
```

### Two Search Variants

The Claude Developer Platform provides two built-in search strategies, each suited to different use cases:

| Variant   | Type ID                           | How Claude Searches                                                                                | Best For                                                                                                         |
| --------- | --------------------------------- | -------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **Regex** | `tool_search_tool_regex_20251119` | Claude constructs regex patterns like `github\|pull.?request` to match tool names and descriptions | Precise, structured lookups where Claude knows the exact tool name or keyword pattern                            |
| **BM25**  | `tool_search_tool_bm25_20251119`  | Claude writes natural language queries like `"create a pull request on GitHub"`                    | Fuzzy, intent-based discovery where Claude describes what it wants to do rather than knowing the exact tool name |

**Regex** is great when your tools have consistent naming conventions (e.g., `github.createPullRequest`, `slack.sendMessage`) — Claude can pattern-match directly. **BM25** shines when tool names are less predictable or when Claude needs to describe a capability rather than guess a name (e.g., searching for `"send a notification to a user"` instead of guessing whether the tool is called `notify-user`, `send-notification`, or `user-alert`).

You can also implement a **custom search tool** using embeddings or any other strategy by defining your own tool that accepts a search query and returns matching tool references.

Here's how to choose:

```
Use Case                              → Variant
─────────────────────────────────────────────────────
Tools with clear, consistent names     → Regex
Tools with verbose/varied descriptions → BM25
Need semantic understanding            → Custom (embeddings)
```

### Implementation Example

```json
{
  "tools": [
    // The search tool itself — always loaded (choose one variant)
    // Option A: Regex-based search
    {
      "type": "tool_search_tool_regex_20251119",
      "name": "tool_search_tool_regex"
    },
    // Option B: BM25-based search (use one or the other, not both)
    // {
    //   "type": "tool_search_tool_bm25_20251119",
    //   "name": "tool_search_tool_bm25"
    // },

    // GitHub tools — discoverable on demand
    {
      "name": "github.createPullRequest",
      "description": "Create a pull request in a GitHub repository",
      "input_schema": { "...": "..." },
      "defer_loading": true
    },
    {
      "name": "github.listIssues",
      "description": "List open issues for a GitHub repository",
      "input_schema": { "...": "..." },
      "defer_loading": true
    },

    // Slack tools — discoverable on demand
    {
      "name": "slack.sendMessage",
      "description": "Send a message to a Slack channel",
      "input_schema": { "...": "..." },
      "defer_loading": true
    },
    {
      "name": "slack.createChannel",
      "description": "Create a new Slack channel",
      "input_schema": { "...": "..." },
      "defer_loading": true
    }
    // ... hundreds more deferred tools from Jira, Sentry, PagerDuty...
  ]
}
```

For MCP servers, you can defer loading **entire servers** while keeping specific high-use tools loaded:

```json
{
  "type": "mcp_toolset",
  "mcp_server_name": "google-drive",
  "default_config": {
    "defer_loading": true
  },
  "configs": {
    "search_files": {
      "defer_loading": false
    }
  }
}
```

In this example, the entire Google Drive server is deferred, but `search_files` — the tool people use most — stays loaded at all times. It's like keeping a flashlight on your belt while the rest of your gear stays in the truck.

### Example: An E-commerce Support Agent

Let's say you're building a customer support agent with these MCP servers:

- **Shopify** (25 tools): orders, products, inventory, fulfillment...
- **Zendesk** (15 tools): tickets, users, macros, triggers...
- **Stripe** (20 tools): payments, refunds, subscriptions, invoices...
- **Twilio** (10 tools): SMS, calls, voice messages...

That's 70 tools at ~60K tokens. But most support conversations only need 2-3 tools.

When a customer asks _"Where's my order #12345?"_, Claude searches for "order tracking" and loads just:

- `shopify.getOrder`
- `shopify.getShipmentTracking`

When another customer says _"I was charged twice, please refund one"_, Claude searches for "refund" and loads:

- `stripe.createRefund`
- `stripe.getPaymentIntent`
- `zendesk.createTicket` (to log the refund)

Each conversation uses ~3K tokens for tools instead of 60K. Over thousands of daily conversations, the cost savings are enormous.

> **Prompt caching note:** Tool Search Tool doesn't break prompt caching because deferred tools are excluded from the initial prompt entirely. They're only added to context after Claude searches for them, so your system prompt and core tool definitions remain perfectly cacheable.

### When to Use It

| Scenario                                 | Recommendation                  |
| ---------------------------------------- | ------------------------------- |
| 50+ tools across multiple MCP servers    | ✅ Strong recommendation        |
| Tool definitions consuming >10K tokens   | ✅ Clear benefit                |
| Experiencing wrong-tool-selection errors | ✅ Accuracy improvement         |
| 10-50 tools across a few servers         | ✅ Worth trying                 |
| <10 tools, all used frequently           | ⚠️ Overhead may not be worth it |
| All tools needed in every conversation   | ⚠️ Better to keep them loaded   |

## 2. Programmatic Tool Calling (PTC)

### The Problem: Death by a Thousand Tool Calls

Traditional tool calling works like a conversation between a manager and an assistant. The manager (Claude) says: _"Get me Alice's expenses."_ The assistant comes back with a stack of receipts. The manager reads them all, then says: _"Now get me Bob's expenses."_ Another stack of receipts. Repeat 20 times.

This creates two serious problems:

**Problem 1: Context pollution.** Every intermediate result piles into Claude's context window. If Claude analyzes server logs for error patterns, the entire 10MB log file enters context — even though Claude only needs a one-line summary like _"Found 47 timeout errors, mostly from the payments service."_ It's like printing out an entire phone book to find one number.

**Problem 2: Inference overhead.** Each tool call requires a full model inference pass. After each one, Claude must re-read everything, figure out what it learned, and decide what to do next. A 20-tool workflow means 20 inference passes — that's 20 times Claude "thinking" when most of that thinking is just bookkeeping.

### Example 1: Budget Compliance Check

Let's make this concrete. Your boss asks: _"Which team members exceeded their Q3 travel budget?"_

You have three tools:

- `get_team_members(department)` — Returns list of team members
- `get_expenses(user_id, quarter)` — Returns expense line items
- `get_budget_by_level(level)` — Returns budget limits

**The traditional approach is painfully wasteful:**

```
Claude: "Get team members for engineering"
  → Returns 20 people                                    [Tool call #1]

Claude: "Get expenses for Alice, Q3"
  → Returns 87 line items (flights, hotels, meals...)     [Tool call #2]

Claude: "Get expenses for Bob, Q3"
  → Returns 63 line items                                [Tool call #3]

... repeat 18 more times ...

Claude: "Get budget for junior level"
  → Returns budget details                               [Tool call #22]

Claude: "Get budget for senior level"
  → Returns budget details                               [Tool call #23]

→ Total: 23 tool calls, 2,000+ expense line items in context (50KB+)
→ Claude manually sums each person's expenses and compares to budget
```

That's a lot of wasted tokens and inference passes for a question that has a simple answer.

### The Solution: Let Claude Write Code Instead

The core idea of Programmatic Tool Calling is: **let Claude express orchestration logic in Python instead of natural language**. Claude writes a script that runs in a sandbox, calls tools, processes results, and only sends the final answer back to context.

It's the difference between:

- **Traditional:** Claude is a manager who reads every receipt and does mental math
- **PTC:** Claude is a programmer who writes a script to crunch the numbers

Here's what Claude generates for the budget check:

```python
import json
import asyncio

team = await get_team_members("engineering")

# Fetch budgets for each unique level (parallel!)
levels = list(set(m["level"] for m in team))
budget_results = await asyncio.gather(*[
    get_budget_by_level(level) for level in levels
])
budgets = {level: budget for level, budget in zip(levels, budget_results)}

# Fetch ALL expenses in parallel — not one by one
expenses = await asyncio.gather(*[
    get_expenses(m["id"], "Q3") for m in team
])

# Crunch the numbers in code
exceeded = []
for member, exp in zip(team, expenses):
    budget = budgets[member["level"]]
    total = sum(e["amount"] for e in exp)
    if total > budget["travel_limit"]:
        exceeded.append({
            "name": member["name"],
            "spent": total,
            "limit": budget["travel_limit"]
        })

print(json.dumps(exceeded, indent=2))
```

What Claude's context actually sees at the end:

```json
[
  { "name": "Alice", "spent": 12500, "limit": 10000 },
  { "name": "David", "spent": 8700, "limit": 7500 }
]
```

That's it. **1KB instead of 200KB.** The 2,000+ expense line items, the summation, the budget lookups — all happened inside the sandbox and never touched Claude's context.

### Example 2: Log Analysis Across Multiple Servers

Here's another scenario where PTC really shines. Imagine you're building an SRE agent and the user asks: _"What were the top 5 error types across all production servers in the last hour?"_

You have tools:

- `list_servers(environment)` — Lists servers
- `get_logs(server_id, time_range, level)` — Returns log entries

**Without PTC:** Claude fetches logs from each of 15 servers. Each returns 500+ log entries. That's 7,500+ log entries flooding the context. Claude then tries to mentally categorize and count error types across all of them. Good luck.

**With PTC:** Claude writes a script:

```python
import asyncio
from collections import Counter

servers = await list_servers("production")

# Fetch error logs from all servers in parallel
all_logs = await asyncio.gather(*[
    get_logs(s["id"], "last_1h", "ERROR") for s in servers
])

# Count error types across all servers
error_counts = Counter()
for server_logs in all_logs:
    for entry in server_logs:
        error_counts[entry["error_type"]] += 1

# Return only the top 5
top_errors = error_counts.most_common(5)
for error_type, count in top_errors:
    print(f"{error_type}: {count} occurrences")
```

Claude sees:

```
DatabaseTimeout: 142 occurrences
ConnectionRefused: 87 occurrences
OutOfMemory: 34 occurrences
RateLimitExceeded: 28 occurrences
CertificateExpired: 15 occurrences
```

Five lines instead of 7,500 log entries. Claude can now give the user a clean, actionable summary.

### Example 3: Multi-API Data Enrichment

Your marketing agent needs to _"Find all inactive users who haven't logged in for 30 days and check their subscription status."_

Tools: `get_inactive_users(days)`, `get_subscription(user_id)`, `get_last_activity(user_id)`

```python
import asyncio

# Step 1: Get inactive users
inactive = await get_inactive_users(30)

# Step 2: Enrich with subscription + activity data in parallel
subs = await asyncio.gather(*[get_subscription(u["id"]) for u in inactive])
activities = await asyncio.gather(*[get_last_activity(u["id"]) for u in inactive])

# Step 3: Build a concise summary — only paying subscribers matter
at_risk = []
for user, sub, activity in zip(inactive, subs, activities):
    if sub["status"] == "active" and sub["plan"] in ["pro", "enterprise"]:
        at_risk.append({
            "name": user["name"],
            "email": user["email"],
            "plan": sub["plan"],
            "monthly_value": sub["amount"],
            "last_seen": activity["last_login"]
        })

# Sort by revenue impact
at_risk.sort(key=lambda x: x["monthly_value"], reverse=True)
print(json.dumps(at_risk[:10], indent=2))
```

Claude only sees the top 10 at-risk paying customers — not the full list of 500 inactive users with all their subscription and activity details.

### Performance Gains

| Metric                     | Traditional | With PTC     | Improvement                    |
| -------------------------- | ----------- | ------------ | ------------------------------ |
| Average token usage        | 43,588      | 27,297       | **37% reduction**              |
| API round-trips (20 items) | 20+         | 1 code block | **~19 fewer inference passes** |

Additional accuracy improvements from internal benchmarks:

- Internal knowledge retrieval: 25.6% → 28.5%
- GIA benchmarks: 46.5% → 51.2%

And there's a real production use case: [Claude for Excel](https://www.claude.com/claude-for-excel) uses PTC to read and modify spreadsheets with thousands of rows without overloading the model's context window.

### How to Implement PTC

The implementation follows four clear steps:

#### Step 1: Mark tools as callable from code

Add `code_execution` and set `allowed_callers` on tools you want Claude to call programmatically:

```json
{
  "tools": [
    {
      "type": "code_execution_20250825",
      "name": "code_execution"
    },
    {
      "name": "get_team_members",
      "description": "Get all members of a department. Returns list of {id, name, level, email}.",
      "input_schema": { "...": "..." },
      "allowed_callers": ["code_execution_20250825"]
    },
    {
      "name": "get_expenses",
      "description": "Get expense items for a user. Returns list of {amount, category, date, vendor}.",
      "input_schema": { "...": "..." },
      "allowed_callers": ["code_execution_20250825"]
    },
    {
      "name": "get_budget_by_level",
      "description": "Get budget limits for an employee level. Returns {travel_limit, equipment_limit}.",
      "input_schema": { "...": "..." },
      "allowed_callers": ["code_execution_20250825"]
    }
  ]
}
```

The API automatically converts these into Python async functions that Claude can `await`.

#### Step 2: Claude writes orchestration code

Instead of requesting tools one at a time, Claude generates a complete Python script:

```json
{
  "type": "server_tool_use",
  "id": "srvtoolu_abc",
  "name": "code_execution",
  "input": {
    "code": "team = await get_team_members('engineering')\n..."
  }
}
```

#### Step 3: Tools execute in the sandbox, not Claude's context

When the code calls `get_expenses()`, the API sends you a tool request with a `caller` field indicating it came from the code sandbox:

```json
{
  "type": "tool_use",
  "id": "toolu_xyz",
  "name": "get_expenses",
  "input": { "user_id": "emp_123", "quarter": "Q3" },
  "caller": {
    "type": "code_execution_20250825",
    "tool_id": "srvtoolu_abc"
  }
}
```

You return the result, and it goes straight to the sandbox — **not** to Claude's context. This repeats for every tool call in the script.

#### Step 4: Only the final `print()` output enters context

```json
{
  "type": "code_execution_tool_result",
  "tool_use_id": "srvtoolu_abc",
  "content": {
    "stdout": "[{\"name\": \"Alice\", \"spent\": 12500, \"limit\": 10000}...]"
  }
}
```

This is **everything** Claude sees. The 2,000+ line items? Processed and discarded.

### When to Use PTC

| Scenario                                          | Recommendation                         |
| ------------------------------------------------- | -------------------------------------- |
| Processing large datasets, need only summaries    | ✅ Huge benefit                        |
| Multi-step workflows with 3+ dependent tool calls | ✅ Reduces round-trips dramatically    |
| Parallel operations across many items             | ✅ `asyncio.gather` is your friend     |
| Filtering/transforming data before Claude reasons | ✅ Keeps context clean                 |
| Single-tool lookups with small responses          | ⚠️ Overhead not worth it               |
| Claude needs to see all intermediate reasoning    | ⚠️ Let it use traditional tool calling |

## 3. Tool Use Examples

### The Problem: JSON Schema Tells You _What_, Not _How_

JSON Schema is like a grammar rulebook. It tells you that a sentence needs a subject and a verb, but it doesn't teach you how to write well. Schemas define structure — types, required fields, allowed enums — but they can't express the **nuances** of how a tool should actually be used.

Let's look at a real example. Here's a support ticket API schema:

```json
{
  "name": "create_ticket",
  "input_schema": {
    "properties": {
      "title": { "type": "string" },
      "priority": { "enum": ["low", "medium", "high", "critical"] },
      "labels": { "type": "array", "items": { "type": "string" } },
      "reporter": {
        "type": "object",
        "properties": {
          "id": { "type": "string" },
          "name": { "type": "string" },
          "contact": {
            "type": "object",
            "properties": {
              "email": { "type": "string" },
              "phone": { "type": "string" }
            }
          }
        }
      },
      "due_date": { "type": "string" },
      "escalation": {
        "type": "object",
        "properties": {
          "level": { "type": "integer" },
          "notify_manager": { "type": "boolean" },
          "sla_hours": { "type": "integer" }
        }
      }
    },
    "required": ["title"]
  }
}
```

Looks complete, right? But imagine you're Claude seeing this for the first time. You'd have so many questions:

- **Date format?** `"2024-11-06"`, `"Nov 6, 2024"`, or `"2024-11-06T00:00:00Z"`?
- **ID convention?** Is `reporter.id` a UUID like `"550e8400-e29b"`, a prefixed string like `"USR-12345"`, or just `"12345"`?
- **When to include nested objects?** Should `reporter.contact` always be filled, or only for urgent issues?
- **How do parameters relate?** If `priority` is `"critical"`, should `escalation.sla_hours` be 4 hours or 24 hours?

The schema says all these inputs are valid. But your API has **conventions** — and without examples, Claude is guessing.

### The Solution: Show, Don't Tell

Tool Use Examples work like training a new team member. Instead of handing them a 50-page style guide, you show them three real tickets and say: _"Make yours look like these."_

You add an `input_examples` array directly to your tool definition:

```json
{
  "name": "create_ticket",
  "input_schema": { "...": "same schema as above" },
  "input_examples": [
    {
      "title": "Login page returns 500 error",
      "priority": "critical",
      "labels": ["bug", "authentication", "production"],
      "reporter": {
        "id": "USR-12345",
        "name": "Jane Smith",
        "contact": {
          "email": "jane@acme.com",
          "phone": "+1-555-0123"
        }
      },
      "due_date": "2024-11-06",
      "escalation": {
        "level": 2,
        "notify_manager": true,
        "sla_hours": 4
      }
    },
    {
      "title": "Add dark mode support",
      "labels": ["feature-request", "ui"],
      "reporter": {
        "id": "USR-67890",
        "name": "Alex Chen"
      }
    },
    {
      "title": "Update API documentation"
    }
  ]
}
```

Notice how the three examples tell a story — from fully-specified to minimal:

| Example                               | What It Teaches                                                                                                                                                        |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Critical bug** (full specification) | Dates are `YYYY-MM-DD`. IDs use `USR-XXXXX`. Critical issues get full contact info, escalation with `sla_hours: 4`, and manager notification. Labels are `kebab-case`. |
| **Feature request** (partial)         | Feature requests get a reporter but **no** contact details or escalation. Labels describe the request type. No due date needed.                                        |
| **Internal task** (minimal)           | Simple internal tasks only need a title. No reporter, no labels, no escalation.                                                                                        |

From just three examples, Claude learns all the implicit conventions that the schema can't express:

| What Claude Learns                  | From Which Example                                |
| ----------------------------------- | ------------------------------------------------- |
| Date format: `YYYY-MM-DD`           | Example 1                                         |
| ID convention: `USR-XXXXX`          | Examples 1 & 2                                    |
| Label style: `kebab-case`           | Examples 1 & 2                                    |
| Nested contact for critical only    | Example 1 has it, Example 2 doesn't               |
| Escalation correlates with priority | Example 1 (critical) has escalation, others don't |

### Example 2: Search API with Complex Filters

Here's another case where examples shine — a search API with many optional filters:

```json
{
  "name": "search_products",
  "description": "Search the product catalog with filters",
  "input_schema": {
    "properties": {
      "query": { "type": "string" },
      "category": { "type": "string" },
      "price_range": {
        "type": "object",
        "properties": {
          "min": { "type": "number" },
          "max": { "type": "number" },
          "currency": { "type": "string" }
        }
      },
      "availability": { "enum": ["in_stock", "pre_order", "all"] },
      "sort_by": {
        "enum": ["relevance", "price_asc", "price_desc", "rating", "newest"]
      },
      "limit": { "type": "integer" }
    },
    "required": ["query"]
  },
  "input_examples": [
    {
      "query": "wireless noise-cancelling headphones",
      "category": "electronics/audio",
      "price_range": { "min": 50, "max": 300, "currency": "USD" },
      "availability": "in_stock",
      "sort_by": "rating",
      "limit": 10
    },
    {
      "query": "organic green tea",
      "category": "grocery/beverages",
      "sort_by": "price_asc"
    },
    {
      "query": "MacBook Pro M4"
    }
  ]
}
```

Without examples, Claude might write `"category": "headphones"` or `"category": "Electronics"`. With examples, it learns the path-based, lowercase convention: `"electronics/audio"`, `"grocery/beverages"`. It also learns that `currency` defaults to USD, `limit` is optional, and simple searches just need a query string.

### Example 3: Similar Tools That Need Differentiation

When you have tools with similar names, examples clarify **when to use which**:

```json
// Tool 1: For customer-reported issues
{
  "name": "create_incident",
  "description": "Create an incident for customer-impacting issues",
  "input_examples": [
    {
      "title": "Payment processing failing for EU customers",
      "severity": "SEV1",
      "affected_customers": 2500,
      "region": "eu-west-1"
    }
  ]
}

// Tool 2: For internal work items
{
  "name": "create_ticket",
  "description": "Create a ticket for internal work tracking",
  "input_examples": [
    {
      "title": "Upgrade Redis from 6.x to 7.x",
      "priority": "medium",
      "labels": ["infrastructure", "upgrade"]
    }
  ]
}
```

Now Claude understands: customer-facing outages → `create_incident`, internal engineering work → `create_ticket`. Without examples, it might confuse the two.

In Anthropic's internal testing, tool use examples improved accuracy from **72% to 90%** on complex parameter handling.

### When to Use Tool Use Examples

| Scenario                                         | Recommendation                                 |
| ------------------------------------------------ | ---------------------------------------------- |
| Complex nested structures with ambiguous usage   | ✅ High impact                                 |
| Many optional parameters with inclusion patterns | ✅ Examples show what to include when          |
| Domain-specific conventions (ID formats, naming) | ✅ Show the convention once, Claude follows it |
| Similar tools needing differentiation            | ✅ Examples clarify which to use               |
| Simple single-parameter tools                    | ⚠️ Schema is usually enough                    |
| Standard formats (URLs, emails)                  | ⚠️ Claude already knows these                  |

---

## Best Practices: Combining the Three Features

These three features solve different bottlenecks and work together as a complementary system. Here's how to combine them effectively.

### Layer Features Strategically

Start with your biggest bottleneck:

| Bottleneck                                   | Solution                    |
| -------------------------------------------- | --------------------------- |
| Context bloat from tool definitions          | → Tool Search Tool          |
| Large intermediate results polluting context | → Programmatic Tool Calling |
| Parameter errors and malformed calls         | → Tool Use Examples         |

Then layer additional features as needed. They're complementary:

- **Tool Search Tool** ensures the right tools are found
- **Programmatic Tool Calling** ensures efficient execution
- **Tool Use Examples** ensures correct invocation

### Optimize Tool Search Discovery

Tool search matches against names and descriptions, so clear definitions improve discovery:

```json
// ✅ Good — descriptive and searchable
{
  "name": "search_customer_orders",
  "description": "Search for customer orders by date range, status, or total amount. Returns order details including items, shipping, and payment info."
}

// ❌ Bad — vague and hard to discover
{
  "name": "query_db_orders",
  "description": "Execute order query"
}
```

Add system prompt guidance so Claude knows what's available:

```text
You have access to tools for Slack messaging, Google Drive file management,
Jira ticket tracking, and GitHub repository operations. Use the tool search
to find specific capabilities.
```

Keep your **3-5 most-used tools always loaded** (`defer_loading: false`), defer the rest.

### Set Up PTC for Correct Execution

Since Claude writes code to parse tool outputs, **document return formats clearly**:

```json
{
  "name": "get_orders",
  "description": "Retrieve orders for a customer.\nReturns:\n    List of order objects, each containing:\n    - id (str): Order identifier\n    - total (float): Order total in USD\n    - status (str): One of 'pending', 'shipped', 'delivered'\n    - items (list): Array of {sku, quantity, price}\n    - created_at (str): ISO 8601 timestamp"
}
```

Opt-in tools that benefit from programmatic orchestration:

- Tools that can run in **parallel** (independent operations)
- Operations safe to **retry** (idempotent)

### Craft Effective Tool Use Examples

- Use **realistic data** — real city names, plausible prices, not "string" or "value"
- Show **variety** — minimal, partial, and full specification patterns
- Keep it **concise** — 1-5 examples per tool
- Focus on **ambiguity** — only add examples where correct usage isn't obvious from the schema

## Getting Started

These features are available in beta. To enable them, add the beta header and include the tools you need:

```python
import anthropic

client = anthropic.Anthropic()

response = client.beta.messages.create(
    betas=["advanced-tool-use-2025-11-20"],
    model="claude-sonnet-4-5-20250929",
    max_tokens=4096,
    tools=[
        # Tool Search Tool for on-demand discovery
        {
            "type": "tool_search_tool_regex_20251119",
            "name": "tool_search_tool_regex"
        },
        # Code Execution for programmatic tool calling
        {
            "type": "code_execution_20250825",
            "name": "code_execution"
        },
        # Your tools with defer_loading, allowed_callers, and input_examples
        {
            "name": "search_customer_orders",
            "description": "Search for customer orders...",
            "input_schema": { "..." : "..." },
            "defer_loading": True,
            "allowed_callers": ["code_execution_20250825"],
            "input_examples": [
                {"customer_id": "CUST-001", "status": "shipped"}
            ]
        }
    ]
)
```

### Useful Resources

- [Tool Search Tool Documentation](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool)
- [Tool Search Cookbook (Embeddings)](https://github.com/anthropics/claude-cookbooks/blob/main/tool_use/tool_search_with_embeddings.ipynb)
- [Programmatic Tool Calling Documentation](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling)
- [PTC Cookbook](https://github.com/anthropics/claude-cookbooks/blob/main/tool_use/programmatic_tool_calling_ptc.ipynb)
- [Tool Use Examples Documentation](https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use#providing-tool-use-examples)

## Summary

| Feature                       | Problem It Solves                           | Key Benefit              | Key Metric          |
| ----------------------------- | ------------------------------------------- | ------------------------ | ------------------- |
| **Tool Search Tool**          | Too many tool definitions consuming context | On-demand discovery      | 85% token reduction |
| **Programmatic Tool Calling** | Intermediate results polluting context      | Code-based orchestration | 37% token reduction |
| **Tool Use Examples**         | Schema ambiguity causing wrong parameters   | Learning from examples   | 72% → 90% accuracy  |

These features move tool use from simple function calling toward **intelligent orchestration**. As agents tackle more complex workflows spanning dozens of tools and large datasets, dynamic discovery, efficient execution, and reliable invocation become foundational capabilities for building production-ready AI systems.

## References

- [Anthropic Engineering: Introducing Advanced Tool Use](https://www.anthropic.com/engineering/advanced-tool-use)
- [Anthropic: Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
- [Anthropic Engineering: Code Execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp)
- [GIA Benchmarks Paper](https://arxiv.org/abs/2311.12983)

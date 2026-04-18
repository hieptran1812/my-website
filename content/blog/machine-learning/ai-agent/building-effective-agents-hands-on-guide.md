---
title: "Building Effective Agents: A Hands-On Guide with Case Studies"
publishDate: "2026-04-17"
category: "machine-learning"
subcategory: "AI Agent"
tags: ["ai-agent", "agent-building", "llm", "tool-use", "prompt-engineering", "architecture", "case-studies", "practical"]
date: "2026-04-17"
author: "Hiep Tran"
featured: false
aiGenerated: true
excerpt: "A practical, step-by-step guide to building LLM agents that actually work in production. We start from a minimal agent loop and build up to a polished system, then walk through three concrete case studies — a code review agent, a customer support agent, and a research agent — showing every design decision along the way."
---

## Who This Guide Is For

This is a hands-on, *builder's* guide. If you want theory and patterns, read the companion piece on [agentic design patterns](/blog/machine-learning/ai-agent/agentic-design-patterns-and-case-studies). This article assumes you've decided to build an agent and want a practical walkthrough of **how** — what to write on day one, what to worry about on day seven, and what three real agents look like from the inside.

We'll do three things:

1. **Build a minimal agent from scratch** — so you see every moving part.
2. **Upgrade it into a production-grade system** — with the non-obvious bits that separate demos from products.
3. **Walk through three case studies** — a code review agent, a customer support agent, and a research agent, designed end to end.

Let's go.

## Part 1: The Minimal Agent

### The Five Lines of Code That Matter

Every agent, stripped to the bone, is this:

```python
while not done:
    response = llm(messages, tools=tools)
    messages.append(response)
    if response.wants_tool:
        result = execute_tool(response.tool_call)
        messages.append(result)
    else:
        done = True
```

That's it. The LLM either picks a tool (the agent acts) or returns a final answer (the agent stops). Everything else in this guide is elaboration on those five lines.

### A Real Minimal Example

Let's build a weather agent that answers "should I bring a jacket?" questions. Using the Anthropic SDK:

```python
import anthropic

client = anthropic.Anthropic()

tools = [{
    "name": "get_weather",
    "description": "Get current weather for a city. Returns temperature in Celsius and conditions.",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name, e.g. 'Tokyo'"}
        },
        "required": ["city"]
    }
}]

def execute_tool(name, inputs):
    if name == "get_weather":
        # Call real weather API here
        return {"temp_c": 12, "conditions": "cloudy"}

def run_agent(user_query: str):
    messages = [{"role": "user", "content": user_query}]

    while True:
        response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=1024,
            tools=tools,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            return response.content[-1].text

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result),
                    })
            messages.append({"role": "user", "content": tool_results})
```

Sixty lines, and you have a real agent. It plans, acts, observes, decides when to stop. Every production agent starts here.

### What You Have (And What You Don't)

You have: a working tool-use loop. You can give it more tools and it will use them.

You don't have: step budgets, error handling, context management, evaluation, safety, observability, streaming, memory, or anything else you'd need to ship. We'll add those.

## Part 2: Upgrading to Production

Here's the list of things I'd add, in roughly the order I'd add them, to turn the minimal agent into something shippable.

### Upgrade 1: A Step Budget

Without a budget, an agent can loop forever. A bug in a tool, an ambiguous user request, or just bad luck can burn through your API quota in minutes.

```python
MAX_STEPS = 25
for step in range(MAX_STEPS):
    response = client.messages.create(...)
    # ... handle response ...
    if done: break
else:
    # Agent didn't finish in budget — fall back or escalate
    return "I couldn't complete this task in the allowed steps. Here's what I learned so far: ..."
```

A good default: **25 steps for most tasks, 50-100 for coding/research agents**. Track step usage in your observability — consistent budget exhaustion means either the task is too hard or your tools are too noisy.

### Upgrade 2: Error Handling at the Tool Layer

Tools fail. APIs time out. JSON fails to parse. The worst thing your tool can do is crash; the second worst is return an uninformative error that the LLM can't act on.

```python
def execute_tool_safely(name, inputs):
    try:
        return {"ok": True, "result": execute_tool(name, inputs)}
    except requests.Timeout:
        return {"ok": False, "error": "The API timed out. Consider retrying or trying a different query."}
    except requests.HTTPError as e:
        if e.response.status_code == 429:
            return {"ok": False, "error": "Rate limit hit. Wait 30 seconds before retrying this tool."}
        return {"ok": False, "error": f"API returned {e.response.status_code}: {e.response.text[:200]}"}
    except Exception as e:
        return {"ok": False, "error": f"Unexpected error: {type(e).__name__}: {str(e)[:200]}"}
```

Good tool errors **tell the LLM what to do next**. "Rate limit hit, wait 30s" is actionable. "HTTP 429" is not.

### Upgrade 3: Context Management

Your context will fill up. Tool results can be huge (API dumps, file contents, search results). If you pass all of it into the next step, quality degrades and costs explode.

Three techniques, in order of sophistication:

**a) Truncate verbose tool outputs.** Cap tool results at, say, 4000 tokens. Add "...[truncated]" so the model knows.

**b) Summarize old turns.** After N turns, replace the oldest turns with a short summary.

```python
if len(messages) > 30:
    old_turns = messages[:20]
    summary = summarize(old_turns)
    messages = [{"role": "user", "content": f"[Earlier conversation summary: {summary}]"}] + messages[20:]
```

**c) Use a scratchpad.** Give the agent a `note_to_self(content)` tool. The agent writes key facts there. Old verbose turns can be dropped because the scratchpad preserves what matters.

### Upgrade 4: A Real System Prompt

The system prompt is where most agent quality lives. A good one answers:

- **Who is the agent?** ("You are a senior customer support agent for Acme Corp's billing system.")
- **What can it do?** ("You can look up invoices, issue refunds up to $100, and escalate to a human.")
- **What can it *not* do?** ("Never discuss competitors. Never promise refunds without first checking the refund policy tool.")
- **How should it reason?** ("Before calling a tool, state in one sentence why you're calling it.")
- **How should it format output?** ("End every user-facing response with 'Is there anything else I can help with?'")

A concrete template:

```
You are {role}. You help {audience} with {task_scope}.

## Capabilities
You have the following tools:
- {tool_1}: use when {when_to_use}
- {tool_2}: use when {when_to_use}

## Constraints
- Do not {forbidden_action_1}
- Always {required_action_1}
- When uncertain about {X}, ask the user rather than guessing

## Style
- {tone}
- {output_format}

## Thinking
Before acting, briefly reason about what you're doing and why.
If a tool fails, consider the error message and decide whether to retry,
try a different approach, or escalate.

If after {N} steps you haven't made progress, stop and explain what you tried.
```

Specificity beats length. A focused 400-token system prompt beats a rambling 2000-token one.

### Upgrade 5: Tool Design as a First-Class Concern

I'll say this louder: **tool design is where most agent quality lives.** A week spent on tools pays more than a month spent on prompts.

What makes a tool good:

**Clear name.** `search_customer_orders` not `search` or `query_db`.

**Clear description.** A full sentence, sometimes two. What it does, when to use it, what it returns.

```python
{
    "name": "search_customer_orders",
    "description": (
        "Search for a customer's orders by customer ID or email. "
        "Returns up to 20 most recent orders with order ID, date, total, and status. "
        "Use this when the user asks about their order history or a specific past order. "
        "If you need full details of a single order, use `get_order_details` instead."
    ),
    ...
}
```

**Narrow, well-typed parameters.** Fewer and stricter is better. Enum types where applicable. Required vs. optional made explicit.

**Structured, concise returns.** Don't dump raw API responses. Shape the output for the LLM:

```python
# Bad: returns 400 lines of JSON
return api_response.json()

# Good: shape it
return {
    "found": len(orders),
    "orders": [
        {"id": o["order_id"], "date": o["created_at"][:10], "total": o["total_usd"], "status": o["status"]}
        for o in orders[:20]
    ]
}
```

**Idempotent where possible.** If the agent retries the same tool call, it shouldn't cause duplicate side effects.

**Unit tests.** Tools are regular code. They can be tested without the LLM.

### Upgrade 6: Observability

You cannot debug what you cannot see. At minimum, log for every agent run:

- The full message trace (every turn, every tool call, every result)
- Step count and total tokens
- Tool call success/failure rates
- Latency per step
- Final outcome (success, failure, timeout, escalation)

Tools like LangSmith, Langfuse, Braintrust, or a plain structured JSON log are all fine. What matters is that six weeks from now, when someone says "the agent gave me a weird answer on Tuesday," you can pull the trace and see exactly what it did.

### Upgrade 7: Evaluation

You need a way to answer "is this agent good?" that isn't vibes.

**Start with a golden set.** 20-50 representative user inputs with either expected outputs or grading criteria. Run the agent against them whenever you change anything.

**Add programmatic checks.** For each task, what does success look like? Did it call the right tool? Did the output validate against a schema? Did it finish within budget? These can be automated.

**Use LLM-as-judge carefully.** For subjective criteria (tone, helpfulness), a separate LLM can score outputs. But you need to *validate the judge* — hand-label 50 examples and check the judge agrees with humans.

**Track regressions.** A dashboard showing pass rate on your golden set over time. Every prompt change, every tool change, gets a number.

### Upgrade 8: Safety Rails for Irreversible Actions

Some actions cannot be undone. Sent emails. Charged credit cards. Deleted files. Posted messages.

Rule: **the agent does not take irreversible actions autonomously.** Options:

- **Human confirmation** — pause and ask the user before the action
- **Two-agent check** — a "supervisor" LLM reviews the proposed action before execution
- **Whitelist** — only specific pre-approved action patterns are allowed
- **Dry-run mode** — the agent produces a plan; a human approves before execution

Which mix you use depends on stakes. A refund up to $10 might be automatic. A refund over $500 might need confirmation. A refund over $5000 might need a human decision.

### Upgrade 9: Streaming and Latency UX

Agents are slow. A five-step task might take 20 seconds. Users will bounce unless you show progress.

Stream everything you can:
- Stream the LLM's thinking tokens as they come
- Announce each tool call when it starts ("Searching your orders...")
- Stream final response tokens
- Show a spinner with specific text ("Checking inventory...") not a generic one

Perceived latency matters more than actual latency. An agent that shows work for 20 seconds feels faster than one that shows a spinner for 10.

### Upgrade 10: Feedback Loops

Once the agent is in production, instrument for improvement:

- Thumbs up/down on responses
- Users marking escalations or corrections
- Logged failure modes
- A review queue for the worst traces

This feedback goes back into your golden set, your system prompt tweaks, and your tool improvements. The best agents are built by teams who iterate on real user data weekly.

## Part 3: Case Study 1 — A Code Review Agent

Let's design a real agent end to end. The task: **review a pull request and leave useful comments.**

### Scoping

First, what does "useful" mean here? Let's be specific:

- Catch bugs, logic errors, and security issues
- Flag style inconsistencies with the team's existing code
- Suggest improvements but not bike-shed
- Not repeat what linters already say
- Comment inline on specific lines

This scoping is half the work. Without it you get an agent that just says "looks good!" or writes 50 nit comments on every PR.

### Tool Set

```
read_pr_metadata(pr_id)
  → Returns title, description, changed files, author, linked issues.

read_file_at_ref(path, ref)
  → Read the file at a specific git ref. Needed to see context around changes.

read_diff(pr_id)
  → Returns the unified diff with line numbers.

search_repo(query)
  → grep-like search across the repo. Finds related code, existing patterns.

run_tests(pattern)
  → Execute the test suite, return pass/fail.

leave_inline_comment(path, line, comment)
  → Post a comment on a specific line.

leave_summary_comment(comment)
  → Post a summary comment on the PR.

finish(verdict)
  → verdict ∈ {"approve", "request_changes", "comment_only"}. Ends the session.
```

Eight tools, narrowly scoped. The agent reads, investigates, decides, acts.

### System Prompt Sketch

```
You are a senior code reviewer for the ${team} team.

Your job: review pull requests and leave comments that save the team time.

## What makes a good review
- Catch real bugs, security issues, or logic errors before they ship.
- Flag inconsistencies with existing patterns in this codebase.
- Suggest improvements only when they clearly matter. Skip bike-shedding.
- Prefer fewer, higher-signal comments over many low-signal ones.

## Process
1. Read the PR description and linked issue.
2. Read the diff. Understand what changed and why.
3. For each non-trivial change, consider: could this break something?
   Is there an existing pattern it should follow?
4. Use search_repo when you need to see how similar code is structured elsewhere.
5. Run the tests if the change is risky.
6. Leave inline comments for specific issues.
7. Leave a summary comment with overall feedback.
8. Finish with your verdict.

## Do not
- Comment on things covered by the linter (style, formatting).
- Leave nits without a clear suggestion.
- Approve a PR with failing tests.
- Leave more than 15 comments unless the PR genuinely needs it.

## Uncertainty
If you're unsure whether something is a bug, ask in a comment rather than asserting.
Preface speculation with "Possible issue:" or "Consider:".
```

### The Agent Loop

The agent reads the PR, investigates, comments, and finishes. A step budget of 40 is reasonable — PRs can need exploration.

Key design choices:

- **Separate read tools from write tools.** The agent reads freely; writes (comments, test runs) are tracked. Makes review traces clean.
- **finish() is explicit.** The agent must decide when it's done. This forces a verdict rather than drifting.
- **No merge power.** The agent *never* merges the PR. Only humans merge. Safety rail.

### What Goes Wrong (And How We Fix It)

**Problem: agent leaves 40 low-signal comments.**
Fix: in the system prompt, add "before leaving each comment, ask yourself: would a senior engineer thank me for this?" Plus a hard cap of 15 comments per PR.

**Problem: agent doesn't understand the change in context.**
Fix: add a `read_related_files(path)` tool that auto-fetches files imported by the changed file. Most bugs come from context the diff doesn't show.

**Problem: agent takes 5 minutes on simple PRs.**
Fix: a trivial-diff short-circuit. If `read_diff` returns under N lines, use a lightweight prompt that just checks for obvious issues and finishes.

**Problem: agent hallucinates symbols that don't exist.**
Fix: force it to use `search_repo` before asserting any symbol name. Reject comments that reference symbols not found in search.

### Evaluation

Golden set: 50 historical PRs with known human reviews. Grade by:

- **Precision** — what fraction of agent comments were also raised by the human reviewer (or are clearly valid)?
- **Recall** — what fraction of human-flagged issues did the agent also catch?
- **Verdict agreement** — does the agent's approve/request-changes match the human?

Iterate on prompt and tools until precision > 0.7 and recall > 0.5. Those are realistic targets for a first-pass reviewer.

## Part 4: Case Study 2 — A Customer Support Agent

The task: **handle tier-1 customer support for a SaaS product.**

### Scoping

A support agent lives or dies on scope. Let's decide:

**In scope:**
- Password resets, account lookups
- Explaining features
- Troubleshooting common issues
- Issuing refunds up to $50
- Escalating anything else to a human

**Out of scope:**
- Sales questions (route to sales)
- Legal or compliance questions (route to legal)
- Anything involving more than one account (route to human)

This narrow scope is what makes the agent reliable.

### Tool Set

```
lookup_user(email)
  → Returns user profile, subscription status, last login.

get_subscription(user_id)
  → Plan, billing status, renewal date.

reset_password(user_id)
  → Triggers a reset email.

search_knowledge_base(query)
  → Returns up to 3 relevant help articles.

issue_refund(user_id, amount, reason)
  → Refund up to $50 directly. Larger amounts fail with "requires human approval".

escalate_to_human(reason, context_summary)
  → Opens a ticket, hands off, ends the conversation.

end_conversation(summary)
  → User is satisfied; summary is logged.
```

### System Prompt Sketch

```
You are a support agent for ${product}. You help users with account and billing issues.

## Your approach
1. Understand the user's issue. Ask one clarifying question if needed.
2. Check the knowledge base for a relevant article. Most questions are answered there.
3. If it's an account action (password reset, refund, etc.), verify the user's identity
   by looking them up. Then take the action.
4. If the issue is outside your scope or you're stuck, escalate to a human.
5. End the conversation when the user confirms the issue is resolved.

## Identity verification
Before taking any account action, confirm the user's email matches a record
in `lookup_user`. Do not take actions on behalf of anonymous users.

## Refunds
You can issue refunds up to $50 for clear issues (duplicate charge,
failed delivery, documented bug affecting their account).
For anything above $50 or ambiguous, escalate.

## Tone
Friendly, concise, solution-first. Never defensive, never robotic.
If you're taking an action, tell the user what you're doing ("Let me look
that up for you") before doing it.

## Escalation triggers (escalate immediately)
- User is frustrated or asks for a human
- Legal, compliance, or security question
- Billing issue over $50
- Anything involving multiple accounts or data export
- Two failed attempts to resolve the same issue
```

### Design Decisions Worth Calling Out

**Identity verification as a prompt rule.** You could build it into tools (e.g., every tool requires proof of identity). But making it a prompt rule that applies to a category of actions is cheaper and often more flexible. The system prompt makes it a soft invariant.

**A hard cap on refund amount, enforced in the tool.** Don't trust the LLM with money. The `issue_refund` tool rejects amounts over $50 regardless of what the prompt says.

**Explicit escalation triggers.** An agent that escalates too much is annoying but safe. An agent that never escalates is dangerous. Listing triggers explicitly shifts the balance toward safe.

**end_conversation as a tool.** Makes "success" a first-class event you can log and measure.

### Observability and Iteration

Key metrics:
- **Resolution rate** — fraction of conversations where `end_conversation` fired with a satisfied user
- **Escalation rate** — target range: 15-30%. Lower means the agent is over-reaching; higher means it's too timid.
- **Average turns per conversation** — long conversations indicate friction
- **User thumbs-up on agent responses** — direct quality signal

The first weeks of deployment are almost entirely about tuning the prompt and the KB content based on real traces. Expect to make 50+ small changes before it feels solid.

## Part 5: Case Study 3 — A Research Agent

The task: **given a research question, produce a well-cited briefing.** Think mini-Perplexity for internal knowledge plus the web.

### Why This Is Different

Unlike the previous two, this task has natural parallel structure. A question like *"compare the three leading approaches to X"* decomposes into three roughly independent investigations. This is where the **orchestrator-workers** pattern earns its keep.

### Architecture

```
                ┌─────────────┐
  User query →  │   Planner   │ → decomposes into sub-questions
                └─────┬───────┘
                      │
         ┌────────────┼────────────┐
         │            │            │
    ┌────▼────┐  ┌────▼────┐  ┌────▼────┐
    │ Worker  │  │ Worker  │  │ Worker  │   parallel search agents
    │   A     │  │   B     │  │   C     │
    └────┬────┘  └────┬────┘  └────┬────┘
         │            │            │
         └────────────┼────────────┘
                      │
                ┌─────▼───────┐
                │ Synthesizer │ → writes briefing with citations
                └─────────────┘
```

### The Planner

A single LLM call that takes the user's question and returns a list of 3-7 focused sub-questions. System prompt emphasizes:

- Each sub-question should be independently researchable
- Together they should cover the user's intent
- Avoid overlap (each sub-question should have a distinct angle)

Output format: structured JSON with sub-questions and why each one is relevant.

### The Workers

Each worker is a small ReAct agent with:

- A `web_search(query)` tool
- A `fetch_page(url)` tool
- An `internal_docs_search(query)` tool (for a local corpus if applicable)
- A `take_notes(content, source_url)` tool that appends to a shared scratchpad

The worker explores for up to 10 steps, takes notes with citations, and finishes. Critically, the worker's context is **isolated** — it only sees its own sub-question and its own notes, not the other workers' work.

### The Synthesizer

Takes the notes from all workers plus the original user question and produces the final briefing. System prompt:

- Use only information in the notes; do not invent facts
- Every substantive claim must have a citation pointing back to a note
- Structure the response by theme, not by worker
- Flag contradictions between sources explicitly
- If evidence is thin, say so rather than speculating

### The Non-Obvious Design Decisions

**Isolation of worker contexts.** Workers don't share state. This keeps each context clean and prevents one worker's errors from corrupting others. The synthesizer sees only the clean notes.

**Notes, not conversation.** Workers don't hand back transcripts. They hand back a structured list of `{claim, source_url, quote}` notes. The synthesizer works with structured data.

**Bounded worker time.** Each worker has a strict step budget. An infinite research rabbit hole is a failure mode; we'd rather return "incomplete" than run forever.

**Explicit contradiction surfacing.** Research tasks often produce conflicting sources. Hiding that is worse than surfacing it. The synthesizer's prompt emphasizes disagreement as information.

### Cost and Quality

Multi-agent systems are expensive. A research agent like this uses ~10-20× the tokens of a single-agent approach. That cost is only justified if:

- The task genuinely benefits from parallel exploration
- You care about breadth
- The user is willing to pay for depth

For simpler research tasks ("when was X founded?"), a single-agent ReAct loop with a web search tool is cheaper and just as good. Not every research question needs the full orchestra.

### Evaluation

Evaluation here is hard. Some ideas that work in practice:

- **Ground-truth questions.** A test set where you know the answer (e.g., "what are the three founders of X?") and can grade exact match.
- **Citation checking.** For every claim in the output, verify the cited source actually contains that claim. A separate LLM pass can do this reliably.
- **Coverage checking.** Given a rubric of what a good answer should cover, an LLM grader checks which points are addressed.
- **Human spot-checks.** Weekly review of 20 random outputs by a human rater.

Don't skip this. A research agent that hallucinates confidently is worse than no research agent.

## Part 6: A Builder's Checklist

You're about to ship an agent. Before you do:

### Scoping
- [ ] Can you describe in one sentence what the agent does?
- [ ] Can you describe in one sentence what it does *not* do?
- [ ] Do you have a list of escalation or hand-off triggers?

### Tools
- [ ] Each tool has a clear name and a full-sentence description
- [ ] Tool parameters are narrow and strictly typed
- [ ] Tool outputs are shaped for the LLM (structured, concise)
- [ ] Errors return actionable messages, not stack traces
- [ ] Destructive actions have hard limits built into the tool
- [ ] Each tool has unit tests

### Prompt
- [ ] System prompt defines role, capabilities, constraints, style
- [ ] Uncertain cases have an explicit fallback ("ask", "escalate")
- [ ] Output format is specified
- [ ] Prompt is under 500 tokens for simple agents, under 2000 for complex

### Orchestration
- [ ] Step budget is set
- [ ] Context management strategy is chosen (truncate / summarize / scratchpad)
- [ ] Final-answer / done detection is explicit
- [ ] Loop exits cleanly on budget, error, or user cancel

### Safety
- [ ] Irreversible actions require confirmation or human approval
- [ ] Hard limits on money / bulk actions enforced in tool code
- [ ] User input is not directly concatenated into tool arguments without validation
- [ ] Prompt injection is considered for any tool that reads external content

### Observability
- [ ] Every run produces a traceable log (messages, tool calls, outcomes)
- [ ] Step count, token count, latency are recorded
- [ ] Outcome (success / failure / escalation) is tagged

### Evaluation
- [ ] Golden set of 20-50 representative inputs
- [ ] Automated pass/fail criteria where possible
- [ ] LLM judge validated against human labels (if used)
- [ ] Regression run on every change

### Production
- [ ] Streaming or progress indicators for tasks over 5 seconds
- [ ] User-visible error messages (not raw exceptions)
- [ ] Feedback capture (thumbs up/down, flag button)
- [ ] Review queue for bad traces
- [ ] Alerting on step-budget exhaustion, tool failure rate spikes

If you check every box, you have a real production agent — not a demo. Most teams get here by iterating for a few weeks on whichever boxes they left unchecked. That iteration is the work.

## Closing

Building a good agent is less about picking the right architecture and more about doing the unglamorous work well:

- Scoping the task narrowly
- Designing tools with care
- Writing prompts that are specific
- Adding safety rails where they matter
- Measuring outputs honestly
- Iterating on real data

The three case studies — code review, customer support, research — all follow the same arc: decide what success looks like, build a minimal loop, identify what goes wrong, fix it, measure, repeat. No magical prompt unlocks any of them. Steady engineering does.

Start minimal. Ship something small. Watch it fail. Fix the failures. That loop, run for a few weeks, produces agents that work.

## Further Reading

Related articles:
- [Agentic Design Patterns and Case Studies](/blog/machine-learning/ai-agent/agentic-design-patterns-and-case-studies)
- [Effective Context Engineering for AI Agents](/blog/machine-learning/ai-agent/effective-context-engineering-for-ai-agents)
- [Advanced Tool Use](/blog/machine-learning/ai-agent/advance-tool-use)
- [Eval Agents](/blog/machine-learning/ai-agent/eval-agents)
- [Model Context Protocol](/blog/machine-learning/ai-agent/model-context-protocol)
- [Claude Code Tips and Best Practices](/blog/machine-learning/ai-agent/claude-code-tips-and-best-practices)

External references:
- Anthropic — *Building Effective Agents*
- Anthropic — *How We Built Our Multi-Agent Research System*
- OpenAI — *A Practical Guide to Building Agents*
- Lilian Weng — *LLM-Powered Autonomous Agents*
- Cognition AI — *Don't Build Multi-Agents* (the opposing view — worth reading)

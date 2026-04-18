---
title: "Designing Agents for Long-Running, Complex Tasks — Part 1: Architecture and Foundations"
publishDate: "2026-04-18"
category: "machine-learning"
subcategory: "AI Agent"
tags:
  [
    "ai-agent",
    "long-running-agents",
    "architecture",
    "planning",
    "task-decomposition",
    "state-management",
    "interview",
    "system-design",
    "llm",
  ]
date: "2026-04-18"
author: "Hiep Tran"
featured: false
aiGenerated: true
excerpt: "Short agent demos are easy. Agents that run for hours, hand off between subtasks, recover from crashes, and produce a correct answer at the end are where the real design work lives. This article walks through every architectural decision you face — task decomposition, planning patterns (ReAct, Plan-and-Execute, Tree of Thoughts, Reflexion), state and context management, sub-agent patterns, and checkpointing — with a dedicated interview Q&A section covering the questions that actually come up."
---

## Why Long-Running Agents Are a Different Design Problem

An agent that answers a single question is a function. An agent that spends three hours refactoring a codebase, triaging a customer's case across five tools, or researching a topic across fifty sources is a **system**.

The difference isn't the model. It's everything around the model:

- The task has to be **decomposed** into steps the model can actually reason about.
- State must survive **beyond a single context window**.
- Failures are normal, not exceptional; **recovery** has to be designed in.
- Tools behave unpredictably; **reliability** becomes a first-class concern.
- Evaluating "did it work?" is no longer obvious.

This article is the architectural half of the design problem: how to decompose, plan, represent state, and structure control flow. The [companion article](/blog/machine-learning/ai-agent/designing-long-running-agents-reliability-production) covers reliability, execution, observability, and production concerns.

Both articles include an interview Q&A section at the end. If you're preparing for a staff-level ML or agent-systems interview, those are written to match what actually gets asked.

## Part 1: A Taxonomy of "Long-Running" and "Complex"

The words "long-running" and "complex" hide different problems. Get clear about which one you're solving before you architect anything.

### Axis 1: Duration

| Duration | Design implication |
| --- | --- |
| Seconds (single turn) | One context, one forward pass. Classic chat. |
| Minutes (a handful of tool calls) | Context mostly fits. Retry is cheap. |
| Hours (dozens–hundreds of steps) | State must live outside context. Recovery matters. |
| Days+ (async, durable workflows) | Durable queues, checkpoints, partial results, human checkpoints. |

### Axis 2: Complexity

| Complexity | Example | Design implication |
| --- | --- | --- |
| Linear | "Summarize this paper" | Straight-through pipeline. |
| Branching | "Find the cheapest flight" | Decisions at each step, may backtrack. |
| Graph | "Plan and book a 10-day trip" | Interdependent subtasks, parallelism. |
| Open-ended | "Investigate this bug in the codebase" | Unknown branching factor, unknown termination. |

### Axis 3: Interactivity

| Interactivity | Design implication |
| --- | --- |
| Autonomous end-to-end | No user in loop. Needs strong self-verification. |
| Checkpoint-interactive | User approves milestones. Needs handoff UX. |
| Turn-interactive | User converses throughout. Needs session memory. |

A five-minute autonomous task and a five-hour checkpoint-interactive task are **different systems**, not the same system with a bigger timeout.

## Part 2: The Core Agent Loop — and Where It Breaks

Every agent starts from this loop:

```python
while not done:
    thought = llm(context)
    if thought.is_action:
        result = execute_tool(thought.action)
        context.append(thought, result)
    else:
        final_answer = thought.text
        done = True
```

This five-line pattern handles the 10-step case beautifully. It collapses around step 50 for specific, predictable reasons:

### Breakage 1: Context Overflows

Each step appends `thought + result` to context. After 50 steps, your context has the original task, 50 reasoning chunks, and 50 tool outputs — easily tens of thousands of tokens. You hit the context limit, or worse, you hit the **useful attention limit** (models attend poorly to the middle of long contexts) long before the hard limit.

### Breakage 2: Drift

With every step, the model re-reads the accumulated history and re-interprets the task slightly. By step 30, many agents have quietly changed what they think they're doing. Without anchoring mechanisms, the loop drifts.

### Breakage 3: Looping

The model takes action A. A fails or gives no new info. The model re-reasons, decides on A again. Loop. Classic failure mode, and the naive loop has no defense against it.

### Breakage 4: Loss on Crash

Anything that crashes — a timeout, a tool failure that raises, an OOM — loses the full context. No retry is possible without re-running everything.

### Breakage 5: No Parallelism

The loop is strictly sequential. For tasks with independent subtasks, that leaves a lot of latency on the table.

**These five failure modes drive essentially every architectural decision below.** Each pattern in this article is a fix for one or more of them.

## Part 3: Task Decomposition

The first real design choice: how do you break the task into pieces the model can handle one at a time?

### Decomposition Patterns

**Fixed pipeline (hand-written).**
Divide the task into known stages, code each stage as its own LLM call. Great when the steps are predictable — RAG pipelines, classification-then-generation flows.

```
load_data → extract_entities → enrich → summarize → format
```

Cheap, predictable, easy to debug. Zero flexibility.

**LLM-driven decomposition (Plan-and-Execute).**
Ask the model to produce a plan (list of subtasks), then execute each subtask. The planner is a separate call from the executors. More on this in Part 4.

**Recursive decomposition.**
Each subtask can itself be decomposed if it's too complex. The agent recursively expands until tasks become small enough to solve directly. Powerful but can explode in depth.

**Dynamic decomposition (ReAct-style).**
No upfront plan. The model decides the next step based on the current state. More flexible, harder to control.

### Choosing a Pattern

| Task property | Good pattern |
| --- | --- |
| Known stages, stable | Fixed pipeline |
| Clear but varied plan per task | Plan-and-Execute |
| Highly exploratory, unknown shape | ReAct |
| Deeply hierarchical (planning a project) | Recursive decomposition |
| Needs parallelism | Plan-and-Execute (with DAG plans) |

The mistake most teams make is picking the most dynamic pattern ("the model will figure it out") when their task is actually structured. Structure is cheaper to engineer than dynamism. Use the most static decomposition your task admits.

### Decision Framework: A Worked Example

Three concrete tasks, three different answers — and the *trade-off* you accept in each:

**(a) "Summarize this PDF and extract action items" → Fixed pipeline.**
Task shape: predictable stages (load → parse → chunk → summarize → extract → format). The signals: zero exploration, no branching, output shape is known. Pattern: a hand-coded chain of five LLM calls, each with a narrow prompt. Trade-off accepted: *zero flexibility*. If the PDF is a scanned contract instead of a report, the pipeline produces nonsense. You pay for that by adding a classifier upfront and routing to a different pipeline — not by making one pipeline smarter.

**(b) "Plan a 2-week Japan trip with a $4k budget" → Plan-and-Execute with a DAG.**
Task shape: 15–30 subtasks (flights, hotels, JR Pass, day-by-day itineraries), many parallelizable, strong interdependence (Kyoto hotel dates depend on Tokyo arrival). The signals: benefits from a global plan, user wants to approve before booking, parallelism pays off (search flights and hotels concurrently). Pattern: a planner produces a dependency graph, executors run branches in parallel, a replanner handles "Kyoto hotel sold out." Trade-off accepted: *up-front planning latency and cost* (one expensive planner call before any work). You pay for that by letting each subtask run cheap; without planning, you'd drift and book flights that don't fit hotel dates.

**(c) "Debug why checkout latency spiked at 3am" → ReAct.**
Task shape: unknown branching — could be DB, cache, deploy, upstream, cert expiry, anything. The signals: path isn't predictable, <20 steps in most cases, user wants to watch the agent reason. Pattern: single ReAct loop with observability tools (metrics, logs, traces, recent deploys). Trade-off accepted: *drift risk and loop risk* in exchange for flexibility. You pay for that by adding loop detection and a hard 15-step budget — not by trying to pre-plan an incident the agent has never seen.

The meta-lesson: *the pattern you pick encodes what you're willing to pay for*. Predictability costs flexibility. Planning costs upfront latency. Dynamism costs drift. Name the cost before you name the pattern.

### Decomposition Anti-Patterns

- **"Let the agent figure out how to split the work."** Works sometimes. Fails in weird, hard-to-debug ways the rest of the time.
- **Steps too granular.** If each subtask is "extract one field," you spend more on orchestration than work.
- **Steps too coarse.** If "build the feature" is a single step, the LLM is doing everything inside one ungoverned context.
- **No contract between steps.** If step 2 doesn't know what shape of output to expect from step 1, the joint fails.

## Part 4: Planning Architectures — The Named Patterns

These are the canonical patterns you will be asked about. Knowing when each shines is more important than knowing their names.

### ReAct (Reason + Act)

The classic:

```
Thought: I need to find the user's recent orders.
Action: search_orders(user_id=123)
Observation: [3 orders returned]
Thought: The most recent is from March. Let me check its status.
Action: get_order_status(order_id=...)
Observation: delivered
Thought: I have enough to answer.
Final Answer: Your last order was delivered in March.
```

**Strengths:** simple, flexible, works for unstructured tasks, one model decides everything.

**Weaknesses:** weak for long plans (no global view), prone to loops, prone to drift, context bloats with `Thought + Observation` pairs.

**Use when:** the path isn't predictable, total steps are modest (<20), the task is mostly "look stuff up until I can answer."

### Plan-and-Execute

Two separate components:

```
Planner LLM:  given a task, produce a plan (ordered steps with dependencies)
Executor:     run each step (which may itself use an LLM or a tool)
Replanner:    after each step, revise the plan if needed
```

**Strengths:** plan is inspectable (and approvable by humans), steps can run in parallel, executor can be smaller/cheaper than planner.

**Weaknesses:** planner has to be good at planning (non-trivial for complex tasks), replanning is its own loop.

**Use when:** tasks are long (20+ steps), benefit from global planning, or need human checkpoints.

### Tree / Graph of Thoughts

Instead of one reasoning trace, the agent explores **multiple branches** of reasoning — like a search tree. A scoring mechanism prunes branches. Good for problems with non-obvious solutions where one reasoning path may fail.

**Use when:** the task benefits from search (puzzles, complex coding, difficult reasoning) and you can afford the cost. It is expensive: branching explodes token counts.

### Reflexion

After an attempt, the agent **reflects** on what went wrong and incorporates the reflection into the next attempt. Essentially self-feedback. Strong for iterative refinement (code generation with test feedback, proof attempts).

**Use when:** you have a **verifier** — something that tells the agent objectively that the attempt is wrong. Without a verifier, reflection becomes hallucinated self-criticism.

### Hierarchical / Orchestrator–Worker

A top-level orchestrator agent decomposes and delegates; worker agents handle specific subtasks. The orchestrator collates results.

**Use when:** task is naturally hierarchical (research project → per-topic investigations → per-source analysis) and you can provide each worker with a self-contained context.

### A Quick Comparison

| Pattern | Parallelism | Inspectability | Recovery | Token cost | When it shines |
| --- | --- | --- | --- | --- | --- |
| ReAct | low | medium | weak | low-medium | unpredictable, short |
| Plan-and-Execute | high | high | strong | medium | long, structured |
| Tree of Thoughts | n/a | medium | weak | high | hard reasoning |
| Reflexion | none | medium | strong (with verifier) | high | iterative w/ verifier |
| Orchestrator–Worker | high | high | per-worker | medium | hierarchical |

### Combining Patterns — Real Systems Are Hybrids

Production agents rarely run a pure pattern. Two hybrids worth knowing:

**Plan-and-Execute with ReAct executors.** The planner produces a high-level subtask list. Each subtask then runs a *small* ReAct loop inside its own bounded context. The planner brings structure; ReAct inside each subtask brings flexibility where the shape of the work genuinely varies. Use when: subtasks are named ("collect pricing data from vendor A") but the inside of each subtask is exploratory. Trade-off: you pay for two planning layers — one global, one per subtask — and you must bound each ReAct loop hard (≤10 steps, specific termination) or the subtask drifts and replanning cascades.

**Orchestrator–Worker with Reflexion per worker.** The orchestrator dispatches; each worker runs its task, checks its own output against a verifier (tests, schema validator, reranker), and retries once or twice if it fails. The orchestrator sees only the *final* verified output. Use when: workers do generative work with a machine-checkable verifier (code generation, data extraction, schema-bound structured output). Trade-off: worker token cost roughly 2× (two attempts on average), but orchestrator's final-stage verification gets much simpler because each worker already self-cleaned.

The composition is only worth its complexity when each pattern solves a *different* problem. Plan-and-Execute solves global coherence; ReAct solves local exploration; Reflexion solves local correctness. If you're adding a pattern and it solves the same problem as one you already have, you're just adding coordination cost.

Anti-pattern: stacking three patterns "because quality should go up." Each layer is a new drift/failure surface. Default to the simplest single pattern that fits, promote to hybrid only after a specific failure mode justifies the extra layer.

## Part 5: State and Memory for Long Horizons

A five-step agent holds state in the context. A 500-step agent cannot. State design is the biggest architectural lever for long-running agents.

### The Three State Layers

**1. Working state (volatile).**
The current reasoning trace. Lives in the context. Reset between steps.

**2. Task state (structured, durable).**
A structured object — JSON, a database row — tracking progress on the current task:

```json
{
  "task_id": "T-42",
  "goal": "Refactor auth module to support SSO",
  "subtasks": [
    {"id": 1, "name": "catalog current auth surface", "status": "done", "artifact": "..."},
    {"id": 2, "name": "design SSO integration",       "status": "in_progress"},
    {"id": 3, "name": "implement and test",           "status": "pending"}
  ],
  "artifacts": {...},
  "notes": [...]
}
```

The agent **reads and writes** this object across steps. It's the durable source of truth — not the context.

**3. Long-term memory (cross-task).**
Things the agent learned that might matter for future tasks. Covered in detail in [the companion article on context engineering](/blog/machine-learning/ai-agent/effective-context-engineering-for-ai-agents).

### Storage Trade-offs: Where Does Each Layer Live?

The three layers have fundamentally different access patterns; pushing them into one store is the single most common state-design mistake.

| Layer | Good default | Why | What breaks it |
| --- | --- | --- | --- |
| Working state | In-memory (process) or Redis | Sub-ms reads, gone on crash is fine | Running across multiple workers — then needs Redis |
| Task state | Postgres (JSON column) or Firestore | Durable, transactional, queryable by task_id, schema-evolvable | High write throughput per task — then shard or add Redis caching layer |
| Scratchpad | Append-only log (Postgres table, S3, or Loki) | Write-heavy, rarely re-read wholesale, debugging-first | Treating as queryable state — it isn't |
| Artifacts | Object store (S3, GCS) | Large, immutable, cheap per GB, referenced by URL | Small items (<4KB) — round-trip latency dominates; inline them |
| Long-term memory | Vector DB + Postgres for metadata | Semantic search across tasks | Using vector DB alone — you lose authoritative metadata |

Quick-decision heuristic: **if it's less than 4KB, transactional, and read every step, put it in the task-state row. If it's large, immutable, and read occasionally, put it in object storage with a reference. If it's searched by semantic similarity, put it in a vector DB with a pointer back to the authoritative source.**

Trade-offs to name out loud:
- **One store for all layers (e.g., "just use Postgres").** Simple until you need semantic search over scratchpad content, at which point retrofit costs dominate.
- **Vector-DB-for-everything.** Pricing model assumes occasional retrieval; hammering it every step is expensive and slow.
- **In-memory only.** Fine for prototypes, indefensible for anything that must survive a process crash. Most production agents cross this line earlier than teams expect.

### The Scratchpad Pattern

Most long-running agents benefit from an explicit scratchpad — a write-only append-only log the agent uses to externalize thought:

```
[step 12] Checking if the `auth_config` table exists...
[step 13] It does. Schema: id, provider, client_id, secret, enabled.
[step 14] Found a bug in row 4: secret is null. Flagging for review.
```

The scratchpad persists independently of context. When context is pruned or rebuilt, the scratchpad can be reread selectively. It also serves as a debugging artifact for humans.

### The Artifact Store

Many long-running tasks produce intermediate outputs too large for context — code files, extracted documents, intermediate analysis. These go into an **artifact store**, and the agent holds **references** in context, not contents:

```
context (small): "See artifact ref://session/123/auth_catalog.md"
store (large):   <actual 12KB catalog>
```

This is the same principle behind RAG. It applies more broadly: **anything large and reread rarely should be a reference, not content.**

### Structured Output As State Update

Instead of letting the model narrate what it just did, force it to emit a **structured update** to task state:

```python
response = llm(
    context,
    output_schema={
        "subtask_progress": {"id": int, "status": str, "notes": str},
        "next_action": {"tool": str, "args": dict},
    }
)
task_state.update(response["subtask_progress"])
```

Structured output dramatically reduces state drift. It's also the foundation for making state durable — free-form narration is much harder to persist cleanly.

## Part 6: Context Engineering for Long Horizons

For agents running 100+ steps, context is not "just append everything." You need an explicit strategy — what stays, what goes, what gets summarized.

### The Context Layout

```
[SYSTEM PROMPT]           — stable, cached
[PERSONA / ROLE]          — stable, cached
[TOOL DEFINITIONS]        — stable, cached
[TASK STATE SUMMARY]      — rebuilt each step from structured state
[RELEVANT MEMORY / ARTIFACTS]  — retrieved per step
[COMPRESSED HISTORY]      — summarized rolled-up older steps
[RECENT WINDOW]           — last N raw steps in full
[CURRENT STEP REQUEST]    — the thing to decide next
```

Every section has a different write policy. The stable sections rarely change (and hit the prompt cache). The dynamic sections are rebuilt every step.

### Compaction

When history exceeds a threshold, compact the oldest portion into a summary. The summary has to preserve what matters for future steps; the details become artifacts. Compaction is a lossy operation — do it deliberately, not by accident.

```
Before: [50 raw steps, ~40k tokens]
After:  [summary of steps 1–35] + [raw steps 36–50]
```

### The Recency vs. Importance Tradeoff

Pure-recency (keep the last N) loses the task goal quickly. Pure-importance (keep what seems relevant) is hard to automate well. The practical answer: **always keep stable anchors** (goal, current subtask, recent actions) and *supplement* with retrieval for history.

### Prompt Caching for Long Agents

Long-running agents are where prompt caching pays most. The stable prefix — system prompt + tool defs + goal — barely changes across hundreds of steps. Every cached step saves a real chunk of cost and latency. Structure your context so the top is maximally stable.

## Part 7: Sub-Agent and Multi-Agent Patterns

When one agent isn't enough.

### When To Add a Sub-Agent

Specific signals:

- A subtask has a **clean self-contained input and output**.
- That subtask is either **very different in nature** from the parent's work or **very verbose** in its intermediate steps.
- You want to bound the parent's context; sub-agent runs in its own context and returns only a summary.

Counter-signals (do NOT split):

- The subtask is inherently woven into the parent's reasoning.
- The subtask returns a large, complex state the parent needs full access to.
- You're adding agents because "more agents sounds more powerful." Every interface between agents is a new failure surface.

### Orchestrator–Worker in Detail

```
Orchestrator:
   plans the overall task
   dispatches subtasks to workers
   collates results
   decides when to re-dispatch or finalize

Workers:
   receive a self-contained task
   run to completion
   return a structured result (not a transcript)
```

Key design rules:

- **Workers return structured results, not conversations.** The orchestrator shouldn't have to read a worker's whole reasoning trace.
- **Workers share nothing except via the orchestrator.** No global state.
- **Dispatch is explicit.** The orchestrator knows which worker is doing what; debugging is possible.

### Peer Agents Negotiating

Occasionally you want multiple agents discussing — debate, critique, peer review. This is powerful for quality, expensive in tokens. Use sparingly: when a single pass has a hard time being reliably correct and the stakes justify the cost.

## Part 8: Checkpointing and Resumption

Any agent running longer than a few minutes will, eventually, be interrupted. The question is whether that interruption loses work.

### What To Checkpoint

At minimum:

- Task state (structured)
- Artifacts
- Scratchpad
- Enough of the recent context to resume reasoning

What *not* to checkpoint:

- The full raw context. It's rebuilt from the above.
- Transient tool call attempts.
- The reasoning LLM's internal state. You don't have access anyway.

### Write-Ahead Logging for Agents

Before taking an action, write the intent:

```
[log] about to call send_email(to="alice@…", body="…")
→ call the tool
[log] call returned: success, message id 42
```

On crash recovery:

- If the "about to call" line is present but no result line: the action may or may not have completed. Check the tool's idempotency and either retry safely or escalate.
- If both lines are present: safe to skip and continue.
- If neither: replay from last checkpoint.

### WAL Failure Scenarios — Three Real Cases

The reason WAL is non-trivial is that the *interesting* crashes happen between the log line and the action, and every one has a correct recovery that depends on what the tool actually does.

**Case 1: Crash between "about to send_email" and the network call.**
WAL shows intent; no result. What actually happened: the email was *not* sent (no network request left the host). Correct recovery: retry with same idempotency key. Trade-off if you escalate instead: you've made a reliable send look like a human-intervention case — unnecessary load on ops. Trade-off if you assume "succeeded" and skip: user never gets the email; silent failure.

**Case 2: Crash after the request left but before the response was recorded.**
WAL shows intent; no result. What actually happened: *unknown* — the email might be sent, or might have failed mid-flight. Correct recovery: *only* retry if the tool is idempotent; otherwise perform an existence check ("has message X been sent in the last N minutes?") before retrying. Trade-off if you retry blindly: duplicate emails, double-charges, angry users. Trade-off if you always escalate: operational toil on a case that's usually safe with idempotency keys.

**Case 3: Log line and result both present, crash during post-processing.**
WAL is complete. What happened: action succeeded; the agent's in-memory update to task state was lost. Correct recovery: skip the action, re-derive task-state update from the result. Trade-off if you retry the action: guaranteed duplicate — the tool saw no new key. Trade-off if you restart from the previous checkpoint without reading the WAL: duplicate action again.

The general principle: WAL turns "unknown state after crash" into "one of N enumerable states, each with a known recovery." Without WAL, you only have the first option. Without idempotency, even WAL can't save you from Case 2 cleanly.

### Idempotency Everywhere

For any action with a side effect, design idempotency in from day one. Either:

- The tool supports idempotency keys natively.
- You wrap the tool in a layer that dedupes by (task_id, step_id).

Without idempotency, crash-recovery is a minefield. Users end up receiving duplicate emails, getting double-charged, or seeing rows inserted twice.

### Resumption Is a First-Class UX

If your agent runs for hours, users will close their laptop. A good system:

- Shows the user that their task is still running (or paused).
- Sends a notification when it completes.
- Allows them to re-open and see progress, intermediate artifacts, and past steps.
- Lets them correct the agent mid-run if it's going wrong.

The difference between "novelty demo" and "actually useful long-running agent" is mostly this UX layer.

## Part 9: Termination — Knowing When to Stop

The silent killer of long-running agents is **not knowing when to stop**. They loop, they over-polish, they add features you didn't ask for.

### Termination Signals

Any combination of:

- **Goal-met predicate.** An explicit check ("do we have a final answer?"). Works when success is machine-checkable.
- **Subtask list exhausted** (for Plan-and-Execute).
- **Budget limits** — max steps, max tokens, max wall-clock time, max cost. Hard limits that trigger graceful stop.
- **No-progress detection.** If the last K steps haven't moved the task state forward, stop.
- **Human checkpoint.** The agent asks for approval at milestones.

### Budgets Are Non-Negotiable

Every long-running agent should have, as a hard system-level limit:

```
max_steps = 200
max_wall_time = 30 minutes
max_cost = $5
max_same_action_repetitions = 3
```

These are not polite suggestions. They are guardrails that prevent the worst outcomes (runaway cost, infinite loops, user frustration).

### Choosing Budgets: A Cost/Success Frontier

Budgets are *not* set by intuition. They come from measuring the distribution of successful runs on a golden eval set and then adding headroom.

**Customer support agent.** Successful resolutions on your eval suite take 5–20 steps at the 50th–95th percentile. Set `max_steps = 50` (2.5× the 95th percentile), `max_cost = $0.50`, `max_wall_time = 2 min`. Trade-off of tighter budget (`max_steps = 25`): ~5% of legitimate long conversations get cut off and route to human — sometimes good (faster escalation), sometimes bad (user frustration). Trade-off of looser budget (`max_steps = 200`): the 1% of stuck runs cost 10× and take 10 minutes before timing out.

**Research assistant.** Successful reports take 100–400 steps across 50–100 sub-questions. Set `max_steps = 500`, `max_cost = $20`, `max_wall_time = 30 min`. Tighter: you cut off the deep-dive cases that are the feature's main value. Looser: one misbehaving run eats a full day of cost budget; you also lose the "I'll come back when it's ready" UX because SLA vanishes.

**Autonomous coding agent.** Well-scoped tasks finish in 20–150 steps. Set `max_steps = 200`, `max_cost = $5`, `max_wall_time = 15 min`. The interesting trade-off: coding agents benefit disproportionately from *higher* step budgets but suffer disproportionately from cost overruns (dev-loop tasks can loop on a failing test). Pair the step budget with a *no-progress* budget (abort if 10 steps pass with no new file touched or test green) — that's a sharper knife than raw step count.

The general principle: set budgets at the 95th percentile of your *successful* distribution, times a small safety factor (2–3×). Not at your worst case — that's how agents run for an hour on tasks they should have bailed on in ten minutes.

## Part 10: Putting It All Together — Reference Architecture

A long-running agent that actually holds up in production looks roughly like this:

```
                  ┌──────────────────────────────┐
                  │      TASK INTAKE             │
                  │  (goal, constraints,         │
                  │   human checkpoints)         │
                  └────────────┬─────────────────┘
                               │
                  ┌────────────▼─────────────────┐
                  │        PLANNER               │
                  │  (produces plan or           │
                  │   initial subtask list)      │
                  └────────────┬─────────────────┘
                               │
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
        Subtask 1          Subtask 2         Subtask 3
        (worker agent)     (worker agent)    (worker agent)
             │                 │                 │
             └────────┬────────┴─────────────────┘
                      │ each writes task state + artifacts
                      ▼
         ┌────────────────────────────────┐
         │      DURABLE STATE             │
         │  - task state (structured)     │
         │  - artifact store              │
         │  - scratchpad / WAL            │
         │  - step log                    │
         └────────────┬───────────────────┘
                      │
         ┌────────────▼───────────────────┐
         │        REPLANNER / JUDGE       │
         │  - checks progress              │
         │  - revises plan                │
         │  - triggers termination         │
         └────────────┬───────────────────┘
                      │
                      ▼
                  FINAL OUTPUT
                      │
        ┌─────────────┴─────────────┐
        │  Monitoring + observability │
        │  spans every step           │
        └─────────────────────────────┘
```

Every box in this diagram is a component you've made at least one explicit design choice for — decomposition, state shape, worker structure, replanning trigger, termination rule. That set of choices is what separates a production-capable design from a demo.

## Part 11: Interview Questions

The questions below come up repeatedly in staff-level ML / agent-systems interviews. I've included what the interviewer is actually testing, a strong answer shape, and the follow-ups that usually come next.

---

### Q1. "Walk me through how you'd design an agent to handle a task that takes four hours."

**What they're testing:** whether you think in terms of the five breakage modes above, or whether you handwave with "we'd prompt-engineer it."

**Strong answer shape:**
- State the task type and duration → pick a decomposition pattern (likely Plan-and-Execute).
- Durable state outside context.
- Checkpointing with WAL.
- Explicit budgets and no-progress detection.
- Observability at every step.
- Termination + human-in-loop checkpoints for a 4-hour task.

**Follow-ups:** "What if the task fails at step 150?" → WAL, idempotency, resume from checkpoint. "What if the plan turns out wrong at step 20?" → replanner, allow plan revision.

---

### Q2. "What's the difference between ReAct and Plan-and-Execute? When would you use each?"

**What they're testing:** do you understand the tradeoff between dynamism and structure.

**Strong answer shape:**
- ReAct: single LLM, alternating thought/action, dynamic. Great for short, unpredictable tasks.
- Plan-and-Execute: separate planner and executor. Better for long, structured tasks, enables parallelism, enables human approval of the plan.
- Mixed pattern exists: produce a plan, then let each subtask run ReAct-style internally.

**Follow-ups:** "How do you pick between them for, say, a customer support agent?" (depends on how structured the task is — if you have policies and known steps, plan-based; if it's free-exploration, ReAct.)

---

### Q3. "How would you handle an agent that keeps looping on the same action?"

**What they're testing:** do you know the common failure modes and have pragmatic fixes.

**Strong answer shape:**
- **Detect:** track the last K actions; if repeated, trigger intervention.
- **Intervene:** inject a meta-prompt ("you've tried X twice, try something different"), or escalate to replanner, or terminate.
- **Prevent:** force structured tool call outputs with signatures you can hash; force the model to explicitly state what information each new call should gather.

**Follow-ups:** "What if the action is legitimately retriable?" → distinguish retry-because-transient (OK) from retry-because-stuck (bad). Usually done by checking whether the result changed.

---

### Q4. "How do you deal with the context window filling up during a long run?"

**What they're testing:** context engineering fluency.

**Strong answer shape:**
- Don't rely on full history. Store state structurally outside context.
- Layered context: stable prefix (cached) + rolling summary + recent window.
- Compaction: summarize old steps into a summary, keep artifacts out of context and reference them.
- Retrieval: pull past steps on demand, not all the time.

**Follow-ups:** "What do you summarize away and what do you keep?" → preserve goals, decisions, and links to artifacts; drop detailed reasoning traces.

---

### Q5. "Design the memory / state system for a research agent that may run for a day."

**What they're testing:** state modeling at a non-trivial scale.

**Strong answer shape:**
- Task state: goal, subtasks (with status, dependencies), decisions, deadlines.
- Artifact store: per-source content, extractions, synthesized notes.
- Scratchpad: agent's running notes.
- Cross-task memory (optional): lessons learned, preferences, templates.
- Structured writes (schema'd), clear update APIs, versioning if corrections matter.

**Follow-ups:** "How do you handle contradictions between notes from different sources?" (flag, don't merge silently; structured conflict record.)

---

### Q6. "When do you use multi-agent instead of one agent with more tools?"

**What they're testing:** you understand that multi-agent is not a free win.

**Strong answer shape:**
- Only when you have **cleanly separable subtasks** with **bounded contracts**.
- Only when bounding context per agent is valuable.
- Every inter-agent interface is a new failure mode and new token cost.
- Default: single agent with tools; promote to multi-agent only when the single-agent version has specific, measured shortcomings.

**Follow-ups:** "What's the most common mistake with multi-agent?" — agents sharing state implicitly via free-text messages, which drifts and hallucinates.

---

### Q7. "Your agent produced the right answer in test but wrong in prod after 50 steps. How do you debug?"

**What they're testing:** debugging intuition for long-horizon systems.

**Strong answer shape:**
- Reproduce the failing trace from logs (requires you having them).
- Inspect where task state diverges from expected.
- Common culprits: context drift (model misinterprets task by step 30), tool output changes the plan, compaction dropped something important, a retriable error wasn't retried.
- Fix: add invariants / checks that the agent self-verifies at milestones; tighten termination; improve structured state.

**Follow-ups:** "How would you prevent this class of bug?" — golden-path traces, regression tests on fixed scenarios, output contracts at each subtask boundary.

---

### Q8. "How would you checkpoint an agent so it can resume after a crash?"

**What they're testing:** durable-state thinking.

**Strong answer shape:**
- Checkpoint task state, scratchpad, and WAL. Not the raw context.
- Write-ahead log before tool calls with side effects.
- Idempotent tools (or a wrapper that enforces idempotency via keys).
- On resume: reconstruct context from state + scratchpad, replay any unacknowledged tool call safely, continue.

**Follow-ups:** "What if the tool isn't idempotent and you don't know if it succeeded?" — escalate to human, or have an explicit "check whether the action actually took effect" step first.

---

### Q9. "You have a complex task. The first step is planning. The plan has 20 steps but turns out to be wrong at step 10. How do you handle this?"

**What they're testing:** replanning strategy.

**Strong answer shape:**
- Replanner runs after each step (or periodically). Compares expected vs. actual progress.
- If drift exceeds threshold, produces a revised plan from current state.
- Avoid thrashing: limit how often a plan can be fully rewritten; cheap local edits preferred over full replans.
- Every replan is logged and visible — to the human and to debugging.

**Follow-ups:** "How do you avoid infinite replan loops?" — budgets; also, if the Nth replan looks similar to the (N-1)th, something upstream is broken; escalate.

---

### Q10. "What metrics would you track for a long-running agent?"

**What they're testing:** eval-for-agents fluency. Detailed answer in [the production article](/blog/machine-learning/ai-agent/designing-long-running-agents-reliability-production), but the outline:
- Task success rate (end-to-end).
- Sub-task success rates.
- Steps per task / time per task / cost per task.
- Retry / replan / loop rates.
- Human intervention rate.
- Drift metrics at each milestone.

---

## Closing

Architecture for long-running agents is not more complicated than architecture for any other hard distributed system. It's just **unusual** — the building blocks (LLMs, unreliable tools, semi-structured state) don't look like the ones most engineers have experience with.

Once you internalize the five breakage modes of the naive loop, the rest of this article reads as "one fix per breakage." Pick the pattern that matches your task shape, externalize state, build checkpointing in from day one, and design termination as carefully as you design the loop.

The [companion article](/blog/machine-learning/ai-agent/designing-long-running-agents-reliability-production) picks up where this one ends: tool design, error handling, async execution, evaluation, observability, safety, and cost — with another full interview Q&A block.

---

**Related reading**

- [Building Effective Agents: A Hands-On Guide](/blog/machine-learning/ai-agent/building-effective-agents-hands-on-guide) — the minimal agent scaffolding this article builds on.
- [Effective Context Engineering for AI Agents](/blog/machine-learning/ai-agent/effective-context-engineering-for-ai-agents) — deeper on the context-engineering side of long horizons.
- [Scaling Managed Agents: Decoupling the Brain from the Hands](/blog/machine-learning/ai-agent/scaling-managed-agents-decoupling-brain-from-hands) — the infrastructure pattern that makes long-running agents survive.
- [Designing AI Companion and Assistant Agents](/blog/machine-learning/ai-agent/designing-ai-companion-assistant-agents) — related design space, different task shape.

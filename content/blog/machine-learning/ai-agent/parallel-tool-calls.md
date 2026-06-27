---
title: "Parallel Tool Calls: Executing Agent Actions Concurrently Without Breaking Things"
date: "2026-06-27"
description: "When and how to parallelize agent tool calls — dependency analysis, race conditions, state conflicts, result aggregation, and the latency gains you can expect."
tags: ["ai-agents", "tool-use", "parallelism", "concurrency", "llm", "machine-learning", "system-design", "production-ml"]
category: "machine-learning"
subcategory: "AI Agent"
author: "Hiep Tran"
featured: true
readTime: 34
---

Every agent eventually hits the same wall. The user asks a question that requires pulling from five different sources — a database, two APIs, a file system, and a web search. You implement it the obvious way: call each tool in sequence, wait for the result, then call the next. The agent works correctly. But it's slow. Painfully slow.

If each tool takes 500ms, five sequential calls cost you 2.5 seconds before the LLM even sees the results. Add the LLM's own inference time on top, and you're looking at 3–4 second round-trips for what should feel like an instant assistant. Users notice latency at 200ms. At 3 seconds, they assume the system is broken.

Parallelism is the obvious solution. If those five tool calls don't depend on each other, you can run them concurrently and pay the cost of the slowest one — not the sum. That's the difference between a 500ms response and a 2500ms response. But "just run them in parallel" papers over a set of real engineering problems: dependency ordering, shared state conflicts, partial failures, rate limit coordination, and result aggregation.

This post works through all of it. By the end you'll have a complete mental model for when to parallelize, how to detect conflicts, what to do when things go wrong, and a working Python implementation you can adapt to your own agent.

---

## The Latency Problem: Sequential Tool Calls Add Up Fast

Before optimizing anything, measure the problem clearly.

Consider a research agent that answers the question: "What's the current price of AAPL, what was the Q4 earnings, and what are the top three analyst recommendations?" To answer this, the agent needs four tool calls: `search_web` for analyst recommendations, `get_stock_price` for the current price, `query_earnings_db` for Q4 data, and `fetch_analyst_ratings` for structured ratings data.

In a sequential implementation — which is the default when you implement a basic [agent loop](/blog/machine-learning/ai-agent/agent-loop-anatomy) — each call waits for the previous to complete:

| Tool | Latency | Start (sequential) | End |
|---|---|---|---|
| search_web | 800ms | 0ms | 800ms |
| get_stock_price | 150ms | 800ms | 950ms |
| query_earnings_db | 600ms | 950ms | 1550ms |
| fetch_analyst_ratings | 400ms | 1550ms | 1950ms |

Total: **1950ms** just for tool execution. Then add LLM latency on top.

None of these calls depend on each other. They can all start at time 0. With parallelism:

| Tool | Latency | Start (parallel) | End |
|---|---|---|---|
| search_web | 800ms | 0ms | 800ms |
| get_stock_price | 150ms | 0ms | 150ms |
| query_earnings_db | 600ms | 0ms | 600ms |
| fetch_analyst_ratings | 400ms | 0ms | 400ms |

Total: **800ms** — the maximum of all latencies. A 2.44× speedup with zero change to correctness.

![Sequential versus parallel tool execution — sequential stacks to 2000ms, parallel resolves at 800ms](/imgs/blogs/parallel-tool-calls-1.webp)

The math generalizes. For N independent tools with latencies L₁, L₂, ..., Lₙ:

- Sequential cost: `sum(L₁..Lₙ)`
- Parallel cost: `max(L₁..Lₙ)`
- Speedup factor: `sum(L₁..Lₙ) / max(L₁..Lₙ)`

The speedup grows with the number of tools and becomes especially pronounced when you have one slow tool among many faster ones. If your agent calls one 2000ms database query and five 50ms cache lookups sequentially, you pay 2250ms. In parallel, you pay 2000ms — barely any savings. But if all six calls take 500ms, sequential costs 3000ms while parallel costs 500ms — a 6× improvement.

The practical implication: measure your tool latencies. The speedup depends entirely on the distribution. Tools that hit external APIs, do network I/O, or query large databases typically have latencies in the 200–2000ms range. These are the ones where parallelism pays off. Tools that run locally and return in <10ms might not be worth the complexity overhead of async coordination.

Beyond raw latency, sequential tool calls compound your agent's turn latency in a way that feels particularly bad. Each tool call round-trip includes: function dispatch overhead, any authentication/rate-limit checks, the actual network or compute time, result serialization, and finally injection back into the context. Even if the sequential overhead per call is only 5ms, ten calls means 50ms of pure overhead that disappears entirely with parallelism.

The real killer is user-perceived latency. Users don't perceive 800ms vs 2000ms as "2.5× faster." They perceive 800ms as "fast" and 2000ms as "slow." The psychological impact of crossing the 1-second barrier is much larger than the arithmetic ratio suggests. Getting under 1 second for tool execution is often the difference between an agent that feels like a tool versus one that feels like a collaborator.

---

## When Parallelism Is Safe: Independent Tool Calls with No Shared State

Not all tool calls can safely run in parallel. The rule is simple to state and harder to apply: **two tool calls are safe to parallelize if and only if neither depends on the output of the other, and they do not share mutable state.**

Two kinds of independence matter here:

**Data independence**: Tool B doesn't need the result of Tool A as an input. This is easy to check statically by inspecting whether Tool B's parameters reference any value that would come from Tool A's output.

**State independence**: Tool A and Tool B don't read-then-modify the same piece of mutable state in a way that creates a race condition. This is much harder to check automatically because it requires understanding the semantics of what each tool does to the world.

Let's work through concrete examples.

**Safe to parallelize:**

```python
# These share no state and don't depend on each other
search_web(query="AAPL analyst recommendations")
get_stock_price(ticker="AAPL")
query_earnings_db(ticker="AAPL", quarter="Q4-2024")
```

Each call reads from different data sources (web index, price feed, earnings database). They write nothing. No shared state. Run them all at once.

```python
# Multiple file reads are safe
read_file(path="/data/config.json")
read_file(path="/data/users.json")
read_file(path="/data/settings.json")
```

Reads are always safe to parallelize with other reads. Reading the same file from two concurrent threads is perfectly fine — both get the same bytes.

**Unsafe to parallelize:**

```python
# Tool B uses the result of Tool A — sequential only
user_id = get_user_id(email="alice@example.com")
profile = get_user_profile(user_id=user_id)  # needs user_id from above
```

This is a data dependency. `get_user_profile` can't run until `get_user_id` completes and returns its result. Any attempt to run them concurrently would fail because you don't have `user_id` yet.

```python
# Both write to the same resource — race condition risk
update_document(doc_id="D1", section="intro", content="...")
update_document(doc_id="D1", section="conclusion", content="...")
```

These might be safe or unsafe depending on the implementation. If the database handles section-level locking and these are genuinely separate sections, it might be fine. If the implementation reads the full document, modifies a section, and writes the full document back, you have a classic read-modify-write race: both reads see the original, both modifications diverge, and one write stomps the other.

**The gray zone:**

```python
# Read-then-conditional-write — it depends
balance = get_account_balance(account_id="A1")
transfer(from_account="A1", amount=100, to_account="A2")  # depends on balance
```

If `transfer` doesn't check the balance independently, you might transfer even when the account goes negative. If another parallel call is also draining the account, both might succeed when only one should. This is the classic TOCTOU (time-of-check to time-of-use) race.

The general principle: **reads are safe to parallelize with reads. Reads are risky to parallelize with writes to the same resource. Writes are dangerous to parallelize with other writes to the same resource.**

For tool-use in agents, you also need to consider external side effects. Sending an email, triggering a webhook, or writing to an append-only log might be technically safe to parallelize (no read-modify-write race) but semantically wrong if the order matters — "notify user of account creation" must happen after "create account."

![Tool call dependency DAG — independent fetch nodes can parallelize; process_data and send_report must sequence](/imgs/blogs/parallel-tool-calls-2.webp)

The relationship between [tool schema design](/blog/machine-learning/ai-agent/tool-schema-design-principles) and parallelism is real: tools with well-defined, narrow responsibilities are easier to reason about for parallelism. A tool called `update_everything` is impossible to safely parallelize because you can't know what it touches. A tool called `update_user_email(user_id, new_email)` is easy — it touches exactly one record.

---

## Dependency Analysis: Building a DAG Over Tool Calls

Once you have a set of candidate tool calls, the formal way to figure out what can parallelize is to build a directed acyclic graph (DAG) where nodes are tool calls and directed edges represent "must complete before" relationships.

The topological sort of this DAG gives you **execution waves**: groups of tool calls at the same topological level that can all run in parallel. Each wave must complete before the next wave starts.

Here's how to build the dependency graph in Python:

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ToolCall:
    id: str
    name: str
    params: dict[str, Any]
    depends_on: list[str] = field(default_factory=list)  # IDs of prerequisite calls

def build_execution_waves(tool_calls: list[ToolCall]) -> list[list[ToolCall]]:
    """Topological sort into execution waves."""
    call_map = {tc.id: tc for tc in tool_calls}
    in_degree = {tc.id: len(tc.depends_on) for tc in tool_calls}
    dependents: dict[str, list[str]] = {tc.id: [] for tc in tool_calls}
    
    for tc in tool_calls:
        for dep in tc.depends_on:
            dependents[dep].append(tc.id)
    
    waves = []
    ready = [tc_id for tc_id, deg in in_degree.items() if deg == 0]
    
    while ready:
        current_wave = [call_map[tc_id] for tc_id in ready]
        waves.append(current_wave)
        next_ready = []
        for tc_id in ready:
            for dep_id in dependents[tc_id]:
                in_degree[dep_id] -= 1
                if in_degree[dep_id] == 0:
                    next_ready.append(dep_id)
        ready = next_ready
    
    return waves
```

For an agent that needs to: (1) look up user preferences and (2) fetch recent activity, then (3) generate a personalized summary, the dependency graph looks like:

```python
calls = [
    ToolCall(id="prefs", name="get_user_prefs", params={"user_id": "U1"}),
    ToolCall(id="activity", name="get_recent_activity", params={"user_id": "U1"}),
    ToolCall(id="summary", name="generate_summary", params={}, depends_on=["prefs", "activity"]),
]

waves = build_execution_waves(calls)
# Wave 1: [get_user_prefs, get_recent_activity]  — parallel
# Wave 2: [generate_summary]                     — sequential
```

The key question is: who builds the dependency graph? There are two answers depending on your architecture.

**Static analysis** works when the LLM generates a complete plan upfront: "I need to call tools A, B, C, and D. A and B are independent. C depends on A. D depends on both B and C." You parse this plan, build the DAG, and execute it. This requires the LLM to reason about dependencies explicitly, which modern models can do but imperfectly.

**Streaming dependency detection** works when the LLM generates tool calls one at a time in an event-driven loop. You build dependencies dynamically: when a new tool call arrives, check if any of its parameter values match the output IDs of pending calls. If so, add a dependency edge. This is more robust to imperfect LLM planning because you're enforcing data dependencies mechanically.

**Annotation-based detection** is the most reliable approach: your tool schemas include metadata about what resources each tool reads and writes. At execution time, you check for read-write conflicts between any two candidate calls. If tool A writes to `resource:user_profile:U1` and tool B reads from `resource:user_profile:U1`, you force them sequential. If both only read from the same resource, they can parallelize.

```python
@dataclass
class ToolSchema:
    name: str
    reads: set[str]   # resource keys this tool reads
    writes: set[str]  # resource keys this tool writes

def can_parallelize(a: ToolSchema, b: ToolSchema) -> bool:
    # Safe if: no write-write conflicts and no read-write conflicts
    if a.writes & b.writes:      return False  # both write same resource
    if a.writes & b.reads:       return False  # a writes what b reads
    if a.reads & b.writes:       return False  # b writes what a reads
    return True
```

This scales to automated dependency analysis across large sets of tool calls, and it makes the parallelism decision auditable — you can log exactly why two tools were forced sequential.

---

## LLM-Driven Parallelism: Models That Emit Multiple Tool Calls in One Response

Modern LLMs — Claude, GPT-4, Gemini — support emitting multiple tool calls in a single response. When the model determines that several tool calls are independent, it outputs them all at once rather than waiting for each to complete before requesting the next. This is the protocol-level enabler for agent parallelism.

In the Anthropic API, a parallel tool call response looks like this:

```json
{
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "I'll look up the stock price, earnings, and analyst ratings simultaneously."
    },
    {
      "type": "tool_use",
      "id": "tc_001",
      "name": "get_stock_price",
      "input": {"ticker": "AAPL"}
    },
    {
      "type": "tool_use",
      "id": "tc_002",
      "name": "query_earnings_db",
      "input": {"ticker": "AAPL", "quarter": "Q4-2024"}
    },
    {
      "type": "tool_use",
      "id": "tc_003",
      "name": "search_web",
      "input": {"query": "AAPL analyst recommendations 2024"}
    }
  ]
}
```

The model emits three `tool_use` blocks in one assistant message. Your agent framework receives these and — if you've implemented it correctly — dispatches all three concurrently. The results come back asynchronously, you collect them, format them as `tool_result` messages, and inject them into the next turn.

The key implementation detail is that you must send all tool results back in a single `user` turn. The API expects tool results keyed by their `tool_use_id`:

```json
{
  "role": "user",
  "content": [
    {
      "type": "tool_result",
      "tool_use_id": "tc_001",
      "content": [{"type": "text", "text": "AAPL: $189.43"}]
    },
    {
      "type": "tool_result",
      "tool_use_id": "tc_002",
      "content": [{"type": "text", "text": "Q4-2024 EPS: $2.18, Revenue: $119.6B"}]
    },
    {
      "type": "tool_result",
      "tool_use_id": "tc_003",
      "content": [{"type": "text", "text": "Goldman: Buy $225, JPMorgan: Overweight $220..."}]
    }
  ]
}
```

The order of results doesn't matter — the model uses `tool_use_id` to correlate each result with its request. This is crucial: your async executor can collect results in any completion order, sort them by ID, and return a coherent batch to the model.

Not every LLM will emit multiple tool calls by default. You may need to tune the system prompt or use few-shot examples to teach the model to parallelize. Something like: "When multiple independent tool calls are needed, emit them all in a single response rather than one at a time." Claude tends to be good at this natively; older GPT-3.5 models needed more explicit guidance.

![LLM parallel tool call flow — single assistant response with three tool_use blocks dispatched concurrently](/imgs/blogs/parallel-tool-calls-4.webp)

One nuance: the model's decision to parallelize is based on its understanding of the task, not mechanical analysis. It might incorrectly parallelize two calls that actually have a data dependency (because the dependency was implicit rather than explicit in the prompt). Your executor needs a safety check: if Tool B's parameters contain a placeholder that references Tool A's output, enforce sequencing regardless of what the model requested.

The model's parallelism decisions also depend on the task framing. "Get the stock price, then get the earnings" will often be executed sequentially even if the calls are independent, because "then" implies ordering. "Get the stock price and the earnings" is more likely to trigger parallel emission. This is a prompt engineering consideration worth knowing.

---

## Race Conditions in Agent State: What Goes Wrong When Tools Share State

A race condition occurs when the correctness of your program depends on the relative timing of two concurrent operations. In agent tool calls, race conditions emerge whenever two parallel tools touch the same piece of mutable state.

There are three fundamental categories, each with a different failure mode:

**Write-write conflicts** happen when two tools both write to the same resource. The last writer wins, and the first writer's change is silently lost. Example: an agent editing a document runs `append_section(doc_id="D1", section="intro", text="...")` and `append_section(doc_id="D1", section="conclusion", text="...")` in parallel. If the implementation is:

```python
def append_section(doc_id, section, text):
    doc = db.read(doc_id)          # both reads see original
    doc["sections"][section] = text # modify different keys
    db.write(doc_id, doc)          # last write wins — one modification is lost
```

...then one modification will be silently overwritten. The section that commits second will stomp the section that committed first, because both started from the same original state.

**Read-write hazards** (also called dirty reads and phantom reads) occur when one tool reads data that another tool is in the middle of writing. If a tool reads a value that is mid-update, it sees an inconsistent state. For example, a "transfer funds" tool that reads balance, deducts, and writes back can leave the account in an incorrect state if a concurrent "read balance" tool sees the intermediate state.

**Check-then-act races** (TOCTOU) are the trickiest. The pattern is: (1) check a condition, (2) act based on that condition. If the condition changes between check and act, you've acted on stale information. Classic example: "check if file exists, then create it if it doesn't." If two parallel tool calls both check before either creates, both will try to create.

```python
# TOCTOU in an agent tool
def create_user_if_not_exists(email):
    if not db.user_exists(email):    # both tools check: user doesn't exist
        db.create_user(email)        # both tools try to create — duplicate!
```

The severity depends on whether the underlying operations are idempotent. If `create_user` is idempotent (second call is a no-op), the TOCTOU doesn't matter. If it fails on duplicate, or creates two records, you have a real bug.

![Race condition types vs mitigation strategies matrix](/imgs/blogs/parallel-tool-calls-5.webp)

In practice, the most common agent race condition I've seen is not dramatic data corruption — it's subtle state divergence. Two tools both fetch a resource, make different modifications, and write back. The final state is a mix of the two writes that's internally inconsistent. In a user profile context, this might mean the user's name is from one update and their email is from another, when the intent was to update both atomically.

The safest heuristic: **if a tool has any write operations, scrutinize it carefully before parallelizing it with anything**. Read-only tools are almost always safe to parallelize. Write tools need case-by-case analysis.

---

## Write Conflict Patterns: Idempotency, Optimistic Locking, Last-Write-Wins

When you must run multiple write-capable tools in parallel — or when your application architecture makes purely sequential writes too slow — you need explicit conflict resolution strategies.

**Idempotency** is the first and most powerful tool. An idempotent operation produces the same result regardless of how many times it's called with the same inputs. `PUT /users/U1 {email: "new@example.com"}` is idempotent: calling it twice gives the same result. `POST /users {email: "new@example.com"}` is not: calling it twice creates two users.

Design your agent tools to be idempotent wherever possible. This means:

- Use upserts instead of inserts: "create or update" rather than "create"
- Use PUT semantics (full resource replacement) rather than PATCH (partial update) when you control the API
- Include idempotency keys: `create_payment(idempotency_key="req_123", ...)` — the server deduplicates retries with the same key
- For operations with side effects (emails, webhooks), gate on a "has this already run?" check

When two tools both call `set_user_preference(user_id="U1", key="theme", value="dark")`, idempotency means the second call is a no-op. No conflict.

**Optimistic locking** allows concurrent reads but detects conflicting writes at commit time. The pattern: read the resource along with a version number, make modifications locally, then write back with "if version still equals what I read." If another write happened in between, the version check fails and you retry.

```python
def update_document_section(doc_id, section, text, max_retries=3):
    for attempt in range(max_retries):
        doc, version = db.read_with_version(doc_id)
        doc["sections"][section] = text
        success = db.write_if_version_matches(doc_id, doc, expected_version=version)
        if success:
            return
        # Version mismatch: someone else wrote; re-read and retry
        time.sleep(0.05 * (2 ** attempt))  # exponential backoff
    raise ConflictError(f"Could not update {doc_id} after {max_retries} retries")
```

Optimistic locking works well when conflicts are rare (writes don't frequently stomp each other) and retries are cheap. It's the right choice when you want parallelism but need correctness.

**Last-write-wins** is an explicit choice to accept that concurrent writes will stomp each other, and the most recent one is authoritative. This is sometimes exactly what you want: if two parallel tools both update a "last_accessed" timestamp, it doesn't matter which one commits last — the final value is still a recent timestamp. For user preferences, you might accept last-write-wins on the grounds that the user's intent is to end up in a specific state, not to preserve every intermediate update.

Document when you're using last-write-wins. It's not a race condition bug if you've explicitly decided it's the correct semantics.

**Serialization queues** are the nuclear option: force all writes to a shared resource through a single queue that processes them one at a time. This eliminates races entirely but sacrifices parallelism for writes to that resource. In Python:

```python
import asyncio

class ResourceLock:
    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = {}
    
    async def acquire(self, resource_id: str):
        if resource_id not in self._locks:
            self._locks[resource_id] = asyncio.Lock()
        return self._locks[resource_id]
    
    async def run_exclusive(self, resource_id: str, coro):
        lock = await self.acquire(resource_id)
        async with lock:
            return await coro
```

This is appropriate for high-value, low-frequency writes where correctness is non-negotiable — financial transactions, user account operations, configuration changes.

---

## Result Aggregation: Collecting and Injecting Parallel Results

Once your parallel tool calls complete, you need to collect their results and inject them back into the LLM's context in a coherent way. This sounds simple but has several nuances.

**Collection order vs. injection order**: Async tool calls complete in unpredictable order. The first tool to finish is not necessarily the first tool in your list. Your aggregator needs to collect results by `tool_use_id` (not position) and then inject them in a stable order. Most LLM APIs expect results ordered by their original tool call positions, so sort by ID before injecting.

```python
import asyncio
from typing import Any

async def execute_parallel_tool_calls(
    tool_calls: list[dict],
    executor: dict  # {tool_name: async callable}
) -> list[dict]:
    async def run_one(tc: dict) -> dict:
        tool_fn = executor[tc["name"]]
        try:
            result = await tool_fn(**tc["input"])
            return {
                "type": "tool_result",
                "tool_use_id": tc["id"],
                "content": [{"type": "text", "text": str(result)}]
            }
        except Exception as e:
            return {
                "type": "tool_result",
                "tool_use_id": tc["id"],
                "is_error": True,
                "content": [{"type": "text", "text": f"Error: {e}"}]
            }
    
    results = await asyncio.gather(*[run_one(tc) for tc in tool_calls])
    
    # Sort by original tool call order
    id_to_order = {tc["id"]: i for i, tc in enumerate(tool_calls)}
    results.sort(key=lambda r: id_to_order[r["tool_use_id"]])
    
    return results
```

![Result aggregation — parallel results flow into merge, then inject into next context, then LLM next turn](/imgs/blogs/parallel-tool-calls-6.webp)

**Context window management**: Parallel tool calls can return a lot of data at once. Five web searches might each return 2KB of results — 10KB total in a single injection. If you're deep into a long conversation, this might push you over the context limit. Pre-aggregate: summarize or truncate individual results before injecting if they're verbose. A `search_web` result doesn't need to include full page content — 200 words of relevant excerpts is usually enough.

**Semantic ordering**: Even if the API accepts results in any order, there's often a natural reading order that helps the LLM respond coherently. If you're gathering data about a company, presenting financial results before analyst recommendations makes sense. Consider the LLM's "reading experience" when choosing injection order.

**Partial result injection**: If one tool call is still running after the others complete (due to high latency variance), do you wait or inject what you have? The right answer depends on the use case. For a dashboard that needs all metrics, wait. For a research assistant that can reason about partial information, inject what's ready and continue. This requires a timeout-based collection strategy rather than waiting for all results unconditionally.

```python
async def collect_with_timeout(
    tasks: list[asyncio.Task],
    timeout: float = 5.0,
    on_timeout: str = "continue"  # "continue" | "raise"
) -> list[Any]:
    done, pending = await asyncio.wait(tasks, timeout=timeout)
    if pending and on_timeout == "raise":
        for task in pending:
            task.cancel()
        raise TimeoutError(f"{len(pending)} tool calls timed out")
    # Cancel remaining tasks and collect what we have
    results = []
    for task in done:
        results.append(task.result())
    for task in pending:
        task.cancel()
        results.append({"error": "timeout", "task": task.get_name()})
    return results
```

---

## Partial Failure Handling: What to Do When 1 of 5 Parallel Calls Fails

Running five tools concurrently means you've given five things the opportunity to fail simultaneously. The default `asyncio.gather()` behavior cancels all remaining tasks when any one fails (with `return_exceptions=False`), which is often not what you want.

Your failure handling strategy should match the semantic relationship between the tool calls:

**Strategy 1: Retry the failed call only**

When the failure is likely transient (network timeout, rate limit, temporary service outage) and the failed result is critical to the overall task, retry just that one call while the others complete normally.

```python
import asyncio
import random

async def with_retry(coro_fn, max_retries=3, base_delay=0.5):
    last_error = None
    for attempt in range(max_retries):
        try:
            return await coro_fn()
        except (ConnectionError, TimeoutError) as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                await asyncio.sleep(delay)
    raise last_error
```

Wrap each tool call with retry logic before gathering. This way, transient failures are handled within the parallel execution without disrupting the other calls.

**Strategy 2: Continue without the failed result**

When partial information is still useful and the agent can reason appropriately about missing data. A research agent looking for information across five sources can still provide a good answer if one source returns an error — just note the gap.

```python
results = await asyncio.gather(*tasks, return_exceptions=True)

successful = []
failed = []
for result in results:
    if isinstance(result, Exception):
        failed.append(result)
    else:
        successful.append(result)

# Inject successful results, annotate failures
context_addition = format_results(successful)
if failed:
    context_addition += f"\nNote: {len(failed)} tool calls failed: {[str(e) for e in failed]}"
```

![Partial failure decision tree — retry / continue-without / abort branches with when-to-use guidance](/imgs/blogs/parallel-tool-calls-7.webp)

**Strategy 3: Abort all on any failure**

When the task is all-or-nothing — you need all results to proceed safely, and partial results would cause the agent to take incorrect actions. Financial calculations, safety checks, or any workflow where missing data leads to a worse outcome than no output.

```python
results = await asyncio.gather(*tasks, return_exceptions=True)
errors = [r for r in results if isinstance(r, Exception)]

if errors:
    # Cancel any still-running tasks
    for task in tasks:
        if not task.done():
            task.cancel()
    raise ToolExecutionError(
        f"Required tool calls failed: {errors}. Aborting agent turn."
    )
```

**Strategy 4: Majority-succeeds threshold**

For redundancy patterns where you've deliberately called multiple tools to cross-validate results. If 4 of 5 sources return the same answer and one fails, you probably have enough confidence to proceed. This is common in research agents and fact-checking workflows.

```python
min_success_fraction = 0.6

results = await asyncio.gather(*tasks, return_exceptions=True)
successes = [r for r in results if not isinstance(r, Exception)]

if len(successes) / len(tasks) < min_success_fraction:
    raise InsufficientDataError("Too many tool calls failed to proceed safely")
```

The right default is "retry then continue without" for most agent use cases. Agents are often asking questions rather than executing transactions, and partial information with appropriate uncertainty is usually better than failing completely. Reserve "abort all" for transaction-like operations where partial execution leaves the system in an inconsistent state.

---

## Rate Limit Management: Coordinating Parallel Calls Against API Limits

Parallelism without rate limit coordination will hammer your API providers. Fire 20 concurrent calls at an API that allows 60 requests per minute and you'll hit 429s almost immediately, turning your speedup into a retry storm.

The standard solution is a **token bucket** implemented as a shared semaphore across all parallel workers:

```python
import asyncio
import time

class RateLimiter:
    def __init__(self, requests_per_second: float):
        self.rate = requests_per_second
        self.tokens = requests_per_second
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
            self.last_refill = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return
            
            # Wait until we have a token
            wait_time = (1 - self.tokens) / self.rate
        
        await asyncio.sleep(wait_time)
        await self.acquire()  # re-check after sleeping

# Global rate limiter, shared across all parallel workers
_search_limiter = RateLimiter(requests_per_second=2.0)  # 120 req/min

async def search_web_rate_limited(query: str) -> str:
    await _search_limiter.acquire()
    return await search_web(query)
```

![Rate limit coordination — token bucket shared across parallel workers, backpressure delays Workers 3 and 4](/imgs/blogs/parallel-tool-calls-8.webp)

For multi-API agents, maintain separate rate limiters per API provider:

```python
RATE_LIMITERS = {
    "openai": RateLimiter(requests_per_second=0.5),      # 30 req/min
    "serpapi": RateLimiter(requests_per_second=1.0),     # 60 req/min
    "database": RateLimiter(requests_per_second=10.0),   # 600 req/min
    "file_system": RateLimiter(requests_per_second=50.0), # local, generous
}
```

Beyond simple rate limiting, consider **concurrency caps** — a hard limit on how many calls of a given type run simultaneously, regardless of rate limits:

```python
class ConcurrencyLimiter:
    def __init__(self, max_concurrent: int):
        self._semaphore = asyncio.Semaphore(max_concurrent)
    
    async def __aenter__(self):
        await self._semaphore.acquire()
        return self
    
    async def __aexit__(self, *args):
        self._semaphore.release()

# Max 3 concurrent web searches
_search_concurrency = ConcurrencyLimiter(max_concurrent=3)

async def search_web_bounded(query: str) -> str:
    async with _search_concurrency:
        return await search_web(query)
```

Combine both: rate limiter for throughput control, concurrency limiter for resource protection. Some APIs have both a requests-per-minute limit (rate) and a concurrent-connections limit (concurrency). Your tools should respect both.

One more consideration: **retry-after headers**. When you hit a 429, the API response often includes a `Retry-After` header telling you when to retry. Respect it — don't retry immediately. A good rate limiter parses this header and backs off accordingly:

```python
async def with_retry_after(coro_fn, tool_name: str):
    while True:
        try:
            return await coro_fn()
        except RateLimitError as e:
            retry_after = e.retry_after_seconds or 1.0
            print(f"[{tool_name}] Rate limited, waiting {retry_after}s")
            await asyncio.sleep(retry_after)
```

---

## Latency Math: Expected Speedup with N Parallel Calls (Amdahl's Law Applied)

Amdahl's Law gives the theoretical maximum speedup from parallelism given a fixed sequential fraction. For agent tool calls, the "sequential fraction" is everything that can't be parallelized: the LLM's own processing time, result aggregation, prompt construction, and any tools that must run in series due to dependencies.

The formula: `Speedup(N) = 1 / (S + (1-S)/N)`

Where:
- `S` = fraction of total work that must remain sequential
- `N` = number of parallel workers
- `(1-S)` = fraction that can be parallelized

For an agent turn where the LLM takes 1000ms, tool calls take 2000ms total (all independent), and aggregation takes 100ms:

- Total sequential work: 1000ms (LLM) + 100ms (aggregation) = 1100ms
- Total work: 1100ms + 2000ms = 3100ms
- Sequential fraction S: 1100/3100 ≈ 0.355
- With 4 parallel workers: Speedup = 1 / (0.355 + 0.645/4) = 1 / (0.355 + 0.161) = 1 / 0.516 ≈ 1.94×
- Latency with parallelism: 3100ms / 1.94 ≈ 1598ms (vs 3100ms sequential)

Actual latency with 4 tools in parallel: 1000ms (LLM) + 500ms (slowest tool) + 100ms (aggregation) = 1600ms. Close to the theoretical prediction.

The insight from Amdahl: **diminishing returns set in quickly**. Going from 1→2 parallel calls gives you the biggest speedup. Going from 8→16 gives almost nothing, because by that point you're dominated by the sequential fraction (LLM inference + aggregation).

![Amdahl's law applied to tool parallelism — 80% and 50% parallelizable curves show diminishing returns past N=8](/imgs/blogs/parallel-tool-calls-9.webp)

The practical implication for agent design: if your LLM inference takes 2000ms and your tools take 500ms each, maximizing tool parallelism barely moves the needle. The bottleneck is the LLM. But if your agent uses fast, cached LLM responses (or smaller models) and slow external APIs, tool parallelism is the critical path.

To identify your actual speedup potential:

1. Measure sequential tool call latency: `sum(all tool latencies)`
2. Measure irreducible sequential work: LLM inference + aggregation + prompt construction
3. Compute theoretical max speedup: `(sequential + parallel) / (sequential + max_parallel)`
4. Compare to actual measured speedup after implementation

If your actual speedup is significantly less than theoretical, investigate:
- Are some tools still running sequentially that could parallelize?
- Is there lock contention in shared state?
- Is rate limiting creating a bottleneck?
- Is result aggregation taking longer than expected due to large payloads?

---

## Implementation: Python Async Tool Executor with Dependency Graph

Here's a complete, production-ready parallel tool executor that handles dependency ordering, rate limiting, partial failures, and result aggregation:

```python
import asyncio
import time
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

@dataclass
class ToolCall:
    id: str
    name: str
    params: dict[str, Any]
    depends_on: list[str] = field(default_factory=list)

@dataclass
class ToolResult:
    tool_use_id: str
    content: str
    is_error: bool = False

class RateLimiter:
    def __init__(self, rps: float):
        self.rate = rps
        self.tokens = rps
        self.last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            self.tokens = min(self.rate, self.tokens + (now - self.last) * self.rate)
            self.last = now
            if self.tokens >= 1:
                self.tokens -= 1
                return
            wait = (1 - self.tokens) / self.rate
        await asyncio.sleep(wait)
        await self.acquire()

class ParallelToolExecutor:
    def __init__(
        self,
        tools: dict[str, Callable[..., Awaitable[Any]]],
        rate_limiters: dict[str, RateLimiter] | None = None,
        max_retries: int = 2,
        partial_failure: str = "continue"  # "continue" | "abort" | "retry_only"
    ):
        self.tools = tools
        self.rate_limiters = rate_limiters or {}
        self.max_retries = max_retries
        self.partial_failure = partial_failure

    def _topological_waves(self, calls: list[ToolCall]) -> list[list[ToolCall]]:
        call_map = {c.id: c for c in calls}
        in_degree = {c.id: len(c.depends_on) for c in calls}
        dependents: dict[str, list[str]] = {c.id: [] for c in calls}
        for c in calls:
            for dep in c.depends_on:
                dependents[dep].append(c.id)
        waves, ready = [], [cid for cid, d in in_degree.items() if d == 0]
        while ready:
            waves.append([call_map[cid] for cid in ready])
            nxt = []
            for cid in ready:
                for dep_id in dependents[cid]:
                    in_degree[dep_id] -= 1
                    if in_degree[dep_id] == 0:
                        nxt.append(dep_id)
            ready = nxt
        return waves

    async def _run_one(self, call: ToolCall, results_so_far: dict[str, Any]) -> ToolResult:
        # Resolve param references to prior results
        resolved = {}
        for k, v in call.params.items():
            if isinstance(v, str) and v.startswith("$ref:"):
                ref_id = v[5:]
                resolved[k] = results_so_far.get(ref_id, v)
            else:
                resolved[k] = v

        limiter = self.rate_limiters.get(call.name)

        for attempt in range(self.max_retries + 1):
            try:
                if limiter:
                    await limiter.acquire()
                result = await self.tools[call.name](**resolved)
                return ToolResult(tool_use_id=call.id, content=str(result))
            except Exception as e:
                if attempt == self.max_retries:
                    return ToolResult(tool_use_id=call.id, content=str(e), is_error=True)
                await asyncio.sleep(0.2 * (2 ** attempt) + random.uniform(0, 0.05))

    async def execute(self, calls: list[ToolCall]) -> list[ToolResult]:
        waves = self._topological_waves(calls)
        all_results: dict[str, Any] = {}
        final_results: list[ToolResult] = []

        for wave in waves:
            wave_results = await asyncio.gather(
                *[self._run_one(c, all_results) for c in wave],
                return_exceptions=False
            )
            errors = [r for r in wave_results if r.is_error]
            if errors and self.partial_failure == "abort":
                raise RuntimeError(f"Aborting: {len(errors)} tool calls failed")
            for r in wave_results:
                all_results[r.tool_use_id] = r.content
                final_results.append(r)

        # Return sorted by original call order
        order = {c.id: i for i, c in enumerate(calls)}
        final_results.sort(key=lambda r: order[r.tool_use_id])
        return final_results
```

Usage:

```python
# Define your tool implementations
async def search_web(query: str) -> str:
    await asyncio.sleep(0.8)  # simulate network call
    return f"Results for: {query}"

async def get_stock_price(ticker: str) -> str:
    await asyncio.sleep(0.15)
    return f"{ticker}: $189.43"

executor = ParallelToolExecutor(
    tools={"search_web": search_web, "get_stock_price": get_stock_price},
    rate_limiters={"search_web": RateLimiter(rps=2.0)},
    partial_failure="continue"
)

calls = [
    ToolCall(id="tc1", name="search_web", params={"query": "AAPL news"}),
    ToolCall(id="tc2", name="get_stock_price", params={"ticker": "AAPL"}),
]

results = asyncio.run(executor.execute(calls))
```

This implementation handles the full dependency graph, rate limiting per tool, exponential backoff retries, partial failure modes, and stable result ordering in roughly 80 lines.

---

## Case Studies: Parallelism Wins and Race Condition Bugs

### Case Study 1: Research Agent — 5× Latency Reduction

A market research agent answered questions by searching multiple news sources, databases, and public APIs. The initial implementation called each source sequentially.

**Baseline**: Five sources × 400ms average = 2000ms tool time + 1200ms LLM = 3200ms total.

After identifying that all five sources were independent (different URLs, no shared state, read-only), the team implemented the parallel executor. **Result**: tool time dropped to 480ms (slowest source) + 1200ms LLM = 1680ms — a 1.9× end-to-end speedup and a 4.2× speedup in tool execution alone.

The implementation required one significant change: rate limiting. The original code naturally stayed within rate limits because it was sequential. With parallel execution, it could fire all five calls simultaneously. One source had a 10 req/min limit, meaning burst calls from parallel agents at scale could exhaust the quota immediately. Adding per-source rate limiters resolved this with no correctness changes.

### Case Study 2: Calendar Assistant Race Condition Bug

A scheduling agent could both `read_calendar` and `create_event` in parallel. A bug report came in: "The agent double-booked me." Investigation revealed the sequence:

1. User asks: "Schedule a meeting with Alice and Bob tomorrow at 2pm, and add a reminder for the dentist at 3pm"
2. Agent emits two parallel `create_event` calls
3. Both calls fetch the calendar to check for conflicts: no conflicts seen at time of read
4. First call creates "Meeting with Alice and Bob 2-3pm"
5. Second call creates "Dentist 3-4pm"
6. But "Meeting with Alice and Bob" was actually supposed to be 2-4pm — the agent made a mistake in the first call
7. User asks the agent to fix it — agent emits `delete_event("Meeting")` and `create_event("Meeting 2-4pm")` in parallel
8. Race: delete and new create race with each other; new create sees the old event, detects "no conflict with itself," deletes nothing
9. Result: three events on the calendar, including the one that should have been deleted

The fix: serialize all write operations on a per-calendar basis using the `ConcurrencyLimiter` pattern. Reads could still parallelize with other reads. Writes serialized through a per-calendar lock. Race conditions eliminated.

### Case Study 3: Data Pipeline Fan-Out — The Happy Path

An ETL agent needed to ingest data from 12 different database tables into a data warehouse. Each table was independent. Sequential execution: 12 tables × 800ms = 9.6 seconds per run. With the parallel executor configured to run 4 concurrent operations (limited by database connection pool):

- Wave 1: Tables 1-4 concurrently (800ms)
- Wave 2: Tables 5-8 concurrently (800ms)  
- Wave 3: Tables 9-12 concurrently (800ms)
- Total: 2400ms

**3.75×** speedup with no code changes to the individual table-read logic. The only engineering work was the parallel executor infrastructure. This is the cleanest parallelism win: genuinely independent read operations with no shared state.

### Case Study 4: Document Editing Agent — Write Conflict Fixed with Optimistic Locking

A document editing agent that could update multiple sections of a markdown document simultaneously. Initial parallel implementation caused silent data loss: when two sections both modified the same document file concurrently, one write would overwrite the other.

Fix: converted the document write to use optimistic locking with a file hash as the version token:

```python
async def update_section_safe(doc_path, section, content, max_retries=3):
    for _ in range(max_retries):
        current_content = read_file(doc_path)
        current_hash = hashlib.md5(current_content.encode()).hexdigest()
        
        new_content = apply_section_update(current_content, section, content)
        
        # Only write if file hasn't changed since we read it
        if try_atomic_write_if_unchanged(doc_path, current_hash, new_content):
            return
        # Hash mismatch: file changed, re-read and retry
        await asyncio.sleep(0.05)
    raise ConflictError("Document modified too frequently for safe parallel update")
```

Now multiple section updates can proceed in parallel. Conflicts are detected and retried automatically. In practice, conflicts are rare because sections are typically independent regions of the document.

### Case Study 5: API Fan-Out with Partial Failure Handling

An agent that needed pricing data from three sources (Bloomberg API, Reuters API, in-house model) and would average the results. If any one source was unavailable, using the remaining two was acceptable. If all three failed, the agent should report an error.

Implementation using the majority-threshold strategy:

```python
calls = [
    ToolCall(id="bloomberg", name="fetch_bloomberg_price", params={"ticker": "AAPL"}),
    ToolCall(id="reuters", name="fetch_reuters_price", params={"ticker": "AAPL"}),
    ToolCall(id="inhouse", name="fetch_model_price", params={"ticker": "AAPL"}),
]

results = await executor.execute(calls)  # partial_failure="continue"

successes = [r for r in results if not r.is_error]
if len(successes) == 0:
    raise InsufficientDataError("All pricing sources failed")

avg_price = mean(float(r.content) for r in successes)
confidence = "high" if len(successes) == 3 else "medium"
```

During an outage that took Bloomberg offline for 30 minutes, the agent continued operating at full speed using Reuters and the in-house model. Without the partial failure handling, the agent would have been completely unavailable for that period.

### Case Study 6: Multi-Tool Authentication Race

An agent that needed to perform several authenticated API calls in parallel. Initial implementation: each tool call independently refreshed its auth token if it found the token expired.

Race condition: token expires at T=0. Three parallel tool calls all see the expired token. All three call `refresh_token()` simultaneously. The auth server creates three tokens and invalidates all previous ones — including the tokens just created by the first two calls. By the time the third call refreshes, the tokens from the first two calls are invalid. All three calls fail with "invalid token" despite having just refreshed.

Fix: add a shared token lock so only one refresh happens at a time:

```python
_token_lock = asyncio.Lock()

async def get_valid_token() -> str:
    async with _token_lock:
        if token_is_valid(current_token):
            return current_token
        return await refresh_and_store_token()
```

All subsequent calls acquire the lock, check if the token is still valid (it probably is by now, because another call just refreshed it), and proceed without re-refreshing. The serialization is fine here because token refreshes are rare — the lock is almost always uncontested.

### Case Study 7: The Sequential Fallback

An agent designed to optimize a database schema needed to: (1) analyze current schema, (2) propose changes, (3) validate changes against existing queries, (4) apply changes. This looks like it could parallelize steps 3 and 4.

It cannot. Step 4 applies schema changes. Step 3 validates that queries still work against the new schema. If both run in parallel, step 3 might validate against the old schema while step 4 is already applying the new one — or vice versa. The validation test becomes meaningless.

Sometimes the right answer is "don't parallelize." The latency cost of sequential execution (maybe 2× slower) is trivially worth the correctness guarantee. Identify these cases explicitly and document the reason in code.

---

## When to Parallelize / When Sequential Is Safer

A clear decision framework for production agents:

**Parallelize when:**

1. All tool calls are reads with no shared mutable state
2. You've verified data independence: no call's output feeds another call's input
3. The tool schemas declare non-overlapping write targets
4. Latency savings are meaningful (>100ms total improvement)
5. Individual tool failures can be handled gracefully (retry or continue-without)

**Keep sequential when:**

1. Any tool call writes, and subsequent tools read from the same resource
2. Operations must occur in a specific order for semantic correctness (create account → send welcome email)
3. Each tool call's behavior depends on the result of the previous one
4. External side effects must be ordered (audit logs, event streams, financial transactions)
5. Your total tool latency is <200ms — parallelism overhead may wash out the gain

**Partial parallelism (wave execution) when:**

1. Some calls are independent and some have dependencies
2. You have distinct "read phase" → "process phase" → "write phase" operations
3. The first N calls gather data and the last call uses all that data

![Tool call pattern grid — 8 patterns with parallelism recommendation and risk level](/imgs/blogs/parallel-tool-calls-10.webp)

The mental model that works in practice: think in terms of execution waves. Which calls have no dependencies on anything? Those form wave 1. Which calls depend only on wave 1 results? Those form wave 2. Continue until all calls are assigned to waves. Within each wave, run in parallel. Between waves, wait for all to complete.

When in doubt, profile first. The latency math from Amdahl's Law will tell you if the speedup is worth the implementation complexity. If your tools are all fast (<50ms) and the LLM dominates, parallelism gives you little. If your tools are slow and the LLM is fast (common with smaller models or cached responses), parallelism is your biggest lever.

The [agent action space](/blog/machine-learning/ai-agent/the-agent-action-space) discussion is directly relevant here: agents with narrow, well-defined tools are easier to parallelize safely. Tools that do one thing and read/write from clearly defined resources can be analyzed statically. Tools that have broad, unbounded effects are risky to parallelize because you can't reason about their conflict potential without runtime analysis.

---

## Putting It Together

Parallel tool calls are not a micro-optimization. They're often the difference between an agent that feels instant and one that feels sluggish. The math is straightforward — if you have N independent tool calls averaging 500ms each, sequential execution costs 500N milliseconds while parallel execution costs 500ms. For N=5, that's 2500ms vs 500ms. Users perceive this gap as the difference between "broken" and "excellent."

The engineering discipline is what makes it safe. Build the dependency graph. Annotate your tools with the resources they read and write. Use the topological wave executor. Wrap writes with idempotency or optimistic locking. Coordinate rate limits across workers. Handle partial failures gracefully. Test your parallel paths explicitly — race conditions rarely surface in serial unit tests.

Start with the easy wins: any agent that issues multiple independent read calls is a safe candidate. Move to writes only after you've thought carefully about shared state. And when sequential is the right answer, document why — it's not a performance failure, it's a correctness decision.

The implementation in this post handles the common cases. Adapt it to your tool schemas, your failure semantics, and your rate limits. Profile before and after. The speedup numbers will speak for themselves.

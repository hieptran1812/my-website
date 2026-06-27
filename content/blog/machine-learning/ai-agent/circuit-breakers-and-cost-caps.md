---
title: "Circuit Breakers and Cost Caps: Preventing Runaway Agents from Burning Your Budget"
date: "2026-06-27"
description: "How to implement circuit breakers, cost caps, token budgets, and iteration limits that stop runaway agents before they cause financial or operational damage."
tags: ["ai-agents", "safety", "cost-control", "circuit-breakers", "reliability", "llm", "machine-learning", "production-ml"]
category: "machine-learning"
subcategory: "AI Agent"
author: "Hiep Tran"
featured: true
readTime: 50
---

The first time your production agent runs up a $400 bill overnight on a task that should have cost $0.50, you get religion about cost controls very quickly. The second time it happens — because you only fixed the one obvious loop and missed three others — you start building infrastructure.

This post is about building that infrastructure: the circuit breakers, cost caps, token budgets, iteration limits, velocity detectors, and hierarchical budget systems that make production agents financially safe to operate. Not "I hope this works" safe. Defense-in-depth safe, where the agent would have to breach six independent controls to actually damage your budget.

The diagram below shows the mental model: a circuit breaker that wraps every agent execution, watching three dimensions simultaneously — token consumption, iteration count, and dollar cost — and interrupting execution before any one of them crosses a threshold that would hurt.

![Circuit Breaker State Machine: Closed → Open → Half-Open](/imgs/blogs/circuit-breakers-and-cost-caps-1.webp)

We will build this from first principles, then expand to the full production story: hierarchical budgets, cost attribution, alerting thresholds, graceful degradation, and the tuning process that makes your limits tight enough to protect you without blocking legitimate work.

## 1. The Runaway Agent Problem

Before we build the solution, let us be precise about the failure mode. There are three distinct ways agents blow budgets, and each requires a different kind of control.

**The infinite retry loop.** An agent calls a tool that returns an error — perhaps a rate-limited API or a temporarily unavailable database. The agent, having been trained to be persistent, retries. The retry also fails. The agent retries again. With each retry, it includes the full conversation history in its context window, so token consumption grows linearly with iteration count. After 100 iterations, what should have been a $0.05 tool-retrieval task has consumed $5 in tokens — and the task still isn't done because the tool is still unavailable.

**The context explosion.** An agent is handling a task that involves reading a large document. The first LLM call reads 10,000 tokens of the document. The agent decides it needs more context and reads another 10,000 tokens, appending them to the conversation. Then it calls a web search tool and appends 5,000 more tokens of search results. By the third planning cycle, the context window contains 80,000 tokens of material, nearly all of which is irrelevant to the current planning step. Each LLM call now costs $0.10 instead of $0.005. A task designed for $0.50 has become a $15 task.

**The goal misgeneralization loop.** This is the scariest one. An agent is asked to "find the cheapest flight from Hanoi to Singapore." It calls a flight search API, gets results, but decides the results don't match its quality bar for "cheapest" (maybe the API returned results sorted by price but the agent expected sorted by value-adjusted price). It decides to try a different search strategy. And another. And another. Each strategy involves multiple LLM calls to plan, execute, and evaluate. The agent isn't retrying a failed tool — it's pursuing a goal it can't adequately satisfy, and it will keep pursuing it until something stops it.

![Runaway Agent Spiral: How a Retry Loop Blows the Budget](/imgs/blogs/circuit-breakers-and-cost-caps-2.webp)

The diagram above shows the common shape of all three failure modes: a monotonically increasing cost curve with no natural stopping point. The agent has a task, the task isn't completing satisfactorily, and the agent's only response is to try harder — which means more LLM calls, more tool calls, and more tokens.

The fundamental asymmetry is this: the agent will spend money until it succeeds or until something external stops it. If the task is impossible (the API is permanently down, the goal is underspecified, the document doesn't contain the answer), the agent will spend infinite money. Your job is to be that external stopping mechanism.

### Why built-in model limits aren't enough

Newer model APIs include context window limits and some have configurable max_tokens, but these don't solve the problem. A context window limit causes the API to return an error, which the agent may interpret as a transient failure and retry — now with a truncated context, which may produce a different response that also isn't satisfactory, triggering another loop. Max_tokens controls how long the model's *output* can be, not how much you'll spend across an entire agent session.

What you need is external enforcement: controls that live outside the model and intercept every call before it happens, accumulating state across the entire session.

## 2. The Circuit Breaker Pattern Applied to Agents

The circuit breaker pattern comes from electrical engineering, where a physical breaker interrupts a circuit when the current exceeds a safe level, protecting downstream components from damage. Michael Nygard popularized the software version in *Release It!*: a state machine that wraps a remote call and stops making that call when failures accumulate beyond a threshold.

For agents, we apply the same pattern, but we measure three things instead of one: failure count, total cost, and call velocity.

### The three-state machine

The circuit breaker moves between three states:

**CLOSED** (normal operation): the agent runs freely. Every LLM call goes through. The breaker accumulates metrics — token count, iteration count, cost, call rate — but doesn't interfere.

**OPEN** (circuit tripped): no LLM calls are allowed. Any attempt to make an LLM call is immediately rejected with a `CircuitOpenError`. The agent either catches this and terminates gracefully, or the error propagates up to the task executor which terminates it forcefully.

**HALF-OPEN** (recovery probe): after a cooldown period (typically 60 seconds), the breaker allows one test call through. If that call succeeds without triggering any of the circuit's thresholds, the breaker resets to CLOSED. If the test call triggers a threshold, the breaker returns to OPEN for another cooldown period.

The transition conditions are what make this useful for cost control rather than just failure handling:

| Transition | Condition |
|---|---|
| CLOSED → OPEN | `failure_count > max_failures` OR `total_cost > cost_cap` OR `iteration_count > max_iterations` OR `call_velocity > velocity_limit` |
| OPEN → HALF-OPEN | `time_since_opened > cooldown_seconds` |
| HALF-OPEN → CLOSED | probe call succeeds without triggering any condition |
| HALF-OPEN → OPEN | probe call triggers any condition |

The key insight is that `total_cost > cost_cap` and `iteration_count > max_iterations` are first-class circuit conditions, not afterthoughts. An agent that's spending too fast is just as dangerous as an agent hitting repeated API errors.

### Why cooldown matters

The cooldown period before HALF-OPEN serves a different purpose for cost-circuit-breakers than for failure-circuit-breakers. In a failure context, the cooldown gives the downstream service time to recover. For a cost-circuit-breaker, the cooldown:

1. Forces a human to look at the alert that was fired when the circuit opened
2. Prevents the circuit from immediately re-tripping if it closes and the same problematic task resumes
3. Creates a natural audit window — you can inspect what the agent was doing during the 60 seconds before the circuit opened

For task-level circuits, I typically use a 60-second cooldown for velocity violations, a 5-minute cooldown for cost-cap violations, and no automatic cooldown (require manual reset) for iteration-limit violations, since those usually indicate a more fundamental problem with the agent's goal or the tools it's using.

## 3. Token Budget Enforcement

Token budgets are the most granular form of cost control — they operate at the level of individual LLM calls and accumulate across a session. The right way to think about them is as a resource tank: when you start a task, you allocate a token budget. Every call draws down that budget. When the budget is empty, the task stops.

### Estimating tokens before the call

Before every LLM call, you should estimate the token count of the request. This is easier than it sounds because most modern LLM providers give you access to their tokenizer (or a compatible approximate tokenizer):

```python
import tiktoken  # works for OpenAI-family models
from anthropic import Anthropic

# For Anthropic models, use the official token counting endpoint
# (billed at $0 for the first 100k/month, then normal input rates)
def estimate_tokens(client: Anthropic, messages: list, model: str) -> int:
    response = client.messages.count_tokens(
        model=model,
        messages=messages,
    )
    return response.input_tokens

def check_budget_before_call(
    client: Anthropic,
    messages: list,
    model: str,
    budget_tracker: "BudgetTracker",
    max_output_tokens: int = 4096,
) -> None:
    """Raise BudgetExceededError before the call if the estimated cost would exceed budget."""
    input_tokens = estimate_tokens(client, messages, model)
    estimated_cost = (
        input_tokens * PRICE_PER_INPUT_TOKEN[model]
        + max_output_tokens * PRICE_PER_OUTPUT_TOKEN[model]
    )
    budget_tracker.check_and_reserve(estimated_cost)
```

The `check_and_reserve` call atomically checks whether the budget has enough headroom and, if so, reserves the estimated amount. If not, it raises `BudgetExceededError` before the call is made — so you never make a call that would push you over budget.

### Three horizons of token budgets

Token budgets should be enforced at three levels simultaneously:

**Per-request budget**: the maximum tokens for a single LLM call. This prevents a single call from being absurdly expensive — if you're seeing 50,000-token requests, something has gone wrong with context management. Typical limit: 16,000 tokens for a 128k-context model.

**Per-task budget**: the total tokens allocated across all LLM calls for one task execution. This is the main line of defense against retry loops. Typical limit: 50,000 tokens for a moderate-complexity task.

**Per-session budget**: the rolling token budget for an agent session (might span multiple tasks). This catches agents that are individually within per-task limits but are running more tasks than expected. Typical limit: 200,000 tokens per hour.

These limits compound: a per-request limit of 16k tokens AND a per-task limit of 50k means a task gets at most ~3 LLM calls at maximum request size, or more calls if each call uses fewer tokens. The combination is what gives you real protection.

### Context window management as a budget defense

One underappreciated form of token budget enforcement is active context window management. Instead of letting the context grow unbounded until it hits the model's maximum, you can implement a summarization policy:

```python
class ContextBudgetManager:
    """Manages context window to stay within token budget."""
    
    def __init__(self, max_context_tokens: int = 16_000):
        self.max_context_tokens = max_context_tokens
        self.messages: list[dict] = []
        self.token_count: int = 0
    
    def add_message(self, role: str, content: str, token_count: int) -> None:
        self.messages.append({"role": role, "content": content})
        self.token_count += token_count
        
        # If we're over budget, summarize the oldest messages
        if self.token_count > self.max_context_tokens * 0.8:
            self._compress_context()
    
    def _compress_context(self) -> None:
        """Replace the oldest half of messages with a summary."""
        # Keep the system message and the most recent N messages
        keep_recent = len(self.messages) // 2
        old_messages = self.messages[1:-keep_recent]  # skip system message
        
        # Ask the LLM to summarize the old messages
        # This is itself a token cost, but far cheaper than keeping them all
        summary = self._summarize(old_messages)
        
        self.messages = (
            self.messages[:1]  # system message
            + [{"role": "assistant", "content": f"[Context summary: {summary}]"}]
            + self.messages[-keep_recent:]
        )
        # Recount tokens after compression
        self.token_count = self._count_tokens(self.messages)
```

Context compression isn't free — it costs tokens and can lose information. But it's far cheaper than letting context grow to 128k tokens over 30 retry iterations.

### Token budget vs. dollar budget: which to use?

You need to track both, but they serve different purposes.

**Token budgets** are the right internal control surface. They're model-price-independent, stable across pricing changes, and directly reflect compute consumed. When you set a limit of 50,000 tokens per task, that limit stays meaningful whether the model costs $3/M or $30/M per input token.

**Dollar budgets** are the right user-facing and business control surface. Your SLAs, invoices, and budget conversations are all in dollars. Users don't think in tokens — they think in dollars.

The right architecture enforces both: a token cap per call (prevents context explosion), a token cap per task (prevents retry loops), AND a dollar cap per task (ensures you don't accidentally deploy a cheap agent against an expensive model and have costs jump 10×). When prices change, only the dollar cap needs updating; the token cap remains valid.

A practical rule of thumb: set the dollar cap at the price you'd pay if the token cap were fully consumed at the current model's pricing, with a 20% buffer. If your task token cap is 50,000 tokens and the model costs $3/M input + $15/M output, and you assume a 70/30 input/output split, then: 35,000 × $3/M + 15,000 × $15/M = $0.105 + $0.225 = $0.33. Set the dollar cap at $0.40 (20% buffer). This ensures the two caps are consistent and neither can be circumvented by the other.

## 4. Iteration Caps: Maximum Steps, Tool Calls, LLM Calls

Token budgets catch cost runaway, but iteration limits catch a different failure mode: agents that are making many cheap calls in rapid succession. A hundred 50-token calls is 5,000 tokens — not expensive, but indicative of a loop.

Iteration caps come in three flavors:

**Maximum LLM calls per task**: the total number of times the agent can call the model within a single task execution. This is the most direct defense against retry loops. Setting this at 20–40 for typical tasks means a runaway retry loop will self-limit after at most 40 iterations instead of running forever.

**Maximum tool calls per task**: some tools are expensive (web search with browser automation, database queries that spawn subqueries) even though the LLM call itself is cheap. A separate tool-call limit catches agents that have discovered they can accomplish something via many cheap tool calls.

**Maximum sequential tool failures**: if the last N tool calls have all failed, the task is almost certainly hitting a systematic problem (the API is down, the credentials are wrong, the resource doesn't exist) and continuing will only waste money. A failure streak limit of 3–5 catches this immediately.

```python
class IterationTracker:
    def __init__(
        self,
        max_llm_calls: int = 30,
        max_tool_calls: int = 50,
        max_sequential_failures: int = 3,
    ):
        self.max_llm_calls = max_llm_calls
        self.max_tool_calls = max_tool_calls
        self.max_sequential_failures = max_sequential_failures
        
        self.llm_call_count = 0
        self.tool_call_count = 0
        self.sequential_failure_count = 0
    
    def record_llm_call(self) -> None:
        self.llm_call_count += 1
        if self.llm_call_count >= self.max_llm_calls:
            raise MaxIterationsError(
                f"Task exceeded {self.max_llm_calls} LLM calls. "
                f"Stopping to prevent runaway cost."
            )
    
    def record_tool_call(self, success: bool) -> None:
        self.tool_call_count += 1
        if self.tool_call_count >= self.max_tool_calls:
            raise MaxIterationsError(f"Task exceeded {self.max_tool_calls} tool calls.")
        
        if success:
            self.sequential_failure_count = 0
        else:
            self.sequential_failure_count += 1
            if self.sequential_failure_count >= self.max_sequential_failures:
                raise MaxSequentialFailuresError(
                    f"{self.max_sequential_failures} consecutive tool failures. "
                    f"Stopping — likely a systematic issue (API down, bad credentials)."
                )
```

### Tuning iteration limits without breaking legitimate tasks

The hard part is setting limits that catch runaway agents without also stopping legitimate complex tasks. A research agent genuinely needs more iterations than a simple Q&A agent. My approach is to tier the limits by task type:

| Task class | Max LLM calls | Max tool calls | Rationale |
|---|---|---|---|
| Simple Q&A / lookup | 5 | 3 | Should complete in 1–2 LLM calls; 5 is generous |
| Code generation | 15 | 10 | Plan + write + test + debug cycle |
| Research and synthesis | 30 | 40 | Multiple search + read + synthesize cycles |
| Multi-step workflow | 50 | 80 | Complex orchestration with branching |
| Long-running agent | 100 | 200 | Explicit override required, with extra monitoring |

The key discipline: every new task type gets an explicit limit assignment before it goes to production, not an implicit "inherit the default." Defaults are for task types you forgot to classify — they should be conservatively low.

## 5. Cost Estimation: Predicting Before Starting

The most underutilized cost control is pre-execution cost estimation. Before starting a task, you can estimate its likely cost and refuse to run it if the estimate exceeds the budget. This catches obviously-expensive tasks before they waste a single dollar.

Cost estimation works best for tasks with well-defined structure:

```python
from dataclasses import dataclass
from typing import Optional

@dataclass  
class CostEstimate:
    min_cost_usd: float
    expected_cost_usd: float
    max_cost_usd: float
    confidence: str  # "high" | "medium" | "low"
    reasoning: str

PRICES = {
    "claude-3-5-sonnet-20241022": {
        "input": 3.00 / 1_000_000,   # $3 per million input tokens
        "output": 15.00 / 1_000_000,  # $15 per million output tokens
    },
    "claude-3-haiku-20240307": {
        "input": 0.25 / 1_000_000,
        "output": 1.25 / 1_000_000,
    },
}

def estimate_task_cost(
    task_type: str,
    input_size_tokens: int,
    model: str,
    expected_iterations: int,
    max_output_tokens_per_call: int = 4096,
) -> CostEstimate:
    price = PRICES[model]
    
    # Each iteration: input grows with history, output is fixed per call
    # Approximate: context grows by output_tokens each iteration
    total_input = sum(
        input_size_tokens + i * max_output_tokens_per_call
        for i in range(expected_iterations)
    )
    total_output = expected_iterations * max_output_tokens_per_call
    
    expected_cost = (
        total_input * price["input"]
        + total_output * price["output"]
    )
    
    return CostEstimate(
        min_cost_usd=expected_cost * 0.3,  # optimistic: short outputs
        expected_cost_usd=expected_cost,
        max_cost_usd=expected_cost * 3.0,  # pessimistic: retry overhead
        confidence="medium",
        reasoning=(
            f"{expected_iterations} LLM calls × "
            f"avg {total_input//expected_iterations:,} input tokens, "
            f"{max_output_tokens_per_call:,} output tokens at {model}"
        ),
    )
```

For tasks where you don't know the expected number of iterations, you can use historical data: record the p50, p90, and p99 iteration count for each task type over the last 30 days, and use those to produce a distribution-based estimate.

### Using estimate confidence to gate task admission

When pre-execution cost estimates have low confidence — because the task type is new, the input length is highly variable, or the required number of iterations is genuinely unknown — you have two options: refuse the task, or accept it with a tighter default cap to limit your downside.

```python
def task_admission_decision(
    estimate: CostEstimate,
    task_cap: float,
) -> tuple[bool, float]:
    """
    Returns (admitted: bool, effective_cap: float).
    Tighter cap when confidence is low.
    """
    if estimate.max_cost_usd <= task_cap:
        # Even worst case is within budget: admit with full cap
        return True, task_cap
    
    if estimate.expected_cost_usd > task_cap:
        # Expected case already over budget: reject
        return False, 0.0
    
    # Expected is under budget but worst case is over
    # Admit with a tighter cap equal to 1.2× expected cost
    if estimate.confidence == "low":
        tight_cap = min(task_cap, estimate.expected_cost_usd * 1.2)
        return True, tight_cap
    
    return True, task_cap
```

This approach lets you run tasks with uncertain cost profiles while limiting the damage from a bad estimate. The cost is that some tasks that would have succeeded with the full cap will hit the tighter cap — but that's a tradeoff in favor of predictable spend.

### Dynamic repricing mid-task

Pre-execution estimation only gets you so far — you can't predict retry storms. Dynamic repricing checks the running cost against the budget at regular intervals during execution:

```python
class DynamicCostGuard:
    """Checks cost against budget at every LLM call."""
    
    def __init__(self, cost_cap: float, warn_threshold: float = 0.8):
        self.cost_cap = cost_cap
        self.warn_threshold = warn_threshold
        self.spent = 0.0
        self.callbacks: list = []
    
    def record_call_cost(self, input_tokens: int, output_tokens: int, model: str) -> None:
        call_cost = (
            input_tokens * PRICES[model]["input"]
            + output_tokens * PRICES[model]["output"]
        )
        self.spent += call_cost
        
        ratio = self.spent / self.cost_cap
        
        if ratio >= 1.0:
            raise CostCapExceededError(
                f"Task cost ${self.spent:.4f} exceeded cap ${self.cost_cap:.4f}. "
                f"Stopping to prevent further spend."
            )
        elif ratio >= self.warn_threshold:
            self._fire_warning(ratio)
    
    def _fire_warning(self, ratio: float) -> None:
        for cb in self.callbacks:
            cb(spent=self.spent, cap=self.cost_cap, ratio=ratio)
```

The 80% warning threshold is important: it gives the agent time to complete its current step gracefully before the hard limit hits at 100%. An agent that's at 80% of budget should start winding down — completing the current tool call, summarizing progress, returning a partial result — rather than suddenly hitting a hard stop mid-sentence.

## 6. Velocity Limits: Detecting Abnormal Spending Rate

A cost cap catches total spend, but it doesn't catch spend *rate*. An agent spending $0.40 over 8 hours is fine. An agent spending $0.40 in 30 seconds is probably in a loop.

Velocity limits measure the rate of token consumption or cost accumulation and fire an alert — or trigger the circuit breaker — when that rate exceeds a multiple of the baseline.

![Cap Types vs Enforcement Point Matrix](/imgs/blogs/circuit-breakers-and-cost-caps-5.webp)

```python
import time
from collections import deque

class VelocityMonitor:
    """Sliding-window velocity monitor for LLM spend rate."""
    
    def __init__(
        self,
        window_seconds: int = 60,
        baseline_tokens_per_second: float = 50.0,
        alert_multiplier: float = 3.0,
        critical_multiplier: float = 10.0,
    ):
        self.window_seconds = window_seconds
        self.baseline = baseline_tokens_per_second
        self.alert_multiplier = alert_multiplier
        self.critical_multiplier = critical_multiplier
        
        # Ring buffer: (timestamp, tokens) pairs
        self.window: deque[tuple[float, int]] = deque()
    
    def record_call(self, tokens: int) -> str:
        """Record a call and return alert level: 'normal', 'warn', 'critical'."""
        now = time.time()
        self.window.append((now, tokens))
        
        # Expire old entries
        cutoff = now - self.window_seconds
        while self.window and self.window[0][0] < cutoff:
            self.window.popleft()
        
        # Compute current rate
        total_tokens = sum(t for _, t in self.window)
        elapsed = max(now - self.window[0][0], 1.0) if self.window else 1.0
        current_rate = total_tokens / elapsed
        
        multiplier = current_rate / max(self.baseline, 1.0)
        
        if multiplier >= self.critical_multiplier:
            return "critical"
        elif multiplier >= self.alert_multiplier:
            return "warn"
        return "normal"
```

A sliding window is important here: a fixed-period window would miss a burst that straddles a window boundary. The sliding window always represents the most recent N seconds regardless of when you query it.

### What baseline to use

The hardest part of velocity monitoring is choosing the right baseline. Options:

**Static baseline**: set once based on expected task cost. Simple, but breaks when you add new expensive task types.

**Per-task-type baseline**: maintain a separate baseline for each task type. More accurate, but requires enough historical data per task type to be meaningful.

**Rolling p50 of recent calls**: compute the p50 token rate over the last N completed tasks and use that as the baseline. Self-adjusting, but can drift upward if your agents gradually get more expensive over time (because you're adding more complex tasks).

My recommendation: start with a static baseline set to 2–3× the average token rate of your most common task type. After 2–4 weeks of production data, switch to per-task-type baselines.

### Velocity vs. burstiness

Velocity monitoring catches sustained fast spend. It is less effective at catching burst-then-idle patterns — an agent that makes 50 LLM calls in 10 seconds, then sits idle for 50 seconds, then makes 50 more calls. The 60-second sliding window will see an average of 50 calls/minute, which might be within your velocity threshold even though the burst pattern is clearly abnormal.

For burst detection, add a separate short-window check:

```python
class BurstDetector:
    """Detects short-window token bursts that velocity averaging would miss."""
    
    def __init__(
        self,
        burst_window_seconds: float = 5.0,
        burst_token_threshold: int = 10_000,
    ):
        self.burst_window = burst_window_seconds
        self.burst_threshold = burst_token_threshold
        from collections import deque
        self._window: deque = deque()
    
    def record(self, tokens: int) -> bool:
        """Returns True if a burst is detected."""
        now = time.time()
        self._window.append((now, tokens))
        cutoff = now - self.burst_window
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()
        total = sum(t for _, t in self._window)
        return total >= self.burst_threshold
```

A burst of 10,000 tokens in 5 seconds is a strong signal of a retry loop — no legitimate task should need that many tokens that quickly. Burst detection and velocity monitoring are complementary: velocity catches sustained high-rate spend, burst detection catches the initial spike of a retry storm before it sustains long enough to cross the velocity threshold.

## 7. Graceful Degradation Under Caps

The worst possible outcome when a cost cap fires is a hard crash that loses all the work the agent has done so far. The best outcome is the agent returning a partial result with metadata explaining exactly where it stopped, what it accomplished, and what it would take to continue.

![Graceful Degradation Pipeline](/imgs/blogs/circuit-breakers-and-cost-caps-7.webp)

Graceful degradation requires two things: checkpointing and partial-result scoring.

### Checkpointing agent state

A checkpoint is a serializable snapshot of everything the agent needs to resume from the current point:

```python
@dataclass
class AgentCheckpoint:
    task_id: str
    task_description: str
    completed_steps: list[dict]
    current_step_index: int
    total_steps_planned: int
    intermediate_results: dict
    cost_spent: float
    tokens_used: int
    timestamp: float
    
    def to_partial_result(self) -> "PartialResult":
        return PartialResult(
            task_id=self.task_id,
            completed_steps=self.completed_steps,
            completion_fraction=self.current_step_index / max(self.total_steps_planned, 1),
            intermediate_results=self.intermediate_results,
            cost_usd=self.cost_spent,
            tokens_used=self.tokens_used,
            stopped_reason="budget_exhausted",
            resume_checkpoint=self,
        )
```

Every agent that runs in production should checkpoint at regular intervals — at minimum, after every successfully completed step. This serves two purposes: it enables graceful degradation when cost caps fire, and it enables resumption if the agent process crashes.

Store checkpoints in a durable store (Redis with a 24-hour TTL is typical; Postgres for longer-lived tasks). The checkpoint storage cost is negligible compared to the LLM cost of starting over.

### Partial result scoring

Not all partial results are equal. Completing 9 of 10 steps of a research task is nearly as useful as completing all 10. Completing 1 of 10 steps is nearly useless. A partial result scorer decides whether to return the partial result or simply fail:

```python
def score_partial_result(checkpoint: AgentCheckpoint) -> float:
    """Return a score 0.0-1.0 indicating how useful this partial result is."""
    # Basic fraction of steps completed
    base_score = checkpoint.current_step_index / max(checkpoint.total_steps_planned, 1)
    
    # Bonus if key intermediate results are present
    key_results_present = len(checkpoint.intermediate_results) / max(checkpoint.total_steps_planned, 1)
    
    # Combined score
    return 0.7 * base_score + 0.3 * key_results_present

PARTIAL_RESULT_THRESHOLD = 0.25  # return partial if at least 25% done

def handle_budget_exhausted(checkpoint: AgentCheckpoint) -> "TaskResult":
    score = score_partial_result(checkpoint)
    
    if score >= PARTIAL_RESULT_THRESHOLD:
        partial = checkpoint.to_partial_result()
        return TaskResult(
            status="partial",
            result=partial.intermediate_results,
            metadata={
                "completion_fraction": partial.completion_fraction,
                "cost_usd": partial.cost_usd,
                "stopped_reason": "budget_exhausted",
                "resume_available": True,
                "resume_checkpoint_id": checkpoint.task_id,
            },
        )
    else:
        return TaskResult(
            status="failed",
            result=None,
            metadata={
                "stopped_reason": "budget_exhausted_too_early",
                "completion_fraction": score,
                "cost_usd": checkpoint.cost_spent,
            },
        )
```

The threshold of 25% (at least 1 of 4 steps done) is a heuristic. For some task types — especially those where the first step is "gather all inputs" — even 90% completion without the final synthesis step is not useful. You'll want to override the threshold per task type as you learn which partial results users actually find useful.

## 8. Hierarchical Budget: Per-Agent, Per-User, Per-Tenant, Global

In a production multi-tenant system, individual task-level budgets aren't enough. One user running many tasks can exhaust a global budget that affects other tenants. One tenant running a poorly designed workflow can exhaust their per-tenant budget in an hour, then complain that the system is broken.

The solution is a hierarchy of budget layers, where each layer independently enforces a limit:

![Hierarchical Budget Stack](/imgs/blogs/circuit-breakers-and-cost-caps-3.webp)

Each layer in the hierarchy uses a token bucket or quota system:

```python
from threading import Lock
import time

class BudgetLayer:
    """Thread-safe budget tracker for one layer of the hierarchy."""
    
    def __init__(self, cap_usd: float, window_seconds: int = 86400):
        self.cap_usd = cap_usd
        self.window_seconds = window_seconds
        self.spent_usd: float = 0.0
        self.window_start: float = time.time()
        self._lock = Lock()
    
    def check_and_spend(self, amount_usd: float) -> None:
        """Atomically check budget and deduct. Raises BudgetExceededError if over cap."""
        with self._lock:
            now = time.time()
            # Reset window if expired
            if now - self.window_start >= self.window_seconds:
                self.spent_usd = 0.0
                self.window_start = now
            
            if self.spent_usd + amount_usd > self.cap_usd:
                raise BudgetExceededError(
                    f"Budget exhausted: ${self.spent_usd:.4f} + ${amount_usd:.4f} "
                    f"> ${self.cap_usd:.4f} cap"
                )
            self.spent_usd += amount_usd
    
    def remaining(self) -> float:
        with self._lock:
            return max(0.0, self.cap_usd - self.spent_usd)

class HierarchicalBudget:
    """Enforces budgets at all levels simultaneously."""
    
    def __init__(
        self,
        global_daily_cap: float = 500.0,
        tenant_daily_cap: float = 50.0,
        user_daily_cap: float = 5.0,
        agent_session_cap: float = 2.0,
        task_cap: float = 0.50,
        call_cap: float = 0.10,
    ):
        self.global_budget = BudgetLayer(global_daily_cap)
        self.tenant_budgets: dict[str, BudgetLayer] = {}
        self.user_budgets: dict[str, BudgetLayer] = {}
        self.agent_cap = agent_session_cap
        self.task_cap = task_cap
        self.call_cap = call_cap
    
    def check_call(
        self,
        tenant_id: str,
        user_id: str,
        call_cost: float,
    ) -> None:
        """Check all budget layers before making an LLM call."""
        # Most granular first — fail fast on per-call cap
        if call_cost > self.call_cap:
            raise BudgetExceededError(
                f"Single call cost ${call_cost:.4f} exceeds per-call cap ${self.call_cap:.4f}"
            )
        
        # Per-user budget
        if user_id not in self.user_budgets:
            self.user_budgets[user_id] = BudgetLayer(5.0)
        self.user_budgets[user_id].check_and_spend(call_cost)
        
        # Per-tenant budget
        if tenant_id not in self.tenant_budgets:
            self.tenant_budgets[tenant_id] = BudgetLayer(50.0)
        self.tenant_budgets[tenant_id].check_and_spend(call_cost)
        
        # Global budget — most expensive to check, do last
        self.global_budget.check_and_spend(call_cost)
```

The ordering matters: check the most granular limits first. This is both faster (per-call limits fail immediately) and safer (it prevents the common case where one user exhausts the global budget before you can stop them).

### Rate limiting vs. hard limits

For user-facing systems, hard cuts — where the agent suddenly fails — are often a bad user experience. Consider using rate limiting for the warning zone and hard cuts only for the critical zone:

| Budget remaining | Behavior |
|---|---|
| > 20% | Normal operation |
| 10–20% | Warning notification to user; agent continues |
| 5–10% | Rate-limit: introduce 500ms delay between LLM calls to allow user to intervene |
| < 5% | Hard stop, return partial result, require explicit user action to continue |

The delay in the 5–10% zone is a useful trick: it slows down the spend rate without stopping the task, giving the user time to see the warning and decide whether to intervene.

## 9. Cost Attribution and Chargeback

You cannot control what you cannot see, and you cannot see what you cannot attribute. Every LLM call should carry tags that allow you to trace it back to the agent, user, task, and tenant that caused it.

![Cost Attribution Flow](/imgs/blogs/circuit-breakers-and-cost-caps-8.webp)

The key principle is attribution at call time, not batch time. If you log the LLM call and then try to join it with your task records later, you'll discover that task IDs aren't always passed through correctly, agent IDs are sometimes missing from logs, and any call that failed mid-task may have no record at all. Tag the call before you make it.

```python
@dataclass
class CallMetadata:
    agent_id: str
    task_id: str
    user_id: str
    tenant_id: str
    task_type: str
    model: str
    timestamp: float

class AttributedLLMClient:
    """Wraps an LLM client to inject attribution metadata and write to cost ledger."""
    
    def __init__(self, client, ledger: "CostLedger", budget: HierarchicalBudget):
        self.client = client
        self.ledger = ledger
        self.budget = budget
    
    def complete(
        self,
        messages: list,
        metadata: CallMetadata,
        **kwargs,
    ) -> dict:
        # Check budget before the call
        estimated_cost = self._estimate_cost(messages, metadata.model, kwargs)
        self.budget.check_call(
            tenant_id=metadata.tenant_id,
            user_id=metadata.user_id,
            call_cost=estimated_cost,
        )
        
        # Make the call
        response = self.client.messages.create(messages=messages, **kwargs)
        
        # Record actual cost
        actual_cost = self._compute_cost(response, metadata.model)
        self.ledger.record(
            metadata=metadata,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=actual_cost,
        )
        
        # Update budgets with actual cost (correct the reservation)
        # If actual < estimated, the difference is returned to the budget
        self.budget.settle_call(
            tenant_id=metadata.tenant_id,
            user_id=metadata.user_id,
            estimated=estimated_cost,
            actual=actual_cost,
        )
        
        return response
```

### The cost ledger

The cost ledger is an append-only log of every attributed LLM call. Its schema should be optimized for the queries you'll actually run:

```sql
CREATE TABLE llm_cost_events (
    id           BIGSERIAL PRIMARY KEY,
    timestamp    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id    TEXT NOT NULL,
    user_id      TEXT NOT NULL,
    agent_id     TEXT NOT NULL,
    task_id      TEXT NOT NULL,
    task_type    TEXT NOT NULL,
    model        TEXT NOT NULL,
    input_tokens  INT NOT NULL,
    output_tokens INT NOT NULL,
    cost_usd     NUMERIC(10, 6) NOT NULL
);

-- Indexes for the queries you'll actually run:
CREATE INDEX idx_cost_tenant_time ON llm_cost_events (tenant_id, timestamp);
CREATE INDEX idx_cost_user_time   ON llm_cost_events (user_id, timestamp);
CREATE INDEX idx_cost_task        ON llm_cost_events (task_id);
```

For high-volume systems (> 10,000 LLM calls per day), consider a time-series database (InfluxDB, TimescaleDB) instead of Postgres. The write pattern is append-only and the query pattern is time-range aggregations — an ideal fit for time-series storage.

## 10. Alerting and On-Call: What Thresholds Warrant Waking Someone Up

Not every budget event is worth waking up an engineer at 2 AM. The challenge is calibrating alerts so that critical events always fire and routine noise never fires.

![Alert Threshold Calibration Matrix](/imgs/blogs/circuit-breakers-and-cost-caps-9.webp)

The matrix above shows the calibration I use for a typical production agent deployment. The key principles:

**Warning alerts go to Slack, critical alerts go to PagerDuty.** A warning should be something an on-call engineer can investigate at their leisure during business hours. A critical alert is something that requires immediate action — typically because money is being spent right now at a rate that will cause real financial damage if not stopped.

**Rate-based alerts are more urgent than total-cost alerts.** A task that has spent $0.40 out of a $0.50 budget is fine — it's nearly done. A task spending $0.15/minute is spending at 3× the baseline rate and is almost certainly in a loop, regardless of how much it has spent so far.

**Distinguish between per-task alerts and system-level alerts.** A single task hitting its budget cap is not a critical alert — it's expected behavior of the circuit breaker. A system-level alert fires when the global daily budget is 80% consumed before noon, when the per-tenant velocity is 10× baseline, or when the circuit breaker trip rate is above 5% of tasks (indicating a systemic problem with agent quality).

### Alert escalation chain

```python
from enum import Enum

class AlertLevel(Enum):
    INFO = "info"      # Log only, no human notification
    WARNING = "warn"   # Slack notification to #agent-alerts
    CRITICAL = "crit"  # PagerDuty alert, immediate human response required

@dataclass
class AlertRule:
    name: str
    condition: str
    level: AlertLevel
    message_template: str
    cooldown_minutes: int  # Don't re-fire same alert within this window

ALERT_RULES = [
    AlertRule(
        name="task_budget_80pct",
        condition="task_spent >= 0.8 * task_cap",
        level=AlertLevel.WARNING,
        message_template="Task {task_id} is at {pct:.0f}% of budget (${spent:.3f}/${cap:.3f})",
        cooldown_minutes=5,
    ),
    AlertRule(
        name="velocity_3x_baseline",
        condition="velocity >= 3 * baseline",
        level=AlertLevel.WARNING,
        message_template="Agent {agent_id} spend rate {rate:.2f}/min is 3× baseline",
        cooldown_minutes=15,
    ),
    AlertRule(
        name="velocity_10x_baseline",
        condition="velocity >= 10 * baseline",
        level=AlertLevel.CRITICAL,
        message_template="CRITICAL: Agent {agent_id} spend rate {rate:.2f}/min is 10× baseline — circuit may be open",
        cooldown_minutes=60,
    ),
    AlertRule(
        name="circuit_breaker_open",
        condition="circuit_state == 'OPEN'",
        level=AlertLevel.CRITICAL,
        message_template="Circuit breaker OPEN for agent {agent_id}: {reason}. Cost ${cost:.3f}",
        cooldown_minutes=60,
    ),
    AlertRule(
        name="global_budget_80pct",
        condition="global_spent >= 0.8 * global_cap",
        level=AlertLevel.CRITICAL,
        message_template="Global daily budget at {pct:.0f}% (${spent:.0f}/${cap:.0f}) — review all active agents",
        cooldown_minutes=30,
    ),
]
```

The cooldown on the circuit-breaker-open alert deserves explanation: 60 minutes might seem long, but circuit-breaker events are typically investigated by the same on-call engineer. Re-paging them every time the circuit re-opens (because the half-open probe keeps failing) creates alert fatigue and risks training engineers to ignore the alert.

## 11. Python Implementation: Agent Executor with Circuit Breaker and Budget Tracking

Here is a complete, production-ready agent executor that wires together everything we've discussed: circuit breaker state machine, hierarchical budget checks, velocity monitoring, iteration limits, checkpointing, and graceful degradation.

```python
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import anthropic

logger = logging.getLogger(__name__)

# ─── Exceptions ──────────────────────────────────────────────────────────────

class BudgetExceededError(Exception):
    pass

class MaxIterationsError(Exception):
    pass

class CircuitOpenError(Exception):
    pass

# ─── Circuit Breaker ─────────────────────────────────────────────────────────

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    cost_cap: float = 0.50
    iteration_cap: int = 30
    cooldown_seconds: float = 60.0

    state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    failure_count: int = field(default=0, init=False)
    total_cost: float = field(default=0.0, init=False)
    iteration_count: int = field(default=0, init=False)
    opened_at: Optional[float] = field(default=None, init=False)

    def before_call(self) -> None:
        if self.state == CircuitState.OPEN:
            elapsed = time.time() - (self.opened_at or 0)
            if elapsed >= self.cooldown_seconds:
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit half-open: probing")
            else:
                raise CircuitOpenError(
                    f"Circuit OPEN. Retry in {self.cooldown_seconds - elapsed:.0f}s"
                )

        if self.total_cost >= self.cost_cap:
            self._trip(f"cost cap ${self.cost_cap:.3f} exceeded (${self.total_cost:.3f} spent)")
        if self.iteration_count >= self.iteration_cap:
            self._trip(f"iteration cap {self.iteration_cap} exceeded")

    def after_call(self, cost: float, success: bool) -> None:
        self.total_cost += cost
        self.iteration_count += 1

        if not success:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self._trip(f"{self.failure_count} consecutive failures")
        else:
            self.failure_count = 0
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                logger.info("Circuit closed: probe succeeded")

    def _trip(self, reason: str) -> None:
        self.state = CircuitState.OPEN
        self.opened_at = time.time()
        logger.warning(f"Circuit OPEN: {reason}")
        raise CircuitOpenError(f"Circuit opened: {reason}")

# ─── Budget Tracker ───────────────────────────────────────────────────────────

@dataclass
class BudgetTracker:
    task_cap: float = 0.50
    warn_at: float = 0.80  # fraction of cap

    spent: float = field(default=0.0, init=False)
    _warned: bool = field(default=False, init=False)

    def record(self, cost: float) -> None:
        self.spent += cost
        ratio = self.spent / self.task_cap
        if ratio >= 1.0:
            raise BudgetExceededError(
                f"Task budget ${self.task_cap:.3f} exceeded "
                f"(${self.spent:.4f} spent)"
            )
        if ratio >= self.warn_at and not self._warned:
            logger.warning(
                f"Task budget {ratio:.0%} consumed "
                f"(${self.spent:.4f} / ${self.task_cap:.3f})"
            )
            self._warned = True

# ─── Velocity Monitor ────────────────────────────────────────────────────────

class VelocityMonitor:
    def __init__(
        self,
        window_s: int = 60,
        baseline_tok_per_s: float = 100.0,
        alert_mult: float = 3.0,
    ):
        self.window_s = window_s
        self.baseline = baseline_tok_per_s
        self.alert_mult = alert_mult
        from collections import deque
        self._window: deque = deque()

    def record(self, tokens: int) -> str:
        now = time.time()
        self._window.append((now, tokens))
        cutoff = now - self.window_s
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()
        total = sum(t for _, t in self._window)
        elapsed = max(now - self._window[0][0], 1.0) if self._window else 1.0
        rate = total / elapsed
        mult = rate / max(self.baseline, 1.0)
        if mult >= self.alert_mult:
            logger.warning(f"Velocity {rate:.0f} tok/s is {mult:.1f}× baseline")
            return "warn"
        return "normal"

# ─── Agent Executor ───────────────────────────────────────────────────────────

PRICES = {
    "claude-3-5-sonnet-20241022": {"input": 3e-6, "output": 15e-6},
    "claude-3-haiku-20240307":    {"input": 0.25e-6, "output": 1.25e-6},
}

def call_cost(model: str, in_toks: int, out_toks: int) -> float:
    p = PRICES.get(model, {"input": 5e-6, "output": 15e-6})
    return in_toks * p["input"] + out_toks * p["output"]

@dataclass
class AgentExecutor:
    client: Any
    model: str = "claude-3-5-sonnet-20241022"
    circuit: CircuitBreaker = field(default_factory=CircuitBreaker)
    budget: BudgetTracker = field(default_factory=BudgetTracker)
    velocity: VelocityMonitor = field(default_factory=VelocityMonitor)

    messages: list = field(default_factory=list, init=False)
    steps_done: list = field(default_factory=list, init=False)

    def run(self, task: str) -> dict:
        self.messages = [{"role": "user", "content": task}]
        try:
            return self._loop()
        except (CircuitOpenError, BudgetExceededError, MaxIterationsError) as exc:
            logger.error(f"Task stopped: {exc}")
            return self._partial_result(str(exc))

    def _loop(self) -> dict:
        while True:
            self.circuit.before_call()
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=self.messages,
            )
            usage = response.usage
            cost = call_cost(self.model, usage.input_tokens, usage.output_tokens)
            self.velocity.record(usage.input_tokens + usage.output_tokens)
            self.budget.record(cost)
            self.circuit.after_call(cost, success=True)

            content = response.content[0].text
            self.messages.append({"role": "assistant", "content": content})
            self.steps_done.append(content)

            if response.stop_reason == "end_turn":
                return {"status": "complete", "result": content,
                        "cost_usd": self.budget.spent,
                        "iterations": self.circuit.iteration_count}

    def _partial_result(self, reason: str) -> dict:
        return {
            "status": "partial",
            "result": self.steps_done[-1] if self.steps_done else None,
            "steps_completed": len(self.steps_done),
            "cost_usd": self.budget.spent,
            "stopped_reason": reason,
        }
```

This executor handles the common case. In production you'll want to extend it with: tool-call support with per-tool retry limits, Redis-based checkpoint persistence, async execution with asyncio-safe locks, and per-call metadata tagging for the cost ledger.

## 11b. Testing Your Budget Controls Before Production

The most dangerous illusion in production-readiness is "we have the controls implemented, therefore we are protected." Budget controls are code, and code has bugs. More importantly, budget controls interact with agent behavior in non-obvious ways — the controls may be logically correct but trigger at the wrong moment, cause the agent to enter an unexpected error-handling path, or fail silently because an exception was caught by the wrong handler.

You need to deliberately break things in staging before anything reaches production.

### Chaos engineering for budget controls

The principle is simple: inject failures that should trigger your budget controls, and verify that they actually do. Specifically:

**Test 1: Force an infinite retry loop.** Create a mock tool that always returns an error. Point an agent at a task that uses this tool. Verify that the iteration cap fires within `max_iterations` calls, that a partial result is returned, and that the total cost is within `max_iterations × max_cost_per_call`.

```python
class AlwaysFailTool:
    """Mock tool for testing iteration limits."""
    call_count = 0
    
    def __call__(self, *args, **kwargs):
        self.call_count += 1
        raise ToolError(f"Simulated failure #{self.call_count}")

def test_iteration_cap_fires():
    tool = AlwaysFailTool()
    executor = AgentExecutor(
        client=real_anthropic_client,
        circuit=CircuitBreaker(iteration_cap=5),
        budget=BudgetTracker(task_cap=1.00),
    )
    result = executor.run("Use the search tool to find X", tools=[tool])
    assert result["status"] in ("partial", "failed")
    assert tool.call_count <= 5, f"Tool called {tool.call_count} times, expected ≤ 5"
    assert result["cost_usd"] < 0.50, "Should not have spent more than $0.50 on a 5-iteration test"
```

**Test 2: Inject a cost spike.** Substitute a fake LLM client that reports artificially high token counts. Verify that the cost cap fires before the total cost exceeds the cap.

```python
class HighCostFakeLLM:
    """Mock LLM that reports 100k tokens per call."""
    
    def messages_create(self, *args, **kwargs):
        return FakeResponse(
            content="fake response",
            usage=FakeUsage(input_tokens=50_000, output_tokens=50_000)
        )

def test_cost_cap_fires():
    executor = AgentExecutor(
        client=HighCostFakeLLM(),
        budget=BudgetTracker(task_cap=0.50),
    )
    result = executor.run("Do a task")
    assert result["status"] in ("partial", "failed")
    # At $3/M input + $15/M output × 100k tokens = $1.80 per call
    # Should fire on the first call since $1.80 > $0.50
    assert result["iterations"] <= 1
```

**Test 3: Verify partial result quality.** For each task type, verify that when the circuit breaker fires mid-task, the returned partial result contains actionable data, not just a failure message.

**Test 4: Test the alert routing.** Fire a circuit-breaker trip event in staging and verify that the Slack notification arrives within 2 minutes and the PagerDuty alert fires for critical-level events. Test this monthly — alerting infrastructure can silently fail (channel renamed, webhook rotated, Slack app disconnected).

### Load testing budget enforcement

Individual task tests aren't enough — you also need to verify that budget enforcement holds under concurrency. If 10 tasks are running simultaneously and each tries to spend its full $0.50 budget, the total spend should not exceed $5 × (number of permitted concurrent tasks), even if all 10 tasks spike at the exact same second.

```python
import asyncio

async def concurrent_budget_test(budget: HierarchicalBudget, n_tasks: int = 20):
    """Launch N concurrent tasks and verify total spend stays within budget."""
    total_spent = 0.0
    errors = []
    
    async def run_one_task(task_id: str):
        nonlocal total_spent
        try:
            budget.check_call(
                tenant_id="test-tenant",
                user_id=f"user-{task_id}",
                call_cost=0.50,  # each task tries to spend $0.50
            )
            total_spent += 0.50
        except BudgetExceededError as e:
            errors.append(str(e))
    
    await asyncio.gather(*[run_one_task(f"task-{i}") for i in range(n_tasks)])
    
    # With a per-user cap of $5, all 20 tasks should succeed
    # With a per-tenant cap of $5, only 10 should succeed
    assert total_spent <= budget.tenant_budgets.get("test-tenant", BudgetLayer(999)).cap_usd
    return total_spent, errors
```

This test regularly surfaces race conditions in budget tracking code — cases where two concurrent calls both read the budget as having headroom, both spend, and the total exceeds the cap. The fix is atomic check-and-decrement (using database transactions or Redis atomic operations, not in-memory Python objects).

## 11c. Observability: What to Measure

Budget controls are only as good as the data they produce. For every circuit breaker trip or budget cap event, you should capture enough context to answer: *why did this trip, and could we have caught it earlier?*

The minimum telemetry set:

```python
@dataclass
class CircuitTripEvent:
    """Emitted whenever the circuit breaker opens."""
    timestamp: float
    agent_id: str
    task_id: str
    user_id: str
    tenant_id: str
    task_type: str
    trip_reason: str          # "cost_cap" | "iteration_cap" | "failure_count" | "velocity"
    total_cost_at_trip: float
    iterations_at_trip: int
    tokens_at_trip: int
    velocity_at_trip: float   # tokens/second at the moment of trip
    task_age_seconds: float   # how long the task had been running
    partial_result_available: bool
    last_tool_calls: list[str]  # names of the last 5 tool calls
    last_error: str | None    # most recent error message, if any
```

These fields enable the queries that matter for debugging and tuning:

- "Which task types have the highest circuit-trip rate?" → sort by `task_type`, filter `trip_reason == "iteration_cap"`
- "Are trips happening early in task lifetime or late?" → histogram of `task_age_seconds` at trip
- "Is one agent causing most of the trips?" → group by `agent_id`
- "Are trips concentrated in specific users/tenants?" → group by `user_id`, `tenant_id`
- "What was the agent doing right before the trip?" → inspect `last_tool_calls` and `last_error`

Without these fields, a circuit trip is just a noise event in your logs. With them, it's a diagnostic signal that tells you whether the problem is in the agent logic, the tool implementation, the budget calibration, or the user behavior.

## 12. Case Studies

### Case Study 1: The $3,200 Overnight Research Agent

A team deployed a research agent designed to monitor competitor pricing. The agent was supposed to run once per hour, fetch 5–10 URLs, summarize them, and write a report. The tool that fetched URLs had a bug: it would occasionally hang for 30 seconds before returning a timeout error. The agent, not having been given a retry limit, would retry the failed URL up to 100 times.

At 3 AM, a particularly slow period for the target website, nearly every URL fetch timed out. The agent ran 100-iteration retry loops on each of 8 URLs, generating 800 LLM calls. Each call included the full conversation history (which by the 50th retry was 40,000 tokens long), so the average call cost was $0.12. Total bill: $96 in one hour, $3,200 overnight.

The fix applied three controls simultaneously: a max_retries=3 on the URL fetch tool, a max_iterations=40 on the task, and a per-task cost cap of $2. Any one of these would have caught the problem.

**Lesson**: tool-call retry limits are as important as LLM-call limits. The agent's looping wasn't at the LLM level — it was at the tool level, with the LLM faithfully (and expensively) processing each retry's failure.

### Case Study 2: The Context Window Inflation Attack

A B2B SaaS company's customer-facing agent allowed users to paste text for the agent to analyze. A customer (inadvertently) pasted a 50,000-word legal document as the input to a "summarize this paragraph" task. The agent, receiving 50,000 tokens of context, made its first call — which cost $0.15 due to input length — then decided it needed to re-read the document for its second reasoning step, appending another 50,000 tokens. By the third LLM call, the context was 150,000 tokens, costing $0.45 per call.

The task had a per-task cap of $1.00, so it only ran 2–3 calls before hitting the cap. But the user got charged $0.90 for a task they expected to cost $0.02, and the partial result was useless because the agent never reached the summarization step.

The real fix was input-length validation at task admission: reject tasks where the total input exceeds 10,000 tokens, or require explicit user confirmation for large inputs with a cost estimate displayed upfront.

**Lesson**: pre-execution cost estimation is especially valuable for tasks with variable-length user input. The budget cap caught the damage, but a pre-admission check would have given a better user experience.

### Case Study 3: The Half-Open Loop

A cloud infrastructure agent was configured with a 60-second cooldown before the circuit transitioned to HALF-OPEN. The circuit opened because the agent hit a velocity limit during a period of high load. During the 60-second cooldown, the underlying load didn't subside. When the circuit transitioned to HALF-OPEN and issued a probe call, the probe also hit the velocity limit. The circuit re-opened immediately.

This pattern repeated every 60 seconds, creating a feedback loop: the circuit would half-open, send a probe, the probe would trigger the velocity limit, and the circuit would re-open. Meanwhile, the actual task had been sitting idle for 20 minutes while the circuit kept trying to recover.

The fix was twofold: first, use an exponential backoff for the cooldown (60s, 120s, 240s, 480s) so that repeated failures extend the wait time. Second, distinguish between velocity-limit trips and failure-count trips: velocity limits should NOT trigger the circuit breaker at all — they should trigger an alert and a temporary request rate limit, but the circuit should remain CLOSED so the agent can continue when the velocity normalizes.

**Lesson**: circuit breaker semantics for financial limits are different from circuit breaker semantics for failure limits. Not every limit type should use the same state machine.

### Case Study 4: The Tenant Isolation Failure

A multi-tenant agent platform had per-tenant daily budgets but no per-user budgets within a tenant. One enterprise customer had 500 users with access to the agent. Twenty of those users, independently and without coordination, all launched heavy research tasks simultaneously on a Monday morning. The collective spend of 20 users, each individually within a reasonable per-user limit, exhausted the entire tenant's daily budget by 10 AM.

The remaining 480 users were locked out for the rest of the day, and the enterprise customer escalated to account management demanding their money back.

The fix added per-user daily budgets at $5/user. The 20 heavy users would each be limited to $5 regardless of what other users were doing, and the collective tenant budget would be consumed more slowly.

**Lesson**: budget isolation must match the actual failure mode. In a multi-user tenant, the failure mode is N users simultaneously exhausting the tenant's budget — which requires per-user limits in addition to per-tenant limits.

### Case Study 5: The Monitoring Blind Spot

A team had implemented all the right controls — per-task cost caps, iteration limits, velocity monitoring — but had never tested whether the alerting actually worked. When a novel task type was deployed that had systematically higher costs than the alert baseline, the alert fired correctly. But the alert was sent to a Slack channel that nobody monitored on weekends.

For three days over a long weekend, the new task type ran at 5× the cost baseline. Each task individually stayed within the per-task cap, so no hard stops fired. But the collective spend was $800 over the three-day weekend rather than the expected $160. The alerts were in the channel, but nobody saw them.

The fix was a proper alert routing setup: warning alerts to Slack, critical alerts (triggered when the daily budget was 80% consumed) to PagerDuty. The daily budget alert would have fired by Friday evening and woken up the on-call engineer in time to investigate.

**Lesson**: an alert that nobody sees is equivalent to no alert. Test the complete alerting path — including weekend coverage, PagerDuty escalation, and the full human response chain — before going to production.

### Case Study 6: The Budget Tuning Overcorrection

After the $3,200 overnight incident (Case Study 1), an engineering team set very conservative limits: max 10 LLM calls per task, $0.10 per task. This caught the runaway agent, but it also broke every legitimate complex task. The team's data analysis agent, which needed 15–20 LLM calls to complete its planning-execution-verification cycle, was now hitting the iteration limit on every run.

Users started complaining that the agent "always fails." The team raised the limits, but without a clear methodology, they raised them too high — to 100 calls and $5 per task — which was still too conservative for some tasks and too permissive for others.

The right approach: instrument every production task for two weeks after setting conservative limits. Collect the p50 and p95 iteration count and cost for each task type. Set the limit to p95 × 1.5. This gives 95% of legitimate tasks enough headroom while keeping the 5% of runaway tasks firmly contained.

**Lesson**: limits set in a panic are wrong. Instrument first, then set limits based on the measured distribution of legitimate task costs.

### Case Study 7: The Model Upgrade Cost Surprise

A team had carefully tuned their budget limits over three months using Claude 3 Haiku at $0.25 per million input tokens. They decided to upgrade their agent to Claude 3.5 Sonnet for better reasoning quality. Claude 3.5 Sonnet costs $3.00 per million input tokens — 12× more expensive.

Every single existing budget limit was now 12× too generous. A per-task cap set at $0.50 (which corresponded to roughly 2 million tokens on Haiku) now corresponded to only 167,000 tokens on Sonnet — enough for perhaps 3–4 complex LLM calls instead of the 20+ the task needed.

Within the first day of the model upgrade, the circuit breaker trip rate jumped from 0.3% of tasks to 31% of tasks. The engineering team spent two days debugging before realizing the root cause was the price change, not a regression in agent logic.

The fix was to parameterize all limits in token terms rather than dollar terms, then compute the dollar cap dynamically based on the configured model price at system startup. A token budget of 100,000 tokens per task would automatically cost $0.025 on Haiku and $0.30 on Sonnet — reflecting the actual compute consumed, not an arbitrary dollar figure.

**Lesson**: token-denominated limits are more stable across model upgrades than dollar-denominated limits. Dollar limits are the right user-facing interface (you're ultimately managing cost), but internal enforcement should be token-based with a price table applied at evaluation time.

### Case Study 8: The Cascade Budget Drain

A workflow orchestration system ran five agents in sequence: agent A did research, agent B summarized, agent C fact-checked, agent D wrote a report, and agent E formatted and published. Each agent had a per-task budget of $1.00, so the workflow had an effective total budget of $5.00.

When agent A encountered a particularly complex research topic and used its full $1.00 budget, it returned a partial result to agent B. Agent B, receiving an incomplete input, needed to ask clarifying questions — which consumed additional tokens. By agent C, the conversation history had grown to include all of A's and B's exchanges, and each LLM call was 40,000 tokens long. Agent C hit its $1.00 limit after only 3 calls.

The workflow had no aggregate budget. Each agent was within its individual limit, but the cascade caused the total spend to be $4.20 on a workflow that usually cost $0.80. The effective cost was 5.25× higher than normal because each downstream agent was penalized by the expanded context from upstream failures.

The fix added a workflow-level budget that all five agents drew from collectively. When the workflow's aggregate budget was 80% consumed — regardless of which agent was responsible — all agents received a "wind down" signal and the workflow moved toward completion with compressed outputs.

**Lesson**: individual agent budgets don't add up to workflow safety in multi-agent pipelines. Add a workflow-aggregate budget layer that sits above the individual agent layers in the hierarchy.

## 13. Tuning Limits: Setting Budgets That Protect Without Over-Constraining

The previous case studies highlight the central challenge: limits that are too tight break legitimate tasks, limits that are too loose don't protect you. The right answer comes from measurement, not intuition.

![Budget Tuning Process Timeline](/imgs/blogs/circuit-breakers-and-cost-caps-10.webp)

### The measurement-first approach

Before setting any production limits, deploy with limits high enough that they will never fire on legitimate tasks (e.g., 500 LLM calls, $50 per task). Log every task's actual cost and iteration count. After 2–4 weeks, you have enough data to set meaningful limits.

The target formula for each task type:

```
cost_cap = p95_cost × 1.5
iteration_cap = p95_iterations × 1.5
```

The 1.5× multiplier gives real tasks headroom above their p95 cost while still cutting off the tail of runaways. If your p95 task cost is $0.35, set the cap at $0.52. A legitimate task that costs $0.40 (slightly above p95) will succeed. An agent in a loop that's spent $0.52 in 30 seconds on a task that should take 5 minutes will be stopped.

### Adaptive limits

For task types with high variance (e.g., research tasks where query complexity varies enormously), fixed limits will either be too tight for complex queries or too loose for simple ones. Consider adaptive limits:

```python
def compute_adaptive_cap(
    task_type: str,
    complexity_score: float,  # 0.0 = simplest, 1.0 = most complex
    base_cap: float = 0.50,
    max_cap: float = 2.00,
) -> float:
    """Scale the cost cap based on task complexity."""
    # Complexity score can come from: input length, number of required tools,
    # estimated search depth, user-stated complexity level
    return base_cap + (max_cap - base_cap) * complexity_score
```

The complexity score can be computed from features available at task admission: input length, number of tools the task will use, the presence of open-ended research goals, and so on.

### Reviewing limits over time

Limits are not set-and-forget. They should be reviewed:

- **Monthly**: check whether the hit rate (fraction of tasks stopped by a limit) has drifted significantly from the target ~1% of tasks. If it's 5%, the limits are too tight. If it's 0.01%, you're likely not catching real runaways.
- **When adding new task types**: measure the new task type's cost profile in staging before deploying with production limits.
- **When model pricing changes**: model prices change over time. A limit set for Claude 3 Haiku may be wrong for a newer model with different pricing.
- **When adding new tools**: tools that do heavy computation (web scraping, code execution, database queries) can make individual iterations much more expensive than a pure LLM-call task.

### The two-number system

For each task type, maintain two numbers: a **soft limit** and a **hard limit**.

The soft limit (typically 70–80% of the hard limit) triggers the warning path: the agent is notified that it's approaching its budget, it should start wrapping up, and a WARNING alert fires.

The hard limit triggers the stop path: the agent is stopped, a CRITICAL alert fires, and the task returns a partial result.

This two-number system gives the agent a chance to complete gracefully. A well-designed agent that receives a "budget running low" signal can compress its remaining steps, skip optional elaborations, and produce a complete if less detailed result rather than a hard-cut partial.

### Limits for different agent architectures

Not all agents are structured the same way. The optimal limits differ significantly between architectures:

**ReAct-style agents** (Reason + Act in a loop): these have naturally bounded iteration counts when the task is well-defined. Limits of 15–25 LLM calls and 20–30 tool calls are typical for moderate-complexity ReAct tasks. The most important limit is the sequential-failure cap — a ReAct agent stuck on a tool failure will reason about the failure, decide to retry, retry, reason about the new failure, and so on indefinitely.

**Plan-then-execute agents**: these generate a full plan upfront, then execute steps. The risk profile is different: the planning phase may be expensive (many LLM calls with large context), but execution is more predictable. Consider separate limits for the planning phase and the execution phase. The planning phase limit can be tight (3–5 LLM calls should be enough to generate any reasonable plan). The execution phase limit scales with the plan length.

**Multi-agent systems** (supervisor + subagents): the supervisor's token budget must account for communication with all subagents. Each subagent response that gets added to the supervisor's context increases its token cost. For a supervisor orchestrating 10 subagents, you might see 10× the expected token cost in the supervisor alone. Separate limits for supervisor and subagent tiers are essential.

**Long-running autonomous agents** (run for hours or days): traditional per-task limits don't apply. Instead, use hourly and daily budgets with rolling windows. An agent running for 8 hours might have an $8 hourly budget (averaging $1/hour) and a $20 daily budget. Velocity monitoring becomes more important for long-running agents since they have more time to accumulate quiet background spend.

### The cost-quality tradeoff

Tighter limits reduce cost risk but also reduce the quality ceiling for tasks that genuinely need more compute. When you set a per-task cap at $0.50, you're implicitly saying: "no task is worth more than $0.50 to us." That might be wrong for some tasks.

Rather than one universal limit, consider a tiered system:

| Task tier | Cost cap | Iteration cap | Who can invoke |
|---|---|---|---|
| Standard | $0.50 | 20 calls | Any user |
| Professional | $2.00 | 50 calls | Pro plan users |
| Enterprise | $10.00 | 100 calls | Enterprise accounts |
| Research | $50.00 | 500 calls | Internal team only, requires explicit approval |

Users self-select into tiers by task type or by explicitly requesting a higher-tier task. This approach gives you cost protection at every tier while allowing legitimately expensive tasks to run to completion.

## 14. Common Implementation Mistakes

After seeing many production agent deployments, the same mistakes appear repeatedly. Here are the ones that cost the most real money and engineer-hours.

**Mistake 1: Checking budget after the call, not before.** Some implementations check the budget by examining the response's usage field after the call completes. This means every call that would go over budget actually makes it through — you just record the overrun and stop future calls. For low-cost tasks this is a minor issue. For an agent that makes one 50,000-token call, checking after the call means you discover the $0.75 spend only after it's gone. Always check a budget estimate before the call.

**Mistake 2: Race conditions in concurrent budget checks.** In a system with concurrent agents, two agents might simultaneously check the budget, see remaining capacity, both proceed, and together exhaust the budget by $0.40. The fix is atomic check-and-deduct, not read-then-write. In Python with threading: use a Lock. In distributed systems: use Redis INCRBY with a max check, or use Postgres advisory locks.

**Mistake 3: Not resetting the circuit breaker correctly.** A circuit breaker that opens based on cost never automatically resets — the task is over. But if you share circuit breakers across tasks (one circuit per agent instance, not per task), a previous task's cost accumulation will affect the next task's starting state. Each task should get a fresh circuit breaker state, or the circuit should reset its `total_cost` counter when a new task begins.

**Mistake 4: Swallowing budget exceptions.** Agent frameworks often have broad exception handlers that catch any exception during tool execution or LLM calls and convert it into a "tool failed" response for the agent to process. If `BudgetExceededError` is caught by this handler, the agent will see a "tool failed" response and may retry — which triggers another budget check, which throws another `BudgetExceededError`, which gets caught again. Budget exceptions must propagate past the retry machinery to the task executor level.

**Mistake 5: No budget on the monitoring path.** Teams often add velocity monitoring and alerting, but forget to budget for the monitoring calls themselves. If your monitoring system makes LLM calls (e.g., to summarize an anomaly before alerting), those calls also need budgets. Unbudgeted monitoring can itself become a runaway cost source if an alert loop causes repeated monitoring calls.

**Mistake 6: Assuming model prices are stable.** Cloud provider LLM prices change frequently — both upward and downward. A budget system that hardcodes prices will silently become wrong when prices change. Store prices in configuration, not in code, and refresh them at startup or via a regular sync job.

**Mistake 7: Setting the same limits for all environments.** Development and staging environments should have much tighter limits than production — both because you're testing with smaller tasks and because a misconfig that blows your production budget should be cheaper to discover in staging. Use environment-specific limit configurations from the start.

```python
import os

BUDGET_PROFILES = {
    "development": {
        "task_cap": 0.05,
        "iteration_cap": 10,
        "daily_user_cap": 0.50,
    },
    "staging": {
        "task_cap": 0.25,
        "iteration_cap": 20,
        "daily_user_cap": 2.00,
    },
    "production": {
        "task_cap": 0.50,
        "iteration_cap": 30,
        "daily_user_cap": 5.00,
    },
}

def get_budget_profile() -> dict:
    env = os.getenv("APP_ENV", "development")
    return BUDGET_PROFILES.get(env, BUDGET_PROFILES["development"])
```

**Mistake 8: Not tracking the cost of cost tracking.** Token counting calls, budget check calls, checkpoint writes — these all have their own costs and latency. On a high-volume system (1,000 tasks/minute), the overhead of per-call token counting can be meaningful. Profile the monitoring stack and ensure it's not adding >5% overhead to your total LLM cost. If your token counter makes a separate API call for each pre-flight estimate, consider batching estimates or switching to a local tokenizer library (like `tiktoken` for OpenAI-family models) that runs in-process at zero API cost.

## When to Use These Controls, and When Not To

Use all of these controls for any agent that:
- Makes real LLM API calls (costs real money per call)
- Runs in an automated or scheduled context (no human in the loop to catch runaway behavior)
- Operates in a multi-user or multi-tenant environment
- Has access to tools that can themselves be expensive (web scraping, database operations, external APIs with rate limits)

Be more conservative (tighter limits, more levels of defense) when:
- The agent operates with real-world consequences (sending emails, making API calls with side effects, modifying data)
- The agent uses tools that can trigger cascading costs (a web scraping tool that itself makes LLM calls)
- You're serving untrusted users who might attempt to exhaust your resources deliberately

You can relax some controls when:
- You're running in a closed research environment where cost overrun is not a business risk
- The task type is extremely well-characterized and you have high confidence in cost estimation
- You have other mechanisms in place (human oversight at each step, hard infrastructure cost caps at the cloud provider level)

The circuit breaker pattern is almost always worth implementing. Even if you relax every other limit, having a circuit that can open when something goes catastrophically wrong — at any of the cost, iteration, or velocity dimensions — is cheap insurance against the failure modes that happen once in a million runs.

### A final note on agent trust levels

Not all agents are created equal. An internal engineering tool agent that your team uses 50 times per day can have generous limits — you know the users, you know the tasks, and a misfire is a $5 problem. A public-facing agent used by thousands of untrusted users needs much tighter limits, because an adversarial user who discovers they can exhaust your agent's token budget with a carefully crafted input can run up your costs by design.

For public-facing agents:

- Never display actual costs to the user in real time — this gives them a meter to calibrate against.
- Add input-length validation at the API boundary before any LLM call is made.
- Rate-limit at the user level before the budget system even sees the request.
- Consider hard per-IP and per-user-account daily caps that are separate from, and tighter than, the normal budget hierarchy.
- Log every circuit trip with the full input payload (with PII handling) for abuse analysis.

Budget controls and circuit breakers are not just cost management — they're also a primary defense layer against the class of attacks that attempt to exhaust your LLM resources deliberately. Design and test them with that adversarial framing in mind, not just the optimistic assumption that all users are acting in good faith. Treat them as security controls first, financial controls second — because the financial damage from a deliberate resource-exhaustion attack can be far larger than that from any accidental runaway loop.

Cross-reference: the controls described here work in concert with [agent sandboxing strategies](/blog/machine-learning/ai-agent/agent-sandboxing-strategies) (which address the blast radius of what an agent *does* in the world, independent of its cost) and with [agent observability and tracing](/blog/machine-learning/ai-agent/agent-observability-and-tracing) (which provides the telemetry these budget controls depend on). For recovering from tool-level failures — which are often what triggers the retry loops in the first place — see [tool error recovery](/blog/machine-learning/ai-agent/tool-error-recovery). The cost optimization perspective — how to reduce cost rather than just cap it — is covered in [cost optimization for agents](/blog/machine-learning/ai-agent/cost-optimization-for-agents).

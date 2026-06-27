---
title: "Tool Error Recovery: How Agents Survive Failures Without Spiraling"
date: "2026-06-27"
description: "How to design agents that handle tool errors gracefully — error taxonomy, retry strategies, graceful degradation, escalation patterns, and the feedback loop back to the LLM."
tags: ["ai-agents", "tool-use", "error-handling", "reliability", "llm", "machine-learning", "production-ml", "system-design"]
category: "machine-learning"
subcategory: "AI Agent"
author: "Hiep Tran"
featured: true
readTime: 41
---

Here is a failure mode that does not announce itself. Your agent is working through a ten-step research task — fetching documents, querying APIs, calling external services. Step three hits a transient 503. The agent receives the error, decides to retry. Step three hits a 503 again. The agent retries again. Five minutes later, the agent is still stuck on step three, burning tokens and context window, with the user wondering why nothing is happening. Nobody panicked, no alarm fired, nothing crashed. The agent simply never learned that the right move was not to retry.

Tool errors in agents are architecturally distinct from errors in conventional software. In a traditional service, an error is an exception that propagates up a call stack until something catches it or the process dies. The process has no memory of what it was doing. In an LLM agent, an error is an observation — it gets serialized into the conversation history as a `tool` message and fed back into the model's context for the next step. The model then decides what to do with it.

That distinction has enormous implications. The quality of your error messages determines whether the model makes a good decision. The structure of your retry logic determines whether the model even gets a chance to decide. And the absence of a circuit breaker means a bad model decision can compound indefinitely.

This post is a complete treatment of tool error recovery for production agents: taxonomy, formatting, retry strategies, the infinite retry trap, graceful degradation, escalation, error cascades, idempotency, and an error budget. We'll close with six detailed case studies of real failure modes.

![Tool error taxonomy — six classes each with a distinct recovery strategy](/imgs/blogs/tool-error-recovery-1.webp)

The diagram above is the mental model: tool errors are not a single category. Each class has a natural recovery action, and applying the wrong action — retrying a permanent error, escalating a transient one — wastes resources or annoys users unnecessarily.

## 1. Why Tool Errors Are Different: They Feed Back Into the LLM's Next Decision

When a tool call fails in a ReAct-style agent, the execution loop does not stop. The error is captured, converted to a string, and injected into the message history as a tool response. The LLM reads it on the next completion. This creates a tight feedback loop: the quality of the error message directly shapes the model's next action.

![Error feedback loop — tool error injected into LLM context as a new observation](/imgs/blogs/tool-error-recovery-2.webp)

Consider what happens with a raw Python traceback:

```
Traceback (most recent call last):
  File "/app/tools.py", line 47, in execute
    response = requests.get(url, timeout=30)
  File "/usr/local/lib/python3.11/site-packages/requests/adapters.py", line 665, in send
    raise ConnectionError(e, request=request)
requests.exceptions.ConnectionError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))
```

This costs 150+ tokens. It tells the model nothing actionable: is this transient? Should it retry? Wait? Use a different URL? The model may pattern-match on `ConnectionError` and generate a retry, or it may decide to abandon the task, or it may attempt to debug the Python code it never wrote. All of these are reasonable-looking responses to an ambiguous error signal.

Compare that to a structured error:

```json
{
  "error_type": "transient",
  "code": "CONNECTION_RESET",
  "message": "Remote server closed connection",
  "retryable": true,
  "suggested_action": "retry_with_backoff",
  "retry_after_seconds": 2,
  "attempt": 1,
  "max_attempts": 3,
  "tool": "web_fetch",
  "url": "https://api.example.com/data"
}
```

This costs 60 tokens. The model reads `retryable: true`, `retry_after_seconds: 2`, `attempt: 1 of 3` and has a clear path: wait two seconds, call again, and give up after three attempts total. The model is not guessing.

This is the core principle: **error messages are part of your tool API**. They deserve the same design attention as the success return value.

### The three decisions the LLM makes from an error

Every time an error lands in context, the model must choose among three paths:

1. **Retry** — call the same tool again, possibly with modified parameters or after a delay
2. **Adapt** — change approach: use a different tool, skip this step, synthesize from what's already available
3. **Escalate** — surface the error to the user or a human operator

The right choice depends entirely on the error type. A transient 503 suggests retry. A 404 for a document the user explicitly named suggests escalation (the user might have the wrong URL). A `missing_required_field` semantic error suggests adaptation (the model generated a bad call; try again with fixed parameters).

If your error messages don't convey error type, retryability, and context, the model is making these three-way decisions with essentially no signal. In practice, this leads to agents that retry everything until they hit the context window, or give up immediately on errors that would have resolved with a single backoff retry.

## 2. Tool Error Taxonomy: Six Classes, Six Recovery Strategies

The first step in designing robust error handling is recognizing that "tool error" is not a monolithic category. Different errors require fundamentally different responses. Here is a taxonomy that maps cleanly to recovery actions.

### Transient errors — retry with backoff

Transient errors are failures caused by temporary conditions that will likely resolve on their own: network resets, brief service unavailability, momentary load spikes. The server is fine; it just tripped. These are your 503 Service Unavailable and 504 Gateway Timeout responses, connection resets (`ECONNRESET`), and any error the service's documentation describes as "try again."

The critical property of a transient error is **time-dependence**: the same call made five seconds later has a reasonable chance of succeeding. Immediate retry might also succeed, but it increases load on an already-struggling service. The right strategy is exponential backoff with jitter.

Detection heuristics: HTTP 503/504, network-level exceptions (`ConnectionError`, `TimeoutError`), any error accompanied by a `Retry-After` response header.

### Rate-limit errors — backoff and respect the window

Rate-limit errors (HTTP 429 Too Many Requests, `QUOTA_EXCEEDED`, `RATE_LIMIT_REACHED`) are a distinct class from transient errors because they carry a specific time signal: the server is telling you exactly when to retry. They are also additive — continuing to fire requests while rate-limited increases the window and may get your key banned.

The correct response to a 429 is: read the `Retry-After` header or `X-RateLimit-Reset` timestamp, wait that exact duration, and then retry once. If no header is present, use exponential backoff starting from 60 seconds. Do not use immediate retry or short-interval retry.

At the fleet level, rate limit errors often indicate that multiple agent instances are sharing a key without coordination. This is an architecture problem, not a per-request problem.

### Timeout errors — circuit break before exhausting the budget

Timeout errors occur when a tool call exceeds a configured time limit. Unlike transient errors, timeouts are not necessarily self-correcting. A database query that exceeded 30 seconds on the first attempt is likely to exceed 30 seconds again on the second — if the underlying query plan is pathological, you will burn 30 seconds per retry indefinitely.

Timeouts require a circuit breaker: after N consecutive timeouts on the same tool, trip the circuit and route around the tool for a configurable cooldown period. See Section 5 on the circuit breaker pattern.

Detection: any exception with "timeout" in the name, HTTP 408 Request Timeout, your tool wrapper's explicit `asyncio.TimeoutError`.

### Semantic errors — reformulate the call

Semantic errors are failures caused by incorrect input: wrong parameter type, missing required field, value outside valid range, malformed JSON schema. These errors cannot be resolved by retrying the same call — the call is structurally wrong.

Semantic errors are the most interesting class for LLM agents because they often indicate a **model generation error**: the LLM produced a tool call that doesn't match the tool's schema. The right response is to give the model the validation error message with enough context to generate a corrected call.

Injection example:

```json
{
  "error_type": "semantic",
  "code": "INVALID_PARAMETER",
  "retryable": false,
  "message": "Parameter 'date' must be ISO 8601 format (YYYY-MM-DD), got: 'June 15th 2025'",
  "suggested_action": "reformulate_call",
  "corrected_example": {"date": "2025-06-15"}
}
```

With this error message, the model has everything it needs to generate a correct retry.

### Auth errors — escalate immediately

Authentication and authorization failures (HTTP 401 Unauthorized, 403 Forbidden) indicate that the credentials are wrong, expired, or missing the required permission scope. These are not retryable — repeating the same call with the same credentials will fail identically.

The correct response is to escalate. For 401 (token expired), the fix may be automated (token refresh), but that logic belongs in the auth layer, not the agent retry loop. For 403 (permission denied), a human must configure the correct access. Under no circumstances should the agent retry a 401 or 403.

A common anti-pattern is treating 401 as transient and including it in a general retry policy. This results in agents that make dozens of unauthorized requests before giving up, burning API quotas and potentially triggering abuse detection.

### Permanent errors — fail fast

Permanent errors are failures where the resource or operation cannot succeed regardless of retry: HTTP 404 Not Found (the resource doesn't exist), 410 Gone (the resource was explicitly removed), 422 Unprocessable Entity (the request is valid but the server cannot process it). These should fail immediately.

The distinction between semantic errors and permanent errors is subtle: semantic errors mean the call was wrong; permanent errors mean the call was correctly formed but the resource it targets doesn't exist. With a semantic error, you can fix the call. With a permanent error, you need to change your strategy entirely — find a different resource, use a different source, or escalate to the user.

| Error Class | HTTP Codes | Retry? | Action |
|---|---|---|---|
| Transient | 503, 504 | Yes, with backoff | Exponential backoff, max 5 retries |
| Rate Limit | 429 | Yes, after window | Wait Retry-After header, then once |
| Timeout | 408, client timeout | Maybe | Circuit break after N consecutive |
| Semantic | 400, 422 | No (reformulate) | Send validation error back to LLM |
| Auth | 401, 403 | No | Escalate to human or refresh token |
| Permanent | 404, 410 | No | Fail fast, change strategy |

## 3. Formatting Errors for the LLM: What Actually Helps

The error message that lands in the model's context is the sole signal it has about what went wrong and what to do next. This section covers what information belongs in that message, and how to structure it.

![Before and after: raw exception vs structured error message](/imgs/blogs/tool-error-recovery-5.webp)

### What the LLM needs to decide

The model needs exactly four pieces of information:

1. **Error class** — is this transient, semantic, auth, permanent? This determines which of the three decision paths (retry/adapt/escalate) is available.
2. **Retryability flag** — should I try again? A boolean `retryable` field is cleaner than asking the model to infer from the error code.
3. **Action suggestion** — if retryable, when and how? If reformulable, what's wrong? If escalatable, what's the human-readable explanation?
4. **Context preservation** — which tool failed, on what parameters, in what attempt number? Without this, the model cannot communicate the failure clearly if it needs to ask the user.

### What the LLM doesn't need

- **Full stack traces** — the model is not debugging your implementation
- **Internal service names** — `com.company.internal.ServiceException` means nothing to the model
- **Raw HTTP response bodies** from failed endpoints — usually HTML error pages or cryptic service messages
- **Retry timing that requires arithmetic** — give the concrete `retry_after_seconds`, not the formula

### Error message schema

Here is the schema we use in production:

```python
from dataclasses import dataclass
from typing import Optional, Literal
from enum import Enum

class ErrorType(str, Enum):
    TRANSIENT = "transient"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    SEMANTIC = "semantic"
    AUTH = "auth"
    PERMANENT = "permanent"

class SuggestedAction(str, Enum):
    RETRY = "retry"
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    WAIT_AND_RETRY = "wait_and_retry"
    REFORMULATE_CALL = "reformulate_call"
    ESCALATE = "escalate"
    FAIL_FAST = "fail_fast"
    USE_FALLBACK = "use_fallback"

@dataclass
class ToolError:
    error_type: ErrorType
    code: str                         # machine-readable, SCREAMING_SNAKE_CASE
    message: str                      # human-readable, one sentence
    retryable: bool
    suggested_action: SuggestedAction
    tool: str                         # which tool failed
    attempt: int                      # current attempt number
    max_attempts: int                 # configured maximum
    retry_after_seconds: Optional[int] = None  # for rate limits
    corrected_example: Optional[dict] = None   # for semantic errors
    context: Optional[dict] = None    # original call parameters

    def to_context_message(self) -> str:
        """Serialize to compact JSON for LLM context injection."""
        import json
        d = {
            "error_type": self.error_type.value,
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "suggested_action": self.suggested_action.value,
            "tool": self.tool,
            "attempt": f"{self.attempt}/{self.max_attempts}",
        }
        if self.retry_after_seconds:
            d["retry_after_seconds"] = self.retry_after_seconds
        if self.corrected_example:
            d["corrected_example"] = self.corrected_example
        return json.dumps(d, separators=(',', ':'))
```

The `to_context_message()` method produces a compact JSON string that costs roughly 50-80 tokens regardless of how verbose the underlying error was. That's a 5-10× reduction from a raw traceback on most tools.

### Calibrating verbosity by error class

Not all errors deserve equal verbosity. A transient 503 needs only 40 tokens — `error_type: transient, retryable: true, retry_after: 2`. A semantic error might need 120 tokens — the field name, what value was given, what type was expected, and a corrected example. An auth error needs 30 tokens and escalation instructions. Match verbosity to the information the model actually needs.

| Error Class | Token Budget | Must Include | Can Omit |
|---|---|---|---|
| Transient | 40-60 | error_type, retryable, suggested_action | Details of failure |
| Rate Limit | 50-70 | retry_after_seconds, quota context | HTTP headers |
| Timeout | 50-80 | tool, attempt, max_attempts | Internal timing |
| Semantic | 80-150 | field name, got/expected, corrected_example | Full schema |
| Auth | 30-50 | error_type, escalate message | Credential details |
| Permanent | 40-60 | error_type, fail_fast, resource identifier | Nothing |

## 4. Retry Strategies: Immediate, Exponential Backoff, Jitter, Max Retries

Once you have classified an error as retryable, you need a retry strategy. The choice between immediate retry, fixed delay, exponential backoff, and jitter has concrete operational consequences.

![Exponential backoff with full jitter — four retry attempts with randomized delays](/imgs/blogs/tool-error-recovery-4.webp)

### Immediate retry

For transient errors that are extremely brief (a connection reset on a single server in a round-robin pool, for example), immediate retry is appropriate. The server that reset your connection is typically back within milliseconds. One immediate retry before switching to backoff catches these cases.

```python
def retry_immediate(fn, max_immediate=1):
    for attempt in range(max_immediate + 1):
        try:
            return fn()
        except TransientError as e:
            if attempt == max_immediate:
                raise
    # fall through to backoff strategy
```

Limit immediate retries to one. More than one immediate retry increases load on a struggling service and is indistinguishable from a thundering herd to the server.

### Exponential backoff

For persistent transient errors and rate limits, exponential backoff is the correct default. The base formula is:

```
delay = min(base_delay × 2^attempt, max_delay)
```

With `base_delay = 1s` and `max_delay = 32s`, the sequence is 1s, 2s, 4s, 8s, 16s, 32s, 32s, 32s... This backs off quickly during a brief outage and caps at a reasonable ceiling during extended outages.

### Full jitter

The problem with pure exponential backoff is that when many clients experience the same failure simultaneously (a service hiccup that hits 1000 agents at once), they all back off by the same computed delay and then all retry simultaneously — the thundering herd. Full jitter solves this:

```
delay = random(0, min(base_delay × 2^attempt, max_delay))
```

By randomizing uniformly between 0 and the computed ceiling, clients spread their retries across the entire window. The expected delay is half of the backoff ceiling, but the _distribution_ prevents synchronized retry storms.

```python
import random
import time

def retry_with_backoff(
    fn,
    max_attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 32.0,
    jitter: bool = True,
):
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except RetryableError as e:
            last_exc = e
            if attempt == max_attempts - 1:
                break  # don't sleep after the final attempt
            ceiling = min(base_delay * (2 ** attempt), max_delay)
            delay = random.uniform(0, ceiling) if jitter else ceiling
            time.sleep(delay)
    raise last_exc
```

### Max retries and the attempt counter

Every error message injected into the LLM context should carry `attempt` and `max_attempts`. When the model sees `attempt: 3/5`, it knows it has two more retries available. When it sees `attempt: 5/5`, it knows it must change strategy. Without this counter, the model has no way to reason about whether to keep retrying or give up.

A common mistake is setting `max_attempts` too high for the context window. If each attempt takes 1000 tokens of context (tool call + error response), five retries on one tool consume 5000 tokens before the agent makes any progress. In a 32K context window, that's 15% of the budget on a single failure. Choose `max_attempts` based on context budget, not just the probability of recovery.

### Retry policy by error class

| Error Class | Immediate | Exponential Backoff | Max Attempts | Special |
|---|---|---|---|---|
| Transient | 1 | Yes, base=1s | 5 | None |
| Rate Limit | 0 | After Retry-After | 3 | Honor Retry-After header |
| Timeout | 0 | Yes, base=5s | 3 | Circuit break at 3 |
| Semantic | 0 | No | 2 | LLM must reformulate |
| Auth | 0 | No | 0 | Escalate immediately |
| Permanent | 0 | No | 0 | Fail immediately |

## 5. The Infinite Retry Trap: How Agents Get Stuck Retrying Unrecoverable Errors

The infinite retry trap is the most common failure mode in production agents. It occurs when the retry policy is too permissive and the error classification is too coarse. The classic scenario: every HTTP error is treated as transient and retried up to 10 times. A 404 (permanent) gets retried 10 times. A 403 (auth) gets retried 10 times. The agent burns tokens, makes unwanted API calls, and eventually exhausts its context window with a stack of identical error messages.

But the trap has a subtler variant: the agent correctly identifies the error as transient, but the underlying condition never resolves within the retry window. The service is down for maintenance, or the network partition is persistent, or the rate limit quota won't reset for six hours. Each retry confirms the failure, adds to the context, and brings the model no closer to a useful answer.

![Circuit breaker pattern — detecting repeated failures and opening the circuit](/imgs/blogs/tool-error-recovery-6.webp)

### The circuit breaker pattern

The circuit breaker is the right solution for both variants. It sits above the retry logic and tracks failure rates across calls. When the failure rate exceeds a threshold, the circuit trips to OPEN state and stops allowing calls to the failing tool for a cooldown period.

The circuit breaker has three states:

**CLOSED** — normal operation. Calls go through. Failures are recorded. When the failure count in the last N seconds exceeds the threshold, transition to OPEN.

**OPEN** — circuit tripped. All calls to this tool fail immediately with a circuit-open error, without contacting the tool at all. After the cooldown period, transition to HALF-OPEN.

**HALF-OPEN** — probe state. Allow one call through. If it succeeds, transition to CLOSED. If it fails, transition back to OPEN.

```python
import time
from threading import Lock
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        window_seconds: int = 60,
        cooldown_seconds: int = 30,
    ):
        self.failure_threshold = failure_threshold
        self.window_seconds = window_seconds
        self.cooldown_seconds = cooldown_seconds
        self.state = CircuitState.CLOSED
        self.failure_times: list[float] = []
        self.tripped_at: float | None = None
        self._lock = Lock()

    def record_failure(self):
        with self._lock:
            now = time.time()
            # Prune failures outside the window.
            self.failure_times = [
                t for t in self.failure_times if now - t < self.window_seconds
            ]
            self.failure_times.append(now)
            if (
                self.state == CircuitState.CLOSED
                and len(self.failure_times) >= self.failure_threshold
            ):
                self.state = CircuitState.OPEN
                self.tripped_at = now

    def record_success(self):
        with self._lock:
            self.failure_times = []
            self.state = CircuitState.CLOSED
            self.tripped_at = None

    def allow_request(self) -> bool:
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            if self.state == CircuitState.OPEN:
                if time.time() - self.tripped_at > self.cooldown_seconds:
                    self.state = CircuitState.HALF_OPEN
                    return True  # allow probe
                return False
            # HALF_OPEN: allow exactly one probe (next failure re-trips)
            return True
```

### The circuit-open error message

When the circuit is open, the error message injected into the LLM context should tell the model:

```json
{
  "error_type": "circuit_open",
  "code": "CIRCUIT_BREAKER_OPEN",
  "message": "Tool 'web_fetch' is temporarily unavailable (5 failures in 60s window)",
  "retryable": false,
  "suggested_action": "use_fallback",
  "cooldown_remaining_seconds": 22,
  "tool": "web_fetch"
}
```

The model reads `retryable: false` and `suggested_action: use_fallback` and immediately routes to the fallback tool or degraded path. It does not attempt to override the circuit breaker.

### Detecting the infinite retry trap in production

The infinite retry trap shows up in monitoring as: agent completion time spikes, token consumption per task spikes, tool error rate for a specific tool is 100% for an extended window. The circuit breaker handles this automatically in production, but you also want log-level signals: `consecutive_failures_on_tool=N` and `circuit_state=open` should emit a metric that feeds your error budget dashboard.

## 6. Graceful Degradation: Partial Results, Fallback Tools, Reduced Functionality

The goal of graceful degradation is to maintain forward progress at reduced fidelity when a primary tool fails. Instead of stopping the agent dead on the first failure, you cascade through a hierarchy of alternatives, each sacrificing some capability for availability.

![Graceful degradation chain — primary tool fails through fallback to partial result](/imgs/blogs/tool-error-recovery-7.webp)

### The degradation hierarchy

Design each tool with a degradation hierarchy at build time, not when things break:

1. **Primary tool** — the highest-fidelity path. Live database query, real-time API, full-resolution source.
2. **Cached version** — same data from a cache, potentially stale. Acceptable for most informational queries where a few minutes of staleness doesn't matter.
3. **Fallback tool** — a different source that provides approximately the same data. A secondary API, a backup service, a different search index.
4. **Partial result** — a subset of the requested data, computed from what's available. If you can't fetch all 20 fields, return the 12 you can.
5. **Absence signal** — an explicit message that the data is unavailable, allowing the LLM to reason about incomplete information rather than silently operating on null values.

### Communicating degradation level to the LLM

Every step down the degradation hierarchy must be communicated to the model. If you silently return a stale cached value, the model will present it to the user as current. Instead:

```python
@dataclass
class DegradedResult:
    data: dict
    degradation_level: str  # "primary" | "cached" | "fallback" | "partial" | "absent"
    staleness_seconds: int | None
    missing_fields: list[str]
    caveat: str  # human-readable explanation

    def to_tool_result(self) -> str:
        import json
        result = {"data": self.data}
        if self.degradation_level != "primary":
            result["_degradation"] = {
                "level": self.degradation_level,
                "caveat": self.caveat,
            }
            if self.staleness_seconds:
                result["_degradation"]["stale_seconds"] = self.staleness_seconds
            if self.missing_fields:
                result["_degradation"]["missing_fields"] = self.missing_fields
        return json.dumps(result)
```

The LLM then knows to qualify its answers: "Based on data from approximately 3 minutes ago..." or "The following fields were unavailable: [list]..."

### Fallback tool selection

Fallback tools should be registered at tool initialization time, not discovered dynamically:

```python
TOOL_FALLBACKS = {
    "web_fetch": ["web_fetch_mirror", "web_cache"],
    "database_query": ["database_replica", "redis_cache", "static_export"],
    "search_api": ["backup_search_api", "local_search_index"],
    "payment_gateway": [],  # no fallback — escalate immediately
}
```

Some tools have no acceptable fallback. A payment processing tool with a double-charge risk is not degradable — it either works or it escalates. Document these explicitly rather than letting the agent discover them at runtime.

## 7. Escalation Patterns: When to Surface Errors to Humans vs. Handle Autonomously

Escalation is the decision to stop trying to resolve an error autonomously and hand it to a human. Getting the escalation threshold right is one of the hardest design decisions in agent infrastructure.

![Escalation decision matrix — error severity × user impact × recovery probability](/imgs/blogs/tool-error-recovery-9.webp)

### The three dimensions of escalation

**Error severity** — how bad is the worst-case outcome if the agent proceeds with degraded data or fails silently? A recommendation agent operating on stale user preferences has low severity. A financial agent operating on missing transaction data has high severity.

**User impact** — is the user blocked, or can they proceed with a partial result? If the error prevents task completion, escalate. If it degrades output quality but the task can still complete, auto-handle.

**Recovery probability** — what's the realistic chance this resolves in the next few minutes? Rate limits (yes, recoverable). Auth failures (no, requires human action). Network partitions (unclear). For low-recovery-probability failures, escalation avoids burning the context window on a futile retry loop.

### Escalation message patterns

When escalating, the error message sent to the user should:
1. Explain what the agent was trying to do
2. Explain what went wrong in non-technical terms
3. Tell the user what information or action is needed to proceed
4. Offer to continue with degraded capability if that's possible

```python
def format_escalation_message(
    task: str,
    tool: str,
    error: ToolError,
    can_continue_degraded: bool,
) -> str:
    msg = f"I ran into a problem while {task}.\n\n"
    msg += f"**What happened:** {error.message}\n\n"
    
    if error.error_type == ErrorType.AUTH:
        msg += "**What's needed:** The credentials for this service need to be refreshed. "
        msg += "Please check the API key configuration.\n\n"
    elif error.error_type == ErrorType.PERMANENT:
        msg += "**What's needed:** The resource I was looking for doesn't exist. "
        msg += "Could you verify the URL or resource name?\n\n"
    
    if can_continue_degraded:
        msg += "**Option:** I can continue with reduced capability (using cached data from "
        msg += "3 minutes ago). Would you like me to proceed on that basis?"
    
    return msg
```

### Async escalation vs. blocking escalation

Not all escalations require an immediate human response. For background tasks or batch jobs, a non-blocking escalation (log an incident ticket, send a notification) allows the agent to continue with degraded capability while a human investigates. For interactive tasks, blocking escalation (ask the user directly) is usually correct.

Design your escalation system to distinguish these two modes based on the task context, not just the error type.

## 8. Error Propagation in Multi-Step Chains: How One Error Cascades

Single-tool failures are straightforward. Multi-step failures are where things get genuinely hard. When a tool call in step 2 of a 10-step chain fails, steps 3 through 10 must decide how to proceed with incomplete input. Without explicit cascade handling, they silently receive null values, produce nonsense output, and the error is invisible in the final result.

![Error cascade in a 5-step tool chain — step 2 failure propagates to later steps](/imgs/blogs/tool-error-recovery-8.webp)

### Propagating structured error context

Every intermediate result in a multi-step chain should carry an explicit error context:

```python
@dataclass
class StepResult:
    success: bool
    data: dict | None
    errors: list[ToolError]  # errors that occurred in this step
    upstream_errors: list[ToolError]  # errors from previous steps
    confidence: float  # 0.0-1.0, degraded by upstream errors
    missing_fields: list[str]  # fields that couldn't be populated

    @property
    def is_degraded(self) -> bool:
        return bool(self.upstream_errors) or bool(self.missing_fields)
```

Each step receives the previous step's `StepResult` and can inspect `upstream_errors` before deciding how to proceed. A step that requires `transaction_list` from step 2 can check whether step 2 succeeded before attempting to use that field.

### Chain abort vs. chain continuation

The choice between aborting the chain and continuing with degraded input depends on whether each step has a defined behavior for missing upstream data:

**Hard dependencies** — if step 4 literally cannot produce any output without step 2's result, abort the chain at step 2 and report the failure. Running step 4 on null input produces garbage output that may look plausible.

**Soft dependencies** — if step 4 can produce a lower-quality result without step 2's data, continue with degraded input but annotate the result. "Risk score computed without transaction history — using default baseline of 0.5."

Design steps to declare their dependency type explicitly. This avoids the common pattern where a developer writes a step assuming all inputs are always present, and the step silently fails in production when upstream data is missing.

```python
class StepDependency(Enum):
    HARD = "hard"    # abort chain if missing
    SOFT = "soft"    # continue with degraded result
    OPTIONAL = "optional"  # ignore if missing

@dataclass
class StepDefinition:
    name: str
    required_fields: dict[str, StepDependency]
    
    def validate_input(self, upstream: StepResult) -> tuple[bool, list[str]]:
        missing_hard = []
        for field, dep in self.required_fields.items():
            if field in upstream.missing_fields:
                if dep == StepDependency.HARD:
                    missing_hard.append(field)
        return len(missing_hard) == 0, missing_hard
```

### Error context amplification

A subtle failure mode: each step adds its own error context to the chain result, and by step 8 the LLM context contains 800 tokens of accumulated error messages. The model starts pattern-matching on the accumulated failures instead of the actual task.

The fix is error summarization at chain boundaries. Rather than appending every upstream error verbatim, summarize them:

```python
def summarize_upstream_errors(errors: list[ToolError]) -> str:
    if not errors:
        return ""
    by_type = {}
    for e in errors:
        by_type.setdefault(e.error_type.value, []).append(e.tool)
    parts = [f"{len(v)} {k} error(s) in {', '.join(set(v))}" for k, v in by_type.items()]
    return "Upstream: " + "; ".join(parts)
```

This keeps chain error context at 30-50 tokens regardless of chain length.

## 9. Idempotency and Safe Retries: Ensuring Retried Tool Calls Don't Cause Double-Effects

Before you can safely retry a tool call, you must know whether the tool is idempotent. An idempotent operation produces the same result whether executed once or N times. Read operations are always idempotent. Write operations are usually not.

![Tool types with idempotency requirements and retry safety](/imgs/blogs/tool-error-recovery-10.webp)

### Non-idempotent writes are the killer

Consider a payment tool: `charge_card(amount=100, card_token=tok_abc)`. Your agent calls this and receives a network timeout. Should it retry? If the charge succeeded before the timeout, a retry will charge the card twice. If the charge failed before the timeout, a retry will charge it once (correctly). The timeout gives you no information about which scenario occurred.

The standard solution is the idempotency key: a caller-generated unique identifier for each logical operation.

```python
import uuid

def charge_card(
    amount: int,
    card_token: str,
    idempotency_key: str,  # required
) -> ChargeResult:
    """
    Idempotency key: if this key has been seen before, return the original result
    without re-processing. The key should be a UUID generated at the start of the
    operation and reused on all retries.
    """
    pass
```

The tool implementation stores `(idempotency_key → result)` in a durable store. On retry, it returns the stored result instead of re-processing. The caller generates the key once and passes it on every retry:

```python
def safe_charge_with_retry(amount, card_token, max_retries=3):
    idem_key = str(uuid.uuid4())  # generate once, reuse on retries
    for attempt in range(max_retries):
        try:
            return charge_card(amount, card_token, idempotency_key=idem_key)
        except TransientError:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
```

### Database operations

Database inserts are not idempotent by default. The SQL pattern `INSERT INTO ... ON CONFLICT (key) DO NOTHING` (or `DO UPDATE`) provides idempotency for inserts keyed on a natural identifier:

```sql
-- Idempotent insert: same external_id won't create duplicate records
INSERT INTO orders (external_id, user_id, amount, created_at)
VALUES ($1, $2, $3, NOW())
ON CONFLICT (external_id) DO UPDATE
  SET updated_at = NOW()
RETURNING *;
```

For updates, include a version or timestamp check: `WHERE version = $1` ensures that a retry doesn't clobber an update made by another process between retries.

### Check-then-act for file operations

File write tools should follow check-then-act: before writing, check if the file already contains the expected output. This is weaker than true idempotency (there's a TOCTOU window) but sufficient for most agent use cases:

```python
def write_report_safe(path: str, content: str) -> bool:
    import os
    import hashlib
    expected_hash = hashlib.sha256(content.encode()).hexdigest()
    if os.path.exists(path):
        with open(path) as f:
            existing = f.read()
        if hashlib.sha256(existing.encode()).hexdigest() == expected_hash:
            return True  # already written, safe to return success
    with open(path, 'w') as f:
        f.write(content)
    return True
```

### Email and notification deduplication

Email sends are among the most dangerous non-idempotent operations because the failure mode (duplicate email to user) is user-visible. Use a `message_id` field that incorporates the recipient, subject, and content hash:

```python
def send_email_idempotent(to: str, subject: str, body: str) -> bool:
    import hashlib, time
    msg_id = hashlib.sha256(f"{to}:{subject}:{body}".encode()).hexdigest()[:16]
    # Check deduplication store (Redis, DynamoDB, etc.) before sending
    if dedup_store.exists(msg_id):
        return True  # already sent
    dedup_store.set(msg_id, time.time(), ex=86400)  # 24h TTL
    return email_client.send(to, subject, body, message_id=msg_id)
```

## 10. Error Budget: Tracking Tool Failure Rates in Production

An error budget is a quantitative measure of how much failure your system can tolerate before it starts violating its reliability commitments. For agent infrastructure, we define error budgets at the tool level, not just the service level.

### Tool-level SLOs

Define a separate SLO for each tool, not just for the agent as a whole:

| Tool | Target Success Rate | Error Budget (30d) | Alert Threshold |
|---|---|---|---|
| `web_fetch` | 99% | 1% × calls | > 2% in 1-hour window |
| `database_query` | 99.9% | 0.1% × calls | > 0.5% in 15-min window |
| `search_api` | 99.5% | 0.5% × calls | > 1% in 30-min window |
| `payment_tool` | 99.99% | 0.01% × calls | > 0.05% in 5-min window |

The payment tool's SLO is two orders of magnitude stricter than the web fetch tool because the consequence of failure is user-visible financial impact.

### Metrics to track per tool

Every tool call should emit at least three metrics:

1. `tool_call_total{tool=X, result=success|error}` — the raw success/failure counter
2. `tool_call_duration_seconds{tool=X, percentile=p50|p95|p99}` — latency distribution
3. `tool_error_total{tool=X, error_type=transient|rate_limit|timeout|semantic|auth|permanent}` — error breakdown by class

The error type breakdown is particularly valuable. A spike in `semantic` errors usually means a model regression (new model generates different tool call syntax). A spike in `auth` errors means credential rotation is needed. A spike in `transient` errors means a dependency is degrading. These are different operational responses.

### Budget burn rate alerts

A burn rate alert fires when you're consuming your error budget faster than sustainable. If your monthly budget is 1% failure rate and you've consumed 80% of it in the first week, you're burning at 4× the sustainable rate.

```python
def compute_burn_rate(
    current_failures: int,
    current_calls: int,
    budget_fraction: float,  # e.g. 0.01 for 1% budget
    window_fraction: float,  # fraction of month elapsed, e.g. 7/30
) -> float:
    """
    Returns the burn rate multiplier.
    1.0 = on track. 2.0 = consuming budget 2× faster than sustainable.
    """
    if current_calls == 0:
        return 0.0
    current_failure_rate = current_failures / current_calls
    sustainable_rate = budget_fraction  # use whole budget over 1 period
    return current_failure_rate / (sustainable_rate * 1.0)  # annualized
```

Alert at `burn_rate > 2.0` with a 1-hour window for high-SLO tools (payment), `burn_rate > 3.0` with a 6-hour window for medium-SLO tools (search).

## 11. Implementation: Python Tool Executor with Retry, Fallback, and Escalation

Here is a complete, production-ready tool executor that ties together the patterns from sections 3 through 10. It handles retry with backoff, circuit breaker, fallback routing, error message formatting, and idempotency key propagation.

```python
import uuid
import time
import random
import logging
from dataclasses import dataclass, field
from typing import Callable, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorClass(str, Enum):
    TRANSIENT = "transient"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    SEMANTIC = "semantic"
    AUTH = "auth"
    PERMANENT = "permanent"


@dataclass
class ToolCallResult:
    success: bool
    data: Any | None
    error_message: str | None  # serialized for LLM context injection
    attempts: int
    degradation_level: str = "primary"  # primary | cached | fallback | partial


def classify_error(exc: Exception) -> tuple[ErrorClass, bool]:
    """
    Returns (error_class, retryable).
    Customize this to match your tool stack's exception types.
    """
    msg = str(exc).lower()
    if hasattr(exc, 'status_code'):
        code = exc.status_code
        if code == 429:
            return ErrorClass.RATE_LIMIT, True
        if code in (401, 403):
            return ErrorClass.AUTH, False
        if code == 404:
            return ErrorClass.PERMANENT, False
        if code in (503, 504):
            return ErrorClass.TRANSIENT, True
        if code == 400:
            return ErrorClass.SEMANTIC, False
    if 'timeout' in msg or 'timed out' in msg:
        return ErrorClass.TIMEOUT, True
    if 'connection' in msg and ('reset' in msg or 'refused' in msg):
        return ErrorClass.TRANSIENT, True
    if 'unauthorized' in msg or 'forbidden' in msg:
        return ErrorClass.AUTH, False
    if 'not found' in msg:
        return ErrorClass.PERMANENT, False
    # Unknown — treat as transient, conservative max retries
    return ErrorClass.TRANSIENT, True


class CircuitBreaker:
    def __init__(self, threshold=5, window=60, cooldown=30):
        self.threshold = threshold
        self.window = window
        self.cooldown = cooldown
        self.failures: list[float] = []
        self.tripped_at: float | None = None
        self.is_open = False

    def allow(self) -> bool:
        now = time.time()
        if self.is_open:
            if now - self.tripped_at > self.cooldown:
                self.is_open = False  # half-open: allow probe
                return True
            return False
        return True

    def record(self, success: bool):
        now = time.time()
        self.failures = [t for t in self.failures if now - t < self.window]
        if not success:
            self.failures.append(now)
            if len(self.failures) >= self.threshold:
                self.is_open = True
                self.tripped_at = now
                logger.warning(f"Circuit breaker tripped after {len(self.failures)} failures")
        else:
            self.failures = []
            self.is_open = False


class ToolExecutor:
    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._fallbacks: dict[str, list[str]] = {}
        self._tools: dict[str, Callable] = {}

    def register(
        self,
        name: str,
        fn: Callable,
        fallbacks: list[str] | None = None,
        circuit_threshold: int = 5,
    ):
        self._tools[name] = fn
        self._breakers[name] = CircuitBreaker(threshold=circuit_threshold)
        self._fallbacks[name] = fallbacks or []

    def call(
        self,
        tool_name: str,
        params: dict,
        max_attempts: int = 3,
        idempotency_key: str | None = None,
    ) -> ToolCallResult:
        idem_key = idempotency_key or str(uuid.uuid4())
        breaker = self._breakers.get(tool_name)

        # Check circuit breaker
        if breaker and not breaker.allow():
            return ToolCallResult(
                success=False,
                data=None,
                error_message=(
                    f'{{"error_type":"circuit_open","code":"CIRCUIT_OPEN",'
                    f'"tool":"{tool_name}","retryable":false,'
                    f'"suggested_action":"use_fallback"}}'
                ),
                attempts=0,
            )

        fn = self._tools.get(tool_name)
        if not fn:
            return ToolCallResult(
                success=False, data=None,
                error_message=f'{{"error_type":"permanent","code":"UNKNOWN_TOOL","tool":"{tool_name}"}}',
                attempts=0,
            )

        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                result = fn(**params, idempotency_key=idem_key)
                if breaker:
                    breaker.record(True)
                return ToolCallResult(success=True, data=result, error_message=None, attempts=attempt + 1)
            except Exception as exc:
                last_exc = exc
                error_class, retryable = classify_error(exc)
                if breaker:
                    breaker.record(False)

                if not retryable:
                    # Fail fast — no more attempts
                    return ToolCallResult(
                        success=False,
                        data=None,
                        error_message=self._format_error(
                            tool_name, error_class, exc, attempt + 1, max_attempts
                        ),
                        attempts=attempt + 1,
                    )

                if attempt < max_attempts - 1:
                    delay = random.uniform(0, min(1.0 * (2 ** attempt), 32.0))
                    logger.info(f"{tool_name}: attempt {attempt+1} failed ({error_class}), retry in {delay:.1f}s")
                    time.sleep(delay)

        # All retries exhausted — try fallback
        for fallback_name in self._fallbacks.get(tool_name, []):
            logger.info(f"{tool_name}: falling back to {fallback_name}")
            fallback_result = self.call(fallback_name, params, max_attempts=2)
            if fallback_result.success:
                fallback_result.degradation_level = "fallback"
                return fallback_result

        return ToolCallResult(
            success=False,
            data=None,
            error_message=self._format_error(
                tool_name, ErrorClass.TRANSIENT, last_exc, max_attempts, max_attempts
            ),
            attempts=max_attempts,
        )

    def _format_error(
        self, tool: str, error_class: ErrorClass, exc: Exception,
        attempt: int, max_attempts: int
    ) -> str:
        import json
        return json.dumps({
            "error_type": error_class.value,
            "code": type(exc).__name__.upper(),
            "message": str(exc)[:120],
            "retryable": error_class in (ErrorClass.TRANSIENT, ErrorClass.RATE_LIMIT, ErrorClass.TIMEOUT),
            "tool": tool,
            "attempt": f"{attempt}/{max_attempts}",
        }, separators=(',', ':'))
```

This executor is approximately 100 lines of meaningful logic. It handles: circuit breaking, error classification, exponential backoff with jitter, fallback routing, idempotency key propagation, and structured error formatting. Wire it up once and use it for every tool in your agent.

## 12. Case Studies

### Case study 1: The over-retrying research agent

A document research agent was tasked with fetching and summarizing 15 academic papers. Three of the papers were behind paywalls (HTTP 403 Forbidden). The agent's retry policy treated all HTTP errors as transient and retried up to five times with backoff. The result: 15 API calls per paper × 3 papers = 45 failed API calls, consuming 12,000 tokens of context and eight minutes of wall clock time. The agent then ran out of context window and returned an incomplete summary.

**Root cause:** No error classification. All errors got the same transient retry policy.

**Fix:** Add `AUTH` class detection. HTTP 403 → no retries, immediate escalation with message: "Paper X is behind a paywall. Please provide access credentials or use an open-access alternative."

**Outcome after fix:** 3 × 1 call (fails immediately), 20-token error message per paper, agent continues with 12 of 15 papers and explicitly reports which three were inaccessible.

### Case study 2: The rate-limited LLM tool call that thundered the herd

An agent system running 200 parallel agents all made simultaneous calls to a search API. The API returned 429 with `Retry-After: 60`. The retry policy used fixed backoff: `sleep(5)` between attempts. All 200 agents woke up together at t+5, all got 429, all slept until t+10, all got 429 again. The pattern repeated for 12 minutes, with the API rate limiter seeing solid 200-request bursts every 5 seconds.

**Root cause:** Fixed retry interval instead of jitter. The `Retry-After` header was ignored.

**Fix:** Parse `Retry-After` header into the delay. Add full jitter within a ±20% window around the header value. Result: requests spread uniformly across the 60-second window.

**Lesson:** Synchronized retries are as damaging as synchronized requests. Jitter is not optional at scale.

### Case study 3: The semantic error that became an infinite loop

A code generation agent was using a file creation tool with a strict path schema: paths must start with `/workspace/`. The agent generated `path: "src/utils/helper.py"` (without the prefix). The tool returned a 400 with the raw error: `Invalid path: must start with /workspace/`. The agent, not recognizing this as a semantic error, classified it as transient and retried the same call three more times.

**Root cause:** The raw error string didn't contain `error_type: semantic` or `retryable: false`. The agent defaulted to transient.

**Fix:** The tool now returns:
```json
{
  "error_type": "semantic",
  "code": "INVALID_PATH",
  "message": "Path must start with /workspace/",
  "retryable": false,
  "corrected_example": {"path": "/workspace/src/utils/helper.py"}
}
```
On seeing `retryable: false` and the corrected example, the agent generates the correct call on the first retry.

### Case study 4: The silent cascade

A customer analytics agent ran a 6-step chain: (1) fetch customer profile, (2) fetch order history, (3) fetch support tickets, (4) compute churn risk score, (5) generate personalized offer, (6) return recommendation. Step 2 hit a database timeout and returned an empty list instead of raising an error. Steps 4 and 5 computed a churn risk of 0 (no orders = no churn signal) and generated an offer for a customer who had placed 47 orders in the last month. The recommendation was entirely wrong.

**Root cause:** Step 2 silently swallowed the timeout and returned an empty list. No error propagated. Steps 4 and 5 had no way to know the input was garbage.

**Fix:** Step 2 returns a `StepResult` with `success=False`, `errors=[timeout_error]`, `missing_fields=["order_history"]`. Step 4 checks for `"order_history" in upstream.missing_fields` before computing churn score. If missing, it returns `confidence=0.1` and sets `degraded=True`. Step 5 refuses to generate an offer with `confidence < 0.5` and escalates to a human.

**Lesson:** Silent failures in chains are worse than loud ones. Every step that can fail must communicate that failure explicitly to downstream steps.

### Case study 5: The idempotency disaster

An email notification agent was tasked with sending a survey to 10,000 users. A network timeout during the SMTP handshake caused the send to fail after the email was already queued. The retry policy sent the same email again. The deduplication window was 5 minutes, but the retry backoff took 6 minutes, so the second send was outside the dedup window. 10,000 users received two survey emails. The company's unsubscribe rate spiked 15× for three days.

**Root cause:** The idempotency key was generated at each retry instead of once per logical operation. The dedup window was too short for the backoff window.

**Fix:** Idempotency key generated once per batch item, passed through all retries. Dedup window extended to 48 hours. `message_id` incorporates recipient + subject + content hash. Any second send with the same `message_id` is silently dropped.

**Lesson:** Idempotency key generation and dedup window sizing are a matched pair. The dedup window must be larger than the maximum backoff window.

### Case study 7: The LLM that escalated too eagerly

A customer support agent was configured to escalate to a human whenever it saw an auth error. This was correct — 401 and 403 should not be retried. But the database tool also returned a custom `AuthorizationError` subclass for row-level security violations: a query for `user_id=123`'s data by an agent running in `user_id=456`'s context. This was not a credentials issue — it was a query scoping bug. But the error class name matched the auth pattern.

The result: every time the agent queried the wrong user's data (a logic bug in the chain), it escalated to a human instead of fixing the query. The support team received 847 false-positive escalations in a single day before someone noticed that every escalation came from the same tool call with the same root cause.

**Root cause:** Over-broad auth error classification. The code classified all `AuthorizationError` subclasses as auth, without distinguishing between credential auth and row-level authorization.

**Fix:** Introduce a `QUERY_SCOPE` error class distinct from `AUTH`. Credential-level auth → escalate. Row-level authorization → classify as semantic (the query itself is wrong), inject a structured error with `suggested_action: reformulate_call`, and let the model fix the user-scope in the query parameters.

**Lesson:** Error classification must be specific enough to distinguish different *causes* of the same exception hierarchy. A single `AuthorizationError` base class can represent at least three different root causes, each with a different recovery path.

### Case study 8: The timeout that was actually a data size problem

A document analysis agent called a PDF extraction tool that had a 30-second timeout. A user uploaded a 400-page academic paper. The extraction timed out. The circuit breaker tracked three timeouts in 60 seconds, tripped the circuit, and routed to the fallback text extractor.

The fallback text extractor also timed out on the 400-page document (it had the same underlying bottleneck — parsing very large PDFs). The agent degraded to "partial result" and returned analysis of only the first few pages that had been extracted before the timeout.

The user received a partial analysis with a note that "the document was too large for full processing," which was technically accurate but operationally wrong. The tool wasn't broken — it just needed a longer timeout and chunked processing for large documents.

**Root cause:** Timeout configured for average-case input (10-page documents), not worst-case (400-page documents). The circuit breaker correctly detected repeated timeouts but routed to a fallback that had the same limitation.

**Fix:** 
1. Error message now includes `document_pages: N` in context, enabling the LLM to reason about document size as a factor.
2. Tool registers two timeout tiers: 30s for documents ≤50 pages, 120s for documents ≤400 pages, immediate rejection with `size_limit_exceeded` for >400 pages.
3. The `size_limit_exceeded` error is permanent (not retryable) and includes `suggested_action: split_document` — the model now knows to ask the user to split large documents rather than timing out.

**Lesson:** Timeout configuration must account for input size distribution. A timeout that works for 99th-percentile inputs is different from a timeout that works for 99th-percentile of *document sizes*.

### Case study 6: The circuit breaker that saved a production incident

A code review agent used a vector similarity API to find related code snippets. During a routine deployment of the API service, the new version had a memory leak that caused it to fall over under load after 20 requests. Without a circuit breaker, 500 parallel agents × 20 calls per agent = 10,000 calls before any timeout. The deployment would have been invisible until the SRE team got paged 10 minutes later.

With the circuit breaker configured at `threshold=5, window=60, cooldown=30`: after 5 failures in 60 seconds, the circuit tripped. The other 495 agents received `circuit_open` errors immediately, fell back to a simpler keyword search, and continued working. The monitoring dashboard showed a circuit-open spike at exactly the deployment timestamp. The SRE team rolled back the deployment in 4 minutes.

**Lesson:** The circuit breaker is not just a recovery mechanism — it's a diagnostic signal. `circuit_state=open` is a better incident indicator than `error_rate > threshold` because it fires based on failure patterns, not raw counts.

## 12b. Testing Tool Error Handling

Error handling code is the most undertested code in most agent systems. The happy path gets tested in every demo. The error paths get tested never, until they fail in production.

Here is a testing strategy that gives you confidence before you discover these failure modes from a user.

### Unit testing error classification

Test the `classify_error` function exhaustively against all the HTTP status codes and exception types your tools can produce. Don't assume — enumerate:

```python
import pytest
from unittest.mock import MagicMock

class MockHTTPError(Exception):
    def __init__(self, status_code):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}")

@pytest.mark.parametrize("exc,expected_class,expected_retryable", [
    (MockHTTPError(503), ErrorClass.TRANSIENT, True),
    (MockHTTPError(504), ErrorClass.TRANSIENT, True),
    (MockHTTPError(429), ErrorClass.RATE_LIMIT, True),
    (MockHTTPError(401), ErrorClass.AUTH, False),
    (MockHTTPError(403), ErrorClass.AUTH, False),
    (MockHTTPError(404), ErrorClass.PERMANENT, False),
    (MockHTTPError(410), ErrorClass.PERMANENT, False),
    (MockHTTPError(400), ErrorClass.SEMANTIC, False),
    (TimeoutError("connection timed out"), ErrorClass.TIMEOUT, True),
    (ConnectionError("Connection refused"), ErrorClass.TRANSIENT, True),
])
def test_classify_error(exc, expected_class, expected_retryable):
    error_class, retryable = classify_error(exc)
    assert error_class == expected_class
    assert retryable == expected_retryable
```

This test takes 5 minutes to write and catches the class of bug that caused Case Study 7 (over-broad auth classification).

### Integration testing retry behavior

Test that the executor actually retries the correct number of times with the correct delay:

```python
import time
from unittest.mock import patch, Mock

def test_retry_count_on_transient_error():
    call_times = []
    
    def flaky_tool(**kwargs):
        call_times.append(time.time())
        raise ConnectionError("connection reset")
    
    executor = ToolExecutor()
    executor.register("flaky", flaky_tool)
    
    with patch("time.sleep") as mock_sleep:
        result = executor.call("flaky", {}, max_attempts=3)
    
    assert not result.success
    assert result.attempts == 3
    assert len(call_times) == 3
    # Should have slept twice (between attempt 1-2 and 2-3)
    assert mock_sleep.call_count == 2

def test_no_retry_on_permanent_error():
    call_count = [0]
    
    def missing_tool(**kwargs):
        call_count[0] += 1
        err = Exception("not found")
        err.status_code = 404
        raise err
    
    executor = ToolExecutor()
    executor.register("missing", missing_tool)
    
    result = executor.call("missing", {}, max_attempts=5)
    assert not result.success
    assert call_count[0] == 1  # should have stopped after first failure
```

### Chaos testing in staging

For systems that matter, inject artificial failures in staging before production. A chaos tool wrapper that randomly returns errors according to a configured distribution:

```python
import random
from typing import Callable

class ChaosWrapper:
    def __init__(
        self,
        fn: Callable,
        transient_rate: float = 0.0,   # fraction of calls that raise 503
        timeout_rate: float = 0.0,     # fraction of calls that timeout
        latency_ms: float = 0.0,       # added latency per call
    ):
        self.fn = fn
        self.transient_rate = transient_rate
        self.timeout_rate = timeout_rate
        self.latency_ms = latency_ms
    
    def __call__(self, **kwargs):
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000)
        
        r = random.random()
        if r < self.transient_rate:
            err = Exception("503 Service Unavailable")
            err.status_code = 503
            raise err
        if r < self.transient_rate + self.timeout_rate:
            raise TimeoutError("injected timeout")
        
        return self.fn(**kwargs)

# Register chaos-wrapped tools in staging
executor.register(
    "web_fetch",
    ChaosWrapper(real_web_fetch, transient_rate=0.1, timeout_rate=0.05),
)
```

Running your full agent task suite against a 10% transient error rate and 5% timeout rate will surface retry, circuit breaker, and fallback bugs that only appear under failure conditions.

### Testing idempotency keys

The most important property to test is that your idempotency key is actually working — that a second call with the same key returns the first result without re-executing:

```python
def test_idempotency_prevents_duplicate_charge():
    charges = []
    
    def charge_fn(amount, card_token, idempotency_key):
        if idempotency_key in [c['key'] for c in charges]:
            # Return existing charge — idempotency satisfied
            return next(c for c in charges if c['key'] == idempotency_key)
        charge = {'key': idempotency_key, 'amount': amount, 'id': f'ch_{len(charges)}'}
        charges.append(charge)
        return charge
    
    executor = ToolExecutor()
    executor.register("charge", charge_fn)
    
    idem_key = "test-key-123"
    # First call — should execute
    r1 = executor.call("charge", {"amount": 100, "card_token": "tok_x"}, idempotency_key=idem_key)
    # Second call with same key — should return same result without charging again
    r2 = executor.call("charge", {"amount": 100, "card_token": "tok_x"}, idempotency_key=idem_key)
    
    assert len(charges) == 1  # only one actual charge
    assert r1.data['id'] == r2.data['id']  # same result both times
```

This test directly validates that Case Study 5's disaster cannot happen in your system.

### Property-based testing for error message schema

Use Hypothesis to verify that your error formatting function always produces valid JSON within the token budget:

```python
from hypothesis import given, strategies as st

error_types = st.sampled_from(list(ErrorClass))
status_codes = st.one_of(st.integers(400, 599), st.none())

@given(
    error_class=error_types,
    message=st.text(max_size=500),
    attempt=st.integers(1, 10),
    max_attempts=st.integers(1, 10),
)
def test_format_error_always_valid_json(error_class, message, attempt, max_attempts):
    import json
    executor = ToolExecutor()
    # Force attempt <= max_attempts
    attempt = min(attempt, max_attempts)
    
    exc = Exception(message)
    formatted = executor._format_error("test_tool", error_class, exc, attempt, max_attempts)
    
    # Must be valid JSON
    parsed = json.loads(formatted)
    
    # Must have required fields
    assert "error_type" in parsed
    assert "retryable" in parsed
    assert isinstance(parsed["retryable"], bool)
    
    # Must fit within token budget (rough approximation: 300 chars ≈ 100 tokens)
    assert len(formatted) < 600
```

Property-based testing finds edge cases in error message formatting that unit tests miss — empty messages, unicode in error codes, very long tracebacks that get truncated.

## 13. When to Retry / When to Fail Fast / When to Escalate

After all the theory and case studies, here is the decision table:

### Retry when

- **Error type is transient or rate-limit** — the server is telling you it's recoverable
- **The tool is idempotent OR you have an idempotency key** — retrying won't cause side effects
- **Attempt number is below max_attempts** — you haven't exhausted your budget
- **Circuit breaker is CLOSED** — the tool isn't in a systematic failure mode
- **Context window can absorb another round** — you have enough tokens to continue

### Fail fast when

- **Error type is permanent (404, 410)** — the resource isn't coming back
- **Error type is semantic** — the call itself is wrong; retrying won't help without reformulation
- **Circuit breaker is OPEN** — the tool is systematically failing; stop adding to the noise
- **The tool is non-idempotent and you have no idempotency key** — the retry risk outweighs the benefit
- **Max retries have been reached** — additional retries are sunk-cost thinking

### Escalate when

- **Error type is auth (401, 403)** — human must configure credentials
- **The failure blocks the user's stated goal** — the agent cannot deliver value without this tool
- **Recovery probability is low and user impact is high** — don't waste time on a futile retry loop
- **The same failure has occurred across multiple tasks** — pattern suggests systemic issue
- **The error involves ambiguity about user intent** — 404 on a user-provided URL may mean the user gave the wrong URL; ask before failing silently

### The meta-rule

Tools fail regularly in production. The agents that survive are not the ones that never encounter errors — they are the ones that treat errors as information, route that information correctly, and maintain forward progress at the highest fidelity the current conditions support.

Investing in structured error messages, error classification, circuit breakers, and idempotency is not defensive programming. It is the core infrastructure that transforms an agent from a prototype that works 80% of the time in demos into a system that delivers value 99% of the time in production.

For related patterns on agent system design, see [agent loop anatomy](/blog/machine-learning/ai-agent/agent-loop-anatomy), [circuit breakers and cost caps](/blog/machine-learning/ai-agent/circuit-breakers-and-cost-caps), [tool schema design principles](/blog/machine-learning/ai-agent/tool-schema-design-principles), and [parallel tool calls](/blog/machine-learning/ai-agent/parallel-tool-calls).

---
title: "Reliability for inference services: Timeouts, retries, hedging, and degraded modes"
date: "2026-08-03"
publishDate: "2026-08-03"
description: "Build an inference endpoint that stops abandoned GPU work, retries only when safe, hedges tails without multiplying load, and degrades quality before availability."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "reliability",
    "timeouts",
    "retries",
    "hedging",
    "streaming",
    "latency",
    "ml-systems",
    "vllm",
    "pytorch",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 30
---

Your client times out at two seconds. The GPU keeps decoding for another eighteen. A retry arrives, gets admitted beside the orphan, and now one user request occupies two KV leases. The first stream eventually closes, the second stream produces a different answer, and the dashboard reports two successful generations even though the user saw one error.

That is not an HTTP problem with a little GPU work behind it. It is a resource-ownership problem that happens to expose an HTTP interface. Figure 1 is the mental model: a request has a client clock and a GPU lease, and reliability means the cancellation path connects those clocks. If the socket disappears but the lease does not, the endpoint is still spending capacity on a request that no longer has a consumer.

This is post 50 in the [Inference Engineering series](/blog/machine-learning/inference-engineering/what-inference-engineering-is). We will add a reliability boundary to `nanoserve`: absolute deadlines, cancellation at scheduler step boundaries, bounded retry metadata, a delayed hedge policy, explicit partial-stream terminal states, and a degradation ladder. The point is not to turn a toy server into a service mesh. The point is to make every expensive action have an owner, a deadline, and a terminal outcome.

![A request branches at the API into normal GPU work and an explicit cancellation path before both reach a terminal result](/imgs/blogs/reliability-timeouts-retries-hedging-and-degraded-modes-1.webp)

The production comparison target is vLLM, not a library we will pretend to reimplement. Its official [speculative-decoding guide](https://docs.vllm.ai/en/v0.18.2/features/speculative_decoding/) describes speculation as a low- to medium-QPS latency tool, and its [2024 performance post](https://vllm.ai/blog/2024-10-17-spec-decode), dated 2024-10-17, reports both speedups at QPS 1 and slowdowns at high QPS. That is exactly the reliability lesson: an optimization is part of the failure policy when its extra work changes the system’s capacity.

## 1. Reliability starts with ownership, not a timeout number

The common implementation is a deadline in the HTTP handler:

```python
result = await asyncio.wait_for(engine.generate(prompt), timeout=2.0)
```

This protects the handler. It does not necessarily protect the engine. If `engine.generate` submitted a request to a queue and then returned a future, `wait_for` cancels the future visible to Python. It may not remove the request from the scheduler, release its KV blocks, stop a CUDA graph launch already in flight, or tell a remote replica that the work is no longer wanted.

The contract we want is narrower and stronger:

1. The edge assigns a request ID and an absolute deadline.
2. The scheduler owns the request while it is queued or running.
3. The runner checks cancellation at a safe step boundary.
4. The scheduler transitions the request exactly once to a terminal state.
5. Cleanup releases KV blocks, removes the request from admission counts, and records why it ended.

The words “exactly once” describe the state transition, not the model computation. A GPU kernel cannot be rolled back. If a timeout fires during a kernel, the safest action is usually to mark cancellation and reclaim at the next boundary. Trying to kill arbitrary device work from a Python exception creates a different class of correctness bug: a later request may observe buffers that are still being written.

### Deadline versus timeout

A timeout is a duration local to one operation. A deadline is an absolute end time carried through the call graph. If ingress gives a request a deadline of `t0 + 2.0`, the queue, prefill, decode, stream writer, and any internal RPC all ask how much time remains. They do not each receive a fresh two seconds.

![A timeline shows a 2,000 millisecond budget shrinking through ingress, queue, prefill, decode, streaming, and a terminal abort](/imgs/blogs/reliability-timeouts-retries-hedging-and-degraded-modes-2.webp)

At hop $i$, let $t_i$ be the current monotonic clock and $D$ the absolute deadline. The remaining budget is an explanatory model, not a formula quoted from a paper:

$$
R_i = \max(0, D - t_i)
$$

The `max` matters. Negative time is not a useful timeout, and a request that has no budget should not enter another queue merely because the queue API accepts a non-negative duration.

```python
# nanoserve/reliability.py
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Literal

Terminal = Literal["completed", "canceled", "timed_out", "failed"]


@dataclass
class RequestLease:
    request_id: str
    deadline_ns: int
    max_new_tokens: int
    canceled: bool = False
    terminal: Terminal | None = None
    output_tokens: int = 0
    cancel_reason: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def remaining_s(self) -> float:
        return max(0.0, (self.deadline_ns - time.monotonic_ns()) / 1e9)

    def expired(self) -> bool:
        return time.monotonic_ns() >= self.deadline_ns

    def request_cancel(self, reason: str) -> None:
        if self.terminal is None:
            self.canceled = True
            self.cancel_reason = reason

    def close(self, terminal: Terminal) -> None:
        if self.terminal is not None:
            raise RuntimeError(f"request {self.request_id} closed twice")
        self.terminal = terminal
```

This is deliberately a lease, not a boolean passed through three unrelated functions. The lease carries the ownership facts that must survive a stream disconnect: identity, deadline, cancellation reason, output count, and terminal state. A real server should protect `close` with the scheduler’s event-loop ownership or a lock; the example makes the invariant visible.

#### Worked example: five hops and one budget

Suppose a client grants 2,000 ms, and the request spends 300 ms in ingress and queue, 600 ms in prefill, and 680 ms across the first decode interval. The remaining budget is derived arithmetic:

$$2{,}000 - 300 - 600 - 680 = 420\text{ ms}$$

The decode loop may start only if its estimated next step fits inside 420 ms. If the edge instead gives ingress, queue, prefill, decode, and streaming five independent 2,000 ms timeouts, the worst-case wall-clock allowance is:

$$5 \times 2{,}000 = 10{,}000\text{ ms}$$

That 10-second value is not a measured latency; it is the upper bound created by the bad policy. The timeline figure shows the intended policy. Source: `derived` from the stated hop budgets.

### Cancellation belongs at a step boundary

`nanoserve` should not poll cancellation inside every fused kernel. It should poll at the boundary where the scheduler already gathers completed tokens, updates sequence lengths, and chooses the next batch. That boundary has three useful properties:

- the previous kernel has finished from the scheduler’s point of view;
- the request’s block table can be removed without changing another request’s indexing;
- the next launch can omit the canceled row.

```python
# nanoserve/engine.py
async def decode_step(self) -> None:
    finished = await self.runner.run_one_step(self.active_rows)
    for row, token in finished:
        lease = self.leases[row.request_id]
        lease.output_tokens += 1
        if lease.expired():
            lease.request_cancel("deadline_exceeded")
        if lease.canceled:
            await self._finish_and_release(lease, lease.cancel_reason or "canceled")
        elif token in self.stop_tokens or lease.output_tokens >= lease.max_new_tokens:
            await self._finish_and_release(lease, "completed")
        else:
            self._append_token(row, token)

async def _finish_and_release(self, lease: RequestLease, reason: str) -> None:
    if lease.terminal is not None:
        return
    terminal = "completed" if reason == "completed" else (
        "timed_out" if reason == "deadline_exceeded" else "canceled"
    )
    lease.close(terminal)
    self.kv_manager.release(lease.request_id)
    self.scheduler.remove(lease.request_id)
    self.metrics.observe_terminal(lease, reason)
```

The loop has a subtle ordering rule: observe the completed token, then decide whether it belongs in the visible output. If the deadline expired while the GPU was working, the token may be omitted from the client stream while still counted as device work in internal metrics. Do not report it as user-visible output unless the stream writer actually sent it.

## 2. Timeouts are a budget hierarchy

One timeout is not enough. Inference has at least four different waits: admission, first token, inter-token progress, and total generation. A single large total timeout hides queue collapse; a tiny per-token timeout kills healthy long-context work; a client timeout without server cancellation leaks capacity.

| Budget | Starts at | Ends at | What it protects | Source |
| --- | --- | --- | --- | --- |
| Admission | request accepted | scheduler admits | queue memory and fairness | derived policy |
| TTFT | request accepted | first visible token | user-perceived start | cited: vLLM metric definitions, [2025-09-05](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) |
| ITL | previous token | next visible token | stream progress | cited: vLLM metric definitions, [2025-09-05](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) |
| Total | request accepted | terminal event | end-to-end cost | derived policy |
| Shutdown drain | SIGTERM | process exit | deployment safety | cited: vLLM deep dive, [2025-03-03](https://vllm.ai/blog/2025-03-03-vllm-deep-dive) |

The vLLM anatomy post defines TTFT as submit to first token and TPOT as average time per output token. We can borrow those names without borrowing a benchmark result. In `nanoserve`, record the timestamps separately because queue time and GPU time have different fixes.

### The timeout budget equation

Let $T$ be the total deadline, $Q$ queue time, $P$ prefill time, and $D$ decode-plus-stream time. The request succeeds only if:

$$
Q + P + D \le T
$$

This is an accounting identity, not a performance model. If the request spends 1,200 ms in the queue and the total deadline is 2,000 ms, the engine has 800 ms left for prefill, decode, and writing. A handler that starts a fresh 2-second timer after admission silently changes the contract to $Q + 2{,}000$ ms.

```python
def stage_timeout(lease: RequestLease, safety_ms: float = 5.0) -> float:
    """Return a non-negative budget for one stage."""
    remaining = lease.remaining_s() - safety_ms / 1000.0
    if remaining <= 0:
        lease.request_cancel("deadline_exceeded_before_stage")
        return 0.0
    return remaining


async def admit_or_reject(lease: RequestLease, estimate_ms: float) -> bool:
    if lease.expired():
        lease.request_cancel("deadline_exceeded_in_queue")
        return False
    if estimate_ms > lease.remaining_s() * 1000.0:
        lease.request_cancel("deadline_too_short_for_estimate")
        return False
    await scheduler.admit(lease)
    return True
```

The estimate is allowed to be conservative. Rejecting a request with 20 ms left is not a model failure; it is an honest refusal to start work that cannot meet its own contract. If the service has a fallback, the caller can choose it before GPU admission.

### Per-token timeouts are dangerous

An ITL SLO is a distribution, not a kill switch. A single decode step can exceed its usual interval because of batch composition, a long KV traversal, CUDA graph selection, or a temporary host scheduling pause. Killing the request after one 80 ms interval can waste all previous tokens and trigger a retry exactly when the batch is full.

Use a two-stage rule instead:

- total deadline is hard;
- progress timeout is a warning or a circuit-breaker signal until it repeats across a short window.

For example, flag three consecutive intervals above 100 ms, but still let the total deadline decide the request’s terminal state. Those numbers are policy placeholders, not universal defaults. Label them in configuration and measure the tail that your users actually experience.

## 3. Retries duplicate GPU work unless the server settles the first attempt

Network clients retry because a timeout is ambiguous. The response may have been lost after the server finished, or the request may still be running, or the server may never have accepted it. For an inference endpoint, all three cases matter because a retry can allocate another KV sequence and run another prefill.

![A before-and-after comparison shows blind retries creating three GPU jobs, while cancellation and a terminal request state keep one active job](/imgs/blogs/reliability-timeouts-retries-hedging-and-degraded-modes-3.webp)

The arithmetic is simple. Let $W$ be the GPU work for one generation and $r$ the fraction of attempts that are retried while the first attempt is still active. With one original attempt and one retry, expected work is:

$$
E[W_{\text{total}}] = W + rW = W(1+r)
$$

With two blind retries, it becomes $W(1+2r)$ if each retry overlaps with probability $r$. In the pathological case where every timeout precedes completion, three concurrent attempts consume $3W$ for one logical request. This is an explanatory model; it intentionally ignores cache sharing and different output lengths.

The operational rule is more important than the equation: **retry ownership must be explicit, and a retry must carry an idempotency key that lets the server find or settle the original attempt.** A request ID is not enough if the client generates a new ID for every retry.

### Retry only transient failures

The gRPC [request hedging guide](https://grpc.io/docs/guides/request-hedging/), last modified 2023-10-03, makes the same distinction in a different RPC setting: retry and hedge policies are bounded by maximum attempts, delay, eligible status codes, and throttling. The [service-config guide](https://grpc.io/docs/guides/service-config/), updated 2025, also treats call timeout, retry policy, and hedging policy as separate call behavior.

| Failure | Retry? | Required condition | Backoff | Source |
| --- | --- | --- | --- | --- |
| Queue deadline before admission | Usually no | caller may choose fallback | none | derived policy |
| HTTP 429 / overload | Maybe | honor retry-after and budget | full jitter | cited: [gRPC service config](https://grpc.io/docs/guides/service-config/), 2025 |
| HTTP 503 / transient replica fault | Yes, once | idempotency key and fresh capacity | exponential + jitter | cited: [gRPC request hedging](https://grpc.io/docs/guides/request-hedging/), 2023-10-03 |
| Client disconnect after tokens | Not automatically | server must settle original lease | none | derived from stream contract |
| Deterministic 400 / invalid schema | No | fix request | none | derived policy |
| Non-idempotent side effect in a tool call | No | explicit application key | none | derived safety rule |

Every row has a Source because “retryable” is not a property of the status code alone. It is the status, the operation’s side effects, whether the server accepted the request, and whether the caller can identify the attempt.

### A bounded retry policy

```python
from dataclasses import dataclass
import random


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 2
    base_delay_s: float = 0.05
    max_delay_s: float = 0.5
    retry_budget_fraction: float = 0.10

    def delay_s(self, retry_number: int, rng: random.Random) -> float:
        cap = min(self.max_delay_s, self.base_delay_s * (2 ** retry_number))
        return rng.uniform(0.0, cap)  # full jitter

    def permits(self, retries: int, originals: int) -> bool:
        if originals <= 0:
            return False
        return retries / originals < self.retry_budget_fraction


def can_retry(status: int, *, idempotent: bool, original_settled: bool) -> bool:
    transient = status in {408, 429, 500, 502, 503, 504}
    return transient and idempotent and not original_settled
```

The budget is measured over a window, not per request. A ten-percent retry budget means at most one retry for every ten original attempts in the policy window; it does not mean every request may retry once. If the service is already overloaded, the budget should tighten or close. Retry storms are feedback: a slow server generates timeouts, timeouts generate retries, and retries increase the work that made the server slow.

### Retry at one layer

If the API gateway retries twice and the client library retries twice and a replica-side connector retries twice, one logical request can create $3^3 = 27$ attempts in the worst case. This multiplication is derived from three layers each allowing three attempts; it is not a measurement. Pick one owner, normally the edge that understands the user deadline and idempotency key. Inner layers should return a classified failure quickly.

## 4. Hedging attacks the tail by spending capacity

Hedging is not “retry faster.” It sends a second copy while the first is still alive, usually after a delay, and returns the first complete response. The official [gRPC hedging guide](https://grpc.io/docs/guides/request-hedging/) describes the same race and says outstanding requests are canceled when a winner returns. It also caps `maxAttempts` at five and describes server pushback and retry throttling. Those are useful guardrails for an inference service, but the GPU makes the loser-cancellation requirement non-negotiable.

![A request fans out to two replicas, the first complete response wins, and the losing GPU job is canceled under a ten percent extra-load budget](/imgs/blogs/reliability-timeouts-retries-hedging-and-degraded-modes-4.webp)

Let $X_1$ and $X_2$ be independent response times for two replicas. A hedge returning the first response has latency:

$$
X_{\text{hedged}} = \min(X_1, X_2)
$$

The minimum is usually smaller than either variable’s tail, but independence is the hidden assumption. If both replicas share a saturated GPU pool, the hedge may produce two correlated slow requests and make the tail worse. If the hedge launches after delay $h$, the extra work is incurred only when the first request has not completed by $h$:

$$
P(\text{extra copy}) = P(X_1 > h)
$$

This is why a hedge budget should be expressed as observed extra attempts divided by originals. If 8 of 100 requests launch a hedge, extra attempt load is 8%, before counting any retries from failures. That 8% is reproducible from request logs; it is not an assumed speedup.

### Do not hedge streaming generation blindly

For a non-streaming classification request, the winner is one response body. For a token stream, the first token from replica A is not necessarily a complete winner: replica B may have a lower eventual total latency, and switching streams would duplicate or reorder tokens. The safe choices are:

- hedge only before the first token, then commit to the first replica that emits a valid first event;
- hedge a request with a deterministic seed and discard the loser before any visible token;
- hedge only a small, bounded prefill admission probe, not the whole decode;
- do not hedge at all when the pool is above the pressure threshold.

The first policy is usually the best `nanoserve` starting point. It treats first-token selection as the commit point. After commitment, the losing lease must be canceled, and the winner owns the user stream.

```python
async def first_token_hedge(request, replicas, delay_s: float = 0.12):
    primary = asyncio.create_task(replicas[0].start_stream(request))
    await asyncio.sleep(delay_s)
    if primary.done():
        return await primary

    hedge = asyncio.create_task(replicas[1].start_stream(request))
    done, pending = await asyncio.wait(
        {primary, hedge}, return_when=asyncio.FIRST_COMPLETED
    )
    winner = next(iter(done))
    stream = await winner
    for task in pending:
        task.cancel()
    await replicas.cancel_lease_for_loser(request.request_id, winner)
    return stream
```

This snippet is a control-flow sketch around a real async interface, not a benchmark. A production implementation needs cancellation acknowledgements, a timeout around the loser cleanup, and a rule for what happens if the winner’s first event is an error. If the loser cannot acknowledge cancellation, the hedge should count as leaked capacity and trip the hedge breaker.

### Hedge breakers

The hedge decision should use recent pressure, not only per-request latency. A simple policy has three gates:

1. the request is still before first token;
2. the primary has exceeded a delay chosen from its queue-time histogram;
3. the replica pool’s hedge ratio and KV pressure are below budget.

When any gate fails, wait for the primary or fail within the deadline. The goal is not the smallest p99 in an empty lab. It is the highest goodput under the workload that pays for the extra copy.

## 5. Partial-stream failure is an API state, not an exception string

Streaming changes what “success” means. A non-streaming endpoint can return one status and one body. An SSE endpoint may have sent 37 tokens before the network breaks. The client needs to know whether it saw a complete answer, an intentionally canceled answer, or an answer that stopped because the server failed.

![A tree separates clean completion, mid-stream failure, and client disconnect, then records usage, retry policy, or KV release for each terminal state](/imgs/blogs/reliability-timeouts-retries-hedging-and-degraded-modes-5.webp)

Use explicit terminal events rather than relying on the TCP close reason:

```json
{"type":"token","request_id":"7f3a","text":"The"}
{"type":"token","request_id":"7f3a","text":" answer"}
{"type":"error","request_id":"7f3a","code":"upstream_reset","partial":true,"output_tokens":2}
```

For a clean finish:

```json
{"type":"done","request_id":"7f3a","finish_reason":"stop","usage":{"prompt_tokens":42,"completion_tokens":18}}
```

For a client disconnect, the server may not be able to deliver a final event. It should still write an internal terminal record with `finish_reason=client_disconnect`, `partial=true`, and the number of tokens actually sent or acknowledged by the stream writer. “Usage” must be defined: generated device tokens, serialized tokens, and received tokens are three different counts.

### Stream delivery and GPU ownership

The writer should not own the GPU lease. It consumes events from a bounded queue; the scheduler owns the lease and receives a cancellation signal when the writer sees a broken pipe. This avoids a common deadlock where the writer awaits the GPU future while the GPU waits for the writer’s queue to drain.

```python
async def stream_response(lease, event_queue, send):
    try:
        while True:
            event = await asyncio.wait_for(
                event_queue.get(), timeout=lease.remaining_s()
            )
            await send(event)
            if event["type"] in {"done", "error"}:
                return
    except (BrokenPipeError, ConnectionResetError):
        lease.request_cancel("client_disconnect")
        scheduler.signal_cancel(lease.request_id)
    except asyncio.TimeoutError:
        lease.request_cancel("stream_deadline")
        scheduler.signal_cancel(lease.request_id)
```

Do not turn a client disconnect into a retry by default. The user may have received a useful partial answer, and a retry doubles work. If the product wants resumable generation, make it a separate protocol with a stable request ID, a token prefix checksum, and a stated policy for whether the model is allowed to continue from the partial sequence. A new request with a new sampling seed is not a resume.

### UTF-8 and event boundaries

SSE data is text, but tokenizer output is bytes that may become text only after several tokens. Accumulate decoded bytes and emit valid UTF-8 boundaries; never assume one token equals one Unicode scalar. The same principle applies to JSON fragments: a stream may contain a partial string or escaped sequence when it fails. The terminal event should say `partial=true`; it should not claim that the partial payload is a valid completed JSON document.

## 6. Graceful degradation spends quality before availability

When the GPU is under pressure, “keep serving” is not one switch. We can reduce context, disable optional speculative work, lower output limits, route to a smaller model, or reject new work. Those choices spend different parts of the product contract.

![A matrix maps normal, pressure, overload, and GPU-fault states to shorter context, model fallback, decoding changes, and an explicit outcome](/imgs/blogs/reliability-timeouts-retries-hedging-and-degraded-modes-6.webp)

The degradation ladder should be ordered by reversibility and user harm:

1. Stop admitting work that cannot fit its deadline.
2. Disable speculative decoding if its draft work is hurting capacity.
3. Cap or summarize old context if the product permits it.
4. Route eligible requests to a smaller model with a declared capability boundary.
5. Return a useful partial or an explicit overload error rather than an orphaned request.

The order is not universal. A code-completion product may prefer a smaller model before truncating context; a retrieval product may prefer truncation only after a retrieval summary is available. The important property is that the ladder is explicit, observable, and reversible.

### Shorter context is not free quality

If the prompt has $P$ tokens and the model’s configured context limit is $C$, truncating to $C' < C$ removes $C-C'$ tokens. That arithmetic is obvious; the product decision is not. Removing the oldest chat turns may be harmless for a short-lived question and disastrous for an agent holding a tool contract near the beginning of the context.

Use a structured truncation policy:

```python
def fit_context(messages, max_input_tokens, tokenizer):
    pinned = [m for m in messages if m.get("pinned")]
    tail = [m for m in messages if not m.get("pinned")]
    kept = list(pinned)
    for message in reversed(tail):
        candidate = [*kept, message]
        if len(tokenizer.encode(render(candidate))) > max_input_tokens:
            break
        kept.insert(len(pinned), message)
    return kept, {"truncated": len(kept) != len(messages)}
```

This example is intentionally conservative about pinned messages. A production implementation needs a token budget for the system prompt, tool schema, and output reservation separately. The degradation event should include the old and new input token counts and the reason; otherwise a later quality regression will be impossible to correlate with pressure.

### Smaller model fallback

A smaller model is a product fallback, not an invisible infrastructure trick. The response should carry a model identifier, and evaluation should establish where the fallback is acceptable. A model cascade can be useful for short factual requests and unacceptable for code edits or safety-critical classification.

```python
async def choose_model(request, pressure):
    if pressure < 0.70:
        return "llama-3.1-8b"
    if request.task in {"short_chat", "classification"} and pressure < 0.90:
        return "qwen3-8b"
    if request.task == "summarize" and pressure < 0.95:
        return "small-summary-model"
    raise Overload("no model tier can meet the deadline")
```

The names above are policy labels, not a claim that a particular model is interchangeable. The fixed series spine is Llama-3.1-8B on an RTX 4090 or A100; use a real smaller checkpoint only after measuring task quality.

### No speculative decoding under pressure

Speculative decoding runs a draft model and asks the target model to verify proposed tokens. The vLLM team’s [official post](https://vllm.ai/blog/2024-10-17-spec-decode), dated 2024-10-17, reports up to 1.5× speedup for Llama 3 70B on 4×H100 at QPS 1 with a draft model and up to 2.8× for prompt lookup on CNN/DailyMail at QPS 1. The same post reports 1.4× and 1.8× slowdowns at high QPS for those two workloads. “1.4× slowdown” means latency increased by a factor of 1.4 in that cited setup; it is not a `nanoserve` result.

The policy implication is derived: if draft work adds capacity cost while the target is already saturated, turning speculation off can increase goodput even if it reduces empty-system token speed. Record `speculation_enabled` and the reason for switching it so an operator can distinguish an optimization choice from a model regression.

## 7. The nanoserve diff: one reliability contract across layers

The cleanest implementation has one request object crossing the API, scheduler, and runner. The API does not directly free KV. The runner does not invent HTTP status codes. The scheduler is the authority for admission, cancellation, and terminal state.

![A nine-cell grid maps API request facts to scheduler state and runner cleanup, ending in metrics with usage and terminal reason](/imgs/blogs/reliability-timeouts-retries-hedging-and-degraded-modes-7.webp)

```python
# nanoserve/protocol.py
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GenerateRequest:
    request_id: str
    prompt_token_ids: list[int]
    max_new_tokens: int
    deadline_ns: int
    idempotency_key: str
    stream: bool = True
    allow_degrade: bool = True


@dataclass(frozen=True)
class TerminalEvent:
    request_id: str
    reason: str
    partial: bool
    generated_tokens: int
    model: str


def validate_request(body: dict[str, Any], now_ns: int) -> GenerateRequest:
    request_id = str(body["request_id"])
    deadline_ns = int(body["deadline_ns"])
    if deadline_ns <= now_ns:
        raise ValueError("deadline must be in the future")
    max_new_tokens = min(int(body.get("max_new_tokens", 256)), 2048)
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    return GenerateRequest(
        request_id=request_id,
        prompt_token_ids=list(body["prompt_token_ids"]),
        max_new_tokens=max_new_tokens,
        deadline_ns=deadline_ns,
        idempotency_key=str(body.get("idempotency_key", request_id)),
        stream=bool(body.get("stream", True)),
        allow_degrade=bool(body.get("allow_degrade", True)),
    )
```

The endpoint should reject a reused idempotency key if the original is still running with incompatible parameters. If the original is terminal, it can return the stored terminal result or a conflict, depending on whether the product stores generated text. Do not silently start a second generation.

### Scheduler cancellation and idempotency

```python
# nanoserve/scheduler.py
class Scheduler:
    def __init__(self, kv_manager, metrics):
        self.kv_manager = kv_manager
        self.metrics = metrics
        self.by_id = {}
        self.by_idempotency = {}

    def submit(self, request):
        old = self.by_idempotency.get(request.idempotency_key)
        if old is not None:
            if old.request_id != request.request_id:
                raise Conflict("idempotency key belongs to another request")
            return old
        lease = RequestLease(
            request_id=request.request_id,
            deadline_ns=request.deadline_ns,
            max_new_tokens=request.max_new_tokens,
            metadata={"idempotency_key": request.idempotency_key},
        )
        self.by_id[lease.request_id] = lease
        self.by_idempotency[request.idempotency_key] = lease
        return lease

    def cancel(self, request_id, reason="client_cancel"):
        lease = self.by_id.get(request_id)
        if lease is not None and lease.terminal is None:
            lease.request_cancel(reason)

    def remove(self, request_id):
        lease = self.by_id.pop(request_id, None)
        if lease is not None:
            key = lease.metadata.get("idempotency_key")
            if key:
                self.by_idempotency.pop(key, None)
```

Keep the idempotency record longer than the GPU lease if the client may retry after a network timeout. A short TTL is fine, but it must exceed the longest retry delay plus the time in which a completed response can be lost. The exact TTL is workload policy; measure duplicate submissions and choose it from observed client behavior.

### API disconnect handling

The HTTP framework should translate disconnect into `scheduler.cancel(request_id)`, not into `task.cancel()` alone. With an OpenAI-compatible endpoint, a client may disconnect without a new request ever arriving. The server should still release work.

```python
# nanoserve/api.py
async def generate_endpoint(payload, request, scheduler):
    lease = scheduler.submit(validate_request(payload, time.monotonic_ns()))
    events = scheduler.events(lease.request_id)
    try:
        async for event in events:
            yield encode_sse(event)
            if await request.is_disconnected():
                scheduler.cancel(lease.request_id, "client_disconnect")
                break
    finally:
        if lease.terminal is None:
            scheduler.cancel(lease.request_id, "stream_closed_before_terminal")
```

The `finally` is not redundant. A transport can close between the final token and the final `done` event. The scheduler’s idempotent cleanup makes the second cancellation harmless while still ensuring that a stream path never leaves a lease without an owner.

## 8. Measuring reliability without fooling yourself

A throughput benchmark that counts every generated token as success will reward orphaned GPU work. A reliability benchmark must preserve request-level records, including failures and partial streams.

```python
# bench/reliability_load.py
import asyncio
import json
import time
import uuid


async def one(client, prompt, total_timeout_s):
    request_id = str(uuid.uuid4())
    started = time.monotonic_ns()
    first = None
    tokens = 0
    reason = "unknown"
    try:
        async with asyncio.timeout(total_timeout_s):
            async for event in client.stream(
                request_id=request_id,
                idempotency_key=request_id,
                prompt=prompt,
            ):
                if event["type"] == "token":
                    tokens += 1
                    first = first or time.monotonic_ns()
                if event["type"] in {"done", "error"}:
                    reason = event.get("finish_reason", event["type"])
                    break
    except TimeoutError:
        reason = "client_timeout"
    finished = time.monotonic_ns()
    return {
        "request_id": request_id,
        "ttft_ms": None if first is None else (first - started) / 1e6,
        "total_ms": (finished - started) / 1e6,
        "tokens_received": tokens,
        "reason": reason,
    }


async def main(client, prompts, arrival_s):
    results = []
    for prompt in prompts:
        results.append(asyncio.create_task(one(client, prompt, 2.0)))
        await asyncio.sleep(arrival_s)
    rows = await asyncio.gather(*results)
    for row in rows:
        print(json.dumps(row))
```

This harness is reader-reproducible, not a claim about a result. Run it against a local `nanoserve` build and record the model revision, GPU, driver, CUDA, PyTorch, prompt manifest, arrival process, total deadline, and retry policy. The expected range is intentionally not fabricated here: the reader should first establish a baseline on the fixed RTX 4090 or A100 matrix, then compare changes on the same environment. A useful experiment is a CPU-only fake runner that sleeps for controlled durations; it can validate cancellation, idempotency, and accounting before any GPU is involved.

### Metrics that expose the failure

| Metric | Formula or definition | Healthy signal | Failure signal | Source |
| --- | --- | --- | --- | --- |
| Goodput | completed within SLO / original requests | stable as load rises | falls while GPU utilization stays high | derived definition |
| Retry amplification | all attempts / original requests | near 1.0 | rises with timeout rate | derived |
| Hedge ratio | hedged attempts / original requests | bounded by policy | tracks saturation | derived |
| Orphan gauge | active leases with no consumer | zero or quickly draining | grows after client timeouts | derived |
| Partial-stream rate | partial terminal streams / streams | workload-specific low rate | rises with resets or deadline | derived |
| Cancel-to-free delay | KV release time − cancel time | bounded by one step | grows with queue or blocked runner | reproduce: load script |
| TTFT / TPOT | submit→first token / mean output interval | tails meet SLO | p99 rises before throughput falls | cited: [vLLM anatomy](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm), 2025-09-05 |

Goodput is the scoreboard because throughput can improve when the server keeps doing work for users who already left. Include timed-out requests in the denominator. A timeout is not a zero-latency completion and should never be silently dropped from a percentile calculation.

### Fault-injection matrix

| Fault | Injection | Expected terminal behavior | Expected resource behavior | Source |
| --- | --- | --- | --- | --- |
| Client disconnect after 5 tokens | close stream | `canceled`, partial | KV released at next step | reproduce: fake client |
| Replica reset before first token | return 503 | one bounded retry | first lease settled | reproduce: fake replica |
| Replica reset after first token | break SSE | `failed`, partial | no automatic duplicate decode | derived policy |
| Queue delay exceeds deadline | sleep before admit | `timed_out` | no KV allocation | derived |
| Hedge loser ignores cancel | drop ack | hedge breaker opens | leaked lease visible | derived safety test |
| GPU runner step hangs | stall fake runner | total deadline trips | process-level health signal required | reproduce: fault injection |

All rows are test plans rather than measured outcomes. The point of the table is to make a reliability property executable. If the last column says `reproduce`, the test should print a JSON record and an expected state, not a hand-written “pass.”

## 9. Case studies and sourced numbers

### 9.1 vLLM speculative decoding: speedup is load-dependent

The vLLM team’s 2024-10-17 post reports up to 1.5× speedup on ShareGPT at QPS 1 for Llama 3 70B on 4×H100 with a draft model, and up to 2.8× on CNN/DailyMail prompt lookup at QPS 1. Under high QPS, the same post reports 1.4× slowdown for the draft-model case and 1.8× slowdown for prompt lookup. These are cited values with the model, hardware, workload, and load setup in the source; they are not estimates for `nanoserve`.

The reliability conclusion is not “never use speculation.” It is “treat speculation as a mode with a capacity cost.” The scheduler can disable it under pressure, just as it can shorten context. The control plane should record the mode in every benchmark row so a change in TPOT is explainable.

### 9.2 gRPC hedging: bounded races are a protocol feature

The gRPC request-hedging guide, last modified 2023-10-03, defines `maxAttempts`, `hedgingDelay`, non-fatal status codes, cancellation of outstanding requests after a successful response, retry throttling, and server pushback. The guide caps `maxAttempts` at five. That cap is a cited property of gRPC, not an ideal number for GPU inference.

The useful design pattern is the set of controls. A hedge is not a free duplicate; it is a bounded, delayed, cancelable attempt under server feedback. `nanoserve` should borrow the shape and choose its own policy from hedge ratio, KV pressure, and goodput measurements.

### 9.3 vLLM’s operational metrics: TTFT and TPOT are not enough

The vLLM anatomy post dated 2025-09-05 defines TTFT, ITL, and TPOT and describes a latency benchmark with stated input/output and load conditions. Those names are valuable because they separate “the user waited for admission and prefill” from “the user waited between decode tokens.” They do not tell us whether a request that timed out released its KV blocks.

Add terminal reason, cancel-to-free delay, retries, hedges, and partial status to the same trace. A p99 TTFT improvement achieved by dropping timed-out requests from the sample is a reporting bug, not an optimization.

### 9.4 A derived duplicate-work incident

Consider a service with 20 original requests per second. If 15% time out and every timeout retries once before the original is settled, the attempt rate is:

$$20 + (0.15 \times 20) = 23\text{ attempts/s}$$

That is 15% extra offered work. If the first attempt remains active for every timeout, the instantaneous overlap can be higher than 23 attempts/s because the original and retry coexist. If a second retry is also allowed, the worst-case attempt rate under the same independent 15% timeout fraction is:

$$20 + (0.15 \times 20) + (0.15^2 \times 20) = 23.45\text{ attempts/s}$$

These are derived arithmetic examples, not a production measurement. The result is enough to justify a retry budget and a dashboard counter before tuning model kernels.

## 10. When to reach for each mechanism

### Use hard deadlines when

- the caller has a real user or tool budget;
- queue time is a first-class part of the SLO;
- the server can cancel and release at a scheduler boundary;
- terminal records remain available after the client disconnects.

### Use retries when

- the failure is transient and classified;
- the operation is idempotent or has a stable idempotency key;
- one layer owns the retry budget;
- the retry can fit the remaining deadline and fresh capacity exists.

### Use hedges when

- tail latency comes from replica variance rather than shared saturation;
- the hedge happens before any token is visible;
- the loser can be canceled and its KV lease observed as free;
- the extra-attempt ratio is bounded and improves goodput, not just p99.

### Degrade when

- preserving a useful answer is better than preserving maximum context or optional decoding;
- fallback models have task-specific quality gates;
- the response identifies the degraded mode;
- returning a partial result is a documented product behavior.

### Do not build this yourself when

You need broad model coverage, mature continuous batching, production-grade KV management, multi-GPU scheduling, and a maintained OpenAI-compatible endpoint today. Use vLLM or another established serving engine and add the policy at a layer where its cancellation and metrics are supported. `nanoserve` is valuable here because the invariants are small enough to inspect: one lease, one deadline, one terminal transition, one cleanup owner.

Do not add hedging to hide a saturated shared GPU pool. Do not retry a stream after the client has received tokens unless you have designed resume semantics. Do not silently truncate context or swap models without exposing the mode. Do not treat a high GPU utilization number as proof of useful work; orphaned and duplicated generations can keep the device busy while goodput collapses.

## Key takeaways

1. A client timeout that does not cancel the scheduler lease is only a UI timeout.
2. Carry an absolute deadline through queue, prefill, decode, and stream writing; never reset a fresh full timeout at each hop.
3. Retry only classified transient failures, with idempotency and one explicit retry owner.
4. A hedge is a capacity-spending race: delay it, bound it, cancel the loser, and disable it under pressure.
5. A stream needs explicit clean, failed-partial, canceled, and timed-out terminal states.
6. Release KV at a scheduler step boundary, and measure cancel-to-free delay.
7. Degrade optional quality before availability: shorter context, no speculation, bounded output, eligible smaller model, then explicit overload.
8. Count timed-out and partial requests in goodput denominators.
9. Treat cited vLLM performance numbers as evidence about their setup, never as `nanoserve` measurements.
10. Use vLLM for production breadth; use `nanoserve` to understand and test the reliability contract.

## Further reading

- [What inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is) and [the inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook).
- [The scheduler as a policy problem](/blog/machine-learning/inference-engineering/the-scheduler-as-a-policy-problem) for admission and scheduling ownership.
- [An experiment protocol for inference benchmarks](/blog/machine-learning/inference-engineering/an-experiment-protocol-for-inference-benchmarks) for load generation and honest measurement.
- [Request hedging](https://grpc.io/docs/guides/request-hedging/) and [gRPC service config](https://grpc.io/docs/guides/service-config/) for bounded retry and hedge controls.
- [How speculative decoding boosts vLLM performance](https://vllm.ai/blog/2024-10-17-spec-decode), 2024-10-17, for the cited speedup/slowdown trade-off.
- [Inside vLLM: Anatomy of a High-Throughput Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm), 2025-09-05, for TTFT, ITL, TPOT, and engine context.

<figure class="blog-anim">
<svg viewBox="0 0 760 260" role="img" aria-label="A request token stream fills a GPU work bar, then a timeout cancels the stream and the bar drains" style="width:100%;height:auto;max-width:860px">
<style>
.rt-track{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}
.rt-fill{fill:var(--accent,#6366f1)}
.rt-stop{fill:#dc2626;opacity:.12}
.rt-label{font:600 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.rt-small{font:500 14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.rt-dot{fill:var(--accent,#6366f1)}
@keyframes rt-work{0%,12%{transform:scaleX(.05)}42%{transform:scaleX(.72)}52%,100%{transform:scaleX(.05)}}
@keyframes rt-cancel{0%,42%{opacity:0}48%,100%{opacity:1}}
@keyframes rt-dot{0%,12%{transform:translateX(0);opacity:0}20%{opacity:1}42%{transform:translateX(430px);opacity:1}52%,100%{transform:translateX(430px);opacity:0}}
.rt-work{animation:rt-work 8s ease-in-out infinite;transform-box:fill-box;transform-origin:left center}
.rt-cancel{animation:rt-cancel 8s ease-in-out infinite}
.rt-dot{animation:rt-dot 8s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.rt-work{animation:none;transform:scaleX(.05)}.rt-cancel{animation:none;opacity:1}.rt-dot{animation:none;opacity:0}}
</style>
<text class="rt-label" x="40" y="42">GPU decode lease</text>
<rect class="rt-track" x="40" y="62" width="680" height="34" rx="8"/>
<rect class="rt-fill rt-work" x="40" y="62" width="680" height="34" rx="8"/>
<circle class="rt-dot" cx="40" cy="79" r="10"/>
<rect class="rt-stop" x="450" y="52" width="140" height="54" rx="8"/>
<text class="rt-label rt-cancel" x="465" y="142">deadline fires → cancel</text>
<text class="rt-small" x="40" y="150">0 ms</text>
<text class="rt-small" x="386" y="150">2,000 ms</text>
<text class="rt-small" x="650" y="150">next step</text>
<text class="rt-label" x="40" y="205">Safe timeout</text>
<text class="rt-small" x="40" y="230">client stops waiting and the runner releases KV at a scheduler step boundary</text>
</svg>
<figcaption>The moving lease grows while decode runs; when the deadline fires, cancellation drains the work instead of leaving an orphaned GPU job.</figcaption>
</figure>

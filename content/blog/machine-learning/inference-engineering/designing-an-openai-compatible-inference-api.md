---
title: "Designing an OpenAI-compatible inference API: The engine boundary is the product"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Build an OpenAI-shaped API for nanoserve with a precise SSE contract, cancellation that reaches GPU work, exact usage accounting, and retry-safe idempotency."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "api-design",
    "streaming",
    "cancellation",
    "idempotency",
    "latency",
    "pytorch",
    "gpu",
    "ml-systems",
    "vllm",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 28
---

Your model is already generating tokens. The Python client can already send a prompt. Then a user closes a browser tab halfway through a 2,000-token answer, and the GPU continues decoding the request until `max_tokens`. The user sees nothing, the allocator sees a live sequence, and the next request waits behind work that has no consumer.

That is not an HTTP bug. It is an engine-boundary bug. The API is the place where a socket, a retry, a token counter, a scheduler request, and a CUDA stream become one lifecycle. Figure 1 is the mental model: the public surface is deliberately familiar, but the adapter turns it into a small internal contract before `nanoserve` admits any GPU work.

![The API adapter normalizes an OpenAI request then branches into tokenization and policy defaults before engine admission](/imgs/blogs/designing-an-openai-compatible-inference-api-1.webp)

By the end, we will have the design and runnable Python diffs for `nanoserve/api.py`, `nanoserve/protocol.py`, and `nanoserve/request_registry.py`. We will define `/v1/models` and `/v1/chat/completions`, stream one JSON object per SSE record, propagate disconnects to a cooperative decode stop point, finalize usage exactly once, and make an `Idempotency-Key` retry replay a result instead of creating a second sequence. This is a design post, not a GPU benchmark: I have no GPU and have run none of these measurements.

The fixed spine is Llama-3.1-8B on one RTX 4090 or A100. The implementation does not pretend that a production engine is a few FastAPI routes. It exposes the seam where the already-built tokenizer, KV cache, scheduler, sampler, and runner meet a transport that users can actually depend on. That is the next step after [what inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is), and it points toward [the inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook).

## 1. The OpenAI-shaped surface

The compatibility target is a shape, not a promise that every field has identical semantics. A client should be able to replace `base_url` and keep its request construction, streaming loop, and error handling. The server should still reject fields it cannot honor rather than silently pretending to support them.

The smallest useful public surface has three endpoints:

| Endpoint | Contract | nanoserve owner | Source |
| --- | --- | --- | --- |
| `GET /v1/models` | list model IDs and capabilities | adapter metadata | derived: one configured model |
| `POST /v1/chat/completions` | messages, sampling, stream, usage | API adapter + engine | cited: OpenAI API shape, accessed 2026-08-03 |
| `GET /healthz` | process and model readiness | process supervisor | derived: local health state |

The request body can accept the familiar fields `model`, `messages`, `temperature`, `top_p`, `max_tokens` or `max_completion_tokens`, `stop`, `stream`, `stream_options`, `tools`, and `user`. Normalize aliases once. Do not let the scheduler learn that OpenAI called the field `max_tokens`; it needs `max_new_tokens` and a stop policy.

### The compatibility boundary

There are three categories of fields.

| Field class | Examples | Policy |
| --- | --- | --- |
| model semantics | `messages`, `temperature`, `top_p`, `stop` | validate and pass to tokenizer/sampler |
| transport semantics | `stream`, `stream_options.include_usage`, request headers | consume at the adapter |
| engine policy | batch cap, KV reservation, priority, preemption | server config only, never client-controlled by default |

The third row is where naïve wrappers become operational liabilities. If a client can ask for arbitrary `max_num_batched_tokens`, it can influence every other tenant's latency. The public API may expose a bounded `priority` later, but it should not expose a raw block allocator or CUDA graph choice.

Here is the first diff. It is runnable on CPU because it only normalizes data; the real engine adapter can replace `EngineRequest` without changing the public route.

```python
# nanoserve/api_types.py
from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class EngineRequest:
    request_id: str
    model: str
    prompt_tokens: tuple[int, ...]
    max_new_tokens: int
    temperature: float
    top_p: float
    stop: tuple[str, ...]
    stream: bool
    include_usage: bool
    metadata: dict[str, str] = field(default_factory=dict)

def normalize_request(body: dict[str, Any], request_id: str, tokens: list[int]) -> EngineRequest:
    limit = body.get("max_completion_tokens", body.get("max_tokens", 256))
    if not isinstance(limit, int) or not 1 <= limit <= 8192:
        raise ValueError("max_completion_tokens must be between 1 and 8192")
    temperature = float(body.get("temperature", 1.0))
    top_p = float(body.get("top_p", 1.0))
    if not 0.0 <= temperature <= 2.0 or not 0.0 < top_p <= 1.0:
        raise ValueError("temperature or top_p is outside the supported range")
    raw_stop = body.get("stop", ())
    stop = (raw_stop,) if isinstance(raw_stop, str) else tuple(raw_stop or ())
    stream_options = body.get("stream_options") or {}
    return EngineRequest(
        request_id=request_id,
        model=str(body["model"]),
        prompt_tokens=tuple(tokens),
        max_new_tokens=limit,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
        stream=bool(body.get("stream", False)),
        include_usage=bool(stream_options.get("include_usage", False)),
    )
```

The exact validation range above is a nanoserve policy, not an OpenAI guarantee. It is intentionally local and testable. If a model supports a larger context, change the policy and its tests together. Never accept a field merely because the JSON parser can store it.

#### Worked example: one request crossing the boundary

Suppose the tokenizer returns 37 prompt tokens and the client asks for `max_tokens: 128`. The adapter stores `prompt_tokens=37` and `max_new_tokens=128`; it does not store a character count. The engine's maximum possible output-token accounting is therefore $37 + 128 = 165$ tokens before any stop sequence or early finish. This is derived from the request after tokenization, not a benchmark.

## 2. The request lifecycle

The hardest API bugs are state bugs. A request can be accepted by HTTP, waiting in the engine queue, holding KV blocks, streaming a partial answer, and already disconnected from the client. If those facts live in separate booleans, one error path will eventually charge twice, free twice, or keep decoding forever.

![The request lifecycle orders admission prefill decode streaming cancellation usage and one terminal outcome](/imgs/blogs/designing-an-openai-compatible-inference-api-2.webp)

Use one explicit state machine and monotonic timestamps. The state names are observable enough for logs but private enough to change the scheduler later:

```python
# nanoserve/lifecycle.py
from dataclasses import dataclass, field
from enum import Enum
from time import monotonic

class Phase(str, Enum):
    RECEIVED = "received"
    QUEUED = "queued"
    PREFILL = "prefill"
    DECODE = "decode"
    STREAMING = "streaming"
    TERMINAL = "terminal"

@dataclass
class Lifecycle:
    request_id: str
    phase: Phase = Phase.RECEIVED
    events: list[tuple[str, float]] = field(default_factory=list)
    finish_reason: str | None = None

    def move(self, phase: Phase) -> None:
        allowed = {
            Phase.RECEIVED: {Phase.QUEUED, Phase.TERMINAL},
            Phase.QUEUED: {Phase.PREFILL, Phase.TERMINAL},
            Phase.PREFILL: {Phase.DECODE, Phase.TERMINAL},
            Phase.DECODE: {Phase.STREAMING, Phase.TERMINAL},
            Phase.STREAMING: {Phase.DECODE, Phase.TERMINAL},
            Phase.TERMINAL: set(),
        }
        if phase not in allowed[self.phase]:
            raise RuntimeError(f"illegal transition {self.phase} -> {phase}")
        self.phase = phase
        self.events.append((phase.value, monotonic()))

    def finish(self, reason: str) -> None:
        if self.phase is Phase.TERMINAL:
            return
        self.finish_reason = reason
        self.move(Phase.TERMINAL)
```

The `finish` method is idempotent. That matters because three independent events can race: the model emits EOS, the client disconnect watcher fires, and a request deadline expires. The first terminal reason wins; later paths observe `TERMINAL` and only release their local resources.

The lifecycle also gives us TTFT and TPOT without guessing. If \`queued_at\` is the timestamp for \`QUEUED\`, \`first_token_at\` is the first \`STREAMING\` timestamp, and \`last_token_at\` is the final token timestamp, then $\text{TTFT} = \text{first_token_at} - \text{queued_at}$. If $n$ output tokens took the interval from first to last token, then the average TPOT is $(\text{last_token_at} - \text{first_token_at}) / \max(1,n-1)$. These are server-side clocks, not client arrival times.

## 3. SSE is a wire contract

Streaming is not `yield token_text`. Server-Sent Events (SSE) is a framing protocol: the response uses `Content-Type: text/event-stream`, each event is separated by a blank line, and each `data:` field carries a payload. The [MDN SSE guide](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events), accessed 2026-08-03, documents the `event`, `data`, `id`, and `retry` fields. OpenAI-compatible chat streaming conventionally sends JSON chat-completion chunks followed by `data: [DONE]`.

![The SSE comparison shows arbitrary text chunks on the left and complete data records with a terminal DONE event on the right](/imgs/blogs/designing-an-openai-compatible-inference-api-3.webp)

The contract should survive a proxy that splits a TCP write anywhere. Never make a client infer event boundaries from token text. The server writes one complete SSE record at a time, but TCP and HTTP/2 may split that record into several reads; the client parser joins bytes until the blank-line delimiter.

```python
# nanoserve/sse.py
import json
from collections.abc import Iterator
from typing import Any

def sse(data: dict[str, Any] | str, *, event: str | None = None, event_id: str | None = None) -> bytes:
    payload = data if isinstance(data, str) else json.dumps(data, separators=(",", ":"))
    lines = []
    if event:
        lines.append(f"event: {event}")
    if event_id:
        lines.append(f"id: {event_id}")
    for line in payload.splitlines() or [""]:
        lines.append(f"data: {line}")
    return ("\n".join(lines) + "\n\n").encode("utf-8")

def stream_chunks(chunks: Iterator[dict[str, Any]], usage: dict[str, int] | None = None):
    for chunk in chunks:
        yield sse(chunk)
    if usage is not None:
        yield sse({"object": "chat.completion.chunk", "choices": [], "usage": usage})
    yield sse("[DONE]")

if __name__ == "__main__":
    sample = list(stream_chunks(iter([{"id": "chatcmpl_demo", "choices": [{"delta": {"content": "hi"}}]}]), {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}))
    print(b"".join(sample).decode())
```

The final usage object is optional only when the client did not request it. When `include_usage` is true, send it before `[DONE]`, with an empty `choices` array, so the final token delta and the accounting record are distinct. If the connection fails before that record reaches the client, the server ledger can still be correct; the client response is not the billing database.

The response headers are part of the contract too:

```python
# FastAPI route fragment
from fastapi.responses import StreamingResponse

return StreamingResponse(
    body_iterator,
    media_type="text/event-stream",
    headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Request-Id": request_id,
        "X-Accel-Buffering": "no",
    },
)
```

`X-Accel-Buffering: no` is a deployment hint for nginx, not an application-level guarantee. Test the actual ingress. A proxy that buffers 32 KB can turn a token stream into a bursty pseudo-batch even when the GPU is producing one token per step.

## 4. Cancellation must reach the GPU

Closing the HTTP response is an observation. Cancellation is a control message. The distance between those two statements is where GPU waste hides.

![Cancellation travels from client disconnect through a request event and scheduler safe point before KV memory is released](/imgs/blogs/designing-an-openai-compatible-inference-api-4.webp)

<figure class="blog-anim">
<svg viewBox="0 0 760 210" role="img" aria-label="A request flows from HTTP through the queue and model steps to an SSE stream while a cancellation marker stops the next decode step" style="width:100%;height:auto;max-width:860px">
<style>
.api-flow-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}.api-flow-label{font:600 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}.api-flow-line{stroke:var(--border,#d1d5db);stroke-width:3}.api-flow-dot{fill:var(--accent,#6366f1)}.api-flow-stop{fill:#ffc9c9;stroke:var(--text-primary,#1f2937);stroke-width:2;opacity:0}.api-flow-run{animation:api-flow-move 8s ease-in-out infinite}.api-flow-cancel{animation:api-flow-stop 8s ease-in-out infinite}@keyframes api-flow-move{0%{transform:translateX(0);opacity:0}12%{opacity:1}76%{transform:translateX(520px);opacity:1}88%,100%{transform:translateX(520px);opacity:0}}@keyframes api-flow-stop{0%,48%{opacity:0}58%,76%{opacity:1}88%,100%{opacity:0}}@media (prefers-reduced-motion:reduce){.api-flow-run,.api-flow-cancel{animation:none}.api-flow-run{transform:translateX(260px);opacity:1}.api-flow-cancel{opacity:1}}
</style>
<line class="api-flow-line" x1="70" y1="105" x2="690" y2="105"/>
<rect class="api-flow-box" x="20" y="65" width="120" height="80" rx="10"/><rect class="api-flow-box" x="180" y="65" width="120" height="80" rx="10"/><rect class="api-flow-box" x="340" y="65" width="120" height="80" rx="10"/><rect class="api-flow-box" x="500" y="65" width="120" height="80" rx="10"/><rect class="api-flow-box" x="650" y="65" width="90" height="80" rx="10"/>
<text class="api-flow-label" x="80" y="100">HTTP</text><text class="api-flow-label" x="80" y="122">request</text><text class="api-flow-label" x="240" y="100">queue</text><text class="api-flow-label" x="240" y="122">admit</text><text class="api-flow-label" x="400" y="100">prefill</text><text class="api-flow-label" x="400" y="122">KV</text><text class="api-flow-label" x="560" y="100">decode</text><text class="api-flow-label" x="560" y="122">safe point</text><text class="api-flow-label" x="695" y="100">SSE</text><text class="api-flow-label" x="695" y="122">delta</text>
<circle class="api-flow-dot api-flow-run" cx="80" cy="40" r="10"/><rect class="api-flow-stop api-flow-cancel" x="360" y="18" width="120" height="34" rx="8"/><text class="api-flow-label api-flow-cancel" x="420" y="41">cancel: stop next step</text>
</svg>
<figcaption>The request advances through engine-owned safe points; cancellation prevents the next decode step instead of pretending to interrupt a running kernel.</figcaption>
</figure>

In an async server, `await request.is_disconnected()` tells the route that the peer is gone. It does not interrupt a CUDA kernel already executing, and it should not mutate a scheduler queue from the HTTP task. The route marks a cancellation event; the engine observes it at a safe point between model steps, removes the sequence from the next batch, and releases KV blocks on the engine's owning thread.

```python
# nanoserve/cancellation.py
import asyncio
from dataclasses import dataclass, field

@dataclass
class Cancellation:
    event: asyncio.Event = field(default_factory=asyncio.Event)
    reason: str | None = None

    def cancel(self, reason: str) -> None:
        if not self.event.is_set():
            self.reason = reason
            self.event.set()

    def cancelled(self) -> bool:
        return self.event.is_set()

async def watch_disconnect(request, cancellation: Cancellation) -> None:
    while not cancellation.cancelled():
        if await request.is_disconnected():
            cancellation.cancel("client_disconnect")
            return
        await asyncio.sleep(0.05)
```

The 50 ms polling interval above is a policy example, not a measured latency. A reader can reproduce its detection envelope by running a local ASGI server and closing a client socket at known times; expected detection is roughly one polling interval plus event-loop scheduling, not zero. For a production server, prefer a framework disconnect event when it is reliable, and retain a deadline as a second stop source.

The engine hook must be cooperative:

```python
# nanoserve/engine.py — interface used by the scheduler
class Engine:
    async def generate(self, req, cancellation, on_token):
        self.scheduler.enqueue(req)
        try:
            while True:
                if cancellation.cancelled():
                    self.scheduler.cancel(req.request_id, cancellation.reason or "cancelled")
                    return "cancelled"
                batch = self.scheduler.next_batch()
                if not batch:
                    await asyncio.sleep(0)
                    continue
                # The model runner owns the CUDA stream. Never call blocking
                # torch work from the HTTP event-loop thread in real code.
                results = await self.runner.step(batch)
                for result in results:
                    if result.request_id == req.request_id:
                        await on_token(result)
                if self.scheduler.finished(req.request_id):
                    return "stop"
        finally:
            self.scheduler.release(req.request_id)
```

That comment is important: `asyncio` cannot preempt a synchronous CUDA call. If `runner.step` is a Python function that launches a kernel and immediately returns while the GPU is still executing, the cancellation flag can stop the next step but not the current one. This is the right guarantee: stop at the next safe engine boundary, not “kill a kernel” by wishful thinking.

What counts as a safe point? After the runner has completed the forward pass, copied the selected token to host-visible state, appended KV for that token, and before the scheduler builds the next batch. Stopping earlier risks a half-updated request. Stopping later spends another decode step. The [vLLM anatomy reference](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm), dated 2025-09-05, is a useful benchmark target here: a mature engine owns request queues and KV management centrally rather than allowing route handlers to reach into device memory.

### Cancellation and output semantics

Do not emit a fake normal finish after cancellation. If a client disconnects, there may be no useful final SSE record at all. If the client explicitly asks for cancellation through a control endpoint, return a terminal status such as `cancelled` in that control response and record `finish_reason="cancelled"` server-side. The next retry must be a new logical request unless it carries the same idempotency key and the previous operation is replayable.

## 5. Usage is a ledger

Token usage is not `len(text)`. It is also not necessarily the number of visible output tokens. A reasoning-capable model, a cached prefix, a tokenizer with special chat-template tokens, and a disconnected stream all make character-based billing wrong.

![The usage matrix assigns prompt output reasoning and cached tokens to tokenizer decoder cache and terminal ledger boundaries](/imgs/blogs/designing-an-openai-compatible-inference-api-5.webp)

At minimum, record these fields:

| Counter | Meaning | Owner | Source |
| --- | --- | --- | --- |
| `prompt_tokens` | tokens accepted by the engine after chat templating | tokenizer boundary | derived: `len(input_ids)` |
| `completion_tokens` | tokens actually emitted by the runner | decode boundary | derived: one increment per selected token |
| `cached_tokens` | prompt tokens served from a verified prefix cache | cache lookup | derived: cache hit metadata |
| `total_tokens` | sum of billable input and output counters | terminal ledger | derived: addition shown below |
| reasoning fields | hidden/reasoning output if model exposes it | model runner | cited: model/API-specific |

For the simple non-reasoning case:

$$
\text{total\_tokens} = \text{prompt\_tokens} + \text{completion\_tokens}
$$

If the provider's billing rule discounts cached prompt tokens, that is a separate cost calculation. Do not overwrite `prompt_tokens` with `prompt_tokens - cached_tokens`; clients use the former to understand the request, while billing uses an explicit price policy.

```python
# nanoserve/usage.py
from dataclasses import dataclass
from threading import Lock

@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens + self.reasoning_tokens

class UsageLedger:
    def __init__(self, prompt_tokens: int, cached_tokens: int = 0):
        self._lock = Lock()
        self._prompt = prompt_tokens
        self._cached = cached_tokens
        self._completion = 0
        self._reasoning = 0
        self._final: Usage | None = None

    def token(self, *, reasoning: bool = False) -> None:
        with self._lock:
            if self._final is not None:
                return
            if reasoning:
                self._reasoning += 1
            else:
                self._completion += 1

    def finalize(self) -> Usage:
        with self._lock:
            if self._final is None:
                self._final = Usage(self._prompt, self._completion, self._cached, self._reasoning)
            return self._final
```

The lock is not there because Python integers are difficult. It documents the ownership boundary: token callbacks, cancellation, and terminal cleanup may arrive from different tasks. More importantly, `finalize` is idempotent. A cancelled request with 12 output tokens still has 12 output tokens; cancellation changes the finish reason, not history.

The ledger must be created after tokenization and before admission. If the tokenizer fails, there is no engine request and no usage record except an error metric. If a request is rejected for lack of KV capacity before prefill, record the rejection separately; do not claim zero-token completion.

#### Worked example: cached prompt accounting

A request has 1,024 input tokens, of which 768 match a verified prefix cache, and generates 96 visible output tokens. Derived usage is `prompt_tokens=1024`, `cached_tokens=768`, `completion_tokens=96`, and `total_tokens=1120` because $1024 + 96 = 1120$. The cached count is metadata, not a subtraction from the public prompt count. If the price policy charges only 25% for cached input, a separate cost ledger would calculate $0.25 \times 768 + 1 \times (1024-768)$ input-token units; that pricing expression is illustrative and must be replaced with the deployment's published price.

## 6. Idempotency prevents duplicate GPU work

Retries are normal. A mobile client loses Wi-Fi after receiving token 42; a load balancer retries a request after a 504; an SDK retries a connection reset. Without idempotency, the same logical user action is admitted twice and consumes two KV allocations.

![The idempotency decision tree checks key absence pending completion and payload mismatch before engine admission](/imgs/blogs/designing-an-openai-compatible-inference-api-6.webp)

An `Idempotency-Key` is a client-provided name for one operation. The registry stores a canonical request fingerprint, lifecycle state, and either a replayable result or a stream-resumption policy. The key is not the request ID: a request ID identifies one server attempt; the key identifies the client operation across attempts.

```python
# nanoserve/request_registry.py
import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from threading import Lock

class IdempotencyState(str, Enum):
    PENDING = "pending"
    COMPLETE = "complete"
    FAILED = "failed"

@dataclass
class Entry:
    fingerprint: str
    state: IdempotencyState
    request_id: str
    result: dict | None = None

def fingerprint(body: dict) -> str:
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()

class Registry:
    def __init__(self):
        self._lock = Lock()
        self._entries: dict[str, Entry] = {}

    def reserve(self, key: str, body: dict, request_id: str) -> tuple[str, Entry]:
        fp = fingerprint(body)
        with self._lock:
            old = self._entries.get(key)
            if old is None:
                entry = Entry(fp, IdempotencyState.PENDING, request_id)
                self._entries[key] = entry
                return "admit", entry
            if old.fingerprint != fp:
                return "mismatch", old
            if old.state is IdempotencyState.PENDING:
                return "attach", old
            return "replay", old

    def complete(self, key: str, result: dict) -> None:
        with self._lock:
            entry = self._entries[key]
            entry.state = IdempotencyState.COMPLETE
            entry.result = result
```

There is a deliberate limitation: replaying a completed non-streaming result is straightforward; replaying an SSE stream requires a retained event log or a resumable sequence. We will not claim that `Last-Event-ID` alone solves it. The [MDN SSE documentation](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events) says an event `id` updates the client's last event ID, but the server still has to retain and replay events for reconnect to be meaningful.

For streaming, choose one of three policies and document it:

| Policy | Retry behavior | GPU consequence | Source |
| --- | --- | --- | --- |
| no replay | return a stable error after disconnect | no duplicate if key remains pending | derived: registry state |
| event log | replay retained deltas, then attach live tail | one engine request | derived: bounded log |
| restartable | replay prefix then start a new engine attempt | duplicate GPU work after disconnect | derived: new admission |

For nanoserve, start with no replay for streaming and replay for non-streaming. It is less magical and makes the cost explicit. Add a bounded event log only after measuring how often clients need reconnects and how much memory a log consumes.

The registry needs expiry. A key held forever is a memory leak; a key expired too early permits a duplicate. Let the retention window be $T$ seconds and the arrival rate be $\lambda$ keys per second. The expected live-key count is approximately $\lambda T$ by Little's law. At 20 keys/s and a 10-minute retention window, that is $20 \times 600 = 12{,}000$ entries before payload storage. This is derived sizing arithmetic; profile the actual Python object size with `tracemalloc` on the reader's machine.

## 7. The engine boundary

The API should not call `model.generate()` and hope that cancellation, usage, and streaming appear. The endpoint needs a narrow engine interface with explicit ownership:

![The API stack keeps transport and protocol above a narrow request registry while nanoserve owns scheduler KV cache runner and GPU execution](/imgs/blogs/designing-an-openai-compatible-inference-api-7.webp)

```python
# nanoserve/engine_boundary.py
from collections.abc import AsyncIterator
from dataclasses import dataclass

@dataclass(frozen=True)
class TokenEvent:
    request_id: str
    token_id: int
    text: str
    is_reasoning: bool = False
    finished: bool = False
    finish_reason: str | None = None

class InferenceEngine:
    async def submit(self, request: EngineRequest, cancel) -> AsyncIterator[TokenEvent]:
        """Yield engine-owned token events; caller owns transport framing."""
        raise NotImplementedError

    def cancel(self, request_id: str, reason: str) -> None:
        raise NotImplementedError

    def snapshot(self) -> dict[str, int | str]:
        raise NotImplementedError
```

The adapter owns JSON validation, authentication, request IDs, SSE serialization, and HTTP errors. The engine owns token IDs, KV blocks, scheduler admission, CUDA streams, stop conditions, and the usage callbacks. The registry owns idempotency. Each side can be tested without booting a GPU.

Here is the route shape tying those pieces together. It intentionally leaves `tokenizer.encode_messages` and `engine.submit` as injected dependencies.

```python
# nanoserve/api.py
import uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    request_id = "req_" + uuid.uuid4().hex
    key = request.headers.get("Idempotency-Key")
    if not isinstance(body.get("messages"), list) or not body.get("model"):
        raise HTTPException(400, "model and messages are required")
    tokens = tokenizer.encode_messages(body["messages"])
    if key:
        action, entry = registry.reserve(key, body, request_id)
        if action == "mismatch":
            raise HTTPException(409, "Idempotency-Key was reused with a different request")
        if action == "replay":
            return JSONResponse(entry.result)
        if action == "attach":
            raise HTTPException(409, "stream replay is not enabled for this deployment")
    req = normalize_request(body, request_id, tokens)
    cancel = Cancellation()
    watcher = asyncio.create_task(watch_disconnect(request, cancel))
    try:
        events = engine.submit(req, cancel)
        if not req.stream:
            result = await collect_completion(events, req)
            if key:
                registry.complete(key, result)
            return JSONResponse(result)
        return StreamingResponse(
            stream_response(events, req, key),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Request-Id": request_id},
        )
    finally:
        watcher.cancel()
```

The snippet is a route skeleton because the ASGI response body is consumed after the route returns; in a production implementation, the watcher lifetime belongs inside `stream_response` and its `finally` block. That distinction is not cosmetic. A local variable can leave scope while the stream is still alive. The complete adapter should make stream ownership obvious:

```python
async def stream_response(events, req, key):
    ledger = UsageLedger(len(req.prompt_tokens))
    try:
        async for event in events:
            ledger.token(reasoning=event.is_reasoning)
            yield sse(make_chunk(req, event))
            if event.finished:
                usage = ledger.finalize()
                if req.include_usage:
                    yield sse(make_usage_chunk(req, usage))
                yield sse("[DONE]")
                return
    except asyncio.CancelledError:
        req_cancel(req.request_id, "response_task_cancelled")
        raise
    finally:
        # This is a control-plane cleanup; the engine owns device release.
        req_cancel(req.request_id, "stream_closed")
```

Do not increment completion usage for a synthetic finish event. Count selected tokens, then finalize once. If the runner emits a final event with no token, `ledger.token` must not run for it; model that distinction in `TokenEvent` or use a separate `FinishEvent` type.

### What the boundary buys us

Once the seam is explicit, the API can support both an in-process runner and a remote engine. The transport tests can feed fake token events and assert exact bytes. The engine tests can feed cancellation at every safe point and assert that `scheduler.cancel` is called once. The accounting tests can run on CPU. The GPU does not need to be present to prove protocol behavior.

This is also where the [API design playbook](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2) belongs: an API is a set of observable invariants, not a list of routes. The inference-specific addition is that an invariant such as “one request, one terminal outcome” must align with KV ownership and a CUDA step boundary.

## 8. Errors, timeouts, and backpressure

An API that streams quickly in the happy path can still collapse under overload. The adapter needs a bounded queue and a clear distinction between “request rejected before engine work” and “request failed after admission.”

| Condition | HTTP behavior | Engine behavior | Client interpretation | Source |
| --- | --- | --- | --- | --- |
| invalid body | 400 | no admission | fix request | derived: validation |
| unknown model | 404 | no admission | choose supported model | derived: model registry |
| queue full | 429 or 503 with `Retry-After` | no KV allocation | retry with backoff | derived: admission policy |
| deadline before first token | 408/504 | cancel at safe point | no complete answer | derived: timeout policy |
| disconnect after admission | no response possible | cancel and release | caller owns retry decision | derived: lifecycle |
| model failure | 500 with request ID | terminal failure and cleanup | retry only if safe | derived: terminal state |

Backpressure is not an optimization. If the HTTP layer accepts unlimited bodies while the engine queue is bounded, memory moves from the GPU queue into Python tasks and proxy buffers. Limit body bytes, input tokens, concurrent streams, and pending requests independently. The right number is deployment-specific; produce it with a load script and an expected range rather than inventing a universal setting.

```python
# nanoserve/admission.py
from asyncio import Semaphore

class Admission:
    def __init__(self, max_pending: int, max_streams: int):
        self.pending = Semaphore(max_pending)
        self.streams = Semaphore(max_streams)

    async def acquire_pending(self):
        if self.pending._value <= 0:
            raise HTTPException(429, "inference queue is full", headers={"Retry-After": "1"})
        await self.pending.acquire()

    def release_pending(self):
        self.pending.release()
```

Do not inspect a private semaphore value for a hard production decision; this minimal snippet makes the idea runnable but a real admission controller should maintain an atomic counter and acquire with a timeout. The important ordering is: validate and tokenize, reserve idempotency, acquire pending capacity, then submit. Release pending when the request leaves the queue, not only when the stream finishes.

## 9. Measuring the API honestly

The API changes what a benchmark measures. A non-streaming client measures completion latency. A streaming client can measure TTFT and TPOT, but only if it records every event boundary and server timestamps. A benchmark that reports only aggregate tok/s cannot tell whether cancellation saved GPU work or whether a proxy buffered all the tokens.

Use a labeled script such as `bench_api.py` with a fixed model ID, prompt manifest, concurrency, arrival process, output cap, client timeout, and server commit. Warm up the model and tokenizer before collecting samples. For GPU timing inside the engine, use `torch.cuda.synchronize()` before starting a CUDA event pair and synchronize after the end event. Never include model load, HTTP connection setup, or tokenizer initialization in decode TPOT unless that is the metric under study.

```python
# bench_api.py — reader-reproducible protocol, not a result claim
import asyncio, json, statistics, time
import httpx

async def one(client, url, body):
    started = time.perf_counter()
    first = None
    tokens = 0
    async with client.stream("POST", url, json={**body, "stream": True}) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            if first is None:
                first = time.perf_counter()
            payload = json.loads(line[6:])
            tokens += sum(bool(c.get("delta", {}).get("content")) for c in payload.get("choices", []))
    ended = time.perf_counter()
    return {"ttft_ms": (first - started) * 1000 if first else None,
            "elapsed_ms": (ended - started) * 1000, "chunks": tokens}

async def main(url, body, concurrency=4):
    async with httpx.AsyncClient(timeout=None) as client:
        results = await asyncio.gather(*(one(client, url, body) for _ in range(concurrency)))
    print(json.dumps({"n": len(results), "ttft_p50_ms": statistics.median(r["ttft_ms"] for r in results), "results": results}, indent=2))

if __name__ == "__main__":
    asyncio.run(main("http://127.0.0.1:8000/v1/chat/completions", {"model": "llama-3.1-8b", "messages": [{"role": "user", "content": "Count to ten."}], "max_tokens": 32}))
```

This script counts non-empty content deltas, not tokens. For usage truth, parse the final usage chunk or query server-side ledger records. The expected range is intentionally not hard-coded: on a given RTX 4090 or A100, run the script across at least 30 steady-state requests and report the range together with driver, CUDA, PyTorch, model revision, prompt length, output cap, and concurrency. That is reproducible evidence; a precise number without that envelope is theater.

The experiment matrix should vary one axis at a time:

| Axis | Values | Why it changes the answer | Source |
| --- | --- | --- | --- |
| prompt | chat, RAG, code, translation | prefill and cache shape differ | reproduce: `bench_api.py` |
| output cap | 32, 256, 1024 tokens | decode work and cancellation opportunity differ | reproduce: `bench_api.py` |
| concurrency | 1, 4, 16, 64 | scheduler batch and queueing change | reproduce: `bench_api.py` |
| stream | false, true | TTFT/TPOT versus completion latency | reproduce: `bench_api.py` |
| disconnect | after first, tenth, random token | tests cancellation propagation | reproduce: `bench_api.py` extension |

For open-loop load, send arrivals according to a Poisson process and record queue delay; for closed-loop load, start the next request after the previous completes and report that choice. The same engine can look healthy under closed-loop load and collapse under open-loop arrivals because the latter allows a queue to form. Use [the series benchmark protocol](/blog/machine-learning/inference-engineering/an-experiment-protocol-for-inference-benchmarks) for the full harness discipline.

## 10. Failure drills before production

The shortest route to confidence is to make the ugly paths deterministic. A protocol test suite should not wait for a real browser or a real GPU to happen to disconnect at the wrong moment. Give the fake engine a clock, a sequence of token events, and a cancellation injection point. Then assert the observable invariant and the ownership side effect.

| Drill | Injected event | Expected invariant | Expected engine side effect | Source |
| --- | --- | --- | --- | --- |
| cancel while queued | cancel before first step | no SSE token | remove from waiting queue | reproduce: CPU fake engine |
| cancel after prefill | cancel after KV allocation | no further delta | release request blocks once | reproduce: CPU fake engine |
| EOS and disconnect race | two tasks call `finish` | one finish reason | one terminal ledger | derived: idempotent `finish` |
| retry before completion | same key twice | one request ID | one engine admission | derived: registry lock |
| retry with changed body | same key, changed prompt | 409 response | no second admission | derived: fingerprint |
| proxy split | split one SSE record at every byte | same JSON event | no engine impact | reproduce: SSE parser test |

The proxy-split drill is especially valuable. Generate a record such as `data: {"delta":"hello"}\n\n`, then feed it to the client parser one byte at a time. The parser should emit one event only after the blank-line separator. Feed a record containing a newline inside JSON string content as escaped JSON and confirm that the serializer produces one `data:` line per encoded line. This catches the common mistake of concatenating arbitrary token text before adding a delimiter.

The cancellation drill should assert allocation ownership, not just a boolean. If request `r1` owns blocks `[7, 8, 9]` and disconnects after token 12, the scheduler should receive one cancellation message and the allocator should see those three blocks returned once. A second `finally` callback must be harmless. A test that only checks HTTP status can pass while the GPU pool leaks.

The idempotency drill should include process boundaries in the design review. An in-memory registry protects one worker process. If a load balancer can route the first attempt to worker A and the retry to worker B, the key store must be shared or the routing must be sticky. Redis or a database introduces its own expiry and serialization semantics; it does not make a streaming replay log free. Start with one process, define the invariant, then choose the smallest distributed store that preserves it.

Finally, test an overloaded queue with a client that never reads the response body. The server must either apply a bounded send buffer and cancel after a write deadline or eventually observe the disconnect. Otherwise the engine can have a healthy decode loop while the network task retains completed chunks and the request registry retains pending state. This is why backpressure, cancellation, and usage finalization are one problem viewed from three layers.

## 11. Case studies / real numbers

### The vLLM paper and the memory contract

The [PagedAttention paper](https://arxiv.org/abs/2309.06180), accessed 2026-08-03, is not an API specification. Its lesson for this post is that a user-visible serving contract rests on a memory manager with explicit ownership. If an API route cannot tell the engine “cancel request X and release its blocks,” the route cannot provide a truthful cancellation promise. We borrow the ownership idea, not a throughput number.

### The vLLM anatomy post and queue ownership

The vLLM team's [Anatomy of a High-Throughput Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm), dated 2025-09-05, names the queues, scheduler, model executor, and KV cache manager as one serving system. That supports the boundary in Figure 7: route handlers should not own those internals. The API can be compatible while the engine remains free to schedule differently.

### OpenAI streaming events

The [OpenAI Responses streaming reference](https://platform.openai.com/docs/api-reference/responses-streaming?lang=python), accessed 2026-08-03, documents typed events, sequence numbers, and usage-bearing response objects for a newer API surface. Chat completions and Responses are not byte-for-byte interchangeable, so nanoserve should advertise the subset it supports. The durable lesson is to make event types and terminal status explicit rather than treating a stream as an untyped text pipe.

### The reader's benchmark

The only numbers this post asks you to obtain locally are from the supplied script. Use a fixed Llama-3.1-8B revision on the named RTX 4090 or A100, warm up, run 30 or more steady-state requests at each matrix cell, and record p50/p95 TTFT, p50/p95 TPOT, completion tokens, cancelled tokens, queue time, and GPU memory. Expected ranges are hardware- and software-dependent; report the observed interval and commit hash. Do not label it a nanoserve result until the script has emitted a manifest.

## When to reach for this (and when not to)

Build this adapter when you need a stable client contract over an engine you are actively writing, when cancellation changes GPU economics, or when you need to test protocol behavior without a GPU. The API is also worth separating when several frontends share one scheduler: chat, batch jobs, and an internal evaluation runner can all feed the same `EngineRequest`.

Do not write an OpenAI-compatible server just to avoid choosing an existing one. If the requirement is production traffic, broad model coverage, multi-GPU execution, mature prefix caching, or hardened observability, use vLLM, SGLang, TGI, or a managed provider and put your product-specific policy in a gateway. The [vLLM deep dive](/blog/machine-learning/model-serving/vllm-deep-dive) is a better benchmark target than a homegrown route.

The compatibility promise should be narrow and honest. “Works with this SDK” means request validation, response fields, streaming delimiters, errors, and usage semantics have tests. It does not mean every proprietary parameter, tool event, multimodal input, or background job is silently accepted. Unsupported fields should fail with a useful 400 response.

## Key takeaways

- Normalize the public request into a small immutable engine request before admission.
- Treat lifecycle transitions, not booleans, as the source of truth for cleanup and metrics.
- SSE is framing: complete `data:` records, blank-line delimiters, and an explicit terminal event.
- A client disconnect must become an engine cancellation signal at the next safe point.
- The route must never mutate GPU state directly; the engine owns scheduling, KV blocks, and device work.
- Count tokens at tokenizer and decoder boundaries, then finalize usage exactly once.
- Idempotency keys identify logical operations; request IDs identify server attempts.
- Replay completed non-streaming results first; do not claim resumable streaming without an event log.
- Bound pending requests, streams, bodies, and tokens independently.
- Benchmark TTFT, TPOT, queue delay, cancellation, and goodput; aggregate tok/s is not a service contract.

## Further reading

- [OpenAI Responses streaming events](https://platform.openai.com/docs/api-reference/responses-streaming?lang=python), accessed 2026-08-03.
- [MDN: Using server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events), accessed 2026-08-03.
- [PagedAttention paper](https://arxiv.org/abs/2309.06180), accessed 2026-08-03.
- [Inside vLLM: Anatomy of a High-Throughput Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm), 2025-09-05.
- [An experiment protocol for inference benchmarks](/blog/machine-learning/inference-engineering/an-experiment-protocol-for-inference-benchmarks).
- [The inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook).

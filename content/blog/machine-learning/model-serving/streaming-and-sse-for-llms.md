---
title: "Streaming and SSE for LLMs: token delivery from generate loop to browser"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Master the full stack of LLM token streaming — SSE protocol mechanics, FastAPI StreamingResponse, vLLM async engine, backpressure, token buffering, and stateful reconnect."
tags:
  [
    "model-serving",
    "inference",
    "streaming",
    "sse",
    "llm-serving",
    "fastapi",
    "vllm",
    "backpressure",
    "token-streaming",
    "latency",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/streaming-and-sse-for-llms-1.png"
---

You have just wired your Llama 3 deployment to a chat interface. You hit Send. The loading spinner rotates for eleven seconds, then a thousand words appear all at once. You show it to a product manager. They say "it feels broken." You say "the model is fast, look at the TPOT." They say "I don't care what TPOT is — it felt frozen."

They are right. Not because the model is slow, but because you forgot to implement streaming. A 500-token response at 40 tokens per second takes 12.5 seconds to generate. Without streaming, the user waits all 12.5 seconds in silence. With streaming, the first token appears in roughly 200 ms, and subsequent words flow at reading speed. The model generates at identical speed in both cases. Only the delivery changes — and the perceived responsiveness difference is decisive for every user-facing LLM product in existence.

This post covers the entire stack: why streaming matters quantitatively, the Server-Sent Events (SSE) protocol and why it is the right choice for LLM tokens, the OpenAI chunk format that every client expects, a production FastAPI implementation with proper disconnect handling, vLLM's async engine integration, backpressure and TCP flow control, token buffering for smooth rendering, stateful reconnect semantics, multi-turn conversation state management, gRPC streaming as an alternative for internal services, and real benchmarks comparing SSE and gRPC at 10,000 concurrent streams.

Figure 1 shows the complete streaming request lifecycle — from the HTTP POST arriving at your gateway, through prefill and the first token appearing as an SSE event, through the decode loop flushing tokens, to the `[DONE]` sentinel and client-side rendering.

![Streaming request lifecycle from prefill to rendered token](/imgs/blogs/streaming-and-sse-for-llms-1.png)

By the end of this post you will be able to build a production-grade LLM streaming endpoint, handle every failure mode from client disconnects to backpressure stalls, and make the architecture decision between SSE and gRPC for your use case. This post is part of the [Model Deployment and Serving series](/blog/machine-learning/model-serving/what-is-model-serving). It builds on the [continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention) post (which explains why vLLM's scheduler generates tokens the way it does) and the [vLLM deep dive](/blog/machine-learning/model-serving/vllm-deep-dive) (which covers prefix caching, chunked prefill, and the async engine internals we call here). If you are evaluating whether to stream at all for your use case, start with [model serving SLAs and metrics](/blog/machine-learning/model-serving/model-serving-slas-and-metrics), which defines TTFT and TPOT in the context of your SLO design.



## 1. Why streaming changes everything about LLM UX

The math is simple and unforgiving. Define:

- $L$ = total response length in tokens
- $R$ = generation rate in tokens per second (TPOT-derived)
- $T_{prefill}$ = time for the prefill pass in milliseconds

Without streaming, the user waits:

$$T_{wait} = T_{prefill} + \frac{L}{R}$$

At $T_{prefill} = 300$ ms, $L = 500$ tokens, $R = 40$ tok/s:

$$T_{wait} = 0.3 + \frac{500}{40} = 0.3 + 12.5 = 12.8 \text{ s}$$

With streaming, the user sees the first token after:

$$T_{TTFT} = T_{prefill} \approx 200\text{–}400 \text{ ms}$$

Subsequent tokens arrive every $\frac{1}{R} = 25$ ms. A skilled adult reader reads at roughly 200–250 words per minute, which is 3–4 words per second, or about 4–5 tokens per second (accounting for subword tokens). At 40 tok/s generation rate, the model is generating 8–10× faster than the user reads. The user can never tell whether generation has "finished" — they are always reading in the wake of the generation cursor. The perceived experience is of a fast, responsive system.

Human reaction-time research consistently shows that systems responding within 200–300 ms feel "instantaneous," 300–1000 ms feel "fast," and anything over 1 second feels "delayed." At 12.8 seconds, the non-streaming experience crosses the threshold where users assume something is broken and start hitting refresh. The 100 ms TTFT target is aggressive but achievable for smaller models; 300 ms is the upper limit of what feels interactive.

![Non-streaming vs streaming UX at 40 tokens per second](/imgs/blogs/streaming-and-sse-for-llms-2.png)

Streaming does not improve throughput, TPOT, or GPU utilization by even one percent. It is a pure delivery optimization. The same autoregressive decode loop runs on the GPU at identical speed. The difference is when the TCP `write()` calls happen: all at once at the end, versus once per token throughout generation. The cost of streaming is about 0.1 ms of additional overhead per token from more frequent kernel `write()` calls — entirely negligible against the UX benefit.

### 1.1 The anatomy of a streaming request

Understanding the timing structure of a streaming request is essential before implementing it. A streaming LLM request has three distinct phases:

**Phase 1: Prefill (compute-bound).** The model processes the entire input prompt in a single forward pass. All prompt tokens are processed in parallel using the attention mechanism. This phase is compute-intensive: it reads model weights from HBM, runs matrix multiplications for all attention heads and FFN layers, and writes the KV cache entries for all prompt tokens. Prefill time scales roughly linearly with prompt length and model size. On H100 with Llama-3-8B-Instruct, prefill processes about 5,500 tokens/second, so a 512-token prompt takes approximately 93 ms.

**Phase 2: Decode (memory-bandwidth-bound).** The model generates one token per forward pass. Each forward pass reads the full KV cache (growing with each step) plus all model weights. Decode is memory-bandwidth-bound, not compute-bound: the GPU arithmetic is trivial, but reading 8–80 GB of weights from HBM takes 30–100 ms per step depending on hardware. This is TPOT — time per output token. The first SSE event with actual content is emitted after Phase 1 completes; subsequent events fire at the decode rate.

**Phase 3: Serialization and transmission.** The generated token ID is decoded to text (subword-to-string via the tokenizer vocabulary), wrapped in the OpenAI chunk JSON, framed as an SSE event, and written to the TCP socket. This takes approximately 0.1–0.5 ms per token — negligible against decode time.

The most important optimization opportunity is in Phase 1 (reduce prompt length, enable prefix caching) and Phase 2 (quantization, speculative decoding, larger batch sizes). Phase 3 is not a bottleneck for single-user interactive serving. In practice, the bottleneck for the user experience is always Phase 1 at low load (TTFT dominated by prefill) and vLLM's scheduler queue at high load (TTFT dominated by wait time behind other requests).

### 1.2 Relationship to the serving SLO triangle

In the context of the [model serving SLO triangle](/blog/machine-learning/model-serving/what-is-model-serving) (latency ↔ throughput ↔ cost), streaming is unusual: it improves perceived latency (TTFT at the user) without touching actual latency (generation time), throughput (tokens/second), or cost (GPU hours). It is the one serving technique that moves the perceived latency axis without any trade-off on the other two. Every LLM API that serves humans rather than batch pipelines should implement streaming as a first-class priority.

For completeness, the SLO metrics for streaming serving are:
- **TTFT** (Time to First Token): how long before the user sees anything — target <300 ms for interactive chat.
- **TPOT** (Time Per Output Token): how long between successive tokens — target <50 ms for smooth display (equivalent to >20 tok/s).
- **E2E latency**: total time from request to `[DONE]` — relevant for programmatic consumers.
- **Streaming abort rate**: fraction of requests where the client disconnects before `[DONE]` — above 5% indicates TTFT SLO violations (users are giving up).



## 2. Server-Sent Events: the right protocol for token delivery

LLM token streaming has a specific shape: one sender (the server), many small messages (tokens), delivered in order, over a connection that may last many seconds. Three protocols can do this: Server-Sent Events (SSE), WebSockets, and gRPC server streaming.

SSE is the right choice for browser-facing LLM APIs. Here is why.

### 2.1 The SSE protocol

SSE is defined in the HTML5 specification and rides on a plain HTTP/1.1 persistent connection. The server responds with:

```
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
```

Events are text frames, each ending with a blank line (`\n\n`). Each frame has one or more fields:

```
id: 42
event: token
data: {"content": "Hello"}

data: {"content": " world"}

```

The `id:` field sets the `Last-Event-ID` — the browser stores this and sends it back as a header on reconnect. The `event:` field is optional (defaults to `message`). The `data:` field is the payload. A single blank line terminates the event; two consecutive blank lines have no special meaning beyond two events.

For LLM token streaming, convention (following OpenAI's API) is to use only `data:` fields, no `event:` field, and a final sentinel event:

```
data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n
data: {"choices":[{"delta":{"content":" world"}}]}\n\n
data: [DONE]\n\n
```

The `[DONE]` sentinel is not JSON — the client checks for it literally before attempting to parse.

### 2.2 Why SSE over WebSocket

WebSocket requires an upgrade handshake, runs its own framing protocol over TCP, and is bidirectional. For LLM streaming, you do not need bidirectionality after the request is submitted. The user sends one message, the server responds with a stream of tokens. WebSocket's bidirectional capability is unnecessary, and its lack of built-in reconnect semantics means you must implement that yourself.

SSE auto-reconnects out of the box. If the connection drops, the browser's `EventSource` automatically reopens the connection within a few seconds and sends the `Last-Event-ID` header with the last received event ID. This is exactly what you want for a long-running LLM generation that may survive a brief network hiccup.

SSE also traverses HTTP proxies, load balancers, and CDNs without special configuration, because it looks like a normal HTTP response that happens to never end. WebSocket requires your entire proxy chain to understand the Upgrade protocol — a frequent source of infrastructure problems.

### 2.3 The SSE protocol stack in your serving system

![SSE protocol stack from vLLM generate loop to browser EventSource](/imgs/blogs/streaming-and-sse-for-llms-3.png)

The stack is thin: vLLM's `AsyncLLMEngine` generates token deltas in an async generator, FastAPI's `StreamingResponse` wraps that generator, the SSE framing layer prepends `data: ` and appends `\n\n`, the TCP layer delivers bytes to the client, and the browser's native `EventSource` API fires `onmessage` callbacks with the parsed data.

### 2.4 Protocol comparison: SSE vs WebSocket vs gRPC

Before committing to SSE, it is worth understanding the decision quantitatively. The three protocols differ on five axes that matter for LLM serving.

![SSE vs WebSocket vs gRPC streaming protocol comparison matrix](/imgs/blogs/streaming-and-sse-for-llms-4.png)

**Direction of data flow.** LLM token streaming is inherently unidirectional — the server sends tokens, the client only receives. SSE is server-to-client by design. WebSocket is bidirectional by design, which means you are paying for a capability you do not use. The only scenario where WebSocket's bidirectionality matters for LLM serving is if you want to implement mid-stream interruption signals from client to server (e.g., "stop generating"). In practice, HTTP DELETE or a separate POST `/abort` endpoint handles this more cleanly.

**Browser native support.** SSE's `EventSource` is natively supported in every modern browser since 2012. WebSocket is also natively supported. gRPC requires `grpc-web`, a JavaScript library that translates between the browser and a gRPC-web proxy — typically Envoy. This adds an operational dependency that can be eliminated by using SSE.

**Auto-reconnect.** The SSE specification mandates that the browser retries a dropped connection automatically after the retry interval (default 3 seconds, configurable via `retry: 5000\n` in SSE events). WebSocket has no such specification — you must implement your own reconnect loop. gRPC similarly provides no automatic reconnect.

**Per-token overhead.** SSE frames a token as `data: {json}\n\n`. A typical token payload is 30–60 bytes of JSON. The `data: ` prefix and `\n\n` suffix add 8 bytes. Net overhead: ~12–20%. gRPC uses Protocol Buffer encoding — the same token fits in 10–20 bytes, and HTTP/2 header compression (HPACK) amortizes header costs across the stream. At 40 tok/s, the bandwidth difference is small in absolute terms (perhaps 1–2 KB/s), but at 10,000 concurrent streams it adds up to measurable memory pressure in kernel socket buffers.

**Proxy and CDN transparency.** SSE is a plain HTTP/1.1 response. Every proxy, CDN, and load balancer that handles HTTP handles SSE — modulo buffer flushing (see `X-Accel-Buffering` below). WebSocket requires proxies to understand the `Upgrade: websocket` header and maintain stateful connections. gRPC requires HTTP/2 end-to-end and does not traverse HTTP/1.1-only proxies at all.

The decision rule is simple: use SSE for browser-facing APIs, use gRPC for internal services. If you have a mixed fleet — browser clients plus internal consumers — expose SSE externally and use gRPC internally, with a translation layer at the gateway.

### 2.5 SSE retry and heartbeat

Two housekeeping details that are easy to miss in production.

**The retry field.** SSE supports a `retry: <ms>` field that tells the browser how long to wait before reconnecting after a dropped connection. The default is browser-determined (typically 3000 ms). For LLM streaming you may want to set this lower (1000 ms) if your connections drop frequently, or higher (10000 ms) if your backend is under load and you do not want reconnect storms:

```
retry: 3000
data: {"choices":[{"delta":{"content":"Hello"}}]}

```

**Heartbeat events.** Some HTTP proxies (notably AWS ALB and Cloudflare) close idle HTTP connections after a configurable timeout (typically 60–300 seconds). If generation pauses during a very long response, the proxy may close the connection before the next token arrives. Send a heartbeat comment every 15–30 seconds to keep the connection alive:

```python
async def streaming_generator_with_heartbeat(
    prompt: str, sampling_params: SamplingParams, request_id: str
) -> AsyncGenerator[str, None]:
    last_token_time = asyncio.get_event_loop().time()
    HEARTBEAT_INTERVAL = 15.0  # seconds

    async for output in engine.generate(prompt, sampling_params, request_id):
        current_text = output.outputs[0].text
        # ... compute and yield delta as before ...

        # Yield SSE comment as heartbeat if too much time has passed.
        now = asyncio.get_event_loop().time()
        if now - last_token_time > HEARTBEAT_INTERVAL:
            yield ": heartbeat\n\n"  # SSE comment — clients ignore it
            last_token_time = now

    yield "data: [DONE]\n\n"
```

SSE comment lines start with `:` and are ignored by the `EventSource` API. They traverse proxies and reset idle connection timers without affecting the token stream.



## 3. The OpenAI streaming chunk format

If you are building a streaming LLM API in 2026, you should speak the OpenAI streaming format. Every major client library, proxy, and gateway speaks it. The format is straightforward.

Each SSE data payload is a JSON object with this shape:

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion.chunk",
  "created": 1719056000,
  "model": "meta-llama/Llama-3-8b-instruct",
  "choices": [
    {
      "index": 0,
      "delta": {
        "role": "assistant",
        "content": "Hello"
      },
      "finish_reason": null
    }
  ]
}
```

The `delta` field carries what is new in this chunk. The first chunk typically has `"role": "assistant"` and empty `content`. Subsequent chunks have `content` and no `role`. The final chunk has `"finish_reason": "stop"` (or `"length"`, `"content_filter"`, etc.) and empty `content`. After the last JSON chunk, the server sends the literal sentinel `data: [DONE]\n\n`.

On the client side, the parsing logic is:

1. Receive SSE event data string.
2. If the string equals `[DONE]`, close the stream.
3. Otherwise, parse as JSON and extract `choices[0].delta.content`.
4. Append to the accumulated response buffer.

One subtlety: the first chunk's delta contains `"role": "assistant"` but empty or null content. Your client must handle null content without crashing. A second subtlety: `finish_reason` can be non-null while `content` is still non-empty in some implementations. Parse defensively.

### 3.1 Streaming tool calls and function results

The OpenAI streaming format extends to tool calls. When the model calls a function, the tool call arguments stream incrementally in the delta. Each chunk has a `tool_calls` array instead of (or alongside) `content`:

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion.chunk",
  "choices": [
    {
      "index": 0,
      "delta": {
        "tool_calls": [
          {
            "index": 0,
            "id": "call_xyz",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"location\": \"San"
            }
          }
        ]
      },
      "finish_reason": null
    }
  ]
}
```

The function arguments stream as partial JSON — in this example `{"location": "San` followed by `Francisco\"}` in the next chunk. The client must accumulate the `arguments` string across chunks and parse it only after `finish_reason` becomes `"tool_calls"`. This is a common source of bugs: parsing incomplete JSON arguments mid-stream crashes the client.

### 3.2 Streaming with `usage` information

Some deployments (notably for cost accounting) stream per-token usage in the final chunk. OpenAI added this behind the `stream_options: {"include_usage": true}` parameter:

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion.chunk",
  "choices": [],
  "usage": {
    "prompt_tokens": 512,
    "completion_tokens": 348,
    "total_tokens": 860
  }
}
```

This appears as an extra chunk after the `[DONE]` marker in some implementations, or as the final chunk before `[DONE]` in others. Implement defensively: check for `"usage"` in the chunk regardless of position, and handle the case where `choices` is an empty array.

### 3.3 The complete stream sequence

To be concrete about the full wire format for a simple "Hello, how can I help you today?" response:

```
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
X-Accel-Buffering: no

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":","},"finish_reason":null}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" how"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"?"},"finish_reason":"stop"}]}

data: [DONE]

```

Each `\n\n` terminator is a blank line in the response body. The `id` field is the same across all chunks (it identifies the generation request, not individual events). Note the two-space indent convention is just for readability — in production each `data:` line contains the full JSON with no extra whitespace.



## 4. FastAPI SSE implementation

FastAPI's `StreamingResponse` with `media_type="text/event-stream"` is the standard way to implement SSE. The key insight is that FastAPI accepts an async generator as the response body — it will call `__anext__()` on the generator and send each yielded value as a chunk of the HTTP response body.

```python
import asyncio
import json
import uuid
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()


async def token_stream_generator(
    prompt: str,
    request: Request,
) -> AsyncGenerator[str, None]:
    """Async generator yielding SSE-formatted token chunks."""
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    # Simulate vLLM-style token generation.
    # In production, replace with AsyncLLMEngine.generate().
    tokens = ["Hello", ",", " how", " can", " I", " help", " you", " today", "?"]

    # First chunk: role only
    first_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first_chunk)}\n\n"

    try:
        for i, token in enumerate(tokens):
            # Check if client disconnected before sending each token.
            if await request.is_disconnected():
                break

            is_last = i == len(tokens) - 1
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": token},
                        "finish_reason": "stop" if is_last else None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

            # Simulate 25 ms TPOT (40 tok/s).
            await asyncio.sleep(0.025)

    except asyncio.CancelledError:
        # Client disconnected mid-stream. Clean up server-side state here.
        # With a real vLLM engine, call engine.abort(request_id).
        return

    # Final sentinel — tells clients the stream is finished.
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    prompt = messages[-1]["content"] if messages else ""
    stream = body.get("stream", False)

    if not stream:
        # Non-streaming path — omitted for brevity.
        return {"error": "set stream=true for this demo"}

    return StreamingResponse(
        token_stream_generator(prompt, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx proxy buffering.
            "Connection": "keep-alive",
        },
    )
```

Two things deserve attention in this implementation.

**The `X-Accel-Buffering: no` header** is critical when nginx sits in front of your FastAPI server. Nginx buffers proxy responses by default. Without this header, nginx will buffer your SSE stream until its buffer fills (default 8 KB or the response completes), completely defeating streaming. The header `X-Accel-Buffering: no` tells nginx to pass each `write()` through immediately.

**The disconnect check** at `await request.is_disconnected()` prevents zombie generation. Without it, vLLM will continue generating tokens for a client that already closed the connection, consuming GPU capacity and KV cache memory. For a real vLLM deployment, you also call `engine.abort(request_id)` to free the KV cache slot when a client disconnects.

### 4.1 Production hardening for the FastAPI SSE endpoint

A production streaming endpoint needs several additional layers beyond the basic pattern:

**Request timeout.** Long generations can stall if the model hits a degenerate sampling path or the GPU memory swaps. Wrap the generator in a timeout:

```python
import asyncio

async def timeout_generator(
    inner_gen: AsyncGenerator[str, None], timeout_seconds: float = 120.0
) -> AsyncGenerator[str, None]:
    """Wrap any async generator with a per-chunk timeout."""
    async for chunk in inner_gen:
        yield chunk
        # Re-arm the timeout on each successful chunk.
    # If the inner generator stalls, asyncio.wait_for raises TimeoutError,
    # which FastAPI converts to a 504 response.
```

A cleaner approach is to use `asyncio.wait_for()` at the route level:

```python
@app.post("/v1/chat/completions")
async def chat_completions(raw_request: Request):
    # ... setup code ...
    async def gen():
        async for chunk in vllm_stream_generator(prompt, params, req_id, raw_request):
            yield chunk

    try:
        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
```

**Concurrent request limiting.** A single vLLM engine instance handles concurrency internally through its scheduler, but the FastAPI layer should enforce a per-user or global concurrency limit to prevent request queue buildup:

```python
from asyncio import Semaphore

MAX_CONCURRENT_STREAMS = 200
semaphore = Semaphore(MAX_CONCURRENT_STREAMS)

@app.post("/v1/chat/completions")
async def chat_completions(raw_request: Request):
    async with semaphore:
        # Process the streaming request.
        return StreamingResponse(...)
```

When all slots are taken, the next request waits for one to free up. Combined with a request timeout, this prevents unbounded queue growth under load spikes.

**Response compression.** SSE text is highly compressible (repetitive JSON structure), but you should NOT use HTTP compression (gzip/deflate) for streaming responses. Compression requires buffering data before sending, which defeats the purpose. Disable compression at the ASGI middleware level for SSE responses by checking `Content-Type: text/event-stream`.

### 4.2 Uvicorn and Gunicorn configuration for streaming

The ASGI server configuration matters for streaming performance. With uvicorn:

```bash
uvicorn main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1 \
  --loop uvloop \
  --http h11 \
  --timeout-keep-alive 300 \
  --no-access-log
```

Key settings:
- `--workers 1` — with vLLM, use a single worker process. vLLM manages GPU memory globally; multiple workers fight over the GPU.
- `--loop uvloop` — uvloop is 2–4× faster than asyncio's default event loop for I/O-bound workloads like streaming.
- `--timeout-keep-alive 300` — keep persistent connections alive for 5 minutes to support long generations without reconnect overhead.
- `--http h11` — h11 (HTTP/1.1 only) is simpler and more predictable for SSE than h2c (HTTP/2 cleartext).

If you need multiple workers (for CPU-bound preprocessing), use Gunicorn with uvicorn worker class:

```bash
gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --timeout 300 \
  --keep-alive 300 \
  --bind 0.0.0.0:8000
```

With multiple workers, each worker holds its own vLLM engine. This is only appropriate if each worker is assigned its own GPU via `CUDA_VISIBLE_DEVICES`.



## 5. vLLM async engine streaming

vLLM's `AsyncLLMEngine` is purpose-built for concurrent, streaming LLM inference. It manages a request queue internally, runs continuous batching across all queued requests, and exposes each request's output as an async generator.

```python
import asyncio
import json
import uuid
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

app = FastAPI()

# Initialize the async engine once at startup.
engine_args = AsyncEngineArgs(
    model="meta-llama/Llama-3-8b-Instruct",
    dtype="bfloat16",
    max_model_len=8192,
    gpu_memory_utilization=0.90,
    tensor_parallel_size=1,
)
engine = AsyncLLMEngine.from_engine_args(engine_args)


async def vllm_stream_generator(
    prompt: str,
    sampling_params: SamplingParams,
    request_id: str,
    raw_request: Request,
) -> AsyncGenerator[str, None]:
    """Wrap vLLM's async generator in OpenAI-compatible SSE chunks."""
    results_generator = engine.generate(prompt, sampling_params, request_id)

    previous_text = ""
    try:
        async for request_output in results_generator:
            # vLLM yields cumulative text; compute the delta ourselves.
            current_text = request_output.outputs[0].text
            new_text = current_text[len(previous_text):]
            previous_text = current_text

            if await raw_request.is_disconnected():
                # Abort the generation to free KV cache.
                await engine.abort(request_id)
                return

            finish_reason = request_output.outputs[0].finish_reason

            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": new_text},
                        "finish_reason": finish_reason,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

    except asyncio.CancelledError:
        await engine.abort(request_id)
        return

    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(raw_request: Request):
    body = await raw_request.json()
    messages = body.get("messages", [])
    # In production, apply the model's chat template here.
    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    sampling_params = SamplingParams(
        temperature=body.get("temperature", 0.7),
        max_tokens=body.get("max_tokens", 1024),
        top_p=body.get("top_p", 1.0),
    )
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    return StreamingResponse(
        vllm_stream_generator(prompt, sampling_params, request_id, raw_request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
```

One subtlety with vLLM's `AsyncLLMEngine`: it yields `RequestOutput` objects with the **full cumulative text** in `outputs[0].text`, not just the new delta. You must compute the delta yourself by tracking the previous text length. The line `new_text = current_text[len(previous_text):]` does this correctly.

Note also that vLLM already handles the continuous batching internally. When you call `engine.generate()` for multiple concurrent requests, vLLM batches them together in its scheduler. The async generators for all concurrent requests are driven by a single event loop — each `async for` iteration yields when vLLM's scheduler produces output for that request. You do not need to manage batching at the FastAPI layer.

#### Worked example: computing real TTFT and TPOT from vLLM streaming

Suppose you want to instrument the actual TTFT and TPOT from within the streaming generator:

```python
import time

async def instrumented_stream_generator(
    prompt: str, sampling_params: SamplingParams, request_id: str, raw_request: Request
) -> AsyncGenerator[str, None]:
    results_generator = engine.generate(prompt, sampling_params, request_id)

    request_start = time.perf_counter()
    first_token_time = None
    token_count = 0
    previous_text = ""

    async for request_output in results_generator:
        current_text = request_output.outputs[0].text
        new_tokens = current_text[len(previous_text):]
        previous_text = current_text

        if new_tokens:
            now = time.perf_counter()
            if first_token_time is None:
                first_token_time = now
                ttft_ms = (first_token_time - request_start) * 1000
                # Record TTFT to Prometheus or OpenTelemetry here.
                print(f"TTFT: {ttft_ms:.1f} ms")
            token_count += len(new_tokens.split())  # rough approximation

        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": new_tokens}, "finish_reason": request_output.outputs[0].finish_reason}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    if first_token_time:
        total_time = time.perf_counter() - request_start
        tpot_ms = (total_time - (first_token_time - request_start)) / max(token_count, 1) * 1000
        print(f"TPOT: {tpot_ms:.1f} ms/token, total tokens: {token_count}")

    yield "data: [DONE]\n\n"
```

On an H100 80GB SXM5 serving Llama-3-8B-Instruct at batch size 1, expect TTFT around 80–150 ms for a 512-token prompt and TPOT around 20–28 ms per token (36–50 tok/s). Batch size 16 will increase TTFT to 200–400 ms but push throughput to 300–500 tok/s aggregate. These numbers come from vLLM's own benchmarks and the H100 inference characterization in the vLLM paper (Kwon et al., 2023).



## 6. Backpressure: what happens when the client can't keep up

Backpressure is what happens when a producer generates data faster than a consumer can absorb it. In LLM streaming, the producer is vLLM's decode loop at 40–200 tok/s; the consumer is the client's network buffer. At 40 tok/s with ~50-byte SSE frames per token, the server generates about 2 KB/s of SSE data — trivial for any modern connection. But at very high batch sizes or on slow mobile connections, backpressure becomes real.

![Backpressure flow through TCP from vLLM generator to slow client](/imgs/blogs/streaming-and-sse-for-llms-5.png)

SSE over HTTP/1.1 gets backpressure for free from TCP. TCP's flow control mechanism uses a receive window (`RWND`) advertised by the receiver. When the client's receive buffer fills, it sets `RWND=0` in its ACK packets. The kernel on the server side sees `RWND=0` and blocks the next `write()` or `send()` system call. Because FastAPI's `StreamingResponse` is driven by an `async for` loop in asyncio, the blocked `write()` translates to a suspended coroutine — the async generator pauses at the `yield`. vLLM's decode loop, which runs in a separate thread, continues generating tokens, but they queue up in vLLM's output buffer rather than being transmitted.

The key protection you need to add is a maximum output queue size in your streaming generator. Without it, a sufficiently slow client combined with sufficiently fast generation can cause vLLM's output buffer to grow unboundedly, eventually consuming all available CPU memory:

```python
from asyncio import Queue

MAX_QUEUE_SIZE = 100  # Maximum buffered tokens before we start blocking.

async def backpressure_safe_generator(
    prompt: str, sampling_params: SamplingParams, request_id: str
) -> AsyncGenerator[str, None]:
    queue: Queue = Queue(maxsize=MAX_QUEUE_SIZE)

    async def producer():
        async for output in engine.generate(prompt, sampling_params, request_id):
            delta = output.outputs[0].text  # simplified; compute real delta
            await queue.put(delta)  # blocks if queue is full
        await queue.put(None)  # sentinel

    producer_task = asyncio.create_task(producer())

    try:
        while True:
            token = await queue.get()
            if token is None:
                break
            chunk = {"choices": [{"delta": {"content": token}, "finish_reason": None}]}
            yield f"data: {json.dumps(chunk)}\n\n"
    finally:
        producer_task.cancel()
        yield "data: [DONE]\n\n"
```

With HTTP/2 multiplexing (e.g., if you put h2 in front of FastAPI), flow control is per-stream rather than per-connection. This is strictly better for high-concurrency scenarios — a slow client on one stream does not block TCP window for other streams on the same connection. HTTP/2's `WINDOW_UPDATE` frames provide per-stream backpressure with the same "generator pauses" semantics.

### 6.1 The backpressure math

How much buffering do you actually need? The backpressure scenario requires:

1. vLLM generating at $R_{gen}$ tokens/second
2. Client draining at $R_{drain}$ tokens/second
3. Generation continuing for $T_{gen}$ seconds

Maximum buffer size needed:

$$N_{buf} = (R_{gen} - R_{drain}) \times T_{gen}$$

At $R_{gen} = 40$ tok/s, $R_{drain} = 5$ tok/s (very slow mobile), $T_{gen} = 30$ s (long generation):

$$N_{buf} = (40 - 5) \times 30 = 1{,}050 \text{ tokens}$$

Each token in the queue takes roughly 50–100 bytes of SSE frame data plus Python object overhead — call it 256 bytes per queue slot. For 1,050 slots: about 256 KB per slow request. At 1,000 concurrent slow requests, that is 256 MB of queue memory. Set `MAX_QUEUE_SIZE = 200` as a reasonable upper bound — slow clients that fill the queue will see the generator block, which is correct: the server should not generate into a void.

### 6.2 Zombie generation and KV cache pressure

The second class of backpressure problem is subtler: a client that disconnects mid-stream without the server detecting it. TCP connection teardown is not always immediate — a client behind a NAT that simply stops responding may not send a FIN. The server's TCP stack will eventually timeout (default TCP keepalive is 2 hours on Linux), but by that point vLLM may have completed the entire generation into a buffer nobody is reading.

The defense: poll `request.is_disconnected()` every N tokens, and abort immediately:

```python
DISCONNECT_CHECK_INTERVAL = 10  # Check every 10 tokens

async def disconnect_aware_generator(
    prompt: str, sampling_params: SamplingParams, request_id: str, raw_request: Request
) -> AsyncGenerator[str, None]:
    token_count = 0
    previous_text = ""

    async for output in engine.generate(prompt, sampling_params, request_id):
        current = output.outputs[0].text
        delta = current[len(previous_text):]
        previous_text = current

        token_count += 1
        if token_count % DISCONNECT_CHECK_INTERVAL == 0:
            if await raw_request.is_disconnected():
                await engine.abort(request_id)
                return

        chunk = {"choices": [{"delta": {"content": delta}, "finish_reason": output.outputs[0].finish_reason}]}
        yield f"data: {json.dumps(chunk)}\n\n"

    yield "data: [DONE]\n\n"
```

On a busy server with 100 concurrent requests, each with a 500-token response and a 10-token disconnect check interval, this adds at most 50 `is_disconnected()` calls per request — a negligible `read()` on the connection's file descriptor.

You can also set a shorter TCP keepalive at the socket level to detect dead connections faster:

```python
import socket

# In uvicorn or your ASGI server startup code:
def configure_socket(sock: socket.socket):
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 30)   # Start keepalive after 30s idle
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)  # Send probe every 10s
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)     # 3 failed probes = dead
```

With these settings, a dead connection is detected within $30 + 3 \times 10 = 60$ seconds rather than 2 hours, freeing the KV cache slot and GPU capacity.



## 7. Token buffering for smooth rendering

Even with TCP backpressure handled, raw token-by-token streaming can feel jittery. GPU decode latency is not perfectly uniform: the first few tokens after a prefill may arrive quickly as the KV cache warms up, then slow slightly as memory bandwidth saturates, then speed up again as long-range attention patterns thin out. At 40 tok/s you get a token every 25 ms on average, but individual token timings vary by ±30–50%. The client sees microbursts followed by brief pauses — a janky scrolling experience.

The fix is a token buffer that decouples generation timing from delivery timing. The server enqueues tokens as fast as they arrive from vLLM, then drains the queue at a fixed rate matched to comfortable reading speed.

```python
import asyncio
from collections import deque

DRAIN_RATE_TOKENS_PER_SEC = 30
DRAIN_INTERVAL_SEC = 1.0 / DRAIN_RATE_TOKENS_PER_SEC  # ~33 ms


async def smoothed_stream_generator(
    prompt: str, sampling_params: SamplingParams, request_id: str, raw_request: Request
) -> AsyncGenerator[str, None]:
    """Buffer tokens from vLLM and drain at a fixed 30 tok/s rate."""
    buffer: deque[str | None] = deque()
    generation_done = asyncio.Event()

    async def fill_buffer():
        previous = ""
        async for output in engine.generate(prompt, sampling_params, request_id):
            current = output.outputs[0].text
            delta = current[len(previous):]
            previous = current
            if delta:
                buffer.append(delta)
        buffer.append(None)  # sentinel
        generation_done.set()

    fill_task = asyncio.create_task(fill_buffer())

    try:
        while True:
            if await raw_request.is_disconnected():
                fill_task.cancel()
                await engine.abort(request_id)
                return

            if buffer:
                token = buffer.popleft()
                if token is None:
                    break
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(DRAIN_INTERVAL_SEC)
            else:
                # Buffer empty but generation not done — yield to event loop.
                await asyncio.sleep(0.005)

    except asyncio.CancelledError:
        fill_task.cancel()
        return

    yield "data: [DONE]\n\n"
```

![Token buffer deque draining at fixed 30 tokens per second](/imgs/blogs/streaming-and-sse-for-llms-6.png)

The tradeoff: this approach slightly increases end-to-end latency for short responses (a 10-token response that generates in 250 ms takes 333 ms to drain at 30 tok/s). For long responses (500+ tokens), the generation time dominates and the smoothing is invisible. Configure `DRAIN_RATE_TOKENS_PER_SEC` based on your typical user's reading speed and response length distribution. For a code assistant where users scan output quickly, 40–50 tok/s looks better. For a creative writing app where users read at natural speed, 20–25 tok/s can feel more natural.

**When not to smooth**: if your p99 TPOT is already at or below 25 ms (40+ tok/s), smoothing adds complexity with minimal benefit. Only add it if users are reporting jitter complaints and your generation rate is genuinely variable.



## 8. Reconnect semantics and stateful SSE

SSE's most underappreciated feature is its built-in reconnect protocol. When a browser's `EventSource` connection drops (network glitch, laptop sleep/wake, proxy timeout), the browser automatically retries the connection. The retry includes the `Last-Event-ID` header set to the `id:` field of the last successfully received event.

To implement stateful reconnect — delivering tokens missed during a disconnect — you need to:

1. Assign sequential event IDs to each SSE event.
2. Cache the last N events per request in a server-side ring buffer, keyed by `request_id`.
3. On reconnect, look up the ring buffer by request ID, find the first event after `Last-Event-ID`, and replay from there.

![SSE reconnect state machine with ring buffer replay](/imgs/blogs/streaming-and-sse-for-llms-7.png)

```python
from collections import deque
from typing import Dict

# Per-request ring buffer: {request_id: deque of (event_id, sse_chunk)}
REPLAY_BUFFERS: Dict[str, deque] = {}
BUFFER_TTL_SECS = 30
MAX_BUFFER_SIZE = 64


async def stateful_stream_generator(
    prompt: str,
    sampling_params: SamplingParams,
    request_id: str,
    raw_request: Request,
    resume_from_event_id: int | None = None,
) -> AsyncGenerator[str, None]:
    # Initialize or retrieve ring buffer.
    if request_id not in REPLAY_BUFFERS:
        REPLAY_BUFFERS[request_id] = deque(maxlen=MAX_BUFFER_SIZE)
    ring = REPLAY_BUFFERS[request_id]

    event_id = 0
    previous_text = ""

    # If resuming, replay buffered events after the last-seen ID.
    if resume_from_event_id is not None:
        for buffered_id, buffered_chunk in ring:
            if buffered_id > resume_from_event_id:
                yield buffered_chunk

    async for output in engine.generate(prompt, sampling_params, request_id):
        current = output.outputs[0].text
        delta = current[len(previous_text):]
        previous_text = current

        if not delta:
            continue

        event_id += 1
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": output.outputs[0].finish_reason}],
        }
        sse_frame = f"id: {event_id}\ndata: {json.dumps(chunk)}\n\n"
        ring.append((event_id, sse_frame))
        yield sse_frame

    yield "data: [DONE]\n\n"
    # Schedule buffer cleanup after TTL.
    asyncio.get_event_loop().call_later(
        BUFFER_TTL_SECS, lambda: REPLAY_BUFFERS.pop(request_id, None)
    )
```

On the server endpoint, extract the `Last-Event-ID` header to detect reconnects:

```python
@app.post("/v1/chat/completions/stream/{request_id}")
async def resume_stream(request_id: str, raw_request: Request):
    last_event_id_header = raw_request.headers.get("Last-Event-ID")
    resume_from = int(last_event_id_header) if last_event_id_header else None

    if request_id not in REPLAY_BUFFERS and resume_from is not None:
        # Buffer expired — generation already finished or TTL exceeded.
        from fastapi.responses import Response
        return Response(status_code=410)  # Gone

    # Re-attach to the in-flight generation using the existing request_id.
    # vLLM uses the request_id to locate the in-flight request.
    sampling_params = SamplingParams(max_tokens=1024)  # fetch from cache in production
    return StreamingResponse(
        stateful_stream_generator(None, sampling_params, request_id, raw_request, resume_from),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

The 410 Gone response on buffer expiry is important — it tells the client that reconnect is impossible (the generation finished or the TTL elapsed) so the client can stop retrying. Without it, a client that reconnects after 30 seconds will keep receiving 200 OK with no data and looping forever.



## 9. Client-side implementation

### 9.1 JavaScript EventSource API

The browser's native `EventSource` API handles SSE with auto-reconnect out of the box. For LLM chat, the pattern is:

```javascript
async function streamChat(messages, onToken, onDone, onError) {
  // Note: EventSource only supports GET. For POST with body,
  // use fetch() with a ReadableStream instead.
  const response = await fetch("/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages, stream: true }),
  });

  if (!response.ok) {
    onError(new Error(`HTTP ${response.status}`));
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    // Decode bytes, accumulate incomplete SSE frames across read() calls.
    buffer += decoder.decode(value, { stream: true });

    // Split on SSE event boundaries (\n\n).
    const events = buffer.split("\n\n");
    buffer = events.pop(); // last element may be incomplete

    for (const event of events) {
      if (!event.trim()) continue;
      const dataLine = event.split("\n").find((l) => l.startsWith("data: "));
      if (!dataLine) continue;
      const data = dataLine.slice(6); // remove "data: "

      if (data === "[DONE]") {
        onDone();
        return;
      }

      try {
        const chunk = JSON.parse(data);
        const content = chunk.choices?.[0]?.delta?.content ?? "";
        if (content) onToken(content);
      } catch (e) {
        // Malformed JSON — log and skip.
        console.warn("SSE parse error:", e, data);
      }
    }
  }
}
```

Why use `fetch()` with a `ReadableStream` rather than the native `EventSource`? Because `EventSource` only supports GET requests. LLM chat APIs require POST with a JSON body. The `fetch()` approach gives you a `ReadableStream` that you read chunk by chunk — functionally identical to `EventSource` minus the auto-reconnect (which you would need to implement manually for this pattern).

The `buffer` variable accumulates partial SSE frames across multiple `read()` calls, which is essential. A single `read()` call may return less than a full SSE event — for example, it might return `data: {"choices":[{"delta"` and the rest comes in the next call. The split-on-`\n\n` and save-the-tail pattern handles this correctly.

### 9.2 Python SSE client with `sseclient-py`

For server-to-server or testing use cases, `sseclient-py` handles SSE parsing:

```python
import json
import requests
import sseclient


def stream_completion(prompt: str, base_url: str = "http://localhost:8000") -> str:
    """Stream a completion and return the full text."""
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": prompt}], "stream": True},
        stream=True,
        timeout=300,
    )
    response.raise_for_status()

    client = sseclient.SSEClient(response)
    full_text = ""

    for event in client.events():
        if event.data == "[DONE]":
            break
        try:
            chunk = json.loads(event.data)
            delta = chunk["choices"][0]["delta"].get("content", "")
            if delta:
                print(delta, end="", flush=True)  # real-time terminal output
                full_text += delta
        except (json.JSONDecodeError, KeyError):
            continue

    print()  # newline after completion
    return full_text


if __name__ == "__main__":
    result = stream_completion("Explain backpressure in three sentences.")
    print(f"\nFull response ({len(result)} chars): {result[:100]}...")
```

### 9.3 Async Python client with httpx

For async Python consumers, `httpx` with `AsyncClient` is cleaner than `requests` + `sseclient-py`:

```python
import asyncio
import json
import httpx


async def async_stream_completion(prompt: str, base_url: str = "http://localhost:8000") -> str:
    """Async streaming completion using httpx."""
    full_text = ""
    buffer = ""

    async with httpx.AsyncClient(timeout=300.0) as client:
        async with client.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": prompt}], "stream": True},
        ) as response:
            response.raise_for_status()

            async for chunk in response.aiter_text():
                buffer += chunk
                # Process complete SSE events from buffer.
                while "\n\n" in buffer:
                    event, buffer = buffer.split("\n\n", 1)
                    for line in event.split("\n"):
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            return full_text
                        try:
                            parsed = json.loads(data)
                            content = parsed["choices"][0]["delta"].get("content", "")
                            if content:
                                print(content, end="", flush=True)
                                full_text += content
                        except (json.JSONDecodeError, KeyError):
                            pass

    return full_text


if __name__ == "__main__":
    result = asyncio.run(async_stream_completion("What is LLM token streaming?"))
```

### 9.4 Handling partial UTF-8 in multi-byte tokens

LLMs produce tokens that can split Unicode code points across SSE events. A Chinese character like 你 (U+4F60) encodes as three bytes `\xe4\xbd\xa0` in UTF-8. If vLLM yields the first byte as one token and the remaining two bytes as the next token, naive `str.decode('utf-8')` on each SSE event individually will throw a `UnicodeDecodeError`. This is especially common with multilingual models serving Chinese, Japanese, Korean, or Arabic content, where a single displayed character may require 2–4 bytes in UTF-8.

The solution is to use Python's incremental decoder:

```python
import codecs

decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')

for raw_bytes in byte_stream:
    text = decoder.decode(raw_bytes, final=False)
    if text:
        process_token(text)

# Flush remaining bytes at end of stream.
final = decoder.decode(b'', final=True)
if final:
    process_token(final)
```

In JavaScript, the `TextDecoder` with `{ stream: true }` handles this automatically, which is why the JavaScript example uses `decoder.decode(value, { stream: true })`.



## 10. Streaming in multi-turn conversations

Multi-turn chat creates a state management challenge: each turn must include the full message history, but the current assistant turn is being streamed. You need to accumulate the streamed response and include it as the assistant message for the next turn.

The client-side pattern is straightforward:

```javascript
const conversationHistory = [];

async function sendMessage(userText) {
  // Add user message to history.
  conversationHistory.push({ role: "user", content: userText });

  let assistantMessage = "";

  await streamChat(
    conversationHistory,
    (token) => {
      assistantMessage += token;
      renderToken(token); // Update UI incrementally.
    },
    () => {
      // Stream complete — add full assistant response to history.
      conversationHistory.push({
        role: "assistant",
        content: assistantMessage,
      });
    },
    (error) => console.error("Stream error:", error)
  );
}
```

On the server side, if you are using vLLM with continuous batching, each multi-turn turn is a new generation request with the full serialized context. The KV cache from previous turns is not reused by default (unless you enable prefix caching). With [prefix caching](/blog/machine-learning/model-serving/vllm-deep-dive) enabled in vLLM, repeated context prefixes (system prompt + conversation history) are cached, dramatically reducing TTFT for later turns in a long conversation.

#### Worked example: multi-turn context growth and TTFT impact

Suppose a system prompt is 200 tokens and each user/assistant turn averages 150 tokens. After 10 turns:

- Total context: $200 + 10 \times 150 = 1{,}700$ tokens
- Without prefix caching on H100 SXM5: TTFT ≈ $1{,}700 \times 0.22$ ms $\approx 374$ ms
- With prefix caching (all but last turn cached): TTFT ≈ $(150) \times 0.22$ ms $\approx 33$ ms + cache lookup

The 11× TTFT reduction from prefix caching makes multi-turn conversations feel qualitatively more responsive by the 5th–10th turn. Without it, TTFT grows linearly with conversation length, eventually crossing the 1-second threshold that feels "slow" to users. See [vLLM deep dive](/blog/machine-learning/model-serving/vllm-deep-dive) for prefix caching configuration details.

### 10.1 Server-side session state for multi-turn streaming

For multi-turn conversations, the server needs to know the conversation history. Three patterns:

**Stateless (send full history each request).** The simplest approach: the client sends the full `messages` array on every request. The server has no state. Each request is independent and can be routed to any server replica. This is what OpenAI's API does and what vLLM's OpenAI-compatible endpoint expects. The cost is growing prompt tokens: a 10-turn conversation with 150 tokens per turn adds 1,500 tokens to the prefill on every turn.

**Stateful with session affinity.** The server caches conversation state and the client sends only a `session_id` and the new user message. The router must always route a session to the same replica. This reduces prompt tokens but creates sticky routing requirements that complicate horizontal scaling. With prefix caching enabled in vLLM, stateless with full history is nearly equivalent in TTFT to stateful, so prefer stateless.

**Hybrid with KV cache transfer.** The most advanced approach (described in [prefill-decode disaggregation](../prefill-decode-disaggregation) literature): the KV cache for the context is transferred between nodes when the request is rerouted, so session affinity is not required. This is how production systems like Kimi K2 handle multi-turn at scale.

For most deployments, stateless full-history is the right choice. Its TTFT at turn N is:

$$T_{TTFT}^{(N)} = (S + N \cdot L_{avg}) \cdot t_{prefill\,per\,token}$$

where $S$ is system prompt tokens, $L_{avg}$ is average tokens per turn, and $t_{prefill\,per\,token}$ ≈ 0.22 ms on H100. With prefix caching, all but the latest turn hits the cache, reducing effective prefill time to approximately $L_{avg} \cdot t_{prefill\,per\,token}$ after the first turn.



## 11. Performance implications of streaming

Streaming does not change the model's computational throughput. The GPU runs the identical decode loop at identical speed. The KV cache usage is identical. The only differences are in the I/O path between the GPU and the client.

**What changes:**
- The server calls `write()` approximately once per token instead of once at the end. At 40 tok/s and 50-byte frames, this is 40 `write()` calls per second, each taking ~0.05–0.1 ms kernel time — about 2–4 ms total overhead per request per second of generation. Negligible.
- TCP ACK traffic increases proportionally with the number of chunks. Each `write()` triggers an ACK from the client. For a 500-token response, this is 500 additional ACK packets over what a non-streaming response would generate. At 100 bytes per ACK packet and gigabit LAN speeds, this is noise.
- With Nagle's algorithm enabled (default), the kernel may buffer small `write()` calls to fill a segment. For SSE this is counterproductive — you want each token delivered immediately. Disable Nagle by setting `TCP_NODELAY` on the socket:

```python
# In FastAPI with uvicorn, configure TCP_NODELAY in the uvicorn.Config.
import uvicorn

if __name__ == "__main__":
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        http="h11",
        # TCP_NODELAY is set per-connection in recent uvicorn versions.
        # Ensure you're on uvicorn >= 0.20.0.
    )
    server = uvicorn.Server(config)
    server.run()
```

In practice, modern uvicorn and gunicorn versions set `TCP_NODELAY` by default. If you observe tokens being batched in groups of 2–4 at the network layer, check whether Nagle's algorithm is enabled.

**What does not change:** TPOT (time per output token), GPU utilization, KV cache memory consumption, batch throughput, or queue depth at the vLLM scheduler. All of these are determined by the GPU compute and memory bandwidth, not by SSE framing overhead.

### 11.1 Measuring and instrumenting streaming latency

TTFT and TPOT are the two metrics that matter for streaming serving quality. Both need Prometheus instrumentation. Here is a Prometheus histogram setup for a FastAPI SSE endpoint:

```python
from prometheus_client import Histogram, Counter
import time

TTFT_HISTOGRAM = Histogram(
    "llm_ttft_seconds",
    "Time to first token in seconds",
    buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0],
)

TPOT_HISTOGRAM = Histogram(
    "llm_tpot_milliseconds",
    "Time per output token in milliseconds",
    buckets=[5, 10, 15, 20, 25, 30, 50, 100, 200],
)

TOKENS_GENERATED = Counter(
    "llm_tokens_total",
    "Total output tokens generated",
)

REQUESTS_TOTAL = Counter(
    "llm_requests_total",
    "Total streaming requests",
    ["status"],  # "success", "aborted", "error"
)


async def instrumented_vllm_generator(
    prompt: str, sampling_params: SamplingParams, request_id: str, raw_request: Request
) -> AsyncGenerator[str, None]:
    request_start = time.perf_counter()
    first_token_time = None
    last_token_time = None
    token_count = 0
    previous_text = ""

    try:
        async for output in engine.generate(prompt, sampling_params, request_id):
            if await raw_request.is_disconnected():
                await engine.abort(request_id)
                REQUESTS_TOTAL.labels(status="aborted").inc()
                return

            current = output.outputs[0].text
            delta = current[len(previous_text):]
            previous_text = current

            if delta:
                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now
                    TTFT_HISTOGRAM.observe(first_token_time - request_start)
                elif last_token_time is not None:
                    tpot_ms = (now - last_token_time) * 1000
                    TPOT_HISTOGRAM.observe(tpot_ms)
                last_token_time = time.perf_counter()
                token_count += 1
                TOKENS_GENERATED.inc()

                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": output.outputs[0].finish_reason}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

        REQUESTS_TOTAL.labels(status="success").inc()

    except Exception as e:
        REQUESTS_TOTAL.labels(status="error").inc()
        raise

    yield "data: [DONE]\n\n"
```

The corresponding Prometheus alert rules for SLO monitoring:

```yaml
# prometheus-rules.yaml
groups:
  - name: llm_streaming_slo
    rules:
      - alert: HighTTFT
        expr: |
          histogram_quantile(0.95, sum(rate(llm_ttft_seconds_bucket[5m])) by (le)) > 0.5
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "p95 TTFT {{ $value | humanizeDuration }} exceeds 500ms SLO"

      - alert: HighTPOT
        expr: |
          histogram_quantile(0.95, sum(rate(llm_tpot_milliseconds_bucket[5m])) by (le)) > 50
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "p95 TPOT {{ $value }}ms exceeds 50ms threshold"

      - alert: HighAbortRate
        expr: |
          rate(llm_requests_total{status="aborted"}[5m]) /
          rate(llm_requests_total[5m]) > 0.05
        for: 5m
        annotations:
          summary: "Abort rate {{ $value | humanizePercentage }} — check for client disconnect storms"
```

A high abort rate (>5%) indicates clients are disconnecting before generation completes — a signal that either TTFT is too high (clients gave up waiting) or the application has a client-side bug in disconnect handling. High TPOT usually indicates GPU memory pressure from too many concurrent KV caches — the signal to scale out or reduce `max_model_len`. For a complete observability setup, see [observability for model servers](/blog/machine-learning/model-serving/the-model-serving-playbook).

### 11.2 Streaming and the SLO triangle

The [SLO triangle](/blog/machine-learning/model-serving/what-is-model-serving) for streaming serving has specific trade-off characteristics:

**Latency (TTFT)** is improved by: smaller models, faster hardware (H100 > A100 > T4), shorter prompts, lower concurrent request count, chunked prefill enabled.

**Throughput (aggregate tok/s)** is improved by: larger batch sizes, continuous batching, speculative decoding, quantization (more requests fit in GPU memory).

**Cost (\$ per million tokens)** is reduced by: quantization (serving more requests per GPU hour), larger batches (better GPU utilization), spot instances for batch processing.

Streaming sits on the latency axis — it is a zero-cost improvement to perceived latency that trades nothing on throughput or cost. The buffering and reconnect machinery described in this post adds a few hundred lines of code and negligible CPU overhead, but no GPU or network cost. For any user-facing LLM product, the streaming investment is the highest-ROI latency optimization available — better than hardware upgrades, better than quantization, because it is free.



## 12. gRPC server streaming: the alternative for internal services

For microservice-to-microservice communication where browser compatibility is not required, gRPC server streaming is more efficient than SSE. It uses Protocol Buffers for binary encoding (smaller wire size, faster serialization), HTTP/2 multiplexing (multiple streams over one connection, per-stream flow control), and built-in HPACK header compression.

The protobuf service definition for a streaming LLM endpoint:

```protobuf
syntax = "proto3";

package llmserving.v1;

service LLMService {
  rpc StreamCompletion(CompletionRequest) returns (stream CompletionChunk);
}

message CompletionRequest {
  string prompt = 1;
  float temperature = 2;
  int32 max_tokens = 3;
  string request_id = 4;
}

message CompletionChunk {
  string request_id = 1;
  string delta_text = 2;
  string finish_reason = 3;
  int32 event_id = 4;
}
```

The Python gRPC server implementation:

```python
import asyncio
import grpc
from concurrent import futures
from vllm import AsyncLLMEngine, SamplingParams

import llmserving_pb2
import llmserving_pb2_grpc


class LLMServicer(llmserving_pb2_grpc.LLMServiceServicer):
    def __init__(self, engine: AsyncLLMEngine):
        self.engine = engine

    async def StreamCompletion(self, request, context):
        sampling_params = SamplingParams(
            temperature=request.temperature or 0.7,
            max_tokens=request.max_tokens or 1024,
        )
        previous_text = ""
        event_id = 0

        try:
            async for output in self.engine.generate(
                request.prompt, sampling_params, request.request_id
            ):
                current = output.outputs[0].text
                delta = current[len(previous_text):]
                previous_text = current

                if delta:
                    event_id += 1
                    yield llmserving_pb2.CompletionChunk(
                        request_id=request.request_id,
                        delta_text=delta,
                        finish_reason=output.outputs[0].finish_reason or "",
                        event_id=event_id,
                    )
        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, str(e))


async def serve():
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    server = grpc.aio.server()
    llmserving_pb2_grpc.add_LLMServiceServicer_to_server(
        LLMServicer(engine), server
    )
    server.add_insecure_port("[::]:50051")
    await server.start()
    await server.wait_for_termination()
```

![SSE vs gRPC stream overhead at 10k concurrent connections](/imgs/blogs/streaming-and-sse-for-llms-8.png)

At 10,000 concurrent streams, gRPC uses roughly 40% less CPU and 60% less memory per stream compared to SSE over HTTP/1.1. The primary reason is HTTP/2 multiplexing: all 10,000 gRPC streams share a small number of TCP connections, eliminating the per-connection kernel state that HTTP/1.1 requires. However, gRPC requires `grpc-web` or a proxy for browser clients, adding infrastructure complexity. The comparison table in section 12 summarizes when to use each.

### 12.1 gRPC client for streaming

The Python gRPC client for consuming a streaming LLM endpoint:

```python
import asyncio
import grpc

import llmserving_pb2
import llmserving_pb2_grpc


async def stream_grpc_completion(
    prompt: str,
    host: str = "localhost:50051",
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> str:
    """Stream a completion from the gRPC LLM service."""
    async with grpc.aio.insecure_channel(host) as channel:
        stub = llmserving_pb2_grpc.LLMServiceStub(channel)
        request = llmserving_pb2.CompletionRequest(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            request_id=f"req-{id(prompt)}",
        )

        full_text = ""
        async for chunk in stub.StreamCompletion(request):
            print(chunk.delta_text, end="", flush=True)
            full_text += chunk.delta_text
            if chunk.finish_reason in ("stop", "length"):
                break

        print()  # newline
        return full_text


if __name__ == "__main__":
    result = asyncio.run(stream_grpc_completion("What is the capital of France?"))
    print(f"Result: {result}")
```

### 12.2 gRPC flow control configuration

gRPC's HTTP/2 flow control has configurable initial window sizes. The defaults (65,535 bytes initial window, 1 MB maximum window) are appropriate for most LLM streaming use cases. For very high-throughput internal services, increase the initial window:

```python
import grpc

async def serve_with_large_window():
    options = [
        ("grpc.max_concurrent_streams", 1000),
        ("grpc.initial_window_size", 1048576),         # 1 MB per-stream window
        ("grpc.initial_connection_window_size", 10485760),  # 10 MB per-conn window
        ("grpc.keepalive_time_ms", 30000),             # Keepalive ping every 30s
        ("grpc.keepalive_timeout_ms", 10000),           # Wait 10s for keepalive ack
        ("grpc.keepalive_permit_without_calls", True),  # Allow keepalive on idle
    ]
    server = grpc.aio.server(options=options)
    # ... add servicer and start ...
```

The `keepalive` settings prevent gRPC connections from being silently dropped by network intermediaries — the same problem that SSE heartbeats address.

### 12.3 SSE-to-gRPC translation gateway pattern

In a mixed deployment (browser clients via SSE, internal services via gRPC), a translation gateway converts between protocols:

```
Browser → [SSE] → Gateway → [gRPC] → vLLM backend
Internal service → [gRPC] → Gateway → [gRPC] → vLLM backend
```

The gateway implementation in Python:

```python
import asyncio
import json
import grpc
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

import llmserving_pb2
import llmserving_pb2_grpc

app = FastAPI()

# Single gRPC channel to the vLLM backend cluster.
# In production, use a channel pool for load balancing.
grpc_channel = grpc.aio.insecure_channel(
    "vllm-backend:50051",
    options=[
        ("grpc.max_reconnect_backoff_ms", 5000),
        ("grpc.initial_reconnect_backoff_ms", 500),
    ],
)
grpc_stub = llmserving_pb2_grpc.LLMServiceStub(grpc_channel)


async def grpc_to_sse_generator(
    request: llmserving_pb2.CompletionRequest,
) -> AsyncGenerator[str, None]:
    """Translate gRPC stream to SSE format for browser clients."""
    event_id = 0
    try:
        async for chunk in grpc_stub.StreamCompletion(request):
            event_id += 1
            sse_payload = {
                "id": chunk.request_id,
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk.delta_text},
                        "finish_reason": chunk.finish_reason or None,
                    }
                ],
            }
            yield f"id: {event_id}\ndata: {json.dumps(sse_payload)}\n\n"
    except grpc.RpcError as e:
        error_payload = {"error": {"message": str(e.details()), "code": e.code().value[0]}}
        yield f"data: {json.dumps(error_payload)}\n\n"

    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_via_gateway(raw_request: Request):
    body = await raw_request.json()
    messages = body.get("messages", [])
    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    grpc_request = llmserving_pb2.CompletionRequest(
        prompt=prompt,
        temperature=body.get("temperature", 0.7),
        max_tokens=body.get("max_tokens", 1024),
        request_id=f"req-{id(body)}",
    )

    return StreamingResponse(
        grpc_to_sse_generator(grpc_request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

This gateway pattern is how production LLM serving infrastructure typically works: the user-facing API layer speaks SSE (for browser compatibility), the internal compute layer speaks gRPC (for efficiency), and a thin translation layer bridges them.



## 13. Case studies and benchmarks

### 13.1 OpenAI's SSE implementation

OpenAI's ChatGPT API uses SSE with the exact format described in section 3. Their public benchmarks (reported in the GPT-4 Technical Report, 2023) show median TTFT of approximately 400–700 ms for GPT-4, and 200–400 ms for GPT-3.5-turbo, depending on prompt length and server load. The streaming delivery is standard SSE with the `[DONE]` sentinel, and their Python SDK (`openai.ChatCompletion.create(stream=True)`) wraps `sseclient` internally.

### 13.2 vLLM streaming throughput characterization

The vLLM paper (Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023) reports that vLLM's AsyncLLMEngine with streaming adds less than 2% overhead compared to non-streaming serving, measured by total throughput tokens/second at the server. Their benchmarks on A100 40GB with OPT-13B show 22.5 req/s for streaming vs 23.1 req/s without streaming — within measurement noise.

### 13.3 TGI streaming benchmarks

HuggingFace's Text Generation Inference (TGI) blog post on streaming performance (2023) reports that with Falcon-7B on A100 80GB at 50 concurrent users, streaming mode achieves 1,200 tok/s aggregate throughput with median TTFT of 180 ms. Without streaming, all tokens are delivered at the end, but the underlying throughput is identical at 1,195 tok/s.

### 13.4 gRPC vs SSE at scale: infrastructure comparison

gRPC server streaming achieves approximately 38% lower CPU utilization than SSE at 5,000+ concurrent streams. This is consistent with the HTTP/2 multiplexing argument: fewer TCP connections, better kernel scheduling. The primary reason for choosing SSE for external-facing LLM APIs remains browser compatibility and proxy transparency.

| Metric | SSE/HTTP/1.1 | SSE/HTTP/2 | gRPC/HTTP/2 |
|---|---|---|---|
| CPU @ 10k streams | ~18%/core | ~14%/core | ~11%/core |
| Memory per stream | ~4 KB | ~1.5 KB | ~0.8 KB |
| Latency per token | ~0.05 ms | ~0.04 ms | ~0.01 ms |
| Browser native | Yes | Partial | No |
| Auto-reconnect | Built in | Built in | Manual |
| Proxy transparent | Yes | Yes | No |

### 13.5 TTFT benchmarks across hardware and model sizes

To give concrete reference numbers for capacity planning, here are TTFT measurements for streaming serving at batch size 1 (single concurrent request), measured as the wall-clock time from request receipt to the first SSE event arriving at the client. These numbers are derived from vLLM benchmark documentation, TGI benchmarks, and the vLLM paper characterization on standard hardware:

| Model | Hardware | Prompt Tokens | TTFT (median) | TPOT (median) | Throughput (1 req) |
|---|---|---|---|---|---|
| Llama-3-8B-Instruct | H100 80GB SXM5 | 128 | 65 ms | 22 ms | 45 tok/s |
| Llama-3-8B-Instruct | H100 80GB SXM5 | 1024 | 220 ms | 22 ms | 45 tok/s |
| Llama-3-70B-Instruct | H100 80GB SXM5 (TP=2) | 512 | 380 ms | 38 ms | 26 tok/s |
| Llama-3-8B-Instruct | A100 40GB | 512 | 280 ms | 30 ms | 33 tok/s |
| Mistral-7B-Instruct | A100 80GB | 512 | 220 ms | 28 ms | 36 tok/s |
| Llama-3-8B-Instruct | T4 16GB | 512 | 820 ms | 95 ms | 10 tok/s |

At high concurrency (50 simultaneous requests), TTFT increases substantially due to queuing at the vLLM scheduler:

| Model | Hardware | Concurrent Requests | TTFT p50 | TTFT p99 | Aggregate tok/s |
|---|---|---|---|---|---|
| Llama-3-8B-Instruct | H100 80GB | 50 | 420 ms | 1,800 ms | 1,200 tok/s |
| Llama-3-8B-Instruct | H100 80GB | 100 | 850 ms | 3,500 ms | 1,400 tok/s |
| Llama-3-70B-Instruct | 2× H100 80GB | 20 | 610 ms | 2,200 ms | 380 tok/s |

The pattern is clear: TTFT scales roughly linearly with queue depth (more concurrent requests → longer wait for prefill scheduling), but aggregate throughput approaches a ceiling as the batch size grows. For a user-facing API with a TTFT SLO of 500 ms p99, Llama-3-8B on a single H100 can handle roughly 30–40 concurrent active requests before the SLO is violated.

#### Worked example: capacity planning for a 500 ms TTFT SLO

You are launching a chat product that must deliver first tokens within 500 ms at p95. Your model is Llama-3-8B-Instruct on H100 80GB. Assume:
- Median prompt length: 512 tokens
- Expected QPS at peak: 20 requests/second
- Average generation: 300 tokens per response

From the benchmark table, at 50 concurrent requests on H100, p99 TTFT is 1,800 ms — too high. At 30 concurrent requests (estimated from linear interpolation), p95 TTFT is approximately 750 ms — still too high for the 500 ms target.

Options:
1. **Reduce concurrent requests per instance.** Cap at 20 active requests per H100 instance. At 20 concurrent with Llama-3-8B, estimated p95 TTFT ≈ 380 ms. Scale horizontally to serve 20 QPS across instances. With 300-token responses at 45 tok/s, each request takes 300/45 ≈ 6.7 seconds. At 20 concurrent, you saturate a single instance at 20/6.7 ≈ 3 QPS. To serve 20 QPS you need approximately 7 H100 instances.

2. **Use chunked prefill to bound TTFT under load.** vLLM's `chunked_prefill` splits long prompts into chunks processed over multiple scheduling rounds, limiting the per-round prefill time. With `max_chunked_prefill_tokens=512`, even a 2,048-token prompt only adds 4 scheduling rounds of prefill latency, each 25 ms, rather than one 200 ms prefill that stalls all other requests. This can reduce p99 TTFT by 40–60% at high concurrency.

3. **Deploy a smaller model for latency-critical traffic.** A Llama-3-8B at FP8 quantization on H100 achieves ~70 tok/s TPOT, reducing generation time from 6.7 s to 4.3 s, allowing each instance to handle 4.7 QPS at 20 concurrent — cutting the fleet from 7 to 5 instances for the same SLO.

The right answer for most products is option 2 first (it is free), then option 3 (quantization), then option 1 (scale out). See [quantization for LLM serving](/blog/machine-learning/model-serving/quantization-for-llm-serving) for the quantization trade-off analysis.



## 14. When to use streaming (and when not to)

**Always use streaming for:**
- Any user-facing LLM chat interface. The UX improvement is mandatory.
- LLM APIs intended for third-party developers — they expect the OpenAI streaming format.
- Long responses (≥100 tokens) where generation takes more than 2 seconds.
- Agentic flows where the user is watching the assistant "think" — intermediate reasoning should stream.

**Skip streaming when:**
- The use case is programmatic, batch processing where latency doesn't matter (e.g., embedding generation, offline document summarization, dataset labeling).
- The response is very short (≤20 tokens) and generation time is under 500 ms — the streaming overhead and protocol complexity exceed the UX benefit.
- Your downstream processing requires the complete response before it can proceed (e.g., passing the response to a classifier or structured output parser). In this case, buffer the stream server-side and return a complete response.
- You are behind a proxy or load balancer that does not support SSE and cannot be reconfigured. Check with your infrastructure team before building streaming.

**gRPC streaming over SSE when:**
- The consumer is an internal microservice, never a browser.
- You have ≥1,000 concurrent streams and CPU efficiency matters.
- You are already running a gRPC service mesh (e.g., Istio) and have the tooling for it.
- Strict binary protocol requirements (e.g., compliance constraints on traffic inspection of text-format SSE).

**SSE over gRPC when:**
- Any browser client needs to consume the stream directly.
- You need auto-reconnect without implementing it yourself.
- Your infrastructure has proxy/CDN layers that do not speak gRPC.
- You want to be compatible with OpenAI client libraries.

### 14.1 Common implementation mistakes

To close out this section, here are the five most common streaming implementation mistakes observed in real deployments, and how to avoid them:

**Mistake 1: Not setting `X-Accel-Buffering: no`.** The symptom is that streaming appears to work in development (no nginx) but all tokens arrive at once in production (behind nginx). Fix: always include the header in `StreamingResponse`.

**Mistake 2: Not aborting the vLLM engine on disconnect.** The symptom is GPU memory climbing over time as KV caches accumulate for disconnected clients, eventually triggering OOM. Fix: check `is_disconnected()` every 10 tokens and call `engine.abort(request_id)`.

**Mistake 3: Forgetting the `[DONE]` sentinel.** Some clients (especially simple Python scripts using `requests.iter_content()`) loop forever waiting for more data after the last token. The `[DONE]` sentinel is the hard-coded signal to close the stream. Without it, clients hang.

**Mistake 4: Parsing the `[DONE]` sentinel as JSON.** The sentinel is the literal string `[DONE]`, not valid JSON. Trying to `json.loads("[DONE]")` raises `JSONDecodeError`. Always check for the sentinel before parsing.

**Mistake 5: Using `StreamingResponse` with a synchronous generator.** If you use `def` instead of `async def` for the generator, FastAPI runs it in a thread pool, blocking the event loop. Use `async def` and `async for` throughout, especially when calling vLLM's async API.



## 15. Key takeaways

1. **TTFT is the user-visible metric that streaming optimizes.** It changes perceived latency from $T_{prefill} + L/R$ to $T_{prefill}$ alone — a 10–60× improvement on typical LLM responses.

2. **SSE is the right default for browser-facing LLM APIs.** It is HTTP/1.1 compatible, has auto-reconnect built into `EventSource`, traverses proxies transparently, and adds less than 0.1 ms per token in overhead.

3. **Set `X-Accel-Buffering: no` whenever nginx is in the path.** Without it, nginx buffers your SSE stream and defeats streaming entirely.

4. **Handle `asyncio.CancelledError` and call `engine.abort(request_id)`.** Client disconnects must free the KV cache slot or vLLM will continue generating tokens for nobody and exhaust memory.

5. **vLLM's `AsyncLLMEngine` yields cumulative text, not deltas.** Always compute the delta by subtracting the previous length: `delta = current[len(previous):]`.

6. **Token buffering smooths jitter but adds end-to-end latency.** Only add it if your generation rate varies significantly and users report jitter. Set the drain rate to match your users' reading speed (typically 25–40 tok/s).

7. **Stateful reconnect requires a per-request ring buffer on the server.** Assign sequential event IDs, cache the last 64 events per request, expire after 30 seconds, and return 410 Gone when the buffer has expired.

8. **gRPC server streaming uses 40% less CPU than SSE at 10k concurrent streams** due to HTTP/2 multiplexing and binary encoding. Use it for internal services; use SSE for browser-facing APIs.

9. **TCP flow control provides natural backpressure for SSE.** When the client receive buffer fills, the server's `write()` blocks and the async generator pauses. Add an explicit `asyncio.Queue(maxsize=...)` to cap memory consumption from slow clients.

10. **Streaming adds zero GPU overhead.** TPOT, KV cache usage, and batch throughput are identical between streaming and non-streaming serving.

11. **Send heartbeat SSE comment events every 15–30 seconds.** Proxy servers (AWS ALB, Cloudflare) close idle HTTP connections on timeout. A comment event `": heartbeat\n\n"` keeps the connection alive without affecting the token stream.

12. **Use `httpx.AsyncClient` or `sseclient-py` for Python consumers.** The `requests` library does not handle streaming natively without explicit `stream=True`; `httpx` async with `aiter_text()` is the clean modern pattern. For JavaScript, use `fetch()` with `ReadableStream` rather than `EventSource` to support POST requests with a body.

13. **Monitor TTFT, TPOT, and the abort rate as your three core streaming SLO metrics.** TTFT tells you about queue depth and prefill scheduling; TPOT tells you about GPU memory pressure and batch size health; the abort rate tells you whether users are giving up because TTFT is too high — it is the canary that fires before user complaints reach your inbox.



## 16. Further reading

- [Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023](https://dl.acm.org/doi/10.1145/3600006.3613165) — the vLLM paper; appendix covers streaming latency measurements.
- [vLLM documentation: async engine](https://docs.vllm.ai/en/latest/dev/engine/async_llm_engine.html) — `AsyncLLMEngine` API reference with streaming examples.
- [MDN Web Docs: Server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events) — authoritative SSE protocol reference including `Last-Event-ID` reconnect behavior.
- [gRPC Python Async API](https://grpc.github.io/grpc/python/grpc_asyncio.html) — server streaming servicer patterns.
- **Series: [What is model serving](/blog/machine-learning/model-serving/what-is-model-serving)** — the foundation: latency/throughput/cost SLO triangle.
- **Series: [vLLM deep dive](/blog/machine-learning/model-serving/vllm-deep-dive)** — chunked prefill, prefix caching, speculative decoding: the engine that drives the streaming generator.
- **Series: [Text Generation Inference deep dive](/blog/machine-learning/model-serving/text-generation-inference-deep-dive)** — TGI's token streaming implementation and Flash Attention integration.
- **Series: [The model serving playbook](/blog/machine-learning/model-serving/the-model-serving-playbook)** — capstone: complete decision tree from notebook to production.

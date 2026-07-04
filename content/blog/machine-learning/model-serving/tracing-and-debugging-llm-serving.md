---
title: "Tracing and Debugging LLM Serving: Finding Out Why a Request Was Slow"
date: "2026-07-04"
publishDate: "2026-07-04"
description: "A field guide to distributed tracing and systematic debugging for LLM serving — decompose end-to-end latency with span waterfalls, instrument vLLM and your gateway with OpenTelemetry, correlate metrics-traces-logs, drop to GPU-level profiling, and follow a symptom-to-fix decision tree."
tags:
  [
    "model-serving",
    "inference",
    "ml-infrastructure",
    "distributed-tracing",
    "opentelemetry",
    "observability",
    "llm-serving",
    "vllm",
    "debugging",
    "gpu-profiling",
    "latency",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/tracing-and-debugging-llm-serving-1.webp"
---

At 14:07 a Slack alert fired: **E2E p99 latency = 1.35 s**, well over our 800 ms target for the chat API. The on-call engineer did what most of us have done at least once. Restarted the pods — no change. Scaled the deployment from 6 to 8 GPUs — p99 stayed at 1.35 s and now the cloud bill was higher. Someone typed "maybe the model got slower?" into the incident channel, with no evidence at all. Forty minutes in, three engineers were staring at the same Grafana dashboard, watching a red line, with no idea *which part* of a request was slow. The dashboard could tell them the request was slow. It could not tell them *why*.

The problem was not a resource leak, a bad deploy, or a slow model. The problem was that an averaged latency metric is a single number stapled to the *end* of a request, and a request to an LLM server is a journey through at least six distinct stages — client to gateway, gateway to engine, a wait in the scheduler's queue, a compute-bound prefill, a memory-bound decode loop that runs once per output token, detokenization, and the stream back. A p99 number tells you the journey took 1.35 seconds. It hides the fact that 340 of those milliseconds were spent *waiting in a queue* because the engine was admitting too few concurrent sequences. No amount of restarting fixes a queue-wait problem, and no amount of adding GPUs fixes it either if the bottleneck is a scheduler config, not capacity.

This post is about the discipline that turns "it's slow" into "the queue-wait span is 340 ms because `max_num_seqs` is too low, and here is the one-line fix." That discipline is **distributed tracing**: instrument every stage of the request so each becomes a timed *span*, stitch the spans of one request into a single *trace*, and read the resulting waterfall to see exactly where time went. The figure below is that waterfall for the 1.35 s request — and the single glance it enables is the entire value proposition. Queue wait, not the model, dominates the controllable latency budget, and the trace makes it obvious in five seconds what forty minutes of guessing could not.

![A span waterfall for a 1.35-second request, breaking end-to-end latency into gateway, queue-wait, prefill, decode, and stream spans with milliseconds on each](/imgs/blogs/tracing-and-debugging-llm-serving-1.webp)

By the end you will be able to instrument a vLLM-based service end to end with OpenTelemetry, propagate trace context across your gateway and engine so the whole request reads as one trace, decompose a latency spike into its queue / prefill / decode terms, correlate a metric alert to the exact slow trace and its logs, drop to a GPU kernel profile with Nsight Systems or `torch.profiler` when the model itself is the suspect, and walk a decision tree from symptom to root cause to fix. Everything ties back to the serving SLO triangle — latency, throughput, cost — because tracing is the instrument that tells you *which* corner of that triangle you are actually paying for on any given request.

## The anatomy of a slow request: decomposing end-to-end latency

Before you can debug a latency number you have to know what it is made of. For a single LLM completion, the end-to-end wall-clock time decomposes into a small number of additive terms:

$$T_\text{e2e} = t_\text{gateway} + t_\text{queue} + t_\text{prefill} + t_\text{decode} + t_\text{detok+stream}$$

Two of these terms are the ones your users actually feel as distinct experiences. **TTFT** (time to first token) is how long the user waits before *anything* appears on screen; it is the sum of everything that must happen before the first output token is emitted:

$$T_\text{ttft} = t_\text{gateway} + t_\text{queue} + t_\text{prefill}$$

**TPOT** (time per output token, sometimes called inter-token latency) is the steady cadence of tokens after the first, and the decode term is just TPOT multiplied by the number of output tokens:

$$t_\text{decode} = n_\text{out} \times \text{TPOT}$$

These two formulas are the spine of every latency-debugging session you will ever run on an LLM server. If TTFT is bad and TPOT is fine, the problem lives in `gateway + queue + prefill`. If TPOT is bad, the problem lives in the decode loop. If both are bad, you are almost certainly overloaded and preempting. That single fork — *which term is abnormal* — eliminates more than half of the candidate causes before you have read a line of logs.

Each term is bounded by a different physical resource, and knowing which resource bounds which term tells you what a fix even *looks* like:

| Term | What it is | Bounded by | Typical value (13B on H100) | Who controls it |
|---|---|---|---|---|
| `t_gateway` | Auth, routing, serialization | Network + CPU | 3–15 ms | Gateway code, LB config |
| `t_queue` | Wait for a scheduler slot | Concurrency limits, load | 0 ms (idle) → 1 s+ (overload) | `max_num_seqs`, admission control, autoscaling |
| `t_prefill` | One forward pass over the prompt | GPU FLOP/s (compute-bound) | ~95 ms for 512 tokens | Prompt length, chunked prefill, batch |
| `t_decode` | `n_out` autoregressive steps | HBM bandwidth (memory-bound) | 7 ms/token → 896 ms for 128 | Output length, batch size, KV precision |
| `t_detok+stream` | Detokenize + network flush | CPU + client link | 5–20 ms | Tokenizer, SSE buffering |

The mechanics behind two of those "bounded by" claims are worth deriving, because they are the reason prefill and decode fail in different ways.

**Prefill is compute-bound.** A prefill pass computes attention and MLP over all `p` prompt tokens at once. The dominant cost is roughly ${2 N_\text{params} \cdot p}$ floating-point operations (two FLOPs per parameter per token for the matmuls), plus an $O(p^2)$ attention term. For a 13B model and a 512-token prompt that is about ${2 \times 13\times10^9 \times 512 \approx 1.3\times10^{13}}$ FLOPs. Divide by the *effective* throughput the engine actually sustains under a shared batch — call it ~150 TFLOP/s once you account for attention overhead and imperfect utilization — and you get roughly 90 ms. Prefill time therefore scales with prompt length, and its failure mode is a very long prompt monopolizing the GPU. The fix vocabulary is "chunked prefill," "prompt length caps," "prefix caching."

**Decode is memory-bound.** Each decode step generates exactly one token for each sequence in the batch, and to do so it must read the entire model's weights from HBM once. At batch size 1 there is no reuse: the per-step time floor is simply the model's byte footprint divided by memory bandwidth. A 13B model in fp16 is about 26 GB; an H100 SXM has 3.35 TB/s of HBM bandwidth; so the floor is

$$\text{TPOT}_\text{floor} \approx \frac{26 \times 10^9 \text{ bytes}}{3.35 \times 10^{12} \text{ bytes/s}} \approx 7.8 \text{ ms/token}$$

which is why our example shows 7 ms/token and 896 ms for 128 output tokens. This is also why batching *helps* decode: a bigger batch reads the same weights once but produces more tokens per read, raising throughput — right up until the KV cache reads and the growing per-token compute push you back toward compute-bound, at which point TPOT climbs. The decode failure mode is "batch too big / memory-bandwidth saturated," and the fix vocabulary is "cap the batch," "quantize the KV cache," "shorter outputs."

**Queue wait is a load phenomenon, not a per-request property.** The other terms are properties of the request itself — prompt length, output length — so they are stable for a given request no matter what else the system is doing. Queue wait is different: it is a property of the *system's occupancy* at the instant the request arrives, and it is the term that behaves nonlinearly. A serving engine has a finite number of concurrent sequence slots (`max_num_seqs`) and a finite KV-cache budget; when arrivals outpace completions, requests wait for a slot. Queuing theory gives the shape. For a system at utilization $\rho$ — the fraction of capacity in use — the expected wait scales roughly as $\frac{\rho}{1-\rho}$ times the service time. The load-bearing feature of that expression is the ${1-\rho}$ in the denominator: as utilization approaches 1, wait does not rise linearly, it rises asymptotically. At 80% the multiplier is 4; at 90% it is 9; at 95% it is 19. That is why queue wait can jump from 2 ms to 340 ms — a 170× swing — over a load increase that only moved utilization from 70% to 95%. The system did not get 170× busier; the request crossed the knee of the utilization curve.

The subtlety that trips up capacity dashboards is *which* utilization matters. The $\rho$ that governs queue wait is the occupancy of the engine's **sequence slots**, not GPU compute utilization, and the two diverge sharply for LLM decode. In the opening incident, slot occupancy was pinned near 100% — all 64 `max_num_seqs` slots full, everyone else queued — while GPU *compute* utilization read only 61%, because each admitted sequence's decode step is memory-bound and light on FLOPs. The bottleneck was slot capacity, not compute. A dashboard showing "61% GPU utilization" looks healthy and actively misleads; the trace, which measures the wait directly, does not. Raising `max_num_seqs` added slots, drained the queue, and — because the GPU had compute headroom — pushed utilization *up* to 88% with no new hardware. Adding GPUs, the instinctive fix, would have lowered utilization further and left the queue exactly where it was.

The uncomfortable consequence of these different bounds is that the same 1.35 s number decomposes completely differently under different load. The grid below reads each latency term across three load regimes. Prefill barely moves — it is a property of the prompt, not the fleet. Decode grows with output length. Queue wait is nearly zero at light load and explodes only under overload, a 170× swing from 2 ms to 340 ms. The term you attack depends entirely on where you are on this grid, which is exactly why a single averaged metric — which blends all three regimes together — is useless for localization.

![A three-by-three grid showing queue, prefill, and decode latency across light, normal, and overload regimes, with queue wait exploding 170x under overload while prefill stays flat](/imgs/blogs/tracing-and-debugging-llm-serving-8.webp)

#### Worked example: decomposing a 1.35 s p99 spike

Take the alert that opened this post. The metric said E2E p99 = 1.35 s. Pull the exemplar trace behind that percentile and its spans read:

- `gateway.forward` — 8 ms
- `llm_request` → `time_in_queue` attribute — **340 ms**
- prefill portion (TTFT − queue) — 95 ms (512 prompt tokens)
- decode portion (E2E − TTFT) — 896 ms (128 output tokens at ~7 ms each)
- detokenize + stream flush — 12 ms

Sum: `8 + 340 + 95 + 896 + 12 = 1351 ms`. Now decompose by *who controls each term*. The 896 ms decode is inherent to generating 128 tokens on this model — you cannot cut it without a smaller model, shorter outputs, or speculative decoding, none of which is an incident fix. The 95 ms prefill is inherent to a 512-token prompt. The 340 ms queue wait is the only term that is both large *and* controllable at the config level. That is the anomaly. TTFT = `8 + 340 + 95 = 443 ms`, and 77% of TTFT is queue wait. The trace has localized the problem to a single term in under a minute; the rest of this post is about how to produce and read that trace, and what to do once you have.

## Guessing is not debugging

It is worth dwelling on *why* the opening incident went the way it did, because the failure was not one of effort — three competent engineers worked it for forty minutes — but one of method. Without a trace, a slow request is opaque. You know the total time and nothing about its composition, so every hypothesis is equally plausible and equally unfalsifiable: maybe it's the model, maybe it's the GPU, maybe it's the network, maybe it's memory. In that state the natural move is to attack whatever resource is easiest to change — restart the pods, add GPUs — because those actions *feel* like progress. They are not. Restarting a stateless engine that is not leaking anything does nothing. Adding GPUs when the bottleneck is a scheduler admission limit does nothing, because the new GPUs sit at low utilization while requests still queue behind the same per-replica concurrency cap.

The contrast is stark once you put the two workflows side by side. On the left, blind remediation burns two GPUs and forty minutes and moves p99 not at all. On the right, one engineer reads the span waterfall, sees the 340 ms queue-wait span, recognizes it as the dominant controllable term, and ships a single config change. The difference is not intelligence or seniority. It is whether the request's internal structure was *visible*.

![Before-and-after contrast: blind guessing restarts pods and adds GPUs with no effect, while trace-guided debugging reads the waterfall, localizes queue wait, and fixes p99 with one config change](/imgs/blogs/tracing-and-debugging-llm-serving-4.webp)

The one-line fix in this case was to raise the engine's concurrency ceiling and add a small admission-control queue-depth cap so the scheduler admitted more sequences per step instead of letting them pile up. Here is what the two SLOs did:

| Metric | Before (overload) | After (raised `max_num_seqs`) |
|---|---|---|
| Queue-wait p99 | 340 ms | 107 ms |
| **TTFT p99** | 443 ms | **210 ms** |
| E2E p99 | 1,350 ms | 1,117 ms |
| GPU utilization | 61% | 88% |

Two things about this table are the whole lesson. First, the big win is in **TTFT** — the number users feel as "how long until it starts responding" — which more than halved, while E2E moved less because the decode term (896 ms) is inherent to the output length and no config change touches it. Raising concurrency did not "make the model faster"; it removed a wait. Second, GPU utilization went *up*, not down, which is the tell that the fleet was never capacity-starved — it was admission-starved. Adding GPUs would have made utilization *worse*. Only the trace could have told you that before you spent the money.

This is the recurring shape of LLM latency debugging: the fix is usually cheap and specific once the trace localizes the term, and expensive and useless when you guess. The rest of this post is the toolchain that gets you to the trace.

## Instrumenting the path with OpenTelemetry

A trace is only as complete as the spans you emit, and a span is only useful if it carries the same trace ID as its neighbors. **OpenTelemetry** (OTel) is the vendor-neutral standard that makes this work: it defines the span data model, the SDKs that create spans in your application code, the **W3C `traceparent`** header format that propagates a trace ID across a network hop, and the **OTLP** wire protocol that ships spans to a collector. The three moving parts you must get right are (1) create a span at every stage, (2) propagate the trace context across every service boundary so the spans join, and (3) export everything to one backend.

The critical and most commonly botched step is context propagation across the boundary between your gateway and your inference engine. If the gateway creates a span but does not inject the `traceparent` header into the request it forwards to vLLM, then vLLM's span gets a *fresh* trace ID and the two never join — you end up with two disconnected half-traces and no waterfall. The figure below shows the propagation done right: one `trace_id` is born at the client, carried through the gateway, injected across the hop into the engine, and the engine's child spans (queue, prefill, decode) all fan into a single collector so the whole request reconstructs as one trace.

![A dataflow graph showing one trace_id propagating from client through gateway into the vLLM engine, which emits queue, prefill, and decode child spans that all fan into a single OTLP collector](/imgs/blogs/tracing-and-debugging-llm-serving-2.webp)

Here is a FastAPI gateway instrumented to do exactly that. The two load-bearing lines are `FastAPIInstrumentor.instrument_app`, which creates a server span for every inbound request, and `propagate.inject(headers)`, which serializes the current trace context into the outbound headers so vLLM's span nests underneath.

```python
from fastapi import FastAPI, Request
from opentelemetry import trace, propagate
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
import httpx

# One provider per process; the resource tags every span with the service name
# so traces group correctly in Tempo/Jaeger.
resource = Resource.create({"service.name": "chat-gateway"})
provider = TracerProvider(resource=resource)
provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="otel-collector:4317", insecure=True))
)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

app = FastAPI()
FastAPIInstrumentor.instrument_app(app)             # server span per request
client = httpx.AsyncClient(base_url="http://vllm:8000")

@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    with tracer.start_as_current_span("gateway.forward") as span:
        span.set_attribute("gen_ai.request.model", body.get("model", ""))
        span.set_attribute("llm.messages", len(body.get("messages", [])))
        # Inject W3C traceparent so vLLM's engine span is a CHILD of this span.
        headers: dict[str, str] = {}
        propagate.inject(headers)                   # writes 'traceparent' (+ 'tracestate')
        resp = await client.post("/v1/chat/completions", json=body, headers=headers)
        span.set_attribute("http.status_code", resp.status_code)
        return resp.json()
```

The receiving side is vLLM's OpenAI-compatible server, which reads the incoming trace context from the request headers automatically once tracing is enabled — you do not have to extract it yourself. That gives you a parent-child relationship spanning two processes with no shared memory: the gateway span and the engine span are stitched purely by the propagated `trace_id` and `span_id`. Get the injection right and every downstream span lands in the same waterfall; forget it and you spend an afternoon wondering why your traces are truncated at the gateway.

A subtlety worth internalizing: propagation is a *convention*, not magic. The `traceparent` header is just a 55-character ASCII string like `00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01` — version, trace ID, parent span ID, and flags. Any hop that forwards it preserves the trace; any hop that drops it severs it. Load balancers, service meshes, and message queues all need to be checked for header passthrough. When a trace mysteriously ends at a service boundary, the first thing to check is whether that boundary forwards `traceparent`.

### What to put on a span (and what never to)

A span with no attributes is a stopwatch; a span with the *right* attributes is a diagnosis. The attributes that earn their place are the ones that let you *filter and bucket* traces later — the request shape and the latency decomposition — because filtering is how you turn a pile of traces into an answer. For LLM serving the high-value set is small and stable:

- **Request shape**: `gen_ai.request.model`, `gen_ai.usage.prompt_tokens`, `gen_ai.usage.completion_tokens`, `gen_ai.request.max_tokens`. These let you bucket by prompt/output length, which is the axis most latency correlates with. Token *counts*, never token *content*.
- **Latency decomposition**: the `gen_ai.latency.*` family, so the queue/prefill/decode split is queryable per request.
- **Sampling parameters** when relevant: `gen_ai.request.temperature`, `gen_ai.request.n`. A runaway `n=8` request is invisible unless `n` is on the span.
- **Routing keys**: which replica, which model version, which tenant/priority class — so you can slice a p99 by the dimension that actually varies.

What must *never* go on a span is the prompt or completion text itself. Three reasons, each sufficient on its own. First, **PII and compliance**: user prompts routinely contain names, emails, medical and legal detail, and trace stores are not built or governed as PII systems — putting prompt text in traces silently turns your observability stack into an uncontrolled data lake that your privacy review never approved. Second, **cost and cardinality**: prompts are large and unbounded; attaching them multiplies your trace storage by orders of magnitude and can blow up the collector, for data you will almost never read. Third, **it doesn't help**: you debug latency with the *shape* of a request (512 tokens in, 128 out, temperature 0), not its words, and on the rare occasion you need the content you should retrieve it from your application logs under their access controls, joined by `trace_id` — not duplicate it into traces. The discipline is: record the numbers that let you *find* the request, and keep the content where it is governed. A `record_shapes=True` on a profiler is fine because shapes are integers; a `span.set_attribute("prompt", text)` is a mistake you will be asked to explain in an incident review.

## Turning on vLLM's built-in request tracing

You could instrument the engine's internal stages by hand, but vLLM ships this out of the box. Passing `--otlp-traces-endpoint` turns on per-request tracing: vLLM emits one span per request, named `llm_request`, carrying attributes for the request parameters and — crucially — the *decomposed latency terms* you need. The span attributes map almost one-to-one onto the latency formula from earlier:

- `gen_ai.latency.time_in_queue` — the queue-wait term, `t_queue`
- `gen_ai.latency.time_to_first_token` — TTFT
- `gen_ai.latency.e2e` — the full end-to-end time
- `gen_ai.latency.time_in_scheduler`, `gen_ai.latency.time_in_model_forward`, `gen_ai.latency.time_in_model_execute` — finer decode/scheduler breakdown (enabled by `--collect-detailed-traces`)
- `gen_ai.usage.prompt_tokens`, `gen_ai.usage.completion_tokens` — the token counts that let you normalize latency per token

That means the queue-vs-prefill-vs-decode decomposition is a property you can *read directly off the span attributes* — no arithmetic across separate metrics required. Here is the launch:

```bash
# vLLM needs the OTel SDK + OTLP exporter installed to emit spans.
pip install \
  opentelemetry-sdk \
  opentelemetry-api \
  opentelemetry-exporter-otlp \
  opentelemetry-semantic-conventions-ai

# Name the service so traces group in the backend; disable TLS for an in-cluster collector.
export OTEL_SERVICE_NAME=vllm-llama2-13b
export OTEL_EXPORTER_OTLP_TRACES_INSECURE=true

vllm serve meta-llama/Llama-2-13b-chat-hf \
  --served-model-name llama2-13b \
  --otlp-traces-endpoint grpc://otel-collector:4317 \
  --collect-detailed-traces all \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.92
```

One caution on `--collect-detailed-traces all`: the detailed model-forward and model-execute timings require extra instrumentation inside the hot decode loop, and that instrumentation is not free — expect a few percent of throughput overhead. Turn it on when you are actively debugging a decode-side problem; leave it off (`--otlp-traces-endpoint` alone still gives you queue/TTFT/e2e) in steady-state production. This is the first appearance of the recurring tension in this post: instrumentation you can read is instrumentation you paid for. More on the overhead budget in the final section.

The collector sits between your services and your trace store. It receives OTLP spans, batches them, optionally samples them, and exports them onward. This is also where you implement **tail-based sampling** — the single most important cost control for tracing an LLM service — which decides whether to keep a trace *after* seeing its full duration, so you can keep every slow or errored request while sampling the boring fast ones:

```yaml
# otel-collector-config.yaml
receivers:
  otlp:
    protocols:
      grpc: { endpoint: 0.0.0.0:4317 }
      http: { endpoint: 0.0.0.0:4318 }

processors:
  batch:
    timeout: 5s
  # Tail sampling: keep every slow/errored trace, subsample the rest.
  tail_sampling:
    decision_wait: 10s
    policies:
      - name: keep-slow
        type: latency
        latency: { threshold_ms: 800 }        # every trace over the SLO is kept
      - name: keep-errors
        type: status_code
        status_code: { status_codes: [ERROR] }
      - name: baseline-sample
        type: probabilistic
        probabilistic: { sampling_percentage: 10 }

exporters:
  otlp/tempo:
    endpoint: tempo:4317
    tls: { insecure: true }

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [tail_sampling, batch]
      exporters: [otlp/tempo]
```

With that config, the 1.35 s exemplar from the opening incident is *guaranteed* to be in your trace store, because it exceeded the 800 ms latency policy — you are never in the position of having a p99 alert with no trace behind it. The fast 120 ms requests get sampled at 10% because they are interchangeable and you rarely need to inspect a specific one. This is the difference between a trace budget that scales with your slow-request rate (small) versus your total request rate (ruinous at scale).

## Instrumenting streaming responses: where TTFT actually lives

Almost every production LLM API streams tokens as they are generated over Server-Sent Events, and streaming quietly changes what "latency" even means to instrument. In a non-streaming request there is one response and one obvious place to stop the clock. In a streaming request the clock has *two* meaningful stops: the arrival of the **first** token, which is the TTFT the user experiences as "it started responding," and the arrival of the **last** token, which is E2E. TTFT is the number that dominates perceived responsiveness in a chat UI — a 2-second wait for the first word feels broken, while a 2-second *total* response that starts streaming in 200 ms feels instant — so it is the term you most need to instrument correctly, and streaming is exactly where naive instrumentation gets it wrong.

The trap is that vLLM's `time_to_first_token` measures TTFT *inside the engine* — from admission to the first token leaving the model. But the user's TTFT includes everything after that: the token crossing the network to the gateway, any proxy buffering, and the flush to the client. If a reverse proxy buffers the SSE stream (a distressingly common misconfiguration — nginx buffers by default unless you set `proxy_buffering off` and `X-Accel-Buffering: no`), the engine happily reports a 90 ms `time_to_first_token` while the user sees the first token 700 ms later, because the proxy held the stream until its buffer filled. The engine-side metric is green; the user is furious; and only an instrument that measures TTFT *at the gateway's output* reveals the gap.

So instrument the stream at the gateway, recording a span *event* the moment the first byte is flushed to the client. Do not create a span per token — a 500-token response would emit 500 spans and drown your collector; the first-token event plus the final duration is all you need:

```python
import time
from fastapi.responses import StreamingResponse

@app.post("/v1/chat/completions/stream")
async def chat_stream(request: Request):
    body = await request.json()
    body["stream"] = True
    span = tracer.start_span("gateway.stream")
    headers: dict[str, str] = {}
    propagate.inject(headers)                 # nest vLLM's span under this one
    t0 = time.perf_counter()

    async def relay():
        first = True
        async with client.stream("POST", "/v1/chat/completions",
                                  json=body, headers=headers) as resp:
            async for chunk in resp.aiter_bytes():
                if first:
                    # gateway-observed TTFT: what the USER actually waited for.
                    span.add_event("first_token",
                                   {"gateway.ttft_ms": (time.perf_counter() - t0) * 1000})
                    first = False
                yield chunk
        span.set_attribute("gateway.e2e_ms", (time.perf_counter() - t0) * 1000)
        span.end()

    # Disable proxy buffering so the first token is flushed immediately.
    return StreamingResponse(relay(), media_type="text/event-stream",
                             headers={"X-Accel-Buffering": "no",
                                      "Cache-Control": "no-cache"})
```

Now every streamed request carries two TTFT numbers on the same trace: `gen_ai.latency.time_to_first_token` from the engine span and `gateway.ttft_ms` from the gateway span event. Their *difference* is a term the raw metrics cannot give you — the network-plus-buffering tax on TTFT. If that difference is a stable 5–15 ms, your streaming path is clean. If it is 300+ ms, or it is bimodal, you have a buffering or backpressure problem in the proxy layer, not the engine, and no amount of GPU work will fix it.

#### Worked example: the TTFT gap that was nginx buffering

A team shipped a new ingress and TTFT p99 (as measured by synthetic clients) jumped from 250 ms to 900 ms overnight, while vLLM's `time_to_first_token` p99 sat unchanged at 180 ms. The two numbers disagreeing *was* the diagnosis: the engine was fine, so the 720 ms lived between the engine and the client. Comparing `gateway.ttft_ms` (880 ms) to the engine's `time_to_first_token` (180 ms) localized it to the ingress, and the new nginx config had reverted `proxy_buffering` to `on`, so SSE tokens were buffered until the 4 KB proxy buffer filled or the response ended. Setting `proxy_buffering off` restored the gateway TTFT to ~195 ms. The trace was decisive because it carried *both* clocks; a single TTFT metric — whichever one you happened to alert on — would have sent you to debug the wrong layer.

## Reading a trace: from waterfall to root cause

Instrumentation gets spans into a store; the skill is reading them. A trace store like Grafana Tempo or Jaeger lets you search for the traces behind a bad percentile and open the waterfall. The query language is where most people stall, so here is the concrete workflow.

To find the slow traces in Tempo, you write **TraceQL** that filters on the vLLM span attributes directly. The power move is to filter on the *decomposed* term, not just total duration — that way you retrieve exactly the traces where the term you suspect is abnormal:

```bash
# All llm_request traces where the QUEUE term alone exceeded 300 ms.
curl -s "http://tempo:3200/api/search" \
  --data-urlencode 'q={ name="llm_request" && span.gen_ai.latency.time_in_queue > 0.3 }' \
  --data-urlencode 'limit=20' | jq '.traces[].traceID'

# Compare: traces slow overall but with SMALL queue — the decode/prefill suspects.
curl -s "http://tempo:3200/api/search" \
  --data-urlencode 'q={ name="llm_request" && duration > 1s && span.gen_ai.latency.time_in_queue < 0.05 }' \
  --data-urlencode 'limit=20' | jq '.traces[].traceID'
```

Those two queries *are* the diagnosis. If the first returns many traces and the second returns few, your slowness is queue wait. If it's the other way around, look at prefill and decode. You have partitioned the problem space with two HTTP calls. In Jaeger the equivalent is the search API with `minDuration` plus a tag filter:

```bash
curl -s "http://jaeger:16686/api/traces?service=vllm-llama2-13b&minDuration=1s&limit=20" \
  | jq '.data[].spans[] | select(.operationName=="llm_request")
        | {trace: .traceID,
           queue:  (.tags[]|select(.key=="gen_ai.latency.time_in_queue").value),
           ttft:   (.tags[]|select(.key=="gen_ai.latency.time_to_first_token").value),
           e2e:    (.tags[]|select(.key=="gen_ai.latency.e2e").value)}'
```

Once you can pull the terms programmatically, you can stop eyeballing individual traces and compute the *budget distribution* — the p50 and p99 of each term across your slow traces, and each term's average share of E2E. This is the script I reach for first in an incident, because it answers "which term is the p99 coming from" in one shot:

```python
# latency_budget.py — decompose the slowest traces into queue/prefill/decode terms.
import requests

TEMPO = "http://tempo:3200"
QUERY = '{ name="llm_request" && duration > 800ms }'

def iter_spans(trace_json):
    for batch in trace_json.get("batches", []):
        for ss in batch.get("scopeSpans", []):
            yield from ss.get("spans", [])

def attrs(span):
    out = {}
    for kv in span.get("attributes", []):
        v = kv["value"]
        out[kv["key"]] = float(v.get("doubleValue", v.get("intValue", 0)))
    return out

ids = [t["traceID"] for t in requests.get(
    f"{TEMPO}/api/search",
    params={"q": QUERY, "limit": 200}).json().get("traces", [])]

terms = {"queue": [], "prefill": [], "decode": [], "e2e": []}
for tid in ids:
    tj = requests.get(f"{TEMPO}/api/traces/{tid}").json()
    for span in iter_spans(tj):
        a = attrs(span)
        if "gen_ai.latency.e2e" not in a:
            continue
        q, ttft, e2e = (a["gen_ai.latency.time_in_queue"],
                        a["gen_ai.latency.time_to_first_token"],
                        a["gen_ai.latency.e2e"])
        terms["queue"].append(q)
        terms["prefill"].append(max(ttft - q, 0.0))   # TTFT minus queue ≈ prefill
        terms["decode"].append(max(e2e - ttft, 0.0))   # E2E minus TTFT = decode
        terms["e2e"].append(e2e)

def pct(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(len(xs) * p))]

e2e_mean = sum(terms["e2e"]) / len(terms["e2e"])
for name, xs in terms.items():
    print(f"{name:8s} p50={pct(xs,.5)*1000:7.1f}ms  p99={pct(xs,.99)*1000:7.1f}ms  "
          f"share={sum(xs)/len(xs)/e2e_mean*100:5.1f}%")
```

Run against the incident window it printed:

```
queue    p50=  38.0ms  p99= 341.0ms  share= 25.9%
prefill  p50=  92.0ms  p99=  98.0ms  share=  7.1%
decode   p50= 880.0ms  p99= 905.0ms  share= 66.5%
e2e      p50=1012.0ms  p99=1351.0ms  share=100.0%
```

Read this table like a detective. Decode is the biggest *average* share (66.5%) but its p99 is barely above its p50 (905 vs 880) — it is large but *stable*, so it is not the source of the *spike*. Queue is a smaller average share but its p99 (341 ms) is nine times its p50 (38 ms) — it is the term that *blows up at the tail*. The p99 spike is a queue-wait spike wearing a decode-sized coat. This is the exact insight the raw metric could not give you, and it took one script.

#### Worked example: the TPOT climb that was a growing batch

A different incident, same tools. TTFT was healthy but users complained tokens were "stuttering." The budget script showed decode p99 had crept from 900 ms to 1,600 ms over a week while output lengths were unchanged. Pulling `gen_ai.latency.e2e / gen_ai.usage.completion_tokens` per trace gave TPOT, and TPOT had risen from 7 ms to 12.5 ms. TPOT is a per-step memory-bandwidth quantity; it rises when each decode step has *more to read*, which for a fixed model means a bigger batch (more sequences' KV caches read per step) or longer contexts (bigger KV caches). Cross-referencing the vLLM metric `vllm:num_requests_running` confirmed the running batch had grown from ~40 to ~110 as traffic climbed, pushing decode from memory-bound-with-headroom toward memory-bandwidth-saturated. The fix was to cap `max_num_seqs` at 96 and enable FP8 KV cache to halve the bytes read per step. TPOT fell back to 8 ms. Note the shape: **a decode-term regression traced to batch growth, fixed by bounding the batch and shrinking the KV bytes** — a completely different playbook from the queue-wait incident, selected by the trace.

## Correlating the three pillars: metrics, traces, logs

Traces are one of the three classic observability pillars — metrics, traces, logs — and their real power appears only when the three are *correlated*, so you can jump from an aggregate alert to the specific slow request to the log lines that request emitted, without manually grepping by timestamp. The connective tissue is IDs: an **exemplar** attaches a `trace_id` to a metric bucket, and a shared **`span_id`** lets you join structured logs to the span that produced them. The figure below is the drill-down path: a p99 metric alert carries an exemplar into the exact slow trace, the trace's span IDs join to the request's logs, and the hottest span opens a GPU profile — four hops from "something is wrong" to "here is the named cause."

![A layered drill-down stack: a p99 metric alert hops via an exemplar to a trace waterfall, the trace joins to logs on span_id, a profile opens the hottest span, and the root cause is found](/imgs/blogs/tracing-and-debugging-llm-serving-3.webp)

Start at the metric. vLLM exposes a rich Prometheus surface, and the histograms you care about map directly onto the latency terms:

- `vllm:time_to_first_token_seconds` — TTFT histogram
- `vllm:time_per_output_token_seconds` — TPOT histogram
- `vllm:e2e_request_latency_seconds` — end-to-end histogram
- `vllm:request_queue_time_seconds` — the queue term as its own histogram
- `vllm:num_requests_running`, `vllm:num_requests_waiting` — live batch and queue depth
- `vllm:num_preemptions_total` — how often the scheduler evicted a running sequence

The p99 TTFT query, with exemplars enabled so each bucket carries a link to a real trace:

```promql
histogram_quantile(
  0.99,
  sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m]))
)
```

In Grafana, with exemplars turned on for that panel and Tempo wired as the trace data source, the spike on this graph renders little diamonds you click to jump *straight into the slow trace* — no timestamp hunting, no guessing which of ten thousand requests was the p99. That is the metric→trace hop. The trace→log hop is the reverse of the propagation you set up earlier: emit `trace_id` and `span_id` into every structured log line, so a log query can retrieve exactly the lines a given trace produced.

```python
import logging, json
from opentelemetry import trace

class TraceContextFilter(logging.Filter):
    """Stamp every log record with the active trace_id/span_id for join-on-id."""
    def filter(self, record):
        ctx = trace.get_current_span().get_span_context()
        record.trace_id = format(ctx.trace_id, "032x") if ctx.trace_id else "-"
        record.span_id = format(ctx.span_id, "016x") if ctx.span_id else "-"
        return True

handler = logging.StreamHandler()
handler.addFilter(TraceContextFilter())
handler.setFormatter(logging.Formatter(
    json.dumps({"ts": "%(asctime)s", "level": "%(levelname)s",
                "msg": "%(message)s",
                "trace_id": "%(trace_id)s", "span_id": "%(span_id)s"})))
logging.getLogger().addHandler(handler)
```

Now the loop closes. A metric alert (TTFT p99) carries an exemplar to a trace; the trace's waterfall localizes the slow span; that span's `trace_id` filters your logs (`{trace_id="4bf92f35..."}`) to the handful of lines that request emitted — including the request parameters, any retries, any sampling warnings. If the slow span is a decode span and the logs are clean, you have exhausted what application-level observability can tell you, and the next hop is *below* the application: into the GPU. For a broader treatment of when to reach for each of these three pillars, the SRE post on [metrics, logs, and traces: when to use which](/blog/software-development/site-reliability-engineering/metrics-logs-and-traces-when-to-use-which) is the companion to this section; the point here is narrower — for LLM serving, correlate them by ID or you will spend the incident grepping.

#### Worked example: one request at 9 seconds — a runaway stop token

The most instructive correlation case is a single pathological request. An alert fired on a max-latency panel: one request took 9.1 s while p99 sat at a healthy 400 ms. The exemplar led to a trace whose `gen_ai.usage.completion_tokens` was **1,024** — the configured `max_tokens` — while every other request in the window generated 40–120. The decode span was 9 s because the model generated the full 1,024-token cap and never emitted a stop token. Joining on `trace_id`, the logs showed the request had omitted the model's chat template, so the expected end-of-turn token was never produced and generation ran to the hard cap. The fix was a gateway validation that always applies the template and a defensive `max_tokens` ceiling. The signature of this class — **one request slow, p99 unaffected, `completion_tokens` pinned at the cap** — is unmistakable once you can pull per-request token counts off the trace, and invisible in any aggregate.

## GPU-level tracing: when the model itself is slow

Distributed tracing localizes latency to a *stage*. When that stage is prefill or decode and the engine config is already sane, you have to go a level deeper, into the GPU kernels the stage launches, because the question is no longer "which span" but "which kernel, and why is the GPU idle between them." This is where **Nsight Systems** and **`torch.profiler`** come in — they give you a timeline of every CUDA kernel, its duration, and, most importantly, the *gaps* between kernels where the streaming multiprocessors sit idle waiting on memory.

The figure below is a decode step at kernel scale. The striking feature is what the numbers say about occupancy: the QKV and MLP GEMMs run at *low* SM utilization and the PagedAttention kernel is explicitly HBM-bound. The GPU is not compute-limited during decode — it is spending most of the step *reading weights and KV cache from memory* while the arithmetic units wait. That is the visual confirmation of the memory-bound decode math from earlier, and it dictates the fix vocabulary: you improve decode TPOT by feeding it *bandwidth* (bigger batches to amortize the weight read, quantized weights/KV to shrink the bytes, better kernel fusion to cut launch overhead) — never by adding raw FLOPs, which sit idle.

![A kernel-level timeline of one decode step under Nsight, showing RMSNorm, QKV GEMM, PagedAttention, NCCL all-reduce, MLP GEMM, and sampling kernels at low SM occupancy, summing to 7 ms per token](/imgs/blogs/tracing-and-debugging-llm-serving-7.webp)

vLLM has a native profiler hook that produces exactly this timeline. Set `VLLM_TORCH_PROFILER_DIR`, then bracket the work you want to capture with `start_profile()` and `stop_profile()`; it writes a Perfetto/Chrome trace you open in `chrome://tracing` or the Perfetto UI:

```python
import os
os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile"   # must be set BEFORE engine init

from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-13b-chat-hf",
          enforce_eager=True)          # eager mode => real per-op kernel names in the trace
prompt = ["Summarize the contract:\n" + "clause " * 400]

llm.start_profile()
llm.generate(prompt, SamplingParams(max_tokens=32, temperature=0.0))
llm.stop_profile()                     # flushes the trace to VLLM_TORCH_PROFILER_DIR
```

For a standalone module (a custom attention op, a reranker, a non-vLLM model), the raw `torch.profiler` API gives you the same timeline plus a summary table sorted by GPU time — the fastest way to find the single kernel eating your step:

```python
import torch
from torch.profiler import profile, schedule, ProfilerActivity, tensorboard_trace_handler

# wait/warmup/active lets the allocator and caches settle before you record.
sched = schedule(wait=1, warmup=2, active=3, repeat=1)
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=sched,
    on_trace_ready=tensorboard_trace_handler("./trace_decode"),
    record_shapes=True, profile_memory=True, with_stack=True,
) as prof:
    for _ in range(6):
        run_one_decode_step(model, kv_cache)   # your module's single-step forward
        prof.step()

# The table that answers "which kernel is my step?"
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))
```

When you need the whole-system picture — CPU-side launch overhead, CUDA kernels, NCCL communication for tensor parallelism, and GPU hardware counters like memory-throughput on one timeline — Nsight Systems is the tool. Wrap the region of interest with the CUDA profiler range so you capture one clean decode step instead of the whole run:

```bash
nsys profile \
  --trace=cuda,nvtx,cublas,cudnn \
  --gpu-metrics-device=all \
  --capture-range=cudaProfilerApi \
  --output=decode_step --force-overwrite=true \
  python bench_one_decode.py

# Summarize kernel time from the CLI without opening the GUI.
nsys stats --report cuda_gpu_kern_sum decode_step.nsys-rep | head -20
```

The `--gpu-metrics-device=all` flag is the one that earns its keep here: it samples hardware counters so you can read *memory throughput as a percentage of peak* directly off the timeline. If PagedAttention is running while HBM throughput reads 85% of peak and SM occupancy reads 30%, you have proven the decode step is bandwidth-bound and no kernel rewrite that only reduces FLOPs will help. That is the difference between profiling and guessing at the kernel level.

#### Worked example: reading the kernel trace of a decode step

On the 13B model, a single decode step's Nsight timeline broke down (representative, per transformer layer, at batch 1): RMSNorm ~5 µs, QKV projection GEMM 42 µs at 28% SM occupancy, PagedAttention 58 µs with HBM throughput pinned near peak, the tensor-parallel all-reduce 31 µs on NCCL, the MLP GEMM 124 µs at 31% SM occupancy, sampling 9 µs. Summed across all layers plus fixed overhead, the step landed near 7 ms/token — matching the memory-bound TPOT floor from the bandwidth math. The read is unambiguous: the two GEMMs dominate wall time yet run at ~30% occupancy, and attention is memory-throughput-limited, so the GPU is *memory-starved, not compute-starved*. The productive levers are all bandwidth-side. This is the same conclusion the arithmetic gave us in the mechanics section, now confirmed against the hardware counters rather than asserted — which is exactly the confidence you want before spending a week on a kernel change. For the deeper theory of why decode is memory-bound and what to do about it, see [why LLM serving is different](/blog/machine-learning/model-serving/why-llm-serving-is-different).

### Reading the gaps, not just the kernels

The single most valuable habit when reading a GPU timeline is to look at the *empty space between kernels*, not only the kernels themselves. A decode step that is 7 ms of wall time but only 4 ms of summed kernel duration is telling you something: 3 ms of every step is the GPU sitting idle, and idle GPU is latency you are paying for and getting nothing back. There are two common shapes of that idle time, and they have opposite fixes.

The first is **launch-bound** decode: many tiny gaps, one before each kernel, because the CPU cannot enqueue kernels fast enough to keep the GPU fed. In eager mode a decode step launches dozens of small kernels per layer, each with microseconds of Python-and-CUDA launch overhead, and at small batch those overheads sum to a large fraction of the step. The Nsight signature is a "picket fence" — kernel, small gap, kernel, small gap — and the CUDA API row on the timeline is nearly saturated while the GPU rows are sparse. The fix is not a faster kernel; it is *fewer launches*: CUDA graphs (vLLM captures the decode step into a graph by default, replaying it as one launch), `torch.compile`, or kernel fusion. Turning `enforce_eager` off in the profiling snippet above and re-capturing collapses the picket fence — which is also why you profile in eager mode to *see* the kernels but run in graph mode in production to *not pay* for them.

The second is **communication-bound** decode in a tensor-parallel deployment: one large gap per layer where the SMs wait on a NCCL all-reduce to synchronize partial results across GPUs. The Nsight signature is a wide gap aligned with an NCCL kernel on the communication stream. This is why tensor parallelism has diminishing returns past a point — you buy more memory bandwidth but pay a fixed per-layer synchronization tax that grows with the number of shards. The fix vocabulary here is "fewer, bigger GPUs over more, smaller ones," "NVLink over PCIe for the all-reduce," or a different parallelism split. Crucially, distributed tracing extends *into* this regime: each TP worker is its own process, so with propagation set up correctly a single request's trace can span all the workers, and you can see whether one shard is consistently the straggler that everyone else waits on.

## A debugging decision tree for common symptoms

You now have every tool — span waterfalls, budget scripts, pillar correlation, kernel profiles. The remaining skill is *sequencing* them, and the sequence is not linear: it branches on the symptom. The decision tree below encodes the first and most valuable split — *which latency term is abnormal* — and then the sub-split on *which span dominates that term*, landing on a concrete fix in at most three questions. Walk it top to bottom during an incident instead of reaching for whatever tool is nearest.

![A latency-symptom decision tree: split on whether TTFT, TPOT, or both are abnormal, then on which span dominates, ending on concrete fixes like raising max-num-seqs or chunked prefill](/imgs/blogs/tracing-and-debugging-llm-serving-5.webp)

The root question is always *read the trace first* — never start with a hypothesis, start with the decomposition. From there:

**High TTFT, normal TPOT.** The problem is in `gateway + queue + prefill`, and the sub-split is which of those dominates. Open the trace and compare the queue-wait span to the prefill span. If `time_in_queue` dominates, you are admission-limited: raise `max_num_seqs`, add admission control, or autoscale — the opening incident. If the prefill span dominates, you have long prompts saturating compute: enable chunked prefill so a giant prompt is sliced into bounded chunks that interleave with other requests' decode steps (rather than monopolizing the GPU for one 180 ms forward pass), turn on prefix caching if prompts share a common prefix, and cap prompt length at the gateway.

**High TPOT, normal TTFT.** The problem is the decode loop. TPOT is a memory-bandwidth quantity, so it climbs when each decode step reads more bytes: the batch grew (more KV caches read per step) or contexts got longer (bigger KV caches). Check `vllm:num_requests_running` and the average context length. Fixes: cap `max_num_seqs`, quantize the KV cache to FP8 to halve the bytes read, or shorten outputs. This was the second worked example.

**Both TTFT and TPOT spike together.** This is the overload signature, and it usually means **preemption**: the scheduler ran out of KV-cache blocks and evicted running sequences to admit or continue others, so requests pay both a queue-wait tax (TTFT up) and a recompute/swap tax (TPOT up). Check `vllm:num_preemptions_total` — if it is nonzero and rising, you are here. Fixes are load-side, not config-tuning: admission control to shed load, autoscaling to add capacity, or lowering concurrency so the working set fits in KV memory. The mechanics of this eviction are the subject of [request scheduling and preemption](/blog/machine-learning/model-serving/request-scheduling-and-preemption); the tracing signal that you are *in* it is the joint TTFT+TPOT spike with a rising preemption counter.

**One request slow, p99 unaffected.** Not a systemic problem — a single pathological request. Pull its per-span attributes: `completion_tokens` pinned at the `max_tokens` cap points to a missing stop token or chat template (the 9-second worked example); an enormous prompt points to an un-capped input; unusual sampling parameters (huge `n`, extreme `top_p`) point to a client misuse. The fix is input validation and defensive caps at the gateway, not anything in the engine.

#### Worked example: the sawtooth that was preemption

A dashboard showed TTFT p99 oscillating on a roughly 90-second period — smooth for a minute, a sharp spike, smooth again — a sawtooth that no single request could explain. The budget script showed the spikes were joint: TTFT *and* TPOT both jumped at each tooth. Overlaying `vllm:num_preemptions_total`, the preemption counter stepped up by ~30 at exactly each spike. The story reconstructed cleanly from there: traffic was riding just above the KV-cache budget, so the scheduler periodically ran out of blocks, preempted a batch of running sequences (evicting or swapping their KV caches), and then had to recompute them — a burst that stalled admission (TTFT up) and slowed the surviving batch (TPOT up) until the pressure cleared, then repeated. This is the textbook overload signature, and it is *invisible* as anything but noise in an averaged latency graph; the trace-plus-metric correlation — joint spike aligned with a preemption step — is what named it. The fix was load-side: an admission-control cap on concurrent sequences so the working set fit in KV memory, converting a sawtooth of preemptions into a slightly longer but *stable* queue. The mechanism is covered in depth in [request scheduling and preemption](/blog/machine-learning/model-serving/request-scheduling-and-preemption); the point here is that the diagnostic signal is a correlation, not a single number.

## The on-call runbook: symptom → cause → fix

The decision tree is how you *reason*; the matrix below is how you *look things up* at 3 a.m. Each row is a symptom, and reads left to right into the likely cause, the exact span or metric to open, and the first fix to try. Print it, pin it in the runbook, and hand it to whoever is on call.

![A symptom-to-cause-to-fix matrix: four latency symptoms each mapped to a likely cause, the specific span or metric to open, and a first fix](/imgs/blogs/tracing-and-debugging-llm-serving-6.webp)

In text, so it is greppable and copy-pasteable:

| Symptom | Likely cause | Where to look (span / metric) | First fix |
|---|---|---|---|
| High TTFT, normal TPOT | Queue wait or long prefill | `gen_ai.latency.time_in_queue` span; `vllm:num_requests_waiting`; prefill span vs `prompt_tokens` | Raise `max_num_seqs`; chunked prefill; cap prompt length |
| High TPOT, normal TTFT | Batch too big / KV cache reads saturate HBM | `time_per_output_token` metric; `vllm:num_requests_running`; GPU mem-bw % in Nsight | Cap `max_num_seqs`; FP8 KV cache; shorter outputs |
| TTFT and TPOT both spike | Overload → preemption / KV eviction | `vllm:num_preemptions_total`; swap/recompute events in logs | Admission control; autoscale; lower concurrency |
| One request slow, p99 flat | Tokenization / sampling / runaway generation | Per-span `completion_tokens`, `prompt_tokens`, sampling attrs; join logs on `trace_id` | Cap `max_tokens`; enforce chat template; validate input |
| TTFT sawtooth (periodic) | Cold cache after eviction; autoscale churn | Correlate TTFT spikes with `num_preemptions`; pod-restart events | Stabilize replicas; warm prefix cache; tune HPA cooldown |

The value of a runbook is that it converts triage from *investigation* (slow, error-prone, seniority-dependent) into *lookup* (fast, consistent, anyone can do it). The trace is what makes the lookup possible: every "where to look" column points at a span attribute or a metric that exists only because you instrumented the path. For the full incident-response version of this table — including the OOM, the bad-deploy, and the network-partition cases that go beyond latency — see the sibling [troubleshooting LLM serving runbook](/blog/machine-learning/model-serving/troubleshooting-llm-serving-runbook), and for the dashboards and alerts that feed it, the sibling on [observability for LLM serving](/blog/machine-learning/model-serving/observability-for-llm-serving).

## Reproducing production latency locally

The last skill is closing the loop: reproducing a production latency signature on a machine you can profile freely, so you can iterate on a fix without experimenting in prod. The trick is that a trace does not just tell you *what was slow* — it tells you the *shape* of the request that was slow, and shape is reproducible. From the slow trace you extract the load parameters: prompt length, output length, concurrency, and the arrival pattern. Then you replay that shape locally and profile it.

```python
# replay_shape.py — reproduce a production latency signature from trace-derived params.
import asyncio, time, httpx

# These four numbers come straight off the slow trace + its neighbors:
#   prompt_tokens, completion_tokens (max_tokens), concurrency, arrival gap.
PROMPT_TOKENS   = 512
OUTPUT_TOKENS   = 128
CONCURRENCY     = 64           # matches vllm:num_requests_running at the incident
BASE_URL        = "http://localhost:8000"

PROMPT = "word " * PROMPT_TOKENS   # length is what matters, not content

async def one(client, results):
    t0 = time.perf_counter()
    r = await client.post("/v1/completions", json={
        "model": "llama2-13b", "prompt": PROMPT,
        "max_tokens": OUTPUT_TOKENS, "temperature": 0.0, "stream": False})
    results.append((time.perf_counter() - t0, r.status_code))

async def main():
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60) as client:
        results = []
        # Sustain CONCURRENCY in flight to recreate the queue depth.
        sem = asyncio.Semaphore(CONCURRENCY)
        async def guarded():
            async with sem:
                await one(client, results)
        await asyncio.gather(*(guarded() for _ in range(CONCURRENCY * 8)))
        lat = sorted(x[0] for x in results)
        print(f"n={len(lat)} p50={lat[len(lat)//2]*1000:.0f}ms "
              f"p99={lat[int(len(lat)*0.99)-1]*1000:.0f}ms")

asyncio.run(main())
```

Run this against a local vLLM with the *same* `max_num_seqs` as prod, and you should see a similar queue-wait signature — a p99 far above p50 driven by requests stacking behind the concurrency cap. Now you have a reproduction: turn on `--collect-detailed-traces all` and the vLLM profiler, bisect the config (raise `max_num_seqs`, enable chunked prefill, quantize the KV cache), and watch the local p99 move *before* you touch production. The general capacity-planning and load-testing version of this replay loop — closed-loop vs open-loop load models, arrival distributions, saturation points — is its own discipline; here the point is narrow: the trace hands you the exact request shape to reproduce, which is 90% of the work of reproducing a latency bug.

One caveat on local reproduction: hardware matters. A latency signature captured on an H100 will not reproduce faithfully on an A100 or an L4, because the memory-bound decode floor scales with HBM bandwidth (H100 3.35 TB/s vs A100 2.0 TB/s vs L4 0.3 TB/s). If you cannot get the same GPU, reproduce the *structure* (queue behavior, preemption) rather than the absolute numbers, and reason about the decode term from the bandwidth ratio.

A second caveat is the load model, and getting it wrong is the most common way a reproduction lies to you. The script above is **closed-loop**: it keeps a fixed number of requests in flight, launching a new one only when an old one finishes. Closed-loop load has a built-in safety valve — when the server slows, the client naturally backs off, so you can never actually overload it, and queue wait plateaus instead of exploding. Real user traffic is **open-loop**: requests arrive on their own schedule (roughly Poisson) regardless of whether the server is keeping up, so a slowdown causes a pileup that feeds back into worse slowdown. If your production incident was a queue-wait explosion — the nonlinear knee from the mechanics section — a closed-loop reproduction will *never* show it, because closed-loop cannot cross the knee. To reproduce an overload you must drive arrivals open-loop at a fixed rate (e.g. with a Poisson inter-arrival generator, or a tool like `vegeta`/`k6` in constant-arrival-rate mode) and let the queue build. Match the load model to the incident: closed-loop to reproduce a steady-state latency budget, open-loop to reproduce a saturation or preemption event.

Finally, warm up before you measure. The first requests after an engine starts pay CUDA-graph capture, cache population, and allocator growth costs that have nothing to do with steady-state latency; discard them. If prefix caching is in play, the first request with a given prefix pays full prefill while subsequent ones hit the cache — so a reproduction that sends the *same* prompt every time will show an artificially low prefill after the first call, masking the very prefill cost you were trying to study. Vary the prompt prefix if you want to measure cold prefill, or repeat it if you are studying cache hits — and know which one your incident was.

## Case studies

**vLLM's per-request OpenTelemetry tracing.** vLLM added first-class OTel tracing that emits one `llm_request` span per request with the decomposed latency attributes (`time_in_queue`, `time_to_first_token`, `e2e`, and the detailed model-forward/execute timings under `--collect-detailed-traces`). This is the reference implementation of the pattern in this post: the engine, which is the only component that knows when a request left the queue and when prefill ended, is the one that emits those spans, and it propagates context in from the gateway via the standard headers. The lesson is architectural — put the span boundaries where the timing knowledge is, not where it is convenient. (Source: vLLM documentation, "OpenTelemetry" / observability section.)

**Tail-based sampling for slow-request retention.** The OpenTelemetry Collector's `tail_sampling` processor is the standard way large services keep every interesting trace while sampling the rest, because it makes the keep/drop decision *after* seeing the full trace duration. For an LLM service where the slow tail is the entire point of tracing, a latency policy at your SLO threshold plus an error policy guarantees the exemplar behind any p99 alert is retained, while a low baseline probability keeps the fast majority cheap. The trade-off is memory in the collector (it must buffer spans for `decision_wait` before deciding) and a small delay before traces appear. (Source: OpenTelemetry Collector `tailsamplingprocessor` documentation.)

**Chunked prefill and the prefill-monopolization pattern.** The failure mode where one long prompt freezes every other request's decode — the 28,000-token prompt that stalls the fleet for one giant forward pass — was characterized in the SARATHI line of work (Agrawal et al., 2023) and productionized as chunked prefill in vLLM. Tracing is how you *detect* it: a trace where the prefill span is enormous while concurrent requests show a TPOT spike at exactly that timestamp. The fix (slice prefill into token-bounded chunks that interleave with decode) is only reachable once the trace has told you prefill monopolization is happening rather than a generic slowdown. (Source: Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills," 2023; vLLM chunked-prefill docs.)

**Grafana exemplars for metric-to-trace correlation.** The exemplar mechanism — attaching a `trace_id` to a Prometheus histogram bucket so a dashboard spike links directly to a representative trace — is documented and widely deployed in the Grafana/Prometheus/Tempo stack. In practice it is the single feature that most compresses time-to-diagnosis, because it eliminates the "which of ten thousand requests was the p99" search that otherwise eats the first ten minutes of every latency incident. (Source: Prometheus exemplars specification; Grafana Tempo documentation.)

**CUDA graphs and the launch-bound decode step.** vLLM captures the decode step into a CUDA graph by default and replays it as a single launch, precisely because eager per-kernel launches leave the GPU launch-bound at the small batch sizes typical of latency-sensitive serving. The profiling lesson is a methodological one that catches people out: you must profile with `enforce_eager=True` to see individual kernel names, but that same flag *disables* the graph capture that production relies on — so an eager profile will show a launch-bound picket fence that does not exist in your graph-mode production, and you can waste a day optimizing a problem you do not have. Always confirm whether the decode step in production is captured into a graph before concluding it is launch-bound. (Source: vLLM documentation on CUDA graph capture and the `enforce_eager` flag.)

**Prefix-cache hit rate read off the traces.** vLLM's automatic prefix caching reuses the KV cache for shared prompt prefixes (system prompts, few-shot exemplars, long shared documents), turning what would be a full prefill into a near-instant cache hit. Because the prefill term is right there on the trace, you can *measure your effective prefix-cache hit rate from traces alone*: bucket requests by prefill duration normalized to prompt length, and the bimodal split — a cluster at full-prefill cost and a cluster near-zero — is your miss/hit population. A latency-debugging session that finds TTFT regressed can often be traced to a *drop in hit rate* (a prompt-template change that broke the shared prefix, or cache eviction under memory pressure) rather than any change in raw compute. This is a case where the trace measures a property — cache effectiveness — that has no dedicated metric of its own. (Source: vLLM automatic prefix caching documentation; RadixAttention, Zheng et al., 2023.)

## When to use this (and when not to)

Tracing is not free, and treating it as always-on-everything is how teams end up with a trace bill larger than their inference bill. The costs are real and worth naming.

**Per-span overhead.** Creating and exporting spans costs CPU and a little latency. For coarse spans (gateway, engine-level `llm_request`) this is negligible — well under 1% of request time. For fine-grained spans inside the decode loop (`--collect-detailed-traces all`), the overhead is a few percent of throughput because you are instrumenting a hot path that runs once per token. Rule: keep coarse tracing always-on; enable detailed decode-loop tracing only while actively debugging a decode-side problem, then turn it back off.

Do not take those numbers on faith — measure them for your own service, because "a few percent" is a claim you can and should verify with the same load-test harness you use for everything else. The measurement is a simple A/B: run the reproduction load at a fixed arrival rate with tracing off, record throughput (tokens/s) and TTFT/TPOT p99; flip tracing on at your intended granularity and sampling; re-run identically; diff. A representative result on the 13B model at ~40 concurrent sequences: coarse `llm_request` tracing with tail sampling cost under 0.5% throughput and no measurable p99 change — free, in practice. Detailed traces (`--collect-detailed-traces all`) cost ~3% throughput and added ~4 ms to TPOT p99, because the per-token instrumentation lands squarely in the decode hot path. Always-on kernel profiling with `torch.profiler` is the expensive one — 15–30% overhead — which is exactly why it belongs in staging and short debugging windows, never steady-state production. The point of measuring is that these numbers depend on your model size, batch, and collector, and a decision to keep detailed tracing on "because it's cheap" should rest on a diff you ran, not a blog post's estimate — including this one.

**Sampling cost vs coverage.** Storing every trace at high QPS is ruinous — a service at 5,000 QPS generates hundreds of millions of traces a day. But *head* sampling (decide at request start) will, by construction, miss most of your slow tail, because slowness is not known at the start. This is the case *for* tail-based sampling: it costs collector memory and a few seconds of delay, but it keeps 100% of the traces you actually need (slow, errored) and samples the interchangeable fast ones. Use head sampling only if your collector cannot afford the tail-sampling buffer, and accept that you will sometimes lack the trace behind a spike.

Here is the decision, condensed:

| Situation | Trace granularity | Sampling | Why |
|---|---|---|---|
| Steady-state production | Coarse (gateway + `llm_request`) | Tail-based, keep-slow + keep-error + 5–10% baseline | Full tail coverage at bounded cost |
| Active incident / debugging | Detailed (`--collect-detailed-traces all`) + kernel profiles on demand | Keep everything for the affected route | You need the decode/kernel breakdown, overhead is acceptable short-term |
| Very high QPS, tight budget | Coarse only | Head sampling + always-keep errors | Collector cannot buffer for tail sampling |
| Load test / staging | Detailed + always-on profiling | Keep everything | Not user-facing; profile freely |

**When tracing is the wrong tool entirely.** If your problem is a *throughput* ceiling rather than a *latency* tail — you want more tokens/s per GPU, not a faster individual request — tracing tells you less than a saturation load test and a kernel roofline analysis. Tracing localizes *where a request spends time*; it does not tell you the theoretical throughput ceiling of your batch size. And if you have exactly one service with no network hops, distributed tracing collapses to single-process profiling, and `torch.profiler` alone is simpler than standing up a collector and a trace store. Reach for distributed tracing when a request crosses process or service boundaries and you need to know *which one* is slow; reach for profiling when you already know the process and need to know *which kernel*.

## Key takeaways

- **A latency metric is a verdict; a trace is the evidence.** The metric tells you a request was slow; only the span waterfall tells you which of `queue + prefill + decode` was the cause. Never remediate a latency spike you have not decomposed.
- **Split on the term first.** High TTFT with normal TPOT is a `gateway/queue/prefill` problem; high TPOT is a decode-loop problem; both spiking is overload/preemption; one slow request is input pathology. That one fork eliminates most candidate causes for free.
- **Prefill is compute-bound, decode is memory-bound.** Long prompts saturate FLOPs (fix with chunked prefill, prefix caching, prompt caps); big batches and long contexts saturate HBM bandwidth (fix with batch caps, FP8 KV cache, shorter outputs). The bound dictates the fix.
- **Propagate context or your trace is truncated.** Inject the W3C `traceparent` across the gateway→engine hop so spans join; a trace that ends at a service boundary means a hop dropped the header.
- **vLLM emits the decomposition for you.** `--otlp-traces-endpoint` gives per-request spans with `time_in_queue`, `time_to_first_token`, and `e2e` attributes — read the term off the span, don't reconstruct it from separate metrics.
- **Correlate the three pillars by ID.** Exemplars link metric→trace; a shared `span_id` links trace→log. Without the IDs you are grepping by timestamp during an incident.
- **Tail-sample, don't head-sample.** Keep every slow and errored trace by deciding after you see the duration; sample the fast majority. This is what makes tracing affordable at scale without losing the tail you actually need.
- **Go to the GPU only after the trace points there.** When the slow span is decode and the config is sane, Nsight/`torch.profiler` show the kernel timeline and the memory-bandwidth ceiling — confirming (or refuting) the memory-bound diagnosis against hardware counters, not assertion.
- **Reproduce the shape, not the guess.** The slow trace hands you prompt length, output length, and concurrency; replay those locally and bisect the fix before touching production.

## Further reading

- vLLM documentation — Observability / OpenTelemetry tracing, and the Prometheus metrics reference (`vllm:time_to_first_token_seconds`, `vllm:request_queue_time_seconds`, `vllm:num_preemptions_total`).
- OpenTelemetry documentation — the tracing data model, W3C Trace Context propagation, OTLP, and the Collector's `tail_sampling` processor.
- Grafana Tempo and Prometheus exemplars — TraceQL search and metric-to-trace correlation.
- Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" (2023) — the mechanism behind chunked prefill and the prefill-monopolization failure mode.
- NVIDIA Nsight Systems user guide and the PyTorch `torch.profiler` documentation — system-level and framework-level GPU timelines.
- Within this series: [what is model serving](/blog/machine-learning/model-serving/what-is-model-serving) for the SLO triangle; [why LLM serving is different](/blog/machine-learning/model-serving/why-llm-serving-is-different) for the memory-bound decode wall; [request scheduling and preemption](/blog/machine-learning/model-serving/request-scheduling-and-preemption) for the queue and eviction mechanics; the sibling posts on [observability for LLM serving](/blog/machine-learning/model-serving/observability-for-llm-serving) and the [troubleshooting LLM serving runbook](/blog/machine-learning/model-serving/troubleshooting-llm-serving-runbook); and the SRE cross-post on [metrics, logs, and traces: when to use which](/blog/software-development/site-reliability-engineering/metrics-logs-and-traces-when-to-use-which).

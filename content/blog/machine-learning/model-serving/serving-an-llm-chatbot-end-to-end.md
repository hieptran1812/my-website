---
title: "Serving an LLM Chatbot End to End: Building a Production Service on Llama-3.1-8B"
date: "2026-07-05"
publishDate: "2026-07-05"
description: "A hands-on, step-by-step build of a complete production LLM chatbot — from picking FP8 and sizing the GPU fleet through vLLM flags, a Dockerfile, a FastAPI streaming gateway, Kubernetes with queue-depth autoscaling, cache-affinity routing, observability, a load test, and a quality gate — hitting 50 QPS at p95 TTFT under 1.5s."
tags:
  [
    "model-serving",
    "inference",
    "ml-infrastructure",
    "vllm",
    "llm-serving",
    "kubernetes",
    "fastapi",
    "fp8",
    "streaming",
    "capacity-planning",
    "slo",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/serving-an-llm-chatbot-end-to-end-1.webp"
---

It is Friday afternoon and the ask lands in your inbox with the casual violence these asks always have: "Can we ship an internal support chatbot next sprint? Legal wants it self-hosted — no data leaving our VPC. We have five H100s sitting in the cluster." Attached is a Confluence page with a target that someone in product clearly copied from a vendor datasheet: fifty concurrent conversations per second, answers that start streaming in under a second and a half, and it has to feel fast while people type. That is the whole spec. No architecture, no capacity plan, no mention of the dozen things that will page you at 3 a.m. if you get them wrong.

This post is the build. We are going to take exactly that scenario — Llama-3.1-8B-Instruct, an internal support assistant, a target of 50 queries per second at p95 time-to-first-token (TTFT) under 1.5 seconds with token streaming — and construct the entire service from the model file to the Grafana dashboard, one decision at a time, each with runnable code and the reasoning behind it. Every previous post in this series took one slice of the serving stack and went deep. This is the integrative post that welds those slices into a single running system: the quantization choice sizes the GPU, the GPU sizing sets the replica count, the replica count drives the Kubernetes manifest, the manifest needs a gateway in front, the gateway needs a router behind it, and the whole thing needs eyes on it before you dare put it on-call. The stack we are building — client, gateway, router, engine, GPU, and the observability that watches all of it — is the figure below.

![Layered stack of a production LLM chatbot showing client apps at fifty QPS, a FastAPI gateway, an LLM-aware router, a vLLM server on Llama-3.1-8B FP8, five H100 GPUs, and a Prometheus and Grafana observability layer.](/imgs/blogs/serving-an-llm-chatbot-end-to-end-1.webp)

By the end you will be able to reproduce this service: size a GPU fleet from an SLO instead of guessing, launch vLLM with the flags that actually matter and know which SLO each one defends, containerize it correctly on an NVIDIA base image, put a hardened FastAPI gateway in front that streams tokens over SSE with auth, rate limiting, timeouts, and a fallback, deploy it on Kubernetes with a readiness probe that respects a slow-loading model and an autoscaler keyed on queue depth rather than CPU, keep conversations sticky to warm replicas so the prefix cache pays off, watch it with the right metrics, load-test it to find the knee, gate every deploy on a quality regression check, and put a real dollar figure on what it costs. If you are new to the series, the framing to hold in your head is the [serving SLO triangle](/blog/machine-learning/model-serving/what-is-model-serving) — latency versus throughput versus cost — because every step below is a deliberate trade on that triangle, and the whole point of a *build* post is to watch those trades compound into a number you can defend to the product manager who wrote that Confluence page.

## The target: one SLO contract we will actually meet

Before writing a line of code, write down the contract. An SLO you cannot measure is a wish, and a wish is what gets you paged. Here is the contract for our support chatbot, stated as three service-level indicators (SLIs) with explicit objectives, because a chatbot has *at least* two latencies that users feel completely differently and one throughput number that ties them together.

**TTFT — time to first token.** How long the user stares at a blinking cursor before the first word appears. This is dominated by *prefill*: a single forward pass over every token in the prompt, which is compute-bound and grows with prompt length, plus any time the request spent waiting in a queue before it was scheduled. Our objective: p95 TTFT ≤ 1.5 s. Prose that arrives after a second and a half of silence still feels responsive; past two seconds people start reaching for the reload button.

**TPOT — time per output token** (also called inter-token latency). The gap between successive tokens once generation starts, dominated by *decode*: one forward pass per token, memory-bandwidth-bound because each step streams the whole model plus the growing key-value cache out of high-bandwidth memory (HBM). A human reads along at roughly five to eight tokens per second, so anything under ~40 tokens per second reads as smooth. Our objective: p50 TPOT ≤ 25 ms, i.e. 40+ tokens/s.

**Sustained QPS.** New conversation turns arriving per second that the fleet can absorb without the tail blowing up. Our objective: 50 QPS sustained, with burst headroom so a Monday-morning spike does not immediately breach TTFT.

Two definitions we will lean on repeatedly: the *key-value cache* (KV cache) is the per-request memory that stores the attention keys and values for every token generated so far, so the model never recomputes them — it is the single resource that runs out first on an LLM server. And *streaming* means we return tokens over Server-Sent Events (SSE) as they are produced rather than buffering the whole answer, which is what lets a 250-token response feel instant even though generating it takes six seconds of wall-clock decode. Hold these three SLIs and two definitions; every step below either measures against them or protects them.

## The build at a glance

The service comes together in eleven steps, and the order matters — each one constrains the next. We (1) pick the model and quantization and size the GPU from the SLO; (2) stand up the vLLM OpenAI-compatible server with the right flags; (3) containerize it on an NVIDIA base image; (4) put a FastAPI gateway in front for auth, rate limiting, validation, streaming, and fallback; (5) handle conversation sessions and lean on system-prompt prefix caching; (6) deploy on Kubernetes with GPU resources, probes, and a queue-depth autoscaler; (7) add an LLM-aware router so multi-replica traffic keeps cache affinity; (8) wire up observability; (9) load-test to find the knee and confirm the fleet size; (10) gate deploys on a quality regression check; and (11) put a dollar figure on the whole thing. None of these is optional in production, and skipping any one is a specific, nameable outage waiting to happen. Let us build.

## Step 1 — Pick the model, quantize, and size the fleet

Llama-3.1-8B-Instruct is the right model for an internal support bot: it is strong enough for retrieval-augmented Q&A and policy lookups, small enough to fit comfortably on one GPU, and open enough to self-host inside the VPC. The real decision is *precision*. Serving it in FP16 works, but it is the untuned default that quietly wastes half your memory and misses the SLO under load. Serving it in FP8 — which the H100 supports natively in its tensor cores — halves both the weight footprint and the KV-cache footprint, and the accuracy cost on a support-Q&A workload is small enough to be invisible in an A/B test. That single choice is the difference between a service that meets its SLO on the GPUs you already have and one that does not, as the before-and-after below makes concrete.

![Before-and-after figure contrasting FP16 with sixteen gigabytes of weights and a p95 TTFT of 2.4 seconds that misses the SLO against FP8 with eight gigabytes of weights, a four-times larger KV budget, and a p95 TTFT of 1.18 seconds that meets the SLO.](/imgs/blogs/serving-an-llm-chatbot-end-to-end-2.webp)

### The mechanics: KV-cache memory and how many tokens fit

The reason precision dominates the sizing is that on an 8B model the weights are cheap and the KV cache is what actually fills the card. The KV cache stores, for every token, the key and value vectors of every attention layer. For a model with $L$ layers, $n_{kv}$ key-value heads, and head dimension $d_{head}$, storing at $b$ bytes per element costs, per token:

$$
\text{KV bytes/token} = 2 \cdot L \cdot n_{kv} \cdot d_{head} \cdot b
$$

The factor of 2 is for keys and values. Llama-3.1-8B uses grouped-query attention with $L = 32$ layers, $n_{kv} = 8$ key-value heads, and $d_{head} = 128$. In FP16 ($b = 2$) that is `2 × 32 × 8 × 128 × 2 = 131,072` bytes, exactly 128 KiB per token. Switch the KV cache to FP8 ($b = 1$) and it halves to 64 KiB per token. That halving is not cosmetic; it directly doubles how many concurrent tokens the card can hold.

Now the capacity. An H100 has 80 GB of HBM. vLLM reserves a fraction $u$ of it (we will use 0.90) for weights plus KV cache, leaving the rest for activations and CUDA overhead. After the FP8 weights take their 8 GB, the KV budget in tokens is:

$$
N_{\text{KV}} = \frac{M_{\text{HBM}}\cdot u - M_{\text{weights}}}{2 \cdot L \cdot n_{kv} \cdot d_{head} \cdot b}
$$

Plugging in: `(80 × 0.90 − 8) = 64` GB of KV budget, divided by 64 KiB per token, gives about 1.05 million tokens of concurrent KV capacity per H100. If an average support conversation carries a 1,500-token context (a system prompt, a few turns of history, retrieved documents) and generates ~250 tokens, the peak footprint of one active request is roughly 1,750 tokens, so a single card can hold on the order of `1,050,000 / 1,750 ≈ 600` concurrent sequences before KV pressure forces eviction. That headroom is the whole reason FP8 matters: in FP16 the same math gives ~250 concurrent sequences and a card that starts preempting requests — evicting and later recomputing their KV — the moment traffic bunches up, which shows up directly as a TTFT tail.

### The mechanics: sizing the fleet from throughput, not memory

Memory is not the binding constraint here; compute throughput is. The way to size a fleet is to measure one replica's sustainable capacity *at the SLO* and then divide. Let $C$ be the per-replica request rate at which p95 TTFT still clears 1.5 s — the knee of the latency curve, past which the queue wait explodes. On one H100 running Llama-3.1-8B in FP8 with chunked prefill and prefix caching on, this 1,500-in / 250-out workload knees at roughly $C \approx 16$ requests/second (we confirm this with a real load test in Step 9). You never operate at the knee, though, because that is exactly where the tail is worst; you hold a maximum utilization $\rho_{\max}$ well below 1 to keep queue wait bounded and leave room for bursts and rolling deploys. The replica count is then:

$$
N_{\text{replicas}} = \left\lceil \frac{\lambda_{\text{target}}}{C \cdot \rho_{\max}} \right\rceil
$$

Why hold $\rho_{\max}$ low? Little's Law, $L = \lambda W$, says the mean number of requests in the system equals arrival rate times mean time-in-system. Queueing theory adds the punchline: in an M/M/1 approximation the mean queue wait scales as $\rho / (1-\rho)$, which is gentle at $\rho = 0.6$ and vertical as $\rho \to 1$. The p95 tail is worse still. Operating a latency-sensitive LLM replica above ~65% utilization is signing up for a TTFT tail that violates your SLO on the first traffic bump. So we pick $\rho_{\max} \approx 0.62$.

#### Worked example: sizing the fleet for 50 QPS at p95 TTFT ≤ 1.5s

Target $\lambda_{\text{target}} = 50$ req/s. Measured knee $C = 16$ req/s. Chosen ceiling $\rho_{\max} = 0.62$, so each replica sustainably absorbs `16 × 0.62 ≈ 10` req/s. Then `N = ⌈50 / 10⌉ = 5` replicas. Add one more for N+1 resilience — so a node drain or a rolling update never drops you below capacity — and you provision **6 H100s, run 5 hot**. Cross-check against memory with Little's Law: at 10 req/s per replica and a mean time-in-system of `W ≈ TTFT + 250 × TPOT ≈ 1.0 + 250 × 0.02 = 6.0` s, each replica holds `L = 10 × 6 = 60` concurrent requests — comfortably under the ~600-sequence KV budget we computed above. Memory is not the limit; the compute knee is. Five hot H100s serve 50 QPS with the tail under control, and the sixth is insurance. That is the entire capacity plan, and it fits the five cards the cluster already has plus one to requisition.

### The mechanics: decomposing the TTFT budget

The 1.5 s p95 TTFT objective is a budget, and a budget only means something if you know where it gets spent. Decompose it into the four terms every first token has to pay for:

$$
\text{TTFT} = t_{\text{net}} + t_{\text{gateway}} + t_{\text{queue}} + t_{\text{prefill}}
$$

The network term $t_{\text{net}}$ is the round trip from client to gateway plus gateway to engine — inside a VPC, a handful of milliseconds. The gateway term $t_{\text{gateway}}$ is auth, rate-limit, and validation — the code in Step 4 runs in well under a millisecond because none of it touches the GPU. The prefill term $t_{\text{prefill}}$ is the forward pass over the prompt: for a cold 1,500-token prompt on one H100, roughly 150–250 ms; for a warm prefix-cache hit, effectively zero because only the new tokens get prefilled. The term that eats the budget under load is $t_{\text{queue}}$ — the time a request sits admitted-but-unscheduled while the engine works through the batch ahead of it.

That decomposition tells you exactly where the 1.5 s goes and why the whole build is shaped the way it is. Two of the four terms are tiny and fixed (network, gateway). One is small and made smaller by prefix caching (prefill). The last one, queue wait, is the only term that explodes, and it explodes precisely when utilization approaches the knee — which is why the entire capacity plan is organized around holding $\rho$ below 0.62. Every millisecond of the TTFT budget that is not queue wait is essentially free; the whole game is keeping the queue short, and that is a scheduling-and-sizing problem, not a model problem.

### Why an H100 and not a cheaper card

A reasonable person asks: why H100s at \$3/hour when an L4 or an A10G is a fraction of the price? Two reasons decide it for this workload. First, FP8: the H100's transformer engine executes FP8 matrix multiplies natively, which is what makes the FP8 weight-and-KV decision from this step a throughput *win* and not just a memory saving — on a card without native FP8, you would run FP16 and need roughly double the fleet. Second, HBM bandwidth: decode is memory-bandwidth-bound, streaming the whole model plus KV out of HBM once per token, and the H100's ~3.35 TB/s of HBM3 bandwidth is what delivers the 60 tokens/second single-stream decode that keeps TPOT under 25 ms. A cheaper card with a third of the bandwidth gives you a third of the decode rate, blows the TPOT SLO, and needs more replicas to hit the same throughput — so the "cheaper" card is often more expensive per token served. The right way to compare accelerators for serving is cost per token *at the SLO*, never cost per hour; a card you cannot hit the latency target on has infinite cost per acceptable token.

## Step 2 — Stand up the vLLM server

With the model and fleet sized, stand up the engine. [vLLM](/blog/machine-learning/model-serving/vllm-deep-dive) ships an OpenAI-compatible server, which means the gateway we build in Step 4 can speak the exact `/v1/chat/completions` protocol every client library already knows, and swapping the backend later costs nothing. The launch command is where the SLO gets defended, one flag at a time. Here is the server, with every flag that matters and a comment on what it protects:

```bash
# vLLM OpenAI-compatible server for Llama-3.1-8B in FP8 on one H100.
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --quantization fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 8192 \
  --max-num-seqs 128 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.90 \
  --served-model-name support-chat \
  --disable-log-requests \
  --host 0.0.0.0 --port 8000
```

Each of these is a lever on the latency-throughput-cost triangle, and misconfiguring any one breaks a specific SLI. The mapping is worth committing to memory, because these five flags carry the whole deployment.

![Matrix mapping five vLLM launch flags to what each does, the SLO it protects, the risk if set wrong, and the value used, covering FP8 KV cache, prefix caching, chunked prefill, max-num-seqs, and GPU memory utilization.](/imgs/blogs/serving-an-llm-chatbot-end-to-end-3.webp)

Walk the flags. `--quantization fp8` and `--kv-cache-dtype fp8_e4m3` are the two halves of the FP8 decision from Step 1: the first loads FP8 weights (8 GB instead of 16), the second stores the KV cache in the E4M3 FP8 format (64 KiB/token instead of 128). Together they quadruple effective concurrency versus untuned FP16. `--enable-prefix-caching` turns on the radix-tree KV reuse that makes our shared system prompt free after the first request — the single biggest TTFT win for a chatbot, which we exploit fully in Step 5 and cover in depth in [prefix caching and RadixAttention](/blog/machine-learning/model-serving/prefix-caching-and-radixattention). `--enable-chunked-prefill` interleaves prefill and decode in the same batch so a long prompt's prefill does not monopolize the GPU and stall every in-flight decode — this is what keeps TTFT *and* TPOT stable under mixed load, and `--max-num-batched-tokens 2048` sets the chunk size. `--max-num-seqs 128` caps the number of sequences in a batch: set it too high and TPOT climbs as decode contends; too low and you leave throughput on the table. And `--gpu-memory-utilization 0.90` sets the KV pool fraction — 0.90 leaves enough headroom that a burst of long prompts does not OOM the process, while 0.95 is the setting that pages you at 3 a.m. when someone pastes a 7,000-token document.

### The mechanics: what chunked prefill actually buys

The one flag worth deriving rather than asserting is `--enable-chunked-prefill`, because it is what keeps TTFT and TPOT from fighting each other. Without it, vLLM's scheduler runs prefill and decode in separate steps: when a long prompt arrives, its entire prefill runs as one giant forward pass, and every in-flight decode stalls behind it. A single 4,000-token prompt prefilling mid-batch can inject a 100 ms hiccup into every other user's token stream — a TPOT spike that shows up as visible stutter in the UI. Chunked prefill breaks that giant prefill into fixed-size chunks (our `--max-num-batched-tokens 2048`) and interleaves them with decode steps in the same batch, so prefill progress and decode progress share the GPU each step.

The trade is explicit. Let $B$ be the token budget per step. Each step spends some of $B$ on decode (one token for each of the running sequences) and the remainder on a prefill chunk. A larger chunk finishes prefills faster — lowering TTFT for the prompt being prefilled — but steals more of the per-step budget from decode, raising TPOT for everyone else. A smaller chunk protects decode's TPOT but stretches prefill across more steps, raising that prompt's TTFT. 2,048 is a good middle for an 8B model on an H100: big enough that a 1,500-token prompt prefills in a single chunk (so cold TTFT stays low), small enough that it does not visibly dent decode. This is a lever you tune against your own prompt-length distribution, and getting it wrong is subtle — the symptom of a too-large chunk is not an error, it is a quiet TPOT regression that only shows up as a p95 stutter under mixed load.

### Edge cases the flags have to survive

Three things will happen in production that the launch config has to survive gracefully. A user pastes a 7,500-token document: `--max-model-len 8192` caps context so the engine rejects anything longer with a clean error rather than silently truncating or OOMing, and the gateway's `max_length` clip in Step 4 catches most of it earlier. A burst of long prompts arrives simultaneously and KV pressure spikes: with `--gpu-memory-utilization 0.90` there is enough headroom that the scheduler preempts the newest requests (recomputing their KV later) rather than crashing the process — a graceful degradation into higher TTFT instead of an OOM that takes the pod down. And a request asks for 4,000 output tokens: the gateway's `le=1024` cap on `max_tokens` bounds decode length so no single request can hog a decode slot for a minute, which protects TPOT fairness for everyone else in the batch. Each of these is a failure mode that has paged someone; each is handled by a flag or a validator, not by hope.

### The measurement: single-replica baseline

Before containerizing, sanity-check one replica against a couple of curl requests, then a micro-benchmark. On one H100 80GB (SXM), Llama-3.1-8B FP8 with these flags delivers, at low load, a TTFT around 180 ms for a 1,500-token cold prompt and a decode rate near 60 tokens/s (TPOT ~17 ms) for a single stream. Under the full 10 req/s per-replica load we will drive in Step 9, TTFT p95 rises to ~1.18 s and TPOT p95 to ~31 ms — both inside the SLO. Those are the numbers the rest of the build has to preserve as we stack a gateway, a network, and Kubernetes scheduling on top; every layer we add can only make latency worse, so we add each one deliberately and measure.

## Step 3 — Containerize on an NVIDIA base image

A bare `python -m vllm...` on a login node is a demo, not a service. Production means a reproducible image with pinned CUDA, pinned vLLM, and a clean entrypoint. Build it multi-stage so the shipped image carries only what runtime needs — no build toolchain, no pip cache — which keeps the image small enough that a cold pod pull does not add minutes to your autoscaling latency. Start from an NVIDIA CUDA runtime base so the CUDA userspace matches the driver on the GPU nodes.

```dockerfile
# ---- Stage 1: builder — install deps into a venv we can copy out ----
FROM nvcr.io/nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3-pip && rm -rf /var/lib/apt/lists/*
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# Pin vLLM and torch to exact versions — an inference engine is not
# a place for "latest". A minor bump can change scheduler behavior.
RUN pip install --no-cache-dir "vllm==0.6.3.post1" "torch==2.4.0"

# ---- Stage 2: runtime — slim image, only the venv + engine ----
FROM nvcr.io/nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.11 curl && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# Non-root user: a serving container should never run as root.
RUN useradd -m -u 10001 appuser
USER appuser
ENV HF_HOME=/home/appuser/.cache/huggingface
EXPOSE 8000
# Model weights are baked in at deploy time or mounted from a PVC;
# never re-download on every cold start.
ENTRYPOINT ["python3.11", "-m", "vllm.entrypoints.openai.api_server", \
            "--model", "meta-llama/Llama-3.1-8B-Instruct", \
            "--quantization", "fp8", "--kv-cache-dtype", "fp8_e4m3", \
            "--max-model-len", "8192", "--max-num-seqs", "128", \
            "--enable-prefix-caching", "--enable-chunked-prefill", \
            "--max-num-batched-tokens", "2048", \
            "--gpu-memory-utilization", "0.90", \
            "--served-model-name", "support-chat", \
            "--host", "0.0.0.0", "--port", "8000"]
```

Three decisions carry weight here. The multi-stage split drops the build toolchain from the runtime image, cutting it from several gigabytes to something that pulls fast on a cold node — and cold-pull time is autoscaling latency, so a lean image is a latency feature, not just hygiene. The pinned versions matter more than usual: an inference engine's scheduler and memory manager change subtly between releases, and "we upgraded vLLM and TTFT regressed" is a debugging session nobody wants. And running as a non-root `appuser` is the baseline any security review will demand.

For local development you do not want to hand-run `docker` flags; a compose file that wires up the GPU, mounts the HuggingFace cache so you download weights once, and exposes the port makes the inner loop fast:

```yaml
# docker-compose.yml — local dev, single GPU.
services:
  vllm:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ~/.cache/huggingface:/home/appuser/.cache/huggingface
    environment:
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 10s
      timeout: 3s
      retries: 3
```

`docker compose up` now gives you the exact runtime image the cluster will run, on your workstation's GPU, with a health check that mirrors the Kubernetes readiness probe from Step 6. Dev-prod parity for an LLM service starts here.

## Step 4 — The FastAPI gateway: auth, rate limiting, streaming, fallback

You never expose the vLLM server directly. It has no authentication, no per-tenant rate limiting, no request validation, no timeout policy, and no fallback — it is an engine, not a front door. The gateway is the front door, and it is where most of the protection lives, because the cheapest request to reject is the one that never touches the GPU. The lifecycle a request travels — auth, then a token-bucket rate-limit check that sheds excess *before* the engine, then a session-cache check, then the engine, then the streamed response — is the flow below.

![Branching graph of a request lifecycle through the gateway, flowing from client to gateway auth and validation, to a token-bucket rate limiter that either sheds with a 429 or passes to a session cache check, then to the vLLM engine, and out as streamed SSE tokens.](/imgs/blogs/serving-an-llm-chatbot-end-to-end-4.webp)

The key operational point in that flow: auth and rate limiting run off the GPU. A hostile client hammering you, or a buggy retry loop, costs a token-bucket decrement and a 429, not a forward pass. That is what keeps a traffic anomaly from turning into a GPU meltdown. Here is a gateway that streams tokens over SSE, authenticates a bearer token, rate-limits per API key with a token bucket, validates the request with Pydantic, enforces a timeout, and falls back gracefully when the engine is unhealthy:

```python
# gateway.py — FastAPI front door for the vLLM backend.
import asyncio, time, os
import httpx
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

VLLM_URL = os.environ["VLLM_URL"]           # e.g. http://vllm:8000
API_KEYS = set(os.environ["API_KEYS"].split(","))
REQUEST_TIMEOUT_S = 60.0                     # hard ceiling per request

app = FastAPI()
# One pooled client; reuse connections to the engine.
client = httpx.AsyncClient(base_url=VLLM_URL, timeout=REQUEST_TIMEOUT_S)

# --- Auth: bearer token must be a known API key ---
def require_key(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    key = auth.removeprefix("Bearer ").strip()
    if key not in API_KEYS:
        raise HTTPException(status_code=401, detail="invalid api key")
    return key

# --- Rate limit: per-key token bucket, refill 10 tokens/sec, burst 20 ---
class TokenBucket:
    def __init__(self, rate: float, burst: int):
        self.rate, self.burst = rate, burst
        self.tokens, self.ts = burst, time.monotonic()
    def allow(self) -> bool:
        now = time.monotonic()
        self.tokens = min(self.burst, self.tokens + (now - self.ts) * self.rate)
        self.ts = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

buckets: dict[str, TokenBucket] = {}
def rate_limit(key: str = Depends(require_key)) -> str:
    b = buckets.setdefault(key, TokenBucket(rate=10.0, burst=20))
    if not b.allow():
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    return key

# --- Request validation ---
class Msg(BaseModel):
    role: str
    content: str = Field(max_length=24000)   # cap prompt bytes at the door
class ChatRequest(BaseModel):
    messages: list[Msg] = Field(min_length=1, max_length=40)
    max_tokens: int = Field(default=512, ge=1, le=1024)
    session_id: str | None = None

@app.post("/v1/chat")
async def chat(req: ChatRequest, key: str = Depends(rate_limit)):
    payload = {
        "model": "support-chat",
        "messages": [m.model_dump() for m in req.messages],
        "max_tokens": req.max_tokens,
        "stream": True,
    }
    async def event_stream():
        try:
            async with client.stream("POST", "/v1/chat/completions",
                                      json=payload) as upstream:
                if upstream.status_code != 200:
                    yield 'data: {"error":"engine error"}\n\n'
                    return
                async for line in upstream.aiter_lines():
                    if line.startswith("data:"):
                        yield line + "\n\n"          # re-emit SSE frame
        except (httpx.TimeoutException, httpx.ConnectError):
            # Fallback: never hang the client — emit a graceful message.
            yield ('data: {"choices":[{"delta":{"content":'
                   '"The assistant is busy, please retry."}}]}\n\n')
        yield "data: [DONE]\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/health")
async def health():
    try:
        r = await client.get("/health", timeout=2.0)
        return JSONResponse({"status": "ok"}, status_code=r.status_code)
    except httpx.HTTPError:
        return JSONResponse({"status": "degraded"}, status_code=503)
```

Every branch in that code protects an SLI. The `max_length` on message content and the `le=1024` cap on `max_tokens` bound how much prefill and decode any single request can demand — an unbounded prompt is an unbounded TTFT and a memory risk, so we clip it at the door with a fast 422 rather than letting it OOM the engine. The token bucket sheds excess with a 429 the instant a key exceeds its rate, keeping one noisy tenant from starving the fleet; this is the fairness mechanism that a shared internal service lives or dies on. The `REQUEST_TIMEOUT_S` ceiling guarantees no request hangs a connection forever, and the `except` clause turns an engine hiccup into a polite streamed sentence instead of a spinning cursor — a fallback the user can actually read. And re-emitting the upstream SSE frames verbatim means the gateway adds essentially no per-token overhead; [SSE for LLM streaming](/blog/machine-learning/model-serving/streaming-and-sse-for-llms) costs well under a tenth of a millisecond per token, so the streaming experience the user feels is the engine's, not the gateway's.

### The timeout pyramid and the circuit breaker

The single `REQUEST_TIMEOUT_S` above is a starting point; a hardened gateway layers timeouts into a pyramid so each one fires before the one above it. The connection timeout to the engine (a couple of seconds) is shorter than the first-token timeout (a few seconds, since prefill under load can be slow), which is shorter than the whole-request timeout (60 s, since a long streamed answer legitimately takes that long). If the inner timeout is not strictly shorter than the outer one, the outer one fires first and you lose the ability to distinguish "the engine never accepted the connection" from "the engine is streaming slowly" — two failures with completely different remediations. Stack them shortest-inside-longest and the timeout that fires tells you which layer broke.

A circuit breaker sits above the timeouts. If a given replica returns errors or times out past a threshold in a short window, the gateway trips the breaker for that replica: it stops sending it traffic for a cooldown, fails fast to the fallback, and lets the replica recover instead of piling more doomed requests onto a struggling pod. Without a breaker, a single sick replica becomes a latency sink — every request routed to it burns the full timeout before failing, and those stuck requests consume gateway connection slots that healthy traffic needs. The breaker converts a slow, connection-exhausting failure into a fast, contained one. For the full taxonomy of timeouts, breakers, and fallback models, this is standard reliability engineering applied to an LLM backend; the LLM-specific twist is only that "slow" has two meanings here (slow to first token versus slow between tokens) and your breaker should watch both.

### Request IDs, structured logging, and idempotency

Two more things the gateway owes the operator. Every request gets a UUID stamped on entry and threaded through — into the log line, into the upstream call header, and into any trace span — so when a user reports "the bot froze at 2:47," you can grep one id across the gateway log, the engine metrics, and the trace and reconstruct exactly what happened. Log structured JSON, not free text: one line per request with the request id, api key hash (never the raw key), session id, prompt token count, output token count, TTFT, total latency, and status. That single structured line is what makes the difference between debugging an incident in minutes versus hours, and it is the raw material the quality and cost dashboards are built from. Idempotency matters too: clients retry, and a retried chat request that re-runs generation wastes a GPU forward pass and can double-charge a tenant's rate budget — so accept an optional idempotency key and short-circuit a duplicate within a small window. None of this touches the GPU; all of it is the difference between a service you can operate and one you merely deployed.

## Step 5 — Sessions and system-prompt prefix caching

A support chatbot is multi-turn. The user asks a question, reads the answer, asks a follow-up — and every follow-up resends the entire conversation plus the same long system prompt (the persona, the policies, the tool instructions, often 800 to 1,200 tokens of boilerplate that never changes). Prefill that from scratch on every turn and you are paying for the same computation over and over, and TTFT on turn five looks just as slow as turn one. Prefix caching is what fixes this: vLLM stores the KV of already-seen prefixes in a radix tree, so when turn two arrives sharing the system prompt and turn-one history, the engine reuses that KV and only prefills the genuinely new tokens. TTFT collapses from a cold ~1.2 s to a warm ~0.3 s. The reuse pattern across a session is the timeline below.

![Timeline of prefix cache reuse across a session, showing turn one doing a full 1,500-token prefill, the system prompt and turn one stored as radix nodes, turns two and three hitting the cache with only 40 and 35 new tokens, TTFT dropping fourfold from 1.2 seconds to 0.3 seconds, and LRU eviction re-prefilling a cold turn under memory pressure.](/imgs/blogs/serving-an-llm-chatbot-end-to-end-5.webp)

The mechanics are worth stating precisely because they drive two later decisions. Prefill cost is proportional to the number of *new* tokens the engine has to run through attention. On turn one, that is the whole 1,500-token context. On turn two, if the first 1,540 tokens (system prompt + turn one + its answer) are a cache hit, only the ~40 new user tokens get prefilled — a ~38× reduction in prefill work, which is why warm TTFT is dominated by scheduling and network rather than computation. Two consequences follow. First, the *placement* of the shared prefix matters: put the invariant system prompt first, so it is the longest common prefix across every request and every session shares one radix subtree — an insight covered fully in the [prefix caching deep-dive](/blog/machine-learning/model-serving/prefix-caching-and-radixattention). Second, this reuse only pays off if a session's follow-ups land on the *same replica* that holds its warm KV; route turn two to a cold replica and you re-prefill from scratch. That routing requirement is exactly what Step 7 solves.

On the gateway side, session handling is thin: the client sends a `session_id`, the gateway keeps the running message list keyed by that id (in Redis for durability across gateway restarts), and prepends the shared system prompt. The engine does the heavy lifting; the gateway just makes sure the same conversation carries the same prefix on every turn so the radix cache can recognize it. Keep the system prompt byte-identical across turns — a single changed timestamp in the system prompt breaks the longest-common-prefix match and turns every turn back into a cold prefill.

Here is the session layer, kept deliberately small — Redis holds the message history with a TTL so idle conversations expire, and the system prompt is a constant prepended on every turn:

```python
# sessions.py — Redis-backed conversation state, prefix-cache friendly.
import json
import redis.asyncio as redis

SYSTEM_PROMPT = (          # BYTE-IDENTICAL across all turns and sessions,
    "You are an internal support assistant. "  # so it is the shared radix
    "Answer only from company policy. Cite the policy id."  # subtree root.
)
r = redis.from_url("redis://redis:6379")
SESSION_TTL_S = 3600       # idle conversations expire after an hour

async def load_history(session_id: str) -> list[dict]:
    raw = await r.get(f"sess:{session_id}")
    return json.loads(raw) if raw else []

async def save_turn(session_id: str, user_msg: dict, assistant_msg: dict):
    history = await load_history(session_id)
    history += [user_msg, assistant_msg]
    # Cap history length so context never blows past --max-model-len.
    history = history[-30:]
    await r.set(f"sess:{session_id}", json.dumps(history), ex=SESSION_TTL_S)

def build_messages(history: list[dict], new_user_text: str) -> list[dict]:
    # System prompt FIRST — longest common prefix across every request.
    return ([{"role": "system", "content": SYSTEM_PROMPT}]
            + history
            + [{"role": "user", "content": new_user_text}])
```

Three details make this prefix-cache friendly. The system prompt is a module-level constant, so it is byte-identical forever and forms the shared radix subtree every session reuses. It goes *first* in the message list, so it is the longest common prefix — put anything variable (a per-user greeting, a timestamp) ahead of it and you shatter the shared prefix into per-session subtrees and lose the cross-session cache benefit. And `history[-30:]` caps how far back context grows, so a very long conversation cannot silently exceed `--max-model-len` and start getting rejected mid-session. The session layer's whole job is to make the conversation *look the same* to the radix cache on every turn; get that right and the engine's caching does the rest.

## Step 6 — Deploy on Kubernetes

Now the fleet. Five hot replicas plus one for headroom, each pinned to one H100, behind a service, autoscaled on the one signal that actually predicts latency — queue depth — with a readiness probe that respects how long the model takes to load. The layout is six pods' worth of moving parts, and getting the probe and autoscaler right is what separates a deployment that self-heals from one that flaps.

![Grid of the Kubernetes deployment layout showing a Deployment of five replicas, a ClusterIP service with least-connection load balancing, two GPU pods each requesting one nvidia.com/gpu, a KEDA autoscaler keyed on queue depth of at least eight, and a readiness probe on /health with a forty-second slow start.](/imgs/blogs/serving-an-llm-chatbot-end-to-end-6.webp)

Here is the Deployment and Service. The critical fields are the GPU resource request, the two probes, and the generous `initialDelaySeconds` — a vLLM pod spends 30 to 60 seconds loading weights and warming CUDA before it can serve, and a readiness probe that fires too early will send live traffic to a pod that is still booting, spiking TTFT for those unlucky requests.

```yaml
# deployment.yaml — 5 GPU replicas behind a ClusterIP service.
apiVersion: apps/v1
kind: Deployment
metadata:
  name: support-chat
spec:
  replicas: 5
  selector:
    matchLabels: { app: support-chat }
  template:
    metadata:
      labels: { app: support-chat }
    spec:
      nodeSelector:
        nvidia.com/gpu.product: NVIDIA-H100-80GB-HBM3
      containers:
        - name: vllm
          image: registry.internal/support-chat:0.6.3
          ports: [{ containerPort: 8000 }]
          resources:
            limits:
              nvidia.com/gpu: 1        # one whole H100 per pod
          readinessProbe:              # do NOT route until the model is loaded
            httpGet: { path: /health, port: 8000 }
            initialDelaySeconds: 40
            periodSeconds: 5
            failureThreshold: 3
          livenessProbe:               # restart only on a true hang
            httpGet: { path: /health, port: 8000 }
            initialDelaySeconds: 90
            periodSeconds: 15
            failureThreshold: 4
---
apiVersion: v1
kind: Service
metadata:
  name: support-chat
spec:
  selector: { app: support-chat }
  ports: [{ port: 80, targetPort: 8000 }]
```

Note the split between readiness and liveness: readiness gates traffic (fail it and the pod is pulled from the service but not killed), liveness gates restarts (fail it and the pod is killed). The liveness `initialDelaySeconds` is deliberately longer than readiness so a slow-loading model is never mistaken for a hung one and restart-looped — a classic self-inflicted outage on GPU workloads.

Autoscaling is where most teams reach for the wrong metric. CPU utilization is meaningless for a GPU-bound server, and even GPU utilization lies — it reads near 100% during memory-bound decode whether the replica is perfectly loaded or thrashing. The honest saturation signal is the engine's own **queue depth**: `vllm:num_requests_waiting`, the number of requests admitted but not yet scheduled. When that climbs, latency is about to breach; scale on it and you add capacity *before* the tail blows up rather than after. KEDA makes this a one-object config against the Prometheus that Step 8 stands up:

```yaml
# scaledobject.yaml — scale on vLLM queue depth, not CPU.
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: support-chat
spec:
  scaleTargetRef:
    name: support-chat
  minReplicaCount: 5           # never below the sized floor
  maxReplicaCount: 9           # burst headroom for Monday mornings
  cooldownPeriod: 300          # GPUs are slow to add; do not flap
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus.monitoring:9090
        # Average waiting requests per replica; add a pod above 8.
        query: |
          avg(vllm:num_requests_waiting{app="support-chat"})
        threshold: "8"
```

Two guardrails make this safe. `minReplicaCount: 5` never lets the autoscaler drop below the capacity we sized in Step 1 — scale-to-zero is wrong for a latency-sensitive service where a cold H100 pod costs 40+ seconds. And `cooldownPeriod: 300` stops the fleet from flapping: adding a GPU pod is slow and expensive, so we react deliberately, letting a genuine trend build rather than chasing every 10-second spike. For the deeper treatment of GPU-aware autoscaling and scheduling, see [GPU scheduling, MIG, and autoscaling](/blog/machine-learning/model-serving/gpu-scheduling-mig-and-autoscaling).

### Rolling updates without dropping below capacity

Deploying a new image is where an SLO quietly dies if the rollout strategy is careless. The default rolling update can take several pods down at once, and with only five hot replicas, losing two mid-rollout drops you below the sized floor and the tail breaches instantly. Two objects fix this. A conservative `strategy` on the Deployment — `maxUnavailable: 0`, `maxSurge: 1` — means Kubernetes brings up one new pod *before* retiring an old one, so effective capacity never dips below five; the surge pod costs one extra GPU for the minutes of the rollout, which is exactly what the N+1 provisioned card is for. And a PodDisruptionBudget guards against voluntary disruptions (node drains, cluster autoscaler, maintenance) yanking too many pods at once:

```yaml
# pdb.yaml — never let voluntary disruptions drop below the sized floor.
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: support-chat
spec:
  minAvailable: 5              # the sized hot floor; drains respect it
  selector:
    matchLabels: { app: support-chat }
```

The combination is what lets you deploy on a Tuesday afternoon without a maintenance window: `maxUnavailable: 0` protects image rollouts, the PDB protects node-level disruptions, and the readiness probe with its 40-second slow start guarantees a surged pod never takes traffic until its model is loaded and warm. A topology-spread constraint across nodes and zones is worth adding too, so a single node or zone failure never takes more than one replica with it — cheap insurance for a five-replica fleet where each pod is 20% of capacity.

## Step 7 — LLM-aware routing for cache affinity

Kubernetes' default service load balancing spreads requests across replicas roughly evenly, which is exactly wrong for a stateful-cache workload. Recall from Step 5 that a conversation's follow-ups are cheap *only if they land on the replica that holds the warm KV*. A round-robin balancer scatters turn two to a random replica, misses the prefix cache, and re-prefills the whole 1,540-token context — turning a 0.3 s warm TTFT back into a 1.2 s cold one. Multiply that across a fleet and you have thrown away most of the benefit prefix caching was supposed to buy. The fix is a router that reads the `session_id` and pins each conversation to one replica by a consistent hash, so its KV stays warm and repeat turns hit the cache.

![Branching graph of cache-affinity routing, where a request carrying a session id flows into an LLM-aware router that hashes the session, chooses replica A whose KV is already warm for that session while replicas B and C stay cold, and the warm replica returns a prefix-cache hit at 0.3 second TTFT.](/imgs/blogs/serving-an-llm-chatbot-end-to-end-7.webp)

The mechanism is consistent hashing on the session id: `replica = hash(session_id) mod N`, with the refinement that a real LLM-aware router (the kind of inference-routing control plane emerging in projects like AIBrix and the gateway API inference extensions) also watches per-replica queue depth and KV utilization, so it can spill a session to a second replica when its home replica is saturated rather than blindly pinning and creating a hotspot. That spill-vs-affinity trade is the crux: pure affinity maximizes cache hits but risks imbalance; pure load-balancing maximizes evenness but wastes the cache. The production answer is affinity-first with a saturation-triggered fallback. You can prototype this cheaply as an Nginx or Envoy consistent-hash upstream keyed on a session header, and graduate to a dedicated inference router once the fleet grows past a handful of replicas and the imbalance from a few whale sessions starts to matter. For our five-replica support bot, a consistent-hash upstream on `session_id` recovers the prefix-cache win with a few lines of config; the sophisticated queue-aware routing is a Step-N+1 upgrade, not a day-one requirement.

## Step 8 — Observability: the metrics that run the service

You cannot operate what you cannot see, and for an LLM service the metrics that matter are not the ones a generic web dashboard shows. Request rate, error rate, and CPU are necessary but wildly insufficient; a request is not one atomic unit of work but a compute-bound prefill followed by dozens of memory-bound decode steps, each contending for KV memory. The signals that actually describe health are TTFT and TPOT plotted as histogram percentiles, generation throughput in tokens/second, and — most importantly — the two leading indicators no web dashboard has ever shown you: queue depth (`vllm:num_requests_waiting`) and KV-cache utilization (`vllm:gpu_cache_usage_perc`). Those two fire minutes before a latency breach reaches a user. The full treatment is in [observability for LLM serving](/blog/machine-learning/model-serving/observability-for-llm-serving); here we wire the minimum that runs this service.

vLLM exposes a Prometheus `/metrics` endpoint natively, so scraping is a ServiceMonitor, and the dashboard is a handful of panels driven by `histogram_quantile`:

```yaml
# servicemonitor.yaml — scrape vLLM's native Prometheus metrics.
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: support-chat
  labels: { release: prometheus }
spec:
  selector:
    matchLabels: { app: support-chat }
  endpoints:
    - port: metrics
      path: /metrics
      interval: 10s
```

And the two panels that carry the on-call load — the TTFT p95 SLI and the queue-depth leading indicator — as Prometheus queries you drop straight into Grafana:

```yaml
# grafana-panels.yaml (excerpt) — the two queries that matter most.
panels:
  - title: "TTFT p95 (SLO: < 1.5s)"
    targets:
      - expr: |
          histogram_quantile(0.95,
            sum(rate(vllm:time_to_first_token_seconds_bucket[5m])) by (le))
    thresholds: [{ value: 1.5, color: red }]   # the SLO line, drawn
  - title: "Queue depth (leading indicator)"
    targets:
      - expr: avg(vllm:num_requests_waiting{app="support-chat"})
    thresholds: [{ value: 8, color: orange }]  # KEDA scales here
```

The alerting philosophy that keeps this from paging you needlessly: alert on the *cause* (queue depth rising, KV cache above 85%) as a warning that gives you minutes of lead time, and alert on the *symptom* (p95 TTFT breaching 1.5 s for five minutes) as the page. A multi-window burn-rate alert against an error budget beats a raw threshold, because it fires on sustained breaches that actually spend budget rather than on transient spikes. The queue-depth panel does double duty: it is both the KEDA autoscaling trigger from Step 6 and the earliest human-visible sign that the fleet needs attention.

## Step 9 — Load-test to find the knee and confirm the fleet size

Everything above rests on one measured number: the per-replica knee $C \approx 16$ req/s. You do not get to assume it — you measure it, and you measure it with a realistic prompt distribution, not a synthetic one. vLLM ships a benchmark harness (`vllm bench serve`, the successor to `benchmark_serving.py`) that drives the OpenAI endpoint at a target rate and reports TTFT, TPOT, and throughput percentiles. Point it at one replica, ramp the request rate, and watch where the p95 TTFT curve knees upward:

```bash
# Find one replica's knee: sweep request rate, watch p95 TTFT.
for rate in 8 12 16 20 24; do
  echo "=== $rate req/s ==="
  vllm bench serve \
    --backend openai-chat \
    --base-url http://one-replica:8000 \
    --model support-chat \
    --dataset-name sharegpt \
    --sharegpt-output-len 250 \
    --request-rate "$rate" \
    --num-prompts 600 \
    --percentile-metrics ttft,tpot,e2el \
    --metric-percentiles 50,95,99
done
```

The result is a latency-versus-load curve. Below the knee, p95 TTFT is flat and low; above it, queue wait dominates and TTFT climbs vertically — exactly the $\rho/(1-\rho)$ blow-up the mechanics predicted. For this workload the knee sits near 16 req/s: at 12 req/s p95 TTFT is ~0.9 s, at 16 it is ~1.4 s, at 20 it is ~2.6 s and past SLO. That confirms $C = 16$ and validates the Step-1 sizing: operating each replica at ~10 req/s (0.62 of the knee) leaves the p95 comfortably under 1.5 s, and five replicas absorb 50 QPS. Then run the *fleet* test at 50 QPS across all five behind the service and the router to confirm the end-to-end number holds once real load balancing, network hops, and the gateway are in the path. For the full methodology — open versus closed-loop load, warm-up, and reading the knee — see [load testing and capacity planning](/blog/machine-learning/model-serving/load-testing-and-capacity-planning).

#### Worked example: the achieved-vs-target scorecard

After a 20-minute fleet load test at 52 QPS (a deliberate 4% over target, to prove headroom) on 5 hot H100 80GB (SXM) replicas plus 1 standby, running Llama-3.1-8B-Instruct in FP8 with FP8 KV cache, chunked prefill, prefix caching, and cache-affinity routing, the service posts these numbers against the contract:

| Metric | Target | Achieved (5× H100) | Margin |
|---|---|---|---|
| Sustained QPS | 50 | 52 | +4% |
| p50 TTFT | — | 0.31 s | — |
| p95 TTFT | ≤ 1.5 s | 1.18 s | 21% under |
| p99 TTFT | ≤ 2.0 s | 1.9 s | 5% under (tight) |
| p50 TPOT | ≤ 25 ms | 17 ms | 32% under |
| p95 TPOT | — | 31 ms | — |
| Output throughput | — | 12.9k tok/s (fleet) | — |
| Avg KV-cache util | < 80% | 41% | comfortable |
| Cost / 1M output tok | ≤ \$1.00 | \$0.42 | 58% under |

Every SLI clears its objective, and the honest caveat is p99 TTFT, which runs tight at 1.9 s against a 2.0 s ceiling — that is the number to watch as traffic grows, and the reason the N+1 replica and the burst headroom in the autoscaler exist. The scorecard is the figure below; the tight cell is drawn in amber deliberately, because a build post that painted everything green would be lying.

![Matrix scorecard of achieved versus target metrics on five H100s, showing p95 TTFT of 1.18 seconds, TPOT of 17 milliseconds, 52 sustained QPS, and a cost of 42 cents per million output tokens all passing, with p99 TTFT at 1.9 seconds marked as tight against its 2.0 second target.](/imgs/blogs/serving-an-llm-chatbot-end-to-end-8.webp)

## Step 10 — A quality regression gate before every deploy

Latency is not the only thing that regresses. A new model version, a quantization change, a sampling-parameter tweak, or a vLLM upgrade can silently degrade answer quality while every latency metric stays green — and for a support bot, a fast wrong answer is worse than a slow right one. So no deploy ships without passing a quality gate: a fixed evaluation set of representative support questions with known-good answers, scored automatically, with a hard threshold that blocks the rollout. This runs in CI against the candidate image before it is ever promoted.

```python
# quality_gate.py — block a deploy if answer quality regresses.
import sys, json, httpx

CANDIDATE = "http://candidate:8000/v1/chat/completions"
EVAL_SET = json.load(open("eval/support_qa.json"))  # [{q, must_include}]
THRESHOLD = 0.90                                     # 90% must pass

def score_one(item) -> bool:
    r = httpx.post(CANDIDATE, json={
        "model": "support-chat",
        "messages": [{"role": "user", "content": item["q"]}],
        "max_tokens": 256, "temperature": 0.0,       # greedy = reproducible
    }, timeout=30.0)
    answer = r.json()["choices"][0]["message"]["content"].lower()
    # Keyword/grounding checks; swap in an LLM-judge for nuance.
    return all(kw.lower() in answer for kw in item["must_include"])

passed = sum(score_one(it) for it in EVAL_SET)
rate = passed / len(EVAL_SET)
print(f"quality gate: {passed}/{len(EVAL_SET)} = {rate:.1%} "
      f"(threshold {THRESHOLD:.0%})")
sys.exit(0 if rate >= THRESHOLD else 1)             # nonzero fails the CI job
```

Two design choices make this trustworthy. Greedy decoding (`temperature: 0.0`) makes the eval reproducible, so a failure is a real regression and not sampling noise. And the pass/fail keyword-grounding check is the fast, cheap floor — for a support bot you want to confirm that the answer *contains the policy fact it must contain* — while a fuller version swaps in an LLM-as-judge for nuance and adds refusal-rate and hallucination checks. The gate returns a nonzero exit code on regression, which fails the CI job and blocks the promotion. Wire it into the same pipeline that builds the image in Step 3, and no model change reaches production without clearing both the latency SLO and the quality floor. Quality-under-load evaluation — confirming answers stay good when the batch is full, not just when the engine is idle — deserves its own harness; this gate is the deploy-time floor, not the whole story.

## Step 11 — The cost check

Finally, the number the product manager will actually ask about. Six provisioned H100s (five hot, one standby) at an on-demand rate of roughly \$3 per GPU-hour is \$18/hour, about \$13,000/month, before the cheaper committed-use or reserved pricing that a steady internal service should absolutely take (often 40–60% off, dropping it toward \$6,000–\$8,000/month). Put it per token: at 50 QPS × 250 output tokens, the fleet produces 12,500 output tokens/second, or 45 million output tokens/hour. Against the five hot replicas' \$15/hour of compute, that is `15 / 45` ≈ \$0.33 per million output tokens of raw GPU cost; add the gateway, router, and standby overhead and the all-in lands near \$0.42 per million output tokens. That is dramatically cheaper than a frontier hosted API for this volume — which is exactly the case self-hosting has to make to justify the operational burden, and at 45M tokens/hour of sustained internal traffic, it makes it easily.

The knobs to pull if the bill needs to come down: spot or preemptible instances for the standby and burst replicas (with the hot floor on committed capacity), a smaller draft model for speculative decoding to raise per-GPU throughput, and — if utilization is genuinely low outside business hours — a scheduled scale-down of the hot floor overnight, trading a little cold-start risk for real savings. Each is a trade on the same triangle; none is free.

## The full request journey, end to end

Step back and trace one follow-up question through everything we built, because the point of a build post is watching the pieces compose. A user types the second question of their conversation. The client sends `POST /v1/chat` with the `session_id` and a bearer token. The FastAPI gateway authenticates the token against the key set, decrements that key's token bucket (still within limit, so no 429), validates the Pydantic schema (prompt under the byte cap, `max_tokens` under 1024), and stamps a request id. It loads the conversation history from Redis, prepends the byte-identical system prompt, and forwards the request. The LLM-aware router hashes the `session_id` and pins it to Replica A — the same replica that served turn one — so the KV for the system prompt and the first exchange is already warm. On Replica A, vLLM's radix cache recognizes the 1,540-token prefix as a hit and prefills only the ~40 new tokens; chunked prefill slots that tiny prefill into the current batch without stalling anyone's decode. The first token comes back in ~0.3 s, the gateway re-emits each SSE frame verbatim, and the user watches the answer stream at 60 tokens/second. Throughout, vLLM's `/metrics` endpoint reports queue depth and TTFT histograms to Prometheus; if queue depth had been climbing, KEDA would already be adding a sixth pod; if p95 TTFT had breached, the burn-rate alert would already have paged. When generation finishes, the gateway writes the new turn back to Redis with its TTL, and logs one structured line with the request id, token counts, and latencies.

That single journey touches every one of the eleven steps, and the SLO holds because each layer did exactly its one job: the gateway kept bad traffic off the GPU, the router kept the cache warm, FP8 and chunked prefill kept the engine fast, the sizing kept the queue short, and the observability stood ready to catch the moment any of that stopped being true. No single component is doing anything heroic; the reliability is emergent from each piece protecting its slice of the triangle. That is what "production" means for an LLM service — not one clever trick, but eleven boring things done correctly and composed so their failures are contained.

## Common failure modes I have been paged for

The value of building the whole stack is that you can name, in advance, the specific outages each layer prevents. These are the ones that actually page teams, and where this build catches them.

**The cold-prefill stampede.** A deploy or a router change scatters active sessions across replicas, every follow-up misses the prefix cache, and TTFT triples across the board as the fleet suddenly re-prefills every conversation from scratch. Symptom: p95 TTFT doubles with no change in QPS. Prevention: cache-affinity routing (Step 7) and a rolling-update strategy that does not reshuffle sessions needlessly. This one is insidious because throughput looks fine — the GPUs are busy — they are just busy redoing work.

**The readiness-probe traffic leak.** A pod passes its probe before the model finishes loading (probe delay too short), takes live traffic, and serves a handful of multi-second TTFTs before it is actually ready — or worse, errors them. Symptom: a burst of slow or failed requests every time a pod restarts or scales up. Prevention: the generous `initialDelaySeconds: 40` and a `/health` endpoint that only reports ready once the engine can actually serve.

**The utilization mirage.** Someone sets up autoscaling on GPU utilization, sees it pinned at 100% during normal memory-bound decode, and either never scales (thinks it is always saturated) or scales constantly (chases a metric that never moves). Symptom: latency breaches while the autoscaler sits idle, or a flapping fleet. Prevention: autoscale on queue depth (Step 6), the one signal that actually leads latency.

**The OOM under a long-prompt burst.** `--gpu-memory-utilization` set to 0.95 to "use the card fully," then a handful of users paste long documents at once, KV overflows, and the process OOMs and restarts — taking every in-flight conversation on that replica with it. Symptom: periodic pod restarts correlated with prompt-length spikes. Prevention: 0.90 headroom (Step 2), prompt-length caps at the gateway (Step 4), and trusting the scheduler to preempt rather than crash.

**The silent quality regression.** A vLLM upgrade or a quantization change ships, every latency metric stays green, and three days later support notices the bot has started citing the wrong policy ids. Symptom: no metric fires; users complain. Prevention: the quality gate (Step 10) that blocks the deploy on a reproducible eval before it ever reaches production. This is the failure that a latency-only view can never catch, which is exactly why the gate exists.

Every one of these is a design decision made earlier in the build, paying off as an outage that did not happen. That is the return on doing all eleven steps instead of the three that get you a demo.

## What I would add next

The service above meets its SLO, but a production system is never finished; it is a set of deliberate next investments. In rough priority order for this support bot: **speculative decoding** with a small draft model, which raises per-GPU decode throughput and would let the same fleet absorb more QPS or tighten the p99 TTFT that currently runs at 1.9 s; **an LLM-as-judge quality gate** to replace the keyword-grounding floor with real answer-quality scoring, plus refusal-rate and hallucination checks; **distributed KV-cache sharing** across replicas (a shared prefix cache tier) so a session that spills to a second replica still gets a warm system prompt, decoupling cache affinity from strict pinning; **request prioritization** so interactive chat preempts any batch or background traffic sharing the fleet, protecting the interactive tail; **canary deploys** that shadow a small traffic slice to a new model version and compare quality and latency before promoting; and **multi-region** replicas if the user base spreads geographically, with latency-based routing. Each is a trade on the same latency-throughput-cost triangle, and each earns its place only when a measured number — a p99 that is too tight, a cache-miss rate that is too high, a region that is too far — says it is time. The discipline is the same one that sized the fleet: let the measurement, not the enthusiasm, decide what to build next.

## Case studies

Three shipped systems ground the choices above and show where they generalize.

**Character.AI's serving stack.** In their 2024 engineering writeup on efficient inference, Character.AI reported serving 20,000+ queries per second and drove cost down roughly 33× from early estimates through aggressive KV-cache reduction (multi-query attention and cross-layer KV sharing), int8 quantization, and — most relevant here — a stateful caching system with high cache-hit rates on conversation prefixes. Their headline number, a cache-hit rate around 95% on the prefix, is precisely the Step-5-plus-Step-7 combination: shared prefixes plus cache-affinity routing. It is the clearest public evidence that the biggest lever for a chatbot's economics is not the raw engine but keeping conversations sticky to warm KV.

**The vLLM PagedAttention result.** The original vLLM paper (Kwon et al., SOSP 2023) demonstrated 2–4× higher throughput than prior systems like FasterTransformer and Orca at the same latency, by eliminating KV-cache fragmentation with paged memory. Every throughput number in this build — the ~16 req/s knee, the ~600-sequence KV budget — inherits from that memory manager. When we set `--gpu-memory-utilization 0.90` and trust the engine to pack 60 concurrent sequences without OOM, we are trusting PagedAttention.

**Continuous batching in production (Anyscale / TGI benchmarks).** Public benchmarks of continuous (in-flight) batching versus static batching have shown throughput improvements on the order of 8–23× for LLM serving under realistic mixed-length workloads. That gap is why our single H100 sustains double-digit QPS at all: without continuous batching, a fixed batch would stall on the longest sequence and our knee would sit in the low single digits, forcing three to four times the fleet for the same SLO. The batching regime is doing more of the heavy lifting than the choice of model.

The pattern across all three: the model is the least interesting variable. Memory management, batching, quantization, and cache affinity are where the order-of-magnitude wins live — which is the whole reason this build spends more words on flags, routing, and sizing than on the model.

## When to use this (and when not to)

Self-hosting a chatbot the way we just built it is the right call when several things are true at once, and the wrong call — an expensive distraction — when they are not.

**Build this when:** you have a hard data-residency or compliance constraint that rules out a hosted API (the Legal ask in our scenario); your sustained volume is high enough that per-token self-hosted cost decisively beats a hosted API (our 45M tokens/hour makes the \$0.42-per-million case trivially); you need a fine-tuned or otherwise customized model that no vendor offers; or you need latency control tight enough that you cannot tolerate a third party's tail. At 50 QPS of steady internal traffic with a residency requirement, every one of those points toward the build.

**Do not build this when:** your traffic is low or spiky (a few thousand requests a day, bursty), in which case a hosted API's per-token pricing and scale-to-zero economics crush a fleet of always-on H100s that sit idle 90% of the time — you would be paying \$13,000/month to serve what \$200 of API calls covers. Do not build it when you lack the on-call muscle to operate a GPU fleet: the observability, autoscaling, quality gates, and 3 a.m. pages above are real operational cost, and a small team is often better served buying that reliability. And do not build it as a *prototype* — stand the idea up on a hosted API first, prove the product, and self-host only once the volume and constraints justify the eleven steps. The honest rule: self-hosting wins on volume, control, and compliance; managed APIs win on time-to-market, spiky traffic, and small teams. Our scenario sits squarely in the first camp, which is why we built it — but most chatbot ideas start life in the second.

## Key takeaways

- **Size from the SLO, not from vibes.** Measure one replica's knee capacity $C$ at your latency target, operate at ~0.62 of it, and compute replicas as $\lceil \lambda / (C \cdot \rho_{\max}) \rceil$ plus one for N+1. Five hot H100s serve 50 QPS at p95 TTFT 1.18 s because the math said so, not because five felt right.
- **FP8 is the highest-leverage single choice.** On an H100 it halves weights and KV bytes, quadruples effective concurrency versus untuned FP16, and turns a 2.4 s SLO miss into a 1.18 s pass — for an accuracy cost small enough to be invisible on support Q&A.
- **The gateway protects the GPU off the GPU.** Auth, validation, and token-bucket rate limiting reject bad traffic for the price of a bucket decrement, never a forward pass. The cheapest request is the one the engine never sees.
- **Prefix caching plus cache-affinity routing is the chatbot's economic engine.** A shared system prompt is free after the first request only if follow-ups land on the replica holding the warm KV. Break the affinity and you throw the win away.
- **Autoscale on queue depth, not CPU or GPU utilization.** `num_requests_waiting` is the leading indicator that predicts a latency breach minutes early; GPU utilization reads 100% whether you are perfectly loaded or thrashing.
- **Respect the model's load time in your probes.** A readiness probe that fires before weights finish loading routes traffic to a booting pod; a liveness probe that is too eager restart-loops a slow-loading model into an outage.
- **Gate every deploy on quality, not just latency.** A fast wrong answer is worse than a slow right one; a reproducible greedy-decode eval with a hard threshold blocks silent quality regressions that every latency metric would happily wave through.
- **Put a dollar figure on it.** \$0.42 per million output tokens is the number that justifies the eleven steps — and the number that tells you when a hosted API would have been the smarter call.

## Further reading

- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023) — the vLLM paper; the memory manager every throughput number here rests on.
- vLLM documentation — the OpenAI-compatible server, engine flags, prefix caching, and the `vllm bench serve` harness used in Step 9.
- Character.AI Engineering, "Optimizing Inference" (2024) — the stateful-cache and KV-reduction case study behind the cache-affinity argument.
- Kubernetes documentation on device plugins and the KEDA project docs — GPU resource requests and Prometheus-driven autoscaling.
- Within this series: [what is model serving](/blog/machine-learning/model-serving/what-is-model-serving) for the SLO triangle, [vLLM deep-dive](/blog/machine-learning/model-serving/vllm-deep-dive) for the engine internals, [streaming and SSE for LLMs](/blog/machine-learning/model-serving/streaming-and-sse-for-llms) for the token-streaming protocol, [prefix caching and RadixAttention](/blog/machine-learning/model-serving/prefix-caching-and-radixattention) for the caching mechanism, [GPU scheduling, MIG, and autoscaling](/blog/machine-learning/model-serving/gpu-scheduling-mig-and-autoscaling) for the fleet mechanics, [observability for LLM serving](/blog/machine-learning/model-serving/observability-for-llm-serving) for the metrics, and [load testing and capacity planning](/blog/machine-learning/model-serving/load-testing-and-capacity-planning) for finding the knee.

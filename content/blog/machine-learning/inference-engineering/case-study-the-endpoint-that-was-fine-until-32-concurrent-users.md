---
title: "The endpoint that was fine until 32 concurrent users: Finding queueing collapse"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Diagnose flat-GPU queueing collapse by splitting queue time from service time, then protect p99 with token-aware admission control and chunked prefill."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "latency",
    "throughput",
    "batching",
    "scheduler",
    "goodput",
    "pytorch",
    "cuda",
    "ml-systems",
    "vllm",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 30
---

At one request, the endpoint is quick. At eight, it is still pleasant. At thirty-two, the p99 jumps from “a little slower” to “the product is broken,” while the GPU dashboard remains an unhelpful horizontal line. The obvious response is to increase the batch limit. That can make the incident worse: a long prefill enters the batch, occupies the scheduler’s attention, and makes every short decode wait behind work that was not visible in the utilization average.

![A request branches from queue wait into GPU work and either useful tokens or a timeout](/imgs/blogs/case-study-the-endpoint-that-was-fine-until-32-concurrent-users-1.webp)

The diagram above is the mental model: a request has an arrival clock, a queue clock, a prefill clock, and a decode clock. GPU utilization observes only part of the service path. By the end, we will have a concrete `nanoserve` admission controller, a chunked-prefill scheduler, a queue-time split, an honest load-test protocol, and a symptom → measurement → fix decision path. The key result is not a magic concurrency number. It is a bounded overload policy that makes waiting explicit instead of allowing it to become p99.

This is post #52 in [Inference Engineering](/blog/machine-learning/inference-engineering/what-inference-engineering-is). It extends the scheduler and goodput ideas from [observability for inference goodput, not throughput](/blog/machine-learning/inference-engineering/observability-for-inference-goodput-not-throughput), and it is a case study in the same `nanoserve` engine that ends in [the inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook). The running model is Llama-3.1-8B on one GPU from the fixed series matrix: RTX 4090 24GB for the consumer baseline, with A100 80GB SXM as the larger-memory comparison.

## 1. The incident: nothing looked saturated

The endpoint accepted chat, RAG, code-completion, and translation requests. The single-request path was healthy. A short chat prompt entered prefill, produced a first token, and decoded at a stable cadence. The first load test also looked healthy because it used a closed-loop client: each client submitted a request only after the previous request finished. That test measured the engine’s response when it was allowed to regulate its own arrivals. It did not measure what happens when arrivals continue during a slow tail.

The production symptom was an abrupt p99 change at a concurrency level that looked arbitrary. The number “32” is not a hardware law and should not be copied to another deployment. It is the incident’s load point. The useful question is why a threshold exists at all. A queue appears whenever the long-run arrival rate approaches the service rate, and the tail reacts much more sharply than the mean. If the service rate is $\mu$ requests per second and the arrival rate is $\lambda$, utilization in a simple queueing model is $\rho = \lambda / \mu$. As $\rho$ approaches one, the amount of spare capacity available to absorb variability approaches zero.

That last sentence is a queueing model, not a benchmark result. It tells us what to measure, not what this repository measured. Real LLM serving is harder than an M/M/1 queue because jobs have different prompt lengths, output lengths, KV-cache footprints, cancellation rates, and preemption costs. Still, the direction is dependable: average GPU utilization can be steady while the waiting distribution grows because utilization is an aggregate over busy intervals, not a per-request deadline.

### The wrong first fixes

Three reactions recur in this incident:

1. Increase the maximum batch size. This may raise useful throughput when there is work to combine, but it also admits more work into a shared service interval. It does not bound the waiting queue.
2. Add GPUs without measuring queue time. More capacity can help, but the bottleneck may be a CPU tokenizer, an event-loop lock, a conservative KV budget, or one large prefill monopolizing scheduling decisions.
3. Tune the alert on GPU utilization. A flat utilization trace is compatible with queue growth, host-side admission delay, and decode starvation. It is a capacity clue, not a user-latency explanation.

The incident becomes tractable when we ask two questions for every request: how long did it wait before admission, and how long did admitted work take? TTFT, or time to first token, is the sum of those intervals plus the work between admission and the first token. TPOT, or time per output token, describes the steady decode cadence after the first token. ITL is the interval for an individual pair of output tokens.

## 2. Split TTFT before optimizing it

![A timeline separates queue wait, prefill service, first token, and decode cadence](/imgs/blogs/case-study-the-endpoint-that-was-fine-until-32-concurrent-users-2.webp)

The useful split is:

$$
\text{TTFT} = \text{queue wait} + \text{admission overhead} + \text{prefill service} + \text{first-token handoff}.
$$

This is an explanatory accounting identity for our endpoint. It is not a formula copied from a paper. The terms must be timestamped rather than inferred from GPU utilization. For a request that enters the queue at $t_q$, is admitted at $t_a$, starts prefill at $t_p$, and exposes its first token at $t_f$:

$$
\text{queue_ms} = t_a - t_q, \qquad
\text{prefill\_ms} = t_f - t_p, \qquad
\text{TTFT} = t_f - t_q.
$$

Suppose a worked trace records queue wait of 210 ms, prefill service of 96 ms, and handoff of 0 ms for an intentionally simplified example. Its TTFT is $210 + 96 = 306$ ms. The arithmetic is derived from the displayed trace; it is not a measured result. If the p99 trace contains 210 ms of queue wait and 96 ms of service, optimizing a CUDA kernel that saves 5 ms cannot remove the dominant tail. Admission or scheduling must change first.

The same split catches the opposite failure. If queue wait is 12 ms and prefill is 480 ms for a long RAG prompt, a queue cap alone will reject more work without making admitted work faster. Chunked prefill, prompt limits, prefix reuse, a faster attention backend, or prefill/decode disaggregation may be appropriate. Measurement tells us which branch we are on.

### Instrument the clocks

The first `nanoserve` change is deliberately boring. It records monotonic timestamps at boundaries. A monotonic clock is required for duration arithmetic because wall time may jump during time synchronization. The wall timestamp remains useful for correlating a trace with logs.

```python
# nanoserve/observability.py
from dataclasses import dataclass, field
from time import monotonic_ns


def now_ms() -> float:
    return monotonic_ns() / 1_000_000.0


@dataclass
class RequestTrace:
    request_id: str
    prompt_tokens: int
    max_new_tokens: int
    received_ms: float = field(default_factory=now_ms)
    queue_enter_ms: float | None = None
    admitted_ms: float | None = None
    prefill_start_ms: float | None = None
    first_token_ms: float | None = None
    finish_ms: float | None = None
    output_times_ms: list[float] = field(default_factory=list)

    def mark_queue(self) -> None:
        self.queue_enter_ms = now_ms()

    def mark_admitted(self) -> None:
        self.admitted_ms = now_ms()

    def mark_prefill(self) -> None:
        self.prefill_start_ms = now_ms()

    def mark_token(self) -> None:
        stamp = now_ms()
        self.output_times_ms.append(stamp)
        if self.first_token_ms is None:
            self.first_token_ms = stamp

    def mark_finish(self) -> None:
        self.finish_ms = now_ms()

    @property
    def queue_ms(self) -> float | None:
        if self.admitted_ms is None:
            return None
        start = self.queue_enter_ms or self.received_ms
        return self.admitted_ms - start

    @property
    def ttft_ms(self) -> float | None:
        if self.first_token_ms is None:
            return None
        return self.first_token_ms - self.received_ms

    @property
    def tpot_ms(self) -> float | None:
        if len(self.output_times_ms) < 2:
            return None
        intervals = [b - a for a, b in zip(self.output_times_ms,
                                            self.output_times_ms[1:])]
        return sum(intervals) / len(intervals)
```

The `zip` expression measures adjacent output intervals, not the time from receipt to the first token. That distinction prevents a slow queue from contaminating TPOT. In production, add cancellation, preemption, cache-hit tokens, and deadline fields. The first implementation does not need a metrics vendor to prove the mechanism; a structured record and a percentile script are enough.

### Aggregate by mechanism, not only by percentile

For every load-test window, report p50 and p99 for queue time, prefill time, TTFT, and TPOT. Also report the fraction of completed requests that met the declared SLO. A single p99 TTFT value cannot tell a scheduler engineer whether to change admission or kernels.

| Signal | If it rises while the other stays flat | First suspect | Source |
|---|---|---|---|
| Queue p99 rises; service p99 flat | waiting grows before work starts | admission cap or arrival burst | derived: timestamp difference |
| Service p99 rises; queue p99 flat | admitted work takes longer | prompt length, prefill chunk, kernel | derived: timestamp difference |
| TTFT rises; TPOT flat | first-token path is delayed | queue or prefill | derived: TTFT and TPOT definitions |
| TPOT rises; TTFT flat | decode cadence is worse | active set, long context, kernel | reproduce: `bench_load.py` |
| GPU util flat; queue p99 rises | busy work is not useful capacity | scheduler or host bottleneck | reproduce: `bench_load.py` |
| Raw tok/s rises; goodput falls | throughput is serving late work | overload and deadlines | explanatory abstraction |

The row labels are diagnostics, not empirical claims. A reader can reproduce the last three by running the harness later in the post and comparing the ranges produced on their named GPU.

## 3. Why GPU utilization can stay flat

GPU utilization answers a narrow question: was at least some work resident or executing during sampled intervals? It does not answer whether the right request received service, whether a request was waiting in host memory, whether a kernel was memory-bound, or whether output arrived before cancellation. A 55% utilization trace can mean “there is room for more useful work,” “the engine has unavoidable synchronization gaps,” or “the engine is continuously doing work on a queue that is already too old.” Those cases require different fixes.

The distinction is especially sharp for autoregressive decode. Decode advances one token per active sequence per step. A scheduler can keep the device occupied with a small active set while dozens of requests remain in a waiting queue. If the active set contains long-context sequences, attention reads more KV state per step, and short requests may wait for a scheduling opportunity even though the utilization sample does not fall.

The [vLLM anatomy article](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm), dated 2025-09-05, describes the production pattern as waiting and running queues coordinated with a KV manager, and defines TTFT, ITL, and TPOT separately. That is useful grounding, not a measurement of `nanoserve`. The engineering lesson is to make the same separation visible in our trace.

### Service time is not one kind of work

Prefill consumes the prompt in parallel and builds the initial KV state. Decode consumes one new token at a time, attending to the accumulated KV state. They have different shapes and different fairness needs. A long prefill can be excellent matrix work while still being a poor neighbor for interactive decode because it occupies a scheduling interval whose duration scales with prompt length.

The engine therefore needs a work unit smaller than “one request.” For a prefill request with $P$ uncached prompt tokens, choose a chunk size $C$ and process at most $C$ new prompt tokens in one scheduler step. The number of chunks is:

$$
N_{\text{chunks}} = \left\lceil \frac{P}{C} \right\rceil.
$$

For a derived example with $P=4096$ and $C=256$, $N_{\text{chunks}} = \lceil 4096/256 \rceil = 16$. The chunk count does not claim a kernel duration. It only establishes sixteen possible yield points. If a decode step is scheduled between chunks, the maximum prefill interference is bounded by one chunk plus scheduler overhead rather than by the whole 4096-token prefill.

## 4. The scheduler failure: work entered in the wrong shape

![A monolithic prefill blocks decode while bounded chunks create yield points](/imgs/blogs/case-study-the-endpoint-that-was-fine-until-32-concurrent-users-3.webp)

The broken implementation treats a request as an indivisible object. Admission means “put this request into the running set,” and the running set means “try to process all available prompt work before considering the next policy decision.” That is a reasonable first loop for a toy engine. It is a poor production policy because prompt length is a hidden service-time multiplier.

The corrected implementation treats the scheduler decision as a token budget: each request receives a number of new tokens this step. This resembles the design described in [vLLM V1: A Major Upgrade to vLLM's Core Architecture](https://vllm.ai/blog/2025-01-27-v1-alpha-release), dated 2025-01-27, where a decision is represented as request-to-token work and the scheduler can combine chunked prefill, prefix caching, and speculative work. We build a smaller version and make its limitation explicit: there is no distributed engine, CUDA graph planner, or production KV manager in this post.

### A bounded work decision

```python
# nanoserve/scheduler.py
from dataclasses import dataclass


@dataclass
class WorkItem:
    request_id: str
    prompt_remaining: int
    decode_ready: bool
    max_new_tokens: int
    generated: int = 0

    @property
    def finished(self) -> bool:
        return self.prompt_remaining == 0 and self.generated >= self.max_new_tokens


def allocate_step(running: list[WorkItem], token_budget: int,
                  prefill_chunk: int) -> dict[str, int]:
    """Return new-token work; decode gets a turn before long prefills."""
    plan: dict[str, int] = {}
    remaining = token_budget

    for item in running:
        if remaining == 0:
            break
        if item.decode_ready and item.generated < item.max_new_tokens:
            plan[item.request_id] = 1
            remaining -= 1

    for item in running:
        if remaining == 0:
            break
        if item.prompt_remaining > 0:
            take = min(item.prompt_remaining, prefill_chunk, remaining)
            plan[item.request_id] = plan.get(item.request_id, 0) + take
            remaining -= take

    return plan
```

This function is runnable as a pure Python unit. It gives decode-ready requests first priority, then assigns bounded prefill. That is not universally optimal. A decode-first policy may starve new requests under a permanently full active set. The production policy needs an age or deadline term, and it should record when a request has waited long enough to receive service.

The `token_budget` is a scheduler budget, not a claim about how many tokens a GPU can process. The caller must derive or configure it from the model, dtype, KV capacity, and measured kernel behavior. The function does not touch CUDA and therefore cannot prove throughput. Its value is that policy can be tested independently from kernels.

### Animated figure: why chunking carries meaning

The following motion shows prompt chunks advancing while decode keeps a turn. A still image could show the two states, but it could not show the repeated yield points that bound interference. The reduced-motion state freezes on the safe interleaving state.

<figure class="blog-anim">
<svg viewBox="0 0 720 220" role="img" aria-label="Prompt chunks advance in bounded steps while decode receives repeated turns" style="width:100%;height:auto;max-width:820px"><style>.c1-cell{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}.c1-prefill{fill:var(--accent,#6366f1)}.c1-decode{fill:var(--success,#16a34a)}.c1-label{font:600 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}.c1-active{animation:c1-step 6s steps(4,end) infinite}@keyframes c1-step{0%,20%{transform:translateX(0)}25%,45%{transform:translateX(88px)}50%,70%{transform:translateX(176px)}75%,95%{transform:translateX(264px)}100%{transform:translateX(0)}}@media (prefers-reduced-motion:reduce){.c1-active{animation:none;transform:translateX(176px)}}</style><text class="c1-label" x="360" y="28">scheduler steps</text><rect class="c1-cell" x="40" y="60" width="72" height="60" rx="8"/><rect class="c1-cell" x="128" y="60" width="72" height="60" rx="8"/><rect class="c1-cell" x="216" y="60" width="72" height="60" rx="8"/><rect class="c1-cell" x="304" y="60" width="72" height="60" rx="8"/><rect class="c1-prefill c1-active" x="40" y="60" width="72" height="60" rx="8"/><text class="c1-label" x="76" y="97">P1</text><text class="c1-label" x="164" y="97">P2</text><text class="c1-label" x="252" y="97">P3</text><text class="c1-label" x="340" y="97">P4</text><rect class="c1-decode" x="470" y="60" width="180" height="60" rx="8"/><text class="c1-label" x="560" y="97">decode turn</text><text class="c1-label" x="360" y="170">256-token chunk → yield → decode → next chunk</text></svg><figcaption>Each chunk creates a yield point, so a long prompt cannot hide an unbounded decode delay.</figcaption>
</figure>

The animation is intentionally declarative CSS. There is no script in the post and no claim that the four visual chunks correspond to four kernel launches on every backend. They represent the policy’s four opportunities to yield.

## 5. Admission control: make overload a product decision

![A decision tree maps p99 TTFT to queue caps, chunking, or kernel profiling](/imgs/blogs/case-study-the-endpoint-that-was-fine-until-32-concurrent-users-4.webp)

Admission control answers whether a request may enter the running set now. Backpressure answers what happens when it may not: wait briefly, return a retryable error, route elsewhere, degrade the request, or shed it. Without a policy, the waiting queue becomes an accidental buffer whose size is determined by memory and patience.

The simplest safe rule is to cap both active sequences and estimated token state. Sequence count protects scheduler overhead. Token budget protects KV memory and the work already promised to admitted requests. For request $i$, let $p_i$ be uncached prompt tokens and $o_i$ be the configured maximum new tokens. A conservative reservation is:

$$
R_i = p_i + o_i.
$$

This is a planning quantity, not a claim that every output reaches its maximum. If the engine has a token reservation budget $B$, admit only when:

$$
\sum_{i \in \text{active}} R_i + R_{\text{new}} \le B.
$$

For a derived example with three active reservations of 1024, 2048, and 4096 tokens and a new reservation of 2048, the total is $1024+2048+4096+2048=9216$ tokens. A budget of 8192 rejects or delays the new request; a budget of 12288 admits it with 3072 tokens of headroom. The values are arithmetic examples, not a hardware capacity claim.

The cache is not the entire memory budget. Activations, temporary buffers, CUDA graph pools, weights, and allocator fragmentation consume memory too. A production controller should reserve a safety margin and use actual free-block accounting from the KV manager. Treat the reservation formula as a planning estimate and label it as such in code and metrics.

### Admission implementation

```python
# nanoserve/admission.py
from dataclasses import dataclass
from collections import deque


@dataclass(frozen=True)
class RequestEstimate:
    request_id: str
    prompt_tokens: int
    max_new_tokens: int

    @property
    def reserved_tokens(self) -> int:
        return self.prompt_tokens + self.max_new_tokens


class AdmissionController:
    def __init__(self, max_active: int, token_budget: int, queue_cap: int):
        self.max_active = max_active
        self.token_budget = token_budget
        self.queue_cap = queue_cap
        self.active: dict[str, RequestEstimate] = {}
        self.waiting: deque[RequestEstimate] = deque()

    def _reserved(self) -> int:
        return sum(req.reserved_tokens for req in self.active.values())

    def offer(self, req: RequestEstimate) -> str:
        if len(self.waiting) >= self.queue_cap:
            return "reject_queue_full"
        if (len(self.active) < self.max_active and
                self._reserved() + req.reserved_tokens <= self.token_budget):
            self.active[req.request_id] = req
            return "admit"
        self.waiting.append(req)
        return "wait"

    def finish(self, request_id: str) -> None:
        self.active.pop(request_id)

    def promote(self) -> list[str]:
        promoted: list[str] = []
        while self.waiting and len(self.active) < self.max_active:
            candidate = self.waiting[0]
            if self._reserved() + candidate.reserved_tokens > self.token_budget:
                break
            self.waiting.popleft()
            self.active[candidate.request_id] = candidate
            promoted.append(candidate.request_id)
        return promoted
```

This controller returns `wait` rather than sleeping inside the request handler. That matters: an application thread blocked on an internal queue can consume CPU and make the overload harder to observe. The HTTP layer should attach a deadline, expose queue position only if it can do so honestly, and return a retryable status when the queue cap is reached. A retry policy belongs to the client contract; unbounded automatic retries turn a full queue into a larger arrival rate.

### Fairness and priority

FIFO is easy to explain and can still be unfair when one long RAG request sits at the head. Strict shortest-job-first improves mean latency in some distributions but can starve long requests. Priority can protect interactive chat while violating batch fairness. Deadline-aware scheduling is closer to the product objective but requires reliable estimates and a policy for impossible deadlines.

| Policy | Protects | Failure mode | Good first use |
|---|---|---|---|
| FCFS | arrival order | head-of-line blocking | uniform short chat | 
| shortest estimated work | mean latency | long-job starvation | bounded batch jobs |
| priority | premium or interactive traffic | low-priority starvation | explicit product tiers |
| weighted fair share | tenant fairness | bookkeeping overhead | multi-tenant service |
| deadline-aware | SLO goodput | bad estimates cause churn | known request budgets |

The table describes policy trade-offs, not measured rankings. Choose one and instrument starvation, queue age, and SLO pass rate. A scheduler that lowers p99 for one tier by silently making another tier time out has moved the incident, not fixed it.

## 6. Capacity math for Llama-3.1-8B on the fixed matrix

![A matrix connects workload shape with KV pressure, prefill shape, and admission policy](/imgs/blogs/case-study-the-endpoint-that-was-fine-until-32-concurrent-users-5.webp)

The fixed series model is Llama-3.1-8B. Its model architecture is published in the [official Hugging Face model card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), which should be consulted for the exact checkpoint configuration. For a reproducible arithmetic example, use 32 layers, 8 KV heads, head dimension 128, and bf16’s 2 bytes per scalar. The per-token K and V storage is:

$$
2 \cdot L \cdot H_{kv} \cdot d \cdot b
= 2 \cdot 32 \cdot 8 \cdot 128 \cdot 2
= 131{,}072\ \text{bytes}
= 128\ \text{KiB per token}.
$$

Those architecture values must be checked against the loaded config before using the number for a deployment. The arithmetic is derived from the stated inputs. At 8192 cached tokens, the same estimate is $8192 \times 131{,}072 = 1{,}073{,}741{,}824$ bytes, or 1 GiB. At 4096 tokens it is 512 MiB. This is KV storage only, not total model memory.

The same formula explains why “32 concurrent users” is not portable. If each active request holds 4096 tokens of context under those assumptions, 32 requests need $32 \times 512$ MiB = 16 GiB of KV. If each holds 8192 tokens, they need 32 GiB, which exceeds a 24GB card before weights, activations, and runtime allocations. An A100 80GB SXM has a different headroom picture, but the same queueing logic applies: more memory allows more residency; it does not guarantee bounded latency when arrivals exceed service.

For Qwen3-8B and Gemma-3-12B, do not reuse Llama’s KV number without reading each `config.json`. The number of layers, KV heads, head dimension, and dtype determine the estimate. A small MoE such as Qwen3-30B-A3B has a different weight and routing profile, but its active-parameter label does not remove the scheduler’s need to bound prompt tokens and active state. DeepSeek-V3-family MLA is a cited architecture reference, not part of this post’s first-hand comparison.

#### Worked example: a reservation that does not fit

Assume a derived controller budget of 8192 planned tokens. Three active requests reserve 1024, 2048, and 4096 tokens. A new RAG request reserves 2048. The existing sum is $7168$; adding the new request yields $9216$. The controller returns `wait` even if there is a free sequence slot. This is the important distinction between sequence capacity and token capacity. If one active request finishes and releases 4096, the active sum becomes 3072 and the waiting request can be promoted, leaving $8192 - (3072+2048) = 3072$ planned tokens of headroom. Every quantity in this example is derived arithmetic.

### Fragmentation and blocks

Real engines do not reserve one giant contiguous KV tensor per request. They allocate blocks. The vLLM anatomy article describes a default block size of 16 tokens and a request-to-block mapping, with per-block storage derived as $2 \cdot \text{block size} \cdot H_{kv} \cdot d \cdot b$. With the Llama example, one 16-token bf16 block is:

$$
2 \cdot 16 \cdot 8 \cdot 128 \cdot 2 = 65{,}536\ \text{bytes} = 64\ \text{KiB}.
$$

That is a cited mechanism plus derived arithmetic, not a claim about the exact runtime allocator in this repository. Partial final blocks, prefix sharing, and preemption can change the accounting. Admission should ask the KV manager for allocatable blocks rather than trusting only a token sum once the toy engine grows a paged allocator.

## 7. Chunked prefill in `nanoserve`

Chunked prefill is not “make prefill slower.” It is a scheduling choice: cap how much uncached prompt work one request may inject into a step, then let the policy interleave that work with decode. A chunk that is too small increases launch and bookkeeping overhead. A chunk that is too large recreates the tail. There is no universal 256-token answer; 256 is a concrete starting point for the example and must be swept by the reader.

```python
# nanoserve/prefill.py
from dataclasses import dataclass


@dataclass
class PrefillState:
    request_id: str
    remaining: int
    chunk_size: int

    def take(self, available: int) -> int:
        if self.remaining <= 0 or available <= 0:
            return 0
        amount = min(self.remaining, self.chunk_size, available)
        self.remaining -= amount
        return amount


def make_prefill_plan(states: list[PrefillState], budget: int) -> dict[str, int]:
    plan: dict[str, int] = {}
    left = budget
    for state in states:
        amount = state.take(left)
        if amount:
            plan[state.request_id] = amount
            left -= amount
        if left == 0:
            break
    return plan


if __name__ == "__main__":
    work = [PrefillState("rag", 4096, 256),
            PrefillState("code", 512, 256)]
    print(make_prefill_plan(work, 512))
```

The expected console output is `{'rag': 256, 'code': 256}` for the displayed deterministic inputs. That output is reader-reproducible Python behavior, not a GPU run. A real scheduler should rotate the starting point or use virtual finish time so the first list entry does not monopolize every step. It should also account for prefix-cache hits by reducing `remaining` to uncached tokens.

The implementation boundary is important. The prefill plan says how many tokens to compute; the model runner must build the correct positions, attention metadata, and KV block writes for exactly that slice. It cannot simply truncate `input_ids` and forget that the position offset starts after the cached prefix. A mismatch can produce plausible but wrong text, so add a parity test against an unchunked reference for fixed seeds and logits.

### Chunk-size trade-offs

| Chunk choice | Queue effect | Kernel effect | Failure mode | Source |
|---|---|---|---|---|
| full prompt | worst decode interference | efficient large work | p99 tail | derived: one yield point |
| 1024 tokens | fewer yield points | better amortization | long prompt still blocks | reproduce: `bench_load.py` |
| 256 tokens | bounded interference | more scheduling overhead | launch overhead | reproduce: `bench_load.py` |
| 64 tokens | fine-grained fairness | small, less efficient work | overhead dominates | reproduce: `bench_load.py` |

The table is a test plan, not a performance result. On an RTX 4090, L4, A100 80GB SXM, or H100 80GB SXM, run the same sweep and report the expected range from your own machine rather than copying someone else’s number. The fixed model and prompt suite make the comparison meaningful, but hardware clocks, software versions, and kernel selection still matter.

## 8. Measure open-loop load, not only concurrency

The incident was hidden by a closed-loop test. A proper harness controls arrival behavior independently of completion. An open-loop Poisson generator draws inter-arrival times from an exponential distribution with mean $1/\lambda$ seconds. That is a workload model, not a claim that production arrivals are Poisson. It is useful because it can continue submitting while the queue grows.

```python
# bench_load.py: reader-runnable arrival skeleton
import argparse
import asyncio
import random
import time


async def arrivals(rate: float, duration_s: float):
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        yield time.monotonic()
        await asyncio.sleep(random.expovariate(rate))


async def run(rate: float, duration_s: float):
    async for submitted_at in arrivals(rate, duration_s):
        # Replace this line with an HTTP request carrying a prompt fixture.
        print({"submitted_at": submitted_at, "rate": rate})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=10.0)
    args = parser.parse_args()
    asyncio.run(run(args.rate, args.duration))
```

The skeleton deliberately does not pretend to be a complete benchmark. Replace the marked line with a client that records request ID, response timestamps, cancellations, prompt tokens, output tokens, and the server’s queue split. A valid run needs warmup, a steady-state window, a drain window, fixed model revision, fixed dtype, and a prompt suite containing chat, RAG, code completion, and translation. Record input and output lengths rather than reporting only request count.

For GPU timings, synchronize correctly. CPU wall time around an asynchronous CUDA launch is not kernel duration. Use CUDA events around the operation and call `torch.cuda.synchronize()` at the boundary where you need a complete result. Warm up compilation and allocator paths before the measurement window. If the experiment controls clocks, record that fact; if it does not, report the limitation. [Setting up a reproducible benchmark](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) is the series reference for these controls.

```python
# measure_cuda.py
import torch


def elapsed_ms(fn, warmup: int = 10, repeats: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats
```

This function returns a reader’s device-time result when run with a CUDA function. It does not supply an expected range because the operation and hardware are intentionally parameters. A report should include model, GPU, driver, framework, batch, prompt length, output length, chunk size, arrival rate, p50, p99, queue share, TTFT, TPOT, output tok/s, and goodput.

### What “goodput” means here

Throughput counts work emitted. Goodput counts work that met the endpoint objective. For this case study, define request goodput as SLO-compliant completed requests divided by the measurement window. This is an explanatory product metric, not a universal standard:

$$
G_{req} = \frac{\#\{i : \text{TTFT}_i \le T_{first} \land \text{finish}_i \le T_{deadline}\}}{W}.
$$

If a ten-second window contains 80 completed requests and 68 meet the declared SLO, request goodput is $68/10 = 6.8$ SLO-compliant requests per second. The arithmetic is derived. It is not evidence that this endpoint achieves 6.8 requests per second. Output-token goodput can be defined separately, but do not mix the two in one dashboard label.

## 9. The remediation: reserve, chunk, and shed explicitly

![Open admission creates an unbounded queue while bounded admission makes overload explicit](/imgs/blogs/case-study-the-endpoint-that-was-fine-until-32-concurrent-users-6.webp)

The fix has three cooperating parts:

1. Admission reserves active sequence slots and a token/KV budget.
2. Chunked prefill bounds how much prompt work can delay decode in one decision.
3. A queue cap turns overload into a retryable response or a product-defined degraded mode.

None of the three is sufficient alone. A queue cap with monolithic prefill still lets admitted long prompts hurt active users. Chunked prefill without admission still allows the waiting queue to grow without bound. Admission without a queue cap simply moves the unbounded buffer to the host.

The first rollout should expose policy counters: admitted, queued, rejected, promoted, deadline-expired, canceled, preempted, and completed. Include the estimated reservation and actual prompt/output tokens. If the controller rejects 20% of requests while goodput improves, that may be a valid overload response; if it rejects short requests behind one oversized request, the policy needs fairness or a maximum request budget.

### A simple policy loop

```python
# nanoserve/engine_loop.py
def scheduler_step(admission, running, waiting, token_budget, chunk_size):
    # Finish requests before promoting new work so reservations are released.
    for req in list(running):
        if req.finished:
            admission.finish(req.request_id)
            running.remove(req)

    for request_id in admission.promote():
        request = waiting.pop(request_id)
        running.append(request)

    plan = allocate_step(running, token_budget, chunk_size)
    for req in running:
        assigned = plan.get(req.request_id, 0)
        if req.prompt_remaining:
            req.prompt_remaining -= min(assigned, req.prompt_remaining)
            req.decode_ready = req.prompt_remaining == 0
        elif assigned:
            req.generated += assigned
    return plan
```

This sketch assumes `waiting` is an indexed container and `finished` is maintained by the model runner. It is intentionally not a drop-in vLLM replacement. The valuable invariant is that every step has a bounded token plan and every promotion follows the same reservation check. Add assertions: no assigned work exceeds remaining prompt tokens, no request decodes before prefill completion, and active reservations never exceed the configured budget.

### Failure modes after the fix

Admission control can fail in less obvious ways:

- A client retries every 429 immediately. The endpoint’s effective arrival rate increases. Use exponential backoff with jitter and a retry budget.
- The maximum output reservation is enormous. A request that usually emits 20 tokens can block admission as if it will emit 8192. Use a product limit, a calibrated estimate, or a two-stage reservation with explicit risk.
- A prefix cache hit is not subtracted. The controller under-admits and leaves capacity idle. Count uncached tokens at admission, then reserve output and a small uncertainty margin.
- A request disconnects but its GPU work continues. Cancellation must remove future decode work and release blocks as soon as the runner reaches a safe point.
- A priority queue never ages. Low-priority work starves and its goodput becomes zero. Record maximum queue age by class.
- Chunking splits positions incorrectly. Text still looks plausible, so tests must compare logits or token IDs against the unchunked path.
- A queue cap is shared across tenants. One noisy tenant can consume all waiting slots. Partition or apply weighted quotas.

The correct response is not to hide these cases behind a larger timeout. Expose them as counters and make the product choice visible.

## 10. A reproducible comparison plan

![A grid joins arrival rate and prompt shape to queue split, tail latency, and goodput](/imgs/blogs/case-study-the-endpoint-that-was-fine-until-32-concurrent-users-7.webp)

Run four configurations on one fixed model and one fixed GPU at a time:

| Configuration | Admission | Prefill | What it isolates | Source |
|---|---|---|---|---|
| baseline | open | monolithic | original incident | reproduce: `bench_load.py` |
| cap only | bounded | monolithic | queue protection | reproduce: `bench_load.py` |
| chunks only | open | 256-token chunks | interference control | reproduce: `bench_load.py` |
| combined | bounded | swept chunks | production candidate | reproduce: `bench_load.py` |

Sweep arrival rate until the SLO pass rate falls below the product target. Do not call the crossing point “maximum throughput” unless the workload, SLO, and arrival model are named. For each point, run the same prompt fixtures and keep output limits fixed. Use at least one run for warmup and a separate steady-state window; the exact repetition count is a reader-controlled experimental parameter.

The expected qualitative range is reproducible rather than claimed: baseline should show queue p99 increasing first when admission is open and arrivals exceed service; cap-only should bound queue age but leave long-prefill TPOT interference; chunks-only should reduce decode disruption while the waiting queue can still grow; combined should make overload visible through rejections or bounded waits. If your trace does not show those relationships, inspect the harness before declaring the policy ineffective. A closed-loop client, a too-short window, or a prompt fixture with no length variance can erase the mechanism.

### Hardware comparison without invented benchmarks

The fixed matrix contains RTX 4090 24GB, L4 24GB, A100 80GB SXM, and H100 80GB SXM. The point of repeating the protocol on all four is not to publish a made-up tok/s table. It is to separate capacity effects from policy effects. The RTX 4090 and L4 have the same nominal memory capacity in the matrix but different server roles and memory systems; the A100 and H100 have more memory headroom, yet a long prompt can still monopolize a scheduler step. Cite the vendor specification for any bandwidth or memory number you report, and label local measurements as `reproduce: bench_load.py`.

| Quantity | How to derive or obtain it | Allowed report |
|---|---|---|
| KV bytes/token | model config and $2LdH_{kv}b$ | derived arithmetic |
| active reservation | sum of per-request estimates | derived arithmetic |
| queue p99 | admission minus queue-enter timestamps | reproduce: trace export |
| prefill p99 | first-token path minus prefill start | reproduce: trace export |
| TPOT | mean adjacent output intervals | reproduce: trace export |
| GPU memory | `torch.cuda.max_memory_allocated()` plus environment | reproduce: named script |
| GPU bandwidth | vendor datasheet | cited: vendor specification |
| goodput | SLO-compliant completions / window | explanatory definition + reproduce |

This is the honesty boundary for the case study. There is no first-hand GPU run behind the prose. Every number above is either derived from shown inputs, cited to a source, or assigned to a reader-runnable measurement with an expected qualitative range. Do not turn the worked examples into benchmark claims when adapting this post.

## Case studies and public grounding

### 1. PagedAttention made the memory problem explicit

The [PagedAttention paper](https://arxiv.org/abs/2309.06180), published in 2023, frames KV-cache memory management as a paging problem and reports large throughput improvements for its evaluated serving system relative to the baselines in its experiments. Those reported improvements belong to the paper’s models, hardware, software, and workload; they are not transferable measurements for `nanoserve`. The mechanism is directly relevant here: when KV state is allocated in blocks, the scheduler can reason about physical capacity and reuse rather than pretending every request needs one contiguous reservation.

The paper does not make admission control optional. Better memory packing can increase the number of requests that fit, which can increase useful capacity, but admitting work beyond the service rate still creates a queue. Memory efficiency changes the capacity curve; it does not remove queueing collapse.

### 2. vLLM’s scheduler unifies different token work

The [vLLM V1 architecture note](https://vllm.ai/blog/2025-01-27-v1-alpha-release), dated 2025-01-27, describes request-to-token scheduling and the ability to combine chunked prefill with other engine features. We cite it as a production design reference. Our `nanoserve` plan is intentionally smaller: it gives decode priority, caps prefill work, and uses a reservation budget, but it does not implement the full engine-core architecture or its optimized kernels.

### 3. Goodput is an SLO decision, not a GPU counter

The [vLLM anatomy article](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm), dated 2025-09-05, names TTFT, ITL, and TPOT as distinct latency concepts and describes benchmark setups with explicit input/output lengths and concurrency. That naming discipline is the useful lesson for this incident. A result that reports only output tok/s cannot tell a reader whether a request waited, whether decode was smooth, or whether the response arrived before its deadline.

### 4. A benchmark protocol is part of the result

The [MLPerf Inference rules](https://github.com/mlcommons/inference_policies) and the repository’s [reproducible benchmark guide](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) illustrate the broader principle: a number without scenario, system, and measurement boundaries is not portable evidence. For this case study, the scenario must include arrival behavior and prompt shape. “32 concurrent users” is a symptom label, not a universal capacity specification.

## When to reach for this, and when not to

Reach for token-aware admission and chunked prefill when:

- queue p99 is the dominant part of TTFT;
- GPU utilization is flat while waiting requests age;
- prompt lengths vary enough that one request can monopolize a step;
- interactive decode must be protected from RAG or code-prefill bursts;
- the product can express a retry, shed, or degraded-mode contract.

Do not add this policy merely because a single-request benchmark is slow. If service time dominates while queue time is negligible, profile the model runner, attention backend, memory transfers, and tokenizer first. Do not chunk aggressively when prompts are uniformly tiny and the added launch overhead consumes the available budget. Do not build a scheduler from scratch for a production endpoint just to rediscover these invariants: use vLLM, SGLang, TensorRT-LLM, or another established engine unless you need a research surface, a specialized policy, or a learning exercise. The value of `nanoserve` is making the mechanism inspectable.

The strongest recommendation is conditional: first instrument the split, then choose the smallest fix that attacks the dominant interval. If queue time is large, cap admission. If prefill service is large, chunk or optimize prefill. If decode TPOT is large, inspect active context, KV layout, and kernels. If goodput is low despite stable raw throughput, change the overload contract rather than celebrating utilization.

## Key takeaways

1. A flat GPU-utilization line cannot prove that the endpoint has spare interactive capacity.
2. Split TTFT into queue wait, admission, prefill, and handoff before tuning kernels.
3. Treat prompt tokens and output reservations as capacity, not just active sequence count.
4. Chunked prefill creates yield points; it does not magically increase raw compute speed.
5. A bounded queue is a product decision: wait, retry, shed, route, or degrade explicitly.
6. Protect decode-ready requests from a long prefill, then add fairness so new work cannot starve.
7. Report queue p99, prefill p99, TTFT, TPOT, goodput, and raw throughput together.
8. Use open-loop load when testing overload; closed-loop load can hide queue growth.
9. Label every number as derived, cited, or reader-reproducible with a named script.
10. Build the policy in `nanoserve` to understand it, then use a production engine when the endpoint matters more than the lesson.

## Further reading

- [Inside vLLM: Anatomy of a High-Throughput Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm), vLLM, 2025-09-05.
- [vLLM V1: A Major Upgrade to vLLM's Core Architecture](https://vllm.ai/blog/2025-01-27-v1-alpha-release), vLLM, 2025-01-27.
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180), 2023.
- [Observability for inference goodput, not throughput](/blog/machine-learning/inference-engineering/observability-for-inference-goodput-not-throughput).
- [Request scheduling and preemption](/blog/machine-learning/model-serving/request-scheduling-and-preemption).
- [Prefill-decode disaggregation](/blog/machine-learning/model-serving/prefill-decode-disaggregation).
- [The inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook).

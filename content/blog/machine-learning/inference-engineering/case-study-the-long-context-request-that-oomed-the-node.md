---
title: "The long-context request that OOMed the node: Memory-aware admission for prefill"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Derive the memory spike behind a 200k-token prefill, then add chunked prefill, context-aware admission, per-request budgets, and graceful failure to nanoserve."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "kv-cache",
    "batching",
    "latency",
    "throughput",
    "pytorch",
    "cuda",
    "ml-systems",
    "vllm",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 28
---

The endpoint looked healthy until one customer pasted a 200,000-token transcript. Eight ordinary requests were decoding. Their KV blocks were resident. The long request passed a maximum-context check, entered prefill, and turned a small positive memory margin into a device-wide out-of-memory failure. The visible error named CUDA. The actual bug was admission control.

![A memory budget separates weights, runtime reserve, KV cache, prefill activations, and the negative headroom that causes eviction](/imgs/blogs/case-study-the-long-context-request-that-oomed-the-node-1.webp)

The diagram above is the mental model: GPU memory is a budget with four claimants, and the transient claimant is the one most serving loops omit. By the end of this post we will derive the KV number, write a peak-memory estimator, split prefill into bounded chunks, reserve memory before admission, and return a useful error when the request cannot fit. The implementation lands in the `nanoserve` scheduler rather than in a model layer.

This is a case study, not a report of a first-hand production run. The incident is a constructed scenario whose constants are either derived below, cited inline, or exposed as reader-reproducible checks. No GPU run is claimed. The fixed reference is Llama 3.1 8B on one RTX 4090 / A100, with the long input intentionally set to 200,000 tokens. Meta's [Llama 3.1 model card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), dated 2024-07-23, lists a 128k context length for the 8B model, so a 200k request is also a useful policy failure: a node must reject it before trying to make a kernel process an unsupported context.

## 1. The symptom is an allocation policy bug

The first bad assumption is that a request's memory is the KV cache it will eventually leave behind. That is only the steady-state part. Prefill also creates intermediate tensors: hidden states, attention tiles or workspaces, temporary projections, and allocator reservations. Some are released layer by layer; some remain live across a fused operation; the exact peak depends on the kernel and implementation. The scheduler cannot safely infer the peak from the final KV size alone.

The second bad assumption is that `nvidia-smi`'s free number is a reservation that belongs to the next request. In a serving process, CUDA context state, PyTorch's caching allocator, graph pools, kernel workspaces, NCCL buffers, and fragmentation all compete for the same device. PyTorch exposes both allocated and reserved memory through [its CUDA memory API](https://docs.pytorch.org/docs/stable/cuda), but neither counter is a promise that an arbitrary future allocation will succeed.

The third bad assumption is that a maximum context check is admission control. A context limit answers “is this input valid for the model?” Admission must answer a different question: “can this input run now without invalidating work already accepted?” The former is static model metadata. The latter is a live capacity decision.

| Assumption | What it misses | Safer rule | Source |
|---|---|---|---|
| Context limit equals memory limit | transient activations and runtime reserve | estimate peak bytes before launch | derived policy |
| Free VRAM belongs to the next request | allocator reservation and live KV blocks | keep a protected reserve | derived policy |
| OOM can be caught and retried | the failed allocation may have disrupted a batch | reject or defer before launch | PyTorch CUDA semantics |
| Long prefill is one indivisible operation | it monopolizes the scheduler and spikes memory | make prefill a sequence of budgeted chunks | explanatory abstraction |

> A request is not admitted because it is valid. It is admitted because its worst live byte has a reservation.

The repair has four pieces. First, count steady-state KV bytes. Second, add a conservative activation reserve for the chunk being launched. Third, account for live requests and allocator headroom. Fourth, make the response policy explicit: admit, chunk, defer, truncate with consent, or reject. We will keep decoding work protected even when a new long-context request is interesting.

## 2. Derive the memory number before choosing a policy

Llama 3.1 8B uses grouped-query attention. For the memory calculation, the relevant quantities are the number of transformer layers $L$, key/value heads $H_{kv}$, head dimension $d$, bytes per scalar $b$, and sequence length $S$. Each token stores both a key and a value, so the explanatory KV-cache law is:

$$
M_{KV}(S) = S \cdot L \cdot H_{kv} \cdot d \cdot 2 \cdot b.
$$

This is the engine's byte-count abstraction; it is not a formula quoted as a single equation in the model card. For Llama 3.1 8B, the model configuration exposes $L=32$, $H_{kv}=8$, and $d=128$. In bf16, $b=2$ bytes. Therefore:

$$
32 \cdot 8 \cdot 128 \cdot 2 \cdot 2 = 131{,}072\ \text{bytes/token} = 128\ \text{KiB/token}.
$$

At 8,192 tokens, the arithmetic is $8{,}192 \times 131{,}072 = 1{,}073{,}741{,}824$ bytes, or exactly 1 GiB under the binary conversion. At 200,000 tokens, it is $200{,}000 \times 131{,}072 = 26{,}214{,}400{,}000$ bytes, or about 24.41 GiB. The decimal 25.6 GB figure is the same byte count divided by $10^9$; the difference is units, not a second estimate.

![A fixed token block maps to physical KV storage and makes the 200k-token reservation arithmetic visible](/imgs/blogs/case-study-the-long-context-request-that-oomed-the-node-6.webp)

The number already exceeds the nominal 24 GB capacity of an RTX 4090 before weights, runtime state, or activations. It also exceeds the Llama 3.1 8B model's documented 128k context. On an A100 80GB, 24.41 GiB of KV is mathematically possible, but only after weights and the runtime have been placed. The model card's context limit still makes 200k an application-level error unless the deployment explicitly uses a supported context-extension configuration.

### The 128k boundary is not a magic allocator boundary

At the documented 128k context, the same derived KV number is $131{,}072 \times 131{,}072 = 17{,}179{,}869{,}184$ bytes, or 16 GiB. That is a clean reference point, not a safe admission value. A bf16 checkpoint's raw parameter storage is approximately $8\times10^9\times2 = 16\times10^9$ bytes, or about 14.9 GiB, before metadata and any non-weight tensors. The exact checkpoint footprint depends on serialization and runtime representation, so the parameter arithmetic is an estimate, not a measured load.

On an 80 GiB A100, a deliberately simple upper-bound ledger could reserve 16 GiB for weights, 4 GiB for runtime and fragmentation, 16 GiB for a 128k KV cache, and 8 GiB for a prefill chunk's activation envelope. That sum is 44 GiB, leaving 36 GiB of uncommitted capacity in the hypothetical ledger. The ledger is useful because every term is visible. It is not a promise of a specific kernel's peak.

#### Worked example: why the long request is rejected on the consumer baseline

Suppose the deployment uses the fixed Llama 3.1 8B configuration in bf16 and an RTX 4090 with 24 GB of advertised VRAM. The derived 200k KV requirement is 26.2144 billion bytes, while the entire device advertises 24 billion bytes in decimal units. The request fails the KV-only check: $26.2144 - 24 = 2.2144$ billion bytes short before weights. No activation measurement is needed to reject it, and attempting the prefill would only turn a predictable 413 response into a process-wide CUDA error. Source: derived arithmetic; device capacity is a cited class specification from [NVIDIA's GeForce RTX 4090 page](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/), accessed 2026-08-04.

On an A100 80GB, the decision changes but does not disappear. If weights use an estimated 16 GiB, runtime reserve is 4 GiB, and eight active requests collectively hold 20 GiB of KV, a 200k request's 24.41 GiB KV footprint leaves $80 - 16 - 4 - 20 - 24.41 = 15.59$ GiB for activations and allocator slack. If the policy's conservative activation envelope is 8 GiB, it may fit; if the service must preserve an additional 12 GiB decode reserve, it must defer. The same input is legal in one state and illegal in another because admission is stateful.

## 3. Chunked prefill bounds the transient spike

Prefill computes logits for a prompt that already exists. Decode computes one next token at a time. A naïve scheduler launches all 200,000 prompt tokens as one prefill batch. That choice creates a large temporary working set, delays the next decode step, and makes the allocator face one enormous request. Chunked prefill changes the unit of work: the scheduler submits, for example, 8,192 prompt tokens, releases chunk-local intermediates, services eligible decodes, then submits the next chunk.

![The same long request changes from one unbounded prefill into bounded chunks that leave decode work protected](/imgs/blogs/case-study-the-long-context-request-that-oomed-the-node-2.webp)

The chunk size is not a universal constant. A larger chunk improves matmul efficiency and reduces scheduling overhead, but raises activation peak and delays decodes. A smaller chunk lowers the memory envelope and improves interleave granularity, but can make launch and synchronization overhead visible. Pick it from the memory budget and validate it with a reproducible latency sweep.

The animated figure below shows the scheduling relationship, not a measured timing. The moving prefill window is intentionally interleaved with decode steps. A still image could show two queues, but it could not show why a long prefill must give the decode loop turns.

<figure class="blog-anim">
<svg viewBox="0 0 720 210" role="img" aria-label="Prefill chunks advance across the prompt while decode steps keep receiving scheduler turns" style="width:100%;height:auto;max-width:820px">
<style>
.lc-stage{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}.lc-txt{font:600 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}.lc-small{font:500 14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}.lc-chunk{fill:var(--accent,#6366f1);opacity:.82}.lc-decode{fill:var(--text-secondary,#6b7280)}
@keyframes lc-sweep{0%,8%{transform:translateX(0)}24%,34%{transform:translateX(116px)}50%,60%{transform:translateX(232px)}76%,86%{transform:translateX(348px)}100%{transform:translateX(464px)}}
@keyframes lc-pulse{0%,18%{opacity:.22}24%,34%{opacity:1}50%,60%{opacity:.22}76%,86%{opacity:1}100%{opacity:.22}}
.lc-mv{animation:lc-sweep 12s ease-in-out infinite}.lc-p{animation:lc-pulse 12s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.lc-mv,.lc-p{animation:none}.lc-mv{transform:translateX(232px)}.lc-p{opacity:1}}
</style>
<text class="lc-txt" x="360" y="28">one scheduler cycle</text>
<rect class="lc-stage" x="24" y="56" width="560" height="54" rx="9"/><text class="lc-small" x="304" y="88">long prompt · 200k logical tokens</text>
<rect class="lc-chunk lc-mv" x="30" y="62" width="104" height="42" rx="7"/><text class="lc-txt" x="82" y="88">chunk</text>
<rect class="lc-stage" x="24" y="140" width="560" height="42" rx="9"/><rect class="lc-decode lc-p" x="42" y="149" width="74" height="24" rx="5"/><rect class="lc-decode" x="142" y="149" width="74" height="24" rx="5"/><rect class="lc-decode lc-p" x="242" y="149" width="74" height="24" rx="5"/><rect class="lc-decode" x="342" y="149" width="74" height="24" rx="5"/><rect class="lc-decode lc-p" x="442" y="149" width="74" height="24" rx="5"/>
<text class="lc-small" x="620" y="166">decode turns</text>
</svg>
<figcaption>Prefill advances in bounded chunks while the scheduler gives decode requests recurring turns.</figcaption>
</figure>

### A runnable chunk planner

The first implementation does not need CUDA. It needs a deterministic contract between the scheduler and the model runner. The planner below emits chunk boundaries and makes the last partial chunk explicit.

```python
# nanoserve/prefill.py
from dataclasses import dataclass


@dataclass(frozen=True)
class PrefillChunk:
    request_id: str
    start: int
    stop: int

    @property
    def tokens(self) -> int:
        return self.stop - self.start


def chunks(request_id: str, prompt_tokens: int, chunk_size: int):
    if prompt_tokens < 1:
        raise ValueError("prompt must contain at least one token")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    for start in range(0, prompt_tokens, chunk_size):
        yield PrefillChunk(request_id, start, min(start + chunk_size, prompt_tokens))


if __name__ == "__main__":
    plan = list(chunks("long-17", 200_000, 8_192))
    print(len(plan), plan[0].tokens, plan[-1].tokens)
```

The reader can run this on any Python installation. The expected output is `25 8192 3392`, derived from $\lceil200{,}000/8{,}192\rceil=25$ and $200{,}000 - 24\times8{,}192 = 3{,}392$. This script tests scheduling arithmetic only; it is not a GPU benchmark.

The runner must keep the request's position offset. Each chunk attends to the previous KV blocks plus the current chunk, writes only the newly produced K/V, and returns a final hidden state for the next stage. It must not reset `position_ids` to zero at every chunk. That bug produces plausible but wrong text, which is harder to detect than an OOM.

```python
def run_prefill(request, model_runner, scheduler, chunk_size):
    offset = 0
    while offset < len(request.input_ids):
        stop = min(offset + chunk_size, len(request.input_ids))
        token_slice = request.input_ids[offset:stop]
        position_ids = range(offset, stop)
        scheduler.before_prefill_chunk(request, stop - offset)
        model_runner.prefill(
            request_id=request.id,
            input_ids=token_slice,
            position_ids=position_ids,
            kv_block_table=request.kv_block_table,
        )
        offset = stop
        scheduler.after_prefill_chunk(request)
```

This is a runnable-shaped integration seam, not a claim that a reader can paste it into a missing repository API. The exact model-runner implementation belongs to the earlier forward-pass and KV-cache posts. The important invariants are testable: offsets are monotonic, the block table is stable, and `after_prefill_chunk` is a scheduling point.

## 4. Context-aware admission protects live decodes

Chunking controls the transient size of one work item. It does not decide whether the complete request can ever fit. A 200k request split into 25 chunks still needs a final KV cache proportional to 200k. Admission therefore needs two estimates:

1. `kv_bytes_after`: the request's eventual KV allocation after its prompt and output budget are accounted for.
2. `peak_bytes_now`: the bytes that must be available for the next chunk, including its activation reserve.

The scheduler compares the sum of these values with a protected device budget. The exact activation term is deployment-specific. A useful conservative interface is a function of chunk tokens, batch tokens, dtype, and runner profile, rather than a hard-coded percentage of VRAM.

![Request length and live work converge on a peak-byte check before the scheduler chooses chunk, defer, or reject](/imgs/blogs/case-study-the-long-context-request-that-oomed-the-node-5.webp)

The graph has one deliberate asymmetry: a request can be valid but not fit. That branch must be visible in the code and in the API response. “Try and catch CUDA OOM” is too late because the allocator may have already split blocks and the scheduler may have already promised work to the client.

### A byte estimator with explicit units

```python
# nanoserve/memory.py
from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class ModelMemory:
    layers: int
    kv_heads: int
    head_dim: int
    bytes_per_value: int
    bytes_per_block_token: int = 16

    @property
    def kv_bytes_per_token(self) -> int:
        return (self.layers * self.kv_heads * self.head_dim
                * 2 * self.bytes_per_value)

    def kv_bytes(self, tokens: int) -> int:
        return tokens * self.kv_bytes_per_token

    def kv_blocks(self, tokens: int) -> int:
        return ceil(tokens / self.bytes_per_block_token)


LLAMA31_8B_BF16 = ModelMemory(32, 8, 128, 2)

if __name__ == "__main__":
    m = LLAMA31_8B_BF16
    print(m.kv_bytes_per_token, m.kv_bytes(200_000), m.kv_blocks(200_000))
```

The expected values are `131072 26214400000 12500`. The first two values follow the derivation above. The block count is $\lceil200{,}000/16\rceil=12{,}500$ with a 16-token block. The block size is an engine choice, not a Llama model fact. The grid figure shows why the allocator should reserve blocks rather than pretend that a prompt is one contiguous tensor.

For the complete peak estimate, keep every term named:

$$
M_{\text{peak}} = M_{\text{weights}} + M_{\text{runtime}} + M_{\text{live KV}} + M_{\text{request KV}} + M_{\text{chunk act}} + M_{\text{slack}}.
$$

This is an explanatory accounting identity. It is not a claim that PyTorch exposes each term perfectly. The value is operationally useful because the scheduler can reject an estimate that exceeds policy even if the allocator would sometimes succeed. False negatives waste capacity; false positives evict users. For an interactive service, protecting accepted work is usually the right bias.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Admission:
    capacity_bytes: int
    weights_bytes: int
    runtime_reserve_bytes: int
    live_kv_bytes: int
    slack_bytes: int
    activation_bytes_per_token: int
    chunk_tokens: int

    def available_for_request(self) -> int:
        return (self.capacity_bytes - self.weights_bytes
                - self.runtime_reserve_bytes - self.live_kv_bytes
                - self.slack_bytes)

    def peak_for_chunk(self, request_kv_bytes: int) -> int:
        chunk_act = self.activation_bytes_per_token * self.chunk_tokens
        return request_kv_bytes + chunk_act

    def can_admit(self, request_kv_bytes: int) -> bool:
        return self.peak_for_chunk(request_kv_bytes) <= self.available_for_request()


policy = Admission(
    capacity_bytes=80 * 2**30,
    weights_bytes=16 * 2**30,
    runtime_reserve_bytes=4 * 2**30,
    live_kv_bytes=20 * 2**30,
    slack_bytes=4 * 2**30,
    activation_bytes_per_token=32 * 1024,
    chunk_tokens=8_192,
)
print(policy.can_admit(LLAMA31_8B_BF16.kv_bytes(128_000)))
```

The illustrative activation coefficient is a policy input, not a measured property of Llama. The example's available budget is $80-16-4-20-4=36$ GiB. The request KV is 16 GiB, and the chunk activation allowance is $32\times1024\times8192=256$ MiB, so the boolean is `True` under this deliberately conservative ledger. A production runner should replace the coefficient with a profile captured on the target kernel and keep a margin for allocator behavior.

## 5. Per-request budgets stop one customer owning the node

Global capacity is necessary but insufficient. If the scheduler admits a long request whenever global free bytes are positive, the long request can consume the entire remaining budget and leave no room for already accepted output tokens. Per-request budgets give each request a second ceiling: a maximum prompt-plus-output KV allocation and a maximum chunk activation reservation.

The budget should be attached to the request at admission, not reconstructed from a log after failure. It must include prompt tokens, maximum new tokens, block rounding, and any shared-prefix treatment. For a request with prompt length $P$, generation cap $G$, and block size $B$, reserve $\lceil(P+G)/B\rceil$ KV blocks. If the API permits unbounded generation, the correct budget is not infinite; it is a product-level limit.

| Policy field | Meaning | Example | Source |
|---|---|---:|---|
| `max_input_tokens` | request validation ceiling | 128,000 | cited: Llama 3.1 model card |
| `max_new_tokens` | output reservation ceiling | 1,024 | API policy |
| `chunk_tokens` | transient prefill unit | 8,192 | policy, reader-tunable |
| `kv_block_tokens` | allocator granularity | 16 | engine choice |
| `protected_decode_bytes` | capacity unavailable to new prefill | 6 GiB | policy |
| `activation_bytes_per_token` | chunk peak approximation | 32 KiB | reproduce: profile script |

```python
from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class RequestBudget:
    prompt_tokens: int
    max_new_tokens: int
    block_tokens: int
    kv_bytes_per_token: int
    activation_bytes: int

    @property
    def total_tokens(self):
        return self.prompt_tokens + self.max_new_tokens

    @property
    def blocks(self):
        return ceil(self.total_tokens / self.block_tokens)

    @property
    def kv_bytes(self):
        return self.blocks * self.block_tokens * self.kv_bytes_per_token

    @property
    def peak_bytes(self):
        return self.kv_bytes + self.activation_bytes


budget = RequestBudget(128_000, 1_024, 16, 131_072, 256 * 2**20)
print(budget.blocks, budget.kv_bytes / 2**30, budget.peak_bytes / 2**30)
```

The expected block count is $\lceil129{,}024/16\rceil=8{,}064$. Because the total is divisible by 16, the reserved KV bytes are $8{,}064\times16\times131{,}072 = 16{,}911{,}433{,}728$ bytes, about 15.75 GiB. That last multiplication should still be performed by the script rather than memorized: the code prints the exact floating value after dividing by $2^{30}$, and the reader can change the block size or dtype without trusting a copied rounded number. The point is that block rounding is visible and charged to the request.

A subtle choice is whether to reserve all output KV up front. Reserving all of it is safest but can reject requests that would finish early. Reserving only the prompt and a small output allowance increases utilization but needs a reservation growth operation at every decode step. That operation must be atomic with the scheduler's batch decision. If it cannot grow, the request enters a documented degraded path; it must not silently evict another request.

## 6. The scheduler needs a state machine, not an OOM handler

The long request has at least six states: `received`, `validated`, `reserved`, `prefilling`, `decoding`, and one terminal outcome. A request that fails the model context check is not the same as one that is valid but waiting for capacity. The client should receive different error codes and different retry advice.

![A timeline separates validation, admission, memory spike, eviction, and the stable fail-fast outcome](/imgs/blogs/case-study-the-long-context-request-that-oomed-the-node-3.webp)

The incident's most important timestamp is the admission decision. By the time an allocator emits an OOM, the system has already chosen the wrong state transition. A robust scheduler makes the transition conditional on a reservation token. The runner consumes that token for each chunk; the allocator returns it when the chunk's temporary tensors die; the request's KV reservation persists until completion or cancellation.

```python
from enum import Enum


class State(Enum):
    RECEIVED = "received"
    REJECTED = "rejected"
    WAITING = "waiting"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    COMPLETE = "complete"
    FAILED = "failed"


def admit(request, memory, budget):
    if request.input_tokens > budget.max_input_tokens:
        return State.REJECTED, "input exceeds model policy"
    needed = request.estimated_peak_bytes
    if not memory.try_reserve(needed, owner=request.id):
        return State.WAITING, "capacity unavailable; retry or queue"
    request.reservation = needed
    return State.PREFILLING, "reserved"


def fail_response(state, request_id):
    if state == State.REJECTED:
        return {"status": 413, "code": "context_limit", "request_id": request_id}
    if state == State.WAITING:
        return {"status": 429, "code": "capacity_deferred", "request_id": request_id}
    return {"status": 500, "code": "inference_failed", "request_id": request_id}
```

This snippet deliberately distinguishes 413 from 429. HTTP 413 says the request is too large for the service's declared policy. HTTP 429 says it may fit later but is not admitted now. Whether an API chooses those exact codes is a contract decision; the key is that clients can stop retrying an impossible request and can back off from temporary pressure.

### Preemption is a last resort, not a memory strategy

If the scheduler cannot fit a new request, it can queue it, reject it, truncate it with explicit consent, spill it to another capacity pool, or preempt an existing request. Preemption may save a node, but it is not free. With KV recompute, the prompt is processed again. With swap, device-to-host transfer and host memory become part of latency. With an active stream, the client may see a pause or cancellation.

The series' earlier [request scheduling and preemption](/blog/machine-learning/model-serving/request-scheduling-and-preemption) discussion is the right place for the general policy. This incident adds a constraint: do not preempt a healthy decode merely to admit a request whose eventual KV is already impossible. First run the per-request budget check. Preemption cannot create bytes that the final request needs.

## 7. Graceful failure should preserve user agency

There is no honest policy that makes 200k tokens free. A product can choose a larger GPU, a model with a supported context extension, retrieval that selects fewer passages, summarization before inference, a separate long-context pool, or a paid batch path. The engine should expose which choice it made.

![A policy matrix maps short, long, and over-budget contexts to interactive, batch, and degraded outcomes](/imgs/blogs/case-study-the-long-context-request-that-oomed-the-node-4.webp)

For interactive traffic, the default should be: validate against the documented context, estimate peak bytes, reserve, chunk, and reject with a clear limit when the estimate does not fit. For offline batch traffic, defer or spill to a queue with a deadline. For a RAG endpoint, truncate only if the application has declared an ordering and a maximum information loss; silent truncation turns a memory fix into a relevance bug.

```python
def choose_overflow_action(request, estimate, policy):
    if estimate.input_tokens > policy.max_input_tokens:
        return "reject_context_limit"
    if estimate.peak_bytes <= policy.available_bytes:
        return "admit_chunked"
    if request.mode == "batch" and estimate.peak_bytes <= policy.batch_pool_bytes:
        return "queue_batch_pool"
    if request.allow_truncation and estimate.input_tokens > policy.min_input_tokens:
        return "truncate_with_metadata"
    return "defer_retryable"


def public_error(action, request_id, retry_after_s=None):
    if action == "reject_context_limit":
        return {"status": 413, "code": action, "request_id": request_id}
    if action == "defer_retryable":
        result = {"status": 429, "code": action, "request_id": request_id}
        if retry_after_s is not None:
            result["retry_after"] = retry_after_s
        return result
    return {"status": 202, "code": action, "request_id": request_id}
```

The response must not claim that truncation preserved meaning. It should include the original token count, retained token count, and policy name. If a caller did not grant truncation, do not perform it. Long-context failures are often data-quality failures disguised as infrastructure errors.

The least surprising response also includes a stable error identifier and the server's capacity class. A client should distinguish `context_limit` from `capacity_deferred` without parsing prose. Operators should aggregate the two counters separately: a rising context-limit count means the product contract is too small for the workload; a rising deferred count means the node is overloaded or the ledger is too conservative. Combining them into one “OOM” counter destroys the signal needed to choose a fix.

For streaming APIs, fail before opening the stream when the request is impossible. If the request is merely waiting, the server can either hold an HTTP connection with a bounded deadline or return 429 immediately. Holding connections consumes host memory and file descriptors, so that choice belongs in the admission policy too. A long-context queue is not free just because its tensors are not on the GPU.

The same distinction matters for retry. A 413-style response should not be retried unchanged. A 429-style response should use exponential backoff with a server-provided upper bound. An idempotency key prevents a client from turning one rejected request into many queue entries. These are API details, but they are part of the memory fix: uncontrolled retries increase the number of budgets the scheduler must inspect and can recreate the pressure the policy just avoided.

![A decision tree turns a peak-byte estimate into admission, truncation, deferral, spill, or a clear rejection](/imgs/blogs/case-study-the-long-context-request-that-oomed-the-node-7.webp)

The decision tree is intentionally conservative. “No” at the budget check does not immediately mean “the user made a bad request”; it means this capacity pool cannot honor the request under its current contract. A batch queue, a larger GPU, or an approved truncation path may still be valid. What is forbidden is an implicit transition from “cannot reserve” to “launch and hope.”

This is also where observability becomes part of correctness. Emit one admission event with the request length, estimated KV bytes, chunk size, available bytes, selected action, and policy revision. Emit a second event when a reservation is released. The pair lets operators find leaks and lets a postmortem reconstruct the exact decision without inspecting a destroyed CUDA context. Never log raw prompt content merely to explain a memory decision; token counts and identifiers are enough.

#### Worked example: a safe A100 decision with a protected decode reserve

Use the derived 128 KiB per-token KV value. Suppose eight live requests hold 20 GiB of KV, the model and runtime reservation is 20 GiB total, and policy keeps 4 GiB of allocator slack. The 80 GiB device leaves $80 - 20 - 20 - 4 = 36$ GiB for a new request's KV plus chunk activation. A 128k prompt plus a 1,024-token output cap rounds to 8,064 blocks of 16 tokens and reserves 16.0 GiB of KV in binary arithmetic, plus a 256 MiB activation allowance. It fits under the 36 GiB request envelope. A 200k prompt would require approximately 24.41 GiB of KV, but it fails the model's documented 128k context policy before capacity is considered. Every number is derived from the displayed assumptions; no GPU result is claimed.

## 8. Measure the peak without fooling yourself

Memory instrumentation must separate allocated, reserved, and peak values. `torch.cuda.max_memory_allocated()` measures tensors tracked as allocated by PyTorch. `torch.cuda.max_memory_reserved()` includes memory held by the caching allocator. The device can still have driver-level allocations that neither represents. PyTorch's [CUDA memory documentation](https://docs.pytorch.org/docs/stable/cuda) describes these counters and allocator controls; use that documentation's terminology in dashboards.

The smallest useful profile runs one request at a time, synchronizes before and after each phase, resets peak stats, and records prompt length, chunk size, dtype, batch token count, and model revision. Then repeat under a live decode batch. Do not compare a cold first call with a warm steady state. Do not call a Python wall-clock interval around asynchronous CUDA work without synchronization.

```python
import torch


def memory_snapshot(label):
    return {
        "label": label,
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
    }


def profile_prefill(run, device="cuda"):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    before = memory_snapshot("before")
    run()
    torch.cuda.synchronize(device)
    after = memory_snapshot("after")
    return {"before": before, "after": after}
```

Run this with a labeled matrix: prompt lengths 4k, 32k, 128k; chunk sizes 2,048, 4,096, 8,192; batch decode counts 0, 4, 8; and bf16 on the target GPU. The expected relationship is not a fabricated range: `peak_reserved` should be nondecreasing as request KV grows, and smaller chunks should generally lower the chunk-local activation term. A reader should report the actual values from their hardware rather than copying a number from this article.

For latency, use CUDA events around the prefill kernel region and `torch.cuda.synchronize()` before reading the result. For service behavior, use an open-loop arrival generator and report queue time separately from compute time. An artificially closed loop that sends the next request only after the previous one completes hides queue collapse. The earlier [reproducible inference benchmark protocol](/blog/machine-learning/inference-engineering/an-experiment-protocol-for-inference-benchmarks) defines the broader measurement contract.

| Quantity | Measurement method | Honest result format |
|---|---|---|
| KV bytes | formula from model config | exact derived bytes/token and blocks |
| Activation peak | reset stats, synchronized run | reader's range by GPU, dtype, chunk |
| TTFT | submit to first token, split queue/compute | p50/p95/p99 with prompt length |
| TPOT | synchronized decode intervals | p50/p95 with active decode count |
| OOM rate | count rejected versus allocator failures | workload, arrival process, window |
| Goodput | SLO-compliant completions / window | SLO definition beside number |

### A profile script with expected behavior

```python
import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=4096)
    args = parser.parse_args()
    x = torch.empty((1, args.tokens, args.hidden), dtype=torch.bfloat16, device="cuda")
    torch.cuda.synchronize()
    print({
        "tokens": args.tokens,
        "allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
        "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 3),
        "tensor_bytes": x.numel() * x.element_size(),
    })


if __name__ == "__main__":
    main()
```

This is a deliberately simple allocation probe, not a model activation benchmark. Its tensor alone uses $1\times8192\times4096\times2 = 67{,}108{,}864$ bytes, about 64 MiB. The output's allocator counters will vary with PyTorch version and process state. The probe is valuable precisely because it labels the part that is derived and leaves the allocator result to the reader.

## 9. Case studies / real numbers

The constructed incident is easier to reason about when placed beside public systems work. These are cited results, not measurements from `nanoserve`.

### The vLLM memory-management result

The vLLM project describes PagedAttention as partitioning each sequence's KV cache into fixed-token blocks and allocating physical blocks on demand. Its [2023 vLLM blog post](https://vllm-project.github.io/2023/06/20/vllm.html), dated 2023-06-20, says prior systems could waste 60–80% of memory through fragmentation and over-reservation. That number belongs to the cited vLLM comparison and should not be transplanted into this incident as a measured waste rate.

The design lesson is narrower and more useful here: physical block allocation makes the scheduler's reservation unit explicit. Our derived 16-token block example is intentionally compatible with that mental model, but it is not a claim about vLLM's current default. The remaining issue is peak transient memory: paging solves fragmentation and incremental KV growth; it does not make an unsupported 200k context fit inside a 128k model policy or remove activation work from prefill.

### FlashAttention's memory principle

The [FlashAttention paper](https://arxiv.org/abs/2205.14135), published 2022-05-27, frames attention as an I/O problem and uses tiling to reduce reads and writes between HBM and on-chip SRAM. Its abstract reports lower HBM access complexity and several benchmark speedups under the paper's stated setups. Those results explain why a fused attention implementation can lower working-set pressure relative to materializing the full attention matrix. They do not supply an activation coefficient for our scheduler.

The engineering consequence is to measure with the actual backend. A chunk policy tuned against naïve attention may be too conservative for a tiled backend, while a policy tuned on one FlashAttention version may be invalid after a kernel change. Keep the coefficient versioned with the runner.

### PyTorch allocator behavior

PyTorch documents `max_memory_reserved`, allocator backends, `max_split_size_mb`, and `expandable_segments` in its [CUDA notes](https://docs.pytorch.org/docs/main/notes/cuda.html), accessed 2026-08-04. The documentation also says that disabling caching is a last-resort debugging tool for some OOM patterns. That is a diagnostic lever, not a production admission policy. A service that “fixes” this incident by disabling caching may trade one failure mode for more allocation overhead and synchronization.

The practical lesson is to record both allocated and reserved counters, but make the scheduler's own reservation ledger authoritative. Counters explain why a run failed; reservations decide what may start.

### The model-card boundary

Meta's Llama 3.1 card lists the 8B model's 128k context length. A request larger than that must have a product-level explanation before it reaches a GPU: a supported rope scaling configuration, a different checkpoint, a retrieval reduction, or a rejection. Treating 200k as “just a long prompt” mixes model validity with hardware capacity and makes incident response ambiguous.

## 10. Stress the policy where it is likely to break

The happy path is a single 8k prompt. The service is defined by the edges.

| Stress case | Failure to look for | Required invariant |
|---|---|---|
| 4k prompt, 64 active decodes | a small request starves behind a giant batch | decode reservation remains protected |
| 128k prompt on A100 | final KV fits but chunk peak does not | reject before launch |
| 200k prompt on 4090 | KV-only request exceeds device | 413, no allocator call |
| cancellation during chunk 12 | leaked blocks | reservation returns exactly once |
| prompt ends at 200,000 tokens | partial final block | block table and position offset agree |
| shared prefix | double-counted or unsafe shared bytes | reference counts and tenant key are explicit |
| retry after 429 | retry storm | bounded `Retry-After` and client backoff |
| allocator fragmentation | reserved bytes exceed ledger | conservative slack and profile alert |

### Cancellation and partial chunks

Cancellation is a memory event. If a client disconnects during chunk 12, the scheduler must stop launching future chunks, keep already written KV only if the request can resume, and release the remaining reservation. A double release is as dangerous as a leak because the next request may be handed bytes that still contain live state. The block manager should make release idempotent and attach an owner identifier to every allocation in debug mode.

### Prefix sharing

Prefix caching can lower the incremental KV cost for repeated prompts, but it complicates admission. A shared prefix is not free unless its blocks are already resident and reference-counted. The request budget should charge private suffix blocks and the possibility of a cache miss. Tenant isolation also matters; the earlier [prefix-cache security case](/blog/machine-learning/inference-engineering/prompt-caching-semantics-engine-side-and-provider-side) explains why a timing or content leak is not an acceptable memory optimization.

### Long context on an L4

The fixed hardware matrix includes the L4 24GB accelerator. NVIDIA lists the L4's memory capacity and bandwidth on its [official product page](https://www.nvidia.com/en-us/data-center/l4/), accessed 2026-08-04. The same 200k KV arithmetic already fails the 24GB capacity before weights. A policy shared between an L4 and an A100 must therefore be parameterized by device profile; a single “max input tokens” constant is not portable.

### Chunk size and fairness

Chunking is not automatically fair. If the scheduler always completes every chunk of the oldest long request before running new work, it has simply moved monopolization from one kernel launch to 25 smaller launches. Use a token budget per scheduling round, and charge prefill tokens separately from decode turns. A common policy is one prefill chunk followed by one decode iteration for each runnable decode request, subject to the peak-byte check. The exact ratio is a workload choice; expose it as a metric so a latency regression has a visible cause.

## 11. When to reach for this, and when not to

Build this policy into `nanoserve` when the service has mixed prompt lengths, interactive decode traffic, a finite GPU pool, or a requirement to explain failures to clients. The code is most valuable before the first serious OOM because it turns a process-level exception into a request-level decision.

Use a simpler path when there is one fixed prompt shape, one request at a time, or a batch job that can tolerate a process restart. Even then, keep the model context check and a basic byte estimate. A toy benchmark that never admits concurrent work does not need a full fairness scheduler.

Use vLLM or another mature engine when you need production-grade paged attention, prefix caching, multi-GPU scheduling, speculative decoding, or years of kernel and allocator compatibility. The [vLLM deep dive](/blog/machine-learning/model-serving/vllm-deep-dive) is the right comparison point. `nanoserve` is a learning artifact and a place to make the reservation contract obvious; it is not a claim that a small Python scheduler should replace a mature serving stack.

Do not use truncation as a silent OOM workaround. Do not turn a 429 into an infinite client retry loop. Do not set the activation reserve to zero because the first small prompt succeeded. Do not report only GPU utilization: a node can be busy recomputing evicted KV blocks while goodput collapses.

## Key takeaways

1. A context limit is a model-validity check; admission is a live memory decision.
2. For Llama 3.1 8B in bf16, the derived KV cost is 128 KiB per token under the stated 32-layer, 8-KV-head, 128-dimension configuration.
3. A 200k-token request needs about 24.41 GiB of KV before activations and is beyond the model card's 128k context policy.
4. Chunked prefill bounds transient work but does not reduce final KV requirements.
5. Reserve bytes for accepted decode work before considering new prefill.
6. Charge block rounding and output allowance to each request, not to an invisible global pool.
7. Measure allocated and reserved memory separately, with synchronization and a labeled workload matrix.
8. Reject impossible requests before launching CUDA; defer temporary pressure with a retryable response.
9. Treat cancellation, prefix sharing, and preemption as reservation-lifetime problems.
10. If the policy becomes a second inference engine, use a mature serving system and keep only the contracts you need to understand.

## Further reading

- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180), Kwon et al., 2023.
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135), Dao et al., 2022.
- [Llama 3.1 8B Instruct model card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), Meta, 2024-07-23.
- [PyTorch CUDA memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html), accessed 2026-08-04.
- [Continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention).
- [Inference engineering: what this series builds](/blog/machine-learning/inference-engineering/what-inference-engineering-is) and [the inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook).

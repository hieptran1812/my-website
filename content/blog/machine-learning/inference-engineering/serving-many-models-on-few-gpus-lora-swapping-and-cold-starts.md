---
title: "Serving many models on few GPUs: LoRA swapping and cold starts"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Build a residency-aware router that batches LoRA adapters, hides safe transfers, and escalates difficult requests without turning every cache miss into a latency incident."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "lora",
    "multi-tenancy",
    "batching",
    "latency",
    "throughput",
    "ml-systems",
    "pytorch",
    "vllm",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 31
---

An inference service rarely has one model and one customer. It has a shared base model, dozens of fine-tuned behaviors, a few heavyweight specialists, and a traffic pattern that changes faster than a GPU can load a checkpoint. The embarrassing failure mode is not that a model is too slow. It is that the right model is absent from device memory when the request arrives. The first token waits behind a transfer, an adapter initialization, a CUDA graph warm-up, and sometimes an eviction of the model that was serving perfectly well one request earlier.

The diagram below is the mental model for this post: a router is also a residency controller. It chooses a model, an adapter, and a queue placement together; the GPU is a finite memory budget, not a magical shared shelf.

![A router fans requests toward a hot base model, adapter pool, cold model store, and finite GPU budget](/imgs/blogs/serving-many-models-on-few-gpus-lora-swapping-and-cold-starts-1.webp)

We will extend `nanoserve` with a small placement layer that makes those choices explicit. The code is intentionally plain Python and PyTorch: a model registry, an adapter-aware batch key, an asynchronous residency manager, a cascade decision, and admission control. It is not a replacement for vLLM. It is a compact engine component you can run, profile, and compare against the production systems described in the [vLLM deep dive](/blog/machine-learning/model-serving/vllm-deep-dive).

By the end, you should be able to answer five operational questions before shipping a new fine-tune:

- Can requests using different adapters share the same base-model forward pass?
- When should an adapter be copied, prefetched, pinned, or evicted?
- Which part of TTFT is queueing, transfer, initialization, or actual model compute?
- Which requests are safe to send to a small model first?
- How do we refuse work before promising a token that cannot fit in model, adapter, and KV memory?

The recurring example is Llama-3.1-8B on one RTX 4090 or one A100. When a number is not derived from a displayed formula, it is marked as cited or reproducible. The sample thresholds are policy examples, not measurements from this machine.

## 1. The problem is residency, not model count

The first rule is simple: count resident bytes, not catalog entries. A model registry can list 100 models while a GPU may hold only one dense 8B base plus adapters and KV cache. The registry is control-plane state. Device residency is data-plane state. Confusing them makes the API look healthy while the first token waits for a loader.

Three kinds of sharing matter:

1. **Weight sharing.** A LoRA adapter changes a base layer by adding a low-rank update. The expensive base weights remain common to all adapters that target that base.
2. **Batch sharing.** Requests with different adapter IDs can enter one decode step if the kernel can select the corresponding low-rank matrices per row.
3. **Cache sharing.** KV blocks belong to sequences, not adapters. They consume device memory even when the base weights are already warm.

The placement controller must keep these dimensions separate. A cache hit for an adapter does not imply a free KV slot. A free KV slot does not imply that the selected base model is resident. A warm base does not imply that the adapter’s tensors are on the same CUDA device.

### A small memory derivation

For a dense Llama-3.1-8B in bf16, a rough weight footprint is derived from parameter count and bytes per parameter: $8 \times 10^9 \text{ parameters} \times 2 \text{ bytes} = 16 \times 10^9 \text{ bytes}$, or about $14.9$ GiB after dividing by $2^{30}$. The model card reports the parameter scale; this conversion is derived, not a benchmark. On a 24 GiB card, that leaves a little over 9 GiB before allocator overhead, activations, CUDA runtime allocations, adapters, and KV cache. The exact usable amount depends on the checkpoint format and runtime.

The KV law is more useful than a vague “the cache is large.” For Llama-3.1-8B, use the model configuration values $L=32$, $H_{kv}=8$, head dimension $d=128$, and bf16 bytes $b=2$. One token stores K and V, so:

$$
\text{KV bytes/token} = 2 \cdot L \cdot H_{kv} \cdot d \cdot b
= 2 \cdot 32 \cdot 8 \cdot 128 \cdot 2
= 131{,}072\text{ bytes} = 128\text{ KiB}.
$$

At 8,192 tokens, that is $8{,}192 \times 128\text{ KiB} = 1{,}048{,}576\text{ KiB} = 1$ GiB per sequence, before block metadata. One sequence can therefore consume more memory than a small adapter pool. This is why “keep every adapter hot” is not a complete policy.

The [KV cache post](/blog/machine-learning/large-language-model/kv-cache) explains the tensor layout in isolation. Here the important product constraint is the sum:

$$
M_{\text{resident}} = M_{\text{weights}} + M_{\text{adapters}} + M_{\text{KV}} + M_{\text{runtime}} \leq M_{\text{usable GPU}}.
$$

This is an explanatory abstraction for `nanoserve`, not a formula quoted from a paper. It is the admission invariant we will implement later.

#### Worked example: one 24 GiB card

Suppose a deployment reserves 16 GiB for the derived bf16 base estimate, 512 MiB for a hot adapter pool, and 2 GiB for runtime and safety headroom. The remaining budget is $24 - 16 - 0.5 - 2 = 5.5$ GiB. Dividing by the derived 128 KiB per token gives $5.5 \times 2^{30} / 131{,}072 = 45{,}056$ token-equivalents. That is a budget, not a promise: multiple sequences, block rounding, temporary activations, and fragmentation reduce the schedulable count. A reproducible allocator script later should report the actual ceiling on the reader’s driver and checkpoint.

The conclusion is deliberately uncomfortable: the model catalog is not the serving capacity. The resident set is.

## 2. Multi-LoRA batching: one base, many deltas

LoRA represents an update to a weight matrix $W$ as a low-rank product $BA$, usually scaled by a factor derived from the adapter rank. For an input activation $x$, a layer computes the base path and adds the adapter path:

$$
y = xW^T + s\,((xA^T)B^T).
$$

This equation is an explanatory abstraction of the implementation. It shows the important sharing boundary: $W$ is common, while $A$ and $B$ depend on the adapter ID. If a batch contains rows from adapters `support`, `legal`, and `code`, the base GEMM can still be shared in principle. The low-rank updates need a per-row adapter selection.

![A matrix compares a shared base and grouped adapter batch with separate replicas across memory and cold-start behavior](/imgs/blogs/serving-many-models-on-few-gpus-lora-swapping-and-cold-starts-2.webp)

The naive design creates one model replica per adapter. For $N$ adapters, the base footprint becomes $N \times M_{\text{base}}$. With three adapters and the derived 16 GB base estimate, that is $3 \times 16 = 48$ GB before KV cache. The shared-base design pays roughly one base plus the adapter tensors. The adapter tensor size for one linear layer with input width $d_{in}$, output width $d_{out}$, and rank $r$ is $(d_{in}r + rd_{out})$ values. For bf16, multiply by 2 bytes. The total depends on which projections are targeted, so the code should load metadata rather than invent a universal adapter size.

### Batch keys are a correctness boundary

The scheduler must not merge incompatible requests merely because they arrived in the same millisecond. A batch key should include at least base model, dtype, tensor-parallel placement, adapter set, and decoding constraints. Adapter IDs can differ inside one multi-LoRA batch only if the kernel contract supports row-wise selection. A grammar mask or a different tokenizer may force a separate path even when the base is identical.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class BatchKey:
    base: str
    dtype: str
    device: str
    adapters: tuple[str, ...]
    tokenizer: str
    grammar: str | None

def compatible(a: BatchKey, b: BatchKey) -> bool:
    same_runtime = (a.base, a.dtype, a.device, a.tokenizer, a.grammar) == (
        b.base, b.dtype, b.device, b.tokenizer, b.grammar
    )
    # An adapter-aware kernel may accept different IDs; a plain kernel may not.
    return same_runtime and bool(set(a.adapters) | set(b.adapters))

keys = [
    BatchKey("llama-3.1-8b", "bf16", "cuda:0", ("support",), "llama", None),
    BatchKey("llama-3.1-8b", "bf16", "cuda:0", ("legal",), "llama", None),
]
print(compatible(*keys))
```

Expected output is `True` because the two rows share every runtime property and differ only in adapter identity. The output is reproducible by running the snippet on any Python 3.10+ interpreter; it does not claim GPU performance.

The production kernel usually takes a dense tensor of adapter indices and a packed representation of $A$ and $B$. A simple reference implementation can make the semantics obvious before optimizing it:

```python
import torch

def lora_linear(x, weight, a_bank, b_bank, adapter_ids, scale=1.0):
    # x: [batch, in_features], weight: [out_features, in_features]
    base = x @ weight.T
    delta = torch.empty_like(base)
    for row, adapter_id in enumerate(adapter_ids):
        a, b = a_bank[adapter_id]
        delta[row] = (x[row] @ a.T) @ b.T
    return base + scale * delta

torch.manual_seed(0)
x = torch.randn(3, 8)
w = torch.randn(5, 8)
bank = {"support": (torch.randn(2, 8), torch.randn(5, 2)),
        "legal": (torch.randn(2, 8), torch.randn(5, 2))}
y = lora_linear(x, w, bank, bank, ["support", "legal", "support"])
print(y.shape)
```

This reference prints `torch.Size([3, 5])`. It is intentionally not a fast implementation: the Python loop exposes the row-to-adapter mapping. A fused kernel replaces the loop with grouped or split-K work. The [vLLM multi-LoRA post](https://vllm.ai/blog/2026-02-26-multi-lora), dated 2026-02-26, describes a fused MoE-LoRA path and reports a GPT-OSS 20B comparison with eight rank-32 adapters; that result is cited evidence for that setup, not a prediction for this reference loop or for dense Llama.

### Grouping versus mixing

There are three workable layouts:

| Layout | Base weights | Adapter work | Failure mode | Source |
|---|---:|---|---|---|
| Separate replica | one copy per adapter | simple dense path | VRAM multiplies with tenants | derived: $N \times M_{base}$ |
| Grouped batch | one shared copy | group rows by adapter | small groups underfill kernels | explanatory design |
| Fused row-wise | one shared copy | per-row index in kernel | more kernel and testing complexity | cited: vLLM multi-LoRA, 2026-02-26 |

Do not promise that multi-LoRA always improves throughput. If every batch has one row and every request uses a different adapter, grouping provides memory sharing but little arithmetic sharing. If the service has long decode sequences, one row per adapter may still be worthwhile because the base weights are not replicated. If the service has short requests, adapter load and grouping overhead can dominate.

### Tenant isolation is part of the batch key

An adapter name is not an authorization decision. Resolve a tenant-scoped adapter handle to an immutable content digest, then use that digest in the batch key. Never let an untrusted request choose an arbitrary filesystem path or a CUDA pointer. The shared base can be public while the adapter remains private; logging must not expose the private name in a cross-tenant metric label.

The cache key should include the base digest, adapter digest, tokenizer digest, and policy version. A model update that preserves the string name but changes the base weights must invalidate every adapter compiled against the old shape or calibration.

## 3. Adapter swapping is a cache policy

An adapter can live in object storage, local SSD, host RAM, pinned host memory, or device memory. These locations have different capacity and transfer costs. A robust service treats each move as a state transition with ownership and cancellation, rather than calling `load_state_dict` inside a request handler.

![A before and after comparison shows blocked decode during a 4 GB transfer versus a 512 MB hot pool with async prefetch](/imgs/blogs/serving-many-models-on-few-gpus-lora-swapping-and-cold-starts-3.webp)

The figure’s values are policy examples. The 4 GB and 512 MB labels illustrate two configurations; they are not claimed measurements. The decision is a cache admission problem:

$$
\text{keep adapter } i \text{ hot if } p_i \cdot c_{swap,i} > c_{memory,i},
$$

where $p_i$ is predicted reuse probability, $c_{swap,i}$ is the latency or queue cost of a miss, and $c_{memory,i}$ is the opportunity cost of occupying device memory. This is an explanatory objective, not an exact vLLM formula. In practice, estimate $p_i$ from a decayed request counter and protect adapters with an SLO or active stream.

### A residency manager with explicit states

```python
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time

class Place(str, Enum):
    DISK = "disk"
    HOST = "host"
    DEVICE = "device"

@dataclass
class Adapter:
    name: str
    bytes: int
    place: Place = Place.DISK
    last_used: float = field(default_factory=time.monotonic)
    active: int = 0

class AdapterPool:
    def __init__(self, capacity_bytes: int):
        self.capacity = capacity_bytes
        self.used = 0
        self.items: dict[str, Adapter] = {}
        self.lock = asyncio.Lock()

    async def acquire(self, name: str, loader):
        async with self.lock:
            item = self.items[name]
            item.active += 1
            item.last_used = time.monotonic()
            if item.place is Place.DEVICE:
                return item
            await self._make_room(item.bytes, loader)
            await loader(item, Place.DEVICE)
            item.place = Place.DEVICE
            self.used += item.bytes
            return item

    async def release(self, item: Adapter):
        async with self.lock:
            item.active -= 1
            item.last_used = time.monotonic()

    async def _make_room(self, needed, loader):
        victims = sorted(self.items.values(), key=lambda x: x.last_used)
        for victim in victims:
            if self.used + needed <= self.capacity:
                break
            if victim.active == 0 and victim.place is Place.DEVICE:
                await loader(victim, Place.HOST)
                victim.place = Place.HOST
                self.used -= victim.bytes
```

This code is a runnable state-machine skeleton, not a CUDA loader. It has two important properties: active streams pin an adapter, and eviction is serialized under one lock. A real implementation must add rollback if `loader` fails, a generation number to reject stale transfers, and a separate CUDA stream for asynchronous copies. It should also use a condition variable instead of waiting while holding the lock if the loader can call back into the pool.

The key policy is not LRU alone. LRU evicts the least recently used object, but a just-started long decode may have a low recent count and a high future cost. Add a lease with an expiry, active-stream protection, and a maximum transfer rate. A queue of 100 misses can otherwise turn one cold adapter into a thundering herd.

### Pinned host memory and overlap

Pinned host memory allows a DMA transfer to proceed without first paging the source buffer. In PyTorch, a CPU tensor can be allocated with `pin_memory=True`, copied with `non_blocking=True`, and synchronized on a CUDA event before the first kernel uses it. The safety rule is simple: do not reuse or free the host buffer until the event signals completion.

```python
import torch

def stage_adapter(cpu_state, device="cuda"):
    staged = {
        name: tensor.pin_memory() if not tensor.is_pinned() else tensor
        for name, tensor in cpu_state.items()
    }
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        device_state = {
            name: tensor.to(device, non_blocking=True)
            for name, tensor in staged.items()
        }
        ready = torch.cuda.Event()
        ready.record(stream)
    return device_state, ready, stream

# The reader can call ready.synchronize() before the first use.
```

This script requires CUDA for the transfer path; on a CPU-only machine it should be treated as an API example, not executed. A benchmark must compare synchronous and overlapped copies with the same adapter bytes, stream priorities, and decode workload. The expected result is qualitative: overlap can hide transfer time only while the GPU has independent work; a completely idle GPU still pays the copy.

### When swapping is the wrong optimization

Swapping is a poor answer when the working set is stable and fits. Repeatedly moving the same adapter means the cache policy is wrong or the traffic is too fragmented for the available memory. It is also a poor answer when the transfer path is not encrypted, authenticated, and integrity-checked for private adapters. Model bytes are supply-chain inputs, not harmless blobs.

## 4. Cold starts are a timeline, not one number

Time to first token, or TTFT, is the elapsed time from accepted request to the first emitted token. It includes queue delay, tokenization if it is on the request path, prompt prefill, and any residency transition. Time per output token, TPOT, begins after the first token and measures decode cadence. Mixing the two hides cold starts: a warm model can have healthy TPOT while a miss has terrible TTFT.

![A timeline shows arrival, routing, weight and adapter loading, warm decode, and eviction as separate cold-start phases](/imgs/blogs/serving-many-models-on-few-gpus-lora-swapping-and-cold-starts-4.webp)

The 0 ms, 20 ms, 100 ms, and 500 ms labels in the figure are illustrative milestones for a reproducible trace, not reported measurements. The arithmetic is additive when phases serialize:

$$
\text{TTFT} = T_{queue} + T_{route} + T_{weights} + T_{adapter} + T_{init} + T_{prefill}.
$$

If routing takes 20 ms, adapter load takes 100 ms, initialization takes 50 ms, and prefill takes 80 ms, the derived cold-path TTFT is $20 + 100 + 50 + 80 = 250$ ms before queueing. If the adapter is already resident, subtract the adapter term: 150 ms. These are a worked arithmetic example, not a service result.

The transfer component itself is bounded by bytes divided by effective bandwidth:

$$
T_{copy} \geq \frac{B_{adapter}}{BW_{effective}}.
$$

For a 4 GiB transfer over an illustrative 12 GiB/s effective path, $4 / 12$ seconds is 333 ms. The effective bandwidth must be measured on the reader’s host-to-device path; a GPU’s peak HBM bandwidth is not the PCIe transfer rate. NVIDIA’s [RTX 4090 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/) list the product’s memory and interface characteristics, but they do not turn a particular server’s PCIe copy into a guaranteed 12 GiB/s.

<figure class="blog-anim">
<svg viewBox="0 0 900 300" role="img" aria-label="Cold start transfers an adapter into GPU memory before decode can begin" style="width:100%;height:auto;max-width:900px" xmlns="http://www.w3.org/2000/svg">
<title id="cold-title">Cold start versus warm adapter residency</title>
<desc id="cold-desc">A request moves from object storage through host memory into GPU memory, then the decode lane becomes available; the animation shows the transfer and the waiting gap.</desc>
<style>
:root{--ink:#1e1e1e;--muted:#6b7280;--blue:#a5d8ff;--amber:#ffec99;--green:#b2f2bb;--red:#ffc9c9;--paper:#f8fafc} .ink{fill:var(--ink)} .muted{fill:var(--muted)} .box{stroke:var(--ink);stroke-width:3;rx:16} .transfer{fill:var(--amber);animation:move 5s ease-in-out infinite}.gpu{fill:var(--blue)} .ready{fill:var(--green);opacity:.25;animation:ready 5s ease-in-out infinite}.wait{fill:var(--red);opacity:.8;animation:wait 5s ease-in-out infinite} @keyframes move{0%,15%{transform:translateX(0);opacity:1}45%,65%{transform:translateX(330px);opacity:1}75%,100%{transform:translateX(660px);opacity:0}} @keyframes ready{0%,58%{opacity:.25}70%,100%{opacity:1}} @keyframes wait{0%,58%{opacity:.85}70%,100%{opacity:0}} @media (prefers-reduced-motion:reduce){.transfer{animation:none;transform:translateX(330px)}.ready{animation:none;opacity:1}.wait{animation:none;opacity:0}}
</style>
<rect x="20" y="20" width="860" height="260" rx="20" fill="var(--paper)" stroke="var(--ink)" stroke-width="2"/>
<text x="55" y="65" font-size="25" class="ink">Cold request: transfer first, decode second</text>
<rect x="55" y="110" width="150" height="90" class="box" fill="var(--muted)"/><text x="82" y="148" font-size="20" class="ink">Object store</text><text x="92" y="177" font-size="18" class="ink">weights</text>
<rect x="280" y="110" width="150" height="90" class="box" fill="var(--amber)"/><text x="308" y="148" font-size="20" class="ink">Host RAM</text><text x="300" y="177" font-size="18" class="ink">pinned buffer</text>
<rect x="505" y="110" width="150" height="90" class="box gpu"/><text x="552" y="148" font-size="20" class="ink">GPU</text><text x="530" y="177" font-size="18" class="ink">resident base</text>
<rect x="730" y="110" width="105" height="90" class="box ready"/><text x="754" y="148" font-size="20" class="ink">Decode</text><text x="750" y="177" font-size="18" class="ink">ready</text>
<rect x="80" y="225" width="580" height="18" rx="9" class="wait"/><text x="55" y="263" font-size="18" class="muted">waiting gap = queue + transfer + initialization</text>
<rect x="90" y="132" width="80" height="45" rx="12" class="transfer"/><text x="108" y="161" font-size="17" class="ink">adapter</text>
<figcaption>Motion makes the cold-start cost visible: the decode lane stays unavailable until residency is complete.</figcaption>
</svg>
</figure>

Instrument every boundary. The request record should carry `accepted_at`, `route_at`, `weights_ready_at`, `adapter_ready_at`, `prefill_done_at`, and `first_token_at`. Emit durations, not only timestamps, so a trace can answer whether p99 rose because the adapter cache missed or because the queue was full.

```python
from dataclasses import dataclass
from time import monotonic

@dataclass
class Stamps:
    accepted: float
    route: float | None = None
    weights: float | None = None
    adapter: float | None = None
    prefill: float | None = None
    first_token: float | None = None

    def durations_ms(self):
        def ms(end, start):
            return None if end is None else round((end - start) * 1000, 3)
        return {
            "queue": ms(self.route, self.accepted),
            "residency": ms(self.adapter, self.route),
            "prefill": ms(self.prefill, self.adapter or self.route),
            "ttft": ms(self.first_token, self.accepted),
        }

stamps = Stamps(monotonic())
print(stamps.durations_ms())
```

The initial output contains `None` values because no stage has completed. That is desirable: an absent stage is not a zero-duration stage. In production, attach the trace ID to each loader and CUDA event. Do not infer TTFT from server wall-clock logs that omit queue time.

## 5. Model cascades: small first, large only when justified

A cascade sends a request to a small model, evaluates whether its answer is safe to accept, and sends uncertain cases to a larger model. It can save expensive work, but only if the gate’s false-accept rate is acceptable for the task. “The small model was confident” is not enough; confidence must be calibrated against labeled outcomes or a conservative verifier.

![A decision tree routes a request through difficulty features, a small model, a confidence gate, and a large fallback](/imgs/blogs/serving-many-models-on-few-gpus-lora-swapping-and-cold-starts-5.webp)

Let $p$ be the fraction of requests accepted by the small model, $C_s$ the small-model cost, and $C_l$ the large-model cost. If rejected requests run both models, the expected compute cost is:

$$
E[C] = C_s + (1-p)C_l.
$$

The cascade saves large-model work only when $p$ is high enough to offset the small-model pass and any verifier. If the large model costs four units and the small model costs one unit, with $p=0.75$, then $E[C] = 1 + 0.25 \times 4 = 2$ units, half of always running the large model at 4 units. This is derived arithmetic. It says nothing about quality until the gate is evaluated.

For latency, a fallback can be worse than a direct large-model route because it serializes the small pass. If a small pass takes 40 ms and a large pass takes 120 ms, an escalated request waits 160 ms of compute before queueing and prefill. Parallel speculative evaluation can change that, but it spends large-model capacity on requests that may have been accepted cheaply.

### A reproducible gate contract

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Decision:
    route: str
    reason: str
    score: float

def route_small_or_large(
    prompt_tokens: int,
    mean_logprob: float,
    verifier_score: float,
    budget_ms: int,
) -> Decision:
    # Policy example: tune from labeled validation data, never by intuition alone.
    difficulty = 0.35 * min(prompt_tokens / 4096, 1.0)
    uncertainty = 1.0 - max(0.0, min(1.0, mean_logprob + 1.0))
    risk = 0.5 * difficulty + 0.3 * uncertainty + 0.2 * (1.0 - verifier_score)
    if budget_ms < 150:
        return Decision("small", "latency budget", risk)
    if risk <= 0.25:
        return Decision("small", "calibrated gate", risk)
    return Decision("large", "fallback", risk)

print(route_small_or_large(512, -0.1, 0.95, 500))
```

The printed decision is a reproducible policy result for those inputs, not a quality claim. The coefficients and threshold are placeholders that must be fit on a held-out set. The feature vector is intentionally logged: prompt length, mean log probability, verifier score, budget, route, and final outcome.

### Difficulty is not one scalar

Length is useful because long prompts cost more prefill and often require more KV reservation, but length does not equal reasoning difficulty. Retrieval disagreement, code syntax risk, tool-call requirements, language pair, output budget, and a verifier’s score are more useful when measured together. The point is not to create a mysterious “difficulty model.” The point is to create an auditable escalation policy that a replay tool can inspect.

## 6. Difficulty routing needs replayable signals

![A grid maps chat, RAG, code, and translation requests to logged length, uncertainty, disagreement, and budget signals](/imgs/blogs/serving-many-models-on-few-gpus-lora-swapping-and-cold-starts-6.webp)

The grid is a logging design, not a benchmark result. Each row is a workload in the series prompt suite. Each column is observable before or during generation. A request can start on the small model, but a signal such as rising token entropy can trigger a controlled fallback only at a safe boundary.

Use normalized features so one large number does not accidentally dominate the score:

$$
s(x) = \sum_i w_i\,\operatorname{clip}\left(\frac{x_i - q_{10,i}}{q_{90,i}-q_{10,i}},0,1\right).
$$

This displayed normalization rule is a policy approximation. The quantiles $q_{10}$ and $q_{90}$ should come from a training or calibration window, and the denominator needs a zero-range guard. The score is not a truth label; it is a routing feature.

```python
def normalize(value, low, high):
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))

def difficulty(*, prompt_tokens, entropy, retrieval_gap, output_budget):
    return (
        0.30 * normalize(prompt_tokens, 128, 4096)
        + 0.30 * normalize(entropy, 0.2, 4.0)
        + 0.25 * normalize(retrieval_gap, 0.0, 1.0)
        + 0.15 * normalize(output_budget, 64, 2048)
    )

score = difficulty(prompt_tokens=1024, entropy=1.4,
                   retrieval_gap=0.2, output_budget=512)
print(round(score, 3))
```

The output is deterministic and reader-reproducible. It is not an expected accuracy. To calibrate it, store the score and route, then join them to an outcome label: human preference, unit-test pass, retrieval citation check, or task-specific evaluator. Tune thresholds per workload; a translation score should not share the code-completion threshold automatically.

### Online fallback and stream semantics

Fallback is easy before the first token and much harder after streaming starts. If a client has received “The answer is”, switching models mid-stream can produce an incoherent continuation. Choose one of three contracts: buffer until the gate passes, allow fallback only before first token, or declare that fallback restarts the answer with an explicit event. The endpoint design in [designing an OpenAI-compatible inference API](/blog/machine-learning/inference-engineering/designing-an-openai-compatible-inference-api) should expose this behavior as a documented event, not hide it in a timeout.

For JSON or tool calls, a small model’s partial output may not be a valid prefix for the large model. A grammar-constrained decoder can preserve syntax, but semantic continuity remains a problem. In practice, fallback is safest for classification, routing, extraction, and verifier-backed tasks. It is riskier for long free-form chat after tokens have already been sent.

## 7. Admission control reserves three budgets

The scheduler should refuse a request before it starts if the selected plan cannot reserve base weights, adapter bytes, and KV blocks. Waiting is a valid answer. Starting and then evicting an active model is usually worse because it turns one miss into a fleet-wide latency storm.

![A layered stack shows tenant quota, model residency, adapter residency, KV reservation, scheduler, and the GPU ceiling](/imgs/blogs/serving-many-models-on-few-gpus-lora-swapping-and-cold-starts-7.webp)

The stack is the admission order. First check policy: is this tenant allowed to use the model and adapter? Then check model residency or a safe load plan. Then reserve adapter space. Then reserve KV capacity for the maximum prompt and output budget. Only then enqueue the request for prefill.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Plan:
    weights: int
    adapter: int
    kv: int
    runtime: int

def can_admit(plan: Plan, usable_gpu_bytes: int, safety_fraction=0.90):
    required = plan.weights + plan.adapter + plan.kv + plan.runtime
    limit = int(usable_gpu_bytes * safety_fraction)
    return required <= limit, required, limit

# Derived example: 24 GiB card, 16 GiB weights, 512 MiB adapter,
# 2 GiB KV reservation, 1 GiB runtime headroom.
gib = 2**30
ok, required, limit = can_admit(
    Plan(16*gib, 512*2**20, 2*gib, 1*gib), 24*gib
)
print(ok, required / gib, limit / gib)
```

The expected output is `True 19.5 21.6`, derived by adding $16 + 0.5 + 2 + 1 = 19.5$ GiB and applying a 90% safety fraction to 24 GiB. The safety fraction is a policy example. The runtime’s actual free memory must be measured with `torch.cuda.mem_get_info()` and reconciled with allocator reservations.

### KV reservation from a request budget

If a request has 1,024 prompt tokens and a 512-token output cap, reserve 1,536 token-equivalents. With the derived 128 KiB per token, that is $1{,}536 \times 128$ KiB = 196,608 KiB = 192 MiB. If the allocator uses 16-token blocks, the logical requirement is $\lceil 1{,}536/16 \rceil = 96$ blocks. The final block can be partially used, so block allocation and byte reservation must agree about internal fragmentation.

```python
import math

KV_BYTES_PER_TOKEN = 2 * 32 * 8 * 128 * 2
BLOCK_TOKENS = 16

def kv_reservation(prompt_tokens, max_new_tokens):
    tokens = prompt_tokens + max_new_tokens
    blocks = math.ceil(tokens / BLOCK_TOKENS)
    return tokens * KV_BYTES_PER_TOKEN, blocks

bytes_needed, blocks = kv_reservation(1024, 512)
print(bytes_needed // 2**20, blocks)
```

The reproducible output is `192 96` for the fixed Llama-3.1-8B example. If the model configuration differs, calculate the bytes from its actual `num_hidden_layers`, `num_key_value_heads`, and `head_dim`; do not copy the constant.

### Queueing and fairness

Admission is not only memory protection. It is fairness. A tenant that submits long requests can occupy all KV blocks and starve short requests, even if the GPU is technically busy. Track per-tenant reserved bytes and active decode slots. Apply a weighted fair policy at admission, then let the scheduler use continuous batching inside each admitted pool.

An adapter cache should also have a quota per tenant. Otherwise one tenant with many low-volume adapters can evict a shared hot adapter for every other tenant. A weighted cache score can include reuse, bytes, active leases, and predicted next-arrival time. Keep the policy inspectable and export the reason for every eviction.

## 8. The `nanoserve` placement loop

The pieces now fit into a loop:

1. Authenticate and resolve the requested base and adapter to immutable digests.
2. Estimate prompt and output token budgets.
3. Compute a route candidate: direct small, cascade, or large specialist.
4. Ask the residency manager for a plan, including possible prefetch.
5. Reserve model, adapter, and KV budgets atomically.
6. Join a compatible batch or wait for a compatible decode step.
7. Record stage timestamps and release leases when the stream ends.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Request:
    tenant: str
    base: str
    adapter: str | None
    prompt_tokens: int
    max_new_tokens: int
    budget_ms: int

def choose_plan(req, registry, free_bytes):
    base = registry[req.base]
    adapter = registry.get(req.adapter) if req.adapter else None
    kv = (req.prompt_tokens + req.max_new_tokens) * KV_BYTES_PER_TOKEN
    plan = Plan(base.weight_bytes, adapter.bytes if adapter else 0, kv, 1 * 2**30)
    ok, required, limit = can_admit(plan, free_bytes)
    if not ok:
        return "reject_or_wait", plan, required, limit
    return "admit", plan, required, limit
```

This function is deliberately side-effect free. A separate transaction commits the reservation after checking the current generation of the residency map. If the map changes between planning and commit, retry with a bounded count; never silently exceed the limit because a victim was evicted in another coroutine.

The placement layer should expose a dry-run endpoint for operators. Given a base, adapter, prompt length, output cap, and tenant, it should return: expected route, current residency, bytes to transfer, KV blocks requested, estimated queue class, and the reason it might wait. This makes a cold-start incident explainable without attaching a debugger to the GPU process.

### A simple load and residency replay

```python
events = [
    (0, "support"), (1, "support"), (2, "legal"),
    (3, "support"), (4, "code"), (5, "legal"),
]

last = {}
for tick, adapter in events:
    hit = adapter in last
    print(tick, adapter, "hit" if hit else "miss")
    last[adapter] = tick
```

The output has misses on the first appearance of each adapter and hits on later appearances. It is not a cache benchmark because no capacity is modeled. Extend it with a byte limit, active leases, and a loader delay to replay a trace before choosing an eviction policy. The useful question is not “what is the average hit rate?” but “which misses occur while a long decode is active, and what is their TTFT contribution?”

## 9. How to measure the system honestly

A serving benchmark needs two axes: workload and residency state. At minimum, run warm-base/warm-adapter, warm-base/cold-adapter, and cold-base/cold-adapter cases. Use the fixed prompt suite: chat with short input and long output, RAG with long input and short output, code completion, and translation. Record base model, adapter digest, prompt tokens, output tokens, batch size, route, and GPU.

Use a warm-up phase and discard it. Before host-to-device copy timing, synchronize the relevant stream. For GPU kernel timing, use CUDA events rather than Python wall-clock around asynchronous launches. For end-to-end TTFT, use monotonic timestamps at the ingress and first-token event. [PyTorch CUDA events](https://pytorch.org/docs/stable/generated/torch.cuda.Event.html) are the appropriate reader-reproducible primitive for device elapsed time.

```python
import torch

def time_cuda(fn, warmup=10, repeats=50):
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

The expected result is a per-call millisecond value on the reader’s GPU; no fixed range is asserted here because adapter size, PCIe topology, driver, and kernel implementation vary. Label the resulting table `reproduce: bench_residency.py`, include the command line, and publish the hardware and software versions.

Do not report only tok/s at batch 1. A residency-aware service needs:

| Metric | Definition | Why it matters | Source |
|---|---|---|---|
| TTFT | accepted request to first token | exposes queue and cold starts | derived: timestamp difference |
| TPOT | mean inter-token decode time | warm decode cadence | derived: timestamp difference |
| p99 TTFT | 99th percentile by route | catches cache-miss tails | reproduce: load script |
| adapter hit rate | resident adapter uses / adapter uses | evaluates cache policy | reproduce: trace replay |
| goodput | requests meeting TTFT and TPOT SLO | rejects fast-but-late work | derived: SLO count / arrivals |
| transfer bytes | host-to-device bytes per request | quantifies swap pressure | reproduce: profiler |

Run both open-loop and closed-loop load. In open-loop, arrivals follow a target process even when the queue grows, exposing overload and admission behavior. In closed-loop, each client waits for its response before sending another request, which can hide queue collapse. Report both if the service is interactive and batch-oriented.

For every result, include a Source column. Derived arithmetic is enough for memory examples. Cited numbers must link to the original source and retain its date and setup. Reproducible numbers need a script and an expected range on named hardware before the benchmark is run. The expected range is not a license to invent a result; it is a contract for what another engineer can check.

#### Worked example: cost of a miss

Take the illustrative 4 GiB adapter and 12 GiB/s effective transfer path. The lower-bound copy time is $4/12 = 0.333$ seconds. Add a derived 50 ms initialization and a derived 80 ms prefill example: $333 + 50 + 80 = 463$ ms before queueing. If the adapter is resident, the same illustrative path is $50 + 80 = 130$ ms. The 333 ms and 130 ms are arithmetic examples, not measured service values. A reader should run `bench_residency.py` and report a range with the GPU, link, adapter bytes, and driver.

## 10. Failure modes that look like model quality problems

### The adapter is loaded but the batch is wrong

A request can use the correct adapter weights and still be placed in a batch with an incompatible tokenizer, grammar mask, or dtype. The symptom is a shape error, a silent fallback to a slower path, or output that differs from the reference. Log the batch key at kernel launch and test one request per adapter against a non-batched reference implementation.

### The cache thrashes at a high hit rate

An aggregate hit rate can look good while the tail is bad. Suppose 99 requests use one hot adapter and one request uses a rotating long-tail adapter; the hot adapter can still be evicted exactly when the long-tail request arrives if the cache has no protected set. Track misses by adapter and by queue state. Protect active streams and use a minimum residency lease.

### Prefetch steals bandwidth from decode

Asynchronous transfer is not free. A prefetch can contend with weight reads, KV traffic, or another tenant’s transfer. Assign prefetch a lower-priority stream, cap its bytes per second, and cancel it when the predicted request disappears. Measure TPOT with and without prefetch; a lower TTFT is not a win if the warm stream’s p99 TPOT violates its SLO.

### Fallback duplicates work

A cascade can increase GPU load precisely when the system is busy. If the small model rejects many requests during an incident, every rejected request runs two models and can make the large-model queue worse. Add a fallback budget: when large-model queue time exceeds a threshold, either route directly to the large model for high-risk requests or return a bounded degraded response. The right behavior depends on the product contract, but “always escalate” is not a resilience policy.

### Weight swapping invalidates compiled state

Compiled graphs and fused kernels may assume tensor shapes, quantization scales, or adapter slots. Swapping bytes without invalidating the associated graph can produce incorrect results or a hard-to-debug CUDA error. Store a residency generation and compilation signature with every device handle. A new generation must be visible to the scheduler before it admits a request.

### Cancellation leaks leases

If a client disconnects while an adapter transfer is in flight, the transfer may complete and the lease may remain active forever. Tie every transfer and decode lease to a cancellation token. On cancellation, mark the request inactive, wait for the stream event, then release the lease. Do not free a tensor while a CUDA kernel may still read it.

## Case studies / real numbers

### 1. PagedAttention changed the unit of residency

The [vLLM PagedAttention paper](https://arxiv.org/abs/2309.06180), published in 2023, frames KV cache management as a paging problem and reports benchmark comparisons for its experimental setup. The relevant lesson here is not to copy one headline percentage. It is that serving capacity depends on how memory is allocated and shared across sequences. A model-placement controller that ignores block allocation can load every adapter correctly and still OOM on the first long prompt.

The constructive takeaway for `nanoserve` is to ask the allocator for blocks before admitting a request. Model residency and KV residency are coupled reservations. This post’s 128 KiB per token is derived from the fixed Llama configuration; the paper’s measured results remain cited to its hardware, workloads, and implementation.

### 2. vLLM Multi-LoRA on MoE models

The vLLM team’s [Multi-LoRA post](https://vllm.ai/blog/2026-02-26-multi-lora) dated 2026-02-26 describes fused MoE-LoRA kernels, Split-K, and a comparison with eight adapters at rank 32 on GPT-OSS 20B. It reports 454% OTPS improvement and 87% lower TTFT for that version and setup. Those are cited values, not portable expectations for dense Llama-3.1-8B. The mechanism is still directly relevant: one base model and many low-rank updates need a kernel that understands adapter identity rather than a process per tenant.

The limitation matters. The post does not establish a universal hot-swap or cold-start latency. Keep the loading policy separate from the fused compute path, and benchmark both. A fast adapter kernel cannot compensate for a 4 GiB transfer on every request.

### 3. vLLM Sleep Mode separates wake policy from serving

The [vLLM Sleep Mode post](https://vllm.ai/blog/2025-10-26-sleep-mode) dated 2025-10-26 describes level 1 offload to CPU memory and level 2 weight discard. It reports example wake times for small models and a five-switch workload comparison, including 112.6 seconds versus 357.1 seconds in its stated setup. These numbers are cited to that post and should not be restated as a promise for this article’s service.

The design lesson is valuable: “not resident” can mean “offloaded and quickly wakeable” or “discarded and reloadable.” `nanoserve` should represent those as different states, with different admission costs and RAM requirements. Level 2-like discard also requires cache invalidation and trusted control-plane access; a public endpoint must never expose an arbitrary wake or load primitive.

### 4. AIBrix makes placement a control-plane concern

The [AIBrix release description](https://vllm.ai/blog/2025-02-21-aibrix-release) dated 2025-02-21 names high-density LoRA management, LLM gateway/routing, heterogeneous serving, and distributed KV cache as separate control-plane components. It is thin on quantitative results, so this article uses it for vocabulary only. The architecture reinforces the boundary: routing, residency, and autoscaling need coordination outside the kernel, while the engine still needs precise per-request reservations inside the GPU process.

## When to reach for this (and when not to)

Reach for a residency-aware multi-model layer when:

- many adapters share one base and requests are small enough that separate replicas waste device memory;
- the hot adapter set is smaller than the catalog and reuse is measurable;
- cold-start TTFT is a product constraint rather than an occasional batch-job cost;
- a small model can produce a useful answer with a calibrated, task-specific fallback;
- operators need an explanation for every load, eviction, route, and admission decision.

Use vLLM instead of extending a toy engine when you need production kernels, continuous batching, tensor or pipeline parallelism, quantization integrations, robust cancellation, and a large compatibility surface. Build `nanoserve` to learn the invariants, to prototype a policy, or to own a narrow deployment where the policy is the product. Do not write a new scheduler because the model registry has an attractive API.

Skip multi-LoRA when adapters are large, traffic is uniformly cold, or each request needs a different base model. Skip cascades when a false accept is more expensive than one large-model call and no reliable verifier exists. Skip aggressive prefetch when it steals bandwidth from a latency-sensitive warm stream. Skip hot-swapping when the complete working set fits comfortably and a simple pinned deployment meets the SLO.

## Key takeaways

1. The capacity of a multi-model GPU service is its resident set, not its catalog size.
2. LoRA shares base weights, but adapter identity must remain explicit in the batch key and authorization path.
3. Treat adapters as cache entries with leases, generations, byte limits, and observable eviction reasons.
4. Cold TTFT is queue plus residency plus initialization plus prefill; do not hide it inside one latency number.
5. Prefetch can hide a transfer only when independent GPU work exists and the copy does not damage warm TPOT.
6. A cascade is a risk policy. Calibrate its gate and budget fallback work during overload.
7. Reserve model, adapter, and KV bytes atomically before admitting a request.
8. Benchmark warm and cold states separately, with open-loop and closed-loop load, and publish provenance for every number.
9. Use vLLM for the production engine unless the placement policy itself is the reason to build.

## Further reading

- [What inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is)
- [The inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook)
- [Continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention)
- [Request scheduling and preemption](/blog/machine-learning/model-serving/request-scheduling-and-preemption)
- [vLLM Multi-LoRA](https://vllm.ai/blog/2026-02-26-multi-lora)
- [vLLM Sleep Mode](https://vllm.ai/blog/2025-10-26-sleep-mode)
- [PagedAttention paper](https://arxiv.org/abs/2309.06180)
- [PyTorch CUDA events](https://pytorch.org/docs/stable/generated/torch.cuda.Event.html)

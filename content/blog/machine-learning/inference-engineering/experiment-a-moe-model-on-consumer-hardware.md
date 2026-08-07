---
title: "Experiment a MoE Model on Consumer Hardware: Why 3B Active Does Not Mean 3B Fast"
date: "2026-08-03"
publishDate: "2026-08-03"
description: "Design an honest Qwen3-30B-A3B experiment for a 24 GB GPU, derive the memory ceiling, and learn when expert offload makes a model fit but makes decode slower."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "mixture-of-experts",
    "qwen3",
    "gpu",
    "quantization",
    "latency",
    "throughput",
    "ml-systems",
    "vllm",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 40
---

The number that gets repeated about a sparse mixture-of-experts model is usually the least useful number for a consumer GPU: active parameters. Qwen3-30B-A3B has 30.5 billion total parameters and 3.3 billion activated parameters, according to the [official Qwen3 model card](https://huggingface.co/Qwen/Qwen3-30B-A3B) (accessed 2026-08-03). That sounds like a 3B model wearing a 30B label. It is not.

The GPU still needs a place for the expert weights, router, attention projections, embeddings, normalization layers, temporary activations, and the KV cache. Routing saves arithmetic for each token. It does not make unused expert weights disappear from storage. The first figure is the mental model: one token wakes eight experts, while the other 120 experts remain part of the deployment footprint.

![A routed token activates eight Qwen3 experts while the remaining resident experts still consume placement memory](/imgs/blogs/experiment-a-moe-model-on-consumer-hardware-1.webp)

This post is an experiment design, not a benchmark report. There is no GPU run behind the ranges below. We will build the arithmetic, write a CPU-runnable planning layer for `nanoserve`, and define a reader-reproducible GPU harness whose output should be reported as a range on a named card. The target is a single RTX 4090 with 24 GB of GDDR6X memory, the capacity NVIDIA lists on its [official product page](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/) (accessed 2026-08-03). The goal is to decide whether Qwen3-30B-A3B or gpt-oss-20b belongs on that card, in what format, and with which failure mode visible.

## 1. The experiment starts with a precise question

The wrong question is “How fast is this MoE?” It hides at least five experiments:

| Question | Configuration that answers it | Metric that matters | Source |
|---|---|---|---|
| Can the model load? | One request, cold load, fixed dtype | peak allocated VRAM | derived from the budget below |
| Is it interactive? | batch one, short prompt, 128 output tokens | TTFT and TPOT p50/p99 | reproduce: `bench_moe.py` |
| Does it serve traffic? | concurrency sweep, mixed lengths | goodput and queue time | reproduce: `bench_moe.py` |
| Does offload help? | same model, same prompts, host experts | TPOT plus PCIe transfer time | reproduce: `bench_moe.py` |
| Is quality preserved? | same seed and task suite | exact match, pass rate, perplexity | reproduce: evaluation harness |

TTFT means time to first token: queue time plus prefill plus scheduling overhead until the first output token. TPOT means time per output token after that first token. A model can have acceptable TTFT and painful TPOT if every decode step streams expert weights. It can also have high throughput and an unusable p99 when the router creates a hot expert or when the host transfer queue backs up.

> Active parameters predict the arithmetic of a token. Resident parameters predict the storage bill. A serving system pays both.

Keep the experiment closed over everything except the question under test. Fix the model revision, tokenizer revision, prompt corpus, maximum new tokens, random seed, dtype, GPU power mode, driver, CUDA build, and serving backend. Run resident and offloaded configurations with the same request order. If the offload path uses a different kernel or a different quantization format, it is a new point on the frontier, not a pure offload comparison.

### A useful baseline: gpt-oss-20b

OpenAI’s [gpt-oss-20b model card](https://huggingface.co/openai/gpt-oss-20b) (accessed 2026-08-03) describes 21 billion total parameters and 3.6 billion active parameters. It also states that the model uses MXFP4 quantization for MoE weights and can run within 16 GB of memory. That is a cited deployment fact for the released model format, not a promise about arbitrary BF16 conversion, arbitrary kernels, or arbitrary context length. It makes gpt-oss-20b a good second candidate: it separates “sparse routing” from “the released checkpoint already has a memory-saving format.”

Qwen3-30B-A3B is a more instructive stress case for `nanoserve`: 128 experts, eight activated experts, 48 layers, 32 query heads, four KV heads, and 32,768 native context tokens are all listed by Qwen. The model card reports 3.3B activated rather than asking us to infer it. Every later budget uses those values or labels the quantity as an assumption.

## 2. What active parameters actually remove

The phrase “active parameters” refers to the parameters in the selected expert paths for one token. In a typical sparse MoE layer, a router produces a score for each expert, selects top-$k$, dispatches the token to those experts, and combines their outputs with routing weights. Qwen3-30B-A3B’s card reports 128 experts and eight activated experts. The activated fraction of the expert set is therefore derived as

$$
f_{\text{expert-active}} = \frac{8}{128} = 0.0625 = 6.25\%.
$$

That is a routing fraction, not a total-model memory fraction. Attention and non-expert layers execute for every token. Router scores execute for every token. The selected expert GEMMs execute for every token. Dispatch and merge execute for every token. The fraction of the total model that runs can be much larger than 6.25% because the dense trunk is always present.

For a simplified feed-forward layer with hidden width $d$, intermediate width $m$, and a gated up/down path, one expert’s dominant multiply-add count is approximately

$$
C_{\text{expert}} \approx 3dm
$$

where the factor three represents two input projections and one output projection. This is an explanatory abstraction, not Qwen3’s exact implementation formula. With eight selected experts, the sparse expert work is approximately $8 \cdot 3dm$ instead of $128 \cdot 3dm$. The router does not give us the ability to pretend that all 128 experts are absent from the checkpoint.

The arithmetic benefit is also sensitive to batch shape. At batch one, each selected expert may receive one token. That creates many skinny matrix multiplications. At larger batch or with balanced routing, tokens assigned to the same expert can be grouped into a more efficient matrix multiplication. The ideal arithmetic reduction can therefore coexist with poor hardware utilization.

### The dispatch tax

Routing converts one dense batch into expert-specific sub-batches. If $B$ is the number of tokens in the decode step, $E$ is the number of experts, and $k$ is top-$k$, the average token count per expert under perfectly uniform routing is

$$
\mathbb{E}[n_e] = \frac{Bk}{E}.
$$

For Qwen3-30B-A3B, $E=128$ and $k=8$. At batch one, this expectation is $8/128=0.0625$ tokens per expert. That does not mean fractional tokens execute; it means most experts receive zero tokens and selected experts receive tiny groups. At $B=128$, the expected assignment count is $128\cdot8/128=8$ assignments per expert. The same model is much easier to batch at 128 decode tokens than at one decode token, assuming the requests can be scheduled together.

This is why “3B active” does not mean “3B fast.” It says the router narrows the expert arithmetic. It says nothing about small GEMM efficiency, memory locality, launch overhead, router imbalance, or the time needed to fetch a cold expert.

![A decode token passes through routing dispatch eight expert GEMMs and merge before repeating through all 48 layers](/imgs/blogs/experiment-a-moe-model-on-consumer-hardware-5.webp)

## 3. Derive the 24 GB VRAM ceiling before loading anything

The memory calculation should happen before `from_pretrained`. The official RTX 4090 product page says 24 GB. Marketing gigabytes are decimal while CUDA allocation tools commonly expose bytes and GiB, so use the explicit conversion

$$
24\ \text{GB} = 24\times10^9\ \text{bytes} \approx 22.35\ \text{GiB}.
$$

That is the physical capacity. A serving process should not plan to consume all of it. Let $R$ be a runtime reserve for the CUDA context, kernels, allocator fragmentation, temporary buffers, and non-model allocations. A planning reserve of 2–3 GiB is a reader-tunable assumption, not a measurement. With the middle value 2.5 GiB, the planning budget is

$$
M_{\text{usable}} \approx 22.35 - 2.5 = 19.85\ \text{GiB}.
$$

![A 24 GB RTX 4090 budget subtracts runtime reserve before weights and KV cache receive capacity](/imgs/blogs/experiment-a-moe-model-on-consumer-hardware-2.webp)

### Dense BF16 arithmetic is a fast rejection test

For a BF16 checkpoint, a first-order weight estimate is two bytes per parameter. Qwen3-30B-A3B has 30.5B total parameters, so

$$
M_{\text{weights,bf16}} \approx 30.5\times10^9\times2\ \text{bytes}
 = 61.0\times10^9\ \text{bytes}
 \approx 56.8\ \text{GiB}.
$$

That already exceeds the approximately 19.85 GiB planning budget by a factor derived as $56.8/19.85\approx2.86$. No KV cache, activations, or fragmentation term can repair that. BF16 Qwen3-30B-A3B is not a one-4090 resident experiment.

For gpt-oss-20b, applying the same rough BF16 arithmetic to 21B parameters gives $42.0\times10^9$ bytes, or approximately $39.1$ GiB. The model card’s released MXFP4 path is a different storage representation, so do not compare the BF16 estimate to its cited “within 16 GB” statement as if they were the same checkpoint format.

### Quantization is not one number

An ideal $b$-bit parameter store uses $b/8$ bytes per parameter, but real formats add scales, zero points, alignment, metadata, and sometimes unquantized layers. The ideal lower-bound arithmetic for Qwen3’s 30.5B parameters is:

| Representation | Ideal weight bytes | Ideal decimal GB | Status |
|---|---:|---:|---|
| BF16 | $30.5\mathrm{B}\times2=61.0$ GB | 61.0 | derived lower-order estimate |
| INT8 | $30.5\mathrm{B}\times1=30.5$ GB | 30.5 | derived, still too large |
| INT4 | $30.5\mathrm{B}\times0.5=15.25$ GB | 15.25 | derived ideal, metadata excluded |
| 3-bit | $30.5\mathrm{B}\times0.375=11.4375$ GB | 11.4375 | derived ideal, kernel-dependent |

The INT4 row is not a deployment result. It is a capacity screen. If the actual format overhead is 15%, the estimate becomes $15.25\times1.15=17.5375$ GB before runtime reserve, KV cache, and activations. That leaves only $19.85-17.54=2.31$ GiB in the middle-reserve plan. A model that “fits” at zero context can still fail when the first request grows its cache.

The gpt-oss-20b model card provides a stronger cited anchor: the released MXFP4 MoE weights are described as fitting within 16 GB of memory. Treat this as a property of the released model and supported software path. The reader should inspect `torch.cuda.max_memory_allocated()` after loading, because “within 16 GB” does not tell us how much KV cache a particular backend leaves available.

#### Worked example: the first request consumes the last GiB

Assume a reader measures 16.0 GiB for a gpt-oss-20b MXFP4 load on an RTX 4090. This is a reader-reproducible measurement to obtain, not a result from this post. With the 22.35 GiB physical conversion and a 2.5 GiB planning reserve, the remaining runtime budget is

$$
22.35 - 2.5 - 16.0 = 3.85\ \text{GiB}.
$$

If a backend allocates 2.0 GiB of temporary workspace and 1.0 GiB of KV cache for a short prompt, the arithmetic leaves $3.85-2.0-1.0=0.85$ GiB. The experiment may pass at short context and fail at a longer prompt even though the weight load itself looked comfortable. The conclusion is derived from the stated assumptions; the 16.0 GiB input must be measured on the named card.

## 4. KV cache still grows with the total architecture

MoE sparsity does not automatically make attention memory sparse. Qwen3’s model card reports 48 layers, four KV heads, and context length 32,768 natively. For a standard grouped-query attention cache, the bytes per token are

$$
B_{\text{KV/token}} = 2\cdot L\cdot H_{kv}\cdot d\cdot b,
$$

where the first two is K plus V, $L$ is the layer count, $H_{kv}$ is the number of KV heads, $d$ is head dimension, and $b$ is bytes per element. Qwen3’s public card does not state `head_dim` in the overview, so the following is explicitly a reader-reproducible calculation once `model.config.head_dim` is printed. Do not substitute a guessed dimension into a claimed model result.

The measurement code is small:

```python
# nanoserve/moe_budget.py
from dataclasses import dataclass

@dataclass(frozen=True)
class AttentionConfig:
    layers: int
    kv_heads: int
    head_dim: int
    bytes_per_element: int = 2

def kv_bytes_per_token(cfg: AttentionConfig) -> int:
    return 2 * cfg.layers * cfg.kv_heads * cfg.head_dim * cfg.bytes_per_element

def kv_gib(tokens: int, cfg: AttentionConfig) -> float:
    return tokens * kv_bytes_per_token(cfg) / (1024 ** 3)

if __name__ == "__main__":
    cfg = AttentionConfig(layers=48, kv_heads=4, head_dim=128)
    print(kv_bytes_per_token(cfg), "bytes/token")
    print(f"{kv_gib(32768, cfg):.3f} GiB at 32,768 tokens")
```

The `head_dim=128` line is an assumption for a planning run, not a cited Qwen3 fact. A reader should replace it with the model configuration and rerun. Under this labeled assumption, the arithmetic is

$$
2\cdot48\cdot4\cdot128\cdot2 = 98{,}304\ \text{bytes/token},
$$

and $98{,}304\times32{,}768=3{,}221{,}225{,}472$ bytes, approximately 3.00 GiB. That cache is per sequence at 32,768 tokens under the assumption. At 8,192 tokens, it is one quarter of that because the cache is linear in token count: approximately 0.75 GiB.

This matters on a 24 GB card because the weight representation determines whether the remaining capacity is measured in GiB or in hundreds of MiB. A 16 GiB checkpoint plus 3 GiB of KV plus 2 GiB of workspace plus 2.5 GiB reserve equals 23.5 GiB in the decimal-style mental arithmetic; after converting consistently to GiB and accounting for allocator overhead, the margin is thin. Never call the margin “free VRAM” until the backend’s peak allocation confirms it.

### Paged allocation changes fragmentation, not physics

Paged KV cache stores token history in fixed-size blocks rather than requiring one contiguous allocation per request. If block size is $P$ tokens and a sequence has $S$ tokens, the allocated blocks are $\lceil S/P\rceil$. The internal slack is

$$
\text{slack}(S,P)=\left\lceil\frac{S}{P}\right\rceil P-S.
$$

At $S=129$ and $P=16$, the allocation is $\lceil129/16\rceil=9$ blocks, or 144 token slots, with 15 slots of slack. That is derived and independent of whether the model is dense or MoE. The value of paging is that free blocks can be scattered and shared by many request lifetimes. It does not reduce the bytes per live token.

The production contrast is vLLM, whose PagedAttention design is the benchmark target for this series. The relevant question for this experiment is not “can `nanoserve` invent a better cache?” It is “does the MoE weight placement leave enough cache blocks for the workload that matters?”

## 5. Expert offload: the escape hatch with a wire attached

If all experts do not fit on the GPU, one option is to keep some weights in pinned host memory and copy selected experts to the device. The trade is simple to state: lower device residency in exchange for host-device traffic. The difficult part is that decode has one token per request per step, so the selected expert group can be too small to amortize the transfer.

![Resident experts avoid transfer traffic while offloaded experts exchange VRAM headroom for PCIe work on every cold route](/imgs/blogs/experiment-a-moe-model-on-consumer-hardware-3.webp)

Let $W$ be the bytes transferred for one selected expert group, and $\beta$ the sustained link bandwidth available to that copy. A lower-bound transfer time is

$$
t_{\text{copy}} \geq \frac{W}{\beta}.
$$

This is a bandwidth lower bound, not an expected end-to-end latency. It omits launch overhead, synchronization, contention, NUMA placement, page faults, and the expert compute itself. If the host path transfers 256 MiB and sustains 12 GB/s, the arithmetic lower bound is $0.256/12$ seconds when units are decimal GB, or about 21.3 ms. If it sustains 24 GB/s, the lower bound is about 10.7 ms. Those are illustrative derived scenarios; the reader must measure the actual copy rate on the motherboard and topology.

That lower bound is already comparable to an interactive TPOT target. If the transfer happens once per layer and the model has 48 layers, multiplying 48 by 10.7 ms would be absurdly pessimistic because reuse and overlap may apply, but it makes the risk obvious: a naive “copy expert, compute expert, discard expert” loop cannot be evaluated with weight size alone. It must expose copies and compute on a timeline.

There are four offload designs worth comparing:

| Design | Device memory | Transfer behavior | Likely failure mode | Source |
|---|---|---|---|---|
| all experts resident | highest | no expert copy in decode | model or KV does not fit | derived |
| cold experts host-resident | lower | copy when route misses | TPOT and p99 spikes | reproduce: profiler |
| expert cache on device | bounded | copy on eviction miss | hot-expert thrashing | reproduce: route trace |
| quantized resident experts | lower | no copy, dequant in kernel | quality or kernel support | reproduce: quality + TPOT |

The expert cache is a real cache with a policy. LRU is a reasonable first implementation, but router popularity can be skewed and can change with the prompt mix. If one expert is hot, LRU works. If the workload alternates between two disjoint expert populations, a small cache thrashes. Measure the route histogram, not only the mean number of selected experts.

### A runnable offload ledger

The following CPU-only class does not pretend to move tensors. It makes the accounting explicit so a GPU implementation has a testable contract. It records misses, bytes, and the decision to keep an expert resident.

```python
# nanoserve/expert_cache.py
from collections import OrderedDict
from dataclasses import dataclass

@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    bytes_loaded: int = 0

class ExpertCache:
    def __init__(self, capacity: int, expert_bytes: int):
        if capacity < 1 or expert_bytes < 1:
            raise ValueError("capacity and expert_bytes must be positive")
        self.capacity = capacity
        self.expert_bytes = expert_bytes
        self.resident = OrderedDict()
        self.stats = CacheStats()

    def touch(self, expert_id: int) -> bool:
        if expert_id in self.resident:
            self.resident.move_to_end(expert_id)
            self.stats.hits += 1
            return True
        self.stats.misses += 1
        self.stats.bytes_loaded += self.expert_bytes
        self.resident[expert_id] = None
        self.resident.move_to_end(expert_id)
        while len(self.resident) > self.capacity:
            self.resident.popitem(last=False)
        return False

if __name__ == "__main__":
    cache = ExpertCache(capacity=2, expert_bytes=256 * 1024 * 1024)
    for expert in [1, 2, 1, 3, 1, 2]:
        cache.touch(expert)
    print(cache.stats)
```

The output is deterministic for the supplied sequence: four misses, two hits, and four times 256 MiB loaded, which is 1,024 MiB by the stated arithmetic. That output tests policy, not GPU performance. The production version should replace `expert_bytes` with the serialized tensor group size, use pinned memory, and record CUDA events around copies and expert kernels.

### The overlap question

An offload design can hide transfer only if the next transfer can overlap the current expert compute and if the next expert is known early enough. Router decisions are available after the router computation, but the full layer schedule, stream dependencies, and memory slots still matter. A copy that is technically asynchronous can become synchronous at the next operation that consumes the tensor.

Use a timeline trace with at least three lanes: router, host-to-device copies, and expert GEMMs. Record the expert ID, copy bytes, CUDA stream, start time, end time, and whether the consumer waited. Do not report “async offload” from the use of `non_blocking=True` alone.

## 6. Build the measurement harness around the model, not the headline

![A decision tree maps fit latency and throughput goals to quantization offload batching and expert-balance experiments](/imgs/blogs/experiment-a-moe-model-on-consumer-hardware-6.webp)

The harness should answer a narrow question with repeatable inputs. Use four prompt families from the series protocol: chat with short input and long output, RAG with long input and short output, code completion, and translation. Fix input and output token budgets separately. A 2,048-token prompt and 128 generated tokens is not interchangeable with a 128-token prompt and 2,048 generated tokens.

The following is a runnable PyTorch timing skeleton. It intentionally does not include a model-specific MoE kernel. It can be adapted to a Transformers or vLLM endpoint and it refuses to claim a result when CUDA is unavailable.

```python
# nanoserve/bench_moe.py
import argparse
import statistics
import time
import torch

def percentile(values, p):
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((p / 100) * (len(values) - 1))))
    return values[index]

def time_cuda(fn, warmup=10, steps=40):
    if not torch.cuda.is_available():
        raise RuntimeError("run this harness on the named CUDA GPU")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return {"p50_ms": percentile(samples, 50),
            "p99_ms": percentile(samples, 99),
            "mean_ms": statistics.mean(samples)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=128)
    args = parser.parse_args()
    x = torch.randn(args.batch, 4096, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    result = time_cuda(lambda: torch.mm(x, w))
    result["batch"] = args.batch
    result["tokens"] = args.tokens
    print(result)

if __name__ == "__main__":
    main()
```

This measures a controlled GEMM, not Qwen3 or gpt-oss. It demonstrates the timing discipline: warm up, record CUDA events, synchronize the event, and report a distribution. The model harness should wrap one decode step and separately record TTFT, then use the same prompts across configurations. The `--tokens` placeholder should become the generated-token budget in a full implementation; do not label the sample GEMM output as model tok/s.

For a reader-reproducible expectation, state ranges before the run. For example: “On an RTX 4090, with the exact released checkpoint, exact backend, batch one, 128 output tokens, and a clean process, expect the TPOT result to land in the range printed by your own 40-step harness; report the observed range and environment.” That is honest but intentionally not a fabricated numeric range. There is no public source in scope that supports a universal Qwen3-30B-A3B TPOT range for every quantization and offload stack. A precise number without those conditions would be theater.

### Open-loop and closed-loop load

Closed-loop load sends the next request only after the previous one finishes. It answers “how fast can this client consume results?” It does not create queueing pressure. Open-loop load samples arrivals independently, for example from a Poisson process, and records queue time. It answers “what happens when users arrive regardless of current service time?”

Use closed-loop batch-one runs for a clean TPOT comparison, then open-loop traffic for goodput and p99. Goodput should count only tokens from requests that satisfy an SLO such as TTFT below a chosen threshold and TPOT below a chosen threshold. The threshold is part of the experiment design; do not smuggle it into the result after seeing the curve.

## 7. Compare resident, quantized, and offloaded routes

The experiment matrix should contain at least these configurations:

![The experiment matrix compares format placement risk and the metric required to make a decision](/imgs/blogs/experiment-a-moe-model-on-consumer-hardware-4.webp)

1. Qwen3-30B-A3B in a supported quantized format with all expert weights resident if it fits.
2. Qwen3-30B-A3B with a device expert cache and pinned-host spill.
3. gpt-oss-20b in its released MXFP4 path with the same prompt suite.
4. A smaller dense model as a control, because sparse routing overhead is impossible to interpret without a dense reference.

The dense control is not meant to win the quality contest. It gives us a way to separate “MoE is slow on this backend” from “this GPU is memory-bound at the chosen batch.” Keep the control at the same dtype where possible and use the same tokenizer accounting.

The right comparison table has no invented cells:

| Configuration | Fits 24 GB? | TTFT | TPOT | p99 | Quality | Provenance |
|---|---|---|---|---|---|---|
| Qwen3 resident quantized | run gate | run | run | run | run | reproduce: harness |
| Qwen3 host expert spill | run gate | run | run | run | run | reproduce: harness |
| gpt-oss-20b MXFP4 | model card says within 16 GB; verify locally | run | run | run | run | cited + reproduce |
| dense control | run gate | run | run | run | run | reproduce: harness |

The phrase “run gate” is deliberate. A configuration that OOMs is a result: record the failing allocation phase, peak allocated memory, reserved memory, prompt length, and backend version. Do not replace it with “not applicable.”

### Expert balance is part of latency

The mean route count does not expose imbalance. Let $c_e$ be the number of token assignments to expert $e$ in one step. The coefficient of variation is

$$
\mathrm{CV}(c)=\frac{\sigma(c_1,\ldots,c_E)}{\mu(c_1,\ldots,c_E)}.
$$

This is a descriptive statistic, not a claim that CV alone predicts latency. A high value says work is uneven; whether that creates a visible tail depends on the kernel and scheduler. Log the maximum expert load and the median expert load as well. If a step sends 32 assignments to one expert and 2 to another, the largest group can determine the completion time even when the total assignment count is fixed.

The vLLM team’s public gpt-oss support post, dated 2025-08-05, describes 32 experts for gpt-oss-20b and top-4 routing, plus MXFP4 for MoE weights. It is useful contrast: production support includes model-specific kernels and backend work that `nanoserve` is intentionally exposing. It is not a consumer RTX 4090 benchmark.

## 8. Why lower active compute can still lose at batch one

There are four independent reasons.

First, the router and dispatch add work around the expert GEMMs. Second, each selected expert may receive so few rows that the GEMM is launch- and utilization-limited. Third, attention and dense layers remain unchanged. Fourth, if the selected expert is offloaded, memory movement adds a lower-bound delay before compute can begin.

The last point is easy to miss in a FLOP-only analysis. A dense model may read a stable resident weight stream. An offloaded MoE may read fewer weights in total but pay more synchronization events. “Fewer bytes overall” and “shorter critical path” are not equivalent.

#### Worked example: 8 experts are not 8 useful GEMMs

Consider batch one, top-8 routing, and 128 experts. The expected assignment count per expert is $8/128=0.0625$. A concrete route can select eight distinct experts, giving one row to each. The model has done sparse expert arithmetic, but the hardware sees eight tiny expert workloads plus router, dispatch, and merge. At batch 128, the expected assignment count becomes $128\cdot8/128=8$ per expert. The latter is a more favorable grouping assumption, but actual routing variance can still create hot experts.

The numbers are derived from Qwen3’s cited 128-expert and eight-active-expert configuration. The latency outcome is not derived; it is reader-reproducible only by running the same kernel and measuring CUDA events. This distinction is the core discipline of the post.

### The batch sweep

Run batch sizes 1, 2, 4, 8, 16, 32, and 64 if memory permits. For each, report:

* tokens presented to the model per decode step;
* selected expert IDs and per-expert assignment counts;
* router, dispatch, expert, merge, and attention time;
* device memory before and after the step;
* TTFT, TPOT, p50, p99, and completed output tokens;
* whether any request was preempted or evicted.

Batch size is not the same as active tokens under continuous batching. If requests finish at different times, the active set changes every step. A good `nanoserve` scheduler should expose the token batch and route histogram as first-class trace fields.

## 9. Implement a route trace before writing a faster kernel

The first `nanoserve` component should be a route trace. A route trace is cheap, debuggable, and tells us whether the problem is imbalance, cold expert misses, or poor grouping.

```python
# nanoserve/route_trace.py
from collections import Counter
from dataclasses import dataclass, field

@dataclass
class RouteTrace:
    layer: int
    top_k: int
    assignments: Counter = field(default_factory=Counter)
    misses: int = 0
    bytes_loaded: int = 0

    def add(self, expert_ids, was_hit, loaded_bytes=0):
        if len(expert_ids) != self.top_k:
            raise ValueError(f"expected {self.top_k} experts, got {len(expert_ids)}")
        self.assignments.update(expert_ids)
        self.misses += int(not was_hit)
        self.bytes_loaded += loaded_bytes

    def summary(self):
        loads = list(self.assignments.values())
        return {
            "layer": self.layer,
            "experts_touched": len(loads),
            "max_assignments": max(loads, default=0),
            "total_assignments": sum(loads),
            "misses": self.misses,
            "bytes_loaded": self.bytes_loaded,
        }

if __name__ == "__main__":
    trace = RouteTrace(layer=0, top_k=8)
    trace.add([1, 2, 3, 4, 5, 6, 7, 8], was_hit=False,
              loaded_bytes=8 * 1024 * 1024)
    print(trace.summary())
```

The sample summary contains eight touched experts, eight total assignments, one miss event, and 8 MiB loaded by the supplied argument. It is not a claim about Qwen3 expert tensor size. The trace is useful because it keeps those distinctions visible.

Once the trace is correct, test three implementations in order: a reference PyTorch route on CPU, a GPU route with resident expert tensors, and only then an offloaded route. Compare logits or a stable checksum between implementations for fixed inputs. A faster route that silently changes expert ordering or routing weights is not a performance win.

### Capacity-aware admission

The scheduler must know that an MoE request has two memory curves: a relatively fixed weight footprint and a context-growing KV footprint. A request admission rule can reserve

$$
M_{\text{request}} = M_{\text{KV}}(S_{\text{prompt}}+S_{\text{generated}}) + M_{\text{activation-peak}} + M_{\text{temporary}}.
$$

This is an explanatory accounting abstraction, not an exact backend allocation equation. The measured peak may be higher because kernels use workspace and because allocations overlap differently. Still, without the terms, a scheduler sees only “model loaded” and admits requests until the first long context causes an OOM.

The admission policy should reserve KV blocks for the maximum allowed output or use a clearly documented preemption policy. If expert offload has a device cache, reserve its capacity too. An expert cache that competes invisibly with KV blocks is not a cache policy; it is an OOM lottery.

## 10. What a fair consumer-hardware result looks like

The result should be a Pareto frontier, not one winner. Plot memory against TPOT, TPOT against throughput, and quality against memory. Mark OOM configurations at the point where they fail. Annotate every point with model revision, format, batch, prompt length, output length, concurrency, GPU, driver, and backend version.

![Resident quantized weights and host-offloaded experts occupy different corners of a memory latency throughput frontier](/imgs/blogs/experiment-a-moe-model-on-consumer-hardware-7.webp)

At minimum, produce these plots from the same raw event log:

1. peak allocated GiB versus maximum context length;
2. TPOT p50 and p99 versus active token batch;
3. completed output tok/s versus concurrency;
4. expert cache hit rate versus TPOT;
5. quality score versus effective weight format.

The first plot should show the distinction between fixed model memory and linear KV growth. The second should reveal whether offload adds a long tail. The third should show whether batching turns tiny expert groups into useful work. The fourth connects the route trace to the latency result. The fifth stops a “faster” quantized model from being accepted when it has silently degraded the application task.

### Reproducibility envelope

Name the hardware exactly: RTX 4090 24 GB, not “a 4090-class GPU.” Name the model commit, quantization artifact, backend commit, PyTorch version, CUDA version, driver, and whether the display server shares the card. Lock clocks only if the reader is allowed to do so, and record the command. Warm up until allocator and kernel selection stabilize, then measure steady state. Include the first-run load time separately.

If a result is expected rather than observed, label it as an expected range on the named hardware and explain how to reproduce it. For example, the safe statement is: “On an RTX 4090, a resident quantized configuration should have lower transfer time than the same configuration with host expert spill, all else equal; the magnitude is expected to vary with the selected backend, expert reuse, and PCIe topology. Run the harness and report p50/p99.” The unsafe statement is “offload costs 12 ms” without a trace and setup.

## 11. Failure modes that a single benchmark hides

An MoE experiment can be numerically tidy and operationally wrong. The most common mistake is to report only the steady-state decode loop after manually loading a model, with one prompt, one batch, and no memory pressure. That answers a narrow kernel question. It does not answer whether a consumer endpoint can start, admit requests, survive long contexts, or preserve a tail-latency objective.

### Cold start is a separate phase

Record model download, safetensors index parsing, CPU-to-GPU transfer, quantization unpacking, kernel compilation, CUDA graph capture, and the first request separately. A warm process can reuse compiled kernels and allocator arenas. A new process cannot. A user who starts a local server once per experiment experiences cold-start time; a long-running service usually cares about warm steady state. Both are legitimate measurements, but they must not share one column called “latency.”

The phase ledger should contain at least:

| Phase | Start event | End event | Why it matters | Provenance |
|---|---|---|---|---|
| load | process start | weights available | deployment readiness | reproduce: process trace |
| prepare | weights available | first warmup complete | kernels and graphs | reproduce: CUDA events |
| prefill | request accepted | first token | prompt cost | reproduce: TTFT |
| decode | first token | final token | autoregressive cost | reproduce: TPOT |
| teardown | final token | allocator released | leak detection | reproduce: memory trace |

If expert offload is enabled, add host-page registration and pinned-buffer preparation. A route miss during the first request can include an allocation cost that does not appear after the expert cache is warm. That is useful information for an interactive application, not noise to delete.

### OOM is a state transition, not a failed number

Sweep prompt length until the configuration fails. Record the last passing length and first failing length. If the KV block size is $P$ tokens, a boundary near a multiple of $P$ can reveal the allocator’s granularity. For example, with $P=16$, moving from 128 to 129 tokens increases the block count from $\lceil128/16\rceil=8$ to $\lceil129/16\rceil=9$, adding one block. The 129-token transition is derived; the block bytes must come from the actual model configuration.

Also record whether the failure came from allocated memory, reserved memory, workspace, or an external process. PyTorch’s `memory_allocated()` and `memory_reserved()` answer different questions. The first is live tensor allocation. The second includes allocator-held segments that may be reused. A server that reports only the first can look healthy while reserve fragmentation prevents the next request.

```python
# nanoserve/memory_snapshot.py
import json
import torch

def memory_row(label: str) -> dict:
    if not torch.cuda.is_available():
        return {"label": label, "cuda": False}
    return {
        "label": label,
        "allocated_gib": torch.cuda.memory_allocated() / (1024 ** 3),
        "reserved_gib": torch.cuda.memory_reserved() / (1024 ** 3),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / (1024 ** 3),
    }

def write_row(path: str, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")

if __name__ == "__main__":
    print(memory_row("startup"))
```

Call the function at startup, after model load, after expert-cache warmup, after prompt prefill, and after the longest decode. This turns an OOM story into a sequence of state changes. It also gives the reader enough evidence to distinguish “the model did not fit” from “the model fit but the selected context did not.”

### Route skew and cache churn

An expert cache can fail even when its average hit rate looks good. Suppose a route trace has 95% hits but the 5% misses all occur in the longest requests. The mean TPOT can improve while p99 gets worse. Conversely, misses during a short warmup may have no user-visible impact. Join each miss with request ID, prompt class, active token batch, expert ID, and output position.

Define a per-request miss rate as

$$
r_i=\frac{m_i}{a_i},
$$

where $m_i$ is expert-cache misses and $a_i$ is expert assignments for request $i$. This statistic is useful for diagnosis, but it is not a latency model. Two misses of different byte sizes or at different layers are not interchangeable. Keep both the count and bytes loaded.

The route trace should also record top-k probabilities if the backend exposes them. A stable, highly confident router can have a cache-friendly population. A diffuse router can rotate through many experts even when top-k remains fixed. That is a workload property, not a universal property of the architecture.

### Quality must be checked at the same format

Quantization and offload should not change which weights are used, but format changes can alter logits. Run a small parity suite before a performance sweep. For deterministic greedy decoding, compare token IDs. For sampling, compare distributions or fix a seed and compare a tolerance-aware statistic rather than demanding identical sampled text.

Use at least four task shapes:

* a short factual prompt to expose obvious load or chat-template mistakes;
* a code completion with exact-match checks;
* a long-context retrieval prompt where the answer location is controlled;
* a structured-output prompt validated by a parser.

The scores are reader-produced. They should be stored beside the performance JSONL with model revision, tokenizer revision, format, and prompt hash. A model that wins TPOT but fails the structured-output parser is not a faster deployment for that task.

### Prompt accounting is part of fairness

Tokenizers can produce different counts for the same human prompt, and Qwen3’s chat template includes model-specific control tokens. Count input IDs after applying the official template, not characters before templating. For gpt-oss, the model card warns that the harmony response format is required. A benchmark that bypasses the required format is not a valid model comparison.

Report both user-visible text length and token length. If a prompt contains $S$ input tokens and the request generates $T$ output tokens, the total work is not summarized by $S+T$ alone: prefill has a different shape from decode, and the KV cache remains live for the full prefix during each output step. The same total token count can be cheap or expensive depending on how it is split.

## 12. A practical `nanoserve` experiment protocol

The implementation in this post is intentionally a measurement seam rather than a complete MoE runtime. A useful next commit would connect the route trace to the engine’s existing request object. Each request should carry model ID, prompt token count, expected maximum output, and a reservation handle. The scheduler should ask the memory ledger for capacity before admitting it.

```python
# nanoserve/admission.py
from dataclasses import dataclass
from math import ceil

@dataclass(frozen=True)
class Reservation:
    request_id: str
    kv_blocks: int
    bytes_reserved: int

def reserve(request_id, prompt_tokens, max_new_tokens, block_tokens,
            bytes_per_token, activation_bytes):
    total_tokens = prompt_tokens + max_new_tokens
    blocks = ceil(total_tokens / block_tokens)
    kv_bytes = blocks * block_tokens * bytes_per_token
    return Reservation(request_id, blocks, kv_bytes + activation_bytes)

if __name__ == "__main__":
    item = reserve("req-7", 129, 64, 16, 98304, 64 * 1024 * 1024)
    print(item)
```

Under the labeled planning assumptions used earlier—16-token blocks, 98,304 KV bytes per token, and 64 MiB activation reservation—129 input tokens plus 64 output tokens requires $\lceil193/16\rceil=13$ blocks. The KV reservation is $13\cdot16\cdot98{,}304=20{,}447{,}232$ bytes, approximately 19.5 MiB, before the 64 MiB activation term. This is a planning reservation, not a claim about the backend’s exact allocation.

The important detail is that the reservation uses the maximum output budget. If a scheduler reserves only the current prefix, it can admit a request that has no room for its own next token. If it reserves the maximum for every request, it may reject useful work. That is a policy decision. The experiment should expose both the conservative reservation and a measured preemption policy.

### The five-run progression

Use a progression that narrows uncertainty:

1. Load the candidate without serving and capture peak model memory.
2. Run a single prefill and decode with a tiny prompt to validate logits and cache writes.
3. Sweep context length at batch one until the first OOM or SLO failure.
4. Sweep active token batch with resident experts and log route histograms.
5. Repeat the same sweep with host spill, then compare the timeline rather than only the final tok/s.

At each stage, keep the prior raw files. Do not overwrite a passing short-context result with a later OOM result. Add a `status` field with `pass`, `oom`, `quality_fail`, or `slo_fail`. This makes the experiment useful to another engineer instead of turning it into a screenshot.

### Interpreting an apparent win

Suppose resident quantized weights produce lower TPOT but support only a short context, while host offload produces higher TPOT but supports a much longer context. Neither dominates. For a chat workload with short prompts, resident wins. For a RAG workload where context is the constraint, offload may be the only feasible point. For a batch service, a third configuration may win because larger token groups amortize routing and transfer.

This is why the final chart should show a frontier and workload labels. A single scalar “speed” deletes the decision. The reader needs to know whether a point is memory-limited, transfer-limited, compute-limited, quality-limited, or queue-limited.

## Case studies / real numbers

### The Qwen3 card separates total from activated parameters

The [Qwen3-30B-A3B model card](https://huggingface.co/Qwen/Qwen3-30B-A3B) reports 30.5B total and 3.3B activated, 48 layers, 128 experts, and eight activated experts. Those are cited model facts. The derived routing fraction is 6.25% of the expert set. The lesson is not that the model is cheap; it is that storage and arithmetic have different denominators.

### The gpt-oss card ships a memory-oriented format

The [OpenAI gpt-oss-20b card](https://huggingface.co/openai/gpt-oss-20b) reports 21B total and 3.6B active parameters, and says the released MXFP4 MoE weights allow the model to run within 16 GB of memory. That statement is tied to the released format and evaluation setup. It is a useful starting point for a consumer experiment, not permission to generalize to BF16 weights or another kernel.

### vLLM support is the benchmark target, not our measurement

The [vLLM gpt-oss support announcement](https://vllm-project.github.io/2025/08/05/gpt-oss.html), dated 2025-08-05, describes 32 experts and top-4 routing for gpt-oss-20b, MXFP4 MoE weights, and support on Blackwell, Hopper, and AMD MI300x/MI355x. The page gives us vocabulary and a production contrast. It does not give a Qwen3-on-RTX-4090 result for this post. For a production endpoint, use vLLM, SGLang, or another maintained engine and compare your `nanoserve` experiment against it under the same prompt suite.

### A hardware citation is a capacity constraint

NVIDIA’s [RTX 4090 product page](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/) identifies 24 GB of G6X memory. That is not a throughput result. It is the physical ceiling used in the arithmetic above. The difference matters: a hardware specification constrains what can fit; it does not predict a model’s TPOT.

## When to reach for this experiment, and when not to

Run this experiment when you need to answer one of these questions:

* whether a released quantized MoE checkpoint can coexist with a useful KV cache on a 24 GB card;
* whether expert reuse is high enough for a device expert cache;
* whether host offload improves capacity without violating the TPOT SLO;
* whether the router creates a long tail at your actual prompt mix;
* whether a sparse model beats a dense control after all dispatch and memory costs are counted.

Do not build an expert offloader just because a model has a large total parameter count. If the released quantized model fits resident and your workload is batch one, host offload adds complexity without a capacity benefit. If you need multi-tenant production, failure recovery, tensor parallelism, quantized kernels, prefix caching, and mature observability, use vLLM or another production engine. `nanoserve` is the instrument panel for understanding the trade; it is not a replacement for the engine that has already paid the integration cost.

Do not optimize the router before measuring route imbalance. Do not optimize expert GEMMs before measuring assignment counts. Do not tune KV blocks before measuring the weight and workspace footprint. The order is: prove the model fits, trace the route, measure the timeline, then optimize the bottleneck.

## 13. How to read the result without fooling yourself

The final report should begin with a one-paragraph verdict that names the workload. “Qwen3 fits” is incomplete. A defensible verdict looks more like: “The tested quantized artifact fits on the named RTX 4090 with a measured peak of X GiB, supports Y-token prompts at batch one, and meets the stated TPOT target only for the resident configuration; host spill increases the p99 because expert misses are serialized.” The placeholders must be filled from the reader’s trace. The shape of the sentence is the important part: capacity, context, workload, and failure mode are all named.

### Separate feasibility from performance

Feasibility has a binary boundary: a configuration either loads and serves the chosen request or it does not. Performance is a distribution over inputs and system states. Do not average an OOM into a throughput number. Do not drop the first request because its expert cache was cold without publishing the warmup rule. Do not compare a resident run at 8,192 context with an offload run at 32,768 context and call the result an offload penalty; the experiments have different memory obligations.

Create a result record with explicit dimensions:

```json
{
  "model": "Qwen/Qwen3-30B-A3B",
  "model_revision": "fill-in-commit",
  "format": "fill-in-artifact",
  "gpu": "RTX 4090 24GB",
  "placement": "resident-or-host-spill",
  "batch_tokens": 8,
  "input_tokens": 2048,
  "output_tokens": 128,
  "status": "pass-or-oom-or-slo-fail",
  "peak_allocated_gib": null,
  "ttft_p50_ms": null,
  "tpot_p99_ms": null,
  "output_tok_s": null,
  "quality_score": null,
  "provenance": "reader-reproducible harness"
}
```

The example uses 8, 2,048, and 128 as experiment knobs, not results. The JSON deliberately contains null measurements. A report that leaves those fields null is incomplete, but it is more trustworthy than one that fills them with plausible-looking values from another GPU.

### Use the slope, not only the endpoint

The memory curve should be approximately affine in token count over a fixed architecture:

$$
M(S) \approx M_0 + S\cdot B_{\text{KV/token}} + M_{\text{fragmentation}}(S),
$$

Here $M_0$ denotes the loaded model plus fixed runtime state, $S$ denotes live token count, and the last term captures block rounding and allocator effects. This is a planning approximation. The measured curve can have jumps at block boundaries and workspace thresholds. Fit the slope only over a stated range and compare it with the derived KV formula.

The same discipline applies to throughput. If output tok/s rises from batch one to batch eight, that does not prove the model is better for interactive use. It may simply mean the expert GEMMs are finally large enough to use the GPU. Plot p99 TPOT beside aggregate output tok/s. A service can gain throughput while violating every user-facing latency target.

### Name the bottleneck by evidence

Call a run memory-bound only when the trace supports it: high device-memory traffic, low arithmetic utilization, and a TPOT response that improves when the resident representation becomes smaller. Call it transfer-bound only when copies occupy the critical path or the consumer waits on them. Call it routing-bound only when router or dispatch events dominate the step. Call it scheduler-bound only when queue time grows while device work remains available.

Avoid using “GPU utilization” as the diagnosis. A card can report high utilization while doing inefficient tiny kernels, and low utilization while waiting for a host copy or a synchronization. Use CUDA events and a timeline, then use utilization as supporting evidence.

### The production decision

After the sweep, choose one of three conclusions. First, resident quantized deployment dominates for the target workload: it fits, keeps a useful KV budget, and has a clean tail. Ship the measured configuration and keep the trace as a regression test. Second, offload is the only feasible point for the required context: accept that its p99 is part of the product contract, or change the context and concurrency policy. Third, neither point meets the SLO: stop optimizing the wrong layer. Use a smaller model, a different GPU, a production MoE backend, or a workload-specific router.

The third answer is often the best engineering answer. A consumer GPU is a constrained laboratory, not a moral test. The experiment has done its job when it tells us which constraint is real and what a larger system would need to remove.

### A note on comparing with a production engine

The production comparison should be a controlled contrast, not a contest with mismatched defaults. Start vLLM with the model’s documented serving command and record its version, quantization mode, maximum sequence length, GPU memory utilization setting, and scheduling flags. Then give `nanoserve` the same model revision, tokenizer, prompt order, output budget, and concurrency schedule. If the engines expose different token accounting, normalize from the returned token IDs or usage fields rather than from wall-clock request counts.

The point of the contrast is diagnostic. If vLLM fits a resident artifact and `nanoserve` does not, inspect weight packing, unquantized modules, workspace, and allocator reserve before blaming the model. If both fit but vLLM has a better TPOT tail, inspect expert grouping, fused dispatch, CUDA graph capture, and host synchronization. If `nanoserve` appears faster on a tiny batch, check whether it has skipped a safety feature such as a maximum sequence reservation, prefix cache accounting, or proper queue-time measurement. “Faster” is not a property of a command line; it is a property of a fully specified workload.

Keep this contrast in a separate result namespace. A `nanoserve` reference implementation is allowed to be slower. Its value is that each event is legible: the router decision, the expert cache miss, the block reservation, and the synchronization are code you can inspect. A production engine is the benchmark target because it has already fused and scheduled these operations. The experiment is successful when it explains the gap and identifies the next measurement, not when a toy engine wins a cherry-picked cell.

### What to store for a future regression

Store one compact manifest beside each raw benchmark file. Include a hash of the prompt file, the model and tokenizer revisions, the quantization artifact, the exact server command, environment variables, GPU name, driver, CUDA runtime, and git revision of `nanoserve`. Include the warmup count, measured-step count, random seed, clock policy, and whether the process was started with an empty expert cache. The manifest is part of the result.

Store event rows rather than only aggregate summaries. A useful row has request ID, sequence ID, layer, decode position, active token count, selected expert IDs, expert-cache hit or miss, bytes copied, router duration, dispatch duration, expert duration, merge duration, attention duration, allocated bytes, reserved bytes, and completion status. The row can be large; that is preferable to losing the evidence needed to explain one p99 outlier.

From those rows, recompute aggregates in a separate script. This protects the report from a bug in a live dashboard and lets a reviewer ask for a different percentile or workload slice without rerunning the GPU. It also permits honest exclusions: if a kernel compilation happened in step zero, label it as setup and show both cold and warm views.

Finally, keep a failure corpus. Include the shortest prompt that caused OOM, the prompt that created the largest route imbalance, the request that caused the largest expert-cache miss burst, and a quality example where the quantized path diverged. These cases become regression tests when the engine gains a fused MoE kernel or a new allocator. A benchmark suite that contains only average prompts will tell you that the happy path is still happy; it will not tell you whether the sharp edge moved.

## Key takeaways

1. Active parameters describe per-token expert arithmetic, not total model residency.
2. Qwen3-30B-A3B’s cited 30.5B total and 3.3B activated parameters imply a 6.25% expert-selection fraction, not a 3.3B-parameter checkpoint.
3. On a 24 GB RTX 4090, BF16 Qwen3-30B-A3B fails the first-order weight budget before KV cache exists: 30.5B times two bytes is about 56.8 GiB.
4. Ideal INT4 arithmetic is a capacity screen; metadata, unquantized layers, workspace, and kernels decide whether a real artifact fits.
5. KV memory still grows with tokens and attention layers; MoE sparsity does not make the transformer cache disappear.
6. Expert offload exchanges VRAM pressure for host-device traffic and tail latency.
7. At batch one, top-8 routing across 128 experts creates tiny expert groups; larger token batches can improve grouping but can harm interactive latency.
8. A route trace and memory ledger are more valuable than an early custom kernel.
9. Report a Pareto frontier across memory, TTFT, TPOT, throughput, p99, and quality.
10. Use vLLM as the production benchmark target; use `nanoserve` to understand why the target’s optimizations exist.

## Further reading

* [What inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is) — the series frame from weights to product.
* [The inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) — the later capstone comparison.
* [Paged KV cache: implementing blocks and a block table](/blog/machine-learning/inference-engineering/paged-kv-cache-implementing-blocks-and-a-block-table) — the cache structure used by the budget.
* [The scheduler as a policy problem](/blog/machine-learning/inference-engineering/the-scheduler-as-a-policy-problem) — admission and goodput consequences.
* [Grouped GEMM for MoE kernels](https://vllm.ai/blog/2025-05-13-moe-optimizations) — a vLLM contrast for expert execution and grouping.
* [Qwen3-30B-A3B model card](https://huggingface.co/Qwen/Qwen3-30B-A3B) and [gpt-oss-20b model card](https://huggingface.co/openai/gpt-oss-20b) — the source of truth for the model facts used here.

<figure class="blog-anim">
<svg viewBox="0 0 760 220" role="img" aria-label="A token enters a router and the active expert highlight moves across eight expert cells while resident cells remain visible" style="width:100%;height:auto;max-width:860px">
<style>
.moe-cell{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}.moe-label{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}.moe-hot{fill:var(--accent,#6366f1);opacity:.2}@keyframes moe-route{0%,12%{transform:translateX(0)}25%,37%{transform:translateX(78px)}50%,62%{transform:translateX(156px)}75%,87%{transform:translateX(234px)}100%{transform:translateX(312px)}}.moe-move{animation:moe-route 10s steps(1,end) infinite}@media (prefers-reduced-motion:reduce){.moe-move{animation:none}}
</style>
<rect class="moe-cell" x="24" y="78" width="116" height="64" rx="8"/><text class="moe-label" x="82" y="116">token</text><path d="M140 110 H210" stroke="var(--text-secondary,#6b7280)" stroke-width="2"/><polygon points="210,110 200,104 200,116" fill="var(--text-secondary,#6b7280)"/><rect class="moe-cell" x="210" y="78" width="116" height="64" rx="8"/><text class="moe-label" x="268" y="116">router</text><path d="M326 110 H386" stroke="var(--text-secondary,#6b7280)" stroke-width="2"/><polygon points="386,110 376,104 376,116" fill="var(--text-secondary,#6b7280)"/>
<g><rect class="moe-cell" x="386" y="42" width="62" height="52" rx="7"/><rect class="moe-cell" x="464" y="42" width="62" height="52" rx="7"/><rect class="moe-cell" x="542" y="42" width="62" height="52" rx="7"/><rect class="moe-cell" x="620" y="42" width="62" height="52" rx="7"/><rect class="moe-cell" x="386" y="126" width="62" height="52" rx="7"/><rect class="moe-cell" x="464" y="126" width="62" height="52" rx="7"/><rect class="moe-cell" x="542" y="126" width="62" height="52" rx="7"/><rect class="moe-cell" x="620" y="126" width="62" height="52" rx="7"/><rect class="moe-hot moe-move" x="386" y="42" width="62" height="52" rx="7"/></g><text class="moe-label" x="535" y="24">selected expert group</text><text class="moe-label" x="535" y="208">one decode step, then repeat</text>
</svg>
<figcaption>The highlight moves through selected expert groups while the full resident expert set stays allocated; motion carries the difference between execution and storage.</figcaption>
</figure>

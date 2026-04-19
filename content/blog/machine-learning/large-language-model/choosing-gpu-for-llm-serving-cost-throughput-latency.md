---
title: "Choosing a GPU for LLM Serving: A Deep Trade-off Analysis of Cost, Bandwidth, Throughput, and Latency"
publishDate: "2026-04-19"
category: "machine-learning"
subcategory: "Large Language Model"
tags:
  [
    "LLM",
    "GPU",
    "inference",
    "serving",
    "throughput",
    "latency",
    "memory-bandwidth",
    "TCO",
    "vLLM",
    "TensorRT",
    "NVIDIA",
    "H100",
    "H200",
    "B200",
    "MI300X",
  ]
date: "2026-04-19"
author: "Hiep Tran"
featured: true
aiGenerated: true
excerpt: "An exhaustive, trade-off-first guide to choosing GPUs for LLM inference. Every section derives the trade-off from first principles with numbers: bandwidth vs FLOPs vs VRAM, batch size vs latency vs throughput, quantization quality vs speed vs memory, single-GPU vs tensor parallel vs pipeline parallel, NVLink vs PCIe vs InfiniBand, self-host vs API, spot vs reserved, dense vs MoE, and more. Detailed worked examples across H100, H200, B200, L40S, A100, MI300X."
---

# Choosing a GPU for LLM Serving: A Deep Trade-off Analysis of Cost, Bandwidth, Throughput, and Latency

Every decision in LLM serving is a trade-off. There is no "best GPU", no "right batch size", and no "optimal quantization" — there is only *the best point on a multi-dimensional frontier for your specific workload*. When someone gives you a categorical answer ("use H100"), they are either abstracting over your constraints or haven't thought carefully.

This article treats GPU selection as an optimization problem in disguise. Every section names the axes of trade-off, quantifies them with first-principles arithmetic, shows you where the frontier bends, and explains why real teams land where they do. Nothing is mentioned without a reason why it matters.

The article covers eleven sections. The first three are foundation and should not be skipped, because everything after them builds on their framing:

1. The fundamental trade-off: prefill vs decode, and why they pull you in opposite directions.
2. Resource trade-offs: bandwidth vs FLOPs vs VRAM vs interconnect.
3. Workload trade-offs: the iron triangle of latency, throughput, concurrency.
4. Software trade-offs: quantization, batching, parallelism, caching, speculative decoding.
5. Economic trade-offs: cloud vs owned, spot vs reserved, self-host vs API.
6. Hardware deep-dives for each major GPU with "when it wins / when it loses".
7. A full decision framework.
8. A complete worked example.
9. Common pitfalls.
10. Quick-reference tables.
11. Conclusion and habits of teams who get this right.

## 1. The Fundamental Trade-off: Prefill vs Decode

Every LLM inference request does two completely different kinds of work. Understanding their trade-off is the foundation of everything else because it is the reason a "good inference GPU" is not the same as a "good training GPU" and not even the same across inference workloads.

### 1.1 What prefill actually does

When a prompt arrives, the model must process all prompt tokens through every layer of the network to produce the key-value (KV) states that feed generation. This runs as one big batched matmul. Formally, for a dense transformer with `N` parameters, a prompt of length `L` performs on the order of `2 × N × L` FLOPs (the `2` accounts for multiply+add).

Three properties of prefill matter:

First, **arithmetic intensity is high**. Each weight is loaded once from HBM and multiplies a vector of `L` tokens. So for each byte of weights streamed, the GPU executes `L` multiply-accumulates. With `L` in the thousands, this ratio is hundreds to thousands of FLOPs per byte — well inside the compute-bound regime. The bottleneck is the tensor cores, not memory.

Second, **FLOPs scale linearly with `L`**. A prompt of 8,000 tokens costs roughly 16× a prompt of 500 tokens. This is unavoidable for a standard transformer and it explains why very long contexts are expensive even before generation begins.

Third, **attention has a quadratic component** (`O(L²)` in the attention matmuls), but FlashAttention and its descendants keep this close to linear in *wall-clock* by fusing the softmax and avoiding materializing the attention matrix. The FLOPs are still quadratic; the bandwidth hit is near-linear.

### 1.2 What decode actually does

After prefill, the model emits one token at a time. Each new token requires a forward pass through all layers with `L=1` (a single query token) against the already-populated KV cache.

Three properties of decode matter and they are the inverse of prefill:

First, **arithmetic intensity is terrible**. Per decode step, the GPU streams the full model weights (tens to hundreds of GB) to multiply by a single-token query. Ratio of FLOPs per byte is roughly `1–2`. The GPU is bandwidth-bound, and its tensor cores spend most of their time idle.

Second, **wall-clock per token is governed by HBM bandwidth**. A rough upper bound: `decode_tokens_per_second ≈ HBM_GB_per_s / model_size_in_GB`. If the model is 70 GB and HBM bandwidth is 3.35 TB/s, you cannot exceed about 48 tokens per second per user, no matter how powerful the tensor cores. The entire enterprise of inference GPU design is about reducing this floor.

Third, **batching reuses weight reads**. The same weight loaded from HBM can be multiplied by 32 query vectors (32 concurrent users) in the same pass. This is why throughput per GPU scales nearly linearly with batch size in decode, up to a knee where other costs (attention over longer sequences, scheduling overhead, KV cache size) take over.

### 1.3 The asymmetry quantified

Take Llama-3-70B in FP8 on one H100 SXM (3.35 TB/s bandwidth, ~1980 TF FP8 peak compute). Approximate per-request costs at batch = 1:

```
Prefill (per prompt token): ~2 × 70e9 / 1.98e15 ≈ 70 µs per token at peak FLOPs
Decode   (per output token): ~70 GB / 3.35 TB/s ≈ 21 ms per step
```

This gives a ratio of roughly `21 ms / 70 µs = 300×`. A single output token costs about 300 times as many wall-clock microseconds as a single prompt token at batch=1. This single number drives almost every economic fact in LLM serving:

Commercial APIs price output tokens 3–10× higher than input tokens because that is roughly the actual cost ratio once batching amortizes some (but not all) of decode's overhead. Batching is non-negotiable precisely because you need to amortize the 21 ms step across many concurrent users; a single-user server is wasting most of its GPU. Two workloads with the same total tokens per request but different prompt/output splits cost wildly different amounts; a 4000+400 request (prefill-heavy) is 2–4× cheaper than a 400+4000 request (decode-heavy) in $/token terms.

### 1.4 The trade-off axis

The practical implication is this table of inversions:

| Axis | Prefill | Decode |
|---|---|---|
| Which GPU spec scales throughput | Peak TFLOPs (FP16/FP8/FP4) | HBM GB/s |
| What a bigger batch buys | Little (already compute-saturated) | A lot (amortizes weight loads) |
| Cost of long context | Linear in prompt length (FLOPs) | Linear in sequence length (KV cache) |
| Effect of FP8/FP4 | Large (directly doubles/quadruples FLOPs) | Medium (halves/quarters bytes per token) |
| Effect of larger HBM | Indirect (fits larger batch) | Direct (fits more KV, more concurrency) |
| Effect of better interconnect | Medium (TP over matmul) | Large (TP over decode all-reduce) |

A GPU that is great at prefill (abundant FLOPs, modest bandwidth — e.g. L40S) can be mediocre at decode. A GPU that is great at decode (abundant bandwidth — e.g. H200) can be modestly over-spec'd for prefill-dominated workloads. Real traffic mixes the two, so you are always choosing a compromise.

### 1.5 How the mix depends on the workload

Profile your actual request distribution before picking hardware. Here is a reasonable table of typical workloads and what they imply:

| Workload | Prompt tokens | Output tokens | Dominant phase | What to optimize |
|---|---|---|---|---|
| Voice assistant | 150–300 | 50–150 | Roughly balanced | End-to-end latency, speculative decoding |
| Customer service chat | 300–800 | 200–800 | Decode-leaning | HBM bandwidth |
| RAG / QA | 3,000–8,000 | 200–600 | Prefill-leaning | FLOPs, prefix caching, chunked prefill |
| Code completion | 4,000–12,000 | 20–200 | Prefill-dominated | FLOPs, prefix caching |
| Long-document summarization | 30,000–200,000 | 500–3,000 | Prefill-dominated, extreme | FP8 FLOPs, KV memory |
| Agent loops (multi-turn tool use) | 2,000–10,000 | 3,000–20,000 | Decode-dominated | Bandwidth, speculative decoding, prefix cache |
| Batch classification/extraction | 500–3,000 | 50–500 | Throughput-maximizing | Any GPU at high batch |
| Creative writing | 200–500 | 1,000–4,000 | Decode-dominated | Bandwidth |

Two workloads with the same total tokens per request can want completely different GPUs. "Average 4000 total tokens" covers both code completion on an L40S and agentic chat on an H200 — and those have 3× different dollar costs per token.

### 1.6 Decision takeaway

Before you go further: measure your real traffic. You need p50, p95, and p99 of prompt length, output length, and end-to-end latency. If you are guessing, you will over-buy in one dimension and under-buy in another. This single hour of measurement saves more money than any hardware optimization below.

## 2. The Resource Trade-off Space

When you pick a GPU, you are not picking a single number — you are picking a point in a four-dimensional budget. Every GPU vendor ships a particular ratio of these four axes, and your job is to pick whichever ratio best matches your workload's ratio.

### 2.1 The four axes

The four axes that matter for LLM inference:

The first is **HBM capacity** (GB). This determines whether the model plus KV cache plus activations plus framework overhead fits at all, and how much concurrency the GPU can support. Capacity is binary at the cliff: either you have enough or you do not. There is no graceful degradation.

The second is **HBM bandwidth** (TB/s). This determines decode speed. Single-user tokens per second is roughly `bandwidth / model_size`. Aggregate throughput is bounded by the same factor, scaled by batch effects.

The third is **peak tensor FLOPs** (TF at FP16/FP8/FP4). This determines prefill speed. TTFT for a long prompt is essentially `FLOPs_needed / peak_FLOPs / utilization`. Utilization for well-tuned prefill is typically 40–65%.

The fourth is **interconnect bandwidth** (NVLink GB/s, or PCIe, or xGMI). This determines how efficiently multiple GPUs cooperate on one model. At TP=2 the interconnect tax is small (5–10%); at TP=8 it can exceed 30%.

### 2.2 Vendor ratios

Different GPUs occupy different points in this four-dimensional space. The table below shows the ratio signature of the main 2026 inference-relevant GPUs:

| GPU | VRAM | HBM BW | FP8 TF | NVLink (per GPU) | Ratio signature |
|---|---|---|---|---|---|
| H100 SXM5 | 80 GB | 3.35 TB/s | 1980 | 900 GB/s | "Balanced Hopper" |
| H100 PCIe | 80 GB | 2.0 TB/s | 1513 | NVL pair only | "H100 but bandwidth-starved" |
| H200 SXM | 141 GB | 4.8 TB/s | 1980 | 900 GB/s | "Decode-optimized Hopper" |
| B200 SXM | 192 GB | 8.0 TB/s | 4500 (FP8) / 9000 (FP4) | 1800 GB/s | "All axes up, at a price" |
| L40S | 48 GB | 0.864 TB/s | 733 | no NVLink | "Cheap FLOPs, no bandwidth" |
| L4 | 24 GB | 0.3 TB/s | 242 | no NVLink | "Low density, low power" |
| A100 SXM 80 | 80 GB | 2.0 TB/s | (FP16 only: 312 TF) | 600 GB/s | "Legacy, cheap, mature" |
| MI300X | 192 GB | 5.3 TB/s | ~2600 | 896 GB/s (xGMI) | "Huge VRAM, software risk" |
| MI325X | 256 GB | 6.0 TB/s | ~2600 | 896 GB/s | "Even more VRAM" |

Read the ratio signature column as describing which workload the GPU naturally fits:

H100 SXM is balanced and has been the serving default for three years because its ratio is reasonable for almost any workload. It has enough VRAM for 7B–34B comfortably and for 70B with tensor parallelism, enough bandwidth for decode, enough FLOPs for prefill, and strong NVLink.

H200 is an inference-specific Hopper refresh. Same compute as H100 SXM but 1.76× VRAM and 1.43× bandwidth. This precisely targets the two bottlenecks that H100 hit in 70B-class serving: VRAM (couldn't fit with headroom) and bandwidth (decode-bound). Same software, same power, same interconnect. It is the inference sweet spot of 2025–2026.

B200 is the next-generation Blackwell. Everything goes up, most notably FP4 support doubling compute again over FP8. Price, power, and density cost all go up as well. Ecosystem maturity for FP4 lags the silicon in early 2026.

L40S is a "pragmatic inference" card. Cheaper, no NVLink, GDDR6 instead of HBM. The bandwidth gap is severe: 864 GB/s vs 3350 GB/s is a 3.9× penalty on decode. But for small models where bandwidth is already plenty and where density matters, L40S is cost-effective.

A100 is legacy but still economically relevant because rental prices have fallen to 40–60% of H100. No FP8 hardware means you cannot get the prefill boost from quantization; weight-only INT8 still helps decode.

MI300X has class-leading VRAM at 192 GB and competitive bandwidth at 5.3 TB/s. Its constraint is software maturity rather than silicon. If your stack is on ROCm and your kernels are supported, it is price-competitive. If it is not, you will lose weeks to infrastructure debugging.

### 2.3 Trade-off: bandwidth vs FLOPs, worked out

A very common question: "Should I add a second H100 for tensor parallelism, or upgrade one H100 to an H200?" Both cost roughly the same. The trade-off breaks down by which resource you are short on.

For a 70B FP8 model serving decode-heavy traffic:

Option A, 1× H100 SXM: 80 GB VRAM (tight for 70B with KV), 3.35 TB/s bandwidth, 1980 TF FP8. Decode ceiling single-user ≈ 48 tok/s.

Option B, 2× H100 SXM TP=2: 160 GB VRAM (comfortable), 6.70 TB/s effective bandwidth because each GPU loads half the weights, 3960 TF FP8. Decode ceiling single-user ≈ 96 tok/s minus 10–15% TP tax, so ~82 tok/s.

Option C, 1× H200 SXM: 141 GB VRAM (comfortable), 4.80 TB/s bandwidth, 1980 TF FP8. Decode ceiling single-user ≈ 68 tok/s.

Interpretation for decode-heavy traffic: Option B (2× H100) is the fastest per user by a meaningful margin (~82 vs 68 tok/s for H200), but costs roughly 2× the H100 hourly rate or ~45% more than H200's price premium. Option C (H200) is the best tokens-per-dollar because it removes the TP tax and the extra GPU, even though it is slightly slower per user.

For a prefill-heavy workload, the math flips. More GPUs scale FLOPs nearly linearly because prefill has high arithmetic intensity and TP tax at prefill is small. Option B wins outright on TTFT (2× FLOPs available), while the bandwidth advantage of H200 matters less.

The general rule: if you are **decode-bottlenecked and fit-constrained**, pay for bandwidth per GPU (H200 over H100) before you add GPUs. If you are **prefill-bottlenecked** and you already have enough VRAM, adding GPUs scales close to linearly and is often the right move.

### 2.4 Trade-off: VRAM has a cliff

VRAM is the one dimension where there is no gradient. You cannot trade bandwidth for VRAM. If the model plus KV cache plus activations exceeds VRAM, inference fails — OOM, not slow. The options are:

More GPUs via tensor or pipeline parallelism, which adds interconnect tax and doubles cost for the case where you are barely over. Aggressive quantization (INT4 AWQ, FP4), which reduces quality. Shorter max context or fewer concurrent users, which reduces product capability. KV cache offload to CPU memory or NVMe, which tanks latency (tens of ms per token cost becomes hundreds).

The cliff shape has an important economic implication: moving from 99% full VRAM to 101% full VRAM roughly doubles serving cost. Moving from 70% to 90% costs nothing. This is why a bigger-VRAM GPU at a modestly higher hourly rate is often dramatically cheaper per token: it keeps you on the cheap side of the cliff.

H200 at 141 GB versus H100 at 80 GB was transformational not because 1.76× more VRAM is "nice to have" but because it moved many 70B and 120B MoE workloads from TP=2 to TP=1. That single change collapsed their serving cost by 30–50% because it eliminated the extra GPU and the TP tax.

### 2.5 Trade-off: NVLink vs standalone bandwidth

NVLink 4 at 900 GB/s sounds enormous until you compute how many bytes a tensor-parallel decode step must send. Each layer requires an all-reduce (or reduce-scatter + all-gather) over the hidden-state activations to merge partial products. For a 70B model with TP=4:

Per token, per layer, activations are roughly `hidden × 2 bytes` (BF16 activations) times the batch size. For hidden=8192, batch=32, that is 512 KB per layer. With 80 layers, a token requires about 40 MB of all-reduce traffic.

At 900 GB/s, transferring 40 MB takes 44 µs. At TP=4, a ring all-reduce visits each GPU `(n-1)/n × 2` times, so the effective time is roughly `40 MB × (3/2) / 900 GB/s ≈ 67 µs` per step. Compare to the underlying decode step at ~20 ms: this is a 0.3% overhead in the ideal case.

The reason real TP=4 has 20–30% overhead is that the all-reduce cannot overlap with compute as perfectly as the theoretical analysis suggests, and that kernel launch overhead, small-batch inefficiencies, and attention load imbalance all compound. At TP=8 on NVLink 4, these overheads stack to 30–45% in practice.

The practical breakdown:

| TP degree | Expected overhead on NVLink 4 | Suitability for latency-sensitive serving |
|---|---|---|
| 2 | 5–10% | Excellent |
| 4 | 15–25% | Good |
| 8 | 25–40% | Use only if VRAM forces it |
| >8 | 35–55% | Avoid; use PP or multi-replica |

On PCIe Gen5 (63 GB/s), every number in the table roughly triples. This is why "TP over PCIe" is a bad idea for latency-sensitive decode: you lose half your throughput to the interconnect. If your GPUs are not NVLinked, treat them as independent replicas, not as cooperators on a single model.

### 2.6 Arithmetic to derive VRAM needs

The practical VRAM budget for serving a dense transformer:

```
total_vram ≥ weights
           + kv_per_token × avg_context_tokens × concurrent_requests
           + activations_and_overhead
           + headroom (10–20%)
```

Weights: `N_params × bytes_per_param`. For 70B FP8, that is 70 GB.

KV cache per token: `2 × n_layers × n_kv_heads × head_dim × bytes`. For Llama-3-70B (80 layers, 8 KV heads with GQA, head_dim=128) at FP16: `2 × 80 × 8 × 128 × 2 = 327,680 bytes ≈ 320 KB/token`. With FP8 KV quantization, halve to ~160 KB/token.

Activations and overhead: depends on batch size and framework. vLLM with paged attention typically adds 2–8 GB for its metadata and runtime buffers.

Worked example: Llama-3-70B FP8 weights, FP8 KV, batch = 32, average context 4k tokens.

- Weights: 70 GB
- KV: `160 KB × 4000 × 32 = 20.5 GB`
- Activations/overhead: 6 GB
- Subtotal: 96.5 GB
- 15% headroom: ~111 GB

This fits one H200 (141 GB) with room to spare, but needs TP=2 on H100 (2×80=160 GB). If you push context to 16k average, KV becomes `160 KB × 16000 × 32 = 82 GB`, subtotal jumps to 158 GB, and now H200 barely fits (tight), while 2×H100 still has room (160 GB total and 15% headroom would need 182 GB — you would need TP=3 which is awkward, or 2×H200). This is how small-looking changes (4k to 16k average context) force substantial hardware changes.

## 3. The Iron Triangle: Latency, Throughput, Concurrency

The most important trade-off in serving, because it determines both capacity planning and SLO compliance, is the three-way tension between per-user latency, aggregate throughput, and concurrency.

### 3.1 Defining the three axes

Per-user latency has two components: TTFT (time to first token, dominated by prefill) and ITL (inter-token latency, dominated by decode step time). Users care about both. TTFT is "how long until something appears"; ITL is "how fast does it type".

Aggregate throughput is the total tokens per second the GPU serves across all users. This is what determines your cost per token because GPU hours are divided by tokens produced.

Concurrency is the number of requests in flight simultaneously. With continuous batching, this is the batch size at each decode step.

### 3.2 The trade-off, stated formally

Holding GPU and model fixed:

Increasing concurrency increases the batch size at each decode step. Because decode is bandwidth-bound, a bigger batch amortizes the same weight load across more users, raising aggregate throughput. But each user now shares the decode step with more peers, so per-user ITL rises because attention over longer KV caches grows and because the attention compute per step grows.

Decreasing concurrency reduces batch size, lowering per-user latency but also throughput. Taken to the extreme, batch = 1 gives minimum latency and maximum cost per token.

Given a latency SLO, the maximum sustainable throughput is determined by the largest batch size that still meets the SLO. Everything follows from that.

### 3.3 Quantified example

For Llama-3-70B FP8 on 1× H200 with reasonable tuning, illustrative numbers:

| Batch | ITL (ms) | Per-user tok/s | Aggregate tok/s | GPU utilization |
|---|---|---|---|---|
| 1 | ~15 | 66 | 66 | ~12% |
| 4 | ~17 | 59 | 235 | ~25% |
| 8 | ~19 | 53 | 421 | ~40% |
| 16 | ~22 | 45 | 725 | ~55% |
| 32 | ~30 | 33 | 1,066 | ~75% |
| 64 | ~45 | 22 | 1,410 | ~88% |
| 128 | ~72 | 14 | 1,775 | ~94% |
| 256 | ~130 | 8 | 1,970 | ~97% |

Two properties of this table deserve attention:

First, aggregate throughput saturates asymmetrically. From batch 1 to batch 32 you gain 16× throughput for 2× latency — a great trade. From batch 32 to 256 you gain 1.85× throughput for 4.3× latency — a bad trade. There is a knee around batch 32–64 for most configurations, and pushing past it is usually a mistake for interactive workloads.

Second, ITL grows sublinearly in the middle and accelerates at the ends. The sublinear region reflects that the dominant cost (weight loads) is amortized well. The acceleration at high batch reflects attention cost (quadratic in sequence length, summed over all sequences in the batch) and scheduling overheads that no longer hide behind compute.

### 3.4 SLO determines cost per token

The same GPU, same model, same software, produces dramatically different $/token numbers based entirely on the SLO you commit to:

| SLO (ITL p95) | Sustainable batch | Aggregate tok/s | $/Mtok at $5/hr |
|---|---|---|---|
| 20 ms | ~8 | ~420 | ~$3.31 |
| 30 ms | ~32 | ~1,066 | ~$1.30 |
| 50 ms | ~80 | ~1,600 | ~$0.87 |
| 100 ms | ~200 | ~1,900 | ~$0.73 |
| Batch (no SLO) | ~256 | ~1,970 | ~$0.70 |

Between a tight interactive SLO and a relaxed batch workload, the cost per token varies by roughly 4.7×. This is larger than any hardware-choice difference, larger than the BF16-to-FP8 delta, and larger than any reasonable reserved-price discount. The SLO is the single most important economic variable under your control.

The practical implication: negotiate SLOs with your product team explicitly. Many teams default to "as fast as possible" when the real user bar is much looser, and they pay 3–4× more than necessary. Conversely, some teams relax SLOs for cost reasons without realizing that user engagement drops below certain ITL thresholds (the "feels slow" line is typically around 40–60 ms for chat).

### 3.5 The fourth hidden axis: request shape variance

Production workloads have non-trivial variance in prompt and output lengths. A batch of 32 users where one has a 32k prompt and 31 have 200-token prompts behaves dramatically differently from 32 uniform requests. Three distinct problems arise:

Prefill bubbles: a long prefill monopolizes the GPU for hundreds of milliseconds. If prefill and decode share the GPU, all decoders stall. Solution: chunked prefill, which splits long prefills into 512–2048 token chunks and interleaves them with ongoing decode steps. This bounds the stall at "one chunk worth" of time (tens of ms) rather than the full prefill duration.

Padding waste: in classical static batching, all sequences in a batch are padded to the longest. Short sequences waste compute and memory. Solution: continuous batching (vLLM, SGLang, TensorRT-LLM), which treats each sequence independently. Sequences enter, decode for their natural length, and exit; new sequences fill the freed slots.

Head-of-line blocking: a single slow request (long output, stuck in attention) stalls the batch behind it. Solution: either prioritize scheduling for new requests, or separate prefill and decode onto different GPU pools ("disaggregated serving"). Disaggregated pools scale independently, decode pool runs at very high batch because it is homogeneous, and prefill pool runs at peak FLOPs without decode contention.

The trade-off for disaggregated serving: higher throughput (1.3–2× commonly reported) at the cost of system complexity and KV transfer bandwidth. Moving the KV cache from prefill GPU to decode GPU requires NVLink-C2C or fast InfiniBand; typical transfer sizes are hundreds of MB per request for long contexts. For low-to-medium QPS (under ~100 QPS per model), disaggregated serving is overkill. At scale (1000+ QPS), it is usually the right choice.

### 3.6 Why "tokens per second" alone is misleading

Benchmarks that report "2000 tokens/sec on H100" are incomplete without specifying batch and ITL. The same hardware can produce:

At batch 1, 30 tok/s single-user with 30 ms ITL — latency-optimized.
At batch 32, 1000 tok/s aggregate with 30 ms ITL — sweet spot.
At batch 256, 2000 tok/s aggregate with 130 ms ITL — throughput-optimized.

All three are "on H100" and all three are correct; they serve different product requirements. When you read a throughput number, always demand the accompanying latency number. Conversely, when quoting your own, always report both.

## 4. Software Trade-offs

Software choices multiply or divide the hardware numbers by factors of 2–5. These are among the highest-leverage decisions you make, and most of them are reversible.

### 4.1 Quantization: quality, speed, memory

Quantization is the single biggest lever. The trade-offs stack in a specific way that matters:

| Quant | Weight bytes | Prefill speedup (H100+) | Decode speedup | Quality impact (typical) |
|---|---|---|---|---|
| BF16 | 2 | 1.0× | 1.0× | baseline |
| FP8 (E4M3 per-tensor) | 1 | ~1.9× | ~1.85× | near-lossless |
| FP8 (per-channel / per-token) | 1 | ~1.9× | ~1.85× | often indistinguishable |
| INT8 weight-only | 1 | no speedup | ~1.85× | near-lossless |
| INT4 (AWQ / GPTQ) | 0.5 | no speedup on H100; ~1.5× on Blackwell | ~1.75× (bw-limited) | 0.5–2 pt drop on MMLU-like |
| FP4 (Blackwell native) | 0.5 | ~3.7× | ~3.1× | 0–1 pt drop with good calibration |
| INT3 / INT2 | <0.5 | N/A | N/A | significant drops, research-grade |

Three important trade-off insights usually missed:

FP8 is close to free *only if* both weights and activations are FP8 and the kernel uses the FP8 tensor cores. Running FP8 weights with BF16 activations ("W8A16") does not use the FP8 compute path, so you save memory and bandwidth but not FLOPs. For prefill-heavy workloads, this gives up half the potential speedup. Always aim for W8A8 when FP8 is supported.

INT4 weight-only compresses weights to 4 bits but dequantizes them to BF16/FP16 inside the kernel before the matmul. Compute is unchanged. This helps decode (bandwidth halved) and VRAM (halved), but prefill is unchanged because it is compute-bound and you are not using lower-precision tensor cores. For prefill-heavy workloads, INT4 is an underwhelming optimization compared to FP8. For decode-heavy or VRAM-tight workloads, INT4 is excellent.

FP4 on Blackwell compounds both benefits: halves bandwidth and quadruples compute vs BF16 (doubles vs FP8). Quality risk is real but manageable with good calibration. As of early 2026, production-grade FP4 kernels exist for mainstream architectures (Llama, Mistral, Qwen) and calibration tooling is maturing. Edge cases (unusual architectures, non-standard attention) are where FP4 support lags.

KV cache quantization to FP8 is often the highest-leverage single change. Halves KV VRAM, halves KV bandwidth during decode, quality impact is typically negligible (per-head scaling helps). Long-context workloads benefit enormously. If your framework supports FP8 KV, turn it on by default.

The defensible default in 2026: FP8 weights + FP8 activations + FP8 KV on H100+ hardware; INT8 weight-only + BF16 activations + FP8 KV (or INT8 KV) on A100; FP4 on Blackwell *after* you have validated quality on your evaluations.

### 4.2 Batching strategy: static vs continuous vs chunked prefill vs disaggregated

| Strategy | Throughput | Latency behavior | Complexity | Best for |
|---|---|---|---|---|
| Static batching | Low (padding + HOL blocking) | Unpredictable | Low | Don't use for LLMs |
| Continuous batching | High | Predictable | Medium (vLLM, SGLang, TensorRT-LLM all do this) | Default |
| Continuous + chunked prefill | Highest at reasonable scale | Best tail latency | Slightly higher | Default for mixed workloads |
| Disaggregated prefill/decode | Highest at large scale | Best throughput + latency | High | 1000+ QPS, mature teams |

The incremental cost of moving from static to continuous batching is negligible — every production serving framework supports continuous batching, and the default is correct. The surprising thing is that some teams still run static batching because they wrote their serving infrastructure before continuous batching became common. If you are, swap; it is free 2–3× throughput.

Chunked prefill is worth it whenever your prompt length distribution has a long tail. Without chunking, a single 32k prompt stalls all active decodes for 2+ seconds, spiking tail latency. With chunking, the stall is bounded by chunk size (typically 2048 tokens, ~130 ms).

Disaggregated serving separates prefill and decode onto distinct GPU pools. Each pool scales and tunes independently. The decode pool can run at very high batch size (homogeneous workload) while the prefill pool runs at peak FLOPs (also homogeneous). Trade-off: you now have two autoscalers, a KV-transfer mechanism between pools (needs NVLink-C2C or CX-7 InfiniBand-class networking), and operational complexity. Worth it past ~1000 QPS of a single model; overkill below 100 QPS.

### 4.3 Parallelism strategies: TP, PP, EP, DP

Each strategy is a different answer to "how do we split the model across GPUs?" with different trade-offs.

Tensor parallelism (TP): split each layer's weight matrices across GPUs. Every layer performs a small all-reduce per token. Minimizes latency (all GPUs work on every step), pays interconnect cost per step. Best for: one model that does not fit on one GPU, where latency matters. Use on NVLinked GPUs only.

Pipeline parallelism (PP): split layers across GPUs. Each GPU owns a contiguous slice of layers. No per-layer all-reduce — only forward the activations from one slice to the next. Interconnect cost is much lower than TP. But PP introduces pipeline bubbles: while GPU 4 works on token T, GPUs 1–3 are either idle or processing later tokens in a micro-batch. Pipeline bubble cost grows with depth and shrinks with micro-batch count. Best for: batch workloads where throughput matters and latency is flexible, or cross-node parallelism where TP would be too bandwidth-demanding.

Expert parallelism (EP): for Mixture-of-Experts (MoE) models. Different experts live on different GPUs; each token is routed to a subset of experts. Scales beautifully when routing is balanced, pays for rebalancing when it is not. Essential for Mixtral, DBRX, DeepSeek, GPT-OSS, and other MoE architectures at scale.

Data parallelism (DP): replicate the full model on each GPU and split requests across replicas. Zero interconnect between replicas. Best for: scaling throughput when you have QPS headroom and models fit on single GPUs.

The practical layering: pick the minimum TP degree that fits your model with headroom, replicate for throughput via DP, add EP for MoE, avoid PP for interactive serving. Cross-node TP via InfiniBand is almost never worth it for decode; if your model does not fit in one NVLink island, either buy bigger-VRAM GPUs or use PP across nodes with large micro-batches.

### 4.4 Prefix caching

Prefix caching stores the KV state for the shared prefix of requests (system prompt, few-shot examples, retrieved context) and reuses it across subsequent requests. The trade-offs:

Benefit: for cached prefixes, prefill is skipped entirely. TTFT drops from hundreds of ms or seconds down to the prefill time for only the non-shared suffix, often 5–50× faster depending on prefix length and total prompt length.

Cost: cached KV consumes VRAM that could otherwise serve active requests. Typical overhead is 1–5 GB per few thousand tokens of cached prefix, depending on model size and KV quantization. You are trading raw concurrency for TTFT.

Hit rate sensitivity: the economic win is directly proportional to hit rate. A 10% absolute increase in hit rate can swing cost per token by 30–40% for prefill-heavy workloads. Measure hit rate in production and tune cache eviction policy.

Invalidation: when does a cache entry expire? Options range from LRU by time-since-last-use, to pinned entries (system prompts never evict), to per-customer isolation. Most production systems pin frequently-used system prompts and LRU the rest.

For any workload with shared prefixes — chat apps with a system prompt, RAG where the same template wraps every request, few-shot classification — prefix caching is almost always worth enabling. vLLM, SGLang, and TensorRT-LLM all support it out of the box.

### 4.5 Speculative decoding

A smaller "draft" model proposes several tokens (typically 4–8); the main model verifies them in a single forward pass. Because verification costs only slightly more than producing one token, if many drafts are accepted, effective tokens per second rise substantially.

Benefit: up to ~3× per-user decode tokens per second on high-acceptance workloads (code, structured JSON, common English prose).

Cost: the draft model consumes VRAM (typically 0.5–2 GB for a 1B draft). It also runs inference per step, though the per-step cost is much lower than the main model. When acceptance is low, speculative decoding can be slower than plain decoding because of verification overhead.

Acceptance rate is everything. A 70% acceptance rate gives roughly 2× speedup; a 30% acceptance rate gives ~1.2× or sometimes slowdown. Acceptance depends on draft quality, task similarity between draft and main model, and entropy of the output (structured output has low entropy and high acceptance; creative writing has high entropy and lower acceptance).

Speculative decoding compounds poorly with high batch. At batch = 1, the saved bandwidth is a fraction of the GPU's capacity. At batch = 64, the GPU is already saturated on weight loads; speculative decoding adds verification overhead without freeing bandwidth. Rule of thumb: speculative decoding is worth the complexity at batch ≤ 8, optional at batch 16–32, and usually counterproductive at batch ≥ 64.

Best use case: single-user latency-critical workloads like voice assistants, code autocomplete in IDEs, and agentic loops where tail latency matters.

### 4.6 Serving framework choice

The three dominant open-source stacks as of 2026:

vLLM: most widely adopted, strong continuous batching, excellent PagedAttention, good multi-GPU support, active community. Somewhat heavyweight for small deployments. Default safe choice.

SGLang: smaller, faster in many benchmarks, particularly strong on structured output and multi-turn scenarios. Younger ecosystem, fewer integrations.

TensorRT-LLM: NVIDIA's own serving stack. Best peak performance on NVIDIA hardware because it uses TensorRT's kernel autotuning. Steeper learning curve, NVIDIA-only, compile step required per model version.

Rough performance: on the same H100 with the same model, well-tuned TensorRT-LLM is typically 10–25% faster than vLLM, which is roughly on par with SGLang. For NVIDIA-only production deployments where peak $/token matters and you have engineering bandwidth for compilation and tuning, TensorRT-LLM wins. For heterogeneous hardware (including AMD) or teams valuing operational simplicity, vLLM is the right default.

## 5. Economic Trade-offs

### 5.1 The $/token formula and its levers

The single number that summarizes serving economics:

```
$/token = (Σ GPU $/hr across all GPUs) / (aggregate tok/s × 3600)
```

This is deceptively simple because every input has multiple sub-levers:

| Lever | Direction | Typical $/token impact |
|---|---|---|
| Quantize BF16 → FP8 (weights + activations + KV) | Improves tok/s | –35% to –45% |
| Add KV-only FP8 quant (BF16 compute remains) | Improves tok/s | –10% to –25% |
| Switch serving stack from vanilla HF to vLLM/SGLang/TensorRT-LLM | Improves tok/s | –30% to –60% |
| Raise SLO from 25ms to 50ms ITL | Raises sustainable batch | –30% to –50% |
| Swap H100 → H200 (same SLO) | More tok/s per GPU | –20% to –35% |
| Enable prefix caching (RAG) | Skips prefill for cached prefixes | –15% to –40% |
| Speculative decoding at batch ≤ 8 | Raises per-user tok/s | –20% to –50% |
| Spot pricing vs on-demand | Reduces hourly rate | –40% to –60% (plus preemption risk) |
| 1-year reserved vs on-demand | Reduces hourly rate | –30% to –40% |
| 3-year reserved vs on-demand | Reduces hourly rate | –50% to –60% |
| Disaggregated prefill/decode | Higher throughput at scale | –20% to –40% |

These compound multiplicatively. A team going from "BF16 on a naive HuggingFace server on on-demand H100" to "FP8 weights + activations + KV on well-tuned SGLang on reserved H200" can cut $/token by 5–8×. This is why new teams often look at their first cost report and think serving is impossibly expensive — it is, until you apply the obvious optimizations, after which it becomes reasonable.

### 5.2 Cloud vs owned hardware

The table of economic trade-offs:

| Dimension | Cloud on-demand | Cloud reserved (1–3 yr) | Owned (colo/on-prem) |
|---|---|---|---|
| Capex | $0 | $0 | $200K–$5M for a node |
| Unit cost ($/GPU/hr equivalent) | $3–$9 | $2–$5 | $1.00–$1.80 all-in |
| Provisioning time | minutes | days | months |
| Capacity risk | low | medium | none once deployed |
| Breakeven utilization | any | >60% | >75% |
| Ops burden | none | none | substantial (datacenter, power, cooling, hands-on support) |
| Hardware obsolescence risk | vendor absorbs | partially | you absorb |

Worked break-even for an 8× H100 SXM node used steadily for 3 years:

Cloud on-demand at $4/GPU/hr × 8 × 24 × 365 × 3 = ~$841K over three years.
Cloud reserved 3-year at ~$2.50/GPU/hr × 8 × 24 × 365 × 3 = ~$526K.
Owned: ~$250K capex for the node, plus ~$40K/yr for power (roughly 6–8 kW × $0.10–0.12/kWh × 8760 hr), plus ~$15K/yr colo space and network, plus ~$15K/yr ops (partial FTE, spares). Three-year total: ~$250K + $210K = ~$460K.

Owned wins by ~13% over 3-year reserved and by ~45% over on-demand, *if utilization is high and demand is stable for three years*. The break-even is sensitive to utilization: if you average 50% utilization instead of 100%, the owned cost per used hour doubles but your cloud cost scales down with actual usage. Cloud wins at low utilization; owned wins at high utilization.

Most teams serving production traffic land on a hybrid: base load on owned hardware or 3-year reserved cloud, burst capacity on on-demand or spot cloud, and offline batch workloads on spot. This matches capacity to demand profile rather than over-provisioning one tier.

### 5.3 Self-host vs commercial API

The right question is not "which is cheaper per token in isolation" but "which is cheaper after all costs, at my actual demand profile". Dimensions to compare:

| Dimension | Self-host | Commercial API |
|---|---|---|
| Per-token cost (70B open model) | $0.20–$1.00 / Mtok at high utilization | $0.20–$2.00 / Mtok |
| Per-token cost (frontier closed model) | N/A | $3–$30 / Mtok |
| Latency control | Full (you choose hardware and serving) | None (depends on provider) |
| Data residency | Full control | Contractual, varies |
| Custom fine-tunes | Unlimited | Limited to provider's offering |
| Model variety | Open models only | Frontier closed + open |
| Ops burden | Real (oncall, autoscaling, model updates, monitoring) | Zero |
| Time to ship | Weeks | Hours |
| Peak handling | You must provision | Provider absorbs |

The crossover heuristic for open models: self-host wins above roughly 100M tokens per day sustained on a 70B-class model, assuming you can actually achieve 60%+ utilization on your provisioned capacity. Below that, commercial APIs almost always win on total cost including engineering.

The trap is peak-to-average ratio. If your traffic has a 10× peak-to-average ratio and you provision for peak, you pay for 10× the capacity you actually use. Autoscaling helps but adds complexity and cold-start latency (LLM serving typically needs 30–120 seconds to load a fresh replica). APIs absorb this cost implicitly because they pool demand across many customers.

The second trap is operational cost. A self-hosted deployment requires oncall rotations, version upgrades (new model releases), security patches, monitoring, capacity planning, and incident response. Count on 0.3–0.7 FTE per model per year, which at realistic loaded engineering costs is $50K–$150K annually. If your self-hosted savings don't exceed that, use the API.

Frontier closed models (GPT-5, Claude Sonnet 5, Gemini 3) are a different category. You cannot self-host them at any price; the only questions are which provider, what commitments, and whether to route some traffic to cheaper open models.

### 5.4 Spot, reserved, on-demand

Three pricing tiers trade discount for risk and flexibility:

On-demand: no commitment, no discount. Full price, full flexibility. Best for unpredictable workloads, proof-of-concept, or peak burst.

Reserved 1-year: 30–40% discount, one-year commitment. Best for steady-state production on mature hardware.

Reserved 3-year: 50–60% discount, three-year commitment. Best for very steady production on hardware generation where you expect to remain (H100/H200 Hopper family looks stable for at least 2 more years of usefulness in 2026). Risk: if Blackwell or its successor delivers 3–5× better $/token mid-reservation, you are locked in to yesterday's economics.

Spot / preemptible: 50–70% discount, preemptible at any time with minutes of notice. Best for stateless batch workloads that can checkpoint and resume. For interactive serving, spot alone is impractical because preemptions cause user-visible outages; but spot for the auto-scaled tier above a reserved baseline is a proven pattern.

Committed use discounts (GCP) and savings plans (AWS): 20–50% discount for a dollar commitment rather than specific instance type. More flexible than reserved instances, useful when you might change hardware mid-commitment.

The practical approach: reserve baseline capacity for predictable load (matches steady-state users), use on-demand for normal peaks, and use spot for batch/offline workloads or as a cheap second tier for burst capacity with fallback to on-demand if spot is unavailable.

### 5.5 Hardware generation timing

Buying or reserving GPUs during a hardware transition is a significant trade-off. In 2026, the transition is Hopper (H100/H200) to Blackwell (B200/B100/GB200).

Reserving H200 3-year in early 2026 locks you in to current prices while Blackwell inventory and software ecosystems mature. If Blackwell cost per token drops 2–3× by 2027 (plausible, especially once FP4 is widely validated), you will be paying double the market rate for years.

Reserving B200 3-year in early 2026 is locking in premium prices when supply is tight. If you can absorb the early-adopter software tax and have validated FP4 quality, this can be the best long-run $/token decision.

The middle path many teams take: reserve current-gen (H200) for 1 year while exploring next-gen (B200) on on-demand and spot. Commit more heavily to next-gen as software matures and your workload is validated. This gives up some discount for flexibility, which is usually worth it during transitions.

## 6. Hardware Deep-dives with Trade-off Framing

Each GPU gets a consistent treatment: spec, strengths, weaknesses, and "when it wins / when it loses" — that is the part that most buying decisions actually need.

### 6.1 H100, the 2022–2025 workhorse

| Variant | VRAM | BW | FP8 TF | NVLink | Notes |
|---|---|---|---|---|---|
| H100 SXM5 | 80 GB | 3.35 TB/s | 1980 | 900 GB/s | Standard HGX 8× node |
| H100 PCIe | 80 GB | 2.0 TB/s | 1513 | NVL pair only | Air-cooled, dense |
| H100 NVL | 94 GB | 3.9 TB/s | 1513 | 600 GB/s via NVL | 2-GPU pair for 70B |

H100 SXM is the workhorse. It is mature, well-supported by every serving framework, and has enough of every resource for 7B–34B models comfortably and for 70B models with TP=2. It has been the default for three years because of ecosystem and availability rather than because it was ever the cost-optimal choice for any specific workload.

The PCIe variant is a common trap. It has 40% less bandwidth than SXM and no NVLink mesh. For decode-heavy workloads, PCIe H100 performs roughly like A100 SXM despite newer silicon. Choose PCIe only when density or thermals force you into a PCIe server form factor; never choose it for the compute.

The NVL variant is a specialty product: two H100 cards bridged with NVLink, sold as a pair for fitting 70B models specifically. It has more VRAM (94 GB per card) than SXM and is cheaper than two SXM cards. If your exact workload is 70B inference on limited infrastructure, NVL can be the price/performance winner, but the niche is narrow.

When H100 wins: mainstream 7B–34B workloads where 80 GB and 3.35 TB/s are plenty, and where its ecosystem maturity and widespread availability are worth more than H200's performance edge. Also wins on short supply timelines — H100 is typically available immediately from any major cloud.

When H100 loses: any 70B+ workload where H200's additional VRAM and bandwidth eliminate the need for TP=2. Also loses on new long-context features, where the VRAM gap to H200 hurts KV capacity.

### 6.2 H200, the inference-optimized Hopper

Same die as H100; refreshed HBM package. 141 GB HBM3e at 4.8 TB/s. Same compute, same NVLink, same power envelope, same software.

H200 is the inference sweet spot of 2025–2026. It precisely targets the two bottlenecks that H100 hit in 70B-class serving: VRAM (too tight to fit 70B plus KV headroom plus concurrency) and bandwidth (decode-bound). Adding 1.76× VRAM and 1.43× bandwidth without changing anything else fixes both problems with no software work.

When H200 wins: this is the best mainstream choice for 70B-class and 120B MoE serving. It moves most 70B workloads from TP=2 on H100 to TP=1 on H200, eliminating an entire GPU and the TP tax. The hourly price premium (typically 30–50% over H100) is more than offset by halving the number of GPUs and gaining 1.43× bandwidth each.

When H100 still wins: smaller models (7B–34B) that fit on H100 with bandwidth to spare. There, H200's extra resources are idle and the hourly premium is pure loss. Also wins when H200 availability is tight in your region; a readily-available H100 is worth more than a waitlisted H200.

When H200 loses to B200: frontier-scale workloads where even H200's capacity is tight, or where FP4 validation has already been done and its 2× compute boost is achievable. For now B200 has a supply and software premium; H200 is the risk-averse choice.

### 6.3 B200, Blackwell's step change

192 GB HBM3e at 8 TB/s. 4.5 PF FP8 and 9 PF FP4. NVLink 5 at 1.8 TB/s per GPU. GB200 NVL72 extends this to a 72-GPU NVLink domain — effectively one colossal GPU for trillion-parameter models.

B200's compound advantages are: 1.67× bandwidth over H200, 2.27× FP8 compute, 4× effective compute with FP4 (when quality is validated), 36% more VRAM, 2× NVLink bandwidth. Against H200, this is a step change in every dimension.

When B200 wins: frontier-scale serving (400B+ MoE), extreme-throughput deployments where small $/token differences justify the premium, and workloads where FP4 has been validated for the model. Also wins for models that don't fit on H200 single-GPU, where B200 collapses them to single-GPU and eliminates TP entirely.

When B200 loses (relative to H200): mainstream 70B workloads where H200's capacity is already sufficient. The hourly premium (typically 40–70% over H200) requires 50%+ throughput gain to justify, and FP8 alone doesn't always deliver that on models that are not bandwidth-bottlenecked even on H200. FP4 delivers it, but only if you've accepted the quality risk.

Early-2026 caveats for B200: supply is tight, datacenter requirements are demanding (liquid cooling, ~1000W per GPU on SXM), FP4 kernels are complete for mainstream architectures but edge cases still catch teams out, and the FP4 calibration toolchain is evolving. None of these are deal-breakers — the silicon is excellent — but they argue against 3-year reservations in early 2026.

### 6.4 L40S, the cost-density workhorse

48 GB GDDR6 at 864 GB/s, 733 TF FP8, PCIe card (no NVLink). Ada Lovelace architecture, consumer-derived silicon optimized for datacenter use.

L40S lives on a different axis than H-class GPUs. It has FP8 support (good for prefill), but its bandwidth is 3.9× lower than H100 SXM and 5.6× lower than H200. It has no NVLink, so multi-GPU serving depends on PCIe, which is too slow for tensor parallelism in decode. Effectively it is a single-GPU serving device.

When L40S wins: serving many small-to-medium models (7B–13B in FP8) where single-GPU fit is fine and where per-hour price matters more than per-GPU throughput. Particularly strong for high-density workloads in standard 2U servers where H-class thermals don't fit. Also wins when power budget is constrained: 350W vs 700W+ for H100 SXM.

When L40S loses: any 70B+ workload (needs multi-GPU, and PCIe TP is impractical for decode). Any latency-critical decode where its GDDR6 bandwidth underperforms HBM. Any high-concurrency serving where its 48 GB VRAM caps the batch size.

There is a particular niche where L40S is genuinely the best answer: internal inference for enterprise tools where models are small, QPS is modest, density matters (one rack = many models), and $/token is more important than absolute throughput.

### 6.5 A100, the still-cheap legacy

80 GB HBM2e at 2.0 TB/s, 312 TF FP16, no native FP8. Ampere architecture shipped in 2020.

A100 is the "still useful legacy" GPU. Rental prices have fallen to 40–60% of H100 on secondary clouds, making it economically competitive for workloads where peak performance is not needed. No FP8 hardware is the main limitation: weight-only INT8 still helps decode (halves bandwidth requirements), but prefill stays BF16 compute and cannot exploit the speedups that FP8-capable GPUs enjoy.

When A100 wins: budget-constrained deployments, internal tools and research where cost discipline matters more than latency, small to medium models (8B–13B) where the 2.0 TB/s bandwidth is adequate and the hourly price advantage compounds.

When A100 loses: any workload where FP8 speedup would meaningfully reduce hardware footprint. For example, serving 70B FP8 on H100 gives you roughly 2× the decode throughput of serving 70B BF16 on A100 at only ~1.3–1.6× hourly price — a clear win for H100. Also loses on long contexts, where its bandwidth cannot keep up with KV cache reads.

Practical advice: if you are new to LLM serving in 2026, skip A100 and go to H100 as the floor. If you already have A100 capacity reserved, use it for batch workloads, non-customer-facing internal tools, and overflow, but don't expand your A100 fleet.

### 6.6 MI300X and MI325X, AMD's credible alternative

MI300X: 192 GB HBM3 at 5.3 TB/s, ~2.6 PF FP8, 896 GB/s xGMI interconnect. MI325X: 256 GB HBM3e at 6.0 TB/s, similar compute.

AMD's strengths are VRAM (class-leading among NVIDIA and AMD) and competitive pricing. A single MI300X holds what two H100s hold, and at aggressive pricing tiers (mainly via cloud partnerships) the $/GB/hr of VRAM is excellent. For models that are VRAM-bound — large dense (200B+) or MoE (400B+) models — MI300X is genuinely the right answer.

AMD's weaknesses are software. ROCm has improved dramatically; vLLM, SGLang, and PyTorch all work on ROCm and AMD inference kernels are competitive for mainstream architectures. But the ecosystem trails NVIDIA by a generation. Bespoke kernels, custom quantization schemes, non-standard architectures, and niche optimizations are more likely to be missing, slow, or buggy on ROCm than on CUDA. Debugging help is thinner; community mass is smaller; the "did someone fix this already" rate is lower.

When MI300X wins: very large dense models where one MI300X replaces two H100s/H200s on VRAM alone, organizations that already have ROCm expertise, and customers of clouds where AMD pricing is aggressive (some specialized clouds offer MI300X at substantial discounts to H100/H200).

When MI300X loses: standard 70B workloads where NVIDIA ecosystem maturity matters more than AMD's VRAM edge, teams without ROCm expertise who can't afford the debugging time, and workloads that rely on NVIDIA-specific kernels (certain FP8 recipes, some attention variants).

The 2026 arithmetic: if your workload is in ROCm's well-supported subset (Llama family, Qwen, Mistral, standard attention, standard quantization), MI300X is a legitimate choice with 10–30% $/token savings possible. If you have unusual model architecture, specialized kernels, or cutting-edge quantization, NVIDIA is the safer bet.

### 6.7 Summary: best GPU by workload

Putting the deep-dives together into a concise map:

| Workload | Best GPU for $/token | Runner-up | Why |
|---|---|---|---|
| 7B–8B high-throughput chat | L40S or A100 | H100 | Bandwidth not the bottleneck at this size |
| 7B–8B ultra-low-latency voice | H100 | H200 | Bandwidth matters at batch 1–4 |
| 13B–34B chat | H100 | H200 | Fits single-GPU, bandwidth adequate |
| 70B interactive | H200 (TP=1) | 2× H100 TP=2 | VRAM + bandwidth eliminates TP tax |
| 70B batch / offline | 2× H100 reserved | H200 | Lower hourly wins when SLO is relaxed |
| 120B MoE | H200 (TP=1) or B200 | 2× H100 TP=2 | MoE decode is cheap; VRAM for experts |
| 200B+ dense | B200 (TP=2) or MI300X (TP=1) | H200 TP=4 | VRAM forces multi-GPU; minimize count |
| 400B+ MoE | 8× B200 with EP | 8× H200 with EP | Frontier scale only |
| Long-context RAG (32k–128k) | H200 or B200 | H100 TP=2 | KV cache dominates; bandwidth + VRAM win |
| High-density edge | L4 | L40S | Power and cost per unit matter most |

## 7. Decision Framework, with Trade-offs Explicit

A practical decision process, from workload analysis to deployment. Each step names the trade-off it resolves.

### 7.1 Step 1: Quantify the workload

Measure, do not guess. You need:

Prompt tokens (p50, p95, p99): drives prefill cost and TTFT distribution.
Output tokens (p50, p95, p99): drives decode cost and wall-clock.
QPS (sustained, peak, peak-to-average ratio): drives concurrency and capacity.
Latency SLO (TTFT p95, ITL p95, E2E p95): drives batch size and hardware tier.
Concurrency target (derived from QPS × avg request duration).
Growth expectation over 12 months.

Trade-off at this step: the effort of measurement vs the accuracy of sizing. A half-day of logging production or a realistic staging load saves weeks of either over-provisioning or under-provisioning.

### 7.2 Step 2: Pick a model and quantization

Trade-off: quality vs hardware cost.

Fine-grained quality evaluation on your own tasks comes first. Public benchmarks (MMLU, HumanEval, etc.) are weakly predictive for specialized tasks. Pick the smallest model that meets your product quality bar.

Quantization: choose FP8 if hardware supports it, with per-channel or per-token scaling. Test on your evaluation set — quality impact is near-zero on most 2026 models, but not always on every model or every task. KV cache quantization to FP8 is almost always safe. INT4 or FP4 only after FP8 has been validated, and only if VRAM or bandwidth are binding after FP8.

### 7.3 Step 3: Compute VRAM and bandwidth budgets

Apply the formulas from section 2.6 to your specific model and workload. Include KV cache at realistic concurrency and context length, activations, framework overhead, and 15% headroom.

Map VRAM requirements against the hardware table in section 6. Identify 2–3 candidate hardware configurations that satisfy VRAM with headroom.

Trade-off at this step: conservative vs aggressive sizing. Being too aggressive means OOMs under traffic spikes; being too conservative means paying for unused capacity. The 15% headroom rule is a reasonable default; adjust based on how predictable your traffic is.

### 7.4 Step 4: Estimate throughput at your SLO

For each candidate hardware configuration, estimate aggregate throughput at the batch size that keeps ITL p95 within your SLO. Use the decode bandwidth formula as a first-order estimate, then cross-check against published vLLM / SGLang / TensorRT-LLM benchmarks for your specific model on your specific hardware.

Apply a realism factor: theoretical maximum is rarely achieved. Well-tuned production serving delivers 60–85% of the theoretical bandwidth-limited ceiling. Use 70% as a planning number unless you have benchmark data specific to your situation.

### 7.5 Step 5: Check latency feasibility

TTFT: prompt_tokens / (FLOPs × GPU count × utilization). Must beat SLO at p95 prompt length.

ITL: model_size / (bandwidth × GPU count for TP × 0.7). Must beat SLO at target batch size.

If either fails, options are: more GPUs (pays TP tax), smaller model (quality trade), shorter prompts (product trade), more aggressive quantization (quality trade), or chunked prefill + disaggregated serving (complexity trade). Pick based on which trade you can live with.

### 7.6 Step 6: Compute $/token for each option

For each shortlisted hardware configuration, compute:

```
hourly cost of the minimum viable configuration
÷ estimated tok/s at your SLO × 3600
= $/token
```

Then apply the pricing tier: on-demand if demand is uncertain, reserved if stable, spot for batch. Compare the resulting numbers side by side.

Compare against commercial API prices for equivalent quality. If your self-host number isn't at least 30% cheaper after including operational cost (0.3–0.7 FTE engineering per model per year), the API is the right choice for now.

### 7.7 Step 7: Stress test the choice

Run "what if" scenarios:

What if QPS doubles? Does the plan scale linearly, or does latency collapse at 2× load?
What if prompt length distribution shifts (average moves from 2k to 8k)? Prefill may become bottleneck.
What if output length distribution shifts? Decode may become bottleneck.
What if a GPU fails? Redundancy and failover.
What if the model changes? Have you over-specialized to one architecture?
What if pricing shifts? Are you locked in to yesterday's prices via 3-year reservation?

The best plan survives all scenarios with graceful degradation, not cliff failures.

### 7.8 Step 8: Deploy, measure, iterate

Paper math is accurate to ±25%. Production reveals surprises. Plan for 20% over-provisioning until you have at least a month of real data. After that, tighten based on actual utilization and latency metrics.

Revisit the decision quarterly. Hardware availability, pricing, and software maturity shift every 3–6 months. A decision that was right in Q1 2026 may not be right in Q3 2026.

## 8. Worked End-to-End Example

A detailed walkthrough of the full decision process for a realistic enterprise scenario.

### 8.1 Scenario

Enterprise RAG product serving knowledge workers. Llama-3-70B-Instruct in FP8. Traffic profile from production measurement:

SLOs: TTFT p95 < 1.5 seconds, ITL p95 < 35 ms. These match user research showing noticeable unresponsiveness above 2 second TTFT and sluggish typing feel above 40 ms ITL.

Traffic: 3 QPS sustained during business hours, peaks to 15 QPS during morning hours (10-11am user local time). Outside business hours, 0.3 QPS. Average prompt 3,200 tokens (RAG context included). Average output 380 tokens. Max context 32k tokens (rare, <2% of requests).

Expected growth: 50% YoY in QPS over 12 months.

### 8.2 Step 1: Concurrency estimate

Average request duration ≈ TTFT + output_tokens × ITL = 1.0 s + 380 × 0.025 s = 10.5 s.

Peak concurrency = 15 QPS × 10.5 s = ~158 in-flight requests at peak.

### 8.3 Step 2: VRAM budget at peak concurrency

Llama-3-70B FP8 weights: 70 GB.

KV cache per token with GQA (8 KV heads, 128 head dim, 80 layers) and FP8 KV quantization: 160 KB/token.

KV cache total at peak: 158 concurrent × 3580 avg tokens per sequence × 160 KB = ~90 GB.

Activations and framework overhead: ~8 GB.

Subtotal: 168 GB. With 15% headroom: ~193 GB.

### 8.4 Step 3: Hardware candidates

Apply the VRAM budget against available hardware:

Option A, 2× H100 SXM (160 GB VRAM total): fails headroom. Would need to cap concurrency or context. Not recommended.

Option B, 1× H200 (141 GB VRAM): fails even tighter. Would require aggressive KV management or dropping to INT4. Not recommended.

Option C, 2× H200 (282 GB VRAM): comfortable headroom. TP=2 over NVLink. Plausible.

Option D, 1× B200 (192 GB VRAM): just within budget. TP=1 eliminates interconnect tax. Plausible.

Option E, 1× MI300X (192 GB VRAM): fits. Software ecosystem risk requires team expertise to evaluate.

### 8.5 Step 4: Throughput at the SLO

Estimated aggregate decode throughput at ITL ≤ 35 ms (based on published benchmarks and first-principles math):

| Option | tok/s aggregate | Max supportable batch at 35 ms ITL |
|---|---|---|
| 2× H200 | ~2,500 | ~80 |
| 1× B200 | ~3,000 | ~100 |
| 1× MI300X | ~2,200 | ~70 |

Required aggregate tok/s at peak (158 concurrent users each decoding): 158 / 0.035 = ~4,500 tok/s.

None of the single-replica options meets peak alone. You need multiple replicas.

### 8.6 Step 5: Replica plan

Two replicas provide both capacity and failover redundancy (non-negotiable for production).

Option C: 2 replicas × 2× H200 = 4 H200s total. Aggregate ~5,000 tok/s at SLO. Meets peak.

Option D: 2 replicas × 1× B200 = 2 B200s total. Aggregate ~6,000 tok/s. Meets peak with comfortable margin.

Option E: 2 replicas × 1× MI300X = 2 MI300X total. Aggregate ~4,400 tok/s. Marginal; might fail at unexpected spikes.

### 8.7 Step 6: Cost

Representative 2026 pricing (varies by provider and region, validate yours):

| Plan | Hourly | Monthly (×720h) |
|---|---|---|
| 4× H200 on-demand @ $5.5 | $22 | $15,840 |
| 4× H200 1-yr reserved @ $3.8 | $15.2 | $10,944 |
| 2× B200 on-demand @ $8 | $16 | $11,520 |
| 2× B200 1-yr reserved @ $5.5 | $11 | $7,920 |
| 2× MI300X on-demand @ $4.5 | $9 | $6,480 |

At average business-hour load (assume 2,500 tok/s aggregate sustained):

$/Mtok for 4× H200 reserved: $15.2 / (2,500 × 3600 / 1e6) = ~$1.69.
$/Mtok for 2× B200 reserved: $11 / (3,000 × 3600 / 1e6) = ~$1.02.
$/Mtok for 2× MI300X on-demand: $9 / (2,200 × 3600 / 1e6) = ~$1.14.

### 8.8 Step 7: Stress tests

What if average output doubles to 800 tokens? Concurrency at 15 QPS rises to ~245 requests. VRAM for KV roughly doubles to ~180 GB. Option D (B200 single-GPU) starts to fail; Option C (2× H200 per replica) stays fine; Option E (MI300X) becomes tight.

What if peak QPS doubles to 30 QPS? All plans need 3 replicas instead of 2. Option D (B200) becomes 3 B200s; still cheapest on reserved pricing.

What if traffic halves? Reserved capacity becomes overprovisioned. On-demand becomes more attractive. 3-year reservations would be painful.

What if FP4 validation completes on Llama-3-70B? B200 throughput could rise to ~5,000 tok/s, cutting $/Mtok to ~$0.60 for Option D reserved. H200 and MI300X cannot benefit from FP4. This strengthens the case for B200 long-term.

### 8.9 Step 8: Final recommendation with trade-offs

For a risk-averse enterprise RAG product in early 2026, 4× H200 on 1-year reserved is the safest choice. Mature ecosystem, predictable performance, zero software surprise, $1.69/Mtok is reasonable for the quality tier.

For a cost-focused team with engineering bandwidth to validate newer stacks, 2× B200 on 1-year reserved is compelling. 40% $/token savings, with FP4 as upside once validated, at the cost of early-adopter ecosystem risk.

MI300X is interesting for teams with ROCm expertise, but the software risk is real for a customer-facing production product on a 2026 timeline. Revisit in 2027.

For teams that cannot commit reservations, 2× B200 on-demand at $11,520/month is still cheaper than 4× H200 reserved. The trade-off is flexibility (can scale down or switch hardware) for higher unit cost.

## 9. Common Pitfalls

Trade-offs that get ignored, with their consequences.

Sizing for p50 and discovering p99 in production. Latency SLOs live at p95 and p99. A plan that works at median prompt length and median output length falls apart at the tails. Always capacity-plan from p95/p99 distributions.

Treating prefill and decode as one workload. The asymmetry (section 1) is the central fact. Two-stage disaggregated serving is overkill at low QPS but the framing is useful even then: know which phase your traffic spends time in.

Ignoring KV cache scaling. Fit tests at batch 1 are misleading. A model that "fits" at batch 1 can OOM at batch 32 because KV cache grows linearly with both users and context. Always size for peak concurrency, not for a smoke test.

Blind tensor parallelism. TP=4 can be *slower* than TP=1 for single-user workloads because the interconnect overhead at low batch is large relative to the compute saved. Always measure before committing to a TP degree.

Skipping FP8 on H100+ hardware. In 2026 this is leaving 40% of potential throughput on the floor. Quality delta with proper calibration is near-zero on mainstream models. If you have not migrated to FP8, that is the highest-leverage change available to you.

Using static batching. LLM serving has been dominated by continuous batching for three years. Any serving stack that does not use continuous batching (plain HuggingFace generate, hand-written loops) is 2–5× slower than vLLM or SGLang. Swap.

Benchmarking synthetic uniform workloads. A benchmark of "256 in, 256 out, batch 32" does not match a production distribution of "mean 3200 in, mean 380 out, high-variance arrivals". Throughput numbers from synthetic benchmarks can be off by 3× from production reality. Always profile and benchmark on realistic traffic shapes.

Over-reserving during a hardware transition. Locking in 3-year reservations on H100 in early 2026 is committing to today's economics while Blackwell is actively redefining the frontier. 1-year reservations are the safer choice during transitions.

Over-indexing on hourly GPU price. Hourly price is 40% of the $/token story. Software stack, quantization, batching, SLO, and prefix caching together drive the rest. A team that optimizes only hardware purchase and ignores software is leaving most of the savings on the table.

Ignoring the API alternative. Running 200k tokens/day on a dedicated H200 is almost pure waste compared to using a commercial API. Self-host breaks even above roughly 100M tokens/day for 70B-class open models. Know your volume before building.

Operational cost blindness. Each self-hosted model replica consumes engineering time: oncall rotations, version upgrades, security patches, monitoring, capacity planning. A realistic estimate is 0.3–0.7 FTE per model per year. At fully-loaded engineering costs, that is $50K–$150K annually. This must appear in your TCO comparison; without it, self-host looks cheaper than it is.

Not re-evaluating. GPU supply, pricing, ecosystem maturity, and model architectures all shift every 3–6 months. A quarterly review of your serving economics is a small cost and catches expensive mistakes.

## 10. Quick Reference

### 10.1 Per-GPU single-GPU decode ceiling

Approximate aggregate tok/s at FP8 with well-tuned serving and realistic mixed-length workloads. Real numbers vary ±30% with serving stack, request shape, and tuning.

| GPU | 8B | 13B | 34B | 70B | 120B MoE |
|---|---|---|---|---|---|
| L4 | ~500 | — | — | — | — |
| L40S | ~1,500 | ~1,000 | ~500 | — | — |
| A100 80G | ~2,500 | ~1,800 | ~900 | ~400 (tight) | — |
| H100 SXM | ~5,000 | ~3,500 | ~1,800 | ~1,200 (tight) | ~1,800 |
| H200 | ~6,500 | ~4,500 | ~2,400 | ~1,800 | ~2,800 |
| B200 (FP8) | ~10,000 | ~7,000 | ~4,000 | ~3,200 | ~4,500 |
| B200 (FP4, validated) | ~15,000 | ~11,000 | ~6,500 | ~5,000 | ~7,500 |
| MI300X | ~5,500 | ~4,000 | ~2,200 | ~1,800 | ~2,800 |

### 10.2 VRAM sizing at batch 32, 8k context, GQA, FP8 weights + FP8 KV

| Model | Weights | KV cache | Activations + overhead | Total + 15% headroom |
|---|---|---|---|---|
| 8B | 8 GB | 16 GB | 4 GB | ~32 GB |
| 13B | 13 GB | 24 GB | 5 GB | ~48 GB |
| 34B | 34 GB | 48 GB | 7 GB | ~102 GB |
| 70B | 70 GB | 40 GB (GQA) | 8 GB | ~135 GB |
| 120B MoE | 120 GB | 30 GB | 10 GB | ~184 GB |

### 10.3 When to use which hardware (at a glance)

| Situation | Default choice |
|---|---|
| Under 30M tokens/day, uncertain future | Commercial API |
| 30M–100M tokens/day, predictable | API or 1 rented H100/H200 |
| 100M–1B tokens/day, steady | Reserved H200 or B200 cluster |
| Over 1B tokens/day, core product | Owned/colo + reserved cloud burst |
| Frontier closed model needed | Commercial API (only option) |
| Strict data residency requirement | Self-host regardless of volume |
| Latency under 20 ms ITL | H200/B200 only; may need speculative decoding |
| Model > 200B or > 400B MoE | B200 or MI300X cluster |

### 10.4 $/token lever cheat sheet

| Lever | $/token impact |
|---|---|
| BF16 → FP8 (full W8A8 + KV) | –35% to –45% |
| KV quantization only (BF16 weights) | –10% to –25% |
| HF Transformers → vLLM/SGLang/TensorRT-LLM | –30% to –60% |
| ITL SLO 25 ms → 50 ms | –30% to –50% |
| H100 → H200 (70B model) | –20% to –35% |
| Prefix caching (RAG with templates) | –15% to –40% |
| Speculative decoding (batch ≤ 8) | –20% to –50% |
| Spot vs on-demand (batch workloads) | –40% to –60% |
| Reserved 1-yr vs on-demand | –30% to –40% |
| Reserved 3-yr vs on-demand | –50% to –60% |
| Disaggregated prefill/decode (large scale) | –20% to –40% |

Compounding 3–4 of these moves typical new deployments from "shockingly expensive" to "reasonable" by 5–8×. Most of the easy wins come from quantization and serving stack; the rest from SLO calibration and pricing tier.

## 11. Conclusion

Choosing a GPU for LLM serving is not a product comparison. It is a multi-dimensional optimization problem with at least a dozen levers, each of which trades off against one or two others. The levers in summary form:

Prefill vs decode: your traffic shape (prompt/output ratio) determines whether you need FLOPs-rich or bandwidth-rich hardware. Two workloads with identical total tokens can require different GPUs.

Bandwidth vs FLOPs vs VRAM vs interconnect: every vendor picks a ratio, and your job is to match yours to your workload's. H100 is balanced, H200 is decode-tilted, B200 is everything-tilted, L40S is FLOPs-rich but bandwidth-poor, MI300X is VRAM-rich.

Latency vs throughput vs concurrency: the iron triangle. Your SLO is the binding constraint; batch size follows from the SLO; hardware sizing follows from the batch size. Relaxing the SLO by 2× typically cuts cost by 2–3×.

Quality vs speed vs memory: quantization compounds across all three. FP8 is nearly free on H100+. FP4 is compelling on Blackwell after quality validation. KV quantization is almost always worth enabling.

Cloud vs owned, spot vs reserved: utilization and predictability drive the answer. Owned hardware wins at high utilization on mature hardware; cloud wins at low or uncertain utilization; spot wins for interruptible batch workloads.

Self-host vs API: volume, data residency, and operational capacity drive the answer. Below roughly 100M tokens/day of a 70B-class model, commercial APIs almost always win on total cost. Frontier closed models are API-only.

Hardware generation: H200 is the 2026 mainstream inference sweet spot. B200 is where frontier and bleeding-edge go. MI300X is the credible non-NVIDIA option for teams with AMD expertise. H100 remains viable for smaller models. A100 remains viable for budget workloads. L40S remains viable for density-focused small-model serving.

Two habits separate teams who get this right from teams who don't:

The first habit is measurement. Paper math gets you within 25%; the last 25% is empirical. Before committing capital, benchmark your model on the candidate hardware with your actual prompt distribution, your actual SLO, and your actual serving stack. The cost of a week of rental for benchmarking is trivial compared to the cost of a wrong multi-year reservation.

The second habit is re-evaluation. The right answer in Q1 2026 is not necessarily the right answer in Q4 2026. New hardware ships. New kernels ship. New quantization schemes ship. Pricing shifts. Set a quarterly review cadence on serving economics and ship upgrades when the frontier has moved meaningfully in your favor.

Everything else in LLM serving economics is tactics. Get the strategy right — know your workload, understand the trade-offs, measure, re-evaluate — and the tactics fall into place.

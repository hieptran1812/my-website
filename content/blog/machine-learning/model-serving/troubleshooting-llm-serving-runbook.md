---
title: "Troubleshooting LLM Serving: A 3am Production Runbook"
date: "2026-07-04"
publishDate: "2026-07-04"
description: "The failures that page you at 3am when serving LLMs — OOM, preemption thrash, TTFT cliffs, NCCL hangs, garbage output — each as a named incident with symptom, confirming signal, root cause, and the exact config change that fixes it."
tags:
  [
    "model-serving",
    "inference",
    "ml-infrastructure",
    "llm-serving",
    "vllm",
    "gpu",
    "troubleshooting",
    "kv-cache",
    "nccl",
    "observability",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/troubleshooting-llm-serving-runbook-1.webp"
---

The pager goes off at 03:12. The alert says `TTFT p99 = 4.2s` against a 1-second SLO, and the on-call channel is filling with screenshots of a chat product that now takes five seconds to start streaming. You have a laptop, a VPN connection, and a serving fleet you did not deploy. What do you type first?

This post is the answer to that question, written as a runbook rather than a survey. Every serving incident I have ever been paged for falls into one of four symptom classes — an out-of-memory crash, a latency SLO breach, a throughput collapse, or bad output — and each class has a single confirming signal that points almost immediately at the root cause. The skill that separates a five-minute mitigation from a two-hour outage is not knowing more failure modes; it is knowing which signal to read first so you stop guessing. The decision tree in Figure 1 is the shape of that skill, and the rest of this post fills in each branch.

![A four-way decision tree that classifies an LLM serving page by symptom class then routes each to its single confirming signal](/imgs/blogs/troubleshooting-llm-serving-runbook-1.webp)

Every technique here is a move on the same triangle that governs all of model serving: **latency, throughput, and cost trade against each other**, and most incidents are the triangle taking revenge on a config that pushed one corner too far. Someone raised `max_num_seqs` to chase throughput and now the KV cache thrashes. Someone lowered `gpu_memory_utilization` to avoid an OOM and now concurrency is starved. The runbook below is organized as named incidents — each with **symptom, how to confirm, root cause, immediate mitigation, and permanent fix** — but the connective tissue is always the triangle and the two equations in the next section. Read those first; they are what let you triage a system you have never seen. If you are new to the vocabulary here (TTFT, TPOT, KV cache, continuous batching), the series intro [what is model serving](/blog/machine-learning/model-serving/what-is-model-serving) defines every term from scratch.

## 1. The two equations every serving on-call needs

You cannot troubleshoot a system you cannot do arithmetic about. Ninety percent of LLM-serving incidents reduce to one of two questions: **does it fit in GPU memory**, and **is the queue backing up or is the model slow**. Two small equations answer both, and once you can evaluate them on the back of an envelope you can triage a serving process you have never logged into before.

### The GPU memory equation

An inference process holds four things in high-bandwidth memory (HBM), and their sum must stay under the card's capacity:

$$M_{\text{weights}} + M_{\text{KV}} + M_{\text{act}} + M_{\text{overhead}} \le C_{\text{GPU}}$$

The four terms have very different behavior, and knowing which one dominates is the entire OOM triage:

- **Weights** ($M_{\text{weights}}$) are fixed the moment you pick a model and a dtype. For a model with $P$ parameters at $s$ bytes each, $M_{\text{weights}} = P \cdot s$. A 32B model in fp16 is ${32 \times 10^9 \cdot 2 = 64}$ GB. This term does not move under load — it is either too big on startup or it is not your problem.
- **KV cache** ($M_{\text{KV}}$) is the one that grows with traffic, and it is the term that OOMs you under load. Every token every active request has generated must keep its key and value vectors resident so attention can look back at them. The per-token cost is:

$$b_{\text{KV}} = 2 \cdot L \cdot h_{kv} \cdot d_{head} \cdot s$$

where $L$ is layers, $h_{kv}$ is the number of key/value heads (smaller than the number of attention heads under grouped-query attention), $d_{head}$ is head dimension, $s$ is bytes per element, and the leading 2 is because you store both K and V. For Qwen2.5-32B ($L=64$, $h_{kv}=8$, $d_{head}=128$, fp16) that is ${2 \cdot 64 \cdot 8 \cdot 128 \cdot 2 = 262{,}144}$ bytes, or **256 KB per token**. The total KV footprint is $b_{\text{KV}}$ times the sum of sequence lengths across all in-flight requests, so it scales with *concurrency times context length* — the two knobs you reach for when you OOM under load.
- **Activations** ($M_{\text{act}}$) are the transient tensors of the forward pass. They spike during prefill (processing the whole prompt at once) in proportion to the number of tokens processed in a step, and they are usually small during decode. A large prefill batch is the hidden third OOM cause: the weights fit, the KV fits, and then a 32k-token prompt arrives and the activation spike tips the process over.
- **Overhead** ($M_{\text{overhead}}$) is the CUDA context, NCCL buffers, allocator fragmentation, and the framework's own bookkeeping — typically 2–8 GB and larger under tensor parallelism.

The single most useful derived quantity is the **token budget**: given whatever KV memory is left after weights, activations, and overhead, how many token-slots can you hold?

$$N_{\text{tokens}} = \frac{M_{\text{KV}}}{b_{\text{KV}}}$$

That number, divided by your average context length, is your real concurrency ceiling. It is worth pausing on two subtleties that trip people up. First, the activation term is not a fixed reservation — it is a *transient peak* that the serving framework has to leave room for, which is why vLLM runs a "memory profiling" forward pass at startup: it executes a dummy batch at `max_num_batched_tokens` to measure the real activation peak, subtracts weights plus that peak from the utilization target, and gives whatever remains to KV blocks. If the profiling pass itself does not fit, you OOM before serving a single request, and the fix is to shrink the profiling batch (`max_num_batched_tokens`) or the utilization target — not to add KV. Second, the overhead term grows under tensor parallelism: each rank holds its shard of the weights *plus* NCCL communication buffers, and those buffers scale with the number of ranks and the hidden size. On an 8-way tensor-parallel deployment the per-GPU overhead can be several gigabytes larger than on a single card, which is why a model that "should" fit across 8 GPUs at one-eighth the weights sometimes does not — the overhead ate the margin you were counting on.

Figure 2 shows the whole budget on one H100 80GB across three configs, and it is worth internalizing because it is the picture you sketch on every OOM page.

![A grid showing the GPU memory budget on one H100 split into weights, KV cache, activations, and overhead across naive, tuned, and quantized configurations](/imgs/blogs/troubleshooting-llm-serving-runbook-2.webp)

Read the columns left to right. The naive fp16 config sums to 82 GB and OOMs — weights alone eat 64 GB, leaving so little for KV that the cache is starved and the process crashes on the first activation spike. Capping utilization and context brings it to exactly 80 GB, but with only 8 GB of KV you can hold ${8 \times 10^9 / 262{,}144 \approx 30{,}500}$ token-slots — perhaps 15 concurrent requests at 2k context. Quantizing the weights to 4-bit AWQ collapses $M_{\text{weights}}$ from 64 GB to 18 GB; that freed 46 GB becomes KV, and with fp8 halving $b_{\text{KV}}$ to 128 KB/token you now hold roughly 390,000 token-slots — a 12× jump in the concurrency you can serve from the identical card. Every OOM fix in this runbook is a move within that picture: shrink a term, or convert a shrunk term into more of the term you actually need. The details of squeezing the most out of KV memory are a topic of their own in [KV cache optimization](/blog/machine-learning/model-serving/kv-cache-optimization).

### The overload-versus-slow-model signal

The second equation is Little's Law, and it is what tells you whether a latency page is a capacity problem or a performance problem — a distinction that determines whether you add replicas or change a flag. For a stable queue with arrival rate $\lambda$ (requests per second) and mean time-in-system $W$, the average number of requests resident is:

$$L = \lambda \cdot W$$

The trap is that when latency ($W$) rises, two completely different causes produce the identical symptom. Either the *service time* rose (the model genuinely got slower — longer prompts, fragmentation, a lost CUDA graph) or the *arrival rate* exceeded service capacity and requests are piling up in the queue. The fix for the first is a config change; the fix for the second is admission control or more replicas. Applying the wrong one makes it worse.

The distinguishing signal is to **decompose latency into queue time and compute time**. Any serious LLM server exposes both. In vLLM the relevant Prometheus metrics are `vllm:request_queue_time_seconds` (time spent waiting before the engine picked the request up) and `vllm:time_to_first_token_seconds` (the compute to produce token one). The rule is mechanical:

- If **queue time dominates** the rising latency and GPU utilization is pinned near 100%, you are overloaded — arrivals outrun capacity. Shed load or scale out.
- If **compute time itself rose** while GPU utilization is *low*, the GPU is not your bottleneck. Something off-device is — CPU-side tokenization, a lock, a slow embedding lookup, a synchronous downstream call. Adding GPUs will do nothing.
- If compute time rose *and* GPU utilization is high, the work per request genuinely grew — longer contexts, a batch-size or fragmentation change. Look at the config and the input distribution, not the replica count.

Those three cases, read off two numbers, cover almost every latency incident you will see. Keep both equations within reach; the rest of this runbook is their application to named failures.

## 2. Incident: CUDA out of memory, at startup and under load

**Symptom.** The process dies with `torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate ... GiB` — either during model loading before it serves a single request, or minutes into a traffic ramp after it has been happily serving.

**How to confirm.** The two sub-cases have different signatures and different fixes, so separate them first. Run `nvidia-smi` and read the memory column, and check *when* the crash happens relative to the first served request.

- **Startup OOM**: the crash lands during "Memory profiling" or weight loading, before `Uvicorn running on ...` appears in the log. `nvidia-smi` at the moment of death shows memory nearly full from weights and the profiling forward pass alone.
- **Load OOM**: the server starts, serves for a while, and dies when concurrency or context length climbs. The crash correlates with `vllm:num_requests_running` and `vllm:gpu_cache_usage_perc` approaching their ceilings just before it falls over.

**Root cause.** Startup OOM means the *fixed* terms do not fit: weights plus overhead plus the activation headroom vLLM needs for its profiling batch exceed the card, usually because `gpu_memory_utilization` was set too high (leaving no room for the profiling forward pass) or the model simply needs more than one GPU. Load OOM means the *growing* term escaped its cap: `max_num_seqs` and `max_model_len` were left unbounded, so a burst of long-context requests grew the KV cache past the budget. The triage flow in Figure 3 is the branch you walk: measure, decide which term dominates, apply the matching fix.

![A branching graph that starts at nvidia-smi, splits on which memory term dominates, and converges each branch on a configuration fix that fits the process under 80GB](/imgs/blogs/troubleshooting-llm-serving-runbook-3.webp)

**Immediate mitigation.** For a startup OOM you cannot afford to debug live, the fastest safe move is to lower `gpu_memory_utilization` (e.g. 0.95 → 0.88) and `max_model_len` to a value your traffic actually uses (32768 → 8192). Both shrink the pressure without touching the model. For a load OOM, cap `max_num_seqs` immediately — that hard-bounds the KV cache regardless of what traffic does. Restart, confirm it holds, then plan the permanent fix.

**Permanent fix.** The durable answer depends on which term dominates. If weights dominate, quantize (AWQ or GPTQ 4-bit, or FP8 on H100) or shard across GPUs with tensor parallelism. If KV dominates, cap concurrency to your real token budget and switch `kv_cache_dtype` to fp8 to double it. If activations dominate (long prompts), enable chunked prefill so a giant prompt is processed in bounded chunks rather than one memory-spiking pass. Here is the concrete vLLM configuration that applies all three:

```python
from vllm import LLM

# BEFORE — defaults OOM on one H100 80GB:
#   64 GB weights + a 0.95-utilization KV reservation + a 32k-token
#   profiling batch = ~82 GB > 80 GB, dies during memory profiling.
# llm = LLM(model="Qwen/Qwen2.5-32B-Instruct")

# AFTER — fit the process under 80 GB and bound every growing term.
llm = LLM(
    model="Qwen/Qwen2.5-32B-Instruct-AWQ",  # 4-bit weights: 64 GB -> ~18 GB
    quantization="awq",
    kv_cache_dtype="fp8",          # halve KV bytes/token (H100 native fp8)
    gpu_memory_utilization=0.90,   # fraction of the 80 GB vLLM may reserve
    max_model_len=8192,            # caps per-sequence KV (context length)
    max_num_seqs=48,               # caps concurrency -> caps total KV
    enable_chunked_prefill=True,   # bound the prefill activation spike
    max_num_batched_tokens=2048,   # prefill chunk budget per engine step
    enforce_eager=False,           # keep CUDA graphs for decode throughput
)
```

Two gotchas worth stating plainly. First, `quantization="awq"` requires an AWQ-quantized checkpoint — pointing it at an fp16 checkpoint will not quantize on the fly; use the `-AWQ` model or quantize offline. Second, `gpu_memory_utilization` is a fraction of *total* card memory, not of the memory left after weights; if you have other processes on the GPU, that fraction includes their footprint, and the profiling pass will still OOM even at 0.90.

There is a fourth OOM variant that the memory equation predicts but that surprises people: the **tensor-parallel OOM**, where a model that clearly fits across N GPUs by weight count still OOMs. Tensor parallelism splits the weights across ranks, so each of 8 GPUs holds one-eighth of $M_{\text{weights}}$ — but it does *not* split the KV cache or the activation buffers proportionally in the way you might assume, and it *adds* per-rank NCCL communication buffers to $M_{\text{overhead}}$. The result is that the freed weight memory is partly re-consumed by overhead and by the KV cache each rank must hold for its share of the attention heads. The practical failure: a 72B model at ${72 \times 10^9 \cdot 2 = 144}$ GB fp16 needs at least two 80 GB cards for weights, but a 2-way split leaves so little KV headroom on each that concurrency is starved, and a naive operator concludes "it doesn't fit" when the real problem is that they need TP=4 or quantization to leave room for KV. The fix is the same discipline as everywhere else: budget all four terms *per rank* including the enlarged overhead, and pick the parallelism degree that leaves a usable KV budget, not merely one that fits the weights.

#### Worked example: triaging a startup OOM with numbers

The page: Qwen2.5-32B on a single H100 80GB dies during startup. `nvidia-smi` at crash shows `79.6 GiB / 80 GiB` used, and the traceback is inside vLLM's memory profiler. Walk the equation.

- Weights: ${32.5 \times 10^9 \cdot 2 = 65}$ GB fp16. That is measured, not guessed — the safetensors shards sum to about 65 GB.
- Configured `gpu_memory_utilization=0.95` targets ${0.95 \cdot 80 = 76}$ GB for the whole process. After 65 GB of weights, that leaves 11 GB for KV blocks plus the profiling activation pass.
- The profiling pass runs a forward at `max_num_batched_tokens`, and the operator had left `max_model_len=32768` with default batched tokens, so the activation peak for a full-context prefill is roughly 6–7 GB. 65 + a few GB of KV probe + 7 GB activation overshoots the 76 GB target and trips CUDA OOM before serving starts.

The fix path, in order of least disruptive first: drop `gpu_memory_utilization` to 0.90 (target 72 GB, weights 65, ~7 GB headroom), cap `max_model_len` to 8192 (activation peak drops to ~2 GB), and enable chunked prefill with `max_num_batched_tokens=2048` (activation peak becomes a bounded 2k-token chunk regardless of prompt length). Now the process fits: 65 GB weights + 8 GB KV + 2 GB activations + 4 GB overhead = 79 GB. It starts — but with only 8 GB of KV you have ${8 \times 10^9 / 262{,}144 \approx 30{,}500}$ token-slots, roughly 15 concurrent 2k-context requests, which is probably too few. So the *real* permanent fix is the AWQ path above: 18 GB weights frees 46 GB, most of which becomes KV, and concurrency jumps roughly 12×. The arithmetic told you the mitigation *and* revealed that the mitigation alone was not enough — that is the whole point of doing it.

## 3. Incident: KV-cache exhaustion and preemption thrash

**Symptom.** Throughput was healthy, then it collapsed — tokens per second fell by 3–5× — without an OOM crash and without a traffic drop. Latency got spiky and unpredictable. Nothing errored; the system just started grinding.

**How to confirm.** This is the signature failure of admitting more sequences than KV memory can hold, and it has an unambiguous metric fingerprint. Scrape the engine metrics and look at two counters: `vllm:num_preemptions_total` (climbing fast) and `vllm:gpu_cache_usage_perc` (pinned near 100%). If preemptions are accumulating while cache usage sits at the ceiling, you are thrashing. In the server log you will see lines like `WARNING: Sequence group ... is preempted by PreemptionMode.RECOMPUTE`. That word — *preempted* — is the diagnosis.

**Root cause.** When the scheduler admits a batch whose combined KV requirement exceeds the free blocks, it must evict some sequences to make room for others, then bring them back later. vLLM has two preemption modes, and both cost you. **Recompute** (the default) discards the evicted sequence's KV cache entirely and, when the sequence is rescheduled, recomputes its whole prefill from scratch — pure wasted compute. **Swap** copies the KV cache to CPU RAM over PCIe and back, which is bandwidth-bound and stalls the batch. Under sustained oversubscription the engine spends most of its cycles evicting and restoring rather than generating new tokens, so throughput craters even though the GPU looks busy. Figure 4 contrasts the thrashing state with the tuned steady state.

![A before-and-after figure contrasting oversubscribed preemption thrash that collapses throughput from 2100 to 400 tokens per second against a capped configuration with stable throughput](/imgs/blogs/troubleshooting-llm-serving-runbook-4.webp)

**Immediate mitigation.** Cap `max_num_seqs` to a value the KV budget can actually hold, and restart. This trades a little peak concurrency for enormous stability — a batch that fits and never preempts beats a bigger batch that thrashes every step. If you cannot restart, some gateways let you throttle admitted concurrency upstream, which starves the oversubscription indirectly.

**Permanent fix.** Size `max_num_seqs` from the token budget rather than from optimism. Compute $N_{\text{tokens}} = M_{\text{KV}} / b_{\text{KV}}$, divide by your *observed* average sequence length (not the maximum), and set the cap there with a safety margin. If that number is embarrassingly small, the real fix is to enlarge $M_{\text{KV}}$ — quantize weights to free memory, or switch to fp8 KV — rather than to keep admitting sequences you cannot hold. If your workload has a heavy tail of long sequences, prefer `preemption_mode="swap"` with enough CPU RAM so the occasional eviction pays PCIe cost instead of full recompute; for mostly-short sequences, recompute is cheaper. The scheduler mechanics behind admission and eviction are covered in depth in [request scheduling and preemption](/blog/machine-learning/model-serving/request-scheduling-and-preemption).

It helps to know the actual cost of each preemption mode so you can pick correctly rather than by folklore. **Recompute** discards a sequence's KV and, on reschedule, re-runs its prefill — a cost proportional to the sequence length times the per-token prefill FLOPs. For a sequence of $n$ tokens, that is roughly the same compute as generating the first token from scratch, and it is paid *every time* the sequence is evicted and brought back. Under heavy thrash a long sequence can be recomputed many times, so the wasted work grows super-linearly in how oversubscribed you are. **Swap** instead copies the sequence's KV cache to CPU RAM and back over PCIe; for a sequence holding $n \cdot b_{\text{KV}}$ bytes of KV, the cost is that many bytes divided by PCIe bandwidth (roughly 25–30 GB/s on PCIe 4.0), paid twice (out and back). The crossover is length-dependent: for a short sequence, recomputing its handful of tokens is far cheaper than a round-trip over PCIe; for a very long sequence, the recompute FLOPs dominate and swapping the bytes is cheaper. A 4k-token sequence at 256 KB/token holds 1 GB of KV — swapping it costs about ${2 \cdot 1 / 27 \approx 74}$ ms of PCIe time, while recomputing a 4k prefill on an H100 is often faster than that, so recompute wins for typical chat contexts and swap only pays for the long-document tail. The deeper point stands either way: *neither mode is free*, and the correct move is almost always to admit fewer sequences so you never preempt, not to optimize the preemption you should not be doing.

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen2.5-32B-Instruct-AWQ",
    quantization="awq",
    kv_cache_dtype="fp8",
    gpu_memory_utilization=0.90,
    max_model_len=8192,
    # Size this from the token budget, not from a round number.
    # ~50 GB KV / 128 KB per fp8 token = ~390k slots; at 2k avg context
    # that is ~190 concurrent, so 128 leaves comfortable headroom.
    max_num_seqs=128,
    preemption_mode="recompute",   # "swap" if you have a long-sequence tail
)
```

#### Worked example: diagnosing preemption thrash

The page: a fleet that was steady at 2,100 tokens/second per replica drops to 400 tokens/second after a product launch pushed traffic up. No OOM, no errors. The dashboard shows GPU utilization still at 95% — which is exactly why the team first assumed the GPUs were healthy and looked at the network. They were looking in the wrong place.

Scrape the metrics. `vllm:gpu_cache_usage_perc` reads 99.8% and `vllm:num_preemptions_total` is climbing by hundreds per minute. That is the tell: the GPU is busy, but it is busy *recomputing evicted prefills*, not producing new tokens. Do the budget arithmetic. The KV budget is 8 GB fp16 (this replica was never quantized), which is ${8 \times 10^9 / 262{,}144 \approx 30{,}500}$ token-slots. The launch pushed average context to 2,300 tokens and offered concurrency to ~200 sequences, which needs ${200 \cdot 2300 = 460{,}000}$ token-slots — fifteen times what fits. The scheduler admits, discovers it cannot hold them, preempts, recomputes, and repeats. Measured recompute overhead consumes roughly 80% of compute, which is precisely the 2,100 → 400 collapse (a ~5× loss).

Two fixes, layered. Immediately: cap `max_num_seqs` to ~13 (30,500 / 2,300) so the batch always fits — throughput returns to ~2,100 tokens/second at the cost of a longer queue, which you handle with admission control (Section 4). Durably: quantize this replica to AWQ + fp8 KV like the healthy replicas, lifting the KV budget to ~50 GB and the token budget to ~390k slots, so `max_num_seqs=128` fits with margin and the queue drains. The thrash was never a network problem; it was a budget the config had written a check it could not cash.

## 4. Incident: the TTFT cliff under burst

**Symptom.** Time-to-first-token is fine at steady load and then falls off a cliff during bursts — p99 TTFT jumps from 300 ms to several seconds while the fleet is not obviously saturated. Users see the "thinking" spinner hang before the first word streams.

**How to confirm.** This is the latency case from Section 1, so apply the decomposition. Compare `vllm:request_queue_time_seconds` against `vllm:time_to_first_token_seconds`. During a TTFT cliff the queue time is almost the entire TTFT — requests are waiting, not computing. Confirm with `vllm:num_requests_waiting`: if the waiting queue is deep (dozens to hundreds) while running requests sit at the concurrency cap, the cliff is a queueing phenomenon, full stop. Figure 5 is the lifecycle of exactly this incident, from the 03:12 alert to the postmortem.

![A timeline of a TTFT-cliff incident moving from the 03:12 alert through confirming queue depth, mitigating with load shedding, fixing with chunked prefill, to resolution and postmortem](/imgs/blogs/troubleshooting-llm-serving-runbook-5.webp)

**Root cause.** Two mechanisms compound. First, plain queueing: when arrivals momentarily exceed the rate at which the engine can start requests, the waiting queue grows and every new request inherits the whole backlog as its TTFT. Second, and more insidious, **head-of-line blocking from prefill**. A long prompt monopolizes the GPU during its prefill pass, and every decode step for already-running requests — and every new request's first token — waits behind it. One 30k-token prompt can stall the first token of a hundred short requests. Bursts make both worse at once: more arrivals, and a higher chance that a giant prompt lands in the middle of them.

The reason latency *cliffs* rather than rising smoothly is worth understanding, because it explains why the incident feels so sudden. Model the server as a queue with utilization $\rho = \lambda / \mu$, where $\lambda$ is the arrival rate and $\mu$ is the service rate. Standard queueing theory says the expected waiting time scales like $W_q \propto \frac{\rho}{1 - \rho}$ — a curve that is nearly flat while $\rho$ is below about 0.7 and then shoots to infinity as $\rho$ approaches 1. A fleet running comfortably at 60% utilization has almost no queue wait; the same fleet at 92% utilization has a queue wait several times its service time, and at 99% the wait is effectively unbounded. A burst does not add latency linearly — it pushes $\rho$ from the flat part of the curve onto the vertical part, and TTFT cliffs. This is why "the fleet was only at 90% and then everything fell over" is such a common and confusing report: 90% is already on the knee of the curve. It is also why admission control works: by capping admitted concurrency you clamp $\rho$ below the knee, trading a few rejected requests for a queue that never explodes. And it is why the correct capacity target for a latency-sensitive service is 60–70% utilization, not 95% — the last 30% of the card is latency insurance, not waste.

One more lever belongs in this section: **prefix caching**. If your traffic shares long common prefixes — a fixed system prompt, a few-shot preamble, a RAG context reused across turns — then caching the KV for that shared prefix means new requests skip re-computing it, which directly shrinks the prefill work that causes head-of-line blocking. On workloads with heavy prefix sharing, enabling prefix caching can cut TTFT more than any queueing change, because it attacks the size of the prefill rather than its scheduling.

**Immediate mitigation.** Shed load at the edge. It is counterintuitive under a latency alert, but rejecting or queueing the marginal request with a fast `429 Too Many Requests` protects the requests already in flight — a served-slower-for-some beats timed-out-for-all. Cap admitted concurrency at the gateway to the number your replicas can start promptly.

**Permanent fix.** Turn on **chunked prefill** and set an explicit **admission control** policy. Chunked prefill splits a long prompt into fixed-size chunks (`max_num_batched_tokens`) and interleaves them with ongoing decode steps, so a giant prompt no longer monopolizes the GPU — it takes its turn a chunk at a time, and decode latency for everyone else stays smooth. Admission control bounds the queue so TTFT cannot cliff past your SLO in the first place. The engine-side config:

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen2.5-32B-Instruct-AWQ",
    quantization="awq",
    kv_cache_dtype="fp8",
    enable_chunked_prefill=True,   # interleave prefill chunks with decode
    max_num_batched_tokens=2048,   # a 30k prompt is 15 chunks, not one stall
    max_num_seqs=128,
)
```

Admission control belongs at the gateway, not the engine, because the engine cannot reject what it has already accepted. A minimal token-bucket in front of the server:

```python
import asyncio, time
from fastapi import FastAPI, HTTPException

app = FastAPI()
# Admit at most `capacity` in-flight requests; reject the rest fast.
_sem = asyncio.Semaphore(128)          # match engine max_num_seqs
_MAX_QUEUE_WAIT_S = 0.5                 # if we cannot admit in 500 ms, shed

@app.post("/v1/chat/completions")
async def chat(req: dict):
    deadline = time.monotonic() + _MAX_QUEUE_WAIT_S
    try:
        await asyncio.wait_for(
            _sem.acquire(), timeout=max(0.0, deadline - time.monotonic())
        )
    except asyncio.TimeoutError:
        # Fail fast so the caller can retry another replica.
        raise HTTPException(status_code=429, detail="server at capacity")
    try:
        return await forward_to_engine(req)   # your engine client
    finally:
        _sem.release()
```

The rule of thumb: set the gateway's admitted-concurrency limit at or slightly below the engine's `max_num_seqs`, and set a short queue-wait deadline so a burst is shed in half a second rather than absorbed into a five-second TTFT. Do not disaggregate prefill and decode, or add replicas, until you have confirmed chunked prefill and admission control cannot hold the SLO — those are heavier interventions for a problem two flags usually solve.

#### Worked example: the burst that cliffed at 90% utilization

The page from the intro: 03:12, TTFT p99 = 4.2s against a 1s SLO. The team's first instinct was to add replicas, because the dashboard showed GPU utilization at 90% and that felt saturated. Decompose the latency first. `vllm:request_queue_time_seconds` averages 3.9s while `vllm:time_to_first_token_seconds` minus queue time — the actual prefill compute — is 0.3s. So 93% of TTFT is waiting, not computing: this is a queue, confirmed. `vllm:num_requests_waiting` reads 180. Now the queueing curve explains the suddenness. Before the burst, arrivals sat at ~55 req/s against a service capacity of ~62 req/s, so $\rho \approx 0.89$ — already on the knee. A marketing push lifted arrivals to ~61 req/s, pushing $\rho$ to ~0.98, and the $\frac{\rho}{1-\rho}$ term jumped from about 8 to about 49 — a 6× blow-up in queue wait from a 10% traffic increase. That is the cliff.

The mitigation, applied live: turn on admission control at the gateway with the admitted-concurrency limit set to the engine's `max_num_seqs` and a 500 ms queue-wait deadline. Requests beyond capacity get a fast `429`, callers retry other replicas, and $\rho$ on each replica clamps back below the knee — p99 TTFT drops to 0.8s within a minute. The permanent fix, shipped that week: enable chunked prefill (so the occasional 20k-token prompt stops monopolizing prefill and inflating the tail) and add two replicas so the steady-state $\rho$ sits at 0.7 with headroom for the next burst. Adding replicas *was* part of the answer — but only after admission control stopped the bleeding, and sized to hold utilization off the knee rather than to chase 95%.

## 5. Incident: TPOT creep

**Symptom.** Time-per-output-token — the inter-token latency during streaming — slowly rises over hours or days. Nothing crashes; the stream just gets choppier, and eventually the p99 TPOT alert fires. A restart makes it briefly better.

**How to confirm.** Distinguish TPOT (steady-state decode speed) from TTFT (the first-token latency of Section 4); they have different causes. Read `vllm:time_per_output_token_seconds` and correlate it with `vllm:num_requests_running` and `vllm:gpu_cache_usage_perc` over time. Two patterns separate the causes: if TPOT tracks the running batch size — high when the batch is large — the decode is bandwidth-bound and the batch is simply too big for your latency target. If TPOT rises even at *constant* batch size, and a restart fixes it, you are looking at memory fragmentation or a slow leak in KV block reuse.

**Root cause.** Decode is memory-bandwidth-bound: each decode step reads the entire KV cache for every sequence in the batch, so per-token latency grows with the total resident KV, which grows with batch size and context length. A batch tuned for throughput can blow the TPOT SLO because throughput and inter-token latency are the two ends of the batching trade-off. The fragmentation variant is subtler: over a long uptime, KV block allocation and free patterns fragment the paged cache, so the effective usable KV shrinks and the scheduler runs smaller, less efficient batches — the restart "fixes" it by resetting the allocator.

**Immediate mitigation.** If TPOT tracks batch size, lower `max_num_seqs` to pull inter-token latency back under SLO — you are explicitly trading some throughput for latency, which is the correct move when the latency corner of the triangle is the one breaching. If it is fragmentation, a rolling restart buys time.

**Permanent fix.** For the batch-size case, set `max_num_seqs` from your TPOT SLO, not from your throughput target, and let horizontal scaling carry throughput. Paged attention (vLLM's block-based KV manager) already reduces fragmentation to near zero by design, so genuine fragmentation creep usually points at an older serving stack or a custom KV manager; upgrading to a paged-attention engine is the durable fix. If you are on vLLM already and still see creep, check whether prefix caching is accumulating cached blocks it never evicts and bound the cache. The relationship between batch size, TPOT, and throughput is the core batching trade-off; the mechanics live in the batching post of this series, and the KV-side of the story continues in [KV cache optimization](/blog/machine-learning/model-serving/kv-cache-optimization).

The mechanics are worth making explicit because they tell you the shape of the trade-off you are managing. Decode reads the entire model — all $M_{\text{weights}}$ — and the entire resident KV cache once per generated token, so the time per token is bounded below by (weights + KV bytes) divided by HBM bandwidth. On an H100 with ~3.35 TB/s of HBM bandwidth, reading 18 GB of AWQ weights alone sets a floor of roughly ${18 / 3350 \approx 5.4}$ ms per token *for a single sequence*, before any KV. Batching amortizes the weight read across every sequence in the batch — that is why throughput rises with batch size — but each sequence's KV must still be read individually, so as the batch grows, the KV-read term grows and TPOT rises even though tokens-per-second-across-the-batch keeps climbing. That is the trade-off in one sentence: **larger batches convert latency into throughput at a bandwidth-fixed exchange rate.** When TPOT creeps up, you have drifted too far toward the throughput end; pulling `max_num_seqs` down moves you back, and the tokens-per-second you give up is recovered by adding a replica. There is no config that gives you both low TPOT and high per-replica throughput at once — the bandwidth is the constraint, and the only real escape is a faster memory system, a smaller KV (fp8, fewer heads via GQA), or fewer weight bytes (quantization).

#### Worked example: TPOT creep after a max_num_seqs bump

The change that caused it: a well-meaning throughput optimization raised `max_num_seqs` from 64 to 192 to push per-replica throughput, and it worked — aggregate tokens/second rose about 40%. Two days later the TPOT alert fires at p99 = 95 ms against an 80 ms SLO. Read the metrics: TPOT tracks the running batch size tightly, high when the batch is full. This is the batch-too-large case, not fragmentation (a restart would not help, and indeed does not). Do the bandwidth arithmetic: at batch 192 with 2k-context sequences in fp8 KV, the per-step read is 18 GB weights plus ${192 \cdot 2000 \cdot 128\text{ KB} \approx 49}$ GB of KV, so ${(18 + 49) / 3350 \approx 20}$ ms of pure bandwidth per token, and with kernel and scheduling overhead the observed 95 ms is consistent. Dropping `max_num_seqs` to 96 halves the KV-read term, pulling TPOT back to ~62 ms — comfortably under SLO — at the cost of ~20% per-replica throughput, which is restored by scaling out. The postmortem line: `max_num_seqs` is a *latency* knob dressed up as a throughput knob, and it must be sized against the TPOT SLO, not the throughput target.

## 6. Incident: throughput far below expectation

**Symptom.** The benchmark said 3,000 tokens/second per H100; production does 700. No errors, no OOM, latency is fine — the GPU is just quietly underperforming, and the cost-per-token is three times what the capacity plan assumed.

**How to confirm.** This is a diagnosis by process of elimination, and each cause has a distinct utilization signature. Watch `nvidia-smi dmon` or the DCGM exporter for GPU utilization and SM occupancy while load-testing, and read the engine's startup log and batch metrics. The four common causes and their fingerprints are laid out in Figure 6, which is the fastest way to route the diagnosis.

![A matrix mapping four throughput-killing misconfigurations to their utilization signatures and one-line fixes](/imgs/blogs/troubleshooting-llm-serving-runbook-6.webp)

**Root cause.** In rough order of how often they bite:

- **Eager mode.** Someone set `enforce_eager=True` (often to debug an unrelated error) and never removed it, so decode runs without CUDA graphs — every step pays kernel-launch overhead. GPU utilization looks moderate but decode is 3–4× slower than it should be. The tell is `enforce_eager=True` in the startup log.
- **No continuous batching.** The server processes one sequence at a time, or uses static batching that waits for the whole batch to finish before admitting new work. Throughput is flat regardless of offered concurrency, and the running batch size sits at 1. The fix is to use an engine that does iteration-level continuous batching (vLLM, TGI) rather than request-at-a-time serving.
- **Starved KV from a low utilization cap.** `gpu_memory_utilization` was set conservatively (say 0.5) to be "safe," leaving so little KV that only a handful of sequences fit. Concurrency is capped by memory you deliberately left on the table. Raise the cap.
- **CPU bottleneck.** Tokenization, detokenization, or sampling runs single-threaded on the request hot path, so the GPU sits idle waiting for the CPU. The signature is damning and unmistakable: GPU utilization low, one CPU core pinned at 100%. Batch and/or offload the CPU work.

**Immediate mitigation.** The eager-mode and utilization-cap cases are one-flag fixes and safe to apply live via a rolling restart. The CPU-bottleneck case needs the tokenizer moved off the hot path, which is a code change, so the interim mitigation is to add replicas to spread the CPU work — expensive, but it holds the SLO until the real fix ships.

**Permanent fix.** Beyond the per-cause flags, standardize a health-and-triage script so the next person confirms the cause in thirty seconds instead of thirty minutes. This is the script I keep in every serving repo:

```bash
#!/usr/bin/env bash
# oncall-triage.sh — the first 30 seconds of any LLM-serving page.
set -euo pipefail
ENDPOINT="${1:-localhost:8000}"

echo "== GPU memory + utilization (per device) =="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu \
           --format=csv,noheader,nounits

echo "== Per-process GPU memory (who is on the card?) =="
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
           --format=csv,noheader

echo "== CPU: is one core pinned? (tokenizer on the hot path) =="
top -b -n1 | head -12

echo "== Engine metrics: queue vs compute vs cache vs preemptions =="
curl -s "http://${ENDPOINT}/metrics" | grep -E \
  'vllm:num_requests_(running|waiting)|vllm:request_queue_time_seconds_sum|vllm:time_to_first_token_seconds_sum|vllm:gpu_cache_usage_perc|vllm:num_preemptions_total'

echo "== Startup flags that silently kill throughput =="
# Adjust to your log location / journalctl unit.
grep -E 'enforce_eager|gpu_memory_utilization|max_num_seqs|enable_chunked_prefill|Chunked' \
     /var/log/vllm/server.log | tail -20 || true
```

Run it first, always. It reads every signal this runbook keys on — memory per device, per-process attribution, CPU saturation, the queue-versus-compute split, cache usage, preemptions, and the throughput-killing flags — in one pass. Most of the time the answer is visible before you have finished reading the output. For turning these ad-hoc scrapes into standing dashboards and traces, see [tracing and debugging LLM serving](/blog/machine-learning/model-serving/tracing-and-debugging-llm-serving).

The CPU-bottleneck case deserves a word on the actual fix, because it is the one that resists a flag. If `top` shows one core pinned while the GPU idles, the tokenizer or a per-request Python callback is on the critical path. The three durable fixes, in order of preference: use the fast Rust-backed tokenizers (`use_fast=True`, the default for most HuggingFace models) rather than the pure-Python ones, which are often 5–10× slower; move tokenization off the request handler thread so a slow tokenize does not block the event loop; and, if you own the client, tokenize in a batch upstream and send token IDs to the server rather than raw strings. A minimal offload of the blocking tokenizer call out of an async handler:

```python
import asyncio
from functools import partial
from transformers import AutoTokenizer

_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct", use_fast=True)
_loop = asyncio.get_event_loop()

async def tokenize_async(text: str) -> list[int]:
    # Run the CPU-bound tokenize in the default thread pool so it never
    # blocks the async event loop that is streaming tokens to other clients.
    return await _loop.run_in_executor(
        None, partial(_tok.encode, text, add_special_tokens=False)
    )
```

The signature that this fixed it is unambiguous on the next load test: the pinned CPU core relaxes and GPU utilization climbs toward its real ceiling. If it does not, the bottleneck is elsewhere on the host path — detokenization, JSON serialization of a huge response, or a synchronous logging call — and the triage script's `top` output is where you keep looking.

#### Worked example: the 4× throughput miss

The page is not a page — it is a capacity review. Finance asks why the LLM bill is 4× the plan. The plan assumed 3,000 tokens/second per H100 from the vLLM benchmark; production logs show a steady 750. Run the triage script. GPU utilization reads 45%, no CPU core is pinned, and cache usage is a healthy 70% — so it is not CPU-bound and KV is not starved. The startup-flags grep prints `enforce_eager=True`. There it is: an engineer had set it three weeks ago to get a readable traceback for an unrelated tokenizer bug, fixed the bug, and left the flag. Without CUDA graphs, every decode step pays ~50 µs of kernel-launch overhead across dozens of kernels, and at the short per-step compute of decode that overhead dominates — the measured 3–4× decode slowdown that turns 3,000 into ~750. The fix is deleting one flag and a rolling restart; throughput returns to ~2,900 tokens/second and the bill drops by nearly 4×. The lesson worth writing into the postmortem: `enforce_eager` and `gpu_memory_utilization` are the two flags most likely to quietly cost you money, so grep for them on every performance investigation before you touch anything else.

## 7. Incident: multi-GPU and multi-node NCCL hang

**Symptom.** A tensor-parallel or multi-node deployment hangs. No crash, no error, no output — the process is alive, one or more GPUs read 100% utilization on `nvidia-smi`, and nothing progresses. The request that triggered it never returns. This is the scariest page because there is no traceback to grep.

**How to confirm.** A silent multi-GPU hang is almost always a stuck collective — an `all-reduce` or `all-gather` where one rank is waiting for peers that will never arrive. The single most useful action is to relaunch with NCCL tracing so the collective library tells you what it is doing:

```bash
# Every rank writes its own trace; INIT/NET/GRAPH cover transport + topology.
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH \
NCCL_ASYNC_ERROR_HANDLING=1 \
TORCH_NCCL_BLOCKING_WAIT=1 \
vllm serve Qwen/Qwen2.5-72B-Instruct \
  --tensor-parallel-size 8 \
  2>&1 | tee /var/log/vllm/nccl-rank.log
```

Then read the first hundred lines of the log. Three signatures cover the overwhelming majority of hangs, and Figure 7 is the flow from the hang to the matching fix.

![A branching graph from a silent NCCL collective hang through NCCL_DEBUG=INFO to three transport or rank causes, each with its targeted fix, converging on a completed all-reduce](/imgs/blogs/troubleshooting-llm-serving-runbook-7.webp)

**Root cause.** The three signatures, and what each looks like in the log:

- **InfiniBand not detected, socket fallback.** NCCL could not find or open the IB devices and silently fell back to TCP sockets over Ethernet — often 10–40× slower, and on a misconfigured network it can stall entirely. The log shows `NET/Socket` where you expected `NET/IB`, and lines like `NCCL INFO NET/IB : No device found.` A cross-node all-reduce that should take milliseconds takes seconds or hangs.
- **Topology mismatch.** Ranks disagree about the communication ring or tree — usually a mismatched `world_size`, a wrong `RANK`/`LOCAL_RANK`, or GPUs visible in a different order on different nodes. The log shows one rank building a different ring than its peers, and the collective never forms.
- **A dead or stuck rank.** One rank hit an OOM, a segfault, or its own error, and died or wedged; the surviving ranks entered the collective and now block forever waiting for it. The tell is that rank's log ends abruptly (or shows its own OOM) while the others sit inside `ncclAllReduce`.

**Immediate mitigation.** For the socket-fallback case, force IB explicitly and restart. For a dead rank, the collective cannot recover — kill and restart the whole job (a stuck collective does not un-stick itself). For a suspected topology mismatch, verify the rank environment variables and GPU ordering match across nodes before relaunching.

**Permanent fix.** Pin the network transport and topology so NCCL never guesses wrong:

```bash
# Force InfiniBand and name the HCAs so there is no silent socket fallback.
export NCCL_NET=IB
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0        # control-plane iface, not the data path
export NCCL_ALGO=Ring                 # pin the algorithm; avoids ring/tree drift
# Fail fast instead of hanging forever on a wedged peer.
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576   # flight recorder for postmortems
```

Reading the trace is a skill worth practicing before you need it. In the first hundred lines, three fields tell you almost everything. The transport line — `NCCL INFO NET/IB : Using [0]mlx5_0:1/IB` versus `NCCL INFO NET/Socket : Using [0]eth0` — tells you whether you got InfiniBand or fell back to sockets; anything but `NET/IB` on a cluster that has IB is the socket-fallback bug. The ring-construction lines — `NCCL INFO Channel 00/0 : 0 1 2 3 4 5 6 7` — should show a consistent ring across all ranks; if one rank prints a different channel membership or a different count, that is a topology mismatch and the collective will never form. And each rank's final lines tell you liveness: a healthy rank enters the collective and proceeds, while a dead rank's log ends mid-init or shows its own `CUDA error` / OOM before the survivors wedge. Grepping all rank logs for the last timestamp and diffing the channel lines usually pins the cause in under a minute. The two settings that convert a hang into a *diagnosable* failure are `NCCL_ASYNC_ERROR_HANDLING` and `TORCH_NCCL_BLOCKING_WAIT` — with them, a wedged rank surfaces an error and tears down the job instead of hanging silently, which is the difference between a five-minute restart and a two-hour mystery. Set `NCCL_IB_HCA` to your actual HCA names (list them with `ibstat`), and make GPU ordering deterministic across nodes with `CUDA_VISIBLE_DEVICES` so no two ranks disagree about the topology. Multi-node serving has an entire failure surface of its own — sharding strategy, P2P transfer, cross-node bandwidth — and it is the subject of [multi-node LLM serving for 100B-plus models](/blog/machine-learning/model-serving/multi-node-llm-serving-100b-plus).

#### Worked example: the socket-fallback hang

Two-node, 8-way tensor-parallel Qwen2.5-72B. Single-node testing passed; the two-node deployment hangs on the first request with all 16 GPUs pinned at 100% and no output for 30 minutes. Relaunch with `NCCL_DEBUG=INFO`. Rank 0's log shows `NCCL INFO NET/IB : No device found.` followed by `NCCL INFO NET/Socket : Using [0]eth0`. That is the whole diagnosis: the container did not have the IB devices mapped in, so NCCL fell back to sockets over the control-plane Ethernet, and a large cross-node all-reduce over that path stalled. The immediate fix is to map the IB devices into the container and set `NCCL_NET=IB` with the HCA names; the collective completes in milliseconds and the request returns. The permanent fix is the environment block above baked into the deployment plus a startup assertion that fails loudly if `NET/IB` is not selected — because a silent 40× slowdown that happens to *not* hang is even harder to catch than one that does.

## 8. Incident: long-prompt OOM, garbage output, slow cold start, and health-check flapping

The last four incidents are quieter than the loud ones above, and they share a property worth stating: none of them is really a GPU problem. Their root causes are a length budget, a chat template, a warmup cost, and a probe timeout — configuration and lifecycle, not compute. Figure 8 collects them.

![A matrix of four output and lifecycle incidents mapping each symptom to its non-GPU root cause and its config or probe fix](/imgs/blogs/troubleshooting-llm-serving-runbook-8.webp)

**OOM only on long prompts.** The service is fine for short requests and OOMs specifically when a prompt is long. This is the KV equation from Section 1 applied to one request: $b_{\text{KV}}$ times a long context can exceed what a single sequence's KV plus its activation spike can hold, even when average traffic fits. Confirm by correlating the crash with input length; the fix is to lower `max_model_len` to a value your product actually needs (rejecting over-long prompts at the gateway with a clear error), enable chunked prefill to bound the activation spike, and switch to fp8 KV to halve the per-token cost. If long contexts are core to the product, the honest answer is that this model on this card needs quantization or more GPUs — the budget does not lie.

**Garbage or repeated output.** The model emits repetition loops, ignores its stop tokens, or produces fluent nonsense — and crucially, it did so *without any infrastructure change*. This is almost never a serving bug and almost always a prompt-formatting or sampling bug. The two usual causes: the request bypassed the model's **chat template** (feeding raw user text where the model expects `<|im_start|>`-delimited turns), so the model has no idea it is in a conversation; or the **sampling parameters** are pathological (temperature near zero can lock a model into a repetition loop, temperature too high produces gibberish, missing `stop` tokens let it run past the turn boundary). Confirm by logging the exact string sent to the engine and diffing it against `tokenizer.apply_chat_template`. The fix:

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")
messages = [{"role": "user", "content": "Explain the KV cache in one paragraph."}]

# WRONG: raw text, no template -> model rambles, ignores turn structure.
# prompt = "Explain the KV cache in one paragraph."

# RIGHT: apply the model's own chat template with a generation prompt.
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

params = SamplingParams(
    temperature=0.7,           # not 0.0 (loop risk) or 1.5 (gibberish)
    top_p=0.9,
    repetition_penalty=1.05,   # gently damp loops
    max_tokens=512,
    stop=["<|im_end|>", "<|endoftext|>"],  # honor the model's real stop tokens
)
out = LLM(model="Qwen/Qwen2.5-32B-Instruct").generate(prompt, params)
```

When you serve through the OpenAI-compatible endpoint, vLLM applies the chat template for you *if you post to `/v1/chat/completions` with a `messages` array*; the classic bug is posting pre-formatted text to `/v1/completions` and double-applying (or skipping) the template. Match the endpoint to whether you are sending messages or a raw prompt.

**Slow cold start.** New pods take 90–180 seconds to become ready, so autoscaling reacts too slowly and scale-from-zero is unusable during a burst. This is not a bug; it is the real cost of loading tens of gigabytes of weights from storage into HBM plus capturing CUDA graphs at startup. Confirm by timing from container start to the first successful `/health`. The mitigations are structural: keep a small warm pool so you never scale from zero under load, pre-bake weights into the image or a fast local cache (loading 64 GB from remote object storage is often the dominant term), and — critically — set the Kubernetes readiness probe's `initialDelaySeconds` above the true startup time so the pod is not declared unhealthy before it has finished loading.

**Health-check flapping.** Pods cycle between Ready and NotReady, triggering restart loops and thrashing the fleet. The classic cause is a readiness probe whose `timeoutSeconds` is shorter than the server's response latency under load — or, worse, a probe that hits a *generation* endpoint, so the health check itself competes for the GPU and times out exactly when the server is busiest. Confirm by checking whether flapping correlates with load and what path the probe hits. The fix is a cheap, dedicated liveness path and honest timeouts:

```yaml
readinessProbe:
  httpGet:
    path: /health          # a cheap check, NOT /generate or /v1/chat/...
    port: 8000
  initialDelaySeconds: 180  # cover weight load + CUDA graph capture
  periodSeconds: 10
  timeoutSeconds: 5         # > p99 of /health, decoupled from generation latency
  failureThreshold: 3       # tolerate transient blips before cycling the pod
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 200  # strictly longer than readiness
  periodSeconds: 20
  timeoutSeconds: 5
  failureThreshold: 5       # liveness must be very slow to fire; it kills the pod
```

The principle: readiness gates traffic and can be moderately strict; liveness kills the pod and must be very forgiving, or a momentary load spike will restart healthy replicas at the exact moment you need them most. Never point either probe at an endpoint that does real inference work.

## The master incident triage table

When you are on-call, you do not read a runbook top to bottom — you jump to the row that matches your symptom. This is that index. It is the single artifact worth pinning above your desk, because it collapses everything above into symptom → signal → cause → fix.

| Incident | Symptom | Confirming signal | Root cause | Fix |
|---|---|---|---|---|
| Startup OOM | Dies during load/profiling | `nvidia-smi` full before first request | Weights + profiling exceed budget | Lower `gpu_memory_utilization`, `max_model_len`; quantize or shard |
| Load OOM | OOMs minutes into a ramp | Crash tracks `gpu_cache_usage_perc` | KV grew past budget | Cap `max_num_seqs`; fp8 KV |
| Preemption thrash | Throughput collapses 3–5×, no crash | `num_preemptions_total` climbing, cache 100% | More sequences than KV holds | Cap `max_num_seqs` to token budget; quantize |
| TTFT cliff | First-token latency spikes on burst | `request_queue_time` ≫ compute; deep waiting queue | Queue backlog + prefill head-of-line | Chunked prefill + admission control |
| TPOT creep | Inter-token latency rises over time | TPOT tracks batch size, or restart helps | Batch too large, or fragmentation | Lower `max_num_seqs` to TPOT SLO; paged attention |
| Low throughput | Far below benchmark, no errors | GPU util low; `enforce_eager` in log | Eager mode / no batching / starved KV / CPU-bound | Remove `enforce_eager`; continuous batching; raise util; batch tokenize |
| NCCL hang | Silent multi-GPU stall, 100% util | `NET/Socket` not `NET/IB`, or a dead rank | IB fallback / topology / dead rank | Pin `NCCL_NET=IB`, HCAs; async error handling; restart |
| Long-prompt OOM | Fine short, OOMs at long context | Crash correlates with input length | `max_model_len` × KV exceeds budget | Lower `max_model_len`; fp8 KV; chunked prefill |
| Garbage output | Repeats, ignores stops, no infra change | Sent string ≠ chat template | Missing template / bad sampling | Apply chat template; set `stop`, `temperature`, `repetition_penalty` |
| Slow cold start | 90–180s to Ready | Time from container start to `/health` | Weight load + graph capture | Warm pool; bake weights; raise `initialDelaySeconds` |
| Health flapping | Ready/NotReady cycling | Flap correlates with load; probe hits `/generate` | Probe timeout < latency; probe does inference | Cheap `/health` path; honest timeouts; `failureThreshold` |

The second table below is the one I reach for during a capacity or config review rather than an active page — it is the trade-off view of the same knobs, so you can see what you are spending when you turn each one.

| Config knob | Turn it up to... | Cost of turning it up | Turn it down to... |
|---|---|---|---|
| `gpu_memory_utilization` | Fit more KV, raise concurrency | Less headroom → startup/activation OOM risk | Avoid OOM on a shared or tight card |
| `max_num_seqs` | Raise throughput | Preemption thrash, higher TPOT | Protect TPOT SLO, stop thrash |
| `max_model_len` | Support long context | Bigger per-seq KV → long-prompt OOM | Free KV for more concurrency |
| `max_num_batched_tokens` | Faster prefill of long prompts | Bigger activation spike, TTFT jitter | Smoother decode via chunked prefill |
| `kv_cache_dtype=fp8` | Double token budget | Small accuracy hit on some models | Keep fp16 KV for max fidelity |

## Measurement: what each fix buys on an H100 80GB

A runbook that only names fixes is half a runbook; the other half is knowing what each fix is worth so you can prioritize under pressure. The table below is a before-and-after for the major incidents on a single H100 80GB serving Qwen2.5-32B, at an average context of 2k tokens. The numbers are representative of what these configuration changes deliver in practice — anchor points for the arithmetic, not a claim about your exact model and traffic. Measure your own; the *direction and rough magnitude* is what transfers.

| Incident and fix | Throughput (tok/s) | TTFT p99 (ms) | TPOT p99 (ms) | Max concurrency | GPU util |
|---|---|---|---|---|---|
| OOM (fp16, naive) | crashes | — | — | 0 | — |
| → AWQ 4-bit + fp8 KV | 2,900 | 320 | 22 | ~190 | 94% |
| Preemption thrash (max_num_seqs 256) | 400 | 5,200 | 140 | 256 (thrashing) | 96% |
| → cap max_num_seqs to budget | 2,100 | 480 | 34 | 128 | 92% |
| TTFT cliff (burst, no admission) | 2,050 | 4,200 | 30 | uncapped | 90% |
| → chunked prefill + admission | 2,050 | 780 | 31 | 128 | 88% |
| TPOT creep (max_num_seqs 192) | 2,850 | 410 | 95 | 192 | 95% |
| → max_num_seqs 96 (TPOT SLO) | 2,300 | 360 | 62 | 96 | 90% |
| Low throughput (enforce_eager) | 750 | 340 | 78 | 128 | 45% |
| → drop enforce_eager (CUDA graphs) | 2,900 | 320 | 22 | 128 | 94% |

Read the table as a priorities list. The two changes with the largest absolute payoff are removing `enforce_eager` (a 3.9× throughput swing from a single deleted flag) and quantizing to escape OOM (the difference between a crashing service and a 190-concurrency one). The preemption-thrash fix trades peak concurrency you could not actually use (256 sequences thrashing) for a 5× throughput recovery. The TTFT and TPOT fixes barely move throughput but pull the tail latencies back under SLO — they are latency insurance, not capacity gains. When you are triaging with limited time, this ordering is the guide: chase the flag-level throughput swings first, then the OOM budget, then the latency tails.

Two measurement caveats worth stating so you do not over-read the numbers. First, GPU utilization is a *misleading* health signal in isolation — the preemption-thrash row sits at 96% while delivering one-fifth the useful throughput, because the GPU is busy recomputing evicted prefills. Always pair utilization with a *useful-work* metric (tokens/second, `gpu_cache_usage_perc`, `num_preemptions_total`) before concluding the card is healthy. Second, TTFT and TPOT are tail metrics; report p99, not the mean, because the incidents in this runbook live almost entirely in the tail — the mean can look fine while the p99 breaches, which is exactly the report that generates a 3am page.

## Case studies

These are drawn from public benchmarks, papers, and the shape of incidents common enough to be representative rather than proprietary. Where a specific number comes from a source, treat it as that source's measurement; where it is a round figure, treat it as illustrative of the mechanism.

**PagedAttention and the fragmentation that used to OOM you.** The vLLM paper (Kwon et al., SOSP 2023) frames the KV-cache problem exactly as an OOM-and-waste problem: pre-paged, contiguous KV allocation wasted 60–80% of KV memory to internal and external fragmentation and over-reservation, which is precisely the "starved KV" and "fragmentation creep" failure modes in Sections 2 and 5. PagedAttention's block-based virtual paging cut that waste to under 4% and raised throughput several-fold on the same hardware. The practical takeaway for a runbook: if you are on a serving stack *without* paged attention and you see KV OOMs or throughput creep on long uptimes, the durable fix is not tuning — it is moving to a paged-attention engine, because the memory model itself is the bug.

**Continuous batching and the throughput a benchmark promised.** The Orca work (Yu et al., OSDI 2022) and the continuous-batching results that TGI and vLLM later shipped repeatedly show an order-of-magnitude throughput gap between request-at-a-time or static batching and iteration-level continuous batching on identical GPUs. That gap is the "low throughput, running batch size 1" row of Figure 6 in the field: teams benchmark a continuous-batching engine, then accidentally deploy behind a router or wrapper that serializes requests, and wonder where the 10× went. The confirming signal is always the same — the running batch size in the metrics sits at 1 regardless of offered concurrency.

**The NCCL socket fallback that looked like a bandwidth problem.** Silent InfiniBand-to-socket fallback is common enough in multi-node LLM deployments to be a first-check rather than a last resort. The failure is insidious because it does not always hang — sometimes it just makes cross-node collectives 10–40× slower, which surfaces as a throughput or TTFT problem three layers away from its cause. The fix that repeatedly resolves it is the same explicit-transport pinning in Section 7 (`NCCL_NET=IB`, named HCAs) plus a startup assertion that the IB path was actually selected. Teams that add that one assertion stop being paged for "mysterious multi-node slowness."

**FP8 KV cache as the concurrency multiplier.** H100-class hardware with native fp8 lets you store the KV cache in 8-bit, halving $b_{\text{KV}}$, and public vLLM and TensorRT-LLM results show this roughly doubling the token budget with negligible quality loss on most models. In budget terms, it is the cheapest move on the memory equation — it does not touch the weights and it converts directly into either more concurrency or longer supported context. For a fleet OOMing under load, it is often the single highest-leverage flag, which is why it appears in nearly every fix column above.

## When to use this (and when not to)

A runbook is a tool for the incident, not a substitute for prevention, and a few of its moves are genuinely the wrong call in the wrong context. Be decisive about both.

**Use the fast mitigations — load shedding, capping concurrency, lowering utilization — the moment an SLO is breaching.** Their entire purpose is to trade a little capacity or peak concurrency for stability under a live incident. Shedding load with a `429` during a TTFT cliff feels wrong under a latency alert, but it is correct: protecting in-flight requests beats letting all of them time out.

**Do not reach for the heavy interventions first.** Prefill/decode disaggregation, adding replicas, multi-node sharding, and speculative decoding are powerful, but they are the wrong first move for most pages. If chunked prefill and admission control hold your TTFT SLO, disaggregation is complexity you do not need — as a rule, do not disaggregate below a few hundred QPS. If a single flag (`enforce_eager`, `gpu_memory_utilization`) explains a throughput miss, do not restructure the fleet. The failure mode of experienced engineers is over-fixing: they see an OOM and immediately reach for tensor parallelism when quantization plus a concurrency cap would have solved it on one card.

**Do not quantize reflexively if quality is the product.** fp8 KV and 4-bit weights are the highest-leverage OOM fixes, but they are not free on every model — some are more sensitive than others, and a quality regression in a customer-facing product can cost more than the GPU you saved. Measure the accuracy delta on your own evals before shipping quantization as a permanent fix, even when it is the obvious mitigation.

**Prevention beats the runbook every time.** The incidents above are overwhelmingly preventable with three habits. First, **do the budget arithmetic before you deploy** — compute $M_{\text{weights}}$, $b_{\text{KV}}$, and the token budget, and set `max_num_seqs`, `max_model_len`, and `gpu_memory_utilization` from those numbers rather than from round figures. Second, **export and alert on the leading indicators**, not just the lagging ones: `num_preemptions_total`, `gpu_cache_usage_perc`, and the queue-versus-compute latency split fire *before* the user-visible SLO does, giving you minutes of warning. Third, **assert your config at startup** — fail loudly if `enforce_eager` is on in production, if the IB path was not selected, or if the readiness probe points at a generation endpoint. Most 3am pages are a config that was wrong the moment it deployed and only surfaced under load; an assertion turns that into a failed deploy at 3pm instead.

## Key takeaways

- **Classify by symptom class first.** Every serving page is OOM, latency, throughput, or bad output. The class picks the confirming signal; the signal picks the cause. Do not guess before you classify.
- **Two equations triage almost everything.** The memory budget ($M_{\text{weights}} + M_{\text{KV}} + M_{\text{act}} + M_{\text{overhead}} \le C_{\text{GPU}}$) tells you whether it fits; Little's Law with a queue-versus-compute split tells you whether you are overloaded or slow.
- **OOM is a term that dominated.** Read `nvidia-smi`, decide which of the four terms is largest, and apply the matching fix — quantize weights, cap KV, or chunk prefill. Do not shard when quantization fits.
- **Throughput collapse without a crash is preemption thrash.** `num_preemptions_total` climbing with cache at 100% is the fingerprint. You admitted more sequences than KV holds; cap `max_num_seqs` to the token budget.
- **A TTFT cliff is a queue, not a slow model.** Confirm with queue time ≫ compute time. Fix with chunked prefill and admission control before you add replicas.
- **`enforce_eager` and a low `gpu_memory_utilization` are the silent money-losers.** Grep for them on every throughput investigation before you touch anything else.
- **A silent multi-GPU hang is a stuck collective.** `NCCL_DEBUG=INFO` names the cause in a hundred log lines; `NCCL_ASYNC_ERROR_HANDLING` and `TORCH_NCCL_BLOCKING_WAIT` turn a hang into a diagnosable failure.
- **Garbage output and flapping health checks are not GPU problems.** Check the chat template and sampling params for the former, and the probe path and timeouts for the latter.
- **Prevention is arithmetic plus assertions.** Size knobs from the budget, alert on leading indicators, and fail deploys loudly on bad config. The best 3am page is the one that failed at 3pm.

## Further reading

- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023) — the KV-fragmentation-as-OOM framing behind Sections 2 and 5.
- Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models" (OSDI 2022) — the origin of iteration-level continuous batching and the throughput gap in Section 6.
- vLLM documentation — `EngineArgs` reference (`gpu_memory_utilization`, `max_num_seqs`, `kv_cache_dtype`, `enable_chunked_prefill`), the Prometheus metrics list, and the production-serving guide.
- NVIDIA NCCL documentation — environment variables (`NCCL_DEBUG`, `NCCL_NET`, `NCCL_IB_HCA`) and the troubleshooting guide for hangs and transport selection.
- Within this series: [what is model serving](/blog/machine-learning/model-serving/what-is-model-serving) for the vocabulary and the SLO triangle; [KV cache optimization](/blog/machine-learning/model-serving/kv-cache-optimization) and [request scheduling and preemption](/blog/machine-learning/model-serving/request-scheduling-and-preemption) for the mechanics behind the OOM and thrash incidents; [tracing and debugging LLM serving](/blog/machine-learning/model-serving/tracing-and-debugging-llm-serving) for standing observability; and [multi-node LLM serving for 100B-plus models](/blog/machine-learning/model-serving/multi-node-llm-serving-100b-plus) for the distributed failure surface behind the NCCL section.

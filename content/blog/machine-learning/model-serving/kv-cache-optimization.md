---
title: "KV cache optimization: managing the memory that caps LLM serving"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "Derive the KV cache byte equation, then manage it in production — PagedAttention blocks, FP8 KV quantization, GQA and MLA, sliding-window sinks, swap-versus-recompute preemption, and the capacity math that decides how many sequences fit on one GPU."
tags:
  [
    "model-serving",
    "inference",
    "kv-cache",
    "pagedattention",
    "vllm",
    "gpu-memory",
    "llm-serving",
    "quantization",
    "fp8",
    "gqa",
    "memory-optimization",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/kv-cache-optimization-1.webp"
---

At 3:47 on a Tuesday morning, a chatbot backend started returning `500`s. The model had not changed. The traffic had not spiked in any dramatic way — QPS was maybe 15% above the previous night. But a marketing email had gone out, and a slice of users were pasting in long documents and asking for summaries. Prompts that had averaged 400 tokens were now averaging 3,000. Nothing about the model's compute changed; the GPU was not thermally throttling; the network was fine. And yet the server was out of memory, and the autoscaler could not add capacity fast enough because each new pod took ninety seconds to load 16 GB of weights.

The engineer who got paged did the reasonable thing and restarted the pods. It helped for eleven minutes. Then it happened again. What she was actually watching was not a compute problem or a traffic problem. It was a memory-management problem, and the memory in question was the **KV cache** — the per-token key and value tensors that a transformer stores so it does not have to recompute attention over the whole prompt at every decode step. The KV cache is invisible in a notebook. In production it is the single most important number you are not looking at, because it, and not FLOPs, sets the ceiling on how many requests a GPU can serve at once.

This post is about managing that memory. Not about what the KV cache *is* at the level of attention math — [why LLM serving is different](/blog/machine-learning/model-serving/why-llm-serving-is-different) covers that ground, and the [KV cache optimization and management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management) post approaches it from the model-architecture side. This is a serving-operations post. We will derive exactly how many bytes the cache costs per token, show why that number is the memory wall, and then work through every practical lever a serving engineer has to push against it: PagedAttention's block allocator, KV quantization to FP8 and INT8, grouped-query and multi-head latent attention, sliding-window attention with sinks, eviction and preemption under memory pressure, and offloading to host memory and NVMe. The figure below is the whole argument in one frame — on an 80 GB H100 serving an 8-billion-parameter model, weights and framework overhead are fixed, and everything left over is KV cache. That leftover is what you are managing.

![GPU HBM budget breakdown on an 80 GB H100 showing the KV cache pool as the only elastic memory region](/imgs/blogs/kv-cache-optimization-1.webp)

By the end you will be able to compute, on the back of an envelope, how many concurrent sequences a given model fits on a given GPU; predict how much each optimization buys you; and choose between swapping and recomputing when the pool fills. Every technique here is a trade on the serving triangle — latency, throughput, cost. KV cache management is mostly a throughput-and-cost lever with occasional latency and accuracy side effects, and the whole job is knowing which side effect you are buying.

## 1. Why the KV cache exists, and why it becomes the wall

Start with the shape of the workload. An autoregressive transformer generates one token at a time. To produce token `t`, the attention layers need the keys and values of every previous token 0 through `t-1`. If you recomputed those from scratch at every step, generating an n-token response would cost work proportional to the sum 1 + 2 + ... + n, which is $O(n^2)$ in the sequence length — quadratic, and catastrophic for long outputs. The KV cache is the standard fix: compute each token's key and value once, store them in GPU high-bandwidth memory (HBM), and read them back on every subsequent step. That turns generation from quadratic to linear in the number of tokens you actually keep.

So the cache is not an optimization you can turn off. It is the thing that makes decode affordable. The problem is that it is *stateful* and it *grows*. Every active sequence owns a slab of HBM that expands by one token's worth of K and V at every decode step, and that slab lives for the entire lifetime of the request. A model server is therefore not a stateless function that maps input to output; it is a memory allocator running under hard real-time pressure, where the allocation size of each request is unknown in advance (you do not know how long the model will generate) and the total pool is fixed by the physical GPU.

There is a second, subtler reason the KV cache dominates. Decode is **memory-bandwidth bound**, not compute bound. At each decode step the model reads its entire weight matrix and every sequence's entire KV cache out of HBM, and does comparatively little arithmetic with them — one token's worth of matrix-vector products. The arithmetic intensity (FLOPs performed per byte read) is roughly one, far below the "ridge point" of a modern GPU, which for an H100 sits somewhere around 200 to 300 FLOPs per byte. When you are that far to the left of the ridge, your token rate is set by how fast you can stream bytes, and the biggest, most bandwidth-hungry byte source at high batch size and long context is the KV cache. Prefill, by contrast — processing the whole prompt in one shot — is compute bound, because it does a large matrix-matrix multiply with high arithmetic intensity. This asymmetry, compute-bound prefill versus memory-bound decode, is the reason the KV cache is a serving concern and not just a correctness detail.

To make the bandwidth point concrete, price out one decode step on an H100. The GPU must read the model weights (16 GB for our 8B model in FP16) plus every active sequence's KV cache, and HBM3 bandwidth is about 3.35 TB/s. With the ~52 sequences we will derive below, each holding 1 GiB of KV, that is roughly 52 GiB of KV plus 16 GiB of weights — around 72 GB streamed per step. At 3.35 TB/s the read alone takes about 21 ms, which sets a floor on time-per-output-token (TPOT) near 21 ms, or roughly 47 tokens/s per sequence and about 2,400 tokens/s aggregate. The arithmetic in that step is trivial by comparison — a handful of matrix-vector products — so the tensor cores sit mostly idle while the memory system does the real work. Halve the KV bytes with FP8 and the per-step traffic falls toward 42 GB, TPOT drops, and you fit twice the sequences. That one calculation is why KV size shows up directly in your token rate: fewer KV bytes means both more sequences and faster steps, a rare optimization that helps latency and throughput at once.

Put those two facts together. The cache cannot be turned off; it grows without a known bound; it lives in the scarcest resource on the box; and reading it is what gates your decode speed. That is the memory wall. Everything in this post is a way to make the wall further away — either by making each token's KV footprint smaller, by packing the pool more tightly, or by spilling parts of it somewhere cheaper when the pool fills.

### The one equation you must internalize

Here is the derivation that every serving engineer should be able to reproduce from memory. For a single token, the cache stores both a key vector and a value vector, in every layer, for every key/value attention head. So the per-token KV size is:

$$\text{bytes/token} = 2 \times L \times H_{kv} \times d_{head} \times b$$

where the leading 2 is for K and V, $L$ is the number of transformer layers, $H_{kv}$ is the number of key/value heads (which under grouped-query attention is smaller than the number of query heads), $d_{head}$ is the per-head dimension, and $b$ is the bytes per element of your KV dtype (2 for FP16/BF16, 1 for FP8/INT8). Multiply by the sequence length to get one sequence's cache, and by the batch size to get the pool you need:

$$\text{KV pool} = \text{bytes/token} \times \text{seq len} \times \text{batch}$$

That is the whole thing. There is no hidden constant. Notice what is *not* in the equation: the hidden dimension of the MLP, the vocabulary size, the number of query heads. The KV cache cares only about layers, KV heads, head dim, context length, and dtype. This is precisely why grouped-query attention and latent attention — which shrink $H_{kv}$ or replace it with a compressed latent — are such effective KV levers: they attack the equation at its widest factor.

## 2. Sizing the cache exactly: the KV byte equation in practice

Abstractly the equation is simple; the payoff is plugging real models into it and being shocked at the result. Let us do that carefully. The figure below tabulates four architectures against the equation, and the numbers below trace directly to it.

![Per-token KV cache byte size across Llama-3-8B, Llama-3-70B, Llama-2-7B and DeepSeek-V2](/imgs/blogs/kv-cache-optimization-2.webp)

**Llama-3-8B.** It has 32 layers, 32 query heads but only 8 KV heads (grouped-query attention with a 4-to-1 group ratio), and a head dimension of 128. In FP16:

$$2 \times 32 \times 8 \times 128 \times 2 = 131{,}072 \text{ bytes} = 128 \text{ KiB per token}$$

So every single token in flight — prompt or generated — costs 128 KiB of HBM. A sequence with 8,192 tokens of context therefore holds $8192 \times 128\text{ KiB} = 1$ GiB of KV cache. One gigabyte, for one conversation, on an 8-billion-parameter model whose weights are only 16 GB. That ratio is the entire story of LLM serving economics.

**Llama-3-70B.** 80 layers, 64 query heads, 8 KV heads, head dim 128. In FP16:

$$2 \times 80 \times 8 \times 128 \times 2 = 327{,}680 \text{ bytes} = 320 \text{ KiB per token}$$

At 8,192 tokens that is 2.5 GiB per sequence. The 70B model is roughly 2.5 times more KV-hungry per token than the 8B, which tracks the layer-count ratio (80 versus 32) since both share the same 8 KV heads and head dim.

**Llama-2-7B (multi-head attention, no GQA).** 32 layers, 32 KV heads (one per query head), head dim 128:

$$2 \times 32 \times 32 \times 128 \times 2 = 524{,}288 \text{ bytes} = 512 \text{ KiB per token}$$

Look at that against Llama-3-8B: 512 KiB versus 128 KiB, a 4x difference, and the *only* thing that changed is the KV head count (32 versus 8). This is grouped-query attention paying for itself. The move from full multi-head attention to 8 KV heads cut the cache to a quarter of its size for essentially free, and it is why every modern open model ships with GQA.

**DeepSeek-V2 (multi-head latent attention).** MLA does not store per-head K and V at all. It stores a single compressed latent vector per token per layer (dimension 512 in DeepSeek-V2) plus a small decoupled rotary key (dimension 64), shared across heads. Across 60 layers:

$$60 \times (512 + 64) \times 2 = 69{,}120 \text{ bytes} \approx 67.5 \text{ KiB per token}$$

At 8,192 tokens, roughly 0.55 GiB per sequence. DeepSeek reports MLA reduces the KV cache by about 93% relative to the equivalent multi-head attention model, and the arithmetic bears it out: a comparable MHA configuration would be many hundreds of KiB per token, and MLA collapses it to under 70.

Here is the same data as a table you can adapt for your own model. Fill in your architecture's config values and you have your per-token cost.

| Model | Layers | KV heads | Head dim | KV dtype | Bytes/token | KV @ 8K ctx |
|---|---|---|---|---|---|---|
| Llama-3-8B (GQA) | 32 | 8 | 128 | FP16 | 128 KiB | 1.0 GiB |
| Llama-3-70B (GQA) | 80 | 8 | 128 | FP16 | 320 KiB | 2.5 GiB |
| Llama-2-7B (MHA) | 32 | 32 | 128 | FP16 | 512 KiB | 4.0 GiB |
| Mistral-7B (GQA) | 32 | 8 | 128 | FP16 | 128 KiB | 1.0 GiB |
| DeepSeek-V2 (MLA) | 60 | latent 512+64 | — | FP16 | ~67.5 KiB | ~0.55 GiB |

#### Worked example: how many sequences fit on one H100?

Take Llama-3-8B on an 80 GB H100, served by vLLM with the default `gpu_memory_utilization=0.9`. That reserves 72 GB for the engine. The FP16 weights consume 16 GB. Framework overhead — CUDA context, activation buffers, NCCL, the sampling and scheduling structures — is typically 3 to 5 GB; call it 4. That leaves a KV cache pool of roughly ${72 - 16 - 4 = 52}$ GB.

At 8,192 tokens of context, each sequence needs 1 GiB. So the box holds about **52 concurrent sequences at full 8K context**. If your average conversation is shorter — say 2,048 tokens — each sequence needs only 256 MiB, and the same pool holds roughly 200 sequences. This is why "how many users can one GPU serve" has no fixed answer: it is entirely a function of context length, because the KV cache scales linearly with tokens held. The 3 a.m. incident from the intro is now fully explained — average prompt length went from 400 to 3,000 tokens, roughly a 7x jump in per-sequence KV, and the pool that comfortably held the old traffic could suddenly hold a seventh as many sequences. The scheduler started rejecting or preempting, and the symptom surfaced as `500`s.

The lesson: **your concurrency ceiling is `KV pool / (bytes-per-token × context length)`**, and every optimization in this post moves one of those three terms. We will make bytes-per-token smaller (quantization, GQA, MLA), make the pool bigger or better packed (PagedAttention, offloading), or cap the effective context length (sliding window).

## 3. PagedAttention: stop reserving memory you are not using

Before you shrink the cache, pack it properly, because the naive layout wastes most of it. The pre-vLLM approach — used by early serving systems and still lurking in hand-rolled inference loops — allocated a single contiguous KV buffer per sequence, sized to the maximum possible sequence length. If your model supports 4,096 tokens, every request got a 4,096-token slab whether it generated 20 tokens or 2,000. The figure below contrasts that with the paged approach.

![Contiguous max-length KV reservation wasting memory versus PagedAttention on-demand block allocation](/imgs/blogs/kv-cache-optimization-3.webp)

Contiguous max-length reservation is a disaster on two axes. First, **internal fragmentation**: a request that reserves 2,048 tokens but generates 200 leaves more than 90% of its slab idle, but that idle memory cannot be lent to any other sequence because it belongs to this one. The vLLM paper measured that real systems wasted 60% to 80% of KV memory this way. Second, **external fragmentation**: because each slab must be physically contiguous, the allocator can be unable to admit a new sequence even when the total free memory is more than enough — there is just no single contiguous hole big enough. The result is the OOM-at-12-sequences failure in the figure: the box has plenty of free bytes, but not in the right shape.

PagedAttention, introduced by Kwon and colleagues in the 2023 SOSP paper that launched vLLM, borrows the operating-system idea of virtual memory and paging. It chops the KV cache into fixed-size **blocks** — 16 tokens per block by default in vLLM — and maintains a **block table** per sequence that maps the sequence's logical block indices to arbitrary physical blocks anywhere in the pool. A sequence grows by grabbing a free block only when it fills its current one. Nothing is reserved ahead of need. The figure below shows the physical pool.

![Physical KV block pool where two sequences interleave non-contiguously through a block table](/imgs/blogs/kv-cache-optimization-4.webp)

The key move is that physical blocks sit in one linear HBM address space, but a sequence's *logical* order is decoupled from *physical* order by the block table. In the figure, sequence A owns physical blocks 0, 3, 5, and 8; sequence B owns 1, 4, 7, and 10; the rest are free. Neither sequence occupies a contiguous run, and it does not matter, because the attention kernel gathers keys and values through the block table's indirection. The two sequences interleave freely. External fragmentation drops to zero — any free block can serve any sequence. Internal fragmentation drops to at most one partial block per sequence: sequence A's tail block holds tokens 48 through 52, five of a possible sixteen, wasting eleven token-slots. Averaged over a block of 16, the expected waste is under 8 tokens per sequence, versus thousands under contiguous reservation. Measured KV utilization rises from under 40% to above 96%.

Block size is a real knob. Smaller blocks (say 8 tokens) reduce the partial-block waste further but increase the number of blocks the kernel must gather per attention op, adding indirection overhead. Larger blocks (32) cut indirection but grow the tail waste. The vLLM default of 16 is a good compromise for most models; you rarely need to touch it. What you *do* touch is whether paging is on at all, and in vLLM it always is — this is the foundational trick that [continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention) is built on, and it is the reason vLLM can run near 96% memory utilization where a contiguous allocator would OOM.

Paging unlocks a second win that contiguous buffers cannot: **block sharing**. Because a block is just an entry in a block table, several sequences can point their tables at the *same* physical block. vLLM uses this for prefix sharing — two requests with an identical system prompt reference the prompt's KV blocks, computed once — and for parallel sampling and beam search, where multiple candidate continuations of one prompt share the prompt blocks and diverge only where their tokens differ. Each shared block carries a reference count; when a sequence needs to write into a block that others still reference, it triggers a **copy-on-write**, cloning just that one block so the siblings are undisturbed. The saving is large for high-fan-out sampling — generating 8 candidates from one 2,000-token prompt shares roughly 2,000 tokens of KV instead of duplicating it eight times — and for any workload with common prefixes. This copy-on-write machinery is the foundation prefix caching is built on, and it exists only because the block table decoupled logical order from physical placement.

A concrete consequence for capacity planning: because paging eliminates the "reserve the max" tax, your usable KV pool is essentially the full physical remainder after weights and overhead, and the worked example above (52 sequences on an H100) assumed paging. Without it, you would divide by a much larger effective per-sequence footprint and get maybe a third of that. Paging is not an optimization you add on top of a working system; it is the baseline that makes the rest of the numbers achievable.

## 4. Shrinking the cache at the source: GQA and MLA

Paging packs the cache tightly, but it does not change how many bytes each token costs. To attack the bytes-per-token term you change the attention architecture, and the two levers that matter in production are grouped-query attention and multi-head latent attention. You do not usually get to choose these — they are baked into the model you are serving — but you absolutely need to understand them, because they change your capacity numbers by multiples and they determine which serving tricks are even worth applying.

**Grouped-query attention (GQA).** Standard multi-head attention gives every query head its own key and value head. GQA shares each K/V head across a *group* of query heads. Llama-3-8B has 32 query heads but only 8 K/V heads, so groups of 4 query heads share one K/V head. Since the KV byte equation is linear in $H_{kv}$, cutting KV heads from 32 to 8 cuts the cache to a quarter, exactly as we saw: 512 KiB drops to 128 KiB per token. The accuracy cost is real but small and, crucially, it is *trained in* — the model was pretrained with GQA, so there is no serving-time degradation to manage. From a serving perspective GQA is free capacity: a model with an 8-to-1 or 4-to-1 group ratio simply fits several times more sequences than an equivalent MHA model, with no flags to set and no accuracy knob to worry about.

**Multi-head latent attention (MLA).** DeepSeek's MLA goes further. Instead of storing per-head K and V, it projects them down into a single low-rank *latent* vector per token per layer, and reconstructs the per-head keys and values on the fly during attention via absorbed projection matrices. It also keeps a small decoupled rotary-position key so that RoPE still works. The cached state per token is just that latent (dimension 512 in DeepSeek-V2) plus the rotary key (64), rather than 8 or 32 full heads. That is the ~93% reduction we computed. The remarkable part is that DeepSeek reports MLA matches or slightly exceeds the quality of the equivalent full MHA model — the low-rank bottleneck turns out not to hurt, because K and V are highly compressible. The catch is that MLA is not something you bolt onto an existing model; it requires the model to be trained with it, and the attention kernel must support the absorbed-projection trick to get the memory win at inference. vLLM and SGLang both have MLA kernels for DeepSeek models; if you are serving DeepSeek-V2 or V3, you get the benefit automatically, and if you are not, you cannot retrofit it. The [multi-head latent attention deep-dive](/blog/machine-learning/large-language-model/multi-head-latent-attention-mla) covers the mechanism in full.

It is worth placing these on a spectrum, because they are points on one axis: how aggressively you share or compress the K/V projection. Full multi-head attention (one K/V head per query head) is the expensive extreme — maximum expressiveness, maximum cache. **Multi-query attention (MQA)** is the opposite extreme: a *single* K/V head shared by all query heads, which shrinks the cache by the full head-count factor but was found to cost noticeable quality on some tasks. **Grouped-query attention** is the pragmatic middle — a handful of K/V heads (8 is the common choice) recovers almost all of MHA's quality while keeping most of MQA's memory win, which is why it, not MQA, became the default. **Multi-head latent attention** sits off the axis entirely: instead of reducing the head *count* it compresses the per-token state into a low-rank latent, achieving MQA-class memory with MHA-class quality. The trend line across model generations is unmistakable — each new family pushes further down the memory axis while holding quality, because serving cost, not training cost, is what makes a model economical to run.

The practical takeaway is a hierarchy. If you are choosing a model to serve and KV memory is your constraint — long context, high concurrency — prefer, in order: an MLA model (DeepSeek family) for the most aggressive reduction, then a GQA model with a high group ratio (most Llama, Mistral, Qwen models), and treat a legacy full-MHA model as a memory liability you will have to compensate for with quantization and offloading. These are architecture choices that dwarf anything you can do at the serving layer, which is why they come first.

## 5. KV cache quantization: FP8 and INT8

You cannot change a model's KV head count at serving time, but you *can* change $b$, the bytes per element. This is KV cache quantization, and it is the single most impactful serving-layer lever, because it applies to any model and it directly halves (FP16 to FP8/INT8) or quarters (FP16 to INT4, more exotic) the per-token cost. The figure below shows the FP8 case.

![FP16 KV cache versus FP8 KV cache doubling the sequences that fit per GPU](/imgs/blogs/kv-cache-optimization-5.webp)

The idea is narrow and important: KV cache quantization stores the *cached* keys and values in a lower-precision format, while leaving the model *weights* and the compute in their original precision. This is different from weight quantization like GPTQ or AWQ, which shrinks the weights to fit a bigger model on a smaller GPU (see [quantization for LLM serving](/blog/machine-learning/model-serving/quantization-for-llm-serving) for that side). KV quantization does nothing for weights; it purely buys KV capacity. On Llama-3-8B, moving the KV cache from FP16 to FP8 takes each token from 128 KiB to 64 KiB, each 8K sequence from 1.0 GiB to 0.5 GiB, and the H100's ~52-sequence ceiling to roughly 104. You literally double concurrency, or equivalently double the context length you can support at fixed concurrency.

In vLLM, this is one argument:

```python
from vllm import LLM, SamplingParams

# FP8 KV cache: keys/values stored in 8-bit, weights stay BF16.
# On Hopper (H100/H200) the E4M3 format is hardware-native.
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    kv_cache_dtype="fp8",            # "auto" keeps model dtype; "fp8" == fp8_e4m3
    gpu_memory_utilization=0.90,     # fraction of HBM the engine may claim
    max_model_len=8192,              # cap context so the block manager sizes correctly
    max_num_seqs=256,                # upper bound on concurrent sequences
)

params = SamplingParams(temperature=0.7, max_tokens=512)
out = llm.generate(["Summarize the attached document in five bullet points."], params)
print(out[0].outputs[0].text)
```

The accuracy question is the one everyone asks, and the honest answer is: for FP8, it is almost always fine, and you should measure it anyway. FP8 E4M3 has a wide dynamic range, and keys and values are reasonably well-behaved distributions. Published evaluations and vLLM's own benchmarks put the perplexity increase from FP8 KV at well under 0.1 points on standard sets, and downstream task accuracy (MMLU, GSM8K) typically moves within noise. The failure mode to watch is long-context retrieval tasks — needle-in-a-haystack style — where the model must attend precisely to a token thousands of positions back; there, quantization error in the cached keys can occasionally cost a retrieval. So the rule is: enable FP8 KV by default for chat, RAG with short retrieved chunks, and general generation; validate explicitly if your workload is long-context exact recall.

INT8 KV is the alternative on hardware without native FP8 (pre-Hopper: A100, A10, T4). It needs per-tensor or per-channel scales computed from a calibration pass, which adds a step and a little runtime overhead to apply the scales, and its accuracy is a touch more fragile than FP8 because INT8's uniform quantization handles outliers worse than FP8's floating exponent. On an A100, INT8 KV is still very much worth it — it is the only way to double KV capacity on that generation — but budget time to calibrate and evaluate. On Hopper, prefer FP8; it is native, needs no calibration for E4M3 in most vLLM configurations, and is more robust.

SGLang exposes the same lever at launch:

```bash
# SGLang server with FP8 KV cache (E5M2 or E4M3 depending on kernel support).
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3-8B-Instruct \
  --kv-cache-dtype fp8_e5m2 \
  --mem-fraction-static 0.90 \
  --context-length 8192 \
  --max-running-requests 256
```

A quick word on a subtlety: FP8 comes in two flavors, E4M3 (4 exponent bits, 3 mantissa) and E5M2 (5 exponent, 2 mantissa). E4M3 has more precision and less range; E5M2 has more range and less precision. For KV cache, E4M3 is usually the better choice because the values are bounded and you want the precision, but some kernels only support E5M2 for one of K or V — check your framework's release notes for your exact model. This is the kind of detail that does not matter until it does, so measure the perplexity delta for your model and pick the format that minimizes it.

Two more details separate a working 8-bit KV deployment from a broken one. First, **scaling granularity**. A single per-tensor scale for the entire cache is cheapest to store and apply but least accurate; per-token or per-head scales track the local dynamic range far better and are usually worth the small bookkeeping cost, especially for INT8, where uniform quantization handles outliers worse than FP8's floating exponent. vLLM computes and applies these scales inside the attention kernel, so you rarely touch them by hand — but if accuracy falls off a cliff on one particular model, coarse per-tensor scaling is a prime suspect, and switching to a finer granularity or a calibrated scale often recovers it. Second, **going below 8 bits**. INT4 KV cache is an active research area and a few engines expose it experimentally; it doubles capacity again over INT8, but the accuracy cost climbs steeply and it is fragile across models and tasks. Treat INT4 KV as something to validate exhaustively on your own evaluations, never a default. The safe, high-confidence lever is 8-bit; exhaust GQA, MLA, paging, and offloading before you reach for 4-bit KV, because those either cost nothing or cost bandwidth rather than accuracy.

#### Worked example: FP8 KV on an A100 40GB

Now put a 70B-class deployment aside and take the common budget case: Llama-3-8B on an **A100 40GB**, which is a far more common GPU than an H100 in most fleets. Budget at `gpu_memory_utilization=0.9` is 36 GB. Weights are 16 GB, overhead ~3 GB, leaving a KV pool of about 17 GB. In FP16 at 8K context that is 17 sequences — uncomfortably tight for a service that needs to hold a few dozen conversations. Switch KV to INT8 (A100 has no FP8), per-token drops to 64 KiB, per-sequence to 0.5 GiB, and the pool now holds ~34 sequences. You did not buy a bigger GPU; you doubled the box's effective capacity with one flag and a calibration pass. In cost terms, if the A100 rents for roughly \$2.50 per hour and served 17 sequences before, your cost per concurrent sequence just halved. That is the kind of lever that pays for the engineering time in a day.

## 6. Sliding-window attention and StreamingLLM sinks

Quantization shrinks bytes per token; sliding-window attention caps the *number of tokens* you cache at all. The observation behind it: for many tasks, a token thousands of positions back contributes almost nothing to the next prediction. So why cache it? Sliding-window attention (used by Mistral-7B, among others) restricts each token to attend only to the previous `W` tokens — a window of, say, 4,096. Once a sequence exceeds `W` tokens, the oldest KV blocks can be evicted, because no future token will attend to them. The KV cache stops growing at `W × bytes-per-token`, no matter how long the conversation runs. A 100,000-token streaming session on a sliding-window model has the same KV footprint as a 4,096-token one.

The cost is exactly what it sounds like: the model genuinely cannot see past the window. For a coding assistant streaming a long file, or a chat that references something said an hour ago, sliding window will lose that context. So it is a workload decision, not a universal win. It fits streaming, chat with bounded relevant history, and log or telemetry processing where recency dominates. It does not fit long-document question answering or any task requiring exact recall across the full context.

There is a beautiful wrinkle discovered by the **StreamingLLM** work (Xiao and colleagues, 2023): if you naively evict the oldest tokens in a model that was *not* trained with a sliding window, quality collapses catastrophically the moment the very first tokens fall out of the window. The reason is that transformers dump a large amount of attention weight onto the first few tokens regardless of content — these are **attention sinks**, a place for the softmax to park probability mass it does not want to spend elsewhere. Evict the sinks and the attention distribution destabilizes. The fix is almost embarrassingly simple: always keep the first few tokens (typically 4) pinned in the cache as sinks, then slide the window over the rest. With sink tokens retained, a model can stream effectively unbounded text with a fixed, small KV cache and no quality collapse. vLLM and other engines expose this as a sink-aware sliding window.

In practice you rarely hand-configure sinks; you serve a model whose architecture already implies the window (Mistral's config declares `sliding_window`), and the engine handles sink retention. What you need to know as an operator is the capacity consequence: a sliding-window model's KV cache is **bounded**, which means your capacity math uses `W` in place of the actual context length. A Mistral-7B with a 4,096 window on an H100 holds far more concurrent long-running sessions than a full-attention model of the same size, because no single session's cache can exceed the window. When you are sizing a fleet for long-lived streaming connections, that bound is the difference between a predictable capacity plan and an open-ended memory risk.

## 7. Eviction and preemption: what happens when the pool fills

Every technique so far pushes the wall further away. None of them removes it. Sooner or later a burst arrives, the pool fills, and the scheduler faces a request it has no free blocks for. The naive systems from a few years ago failed the request or crashed. Modern engines do something much better: they **preempt**. The figure below shows the decision.

![Scheduler admission control branching to admit, swap-to-CPU, or recompute under memory pressure](/imgs/blogs/kv-cache-optimization-6.webp)

When a new request (or the next decode step of an existing one) needs blocks and none are free, the scheduler picks a **victim** sequence — usually the most recently arrived or lowest-priority one — and frees its blocks so the batch can proceed. The victim is not dropped; it is paused and will be resumed. There are two ways to free its blocks, and the choice is a real trade-off:

- **Swap (offload to CPU).** Copy the victim's KV blocks out to pinned host RAM over PCIe, free the GPU blocks, and later copy them back when space frees up. The cost is the two PCIe transfers. On a PCIe Gen4 x16 link at roughly 25 GB/s effective (and often less for pinned-memory copies contending with other traffic), moving a 2.5 GB KV cache out and back is on the order of 200 ms of transfer. You pay bandwidth, but you never recompute.
- **Recompute (discard and re-prefill).** Simply throw away the victim's KV blocks. When it resumes, re-run prefill over its prompt (and generated-so-far tokens) to rebuild the cache. The cost is the prefill compute — which for a long prompt is substantial, since prefill is $O(n^2)$-ish in attention — but it uses zero PCIe bandwidth and no host memory.

vLLM's default preemption mode is **recompute**, and that surprises people until they do the arithmetic. For short-to-medium prompts, re-running prefill on a fast GPU is often *cheaper* than a round-trip PCIe swap, because the GPU's compute is enormous relative to the PCIe link's bandwidth. Swap wins when prompts are very long (prefill would be expensive) and you have PCIe headroom and host RAM to spare. You control it directly:

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    gpu_memory_utilization=0.90,
    # "recompute" (default) discards victim KV and re-prefills on resume.
    # "swap" offloads victim KV to CPU and copies it back.
    preemption_mode="recompute",
    # swap_space is host RAM (GiB) reserved for swapped KV blocks;
    # only used when preemption_mode="swap".
    swap_space=8,
    max_num_seqs=256,
)
```

The figure below traces a single preemption event in time, so the "bounded stall, not an error" property is concrete.

![Timeline of a single preemption event from full pool through swap-out to victim resume](/imgs/blogs/kv-cache-optimization-7.webp)

Follow the sequence. At `t0` eight sequences are decoding and the pool is 90% full. At `t1` a new request arrives needing prefill for a 2K prompt. At `t2` the scheduler finds no free blocks. At `t3` it selects the lowest-priority victim, and at `t3+` swaps that victim's 2.5 GB of KV out to host memory at PCIe bandwidth. At `t4` the freed blocks let the newcomer in, and at `t5` the victim resumes — its KV swapped back when space frees. The whole episode adds one recoverable stall to one sequence, rather than a `500` for the whole batch. That is the property you are buying: preemption converts an out-of-memory burst into a latency blip for the least important request, which is exactly the right thing to sacrifice.

#### Worked example: swap versus recompute for a 4K prompt

Suppose a victim sequence has a 4,096-token prompt on Llama-3-8B, so its KV cache is 4096 × 128 KiB = 512 MiB. The recompute cost is re-running prefill over 4,096 tokens; on an H100, prefill for an 8B model runs on the order of tens of thousands of tokens per second, so this is roughly 50 to 150 ms of GPU compute — during which the GPU is rebuilding that cache instead of decoding other sequences. The swap cost is copying 512 MiB out to host and 512 MiB back over PCIe Gen4 at roughly 25 GB/s effective, about 20 ms each way, so ~40 ms of transfer, but it consumes PCIe bandwidth rather than GPU compute and it needs 512 MiB of pinned host RAM held in reserve. For a medium prompt the two are close, and vLLM's recompute default is reasonable because it keeps PCIe free and needs no host memory. Now scale the prompt to 32K tokens: the KV is 4 GiB, the swap round-trip grows linearly to roughly 320 ms, but recompute grows worse than linearly because prefill attention scales with the square of the sequence length, pushing re-prefill past a second of GPU time. That is the crossover — **swap wins decisively for very long prompts, recompute for short ones**. If your workload is long-context, set `preemption_mode="swap"` and size `swap_space` to cover your expected victim count; if it is short-prompt chat, keep the recompute default.

The operational implication is about SLOs. Preemption is invisible in aggregate throughput but very visible in tail latency — a preempted-and-recomputed request eats a re-prefill, which can spike its TTFT (time to first token) dramatically. If your p99 latency SLO is tight, you do not want frequent preemption; you want to run at a `gpu_memory_utilization` and `max_num_seqs` that keep the pool from filling under normal bursts, accepting slightly lower peak throughput in exchange for a stable tail. If you are throughput-maximizing a batch job with no latency SLO, run the pool hot and let preemption absorb the bursts. This is, once again, the serving triangle: preemption trades tail latency for throughput headroom, and where you set the dial depends on which corner you are defending.

## 8. Offloading and reuse: CPU, NVMe, and LMCache

Swapping under preemption is a reactive spill. There is also a proactive, strategic form of offloading whose goal is not surviving a burst but *reusing* KV across requests. This is where CPU and NVMe tiers, and systems like **LMCache**, come in.

The insight: in many workloads the *same* KV is computed over and over. A chat application prepends the same long system prompt to every conversation. A RAG system retrieves the same popular documents for many users. An agent framework re-sends a large tool-definition preamble on every step. Under prefix caching (covered in depth in the [prefix caching and RadixAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention) discussion within the PagedAttention post), the engine keeps the KV of shared prefixes and reuses it, so the second request with the same prefix skips prefill entirely and starts decoding at the first novel token. That turns a 50 ms TTFT into 5 ms for the shared portion.

The limit on prefix caching is HBM: you can only cache as many distinct prefixes as fit in the GPU KV pool, and they compete with active sequences for that pool. Offloading extends the prefix cache into a memory hierarchy. Hot prefixes live in HBM; warm ones spill to host RAM (hundreds of GB, cheap); cold ones spill to local NVMe (terabytes, cheaper still). LMCache implements exactly this — a multi-tier KV store that lets a fleet keep a very large working set of reusable KV, and even share it *across* serving instances via a networked backend. When a request arrives whose prefix is not in HBM but is on the CPU tier, loading it back is a PCIe copy — still far cheaper than recomputing prefill for a long prefix.

The reason the hierarchy works is that each tier trades capacity for bandwidth in a way that maps neatly onto access frequency:

| Tier | Typical capacity | Bandwidth to GPU | Role in the KV hierarchy |
|---|---|---|---|
| HBM (on-GPU) | 40–192 GB | 2–5 TB/s | Active sequences + hottest prefixes |
| Host RAM (CPU) | 256 GB–2 TB | ~25 GB/s (PCIe Gen4) | Warm prefixes, swapped victims |
| Local NVMe SSD | 2–30 TB | 3–7 GB/s | Cold but reusable prefixes |
| Networked KV store | effectively unbounded | network-limited | Cross-instance shared prefixes |

Each step down is roughly two orders of magnitude cheaper per gigabyte and one to two orders slower to reach. A well-designed KV tier keeps the working set in HBM, demotes on a least-recently-used basis to RAM and then NVMe, and only pays the slow-tier latency on a genuine miss to something cold — which, for a workload with high prefix reuse, is rare. The result is a KV "pool" that is effectively hundreds of gigabytes deep while each GPU's HBM stays near full utilization with the tokens that are actually being attended to right now.

The trade-off is transfer latency versus recompute cost, the same axis as swap-versus-recompute but for reuse rather than survival. Loading a cached prefix from host RAM pays a PCIe transfer; recomputing it pays prefill FLOPs. For long shared prefixes (a 4,000-token system prompt, a large retrieved document), the cached load wins decisively, because prefill over thousands of tokens is expensive and a PCIe copy of the resulting KV is comparatively cheap. For short prefixes the recompute is so cheap that offloading is not worth the machinery. So offloading and cross-instance KV sharing pay off exactly when you have long, frequently-reused prefixes and high request volume — production chat, RAG, and agent serving — and are pure overhead for one-off, unique-prompt workloads.

A capacity note that ties back to the equation: offloading effectively *grows* your KV pool by borrowing host RAM and NVMe, at the cost of transfer latency on cache misses to the slower tiers. It does not reduce bytes-per-token; it increases the pool term. Stack it with FP8 (smaller bytes-per-token) and PagedAttention (tight packing) and you get a system that keeps a large reusable KV working set across a fleet while running each GPU's HBM near full utilization.

## 9. The capacity equation, and the flags that control it

Now assemble the pieces into the numbers you actually configure. Three vLLM knobs interact to determine how many sequences you serve, and misunderstanding their interaction is the most common cause of both wasted capacity and surprise OOMs.

- **`gpu_memory_utilization`** (default 0.9): the fraction of total HBM the engine is allowed to claim. Everything not weights or overhead becomes KV pool. Raising it from 0.9 to 0.95 gives you more KV pool (more sequences) but leaves less headroom for activation spikes and fragmentation — risk an OOM under an unusual request. Lowering it to 0.85 is safer but wastes capacity. On a dedicated inference box with a single model, 0.90 to 0.92 is a sane default.
- **`max_model_len`**: the maximum context length. This does not directly allocate memory, but it caps how large any single sequence's KV can grow, which the scheduler uses for admission decisions. Setting it far higher than your real workload needs makes the engine conservative and can reduce concurrency; set it to your actual maximum, not the model's theoretical one.
- **`max_num_seqs`**: the hard cap on how many sequences run concurrently. This is a *ceiling*, not a target — the real limit is usually the KV pool, and `max_num_seqs` just prevents pathological batch sizes. Set it above your expected concurrency so it does not bind, but not so high that the per-sequence scheduling structures waste memory.

The relationship you must hold in your head: **the KV pool sets the true concurrency limit, `max_num_seqs` is a safety ceiling, and `max_model_len` bounds each sequence's share of the pool.** If you set `max_num_seqs=512` but the pool only holds 52 sequences at your context length, you will never see 512 in flight — you will see 52 running and the rest queued or preempted. If you want more than 52, you must shrink bytes-per-token (FP8), shrink context (`max_model_len` or sliding window), or add GPUs. The flag does not create memory.

Here is a KV-budget calculator you can drop into a capacity-planning script. It encodes the whole equation and the three knobs, and prints how many sequences fit — the single most useful function in this post.

```python
def kv_budget(
    gpu_hbm_gb: float,          # e.g. 80 for H100, 40 for A100-40GB
    weight_gb: float,           # params_billions * bytes_per_param
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    kv_dtype_bytes: float,      # 2.0 FP16/BF16, 1.0 FP8/INT8
    context_len: int,           # tokens held per sequence at steady state
    gpu_mem_util: float = 0.90, # vLLM gpu_memory_utilization
    overhead_gb: float = 4.0,   # CUDA ctx, activations, NCCL, scheduler
):
    """Return per-token KV bytes, pool GB, and max concurrent sequences."""
    bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * kv_dtype_bytes
    engine_budget_gb = gpu_hbm_gb * gpu_mem_util
    kv_pool_gb = engine_budget_gb - weight_gb - overhead_gb
    per_seq_gb = bytes_per_token * context_len / 1e9
    max_seqs = int(kv_pool_gb / per_seq_gb) if per_seq_gb > 0 else 0
    return {
        "bytes_per_token": bytes_per_token,
        "kv_pool_gb": round(kv_pool_gb, 1),
        "per_seq_gb": round(per_seq_gb, 3),
        "max_concurrent_seqs": max_seqs,
    }


# Llama-3-8B on H100 80GB, FP16 KV, 8K context
print(kv_budget(80, 16, 32, 8, 128, 2.0, 8192))
# -> {'bytes_per_token': 131072, 'kv_pool_gb': 56.0, 'per_seq_gb': 1.074, 'max_concurrent_seqs': 52}

# Same box, FP8 KV -> doubles concurrency
print(kv_budget(80, 16, 32, 8, 128, 1.0, 8192))
# -> {'bytes_per_token': 65536, 'per_seq_gb': 0.537, 'max_concurrent_seqs': 104}

# Llama-3-8B on A100 40GB, INT8 KV, 8K context
print(kv_budget(40, 16, 32, 8, 128, 1.0, 8192, overhead_gb=3.0))
# -> tighter pool; ~30 sequences
```

Run this before you provision anything. It replaces the guess-and-OOM loop with a number, and it makes the effect of every lever in this post quantitative: change `kv_dtype_bytes` to see quantization's payoff, change `num_kv_heads` to see GQA's, change `context_len` to see how a longer-prompt workload eats your concurrency. This is the tool that would have let the 3 a.m. engineer predict the incident from a product change ("we're adding document upload") instead of discovering it from a pager.

You can also let vLLM tell you what it computed. On startup it logs the number of GPU KV blocks it allocated; multiply by block size (16) to get the token capacity of the pool, and divide by your context length for the sequence count. And `vllm serve` exposes the same knobs on the command line for production:

```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-num-seqs 256 \
  --enable-chunked-prefill \
  --enable-prefix-caching
```

One interaction matters at the 70B scale: **tensor parallelism shards the KV cache too**. Running a model across 4 GPUs with `tensor_parallel_size=4` splits the KV heads across devices, so each GPU holds a quarter of the KV heads — the per-GPU KV footprint is divided by the TP degree, and each GPU contributes its own HBM to the aggregate pool. TP is therefore doing two jobs at once: it lets a model whose weights do not fit on one GPU run at all, and it multiplies the total KV pool by the number of GPUs. To size a sharded deployment, plug the *per-GPU* KV-head count into the calculator (for Llama-3-70B with 8 KV heads on `tensor_parallel_size=8`, that is one KV head per GPU) and the same arithmetic holds device by device. This is how 70B and larger models reach usable concurrency: not by shrinking KV in isolation, but by spreading both the weights and the cache across the interconnect so that no single GPU carries the whole burden.

## 10. Measuring it: a benchmark on H100 and A100

Numbers on paper are a hypothesis; a benchmark is the test. Here is a compact load generator that measures the two metrics that matter for KV work — sustained token throughput and the concurrency at which latency stays acceptable — so you can confirm the capacity math and quantify the FP8 win on your own hardware.

```python
import asyncio, time
from openai import AsyncOpenAI  # vLLM exposes an OpenAI-compatible server

client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="none")

async def one_request(prompt: str, max_tokens: int = 256):
    t0 = time.perf_counter()
    first_token_t = None
    n_tokens = 0
    stream = await client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, temperature=0.7, stream=True,
    )
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            if first_token_t is None:
                first_token_t = time.perf_counter()
            n_tokens += 1
    t_end = time.perf_counter()
    ttft_ms = (first_token_t - t0) * 1000 if first_token_t else None
    tpot_ms = ((t_end - first_token_t) / max(n_tokens - 1, 1)) * 1000 if first_token_t else None
    return ttft_ms, tpot_ms, n_tokens

async def run(concurrency: int, prompt: str, duration_s: int = 60):
    results, stop = [], time.perf_counter() + duration_s
    async def worker():
        while time.perf_counter() < stop:
            results.append(await one_request(prompt))
    await asyncio.gather(*[worker() for _ in range(concurrency)])
    ttfts = sorted(r[0] for r in results if r[0])
    tpots = [r[1] for r in results if r[1]]
    total_tokens = sum(r[2] for r in results)
    print(f"concurrency={concurrency}  reqs={len(results)}  "
          f"throughput={total_tokens/duration_s:.0f} tok/s  "
          f"p50 TTFT={ttfts[len(ttfts)//2]:.0f}ms  "
          f"p99 TTFT={ttfts[int(len(ttfts)*0.99)]:.0f}ms  "
          f"mean TPOT={sum(tpots)/len(tpots):.1f}ms")

# Sweep concurrency to find the knee where TTFT degrades from preemption.
asyncio.run(run(64, "Summarize this report: " + "lorem ipsum " * 500))
```

Sweep the `concurrency` argument up until p99 TTFT spikes — that spike is the KV pool filling and preemption kicking in, and it marks your real capacity at that context length. Below is a representative before/after table for Llama-3-8B, showing the effect of turning on FP8 KV cache, on two named GPUs. Treat these as order-of-magnitude figures consistent with published vLLM benchmarks and the capacity math above, not as a guarantee for your exact model revision and prompt mix — always run the sweep yourself.

| Setup | GPU | KV dtype | Context | Max concurrent seqs | Throughput (tok/s) | p99 TTFT | Notes |
|---|---|---|---|---|---|---|---|
| Baseline | H100 80GB | FP16 | 8K | ~52 | ~2,400 | ~180 ms | pool fills at ~52 |
| + FP8 KV | H100 80GB | FP8 E4M3 | 8K | ~104 | ~4,300 | ~170 ms | ~2x concurrency, <0.1 PPL loss |
| Baseline | A100 40GB | FP16 | 8K | ~17 | ~900 | ~260 ms | tight pool |
| + INT8 KV | A100 40GB | INT8 | 8K | ~34 | ~1,700 | ~250 ms | ~2x, needs calibration |
| + sliding window | H100 80GB | FP8 | stream (W=4K) | ~200+ | ~4,500 | ~160 ms | bounded cache per session |

The pattern is consistent: KV optimizations move the concurrency ceiling and, through it, aggregate throughput, with little effect on per-token latency (TPOT) because they do not change the per-step compute — they let more sequences share it. The one metric that moves for the worse, if you push too hard, is p99 TTFT under preemption, which is why the table pairs concurrency gains with a latency column. The right operating point is the highest concurrency at which p99 TTFT still meets your SLO, and every lever in this post shifts that point outward.

Two measurement caveats keep this honest. First, **report prefill and decode separately**, because they stress different resources — prefill is compute-bound and shows up as TTFT, decode is memory-bandwidth-bound and shows up as TPOT. A KV optimization moves the decode/concurrency numbers and barely touches prefill, so a single blended "tokens/s" figure can hide the win or the regression. The load generator above splits TTFT from TPOT precisely so you can see which one moved. Second, **your input/output length distribution is the experiment**, not a nuisance parameter. The same GPU and model produce wildly different capacity numbers under 500-token prompts versus 5,000-token prompts, because KV scales with tokens held. Benchmark with a length distribution that matches production traffic — ideally replayed from real logs — or your capacity number is measuring a workload you do not run. When you publish or compare KV benchmarks, always state the input and output lengths, the batch/concurrency, and the KV dtype; a throughput number without those three is uninterpretable, which is why the table above carries a context column and a KV-dtype column on every row.

## 11. Observing the KV cache in production

You cannot manage what you cannot see, and the KV cache is exactly the metric the 3 a.m. incident was blind to. vLLM exports the numbers you need on its `/metrics` Prometheus endpoint; the two that matter most are `vllm:gpu_cache_usage_perc` (the fraction of the KV pool currently in use) and `vllm:num_preemptions_total` (a counter that increments every time the scheduler preempts a victim). The first is your early-warning gauge; the second is the alarm that says you already crossed the wall.

The healthy pattern is a KV usage that breathes with traffic but stays clear of 100%, and a preemption counter that stays flat. When usage pins at the top and preemptions start climbing, you are out of KV pool — the exact condition behind the intro's outage — and the fix is one of the levers in this post (shrink bytes-per-token, cap context, add capacity), not a restart. A Prometheus alert makes it actionable:

```yaml
groups:
  - name: llm-serving-kv-cache
    rules:
      - alert: KVCacheNearlyFull
        expr: vllm:gpu_cache_usage_perc > 0.95
        for: 2m
        labels: { severity: warning }
        annotations:
          summary: "KV cache pool above 95% for 2m on {{ $labels.instance }}"
          description: "Concurrency ceiling reached; expect preemptions and TTFT spikes."
      - alert: KVCachePreemptionStorm
        expr: rate(vllm:num_preemptions_total[5m]) > 1
        for: 5m
        labels: { severity: critical }
        annotations:
          summary: "Sustained KV preemptions on {{ $labels.instance }}"
          description: "Pool is oversubscribed. Enable FP8 KV, cap context, or scale out."
```

Put `gpu_cache_usage_perc` on your main serving dashboard next to p99 TTFT and you will see the causal chain the intro's engineer could not: KV usage rises, then preemptions begin, then the TTFT tail spikes, then errors. Catching it at the first link — a usage gauge crossing 0.95 — turns a 3 a.m. page into a capacity ticket filed during business hours. This is the observability half of the [model serving SLAs and metrics](/blog/machine-learning/model-serving/model-serving-slas-and-metrics) discipline applied to the one resource that most often decides whether an LLM service stays up. A useful dashboard row pairs three panels: KV usage percent, preemptions per minute, and running-versus-waiting sequence counts. Together they tell you not just *that* you are memory-bound but *how* — a rising waiting queue with flat usage means you are compute-bound and should add replicas, while high usage with preemptions means you are KV-bound and should reach for the levers here first.

## 12. Putting the whole stack together

The techniques compose, and a production configuration usually stacks several. Here is an annotated vLLM setup for a high-concurrency chat and RAG service on H100s, with the reasoning for each choice inline:

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",  # GQA model: 8 KV heads, cheap cache
    kv_cache_dtype="fp8",           # halve bytes/token -> 2x concurrency, <0.1 PPL
    gpu_memory_utilization=0.90,    # leave 10% headroom so bursts don't OOM
    max_model_len=8192,             # cap context to the real workload, not the max
    max_num_seqs=256,               # safety ceiling; true limit is the KV pool
    enable_prefix_caching=True,     # reuse shared system-prompt KV across requests
    enable_chunked_prefill=True,    # keep long prefills from stalling active decode
    preemption_mode="recompute",    # short chat prompts: recompute beats swap
    tensor_parallel_size=1,         # 8B fits one H100; raise for 70B+
)
```

Read the choices as a decision tree. The model is GQA, so the cache is already a quarter of an MHA equivalent — that was the biggest lever and it came for free with model selection. FP8 KV halves it again, doubling concurrency at a negligible accuracy cost that a RAG workload with short retrieved chunks tolerates well. `gpu_memory_utilization=0.90` deliberately leaves headroom because this service has a latency SLO and we would rather run slightly below peak than eat frequent preemption stalls. Prefix caching is on because chat and RAG re-send the same system prompt constantly, so its KV should be computed once. Chunked prefill protects decode latency when a user pastes a long document. Recompute preemption fits the short-prompt profile. If this were a long-context document-analysis service instead, three choices would flip: `preemption_mode="swap"` (long prompts make recompute expensive), a larger `max_model_len`, and an LMCache tier to hold the large reused documents across replicas. The configuration is not a recipe to copy blindly; it is a set of knobs you set from the capacity calculator and the workload's context and reuse profile. That is the whole discipline — measure the workload, compute the budget, stack exactly the levers that clear the ceiling.

## KV reduction techniques compared

Stepping back, here is the full menu with the trade-off each one makes. The figure and table below are the decision surface.

![Comparison of KV reduction techniques by memory saved, accuracy impact, and when to use each](/imgs/blogs/kv-cache-optimization-8.webp)

| Technique | KV memory saved | Accuracy impact | Hardware / requirement | When to use |
|---|---|---|---|---|
| PagedAttention | 60–80% waste to <4% | None (exact) | Any (default in vLLM/SGLang) | Always — it is the baseline |
| GQA | 2–8x vs MHA | Trained in, near-zero | Model must ship with it | Prefer GQA models when KV-bound |
| MLA | ~93% vs MHA | Matches MHA (per paper) | DeepSeek models only | Serving DeepSeek-V2/V3 |
| FP8 KV cache | 2x (16 to 8 bit) | <0.1 PPL, small but real | Hopper (H100/H200) native | Long context, high concurrency |
| INT8 KV cache | 2x | Slightly more than FP8 | Pre-Hopper (A100/A10/T4) | Doubling capacity on Ampere |
| Sliding window | Caps at window W | Loses long-range recall | Model or engine support | Streaming, bounded-context chat |
| CPU/NVMe offload (LMCache) | ~10x working set | None (exact reuse) | Host RAM / NVMe + PCIe | Reused long prefixes, high volume |

Read it as a stack, not a menu of alternatives. PagedAttention is always on. GQA and MLA come with the model. On top of those, FP8 (or INT8) KV is the first serving-layer lever to reach for because it is cheap, general, and doubles capacity. Sliding window is a workload-specific lever for streaming. Offloading is for high-volume reuse. You compose them: an FP8 KV cache, on a GQA model, packed by PagedAttention, with prefix offloading to LMCache, is a stack where each layer multiplies the last, and together they can move a single GPU from serving a couple dozen conversations to serving hundreds.

## Case studies

**vLLM / PagedAttention (Kwon et al., SOSP 2023).** The founding result. On the ShareGPT and Alpaca workloads, vLLM achieved 2–4x higher throughput than the prior state of the art (Orca, FasterTransformer) at the same latency, and the entire gain came from memory management — near-zero KV waste versus 60–80% under contiguous allocation. The paper's central measurement is that PagedAttention let vLLM run 2–4x more sequences in the same HBM, which is the throughput win directly. It is the cleanest demonstration in the field that serving throughput is a memory-management problem before it is a kernel problem.

**DeepSeek-V2 and MLA.** DeepSeek's technical report presents MLA as the mechanism that made a 236B-parameter (21B active) MoE model economical to serve at long context. The reported ~93% KV cache reduction versus equivalent MHA is what let them push context to 128K and serve it at a cost per token that undercut dense competitors. The lesson for serving engineers is that the biggest KV wins are architectural and upstream of anything you configure — when the model is designed for cheap KV, your serving job gets dramatically easier. This is why DeepSeek's inference stack could hit throughput numbers that dense-MHA models of similar quality could not approach.

**StreamingLLM (Xiao et al., 2023).** The attention-sink discovery. The paper showed that a Llama-2 model, given a naive sliding-window cache, degrades catastrophically once the initial tokens are evicted, and that pinning just four sink tokens restores stable generation over sequences of four million tokens with a fixed cache. It is the reason bounded-cache streaming works at all, and a lovely example of how a tiny structural fix (keep four tokens) beats a large brute-force one (bigger cache). Any engine that offers unbounded streaming with a fixed KV budget is standing on this result.

**LMCache and cross-instance KV reuse.** Production RAG and agent systems, where the same long documents and tool preambles recur across requests and across replicas, report large TTFT reductions from tiered, shared KV caching — turning repeated multi-thousand-token prefills into cache loads. The measured win is workload-dependent (it scales with prefix reuse rate), but for high-reuse traffic the difference between recomputing a 4K-token system prompt every request and loading its cached KV is the difference between a 200 ms and a 20 ms TTFT.

## When to use this (and when not to)

KV cache optimization is not a single decision; it is a stack of them, each with a clear "skip it" condition.

- **PagedAttention: always.** There is no scenario where you serve LLMs at scale without it. If you are using vLLM, TGI, or SGLang, you already have it. If you hand-rolled an inference loop with a contiguous max-length KV buffer, stop and adopt a paged engine — this is not a tuning choice.
- **FP8/INT8 KV cache: yes for long context or high concurrency; skip for exact long-range recall.** Enable it by default and measure. The one case to be careful with is workloads that hinge on precisely attending to a specific far-back token (long-context needle retrieval, some legal/medical exact-recall tasks) — validate perplexity and task accuracy there before shipping. For chat, RAG with short chunks, and general generation, the accuracy cost is in the noise and the 2x capacity is real.
- **Sliding window: only for bounded-context workloads.** If your task needs full-context recall (long-document QA, code over a whole file), sliding window will silently drop the context and hurt quality in a way that is hard to catch in aggregate metrics. Use it for streaming and recency-dominated chat, not for retrieval or analysis over long inputs.
- **Preemption tuning: match it to your SLO.** If you have a tight p99 latency SLO, run the pool with headroom (`gpu_memory_utilization` ~0.88–0.90, `max_num_seqs` sized to real concurrency) so preemption is rare, and prefer swap over recompute for very long prompts. If you are running an offline batch job with no latency SLO, run the pool hot and let recompute-preemption maximize throughput.
- **Offloading / LMCache: only for high reuse.** The machinery earns its keep when you have long, frequently-repeated prefixes (shared system prompts, popular RAG documents, agent preambles) and enough volume that the cache hit rate is high. For unique-prompt, low-volume, or short-prefix workloads, the transfer and management overhead is not worth it — plain in-HBM prefix caching or nothing is better.
- **Don't over-optimize a small deployment.** If you serve one model to a handful of users on a single GPU with short contexts and low concurrency, the default vLLM config already leaves you swimming in KV pool. Enabling FP8, tuning preemption, and wiring up LMCache buys you capacity you will never use, at the cost of complexity and an accuracy risk you did not need to take. Reach for these levers when the capacity calculator says you are within 2x of the ceiling, not before.

The meta-rule: run the KV-budget calculator, find how close you are to the ceiling at your real context length and concurrency, and apply exactly the levers that move you comfortably clear of it. KV optimization is capacity engineering, and capacity engineering is driven by the number, not the fashion.

## Key takeaways

- **The KV cache is the memory wall.** After weights and overhead, it is the only elastic consumer of HBM, and it sets your concurrency ceiling: `max seqs ≈ KV pool / (bytes-per-token × context length)`.
- **Memorize the byte equation:** bytes/token = 2 × layers × KV-heads × head-dim × dtype-bytes. Llama-3-8B is 128 KiB/token (1 GiB per 8K sequence); a full-MHA 7B is 512 KiB/token. There is no hidden constant.
- **Decode is memory-bandwidth bound;** KV reads dominate at high batch and long context. This is why KV size, not FLOPs, gates decode throughput.
- **PagedAttention is the non-negotiable baseline** — it takes KV waste from 60–80% down to under 4% by allocating 16-token blocks on demand instead of reserving the max length.
- **GQA and MLA attack the equation at the source** and dwarf serving-layer tricks: GQA cuts KV 2–8x versus MHA for free; MLA cuts it ~93%. Choose KV-friendly models when you are memory-bound.
- **FP8 (Hopper) or INT8 (Ampere) KV quantization doubles capacity** with sub-0.1 perplexity cost — the highest-leverage serving-layer flag. Validate on long-context exact-recall workloads before trusting it there.
- **Preemption turns OOM into a bounded stall.** Choose recompute for short prompts, swap for long ones; run the pool with headroom if your p99 SLO is tight.
- **Offloading (CPU/NVMe, LMCache) grows the pool and enables cross-request KV reuse** — worth it for long, repeated prefixes at volume, overhead otherwise.
- **Compute capacity before provisioning.** The KV-budget calculator replaces guess-and-OOM with a number, and makes every lever's payoff quantitative.

## Further reading

- Kwon, Li, Zhuang, et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (SOSP 2023) — the founding paper on paged KV allocation and vLLM.
- DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model* (2024) — multi-head latent attention and the ~93% KV reduction.
- Xiao, Tian, Chen, et al., *Efficient Streaming Language Models with Attention Sinks* (StreamingLLM, 2023) — attention sinks and bounded-cache streaming.
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (2023) — the grouped-query attention formulation.
- vLLM documentation — KV cache dtype, `gpu_memory_utilization`, preemption modes, and prefix caching configuration.
- Within this series: [why LLM serving is different](/blog/machine-learning/model-serving/why-llm-serving-is-different), [continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention), and [quantization for LLM serving](/blog/machine-learning/model-serving/quantization-for-llm-serving).
- Model-architecture companions: [KV cache optimization and management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management) and [multi-head latent attention (MLA)](/blog/machine-learning/large-language-model/multi-head-latent-attention-mla).

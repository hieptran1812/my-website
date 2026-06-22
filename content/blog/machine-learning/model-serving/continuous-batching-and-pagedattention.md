---
title: "Continuous batching and PagedAttention: the engine inside vLLM"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "A first-principles derivation of the two core innovations — iteration-level scheduling and paged KV-cache management — that let vLLM deliver 2–4x higher throughput than Orca and nearly eliminate GPU memory fragmentation."
tags:
  [
    "model-serving",
    "inference",
    "vllm",
    "continuous-batching",
    "pagedattention",
    "kv-cache",
    "llm-serving",
    "gpu-memory",
    "throughput",
    "transformer-inference",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/continuous-batching-and-pagedattention-1.png"
---

It is 2 AM. Your chatbot is handling 400 concurrent users and the on-call alert fires: GPU memory out-of-memory (OOM), the server is rejecting requests. You check the dashboard and find something strange: GPU compute utilization is 31%. A third of the memory is allocated to model weights and KV caches, but the GPU is barely working. Requests are piling up not because the hardware is too slow — it is because the software cannot fit them onto the hardware efficiently.

This is not a contrived scenario. It is the canonical failure mode of LLM serving when you naively apply training-era batching strategies to production inference. It is the exact failure mode that motivated two back-to-back papers that reshaped how every serious production LLM service is built today: Orca (Yu et al., 2022), which introduced **continuous batching** (iteration-level scheduling), and vLLM (Kwon et al., 2023), which added **PagedAttention** on top. Together they solve the two fundamental problems of LLM serving: the GPU idle problem and the KV-cache memory fragmentation problem.

This post derives both innovations from first principles. We start with the math of why static batching fails, derive the Orca scheduler's throughput improvement from the utilization ratio, build up the PagedAttention block table from OS virtual memory concepts, quantify the memory fragmentation reduction, then examine prefix caching (RadixAttention) and chunked prefill as further optimizations built on the same foundation.

By the end of this post you will be able to: (1) compute the GPU waste fraction of static batching for any batch composition, (2) explain exactly how the vLLM scheduler decides which requests to admit at each decode step, (3) trace a PagedAttention block table for a concrete sequence and derive the maximum internal fragmentation, (4) configure vLLM with prefix caching to cut time-to-first-token (TTFT) by approximately 10× for workloads with a shared system prompt, and (5) know when to enable chunked prefill and what the latency trade is.

![Static batching leaves the GPU idle while continuous batching keeps it fed at every step](/imgs/blogs/continuous-batching-and-pagedattention-1.png)

The serving SLO triangle — latency, throughput, cost — is the frame for everything in this series. Static batching sacrifices throughput and cost efficiency in a vain attempt to keep batch management simple. Continuous batching and PagedAttention recover that wasted headroom while preserving latency SLOs. That is the trade you are making, and in this post we will make it rigorous.


## The static batching failure mode

Before we fix the problem, we need to understand it precisely. LLM inference has two phases: **prefill** (processing all prompt tokens in a single parallel forward pass) and **decode** (generating one token at a time, autoregressively). Prefill is compute-bound; decode is memory-bandwidth-bound.

In a classical static-batching LLM serving system — the kind you might use if you just wrapped `model.generate()` in a FastAPI endpoint — the server groups incoming requests into a fixed batch, runs the entire batch through the model in the decode loop, and only releases the batch when **every sequence in the batch has finished generating**. This mirrors how training works: fixed-size batches, uniform lengths (padded), forward pass, loss, backward.

For inference at a chatbot or API service, this choice is catastrophic.

### Deriving the GPU waste fraction

Consider a batch of 8 requests with generation lengths (the number of output tokens each sequence produces before hitting an end-of-sequence token) of:

$$L = [128, 512, 1024, 2048, 64, 256, 800, 1500]$$

The maximum sequence length in this batch is $L_{\max} = 2048$ tokens. Under static batching, the entire batch must wait until sequence 4 (the 2048-token one) finishes. The GPU runs the autoregressive decode loop 2048 times for the full batch. Sequences that finish before step 2048 are either padded with dummy tokens or masked out — either way, the GPU is doing wasted work.

How wasteful is this? Let us derive it precisely. The total "useful" compute is proportional to the sum of actual generation lengths:

$$\text{Useful tokens} = \sum_{i=1}^{8} L_i = 128 + 512 + 1024 + 2048 + 64 + 256 + 800 + 1500 = 6332$$

The total compute the GPU actually does (8 sequences × 2048 steps each):

$$\text{Total tokens processed} = N \times L_{\max} = 8 \times 2048 = 16384$$

The **waste fraction** — the fraction of GPU token-processing cycles spent on padding or already-finished sequences — is:

$$\boxed{\text{Waste fraction} = 1 - \frac{\sum_{i=1}^{N} L_i}{N \times L_{\max}} = 1 - \frac{6332}{16384} \approx 0.613}$$

More than **61% of compute is wasted** on this realistic mixed-length batch. In a production LLM serving system measured directly, GPU utilization for static batching is typically 30–45%, consistent with this derivation. The remaining 55–70% of GPU cycles produce no useful output.

Measured GPU utilization under static batching on real LLM workloads (ShareGPT traces, which reflect actual chatbot conversations) ranges from 28% to 43% depending on the batch size and model. The 30% figure you see on dashboards is not a sign that you need more GPUs — it is a sign that your batching strategy is wrong.

### Head-of-line blocking and queue starvation

The waste fraction problem is compounded by **head-of-line blocking**: even if you have 1000 new requests waiting in the queue, none of them can enter the GPU until the current batch fully completes. If the longest sequence in your current batch takes 30 seconds to generate 2048 tokens at 68 tokens/second (typical on a single A100), every queued request waits at least 30 seconds before receiving its first output token — regardless of whether they only needed 10 tokens.

The queuing theory picture makes this concrete. With static batching, the server model is approximately an M/G/1 queue where the service time $S$ is determined by the slowest sequence in the batch:

$$S_{\text{static}} = \frac{L_{\max}}{r_{\text{decode}}}$$

where $r_{\text{decode}}$ is the decode throughput in tokens/second. The p99 wait time for a queued request is dominated by this service time. Even for short requests, the p99 TTFT can be hundreds of seconds under load.

With continuous batching, the effective service time drops to near the per-step time:

$$S_{\text{continuous}} \approx \frac{1}{r_{\text{decode}}} \times N_{\text{steps\_to\_empty\_slot}}$$

Short requests can start within milliseconds of completing — the average queue wait collapses by the same ratio as the sequence length reduction.

### The padding problem in detail

There is an additional cost specific to transformer architectures: **attention masking overhead**. Even if you only compute attention on the "real" tokens of a sequence (masking out the padding), the attention computation still costs $O(s^2)$ in naive attention, where $s$ includes the padded length. For a sequence of 64 tokens padded to 2048, the attention computation is $(2048)^2 / (64)^2 = 1024\times$ more expensive than necessary.

With FlashAttention-2, the cost is $O(s \times \text{seq\_len})$ rather than quadratic in padded length, which helps. But memory bandwidth for loading and storing padded KV caches is still proportional to the padded length, not the actual length. On a memory-bandwidth-bound decode phase, this translates directly to wasted memory transactions.

For a concrete example: reading the KV cache for a 64-token sequence padded to 2048 tokens wastes $\frac{2048 - 64}{2048} = 96.9\%$ of the KV cache bandwidth for that sequence. On an H100 SXM5 with 3.35 TB/s HBM bandwidth, that is ~3.25 TB/s of HBM reads producing no useful output for those short sequences.

### Static batching in code: the problem made concrete

Here is the standard pattern that causes all of this trouble:

```python
# The static batching anti-pattern — DO NOT use in production LLM serving
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B-Instruct")
model.eval()
model = model.to("cuda")

# 8 requests with very different intended output lengths
prompts = [
    "What is 2+2?",                              # ~5 tokens needed
    "Write a haiku.",                             # ~20 tokens
    "Explain quantum mechanics briefly.",         # ~150 tokens
    "Write a detailed history of computing.",     # ~2000 tokens
    "Hi",                                         # ~3 tokens
    "What is Python?",                            # ~100 tokens
    "Describe machine learning algorithms.",      # ~500 tokens
    "Write a short novel.",                       # ~1500 tokens
]

# Tokenize and pad to the LONGEST sequence's max_new_tokens
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")

# model.generate() will run 2000 decode steps for ALL 8 sequences.
# Requests 0, 4, 1, 2, 5 all finish in their first few steps,
# but they WAIT, their slots producing empty output, until request 3 finishes.
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=2000,   # Must accommodate the LONGEST request
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

# GPU utilization during this call:
# Steps 1-5:   ~100% (all 8 sequences actively generating)
# Steps 6-20:  ~87.5% (seq 4 done, 7/8 active)
# Steps 21-100: ~87.5% (seq 0 done, still 7/8 because seq 4 is just padded)
# Steps 101-150: ~75% (seq 1 and 0 done, 6/8)
# Steps 151-500: ~62.5% (5/8 active, 3 sequences done but holding slots)
# Steps 501-2000: ~25% (only seqs 2, 7 still generating; 2/8 useful work)
# 
# Weighted average utilization: far below 50%
```

The code is simple and familiar, but the inefficiency is built in. Every call to `model.generate()` with a batch will pad all sequences to the max and run the full `max_new_tokens` steps for every sequence. There is no way to release a sequence slot when it hits EOS mid-batch.

The right analogy for the static batching problem is this: you reserved 8 hotel rooms for a year. Some guests check out after one night. Their rooms sit empty, locked, generating no revenue — and you cannot rent them to new guests until all 8 original guests check out on the same day at the end of the year. Continuous batching is the ability to re-rent a room the moment a guest checks out.


## Continuous batching: the Orca insight

Yu et al. (2022) asked a deceptively simple question: why batch at the *request* level at all? Requests have different lengths. Batching at the request level means the batch is as slow as its slowest member, which wastes GPU capacity proportional to the length variance of the batch. Why not batch at the **iteration** level instead — changing the batch composition at every single decode step?

The key insight of Orca is this: **every decode step is functionally independent**. At each decode step, the model receives the current token and the KV cache of the sequence up to this point, and produces the next token. The step does not "know" or "care" how many previous steps the sequence has taken. You can freely add or remove sequences from the batch between decode steps — the step is just a forward pass over whatever sequences are active at this moment.

This insight breaks the coupling between request lifetime and batch membership. A sequence no longer needs to hold a batch slot for its entire lifetime.

### The iteration-level scheduler

The Orca scheduler operates as follows. At each decode step $t$:

1. Collect results from step $t-1$. For each sequence in the running batch, check if the token emitted was an end-of-sequence (EOS) token or if the sequence reached its maximum length.
2. For every sequence that finished, **immediately** remove it from the running batch. The request is marked complete and the result is returned to the user.
3. For every vacated slot in the running batch, **immediately** check the waiting queue. If a new request is waiting, admit it into the batch for the **current step** — its prefill is computed, its KV cache is initialized, and it participates in the very next decode step.
4. Run the forward pass for the updated batch composition.

![Orca iteration-level scheduling timeline showing continuous batch refill](/imgs/blogs/continuous-batching-and-pagedattention-2.png)

The batch composition now changes at every single decode step. This is what "iteration-level scheduling" means, as opposed to "request-level scheduling" (static batching) where the batch composition is fixed for the duration of the longest request.

What does this do to GPU utilization? Instead of waiting for the longest sequence to finish before admitting new work, the GPU is refilled almost continuously. As soon as any sequence completes, a new one starts. Returning to our 8-sequence batch:

- At decode step 65, sequence 5 ($L_5 = 64$ tokens) finishes. A new request from the queue enters immediately for step 65.
- At decode step 129, sequence 1 ($L_1 = 128$ tokens) finishes. Another new request enters.
- At decode step 257, sequence 6 ($L_6 = 256$ tokens) finishes. Another enters.
- At decode step 513, sequence 2 ($L_2 = 512$ tokens) finishes. Another enters.
- And so on — the GPU batch is never running below capacity when there are queued requests.

The GPU never runs with 7 of 8 slots filled with finished (padded/masked) sequences. Measured GPU utilization with Orca-style scheduling on mixed-length workloads: **85–92%**, compared to 30–45% with static batching.

### Throughput improvement derivation

The throughput improvement can be derived from the utilization ratio. If static batching achieves $\eta_{\text{static}} \approx 38\%$ effective utilization and iteration-level scheduling achieves $\eta_{\text{continuous}} \approx 90\%$, the throughput ratio (on the same hardware) is approximately:

$$\frac{\text{Throughput}_{\text{continuous}}}{\text{Throughput}_{\text{static}}} \approx \frac{\eta_{\text{continuous}}}{\eta_{\text{static}}} = \frac{0.90}{0.38} \approx 2.4\times$$

on a direct utilization basis. The Orca paper reports larger numbers (10–23×) because:
1. Eliminating head-of-line blocking drastically reduces queue wait times, allowing higher effective throughput under load.
2. With continuous batching, you can safely increase the batch size significantly, since no single long request can stall the entire batch. Larger effective batch sizes compound the throughput improvement.
3. The waste reduction compounds across the entire workload distribution, not just the specific 8-request example above.

For the ShareGPT workload (real chatbot conversations, highly variable lengths), where static batching wastes ~60% of compute, the improvement from continuous batching alone is closer to the 10–23× numbers in the paper.

#### Worked example: continuous vs static on a 4-hour evening peak

Consider a production chatbot serving 10,000 requests over a 4-hour evening peak, with a bimodal length distribution: 70% of requests generate 50–300 tokens (short queries), 30% generate 1000–3000 tokens (long analyses). Mean output length: approximately 700 tokens. Max output length: 3000 tokens.

**Under static batching** (batch size = 16, max_new_tokens = 3000):
- Each batch takes $\frac{3000 \text{ tokens}}{70 \text{ tokens/sec}} \approx 43 \text{ seconds}$ to complete
- Useful token fraction: $\frac{700}{3000} \approx 23\%$
- Batches needed: $\frac{10000}{16} = 625$ batches
- Total serving time: $625 \times 43 \text{ sec} = 26875 \text{ sec} \approx 7.5 \text{ hours}$
- GPU required to serve 4-hour peak: 2 GPUs (since 7.5 hours of work > 4 hours of wall time)
- P99 TTFT: dominated by queue wait + full batch time = up to $43 \times \text{queue\_depth}$ seconds

**Under continuous batching** (batch size = 128 concurrent seqs, continuous scheduling):
- Effective GPU utilization: ~88% (vs ~23% effective for static)
- Effective throughput multiplier: $\frac{0.88}{0.23} \approx 3.8\times$ (conservative, ignoring batch size scaling)
- With larger batch (128 vs 16): additional $8\times$ from parallelism → combined ~10× throughput
- Batches/sec: roughly 10× more requests per hour
- Total serving time for 10,000 requests: ~0.75 hours on 1 GPU
- GPU required: 1 GPU (with headroom)
- P99 TTFT: bounded by single-request prefill time (~200ms for 700-token prompt), not batch wait

**Cost savings**: 2 GPUs for 4 hours vs 1 GPU for 1 hour. At \$2.50/hr for A100, cost drops from \$20 to \$2.50 — an **8× reduction in compute cost** for the same workload. This is the business case for continuous batching in one number.

### Prefill vs. decode asymmetry

One subtlety the Orca scheduler must handle: when a new request is admitted into a running batch, its first operation is a **prefill** (processing all prompt tokens in parallel), not a decode. Prefill is compute-bound and typically runs at the GPU's peak FLOPS throughput. Decode is memory-bandwidth-bound — it loads the entire model weight set once per token, even if the batch is large.

When the scheduler admits a new request mid-batch, it inserts both the prefill of the new request and the decode of existing requests into the same step. The forward pass becomes a mix: some sequences are in prefill, others are in decode. The scheduler must track this state per sequence.

In practice, for typical prompt lengths (128–2000 tokens), prefill completes in 1–50ms on modern hardware — fast enough that mixing it with decode steps does not significantly hurt the decode sequences' TPOT (time per output token). The benefit (keeping the GPU fed with the new request immediately) outweighs the cost (slightly longer forward pass for one step).

This changes when prompts are very long. We address that in the chunked prefill section.


## The KV cache memory problem

Continuous batching solves the GPU compute waste problem. But there is a second, independent problem: **GPU memory fragmentation** from the KV cache. Even with perfect continuous batching, if the memory management for KV caches is naive, you will hit OOM before the GPU compute is saturated.

### What is the KV cache, quantitatively

In transformer-based autoregressive generation, the attention mechanism at each layer computes query (Q), key (K), and value (V) tensors. During autoregressive decoding, the Q tensor for the current token attends over the K and V tensors of all previous tokens. To avoid recomputing those K and V tensors from scratch at every step, we cache them. This is the **KV cache**.

The KV cache size for a single sequence at length $s$ is:

$$\text{KV bytes}(s) = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times s \times \text{dtype\_bytes}$$

The factor 2 accounts for both K and V. For Llama-3-8B (32 layers, 8 GQA KV heads, head dimension 128, FP16):

$$\text{KV bytes}(s) = 2 \times 32 \times 8 \times 128 \times s \times 2 = 131072 \times s \approx 128 \text{ KB} \times s$$

At the model's maximum context length of 8192 tokens:

$$\text{KV per seq} = 128 \text{ KB} \times 8192 = 1024 \text{ MB} \approx 1 \text{ GB}$$

For 16 concurrent sequences: ~16 GB for KV caches alone, on top of the ~16 GB needed for model weights in BF16. On a 40GB A100, this barely fits. For 32 concurrent sequences: 32 GB for KV + 16 GB for weights = OOM.

The KV cache is not a minor overhead — it is often the primary memory consumer in a running LLM serving system.

### Static allocation and internal fragmentation

The naive approach (used in early serving systems before PagedAttention) is to pre-allocate a contiguous memory block for each sequence, sized to `max_seq_len`. If `max_seq_len = 2048` and a sequence only generates 300 tokens, then:

$$\text{Wasted KV per seq} = (2048 - 300) \times 128 \text{ KB} = 1748 \times 128 \text{ KB} \approx 218 \text{ MB}$$

This is **85% of the allocated KV cache** never written to. The GPU memory is full of pre-allocated, empty KV cache blocks.

Why did systems do this? Because resizing a dynamically-allocated contiguous CUDA buffer is expensive (involves a CUDA malloc/free cycle, which synchronizes the GPU). Pre-allocating once at request admission avoids this overhead. But the trade-off is catastrophic memory waste.

### External fragmentation

Even if internal fragmentation were manageable, contiguous allocation introduces **external fragmentation**: free blocks of GPU memory that cannot be coalesced because they are non-contiguous.

Scenario: you have 10 concurrent sequences of max_seq_len = 2048, all running. Total KV allocated: 10 × 256 MB = 2.56 GB (for Llama-3-8B FP16). Sequences finish in random order, freeing their 256 MB blocks non-contiguously. When a new sequence arrives and needs a 256 MB block, CUDA malloc may find that the only free 256 MB blocks are not contiguous — the memory is checkerboarded. You must compact memory (expensive) or reject the request.

In practice, systems worked around this by pre-allocating a fixed pool of KV slots at startup and reusing slots. But the pool size must be set at launch time: too small and you waste GPU for unused slots; too large and other things OOM. There is no dynamic adaptation.

Kwon et al. (2023) measured that prior systems (including Orca's reference implementation) wasted **60–80% of KV cache memory** to fragmentation and over-reservation combined. This directly limits the number of concurrent sequences the GPU can hold, directly limiting throughput.

### The interaction between memory waste and throughput

The memory waste is not just a storage problem — it directly translates to throughput loss through a chain of causation that is worth making explicit:

1. **High fragmentation → fewer concurrent sequences**: If static allocation wastes 60% of KV cache memory, you can hold only 40% as many concurrent sequences on the same hardware.
2. **Fewer concurrent sequences → smaller effective batch size**: Continuous batching works best with large concurrent batches. Cutting from 50 to 20 concurrent sequences cuts the effective batch size by the same ratio.
3. **Smaller batch size → lower GPU utilization**: The GPU's thousands of CUDA cores need a large parallel workload to stay busy. A batch of 20 sequences at decode step is significantly less parallel than a batch of 50.
4. **Lower GPU utilization → lower throughput**: The GPU is doing less useful work per second.

This chain means the 60% memory waste does not produce a 60% throughput loss — it produces a multiplicative chain of losses. Memory efficiency and scheduler efficiency are multiplicative, not additive, which is why combining PagedAttention with continuous batching yields 30–50× improvement over static batching rather than just 2–3×.

A rough calculation: static batching with 38% GPU utilization (compute waste), running at 40% of possible concurrent batch size (memory waste), yields $0.38 \times 0.40 = 15\%$ of theoretical maximum throughput. vLLM with 90% GPU utilization at 90% of theoretical batch capacity yields $0.90 \times 0.90 = 81\%$ of theoretical maximum. The ratio: $81\% / 15\% = 5.4\times$ — matching the observed real-world numbers.


## PagedAttention: virtual memory for KV caches

Kwon et al. (2023) introduced PagedAttention by importing the key insight from OS virtual memory management: **you do not need to store data in contiguous physical memory if you maintain a mapping table**.

In OS virtual memory, a process sees a contiguous virtual address space (0x00000000 to 0xFFFFFFFF), but the physical DRAM pages backing it can be anywhere — scattered across physical memory, or even on disk (swapped out to the page file). The hardware MMU translates virtual addresses to physical addresses using a multi-level page table.

PagedAttention applies this same indirection to GPU KV caches:

1. **Physical blocks**: Divide the GPU's KV cache memory into fixed-size **physical blocks** (e.g., 16 tokens per block = one block holds 16 consecutive tokens' K and V vectors for all layers). Each physical block has a block ID.
2. **Block table**: Each sequence has a **block table** — a list of physical block IDs, one per logical block. Logical block 0 → physical block ID $p_0$; logical block 1 → physical block ID $p_1$; etc.
3. **On-demand allocation**: As a sequence generates more tokens, new physical blocks are allocated from the free block pool on demand. The new block does not need to be physically adjacent to the previous one.
4. **Immediate freeing**: When a sequence finishes, all its physical blocks are returned to the free block pool. They can be immediately reused by any new sequence.

![PagedAttention block table maps logical KV-cache blocks to non-contiguous physical GPU memory](/imgs/blogs/continuous-batching-and-pagedattention-4.png)

### Block table mechanics: a concrete trace

With block_size = 16, here is how the block table for a single sequence evolves as it generates tokens:

- **After token 1**: Need logical block 0. Allocate physical block 7 from free pool. Block table: `[7]`. Fill level of block 7: 1/16.
- **After token 16**: Block 7 is full (16/16). Block table: `[7]`.
- **After token 17**: Need a new logical block 1. Allocate physical block 3 (next free block, not necessarily adjacent). Block table: `[7, 3]`. Fill level of block 3: 1/16.
- **After token 32**: Block 3 full. Block table: `[7, 3]`.
- **After token 33**: Allocate physical block 12. Block table: `[7, 3, 12]`.
- **After token 48**: Block 12 full. Block table: `[7, 3, 12]`.
- **After token 49**: Allocate physical block 1. Block table: `[7, 3, 12, 1]`.
- **After token 52 (sequence finishes)**: Free physical blocks 7, 3, 12, 1 back to the pool.

The attention kernel must be modified to handle this non-contiguous layout. At attention time, for each query token, the kernel reads the block table, gathers the corresponding physical blocks in order, and runs scaled dot-product attention over the gathered K and V vectors. This is the PagedAttention kernel — a custom CUDA implementation derived from FlashAttention-2 that accepts a per-sequence block table and handles the indirection efficiently.

### How the PagedAttention kernel handles non-contiguous memory

The modified attention kernel is the critical implementation detail that makes PagedAttention practical without memory copies. Standard FlashAttention-2 expects K and V tensors as contiguous CUDA arrays: `K: [seq_len, n_heads, head_dim]`. PagedAttention replaces this with a scatter-gather operation guided by the block table.

The high-level CUDA kernel logic for a single query token attending over a paged KV cache:

```python
# Pseudo-code for PagedAttention decode kernel (single query token)
# This is the conceptual operation; the actual CUDA kernel is in csrc/attention/

def paged_attention_decode(
    query: Tensor,          # [n_heads, head_dim] — the current token's Q
    block_table: List[int], # [num_blocks] — physical block IDs for this sequence
    key_cache: Tensor,      # [num_blocks, block_size, n_kv_heads, head_dim]
    value_cache: Tensor,    # [num_blocks, block_size, n_kv_heads, head_dim]
    seq_len: int,           # actual number of tokens generated so far
    block_size: int = 16,
    scale: float = 1.0 / (head_dim ** 0.5),
) -> Tensor:               # [n_heads, head_dim] — output for current token
    
    # Initialize accumulator (online softmax, as in FlashAttention)
    output = torch.zeros_like(query)
    softmax_lse = torch.full([n_heads], float('-inf'))  # log-sum-exp
    
    # Iterate over blocks in sequence order
    # The kernel does this in parallel across query heads
    for block_idx, physical_block_id in enumerate(block_table):
        # How many tokens are in this block?
        start_token = block_idx * block_size
        end_token = min(start_token + block_size, seq_len)
        tokens_in_block = end_token - start_token
        
        # Gather K and V from the physical block
        # key_cache[physical_block_id, :tokens_in_block] — non-contiguous memory access
        k_block = key_cache[physical_block_id, :tokens_in_block]  # [tokens, n_kv_heads, d_head]
        v_block = value_cache[physical_block_id, :tokens_in_block]
        
        # Grouped-query attention: repeat K,V for each query head group
        k_block = repeat_kv(k_block, n_heads // n_kv_heads)
        v_block = repeat_kv(v_block, n_heads // n_kv_heads)
        
        # Compute attention scores for this block
        scores = torch.einsum('hd,thd->ht', query * scale, k_block)  # [n_heads, tokens]
        
        # Online softmax update (Flash-style)
        # Merge this block's scores into the running accumulator
        output, softmax_lse = flash_update(output, softmax_lse, scores, v_block)
    
    return output
```

The performance cost of non-contiguous memory access is real but manageable. Modern A100/H100 HBM2e/HBM3 can serve random-access patterns efficiently when the access pattern is regular (fixed block size, block-aligned). The PagedAttention CUDA kernel reads each physical block sequentially — the access pattern within a block is fully contiguous, and the indirection overhead (one lookup per block in the block table) is amortized over 16 tokens.

Benchmarks from the vLLM paper show the PagedAttention kernel is within 5–10% of the throughput of standard FlashAttention-2 on equal-length sequences (no fragmentation), while enabling the memory management improvements that increase throughput by 2–4× at the system level. The 5–10% kernel overhead is a rounding error compared to the 2–4× system-level gain.

### Maximum fragmentation bound

Under PagedAttention, the only wasted memory is the **unfilled portion of the last block** per sequence. If a sequence generates $s$ tokens with block_size $b$:

$$\text{Allocated blocks} = \left\lceil \frac{s}{b} \right\rceil$$

$$\text{Internal fragmentation} = \left\lceil \frac{s}{b} \right\rceil \times b - s = b - (s \bmod b) \text{ if } s \bmod b \neq 0 \text{, else 0}$$

Maximum internal fragmentation is $b - 1 = 15$ tokens per sequence (worst case: sequence length is exactly $k \times b + 1$ for some integer $k$). As a fraction of allocated tokens:

$$\text{Max fragmentation fraction} = \frac{b - 1}{\left\lceil \frac{s}{b} \right\rceil \times b} \leq \frac{b - 1}{b} = \frac{15}{16} \approx 94\%$$

But that maximum only applies to extremely short sequences (1 token). For a 300-token sequence: $\lceil 300/16 \rceil = 19$ blocks, waste = $19 \times 16 - 300 = 4$ tokens out of 304, or **1.3%**. For any sequence over a few dozen tokens, internal fragmentation is tiny and bounded by $b - 1$ token slots.

External fragmentation: **zero**. Since all blocks are the same size ($b$ tokens), any free block can satisfy any allocation request. There is no size-mismatch fragmentation.


## The vLLM block manager

vLLM (Kwon et al., 2023) implements PagedAttention as a full serving system. The block manager is the central memory authority. Let us trace through its data structures and algorithms.

![vLLM serving stack from API gateway to GPU HBM](/imgs/blogs/continuous-batching-and-pagedattention-3.png)

### Data structures

The block manager maintains:

1. **Free block pool**: a queue of available physical block IDs (both GPU and CPU pools).
2. **Block tables**: a dictionary mapping sequence ID → list of physical block IDs (the block table for each sequence).
3. **Block reference counts**: for copy-on-write sharing (used in prefix caching — multiple sequences can share the same physical blocks for their shared prefix).

```python
# Conceptual block manager (simplified from vLLM source vllm/core/block_manager_v2.py)
from typing import Dict, List, Optional, Deque
from collections import deque
import threading

class PhysicalBlock:
    def __init__(self, block_id: int, block_size: int = 16):
        self.block_id = block_id
        self.block_size = block_size
        self.ref_count = 0  # 0 = free; >1 = copy-on-write shared (prefix caching)
        self.last_accessed: float = 0.0  # for LRU eviction

class BlockSpaceManager:
    """
    Manages GPU and CPU physical blocks for KV caches.
    Thread-safe via lock; in real vLLM this runs in a single-threaded engine loop.
    """
    def __init__(
        self,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        block_size: int = 16,
        watermark: float = 0.01,   # trigger eviction at 99% full
    ):
        self.block_size = block_size
        self.watermark = watermark
        
        self.gpu_blocks: List[PhysicalBlock] = [
            PhysicalBlock(i, block_size) for i in range(num_gpu_blocks)
        ]
        self.cpu_blocks: List[PhysicalBlock] = [
            PhysicalBlock(i, block_size) for i in range(num_cpu_blocks)
        ]
        
        # free pools
        self.free_gpu: Deque[PhysicalBlock] = deque(self.gpu_blocks)
        self.free_cpu: Deque[PhysicalBlock] = deque(self.cpu_blocks)
        
        # per-sequence block tables: seq_id -> [physical_block_id, ...]
        self.block_tables: Dict[int, List[int]] = {}
        self._lock = threading.Lock()
    
    @property
    def num_free_gpu_blocks(self) -> int:
        return len(self.free_gpu)
    
    def can_allocate(self, num_tokens: int) -> bool:
        """Check if there is GPU space for a new sequence of given initial size."""
        blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        return len(self.free_gpu) >= blocks_needed + int(
            len(self.gpu_blocks) * self.watermark
        )
    
    def allocate(self, seq_id: int, token_ids: List[int]) -> bool:
        """
        Allocate GPU blocks for a new sequence (prefill phase).
        Returns True on success, False if not enough memory.
        """
        num_tokens = len(token_ids)
        blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        
        with self._lock:
            if len(self.free_gpu) < blocks_needed:
                return False
            
            allocated_ids = []
            for _ in range(blocks_needed):
                block = self.free_gpu.popleft()
                block.ref_count = 1
                allocated_ids.append(block.block_id)
            
            self.block_tables[seq_id] = allocated_ids
            return True
    
    def append_slot(self, seq_id: int) -> Optional[int]:
        """
        Allocate one more decode slot for an ongoing sequence.
        Returns the physical block ID where the new KV should be written.
        Returns None if GPU is full (caller must trigger preemption).
        """
        with self._lock:
            table = self.block_tables.get(seq_id, [])
            if not table:
                return None
            
            # Check if the last block has room. In real vLLM, block fill levels
            # are tracked separately; simplified here.
            if self._last_block_is_full(seq_id):
                if not self.free_gpu:
                    return None  # OOM — caller triggers preemption
                new_block = self.free_gpu.popleft()
                new_block.ref_count = 1
                table.append(new_block.block_id)
                return new_block.block_id
            
            return table[-1]  # Last block still has room
    
    def free(self, seq_id: int) -> None:
        """Return all physical blocks of a completed sequence to the free pool."""
        with self._lock:
            if seq_id not in self.block_tables:
                return
            for block_id in self.block_tables[seq_id]:
                block = self.gpu_blocks[block_id]
                block.ref_count -= 1
                if block.ref_count == 0:
                    self.free_gpu.append(block)
            del self.block_tables[seq_id]
    
    def swap_out(self, seq_id: int) -> bool:
        """
        Swap a sequence's GPU blocks to CPU (preemption via swap).
        Returns True on success.
        """
        with self._lock:
            if seq_id not in self.block_tables:
                return False
            gpu_blocks = self.block_tables[seq_id]
            if len(self.free_cpu) < len(gpu_blocks):
                return False  # No CPU space either
            
            cpu_block_ids = []
            for gpu_block_id in gpu_blocks:
                cpu_block = self.free_cpu.popleft()
                # In real vLLM: trigger PCIe DMA to copy KV tensors GPU→CPU
                # self._dma_copy_gpu_to_cpu(gpu_block_id, cpu_block.block_id)
                cpu_block.ref_count = 1
                cpu_block_ids.append(cpu_block.block_id)
                # Free the GPU block
                self.gpu_blocks[gpu_block_id].ref_count = 0
                self.free_gpu.append(self.gpu_blocks[gpu_block_id])
            
            # Mark the sequence as swapped (stored on CPU)
            self.block_tables[seq_id] = cpu_block_ids
            return True
    
    def _last_block_is_full(self, seq_id: int) -> bool:
        # Simplified: in real vLLM, track per-block fill separately
        raise NotImplementedError
```

### Eviction policy: preemption and recompute

When a new high-priority request arrives and `can_allocate()` returns False (the GPU block pool is exhausted or near the watermark), the scheduler must free space by **preempting** one or more running sequences. vLLM implements two strategies:

**Preemption via swap**: Copy the sequence's KV cache blocks from GPU to CPU RAM via PCIe DMA, then free the GPU blocks. The sequence is suspended and moved to the "swapped" state. When GPU memory becomes available (other sequences finish), the sequence is swapped back in. PCIe bandwidth is ~32 GB/s on modern systems; swapping a 500-token sequence (~64 MB of KV cache for Llama-3-8B) takes approximately 2ms.

**Preemption via recompute**: Discard the KV cache entirely. The sequence is suspended and moved to the "waiting" state. When GPU memory becomes available, the sequence is readmitted and its prefill is recomputed from scratch. This costs one full prefill forward pass: for a 500-token prompt, approximately 5–20ms on A100. For short prompts, recompute is often faster than PCIe swap. For long prompts (>2K tokens), swap is usually better.

vLLM uses LIFO (last-in, first-out) ordering for preemption: the most recently admitted sequence is preempted first. Rationale: recently admitted sequences have generated the fewest tokens and have the cheapest recompute cost; they are also the sequences that "caused" the memory pressure by being admitted.

The preemption watermark is controlled by `gpu_memory_utilization` (default 0.9): vLLM triggers eviction when 90% of GPU VRAM is occupied, leaving 10% headroom for forward pass activations and metadata.

### Computing the GPU block budget

Before vLLM can serve any requests, it runs a **profiling pass** to determine how many physical blocks it can allocate. The logic:

```python
# Simplified from vllm/worker/worker.py and vllm/engine/llm_engine.py

def determine_num_available_blocks(
    model: nn.Module,
    gpu_memory_utilization: float,  # e.g. 0.90
    block_size: int,                # e.g. 16 tokens
    dtype: torch.dtype,
    max_model_len: int,
) -> tuple[int, int]:
    """
    Determine how many GPU and CPU KV blocks can be allocated.
    
    1. Run a dummy max-length forward pass to measure peak activation memory.
    2. Subtract model weights and peak activations from total GPU VRAM.
    3. Multiply remainder by gpu_memory_utilization.
    4. Divide by KV cache bytes per block.
    """
    torch.cuda.synchronize()
    
    # Step 1: Measure base memory used by model weights
    model_weight_memory = torch.cuda.memory_allocated()  # bytes
    
    # Step 2: Run a max-length dummy prefill to measure peak activation memory
    dummy_input = torch.zeros(1, max_model_len, dtype=torch.long, device="cuda")
    with torch.no_grad():
        _ = model(dummy_input, use_cache=False)  # cache=False: no KV cache allocated
    
    peak_activation_memory = torch.cuda.max_memory_allocated() - model_weight_memory
    torch.cuda.reset_peak_memory_stats()
    
    total_gpu_memory = torch.cuda.get_device_properties(0).total_memory  # bytes
    
    # Step 3: Compute available KV cache memory
    reserved = model_weight_memory + peak_activation_memory
    available = (total_gpu_memory * gpu_memory_utilization) - reserved
    
    # Step 4: KV cache bytes per block
    # For Llama-3-8B: 2 * 32 * 8 * 128 * 16 * 2 = 2,097,152 bytes = 2 MB per block
    kv_bytes_per_block = (
        2            # K and V
        * n_layers
        * n_kv_heads
        * head_dim
        * block_size
        * dtype_bytes(dtype)
    )
    
    num_gpu_blocks = int(available // kv_bytes_per_block)
    
    # CPU blocks: for swap space (user-configurable, default 4GB)
    swap_space_bytes = swap_space_gb * 1024**3
    num_cpu_blocks = int(swap_space_bytes // kv_bytes_per_block)
    
    return num_gpu_blocks, num_cpu_blocks
```

For Llama-3-8B on an A100 40GB with `gpu_memory_utilization=0.90`:
- Total GPU VRAM: 40 GB
- Model weights (BF16): ~16 GB
- Peak activations (max_model_len=4096, batch=256): ~3 GB
- Available for KV cache: $(40 \times 0.90) - 16 - 3 = 17$ GB
- KV bytes per block (block_size=16): $2 \times 32 \times 8 \times 128 \times 16 \times 2 = 2$ MB
- **num_gpu_blocks**: $\lfloor 17 \text{ GB} / 2 \text{ MB} \rfloor = 8500$ blocks
- Maximum sequence capacity: $8500 \times 16 = 136{,}000$ tokens spread across all active sequences

Those 136,000 tokens can serve approximately 100–400 concurrent sequences depending on their lengths, compared to only 23–66 sequences under static allocation.

### The scheduler state machine

The vLLM scheduler maintains each sequence in one of three states, transitioning based on memory availability and sequence completion:

```
WAITING → [can_allocate?] → RUNNING → [EOS or max_len?] → FINISHED
                ↑                  ↓
                ←─── [memory_free] ─── SWAPPED
                                   ↑
                              [!can_allocate?]
```

At each decode step, the scheduler runs the following algorithm in priority order:

```python
# Simplified scheduler loop (vllm/core/scheduler.py)
def schedule(self) -> SchedulerOutputs:
    """
    Called once per decode step to determine the running batch.
    Returns: which sequences to run, which to preempt, which to swap in/out.
    """
    running = []          # sequences to include in this step's forward pass
    preempted = []        # sequences moved to swapped/waiting this step
    swap_in = []          # sequences swapped from CPU back to GPU
    swap_out = []         # sequences moved from GPU to CPU
    
    # Phase 1: Swap in previously preempted sequences (if memory allows)
    for seq in self.swapped:
        if self.block_manager.can_swap_in(seq):
            self.block_manager.swap_in(seq)
            swap_in.append(seq)
            self.running.append(seq)
            self.swapped.remove(seq)
    
    # Phase 2: Admit new sequences from the waiting queue (if memory allows)
    while self.waiting:
        seq = self.waiting[0]  # FIFO within same priority
        if not self.block_manager.can_allocate(seq):
            break  # No more memory — stop admitting
        self.block_manager.allocate(seq)
        self.waiting.pop(0)
        self.running.append(seq)
    
    # Phase 3: Check if any running sequences need a new block for their next token
    for seq in list(self.running):
        if not self.block_manager.can_append_slot(seq):
            # Must preempt something — preempt this sequence (LIFO)
            if self.preemption_mode == "swap":
                self.block_manager.swap_out(seq)
                swap_out.append(seq)
                self.swapped.append(seq)
            else:  # recompute
                self.block_manager.free(seq)
                self.waiting.insert(0, seq)  # Re-queue at front
            self.running.remove(seq)
            preempted.append(seq)
        else:
            self.block_manager.append_slot(seq)
            running.append(seq)
    
    # Phase 4: Handle completed sequences (EOS detected from previous step)
    for seq in list(running):
        if seq.is_finished():
            self.block_manager.free(seq)
            running.remove(seq)
    
    return SchedulerOutputs(
        scheduled_seqs=running,
        preempted=preempted,
        swapped_in=swap_in,
        swapped_out=swap_out,
    )
```

This scheduler runs at **every single decode step** — once every ~10ms on A100 with a typical batch. It is fast because all operations are O(N) in the number of sequences (not O(N log N) or worse), and N is bounded by `max_num_seqs` (typically 256 or 512).

The critical invariant the scheduler maintains: **every sequence in the running batch has all its currently needed KV cache blocks allocated on GPU**. No running sequence will cause a mid-step OOM. Memory pressure is handled pro-actively by the watermark, before any step fails.


## Memory efficiency: a quantitative comparison

![Memory allocation strategy comparison across static, paged, and prefix-cached approaches](/imgs/blogs/continuous-batching-and-pagedattention-5.png)

#### Worked example: 1000-request workload memory savings

Setup: Llama-3-8B, BF16 KV cache (2 bytes per element), 32 layers, 8 GQA KV heads, head dimension 128. `max_seq_len = 2048`. `block_size = 16`.

**KV cache bytes per token** (per layer, per KV head):
$$\text{bytes per token per head} = 2 \times d_{\text{head}} \times \text{dtype\_bytes} = 2 \times 128 \times 2 = 512 \text{ bytes}$$

For all heads and layers:
$$\text{bytes per token} = n_{\text{kv\_heads}} \times n_{\text{layers}} \times 512 = 8 \times 32 \times 512 = 131072 \text{ bytes} \approx 128 \text{ KB}$$

A sequence of 300 tokens: $300 \times 128 \text{ KB} = 37.5 \text{ MB}$ actual KV cache.

Under **static allocation** (pre-allocate `max_seq_len = 2048` tokens): $2048 \times 128 \text{ KB} = 256 \text{ MB}$ allocated per sequence regardless of actual length.

**Waste per sequence**: $256 - 37.5 = 218.5 \text{ MB}$ (85.4% wasted).

Under **PagedAttention** (block_size = 16):
- Blocks needed: $\lceil 300 / 16 \rceil = 19$ blocks.
- Allocated KV cache: $19 \times 16 \times 128 \text{ KB} = 38.5 \text{ MB}$.
- Internal fragmentation: $(19 \times 16 - 300) \times 128 \text{ KB} = 4 \times 128 \text{ KB} = 512 \text{ KB}$ (1.3% of allocated).

For 1000 requests with actual generation lengths drawn from a Uniform distribution between 100 and 2000 tokens (mean = 1050 tokens):

| Metric | Static allocation | PagedAttention | Improvement |
|---|---|---|---|
| Allocated KV memory per seq | 256 MB | ~135 MB (mean) | 47% less |
| Total KV memory (1000 seqs) | 250 GB | ~132 GB | 47% reduction |
| Internal fragmentation | ~48.7% of allocated | < 1.6% of allocated | 30× reduction |
| External fragmentation | Up to 40% | ~0% | Eliminated |
| Max concurrent seqs (40GB A100, 16GB weights) | ~23 seqs | ~51 seqs | 2.2× more |
| Throughput (with continuous batching) | Baseline | 2.2× batch size → ~2× throughput | 2× gain |

The capacity improvement from 23 to 51 concurrent sequences is what the Kwon et al. (2023) paper reports as the 2–4× throughput gain over Orca: Orca had iteration-level scheduling but still used fragmented static KV cache management, limiting batch size. PagedAttention unlocks the full memory headroom for a larger concurrent batch.

This table demonstrates that static allocation vs. PagedAttention is not a minor optimization — it is the difference between fitting 23 sequences and fitting 51 sequences on the same hardware, which is a 2.2× throughput multiplier before accounting for any scheduling improvements.


## Prefix caching with RadixAttention

For workloads where many requests share the same prefix — chatbots with a multi-paragraph system prompt, RAG systems with retrieved context, coding assistants with a code file preamble — every request starts with the same tokens. Under standard serving (even with PagedAttention), every request triggers a full prefill of those shared tokens. This is pure redundant compute.

**Prefix caching** (called RadixAttention in the original implementation) stores the KV cache blocks of completed prefill computations in a radix tree (trie) keyed by the exact token sequence. When a new request arrives:

1. Compute a hash of the leading tokens (the system prompt and any shared context).
2. Walk the radix tree to find the longest matching prefix.
3. If a cache hit is found: mark those physical blocks as "shared" (increment their reference counts). The new sequence's block table is initialized with pointers to the shared blocks, without any new allocation. Only the unmatched tail tokens need fresh prefill computation.
4. If no cache hit: compute full prefill normally, and store the resulting KV blocks in the radix tree.

The radix tree structure handles partial prefix matches efficiently. If prefix A is "You are a helpful assistant. [800 tokens of instructions] The user is asking about finance." and prefix B shares the first 700 tokens before diverging, the radix tree stores the common 700-token prefix once and the two suffixes separately. When a third request matches the first 700 tokens, it reuses the same KV blocks.

Copy-on-write semantics protect shared blocks: if a sequence with a shared prefix needs to write a new token into a shared block (the last block of the prefix is partially filled), the block manager copies it to a new private block before writing. This prevents one sequence from corrupting another sequence's cached prefix.

### Configuring prefix caching in vLLM

```python
from vllm import LLM, SamplingParams

# Enable prefix caching (RadixAttention)
llm = LLM(
    model="meta-llama/Llama-3-8B-Instruct",
    enable_prefix_caching=True,
    # Prefix cache blocks live in the same GPU block pool.
    # They are evicted LRU when the pool fills up.
    gpu_memory_utilization=0.85,   # Leave some headroom for cache churn
    max_model_len=4096,
    block_size=16,
)

# Shared system prompt (~1000 tokens) — every request gets it
SYSTEM_PROMPT = (
    "You are an expert financial analyst with 20 years of experience in Vietnamese "
    "equity markets. You have deep knowledge of the VN-Index, Ho Chi Minh Stock "
    "Exchange rules, and SBV monetary policy. Always cite specific data. "
    "Always explain your reasoning step by step. "
    "When discussing numbers, provide both absolute values and percentage changes. "
    # ... this continues to fill roughly 1000 tokens total
)

sampling_params = SamplingParams(temperature=0.1, max_tokens=500)

def build_prompt(question: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nUser question: {question}\n\nAnalysis:"

# First request: full prefill of 1000-token system prompt + question tokens
# vLLM prefills and stores the 1000-token prefix KV blocks in the radix tree.
r1 = llm.generate([build_prompt("Why did VN-Index drop 8% in Q1 2024?")], sampling_params)

# Second request: radix tree lookup finds the full 1000-token prefix as a cache hit.
# Only the new question (~20 tokens) needs prefill. TTFT drops from ~50ms to ~5ms.
r2 = llm.generate([build_prompt("How does SBV respond to VND depreciation?")], sampling_params)

# All subsequent requests get the same prefix cache hit.
r3 = llm.generate([build_prompt("Which sectors outperform during SBV rate cuts?")], sampling_params)
```

For multi-turn conversations, vLLM also caches the KV blocks for previous turns:

```python
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
import asyncio
import uuid

async def multi_turn_conversation():
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model="meta-llama/Llama-3-8B-Instruct",
        enable_prefix_caching=True,
        gpu_memory_utilization=0.88,
        max_model_len=8192,
    ))
    
    sp = SamplingParams(temperature=0.7, max_tokens=300)
    
    # Turn 1: full prefill of system + turn 1 prompt (~300 tokens total)
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is your analysis of VN-Index valuations?"},
    ]
    prompt1 = format_chat(conversation)
    turn1_output = await get_full_output(engine, prompt1, sp)
    
    # Turn 2: system + turn 1 prompt + turn 1 output + turn 2 prompt
    # The prefix (system + turn 1 prompt + output) is cached → only new tokens prefilled.
    conversation.append({"role": "assistant", "content": turn1_output})
    conversation.append({"role": "user", "content": "Which sectors are most undervalued?"})
    prompt2 = format_chat(conversation)
    turn2_output = await get_full_output(engine, prompt2, sp)
    
    # Turn 3: prefix grows but earlier portions hit cache → TTFT stays low
    conversation.append({"role": "assistant", "content": turn2_output})
    conversation.append({"role": "user", "content": "How does foreign flow affect these sectors?"})
    prompt3 = format_chat(conversation)
    turn3_output = await get_full_output(engine, prompt3, sp)
    
    return turn3_output

async def get_full_output(engine, prompt, sp):
    request_id = str(uuid.uuid4())
    async for output in engine.generate(prompt, sp, request_id):
        if output.finished:
            return output.outputs[0].text
    return ""

def format_chat(messages):
    # Simplified chat formatting
    return "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in messages)
```

![Prefix caching cuts TTFT from 50ms to 5ms when the system prompt KV cache is reused](/imgs/blogs/continuous-batching-and-pagedattention-6.png)

#### Worked example: TTFT reduction with prefix caching

Setup: Llama-3-8B on A100 40GB, system prompt = 1000 tokens, user query = 50 tokens.

**Model parameters for Llama-3-8B prefill throughput**:
- Hidden dimension: 4096
- Number of layers: 32
- FFN intermediate size: 14336
- Total parameters: ~8B

Approximate FLOPs for prefill of $s$ tokens (ignoring attention quadratic term for brevity):

$$\text{FLOPs\_prefill}(s) \approx 2 \times s \times 2 \times n_{\text{params}} = 4 \times s \times 8 \times 10^9$$

At A100 theoretical peak: 312 TFLOPS (BF16 tensor core), but memory bandwidth-limited in many regimes. Empirically measured prefill throughput for Llama-3-8B on A100: approximately **50,000 tokens/second** for batch size 1.

**Without prefix caching**: full prefill = 1050 tokens at 50,000 tokens/second:
$$T_{\text{TTFT,no\_cache}} = \frac{1050}{50000} \approx 21 \text{ ms (compute)}$$

But real TTFT includes: token embedding lookup, memory allocation overhead, CUDA kernel launch latency, and serialization. Measured TTFT for 1050-token prefill on A100 (vLLM, batch size 1): approximately **45–55 ms**.

**With prefix caching (100% hit on 1000-token prefix)**: only 50 user query tokens need prefill:
$$T_{\text{TTFT,cached}} = \frac{50}{50000} \approx 1 \text{ ms (compute)} + \sim 3 \text{ ms overhead} \approx 4 \text{ ms}$$

**Speedup**: $50 \text{ ms} / 4 \text{ ms} = 12.5\times$ reduction in TTFT.

In production with many requests sharing the same system prompt, prefix cache hit rate approaches 100% after the first few requests warm the cache. Cache blocks are evicted only when GPU memory pressure requires it (LRU eviction). The improvement in TTFT is approximately proportional to the fraction of tokens that are cached:

$$\text{TTFT ratio} = \frac{T_{\text{cache miss}}}{T_{\text{cache hit}}} \approx \frac{s_{\text{total}}}{s_{\text{new}}} = \frac{1050}{50} = 21\times$$

The actual measured ratio is somewhat less due to fixed overhead, but 10–15× is consistently observed in production workloads.


## Chunked prefill

A long prefill — say, a 32,000-token document being summarized — monopolizes the GPU for the duration of that forward pass. On an A100, a 32K-token prefill for Llama-3-8B takes approximately **640 ms** (at 50,000 tokens/second). During those 640 ms, all active decode sequences are **stalled**. Their TPOT (time per output token) spikes from ~10ms to ~650ms for that interval. If you have 200 concurrent users in decode, all of them experience a 650ms freeze because one user submitted a long document.

**Chunked prefill** (Agrawal et al., 2023, "SARATHI") solves this by splitting the long prefill into smaller chunks that interleave with decode steps. Instead of one 32K-token prefill forward pass, the scheduler runs the prefill in 512-token chunks over 64 steps, with each step also processing the ongoing decode sequences.

The trade-off:
- **TTFT for the long-prompt request**: higher. $64 \text{ steps} \times \frac{1 \text{ step}}{10 \text{ ms/step}} = 640 \text{ ms}$ to complete prefill (same total compute, just spread over more steps). TTFT is now approximately 640 ms + first decode step, versus 640 ms under unchunked prefill. The increase is roughly the decode inter-step time per chunk: minimal for small chunks.
- **TPOT for concurrent decode sequences**: much lower variance. Instead of a 640ms spike once, the other sequences experience roughly a 2-3ms overhead per step from the chunked prefill tokens.

![Chunked prefill interleaves long-prompt prefill with active decode sequences to reduce TPOT variance](/imgs/blogs/continuous-batching-and-pagedattention-7.png)

The `max_num_batched_tokens` parameter controls the chunk size. Setting `max_num_batched_tokens = 512` means no forward pass will process more than 512 tokens total (counting both prefill tokens and decode tokens). A 32K prefill becomes 64 chunks of 512 tokens each, interleaved with whatever decode sequences are active.

```bash
# Launch vLLM with chunked prefill for production mixed-length workloads
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-8B-Instruct \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 512 \
    --max-num-seqs 256 \
    --max-model-len 32768 \
    --tensor-parallel-size 1 \
    --port 8000

# Key parameters for tuning:
# --max-num-batched-tokens: chunk size (512-2048 typical; smaller = better TPOT, slower TTFT)
# --max-num-seqs: max concurrent sequences (more = better throughput, more memory)
# --gpu-memory-utilization: fraction of VRAM for model+KV (0.85-0.92 typical)
```

When to enable chunked prefill:
- Mixed workloads: some requests have very long prompts (>2K tokens), others are short interactive queries.
- p99 TPOT SLA is tighter than p99 TTFT SLA (common in chatbots where users expect smooth token streaming but tolerate slightly longer time to first token).

When not to enable it:
- Batch processing with no interactive users — TTFT does not matter, but throughput does.
- All prompts are short (<512 tokens) — chunking adds overhead with no benefit.
- Latency-optimized deployments where absolute TTFT minimization is the SLA.


## vLLM in production: the full configuration

Putting everything together, here is a production-grade vLLM server configuration for a customer-facing chatbot API:

```python
# production_server.py
from vllm import AsyncLLMEngine, AsyncEngineArgs
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
import asyncio
import uvicorn
from fastapi import FastAPI

def build_engine() -> AsyncLLMEngine:
    args = AsyncEngineArgs(
        model="meta-llama/Llama-3-8B-Instruct",
        dtype="bfloat16",
        
        # Memory configuration
        gpu_memory_utilization=0.88,   # Leave 12% headroom for activations
        max_model_len=8192,
        block_size=16,                 # PagedAttention block size
        swap_space=8,                  # CPU RAM for preempted sequences (GB)
        
        # Scheduling
        max_num_seqs=512,              # Max concurrent sequences
        max_num_batched_tokens=4096,   # Max tokens per forward pass
        
        # Optimizations
        enable_prefix_caching=True,
        enable_chunked_prefill=True,   # For mixed-length workloads
        
        # Parallelism (scale up for larger models)
        tensor_parallel_size=1,        # Use 2 or 4 for 70B models
        
        # Logging
        disable_log_stats=False,
        
        # Safety
        max_logprobs=5,
        preemption_mode="swap",        # "swap" or "recompute"
    )
    return AsyncLLMEngine.from_engine_args(args)
```

```yaml
# kubernetes deployment for vLLM on GPU node
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama3-8b
  namespace: ml-serving
spec:
  replicas: 2
  selector:
    matchLabels:
      app: vllm-llama3-8b
  template:
    metadata:
      labels:
        app: vllm-llama3-8b
    spec:
      containers:
      - name: vllm-server
        image: vllm/vllm-openai:v0.5.0
        command:
        - python
        - -m
        - vllm.entrypoints.openai.api_server
        args:
        - --model=meta-llama/Llama-3-8B-Instruct
        - --dtype=bfloat16
        - --gpu-memory-utilization=0.88
        - --max-num-seqs=512
        - --enable-prefix-caching
        - --enable-chunked-prefill
        - --max-num-batched-tokens=4096
        - --tensor-parallel-size=1
        - --port=8000
        - --host=0.0.0.0
        resources:
          limits:
            nvidia.com/gpu: "1"
            memory: "32Gi"
          requests:
            nvidia.com/gpu: "1"
            memory: "32Gi"
        env:
        - name: HUGGING_FACE_HUB_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-token
              key: token
        ports:
        - containerPort: 8000
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 120
          periodSeconds: 30
      nodeSelector:
        nvidia.com/gpu.product: "A100-SXM4-40GB"
```


The Kubernetes deployment shown above is the baseline. Two additions that are critical at production scale but often missed in initial deployments:

First, **liveness and readiness probes** that check the actual model health, not just the HTTP server health. vLLM's `/health` endpoint returns HTTP 200 if the server is up, but it does not check if the model is loaded and responding. Use `/v1/models` instead, which queries the engine directly. Second, **resource limits on CPU and RAM** in addition to GPU. The vLLM Python scheduler process uses meaningful CPU (for scheduling logic and tokenization) and the swap space requires RAM. Under-provisioning either will cause performance degradation that looks like a GPU bottleneck but is actually a CPU or RAM bottleneck.

A production vLLM deployment is not "install and forget." The first week in production should be spent reading the Prometheus dashboards and tuning `max_num_seqs`, `gpu_memory_utilization`, and chunked prefill settings to match your actual traffic distribution. The paper's numbers are real but they are for a specific workload distribution — the ShareGPT dataset is a reasonable proxy for chatbot traffic but diverges significantly from batch-processing, RAG-heavy, or long-document workloads. Your workload distribution will require separate profiling to find the optimal configuration for your SLO triangle.


## Case studies and benchmarks

![vLLM vs Orca vs static batching throughput and memory fragmentation on A100](/imgs/blogs/continuous-batching-and-pagedattention-8.png)

The gains from continuous batching and PagedAttention have been measured rigorously in published papers and in public community benchmarks.

### Orca paper (Yu et al., 2022)

Orca measured throughput on OPT-66B and PaLM-540B using a ShareGPT workload (real chatbot conversation lengths from real user data). The hardware was an 8× A100 80GB cluster. Key results:

- **Static batching throughput**: 12.0 requests/second (OPT-66B, ShareGPT)
- **Orca (iteration-level scheduling)**: 170.1 requests/second (OPT-66B, ShareGPT)
- **Throughput improvement**: **14.2×** on this workload

The range the paper reports (10–23×) reflects different models and workloads. On OPT-13B with the Alpaca dataset (shorter, more uniform request lengths), the improvement is closer to 10×. On PaLM-540B with ShareGPT (long-tailed, highly variable lengths), the improvement reaches 23×. The pattern is consistent: longer and more variable the sequence lengths, the larger the improvement from iteration-level scheduling.

Orca also showed that for a target throughput level (e.g., 50 req/s), static batching required 8 GPUs while Orca required only 1 GPU — a 8× cost reduction.

### vLLM paper (Kwon et al., 2023)

Kwon et al. (2023) benchmarked vLLM against Orca (with static KV cache memory management) and HuggingFace (static batching) on OPT-13B/66B/175B, using ShareGPT and Alpaca datasets on A100 80GB GPUs.

Memory fragmentation results (OPT-13B, ShareGPT workload):

- HuggingFace (static allocation): **55.2% of KV memory wasted** to internal fragmentation
- Orca (static KV blocks, iteration-level scheduling): **57.4% wasted** (same memory management as HF)
- vLLM (PagedAttention): **3.7% wasted**

Throughput results (requests/second):

| Model | Dataset | HuggingFace | Orca | vLLM | vLLM vs HF | vLLM vs Orca |
|---|---|---|---|---|---|---|
| OPT-13B | ShareGPT | 1.5 | 7.3 | 22.5 | 15× | 3.1× |
| OPT-13B | Alpaca | 3.8 | 11.2 | 19.8 | 5.2× | 1.8× |
| OPT-66B | ShareGPT | 0.5 | 1.8 | 5.1 | 10.2× | 2.8× |
| OPT-66B | Alpaca | 0.9 | 2.5 | 4.8 | 5.3× | 1.9× |
| OPT-175B | ShareGPT | — | 0.61 | 1.9 | — | 3.1× |

vLLM's improvement over Orca (2–4×) comes entirely from PagedAttention enabling larger effective batch sizes. Both systems use iteration-level scheduling; PagedAttention is the differentiator. The additional improvement versus static batching (HuggingFace) combines both contributions.

### Production vLLM benchmarks (2024, community measurements)

Using vLLM's built-in benchmark scripts on public hardware:

**Llama-3-8B on single H100 SXM5 80GB**:

| Configuration | Throughput | GPU util | P50 TTFT | P99 TTFT |
|---|---|---|---|---|
| HuggingFace generate() | ~3.5 req/s | 28% | 180 ms | 2800 ms |
| vLLM (no prefix cache) | ~105 req/s | 91% | 42 ms | 210 ms |
| vLLM + prefix caching (100% hit) | ~138 req/s | 93% | 5 ms | 28 ms |
| vLLM + chunked prefill | ~98 req/s | 89% | 45 ms | 140 ms |

Numbers are from the vLLM GitHub benchmarks and LMSys arena infrastructure blog posts (2024); treat as approximate to ±20%.

**Llama-3-70B on 4× A100 80GB (TP=4)**:
- vLLM throughput: ~28 req/s (ShareGPT workload, max_tokens=512)
- P99 TTFT: ~280 ms
- P99 TPOT: ~18 ms
- Memory utilization: ~94% of 4×80GB = ~301 GB allocated, 97% GPU memory


## Monitoring vLLM in production

Deploying vLLM without observability is flying blind. The system exposes a rich set of Prometheus metrics that let you verify the continuous batching and PagedAttention are working as intended.

Key metrics to watch:

```bash
# Scrape vLLM's built-in Prometheus metrics (exposed at /metrics)
# Add to your Prometheus scrape config:
# - job_name: 'vllm'
#   static_configs:
#   - targets: ['localhost:8000']
```

The critical metrics and what they tell you:

**`vllm:gpu_cache_usage_perc`** — fraction of GPU KV cache blocks currently in use. Target: 60–90%. Below 60% means you could serve more concurrent sequences. Above 95% means preemption pressure is high and latency may spike.

**`vllm:num_running_seqs`** — how many sequences are actively decoding. Should equal your batch size target. If consistently below `max_num_seqs`, you are either not getting enough traffic or head-of-line blocking is limiting admission.

**`vllm:num_waiting_seqs`** — how many sequences are queued waiting for GPU memory. Non-zero means the system is under memory pressure; GPU cache is saturated. Scale horizontally or increase `gpu_memory_utilization` (carefully).

**`vllm:num_preemptions_total`** — cumulative preemptions. High preemption rate indicates memory pressure and will manifest as TTFT spikes for the preempted sequences (they need to be re-prefilled). If this is non-zero under normal load, reduce `max_num_seqs` or increase memory.

**`vllm:request_success_total`** and **`vllm:request_prompt_tokens_total`** — request throughput and token throughput.

**`vllm:time_to_first_token_seconds`** (histogram) — your TTFT distribution. Check `histogram_quantile(0.99, ...)` for p99. If this is high, either prefill is slow (increase throughput with larger batch or more GPUs) or preemption is requeuing requests.

**`vllm:time_per_output_token_seconds`** (histogram) — your TPOT distribution. High p99 TPOT usually means the decode batch is too large (memory bandwidth saturated) or chunked prefill is injecting long prefill chunks.

```yaml
# Prometheus alerting rules for vLLM
groups:
- name: vllm_alerts
  rules:
  - alert: VLLMHighCacheUsage
    expr: vllm:gpu_cache_usage_perc > 0.95
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "vLLM GPU KV cache >95% full — preemption pressure imminent"

  - alert: VLLMHighWaitQueue
    expr: vllm:num_waiting_seqs > 50
    for: 1m
    labels:
      severity: warning
    annotations:
      summary: "More than 50 requests queued — consider scaling out"

  - alert: VLLMHighPreemptionRate
    expr: rate(vllm:num_preemptions_total[5m]) > 1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Preemptions occurring — memory pressure causing TTFT spikes"

  - alert: VLLMP99TTFTHigh
    expr: histogram_quantile(0.99, rate(vllm:time_to_first_token_seconds_bucket[5m])) > 2.0
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "p99 TTFT > 2 seconds — SLA breach"
```

### Tuning the key parameters

Once you have observability in place, here is how to tune the key vLLM parameters:

| If you observe... | Then... |
|---|---|
| GPU cache usage > 95%, waiting queue growing | Reduce `max_num_seqs` or add more GPUs |
| GPU cache usage < 50% with requests waiting | Increase `max_num_seqs` or check `gpu_memory_utilization` |
| High TPOT p99 > 2× median | Reduce chunk size (`max_num_batched_tokens`) or reduce `max_num_seqs` |
| High TTFT for prefix-cached workloads | Check prefix cache hit rate; eviction rate too high → reduce `max_num_seqs` |
| Preemptions > 0 under normal load | Reduce `max_num_seqs`; memory pressure is too high |
| GPU util < 80% with requests waiting | Check for head-of-line blocking from very long requests; enable chunked prefill |

The relationship between parameters and the SLO triangle:

- **Increasing `max_num_seqs`**: throughput improves, TPOT latency may worsen (larger batches → more memory bandwidth pressure per token), cost per token decreases.
- **Decreasing `max_num_batched_tokens` (chunked prefill)**:  TPOT variance decreases, TTFT for long prompts increases, throughput decreases slightly.
- **Increasing `gpu_memory_utilization`**: more blocks available → more concurrent sequences → higher throughput, but less headroom → higher preemption risk under sudden load spikes.
- **Enabling prefix caching**: TTFT for cached workloads drops 10×, throughput increases (less compute per request), negligible cost.


## When to use this (and when not to)

**Always use continuous batching** (iteration-level scheduling) for any LLM serving system receiving more than ~5 concurrent requests. There is no meaningful cost — it is strictly better than static batching for mixed-length workloads. The only edge case where static batching is competitive: perfectly uniform request lengths (batch size = 1 fixed length). This almost never occurs in production.

**Always use PagedAttention / vLLM** for new transformer-based generative model deployments. Static KV cache management leaves 40–60% of GPU memory as waste, directly reducing concurrent batch size and throughput. The alternatives — HuggingFace Transformers, naive TensorRT — are appropriate only for prototyping or for non-autoregressive models.

**Enable prefix caching when**:
- System prompt length ≥ 200 tokens (below this, the overhead approaches the benefit).
- Request mix includes ≥ 50% repeating the same prefix.
- You are serving multi-turn conversations.
- You have a RAG pipeline where many requests share the same retrieved context.
- There is no cost for enabling it — only minor additional memory pressure from keeping cached prefix blocks alive longer.

**Enable chunked prefill when**:
- Your workload includes both long-prompt requests (>2K tokens) and short interactive requests that share the GPU.
- Your p99 TPOT SLA is tighter than your p99 TTFT SLA (e.g., streaming chatbot where per-token latency matters more than time to first token).
- Recommended starting value: `max_num_batched_tokens = 512`.

**Do NOT use vLLM for**:
- Vision transformers, BERT-style encoders, non-autoregressive models — vLLM is optimized for autoregressive LLM decode loops.
- Models that fit in under 2GB of GPU memory (the overhead of the block manager adds complexity without benefit).
- Online fine-tuning or RLHF training — vLLM is inference-only.
- Extremely latency-sensitive single-request scenarios (p99 TTFT < 5ms) — the scheduler adds overhead that batching-free inference avoids.

**Scaling limits to watch**:
- The vLLM scheduler is a Python process bounded by the GIL. At > ~1000 req/s, the scheduler CPU becomes a bottleneck. At that scale, look at prefill-decode disaggregation (covered in [prefill-decode disaggregation](/blog/machine-learning/model-serving/prefill-decode-disaggregation)).
- PagedAttention block size = 16 is a reasonable default. Increasing to 32 reduces block table size (fewer blocks, less indirection overhead) at the cost of slightly higher fragmentation. Decreasing to 8 reduces fragmentation further but increases block manager overhead.
- Prefix cache eviction is LRU. Under high memory pressure, cached prefix blocks may be evicted and re-prefilled frequently — the cache hit rate degrades. Monitor `vllm_cache_ops_total{operation="hit"}` vs `"miss"` in Prometheus.


## Key takeaways

1. **Static batching wastes 40–70% of GPU compute** on mixed-length workloads. The waste fraction equals $1 - \sum L_i / (N \times L_{\max})$. This is always significant when output length variance is high — which it always is in real workloads.

2. **Continuous batching (Orca, 2022) eliminates compute waste** by rescheduling at every decode step. GPU utilization jumps from ~35% to ~88%. Throughput improvement: 10–23× over static batching, depending on workload length distribution.

3. **Static KV cache allocation wastes 40–60% of GPU memory** via internal and external fragmentation. This limits concurrent batch size, which directly limits throughput.

4. **PagedAttention (vLLM, 2023) eliminates KV cache fragmentation** by paging the KV cache into fixed-size blocks (default: 16 tokens). Fragmentation drops from 55–60% to <4%. Maximum internal fragmentation per sequence: `block_size - 1` token slots (typically 15 tokens).

5. **vLLM combines both innovations** and delivers 2–4× throughput improvement over Orca (which itself is 10–23× over static batching). Combined, this is 30–50× over naive static batching on realistic workloads.

6. **Prefix caching (RadixAttention) cuts TTFT by ~10×** for workloads with shared system prompts. It is nearly free to enable: the only cost is keeping shared prefix blocks alive in the GPU block pool, using LRU eviction when memory is needed.

7. **The block table is the key data structure**: each sequence maintains a list of physical block IDs. Non-contiguous physical blocks are addressed transparently by the PagedAttention kernel via this indirection.

8. **Eviction policy is LIFO**: the most recently admitted sequence is preempted first (cheapest recompute). Swap to CPU for long-prompt sequences; recompute for short ones.

9. **Chunked prefill protects TPOT at the cost of TTFT**: split long prefills into 512-token chunks interleaved with decode steps. Use when your p99 TPOT SLA is tighter than your TTFT budget.

10. **GPU utilization > 90% is the target**: below 80% usually signals static batching, KV cache fragmentation, or insufficient batch size. With vLLM properly configured, hitting 90–95% on production workloads is routine.

11. **Memory efficiency and scheduling efficiency are multiplicative**: static batching captures roughly 15% of theoretical maximum throughput (38% utilization × 40% memory efficiency). vLLM captures ~81% (90% × 90%). The 5× gap is why the paper reports 30–50× improvement over naive static batching on real workloads.

12. **The monitoring minimum**: instrument `vllm:gpu_cache_usage_perc`, `vllm:num_waiting_seqs`, `vllm:num_preemptions_total`, and p99 TTFT from `vllm:time_to_first_token_seconds`. Cache usage above 95% and non-zero preemptions under normal load are your early warning signals of over-provisioned concurrency.


## Further reading

**Seminal papers**:
- Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models", OSDI 2022. The iteration-level scheduling paper that started this line of work.
- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023. The vLLM paper; the block manager design is in sections 3–4.
- Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills", 2023. The chunked prefill proposal.
- Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs", 2024. Extends RadixAttention for structured generation patterns.

**Official documentation**:
- [vLLM documentation](https://docs.vllm.ai/) — engine arguments, prefix caching, quantization, distributed serving, production deployment.
- [vLLM block manager source](https://github.com/vllm-project/vllm/blob/main/vllm/core/block_manager_v2.py) — the actual implementation; read alongside the paper for full clarity.
- [PagedAttention CUDA kernel](https://github.com/vllm-project/vllm/tree/main/csrc/attention) — the custom CUDA implementation of the attention with block table indirection.

**Within this series**:
- [Why LLM serving is different](/blog/machine-learning/model-serving/why-llm-serving-is-different) — the KV cache memory wall and autoregressive bottleneck that motivates this post.
- [vLLM deep dive](/blog/machine-learning/model-serving/vllm-deep-dive) — the next post in Track C: speculative decoding, multi-LoRA serving, and advanced chunked prefill tuning in vLLM.
- [KV cache optimization](/blog/machine-learning/model-serving/kv-cache-optimization) — prefix caching eviction policies, multi-level KV cache, and memory pressure management.
- [Batching fundamentals: latency-throughput tradeoff](/blog/machine-learning/model-serving/batching-fundamentals-latency-throughput-tradeoff) — the queuing theory and Little's Law foundation for understanding all batching strategies.
- [The model serving playbook](/blog/machine-learning/model-serving/the-model-serving-playbook) — the series capstone with the complete decision tree from model to production.

---
title: "580 Tokens a Second on a 397B Model: Inside the TokenSpeed Qwen3.5 Speed Record"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A technique-by-technique teardown of how TokenSpeed sustains ~580 tokens/second at batch size one on Qwen3.5-397B-A17B — the hybrid GDN/Mamba state machinery, the kernel fusions, the multi-stream overlap, and the CUDA-graph plumbing that takes the CPU off the decode loop."
tags:
  - tokenspeed
  - qwen3-5
  - gated-deltanet
  - hybrid-attention
  - llm-inference
  - agentic-workloads
  - blackwell
  - speculative-decoding
  - kernel-fusion
  - cuda-graphs
  - mixture-of-experts
  - nvfp4
category: "machine-learning"
subcategory: "Open Source Library"
author: "Hiep Tran"
featured: true
image: "/imgs/blogs/tokenspeed-580-tps-qwen3-5-hybrid-mamba-blackwell-1.webp"
readTime: 50
---

There is a number in the latest TokenSpeed post that should stop any inference engineer mid-scroll: **~580 tokens per second on Qwen3.5-397B-A17B — at batch size one.** Not aggregate. Not summed across a packed batch of two hundred users. One user, zero concurrency, a 397-billion-parameter model, one NVIDIA B200 Blackwell node in an eight-way tensor-parallel configuration, weights quantized to NVFP4. A single stream of tokens leaving the GPU faster than most engines can manage when they are batching aggressively to hide the cost of weight loads.

That distinction is the whole story. Batch-size-one throughput is the cruelest benchmark in serving because every trick the field has spent five years perfecting — wide batches, continuous batching, packing the matmul tiles until the Tensor Cores saturate — is unavailable to you. At batch one there is nothing to amortize against. The decode step is brutally memory-bound, the GPU spends most of its time waiting on HBM, and the CPU's per-kernel launch overhead, normally invisible behind a fat batch, becomes a first-order tax. Hitting 580 tok/s in that regime is not a throughput win. It is a **latency** win, and latency wins come only from eliminating stalls one at a time until there are none left to remove.

![The 580 tok/s record sits on four stacked optimizations](/imgs/blogs/tokenspeed-580-tps-qwen3-5-hybrid-mamba-blackwell-1.webp)

The diagram above is the mental model for the rest of this article: the record is not one trick, it is four stacked layers of stall-removal, each attacking a different part of the decode step, sitting on top of a Blackwell-plus-NVFP4 substrate. From the bottom up — hybrid GDN/Mamba state management that makes the agentic prefix essentially free; multi-stream overlap that hides the shared-expert compute behind the routed-expert GEMM; kernel fusion that collapses five HBM round-trips into one register-resident kernel; and an async-CPU layer that captures the entire decode loop into a CUDA graph and replaces every device-to-host scalar read with an on-device sentinel. Pull out any one layer and the number drops. This is a teardown of all four, plus the benchmark methodology that produced the headline, written for engineers who serve these models for a living.

A word on provenance before we trust any of it. TokenSpeed is an early-preview, MIT-licensed engine from the **LightSeek Foundation**, built with collaborators including the Alibaba Tongyi team (who train Qwen), NVIDIA DevTech, and the Mooncake team. The numbers come from the vendor's own blog and an EvalScope harness, on hardware most of us cannot rent yet. Treat them as a credible engineering report from people who run these models in anger, not as an independently reproduced benchmark. The *techniques*, which is what we care about, are the durable part — every one of them is a transferable lesson about serving hybrid-attention MoE models on Blackwell, and they will outlive this particular leaderboard entry. (For the engine's broader four-layer architecture and its MLA kernels on the DeepSeek-shaped models, see the companion piece, [TokenSpeed: inside a speed-of-light inference engine](/blog/machine-learning/open-source-library/tokenspeed-agentic-inference-engine); this post is specifically about the Qwen3.5 hybrid path and the 580 tok/s run.)

## 1. The workload: why batch size one is the hard number

**Senior rule of thumb: if your benchmark batches to hide latency, it is measuring your finance team's cost per token, not your user's experience.**

The reason this record matters is that the workload it targets has quietly become the dominant one. A coding agent — Claude Code, Codex, Cursor, an in-house planner-executor loop — does not behave like a chatbot. It arrives with **tens of thousands of tokens of accumulated context** before it asks for its first new token; it runs for **dozens of turns**, each appending a tool result or a diff to the same growing conversation; and on the other end a human (or a downstream agent blocked on this one) is waiting for each token to appear. The metric that decides whether that loop feels alive is **per-user tokens per second**, and the only way to push it up is to make a single decode step finish faster.

Here is the mismatch laid out against the assumptions that chatbot-tuned engines were built on. None of these assumptions is wrong for chatbots; all of them mislead you for agents.

| Assumption | The naive view | The reality for agents |
|---|---|---|
| Throughput is the metric. | Maximize fleet-wide tokens/sec by batching wide. | Each user needs a per-user TPS floor; a fleet record is worthless if any one stream stalls. |
| Prompts are short, prefill is cheap. | Decode dominates; prefill is a one-time rounding error. | Agentic first turns run 50K+ tokens and grow every turn — prefill and cache pressure are first-order. |
| Decode is memory-bound, so widen the batch. | Bigger batch → better Tensor-Core fill → more tok/s. | At batch one there is no batch to widen; you must make the single stream itself fast. |
| The CPU is free behind a fat batch. | Per-kernel launch overhead hides under compute. | At batch one the GPU finishes each kernel so fast the host dispatch becomes the bottleneck. |

The thread tying these together is that batch-size-one decode is a **stall-elimination** problem, not a **utilization** problem. You are not trying to keep 132 Blackwell SMs busy on a fat matmul — at batch one and 17B active parameters, the per-step FLOPs are modest. You are trying to ensure that between one token leaving the sampler and the next token's first kernel launching, *nothing* on the critical path is idle: not the GPU waiting on a CPU branch, not the CPU waiting on a device-to-host copy, not the routed-expert GEMM waiting on the shared expert, not the attention kernel waiting on a memory copy of recurrent state. Every section below is one class of stall, and the engine's job is to drive each to zero.

> An engine that wins the throughput benchmark by batching wide can still feel broken to every individual user. The agentic record is a constrained optimization, and the constraint — per-user latency — is exactly the thing aggregate throughput throws away.

### The latency budget, in numbers

It is worth doing the arithmetic, because it tells you both how hard 580 tok/s is and how much headroom remains. A rate of 580 tokens per second means **1.72 ms per token**. Across roughly 60 decoder layers, that is about **29 µs per layer per token** — and into that 29 µs you have to fit attention (GDN or full), the MoE GEMM, the norms, the all-reduce, the sampler, and a slice of the draft model. There is no room for a stray synchronize.

Now the floor. At batch one, decode is bandwidth-bound on weight loads. With 17B active parameters at NVFP4 (4 bits, ½ byte each), a single token streams about $17\times10^9 \times 0.5 = 8.5\ \text{GB}$ of weights. A B200's HBM delivers on the order of 8 TB/s, so the pure weight-streaming time is:

$$t_{\text{floor}} = \frac{8.5\ \text{GB}}{8\ \text{TB/s}} \approx 1.06\ \text{ms} \;\Rightarrow\; \sim 940\ \text{tok/s}.$$

So the bandwidth-bound ceiling for this model on this hardware is roughly 940 tok/s, and 580 tok/s is about **62% of that ceiling** — with the remaining gap consumed by the KV/state reads, the kernels' arithmetic, and whatever stalls survive. That single ratio reframes the entire engineering effort: the four optimization layers are a campaign to close the distance between the measured 580 and the theoretical 940, one stall at a time, and the un-closed ~38% is exactly why FlashAttention-4 and further fusion still have room to push the number up.

There is one more reason batch one is the right target for *this* model specifically, and it is architectural. Qwen3.5-397B-A17B activates only ~17B of its 397B parameters per token (the A17B suffix), so the per-token compute is small relative to the memory you must stream. That makes it gloriously cheap per token *if* you can keep the pipeline fed — and pathologically stall-sensitive if you cannot. The model practically begs for a latency-first engine. TokenSpeed is built to answer.

## 2. The model underneath: Qwen3.5's hybrid GDN and sparse MoE

**Senior rule of thumb: you cannot tune an engine for a model whose memory state you have not drawn on a whiteboard.**

Everything TokenSpeed does is shaped by an architectural fact: Qwen3.5 is not a pure Transformer. It is a **hybrid** that interleaves two completely different token-mixing layers, and they keep two completely different kinds of per-token state.

![Two token-mixers, two memory systems](/imgs/blogs/tokenspeed-580-tps-qwen3-5-hybrid-mamba-blackwell-2.webp)

Roughly three of every four layers are **Gated DeltaNet (GDN)** linear-attention layers; the remaining one in four is a standard **full-attention** layer. That 3:1 ratio is the default Qwen3.5 pattern — three linear layers, then one full-attention layer, repeat. The two layer types differ on nearly every axis that matters to a kernel author, as the matrix above lays out, and the consequences ripple through the entire engine.

A **GDN layer** is a linear-attention recurrence. Instead of a softmax over all past keys, it maintains a fixed-size recurrent state and updates it token by token. Gated DeltaNet specifically combines four ingredients that are worth naming because they each leave a fingerprint on the state you must manage:

- A **delta rule** for error-correcting memory updates — the layer writes the *difference* between what it would retrieve and the new value, which is what gives linear attention competitive retrieval quality.
- **Exponential gating** for adaptive decay, so the recurrent memory does not saturate over a long context.
- A **causal `Conv1D`** for local mixing, which is why the per-layer state has a `conv_state` component — a short rolling buffer of recent activations.
- **L2 normalization on Q and K** in place of the softmax normalizer.

The practical upshot is that a GDN layer's entire memory of the past is two small, *fixed-size* tensors per request: a `conv_state` (the convolution's rolling window) and an `ssm_state`/`temporal_state` (the recurrent summary). Crucially, that size does **not** grow with context length. For Qwen3.5-397B-A17B the GDN layers use 64 heads for the value path and 16 for the query/key path at head dimension 128.

A **full-attention layer** is the opposite. It uses grouped-query attention — 32 query heads sharing 2 key/value heads — at a head dimension of **256**, and it keeps a conventional KV cache that grows linearly with the sequence. Head dimension 256 is not a footnote; it is double the usual 128 and it places real demands on the attention kernel's register and shared-memory budget, which is exactly why FlashAttention-4 support for `head_dim=256` is on TokenSpeed's roadmap (more on that later).

So the model hands the engine two memory subsystems with opposite cost curves. The GDN state is small and constant but *stateful and mutable* — you cannot page it like a KV cache, and you cannot recompute it cheaply because it is a recurrence. The KV cache is large and growing but *append-only and pageable* — the well-trodden territory every modern engine already handles. An engine that treats Qwen3.5 as "a Transformer with some weird layers" will mismanage the GDN state and leave most of the latency win on the table. TokenSpeed treats the two as first-class peers, and that decision is the root of nearly every design choice that follows. If the recurrence itself is unfamiliar, the [Nemotron-H hybrid Mamba-Transformer](/blog/machine-learning/large-language-model/nemotron-h-hybrid-mamba-transformer) writeup walks through why linear-attention state behaves so differently from a KV cache.

### The recurrence, in one equation

The reason the GDN state stays constant-size is worth making precise, because it is the property the whole engine exploits. A linear-attention layer maintains a state matrix $S_t \in \mathbb{R}^{d_k \times d_v}$ and updates it per token with a gated delta rule of the shape

$$S_t = \big(\alpha_t\, S_{t-1}\big) + \beta_t\, k_t \big(v_t - S_{t-1} k_t\big)^{\!\top}, \qquad o_t = S_t\, q_t,$$

where $\alpha_t$ is the exponential gate (adaptive decay), $\beta_t$ is the delta-rule write strength, and $k_t, v_t, q_t$ are the per-token key, value, and query after the causal `Conv1D` and L2 normalization. The output $o_t$ is read straight out of the current state. Notice what is *absent*: there is no sum over past positions, no $O(t)$ scan at decode time, and no tensor whose size depends on $t$. The state $S_t$ has the same shape at token 1 and at token 1,000,000. That is the entire reason the long-context benchmark in Section 10 decays by only 16% out to a million tokens — three of every four layers carry a memory of the past that costs the same to read no matter how long the past is. The full-attention layers, by contrast, compute $o_t = \text{softmax}(q_t K_{1:t}^\top) V_{1:t}$, where $K_{1:t}$ and $V_{1:t}$ grow with the context — which is why those one-in-four layers are the only ones that feel a 1M-token prompt, and the only ones that need a paged KV cache.

The serving consequence is that the two layer types want opposite things from the memory system. The GDN layer wants a small, fixed-size, *mutable* slab it can read-modify-write in place every token; the full-attention layer wants a large, *append-only*, pageable cache. An engine that offers only one of those — only paged KV, as a Transformer-tuned engine does — will bolt the GDN state onto the wrong kind of allocator and pay for it in copies and stalls. That mismatch is the seed of every section that follows.

### Second-order consequence: quantization meets two state types

The benchmark model is `Qwen3.5-397B-A17B-NVFP4` — weights in NVIDIA's 4-bit floating-point format. NVFP4 shrinks the weight footprint roughly 4× versus FP16, which at batch one directly buys decode speed because decode is bandwidth-bound on weight loads. But the two state types react differently to a 4-bit regime: the KV cache can be quantized with well-understood techniques, while the GDN recurrence is numerically touchier — a recurrence accumulates error, so its state typically stays in higher precision even when the weights are NVFP4. If you want the broader picture of why 4-bit is the current frontier and where it breaks, the [past the 4-bit wall](/blog/machine-learning/large-language-model/past-4-bit-wall-frontier-llm-quantization) deep-dive covers the numerics. The point here is narrower: quantization is not uniform across a hybrid model, and the engine has to know which state lives where.

## 3. Two memory systems, one scheduler

**Senior rule of thumb: the scheduler is where a hybrid model's two memory systems either compose cleanly or corrupt each other.**

Because the model keeps two kinds of state, the scheduler manages **two separate resource pools** in lockstep, and routes every layer's forward call to the right one.

![One scheduler, routed by layer_id into two pools](/imgs/blogs/tokenspeed-580-tps-qwen3-5-hybrid-mamba-blackwell-3.webp)

The first pool is the familiar **KV-cache pool**: block indices for the paged KV cache that the full-attention layers read and append to. (If paged KV allocation is new, our [KV cache optimization and management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management) post covers the page-table mechanics this builds on.) The second is the **Mamba state pool**: a flat array of slot indices — TokenSpeed calls them `mamba_pool_indices` — where each slot holds one request's `conv_state` and `ssm_state` for the GDN layers. The distinction in how the two are *addressed* is the crux: KV caches are indexed through page tables (a request's logical position maps through a table to physical pages), whereas Mamba states are indexed by **flat slot IDs** handed out directly by the scheduler. One is a virtual-memory system; the other is a slab allocator. Conflating them is how subtle corruption bugs are born.

A `HybridLinearAttnBackend` sits at the dispatch point. Each decoder layer's `forward` is called with its `layer_id`; the backend reads the id, decides whether this is a GDN layer or a full-attention layer, and routes to the matching kernel path with backend-specific metadata initialized separately for each type. A request's lifecycle threads through both pools at once:

```python
# Illustrative reconstruction of the dual-pool request lifecycle.
# The real engine splits a C++ control plane from a Python execution plane;
# this captures the state transitions, not the exact API.

class HybridRequest:
    def on_arrival(self, sched):
        # One Mamba slot reserved up front for the whole request lifetime.
        self.mamba_slot = sched.mamba_pool.alloc()          # flat slot id
        self.kv_blocks  = []                                # paged, grows on demand

    def prefill(self, sched, prompt_tokens):
        # Either populate GDN state from scratch, or copy-on-write from a
        # prefix-cache checkpoint (Section 4). KV blocks fill page by page.
        sched.kv_pool.reserve(self, n_blocks=ceil(len(prompt_tokens) / BLOCK))
        run_prefill(self.mamba_slot, self.kv_blocks, prompt_tokens)

    def decode_step(self, sched):
        # GDN layers mutate conv/ssm state in place; full-attn layers append KV.
        run_decode(self.mamba_slot, self.kv_blocks)         # one token out

    def on_finish(self, sched):
        sched.mamba_pool.free(self.mamba_slot)              # release both pools
        sched.kv_pool.free(self.kv_blocks)
```

The non-obvious hard part is that the two pools have different *liveness semantics*. A KV block, once written, is immutable for the life of the sequence — append-only. A Mamba slot is **mutated in place on every decode step**: the recurrence reads the slot, updates it, and writes it back. That mutability is what makes the GDN state both cheap (constant size) and dangerous (every read-after-write must be ordered correctly across CUDA streams). Hold that thought — it is the reason the prefix cache in the next section needs an entire copy-on-write protocol that a pure-KV engine would never bother with.

### Second-order consequence: speculative decoding needs a state snapshot

The scheduler also has to support speculative decoding over a recurrence, which is genuinely harder than over a KV cache. With a KV cache you can speculate forward and simply not commit the extra blocks if verification rejects them. With a recurrence you have already *mutated* the state to produce the draft tokens, so a rejection means you have to roll the state back. TokenSpeed's scheduler maintains a `spec_cache` that snapshots the `conv`/`ssm` state for each speculative step precisely so a verification failure can restore the pre-draft state. The naive way to do that is ruinously expensive, and fixing it is the subject of Section 5.

## 4. The agentic killer feature: hybrid prefix cache

**Senior rule of thumb: in agentic serving, the cheapest token is the one you never recompute — and the second cheapest is the state you never re-derive.**

Here is the move that makes the agentic workload tractable. An agent's turns share an enormous prefix: the 50K-token first turn, then turn two is that same 50K plus 800 new tokens, turn three is 50.8K plus another 800, and so on. If you re-prefill the shared prefix every turn you are doing tens of thousands of tokens of redundant work before the model emits anything — and at that point your "per-user TPS" is dominated by prefill latency the user should never have paid twice. KV-cache prefix sharing solves this for the full-attention layers. But the GDN layers have *recurrent state*, and you cannot share a recurrence by pointing two requests at the same physical KV pages, because the moment one request decodes a token it mutates the state the other is reading.

![Hybrid prefix cache: copy-on-write off a clean checkpoint](/imgs/blogs/tokenspeed-580-tps-qwen3-5-hybrid-mamba-blackwell-4.webp)

TokenSpeed's **hybrid prefix cache** solves both halves at once, with a two-layer design. A **C++ layer** does the bookkeeping: radix-tree matching of shared prefixes, page IDs, eviction, and Mamba-slot lifetime management. A **Python layer** does the GPU work: managing KV pages and the `conv_state`/`ssm_state` tensors, stream ordering, copy-on-write, zeroing, and snapshot copies. The protocol hinges on a distinction between two kinds of Mamba slot per request:

- A **working slot** — the mutable state for the current forward step, which the recurrence writes to every token.
- A **checkpoint slot** — a snapshot taken at a block-aligned boundary, destined to be published into the prefix tree so a *future* request can reuse it.

The correctness invariant that holds the whole thing together is stated plainly in the design: **every Mamba slot reachable from the prefix tree contains a clean, block-aligned state — never an arbitrary intermediate.** A slot is only ever published at a block boundary, so the tree never advertises a half-updated recurrence. When a new request arrives and its prefix matches a cached node, the scheduler returns a `mamba_cow_src_index`, and the Python layer **copy-on-writes** the cached checkpoint into the request's private working slot. The cached tree slot stays immutable; only the private copy is mutated as the new request decodes. Two requests can share the same 50K-token prefix and then diverge without ever corrupting each other's recurrence.

There is a second safety rail. A freshly allocated slot is only ever used along one of two provably-clean paths: either it received a copy-on-write copy from a known-clean checkpoint, or the Python layer **explicitly zeroed it** before first use. There is no third path where a slot inherits whatever garbage a previous request left behind — that is what would silently poison generation quality in a way no unit test catches until a customer reports nonsense after turn nine.

The hardest case is **chunked prefill under overlap scheduling**, where the engine is publishing checkpoints into the tree *while* a later chunk is already executing on another stream. The ordering discipline is precise: chunk N writes its state on the execution stream; the default stream waits on the execution stream; the C++ layer inserts the previous chunk into the tree and detaches its checkpoint slot so it is never reused; and the next chunk gets a fresh checkpoint slot. The invariant the engineers protect is that even though C++ publishes the checkpoint during the overlap window, every downstream GPU consumer is ordered *after* the snapshot write — so no reader ever sees a slot mid-flight.

```python
# Illustrative: copy-on-write fork of a GDN checkpoint into a private slot.
def admit_with_prefix_cache(req, tree, mamba_pool, stream):
    node = tree.longest_prefix_match(req.token_ids)        # radix-tree match
    req.kv_blocks = node.kv_pages[:]                       # share KV pages (read-only)
    if node.mamba_checkpoint is not None:
        src = node.mamba_checkpoint                        # clean, block-aligned
        dst = mamba_pool.alloc()                           # private working slot
        with torch.cuda.stream(stream):
            dst.conv.copy_(src.conv, non_blocking=True)    # CoW: copy, don't alias
            dst.ssm.copy_(src.ssm,  non_blocking=True)
        req.mamba_slot = dst                               # only the copy is mutated
    else:
        req.mamba_slot = mamba_pool.alloc().zero_()        # the only other clean path
    return req
```

It is worth quantifying what the cache saves. Take the benchmark's own shape: a 50K first-turn context, 800 tokens appended per turn, 15 turns. Without prefix reuse, turn $n$ must prefill the entire accumulated context — turn 2 re-processes ~50.8K tokens, turn 3 ~51.6K, and so on. Summed across 15 turns that is roughly **$15 \times 50\text{K} \approx 750\text{K}$ tokens of prefill**, almost all of it redundant recomputation of context the engine already processed moments earlier. Each of those turns produces only 800 new tokens, so without the cache the user pays for a 50K-token prefill to get 800 tokens back — a ~98% tax on every turn after the first. With a >90% hit rate, that 750K collapses to roughly the 50K of the genuinely-new first turn plus the ~12K of per-turn appends; the rest is served from reused KV pages and copy-on-written GDN checkpoints. The cache is not a nice-to-have optimization on this workload — it is the difference between an interactive agent and one that visibly re-reads its entire history before every reply.

The payoff is the number that makes the agentic benchmark work at all: a **KV-cache hit rate above 90%** on the multi-turn tool-call workload. More than nine times in ten, a turn's shared context is served from the cache — KV pages reused, GDN state copy-on-written from a checkpoint — instead of being recomputed from scratch. That is the difference between an agent that feels instantaneous between turns and one that visibly thinks for a second before every reply.

### Second-order consequence: eviction has to evict two things

A subtle operational hazard: when the radix tree evicts a stale branch under memory pressure, it must free *both* the KV pages and the Mamba checkpoint slot, and it must do so without yanking a slot out from under a request that copy-on-wrote from it microseconds ago. The C++ lifetime manager is what keeps the slab allocator and the page allocator evicting in agreement. Get this wrong and you get a use-after-free that manifests as occasional garbage tokens — the worst kind of bug, because it is rare, non-deterministic, and invisible to throughput dashboards.

## 5. Speculative decoding without the state-copy tax

**Senior rule of thumb: when a fast path forces a big memory copy, the copy is the new bottleneck — move pointers, not data.**

Speculative decoding (here, multi-token prediction, MTP) is the single biggest latency lever at batch one, and we will quantify its gains in Section 10. But applying it to a recurrence creates a nasty cost that does not exist for a pure Transformer, and the way TokenSpeed removes that cost is one of the most elegant ideas in the whole engine.

Recall the problem from Section 3: to verify $k$ speculative tokens, the engine runs the recurrence forward through all $k$ drafts, buffering the intermediate states; then the target model verifies; then the accepted prefix's state must become the new canonical state. The naive implementation **copies the accepted state back** into the working slot — and that copy is over the full recurrent state, $O(L \cdot D)$ in the number of layers $L$ and state dimension $D$. You pay that copy on *every* speculative round. At batch one, where you are counting microseconds, a full-state memcpy per step is exactly the kind of "innocent" cost that quietly eats your latency budget.

![Speculative decoding without the state-copy tax](/imgs/blogs/tokenspeed-580-tps-qwen3-5-hybrid-mamba-blackwell-5.webp)

TokenSpeed's fix is **index indirection**: move pointers, not data. The state buffer is extended so that beyond the scheduler-allocated base slots there is a per-request **draft region** — a handful of extra rows reserved for speculative steps. A lightweight index table, `current_input_indices[req]`, records *which physical row* currently holds the canonical Mamba state for each request. The recurrence kernels are taught to read and write through this indirection:

- **Input redirection.** Each kernel reads its initial state from the row named by `current_input_indices[req]`. No data is moved to position it; the kernel just dereferences an index.
- **Output routing.** A companion `output_state_indices` tensor tells the kernel where to write each step's result — slot 0 is the working row, slots 1..N are the request-private draft rows. The kernel writes directly to its pre-assigned destination.
- **Post-verify bookkeeping.** Once the target verifies and accepts up to token $j$, the engine sets `current_input_indices[req]` to the draft row holding token $j$'s state. That is a single integer write — $O(1)$ — instead of an $O(L \cdot D)$ tensor copy.

```python
# Illustrative: O(1) accept instead of an O(L·D) state copy-back.
def commit_speculative(req, accepted_len, current_input_indices, draft_rows):
    # draft_rows[j] is the physical state row produced after the j-th draft token.
    winner_row = draft_rows[accepted_len - 1]   # last accepted token's state row
    current_input_indices[req] = winner_row      # <-- one integer write, no memcpy
    # The next decode step reads its initial state from winner_row via the
    # same indirection; nothing is copied. Rejected draft rows are simply
    # overwritten on the next round.
```

The result is that the intermediate cache disappears entirely and the per-step speculative overhead collapses from a full-state copy to a pointer update. It is the same idea that makes copy-on-write filesystems and persistent data structures fast — never copy what you can re-point — applied to the one place in a hybrid model where everyone else copies. Combined with the MTP gains in Section 10, this is why speculative decoding is a *net* win at batch one rather than a wash eaten by bookkeeping.

### Second-order consequence: the draft region sizes the slab

The draft region is not free — it is extra rows in the Mamba state pool, multiplied across every in-flight request. Sizing it is a real tradeoff: too few rows and you cap the speculative depth; too many and you waste the slab capacity that bounds your max concurrency. This is one of those parameters that looks like a constant in the code and is actually a load-bearing capacity-planning decision, the kind you only tune correctly after watching the pool occupancy under a real agent fleet.

## 6. Overlap is all you need: multi-stream parallelism

**Senior rule of thumb: two independent kernels that run back-to-back are a scheduling bug; if they do not depend on each other, they should be running on different streams.**

At batch one, the GPU finishes each individual kernel quickly, which means the *gaps between kernels* and any *serialized-but-independent work* dominate. The Qwen3.5 MoE layer hands TokenSpeed a gift here: it has both **shared experts** (always active) and **routed experts** (selected by top-k), and the two are naturally independent. There is no reason to run them one after the other.

<figure class="blog-anim">
<svg viewBox="0 0 720 340" role="img" aria-label="Two GPU streams run the same decode step concurrently; the auxiliary stream's shared-expert work finishes before the main stream's routed-expert GEMM, so its latency is hidden" style="width:100%;height:auto;max-width:820px">
<style>
.a6-ttl{font:600 19px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.a6-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.a6-note{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.a6-track{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.a6-fill{fill:var(--accent,#6366f1);transform-box:fill-box;transform-origin:left center}
.a6-head{stroke:var(--accent,#6366f1);stroke-width:3}
.a6-sync{fill:var(--accent,#6366f1)}
.a6-syncg{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
@keyframes a6-main{0%{transform:scaleX(0);opacity:0}6%{opacity:1}14%{transform:scaleX(0)}68%{transform:scaleX(1)}90%{transform:scaleX(1);opacity:1}98%{opacity:0}100%{transform:scaleX(1);opacity:0}}
@keyframes a6-aux{0%{transform:scaleX(0);opacity:0}6%{opacity:1}14%{transform:scaleX(0)}46%{transform:scaleX(1)}90%{transform:scaleX(1);opacity:1}98%{opacity:0}100%{transform:scaleX(1);opacity:0}}
@keyframes a6-sweep{0%{transform:translateX(0);opacity:0}14%{transform:translateX(0);opacity:1}68%{transform:translateX(460px)}90%{transform:translateX(460px);opacity:1}98%{opacity:0}100%{opacity:0}}
@keyframes a6-pop{0%,60%{opacity:0}72%,92%{opacity:1}100%{opacity:0}}
.a6-m{animation:a6-main 7s ease-in-out infinite}
.a6-a{animation:a6-aux 7s ease-in-out infinite}
.a6-h{animation:a6-sweep 7s ease-in-out infinite}
.a6-p{animation:a6-pop 7s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.a6-m,.a6-a{animation:none;transform:scaleX(1);opacity:1}.a6-h{animation:none;opacity:0}.a6-p{animation:none;opacity:1}}
</style>
<text class="a6-ttl" x="20" y="34">Two GPU streams, one decode step</text>
<text class="a6-lbl" x="20" y="92">main stream — routed expert GEMM</text>
<rect class="a6-track" x="180" y="100" width="460" height="48" rx="8"/>
<rect class="a6-fill a6-m" x="180" y="100" width="460" height="48" rx="8"/>
<text class="a6-lbl" x="20" y="182">aux stream — shared expert</text>
<rect class="a6-track" x="180" y="190" width="300" height="48" rx="8"/>
<rect class="a6-fill a6-a" x="180" y="190" width="300" height="48" rx="8"/>
<text class="a6-note" x="184" y="270">aux finishes first → its latency is hidden under the GEMM</text>
<line class="a6-head a6-h" x1="180" y1="92" x2="180" y2="246"/>
<circle class="a6-sync a6-p" cx="640" cy="124" r="9"/>
<text class="a6-syncg a6-p" x="540" y="300">event sync → combine results</text>
<line class="a6-head a6-p" x1="640" y1="148" x2="640" y2="190" stroke-dasharray="4 4"/>
</svg>
<figcaption>The shared-expert forward runs on an auxiliary stream concurrently with the routed-expert GEMM on the main stream; because it is the smaller op, it completes first and its cost disappears behind the GEMM, with an event syncing the two before results combine.</figcaption>
</figure>

The figure animates the idea. The **main stream** runs the routed-expert path — top-k routing, expert dispatch, and the big MoE GEMM. On an **auxiliary stream**, concurrently, the shared expert does its forward: `gate_up → SiLU → down`, with sigmoid gating. Because the shared expert is the smaller of the two, it finishes well within the routed GEMM's runtime, and its latency is *hidden* — you pay the GEMM's time and get the shared expert for free. The two streams synchronize via CUDA events before their results combine. TokenSpeed wraps the fork/join in a `StreamFork` helper:

```python
# Illustrative reconstruction of the shared/routed expert overlap.
import torch

class StreamFork:
    def __init__(self):
        self.aux = torch.cuda.Stream()
        self.done = torch.cuda.Event()

    def run(self, hidden, router, shared_expert, routed_experts):
        # Fork: launch the shared expert on the auxiliary stream.
        self.aux.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.aux):
            shared_out = shared_expert(hidden)        # gate_up -> SiLU -> down
            self.done.record(self.aux)

        # Main stream: routing + dispatch + the big routed-expert GEMM.
        topk_ids, topk_w = router(hidden)
        routed_out = routed_experts(hidden, topk_ids, topk_w)

        # Join: main stream waits on the (already finished) shared expert.
        torch.cuda.current_stream().wait_event(self.done)
        return routed_out + shared_out
```

The same trick appears one level down, inside the GDN layer itself. Gated DeltaNet's input projection is two independent linear layers: a large `in_proj_qkvz` and a small `in_proj_ba`. TokenSpeed runs the large projection on the main stream and the small one on an alternate stream, so the small projection is fully hidden behind the large one. This overlap is activated specifically **during CUDA-graph capture** (the subject of Section 8), so the two-stream structure gets baked into the replayed graph and costs nothing to set up at runtime.

The discipline here is the senior lesson: *any* two kernels on the critical path that do not have a data dependency are a latency opportunity. The MoE shared/routed split and the GDN dual projection are the two fattest such opportunities in Qwen3.5, but the mindset generalizes — every time you see two independent ops serialized on one stream, you are leaving microseconds on the floor. For the broader MoE picture this builds on, see [optimizing MoE training and inference](/blog/machine-learning/large-language-model/optimizing-moe-training-and-inference).

### Second-order consequence: overlap competes for the same SMs

Multi-stream overlap is not a free lunch in the limit. Two concurrent kernels share the same SMs, the same L2, and the same memory controllers. At batch one this is pure win because no single kernel saturates the GPU — there is headroom for both. At high concurrency, where the routed GEMM already fills the machine, forking the shared expert onto a second stream can *contend* rather than overlap, and the win shrinks. This is the first hint of a theme that recurs in the benchmarks: the optimizations that make batch one fly are tuned for the latency-bound regime, and several of them quietly stop paying off once the batch is wide enough to saturate the hardware on its own.

There is also a correctness cost to multi-stream work that is easy to underestimate. The moment two streams touch overlapping state, you have introduced a happens-before relationship that the CUDA runtime will not enforce for you — you have to. Every `StreamFork` needs its matching event wait, and every event wait has to be on the *right* event recorded at the *right* point; a fork without a correctly placed join does not error, it produces a data race that surfaces as occasional wrong numbers under load. The GDN dual-projection overlap is the more delicate of the two because the two projections feed the same downstream recurrence, so their join has to complete before the state update reads either result. The reason TokenSpeed activates these overlaps specifically during CUDA-graph capture is partly performance and partly safety: capturing the fork/join structure into the graph freezes the ordering into the replayed unit, so the synchronization cannot be accidentally dropped by a later code path. Overlap is a latency win and a concurrency-bug surface in equal measure, and the engineering is as much about the events as the streams.

## 7. The more you fuse, the less latency: kernel fusions

**Senior rule of thumb: at batch one, decode is bandwidth-bound, so every kernel that writes an intermediate to HBM and reads it back is paying a tax you can often delete entirely.**

A small kernel that loads a tensor from HBM, does a cheap elementwise op, and writes it back is almost pure memory traffic — the arithmetic is trivial, the round-trip to HBM is the cost. String five of those together and you have made five round-trips to move data that never needed to leave the chip. Fusion collapses the chain into one kernel that loads once, does all the work in registers, and writes once.

![Fusing QK-norm, RoPE and gate-split into one kernel](/imgs/blogs/tokenspeed-580-tps-qwen3-5-hybrid-mamba-blackwell-7.webp)

The marquee example is the attention pre-amble. Naively, producing the query and key tensors for an attention layer is five separate launches, each bouncing its result through HBM: the Q/K GEMM writes `q` to HBM; a Q-RMSNorm kernel reads it back, normalizes, writes it again; a RoPE kernel reads, rotates, writes; the same dance for K; and finally a gate-split-plus-contiguous-copy writes the gated tensor out. Five kernels, five HBM round-trips, for a sequence of cheap pointwise transforms. TokenSpeed's `fused_qk_rmsnorm_rope_gate` Triton kernel does all of it in one launch — the intermediates stay in registers, there are zero wasted HBM trips, and the result is written exactly once.

![Three fusions, three parts of the decoder layer](/imgs/blogs/tokenspeed-580-tps-qwen3-5-hybrid-mamba-blackwell-8.webp)

That is one of three fusions, each attacking a different cluster of launches in the decoder layer, summarized in the grid above and the table below:

| Fusion | What it merges | Before → after | Where it applies |
|---|---|---|---|
| Gemma AllReduce + residual + RMSNorm | The tensor-parallel all-reduce, the residual add, and the RMSNorm into one kernel | 3 launches → 1 | Every Qwen3.5 decoder layer; auto-enabled on SM90+ single-node TP |
| Fused QK-RMSNorm + partial RoPE + gate split | Q/K norm, rotary embedding, and the gate split, register-resident | 5 launches → 1 | Attention pre-amble |
| Fused gate-sigmoid-mul-add | The MoE shared-expert gating: $\text{final} \mathrel{+}= \sigma(\sum_i h_i w_i)\cdot \text{shared}$ | 5 launches → 1 | MoE shared-expert path |

The first fusion has a delightful wrinkle worth dwelling on, because it shows how a tiny numerical detail can block a fusion until someone notices. Standard fused AllReduce+RMSNorm kernels compute $x \cdot \text{weight}$. But **GemmaRMSNorm** computes $x \cdot (1 + \text{weight})$ — the extra $+1$ is baked into Gemma-lineage norm layers and it does not fit the standard kernel's contract, so for a long time these layers could not use the fused path. The fix is almost insultingly simple: precompute $\text{gemma\_weight} = \text{weight} + 1.0$ once at load time, hand *that* to the standard fused kernel as its $\gamma$ parameter, and the arithmetic lines up. Three kernel launches become one across every decoder layer, automatically on SM90-and-newer single-node tensor-parallel deployments.

```python
# Illustrative: the +1 reparameterization that unblocks the AllReduce+RMSNorm fusion.
# GemmaRMSNorm computes  x * (1 + weight); the standard fused kernel wants x * gamma.
# Fold the +1 into gamma once at load time, then the layer takes the fused path.
def fold_gemma_rmsnorm(weight: torch.Tensor) -> torch.Tensor:
    return weight + 1.0     # gemma_weight; pass as gamma to fused_allreduce_residual_rmsnorm

# Hot path (per layer), now a single fused launch instead of three:
#   hidden = fused_allreduce_residual_rmsnorm(hidden, residual, gemma_weight)
```

The third fusion targets the MoE shared expert's gating arithmetic. Naively it is an elementwise multiply (`h[i] * w[i]`) written to HBM, a reduction to a per-token scalar written to HBM, a sigmoid written to HBM, a broadcast multiply against the shared-expert output written to HBM, and a final accumulate — five launches, five round-trips, for a single fused expression $\text{final} \mathrel{+}= \sigma\!\big(\sum_i h_i w_i\big)\cdot \text{shared}$. The `fused_gate_sigmoid_mul_add` kernel computes the whole thing in place within one thread block per token, intermediates never leaving registers.

None of these three fusions is individually dramatic — each saves a handful of microseconds. But decode is a loop, and these kernels run on *every layer of every token*. A few microseconds saved per layer, times ~60 layers, times every token in a 50K-context agentic turn, is the difference between 500 and 580 tokens per second. Latency engineering is the discipline of caring about microseconds because you are about to multiply them by a very large number.

### Second-order consequence: fusion fights flexibility

Every fusion is a bet that a specific sequence of ops will always appear together. The Gemma `+1` story is the warning: when a model variant changes the norm's arithmetic, the fused kernel silently stops applying (or, worse, applies incorrectly if no one guards it). Fused kernels are also harder to write, harder to debug, and Triton-version-sensitive. The engineering judgment is to fuse the hottest, most stable chains — the attention pre-amble and the per-layer norm, which every token hits — and leave the rare paths unfused. A fused kernel for a code path that runs once per request is wasted effort and a maintenance liability.

## 8. Killing the CPU: CUDA graphs and async everything

**Senior rule of thumb: at batch one, the host is on the critical path; a single `cudaStreamSynchronize` to read one scalar can cost more than the kernel you launched.**

This is the layer that separates a fast engine from a record-setting one, and it is the least glamorous: getting the CPU out of the decode loop entirely. At batch one, kernels finish so quickly that the *host* — launching kernels, reading back scalars to make branching decisions, preparing the next step — becomes the bottleneck. Every microsecond the GPU spends idle waiting on the CPU is a microsecond stolen from your token rate.

![Taking the CPU off the decode critical path](/imgs/blogs/tokenspeed-580-tps-qwen3-5-hybrid-mamba-blackwell-9.webp)

The foundational move is a **CUDA graph** that captures the entire decode loop — target model forward, sampler, and draft model — as one replayable unit. Once captured, thousands of GPU kernels replay with a *single* CPU launch, eliminating per-kernel dispatch overhead wholesale. The host stops launching a thousand kernels per token and starts launching one graph per token.

But a captured graph is only as good as the synchronization points it does *not* contain, and the subtle enemy is the **innocent device-to-host query**: any place the code reads a single scalar back from the GPU to make a CPU-side decision. Reading "how many tokens were accepted?" or "is this sequence finished?" off the device forces a `cudaStreamSynchronize`, which drains the pipeline and stalls everything behind it. TokenSpeed's playbook for eliminating these is worth memorizing because it transfers to any latency-critical CUDA codebase:

1. **Pre-compute worst-case bounds at initialization.** If a buffer's size depends on a runtime quantity, allocate for the maximum so you never need to read the actual value to size it.
2. **Capture CPU-side maxima before the H2D transfer**, so the host already knows the bound without reading it back from the device.
3. **Use GPU-side sentinel values.** Instead of reading "k tokens accepted" to the host and branching, write a sentinel into the unused slots and let downstream kernels skip invalid entries on-device. The decision tree never leaves the GPU.
4. **Keep the entire decision on the device.** No D2H read, no host branch, no sync.

On top of that, TokenSpeed uses `torch.compile` on the scheduling routines so Inductor fuses the index arithmetic — the per-step bookkeeping that manipulates slot indices and page tables — from 10–14 individual launches down to 1–2 elementwise kernels, with the index manipulation flowing through registers. And the H2D path is made fully asynchronous: pinned host memory with non-blocking copies, the host polling pinned-host counters instead of calling `synchronize()`, event-based barriers that wake only the layers that need a particular piece of state, and the CPU preparing the next batch while the current one is still in flight.

```python
# Illustrative: capture once, replay per token; no host scalar reads inside.
g = torch.cuda.CUDAGraph()
static_in  = torch.empty(..., device="cuda")      # fixed-address input
static_out = torch.empty(..., device="cuda")

# Warm up so all lazy allocations happen before capture.
for _ in range(3):
    static_out.copy_(decode_step(static_in))       # target + sampler + draft

with torch.cuda.graph(g):
    static_out.copy_(decode_step(static_in))        # thousands of kernels, captured

# Steady-state decode loop: one replay per token, zero per-kernel dispatch,
# and no device-to-host sync — finished-ness is a GPU sentinel, not a host read.
def next_token(input_ids):
    static_in.copy_(input_ids, non_blocking=True)
    g.replay()                                       # one CPU launch
    return static_out                                # consumed on-device next step
```

The cumulative effect is "asynchronous everything": the host spends its time submitting work, not waiting for it, and the GPU never idles on a CPU decision. This is the layer that takes an engine that is *fast* and makes it *record-fast*, and it is invisible in any profile that only looks at kernel times — the stalls it removes are the gaps *between* kernels, which is exactly where batch-one latency hides.

### Second-order consequence: CUDA graphs ossify shapes

CUDA graphs capture a *fixed* set of kernels operating on *fixed* memory addresses. That rigidity is the source of the speed and also the source of the pain: anything dynamic — a variable batch size, a variable sequence length, a variable number of accepted speculative tokens — breaks naive capture. This is precisely why so much engineering goes into pre-computing worst-case bounds and using sentinels: the graph must be shape-stable, so every runtime variability has to be expressed as "compute the max, mask the rest" rather than "branch on the actual value." It is a different way of writing code, and it is the price of admission for graph replay.

## 9. Scaling out: PD disaggregation that also moves Mamba state

**Senior rule of thumb: disaggregating prefill from decode is well-trodden for KV caches; doing it for a recurrence means shipping mutable state across the wire with perfect layer alignment.**

Prefill and decode have opposite hardware appetites — prefill is compute-bound and bursty, decode is memory-bound and steady — so production systems increasingly run them on **separate nodes** (PD disaggregation). The prefill node chews through the long prompt; the decode node streams tokens. For a pure Transformer, the only thing you transfer between them is the KV cache, indexed by page tables. For a hybrid model you also have to transfer the **GDN recurrent state**, and that is materially harder because the state is mutable, per-layer, and must arrive with exact layer-wise alignment or the recurrence is corrupted.

![Prefill-decode disaggregation that also moves Mamba state](/imgs/blogs/tokenspeed-580-tps-qwen3-5-hybrid-mamba-blackwell-10.webp)

TokenSpeed builds a **unified state-transfer** path for this. Each node maintains two pre-allocated contiguous GPU memory pools: a **convolutional state pool** (the short-term causal-convolution memory) and a **recurrent SSM state pool** (the long-term compressed history). Each request owns exactly one slot per layer in each pool. To move state, the nodes exchange buffer descriptors — base addresses, per-slot sizes, and layer-ID mappings — and the system maps slot indices to physical byte offsets, groups contiguous slots into scatter-gather blocks, and issues **bulk RDMA writes** with no serialization and no staging buffer. (The KV pages travel the same way, but indexed by page tables; the Mamba states by flat slot IDs — the same dual-addressing split from Section 3, now stretched across the network.) For the Mooncake-style transfer fabric this rides on, the [serving LLMs at scale](/blog/machine-learning/large-language-model/serving-llms-at-scale-production-systems) post covers the disaggregation patterns in general.

The clever part is how the transfer overlaps with compute on both ends, via a **unified step counter** that ticks after each layer's forward completes, regardless of layer type:

- On the **prefill side**, a transfer thread watches the counter. For each layer window it ships the appropriate payload — KV pages for full-attention layers, state slots for Mamba layers — so the layer-type pattern is invisible to the transfer logic; it just sends whatever the current layer produced.
- On the **decode side**, a mirror **layer-done barrier** lets the model start layer 0 before layer 15's state has even arrived. Each layer calls into the state pool and blocks only if its own state is not yet loaded, so network reception overlaps with early-layer execution instead of serializing behind it.

Finally, a **three-phase handshake** guarantees decode never starts with incomplete state or re-derives the first token:

1. **Transfer completes** — all KV pages and Mamba states for the final layer group are shipped; the transfer thread waits at a barrier.
2. **Token produced** — the prefill forward finishes and emits the first output token; the event loop records it and signals the waiting transfer thread.
3. **Status delivered** — the transfer thread sends a lightweight status, carrying the bootstrap token, over a side channel; the decode node begins only when it has received *both* the bulk state data and that token.

The invariant is that decode never begins generation with incomplete recurrent state and never has to re-derive the first token it was handed. That is the correctness floor; the layer-done barrier is the performance ceiling, letting the two nodes work concurrently rather than in a strict prefill-then-transfer-then-decode relay.

### Second-order consequence: the recurrence makes transfers unforgiving

With a KV cache, a slightly-late page is tolerable — the attention kernel for a late layer simply waits, and a missing page is a clean "not ready yet." With a recurrence, a *wrong* state is silently catastrophic: feed layer 12 the state meant for layer 11 and you do not crash, you generate plausible-looking garbage. The per-layer slot ownership and the layer-ID mappings in the buffer descriptors exist precisely to make misalignment impossible by construction. This is the recurring tax of hybrid models — every mechanism that is "append-only and forgiving" for a KV cache becomes "mutable and unforgiving" for the GDN state, and the engine pays for that difference in protocol complexity.

## 10. What the benchmarks actually say

**Senior rule of thumb: a headline number is a marketing artifact; the *shape* of the benchmark curve is the engineering truth.**

The setup, stated for reproducibility: NVIDIA B200 Blackwell GPUs, the `lightseekorg/tokenspeed-runner:latest` container, the EvalScope benchmark harness, and the `Qwen3.5-397B-A17B-NVFP4` model. Two parallelism families are tested — attention-TP with MoE-TP, and attention-TP with MoE expert-parallelism (EP). The numbers below are the vendor's; the value for us is in the curves, not the peaks.

### Decode throughput and the speculative-decoding payoff

The headline agentic result: at batch one, all four parallelism configurations — TP4, TP4EP4, TP8, TP8EP8 — sustain **500+ tok/s**, with **TP8 peaking around 580 tok/s**. Pure-TP and TP+EP land in comparable throughput-latency territory for the same GPU count, which is a genuinely useful finding: you can pick the parallelism layout for operational reasons (memory headroom, expert routing balance) without paying a throughput penalty.

The more instructive curve is **where speculative decoding (MTP) helps and where it does not.**

![Where speculative decoding (MTP) actually pays off](/imgs/blogs/tokenspeed-580-tps-qwen3-5-hybrid-mamba-blackwell-11.webp)

The matrix above is the single most important benchmark in the post, because it tells you when to turn MTP *off*:

| Regime | Batch | Output length | MTP gain |
|---|---|---|---|
| Latency-bound | 1 | any | **+100% to +159%** |
| Long-output, mid concurrency | 32–64 | > 4096 tokens | +38% to +90% |
| Throughput-bound | 64 | ~1024 tokens | ~0% or slightly negative |

The pattern is the one the Section 6 second-order note foreshadowed: MTP doubles-to-triples your speed in the latency-bound corner (small batch, where the GPU is starved and extra speculative work is free), pays off handsomely for long outputs at moderate concurrency, and goes to *zero or negative* when the batch is already throughput-bound. Once the hardware is saturated, the speculative work is no longer free — it competes for the same compute, and its verification overhead can exceed its benefit. This is a general law of speculative decoding, and the matrix is a clean empirical statement of it: **spec decoding is a latency tool, not a throughput tool.** (The [efficient LLM inference techniques](/blog/machine-learning/large-language-model/efficient-llm-inference-techniques) overview puts MTP in context with the other decode-time levers.)

### The agentic workload, specified

The agentic benchmark is built to mirror real agent traffic rather than a synthetic fixed-length sweep: a **first-turn context of 50K tokens**, **800 tokens appended per turn**, **10–15 turns total**, dominated by tool-call histories and multi-turn dialogue. The prefix cache from Section 4 achieves a **>90% KV-cache hit rate** on this pattern, which is the entire reason the per-user rate stays high across turns — without it, every turn would re-prefill tens of thousands of tokens and the user would feel a stall before each reply. Peak single-user throughput lands at **500+ tok/s on TP4** and **~580 tok/s on TP8**; at 16 concurrent agents the system delivers on the order of **1–2K tok/min/GPU**, depending on configuration.

It is worth reconciling the two throughput numbers, because they measure different things. The 580 tok/s peak is *per user* at batch one — the latency a single agent feels. The "1–2K tok/min/GPU" at 16 concurrent agents is *aggregate system* throughput, and it works out to roughly 17–33 tok/s summed across all sixteen streams — far below 16 × 580, because the engine is deliberately *not* trying to maximize aggregate throughput on this workload. It is holding each of the sixteen users above an interactive floor while they each sit on a 50K-token context, which is a far harder constraint than packing sixteen short chatbot turns into one fat batch. The two numbers are not in tension; they are the two axes of the constrained optimization from Section 1 — per-user TPS as the constraint, per-GPU aggregate as the thing maximized subject to it. An engine tuned only for the second number would batch these sixteen agents wide and let each one's latency sag, which is exactly the failure mode the whole design exists to avoid.

The shape of this workload — shallow requests on deep context (800-token outputs on 50K-token prefixes) — is precisely why the engine pours its budget into the *decode* path and the *prefix cache* rather than raw prefill throughput. The work is mostly remembering, not computing.

### Long context: graceful decay to 1M

The final benchmark stresses context length with a needle-in-a-haystack (NIAH) test at 128K, 256K, 512K, and 1M tokens, TP8:

| Context length | Decode throughput | Degradation vs 128K |
|---|---|---|
| 128K | ~530 tok/s | baseline |
| 256K | ~495 tok/s | −6.6% |
| 512K | ~470 tok/s | −11.3% |
| 1M | ~445 tok/s | −16% |

A 16% decode-throughput decay from 128K to **1M** tokens is the quiet vindication of the hybrid architecture. In a pure Transformer, decode cost grows with the full KV cache, and a 1M context would crush per-token latency. Here, three of every four layers are GDN with *constant-size* state, so only the one-in-four full-attention layers feel the context growth — and the decay stays gentle. The hybrid design is not just a training-efficiency story; it is a serving-latency story, and this curve is the proof.

### The next unlock: FlashAttention-4 at head_dim 256

One number is conspicuously *not* yet realized: native FlashAttention-4 for the full-attention layers. The `head_dim=256` support has been contributed and merged upstream, but native FA4 for Qwen3.5 inside TokenSpeed is still under active development. When it lands, it should unlock more of Blackwell's compute on exactly the layers that feel long context — the one-in-four full-attention layers — which is the most plausible path to pushing past 580. The record, in other words, is the engineers' own floor, not their ceiling.

## Case studies from production

These are the failure modes and lessons that live underneath the techniques above — the kind of thing that does not show up in a benchmark table but decides whether the engine survives contact with a real agent fleet. Several are reconstructed from the design's stated invariants; all are the genuine hazards of serving a hybrid-attention MoE model at batch one.

### 1. The innocent scalar read that halved the token rate

The symptom: a captured decode graph that benchmarked beautifully in isolation but ran at half the expected rate in the serving loop. The wrong first hypothesis was kernel inefficiency — someone spent a day profiling the attention kernel. The actual root cause was a single line that read the accepted-token count back to the host to decide how many tokens to append, forcing a `cudaStreamSynchronize` on every step that drained the pipeline behind the graph replay. The fix was to write the accepted count as a GPU sentinel and let the append kernel mask invalid slots on-device, removing the D2H read entirely. The lesson, and the reason Section 8 exists: at batch one, a synchronize is more expensive than the kernel it guards, and the worst stalls are the ones that never appear in a kernel profile because they live in the gaps between kernels.

### 2. The turn-nine garbage tokens

The symptom: an agent that produced perfect output for eight turns and then, occasionally, emitted nonsense on a later turn — non-deterministically, never reproducible in a unit test. The wrong hypothesis was a sampling bug. The actual root cause was a Mamba working slot that had been allocated and used *without* either a copy-on-write from a clean checkpoint or an explicit zero — it inherited stale recurrent state from a previous request that had occupied the slot. Because the GDN state is a recurrence, the corruption did not crash; it slowly steered generation off the rails as the bad state propagated. The fix is the Section 4 invariant: a slot is only ever used along one of two provably-clean paths. The lesson: mutable per-request state in a pooled allocator is a use-after-free waiting to happen, and the only defense is a hard invariant that every allocation is clean by construction.

### 3. The spec-decode regression nobody expected

The symptom: a team enabled MTP across the board to chase the +159% batch-one number and watched their *throughput* at high concurrency drop. The wrong hypothesis was a bug in the speculative verifier. The actual root cause was the Section 10 matrix made real: at batch 64 with short outputs, the speculative work competed for compute the routed-expert GEMM was already using, and the verification overhead exceeded the acceptance benefit — MTP went slightly negative. The fix was to gate MTP on the operating regime: on at low batch, off when the batch is throughput-bound. The lesson: speculative decoding is a latency tool, and applying a latency tool to a throughput-bound workload makes things worse, not better. Measure the curve, not the peak.

### 4. Head dimension 256 and the under-filled tile

The symptom: the full-attention layers ran slower than a back-of-envelope FLOP count predicted, even on Blackwell. The wrong hypothesis was memory bandwidth. The actual root cause was head dimension 256 — double the usual 128 — straining the attention kernel's register and shared-memory budget so the kernel could not tile the problem efficiently, leaving Tensor Cores under-fed. This is exactly the gap that the upstreamed FA4 `head_dim=256` support targets. The lesson: a model's hyperparameters are an interface contract with your kernels, and an unusual head dimension can silently leave compute on the floor until a kernel is written specifically for it. Until then, the cost is real and it hides inside layers that look like ordinary attention.

### 5. The copy-on-write that became a copy storm

The symptom: under a burst of new agents all sharing the same 50K system prompt, latency spiked instead of benefiting from the shared prefix. The wrong hypothesis was lock contention in the radix tree. The actual root cause was that every admitted request copy-on-wrote the full GDN checkpoint into a private slot at once, and a thundering herd of simultaneous admissions turned the "cheap" CoW copies into a bandwidth storm on the state pool. The fix was to stagger admissions and let the copies pipeline on the copy stream rather than firing them all on one event. The lesson: copy-on-write is cheap *per copy* but not *per stampede*; the prefix cache turns recomputation into copying, and copying has its own saturation point you have to schedule around.

### 6. The disaggregation handshake race

The symptom: a disaggregated deployment that, rarely, produced a first token inconsistent with the prompt. The wrong hypothesis was a tokenizer mismatch between nodes. The actual root cause was a decode node that began generating after the bulk RDMA state transfer completed but *before* the bootstrap token arrived over the side channel, so it re-derived its own first token from the transferred state — and a tiny numerical difference between prefill and decode nodes produced a different token. The fix is the Section 9 three-phase handshake: decode waits for *both* the state and the token. The lesson: in disaggregation, "the data arrived" and "I have permission to start" are two different events, and conflating them is a race that only shows up under production timing.

### 7. The EP-versus-TP wash that wasn't a wash everywhere

The symptom: a team read "TP and TP+EP are comparable" and switched to expert-parallelism for its memory headroom, then saw routing imbalance hurt tail latency. The wrong hypothesis was that the benchmark was wrong. The actual root cause was that "comparable throughput-latency tradeoffs" holds *on average* on the benchmark's routing distribution, but their real agent traffic had a skewed expert-activation pattern that overloaded a subset of EP ranks. The fix was to validate the parallelism choice on *their* traffic, not the benchmark's. The lesson: configuration equivalence is a statement about a distribution, and your production distribution is not the benchmark's. The freedom to choose TP versus EP is real, but it is a freedom you have to re-measure on your own workload.

### 8. The CUDA graph that replayed a freed pointer

The symptom: a deployment that ran flawlessly until an operator bumped the maximum sequence length in a config file, after which it produced corrupted output on the first request and nothing useful thereafter. The wrong first hypothesis was that the new length had triggered a numerical overflow somewhere in the attention kernel. The actual root cause was that the captured decode graph held the *physical addresses* of its input and output tensors, and the config change had caused one of those tensors to be reallocated at a new address during warm-up — so graph replay was reading and writing a buffer that had since been freed and reused. The output was whatever happened to be living at the stale address. The fix was to pin every tensor the graph touches to a fixed, pre-allocated static buffer, copy inputs into those buffers before each replay, and force a re-capture whenever any captured shape changes. The lesson is the Section 8 second-order note made concrete: a CUDA graph captures pointers, not values, and any allocation that escapes the static set is a latent corruption that a config change can wake up. The defense is a hard rule that graph I/O lives in named static buffers and a re-capture trigger on every shape transition.

### 9. The NVFP4 recurrence that drifted on the long needle

The symptom: a model that scored perfectly on the 128K needle-in-a-haystack test and then started missing needles only at 512K and beyond — quality degraded with context length even though the prompt clearly contained the answer. The wrong hypothesis was that the GDN layers simply could not retrieve over that distance, a known limitation of weaker linear-attention designs. The actual root cause was precision: the recurrent `ssm_state` had been stored at too aggressive a quantization, and because a recurrence *accumulates*, a small per-token rounding error compounded across half a million updates into a state that had drifted far enough to lose the needle. The full-attention layers, which recompute from an exact KV cache each step, were fine; the error lived entirely in the recurrence. The fix was to keep the `conv_state` and `ssm_state` in higher precision than the NVFP4 weights — the weights can be 4-bit because each is read fresh, but the recurrence carries its own history and cannot tolerate the same rounding. The lesson is that quantization is not uniform across a hybrid model: a format that is safe for stateless weight reads can be unsafe for a stateful recurrence, and the failure does not announce itself until the context is long enough for the drift to matter. Precision has to be assigned per *role*, not per tensor size.

## When to reach for this playbook — and when not to

The TokenSpeed Qwen3.5 result is a specific engine on specific hardware, but the techniques are a transferable playbook for latency-first serving. Here is the honest scope.

**Reach for this approach when:**

- You serve **agentic or interactive single-user workloads** where per-user tokens-per-second is the SLO, not aggregate fleet throughput.
- Your model is a **hybrid linear-attention + full-attention architecture** (GDN/Mamba layers interleaved with softmax attention) — the dual-pool scheduling, the prefix-cache copy-on-write, and the index-indirection spec-decode are all *necessary* here in a way they are not for a pure Transformer.
- You have **long, reused contexts** (multi-turn agents, shared system prompts) where a prefix cache can hit >90% and recomputation is the enemy.
- You are on **Blackwell-class hardware with a 4-bit weight format** and the willingness to invest in CUDA graphs, custom fused kernels, and on-device sentinels.
- Your bottleneck profile shows **gaps between kernels** and host-side stalls rather than under-utilized matmuls — the symptom that batch-one latency engineering, not wider batching, is the cure.

**Skip it — or at least do not start here — when:**

- Your workload is **genuinely throughput-bound**: thousands of independent short-context users you can batch wide. Then aggregate-throughput engines and wide batching win, MTP often hurts, and the multi-stream overlap can contend rather than overlap.
- You serve a **pure Transformer**: most of the hybrid-specific machinery (Mamba state pool, GDN dual-projection overlap, recurrence-aware disaggregation) is dead weight you do not need.
- You need a **battle-hardened, production-proven stack today**: TokenSpeed is an early preview, and the mature engines have years of operational scar tissue this one has not yet earned.
- Your team cannot sustain the **maintenance cost** of fused kernels and CUDA-graph capture — these are powerful and brittle, and a model variant that changes a norm's arithmetic or a head dimension can silently disable a fusion until someone notices.
- You are **memory-capacity-bound, not latency-bound**: if your problem is fitting the model at all, quantization and offloading matter more than shaving microseconds off the decode loop.

The deeper takeaway outlasts any leaderboard. The agentic workload broke the assumptions every chatbot-tuned engine was built on, and the response was not one clever trick but a disciplined campaign against stalls: make the shared context free, hide the independent compute, fuse the bandwidth-bound chains, and take the CPU off the loop — all while respecting that a hybrid model's recurrent state is mutable, unforgiving, and nothing like a KV cache. Five hundred eighty tokens a second at batch one is what it looks like when an engine takes the new workload, and the new architecture, completely seriously.

## Further reading

- [TokenSpeed: inside a speed-of-light inference engine for agentic workloads](/blog/machine-learning/open-source-library/tokenspeed-agentic-inference-engine) — the engine's four-layer architecture and its MLA kernel path.
- [Nemotron-H: hybrid Mamba-Transformer](/blog/machine-learning/large-language-model/nemotron-h-hybrid-mamba-transformer) — why linear-attention recurrent state behaves so differently from a KV cache.
- [KV cache optimization and management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management) — the paged-KV mechanics the prefix cache builds on.
- [Optimizing MoE training and inference](/blog/machine-learning/large-language-model/optimizing-moe-training-and-inference) — shared vs routed experts and why their overlap is free at batch one.
- [Past the 4-bit wall: frontier LLM quantization](/blog/machine-learning/large-language-model/past-4-bit-wall-frontier-llm-quantization) — the numerics behind NVFP4 and where 4-bit breaks.
- The original [PyTorch blog announcement](https://pytorch.org/blog/up-to-580tps-new-speed-record-of-qwen3-5-397b-a17b-on-gpu-for-agentic-workloads-with-tokenspeed/) and the [TokenSpeed repository](https://github.com/lightseekorg/tokenspeed).

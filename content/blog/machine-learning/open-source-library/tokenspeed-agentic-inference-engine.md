---
title: "TokenSpeed: Inside a Speed-of-Light Inference Engine for Agentic Workloads"
publishDate: "2026-06-13"
date: "2026-06-13"
category: "machine-learning"
subcategory: "Open Source Library"
tags:
  - tokenspeed
  - llm-inference
  - agentic-workloads
  - mla
  - multi-head-latent-attention
  - blackwell
  - speculative-decoding
  - kv-cache
  - cuda-kernels
  - tensorrt-llm
  - vllm
  - inference-engine
description: "A deep dive into LightSeek's TokenSpeed: how a four-layer engine — local-SPMD modeling, a typed-FSM C++ scheduler, a pluggable multi-silicon kernel library, and a low-overhead AsyncLLM entrypoint — chases TensorRT-LLM-level speed with vLLM-level usability for the agentic-inference regime, with a tour of its MLA kernels, the fold_sq_factor BMM1 trick, and runnable code."
author: "Hiep Tran"
featured: true
image: "/imgs/blogs/tokenspeed-agentic-inference-engine-1.png"
readTime: 51
---

Every inference engine in production today was tuned for a workload that is quietly disappearing. The benchmark that made vLLM and TensorRT-LLM famous is the chatbot: a few hundred tokens of prompt, a few hundred tokens of completion, thousands of independent users you can batch together to amortize the cost of weight loads. That regime rewards one thing above all — aggregate throughput. Pack the batch as wide as it will go, keep the GPU's matmul units saturated, and report tokens per second across the whole fleet.

Then coding agents arrived, and the shape of the work changed underneath us. A single Claude Code, Codex, or Cursor session does not look like a chatbot turn. It looks like a conversation that has already accumulated **50,000 tokens of context** before the model emits its first new token, that will run for **dozens of turns**, and where the human on the other end is staring at a cursor waiting for the next token to appear. Aggregate throughput is no longer the metric that matters; **per-user generation speed** is. If your engine can serve a thousand agents at a blistering fleet-wide token rate but each individual agent only sees 18 tokens per second, every one of those agents feels broken.

[TokenSpeed](https://github.com/lightseekorg/tokenspeed), from the LightSeek Foundation, is an inference engine built from first principles for this new regime. Its pitch is deliberately provocative — "**TensorRT-LLM-level performance and vLLM-level usability**" — and its tagline is "a speed-of-light LLM inference engine." Those are big claims, and TokenSpeed is, at the time of writing, an early preview rather than a battle-hardened production system. But the *architecture* is the interesting part, because nearly every design decision in it is a direct, traceable consequence of taking the agentic workload seriously. This is a tour of that architecture: the four layers, the local-SPMD compiler that writes your collective communication for you, the scheduler that turns KV-cache safety into a compile-time property, and the MLA kernels on NVIDIA Blackwell where the "speed-of-light" claim actually has to cash out.

![The four layers of TokenSpeed](/imgs/blogs/tokenspeed-agentic-inference-engine-1.png)

The diagram above is the mental model, and the rest of this article is a tour of it. Read top to bottom, TokenSpeed is four decoupled layers. The **entrypoint** is an `AsyncLLM` wrapper integrated with a serving framework (SMG) whose only job is to keep CPU-side overhead off the critical path. The **scheduler** is a C++ control plane paired with a Python execution plane, and it encodes the request lifecycle as a typed finite-state machine. The **modeling** layer uses a local-SPMD design where a static compiler generates the parallel communication for you. And the **kernels** sit at the bottom as a first-class, swappable subsystem — `tokenspeed-kernel` — with the multi-head latent attention (MLA) kernels as its showpiece. The layers are decoupled on purpose: you can drop a faster attention kernel in without touching the scheduler, or retarget a new accelerator without rewriting the model.

A word on provenance, because it shapes how much to trust the numbers. TokenSpeed is published by the **LightSeek Foundation** and developed in collaboration with a striking roster — **NVIDIA DevTech** and **NVIDIA Dynamo**, **AMD's Triton** team, **Qwen Inference**, **Together AI**, **Mooncake**, **LongCat**, **FluentLLM**, and **EvalScope** among them. That breadth tells you two things. The kernel work has real silicon-vendor involvement on both the NVIDIA and AMD sides, which is why the multi-silicon story reads as credible rather than aspirational. And the model coverage — Kimi K2.5, Qwen 3.6, DeepSeek V4, GPT-OSS on AMD, Minimax M2.7, Nemotron — reflects the inference teams who actually run these models in anger. The codebase is roughly **90% Python and 9% C++**, which matches the architecture exactly: a thin, hot C++ control plane under a broad Python execution and modeling surface. None of this makes the preview production-ready, but it does mean the design reflects production experience rather than a research prototype.

## Why agentic inference breaks chatbot-tuned engines

The reason this problem ambushes teams is that the agentic workload violates three assumptions baked deep into the engines they already run. None of those assumptions is wrong for chatbots. All of them are wrong for agents.

| Assumption | The naive view | The reality for agents |
|---|---|---|
| "Throughput is the metric." | Maximize fleet-wide tokens/second by batching aggressively. | Each user needs a **per-user TPS floor** (≈70 TPS, often 200+); fleet throughput is worthless if any single stream drops below it. |
| "Prompts are short, so prefill is cheap." | Prefill is a small one-time cost; decode dominates. | Agentic contexts routinely exceed **50K tokens** and grow every turn — prefill and KV-cache pressure become first-order costs. |
| "Decode is memory-bound, so just widen the batch." | Bigger batch → better Tensor-Core utilization → more tokens/sec. | At small effective batch with few heads (MLA), the attention matmul **under-fills its tiles** and the Tensor Cores sit idle no matter how wide the batch is. |
| "The model author writes the parallelism." | Hand-write tensor-parallel all-reduces into the model code. | Parallelism strategy changes per deployment (TP4 vs TP8, attention-TP vs MoE-TP); hand-written collectives ossify the model. |

The thread connecting all four rows is **latency under a per-user service-level objective**. A chatbot engine maximizes a single global number. An agentic engine maximizes a global number — per-GPU tokens per minute (TPM) — *subject to a hard constraint*: every individual user's token stream must stay above an interactive floor. That constraint is not a footnote. It is the load-bearing wall of the whole design, and TokenSpeed's documentation states it plainly: the goal is to **maximize per-GPU TPM while maintaining a per-user TPS floor**, typically 70 TPS, with 200 TPS or higher as the target for a snappy agent.

> An engine that optimizes aggregate throughput will happily make every individual user miserable to win a benchmark. Agentic serving is a constrained optimization, and the constraint is the user.

Hold that distinction in your head, because it explains why TokenSpeed makes choices that look strange from a throughput-maximizing point of view — why it bounds batch size on purpose, why it pours engineering into the *decode* kernel specifically, and why the MLA attention path is where it spends its hardest optimization budget.

## 1. The agentic-inference regime

**Senior rule of thumb: if you cannot name the SLO your engine is optimizing for, you are optimizing for the wrong one.**

Let us make the objective concrete, because the entire engine is shaped around it. Define two quantities. **TPM** (tokens per minute, per GPU) is the throughput metric your finance team cares about — it sets your cost per token. **TPS** (tokens per second, per user) is the latency metric your user feels — it sets whether the agent feels alive or sluggish. A pure-throughput engine maximizes TPM and lets TPS fall where it may. TokenSpeed maximizes TPM *subject to* TPS staying above a floor.

![Throughput-only batching versus the TPS-floor objective](/imgs/blogs/tokenspeed-agentic-inference-engine-2.png)

The figure above is the whole argument in two columns. On the left, the throughput-only strategy: pack the batch as wide as possible to amortize the cost of streaming weights through the GPU on every decode step. The aggregate TPM is gorgeous. But each user's slice of the batch is now tiny, the per-user token rate collapses — call it 18 tokens per second in the illustration — and it drops below the 70-TPS floor. The agent stalls; the human watches a frozen cursor. On the right, TokenSpeed's strategy: bound the batch so that the per-user rate stays above the floor (90+ TPS in the illustration), accept a slightly lower theoretical peak TPM, and in exchange keep *every* user in the interactive zone. The numbers are illustrative, but the shape is the real engineering: you are trading a sliver of peak throughput for a guarantee that nobody falls off the floor.

Why does this constraint bite so hard in the agentic regime specifically? Two reasons, both structural:

- **Decode is the dominant cost, and decode is bandwidth-bound.** For a long-running agent, the model spends most of its wall-clock generating tokens one at a time. Each decode step must read the entire KV cache and (for a dense layer) stream the weights. With 50K-token contexts, the KV cache is enormous, and the per-step cost is gated by HBM bandwidth, not FLOPs. Widening the batch helps amortize *weights* but does nothing for the *per-request* KV read — and it directly steals per-user TPS.
- **The batch is naturally small and skewed.** Agents are expensive and long-lived, so you serve fewer of them concurrently than chatbot users, and their context lengths vary wildly (one agent is on turn 2, another on turn 40). That ragged, small-batch decode workload is exactly the regime where naive attention kernels waste the most silicon, which is the gap TokenSpeed's MLA kernels are built to close.

### A worked example: where a decode step's time goes

It helps to put numbers on why decode is bandwidth-bound, because the entire MLA kernel effort follows from it. MLA stores, per token per layer, a compressed latent plus a small decoupled-RoPE vector — for DeepSeek-class configurations that is roughly `kv_lora_rank = 512` plus `qk_rope_head_dim = 64`, about **576 elements per token per layer**. In BF16 (2 bytes) that is ~1.15 KB per token per layer; across ~60 layers, ~69 KB per token. A single 50K-token agent context is therefore about **3.4 GB of KV cache — for one request**.

Now the binding constraint: every decode step must *read that entire KV cache from HBM* to attend over the context. On a B200 with roughly 8 TB/s of HBM bandwidth, streaming 3.4 GB is about **425 µs** of pure memory traffic — before you touch a single weight matrix. That sets a hard per-request ceiling of roughly 2,300 tokens/second from KV reads alone, and it only gets worse as the context grows. This is why the optimization levers are what they are:

| Lever | Effect on the decode step | Why it matters for the floor |
|---|---|---|
| FP8 KV cache | Halves the 3.4 GB read to ~1.7 GB | Doubles the KV-bound token ceiling, ~halves the floor cost |
| `fold_sq_factor` | Fills the BMM1 tile so the matmul isn't wasted | Stops Tensor Cores idling while you wait on HBM |
| Split-KV | Parallelizes the 3.4 GB read across CTAs | Hides the latency behind overlapped loads |
| Widening the batch | **No effect** on the per-request KV read | The lever throughput engines reach for — and it does nothing here |

The last row is the punchline of the whole article. The instinct that "more batch equals more speed" is exactly backwards for the metric agents care about: the per-request KV read is fixed by context length and precision, and the only ways to move it are to make the cache smaller (FP8), read it faster (split-KV), or stop wasting the matmul that consumes it (`fold_sq_factor`). Every one of those is something TokenSpeed does, and none of them is "batch wider."

This is also why TokenSpeed targets **NVIDIA Blackwell** (B200-class) hardware first and reports its headline numbers there. Blackwell's Tensor Cores and memory subsystem are fast enough that the *software* overheads — CPU-side scheduling, collective communication, kernel-launch latency, tile under-utilization — become the binding constraint. An engine that wants to be "speed-of-light" on Blackwell has to attack those overheads directly, layer by layer. That is the tour we are about to take.

The project's stated results frame the target: **580 TPS on Qwen3.5-397B-A17B** for agentic workloads, and on Kimi K2.5 against TensorRT-LLM, roughly **9% faster in the min-latency case** and roughly **11% higher throughput** at the ~100-TPS-per-user operating point. These are vendor-reported, preview-stage numbers — treat them as a statement of intent, not an independent benchmark — but they tell you where the engine is pointed: the *latency-constrained* corner of the design space, not the throughput-maximal one.

### The reported numbers, with caveats

It is worth tabulating the public results in one place, with the caveat repeated: these are vendor-reported, preview-stage figures on specific hardware, not an independent benchmark, and you should reproduce them on your own workload before betting on them.

| Result | Configuration | What it claims |
|---|---|---|
| 580 TPS | Qwen3.5-397B-A17B, agentic workload | Headline throughput (May 2026) |
| ~9% lower latency vs TRT-LLM | Kimi K2.5, B200, min-latency point | Attention TP4 + MoE TP4 |
| ~11% higher throughput vs TRT-LLM | Kimi K2.5, B200, ~100 TPS/user | Same split-mesh config |
| ~2× (nearly halved) decode latency | Typical decode + speculative decoding | MLA decode kernel vs TRT-LLM |

Read these as a *direction*, not a guarantee. The pattern across all four rows is consistent — the wins concentrate in the **latency-constrained, agentic-decode** corner, which is exactly where the architecture is aimed. They are *single-digit-percent* against TensorRT-LLM at the steady-state operating point, but *~2×* on the specific decode-with-speculation path the kernels were built for. That shape is exactly what you would expect from an engine that out-engineered its competitor on one specific kernel interaction rather than uniformly across the board — and it is a useful reminder that "2× faster" and "9% faster" can both be true of the same system depending on which workload you measure.

If you want the broader landscape of inference optimizations this builds on — paged KV, continuous batching, quantization — the blog has a [complete guide to optimizing LLM inference](/blog/machine-learning/large-language-model/optimizing-llm-inference-complete-guide) and a survey of [efficient LLM inference techniques](/blog/machine-learning/large-language-model/efficient-llm-inference-techniques) that set the baseline TokenSpeed is trying to beat.

## 2. The modeling layer: local-SPMD and the collective-generating compiler

**Senior rule of thumb: the moment your parallelism strategy lives inside your model code, every new deployment shape becomes a model rewrite.**

Here is a problem that does not show up in a single-GPU benchmark but dominates real serving: **how does the model get sharded across GPUs, and who writes the communication?** A 400-billion-parameter MoE model does not fit on one B200. You split it — tensor parallelism inside each layer, expert parallelism across the MoE, maybe a different split for the attention blocks than for the experts. Traditionally, the model author hand-writes the `all_reduce` after each tensor-parallel matmul and the `all_gather`/`all_to_all` around the MoE dispatch. That code is tedious, easy to get wrong, and — worst of all — it bakes one parallelism strategy into the model. Want to switch from TP8 to "attention TP4 + MoE TP4"? Edit the model.

TokenSpeed's modeling layer attacks this with a **local-SPMD** (Single Program, Multiple Data) design. You write the model as if it runs on one device, and you annotate the *placement* of tensors at **module boundaries** — this activation is sharded along the head axis here, replicated there. A lightweight **static compiler** then runs during model construction, reads those boundary annotations, and **generates the required collective operations automatically**.

![Local-SPMD: collectives generated, not hand-written](/imgs/blogs/tokenspeed-agentic-inference-engine-3.png)

The figure walks the pipeline: an annotated model carrying I/O placement at its module boundaries goes into a build-time static compiler, which analyzes where data must move and *inserts* the collectives — the all-gathers and all-reduces — and emits a per-GPU SPMD executable. The author never types `dist.all_reduce`. The parallelism strategy becomes a *configuration* of the compiler, not a property of the model source.

Why this matters for the agentic SLO specifically: the optimal parallelism layout is **workload-dependent and changes the latency profile**. TokenSpeed's own best configuration for Kimi K2.5 is **Attention TP4 + MoE TP4** — a *split* strategy where the attention blocks and the MoE experts use independent tensor-parallel groups, because they have different communication-to-compute ratios. Discovering that layout by hand-editing model code for each candidate is glacial. With a placement-annotation compiler, sweeping `attention_tp ∈ {2,4,8}` against `moe_tp ∈ {2,4,8}` is a config sweep, and the compiler regenerates correct collectives for each point.

Conceptually, the model-author experience looks like this — placement annotations at the boundaries, and nothing else:

```python
## Illustrative shape of the local-SPMD authoring model.
## You declare WHERE tensors live; the compiler decides HOW they move.
import tokenspeed as ts

class MoEBlock(ts.Module):
    def __init__(self, cfg):
        super().__init__()
        # Attention projections live in the "attention" mesh (TP=4).
        self.qkv = ts.Linear(cfg.d_model, cfg.d_qkv,
                              shard=ts.Shard.head, mesh="attn")
        self.o   = ts.Linear(cfg.d_attn, cfg.d_model,
                              shard=ts.Shard.row,  mesh="attn")
        # Experts live in a SEPARATE mesh (EP/TP=4) — different comm profile.
        self.experts = ts.MoE(cfg.n_experts, cfg.d_ff,
                              shard=ts.Shard.expert, mesh="moe")

    def forward(self, x):
        # No all_reduce / all_gather written here. The static compiler sees
        # `Shard.row` on `self.o` and inserts the all-reduce at this boundary;
        # it sees the mesh change attn -> moe and inserts the resharding.
        h = self.o(self.attn(self.qkv(x)))
        return self.experts(h)
```

The payoff is that the same model source compiles to TP4, TP8, or a split attention/MoE layout with no edits — only a change to the mesh configuration handed to the compiler. The cost is a layer of indirection you have to trust: when a collective is wrong, you are debugging the compiler's placement inference, not a line you wrote. In practice that is a good trade, because the compiler's collectives are *correct by construction* for a given placement, whereas hand-written ones drift out of sync the moment someone refactors a layer. If you have ever chased a silent numerical mismatch caused by a missing `all_reduce` after a row-parallel projection, you already know why moving this into a compiler is worth a layer of indirection.

To see what the compiler is actually saving you, compare the two worlds directly:

| Concern | Hand-written collectives | Compiler-generated (local-SPMD) |
|---|---|---|
| Where parallelism lives | Inside the model's `forward()` code | In a mesh config handed to the compiler |
| Switching TP4 → TP8 | Edit and re-test the model | Change one config value, recompile |
| Split attention-TP / MoE-TP | A second hand-written code path | A second mesh over the same model source |
| Correctness after a refactor | Silent races if a collective is dropped | Correct by construction for the declared placement |
| Adding expert parallelism | Hand-write the `all_to_all` dispatch/combine | Annotate the expert shard; the compiler emits it |

The collectives the compiler emits are the usual suspects, and each maps to a placement transition: an **all-reduce** after a row-parallel matmul (summing partial outputs), an **all-gather** to reassemble a column-sharded activation, and an **all-to-all** around MoE dispatch and combine (routing each token to its expert's shard and back). Hand-writing these is not hard for one layout; the pain is that you must redo them, correctly, for *every* parallelism shape you ship — and the MoE `all_to_all` in particular is a notorious source of subtle, shape-dependent bugs. Pushing them into a compiler turns a combinatorial test matrix into a config sweep.

### Second-order effect: placement annotations are a contract, not a hint

The subtle gotcha is that module-boundary annotations are a **contract the compiler enforces**, not a suggestion it optimizes around. If you annotate an activation as head-sharded but a downstream module expects it replicated, the compiler must insert a resharding collective — and if you got the annotation wrong, you get a *correct* program that does *unnecessary* communication. The failure mode here is not a crash; it is a quietly slow model. The discipline this demands is that you treat placement annotations with the same care you treat type signatures: they are the interface between modules, and a sloppy one costs you bandwidth on every single token. This is the same philosophy the scheduler applies to KV-cache ownership, which is the next layer.

## 3. The scheduler: a C++ control plane as a typed finite-state machine

**Senior rule of thumb: in a long-running serving loop, the bugs that hurt most are not crashes — they are silent data races on shared GPU memory.**

The scheduler is where TokenSpeed makes its most opinionated bet, and it is the layer most directly shaped by the "dozens of turns, 50K-token context" workload. The hard problem a scheduler solves is **resource ownership over time**: a request arrives, gets admitted (or rejected), waits in a queue, runs prefill (which *allocates* KV-cache pages), runs decode (which *holds* those pages and produces one token per step), and eventually finishes (*freeing* the pages) or gets evicted under memory pressure (*spilling* them). At every moment, dozens of requests are at different points in this lifecycle, and the KV cache is a single shared pool of GPU memory they are all contending for.

Get the ownership wrong and you get the worst class of bug in systems programming: two requests believing they own the same KV page, one overwriting the other's attention state, producing *plausible but wrong* tokens with no crash to tell you something broke. In a chatbot that surfaces as an occasional weird completion. In a 40-turn agent, a single corrupted KV page poisons every subsequent turn.

TokenSpeed's answer is to make the request lifecycle a **finite-state machine that works with the type system to enforce safe resource management at compile time**. The control plane is written in **C++**; the execution plane is **Python**. Request lifecycle, KV-cache resources, and overlap timing are represented as **explicit FSM transitions with ownership semantics** — and crucially, the type system is enlisted so that a piece of code *cannot* touch a KV resource it does not own. Unsafe reuse becomes a compile error, not a runtime corruption.

![Scheduler as a typed FSM over KV ownership](/imgs/blogs/tokenspeed-agentic-inference-engine-4.png)

The figure traces the lifecycle as the FSM sees it. `admit` checks the KV budget; if there is no room, the request transitions straight to `rejected` (admission control is how you protect the floor — better to reject than to admit a request that drags everyone below the SLO). An admitted request `waits`, then is `scheduled` into `prefill`, which **allocates KV pages**. Prefill hands ownership to `decode`, which **holds** the pages and emits one token per step. From decode there are two exits: `finished` on EOS, which **frees** the pages, or `evicted` under memory pressure, which **spills** the KV to host memory and `requeues` the request to reclaim that capacity. Each arrow is an ownership handoff, and the type system makes each handoff explicit.

The reason to split the control plane into C++ while keeping execution in Python deserves its own paragraph, because it is a recurring pattern in high-performance serving. The control plane runs on **every token, for every request** — it is the hottest CPU code in the system. Python's interpreter overhead and the GIL make it a poor place for that loop; a microsecond of scheduling overhead per token, multiplied by a wide batch and a high token rate, becomes a real chunk of your latency budget and directly eats into the per-user TPS floor. By pushing the lifecycle FSM and ownership bookkeeping into C++, TokenSpeed keeps the per-token control overhead low while keeping the *execution* plane — model invocation, kernel dispatch — in Python where it is ergonomic. This is the same instinct behind the AsyncLLM entrypoint, whose entire purpose is "low CPU-side overhead."

The "overlap timing" piece of the FSM is the third thing it encodes, and it is subtle. Modern inference overlaps work to hide latency: while the GPU computes decode for batch *N*, the CPU is preparing batch *N+1*, copying the next set of tokens, updating block tables. If that overlap is mistimed — if batch *N+1*'s setup reads a KV page that batch *N* is still writing — you have the same race as before, just in the time dimension instead of the space dimension. By encoding overlap timing as FSM transitions with ownership semantics, the scheduler makes the *temporal* hazards as type-checked as the *spatial* ones.

### Prefix caching: the agentic memory that makes long context affordable

There is a structural gift hiding in the agentic workload, and the scheduler's KV-ownership model is what lets TokenSpeed collect it. Agents are *repetitive*: turn N+1's prompt is turn N's prompt plus a little more. The 50K-token context is overwhelmingly a **shared prefix** — the system prompt, the tool definitions, the accumulated conversation — and only the last few hundred tokens are new. Re-running prefill over the entire 50K tokens every turn would be ruinous; it would make each turn's time-to-first-token grow without bound as the conversation lengthens.

The paged KV layout makes the fix natural. Because the KV cache is stored in fixed-size pages addressed through a per-request block table, two requests — or two turns of the same request — that **share a prefix can share the underlying KV pages**. Turn N+1 reuses turn N's pages for the shared portion and only prefills the new suffix. The scheduler's job, and the reason KV ownership has to be a typed, explicit thing, is to track which pages are shared, refcount them so a page is never freed while another turn still references it, and copy-on-write only when a branch actually diverges. This is precisely the class of bug the FSM's ownership semantics exist to prevent: a shared prefix page freed one turn too early silently corrupts every turn that still points at it.

The payoff is enormous and specific to agents. In the chatbot regime, prefixes are short and prefix caching is a minor win. In the agentic regime, the shared prefix is the *majority* of the context, so prefix reuse turns an O(context) per-turn prefill into an O(new tokens) one — the difference between a coding agent that answers in a second and one that pauses for ten while it re-reads its own history. It also compounds across the fleet: many agents share the same system prompt and tool definitions, so those pages can be cached once and shared across thousands of requests instead of recomputed per session.

### Second-order effect: admission control is a latency feature, not a capacity feature

The non-obvious consequence is that `rejected` is a *good* state. A throughput-maximizing engine treats rejection as failure — it wants to admit everything. An SLO-constrained engine treats rejection as **the primary tool for protecting the floor**. If admitting one more request would drop the existing requests below 70 TPS, the right move is to reject (or queue) it. The FSM makes this a first-class transition rather than an afterthought, which means capacity planning for TokenSpeed is really *floor* planning: you size the fleet so that the admission controller rarely has to reject, and when it does, it sheds load gracefully instead of degrading everyone. If you want the background on why KV-cache management is the crux of all of this, the deep dives on the [KV cache](/blog/machine-learning/large-language-model/kv-cache) and [KV-cache optimization and management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management) cover the data-structure side that this scheduler is governing.

## 4. The kernel subsystem: `tokenspeed-kernel` and pluggable multi-silicon backends

**Senior rule of thumb: the fastest kernel for a given (silicon, shape, dtype) changes every six months — so make the kernel a swappable part, not a hardcoded call.**

If the scheduler is TokenSpeed's most opinionated layer, the kernel subsystem is its most *pragmatic* one. The core engine does not call CUDA directly. It calls into `tokenspeed-kernel`, a separate package whose stated goal is "a collection of the best portable and performant kernels for multi-silicon AI inference." The engine treats this package as its **only kernel boundary** — a single, vendor-neutral interface — and `tokenspeed-kernel` is responsible for resolving each call to the fastest available implementation for the hardware it is running on.

The public API is deliberately small and solution-agnostic. You call operations by *what they do*, not *how they do it*:

```python
## tokenspeed-kernel's public surface: name the operation, not the backend.
from tokenspeed_kernel import (
    mha_prefill, mha_prefill_with_kvcache, mha_decode_with_kvcache,  # attention
    mm,                                                              # GEMM
    moe_route, moe_dispatch, moe_experts, moe_combine, moe_fused,    # MoE
)

## The SAME call resolves to a different backend on Blackwell vs. on AMD,
## and to a different kernel for prefill (compute-bound) vs. decode (bw-bound).
out = mha_decode_with_kvcache(query, kv_cache, block_tables, seq_lens,
                              softmax_scale=scale)
```

Behind that flat API sits a **layered backend strategy**, and understanding the layers is understanding the engine's whole philosophy of portability-with-an-escape-hatch.

![Kernel selection: one API, many backends](/imgs/blogs/tokenspeed-agentic-inference-engine-5.png)

The selection tree above shows the four kinds of backend a single API call can resolve to:

- **Triton — the default portable JIT path.** Triton kernels run across vendors, so they are the baseline that guarantees the engine works *somewhere* on any silicon. When no faster specialized kernel exists, Triton is the floor.
- **CuteDSL / Triton Gluon — the performant JIT path for key kernels.** For the operations that dominate the latency budget, TokenSpeed reaches for a faster JIT: **CuteDSL** for NVIDIA GPU kernels (the MLA kernels are CuteDSL), **Triton Gluon** for AMD GPU kernels. These are not portable, but they are fast, and they are reserved for the kernels where the speedup pays for the specialization.
- **Vendor wraps.** Where a vendor library is already best-in-class — FlashAttention, TensorRT-LLM's kernels — `tokenspeed-kernel` simply wraps it rather than reinventing it. The directory convention is `ops/<family>/<solution>`, e.g. `gemm/trtllm.py`, `attention/triton/`.
- **PyTorch reference.** Every operation has a plain-PyTorch reference implementation used as **ground truth** for correctness testing. When a fast kernel produces wrong numbers, the reference is what tells you.

The mechanism that makes this extensible is a single decorator. Backends — including **out-of-tree** ones from third-party packages — register themselves via `@register_kernel`, and they participate in selection on equal footing with the built-in kernels. This is the "pluggable mechanism for heterogeneous accelerators" the architecture promises, and it is what lets a hardware vendor add support for new silicon **without forking the engine**.

```python
## Registering a custom backend from your OWN package — no fork required.
from tokenspeed_kernel import register_kernel

@register_kernel(
    op="mha_decode_with_kvcache",
    # This backend is only eligible when these predicates hold.
    silicon="amd:mi355x",
    dtype=("fp8_e4m3", "bf16"),
    constraint=lambda shape: shape.num_heads <= 128,
)
def my_gluon_decode(query, kv_cache, block_tables, seq_lens, *, softmax_scale):
    # ... a Triton Gluon kernel tuned for this accelerator ...
    return out

## At dispatch time, tokenspeed-kernel scores all registered backends whose
## predicates match (silicon, dtype, shape) and picks the best — yours included.
```

The design conventions around this are tellingly disciplined: a module named `_triton.py` centralizes *all* direct Triton imports so the rest of the codebase never imports Triton ad hoc; third-party code lives in a `thirdparty/` directory and is re-exported into `ops/`; and the project's own contribution guide says any dependency that keeps causing version conflicts should be "removed entirely or at least made optional." That is the posture of a library that intends to be embedded in many different stacks without dragging a brittle dependency tree behind it.

It is worth noticing what is in that API list beyond attention, because it tells you what TokenSpeed thinks the expensive operations in a modern MoE model actually are. Alongside `mm` (the workhorse GEMM) sit five MoE primitives: `moe_route` (the gating that picks each token's experts), `moe_dispatch` (scattering tokens to the GPUs that hold their experts), `moe_experts` (the per-expert feed-forward compute), `moe_combine` (gathering the expert outputs back), and `moe_fused` (a fused path that collapses several of those steps into one kernel to avoid round-tripping activations through HBM between them). For the large MoE models TokenSpeed targets — Kimi K2.5, DeepSeek V4 — the MoE layers, not attention, are where most of the parameters and a large share of the FLOPs live, and the dispatch/combine steps are communication-heavy (recall the `all_to_all` the local-SPMD compiler generates). Exposing these as first-class, swappable kernels lets the engine apply the same fallback-with-a-fast-ceiling strategy to the MoE path that it applies to attention: a portable Triton implementation for coverage, a fused vendor or CuteDSL kernel for the configurations where the speedup pays. If you want the deeper story on why MoE dispatch is the hard part of serving these models, the blog covers [optimizing MoE training and inference](/blog/machine-learning/large-language-model/optimizing-moe-training-and-inference) in detail.

### Second-order effect: portability is a fallback, not a tax

The thing to appreciate is the *ordering*. A lot of "portable" frameworks pay a permanent performance tax to run everywhere — they target the lowest common denominator and never beat a hand-tuned vendor kernel. TokenSpeed inverts this: portability (Triton) is the **fallback** that guarantees coverage, while the **fast path is always a specialized kernel** (CuteDSL, Gluon, or a vendor wrap) when one exists for your silicon and shape. You get vLLM-level "it just runs" usability from the Triton floor and TensorRT-LLM-level speed from the specialized ceiling, selected automatically. The cost is a larger kernel matrix to maintain and test — which is exactly why the PyTorch reference implementations and the `@register_kernel` selection harness exist. For a comparison point on how a fully vendor-specific compiler approaches the same problem, see the deep dive on [TensorRT as an end-to-end inference compiler](/blog/machine-learning/mlops/tensorrt-end-to-end-inference-compiler).

## 5. The MLA kernels: where "speed-of-light" has to cash out

**Senior rule of thumb: in long-context decode, attention is the kernel that decides whether your GPU is a supercomputer or a very expensive memory controller.**

Everything above is plumbing. This is the engine room. TokenSpeed's headline kernel claim is "one of the fastest **MLA** implementations on Blackwell" for agentic workloads, and MLA — Multi-head Latent Attention — is the attention variant that DeepSeek popularized and that Kimi, Qwen, and friends now use. If you need the full background on the mechanism, the blog has a dedicated piece on [Multi-head Latent Attention](/blog/machine-learning/large-language-model/multi-head-latent-attention-mla); the one-paragraph version is that MLA compresses the per-token key/value into a **low-rank latent** (of size `kv_lora_rank`) plus a small **decoupled RoPE** component (of size `qk_rope_head_dim`), so the KV cache you must store and re-read on every decode step is dramatically smaller than vanilla multi-head attention. Smaller KV cache means less HBM traffic per token — exactly what a bandwidth-bound, 50K-context decode loop needs.

MLA splits cleanly into two kernels with very different characters, and TokenSpeed optimizes them separately.

### 5.1 Prefill: ragged, varlen, and FLOP-bound

The **prefill** kernel processes the entire existing context to populate the KV cache before generation begins. With agents on different turns, the batch is **ragged**: one request has 2K tokens of context, another has 48K. TokenSpeed's prefill kernel handles this with true **variable-length (varlen) ragged inputs — no padding** — so you do not waste compute padding the 2K request out to 48K. It supports causal and non-causal masking, returns the optional **log-sum-exp (LSE)** statistics needed for downstream chunked attention, and offers a **PDL** (programmatic dependent launch) path to overlap kernel launches. The signature is concrete:

```python
from tokenspeed_mla import tokenspeed_mla_prefill

## Ragged batch: queries/keys/values are concatenated, with cumulative
## sequence-length offsets marking each request's boundaries. No padding.
out, lse = tokenspeed_mla_prefill(
    query,            # [sum(q_lens), h_q, d_qk]
    key,              # [sum(kv_lens), h_k, d_qk]
    value,            # [sum(kv_lens), h_k, d_v]
    seq_lens,         # per-request KV lengths
    cum_seq_lens,     # prefix sums -> ragged offsets for KV
    max_seq_len,
    batch_size,
    softmax_scale,
    is_causal=True,
    return_lse=True,           # FP32 LSE for chunked / split attention
    cum_seq_lens_q=None,       # set when Q and KV lengths differ
    max_seq_len_q=None,
    enable_pdl=True,           # programmatic dependent launch overlap
)
```

Prefill ships two backends behind that one call: a **CuTe DSL JIT** path (the default, which compiles the kernel for your exact static configuration and caches it) and an **AOT binary** path that leverages NVIDIA-internal optimizations for the configurations it covers. The kernel keeps a **fine-tuned softmax** implementation — softmax is deceptively expensive in attention because of the max-subtraction and normalization passes, and a tuned one matters at long context. Prefill is fundamentally **FLOP-bound** (you are doing the full quadratic attention over the context once), so the wins here come from feeding the Tensor Cores cleanly and overlapping launches.

The two-backend split is a small lesson in its own right. A **JIT** kernel compiles for your exact static configuration — head count, head dim, dtype, causal or not — and caches the result, so you pay a one-time compile cost and then run a kernel specialized to your shape; this is what makes CuTe DSL attractive for the long tail of model configurations. An **AOT binary** is pre-compiled and ships ready to run, trading the JIT's per-shape specialization for zero warmup and the ability to embed NVIDIA-internal optimizations that are not expressible in the public DSL. TokenSpeed defaults to the JIT for flexibility and reaches for the AOT binary where it is faster on the configurations it covers — the same fallback-with-a-fast-ceiling pattern the kernel selector uses one layer up. The practical consequence for you: the first request of a new shape may pay a compile, so a warmup pass over your expected shapes is worth running before a server enters the rotation.

### 5.2 Decode: the fold_sq_factor trick and the BMM1 M tile

Decode is the kernel that runs on **every single generated token**, and it is where the agentic SLO lives or dies. Here is the structural problem that TokenSpeed's decode kernel is built to solve, and it is subtle enough that it is worth slowing down for.

Attention's first matmul — call it **BMM1**, the batched `Q · Kᵀ` that produces the attention scores — is mapped onto the GPU's Tensor Cores as a tiled GEMM. The Tensor Core wants a tile with a healthy **M dimension** (the number of rows in the output) to be efficient; on Blackwell, the matmul instruction effectively wants the M tile filled toward **128 rows**. In ordinary multi-head attention with many heads and a wide batch, M is naturally large and the tile fills.

But MLA decode is a small-M nightmare. During token-by-token decode, the query sequence length is **1** (you are generating one token), and MLA's structure means the number of heads feeding BMM1 can be **well below 128**. So the BMM1 tile arrives with, say, 16 rows out of a possible 128 — **12% filled** — and the Tensor Core does the same fixed-cost work it would for a full tile while producing a fraction of the useful output. You are paying full freight for 12% of a matmul. No amount of widening the batch fixes this, because batch widens a *different* axis; the M-tile under-fill is intrinsic to small-head, single-query decode.

TokenSpeed's fix is the **`fold_sq_factor`** optimization, and it is the cleverest single idea in the kernel. When `num_heads < 128`, the decode kernel **folds the query-sequence axis into the head axis** to fill the BMM1 M tile. Concretely, it picks a fold factor `F` such that `q_seqlen % F == 0` and `num_heads * F <= 128`, then groups `q_seqlen` and `num_heads` together into the BMM1 M dimension. The under-filled tile becomes a full one, and the Tensor Cores go from idling to saturated.

![fold_sq_factor: filling the BMM1 M tile](/imgs/blogs/tokenspeed-agentic-inference-engine-6.png)

The figure makes the arithmetic vivid. On the left, the *before*: `num_heads = 16`, `q_seqlen = 1`, so BMM1 runs with `M = 16 / 128` — a 12%-filled tile and Tensor Cores 88% idle. On the right, the *after*: fold with `F = 8`, so `num_heads × F = 128`, the tile is **fully filled**, and the Tensor Cores are saturated. Same FLOPs of useful work, but now packed into a tile the hardware actually wants.

Where does the `q_seqlen` to fold *come from* during decode, when you are generating one token at a time? Two places, and both are central to the agentic workload. First, **speculative decoding**: a draft model proposes several tokens at once, and the target model verifies all of them in one decode call — so `q_seqlen` is the number of speculative tokens, not 1. Second, **multi-token prediction** and similar schemes that score several candidate next-tokens together. Folding those query tokens into the head axis is what turns the small-batch, small-head MLA decode from a Tensor-Core-starved kernel into a saturated one. This is the mechanism behind the project's claim that the decode kernel "**nearly halves latency** relative to TensorRT-LLM on typical decode workloads with speculative decoding."

```python
from tokenspeed_mla import tokenspeed_mla_decode

## Choose the fold factor: pack q_seqlen into the head axis so that
## num_heads * F fills the 128-wide BMM1 M tile, given F | q_seqlen.
def choose_fold_factor(num_heads: int, q_seqlen: int, m_tile: int = 128) -> int:
    best = 1
    f = 1
    while f <= q_seqlen:
        if q_seqlen % f == 0 and num_heads * f <= m_tile:
            best = f                       # largest valid fold wins
        f += 1
    return best

## e.g. speculative decode verifying 8 draft tokens with 16 MLA heads:
F = choose_fold_factor(num_heads=16, q_seqlen=8)   # -> 8  (16 * 8 = 128)

out = tokenspeed_mla_decode(
    query,             # [B, q_len, H, D_qk]   (q_len carries the spec tokens)
    kv_cache,          # [num_pages, page_size, D_total]  paged latent KV
    workspace_buffer,  # torch.int8, 1D scratch for split-KV reduction
    kv_lora_rank,      # MLA latent dim
    qk_rope_head_dim,  # decoupled RoPE dim
    block_tables,      # [B, max_pages]  paged KV indirection
    seq_lens,          # [B]  per-request context length
    max_seq_len,
    softmax_scale,
    enable_pdl=False,
)
```

One subtlety worth flagging: the fold targets **BMM1**, the `Q·Kᵀ` score matmul, because that is where the M-tile under-fill bites in MLA decode. The *second* attention matmul, **BMM2** (`softmax(scores)·V`), has a different shape and a different bottleneck — it is dominated by the streaming read of the value cache, which is exactly what split-KV addresses. The two optimizations are therefore complementary, not redundant: `fold_sq_factor` rescues the compute-side waste in BMM1, while split-KV rescues the bandwidth-side cost of BMM2 and the KV read. A decode kernel that fixed only one of them would still be bottlenecked on the other, which is why TokenSpeed ships both in the same kernel.

### 5.3 Split-KV: hiding HBM latency behind the reduction

The second half of the decode kernel's speed comes from how it reads the KV cache. A 50K-token context is far too long for one CTA (cooperative thread array — a thread block) to chew through while keeping the latency low; if a single CTA streamed the whole sequence, the kernel would be a long, serial, bandwidth-bound crawl. TokenSpeed uses **split-KV**: it partitions the KV sequence across multiple CTAs, each of which computes a **partial softmax** over its slice and emits the partial result plus the **LSE** needed to combine slices correctly. A **second kernel** then **reduces** the partials — using the log-sum-exp trick to merge the partial softmaxes exactly — into the final output.

![MLA decode: split-KV partials reduced via log-sum-exp](/imgs/blogs/tokenspeed-agentic-inference-engine-7.png)

The figure shows the two-kernel structure. The query and the **paged KV cache** (with `page_size = 64`) fan out across CTAs; each CTA owns a contiguous slice of the sequence and computes a partial softmax with its LSE. Those partials fan in to a second kernel that performs the LSE reduction and writes the final BF16 output. The reason this is fast is **latency hiding**: with the sequence split across CTAs, the memory loads happen in parallel and overlap, and the GPU's L2 cache prefetches effectively because each CTA streams a predictable contiguous slice. The `workspace_buffer` you pass in is the scratch space the split-KV reduction uses; its size is auto-computed and cached per configuration.

The kernel is dense with Blackwell-specific micro-optimizations, and they are worth listing because each one is a small, concrete latency win:

| Technique | What it does | Why it helps decode |
|---|---|---|
| `fold_sq_factor` | Folds `q_seqlen` into the head axis when `num_heads < 128` | Fills the BMM1 `M` tile, saturating Tensor Cores |
| Split-KV + LSE reduce | Partitions the sequence across CTAs, two-kernel reduction | Parallelizes long-context KV reads, hides HBM latency |
| **2CTA UTCMMA** | Two-CTA cooperative tensor-core matmul instruction | Reduces shared-memory pressure per CTA |
| Minimal mbarrier count | Fewer memory barriers in the pipeline | Less synchronization overhead in the inner loop |
| Split-KV loading warps | Dedicated warps for KV loads, leaning on L2 prefetch | Overlaps loads with compute |
| Multi-stage STG epilogue | Multi-stage store-global in the output phase | Keeps the write-out from stalling the pipeline |

### 5.4 FP8 numerics: store cheap, compute stable

The last piece of the decode story is precision, and it interacts directly with the long-context memory pressure. The MLA kernels accept FP16, BF16, and **FP8** inputs (both `E4M3FN` and `E5M2`). Storing the KV cache in FP8 roughly halves its memory footprint versus BF16 — which, at 50K tokens across a wide batch, is the difference between fitting and OOM-ing. But FP8 has very little mantissa, and accumulating a long softmax in it would bleed accuracy. TokenSpeed's rule is pragmatic: **FP8 decode writes a BF16 output** for downstream stability. You get the storage and bandwidth win of FP8 on the KV cache, and you avoid propagating FP8's noise into the residual stream. There is even a fused **K/V pack + FP8 quantize** Triton kernel that combines the concatenation and the cast into one pass, so quantizing the new token's KV into the cache is not a separate memory round-trip.

The dtype contract is worth pinning down precisely:

| Stage | Accepted input | Output | Note |
|---|---|---|---|
| Prefill | FP16 / BF16 / FP8 (E4M3FN, E5M2) | BF16 (+ optional FP32 LSE) | LSE returned for chunked attention |
| Decode | FP16 / BF16 / FP8 | BF16 | FP8 decode → BF16 out for stability |
| K/V pack | new K, V | FP8 in cache | fused concat + cast (Triton) |

This is the same store-cheap-compute-stable philosophy that runs through modern quantized inference; for the broader picture on where FP8 and INT4 each win, see [how quantization works](/blog/machine-learning/large-language-model/how-quantization-works-gguf-quant-types-decoded) and the discussion of [the FP8/INT4 tradeoffs at the edge](/blog/machine-learning/mlops/quantization-int8-fp16-int4-edge-tradeoffs).

### Second-order effect: the kernel is co-designed with the scheduler

The non-obvious thing about all this kernel work is that it only pays off because the scheduler feeds it the right shapes. `fold_sq_factor` needs `q_seqlen > 1`, which means the scheduler must be running speculative decoding or multi-token prediction and batching the speculative tokens into the decode call. Split-KV needs the `workspace_buffer` sized and the paged `block_tables` laid out so the CTAs get contiguous slices. The FP8 path needs the KV cache allocated in FP8 from the start. None of these kernels is a drop-in that magically speeds up an arbitrary engine — they are co-designed with the scheduler and the KV-cache layout above them. That co-design is the whole argument for the four-layer architecture: the layers are decoupled enough to swap, but designed *together* so the fast paths line up.

## Cross-cutting concerns: speculative decoding, FP8, and parallelism config

Three concerns cut across every layer, and they are where a deployer actually spends their tuning time.

### Speculative decoding as the engine's native mode

We have met speculative decoding twice already — as the source of the `q_seqlen` that `fold_sq_factor` folds, and as the workload where the decode kernel "nearly halves latency." It is worth seeing the full step, because the interaction between the algorithm and the MLA kernel is the crux of TokenSpeed's decode speed.

![Speculative decode: K tokens verified in one MLA call](/imgs/blogs/tokenspeed-agentic-inference-engine-8.png)

The timeline walks one speculative decode step. A small **draft model proposes K = 4 tokens** cheaply. The target model — the big one — **verifies all four in a single MLA decode call**, and this is the key move: those four query tokens become the `q_seqlen` that gets folded into the BMM1 M tile, so the verification runs on a *full* Tensor-Core tile instead of four wasteful single-token calls. The kernel **accepts the longest matching prefix**, **resamples one token from the target** at the first mismatch (preserving the exact target distribution), and **commits up to K+1 tokens in one step**. The net effect: one expensive target forward pass yields multiple tokens, and because folding made that forward pass tile-efficient, the per-token cost drops hard. Speculative decoding on a naive kernel is often disappointing precisely because the verification call is small-M and Tensor-Core-starved; `fold_sq_factor` is what makes the speedup real on MLA.

### Parallelism configuration is your main tuning knob

Because the local-SPMD compiler decouples parallelism from model code, your main deployment knob is the **mesh configuration**, and it is genuinely consequential. TokenSpeed's reported best configuration for Kimi K2.5 is **Attention TP4 + MoE TP4** — independent tensor-parallel groups for attention and for the experts. The reason a *split* config wins is that attention and MoE have different communication-to-compute ratios: attention's all-reduce is relatively cheap per FLOP, while MoE's dispatch/combine is communication-heavy, so the optimal shard count differs between them. A launch looks roughly like:

```bash
## Launch a TokenSpeed server with a split attention/MoE parallelism layout.
## The static compiler regenerates the correct collectives for this mesh;
## no model edits required to change these numbers.
python -m tokenspeed.serve \
  --model kimi-k2.5 \
  --attention-tp 4 \
  --moe-tp 4 \
  --kv-cache-dtype fp8_e4m3 \
  --page-size 64 \
  --speculative-draft kimi-k2.5-draft \
  --speculative-num-tokens 4 \
  --tps-floor 70 \
  --max-model-len 65536
```

Two flags there encode the whole thesis of this post: `--tps-floor 70` tells the admission controller the SLO it must protect, and `--speculative-num-tokens 4` sets the `q_seqlen` that the decode kernel will fold. The rest — `--kv-cache-dtype fp8_e4m3`, `--page-size 64`, `--max-model-len 65536` — are the long-context, memory-pressure knobs the agentic regime forces you to think about.

### The AsyncLLM entrypoint

At the very top, the entrypoint is an **`AsyncLLM`** integrated with a serving framework (SMG) and engineered for **low CPU-side overhead**. For a Python user, the multi-turn agent loop looks ordinary — you `await` a stream of tokens — but the overhead-minimizing work happens underneath, in the C++ control plane and the overlap timing the scheduler manages:

```python
import asyncio
from tokenspeed import AsyncLLM, SamplingParams

async def run_agent(llm: AsyncLLM, tools):
    # A long-running agent: dozens of turns, growing 50K+ context.
    history = []
    params = SamplingParams(temperature=0.7, max_tokens=2048)

    for turn in range(40):                     # the agentic regime: many turns
        prompt = render(history, tools)        # context grows every turn
        stream = llm.generate(prompt, params, request_id=f"agent-{turn}")

        text = ""
        async for out in stream:               # tokens arrive above the TPS floor
            text += out.delta
            emit_to_ui(out.delta)              # the human watches this stream

        action = parse_tool_call(text)
        observation = await tools.run(action)  # tool use between turns
        history += [("assistant", text), ("tool", observation)]

asyncio.run(run_agent(AsyncLLM.from_pretrained("kimi-k2.5"), my_tools))
```

The `request_id` per turn is what lets the scheduler track the request's KV ownership through its FSM; the growing `prompt` is what makes prefix-caching and the long-context kernels matter; and the `async for` is what makes the per-user TPS floor a thing the user actually experiences.

## Building and embedding TokenSpeed

Because TokenSpeed intends to be *embedded* in other people's serving stacks, its build and contribution conventions are part of the design, not an afterthought — and they are worth knowing if you plan to extend it. The runtime package, `tokenspeed`, keeps its dependencies **vendor-neutral** and treats `tokenspeed-kernel` as its single kernel boundary; you can therefore pull in the kernel library, or even just the MLA kernels, on their own.

```bash
## The MLA kernels are usable standalone — you do not need the whole engine.
pip install tokenspeed-mla          # CuTe DSL JIT + AOT MLA prefill/decode
pip install tokenspeed-kernel       # the portable multi-silicon kernel library
pip install tokenspeed              # the full engine: scheduler + entrypoint

## Contributor hygiene the project enforces before a commit lands:
pre-commit run --all-files          # formatting, lint, import-gating checks
pytest tests/ -k "reference"        # every kernel is checked vs torch ground truth
```

A few conventions reveal the engineering temperament. All direct Triton imports are funneled through a single `_triton.py` module, so the rest of the codebase never reaches for Triton ad hoc and the dependency can be swapped or gated in one place. Third-party code lives under `thirdparty/` and is re-exported into `ops/` rather than imported scattershot, so the provenance of every kernel stays visible. The `ops/<family>/<solution>` directory layout — `gemm/trtllm.py`, `attention/triton/` — means adding a backend is creating a file in a predictable place, not threading a new branch through a dispatcher. And the rule that any dependency causing repeated version conflicts should be "removed entirely or made optional" is the posture of a library that expects to live inside stacks it does not control.

The testing convention is the one to copy even if you never touch TokenSpeed: **every kernel has a plain-PyTorch reference implementation that serves as ground truth.** When a hand-tuned CuTe DSL or Gluon kernel produces a wrong number — and at FP8, on a new accelerator, it will — the reference is what tells you the *kernel* is wrong rather than the model. This is the unglamorous infrastructure that makes a multi-backend kernel library trustworthy: you cannot safely keep five implementations of `mha_decode_with_kvcache` unless you have a sixth, obviously-correct one to check them all against. If your team maintains custom kernels and does not have this, build it before you build anything else.

## Case studies you can borrow

These are not incident post-mortems from a mature production system — TokenSpeed is a preview, and honest engineering means saying so. They are **design walkthroughs**: concrete situations where TokenSpeed's choices either earn their keep or do not, written so you can map them onto your own stack and decide what to copy.

### 1. The 80K-context coding agent that stalls on a chatbot-tuned server

**The symptom.** You stand up a coding agent on a vLLM deployment that benchmarks beautifully on ShareGPT-style chat. In production, every agent session feels like it is wading through mud — tokens trickle out at 15–20 per second once the context passes 40K tokens, and your users complain the agent "thinks too slowly."

**The wrong first hypothesis.** "We're GPU-bound; buy more GPUs." You add nodes, fleet-wide throughput climbs, and per-session speed does not move. The benchmark that sold you the engine measured the wrong number.

**The actual root cause.** Your engine is maximizing aggregate throughput by batching wide, and at 80K context the per-step KV read dominates while the per-user slice of the batch shrinks. The per-user TPS has fallen below the interactive floor and no amount of fleet capacity raises it, because the constraint is per-request bandwidth, not fleet FLOPs.

**The fix.** Switch to an engine that treats the **per-user TPS floor as a first-class constraint** — bound the batch to hold the floor, run speculative decoding so each decode step commits multiple tokens, and store the KV in FP8 so the 80K context fits without thrashing. The lesson: in the agentic regime, *measure and optimize per-user TPS, not fleet TPM*, and pick an engine whose admission controller knows the floor.

### 2. Kimi K2.5 on a single B200 node

**The setup.** You are serving Kimi K2.5 — a large MoE — for an agent product on one B200 node, and you want the best latency at roughly 100 TPS per user.

**The exploration.** The naive layout is uniform tensor parallelism: TP8 across everything. It works, but it is not optimal, because attention and the MoE experts have different communication profiles and forcing them into the same shard count over-communicates somewhere.

**The result.** TokenSpeed's reported optimum here is **Attention TP4 + MoE TP4** — split meshes — which lands roughly **9% faster in the min-latency case** and roughly **11% higher throughput** at the ~100-TPS operating point versus TensorRT-LLM. Because the local-SPMD compiler generates the collectives, finding this was a config sweep over `(attention_tp, moe_tp)`, not a model rewrite per candidate. The lesson: **the optimal parallelism layout is rarely uniform**, and a compiler that lets you sweep it cheaply is worth more than any single hand-tuned kernel.

### 3. The small-head decode wall

**The symptom.** You profile MLA decode and find the attention kernel is the bottleneck even though the batch is wide and the GPU "looks busy." Tensor-Core utilization, when you actually measure it, is in the teens.

**The wrong first hypothesis.** "Widen the batch more." You do, utilization does not improve, and now per-user TPS is worse because the batch slices are thinner.

**The actual root cause.** MLA decode has `q_seqlen = 1` and `num_heads < 128`, so BMM1 arrives with an M tile filled to maybe 12%. The Tensor Core pays full instruction cost for a sliver of useful output. Batch width is the wrong axis — it does not touch M.

**The fix.** `fold_sq_factor`: fold the query-sequence axis into the head axis so `num_heads * F` reaches 128, filling the tile. The catch is that you need `q_seqlen > 1` to fold, which means you must be running speculative decoding or multi-token prediction. The lesson: **tile under-utilization is invisible to throughput metrics** — you have to look at Tensor-Core occupancy, and the fix is to reshape the matmul, not resize the batch.

### 4. Speculative decoding that finally pays off

**The symptom.** You enable speculative decoding expecting a big speedup and get a disappointing 1.2×, well below the acceptance-rate math that promised 2×+.

**The wrong first hypothesis.** "Our draft model's acceptance rate is too low." You train a better draft model; the speedup barely moves.

**The actual root cause.** The verification step — running the target model on the K draft tokens — was executing as a small-M, Tensor-Core-starved call on your attention kernel. The algorithm was proposing tokens efficiently, but the kernel was wasting the verification. Your speedup was capped by kernel inefficiency, not acceptance rate.

**The fix.** A decode kernel that **folds the K speculative tokens into the BMM1 M tile** turns the verification into a full-tile matmul. This is precisely the interaction behind TokenSpeed's "nearly halves latency with speculative decoding" claim. The lesson: **speculative decoding and the attention kernel are co-designed** — a great draft model behind a tile-starved verification kernel leaves most of the win on the table.

### 5. FP8 KV cache under long-context memory pressure

**The symptom.** Around 60K tokens of context on a wide batch, the server starts evicting requests and the eviction churn tanks throughput — requests get admitted, spilled, requeued, and re-prefilled in a thrash loop.

**The wrong first hypothesis.** "We need more HBM." More HBM helps linearly, but the KV cache is growing with context × batch and you are fighting the wrong axis.

**The actual root cause.** A BF16 KV cache at 60K tokens across the batch simply does not fit, so the scheduler's `evicted → spill → requeue` path is firing constantly, and every requeue pays a re-prefill cost.

**The fix.** Store the KV cache in **FP8 E4M3** — roughly halving its footprint — using the fused K/V-pack-plus-quantize kernel so the cast is not a separate pass, and let decode **write BF16 output** so accuracy holds. The context now fits, eviction churn subsides, and the admission controller stops thrashing. The lesson: **at long context, KV precision is a capacity decision**, and the store-FP8/compute-BF16 split buys you the capacity without the accuracy hit. The [TurboQuant deep dive on KV-cache quantization](/blog/machine-learning/open-source-library/turboquant-kv-cache-quantization-deep-dive) covers the accuracy side of this trade in more depth.

### 6. Bring-your-own-silicon without forking the engine

**The setup.** You run AMD MI355X accelerators and want TokenSpeed's scheduler and entrypoint, but the NVIDIA CuteDSL MLA kernel obviously does not apply to you.

**The wrong first hypothesis.** "We have to fork the engine and rip out the NVIDIA kernels." That path forks you off upstream forever and you inherit every future merge conflict.

**The actual approach.** Write a **Triton Gluon** decode kernel tuned for your accelerator and register it from *your own package* via `@register_kernel`, scoped to `silicon="amd:..."` and the dtype/shape constraints it supports. At dispatch, `tokenspeed-kernel` scores it alongside the built-ins and selects it on your hardware; on NVIDIA hardware the CuteDSL kernel still wins. The lesson: **the plugin boundary is the portability story** — a kernel author extends coverage out-of-tree, and the Triton portable path is the fallback that keeps everything else running while you optimize the one kernel that matters.

### 7. Parallelism without rewriting the model

**The symptom.** Your team supports five deployment shapes — single-GPU dev, TP2 staging, TP8 prod, a TP4/TP4 split for the MoE model, and a quantized edge variant — and the model code has accreted five code paths of hand-written `all_reduce`/`all_gather` guarded by config flags. Every new shape is a model PR, and the collectives are a recurring source of subtle numerical bugs.

**The actual approach.** Move to the **local-SPMD** model: annotate tensor placement at module boundaries once, and let the static compiler emit the correct collectives for whatever mesh you configure. The five shapes become five compiler configurations over *one* model source. The lesson: **parallelism belongs in a compiler, not in the model** — hand-written collectives are correct only until the next refactor, while generated ones are correct by construction for the declared placement. This is the same lesson the multi-node training world learned; the blog's [multi-node LLM training recipe](/blog/machine-learning/mlops/multi-node-llm-training-recipe-troubleshooting) catalogs the bugs that hand-written collectives cause.

### 8. Designing to an SLO, not a benchmark

**The symptom.** Your capacity planning is built around a throughput benchmark, and in production you keep getting paged for "the agent feels slow" even though the dashboard shows healthy fleet TPM and GPUs at high utilization.

**The wrong first hypothesis.** "Utilization is high, so we're efficient; the complaints are subjective." They are not subjective — they are a per-user TPS violation your fleet-level dashboard cannot see.

**The actual approach.** Re-instrument around the **per-user TPS floor** and let the scheduler's **admission controller** protect it: size the fleet so that admission rarely rejects, and when load spikes, shed it by rejecting or queueing new requests rather than degrading the in-flight ones below the floor. Capacity planning becomes *floor* planning. The lesson: **a healthy fleet metric can hide a broken user experience** — the FSM's `rejected` state is a feature, and the SLO you protect should be the one the human feels.

### 9. Prefix caching across agent turns

**The symptom.** Your agent's *time-to-first-token* grows every turn. Turn 2 responds quickly; by turn 30 the user waits eight seconds before the first token appears, even though the actual generation, once it starts, is fast.

**The wrong first hypothesis.** "Generation is slow at long context." But you measure, and decode TPS is fine — the latency is all *before* the first token, in prefill.

**The actual root cause.** The engine is re-running prefill over the entire accumulated context every turn. At turn 30 that is 45K tokens of prompt re-processed from scratch to produce a few hundred new ones. The shared prefix — the system prompt, the tools, and the first 29 turns — is being recomputed every single turn, and prefill is FLOP-bound, so the cost grows with the context you have already paid for once.

**The fix.** A paged KV layout with **prefix reuse**: turn N+1 reuses turn N's KV pages for the shared prefix and only prefills the new suffix, with the scheduler refcounting shared pages so none is freed while a turn still references it. Time-to-first-token drops from O(total context) to O(new tokens). The lesson: in a multi-turn agent, **most of every prompt is something you already computed** — an engine that cannot reuse prefix KV is doing quadratic work for a linear conversation.

### 10. Choosing `page_size` for the paged MLA KV cache

**The setup.** You are tuning the paged KV cache and have to pick a `page_size`. TokenSpeed defaults to **64**, and you wonder whether to change it.

**The exploration.** Smaller pages (say 16) pack memory tightly — less waste in the partially-filled last page of each request — but multiply the number of entries in every block table, adding indirection overhead to each KV access and bloating the page-table metadata. Larger pages (say 256) cut indirection and metadata but waste more memory in the tail page of every request, which at a wide batch of variable-length agents adds up fast.

**The result.** A `page_size` of 64 is the balance the kernels are tuned around: large enough that block-table indirection is cheap and the split-KV loads stay contiguous (which is what lets L2 prefetch work), and small enough that tail-page fragmentation across a ragged agent batch stays modest. The lesson: **`page_size` is a fragmentation-versus-indirection knob**, the kernels are co-tuned for the default, and you should only move it with a benchmark of *your* context-length distribution in hand — because the right answer depends on how ragged your batch actually is.

## When to reach for TokenSpeed, and when not to

![Choosing an inference engine for agentic workloads](/imgs/blogs/tokenspeed-agentic-inference-engine-9.png)

The matrix above is the honest decision aid. TokenSpeed is not strictly better than vLLM, TensorRT-LLM, or SGLang on every axis — it makes a specific bet, and the bet is "best-in-class agentic decode latency plus out-of-tree kernel extensibility, in exchange for preview-stage maturity." Read the columns as a chooser, not a scoreboard.

**Reach for TokenSpeed when:**

- Your workload is genuinely **agentic** — long contexts (tens of thousands of tokens), many turns, and a **per-user TPS floor** you must hold rather than a fleet throughput number you must maximize.
- You are on **NVIDIA Blackwell** (B200-class) and want the MLA decode kernel and `fold_sq_factor` win, ideally paired with **speculative decoding**.
- You serve **MLA-based MoE models** (Kimi K2.5, DeepSeek V4, Qwen 3.6, Minimax M2.7, Nemotron) where the attention compression and split-mesh parallelism pay off.
- You need to **add or tune kernels for new silicon** and want to do it out-of-tree via `@register_kernel` instead of forking.
- You want to **sweep parallelism layouts** (attention-TP vs MoE-TP) without rewriting model code.

**Skip TokenSpeed when:**

- You are running a **classic chatbot** workload — short prompts, short completions, throughput is the metric — where mature engines are already excellent and the agentic optimizations buy you little.
- You need **production-grade stability today**; TokenSpeed is an early preview, and vLLM/SGLang/TensorRT-LLM have years of hardening, broader model coverage, and larger communities.
- Your hardware is **not Blackwell (or a supported AMD path)** — the headline kernels target Blackwell, and on older silicon you fall back to the portable Triton path and lose the marquee speedups.
- Your models use **vanilla multi-head or GQA attention**, not MLA — the MLA-specific kernels are the whole point, and a non-MLA model leaves the best optimizations unused.
- You need a **broad ecosystem of integrations** (structured output, LoRA hot-swapping, the long tail of features mature servers accumulate) right now.

The meta-lesson, and the reason TokenSpeed is worth studying even if you never deploy it, is that it is a clean worked example of **designing an entire system around one SLO**. Every layer — the placement-annotation compiler, the typed-FSM scheduler with its admission control, the pluggable kernels, the `fold_sq_factor` decode trick — traces back to a single sentence: *maximize per-GPU TPM subject to a per-user TPS floor, for long-context multi-turn agents on Blackwell.* When you can write your objective that precisely, the architecture mostly writes itself. That clarity is the thing to borrow, whatever engine you ship.

## Further reading

- [TokenSpeed on GitHub](https://github.com/lightseekorg/tokenspeed) — the engine, the `tokenspeed-kernel` library, and the `tokenspeed-mla` kernels.
- [Multi-head Latent Attention](/blog/machine-learning/large-language-model/multi-head-latent-attention-mla) — the attention mechanism the MLA kernels implement.
- [DeepSeek's open infra: DeepEP, DeepGEMM, FlashMLA](/blog/machine-learning/mlops/deepseek-open-infra-deepep-deepgemm-flashmla) — the sibling kernel stack TokenSpeed's MLA path is racing against.
- [TensorRT as an end-to-end inference compiler](/blog/machine-learning/mlops/tensorrt-end-to-end-inference-compiler) — the vendor-specific baseline TokenSpeed benchmarks itself against.
- [Optimizing LLM inference: a complete guide](/blog/machine-learning/large-language-model/optimizing-llm-inference-complete-guide) — the broader set of techniques this engine composes.
- [Choosing a GPU for LLM serving](/blog/machine-learning/large-language-model/choosing-gpu-for-llm-serving-cost-throughput-latency) — the cost/throughput/latency framing behind the TPM-vs-TPS objective.

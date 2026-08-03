---
title: "Hybrid versus transformer at long context: the experiment you should run before you believe anyone"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Design the head-to-head benchmark that settles whether a hybrid attention plus state-space model actually beats a pure transformer at 128k context, derive every curve it should produce before you run it, and find the workloads where the hybrid honestly loses."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "benchmarking",
    "long-context",
    "hybrid-models",
    "state-space-models",
    "kv-cache",
    "latency",
    "throughput",
    "pytorch",
    "gpu",
    "vllm",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 52
---

Here are two real, published claims about hybrid attention plus state-space models. NVIDIA's [Nemotron-H technical report (arXiv 2504.03624)](https://arxiv.org/abs/2504.03624) says its models are "up to 3× faster at inference" than similarly-sized transformers such as Qwen-2.5-7B and Llama-3.1-8B. The vLLM team's [MiniMax-M1 post (2025-06-30)](https://vllm.ai/blog/2025-06-30-minimax-m1) says lightning attention "reduces memory 83% and inference latency 67% for 100k-token sequences." Both are from credible primary sources. Both are almost certainly true as stated.

And you cannot act on either of them.

"Up to 3× faster" — at what batch size, at what context length, measured as time-to-first-token or as steady-state tokens per second, under closed-loop or open-loop load, on which GPU, against which serving stack? The abstract does not say. "83% less memory" — less than what baseline, at what precision, and is that the whole request footprint or just the sequence-mixing layers? The post does not say. These are not sloppy claims; they are *compressed* claims, and the compression threw away exactly the information you need to decide whether to change the model behind your production endpoint. A number without its setup is a rumour with a decimal point.

![Two columns comparing a pure transformer and a hybrid model on the same eighty gigabyte GPU at a context of one hundred twenty-eight thousand tokens, showing the concurrency each one reaches](/imgs/blogs/experiment-hybrid-vs-transformer-at-long-context-1.webp)

The figure above is the result this post is going to derive, and then teach you to verify. Same GPU — one 80 GB A100. Same context — 128k tokens per request. Same precision — bf16. Change only the architecture, and the concurrency ceiling moves from three requests to twenty-six, and the aggregate decode throughput from about 90 tokens per second to about 721. That is not a 3× claim or an 83% claim. It is a specific number attached to a specific configuration, derived from arithmetic you will see in full, and it comes with an equally specific list of the workloads where the hybrid is the *worse* choice.

This is the experiment post that closes the hybrid track. By the end you will be able to: state the benchmark protocol that makes an architecture comparison trustworthy (locked clocks, warmup, CUDA events, open-loop arrivals, steady state, seeds, a fixed prompt suite); derive — before running anything — the memory curve, the TTFT curve, the TPOT curve and the goodput curve that each architecture *must* produce if the physics is what we think it is; run `nanoserve/bench/longcontext.py` yourself and check your machine against those predictions; and give an honest per-workload verdict instead of a headline. You will also learn the two things the hybrid gives up, and why every serious hybrid keeps some full-attention layers rather than going all the way.

Two promises inherited from [the series introduction](/blog/machine-learning/inference-engineering/what-inference-engineering-is), and they bind harder in an experiment post than anywhere else. **First: I have no GPU and I have run nothing.** There is not one measured number in this post. Every figure is either derived from arithmetic I show you, cited from a paper or vendor post with a link and its setup, or framed as something *you* will produce with a named script and an expected range. Every results table carries a `Source` column that says which. **Second: I will not name a model or a ratio I could not verify against a primary source.** Several plausible hybrids are absent from this post for exactly that reason.

---

## 1. Why the honest version of this question is harder than it looks

Start by admitting what makes an architecture comparison uniquely easy to get wrong.

When you benchmark two *kernels*, you can hold everything else constant: same model, same shapes, same data, same machine. The only thing that changes is the kernel. When you benchmark two *architectures*, almost nothing is constant. The two models have different parameter counts, different training data, different tokenizers, different context windows they were actually trained for, different quality on your task. If you measure tokens per second and the hybrid wins, you have not learned that hybrids are faster — you may have learned that this particular hybrid checkpoint is smaller, or that its tokenizer emits fewer tokens for your prompts, or that its default `max_num_seqs` is higher.

So the experiment has to be designed around a specific, narrow, answerable question. Here is the one worth answering:

> **At a fixed VRAM budget and a fixed latency SLO, how does each architecture's serving capacity change as context length grows from 4k to 128k, and what does the hybrid give up in return?**

Notice what that question does. It fixes the resource (VRAM), fixes the quality bar (an SLO), and makes context length the independent variable. It asks for a *curve*, not a point. And it demands the cost side be reported alongside the benefit, so that "faster" cannot quietly mean "faster and worse."

The two systems under test, and why these two:

- **Pure transformer**: Llama-3.1-8B. The spine model of this whole series. From its config: 32 layers, 32 query heads, 8 key-value heads (grouped-query attention), head dimension 128, bf16. About 8.03B parameters.
- **Hybrid**: Nemotron-H-8B. Its config declares 52 layers with the `hybrid_override_pattern` string, which decomposes into 24 Mamba-2 layers, 24 feed-forward layers, and exactly **4** self-attention layers — a 1:6 attention-to-recurrent ratio among the sequence-mixing layers. Same grouped-query shape on those four attention layers: 8 key-value heads, head dimension 128. About 8.2B parameters. The architecture is documented in the [Nemotron-H report (arXiv 2504.03624)](https://arxiv.org/abs/2504.03624), and Mamba-2 itself in [Dao and Gu's state space duality paper (arXiv 2405.21060)](https://arxiv.org/abs/2405.21060).

Both roughly 8B. Both bf16. Both grouped-query attention where they have attention at all. That is about as close to a controlled comparison as two independently-trained checkpoints get, and the residual confounds — different training data, different tokenizers — are exactly why the recall measurement in section 10 is not optional.

One honest caveat up front, and it is a big one. The published `Nemotron-H-8B-Base-8K` checkpoint is trained for an 8K context window; the suffix says so. Everything this post derives about its behaviour at 128k is about the **architecture's memory and bandwidth shape**, not a claim that this specific checkpoint produces good text at 128k. If you want to actually serve 128k on a hybrid you need a hybrid trained for it — the vLLM team's [Qwen3-Next post (2025-09-11)](https://vllm.ai/blog/2025-09-11-qwen3-next) describes a hybrid that interleaves Gated DeltaNet with full attention and states a context of "65K and beyond," and the MiniMax-M1 post names M1-40k and M1-80k variants. Use the shape arithmetic here; pick a checkpoint whose model card claims the context you need.

---

## 2. The scoreboard, and what "win" is allowed to mean

Before a protocol, a scoreboard. These are the seven quantities the experiment reports, and each exists because some real decision depends on it.

**TTFT — time to first token.** From request submission to the first streamed token. Dominated by prefill: the single forward pass over the prompt that populates the cache. This is the number a user *feels* as "did it hear me." At long context it is the number that goes catastrophically wrong first.

**TPOT — time per output token.** The average gap between consecutive output tokens once generation is underway (vLLM's [anatomy post](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) defines the per-gap version as ITL, inter-token latency, and TPOT as its average over the output). This sets the perceived reading speed. A TPOT of 25 ms reads as about 40 tokens per second, roughly comfortable reading pace.

**Output tokens per second, per request.** Just $1/\text{TPOT}$, but report it because humans reason about it more naturally than milliseconds.

**Aggregate throughput (goodput).** Total output tokens per second across all concurrent requests — but only counting requests that met the SLO. The distinction matters enormously and the industry is sloppy about it. A server that emits 5,000 tokens per second while every single request violates a 1-second TTFT budget has a throughput of 5,000 and a goodput of zero.

**p99 latency.** Not p50. The tail is where preemption, queueing and long-prompt stalls live, and the tail is what your users complain about.

**Peak device memory, and the concurrency it permits.** `torch.cuda.max_memory_allocated()` plus the reserved figure from `torch.cuda.memory_reserved()`, and from those the largest batch that fits.

**A recall metric.** This is the one people skip, and skipping it is how you ship a regression. Report accuracy on a long-context retrieval task at *each* context length in the sweep. The reference benchmark here is [RULER (Hsieh et al., arXiv 2404.06654)](https://arxiv.org/abs/2404.06654), which extends the popular needle-in-a-haystack test with multi-needle retrieval, variable tracking and aggregation tasks. Its headline finding is exactly why you need it: evaluating 17 long-context models across 13 tasks, the authors report that although models claim 32K-token context or more, **only about half maintain satisfactory performance at 32K**, and nearly all degrade sharply with length despite near-perfect scores on the vanilla single-needle test. A model that "supports 128k" and a model that is *useful* at 128k are different claims.

And the derived business number:

$$\text{cost per 1M tokens} \;=\; \frac{10^{6}}{\text{goodput} \times 3600} \times r$$

where $r$ is your dollars-per-GPU-hour. Note what this expression does: the **ratio** between two architectures is independent of $r$, so the comparison is portable even though the absolute figure is not. Any absolute \$-per-million number in this post uses an illustrative placeholder rate that I state explicitly, because I am not going to quote a market price I cannot source.

A win, then, is not "higher tok/s." A win is: **more goodput at the same SLO and the same VRAM, without losing recall on the workloads you actually serve.** Three conditions. Drop any one and the comparison stops meaning anything.

---

## 3. The protocol: everything you do before you are allowed to record a number

![Seven ordered protocol steps running from locking the GPU clocks through warmup and open-loop load generation to reporting percentiles and a recall score](/imgs/blogs/experiment-hybrid-vs-transformer-at-long-context-2.webp)

The timeline above is the whole protocol, and the point it makes visually is that six of the seven steps happen *before* any number is written down. Each one removes a specific source of variance. Skip one and your two curves differ by an amount that has nothing to do with architecture. (A fuller treatment of benchmark hygiene lives in [setting up a reproducible benchmark](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark); what follows is the long-context-specific version.)

### 3.1 Lock the clocks

A modern GPU boosts and throttles continuously based on temperature and power. Run the same kernel twice, once on a cold card and once after ten minutes of load, and you can see double-digit percentage differences with no code change at all. That is enough to invent an architecture "result" out of thin air.

```bash
# Persistence mode on, then pin the SM and memory clocks to a fixed point.
sudo nvidia-smi -pm 1
nvidia-smi --query-supported-clocks=gr --format=csv | head
sudo nvidia-smi -lgc 1350,1350     # pick a value your card sustains
sudo nvidia-smi -i 0 --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv
```

Pick a clock the card can hold indefinitely, not its boost peak. Then confirm during the run, not just before it: sample `clocks.sm` and `clocks_throttle_reasons.active` every few seconds, and if throttling appears anywhere in the window, the run is invalid. Reset with `sudo nvidia-smi -rgc` afterwards.

If you are on a shared or cloud instance where you cannot lock clocks, say so in the report and widen your error bars. An honest "±15% because clocks were unlocked" beats a precise-looking number that is not reproducible.

### 3.2 Warm up, then throw the warmup away

The first invocation of any CUDA path in a process is not representative of the tenth. Lazy module loading, cuBLAS handle creation, autotuning of Triton kernels, `torch.compile` tracing and compilation, CUDA graph capture, and allocator growth all happen once. For an engine using `torch.compile` with piecewise CUDA graphs — the default in vLLM V1, per the [V1 architecture post](https://vllm.ai/blog/2025-01-27-v1-alpha-release) — the first few steps can be *seconds* slower than steady state.

Rule: run at least 30 full decode steps at the target shape, discard every one, and only then start recording. And critically, **warm up at each context length separately**, because a 4k prefill and a 128k prefill may select different attention kernels entirely.

### 3.3 Synchronize before you time, and prefer CUDA events

CUDA kernel launches are asynchronous. `time.perf_counter()` around a `model.forward()` measures how long it took the CPU to *enqueue* work, which on a well-pipelined engine is nearly zero. This produces the classic beginner result of a 10,000 tok/s language model.

```python
import torch

# WRONG: measures enqueue time, not execution time.
t0 = time.perf_counter()
out = model(input_ids)
t1 = time.perf_counter()          # meaningless

# Acceptable: synchronize on both sides.
torch.cuda.synchronize()
t0 = time.perf_counter()
out = model(input_ids)
torch.cuda.synchronize()
t1 = time.perf_counter()

# Better: CUDA events time on the device timeline, no host stall in between.
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
out = model(input_ids)
end.record()
torch.cuda.synchronize()          # one sync, after both records
elapsed_ms = start.elapsed_time(end)
```

Events are better because a `torch.cuda.synchronize()` inside your timing loop drains the pipeline on every iteration, which is exactly the overlap a real server depends on. Measure with events; sync once at the end.

For an end-to-end server benchmark over HTTP, you are measuring wall-clock at the client anyway, and there the requirement flips: measure from the client, include queueing, and do not let the engine's internal timers substitute for it. Your users experience the queue.

### 3.4 Open loop, not closed loop

This is the single most common structural error in serving benchmarks, and it silently favours whichever system has lower latency in a way that has nothing to do with capacity.

A **closed-loop** load generator keeps $N$ requests in flight: as soon as one finishes, it issues the next. A **open-loop** generator issues requests at a rate $\lambda$ drawn from a Poisson process regardless of whether previous ones finished. Real traffic is open-loop. Users do not wait for your server to be ready before typing.

Why it matters: in closed loop, a slow server automatically receives less load, because the generator is throttled by the server's own latency. Queueing never builds. You cannot observe latency collapse, you cannot observe the tail, and you cannot find the capacity limit — the system appears to degrade gracefully forever. In open loop, when arrival rate exceeds service rate the queue grows without bound and p99 goes vertical, which is the failure you are trying to locate.

Practically: sweep $\lambda$ upward, and for each $\lambda$ record p50 and p99 TTFT and TPOT plus the fraction of requests meeting SLO. The capacity number you report is the highest $\lambda$ at which the SLO fraction stays above your threshold. That is goodput, and it is the number that predicts your production behaviour. (This is the same framing vLLM used in its [MoRIIO connector post](https://vllm.ai/blog/2026-04-07-moriio-kv-connector), where the SLO was stated as TTFT under one second and ITL under 50 ms, and the goodput comparison counted only requests meeting both.)

### 3.5 Steady state only

Even under open-loop arrivals, the first minute of a run is a transient: the KV cache is empty, prefix caching has no hits, the scheduler queue is short, the allocator has not reached its working-set size. Discard the first 60 seconds and the last 10 (which is drain), and analyse only the middle. Report the window you used.

### 3.6 Control the seeds, and control the prompts

Sampling is stochastic. Two runs of the same request will produce different output lengths, which changes the number of decode steps, which changes everything downstream. For a latency benchmark, fix `seed`, and better still fix the *output length* by setting `min_tokens == max_tokens` and disabling stop strings, so every request does exactly the same amount of decode work. You are measuring the engine, not the model's tendency to be chatty.

The prompt suite, held constant across both models and every context length:

| Workload | Input | Output | What it stresses |
| --- | --- | --- | --- |
| Chat | 4k tokens | 512 tokens | Decode-dominated; weights and per-step overhead |
| RAG | 4k → 128k tokens | 256 tokens | Prefill-dominated; the KV memory wall |
| Code completion | 32k tokens (repo map) | 256 tokens | Mid-length, prefix-reuse sensitive |
| Translation | 2k tokens | 2k tokens | Balanced; long decode at modest context |
| Needle recall | 4k → 128k tokens | 32 tokens | Exact retrieval quality, not speed |

One subtlety that will bite you: **the two models have different tokenizers**, so "the same prompt" is not the same number of tokens. Build the suite by *token count*, not by character count — generate filler until `len(tokenizer(text).input_ids)` hits the target for each model independently. Otherwise you are comparing a 128k-token request against a 119k-token request and attributing the difference to architecture.

### 3.7 What you report

For every (model, context, arrival-rate) cell: p50/p99 TTFT, p50/p99 TPOT, per-request output tok/s, aggregate goodput at SLO, peak allocated and reserved memory, the largest batch that ran without OOM, the recall score at that context, and the derived \$-per-million-output-tokens at a stated rate. Plus the invariants: GPU model, driver, CUDA, torch, engine version, clock lock, dtype, and the exact commit of your harness.

That is more bookkeeping than most benchmarks do. It is also the difference between a result and an anecdote.

---

## 4. Prediction one: the memory ceiling, and where each architecture stops

Now the derivations. The value of deriving before measuring is that it turns your benchmark from an exploration into a *test*: you know what the numbers should be, so a deviation is information rather than noise.

### 4.1 The two memory laws

For a pure transformer, the per-request cache is a line through the origin. With $L$ layers, $H_{kv}$ key-value heads, head dimension $d$, and $b$ bytes per element, storing both K and V:

$$M_{\text{full}}(S) \;=\; \underbrace{2 \cdot L \cdot H_{kv} \cdot d \cdot b}_{B_{\text{tok}}} \;\cdot\; S$$

For Llama-3.1-8B in bf16: $2 \times 32 \times 8 \times 128 \times 2 = 131{,}072$ bytes per token, exactly 128 KiB. Useful shorthand: **one GiB of cache buys 8,192 tokens.**

For a hybrid where a fraction $f$ of the $L$ mixing layers are attention and the rest carry a fixed per-layer state $\sigma$, the law grows an intercept:

$$M_{\text{hyb}}(S) \;=\; \underbrace{2 \cdot f L \cdot H_{kv} \cdot d \cdot b \cdot S}_{\text{slope: grows with } S} \;+\; \underbrace{(1-f)\,L \cdot \sigma}_{\text{intercept: constant}}$$

For Nemotron-H-8B: four attention layers give $2 \times 4 \times 8 \times 128 \times 2 = 16{,}384$ bytes per token, exactly 16 KiB — one eighth of Llama's slope. The 24 Mamba-2 layers contribute a constant. [The previous post in this track](/blog/machine-learning/inference-engineering/hybrid-models-and-the-end-of-the-kv-cache-assumption) derives the per-layer state from the config as 2,158,592 bytes, so the total intercept is $24 \times 2{,}158{,}592 = 51{,}806{,}208$ bytes, or 49.4 MiB. Shorthand: **one GiB of hybrid cache buys 65,536 tokens**, plus a flat 49.4 MiB toll.

The full derivation of $B_{\text{tok}}$ and why grouped-query attention is the term that matters lives in [the memory math of the KV cache](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache); here we only need the result.

### 4.2 Solving for the ceiling

Serving capacity at a fixed VRAM budget $V$ is not a matter of taste. Let $W$ be weight bytes and $R$ a reserve for activations, workspace and CUDA context. The free budget is $F = V - W - R$, and the maximum total cached tokens across all concurrent requests is $F / B_{\text{tok}}$ for the transformer, and $(F - n\sigma_{\text{tot}})/B_{\text{tok}}^{\text{hyb}}$ for the hybrid at concurrency $n$. Set $n = 1$ to find the single-request context ceiling:

$$S_{\max}^{\text{full}} \;=\; \frac{F}{B_{\text{tok}}}, \qquad S_{\max}^{\text{hyb}} \;=\; \frac{F - \Sigma}{B_{\text{tok}}^{\text{hyb}}}$$

#### Worked example: at what context does each model OOM a 24 GB RTX 4090?

The consumer baseline of this series. NVIDIA's [GeForce RTX 4090 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/) list 24 GB of GDDR6X at 1008 GB/s. `nvidia-smi` typically reports about 24,564 MiB usable; after the CUDA context and driver overhead, budget **23.0 GiB** and verify on your own card.

**Llama-3.1-8B.** Weights: $8.03 \times 10^{9} \times 2 = 16.06$ GB, which is 14.96 GiB. Reserve 1.5 GiB for activations, logits and a 2048-token prefill chunk's workspace. Free for cache:

$$F \;=\; 23.0 - 14.96 - 1.5 \;=\; 6.54 \text{ GiB}$$

$$S_{\max} \;=\; 6.54 \times 8192 \;=\; 53{,}575 \text{ tokens}$$

**Fifty-three thousand tokens.** Not 128k — not even 64k. A single 128k-token request needs 16.0 GiB of cache on a card that has 6.54 GiB free. The transformer does not run this workload on this GPU at all, at any batch size, and no scheduler tuning fixes it.

**Nemotron-H-8B.** Weights: about $8.2 \times 10^{9} \times 2 = 16.4$ GB, or 15.27 GiB. Same 1.5 GiB reserve. Subtract the 49.4 MiB (0.048 GiB) state intercept:

$$F \;=\; 23.0 - 15.27 - 1.5 - 0.048 \;=\; 6.18 \text{ GiB}$$

$$S_{\max} \;=\; 6.18 \times 65{,}536 \;=\; 405{,}012 \text{ tokens}$$

Source: derived, using the RTX 4090 memory figure cited from NVIDIA's specification page and the two config-derived per-token constants. **7.6× more context on the same card**, and the difference is entirely the slope of one line.

That last sentence deserves the caveat from section 1 restated at full volume: 405k is the *architecture's* memory headroom, not a claim that the 8K-trained checkpoint produces sensible text at 405k. Memory headroom and trained context are independent constraints, and a benchmark that reports the first while implying the second is dishonest. The correct statement is: the hybrid's memory stops being the binding constraint long before its training does, and for the transformer the opposite is true.

![A memory budget branching into a transformer path that caps early and a hybrid path that reaches far longer context, with a recall caution rejoining both into a per-workload verdict](/imgs/blogs/experiment-hybrid-vs-transformer-at-long-context-3.webp)

The graph above is the shape of the whole result and worth pausing on: one number at the top — free VRAM — divides by two different per-token costs and produces two wildly different fates, and the two branches only rejoin because the hybrid path carries a caution the transformer path does not. That caution node is section 10, and it is the reason the verdict at the bottom says "per workload" rather than naming a winner.

Now watch the two curves race the ceiling.

<figure class="blog-anim">
<svg viewBox="0 0 700 330" role="img" aria-label="Two memory bars growing as context length increases from four thousand to two hundred fifty-six thousand tokens, where the transformer bar reaches the free-memory ceiling and overflows while the hybrid bar stays low" style="width:100%;height:auto;max-width:860px">
<style>
.mx-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.mx-sub{font:400 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.mx-ceil{font:600 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.mx-track{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.mx-bar{fill:var(--accent,#6366f1);transform-box:fill-box;transform-origin:bottom}
.mx-line{stroke:var(--text-secondary,#6b7280);stroke-width:2;stroke-dasharray:6 5}
.mx-oomtxt{font:700 15px ui-sans-serif,system-ui;fill:var(--accent,#6366f1);text-anchor:middle}
.mx-step{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
@keyframes mx-tf{0%{transform:scaleY(.078)}25%{transform:scaleY(.61)}50%{transform:scaleY(1)}75%{transform:scaleY(1)}100%{transform:scaleY(.078)}}
@keyframes mx-hy{0%{transform:scaleY(.017)}25%{transform:scaleY(.084)}50%{transform:scaleY(.313)}75%{transform:scaleY(.619)}100%{transform:scaleY(.017)}}
@keyframes mx-oom{0%,49%{opacity:0}50%,99%{opacity:1}100%{opacity:0}}
@keyframes mx-s1{0%{opacity:1}25%,100%{opacity:.12}}
@keyframes mx-s2{0%{opacity:.12}25%{opacity:1}50%,100%{opacity:.12}}
@keyframes mx-s3{0%,25%{opacity:.12}50%{opacity:1}75%,100%{opacity:.12}}
@keyframes mx-s4{0%,50%{opacity:.12}75%{opacity:1}100%{opacity:.12}}
.mx-a1{animation:mx-tf 14s steps(1,end) infinite}
.mx-a2{animation:mx-hy 14s steps(1,end) infinite}
.mx-a3{animation:mx-oom 14s steps(1,end) infinite}
.mx-b1{animation:mx-s1 14s steps(1,end) infinite}
.mx-b2{animation:mx-s2 14s steps(1,end) infinite}
.mx-b3{animation:mx-s3 14s steps(1,end) infinite}
.mx-b4{animation:mx-s4 14s steps(1,end) infinite}
@media (prefers-reduced-motion:reduce){.mx-a1{animation:none;transform:scaleY(1)}.mx-a2{animation:none;transform:scaleY(.313)}.mx-a3{animation:none;opacity:1}.mx-b1,.mx-b2,.mx-b4{animation:none;opacity:.12}.mx-b3{animation:none;opacity:1}}
</style>
<rect class="mx-track" x="150" y="60" width="110" height="190" rx="6"/>
<rect class="mx-track" x="430" y="60" width="110" height="190" rx="6"/>
<rect class="mx-bar mx-a1" x="150" y="60" width="110" height="190" rx="6"/>
<rect class="mx-bar mx-a2" x="430" y="60" width="110" height="190" rx="6"/>
<line class="mx-line" x1="110" y1="60" x2="600" y2="60"/>
<text class="mx-ceil" x="110" y="50">free VRAM ceiling · 6.54 GiB on a 24 GB RTX 4090</text>
<text class="mx-oomtxt mx-a3" x="205" y="42">OOM</text>
<text class="mx-lbl" x="205" y="275">Llama-3.1-8B</text>
<text class="mx-sub" x="205" y="293">128 KiB per token</text>
<text class="mx-lbl" x="485" y="275">Nemotron-H-8B</text>
<text class="mx-sub" x="485" y="293">16 KiB + 49 MiB flat</text>
<text class="mx-step mx-b1" x="350" y="318">context 4k · 0.50 GiB vs 0.11 GiB</text>
<text class="mx-step mx-b2" x="350" y="318">context 32k · 4.00 GiB vs 0.55 GiB</text>
<text class="mx-step mx-b3" x="350" y="318">context 128k · 16.0 GiB vs 2.05 GiB</text>
<text class="mx-step mx-b4" x="350" y="318">context 256k · 32.0 GiB vs 4.05 GiB</text>
</svg>
<figcaption>Per-request cache for one request as context grows across the sweep. The transformer bar reaches the free-memory ceiling between 32k and 128k and stays pinned there because the real value is off the chart; the hybrid bar is still under two thirds of the ceiling at 256k. All values derived from the two per-token constants.</figcaption>
</figure>

The motion carries the argument better than a table does: the transformer bar is not merely taller, it *leaves the chart*, and it does so between two adjacent points in a normal sweep. If your benchmark grid is 4k, 8k, 16k, 32k you will never see it happen and you will conclude the two architectures are similar.

#### Worked example: concurrency and cost on one 80 GB A100

Switch to the datacentre card, where 128k actually fits for both. NVIDIA's [A100 product page](https://www.nvidia.com/en-us/data-center/a100/) lists the 80 GB SXM part at 2039 GB/s of HBM2e and 312 TFLOP/s of bf16 tensor throughput. 80 GB is 74.5 GiB. Reserve 4 GiB.

- Llama free: $74.5 - 14.96 - 4.0 = 55.54$ GiB. At 16.0 GiB per 128k request: $\lfloor 55.54/16.0 \rfloor = \mathbf{3}$ concurrent.
- Nemotron-H free: $74.5 - 15.27 - 4.0 = 55.23$ GiB. At 2.05 GiB per 128k request: $\lfloor 55.23/2.05 \rfloor = \mathbf{26}$ concurrent.

Source: derived, with the 80 GB capacity cited from NVIDIA. (If you assume identical weights for both models you get 27 rather than 26 — the difference is Nemotron-H's slightly larger parameter count, and it is a good illustration of how sensitive a capacity number is to assumptions you did not write down.)

The full memory curve, all derived from the two constants:

| Context $S$ | Llama-3.1-8B cache | Nemotron-H-8B cache | Ratio | Fits on 4090? | Source |
| --- | --- | --- | --- | --- | --- |
| 4,096 | 512 MiB | 49.4 + 64.0 = 113.4 MiB | 4.51× | both | derived |
| 16,384 | 2.00 GiB | 49.4 MiB + 256 MiB = 305 MiB | 6.71× | both | derived |
| 32,768 | 4.00 GiB | 49.4 MiB + 512 MiB = 561 MiB | 7.30× | both | derived |
| 65,536 | 8.00 GiB | 49.4 MiB + 1.00 GiB = 1.05 GiB | 7.63× | hybrid only | derived |
| 131,072 | 16.0 GiB | 49.4 MiB + 2.00 GiB = 2.05 GiB | 7.81× | hybrid only | derived |
| 262,144 | 32.0 GiB | 49.4 MiB + 4.00 GiB = 4.05 GiB | 7.90× | hybrid only | derived |
| 1,048,576 | 128 GiB | 49.4 MiB + 16.0 GiB = 16.05 GiB | 7.97× | neither | derived |

Read the ratio column carefully, because it contains a result the marketing never mentions: **the advantage saturates.** As $S \to \infty$ the ratio approaches the pure layer-count ratio of 8, and it is already at 7.30 by 32k — 91% of the theoretical maximum. Going from 32k to 128k quadruples your context and buys 7% more relative advantage. Hybrids do not "get better and better" at longer context in ratio terms. They get *absolutely* better while the ratio flatlines.

---

## 5. Prediction two: what TTFT should do as context grows

Prefill is a compute problem, not a memory problem, and the two architectures diverge for a completely different reason than they did in section 4.

The forward pass over a prompt of $S$ tokens has a **linear term** — every matmul against the weights, about $2N$ FLOPs per token for $N$ parameters — and a **quadratic term** from attention itself:

$$\text{FLOPs}_{\text{linear}} \approx 2 N S, \qquad \text{FLOPs}_{\text{attn}} \approx 2 \, L_{\text{attn}} \, H_q \, d \, S^{2}$$

where $L_{\text{attn}}$ is the number of *full-attention* layers. That subscript is the entire story. Llama-3.1-8B has 32 of them; Nemotron-H-8B has four. The quadratic constant for Llama is $2 \times 32 \times 32 \times 128 = 262{,}144$; for the hybrid it is $2 \times 4 \times 32 \times 128 = 32{,}768$. One eighth.

Set the two terms equal to find where prefill stops being a matmul machine and becomes an $S^2$ machine:

$$S_{\text{cross}} \;=\; \frac{N}{L_{\text{attn}} \, H_q \, d}$$

- **Llama-3.1-8B**: $8.03\times10^{9} / (32 \cdot 32 \cdot 128) = 61{,}264$ tokens.
- **Nemotron-H-8B**: $8.2\times10^{9} / (4 \cdot 32 \cdot 128) = 500{,}488$ tokens.

The transformer turns quadratic-dominated inside the range you actually serve. **The hybrid does not turn quadratic-dominated until half a million tokens**, which is past where anyone is serving. That is a genuinely different regime, not a constant-factor improvement, and it is the reason the prefill curves have different *shapes* and not merely different heights. The derivation of the transformer side, along with why FlashAttention makes the quadratic fast without making it not-quadratic, is in [long-context inference: RoPE scaling, sinks, and the prefill cost curve](/blog/machine-learning/inference-engineering/long-context-inference-rope-scaling-sinks-and-the-prefill-cost-curve).

| Context | Llama prefill | Hybrid prefill | Llama attn share | Hybrid attn share | Ratio | Source |
| --- | --- | --- | --- | --- | --- | --- |
| 4,096 | 0.070 PFLOP | 0.068 PFLOP | 6.3% | 0.8% | 1.04× | derived |
| 32,768 | 0.808 PFLOP | 0.573 PFLOP | 34.8% | 6.1% | 1.41× | derived |
| 131,072 | 6.61 PFLOP | 2.71 PFLOP | 68.2% | 20.8% | 2.44× | derived |

Turning FLOPs into a TTFT prediction needs an effective throughput, and this is where you must be careful not to over-claim. An A100 80 GB lists 312 TFLOP/s of bf16 dense tensor throughput, but long-context prefill does not sustain peak: the attention kernel is memory-movement-heavy, and the non-matmul work (normalization, RoPE, activation) does not use tensor cores at all. Budget roughly a third of peak, about 110 TFLOP/s effective, and treat the result as an order-of-magnitude expectation:

| Context | Llama TTFT | Hybrid TTFT | Source |
| --- | --- | --- | --- |
| 4,096 | ~0.6 s | ~0.6 s | derived at 110 TFLOP/s effective |
| 32,768 | ~7 s | ~5 s | derived at 110 TFLOP/s effective |
| 131,072 | ~60 s | ~25 s | derived at 110 TFLOP/s effective |

**Now the honest correction, and it matters.** That table is an upper bound on the hybrid's advantage, not a prediction. Two reasons:

1. **A Mamba-2 chunked scan is not a dense GEMM.** It has lower arithmetic intensity than the big projection matmuls and a more complex access pattern, so its *effective* utilization of the tensor cores is likely lower than attention's. The FLOP ratio of 2.44× is therefore an optimistic bound; a realistic expectation is somewhere between 1.3× and 2.4×, and the only way to know is to measure.
2. **I folded the scan's own arithmetic into the linear term.** The $2NS$ approximation counts the weight matmuls. The state update itself does work proportional to $S$ with a nontrivial constant that I have not separately accounted for, which biases the hybrid's number *low*. Two errors in opposite directions, both unquantified. Say so rather than pretending the table is tighter than it is.

Which is the whole point of deriving first. You now know the hybrid's TTFT advantage should be somewhere in the 1.3× to 2.4× band at 128k and essentially zero at 4k. If your run shows 5×, you have a confound — probably a different attention backend, or chunked prefill enabled on one side and not the other. If it shows 0.8×, the scan implementation is inefficient on your hardware. Either way you learned something, which is not true of a benchmark you had no expectation for.

---

## 6. Prediction three: what TPOT should do as context grows

Decode is a bandwidth problem, and this is where the two architectures diverge most cleanly.

Every decode step must read, from HBM, the entire weight matrix set plus the entire cache of every active request. At batch $n$, sequence length $S$, HBM bandwidth $\text{BW}$:

$$t_{\text{step}} \;\approx\; \frac{W \;+\; n\,\bigl(B_{\text{tok}} \cdot S \;+\; \Sigma\bigr)}{\text{BW}}$$

That is a roofline: a lower bound on step time assuming perfect bandwidth utilization and zero launch overhead. Real kernels hit some fraction of it. The reason it is still the right model is that at batch 1 an 8B decode step performs about $2N n = 1.6\times10^{10}$ FLOPs against roughly $1.6\times10^{10}$ bytes of traffic — an arithmetic intensity near 1 FLOP per byte, three orders of magnitude below the A100's ratio of roughly 153. Decode is not close to compute-bound; it is a bandwidth benchmark wearing a model's clothes. (The general framing is in [the roofline model](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound).)

![Layered bands showing the bytes a single decode step reads at long context, with a shared weight band on top and much smaller cache bands for the hybrid below the transformer's](/imgs/blogs/experiment-hybrid-vs-transformer-at-long-context-4.webp)

The stack above makes the mechanism visible in one frame. The top band — 16.06 GB of weights — is paid by both architectures on every single step and cannot be avoided. Everything below it is cache, and that is the only place the two models differ. At 4k context the cache bands are slivers and the two step times are indistinguishable. At 128k the transformer's cache band is *larger than the weight band*, and the hybrid's is one eighth of that.

#### Worked example: batch-1 TPOT at 4k versus 128k, and what you should actually observe

A100 80 GB SXM, 2039 GB/s cited from NVIDIA, batch 1, bf16.

**Llama-3.1-8B at 4k:** weights 16.06 GB, cache $131{,}072 \times 4096 = 0.537$ GB.

$$t \;=\; \frac{16.06 + 0.537}{2039} \;=\; 8.14 \text{ ms} \;\Rightarrow\; 123 \text{ tok/s ceiling}$$

**Llama-3.1-8B at 128k:** cache $131{,}072 \times 131{,}072 = 17.18$ GB.

$$t \;=\; \frac{16.06 + 17.18}{2039} \;=\; 16.30 \text{ ms} \;\Rightarrow\; 61 \text{ tok/s ceiling}$$

The cache read is *more than half the step*. TPOT exactly doubles from 4k to 128k.

**Nemotron-H-8B at 4k:** weights 16.4 GB, state 0.052 GB, cache $16{,}384 \times 4096 = 0.067$ GB.

$$t \;=\; \frac{16.4 + 0.052 + 0.067}{2039} \;=\; 8.10 \text{ ms} \;\Rightarrow\; 123 \text{ tok/s ceiling}$$

**Nemotron-H-8B at 128k:** cache $16{,}384 \times 131{,}072 = 2.147$ GB.

$$t \;=\; \frac{16.4 + 0.052 + 2.147}{2039} \;=\; 9.12 \text{ ms} \;\Rightarrow\; 110 \text{ tok/s ceiling}$$

| Model | TPOT at 4k | TPOT at 128k | Degradation | Reproduce-range at 128k | Source |
| --- | --- | --- | --- | --- | --- |
| Llama-3.1-8B | 8.14 ms | 16.30 ms | **2.00×** | 20–28 ms | derived + reproduce: `longcontext.py` |
| Nemotron-H-8B | 8.10 ms | 9.12 ms | **1.13×** | 12–20 ms | derived + reproduce: `longcontext.py` |

The reproduce-ranges come from applying a realistic 60–80% of peak bandwidth to the roofline, the same discount that makes the well-known figure of roughly 40–60 tok/s at batch 1 for an 8B bf16 model on an RTX 4090 come out of that card's 1008 GB/s. Run the harness and report your own; if you land far outside these bands, the interesting question is which assumption broke.

Two things this table says that a single "3× faster" claim cannot:

**At 4k the two architectures are indistinguishable.** Both are 8.1 ms, because at short context the weight read is 97% of the traffic and the cache is a rounding error. If someone benchmarks a hybrid at 2k context and reports a large speedup, the speedup is coming from somewhere other than the architecture — a smaller model, a different kernel, a different engine version.

**The right statistic is degradation, not absolute latency.** 2.00× versus 1.13× is architecture-attributable in a way that "16.3 ms versus 9.1 ms" is not, because the ratio cancels the weight term that both models pay. When you report your run, report the *slope* of TPOT against context. It is far more robust to confounds than any single point.

And the honest caveat on the other side: I expect the hybrid to hit a *lower fraction* of its roofline than the transformer does. A recurrent decode step is a sequence of small, low-intensity kernels updating a state tensor, and it carries launch overhead that a fused attention kernel does not. Dao and Gu's [state space duality work](https://arxiv.org/abs/2405.21060) exists precisely because the naive selective-scan was slow; their SSD formulation is reported as 2–8× faster than Mamba's original selective SSM, which tells you how much implementation quality matters here. Expect the observed ratio to be smaller than the derived ratio, and treat a large gap between them as a kernel problem rather than an architecture result.

---

## 7. Prediction four: goodput at fixed VRAM, and the dollar figure

Sections 4 and 6 combine into the number that actually decides deployments. Concurrency comes from memory; step time comes from bandwidth; goodput is their product.

At the memory ceiling on an 80 GB A100 at 128k context:

**Llama-3.1-8B, batch 3.** Bytes per step: $16.06 + 3 \times 17.18 = 67.60$ GB.

$$t_{\text{step}} = \frac{67.60}{2039} = 33.2 \text{ ms} \;\Rightarrow\; \text{TPOT } 33.2 \text{ ms}, \;\; \frac{3}{0.0332} = 90 \text{ tok/s aggregate}$$

**Nemotron-H-8B, batch 26.** Bytes per step: $16.4 + 26 \times (2.147 + 0.052) = 73.57$ GB.

$$t_{\text{step}} = \frac{73.57}{2039} = 36.1 \text{ ms} \;\Rightarrow\; \text{TPOT } 36.1 \text{ ms}, \;\; \frac{26}{0.0361} = 721 \text{ tok/s aggregate}$$

**Eight times the aggregate throughput at essentially the same per-user latency.** 33.2 ms versus 36.1 ms of TPOT is a difference no user perceives; 90 versus 721 tokens per second is the difference between one GPU serving three long-context conversations and one GPU serving twenty-six.

That result is worth staring at because it is not what the naive reading of section 6 predicts. At batch 1 the hybrid's TPOT advantage was 1.79×. Here it is *negative* — the hybrid is 3 ms slower per token. The entire win came from concurrency, not from speed. The hybrid is not a faster model; it is a model that lets you run more copies of the same speed. That distinction changes how you tune: the lever is `max_num_seqs` and admission policy, not kernel optimization.

And the cost, using the formula from section 2 with an **illustrative** rate of \$1.50 per GPU-hour (a placeholder, not a quoted market price — substitute your own):

| Config | Concurrency | TPOT | Aggregate goodput | GPU-hours per 1M tok | Cost per 1M tok | Source |
| --- | --- | --- | --- | --- | --- | --- |
| Llama-3.1-8B @ 128k | 3 | 33.2 ms | 90 tok/s | 3.09 | \$4.63 | derived at \$1.50/GPU-hr |
| Nemotron-H-8B @ 128k | 26 | 36.1 ms | 721 tok/s | 0.385 | \$0.58 | derived at \$1.50/GPU-hr |
| Llama-3.1-8B @ 4k | 111 | 37.1 ms | 2,992 tok/s | 0.093 | \$0.14 | derived, roofline invalid |
| Nemotron-H-8B @ 4k | 498 | 37.1 ms | 13,423 tok/s | 0.021 | \$0.03 | derived, roofline invalid |

The **8× cost ratio at 128k is the portable result** — it does not depend on the rate you plug in. The absolute dollars do, entirely.

The last two rows carry a deliberate warning label. At batch 111 and above, the decode matmul stops being a skinny GEMV and starts behaving like a real GEMM with meaningful tensor-core utilization; the bandwidth roofline is no longer the binding constraint, and other limits (scheduler bookkeeping, `max_num_seqs` caps, activation memory that I modelled as a flat reserve) bind first. Treat those two rows as upper bounds that your measurement will not reach, and note that this is precisely the regime where the [skinny matrix problem](/blog/machine-learning/inference-engineering/gemm-for-decode-the-skinny-matrix-problem) stops being the dominant concern. I include them because leaving them out would hide the more interesting fact: **at 4k the architectures are 4.5× apart, at 128k they are 8× apart, and at 512 tokens the hybrid is worse.** The advantage is a function of context, and reporting a single number for it is the original sin this post is trying to correct.

---

## 8. The harness: `nanoserve/bench/longcontext.py`

Everything above is a prediction. This section is the instrument that tests it. The design goal is that a stranger with one GPU can run it and produce numbers that are comparable to another stranger's.

### 8.1 The environment record

Before anything else, capture the invariants. A result without this block is not reproducible and should not be published.

```python
# nanoserve/bench/env.py
import json, os, platform, subprocess, torch

def capture_environment(device: int = 0) -> dict:
    props = torch.cuda.get_device_properties(device)
    def smi(field: str) -> str:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader",
             "-i", str(device)],
            capture_output=True, text=True,
        )
        return out.stdout.strip()

    return {
        "gpu": props.name,
        "vram_gib": round(props.total_memory / 2**30, 2),
        "sm_count": props.multi_processor_count,
        "capability": f"{props.major}.{props.minor}",
        "driver": smi("driver_version"),
        "clocks_locked_sm_mhz": smi("clocks.applications.graphics"),
        "current_sm_mhz": smi("clocks.sm"),
        "throttle_reasons": smi("clocks_throttle_reasons.active"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "harness_commit": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True).stdout.strip(),
        "env_flags": {k: v for k, v in os.environ.items()
                      if k.startswith(("VLLM_", "TORCH", "CUDA_", "NCCL_"))},
    }
```

The `throttle_reasons` field is the one people forget. If it reads anything other than the "not active" bitmask during your run, the run is thermally or power limited and the numbers are about your cooling, not your architecture.

### 8.2 The prompt suite, built by token count

```python
# nanoserve/bench/suite.py
import random
from dataclasses import dataclass
from transformers import AutoTokenizer

@dataclass(frozen=True)
class Prompt:
    workload: str
    text: str
    n_input: int
    n_output: int          # forced exact, so decode work is identical
    needle: str | None     # ground-truth answer for recall scoring
    needle_depth: float    # 0.0 = start of context, 1.0 = end

_FILLER = (
    "The quarterly maintenance window for the storage tier is scheduled "
    "after the regional failover drill completes. Operators should confirm "
    "the replica lag is under one second before proceeding. "
)

def build(tokenizer, workload: str, n_input: int, n_output: int,
          seed: int, depth: float = 0.5) -> Prompt:
    """Build a prompt of EXACTLY n_input tokens under THIS tokenizer.

    Building by token count rather than character count is what makes the
    comparison fair: two models with different vocabularies would otherwise
    receive different amounts of context from the same string.
    """
    rng = random.Random(seed)
    needle = None

    if workload == "needle":
        magic = f"{rng.randrange(10**7):07d}"
        needle = magic
        sentence = f"The vault access code for building seven is {magic}. "
        question = ("\n\nWhat is the vault access code for building seven? "
                    "Answer with the digits only.")
    else:
        sentence, question = "", "\n\nSummarize the passage above in one sentence."

    # Grow filler until the tokenized length hits the target, then bisect.
    body = _FILLER * max(1, n_input // 20)
    ids = tokenizer(body, add_special_tokens=False).input_ids
    while len(ids) < n_input * 1.2:
        body += _FILLER * 64
        ids = tokenizer(body, add_special_tokens=False).input_ids

    budget = n_input - len(tokenizer(question, add_special_tokens=False).input_ids)
    if sentence:
        budget -= len(tokenizer(sentence, add_special_tokens=False).input_ids)
        cut = int(budget * depth)
        text = (tokenizer.decode(ids[:cut]) + sentence
                + tokenizer.decode(ids[cut:budget]) + question)
    else:
        text = tokenizer.decode(ids[:budget]) + question

    actual = len(tokenizer(text, add_special_tokens=False).input_ids)
    assert abs(actual - n_input) <= 8, f"length drift: {actual} vs {n_input}"
    return Prompt(workload, text, actual, n_output, needle, depth)
```

The `assert` is deliberate. A silent 10% length mismatch between the two models would produce a 10% "architecture difference" that is entirely tokenizer.

### 8.3 Open-loop arrivals

```python
# nanoserve/bench/load.py
import asyncio, random, time

async def poisson_driver(send, prompts, rate_rps: float,
                         duration_s: float, seed: int = 0):
    """Issue requests at Poisson-distributed intervals, NOT waiting for
    completion. Inter-arrival times are exponential with mean 1/rate.

    Closed-loop drivers (issue next when previous returns) throttle
    themselves against a slow server and can never show queue growth,
    which is the exact failure this benchmark is looking for.
    """
    rng = random.Random(seed)
    tasks, t_end = [], time.perf_counter() + duration_s
    i = 0
    while time.perf_counter() < t_end:
        p = prompts[i % len(prompts)]
        tasks.append(asyncio.create_task(send(p, submitted_at=time.perf_counter())))
        i += 1
        await asyncio.sleep(rng.expovariate(rate_rps))
    return await asyncio.gather(*tasks, return_exceptions=True)
```

Two details that are easy to get wrong. `rng.expovariate(rate)` gives the exponential inter-arrival times that make the process Poisson — a fixed `1/rate` sleep gives a *deterministic* arrival process, which is far gentler on the queue and will overstate your capacity. And `submitted_at` is stamped at issue time on the client, not at engine admission, so queueing time is inside TTFT where it belongs.

### 8.4 Per-request instrumentation

```python
# nanoserve/bench/record.py
import time
from dataclasses import dataclass, field

@dataclass
class Record:
    workload: str
    n_input: int
    n_output_requested: int
    submitted_at: float
    first_token_at: float | None = None
    token_times: list[float] = field(default_factory=list)
    text: str = ""
    error: str | None = None

    @property
    def ttft_ms(self) -> float | None:
        if self.first_token_at is None:
            return None
        return (self.first_token_at - self.submitted_at) * 1e3

    @property
    def tpot_ms(self) -> float | None:
        """Mean inter-token latency AFTER the first token.

        Dividing total time by total tokens folds prefill into TPOT and
        makes long-context requests look like slow decoders. Excluding
        the first gap is the whole point.
        """
        if len(self.token_times) < 2:
            return None
        gaps = [b - a for a, b in zip(self.token_times, self.token_times[1:])]
        return sum(gaps) / len(gaps) * 1e3

    @property
    def output_tok_s(self) -> float | None:
        t = self.tpot_ms
        return None if t is None else 1e3 / t

async def stream_request(client, model: str, p, submitted_at: float) -> Record:
    rec = Record(p.workload, p.n_input, p.n_output, submitted_at)
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": p.text}],
            max_tokens=p.n_output,
            temperature=0.0,
            seed=1234,
            extra_body={"min_tokens": p.n_output, "ignore_eos": True},
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if not delta:
                continue
            now = time.perf_counter()
            if rec.first_token_at is None:
                rec.first_token_at = now
            rec.token_times.append(now)
            rec.text += delta
    except Exception as exc:                    # OOM, timeout, 400 on context
        rec.error = f"{type(exc).__name__}: {exc}"
    return rec
```

`min_tokens` with `ignore_eos` is what makes decode work identical across models. Without it, one model writes 180 tokens and the other 340, and your "throughput" measurement is partly a measurement of verbosity.

### 8.5 The capacity probe

Rather than guessing the batch that fits, find it. Binary search on concurrency until you OOM, which also gives you the OOM boundary as a reportable number.

```python
# nanoserve/bench/capacity.py
import torch

def max_concurrency(run_batch, lo: int = 1, hi: int = 512) -> tuple[int, dict]:
    """Largest concurrency that completes one decode step without OOM.

    run_batch(n) must allocate the cache for n requests at the target
    context and step once. Reset the allocator between probes or the
    fragmentation from a failed attempt poisons the next one.
    """
    best, peaks = lo, {}
    while lo <= hi:
        mid = (lo + hi) // 2
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            run_batch(mid)
            peaks[mid] = {
                "allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            }
            best, lo = mid, mid + 1
        except torch.cuda.OutOfMemoryError:
            hi = mid - 1
    return best, peaks
```

Report both `allocated` and `reserved`. The gap between them is allocator fragmentation, and on a long-context workload with variable prompt lengths it can be several gigabytes — which is exactly the problem paged attention was invented to solve, covered in [paged KV cache: implementing blocks and a block table](/blog/machine-learning/inference-engineering/paged-kv-cache-implementing-blocks-and-a-block-table).

### 8.6 The sweep driver

```python
# nanoserve/bench/longcontext.py
import asyncio, json, statistics as st
from nanoserve.bench import env, suite, load, record

CONTEXTS = [4_096, 16_384, 32_768, 65_536, 131_072]
WORKLOADS = [("chat", 4_096, 512), ("rag", None, 256),
             ("code", 32_768, 256), ("needle", None, 32)]
SLO = {"ttft_ms": 5_000.0, "tpot_ms": 50.0}

def summarize(recs, window_s: float, seed_note: str) -> dict:
    ok = [r for r in recs if r.error is None and r.ttft_ms is not None]
    oom = sum(1 for r in recs if r.error and "OutOfMemory" in r.error)
    if not ok:
        return {"status": "all_failed", "oom": oom, "n": len(recs)}
    ttfts = sorted(r.ttft_ms for r in ok)
    tpots = sorted(r.tpot_ms for r in ok if r.tpot_ms is not None)
    met = [r for r in ok
           if r.ttft_ms <= SLO["ttft_ms"] and (r.tpot_ms or 0) <= SLO["tpot_ms"]]
    out_tokens = sum(len(r.token_times) for r in met)
    return {
        "n": len(recs), "ok": len(ok), "oom": oom,
        "ttft_p50_ms": round(st.median(ttfts), 1),
        "ttft_p99_ms": round(ttfts[int(0.99 * (len(ttfts) - 1))], 1),
        "tpot_p50_ms": round(st.median(tpots), 2),
        "tpot_p99_ms": round(tpots[int(0.99 * (len(tpots) - 1))], 2),
        "slo_met_frac": round(len(met) / len(ok), 3),
        "goodput_tok_s": round(out_tokens / window_s, 1),
        "seeds": seed_note,
    }

async def main(client, model, tokenizer, rates=(0.25, 0.5, 1.0, 2.0)):
    results = {"env": env.capture_environment(), "model": model, "cells": []}
    for ctx in CONTEXTS:
        for name, fixed_in, n_out in WORKLOADS:
            n_in = fixed_in or ctx
            prompts = [suite.build(tokenizer, name, n_in, n_out, seed=s)
                       for s in range(5)]                  # 5 seeds per point
            for rate in rates:
                # 30 warmup steps at THIS shape, discarded entirely.
                await load.poisson_driver(
                    lambda p, submitted_at: record.stream_request(
                        client, model, p, submitted_at),
                    prompts, rate_rps=rate, duration_s=30, seed=999)
                recs = await load.poisson_driver(
                    lambda p, submitted_at: record.stream_request(
                        client, model, p, submitted_at),
                    prompts, rate_rps=rate, duration_s=180, seed=7)
                keep = [r for r in recs
                        if 60 <= r.submitted_at - recs[0].submitted_at <= 170]
                cell = summarize(keep, window_s=110.0, seed_note="0-4")
                cell |= {"context": ctx, "workload": name, "rate_rps": rate}
                results["cells"].append(cell)
                print(json.dumps(cell))
    return results
```

Run it for each model and diff the JSON. Note the structure of the honest reporting: `oom` is a first-class output, not an exception that aborts the sweep. A cell where the transformer OOMs and the hybrid does not **is the result**, and a harness that crashes there throws away the most important data point.

### 8.7 Scoring recall

Speed numbers without this section are how regressions ship.

```python
# nanoserve/bench/recall.py
import re

def score_needle(recs) -> dict:
    """Exact-match recall on the planted needle, bucketed by depth.

    Bucketing by depth matters: 'Lost in the Middle' style degradation
    shows up as a dip at depth 0.4-0.6 that an aggregate score hides.
    """
    buckets: dict[str, list[int]] = {}
    for r in recs:
        if r.error or r.needle is None:
            continue
        hit = int(bool(re.search(rf"\b{re.escape(r.needle)}\b", r.text)))
        buckets.setdefault(f"depth_{r.needle_depth:.1f}", []).append(hit)
    per_depth = {k: sum(v) / len(v) for k, v in buckets.items() if v}
    flat = [h for v in buckets.values() for h in v]
    return {"overall": sum(flat) / len(flat) if flat else None,
            "per_depth": per_depth, "n": len(flat)}
```

Run this at every context in the sweep, at depths 0.1 through 0.9, for both models. Then run the multi-needle and variable-tracking tasks from [RULER](https://arxiv.org/abs/2404.06654), because single-needle retrieval is the test everything passes — that is the RULER paper's central complaint about it.

**A results table with an empty recall column is not a result.** If you take one thing from this post, take that.

---

## 9. The shape your sweep should trace, and how to read a surprise

![Six ordered sweep points from four thousand to one million tokens showing where the two architectures tie, where they diverge, and where one stops running](/imgs/blogs/experiment-hybrid-vs-transformer-at-long-context-5.webp)

The timeline above is the prediction the harness is testing, laid out along the axis you sweep. The way to use it is as a checklist: at each point, does your run reproduce the expected relationship? Deviations are the interesting part.

| Sweep point | Expected relationship | Source |
| --- | --- | --- |
| 4k | TPOT within a few percent; memory 4.5× apart | derived |
| 16k | TPOT gap opens to about 1.2×; memory 6.7× apart | derived |
| 32k | TPOT about 1.3×; TTFT about 1.4×; memory 7.3× apart | derived |
| 128k | TPOT about 1.8× at batch 1; goodput about 8× at capacity | derived |
| 256k | Transformer needs 32 GiB per request; OOM on 80 GB at batch 2 | derived |
| 1M | Transformer needs 128 GiB per request; hybrid 16.05 GiB | derived |

Now the diagnostic table — what each surprise means:

**Your hybrid shows a big win at 4k.** The physics says it should not. Check that both models are on the same engine version, the same attention backend, and the same `max_num_seqs`; check the tokenizers are producing the same input length; check whether prefix caching is on for one and off for the other. The vLLM Qwen3-Next post notes that prefix caching for hybrid models was a **roadmap gap**, so a hybrid-versus-transformer comparison with prefix caching enabled may be comparing "cache hit" against "cache miss" and calling it architecture.

**Your transformer beats the derived roofline.** Almost certainly prefix caching or chunked prefill silently reusing work between your seeded prompts. Vary the prompt prefix per seed, or disable prefix caching on both sides for the architecture comparison and re-enable it for the "what should I deploy" comparison. Both runs are valid; conflating them is not.

**Your hybrid's TPOT is worse than derived at every context.** Kernel quality, not architecture. The recurrent path is launch-overhead-sensitive and its Triton kernels may not be tuned for your GPU. Check whether CUDA graphs are capturing the recurrent layers at all — vLLM's [Model Runner V2 post](https://vllm.ai/blog/2026-03-24-mrv2) notes that as of v0.18.0 the new runner did **not** support linear-attention models, which is the kind of version-dependent gap that produces a large, entirely non-architectural difference.

**p99 diverges from p50 far more for one model.** Look at preemption. The hybrid's fixed state cannot be dropped and recomputed the way a KV block can — there is no per-token index to recompute *from* — so an engine under memory pressure has different options for the two architectures. This is one of the engine-level consequences developed in [hybrid models and the end of the KV-cache assumption](/blog/machine-learning/inference-engineering/hybrid-models-and-the-end-of-the-kv-cache-assumption).

**Both models degrade identically.** Then your bottleneck is not the cache. Check whether you are actually reaching the memory ceiling: if `max_num_seqs` is capping concurrency below what memory allows, you are benchmarking a config default.

---

## 10. Where hybrids give ground: the part the headline number omits

Everything so far has been about cost. This section is about capability, and it is the reason this post refuses to declare a winner.

A fixed-size recurrent state is a lossy compressor with a hard information budget. A Nemotron-H Mamba-2 layer holds about a million numbers, and that budget is the same whether the context is 500 tokens or 500,000. At 128k tokens the layer is retaining on the order of eight numbers per input token. It cannot keep everything, so the gating decides what to keep — and it makes that decision **at write time, before the model has seen the query.** Attention makes the equivalent decision at *read* time, when the query is known, which is precisely why it can retrieve anything from anywhere.

That is not a hand-wave; it is the finding of a specific body of work:

- [Jelassi et al., *Repeat After Me* (arXiv 2402.01032)](https://arxiv.org/abs/2402.01032) isolates **copying** as the task where transformers beat state-space models, with the argument that a fixed-size state cannot store an arbitrarily long string for later verbatim reproduction.
- [Arora et al., *Zoology* (arXiv 2312.04927)](https://arxiv.org/abs/2312.04927) isolates **multi-query associative recall** — retrieving several distinct key-value bindings from context — as the specific capability on which efficient gated-convolution architectures fall behind attention, and attributes a substantial share of the residual perplexity gap to it.
- [RULER (arXiv 2404.06654)](https://arxiv.org/abs/2404.06654) gives you the instrument: multi-needle retrieval, variable tracking, aggregation, all parameterized by length. Its finding that only about half of 17 evaluated long-context models hold up at 32K applies to transformers too — this is not a hybrid-only problem, which is exactly why you must measure *both* arms.

Now name the production workloads those results describe. Retrieve the function signature from the file pasted at token 3,000. Quote the indemnity clause from the contract. Reuse the tool-call ID that appeared eleven turns ago. Copy this UUID verbatim. Every one is exact retrieval of a specific token from an arbitrary earlier position, and if you serve RAG, agentic tool loops, or code assistance, these are not edge cases — they are the workload.

![A five-row comparison of workloads against the hybrid verdict, the transformer verdict, and the quantity that decides each one](/imgs/blogs/experiment-hybrid-vs-transformer-at-long-context-6.webp)

The matrix above is the honest summary, and the thing to notice is that the winning column changes three times going down five rows. Row by row:

**Chat, 4k in / 512 out — a tie.** Both land at about 8.1 ms of derived TPOT because the weight read is 97% of the traffic. Choose on quality, not on serving cost.

**RAG, 128k in / 256 out — the hybrid, decisively.** 721 versus 90 tokens per second of derived aggregate goodput at the memory ceiling, an 8× gap, at indistinguishable per-user latency. This is the workload hybrids were built for. But see the needle row before you commit: RAG is *also* exact retrieval, and if your retrieved chunks require verbatim quotation you are in the row below.

**Code, 32k repo map — the hybrid, modestly.** 2.71 versus 6.61 PFLOP of derived prefill at 128k, about 1.4× at 32k, which shows up as TTFT. Real code assistance leans heavily on prefix caching, so measure with prefix caching in the configuration you will actually deploy, and remember it may not be available on the hybrid path.

**Needle recall at 128k — the transformer, on principle and probably in measurement.** This is the row where you must produce your own number rather than trusting mine, because the answer depends on the specific checkpoint's training, not only on its architecture. Do not accept a hybrid for a retrieval-critical workload without a RULER score at your context length.

**Classification, 200-token prompts — the transformer, and it is not close.** The hybrid's 49.4 MiB state intercept is pure overhead below the break-even length. The previous post derives that break-even as $S^{*} = \sigma / (2 H_{kv} d b) = 527$ tokens for Nemotron-H, and remarkably it does not depend on the interleave ratio at all. At a 200-token prompt the hybrid holds 53.4 MiB per request against the transformer's 32.0 MiB — 67% *more* memory. If your endpoint is short-prompt classification, a hybrid is a straightforward regression and you should say so out loud rather than adopting one because the architecture is fashionable.

This is also the whole answer to "why interleave at all?" If pure recurrence were free, nobody would keep any attention layers. Every serious hybrid keeps some — Nemotron-H keeps four of 28 mixers, Qwen3-Next interleaves Gated DeltaNet with full attention (the vLLM post is explicit that it interleaves; it does **not** state a ratio, so do not quote one from that source), MiniMax-M1 interleaves lightning attention with softmax attention. The interleave ratio is the dial between memory cost and exact recall, and every architect who has published one has chosen a value strictly between zero and one.

---

## 11. Case studies: what the public record actually says, with its setup attached

Four published results worth knowing, each quoted with the setup that makes it interpretable — which is the discipline this whole post is arguing for.

**MiniMax-M1's lightning attention.** The vLLM team's [MiniMax-M1 post (2025-06-30)](https://vllm.ai/blog/2025-06-30-minimax-m1) states that lightning attention "reduces memory 83% and inference latency 67% for 100k-token sequences." Setup as published: MiniMax-M1, a hybrid interleaving lightning (linear) attention with softmax attention, MoE with 456B total and about 45.9B active parameters, served with `--tensor-parallel-size 8 --quantization experts_int8`, variants M1-40k and M1-80k. **What the post does not say**, and you should not assume: the baseline the 83% is measured against, the GPU, the batch size, or whether "memory" means the whole request footprint or only the sequence-mixing layers. The post also explicitly does not detail how the linear state and the full-attention KV coexist, flagging a future "hybrid allocator." Compare it to my derived ratio: a 1:6 interleave gives an 87.5% cache reduction from the slope alone, which brackets the 83% figure nicely — but that agreement is suggestive, not confirmatory, since the setups are not the same.

**Nemotron-H's inference speedup.** The [technical report (arXiv 2504.03624)](https://arxiv.org/abs/2504.03624) claims "up to 3× faster at inference" against Qwen-2.5-7B/72B and Llama-3.1-8B/70B. The abstract does not state the context length, batch size, hardware, or serving stack. Set against the derivation in section 6: at batch 1 and 128k, the derived roofline ratio is 1.79×; at the memory ceiling the *goodput* ratio is 8×. A "3×" sits between those, which is consistent with a mid-range context at a moderate batch — but "consistent with" is not "reproduced," and the honest reading is that you need the paper's full experimental section, not its abstract, before you use that number in a capacity plan.

**Disaggregated serving for hybrids.** vLLM's [hybrid SSM disaggregation post (2026-04-21)](https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg) reports that with `Nemotron-3-Super-120B-A12B-FP8` on 8×H200 running disaggregated prefill and decode (prefill TP4 plus decode TP4), the disaggregated configuration "Pareto-dominates the co-located baseline at higher batch sizes." Two things to take from it. First, the *framing*: a Pareto claim over a throughput-versus-latency curve is the honest shape for a serving result, because there is no single operating point. Second, the engineering: the post describes decomposing the conv state into three sub-projections plus the SSM state in a `(dim, state_len)` layout so each decode rank reads only its own slice over RDMA, saving roughly 50 MB per request in bf16 by never transferring padding. That is a concrete reminder that a hybrid's serving cost is as much about the engine as the architecture. The post also lists real limits worth knowing before you benchmark: Mamba-1 unsupported, gated DeltaNet pending, and speculative-decoding interaction "not extensively validated."

**The transformer side is not standing still.** Two vLLM posts matter for a fair comparison. The [FP8 KV-cache post (2026-04-22)](https://vllm.ai/blog/2026-04-22-fp8-kvcache) reports that for Llama-3.1-8B on an H100, FP8 KV brings the inter-token-latency slope down to 54% of bf16 with a break-even around 7k tokens, and under load gives +14.9% throughput and −14.8% median ITL, halving KV storage. Halving the transformer's slope halves the memory gap in section 4 — so **an FP8-KV transformer versus a bf16 hybrid is not the comparison you think you are running.** That same post also carries the sharpest cautionary tale in this whole area: on Hopper, imprecise FP32 accumulation at 100k-plus contraction lengths collapsed needle-retrieval accuracy from 91% to 13%, fixed by two-level accumulation at a TTFT cost. A numerical detail in a kernel destroyed long-context recall, and only a recall metric would have caught it. Separately, the [DeepSeek-V4 post (2026-04-24)](https://vllm.ai/blog/2026-04-24-deepseek-v4) reports bf16 KV of only 9.62 GiB per sequence at 1M tokens through compression and sparse attention — roughly 8.7× smaller than a V3.2-style estimate of 83.9 GiB — which is squarely in hybrid territory using attention-family techniques.

The synthesis: **"hybrid versus transformer" is not a fixed comparison.** It is a comparison between two moving targets, and any result you publish is valid for the precise pair of configurations you tested. Which is why the environment record in section 8.1 is not bureaucracy.

---

## 12. When to reach for a hybrid, and when to keep the transformer

![A decision tree splitting a long-context service on exact-retrieval need, throughput pressure, and prompt length, with the recommendation each branch reaches](/imgs/blogs/experiment-hybrid-vs-transformer-at-long-context-7.webp)

The tree above is the decision, and its most important property is that only one of the three branches ends in an unqualified recommendation. The other two end in a measurement, which is the correct output when the answer genuinely depends on your data.

**Reach for a hybrid when:** your context is routinely above about 16k and your bottleneck is concurrency at a fixed VRAM budget; your workload is summarization, long-document reasoning, extended chat, or long-form generation where the model needs the *gist* of the context rather than verbatim retrieval from it; you are memory-limited on consumer hardware and the transformer simply does not fit; or you serve enough volume that an 8× cost ratio at 128k pays for the migration.

**Keep the transformer when:** exact retrieval is on the critical path and you have not scored recall at your context length; your prompts are short — below the 527-token break-even, the hybrid is strictly worse on memory; your engine depends on features the hybrid path does not have yet (the vLLM posts name prefix caching for Qwen3-Next as a roadmap gap, Model Runner V2 as not supporting linear attention as of v0.18.0, and speculative decoding on hybrids as not extensively validated); or you have not yet applied the transformer-side wins — FP8 KV alone halves the slope, and prefix caching may delete your prefill entirely.

**And the option most teams should take first:** neither. Before you change architecture, exhaust the cheaper levers on the model you already run. Prefix caching, chunked prefill, FP8 KV, a bigger `max_num_seqs`, and admission control that stops overcommitting are all configuration changes with no quality risk and no migration. An architecture swap is a model swap: new quality profile, new eval suite, new failure modes, new operational unknowns. Spend that only when the arithmetic in sections 4 through 7 says the ceiling you are hitting is structural.

**When to use vLLM rather than your own harness:** for the *engine*, always — nothing in `nanoserve` should serve production traffic. But for the *measurement*, own the harness. A benchmark you wrote, whose assumptions you can enumerate, that emits an environment record and a recall column, is worth more than any number someone else publishes, precisely because you know what it does and does not control.

---

## Key takeaways

1. **A number without its setup is a rumour.** "Up to 3× faster" and "83% less memory" are both true and both unusable; the setup is the number.
2. **Derive the curves before you run them.** A benchmark with a prediction is a test; a benchmark without one is an exploration that will accept any result.
3. **The protocol is most of the work.** Locked clocks, 30 discarded warmup steps, CUDA events, open-loop Poisson arrivals, 60 seconds of transient dropped, fixed seeds, forced output length, and a prompt suite built by token count per tokenizer.
4. **Closed-loop load generators cannot find your capacity limit.** They throttle themselves against a slow server, so the queue never grows and the tail never appears.
5. **The memory advantage saturates early.** The derived ratio between Llama-3.1-8B and Nemotron-H-8B is 7.30× at 32k and 7.81× at 128k against a ceiling of 8. Quadrupling the context buys 7% more relative advantage.
6. **At short context the architectures are identical, and below 527 tokens the hybrid is worse.** Both sit at about 8.1 ms of derived batch-1 TPOT at 4k because the weight read is 97% of the traffic.
7. **The win is concurrency, not speed.** At the 80 GB memory ceiling at 128k, the derived per-user TPOT is 33.2 ms for the transformer and 36.1 ms for the hybrid — the hybrid is *slower* per token — while aggregate goodput is 90 versus 721 tokens per second.
8. **Report degradation, not absolutes.** TPOT slope against context (2.00× versus 1.13× from 4k to 128k) cancels the weight term both models pay and is far more robust to confounds than any single latency point.
9. **A results table with an empty recall column is not a result.** Fixed state decides what to keep at write time; attention decides at read time. Score RULER at every context, on both arms.
10. **Both sides are moving.** FP8 KV halves the transformer's slope; the hybrid path is still missing prefix caching, Model Runner V2 support, and validated speculative decoding. Your comparison is valid for the exact pair of configurations you tested and no others.

---

## Further reading

- [Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models (arXiv 2504.03624)](https://arxiv.org/abs/2504.03624) — the architecture and the "up to 3× faster" claim.
- [Transformers are SSMs: state space duality and Mamba-2 (arXiv 2405.21060)](https://arxiv.org/abs/2405.21060) — Dao and Gu, the recurrent layer underneath most current hybrids.
- [RULER: What's the Real Context Size of Your Long-Context Language Models? (arXiv 2404.06654)](https://arxiv.org/abs/2404.06654) — the recall instrument this experiment requires.
- [Repeat After Me (arXiv 2402.01032)](https://arxiv.org/abs/2402.01032) and [Zoology (arXiv 2312.04927)](https://arxiv.org/abs/2312.04927) — where fixed-state models lose to attention, isolated as tasks.
- [vLLM: Disaggregated Serving for Hybrid SSM Models (2026-04-21)](https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg) — the dual-descriptor engine work and the Pareto framing.
- [vLLM: The State of FP8 KV-Cache and Attention Quantization (2026-04-22)](https://vllm.ai/blog/2026-04-22-fp8-kvcache) — the transformer-side counter-move, and the needle-collapse cautionary tale.
- [vLLM: MiniMax-M1 (2025-06-30)](https://vllm.ai/blog/2025-06-30-minimax-m1) and [Qwen3-Next (2025-09-11)](https://vllm.ai/blog/2025-09-11-qwen3-next) — two production hybrids and the hybrid KV-cache manager.
- Within this series: [what inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is), [the memory math of the KV cache](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache), [long-context inference: RoPE scaling, sinks, and the prefill cost curve](/blog/machine-learning/inference-engineering/long-context-inference-rope-scaling-sinks-and-the-prefill-cost-curve), [hybrid models and the end of the KV-cache assumption](/blog/machine-learning/inference-engineering/hybrid-models-and-the-end-of-the-kv-cache-assumption), and the capstone [the inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook).
- Out of series: [setting up a reproducible benchmark](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) for the general benchmark-hygiene treatment.

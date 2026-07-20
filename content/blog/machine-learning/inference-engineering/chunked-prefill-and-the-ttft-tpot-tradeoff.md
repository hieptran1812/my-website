---
title: "Chunked prefill and the TTFT/TPOT trade-off: one knob, two victims"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Derive exactly how much one long prefill costs forty streaming users, then build the token budget that bounds it, plot the frontier it buys you, and learn when to split your GPUs in two instead."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "chunked-prefill",
    "scheduling",
    "batching",
    "latency",
    "throughput",
    "kv-cache",
    "pytorch",
    "gpu",
    "ml-systems",
    "vllm",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 52
---

Here is a bug report you cannot reproduce.

Forty people are chatting with your service. Tokens are arriving smoothly, roughly one every thirteen milliseconds, which reads on screen as a comfortable typewriter. Then, for a little over five seconds, every one of those forty streams stops dead. No error. No timeout. No log line above `INFO`. Five seconds later all forty resume at exactly the cadence they had before, and your dashboard — which averages latency over a minute — shows a barely perceptible bump. Nobody can reproduce it, because reproducing it requires that a forty-first person, somewhere else entirely, pasted a 32,000-token contract into the same endpoint at the same moment.

![Seven engine iterations laid left to right where a single long prefill occupies one iteration and forty decoding streams emit nothing for its whole duration](/imgs/blogs/chunked-prefill-and-the-ttft-tpot-tradeoff-1.webp)

That is the picture above, and it is the entire subject of this post. The forty-first request needed its prompt processed — its *prefill* — and prefill is a single enormous compute job that your engine ran as one indivisible iteration. While it ran, the decode loop could not run, because there is one GPU and it does one thing at a time. Everyone paid for one person's document.

By the end of this post you will be able to derive that five-second number from a model config and a datasheet, without a GPU; implement the fix — a per-iteration **token budget** with a prefill cursor — on top of the continuous batching loop from [the previous post](/blog/machine-learning/inference-engineering/writing-a-continuous-batching-loop); plot the frontier that budget buys you and read a defensible operating point off it; state a closed-form rule that turns an inter-token-latency SLO into a budget number; and know precisely when chunked prefill is the wrong answer and you should be splitting your fleet into prefill and decode pools instead. This post writes `nanoserve/budget.py` and rewrites `nanoserve/scheduler.py`'s `step()`.

Two standing promises, restated from [the series introduction](/blog/machine-learning/inference-engineering/what-inference-engineering-is). First, **I have no GPU and I have run none of this.** Every number below is derived from arithmetic I show you, cited from a public source with a link, or framed as something you should reproduce yourself. Results tables carry a `Source` column. Second, the centerpiece of this post — the frontier table — is *pure arithmetic*, which means you can run the script and get byte-identical numbers to mine, on a laptop, with no accelerator at all. That is the strongest form of reproducibility available to me and I lean on it hard.

---

## 1. Two workloads that hate each other

Every request to an LLM server has two phases, and they are so different that treating them as one workload is the root cause of most latency pathology in serving.

**Prefill** processes the prompt. All $P$ prompt tokens go through the model at once, in parallel, as one big batched forward pass. It produces the KV cache entries for all $P$ positions and exactly one output token — the first one. The metric it owns is **TTFT** (time to first token): the wall-clock gap between the user pressing enter and the first character appearing.

**Decode** produces every subsequent token. It runs the model on exactly one new token per sequence, attends over the cached keys and values of everything before it, and appends one new KV entry. The metrics it owns are **ITL** (inter-token latency — the gap between two consecutive tokens of one request) and **TPOT** (time per output token — the average ITL over a request's output). The vLLM team's *Anatomy of a High-Throughput Inference System* post ([vllm.ai, 2025-09-05](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm)) defines these three exactly this way, and I use their definitions throughout.

![A request queue fanning into a compute-bound prefill path and a memory-bound decode path which merge back into one serialized engine iteration feeding the two latency clocks](/imgs/blogs/chunked-prefill-and-the-ttft-tpot-tradeoff-2.webp)

The figure above is the shape of the whole problem: two paths with opposite characters, forced through one serialized step. Let me put numbers on "opposite characters", because the gap is not a matter of degree.

Take the spine model of this series: **Llama-3.1-8B in bf16 on an A100 80GB SXM**. From its config: $L = 32$ layers, $H = 32$ query heads, $H_{kv} = 8$ key/value heads, $d = 128$ per-head dimension, $N = 8.03 \times 10^9$ parameters. From [NVIDIA's A100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf): 312 TFLOP/s dense BF16 tensor throughput and 2,039 GB/s of HBM2e bandwidth on the 80GB SXM part.

Three constants fall out of that config and I will use them for the rest of the post:

$$\underbrace{2N = 1.606 \times 10^{10}}_{\text{FLOPs per token, linear layers}} \qquad \underbrace{A = 4Lhd = 524{,}288}_{\text{FLOPs per query-key pair}} \qquad \underbrace{B_{\text{tok}} = 2 L H_{kv} d b = 131{,}072}_{\text{KV bytes per token}}$$

The first is the standard "two FLOPs per parameter per token" for the matrix multiplications. The second is the attention cost: for one query token attending one key position, each layer and each head does ${2d}$ FLOPs for the $QK^\top$ dot product and ${2d}$ for the $AV$ weighted sum, so ${4d}$ per head, times $H$ heads times $L$ layers gives $4Lhd = 4 \cdot 32 \cdot 32 \cdot 128 = 524{,}288$. The third is [the memory math from Track B](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache): 128 KiB per token of context, all layers included.

Now the two workloads.

**A decode step with 40 running sequences at an average context of 2,000 tokens.**

$$\text{FLOPs} = 2N \cdot 40 + A \cdot (40 \cdot 2000) = 6.42\times10^{11} + 4.19\times10^{10} = 6.84 \times 10^{11}$$

$$\text{bytes} = \underbrace{16.06\times10^{9}}_{\text{read all weights}} + \underbrace{131072 \cdot 80{,}000}_{\text{read all KV}} = 2.655 \times 10^{10}$$

Arithmetic intensity $\text{AI} = 6.84\times10^{11} / 2.655\times10^{10} = 25.8$ FLOPs per byte. The A100's ridge point — the intensity at which a kernel stops being memory-bound and starts being compute-bound — is $312\times10^{12} / 2.039\times10^{12} = 153$ FLOPs per byte. Decode at 25.8 is a factor of six *below* the ridge: it is memory-bound, and its time is set by bandwidth. That step takes $2.655\times10^{10} / 2.039\times10^{12} = 13.0$ ms. Forty streams, one token each, thirteen milliseconds. If you have read [the roofline post](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound), none of this is new; if you have not, the one-line version is that decode spends its life dragging 16 GB of weights out of HBM to do a rounding error's worth of math.

**A prefill of 32,768 tokens.**

$$\text{FLOPs} = 2NP + \tfrac{A}{2} P^2 = 1.606\times10^{10}\cdot 32768 + 262144 \cdot 32768^2$$

$$= 5.263\times10^{14} + 2.815\times10^{14} = 8.078\times10^{14}$$

The $\tfrac{A}{2}P^2$ term is the causal attention triangle: query $i$ attends keys $0 \ldots i$, so summing over all $i$ gives $P^2/2$ query-key pairs, not $P^2$. Bytes moved are the weights once, plus writing $32768 \cdot 131072 = 4.30$ GB of KV, plus reading it back inside the attention kernel: call it $2.5\times10^{10}$ bytes. Arithmetic intensity $\approx 32{,}800$ FLOPs per byte — over two hundred times past the ridge point. Prefill is compute-bound to a degree that few workloads on a GPU ever are.

At a realistic **50% model FLOPs utilization** — I will call this rate $R = 0.5 \cdot 312 = 156$ TFLOP/s and flag it as an assumption every time it matters — that prefill takes

$$T_{\text{prefill}} = \frac{8.078\times10^{14}}{1.56\times10^{14}} = 5.18 \text{ s}$$

Five point one eight seconds of GPU, uninterruptible, for one request. Meanwhile a decode step costs thirteen milliseconds. **The prefill is 398 decode steps long.** That is the bug report at the top of this post, derived from a config file and a datasheet.

#### Worked example: where the prefill FLOPs actually go

It is worth knowing which of those two terms dominates, because the answer changes how you reason about long context. Set them equal:

$$2NP = \tfrac{A}{2}P^2 \implies P^\star = \frac{2N}{A/2} = \frac{1.606\times10^{10}}{262144} = 61{,}264 \text{ tokens}$$

Below about 61k tokens of prompt, Llama-3.1-8B's prefill is dominated by the linear layers and scales *linearly* with prompt length. Above it, attention dominates and the cost goes quadratic. At our 32,768-token prompt attention is already 35% of the bill ($2.815\times10^{14}$ of $8.078\times10^{14}$), which is why "just double the context window" is never a factor-of-two conversation. At 131,072 tokens the same arithmetic gives $2.10\times10^{15} + 4.50\times10^{15} = 6.60\times10^{15}$ FLOPs — attention is now 68% of it, and the whole prefill is 42.3 s at $R = 156$ TFLOP/s. Source: derived.

| Prompt length | Linear FLOPs | Attention FLOPs | Total | Time at 156 TFLOP/s | Source |
| --- | --- | --- | --- | --- | --- |
| 1,024 | 1.64e13 | 2.75e11 | 1.67e13 | 0.107 s | derived |
| 8,192 | 1.32e14 | 1.76e13 | 1.49e14 | 0.956 s | derived |
| 32,768 | 5.26e14 | 2.81e14 | 8.08e14 | 5.18 s | derived |
| 131,072 | 2.10e15 | 4.50e15 | 6.60e15 | 42.3 s | derived |

Read the last row again. On one A100, one 128k-context request costs forty-two seconds of exclusive GPU time if you run it as a single step. There is no scheduler policy that survives that, and no amount of "we'll add more replicas" that helps, because the stall is *per-request*, not per-QPS.

---

## 2. Deriving the stall: what one prefill does to everyone else's ITL

Now make the damage precise, because "it feels janky" is not an engineering statement and you will need the formula to size the fix.

Model an engine iteration as: pick a set of work, run one forward pass, emit the tokens. Under the naive continuous batching loop, an iteration is either *all decode* (every running sequence advances one token) or *the prefill of one newly admitted request*. Let $n_d$ be the number of running decoders, $T_d$ the decode-step time, and $T_p(P)$ the prefill time for a $P$-token prompt.

A decoder's ITL is normally $T_d$. If a prefill lands between two of its tokens, that one gap becomes

$$\text{ITL}_{\text{spike}} = T_d + T_p(P) = T_d + \frac{2NP + \tfrac{A}{2}P^2}{R}$$

and the **spike ratio** — how much worse that one gap is than the normal cadence — is

$$\rho(P) = \frac{T_d + T_p(P)}{T_d} = 1 + \frac{2NP + \tfrac{A}{2}P^2}{R \cdot T_d}$$

With $T_d = 13.0$ ms and $R = 156$ TFLOP/s, the denominator $R \cdot T_d = 2.028\times10^{12}$ FLOPs. So:

| Interloping prompt | $T_p$ | ITL spike | $\rho$ | Source |
| --- | --- | --- | --- | --- |
| 512 tokens | 53 ms | 66 ms | 5.1x | derived |
| 2,048 tokens | 0.22 s | 0.23 s | 17.8x | derived |
| 8,192 tokens | 0.96 s | 0.97 s | 74.5x | derived |
| 32,768 tokens | 5.18 s | 5.19 s | 399x | derived |
| 131,072 tokens | 42.3 s | 42.3 s | 3255x | derived |

Three things to take from this table.

**The spike is superlinear in prompt length** because of the $P^2$ term, so your worst case is set by your *longest allowed* prompt, not your median. A service whose p50 prompt is 400 tokens and whose `max_model_len` is 128k has a worst-case ITL of forty-two seconds. The median tells you nothing.

**One prefill damages every concurrent decoder, not one.** If 40 streams are running, one 32k prefill injects one 5.19 s gap into 40 different users' token streams. The blast radius is the batch size. This is the single most counterintuitive property of shared-GPU serving and the reason p99 ITL is the metric that actually correlates with people complaining.

**The damage is invisible to averages.** A user generating 500 tokens whose stream ate one 5.19 s stall has a TPOT of $(499 \cdot 0.013 + 5.19)/500 = 23$ ms — barely double the clean 13 ms, and comfortably inside most TPOT SLOs. The TPOT metric *hides* the exact failure the user noticed. If you take one operational lesson from this post: **alert on p99 ITL, not on TPOT.** TPOT is an average and averages launder stalls.

#### Worked example: how much prefill traffic can you absorb?

Suppose your workload is 20 chat requests per second at 400 prompt tokens each, plus 0.2 requests per second of document summarization at 32,768 tokens. Prefill FLOPs per second:

$$20 \cdot (1.606{\times}10^{10} \cdot 400 + 262144 \cdot 400^2) + 0.2 \cdot 8.078\times10^{14}$$
$$= 20 \cdot (6.42{\times}10^{12} + 4.19{\times}10^{10}) + 1.616\times10^{14} = 1.29\times10^{14} + 1.616\times10^{14} = 2.91\times10^{14}$$

That is $2.91\times10^{14}$ FLOPs/s of prefill demand against $R = 1.56\times10^{14}$ FLOPs/s of supply. **You are over capacity by 1.9x before a single token is decoded.** The 0.2 rps of long documents — one percent of your request volume — is 56% of your prefill compute. Source: derived.

This is the real reason long-context traffic is dangerous: not that any one request is slow, but that a trickle of it consumes a majority of your compute while contributing almost nothing to your request count. Any capacity model built on requests per second rather than *tokens* per second will be wrong by a factor of two on a workload like this. Chunked prefill does not fix that — nothing fixes an over-subscribed GPU except more GPUs or fewer tokens — but it does stop the over-subscription from expressing itself as multi-second freezes.

---

## 3. The fix: stop scheduling requests, start scheduling tokens

The insight is small and the consequences are large. The engine's scheduling decision should not be *"which requests run this step"* — it should be *"how many tokens does each request get this step"*, subject to a cap on the total.

vLLM's V1 architecture made exactly this change, and their write-up is the clearest statement of it I know: the V1 scheduler represents each scheduling decision as a simple dictionary, `{request_id: num_tokens}`, which "erases the distinction between prefill and decode" — a decoding request is just a request that gets one token, and a prefilling request is one that gets many ([vLLM V1 alpha announcement, 2025-01-27](https://vllm.ai/blog/2025-01-27-v1-alpha-release)). Once the decision is a token count, chunking is free: you hand a long prefill 472 tokens this step and 472 more next step, and nothing else in the engine has to know.

The cap on the total is the **token budget** — vLLM exposes it as `max_num_batched_tokens`, with `long_prefill_token_threshold` as the per-request cap that specifically enables chunking when set to a positive integer, capping the new tokens any one prefill contributes per step (per the *Anatomy* post linked above). The technique itself comes from the Sarathi line of work: [*SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills*](https://arxiv.org/abs/2308.16369) introduced chunked prefills plus decode-maximal batching, and [*Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*](https://arxiv.org/abs/2403.02310) (OSDI '24) named the resulting property **stall-free batching** and reports serving-capacity improvements of up to 2.6x for Mistral-7B on a single A100 under latency SLOs. Cite those two when someone asks where this came from.

![Two columns comparing one unbudgeted iteration against 152 budgeted iterations with their time to first token and worst-case inter-token latency](/imgs/blogs/chunked-prefill-and-the-ttft-tpot-tradeoff-3.webp)

The comparison above holds a fact that is easy to miss and worth stating as a law, because everything downstream depends on it.

> **Total prefill FLOPs are chunk-size invariant.** Splitting a $P$-token prefill into chunks changes neither the linear term nor the attention term.

The linear term is obviously invariant: $\sum_j 2Nc_j = 2NP$. The attention term is the interesting one. A chunk of $c$ tokens starting at cursor $s$ attends the $s$ cached positions before it plus a causal triangle within itself, so its query-key pair count is $c \cdot s + c(c-1)/2 \approx c(s + c/2)$. Summing over the chunks, with $s$ advancing by $c$ each time, is a Riemann sum for $\int_0^P s\,ds = P^2/2$ — exactly the pair count of the unchunked prefill. No work is created and none is destroyed.

That invariance is the whole reason chunked prefill is nearly free. You are not doing more math; you are *interleaving* the same math with other people's decode steps. What you *do* pay for is real but bounded, and there are exactly three costs:

1. **Re-reading the weights.** Each step reads all 16.06 GB of weights from HBM, which is 7.9 ms at 2,039 GB/s. Unchunked you pay it once; in $k$ chunks you pay it $k$ times. It is usually hidden — a compute-bound chunk overlaps the weight read — but it sets a hard floor on how small a useful chunk can be.
2. **Re-running the decoders.** Every step you chunk into is a step where the 40 decoders also advance, costing ${2N n_d + A n_d S_d}$ FLOPs. This is not waste — it is the decoders' own work getting done, and it is the entire point — but it does lengthen the prefiller's wall-clock TTFT.
3. **Per-step fixed overhead.** Kernel launches, the Python scheduler, building input tensors, the sampler. Small on a well-built engine, not zero.

Here is the motion, because a still frame genuinely cannot show it: the same prefill work in one lane and in seven, with the decode ticks that survive it.

<figure class="blog-anim">
<svg viewBox="0 0 720 300" role="img" aria-label="Two serving lanes over the same interval: the unchunked lane runs one long prefill while its decode ticks stay dark, the chunked lane advances the same prefill in slices while decode ticks keep firing between them" style="width:100%;height:auto;max-width:820px">
<title>Chunked prefill keeps decode ticks alive while the same prefill work completes</title>
<style>
.cp-h{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.cp-s{font:400 12.5px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.cp-track{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.cp-bar{fill:var(--accent,#6366f1)}
.cp-tick{fill:var(--text-secondary,#6b7280)}
.cp-rule{stroke:var(--border,#d1d5db);stroke-width:1;stroke-dasharray:4 4}
@keyframes cp-fill{0%{transform:scaleX(0)}6%{transform:scaleX(0)}92%{transform:scaleX(1)}100%{transform:scaleX(1)}}
@keyframes cp-late{0%,86%{opacity:.13}93%{opacity:1}100%{opacity:.13}}
@keyframes cp-beat{0%,100%{opacity:.13}45%{opacity:1}}
.cp-smooth{animation:cp-fill 12s linear infinite;transform-box:fill-box;transform-origin:left}
.cp-stepped{animation:cp-fill 12s steps(7,end) infinite;transform-box:fill-box;transform-origin:left}
.cp-frozen{opacity:.13;animation:cp-late 12s linear infinite}
.cp-live{opacity:.13;animation:cp-beat 1.5s ease-in-out infinite}
.cp-d2{animation-delay:.19s}.cp-d3{animation-delay:.38s}.cp-d4{animation-delay:.57s}
.cp-d5{animation-delay:.76s}.cp-d6{animation-delay:.95s}.cp-d7{animation-delay:1.14s}
@media (prefers-reduced-motion:reduce){.cp-smooth,.cp-stepped{animation:none;transform:scaleX(1)}.cp-frozen{animation:none;opacity:.13}.cp-live{animation:none;opacity:1}}
</style>
<text class="cp-h" x="24" y="26">No token budget — one 32768-token prefill owns the step</text>
<text class="cp-s" x="24" y="46">prefill work</text>
<rect class="cp-track" x="128" y="32" width="512" height="26" rx="6"/>
<rect class="cp-bar cp-smooth" x="128" y="32" width="512" height="26" rx="6"/>
<text class="cp-s" x="654" y="50">5.2 s</text>
<text class="cp-s" x="24" y="96">decode ticks</text>
<circle class="cp-tick cp-frozen" cx="152" cy="90" r="8"/>
<circle class="cp-tick cp-frozen" cx="224" cy="90" r="8"/>
<circle class="cp-tick cp-frozen" cx="296" cy="90" r="8"/>
<circle class="cp-tick cp-frozen" cx="368" cy="90" r="8"/>
<circle class="cp-tick cp-frozen" cx="440" cy="90" r="8"/>
<circle class="cp-tick cp-frozen" cx="512" cy="90" r="8"/>
<circle class="cp-tick cp-frozen" cx="584" cy="90" r="8"/>
<text class="cp-s" x="128" y="124">40 streams emit nothing until the prefill ends — p99 ITL 5183 ms</text>
<line class="cp-rule" x1="24" y1="150" x2="696" y2="150"/>
<text class="cp-h" x="24" y="188">Token budget 256 — the same prefill in bounded slices</text>
<text class="cp-s" x="24" y="208">prefill work</text>
<rect class="cp-track" x="128" y="194" width="512" height="26" rx="6"/>
<rect class="cp-bar cp-stepped" x="128" y="194" width="512" height="26" rx="6"/>
<text class="cp-s" x="654" y="212">5.9 s</text>
<text class="cp-s" x="24" y="258">decode ticks</text>
<circle class="cp-tick cp-live" cx="152" cy="252" r="8"/>
<circle class="cp-tick cp-live cp-d2" cx="224" cy="252" r="8"/>
<circle class="cp-tick cp-live cp-d3" cx="296" cy="252" r="8"/>
<circle class="cp-tick cp-live cp-d4" cx="368" cy="252" r="8"/>
<circle class="cp-tick cp-live cp-d5" cx="440" cy="252" r="8"/>
<circle class="cp-tick cp-live cp-d6" cx="512" cy="252" r="8"/>
<circle class="cp-tick cp-live cp-d7" cx="584" cy="252" r="8"/>
<text class="cp-s" x="128" y="286">every stream advances one token per step — p99 ITL 50 ms</text>
</svg>
<figcaption>The same prefill work finishes in both lanes, but only the budgeted lane lets the forty decoding streams keep emitting tokens while it runs.</figcaption>
</figure>

Notice what does *not* change between the lanes: the prefill bar reaches the right edge at almost the same moment. That is the invariance law made visual, and it is why this trade-off is so favorable — you give up a little TTFT and get back two orders of magnitude of ITL.

---

## 4. Implementing it in nanoserve

Time to write code. The engine we are extending is the one from post 12: a waiting queue, a running set, and a `step()` that admits, runs, and retires requests once per iteration. Three things change.

### 4.1 A request needs a cursor

Until now a request was either "waiting" (no KV) or "running" (fully prefilled, decoding). Chunked prefill introduces a third state — *partially prefilled* — and the state is a single integer.

```python
# nanoserve/request.py
from dataclasses import dataclass, field
from nanoserve.blocks import PagedSequence


@dataclass
class Request:
    """One in-flight request. `num_computed` is the whole new idea."""

    req_id: str
    prompt_ids: list[int]
    max_new_tokens: int = 128

    # How many of this request's tokens already have KV entries in the cache.
    # Starts at 0. Rises by the chunk size each prefill step. Once it reaches
    # len(prompt_ids) the request is a decoder and rises by 1 per step.
    num_computed: int = 0

    output_ids: list[int] = field(default_factory=list)
    seq: PagedSequence | None = None      # block table, from post 8

    arrival_s: float = 0.0
    first_token_s: float | None = None    # set when TTFT is observed

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_ids)

    @property
    def is_prefilling(self) -> bool:
        return self.num_computed < self.prompt_len

    @property
    def remaining_prefill(self) -> int:
        return max(0, self.prompt_len - self.num_computed)

    def token_ids(self) -> list[int]:
        """Prompt followed by generated tokens — the full sequence so far."""
        return self.prompt_ids + self.output_ids

    def slice_for(self, n: int) -> list[int]:
        """The next n token ids this request wants processed."""
        ids = self.token_ids()
        return ids[self.num_computed : self.num_computed + n]
```

`num_computed` is doing a lot of work here, and it is worth being explicit about why it unifies the two phases. During prefill it counts prompt tokens whose KV has been written. During decode it counts *everything* whose KV has been written, which is the prompt plus the outputs so far. A decoder is simply a request whose `remaining_prefill` is zero and which therefore takes exactly one token per step. There is no branch in the executor; there is one integer.

### 4.2 The budget planner

This is `nanoserve/budget.py`, and it is the heart of the post. It returns a plan in exactly vLLM's V1 shape: a dict from request id to token count.

```python
# nanoserve/budget.py
from __future__ import annotations
from dataclasses import dataclass
from nanoserve.blocks import BlockAllocator
from nanoserve.request import Request


@dataclass
class BudgetConfig:
    # Total new tokens the engine will process in one forward pass.
    # vLLM calls this max_num_batched_tokens.
    max_num_batched_tokens: int = 256
    # Per-request cap on prefill tokens per step. 0 disables chunking, i.e.
    # a prefill must be admitted whole or not at all. vLLM calls this
    # long_prefill_token_threshold.
    long_prefill_token_threshold: int = 0
    # Never run more than this many sequences concurrently.
    max_num_seqs: int = 64


def plan_step(
    running: list[Request],
    waiting: list[Request],
    alloc: BlockAllocator,
    cfg: BudgetConfig,
) -> dict[str, int]:
    """Decide how many tokens each request gets this iteration.

    Returns {req_id: num_tokens}. A value of 1 for a fully prefilled request
    means "decode one token". A value of c for a prefilling request means
    "process prompt tokens [num_computed, num_computed + c)".

    Invariant: sum(plan.values()) <= cfg.max_num_batched_tokens.
    """
    plan: dict[str, int] = {}
    left = cfg.max_num_batched_tokens

    # --- Phase 1: decoders are paid first, one token each. -----------------
    # This ordering IS the policy. Decoders already have a user watching a
    # half-written sentence; a prefill has a user watching a spinner. Paying
    # decoders first is what bounds ITL.
    for r in running:
        if r.is_prefilling:
            continue
        if left < 1:
            break                      # budget too small for the running set
        if not _has_room(r, 1, alloc):
            continue                   # out of blocks: post 10's preemption path
        plan[r.req_id] = 1
        left -= 1

    # --- Phase 2: finish partially prefilled requests, oldest first. -------
    # In-progress prefills outrank brand-new ones: half-done work is capital,
    # and letting it sit only lengthens someone's TTFT while holding blocks.
    for r in sorted((r for r in running if r.is_prefilling),
                    key=lambda r: r.arrival_s):
        if left <= 0:
            break
        take = _prefill_take(r, left, alloc, cfg)
        if take > 0:
            plan[r.req_id] = take
            left -= take

    # --- Phase 3: admit new requests into whatever budget survives. --------
    while waiting and left > 0 and len(running) + 1 <= cfg.max_num_seqs:
        r = waiting[0]
        take = _prefill_take(r, left, alloc, cfg, fresh=True)
        if take <= 0:
            break                      # cannot even start it: leave it queued
        waiting.pop(0)
        running.append(r)
        plan[r.req_id] = take
        left -= take

    assert sum(plan.values()) <= cfg.max_num_batched_tokens
    return plan
```

Three design decisions in there deserve defending, because each is a place where reasonable engines differ.

**Decoders first, unconditionally.** This is what makes the ITL bound hold. If prefill could outbid decode for budget, a burst of long prompts would starve the running set and you would be back to multi-second gaps — just spread over more steps. Sarathi-Serve calls the resulting property stall-free batching precisely because decodes are never displaced.

**In-progress prefills before new admissions.** A half-finished prefill is holding KV blocks and a user's TTFT clock is running. Preferring it is both fairness and memory hygiene. The alternative — strict FCFS across all prefills — is what vLLM's default scheduler does, and it behaves the same here because in-progress requests arrived earlier by construction.

**A prefill that cannot get any tokens stays queued rather than being admitted at zero.** Admitting a request you cannot advance burns a `max_num_seqs` slot and creates a request whose TTFT clock runs while it does nothing. Post 15's admission control makes this a real policy; here it is one `break`.

### 4.3 Chunk sizing, and the block allocator's veto

`_prefill_take` is where the token budget meets the paged KV cache from [post 8](/blog/machine-learning/inference-engineering/paged-kv-cache-implementing-blocks-and-a-block-table). A chunk is not just "as many tokens as the budget allows" — it is "as many tokens as the budget allows *and the free block pool can back*". A 472-token chunk with a block size of 16 needs up to 30 fresh physical blocks, and if the pool has 12 you must shrink the chunk or you will raise `OutOfBlocks` halfway through a forward pass, which is a far worse failure than a smaller chunk.

```python
# nanoserve/budget.py (continued)

def _room_in_blocks(r: Request, alloc: BlockAllocator) -> int:
    """How many more tokens this request's KV can hold right now.

    Two sources: the unused tail of its last allocated block, plus every
    free block in the pool. Both are expressed in tokens.
    """
    bs = alloc.block_size
    if r.seq is None:
        tail = 0
    else:
        tail = len(r.seq.block_table) * bs - r.seq.num_tokens
    return tail + alloc.num_free * bs


def _has_room(r: Request, n: int, alloc: BlockAllocator) -> bool:
    return _room_in_blocks(r, alloc) >= n


def _prefill_take(
    r: Request,
    left: int,
    alloc: BlockAllocator,
    cfg: BudgetConfig,
    fresh: bool = False,
) -> int:
    """How many prefill tokens this request gets this step."""
    take = min(r.remaining_prefill, left)

    # The chunking knob itself. 0 means "no chunking": either the whole
    # remaining prefill fits in the budget or the request waits.
    if cfg.long_prefill_token_threshold > 0:
        take = min(take, cfg.long_prefill_token_threshold)
    elif fresh and take < r.remaining_prefill:
        return 0                       # unchunked: all or nothing

    # The allocator's veto. Round DOWN to a block boundary when we have to
    # truncate, so we never leave a request straddling a block we cannot
    # complete on the next step for a silly off-by-a-few reason.
    room = _room_in_blocks(r, alloc)
    if take > room:
        bs = alloc.block_size
        take = (room // bs) * bs
    return max(0, take)
```

The `elif fresh and take < r.remaining_prefill: return 0` line is the *old* behaviour, kept deliberately. Setting `long_prefill_token_threshold = 0` gives you back the unchunked engine, which is exactly what you want for A/B measurement and for the frontier script below. A knob that cannot be turned off is not a knob.

![A five-layer stack showing a 512-token budget decomposing into decoder tokens, the prefill chunk, the new KV blocks it needs, and the resulting step cost](/imgs/blogs/chunked-prefill-and-the-ttft-tpot-tradeoff-4.webp)

The accounting above is the picture to hold onto: the budget is spent top-down, decoders take their fixed cut, and the prefill chunk is the residual. Change the number of running decoders and the chunk size changes with it — which is a real dynamic behaviour, not a rounding detail, and it means your prefill throughput degrades as your concurrency rises.

### 4.4 Executing a mixed batch

The forward pass now receives a batch that contains 40 sequences contributing one token each and one sequence contributing 472. The standard way to express that is a **flattened** batch with a start-offset array — vLLM's `query_start_loc`, FlashAttention's varlen interface, and the "one concatenated super sequence with position and mask isolation" that the *Anatomy* post describes.

```python
# nanoserve/engine.py
import torch


def build_step_inputs(plan: dict[str, int], reqs: dict[str, Request], device="cuda"):
    """Flatten the plan into one packed batch the model can run.

    Returns:
      input_ids     [T]        every token processed this step, concatenated
      positions     [T]        each token's absolute position in its sequence
      query_start   [B+1]      exclusive prefix sum: where each request starts
      seq_lens      [B]        total KV length AFTER this step, per request
      slot_mapping  [T]        flat KV slot for each token, from PagedSequence
    """
    input_ids, positions, slot_mapping = [], [], []
    query_start, seq_lens = [0], []

    for rid, n in plan.items():
        r = reqs[rid]
        ids = r.slice_for(n)
        assert len(ids) == n, f"{rid}: wanted {n} tokens, sequence had {len(ids)}"

        input_ids.extend(ids)
        # Positions are absolute. A chunk starting at cursor 30120 produces
        # positions 30120..30591 — RoPE must see the true position, not the
        # offset within the chunk. Getting this wrong is the single most
        # common chunked-prefill bug and it produces fluent nonsense.
        positions.extend(range(r.num_computed, r.num_computed + n))
        # Reserve KV slots; this may allocate new physical blocks.
        slot_mapping.extend(r.seq.append(n))

        query_start.append(query_start[-1] + n)
        seq_lens.append(r.num_computed + n)

    t = lambda xs, dt=torch.int32: torch.tensor(xs, dtype=dt, device=device)
    return (t(input_ids, torch.long), t(positions, torch.long),
            t(query_start), t(seq_lens), t(slot_mapping))
```

The comment about positions is not hypothetical. When you chunk a prefill, the second chunk's tokens are at absolute positions 472 through 943, and if you naively pass `torch.arange(n)` the rotary embedding rotates them as if they were positions 0 through 471. The model does not crash. It does not produce garbage either. It produces *coherent text that ignores the middle of the document*, because every chunk after the first has been positionally aliased onto the beginning. I have seen this class of bug survive a code review, a demo, and a week of production, because the only symptom is that summaries of long documents are subtly bad. Assert on it.

The attention call itself must give the chunk's queries visibility of the whole cached prefix:

```python
# nanoserve/attention.py

def chunked_attention(q, k, v, query_start, seq_lens, block_tables, kv_cache):
    """Attention for a mixed batch of decodes and prefill chunks.

    For request i:
      queries  = q[query_start[i] : query_start[i+1]]        (1 or c of them)
      keys     = all seq_lens[i] cached positions, gathered via block_tables
      mask     = causal, aligned so query j sees keys 0 .. (start_pos + j)

    In nanoserve this is flash_attn_varlen_func; in vLLM it is FlashAttention 3,
    which the V1 write-up credits specifically for handling the dynamism of a
    batch that mixes prefill chunks and single-token decodes in one call.
    """
    ...
```

The alignment clause is the subtle part. A chunk starting at cursor $s$ has queries at absolute positions $s, s+1, \ldots, s+c-1$, and query $j$ must see keys ${0}$ through $s+j$ — the entire prefix plus the causal triangle inside its own chunk. FlashAttention's varlen kernels express this with separate query and key cumulative-length arrays and a bottom-right-aligned causal mask. If your kernel only supports top-left alignment, a chunk of length $c$ at cursor $s$ will mask out everything past position $c$, and the request will attend only to the first $c$ tokens of the document. Same failure mode as the RoPE bug: plausible output, wrong content.

### 4.5 Wiring it into `step()`

```python
# nanoserve/scheduler.py

def step(self) -> list[Output]:
    """One engine iteration under a token budget."""
    plan = plan_step(self.running, self.waiting, self.alloc, self.cfg)
    if not plan:
        return []                              # nothing runnable this step

    ids, pos, qstart, seq_lens, slots = build_step_inputs(
        plan, self.reqs, device=self.device)

    with torch.inference_mode():
        logits = self.model(ids, pos, qstart, seq_lens, slots, self.kv)

    outputs = []
    for i, (rid, n) in enumerate(plan.items()):
        r = self.reqs[rid]
        r.num_computed += n                    # advance the cursor

        if r.is_prefilling:
            continue                           # mid-prefill: no token to emit

        # Only the LAST token of a chunk produces a logit worth sampling, and
        # only when the chunk finished the prompt. Everything else is cache fill.
        tok = self.sampler(logits[qstart[i + 1] - 1])
        r.output_ids.append(tok)
        if r.first_token_s is None:
            r.first_token_s = time.perf_counter()   # TTFT observed here
        outputs.append(Output(rid, tok))

        if self._finished(r):
            self._retire(r)

    self.steps += 1
    return outputs
```

The `if r.is_prefilling: continue` is where the chunking becomes visible at the API boundary: for 151 of the 152 steps that a 32,768-token prefill takes under a 256-token budget, that request produces *no output at all*. It is consuming GPU, holding blocks, advancing a cursor, and streaming nothing. Any component that assumes "a scheduled request produces a token" — a streaming adapter, a metrics exporter, a timeout watchdog — needs to learn about this state, and forgetting to teach it is how you get a request that is making perfect progress being killed by an idle-stream timeout.

### 4.6 What the accounting looks like

Instrument the plan and print it. This log is the single most useful diagnostic in a chunked-prefill engine.

```python
# nanoserve/scheduler.py (continued)

def log_plan(self, plan: dict[str, int]) -> None:
    dec = {k: v for k, v in plan.items() if not self.reqs[k].is_prefilling_before}
    pre = {k: v for k, v in plan.items() if self.reqs[k].is_prefilling_before}
    used = sum(plan.values())
    print(f"step {self.steps:5d} | budget {used:5d}/{self.cfg.max_num_batched_tokens} "
          f"| decode {len(dec):3d} reqs x1 "
          f"| prefill {len(pre)} reqs {list(pre.values())} "
          f"| blocks {self.alloc.num_used}/{self.alloc.num_blocks}")
```

Running the engine with `max_num_batched_tokens=256` while one 32k request prefills alongside 40 chatters prints something with this shape:

```console
step    17 | budget   256/256 | decode  40 reqs x1 | prefill 1 reqs [216] | blocks 5344/16384
step    18 | budget   256/256 | decode  40 reqs x1 | prefill 1 reqs [216] | blocks 5358/16384
step    19 | budget   256/256 | decode  40 reqs x1 | prefill 1 reqs [216] | blocks 5372/16384
step    20 | budget   248/256 | decode  40 reqs x1 | prefill 1 reqs [208] | blocks 5385/16384
step    21 | budget   256/256 | decode  41 reqs x1 | prefill 1 reqs [215] | blocks 5399/16384
...
step   152 | budget   192/256 | decode  40 reqs x1 | prefill 1 reqs [152] | blocks 7392/16384
step   153 | budget    41/256 | decode  41 reqs x1 | prefill 0 reqs []    | blocks 7392/16384
```

Read that log for the three things it tells you. **Budget saturation** near 256 every step means the engine is compute-saturated and the budget is the binding constraint — good. **The block counter rising by about 14 per step** is the chunk's KV landing in the pool: 216 tokens at 16 tokens per block is 13.5 blocks, and the fractional part accumulates. **Step 20's short chunk of 208** is the allocator's veto rounding down to a block boundary. And step 153 is the moment the prefill completes and the budget suddenly goes 84% idle, because 41 decoders only need 41 tokens — the engine is now memory-bound again and the budget is irrelevant. A budget that is idle most of the time is not a bug; it is a cap, and caps are supposed to be slack when there is nothing to cap.

---

## 5. The frontier, derived

Now the payoff. Everything above gives us a cost model precise enough to compute the whole trade-off curve without touching a GPU — which is exactly the kind of number this series is allowed to publish, because you can rerun it and get the same digits.

Per step, with $n_d$ decoders at average context $S_d$, and a prefill chunk of $c$ tokens at cursor $s$:

$$\text{FLOPs}(s, c) = 2N(n_d + c) + A\left[n_d S_d + c\left(s + \tfrac{c-1}{2}\right)\right]$$

$$\text{bytes}(s, c) = W + B_{\text{tok}}\left(n_d S_d + s + c\right)$$

$$T_{\text{step}} = \max\!\left(\frac{\text{FLOPs}}{R},\ \frac{\text{bytes}}{\text{BW}}\right)$$

with $W = 1.606\times10^{10}$ bytes of weights, $R = 1.56\times10^{14}$ FLOP/s, $\text{BW} = 2.039\times10^{12}$ B/s. The `max` is the roofline: a step takes as long as the slower of its compute and its memory traffic. This is an idealization — it assumes perfect overlap and no fixed overhead — and I will say so again when I report the numbers.

Here is the script. It is arithmetic; it runs anywhere.

```python
# nanoserve/bench/chunk_frontier.py
"""Derive the TTFT / ITL frontier for chunked prefill. No GPU required.

Every number this prints comes from the closed-form cost model in section 5
of the post. Run it and you will get exactly the same digits I did, because
there is no hardware in the loop -- only a datasheet and a config.
"""
import math

# --- Llama-3.1-8B, bf16 ------------------------------------------------------
N, L, H, HKV, D, DTYPE_B = 8.03e9, 32, 32, 8, 128, 2
TWO_N = 2 * N                       # 1.606e10 FLOPs/token, linear layers
ATTN = 4 * L * H * D                # 524288 FLOPs per query-key pair
KV_B = 2 * L * HKV * D * DTYPE_B    # 131072 bytes per token of context
W_B = N * DTYPE_B                   # 1.606e10 bytes of weights

# --- A100 80GB SXM (NVIDIA datasheet) + an explicit MFU assumption -----------
PEAK = 312e12                       # dense BF16 TFLOP/s
BW = 2039e9                         # HBM2e bytes/s
MFU = 0.50                          # ASSUMPTION, stated in the prose
R = PEAK * MFU

ND, SD = 40, 2000                   # 40 decoders at 2000 tokens of context


def step_time(cursor: float, chunk: int) -> float:
    pairs = ND * SD + chunk * (cursor + (chunk - 1) / 2)
    flops = TWO_N * (ND + chunk) + ATTN * pairs
    byts = W_B + KV_B * (ND * SD + cursor + chunk)
    return max(flops / R, byts / BW)


def run(prompt: int, budget: int | None) -> dict:
    """budget=None means unchunked: the whole prompt in one step."""
    chunk_cap = prompt if budget is None else max(1, budget - ND)
    cursor, steps, total, worst = 0, 0, 0.0, 0.0
    while cursor < prompt:
        c = min(chunk_cap, prompt - cursor)
        t = step_time(cursor, c)
        total += t
        worst = max(worst, t)
        cursor += c
        steps += 1
    return {"steps": steps, "ttft_s": total,
            "p99_itl_ms": worst * 1e3, "mean_itl_ms": total / steps * 1e3}


if __name__ == "__main__":
    P = 32768
    print(f"Llama-3.1-8B bf16 | A100 80GB | MFU {MFU:.0%} | "
          f"{ND} decoders @ {SD} ctx | prompt {P}")
    print(f"{'budget':>9} {'chunk':>7} {'steps':>7} {'TTFT s':>9} "
          f"{'p99 ITL ms':>12} {'mean ITL ms':>12}")
    for b in [None, 8192, 2048, 1024, 512, 256, 128, 64]:
        r = run(P, b)
        chunk = P if b is None else b - ND
        print(f"{str(b or 'none'):>9} {chunk:>7} {r['steps']:>7} "
              f"{r['ttft_s']:>9.2f} {r['p99_itl_ms']:>12.0f} "
              f"{r['mean_itl_ms']:>12.1f}")
```

Its output:

```console
Llama-3.1-8B bf16 | A100 80GB | MFU 50% | 40 decoders @ 2000 ctx | prompt 32768
   budget   chunk   steps    TTFT s   p99 ITL ms  mean ITL ms
     none   32768       1      5.18         5183       5182.7
     8192    8152       5      5.20         1625       1039.6
     2048    2008      17      5.25          421        308.9
     1024     984      34      5.33          211        156.7
      512     472      70      5.49          104         78.4
      256     216     152      5.85           50         38.5
      128      88     373      6.81           23         18.3
       64      24    1366     11.17           15          8.2
```

![A five-row comparison matrix of token budgets against step count, time to first token, worst-case inter-token latency, and a verdict](/imgs/blogs/chunked-prefill-and-the-ttft-tpot-tradeoff-5.webp)

That table is the frontier, and it has a shape worth internalizing.

**From unchunked down to a budget of 256, TTFT rises 13% and worst-case ITL falls by a factor of 104.** That is not a trade-off so much as a gift. The reason is the invariance law: the prefill FLOPs are identical in every row, and the only TTFT growth comes from the decoders' own work getting interleaved — $k$ steps each costing $2N n_d + A n_d S_d = 6.84\times10^{11}$ FLOPs, or 4.4 ms of the prefiller's wall clock per step. At $k = 152$ that is 0.67 s, which is the entire 5.18 to 5.85 s difference. You can see the whole column in one formula:

$$\text{TTFT}(k) \approx \frac{2NP + \tfrac{A}{2}P^2}{R} + k \cdot \frac{2N n_d + A n_d S_d}{R} = 5.18 + 0.00439k \ \text{s}$$

**Below a budget of about 128, the deal goes bad.** At budget 64 TTFT more than doubles (11.17 s) and ITL improves only from 23 ms to 15 ms. The reason is the roofline `max`: once the chunk's compute time falls below the step's memory time, you are paying a full 16 GB weight read to do almost no work. The break-even chunk size is where the two sides of the `max` meet:

$$\frac{2N c}{R} = \frac{W}{\text{BW}} \implies c_{\min} = \frac{R \cdot W}{2N \cdot \text{BW}} = \frac{1.56\times10^{14} \cdot 1.606\times10^{10}}{1.606\times10^{10} \cdot 2.039\times10^{12}} = 76.5$$

**Chunks below about 77 tokens are pure waste on this hardware.** The GPU spends longer fetching weights than using them. That threshold is a pure hardware ratio — $R/\text{BW}$, the ridge point, times the weight bytes over the per-token FLOPs — so it moves with the card and not with the model size in any simple way. On an H100 (989 TFLOP/s dense BF16, 3.35 TB/s HBM3, per [NVIDIA's H100 datasheet](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-datasheet)) the same formula at 50% MFU gives $c_{\min} = 0.5 \cdot 989\times10^{12} / 3.35\times10^{12} = 147.6$ — the *bigger* GPU needs *bigger* chunks, because its compute grew faster than its bandwidth.

**The knee is at the SLO you actually have.** There is no universally correct row. A code-completion product with a 200 ms ITL budget should sit at 1024. A voice product that needs tokens faster than a person speaks — call it 50 ms — must sit at 256 and eat the 13% TTFT. A batch summarization job with no human watching should turn chunking off entirely and take the 5.18 s.

---

## 6. A rule for picking the budget

Guessing is unnecessary. Invert the cost model.

You have an ITL SLO of $I$ seconds. The worst step is the one with the largest chunk at the largest cursor, which is a prefill chunk of $c$ tokens at cursor $P_{\max}$ (your `max_model_len`). Ignore the memory side, since at any useful chunk size the step is compute-bound. Then:

$$I \ge \frac{2N(n_d + c) + A\left(n_d S_d + c P_{\max}\right)}{R}$$

Solve for $c$, fold the decoders' fixed cost into a small correction, and you get the budget rule:

$$\boxed{\ B^\star \;=\; n_d \;+\; \frac{I \cdot R \;-\; \left(2N n_d + A n_d S_d\right)}{2N + A\,P_{\max}}\ }$$

Check it against the table. With $I = 0.050$ s, $R = 1.56\times10^{14}$, $n_d = 40$, $S_d = 2000$, $P_{\max} = 32768$:

- numerator: $0.050 \cdot 1.56\times10^{14} - 6.84\times10^{11} = 7.80\times10^{12} - 0.068\times10^{13} = 7.12\times10^{12}$
- denominator: $1.606\times10^{10} + 524288 \cdot 32768 = 1.606\times10^{10} + 1.718\times10^{10} = 3.324\times10^{10}$
- $c = 214$, so $B^\star = 254$.

The frontier script says budget 256 gives 50 ms. The closed form says 254. Source: both derived, from the same model — this is a consistency check, not independent evidence, but it means you can carry the formula instead of the table.

What the formula tells you about your knobs is more interesting than the number.

| If you change | Effect on the affordable budget | Why | Source |
| --- | --- | --- | --- |
| Loosen ITL SLO 50 to 200 ms | 254 to 894 | numerator scales with $I$ | derived |
| Upgrade A100 to H100 | 254 to 790 | $R$ triples at the same MFU | derived |
| Downgrade A100 to RTX 4090 | 254 to 155 | $R$ falls to 82.6 TFLOP/s | derived |
| Cap max context 32k to 8k | 254 to 423 | $A P_{\max}$ shrinks 4x | derived |
| Raise concurrency 40 to 128 decoders | 254 to 227 | decoders eat the numerator | derived |
| Halve $S_d$ (shorter chats) | 254 to 262 | small: $A n_d S_d$ is minor | derived |

The RTX 4090 row uses 165.2 TFLOP/s dense BF16 tensor throughput from [NVIDIA's Ada architecture whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf) (the 330.3 TFLOPS figure quoted there is with 2:1 structured sparsity; halve it for dense), at the same 50% MFU assumption.

Two operational conclusions fall straight out. First, **capping `max_model_len` is a latency lever, not just a memory lever.** Refusing 128k prompts lets you run a 4x larger budget at the same ITL, which improves prefill throughput for everyone. Second, **the budget is hardware-specific and must be re-tuned on every fleet migration.** Copying `max_num_batched_tokens=8192` from an H100 config to an L4 deployment will quietly triple your p99 ITL.

---

## 7. The other answer: stop sharing the GPU at all

Chunked prefill solves contention by *scheduling* prefill and decode onto one pool of GPUs. There is a second, structurally different answer: put them on *different* pools. Prefill happens on machines that do nothing else, the resulting KV cache is shipped over the network to a decode machine, and decode happens on machines that never see a prefill. This is **prefill-decode disaggregation**, and the repo has a full treatment of it in [prefill-decode disaggregation](/blog/machine-learning/model-serving/prefill-decode-disaggregation).

![A decision tree splitting prefill-decode contention into a one-pool software-scheduling branch and a two-pool network-transfer branch with their respective costs](/imgs/blogs/chunked-prefill-and-the-ttft-tpot-tradeoff-6.webp)

The tree above is the choice, and it is genuinely a choice — neither branch dominates.

For concrete numbers on the disaggregated branch, the most useful recent public result I know is vLLM's write-up of the **MoRIIO KV connector** ([vllm.ai, 2026-04-07](https://vllm.ai/blog/2026-04-07-moriio-kv-connector)), and I want to give the whole setup because a goodput number without its setup is meaningless. They serve **Qwen3-235B-A22B-FP8 on 8x AMD MI300X, split 4 prefill + 4 decode**, with an input sequence length of 2,000 and an output sequence length of 1,000, at an arrival rate of **8 requests per second**. The SLO is defined as **TTFT under 1 s and ITL under 50 ms**, and requests are counted as good only if they meet both. Against a collocated (non-disaggregated) baseline on the same eight GPUs, they report **2.5x higher goodput**, with **73 of 100 requests meeting the SLO in write mode, 70 of 100 in read mode, versus 30 of 100 collocated**.

That baseline number is the one to stare at. On a collocated deployment under that load, **70% of requests missed the SLO** — and the SLO was a perfectly ordinary "first token within a second, tokens smoother than 50 ms apart". Prefill-decode contention is not an exotic tail problem. At realistic load it is the majority case.

The mechanism has two modes, and the difference matters for your TTFT:

- **Write mode** (the default) pushes KV into the decode instance's memory *per layer, as the prefill computes it*. The transfer overlaps the prefill, so most of it is hidden.
- **Read mode** (`VLLM_MORIIO_CONNECTOR_READ_MODE=1`) has the decode instance pull the KV over RDMA *after* the prefill finishes, which adds a full prefill-length transfer to the critical path. Their own numbers show it slightly behind write mode: 70 versus 73 of 100.

And the honest costs, which the vLLM post states plainly and which you should not skip past:

- **TTFT increases.** You have inserted a network hop and a scheduling boundary into the first-token path. Disaggregation buys ITL and goodput by spending TTFT.
- **Prefix caching must be disabled** (`--no-enable-prefix-caching`). That is a brutal cost for chat and agentic workloads, where system prompts and conversation history are re-sent every turn. If your prefix hit rate is high, disaggregation may delete more value than it creates.
- **Single-node only** in that implementation — the eight GPUs are in one box, connected by the intra-node fabric. Cross-node disaggregation is a harder engineering problem and a different bandwidth regime.

Two more citations worth having in your pocket when this argument comes up at work. [*DistServe*](https://arxiv.org/abs/2401.09670) (OSDI '24) is the academic case for disaggregation and reports serving up to 7.4x more requests, or meeting 12.6x tighter SLOs, than collocated systems on their workloads. [*Splitwise*](https://arxiv.org/abs/2311.18677) (ISCA '24) makes the complementary hardware argument: because prefill is compute-bound and decode is memory-bound, the two phases want *different GPUs*, and a heterogeneous fleet can be cheaper at the same throughput. Both are worth reading in full; both are describing a fleet-level decision, not a code-level one.

Here is how I would actually decide.

| Dimension | Chunked prefill | PD disaggregation | Source |
| --- | --- | --- | --- |
| GPUs required | 1 or more, any count | 2 pools, min 2 nodes worth | derived |
| What it costs | 13% TTFT, per-step overhead | network hop, TTFT increase | derived / cited: MoRIIO |
| ITL spikes | bounded, not eliminated | eliminated by construction | cited: MoRIIO |
| Prefix caching | fully compatible | must be disabled in MoRIIO | cited: MoRIIO |
| Implementation | ~80 lines of scheduler | KV connector, RDMA, discovery | derived |
| Best at | mixed traffic, one box, high prefix reuse | high sustained load, both phases saturated | derived |
| Worst at | very long prompts on small GPUs | low load, high prefix reuse, one GPU | derived |
| Tunable per SLO | yes, continuously | coarse: pool ratio only | derived |

**My rule of thumb.** Chunked prefill first, always — it costs eighty lines and works on one GPU. Reach for disaggregation when all three of these are true: you are running enough sustained load that both a prefill pool and a decode pool would stay busy; your prefix cache hit rate is low enough that giving it up does not hurt (long unique documents, not chat); and you have measured that even a well-tuned budget cannot hit your ITL SLO because your longest prompts force a chunk size below the memory floor. If you are running one A100, the answer is chunked prefill and it is not close.

Note that the two are not mutually exclusive in principle — a disaggregated deployment still chunks prefills within its prefill pool, to bound the latency of *queued* prefills against each other. What disaggregation removes is the coupling between prefill and *decode*.

---

## 8. Stress tests: where the budget stops working

A knob you have not stressed is a knob you do not understand. Four scenarios, all derived from the same cost model.

### 8.1 Many simultaneous long prefills

The budget is shared, so $m$ concurrent prefills each get $(B - n_d)/m$ tokens per step under a fair split. With $B = 256$, $n_d = 40$ and $m = 4$, each prefill gets 54 tokens per step — *below the 77-token memory floor derived above*. All four requests now run at a chunk size where the GPU is fetching weights faster than it uses them, and the aggregate prefill throughput collapses.

The alternative policy — strict FCFS, where the oldest prefill takes the whole 216-token budget and the other three get nothing — finishes the first request in 152 steps and the fourth in 608, with mean TTFTs of 5.85, 11.7, 17.6 and 23.4 s. Fair-sharing gives all four a TTFT of about 23 s but with the memory-floor penalty, so realistically worse than 23 s for everyone.

| Policy for 4 concurrent 32k prefills | TTFT req 1 | TTFT req 4 | Mean TTFT | Source |
| --- | --- | --- | --- | --- |
| FCFS, one prefill at a time | 5.85 s | 23.4 s | 14.6 s | derived |
| Fair split, 54 tok each | 23.4 s | 23.4 s | 23.4 s | derived (floor penalty not modeled) |

**FCFS wins on mean TTFT and never crosses the memory floor.** This is the same result queueing theory gives for any work-conserving system with no deadlines: shortest-remaining-first or first-come-first-served beat processor sharing on mean response time. It is also why vLLM's default scheduler is FCFS, filling the budget with the oldest prefill first rather than spreading it. The policy question — and the fairness question it raises — is [the next post's](/blog/machine-learning/inference-engineering/the-scheduler-as-a-policy-problem) whole subject.

### 8.2 The prefill that never finishes

![Seven points along a 128k prefill showing the per-chunk cost climbing from 53 ms to 260 ms as the cursor advances and the total reaching 43.6 seconds](/imgs/blogs/chunked-prefill-and-the-ttft-tpot-tradeoff-7.webp)

The timeline above is the failure mode that catches people who tuned their budget on a 32k workload and then enabled 128k context.

Run a 131,072-token prompt at budget 512 (chunk 472). It takes $\lceil 131072/472 \rceil = 278$ steps, and the frontier script gives a TTFT of **43.6 s**. Worse, the *per-step cost is not constant*: the attention term $A \cdot c \cdot (s + c/2)$ grows linearly with the cursor $s$, so the chunk that was 53 ms at the start of the document costs 106 ms at cursor 33k, 203 ms at cursor 94k, and **260 ms at the end**. A budget chosen to hold ITL at 104 ms for a 32k prompt delivers 260 ms for a 128k one.

That is what the $P_{\max}$ term in the budget rule is protecting you from, and it is why the rule uses your *maximum* context and not your typical one. Three ways out, in order of how much I like them:

1. **Size the budget for $P_{\max}$.** The rule already does this; you just have to be honest about what $P_{\max}$ is. At $P_{\max} = 131072$, $I = 0.050$: denominator $= 1.606\times10^{10} + 524288 \cdot 131072 = 8.03\times10^{10}$, giving $c = 89$ and $B^\star = 129$. Uncomfortably close to the memory floor of 77 — which is the model telling you that a single A100 cannot serve 128k prompts and hold a 50 ms ITL. That is a real answer, and it is better to know it before the incident.
2. **Cap `max_model_len`.** Refuse the prompts you cannot serve. This is not a cop-out; it is capacity planning with a 400-level status code.
3. **Make the chunk size cursor-dependent.** Shrink the chunk as the cursor advances so that $c \cdot (s + c/2)$ stays roughly constant. Elegant, and it keeps ITL flat, but it makes late chunks tiny and drives you into the memory floor exactly when the request is most expensive. I would reach for this only after the first two.

There is also a starvation flavor of this problem: under a tiny budget with a steady arrival of short requests, an in-progress long prefill can be perpetually outbid if your policy admits new work ahead of it. My `plan_step` avoids this by putting in-progress prefills in phase 2, ahead of new admissions in phase 3, but "the long request eventually finishes" is a *property of your policy*, not a property of chunked prefill.

### 8.3 Prefix caching changes the whole calculus

Now suppose 30,000 tokens of that 32,768-token prompt are a system preamble the request shares with everyone else, and [prefix caching](/blog/machine-learning/inference-engineering/prefix-sharing-radix-trees-and-copy-on-write) is on. The *Anatomy* post is precise about the granularity: block hashes are computed over complete blocks only, so a partial prefix leaves `long_prefix_len % block_size` tokens to recompute. With a block size of 16, a 30,000-token preamble is exactly 1,875 complete blocks and none of it is stranded — but shift that preamble to 30,005 tokens and the last five fall outside a complete block and get recomputed. Design your preambles on block boundaries if you can.

Effective prefill length drops from 32,768 to 2,768 tokens, *starting at cursor 30,000*. Running that through the cost model:

$$\text{FLOPs} \approx 2N \cdot 2768 + A \cdot \frac{32768^2 - 30000^2}{2} = 4.45\times10^{13} + 4.55\times10^{13} = 9.00\times10^{13}$$

$$\text{TTFT} \approx \frac{9.00\times10^{13}}{1.56\times10^{14}} + k \cdot 0.00439 = 0.577 + 6 \cdot 0.00439 = 0.603 \text{ s}$$

**TTFT falls from 5.85 s to 0.61 s — a 9.7x cut.** But look at what does *not* change: the chunks are still 472 tokens at a cursor around 30,000, so each step still costs about 100 ms and the p99 ITL is unchanged. Source: derived.

That asymmetry is the takeaway and it is genuinely useful:

> **Prefix caching cuts TTFT. Chunked prefill cuts ITL. They are orthogonal and you want both.**

It also means the two features interact through the budget in a way worth knowing: a high prefix hit rate makes prefills *shorter*, which means fewer of them are long enough to need chunking at all, which means the budget binds less often. On a chat workload with an 80% hit rate you may find the budget almost never engages — and then one 128k document arrives with no cache hit and it engages very much indeed. Tune for that request, not the average one.

One more caution from the *Anatomy* post worth passing on: the block hash is a function of the previous block's hash plus the current block's tokens, so a cache hit is a *prefix* match, not a substring match. Changing one token near the start of a 30,000-token system prompt invalidates all 1,875 blocks after it. Version your system prompts deliberately.

### 8.4 Small GPUs make the trade-off worse in both directions

Run the frontier script with `PEAK = 165.2e12` (RTX 4090, dense BF16) and `BW = 1008e9` (its GDDR6X bandwidth). Two things move. The affordable budget at a 50 ms SLO drops to about 155, because $R$ fell by half. And the memory floor $c_{\min} = R \cdot W / (2N \cdot \text{BW})$ becomes $8.26\times10^{13}/1.008\times10^{12} = 81.9$ tokens — barely below the budget you can afford. The usable window between "chunk too big for the SLO" and "chunk too small to be worth a weight read" has narrowed from roughly 77–214 tokens on the A100 to roughly 82–115 on the 4090.

On smaller cards still, or with longer $P_{\max}$, that window closes entirely and there is *no* chunk size that both meets your ITL SLO and does useful work. That is not a tuning failure; it is the hardware telling you the SLO is not purchasable at that context length. Believe it early.

---

## 9. Measuring it honestly

You cannot tune a budget against a metric that hides the thing you are tuning for. Four rules.

**Measure ITL, not TPOT, and report a high percentile.** As shown in section 2, a single 5.19 s stall inside a 500-token response moves TPOT from 13 ms to 23 ms — a number most dashboards would call healthy. The same event puts a 5,190 ms sample in the ITL distribution. Record every inter-token gap, not the per-request average.

**Timestamp at the streaming boundary, not inside the engine.** The number a user experiences includes detokenization, the SSE write, and any buffering your web framework does. If you time inside `step()` you will measure a cleaner system than the one you ship. [The incremental detokenization post](/blog/machine-learning/inference-engineering/the-tokenizer-boundary-and-incremental-detokenization) covers a real gotcha here: a partial UTF-8 sequence can hold a token back for one step, which shows up as an ITL spike that has nothing to do with the scheduler.

**Load-test open-loop.** A closed-loop harness — $n$ workers, each sending the next request only after the previous finishes — cannot produce the pathology this post is about, because the long request and the chat requests never overlap in the way that matters. Use Poisson arrivals at a fixed rate, let the queue grow if the system cannot keep up, and record what happens. This is also the only way to measure goodput, which is [the next post's](/blog/machine-learning/inference-engineering/the-scheduler-as-a-policy-problem) topic and the metric MoRIIO's 2.5x is quoted in.

**Include a long-prompt tail in the trace.** The whole failure mode requires a rare long prompt. A benchmark of uniform 512-token prompts will show you a flat ITL distribution at every budget setting and tell you the knob does nothing. Mix in 1–5% of prompts at your `max_model_len` — that is what real traffic looks like, and it is the traffic the budget exists for.

And the standard GPU-timing hygiene, which applies to every measurement in this series and is covered properly in [setting up a reproducible benchmark](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark):

```python
# nanoserve/bench/itl.py
import time, torch, numpy as np


def measure_itl(engine, trace, warmup_steps=50):
    """Record every inter-token gap under an open-loop arrival trace."""
    for _ in range(warmup_steps):          # warm caches, autotune, CUDA graphs
        engine.step()
    torch.cuda.synchronize()

    last = {}
    gaps = []
    t0 = time.perf_counter()
    for arrival, req in trace:             # trace carries absolute arrival times
        while time.perf_counter() - t0 < arrival:
            for out in engine.step():      # open loop: keep stepping while we wait
                now = time.perf_counter()
                if out.req_id in last:
                    gaps.append((now - last[out.req_id]) * 1e3)
                last[out.req_id] = now
        engine.submit(req)

    g = np.array(gaps)
    print(f"ITL  p50 {np.percentile(g,50):7.1f} ms   "
          f"p95 {np.percentile(g,95):7.1f} ms   "
          f"p99 {np.percentile(g,99):7.1f} ms   max {g.max():7.1f} ms")
    return g
```

Run that at several `max_num_batched_tokens` values against the same trace and plot p99 ITL against mean TTFT. You will get the empirical version of the frontier table, and the two should have the same *shape* even if the absolute numbers differ from my derived ones — my model has no kernel-launch overhead, no Python, no CUDA graph capture, and an MFU assumption. **On an A100 with Llama-3.1-8B I would expect the measured curve to sit above the derived one, plausibly 15–40% higher on both axes, with the same knee location.** If your measured knee is somewhere very different, that discrepancy is information: it usually means either your MFU is far from 50% or your per-step overhead is much larger than you think.

---

## 10. Case studies and public numbers

Four public results that bear directly on this knob. Each with its setup, because a number without a setup is a rumor.

**Sarathi-Serve (OSDI '24).** The paper that named stall-free batching. Its central claim is that chunked prefills plus decode-prioritized batching let you raise load without violating latency SLOs, and it reports serving-capacity improvements of **up to 2.6x for Mistral-7B on a single A100** and larger factors for bigger models on multi-GPU setups, all measured under explicit tail-latency constraints. The framing to steal from it: capacity *at an SLO*, not raw throughput. [arXiv:2403.02310](https://arxiv.org/abs/2403.02310)

**vLLM V1 (2025-01-27).** The architectural change that made chunked prefill the default rather than a special mode: representing every scheduling decision as `{request_id: num_tokens}` so prefill, decode, chunked prefill, prefix caching and speculative decoding all compose in one policy. vLLM reports **up to 1.7x throughput over V0** on Llama 3.1 8B and 3.3 70B with ShareGPT traces. Note the caveats they state at that release: Ampere-or-newer NVIDIA only, and several features (LoRA, pipeline parallelism, structured decoding) not yet supported at the time. [vllm.ai/blog/2025-01-27-v1-alpha-release](https://vllm.ai/blog/2025-01-27-v1-alpha-release)

**MoRIIO KV connector (2026-04-07).** Covered in section 7: Qwen3-235B-A22B-FP8 on 8x MI300X, 4+4 split, ISL 2000 / OSL 1000, 8 rps, SLO of TTFT under 1 s and ITL under 50 ms. **2.5x goodput, 73/100 requests meeting SLO versus 30/100 collocated.** With the honest costs stated: TTFT rises, prefix caching off, single node. [vllm.ai/blog/2026-04-07-moriio-kv-connector](https://vllm.ai/blog/2026-04-07-moriio-kv-connector)

**The Anatomy post (2025-09-05).** The reference for the flag names and the metric definitions used throughout this post: `long_prefill_token_threshold` as a positive integer to enable chunking; block size default 16; the waiting/running queue structure; TTFT, ITL and TPOT defined precisely. It also describes vLLM's continuous batching as flattening all sequences into one concatenated "super sequence" with position and mask isolation — which is exactly the packed batch `build_step_inputs` constructs above. [vllm.ai/blog/2025-09-05-anatomy-of-vllm](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm)

One synthesis worth stating across all four: **every one of these results is quoted against a latency constraint, not as raw throughput.** That is not an accident of presentation. Chunked prefill *slightly reduces* peak throughput — it adds per-step overhead and extra weight reads — and it is a large win only when you score yourself on requests that met an SLO. If your objective function is tokens per second with no latency term, chunked prefill will look like a regression, and you will have optimized for a metric nobody experiences.

---

## 11. When to reach for this (and when not to)

**Turn chunked prefill on, by default, if:** you serve interactive traffic where someone watches tokens appear; your prompt length distribution has any tail at all; you run more than a handful of concurrent requests; or you have ever had a "the stream froze for a few seconds" report you could not reproduce. On modern vLLM this is already the default and you should leave it that way.

**Tune the budget down from the default if:** your ITL SLO is tight (voice, live translation, code completion under a keystroke budget), or your `max_model_len` is large. Use the budget rule from section 6 with *your* $P_{\max}$, not your median prompt.

**Tune it up, or turn chunking off, if:** nobody is watching. Offline batch inference, evaluation harnesses, synthetic data generation, embedding jobs — all of these have no ITL to protect and every per-step overhead you avoid is throughput you keep. A budget of `max_model_len` is the right setting for a batch job.

**Do not reach for chunked prefill when the real problem is something else.** Three misdiagnoses I would look for first: if your prompts are long because you resend an unchanged system preamble every turn, prefix caching is a 10x fix and chunking is a 1.1x one; if your ITL is bad *without* any long prompts in flight, your batch is too large or your KV reads dominate and the answer is in [the memory math](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache), not the scheduler; if your TTFT is bad but your ITL is fine, you have a queueing problem and chunking will make it slightly worse, not better.

**Use vLLM instead of your own scheduler when** you need this to work correctly with prefix caching, speculative decoding, structured output, LoRA, tensor parallelism and CUDA graphs all at once. The eighty lines in this post are a correct implementation of one idea; a production engine is the composition of thirty such ideas, and the composition is where the hard bugs live. Write `nanoserve` to understand the knob. Ship vLLM and set the knob correctly.

---

## Key takeaways

1. **Prefill and decode are opposite workloads on one GPU.** Prefill runs at roughly 32,800 FLOPs per byte and decode at 26, against an A100 ridge point of 153. One is compute-bound, the other bandwidth-bound, and they must serialize.
2. **Derive your worst case, do not measure your average.** A 32k prefill on an A100 costs 5.18 s of exclusive GPU — 398 decode steps — and injects that gap into *every* concurrent stream. The formula is $T_p = (2NP + \tfrac{A}{2}P^2)/R$ and it needs only a config file and a datasheet.
3. **Schedule tokens, not requests.** Making the scheduling decision a `{request_id: num_tokens}` dictionary erases the prefill/decode distinction and makes chunking a one-line policy instead of an architecture.
4. **Total prefill FLOPs are chunk-size invariant.** Chunking creates no new work; it interleaves the same work with other people's decode steps. That invariance is why the trade-off is so favorable.
5. **The frontier is nearly free until it is not.** Budget 32768 to 256 costs 13% of TTFT and buys 104x on p99 ITL. Below a chunk of about 77 tokens on an A100 the weight read dominates and you pay TTFT for nothing.
6. **Pick the budget from the SLO, with your maximum context.** $B^\star = n_d + (I R - 2N n_d - A n_d S_d)/(2N + A P_{\max})$. It is hardware-specific; re-tune it on every fleet migration.
7. **Prefix caching cuts TTFT; chunked prefill cuts ITL.** They are orthogonal, they compose, and you want both — which is a reason to be cautious about disaggregation implementations that force prefix caching off.
8. **Disaggregation is the other answer, at fleet scale.** vLLM's MoRIIO write-up reports 2.5x goodput and 73/100 versus 30/100 requests meeting a TTFT-under-1s / ITL-under-50ms SLO, at the cost of higher TTFT, no prefix caching, and single-node operation.
9. **Alert on p99 ITL.** TPOT is an average and averages launder multi-second stalls into single-digit-millisecond regressions.
10. **A budget that cannot meet your SLO at your maximum context is telling you the truth.** Cap `max_model_len`, buy a bigger GPU, or split the pools — but do not tune a knob whose usable range is empty.

---

## Further reading

- [SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](https://arxiv.org/abs/2308.16369) — the origin of chunked prefills and decode-maximal batching.
- [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310) (OSDI '24) — stall-free batching, and capacity measured against latency SLOs.
- [Inside vLLM: Anatomy of a High-Throughput Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) — `long_prefill_token_threshold`, the scheduler's queue structure, and precise TTFT / ITL / TPOT definitions.
- [vLLM V1: A Major Upgrade to vLLM's Core Architecture](https://vllm.ai/blog/2025-01-27-v1-alpha-release) — the `{request_id: num_tokens}` scheduling decision that unifies prefill and decode.
- [MoRIIO KV connector: single-node prefill-decode disaggregation](https://vllm.ai/blog/2026-04-07-moriio-kv-connector) — the goodput numbers and the honest costs of splitting the pools.
- [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving](https://arxiv.org/abs/2401.09670) (OSDI '24) — the academic case for two pools.
- [Splitwise: Efficient Generative LLM Inference Using Phase Splitting](https://arxiv.org/abs/2311.18677) (ISCA '24) — why the two phases may want different hardware entirely.
- Within this series: [what inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is) for the layer map, [writing a continuous batching loop](/blog/machine-learning/inference-engineering/writing-a-continuous-batching-loop) for the `step()` this post extends, [the scheduler as a policy problem](/blog/machine-learning/inference-engineering/the-scheduler-as-a-policy-problem) for what to do when the budget is contested, and [the inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) for how this knob sits among all the others.

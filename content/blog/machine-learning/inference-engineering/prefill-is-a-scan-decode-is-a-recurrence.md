---
title: "Prefill is a scan, decode is a recurrence: the dual form of a linear-attention layer"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Derive why one recurrent layer needs two completely different kernels, prove the associativity that makes the parallel scan legal, tune the chunk size that decides which side of the roofline your prefill lands on, and see why hybrid decode starves a GPU unless you fill the batch."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "mamba",
    "state-space-models",
    "linear-attention",
    "hybrid-models",
    "batching",
    "pytorch",
    "gpu",
    "ml-systems",
    "throughput",
    "vllm",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 51
---

Here is a sentence that sounds like a bug report and is actually a specification: **the recurrent layer in your hybrid model needs two kernels that share no code, produce bit-comparable results, and differ in arithmetic intensity by a factor of 360.**

Not two kernels for two models. Two kernels for the *same layer*, the same weights, the same request — one that runs while the prompt is being consumed, and one that runs on every token after that. If you write only the recurrence, prefill becomes a Python loop 8,192 iterations deep and your time-to-first-token is measured in seconds. If you write only the scan, every decode step allocates a chunk-sized workspace to process a single token and you burn ten times the memory bandwidth you needed. Attention gets to blur this line — a prefill and a decode are both `scaled_dot_product_attention` with different sequence lengths, and vLLM's V1 scheduler famously erased the distinction entirely by representing every step as `{request_id: num_tokens}`. A recurrent layer will not let you blur it. The dispatch is on `L == 1` versus `L > 1`, and the two branches are genuinely different programs.

![Two columns comparing one recurrent layer running as a parallel chunked scan during prefill against the same layer running as a single in-place state update during decode](/imgs/blogs/prefill-is-a-scan-decode-is-a-recurrence-1.webp)

The figure above is the shape of the whole problem. On the left, prefill: an 8k prompt cut into 32 chunks of 256 tokens, 16.8 MFLOP of work per token per layer, 99.9% of it dense matrix multiplication landing on tensor cores, arithmetic intensity 452 FLOP per byte — comfortably compute-bound on an H100 whose bf16 ridge sits at 295. On the right, decode: one token, one in-place update of a fixed 2 MiB state, 5.24 MFLOP, arithmetic intensity 1.25. Same weights. Same math, even — the second is what the first computes, just evaluated one step at a time. The left side is a machine that wants to be fed a whole prompt at once. The right side is a machine so small that the cost of *launching* it dominates the cost of running it.

By the end of this post you will be able to derive both forms from a three-line proof that a gated linear recurrence is associative, explain exactly which four blocks of the chunked algorithm are matmuls and which one is not, pick a chunk size by solving for where your GPU's roofline ridge sits instead of copying a default, and predict — with arithmetic you can check — how a hybrid model's decode throughput scales with batch size and why it scales so much better than a dense transformer's. You will have written `nanoserve/recurrent.py`: both code paths, the dispatcher that chooses between them, and the equivalence test that catches the single most common hybrid-engine bug, which is a seam between the two paths at the prompt boundary.

Two promises carried over from [the series introduction](/blog/machine-learning/inference-engineering/what-inference-engineering-is), and they bind hard in a post this arithmetic-heavy. **I have no GPU and have run nothing.** Every number below is derived from arithmetic I show you, cited from a paper or vendor page with a link, or framed as a range you should reproduce yourself. Results tables carry a `Source` column. And **I do not name a model or an architecture I could not verify against a primary source** — several plausible candidates are absent for exactly that reason.

This post assumes you have read [the landscape post on hybrid models](/blog/machine-learning/inference-engineering/hybrid-models-and-the-end-of-the-kv-cache-assumption), which derives the *memory* consequence of a fixed-size state. This one derives the *compute* consequence. They are the same architectural fact seen from two sides, and the compute side is the one that decides your batching policy.

---

## 1. Two programs wearing one set of weights

Start with the layer, stated as plainly as it can be stated. A Mamba-2 style sequence mixer maintains, per attention-like head, a state matrix $h \in \mathbb{R}^{P \times N}$ — $P$ is the head dimension, $N$ is the state dimension. At each timestep it receives an input vector $x_t \in \mathbb{R}^{P}$, an input-selection vector $B_t \in \mathbb{R}^{N}$, an output-selection vector $C_t \in \mathbb{R}^{N}$, and a scalar decay $a_t \in (0,1]$. It does two things:

$$h_t \;=\; a_t\,h_{t-1} \;+\; x_t B_t^{\top}, \qquad y_t \;=\; h_t\,C_t$$

That is the entire layer. A decay-and-accumulate on a matrix, then a matrix-vector product to read out. Gated DeltaNet, gated linear attention, and the Kimi Delta Attention variant all differ in what $a_t$ is allowed to be — a scalar, a diagonal, a rank-one correction to the identity — but the shape of the computation is this, and everything in this post follows from the shape rather than the details.

Now notice the problem. The recurrence is *defined* sequentially: $h_t$ depends on $h_{t-1}$. For a 8,192-token prompt, a literal implementation is 8,192 dependent steps, each of which does about $5PN$ floating-point operations per head. On the reference model I will use throughout — NVIDIA's `Nemotron-H-8B-Base-8K`, whose config gives 128 heads, head dimension 64, and state dimension 128, all of which [the sibling post derives from the published config](/blog/machine-learning/inference-engineering/hybrid-models-and-the-end-of-the-kv-cache-assumption) — that is 5.24 MFLOP per token per layer. An H100 SXM is rated by [NVIDIA's datasheet](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c) at 989 dense bf16 TFLOP/s. Feeding it 5.24 MFLOP at a time, 8,192 times in a row, with a data dependency between every pair, uses a rounding error's worth of the machine.

The fix is not a faster kernel. The fix is a different algorithm — and the reason a different algorithm exists is a property of the recurrence itself.

![A layer input branching into a chunked scan path and a single step path that both write the same fixed state tensor before merging into the output projection](/imgs/blogs/prefill-is-a-scan-decode-is-a-recurrence-2.webp)

The figure above is the dispatch, drawn. One input tensor, one set of weights, one state object — and two paths through it that must agree. The scan path takes $L \gt 1$ tokens and produces $L$ outputs plus a final state. The step path takes one token and the current state, and produces one output plus a mutated state. Both write `state`, both feed `out_proj`. The engine picks between them on a single integer, and the correctness of your whole model rests on the two producing the same state tensor at the seam.

Two things follow immediately, and they are worth stating before the derivation because they are the practical payload:

1. **Your engine cannot flatten a mixed batch.** The trick continuous batching relies on — concatenate every request's tokens into one "super sequence" and use position offsets and masks to keep them isolated, as [vLLM's anatomy write-up describes](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) — works because attention is one kernel regardless of how many tokens each request contributes. A recurrent layer is not one kernel. Requests in prefill and requests in decode go through different code. You split the batch, run two kernels, and stitch the results.
2. **The two paths have different numerics.** The scan computes decays as differences of cumulative log-sums inside a chunk; the step multiplies by $a_t$ once per token. Mathematically identical, floating-point-wise not. If you get this wrong the model is perfect through the prompt and subtly wrong from the first generated token onward, which is the worst failure mode there is because it looks like a quality problem rather than a bug.

---

## 2. Why two forms exist at all: the recurrence is associative

The reason a sequential recurrence can be computed in parallel is not a trick. It is a theorem, and it is short enough to prove here.

Rewrite the update with the input term abbreviated:

$$h_t \;=\; a_t\,h_{t-1} \;+\; b_t, \qquad b_t \;=\; x_t B_t^{\top}$$

Each timestep is now a pair $(a_t, b_t)$ describing an affine map $h \mapsto a_t h + b_t$. Composing two such maps — apply the first, then the second — gives

$$h \;\mapsto\; a_2\,(a_1 h + b_1) + b_2 \;=\; (a_1 a_2)\,h + (a_2 b_1 + b_2)$$

so define the combine operator

$$(a_1, b_1) \bullet (a_2, b_2) \;=\; \bigl(a_1 a_2,\;\; a_2 b_1 + b_2\bigr)$$

**Claim: this operator is associative.** Expand both groupings.

$$\bigl[(a_1,b_1) \bullet (a_2,b_2)\bigr] \bullet (a_3,b_3) \;=\; (a_1 a_2,\; a_2 b_1 + b_2) \bullet (a_3, b_3) \;=\; \bigl(a_1 a_2 a_3,\;\; a_3 a_2 b_1 + a_3 b_2 + b_3\bigr)$$

$$(a_1,b_1) \bullet \bigl[(a_2,b_2) \bullet (a_3,b_3)\bigr] \;=\; (a_1, b_1) \bullet (a_2 a_3,\; a_3 b_2 + b_3) \;=\; \bigl(a_1 a_2 a_3,\;\; a_2 a_3 b_1 + a_3 b_2 + b_3\bigr)$$

The two right-hand sides are identical. $\blacksquare$

That is the whole license. Associativity means you may bracket the sequence however you like: fold the first half and the second half independently and combine, or fold in a balanced binary tree of depth $\log_2 L$, or — the option that actually matters on a GPU — fold contiguous chunks of $Q$ tokens in parallel and then run a short serial scan over the $L/Q$ chunk results. Sequential order is preserved; the *evaluation* order is free. This is exactly the guarantee that [PyTorch's `associative_scan` higher-order op](https://docs.pytorch.org/docs/main/higher_order_ops/associative_scan.html) demands of the `combine_fn` you hand it — the docs require that the function "must be pure, satisfy the associative property and have no side-effects."

### 2.1 From associativity to a closed form

Unroll the recurrence all the way down and the pattern is a decayed sum over history:

$$h_t \;=\; \sum_{s=1}^{t}\Bigl(\prod_{r=s+1}^{t} a_r\Bigr)\, b_s$$

Define the running product $A_t = \prod_{r=1}^{t} a_r$. Then $\prod_{r=s+1}^{t} a_r = A_t / A_s$, and

$$h_t \;=\; A_t \sum_{s=1}^{t} A_s^{-1} b_s$$

which turns a recurrence into a **cumulative sum** — the single most parallelizable primitive there is. Substituting $b_s = x_s B_s^{\top}$ and reading out with $C_t$:

$$y_t \;=\; h_t C_t \;=\; \sum_{s \le t} \frac{A_t}{A_s}\,\bigl(C_t \cdot B_s\bigr)\, x_s$$

Look at what that is. Define $M_{ts} = \frac{A_t}{A_s}\,(C_t \cdot B_s)$ for $s \le t$ and zero otherwise. Then $y = M x$ — a causally masked matrix applied to the value sequence. $C$ plays the role of queries, $B$ the role of keys, $x$ the role of values, and the decay ratio is a data-dependent positional mask. **The recurrence and a masked attention matrix are the same object.** This equivalence is the state-space duality that Tri Dao and Albert Gu formalized in [*Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*](https://arxiv.org/abs/2405.21060) (ICML 2024), and it is why the Mamba-2 layer can be executed with the same hardware primitives as attention.

There is a catch, and it is the reason chunking is not merely a performance choice. $A_t / A_s$ is a ratio of products of numbers in $(0,1]$. Over 8,192 tokens with a typical decay, $A_t$ underflows to zero in fp32 long before you reach the end of the prompt, and $A_t / A_s$ becomes $0/0$. The fix is to work in log space — store $\log a_t$, take a cumulative sum, and compute the decay as $\exp(\log A_t - \log A_s)$ — but even that only holds up if the difference is bounded, which means the cumulative sum must be **restricted to a window**. A chunk. This is precisely why the [flash-linear-attention library's gated delta rule kernels](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py) use a `chunk_local_cumsum` rather than a global one. Chunking is doing double duty: it is a parallelization strategy *and* a numerical stabilization strategy, and if you only think of it as the former you will pick a chunk size that quietly wrecks your logits at long context.

---

## 3. The chunked scan: four blocks, three of them matmuls

Now assemble the algorithm. Cut the sequence into $L/Q$ chunks of length $Q$. Within a chunk, index tokens $0 \ldots Q-1$. The output of token $q$ in chunk $c$ splits cleanly into two contributions:

- **What happened inside this chunk**, from tokens $0 \ldots q$ of chunk $c$.
- **What arrived at the chunk boundary**, summarized entirely by the incoming state $h_{c}^{\text{in}}$.

That split is the algorithm. It gives four blocks, and the [Mamba-2 authors' own walkthrough of the SSD algorithm](https://tridao.me/blog/2024/mamba2-part3-algorithm/) names them the same way:

1. **Intra-chunk outputs (the diagonal blocks).** Build the $Q \times Q$ masked matrix $M$ for this chunk from the local decays, and apply it. Two matmuls: $C B^{\top}$ to get the score matrix, then $M x$ to apply it. Pure tensor-core work.
2. **Chunk states.** Compute what this chunk *would* leave behind starting from a zero state: $h_c = \sum_q \text{decay}(Q{-}1 \leftarrow q)\, x_q B_q^{\top}$. One matmul per head.
3. **Inter-chunk recurrence.** Run the actual scan — but only over the $L/Q$ chunk states, not over $L$ tokens. This is the one genuinely serial piece, and it operates on a sequence that is $Q$ times shorter.
4. **Output conversion.** Add each chunk's incoming-state contribution: $C_q\, h_c^{\text{in}}$, decayed by the local cumulative product. One matmul per head.

The authors' own summary of the payoff is the sentence to remember: steps 1, 2 and 4 "leverage matmuls (and hence tensor cores), and can be computed completely in parallel," while only step 3 requires a scan, which "operates on a much shorter sequence and usually only takes a small fraction of the time of the full algorithm." Their stated motivation was blunt — "one of our primary goals with Mamba-2 is to leverage tensor cores to speed up the SSM" — on hardware where matmul FLOPs run up to 16 times faster than non-matmul FLOPs.

![A timeline of a chunked scan showing intra-chunk matmuls, chunk states, a short boundary scan, and the final state handed to the decode loop](/imgs/blogs/prefill-is-a-scan-decode-is-a-recurrence-3.webp)

The timeline above walks one 8k prompt through the algorithm at chunk size 256: 32 chunks whose intra-chunk work is entirely independent, 32 chunk states, a 32-element serial scan across the boundaries, and one final state that becomes the starting point for decoding. That last event is the one your engine cares about most — the handoff. Everything the 8,192 prompt tokens contributed is now compressed into a single 2 MiB tensor, and the KV blocks for the recurrent layers do not exist because there are none.

Let me put an exact number on "a small fraction." Per head, per token, the four blocks cost:

$$\underbrace{2Q(N+P)}_{\text{1: intra-chunk}} \;+\; \underbrace{2NP}_{\text{2: chunk state}} \;+\; \underbrace{2NP}_{\text{4: conversion}} \;+\; \underbrace{3NP/Q}_{\text{3: boundary scan}}$$

The first term grows with $Q$ because a longer chunk means a bigger $Q \times Q$ masked matrix — more of the work is done "quadratically" inside the chunk. The last term *shrinks* with $Q$ because the boundary scan runs once per chunk and gets amortized over $Q$ tokens. The two middle terms do not care.

![A four-part breakdown of one chunk's floating-point work showing three matmul blocks dominating and the serial boundary scan contributing a fraction of one percent](/imgs/blogs/prefill-is-a-scan-decode-is-a-recurrence-4.webp)

With the reference model's numbers — 128 heads, $P = 64$, $N = 128$ — and $Q = 256$, that breakdown is 12.58 MFLOP for the intra-chunk block, 2.10 for chunk states, 2.10 for the conversion, and 0.012 for the boundary scan: **99.93% of the arithmetic is matrix multiplication.** The one serial piece is seven hundredths of one percent of the work. That is the number that makes a recurrent prefill look, to the GPU, exactly like a transformer prefill.

Now watch the two forms run, one after the other, on the same request.

<figure class="blog-anim">
<svg viewBox="0 0 680 300" role="img" aria-label="Prefill lights four prompt chunks at once and then passes a boundary state left to right, after which decode collapses to one small state box that pulses and emits a single token per step" style="width:100%;height:auto;max-width:820px">
<style>
.p1-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.p1-sub{font:400 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.p1-cap{font:600 13px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.p1-cell{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.p1-hot{fill:var(--accent,#6366f1);opacity:.22;stroke:var(--accent,#6366f1);stroke-width:2}
.p1-baton{fill:var(--accent,#6366f1)}
.p1-state{fill:var(--accent,#6366f1);opacity:.85;transform-box:fill-box;transform-origin:center}
.p1-tok{fill:var(--accent,#6366f1)}
.p1-rule{stroke:var(--border,#d1d5db);stroke-width:1}
@keyframes p1-fadeA{0%,44%{opacity:1}50%,94%{opacity:0}100%{opacity:1}}
@keyframes p1-fadeB{0%,44%{opacity:0}50%,94%{opacity:1}100%{opacity:0}}
@keyframes p1-hot{0%,2%{opacity:0}7%,42%{opacity:.22}48%,100%{opacity:0}}
@keyframes p1-b1{0%,9%{opacity:0}13%,42%{opacity:1}48%,100%{opacity:0}}
@keyframes p1-b2{0%,17%{opacity:0}21%,42%{opacity:1}48%,100%{opacity:0}}
@keyframes p1-b3{0%,25%{opacity:0}29%,42%{opacity:1}48%,100%{opacity:0}}
@keyframes p1-pulse{0%,52%{transform:scale(1)}55%{transform:scale(1.14)}58%{transform:scale(1)}63%{transform:scale(1.14)}66%{transform:scale(1)}71%{transform:scale(1.14)}74%{transform:scale(1)}79%{transform:scale(1.14)}82%{transform:scale(1)}100%{transform:scale(1)}}
@keyframes p1-t1{0%,55%{opacity:0}59%,92%{opacity:1}100%{opacity:0}}
@keyframes p1-t2{0%,63%{opacity:0}67%,92%{opacity:1}100%{opacity:0}}
@keyframes p1-t3{0%,71%{opacity:0}75%,92%{opacity:1}100%{opacity:0}}
@keyframes p1-t4{0%,79%{opacity:0}83%,92%{opacity:1}100%{opacity:0}}
.p1-A{animation:p1-fadeA 14s ease-in-out infinite}
.p1-B{animation:p1-fadeB 14s ease-in-out infinite}
.p1-h{animation:p1-hot 14s ease-in-out infinite}
.p1-x1{animation:p1-b1 14s ease-in-out infinite}
.p1-x2{animation:p1-b2 14s ease-in-out infinite}
.p1-x3{animation:p1-b3 14s ease-in-out infinite}
.p1-pu{animation:p1-pulse 14s ease-in-out infinite}
.p1-y1{animation:p1-t1 14s ease-in-out infinite}
.p1-y2{animation:p1-t2 14s ease-in-out infinite}
.p1-y3{animation:p1-t3 14s ease-in-out infinite}
.p1-y4{animation:p1-t4 14s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.p1-A{animation:none;opacity:1}.p1-B{animation:none;opacity:0}.p1-h{animation:none;opacity:.22}.p1-x1,.p1-x2,.p1-x3{animation:none;opacity:1}.p1-pu{animation:none}.p1-y1,.p1-y2,.p1-y3,.p1-y4{animation:none;opacity:1}}
</style>
<g class="p1-A">
<text class="p1-lbl" x="30" y="42">prefill · chunked scan</text>
<text class="p1-sub" x="30" y="62">all four chunks compute at once; only the boundary state is serial</text>
<rect class="p1-cell" x="30" y="90" width="140" height="72" rx="8"/>
<rect class="p1-cell" x="196" y="90" width="140" height="72" rx="8"/>
<rect class="p1-cell" x="362" y="90" width="140" height="72" rx="8"/>
<rect class="p1-cell" x="528" y="90" width="122" height="72" rx="8"/>
<rect class="p1-hot p1-h" x="30" y="90" width="140" height="72" rx="8"/>
<rect class="p1-hot p1-h" x="196" y="90" width="140" height="72" rx="8"/>
<rect class="p1-hot p1-h" x="362" y="90" width="140" height="72" rx="8"/>
<rect class="p1-hot p1-h" x="528" y="90" width="122" height="72" rx="8"/>
<text class="p1-cap" x="100" y="122">chunk 0</text>
<text class="p1-cap" x="266" y="122">chunk 1</text>
<text class="p1-cap" x="432" y="122">chunk 2</text>
<text class="p1-cap" x="589" y="122">chunk 3</text>
<text class="p1-cap" x="100" y="146">256 tok</text>
<text class="p1-cap" x="266" y="146">256 tok</text>
<text class="p1-cap" x="432" y="146">256 tok</text>
<text class="p1-cap" x="589" y="146">256 tok</text>
<line class="p1-rule" x1="30" y1="200" x2="650" y2="200"/>
<circle class="p1-baton p1-x1" cx="183" cy="200" r="10"/>
<circle class="p1-baton p1-x2" cx="349" cy="200" r="10"/>
<circle class="p1-baton p1-x3" cx="515" cy="200" r="10"/>
<text class="p1-sub" x="30" y="232">boundary scan carries one 2.00 MiB state across 3 seams</text>
<text class="p1-sub" x="30" y="256">16.8 MFLOP per token · 99.9% matmul · AI 452 FLOP per byte</text>
</g>
<g class="p1-B">
<text class="p1-lbl" x="30" y="42">decode · single-step recurrence</text>
<text class="p1-sub" x="30" y="62">the same state is rewritten in place, once per generated token</text>
<rect class="p1-cell" x="30" y="90" width="150" height="72" rx="8"/>
<rect class="p1-state p1-pu" x="54" y="106" width="102" height="40" rx="6"/>
<text class="p1-cap" x="105" y="186">state 2.00 MiB</text>
<line class="p1-rule" x1="196" y1="126" x2="650" y2="126"/>
<circle class="p1-tok p1-y1" cx="250" cy="126" r="11"/>
<circle class="p1-tok p1-y2" cx="360" cy="126" r="11"/>
<circle class="p1-tok p1-y3" cx="470" cy="126" r="11"/>
<circle class="p1-tok p1-y4" cx="580" cy="126" r="11"/>
<text class="p1-cap" x="250" y="166">tok 1</text>
<text class="p1-cap" x="360" y="166">tok 2</text>
<text class="p1-cap" x="470" y="166">tok 3</text>
<text class="p1-cap" x="580" y="166">tok 4</text>
<text class="p1-sub" x="30" y="232">no cache grows; nothing is appended; the state is overwritten</text>
<text class="p1-sub" x="30" y="256">5.24 MFLOP per token · 3 tiny kernels · AI 1.25 FLOP per byte</text>
</g>
</svg>
<figcaption>Prefill lights every chunk simultaneously and threads a single state through the boundaries; decode collapses that machinery into one in-place rewrite per token.</figcaption>
</figure>

The motion is the argument. During prefill, the four chunk boxes brighten *together* — that is the parallelism, and it is what a still frame cannot show. Only the three boundary batons appear in order, because only they are serial. Then the whole apparatus folds down to one small box that pulses once per token. Same layer, same weights, and the second machine is roughly a thousandth the size of the first.

---

## 4. The chunk-size knob, derived instead of copied

Chunk size $Q$ is presented in most codebases as a magic constant — `block_len=64` in the Mamba-2 reference implementation, chunk size 64 in the flash-linear-attention gated delta rule kernels, `chunk_size=256` in a number of model configs. It is not magic. It is the solution to a small optimization problem you can write down.

The cost per token per head, from section 3:

$$F(Q) \;=\; 2Q(N+P) \;+\; 4NP \;+\; \frac{3NP}{Q}$$

Compare that to the sequential recurrence, which costs $5PN$ per token per head and cannot use tensor cores. The chunked form does **more** arithmetic — that is the trade. With $P = 64$ and $N = 128$, $5PN = 40{,}960$ FLOPs per token per head, and:

| Chunk size Q | FLOPs per token per layer | vs. pure recurrence | Chunks in an 8k prompt | Mask memory, 8k prompt | AI, fused kernel | Source |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 5.87 MFLOP | 1.12x | 256 serial steps | 67 MB | 158 | derived |
| 64 | 7.39 MFLOP | 1.41x | 128 serial steps | 134 MB | 199 | derived |
| 128 | 10.51 MFLOP | 2.00x | 64 serial steps | 268 MB | 283 | derived |
| 256 | 16.79 MFLOP | 3.20x | 32 serial steps | 537 MB | 452 | derived |
| 512 | 29.37 MFLOP | 5.60x | 16 serial steps | 1,074 MB | 791 | derived |

![A five-row comparison of chunk sizes against floating-point cost, redundancy versus the pure recurrence, serial depth, and arithmetic intensity](/imgs/blogs/prefill-is-a-scan-decode-is-a-recurrence-5.webp)

Read the table with the figure and the trade is unmissable. Every doubling of $Q$ roughly doubles the intra-chunk term, halves the serial depth, and doubles the memory the naive implementation needs for its masked score matrices. The question is which of those costs your hardware actually charges you for.

**The arithmetic-intensity column is the one that decides it.** The scan reads, per token, the projected inputs it consumes and writes the outputs it produces. For the reference layer that is $x$ (8,192 elements), $B$ and $C$ (1,024 each), and the timestep vector (128), for 10,368 elements in; and $y$ (8,192 elements) out. At bf16 that is $18{,}560 \times 2 = 37{,}120$ bytes per token — assuming a fused kernel where the intermediate score matrices never leave on-chip memory, which is exactly the assumption [FlashAttention-style kernel fusion](/blog/machine-learning/high-performance-computing/kernel-fusion-and-flashattention-beating-the-memory-wall) exists to make true. Arithmetic intensity is then $F(Q) \cdot 128 / 37{,}120$.

An H100 SXM's ridge point — the intensity above which you are compute-bound rather than bandwidth-bound, as [the roofline post](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) develops — is 989 TFLOP/s divided by 3.35 TB/s, or 295 FLOP per byte. Setting the intensity equal to that and solving:

$$\frac{128}{37{,}120}\Bigl(384Q + 32{,}768 + \frac{24{,}576}{Q}\Bigr) = 295 \quad\Longrightarrow\quad Q \approx 138$$

**That is the answer to "what chunk size should I use", and it came out of the hardware spec rather than a config file.** Below roughly 138 tokens per chunk, a Mamba-2 prefill on an H100 is memory-bound: you are paying for bandwidth you cannot fill with arithmetic, and making the chunk bigger is free. Above it you are compute-bound and every extra FLOP is an extra FLOP. So the useful range is "the smallest chunk that clears the ridge, plus enough headroom for the matmul shapes to be efficient" — which lands you at 128 or 256, which is exactly where the ecosystem sits. The defaults are not arbitrary; they are this calculation, done on hardware of the same generation.

Two corrections to that clean story, both of which matter in practice:

**Matmul shape, not just FLOP count.** A $64 \times 64$ intra-chunk matmul is small enough that tensor-core utilization suffers — the fixed cost of staging tiles into shared memory is amortized over very little work. A $256 \times 256$ matmul is a comfortable shape. So the achieved fraction of peak rises with $Q$ even as the FLOP count rises, and the two effects partially cancel. This is why a sweep over $Q$ on real hardware usually shows a broad, flat optimum somewhere in the 64-to-256 range rather than a sharp peak.

**Memory, in the unfused case.** The mask-memory column is the killer for a pure-PyTorch implementation. Materializing the per-chunk score matrices for a whole prompt at once costs $b \cdot H \cdot L \cdot Q$ elements — linear in $Q$, and with 128 heads and an 8k prompt that is 537 MB at $Q = 256$ in bf16, or over a gigabyte if you compute the exponentials in fp32 (which you should). A fused kernel never materializes them; a vectorized PyTorch reference does. If you are writing the reference, either loop over chunks or keep $Q$ small, and understand that you are trading throughput for the ability to run at all.

---

## 5. Decode is a recurrence, and it is very, very small

Now the other side. At decode $L = 1$, there is nothing to chunk, and the layer does exactly what its definition says: scale the state by $a_t$, add an outer product, contract with $C_t$. Three operations on a fixed-size tensor.

Count the work per head:

- Scale: $PN$ multiplies.
- Outer product and accumulate: $PN$ multiplies plus $PN$ adds.
- Readout $h_t C_t$: $PN$ multiplies plus $PN$ adds.

Total $5PN$ FLOPs per head, or ${5 H_m P N}$ per layer. For the reference layer: $5 \times 128 \times 64 \times 128 = 5{,}242{,}880$ FLOPs, call it 5.24 MFLOP.

Count the bytes. The state must be read and written: $2 \times H_m P N \times b_s$, which at bf16 is $2 \times 128 \times 64 \times 128 \times 2 = 4{,}194{,}304$ bytes, exactly 4.00 MiB.

$$\text{AI}_{\text{core}} \;=\; \frac{5.24 \times 10^6}{4.19 \times 10^6} \;=\; 1.25 \text{ FLOP per byte}$$

Against a ridge of 295, that is not memory-bound so much as it is *barely present*. On paper, 4.19 MB at 3.35 TB/s takes 1.25 microseconds. A kernel launch costs a few microseconds. **The recurrence, as a standalone kernel, takes less time to run than to launch.**

### 5.1 The layer is not the recurrence

Here is the correction that keeps this honest, and it is the single most common mistake people make when reasoning about SSM decode cost: the recurrence is not the layer. The layer also has projections, and they are large.

For the reference Mamba-2 block at hidden size 4,096 and expansion 2, the input projection produces the gate $z$ (8,192), the convolved branch $x$ (8,192), the group-shared $B$ and $C$ (1,024 each), and the per-head timestep (128) — 18,560 outputs from 4,096 inputs, so $4{,}096 \times 18{,}560 = 76{,}021{,}760$ parameters. The output projection is $8{,}192 \times 4{,}096 = 33{,}554{,}432$. Together 109,576,192 parameters, which at bf16 is **219 MB of weights that must be read from HBM on every single decode step.**

So the honest per-layer decode accounting at batch 1 is:

| Component | FLOPs per token | Bytes moved | Share of bytes | Source |
| --- | --- | --- | --- | --- |
| in_proj + out_proj GEMV | 219.2 MFLOP | 219.2 MB | 98.1% | derived |
| conv state + step | 0.1 MFLOP | 0.06 MB | 0.03% | derived |
| SSM recurrence core | 5.24 MFLOP | 4.19 MB | 1.9% | derived |
| **Layer total** | **224.4 MFLOP** | **223.4 MB** | **100%** | derived |

The recurrence is 2.3% of the layer's arithmetic and 1.9% of its traffic. At batch 1 it is a rounding error, and anyone who tells you "SSM decode is O(1) so it's free" is describing the 2%. What is actually happening at batch 1 is what happens in every decode step in this series: [you are reading weights and generating one token](/blog/machine-learning/inference-engineering/the-naive-decode-loop-and-your-first-baseline), and the sequence mixer barely registers.

**But the two terms scale differently, and that is the entire post.** Weights are read once per step no matter how many requests are in the batch. State is read once per *request*. So:

$$\text{AI}_{\text{layer}}(B) \;=\; \frac{B \cdot F_{\ell}}{W_{\ell} + B \cdot \sigma}, \qquad F_{\ell} = 224.4\,\text{MFLOP},\; W_{\ell} = 219.2\,\text{MB},\; \sigma = 4.19\,\text{MB}$$

which rises with $B$ and saturates at $F_{\ell}/\sigma = 53.5$ FLOP per byte. The crossover where state traffic equals weight traffic is $B = W_{\ell}/\sigma = 52.3$ — **at batch 52, a hybrid model's recurrent states cost as much bandwidth as its weights.**

#### Worked example: one recurrent layer versus one attention layer at 32k context

Take the same decode step and run it through a GQA attention layer of a Llama-3.1-8B shape: 32 query heads, 8 key-value heads, head dimension 128, hidden 4,096. Per token per layer:

- **KV bytes read**: $2 \times 8 \times 128 \times 2 = 4{,}096$ bytes per token of context. At $S = 32{,}768$: 134.2 MB, per request.
- **Attention FLOPs**: $2 H_q S d$ for the score and the same for the weighted sum, so $4 \times 32 \times 32{,}768 \times 128 = 536.9$ MFLOP.
- **Projection weights**: $q$ and $o$ at $4{,}096^2$ each, $k$ and $v$ at $4{,}096 \times 1{,}024$ each — 41,943,040 parameters, 83.9 MB at bf16, 83.9 MFLOP of GEMV.

| Quantity, one layer, S = 32768, batch 1 | Mamba-2 layer | GQA attention layer | Ratio | Source |
| --- | --- | --- | --- | --- |
| Sequence-mixer FLOPs per token | 5.24 MFLOP | 536.9 MFLOP | 102x | derived |
| Sequence-mixer bytes per token | 4.19 MB | 134.2 MB | 32x | derived |
| Weight bytes per step | 219.2 MB | 83.9 MB | 0.38x | derived |
| Total layer bytes at batch 1 | 223.4 MB | 218.1 MB | 0.98x | derived |
| Arithmetic intensity at batch 1 | 1.00 | 2.85 | — | derived |
| Arithmetic intensity at batch 32 | 20.3 | 4.54 | — | derived |
| Arithmetic intensity ceiling | 53.5 | 4.63 | 11.6x | derived |

Three readings of that table, in increasing order of usefulness.

**First, the obvious one.** The attention layer does 102 times more sequence-mixing arithmetic and moves 32 times more sequence-mixing bytes at 32k. The recurrent layer is, on that axis, essentially free.

**Second, the surprising one.** At batch 1 the two layers move *almost the same total bytes* — 223 MB versus 218 MB — and the recurrent layer is slightly worse. Its projections are 2.6 times bigger, because Mamba-2 expands the hidden dimension by 2 and emits five tensors from one matmul. At batch 1, on a 32k context, a hybrid saves you nothing per layer. It saves you at 128k, and it saves you at batch 32, and those are different arguments.

**Third, the one that determines your engine's design.** The ceilings. No matter how large you make the batch, the attention layer's arithmetic intensity converges to 4.63 — because its bytes scale with batch *and* context, exactly in step with its FLOPs. The recurrent layer converges to 53.5, an order of magnitude higher, because its per-request bytes are a constant that batching amortizes weight traffic against. **Neither reaches the H100's ridge of 295. Both are memory-bound at decode, forever.** But one of them has 11.6 times more headroom, and you only collect that headroom by filling the batch.

---

## 6. Writing both paths: `nanoserve/recurrent.py`

Enough derivation. Here is the code.

### 6.1 The state object

The state is the thing both paths share, so define it first. It has two parts: the SSM state and the causal convolution's rolling window, which is easy to forget and produces spectacularly wrong output when you do.

```python
# nanoserve/recurrent.py
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class RecurrentConfig:
    d_model: int = 4096
    expand: int = 2
    n_heads: int = 128         # mamba_num_heads
    head_dim: int = 64         # mamba_head_dim  (n_heads * head_dim == d_inner)
    state_dim: int = 128       # ssm_state_size
    n_groups: int = 8
    conv_kernel: int = 4
    chunk_size: int = 256

    @property
    def d_inner(self) -> int:
        return self.d_model * self.expand

    @property
    def conv_width(self) -> int:
        # the depthwise conv covers x plus the group-shared B and C
        return self.d_inner + 2 * self.n_groups * self.state_dim


@dataclass
class RecurrentState:
    """Per-request state for one recurrent layer. Fixed size, forever."""

    ssm: torch.Tensor    # (B, H, P, N)
    conv: torch.Tensor   # (B, conv_width, K - 1)

    @classmethod
    def alloc(cls, batch: int, cfg: RecurrentConfig, device, dtype=torch.float32):
        return cls(
            ssm=torch.zeros(batch, cfg.n_heads, cfg.head_dim, cfg.state_dim,
                            device=device, dtype=dtype),
            conv=torch.zeros(batch, cfg.conv_width, cfg.conv_kernel - 1,
                             device=device, dtype=dtype),
        )

    def nbytes(self) -> int:
        return (self.ssm.numel() * self.ssm.element_size()
                + self.conv.numel() * self.conv.element_size())


cfg = RecurrentConfig()
st = RecurrentState.alloc(1, cfg, device="cpu")
print(f"ssm  {tuple(st.ssm.shape)}  {st.ssm.numel() * 4 / 2**20:.2f} MiB fp32")
print(f"conv {tuple(st.conv.shape)} {st.conv.numel() * 4 / 2**20:.3f} MiB fp32")
print(f"total per request per layer: {st.nbytes() / 2**20:.2f} MiB")
```

```console
ssm  (1, 128, 64, 128)  4.00 MiB fp32
conv (1, 10240, 3) 0.117 MiB fp32
total per request per layer: 4.12 MiB
```

Note the dtype default. I allocate fp32 even for a bf16 model, because the state accumulates over thousands of steps and bf16 has 8 mantissa bits. That doubles the intercept in the memory law — 4.00 MiB per layer instead of 2.00 — and it is a real cost you should measure rather than assume. The sibling post on [the two-cache engine](/blog/machine-learning/inference-engineering/implementing-a-two-cache-engine-kv-blocks-plus-recurrent-state) is where this allocation gets its own pool and lifetime.

### 6.2 The decode path: one step, in place

The step is short enough to read in one sitting, which is the point.

```python
# nanoserve/recurrent.py  (continued)

def conv_step(x_t: torch.Tensor, conv_state: torch.Tensor,
              weight: torch.Tensor, bias=None) -> torch.Tensor:
    """One timestep of a causal depthwise conv, using a rolling window.

    x_t        : (B, C)          this timestep's input
    conv_state : (B, C, K-1)     the K-1 previous inputs, mutated in place
    weight     : (C, K)          depthwise kernel
    """
    # window = [older ... newer, x_t]
    window = torch.cat([conv_state, x_t.unsqueeze(-1)], dim=-1)   # (B, C, K)
    y = (window * weight).sum(dim=-1)                             # (B, C)
    conv_state.copy_(window[..., 1:])                             # slide by one
    return y if bias is None else y + bias


def ssm_step(x_t, a_t, B_t, C_t, state: torch.Tensor) -> torch.Tensor:
    """The recurrence, one token, state mutated in place.

    x_t   : (B, H, P)      per-head input
    a_t   : (B, H)         per-head scalar decay in (0, 1]
    B_t   : (B, H, N)      input selection
    C_t   : (B, H, N)      output selection
    state : (B, H, P, N)   mutated
    """
    state.mul_(a_t[..., None, None])                      # h <- a * h
    state.addcmul_(x_t[..., :, None], B_t[..., None, :])  # h <- h + x B^T
    return torch.einsum("bhpn,bhn->bhp", state, C_t)      # y = h C
```

Three lines of arithmetic for the recurrence: one `mul_`, one `addcmul_`, one small contraction. Both state updates are in place, so the state tensor is allocated once per request at admission and never reallocated — which is what makes it eligible for a CUDA-graph capture, and what makes it *ineligible* for the copy-on-write tricks that paged KV blocks enjoy.

`addcmul_` is doing the outer product by broadcasting: `x_t[..., :, None]` is $(B, H, P, 1)$ and `B_t[..., None, :]` is $(B, H, 1, N)$, so the product broadcasts to the state's shape without ever materializing an intermediate. That matters here — an explicit `torch.einsum("bhp,bhn->bhpn", ...)` followed by an add would allocate a full second copy of the state on every step, doubling the traffic of the one component whose traffic we are trying to control.

### 6.3 The prefill path: the chunked scan

The scan needs one helper: a segmented sum that produces, for every pair of positions in a chunk, the log of the decay between them.

```python
# nanoserve/recurrent.py  (continued)

def segsum(log_a: torch.Tensor) -> torch.Tensor:
    """Pairwise segment sums of log-decays, in log space.

    log_a : (..., Q)  log of the per-step decay
    returns (..., Q, Q) where out[..., i, j] = sum_{k=j+1..i} log_a[..., k]
    for j <= i, and -inf above the diagonal.
    """
    cum = torch.cumsum(log_a, dim=-1)                       # (..., Q)
    seg = cum.unsqueeze(-1) - cum.unsqueeze(-2)             # out[i, j] = cum_i - cum_j
    Q = log_a.size(-1)
    causal = torch.ones(Q, Q, dtype=torch.bool, device=log_a.device).tril()
    return seg.masked_fill(~causal, float("-inf"))
```

The subtraction-of-cumulative-sums is the log-space form of the $A_t / A_s$ ratio from section 2.1. Because `cum` is a cumulative sum *within a chunk*, its magnitude is bounded by $Q \cdot |\log a|_{\max}$ rather than by the whole sequence length — which is the numerical stabilization I promised, and the reason chunk size shows up in an accuracy discussion and not only a speed one.

Now the scan itself, in four blocks that map one-to-one onto section 3.

```python
# nanoserve/recurrent.py  (continued)

def ssm_scan(x, log_a, B, C, chunk: int, h0=None):
    """Chunked parallel scan. The prefill path.

    x     : (b, L, H, P)
    log_a : (b, L, H)      log of per-step decay
    B, C  : (b, L, H, N)
    h0    : (b, H, P, N) or None
    returns y (b, L, H, P) and h_final (b, H, P, N)
    """
    b, L, H, P = x.shape
    N = B.shape[-1]
    assert L % chunk == 0, "pad or trim the tail before calling"
    nc = L // chunk

    def split(t):                       # (b, L, ...) -> (b, nc, Q, ...)
        return t.reshape(b, nc, chunk, *t.shape[2:])

    x, B, C = split(x), split(B), split(C)
    la = split(log_a).permute(0, 3, 1, 2)          # (b, H, nc, Q)
    cum = la.cumsum(dim=-1)                        # (b, H, nc, Q)

    # 1. intra-chunk outputs: the masked (C B^T) matrix applied to x
    mask = torch.exp(segsum(la))                                   # (b,H,nc,Q,Q)
    y_diag = torch.einsum("bcqhn,bckhn,bhcqk,bckhp->bcqhp", C, B, mask, x)

    # 2. chunk states, computed from a zero initial state
    decay_to_end = torch.exp(cum[..., -1:] - cum)                  # (b,H,nc,Q)
    states = torch.einsum("bckhn,bhck,bckhp->bchpn", B, decay_to_end, x)

    # 3. the only serial part: a scan over nc chunk states
    if h0 is None:
        h0 = torch.zeros_like(states[:, :1])
    else:
        h0 = h0.unsqueeze(1)
    states = torch.cat([h0, states], dim=1)                        # (b,nc+1,H,P,N)
    chunk_la = F.pad(cum[..., -1], (1, 0))                         # (b,H,nc+1)
    carry = torch.exp(segsum(chunk_la))                            # (b,H,nc+1,nc+1)
    states = torch.einsum("bhzc,bchpn->bzhpn", carry, states)
    states, h_final = states[:, :-1], states[:, -1]

    # 4. add each chunk's incoming-state contribution
    decay_from_start = torch.exp(cum)                              # (b,H,nc,Q)
    y_off = torch.einsum("bcqhn,bchpn,bhcq->bcqhp", C, states, decay_from_start)

    return (y_diag + y_off).reshape(b, L, H, P), h_final
```

Every `einsum` in there is a batched matmul, and step 3 — the `carry` contraction — is the serial scan, expressed as a tiny lower-triangular matrix over `nc + 1` chunk boundaries. On a 32-chunk prompt that is a 33-by-33 matrix. This is what "0.07% of the FLOPs" looks like in code.

### 6.4 The test that catches the seam bug

Now the most valuable twenty lines in this post. The two paths must agree, and the only way to know they do is to assert it.

```python
# nanoserve/tests/test_recurrent_equivalence.py
import torch
from nanoserve.recurrent import ssm_scan, ssm_step


def test_scan_matches_step(L=512, chunk=64, b=2, H=4, P=16, N=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    kw = dict(generator=g, dtype=torch.float64)          # fp64: isolate algebra from precision
    x = torch.randn(b, L, H, P, **kw)
    B = torch.randn(b, L, H, N, **kw)
    C = torch.randn(b, L, H, N, **kw)
    log_a = -torch.rand(b, L, H, **kw) * 0.1             # decays in (0.90, 1.0]

    y_scan, h_scan = ssm_scan(x, log_a, B, C, chunk=chunk)

    h = torch.zeros(b, H, P, N, dtype=torch.float64)
    y_step = torch.empty_like(y_scan)
    for t in range(L):
        y_step[:, t] = ssm_step(x[:, t], log_a[:, t].exp(), B[:, t], C[:, t], h)

    assert torch.allclose(y_scan, y_step, atol=1e-9), \
        f"outputs diverge, max |diff| = {(y_scan - y_step).abs().max():.3e}"
    assert torch.allclose(h_scan, h, atol=1e-9), \
        f"final state diverges, max |diff| = {(h_scan - h).abs().max():.3e}"


def test_chunk_size_is_irrelevant(L=512):
    """Chunking is an evaluation-order choice. It must not change the answer."""
    ref = None
    for chunk in (8, 16, 32, 64, 128, 256, 512):
        torch.manual_seed(7)
        ...  # build the same inputs
        y, h = ssm_scan(x, log_a, B, C, chunk=chunk)
        if ref is None:
            ref = (y, h)
        else:
            assert torch.allclose(y, ref[0], atol=1e-9)
            assert torch.allclose(h, ref[1], atol=1e-9)
```

Run it in fp64 first. In fp64 the two paths are algebraically identical and the difference should sit at machine epsilon — if it does not, you have a real bug, not a precision problem. Only then drop to bf16, where you should expect a relative difference on the order of $10^{-2}$ to $10^{-3}$ and where the right assertion is a relative-error bound, not `allclose`. The second test is the one people skip and should not: chunk size is an *evaluation order*, so changing it must not change the answer. If your output shifts when you change `chunk`, you have an off-by-one in the mask, a decay applied at the wrong boundary, or a state captured one token early.

### 6.5 The dispatcher

With both paths written, the layer is a router.

```python
# nanoserve/recurrent.py  (continued)

class RecurrentLayer(torch.nn.Module):
    def __init__(self, cfg: RecurrentConfig, weights):
        super().__init__()
        self.cfg = cfg
        self.in_proj = weights.in_proj      # (d_model -> 2*d_inner + 2*G*N + H)
        self.conv_w = weights.conv_weight   # (conv_width, K)
        self.out_proj = weights.out_proj    # (d_inner -> d_model)
        self.A_log = weights.A_log          # (H,)
        self.dt_bias = weights.dt_bias      # (H,)

    def forward(self, hidden, state: RecurrentState, *, slot):
        """hidden: (b, L, d_model). slot: LongTensor of state rows for this batch."""
        b, L, _ = hidden.shape
        z, xbc, dt = self._project(hidden)
        dt = F.softplus(dt + self.dt_bias)                   # (b, L, H)
        log_a = -dt * torch.exp(self.A_log)                  # (b, L, H), <= 0

        if L == 1:
            xbc = conv_step(xbc[:, 0], state.conv[slot], self.conv_w)
            x, Bt, Ct = self._split(xbc.unsqueeze(1))
            y = ssm_step(x[:, 0] * dt[:, 0, :, None], log_a[:, 0].exp(),
                         Bt[:, 0], Ct[:, 0], state.ssm[slot])
            y = y.unsqueeze(1)
        else:
            xbc, new_conv = self._conv_prefill(xbc)
            state.conv[slot] = new_conv
            x, Bt, Ct = self._split(xbc)
            y, h = ssm_scan(x * dt[..., None], log_a, Bt, Ct,
                            chunk=self.cfg.chunk_size,
                            h0=state.ssm[slot])
            state.ssm[slot] = h

        y = self._gate_and_norm(y, z)
        return self.out_proj(y.reshape(b, L, self.cfg.d_inner))
```

Three details in there are load-bearing and easy to get wrong.

**`h0=state.ssm[slot]`, not zeros.** The scan must accept an incoming state. If it does not, chunked prefill breaks: when the scheduler splits a 32k prompt across four steps of 8k each — the technique developed in [the chunked-prefill post](/blog/machine-learning/inference-engineering/chunked-prefill-and-the-ttft-tpot-tradeoff) — the second call must continue where the first left off. A KV cache gets this for free, because the blocks from step one are simply still there. A recurrent state does not; it is a value you must thread through explicitly.

**`slot` is an index, not a slice.** Requests are not contiguous in the state pool. The batch is a gather of arbitrary rows, so the state tensor is indexed rather than sliced. That gather is another difference from the KV path, where the block table already handles the indirection.

**The conv branch has its own two paths.** During prefill it is an `F.conv1d` with left padding; during decode it is the rolling window from section 6.2. Both must leave the same $K-1$ trailing inputs in `state.conv`.

### 6.6 The cost model, so you can tune without a GPU

The chunk-size table in section 4 is pure arithmetic, so ship it as a script rather than a claim.

```python
# nanoserve/tools/scan_cost.py
"""Derived cost model for a chunked scan. No GPU required, no measurement claimed."""

H, P, N = 128, 64, 128           # Nemotron-H-8B Mamba-2 layer, from its config
BYTES_PER_TOKEN = 37_120         # x + B + C + dt in, y out, bf16
H100_RIDGE = 989e12 / 3.35e12    # dense bf16 TFLOP/s over HBM3 TB/s


def scan_flops_per_token(Q, H=H, P=P, N=N):
    intra = 2 * Q * (N + P)      # block 1: the masked Q x Q matmul
    chunk_state = 2 * N * P      # block 2
    convert = 2 * N * P          # block 4
    boundary = 3 * N * P / Q     # block 3: the only serial part, amortized
    return H * (intra + chunk_state + convert + boundary)


def step_flops_per_token(H=H, P=P, N=N):
    return 5 * H * P * N         # scale + outer product + readout


base = step_flops_per_token()
print(f"ridge point: {H100_RIDGE:.0f} FLOP/byte    "
      f"pure recurrence: {base/1e6:.2f} MFLOP/token\n")
print(f"{'Q':>5} {'MFLOP/tok':>10} {'vs step':>8} {'chunks@8k':>10} "
      f"{'AI':>7}  {'roofline':>13}")
for Q in (32, 64, 128, 256, 512):
    f = scan_flops_per_token(Q)
    ai = f / BYTES_PER_TOKEN
    side = "compute-bound" if ai > H100_RIDGE else "memory-bound"
    print(f"{Q:>5} {f/1e6:>10.2f} {f/base:>7.2f}x {8192//Q:>10} {ai:>7.0f}  {side:>13}")
```

```console
ridge point: 295 FLOP/byte    pure recurrence: 5.24 MFLOP/token

    Q  MFLOP/tok  vs step  chunks@8k      AI       roofline
   32       5.87    1.12x        256     158   memory-bound
   64       7.39    1.41x        128     199   memory-bound
  128      10.51    2.00x         64     283   memory-bound
  256      16.79    3.20x         32     452  compute-bound
  512      29.37    5.60x         16     791  compute-bound
```

Change the three constants at the top to your model's config and your GPU's specs, and the table retunes. On an L4 — 121 dense bf16 TFLOP/s against 300 GB/s of bandwidth, per NVIDIA's L4 datasheet — the ridge is around 403, which pushes the useful chunk size *up*, not down. That is a genuinely counterintuitive result worth sitting with: the weaker the GPU's bandwidth relative to its math, the larger the chunk should be.

### 6.7 The batched step, which is where the throughput lives

The step in section 6.2 already takes a batch dimension. What it does not yet do is handle the fact that in a continuous-batching engine, prefilling and decoding requests arrive in the same scheduler step.

```python
# nanoserve/engine/hybrid_batch.py

def run_recurrent_layer(layer, batch, state: RecurrentState):
    """Split a mixed scheduler step into the two paths, run both, restitch."""
    prefill = [r for r in batch if r.num_new_tokens > 1]
    decode = [r for r in batch if r.num_new_tokens == 1]
    out = {}

    # decode group: one kernel over every stepping request at once
    if decode:
        slots = torch.tensor([r.state_slot for r in decode], device=state.ssm.device)
        h = torch.stack([r.hidden for r in decode])          # (Bd, 1, d_model)
        out.update(zip((r.id for r in decode),
                       layer(h, state, slot=slots).unbind(0)))

    # prefill group: one call per request, because lengths differ
    for r in prefill:
        slot = torch.tensor([r.state_slot], device=state.ssm.device)
        out[r.id] = layer(r.hidden.unsqueeze(0), state, slot=slot)[0]

    return out
```

That loop over prefill requests is the honest version of what a toy engine does, and it is also the exact thing production kernels remove. A varlen scan kernel takes a `cu_seqlens` offset tensor and processes all prefilling requests in one launch, with chunk boundaries snapped to request boundaries so no chunk ever spans two prompts. Writing that kernel is [the next post in this track](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook)'s neighbor, post 62, and it is where the fused version — gates, normalization, conv and state write in one launch — actually lives.

---

## 7. What this does to your batching strategy

![One layer's weight traffic shared across a batch while per-request state traffic accumulates, lifting arithmetic intensity from one to twenty](/imgs/blogs/prefill-is-a-scan-decode-is-a-recurrence-6.webp)

Put the whole model together and the batching consequence becomes a number you can act on. The decode step's lower bound is total bytes over bandwidth — the law from [the naive-decode-loop post](/blog/machine-learning/inference-engineering/the-naive-decode-loop-and-your-first-baseline), unchanged. What changes is what goes into "total bytes."

For a hybrid at batch $B$ and context $S$:

$$\text{bytes}(B, S) \;=\; \underbrace{W}_{\text{weights}} \;+\; B\cdot\Bigl(\underbrace{\Sigma}_{\text{state, flat}} \;+\; \underbrace{S \cdot \kappa}_{\text{KV, per token}}\Bigr)$$

For the reference model: $W \approx 16$ GB for 8B parameters at bf16, $\Sigma = 51.8$ MB across 24 Mamba-2 layers, and $\kappa = 16{,}384$ bytes per token because only 4 of its 52 layers hold a KV cache. For a dense Llama-3.1-8B the same law has $\Sigma = 0$ and $\kappa = 131{,}072$ bytes per token, since all 32 layers cache.

#### Worked example: decode throughput as the batch fills, at 4k context

Divide by an H100's 3.35 TB/s and read off the floor.

| Batch | Hybrid step floor | Hybrid tok/s | Dense 8B step floor | Dense tok/s | Source |
| --- | --- | --- | --- | --- | --- |
| 1 | 4.81 ms | 208 | 4.94 ms | 203 | derived |
| 8 | 5.06 ms | 1,581 | 6.06 ms | 1,320 | derived |
| 32 | 5.91 ms | 5,413 | 9.90 ms | 3,231 | derived |
| 128 | 9.32 ms | 13,734 | 25.29 ms | 5,061 | derived |
| 256 | 13.86 ms | 18,467 | 45.80 ms | 5,590 | derived |

At batch 1 the two are indistinguishable — a 2% difference that no user would notice. At batch 256 the hybrid delivers **3.3 times the tokens per second at less than a third of the per-token latency**, and every bit of that gap comes from the state term refusing to grow. The dense model's curve flattens because its KV traffic grows with the batch exactly as fast as its FLOPs do; the hybrid's keeps climbing because 51.8 MB per request is small enough that 256 of them are still only 13.3 GB against a 16 GB weight read.

**This is the batching argument, and it cuts both ways.** A hybrid that you serve at batch 4 is a hybrid you have paid for and not used. The architecture's entire compute advantage is a slope, and you only travel along it by having requests in flight. That is why the vLLM team's [disaggregated-serving work for hybrid SSM models](https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg) reports that a disaggregated prefill/decode deployment of `Nemotron-3-Super-120B-A12B-FP8` on 8 H200s "Pareto-dominates the co-located baseline at higher batch sizes" — the qualifier is the finding. Separating prefill from decode lets the decode pool run at a batch size the co-located version never reaches, and for a hybrid that matters more than it does for a transformer, because the hybrid's curve has further to climb.

There is a second-order effect worth naming. Because the recurrent layers' capacity does not depend on context length, your admission-control math changes shape: you can admit a request without knowing how long it will get, for 24 of the 52 layers. The 4 attention layers still impose the old constraint. Scheduling a model where half the capacity question has a constant answer and half does not is a genuinely different problem, and it is post 63's.

### 7.1 The launch-bound trap at small batch

The floors above are bandwidth floors. At small batch a hybrid does not reach them, and the reason is not bandwidth.

Count kernels. An unfused PyTorch Mamba-2 decode step is roughly: input projection GEMV, split, conv window concat, conv multiply, conv sum, conv state copy, softplus, exponential, state scale, state accumulate, readout contraction, gate multiply, normalization, output projection GEMV. Call it a dozen launches. Across 24 Mamba layers, 24 feed-forward blocks, 4 attention layers and their normalizations, a single decode step is somewhere in the neighborhood of 400 to 500 kernel launches. Kernel launch overhead is on the order of a few microseconds each — measure yours, do not take mine — so the CPU-side dispatch cost of one step is plausibly in the low milliseconds, against a 4.81 ms bandwidth floor at batch 1.

That is not a small correction. It means a hybrid at batch 1 without CUDA graphs is quite possibly CPU-bound, and the profile will show a GPU that is idle between kernels rather than a GPU that is waiting on HBM. Two consequences:

- **CUDA graph capture is not optional for hybrids, it is structural.** The step path is a fixed sequence of tiny kernels on fixed-shape tensors with in-place updates — close to the ideal case for graph capture, and much more valuable than for a dense transformer where each kernel is large enough to hide its own launch.
- **Fusion pays more here than anywhere else in the model.** Merging the twelve step kernels into one — which is what a real selective-scan decode kernel does, and what the vLLM team's Triton kernels borrowed from flash-linear-attention accomplish for Qwen3-Next, [per their launch post](https://vllm.ai/blog/2025-09-11-qwen3-next) — turns a dozen launches into one. The arithmetic does not change at all. The wall-clock can change by an order of magnitude.

---

## 8. Where it breaks: five stress tests

**Batch 1 on a laptop-class GPU.** Everything above assumed an H100. On an RTX 4090 — 165 dense bf16 TFLOP/s against roughly 1.0 TB/s of GDDR6X, per NVIDIA's specifications — the ridge sits near 165, so a chunk size of about 78 clears it and $Q = 128$ is already comfortable. But the 4090 has 24 GB, so an 8B model in bf16 leaves you about 8 GB for states and KV, which at 51.8 MB of state per request caps you near 150 concurrent requests before KV even enters. That is fine. The real problem on a 4090 is that its CPU-side launch overhead is proportionally larger relative to its smaller kernels, so the launch-bound regime extends to higher batch sizes. Expect the batch at which throughput starts scaling linearly to be higher on consumer hardware, and confirm with a profile rather than a guess.

**A 128k-token prompt.** The scan is linear in sequence length, so a 128k prefill costs 16 times a 8k prefill in the recurrent layers — no surprise there. What does surprise people: at 128k the *attention* layers dominate the hybrid's prefill anyway, because they are quadratic. With 4 attention layers at 32 query heads, a 128k prefill costs $2 H_q d S$ per token per layer, or 1.07 GFLOP per token — 64 times the recurrent layer's 16.8 MFLOP. Four such layers therefore contribute more prefill arithmetic than all 24 recurrent layers combined, by a factor of about 10. **A hybrid's long-context prefill is still an attention problem.** The recurrent layers fixed decode, not prefill.

**The seam at the prompt boundary.** This is the bug you will actually hit, so here is the narrative in full. Symptom: the model scores perfectly on a teacher-forced evaluation where you feed the whole sequence through the scan, and produces subtly degraded output — plausible but wrong facts, drifting style — when you generate autoregressively. Because the degradation starts exactly at the prompt boundary and compounds, it reads like a quality problem with the model. It is not. It is one of three things, in decreasing order of likelihood:

1. **The scan returned the state after a padded tail.** If your prompt length is not a multiple of the chunk size and you padded to make `L % chunk == 0`, the returned `h_final` includes the padding tokens' contributions. A KV cache would have masked them; a state cannot be un-updated. Trim the tail and run it through the step path, or handle a partial final chunk explicitly.
2. **The conv state was captured one position off.** The rolling window must hold the last $K-1$ *inputs to the convolution*, which are pre-convolution values, not post. Off by one and the first generated token sees a shifted receptive field.
3. **The two paths accumulate in different precisions.** The scan sums a chunk's contributions in whatever dtype the einsum promotes to; the step accumulates in the state's dtype. If one is fp32 and the other bf16, the seam is a real discontinuity.

The fix for all three is section 6.4's test, run at fp64 first, and extended to assert that `ssm_scan(seq[:n]) == n steps of ssm_step` for several values of `n` that are *not* multiples of the chunk size.

**Chunked prefill across scheduler steps.** Your scheduler's token budget and your scan's chunk size are two different chunk sizes and they will be confused. If the scheduler hands the layer 2,048 tokens this step and 2,048 next step, the layer must thread `h_final` from the first call into `h0` of the second — and if the budget is not a multiple of $Q$, the first call ends mid-chunk. The clean answer is to require the scheduler's per-request token budget to be a multiple of the scan's chunk size, which is a constraint the KV path never imposed and which your budget-picking logic now has to respect.

**Prefix caching, which mostly stops working.** Two requests sharing a 2,000-token system prompt can share KV blocks by hash, as the [prefix-caching machinery](/blog/machine-learning/model-serving/prefix-caching-and-radixattention) does. Can they share a state? Only if the shared prefix is the *entire* history — a state is a summary of everything so far, with no way to slice off a suffix or splice on a different one. You can cache the state at the end of a fixed system prompt and clone it for every new request, which is real and valuable. What you cannot do is the radix-tree trick where two divergent branches share their common ancestor's blocks and then go their separate ways cheaply, because the divergence requires two states and a state is 2 MiB, not a 16-token block. This is post 63's territory and it is one of the sharpest edges in the whole architecture.

---

## 9. Measuring this honestly

I have not run any of this. Here is how you should, so your numbers mean something.

**Separate the two paths in the profile.** Run `nsys profile` on a single request and look at the timeline twice: once during prefill, once during steady-state decode. In prefill you are looking for whether the scan's matmuls actually land on tensor cores — `ncu` will tell you the tensor-core utilization of the intra-chunk kernel directly. In decode you are looking for gaps. If the GPU timeline is mostly white space between kernels, you are launch-bound and no amount of kernel tuning will help until you capture a graph. The technique is the same one [the reproducible-benchmark post](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) sets out: warm up, `torch.cuda.synchronize()` before you start the clock, time with CUDA events rather than wall clock, lock the clocks, and report a steady-state window rather than a first iteration.

**Sweep the chunk size, and sweep it on your prompt lengths.** The cost model in section 6.6 gives you the FLOP curve for free. What it cannot give you is the achieved fraction of peak, which depends on matmul shapes and on your kernel. So run a sweep over $Q \in \{32, 64, 128, 256, 512\}$ at three prompt lengths — 512, 8k, 32k — and report TTFT for each. What you should expect to see, if the analysis above is right: a broad plateau between 64 and 256 at long prompts, a preference for small $Q$ at short prompts (where the serial depth is short anyway and the extra FLOPs are pure loss), and a degradation past 256 that gets worse as prompts get longer. If your curve is shaped differently, the interesting question is why, and the answer is usually memory pressure from unfused intermediates.

**Sweep the batch size, and report tokens per second, not latency.** At batch 1 a hybrid and a dense model of the same parameter count look the same, because at batch 1 both are reading weights. Every number that makes hybrids interesting appears at batch 32 and above. Run the batch sweep to at least 128 if your memory allows, plot tokens per second against batch, and compare the shape of the curve to the derived floors in section 7. The gap between your curve and the floor is your engine's overhead, and on a hybrid that gap will be dominated by launch overhead until you fix it.

**Assert equivalence in CI, not by eye.** The seam bug is silent. Section 6.4's test should run on every commit, in fp64, at several sequence lengths that are deliberately not multiples of the chunk size. It costs a second and it is the difference between shipping a correct engine and shipping one that quietly loses two points of accuracy.

**What not to report.** Tokens per second at batch 1 tells you nothing about a hybrid server, more so than for a transformer, because batch 1 is the exact regime where the architecture's advantage is zero. And do not report a chunk-size sweep from an unfused PyTorch implementation as if it predicted a fused kernel's behavior — the memory term that dominates the unfused version does not exist in the fused one, so the optima are in different places.

---

## 10. Case studies and public numbers

Four public results, each cited with its setup, none of them mine.

| Result | Setup | Source |
| --- | --- | --- |
| SSD steps 1, 2 and 4 are matmuls; only step 3 is a scan and takes "a small fraction" of the time | Mamba-2 reference implementation, `block_len=64` | cited: [Dao and Gu, SSD algorithm walkthrough](https://tridao.me/blog/2024/mamba2-part3-algorithm/) |
| KV cache reduced by up to 75%; up to 6x decoding throughput at 1M context | Kimi Linear, 48B total / 3B active, 3:1 KDA to full attention | cited: [Kimi Linear paper, arXiv 2510.26692](https://arxiv.org/abs/2510.26692) |
| Memory reduced 83% and inference latency 67% for 100k-token sequences | MiniMax-M1 lightning attention vs. softmax attention | cited: [vLLM MiniMax-M1 post, 2025-06-30](https://vllm.ai/blog/2025-06-30-minimax-m1) |
| Disaggregated prefill/decode "Pareto-dominates the co-located baseline at higher batch sizes" | `Nemotron-3-Super-120B-A12B-FP8`, 8xH200, prefill TP4 + decode TP4 | cited: [vLLM hybrid SSM disaggregation post, 2026-04-21](https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg) |

Two of those deserve a sentence of interpretation, because the headline number is not the interesting part.

**Kimi Linear's 6x at 1M context is a decode number, and it is a memory-traffic result, not a FLOP result.** At a million tokens, a full-attention layer reads a million tokens' worth of KV per step. A KDA layer reads a fixed state. The ratio of those two is what produces a 6x, and the derivation in section 5 is the same derivation at a different scale. Note also the setup — 3B activated parameters — which keeps the weight term small and therefore lets the state-versus-KV difference dominate. On a dense 70B the same architectural change would show a smaller end-to-end ratio because weights would be a larger share of the traffic.

**The vLLM disaggregation result's qualifier is the finding.** "At higher batch sizes" is not hedging; it is section 7 restated as an operational observation. A hybrid's advantage is a slope, and a co-located deployment where prefill work keeps interrupting decode never gets to run decode at a batch large enough to walk up that slope. The same post also lists what does not yet work, and the list is instructive: Mamba1 unsupported, gated DeltaNet pending, speculative-decoding interaction "not extensively validated," mixed-block-size hybrid attention unsupported. Separately, vLLM's [Model Runner V2 announcement](https://vllm.ai/blog/2026-03-24-mrv2) notes that as of v0.18.0 its faster runner did not yet cover linear-attention models. The engine work this track is about is genuinely unfinished in production systems, which is a reason to understand it rather than a reason to avoid it.

**One negative result worth internalizing.** The Qwen3-Next launch post describes vLLM automatically tuning "the 'logical' block size of the full attention layers to ensure that the state for the full attention layers and linear attention layers occupy the same amount of 'physical' GPU memory." That is a workaround for a mismatch this post's dual form creates: two kinds of per-request memory with two completely different granularities, forced into one allocator. It works, and it is also the sort of thing that exists because the clean design does not.

---

## 11. When to reach for this, and when not to

![A decision tree splitting a recurrent layer request into the scan path tuned by chunk size and the step path tuned only by batch size](/imgs/blogs/prefill-is-a-scan-decode-is-a-recurrence-7.webp)

The tree above is the short version: sequence length picks the path, chunk size tunes the scan, batch size tunes the step, and there is no third knob on either side.

**Write both paths yourself when** you are learning how the layer works, when you need a reference implementation to test a fused kernel against, or when you are working with an architecture whose kernels do not exist yet. The scan in section 6.3 is about sixty lines and it is *correct*, which is more than you can say for most first attempts at the fused version.

**Do not write the fused kernel yourself unless you have to.** For Mamba-2 the `mamba_ssm` package's selective-scan kernels exist; for the delta-rule family, [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) provides Triton kernels that vLLM itself uses for Qwen3-Next. The gap between a good PyTorch scan and a good fused kernel is large — the fused version keeps the score matrices on chip, which is the difference between the AI column in section 4's table and something much worse — but it is a gap that other people have already closed for the mainstream architectures.

**Do not serve a hybrid at low batch and expect a win.** If your workload is single-stream — one user, batch 1, interactive — a hybrid buys you memory, not speed. The tok/s table in section 7 shows 208 versus 203, which is noise. Serve it at batch 1 and you have adopted an architecture whose advantage you have structurally excluded.

**Reach for the disaggregated deployment when the batch is the constraint.** If your decode pool cannot reach batch 32 because prefill work keeps preempting it, the fix is not a better kernel, it is separating the pools — and the vLLM result above says that is where hybrids start to dominate.

**Just use vLLM when** your model is one of the supported hybrids and your problem is serving it rather than understanding it. The hybrid KV-cache manager, the logical-block-size tuning, and the FLA kernels represent a lot of careful work that you will not reproduce in a weekend. Build `nanoserve`'s version to know what it is doing; run the production one to ship.

---

## Key takeaways

1. **A linear recurrence is associative, and that is the whole license for the parallel form.** $(a_1,b_1)\bullet(a_2,b_2) = (a_1a_2,\, a_2b_1+b_2)$ composes associatively, so evaluation order is free even though sequence order is not.
2. **The chunked scan is three matmuls and one short serial scan.** At chunk size 256, 99.93% of the arithmetic is tensor-core work and the serial part is 0.07%.
3. **Chunk size is a roofline knob you can solve for.** On an H100 the bf16 ridge is 295 FLOP per byte, which puts the crossover near 138 tokens per chunk — which is why the ecosystem's defaults are 64 to 256 and not 8 or 8,192.
4. **Bigger chunks buy parallelism with redundant arithmetic.** Going from 64 to 256 costs 2.3 times the FLOPs and buys a 4x shorter serial depth plus better matmul shapes. Going to 512 costs 4x the FLOPs for very little more.
5. **Decode's recurrence is 2% of its own layer.** The projections are 219 MB of weights against 4.19 MB of state; at batch 1 the state is invisible and the layer looks like every other GEMV-bound decode step.
6. **The state term is per request and the weight term is not.** That is why hybrid decode's arithmetic intensity climbs to a ceiling of 53.5 while a GQA attention layer's is pinned at 4.6 — an 11.6x larger roof, collected only by filling the batch.
7. **At batch 1 a hybrid and a dense model of equal size are the same speed.** At batch 256 and 4k context the derived floors differ by 3.3x. The architecture's advantage is a slope, not an offset.
8. **You cannot flatten a mixed batch through a recurrent layer.** Prefill and decode are different kernels; the engine splits the batch, runs both, and restitches.
9. **The seam between the two paths is the bug that will cost you a week.** Assert `scan(L tokens) == L steps` in fp64, at lengths that are not multiples of the chunk size, in CI.
10. **Hybrids fix decode, not prefill.** At 128k the handful of remaining attention layers still dominate prefill arithmetic, because quadratic beats linear at scale in the direction you do not want.

---

## Further reading

- [*Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*](https://arxiv.org/abs/2405.21060) — Dao and Gu, ICML 2024. The duality proof and the chunked algorithm, from the source.
- [State Space Duality (Mamba-2) Part III — The Algorithm](https://tridao.me/blog/2024/mamba2-part3-algorithm/) — the authors' own walkthrough of the four blocks, with the reference `block_len=64` implementation.
- [*Kimi Linear: An Expressive, Efficient Attention Architecture*](https://arxiv.org/abs/2510.26692) — Kimi Team, October 2025. A 3:1 delta-rule hybrid at 48B total / 3B active, with the KV and throughput claims quoted in section 10.
- [PyTorch `associative_scan` documentation](https://docs.pytorch.org/docs/main/higher_order_ops/associative_scan.html) — the higher-order op, its associativity requirement, and its prototype-status caveats.
- [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) — Triton kernels for the gated delta rule and friends, including the chunk-local cumulative sums discussed in section 2.1.
- [Disaggregated Serving for Hybrid SSM Models](https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg) — vLLM, April 2026. The production engine's account of what a fixed-size state does to descriptors, transfers and batching.
- [Hybrid models and the end of the KV-cache assumption](/blog/machine-learning/inference-engineering/hybrid-models-and-the-end-of-the-kv-cache-assumption) — the memory side of the same architectural fact, with the state-bytes derivation this post reuses.
- [Chunked prefill and the TTFT/TPOT tradeoff](/blog/machine-learning/inference-engineering/chunked-prefill-and-the-ttft-tpot-tradeoff) — the scheduler's chunk, which is not the scan's chunk, and how the two must be made to agree.
- [The inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) — the capstone, where every track's technique gets weighed against the others on one scoreboard.

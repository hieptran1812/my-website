---
title: "Inkling at FP4: Serving a 975B MoE on NVIDIA and AMD, Explained"
date: "2026-07-17"
description: "How TokenSpeed brought Day-0 inference to Thinking Machines Lab's 975B Inkling model in native FP4 on both NVIDIA and AMD — the quantization, the flat KV cache, the decode-specialized kernels, and multi-token prediction, each built up from intuition to the math."
tags: ["paper-reading", "llm-inference", "fp4", "nvfp4", "mxfp4", "quantization", "kv-cache", "flash-attention", "speculative-decoding", "multi-token-prediction", "mixture-of-experts", "gpu-kernels"]
category: "paper-reading"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 33
paper:
  title: "TML Inkling at Day 0: FP4 Inference on NVIDIA and AMD with TokenSpeed"
  authors: "TokenSpeed Team & Thinking Machines Lab"
  venue: "TokenSpeed / LightSeek Foundation engineering blog, Jul 2026"
  url: "https://lightseek.org/blog/tokenspeed-inkling.html"
---

> [!tldr]
> - **What it is.** A serving-systems write-up: TokenSpeed and Thinking Machines Lab (TML) stood up **Day-0 inference for Inkling** — a 975B-parameter mixture-of-experts model with 41B active parameters per token — in **native 4-bit floating point** on *both* NVIDIA ((G)B200/B300, NVFP4) and AMD (MI350X/MI355X, MXFP4).
> - **The key mechanism.** One accelerator-neutral model layer sits on a **unified kernel API**; everything vendor-specific (the FP4 format, the attention kernels) lives below that line, so the same model and scheduler drive two silicon stacks.
> - **Why it matters.** FP4 shrinks a 975B model from ~1.95 TB to ~520–550 GB, which is what lets it fit on **four** GPUs with room left for a 50k-token KV cache — and the quantized checkpoints stay within ~1–3 points of the BF16 baseline on quality.
> - **Most surprising result.** Multi-token prediction (MTP) delivers a **2.33× per-user decode speedup at batch size 1**, but the same feature *slows decode down* (0.94×, 0.84×) at batch sizes 3–4. The win is real but batch-shaped.
> - **Where it's thin.** It's an engineering snapshot, not a paper: no error bars, "early" AMD numbers, and the FP4 numeric internals are left to the format specs. I fill those in below and flag every place I go beyond the post.

The source is an engineering blog, licensed CC BY 4.0, not a peer-reviewed paper — so I treat its text and figures as the ground truth and mark clearly wherever I reach past it into the FP4 or speculative-decoding literature to explain a mechanism the post only names. Every figure credited "TokenSpeed / TML" is reproduced from that post under its license; the four hand-drawn diagrams are mine.

## The problem: one model, two incompatible accelerators, zero warm-up time

"Day-0 support" is a deceptively small phrase. It means that on the morning a model's weights go public, an inference stack can already serve it at production speed. For a dense 7B model on a single well-trodden GPU, that is a config change. For Inkling, it is not.

Three things stack up at once. **First, scale.** Inkling has 975 billion total parameters. Even though only 41 billion are *active* per token — it is a mixture-of-experts model, so each token is routed to a small subset of the experts — *all* 975B weights have to be resident in GPU memory, because any token might route to any expert. In BF16 that is roughly 1.95 terabytes of weights before you have stored a single key or value. **Second, mixed attention.** Inkling does not use one attention type; it interleaves full attention (every token attends to all previous tokens) with sliding-window attention (each token attends only to a fixed recent window), and it carries convolutional state on top. Each of those wants a differently-shaped cache. **Third, two silicon stacks.** NVIDIA and AMD have different low-precision number formats, different memory hierarchies, and completely different kernel programming models. A stack that only speaks CUDA cannot serve the AMD checkpoint, and vice versa.

The naive answer — "quantize it, then write CUDA kernels" — fails all three at the layer that matters. It leaves the AMD path unserved, it fragments the cache across attention types, and it reuses prefill-shaped attention kernels for the decode loop, where they waste most of the GPU. The Inkling write-up is a tour of the specific engineering that dissolves each of these, and it is unusually honest about where the wins are and are not.

The rest of this post walks the stack from the weights up: the FP4 checkpoints, the flat KV cache that unifies three kinds of state, the attention kernels specialized for decode on each vendor, and multi-token prediction. Each load-bearing technique gets the full treatment — the problem it solves, an intuition, the mechanism, the math with every symbol defined, a worked micro-example, and the regime where it breaks.

## What TokenSpeed actually shipped

Stripped to a list, the contributions are:

1. **Two native FP4 checkpoints.** TML released Inkling in BF16 and NVFP4; TokenSpeed used AMD Quark to quantize it to MXFP4 for MI350X/MI355X and published the result at `lightseekorg/Inkling-MXFP4`. Both stay close to the BF16 baseline on quality.
2. **A flat KV-cache architecture.** A single paged memory pool with *heterogeneous views* holds full-attention KV, sliding-window KV, and convolution state — uniform allocation without padding the small states up to the big one's footprint.
3. **A unified kernel API.** One API surface spans NVIDIA and AMD, so the model layer and scheduler are written once and only the kernels are vendor-specific.
4. **A decode-specialized attention kernel per vendor.** A CuteDSL kernel on NVIDIA and Gluon kernels on AMD, both shaped for the decode regime (one query row, a very long KV) instead of reusing the prefill kernel.
5. **Multi-token prediction (MTP).** A speculative-style decode that proposes several tokens per step and verifies them in parallel, advancing ~3.3 tokens per iteration at low concurrency.

Numbered like that, it reads like five unrelated tricks. It isn't — it is one architectural decision (put the vendor boundary at the kernel API) followed by four exploitations of it. Let's build up to why that boundary is the whole game.

## The model: Inkling and its two FP4 checkpoints

### Reading Inkling's architecture

Before the serving tricks, fix the shape of the thing being served. From the post: Inkling is a transformer-based MoE with **66 layers** and **256 routed experts**, of which **6 routed experts plus 2 shared experts** activate per token, giving **975B total / 41B active** parameters. It interleaves **full and sliding-window attention**, and — as the cache section will reveal — also carries convolutional state.

The 66 layers are not a flat stack. They factor into **11 repeating units of 6 layers each**: five sliding-window-attention (SWA) layers and one full-attention layer per unit. That 5:1 ratio is the single most important number for the cache design, so hold onto it.

How competitive is the model that has to be served this fast? The post's benchmark table places Inkling among current open- and closed-weight frontier models:

![Benchmark comparison: Inkling versus contemporary open- and closed-weight models across reasoning, agentic, factuality, chat, vision, audio, and safety suites. Figure from TokenSpeed / Thinking Machines Lab (CC BY 4.0).](/imgs/blogs/inkling-fp4-inference-nvidia-amd-fig1.webp)

The point of showing this here is not the leaderboard — it is that Inkling is a *frontier-class* model, so the serving work is not a toy exercise. AIME 2026 at 97.1%, GPQA Diamond at 87.9%, and strong agentic and multimodal scores mean the thing being squeezed into FP4 on four GPUs is genuinely capable, which raises the bar on preserving quality through quantization.

### Technique 1 — native FP4 block quantization

**The problem it solves.** 1.95 TB of BF16 weights does not fit on four GPUs, and moving that many bytes from memory to the compute units every forward pass is the dominant cost of decode. You want to store and move each weight in as few bits as possible — without moving the model's outputs.

**Intuition.** Think of a street of houses whose prices you want to text to a friend, cheaply. Sending each exact price (`472,318`, `489,201`, …) is expensive. Instead you send one "these are all around half a million" scale for the whole street, then a tiny one-digit offset per house (`-3`, `+2`, `-1`, …). The friend multiplies each offset by the shared scale to recover a good-enough price. FP4 quantization is exactly this: a **shared scale per block of weights**, plus a **4-bit code per weight**. The block amortizes the cost of the (relatively expensive) scale across many weights.

**The mechanism, step by step.** Take a contiguous block of $B$ weights $w_1, \ldots, w_B$ from a weight matrix.

1. Find the block's largest magnitude, $a = \max_i |w_i|$.
2. Choose a scale $s$ so that the biggest weight maps to the top of the 4-bit range.
3. Round each $w_i / s$ to the nearest representable **4-bit floating-point code** $q_i$.
4. Store the $B$ codes plus the one scale. At compute time, dequantize with $\hat{w}_i = s \cdot q_i$.

The 4-bit code format is **E2M1**: one sign bit, two exponent bits, one mantissa bit. *(This bit layout, and the block/scale details below, come from the NVFP4 and OCP microscaling specifications — the post names the formats but does not spell out their internals; I'm filling that in.)* With 2 exponent and 1 mantissa bits, E2M1 can represent only these magnitudes: $0,\ 0.5,\ 1,\ 1.5,\ 2,\ 3,\ 4,\ 6$. So the maximum representable code magnitude is $q_{\max} = 6$.

**The math.** The scale is chosen to line the block's peak up with the format's peak:

$$
s = \frac{\max_i |w_i|}{q_{\max}}, \qquad q_{\max} = 6 .
$$

Here $s$ is the per-block scale (a small floating-point number stored alongside the block), $w_i$ is the original weight, and $q_{\max}$ is the largest magnitude E2M1 can encode. Each weight is then quantized and later dequantized as

$$
q_i = \operatorname{round}_{\text{E2M1}}\!\left(\frac{w_i}{s}\right), \qquad \hat{w}_i = s \cdot q_i ,
$$

where $\operatorname{round}_{\text{E2M1}}(\cdot)$ snaps its argument to the nearest of the eight representable magnitudes (with sign), and $\hat{w}_i$ is the reconstructed weight the kernel actually multiplies. The reconstruction error $|\hat{w}_i - w_i|$ is bounded by half the gap between adjacent E2M1 levels *times* $s$ — which is why a *tight* scale (small block, so $a$ tracks the local weights) matters.

The **storage cost per weight** is the 4 code bits plus the block's share of the scale bits:

$$
b_{\text{eff}} = 4 + \frac{b_{\text{scale}}}{B},
$$

where $b_{\text{scale}}$ is the number of bits in the shared scale and $B$ is the block size. This one formula is the entire design tension: a **smaller block** $B$ gives a tighter, more accurate scale but spends more bits amortizing it; a **larger block** is cheaper but coarser. NVFP4 and MXFP4 pick different points on this curve.

![Three ways to store one weight: BF16 keeps a full 16-bit float; NVFP4 and MXFP4 both use a 4-bit E2M1 code but differ in block size and shared-scale format, so their effective bits-per-weight land at 4.5 and 4.25.](/imgs/blogs/inkling-fp4-inference-nvidia-amd-1.webp)

- **NVFP4** (NVIDIA): block size $B = 16$, scale in **FP8 E4M3** (8 bits). Effective bits $= 4 + 8/16 = 4.5$.
- **MXFP4** (AMD, the OCP "microscaling" format): block size $B = 32$, scale in **E8M0** — an 8-bit *power-of-two* exponent, no mantissa. Effective bits $= 4 + 8/32 = 4.25$.

The difference is more than the 0.25 bits. NVFP4's smaller 16-wide block and full FP8 scale track local weight statistics more tightly (better accuracy per code); MXFP4's power-of-two scale means dequantization is a cheap exponent add rather than a multiply, which suits AMD's path. Same 4-bit code, two philosophies about the scale.

**Worked micro-example.** Take a block of four weights (using $B=4$ for legibility): $w = [0.42,\ -1.9,\ 5.4,\ 0.03]$.

- $a = \max|w| = 5.4$, so $s = 5.4 / 6 = 0.9$.
- Divide: $w/s = [0.47,\ -2.11,\ 6.0,\ 0.033]$.
- Snap each to E2M1 magnitudes $\{0,0.5,1,1.5,2,3,4,6\}$: $q = [0.5,\ -2,\ 6,\ 0]$.
- Dequantize $\hat{w} = s \cdot q = [0.45,\ -1.8,\ 5.4,\ 0]$.

The big value (5.4) round-trips exactly; the mid value (−1.9 → −1.8) is close; the tiny value (0.03 → 0) is flushed to zero — quantization always sacrifices the small entries first, which is why keeping *shared* experts and sensitive layers at higher precision (a standard trick) matters when the naive scheme costs too much quality.

**Why it works / when it fails.** It works because weights within a small block have similar magnitudes, so one scale plus a coarse code preserves the *ratios* that matter for the matrix product. It fails when a block straddles a huge outlier — then the scale is dragged up to cover the outlier and every other weight in the block collapses toward zero (the 0.03 → 0 effect, at scale). This is exactly why block size is small (16 or 32, not 1024) and why the community obsesses over outlier handling. The proof it worked here is the quality table: across six suites, NVFP4 and MXFP4 stay within ~1–3 points of BF16, and on two suites (AIME 2026, BFCL exact calls) they even edge ahead — which is quantization noise, not a real gain, but confirms nothing broke.

**A memory back-of-envelope (my calculation).** At 4.5 bits/weight, NVFP4 stores 975B weights in $975\times10^9 \times 4.5/8 \approx 548$ GB; MXFP4 at 4.25 bits needs $\approx 518$ GB. A B200 carries 192 GB of HBM and an MI355X 288 GB (public specs), so four B200s (768 GB) hold the NVFP4 weights with ~220 GB to spare for KV cache and activations, and four MI355X (1152 GB) hold the MXFP4 weights with ~630 GB free — which is precisely why the post can say the model "run[s] on four MI355X GPUs while preserving enough cache capacity for 50k+ token contexts." FP4 is not just a speed trick; it is the thing that makes the GPU count small enough to be economical.

## A flat KV cache for three kinds of state

### Technique 2 — one flat pool, heterogeneous views

**The problem it solves.** Inkling's inference carries three persistent per-request states, each with a different per-token footprint:

- a **growing KV state** for the full-attention layers (grows with the whole sequence),
- a **bounded KV state** for the sliding-window layers (capped at the window),
- a **window/convolution state** for the convolutions (tiny, fixed).

Give each its own memory pool and you fragment the cache: free space in the SWA pool can't be lent to the full-attention pool, so you hit out-of-memory while holding free bytes. Put them all in *one* pool with a *uniform page shape* and you waste memory the other way — every small SWA or convolution page gets padded up to the footprint of the largest full-attention page.

**Intuition.** Think of a parking structure where every stall is the same physical size, but what parks in it differs by floor: motorcycles on one floor, sedans on another, trucks on a third. You don't build three separate garages (fragmentation) and you don't size every stall for a truck (waste). You keep **one stall size and one numbering scheme**, and let each floor decide how many vehicles fit in a stall. The stall id `k` means "row `k`" on *every* floor; the floors just hold different things.

That is exactly TokenSpeed's move: **one flat paged pool, with heterogeneous logical views on top of a uniform physical slot.** The post credits the same high-level idea to recent community work — [Jenga](https://arxiv.org/abs/2503.18292) in vLLM, which separates *physical* memory allocation from *logical* memory organization.

![Inkling's flat KV cache: the 66 layers form 11 six-layer units, each mapped to one slab; a single fixed-size slot holds 256 full-attention KV tokens, 128 sliding-window/KV-convolution tokens, or 16 hidden-state-convolution tokens. Figure from TokenSpeed / Thinking Machines Lab (CC BY 4.0).](/imgs/blogs/inkling-fp4-inference-nvidia-amd-fig2.webp)

**The mechanism, step by step.** Recall the 11 units of 6 layers (5 SWA + 1 full). Each unit also owns six KV-side convolutions and six hidden-state convolutions. All of a unit's state maps to **one slab**, so there are **11 slabs**. A **block id** selects the *same* fixed-size slot — the same physical row — across all 11 slabs at once. Because the states have different per-token byte footprints, that one physical slot holds a *different number of tokens* depending on which state occupies it:

- **256 tokens** of full-attention KV,
- **128 tokens** of sliding-window KV, or of KV-side convolution state,
- **16 tokens** of hidden-state convolution state.

The allocation *unit* is uniform (one slot = one row in every slab); the logical *page size* is not.

**The math.** Let a slot be $S$ bytes and let state $t$ cost $c_t$ bytes per token. The logical capacity of a slot for state $t$ is

$$
\text{tokens}_t = \left\lfloor \frac{S}{c_t} \right\rfloor .
$$

The observed capacities (256 / 128 / 16) tell you the *ratio* of footprints directly: full-attention KV costs half as much per token as… no — read it the other way. A larger `tokens_t` means a *smaller* per-token footprint. So hidden-state convolution state is the heaviest per token (only 16 fit), sliding-window KV is $128/16 = 8\times$ lighter, and full-attention KV is lightest per token ($256/16 = 16\times$), which fits intuition: a single KV vector per token is cheaper than a full convolution window. The key property is that **one $S$** serves all three, so allocation, eviction, and scheduling operate on a single uniform quantity.

**Worked micro-example.** A request needs cache for 512 full-attention tokens. At 256 tokens/slot that is $\lceil 512/256\rceil = 2$ block ids. Those same 2 block ids simultaneously reserve, in the SWA slabs, $2 \times 128 = 256$ tokens of sliding-window capacity and, in the convolution slabs, $2 \times 16 = 32$ tokens of hidden-state capacity — all from *two* integers. Freeing the request returns exactly those 2 block ids to the shared pool, where any other request or any other state type can immediately reuse them. No per-type free list, no fragmentation between types.

**Why it works / when it fails.** It works because it decouples the thing you must keep uniform for fast scheduling (the allocation unit) from the thing that is inherently non-uniform (per-token footprint by state type). It degrades if the footprints are *wildly* mismatched: the 16:128:256 spread means the slot is sized so that 16 hidden-conv tokens fit, which is fine here, but a hypothetical state costing, say, one token per slot would waste most of the slot. The design is tuned to Inkling's specific ratios — it is not a free lunch for arbitrary state mixes.

### The management hierarchy: who owns a page

The physical layout is half the story; the other half is *who is allowed to hand out and reclaim a page*. Get this wrong and you either leak memory or, worse, let two requests scribble on the same slot.

![Cache-management hierarchy: a coordinator fans each request out to cache groups, each group keeps a per-request BlockTable of reference-counted BlockRefs into one shared pool, and page id k maps directly to row k of the physical slabs. Figure from TokenSpeed / Thinking Machines Lab (CC BY 4.0).](/imgs/blogs/inkling-fp4-inference-nvidia-amd-fig3.webp)

Reading the hierarchy top-down:

- A **coordinator** fans each incoming request out to the **cache groups** (e.g. `full_attention`, `sliding_attention`).
- Each group maintains its own **per-request BlockTable**. Table entries do not hold raw pages; they hold **reference-counted BlockRefs** into one shared **block pool**.
- **Page id `k` maps directly to row `k`** in the physical slabs — the same identity relation the flat layout relies on.
- The group managers own **matching and eviction policy** (which pages to keep for prefix reuse, which to drop under pressure), but **memory ownership stays centralized** in the single pool.

The BlockRef is the load-bearing detail. Because a page is only ever held through a reference-counted handle, a page shared between requests (prefix caching, where two conversations share a common system prompt) is kept alive exactly as long as *someone* references it, and is freed the instant the last reference drops — safely reusable across groups immediately. This is the classic RAII / reference-counting discipline applied to GPU pages: correctness (no double-free, no use-after-free) falls out of the ownership rule rather than out of careful manual bookkeeping in the scheduler. If you have read about [prefix caching and RadixAttention](/blog/machine-learning/model-serving/prefix-caching-and-radixattention), this is the allocator that makes safe cross-request sharing cheap.

## Attention kernels: prefill and decode want opposite things

Attention is a large share of Inkling's compute, and the single most important systems fact about it is that **prefill and decode are different shapes of the same math**. A kernel tuned for one is wrong for the other.

### Technique 3 — the decode-specialized CuteDSL kernel (NVIDIA)

**The problem it solves.** During **prefill**, you process the whole prompt at once: the query sequence is long, so an attention kernel has thousands of query rows to spread across the GPU's thousands of lanes. During **decode**, you generate one token at a time: the query is a *single row* (or a few, with MTP), while the KV cache it attends to can be tens of thousands of tokens long. A prefill-style kernel is organized around large query tiles; hand it a one-row query and most of the tile — and most of the GPU — sits idle.

**Intuition.** A prefill kernel is a wide harvester built to mow a whole field in parallel passes. Decode hands it a single row of wheat and a mile-long fence to inspect. The harvester's width is wasted; what you want instead is a narrow, fast vehicle that *drives the length of the fence* and packs its few passengers efficiently. Prefill parallelizes across the query (the field); decode must parallelize across the **KV length** (the fence).

![Prefill vs decode: the same softmax attention, opposite shapes. A prefill-tiled kernel wastes compute lanes when the decode query is a single row; a decode-specialized kernel streams the long KV and packs the query heads into each tile.](/imgs/blogs/inkling-fp4-inference-nvidia-amd-2.webp)

**The mechanism, step by step.** For prefill, TokenSpeed reuses TML's **FlashAttention-4 (FA4)** path (developed by Colfax Research) — it already tiles beautifully along a long query sequence. For decode, they write a **dedicated kernel** that:

1. **Streams over the long KV sequence** — the parallelism comes from splitting the 64k keys across the GPU, not from splitting the (length-1) query.
2. **Packs the small query/prediction dimension into each CTA tile.** A CTA (cooperative thread array, a thread block) tile that would be mostly empty with one query row is filled by packing the multiple query heads — and, under MTP, the multiple predicted positions — into it.

The head geometry makes point 2 concrete. The kernel benchmark runs `seqlen_q=1, seqlen_kv=64k, h_q=16, h_kv=2`: one query position, 64k KV positions, 16 query heads sharing 2 KV heads. That 16:2 ratio is grouped-query attention *(my reading of the head counts)* — eight query heads per KV head. Those 16 heads are what the decode kernel packs into the tile so the hardware is busy even though `seqlen_q=1`.

**The math.** Attention is unchanged; only the tiling differs. For a single decode step,

$$
o = \operatorname{softmax}\!\left(\frac{q K^\top}{\sqrt{d}}\right) V,
$$

where $q \in \mathbb{R}^{1 \times d}$ is the single query row of head dimension $d$, $K \in \mathbb{R}^{L \times d}$ and $V \in \mathbb{R}^{L \times d}$ are the cached keys and values over $L$ positions (here $L = 64\text{k}$), and $o \in \mathbb{R}^{1 \times d}$ is the output. The $\sqrt{d}$ divisor keeps the dot products from growing with dimension, which would otherwise push softmax into a near-one-hot regime.

The decisive quantity is **arithmetic intensity** — FLOPs per byte moved. Decode reads all $L \times d$ bytes of $K$ and $V$ to produce a length-1 output, so it does $O(L d)$ work over $O(L d)$ bytes: intensity $\approx O(1)$, i.e. **memory-bound**. Prefill with a length-$L_q$ query does $O(L_q L d)$ work over the same $O(L d)$ bytes: intensity $\approx O(L_q)$, **compute-bound**. *(This intensity analysis is my framing of why the regimes differ; the post states the outcome, not the derivation.)* A memory-bound kernel wins by maximizing bytes-in-flight — exactly what streaming the KV and packing heads achieves, and exactly what a compute-tiled prefill kernel fails to do.

**Worked micro-example — the measured payoff.** The post's full-attention decode benchmark makes the gap concrete:

![Full-attention decode timing: TML's FA4 path (attention core + a small ShearingBias preprocessing bar) versus the decode-specialized kernel, at seqlen_q=1, seqlen_kv=64k, h_q=16, h_kv=2. Figure from TokenSpeed / Thinking Machines Lab (CC BY 4.0).](/imgs/blogs/inkling-fp4-inference-nvidia-amd-fig4.webp)

| Batch | FA4 core + bias (µs) | Decode kernel (µs) | Speedup |
| ----: | -------------------: | -----------------: | ------: |
| 1     | 26.6 + 2.6           | 15.4               | 1.87×   |
| 2     | 34.9 + 2.6           | 25.8               | 1.45×   |
| 4     | 55.0 + 2.6           | 43.2               | 1.33×   |
| 8     | 100.9 + 2.6          | 80.1               | 1.28×   |
| 16    | 200.7 + 2.6          | 152.2              | 1.33×   |

The decode kernel is **1.28×–1.87×** faster across batch sizes, and the win is largest at batch 1 — precisely the memory-bound, lanes-idle regime where a prefill kernel wastes the most.

**Why it works / when it fails.** It works because it matches the parallel axis to the shape of the problem (KV length, not query length) and keeps the tile full by packing heads. It stops helping as batch grows: at batch 16 the many independent query rows across the batch already give a prefill-style kernel enough parallelism, so the specialized kernel's edge narrows (1.87× → 1.33×). The decode kernel is a *low-concurrency* optimization — remember this, because MTP behaves the same way for the same reason.

### ShearingBias: relative bias without a second kernel

Inkling applies a **relative bias before softmax** (a position-dependent additive term inside the attention logits). In the FA4 **prefill** path this is handled by a *separate* `ShearingBias` preprocessing kernel — the little 2.6 µs bar stacked on the FA4 bars above. That extra kernel is affordable in prefill because its cost is **amortized across many query rows**.

In **decode**, spinning up a second kernel to bias a single query row would be pure overhead — 2.6 µs on top of a 15 µs kernel is a 17% tax. So the decode kernel folds the bias in: because the query dimension is tiny, it **computes the relative indices directly inside the online-softmax loop**, at essentially no extra cost. This is a small detail with a clean lesson: an optimization that pays for itself when amortized over a long query (prefill) becomes dead weight when the query is length-1 (decode), so the decode path inlines it instead. Same arithmetic, moved to where it's free.

### Technique 4 — Gluon attention for AMD (persistent prefill, split-K decode)

**The problem it solves.** Everything above is CUDA. AMD's MI350X/MI355X need their own kernels, and rewriting the whole model to get them is exactly the cross-vendor tax the project set out to avoid.

**Intuition.** Two very different assembly lines building the same car. You do not want two different *car designs*; you want one blueprint and two floor layouts. The floor layout is the kernel; the blueprint is the model code.

**The mechanism.** For AMD, TokenSpeed extends its existing **Gluon** attention kernels to cover Inkling's prefill and decode:

- **Prefill** uses a **persistent loop** — a kernel that launches once and keeps thread blocks resident, looping over work internally instead of paying repeated launch and scheduling overhead. Good when there is a lot of query work to grind through.
- **Decode** uses a **split-K** design — the long KV dimension ($K$, the "fence") is split into chunks processed in parallel, then their partial softmax results are combined. This is the AMD-side answer to the same "parallelize over KV length, not query length" problem the CuteDSL kernel solves on NVIDIA.

Crucially, these Gluon kernels **implement the unified kernel API** alongside the NVIDIA backend, so the model code stays accelerator-neutral while the AMD path still gets hand-optimized kernels.

**The math** is the same attention as before — split-K just reorders the reduction. Splitting the $L$ keys into $G$ groups, each group computes a partial numerator and a partial softmax denominator (a running max $m_g$ and sum $\ell_g$), and the groups are merged with the standard online-softmax rescale:

$$
o = \frac{\sum_{g=1}^{G} e^{\,m_g - m^\star}\, \tilde{o}_g}{\sum_{g=1}^{G} e^{\,m_g - m^\star}\, \ell_g}, \qquad m^\star = \max_g m_g,
$$

where $\tilde{o}_g$ is group $g$'s unnormalized output, $m_g$ its local max logit, $\ell_g$ its local exponent-sum, and $m^\star$ the global max used to rescale every group onto a common exponent base so the partial sums can be added without overflow. The $G$ groups run in parallel — that is the whole point of split-K for a memory-bound decode.

**Why it works / when it fails.** Split-K wins when $L$ is long enough that one thread block cannot saturate the GPU (decode's exact situation). It *loses* on short KV, where splitting adds merge overhead without adding useful parallelism — which is why prefill (long query, short-per-row KV) uses the persistent loop instead. The right kernel depends on the shape, and the API lets each vendor pick its own.

### The unified kernel API that makes both possible

Step back and the architecture is one idea: TokenSpeed's modular stack separates the **model layer**, the **scheduler**, and the **kernel** subsystems behind clean boundaries, and puts the vendor line at the kernel API.

Enabling Inkling is then a recipe rather than a rewrite: write the **accelerator-agnostic model logic once**, **reuse the existing scheduler**, and bring up NVIDIA and AMD from the *same* model integration by implementing the kernel API twice. NVFP4 versus MXFP4, CuteDSL versus Gluon, persistent-loop versus split-K — all of that vendor divergence lives *below* the API. Above it, the model and scheduler cannot tell which silicon they are running on.

This is why "Day 0 on both vendors" was achievable at all. If the vendor boundary sat higher — in the model or the scheduler — every new model would need two integrations. Putting it at the kernel API means a new model is one integration, and porting to a new accelerator is a set of kernels, not a fork. (For how much a specific accelerator's quirks still leak into tuning even with a clean API, see [GPU-architecture-specific tuning for LLM serving](/blog/machine-learning/model-serving/gpu-architecture-specific-tuning-for-llm-serving).)

## Technique 5 — multi-token prediction (MTP)

**The problem it solves.** Decode is inherently sequential: token $t+1$ needs token $t$, so you pay one full forward pass — the cost of reading all 41B active parameters (and streaming the KV) from memory — for *one* token. At batch 1, that forward pass is memory-bound, so the compute units are mostly idle while the bytes stream in. You are paying for a forward pass and using a fraction of it. Can you get more than one token out of it?

**Intuition.** A careful editor (the full "target" model) reads slowly and is always right. A fast intern (a small "draft" head) guesses the next several words quickly but is sometimes wrong. Instead of the editor writing each word, the **intern proposes a short run of words** and the **editor checks the whole run in a single glance**, keeping the correct prefix and fixing the first mistake. When the intern is usually right, the editor commits several words per glance instead of one. That is speculative decoding, and **multi-token prediction** is the variant where the "intern" is the model's own extra prediction heads rather than a separate small model.

![Multi-token prediction as a graph: one draft forward proposes several tokens, one parallel target forward verifies them, the longest matching prefix is committed, and a mismatch costs a single correction token before the next iteration.](/imgs/blogs/inkling-fp4-inference-nvidia-amd-3.webp)

**The mechanism, step by step.** With the current context of $t$ tokens:

1. The **draft head proposes $d$ tokens** in one forward pass (the post's configs use $d \in \{3, 5, 8\}$ draft steps).
2. The **target model verifies all $d+1$ positions in one parallel forward pass** — this is the key: verifying $d+1$ candidate positions costs *one* forward pass, not $d+1$, because they are checked in parallel (the decode kernel's head-packing is what makes those extra positions nearly free).
3. **Accept the longest prefix of drafts** that the target agrees with. On the first mismatch, the target's own token for that position is used (one guaranteed-correct "correction" token).
4. Commit the accepted tokens, extend the context, repeat.

Because verification is one forward pass regardless of $d$, every accepted draft beyond the first is a *free* token — you got it without a separate sequential step.

**The math.** Let $\alpha$ be the per-token probability that the target accepts a draft (the "acceptance rate"). If drafts are accepted independently until the first rejection, the expected number of tokens committed per iteration — the $d$ possible draft acceptances plus the one guaranteed correction — is the geometric sum

$$
\mathbb{E}[\text{tokens/iter}] = \frac{1 - \alpha^{\,d+1}}{1 - \alpha},
$$

where $\alpha \in [0,1)$ is the acceptance rate and $d$ is the number of draft steps. *(This is the standard speculative-decoding expectation; the post reports the outcome — ~3.3 tokens/iter — not the formula.)* Each term $\alpha^k$ is the probability that the first $k$ drafts were all accepted; summing them counts the expected committed tokens.

**Worked micro-example.** The post says 3 draft steps ($d=3$) advance **~3.3 tokens per iteration**. Solve $\frac{1-\alpha^4}{1-\alpha} = 3.3$ for $\alpha$: trying $\alpha = 0.85$ gives $\frac{1-0.522}{0.15} = 3.19$, and $\alpha \approx 0.87$ gives $\approx 3.28$. So Inkling's draft head is landing an **~86–87% acceptance rate** on this agentic workload — high, because agentic traffic (tool calls, structured output, ~90% cache-hit prefixes) is unusually predictable. Now convert to throughput. With MTP off, the post measures 152.4 tok/s at 6.6 ms/iter (indeed $1/0.0066 \approx 152$). With MTP on at 317.5 tok/s and 3.3 tokens/iter, each iteration takes $3.3 / 317.5 \approx 10.4$ ms — about 58% longer than the plain 6.6 ms step (the extra draft + wider verify), but it yields 3.3× the tokens, netting the **2.08×** speedup.

| Config (draft steps) | tok/s per user (batch 1) | Speedup vs MTP-off |
| :------------------- | -----------------------: | -----------------: |
| MTP off              | 152.4                    | 1.00×              |
| 3/1/4 (3 steps)      | 317.5                    | 2.08×              |
| 5/1/6 (5 steps)      | 342.5                    | 2.25×              |
| 8/1/9 (8 steps)      | 354.6                    | 2.33×              |

More draft steps help — but with diminishing returns (2.08 → 2.25 → 2.33), because each extra step only pays off if all the earlier drafts were accepted, and $\alpha^k$ decays.

**Why it works / when it fails.** It works when two conditions hold: the draft is accurate (high $\alpha$) *and* the target forward pass has spare compute to absorb the extra verified positions — i.e. the memory-bound, low-batch regime. Both fail as batch grows. The post's per-user decode chart is unusually candid about it:

![Per-user decode speed on the agentic benchmark (B200, NVFP4): MTP delivers up to 2.33× at batch 1 but drops below 1.0× (0.94×, 0.84×) at batch 3 and 4 as the batch itself saturates the GPU. Figure from TokenSpeed / Thinking Machines Lab (CC BY 4.0).](/imgs/blogs/inkling-fp4-inference-nvidia-amd-fig5.webp)

| Batch | MTP off (tok/s/user) | MTP on (tok/s/user) | Speedup |
| ----: | -------------------: | ------------------: | ------: |
| 1     | 152.4                | 317.5               | 2.08×   |
| 2     | 138.5                | 189.4               | 1.37×   |
| 3     | 130.4                | 122.2               | 0.94×   |
| 4     | 122.4                | 102.8               | 0.84×   |

The reason is the same arithmetic-intensity story as the decode kernel. At batch 1 the GPU is memory-bound with idle compute, so verifying $d+1$ positions is nearly free — pure win. By batch 3–4 the batch *already* fills the compute units, so the speculative work is no longer free: rejected drafts are wasted FLOPs that now contend with real work, and the net is a **slowdown**. MTP is a latency optimization for the low-concurrency, latency-sensitive regime (a single user in an agentic loop), not a throughput optimization for a packed server. A serving stack should switch it *off* above a batch threshold — which is exactly the kind of policy the scheduler, sitting above the kernel API, is positioned to make. This is the sharpest practical lesson in the whole post, and it generalizes to any speculative method (see [speculative decoding in production](/blog/machine-learning/model-serving/speculative-decoding-in-production)).

## Experiments & results

Two questions decide whether this stack is real: does FP4 keep the model's quality, and is it actually fast?

**Quality.** The post evaluates all three checkpoints on six suites. Baselines are named (the BF16 reference is the model itself pre-quantization), and the deltas are small:

| Benchmark             | BF16 (ref) | NVFP4 | MXFP4 |
| :-------------------- | ---------: | ----: | ----: |
| GPQA Diamond          | 88.1%      | 86.4% | 85.4% |
| AIME 2026             | 96.4%      | 96.7% | 96.7% |
| BFCL exact calls      | 78.3%      | 78.5% | 79.1% |
| BFCL all-live macro   | 75.4%      | 76.5% | 75.3% |
| MMAU                  | 77.2%      | 76.5% | 76.0% |
| MMMU-Pro (standard10) | 73.6%      | 73.2% | 73.1% |

The worst regression is GPQA Diamond (−1.7 for NVFP4, −2.7 for MXFP4); on the other five suites the FP4 checkpoints are within a point of BF16 or slightly ahead. MXFP4 trails NVFP4 by a hair on most suites — consistent with its coarser 4.25-bit / power-of-two scale — but not enough to matter for most deployments.

**Speed.** On a multi-turn agentic workload (50k+ token contexts, 10–15 turns per conversation, ~90% cache-hit rate), Inkling NVFP4 on **four B200 GPUs** runs at **317 tok/s per user at concurrency 1** with MTP (3 draft steps, ~3.3 tokens/iter). With MTP off it sustains **152 tok/s per user at concurrency 1** (6.6 ms/iter) and **122 tok/s per user at concurrency 4**, where **system throughput reaches 40k tok/s**. On the AMD side, the MXFP4 checkpoint lets the 975B model run on **four MI355X GPUs** with cache headroom for the same 50k+ contexts, and — reusing the *same* MTP path as NVIDIA, untouched at the model layer — early MI355X runs show MTP raising per-user decode speed across a **1.5×–2.4×** range over batch sizes 1–4.

**What's load-bearing in their setup that might not transfer.** Three things prop up the headline numbers. **(1) The ~90% cache-hit rate.** Agentic multi-turn traffic reuses long shared prefixes, which both makes prefill cheap and — more importantly for MTP — makes the next tokens *predictable*, inflating the acceptance rate $\alpha$ toward the 0.87 we backed out. A workload of diverse, cold, single-turn prompts would see a much lower $\alpha$ and a much smaller MTP win. **(2) Concurrency 1.** The flagship 2.33× is a batch-1 number; the honest chart shows it inverting by batch 3. **(3) 50k+ contexts.** Long contexts are what make the decode kernel's KV-streaming and the flat cache's efficiency matter; at 2k contexts these optimizations would barely register. The results are real, but they are results *for the low-concurrency, long-context, high-cache-hit agentic regime* — which, to be fair, is exactly the regime the post targets and names.

## Critique

**What's genuinely strong.** The architecture is the contribution, and it is a good one: putting the vendor boundary at the kernel API is what turns "port to AMD" from a fork into a set of kernels, and the post *demonstrates* the payoff rather than asserting it (two checkpoints, two kernel families, one model layer). The flat cache with heterogeneous views is an elegant, concrete answer to a real fragmentation-versus-padding dilemma, and the reference-counted BlockRef ownership model is the right way to make cross-request sharing safe. Most of all, the post is intellectually honest where it would be easy not to be: it *publishes the chart where its own headline feature loses*, showing MTP dropping to 0.84× at batch 4. That single admission does more for the work's credibility than any speedup number.

**What's weak, unfalsifiable, or cherry-picked.** It is an engineering blog, so the rigor bar is a snapshot, not a paper — and it shows. There are **no error bars or run-to-run variance** on any measurement; a 2.08× that is really $2.08 \pm 0.3$ reads very differently. The AMD numbers are explicitly **"early"** and given as a range ("1.5× to 2.4×") rather than a table, so they are not yet comparable to the NVIDIA results. The quality evals are **six suites**; a skeptic would want perplexity or a long-tail generation eval, because benchmark accuracy can hold while FP4 quietly degrades rare-token fidelity. And the headline **2.33× is a batch-1, ~90%-cache-hit, agentic number** — the most favorable cell in the entire result space.

**What ablation is missing.** The one I most want is **acceptance rate $\alpha$ versus workload**: plot tokens/iter as the cache-hit rate falls from 90% to 50% to cold, and the MTP story becomes falsifiable instead of anecdotal. Second, a **memory-versus-quality sweep** — where does mixing higher-precision layers (shared experts, attention) buy back the GPQA points, and at what memory cost? Third, an **isolation of the decode kernel from MTP**: both are batch-1 wins for the same memory-bound reason, and the end-to-end number entangles them.

**What would change my mind.** If the acceptance-rate-versus-cache-hit ablation showed MTP still clearing, say, 1.5× at a realistic 50% cache-hit rate with cold, diverse prompts, I would upgrade MTP here from "a strong low-concurrency latency trick" to "a general decode win." As it stands, the evidence supports the narrower, honest claim the post actually makes — and I would *lower* my confidence if a follow-up quietly dropped the batch-3/4 chart instead of explaining it.

## What I'd build with this

These are my extrapolations, not claims from the post.

1. **A batch-aware MTP scheduler policy.** The 0.94×/0.84× cliff is a control-plane bug waiting to be fixed: the scheduler (which sits above the kernel API) should turn MTP off — or shrink $d$ — once a request's effective batch crosses the break-even point, and turn it back on when concurrency drops. The data to set that threshold is already in the post.
2. **Adaptive draft depth from live acceptance.** Track a running estimate of $\alpha$ per request and set $d$ to maximize expected tokens/iter given the measured iteration-cost curve. High-$\alpha$ agentic sessions get $d=8$; a session that starts missing gets throttled to $d=3$ or off.
3. **A third heterogeneous view for a fourth state type.** The flat-cache design is parameterized by per-token footprint; adding, say, a compressed-KV or a retrieval-cache view is "pick a tokens-per-slot and register a group," which is a much smaller change than it would be with per-type pools.
4. **Mixed-precision by sensitivity, driven by the quality table.** GPQA's −2.7 on MXFP4 suggests a few sensitive layers dominate the loss; a per-layer precision search that keeps those at FP8 and everything else at FP4 could close most of the gap for a small memory premium.
5. **Port the unified-API discipline to a third backend.** The real test of "the vendor boundary is at the kernel API" is a backend nobody planned for — a TPU or a startup accelerator. If bring-up is genuinely "implement the kernel API," that validates the whole thesis; if the model layer has to change, it doesn't.

## References

- **Source post:** TokenSpeed Team & Thinking Machines Lab, *TML Inkling at Day 0: FP4 Inference on NVIDIA and AMD with TokenSpeed* (LightSeek Foundation engineering blog, Jul 2026), CC BY 4.0 — [lightseek.org/blog/tokenspeed-inkling.html](https://lightseek.org/blog/tokenspeed-inkling.html). All figures credited "TokenSpeed / TML" are reproduced from this post under its license.
- **Jenga (cited by the post):** *Jenga: Effective Memory Management for Heterogeneous LLM Inference* — [arxiv.org/abs/2503.18292](https://arxiv.org/abs/2503.18292) — the community work on separating physical allocation from logical organization that the flat cache echoes.
- **Format specifications** (for the FP4 internals the post names but does not detail): NVIDIA's NVFP4 and the OCP Microscaling (MX) formats, which define E2M1, the 16- and 32-element blocks, and the FP8-E4M3 / E8M0 scales.
- Related on this blog: [quantization for LLM serving](/blog/machine-learning/model-serving/quantization-for-llm-serving) · [speculative decoding in production](/blog/machine-learning/model-serving/speculative-decoding-in-production) · [prefix caching and RadixAttention](/blog/machine-learning/model-serving/prefix-caching-and-radixattention) · [GPU-architecture-specific tuning for LLM serving](/blog/machine-learning/model-serving/gpu-architecture-specific-tuning-for-llm-serving).

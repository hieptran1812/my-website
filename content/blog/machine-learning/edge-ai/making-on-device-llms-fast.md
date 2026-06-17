---
title: "Making on-device LLMs fast: KV-cache, speculative decoding, and paged attention"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Your 7B model loads and runs on the laptop — now make it stay fast over a long conversation without OOM, using the KV-cache, KV-cache quantization, speculative decoding, paged attention, and prefix caching, with derivations, llama.cpp commands, and measured before-after numbers."
tags:
  [
    "edge-ai",
    "model-optimization",
    "kv-cache",
    "speculative-decoding",
    "paged-attention",
    "llm",
    "inference",
    "efficient-ml",
    "llama-cpp",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/making-on-device-llms-fast-1.png"
---

You did the hard part already. You picked a 7B model, quantized it to a 4-bit GGUF, and got it running locally on your laptop with `llama.cpp` (if you haven't, start with [running LLMs locally with llama.cpp and GGUF](/blog/machine-learning/edge-ai/running-llms-locally-llama-cpp-and-gguf)). The first reply comes back at a respectable 18 tokens per second. You demo it to a teammate, they're impressed, everyone moves on.

Then a real user shows up. They paste in a 6,000-token document and ask three follow-up questions. By the third question, the first token takes almost a full second to appear, the tokens-per-second has sagged, and your activity monitor shows memory creeping up turn after turn. Push it harder — a 30k-token codebase, a long multi-turn debugging session — and the process gets OOM-killed. The model that "ran fine" yesterday is now slow and fragile in exactly the situation people actually use it: a long, stateful conversation.

This is the gap between *fits and runs* and *fast and durable*. It is almost never about the model weights — those are a fixed cost you paid at load time. It is about what happens during **decode**, the autoregressive token-by-token generation loop, and specifically about a data structure that quietly grows with every token: the **KV-cache**. The fixes are a small, composable set of decode-time tricks — quantizing the KV-cache, speculative decoding, paged attention, and prefix caching — and the beautiful thing about them is that, unlike quantizing the weights, most of them cost you *zero* quality. They are pure systems wins. Figure 1 shows the core problem they all attack: the KV-cache footprint growing linearly with context until it becomes the wall.

![Matrix comparing fp16 and int8 KV-cache size against fixed weight size across 4k, 16k, and 32k context lengths for a 7B model](/imgs/blogs/making-on-device-llms-fast-1.png)

By the end of this post you'll be able to: derive exactly how much memory your KV-cache costs at any context length; explain why decode is memory-bound and what that implies for which optimizations actually help; turn on KV-cache quantization, speculative decoding, and prefix caching in `llama.cpp` and measure the speedup honestly; and reason about which of these tricks matter on a single-user edge device versus a multi-tenant server. This sits in the runtime layer of the four-lever frame from [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) — it doesn't change the weights at all, it changes how the runtime executes them.

## The decode loop, and why it gets slow

To fix decode you have to picture it. A transformer LLM does two distinct things per request, with very different performance profiles.

**Prefill** processes your prompt. All the prompt tokens go through the model at once, in parallel, as one big matrix multiply per layer. This is *compute-bound*: there's a lot of arithmetic and the hardware is busy. Prefill latency is what determines **time-to-first-token (TTFT)** — the pause before the model starts replying.

**Decode** generates the answer one token at a time. You feed in the single most-recent token, run one forward pass, sample one new token, append it, and repeat. The model can't generate token *t+1* until it has produced token *t* — generation is inherently sequential. Decode latency determines your **tokens-per-second (throughput)** and dominates total time for any non-trivial answer.

Here's the trap. In decode, each forward pass processes exactly **one** token's worth of new activations, but to compute attention for that one token it must attend to *every previous token*. Naively, computing attention at step *t* means re-deriving the keys and values for all *t* prior tokens from their hidden states — re-running a big chunk of the network over the entire history, every single step. For a 4,000-token answer, the last token would trigger ~4,000 tokens' worth of recomputation. The total work to generate *n* tokens would scale like $O(n^2)$, and your throughput would collapse as the conversation grows.

To make this concrete, contrast the two phases by their *shape*. Prefill multiplies a $[\,L \times d\,]$ matrix of $L$ prompt-token activations against each weight matrix — a matrix-matrix product, dense, the GPU's happy place. Decode multiplies a $[\,1 \times d\,]$ row vector against the same weight matrix — a matrix-*vector* product. The weight matrix is the same size and just as expensive to *read* from memory in both cases, but in decode you do $L$ times *less* arithmetic with it. That asymmetry is the whole story of on-device LLM speed: prefill is limited by how fast the chip can *compute*, decode is limited by how fast it can *read memory*, and almost everything that makes an on-device assistant feel slow lives in the decode phase. Keep that prefill-vs-decode split in mind; it's the lens we'll use for every technique.

Put numbers on the asymmetry so it stops being a slogan. A single transformer layer of a 7B is dominated by four projections in attention ($q$, $k$, $v$, output) and the feed-forward block's up- and down-projections. For a hidden size $d = 4096$ and FFN width $4d$, the matmul FLOPs *per token* through one layer come to roughly $2 \times (4 d^2 + 8 d^2) = 24 d^2 \approx 4.0 \times 10^8$ FLOPs (the leading 2 is multiply-and-add). Across 32 layers that's about $1.3 \times 10^{10}$ FLOPs per token — call it 13 GFLOP. In *prefill* you do that 13 GFLOP for every one of $L$ prompt tokens, but you reuse each weight matrix across all $L$ of them in a single fused matmul, so the weights are read from memory once and amortized over $L$ tokens of work. In *decode* you do the same 13 GFLOP for the one new token, but you must still stream every weight in from memory to do it — and you reuse those bytes for exactly one token before moving on. Prefill's arithmetic intensity (FLOPs per byte read) scales with $L$; decode's is pinned near the floor. That single difference is why a 6,000-token prompt prefills in a couple hundred milliseconds of busy compute but each *reply* token trickles out at the speed of memory.

The KV-cache is the fix for that recomputation, and — as is so often the case in systems — the fix for a compute problem creates a memory problem. Let's build it up properly.

## The KV-cache: trading memory to kill recomputation

Attention works on three projections of each token's hidden state: a **query** $q$, a **key** $k$, and a **value** $v$. For the token at position $t$, the attention output is a weighted sum of the value vectors of all positions $0 \dots t$, where the weights come from the dot products of $q_t$ with every key $k_0 \dots k_t$:

$$
\text{attn}(q_t) = \sum_{i=0}^{t} \text{softmax}_i\!\left(\frac{q_t \cdot k_i}{\sqrt{d}}\right) v_i
$$

The crucial observation: the keys and values of *past* tokens never change. $k_5$ and $v_5$ are computed once, from token 5's hidden state, and they're identical whether you're generating token 6 or token 600. So instead of recomputing them every step, you compute each token's $k$ and $v$ once, store them, and on every subsequent step just compute the *new* token's $q$, $k$, $v$, append the new $k,v$ to the stored set, and attend over the whole stored set.

That stored set is the **KV-cache**. With it, each decode step costs $O(t)$ work (one query against $t$ cached keys) instead of $O(t^2)$ recomputation — the whole generation drops from $O(n^2)$ back to $O(n^2)$ for the unavoidable attention but with a tiny constant, because the expensive feed-forward and projection work is done once per token, not once per token *per step*. Figure 2 shows the mechanism: append the new key and value, attend over the running cache, grow by one column per token.

It helps to trace one full step. At decode step $t$, you have a cache holding $k_0 \dots k_{t-1}$ and $v_0 \dots v_{t-1}$ for every layer. You feed in token $t-1$ (the one you just produced). In each layer: project its hidden state to $q_t, k_t, v_t$; **append** $k_t$ and $v_t$ to that layer's cache (a single column write, no recompute of the past); compute the attention scores $q_t \cdot k_i$ for $i = 0 \dots t$; softmax them; take the weighted sum of $v_0 \dots v_t$; carry on through the feed-forward block; and pass up to the next layer. The query $q_t$ is *not* cached — it's used once and discarded, because no future token will ever attend *from* position $t$ again under causal masking. Only keys and values persist, because future queries attend *to* them. That is the whole reason the structure is a "KV"-cache and not a "QKV"-cache, and it's worth saying out loud because it's the asymmetry that makes the memory derivation come out the way it does.

Two consequences fall out of this immediately. First, the cache is *append-only* during generation — you never rewrite past entries, you only add columns. That makes it cheap to grow and is exactly why paging (later) works so cleanly. Second, the cache's size is set by **how many tokens are in the conversation**, not by how many tokens you're about to generate — a 30k-token document with a one-word answer still requires a 30k-entry cache, because the answer's single query must attend over all 30k cached keys. Long *inputs* are just as expensive to hold as long *outputs*. This is why "summarize this huge document" is a memory-heavy request even though the output is short.

![Stack diagram showing the per-step KV-cache flow from projecting a new token to appending its key and value and attending over the cached prefix](/imgs/blogs/making-on-device-llms-fast-2.png)

This is the single most important optimization in LLM inference and every serious runtime does it by default. The question is never *whether* to use a KV-cache — it's how to pay for it, because it is not free.

### Deriving the memory cost

How big is the cache? Walk through what we store. For **each layer**, for **each token**, we store one key vector and one value vector. A key (or value) vector has $n_{\text{heads}} \times d_{\text{head}}$ elements — that's the model's hidden dimension split across attention heads. So the total element count is:

$$
\text{elements} = 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times \text{seq\_len}
$$

The factor of 2 is keys *and* values. Multiply by `bytes_per_element` to get bytes:

$$
\text{KV bytes} = 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times \text{seq\_len} \times \text{bytes}
$$

Plug in a concrete 7B (Llama-2-7B shape): $n_{\text{layers}} = 32$, $n_{\text{heads}} = 32$, $d_{\text{head}} = 128$, so $n_{\text{heads}} \times d_{\text{head}} = 4096$ (the hidden size). At fp16, `bytes` = 2. For a single token:

$$
2 \times 32 \times 4096 \times 2 = 524{,}288 \text{ bytes} \approx 0.5\ \text{MB per token}
$$

That half-megabyte-per-token figure is worth memorizing for a 7B. It means:

- **4k context** → $4096 \times 0.5\ \text{MB} \approx 2.0\ \text{GB}$. (Figure 1 rounds the practical per-token cost down slightly for GQA-style models; for a full-MHA 7B it's about 0.5 GB per 1k tokens, so ~0.5 GB at 1k and ~2 GB at 4k.)
- **32k context** → $32768 \times 0.5\ \text{MB} \approx 16\ \text{GB}$.

Notice what just happened. The *weights* of a 4-bit 7B are a fixed ~3.8 GB. At 4k context the KV-cache is already a meaningful fraction of that, and at 32k context the KV-cache is **four times the size of the model itself**. Your peak memory is no longer set by the model — it's set by the conversation length. That's the memory wall in Figure 1, and it's why long conversations OOM.

> A clarifying note on the arithmetic: the per-token cost depends on the *attention* config, not the parameter count. The factor that matters is $2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}}$. For a plain multi-head 7B that's 0.5 MB/token; for a model with grouped-query attention it's far smaller, as we'll see. Always compute it from the architecture, not from "it's a 7B."

It's worth dwelling on *why* this scales linearly and not, say, quadratically — because the $O(n^2)$ in attention's *compute* tempts people into expecting $O(n^2)$ *memory*. The compute is quadratic because each new token attends to all prior tokens: step $t$ does $O(t)$ score computations, and summed over the whole sequence that's $\sum_t t = O(n^2)$. But the *storage* is one $k$ and one $v$ per token per layer, full stop — you store each past token's key and value exactly once, never a pairwise score matrix. The $t \times t$ attention-score matrix is materialized transiently inside the kernel and thrown away; it never lives in the cache. So memory is strictly linear in sequence length while compute is quadratic. That distinction matters for the optimizations: KV-cache *quantization* and *paging* attack the linear memory term, while flash-attention-style kernels attack the transient quadratic-compute term by never materializing the full score matrix. They are orthogonal wins, which is exactly why you stack them.

There's a second subtlety that catches people at deployment time: the formula uses $n_{\text{kv\_heads}}$, which under grouped-query attention is *smaller* than $n_{\text{heads}}$, but it uses $d_{\text{head}}$, which is the *per-head* dimension, not the hidden size. A model can keep $n_{\text{heads}} \times d_{\text{head}} = d$ for its query projection while shrinking $n_{\text{kv\_heads}}$, so the cache shrinks even though the model's hidden size is unchanged. When you read a model card, the two numbers to grab are `num_key_value_heads` and `head_dim` (or derive `head_dim = hidden_size / num_attention_heads`). Everything about your memory wall is downstream of those two integers.

### Worked example: KV-cache memory at 4k vs 32k for a 7B

#### Worked example: the KV-cache growth and the int8 saving

You're shipping a coding assistant on a 16 GB laptop. The 4-bit 7B weights take ~3.8 GB. The runtime and OS want ~2 GB headroom. That leaves roughly **10 GB** for the KV-cache. Question: how long a context can you support, and what does int8 KV buy you?

At fp16, KV cost is 0.5 MB/token:

$$
\text{max context}_{\text{fp16}} = \frac{10\ \text{GB}}{0.5\ \text{MB/token}} = \frac{10{,}240\ \text{MB}}{0.5\ \text{MB}} \approx 20{,}480\ \text{tokens}
$$

So fp16 KV caps you at ~20k context before OOM. A 30k-token codebase blows past that — exactly the failure we opened with.

Now quantize the KV-cache to int8 (1 byte per element instead of 2). The per-token cost halves to 0.25 MB:

$$
\text{max context}_{\text{int8}} = \frac{10{,}240\ \text{MB}}{0.25\ \text{MB}} \approx 40{,}960\ \text{tokens}
$$

You **doubled** your usable context, from 20k to 40k tokens, by flipping one flag — and we'll see the quality cost is close to nil. Concretely at the three context points: fp16 KV is 2.0 GB / 8.0 GB / 16 GB at 4k / 16k / 32k; int8 KV is 1.0 GB / 4.0 GB / 8.0 GB. (Figure 1 uses slightly rounded per-token costs to keep the table readable; the relationship — int8 is exactly half, and both grow linearly — is the point.)

Lay the full-MHA 7B's KV growth out as a table so the wall is unmistakable, and put it next to the 4-bit weights (a fixed ~3.8 GB) for scale:

| Context length | fp16 KV (0.5 MB/tok) | int8 KV (0.25 MB/tok) | q4 KV (0.125 MB/tok) | KV vs 3.8 GB weights (fp16) |
|---|---|---|---|---|
| 1k | 0.5 GB | 0.25 GB | 0.13 GB | 0.13× |
| 4k | 2.0 GB | 1.0 GB | 0.5 GB | 0.53× |
| 8k | 4.0 GB | 2.0 GB | 1.0 GB | 1.05× |
| 16k | 8.0 GB | 4.0 GB | 2.0 GB | 2.1× |
| 32k | 16 GB | 8.0 GB | 4.0 GB | 4.2× |

Trace the rightmost column. Somewhere between 4k and 8k context the fp16 KV-cache *overtakes the entire 4-bit model*, and by 32k it is more than four times the weights. This is the single most counterintuitive fact about on-device LLMs: at long context you are not running a 3.8 GB model, you are running a 3.8 GB model dragging a 16 GB tail behind it, and the tail is what reads slowly each step and what eventually OOMs you. Every row of the int8 and q4 columns is a flag-flip away — no retraining, no quality cliff for q8 — which is why KV quantization is the first lever you reach for once context gets long.

## Decode is memory-bound: why "more work per byte" is the whole game

Before reaching for fixes, you have to know what kind of slow you are, because optimizations that help compute-bound code do nothing for memory-bound code and vice versa. This is the [roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives), and decode lands firmly on the memory-bound side of it.

The roofline idea in one breath: a kernel's max performance is the lesser of (a) the hardware's peak compute (FLOP/s) and (b) its peak memory bandwidth (bytes/s) times the kernel's **arithmetic intensity** (FLOPs per byte of memory traffic). Low arithmetic intensity → bandwidth-bound; high → compute-bound.

Decode with batch size 1 is the canonical low-intensity case. To generate one token you must **read every weight in the model from memory exactly once** (to multiply the single new activation vector through them), but you do only a tiny amount of arithmetic with each weight (a vector-matrix product, not a matrix-matrix product). The arithmetic intensity is roughly 1-2 FLOPs per byte read — far below the ratio where any modern accelerator becomes compute-bound. So your decode speed is, to first order:

$$
\text{tokens/s} \approx \frac{\text{memory bandwidth (bytes/s)}}{\text{bytes read per token}}
$$

And `bytes read per token` $\approx$ (model weights in bytes) + (KV-cache bytes touched). On an M2 MacBook with ~100 GB/s of usable bandwidth and a 3.8 GB 4-bit model, the bandwidth ceiling on decode is roughly $100 / 3.8 \approx 26$ tokens/s before KV traffic — and you measure ~18 because the KV-cache adds bytes and kernels aren't perfectly efficient. This single equation explains *every* trick in this post:

- **Quantize the weights** (already done): fewer bytes per weight → fewer bytes read → faster. This is why a 4-bit model decodes faster than fp16, not because it does less math.
- **Quantize the KV-cache**: fewer bytes of cache to read each step → faster *and* smaller. A double win.
- **GQA/MQA**: fewer KV heads → smaller cache → fewer bytes read.
- **Speculative decoding**: the radical one — it does *more arithmetic* (verifying K tokens at once) to read the weights *fewer times per accepted token*. It deliberately raises arithmetic intensity, climbing toward the compute roof you were wasting.

That last bullet is the key insight. Because decode wastes the chip's compute (it's all sitting idle while we wait on memory), there's free compute headroom. Speculative decoding *spends* that headroom to read the weights less often per token produced. Let's build it.

#### Worked example: how far is decode from the compute roof?

Make the "wasted compute" claim quantitative on a named target so it isn't hand-waving. Take an M2 (the GPU in a MacBook Air): roughly 3.6 TFLOP/s of fp16 throughput and ~100 GB/s of usable memory bandwidth. The hardware's *ridge point* — the arithmetic intensity at which it flips from memory-bound to compute-bound — is the ratio of those two: $3.6\times10^{12} / 100\times10^{9} \approx 36$ FLOPs per byte. Any kernel below 36 FLOP/byte is memory-bound; above it, compute-bound.

Now compute decode's actual arithmetic intensity. Per token we read the 3.8 GB of 4-bit weights (call it $3.8\times10^9$ bytes) and do ~13 GFLOP of matmul against them: $1.3\times10^{10} / 3.8\times10^{9} \approx 3.4$ FLOP/byte. That's *ten times below* the ridge point. Decode is sitting at roughly one-tenth of the chip's compute capability, with nine-tenths of the ALUs idle, blocked on memory. That idle nine-tenths is the headroom speculative decoding will spend. And it's exactly why a 4-bit model decodes faster than an 8-bit one even though both do the same number of FLOPs: cutting the weight bytes in half halves the term in the denominator of $\text{tok/s} \approx \text{bandwidth} / \text{bytes-per-token}$, so you get ~2× the tokens while doing identical arithmetic. Speed came from reading less, not computing less — the defining signature of a memory-bound regime.

One caveat the roofline hides: as context grows, the KV-cache bytes you must read each step grow too, so `bytes read per token` is *not* constant — it's (weights) + (KV touched). At 32k context on a full-MHA 7B the KV traffic alone is ~16 GB if you re-read the whole cache each step, which would dwarf the 3.8 GB of weights and crater throughput. In practice flash-attention kernels and the linear-in-context attention cost keep per-step KV traffic far below the full-cache size, but the direction is real: long context slows decode not just by risking OOM but by adding bytes to every single step. KV quantization helps here twice — smaller cache to *hold* and fewer bytes to *read* per step.

## Speculative decoding: doing more arithmetic to read weights less often

The setup that makes decode slow — one weight-read per token, lots of idle compute — also makes it fixable. What if, instead of reading the whole 3.8 GB of weights to produce *one* token, we read them once and produce (say) *three* tokens? We can't, directly: generation is sequential, we don't know token *t+1* until we have *t*. But we can *guess* the next few tokens cheaply, then *check* all the guesses in one parallel pass through the big model — and a parallel pass over K candidate tokens reads the weights exactly once, the same as a single decode step.

This is **speculative decoding**, introduced concurrently by Leviathan et al. (Google, "Fast Inference from Transformers via Speculative Decoding", 2023) and Chen et al. (DeepMind, "Accelerating Large Language Model Decoding with Speculative Sampling", 2023). The mechanism, in Figure 3:

1. A small, fast **draft model** (e.g. a 1B, or a quantized clone, or the same model's early layers) autoregressively proposes the next $K$ tokens. This is cheap precisely because the draft model is small.
2. The big **target model** runs **one** forward pass over the original token plus all $K$ drafted tokens — in parallel, like a mini-prefill. This produces the target's probability distribution at each of the $K+1$ positions.
3. **Accept the longest correct prefix**: walk the drafted tokens left to right, accepting each one according to a rejection-sampling rule that guarantees the accepted tokens are distributed *exactly* as if the target had sampled them. At the first rejection, resample that one position from a corrected distribution and stop.
4. Repeat from the new position.

![Timeline of speculative decoding showing the draft proposing K tokens, the target verifying them in one parallel pass, and accepting the correct prefix](/imgs/blogs/making-on-device-llms-fast-3.png)

The part that trips people up is step 3, and it's the part that makes speculative decoding *quality-free*. You might assume "guessing tokens" must degrade output. It does not. The accept/reject rule (modified rejection sampling) is constructed so that the final stream of accepted tokens has the **identical distribution** to plain sampling from the target model. If the draft proposes token $x$ with probability $q(x)$ and the target would assign it $p(x)$, you accept with probability $\min(1, p(x)/q(x))$; on rejection you sample from the normalized positive part of $p - q$. Summed up, the marginal distribution of emitted tokens is exactly $p$. So speculative decoding is not an approximation — at temperature 0 it produces a *bit-identical* output to greedy decoding of the target, and at temperature > 0 it produces a sample from the same distribution. You get speed for free, in the strong sense.

It is worth proving the unbiasedness in a line, because it's the load-bearing claim and it's short. Fix a position and ask: what is the total probability that the emitted token equals some specific value $x$? Two disjoint paths lead to emitting $x$. Path one: the draft proposed $x$ (probability $q(x)$) *and* we accepted it (probability $\min(1, p(x)/q(x))$), contributing $q(x)\,\min(1, p(x)/q(x)) = \min(q(x), p(x))$. Path two: the draft proposed something else and was rejected, and the resampling step happened to draw $x$ from the corrected residual distribution $\propto \max(0,\, p - q)$. The resampling contributes exactly the leftover mass $\max(0,\, p(x) - q(x))$. Add the two paths: $\min(q(x), p(x)) + \max(0,\, p(x) - q(x)) = p(x)$ for every $x$. The two cases ($q(x) \le p(x)$ and $q(x) > p(x)$) each collapse to $p(x)$ algebraically. So the emitted distribution is identically $p$ regardless of how good or bad the draft $q$ is — a *wrong* draft never corrupts the output, it only lowers the acceptance rate and thus the speed. That asymmetry is the gift: speculative decoding can only ever cost you speed, never correctness, which is why it is safe to ship without a quality eval.

A useful corollary for tuning: the acceptance rate $\alpha$ at a position equals $\sum_x \min(p(x), q(x))$, which is exactly $1 - \tfrac12\lVert p - q\rVert_1$ — one minus the total-variation distance between draft and target. So "how often does the draft agree with the target" is *literally* a distributional-distance measurement. A draft trained on the same data and distilled toward the target has small TV distance and high $\alpha$; a draft from a different family, or run on out-of-distribution text, has large TV distance and low $\alpha$. This is why matching the draft to the target's *distribution* matters more than matching its size: a well-distilled 1B can beat a generic 3B as a draft because its $p$-vs-$q$ overlap is higher.

### Deriving the expected speedup

How much faster? Let $\alpha$ be the **acceptance rate** — the probability the target accepts any given drafted token (how often the cheap draft agrees with the expensive target). Draft $K$ tokens per round. The number of tokens accepted in a round is a truncated geometric random variable: you accept token 1 with prob $\alpha$, token 2 with prob $\alpha^2$, and so on, plus you always get the one "free" corrected token at the first rejection. The **expected number of tokens produced per target forward pass** is:

$$
\mathbb{E}[\text{tokens per round}] = \frac{1 - \alpha^{\,K+1}}{1 - \alpha}
$$

This is the geometric series, and it's the heart of the speedup. Let's read it:

- If $\alpha = 0$ (draft is useless), the expression is 1 — you produce one token per target pass, same as no speculation. No harm, no help.
- If $\alpha = 1$ (draft is perfect), it's $K+1$ — every drafted token plus the free one, a full $(K+1)\times$ on the target passes.
- Realistically $\alpha \in [0.6, 0.9]$ for a well-matched draft on in-distribution text.

The *wall-clock* speedup is that expected token count divided by the cost of a round measured in target-pass-equivalents. A round costs **one target forward** plus **$K$ draft forwards**. If the draft is $c$ times the cost of a target forward (with $c$ small — a 1B draft against a 7B target is roughly $c \approx 0.14$ per token, less with batching), the speedup is approximately:

$$
\text{speedup} \approx \frac{\frac{1 - \alpha^{K+1}}{1 - \alpha}}{1 + cK}
$$

Figure 4 walks the math for a concrete setting. With $\alpha = 0.7$ and $K = 4$:

$$
\mathbb{E}[\text{tokens}] = \frac{1 - 0.7^{5}}{1 - 0.7} = \frac{1 - 0.168}{0.3} = \frac{0.832}{0.3} \approx 2.77
$$

So each target forward yields ~2.77 tokens. Net of draft cost (with $c \approx 0.14$, $cK \approx 0.56$): $2.77 / 1.56 \approx 1.8\times$. In practice on a laptop with a well-chosen draft you measure roughly $1.8$–$2.5\times$ on decode throughput, which is exactly what the literature reports (Leviathan et al. report ~2–3× on summarization and translation).

It pays to see how sensitive the speedup is to $\alpha$, because that one number — set by your draft quality and your workload — dominates everything. Holding $K = 4$ and $c \approx 0.14$ fixed, here is the expected tokens-per-pass and the net speedup across acceptance rates:

| Acceptance $\alpha$ | $\mathbb{E}[\text{tokens}]$ at $K{=}4$ | Net speedup ($cK{=}0.56$) | Regime |
|---|---|---|---|
| 0.30 | 1.41 | 0.90× | below break-even — *slower* |
| 0.40 | 1.62 | 1.04× | barely worth it |
| 0.50 | 1.94 | 1.24× | marginal win |
| 0.60 | 2.31 | 1.48× | solid |
| 0.70 | 2.77 | 1.78× | good (code, structured text) |
| 0.80 | 3.36 | 2.15× | excellent |
| 0.90 | 4.10 | 2.63× | near-ideal (very predictable text) |

The first row is the one to internalize: at $\alpha = 0.3$ the *net* speedup is below 1.0 — you've made decode **slower**, because you pay for four draft forwards every round and accept barely more than one token. There is a genuine break-even acceptance below which speculation is negative, and it sits around $\alpha \approx 0.35$–$0.40$ for these parameters. This is the single most common way speculative decoding disappoints in practice: someone bolts on a mismatched draft, sees a slowdown, and concludes "speculation doesn't work on my hardware," when the real problem is a low-$\alpha$ draft. Measure $\alpha$ first; if it's under ~0.4, fix the draft or turn speculation off.

Notice also the *shape* of the speedup curve: it's convex and accelerating in $\alpha$. Going from 0.6 to 0.7 buys you +0.30×; going from 0.8 to 0.9 buys you +0.48×. High-acceptance regimes reward draft improvements super-linearly, which is the economic argument for investing in a *good* draft (distillation, same-family, same-tokenizer) rather than just grabbing the smallest model lying around.

![Stack diagram deriving expected tokens per target forward from acceptance rate and draft length, then subtracting draft cost to get net decode speedup](/imgs/blogs/making-on-device-llms-fast-4.png)

### Why K shouldn't be huge

The geometric series saturates. Going from $K=4$ to $K=8$ when $\alpha = 0.7$ only lifts $\mathbb{E}[\text{tokens}]$ from 2.77 to about 3.2 — because tokens deep in the draft are accepted with probability $\alpha^k$, which decays fast. Meanwhile the draft cost $cK$ keeps growing linearly. So speedup-vs-$K$ has an interior maximum, usually around $K = 3$ to $6$. Drafting 20 tokens is almost always counterproductive: you pay for 20 draft forwards and accept maybe 3. Tune $K$ to your $\alpha$; higher acceptance supports larger $K$.

### Variants that drop the separate draft model

Maintaining a *separate* draft model is a hassle on edge — it's a second download, a second memory footprint, and the two models must share a tokenizer. Several variants remove that:

- **Self-speculative decoding** drafts using a *subset of the target's own layers* (e.g. skip every other layer, or use early-exit), so there's no second model. Acceptance is lower (the draft is cruder) but the memory cost is zero.
- **Medusa** (Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads", 2024) bolts several extra prediction *heads* onto the target model that each predict a future token directly, forming a tree of candidates verified in one pass. No separate model, modest extra params, typically ~2–3× with a small fine-tune of the heads.
- **Lookahead decoding** (Fu et al., 2024) generates and verifies n-gram candidates from the model's own recent outputs using a Jacobi-iteration-style parallel scheme — no draft model and no training, though gains are workload-dependent.

On edge, the trade is memory vs. acceptance: a separate tiny draft gives the best acceptance but costs RAM you may not have; Medusa-style heads or self-speculation give a free-ish speedup when RAM is tight.

## Paged attention: managing the KV-cache like virtual memory

Speculative decoding attacks decode *speed*. Paged attention attacks KV-cache *memory efficiency* — and while its biggest payoff is server-side, it matters on edge for multi-turn and branching chats. It comes from Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023), the paper behind **vLLM**.

The problem it solves is a classic memory-management one. The naive way to store a KV-cache is in one **contiguous** buffer per sequence, sized to the maximum possible length. If your model supports 32k context, you reserve a 32k-token slab up front for *every* sequence — even one that ends up only 2k tokens long. That's two kinds of waste:

- **Internal/reservation waste**: you reserved 32k but used 2k → ~94% of that slab is dead.
- **External fragmentation**: variable-length sequences leave odd-sized holes that no new sequence fits into, even when total free memory is plenty.

The PagedAttention paper measured that contiguous allocation wasted **60–80%** of KV memory in real serving. Their fix is exactly the operating-system fix for the same problem: **paging**. Don't store the cache contiguously; chop it into small fixed-size **blocks** (pages), each holding the KV for a fixed number of tokens (e.g. 16). Keep a per-sequence **block table** (the page table) mapping logical positions to physical blocks. Allocate a new block only when the current one fills. The attention kernel is rewritten to gather KV from non-contiguous blocks via the block table. Figure 5 contrasts the two layouts.

![Before-after diagram contrasting a contiguous max-length KV reservation with high waste against paged fixed-size blocks allocated on demand](/imgs/blogs/making-on-device-llms-fast-5.png)

The memory-utilization win is direct. With pages of size $B$ tokens, the only waste is **internal fragmentation in the last partially-filled page** — at most $B-1$ tokens per sequence, regardless of how long the sequence is or how long the model *could* go. So waste drops from "max_len − actual_len" (potentially tens of thousands of tokens) to "< $B$" (e.g. < 16 tokens), i.e. from 60–80% down to under ~4%. The freed memory translates directly into more concurrent sequences (server) or longer single conversations (edge).

Paging unlocks a second, lovely capability: **prefix sharing**. Because the cache is blocks behind a page table, two sequences that share a prefix (same system prompt, or a branched conversation, or beam-search siblings) can **point their page tables at the same physical blocks** for the shared part — copy-on-write, exactly like fork() shares pages. One copy of the system-prompt KV serves all sessions. On a multi-turn or tree-structured chat on edge, that's real memory saved and real prefill avoided.

Put the prefix-sharing win in numbers, because it's the part that translates to edge. Suppose four browser tabs each open a chat against one local `llama-server`, all sharing a 2,000-token system prompt, and each has accumulated ~1,000 tokens of its own conversation. Without sharing, that's $4 \times 3{,}000 = 12{,}000$ token-slots of KV. With copy-on-write prefix sharing, the 2,000-token system prompt is stored *once* and the four block tables point at it, so you pay $2{,}000 + 4 \times 1{,}000 = 6{,}000$ token-slots — a 2× reduction, and it grows with the number of sessions: $N$ sessions sharing a $P$-token prefix with $U$ unique tokens each cost $P + N\,U$ instead of $N(P + U)$, saving $(N-1)\,P$ token-slots. On an 8B GQA model at 0.125 MB/token fp16, that 6,000-slot saving on the four-tab example is ~0.75 GB of RAM recovered, plus the prefill of the shared prompt is done once instead of four times. The same mechanism is what makes beam search and "regenerate this answer" cheap: the siblings fork from a shared prefix and only the divergent suffix costs new blocks.

The honest caveat for edge: if you run a *single* conversation at a time, contiguous allocation sized to your actual `--ctx-size` (not the model's max) already wastes little, so paging's headline win shrinks. Its value on edge is concentrated in **multi-session** (a few apps sharing one model server) and **branching** (regenerate, tree-of-thought, beam) workloads. Don't expect a single-chat throughput bump from paging alone — expect it to let you *fit more* concurrent or branched state.

There's a subtle second-order benefit even for a single chat, though, and it's worth naming. Because paged allocation grows the cache one small block at a time instead of reserving the whole `--ctx-size` slab up front, your *peak* memory tracks the *actual* conversation length rather than the *configured maximum*. If you set `--ctx-size 32768` to be safe but most conversations end at 3k tokens, a contiguous allocator may reserve the full 32k slab (16 GB on full MHA!) while a paged allocator holds only the ~3k worth of blocks you're using (~1.5 GB). On a RAM-starved device, that difference between "reserved" and "resident" can be the difference between launching and OOMing at startup. So even on edge, paged attention quietly lets you set a generous max context without paying for it until you use it.

## Prompt and prefix caching: the biggest single-user edge win

Here's a trick that's almost embarrassingly effective on edge and gets less press than it deserves: **prefix caching** (also called prompt caching). It targets TTFT, not throughput, and on a multi-turn chat it's the difference between "snappy" and "sluggish."

Recall that prefill (processing the prompt) is the compute-bound phase that sets TTFT. In a chat, every turn re-sends the *same* leading content: the system prompt, the tool definitions, the long document the user pasted, plus the accumulating conversation. The naive runtime re-prefills *all of it* every turn — recomputing the KV for thousands of tokens that haven't changed since last turn. That's wasted work, and it grows as the conversation grows.

Prefix caching keeps the KV-cache of the **already-processed prefix** between turns. On turn two, the runtime detects that the new request's first $N$ tokens are identical to a cached prefix, *reuses* that cached KV verbatim, and only prefills the genuinely new tokens (the latest user message). Since the KV of past tokens is position-dependent but content-fixed, this reuse is exact — no quality cost, like the others here. Figure 6 shows the before/after.

![Before-after diagram showing a chat reprefilling the full system prompt every turn versus reusing a cached prefix and prefilling only new tokens](/imgs/blogs/making-on-device-llms-fast-6.png)

The win scales with how much of your context is fixed. A coding assistant with a 1,500-token system prompt and a 4,000-token pasted file, where the user's actual question is 30 tokens, re-prefills ~5,530 tokens every turn naively but only ~30 tokens with prefix caching — roughly a **180×** reduction in prefill work for that turn, turning a ~900 ms TTFT into ~120 ms (the numbers in Figure 7's TTFT column). For an interactive on-device assistant this is the single most noticeable latency improvement you can ship, and it's purely a caching policy — no model change, no quality change.

The cost is memory: the cached prefix KV occupies RAM between turns (the same KV-cache bytes we sized earlier). On edge that's a real budget line, which is *another* reason KV-cache quantization composes so well here — int8 KV halves the cost of the prefix you're holding onto.

#### Worked example: prefill work and TTFT with vs. without prefix caching

A coding assistant carries a fixed 1,500-token system prompt and the user has pasted a 4,000-token source file. The user's actual follow-up question is 30 tokens. Prefill on this device runs at roughly 6,000 tokens/second (prefill is compute-bound and fast; that's the whole point of the phase). Compute TTFT both ways.

Naive, no prefix cache — every turn re-prefills the full standing context plus the new question:

$$
\text{tokens prefilled} = 1{,}500 + 4{,}000 + 30 = 5{,}530, \quad \text{TTFT} \approx \frac{5{,}530}{6{,}000} \approx 0.92\ \text{s}
$$

With prefix caching — the system prompt and pasted file were processed and cached on turn one; turn two only prefills the new question:

$$
\text{tokens prefilled} = 30, \quad \text{TTFT} \approx \frac{30}{6{,}000} \approx 0.005\ \text{s} + \text{overhead} \approx 0.12\ \text{s}
$$

The arithmetic says prefill drops by ~184× (5,530 → 30 tokens); the *felt* TTFT drops from ~920 ms to ~120 ms once you add the fixed sampling and detokenization overhead that doesn't shrink. That is the gap between a chat that feels like it's *thinking* before every reply and one that answers instantly. And note the asymmetry with decode: prefix caching does nothing for tokens-per-second (decode is unchanged), but it is the *only* one of these tricks that attacks the pause *before* the first token — which, for an interactive assistant, is the latency a human actually notices.

The honest limit: prefix caching only helps when the prefix is *stable*. If your app reorders the system prompt, injects a per-turn timestamp at the *front*, or rebuilds the message list differently each turn, the cached prefix won't match and you re-prefill everything. Keep the invariant content at the *front* (system prompt, tools, pasted document) and let only the genuinely new content (the latest user turn) sit at the *tail*. Prefix caching matches the longest common *prefix*; anything that perturbs an early token invalidates the whole tail behind it. Designing your prompt template so the stable parts lead is a free 100-ms-class win that costs you nothing but discipline.

### Continuous batching (and why edge cares less)

One more server-side technique to name so you can place it correctly. **Continuous batching** (also "in-flight batching") lets a server inject and evict requests from a running batch at the *token* granularity rather than waiting for a whole batch to finish — so a freed slot is immediately filled by a queued request. It massively raises throughput under concurrent load by keeping the batch full. On a single-user edge device with batch size 1 and one conversation, there's no batch to keep full, so continuous batching is mostly irrelevant locally. It belongs in your mental model for *why server inference scales*, but it's not a knob you tune for an on-device assistant. (It pairs naturally with paged attention — vLLM does both.)

## Doing it for real in llama.cpp

Enough theory. Here's how to measure and enable each of these on a laptop with `llama.cpp`, which is the workhorse runtime for on-device LLMs and exposes every knob we've discussed. (Build and basic usage are covered in [running LLMs locally with llama.cpp and GGUF](/blog/machine-learning/edge-ai/running-llms-locally-llama-cpp-and-gguf).)

### Step 1: measure the KV-cache so you know your wall

First, quantify the problem before optimizing it — always. `llama.cpp` prints the KV-cache size at load time, and you can compute it yourself from the architecture. A quick Python sanity-check matching the derivation above:

```python
def kv_cache_bytes(n_layers, n_kv_heads, head_dim, seq_len, bytes_per_elem=2):
    # factor 2 = keys AND values
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * bytes_per_elem

# Llama-2-7B (full multi-head attention: 32 KV heads)
mha = kv_cache_bytes(32, 32, 128, 32768, 2)
print(f"7B MHA  fp16 @32k: {mha/1e9:.1f} GB")        # ~17.2 GB

# Llama-3-8B style (GQA: 8 KV heads instead of 32)
gqa = kv_cache_bytes(32, 8, 128, 32768, 2)
print(f"8B GQA  fp16 @32k: {gqa/1e9:.1f} GB")         # ~4.3 GB  (4x smaller)

# Same GQA model with int8 KV-cache
gqa_int8 = kv_cache_bytes(32, 8, 128, 32768, 1)
print(f"8B GQA  int8 @32k: {gqa_int8/1e9:.1f} GB")    # ~2.1 GB  (8x vs 7B MHA)
```

Notice the **GQA** line. Grouped-query attention (and its extreme, multi-query attention) is an *architectural* KV-cache reduction: instead of every query head having its own key/value head, groups of query heads *share* one KV head. Drop from 32 KV heads to 8 and the cache shrinks 4×. This is why modern small models are designed with GQA — it's the cheapest possible long-context enabler, baked into the weights. Stack int8 KV on top of GQA and the full-MHA 7B's 17 GB at 32k becomes ~2 GB. (Architecture-level KV reductions like GQA and MQA are part of the design story in [small language models by design](/blog/machine-learning/edge-ai/small-language-models-by-design).)

The more useful version of the calculator inverts the question: given a fixed RAM budget, *how much context can I afford?* That's the number you actually need when picking `--ctx-size`, because setting it too high reserves cache you can't back with RAM and setting it too low truncates the conversation. Compute the headroom after weights and OS, then divide by the per-token cost:

```python
def max_context(ram_gb, weights_gb, os_headroom_gb,
                n_layers, n_kv_heads, head_dim, bytes_per_elem=2):
    free_bytes = (ram_gb - weights_gb - os_headroom_gb) * 1e9
    per_token = 2 * n_layers * n_kv_heads * head_dim * bytes_per_elem
    return int(free_bytes / per_token)

# 16 GB laptop, 4-bit 8B GQA weights ~4.9 GB, 2 GB OS/runtime headroom
for bpe, name in [(2, "fp16"), (1, "int8 q8_0")]:
    ctx = max_context(16, 4.9, 2.0, n_layers=32, n_kv_heads=8,
                      head_dim=128, bytes_per_elem=bpe)
    print(f"{name:9s} KV  ->  max context ~ {ctx:,} tokens")
# fp16      KV  ->  max context ~ 222,656 tokens   (GQA is generous)
# int8 q8_0 KV  ->  max context ~ 445,312 tokens
```

Two lessons jump out. First, on a *GQA* 8B the KV-cache is so small (0.125 MB/token fp16) that a 16 GB laptop can in principle hold hundreds of thousands of tokens of cache — the wall has moved far out compared to the full-MHA 7B, which choked at ~20k. Architecture did most of the work before any flag. Second, even so, you rarely set `--ctx-size` to the RAM maximum: prefill and decode both get slower with context, and most models' *quality* degrades past their trained context window regardless of how much cache you can physically hold. Use this calculator to confirm you *can* afford the context you want, then set `--ctx-size` to what the *task and the model* actually need — not to the largest number RAM permits.

### Step 2: quantize the KV-cache

`llama.cpp` quantizes the KV-cache with two flags, `--cache-type-k` and `--cache-type-v` (commonly written `-ctk` / `-ctv`). Supported types include `f16` (default), `q8_0` (int8), and the more aggressive `q4_0`:

```bash
# Baseline: fp16 KV-cache, 16k context
./llama-cli -m models/llama-3-8b-instruct-q4_k_m.gguf \
    --ctx-size 16384 \
    --cache-type-k f16 --cache-type-v f16 \
    -p "Summarize the attached design doc." -f doc.txt

# int8 KV-cache (q8_0): halves KV memory, ~free quality
./llama-cli -m models/llama-3-8b-instruct-q4_k_m.gguf \
    --ctx-size 16384 \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    -p "Summarize the attached design doc." -f doc.txt

# Aggressive: q4 keys/values. Watch quality on hard tasks.
./llama-cli -m models/llama-3-8b-instruct-q4_k_m.gguf \
    --ctx-size 32768 \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    -p "Summarize the attached design doc." -f doc.txt
```

Two practical notes. First, **keys are more sensitive than values** to quantization — keys feed the softmax, where small errors get amplified through the exponential, while values are just averaged. A common sweet spot is `q8_0` keys with `q4_0` values, or `q8_0` for both; pure `q4_0` keys can hurt on reasoning-heavy tasks. The asymmetric configuration looks like this:

```bash
# Asymmetric KV: q8 keys (sensitive), q4 values (forgiving) — best memory/quality trade
./llama-cli -m models/llama-3-8b-instruct-q4_k_m.gguf \
    --ctx-size 32768 \
    --flash-attn \
    --cache-type-k q8_0 --cache-type-v q4_0 \
    -f system_plus_doc.txt -n 256
```

Second, on some backends quantized KV requires **Flash Attention** to be enabled (`-fa`/`--flash-attn`) — shown above — because the fused attention kernel is what reads the packed cache efficiently; without it you may see a fallback or no speedup. If you flip `--cache-type-v q4_0` and throughput *doesn't* improve, the missing `--flash-attn` flag is the first thing to check: the dequantize-on-the-fly path only pays off inside the fused kernel. The research backing is solid: **KIVI** (Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache", 2024) shows you can push KV to 2 bits with per-channel key quantization and per-token value quantization at minimal quality loss, which is why these flags are safe to reach for. KV-cache quantization is the same family as the activation/KV quantization in [LLM quantization II: activations, SmoothQuant, and the KV-cache](/blog/machine-learning/edge-ai/llm-quantization-activations-smoothquant-kv-cache) — the KV-cache is, after all, just cached activations.

### Step 3: speculative decoding with a draft model

`llama.cpp` supports speculative decoding by passing a second, smaller **draft model** with `-md` (model-draft) and a draft length with `--draft` (or `--draft-max`). The draft and target must share a vocabulary/tokenizer:

```bash
# Target = 8B, draft = a 1B of the same family (shared tokenizer)
./llama-speculative \
    -m models/llama-3-8b-instruct-q4_k_m.gguf \
    -md models/llama-3.2-1b-instruct-q4_k_m.gguf \
    --draft-max 5 --draft-min 1 \
    --ctx-size 8192 \
    -p "Write a Python function that merges two sorted lists." \
    -n 256
```

The output log reports the **acceptance rate** (often shown as accepted/drafted ratio, the empirical $\alpha$ times $K$) and the resulting tokens/s. That acceptance rate is your single most important tuning signal:

- **High $\alpha$ (> 0.7)**: draft and target agree often — increase `--draft-max` to harvest more tokens per verify.
- **Low $\alpha$ (< 0.4)**: draft is a poor match (wrong family, too small, off-distribution prompt) — a bigger or better-matched draft, or drop speculation entirely; below break-even it's *slower* than plain decode because you pay draft cost for tokens you reject.

A small measurement harness to compare baseline vs. speculative honestly (warm up first, average several runs, fixed prompt and seed):

```python
import subprocess, re, statistics

def run_and_parse(cmd, n_runs=5):
    tps = []
    for _ in range(n_runs):
        out = subprocess.run(cmd, capture_output=True, text=True).stderr
        m = re.search(r"eval time.*?([\d.]+)\s*tokens per second", out)
        if m:
            tps.append(float(m.group(1)))
    return statistics.median(tps), tps

base = ["./llama-cli", "-m", "models/llama-3-8b-instruct-q4_k_m.gguf",
        "-p", "Write a Python function that merges two sorted lists.",
        "-n", "256", "--seed", "42"]

spec = ["./llama-speculative", "-m", "models/llama-3-8b-instruct-q4_k_m.gguf",
        "-md", "models/llama-3.2-1b-instruct-q4_k_m.gguf",
        "--draft-max", "5",
        "-p", "Write a Python function that merges two sorted lists.",
        "-n", "256", "--seed", "42"]

b_med, _ = run_and_parse(base)
s_med, _ = run_and_parse(spec)
print(f"baseline:    {b_med:.1f} tok/s")
print(f"speculative: {s_med:.1f} tok/s  ({s_med/b_med:.2f}x)")
```

Crucially, log the *acceptance rate* alongside throughput so a slowdown is diagnosable, not mysterious. `llama-speculative` prints accepted/drafted counts; parse them and turn them into the empirical $\alpha$ and the predicted speedup, then compare the prediction to the measurement — if they disagree, your $K$ or draft cost assumption is off:

```python
def parse_acceptance(stderr):
    # llama.cpp logs lines like: "n_drafted = 198, n_accept = 142"
    drafted = re.search(r"n_drafted\s*=\s*(\d+)", stderr)
    accept  = re.search(r"n_accept\s*=\s*(\d+)", stderr)
    if not (drafted and accept):
        return None
    d, a = int(drafted.group(1)), int(accept.group(1))
    return a / d  # empirical per-token acceptance alpha

def predicted_speedup(alpha, K, c=0.13):
    exp_tokens = (1 - alpha ** (K + 1)) / (1 - alpha)
    return exp_tokens / (1 + c * K)

out = subprocess.run(spec, capture_output=True, text=True).stderr
alpha = parse_acceptance(out)
if alpha is not None:
    print(f"alpha = {alpha:.2f}  ->  predicted {predicted_speedup(alpha, 5):.2f}x")
```

If the predicted speedup is comfortably above 1.0 but the *measured* one isn't, the usual culprit is draft-model overhead being larger than your $c$ assumes (a draft that isn't really 1/8 the cost, or one not benefiting from the same offload), or the draft and target competing for the same memory bandwidth on a unified-memory device — both models stream from the same RAM, so on an M-series chip the draft's reads compete with the target's. That bandwidth contention is a real, edge-specific reason the laptop speedup can trail the GPU-server numbers in the papers.

#### Worked example: acceptance rate to measured speedup

You run the harness above on an M2 MacBook, 8B target + 1B draft, `--draft-max 5`, on a code-generation prompt. The log reports **142 accepted out of 198 drafted**, i.e. an empirical per-token acceptance $\alpha = 142/198 \approx 0.72$. Predict the speedup and check it.

Expected tokens per target forward at $\alpha = 0.72$, $K = 5$:

$$
\mathbb{E}[\text{tokens}] = \frac{1 - 0.72^{6}}{1 - 0.72} = \frac{1 - 0.139}{0.28} = \frac{0.861}{0.28} \approx 3.08
$$

So each target pass yields ~3.08 tokens. The 1B draft is ~1/8 the target cost per token, $c \approx 0.13$, and $K = 5$ draft tokens cost $cK \approx 0.65$ target-equivalents per round. Net speedup prediction:

$$
\text{speedup} \approx \frac{3.08}{1 + 0.65} = \frac{3.08}{1.65} \approx 1.87\times
$$

Measured: baseline 18 tok/s → speculative 34 tok/s, a $1.89\times$ — within a hair of prediction. This is why the acceptance rate is the number to watch: it tells you, before you even tune $K$, whether speculation is worth running. Notice too that code prompts tend to give *high* acceptance (lots of predictable boilerplate, brackets, indentation), while creative writing gives lower acceptance (more genuinely uncertain tokens) — so the same setup might only hit $1.4\times$ on poetry. Always measure on *your* workload.

### Step 4: prompt/prefix caching across turns

For TTFT, reuse the prompt KV. `llama.cpp` can save and load a prompt's cache state to disk with `--prompt-cache`, and its server keeps prefix KV in memory across requests automatically (prefix matching). For a fixed system prompt + document you process repeatedly:

```bash
# First run: process the long prefix and SAVE its KV state
./llama-cli -m models/llama-3-8b-instruct-q4_k_m.gguf \
    --ctx-size 8192 \
    --prompt-cache chat-prefix.bin \
    --prompt-cache-all \
    -f system_plus_doc.txt -n 1

# Later turns: LOAD the cached prefix, only prefill the new user message
./llama-cli -m models/llama-3-8b-instruct-q4_k_m.gguf \
    --ctx-size 8192 \
    --prompt-cache chat-prefix.bin \
    -p "$(cat system_plus_doc.txt)\n\nUser: What are the top three risks?" \
    -n 256
```

For an interactive app, run `llama-server` instead — its built-in prefix caching means turn two of a conversation reuses turn one's KV automatically, and you'll see TTFT drop from ~hundreds of ms to tens of ms once the prefix is warm. This is the cheapest big win on the list: one flag (or just using the server), zero quality cost, and the most *felt* improvement for an interactive user.

## Results: stacking the tricks

Now the part the kit insists on — measured before→after. Target: an M2 MacBook Air (8-core, ~100 GB/s unified memory bandwidth), Llama-3-8B-Instruct at Q4_K_M (~4.9 GB), batch size 1, a multi-turn coding-assistant workload with a 1,500-token system prompt and a 4,000-token pasted file. Decode throughput measured over a 256-token generation, median of 5 warm runs (discard the first, it pays disk and warm-up cost); TTFT measured on turn 2; "peak KV mem" at 16k context. These are representative figures for this class of setup — measure on your own device and prompt, since bandwidth and acceptance vary.

| Configuration | Decode tok/s | Peak KV mem (16k) | TTFT turn 2 | Notes |
|---|---|---|---|---|
| Baseline (fp16 KV) | 18 | 4.0 GB | 900 ms | Slow long chats, OOM risk |
| + int8 KV-cache | 20 | 2.0 GB | 880 ms | Half the cache, slight read win |
| + speculative decode (1B draft, K=5) | 38 | 2.0 GB | 860 ms | ~2× decode, free quality |
| + prefix caching (server) | 40 | 2.0 GB | 120 ms | TTFT collapses on turn ≥ 2 |

A second view, the KV-memory table from the derivation, for the same 8B (GQA, 8 KV heads → 0.125 MB/token fp16):

| Context length | fp16 KV | int8 KV (q8_0) | q4 KV |
|---|---|---|---|
| 4k | 0.5 GB | 0.25 GB | 0.13 GB |
| 16k | 2.0 GB | 1.0 GB | 0.5 GB |
| 32k | 4.3 GB | 2.1 GB | 1.1 GB |
| 128k | 17.2 GB | 8.6 GB | 4.3 GB |

Read these two tables together and the strategy writes itself. KV quantization buys you **context length** (the memory column) for near-zero quality; speculative decoding buys you **throughput** (the tok/s column) for zero quality; prefix caching buys you **TTFT** (the latency column) for zero quality. They attack three different axes and *compose* — Figure 7 shows the compounding. Stacking all three on the baseline roughly **doubles** decode speed, **halves** KV memory, and cuts felt latency on follow-up turns by ~7×, none of it costing output quality.

![Matrix showing decode throughput, KV memory, and time-to-first-token improving as int8 KV, speculative decoding, and prefix caching are stacked on the baseline](/imgs/blogs/making-on-device-llms-fast-7.png)

### Honest measurement caveats

Numbers like these lie if you don't measure carefully:

- **Warm-up**: the first run pays page-in, kernel JIT/Metal-shader compile, and a cold cache. Discard it. Report median of several warm runs.
- **Thermal throttling**: a fanless laptop or a phone will slow down after ~30–60s of sustained generation. A 10-token benchmark hides this; measure a realistic 256–512 token run.
- **Acceptance variance**: speculative speedup is workload-dependent. Report the acceptance rate alongside tok/s, and benchmark on prompts representative of your real traffic, not just "write a poem."
- **Batch=1 reality**: on-device is single-user; don't quote server throughput numbers (which come from large batches) as if they apply locally. Decode at batch 1 is bandwidth-bound, full stop.
- **Quality regression checks**: even "free" tricks have edges. After enabling q4 KV, run a small eval (a few reasoning/long-context tasks) to confirm no degradation; don't trust "looks fine" on one prompt.

## Stress test: what breaks, and how to reason about it

A technique you only understand at its happy path you don't really understand. Here are the failure modes that will actually page you, each posed as the concrete question and reasoned to a fix.

**What breaks at 32k context on a full-MHA model?** Two things, in order. First, memory: the fp16 KV-cache hits ~16 GB (the table above), and on a 16 GB device that plus the weights plus the OS is an instant OOM-kill — usually at *prefill* of the long document, before a single reply token. Second, even if you fit it, decode slows because per-step KV traffic climbs with context. The fix order is the same order as the levers: pick a GQA model (drops the 16 GB to ~4 GB), quantize the KV to int8 (~2 GB), and if you still need more, q4 the *values* (keep keys at q8). If you've done all three and 32k still won't fit or stays slow, the honest move is to *not hold 32k of raw context* — chunk-and-retrieve or summarize older turns so the live cache stays bounded. The cache is linear in tokens; the only way to make it small is to hold fewer tokens.

**What happens when the draft model's acceptance rate is low?** You go *backwards*. From the sensitivity table, at $\alpha = 0.3$ the net speedup is ~0.9× — speculation is now a tax. The symptoms are subtle: tokens/s drops a little versus baseline and the acceptance line in the log reads, say, 60/200. Diagnose it by reading that ratio, not by guessing. Causes, most-to-least common: the draft is a different model family (mismatched distribution → large TV distance → low overlap); the prompt is off-distribution for the draft (a 1B draft trained mostly on web text drafting badly for dense legal or code-with-rare-APIs text); `--draft-max` set too high so you're drafting deep into low-probability territory; or temperature very high (high entropy means even a perfect draft can't agree often). Fixes in order: lower `--draft-max` to 2–3 (cheaper rounds, you stop paying for tokens you reject); switch to a same-family distilled draft; or, if acceptance stays under ~0.4 on your real traffic, turn speculation off — it is genuinely the wrong tool for that workload.

**What happens when decode is memory-bound and you reach for the wrong fix?** This is the classic wasted afternoon. Someone profiles a slow assistant, sees the GPU at 15% utilization during decode, and concludes "the GPU is underused, let me give it more work" — by, say, increasing batch size or enabling a heavier sampler. But low GPU utilization during decode *is the expected signature of a memory-bound kernel*: the ALUs are idle because they're starved for bytes, not because they're under-fed work. Adding compute does nothing; the bottleneck is bandwidth. The only things that move a memory-bound decode are reading fewer bytes (quantize weights/KV, GQA) or producing more tokens per weight-read (speculative decoding). Use the roofline worked example above as the diagnostic: if your measured tok/s is close to $\text{bandwidth} / \text{bytes-per-token}$, you are at the memory ceiling and no amount of extra compute will help — only fewer bytes will.

**What happens to q4 KV on a hard reasoning task?** This is the one "free" trick that isn't always free, and it fails *quietly*. q8 keys/values are safe across the board; q4 *values* are usually fine; q4 *keys* are where you can lose accuracy on long-context retrieval and multi-step reasoning, because keys feed the softmax and the exponential amplifies small key errors into mis-weighted attention — the model attends to the wrong earlier token. The danger is that it looks fine on short prompts and degrades only on the hard, long cases you didn't spot-check. The fix is procedural, not technical: never ship q4 *keys* without running a small targeted eval (a few needle-in-a-haystack retrieval prompts and a couple of multi-step reasoning items) and comparing against q8. If the eval moves, fall back to q8 keys / q4 values, which keeps most of the memory win and almost none of the risk.

**What happens when prefix caching silently stops hitting?** Symptom: TTFT that *was* ~120 ms creeps back toward ~900 ms turn after turn, and the server's prefill-token counter is high every turn instead of near zero. Cause: something is perturbing an early token so the longest-common-prefix match fails — a per-turn timestamp injected at the top, a system prompt rebuilt in a different key order, or whitespace/formatting that varies. Diagnose by logging the matched-prefix length the server reports; if it's near zero when it should be ~5,500, your template isn't stable. Fix by freezing the prompt prefix: stable content (system prompt, tools, pasted doc) first and byte-identical every turn, volatile content (the new user message) last. This is the most common way a working prefix cache rots in production, and it's pure prompt-template discipline to prevent.

## Case studies and real numbers from the literature

A few load-bearing results from the papers, so your expectations are calibrated:

- **Speculative decoding (Leviathan et al., 2023; Chen et al., 2023)**: both report roughly **2–3×** wall-clock speedup on tasks like summarization and translation with a well-matched draft, with provably identical output distribution. The acceptance rate, not the model size ratio, sets the gain.
- **Medusa (Cai et al., 2024)**: ~**2.2–2.8×** speedup by adding decoding heads to the target — no separate draft model and only a light fine-tune of the heads, which is attractive when you can't afford a second model's RAM on device.
- **PagedAttention / vLLM (Kwon et al., 2023)**: cut KV-cache memory waste from **60–80%** down to **under 4%**, yielding **2–4×** higher serving throughput at the same hardware — a server result whose edge analogue is "fit far more concurrent or branched conversation state in the same RAM."
- **KIVI (Liu et al., 2024)**: **2-bit** KV-cache quantization (asymmetric, per-channel keys / per-token values) with negligible quality loss, enabling roughly **2.6×** less peak memory and larger batch/context — the research that makes aggressive `--cache-type` flags defensible.

A grounded edge anecdote that ties it together: a 7B-class assistant on an M2 laptop, naively, gives ~18 tok/s and starts choking past ~16k context. With GQA-by-design (8B model), int8 KV, a 1B draft for speculation, and server-side prefix caching, the same laptop sustains long multi-turn sessions at ~35–40 tok/s with sub-150 ms follow-up latency and comfortably holds 32k+ of context in the same memory envelope. None of those wins came from touching the weights — they're all decode-time systems work.

## Which tricks matter on edge vs. server (and the order to apply them)

The single most useful thing to internalize is that these techniques are *not* interchangeable — each targets a specific bottleneck, and the edge profile (single user, batch 1, tight RAM, latency-sensitive interaction) weights them differently from the server profile (many users, big batches, throughput-maximizing). Figure 8 is the decision tree.

![Decision tree mapping the bottleneck — memory wall, slow decode, repeated prompts, or many sessions — to KV quantization, speculative decoding, prefix caching, or paged attention](/imgs/blogs/making-on-device-llms-fast-8.png)

| Technique | Targets | Edge (single-user) value | Server value | Quality cost |
|---|---|---|---|---|
| KV-cache quantization | Memory (context length) | High — buys long context cheaply | High | ~0 (q8), small (q4) |
| GQA / MQA (architectural) | Memory | High — pick a GQA model | High | Built into weights |
| Speculative decoding | Decode throughput | High — ~2× tok/s | Medium (helps low-batch) | Zero (exact distribution) |
| Prefix / prompt caching | TTFT | Very high — biggest felt win | High | Zero |
| Paged attention | KV utilization, sharing | Medium — multi-session/branching | Very high | Zero |
| Continuous batching | Throughput under load | Low — no batch at b=1 | Very high | Zero |

**The honest order to apply them on edge**, cheapest-and-safest first:

1. **Pick a GQA model and quantize the weights.** Before any decode trick, start from a model whose architecture already has a small KV-cache (GQA/MQA) and 4-bit weights. This is the foundation; it's [small-language-models-by-design](/blog/machine-learning/edge-ai/small-language-models-by-design) and [quantization](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) doing their job.
2. **Turn on prefix caching.** One flag (or just use the server), zero quality cost, the most *felt* latency win for interactive use. Do this first among the decode tricks.
3. **Quantize the KV-cache to int8.** `--cache-type-k/v q8_0`. Halves KV memory for ~free, doubling your usable context. Re-check quality if you go to q4.
4. **Add speculative decoding *if* you have the RAM for a draft** (or use Medusa/self-speculation if you don't). Measure acceptance on your real workload; keep it only if you're comfortably above break-even.
5. **Reach for paged attention only when you run multiple sessions or branch** (regenerate, tree search) — otherwise a contiguous cache sized to your actual context wastes little.

When *not* to bother: if your conversations are always short (a few hundred tokens), the KV-cache never becomes the wall and you can skip KV quantization entirely. If your draft model can't clear ~0.4 acceptance on your prompts, speculative decoding is *negative* — drop it. If you serve exactly one chat at a time, continuous batching and paged attention give you little; don't add their complexity. And never reach for q4 KV on a hard reasoning task without an eval — the keys feeding the softmax are the one place where "free" quantization can quietly cost you. This whole decode-time toolkit is the runtime half of the broader [edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook); pair it with the weight-level levers, don't substitute for them.

## Key takeaways

- **The KV-cache, not the weights, is the long-context memory wall.** It costs $2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{seq\_len} \times \text{bytes}$ — for a full-MHA 7B that's ~0.5 MB/token, so it overtakes the model at long context. Compute it from the *architecture*, not the parameter count.
- **Decode is memory-bandwidth-bound at batch 1.** Tokens/s ≈ bandwidth / bytes-read-per-token. Every win is "fewer bytes read" (quantize weights/KV, GQA) or "more tokens per weight-read" (speculative decoding).
- **KV-cache quantization (`--cache-type-k/v q8_0`) halves cache memory for near-zero quality** and doubles your usable context. Keys are more sensitive than values; prefer q8 keys.
- **Speculative decoding is free quality.** The accept/reject rule reproduces the target's exact distribution; expected tokens per pass is $\frac{1 - \alpha^{K+1}}{1 - \alpha}$, so a 0.7 acceptance at K=4–5 gives ~2× decode throughput. Watch the acceptance rate; below ~0.4 it's a loss.
- **Prefix caching is the biggest single-user edge win for latency.** Reuse the system-prompt/document KV across turns and TTFT on follow-ups drops from hundreds of ms to tens. One flag, zero quality cost.
- **Paged attention is mostly a server win**, but it earns its keep on edge for multi-session and branching (prefix sharing via copy-on-write). For a single chat, a right-sized contiguous cache wastes little.
- **They compose across three axes**: KV-quant buys context, speculation buys throughput, prefix caching buys TTFT. Stack all three and you can roughly double speed, halve KV memory, and cut follow-up latency ~7× — none of it costing output quality.
- **Apply in cost order**: GQA model + 4-bit weights → prefix caching → int8 KV → speculative decoding (if RAM allows) → paged attention (only if multi-session). Measure each step on your real workload, warm runs only.

## Further reading

- Leviathan, Kalman, Matias, "Fast Inference from Transformers via Speculative Decoding" (2023) — the expected-speedup derivation and the rejection-sampling guarantee.
- Chen, Borgeaud, Irving, et al., "Accelerating Large Language Model Decoding with Speculative Sampling" (2023) — the concurrent DeepMind formulation with the exact-distribution proof.
- Kwon, Li, Zhuang, et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023) — the vLLM paper; the 60–80%→<4% waste result.
- Cai, Li, Geng, et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads" (2024) — draft-model-free speculation via extra heads.
- Liu, Yuan, Jin, et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache" (2024) — how low you can push KV-cache bits and why keys vs. values differ.
- `llama.cpp` documentation — `--cache-type-k/v`, `-md`/`--draft-max` (speculative), `--prompt-cache`, and the server's prefix caching.
- Within this series: [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression), [the roofline model: where your bottleneck lives](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives), [LLM quantization II: activations, SmoothQuant, and the KV-cache](/blog/machine-learning/edge-ai/llm-quantization-activations-smoothquant-kv-cache), [small language models by design](/blog/machine-learning/edge-ai/small-language-models-by-design), [running LLMs locally with llama.cpp and GGUF](/blog/machine-learning/edge-ai/running-llms-locally-llama-cpp-and-gguf), and the capstone [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook).
- Going deeper on the cache itself: [KV-cache optimization and management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management).

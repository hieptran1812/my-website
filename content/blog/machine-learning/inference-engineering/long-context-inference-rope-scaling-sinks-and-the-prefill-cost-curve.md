---
title: "Long-Context Inference: RoPE Scaling, Sinks, and the Prefill Cost Curve"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Long context breaks inference in two independent places at once — a quadratic prefill bomb and a linear-but-enormous KV cache — and this post derives both, then builds the RoPE-scaling, sliding-window, and sparse-attention knobs that tame them into nanoserve."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "long-context",
    "kv-cache",
    "rope",
    "sparse-attention",
    "attention-sinks",
    "pytorch",
    "gpu",
    "deepseek",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 41
---

A user pastes a 200,000-token codebase into your chat endpoint and asks one question. On a screen somewhere your dashboard goes quiet: the token-per-second counter for forty other in-flight requests drops to zero and stays there for the better part of a minute, then half of them fail with an out-of-memory error that names a tensor nobody allocated. You did not deploy a bug. You deployed a model that advertises a 128k or 1M context window, and a single request just exercised it.

Long context is the one feature that breaks inference in **two independent places at the same time**, and the two failures have nothing to do with each other. The first is **compute**: attention is quadratic in sequence length, so the *prefill* — the one-shot pass that reads the whole prompt before the first output token — does roughly a thousand times more attention work at 128k than at 4k. That work runs on the GPU as one indivisible job, and while it runs, every other request waits. The second is **memory**: the KV cache is only *linear* in sequence length, but the constant is so large that one long request's cache can equal or exceed the entire model's weights. One request becomes a prefill bomb and a memory hog simultaneously, and the fix for one is not the fix for the other.

![Diagram showing a single 200k-token request splitting into a quadratic prefill path that stalls other streams and a linear KV memory path that risks an out-of-memory failure, both converging on the same mitigation](/imgs/blogs/long-context-inference-rope-scaling-sinks-and-the-prefill-cost-curve-1.webp)

This post takes the two costs apart and derives each from scratch, then builds the inference-time machinery that bounds them — into `nanoserve/longctx.py`, the file this post owns. By the end you will be able to: derive the prefill FLOPs and the KV bytes for any prompt length and model, and predict exactly where each one bites; add a **RoPE-scaling knob** (linear interpolation and NTK-aware) so the model can address positions it never trained on; implement a **sliding-window plus attention-sink mask** that caps the cache at a constant no matter how long the context grows; write a **sparse top-k attention** that turns the quadratic into `S · k`; and encode the two scheduling consequences — a long prompt must be *chunked* so it does not freeze the fleet, and *admission-controlled* so it does not OOM the node. Everything is derived, cited to a primary source, or framed as a script you can run yourself; there are no first-hand benchmarks here, because I have no GPU and ran nothing. The running example stays the same as the rest of the series — serving Llama-3.1-8B on one A100 80GB or H100 — with DeepSeek's MLA and sparse-attention stack cited as the reference for what a production long-context engine actually does.

If you have not read [what inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is), start there for the `nanoserve` frame; this post assumes you know what prefill, decode, TTFT (time to first token), TPOT (time per output token), and the KV cache are. The [memory math of the KV cache](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache) derives the per-token bytes we lean on here, and [chunked prefill and the TTFT/TPOT tradeoff](/blog/machine-learning/inference-engineering/chunked-prefill-and-the-ttft-tpot-tradeoff) is the scheduling half of the story.

## 1. Two costs, derived separately — because they fail differently

The single most common mistake in reasoning about long context is to lump "it's expensive" into one number. It is two numbers with different exponents, and they peak at different times in a request's life.

- **Prefill compute is quadratic.** Reading an `S`-token prompt costs `O(S² · d)` in the attention layers, because every token attends to every earlier token. This cost is paid **once**, up front, before the first output token, and it lands entirely inside TTFT. It scales with the *square* of the prompt.
- **KV cache memory is linear.** Storing the keys and values for `S` tokens costs `O(S)` bytes. This cost is paid **continuously** — it exists for the whole lifetime of the request, grows by one token's worth every decode step, and it competes for the same HBM that holds the weights and everyone else's cache. It scales *linearly*, but with a brutal constant.

Because prefill is a one-shot compute spike and KV is a sustained memory occupancy, they hurt different parts of your system. Prefill hurts **latency and fairness**: a long prefill monopolizes the GPU's math units and every concurrent decoder stalls behind it. KV hurts **capacity and stability**: a long cache evicts other requests or OOMs the node. You cannot fix both by "using a smaller model" or "adding a GPU"; you fix prefill with *scheduling* (chunk it) and KV with *admission control plus architecture* (bound it). The rest of this post derives each, then builds the knob that bounds it.

A note on the honesty of every number that follows. Each quantitative claim is one of three kinds, and the prose says which: **derived** from a formula shown in the text; **cited** to a paper, model card, or vendor doc named with a link; or **reproducible** — a script plus an expected range on named hardware, framed as what *you* will see, never what I saw. Results tables carry a `Source` column. That discipline matters more here than anywhere, because long-context numbers are exactly the kind that get quoted out of a slide with no idea where they came from.

## 2. The prefill cost curve: where the quadratic actually lives

Prefill runs the model forward over the entire prompt in one batched pass and fills the KV cache. Its cost has two terms, and they scale differently.

The **linear term** is everything that is a matmul against the weights: the Q/K/V projections, the output projection, and the MLP. A standard result is that a forward pass costs about `2N` FLOPs per token, where `N` is the parameter count (one multiply and one add per weight). So the linear part of prefill is:

$$\text{FLOPs}_{\text{linear}} \approx 2 N S$$

For Llama-3.1-8B, $N \approx 8.03 \times 10^9$. At a 4k prompt that is $2 \cdot 8.03\mathrm{e}9 \cdot 4096 \approx 6.6 \times 10^{13}$ FLOPs; at 128k it is $2 \cdot 8.03\mathrm{e}9 \cdot 131072 \approx 2.1 \times 10^{15}$. Linear — 32× the tokens, 32× the work.

The **quadratic term** is attention itself. For each layer, computing the score matrix $QK^\top$ costs $2 H_q S^2 d_{\text{head}}$ and applying it to $V$ costs another $2 H_q S^2 d_{\text{head}}$; causal masking means each query only attends to earlier tokens, which halves the sum. Across `L` layers:

$$\text{FLOPs}_{\text{attn}} \approx 2 L H_q S^2 d_{\text{head}}$$

For Llama-3.1-8B ($L = 32$ layers, $H_q = 32$ query heads, $d_{\text{head}} = 128$), the constant $2 L H_q d_{\text{head}} = 262{,}144$. So attention FLOPs are $262{,}144 \cdot S^2$. This is the term that explodes.

![Two-panel comparison contrasting full attention with a quadratic pair count and a growing cache against a windowed variant with a bounded token set and a constant cache](/imgs/blogs/long-context-inference-rope-scaling-sinks-and-the-prefill-cost-curve-4.webp)

Here is the crossover that most people never compute. Attention overtakes the linear term when $2 L H_q S^2 d_{\text{head}} = 2 N S$, i.e. at:

$$S_{\text{cross}} = \frac{N}{L \, H_q \, d_{\text{head}}} = \frac{8.03\mathrm{e}9}{32 \cdot 32 \cdot 128} \approx 61{,}000 \text{ tokens}$$

Below ~61k tokens, prefill is dominated by the weight matmuls — attention is a rounding error (at 4k it is 6.7% of the linear cost). Above ~61k, attention dominates, and from there every doubling of the prompt *quadruples* the attention work. That single fact — that there is a length past which your model stops being a matmul machine and becomes an $S^2$ machine — is the whole reason long context is a distinct engineering problem.

A common misconception is worth killing here: **FlashAttention does not fix the quadratic.** FlashAttention (and its descendants, covered in [kernel fusion and FlashAttention](/blog/machine-learning/high-performance-computing/kernel-fusion-and-flashattention-beating-the-memory-wall)) tiles the attention computation so it never materializes the full $S \times S$ score matrix in HBM — it reduces the *memory* traffic from $O(S^2)$ to $O(S)$ and keeps the softmax in on-chip SRAM. That is an enormous constant-factor win, and it is why 128k context is tractable at all. But the *FLOPs* are still $O(S^2)$: every query-key dot product still happens, it just happens in a fused, memory-efficient loop. FlashAttention makes the quadratic *fast*; it does not make it *not quadratic*. When people say "we can do 1M context now," they mean the memory wall moved, not that the compute curve bent.

#### Worked example: the 4k chat prompt versus the 128k RAG prompt

Take two requests to the same Llama-3.1-8B: a chat turn with a 4k prompt, and a retrieval-augmented request that stuffs 128k tokens of retrieved documents into the context. Both ask for a short answer.

| Quantity | 4k prompt | 128k prompt | Ratio | Source |
| --- | --- | --- | --- | --- |
| Linear FLOPs (${2NS}$) | 0.066 PFLOP | 2.10 PFLOP | 32× | derived |
| Attention FLOPs ($2LH_q S^2 d$) | 0.0044 PFLOP | 4.50 PFLOP | **1024×** | derived |
| Total prefill FLOPs | 0.070 PFLOP | 6.60 PFLOP | 94× | derived |
| Attention share of total | 6% | 68% | — | derived |

The attention term grows exactly $32^2 = 1024\times$ — three orders of magnitude — while the prompt grew only 32×. The *total* grows 94× because the linear part dilutes it, but the shape of the curve is dominated by that quadratic. This is why "just cache the documents" (prefix caching, covered in [prefix sharing and radix trees](/blog/machine-learning/inference-engineering/prefix-sharing-radix-trees-and-copy-on-write)) is the single highest-leverage optimization for RAG: if you can avoid *recomputing* the 128k prefill on every request, you delete the 6.5 PFLOP entirely.

#### Worked example: the 200k prefill as a wall-clock stall

Now the failure mode from the intro. A 200k-token prompt on Llama-3.1-8B has a total prefill of:

$$\text{FLOPs}_{\text{linear}} + \text{FLOPs}_{\text{attn}} = 3.21\mathrm{e}15 + 1.05\mathrm{e}16 \approx 1.37 \times 10^{16} = 13.7 \text{ PFLOP}$$

An H100 SXM lists 989 TFLOP/s of bf16 tensor throughput ([NVIDIA H100 datasheet](https://resources.nvidia.com/en-us-tensor-core)). Long-context prefill rarely sustains more than about a third of peak once you account for the attention kernel's memory traffic and the non-matmul work, so budget roughly 330 TFLOP/s of *effective* throughput. That puts a single 200k prefill at $13.7\mathrm{e}15 / 3.3\mathrm{e}14 \approx 42$ seconds of GPU time. The exact wall-clock depends on your measured MFU (model FLOPs utilization) — the point is *tens of seconds*, and it scales as $S^2$, so a 400k prompt is four minutes.

Now recall that a GPU runs one kernel at a time. If forty decode streams are batched and stepping, and a 200k prefill arrives as one monolithic forward pass, those forty streams get **nothing** for 42 seconds. Their TPOT — normally 15–25 ms — spikes to tens of seconds. That is the "stall forty streams" failure: not a crash, a fairness collapse. The fix is not to make prefill faster; it is to slice it. [Chunked prefill](/blog/machine-learning/inference-engineering/chunked-prefill-and-the-ttft-tpot-tradeoff) breaks the 200k pass into, say, 2048-token chunks and interleaves decode steps between them, so the fleet keeps moving while the long prompt ingests over many scheduler iterations. We return to that in §8; for now, hold the picture: prefill is a compute bomb, and its blast radius is everyone else's latency.

## 3. The KV memory wall: linear, but the constant is the whole model

Switch from compute to memory. The KV cache stores, for every token, the key and value vectors at every layer so that future tokens can attend to them without recomputing. [Why recompute is fatal](/blog/machine-learning/inference-engineering/why-recompute-is-fatal-writing-a-kv-cache) makes the case for caching; here we count the bytes. For a model with `L` layers, $H_{kv}$ key/value heads (GQA groups them, which is why $H_{kv} = 8 < H_q = 32$ for Llama-3.1), head dimension $d$, storing both K and V at `b` bytes each:

$$\text{bytes/token} = 2 \cdot L \cdot H_{kv} \cdot d \cdot b$$

For Llama-3.1-8B in bf16: $2 \cdot 32 \cdot 8 \cdot 128 \cdot 2 = 131{,}072$ bytes = exactly **128 KiB per token**. Multiply by context length and the numbers get frightening fast.

![Vertical budget stack for an 80 GB A100 showing model weights, activations, and one long request's KV cache growing until it equals the model and squeezes the space left for other requests](/imgs/blogs/long-context-inference-rope-scaling-sinks-and-the-prefill-cost-curve-2.webp)

The figure above is the whole memory argument in one column. The weights of Llama-3.1-8B in bf16 are $8.03\mathrm{e}9 \cdot 2 = 16.06$ GB (14.96 GiB). Watch what the KV cache of a *single* long request does against that fixed baseline:

| Context length | KV cache (one sequence) | Relative to 16 GB weights | Source |
| --- | --- | --- | --- |
| 4k | 0.50 GiB | 3% | derived |
| 32k | 4.0 GiB | 25% | derived |
| **120k** | **15.0 GiB** | **≈ 100% — KV equals the model** | derived |
| 128k | 16.0 GiB | 107% | derived |
| 200k | 24.4 GiB | 163% | derived |
| 1M | **128 GiB** | 856% | derived |

At about 120k tokens, one request's cache is as large as the entire model. At 200k it is 24.4 GiB — which is why the intro's request OOM'd: on an 80 GB A100 you have the weights (15 GiB), activations (~4 GiB), and whatever the other tenants hold; a 24.4 GiB reservation on top can easily push you past the ~74.5 GiB the card actually gives you. And at 1M tokens the KV cache is 128 GiB — a single request's cache **does not fit on the GPU at all**, at any batch size, before you have stored a single byte of anyone else's context.

This is the crossover the [memory math post](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache) sets up and that a later production-trace post in this series dissects as a live OOM: the moment your dominant memory consumer flips from *weights* (fixed) to *KV cache* (per-request, unbounded). Once you are past it, capacity planning is no longer "how big is the model" but "how many token-slots of KV can I afford, and who gets them" — which is exactly the [admission-control](/blog/machine-learning/inference-engineering/admission-control-backpressure-and-latency-collapse) problem.

**How production models cheat the constant.** The 128 KiB/token above is dense multi-head-with-GQA. Multi-head *latent* attention (MLA), which [the MLA and attention-variants post](/blog/machine-learning/inference-engineering/mla-and-attention-variants-at-inference-time) covers in depth, compresses each token's cache to a small latent. Per vLLM's [DeepSeek-V3.2-Exp post](https://vllm.ai/blog/2025-09-29-deepseek-v3-2) (2025-09-29), DeepSeek's MLA cache is **656 bytes per token** — broken down as 512 bytes of quantized NoPE latent, 16 bytes of scales, and 128 bytes of RoPE key. That is roughly **200× smaller** per token than the Llama-3.1-8B dense cache ($131{,}072 / 656 \approx 200$). It is the difference between a 1M-token request costing 128 GiB and costing well under a gigabyte for the attention latent — which is precisely why the frontier long-context models are all MLA-based. We will keep coming back to that 656-byte figure; it is the number that makes 1M context *serveable*.

## 4. Position extension: teaching the model to count past where it trained

A model trained with a 8k context window has never seen position 100,000. Rotary position embeddings (RoPE) encode a token's position by *rotating* its query and key vectors by an angle proportional to the position. If you feed a position the model never trained on, the rotation lands in a region of angle-space it has no calibration for, and the attention logits go haywire — the model outputs garbage, or its perplexity explodes. Extending context at inference time is the art of choosing *how* to remap positions so the model can operate past its training length without a full retrain.

![Decision tree branching from a model trained at 8k that wants to reach 128k into a failing naive-extrapolation path and a working rescale-the-frequencies path with three methods](/imgs/blogs/long-context-inference-rope-scaling-sinks-and-the-prefill-cost-curve-3.webp)

To see the knobs, you need the mechanism. RoPE rotates dimension pair `i` of a token at position `m` by angle $\theta_i \cdot m$, where the per-dimension frequency is:

$$\theta_i = \text{base}^{-2i/d}, \quad i = 0, 1, \ldots, d/2 - 1$$

with `base` the `rope_theta` hyperparameter (10,000 in the original RoFormer; Llama-3.1 uses 500,000). Low `i` gives high-frequency, short-wavelength rotations (they cycle every few tokens); high `i` gives low-frequency, long-wavelength rotations (they barely move over the whole context). The **wavelength** of dimension `i` — how many tokens before that rotation completes a full turn — is $\lambda_i = 2\pi / \theta_i = 2\pi \cdot \text{base}^{2i/d}$.

**Why naive extrapolation fails.** If the model trained up to position 8192, the largest angle it saw for the highest-frequency dimension is $\theta_0 \cdot 8192$. Ask it about position 100,000 and that dimension's angle is 12× larger than anything in training — an out-of-distribution rotation. The attention dot product, which depends on the *relative* angle between a query and a key, produces values the softmax has never had to normalize. In practice the logits blow up and no amount of prompting saves it. This is the `danger` branch in the tree: extrapolation is free but broken.

Every working method instead **rescales the frequencies** so that the model's known angle range covers the new, longer position range. There are three, in increasing sophistication.

**Position Interpolation (PI), a.k.a. linear scaling.** Chen et al. 2023 ([arXiv:2306.15595](https://arxiv.org/abs/2306.15595)) — and independently the "kaiokendev" experiments — proposed the simplest thing: divide every position by a scale factor $s = L_{\text{target}} / L_{\text{train}}$. To reach 128k from an 8k model, $s = 16$; a token at position 100,000 is treated as if it were at position 6,250, comfortably inside the trained range. Equivalently, it divides every frequency by `s`. It works with only light fine-tuning, but it *compresses* the high-frequency dimensions — the ones that encode fine-grained local position — so the model's ability to distinguish adjacent tokens degrades. You have traded positional *resolution* for *range*.

**NTK-aware interpolation.** The insight (from the "bloc97" community work, later formalized) is that PI over-compresses the high frequencies, which are exactly the ones you want to keep sharp. Instead of scaling all frequencies equally, change the *base* itself: $\text{base}' = \text{base} \cdot s^{d/(d-2)}$. This spreads the interpolation *unevenly* — long-wavelength (low-frequency) dimensions get stretched a lot, short-wavelength (high-frequency) dimensions barely move. High-frequency local detail survives; only the slow, global position gets remapped. NTK-aware can extend context a few-fold with *no* fine-tuning at all, which is why it is the default "free" knob.

**YaRN.** Peng et al. 2023 ([arXiv:2309.00071](https://arxiv.org/abs/2309.00071)) makes the per-dimension treatment explicit: interpolate the low frequencies like PI, extrapolate the high frequencies, and use a smooth *ramp* to blend the bands in between, chosen by wavelength relative to the context length. YaRN adds one more correction that the others miss — an **attention temperature**: as you stretch positions, the softmax over more tokens gets flatter, so YaRN scales the logits by a factor $\sqrt{1/t}$ to restore the distribution's sharpness. It gives the best quality of the three and is what most modern long-context checkpoints ship (Llama-3.1's own `rope_scaling` is a closely related piecewise scheme with `factor: 8`, `low_freq_factor: 1`, `high_freq_factor: 4`, per the model's `config.json`).

| Method | What it does to RoPE frequencies | Fine-tune needed | Source |
| --- | --- | --- | --- |
| Naive extrapolate | nothing — uses unseen angles | fails regardless | derived |
| Linear PI | divide all frequencies by `s` | light | cited: Chen 2023 |
| NTK-aware | rescale the base; low freqs stretched, high freqs kept | often none | cited: bloc97 / YaRN |
| YaRN | per-band ramp + attention temperature | short, best quality | cited: Peng 2023 |

### The RoPE-scaling knob in nanoserve

Here is the whole thing as an inference-time switch on the RoPE we built in [the forward pass by hand](/blog/machine-learning/inference-engineering/a-forward-pass-by-hand-llama-from-scratch). The only decision is how to construct `inv_freq`; everything downstream is unchanged.

```python
# nanoserve/longctx.py
import torch

def build_inv_freq(head_dim: int, rope_theta: float, *,
                   mode: str = "none", factor: float = 1.0,
                   orig_max_pos: int = 8192) -> torch.Tensor:
    """Inverse frequencies for RoPE, with an inference-time scaling knob.

    mode="none"   : vanilla RoPE (extrapolates -> breaks past orig_max_pos)
    mode="linear" : Position Interpolation -- divide positions by `factor`
    mode="ntk"    : NTK-aware -- rescale the base, keep high freqs sharp
    """
    i = torch.arange(0, head_dim, 2, dtype=torch.float32)  # 0,2,4,...
    if mode == "ntk":
        # stretch the base so low freqs move a lot, high freqs barely move
        base = rope_theta * (factor ** (head_dim / (head_dim - 2)))
        inv_freq = 1.0 / (base ** (i / head_dim))
    else:
        inv_freq = 1.0 / (rope_theta ** (i / head_dim))
        if mode == "linear":
            inv_freq = inv_freq / factor  # PI: divide every frequency by s
    return inv_freq  # shape [head_dim/2]

def apply_rope(x: torch.Tensor, positions: torch.Tensor,
               inv_freq: torch.Tensor) -> torch.Tensor:
    """Rotate x [.., seq, head_dim] by position-dependent angles."""
    angles = positions[:, None].float() * inv_freq[None, :]   # [seq, hd/2]
    cos, sin = angles.cos(), angles.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rot = torch.stack([x1 * cos - x2 * sin,
                       x1 * sin + x2 * cos], dim=-1)
    return rot.flatten(-2)
```

The instrumentation that makes the tradeoff visible is the *effective wavelength* of the highest-frequency dimension. Print it for each mode and you can see, before running the model, how much local resolution you are giving up:

```python
def rope_diag(head_dim=128, rope_theta=500000.0, target=131072, orig=8192):
    s = target / orig                    # scale factor, e.g. 16
    for mode, factor in [("none", 1.0), ("linear", s), ("ntk", s)]:
        inv = build_inv_freq(head_dim, rope_theta, mode=mode, factor=factor)
        # wavelength of the fastest-rotating dim (i=0): 2*pi / theta_0
        lam0 = 2 * 3.14159265 / inv[0].item()
        print(f"{mode:7s} factor={factor:>5.1f}  "
              f"fastest wavelength = {lam0:8.1f} tokens")

rope_diag()
```

Expected output (values are `derived` from the formula, not a run of the model):

```console
none    factor=  1.0  fastest wavelength =      6.3 tokens
linear  factor= 16.0  fastest wavelength =    100.5 tokens
ntk     factor= 16.0  fastest wavelength =      7.4 tokens
```

Read that carefully: linear PI blows the fastest wavelength from 6.3 to 100 tokens — it has smeared the model's finest positional distinction across 16× more tokens, which is exactly the "loses local resolution" cost. NTK-aware keeps it at 7.4 tokens — nearly untouched — because it pushed the stretching onto the low-frequency dimensions instead. That single diagnostic tells you why NTK-aware usually wins for training-free extension, and why you would reach for YaRN's per-band ramp when you need to go further than either survives cleanly. **When to reach for which:** NTK-aware for a training-free 2–4× stretch; PI or YaRN with a short fine-tune when you need 8× or more and can afford the tuning run.

## 5. Bounding the cost, part 1: sliding windows and attention sinks

RoPE scaling lets the model *address* long positions, but it does nothing about the two costs — a 1M-token prompt is still 128 GiB of cache and a quadratic prefill. The first way to bound both is to simply **stop attending to most of the context**.

The observation behind sliding-window attention is that most tokens only need recent context. So cap it: each query attends only to the last `W` tokens. The cache never exceeds `W` tokens' worth of KV, and the attention cost drops from $O(S^2)$ to $O(S \cdot W)$ — linear again. But a naive sliding window has a nasty failure. When the earliest tokens fall out of the window, model quality *collapses* — perplexity spikes far more than losing a bit of old context should explain.

Xiao et al. 2023 ("Efficient Streaming Language Models with Attention Sinks," [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)) found the cause and the fix. The softmax in attention must put its probability mass *somewhere*; when nothing in the window is relevant, the model has learned to dump excess attention onto the **first few tokens** of the sequence — regardless of their content. These are **attention sinks**. Evict them and the softmax has no overflow valve, so it distorts the whole distribution. The fix is almost embarrassingly cheap: keep the last `W` tokens **and** the first few (typically 4) sink tokens, permanently. The cache is bounded at `W + sinks` tokens forever.

For Llama-3.1-8B with `W = 2048` and 4 sinks, the cache is $(2048 + 4) \cdot 128 \text{ KiB} = 256.5$ MiB — **constant**, at any context length, versus 128 GiB for dense attention at 1M. That is the entire deal, and it is the `before → after` in the figure above: an unbounded quadratic cache becomes a fixed 256 MiB one.

The motion is worth watching, because a still frame hides what the window actually does as generation proceeds:

<figure class="blog-anim">
<svg viewBox="0 0 660 180" role="img" aria-label="A fixed-size window slides along a row of tokens while the first two tokens stay pinned as sinks" style="width:100%;height:auto;max-width:820px">
<style>
.lc1-cell{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.lc1-sink{fill:var(--accent,#6366f1);opacity:.85}
.lc1-win{fill:var(--accent,#6366f1);opacity:.16;stroke:var(--accent,#6366f1);stroke-width:3}
.lc1-lbl{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.lc1-sl{font:600 12px ui-sans-serif,system-ui;fill:var(--background,#fff);text-anchor:middle}
@keyframes lc1-slide{0%{transform:translateX(0)}100%{transform:translateX(360px)}}
.lc1-mv{animation:lc1-slide 9s ease-in-out infinite alternate}
@media (prefers-reduced-motion:reduce){.lc1-mv{animation:none}}
</style>
<rect class="lc1-cell lc1-sink" x="30"  y="70" width="40" height="50" rx="6"/>
<rect class="lc1-cell lc1-sink" x="80"  y="70" width="40" height="50" rx="6"/>
<rect class="lc1-cell" x="150" y="70" width="40" height="50" rx="6"/>
<rect class="lc1-cell" x="200" y="70" width="40" height="50" rx="6"/>
<rect class="lc1-cell" x="250" y="70" width="40" height="50" rx="6"/>
<rect class="lc1-cell" x="300" y="70" width="40" height="50" rx="6"/>
<rect class="lc1-cell" x="350" y="70" width="40" height="50" rx="6"/>
<rect class="lc1-cell" x="400" y="70" width="40" height="50" rx="6"/>
<rect class="lc1-cell" x="450" y="70" width="40" height="50" rx="6"/>
<rect class="lc1-cell" x="500" y="70" width="40" height="50" rx="6"/>
<rect class="lc1-cell" x="550" y="70" width="40" height="50" rx="6"/>
<rect class="lc1-cell" x="600" y="70" width="40" height="50" rx="6"/>
<text class="lc1-sl" x="50"  y="100">s</text>
<text class="lc1-sl" x="100" y="100">s</text>
<rect class="lc1-win lc1-mv" x="146" y="64" width="150" height="62" rx="8"/>
<text class="lc1-lbl" x="75"  y="150">sinks (pinned)</text>
<text class="lc1-lbl" x="400" y="150">sliding window of recent tokens</text>
</svg>
<figcaption>The first tokens stay pinned as attention sinks while a fixed-width window sweeps forward over recent tokens; anything the window has passed and the sinks do not cover is gone from the cache.</figcaption>
</figure>

The cost is written on the animation: everything the window has slid past, and that is not a sink, **is gone**. The model literally cannot see the middle of the context anymore. On a "needle in a haystack" test — plant one fact at position 60,000 of a 128k prompt and ask about it — a sliding window of 2048 will miss the needle every time, because by the time the window reaches the question the needle is long evicted. This is the same "lost in the middle" degradation Liu et al. 2023 ([arXiv:2307.03172](https://arxiv.org/abs/2307.03172)) documented even for full attention, taken to its extreme. Sliding windows are perfect for *streaming* workloads (a chat that only cares about recent turns, a log tailer) and disqualifying for *retrieval* workloads that must recall an arbitrary earlier token.

### Sliding-window-plus-sink in nanoserve

Two pieces: a prefill mask (for the batched forward pass) and a bounded decode cache (for the step loop). The mask, built on top of the [paged-attention kernel](/blog/machine-learning/inference-engineering/paged-attention-kernel-by-hand), allows query `i` to see the sinks and its own window:

```python
# nanoserve/longctx.py  (continued)
def sliding_window_sink_mask(seq_len: int, window: int, sinks: int,
                             device="cpu") -> torch.Tensor:
    """Boolean [S, S] mask: query i may attend to key j iff
       j < sinks  (a sink)  OR  i - window < j <= i  (the window)."""
    i = torch.arange(seq_len, device=device)[:, None]
    j = torch.arange(seq_len, device=device)[None, :]
    causal   = j <= i
    in_window = j > (i - window)
    is_sink   = j < sinks
    return causal & (in_window | is_sink)

# use it with PyTorch's fused attention
import torch.nn.functional as F
def windowed_attention(q, k, v, window, sinks):
    S = q.shape[-2]
    mask = sliding_window_sink_mask(S, window, sinks, device=q.device)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
```

For decode, you never materialize the full cache — you keep a ring buffer of the last `W` tokens plus a frozen prefix of sinks:

```python
class StreamingKVCache:
    """Keeps `sinks` initial tokens + the most recent `window` tokens.
    Total slots = sinks + window, constant for the whole request."""
    def __init__(self, sinks: int, window: int, n_heads: int, head_dim: int):
        self.sinks, self.window = sinks, window
        self.cap = sinks + window
        self.k = torch.zeros(self.cap, n_heads, head_dim)
        self.v = torch.zeros(self.cap, n_heads, head_dim)
        self.n = 0            # tokens seen so far

    def append(self, k_t, v_t):
        if self.n < self.cap:
            slot = self.n
        else:
            # overwrite oldest *non-sink* slot: ring over [sinks, cap)
            slot = self.sinks + (self.n - self.sinks) % self.window
        self.k[slot], self.v[slot] = k_t, v_t
        self.n += 1
        return self.k[:min(self.n, self.cap)], self.v[:min(self.n, self.cap)]
```

Notice the ring only ever recycles slots in `[sinks, cap)` — the sink slots are written once and never evicted. That is the entire StreamingLLM trick in a dozen lines: constant memory, sinks preserved, and the model streams indefinitely without the perplexity cliff. What you *cannot* get back is the middle.

## 6. Bounding the cost, part 2: sparse attention that keeps the middle

Sliding windows throw away the middle unconditionally. Sparse attention asks a smarter question: instead of attending to *all* past tokens or only *recent* ones, attend to the *most relevant* few — wherever they are. That keeps the door open to a needle at position 60,000, as long as the selector finds it.

The cleanest production example is DeepSeek Sparse Attention (DSA), shipped in DeepSeek-V3.2-Exp. Per vLLM's [DeepSeek-V3.2-Exp post](https://vllm.ai/blog/2025-09-29-deepseek-v3-2) (2025-09-29), the mechanism is a two-stage attention: a cheap **lightning indexer** scores every past token against the current query, and only the **top-2048** highest-scoring tokens are passed to the expensive MLA attention. Each query attends to at most 2048 tokens no matter how long the context is, so the compute is $S \cdot 2048$ instead of $S^2$. At 1M tokens that is the difference between $10^{12}$ query-key pairs and $2 \times 10^9$ — a ~500× reduction — while the model can still reach any token the indexer ranks highly.

![Dataflow where a query fans into a lightning indexer that scores the whole past and a separate compressed cache, the indexer selects a top-k set, and both feed a bounded sparse attention while the paging path splits off](/imgs/blogs/long-context-inference-rope-scaling-sinks-and-the-prefill-cost-curve-5.webp)

The figure shows the two branches that make DSA a genuine *systems* problem, not just a math trick. The MLA KV cache — the 656 bytes per token from §3 — stays exactly as it was; DSA does not touch it. What DSA adds is the indexer, which needs its *own* key cache to score against, and a **top-k selection step** that runs before every attention. That selection is where the engineering hurts:

- **It fights paged attention.** A paged KV cache stores tokens in fixed 16- or 256-token blocks scattered across HBM (see [paged KV cache](/blog/machine-learning/inference-engineering/paged-kv-cache-implementing-blocks-and-a-block-table)). But top-k selection produces an *arbitrary* set of 2048 token indices, which do not align to blocks. Gathering them means a scatter-read across the block table — the exact access pattern paging was designed to avoid.
- **It fights continuous batching.** In a batched decode, each sequence has a different length and a different top-k set, so the causality bounds differ per query. DSA handles this with per-query `ks`/`ke` (key-start/key-end) tensors that encode each query's valid attention range, and vLLM's integration handles the indexer **separately for prefill and decode** because the two phases have completely different selection shapes: prefill selects for many queries at once, decode selects for one query against a growing cache.

That "handled separately for prefill and decode" is the tell that this is real production plumbing, not a paper diagram. If you want the training-side comparison — why DSA and NSA (native sparse attention) make different choices about *what* to make sparse — see [trainable sparse attention: NSA vs DSA](/blog/machine-learning/large-language-model/trainable-sparse-attention-nsa-vs-dsa).

### A simple sparse top-k attention in nanoserve

You will not reproduce DSA's lightning indexer here — it is a trained, low-precision scoring module. But the *shape* of sparse attention is short, and building it makes the paging problem concrete:

```python
# nanoserve/longctx.py  (continued)
def sparse_topk_attention(q_t, K, V, k: int):
    """One decode query q_t [H, d] attends to only the top-k past tokens.
    K, V: [S, H, d].  Returns the attention output [H, d].
    (Real DSA uses a cheap learned indexer for the scores; here we use
    the actual query-key dot as a stand-in.)"""
    S = K.shape[0]
    if S <= k:                                 # short context: dense
        idx = torch.arange(S, device=K.device)
    else:
        # score every past token, pick the top-k per head-averaged query
        scores = torch.einsum("hd,shd->s", q_t, K) / (q_t.shape[-1] ** 0.5)
        idx = torch.topk(scores, k).indices    # [k] arbitrary positions
        idx, _ = torch.sort(idx)               # keep causal order for gather
    Ksel, Vsel = K[idx], V[idx]                # the scatter-gather that
    attn = torch.softmax(                      #   fights the block table
        torch.einsum("hd,khd->hk", q_t, Ksel) / (q_t.shape[-1] ** 0.5), dim=-1)
    return torch.einsum("hk,khd->hd", attn, Vsel)
```

The `K[idx]` line is deceptively innocent. With a contiguous cache it is a fast gather; with a paged cache, `idx` is 2048 arbitrary logical positions that must each be translated through the block table to a physical block and offset, then read from scattered HBM. That is why DSA's real kernels are custom, and why "just add top-k" is not a weekend project on a production engine. The payoff, though, is exactly what the figure claims: **compute $S \cdot k$, not $S^2$**, while retaining the ability to reach any token — the property sliding windows give up.

## 7. Bounding the cost, part 3: compressing the KV cache itself

Windows and sparse attention bound *which* tokens you attend to. KV compression attacks the bytes directly: keep attending to everything, but store a *smaller representation* of each token's key and value. DeepSeek-V4 is the reference here. Per vLLM's [DeepSeek-V4 post](https://vllm.ai/blog/2026-04-24-deepseek-v4) (2026-04-24) — which is past my knowledge cutoff, so treat every number as strictly cited — the model combines four strategies, of which two are compression:

- **`c4a` — roughly 4× compression.** One *compressed* token is a weighted sum of 8 uncompressed tokens, produced with a stride of 4. So the cache holds one compressed entry for every ~4 original tokens (hence ≈1/4 the bytes), and each compressed entry blends an 8-token window (hence a controlled blurring: the model sees an 8:1 average, not the exact tokens). It is a learned pooling of the cache.
- **`c128a` — roughly 128× compression.** The same idea at a far coarser stride, giving a ~1/128 cache at the cost of only a coarse summary of that region.
- Combined with **DSA top-k over the compressed tokens** and a **sliding window of 128 uncompressed tokens** for the recent, exact context — so the recent window is lossless and the distant past is compressed.

The headline number is the one that makes 1M context practical: **9.62 GiB of bf16 KV per sequence at 1M tokens**, which vLLM's post frames as ~8.7× smaller than an 83.9 GiB estimate for a V3.2-style stack. Put that against our derived Llama-3.1-8B dense figure of **128 GiB at 1M**, and you see the full span of the design space — three orders of magnitude of cache, from dense MHA to a compressed MLA stack, for the *same* context length.

The recall tradeoff is the mirror of sliding windows, but graceful instead of a cliff. A query that needs an exact token from the compressed region sees an 8:1 (for `c4a`) or coarser (for `c128a`) blend, so fine details of old tokens are softened rather than deleted; the recent 128-token window stays exact. For a summarization workload that only needs the gist of the far context, that is nearly free. For a workload that must quote an exact token from 900k positions back, the blurring costs you — which is why the compression is *layered* with DSA's exact top-k selection rather than used alone.

![Comparison matrix rating full dense attention, window plus sinks, DSA sparse, and two KV-compression settings across cache size at one million tokens, compute cost, recall quality, and source](/imgs/blogs/long-context-inference-rope-scaling-sinks-and-the-prefill-cost-curve-7.webp)

The matrix is the whole toolbox on one card. Read it as a decision surface: dense attention gives exact recall and pays $O(S^2)$ compute plus a 128 GiB cache; every other row buys down cache or compute and pays in recall. Sliding windows are the cheapest and the bluntest (they delete the middle); DSA keeps the middle but at real systems cost; KV compression trades exact recall for a smooth degradation. There is no free row — which is the honest summary of long-context inference.

| Technique | Cache @ 1M | Compute | Recall | Source |
| --- | --- | --- | --- | --- |
| Full dense (MHA/GQA) | 128 GiB | $O(S^2)$ | exact | derived |
| Window + sinks | 256 MiB | $O(S \cdot W)$ | loses the middle | cited: Xiao 2023 |
| DSA sparse | MLA latent (656 B/tok) | $O(S \cdot 2048)$ | top-k only | cited: DeepSeek-V3.2 |
| KV compress `c4a` | ≈ 1/4 | reduced | blurred 8:1 | cited: DeepSeek-V4 |
| KV compress `c128a` | ≈ 1/128 | reduced | coarse | cited: DeepSeek-V4 |

#### Worked example: does a 1M-token request fit on one 80 GB GPU?

Pose it as an admission question. An 80 GB card gives ~74.5 GiB usable.

- **Dense Llama-3.1-8B:** KV at 1M = **128 GiB**. It does not fit — not with the weights, not without them, not at batch 1. Dense attention at 1M on one GPU is simply impossible; you would need tensor parallelism across four-plus GPUs *just for the cache*.
- **MLA latent alone (656 B/token):** $656 \cdot 1{,}048{,}576 \approx 0.64$ GiB. The attention latent for a 1M-token request fits in under a gigabyte. This is the number that changes the answer from "no" to "trivially yes."
- **DeepSeek-V4 full stack:** 9.62 GiB per sequence at 1M (cited). Comfortably fits on one 80 GB card alongside the model, with room for several such requests.

That is the entire case for MLA-plus-compression in one comparison: the *same* 1M-token context is a 128 GiB non-starter or a 9.62 GiB routine request depending purely on the attention architecture. Long context is not a model-quality problem anymore; it is a KV-representation problem.

## 8. The scheduling consequence: a long request is a bomb and a hog

Everything so far is about a *single* request in isolation. The scheduler's job is that a 200k request must coexist with everyone else, and it is dangerous in both dimensions at once: a prefill bomb (§2) and a memory hog (§3). The [scheduler as a policy problem](/blog/machine-learning/inference-engineering/the-scheduler-as-a-policy-problem) sets up the general machinery; here is the long-context-specific decision.

![Timeline of the scheduler handling a 200k prompt: arrival with a large reservation, an admission check, a KV reservation, chunked prefill, decoders continuing to step, and the failure path if both mitigations are skipped](/imgs/blogs/long-context-inference-rope-scaling-sinks-and-the-prefill-cost-curve-6.webp)

The timeline encodes the two mitigations that have to fire together. **Chunk the prefill** so the 42-second forward pass is sliced into 2048-token steps interleaved with decode iterations — the fleet keeps stepping while the long prompt ingests. **Admission-control the memory** so you reserve the request's full KV footprint *before* you start, rather than discovering mid-decode that you cannot grow the cache. Skip chunking and you freeze forty streams for 42 seconds. Skip admission and you OOM the node at, say, 57% reported memory when the reservation finally exceeds free HBM.

The admission predicate is worth deriving, because it is the exact line of code that stands between a healthy fleet and a latency collapse. A request `r` with a prompt of `S_prompt` tokens and a generation budget of `S_gen` tokens will eventually hold:

$$\text{reserve}(r) = (S_{\text{prompt}} + S_{\text{gen}}) \cdot \text{bytes-per-token}$$

and it is admissible only if that reservation fits alongside the KV already committed to the running set:

$$\text{reserve}(r) + \sum_{q \in \text{running}} \text{held}(q) \;\le\; \text{KV budget}$$

For our 200k request on Llama-3.1-8B, `reserve` is at least the prompt cache — $200{,}000 \cdot 128 \text{ KiB} = 24.4$ GiB — plus headroom for the tokens it will generate. Against the ~44 GiB of KV budget the figure in §3 left after weights and activations, that request is admissible **only if the running set currently holds under ~20 GiB of KV**. If it does not, the honest move is to *reject or queue* the request (backpressure), not to admit it and hope — admitting it forces a preemption cascade that [eviction and KV swapping](/blog/machine-learning/inference-engineering/eviction-preemption-and-kv-swapping) shows is often worse than the wait. A later post in this series walks through exactly this admission failure in a production trace, where a single unbounded long-context reservation triggered the OOM the whole node never recovered from.

```python
# nanoserve/longctx.py  (continued)
def can_admit(s_prompt: int, s_gen: int, running_kv_bytes: int,
              kv_budget_bytes: int, bytes_per_token: int) -> tuple[bool, int]:
    reserve = (s_prompt + s_gen) * bytes_per_token
    fits = reserve + running_kv_bytes <= kv_budget_bytes
    return fits, reserve

# 200k prompt, 2k generation, ~44 GiB budget, 20 GiB already running
GiB = 1024**3
fits, reserve = can_admit(200_000, 2_048, 20*GiB, 44*GiB, 131_072)
print(f"reserve={reserve/GiB:.1f} GiB  admit={fits}")
# -> reserve=24.7 GiB  admit=False   (24.7 + 20 = 44.7 > 44)
```

The predicate is three lines, but it is the whole difference between a scheduler that degrades gracefully under a long-context flood and one that face-plants. Note that it *must* use the full `S_prompt + S_gen` reservation, not the current length — reserving lazily is how you end up 90% of the way through a 24 GiB cache with nowhere to put the next block.

### The full context-cost report

The one piece of instrumentation that ties this post together is a table that grows with context, so you can see both curves before you ever deploy. This is the `derived` scoreboard `nanoserve` prints for a config:

```python
def context_cost_report(N=8.03e9, L=32, Hq=32, Hkv=8, d=128, b=2):
    per_tok = 2 * L * Hkv * d * b                       # KV bytes/token
    const_attn = 2 * L * Hq * d                         # attn FLOP constant
    print(f"{'ctx':>8}  {'KV cache':>10}  {'prefill FLOP':>13}  {'attn share':>10}")
    for S in [4096, 32768, 131072, 200_000, 1_048_576]:
        kv   = S * per_tok
        lin  = 2 * N * S
        attn = const_attn * S * S
        print(f"{S:>8}  {kv/1024**3:>7.2f} GiB  "
              f"{(lin+attn)/1e15:>10.2f} P  {attn/(lin+attn):>9.0%}")

context_cost_report()
```

Expected output (`derived` from the formulas in §2 and §3 — no model is run):

```console
     ctx    KV cache   prefill FLOP  attn share
    4096     0.50 GiB        0.07 P         6%
   32768     4.00 GiB        0.66 P        33%
  131072    16.00 GiB        6.61 P        68%
  200000    24.41 GiB       13.70 P        76%
 1048576   128.00 GiB      338.4 P         98%
```

Two curves, one table. The KV column climbs linearly to the 128 GiB wall; the attention-share column climbs from 6% to 98%, marking the exact point where your model stops being a matmul engine and becomes a quadratic-attention engine. Every long-context decision — chunk, admit, window, sparsify, compress — is a response to one of those two columns.

### Measuring the two curves honestly

The table above is *derived*. When you go to *measure* prefill on real hardware — to confirm your MFU assumption or to find out where your engine actually falls off the curve — the honest-measurement rules of this series apply with extra force, because long-context timings are noisy and easy to fool yourself with. The whole discipline is in [setting up a reproducible benchmark](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark); the long-context-specific traps are these:

- **Warm up, then time.** The first forward pass at a new sequence length triggers kernel autotuning, cuDNN/cuBLAS algorithm selection, and CUDA graph capture. Throw away the first 2–3 iterations at each length or you are timing the compiler, not the model.
- **Synchronize before you read the clock.** CUDA kernels are asynchronous — `time.time()` around a launch measures the *launch*, not the *work*. Use CUDA events, which record on the stream and give true device time.
- **Measure prefill and decode separately.** Prefill is one big compute-bound pass; decode is thousands of tiny memory-bound steps. A single tok/s number blends two completely different regimes and tells you nothing about either — exactly the mistake that hides a prefill bomb until production.

```python
import torch
def time_prefill(model, input_ids, warmup=3, iters=10):
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(warmup):                     # warm autotune + graphs
        model(input_ids)
    torch.cuda.synchronize()                    # drain before timing
    start.record()
    for _ in range(iters):
        model(input_ids)
    end.record()
    torch.cuda.synchronize()                    # wait for the work to finish
    return start.elapsed_time(end) / iters      # ms of true device time
```

And the load-shape trap that long context makes vicious: a **closed-loop** benchmark (fire the next request only after the last finishes) will never surface the prefill-bomb stall, because there is never a queue of decoders to freeze. You must use an **open-loop** generator — Poisson arrivals at a fixed rate, independent of completion — to see a 200k prefill do to your p99 TTFT and TPOT what §2 predicts. tok/s at batch 1 on a closed loop is the number that makes a long-context engine look fine right up until it collapses under real traffic.

## 9. Stress tests: push each knob until it breaks

A technique you have not stressed is a technique you do not understand. Four stresses, each pointed at a different knob.

**1M tokens on one 80 GB GPU.** Derived in §7: dense is 128 GiB and impossible; MLA latent is 0.64 GiB and trivial; DeepSeek-V4's full stack is 9.62 GiB and routine (cited). The lesson is that "supports 1M context" is a claim about the *KV representation*, not the model. If your engine is dense-GQA, "1M context" means "1M context spread across a multi-GPU tensor-parallel cache," and your capacity math must say so. Never let a model card's context number set your batch size; let the `context_cost_report` do it.

**The sliding-window needle cliff.** Plant a fact at position 60,000 of a 128k prompt, set `W = 2048` with 4 sinks, and ask about it after the window has swept past. The recall is not "degraded" — it is *zero*, because the needle's KV was overwritten in the ring buffer. This is not a bug to tune away; it is the definition of the technique. The correct response is workload routing: send streaming/chat traffic to the windowed path and retrieval/QA traffic to a dense or sparse path. A single endpoint that windows everything will pass its latency SLO and silently fail its accuracy one.

**RoPE scaling at 8× past training length.** Take an 8k-trained model to 64k with `mode="ntk"`, `factor=8`. The `rope_diag` output will show the fastest wavelength barely moving (NTK keeps high frequencies sharp), but perplexity on genuinely long-range dependencies still drifts, because no training-free method invents information the model never learned to use at that range. The honest expectation, reproducible with any long-context eval harness: NTK-aware buys you a *usable* 2–4× for free and a *degraded* 8× that a short YaRN fine-tune repairs. If you deploy an 8× stretch with no fine-tune and no eval, you have shipped a model that looks fine on short prompts and quietly falls apart on the long ones you extended it for.

**Sparse attention against a paged cache.** This is the one that bites in production. Wire `sparse_topk_attention` to a paged KV cache and the `K[idx]` gather becomes a scatter-read of 2048 arbitrary logical positions through the block table. On a contiguous cache that is fast; on a paged one it is a latency spike that can eat the compute you saved by going sparse. This is exactly why DSA is "handled separately for prefill and decode" and ships custom kernels — the selection-plus-gather is the hard part, not the sparse matmul. The takeaway: sparse attention is a *co-design* with your memory layout, not a drop-in mask. If you bolt top-k onto a paged engine naively, measure the gather latency before you celebrate the FLOP reduction.

## 10. Case studies: what the public record actually says

Four named, cited results — the primary sources behind the numbers in this post.

**StreamingLLM (Xiao et al. 2023, [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)).** The attention-sink discovery. The paper shows that keeping 4 initial tokens plus a rolling window lets a model stream over millions of tokens with stable perplexity, where a naive window collapses. It is the origin of the 4-sink convention and the reason `nanoserve`'s `StreamingKVCache` never evicts its first slots. Crucially, the paper is honest that this does *not* extend the model's *usable* context — it enables infinite *streaming*, not infinite *recall*.

**DeepSeek-V3.2-Exp DSA (vLLM post, [2025-09-29](https://vllm.ai/blog/2025-09-29-deepseek-v3-2)).** The lightning-indexer-plus-top-2048 mechanism, the 656-byte MLA cache (512B quantized NoPE + 16B scales + 128B RoPE), the `ks`/`ke` causality tensors, and the acknowledgment that the indexer challenges continuous batching and paged attention and is handled separately for prefill and decode. The post reports no throughput numbers — it is a mechanism-level source, which is exactly how it is used here.

**DeepSeek-V4 KV compression (vLLM post, [2026-04-24](https://vllm.ai/blog/2026-04-24-deepseek-v4)).** The `c4a` (≈1/4, one compressed token = weighted sum of 8 uncompressed, stride 4) and `c128a` (≈1/128) compression, the sliding window of 128 uncompressed tokens, and the headline 9.62 GiB bf16 KV per sequence at 1M — ~8.7× smaller than an 83.9 GiB V3.2-style estimate. The post's performance claims are component-level only (5–6% end-to-end at low batch; kernel fusions 1.4–20×); it is an architectural announcement past my cutoff, cited strictly.

**YaRN (Peng et al. 2023, [arXiv:2309.00071](https://arxiv.org/abs/2309.00071)).** The per-band interpolation plus attention-temperature correction that most modern long-context checkpoints build on, and the formalization of why NTK-aware and PI each leave quality on the table. Read alongside Position Interpolation (Chen et al. 2023, [arXiv:2306.15595](https://arxiv.org/abs/2306.15595)) for the linear baseline it improves on.

The common thread across all four: every one is explicit that its win *costs recall or resolution somewhere*, and every one names where. That is the standard to hold your own long-context deployment to.

## 11. When to reach for this (and when not to)

Long-context machinery is not free complexity, and most services do not need all of it. A decisive guide:

- **If your prompts are under ~32k and your model's trained window covers them, do nothing.** Attention is still a minority of prefill (33% at 32k), the KV cache is 4 GiB, and dense attention is exact and simple. Reaching for sparse attention here adds systems risk for no benefit. Just use vLLM's dense path.
- **If your prompts are long but your model's window already covers them, you still need chunked prefill and admission control** (§8) — the scheduling consequences bite regardless of architecture. This is the minimum viable long-context engineering, and it is scheduling, not attention.
- **Reach for RoPE scaling only to exceed the trained window,** and prefer a checkpoint that was *fine-tuned* for its advertised length over a training-free stretch you apply yourself. Use NTK-aware for a free 2–4× when you cannot fine-tune; do not ship an 8× stretch without a long-context eval.
- **Reach for sliding windows + sinks only for streaming/recency workloads** — chat with short memory, log processing, anything that does not need the middle. Never for retrieval or QA.
- **Reach for sparse attention or KV compression only when you are actually serving 128k+ regularly and dense does not fit.** These are co-designs with your kernels and memory layout; DeepSeek ships custom kernels for a reason. If you need this and are not DeepSeek, the honest recommendation is to use a model and engine that already implement it (an MLA-based model on vLLM or SGLang) rather than building DSA yourself. Build the RoPE knob and the windowed mask in `nanoserve` to *understand* them; run the real thing in production.

The meta-rule: measure which of the two curves — quadratic prefill or linear KV — is actually your bottleneck (the `context_cost_report` tells you), and apply only the knob that bends *that* curve. Applying a memory technique to a compute problem, or vice versa, is the most common way long-context "optimizations" make things worse.

## 12. Key takeaways

- **Long context has two costs with different exponents.** Prefill compute is $O(S^2)$ and paid once into TTFT; KV memory is $O(S)$ and paid continuously. They fail differently — one stalls the fleet, the other OOMs the node — and need different fixes.
- **The quadratic overtakes the linear around 61k tokens** (for Llama-3.1-8B). Below that, prefill is a matmul; above it, attention dominates and every doubling quadruples the work. The 4k→128k attention cost grows exactly 1024×.
- **KV memory crosses the model's own size around 120k tokens** and reaches 128 GiB at 1M — one dense request's cache does not fit on an 80 GB GPU. MLA's 656 bytes/token (cited) makes the *same* context 200× cheaper and turns 1M from impossible to routine.
- **RoPE scaling extends *addressing*, not *understanding*.** Naive extrapolation breaks; PI trades resolution for range; NTK-aware keeps high frequencies sharp for free; YaRN adds a per-band ramp and temperature. None invents range the model never learned.
- **Sliding windows + sinks give a constant cache but delete the middle** — perfect for streaming, disqualifying for retrieval. The 4 attention sinks are non-negotiable; evicting them collapses quality.
- **Sparse attention keeps the middle at $S \cdot k$ compute, but the top-k gather fights paged attention and continuous batching** — it is a memory-layout co-design, not a drop-in mask.
- **KV compression (c4a ≈1/4, c128a ≈1/128) trades exact recall for a smooth blur,** and layered with a lossless recent window it delivers 9.62 GiB at 1M (cited DeepSeek-V4).
- **The scheduler must chunk the prefill *and* admission-control the KV** — both, together. The admission predicate reserves the full $S_{\text{prompt}} + S_{\text{gen}}$ footprint up front; reserving lazily is how you OOM mid-decode.

## Further reading

- [What inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is) — the `nanoserve` frame and the TTFT/TPOT/memory scoreboard this post plugs into.
- [The memory math of the KV cache](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache) — the per-token byte derivation the §3 wall is built on.
- [Chunked prefill and the TTFT/TPOT tradeoff](/blog/machine-learning/inference-engineering/chunked-prefill-and-the-ttft-tpot-tradeoff) — the scheduling half of the prefill-bomb fix.
- [MLA and attention variants at inference time](/blog/machine-learning/inference-engineering/mla-and-attention-variants-at-inference-time) — where the 656-byte latent cache comes from and how MLA changes the memory math.
- [Admission control, backpressure, and latency collapse](/blog/machine-learning/inference-engineering/admission-control-backpressure-and-latency-collapse) — the predicate in §8, generalized.
- [Trainable sparse attention: NSA vs DSA](/blog/machine-learning/large-language-model/trainable-sparse-attention-nsa-vs-dsa) — the training-side comparison of the sparse-attention designs cited here.
- StreamingLLM — Xiao et al. 2023, [arXiv:2309.17453](https://arxiv.org/abs/2309.17453); YaRN — Peng et al. 2023, [arXiv:2309.00071](https://arxiv.org/abs/2309.00071); Position Interpolation — Chen et al. 2023, [arXiv:2306.15595](https://arxiv.org/abs/2306.15595).
- vLLM's [DeepSeek-V3.2-Exp](https://vllm.ai/blog/2025-09-29-deepseek-v3-2) and [DeepSeek-V4](https://vllm.ai/blog/2026-04-24-deepseek-v4) posts — the DSA and KV-compression mechanisms and numbers, cited throughout.
- The capstone, [the inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook), assembles these knobs with the rest of the series into one decision framework.

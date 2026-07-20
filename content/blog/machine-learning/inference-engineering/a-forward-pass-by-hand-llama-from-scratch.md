---
title: "A forward pass by hand: Llama from scratch in 200 lines"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Rebuild a Llama-3 decoder in pure PyTorch from config.json alone, derive its 8.03B parameters and its FLOPs from the shapes, then prove it matches the reference logit for logit."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "pytorch",
    "transformers",
    "rope",
    "attention",
    "gqa",
    "rmsnorm",
    "ml-systems",
    "gpu",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 47
---

You can write a Llama-3 forward pass in about two hundred lines of PyTorch with no `transformers` modeling code in sight. Most engineers who try get something that runs on the first afternoon, produces fluent-looking text on the second, and is quietly, subtly wrong the whole time. The model still speaks English. It still finishes sentences. It just picks a slightly different token every twenty or thirty steps, and on a hard prompt it falls apart in a way you will blame on the sampler, or the prompt, or the quantization you added three posts later.

That failure mode is the reason this post exists. Everything the rest of this series does — the KV cache, the paged allocator, the continuous-batching scheduler, the hand-written CUDA kernels, the INT4 dequant-fused GEMM — is an *optimization*, and an optimization is only meaningful relative to a reference you trust. If you cannot say "my logits match Hugging Face to within bf16 rounding," then every speedup you measure afterwards is unfalsifiable. You will not know whether your paged attention kernel is fast or just skipping work.

So this post writes `nanoserve/model.py`: the slowest, most obvious, most readable Llama-3-style decoder that can exist. No cache. No batching tricks. No fused anything. It reads `config.json`, turns eleven fields into a complete set of tensor shapes, and runs them. Then it writes `nanoserve/verify.py`, which loads the same weights into `transformers`, runs both on the same input, and tells you — with a tolerance you can defend — whether they agree.

![A configuration file fanning out into embedding attention and MLP tensor shapes that sum to eight billion parameters](/imgs/blogs/a-forward-pass-by-hand-llama-from-scratch-1.webp)

By the end you will be able to open any decoder-only model's `config.json`, predict its parameter count to the nearest ten million, predict its bf16 footprint in gigabytes, predict its per-token FLOPs, and predict how many tokens of context will fit in whatever VRAM is left over — before you download a single shard. That skill is worth more than the code. The code is the proof that you actually have it.

If you came here from [what inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is), this is the second brick in the wall: the previous post got the weights off disk and onto the device, and this one turns them into logits. The [inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) at the end of the series treats this correctness harness as the thing every later claim is checked against.

## 1. Read the config before you write a line of code

Open `config.json` from `meta-llama/Llama-3.1-8B-Instruct`. Strip the housekeeping and you are left with eleven fields that fully determine the model:

```json
{
  "hidden_size": 4096,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "intermediate_size": 14336,
  "rms_norm_eps": 1e-05,
  "rope_theta": 500000.0,
  "vocab_size": 128256,
  "tie_word_embeddings": false,
  "max_position_embeddings": 131072
}
```

Every one of those becomes a shape. Here is the mapping, which is worth memorizing because you will do it in your head for the rest of your career:

| Config field | Symbol | Value | What it becomes |
| --- | --- | --- | --- |
| `hidden_size` | $d_{\text{model}}$ | 4096 | Width of the residual stream; every layer's input and output |
| `num_hidden_layers` | $L$ | 32 | How many identical decoder blocks stack |
| `num_attention_heads` | $H_q$ | 32 | Query heads; `q_proj` output is $H_q \cdot d_h$ |
| `num_key_value_heads` | $H_{kv}$ | 8 | Key/value heads; `k_proj` output is $H_{kv} \cdot d_h$ |
| `head_dim` | $d_h$ | 128 | Per-head width; the $\sqrt{d}$ in the attention scale |
| `intermediate_size` | $d_{ff}$ | 14336 | MLP width; three matrices of $d_{\text{model}} \times d_{ff}$ |
| `rms_norm_eps` | $\epsilon$ | 1e-5 | Added inside the square root, not outside |
| `rope_theta` | $\theta_{\text{base}}$ | 500000 | Base of the rotary frequency ladder |
| `vocab_size` | $V$ | 128256 | Rows in the embedding table and the output head |
| `tie_word_embeddings` | — | false | Whether `lm_head` is a second copy of the embedding matrix |
| `max_position_embeddings` | — | 131072 | How long the cos/sin tables need to be |

Notice `head_dim` is stated explicitly here. It is 128, and $32 \times 128 = 4096 = d_{\text{model}}$, so the usual assumption `head_dim = hidden_size // num_attention_heads` happens to hold. Do not rely on that. Several recent models decouple the two — the head dimension is set independently of the residual width, and `q_proj` is no longer square. If your loader computes `head_dim` by division and the config states it, you have written a bug that only fires on the next model you try. Read the field; fall back to division only when it is absent.

The second thing to notice is `num_key_value_heads: 8` against `num_attention_heads: 32`. That ratio, four, is grouped-query attention, and it is the single most consequential number in the file for anyone who cares about serving. It does not change the parameter count much. It divides your KV cache by four. We will come back to it.

Third: `tie_word_embeddings: false`. Llama-3.1-8B carries two separate $128256 \times 4096$ matrices — one to turn token IDs into vectors, one to turn vectors back into logits. That is a deliberate choice with a real cost, and small models make the opposite one.

Let me turn the config into a dataclass, because every later file in `nanoserve` will take this object:

```python
# nanoserve/config.py
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LlamaConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    vocab_size: int
    tie_word_embeddings: bool
    max_position_embeddings: int
    rope_scaling: dict | None = None

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "LlamaConfig":
        raw = json.loads((Path(model_dir) / "config.json").read_text())
        n_heads = raw["num_attention_heads"]
        return cls(
            hidden_size=raw["hidden_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=n_heads,
            # older configs omit this entirely and mean MHA
            num_key_value_heads=raw.get("num_key_value_heads", n_heads),
            # read it if present; only divide as a fallback
            head_dim=raw.get("head_dim", raw["hidden_size"] // n_heads),
            intermediate_size=raw["intermediate_size"],
            rms_norm_eps=raw["rms_norm_eps"],
            rope_theta=raw.get("rope_theta", 10000.0),
            vocab_size=raw["vocab_size"],
            tie_word_embeddings=raw.get("tie_word_embeddings", False),
            max_position_embeddings=raw["max_position_embeddings"],
            rope_scaling=raw.get("rope_scaling"),
        )

    @property
    def n_rep(self) -> int:
        """How many query heads share each KV head."""
        return self.num_attention_heads // self.num_key_value_heads
```

The `rope_scaling` field is the one people skip, and skipping it is the reason a large fraction of from-scratch Llama-3.1 implementations never match the reference. We will handle it properly in section 5.

## 2. Where the 8.03 billion parameters actually live

Now derive the parameter count from those shapes alone. This is the mechanism block of the post: not an assertion, a construction.

Start with one decoder layer. It contains seven weight matrices and two norm vectors.

The attention projections:

$$
P_{\text{attn}} = \underbrace{d_{\text{model}} \cdot H_q d_h}_{q\_proj} + \underbrace{d_{\text{model}} \cdot H_{kv} d_h}_{k\_proj} + \underbrace{d_{\text{model}} \cdot H_{kv} d_h}_{v\_proj} + \underbrace{H_q d_h \cdot d_{\text{model}}}_{o\_proj}
$$

Substituting Llama-3.1-8B's numbers: $q$ and $o$ are each $4096 \times 4096 = 16{,}777{,}216$. The $k$ and $v$ projections output only $8 \times 128 = 1024$ features, so each is $4096 \times 1024 = 4{,}194{,}304$. That totals $41{,}943{,}040$ parameters of attention per layer.

The MLP is three matrices, all $d_{\text{model}} \times d_{ff}$ in one direction or the other:

$$
P_{\text{mlp}} = 3 \cdot d_{\text{model}} \cdot d_{ff} = 3 \times 4096 \times 14336 = 176{,}160{,}768
$$

The two RMSNorm weight vectors are $2 \times 4096 = 8192$ parameters — rounding error, but include them if you want the total to land exactly.

So one layer is $41{,}943{,}040 + 176{,}160{,}768 + 8192 = 218{,}112{,}000$ parameters. Thirty-two of them is $6{,}979{,}584{,}000$.

Outside the stack: the embedding table is $V \cdot d_{\text{model}} = 128256 \times 4096 = 525{,}336{,}576$. Because `tie_word_embeddings` is false, `lm_head` is a second matrix of the same size. The final `model.norm` adds 4096.

$$
N = 6{,}979{,}584{,}000 + 2 \times 525{,}336{,}576 + 4096 = 8{,}030{,}261{,}248
$$

Meta's [Llama-3.1-8B model card](https://huggingface.co/meta-llama/Llama-3.1-8B) lists the model at 8.03B parameters. Our shapes-only derivation lands on 8,030,261,248 — 8.03B. That agreement is the first checkpoint of the post: if your parameter formula and the model card disagree, you have misread a shape, and you will find out the hard way when a matmul refuses to contract.

![A layered budget showing the MLP holding most of the eight billion parameters with attention and embeddings smaller](/imgs/blogs/a-forward-pass-by-hand-llama-from-scratch-2.webp)

The proportions in that figure are the useful part. The MLP holds $32 \times 176{,}160{,}768 = 5{,}637{,}144{,}576$ parameters — 70.2% of the model. All attention projections together are $32 \times 41{,}943{,}040 = 1{,}342{,}177{,}280$, just 16.7%. Embeddings and the output head are 13.1%.

That distribution has direct operational consequences. At decode time, generating one token requires streaming essentially every weight from HBM into the SMs, so 70% of your memory traffic is the SwiGLU projections. If you are hunting for a quantization win, that is where the bytes are. If you are hunting for an attention-kernel win, you are optimizing a sixth of the traffic. This is exactly the reasoning the [roofline model](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) formalizes, and it is why so much inference work concentrates on weight-only quantization rather than on cleverer attention.

### From parameters to FLOPs

Every parameter in a linear layer participates in exactly one multiply and one add per token that passes through it. So the forward cost per token is

$$
F_{\text{token}} \approx 2N
$$

which for Llama-3.1-8B is $2 \times 8.03 \times 10^9 = 16.06$ GFLOP per token. Two caveats keep this honest. First, the embedding lookup is a gather, not a matmul — it contributes zero FLOPs despite holding 525M parameters, so ${2N}$ slightly overcounts. Second, the attention score computation ($QK^\top$ and $AV$) uses no parameters at all and is therefore missing from ${2N}$ entirely.

That second term is worth deriving because it is the one that grows with context. For a sequence of length $S$, per layer, $QK^\top$ costs $2 S^2 H_q d_h$ and $A V$ costs the same, so:

$$
F_{\text{attn}} = 4 \cdot L \cdot S^2 \cdot H_q d_h = 4 \cdot 32 \cdot S^2 \cdot 4096
$$

Causal masking means only half the score matrix is ever needed, so a good kernel halves it. At $S = 2048$: $2 \times 32 \times 2048^2 \times 4096 \approx 1.10$ TFLOP, against $2048 \times 16.06\ \text{GFLOP} = 32.9$ TFLOP for the parameter term. Attention is 3.2% of prefill FLOPs at 2k context. At 32k it is roughly a third. At 128k it dominates. Hold that thought — it is the whole reason [long-context prefill](/blog/machine-learning/large-language-model/kv-cache) behaves so differently from short-context prefill.

#### Worked example: prefill and decode cost for one RAG request

Take the RAG shape from this series' standard prompt suite: 2048 input tokens, 256 output tokens, Llama-3.1-8B in bf16, one RTX 4090.

**Prefill.** $2048 \times 16.06\ \text{GFLOP} + 1.10\ \text{TFLOP} \approx 34.0$ TFLOP of arithmetic. NVIDIA's [RTX 4090 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/) list 165.2 TFLOP/s of FP16/BF16 tensor-core throughput with FP32 accumulate (dense). At a perfect 100% of peak that is 206 ms; real dense GEMM stacks on consumer silicon land well below peak, so a reader running this should expect something in the several-hundred-millisecond range and treat anything under 250 ms as suspicious. *(Source: derived from the FLOP formula above plus a cited vendor spec.)*

**Decode.** 256 tokens × 16.06 GFLOP = 4.11 TFLOP — an eighth of the prefill's arithmetic, spread over 256 separate steps. But each of those steps must read all 16.06 GB of weights from HBM, because a batch of one gives you no opportunity to amortize. Total traffic: $256 \times 16.06\ \text{GB} = 4.11$ TB. The 4090 lists 1008 GB/s of memory bandwidth, giving $4.11\ \text{TB} / 1.008\ \text{TB/s} = 4.08$ s, or a hard ceiling of **62.7 tok/s**. *(Source: derived from parameter count and a cited bandwidth spec.)*

The arithmetic in that decode phase would take 4.11 TFLOP / 165.2 TFLOP/s = 25 ms if the GPU could be fed. It takes 4.08 seconds instead. The ratio — about 160× — is the entire justification for the next thirty posts in this series. A batch-1 decode step uses roughly 0.6% of the 4090's compute.

Run it yourself and report what you get: a naive PyTorch loop with no cache, no `torch.compile`, and Python overhead on every step will land far *below* 62.7 tok/s, and the gap between what you measure and the 62.7 ceiling is precisely the budget the rest of the series spends.

## 3. The embedding table: a lookup that costs two gigabytes

The first operation in the model is the cheapest and one of the most expensive. `model.embed_tokens.weight` is a $128256 \times 4096$ matrix. In bf16 that is $128256 \times 4096 \times 2 = 1{,}050{,}673{,}152$ bytes — 1.05 GB. With `tie_word_embeddings: false`, `lm_head.weight` is another 1.05 GB. Two gigabytes of a sixteen-gigabyte model, 13.1%, spent on a table lookup and a final projection.

The forward operation itself is trivial:

```python
# a gather, not a matmul: zero FLOPs, one row read per token
h = embed_tokens[input_ids]          # [B, S] int64 -> [B, S, 4096]
```

Vocabulary size is the knob nobody thinks about until it bites. Compare three models:

| Model | $V$ | $d_{\text{model}}$ | Embedding params | Tied? | Share of total |
| --- | --- | --- | --- | --- | --- |
| Llama-3.1-8B | 128,256 | 4096 | 525M × 2 | no | 13.1% of 8.03B |
| Llama-3.2-1B | 128,256 | 2048 | 263M × 1 | yes | 21.3% of 1.24B |
| Qwen3-8B | 151,936 | 4096 | 622M | no | derived from config |

*(Source: derived from each model's published config fields; parameter totals cross-checked against the respective model cards.)*

Llama-3.2-1B is the instructive case. Its config gives $d_{\text{model}} = 2048$, $L = 16$, $H_q = 32$, $H_{kv} = 8$, $d_h = 64$, $d_{ff} = 8192$, and `tie_word_embeddings: true`. Run the same formula: attention per layer is $2048 \times 2048 + 2 \times (2048 \times 512) + 2048 \times 2048 = 10{,}485{,}760$; the MLP is $3 \times 2048 \times 8192 = 50{,}331{,}648$; with norms that is $60{,}821{,}504$ per layer, $973{,}144{,}064$ for sixteen layers. Add one shared $128256 \times 2048 = 262{,}668{,}288$ embedding matrix and a final norm: **1,235,814,400**, which is the 1.24B the model card advertises.

The embedding is 21.3% of that model. Untying it would add another 263M parameters — a 21% larger model for the same depth and width. That is why every sub-2B model you will meet ties its embeddings, and why the trend reverses above roughly 7B: a fixed 128k vocabulary is a rounding error at 70B and a structural cost at 1B.

There is an inference-time consequence too. When embeddings are tied, the `lm_head` matmul reads the same 1.05 GB the embedding lookup already touched. When they are untied, that is a distinct 1.05 GB of HBM traffic on the last op of every single decode step — 6.5% of your per-token bandwidth budget spent producing logits for 128,256 tokens when you are about to keep one. Later posts in this series will look at whether that projection can be shrunk; for now, note that it is not free.

## 4. RMSNorm: the formula, and the fp32 detail that decides whether you match

LayerNorm subtracts the mean, divides by the standard deviation, then scales and shifts. RMSNorm, introduced by [Zhang and Sennrich (2019)](https://arxiv.org/abs/1910.07467), drops the mean subtraction and the bias entirely:

$$
\text{RMSNorm}(x)_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2 + \epsilon}} \cdot g_i
$$

The paper's argument is that re-centering contributes little to the benefit of normalization and that re-scaling does the work. The engineering argument is simpler: you save one pass over the vector and one buffer, and every op you remove from a layer is an op you remove 32 times per token.

Two details in that formula are where implementations diverge.

**Where $\epsilon$ goes.** It is inside the square root, added to the mean of squares. Putting it outside — dividing by $\sqrt{\overline{x^2}} + \epsilon$ — is a different function. With $\epsilon = 10^{-5}$ and typical activation magnitudes the numerical difference is small enough to hide, which is exactly what makes it dangerous: your model will work, and it will be a few thousandths off, forever.

**Where the fp32 cast goes.** The reference implementation computes the mean of squares in fp32, applies the reciprocal square root in fp32, casts the *normalized activation* back to the input dtype, and only then multiplies by the weight. That ordering matters. Summing 4096 squared bf16 values in bf16 is genuinely bad: bf16 carries eight significand bits, so its machine epsilon is $2^{-7} \approx 0.0078$, and a naive sequential accumulation of thousands of terms loses precision fast. But the *final* multiply by `g` happens in the low precision, because that is what the reference does, and matching the reference is the job.

```python
# nanoserve/model.py
import torch


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Match transformers' LlamaRMSNorm exactly, including the cast order."""
    in_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)      # mean of squares, fp32
    x = x * torch.rsqrt(variance + eps)             # eps INSIDE the sqrt
    return weight * x.to(in_dtype)                  # cast back, then scale
```

If you write this as `(weight.float() * x).to(in_dtype)` instead, you will get a slightly different answer — usually within a few units in the last place, occasionally enough to flip a marginal argmax. When you are bisecting a divergence later in this post and layer 0 is already off by $10^{-3}$, this line is the first place to look.

Llama places two RMSNorms per layer, both pre-norm: one on the input to attention, one on the input to the MLP. The residual stream itself is never normalized in place — it accumulates unnormalized, and each sub-block reads a normalized copy. That is why the residual stream's magnitude grows with depth and why the final `model.norm` exists at all: it is the one normalization standing between a 32-layer accumulation and the output projection. Forgetting it is a classic bug, and it has a distinctive signature we will name in section 9.

## 5. RoPE: rotation, relative position, and the layout that breaks everything

Rotary position embedding, from [Su et al. (2021)](https://arxiv.org/abs/2104.09864), is the piece most likely to be *almost* right. It is also, once you see the derivation, the most elegant idea in the architecture.

### The derivation

Attention scores are dot products $q_m \cdot k_n$ between a query at position $m$ and a key at position $n$. We want the score to depend on the relative offset $m - n$, not on the absolute positions, because "the word three tokens back" is a more useful signal than "the word at index 4,097."

Take a two-dimensional slice of the head. Rotate the query by an angle proportional to its position, $m\theta$, and the key by $n\theta$. Write $R(\alpha)$ for the 2D rotation matrix. Then:

$$
(R(m\theta)\, q) \cdot (R(n\theta)\, k) = q^\top R(m\theta)^\top R(n\theta)\, k = q^\top R\big((n-m)\theta\big)\, k
$$

using the fact that rotations are orthogonal, so $R(\alpha)^\top R(\beta) = R(\beta - \alpha)$. The absolute positions cancel. What survives is a rotation by the *difference*. Absolute rotation in, relative dependence out — with no learned parameters and no extra memory at attention time.

A single 2D rotation only gives you one frequency, which wraps around and cannot distinguish offset 3 from offset $3 + 2\pi/\theta$. So RoPE splits the 128-dimensional head into 64 independent 2D pairs and assigns each pair its own frequency:

$$
\theta_i = \theta_{\text{base}}^{-2i/d_h}, \qquad i = 0, 1, \dots, d_h/2 - 1
$$

Pair 0 rotates fastest ($\theta_0 = 1$ radian per position). Pair 63 rotates slowest ($\theta_{63} = \theta_{\text{base}}^{-126/128}$, which for $\theta_{\text{base}} = 500000$ is about $1.2 \times 10^{-6}$ radians per position, a wavelength of roughly five million tokens). Together they encode offset the way a set of clock hands encodes time: fast hands give precision, slow hands give range.

<figure class="blog-anim">
<svg viewBox="0 0 700 280" role="img" aria-label="Two vectors on a circle sweep together as position advances while the shaded angle between them stays fixed" style="width:100%;height:auto;max-width:720px">
<style>
.ie3-ring{fill:none;stroke:var(--border,#d1d5db);stroke-width:1.5}
.ie3-ax{stroke:var(--border,#d1d5db);stroke-width:1;stroke-dasharray:5 5}
.ie3-q{stroke:var(--accent,#6366f1);stroke-width:4.5;stroke-linecap:round}
.ie3-k{stroke:var(--text-secondary,#6b7280);stroke-width:4.5;stroke-linecap:round}
.ie3-wedge{fill:var(--accent,#6366f1);opacity:.15}
.ie3-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.ie3-sub{font:400 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.ie3-eq{font:600 15px ui-monospace,ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:start}
.ie3-note{font:400 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:start}
@keyframes ie3-spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
.ie3-spin{transform-box:view-box;transform-origin:170px 140px;animation:ie3-spin 16s linear infinite}
@media (prefers-reduced-motion:reduce){.ie3-spin{animation:none}}
</style>
<circle class="ie3-ring" cx="170" cy="140" r="92"/>
<line class="ie3-ax" x1="60" y1="140" x2="280" y2="140"/>
<line class="ie3-ax" x1="170" y1="30" x2="170" y2="250"/>
<g transform="rotate(-55 170 140)"><path class="ie3-wedge ie3-spin" d="M 170 140 L 262 140 A 92 92 0 0 1 222.8 62.8 Z"/></g>
<g transform="rotate(-55 170 140)"><line class="ie3-k ie3-spin" x1="170" y1="140" x2="256" y2="140"/></g>
<g><line class="ie3-q ie3-spin" x1="170" y1="140" x2="256" y2="140"/></g>
<circle cx="170" cy="140" r="5" fill="var(--text-primary,#1f2937)"/>
<text class="ie3-lbl" x="170" y="268">one 2D pair of a 128-dim head</text>
<text class="ie3-eq" x="330" y="70">q at position m  -&gt;  rotate by m x theta</text>
<text class="ie3-eq" x="330" y="100">k at position n  -&gt;  rotate by n x theta</text>
<text class="ie3-eq" x="330" y="146">q . k  =  f( (m - n) x theta )</text>
<text class="ie3-note" x="330" y="180">Advance both positions by the same amount</text>
<text class="ie3-note" x="330" y="200">and both vectors sweep together, but the</text>
<text class="ie3-note" x="330" y="220">shaded angle between them never changes.</text>
<text class="ie3-note" x="330" y="240">That invariance is the whole point of RoPE.</text>
</svg>
<figcaption>Both vectors rotate as the sequence slides forward, yet the angle between them stays fixed, which is why the attention score depends only on the offset and not on absolute position.</figcaption>
</figure>

Watching that loop is the fastest way to internalize why RoPE needs no relative-position bias table and no extra attention-time memory. The information is baked into the geometry of the vectors themselves.

### What `rope_theta` actually does

Llama-2 used $\theta_{\text{base}} = 10000$. Llama-3 uses 500000. Raising the base compresses the frequency ladder toward the slow end: every pair rotates more slowly, so the slowest pair's wavelength stretches from roughly 82,000 positions to roughly 5.2 million. In exchange, the fast pairs lose some of their ability to resolve very short offsets.

The practical reading: a larger base buys long-context capacity at the cost of some short-range positional sharpness. It is one of the few architectural changes that can be applied post-hoc, which is why "just raise rope_theta and fine-tune briefly" became a standard context-extension recipe.

### The scaling function you cannot skip

Llama-3.1 goes further. Its config contains:

```json
{
  "rope_scaling": {
    "rope_type": "llama3",
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192
  }
}
```

This is not a runtime switch that only activates past 8192 tokens. It rewrites `inv_freq` itself, at build time, for every position including position zero. Frequencies whose wavelength exceeds the original 8192-token context get divided by the factor; frequencies well inside it are left alone; the band in between is interpolated. If you implement plain RoPE and skip this, your logits will be wrong on a ten-token prompt — a fact that surprises almost everyone the first time.

```python
# nanoserve/model.py
import math
import torch


def _llama3_inv_freq(inv_freq: torch.Tensor, cfg) -> torch.Tensor:
    """Apply the llama3 frequency rescaling described in config.rope_scaling."""
    s = cfg.rope_scaling
    factor = s["factor"]                                   # 8.0
    low_f = s["low_freq_factor"]                           # 1.0
    high_f = s["high_freq_factor"]                         # 4.0
    old_ctx = s["original_max_position_embeddings"]        # 8192

    low_wavelen = old_ctx / low_f                          # 8192
    high_wavelen = old_ctx / high_f                        # 2048
    wavelen = 2 * math.pi / inv_freq

    # long wavelengths get stretched by the full factor
    scaled = torch.where(wavelen > low_wavelen, inv_freq / factor, inv_freq)
    # the middle band is a smooth blend between scaled and unscaled
    smooth = (old_ctx / wavelen - low_f) / (high_f - low_f)
    blended = (1 - smooth) * scaled / factor + smooth * scaled
    is_mid = (wavelen >= high_wavelen) & (wavelen <= low_wavelen)
    return torch.where(is_mid, blended, scaled)


def build_rope_cache(cfg, device, max_pos: int | None = None):
    """Precompute cos and sin tables once; every layer reuses them."""
    d = cfg.head_dim
    idx = torch.arange(0, d, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (cfg.rope_theta ** (idx / d))         # [d/2]
    if cfg.rope_scaling and cfg.rope_scaling.get("rope_type") == "llama3":
        inv_freq = _llama3_inv_freq(inv_freq, cfg)

    max_pos = max_pos or cfg.max_position_embeddings
    t = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)                       # [max_pos, d/2]
    emb = torch.cat((freqs, freqs), dim=-1)                # [max_pos, d]
    return emb.cos(), emb.sin()                            # keep in fp32
```

Precomputing is not an optimization here, it is table stakes: `cos` and `sin` are functions of position and frequency only, so computing them per layer would repeat 32 identical transcendental evaluations per forward pass for nothing. Build them once at load time, keep them in fp32 (they are small — 131072 × 128 × 4 bytes × 2 tables = 134 MB for the full 128k context, and you should size them to the context you actually serve, not the maximum the config allows).

### The layout bug

Here is where hand-written Llama implementations go to die.

![Two side-by-side columns contrasting interleaved rotary pairing against half-split pairing with only one producing matching logits](/imgs/blogs/a-forward-pass-by-hand-llama-from-scratch-3.webp)

RoPE pairs up dimensions. Which dimensions? There are two conventions:

- **Interleaved**: pair $(x_0, x_1)$, $(x_2, x_3)$, …, $(x_{126}, x_{127})$. Adjacent dimensions form each 2D plane. This is what Meta's original reference code does, using complex arithmetic on `view_as_complex`.
- **Half-split**: pair $(x_0, x_{64})$, $(x_1, x_{65})$, …, $(x_{63}, x_{127})$. The first half pairs with the second half. This is what `transformers` does, via a function called `rotate_half`.

These are genuinely different functions. Applying the wrong one does not raise an error, does not produce NaN, and does not obviously break generation — it produces a model that is positionally confused in a subtle, distributed way.

The resolution is a detail buried in the conversion script: when Meta's checkpoints are converted to the Hugging Face format, the `q_proj` and `k_proj` weight matrices are **permuted** so that the half-split convention reproduces the interleaved result. The weights on disk are not the weights Meta trained; they are those weights with their output rows shuffled. So the rule is mechanical:

- Loading a Hugging Face `.safetensors` checkpoint? Use **half-split** (`rotate_half`).
- Loading Meta's original `consolidated.00.pth`? Use **interleaved**.
- Loading GGUF? Check the converter — llama.cpp handles this explicitly and the answer depends on which path produced the file.

```python
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Half-split pairing: the convention Hugging Face checkpoints expect."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin, positions):
    """q, k: [B, H, S, D]. cos, sin: [max_pos, D]. positions: [S] int64."""
    c = cos[positions].unsqueeze(0).unsqueeze(0)   # [1, 1, S, D]
    s = sin[positions].unsqueeze(0).unsqueeze(0)
    q_out = (q.float() * c + rotate_half(q.float()) * s).to(q.dtype)
    k_out = (k.float() * c + rotate_half(k.float()) * s).to(k.dtype)
    return q_out, k_out
```

Note the `positions` argument rather than an implicit `arange`. It costs nothing now and saves a rewrite in the KV-cache post, where a decode step passes a single position that is nowhere near zero.

Note also the fp32 promotion around the rotation. RoPE is a rotation, so it preserves vector norm exactly in exact arithmetic; in bf16 it does not, and the error compounds over 32 layers. Doing the multiply-add in fp32 and casting back matches the reference and costs a negligible amount of arithmetic on a memory-bound step.

**The diagnostic trick.** If you suspect a RoPE bug, replace `cos` with all-ones and `sin` with all-zeros in both your implementation and the reference. That makes RoPE the identity in both. If they now agree to within bf16 noise, your bug is in RoPE. If they still disagree, it is somewhere else and you have just eliminated the hardest suspect in two lines.

## 6. Attention with grouped queries

Now the part everyone thinks is the hard part, which is actually the easy part once the shapes are straight.

![Four query head nodes converging onto a single key value head that feeds the per-token cache cost](/imgs/blogs/a-forward-pass-by-hand-llama-from-scratch-4.webp)

Llama-3.1-8B projects the 4096-wide residual into 32 query heads of 128 dimensions each — but only 8 key heads and 8 value heads. Query heads 0 through 3 share key/value head 0, heads 4 through 7 share head 1, and so on. That is grouped-query attention, from [Ainslie et al. (2023)](https://arxiv.org/abs/2305.13245), and $n_{\text{rep}} = H_q / H_{kv} = 4$.

The parameter saving is real but modest: `k_proj` and `v_proj` shrink from 16.8M to 4.2M each, saving 25.1M per layer, 803M across the model — 9% of the parameters. The *cache* saving is the one that matters. Bytes of KV cache per token:

$$
B_{\text{token}} = 2 \cdot L \cdot H_{kv} \cdot d_h \cdot \text{sizeof(dtype)} = 2 \times 32 \times 8 \times 128 \times 2 = 131{,}072
$$

That is exactly 128 KiB per token. With multi-head attention ($H_{kv} = 32$) it would be 512 KiB. On a 24 GB RTX 4090 holding 16.06 GB of weights, you have roughly 8 GB of headroom after activations and allocator overhead, which is $8 \times 2^{30} / 131072 = 65{,}536$ tokens of context in aggregate. Under MHA that would be 16,384. GQA quadrupled how many concurrent users fit on the card, and it did so without touching the compute. That is why it was the biggest single inference win of 2023, and it is the starting point for the further compression that [multi-head latent attention](/blog/machine-learning/large-language-model/multi-head-latent-attention-mla) pushes much further.

### Repeating the KV heads

Standard attention math wants $H_q$ keys and $H_q$ values. GQA gives you $H_{kv}$ of each. The bridge is a repeat:

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[B, H_kv, S, D] -> [B, H_kv * n_rep, S, D], each KV head duplicated n_rep times."""
    if n_rep == 1:
        return x
    b, h_kv, s, d = x.shape
    x = x[:, :, None, :, :].expand(b, h_kv, n_rep, s, d)
    return x.reshape(b, h_kv * n_rep, s, d)
```

The ordering here is load-bearing. `expand` inserts the repetition axis *inside* the head axis, so head $j$ of the output maps to KV head $\lfloor j / n_{\text{rep}} \rfloor$. Query head 3 must read KV head 0, not KV head 3. If you instead write `x.repeat(1, n_rep, 1, 1)`, you get the tiling `[kv0, kv1, ..., kv7, kv0, kv1, ...]`, which maps query head 3 to KV head 3. The shapes are identical. The model is wrong. This is the single most common GQA bug and it has a beautiful symptom, described in section 9.

The `expand` is a view, so nothing is copied until `reshape` forces it — and `reshape` will copy here, materializing a 4× larger tensor. A production kernel never does this; FlashAttention-style kernels index the KV heads directly. PyTorch 2.5 added `enable_gqa=True` to `scaled_dot_product_attention`, which pushes the repetition into the kernel where it belongs. We write `repeat_kv` anyway because it is explicit, and explicit is what a reference implementation is for.

### The scores, the mask, and the softmax

Written out by hand, with nothing hidden:

```python
import math
import torch
import torch.nn.functional as F


def attention_manual(q, k, v, mask):
    """q: [B, Hq, S, D]; k, v: [B, Hq, S, D] after repeat_kv; mask: [S, S] additive."""
    d_h = q.shape[-1]
    scale = 1.0 / math.sqrt(d_h)                       # 1/sqrt(128) = 0.08838835
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale   # [B, Hq, S, S]
    scores = scores + mask                             # -inf above the diagonal
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)  # fp32 softmax
    return torch.matmul(probs, v)                      # [B, Hq, S, D]


def causal_mask(seq_len: int, device, dtype):
    """Additive mask: 0 where attention is allowed, -inf where it is not."""
    m = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
    return torch.triu(m, diagonal=1)                   # diagonal=1 keeps self-attention
```

Three details, each of which is a bug someone has shipped.

**The scale.** $1/\sqrt{d_h}$ with $d_h = 128$, not $1/\sqrt{d_{\text{model}}}$. The reason for the scale at all: if $q$ and $k$ have unit-variance independent components, their dot product over $d_h$ dimensions has variance $d_h$, so raw scores grow with head width and drive the softmax toward a one-hot distribution. Dividing by $\sqrt{d_h}$ restores unit variance and keeps gradients — and, at inference, the attention distribution — in a sane range. On models where `head_dim` is decoupled from `hidden_size`, using the wrong one produces a model that is systematically over- or under-confident in its attention.

**The mask diagonal.** `torch.triu(m, diagonal=1)` masks the *strict* upper triangle. Position $t$ can attend to positions ${0}$ through $t$ inclusive. Using `diagonal=0` masks the diagonal too, meaning position 0 can attend to nothing, its entire score row is $-\infty$, softmax produces NaN, and the NaN propagates through every subsequent layer. That failure is loud, which makes it the easiest of these bugs. The quiet version is building the mask with the wrong orientation — masking the lower triangle — which turns your causal decoder into an anti-causal one that only sees the future. It generates fluent text and scores terribly.

**The fp32 softmax.** `torch.softmax(scores.float(), dim=-1)`. Softmax involves an exponential and a sum over the whole sequence. In bf16, with eight significand bits, summing thousands of exponentials loses real precision, and the subtraction of the running max that keeps `exp` from overflowing is itself precision-sensitive. PyTorch's own SDPA kernels accumulate in fp32 internally. If you write the softmax by hand and leave it in bf16, you will not match, and the mismatch will grow with sequence length in a way that looks like a mask bug.

### Or: let SDPA do it

```python
def attention_sdpa(q, k, v, n_rep: int):
    """Same math, one call. PyTorch selects the backend."""
    k = repeat_kv(k, n_rep)
    v = repeat_kv(v, n_rep)
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

`torch.nn.functional.scaled_dot_product_attention` dispatches at runtime among several backends: a FlashAttention-2 kernel, a memory-efficient kernel, a cuDNN kernel, and a fallback that is essentially the manual code above. The selection depends on dtype, head dimension, alignment, whether an explicit mask was passed, and the device's compute capability. `is_causal=True` is not just a convenience — it lets the kernel skip computing the masked half of the score matrix entirely, roughly halving prefill attention FLOPs, and it avoids materializing an $S \times S$ mask tensor. At 8192 tokens that mask alone would be 128 MB in bf16 per batch element.

Two warnings that will matter later:

1. **`is_causal=True` aligns the mask to the top-left.** For prefill, where query length equals key length, that is exactly right. For a decode step with one query against $S$ cached keys, top-left alignment means the single query attends only to position 0 — catastrophically wrong, and silent. The KV-cache post handles this properly; for now our query length always equals our key length, so we are safe.
2. **SDPA's backend choice is not stable across your machines.** The same code can take the Flash path on an A100 and the math fallback on an older card, and the two do not produce bit-identical results. When you verify, pin the reference to `attn_implementation="eager"` so you are comparing against a known quantity.

We ship both implementations in `nanoserve` and keep the manual one behind a flag. It is 40× slower and it is the thing you switch to when SDPA and the reference disagree and you need to see the score matrix.

## 7. SwiGLU: three matrices and a suspicious ratio

The MLP in a classic transformer is two matrices with a nonlinearity between them: $W_2\,\sigma(W_1 x)$, with $d_{ff} = 4 d_{\text{model}}$. Llama uses SwiGLU, from [Shazeer (2020)](https://arxiv.org/abs/2002.05202), which is three matrices:

$$
\text{SwiGLU}(x) = W_{\text{down}}\Big( \text{SiLU}(W_{\text{gate}}\, x) \odot (W_{\text{up}}\, x) \Big)
$$

where $\text{SiLU}(z) = z \cdot \sigma(z)$ and $\odot$ is elementwise multiplication. One branch produces a value, the other produces a gate, and the gate modulates the value multiplicatively before the down-projection. Gating gives the layer a cheap way to express conditional behavior that a single nonlinearity cannot.

```python
def swiglu(x, w_gate, w_up, w_down):
    """x: [B, S, d_model]; w_*: [out, in] as stored in HF checkpoints."""
    gate = F.silu(F.linear(x, w_gate))     # [B, S, d_ff]
    up = F.linear(x, w_up)                 # [B, S, d_ff]
    return F.linear(gate * up, w_down)     # [B, S, d_model]
```

Use `F.linear`, not `x @ w`. Hugging Face stores linear weights as `[out_features, in_features]` and `F.linear` handles the transpose internally. Writing `x @ w_gate` will either throw a shape error (good) or, on a square matrix like `q_proj`, silently compute the transpose of what you wanted (very bad). Square projections are exactly where this bug hides, which is why it shows up in `q_proj` and `o_proj` and never in `k_proj`.

### Why 3.5× and not 4×

Three matrices instead of two means SwiGLU costs 50% more parameters at the same $d_{ff}$. Shazeer's paper handles this by shrinking $d_{ff}$ by two-thirds, so $\frac{8}{3} d_{\text{model}} \approx 2.67 d_{\text{model}}$ across three matrices holds the parameter count equal to $4 d_{\text{model}}$ across two. That is where the folklore "SwiGLU uses 8/3" comes from, and Llama-2-7B follows it: ${11008 / 4096 = 2.6875}$, which is $\tfrac{8}{3} \times 4096 = 10922.7$ rounded up to a multiple of 256.

Llama-3.1-8B does **not** follow it. ${14336 / 4096 = 3.5}$ exactly. That is 31% more MLP parameters than a parameter-matched $4\times$ two-matrix MLP would have:

| MLP variant | Matrices | $d_{ff}$ | Params per layer | vs. Llama-3 |
| --- | --- | --- | --- | --- |
| GELU 2-matrix, $4\times$ | 2 | 16384 | 134,217,728 | 0.76× |
| SwiGLU, $8/3 \times$ (Llama-2 style) | 3 | 11008 | 135,266,304 | 0.77× |
| SwiGLU, $3.5\times$ (Llama-3.1-8B) | 3 | 14336 | 176,160,768 | 1.00× |

*(Source: derived from the shape formula $3 \cdot d_{\text{model}} \cdot d_{ff}$ with $d_{\text{model}} = 4096$.)*

So the honest answer to "why 3.5×" is not a derivation, it is a design decision: Meta spent extra capacity in the MLP rather than in depth or width. If someone tells you 3.5× follows from the 8/3 rule, they have the right rule and the wrong model. What matters operationally is that you read `intermediate_size` from the config instead of computing it, because the ratio varies across model families and getting it wrong is a shape error at best and a silent 30% parameter miscount at worst.

## 8. Assembling the block and the model

Everything above composes into a decoder layer with two pre-norm sub-blocks and two residual adds.

![A left to right sequence of seven layer operations with tensor shapes annotated at each step](/imgs/blogs/a-forward-pass-by-hand-llama-from-scratch-5.webp)

The invariant worth naming: **the residual stream never changes width**. It enters the layer as `[B, S, 4096]` and leaves as `[B, S, 4096]`. Attention temporarily reshapes it into heads; the MLP temporarily expands it to 14336. Both project back. That invariant is what lets you stack 32 identical layers, and it is what makes the residual stream a single continuous channel from the embedding to the final norm.

```python
# nanoserve/model.py
import torch
import torch.nn.functional as F


class Llama:
    """A weights dict plus a config. No nn.Module, no autograd, no surprises."""

    def __init__(self, cfg, weights: dict[str, torch.Tensor], device):
        self.cfg = cfg
        self.w = weights
        self.cos, self.sin = build_rope_cache(cfg, device)

    def layer(self, h, i, cos, sin, positions, mask):
        cfg, w = self.cfg, self.w
        p = f"model.layers.{i}"
        b, s, _ = h.shape
        hq, hkv, dh = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim

        # --- attention sub-block ---------------------------------------
        x = rms_norm(h, w[f"{p}.input_layernorm.weight"], cfg.rms_norm_eps)
        q = F.linear(x, w[f"{p}.self_attn.q_proj.weight"]).view(b, s, hq, dh)
        k = F.linear(x, w[f"{p}.self_attn.k_proj.weight"]).view(b, s, hkv, dh)
        v = F.linear(x, w[f"{p}.self_attn.v_proj.weight"]).view(b, s, hkv, dh)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))   # -> [B, H, S, D]

        q, k = apply_rope(q, k, cos, sin, positions)
        k, v = repeat_kv(k, cfg.n_rep), repeat_kv(v, cfg.n_rep)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        a = a.transpose(1, 2).reshape(b, s, hq * dh)       # merge heads back
        h = h + F.linear(a, w[f"{p}.self_attn.o_proj.weight"])

        # --- MLP sub-block ---------------------------------------------
        x = rms_norm(h, w[f"{p}.post_attention_layernorm.weight"], cfg.rms_norm_eps)
        gate = F.silu(F.linear(x, w[f"{p}.mlp.gate_proj.weight"]))
        up = F.linear(x, w[f"{p}.mlp.up_proj.weight"])
        h = h + F.linear(gate * up, w[f"{p}.mlp.down_proj.weight"])
        return h

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        cfg, w = self.cfg, self.w
        b, s = input_ids.shape
        positions = torch.arange(s, device=input_ids.device)

        h = w["model.embed_tokens.weight"][input_ids]      # [B, S, d_model]
        for i in range(cfg.num_hidden_layers):
            h = self.layer(h, i, self.cos, self.sin, positions, None)

        h = rms_norm(h, w["model.norm.weight"], cfg.rms_norm_eps)   # final norm
        head = (w["model.embed_tokens.weight"] if cfg.tie_word_embeddings
                else w["lm_head.weight"])
        return F.linear(h, head)                           # [B, S, vocab_size]
```

That is the whole model. Counting the config dataclass, `rms_norm`, `build_rope_cache`, `apply_rope`, `repeat_kv` and the class above, it is roughly 200 lines with docstrings.

Some deliberate choices:

- **`@torch.inference_mode()`, not `no_grad()`.** Inference mode goes further: tensors created inside it cannot be recorded by autograd at all and skip version-counter bookkeeping. For a forward-only engine there is no reason to use anything weaker.
- **A plain dict of tensors, not `nn.Module`.** `nn.Module` buys parameter registration, `state_dict`, device movement and hooks, none of which an inference engine needs. What it costs is a layer of indirection between you and the tensor, and later in this series that indirection is exactly what gets in the way when you want to swap a weight for a quantized surrogate or point an attention op at a paged cache.
- **`mask=None` because SDPA's `is_causal` handles it.** The manual path builds the explicit mask.

Loading the weights is the previous post's job; the short version is `safetensors.torch.load_file` per shard, merged into one dict, with each tensor `.to(device, dtype=torch.bfloat16)`.

```python
# nanoserve/run.py
import torch
from transformers import AutoTokenizer

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
cfg = LlamaConfig.from_pretrained(MODEL)
weights = load_safetensors(MODEL, device="cuda", dtype=torch.bfloat16)
model = Llama(cfg, weights, device="cuda")

tok = AutoTokenizer.from_pretrained(MODEL)
ids = tok("The capital of France is", return_tensors="pt").input_ids.cuda()
logits = model.forward(ids)
print(logits.shape)                       # torch.Size([1, 6, 128256])
print(tok.decode(logits[0, -1].argmax()))
```

```console
torch.Size([1, 6, 128256])
 Paris
```

Getting ` Paris` out is satisfying and proves almost nothing. A model with the wrong RoPE convention will also say ` Paris`. That prompt is so overdetermined that a badly broken model still gets it. Which brings us to the actual work.

## 9. The verification harness

This is the part of the post worth keeping. Everything before it is architecture you could read in a paper; this is the practice that makes the architecture trustworthy.

The plan: load the same weights into `transformers`, run both models on the same input, and compare. Not eyeball the text — compare the tensors, with a tolerance you can justify.

```python
# nanoserve/verify.py
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEV = "cuda"


def compare(mine: torch.Tensor, ref: torch.Tensor, name: str) -> dict:
    a, b = mine.float(), ref.float()
    diff = (a - b).abs()
    stats = {
        "name": name,
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        # relative Frobenius error: scale-free, the number to actually watch
        "rel_fro": ((a - b).norm() / b.norm()).item(),
        "cos": F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item(),
        "ref_scale": b.abs().mean().item(),
    }
    print(
        f"{name:28s} max={stats['max_abs']:.3e}  "
        f"rel={stats['rel_fro']:.3e}  cos={stats['cos']:.7f}"
    )
    return stats


tok = AutoTokenizer.from_pretrained(MODEL)
ids = tok(
    "In 1969 the Apollo 11 mission landed the first humans on the Moon, "
    "an achievement that required solving problems in guidance, propulsion, "
    "and thermal control that had never been attempted at that scale.",
    return_tensors="pt",
).input_ids.to(DEV)

ref_model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",   # pin the backend so the comparison is stable
).to(DEV).eval()

cfg = LlamaConfig.from_pretrained(MODEL)
mine = Llama(cfg, load_safetensors(MODEL, DEV, torch.bfloat16), DEV)

with torch.inference_mode():
    ref_logits = ref_model(ids).logits
    my_logits = mine.forward(ids)

compare(my_logits, ref_logits, "logits")
```

Use a real prompt, not `"Hello"`. Short prompts exercise almost nothing: they cannot reveal a mask orientation bug, a positional bug beyond position 2, or a numerical drift that only shows up after a few dozen tokens. Sixty tokens of prose with varied vocabulary is a much better test, and it costs nothing.

Pin `attn_implementation="eager"`. `transformers` defaults to SDPA where available, and SDPA picks its backend based on your hardware. Comparing two moving targets is how you spend an afternoon chasing a discrepancy that is really two different Flash kernels.

### What tolerance is legitimate

`torch.allclose` defaults to `rtol=1e-5, atol=1e-8`. Those are fp64 tolerances. Applying them to a bf16 forward pass is not strictness, it is a category error, and you will conclude your correct implementation is broken.

Derive the right number instead. bf16 has 8 bits of significand precision, so its machine epsilon is $2^{-7} \approx 7.8 \times 10^{-3}$ and round-to-nearest gives a maximum relative representation error of $2^{-8} \approx 3.9 \times 10^{-3}$. Tensor-core matmuls round their *inputs* to bf16 and accumulate in fp32, so a single GEMM contributes roughly $10^{-3}$ relative error. A Llama-3.1-8B forward pass chains seven GEMMs per layer across 32 layers plus the output projection, and any difference in reduction order between two implementations — a different tile size, a different split-K, a different loop order — perturbs each one independently.

The consequence: **do not check absolute logit values.** Check these instead, in this order:

| Check | What passes | What it catches | Source |
| --- | --- | --- | --- |
| Relative Frobenius error on logits | below about 1e-2 | any structural bug | derived from bf16 epsilon |
| Cosine similarity on logits | above 0.9999 | wrong direction in logit space | derived |
| Top-1 argmax agreement, all positions | 100% | anything that changes greedy output | reproduce: `verify.py` |
| Top-5 set overlap per position | 5 of 5, or 4 of 5 on ties | near-degenerate distributions | reproduce: `verify.py` |
| Top-5 *order* agreement | often imperfect | nothing — do not gate on this | derived |
| Greedy generation, 128 tokens | identical token IDs | drift that compounds over steps | reproduce: `verify.py` |

Logit magnitudes for these models sit in the range of roughly 10 to 30 in absolute value, so a relative error of $10^{-3}$ corresponds to absolute differences of order $10^{-2}$. If your `max_abs` is a few hundredths and your cosine similarity is above 0.9999, you have a correct implementation with different rounding. If `max_abs` is above 1 or cosine drops below 0.999, you have a bug, and no amount of tolerance tuning will make that not true.

Top-5 *order* is the trap. When two logits differ by less than the noise floor, their order is arbitrary in both implementations, and demanding stable order means demanding bit-exactness you cannot have. Gate on the argmax and the top-5 *set*; report the order as information.

```python
def agreement(mine, ref, k=5):
    m, r = mine.float(), ref.float()
    top1 = (m.argmax(-1) == r.argmax(-1)).float().mean().item()
    mk = m.topk(k, dim=-1).indices
    rk = r.topk(k, dim=-1).indices
    overlap = torch.stack([
        torch.isin(mk[..., i, :], rk[..., i, :]).float().mean()
        for i in range(m.shape[-2])
    ]).mean().item()
    print(f"top-1 agreement {top1:.4f}   top-{k} set overlap {overlap:.4f}")
```

A correct bf16 implementation should give `top-1 agreement 1.0000` on a 60-token prompt and a top-5 set overlap at or very near 1.0. If top-1 agreement is 0.98 — one position out of fifty disagreeing — do not shrug. Find the position, look at the logit gap there, and confirm the gap is smaller than your measured noise floor. If the gap is large, you have a real bug that happens to fire rarely.

### Bisecting layer by layer

Aggregate logits tell you *that* you diverged. Hooks tell you *where*.

```python
captured = {}


def make_hook(name):
    def hook(module, args, output):
        t = output[0] if isinstance(output, tuple) else output
        captured[name] = t.detach().float().clone()
    return hook


handles = [ref_model.model.embed_tokens.register_forward_hook(make_hook("embed"))]
for i, layer in enumerate(ref_model.model.layers):
    handles.append(layer.register_forward_hook(make_hook(f"layer{i:02d}")))
handles.append(ref_model.model.norm.register_forward_hook(make_hook("final_norm")))

with torch.inference_mode():
    ref_model(ids)

# now run ours, capturing the same points
mine_states = {}
h = mine.w["model.embed_tokens.weight"][ids]
mine_states["embed"] = h.float().clone()
for i in range(cfg.num_hidden_layers):
    h = mine.layer(h, i, mine.cos, mine.sin, positions, None)
    mine_states[f"layer{i:02d}"] = h.float().clone()
h = rms_norm(h, mine.w["model.norm.weight"], cfg.rms_norm_eps)
mine_states["final_norm"] = h.float().clone()

for name in ["embed"] + [f"layer{i:02d}" for i in range(cfg.num_hidden_layers)] + ["final_norm"]:
    compare(mine_states[name], captured[name], name)

for handle in handles:
    handle.remove()
```

Read the resulting table from the top. The shape of the output is the diagnosis:

- **`embed` already differs.** You have a weight-loading bug, a dtype bug, or a tokenizer mismatch. The forward pass is innocent. Check that the two models tokenized the same string into the same IDs — a missing BOS token is the classic cause and it makes every downstream tensor differ.
- **`embed` matches, `layer00` is off by a lot.** The bug is in a single layer's math, and you now get to bisect *inside* one layer. Split the layer into its five checkpoints — post-norm, post-QKV, post-RoPE, post-attention, post-MLP — and compare each. Fifteen minutes, at most.
- **Everything matches, then error grows smoothly from about 1e-3 at layer 0 to about 1e-2 at layer 31.** That is not a bug. That is bf16 rounding compounding through 32 layers, exactly as the epsilon derivation predicts. Growth that is roughly linear or mildly super-linear in depth is the healthy signature.
- **Error jumps by an order of magnitude at one specific layer and stays there.** Something is different about *that* layer. In a homogeneous stack like Llama-3 that usually means an off-by-one in your weight-key formatting — you loaded `model.layers.7` weights into slot 8.
- **Everything matches through `layer31`, `final_norm` diverges.** You forgot `model.norm`, or applied it with the wrong epsilon. This is common enough to be worth its own line in the table.

![A five row lookup mapping each divergence symptom to a likely cause a cheap confirming check and a fix](/imgs/blogs/a-forward-pass-by-hand-llama-from-scratch-6.webp)

### The classic culprits, with their fingerprints

Each of the standard bugs has a distinct signature. Learn the signatures and you will diagnose most of these before opening the code.

**Wrong RoPE layout (interleaved on HF weights).** Logits diverge substantially — relative error in the 0.1 to 1 range — but generation stays fluent. Position 0 matches almost exactly, because rotating by zero is the identity under both conventions, and error grows with position. That last part is the tell: run the comparison on a 60-token prompt and plot `rel_fro` per position. Rising with index means positional encoding; flat means something else.

**Forgetting `repeat_kv`.** A shape error, immediately, because `q` has 32 heads and `k` has 8. Loud and free. This one never survives to production.

**Repeating KV heads with the wrong interleaving.** No shape error. Query head 3 reads KV head 3 instead of KV head 0. All 32 query heads still get *a* key head, the math runs, the output is confidently wrong. Its fingerprint: divergence appears at `layer00` and stays roughly constant across depth rather than growing. Test it directly — build a tensor of shape `[1, 8, 1, 1]` with values `0..7`, run it through `repeat_kv(x, 4)`, and assert the output reads `[0,0,0,0,1,1,1,1,...]` and not `[0,1,...,7,0,1,...,7]`. Three lines, permanent insurance.

**A transposed projection.** `x @ W` instead of `F.linear(x, W)`. On the non-square projections this throws. On `q_proj` and `o_proj`, both $4096 \times 4096$, it silently computes with the transpose. Fingerprint: massive divergence at `layer00`, output text that is grammatical but semantically unrelated to the prompt.

**Causal mask off by one.** `diagonal=0` gives an all-`-inf` first row, NaN after softmax, and NaN everywhere downstream. Trivially diagnosed by `torch.isnan(logits).any()`, which belongs in your harness anyway.

**Mask orientation flipped.** Masking the lower triangle instead of the upper. No NaN, no shape error. Every position attends only to the future. Generation is fluent and the model appears to have amnesia about its own prompt. Fingerprint: the divergence is *largest at early positions* and smallest at the last one, the mirror image of the RoPE signature.

**Missing final norm.** Relative error of roughly 0.2 to 0.5 on the logits, all positions equally affected, and the argmax is right maybe 60% of the time. Confirmed instantly by the layer-by-layer table: everything matches until the last row.

**Tied vs untied `lm_head`.** If you use `embed_tokens` as the head on a model with `tie_word_embeddings: false`, you have substituted a completely different matrix. The logits are unrelated to the reference. The check is one line, and reading it from the config instead of guessing is one line too.

#### Worked example: reading a real bisect table

Suppose your harness prints this shape of output:

```console
embed                        max=0.000e+00  rel=0.000e+00  cos=1.0000000
layer00                      max=3.1e-03    rel=8.4e-04    cos=0.9999996
layer01                      max=4.4e-03    rel=1.1e-03    cos=0.9999994
...
layer15                      max=1.6e-02    rel=3.9e-03    cos=0.9999923
...
layer31                      max=4.2e-02    rel=9.1e-03    cos=0.9999585
final_norm                   max=5.0e-02    rel=9.6e-03    cos=0.9999539
logits                       max=6.8e-02    rel=1.0e-02    cos=0.9999502
top-1 agreement 1.0000   top-5 set overlap 1.0000
```

Read it as: embeddings identical (bit-exact gather, as expected); error appearing at layer 0 at the $10^{-3}$ relative level, which is one GEMM's worth of bf16 rounding; growth to $10^{-2}$ over 32 layers, roughly consistent with accumulating independent perturbations; cosine similarity never dropping below 0.9999; and complete argmax agreement. **This is a passing run.** *(Source: the magnitudes are derived from the bf16 epsilon argument above; the exact digits are illustrative of the pattern to look for, not a measurement — run `verify.py` on your own hardware and compare the shape of the curve, not the digits.)*

The thing to gate your CI on is not the numbers, it is the *shape*: monotone smooth growth with depth, no step change at any single layer, cosine above 0.9999 at the logits, and 100% argmax agreement across a long prompt. Any run that breaks that shape gets investigated.

### Measuring the forward pass honestly

You will want a timing number even though this post is about correctness. Get it right, because a bad measurement here poisons every comparison later in the series.

```python
import torch

def timed_forward(fn, ids, warmup=5, iters=20):
    for _ in range(warmup):          # let cuBLAS pick algorithms, let clocks settle
        fn(ids)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(ids)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters   # milliseconds per forward
```

The non-negotiables: warm up before timing, because the first call pays cuBLAS autotuning and kernel-module loading; use CUDA events rather than `time.perf_counter`, because kernel launches are asynchronous and a host-side timer measures how fast you enqueued work, not how fast it ran; call `torch.cuda.synchronize()` before reading any host-side timer; and report a distribution rather than a single number, because clock throttling makes the tenth iteration slower than the first on a consumer card under sustained load.

Also: **tok/s at batch 1 tells you almost nothing about a server.** It is a latency measure of the memory-bound decode path. A server's throughput depends on batching, and the relationship between the two is the subject of the entire scheduler track. Report both, never conflate them.

## 10. What this deliberately does not do

![A branching tree separating what this post ships from the cache batching and kernel work deferred to later posts](/imgs/blogs/a-forward-pass-by-hand-llama-from-scratch-7.webp)

Four large omissions, all intentional.

**No KV cache.** Generating token $t+1$ re-runs the entire forward pass over all $t$ previous tokens. That makes generating $n$ tokens from a prompt of length $S$ cost $O(n \cdot (S+n))$ token-forwards instead of $O(S + n)$. For a 2048-token prompt and 256 generated tokens, that is roughly $256 \times 2176 = 557{,}056$ token-forwards instead of ${2304}$ — a factor of 240. The cache is post 6 of this series, and the reason it comes *after* this post is that a cache is a place for an off-by-one to hide, and you want a verified uncached path to diff against when it does.

**No batching.** `forward` accepts a batch dimension and will happily process `[4, S]`, but every sequence must be the same length, which in practice means padding to the longest and wasting the difference. Track C covers static batching's padding tax and then the continuous-batching loop that replaces it.

**No paged memory.** Because there is no cache, there is nothing to page. Once there is, contiguous allocation fragments badly, and the block allocator that fixes it is post 8.

**No fused kernels.** Every operation here is a separate PyTorch call, which means separate kernel launches and separate round trips to HBM for intermediate tensors. RMSNorm alone reads and writes the full activation twice. Track E replaces the hot ones with hand-written CUDA and Triton — and the correctness test for every one of those kernels is a diff against the code in this post.

There is one more thing this post does not do: handle a `head_dim` that differs from `hidden_size / num_attention_heads`, sliding-window attention layers, QK-norm, attention sinks, or MoE routing. Several current models use one or more of those; the [modern LLM architectures comparison](/blog/machine-learning/large-language-model/modern-llm-architectures-qwen-llama-gemma-deepseek) walks through which family uses what. Adding each is a small, local change to `layer()` — which is itself an argument for keeping the reference implementation this plain.

## Case studies and real numbers

**The Hugging Face weight permutation.** The `transformers` conversion script for Llama models permutes `q_proj` and `k_proj` before saving, specifically so that `rotate_half` reproduces the interleaved rotation Meta trained with. This is documented in the conversion code rather than in any paper, which is why so many from-scratch implementations get it wrong. The lesson generalizes: the checkpoint format and the modeling code are a matched pair, and a weight file does not carry a note explaining which conventions it assumes.

**Grouped-query attention.** [Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (2023)](https://arxiv.org/abs/2305.13245) introduced GQA as an uptraining recipe: take an existing MHA checkpoint, mean-pool the key and value heads into groups, and fine-tune briefly. The paper's argument is that quality sits close to MHA while inference cost sits close to MQA. Our derivation shows the mechanism concretely — Llama-3.1-8B's 128 KiB per token versus 512 KiB under MHA, a 4× reduction in the memory that limits concurrency.

**RoPE and the base frequency.** [Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)](https://arxiv.org/abs/2104.09864) is the original. The move from $\theta_{\text{base}} = 10000$ in Llama-2 to 500000 in Llama-3, and the `llama3` rescaling function shipped in Llama-3.1's config, are both documented in the respective model cards. What is worth internalizing is that these are the *only* architectural knobs that changed between those releases for the 8B model — the layer count, width, head counts and MLP ratio are identical. Long context came from positional encoding and training, not from a new block.

**SwiGLU.** [Shazeer, "GLU Variants Improve Transformer" (2020)](https://arxiv.org/abs/2002.05202) is two pages long, tries eight gated variants, reports that several beat ReLU and GELU on the same parameter budget, and closes with the memorable line that the author attributes the success to divine benevolence. The 8/3 scaling convention comes from this paper's parameter-matching argument, and Llama-3's departure from it is a reminder to read the config rather than the folklore.

## When to reach for this, and when not to

Write the forward pass by hand when:

- You are building an inference engine and need a reference that you fully control and can instrument at any point.
- You are debugging a discrepancy between two runtimes — a vLLM output that disagrees with `transformers`, a quantized model that disagrees with its bf16 parent — and need a third implementation you trust as a tiebreaker.
- You are about to write kernels. Every custom kernel needs a numerically explicit reference to diff against, and PyTorch's fused ops are not that reference because they hide the intermediates you need.
- You are adding a new architecture and want to understand the block before you optimize it.

Do **not** write it by hand when:

- You want to run a model. Use `transformers` for research, vLLM or SGLang for serving. A hand-written forward pass is 100× slower than a real engine and buys you nothing operationally.
- The architecture is well supported and you have no plan to modify it.
- You are under deadline. This is a two-day exercise done properly, most of it in verification, and the verification is the part you cannot skip.

The trap to avoid is a hand-written forward pass that was *never verified against a reference*. It is worse than no implementation, because it produces plausible output and you will trust it. If you write the model, write `verify.py` the same day, and make it a test.

## Key takeaways

1. **`config.json` fully determines the model.** Eleven fields give you every tensor shape, the parameter count, the bf16 footprint, the per-token FLOPs, and the KV bytes per token. Learn to do that arithmetic in your head before downloading anything.
2. **Check your parameter formula against the model card.** Llama-3.1-8B's shapes sum to 8,030,261,248, which is the published 8.03B. If yours disagrees, you have misread a shape.
3. **The MLP is 70% of the weights.** Attention projections are 17%, embeddings 13%. Optimize where the bytes are.
4. **Forward FLOPs are about ${2N}$ per token**, plus an attention term that scales as $S^2$ and is negligible below a few thousand tokens and dominant above 32k.
5. **Decode is memory-bound by roughly two orders of magnitude.** A batch-1 step on a 4090 needs 16.06 GB of weight reads and 16.06 GFLOP of math; the reads take 160× longer. Everything downstream in this series exists to close that gap.
6. **RoPE layout is the bug you will hit.** Hugging Face checkpoints are permuted for half-split pairing. Interleaved pairing on those weights is silently wrong, and the fingerprint is error that grows with position.
7. **Repeat KV heads with `expand` inside the head axis**, never `repeat`. Test it with a three-line assertion on a toy tensor.
8. **Normalize and softmax in fp32, cast back afterwards.** Match the reference's cast ordering exactly, including multiplying by the RMSNorm weight after the cast back.
9. **`torch.allclose` defaults are fp64 tolerances.** For a bf16 forward pass, gate on relative Frobenius error below 1e-2, cosine similarity above 0.9999, and 100% top-1 agreement on a long prompt — never on top-k order.
10. **Hooks turn "it's wrong" into "layer 7 is wrong" in one run.** Capture every block's output from both implementations and read the divergence curve; its shape names the bug before you open the code.

## Further reading

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — Su et al., the RoPE derivation in full.
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — Ainslie et al., the paper behind `num_key_value_heads`.
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — Zhang and Sennrich, why the mean subtraction went away.
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — Shazeer, two pages, the source of SwiGLU and of the 8/3 convention.
- [PyTorch `scaled_dot_product_attention` documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) — backend selection rules, `is_causal` semantics, and `enable_gqa`.
- [Llama-3.1-8B model card](https://huggingface.co/meta-llama/Llama-3.1-8B) — the config fields and the published parameter count used throughout this post.
- [What inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is) — the layer map this series builds through.
- [The inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) — the capstone, which treats this harness as the correctness gate for every optimization that follows.
- [The KV cache, explained](/blog/machine-learning/large-language-model/kv-cache) — background for the cache we deliberately omitted here.
- [The roofline model: compute-bound vs memory-bound](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) — the framework behind the 160× decode gap.

---
title: "Activation outliers, calibration, and measuring quality loss: how quantization breaks and how to know"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Why a few activation channels break INT8, how per-channel scales, SmoothQuant, AWQ, and learned rounding fix it, and — the part everyone skips — how to measure the damage honestly with an eval harness that runs perplexity AND a task the way a shipping engineer must."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "quantization",
    "calibration",
    "evaluation",
    "perplexity",
    "smoothquant",
    "awq",
    "pytorch",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 42
---

You quantized Llama-3.1-8B to INT4, the memory dropped from 15 GiB to 4 GiB, tokens-per-second nearly doubled, and the perplexity moved from 9.43 to 9.51 — a change so small you rounded it away in the release notes. You shipped it. Three weeks later a customer reports that the model can no longer add two three-digit numbers reliably, and your support queue fills with math errors that the fp16 model never made. The perplexity was fine. The perplexity is *always* fine. That is the trap, and it is the reason this post exists.

The previous three posts made quantization *work mechanically*: [weight-only INT4 at load time](/blog/machine-learning/inference-engineering/weight-only-quantization-in-your-engine-gguf-awq-gptq-at-load-time), [FP8 and FP4 on the hardware](/blog/machine-learning/inference-engineering/fp8-and-fp4-inference-what-the-hardware-actually-gives-you), and [quantizing the KV cache](/blog/machine-learning/inference-engineering/kv-cache-quantization-fp8-int8-and-the-accuracy-cliff). Each one told you how to make the bytes smaller. None of them told you the two things that decide whether you can actually ship: *why quantization sometimes destroys a model that looks fine*, and *how you would ever know it did*. This post is those two halves. The first half is the outlier mechanism — the reason a naive INT8 pass can turn a good model into a bad one on some architectures and not others. The second half is the measurement discipline — why perplexity lies to you, which evals catch what it misses, and how to build a gate that would have stopped you from shipping the broken INT4 model.

![Side-by-side comparison of one quantization scale stretched to cover an outlier channel and crushing the normal channels against a per-channel scheme that gives each channel its own scale](/imgs/blogs/activation-outliers-calibration-and-measuring-quality-loss-1.webp)

By the end you will have written `nanoserve/quant/observe.py` (an outlier-detection pass that plots the per-channel activation magnitudes), `nanoserve/quant/compare.py` (a per-channel-versus-per-tensor quantization-error report), and `nanoserve/eval/harness.py` (an eval harness that runs perplexity *and* a small arithmetic task on the quantized model versus the fp16 model and prints both, so you can catch the perplexity-looks-fine-but-the-task-broke failure yourself). The standing promise from [the series introduction](/blog/machine-learning/inference-engineering/what-inference-engineering-is) holds here too: **I have no GPU and I have run none of this.** Every number below is derived from arithmetic I show you, cited from a paper or an official post with a link, or framed as something you reproduce yourself with a named script and an expected range. Results tables carry a `Source` column. The measurement half is exactly where that discipline matters most — a beautiful quantization with an invented accuracy number is worse than no quantization at all.

---

## 1. A few channels do all the damage

Start with what an activation actually is, because the whole problem lives in its shape. Inside a transformer, between the layers, there is a running vector called the **residual stream** — for Llama-3.1-8B it is 4,096 numbers wide, one vector per token, carried in bf16. Every attention block and every MLP reads this vector, transforms it, and writes back. When we say "quantize the activations" we mean: take that stream of 4,096-wide vectors, and store or compute them in INT8 or FP8 instead of bf16. Weight quantization shrinks the *model*; activation quantization is what unlocks integer matmuls, where both operands are low-precision and the tensor cores run at their fast INT8/FP8 rates. The catch is entirely in the distribution of those 4,096 numbers.

If you histogram the values across the channels (the 4,096 positions) of a real transformer's residual stream, you do not get a nice bell curve. You get a distribution where the overwhelming majority of channels sit in a tight band — say magnitudes under 1 or 2 — and a *tiny handful* of channels, often fewer than ten, carry values 10 to 100 times larger. These are the **outlier features**, and they are not noise. They are structural: the model learned to route certain always-on signals through a few dedicated channels, and it does this consistently across tokens and across inputs.

![Layered anatomy of the residual stream showing the bulk of channels in a narrow band and a small set of outlier channels reaching far higher magnitudes](/imgs/blogs/activation-outliers-calibration-and-measuring-quality-loss-2.webp)

This is a documented, measured phenomenon, not folklore. The LLM.int8() paper (Dettmers et al., 2022, [arXiv:2208.07339](https://arxiv.org/abs/2208.07339)) reports that these outlier features **emerge abruptly** as models grow: below roughly 6.7B parameters they are manageable, and past that scale they appear in essentially every layer, concentrated in a small number of feature dimensions, with magnitudes up to about 20× the typical activation and — critically — they are the dimensions that matter most for the model's predictions. The paper's whole method (a mixed-precision decomposition that keeps the outlier dimensions in fp16 and quantizes the rest to INT8) is a direct response to the fact that you cannot quantize the outlier channels the same way you quantize the normal ones. Whether you hit this depends on the model: some architectures and training recipes produce far sharper outliers than others, which is exactly why a quantization scheme that is flawless on one model can wreck another. The FP8-KV work from the vLLM team makes the same point from the deployment side — their [FP8 KV-cache post](https://vllm.ai/blog/2026-04-22-fp8-kvcache) (2026-04-22) notes that uncalibrated models like Kimi-K2.5 show a persistent downward accuracy shift that only calibration fixes, while other models recover to 97%+ with the default per-tensor scale. Same technique, different model, opposite outcome. The difference is the outliers.

Why would a model *learn* to do this to itself? The outliers are not an accident; they are the model using a few channels as high-gain, always-on signal lines. Two mechanisms are well understood. The first is the **attention sink**: transformers learn to dump attention probability onto a small number of positions (often the first token) as a no-op when a head has nothing useful to attend to, and the residual channels that implement that behavior carry large, persistent magnitudes. The second is **normalization gain**: RMSNorm and LayerNorm each have a learned per-channel scale, and a handful of those learned gains grow large during training, inflating specific channels of the normalized output. Both produce the same signature — a few dimensions, stable across tokens and inputs, sitting far above the bulk. This matters for quantization because "stable across inputs" is what makes calibration *possible* (the outlier channels are the same ones you can measure and plan for), while "a few dimensions carrying disproportionate signal" is what makes crushing them *catastrophic* (you are destroying precision exactly where the model concentrated its information). The outlier is both the reason quantization is hard and the reason it is tractable at all.

Before we fix anything, measure it — this is the first thing `nanoserve` learns to do. You attach a forward hook to a linear layer, capture the input activations across a batch of real tokens, and reduce to the per-channel maximum absolute value. If a few channels tower over the rest, you have outliers.

```python
# nanoserve/quant/observe.py
import torch

class ActivationObserver:
    """Capture per-channel activation magnitude statistics via a forward hook.

    Attach to the linear layers whose *inputs* you plan to quantize
    (q/k/v/o projections, gate/up/down). We track the running per-channel
    max-abs and the mean-abs so we can see the outlier ratio.
    """
    def __init__(self, num_channels: int, device="cuda"):
        self.max_abs = torch.zeros(num_channels, device=device)
        self.sum_abs = torch.zeros(num_channels, device=device)
        self.count = 0

    def __call__(self, module, inputs, output):
        x = inputs[0]                      # [tokens, channels]
        x = x.reshape(-1, x.shape[-1]).float()
        self.max_abs = torch.maximum(self.max_abs, x.abs().amax(dim=0))
        self.sum_abs += x.abs().sum(dim=0)
        self.count += x.shape[0]

    def outlier_report(self, top_k=8):
        mean_abs = self.sum_abs / max(self.count, 1)
        typical = mean_abs.median()                 # robust "normal" scale
        ratio = self.max_abs / (typical + 1e-9)     # per-channel outlier ratio
        vals, idx = ratio.topk(top_k)
        return {"typical_mag": typical.item(),
                "top_channels": idx.tolist(),
                "top_ratios": vals.tolist(),
                "global_max_ratio": ratio.max().item()}
```

Wire it in with a hook on the layer you care about and feed it a few hundred tokens:

```python
# scripts/find_outliers.py
from nanoserve.quant.observe import ActivationObserver

layer = model.model.layers[16].mlp.down_proj      # a mid-stack MLP input
obs = ActivationObserver(num_channels=layer.in_features, device="cuda")
h = layer.register_forward_hook(obs)

with torch.inference_mode():
    for batch in calib_loader:                     # ~256 sequences is plenty
        model(batch.to("cuda"))
h.remove()

print(obs.outlier_report())
```

#### Worked example: reading an outlier report

Run the script above on a mid-stack MLP of an 8B model and you should see something like `global_max_ratio` in the tens — a single channel whose peak magnitude is 20–70× the median channel's typical magnitude — with the offending channel indices staying *stable* if you re-run on a different batch. That stability is the tell: a genuine outlier feature is the same handful of dimensions every time, not a different random channel per batch. If instead you see the ratios hovering near 1–3 and no dimension standing out, this particular model quantizes easily and you can skip most of the machinery in this post for it. *(Source: reproduce with `find_outliers.py`; expected range grounded in the LLM.int8() paper's report of up to ~20× outliers past 6.7B parameters, [arXiv:2208.07339](https://arxiv.org/abs/2208.07339).)* The number you compute is your own; the order of magnitude is what the literature predicts.

The reason this matters so much is not that the outliers themselves are hard to store — INT8 can represent a big number fine. The reason is what a *single shared scale* has to do to accommodate them, which is the derivation in the next section.

---

## 2. Why a single scale wastes almost all of the range

Quantization to INT8 is, at heart, picking a **scale** $s$ and mapping a real value $v$ to the nearest integer $q = \operatorname{round}(v / s)$, clamped to the representable range. Symmetric INT8 gives you integers in $[-127, 127]$, so 255 usable levels. To not clip the largest value in whatever tensor you are quantizing, you must set the scale from the maximum:

$$
s = \frac{\max_i |v_i|}{127}
$$

That is the whole mechanism for a **per-tensor** scale: one number $s$ for the entire activation tensor, chosen so the biggest element just fits. Now watch what it does to a normal channel when an outlier is present. Suppose the outlier channel peaks at magnitude $A$, and a typical channel's values sit around $A/100$ — a 100× ratio, well within what the LLM.int8() paper reports. The per-tensor scale is forced to $s = A / 127$ by the outlier. A typical value $A/100$ therefore maps to:

$$
q = \operatorname{round}\!\left(\frac{A/100}{A/127}\right) = \operatorname{round}(1.27) = 1
$$

The entire typical channel — every value in it — lands on the integers $\{-1, 0, 1\}$. **Three levels.** Out of 255. That channel is now carrying about $\log_2 3 \approx 1.58$ bits of information instead of 8. You spent an 8-bit budget and delivered 1.58 bits to the channels that make up 99.8% of the tensor, so that the 0.2% of channels that are outliers could be represented without clipping. This is the core reason naive per-tensor INT8 fails on outlier-heavy models: the outlier does not get quantized badly — *everything else does*, because the outlier sets the scale.

#### Worked example: how many bits does the outlier steal?

Make the loss explicit as a function of the outlier ratio $r = \max|v| / \text{typical}$. A typical value maps to about $127 / r$ integer levels, so the effective bit depth of the normal channels is roughly $\log_2(2 \cdot 127 / r)$ — the range $[-127/r, 127/r]$ has about $2 \cdot 127 / r$ integers. At $r = 4$ (a mild outlier) the normal channels get $\log_2(63) \approx 6.0$ bits — barely a problem. At $r = 16$ (LLM.int8()'s reported magnitude) they get $\log_2(15.9) \approx 4.0$ bits — you paid for INT8 and got INT4 on 99.8% of the tensor. At $r = 100$ they get $\log_2(2.5) \approx 1.3$ bits — worse than binary. The loss is logarithmic in the ratio, which is why a model with $r$ in the single digits quantizes fine per-tensor and a model with $r$ in the dozens does not: the same scheme, the same code, and the outlier ratio alone decides whether you shipped 6 effective bits or 1.3. *(Source: derived; ratios grounded in [arXiv:2208.07339](https://arxiv.org/abs/2208.07339).)*

An **asymmetric** scheme (a per-tensor scale *plus* a zero-point offset, so the range need not be centered at zero) buys you nothing here, because the problem is not that the range is off-center — it is that the range is *too wide*. Asymmetric quantization helps when a channel's values are one-sided (all positive, like a post-ReLU activation); it does not help when a single outlier stretches the range that a shared scale must span. The only fix is to stop *sharing* the scale across the outlier and the normals, which is what per-channel/per-token scaling does — with the reduction-dimension caveat we hit next.

The animated figure below shows the same tensor under both regimes. Under one shared scale the quantization grid is coarse, set by the far-off spike, and the normal bars fall between just a couple of grid lines. Give each channel its own scale and the grid around the normal bars becomes fine again — the outlier is off in its own scale and stops dictating everyone else's precision.

<figure class="blog-anim">
<svg viewBox="0 0 760 320" role="img" aria-label="An activation histogram with one tall outlier bar and several short normal bars, shown first under a single coarse quantization grid that crushes the normal bars onto three levels and then under per-channel grids that resolve each normal bar finely" style="width:100%;height:auto;max-width:900px">
<title>One outlier channel forces a coarse per-tensor grid that flattens every normal channel onto three levels; per-channel scales restore fine resolution to the normal channels.</title>
<style>
.ao1-bar{fill:var(--accent,#6366f1)}
.ao1-out{fill:var(--text-primary,#1f2937)}
.ao1-grid{stroke:var(--border,#d1d5db);stroke-width:1}
.ao1-fine{stroke:var(--accent,#6366f1);stroke-width:1;opacity:.55}
.ao1-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.ao1-sub{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
@keyframes ao1-A{0%,42%{opacity:1}54%,96%{opacity:0}100%{opacity:1}}
@keyframes ao1-B{0%,42%{opacity:0}54%,96%{opacity:1}100%{opacity:0}}
.ao1-stateA{animation:ao1-A 10s ease-in-out infinite}
.ao1-stateB{animation:ao1-B 10s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.ao1-stateA{animation:none;opacity:1}.ao1-stateB{animation:none;opacity:0}}
</style>
<rect class="ao1-bar" x="70"  y="210" width="40" height="40"/>
<rect class="ao1-bar" x="130" y="222" width="40" height="28"/>
<rect class="ao1-bar" x="190" y="205" width="40" height="45"/>
<rect class="ao1-bar" x="250" y="215" width="40" height="35"/>
<rect class="ao1-out" x="330" y="60"  width="40" height="190"/>
<rect class="ao1-bar" x="410" y="218" width="40" height="32"/>
<rect class="ao1-bar" x="470" y="208" width="40" height="42"/>
<text class="ao1-sub" x="350" y="52">outlier · ~100x</text>
<g class="ao1-stateA">
<line class="ao1-grid" x1="60" y1="90"  x2="540" y2="90"/>
<line class="ao1-grid" x1="60" y1="170" x2="540" y2="170"/>
<line class="ao1-grid" x1="60" y1="250" x2="540" y2="250"/>
<text class="ao1-lbl" x="620" y="150">one scale</text>
<text class="ao1-sub" x="620" y="172">normals get</text>
<text class="ao1-sub" x="620" y="190">3 levels</text>
</g>
<g class="ao1-stateB">
<line class="ao1-fine" x1="60" y1="200" x2="300" y2="200"/>
<line class="ao1-fine" x1="60" y1="212" x2="300" y2="212"/>
<line class="ao1-fine" x1="60" y1="224" x2="300" y2="224"/>
<line class="ao1-fine" x1="60" y1="236" x2="300" y2="236"/>
<line class="ao1-fine" x1="60" y1="248" x2="300" y2="248"/>
<line class="ao1-fine" x1="400" y1="200" x2="540" y2="200"/>
<line class="ao1-fine" x1="400" y1="212" x2="540" y2="212"/>
<line class="ao1-fine" x1="400" y1="224" x2="540" y2="224"/>
<line class="ao1-fine" x1="400" y1="236" x2="540" y2="236"/>
<line class="ao1-fine" x1="400" y1="248" x2="540" y2="248"/>
<text class="ao1-lbl" x="620" y="150">per channel</text>
<text class="ao1-sub" x="620" y="172">normals get</text>
<text class="ao1-sub" x="620" y="190">full range</text>
</g>
</svg>
<figcaption>The same channels under a single shared scale (coarse grid, normals crushed to three levels) versus per-channel scales (fine grid, normals resolved). The outlier magnitude is illustrative at ~100x; the three-level result is the derivation in the text.</figcaption>
</figure>

The fix that section suggests — give each channel its own scale — is correct in spirit but has a subtlety that shapes every real scheme, and you have to see the matmul to understand it. A linear layer computes $Y = X W$, where $X$ is `[tokens, in]` and $W$ is `[in, out]`. To do this in integer arithmetic you quantize $X$ and $W$ to integers and factor the scales out of the dot product. That factoring only works if the scale is *constant along the reduction (contraction) dimension* — the `in` dimension that the dot product sums over. Concretely:

$$
Y[t, o] = s_x[t] \cdot s_w[o] \cdot \sum_{c} X_q[t, c]\, W_q[c, o]
$$

A **per-token** activation scale $s_x[t]$ (one per row) factors out cleanly, because $t$ is not summed over. A **per-output-channel** weight scale $s_w[o]$ (one per column) factors out cleanly, because $o$ is not summed over. But a **per-input-channel** activation scale $s_x[c]$ — which is exactly what you would need to give the outlier channel its own scale — does *not* factor out, because $c$ is the dimension the dot product sums over. You would have to un-scale each product before summing, which defeats the integer matmul. So the natural, cheap pairing is **per-token activations × per-channel weights** — and this is precisely PTPC-FP8 (Per-Token-activation, Per-Channel-weight FP8), which the vLLM team built [specifically to tackle activation outliers](https://vllm.ai/blog/2025-02-24-ptpc-fp8-rocm) (2025-02-24). Their fused rowwise-scaled FP8 GEMM runs up to 2.5× faster than a naive two-step, and — the number that matters for this post — on Llama-3.1-8B the Wikitext perplexity moves from a bf16 baseline of 9.4281 to 9.5093, a **0.86%** degradation. Hold that 0.86% in mind; we return to it when we ask whether perplexity is the right thing to be looking at.

Per-token/per-channel scaling helps, but it does not directly attack the outlier *channel*, because the outlier lives on the reduction dimension where per-token scaling cannot reach it. The next section is the three ideas that do reach it.

---

## 3. The four defenses, mechanism first

There are four techniques you will meet in the wild, and they are easy to confuse because they all "help with outliers." They are actually attacking the problem from different directions. Two rescale, one protects, one re-rounds. Here they are in the order that builds understanding.

### 3.1 Per-channel and per-token scales — give the difficulty its own budget

This is the direct fix from the previous section: instead of one scale for the whole tensor, use a scale per row (per-token, for activations) and a scale per column (per-channel, for weights). It is cheap, it is the default in good kernels, and it recovers most of the range that a per-tensor scale threw away. Its limit, again, is that it cannot give the *reduction*-dimension outlier channel its own scale — so on a model with truly sharp per-input-channel outliers, per-token/per-channel is necessary but not always sufficient. That is why the next two ideas exist.

### 3.2 SmoothQuant — migrate the difficulty from activations to weights

Here is the key insight, and it is genuinely elegant. Activations are hard to quantize (outliers on the reduction dimension). Weights are *easy* to quantize (well-behaved, and you can scale them per-output-channel, which factors out cleanly). So: **move the difficulty from the activations into the weights**, where you have a tool that handles it. SmoothQuant (Xiao et al., 2022, [arXiv:2211.10438](https://arxiv.org/abs/2211.10438)) does this with a per-input-channel diagonal rescale that leaves the product mathematically unchanged.

![Dataflow diagram showing an activation with a wide outlier range and a weight with a narrow range meeting a diagonal scaling that splits the difficulty between them before the two rebalanced tensors merge back into an identical product](/imgs/blogs/activation-outliers-calibration-and-measuring-quality-loss-3.webp)

Pick a per-input-channel vector $s$ (length `in`). Rescale the activation *down* and the weight *up* by the same diagonal:

$$
\hat{X} = X \operatorname{diag}(s)^{-1}, \qquad \hat{W} = \operatorname{diag}(s)\, W
$$

The product is untouched, because the diagonals cancel:

$$
\hat{X}\hat{W} = X \operatorname{diag}(s)^{-1} \operatorname{diag}(s)\, W = X W
$$

Now choose $s$ to *balance* the ranges. For input channel $j$, the activation there has magnitude $\max|X_j|$ and the weight row has magnitude $\max|W_j|$. Set

$$
s_j = \frac{\max|X_j|^{\alpha}}{\max|W_j|^{1-\alpha}}
$$

with a migration strength $\alpha$ (typically around 0.5). A large activation channel gets a large $s_j$, which divides that channel *down* in $\hat{X}$ and multiplies the corresponding weight row *up* in $\hat{W}$. The outlier magnitude is physically moved out of the activation, where you could not scale it away, and into the weight, where per-output-channel weight quantization eats it without trouble. And here is the part that makes it free at inference: $\operatorname{diag}(s)^{-1}$ can be **folded into the previous layer** — the LayerNorm/RMSNorm scale or the preceding linear's weights — so at serve time there is no extra op. You quantize once, offline; the migration is baked into the checkpoint. Nothing on the hot path changed.

The scale computation is short enough to write out, and doing so makes the migration-strength knob concrete. You collect the per-channel activation maxima during calibration (from the same observer as §1), take the per-channel weight maxima from the checkpoint, and combine them:

```python
# nanoserve/quant/smoothquant.py
import torch

def smoothquant_scales(act_amax, weight, alpha=0.5, eps=1e-5):
    """act_amax: [in] per-input-channel activation max-abs (from calibration).
       weight:   [in, out] the linear's weight. Returns s: [in]."""
    w_amax = weight.abs().amax(dim=1).clamp(min=eps)          # [in]
    a_amax = act_amax.clamp(min=eps)                          # [in]
    s = a_amax.pow(alpha) / w_amax.pow(1 - alpha)             # [in]
    return s.clamp(min=eps)

def apply_smoothquant(x, weight, prev_layer, s):
    # fold 1/s into the previous op's output, s into this weight
    prev_layer.weight.mul_(s.reciprocal().unsqueeze(-1))     # activations / s
    weight.mul_(s.unsqueeze(-1))                             # weights * s
    # at inference x already arrives divided by s; product is unchanged
    return weight
```

The `alpha` knob is the whole trade. At `alpha=0` you divide the activations by the weight ranges (all migration onto activations — pointless); at `alpha=1` you divide by the activation ranges (all difficulty onto the weights, which can overload weight quantization). Around 0.5 splits it, and the right value is model-specific — which is one more thing the eval harness at the end of this post exists to check, because there is no way to pick `alpha` correctly without measuring the downstream task.

### 3.3 AWQ — protect the salient channels, do not rescale everything

AWQ (Activation-aware Weight Quantization; Lin et al., 2023, [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)) makes a sharper observation for *weight-only* quantization: not all weights matter equally. A small fraction — roughly 1% — of weight channels are **salient**, and which ones are salient is determined by the *activation* magnitudes flowing into them (a weight channel that multiplies a large activation channel matters far more than one that multiplies a near-zero channel). Keeping just those salient channels in higher precision recovers most of the lost accuracy; but mixed-precision storage is awkward for kernels. So AWQ instead applies a per-channel scaling that *protects* the salient channels — scaling them up before quantizing so they land in a region of the grid with finer effective resolution, then folding the inverse scale into the activation path. The scaling factors are found by minimizing the layer's output error over a small calibration set. It is the same "activation statistics tell you what to protect" idea we used in [the weight-only quantization post](/blog/machine-learning/inference-engineering/weight-only-quantization-in-your-engine-gguf-awq-gptq-at-load-time), applied as a rescale rather than a decomposition — which is why AWQ checkpoints load into a normal kernel with no mixed-precision bookkeeping.

### 3.4 Learned rounding — stop rounding to the nearest level

The last idea attacks a different assumption entirely. Everything above still uses **round-to-nearest**: map each value to its closest grid point. Round-to-nearest minimizes the error of *each weight independently* — but that is not the error you care about. You care about the error of the layer's *output*, and rounding errors on different weights are correlated through the matmul, so a set of deliberately "wrong" roundings can *cancel* in the output and beat round-to-nearest. GPTQ (Frantar et al., 2022, [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)) exploits this with second-order information: it uses a Hessian estimated from a calibration set to compensate each weight's quantization error against the others, column by column. AutoRound (Intel, described in the vLLM team's [AutoRound × LLM Compressor post](https://vllm.ai/blog/2025-12-09-intel-autoround-llmc), 2025-12-09) makes the rounding itself *learnable*: it introduces three parameters per tensor — a rounding offset $V$ and a pair of clipping bounds $\alpha, \beta$ — and optimizes them by **signed gradient descent** to minimize the block's output reconstruction error. It typically uses about 128 calibration samples and around 200 iterations, and because it is an optimization with a stochastic path it is **non-deterministic** — run it twice and you get two slightly different checkpoints. On Qwen3-8B at W4A16 the vLLM post reports a GSM8K score of 0.911. Learned rounding is the most expensive of the four (you are running an optimizer over the model), but on aggressive bit-widths it is often the difference between a model that survives and one that does not.

![Comparison table of the four defenses across what each does, where it moves the difficulty, its calibration need, and a cited anchor](/imgs/blogs/activation-outliers-calibration-and-measuring-quality-loss-4.webp)

The four are not competitors so much as a stack you assemble. A production W4A16 checkpoint often uses SmoothQuant-style migration *and* AWQ-style protection *and* learned rounding, on top of per-channel weight scales. Here is the decision as a table.

| Defense | What it does | Where the difficulty goes | Calibration need | Source |
|---|---|---|---|---|
| Per-channel / per-token | One scale per row/column | Split off the token & output dims | None (compute at runtime or from a few batches) | derived; cited: PTPC-FP8, [vLLM 2025-02-24](https://vllm.ai/blog/2025-02-24-ptpc-fp8-rocm) |
| SmoothQuant | Diagonal rescale, folded into prev layer | Migrated activation → weight | Activation max per channel (a few hundred samples) | cited: [arXiv:2211.10438](https://arxiv.org/abs/2211.10438) |
| AWQ | Protect ~1% salient channels by rescale | Kept in effective higher precision | Activation stats + output-error search | cited: [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) |
| Learned rounding (GPTQ/AutoRound) | Optimize rounding for output error | Correlated errors cancel in the output | ~128 samples, ~200 iters, non-deterministic | cited: [vLLM 2025-12-09](https://vllm.ai/blog/2025-12-09-intel-autoround-llmc) |

Notice that three of the four need a **calibration set**. That is not a footnote — the calibration set is where most quiet quality disasters are actually born, and it is the next section.

---

## 4. Calibration: the set, the size, and the wrong-distribution trap

A **calibration set** is a small collection of representative inputs you run through the model, in inference mode, to *measure* the statistics that the quantization needs — the per-channel activation maxima for SmoothQuant, the salience ranking for AWQ, the Hessian for GPTQ, the reconstruction target for AutoRound. It is not training. No weights are updated by gradient descent on a loss; you are collecting activation statistics, or in the learned-rounding case, fitting a handful of rounding parameters. But it shares one property with training that makes it dangerous: **the statistics you collect are only valid for inputs that look like your calibration set.**

Two questions decide whether calibration helps or hurts: how *many* samples, and from *what distribution*.

**How many.** For scale estimation (SmoothQuant, per-channel maxima), you are estimating a maximum, and maxima converge reasonably fast once you have seen the outlier channels fire — a few hundred sequences of a few hundred tokens is the standard range, and AutoRound's ~128 samples sits right in that band. More samples rarely hurt and quickly hit diminishing returns for the *scale*. But sample count cannot save you from the second question.

**From what distribution.** This is the trap, and it is worth stating as a rule: *calibrate on the distribution you deploy on, or expect to pay for the mismatch on exactly the inputs you did not calibrate for.* If you calibrate on English Wikipedia and deploy on Python source, or on Vietnamese, the outlier channels that fire on the deployment distribution may differ from the ones your calibration saw — and your scales, tuned to the wrong channels, will clip the ones that actually matter at serve time.

![Decision tree for calibration data showing a match to the deployment distribution leading to good recovery and three mismatch branches leading to distinct failure modes](/imgs/blogs/activation-outliers-calibration-and-measuring-quality-loss-5.webp)

This connects directly to something we established in [the tokenizer post](/blog/machine-learning/inference-engineering/the-tokenizer-boundary-and-incremental-detokenization): different languages and code tokenize with very different efficiency, which means they exercise different regions of the model. A calibration set of English prose does not necessarily light up the channels that Vietnamese diacritics or Python indentation tokens light up. The vLLM FP8-KV post's observation about Kimi-K2.5 is the same failure from the model side — an uncalibrated per-tensor scale is implicitly "calibrated" on the assumption `scale = 1.0`, and for a model whose outliers violate that assumption, the result is a persistent downward accuracy shift that only a real calibration pass repairs.

#### Worked example: the input-dependent outlier a small calibration set misses

Here is the nastiest version, and it is why sample count and distribution interact. Suppose a particular outlier channel only fires strongly on a *specific* kind of input — say, a channel that spikes on long strings of digits, which appear in your traffic but are rare in generic web text. Your 128-sample calibration set, drawn from generic text, contains almost no long digit strings, so that channel looks *normal* during calibration and gets a tight scale. In production, a user pastes a spreadsheet, the channel fires at 40× its calibrated magnitude, the tight scale clips it hard, and the model's arithmetic falls apart — on exactly the inputs a user would most expect a computer to get right. Perplexity on your generic eval set never sees it, because your eval set also has no long digit strings. *(Source: derived mechanism; the outlier-emergence and input-dependence are grounded in [arXiv:2208.07339](https://arxiv.org/abs/2208.07339).)* This is not hypothetical hand-waving; it is the structural reason calibration coverage matters, and it is the bridge to the second half of the post. You cannot fix what you cannot see, and you cannot see it with the wrong measurement.

---

## 5. Perplexity's blind spot — the single most important measurement lesson

Everyone reaches for **perplexity** first, because it is cheap, it needs no labels, and it produces one number. Perplexity is the exponentiated average negative log-likelihood the model assigns to a held-out corpus:

$$
\text{ppl} = \exp\!\left(\frac{1}{N} \sum_{i=1}^{N} \text{NLL}_i\right), \qquad \text{NLL}_i = -\log p(x_i \mid x_{<i})
$$

Lower is better; it measures, on average, how surprised the model is by the next token. The word doing all the damage in that sentence is **average**. Perplexity averages the surprise over *every* token in the corpus, and a quantization error that corrupts a *few critical tokens* barely moves an average taken over tens of thousands of them. Let me make that precise, because the arithmetic is the lesson.

Suppose quantization increases the NLL by $\Delta$ nats on a fraction $f$ of tokens and leaves the rest unchanged. The mean NLL rises by $f\Delta$, so the perplexity ratio is:

$$
\frac{\text{ppl}_{\text{quant}}}{\text{ppl}_{\text{fp16}}} = \exp(f \Delta)
$$

Plug in a real scenario. Say quantization badly corrupts $f = 0.002$ of tokens (one in five hundred) by a large $\Delta = 2$ nats each. The perplexity ratio is $\exp(0.002 \times 2) = \exp(0.004) \approx 1.004$ — a **0.4% perplexity increase**. You would round that away. Recall PTPC-FP8's measured 0.86% Wikitext degradation — right in this neighborhood, and reported as a success, which it may well be. The point is that a 0.4–0.9% perplexity move is *consistent with the model being perfectly fine and also consistent with the model being broken on a class of inputs you care about.* Perplexity cannot distinguish the two, and that is its blind spot.

Now the other side. Consider a task where the answer is only correct if *every* token in a chain is correct — arithmetic is the cleanest example. Model a GSM8K solution as a chain of $k$ **pivotal** tokens (the digits and operators where a single mistake fails the whole problem). If quantization raises the per-pivotal-token error rate from $p_0$ to $p_1$, the per-problem success probability goes from $(1 - p_0)^k$ to $(1 - p_1)^k$. With $p_0 = 0.01$, $p_1 = 0.04$, and $k = 5$:

$$
(1 - 0.01)^5 = 0.951 \quad \longrightarrow \quad (1 - 0.04)^5 = 0.815
$$

A **13.6-point** drop in task accuracy — from the *same* quantization whose perplexity moved 0.4%. The exponent $k$ is the amplifier: perplexity sees the *arithmetic mean* of per-token surprise, but a chained task sees the *product* of per-token success, and a product punishes a raised error rate far more brutally than a mean does. The animated figure makes the disparity visible — the same "quantize" event, one needle that barely twitches and one bar that collapses.

<figure class="blog-anim">
<svg viewBox="0 0 720 300" role="img" aria-label="Two meters responding to the same quantization event: a perplexity meter that grows only slightly while a task-accuracy bar shrinks dramatically" style="width:100%;height:auto;max-width:860px">
<title>The same quantization moves the perplexity meter by a hair while the task-accuracy bar collapses, because a chained task multiplies per-token success while perplexity averages it.</title>
<style>
.ao2-track{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.ao2-ppl{fill:var(--text-secondary,#6b7280);transform-box:fill-box;transform-origin:bottom}
.ao2-task{fill:var(--accent,#6366f1);transform-box:fill-box;transform-origin:bottom}
.ao2-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.ao2-sub{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
@keyframes ao2-ppl{0%,20%{transform:scaleY(1)}55%,80%{transform:scaleY(1.05)}100%{transform:scaleY(1)}}
@keyframes ao2-task{0%,20%{transform:scaleY(1)}55%,80%{transform:scaleY(0.63)}100%{transform:scaleY(1)}}
.ao2-pplA{animation:ao2-ppl 9s ease-in-out infinite}
.ao2-taskA{animation:ao2-task 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.ao2-pplA,.ao2-taskA{animation:none}}
</style>
<rect class="ao2-track" x="160" y="60" width="90" height="190" rx="6"/>
<rect class="ao2-ppl ao2-pplA" x="168" y="180" width="74" height="70"/>
<text class="ao2-lbl" x="205" y="275">perplexity</text>
<text class="ao2-sub" x="205" y="293">9.43 to 9.51 · +0.4%</text>
<rect class="ao2-track" x="470" y="60" width="90" height="190" rx="6"/>
<rect class="ao2-task ao2-taskA" x="478" y="72" width="74" height="178"/>
<text class="ao2-lbl" x="515" y="275">task accuracy</text>
<text class="ao2-sub" x="515" y="293">73% to 61% · minus 14 pts</text>
<text class="ao2-sub" x="360" y="40">same quantization event</text>
</svg>
<figcaption>The same INT4 quantization: the perplexity meter barely rises while the task-accuracy bar collapses. Numbers are illustrative, derived from the chained-token model in the text (0.4% perplexity, ~14-point task drop).</figcaption>
</figure>

Which tokens are the pivotal ones? The ones a single-token flip destroys: a **digit** in an arithmetic answer, a **negation** ("not") that inverts a claim, a **tool name** or a JSON key that a downstream parser must match exactly, a variable name in generated code. Perplexity weights these identically to the ten-thousandth filler "the." That is the whole disease. The cure is to measure the tasks where pivotal tokens live, which is the next section.

Before we get there, one cheap metric sits between perplexity and full task evals and is worth knowing: the **per-token KL divergence** between the fp16 and quantized model's output distributions, on the *same* forced context. Perplexity asks "how surprised is the quantized model by the ground-truth token"; KL asks "how far did the quantized model's *whole distribution* drift from the fp16 model's, token by token." The difference matters because KL exposes the *maximum* per-token drift, not just the average, and it needs no labels:

$$
\text{KL}_i = \sum_{v} p^{\text{fp16}}(v \mid x_{<i}) \, \log \frac{p^{\text{fp16}}(v \mid x_{<i})}{p^{\text{quant}}(v \mid x_{<i})}
$$

Track the *tail* of the per-token KL — the 99th percentile and the max — not the mean. A quantization that leaves the mean KL near zero but produces a fat tail of high-KL tokens is precisely the "corrupts one token in five hundred" failure, and the KL tail flags it while perplexity smooths it away. This is the same batch-invariance and numerical-drift concern that [the sampling-numerics post](/blog/machine-learning/inference-engineering/sampling-numerics-determinism-and-batch-invariance) treats for a different cause. KL is not a substitute for a task eval — it tells you the distribution moved, not whether the *answer* is now wrong — but it is a cheap, label-free early-warning signal you can compute on any text, and a fat KL tail is a strong hint to go run the expensive task battery.

---

## 6. The evals that catch what perplexity misses

The discipline is simple to state and constantly skipped: **never quantize on a perplexity check alone.** Run perplexity *and* a small battery of task evals, each chosen because it is sensitive to a failure mode perplexity hides. You do not need a giant harness; you need a handful of tasks that between them cover the pivotal-token failures.

![Matrix mapping each eval to the failure mode it catches, its sensitivity, and a cited or derived anchor number](/imgs/blogs/activation-outliers-calibration-and-measuring-quality-loss-6.webp)

- **GSM8K (grade-school math).** The canonical single-token-sensitive eval. A chain of arithmetic where one wrong digit fails the problem — exactly the amplifier from the last section. PTPC-FP8 reports Llama-3.1-8B GSM8K at 73.2% (bf16) versus 70.8% (PTPC-FP8), a 2.4-point drop that is invisible in the 0.86% perplexity move ([vLLM 2025-02-24](https://vllm.ai/blog/2025-02-24-ptpc-fp8-rocm)). AutoRound reports Qwen3-8B W4A16 at 0.911 ([vLLM 2025-12-09](https://vllm.ai/blog/2025-12-09-intel-autoround-llmc)). Both are exact-match on a final numeric answer, so a quantization that nudges arithmetic shows up here first.
- **A code eval (HumanEval / MBPP-style).** Exact-match on executable code: a single wrong token — an off-by-one bound, a wrong operator, a renamed variable — turns a passing solution into a failing one. Code is where the "calibrated on English, deployed on code" mismatch from §4 shows its teeth.
- **A long-context retrieval eval (needle-in-a-haystack).** This one specifically catches the KV-quantization cliffs from [the KV-cache quantization post](/blog/machine-learning/inference-engineering/kv-cache-quantization-fp8-int8-and-the-accuracy-cliff). The vLLM FP8-KV post documents a needle-retrieval collapse from 91% to 13% at 100k-token contexts from imprecise FP32 accumulation ([vLLM 2026-04-22](https://vllm.ai/blog/2026-04-22-fp8-kvcache)) — a catastrophic failure that a short-context perplexity run cannot see, because it never uses a long context.
- **Instruction-following.** Does the quantized model still obey format constraints, honor system prompts, and stop when told? Quantization can quietly erode the fine control that matters most for agents and structured output, and it does not always show in an accuracy number.

Here is the honest scoreboard, with provenance on every cell.

| Eval | Failure mode it catches | Cited / derived anchor | Source |
|---|---|---|---|
| Perplexity (Wikitext) | Broad, average fluency | Llama-3.1-8B 9.43 → 9.51 (PTPC-FP8) | cited: [vLLM 2025-02-24](https://vllm.ai/blog/2025-02-24-ptpc-fp8-rocm) |
| GSM8K | Single-token arithmetic errors | 73.2% → 70.8%; Qwen3-8B W4A16 0.911 | cited: [vLLM 2025-02-24](https://vllm.ai/blog/2025-02-24-ptpc-fp8-rocm), [2025-12-09](https://vllm.ai/blog/2025-12-09-intel-autoround-llmc) |
| Code exact-match | Off-by-one / wrong-operator regressions | task-drop model: 0.99^k vs 0.96^k | derived |
| Long-context retrieval | KV-quant cliffs at long context | needle 91% → 13% @ 100k | cited: [vLLM 2026-04-22](https://vllm.ai/blog/2026-04-22-fp8-kvcache) |
| Instruction-following | Eroded format/constraint obedience | qualitative; measure per-task | reproduce: `harness.py` |

#### Worked example: how an average hides a broken task

Suppose the fp16 model scores 80 on chat, 78 on code, 73 on math, and 88 on retrieval — average 79.75. The quantized model scores 80, 78, 59, and 88 — the math collapsed 14 points, everything else held. The new average is 76.25. If your gate says "ship if the average dropped less than 5 points," a 3.5-point average drop *passes*, and you ship a model that cannot do arithmetic to users who will absolutely notice. Gate on the *minimum per-task margin* instead — "no single task may drop more than 3 points" — and the same result *fails* on the math task, which is the correct decision. The average is not just uninformative here; it is actively misleading, because a big regression on one task is exactly what averaging is designed to smooth away. *(Source: illustrative; the 14-point drop is the §5 chained-token derivation.)*

Two subtleties make this harder than "run four evals and average." First, **an aggregate score hides long-tail regressions.** If you report one number that blends math, code, retrieval, and chat, a 14-point collapse on math averages into invisibility against three tasks that held. Report and gate **per task**, never on the mean. Second, the thing that decides ship-or-not is not a score, it is a **threshold set before you quantize** — a per-task acceptance bar you commit to while you still have no incentive to move it. That is the discipline the whole [case study on the model that quietly got dumber](/blog/machine-learning/inference-engineering/case-study-the-quantized-model-that-quietly-got-dumber) turns on, and it is the last section.

---

## 7. Building the harness in nanoserve

Now the code that makes all of this real. Three pieces: compare per-channel against per-tensor error on a captured tensor (so you can *see* the crush from §2), and an eval harness that runs perplexity and a task on both models and prints both (so you can *catch* the §5 failure). Start with the error comparison, which turns the derivation into a measurement.

```python
# nanoserve/quant/compare.py
import torch

def quantize_symmetric(x, scale):
    q = torch.clamp(torch.round(x / scale), -127, 127)
    return q * scale                          # dequantized (fake-quant) value

def per_tensor_error(x):
    scale = x.abs().max() / 127               # one scale, set by the outlier
    xq = quantize_symmetric(x, scale)
    return (x - xq)

def per_channel_error(x, ch_dim=-1):
    # one scale per channel along ch_dim
    amax = x.abs().amax(dim=tuple(d for d in range(x.ndim) if d != ch_dim),
                        keepdim=True)
    scale = amax / 127
    xq = quantize_symmetric(x, scale)
    return (x - xq)

def compare(x, ch_dim=-1):
    e_pt = per_tensor_error(x)
    e_pc = per_channel_error(x, ch_dim)
    # error concentrated in the NORMAL channels under per-tensor
    per_ch_mse_pt = e_pt.pow(2).mean(dim=tuple(d for d in range(x.ndim)
                                               if d != ch_dim))
    per_ch_mse_pc = e_pc.pow(2).mean(dim=tuple(d for d in range(x.ndim)
                                               if d != ch_dim))
    return {
        "per_tensor_total_mse": e_pt.pow(2).mean().item(),
        "per_channel_total_mse": e_pc.pow(2).mean().item(),
        "worst_normal_ch_mse_ratio":
            (per_ch_mse_pt / (per_ch_mse_pc + 1e-12)).max().item(),
    }
```

Feed it a captured activation with a planted outlier and the numbers make the point:

```python
# scripts/compare_quant.py
import torch
from nanoserve.quant.compare import compare

torch.manual_seed(0)
x = torch.randn(512, 4096) * 0.5              # 4096 "normal" channels
x[:, 137] *= 60.0                             # one outlier channel, ~60x

print(compare(x, ch_dim=-1))
# Expected shape of the result:
#   per_tensor_total_mse   >>  per_channel_total_mse
#   worst_normal_ch_mse_ratio in the hundreds — the normal channels
#   carry almost all of the per-tensor error, exactly as derived in section 2.
```

The comment is a prediction, not a measurement I made — run it and you will see the per-tensor total MSE dwarf the per-channel one, with the excess concentrated in the normal channels the single scale crushed. *(Source: reproduce with `compare_quant.py`; the qualitative result is forced by the §2 derivation.)* This is the whole outlier story in fifteen lines, and it runs on a CPU.

Now the eval harness — the piece that would have stopped you from shipping the broken model in the intro. It does two things and reports both, side by side, never blended.

```python
# nanoserve/eval/harness.py
import torch, re

@torch.inference_mode()
def perplexity(model, tokenizer, text, ctx=2048, stride=1024, device="cuda"):
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    nlls, n_tokens = [], 0
    for i in range(0, ids.shape[1] - 1, stride):
        window = ids[:, i:i + ctx]
        if window.shape[1] < 2:
            break
        logits = model(window).logits[:, :-1]
        target = window[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
            reduction="sum")
        nlls.append(loss)
        n_tokens += target.numel()
    return torch.exp(torch.stack(nlls).sum() / n_tokens).item()

@torch.inference_mode()
def gsm8k_style(model, tokenizer, problems, device="cuda", max_new=256):
    """Exact-match on the final integer. `problems` = list of
    {"question": str, "answer": int}. This is the single-token-sensitive eval."""
    correct = 0
    for p in problems:
        prompt = f"Question: {p['question']}\nAnswer step by step, end with '#### <int>'.\n"
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False)
        text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        m = re.search(r"####\s*(-?\d+)", text)
        pred = int(m.group(1)) if m else None
        correct += int(pred == p["answer"])
    return correct / len(problems)

def report_both(fp16_model, quant_model, tokenizer, wikitext, problems):
    rows = []
    for name, model in [("fp16", fp16_model), ("quant", quant_model)]:
        rows.append({
            "model": name,
            "perplexity": round(perplexity(model, tokenizer, wikitext), 4),
            "gsm8k": round(gsm8k_style(model, tokenizer, problems), 4),
        })
    # print BOTH, never a blended score
    for r in rows:
        print(f"{r['model']:>6}  ppl={r['perplexity']:<8}  gsm8k={r['gsm8k']}")
    d_ppl = rows[1]["perplexity"] - rows[0]["perplexity"]
    d_task = rows[0]["gsm8k"] - rows[1]["gsm8k"]
    print(f"delta: ppl +{d_ppl:.4f}  gsm8k -{d_task:.4f}")
    return rows
```

The design choice that matters is the last three lines: it prints the perplexity delta and the task delta **separately**, so the exact failure this post is about — a tiny `ppl +0.08` next to a large `gsm8k -0.14` — is impossible to miss. A harness that reported a single blended "quality score" would average that disaster into a shrug.

#### Worked example: what the harness should tell you

Run `report_both` on an fp16 model and an aggressively quantized INT4 version of it, using a Wikitext slice and 200 GSM8K-style problems. On a *healthy* quantization you should see perplexity within roughly 1% and GSM8K within a couple of points — both fine, ship it. On a *broken* one (a per-tensor INT4 pass on an outlier-heavy model, or a checkpoint calibrated on the wrong distribution) you should see perplexity *still* within about 1% while GSM8K drops well into the double digits. That gap is the whole point of running both. *(Source: reproduce with `harness.py`; the direction and rough magnitude are grounded in PTPC-FP8's 2.4-point GSM8K drop at 0.86% perplexity, [vLLM 2025-02-24](https://vllm.ai/blog/2025-02-24-ptpc-fp8-rocm), and the chained-token derivation in §5.)* One warning on honest measurement, the same as everywhere in this series: fix the sampling to greedy (`do_sample=False`) for the task eval so the comparison is deterministic, and evaluate both models with the *identical* prompt template and problem set — a quantization "regression" is easy to fake by accidentally changing the harness between runs.

---

## 8. The decision: quantize, measure both, gate, ship or fall back

Put it together as the pipeline you actually run. It is short, it is boring, and skipping any step is how models quietly get dumber.

![Ordered pipeline from quantization through perplexity and task evaluation to a per-task threshold gate that decides ship or fall back](/imgs/blogs/activation-outliers-calibration-and-measuring-quality-loss-7.webp)

1. **Set per-task thresholds before you quantize.** While you have no stake in the outcome, write down the acceptance bar for each task: "GSM8K within 2 points, code within 3 points, long-context retrieval within 5 points, perplexity within 1%." Commit them to the repo.
2. **Quantize with the right defense and the right calibration.** Choose per-channel/per-token as the floor; add SmoothQuant/AWQ/learned rounding as the bit-width demands; calibrate on a set that *matches deployment* (code in the mix if you serve code, your languages if you serve them).
3. **Measure both, per task, on the deployment distribution.** Perplexity for a cheap smoke signal, the task battery for the truth. Report every task's number; never a blend.
4. **Gate against the pre-committed thresholds.** Any task that busts its bar fails the whole quantization — not "the average passed."
5. **Ship or fall back.** Passed everything: ship the quantized model. Busted a bar: fall back to a less aggressive scheme (INT8 instead of INT4, FP8 instead of INT8, or per-channel instead of per-tensor) and re-measure, or ship fp16 and take the memory hit. Falling back is not a failure of the process; it is the process working.

This is the exact gate whose absence powers [the case study of the model that quietly got dumber](/blog/machine-learning/inference-engineering/case-study-the-quantized-model-that-quietly-got-dumber) — INT4 shipped on a perplexity check alone, a task regression found weeks later in the field. And it is the measurement scaffold that [the quantization quality-vs-speed experiment](/blog/machine-learning/inference-engineering/experiment-quantization-quality-vs-speed-across-models) runs across the whole model matrix. The whole capstone [inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) folds this into the ship checklist for exactly one reason: the speed win is worthless if you cannot prove the quality held.

#### The ship-or-fall-back checklist

- [ ] Outlier report run; you know whether this model is outlier-heavy.
- [ ] Calibration set matches the deployment distribution (languages, code, digit-heavy inputs).
- [ ] Per-task thresholds written down *before* quantizing.
- [ ] Perplexity measured — treated as a smoke signal, never as sufficient.
- [ ] Task battery measured: math, code, long-context retrieval, instruction-following.
- [ ] Every task reported and gated *individually*; no blended score.
- [ ] Deterministic harness (greedy decode, identical prompts/templates across models).
- [ ] Decision recorded: ship, fall back a level, or ship fp16.

---

## Stress-testing before you trust it

A quantization that passes your gate on the happy path can still fail on inputs your gate never tried. Three stress tests catch the failures that hide between the cracks, and each maps to a mechanism from earlier in this post.

**Stress 1 — the model that looks fine on perplexity and fails GSM8K after INT4.** This is the intro, and it is the reason the harness prints two numbers. Take an outlier-heavy 8B model, quantize it to per-tensor INT4, and run `report_both`. The perplexity will land within about 1% of fp16 — genuinely fine — while GSM8K drops into the double digits, because the per-tensor scale crushed the normal channels to the ~1.3 effective bits we derived, and arithmetic is a chain of pivotal tokens that the product-of-successes formula punishes. The fix is per-channel scaling plus a defense from §3, and the *proof* the fix worked is the GSM8K number recovering — not the perplexity, which was never the problem. *(Source: reproduce with `harness.py`; magnitude grounded in the §5 derivation and PTPC-FP8's 2.4-point GSM8K drop.)*

**Stress 2 — calibrated on English, deployed on code and Vietnamese.** Calibrate a SmoothQuant or AWQ checkpoint on an English-only set, then evaluate it three ways: English chat, Python code generation, and Vietnamese. You should see the English task hold (it matches calibration) while code and Vietnamese regress more — because, as [the tokenizer post](/blog/machine-learning/inference-engineering/the-tokenizer-boundary-and-incremental-detokenization) established, those distributions tokenize differently and exercise channels the English calibration never measured, so the scales are tuned to the wrong outliers. The fix is not more English samples; it is a calibration set that *contains* code and Vietnamese in the proportions you serve. This is the single most common way a quantization that "passed eval" fails in production: the eval and the calibration shared a blind spot. *(Source: derived mechanism; tokenization-efficiency differences per the tokenizer post.)*

**Stress 3 — the outlier channel that only appears on certain inputs.** This is the §4 worked example turned into a test. Build a small calibration set from generic web text, quantize, and then evaluate on a held-out set that is *deliberately* heavy on the rare trigger — long digit strings, dense tables, base64 blobs, whatever your traffic contains that generic text does not. If a channel that only fires on those inputs was invisible during calibration, it got a tight scale, and it will clip hard on the trigger inputs. The failure shows up only on the adversarial eval slice, never on the generic one — which is why your eval set, like your calibration set, must include the tails of your real distribution, not just its body. A gate that only measures the average input is blind to exactly the inputs users complain about. *(Source: derived; input-dependence grounded in [arXiv:2208.07339](https://arxiv.org/abs/2208.07339).)*

The common thread across all three: **the gate is only as good as the distribution it measures on.** A per-task threshold on a happy-path eval set is a comfortable lie. Put your real tails — languages, code, digit-heavy inputs, long contexts — into both the calibration set and the eval set, and the gate starts telling the truth.

---

## Case studies and real numbers

Four public results, each cited, that ground everything above.

**PTPC-FP8 on ROCm (vLLM, [2025-02-24](https://vllm.ai/blog/2025-02-24-ptpc-fp8-rocm)).** Per-token-activation, per-channel-weight FP8, built specifically for activation outliers. Llama-3.1-8B: Wikitext perplexity 9.4281 (bf16) → 9.5093 (PTPC-FP8), a 0.86% degradation; GSM8K 73.2% → 70.8%, a 2.4-point drop that is 96.7% recovery. On the 70B model PTPC-FP8 reports 87.3% GSM8K — *above* the bf16 baseline, a reminder that quantization noise is not always a loss and that a single run's task number carries variance. The fused rowwise-scaled GEMM runs up to 2.5× faster than a naive two-step, and 70B throughput is within 1.01× of per-tensor FP8. This is the clean example of the §5 lesson: perplexity moved 0.86%, the task moved 3.3% relative, and only the second number tells you what a user feels.

**AutoRound × LLM Compressor (vLLM, [2025-12-09](https://vllm.ai/blog/2025-12-09-intel-autoround-llmc)).** Learned rounding by signed gradient descent over $V$, $\alpha$, $\beta$, minimizing block reconstruction error with ~128 calibration samples and ~200 iterations. Qwen3-8B W4A16 GSM8K 0.911. The post is explicit that the method is non-deterministic — two calibration runs produce two different checkpoints — which is itself a measurement lesson: if your eval delta is smaller than the run-to-run variance of the quantizer, you have not measured a real difference.

**FP8 KV-cache state-of-the-art (vLLM, [2026-04-22](https://vllm.ai/blog/2026-04-22-fp8-kvcache)).** Two facts we leaned on. First, uncalibrated models like Kimi-K2.5 show a persistent downward accuracy shift that only calibration repairs, while others (Qwen3-30B-A3B-Thinking) recover to 97%+ with the default per-tensor scale — the same technique, opposite outcomes, decided by the model's outliers. Second, the needle-in-a-haystack collapse from 91% to 13% at 100k-token contexts from imprecise FP32 accumulation: a catastrophic failure a short-context perplexity run is structurally blind to, and the reason a long-context retrieval eval belongs in the battery.

**LLM.int8() (Dettmers et al., [arXiv:2208.07339](https://arxiv.org/abs/2208.07339)).** The measurement of the outlier phenomenon itself: emergent outlier features past ~6.7B parameters, concentrated in a few feature dimensions, up to ~20× the typical magnitude, and disproportionately important to the model's predictions. This is the paper that turned "some models quantize badly" from folklore into a mechanism, and it is the empirical backbone of the whole first half.

---

## When to reach for this (and when not to)

**Reach for the full battery when you are shipping.** Any quantized model that touches production traffic gets the outlier report, the calibration-distribution check, and the per-task gate. The cost is a few hours of eval compute; the alternative is the intro's three-weeks-later support queue. There is no version of "we shipped a quantized model to users" where perplexity alone is enough diligence.

**You can skip most of the machinery when the model is not outlier-heavy.** Run the outlier report first; if `global_max_ratio` is small and the model recovers cleanly at your target bit-width on a quick GSM8K check, you do not need SmoothQuant and AWQ and learned rounding stacked three deep. Match the defense to the measured problem, not to a blog post's worst case.

**Do not write your own quantizer for a shipping model.** This post builds the *understanding* and the *measurement harness* — those you should own, because the eval gate is your product's safety net and nobody else will build it for your distribution. But the quantization *itself* — the SmoothQuant migration, the AWQ search, the AutoRound optimization — is exactly the kind of well-trodden, correctness-critical code where you should use a maintained library (LLM Compressor, AutoAWQ, AutoGPTQ, AutoRound) and vLLM's kernels. Build the observability and the gate; buy the quantizer. The engine you are writing needs to *load and serve* the quantized checkpoint well and *measure it honestly* — it does not need to reinvent the PTQ algorithm.

**Do not trust a single task number.** PTPC-FP8's 70B model scoring *above* bf16 on GSM8K is the cautionary tale: task evals have variance, especially at a few hundred problems. Gate on a margin, run the eval more than once when the delta is close, and treat a result inside the quantizer's own run-to-run noise as "no measured difference."

---

## Key takeaways

- **Outliers are the reason naive quantization fails.** A few activation channels carry magnitudes 10–100× the rest; a single per-tensor scale set by the outlier crushes the normal channels to about three levels, delivering ~1.58 bits where you budgeted 8.
- **The outlier lives on the reduction dimension**, which is exactly where per-token/per-channel scaling cannot reach — which is why SmoothQuant migrates the difficulty into the weights, AWQ protects the salient channels, and learned rounding re-rounds for output error.
- **Calibration is only valid for its distribution.** Calibrate on what you deploy; an English calibration set will not light up the channels that code or Vietnamese or long digit strings fire, and a small set can miss an input-triggered outlier entirely.
- **Perplexity has a structural blind spot.** It averages surprise over all tokens; a quantization that corrupts one token in five hundred moves perplexity ~0.4% while a chained task that multiplies per-token success collapses by double digits.
- **Measure both, per task, always.** Perplexity is a smoke signal; GSM8K, a code eval, long-context retrieval, and instruction-following catch what it hides. Report every task; never a blended score.
- **Set thresholds before you quantize**, gate each task individually, and treat falling back to a milder scheme as the process working, not failing.
- **Build the harness, buy the quantizer.** Own the outlier report and the eval gate; use a maintained library for the PTQ algorithm itself.
- **Every number needs a provenance.** Derived, cited with a link, or reproduce-it-yourself with a range — a quantization with an invented accuracy number is worse than none.

---

## Further reading

- **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale** — Dettmers et al., [arXiv:2208.07339](https://arxiv.org/abs/2208.07339). The measurement of the outlier phenomenon and the mixed-precision response.
- **SmoothQuant** — Xiao et al., [arXiv:2211.10438](https://arxiv.org/abs/2211.10438). The difficulty-migration derivation, folded into the previous layer.
- **AWQ: Activation-aware Weight Quantization** — Lin et al., [arXiv:2306.00978](https://arxiv.org/abs/2306.00978). Protecting the ~1% salient channels identified by activation statistics.
- **GPTQ** — Frantar et al., [arXiv:2210.17323](https://arxiv.org/abs/2210.17323). Second-order learned quantization; the ancestor of AutoRound's learned rounding.
- **PTPC-FP8 on ROCm** — vLLM, [2025-02-24](https://vllm.ai/blog/2025-02-24-ptpc-fp8-rocm). Per-token/per-channel FP8 built for outliers, with the perplexity-vs-GSM8K numbers used throughout.
- **AutoRound × LLM Compressor** — vLLM, [2025-12-09](https://vllm.ai/blog/2025-12-09-intel-autoround-llmc). Learned rounding by signed gradient descent, and its non-determinism.
- **The State of FP8 KV-Cache** — vLLM, [2026-04-22](https://vllm.ai/blog/2026-04-22-fp8-kvcache). Calibration necessity and the long-context needle cliff.
- **Within the series** — the [KV-cache quantization post](/blog/machine-learning/inference-engineering/kv-cache-quantization-fp8-int8-and-the-accuracy-cliff), the [FP8/FP4 hardware post](/blog/machine-learning/inference-engineering/fp8-and-fp4-inference-what-the-hardware-actually-gives-you), the [weight-only quantization post](/blog/machine-learning/inference-engineering/weight-only-quantization-in-your-engine-gguf-awq-gptq-at-load-time), and the general [quantization in LLM guide](/blog/machine-learning/large-language-model/quantization-in-llm).

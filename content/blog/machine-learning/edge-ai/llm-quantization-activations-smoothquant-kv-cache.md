---
title: "LLM quantization II: activations, SmoothQuant, FP8, and the KV-cache"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Weight-only int4 shrinks an LLM but leaves activations in fp16 — learn to quantize activations and the KV-cache too, defeat the outlier problem with SmoothQuant and FP8, and pick the right W4A16/W8A8/FP8/W4A8 scheme for your hardware and workload."
tags:
  [
    "edge-ai",
    "model-optimization",
    "quantization",
    "smoothquant",
    "fp8",
    "kv-cache",
    "llm",
    "inference",
    "efficient-ml",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/llm-quantization-activations-smoothquant-kv-cache-1.png"
---

In the [previous post on weight-only quantization](/blog/machine-learning/edge-ai/llm-quantization-weight-only-gptq-awq) we took a 7-billion-parameter model and squeezed its weights down to four bits with GPTQ and AWQ. The model that was 14 GB in fp16 became roughly 3.5 GB, and on a single consumer GPU — or even a laptop with `llama.cpp` — it suddenly fit and ran. For a single user chatting one token at a time, that was the whole game. Decode is memory-bound; the bottleneck is streaming the weights off DRAM; cut the weight bytes by four and you go almost four times faster. Job done.

Then someone asked me to make the *prefill* faster. A user pastes a 16,000-token document and wants the first token back quickly. And separately, someone wanted to serve thirty users at once on the same GPU. Both of those workloads broke my mental model, because both of them are *not* decode-bound in the same way a single chat is. Prefill crunches the entire prompt in one giant matrix multiply — it is compute-bound, and weight-only int4 does almost nothing for it because the math still runs in fp16. And batch serving is dominated by a memory term I had been ignoring entirely: the KV-cache, which at long context and high batch can dwarf the weights themselves.

The fix for both is the same uncomfortable step: you have to quantize the *activations* too, and you have to quantize the KV-cache. That is genuinely harder than quantizing weights, and the reason is a single, stubborn phenomenon that anyone who has tried it has been burned by — **activation outliers**. A few channels in a transformer's activations carry magnitudes tens of times larger than everything else, and those few channels poison a naive per-tensor int8 scale so badly that the model's perplexity can jump by a thousand points. Figure 1 shows exactly how one outlier channel wrecks the scale for every other channel.

![A two column comparison showing fp16 activations with a few outlier channels far larger than the rest, and naive per-tensor int8 where the outlier sets a coarse step that rounds all normal channels down to a couple of levels and perplexity explodes.](/imgs/blogs/llm-quantization-activations-smoothquant-kv-cache-1.png)

By the end of this post you will understand why those outliers exist, and you will have three concrete tools to beat them: **LLM.int8()**, which surgically keeps the outlier dimensions in fp16; **SmoothQuant**, which mathematically migrates the difficulty from activations onto weights so both become quantizable; and **FP8**, a floating 8-bit hardware format that tolerates outliers by construction. You will know how to quantize the KV-cache in `llama.cpp` and vLLM and what it costs you. And you will be able to look at any deployment — single-stream chat, long-context prefill, high-batch serving — and pick the right scheme from the four that matter: W4A16, W8A8, FP8, and W4A8. This is the second half of the quantization lever in the four-lever frame; if you have not seen the overall map, skim [the taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) first.

## The notation: what W4A16 and W8A8 actually mean

Before anything else, fix the vocabulary, because the whole post hinges on it. A quantization scheme for an LLM is named by two numbers: the bit-width of the **weights** (W) and the bit-width of the **activations** (A). So:

- **W4A16** — 4-bit weights, 16-bit activations. This is weight-only quantization, exactly what GPTQ and AWQ produce. The weights are stored in int4 (or a k-quant), de-quantized to fp16 on the fly inside the kernel, and the actual arithmetic happens in fp16. The win is purely memory: smaller weights, fewer bytes streamed.
- **W8A8** — 8-bit weights *and* 8-bit activations, both integer. Now the matrix multiply itself runs on int8 hardware. The win is compute: int8 tensor cores do roughly twice the operations per second of fp16, so the math is faster, not just the memory traffic.
- **FP8** — 8-bit weights and activations, but in a *floating-point* 8-bit format (E4M3 or E5M2) rather than integer. The arithmetic runs on FP8 tensor cores, which Hopper and Ada GPUs have natively.
- **W4A8** — 4-bit weights, 8-bit activations. The best of both worlds in principle: tiny weights *and* fast activation math. In practice the trickiest to make accurate.

The reason this distinction matters so much is that the two numbers attack two different bottlenecks. Quantizing W helps the **memory-bound** part of inference (decode, where you stream weights one token at a time). Quantizing A helps the **compute-bound** part (prefill, where you do a big GEMM over many tokens at once). If you have not internalized why those two phases have opposite bottlenecks, read [the roofline model post](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) — it is the single most useful diagnostic for this entire decision. The short version: a single decode step is a matrix-times-vector (GEMV), which moves a lot of weight bytes and does very little math per byte, so it sits on the memory roof. A prefill over 4,000 tokens is a matrix-times-matrix (GEMM), which reuses each loaded weight across all 4,000 tokens, so it does a lot of math per byte and sits on the compute roof.

That single fact is the spine of everything below. Hold onto it: **weight bits buy decode; activation bits buy prefill.**

## Why activations are so much harder than weights

Quantizing weights was, in retrospect, the easy half. Weights are *static*: you have them all, offline, before a single token flows. You can spend GPU-hours analyzing their distribution, applying second-order corrections like GPTQ's Hessian-based error compensation, and searching for the per-group scales that minimize reconstruction error. The weight distribution of a trained transformer is also relatively well-behaved — roughly Gaussian, with no monstrous outliers in most layers. That is why W4A16 with GPTQ or AWQ loses under half a point of perplexity.

Activations are a different animal for two reasons.

**First, they are dynamic.** The activation values depend on the *input*. You do not know them ahead of time; you only know the distribution from a calibration set, and the real distribution at serving time may differ. So activation quantization is inherently riskier — you are committing to scales based on a sample.

**Second, and far more importantly, transformer activations have extreme outliers.** This is the crux of the whole post. In the residual stream of a large transformer, a small number of feature dimensions — channels — develop systematically huge magnitudes. We are not talking about a 2× spread. Dettmers and colleagues, in the LLM.int8() paper (2022), measured outlier features with magnitudes up to roughly 20× the typical activation, concentrated in a handful of dimensions, and crucially these emerge at scale: they become severe past about 6.7 billion parameters, exactly when LLMs start being useful. Later work measured per-channel maxima 70× and more above the median in certain layers.

### What an outlier does to a per-tensor scale

To see the damage precisely, recall how uniform integer quantization works. To quantize a tensor $X$ to a $b$-bit signed integer, you pick a scale $s$ and compute

$$
X_q = \mathrm{round}\!\left(\frac{X}{s}\right), \qquad s = \frac{\max(|X|)}{2^{b-1}-1}.
$$

For symmetric int8, $2^{b-1}-1 = 127$. The step size $s$ is set by the *single largest absolute value in the tensor*. That is the trap. If 99.9% of your activations live in $[-1, 1]$ but one channel spikes to $70$, then a per-tensor scale becomes

$$
s = \frac{70}{127} \approx 0.55.
$$

Now ask what happens to a perfectly ordinary activation of value $0.4$. It quantizes to $\mathrm{round}(0.4 / 0.55) = \mathrm{round}(0.73) = 1$, then de-quantizes back to $0.55$ — a 38% error on that value. A value of $0.2$ rounds to $0$ and is annihilated entirely. The outlier has consumed almost the entire integer range; everything else is crammed into two or three levels near zero. Figure 1 above is exactly this collapse: a per-tensor int8 scale governed by one outlier rounds the informative channels down to noise.

The information-theoretic way to say this: uniform quantization gives you a fixed number of code points, and a single far-out value forces those points to spread over a range that is mostly empty. The *effective* bit-width for the bulk of your data drops from 8 bits to maybe 2. The signal-to-quantization-noise ratio, which for a well-matched uniform quantizer is about $\text{SQNR} \approx 6.02\,b + 1.76$ dB, falls off a cliff because the assumption behind that formula — that the signal fills the range — is violated by a factor of 70.

### Where the outliers come from (and why they are systematic)

It helps enormously to know that these outliers are not random accidents — they are a structural feature of how transformers compute, which is why every large model has them and why the mitigations are reusable. Three observations from the interpretability and quantization literature:

- **They live in fixed channels.** The outlier dimensions are largely the *same* hidden-dimension indices across tokens and even across inputs. A given layer might have its outliers in dimensions 1,512 and 3,608, more or less regardless of what text you feed it. That consistency is exactly what makes per-channel handling (and SmoothQuant's per-channel factor) work: you can characterize the outlier channels once on a calibration set and trust they will be the same at serving time.
- **They grow through the residual stream.** Transformers add each layer's output back into a running residual stream. Some features get repeatedly reinforced layer after layer, accumulating large magnitudes. This is why outliers are *worse in later layers* and worse in *bigger models* (more layers, more accumulation) — and why Dettmers measured the sharp onset around 6.7B parameters.
- **They often implement attention sinks.** Recent work links large-magnitude features to "attention sinks" — the model parks excess attention probability on a few tokens (often the first token) and uses high-magnitude channels to do it. These are not noise to be cleaned up; they are part of the algorithm the model learned. Destroy them and attention destabilizes. This is the mechanistic reason clipping fails.

The takeaway for a practitioner: treat the outliers as a *known, characterizable* property of the model, not a surprise. You measure where they are, then you choose a method (keep them in fp16, migrate them to weights, or use a format that tolerates them). You never just hope they are small.

### Why not just clip the outliers?

The first instinct of every engineer who meets this problem is: clip the outliers. Set the scale by, say, the 99.9th percentile instead of the max, and saturate the few values above it. Sometimes this helps a little. But for LLMs it usually does not, and the reason is humbling: those outlier channels are not noise. They are *load-bearing*. The model learned to put critical information into a few high-magnitude features — they often act like a per-token bias or attention sink. Clip them and you damage the very channels the model relies on most. Dettmers' team showed that zeroing out as few as a handful of outlier dimensions could collapse the model's accuracy entirely. So you cannot delete them and you cannot squash them. You have to *handle* them. That is what the next three sections are about.

## The granularity dial: per-tensor, per-token, per-channel, per-group

Before the three named methods, there is a single dial that underlies all of them and that you must understand to reason about activation quantization at all: **how many distinct scales do you use, and along which axis?** Every quantization scheme is, at bottom, a choice of how finely you tile the tensor with scales. Coarser scales are cheaper to store and faster (one division for a whole tensor) but more vulnerable to outliers; finer scales cost storage and runtime but isolate outliers into smaller groups. The whole art of activation quantization is choosing the granularity that contains the outliers without paying too much for the scales.

There are four standard granularities, from coarsest to finest:

- **Per-tensor** — one scale for the entire tensor. The cheapest and the most fragile; this is the one Figure 1 shows blowing up. A single outlier anywhere in the tensor sets the scale for every value.
- **Per-token (per-row)** — one scale per token (per row of the activation matrix). Now a giant value in token $i$ only coarsens token $i$'s row, not the whole tensor. This helps a lot when outliers are concentrated in *some tokens* — but in transformers the outliers are concentrated in *channels*, which cut across every token, so per-token scaling alone does not fully solve it (recall the OPT-13B per-token row was still at ~170 perplexity).
- **Per-channel (per-column)** — one scale per hidden dimension. This is the natural axis for transformer activation outliers, because the outliers *are* channels. A per-channel scale gives the outlier channel its own large scale and leaves the normal channels with fine scales. The problem: for activations you cannot easily do per-channel quantization in a standard GEMM, because the channel (the reduction dimension) is summed over inside the matmul — you would need to apply a different scale to each element being multiplied, which the integer tensor-core instruction cannot do. This is exactly the gap SmoothQuant fills: it moves the per-channel difficulty into the weights (where per-channel scaling along the *output* dimension *is* compatible with the GEMM) instead.
- **Per-group (block-wise)** — one scale per contiguous block of, say, 64 or 128 elements. The finest practical granularity, used heavily for *weights* (GPTQ/AWQ k-quants are per-group along the input dimension). It bounds the damage of any outlier to its own small block. For weights this is cheap and standard; for activations it is rarer because of the GEMM-axis constraint.

### Why per-channel is incompatible with the int8 GEMM (and per-token is not)

This is the subtle hardware fact that explains the entire shape of activation quantization, so it is worth being precise. An int8 matmul computes $Y_{io} = \sum_j X_{ij} W_{jo}$ where $j$ runs over the hidden dimension. To make this an *integer* dot product, both operands along $j$ must share scales that factor *out* of the sum. A **per-token** scale on $X$ (indexed by $i$) and a **per-output-channel** scale on $W$ (indexed by $o$) both factor out cleanly:

$$
Y_{io} = s^X_i \, s^W_o \sum_j X^q_{ij} W^q_{jo}.
$$

The integer accumulator $\sum_j X^q_{ij} W^q_{jo}$ runs at full int8 speed; you rescale once at the end by $s^X_i s^W_o$. But a **per-channel** activation scale $s^X_j$ — indexed by the *reduction* dimension $j$ — sits *inside* the sum and cannot be factored out:

$$
Y_{io} = \sum_j (s^X_j X^q_{ij}) W^q_{jo} = \sum_j s^X_j X^q_{ij} W^q_{jo},
$$

so you would have to multiply every product by a different $s^X_j$ before accumulating, which is not what int8 tensor-core instructions do. *That* is why you cannot just "quantize activations per-channel" and be done. The natural axis for the outliers (per-channel) is the one axis the hardware forbids for activations. SmoothQuant's whole elegance is that it converts the forbidden per-channel-activation problem into an allowed per-output-channel-weight problem. FP8 sidesteps it differently — by making even a coarse (per-tensor) scale tolerable, because the floating format itself adapts precision per value.

This is also the reason **per-token dynamic** activation scaling is the cheap, always-available baseline: it is GEMM-compatible (the scale factors out), it needs no calibration (you compute each row's max at runtime), and it captures token-to-token variation. It is the default in vLLM's int8 and FP8 dynamic modes. It just is not *enough* on its own for outlier-heavy models, which is why you layer SmoothQuant on top.

#### The error-variance argument for finer granularity

There is a clean quantitative way to see why finer granularity helps, beyond "it isolates outliers." For a uniform quantizer with step $s$, rounding error is approximately uniform on $[-s/2, s/2]$, so the per-element quantization-error variance is

$$
\sigma_q^2 = \frac{s^2}{12}.
$$

The step $s$ is proportional to the *range* of the group it covers. If a group's range is dominated by an outlier of magnitude $M$ while the useful signal has magnitude $\sigma_x \ll M$, then $s \propto M$ and the noise floor $\sigma_q \propto M$ swamps the signal — the SQNR for the bulk values is $\approx 20\log_{10}(\sigma_x / \sigma_q) = 20\log_{10}(\sigma_x / M) - \text{const}$, which is hugely *negative* when $M \gg \sigma_x$. Shrink the group so the outlier no longer shares a scale with the bulk, and the bulk's step drops from $\propto M$ to $\propto \sigma_x$ — recovering on the order of $20\log_{10}(M/\sigma_x)$ dB of SQNR. For a 70× outlier that is about $37$ dB recovered, roughly 6 effective bits. Granularity is not a minor tuning knob; it is worth several effective bits when outliers are present. Every method below is, in some sense, a clever way to buy that granularity without paying the GEMM-axis penalty.

## LLM.int8(): keep the outliers in fp16, quantize the rest

The first and most direct solution, from Dettmers, Lewis, Belkada, and Zettlemoyer (2022), is conceptually simple: if a small number of dimensions cause all the trouble, treat them separately. Run the bulk of the matrix multiply in int8 for speed and memory, but pull out the outlier columns and run *those* in fp16. Then add the two results. This is **mixed-precision decomposition**, and it is what `bitsandbytes`'s `load_in_8bit=True` does under the hood. Figure 3 shows the two-path structure.

![A branching dataflow graph where the fp16 input is checked for outlier columns above a threshold, the rare outlier columns go through an fp16 matmul, the remaining columns go through an int8 matmul that is then dequantized, and both partial results are summed into the output.](/imgs/blogs/llm-quantization-activations-smoothquant-kv-cache-3.png)

Mechanically, for an activation matrix $X$ of shape $[\text{tokens}, h]$ and weight $W$ of shape $[h, o]$:

1. Identify the set of hidden dimensions $O$ where any activation exceeds a threshold (the paper uses an absolute magnitude of $6.0$). Empirically $|O|$ is tiny — often 0.1% or fewer of the $h$ dimensions, maybe 6 columns out of 4,096.
2. Split the multiply. The columns in $O$ stay in fp16: $Y_{\text{fp16}} = X_{:,O} \, W_{O,:}$. The rest are quantized to int8 with **row-wise** scales on $X$ and **column-wise** scales on $W$ (vector-wise quantization, which already sidesteps the per-tensor problem for the non-outlier part): $Y_{\text{int8}} = \mathrm{dequant}(X_{:,\bar O}^{q}\, W_{\bar O,:}^{q})$.
3. Sum: $Y = Y_{\text{fp16}} + Y_{\text{int8}}$.

Because $|O|$ is so small, the fp16 sub-matmul is cheap, and the int8 path — now free of the outliers that would have wrecked its scale — quantizes cleanly. LLM.int8() recovered full fp16 accuracy on models up to 175B with zero degradation. It was a landmark: it made 8-bit inference of huge models *exact*, not approximate.

Here is the catch, and it is the reason SmoothQuant exists. LLM.int8() is fast for *memory* but not for *compute*. The two-path structure — detecting outliers per batch, gathering those columns, running a separate fp16 GEMM, and scattering the results back — has overhead. On the decode path, where you are memory-bound anyway, it is fine and often a net win. But on the compute-bound prefill path, the irregular outlier handling and the extra fp16 GEMM can make it *slower* than just running fp16. It also does not give you an int8-throughout GEMM, so you never reach the 2× compute speedup that int8 tensor cores promise. It solves accuracy, not prefill latency.

```python
# LLM.int8() via bitsandbytes + transformers — one flag.
# This is W8A8 with outlier columns kept in fp16 automatically.
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,            # triggers LLM.int8() mixed-precision matmul
    device_map="auto",
    torch_dtype=torch.float16,
)
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# The threshold that decides which columns are "outliers" is tunable:
#   BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
# Lower threshold -> more columns kept in fp16 -> safer but slower.
out = model.generate(**tok("The KV-cache dominates memory at",
                            return_tensors="pt").to(model.device),
                     max_new_tokens=40)
print(tok.decode(out[0]))
```

So LLM.int8() is the right tool when your goal is to *fit* a big model in 8-bit memory without losing accuracy and you mostly care about decode. It is not the tool when you need fast int8 prefill. For that, we need to make the activations quantizable *without* a separate outlier path — which means making the outliers smaller. That is SmoothQuant.

## SmoothQuant: migrate the difficulty from activations to weights

SmoothQuant (Xiao, Lin, Seznec, Wu, Demouth, Han, 2022) is one of those ideas that feels obvious only after you see it. The activations have outliers; the weights do not. The matrix multiply is $Y = XW$. What if we could *transfer* the spikiness from $X$ to $W$, so that both end up moderately easy to quantize, while the product $XW$ stays mathematically identical? Figure 2 shows exactly this migration.

![A two column comparison where before smoothing the activations are spiky and dominate quantization error while weights are flat, and after dividing activations by a per channel factor and multiplying weights by it both sides have balanced ranges and the product is unchanged.](/imgs/blogs/llm-quantization-activations-smoothquant-kv-cache-2.png)

### The math: a per-channel smoothing factor

The key observation is that matrix multiplication has a built-in invariance. Insert a diagonal scaling matrix $\mathrm{diag}(s)$ and its inverse between $X$ and $W$:

$$
Y = XW = \big(X\,\mathrm{diag}(s)^{-1}\big)\big(\mathrm{diag}(s)\,W\big) = \hat X \hat W.
$$

Here $s \in \mathbb{R}^{h}$ is a per-channel vector, one scalar per hidden dimension. Dividing column $j$ of $X$ by $s_j$ shrinks that channel's activations; multiplying row $j$ of $W$ by $s_j$ compensates exactly, so $\hat X \hat W = XW$ for any positive $s$. The product is invariant. We get to *choose* $s$ to make both $\hat X$ and $\hat W$ easy to quantize.

This is the whole trick. By choosing $s_j$ large for an outlier channel $j$, we divide that channel of $X$ down to a normal magnitude — the outlier is gone from the activations — and we push the factor into $W$, which had headroom to spare. The "difficulty" of quantization (the dynamic range) is conserved overall but redistributed so neither side is catastrophic.

### Deriving the balance: the $\alpha$ that equalizes ranges

The question is how to pick $s$. If you push *all* the difficulty into $W$ (very large $s$), you just move the outlier problem to the weights — now the weights are unquantizable. If you push none ($s = 1$), nothing changes. You want the sweet spot where activations and weights are *equally* hard. SmoothQuant parameterizes this with a single hyperparameter $\alpha \in [0,1]$, the **migration strength**:

$$
s_j = \frac{\big(\max_i |X_{ij}|\big)^{\alpha}}{\big(\max_i |W_{ij}|\big)^{1-\alpha}}.
$$

Read this carefully because it is the heart of the method. $\max_i |X_{ij}|$ is the largest absolute activation in channel $j$ (over the calibration tokens $i$); $\max_i |W_{ij}|$ is the largest absolute weight in the corresponding input row of $W$. The exponents split the burden:

- At $\alpha = 1$, $s_j = \max|X_{ij}|$, which after dividing makes every activation channel's max equal to $1$ — all the difficulty goes to the weights.
- At $\alpha = 0$, $s_j = 1/\max|W_{ij}|$, normalizing the weights and leaving the activations spiky.
- At $\alpha = 0.5$, the smoothed activation channel max becomes $\sqrt{\max|X_{ij}| \cdot \max|W_{ij}|}$ — the *geometric mean* of the two maxima — and so does the smoothed weight max. Both sides end up with the same per-channel range. That is the balance.

To see the equalization explicitly: after smoothing, the activation max in channel $j$ is $\max|X_{ij}| / s_j$ and the weight max is $\max|W_{ij}| \cdot s_j$. Plug in $s_j$ from the formula:

$$
\frac{\max|X_{ij}|}{s_j} = \big(\max|X_{ij}|\big)^{1-\alpha}\big(\max|W_{ij}|\big)^{1-\alpha},
$$
$$
\max|W_{ij}| \cdot s_j = \big(\max|X_{ij}|\big)^{\alpha}\big(\max|W_{ij}|\big)^{\alpha}.
$$

At $\alpha = 0.5$ both expressions equal $\big(\max|X_{ij}|\,\max|W_{ij}|\big)^{0.5}$ — identical. The smoothing factor literally interpolates, channel by channel, between "normalize the activations" and "normalize the weights," and $\alpha = 0.5$ is the geometric midpoint where their ranges meet. In practice $\alpha = 0.5$ is the default and works for most models; models with very severe outliers (some GLM and OPT variants) want $\alpha$ a bit higher (0.6–0.8) to drag more of the spike onto the weights.

One more thing that makes SmoothQuant *free at inference time*: the division $X \to X/s$ is folded into the *preceding* layer. The activations entering a linear layer come out of a LayerNorm (or another linear). You bake $1/s$ into that LayerNorm's scale parameters and bake $s$ into the linear's weights, both offline. At serving time there is no extra op — the smoothing has been absorbed into existing parameters. The model is now plain W8A8 with no outlier path, so it runs on standard int8 GEMM kernels at full speed. That is the payoff LLM.int8() could not give: fast int8 prefill *and* recovered accuracy.

### SmoothQuant in code

```python
# SmoothQuant: compute per-channel smoothing scales from a calibration set,
# fold them into LayerNorm + linear weights, then quantize to W8A8.
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-6.7b", torch_dtype=torch.float16, device_map="auto")
tok = AutoTokenizer.from_pretrained("facebook/opt-6.7b")

# 1) Collect per-channel activation maxima on a small calibration set.
act_max = {}                                   # name -> running max per channel
def hook(name):
    def fn(mod, inp, out):
        x = inp[0].detach().abs().amax(dim=(0, 1))   # max over batch + tokens
        act_max[name] = torch.maximum(act_max.get(name, x), x)
    return fn

handles = [m.register_forward_hook(hook(n))
           for n, m in model.named_modules()
           if isinstance(m, torch.nn.Linear)]

calib = ["The KV-cache dominates memory at long context.",
         "SmoothQuant migrates difficulty from activations to weights."]  # use ~512 real samples
for s in calib:
    model(**tok(s, return_tensors="pt").to(model.device))
for h in handles:
    h.remove()

# 2) For each linear, compute s_j with alpha=0.5 and fold it in.
ALPHA = 0.5
def smooth_linear(ln, fc, x_max):
    w_max = fc.weight.abs().amax(dim=0)                       # per input-channel
    s = (x_max.pow(ALPHA) / w_max.pow(1 - ALPHA)).clamp(min=1e-5)
    ln.weight.div_(s)                  # X -> X / s, absorbed into LayerNorm
    fc.weight.mul_(s.view(1, -1))      # W -> s * W, exact compensation
# (In real code: pair each LayerNorm with the q/k/v/fc1 linears it feeds.)
```

The production path is even simpler — you do not usually hand-roll this. The modern way is to use `llm-compressor` (the successor to the original `smoothquant` repo, maintained by the vLLM/Neural Magic team), which applies SmoothQuant plus GPTQ-style weight quantization and emits a W8A8 checkpoint that vLLM and TensorRT-LLM load directly:

```python
# Production W8A8 (SmoothQuant + GPTQ weights) with llm-compressor.
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

recipe = [
    SmoothQuantModifier(smoothing_strength=0.5),               # alpha
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]
oneshot(model="meta-llama/Llama-2-7b-hf", dataset="open_platypus",
        recipe=recipe, output_dir="Llama-2-7b-W8A8",
        max_seq_length=2048, num_calibration_samples=512)
# vLLM then serves it natively:  vllm serve Llama-2-7b-W8A8
```

### Static vs dynamic activation scales, and how to calibrate

SmoothQuant equalizes the *ranges*, but you still have to decide *when* the activation scale is computed, and this choice is just as consequential as $\alpha$. There are two regimes:

- **Static quantization** computes the activation scales *once*, offline, from a calibration set, and bakes them into the model. At serving time there is no scale computation — the fastest path. The risk: the served distribution must match the calibration distribution. If you calibrate on Wikipedia and serve source code, the activation statistics shift, the outlier channels move, and your fixed scales clip the wrong things. Static is great when traffic is predictable and you can calibrate on representative data.
- **Dynamic quantization** computes the activation scale *at runtime*, per token (the per-token granularity from above), from the actual values in flight. It adds a small per-step reduction (find the max of each row) but it always fits the real distribution, so it is robust to distribution shift and needs no calibration for activations. vLLM's default int8 and FP8 modes are dynamic per-token. The cost is a few percent of throughput for the runtime max-reduction.

The practical rule: **use dynamic per-token activation scaling unless you have measured that static is faster *and* your traffic is stable.** Dynamic is the safer default and removes a whole class of "it worked in eval, broke in prod" failures caused by calibration mismatch. Weights, being static, are always quantized statically — only activations and the KV-cache have this dynamic option.

When you *do* calibrate (static activation scales, or SmoothQuant's per-channel maxima, or static KV scales), the calibration set matters more than people expect. Three rules that have saved me:

1. **Representative, not random.** Use real samples from your production distribution — the same languages, the same prompt formats, the same domains. A few hundred sequences of the right kind beat tens of thousands of the wrong kind.
2. **Long enough sequences.** Activation outliers and KV statistics depend on position; calibrate at sequence lengths close to what you will serve, not on 32-token snippets if you serve 4k contexts.
3. **512 samples is usually plenty.** GPTQ, AWQ, and SmoothQuant all converge with a few hundred to ~1,000 calibration samples; beyond that there is little gain. The sensitivity is to *representativeness*, not raw count.

```python
# A representative calibration reader for ORT / llm-compressor style PTQ.
# The single most common quantization failure is a non-representative calib set.
from datasets import load_dataset

def build_calibration(tokenizer, n=512, seqlen=2048):
    # Mix domains to match production traffic; avoid a single narrow source.
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    samples = []
    for ex in ds:
        ids = tokenizer(ex["text"], return_tensors="pt").input_ids
        if ids.shape[1] >= seqlen:                  # only long-enough sequences
            samples.append(ids[:, :seqlen])
        if len(samples) >= n:
            break
    return samples                                  # feed these through the model
```

#### Worked example: int8 activations that SmoothQuant rescues

Let me make the recovery concrete with the numbers from the SmoothQuant paper and reproductions. Take OPT-13B, a model notorious for severe activation outliers, and evaluate WikiText-2 perplexity (lower is better). fp16 baseline perplexity is about **14.6**.

| Scheme | Activation handling | WikiText-2 perplexity | Verdict |
|---|---|---|---|
| fp16 | — | 14.6 | baseline |
| W8A8, naive per-tensor | none | 1.4 × 10⁴ (≈ 14,000) | broken — random output |
| W8A8, per-token dynamic | finer scales, still no migration | ~170 | still unusable |
| W8A8, LLM.int8() | outliers kept in fp16 | 14.6 | exact, but slow prefill |
| W8A8, SmoothQuant α=0.5 | outliers migrated to W | ~14.7 | recovered, full-speed int8 |

The naive per-tensor int8 number is not a typo — without handling the outliers, the model produces garbage; perplexity in the thousands means it has essentially lost the ability to predict the next token. Per-token dynamic scaling (a finer scale per row) helps an order of magnitude but is still far from usable on a model this outlier-heavy. LLM.int8() nails the accuracy but at the cost of the mixed-precision overhead. SmoothQuant lands within a tenth of a perplexity point of fp16 *and* keeps a clean int8 GEMM, which is why it became the default activation-quantization recipe for serving. That is the entire value proposition in one table: **a model where naive int8 activations destroy accuracy, fully recovered by migrating the outliers into the weights.**

## FP8: a floating format that tolerates outliers by construction

SmoothQuant fixes the outlier problem in software by reshaping the data to fit a uniform integer grid. FP8 fixes it in *hardware* by changing the grid. The deep reason int8 struggles with outliers is that its 256 code points are spaced *uniformly* — the gap between adjacent representable values is constant. A floating-point format spaces its code points *logarithmically*: dense near zero, sparse far away. That is precisely the shape that matches activation distributions, which have most of their mass near zero and a long tail. Figure 4 contrasts the two.

![A two column comparison of int8 with 256 evenly spaced levels whose step is fixed by the range so outliers coarsen small values, versus FP8 E4M3 with four exponent and three mantissa bits giving logarithmically spaced levels that stay fine near zero while still representing the outliers.](/imgs/blogs/llm-quantization-activations-smoothquant-kv-cache-4.png)

### The two FP8 formats and why there are two

The FP8 standard that Micikevicius and colleagues at NVIDIA, Arm, and Intel proposed (2022) defines two 8-bit floating formats, distinguished by how they split the 8 bits between exponent and mantissa:

- **E4M3** — 1 sign bit, 4 exponent bits, 3 mantissa bits. Range roughly $\pm 448$, with finer precision (3 mantissa bits = 8 levels per binade). Used for **weights and activations** in the forward pass, where precision matters more than range.
- **E5M2** — 1 sign, 5 exponent, 2 mantissa. Range roughly $\pm 57{,}344$, coarser precision (2 mantissa bits). Used for **gradients** in training, where the dynamic range is enormous but precision tolerance is higher.

For *inference* — our concern here — E4M3 is the workhorse for both weights and activations. The crucial property is its range. An E4M3 value can represent magnitudes from tiny denormals up to $448$. So an activation outlier of $70$ fits comfortably *as a single value*, with a few mantissa bits of precision, while the bulk of the distribution near $\pm 1$ also gets several mantissa bits because the exponent adapts per value. Contrast int8: a value of $70$ and a value of $0.4$ must share the *same* fixed step. FP8's per-value exponent is what lets it hold the outlier and the bulk at once.

### Why this beats int8 on activations, quantified

Here is the comparison that makes it click. Suppose your activation range is $[-70, 70]$ because of one outlier, but 99% of values are in $[-1, 1]$.

- **int8** must set its step to $70/127 \approx 0.55$. Every value in $[-1, 1]$ is represented with that 0.55 step — at most 4 distinct levels across the bulk. Catastrophic.
- **E4M3** spends its precision per-binade. A value of $0.4$ sits in the binade $[0.25, 0.5)$ and is represented with 3 mantissa bits, i.e. 8 levels in that binade — a step of about $0.03$, roughly 17× finer than int8's. The outlier at $70$ sits in the binade $[64, 128)$ with a step of about $8$ — coarse, but the *outlier* tolerates coarseness; it is the bulk that needed precision, and the bulk got it.

So FP8 gives you, for free in hardware, much of what SmoothQuant achieves in software: the small values keep their resolution despite the presence of large ones. This is why FP8 activation quantization typically needs *no* SmoothQuant smoothing and *no* outlier path, and still lands under 0.1 perplexity of fp16 — the best activation-quantization quality of any 8-bit scheme. The trade-off is hardware: FP8 tensor cores exist only on Hopper (H100, H200), Ada Lovelace (L40S, RTX 4090), and Blackwell. On an older GPU you simply cannot run FP8 math, and you fall back to W8A8-int + SmoothQuant.

### FP8 in code

vLLM and TensorRT-LLM make FP8 close to a one-liner because the hardware does the heavy lifting.

```bash
# vLLM: dynamic FP8 quantization at load time, no calibration needed for weights.
# Activations are quantized per-token dynamically; weights per-tensor to FP8 E4M3.
vllm serve meta-llama/Llama-2-7b-hf \
    --quantization fp8 \
    --kv-cache-dtype fp8        # FP8 KV-cache too — see the next section

# Or pre-quantize an offline FP8 checkpoint with llm-compressor for static scales:
#   recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC")
```

```python
# TensorRT-LLM / TensorRT Model Optimizer: explicit FP8 PTQ.
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_tensorrt_llm_checkpoint

config = mtq.FP8_DEFAULT_CFG          # E4M3 weights + activations, per-tensor amax
def forward_loop(model):
    for batch in calib_loader:         # ~512 samples to estimate activation amax
        model(batch)

model = mtq.quantize(model, config, forward_loop)
export_tensorrt_llm_checkpoint(model, "llama7b-fp8", dtype="float16",
                               inference_tensor_parallel=1)
# Then: trtllm-build --checkpoint_dir llama7b-fp8 --gemm_plugin fp8
```

The reason FP8 has become the default for serving on Hopper-class hardware is this combination: near-lossless quality, native 2× compute on tensor cores for prefill, half the weight and KV memory of fp16, and almost no tuning. If you have the GPU, it is hard to beat.

#### Worked example: FP8 vs fp16 prefill on an H100

Here is where activation quantization actually pays off, and it is the half that weight-only int4 cannot touch. Take Llama-2-7B serving a RAG workload: a 4,096-token retrieved context per request, and measure **time-to-first-token (TTFT)** — the prefill latency the user feels before any output appears — on an H100 80 GB at batch 1, warmed up, steady-state.

| Scheme | Prefill GEMM math | TTFT @ 4k prompt | Decode tokens/s | Weight + KV mem (4k ctx) |
|---|---|---|---|---|
| fp16 | fp16 tensor cores | ~210 ms | ~70 | 14.0 GB + 2.0 GB |
| W4A16 (AWQ) | fp16 math (weights deq'd) | ~205 ms | ~135 | 3.5 GB + 2.0 GB |
| FP8 E4M3 (+ FP8 KV) | FP8 tensor cores | ~110 ms | ~115 | 7.0 GB + 1.0 GB |

Read the W4A16 row first, because it is the surprising one: 4-bit weights nearly **double** decode throughput (135 vs 70 tokens/s — fewer weight bytes streamed per token) but barely move TTFT (205 vs 210 ms), because the prefill GEMM still runs in fp16. Weight-only quantization did *nothing* for the prefill bottleneck. FP8 is the mirror image: it roughly **halves TTFT** (110 vs 210 ms) by running the prefill GEMM on FP8 tensor cores, while also nearly halving every memory term. That is the entire reason this post exists — **to make prefill fast, you must quantize activations; weights alone cannot do it.** On a chatbot where the user types short prompts, the W4A16 row is fine; on a RAG or document-summarization workload with 4k–32k prompts, the FP8 row is the one your users feel.

## The KV-cache: the memory term that ate your long context

Now the second half of the problem, and the one that dominates at long context and high batch: the **KV-cache**. To make a transformer generate text efficiently, you do not recompute attention over the whole sequence at every step. You cache the **key** and **value** vectors for every token you have already processed, so each new token only computes its own K and V and attends against the stored history. That cache is the KV-cache, and it is the reason generation is fast — but it grows linearly with sequence length and batch size, and at long context it becomes the single largest consumer of GPU memory, often larger than the model weights. (For the full mechanics of why the cache exists and how it is managed with paging, see the dedicated post on [KV-cache optimization and management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management).)

### Sizing it: the formula and a worked number

The KV-cache size, in bytes, for a single sequence is:

$$
\text{KV bytes} = 2 \times L \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times b_{\text{bytes}},
$$

where the leading $2$ is for K *and* V, $L$ is sequence length, and $b_{\text{bytes}}$ is bytes per element (2 for fp16, 1 for int8). Take Llama-2-7B: 32 layers, 32 KV heads, head dimension 128, so $n_{\text{kv\_heads}} \times d_{\text{head}} = 4096$.

$$
\text{KV per token} = 2 \times 32 \times 4096 \times 2\ \text{bytes} = 524{,}288\ \text{bytes} \approx 0.5\ \text{MB/token in fp16}.
$$

At a 32,000-token context, that is $32{,}000 \times 0.5\ \text{MB} \approx 16\ \text{GB}$ — for *one* sequence. The model's int4 weights are about 3.5 GB. So at 32k context the KV-cache is **more than four times the size of the quantized weights**, and it scales with every concurrent user. On a 24 GB consumer GPU, the KV-cache alone bursts your memory budget before you serve a second request. Figure 5 shows how quantizing it claws that memory back layer by layer.

![A vertical stack showing fp16 KV-cache at sixteen gigabytes as the wall at long context, int8 at eight gigabytes for about a tenth of a perplexity cost, int4 at four gigabytes for a few tenths cost, an asymmetric per-token key and per-channel value scheme for best quality per bit, and fixed model weights at the bottom that do not grow with context.](/imgs/blogs/llm-quantization-activations-smoothquant-kv-cache-5.png)

### Quantizing the KV-cache: int8, int4, and the scaling choice

The fix is the same lever: store K and V in int8 or int4 instead of fp16. int8 KV halves the memory; int4 quarters it. But the KV-cache has its own outlier wrinkle, and the recent papers KIVI (Liu et al., 2024) and KVQuant (Hooper et al., 2024) figured out the right scaling axes:

- **Keys have per-channel outliers** — certain channels of the key vectors are consistently large (the same kind of outlier structure as activations). So keys should be quantized **per-channel** (a scale per channel, shared across tokens).
- **Values do not have strong channel structure** but vary token to token, so values should be quantized **per-token** (a scale per token, shared across channels).

KIVI's headline result: 2-bit KV quantization (per-channel K, per-token V) with negligible quality loss, enabling roughly 2.6× larger batch and 2.4–3.5× throughput at long context. KVQuant pushed to 3-bit and even sub-3-bit KV-cache with under 0.1 perplexity degradation by combining per-channel key quantization with handling of a few outlier KV entries in higher precision. The practical sweet spot for most deployments today is **int8 KV** (trivially safe, ~0.1 ppl, 2× memory) or **int4 KV** when you are desperate for context length and can absorb a few tenths of perplexity.

There is a subtle interaction with **rotary position embeddings (RoPE)** that the KV papers had to get right, and it is worth knowing because it explains a non-obvious implementation detail. RoPE rotates the key vectors by an angle that depends on token position, which *mixes* channels — a fixed per-channel outlier structure before RoPE gets smeared across channels after RoPE. KVQuant's important finding was to quantize the keys **before** the rotary embedding is applied, where the per-channel outlier structure is clean and stable, and apply RoPE to the de-quantized keys at attention time. Quantize after RoPE and the per-channel scales no longer line up with the outliers, and quality drops. The lesson generalizes: quantize in the representation where the outliers are *stationary*, then transform.

The asymmetry between keys and values — per-channel for K, per-token for V — is not arbitrary; it falls directly out of the granularity analysis from earlier. Keys carry the per-channel outlier structure (so they need the per-channel axis), while values are comparatively flat across channels but vary token to token (so per-token suffices and is cheaper). Getting this axis assignment wrong is the most common KV-quantization mistake: quantize keys per-token and you hit the same wall as per-tensor activations, because the channel outliers cut across all tokens and a per-token scale cannot isolate them.

The reason this matters so much for the *edge* and for cost: KV quantization is what lets a 24 GB card serve a 32k-context model at batch 8 instead of batch 2, or lets a long-document RAG pipeline fit at all. It directly buys you context length and concurrency, which are exactly the dimensions that explode memory. And because decode is memory-bound — every generated token must read the *entire* KV-cache to attend over the history — a smaller cache is also a *faster* decode: fewer KV bytes to stream per step. This is the rare optimization that improves memory and latency at the same time, which is why it should be near the top of your list for any long-context deployment.

#### Worked example: int8 KV-cache at 32k context

Concrete numbers on Llama-2-7B (W4A16 weights, single sequence) at 32,000 tokens of context, measured-style estimates on an A100-class GPU:

| KV-cache dtype | KV-cache memory | Total memory (W4 + KV) | WikiText-2 ppl | Decode tokens/s | Max batch on 24 GB |
|---|---|---|---|---|---|
| fp16 | 16.0 GB | 19.5 GB | 5.47 (fp16 KV) | ~95 | 1 |
| int8 | 8.0 GB | 11.5 GB | 5.49 (+0.02) | ~110 | 2–3 |
| int4 (per-ch K / per-tok V) | 4.0 GB | 7.5 GB | 5.62 (+0.15) | ~125 | 4–6 |

Read what the int8 row buys: KV memory halved from 16 GB to **8 GB saved**, total footprint down a third, perplexity essentially unchanged (+0.02), and decode is actually *faster* (~110 vs 95 tokens/s) because the cache is half the bytes to stream — remember decode is memory-bound, so fewer KV bytes is fewer bytes moved per step. int4 KV quarters the cache to 4 GB and lets you fit 4–6× the batch on a 24 GB card, for a small but real +0.15 perplexity. **8 GB saved at 32k context for two-hundredths of a perplexity point is one of the best trades in the entire quantization toolbox.**

### Quantizing the KV-cache in practice

In `llama.cpp`, the KV-cache type is a runtime flag — independent of how the *weights* are quantized — so you can run Q4_K_M weights with a Q8_0 KV-cache:

```bash
# llama.cpp: int8 KV-cache (q8_0) with int4 weights.
# K and V cache types are set separately; q8_0 is the safe default, q4_0 for max savings.
./llama-cli -m llama-2-7b.Q4_K_M.gguf \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    -c 32768 \
    -ngl 99 \
    -p "Summarize the attached 30-page contract:"

# Flash-attention is required for quantized V-cache in llama.cpp:
#   add  -fa  (flash attn) when using --cache-type-v below f16
```

In vLLM, FP8 KV-cache is one flag and composes with FP8 or int8 weights:

```bash
# vLLM: FP8 KV-cache (E4M3) — halves KV memory, near-lossless, Hopper/Ada.
vllm serve meta-llama/Llama-2-7b-hf \
    --kv-cache-dtype fp8_e4m3 \
    --max-model-len 32768 \
    --max-num-seqs 16

# int8 KV with calibrated scales (better than dynamic for some models):
#   --kv-cache-dtype fp8  uses dynamic per-tensor scales by default.
```

A practical caution: in `llama.cpp` a quantized *value* cache requires Flash-Attention (`-fa`) to be enabled, because the non-flash attention path does not support a quantized V-cache. And quantizing the K-cache is generally safer than the V-cache; if you see quality drop, keep K at q8_0 and try V at q8_0 before going lower. Asymmetric choices (`--cache-type-k q4_0 --cache-type-v q8_0`) are valid and sometimes the right Pareto point.

### KV quantization composes with grouped-query attention

One more compounding effect worth flagging, because it changes the arithmetic for modern models. Llama-2-70B, Llama-3, Mistral, and most recent models use **grouped-query attention (GQA)** or multi-query attention, where many query heads share a small number of *key/value* heads. The KV-cache size depends on $n_{\text{kv\_heads}}$, not the number of query heads, so GQA already shrinks the cache before you quantize. Llama-3-8B, for example, has 32 query heads but only 8 KV heads — a 4× smaller cache than if every query head had its own KV. Quantization then stacks *on top* of that: an int4 KV-cache on a GQA model is $4 \times 4 = 16\times$ smaller than fp16 multi-head. The levers compose multiplicatively. The general principle, which recurs throughout this series, is that architectural efficiency (fewer KV heads) and numerical efficiency (fewer bits per element) are independent dimensions you can multiply together — and the KV-cache is the place where that multiplication has the biggest payoff at long context.

## The full LLM precision picture

Now we can assemble the four schemes into one decision. Figure 6 lays them out as a matrix: weight bits, prefill compute, quality, and the hardware each needs.

![A four by four matrix comparing W4A16, W8A8 int, FP8 E4M3, and W4A8 across weight bits, prefill compute speedup, quality drop, and the hardware each scheme requires, showing weight-only helps decode while activation quantization helps prefill.](/imgs/blogs/llm-quantization-activations-smoothquant-kv-cache-6.png)

Here is the same picture as a table you can act on, with the KV-cache option folded in:

| Scheme | Weight mem | Prefill (compute) | Decode (memory) | Quality (ppl Δ) | KV-cache | Hardware |
|---|---|---|---|---|---|---|
| **W4A16** (GPTQ/AWQ) | 4× smaller | ~1× (fp16 math) | ~3–4× faster | <0.5 | fp16 or int8 separately | any GPU with int4 kernel |
| **W8A8 int** + SmoothQuant | 2× smaller | ~2× faster | ~2× faster | ~0.1 | int8 | Turing+ (int8 TC) |
| **FP8 E4M3** | 2× smaller | ~2× faster | ~2× faster | <0.1 | FP8 native | Hopper / Ada / Blackwell |
| **W4A8** | 4× smaller | ~2× faster | ~3–4× faster | 0.3–1.0 | int8 | int4×int8 kernel (TRT-LLM) |

The logic of the table:

- **W4A16** is unbeatable for *decode-bound single-stream* inference on *consumer hardware*. The 4-bit weights minimize the bytes streamed per token (the decode bottleneck), and you do not need any special tensor cores because the math is plain fp16. This is what `llama.cpp` on your laptop runs. It does nothing for prefill speed, but a single chatting user spends most time in decode anyway.
- **W8A8 + SmoothQuant** is the workhorse for *prefill-heavy or batch serving* on *any reasonably modern GPU* (Turing and up have int8 tensor cores). You get the ~2× prefill speedup from int8 math and ~2× decode from int8 weights, at the cost of a calibration pass and SmoothQuant tuning. Quality is excellent (~0.1 ppl) once the outliers are migrated.
- **FP8** dominates *if you have Hopper or Ada*. Best activation quality of the 8-bit schemes, native 2× compute, near-zero tuning. The only reason not to use it is the hardware gate.
- **W4A8** is the aggressive option: 4-bit weights (best decode memory) *and* int8 activations (fast prefill). It is the right answer for high-throughput long-context serving where you need both. But it is the hardest to make accurate — mixing int4 weights with int8 activations in one kernel is finicky, quality is more fragile (0.3–1.0 ppl depending on the model and calibration), and you need a kernel that supports the mixed precision (TensorRT-LLM has one). Reach for it last, when W4A16 leaves prefill too slow and W8A8 leaves decode memory too high.

Why is W4A8 so much more fragile than either W4A16 or W8A8 alone? Because the two error sources *compound*. The int4 weights already carry a reconstruction error from being squeezed to 16 levels (GPTQ minimizes it but cannot eliminate it), and the int8 activations carry their own error even after SmoothQuant. In a W4A16 model the activations are exact (fp16), so the only error is the weight error; in a W8A8 model the weights have 256 levels (plenty). W4A8 stacks the coarsest practical weight grid against quantized activations in the *same* matmul, and the errors do not cancel — they accumulate through the network's depth. The mitigation that makes W4A8 viable is finer-grained weight quantization (per-group int4 with group size 64 or 128, not per-channel) plus careful SmoothQuant tuning, and even then you should budget for re-tuning $\alpha$ per model. The honest engineering stance: W4A8 is a power tool for a serving team that has measured a genuine need for both small weights and fast prefill and is willing to invest in per-model calibration. For most teams, W4A16 (decode) or FP8/W8A8 (prefill) is the right answer, and reaching straight for W4A8 is premature optimization that trades quality you can feel for memory you may not need.

Figure 7 reframes the same decision as the prefill-vs-decode contrast that drives it.

![A two column comparison contrasting prefill as a compute-bound big GEMM over all prompt tokens with high arithmetic intensity where W8A8 or FP8 faster math wins, against decode as a memory-bound one-token GEMV that streams weights and KV where W4A16 plus int8 KV moving fewer bytes wins.](/imgs/blogs/llm-quantization-activations-smoothquant-kv-cache-7.png)

### The science of why the bottleneck flips: arithmetic intensity

It is worth one paragraph of rigor on *why* prefill and decode have opposite bottlenecks, because it is the load-bearing fact. Arithmetic intensity is FLOPs divided by bytes moved. For a linear layer with weight matrix $W$ of shape $[h, o]$ processing $T$ tokens:

$$
I = \frac{2 \cdot T \cdot h \cdot o}{(T \cdot h + h \cdot o + T \cdot o) \cdot b_{\text{bytes}}}.
$$

The numerator (the math) scales with $T$. The dominant term in the denominator at small $T$ is $h \cdot o$ — loading the weights — which does *not* scale with $T$. So during **decode** ($T = 1$), $I \approx 2h o / (h o \cdot b) = 2/b_{\text{bytes}}$ — about 1 FLOP/byte for fp16, far below any GPU's ridge point. Decode is memory-bound: you load the whole weight matrix to compute one column of output, doing almost no math per byte. The only lever is fewer weight bytes — quantize W. During **prefill** ($T$ large, say 4,096), the $2Tho$ math term dominates and the weight load $ho$ is amortized across all $T$ tokens, so $I$ climbs above the ridge point and the layer becomes compute-bound. Now the lever is faster math — quantize A so the GEMM runs on int8 or FP8 tensor cores. This is the roofline model applied to the two phases of LLM inference; it is exactly the kind of bottleneck diagnosis [the roofline post](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) teaches, and it is why no single precision scheme is best for both phases.

## When to quantize activations and KV — and when weight-only is enough

This is the decision the whole post has been building toward. The honest answer is that activation and KV quantization are *not free* — they cost calibration effort, hardware constraints, and a real (if small) quality risk — so you should only pay for them when the workload demands it. Figure 8 is the decision tree.

![A decision tree that branches first on whether the workload is decode-bound or prefill-bound, routes decode-bound single streams to W4A16 with int8 KV-cache, and routes prefill-bound batch serving to a hardware check that picks FP8 on Hopper or Ada and otherwise W8A8 with SmoothQuant.](/imgs/blogs/llm-quantization-activations-smoothquant-kv-cache-8.png)

**Weight-only (W4A16) is enough when:**

- You are running **single-stream decode** — a chatbot, an autocomplete, an agent that emits tokens one at a time for one user. Decode is memory-bound; 4-bit weights already give you near the full speedup; activation quantization adds risk for no decode benefit.
- You are on **consumer hardware** without int8/FP8 tensor cores, or you are on CPU (`llama.cpp`, ExecuTorch). Activation quantization needs the integer/FP8 GEMM hardware to pay off; without it you are doing fp16 math anyway, so W4A16 is the natural fit.
- Your context is **short** and batch is **small**, so the KV-cache is a minor memory term and not worth quantizing.
- Quality is paramount and you cannot afford even a 0.1 ppl risk on activations.

**Quantize activations (W8A8 / FP8 / W4A8) when:**

- You are **prefill-heavy**: long prompts, RAG with big retrieved contexts, document summarization, code-completion over large files. Prefill is compute-bound, and only activation quantization speeds up the GEMM. This is where W4A16 leaves performance on the table.
- You are **serving at batch**: many concurrent users. Batching turns even decode into a GEMM (multiple sequences' tokens stacked), pushing arithmetic intensity up so int8/FP8 math helps, and the per-request KV-cache memory makes KV quantization almost mandatory to fit the batch.
- You have **the hardware**: int8 tensor cores (Turing+) for W8A8, or FP8 cores (Hopper/Ada) for FP8. If you have a Hopper GPU, FP8 is close to a free lunch and worth defaulting to.

**Quantize the KV-cache when:**

- Your **context is long** (8k, 32k, 128k) and/or your **batch is large**. The KV-cache scales with both; at 32k it dwarfs the weights. int8 KV is nearly free quality-wise and should be the default at long context. int4 KV when you are memory-desperate and can tolerate a few tenths of perplexity.
- You should *not* quantize the KV-cache aggressively when context is short (the savings are tiny) or when you have already hit a quality floor — keep K at int8 and only push V lower if you must.

### A stress test: what breaks, and what to check

Pose the failure modes explicitly, because they are where engineers get paged.

- **Naive W8A8 with no SmoothQuant on an outlier-heavy model** → perplexity explodes (the OPT-13B 14,000-ppl row above). Fix: SmoothQuant, or fall back to LLM.int8() / FP8. Always evaluate perplexity before shipping; a per-tensor int8 model can look fine on a smoke test and be broken on the benchmark.
- **W4A8 quality drop** → the int4 weights and int8 activations interact badly on a particular model. Fix: raise the calibration set size, try per-group weight scales, or step back to W8A8 if the prefill win is not worth the quality.
- **FP8 on the wrong GPU** → it silently falls back to a slow emulated path (or errors). Check that your GPU has FP8 tensor cores (compute capability 8.9 / 9.0+) before committing to FP8.
- **Quantized V-cache without Flash-Attention in llama.cpp** → it refuses to run. Fix: add `-fa`.
- **Calibration mismatch** → you calibrated activation scales on English prose and serve code; the activation distribution shifts and outliers move. Fix: calibrate on data representative of production traffic, and prefer *dynamic* per-token activation scaling (computed at runtime) over static scales when the distribution is unpredictable.
- **The model is memory-bound and you quantized activations** → you added calibration cost and quality risk and got no speedup, because the bottleneck was bytes, not math. Fix: profile first (roofline), then choose the lever that matches the bottleneck.

That last one is the meta-lesson of the entire series: **measure the bottleneck before you pick the lever.** Quantizing activations on a decode-bound workload is the LLM-inference equivalent of the week I spent tuning a kernel that was never compute-bound.

## Case studies: real numbers from the literature

A few grounded results, with sources, so these are not just my estimates.

**SmoothQuant on OPT-175B and BLOOM-176B (Xiao et al., 2022).** The headline of the paper: W8A8 with SmoothQuant preserved accuracy across OPT, BLOOM, GLM, and LLaMA families where naive W8A8 collapsed, while delivering up to **1.56× speedup and 2× memory reduction** at inference. For the largest models, SmoothQuant enabled serving OPT-175B on a single 8-GPU node where fp16 needed more. The key contribution was making W8A8 *accurate* without a separate outlier path, so it runs on standard int8 kernels.

**LLM.int8() on OPT and BLOOM up to 175B (Dettmers et al., 2022).** Zero-degradation 8-bit inference. The paper's experiments showed that emergent outlier features appear sharply around 6.7B parameters and that the mixed int8/fp16 decomposition recovers full fp16 accuracy where vanilla int8 fails — enabling, for the first time, running a 175B model on a single server's consumer-grade GPUs in 8-bit. This is the result that put 8-bit LLM inference into `bitsandbytes` and `transformers` for everyone.

**FP8 for inference (Micikevicius et al., 2022, and NVIDIA TensorRT-LLM reports).** The FP8 formats paper demonstrated E4M3/E5M2 matching fp16 accuracy across a range of models for both training and inference. In production, NVIDIA's TensorRT-LLM reports FP8 on Hopper delivering roughly **2× throughput** over fp16 for LLM serving with quality within noise, which is why FP8 became the default serving precision on H100-class hardware.

**KIVI and KVQuant for KV-cache (Liu et al., 2024; Hooper et al., 2024).** KIVI showed **2-bit** KV-cache (per-channel keys, per-token values) with negligible accuracy loss, enabling up to **2.6× larger batch sizes and 2.4–3.5× throughput** at long context. KVQuant achieved **sub-4-bit** KV-cache (down to ~3 bits) with under 0.1 perplexity degradation on LLaMA models, by quantizing keys per-channel before the rotary embedding and isolating a small fraction of outlier KV entries. Together they established the per-channel-K / per-token-V recipe that vLLM and TensorRT-LLM now ship.

**The composite, real-world recipe.** A common production stack on Hopper today is FP8 weights and activations *plus* an FP8 KV-cache — roughly halving every memory term and doubling prefill throughput at near-zero quality cost, with one or two flags in vLLM. On older hardware the equivalent is W8A8 + SmoothQuant + int8 KV-cache. On a laptop, it is W4A16 GGUF weights + q8_0 KV-cache in `llama.cpp`. Three different points on the Pareto frontier, each matched to its hardware.

**A cost lens, because someone always asks.** Quantization is, ultimately, a way to serve more tokens per dollar of hardware. If FP8 doubles prefill throughput and halves KV memory, a single H100 that served 100 requests/second at fp16 might serve closer to 180–200, and the KV savings let you raise the batch and context without buying a second card. At cloud GPU rates, that is the difference between \$0.50 and \$0.27 per million output tokens on the same machine — a near-halving of marginal serving cost for one configuration flag, with quality within measurement noise. That economics is why every serious LLM serving stack now ships activation and KV quantization on by default for supported hardware; leaving it off is leaving roughly half your throughput on the table. The one place the math inverts is the single-stream consumer case: if you serve one user on one consumer GPU, the throughput you are not using has no dollar value, and the only thing that matters is fitting the model and keeping decode snappy — which is exactly the W4A16 regime where you should *not* pay the activation-quantization tax.

## How to measure this honestly

Numbers like "2× faster" mean nothing without the measurement protocol, so here is how to get trustworthy ones.

- **Separate prefill and decode.** Report time-to-first-token (prefill latency) and inter-token latency (decode) as *distinct* metrics. A scheme that helps prefill and not decode will look mediocre if you only report end-to-end latency at short output length, and great at long prompt length. Measure both phases.
- **Warm up.** The first few iterations pay for kernel autotuning, CUDA graph capture, and cache population. Discard them; measure steady state.
- **Fix the batch and context.** Throughput is meaningless without the batch size and sequence length. Report tokens/s *at* a stated batch and context, because that is exactly where activation and KV quantization change the answer.
- **Watch for thermal throttling** on edge devices and laptops — run long enough to reach steady-state clocks, and report whether the chip throttled.
- **Evaluate perplexity on a held-out set** (WikiText-2 is the standard, plus a task benchmark like MMLU) *before* shipping any activation-quantized model. A broken int8 model can pass a one-line generation smoke test and still be at 14,000 perplexity. Perplexity is your canary.
- **Measure peak memory**, not just average — the KV-cache peaks at max context, and that peak is what OOMs you. Report peak GPU memory at the worst-case context and batch you intend to serve.

For the broader set of edge metrics — energy, p99, model size on disk, peak SRAM — see [the metrics that actually matter on-device](/blog/machine-learning/edge-ai/the-metrics-that-actually-matter-on-device); the same discipline applies here.

## Key takeaways

- **Weight bits buy decode; activation bits buy prefill.** Decode is memory-bound (stream weights one token at a time) so W4A16 wins; prefill is compute-bound (big GEMM over all tokens) so quantizing activations to int8/FP8 wins. Match the lever to the bottleneck.
- **Activation outliers are the whole difficulty.** A few channels with magnitudes tens of times larger than the rest poison a naive per-tensor int8 scale and can blow perplexity into the thousands. You cannot clip them — they are load-bearing.
- **LLM.int8()** keeps the rare outlier columns in fp16 and runs the rest in int8 — exact accuracy, great for memory, but the mixed-precision overhead means it does not speed up compute-bound prefill.
- **SmoothQuant** migrates the difficulty from activations to weights via a per-channel factor $s_j = (\max|X_j|)^\alpha / (\max|W_j|)^{1-\alpha}$, with $\alpha = 0.5$ equalizing the two ranges at their geometric mean. It folds into LayerNorm offline, so the served model is plain fast int8.
- **FP8 (E4M3)** tolerates outliers in hardware because its log-spaced levels stay fine near zero while still representing large values. Best 8-bit activation quality, native 2× compute — but only on Hopper / Ada / Blackwell.
- **The KV-cache dominates memory at long context** — at 32k tokens it can be 4× the size of int4 weights. int8 KV is near-free (~0.02 ppl) and halves it; int4 KV quarters it for a few tenths. Quantize keys per-channel and values per-token (KIVI/KVQuant).
- **Pick the scheme by workload then hardware:** single-stream decode on consumer HW → W4A16 (+ int8 KV at long context); prefill-heavy or batch serving → FP8 if you have Hopper/Ada, else W8A8 + SmoothQuant; need both tiny weights and fast prefill → W4A8, last, when the others fall short.
- **Measure before you choose.** Profile the bottleneck (roofline), report prefill and decode separately, and always check perplexity before shipping an activation-quantized model — a broken one passes smoke tests.

## When this composes with the other levers

Quantization is one of four levers, and activation/KV quantization composes cleanly with the others. You can distill a smaller student and then run it W8A8. You can prune a model and quantize the survivor. The KV-cache lever in particular stacks with architectural choices like grouped-query attention (fewer KV heads → smaller cache before you even quantize) and with paging (vLLM's PagedAttention manages the quantized cache in blocks).

If you want a concrete sequencing for an LLM serving deployment, this is the order I follow. First, profile the workload to learn whether it is decode-bound or prefill-bound and how long the contexts are — this single measurement determines everything downstream. Second, quantize the weights (W4A16 with GPTQ or AWQ, the previous post's lever) because it is the lowest-risk, highest-reward step and it helps the decode path almost every workload spends time in. Third, if the workload is prefill-heavy or batched and you have the hardware, add activation quantization — FP8 on Hopper/Ada, else W8A8 with SmoothQuant — to speed the compute-bound GEMM. Fourth, if contexts are long or batches are large, quantize the KV-cache (int8 first, int4 if you are memory-starved), because that is where the memory and the long-context latency actually live. At each step, re-measure perplexity and the prefill/decode latencies; stop as soon as you hit your target, because every additional bit of quantization is quality you are spending. Do not jump straight to W4A8 + int4 KV "to be safe" — that is the configuration most likely to lose you accuracy you cannot get back without re-tuning.

For how all four levers sit on the accuracy–efficiency Pareto frontier and how to sequence them across techniques, not just within quantization, see [the taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression), and for the end-to-end decision flow that ties quantization, pruning, distillation, and architecture together on a real deployment, the capstone [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) walks the whole pipeline.

## Further reading

- Dettmers, Lewis, Belkada, Zettlemoyer, *"LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale"* (2022) — the outlier-feature analysis and mixed-precision decomposition; the basis of `bitsandbytes` 8-bit.
- Xiao, Lin, Seznec, Wu, Demouth, Han, *"SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models"* (2022) — the per-channel migration factor and the $\alpha$ balance.
- Micikevicius et al., *"FP8 Formats for Deep Learning"* (2022) — the E4M3 / E5M2 definitions and the case for floating 8-bit.
- Liu et al., *"KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache"* (2024) — per-channel keys, per-token values.
- Hooper et al., *"KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization"* (2024) — sub-4-bit KV with outlier isolation.
- Official docs: `llama.cpp` quantization and `--cache-type-k/v` flags; vLLM quantization and `--kv-cache-dtype`; NVIDIA TensorRT-LLM and TensorRT Model Optimizer (FP8/int8 PTQ); `llm-compressor` recipes.
- Within this series: [LLM quantization I: weight-only with GPTQ and AWQ](/blog/machine-learning/edge-ai/llm-quantization-weight-only-gptq-awq), [the roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives), [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression), [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook). Out of series: [KV-cache optimization and management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management).

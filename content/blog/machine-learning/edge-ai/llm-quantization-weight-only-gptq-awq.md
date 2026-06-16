---
title: "LLM quantization I: weight-only int4 with GPTQ, AWQ, and GGUF k-quants"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Why weight-only int4 is the single technique that put 7B-class LLMs on phones and laptops, and how to actually do it with GPTQ, AWQ, and llama.cpp k-quants without losing your accuracy."
tags:
  [
    "edge-ai",
    "model-optimization",
    "quantization",
    "gptq",
    "awq",
    "gguf",
    "llama-cpp",
    "inference",
    "efficient-ml",
    "llm",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/llm-quantization-weight-only-gptq-awq-1.png"
---

A Llama-2 7B checkpoint in fp16 is about 14 GB of weights. That number alone ends a lot of conversations. It will not fit in the 8 GB of unified memory on a base MacBook Air. It will not fit on a phone. It will not even fit on a single consumer 8 GB or 12 GB GPU once you add the KV cache and the runtime's own overhead. For years that single fact was the wall between "LLMs are amazing" and "LLMs are something I can run myself, offline, on a device I already own."

Weight-only int4 quantization is the technique that tore that wall down. Store each weight in roughly 4 bits instead of 16, and the same 7B model drops to about 3.5 to 4 GB. Now it fits on the laptop. Now it fits on the phone. And here is the part people find surprising the first time: it usually runs *faster*, not just smaller, because single-token LLM decoding is **memory-bound** — the hardware spends almost all of its time reading weights from memory, not doing arithmetic — and you just cut the bytes it has to read by roughly 4×. The figure below is the whole pitch in one picture: the same model, two precisions, one of which fits the device.

![Side by side comparison of a 7B model at fp16 needing about 14 GB and not fitting consumer devices versus int4 at under 4 GB fitting a phone or laptop](/imgs/blogs/llm-quantization-weight-only-gptq-awq-1.png)

This post is the first of two on LLM quantization in the "Optimizing AI Models for the Edge" series. Here we cover **weight-only** quantization: we keep activations in fp16 and squeeze only the stored weights down to 4 bits. That choice is not arbitrary — it is the single most effective thing you can do for memory-bound decode, and it sidesteps the nastiest part of LLM numerics, the activation outlier problem, which we save for the [activations and KV-cache post](/blog/machine-learning/edge-ai/llm-quantization-activations-smoothquant-kv-cache). The second post handles activation quantization, SmoothQuant, and KV-cache quantization, which is where the harder physics and the compute-bound prefill story live.

A word on scope and honesty before we dive in. Every number in this post is either derived from first principles (the byte counts, the variance and SQNR laws, the bits-per-weight arithmetic) or quoted as approximate from the published literature and widely-replicated community benchmarks (the perplexity deltas, the tokens/s figures). Quantization results are notoriously hardware- and runtime-dependent — the same Q4_K_M file runs at different speeds on different Macs, different `llama.cpp` builds, and different thermal states — so treat every absolute tokens/s as an order-of-magnitude guide and the *ratios* (int4 vs fp16, GPTQ vs AWQ) as the durable takeaway. The one thing you should never do is trust a quantization result you did not measure on your own target. The point of this post is to give you the reasoning to know *what* to measure and *why* the numbers come out the way they do.

By the end of this post you will be able to: explain *why* weight-only int4 wins for LLMs from the roofline up; reason about the outlier problem and group-wise quantization; describe how GPTQ, AWQ, and GGUF k-quants each round 16-bit weights to 4 bits while protecting accuracy; run all three toolchains with real commands; and read a bits-versus-perplexity-versus-speed table to pick the right one for your hardware. This post sits on the **quantization lever** of the four-lever frame from [the model-compression taxonomy](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression), and it feeds directly into [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) at the end of the series.

## 1. Why weight-only quantization dominates for LLMs

Start with the physics, because the physics is what makes weight-only int4 a near-free win rather than a painful trade-off.

A transformer doing autoregressive generation runs in two regimes. The **prefill** phase processes your prompt: every token in the prompt is computed in parallel, so the matrix multiplies are large (many rows), the hardware is busy with arithmetic, and the workload is **compute-bound**. The **decode** phase generates one new token at a time: each step multiplies a single new token's hidden vector against every weight matrix in the model. The matrices are huge; the input is one vector. That is a matrix-times-vector operation, and it has terrible arithmetic intensity.

Arithmetic intensity is the number of floating-point operations you do per byte of data you read from memory. We derived the full version of this in [the roofline post](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives), but the one-line summary is: if your arithmetic intensity is below the hardware's ratio of peak FLOP/s to peak memory bandwidth (its "ridge point"), you are memory-bound — you finish the reads slower than the math, so the math units sit idle waiting on memory.

For a single matrix-vector product $y = Wx$ where $W$ is $m \times n$, you do about $2mn$ floating-point operations and you read about $mn$ weight elements. If each weight is 2 bytes (fp16), you read $2mn$ bytes. The arithmetic intensity is

$$
I = \frac{2mn \text{ FLOP}}{2mn \text{ bytes}} = 1 \ \frac{\text{FLOP}}{\text{byte}}.
$$

One FLOP per byte. A modern accelerator's ridge point is often 50 to 200+ FLOP/byte (an A100 is around 100; an M-series Mac's GPU is lower but still tens). So decode is *deeply* memory-bound — by one to two orders of magnitude. The math units are starved; the bottleneck is purely how fast you can stream weights out of memory.

It is worth pausing on *why* the arithmetic intensity is stuck at 1 and cannot be rescued by a better kernel. The number $2mn$ FLOP / $2mn$ bytes is structural: you visit each of the $mn$ weights exactly once and do exactly two FLOPs with it (a multiply and an add into the accumulator). There is no reuse. Compare this to a square matrix-matrix product $C = AB$ where $A$ and $B$ are both $n \times n$: that does $2n^3$ FLOPs over $\sim 2n^2$ bytes of input, an arithmetic intensity of $\sim n$ — every input element is reused $n$ times across the output. Reuse is what feeds the math units. Matrix-vector decode has no reuse to give. This is not a software problem you can tile your way out of; it is the shape of the computation. The only lever that moves the needle is the *bytes* term, and that is exactly the term quantization attacks.

The batch size is the one thing that changes this picture, and understanding it tells you precisely when weight-only int4 helps and when it stops helping. If you decode $B$ sequences at once (batched serving), the single weight read is amortized across $B$ tokens: you read the weight matrix once and do $B$ matrix-vector products with it, so the arithmetic intensity rises to roughly $B$ FLOP/byte. Push $B$ high enough — past the hardware's ridge point of ~100 — and decode becomes compute-bound, and weight-only quantization (which keeps the math in fp16) stops giving you a speedup because you are no longer waiting on memory. This is the crucial caveat for the on-device case: edge inference is almost always **batch size one**, deeply memory-bound, exactly the regime where weight-only int4 is a near-free speedup. Datacenter serving at large batch is a different animal where you want low-precision *math*, not just low-precision storage. We are firmly in the batch-one world here.

This is the key insight that makes weight-only quantization special for LLMs. If decode time is dominated by reading weight bytes, then **reading fewer weight bytes makes decode proportionally faster.** Cut from fp16 (2 bytes) to int4 (0.5 byte) and you read roughly 4× fewer weight bytes per token. In practice you do not get the full 4× — there is dequantization overhead, the KV cache still reads in fp16, the runtime is imperfect — but a 2 to 3× decode speedup on memory-bound hardware is routine. The figure below shows the contrast directly: same model, same math, but far fewer bytes crossing the memory bus.

![Before and after view showing fp16 decode reading 14 GB per token and bandwidth-limited versus int4 decode reading about 4 GB per token at two to three times the tokens per second](/imgs/blogs/llm-quantization-weight-only-gptq-awq-2.png)

Contrast this with quantizing **activations**. Activations are tiny during decode — a single token's hidden state is a few thousand floats, kilobytes, not gigabytes. Quantizing them saves almost no memory traffic in the decode-dominated regime, so it buys you almost no decode speedup. It only helps when you can also turn the matmuls into faster low-precision *math* (int8 tensor-core kernels), which is a compute-bound win that matters for prefill and large batches. And activation quantization is where the dragons live: activations contain extreme **outliers** that wreck naive low-precision rounding. So for the on-device, batch-size-one, decode-heavy case — the case that matters for the edge — weight-only is the high-leverage move, and we keep activations in fp16 to stay safe.

#### Worked example: where does decode time go?

Take a 7B model, fp16, on a laptop GPU with roughly 200 GB/s of usable memory bandwidth (a respectable integrated or low-end discrete GPU). Weights are about 13 to 14 GB. Reading 13 GB at 200 GB/s takes about 65 ms. That is your *floor* for one token — you cannot generate a token faster than you can read the weights once. 65 ms/token is about 15 tokens/s, and that lines up with what people actually see. Now quantize weights to int4: about 3.5 GB to read, 17.5 ms floor, roughly 57 tokens/s if everything else scaled perfectly. Reality lands between 40 and 60 tokens/s because of dequant and KV-cache overhead, but the direction and the rough magnitude come straight from the byte count. You did not change a single FLOP. You changed how many bytes you read.

That is the entire reason weight-only int4 became the default way to run LLMs on consumer hardware. It is not a clever accuracy trick that happens to also save memory. It is a memory-bandwidth trick that happens to keep accuracy.

## 2. The science: what breaks when you naively quantize

Quantizing a weight tensor means mapping its fp16 values onto a small grid of integers. For symmetric int4 you have 16 levels, conventionally $-8$ to $7$. The standard affine (asymmetric) scheme stores a scale $s$ and a zero-point $z$ and rounds:

$$
q = \text{clamp}\left(\text{round}\left(\frac{w}{s}\right) + z,\ 0,\ 2^b - 1\right), \qquad \hat{w} = s\,(q - z).
$$

Here $b$ is the bit-width (4), $w$ is the original weight, $q$ is the stored integer, and $\hat{w}$ is the value you recover at inference time. The scale $s$ stretches the integer grid to cover the weight range; the zero-point $z$ shifts it so that real zero maps exactly to an integer (important so that padding and ReLU-style structure stay exact). For symmetric quantization $z = 0$ and $s = \max|w| / (2^{b-1} - 1)$.

The error of this rounding, $\hat{w} - w$, is what you have to control. If the weights within a quantization group are roughly uniform over $[-r, r]$ and you use $2^b$ levels, the step size is $\Delta = 2r / 2^b$ and the rounding error is approximately uniform on $[-\Delta/2, \Delta/2]$, with variance

$$
\sigma_q^2 = \frac{\Delta^2}{12} = \frac{r^2}{3 \cdot 2^{2b}}.
$$

That $2^{2b}$ in the denominator is the good news: every extra bit quarters the error variance — the classic 6.02 dB per bit of signal-to-quantization-noise. To make that precise, the signal-to-quantization-noise ratio for a signal of power $\sigma_x^2$ against this uniform noise floor is

$$
\text{SQNR} = 10 \log_{10}\frac{\sigma_x^2}{\sigma_q^2} \approx 6.02\,b + \text{const} \ \ (\text{dB}).
$$

Each bit you add buys about 6 dB of headroom against rounding noise. Going from int8 (48 dB) to int4 (24 dB) throws away 24 dB — a 250× increase in noise power. That sounds catastrophic, and for a *random* signal it would be. The reason int4 LLMs work anyway is that the "signal" — the layer's output — is robust to a lot of that noise because of redundancy (we return to this in §6), and because group-wise scaling and the GPTQ/AWQ tricks keep the *effective* range $r$ tight so the constant term stays favorable. But the 6 dB/bit law is why you cannot keep dropping bits forever: by int2 you have given back 36 dB versus int8, and no amount of cleverness fully recovers it.

But notice the $r^2$ in the numerator of the variance. The error scales with the *square of the range* you have to cover. And that is where the trouble starts. The whole game of good quantization is keeping $r$ as small as possible for the weights you actually care about, which is precisely what group-wise quantization, GPTQ's error pushing, and AWQ's scaling each do in a different way.

One more mechanical point before we hit the outlier problem, because it explains why weight-only int4 is "free" at inference rather than requiring exotic hardware. At runtime, the int4 weight $q$ is read from memory (this is the cheap part — few bytes), then **dequantized** back to fp16 as $\hat{w} = s\,(q - z)$, and the matmul runs in fp16 as usual. So the *math* is unchanged; only the *bytes read* changed. This is why weight-only int4 needs no special low-precision matmul unit — any GPU or CPU that does fp16 matmul can run it, the kernel just unpacks 4-bit values on the way in. The dequant adds a little arithmetic, but on a memory-bound workload there were idle math units to spare, so it is nearly free. (This also makes clear why weight-only int4 does *not* speed up a compute-bound workload: you have added dequant FLOPs to an already FLOP-limited kernel.)

### The outlier problem

LLM weight matrices are not uniform. A handful of weights, in a handful of channels, are far larger in magnitude than the rest. When you compute one global scale for an entire matrix (per-tensor quantization), that scale is set by the single largest weight — the outlier. The step size $\Delta$ becomes huge to cover that range, so the *typical* weight, which is small, gets crushed into just two or three of the 16 levels. You have spent your precision budget representing a few giant values and starved everything else.

It gets worse on the activation side, which is *why* we keep activations in fp16 here. As LLMs scale past roughly 6.7B parameters, specific activation feature dimensions develop systematic outliers up to 20× the typical magnitude (this is the finding behind LLM.int8() from Dettmers et al., 2022). Those activation outliers carry real signal — zeroing them collapses the model — so you cannot just clip them. Quantizing weights only, and leaving activations in fp16, lets us dodge that entire failure mode. We will face it head-on with SmoothQuant in [the activations post](/blog/machine-learning/edge-ai/llm-quantization-activations-smoothquant-kv-cache).

### The fix you reach for first: group-wise quantization

Per-tensor quantization is hostage to the single worst weight in millions. Per-channel (one scale per output channel) is much better and is the floor for any serious int8 scheme. But for int4 the winning move is **group-wise** (also called per-group or block) quantization: split each row into fixed-size groups — typically 64 or 128 consecutive weights — and give each group its own scale (and zero-point). Now one outlier only inflates the range of *its* 128-weight group, not the whole matrix. Every other group keeps a tight range and uses its full 16-level grid. The figure below shows the layout: a row chopped into groups, each carrying its own fp16 scale.

![Grid showing one weight row split into four groups of 128 weights with each group carrying its own fp16 scale below it](/imgs/blogs/llm-quantization-weight-only-gptq-awq-3.png)

The cost is the metadata. With group size $g$ and bit-width $b$, plus an fp16 scale (16 bits) and, for asymmetric, an fp16 or low-bit zero-point per group, the effective bits per weight is

$$
\text{bpw} = b + \frac{16 + 16}{g}.
$$

For $b = 4$, $g = 128$: $\text{bpw} = 4 + 32/128 = 4.25$ bits/weight. For $g = 64$: $4 + 32/64 = 4.5$ bits/weight — slightly bigger, slightly more accurate. For $g = 32$: $5.0$ bits/weight. This is the **group-size trade-off**: smaller groups mean tighter ranges and better accuracy, but more scales and a bigger file. Group size 128 is the near-universal default because it sits at the knee of the curve — almost all the accuracy of $g{=}64$ at noticeably less overhead. Production GGUF k-quants get clever and quantize the per-group scales themselves to 6 bits to claw some of that overhead back, which we will see in §6.

#### Worked example: how much does group-wise actually help?

Suppose a 4096-wide weight row has 4095 weights in $[-0.1, 0.1]$ and one outlier at $1.0$. Per-tensor int4: $r = 1.0$, step $\Delta = 2 \cdot 1.0 / 16 = 0.125$. The typical weights live in $[-0.1, 0.1]$, a span of $0.2$, which is only $0.2 / 0.125 \approx 1.6$ steps — they collapse onto essentially two levels. Catastrophic. Now group-wise with $g = 128$: the outlier sits in exactly one group, where $r = 1.0$ and that group is coarse, but the other 31 groups have $r \approx 0.1$, step $\Delta = 0.0125$, and resolve the typical weights to about 16 distinct levels — full precision use. One outlier, isolated. That is the whole reason group-wise quantization is the foundation under GPTQ, AWQ, and k-quants alike.

## 3. GPTQ: minimize the layer's output error, not the weight error

Round-to-nearest with group-wise scales is a strong baseline. But it has a blind spot: it minimizes the error on each *weight* in isolation, when what you actually care about is the error on each *layer's output*. Two weights with the same rounding error do not hurt equally — one might feed a dimension the next layer ignores, the other a dimension it leans on heavily. GPTQ (Frantar, Ashkboos, Hoefler, Alistarh, 2022) is the method that quantizes weights to minimize the output error of the layer, and it is the workhorse behind a huge fraction of int4 LLMs.

### The objective

GPTQ quantizes one linear layer at a time. For a layer with weight matrix $W$ and a set of calibration inputs $X$ (a few hundred sequences run through the model), the goal is to find a quantized $\hat{W}$ that keeps the layer's output as close as possible to the original:

$$
\hat{W} = \arg\min_{\hat{W}}\ \big\| WX - \hat{W}X \big\|_2^2 .
$$

This is a per-layer reconstruction objective. It is the same objective as the classic Optimal Brain Quantization (OBQ) framework, which itself descends from Optimal Brain Surgeon: when you perturb a weight (here, by rounding it), you can compensate by adjusting the *other, still-free* weights to absorb the damage, using second-order curvature information.

### The OBQ / GPTQ update rule

The curvature of that quadratic objective with respect to the weights is the **Hessian**, and for this least-squares problem it has a clean closed form:

$$
H = 2 X X^\top.
$$

$H$ tells you how sensitive the layer's output is to each weight and to *correlations* between weights. OBQ's insight: when you quantize weight $w_q$ (column $q$), the optimal compensating update to all the not-yet-quantized weights, and the resulting increase in error, are both expressible through $H^{-1}$. Quantizing one weight and applying the optimal correction to the rest costs an error of

$$
\delta E_q = \frac{\big(w_q - \text{quant}(w_q)\big)^2}{[H^{-1}]_{qq}},
$$

and the correction applied to the remaining free weights is

$$
\delta W = -\frac{w_q - \text{quant}(w_q)}{[H^{-1}]_{qq}}\, H^{-1}_{:,\,q}.
$$

Read that second equation slowly, because it is the whole idea. You round weight $q$. That introduces an error $w_q - \text{quant}(w_q)$. Instead of accepting the damage, you *push* it into the weights you have not quantized yet, in proportion to how those weights are correlated with $q$ through $H^{-1}$. The weights you have already frozen are untouched; the free weights soak up the error so the layer's output barely moves.

GPTQ makes this scalable. OBQ greedily picks the cheapest weight to quantize next, which is $O(n^3)$ per row and too slow for billion-parameter matrices. GPTQ's key simplification: the *order* of quantization within a row barely matters, so just go **column by column, left to right**, in a fixed order, for the whole matrix at once. It quantizes column $j$, applies the Hessian-weighted correction to columns $j+1 \dots n$, then moves to column $j+1$. It batches the $H^{-1}$ math (one Cholesky factorization of $H$ up front), and processes columns in blocks of 128 for cache efficiency. The result runs a 175B model through quantization in a few GPU-hours. The figure below traces that column sweep: round a column, correct the rest, repeat.

![Timeline of the GPTQ algorithm building a Hessian then rounding each weight column to int4 and correcting the remaining columns until output error is minimized](/imgs/blogs/llm-quantization-weight-only-gptq-awq-4.png)

### Why it works, intuitively

Round-to-nearest treats every weight as independent. GPTQ treats them as a coupled system: it knows, from the calibration statistics in $H$, that some directions in weight space matter far more than others, and it spends its error budget where the layer's output can absorb it. That is why GPTQ at int4 often loses *less* than half the perplexity that naive round-to-nearest does, at the same bit-width. The price is the calibration pass — you need a few hundred representative sequences and a GPU-hour or several — and the result is specific to the data distribution you calibrated on.

A concrete way to feel the difference: take two weights in the same layer that the calibration data shows are strongly anti-correlated through $H^{-1}$ — when one is too big, the layer's output is fine as long as the other shrinks to match. Round-to-nearest, blind to this, might round both up, and the two errors *add* in the output. GPTQ rounds the first, sees the error, and when it reaches the second it nudges it down by exactly the amount that cancels the first error in the output. The weights individually are now *further* from their fp16 values than round-to-nearest would have left them — GPTQ deliberately accepts larger per-weight error — but the *layer output* is closer, which is the only thing that matters. That figure of two weights soaking up each other's error is the entire intuition behind why a second-order method beats a naive one at the same bit count.

There is a subtlety in $H = 2XX^\top$ worth flagging because it explains the `damp_percent` flag you will set in practice. $H$ can be near-singular — some input directions barely appear in the calibration data, so $H$ has tiny eigenvalues, and $H^{-1}$ blows them up. The correction $\delta W \propto H^{-1}_{:,q}$ then explodes and the quantization NaNs out or produces garbage. The fix is a ridge term: replace $H$ with $H + \lambda I$ where $\lambda$ is a small fraction of the mean diagonal (that is what `damp_percent` controls). This is the standard Tikhonov regularization move — it bounds the inverse and trades a hair of optimality for numerical stability. If GPTQ fails to converge on a model, raising the damping is the first thing to try.

### Running GPTQ

The modern path is the `gptqmodel` library (the maintained successor to `auto-gptq`), or `optimum`'s GPTQ integration which wraps it. Here is a real quantization run:

```python
# pip install gptqmodel transformers
from gptqmodel import GPTQModel, QuantizeConfig
from datasets import load_dataset

model_id = "meta-llama/Llama-2-7b-hf"

# 4-bit, group size 128, with activation-order column sorting on.
quant_config = QuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=True,        # quantize columns in descending Hessian order
    damp_percent=0.01,    # ridge term added to H diagonal for stability
)

# A few hundred calibration samples is plenty; more rarely helps.
calib = load_dataset("allenai/c4", "en", split="train", streaming=True)
calib_texts = [next(iter(calib))["text"] for _ in range(256)]

model = GPTQModel.load(model_id, quant_config)
model.quantize(calib_texts)            # runs the column-by-column GPTQ sweep
model.save("Llama-2-7b-gptq-int4-g128")
```

A few flags actually matter here. `group_size=128` is the accuracy/size knee we discussed. `desc_act=True` (activation order) quantizes the most Hessian-sensitive columns first, which improves accuracy at a small speed cost during inference; it is worth it on small models, sometimes skipped on large ones for kernel speed. `damp_percent` adds a tiny ridge to the Hessian diagonal so the Cholesky stays numerically stable — if quantization NaNs out, raise it. The calibration set should look like your real inputs; C4 or WikiText is fine for general models, but if you serve code or a specific language, calibrate on that.

Loading the result for inference is a one-liner in transformers, and vLLM and TGI read GPTQ checkpoints natively for serving:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Llama-2-7b-gptq-int4-g128",
    device_map="cuda",      # uses the fused int4 GPTQ kernels
)
tok = AutoTokenizer.from_pretrained("Llama-2-7b-gptq-int4-g128")
out = model.generate(**tok("The roofline model says", return_tensors="pt").to("cuda"))
print(tok.decode(out[0]))
```

## 4. AWQ: protect the salient weight channels

GPTQ asks "how do I round all the weights to minimize output error?" AWQ — Activation-aware Weight Quantization (Lin et al., 2023) — asks a sharper question: "which weights actually matter, and how do I protect *just those*?" Its answer is elegant and a little counterintuitive, and it produces models that are both accurate and fast to run because the rounding stays plain round-to-nearest in the kernel.

### The salient-channel observation

AWQ's founding observation: not all weights are equally important, and the important ones are revealed by the *activations*, not the weights. About 1% of weight channels — the channels that get multiplied by large-magnitude activation features — dominate the layer's output. If you could keep just those ~1% of channels in fp16 and quantize the rest, you would recover almost all of the accuracy. The AWQ paper shows exactly this: keeping 1% of channels (chosen by activation magnitude) in fp16 nearly closes the gap to the full-precision model.

But mixed-precision storage — fp16 for some channels, int4 for others, in the same matrix — is a kernel nightmare. Hardware hates ragged layouts. So AWQ asks: can we get the *protection* of keeping salient channels in fp16, while keeping the *uniform* int4 layout the hardware loves? Yes — through scaling.

### The scaling derivation

Here is the math, and it is genuinely clever. Consider one weight $w$ multiplied by its input activation $x$, so the output contribution is $wx$. The quantization error on this term, from rounding $w$, is roughly the step size $\Delta$ scaled by $x$:

$$
\text{Err}(wx) \approx \Delta \cdot |x|, \qquad \Delta = \frac{\max|w|}{2^{b-1}}.
$$

Now scale that channel's weight up by a factor $s > 1$ before quantizing, and scale the activation down by $s$ to compensate, so the product $wx = (sw)(x/s)$ is unchanged. Quantizing the scaled weight $sw$ gives a new step size $\Delta'$. The crucial fact: if you scale up only *one* salient channel within a group, the group's overall max barely moves (it was set by other channels), so $\Delta' \approx \Delta$. The new error on the term is

$$
\text{Err}'((sw)(x/s)) \approx \Delta' \cdot \frac{|x|}{s} \approx \frac{\Delta \cdot |x|}{s},
$$

which is the original error divided by $s$. You have reduced the relative quantization error on the salient channel by a factor of $s$ — without keeping a single weight in fp16, without a ragged layout. The whole matrix stays uniform int4; you just pre-multiplied salient channels by a per-channel scale (folded into the weights) and folded the inverse scale into the preceding layer's output. The figure below contrasts naive rounding against this activation-aware scaling.

![Before and after view of naive round to nearest leaving large error on salient channels versus AWQ scaling salient channels up before rounding and down after so their error drops](/imgs/blogs/llm-quantization-weight-only-gptq-awq-5.png)

### Choosing the scales

How big should $s$ be per channel? Scale too little and you under-protect the salient channels; scale too much and you inflate the other channels in the group, hurting them. AWQ picks the per-channel scales by a small grid search that minimizes the layer's output reconstruction error on calibration data — using the activation magnitudes to decide which channels deserve protection. In the paper's formulation the scale for a channel is driven by its average activation magnitude raised to a tuned power $\alpha$:

$$
s_j \propto \big(\bar{|x_j|}\big)^{\alpha},
$$

with $\alpha$ found by a quick search per layer. The whole thing needs only activation *statistics* (which channels are large on average), so AWQ is more robust to the exact calibration set than GPTQ is — it does not overfit to specific calibration examples, just to the distribution of which channels are hot. That robustness is one reason AWQ tends to generalize a touch better out of the calibration domain.

It is worth being precise about why the scaling does not just move the problem around. When you multiply channel $j$'s weights by $s_j > 1$, you have made those weights larger, which seems like it should make *their* group's range $r$ — and therefore step size $\Delta$ — larger, hurting every weight in the group. The reason it nets out positive is the asymmetry between salient and non-salient channels. Salient channels are about 1% of the total, so within any group most channels are non-salient and small. Scaling up the rare salient channel by a modest $s$ (the search typically lands $s$ in the range of a few-fold, not orders of magnitude) raises that one channel toward, but usually not past, the group's existing maximum — so $\Delta$ barely moves. Meanwhile the salient channel's *own* relative error drops by the full factor $s$. The math only works because saliency is sparse; if a third of channels were salient, scaling them all up would inflate every group's range and the trick would collapse. Sparsity of importance is the precondition, and the LLM.int8() outlier findings are exactly the evidence that importance is sparse.

The grid search for $\alpha$ deserves a word because it is what makes AWQ "activation-aware" rather than just "scale the big weights." A pure weight-magnitude criterion would protect the largest *weights*; AWQ protects the largest *weight-times-activation products*, because a large weight multiplied by a tiny activation contributes little to the output and is not worth protecting, while a modest weight multiplied by a huge activation is. The search over $\alpha$ interpolates between "protect by activation magnitude" ($\alpha$ large) and "protect everything equally" ($\alpha = 0$), and it picks, per layer, the exponent that minimizes the measured output reconstruction error. It runs in seconds per layer because evaluating a candidate $\alpha$ is just one forward pass of the layer on the calibration batch.

### Running AWQ

The `autoawq` library is the standard path, and vLLM serves AWQ checkpoints natively:

```python
# pip install autoawq
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_id = "meta-llama/Llama-2-7b-hf"
model = AutoAWQForCausalLM.from_pretrained(model_id)
tok = AutoTokenizer.from_pretrained(model_id)

quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",     # fused int4 GEMM kernels for serving
}
model.quantize(tok, quant_config=quant_config)   # computes per-channel scales
model.save_quantized("Llama-2-7b-awq-int4")
tok.save_pretrained("Llama-2-7b-awq-int4")
```

Serving it under vLLM is a single flag, and this is where AWQ shines because the layout is plain int4 and the fused GEMM kernels are fast:

```bash
# vLLM auto-detects AWQ from the checkpoint config; the flag is explicit here.
python -m vllm.entrypoints.openai.api_server \
    --model Llama-2-7b-awq-int4 \
    --quantization awq \
    --max-model-len 4096
```

GPTQ versus AWQ in one breath: both land near 4 bits/weight with group size 128, both need a small calibration pass, and on big models their perplexity is within noise of each other. GPTQ does explicit Hessian-guided error compensation and has the widest tooling and serving support; AWQ does activation-aware scaling, keeps a clean uniform layout (often slightly faster kernels), and tends to be a bit more robust to calibration-set choice. If you are serving on a GPU and want the absolute lowest perplexity with mature tooling, GPTQ is the safe default; if you want clean fast kernels and calibration robustness, AWQ is excellent. On a laptop CPU or Apple Silicon, neither — you want GGUF.

## 5. GGUF k-quants: the format that runs on your laptop

GPTQ and AWQ target GPUs and the PyTorch/vLLM ecosystem. The moment you want to run on a laptop CPU, on Apple Silicon's unified memory, or on a Raspberry Pi, you are in `llama.cpp` land, and the format is **GGUF** with **k-quants**. This is the format behind the local-LLM explosion — Ollama, LM Studio, and most "run it on your machine" tools are `llama.cpp` underneath, reading GGUF files.

### The block structure

k-quants do not use a single group size and a single scale type. They use a nested **super-block** structure that quantizes the scales themselves to save metadata bits. Take Q4_K, the 4-bit k-quant that underpins the popular `Q4_K_M`:

- A **super-block** holds **256 weights**.
- The super-block is divided into **8 sub-blocks of 32 weights** each.
- Each weight is stored in **4 bits**.
- Each sub-block has its own **scale** and **min**, but those are quantized to **6 bits** (not fp16).
- The whole super-block carries **two fp16 values**: a super-scale and a super-min that the 6-bit sub-scales are themselves quantized against.

So the per-weight cost is: 4 bits for the weight, plus the sub-block scale/min amortized (8 sub-blocks × two 6-bit values over 256 weights ≈ 0.375 bits/weight), plus the two fp16 super-values over 256 weights (≈ 0.125 bits/weight). Total: about **4.5 bits/weight** for Q4_K. The figure below lays out that super-block.

![Stacked layers of a GGUF Q4 k-quant super-block holding 256 weights as eight 32-weight sub-blocks with quantized sub-scales under one fp16 super-scale at about four and a half bits per weight](/imgs/blogs/llm-quantization-weight-only-gptq-awq-6.png)

Quantizing the scales to 6 bits is the trick that lets k-quants use a *small* group (32 weights, tight ranges, great accuracy) without paying the full fp16-scale overhead that group size 32 would otherwise cost (which we computed as 5.0 bits/weight back in §2). k-quants get a 32-wide group's accuracy at 4.5 bits/weight. That is the engineering insight that made them the default.

Step back and notice what just happened, because it is a recurring pattern in good quantization design. The naive way to get tight ranges is small groups, but small groups mean many scales, and scales in fp16 are expensive (16 bits each). k-quants observe that the scales *themselves* are a tensor of numbers with a limited range, so they quantize the scales — recursively applying the same idea one level up. The super-scale is the "scale of the scales." You pay two fp16 numbers per 256 weights for the top level, then 6-bit scales per 32-weight sub-block, then 4-bit weights. Each level is cheap because the level above it has already tightened the range it has to cover. This nested structure is why a Q4_K file lands near 4.5 bits/weight while resolving weights as finely as a 32-wide fp16-scale scheme would at 5.0 bits/weight — a real 10% size saving at equal accuracy, multiplied across every weight in the model.

### Reading the names: Q4_K_M, Q5_K_M, Q6_K

The naming is `Q<bits>_K_<size>` and it tells you a lot once you decode it:

- `Q4`, `Q5`, `Q6` — the *base* bits per weight in the main blocks.
- `_K` — uses the k-quant super-block machinery (quantized scales). The older `Q4_0`, `Q4_1`, `Q8_0` are the "legacy" quants without it; `Q8_0` (8-bit, simple) is still common as a near-lossless reference.
- `_S`, `_M`, `_L` — small, medium, large *mixes*. k-quants are **mixed-precision per tensor type**: the more sensitive tensors (attention output projections, the `feed_forward.w2` down-projections, sometimes embeddings) are stored at a *higher* bit-width than the rest. `Q4_K_M` keeps some of those sensitive tensors at Q6_K while the bulk stays Q4_K; `Q4_K_S` keeps more of them at Q4. The "M" mix is the community's recommended default — it is the accuracy/size sweet spot for 4-bit.

So `Q4_K_M` is roughly 4.5–4.8 bits/weight effective (the mix pushes it slightly above raw Q4_K), `Q5_K_M` about 5.5, `Q6_K` about 6.6, and `Q8_0` is 8.5. As a rule of thumb people repeat for a reason: `Q4_K_M` is the smallest quant most people should run for general use; `Q5_K_M` if you have the RAM and want a margin; `Q6_K`/`Q8_0` when you want quality essentially indistinguishable from fp16 and size is not the constraint; go below Q4 (Q3, Q2) only when you are truly memory-starved and have measured that the quality is still acceptable for your task.

The mixed-precision-by-tensor-type design is the quietly important part, and it is a generalizable lesson. k-quants do not treat every weight matrix equally because every weight matrix is not equally sensitive. The empirical finding baked into the `_M` mix is that a handful of tensor *types* — notably the attention output projection and the feed-forward down-projection, which sit at the end of each block where errors have nowhere left to be corrected — hurt the most when quantized hard. So those get a bit or two more, and the bulk of the model (the larger up-projections and gate projections, the query/key/value projections) gets the base 4 bits. This is exactly the same "spend bits where they matter" principle as GPTQ's Hessian and AWQ's salient channels, just applied at the coarse granularity of whole tensors rather than per-weight or per-channel. When you pick `Q4_K_M` over `Q4_K_S`, you are buying that extra protection on the sensitive tensors for a small size bump — almost always the right trade.

### The conversion and quantization flow

The `llama.cpp` flow is two steps: convert the HuggingFace checkpoint to a GGUF (initially in fp16 or Q8_0), then quantize that GGUF down to the k-quant you want.

```bash
# 1. Build llama.cpp (one time)
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build && cmake --build build --config Release -j

# 2. Convert HF model -> GGUF in fp16
python convert_hf_to_gguf.py /models/Llama-2-7b-hf \
    --outfile llama-2-7b-f16.gguf --outtype f16

# 3. Quantize the fp16 GGUF down to Q4_K_M
./build/bin/llama-quantize llama-2-7b-f16.gguf \
    llama-2-7b-Q4_K_M.gguf Q4_K_M
```

That `llama-quantize` step is fast (minutes on CPU) and needs **no calibration data** — k-quants are a data-free round-to-nearest scheme with the clever block structure, not a Hessian/activation method.

There is, however, an optional **importance matrix** ("imatrix") flavor that does bring calibration data into the k-quant flow, and it is worth knowing because it is how `llama.cpp` closes most of the remaining gap to GPTQ/AWQ at the low bit-widths. You first run the model over a calibration corpus to record, per weight, how much it contributes to the output (an activation-derived importance, closely related to the diagonal of GPTQ's Hessian), then feed that importance file to `llama-quantize`. The quantizer uses it to weight the rounding so that important weights are kept closer to their true values:

```bash
# Build an importance matrix from a calibration text, then quantize with it
./build/bin/llama-imatrix -m llama-2-7b-f16.gguf \
    -f calibration.txt -o llama-2-7b.imatrix

./build/bin/llama-quantize --imatrix llama-2-7b.imatrix \
    llama-2-7b-f16.gguf llama-2-7b-IQ3_M.gguf IQ3_M
```

The imatrix matters most for the sub-4-bit "IQ" quants (IQ3, IQ2), where data-free rounding loses too much; for Q4_K_M and up the data-free version is already close enough that most people skip it. The lesson is that all three method families converge on the same idea — use data to figure out which weights matter and protect them — they just package it differently: GPTQ as a Hessian correction, AWQ as a channel scaling, and llama.cpp as an optional importance weighting on top of a strong data-free block format.

Running it is one command, and the `-ngl` flag controls how many layers go on the GPU — on a Mac that is the Metal GPU sharing unified memory:

```bash
# Run with all layers offloaded to GPU (-ngl 99), 4096-token context
./build/bin/llama-cli -m llama-2-7b-Q4_K_M.gguf \
    -ngl 99 -c 4096 -p "Explain weight-only int4 quantization in one sentence."

# Or serve an OpenAI-compatible endpoint
./build/bin/llama-server -m llama-2-7b-Q4_K_M.gguf -ngl 99 -c 4096 --port 8080
```

On a CPU-only box, drop `-ngl` (or set it to 0) and `llama.cpp` runs the int4 kernels on AVX2/AVX-512/NEON. This is the path that runs a 7B model on a Raspberry Pi 5 (slowly — a few tokens/s) or comfortably on any modern laptop.

## 6. Measuring quality: perplexity, accuracy, and bits

You cannot ship a quantized model on faith. You measure it. Two measurements matter: **perplexity** (intrinsic, fast, sensitive) and **downstream accuracy** (what users feel).

### Perplexity

Perplexity is the exponentiated average negative log-likelihood the model assigns to a held-out text. Lower is better; a perfect predictor has perplexity 1, random over a 32k vocab has perplexity ~32k. For quantization, the metric you actually care about is the **delta**: how much does perplexity rise versus the fp16 baseline on the *same* held-out set (WikiText-2 is the community standard). A delta under ~0.1 is "indistinguishable"; under ~0.5 is "fine for most uses"; over ~1.0 on a small model is "you will feel it." `llama.cpp` computes perplexity directly:

```bash
./build/bin/llama-perplexity -m llama-2-7b-Q4_K_M.gguf \
    -f wikitext-2-raw/wiki.test.raw -ngl 99
# prints e.g. "PPL = 5.84 +/- 0.03"  (compare to the f16 baseline's PPL)
```

Perplexity is sensitive and cheap, but it is not the user-facing truth. A model can hold its perplexity and still lose a few points on a reasoning benchmark, because perplexity averages over all tokens while a benchmark hinges on a few decisive ones. So pair it with a downstream eval — `lm-evaluation-harness` over MMLU, GSM8K, ARC, HellaSwag — at least as a spot check before shipping.

```bash
# Spot-check downstream accuracy of the quantized model
pip install lm-eval
lm_eval --model hf \
    --model_args pretrained=Llama-2-7b-gptq-int4-g128 \
    --tasks mmlu,gsm8k,arc_challenge,hellaswag \
    --batch_size 8
# compare each task's acc to the fp16 baseline; flag any >1-point drop
```

Two failure modes show up here that perplexity hides. The first is **generation degradation under sampling**: perplexity is a teacher-forced, single-step metric (it scores the next token given the *true* prefix), but real generation feeds the model its own outputs, so small per-token errors compound over a long response. A model with a tiny perplexity delta can still produce noticeably worse multi-paragraph or multi-step outputs because the errors accumulate. The second is **task-specific cliffs**: math and code are more brittle to quantization than open chat, because a single wrong token (a digit, a bracket) fails the whole problem, whereas chat tolerates fuzz. So a Q4_K_M chat model that feels great might drop several points on GSM8K. This is exactly why you eval the tasks you actually serve, not just perplexity.

### Why int4 is nearly free on big models but bites small ones

Here is the pattern that surprises people: int4 weight-only loses **less than one perplexity point** on a 7B-and-up model, but loses noticeably more on a 1B or smaller model. Why?

The information-theoretic intuition: quantization injects a fixed *relative* noise floor per weight (recall $\sigma_q^2 \propto r^2 / 2^{2b}$ — same bit-width, same relative noise). Large models are **over-parameterized and redundant** — many weights encode overlapping information, so the network has spare capacity to absorb per-weight noise without changing its function much. The output error from quantization is a sum of many small, partially-cancelling perturbations across a huge, redundant weight space. Small models carry far less redundancy: each weight does more work, so the same relative noise on each weight perturbs the function more, and the errors cancel less. This is the same redundancy story that underlies the lottery-ticket and pruning results — big models have slack, small models do not.

There is a clean statistical way to see the cancellation. If a layer's output dimension is a sum of $n$ weighted inputs and each weight carries independent quantization noise of variance $\sigma_q^2$, the noise injected into that output sums in variance, but the *signal* sums coherently. For a wide layer (large $n$, big models) the signal-to-noise ratio of the output improves relative to the per-weight ratio, because the central-limit averaging smooths independent errors while the intended signal adds up in phase. Wider layers, more of them, deeper redundancy — every dimension along which big models are bigger is a dimension along which quantization noise averages out more. That is why the perplexity-versus-bits curve is nearly flat down to 4 bits for a 7B and falls off a cliff for a 1B: the 1B simply does not have the width and depth to average the noise away.

This also explains the cliff *within* a model as you push below 4 bits. From the SQNR law, each bit you drop is another 6 dB of noise — but the redundancy that absorbs it is finite. Down to int4, big models have enough slack; at int3 you have doubled the noise standard deviation again and started exhausting that slack, so perplexity rises faster than linearly; by int2 the noise overwhelms the redundancy and the model degrades sharply. The frontier methods that make 2- and 3-bit usable (importance matrices, mixed-precision, vector quantization, and learned codebooks) all work by spending bits *unevenly* — more on the weights that matter, fewer on the ones that do not — rather than the uniform budget that the cliff punishes.

The practical consequence is a rule you can act on: **for 7B+, int4 weight-only with group size 128 is a near-free win** — take it. For models under ~2–3B, int4 starts to cost real accuracy; consider int5/int6 (Q5_K_M / Q6_K), or a smarter method (GPTQ/AWQ over naive round-to-nearest), or accept int4 only after measuring your downstream task. The figure below puts the bits-versus-size-versus-perplexity trade in one matrix.

![Matrix comparing fp16 int8 int4 and int3 on a 7B model showing size shrinking with bit width while perplexity stays flat to int4 then degrades sharply at int3](/imgs/blogs/llm-quantization-weight-only-gptq-awq-8.png)

### The group-size / accuracy / size trade

Within int4, the group size is the second knob. From §2, group size 128 costs 4.25 bits/weight, group size 32 costs 5.0. The accuracy gain from 128 → 32 is real but small on big models (often <0.1 perplexity); the size grows ~18%. So group size 128 is the default; drop to 64 if you have a sensitive model and the headroom, and use the k-quant super-block trick (Q4_K) when you want 32-wide-group accuracy at ~4.5 bits instead of 5.0.

## 7. Worked examples on real hardware

Numbers make this concrete. These are representative figures consistent with the published literature (GPTQ and AWQ papers, llama.cpp benchmarks, and widely reported community measurements); treat exact tokens/s as approximate and hardware-dependent, and always re-measure on your own target with warm-up runs and batch size one.

#### Worked example: Llama-2 7B fp16 vs GPTQ-int4 vs AWQ-int4 on a 24 GB GPU

Target: a single 24 GB consumer GPU (e.g., an RTX 3090/4090 class card). WikiText-2 perplexity baseline for Llama-2 7B fp16 is about **5.47**.

| Variant | Disk size | VRAM (weights+KV, bs=1) | Decode tok/s (bs=1) | WikiText-2 PPL | PPL delta |
| --- | --- | --- | --- | --- | --- |
| fp16 | ~13 GB | ~14–15 GB | ~40 tok/s | 5.47 | — |
| GPTQ int4 g128 | ~3.9 GB | ~5–6 GB | ~90–120 tok/s | ~5.6 | +0.1 |
| AWQ int4 g128 | ~3.9 GB | ~5–6 GB | ~100–130 tok/s | ~5.6 | +0.1 |

Read what this says. Both int4 methods cut disk and VRAM by roughly 3.3×, more than double decode throughput on a memory-bound workload, and cost about **0.1 perplexity** — within the range you would struggle to feel on most tasks. AWQ's clean uniform layout often edges out GPTQ on raw kernel speed; GPTQ with `desc_act` sometimes edges out AWQ on perplexity. On a 7B model the difference is in the noise. The real story is the column of perplexity deltas: near-free.

The honest measurement caveats: those tok/s are warm (kernels compiled, caches hot), batch size one (the on-device reality), and short-context. With a long context the KV cache — still fp16 here — grows and starts to dominate memory traffic, eroding the int4 advantage; that is exactly the problem [the activations and KV-cache post](/blog/machine-learning/edge-ai/llm-quantization-activations-smoothquant-kv-cache) solves with KV-cache quantization.

#### Worked example: GGUF Q4_K_M vs Q8_0 vs fp16 on an M2 MacBook

Target: an Apple M2 (or M2 Pro) MacBook with unified memory, running `llama.cpp` with Metal (`-ngl 99`). This is the canonical "run an LLM on my laptop" setup. Llama-2 7B again.

| Variant | File size | Bits/weight | Decode tok/s (Metal) | Quality vs fp16 |
| --- | --- | --- | --- | --- |
| fp16 (F16) | ~13 GB | 16 | ~20–25 tok/s | reference (and may not fit 16 GB) |
| Q8_0 | ~7.2 GB | ~8.5 | ~30–35 tok/s | indistinguishable (PPL +~0.02) |
| Q5_K_M | ~4.8 GB | ~5.7 | ~40–48 tok/s | excellent (PPL +~0.1) |
| Q4_K_M | ~4.1 GB | ~4.8 | ~48–60 tok/s | great (PPL +~0.3) |

The fp16 row is half cautionary tale: on a 16 GB Mac, fp16 7B plus context plus the OS leaves you fighting for memory, and you may swap or fail to load. Q4_K_M loads comfortably with room for a long context, runs at the most tokens/s, and the quality cost is a few tenths of a perplexity point — for a chat assistant or a coding helper, you will not notice. **Q4_K_M is the default for a reason.** If you have a 32 GB+ Mac and want a quality margin, Q5_K_M or Q6_K is a cheap upgrade. Q8_0 exists mainly as a near-lossless reference and for cases where you specifically want minimal degradation and have the RAM.

The measurement discipline still applies: `llama.cpp` numbers swing with the Metal/CPU split (`-ngl`), thread count, context length, and thermal state — a laptop throttles under sustained load, so the tok/s after 30 seconds of generation is lower than the first burst. Measure the steady state, not the first token.

#### Worked example: does a 7B fit a phone budget?

Target: a flagship phone with 8 GB of RAM, of which an app can realistically claim maybe 4 to 5 GB before the OS starts killing it. Can a 7B model fit? In fp16 (~13 GB) it is hopeless. In int4: the weights are ~3.5 to 4 GB. Then add the KV cache. For a 7B Llama-2 with 32 layers, 32 heads, head dimension 128, the KV cache is $2 \times \text{layers} \times \text{heads} \times \text{head\_dim} \times 2\text{ bytes}$ per token in fp16, which is $2 \times 32 \times 32 \times 128 \times 2 \approx 1.05$ MB/token. A 2,048-token context is then ~2.1 GB of KV cache — already as big as a chunk of the weights. So the real budget is roughly 4 GB (int4 weights) + 2 GB (2k-context KV) + runtime overhead, which is right at the edge of a phone's 4 to 5 GB app budget, and only if you cap the context. The takeaways for on-device: int4 weights are *necessary but not sufficient* — once you want long context, the fp16 KV cache becomes the binding constraint, and you need to either cap context, use grouped-query attention models (which shrink the KV cache structurally), or quantize the KV cache. KV-cache quantization is precisely the lever [the next post](/blog/machine-learning/edge-ai/llm-quantization-activations-smoothquant-kv-cache) pulls. Weight-only int4 gets a 7B model *onto* the phone; keeping it there at long context needs the second technique.

## 8. Method comparison and case studies

We have four real weight-only int4 options on the table: GPTQ, AWQ, GGUF k-quants, and bitsandbytes NF4. NF4 (4-bit NormalFloat, from Dettmers et al.'s QLoRA, 2023) deserves a mention because it is the fourth common int4 you will meet. NF4 uses a *non-uniform* 4-bit grid whose levels are spaced according to the quantiles of a normal distribution — because neural-net weights are roughly Gaussian, putting more levels where the weights are dense (near zero) and fewer in the tails is information-theoretically efficient. It is data-free and one line in transformers, and it is the default for QLoRA fine-tuning. For pure inference its perplexity is competitive but its kernels are generally not as fast as AWQ/GPTQ's fused int4 GEMMs, so people reach for it mostly when fine-tuning a quantized model, not when squeezing maximum serving throughput.

The NF4 idea is worth dwelling on for a moment because it is the clearest example of matching the quantization grid to the data. A uniform int4 grid spends its 16 levels equally across the range, but weights are not uniform — they pile up near zero and thin out in the tails. So a uniform grid wastes resolution on the sparse tails and starves the dense center. NF4 instead places its 16 levels at the 16 quantiles of a unit normal, so each level is equally *likely* to be used — maximum entropy per stored code, which is the information-theoretic optimum for a Gaussian source. QLoRA stacks a second trick, **double quantization**, on top: it quantizes the quantization constants themselves (the per-block scales), the same recursive-scale idea k-quants use, shaving the metadata overhead further. Loading an NF4 model is genuinely one line:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # the NormalFloat grid
    bnb_4bit_use_double_quant=True,     # quantize the scales too
    bnb_4bit_compute_dtype="bfloat16",  # dequant target for the matmul
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", quantization_config=bnb, device_map="cuda")
```

This is the fastest possible path from a fp16 checkpoint to a running 4-bit model — no calibration, no separate quantize step, it happens at load time. The trade is speed: bitsandbytes dequantizes in the kernel without the heavily-tuned fused int4 GEMMs that GPTQ and AWQ ship, so for raw serving throughput it trails them. Its sweet spot is QLoRA fine-tuning and quick experiments, not maximum-throughput inference.

Here is the decision matrix.

![Matrix comparing GPTQ AWQ GGUF Q4_K_M and bitsandbytes NF4 across bits per weight calibration needs and best use case](/imgs/blogs/llm-quantization-weight-only-gptq-awq-7.png)

In words:

| Method | Bits/wt | Calibration | Kernels / runtime | Best for |
| --- | --- | --- | --- | --- |
| **GPTQ** | ~4.0–4.25 | needs calib set (few hundred seqs) | fused int4 on GPU; vLLM/TGI/transformers | lowest-PPL GPU serving, widest tooling |
| **AWQ** | ~4.0–4.25 | needs calib (activation stats) | clean uniform int4 GEMM; vLLM | fast GPU serving, calibration-robust |
| **GGUF Q4_K_M** | ~4.5–4.8 | none (data-free) | `llama.cpp` CPU/Metal/CUDA | laptops, Macs, Pi, local apps |
| **bitsandbytes NF4** | ~4.0 | none (data-free) | transformers; slower than fused int4 | QLoRA fine-tuning, quick PTQ |

### A two-minute decision procedure

You rarely need to agonize over which method. The runtime you are targeting usually decides it for you, because each method's fast kernels live in a specific ecosystem:

- **Serving on a GPU with vLLM/TGI, want lowest perplexity and widest tool support?** GPTQ int4 g128 with `desc_act`. It is the most-supported quantized format in the serving stack.
- **Serving on a GPU, want clean fast kernels and calibration robustness?** AWQ int4 g128. Often a touch faster than GPTQ at near-identical perplexity, and friendlier when your serving distribution differs from your calibration set.
- **Running on a laptop, Mac, Raspberry Pi, or in Ollama/LM Studio?** GGUF Q4_K_M. One file, every CPU/Metal/CUDA backend, no calibration, the de facto local-LLM standard. Step up to Q5_K_M / Q6_K if you have the RAM and want margin, especially on smaller models.
- **Fine-tuning a quantized base (PEFT/QLoRA)?** bitsandbytes NF4. It is the format built for training-through, and the LoRA adapters land in 16-bit on top.

Notice that this is a *runtime-first* decision, not a method-first one. The methods are roughly equivalent in accuracy at int4 on big models; what differs is which ecosystem has the optimized kernels. Pick the runtime your deployment needs, and the format follows.

### Case studies from the literature

**GPTQ on OPT and BLOOM, then LLaMA (Frantar et al., 2022).** The original paper showed that 175B-parameter models could be quantized to 3–4 bits with negligible perplexity loss in a few GPU-hours, and crucially that a 175B model in int3 could run on a single 80 GB GPU it could never fit in fp16. The headline: at 4 bits, output-error-minimizing quantization made huge models nearly free to compress, and the perplexity delta on the biggest models was a fraction of a point.

**AWQ on Llama and instruction-tuned models (Lin et al., 2023).** AWQ matched or beat GPTQ's perplexity at int4 across Llama, Llama-2, and instruction-tuned variants, and showed strong robustness on tasks *outside* the calibration domain (e.g., calibrating on a general corpus and evaluating on code or math), because it depends only on which channels are salient, not on specific calibration examples. Its accompanying TinyChat runtime demonstrated 4-bit Llama running interactively on a laptop GPU and even on a Jetson-class edge board.

**QLoRA / NF4 (Dettmers et al., 2023).** QLoRA fine-tuned a 65B model on a single 48 GB GPU by holding the base weights in NF4 (frozen, 4-bit) and training only small LoRA adapters in 16-bit — and matched 16-bit full fine-tuning quality. This is the canonical proof that 4-bit weights retain enough information not just to *run* but to *fine-tune through*. If you are doing PEFT on a quantized base, this is the path; see the series' fine-tuning material for how LoRA composes with quantization.

**The local-LLM ecosystem (llama.cpp / GGUF).** The most-deployed weight-only quantization on Earth is almost certainly GGUF k-quants, by sheer install count — every Ollama and LM Studio user running a Q4_K_M model is running it. The empirical community result, replicated thousands of times, is the one in our M2 worked example: Q4_K_M gives you a 7B model that fits 8 GB of RAM, runs at interactive speed on a laptop CPU or Mac GPU, and is good enough that most users never reach for a larger quant.

**On-device small models (Phi, Gemma, Llama-3.2 1B/3B).** The newer wave of deliberately small models ships with quantized variants meant for phones and laptops, and they are the cautionary half of the int4 story. A 1B or 3B model quantized to int4 loses noticeably more than a 7B does — exactly the redundancy argument from §6 — so vendors often ship these at int4 *with* QAT (quantization-aware training, the next lever) or recommend Q5/Q6 for the smaller sizes. The practical reading: do not blindly apply the "int4 is free" rule you learned on 7B to a 1B model. The break-even shifts with scale; measure it.

**Whisper and speech on-device.** Outside text generation, the same weight-only int4/int5 GGUF approach (via whisper.cpp) is what puts a multi-hundred-megabyte speech model onto a laptop or phone for offline transcription. It is a useful reminder that "weight-only int4 for memory-bound decode" is a property of the *workload shape* (autoregressive, memory-bound), not of language modeling specifically — any model whose inference streams large weights against small inputs benefits the same way.

For a deeper, format-level tour of every GGUF quant type and the exact bit layouts, the companion piece [how quantization works: GGUF quant types decoded](/blog/machine-learning/large-language-model/how-quantization-works-gguf-quant-types-decoded) goes file-format-deep where this post stays method-deep.

## 9. When weight-only int4 is the right call (and when it is not)

Weight-only int4 is the first lever you reach for when you want an LLM on consumer hardware. But like every technique in [the taxonomy](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression), it is a cost with a domain of applicability. Here is the decisive version.

**Reach for weight-only int4 when:**

- Your workload is **memory-bound decode** at **batch size one** — i.e., on-device generation, a single user, interactive latency. This is the case it was built for, and the case where it gives you both smaller *and* faster essentially for free.
- The model is **7B or larger**. Big models have the redundancy to absorb int4 noise at well under a perplexity point. Take the win.
- You are **memory-constrained** — the model does not fit, or barely fits, in fp16. Quantization is what makes it fit at all, KV cache and runtime overhead included.
- You want a **single artifact that runs across CPU, Mac, and edge** — that is GGUF Q4_K_M. One file, every backend.

**Do not reach for it (or reach further) when:**

- You are **compute-bound**: large-batch serving, or long-prompt prefill. There the bottleneck is FLOPs, not weight bytes, and weight-only int4 (which still does the matmul math in fp16 after dequantizing) buys you little speed. You want activation/weight int8 with fast low-precision *math* kernels — the compute-bound win — which is [the activations post](/blog/machine-learning/edge-ai/llm-quantization-activations-smoothquant-kv-cache).
- Your model is **small (≤1–2B)**. Int4 can cost real accuracy here; measure, and consider int5/int6 or a stronger method before committing.
- The **KV cache dominates** your memory (very long contexts, or many concurrent sequences). Weight-only int4 shrinks weights, not the KV cache; once the KV cache is the bigger consumer, you need KV-cache quantization to keep going. Again, the next post.
- You need to **push below 4 bits**. At int3 and especially int2, naive and even GPTQ/AWQ schemes start losing real quality; this is the frontier where mixed-precision, importance matrices, and newer methods are still moving, and where you must measure your downstream task, not trust a perplexity number.

The stress tests are where judgment lives. *What happens at int3?* On a 7B you might pay +1 to +2 perplexity and lose a few benchmark points — sometimes acceptable for a memory-desperate deployment, often not; never assume, measure. *What if the calibration set is tiny or off-distribution?* GPTQ can overfit to it (its Hessian is built from it); AWQ is more robust because it only needs which channels are salient; k-quants do not care because they are data-free. *What if you serve a non-English or code model?* Calibrate GPTQ/AWQ on representative data, or prefer AWQ/k-quants for their distribution robustness. *What if int4 kernels are slower than fp16 on your specific chip?* It happens on hardware without good int4 support — always benchmark the actual kernel, because a "smaller" model that runs slower has bought you nothing but disk space.

One more stress test that bites people in production: *what happens when the workload shifts from decode to prefill?* If your application sends long prompts and generates short answers — retrieval-augmented generation with a big stuffed context, classification, or summarization of a long document — you spend most of your compute in prefill, which is compute-bound, and weight-only int4 gives you little speedup there (it may even be marginally slower because of dequant overhead on a FLOP-limited kernel). The disk and memory savings still apply, so int4 is still worth it to *fit* the model, but do not promise a 2-3× latency win that only materializes for decode-heavy chat. Profile your actual prompt-to-completion ratio before you quote a speedup; the roofline does not care about your marketing slides. And when prefill latency is the problem, the answer is the *next* lever — low-precision activation math — not more weight-only quantization.

The decisive summary: weight-only int4 is the highest-leverage, lowest-risk quantization move for the on-device, batch-one, decode-heavy LLM workload — which is most of what "running an LLM yourself" means. It is smaller and usually faster for almost no quality cost on a 7B-and-up model. Its limits are crisp and predictable: small models, very low bits, compute-bound prefill, and KV-cache-dominated long contexts. Knowing exactly where those limits are — and that the next two levers (activation quantization and KV-cache quantization) pick up precisely where this one stops — is the difference between cargo-culting a Q4_K_M download and engineering a deployment.

## 10. Key takeaways

- **Weight-only int4 wins for LLMs because decode is memory-bound.** A 7B model is ~14 GB in fp16 and reading those bytes per token is the bottleneck; int4 reads ~4× fewer bytes, so it is both ~4× smaller and typically 2–3× faster — a near-linear speedup from cutting weight bytes.
- **Keep activations in fp16 to dodge the outlier problem.** LLM activations carry extreme outliers that wreck naive low-precision rounding; weight-only quantization avoids that failure mode entirely. Activation quantization is a separate, harder problem for the compute-bound regime.
- **Group-wise quantization is the foundation.** One scale per group of 64/128 weights isolates outliers to their own group; group size 128 (~4.25 bits/weight) is the accuracy/size knee everyone uses.
- **GPTQ minimizes layer output error with a Hessian.** It quantizes column by column and pushes each column's rounding error into the not-yet-quantized columns via $H^{-1}$ — better than round-to-nearest at the same bit-width, at the cost of a calibration pass.
- **AWQ protects the ~1% salient channels by scaling, not by mixed precision.** Scale a salient channel up by $s$ before rounding and down after, cutting its relative error by $s$ while keeping a clean uniform int4 layout — fast kernels and calibration robustness.
- **GGUF k-quants are the laptop/Mac/edge default.** Q4_K_M's super-block structure (256-weight super-blocks, 32-weight sub-blocks, 6-bit sub-scales, mixed precision on sensitive tensors) delivers 32-wide-group accuracy at ~4.5 bits/weight, data-free, and runs everywhere `llama.cpp` runs.
- **int4 is near-free on 7B+ and risky on small models.** Big models' redundancy absorbs the noise (<1 perplexity point); small models feel it. Measure on small models; default to int4 on large ones.
- **Always measure both perplexity and a downstream eval, on your hardware, warm, at batch one.** A smaller model that runs slower, or that quietly drops benchmark points, is not a win. Re-measure on your target; never trust a generic tokens/s.

## 11. Further reading

- **Frantar, Ashkboos, Hoefler, Alistarh (2022), "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers."** The layer-wise, Hessian-guided, column-by-column int4 method; the OBQ/OBS lineage and the scaling to 175B in a few GPU-hours.
- **Lin, Tang, Tang, Yang, Dang, Han (2023), "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration."** The salient-channel observation, the scaling derivation, and the TinyChat edge runtime.
- **Dettmers, Pagnoni, Holtzman, Zettlemoyer (2023), "QLoRA: Efficient Finetuning of Quantized LLMs."** NF4, double quantization, and fine-tuning a 65B model on one 48 GB GPU — the proof that 4-bit weights retain enough to train through.
- **Dettmers, Lewis, Belkada, Zettlemoyer (2022), "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale."** The activation-outlier finding that explains why weight-only is the safe path and why activation quantization is hard.
- **llama.cpp k-quants documentation and source (`ggml-org/llama.cpp`).** The authoritative spec for GGUF super-block layouts, the Q*_K mixes, and the `llama-quantize` / `llama-perplexity` tools.
- Within this series: [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) for the four-lever frame, [the roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) for memory-bound vs compute-bound, [LLM quantization II: activations, SmoothQuant, and the KV cache](/blog/machine-learning/edge-ai/llm-quantization-activations-smoothquant-kv-cache) for the compute-bound and long-context story, and [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) for how this lever composes with pruning and distillation.
- Going format-deep: [how quantization works: GGUF quant types decoded](/blog/machine-learning/large-language-model/how-quantization-works-gguf-quant-types-decoded).

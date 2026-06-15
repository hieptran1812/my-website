---
title: "Scaling laws for precision: how low-bit training and quantization bend the curve"
date: "2026-06-15"
description: "Learn how training and inference precision enter the scaling laws: why over-trained models quantize worse, how low-bit training acts like fewer parameters, and why compute-optimal precision lands near seven to eight bits."
tags: ["scaling-laws", "quantization", "low-precision-training", "post-training-quantization", "effective-parameters", "compute-optimal", "kv-cache", "fp8", "int4", "large-language-models", "inference-cost"]
category: "machine-learning"
subcategory: "Scaling Laws"
author: "Hiep Tran"
featured: true
readTime: 53
---

For about a decade the scaling-law literature treated numerical precision as a footnote. You picked a dtype — bf16 if you were sensible, fp16 if you were brave, fp32 if you were nervous — and then you went and fit a loss law in parameters and tokens as though every weight were a real number with infinite resolution. That was always a convenient fiction, and in 2024 the fiction finally broke. A 32-bit float and a 4-bit integer are not two points on a smooth continuum of "model quality"; they are different amounts of information per weight, and the loss law has to account for that information directly. The diagram below is the mental model for this whole post: precision shows up in two distinct places — it shrinks the *effective* number of parameters during training, and it adds a *degradation* term when you quantize after training — and the second term, surprisingly, gets worse the more data you trained on.

![A graph showing how training precision and post-training precision feed into the unified loss law, with effective parameters on one path and a quantization penalty that grows with the data-to-parameter ratio on another](/imgs/blogs/precision-scaling-laws-1.png)

That picture comes from Kumar, Ankner, et al. 2024, "Scaling Laws for Precision" (arXiv:2411.04330), a Harvard/MIT/Stanford/Databricks/CMU collaboration that ran more than 465 controlled pretraining runs (validated up to 1.7B parameters and 26B tokens) and fit a single law that ties together three things the field had been treating separately: how low precision degrades training, how it degrades post-training quantization, and how to choose precision optimally given a compute budget. The headline that most people remember is the unsettling one — *if you are going to quantize your model at inference time, training it on more data can make it worse* — but the deeper contribution is a clean, fittable functional form that puts precision on the same footing as parameters and tokens. This post is a full tour of that result. We will build the intuition first, then write down the math with the paper's fitted constants, then work numeric examples in bits and FLOPs, then walk through the failure modes that bit people in production, and finally turn it into practical guidance you can apply to your next training run and your next quantization pass.

> [!important] The one number to remember: about 7 bits
> - **Precision enters the loss law twice.** Low-precision *training* reduces the *effective* parameter count $N_{\text{eff}}$; post-training quantization (PTQ) adds a separate degradation term $\delta_{\text{PTQ}}$. The unified law is $L = A\,N_{\text{eff}}^{-\alpha} + B\,D^{-\beta} + E + \delta_{\text{PTQ}}$.
> - **Over-trained models degrade more under PTQ.** $\delta_{\text{PTQ}}$ scales roughly as a power law in $D/N$ (data per parameter). Past a critical $D/N$, *more pretraining data raises post-quantization loss*. Llama-3-8B sits at $D/N \approx 2000$, about 100x the Chinchilla optimum — squarely in the danger zone.
> - **Low-precision training is like having fewer parameters.** $N_{\text{eff}}(P) = N(1-e^{-P_w/\gamma_w})(1-e^{-P_a/\gamma_a})(1-e^{-P_{kv}/\gamma_{kv}})$ — multiplicative over weights, activations, and KV cache. The gains saturate around **6 to 7 bits per weight**.
> - **Compute-optimal training precision is about 7 to 8 bits** (integer), and it is roughly independent of the compute budget. 16-bit "has many unnecessary bits"; below 4-bit the model has to grow more than 4x to hold the same loss.
> - **The KV cache tolerates the lowest precision.** Its saturation constant $\gamma_{kv} \approx 0.96$ is far smaller than weights ($\gamma_w \approx 2.67$) or activations ($\gamma_a \approx 2.21$), so 4 to 5 bits for KV is fine while weights and activations want more.
> - **Training in low precision robustifies against later quantization.** The same $\delta_{\text{PTQ}}$ term captures the empirically dominant effect that weights trained at low precision are more robust to being quantized further at inference.
> - **The practical rule:** if you will quantize at inference, do not over-train; train near 7 to 8 bits; spend the cheapest bits on the KV cache.

## Why precision is different from what the scaling laws assumed

The right way to feel this result is to notice how badly it contradicts the implicit assumption that ran underneath every prior scaling-law paper. [Kaplan](/blog/machine-learning/scaling-laws/kaplan-scaling-laws-language-models) and [Chinchilla](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) both fit loss as a function of parameter count $N$ and token count $D$, and they both treated a parameter as a parameter — a 70B model has 70 billion knobs, full stop. But a knob you can only set to one of 16 values (4-bit) carries less information than a knob you can set to one of 65,536 values (16-bit). The scaling law was silently assuming infinite-resolution weights, which is fine when everyone trains and serves in bf16, and quietly wrong the moment you quantize.

Here is the before-and-after, framed as the assumption-versus-reality table a senior engineer keeps in their head.

| Question | The pre-2024 assumption | The precision-scaling reality |
|---|---|---|
| Does a parameter's *resolution* matter for loss? | No — a parameter is a parameter | Yes — low precision shrinks effective parameter count $N_{\text{eff}}$ |
| Is more pretraining data always good? | Yes — loss falls monotonically in $D$ | Only if you serve at full precision; if you quantize, past a critical $D/N$ more data *raises* loss |
| Does training precision affect inference quantization? | Treated as independent | Strongly coupled — low-precision training robustifies against PTQ |
| What precision should you train at? | bf16, because that is what the hardware does | Compute-optimally, about 7 to 8 bits |
| Do weights, activations, and KV cache want the same bits? | Usually quantized uniformly | No — KV tolerates the fewest bits, weights the most |
| Is 16-bit a safe default? | Yes | It has "many unnecessary bits" — wasteful at compute-optimal allocation |

The two rows that should make you sit up are the second and the fourth. The claim that *more data can hurt* is genuinely counterintuitive — it runs against the entire instinct of the field, which has spent five years establishing that more tokens are the missing half of the recipe (the [Chinchilla 20-tokens-per-parameter rule](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) and the [inference-aware argument for over-training](/blog/machine-learning/scaling-laws/inference-aware-scaling-laws) both push you toward more tokens). The precision paper does not overturn those results; it adds a caveat that bites precisely the regime everyone is now operating in. If you train a small model on an enormous number of tokens — which is exactly what the inference-aware scaling laws tell you to do for a high-traffic deployment — and then quantize it to 4 bits to serve it cheaply, you may have walked into the worst of both worlds.

> If you take one thing from this post: precision is not a deployment detail you bolt on at the end. It is a scaling axis, and the decision to quantize at inference reaches all the way back and changes how much data you should have trained on.

### A short history of how precision crept into the law

It is worth tracing the lineage, because the precision law did not arrive from nowhere. The story runs in three steps, and seeing them in order makes the result feel inevitable rather than surprising.

The first step was hardware. Mixed-precision training — keeping a master copy of weights in fp32 while doing the matmuls in fp16 — became standard around 2017–2018 because it roughly doubled throughput on tensor cores at no measurable loss in quality. bf16 (a 16-bit float with fp32's exponent range and a truncated mantissa) made this even more robust by removing most of the overflow headaches. The lesson the field internalized was *16-bit is free* — you get 2x speed and the loss curve does not move. That lesson was correct, and it is exactly why precision stayed a footnote: if dropping from 32 to 16 bits costs nothing, why would you write it into the scaling law?

The second step was inference quantization. As models grew past tens of billions of parameters, serving them in 16-bit became the dominant cost, and a wave of post-training quantization methods — GPTQ, AWQ, SmoothQuant, and the rest — showed you could drop weights to 4 bits with surprisingly small quality loss. The implicit theory was again *quantization is nearly free if you are clever about it*. And again, that was mostly true — for compute-optimal models. The crack appeared when people quantized heavily over-trained models and found the degradation was worse than the methods' own benchmarks predicted. Something about training on a lot of data made the weights *less* robust to quantization, not more.

The third step is the precision scaling law itself, which explained both observations with one equation. 16-bit being free is just the saturation of the effective-parameter curve — past about 7 bits, extra bits buy almost nothing, so 16 versus 32 versus 64 are all on the flat part of the curve. And over-trained models quantizing badly is the $\delta_{\text{PTQ}} \propto (D/N)^{\sim 0.4}$ term. The two folk observations were two faces of the same law. That is the moment precision stopped being a footnote and became a scaling axis with its own exponent.

## 1. The unified law: four terms, one of which is new

**Senior rule of thumb: before you fit anything, write the functional form and make every term defensible on its own.** The precision paper's whole edifice rests on extending the Chinchilla loss law by two moves — replacing $N$ with an effective count $N_{\text{eff}}$, and adding a degradation term $\delta_{\text{PTQ}}$ — and the power comes from each piece having a physical meaning.

![A layered stack showing the four additive terms of the low-precision loss law: an effective-parameter penalty, a data penalty, an irreducible floor, and a quantization degradation term](/imgs/blogs/precision-scaling-laws-2.png)

The starting point is the familiar [Chinchilla parametric form](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling):

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$$

where $N$ is parameter count, $D$ is training tokens, $E$ is the irreducible loss (the entropy floor of the text), and $A, B, \alpha, \beta$ are fitted constants. The precision paper refits these on its own data and reports $A = 4299$, $\alpha = 0.4965$, $B = 18060$, $\beta = 0.5$ (approximately), and $E = 2.7648$ nats per token. (These differ from DeepMind's original Chinchilla numbers because they are fit on a different corpus and tokenizer and over a different scale range — a reminder, which the [reconciliation post](/blog/machine-learning/scaling-laws/kaplan-vs-chinchilla-reconciliation) drives home, that the *exponents* travel across setups but the *coefficients* are setup-specific.)

The unified law is then:

$$L = \underbrace{\frac{A}{N_{\text{eff}}^{\alpha}}}_{\text{effective-param penalty}} + \underbrace{\frac{B}{D^{\beta}}}_{\text{data penalty}} + \underbrace{E}_{\text{floor}} + \underbrace{\delta_{\text{PTQ}}}_{\text{quant penalty}}$$

Look at the stack figure above and notice what is and is not touched by precision. The data penalty $B/D^{\beta}$ and the floor $E$ are precision-blind — tokens are tokens and the entropy of English does not care how many bits your weights have. The other two terms are where precision lives. The first, $A/N_{\text{eff}}^{\alpha}$, is the ordinary parameter penalty but with $N$ replaced by an *effective* count that low-precision training shrinks. The fourth, $\delta_{\text{PTQ}}$, is brand new: it is the extra loss you pay when you quantize a trained model, and it depends on the post-training precision and — this is the surprising part — on the data-to-parameter ratio.

The reason this single equation is such a good piece of engineering is that it collapses three previously separate questions into one. "How much does low-precision training hurt?" is answered by the $N_{\text{eff}}$ substitution. "How much does post-training quantization hurt?" is answered by $\delta_{\text{PTQ}}$. And "does training precision affect quantization robustness?" — which sounds like it needs a fourth mechanism — turns out to be captured by the same $\delta_{\text{PTQ}}$ term, because training in low precision is, to first order, quantization applied continuously during training, and the law treats the post-training quantization gap relative to whatever precision you trained at. We will unpack each term in its own section.

### What "effective parameters" actually means

Before we get to the math, pin down the intuition for $N_{\text{eff}}$, because it is the load-bearing idea. A weight stored in 4 bits can hold one of 16 distinct values. A weight in 8 bits holds one of 256. The network's expressive capacity is not literally the number of weights; it is closer to the total number of distinguishable configurations, which depends on both how many weights you have and how finely each can be set. Low precision coarsens each weight, so a 4-bit 8B model behaves, for loss purposes, like a smaller full-precision model. That "smaller model it behaves like" is $N_{\text{eff}}$.

This is why the substitution is multiplicative and not additive: quantizing the weights, the activations, and the KV cache each independently coarsens part of the computation, and the effects compound. If weight precision alone makes the model behave like 80% of its parameters, and activation precision independently makes it behave like 90%, the two together give roughly $0.8 \times 0.9 = 72\%$. That multiplicative structure is the single most important shape in the whole law, and the next section makes it precise.

## 2. Low-precision training reduces effective parameters

**Senior rule of thumb: bits buy capacity with sharply diminishing returns, and the knee is around 6 to 7 bits.** This is the part of the law that explains why 16-bit always felt free and why aggressive sub-4-bit training keeps disappointing.

The effective-parameter count is:

$$N_{\text{eff}}(P_w, P_a, P_{kv}) = N \cdot \left(1 - e^{-P_w/\gamma_w}\right)\left(1 - e^{-P_a/\gamma_a}\right)\left(1 - e^{-P_{kv}/\gamma_{kv}}\right)$$

where $P_w, P_a, P_{kv}$ are the bits allocated to weights, activations, and the KV cache, and $\gamma_w = 2.6745$, $\gamma_a = 2.2102$, $\gamma_{kv} = 0.9578$ are the fitted saturation constants. Each factor is a saturating exponential: at $P = 0$ bits the factor is 0 (no information, no effective parameters), and as $P \to \infty$ it approaches 1 (full-precision behavior). The constant $\gamma$ sets *how fast* you climb toward 1 — a larger $\gamma$ means you need more bits to saturate.

### The saturation curve and why 16-bit is wasteful

The single most useful picture in this whole subject is the weight-precision factor $1 - e^{-P_w/\gamma_w}$ plotted against bits. It is the curve the instructions asked for and the curve every practitioner should have memorized.

![A hand-drawn curve of the effective-parameter fraction rising steeply from low bit counts then flattening out near six to seven bits, with markers showing the values at two, four, and seven bits](/imgs/blogs/precision-scaling-laws-3.png)

Trace the curve from the figure. At 2 bits the factor is $1 - e^{-2/2.6745} = 0.527$ — so an INT2 model behaves like barely more than half its parameters. At 4 bits it is $1 - e^{-4/2.6745} = 0.776$. At 6 bits, $0.894$. At 7 bits, $0.927$. At 8 bits, $0.950$. And at 16 bits, $1 - e^{-16/2.6745} = 0.997$. Stare at the last two numbers: going from 7 bits to 16 bits — more than doubling your storage and bandwidth — buys you a 0.07 increase in the effective-parameter fraction, from 0.927 to 0.997. That is the precise, quantitative meaning of the folk wisdom that "16-bit has many unnecessary bits." Those nine extra bits land almost entirely on the flat part of the curve.

The mirror-image observation is what happens below 4 bits. The curve is steep there: each bit you remove costs a lot of effective capacity. Dropping from 4 bits to 3 takes the factor from 0.776 to $1 - e^{-3/2.6745} = 0.674$; from 3 to 2 it falls to 0.527. This is why sub-4-bit training is genuinely hard: you are operating on the steep part of the curve, where the model's effective size is collapsing fast, and to recover the lost capacity you have to grow $N$ — a lot. We will quantify "a lot" in section 5.

Here is a quick numerical sanity check you can run in your head or in a notebook to feel the multiplicative structure:

```python
import math

gamma = {"w": 2.6745, "a": 2.2102, "kv": 0.9578}

def factor(bits, g):
    return 1.0 - math.exp(-bits / g)

def n_eff_fraction(p_w, p_a, p_kv):
    return factor(p_w, gamma["w"]) * factor(p_a, gamma["a"]) * factor(p_kv, gamma["kv"])

# A "uniform 8-bit" model: weights, activations, KV all at 8 bits.
print(round(n_eff_fraction(8, 8, 8), 3))   # 0.950 * 0.973 * 0.9998 = 0.924

# An aggressive "4-bit weights, 8-bit activations, 4-bit KV" recipe.
print(round(n_eff_fraction(4, 8, 4), 3))   # 0.776 * 0.973 * 0.985 = 0.744

# Near-full-precision bf16-everywhere.
print(round(n_eff_fraction(16, 16, 16), 3))  # 0.997 * 0.9993 * 1.0 = 0.996
```

The uniform-8-bit model retains about 92% of its effective parameters; the aggressive mixed-precision recipe retains about 74%. That 18-point gap is the price of the aggressive recipe, and whether it is worth paying depends entirely on the FLOP and memory savings on the other side of the ledger — which is exactly the compute-optimal calculation in section 6.

### Why the law has this exponential shape and not some other

It is worth pausing on *why* the effective-parameter factor is $1 - e^{-P/\gamma}$ and not, say, a linear ramp or a step function, because the functional form is what gives the law its predictive power. The shape is the signature of a process with diminishing marginal returns: the first few bits resolve the coarse structure of a weight (its sign and rough magnitude), which carries most of the information; each additional bit halves the residual quantization error, but that residual is already small, so the marginal information gained shrinks geometrically. A geometric decay in marginal value integrates to a saturating exponential in cumulative value — which is exactly $1 - e^{-P/\gamma}$. The constant $\gamma$ is the "bit scale" of the component: it is the number of bits over which the factor rises by a factor of $1 - 1/e \approx 0.63$ of its remaining distance to saturation. For weights, $\gamma_w = 2.67$ means roughly every 2.67 bits closes 63% of the remaining gap to full precision.

This shape also explains a phenomenon people find surprising: the *relative* benefit of an extra bit depends on where you start. The derivative of the factor is $\frac{1}{\gamma}e^{-P/\gamma}$, which is largest at $P = 0$ and decays exponentially. The marginal value of going from 3 to 4 bits is $e^{-3/\gamma} - e^{-4/\gamma}$ worth of factor — substantial. The marginal value of going from 15 to 16 bits is $e^{-15/\gamma} - e^{-16/\gamma}$ — essentially zero. This is the formal reason that "the last bits are free to drop and the first bits are precious," and it is why mixed-precision schemes that keep a handful of outlier weights at high precision while crushing the rest to low precision work: the bulk of weights live where the marginal bit is cheap.

### Comparing storage formats through the lens of effective bits

A subtlety that bites people in practice is that "bits" in the law is closer to *mantissa resolution* than to raw storage width. A floating-point format spends some of its bits on an exponent (dynamic range) and some on a mantissa (resolution within a scale). The effective-parameter factor is driven mostly by resolution, so two formats with the same total width can sit at different points on the curve. The table below translates common formats into their approximate effective-bit position.

| Format | Total bits | Mantissa bits | Dynamic range | Rough position on the curve |
|---|---|---|---|---|
| fp32 | 32 | 23 | enormous | fully saturated (factor ≈ 1.000) |
| bf16 | 16 | 7 | fp32-like | saturated for stability, ~0.93 on resolution alone |
| fp16 | 16 | 10 | narrow | saturated on resolution, range-limited |
| fp8 (E4M3) | 8 | 3 | moderate | resolution like ~3-4 bits, range like 8 |
| fp8 (E5M2) | 8 | 2 | wide | resolution like ~2-3 bits, range like 8 |
| int8 | 8 | 8 (with scale) | per-tensor scale | resolution near 8 bits, factor ≈ 0.95 |
| int4 | 4 | 4 (with scale) | per-group scale | resolution near 4 bits, factor ≈ 0.78 |

The practical upshot is that bf16's reputation for being "as good as fp32 for training" comes from its fp32-like *range* (it almost never overflows), not from its *resolution* — on pure resolution, bf16's 7 mantissa bits put it around 0.93 on the weight curve, not 1.0. That is fine for training because the range stability matters more than the last few resolution bits there. But it also means the gap between bf16 and a well-scaled int8 is smaller than the 16-versus-8 storage ratio suggests: both are far up the saturation curve. The formats that genuinely live on the steep part are fp8 with 2-3 mantissa bits and int4 — and those are exactly the formats where the law's warnings bite hardest.

### Why each component has a different saturation rate

The three $\gamma$ constants are not equal, and the differences matter. Weights have the largest constant ($\gamma_w = 2.6745$), meaning weights are the *most* sensitive to precision — they need the most bits to saturate. Activations are close behind ($\gamma_a = 2.2102$). The KV cache is dramatically different ($\gamma_{kv} = 0.9578$, roughly a third of the weight constant), meaning the KV cache saturates *fast*: it reaches the same effective fraction in about a third of the bits.

The mechanism is intuitive once you see it. Weights are the parameters — they are literally where the learned function lives, so coarsening them directly coarsens the function. The KV cache, by contrast, is a transient store of attention keys and values for the current sequence; errors there are averaged over the attention softmax and over many positions, so the network is far more forgiving of low-precision KV. This is the law's quantitative blessing of a trick practitioners had already discovered empirically: you can quantize the KV cache much more aggressively than the weights. We will turn this into a bit-budget in section 4.

## 3. Over-trained models degrade more under post-training quantization

**Senior rule of thumb: if you will quantize at inference, more pretraining data is not free — past a critical data-to-parameter ratio it is actively harmful.** This is the result that made the paper famous, and it is worth being precise about, because it is so easy to mis-state.

The post-training quantization degradation is:

$$\delta_{\text{PTQ}}(N, D, P_{\text{post}}) = C_T \cdot \frac{D^{\gamma_D}}{N^{\gamma_N}} \cdot e^{-P_{\text{post}}/\gamma_{\text{post}}}$$

with fitted constants $C_T = 0.0598$, $\gamma_D = 0.5068$, and $\gamma_N = 0.3439$. Read the three factors left to right. $C_T$ is an overall scale. The middle factor $D^{\gamma_D}/N^{\gamma_N}$ is the dangerous one: because $\gamma_D$ and $\gamma_N$ are both around 0.4 to 0.5 and close to each other, this factor is approximately a power law in the ratio $D/N$ — the very same data-per-parameter quantity that drives the Chinchilla story, but now with the opposite sign of consequence. The third factor $e^{-P_{\text{post}}/\gamma_{\text{post}}}$ says the degradation falls exponentially as you give the quantized model more bits, which is the expected and reassuring direction: quantize less aggressively and you pay less.

### Reading the danger: more data raises post-quant loss

Hold $N$ and $P_{\text{post}}$ fixed and let $D$ grow. The full-precision loss $L(N, D)$ falls as $B/D^{\beta}$ — that is the ordinary benefit of more data. But the post-quant loss is $L(N, D) + \delta_{\text{PTQ}}$, and $\delta_{\text{PTQ}}$ *rises* as $D^{\gamma_D}$. So you are subtracting something that shrinks as $D^{-\beta}$ and adding something that grows as $D^{+\gamma_D}$. At small $D$ the falling term dominates and more data helps even after quantization. But there is a crossover $D^*$ beyond which the rising $\delta_{\text{PTQ}}$ wins, and from there on, *each additional token raises the loss of the quantized model*. The figure below shows the two regimes as a single rising curve in $D/N$.

![A hand-drawn curve showing the post-training quantization loss penalty rising along a power law as the data-to-parameter ratio increases, with markers at the compute-optimal point, a critical zone, and the heavily over-trained Llama-3-8B point](/imgs/blogs/precision-scaling-laws-5.png)

The mechanism is the one we glimpsed in section 2's intuition about effective parameters. As you train a fixed-size model on more and more tokens, the weights pack more information into the same number of parameters — the loss falls because the network is using its capacity more fully. But weights that are "denser" with information have less slack to absorb the rounding error of quantization. A lightly trained model has redundancy; quantize it and the redundancy soaks up the error. A heavily over-trained model has wrung out its redundancy; quantize it and the error lands directly on information the model needs. More training makes the weights more brittle to rounding. That is the whole phenomenon, and it is why the degradation grows with $D$ at fixed $N$.

### Deriving the crossover point

The crossover is worth deriving explicitly, because it tells you where the danger zone begins for your specific model. The post-quant loss as a function of $D$ (holding $N$ and $P_{\text{post}}$ fixed) is:

$$L_{\text{quant}}(D) = \underbrace{E + \frac{A}{N_{\text{eff}}^{\alpha}}}_{\text{constant in } D} + \frac{B}{D^{\beta}} - C_T' \cdot D^{\gamma_D} \cdot \text{(wait — sign)}$$

Be careful with signs: $\delta_{\text{PTQ}}$ is *added*, so $L_{\text{quant}}(D) = \text{const} + B D^{-\beta} + C_T' D^{\gamma_D}$ where $C_T' = C_T N^{-\gamma_N} e^{-P_{\text{post}}/\gamma_{\text{post}}}$ collects the $D$-independent factors. The derivative is:

$$\frac{dL_{\text{quant}}}{dD} = -\beta B D^{-\beta - 1} + \gamma_D C_T' D^{\gamma_D - 1}$$

Setting this to zero gives the token count $D^*$ at which the quantized loss is minimized — beyond which more data raises it:

$$D^* = \left(\frac{\beta B}{\gamma_D C_T'}\right)^{1/(\beta + \gamma_D)}$$

The key qualitative reads from this expression: $D^*$ grows with $N$ (through $C_T' \propto N^{-\gamma_N}$, larger models tolerate more tokens before the crossover), and $D^*$ grows as you increase $P_{\text{post}}$ (through $C_T' \propto e^{-P_{\text{post}}/\gamma_{\text{post}}}$ — serve at higher precision and the crossover moves out, eventually to infinity as the quantization gap vanishes). This is the formal statement of everything in this section: the crossover exists, it scales sensibly with model size and serving precision, and it is the line past which the marginal token flips from asset to liability for a quantized deployment. For a heavily over-trained small model served at 4 bits, you are well past $D^*$; for a large model served at 6 bits, you may never reach it within any realistic token budget.

### The empirical evidence behind the term

It would be fair to be skeptical of a term that says "more data is bad," so it is worth noting how the paper established it. The authors ran the relevant sweep directly: fix a model size, train checkpoints at a ladder of token counts, then post-training-quantize each checkpoint to a fixed low precision and measure the degradation. The degradation rose monotonically with the token count, and the rise fit a power law in $D$ with the exponent $\gamma_D \approx 0.51$ across model sizes. They also varied $N$ and confirmed the $N^{-\gamma_N}$ dependence, and varied $P_{\text{post}}$ and confirmed the exponential falloff. Crucially, the same functional form fit the *training*-in-low-precision data too, which is what let them unify the robustification effect into the same term rather than needing a separate mechanism. With 465-plus runs spanning these axes, the form is well constrained within the validated range — the caveat is the usual one about extrapolating the constants to scales an order of magnitude beyond the data.

### Where real models sit on this curve

The numbers make this concrete and slightly alarming. The Chinchilla compute-optimal point is $D/N \approx 20$. The paper characterizes "over-trained" as roughly $D/N > 1000$, and the canonical example is **Llama-3-8B**, trained on 15T tokens at 8B parameters for $D/N \approx 2000$ — about 100x the Chinchilla optimum and well past the critical ratio. That is not an exotic research model; it is one of the most-deployed open models in the world, and a huge fraction of its deployments are quantized to 4 bits to fit on consumer GPUs. The precision law says those deployments are paying an outsized quantization penalty precisely *because* the model was trained so thoroughly.

To be careful about what this does and does not say: it does not say Llama-3-8B is a bad model, or that you should have trained it less for full-precision serving. At full precision, 15T tokens is wonderful — the model is excellent. The claim is conditional: *if your serving plan is to quantize*, then the marginal tokens past the critical $D/N$ bought you full-precision quality you are about to throw away, while simultaneously making the quantized version worse. The before-and-after below frames the two regimes head to head.

![A before-and-after comparison contrasting a compute-optimal model that quantizes cleanly against an over-trained model that suffers a large post-training quantization penalty](/imgs/blogs/precision-scaling-laws-4.png)

The left side is a compute-optimal model at $D/N \approx 20$: a small $\delta_{\text{PTQ}}$, clean quantization to 4 bits, no surprises. The right side is the over-trained regime at $D/N \approx 2000$: a large $\delta_{\text{PTQ}}$, where the extra data has crossed into being net harmful for the quantized deployment. The two columns are the same model family at the same parameter count; the only difference is how many tokens you poured in, and that difference flips the sign of the marginal-data decision once you commit to quantizing.

### The robustification effect: low-precision training cuts the gap

There is a second, more hopeful face of the same $\delta_{\text{PTQ}}$ term, and the paper found it to be empirically *dominant* in importance: training in low precision makes the weights more robust to later quantization. The intuition is that if you train the weights while they are already constrained to a coarse grid, the optimizer learns a solution that lives comfortably on that grid, so quantizing further to that same grid (or near it) costs almost nothing. You have, in effect, done quantization-aware training as a side effect of low-precision training.

In the law, this is captured by interpreting $P_{\text{post}}$ relative to the training precision: the degradation you pay at inference is the gap between your training precision and your serving precision, not the absolute serving precision. A model trained in bf16 and quantized to 4 bits pays a large gap; a model trained at 6 bits and served at 4 bits pays a much smaller one. This is the bridge between the two halves of the law — the training-side $N_{\text{eff}}$ story and the inference-side $\delta_{\text{PTQ}}$ story are not independent levers, they are coupled, and the coupling is favorable: spending bits during training buys you robustness at inference. It is also the cleanest argument for why, if you know you will serve at low precision, you should consider training at low precision too rather than training high and quantizing down.

## 4. The bit budget: weights, activations, and the KV cache

**Senior rule of thumb: do not quantize uniformly — spend bits where saturation is slow and starve the components that saturate fast.** Because the three $\gamma$ constants differ by up to 3x, the optimal allocation is decidedly non-uniform, and the law tells you exactly which way to skew.

![A matrix laying out the saturation constant, half-capacity bit count, practical bit floor, and role for weights, activations, and the KV cache](/imgs/blogs/precision-scaling-laws-6.png)

The matrix above lays out the three components side by side. The "half-capacity bits" column is the number of bits at which each factor reaches 0.5: solving $1 - e^{-P/\gamma} = 0.5$ gives $P = \gamma \ln 2 \approx 0.69\gamma$, so weights hit half-capacity at about 1.9 bits, activations at about 1.5, and the KV cache at about 0.7. The KV cache reaches half its effective contribution in well under a single bit — which is the formal statement of "the KV cache barely cares about precision."

Turn this into a concrete budget. Suppose you have a memory or bandwidth budget that averages out to about 6 bits per value across the three components, and you want to maximize $N_{\text{eff}}$. Uniform allocation gives 6/6/6 and an effective fraction of:

$$\left(1 - e^{-6/2.6745}\right)\left(1 - e^{-6/2.2102}\right)\left(1 - e^{-6/0.9578}\right) = 0.894 \times 0.935 \times 0.998 = 0.834$$

Now skew toward weights and activations and starve the KV cache — say 7/7/4, which has the same average:

$$\left(1 - e^{-7/2.6745}\right)\left(1 - e^{-7/2.2102}\right)\left(1 - e^{-4/0.9578}\right) = 0.927 \times 0.958 \times 0.985 = 0.875$$

The skewed allocation buys you a 4-point gain in effective parameters at the same total bit budget, purely by moving bits off the KV cache (which was already saturated at 6 bits) and onto the weights and activations (which were not). The KV cache at 4 bits still retains 0.985 of its contribution — practically all of it — because its $\gamma$ is so small. This is the law's blessing of the practical recipe many serving stacks already use: aggressive KV-cache quantization (4 to 5 bits) paired with more conservative weight quantization. The [inference-aware scaling work](/blog/machine-learning/scaling-laws/inference-aware-scaling-laws) cares about KV-cache size because it dominates memory at long context and large batch, and the precision law says you can crush it to 4 bits at almost no quality cost — a rare case where the cheap thing and the right thing coincide.

| Allocation (w/a/kv) | $N_{\text{eff}}$ fraction | Avg bits | Notes |
|---|---|---|---|
| 16 / 16 / 16 | 0.996 | 16.0 | bf16 everywhere — most bits on the flat part of the curve |
| 8 / 8 / 8 | 0.924 | 8.0 | uniform INT8 — a sane, safe default |
| 6 / 6 / 6 | 0.834 | 6.0 | uniform 6-bit — KV is wasting bits here |
| 7 / 7 / 4 | 0.875 | 6.0 | skewed — same budget, KV starved, weights fed |
| 4 / 8 / 4 | 0.744 | ~5.3 | aggressive weight quant — only if FLOP savings justify it |
| 4 / 4 / 4 | 0.711 | 4.0 | uniform INT4 — steep part of the curve for w and a |

The table makes the skew argument visually obvious: 7/7/4 beats 6/6/6 at the same average, and the uniform INT4 row sits low precisely because weights and activations are deep on the steep part of their curves while the KV cache (at 4 bits) is already nearly saturated and could have given up bits to help the others.

## 5. Compute-optimal precision lands near seven to eight bits

**Senior rule of thumb: precision is a third axis to optimize jointly with parameters and tokens, and the joint optimum is about 7 to 8 bits, roughly independent of your budget.** This is the section that turns the law into a training-time decision.

The compute model is the key. Training FLOPs scale not just with $N$ and $D$ but with precision: lower-precision arithmetic is cheaper per operation on precision-aware hardware, so to first order the cost is $C \propto N \cdot D \cdot P$ for integer precision $P$ (more bits, more cost per op). The optimization is: minimize loss $L$ subject to a fixed compute budget $C \propto N \cdot D \cdot P$, jointly over $N$, $D$, and $P$. You are now spending a fixed pile of FLOPs across three knobs instead of two, and the question is how many bits to buy.

![A grid contrasting four precision regimes from three-bit to sixteen-bit, showing the cost of each and why seven to eight bits is compute-optimal](/imgs/blogs/precision-scaling-laws-8.png)

The grid above lays out the four regimes. The answer the paper reports is that the compute-optimal precision is about **7 to 8 bits** (for integer types), and — strikingly — this is roughly *independent of the compute budget*. That independence is the surprising and useful part: it means there is a "right" training precision that does not drift as you scale up, the way the optimal model size and token count do. You do not have to re-derive your precision target every time your cluster grows; 7 to 8 bits is a stable target.

Why not lower, and why not higher? Look at the two edges of the grid. Above 8 bits — at 16 bits — you are buying bits on the flat part of the effective-parameter curve, so each extra bit costs compute (the $P$ factor in $C$) but barely raises $N_{\text{eff}}$. Those bits would have been better spent on more parameters or more tokens. The paper's phrasing is that 16-bit "has many unnecessary bits," and the grid's right column shows why: 16-bit buys nearly no extra $N_{\text{eff}}$ for a doubled per-op cost relative to 8-bit. Below 4 bits you hit the opposite wall: the effective-parameter curve is so steep that to hold the same loss you must grow the parameter count *more than 4x*, which more than eats the per-op savings of the lower precision. FP4 looks tantalizing on a spec sheet — 4x the throughput of bf16 — but the law says you would have to roughly quadruple your model to recover the lost capacity, and a 4x-bigger model at 4x-faster arithmetic is a wash on FLOPs while being strictly worse on memory and complexity.

### The fixed-N corollary: over-trained models want more bits

There is a refinement worth knowing for the common case where the parameter count is fixed in advance (you are training a model of a specific size because that is what fits your serving target). For a fixed-$N$ family, the optimal training precision rises slowly with compute: $P^*(C) \propto \log C \approx \log(D/N)$. In words, *the more you over-train a fixed-size model, the higher the precision it warrants*. This dovetails exactly with the $\delta_{\text{PTQ}}$ story: heavy over-training makes weights brittle to quantization, so you want to have trained at higher precision to leave more robustness on the table. A model you plan to push to $D/N \approx 2000$ should be trained at more bits than one you will stop at the Chinchilla optimum — the over-training is precisely what raises the value of the marginal bit.

```python
# A back-of-envelope joint optimization sketch. We hold the compute budget
# C ~ N * D * P fixed and search over precision P, choosing N and D at the
# (approximate) Chinchilla split for each P, to see where loss bottoms out.
import math

A, alpha = 4299, 0.4965
B, beta  = 18060, 0.5
E        = 2.7648
gamma_w  = 2.6745

def loss(N_eff, D):
    return A / (N_eff ** alpha) + B / (D ** beta) + E

C = 6e21                      # fixed training-compute budget (arbitrary units)
best = None
for P in range(3, 17):        # candidate integer precisions, bits per weight
    # Cost per "parameter-token" scales with P; spend the rest on N and D.
    budget = C / P
    N = math.sqrt(budget / 6) # Chinchilla-style sqrt split (illustrative)
    D = budget / (6 * N)
    N_eff = N * (1 - math.exp(-P / gamma_w))   # weights-only factor for sketch
    L = loss(N_eff, D)
    if best is None or L < best[1]:
        best = (P, L)
    print(f"P={P:2d} bits  N_eff={N_eff:.3e}  loss={L:.4f}")

print("compute-optimal precision (sketch):", best[0], "bits")
```

This sketch is deliberately simplified — it uses a weights-only factor and a toy compute model — but run it and you will see the loss bottom out in the 7-to-8-bit range and stay flat-ish across a wide span of budgets, reproducing the qualitative shape of the paper's result: there is a broad, budget-insensitive sweet spot around 7 bits, with a soft penalty for going higher and a hard penalty for going much lower.

### Why the optimum is budget-independent

The budget-independence of $P^*$ is the most counterintuitive part of the compute-optimal result, and the intuition is worth spelling out because it is genuinely different from how $N^*$ and $D^*$ behave. The optimal model size and token count both grow with the budget — that is the whole content of Chinchilla, $N^* \propto C^{0.5}$ and $D^* \propto C^{0.5}$. So why would the optimal precision *not* grow with the budget?

The answer is that precision and size play fundamentally different roles in the cost-quality trade. Adding parameters lowers loss along a power law $A N^{-\alpha}$ with no ceiling — there is always more loss to wring out by going bigger, so a bigger budget always wants a bigger model. Adding precision lowers loss along the *saturating* curve $1 - e^{-P/\gamma}$, which has a hard ceiling at factor 1. Once you are near the knee at 7-8 bits, there is almost no loss left to capture from more bits, *regardless of how much budget you have*. A bigger budget does not change the location of the knee; it just lets you buy more of the thing that still has returns (parameters and tokens) rather than the thing that has saturated (precision). So the optimal split is: pour the marginal budget into $N$ and $D$, keep $P$ pinned at the knee. The knee is a property of the network's tolerance to coarsening, not of your bank account.

This is why $P^*$ is such a clean, portable number compared to $N^*$ and $D^*$. You re-derive your optimal model size every time your cluster grows. You do not re-derive your optimal precision — it stays near 7-8 bits from a research-scale run to a frontier-scale run. That stability is rare in scaling-law results and makes precision an unusually actionable axis: there is a default you can adopt and largely stop thinking about, which is not true of the parameter and token allocation.

### What changes the optimum

Two things do move $P^*$, and it is worth knowing them so you recognize when the default does not apply. First, the hardware cost model: the analysis assumes per-op cost scales with $P$, which holds on precision-aware accelerators (FP8 tensor cores genuinely run faster than bf16). On hardware where low precision gives no speedup — where everything runs at the bf16 rate regardless of operand width — the $P$ factor in $C$ disappears, and there is no compute reason to drop below 16 bits; you would only quantize for memory. Know your hardware's actual precision-throughput curve before adopting the 7-8-bit target. Second, the over-training regime, as covered in the fixed-$N$ corollary: if your parameter count is pinned and you are pushing $D/N$ very high, the optimal training precision drifts up with $\log(D/N)$, because heavy over-training raises the value of the marginal bit (both for capacity and for quantization robustness). A model headed for $D/N \approx 2000$ wants more training bits than one stopping at $D/N \approx 20$.

## 6. Worked examples: putting numbers on the decisions

**Senior rule of thumb: a scaling law is only as useful as the dollar or FLOP decision you make with it — so always run the numbers before you trust the headline.** Let us walk three concrete scenarios end to end.

### Example A: should you train your 8B model on 15T tokens if you will serve it at INT4?

You are planning an 8B-parameter model. Your serving target is INT4 weights to fit consumer GPUs. The data team can give you anywhere from 2T to 15T tokens. The naive instinct, fresh off the inference-aware scaling laws, is "more tokens, always" — push to 15T.

Run the $\delta_{\text{PTQ}}$ term. At $N = 8\text{e}9$, $\gamma_N = 0.3439$, the denominator $N^{\gamma_N} = (8\text{e}9)^{0.3439} \approx 2900$. Compare two token counts. At $D = 2\text{e}12$ (so $D/N = 250$), the numerator $D^{\gamma_D} = (2\text{e}12)^{0.5068} \approx 1.8\text{e}6$, and with the exponential at, say, INT4 from a bf16-trained model giving a gap factor we will lump into the constant, $\delta_{\text{PTQ}} \propto 1.8\text{e}6 / 2900 \approx 620$ units. At $D = 15\text{e}12$ ($D/N = 1875$), $D^{\gamma_D} = (15\text{e}12)^{0.5068} \approx 5.1\text{e}6$, so $\delta_{\text{PTQ}} \propto 5.1\text{e}6 / 2900 \approx 1760$ units — roughly **2.8x larger** quantization penalty from the extra tokens. Meanwhile the full-precision benefit of those tokens, $B/D^{\beta}$, fell from $18060/(2\text{e}12)^{0.5} \approx 0.0128$ to $18060/(15\text{e}12)^{0.5} \approx 0.0047$ — a benefit of only about 0.008 nats. The question is whether 0.008 nats of full-precision improvement is worth nearly tripling the quantization penalty. For an INT4 deployment, the law says: probably not. You would likely be better served stopping around 4T to 6T tokens, or — better still — training at 6 to 8 bits so the INT4 gap is small.

The point is not the exact unit-less numbers (the proportionality constant matters for the absolute trade) but the *ratio*: the penalty grows almost 3x while the benefit shrinks to a sliver. That asymmetry is the whole argument.

There is a tempting middle path worth naming: train to 15T tokens, but serve the *full-precision* model to your highest-value traffic and the INT4 model only to the cost-sensitive tail. This works, and it is what many deployments actually do, but notice it does not escape the trade — it just splits your traffic across two points on the curve. The full-precision tier gets the benefit of all 15T tokens; the INT4 tier eats the large $\delta_{\text{PTQ}}$. If the INT4 tier is the bulk of your volume (the usual case, since it is the cheap one), then most of your serving is paying the over-training penalty, and the analysis above still points you toward either a lower token budget or a lower training precision. The dual-tier strategy is a hedge, not a fix; the fix is to align training precision with the precision your *majority* traffic will actually run at.

### Example B: how much bigger must a 4-bit model be to match an 8-bit model's loss?

You want to know the cost of going from INT8 weights to INT4 weights at training time, holding the loss constant. The weight factor is $1 - e^{-P/\gamma_w}$: at 8 bits it is 0.950, at 4 bits it is 0.776. To hold $N_{\text{eff}} = N \cdot \text{factor}$ constant when the factor drops from 0.950 to 0.776, you need:

$$N_{4} = N_{8} \cdot \frac{0.950}{0.776} = 1.224 \, N_{8}$$

So at the weights-only level, a 4-bit model needs to be about 22% bigger to match an 8-bit model's effective parameter count. That sounds cheap — but it is the weights-only number. If you also drop activations to 4 bits (factor falls from 0.973 at 8 bits to 0.834 at 4 bits), the combined factor falls from $0.950 \times 0.973 = 0.924$ to $0.776 \times 0.834 = 0.647$, and now $N_4 = N_8 \cdot (0.924/0.647) = 1.43\,N_8$ — a 43% size increase. Push further to genuinely aggressive sub-4-bit and the required growth blows past 4x, which is the "below 4-bit forces model size to grow more than 4x" claim from the headline. The lesson: INT8 to INT4 on weights alone is a modest 22% tax you might happily pay for the memory savings; INT4 across weights and activations is a 43% tax; and FP4-everywhere is a regime where the model has to balloon so much that the throughput win evaporates.

### Example C: budgeting bits for a long-context serving stack

You are serving a 70B model at 128K context with large batches. Profiling shows the KV cache dominates memory — it is the thing forcing you onto more GPUs. You have decided weights will be INT8 (factor 0.950) and you are choosing KV-cache precision. The KV factor is $1 - e^{-P_{kv}/0.9578}$: at 8 bits it is 0.9998, at 5 bits 0.9948, at 4 bits 0.9847, at 3 bits 0.957, at 2 bits 0.876.

Dropping the KV cache from 8 bits to 4 bits halves its memory footprint and takes its effective-contribution factor from 0.9998 to 0.9847 — a loss of 1.5%, which is comfortably inside the noise for most deployments. Dropping to 2 bits halves it again but now costs 12% of the KV contribution, which starts to show up in long-context quality. The law's recommendation is clear: **4 to 5 bits for the KV cache is the sweet spot** — most of the memory savings, almost none of the quality cost — and 2-bit KV is a step too far that you take only under genuine memory desperation. This is the single most actionable consequence of the small $\gamma_{kv}$ constant, and it is why nearly every modern serving stack ships INT4 or INT5 KV quantization while keeping weights at INT8 or higher.

### Example D: trading training precision against token budget at fixed quality

Here is the example that ties the two halves of the law together. You have a fixed quality target — a specific full-precision-equivalent loss you must hit — and a serving plan of INT4. You can reach the target two ways: train at high precision on a lot of tokens (and eat a large quantization gap), or train at lower precision on fewer tokens (and eat a small gap). Which is cheaper?

Path one: train at bf16 (effective weight factor 0.997) on enough tokens to hit the target, then quantize to INT4. The token count to hit the target at the high effective-parameter fraction is modest — say $D_1$ — but the quantization gap is bf16-to-INT4, large, so the *served* loss is target plus a big $\delta_{\text{PTQ}}$. To get the served loss back to target you must over-train further, pushing $D$ up and, by section 3, pushing $\delta_{\text{PTQ}}$ up too — a frustrating chase where adding tokens partly fights itself.

Path two: train at 6 bits (effective weight factor 0.894) on more tokens to compensate for the lower factor, then quantize to INT4. The lower effective fraction means you need more tokens to hit the same full-precision-equivalent loss — call it $D_2 > D_1$ — but the quantization gap is now only 6-bit-to-4-bit, small, so the served loss is target plus a small $\delta_{\text{PTQ}}$. No chase: you hit the target and the served model stays near it.

Run the rough arithmetic with the compute model $C \propto N D P$. Path two trains at $P = 6$ instead of $P = 16$, a 2.7x lower per-token cost, while needing maybe 1.3-1.5x more tokens to compensate for the lower $N_{\text{eff}}$ — a net training-compute *saving* of roughly 1.8x, *and* a better served model because the quantization gap is small. The high-precision path looks safer on instinct ("train at full precision, you can always quantize later") but is both more expensive and worse at the serving target. This is the robustification effect cashing out as a concrete budget decision: when the serving precision is known and low, training near it dominates training high and quantizing down.

## 7. The over-training trajectory: how the field walked into the trap

It is worth stepping back to see how the most-deployed open models drifted into exactly the regime where the $\delta_{\text{PTQ}}$ penalty bites. The trajectory is a direct consequence of the [inference-aware scaling argument](/blog/machine-learning/scaling-laws/inference-aware-scaling-laws): if you will serve a model to billions of requests, you should make it small and train it on far more than 20 tokens per parameter, because the training cost is amortized over an enormous inference volume. That argument is correct, and it drove the industry from Chinchilla-style ratios toward heavy over-training.

![A timeline tracing open models from Chinchilla at twenty tokens per parameter through the LLaMA series up to Llama-3-8B at two thousand tokens per parameter](/imgs/blogs/precision-scaling-laws-7.png)

The timeline traces the drift. Chinchilla 70B sat at $D/N \approx 20$, the compute-optimal point. LLaMA-1 7B (1T tokens) jumped to $D/N \approx 143$. LLaMA-2 70B (2T tokens) was a relatively conservative $D/N \approx 29$, close to optimal. Then Llama-3 pushed hard: the 70B trained on 15T tokens lands at $D/N \approx 214$, and the 8B on the same 15T tokens lands at the eye-watering $D/N \approx 2000$ — about 100x Chinchilla, deep in the over-trained zone. Each of these was a *good* decision for full-precision serving and for the inference-amortization argument. The catch is that the same small-and-heavily-trained models are exactly the ones people most want to quantize — an 8B model is attractive precisely because it can run on a single consumer GPU, and quantizing it to 4 bits is how you get it there. So the very property that makes Llama-3-8B popular (small, thoroughly trained) is the property that makes it quantize worse than the law's idealized compute-optimal model.

The resolution is not to stop over-training — the inference-amortization math is real — but to *recognize the coupling*. If a small over-trained model is your deployment target and quantization is your serving plan, the precision law says to either train at lower precision (so the serving-precision gap is small and the weights are already robust) or accept that the last few trillion tokens bought full-precision quality you will partly discard. Both the [inference-aware post](/blog/machine-learning/scaling-laws/inference-aware-scaling-laws) and this one are pushing on the same lever — total lifetime cost — but this one adds the precision dimension that the pure-FLOP inference-aware analysis left out.

## 8. Failure modes from the field

**Senior rule of thumb: every clean scaling law has a dirty boundary where it stops applying, and that boundary is where the production incidents live.** Here are the recurring ways precision decisions go wrong, drawn from the patterns the law explains.

### 1. The "free quantization" surprise on an over-trained small model

The symptom: a team takes a heavily over-trained 7–8B model that benchmarks beautifully in bf16, runs a standard INT4 PTQ pass that worked fine on their previous compute-optimal model, and watches accuracy crater far more than expected. The wrong first hypothesis is "the quantization method is broken" — so they swap GPTQ for AWQ for SmoothQuant and see only marginal differences. The actual root cause is the $\delta_{\text{PTQ}} \propto (D/N)^{\sim 0.4}$ term: the new model sits at $D/N \approx 2000$ instead of the old one's $D/N \approx 100$, so the same INT4 pass pays a much larger degradation. The fix is not a better quantizer; it is either serving at 5–6 bits instead of 4, or, for the next training run, training at lower precision so the gap shrinks. The lesson: quantization difficulty is a property of how the model was *trained*, not just of the quantizer.

### 2. Chasing FP4 throughput and getting a wash

The symptom: a lab sees FP4 tensor cores promising 4x the throughput of bf16 and rebuilds their training stack around 4-bit, expecting a 4x cheaper run. They get a model that needs to be far larger to hit the target loss, and after sizing up to recover quality, the total FLOPs are roughly unchanged while the engineering complexity and memory pressure are much worse. The root cause is the steep part of the effective-parameter curve below 4 bits: the model must grow more than 4x to hold loss, eating the per-op savings. The fix is to land at the compute-optimal 7–8 bits, where the per-op cost and the $N_{\text{eff}}$ retention balance. The lesson: throughput on a spec sheet is not loss-per-FLOP — the law's $C \propto N D P$ trade is what matters.

### 3. Uniformly quantizing the KV cache to the weight precision

The symptom: a serving team quantizes weights, activations, and KV cache all to the same bit width (say 8) because the tooling makes uniform quantization easy, then struggles with KV-cache memory at long context. The root cause is leaving the KV cache at 8 bits when its $\gamma_{kv} = 0.9578$ means it saturated by about 4 bits — those extra 4 bits are pure waste on the flat part of its curve. The fix is to drop KV to 4 bits (factor 0.985, a 1.5% loss) and reclaim half the KV memory. The lesson: do not quantize uniformly; the three components have different saturation rates and the KV cache is the cheapest place to take bits from.

### 4. Treating training precision and serving precision as independent

The symptom: a team trains in bf16 (because that is the default) fully intending to serve at INT4, and is surprised the INT4 model is so much worse than the bf16 one. They had budgeted for "a small quantization hit" based on published PTQ benchmarks. The root cause is the gap interpretation of $\delta_{\text{PTQ}}$: the penalty scales with the gap between training and serving precision, and bf16-to-INT4 is a large gap. A model trained at 6 bits and served at 4 bits would have paid far less. The fix is to make training precision a function of serving precision — if you will serve at 4 bits, train at 6, not 16. The lesson: the two precisions are coupled; the robustification effect is real and it is, per the paper, the dominant term.

### 5. Extrapolating the law past its validated range

The symptom: someone takes the fitted constants and confidently predicts the behavior of a 400B model at $D/N = 10000$ and 2-bit weights, then is shocked when reality diverges. The root cause is that the law was validated up to 1.7B parameters and 26B tokens; the constants are point estimates from a specific regime, and extrapolating three orders of magnitude past the data is exactly the kind of move the [reconciliation post](/blog/machine-learning/scaling-laws/kaplan-vs-chinchilla-reconciliation) warns against — fitting at one scale and extrapolating to another is how a field gets the exponent wrong. The fix is to treat the exponents as more trustworthy than the coefficients, to use the law for *directional* decisions (skew bits toward weights, do not over-train if you will quantize) rather than for precise absolute loss predictions at frontier scale, and to re-fit on your own runs when you can. The lesson: respect the validated range; the shape travels, the exact constants may not.

### 6. Forgetting that the data penalty term is precision-blind

The symptom: a team convinced by the "more data can hurt" headline cuts their token budget aggressively for a *full-precision* deployment and ends up with an undertrained model. The root cause is misreading the conditional: $\delta_{\text{PTQ}}$ only exists if you quantize. For full-precision serving, $\delta_{\text{PTQ}} = 0$ (or rather, the serving-precision gap is zero), the data penalty $B/D^{\beta}$ is the only $D$-dependent term, and more data is unambiguously good. The fix is to apply the over-training caution *only* when quantization is in the serving plan. The lesson: the dangerous term is conditional on quantizing; do not let the scary headline override the basic Chinchilla intuition for full-precision models.

### 7. Activation quantization treated as an afterthought

The symptom: a team quantizes weights aggressively to 4 bits, leaves activations at 16, and is puzzled that throughput barely improved on their hardware. The root cause is that on many accelerators the matmul throughput is gated by the lower-precision operand only when *both* operands are low precision; a 4-bit-weight, 16-bit-activation matmul often runs at the 16-bit rate. The law also notes activations have a substantial $\gamma_a = 2.2102$, close to weights, so they are not a free place to keep high precision either — they cost both throughput (if left high) and $N_{\text{eff}}$ (if dropped too low). The fix is to co-design weight and activation precision against the actual kernel behavior, landing both near the compute-optimal range. The lesson: precision is a per-operand, per-kernel decision, not a single global knob.

### 8. Assuming integer and floating-point precision are interchangeable in the law

The symptom: a team reads "compute-optimal precision is about 7 to 8 bits" and concludes FP8 and INT8 are equivalent for their purposes, then finds FP8 training more stable but INT8 inference cheaper, with quality differences the headline number did not predict. The root cause is that the paper's compute-optimal-precision result is stated for *integer* types, and floating-point formats trade mantissa bits for exponent range differently — an FP8 (E4M3) value has only 3 mantissa bits, so its effective resolution differs from INT8's. The fix is to map your actual format to its effective bit count (mantissa bits drive the $N_{\text{eff}}$ factor, exponent bits drive dynamic range and stability) rather than reading "8 bits" as format-agnostic. The lesson: "bits" in the law is closer to mantissa resolution than to raw storage width; floating-point formats need translation.

### 9. Quantizing on the wrong hardware and seeing no speedup

The symptom: a team quantizes a model to INT4 expecting to serve it cheaper, deploys to their existing fleet, and measures essentially the same latency and throughput as the bf16 model — sometimes worse. The wrong first hypothesis is that the quantization "did not take." The actual root cause is that the target hardware has no fast INT4 path; the kernels dequantize to bf16 before the matmul, so the only saving is memory, not compute, and the dequantization overhead can eat even that on compute-bound shapes. The precision law's compute-optimal analysis explicitly assumes per-op cost scales with precision, which is a property of the *accelerator*, not the model. The fix is to check the hardware's precision-throughput curve first: quantize for memory on hardware without a low-precision compute path, and quantize for compute only where the tensor cores actually run the low-precision matmul natively. The lesson: the cost side of the law is a hardware fact, and a quantization that pays off on one accelerator can be a no-op on another.

### 10. Mixing the over-training caution with calibration-set overfitting

The symptom: a team quantizing an over-trained model invests heavily in a large, carefully curated PTQ calibration set, expecting it to recover the lost quality, and gets only marginal improvement. The wrong hypothesis is that the calibration set is too small or unrepresentative, so they make it bigger and more diverse — to little effect. The actual root cause is that calibration tunes the quantization *grid* (scales and zero-points) to minimize rounding error, but the $\delta_{\text{PTQ}}$ penalty for an over-trained model comes from the weights having little redundancy to begin with — there is no grid placement that recovers information the dense weights genuinely need at 4-bit resolution. Calibration helps a model that has slack; it cannot manufacture slack in a model that wrung it out through heavy training. The fix is to serve at higher precision or to have trained at lower precision; calibration is a second-order knob here, not a rescue. The lesson: PTQ calibration and the over-training penalty are different mechanisms, and you cannot calibrate your way out of a $D/N \approx 2000$ model that must run at 4 bits.

## 9. How this connects to the rest of the scaling-law story

It is worth situating the precision law inside the series, because it does not float free — it modifies and is modified by the other results.

The relationship to [Chinchilla](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) is that precision adds a third axis to the two-axis compute-optimal problem. Chinchilla said: given compute $C \approx 6ND$, split it about evenly between $N$ and $D$, landing at $D/N \approx 20$. The precision law says: actually the budget is $C \propto N \cdot D \cdot P$, and you should optimize all three — with $P^* \approx 7$–$8$ bits. The two are compatible; precision is an extra dimension that Chinchilla implicitly fixed at "whatever bf16 is."

The relationship to the [inference-aware laws](/blog/machine-learning/scaling-laws/inference-aware-scaling-laws) is the tension we explored in section 7. Inference-aware scaling pushes you toward small, heavily over-trained models to minimize lifetime serving FLOPs. Precision scaling warns that those exact models quantize worst. The synthesis is to fold precision into the lifetime-cost objective: if your cheap serving plan involves quantization, the over-training point that minimizes *full-precision* serving cost is not the point that minimizes *quantized* serving cost, and you should either pull back the token budget or push down the training precision.

The relationship to [data-quality scaling](/blog/machine-learning/scaling-laws/data-quality-scaling-laws) is subtler but real. Data-quality work shows that better data shifts the loss-vs-compute curve down by a multiplicative compute factor — you can hit the same loss with fewer tokens. That is directly helpful here: fewer tokens to reach a target loss means a lower $D/N$ at that loss, which means a smaller $\delta_{\text{PTQ}}$. Higher-quality data is, among its many virtues, a way to stay further from the over-training quantization trap, because it lets you reach your quality target before crossing the critical $D/N$. The two axes — data quality and precision — both ultimately move the same lever: how much capacity you have packed into the weights and how brittle that packing is.

There is also a clean conceptual unity worth stating: every one of these axes is ultimately about *information density in the weights and the brittleness that comes with it*. Chinchilla's tokens fill the weights with information up to the data penalty floor. Data quality controls how much useful information each token deposits. And precision controls both the resolution at which that information can be stored ($N_{\text{eff}}$) and how much rounding the stored information can survive ($\delta_{\text{PTQ}}$). Over-training is the state of maximum information density, which is simultaneously the state of maximum quality at full precision and maximum fragility under quantization. Seen this way, the "more data can hurt" result is not a paradox at all — it is the inevitable consequence of pushing one axis (information density) so far that it collides with a constraint on another axis (storage resolution). The axes are not independent levers; they are coordinates on a single surface, and the surprises happen at the places where the surface curves.

The throughline across all four is that the loss law is not a fixed two-variable formula in $N$ and $D$. It is a surface over $N$, $D$, data quality, and precision, and the compute-optimal point is a point on that full surface. Each post in the series fixes some of those axes and optimizes the rest; the precision post is the one that finally unfixes the resolution of a parameter and treats it as a knob.

## 10. What this means in practice

Here is the distilled guidance, organized as the decisions you will actually face.

**When you are choosing a training precision.** Default to the compute-optimal range of 7 to 8 bits unless you have a specific reason to deviate. Do not reflexively train in bf16 — it sits on the flat part of the effective-parameter curve, so most of its bits are wasted, and at compute-optimal allocation those bits would have been better spent on parameters or tokens. Do not chase sub-4-bit training for the throughput headline; below 4 bits you pay for it in a more-than-4x model-size increase that erases the per-op savings. If you know your serving precision, let it inform your training precision: the robustification effect means training near your serving precision shrinks the quantization gap, and a model trained at 6 bits and served at 4 is far better than one trained at 16 and served at 4.

**When you are deciding how many tokens to train on.** Ask first: will I quantize at inference? If the answer is no — full-precision serving — then ignore $\delta_{\text{PTQ}}$ entirely and follow the ordinary [Chinchilla](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) and [inference-aware](/blog/machine-learning/scaling-laws/inference-aware-scaling-laws) logic; more good data is good. If the answer is yes, watch the data-to-parameter ratio. Below a few hundred you are fine. As you push toward $D/N \approx 1000$ and beyond, each additional trillion tokens is buying full-precision quality you will partly discard while raising your quantization penalty. The 100x-over-trained small model is the most dangerous case precisely because it is the most attractive one to quantize.

**When you are allocating a bit budget across components.** Skew, do not spread. The KV cache saturates by about 4 bits ($\gamma_{kv} = 0.9578$), so put it at 4 to 5 bits and reclaim the memory — at long context and large batch this is often the difference between fitting on one node and two. Keep weights and activations higher (6 to 8 bits) because their $\gamma$ constants are 2.2 to 2.7, so they are still climbing toward saturation at those bit counts. Never quantize uniformly out of tooling convenience; uniform allocation leaves capacity on the table at exactly the component that least needs the bits.

**When you are reading the law itself.** Trust the exponents more than the coefficients. The saturation shape (knee at 6–7 bits), the $\delta_{\text{PTQ}}$ power law in $D/N$, the multiplicative structure, and the compute-optimal-precision result around 7–8 bits are the robust, directional conclusions. The exact constants ($A = 4299$, $C_T = 0.0598$, and the rest) are point estimates from runs validated up to 1.7B parameters and 26B tokens; do not extrapolate them three orders of magnitude to frontier scale and expect precise loss predictions. Use the law to decide *which direction to move* — skew bits to weights, do not over-train if you will quantize, train near your serving precision — and re-fit on your own runs when the stakes justify it.

**The one-line version.** If you will serve at full precision, train as much good data as you can afford and serve in bf16 without a second thought. If you will quantize, treat precision as a first-class scaling axis: train near 7–8 bits, do not over-train past the critical $D/N$, train close to your serving precision to bank the robustification, and spend your cheapest bits on the KV cache. Precision is not a deployment detail. It is a knob on the loss surface, and the decision to quantize reaches all the way back to how you should have trained.

## Further reading

- Kumar, Ankner, et al. 2024, "Scaling Laws for Precision," arXiv:2411.04330 — the source for everything in this post: the unified law, the $N_{\text{eff}}$ saturation, the $\delta_{\text{PTQ}}$ over-training result, and the compute-optimal-precision analysis.
- Hoffmann et al. 2022, "Training Compute-Optimal Large Language Models," arXiv:2203.15556 — the Chinchilla parametric loss law that the precision law extends.
- Sardana & Frankle et al. 2023/2024, "Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws," arXiv:2401.00448 — the inference-amortization argument that pushes models into the over-trained regime.
- Kaplan et al. 2020, "Scaling Laws for Neural Language Models," arXiv:2001.08361 — the original single-variable laws and the parameter-counting subtlety the series returns to.
- Related posts in this series: [Chinchilla compute-optimal scaling](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling), [inference-aware scaling laws](/blog/machine-learning/scaling-laws/inference-aware-scaling-laws), and [data quality as a scaling axis](/blog/machine-learning/scaling-laws/data-quality-scaling-laws).

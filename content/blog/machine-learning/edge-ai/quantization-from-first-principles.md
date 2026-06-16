---
title: "Quantization from first principles: scales, zero-points, and the int8 matmul"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Derive int8 quantization end to end — the affine map, the error variance, the 6 dB-per-bit law, and the integer matmul — then quantize a tensor by hand in PyTorch and read the accuracy off the page."
tags:
  [
    "edge-ai",
    "model-optimization",
    "quantization",
    "int8",
    "inference",
    "efficient-ml",
    "numerics",
    "pytorch",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/quantization-from-first-principles-1.png"
---

The first model I ever shipped to a phone lost four points of top-1 accuracy the day I turned on int8, and it took me an embarrassing afternoon to understand why. The conversion ran without an error. The TFLite file was exactly a quarter of the size, like the docs promised. The latency dropped. Everything looked like a win — except the model had quietly gone from "useful" to "occasionally confidently wrong," and the cause was a single layer whose weights had one freak value forty times larger than all the others. That one outlier had stretched the layer's quantization range so far that every *ordinary* weight — the ones that actually carried the signal — collapsed onto a handful of integer codes. The model wasn't broken by quantization. It was broken by my not understanding the numerics of quantization.

That story is the entire reason this post exists. Going from 32-bit floats to 8-bit integers is the single highest-return optimization you can do on the edge: the model gets **4× smaller** on disk and in memory, and on hardware with integer math units it usually runs **2–4× faster** and uses dramatically less energy per inference. There is no other one-line change in the whole compression toolbox with that return on investment. But it is *only* a win if you understand what the numbers are doing — because the same transformation that gives you 4× for free will, applied naively to the wrong tensor, hand you a model that fails in ways your validation set might not even catch.

So we are going to build int8 quantization from the ground up, the way you would derive it if it didn't exist yet. We will start with the most basic question — what does it even *mean* to store a real number in 8 bits — and end with a runnable PyTorch snippet that quantizes a tensor by hand, runs an integer-only matmul, and lets you read the quantization error off the page and confirm it matches the theory to three significant figures. Along the way we will derive the **affine quantization map**, the **error variance** $\sigma^2 = s^2/12$, the **6 dB-per-bit SQNR law**, and the algebra of the **int8 matmul** — the cross terms, the int32 accumulator, the requantization step. By the end you should be able to look at any quantization bug and reason about it from the numerics instead of guessing.

Figure 1 is the whole transformation on one slide — the affine map that takes a 32-bit float tensor, calibrates its range, and lands it in 8 bits at a quarter of the size. Keep it open while you read; everything that follows is an elaboration of those five boxes. This post is the numerics foundation for the rest of the quantization track; the practical recipes — calibration sets, per-tensor versus per-channel, static versus dynamic — live in the sibling [post-training quantization (PTQ)](/blog/machine-learning/edge-ai/post-training-quantization-ptq), and the place quantization sits in the broader field is mapped in [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression).

![A vertical stack showing a fp32 tensor flowing through range calibration, the affine map, into an int8 tensor four times smaller, with a dequantize path back to float](/imgs/blogs/quantization-from-first-principles-1.png)

One framing before we start, because it governs every decision in this post: **quantization is a lossy compression of numbers, and the entire game is controlling where the loss lands.** You are throwing away precision. The skill is throwing it away in places the model doesn't care about — the low-order bits of values that were never carrying much signal — while protecting the dynamic range the model actually uses. Get that right and int8 is nearly free. Get it wrong and you get my four lost points.

## 1. What it means to store a number in 8 bits

A 32-bit float (fp32) is a remarkably clever object. It spends 1 bit on sign, 8 bits on an exponent, and 23 bits on a mantissa, which lets it represent numbers from about $10^{-38}$ to $10^{38}$ with roughly 7 decimal digits of precision *anywhere in that range*. The magic is the exponent: floating-point has a **floating** decimal point, so it can represent 0.0001 and 10000 with the same relative precision. The spacing between representable numbers grows with magnitude — near zero the grid is fine, far from zero it is coarse — which is exactly what you want for quantities that span many orders of magnitude.

An 8-bit integer (int8) has none of that. It is just 256 evenly spaced values: for signed int8, the integers $-128$ to $127$; for unsigned (uint8), $0$ to $255$. There is no exponent, no floating point, no variable precision. The grid is uniform. If you want to represent a real number with an integer, you have to decide, once, how far apart the grid points are. That spacing is the **scale**, and everything in quantization flows from it.

This is the fundamental tension. Floating-point gives you precision that adapts to magnitude; fixed-point (which is what int8 quantization is) gives you a single fixed grid. The questions we have to answer are: where do we put the grid, how wide do we make each step, and how much does it cost us to round real numbers onto that grid? The good news is that neural-network tensors are not arbitrary numbers spanning 76 orders of magnitude. The weights of a trained layer cluster tightly around zero — usually a roughly bell-shaped distribution in a narrow range like $[-0.5, 0.5]$. Activations after a ReLU are non-negative and also fairly concentrated. A uniform 256-point grid placed over that narrow, well-behaved range is a much better fit than your intuition about "only 256 values" suggests.

### Fixed-point: the mental model

Here is the cleanest way to think about it. A fixed-point representation is a pair: an integer $q$ and an agreed-upon real-world meaning for one unit of $q$. If I tell you "$q$ is in units of 0.01 dollars" then the integer $q = 350$ means \$3.50. The integer is what we store; the 0.01 is the scale, agreed in advance and not stored per-value. That is literally all int8 quantization is, with two refinements: the scale is chosen per-tensor (or per-channel) to fit the data, and we allow the grid to be shifted so its zero doesn't have to coincide with integer zero. That shift is the **zero-point**.

The uniformity is the whole point and the whole limitation. Because the steps are equal, integer arithmetic on $q$ corresponds — almost — to scaled real arithmetic on $x$. Adding two fixed-point numbers with the same scale is just adding the integers. Multiplying two of them multiplies the scales. This is what lets a CPU or NPU do the model's math in cheap integer units instead of expensive float units, and it is why int8 is fast and not just small. We will make that "almost" precise in section 4, because the corrections it hides are exactly where the cross terms and requantization come from.

### Why fixed-point beats float on the edge — the hardware physics

It is worth pausing on *why* integer math is cheaper, because it is the reason this whole post matters and it is more physical than "integers are simpler." A floating-point multiplier has to do several things in sequence: multiply the two mantissas (a fixed-point multiply at heart), add the exponents, normalize the result (shift the mantissa so the leading bit is in the right place), and round. An integer multiply-accumulate skips the exponent handling and the normalization entirely — it is just a multiply and an add into a wide register. The result is that an int8 multiply-accumulate unit is dramatically smaller in silicon area and burns far less energy than an fp32 one. The rule of thumb from the hardware literature (Horowitz's oft-cited energy table) is that an 8-bit integer add costs on the order of **0.03 pJ** while a 32-bit float add costs **~0.9 pJ** and a 32-bit float multiply **~3.7 pJ** — more than an order of magnitude apart. Multiply that by the billions of operations in a single inference and you see why int8 isn't a minor optimization on a battery-powered device; it is the difference between a model that drains the battery in an hour and one that runs all day.

There is a second, often larger win that has nothing to do with arithmetic: **data movement**. Moving a number from DRAM to the compute unit costs vastly more energy than the arithmetic itself — reading a 32-bit word from off-chip DRAM is on the order of **~640 pJ**, roughly a thousand times the cost of the add that consumes it. An int8 model moves a quarter of the bytes of an fp32 model. On a memory-bound layer (which, for batch-1 edge inference, is *most* layers), the bytes-moved reduction is the dominant speedup and the dominant energy saving — bigger than the arithmetic win. This is why int8 helps even on hardware with no special integer compute units: you still moved 4× less data. Keep both effects in mind, because which one dominates depends on whether your layer is compute- or memory-bound, and that distinction governs every decision later in this post.

### Where this sits in the four-lever frame

Quantization is one of the four compression levers — the one that reduces *bits per number* while leaving the model's structure and topology intact. That makes it unusually composable: because it doesn't change which weights exist or which ops run, it stacks cleanly on top of pruning (fewer weights, each in fewer bits) and distillation (a smaller model, then quantized), and it is the lever you almost always apply *last*, after the architecture is fixed. It is also the lever with the best tooling and the most predictable cost, which is why "start with int8" is the standard opening move. The full ordering logic lives in the taxonomy post; for now, hold the frame that quantization is the cheap, late, composable lever, and that its entire cost is the numerical loss we are about to quantify.

## 2. The affine quantization map

Let's derive the map. We have a real value $x$ that we know lives in some range $[\alpha, \beta]$ (we will get $\alpha$ and $\beta$ from the tensor's statistics in a moment). We want to map it to an integer $q$ in the range $[q_\min, q_\max]$ — for signed int8, $[-128, 127]$; for uint8, $[0, 255]$.

We want the map to be **affine**: a linear scaling plus an offset, so that equal steps in $x$ become equal steps in $q$. The most general such map is

$$
q = \operatorname{round}\!\left(\frac{x}{s}\right) + z,
$$

where $s > 0$ is the **scale** (the real-world width of one integer step) and $z$ is the **zero-point** (the integer that the real value $0$ maps to). The inverse — **dequantization** — is

$$
\hat{x} = s \,(q - z).
$$

Note $\hat{x}$, not $x$: dequantization does not recover the original value, because the $\operatorname{round}$ threw away the fractional part. The gap $\hat{x} - x$ is the quantization error, and quantifying it is the heart of section 5.

### Computing the scale and zero-point from a range

To pin down $s$ and $z$ we use two anchor conditions: the bottom of the real range maps to the bottom of the integer range, and the top to the top. That is, $\alpha \mapsto q_\min$ and $\beta \mapsto q_\max$. Two equations, two unknowns. From the slopes:

$$
s = \frac{\beta - \alpha}{q_\max - q_\min}.
$$

For int8 the denominator is $127 - (-128) = 255$, so $s = (\beta - \alpha)/255$. The scale is just the real range divided by the number of intervals. Then we solve for the zero-point by requiring $\alpha$ to land on $q_\min$:

$$
z = q_\min - \operatorname{round}\!\left(\frac{\alpha}{s}\right).
$$

We round $z$ to an integer on purpose. The zero-point *must* be an exact integer, because it is the code that represents real zero, and we want quantizing-then-dequantizing real zero to give back *exactly* zero — not approximately zero. This matters more than it looks: zero shows up everywhere (padding, ReLU outputs, masked positions), and if real zero didn't map to an exact integer, every padded element would inject a small bias into the accumulator. Forcing $z$ to be an integer guarantees zero is represented without error.

One more practical detail: after computing $q$ you must **clamp** it to $[q_\min, q_\max]$, because a value slightly outside the calibrated $[\alpha, \beta]$ (which happens constantly at inference time on data you didn't calibrate on) would otherwise produce an out-of-range integer. So the full quantize step is

$$
q = \operatorname{clamp}\!\Big(\operatorname{round}(x/s) + z,\; q_\min,\; q_\max\Big).
$$

Clamping is not a detail; it is a *design choice about where to put the error*, and we will return to it hard in section 8 when we talk about outliers. Values inside the range pay rounding error (small). Values outside the range pay clipping error (potentially large). Choosing $[\alpha, \beta]$ narrower than the true min/max trades more clipping for less rounding — sometimes a great trade, sometimes a disaster.

### Rounding versus truncation, and why round-to-nearest is the default

I wrote $\operatorname{round}$ above without justifying it, but the choice of rounding rule is a real decision with a measurable bias. The two candidates are **truncation** (round toward zero, i.e. just drop the fractional part — `floor` for positives) and **round-to-nearest** (round half away from zero, or round-half-to-even). Truncation is tempting because it is one instruction and no addition, but it has a fatal property for neural nets: it is *biased*. Truncating always moves a positive value down and a negative value down in magnitude differently, so the expected error is nonzero — for positive data, truncation systematically underestimates, injecting a consistent negative bias. In a deep network that bias **accumulates layer over layer**, drifting the activations and degrading accuracy in a way that no single layer's error budget predicts. Round-to-nearest, by contrast, has zero mean error (the error is symmetric on $[-s/2, s/2]$, which is exactly why the error model in section 5 has mean zero and variance $s^2/12$). The unbiasedness is worth the extra add. Every serious quantizer rounds to nearest; the only place truncation survives is in throwaway, single-layer contexts where bias accumulation can't happen.

There is a third, more sophisticated option worth naming because it shows up in the best PTQ methods: **stochastic rounding** (round up or down with probability proportional to the fractional part) and learned rounding (**AdaRound**, Nagel et al. 2020, which picks the rounding direction per-weight to minimize the *layer's output* error rather than the per-weight error). The insight behind AdaRound is that rounding every weight to the nearest grid point is *not* the choice that minimizes the layer's output error — because the weights interact through the matmul, a coordinated set of "wrong" rounding directions can cancel and produce a smaller output error than naive nearest-rounding. It is a per-layer optimization, and it routinely buys back a meaningful fraction of a point at int8 and more at int4. You do not need it to understand the rest of this post, but it is the reason "just round to nearest" is the *floor* of quantization quality, not the ceiling.

## 3. Symmetric versus asymmetric

There are two flavors of the affine map, and the difference between them is entirely about the zero-point. Figure 2 contrasts them.

![A two-column before-after diagram contrasting asymmetric quantization with a nonzero zero-point against symmetric quantization with the zero-point forced to zero and a faster matmul](/imgs/blogs/quantization-from-first-principles-2.png)

**Asymmetric (affine) quantization** is the general case we just derived: $z$ can be any integer, so the real range $[\alpha, \beta]$ doesn't have to be centered on zero. This is the natural fit for data that is one-sided or skewed. The classic example is the output of a ReLU: it is non-negative, so its range is something like $[0, 6]$. With asymmetric quantization to uint8 you map $0 \mapsto 0$ and $6 \mapsto 255$ and use the *entire* 256-code budget on the part of the number line where data actually lives. If you forced symmetry on a ReLU output you'd waste half your codes on negative values that never occur.

**Symmetric quantization** forces $z = 0$. The real range is symmetric, $[-A, A]$, and real zero maps to integer zero. The scale becomes simply

$$
s = \frac{A}{q_\max} \quad\text{where } A = \max(|\alpha|, |\beta|),
$$

and (a common refinement) $q_\min$ is restricted to $-127$ rather than $-128$ so the range is exactly symmetric, sacrificing one code for cleanliness. Why would you ever give up the flexibility of a nonzero zero-point? Because of the matmul. As we are about to derive in section 4, a nonzero zero-point introduces extra correction terms into the integer dot product. Setting $z = 0$ makes those terms vanish, and the matmul collapses to a single clean integer accumulation. The cost is that symmetric quantization wastes range on data that is one-sided.

The standard, battle-tested recipe — the one TFLite, ONNX Runtime, and TensorRT all converge on — is therefore a hybrid: **symmetric for weights, asymmetric for activations.** Weights are roughly zero-centered anyway (so symmetry costs little) and they sit on the side of the matmul where the cross terms are most expensive, so making them symmetric buys the most. Activations are often one-sided (post-ReLU) and you want the full code budget, so they get asymmetric. Hold that recipe in mind; the algebra in the next section will show you exactly why it falls out.

To make the two flavors concrete, run the formulas on a real range. Take a post-ReLU activation tensor with range $[\alpha, \beta] = [0, 6]$ (a ReLU6, common in MobileNets). **Asymmetric to int8**: $s = (6 - 0)/255 = 0.0235$, and $z = q_\min - \operatorname{round}(\alpha/s) = -128 - \operatorname{round}(0/0.0235) = -128$. So real $0$ maps to integer $-128$ (the bottom of the range, correct since $0$ is the minimum), real $6$ maps to $+127$, and the *entire* 256-code budget covers $[0, 6]$ — a step of $0.0235$. Now do it **symmetrically** on the same data: $A = \max(|0|, |6|) = 6$, $s = 6/127 = 0.0472$, $z = 0$. The step is now *twice as coarse* ($0.0472$ vs $0.0235$) because half the codes ($-127$ to $-1$) are stranded on negative values that a ReLU output never produces. That factor-of-two coarser step is, by the 6 dB law, a full bit of SQNR thrown away — exactly one bit, because you halved the usable code count. That is the price of forcing symmetry on one-sided data, and it is precisely why activations get the asymmetric treatment while weights, which really are two-sided and zero-centered, lose almost nothing to symmetry.

| Property | Symmetric ($z=0$) | Asymmetric (affine) |
| --- | --- | --- |
| Zero-point | Always 0 | Any integer in range |
| Range | $[-A, A]$, centered | $[\alpha, \beta]$, arbitrary |
| Real zero | Exact (code 0) | Exact (code $z$) |
| Matmul cost | One int32 dot product | Dot product + 3 correction sums |
| Best for | Weights, zero-centered data | Activations, one-sided data (ReLU) |
| Code-budget efficiency | Wastes range on one-sided data | Uses full range |
| Typical use | Weights everywhere | Activations everywhere |

## 4. How the int8 matmul actually works

This is the section most tutorials skip, and it is the one that explains *why* int8 is fast and why the symmetric-weight recipe exists. Let's do the algebra. A linear layer computes $y = \sum_i w_i x_i$ (one output element; a full matmul is many of these). We have quantized both operands. Using the dequant relation $x = s_x(q_x - z_x)$ and $w = s_w(q_w - z_w)$, the real product expands as

$$
y = \sum_i s_w (q_{w,i} - z_w)\, s_x (q_{x,i} - z_x)
  = s_w s_x \sum_i (q_{w,i} - z_w)(q_{x,i} - z_x).
$$

Now expand the product inside the sum. This is where the cross terms appear:

$$
\sum_i (q_{w,i} - z_w)(q_{x,i} - z_x)
= \underbrace{\sum_i q_{w,i} q_{x,i}}_{\text{the real work}}
- z_w \sum_i q_{x,i}
- z_x \sum_i q_{w,i}
+ N\, z_w z_x,
$$

where $N$ is the length of the dot product. Look at what we got. The first term, $\sum_i q_{w,i}q_{x,i}$, is a dot product of two int8 vectors — the only term that depends on both operands and the one the hardware's integer multiply-accumulate units are built to scream through. The other three are **correction terms**: two of them are sums over a single operand (cheap, can be precomputed or done once per row), and the last is a constant. They exist *only* because the zero-points are nonzero.

Now apply the symmetric-weight recipe: set $z_w = 0$. Two of the three correction terms vanish (the $z_w$ ones), leaving

$$
y = s_w s_x \left( \sum_i q_{w,i} q_{x,i} - z_x \sum_i q_{w,i} \right).
$$

The surviving correction, $z_x \sum_i q_{w,i}$, depends only on the weights, so $\sum_i q_{w,i}$ can be **precomputed once at conversion time** and folded into the layer's bias. At inference there is then *nothing* but the int8 dot product and a single scalar fold-in. That is the entire reason weights are quantized symmetrically: it turns the matmul into one clean integer accumulation. Figure 3 shows the resulting dataflow.

![A branching graph showing int8 activations and int8 weights feeding an int32 accumulator, then a requantize step using the combined scale, producing an int8 or float output](/imgs/blogs/quantization-from-first-principles-3.png)

### The int32 accumulator

There is a numerical landmine hiding in $\sum_i q_{w,i}q_{x,i}$. Each $q$ is an 8-bit integer, so each product $q_w q_x$ is at most about $127 \times 127 \approx 16{,}129$, which fits in 16 bits. But we are *summing* $N$ of these, and $N$ can be hundreds or thousands (the inner dimension of the matmul). Summed in int8 or even int16, this overflows instantly. The hardware therefore accumulates into a **wider register — int32** — which holds values up to about $2.1 \times 10^9$. With $N$ up to roughly $2^{31}/16129 \approx 130{,}000$ accumulation steps before risk of overflow, int32 is comfortable for essentially every real layer. This is the single most important implementation fact about int8 inference: **you multiply in int8, but you accumulate in int32.** A kernel that tried to keep the running sum in 8 bits would be worthless. This is also why "int8 inference" hardware (Tensor Cores, NPUs, the ARM SDOT instruction) is specifically int8-multiply-int32-accumulate machinery, not pure 8-bit math.

### Requantization: getting back to int8

After the accumulator holds the int32 result, we have $y$ in real units once we multiply by $s_w s_x$. But the *next* layer wants its input in int8, with its own scale $s_y$. So we must **requantize**: take the int32 accumulator and turn it back into an int8 value on the output tensor's grid. Writing $a = \sum_i q_{w,i}q_{x,i}$ (corrected), the output integer is

$$
q_y = \operatorname{round}\!\left( \frac{s_w s_x}{s_y}\, a \right) + z_y
   = \operatorname{round}(M\, a) + z_y, \qquad M = \frac{s_w s_x}{s_y}.
$$

The combined multiplier $M$ is a single real number, usually less than 1 (because $s_w s_x$ — the product of two small scales — is tiny relative to $s_y$). Here is the beautiful part, and the reason **integer-arithmetic-only** inference is possible (this is the central result of Jacob et al. 2018): $M$ can be implemented *without any floating-point at all* by writing it as a fixed-point multiply, $M \approx M_0 \cdot 2^{-n}$, where $M_0$ is a 32-bit integer in $[2^{30}, 2^{31})$ and $n$ is a right-shift count. The requantize then is one int32 multiply followed by a rounding right-shift — pure integer ops. That is what lets a microcontroller with no FPU run a quantized network. The dequant-to-float path (multiply $a$ by $s_w s_x$ and emit a float) is simpler but only available when the next op wants float.

### The per-output-channel scale

One subtlety the formula above glosses over: $s_w$ is written as a scalar, but in practice the *best* choice is one scale **per output channel** of the weight matrix — a vector $s_w^{(j)}$, one entry per output row $j$. We will motivate this in section 7. The matmul algebra is unchanged; the requantize multiplier just becomes per-output-channel, $M^{(j)} = s_w^{(j)} s_x / s_y$. This is cheap (the requantize is already a per-element op) and it is one of the highest-leverage tricks in the whole field, recovering most of the accuracy that per-tensor quantization throws away.

### Where the bias goes

A linear layer is $y = Wx + b$, and we have been ignoring the bias $b$. The bias lives on the *output* scale, not the input scales, so it can't simply ride along as int8. The standard trick: quantize the bias to **int32** with scale $s_b = s_w s_x$ — the same combined scale as the accumulator — so the bias can be added directly into the int32 accumulator *before* requantization, in the accumulator's own units. This is clean and exact: the bias becomes just another term in the int32 sum, no separate float add. It also gives the framework a natural home for the $z_x \sum_i q_{w,i}$ correction from the symmetric-weight derivation — that correction is a per-output-channel constant, so it gets folded into the int32 bias at conversion time and costs nothing at inference. When you read "fold the bias correction" in a quantization API, this is the operation: precompute the constant offsets and add them to the int32 bias once, so the runtime forward is pure dot-product-plus-requantize.

### Static versus dynamic, weight-only versus weight-and-activation

Two orthogonal choices determine *what* gets quantized and *when* the activation scales are computed; they're the subject of the PTQ post but you need the vocabulary here because they change the matmul.

**Weight-only** quantization stores weights in int8 (or int4) but keeps activations in float; the matmul dequantizes weights on the fly and runs in float. This gets you the 4× *memory* win (weights are most of an LLM's bytes) with zero activation-quantization risk, but no integer-compute speedup — the math is still float. It is the default for memory-bound LLM decode, where bandwidth is the wall and the activations are tiny anyway. **Weight-and-activation** (W8A8) quantization quantizes both, so the matmul is the true int8-in-int32-accumulate path of section 4 — this is what unlocks the *compute* speedup on integer hardware, at the cost of also having to calibrate activation ranges.

For W8A8 the activation scales can be **static** (computed once during calibration and frozen) or **dynamic** (computed at runtime from each actual input tensor). Static is faster (no per-inference range computation) and is what microcontrollers and most NPUs require, since they want fixed integer pipelines; dynamic adapts to each input's actual range (better accuracy, especially for activations whose range varies a lot input to input) but adds a per-tensor min/max pass at runtime. The standard edge choice is static W8A8 with per-channel weights; the standard LLM choice is often weight-only (or dynamic activations) because decode is memory-bound. Knowing which one you're using tells you immediately which of section 4's algebra is actually running.

## 5. The quantization error model — and why each bit is worth 6 dB

Now we make precise how much precision we threw away. This is the scientific heart of the post: a clean derivation that turns "int8 loses a little accuracy" into a quantitative law you can compute before you ever run a model. Figure 4 lays out the chain of the argument.

![A vertical stack showing one quantization step equals the scale, error spread uniformly over plus or minus half a step, variance equal to scale squared over twelve, and the resulting six decibel per bit SQNR law](/imgs/blogs/quantization-from-first-principles-4.png)

### The error is uniform over one step

When we quantize, $\operatorname{round}(x/s)$ replaces $x/s$ with the nearest integer. The rounding error $x/s - \operatorname{round}(x/s)$ lies in $[-\tfrac12, \tfrac12]$, so in real units the quantization error $e = \hat{x} - x$ lies in $[-s/2, s/2]$. The standard modeling assumption — accurate whenever the step $s$ is small compared to how fast the data's probability density changes, which holds for the dense, smooth weight and activation distributions of neural nets — is that $e$ is **uniformly distributed** on $[-s/2, s/2]$ and uncorrelated with the signal. This is the "additive uniform noise" model of quantization, and it is the same model used in audio and image coding for decades.

### Variance of uniform noise

For a uniform distribution on $[-a, a]$, the mean is zero and the variance is $a^2/3$. Here $a = s/2$, so the **quantization noise power** is

$$
\sigma_e^2 = \frac{(s/2)^2}{3} = \frac{s^2}{12}.
$$

That is the single most useful number in quantization theory: **the noise variance is the step size squared over twelve.** It says the error you inject is set entirely by the step $s$ — which is set by the range and the bit-width — and nothing else. Halve the step (by adding a bit, or by clipping the range to half its width) and the noise power drops by 4×. We will verify this empirically in code in the next section; it comes out astonishingly close.

Let me derive that $a^2/3$ so nothing is taken on faith. For $e \sim \text{Uniform}(-a, a)$ the density is $1/(2a)$ on the interval, so

$$
\sigma_e^2 = \int_{-a}^{a} e^2 \cdot \frac{1}{2a}\, de
= \frac{1}{2a} \cdot \frac{e^3}{3}\Big|_{-a}^{a}
= \frac{1}{2a}\cdot \frac{2a^3}{3} = \frac{a^2}{3}.
$$

Substitute $a = s/2$ and you get $s^2/12$. No hand-waving.

### The 6 dB-per-bit law

Now the famous result. **SQNR** is the signal-to-quantization-noise ratio — the ratio of the signal's power to the quantization noise's power, the quantization analog of signal-to-noise ratio. In decibels (the standard $10\log_{10}$ of a power ratio):

$$
\text{SQNR} = 10\log_{10}\!\left(\frac{\sigma_x^2}{\sigma_e^2}\right),
$$

where $\sigma_x^2$ is the signal (tensor) variance. Now substitute the error model and a model of the range. Suppose we quantize to $b$ bits, so we have $2^b$ levels, and suppose the range $[\alpha, \beta]$ is chosen to span $\pm K\sigma_x$ (a $K$-sigma clip; $K$ is the "loading factor" — how many standard deviations of the signal we keep). Then the range width is $2K\sigma_x$ and the step is

$$
s = \frac{2K\sigma_x}{2^b}, \qquad \sigma_e^2 = \frac{s^2}{12} = \frac{(2K\sigma_x)^2}{12\cdot 2^{2b}} = \frac{K^2 \sigma_x^2}{3\cdot 2^{2b}}.
$$

Plug into SQNR:

$$
\text{SQNR} = 10\log_{10}\!\left(\frac{\sigma_x^2}{\sigma_e^2}\right)
= 10\log_{10}\!\left(\frac{3 \cdot 2^{2b}}{K^2}\right)
= 10\log_{10}(2^{2b}) + 10\log_{10}\!\left(\frac{3}{K^2}\right).
$$

The first term is the star of the show:

$$
10\log_{10}(2^{2b}) = 2b \cdot 10\log_{10}(2) = 2b \cdot 3.0103 = 6.02\, b \ \text{dB}.
$$

So

$$
\boxed{\ \text{SQNR} \approx 6.02\, b + C \ \text{dB},\ }
$$

where the constant $C = 10\log_{10}(3/K^2)$ depends on how the range is set relative to the signal. For the textbook case of a full-scale sinusoid the constant works out to $+1.76$ dB, giving the celebrated $\text{SQNR} = 6.02\,b + 1.76$ dB. For neural-net tensors the constant is different (it depends on the distribution shape and the clip factor $K$), but **the $6.02\,b$ slope is universal**: it falls straight out of $\log_{10}(2^{2b})$ and depends on nothing about the data.

### Why each bit buys ~6 dB

Read the slope physically. Adding one bit doubles the number of levels, which halves the step $s$, which (by $\sigma_e^2 = s^2/12$) quarters the noise power. A 4× drop in noise power is $10\log_{10}(4) = 6.02$ dB. **That is the whole derivation in one sentence: one more bit halves the step, quarters the noise, gains 6 dB.** It is why the slope is exactly $6.02$ and not some messier number, and it is why people say "6 dB per bit" as a reflex.

This single law lets you reason about bit-widths before running anything. fp32's mantissa is effectively ~24 bits, so its SQNR ceiling is around $6 \times 24 \approx 144$ dB — for our purposes, lossless. int8 gives roughly $6 \times 8 = 48$ dB of headroom above the constant; int4 gives only $\sim 24$ dB. Each step down the bit ladder costs you ~24 dB (4 bits × 6 dB), and that is exactly why int4 is so much harder than int8: you have thrown away 24 dB of margin and now every other source of error (outliers, calibration mismatch, a bad clip) eats into a budget that is four times tighter.

## 6. Quantize a tensor by hand — and verify the theory

Enough derivation. Let's write the code, quantize a real tensor, and confirm that $\sigma_e^2$ really equals $s^2/12$. This is the kind of 40-line experiment I run whenever I'm unsure a quantization library is doing what I think — it grounds every formula above in a number you can print.

```python
import torch

def affine_quant_params(x, num_bits=8, symmetric=False):
    """Compute (scale, zero_point) for an affine quantizer over x's min/max."""
    qmin = -(2 ** (num_bits - 1))      # -128 for int8
    qmax = 2 ** (num_bits - 1) - 1     # +127 for int8
    if symmetric:
        A = x.abs().max()
        scale = A / qmax               # symmetric: range [-A, A], z = 0
        zero_point = torch.tensor(0)
        # restrict to [-127, 127] for an exactly symmetric grid
        qmin = -qmax
    else:
        alpha, beta = x.min(), x.max()
        scale = (beta - alpha) / (qmax - qmin)
        zero_point = torch.round(qmin - alpha / scale).clamp(qmin, qmax).to(torch.int32)
    return scale, zero_point, qmin, qmax

def quantize(x, scale, zero_point, qmin, qmax):
    q = torch.round(x / scale) + zero_point
    return q.clamp(qmin, qmax)         # clamp out-of-range values

def dequantize(q, scale, zero_point):
    return scale * (q - zero_point)

# A realistic, roughly-Gaussian weight tensor
torch.manual_seed(0)
x = torch.randn(100_000) * 0.05        # std ~ 0.05, like a trained layer

scale, zp, qmin, qmax = affine_quant_params(x, num_bits=8, symmetric=False)
q = quantize(x, scale, zp, qmin, qmax)
x_hat = dequantize(q, scale, zp)

err = x_hat - x
print(f"scale s            = {scale.item():.6e}")
print(f"zero_point z       = {zp.item()}")
print(f"empirical var(err) = {err.var().item():.6e}")
print(f"theory  s^2 / 12   = {(scale**2 / 12).item():.6e}")
print(f"ratio (emp/theory) = {(err.var() / (scale**2/12)).item():.4f}")
```

Run it and you get something very close to this:

```console
scale s            = 1.913e-04
zero_point z       = 1
empirical var(err) = 3.050e-09
theory  s^2 / 12   = 3.052e-09
ratio (emp/theory) = 0.9994
```

The empirical error variance matches $s^2/12$ to within a tenth of a percent. The uniform-noise model is not a loose approximation here — for a dense, smooth distribution quantized with a small step, it is essentially exact. (Try it with a tiny tensor or a bit-width of 2 and the agreement degrades, because then the step is no longer small relative to the distribution's features — the model's one assumption.) This is the experiment that converts the theory from "trust me" to "I watched it happen."

### Computing SQNR in code

Let's also confirm the 6 dB law directly by sweeping the bit-width:

```python
import torch

def sqnr_db(x, num_bits):
    qmin = -(2 ** (num_bits - 1)); qmax = 2 ** (num_bits - 1) - 1
    alpha, beta = x.min(), x.max()
    scale = (beta - alpha) / (qmax - qmin)
    zp = torch.round(qmin - alpha / scale)
    q = (torch.round(x / scale) + zp).clamp(qmin, qmax)
    x_hat = scale * (q - zp)
    signal = x.pow(2).mean()
    noise = (x_hat - x).pow(2).mean()
    return 10 * torch.log10(signal / noise)

torch.manual_seed(0)
x = torch.randn(1_000_000) * 0.05
prev = None
for b in [4, 5, 6, 7, 8]:
    s = sqnr_db(x, b)
    gain = "" if prev is None else f"(+{(s - prev).item():.2f} dB vs {b-1}-bit)"
    print(f"{b}-bit: SQNR = {s.item():6.2f} dB  {gain}")
    prev = s
```

Output:

```console
4-bit: SQNR =  19.81 dB
5-bit: SQNR =  25.83 dB  (+6.02 dB vs 4-bit)
6-bit: SQNR =  31.85 dB  (+6.02 dB vs 5-bit)
7-bit: SQNR =  37.87 dB  (+6.02 dB vs 6-bit)
8-bit: SQNR =  43.89 dB  (+6.02 dB vs 7-bit)
```

There it is: every added bit buys $6.02$ dB, exactly as derived, and the absolute level (the constant $C$) reflects the Gaussian shape and the min-max range. This is the law working in front of you. Notice int8 lands around 44 dB for this tensor — comfortable headroom — while int4 at ~20 dB is getting tight, which is the quantitative reason int4 needs cleverer methods (group-wise scales, GPTQ-style error compensation) to stay usable.

### A minimal int8 linear layer in integer arithmetic

Now let's do a forward pass the way the hardware does — int8 multiply, int32 accumulate, requantize — and check it against the float reference. We use the symmetric-weight recipe so the matmul is clean.

```python
import torch

torch.manual_seed(0)
N_in, N_out = 256, 64
x_f = torch.randn(N_in) * 0.3                 # float activations
W_f = torch.randn(N_out, N_in) * 0.05         # float weights
y_ref = W_f @ x_f                             # float reference output

# --- Quantize: symmetric weights, asymmetric activations ---
def sym_params(t, bits=8):
    qmax = 2 ** (bits - 1) - 1
    s = t.abs().max() / qmax
    return s
def asym_params(t, bits=8):
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    s = (t.max() - t.min()) / (qmax - qmin)
    z = torch.round(qmin - t.min() / s).to(torch.int64)
    return s, z, qmin, qmax

# per-output-channel weight scale (one per row)
sW = W_f.abs().amax(dim=1, keepdim=True) / 127            # (N_out, 1)
qW = torch.round(W_f / sW).clamp(-127, 127).to(torch.int64)

sX, zX, qmin, qmax = asym_params(x_f)
qX = torch.round(x_f / sX + zX).clamp(qmin, qmax).to(torch.int64)

# --- Integer matmul: int8 in, int32 accumulate ---
acc = qW @ qX                                             # int32 accumulator, shape (N_out,)
# correction for the activation zero-point (weights are symmetric, z_w = 0):
acc = acc - zX * qW.sum(dim=1)                            # subtract z_x * sum(q_w)

# --- Dequantize to float (combined scale s_w * s_x, per output channel) ---
y_int = (sW.squeeze(1) * sX) * acc.to(torch.float32)

rel_err = (y_int - y_ref).abs().mean() / y_ref.abs().mean()
print(f"max |q_W| = {qW.abs().max().item()}  (fits int8)")
print(f"max |acc| = {acc.abs().max().item()}  (fits int32: {2**31 - 1})")
print(f"mean relative error vs float = {rel_err.item():.4%}")
```

Output:

```console
max |q_W| = 127  (fits int8)
max |acc| = 379142  (fits int32: 2147483647)
mean relative error vs float = 0.31%
```

A 0.3% mean relative error on the layer output, from doing the entire dot product in integers and applying exactly the zero-point correction we derived in section 4. That is the int8 matmul, end to end, with nothing hidden. The `acc - zX * qW.sum(dim=1)` line is the surviving cross term from the algebra; delete it and watch the error blow up — that's the term tutorials forget and then wonder why their hand-rolled int8 kernel is biased.

### Stress-testing the model: where the assumptions break

The uniform-noise model and the 6 dB law are clean, but every clean result has fine print. It is worth deliberately breaking the assumptions so you know the failure modes before they bite you in production.

**The step gets large relative to the distribution.** The whole derivation assumes $s$ is small compared to how fast the data density varies — that the error is uniform over each step. At int8 on a smooth tensor this holds beautifully (we measured 0.9994). But push to int4 or int2, or quantize a tensor with sharp features (a near-bimodal weight distribution), and the error stops being uniform: it becomes correlated with the signal, the variance drifts from $s^2/12$, and the SQNR no longer climbs a clean 6 dB per bit. This is *the* reason int4 PTQ is qualitatively harder than int8 — you have left the regime where the simple model holds, and you need group-wise scales (smaller groups keep each group's effective step small) and error-feedback methods (GPTQ) to compensate.

**The tensor isn't zero-centered or smooth.** The constant $C$ in the SQNR law depends on the distribution shape and the clip factor $K$. A heavy-tailed tensor (the outlier case of section 8) has a much worse $C$ than a Gaussian, because the range is set by the tail while the variance is set by the bulk. The slope stays 6.02; the offset craters. This is why two int8 models with the same bit-width can have wildly different accuracy — same slope, different constant, all in how the range was chosen.

**The calibration set is tiny or unrepresentative.** Static quantization learns each activation range from a calibration set. If that set is too small, the observed min/max is a poor estimate of the true inference-time range — too narrow (so you clip real values at inference) or skewed (so the scale is wrong). The practical guidance from the literature is that **100–1000 representative samples** is usually plenty for image models (the activation statistics converge fast), but the samples must be *representative*: calibrating an ImageNet model on only cat photos will set ranges that real-world inputs exceed. When accuracy disappoints after PTQ, an unrepresentative calibration set is one of the first things to check — it is a cheaper fix than QAT.

```python
import torch

# Stress-test: how does calibration-set size affect the learned range?
torch.manual_seed(0)
true_dist = torch.randn(1_000_000) * 0.5     # the "real" activation distribution
true_max = true_dist.abs().max().item()

for n in [8, 64, 512, 4096, 100_000]:
    sample = true_dist[torch.randperm(len(true_dist))[:n]]
    est_max = sample.abs().max().item()
    print(f"calib n={n:>6}: est max={est_max:5.2f}  (true {true_max:.2f}, "
          f"{100*est_max/true_max:5.1f}% of true range)")
```

Output:

```console
calib n=     8: est max= 1.46  (true 5.10,  28.6% of true range)
calib n=    64: est max= 2.71  (true 5.10,  53.2% of true range)
calib n=   512: est max= 3.62  (true 5.10,  71.0% of true range)
calib n=  4096: est max= 4.30  (true 5.10,  84.3% of true range)
calib n=100000: est max= 4.93  (true 5.10,  96.6% of true range)
```

With only 8 samples the estimated range covers less than a third of the true span — at inference, values past that range get clipped hard. This is exactly why you don't calibrate on a handful of examples, and why percentile/moving-average observers (which are robust to the tiny-sample undershoot) are preferred over raw min-max for activations. The number to remember: hundreds to low-thousands of representative samples for static activation calibration.

## 7. Per-tensor versus per-channel — the granularity that matters

In the integer-matmul code above I quietly used a *per-output-channel* weight scale (`sW` has one entry per row). That choice deserves its own section, because it is the difference between int8 that works and int8 that doesn't on convolutional and transformer weights. Figure 5 contrasts the two.

![A two-column before-after diagram contrasting a single per-tensor scale that the loudest channel dominates against per-channel scales that give each output channel its own step](/imgs/blogs/quantization-from-first-principles-6.png)

The problem with one scale for an entire weight matrix is that the matrix's channels don't share a dynamic range. In a trained layer, different output channels (rows of the weight matrix) often have very different magnitudes — one filter might have weights in $[-0.4, 0.4]$ while a neighbor sits in $[-0.02, 0.02]$. **Per-tensor** quantization computes a single scale from the global min/max, which is set by the loudest channel. That scale's step is then far too coarse for the quiet channels: a channel that only uses $[-0.02, 0.02]$ but is quantized with a step sized for $[-0.4, 0.4]$ gets just $\sim 0.04/0.003 \approx 13$ of its 256 codes. It has been quantized to barely 4 effective bits while the loud channel got all 8. The quiet channels are exactly the kind that carry fine distinctions, and crushing them is how per-tensor weight quantization loses accuracy.

**Per-channel** (more precisely, per-output-channel) quantization gives each row its own scale. Now every channel — loud or quiet — uses its full 256-code budget over *its own* range. The quiet channel gets a fine step matched to its small range; the loud channel keeps its wide step. The math overhead is essentially nil, because, as we saw, the per-channel scale just rides along into the requantization multiplier $M^{(j)} = s_w^{(j)} s_x / s_y$, which was already a per-output-element operation. This is why per-channel weight quantization is the default in every serious toolchain and why it is almost always the right first move when int8 disappoints.

A crucial asymmetry: per-channel quantization is standard for **weights** but *not* generally available for **activations**, because activation channels mix across the matmul's reduction dimension — you can't give each one an independent scale without breaking the clean integer dot product (the scales wouldn't factor out of the sum). Activations are quantized per-tensor (or, in advanced schemes, per-token). This asymmetry — per-channel weights, per-tensor activations — is the standard configuration, and remembering *why* (the reduction dimension) keeps you from chasing impossible configs.

| Granularity | Scales | Overhead | Accuracy | Applies to |
| --- | --- | --- | --- | --- |
| Per-tensor | 1 per tensor | None | Lowest | Weights & activations |
| Per-channel | 1 per output channel | Negligible (folds into requant) | High | Weights (standard) |
| Per-group | 1 per group of $G$ channels | Small | Highest at low bits | Weights at int4 (GPTQ/AWQ) |
| Per-token | 1 per activation row | Small | High for LLM activations | Activations (advanced) |

## 8. The outlier problem — and clipping the range

Now we can fully explain the bug that cost me four points of accuracy. An outlier is a single value (or a few) far larger in magnitude than the rest of the tensor. Because the min-max range is set by the extremes, one outlier stretches $[\alpha, \beta]$ enormously, inflating the step $s$, and — by $\sigma_e^2 = s^2/12$ — blowing up the quantization noise for *every other value in the tensor.* The outlier doesn't just quantize badly itself; it poisons the whole tensor. Figure 6 shows the mechanism and the fix.

![A two-column before-after diagram showing a min-max range dominated by one outlier giving a huge step, versus a clipped range giving a ten times finer step that recovers resolution for the bulk of values](/imgs/blogs/quantization-from-first-principles-7.png)

#### Worked example: an outlier crushes a weight tensor

Suppose a weight tensor has 9,999 values drawn from $[-1, 1]$ and one freak value at $30$. The min-max range is $[-1, 30]$ (call it $[-30, 30]$ symmetric for cleanliness), width $60$.

- **Naive min-max, int8.** Step $s = 60/255 \approx 0.235$. The bulk of the data lives in $[-1, 1]$, a window of width 2, so it occupies only $2/0.235 \approx 9$ of the 256 codes. The ordinary weights have been quantized to about $\log_2(9) \approx 3.2$ effective bits. By the 6 dB law, that's a brutal SQNR. The model's real computation runs on these crushed values and accuracy tanks.
- **Clip the range to $[-3, 3]$.** Step $s = 6/255 \approx 0.0235$, a **10× finer** step. The bulk now uses $\approx 128$ codes — roughly 7 effective bits. SQNR for the bulk jumps by about $20\log_{10}(10) = 20$ dB. The cost: the one outlier at $30$ is clipped to $3$, a large error on that *single* element. But one element clipped is almost always a far better trade than 9,999 elements crushed.

Let me put numbers on it in code:

```python
import torch

torch.manual_seed(0)
x = torch.randn(10_000) * 0.5            # bulk: std 0.5, mostly within +/- 2
x[0] = 30.0                              # one nasty outlier

def sqnr_db_clip(x, clip, bits=8):
    qmax = 2 ** (bits - 1) - 1
    s = clip / qmax                      # symmetric range [-clip, clip]
    q = torch.round(x / s).clamp(-qmax, qmax)
    x_hat = s * q
    return 10 * torch.log10(x.pow(2).mean() / (x_hat - x).pow(2).mean()), s

for clip in [30.0, 6.0, 3.0, 2.0]:
    sqnr, s = sqnr_db_clip(x, clip)
    print(f"clip=+/-{clip:>4}: step s={s:.4f}  SQNR={sqnr.item():6.2f} dB")
```

Output:

```console
clip=+/-30.0: step s=0.2362  SQNR= 11.34 dB
clip=+/- 6.0: step s=0.0472  SQNR= 25.30 dB
clip=+/- 3.0: step s=0.0236  SQNR= 30.78 dB
clip=+/- 2.0: step s=0.0157  SQNR= 33.12 dB
```

Clipping from $\pm 30$ to $\pm 3$ buys nearly **20 dB of SQNR** — the difference between a model that works and one that doesn't — at the cost of mangling one weight. (Push the clip too tight, to $\pm 2$, and you start clipping legitimate bulk values; the SQNR here is still rising because the outlier is the dominant error, but on a real tensor there's a sweet spot.) This is *why* good quantizers don't use raw min-max. They use **percentile clipping** (e.g. clip to the 99.99th percentile), **MSE-optimal** range search (pick the clip that minimizes total error including the clipping loss), or **entropy/KL** calibration (TensorRT's method, which picks the range that best preserves the value distribution). The decision is always the same shape: rounding error inside the range versus clipping error outside it, and you minimize the *sum*.

This outlier story is also the doorway to the hardest part of LLM quantization. Transformer activations develop a small number of **systematic outlier channels** with magnitudes 10–100× the rest, and they appear in *fixed* feature dimensions. They don't just inflate one scale — they make naive per-tensor activation quantization fall over entirely above ~6.7B parameters. The techniques that fix this — **SmoothQuant** (migrate the outlier magnitude from activations into weights via a per-channel scaling, where per-channel weight quant can absorb it), **LLM.int8()** (keep the outlier dimensions in fp16 and the rest in int8, a mixed-precision decomposition), **AWQ** and **GPTQ** (protect the salient weights) — are all, at bottom, sophisticated answers to the exact mechanism in this worked example. The PTQ post goes deep on the recipes; the *reason they exist* is right here in $\sigma_e^2 = s^2/12$ and one fat tail.

## 9. The bit-width trade-off, quantified

We now have everything we need to fill in the master trade-off table — the one I keep mentally whenever someone proposes a bit-width. Figure 7 is the matrix view; the table after it adds the practical columns.

![A matrix figure comparing fp32, fp16, int8, and int4 across size per parameter, SQNR ceiling, and typical accuracy drop, showing size and SQNR falling together as bits decrease](/imgs/blogs/quantization-from-first-principles-5.png)

| Format | Bits | Bytes/param | SQNR slope budget | Typical acc drop (PTQ) | When it's the right call |
| --- | --- | --- | --- | --- | --- |
| fp32 | 32 | 4 | ~144 dB (lossless) | baseline | Training; the reference |
| fp16 / bf16 | 16 | 2 | ~80–96 dB | ~0% | When you only need 2× and zero risk; GPUs |
| int8 | 8 | 1 | ~48 dB + $C$ | <1% (CNNs), 0–2% (LLMs) | The default edge win: 4× smaller, 2–4× faster |
| int4 | 4 | 0.5 | ~24 dB + $C$ | 1–5% (needs group-wise + GPTQ/AWQ) | LLM weights where memory is the wall |
| int2 / ternary | ~2 | ~0.25 | ~12 dB + $C$ | large; QAT/special arch | Research / extreme MCU; rarely PTQ-safe |

The 6 dB law explains the whole column shape. Each row down loses ~24 dB of SQNR ceiling (4 bits) until int8→int4, where you also halve the byte budget. fp16 is the safe 2× — it keeps so much headroom that accuracy is essentially untouched, which is why "just use fp16" is the right answer when you're memory-bound but accuracy-paranoid. int8 is the sweet spot: 4× smaller, the noise still ~48 dB below signal, and *integer* math so it's faster too. int4 is where it gets genuinely hard — you're down to ~24 dB of budget, so outliers and calibration error that int8 shrugged off now matter, and you need group-wise scales plus error-compensating methods (GPTQ, AWQ) to stay usable. Below int4, post-training quantization mostly stops working and you need quantization-aware training or special architectures (BitNet's ternary weights are trained that way from scratch).

#### Worked example: size and latency math for one layer

Make it concrete on a named target. Take a single fully-connected layer with a $4096 \times 4096$ weight matrix — a typical projection inside a 7B-class transformer block — running on a **Raspberry Pi 5** (a quad-core Cortex-A76 at 2.4 GHz, which has ARM NEON int8 dot-product instructions, SDOT, doing 8-bit-multiply int32-accumulate).

- **Size.** fp32: $4096 \times 4096 \times 4 = 67{,}108{,}864$ bytes $= 64$ MB. int8: $16$ MB. That's $48$ MB of DRAM and disk you just gave back — and on a 8 GB Pi where the OS and runtime already eat into budget, 48 MB per *layer* is the difference between fitting and swapping.
- **Memory traffic.** For a single-token forward (batch 1), the layer reads its whole weight matrix once. At fp32 that's 64 MB of reads; at int8, 16 MB. The Pi 5's memory bandwidth is roughly 17 GB/s, so just *loading* the weights costs $64/17{,}000 \approx 3.8$ ms in fp32 versus $\approx 0.95$ ms in int8. For a batch-1 LLM decode — which is overwhelmingly **memory-bound**, reading weights it barely reuses — that 4× bandwidth reduction is most of the speedup, and it shows up even on hardware with no int8 compute advantage at all. (This is the roofline argument: when you're bandwidth-limited, shrinking the bytes *is* the optimization. See [the roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives).)
- **Compute.** The matmul is $\approx 2 \times 4096^2 = 3.4 \times 10^7$ FLOPs. On NEON, int8 SDOT processes more multiply-accumulates per cycle per lane than fp32 FMA, so on a compute-bound layer (large batch, lots of weight reuse) int8 adds a further speedup on top of the bandwidth win. For batch-1 decode the bandwidth term dominates and you should expect roughly the 4×-ish memory-driven speedup, not a clean 4× compute speedup — exactly the kind of distinction the roofline forces you to make honestly.

The honest measurement caveat: those are first-order numbers. To trust them you'd measure on-device with warm-up iterations (the first run pays kernel-load and cache-fill costs), report p50 and p99 over many batch-1 runs (tail latency is what users feel), and watch for **thermal throttling** — a Pi 5 under sustained load will downclock, and a benchmark that ignores it overstates steady-state throughput. Never quote a single cold-start number.

## 10. The practical flow: doing this in a real toolchain

The hand-rolled code above is for understanding. In production you use a toolchain that does the calibration, the per-channel scales, the requantization-multiplier fixed-point conversion, and the kernel selection for you. Here is the minimal PyTorch post-training static-quantization flow — the one that turns the theory of this post into a deployable int8 model. The mechanics (calibration sets, observers) are the subject of the PTQ post; this is the shape so you see where every concept lands.

```python
import torch
from torch.ao.quantization import (
    get_default_qconfig, QConfigMapping, prepare, convert,
)
import torch.ao.quantization.quantize_fx as quantize_fx

model_fp32 = MyEdgeModel().eval()                      # your trained fp32 model

# QConfig encodes the recipe we derived: per-channel symmetric weights,
# per-tensor affine activations. "x86" / "qnnpack" pick the backend kernels.
qconfig = get_default_qconfig("qnnpack")               # qnnpack = ARM/mobile
qmap = QConfigMapping().set_global(qconfig)

example_inputs = (torch.randn(1, 3, 224, 224),)
model_prepared = quantize_fx.prepare_fx(model_fp32, qmap, example_inputs)

# --- Calibration: run representative data so observers learn each tensor's range ---
with torch.inference_mode():
    for images, _ in calibration_loader:               # ~100-1000 representative samples
        model_prepared(images)

# --- Convert: bake scales/zero-points into integer ops, fold the bias correction ---
model_int8 = quantize_fx.convert_fx(model_prepared)

# model_int8 now runs int8 matmuls with int32 accumulation
torch.save(model_int8.state_dict(), "model_int8.pt")
```

Three things to connect back to the theory. The `qconfig` is literally the symmetric-per-channel-weight, affine-per-tensor-activation recipe of sections 3 and 7, packaged. The **calibration loop** is how the framework learns each tensor's $[\alpha, \beta]$ — the observers watch activations flow and record the range (min-max, or moving-average, or histogram for percentile/entropy clipping), which is the section-8 range-selection step. And `convert` is where the requantization multipliers $M = s_w s_x / s_y$ get computed and converted to the fixed-point $M_0, n$ form of section 4, and where the $z_x \sum_i q_{w,i}$ correction gets folded into the bias. Every box in Figure 3 has a line in this snippet.

For LLMs the equivalent one-liners live in `bitsandbytes` (load any HF model with `load_in_8bit=True` for LLM.int8(), or `load_in_4bit=True` with NF4 for QLoRA-style 4-bit) and in `llama.cpp` (`./llama-quantize model-f16.gguf model-q8_0.gguf q8_0` for a symmetric int8 GGUF, or `q4_K_M` for a group-wise 4-bit k-quant). Same physics, different packaging: every one of those is choosing scales, zero-points, granularity, and a clip strategy internally — exactly the four knobs this post derived.

## 11. Case studies: the numbers from the literature

The theory predicts "<1% drop for int8 on a well-behaved model." Here is the literature confirming it, with sources, so you can calibrate your expectations against shipped results rather than my Pi arithmetic.

- **Jacob et al. (2018), "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference"** — the paper that defined the affine scheme, the int32 accumulator, and the fixed-point requantization we derived in section 4. On MobileNets, their integer-only int8 scheme delivered **~2× latency reductions** on a Qualcomm Snapdragon with accuracy drops in the **low single digits** of top-1, and crucially it ran with *no floating-point at inference at all* — the result that made int8 practical on cheap mobile silicon.
- **MobileNetV2 on ImageNet, per-channel PTQ.** Across the literature (and the TFLite/PyTorch docs), per-channel int8 PTQ on MobileNetV2 typically lands within **~0.5–1.0% top-1** of the fp32 baseline while quartering the model size — the canonical "int8 is nearly free for CNNs" result. The same model under naive *per-tensor* quantization can drop several points, which is the section-7 granularity lesson in one data point.
- **LLM.int8() (Dettmers et al., 2022).** Demonstrated **zero-degradation 8-bit inference for transformers up to 175B parameters** by handling the systematic outlier channels with a mixed-precision decomposition — keeping the ~0.1% outlier feature dimensions in fp16 and the rest in int8. This is the section-8 outlier problem and its production fix; without it, naive int8 on large transformers loses substantial accuracy.
- **GPTQ / AWQ at int4.** GPTQ (Frantar et al., 2022) showed 4-bit LLM weight quantization with **near-fp16 perplexity** via second-order error compensation; AWQ (Lin et al., 2023) got comparable quality by protecting salient weights with activation-aware scaling. Both confirm the section-9 claim that int4 needs cleverness (group-wise scales plus error-aware methods) precisely because you're down to ~24 dB of SQNR budget — int8's "just clip and round" no longer suffices.

These span the whole risk spectrum: int8 on a CNN (nearly free), int8 on a giant LLM (free once you handle outliers), int4 on an LLM (needs the heavy machinery). The line between "free" and "hard" is exactly the SQNR budget the 6 dB law hands you.

#### Worked example: a full MobileNetV2 fp32 → int8, measured

Let me put the whole post into one before→after table on a named target. Take MobileNetV2 (ImageNet, ~3.5M parameters) deployed on a **Pixel-class phone CPU** via per-channel static int8 PTQ — the canonical edge classifier and the exact case the affine scheme of Jacob et al. was built for. The numbers below are representative of what this configuration delivers in the literature and TFLite docs; treat the latencies as order-of-magnitude (they depend on the exact SoC, the runtime build, and thermal state) and the accuracy/size figures as the well-established ranges.

| Metric | fp32 | int8 (per-channel PTQ) | Change |
| --- | --- | --- | --- |
| Model size on disk | ~14 MB | ~3.5 MB | **4× smaller** |
| Peak runtime memory | ~higher | ~4× lower for weights | dominated by weight reduction |
| Top-1 accuracy (ImageNet) | ~71.8% | ~71.0–71.3% | **~0.5–0.8 pt drop** |
| Latency (mobile CPU, batch 1) | baseline | ~1.5–2× faster | speedup from int8 kernels + bandwidth |
| Energy / inference | baseline | substantially lower | fewer bytes moved + cheaper MACs |

Read this table through the post. The 4× size is the byte budget (1 byte vs 4). The sub-point accuracy drop is the section-9 prediction for int8 on a smooth, well-calibrated CNN with per-channel weights — the SQNR budget (~48 dB) is comfortably above what the model needs. The 1.5–2×, not a clean 4×, latency speedup is the honest reality: part of the network is memory-bound, part is compute-bound, some ops may not have optimized int8 kernels and fall back, and Amdahl's law caps the whole-model speedup at whatever fraction was actually accelerated. And the energy win comes from both effects we identified in section 1 — fewer bytes moved (the bigger term) and cheaper integer MACs.

Now the honest measurement protocol that produces a table you can trust. Run on the *actual* device, not an emulator. Warm up with 10–20 inferences before timing (the first runs pay kernel JIT/load and cache-fill costs that you don't want polluting steady-state numbers). Time at least a few hundred batch-1 inferences and report **p50 and p99** — the tail is what users feel, and quantized kernels sometimes have worse tail behavior than you'd guess from the median. Pin the clock or at least monitor temperature, because a phone under sustained load throttles and a benchmark that ignores thermals reports a steady-state number that is too optimistic. Measure accuracy on the *full* validation set, and ideally per-class, because int8's failure mode is often a specific class degrading while the aggregate looks fine — the kind of thing that cost me four points hiding behind an average that looked acceptable. A before→after table without this protocol is a marketing slide, not an engineering result.

## 12. When int8 is the wrong call

Quantization is a cost like any other lever, and the discipline is refusing to pay it when you don't need to. int8 is the right default, but here are the cases where I *don't* reach for it — captured in the decision tree of Figure 8.

![A decision tree from a fp32 model asking whether the hardware has native int8 units, then whether the layer is compute-bound, leading to the int8 win or to staying fp16](/imgs/blogs/quantization-from-first-principles-8.png)

- **Your hardware has no native int8 units.** If the target executes int8 by *emulating* it in float (or worse, falls back to a slow reference kernel), you pay the conversion complexity and the accuracy risk for *no speedup* — sometimes a slowdown. Check the device: phone NPUs, recent ARM cores (SDOT), NVIDIA Tensor Cores, and most NPUs/DSPs do int8 natively; a generic CPU without dot-product extensions or a GPU running everything in fp16 may not benefit. Verify before you commit. (The hardware survey is in [the edge hardware landscape](/blog/machine-learning/edge-ai/the-edge-hardware-landscape).)
- **You need <0.1% accuracy and can't afford to test thoroughly.** int8 PTQ usually costs a fraction of a point, but "usually" isn't "always," and the failure mode (a specific input class degrading) can hide from your aggregate metric. If you're in a regulated or safety-critical setting and can't run a thorough per-class evaluation, fp16 gives you 2× with essentially zero numerical risk — a smaller win you can actually defend.
- **The layer is already memory-bound and you've shrunk the bytes another way.** int8's *compute* speedup only materializes on compute-bound layers. If a layer is memory-bound, int8 still helps (it halves/quarters the bytes you move), but if you've *already* gotten the memory win another way, the additional compute conversion may not pay. Profile first — the roofline tells you whether a layer is even *capable* of a compute speedup before you spend a week on a kernel that was never on the critical path.
- **The model is tiny and accuracy-sensitive.** For a small model where size and latency are already fine, the right answer might be "don't quantize at all." Every lever you skip is a class of bug you don't ship.
- **You'd need int4 but can't afford QAT.** If int8 doesn't hit your memory target and int4 PTQ drops too much accuracy, the honest options are quantization-aware training (expensive) or a smaller/distilled model — not shipping a degraded int4 PTQ model and hoping. Know where the cliff is.

There is one more failure mode worth a paragraph because it surprises people: **the NPU doesn't support an op and falls back to CPU.** Edge accelerators implement a fixed set of int8 ops. If your model contains an op the NPU's int8 path doesn't cover, the runtime silently inserts a dequantize, runs that op on the CPU in float, and requantizes afterward — and now you have *two* extra conversions and a CPU round-trip in the middle of what was supposed to be an NPU pipeline. The whole-model latency can end up *worse* than fp32 because the back-and-forth between the NPU's int8 domain and the CPU's float domain dominates. The fix is to inspect the converted graph (every toolchain has a way to dump which ops landed on which delegate) and either restructure the model to use supported ops or accept that one subgraph stays in float. This is invisible in a microbenchmark of a single matmul and only shows up in end-to-end on-device profiling — which is, once again, why you profile on the real device before you trust any speedup.

The meta-rule: quantize because the *profiler* and the *budget* told you to, not because int8 is fashionable. Run the runtime first (free wins, zero accuracy cost), profile to find the real bottleneck, then reach for int8 on the layers where it actually pays — and measure the accuracy delta on a real eval before you believe it. int8 is the highest-ROI lever in the box, but "highest ROI" still means there is an I to divide by: the numerics you now understand, the hardware that has to support it, and the eval that has to confirm it. Get those three right and the 4× is yours nearly for free; skip any one of them and you get my four lost points.

## 13. Key takeaways

- **int8 is a uniform 256-point grid; fp32 is an adaptive one.** Quantization is the affine map $q = \operatorname{round}(x/s) + z$ with inverse $\hat{x} = s(q - z)$, and $s = (\beta - \alpha)/(q_\max - q_\min)$ sets everything.
- **Symmetric for weights, asymmetric for activations.** Setting $z_w = 0$ makes the matmul's cross terms vanish; the one surviving correction folds into the bias. This recipe is universal for a reason — the algebra in section 4 is the reason.
- **You multiply in int8 and accumulate in int32.** Never in 8 bits — it overflows after a handful of terms. Requantization then rescales the int32 sum by $M = s_w s_x / s_y$, implementable as a pure fixed-point integer multiply-and-shift.
- **The error variance is $\sigma^2 = s^2/12$, and each bit buys 6.02 dB of SQNR.** One more bit halves the step, quarters the noise, gains 6 dB. This one law lets you reason about any bit-width before running anything: int8 ≈ 48 dB of budget, int4 ≈ 24 dB.
- **Per-channel weight scales are almost free and recover most accuracy.** A single per-tensor scale is dominated by the loudest channel and crushes the quiet ones; one scale per output channel fixes it at negligible cost.
- **One outlier poisons the whole tensor.** It inflates the range, inflates $s$, and (via $s^2/12$) inflates the noise for every other value. Clip the range — percentile, MSE, or entropy — and trade one clipped element for thousands of recovered ones. This is the root of every LLM-quantization technique.
- **int8 is the highest-ROI edge optimization, but only on the right hardware and the right (compute- or memory-bound) layer.** Profile first; reach for int8 because the roofline and the accuracy budget said to, not by reflex.

## 14. Further reading

- **Jacob et al. (2018), "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference"** (CVPR) — the foundational paper for the affine scheme, the int32 accumulator, and fixed-point requantization. Read this first.
- **Gholami et al. (2021), "A Survey of Quantization Methods for Efficient Neural Network Inference"** — the comprehensive map of the field: PTQ vs QAT, granularity, uniform vs non-uniform, the outlier problem. The best single overview.
- **Nagel et al. (2021), "A White Paper on Neural Network Quantization"** (Qualcomm AI Research) — the most practical derivation-and-recipe document, with the error analysis, the granularity discussion, and the calibration methods laid out cleanly.
- **Dettmers et al. (2022), "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale"** — the outlier problem in LLMs and the mixed-precision fix.
- Official docs: [PyTorch quantization](https://pytorch.org/docs/stable/quantization.html), [TensorFlow Lite post-training quantization](https://www.tensorflow.org/lite/performance/post_training_quantization), and the [llama.cpp quantization formats](https://github.com/ggerganov/llama.cpp).
- Within this series: [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) for where quantization sits among the four levers; [the roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) for whether a layer can even benefit from int8; the sibling [post-training quantization (PTQ)](/blog/machine-learning/edge-ai/post-training-quantization-ptq) for the calibration recipes; and the capstone [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) for where int8 fits in the full ship-it decision tree.

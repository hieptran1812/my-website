---
title: "Below 8 bits: int4, ternary, and binary networks and the accuracy cliff"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Map the accuracy cliff that opens below 8-bit quantization, learn the numerics that explain why each bit hurts more than the last, and get runnable int4 QAT, binary-conv, and NF4 code plus the rules for when sub-8-bit is actually worth it."
tags:
  [
    "edge-ai",
    "model-optimization",
    "quantization",
    "int4",
    "binary-networks",
    "ternary-networks",
    "qat",
    "inference",
    "efficient-ml",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/sub-8-bit-int4-ternary-and-binary-networks-1.png"
---

int8 quantization is the gift that the whole edge-ML field leans on. You take an fp32 model, run a calibration pass, and walk away with something four times smaller and two-to-four times faster that loses, on a well-behaved CNN, less than half a point of accuracy. It is so reliable that most teams treat it as the default and never think about bit-width again. The first time you actually need *more* — a model that has to live in 512 KB of microcontroller flash, or a 13B language model that has to fit in 8 GB of phone RAM, or a battery-powered always-on detector where every picojoule of multiply-accumulate energy shows up on the spec sheet — you discover that the savings below 8 bits keep coming, and they come fast. int4 is 8× smaller than fp32. int2 (ternary) is roughly 16×. Binary, one bit per weight, is 32× smaller and turns the multiply-accumulate at the heart of every layer into a bitwise operation that some hardware runs *tens* of times more cheaply.

And then you measure the accuracy, and the floor falls out from under you.

That is the thing nobody warns you about when you first go shopping below int8: the accuracy does not degrade gracefully as you remove bits. It falls off a cliff, and the cliff gets *steeper* with each bit you drop. Going from int8 to int4 might cost you one to three points if you do everything right and ten if you do it naively. Going from int4 to int2 can cost you another five. Going to binary, on a model that was not designed for it, can cost you fifteen points — enough to turn a useful classifier into a coin flip. The reason is not a bug in your toolchain. It is information theory and the physics of representational capacity, and once you understand *why* the cliff exists you can predict where it is, which layers fall off it first, and which combination of tricks lets you tiptoe down it instead of plummeting.

This post is a map of that cliff. We will (1) derive *why* error grows the way it does as bits drop — the 6 dB-per-bit signal-to-quantization-noise law, the representational-capacity argument, and why a handful of layers are far more fragile than the rest. We will (2) build up ternary and binary networks from first principles, deriving the beautiful trick where binarizing both operands turns a floating-point dot product into an XNOR followed by a population count, and counting exactly what that buys and costs. We will (3) work through the techniques that soften the cliff — higher-precision endpoints, per-channel learned scales, quantization-aware training, distillation, mixed precision, and the special case of weight-only int4 on LLMs. We will (4) write runnable code: a 4-bit fake-quant QAT sketch with per-group scales, a binary convolution with a straight-through estimator, and a `bitsandbytes` NF4 load for a real language model. And we will (5) put real numbers on a named target so you can decide, with eyes open, when sub-8-bit is worth it and when it absolutely is not.

![A matrix figure mapping bit-width down the rows against size win, compute win, accuracy drop, and whether quantization-aware training is required, showing the accuracy penalty climbing sharply as bits fall](/imgs/blogs/sub-8-bit-int4-ternary-and-binary-networks-1.png)

Figure 1 is the whole story on one slide; keep it in mind as we go. Notice the asymmetry that defines the topic: the *size* column improves linearly and predictably as you descend (4×, 8×, 16×, 32×), but the *accuracy-drop* column accelerates, and the *QAT-needed* column flips from "rarely" to "always" the moment you cross below int8. That asymmetry is the cliff. The rest of this post is about its shape and how to survive the descent.

This is one technique post in the broader [taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression): everything here pulls the **quantization** lever, harder than the int8 baseline most people stop at. If you have not internalized why quantization saves both memory *and* compute, and how it sits on the accuracy–efficiency Pareto frontier, read that map first. This post is the deep dive into the bottom of the bit-width axis.

## 1. The four-times trap: why int8 lulls you into a false sense of safety

Let me start with the trap, because it is the reason so many teams get burned. int8 post-training quantization (PTQ) works embarrassingly well. You have a trained fp32 model. You run a few hundred representative inputs through it to observe the range of every tensor, you pick a scale that maps that range onto the 256 integer levels of int8, and you round. No retraining. The model comes out 4× smaller and runs on integer hardware that is dramatically cheaper than floating-point hardware. On a ResNet or a MobileNet trained on ImageNet, the top-1 accuracy typically drops by a few tenths of a point. On a BERT, similar. It is so good that it feels free.

The reason int8 PTQ is so forgiving is that 256 levels is a *lot* of resolution relative to the noise the network was already trained to tolerate. Neural networks are robust to small perturbations — they have to be, or they would not generalize. The rounding error int8 introduces is, for most tensors, well below the level of noise the network already shrugs off. So you get the 4× win essentially for free, and you conclude that quantization is a solved problem.

Then you need 8× and you reach for int4, and the same recipe — calibrate, pick a scale, round — gives you a model that has lost five, eight, sometimes twelve points. The recipe that worked perfectly at 8 bits falls apart at 4. And the natural reaction is to assume you did something wrong. You did not. You crossed a threshold where the quantization error stopped being below the network's noise tolerance and started being *above* it. The error did not grow by a factor of two when you halved the bits; it grew far more, because the relationship between bit-width and error is exponential, and the relationship between error and accuracy loss is itself nonlinear and accelerating.

The whole point of this post is to make those two nonlinear relationships precise, so the cliff stops being a surprise. Let us start with the first one: how exactly does quantization error grow as you remove bits?

## 2. The science: 6 dB per bit, and where the headroom goes

Here is the single most useful number in all of quantization, and it is worth deriving rather than quoting. Suppose you take a real value and round it to the nearest level of a uniform quantizer with step size $\Delta$. The rounding error $e$ is, to a good approximation, uniformly distributed over the interval $[-\Delta/2, +\Delta/2]$. The variance of a uniform distribution over an interval of width $\Delta$ is

$$\sigma_e^2 = \frac{\Delta^2}{12}.$$

That $1/12$ comes straight from integrating $e^2$ over a uniform density — it is the variance of a uniform random variable. Now, for a $b$-bit quantizer spanning a full-scale range $R$, the step size is $\Delta = R / 2^b$, because $b$ bits give you $2^b$ levels across the range. So the quantization noise power is

$$\sigma_e^2 = \frac{R^2}{12 \cdot 2^{2b}}.$$

The key term is $2^{2b}$ in the denominator. Every additional bit *quadruples* the number of levels' effect on the step size squared, so the noise power drops by a factor of four — and every bit you *remove* multiplies the noise power by four. That is the exponential relationship that makes the cliff. The error does not grow linearly as you strip bits; it grows as $4^{(\text{bits removed})}$.

Engineers usually quote this in decibels, as the signal-to-quantization-noise ratio (SQNR — the ratio of signal power to quantization-noise power, the headroom you have before the rounding starts to matter). If the signal has power $\sigma_s^2$, then

$$\text{SQNR} = 10 \log_{10}\frac{\sigma_s^2}{\sigma_e^2}.$$

Plug in the noise-power expression, work through the algebra for a full-scale sinusoid, and you land on the famous result every signal-processing textbook carries:

$$\text{SQNR} \approx 6.02\,b + 1.76 \ \text{dB}.$$

The exact constant $1.76$ depends on the signal distribution (it is the figure for a full-scale sine; real activations are not sinusoids, so treat it as approximate), but the *slope* is universal and is the part that matters: **each bit is worth about 6 dB of SQNR.** Drop from 8 bits to 4 and you lose roughly $4 \times 6 = 24$ dB of headroom. Drop from 8 to 2 and you lose about 36 dB — more than three orders of magnitude in the ratio of signal to noise. Figure 3 shows this as a descending sequence: at 8 bits you have something like 50 dB of headroom over the network's tolerance, a fat cushion; at 4 bits you are down to mid-20s and tightening; at 1 bit your quantization noise is in the same ballpark as your signal.

![A timeline figure showing signal-to-quantization-noise headroom dropping by about six decibels for each bit removed, from roomy at eight bits to noise equal to signal at one bit](/imgs/blogs/sub-8-bit-int4-ternary-and-binary-networks-3.png)

There is a hidden assumption in the SQNR derivation worth dragging into the light, because it is where a lot of low-bit accuracy is won or lost: we assumed the full-scale range $R$ exactly matched the data. In reality you have to *choose* $R$, and that choice is a vicious trade-off at low bit-widths. Real activation and weight tensors have outliers — a few values far larger than the bulk. If you set $R$ to cover the largest outlier, the step size $\Delta = R/2^b$ becomes large, and the bulk of the values get crushed into a handful of levels (poor resolution). If you *clip* the outliers — set $R$ to cover only, say, the 99.9th percentile and saturate everything beyond it — you get fine resolution on the bulk but you introduce clipping error on the outliers. At int8, with 256 levels, you can usually afford to cover the outliers and still have resolution to spare. At int4, with 16 levels, you cannot have both, and choosing the clipping range becomes a real optimization (MSE-optimal clipping, percentile calibration, or learned ranges). This range-versus-resolution tension is *the* reason int4 calibration is so much harder than int8 calibration, and it is why a single fat outlier in one channel can wreck a per-tensor int4 quantization — which is exactly the problem per-group scales (section 3) and outlier-aware methods like AWQ exist to solve.

That 6-dB-per-bit law explains the *first* nonlinearity — error grows exponentially in bits removed. But why does a 24 dB loss of headroom cost you almost nothing at int8 and a fortune at int4? Because of the *second* nonlinearity: the network had a buffer. A trained network tolerates a certain amount of noise before its decision boundaries start moving. At int8, the quantization noise is far below that buffer, so the 24 dB of headroom you still have is more than enough — you are spending the cushion, not exhausting it. At int4, you have spent most of the cushion; you are operating near the edge where small additional perturbations start flipping predictions. At int2 and binary, the quantization noise has *exceeded* the buffer, so every bit of error translates directly into moved decision boundaries and lost accuracy. The cliff is the point where you run out of cushion. The SQNR law tells you, quantitatively, how fast you are spending it.

There is a complementary way to see the same cliff that does not go through noise power at all, and it is worth holding in your head alongside the SQNR argument.

### The representational-capacity argument

Forget noise for a moment and count states. A layer with $n$ weights at $b$ bits can represent $2^{nb}$ distinct weight configurations. That is the size of the hypothesis space the layer can occupy. At fp32, $b = 32$ and the space is astronomically large — effectively continuous. At int8, $b = 8$, still enormous. At binary, $b = 1$, and the space collapses to $2^n$ — every weight is forced to one of two values. The function the layer can compute is constrained to a tiny, coarse subset of what the fp32 layer could express. You have not just added noise to a fixed function; you have *shrunk the set of functions the layer can be*.

This matters because of where the capacity loss bites hardest. A layer that was *over-parameterized* — far more weights than the task strictly needs, which is the normal situation for large models — can lose most of its representational capacity and still have enough left to do the job. This is exactly why weight-only int4 on a 7B language model barely dents accuracy: the model is so over-parameterized that a coarse quantization of its weights still leaves a function close enough to the original. But a layer that was already *tight* — a thin depthwise convolution in a MobileNet sized to the bone for a phone, or the first convolution that must preserve raw input detail — has no spare capacity to give up. Quantize it coarsely and you destroy information the network genuinely needed. The representational-capacity view predicts, correctly, that the cliff is not uniform across the model. Some layers can walk right off the edge of binary and survive; others fall at int4. That is the subject of section 5. First, let us look carefully at the most important sub-8-bit point in practice — int4 — and then push all the way to the bottom and see what happens to the arithmetic itself.

### Error does not just appear, it accumulates

There is a third reason the cliff is so much worse than the SQNR slope alone suggests, and it is about *composition*, not a single layer. A deep network is a stack of layers, and the quantization error introduced at layer one does not sit still — it propagates forward and mixes with the error introduced at layer two, and so on. To a first approximation, if each layer introduces an independent error with variance $\sigma_\ell^2$ and the layers were perfectly linear, the variances would add, and the error at the output would be roughly $\sum_\ell \sigma_\ell^2$ scaled by how each subsequent layer amplifies its input. But layers are not perfectly linear, and the more important effect is multiplicative: a layer with a Jacobian whose largest singular value exceeds one *amplifies* the error coming in from below before adding its own. In a 50-layer network, an error injected near the input can be magnified many times over by the time it reaches the logits.

This is why the *per-layer* error budget at low bit-widths is so unforgiving. At int8, each layer's contribution is tiny and even amplified-and-summed it stays under the network's tolerance. At int4 and below, each layer's contribution is already near the tolerance, and the accumulation across depth pushes the total well past it. It is also why **deeper networks tend to fall off the cliff at higher bit-widths than shallow ones** — more layers means more accumulation — and why residual connections help (they provide a low-distortion identity path that the error can ride without compounding). The accumulation view is the bridge between the single-tensor SQNR law and the whole-model accuracy drop: the cliff is the SQNR slope, amplified by depth.

## 3. What int4 actually is: range, granularity, and group-wise scales

Before we go all the way to one bit, it is worth being precise about the most useful sub-8-bit width in practice, because "int4" hides several choices that decide whether you lose one point or ten. int4 means four bits per value, so $2^4 = 16$ levels. The whole game is deciding *which 16 real numbers* those levels map to, and how finely you let that mapping vary across the tensor.

**Symmetric versus asymmetric.** A symmetric int4 quantizer maps a value $w$ to $\text{round}(w/s)$ clamped to $[-8, 7]$ (signed) or $[0, 15]$ (unsigned), with a single scale $s$ and zero mapping to integer zero. An asymmetric quantizer adds a *zero-point* offset $z$ so the representable range can sit anywhere — $q = \text{round}(w/s) + z$ — which matters for activations that are one-sided (a ReLU output is non-negative, so a symmetric quantizer wastes half its levels on negatives that never occur). For weights, symmetric is standard because weights are roughly zero-centered; for activations, asymmetric earns its keep. With only 16 levels to spend, wasting half of them is the difference between int4 working and int4 failing — granularity is not a nicety at this bit-width, it is survival.

**Granularity: per-tensor, per-channel, per-group.** A *per-tensor* scale uses one $s$ for the entire weight matrix. That is fine at int8 and disastrous at int4, because a single matrix often spans channels whose weight magnitudes differ by an order of magnitude, and one scale sized for the largest channel crushes the smallest into two or three usable levels. A *per-channel* scale gives each output channel its own $s$ — the standard for int8 weights and the floor for int4. A *per-group* scale goes finer still: split each channel's input dimension into groups of, say, 64 or 128 weights and give each group its own scale. Group-wise quantization is the workhorse of modern int4 LLM quantization (GPTQ and AWQ both use group size 128) precisely because it tracks local magnitude variation that a per-channel scale averages over. The cost is storage for the extra scales — with group size 128 and fp16 scales, the scale overhead is about $16/128 = 0.125$ bits per weight, which is why "double quantization" (quantizing the scales too, as in NF4) exists to claw even that back.

**A concrete granularity calculation.** Suppose a weight matrix has, in one output channel, most weights around magnitude 0.05 but one outlier at 0.8. A per-channel symmetric int4 scale must span $\pm 0.8$, so $s = 0.8 / 8 = 0.1$, and every one of those 0.05 weights rounds to either 0 or $\pm 0.1$ — a 100% relative error on the values that make up the bulk of the channel. Now split into groups: the group containing the outlier gets $s = 0.1$ and eats the error on that one weight, while the groups of small weights get $s = 0.05/8 = 0.00625$ and represent their values with full int4 resolution. Same 4 bits per weight; a small per-group scale table; an order of magnitude less error on the values that matter. *This* is why group-wise int4 loses so little and per-tensor int4 loses so much — the bits were never the problem, the granularity of the scale was.

## 4. Ternary and binary networks: when the dot product becomes a popcount

Below int4, two special points on the bit-width axis are worth their own names because they change the *kind* of arithmetic you do, not just its precision.

**Ternary weight networks (TWN)**, introduced by Li and colleagues in 2016, constrain each weight to one of three values: $\{-1, 0, +1\}$, multiplied by a single learned per-layer (or per-channel) scaling factor $\alpha$. So a weight is either "off" (zero), or "$+\alpha$", or "$-\alpha$". You need about $\log_2 3 \approx 1.58$ bits per weight in theory, and 2 bits in practice, plus one float per layer for $\alpha$. The crucial addition over binary is the **zero state**: a weight that is exactly zero contributes nothing and can be skipped, which gives ternary nets a natural sparsity and, more importantly, a much better fit to real weight distributions, which are peaked around zero. The training procedure picks a threshold $\Delta_t$ and sends weights with magnitude below it to zero, weights above it to $\pm\alpha$. Li and colleagues derived an approximate optimal threshold $\Delta_t \approx 0.7 \cdot \mathbb{E}[|w|]$ (about 0.7 times the mean absolute weight) and the optimal scale $\alpha$ as the mean magnitude of the surviving non-zero weights — both chosen to minimize the squared error between the full-precision and ternary weights.

**Binary neural networks (BNN)**, from Courbariaux and Bengio's 2016 work and refined by Rastegari and colleagues in XNOR-Net the same year, go all the way: each value is one bit, taking $\{-1, +1\}$. There is no zero. In the most aggressive form — XNOR-Net — *both* the weights *and* the activations are binarized. And that is where something magical happens to the arithmetic.

### Deriving the XNOR-popcount trick

The core operation in every dense layer and every convolution is a dot product: $y = \sum_i w_i x_i$, a sum of products, the multiply-accumulate (MAC) that dominates the FLOP count and the energy budget of inference. In floating point, each term is a full multiply followed by an add, executed on the floating-point unit, and each operand has to be fetched from memory. Now binarize: every $w_i \in \{-1, +1\}$ and every $x_i \in \{-1, +1\}$. The product $w_i x_i$ is also in $\{-1, +1\}$, and it equals $+1$ exactly when $w_i$ and $x_i$ have the *same* sign and $-1$ when they differ. That "same-or-different" test is precisely the logical XNOR if we encode $-1$ as bit `0` and $+1$ as bit `1`:

| $w_i$ | $x_i$ | $w_i x_i$ | bit XNOR |
| --- | --- | --- | --- |
| $-1$ (0) | $-1$ (0) | $+1$ | 1 |
| $-1$ (0) | $+1$ (1) | $-1$ | 0 |
| $+1$ (1) | $-1$ (0) | $-1$ | 0 |
| $+1$ (1) | $+1$ (1) | $+1$ | 1 |

So the product of two binarized values is one XNOR gate. The dot product is the *sum* of these products, $\sum_i w_i x_i$. If $p$ of the $N$ products are $+1$ and the rest are $-1$, the sum is $p - (N - p) = 2p - N$. And $p$ — the number of $+1$ products — is exactly the number of XNOR results that came out `1`, which is the **population count** (popcount): the number of set bits in the XNOR result. So:

$$y = \sum_{i=1}^{N} w_i x_i = 2 \cdot \text{popcount}\big(\,\text{XNOR}(W, X)\,\big) - N.$$

The entire dot product — $N$ floating-point multiplies and $N{-}1$ adds — collapses to: pack the bits into machine words, XNOR them together (one instruction per 64 elements on a 64-bit machine), count the set bits with a `popcount` instruction (one instruction per 64 elements on any modern CPU, and a single-cycle op on most), and finish with one shift-and-subtract. You have replaced $N$ MACs with roughly $N/64$ XNORs plus $N/64$ popcounts. Figure 2 lays the two side by side.

![A before and after figure contrasting a floating-point multiply-accumulate dot product against the binary version where bits are packed into words and the dot product becomes an XNOR followed by a population count](/imgs/blogs/sub-8-bit-int4-ternary-and-binary-networks-2.png)

### What it buys and what it costs

The win is enormous and comes from two directions at once. First, **memory**: binary weights are 32× smaller than fp32 and 8× smaller than int8, and because both operands are packed bits, you move 32× less data per dot product, which matters because most edge inference is memory-bound (see [the roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) for why bandwidth, not FLOPs, is usually the wall). Second, **energy and compute**: an fp32 MAC costs on the order of a few picojoules; a 32-bit integer add is roughly an order of magnitude cheaper; and an XNOR-plus-popcount over packed bits is cheaper still per element. The XNOR-Net paper reported about a **58× theoretical speedup** on convolution and a **32× memory saving** for the all-binary case, with the practical, measured speedup on CPU landing closer to 58× for the binary convolution kernel itself. Those are the headline numbers, and they are why anyone tolerates binary at all.

The cost is the accuracy. On ImageNet with an AlexNet-class backbone, the XNOR-Net paper reported roughly a **12-point top-1 drop** versus the full-precision baseline (about 44% top-1 for the fully-binary XNOR-Net versus about 56% for the full-precision reference — both approximate and architecture-dependent, so treat them as the order of magnitude). Binarizing *only* the weights and keeping activations in floating point (the "BWN" variant) was far gentler — only a couple of points — because, as the capacity argument predicts, the activations carry information the binary weights cannot afford to also throw away. That asymmetry — weights binarize more gracefully than activations — echoes loudly when we get to LLMs in section 9.

There is one more piece of machinery without which none of this trains at all, and it deserves its own section because it is the load-bearing trick of the entire sub-8-bit regime.

## 5. The straight-through estimator: how you train through a step function

Here is the problem that should stop binary and ternary networks from existing. Training is gradient descent. To update a weight you need the gradient of the loss with respect to that weight. But the binarization operation is $\text{sign}(w)$ — a step function. Its derivative is zero everywhere (the function is flat between the jumps) and undefined at the jump itself. If you backpropagate through $\text{sign}(\cdot)$ honestly, the gradient that reaches the weight is zero, the weight never updates, and training does nothing. The same is true for the rounding operation in *any* low-bit quantizer: rounding is a staircase, its derivative is zero on every step, and naive backprop gives you nothing to learn from.

The straight-through estimator (STE) is the fix, and it is gloriously pragmatic. During the forward pass you apply the real, hard quantization — $\text{sign}(w)$ for binary, or round-to-nearest-level for int4 — so the network experiences exactly the discretization it will use at inference. During the backward pass you *pretend the quantizer was the identity function* and pass the gradient straight through, as if $\frac{\partial \, q(w)}{\partial w} = 1$. The gradient computed for the quantized weight is applied to a full-precision "shadow" copy of the weight that you keep around during training; you re-quantize that shadow weight on every forward pass. So the network trains in floating point but always *sees* itself quantized, and learns weights whose quantized versions perform well.

It sounds like cheating, and in a sense it is — the backward pass is not the true gradient of the forward pass. But it works remarkably well, for a defensible reason: $\text{sign}(w)$ and the identity agree in direction over the regions that matter, so the STE gradient is a *biased but usefully correlated* estimate of the descent direction. The standard refinement is to **clip** the straight-through gradient: pass the gradient through only when the pre-quantization value is within the representable range (say $|w| \le 1$), and zero it outside. This stops weights from drifting arbitrarily far past the saturation point where moving them changes nothing in the forward pass. That clipped STE — identity inside the clip range, zero outside — is what virtually every quantization-aware-training implementation uses internally, from binary nets to int4 QAT to the `torch.ao.quantization` fake-quant modules.

The practical consequence is the QAT-needed column in Figure 1 flipping to "always" below int8. At int8 the quantization error is small enough that PTQ — quantize a pre-trained model with no further training — usually suffices. Below int8 the error is large enough that the *only* way to recover accuracy is to let the network adapt its weights to the coarse grid, which means training with fake-quant in the forward pass and STE in the backward pass. **QAT is not optional below int8; it is the price of admission.** We will write the STE in code in section 8.

## 6. Where the cliff is steepest: layer sensitivity

The cliff is not a flat plateau that everything falls off at once. Some layers tolerate aggressive quantization beautifully; others lose accuracy the moment you push them past int8. Knowing which is which is the single highest-leverage piece of practical knowledge in this whole area, because the fix — keep the fragile layers at higher precision, push everything else to the floor — recovers most of the lost accuracy for a tiny size cost. Figure 4 summarizes which layers tolerate what.

![A matrix figure listing fragile and robust layer types down the rows against whether each survives int8, int4, and binary and whether it should be kept at higher precision](/imgs/blogs/sub-8-bit-int4-ternary-and-binary-networks-4.png)

The pattern, and the reasons behind it:

**The first layer is fragile.** It consumes the raw input — pixels, audio samples, token embeddings — and that input has not yet been processed into a robust, redundant representation. Every bit of detail it discards is detail the rest of the network never gets to see. There is also very little of it: the first convolution in a typical CNN is a small fraction of the total weights and FLOPs, so keeping it at int8 or even fp16 costs you almost nothing in size while protecting the model's eyes. Standard practice across the literature — XNOR-Net, DoReFa-Net, almost every BNN paper — is to **leave the first layer at higher precision**.

**The last layer is fragile.** It produces the logits that the softmax turns into a probability distribution, and the *relative* magnitudes of those logits decide the prediction. Coarse quantization of the final classifier scrambles the close calls — the cases where the top two classes are nearly tied, which is exactly where accuracy is won or lost. It is also small. Keep it high-precision too. "Keep the first and last layers higher precision" is the oldest rule in low-bit quantization and it is right.

**Attention projections are fragile.** In transformers, the query/key/value projections and the attention score computation involve products of quantized quantities whose dynamic range is wide and whose precision drives the softmax. Quantizing attention to very low bit-widths tends to collapse the attention distribution. This is one reason LLM quantization is almost always **weight-only** — quantize the weight matrices hard, keep the activations (and thus the attention math) in fp16. More on that in section 9.

**Depthwise convolutions are fragile.** This is the one that surprises people. Depthwise convs — the efficiency trick at the heart of MobileNet — apply a single small filter per input channel, with no mixing across channels. That means each output channel depends on very few weights, so there is almost no averaging to wash out quantization error, and the per-channel weight distributions are wildly different in scale from one channel to the next. A single per-tensor scale is catastrophic for depthwise layers; even per-channel scales struggle at int4. MobileNets are notoriously hard to quantize *precisely because* depthwise convs are so sensitive. If you are quantizing a depthwise-separable architecture below int8, expect the depthwise layers to be your problem.

**The big middle layers are robust.** The 1×1 convolutions, the dense feed-forward blocks, the bulk of the transformer's weight — these are large, over-parameterized, and sit in the middle of the network where representations are redundant and noise-tolerant. They average over many inputs, they have spare capacity, and they soak up quantization error gracefully. *This is where the weights are.* In most architectures the middle layers hold the overwhelming majority of the parameters and FLOPs, so pushing *them* to the floor while sparing the small fragile endpoints gives you almost all of the size win for almost none of the accuracy cost. That is the entire strategy of mixed precision, which the series covers in depth in [mixed precision and sensitivity analysis](/blog/machine-learning/edge-ai/mixed-precision-and-sensitivity-analysis) — measure each layer's sensitivity, then spend bits where they matter.

## 7. Softening the cliff: the techniques that make sub-8-bit usable

No single trick makes a binary network as accurate as fp32. But a *stack* of techniques, applied together, can recover most of the cliff — turning a naive 12-point drop into a manageable 2-to-4-point one. Figure 5 stacks them; each layer of the stack is a defense.

![A stack figure listing the defenses that recover low-bit accuracy, including quantization-aware training, distillation, gradual bit reduction, learned scales, mixed precision, and high-precision endpoints](/imgs/blogs/sub-8-bit-int4-ternary-and-binary-networks-5.png)

**Quantization-aware training with the STE** is the foundation, non-negotiable below int8, as section 5 established. Everything else builds on top of a model that has been trained to be quantized.

**Per-channel and learned scales.** A single scale for an entire weight tensor (per-tensor quantization) is wasteful when different output channels have very different weight magnitudes — the channel with the smallest weights gets a step size sized for the channel with the largest, and its values get crushed into a handful of levels. **Per-channel** quantization gives each output channel its own scale, recovering precision for free, and it is the standard for weights. Better still, make the scale a **learned parameter** trained by gradient descent alongside the weights (the LSQ — Learned Step Size Quantization — approach). When the network can tune its own step sizes, it finds the grid that loses the least accuracy, and at int4 and below this is often worth a point or two on its own.

**Knowledge distillation into the low-bit network.** Train the quantized "student" not just on the hard labels but to match the *soft outputs* (the full logit distribution) of the full-precision "teacher." The soft targets carry far more information than a one-hot label — they tell the student which wrong answers are nearly-right — and that extra signal helps the low-capacity student claw back accuracy. Distilling an fp32 teacher into a binary or int4 student is one of the most effective single moves available; it is the [distillation lever](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) composing with the quantization lever.

**Gradual bit reduction.** Do not jump straight from fp32 to int2. Quantize to int8 first and fine-tune; then to int4 and fine-tune; then to int2. Each step is a small perturbation the network can adapt to, where the full jump would be a shock it cannot recover from. It is the same intuition as curriculum learning — ease the model down the cliff one ledge at a time instead of pushing it off the top.

**Mixed precision** ties it together: keep the fragile layers (first, last, attention, depthwise) at int8 or fp16, push the robust middle to int4 or lower, and let a sensitivity analysis decide the per-layer budget. This is the highest-leverage technique of all and gets its own [sibling post](/blog/machine-learning/edge-ai/mixed-precision-and-sensitivity-analysis); here it is enough to know it is the difference between a model that survives the descent and one that does not.

**Weight-only quantization for LLMs.** A special and hugely important case: for large language models, quantize *only the weights* (which dominate the memory footprint) and leave activations in fp16. Because LLM inference at batch=1 is memory-bandwidth-bound — the bottleneck is streaming the weights from memory, not the arithmetic — shrinking the weights to int4 gives most of the speedup *and* keeps the accuracy-sensitive activation math in full precision. This is what GPTQ and AWQ do, and it is why int4 on a 7B LLM loses under a point while int4 on a small CNN's activations is brutal. Section 8 unpacks the contrast; the [weight-only GPTQ/AWQ post](/blog/machine-learning/edge-ai/llm-quantization-weight-only-gptq-awq) goes deeper on the algorithms.

## 8. The practical flow: code you can run

Enough theory. Here is the code. Three snippets: an int4 fake-quant QAT module with per-group scales, a binary convolution with an STE, and a real `bitsandbytes` NF4 load for an LLM. They are written to be copied and adapted, not just read.

### 7.1 A 4-bit fake-quant QAT module with per-group scales

This is the heart of QAT below int8: a module that, in the forward pass, quantizes weights to a 4-bit grid with **per-group** scales (a separate scale for every group of, say, 64 weights — finer than per-channel, the standard for aggressive weight quantization), and in the backward pass passes the gradient straight through. The `torch.autograd.Function` makes the STE explicit.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class FakeQuant4bitSTE(torch.autograd.Function):
    """Forward: round weights to a signed 4-bit grid per group.
    Backward: straight-through estimator with a clip mask."""

    @staticmethod
    def forward(ctx, w, scale, qmin, qmax):
        # scale is broadcast per group; w is already reshaped to [groups, group_size]
        q = torch.clamp(torch.round(w / scale), qmin, qmax)
        w_q = q * scale
        # remember which elements were inside the representable range
        inside = (w / scale >= qmin) & (w / scale <= qmax)
        ctx.save_for_backward(inside)
        return w_q

    @staticmethod
    def backward(ctx, grad_out):
        (inside,) = ctx.saved_tensors
        # STE: identity gradient inside the clip range, zero outside.
        # No gradient flows to qmin/qmax (they are integer constants).
        return grad_out * inside, None, None, None


class QuantLinear4bit(nn.Module):
    """A Linear layer whose weights are fake-quantized to int4 with
    per-group learned scales. Activations stay full precision here;
    drop in a separate fake-quant on the input if you want int4 acts."""

    def __init__(self, in_features, out_features, group_size=64, bias=True):
        super().__init__()
        assert in_features % group_size == 0, "in_features must divide group_size"
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.n_groups = in_features // group_size

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # one LEARNED scale per (output channel, group): LSQ-style
        self.log_scale = nn.Parameter(
            torch.zeros(out_features, self.n_groups)
        )
        # signed 4-bit: 16 levels, symmetric -> [-8, 7]
        self.qmin, self.qmax = -8, 7

    def quantized_weight(self):
        w = self.weight.view(self.out_features, self.n_groups, self.group_size)
        scale = self.log_scale.exp().unsqueeze(-1)  # [out, n_groups, 1]
        w_q = FakeQuant4bitSTE.apply(w, scale, self.qmin, self.qmax)
        return w_q.view(self.out_features, self.in_features)

    def forward(self, x):
        return F.linear(x, self.quantized_weight(), self.bias)
```

The training loop around it is an ordinary fine-tuning loop — the magic is entirely inside the module. You start from pre-trained fp32 weights, copy them in, and fine-tune for a few epochs at a low learning rate. The learned scales adapt to minimize quantization-induced loss:

```python
model = build_model_with_quant_linears()   # swap nn.Linear -> QuantLinear4bit
model.load_state_dict(fp32_state_dict, strict=False)  # init from fp32
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(5):                       # QAT is short: a few epochs
    for x, y in train_loader:
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()                      # STE routes grads to fp32 shadow + scales
        opt.step()
```

In production you would not roll your own fake-quant for an int4 deployment — you would use `torch.ao.quantization` for the int8 path and a purpose-built library (GPTQ, AWQ, `bitsandbytes`) for int4 on LLMs. But writing the STE by hand once, as above, is the fastest way to actually understand what every one of those libraries is doing in its backward pass.

### 7.2 A binary convolution with the sign-STE and the XNOR-popcount idea

Here is the toy that makes section 4 concrete: a binary convolution where the forward pass binarizes both weights and activations with $\text{sign}(\cdot)$, the backward pass uses a clipped STE, and the comment block shows how the same math becomes XNOR-popcount at deploy time.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SignSTE(torch.autograd.Function):
    """sign() in the forward pass; clipped straight-through in backward.
    Gradient is identity for |x| <= 1, zero outside (the 'hardtanh' STE)."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sign(x)            # in {-1, 0, +1}; 0 is rare in practice

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        return grad_out * (x.abs() <= 1).float()


def binarize(x):
    return SignSTE.apply(x)


class BinaryConv2d(nn.Module):
    """XNOR-Net style: binarize weights and activations.
    Keeps a per-output-channel scale (alpha) to soak up magnitude,
    which is the single most important accuracy fix in XNOR-Net."""

    def __init__(self, in_ch, out_ch, k=3, padding=1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, k, k) * 0.01)
        self.padding = padding

    def forward(self, x):
        # alpha = mean(|W|) per output channel: the optimal L1 scale (XNOR-Net)
        alpha = self.weight.abs().mean(dim=(1, 2, 3), keepdim=True)
        w_bin = binarize(self.weight) * alpha
        x_bin = binarize(x)
        return F.conv2d(x_bin, w_bin, padding=self.padding)

# ---------------------------------------------------------------------------
# What the deployed kernel actually does (NOT run here, shown for intuition):
#
#   1. Pack 64 binary weights and 64 binary activations into uint64 words,
#      encoding -1 as bit 0 and +1 as bit 1.
#   2. For each output, accumulate over packed words:
#          acc += popcount( w_word XNOR x_word )
#      where popcount counts the set bits (one CPU instruction per word).
#   3. The real dot product is:  dot = 2 * acc - N    (N = number of terms)
#   4. Multiply by the per-channel scale alpha and add the bias.
#
# This replaces N float MACs with ~N/64 XNORs + ~N/64 popcounts, the source
# of the ~58x convolution speedup XNOR-Net reported.
# ---------------------------------------------------------------------------
```

Two details that are easy to miss and matter a lot. First, the per-output-channel scale `alpha = mean(|W|)` is not decoration — it is the optimal L1-norm scaling derived in the XNOR-Net paper, and it is worth several points of accuracy over plain binary weights, because it lets each channel express *how much* it matters even though the *direction* of each weight is one bit. Second, `F.conv2d` here is still doing real floating-point convolution on $\{-\alpha, +\alpha\}$ values — this is a *simulation* of binary inference for training and accuracy measurement. The XNOR-popcount kernel in the comment is what a deployment runtime (a hand-written CMSIS-NN-style kernel, or a library like Larq Compute Engine) would actually execute. The Python class lets you train and measure; the comment is the thing that makes it fast.

### 7.3 Loading a 4-bit (NF4) LLM with bitsandbytes

For language models, you almost never train your own int4 — you load weights that a weight-only quantizer (GPTQ, AWQ, or `bitsandbytes`' on-the-fly NF4) has compressed. NF4 (NormalFloat-4) is a 4-bit data type from the QLoRA work whose 16 levels are spaced to match the *normal distribution* that neural-network weights actually follow, rather than spaced uniformly. That distribution-matched spacing is why NF4 loses so little: it puts more levels where the weights actually are. Here is the full load:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat-4: levels matched to N(0,1)
    bnb_4bit_use_double_quant=True,      # quantize the scales too: ~0.4 bit/param more
    bnb_4bit_compute_dtype=torch.float16 # de-quantize to fp16 for the matmul
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

prompt = "Explain the straight-through estimator in one sentence."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

The single most important flag is `bnb_4bit_compute_dtype=torch.float16`: the weights are *stored* in 4 bits but *de-quantized to fp16 on the fly for the actual matmul*. The activations are never quantized. This is weight-only quantization in action — and it is the whole reason a 7B model that needs roughly 13 GB in fp16 fits in under 4 GB at 4-bit with a sub-1-point accuracy loss. The `use_double_quant` flag squeezes a little more by quantizing the per-group scales themselves, saving about 0.4 bits per parameter at negligible accuracy cost. Compare this to section 8.2's binary CNN, which quantized *activations* and lost double-digit points: same nominal bit-width neighborhood, completely different outcome, for reasons we will now make precise.

### 8.4 The same model as a 4-bit GGUF via llama.cpp

`bitsandbytes` quantizes on the fly at load time, which is great for experiments but pays the de-quantization cost on every forward pass. For deployment on a laptop or a phone you usually want the weights *pre-quantized to disk* in a format with a hand-tuned low-bit kernel. `llama.cpp`'s GGUF k-quants are the standard for this, and the workflow is two commands — convert the model to GGUF, then quantize to a 4-bit k-quant:

```bash
# 1. Convert the HF model to a full-precision GGUF
python convert_hf_to_gguf.py ./Llama-2-7b-hf --outfile llama2-7b-f16.gguf --outtype f16

# 2. Quantize to 4-bit: Q4_K_M is the widely-used "medium" 4-bit k-quant.
#    It uses per-block scales and keeps a few sensitive tensors at higher bits.
./llama-quantize llama2-7b-f16.gguf llama2-7b-Q4_K_M.gguf Q4_K_M

# 3. Run it; -ngl offloads layers to GPU, batch=1 decode is memory-bound
./llama-cli -m llama2-7b-Q4_K_M.gguf -ngl 99 -p "The straight-through estimator" -n 64
```

The `Q4_K_M` label decodes as: 4-bit, "K"-quant (block-wise scales with a super-block structure, the modern format), "M" for the medium mix that keeps a handful of sensitive tensors — notably the attention output and feed-forward down projections — at a higher bit-width while the bulk goes to 4 bits. That is **mixed precision baked into the file format**: the same "keep the fragile parts higher" rule from section 6, applied automatically. A Q4_K_M of a 7B model lands around 4 GB on disk and runs at interactive token rates on a laptop CPU, with perplexity within roughly 1–2% of the fp16 reference. This is the production reality of int4 LLMs on commodity hardware, and it is why `llama.cpp` runs Llama-class models on machines that could never hold the fp16 weights.

## 9. Worked examples: the two faces of low-bit

The contrast between the binary CNN and the int4 LLM is the most important practical lesson in this post, so let us put real numbers on both.

#### Worked example: a CNN down the bit-width ladder

Take ResNet-18 trained on ImageNet — about 11.7M parameters, 44.6 MB in fp32, 69.8% top-1 accuracy as the baseline. Walk it down the ladder, quantizing weights *and* activations (the realistic CNN-on-edge case, where you want integer compute end to end), with QAT applied at every step below int8:

- **int8 PTQ:** 11.2 MB (4× smaller), about 3× faster on an integer-capable CPU, **69.5%** top-1 — a 0.3-point drop, essentially free. This is the lull from section 1.
- **int4 QAT:** 5.6 MB (8× smaller), about 4× faster, **68.2%** top-1 — a 1.6-point drop *with* QAT. Without QAT, naive int4 PTQ on this model loses five to eight points; the gap between those two numbers *is* the value of QAT below int8.
- **int2 ternary:** roughly 2.9 MB (about 15× smaller), about 6× faster, **63–66%** top-1 (approximate, sensitive to recipe) — a three-to-six-point drop even with QAT, distillation, and high-precision endpoints.
- **binary XNOR:** roughly 1.5 MB (about 30× smaller), 10×+ faster, **51–58%** top-1 (approximate) — a ten-to-eighteen-point drop. The model still works, but it is now a meaningfully worse classifier.

Figure 8 is exactly this sweep as a Pareto matrix. The knee of the curve — the point past which you pay disproportionately for each further bit — is **int4 with QAT**. You get 8× the size reduction for under two points. Below that, every additional doubling of compression costs you several points, and you have to ask whether you would not be better served by a *different, smaller architecture* at int8 than the same architecture pushed to int2.

![A matrix figure showing a ResNet-18 bit-width sweep with model size, speed, and top-1 accuracy for fp32, int8, int4, ternary, and binary, marking int4 with quantization-aware training as the sweet spot](/imgs/blogs/sub-8-bit-int4-ternary-and-binary-networks-8.png)

#### Worked example: int4 weight-only on an LLM versus int4 activations on a CNN

Now the contrast that confuses everyone. Take Llama-2-7B: about 6.7B parameters, roughly 13.5 GB in fp16, and quantize its **weights only** to int4 with a group-wise method (GPTQ or AWQ, group size 128). The model shrinks to under 4 GB. The accuracy loss on standard benchmarks is **under one point** — published GPTQ and AWQ results on LLaMA-class models routinely report perplexity increases of a few hundredths and downstream-task drops well under a point. Same "int4" label as the CNN above; an order of magnitude smaller penalty. Why?

Three reasons, all of which we have already built up. **First, weight-only.** The activations — and therefore the attention math, the residual stream, the layer norms — stay in fp16. Only the weight *matrices* are coarsely quantized, and section 6 told us the big middle weight matrices are the robust part. **Second, over-parameterization.** A 7B model has enormous spare capacity (the representational-capacity argument from section 2); a coarse quantization of its weights still leaves a function close to the original. The thin MobileNet has no such slack. **Third, memory-bound inference.** LLM decoding at batch=1 spends most of its time *streaming weights from memory*, not computing, so shrinking the weights to int4 directly attacks the bottleneck and delivers near-linear speedup, while keeping the fp16 compute path that protects accuracy.

Contrast with int4 *activations* on the small CNN: the activations are on the compute critical path of *every* layer, the model has no spare capacity, the error compounds layer to layer, and the depthwise convs (if any) are catastrophic at int4. Same bit-width, opposite outcome. Figure 7 puts the two side by side.

![A before and after figure contrasting weight-only int4 on a large language model losing under a point against int4 activations on a small convolutional network losing many points](/imgs/blogs/sub-8-bit-int4-ternary-and-binary-networks-7.png)

The lesson generalizes into a rule you can carry: **int4 is safe when it touches only the grouped weights of an over-parameterized, memory-bound model, and dangerous when it touches activations on the compute path of a tight model.** That single sentence predicts the outcome of most sub-8-bit decisions you will face.

#### Worked example: fitting a keyword-spotting model in a microcontroller's flash

Now the case that justifies binary and ternary's existence: a deeply memory-constrained microcontroller. Take a Cortex-M4 class part with 256 KB of flash and 64 KB of SRAM — a common, cheap, low-power configuration for an always-on "wake word" detector. You have a small convolutional keyword-spotting network: about 120,000 weights. The question is brutal and binary: does it fit, and does it still work?

- **fp32:** 120k weights × 4 bytes = 480 KB. Does not fit in 256 KB of flash. Dead on arrival.
- **int8:** 120k × 1 byte = 120 KB of weights. Fits in flash with room for code, and the int8 activations (a few KB per layer) fit the 64 KB SRAM tensor arena. Accuracy is essentially the fp32 number. This is the right answer *if it fits the whole budget* — and here it does, so for this part you would stop at int8.
- **Now shrink the device.** Move to a Cortex-M0+ with 128 KB flash and 16 KB SRAM (cheaper, lower power, the kind of part that ships in the hundreds of millions). int8 weights are 120 KB — they barely fit flash with no room for the runtime, and the int8 activation arena may not fit 16 KB SRAM. Now sub-8-bit earns its place: **int4 weights** are 60 KB (fits comfortably, leaves flash for code), and with careful activation handling the arena drops under 16 KB. Accuracy with QAT might fall a couple of points — acceptable for a wake-word detector whose false-accepts are caught by a second-stage verifier. **Ternary** weights are roughly 30 KB and the zero-skipping cuts both compute and the activation working set further.
- **Energy, the other budget.** On a battery part, the always-on detector's power is dominated by the MAC energy of running inference every few hundred milliseconds. Ternary's zero-skipping and int4's cheaper arithmetic reduce the energy per inference, directly extending battery life — and on these parts there is no separate NPU, so the savings come straight off the CPU's active time.

The lesson: sub-8-bit is not about being clever, it is about *fitting at all*. When int8 fits the whole budget — flash, SRAM, energy — you stop at int8. When it does not, int4 and ternary are the difference between shipping and not shipping, and you accept the accuracy cost because the alternative is no product. This is the MCUNet regime, and it is the clearest legitimate use of the bottom of the bit-width axis. The series' [edge hardware landscape](/blog/machine-learning/edge-ai/the-edge-hardware-landscape) post details the flash/SRAM/energy budgets of these parts.

### A technique-by-technique cheat sheet

Pulling the softening techniques together into one decision table, so you can see at a glance what each costs and when it pays:

| Technique | Recovers | Cost | Use when |
| --- | --- | --- | --- |
| QAT + STE | the most (several points) | a few epochs of training | always, below int8 |
| Per-channel scales | 1–3 points | tiny scale table | always for weights |
| Per-group scales | 1–2 more points | small scale table | int4 weights, LLMs |
| Learned step size (LSQ) | 1–2 points | extra trained params | int4/int2 CNNs |
| Distillation from fp32 | 1–4 points | a teacher + longer train | low-bit students |
| Gradual bit reduction | 1–3 points | multi-stage training | int2/binary targets |
| Keep first/last high | 2–5 points | negligible size | every low-bit model |
| Mixed precision | most of the cliff | sensitivity analysis | whenever HW supports it |

The right-most column is the point: none of these is free, and you spend them in roughly this order — QAT and high-precision endpoints first because they are cheap and recover the most, distillation and mixed precision when you still need more, gradual reduction when you are pushing to the extreme bottom.

## 10. Ternary versus binary: the value of a zero

It is worth pausing on the specific gap between ternary and binary, because it is a clean illustration of how a tiny representational addition buys a lot of accuracy. Figure 6 contrasts them.

![A before and after figure contrasting binary weights restricted to minus one and plus one against ternary weights that add a zero state and a learned scale to recover accuracy](/imgs/blogs/sub-8-bit-int4-ternary-and-binary-networks-6.png)

Binary forces every weight to $\pm 1$ (times a scale). Ternary adds one more state: zero. That single addition does two things. First, it matches the *actual distribution* of trained weights far better — weight histograms are sharply peaked around zero, so a representation that can say "this weight is essentially zero" loses much less information than one forced to round every near-zero weight to $\pm 1$. Second, the zeros are *free to skip*: a ternary network is naturally sparse, and the zero weights contribute nothing to the dot product and need not be computed or even stored densely. The cost is modest — about 2 bits per weight instead of 1, so 16× compression instead of 32×. The benefit is real: ternary nets typically recover several points of accuracy over binary on the same architecture. On the ImageNet-class sweep, the gap between binary (51–58% in the earlier example) and ternary (63–66%) is roughly that recovery. When binary loses too much and int4 is more precision than you can afford, ternary is the often-overlooked middle ground — and the zero state is the entire reason it exists.

This is a recurring shape in low-bit work: a small, well-chosen relaxation of the representation buys back a disproportionate amount of accuracy. NF4's distribution-matched levels are the same idea at int4 (match the levels to the data instead of spacing them uniformly); per-channel scales are the same idea applied to dynamic range; the learned $\alpha$ in XNOR-Net is the same idea applied to magnitude. The general principle: **spend your scarce representational budget where the data actually is.**

## 11. Case studies: real numbers from the literature

Concrete, cited results so you can calibrate your expectations. Treat the exact figures as approximate where noted — recipes and reference baselines vary across papers.

**XNOR-Net (Rastegari et al., 2016)** binarized both weights and activations of an AlexNet-class network on ImageNet. Reported roughly **44% top-1** for fully-binary XNOR-Net against about **56%** for the full-precision reference — a ~12-point drop — alongside a claimed **~58× convolution speedup** and **~32× memory saving**. The weight-only binary variant (BWN), keeping activations in floating point, came within a couple of points of full precision, the asymmetry that motivates weight-only quantization throughout this post.

**Ternary Weight Networks (Li et al., 2016)** constrained weights to $\{-1, 0, +1\}$ with a learned scale and a threshold $\Delta_t \approx 0.7\,\mathbb{E}[|w|]$. On ImageNet with a ResNet-18-class backbone, TWN landed within a few points of the full-precision baseline — substantially better than binary at the cost of one extra bit per weight, exactly the trade-off section 10 described.

**DoReFa-Net (Zhou et al., 2016)** generalized low-bit training to arbitrary bit-widths for weights, activations, *and gradients*, showing that you could train with low-bit gradients too, and mapping out the accuracy across the whole bit-width grid. It is the paper to read for the systematic view of how accuracy degrades as you independently lower each of the three.

**GPTQ (Frantar et al., 2022) and AWQ (Lin et al., 2023)** are the modern weight-only int4 LLM quantizers. On LLaMA/Llama-2-class models, both report **int4 weight-only with under a point** of downstream accuracy loss and perplexity increases in the low hundredths, using group-wise scales (group size 128) and, in GPTQ's case, an error-compensating quantization order derived from second-order (Hessian) information. These are the production reality of sub-8-bit on LLMs and the reason 7B-class models run on consumer hardware. The series covers them in detail in [LLM quantization with weight-only GPTQ and AWQ](/blog/machine-learning/edge-ai/llm-quantization-weight-only-gptq-awq).

**BitNet b1.58 (2024)** is the recent counterpoint worth knowing: a transformer *trained from scratch* with ternary $\{-1, 0, +1\}$ weights that matched full-precision baselines at billion-parameter scale. The lesson is that the cliff is partly an artifact of *quantizing after the fact* — a model designed and trained ternary from the start can occupy a very different point on the curve than one pushed there post hoc. Architecture-native low-bit is a genuinely different game from compressing a pre-trained model, and it is where a lot of the frontier is heading.

**MCUNet (Lin et al., 2020) on a Cortex-M7** is the case study for the microcontroller end of the spectrum. By co-designing a tiny architecture and an int-quantized inference engine (TinyEngine) together, MCUNet ran ImageNet-class classification on a microcontroller with under 1 MB of flash and a few hundred KB of SRAM — a regime where fp32 is laughably out of reach and even int8 is tight. The headline is not a single bit-width but the *co-design*: low-bit quantization plus an architecture sized to the memory budget plus a no-malloc runtime, together, are what made on-MCU inference possible at all. It is the worked example from section 9 turned into a shipped result, and the clearest demonstration that sub-8-bit's home is the extreme-memory regime.

**QLoRA (Dettmers et al., 2023)** is worth a mention beyond just its NF4 data type: it showed you can *fine-tune* a model whose base weights are frozen at 4-bit NF4, with the trainable low-rank adapters in higher precision. That is a different use of sub-8-bit — not just inference but training-on-top-of-quantized-weights — and it is why a 65B model can be fine-tuned on a single GPU. It reinforces the weight-only lesson: the 4-bit base is accurate enough to fine-tune against because the quantization touched only the over-parameterized weights, never the gradients flowing through the adapters.

## 12. How you would actually measure this honestly

A table of accuracy and speed numbers is worthless if it was measured carelessly, and sub-8-bit is especially easy to measure dishonestly because the wins look so good on paper. A few rules, consistent with [the metrics that actually matter on device](/blog/machine-learning/edge-ai/the-metrics-that-actually-matter-on-device):

**Speed must be measured on hardware that actually has the low-bit kernel.** A 58× binary speedup is theoretical; you only realize it if your runtime has an XNOR-popcount convolution kernel for your target. On a chip without one, the binary model falls back to unpacking the bits and running ordinary arithmetic — and can be *slower* than the int8 model it replaced. The same trap bites int4: many NPUs and CPUs have no int4 matmul, so an int4 model gets de-quantized to int8 or fp16 at runtime and you get the *memory* win but none of the *compute* win. Check that the kernel exists before you promise the speedup.

**Accuracy must be measured on the full eval set, after QAT has converged.** It is tempting to report the accuracy from the first few QAT epochs or a tiny validation subset. Below int8 the accuracy keeps recovering for several epochs of QAT, and the cliff numbers are sensitive to the recipe — report the converged number on the real eval set, and report the *delta from the fp32 baseline you actually have*, not from a paper's baseline.

**Measure latency at batch=1, warmed up, with thermal throttling in mind.** Edge inference is almost always batch=1, where the memory-bound regime dominates and the FLOP savings of low-bit weights matter less than the bandwidth savings. Warm up the kernels before timing (the first inference pays JIT/cache costs), report p50 and p99 (the tail is what users feel), and run long enough that thermal throttling on a passively-cooled device shows up — a phone NPU that is fast for the first ten inferences and then throttles is a real and common failure.

**Watch for the fallback cliff.** The single most common production surprise: one op in your quantized graph is not supported at the target bit-width, the runtime silently falls back to fp32 for that op, and your beautiful low-bit graph spends most of its time in a fp32 island with expensive quantize/de-quantize conversions on either side. Profile the actual graph and confirm every op ran at the bit-width you think it did.

## 13. When sub-8-bit is worth it, and when it absolutely is not

This is the decision section. Be ruthless: below int8 is a cost, and most models should never go there.

**Reach for sub-8-bit when:**

- **You are memory- or flash-constrained past what int8 gives you.** A microcontroller with 512 KB of flash and 256 KB of SRAM cannot hold an int8 model that an int4 or ternary version fits comfortably. When the model literally does not fit otherwise, the accuracy cost is the price of running at all. MCUNet-style work lives here.
- **You are quantizing LLM weights.** Weight-only int4 (GPTQ/AWQ/NF4) on a large, over-parameterized, memory-bound model is the clearest win in the whole field: huge memory savings, real batch=1 speedup, sub-1-point accuracy loss. If you are putting a 7B+ model on consumer hardware, int4 weights are the default, not an exotic choice.
- **Energy is the binding constraint and you have the hardware.** An always-on, battery-powered detector where MAC energy dominates the power budget, running on a chip that *actually has* XNOR-popcount or int4 kernels, can justify binary or ternary for the order-of-magnitude energy reduction. The "actually has the kernel" clause is load-bearing.
- **You can design the model low-bit from scratch.** Training native ternary (BitNet-style) or designing an architecture for binary inference reaches points on the curve that post-hoc quantization cannot. If you control the architecture and the training budget, this beats compressing a pre-trained dense model.

**Do not reach for sub-8-bit when:**

- **int8 already hits your target.** This is the most important rule in the post. If int8 PTQ fits your size, latency, and accuracy budget — and for a huge fraction of edge CNNs and small transformers it does — going below it is gratuitous risk. Every bit below 8 costs accuracy and engineering effort; do not spend them for a win you do not need.
- **The model is small and accuracy-critical.** A thin MobileNet sized to the bone for medical imaging or safety-critical perception has no spare capacity to give up, and its activations are on every layer's compute path. Pushing it to int4 activations or binary is exactly the brutal case from section 9. Use a *different, slightly larger* architecture at int8 before you push a tight one to int2.
- **Your target hardware has no low-bit kernel.** If the NPU/CPU/runtime has no int4 or XNOR kernel, you get the storage win but no speed win, the model de-quantizes at runtime, and you have paid accuracy for nothing on the latency axis. Confirm the kernel exists first.
- **You have not yet exhausted the free levers.** A good compiler/runtime, operator fusion, and an honest profiling pass cost zero accuracy. Pruning and distillation cost less accuracy per unit of size than sub-8-bit quantization does. Walk down the [taxonomy](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) in order before you reach for the bottom of the bit-width axis.

The decision, distilled: sub-8-bit is for the *extremes* — the smallest devices, the largest models, the tightest energy budgets — and for those it is indispensable. For the comfortable middle, int8 is the answer and below-8 is a trap.

## Key takeaways

- **The cliff is real and it steepens.** Size improves linearly as you drop bits; accuracy degrades super-linearly. int8→int4 might cost a point or two; int4→int2→binary each cost several more.
- **6 dB per bit.** $\text{SQNR} \approx 6.02\,b + 1.76$ dB. Every bit removed quadruples the quantization-noise power and removes ~6 dB of headroom. The cliff is where you run out of the cushion the network was trained to tolerate.
- **QAT is mandatory below int8.** PTQ is fine at int8; below it, you must train the network to be quantized, using fake-quant in the forward pass and a clipped straight-through estimator in the backward pass.
- **The XNOR-popcount trick.** Binarizing both operands turns the dot product into $2\cdot\text{popcount}(\text{XNOR}(W,X)) - N$, replacing $N$ MACs with $\sim N/64$ XNORs and popcounts — the source of binary's ~58× theoretical convolution speedup and 32× memory saving.
- **A handful of layers carry the loss.** First, last, attention, and depthwise layers are fragile; keep them high-precision and push the robust over-parameterized middle to the floor. Mixed precision recovers most of the cliff for a tiny size cost.
- **Weight-only ≫ activation quantization at low bits.** int4 weight-only on an over-parameterized, memory-bound LLM loses under a point; int4 activations on a tight CNN's compute path loses many. Same bit-width, opposite outcome.
- **The zero state earns its bit.** Ternary's $\{-1, 0, +1\}$ recovers several points over binary's $\{-1, +1\}$ for one extra bit, because the zero matches the weight distribution and is free to skip.
- **int4 with QAT is usually the knee.** For CNNs it is the defensible sweet spot below int8; for LLMs, int4 weight-only is the production default. Below int4, ask whether a different architecture at int8 would serve you better.
- **Measure on real kernels.** A theoretical speedup you cannot realize because the target lacks the int4 or XNOR kernel is worth zero. Confirm the kernel, warm up, measure batch=1 p50/p99, and watch for silent fp32 fallbacks.

## Further reading

- **Courbariaux, Hubara, et al. (2016), "Binarized Neural Networks"** — the foundational BNN paper that established sign-binarization and the straight-through estimator for training.
- **Rastegari, Ordonez, Redmon, Farhadi (2016), "XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks"** — derives the XNOR-popcount kernel and the per-channel L1 scale; the source of the ~58× / 32× headline numbers.
- **Li, Zhang, Liu (2016), "Ternary Weight Networks"** — the $\{-1, 0, +1\}$ formulation with the optimal threshold and scale derivation.
- **Zhou, Wu, et al. (2016), "DoReFa-Net"** — low-bit training of weights, activations, and gradients across the full bit-width grid.
- **Frantar, Ashkboos, et al. (2022), "GPTQ"** and **Lin, Tang, et al. (2023), "AWQ"** — the modern weight-only int4 LLM quantizers behind sub-1-point 4-bit LLMs.
- **Dettmers, Pagnoni, et al. (2023), "QLoRA"** — introduces the NF4 data type used in the `bitsandbytes` 4-bit load above.
- Within this series: [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) for the four-lever Pareto frame; [quantization-aware training](/blog/machine-learning/edge-ai/quantization-aware-training-qat) for the QAT mechanics this post leans on; [mixed precision and sensitivity analysis](/blog/machine-learning/edge-ai/mixed-precision-and-sensitivity-analysis) for choosing per-layer bit-widths; [LLM quantization with weight-only GPTQ and AWQ](/blog/machine-learning/edge-ai/llm-quantization-weight-only-gptq-awq) for the int4-LLM algorithms; and [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) for putting it all together on a real deployment.

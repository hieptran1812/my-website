---
title: "Numerical Formats and Mixed Precision: fp32, tf32, bf16, fp16, fp8"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Learn exactly why bf16 trains stably, fp16 needs loss scaling, tf32 is near-free, and fp8 is the new frontier, so you can flip on AMP knowing what every bit is doing."
tags:
  [
    "high-performance-computing",
    "gpu",
    "mixed-precision",
    "bf16",
    "fp8",
    "tensor-cores",
    "deep-learning",
    "ml-systems",
    "pytorch",
    "amp",
  ]
category: "machine-learning"
subcategory: "High Performance Computing"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/numerical-formats-and-mixed-precision-fp32-tf32-bf16-fp16-fp8-1.png"
---

You flip on `torch.autocast`, your step time drops by 2.7x, your loss curve looks identical, and you move on. That is the happy path, and most of the time it just works. But the reason it works is not magic, and the reason it sometimes does not work — the run that NaNs at step 4,000, the fine-tune that silently loses two points of accuracy, the conversion to fp16 that turns a healthy gradient into a hard zero — is buried in how a handful of bits get split between *range* and *precision*. If you have ever stared at a `nan` in your loss and reflexively added `GradScaler` without knowing why it helps, this post is for you.

Here is the one-sentence version of everything below: a floating-point number is a sign bit, some **exponent** bits that set its *dynamic range* (how big and how small a value it can represent, like scientific notation's power of ten), and some **mantissa** bits that set its *precision* (how many significant figures it keeps). At a fixed 16 bits you can spend them on range or on precision, but not both. **bf16** spends them on range and trains like fp32 with no babysitting. **fp16** spends them on precision and underflows your gradients unless you rescale the loss. That single trade-off explains the entire modern training stack, and once you see it, every choice — tf32 for free matmul speed, fp8 on Hopper, which ops AMP leaves in fp32 — falls out of it.

![matrix comparing the bit layouts of fp32, tf32, bf16, fp16, and the two fp8 formats showing how each splits its bits into sign, exponent, and mantissa](/imgs/blogs/numerical-formats-and-mixed-precision-fp32-tf32-bf16-fp16-fp8-1.png)

The figure above lays out all five formats (six, counting fp8's two variants) side by side. Notice the pattern: the green "wide range" cells all have an 8-bit exponent — fp32, tf32, and bf16 share the same dynamic range. The danger-colored cells, fp16 and fp8, are the ones that pinch the exponent. By the end of this post you will be able to read that table and predict, before you run anything, whether a given format will train stably, whether it needs loss scaling, and roughly what speedup it buys on which hardware. This is the precision wall in the three-walls frame from [why HPC is the bottleneck for modern AI](/blog/machine-learning/high-performance-computing/why-hpc-is-the-bottleneck-for-modern-ai): every byte you do not move is a byte of HBM bandwidth you get back, and every bit you drop from the matmul is throughput you gain on the Tensor Cores — as long as the numerics survive.

We will build it up in layers. First the anatomy of a float and the exact value formula. Then range versus precision made quantitative, with the actual largest and smallest numbers each format can hold. Then why fp16 underflows a `1e-7` gradient that bf16 keeps, and the loss-scaling trick that rescues fp16. Then tf32 as the near-free matmul accelerator, fp8 on Hopper, and finally the whole thing wired together as PyTorch AMP — `autocast`, `GradScaler`, the fp32 master weights — with a Transformer trained in bf16 as our running example and real before-and-after numbers on named A100 and H100 silicon.

## What a floating-point number actually is

Before any format, the value formula. A normal floating-point number is

$$\text{value} = (-1)^s \cdot (1 + m) \cdot 2^{\,e - \text{bias}}$$

where $s$ is the sign bit (0 for positive, 1 for negative), $m$ is the fractional mantissa read as a binary fraction in $[0, 1)$, and $e$ is the stored exponent, an unsigned integer that gets shifted down by a fixed **bias** so the format can represent both large and small magnitudes. The leading `1 +` is the famous "hidden bit": for normal numbers the leading significand digit is always 1, so the format does not waste a bit storing it.

A concrete number makes this real. Take fp32, which has 8 exponent bits (bias 127) and 23 mantissa bits. Suppose $s = 0$, the stored exponent is $e = 128$, and the mantissa bits encode $m = 0.5$. Then

$$\text{value} = (-1)^0 \cdot (1 + 0.5) \cdot 2^{128 - 127} = 1.5 \cdot 2 = 3.0.$$

So those 32 bits hold exactly `3.0`. Change one mantissa bit and you get the *next* representable number above or below 3.0; the gap between consecutive representable numbers is the **ULP** (unit in the last place), and it is *not* constant — it scales with the magnitude of the number. Near 3.0 the fp32 ULP is about $2^{-22} \cdot 2 \approx 4.8 \times 10^{-7}$; near $3 \times 10^6$ it is a thousand times larger. This is the single most important fact about floating point: precision is *relative*, not absolute. You get roughly the same number of significant figures everywhere, and the absolute spacing grows with the value. That is exactly what you want for neural-net weights, which span many orders of magnitude.

The exponent bits set the *range*: the largest and smallest powers of two you can reach. The mantissa bits set the *precision*: how finely you can subdivide between consecutive powers of two. The mantissa is the resolution; the exponent is the reach. Hold that distinction — every format below is just a different split of a fixed bit budget between reach and resolution.

There is one more wrinkle that matters enormously for training: **subnormals** (also called denormals). When the stored exponent hits its minimum and the value would otherwise be too small for the normal formula, the hidden leading 1 is dropped and the format represents

$$\text{value} = (-1)^s \cdot (0 + m) \cdot 2^{\,1 - \text{bias}},$$

a gradual slide toward zero instead of a sudden cliff. Subnormals buy you a little extra reach at the bottom at the cost of precision, and they are where the smallest representable nonzero numbers live. For fp16, subnormals are the difference between a `1e-7` gradient becoming a tiny-but-nonzero number versus becoming a hard zero. We will return to this; it is the crux of why fp16 needs help.

#### Worked example: reading a bf16 value by hand

bf16 has 1 sign bit, 8 exponent bits (bias 127, same as fp32), and 7 mantissa bits. Suppose the 16 bits decode to $s = 0$, $e = 124$, $m = 0.75$. Then $\text{value} = (1 + 0.75) \cdot 2^{124 - 127} = 1.75 \cdot 2^{-3} = 1.75 / 8 = 0.21875$. The ULP here is $2^{-7} \cdot 2^{-3} = 2^{-10} \approx 9.8 \times 10^{-4}$ — about three decimal digits of resolution. Now decode the *same exponent and mantissa* in fp16 (5 exponent bits, bias 15, 10 mantissa bits). The exponent field cannot even hold 124; fp16's max stored exponent is 30 (the value 31 is reserved for infinity and NaN). So a number bf16 represents trivially is simply unreachable by fp16's exponent — that gap, not the mantissa, is the whole story of training stability.

### The bit fields, laid out byte by byte

It helps to see exactly which bits go where, because the field widths are the entire design. In fp32, the 32 bits run sign (1) · exponent (8) · mantissa (23). bf16 is, almost literally, fp32 with the bottom 16 mantissa bits chopped off: sign (1) · exponent (8) · mantissa (7). That is not an accident of naming — it is why converting fp32 to bf16 is nearly free in hardware (truncate or round-to-nearest the low 16 bits) and why bf16 inherits fp32's exact exponent range. fp16 is a different animal entirely: sign (1) · exponent (5) · mantissa (10). It was designed by the graphics world (it is IEEE 754 half precision, born for textures and colors where values live in a bounded $[0, 1]$-ish range) long before anyone trained a neural network in it, and its 5-bit exponent reflects that origin — graphics never needed 76 decades of range. bf16, by contrast, was designed *by* the deep-learning world (Google Brain, for TPUs) specifically to keep fp32's range, because the people who built it had already been burned by fp16's narrowness.

The fp8 formats extend the same chopping logic one rung further. E4M3 is sign (1) · exponent (4) · mantissa (3); E5M2 is sign (1) · exponent (5) · mantissa (2). At 8 bits there is so little to spend that E4M3 even bends the IEEE rules slightly — it reclaims the would-be infinity encodings to represent a couple more finite values, which is why its max is 448 rather than a rounder power of two. You do not need to memorize these layouts, but seeing that every format is just a different cut point between the exponent field and the mantissa field demystifies the whole zoo. There is no deep difference between them; there is only where you put the comma between "how much reach" and "how fine the steps."

### Why relative precision is exactly what neural nets want

Pause on the relative-precision point, because it is the reason low precision works at all for deep learning and it is genuinely non-obvious. A floating-point format gives you roughly constant *relative* error everywhere: whether your number is near $10^{-5}$ or near $10^{5}$, the gap to the next representable value is about the same *fraction* of the number. For bf16 that fraction is about $2^{-8} \approx 0.4\%$; for fp16 about $2^{-11} \approx 0.05\%$; for fp32 about $2^{-24} \approx 6 \times 10^{-6}\%$. Now ask what neural-network values look like: weights, activations, and gradients each span several orders of magnitude, and what matters for learning is almost always the *ratio* of values (a weight twice as large as another, a gradient direction), not their absolute spacing. A format that holds constant relative error across a wide range is precisely the tool for quantities whose meaning is multiplicative. A fixed-point format, by contrast, gives constant *absolute* error — fine near its design magnitude, useless three orders of magnitude away — which is exactly why integer/fixed-point training never caught on and why we use floats at all. The whole game of mixed precision is to ask, format by format, "is 0.4% relative error tolerable for *this* tensor?" and the answer is yes far more often than intuition suggests, because gradient descent is a noisy, averaging process that washes out per-element rounding the same way it washes out minibatch sampling noise.

## Range versus precision, made quantitative

Let us put real numbers on "range" and "precision" so the trade-off stops being vague. The cleanest way to feel a format is to ask `torch.finfo` for its limits.

```python
import torch

for dtype in (torch.float32, torch.float16, torch.bfloat16):
    fi = torch.finfo(dtype)
    print(f"{str(dtype):16s} "
          f"max={fi.max:.3e}  "
          f"tiny(min normal)={fi.tiny:.3e}  "
          f"eps={fi.eps:.3e}  "
          f"mantissa bits={fi.bits - fi.eps.bit_length() if False else ''}")
```

Run that and the headline numbers come out roughly like this (I am rounding; the exact values are spec-defined):

| Format | Exp / Mant bits | Largest normal | Smallest normal | eps (ULP at 1.0) | Decimal digits |
| --- | --- | --- | --- | --- | --- |
| fp32 | 8 / 23 | $\approx 3.4 \times 10^{38}$ | $\approx 1.18 \times 10^{-38}$ | $\approx 1.19 \times 10^{-7}$ | ~7 |
| tf32 | 8 / 10 | $\approx 3.4 \times 10^{38}$ | $\approx 1.18 \times 10^{-38}$ | $\approx 9.77 \times 10^{-4}$ | ~3 |
| bf16 | 8 / 7 | $\approx 3.4 \times 10^{38}$ | $\approx 1.18 \times 10^{-38}$ | $\approx 7.81 \times 10^{-3}$ | ~2–3 |
| fp16 | 5 / 10 | $\approx 65504$ | $\approx 6.10 \times 10^{-5}$ | $\approx 9.77 \times 10^{-4}$ | ~3–4 |
| fp8 E4M3 | 4 / 3 | $\approx 448$ | $\approx 1.95 \times 10^{-3}$ | $\approx 0.125$ | ~1 |
| fp8 E5M2 | 5 / 2 | $\approx 57344$ | $\approx 6.10 \times 10^{-5}$ | $\approx 0.25$ | ~1 |

Read the two halves of that table separately and the design philosophy of each format jumps out.

Look at the **largest/smallest normal** columns — that is range. fp32, tf32, and bf16 all reach $\pm 3.4 \times 10^{38}$ at the top and about $1.2 \times 10^{-38}$ at the bottom. They share an 8-bit exponent, so they share fp32's enormous dynamic range — roughly 76 orders of magnitude. fp16, with only 5 exponent bits, tops out at **65,504** and bottoms out (in normals) at about $6.1 \times 10^{-5}$. That is a range of about 9 orders of magnitude, and it is the single fact that makes fp16 fragile for training. fp8 E4M3 maxes at a laughably small **448**.

Now look at **eps** — that is precision at the value 1.0, i.e. the ULP there. fp16 and tf32 both keep 10 mantissa bits, so their resolution near 1.0 is finest among the 16-bit-and-below formats. bf16, with only 7 mantissa bits, is coarser: a relative resolution of about 0.8%. This is bf16's genuine cost, and we will see exactly where it bites.

![before and after diagram contrasting fp16 spending bits on precision and overflowing against bf16 spending bits on dynamic range so a tiny gradient survives](/imgs/blogs/numerical-formats-and-mixed-precision-fp32-tf32-bf16-fp16-fp8-2.png)

That figure crystallizes the swap. At an identical 16-bit budget, fp16 keeps 10 mantissa bits and a 5-bit exponent — fine resolution, narrow reach, and a `1e-7` gradient falls off the bottom. bf16 keeps an 8-bit exponent and only 7 mantissa bits — the same fp32 reach, so a `1e-7` gradient survives, but each number carries about one fewer significant decimal digit. The entire mixed-precision playbook is a series of answers to the question "where can I afford to lose that decimal digit, and where must I keep the range?"

#### Worked example: how many orders of magnitude does each exponent buy?

An exponent field of $k$ bits, after reserving the all-ones pattern for infinity/NaN and the all-zeros pattern for subnormals, spans about $2^k - 2$ usable exponent values. fp16's 5-bit exponent gives roughly 30 usable powers of two, and $2^{30} \approx 10^9$ — about 9 decades from smallest normal to largest, which matches the table ($6.1\times10^{-5}$ to $65504$ is indeed roughly 9 decades). bf16's 8-bit exponent gives about 254 usable powers, and $2^{254} \approx 10^{76}$ — about 76 decades, the full fp32 span. So bf16 does not buy *more* total numbers than fp16 (both are 16 bits, both represent exactly $2^{16}$ patterns); it *spreads* the same count of numbers across a vastly wider range, accepting coarser steps in exchange for never running out of reach. That redistribution is the whole trick.

### How much precision do you actually lose? SQNR and the mantissa

"Coarser mantissa" sounds scary until you quantify it, so let us put a number on the precision cost using **SQNR** (signal-to-quantization-noise ratio), the standard way to measure how much rounding error a format introduces relative to the signal. When you round a value to a format with $b$ mantissa bits, the rounding error is at most half a ULP, and averaged over many values the noise power works out to a clean rule of thumb: every mantissa bit buys about **6 dB** of SQNR. The famous formula is $\text{SQNR} \approx 6.02\,b + 1.76$ decibels for $b$ mantissa bits.

Plug the formats in. fp32's 23 mantissa bits give about $6.02 \times 23 + 1.76 \approx 140$ dB — effectively perfect for our purposes. fp16 and tf32, at 10 mantissa bits, give about $62$ dB. bf16, at 7 mantissa bits, gives about $44$ dB. fp8 E4M3, at 3 mantissa bits, gives about $20$ dB; E5M2 at 2 bits about $14$ dB. Now the key question: how much SQNR does training actually *need*? Empirically, training tolerates surprisingly low SQNR because gradient descent averages over a minibatch and over many steps — the per-element rounding noise behaves like an extra source of stochasticity, similar to the noise you already accept from minibatch sampling, and it largely cancels in the accumulated update. bf16's 44 dB turns out to be plenty for stable convergence (with fp32 master weights mopping up the swamping problem), which is why it trains to the same accuracy as fp32. fp8's ~20 dB is at the edge, which is exactly why fp8 training needs per-tensor scaling and a precision-sensitive shortlist rather than a blanket cast.

The reason the *mantissa* loss is forgivable but the *exponent* loss is not comes straight from this. Losing mantissa bits adds bounded relative noise — annoying, averaged away, survivable. Losing exponent bits adds *catastrophic* error: a value that falls off the bottom of the range does not get noisy, it becomes exactly zero, an infinite relative error on that element. Underflow is not a small perturbation; it is total destruction of information. That asymmetry — graceful degradation from too few mantissa bits versus a cliff from too few exponent bits — is the deepest reason bf16's "spend bits on range" beats fp16's "spend bits on precision" for training. You can tolerate a noisy gradient; you cannot tolerate a deleted one.

## Why fp16 underflows where bf16 survives

Now we make the headline claim provable: a small gradient that is perfectly healthy in bf16 becomes a hard zero in fp16, and that is what destroys an fp16 run that has no loss scaling.

During backprop, gradients of the loss with respect to deep-layer weights are routinely tiny. In a Transformer with pre-LN and a reasonable learning rate, individual gradient elements often sit in the $10^{-4}$ to $10^{-8}$ range, and after multiplying through a long chain of small factors, plenty of them land around $10^{-7}$ or below. Ask whether $10^{-7}$ is representable.

In **bf16**, the smallest normal is about $1.18 \times 10^{-38}$ and subnormals reach far below that, down to about $9 \times 10^{-41}$. A value of $10^{-7}$ is nowhere near the bottom — it is a comfortable, fully-normal number with several powers of two to spare. It survives the cast exactly the way it would in fp32 (give or take the coarser mantissa). The gradient flows, the optimizer sees it, the weight updates.

In **fp16**, the smallest *normal* number is about $6.1 \times 10^{-5}$. A gradient of $10^{-7}$ is already two orders of magnitude *below* the smallest normal fp16 value. It can only be represented as a subnormal, and fp16 subnormals run down to about $6 \times 10^{-8}$ — so $10^{-7}$ squeaks in as a coarse subnormal, but anything below roughly $6 \times 10^{-8}$ rounds to **exactly zero**. Underflow. The gradient is silently destroyed before the optimizer ever sees it. Multiply a whole tensor of such gradients by zero and that layer simply stops learning. No error, no warning — just a model that trains worse than it should, or a loss curve that stalls.

That is **underflow**: a nonzero value too small for the format rounds to zero. Its mirror is **overflow**: fp16 tops out at 65,504, so a large activation or a loss spike past that becomes `inf`, and the first `inf` that touches a multiply or subtract produces `nan`, which then poisons every weight it reaches. fp16 is squeezed from both ends — too small underflows, too big overflows — because its 9-decade window is just too narrow for the spread of values in a real training step. bf16's 76-decade window has so much headroom that neither end is a practical concern.

This is why, empirically, you can take a network that trains in fp32, switch the compute to bf16, change nothing else, and it just trains. And it is why the same switch to fp16 will, on many networks, NaN or stall within a few thousand steps unless you add loss scaling. The 2018 Micikevicius et al. paper "Mixed Precision Training" — the work that made fp16 training practical at all — is fundamentally a paper about getting around fp16's narrow exponent, and its central tool is the next section's subject.

### The gradient histogram tells the whole story

The most useful diagnostic for any precision question is a histogram of the absolute values of your gradients, plotted on a log scale. Micikevicius et al. made exactly this plot for an SSD detection network and it became the canonical image of the problem: a big mass of gradient magnitudes sitting around $10^{-3}$ to $10^{-1}$, a long tail stretching down toward $10^{-7}$ and below — and the fp16 representable floor (about $2^{-24} \approx 6 \times 10^{-8}$) cutting straight through that tail. Everything to the left of the floor line is gradient signal that fp16 turns into zero. In their measurement, a substantial fraction of gradient values — sometimes more than half by count, depending on the layer — fell into the underflow region, and zeroing them measurably hurt accuracy.

Two things about that histogram explain the fix completely. First, the *shape* of the distribution is roughly fixed; what loss scaling does is slide the whole histogram to the right by $\log_2 S$ powers of two so the underflowing tail lands back inside the representable window. Second, the *top* of the histogram has a lot of headroom below fp16's overflow ceiling of 65,504 — the largest gradients are nowhere near it — so you can shift the whole thing up by a big factor before you risk overflow at the top. That headroom is the budget loss scaling spends. The optimal scale is "as large as possible without overflowing the largest gradient," which is exactly what dynamic loss scaling discovers automatically by pushing up until it overflows and backing off. If you ever want to *see* why your fp16 run is stalling, log `grad.abs()` percentiles and watch where they sit relative to $6 \times 10^{-8}$; the fix will be obvious. And if you switch to bf16, the floor moves to $10^{-40}$ and the entire tail clears it — no histogram-shifting required.

#### Worked example: where exactly is the fp16 underflow cliff?

Let us pin the cliff precisely. fp16's exponent bias is 15, the minimum stored exponent for normals is 1, so the smallest normal is $2^{1-15} = 2^{-14} \approx 6.10 \times 10^{-5}$. Subnormals push lower: the smallest positive subnormal is $2^{-10} \cdot 2^{-14} = 2^{-24} \approx 5.96 \times 10^{-8}$. So the representable nonzero floor is about $6 \times 10^{-8}$. A gradient of $5 \times 10^{-8}$ rounds to zero; a gradient of $7 \times 10^{-8}$ becomes a very coarse subnormal with essentially one bit of precision. Now contrast bf16: bias 127, smallest normal $2^{-126} \approx 1.18 \times 10^{-38}$, smallest subnormal $2^{-7} \cdot 2^{-126} = 2^{-133} \approx 9.2 \times 10^{-41}$. The bf16 floor is about $10^{-40}$ — thirty-three orders of magnitude below fp16's floor. That gap is the difference between a gradient that vanishes and one that flows.

## Loss scaling: rescuing fp16 by shifting the histogram

If fp16's problem is that gradients fall off the bottom of its range, the fix is to move them up before they fall off, and move them back down afterward. That is **loss scaling**, and it is the heart of fp16 mixed-precision training.

The mechanism is arithmetic. The gradient of the loss is linear in the loss by the chain rule: if you multiply the loss $L$ by a constant scale factor $S$ before calling backward, every gradient in the graph comes out multiplied by the same $S$:

$$\frac{\partial (S \cdot L)}{\partial w} = S \cdot \frac{\partial L}{\partial w}.$$

Pick $S = 65536 = 2^{16}$. A gradient that would have been $10^{-7}$ — an fp16 subnormal teetering on the underflow cliff — becomes $65536 \times 10^{-7} \approx 6.6 \times 10^{-3}$, a comfortable, fully-normal fp16 value with plenty of mantissa precision. The whole gradient histogram shifts up by 16 powers of two, lifting the small values out of the subnormal danger zone and into the heart of fp16's representable range. Then, *after* backward but *before* the optimizer step, you divide every gradient by the same $S$ to restore the true gradient values, and the weight update proceeds as if nothing happened. The scale is a temporary lift that never touches the actual update magnitude.

![timeline of loss scaling showing loss multiplied by a scale factor before backward then gradients unscaled before the optimizer step](/imgs/blogs/numerical-formats-and-mixed-precision-fp32-tf32-bf16-fp16-fp8-3.png)

The figure traces the sequence: compute the loss in fp32, multiply by the scale (say 65536), run backward so the now-large gradients land safely in fp16 range, unscale by dividing by the same factor, check whether any gradient went `inf` or `nan` (an overflow — the scale was too aggressive for this batch), and only then take the optimizer step in fp32. The overflow check is what makes it robust: if scaling pushed some gradient past fp16's 65,504 ceiling, you *skip* that step entirely rather than corrupt the weights, and you back off the scale.

The reason a *fixed* scale is fragile — and why you almost never hardcode one — is that the right scale changes during training. Early on, gradients are large and a high scale overflows; later they shrink and a low scale underflows. So production loss scaling is **dynamic**: start with a large scale, multiply it up (e.g. double it) every N successful steps to keep gradients as high as possible, and halve it immediately whenever an overflow is detected. PyTorch's `GradScaler` implements exactly this policy. You never tune the number by hand.

Here is the punchline that explains why this whole section is *optional* for most readers in 2026: **bf16 does not need any of this.** Because bf16 shares fp32's exponent, its gradients never underflow in the first place, so there is no histogram to shift. `GradScaler` is a no-op for bf16 and you simply do not use it. Loss scaling is the price of fp16's narrow exponent. The moment hardware gave us bf16 — first on TPUs, then on A100 and every NVIDIA GPU since — the entire loss-scaling machinery became something you only reach for on older fp16-only silicon (think V100, T4, consumer Turing/early-Ampere cards without good bf16 throughput) or in inference paths that are committed to fp16. For new training, bf16 is the default precisely because it deletes this whole class of problem.

#### Worked example: dynamic loss scale arithmetic

Say you start with scale $S = 2^{16} = 65536$ and the policy is "double every 2000 successful steps, halve on overflow." At step 500 a batch produces a gradient that, after scaling, hits $80{,}000 > 65{,}504$ — overflow. The scaler detects the `inf`, *skips* the optimizer step (weights untouched), and sets $S = 2^{15} = 32768$. Training continues. Over the next 2000 clean steps the scaler doubles back up to $S = 2^{16}$, probes whether it can go higher, hits $2^{17} = 131072$, finds it overflows again, and settles. The scale hovers just below the overflow ceiling, keeping the most gradient signal in range without corrupting a step. The cost of the occasional skipped step is negligible — a few steps in tens of thousands. The benefit is that fp16 training stays numerically alive.

## Which format goes where: the mixed in mixed precision

The word "mixed" in mixed precision is doing real work. You do not pick one format for the whole model. You assign each *tensor* the cheapest format it can tolerate, and you keep a few critical things in fp32 no matter what. Getting this assignment right is the difference between a stable run and a subtly broken one.

![matrix showing which numerical format each tensor uses with master weights and optimizer state in fp32 and compute tensors in bf16](/imgs/blogs/numerical-formats-and-mixed-precision-fp32-tf32-bf16-fp16-fp8-4.png)

The figure lays out the standard assignment, and every row has a reason rooted in the range-versus-precision trade-off:

- **Master weights → fp32.** This is the canonical copy of the parameters that the optimizer updates, and it stays in fp32 even when everything else is low precision. The reason is **precision**, not range: weight updates are tiny relative to the weights. A weight of `1.0` getting an update of `1e-7` needs the update to actually land. In bf16, the ULP at 1.0 is about $7.8 \times 10^{-3}$, so adding `1e-7` to `1.0` rounds *right back to 1.0* — the update vanishes into the gap between representable numbers. This is **swamping**, and it is bf16's real failure mode: not underflow of the gradient, but the update being smaller than one ULP of the weight. Keep the master weight in fp32 (ULP $\approx 1.2 \times 10^{-7}$) and the tiny update accumulates correctly over many steps.
- **Compute weights → bf16.** A bf16 *copy* of the weights is what actually feeds the matmul. It is cheap (2 bytes), it streams fast, and the Tensor Cores want it. The coarse mantissa is fine here because the *forward* and *backward* matmuls accumulate in fp32 inside the Tensor Core regardless of the input format.
- **Activations → bf16.** Half the memory, half the HBM traffic. For a memory-bound model (see [the roofline model](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound)), halving activation bytes can directly halve the time of bandwidth-limited layers like LayerNorm and elementwise ops.
- **Gradients → bf16.** Smaller gradients mean the all-reduce in data-parallel training moves half the bytes, which matters enormously once you are communication-bound across many GPUs.
- **Optimizer state → fp32.** Adam's first and second moments must stay in fp32. They accumulate over the whole run, and a coarse format would let the slow-moving second moment estimate drift. This is the largest fp32 cost in the budget — 8 bytes per parameter for the two moments — and it is why bf16 training saves memory on activations but *not* on optimizer state, which is exactly the gap that ZeRO and FSDP target.

The accumulation rule generalizes: **anything that sums many terms stays in fp32.** A reduction over thousands of elements accumulates rounding error term by term; do it in bf16 and the error compounds; do it in fp32 and it stays bounded. This is why the matmul *inputs* are bf16 but the matmul *accumulator* is fp32, and why softmax, LayerNorm statistics, and the loss are computed in fp32 even under autocast. The pattern: low precision for the bulk multiply-adds where each operation's error is independent, fp32 for the reductions where error accumulates.

#### Worked example: the full memory budget of a 7B Transformer

The format-assignment table is not just about stability — it sets your memory bill, and the numbers are worth computing once so you internalize where the bytes go. Take a 7-billion-parameter Transformer trained with AdamW in bf16 mixed precision. Per parameter, the *persistent* state is: the fp32 master weight (4 bytes), the bf16 compute weight (2 bytes), the bf16 gradient (2 bytes), and the two fp32 Adam moments (8 bytes). That is 16 bytes per parameter before a single activation. For 7B parameters: $7 \times 10^9 \times 16 \approx 112$ GB. That already overflows an 80 GB A100 — which is the entire reason ZeRO and FSDP exist, to *shard* that 16-bytes-per-parameter state across data-parallel ranks so each GPU holds only a slice.

Now compare to what a *naive* "everything in fp32" run would cost: fp32 weight (4) + fp32 gradient (4) + two fp32 moments (8) = 16 bytes per parameter too — but with fp32 activations on top, which is where bf16's win actually lands. Activations for a Transformer scale as roughly (batch × sequence length × hidden × layers), and halving their byte width from 4 to 2 directly halves the dominant memory term during the forward pass and the gradient checkpoint storage. For a training step with large batch×sequence, activations often *exceed* the parameter state, so bf16's halving of activation bytes is frequently the difference between fitting and OOMing. The persistent-state bytes are identical between an fp32 and a bf16-AMP run (both keep fp32 master weights and fp32 moments); the savings come from the activations and the gradients you stream. This is the precise reason "bf16 saves memory" is true but commonly mis-attributed — it is the activations, not the weights, doing the saving.

#### Worked example: why the master weight, not the gradient, is the subtle one

It is tempting to think the fp32 master weight is about not losing the gradient, but it is really about not losing the *accumulated history* of tiny updates. Walk a single weight through 10,000 steps. Suppose the true update each step is $+1 \times 10^{-7}$ and the weight starts at $1.0$. In fp32, after 10,000 steps the weight is $1.0 + 10^4 \times 10^{-7} = 1.001$ — the updates accumulated correctly, because each $10^{-7}$ addition lands (fp32 ULP at 1.0 is $\approx 1.2 \times 10^{-7}$, so the update is just above one ULP and survives). In bf16, the ULP at 1.0 is $\approx 7.8 \times 10^{-3}$, so each $10^{-7}$ update is about 78,000 times smaller than one ULP — `1.0 + 1e-7` rounds straight back to `1.0`, every single step, forever. The weight never moves. After 10,000 steps it is still exactly `1.0`. The gradient was fine, the optimizer computed the right update, and the weight still did not learn — because the *accumulator* could not resolve the update. That is swamping, and the fp32 master weight is the only fix: accumulate in fp32, downcast a bf16 copy for the matmul, repeat. No amount of loss scaling helps here, which is why even bf16 (which needs no loss scaling) still needs fp32 master weights.

## AMP: how PyTorch wires it all together

You almost never implement the format assignment above by hand. PyTorch's **Automatic Mixed Precision** (AMP) does it for you through two cooperating pieces: `torch.autocast`, a context manager that picks the format *per operation*, and `torch.cuda.amp.GradScaler` (now `torch.amp.GradScaler`), which handles loss scaling for fp16. The beauty of AMP is that it encodes the "which format where" table as a built-in policy.

![graph showing autocast routing matmul and convolution ops to bf16 while keeping reductions softmax and loss in fp32](/imgs/blogs/numerical-formats-and-mixed-precision-fp32-tf32-bf16-fp16-fp8-5.png)

The figure shows the routing logic. When an op runs inside `autocast`, PyTorch consults an internal allowlist. Matmuls and convolutions — the multiply-add-heavy ops where low precision is safe and where the Tensor Cores live — get cast to bf16 (or fp16) and dispatched to the fast Tensor Core path. Reductions, softmax, and the loss — the ops where error accumulates — are kept in fp32. The result comes back into the autograd graph and the next op gets routed by the same policy. You write your model in fp32 and AMP transparently runs the cheap parts cheap and the sensitive parts safe.

Here is the canonical bf16 training loop for our running example, a Transformer:

```python
import torch
from torch import nn

device = "cuda"
model = MyTransformer().to(device)            # parameters live in fp32 (master weights)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

# bf16 path: autocast picks the format per op; NO GradScaler needed for bf16.
for step, (x, y) in enumerate(loader):
    x, y = x.to(device), y.to(device)
    opt.zero_grad(set_to_none=True)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(x)                     # matmuls run in bf16 on Tensor Cores
        loss = nn.functional.cross_entropy(logits, y)  # softmax + loss stay fp32

    loss.backward()                           # grads computed; bf16 grads never underflow
    opt.step()                                # update applied to fp32 master weights
```

That is the whole thing. The master weights stay fp32 because `model` was created in fp32 and `autocast` never changes the stored parameters — it only casts *inputs to ops* on the fly. The matmuls run bf16, the loss runs fp32, and because it is bf16 there is no scaler. This loop, on an A100, will run roughly 3x faster than the same loop without `autocast`, at matched accuracy, and you changed three lines.

There is a critical distinction hiding in that paragraph that trips up a lot of engineers: `autocast` is **not** the same as `model.half()` or `model.to(torch.bfloat16)`. If you call `model.bfloat16()`, you have permanently converted the stored parameters to bf16 — you have thrown away the fp32 master weights, and now your weight updates will be swamped exactly as the worked example above described. The model will train, but worse, and you will spend a day wondering why. `autocast` is the right tool precisely because it leaves the parameters in fp32 and only casts the *operands flowing into individual ops*, op by op, keeping the master copy intact. The rule: create your model in fp32, wrap the forward in `autocast`, and never permanently downcast the parameters. The one exception is pure inference, where you have no optimizer and no updates to swamp, so `model.bfloat16()` is fine and saves the memory of the second copy.

It is also worth being precise about what `autocast` does and does not cover. It governs the operations *inside its context block* — the forward pass. The backward pass automatically runs each gradient computation in the dtype its forward op used (PyTorch records this), so you do *not* wrap `loss.backward()` in autocast. And the optimizer step happens outside autocast entirely, operating on the fp32 master weights and fp32 gradients (after unscaling, in the fp16 case). So the mental flow is: fp32 parameters live permanently; autocast casts forward-op inputs to bf16; backward mirrors the forward dtypes; the step is pure fp32. Get that sequence right and mixed precision is robust; confuse it with a blanket `.half()` and you have quietly broken your training.

The **fp16** version needs the scaler. Here is the contrast, and it is exactly the loss-scaling timeline from earlier turned into code:

```python
scaler = torch.amp.GradScaler()              # dynamic loss scaling for fp16

for step, (x, y) in enumerate(loader):
    x, y = x.to(device), y.to(device)
    opt.zero_grad(set_to_none=True)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)

    scaler.scale(loss).backward()            # multiply loss by scale, then backward
    scaler.step(opt)                         # unscale grads, skip step if inf/nan seen
    scaler.update()                          # raise or lower the scale dynamically
```

The three changed lines — `scaler.scale(loss).backward()`, `scaler.step(opt)`, `scaler.update()` — are precisely the multiply, unscale-and-conditionally-step, and dynamic-adjust stages of the loss-scaling timeline. Swap `torch.float16` for `torch.bfloat16` and delete the scaler, and you are back to the simpler bf16 loop. That swap is the practical embodiment of everything in the range-versus-precision discussion: bf16 deletes the scaler because bf16 deletes the underflow.

One more snippet worth keeping in your back pocket: inspecting the formats directly so you stop guessing about their limits.

```python
import torch

for name, dt in [("fp32", torch.float32),
                 ("fp16", torch.float16),
                 ("bf16", torch.bfloat16)]:
    fi = torch.finfo(dt)
    print(f"{name}: max={fi.max:.4g}  min_normal={fi.tiny:.4g}  "
          f"eps={fi.eps:.4g}  resolution={fi.resolution:.4g}")

# A live demonstration of the fp16 underflow cliff:
g = torch.tensor(1e-7, dtype=torch.float32)
print("1e-7 as fp16:", g.to(torch.float16).item())   # -> coarse subnormal, near 1e-7
print("5e-8 as fp16:", torch.tensor(5e-8).to(torch.float16).item())  # -> 0.0  (underflow!)
print("5e-8 as bf16:", torch.tensor(5e-8).to(torch.bfloat16).item()) # -> ~5e-8 (survives)
```

That last block is the entire post in five lines: a `5e-8` gradient becomes a hard `0.0` in fp16 and stays nonzero in bf16. Run it; watching the zero appear is more convincing than any table.

## tf32: a near-free speedup for fp32 code

There is a fourth format that you can turn on without touching your model at all, and most people leave free performance on the table by not knowing about it. **tf32** (TensorFloat-32) is NVIDIA's Ampere-and-later format that lives *inside the Tensor Core*. It is not a storage format — your tensors stay fp32 in memory. tf32 is what the Tensor Core does to fp32 inputs when you let it: it takes the fp32 operands, rounds their mantissas down to 10 bits (keeping the full 8-bit exponent, hence the full fp32 range), multiplies on the Tensor Core, and accumulates the result back in fp32.

![before and after diagram of the tf32 Tensor Core path showing fp32 inputs truncated to ten mantissa bits inside the matmul for an eight times speedup](/imgs/blogs/numerical-formats-and-mixed-precision-fp32-tf32-bf16-fp16-fp8-6.png)

The figure shows the trade. On the left, a true fp32 matmul runs on the CUDA cores with all 23 mantissa bits and hits the A100's fp32 peak of **19.5 TFLOP/s**. On the right, the tf32 path feeds the same fp32 inputs to the Tensor Cores using only 10 mantissa bits and reaches **156 TFLOP/s** — about 8x faster — while inputs and outputs remain fp32 in your code. The exponent is untouched, so range is identical to fp32; you lose precision only inside the matmul, dropping from ~7 decimal digits to ~3. For training, where bf16's 7-bit mantissa is already plenty, tf32's 10-bit mantissa is more than enough, and the speedup is essentially free.

Enabling it is one or two lines:

```python
import torch

# Let cuBLAS/cuDNN use the tf32 Tensor Core path for fp32 matmuls and convolutions.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Newer, preferred knob (PyTorch 2.x): set the global fp32 matmul precision.
torch.set_float32_matmul_precision("high")   # "high" -> tf32; "highest" -> true fp32
```

A subtlety worth knowing: in older PyTorch, `allow_tf32` for matmuls defaulted to `False` (so fp32 matmuls ran in true fp32 unless you opted in), while cuDNN convolutions defaulted to `True`. Defaults have shifted across versions, so the safe move is to set the knob explicitly. If you are running pure-fp32 code and have not set `set_float32_matmul_precision("high")`, you may be leaving an 8x matmul speedup unclaimed for no accuracy benefit. The only time you want `"highest"` (true fp32) is when you genuinely need every mantissa bit — some scientific-computing or ill-conditioned linear-algebra workloads — which is essentially never in deep-learning training.

tf32 is the gentlest rung on the precision ladder: it accelerates code you already wrote, in a format you already use, with a range identical to fp32 and a precision loss that deep learning does not notice. It is the first thing to flip on, before you even reach for autocast.

One clarification that confuses people: tf32 and bf16 are *not* redundant even though both keep the 8-bit fp32 exponent. tf32 has 10 mantissa bits and lives only inside the Tensor Core — it never reduces your memory footprint, because your tensors stay 4-byte fp32 in HBM. bf16 has 7 mantissa bits and is a genuine 2-byte storage format — it halves your memory and your bandwidth. So tf32 buys you matmul *speed* on fp32 code with zero memory change; bf16 buys you matmul speed *and* halved memory but requires you to actually adopt a 16-bit format with autocast and master weights. The progression up the ladder is: tf32 first (free, no code change, no memory change), then bf16 AMP (more speed, half the memory, three lines of code), then fp8 (more speed and memory still, but real complexity). Each rung trades a bit more precision for a bit more performance, and you climb only as far as your accuracy budget and your hardware allow.

There is also a relationship to the matmul accumulation point from earlier. Even in tf32 mode, the *accumulator* inside the Tensor Core is fp32 — only the *inputs* are rounded to 10 mantissa bits before multiplication. This is the same pattern as bf16: low-precision operands, fp32 accumulation. It is what keeps a long matmul (a reduction over the inner dimension, which might be thousands of elements) from compounding rounding error. The Tensor Core hardware was designed around this insight — feed it cheap operands, accumulate the partial sums in fp32 — and it is why all of these formats can be so aggressive on the inputs without wrecking the result.

## fp8 on Hopper: the new frontier

The aggressive end of the precision ladder is **fp8**: eight bits total, and the headline accelerator on NVIDIA's Hopper (H100) and later Blackwell GPUs. At 8 bits you have almost nothing to spend, so fp8 does not even pretend one layout fits all — it ships *two*, and the choice between them is the cleanest illustration of the range-versus-precision trade in the whole post.

![matrix comparing the two fp8 formats E4M3 and E5M2 showing E4M3 keeps more mantissa for the forward pass and E5M2 keeps more exponent for gradients](/imgs/blogs/numerical-formats-and-mixed-precision-fp32-tf32-bf16-fp16-fp8-7.png)

The figure compares the two. **E4M3** spends its bits as 4 exponent, 3 mantissa, topping out at a max value of **448**. The extra mantissa bit (3 versus 2) buys precision, so E4M3 is used for the **forward pass** — weights and activations, where the values are bounded and you want resolution. **E5M2** spends its bits as 5 exponent, 2 mantissa, reaching a max of **57,344**. The extra exponent bit buys range at the cost of a mantissa bit, so E5M2 is used for **gradients in the backward pass**, where values spread across a much wider dynamic range and range matters more than resolution. It is the bf16-versus-fp16 trade replayed at 8 bits: forward wants precision (E4M3), backward wants range (E5M2).

Even with two layouts, fp8's dynamic range is so cramped that training in it requires **per-tensor scaling** — a learned or tracked scale factor applied to each tensor to keep its values centered in the representable window, which is a generalization of loss scaling applied everywhere, not just to the loss. You do not implement this yourself. NVIDIA's **Transformer Engine** library handles the format selection, the scaling factors, and the casts. A minimal fp8 training region looks like this:

```python
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format

# Build the model from Transformer Engine's fp8-aware layers.
model = te.TransformerLayer(hidden_size=4096, ffn_hidden_size=16384,
                            num_attention_heads=32).cuda()

# E4M3 for forward, E5M2 for backward, with delayed per-tensor scaling.
recipe = DelayedScaling(fp8_format=Format.HYBRID,
                        amax_history_len=16, amax_compute_algo="max")

for x, y in loader:
    with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
        out = model(x)                  # GEMMs run in fp8 on Hopper Tensor Cores
    loss = loss_fn(out, y)
    loss.backward()                     # gradient GEMMs use E5M2
    opt.step(); opt.zero_grad()
```

`Format.HYBRID` is exactly the E4M3-forward / E5M2-backward split from the figure, and `DelayedScaling` tracks a short history of each tensor's maximum absolute value (`amax`) to pick the per-tensor scale without a synchronization stall every step. The payoff is large: on an H100, fp8 matmul peaks around **1979 TFLOP/s** (with the sparsity-free dense rate roughly half the marketing "with sparsity" number — always check which the spec sheet quotes), versus about 989 TFLOP/s for bf16 on the same chip. fp8 roughly doubles matmul throughput again on top of bf16, and it halves the bytes once more.

Why two scaling strategies even matter here is worth a sentence. With only 4 or 5 exponent bits, an fp8 tensor's representable window is so narrow that the *whole tensor* must be shifted to sit inside it — and different tensors (a weight matrix, an activation map, a gradient) have wildly different magnitude ranges, so each needs its *own* scale factor, recomputed as those magnitudes drift during training. That is per-tensor scaling, and `DelayedScaling` is the trick that makes it cheap: instead of doing a blocking reduction over the tensor to find its max every step (which would stall the pipeline), it reuses a short rolling history of recent maxima to predict the scale, so the scale computation overlaps with compute instead of gating it. It is loss scaling's idea — keep values in range — generalized to every tensor and made non-blocking.

The honest caveat: fp8 training is not the free lunch bf16 is. It demands the scaling machinery, it is more sensitive to outlier activations, and several reported fp8 training runs keep the most sensitive layers (the first and last, the embeddings, sometimes attention) in bf16 while running only the big middle GEMMs in fp8. It is a frontier technique that pays off at large scale where the matmul dominates, and it is overkill for a small fine-tune. The mental model for when fp8 earns its complexity: if your step is dominated by large matmuls (big hidden dimension, long sequences, lots of layers) and you are on H100 or Blackwell, fp8 can push throughput another ~1.3x past bf16 and halve the matmul bytes again; if your step is dominated by memory movement, small ops, or the data loader, fp8 buys you little and the scaling bookkeeping is pure overhead. We will see real fp8 training numbers in the case studies.

## The running example, measured: bf16 versus fp32 on an A100

Time to make the running Transformer concrete with named-hardware numbers. We train a mid-sized Transformer on a single **A100 80GB SXM** and measure fp32 baseline against bf16 AMP. The A100's relevant peaks, from NVIDIA's datasheet, are **19.5 TFLOP/s** dense fp32 (CUDA cores), **156 TFLOP/s** tf32, **312 TFLOP/s** dense bf16/fp16 (Tensor Cores), and **2.0 TB/s** of HBM2e bandwidth.

![before and after diagram comparing fp32 baseline against bf16 with AMP on an A100 showing higher throughput and halved activation memory](/imgs/blogs/numerical-formats-and-mixed-precision-fp32-tf32-bf16-fp16-fp8-8.png)

The figure summarizes the win: fp32 runs on the 19.5 TFLOP/s CUDA-core path with 4-byte activations at 1.0x throughput; bf16 with AMP runs on the 312 TFLOP/s Tensor Core path with 2-byte activations at roughly 3x throughput. Two things changed — the matmul moved to the Tensor Cores (a 16x peak-FLOP/s jump on paper) and the activation bytes halved — and together they deliver the speedup at matched accuracy.

Here is a representative before-after table. Treat the absolute numbers as approximate and workload-dependent; the *ratios* are what generalize.

| Metric | fp32 (CUDA cores) | tf32 (Tensor Cores) | bf16 AMP (Tensor Cores) |
| --- | --- | --- | --- |
| Peak matmul (A100) | 19.5 TFLOP/s | 156 TFLOP/s | 312 TFLOP/s |
| Achieved (approx.) | ~16 TFLOP/s | ~95 TFLOP/s | ~145 TFLOP/s |
| MFU (achieved/peak) | ~80% (low peak) | ~60% | ~46% |
| Activation memory | 4 bytes/elem (1.0x) | 4 bytes/elem | 2 bytes/elem (0.5x) |
| Step time (relative) | 1.0x | ~0.40x | ~0.37x |
| Throughput | 1.0x | ~2.5x | ~2.7x |
| Final accuracy | baseline | within noise | within noise |

Note the subtlety in the **MFU** column (MFU = model FLOP/s utilization, achieved FLOP/s divided by the hardware's peak FLOP/s). fp32 shows a deceptively high ~80% MFU because its *peak is tiny* — 80% of 19.5 is still slow. bf16 shows a lower ~46% MFU because its peak is huge — but 46% of 312 is roughly 145 TFLOP/s, nearly 9x the fp32 achieved number. **Never compare MFU across precisions without the absolute throughput.** A higher MFU at a lower peak is worse, not better. This is the trap that makes people think they are "efficient" in fp32 when they are simply slow.

#### Worked example: how to measure this honestly

If you want to reproduce the table without fooling yourself, the measurement protocol matters more than the code. The pitfalls: GPU kernels are asynchronous, so you must `torch.cuda.synchronize()` before reading a timer or you will time the launch, not the work; the first few steps include CUDA graph capture, cuDNN autotuning, and allocator warmup, so discard them; and the data loader can silently dominate, making the GPU look slow when it is actually starved.

```python
import torch, time

def benchmark(model, batch, dtype, steps=50, warmup=10):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    use_scaler = (dtype == torch.float16)
    scaler = torch.amp.GradScaler(enabled=use_scaler)

    for i in range(warmup + steps):
        if i == warmup:
            torch.cuda.synchronize(); t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype):
            loss = model(batch).float().mean()
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / steps
    print(f"{dtype}: {dt*1e3:.1f} ms/step  "
          f"peak_mem={torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    torch.cuda.reset_peak_memory_stats()
```

Run `benchmark(model, batch, torch.float32)` then `benchmark(model, batch, torch.bfloat16)` and the ratio of `ms/step` is your real speedup. Watch `peak_mem` halve on the activation side. If the speedup is far below the ~2.7x you expected, your model is probably memory-bound, not compute-bound — the matmul speedup only helps the compute-bound layers, and the [roofline model](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) tells you which regime you are in. That is the single most common reason "bf16 didn't speed me up much": the bottleneck was never the matmul.

## A deliberate failure: overflow to NaN, then fixed by scaling

Let us watch fp16 break and then fix it, because seeing the failure makes the loss-scaling argument visceral. The cleanest reproduction of fp16's fragility is to push a value past its 65,504 ceiling.

```python
import torch

# Overflow: a value beyond fp16's max (65504) becomes inf, and inf - inf = nan.
x = torch.tensor([60000.0, 60000.0], dtype=torch.float16)
y = x * 2                      # 120000 > 65504  ->  inf
print("fp16 overflow:", y)    # tensor([inf, inf], dtype=float16)
print("inf - inf  ->", (y - y))   # tensor([nan, nan]) — the poison spreads

# The same in bf16 is fine: bf16's max is ~3.4e38.
xb = torch.tensor([60000.0, 60000.0], dtype=torch.bfloat16)
print("bf16:", xb * 2)        # tensor([120000., 120000.], dtype=bfloat16) — no overflow
```

In a real fp16 run the overflow is not this blatant; it sneaks in when a loss spike or an unscaled gradient crosses 65,504, produces an `inf`, and the next subtraction or normalization turns it into `nan`, which then propagates to every weight downstream and the loss prints `nan` for the rest of the run. Now the underflow-and-fix demonstration, which is the more common fp16 problem and the one loss scaling targets:

```python
import torch

# Underflow: a tiny gradient rounds to zero in fp16, killing the update.
grad = torch.tensor(3e-8, dtype=torch.float32)
print("3e-8 -> fp16:", grad.to(torch.float16).item())   # 0.0  (underflow)

# Loss scaling fixes it: lift the gradient up, then unscale after.
scale = 2.0 ** 16                       # 65536
scaled = (grad * scale).to(torch.float16)
print("scaled in fp16:", scaled.item())           # ~0.00197, safely representable
recovered = scaled.float() / scale
print("recovered:", recovered.item())              # ~3.0e-8, the update is back
```

The flow is the loss-scaling timeline made literal: `3e-8` becomes a hard `0.0` if cast directly, but multiply by 65536 first and it lands at about `0.00197` — a healthy fp16 value — then divide back out after the backward pass to recover the true `3e-8`. The update survives. And the bf16 version of the same gradient never needed any of this, because `3e-8` is a perfectly ordinary bf16 number. That contrast — same gradient, fp16 needs a 16-bit shift to save it, bf16 just holds it — is the whole reason bf16 won as the training default.

## Case studies and real numbers

Specs are one thing; what actually ships is another. A few load-bearing results from the literature and vendor reports, with the numbers marked approximate where the source rounds.

**bf16 is the LLM training default, and the field switched deliberately.** Google's TPUs used bf16 from the start, and the lesson transferred to GPUs the moment A100 added bf16 Tensor Cores. Meta's LLaMA and most open LLMs since (Mistral, the OLMo line, Qwen) train in bf16 with fp32 master weights and fp32 optimizer state, no loss scaling. The reason given is exactly the one above: bf16's fp32-range exponent removes the gradient-underflow failure mode that made fp16 training a babysitting exercise. The mantissa cost (7 bits versus fp16's 10) turns out not to matter for convergence because the fp32 master weights and fp32 accumulators absorb the precision loss where it would compound. The empirically reported accuracy delta between bf16 and fp32 training is, across many papers, within run-to-run noise — there is no measurable quality penalty for the ~2.5–3x throughput and ~2x activation-memory win.

**Micikevicius et al., "Mixed Precision Training" (2018), is the founding result and a loss-scaling case study.** Training a range of networks (ResNet, Inception, large LSTMs, GNMT translation, DeepSpeech 2, GANs) in fp16 with an fp32 master copy and loss scaling, they matched fp32 accuracy across the board while roughly halving memory and substantially increasing throughput on V100 Tensor Cores. The paper's three ingredients — fp32 master weights, loss scaling, and fp32 accumulation for reductions — are exactly the three rows that stay fp32 in the format-assignment matrix. It is worth reading because it derives, from first principles, why those three and only those three need fp32, and it is the reason the AMP API has the shape it does.

**fp8 training reports show the ceiling rising again.** NVIDIA's Transformer Engine numbers and several 2024–2025 large-model reports describe end-to-end fp8 training (forward GEMMs in E4M3, backward in E5M2, per-tensor delayed scaling) reaching roughly 1.2–1.5x the bf16 throughput on H100 at large scale, with convergence matching bf16 *when the sensitive layers are kept higher precision*. DeepSeek-V3's training, for instance, used an fp8 mixed-precision scheme for the bulk GEMMs while keeping a short list of operations in higher precision, reporting a small and manageable relative loss-error budget. The consistent message: fp8 is real and it works at scale, but it is a careful technique with per-tensor scaling and a precision-sensitive shortlist, not the flip-a-switch experience of bf16. H100's fp8 peak of ~1979 TFLOP/s (dense) versus ~989 bf16 is the carrot; the scaling bookkeeping is the cost.

**Inference is its own story, and it leans on the quantization toolkit.** Serving stacks routinely run weights in int8 or fp8 and activations in fp16/bf16, trading a small accuracy delta for a large throughput and memory win. The mechanics — calibration, per-channel scales, outlier handling — are the inference-time cousins of training's per-tensor scaling, and they are covered in [quantization from first principles](/blog/machine-learning/edge-ai/quantization-from-first-principles) and the serving-side trade-offs in [efficient LLM inference techniques](/blog/machine-learning/large-language-model/efficient-llm-inference-techniques). The thread connecting training and inference is the same one running through this whole post: spend bits where the numerics tolerate it, keep range where the values demand it. One worth noting: inference can often go *more* aggressive than training because there is no gradient to underflow and no accumulator to swamp — a frozen weight in int8 with a good per-channel scale loses very little, which is why int8 and fp8 weight quantization for serving is mature and widespread while fp8 *training* is still a frontier technique.

**The TPU precedent, and why the field trusts bf16.** It is easy to forget that bf16 is not new — Google's TPUs have used bf16 (which Google originally called "brain floating point," hence the *b*) as their native matmul format since roughly 2018, training production models at Google scale in it for years before the A100 brought it to GPUs in 2020. That track record is why, when bf16 Tensor Cores arrived on Ampere, the deep-learning community adopted it almost instantly without the years of skepticism that fp16 training had faced. The accumulated evidence — many models, many domains, training to fp32-matched accuracy — had already been gathered on TPUs. The lesson the field internalized: for *training*, dynamic range beats precision, and 7 mantissa bits with fp32 accumulators is enough. Everything in this post is downstream of that hard-won consensus.

#### Worked example: estimating the dollar cost a precision switch saves

Make the throughput win concrete in money, since that is ultimately what a precision decision buys. Suppose your running Transformer takes 10,000 A100-hours to train in fp32, and an A100 rents for roughly \$2 per GPU-hour (cloud spot pricing is in this ballpark; on-demand is higher). That is a \$20,000 run. Switch to bf16 AMP and assume a 2.7x throughput improvement: the same training now takes about 3,700 A100-hours, or roughly \$7,400 — a saving of about \$12,600 on a single run, for three lines of code and no accuracy loss. Scale that across a research team running dozens of experiments a month and the precision choice is worth six figures a year. Add tf32 to any remaining fp32 work and the savings compound. This is why "just turn on AMP" is not a micro-optimization — it is frequently the single highest-leverage change an ML engineer can make to a training budget, and it is the cheapest of all the levers in the HPC playbook because it costs almost nothing to apply.

## Common pitfalls and how to debug them

Even with bf16's robustness, mixed precision has a handful of failure modes that account for nearly every "it NaN'd" or "it's slower than I expected" support ticket. Knowing them turns a day of confusion into a five-minute fix.

**The NaN that appears mid-run.** A loss that is healthy for thousands of steps and then prints `nan` is almost always an overflow somewhere — a value crossed the format's ceiling, became `inf`, and a subtraction or division turned it into `nan`. In fp16 this is common (ceiling 65,504) and the fix is loss scaling with overflow-skip, which `GradScaler` does automatically. In bf16 it is rare (ceiling $3.4 \times 10^{38}$) and when it happens the culprit is usually not the format but a genuine numerical bug: a division by a near-zero denominator, a `log` of a negative number, an attention score that exploded because of an un-normalized input. The debugging move is the same in both cases: set `torch.autograd.set_detect_anomaly(True)` to get a stack trace at the first NaN-producing op, or sprinkle `assert torch.isfinite(x).all()` checks to bisect which layer first goes non-finite. Do not reflexively add `GradScaler` to a bf16 run that NaNs — it will not help, because bf16 does not underflow; find the real bug.

**The "bf16 didn't speed me up" surprise.** You flip on autocast and step time barely moves. Almost always this means your workload is memory-bound, not compute-bound, so the matmul speedup does not help — your bottleneck is HBM bandwidth or the data loader, not the Tensor Cores. The [roofline model](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) is the right diagnostic: compute your arithmetic intensity and see which side of the ridge you sit on. A second common cause is small matrices: Tensor Cores need reasonably large GEMMs to reach their peak, and a tiny model or a tiny batch leaves them mostly idle regardless of precision. A third is that you forgot tf32 was already on for the fp32 baseline, so your "fp32" number was secretly fast and the bf16 delta looks small. Always benchmark with synchronization and warmup, as in the measurement worked example.

**Silent accuracy loss from over-aggressive casting.** If you bypass autocast and manually cast things — say you `.half()` a normalization layer, or you compute a softmax in bf16, or you let a reduction run in low precision — you can lose accuracy in ways that do not announce themselves. The fix is to trust the autocast allowlist: it keeps softmax, LayerNorm statistics, and the loss in fp32 for a reason. If you write a custom op, register it correctly with autocast or wrap its sensitive parts in `with torch.autocast(enabled=False):` so they run fp32. The most common self-inflicted version of this is a custom loss or a custom attention that sums in bf16; force the accumulation to fp32 with `.float()` before the sum.

**The checkpoint that does not match.** Save your master weights in fp32. If you checkpoint a bf16 copy and resume from it, you have permanently rounded your weights to bf16 precision at the save point, and resuming will not perfectly match continuing — the swamping-prone updates restart from a coarsened state. For exact reproducibility across a resume, the checkpoint must contain the fp32 master weights and the fp32 optimizer state, not the bf16 compute copies.

**fp8 outlier sensitivity.** When you do graduate to fp8, the failure mode shifts: fp8's tiny range makes it acutely sensitive to outlier activations (a single large value can dominate the per-tensor scale and crush everyone else into a couple of mantissa bits). This is why fp8 recipes track an `amax` history and why several production runs keep the layers most prone to outliers — often the first and last blocks, and sometimes attention — in bf16. If an fp8 run diverges, the first thing to try is widening the bf16 shortlist, not retuning the scale.

## When to reach for each format (and when not to)

A decisive recommendation, because every format is a trade and the wrong one costs you accuracy or speed.

**Default to bf16 + AMP for all training on Ampere or newer.** It is the free lunch: ~2.5–3x throughput, ~2x activation-memory savings, no loss scaling, no measurable accuracy loss. If you are on an A100, H100, or any recent GPU and you are training in fp32, you are simply leaving performance on the floor. There is no reason not to flip on `torch.autocast(dtype=torch.bfloat16)`.

**Turn on tf32 unconditionally for any remaining fp32 work.** `torch.set_float32_matmul_precision("high")` costs you nothing in accuracy for deep learning and gives an ~8x matmul speedup on the parts that stay fp32. It is strictly dominant over true fp32 for training. Reserve `"highest"` for the rare ill-conditioned numerical workload that genuinely needs 23 mantissa bits.

**Reach for fp16 only when bf16 is unavailable.** On older hardware without good bf16 throughput — V100, T4, early consumer Turing — fp16 with `GradScaler` is your mixed-precision path, and the loss-scaling machinery earns its keep. On modern hardware, prefer bf16 every time; the only reason to pick fp16 is a deployment target locked to it.

**Reach for fp8 only at large scale, with the right library, and with eyes open.** fp8 pays off when the matmul dominates your step (large models, long sequences) and you are on H100/Blackwell with Transformer Engine. It is not worth the per-tensor-scaling complexity and the precision-sensitivity for a small fine-tune or a model where memory traffic, not matmul, is the bottleneck. Keep the sensitive layers in bf16; do not try to push everything to fp8 on the first attempt.

**Keep fp32 for the three things that need it, always.** Master weights, optimizer state, and reduction accumulators stay fp32 regardless of how aggressive your compute precision is. These are the precision-critical, error-accumulating, tiny-update parts of training, and dropping them to low precision is how you get a run that trains *almost* right and loses a point of accuracy you cannot explain. The savings are not worth it; this is the line you do not cross.

A quick decision table to keep at hand:

| Situation | Use | Why |
| --- | --- | --- |
| Training on A100/H100/newer | bf16 + AMP, no scaler | fp32 range, no underflow, ~3x faster |
| Remaining fp32 matmuls | tf32 (`matmul_precision="high"`) | ~8x free, identical range |
| Training on V100/T4/Turing | fp16 + `GradScaler` | no bf16; loss scaling fixes underflow |
| Large model, H100, matmul-bound | fp8 via Transformer Engine | ~1.3x over bf16, halve bytes again |
| Master weights / optimizer / reductions | fp32, always | tiny updates and accumulation need range and precision |

## Key takeaways

- A float is a sign, an **exponent** (sets dynamic range — how big and small), and a **mantissa** (sets precision — how many significant figures). The value is $(-1)^s (1+m) 2^{e-\text{bias}}$, and precision is *relative*: the ULP grows with magnitude.
- At a fixed 16 bits you choose range or precision. **bf16** keeps fp32's 8-bit exponent (range from ~$10^{-38}$ to ~$3.4\times10^{38}$) and pays in mantissa; **fp16** keeps 10 mantissa bits and pays in range (only ~$6\times10^{-5}$ to 65,504).
- That range gap is why fp16 underflows a `1e-7`-class gradient to zero (its floor is ~$6\times10^{-8}$) while bf16 holds it trivially. fp16 also overflows past 65,504 to `inf`, which becomes `nan`.
- **Loss scaling** rescues fp16 by multiplying the loss by ~$2^{16}$ before backward (lifting tiny gradients into range) and unscaling before the step, skipping any step that overflowed. **bf16 needs none of this** — that is why it is the default.
- Keep **master weights, optimizer state, and reduction accumulators in fp32** always; tiny updates would otherwise be swamped (smaller than one bf16 ULP) and accumulated error would compound.
- **AMP** wires it up: `autocast` routes matmuls to low precision and keeps softmax/loss/reductions in fp32; `GradScaler` handles fp16 loss scaling. bf16 is `autocast` with no scaler.
- **tf32** is a near-free ~8x matmul speedup for fp32 code — fp32 range, 10-bit mantissa inside the Tensor Core. Turn it on with `set_float32_matmul_precision("high")`.
- **fp8** (E4M3 forward for precision, E5M2 backward for range) roughly doubles matmul throughput again on Hopper but needs per-tensor scaling and a precision-sensitive shortlist.
- On named hardware: A100 peaks at 19.5 (fp32) / 156 (tf32) / 312 (bf16) TFLOP/s; H100 fp8 reaches ~1979 TFLOP/s. Compare *absolute* throughput, never MFU across precisions — a high MFU at a low peak is just slow.

## Further reading

- Micikevicius et al., "Mixed Precision Training" (ICLR 2018) — the founding paper: fp32 master weights, loss scaling, fp32 accumulation, and why each is needed.
- NVIDIA A100 and H100 architecture whitepapers — the source for the 19.5 / 156 / 312 / 989 / 1979 TFLOP/s peaks and the tf32 / fp8 Tensor Core paths.
- NVIDIA Transformer Engine documentation — the fp8 recipes (`DelayedScaling`, `Format.HYBRID`) and per-tensor scaling in practice.
- PyTorch AMP documentation (`torch.autocast`, `torch.amp.GradScaler`) and the `torch.set_float32_matmul_precision` reference.
- [Why HPC is the bottleneck for modern AI](/blog/machine-learning/high-performance-computing/why-hpc-is-the-bottleneck-for-modern-ai) — the three-walls frame this post's precision wall sits inside.
- [The roofline model: compute-bound vs memory-bound](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) — why halving activation bytes helps memory-bound layers and why bf16 sometimes "doesn't speed things up."
- [Inside the GPU: SMs, warps, and the SIMT execution model](/blog/machine-learning/high-performance-computing/inside-the-gpu-sms-warps-and-the-simt-execution-model) — where the Tensor Cores that consume these formats actually live.
- [The HPC playbook for AI engineers](/blog/machine-learning/high-performance-computing/the-hpc-playbook-for-ai-engineers) — the capstone that ties precision together with kernels, parallelism, and collectives.
- [Quantization from first principles](/blog/machine-learning/edge-ai/quantization-from-first-principles) — the inference-time cousin of per-tensor scaling.
- [Efficient LLM inference techniques](/blog/machine-learning/large-language-model/efficient-llm-inference-techniques) — int8/fp8 serving and the throughput trade-offs.

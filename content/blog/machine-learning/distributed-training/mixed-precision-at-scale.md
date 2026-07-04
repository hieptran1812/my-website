---
title: "Mixed Precision at Scale: bf16, fp16, fp8, and the Master Weights That Keep Training Stable"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "The numerics of low-precision distributed training, drawn: why fp16 underflows and bf16 does not, how loss scaling and fp32 master weights keep the loss from stalling, and where fp8 on H100 actually pays off."
tags:
  [
    "distributed-training",
    "multi-gpu",
    "mixed-precision",
    "bf16",
    "fp8",
    "pytorch",
    "fsdp",
    "deep-learning",
    "ml-systems",
    "gpu",
  ]
category: "machine-learning"
subcategory: "Distributed Training"
author: "Hiep Tran"
featured: true
readTime: 41
---

The pull request was one line. Someone changed `torch.float32` to `torch.float16` in the autocast context, the CI throughput benchmark jumped 1.9x, and it got merged on a Friday. By Monday the 7B pretraining run had been flat for 30,000 steps — loss frozen at 4.2, grad-norm reading a suspicious `0.0000` on rank after rank, GPUs pinned at 98% utilization doing arithmetic that changed nothing. Nobody had touched the data, the learning rate, or the model. The only thing that changed was the number of bits each gradient was stored in, and that was enough to silently stop learning while burning \$1,800 an hour of H100 time.

That failure is the whole subject of this post. Mixed precision is the single biggest throughput lever you have after choosing a parallelism strategy: a matmul in 16-bit runs roughly twice as fast on Tensor Cores as the same matmul in fp32, and it halves the memory for activations and the bytes you push across the interconnect for gradients. It is not optional at scale — every serious LLM run uses it. But "just use 16-bit" is exactly the naive instinct that froze that run, because the two 16-bit formats behave completely differently, and the one that CI happened to pick was the wrong one. Understanding *why* fp16 stalls where bf16 sails, and what machinery you add to make either one trainable, is the difference between a 1.9x speedup and a dead run that looks alive.

This is the third memory-and-throughput lever in the [Distributed Training in the Trenches](/blog/machine-learning/distributed-training/why-distributed-training) series, after the [memory budget](/blog/machine-learning/distributed-training/the-memory-budget) that tells you where every gigabyte goes and [activation checkpointing](/blog/machine-learning/distributed-training/activation-checkpointing) that trades compute for memory. Here we go one level below the tensors, into the bits: the three float formats, the loss-scaling trick that rescues fp16, the fp32 master weight that is the "mixed" in mixed precision, and the fp8 frontier that doubles throughput again on H100 if you can validate it. Figure 1 is the whole mechanism on one page — keep it in mind as the spine.

![diagram of the mixed precision loop with an fp32 master weight cast to bf16 for a fast matmul then reduced and stepped in fp32](/imgs/blogs/mixed-precision-at-scale-1.webp)

By the end you will be able to read a `MixedPrecision` config and know exactly which dtype belongs where, choose bf16 / fp16 / fp8 for a given piece of hardware without guessing, add loss scaling correctly when you must use fp16, and explain to the next person who edits that autocast line why their 1.9x speedup is about to freeze the loss.

## The lever: what 16-bit actually buys you

Start with why anyone takes the risk. A modern GPU has two kinds of arithmetic units. The general-purpose CUDA cores do fp32 the classic way. The **Tensor Cores** are dedicated matrix-multiply-accumulate units, and they run at their headline rate only on 16-bit (or 8-bit) inputs. On an A100 SXM the numbers are stark: fp32 matmul tops out around 19.5 TFLOP/s on the CUDA cores, TF32 on the Tensor Cores hits ~156 TFLOP/s, and true 16-bit — bf16 or fp16 — hits ~312 TFLOP/s. On an H100 SXM, bf16 is roughly 990 TFLOP/s dense and fp8 doubles that again to ~1,979 TFLOP/s. The Tensor Core is where the FLOPs live, and the ticket to the Tensor Core is 16-bit inputs. Train in fp32 and you leave more than half the machine idle.

One detail makes this safe, and it is the quiet hero of the whole scheme: **the Tensor Core reads 16-bit inputs but accumulates in fp32.** A matmul is a sum of many products; if you multiplied two bf16 numbers and added the result into a bf16 running sum, thousands of times per output element, the rounding error would pile up catastrophically. Instead the hardware multiplies the bf16 (or fp8) inputs and accumulates the partial sums in a full fp32 register, only rounding back to 16-bit at the very end. So the *inputs* are low precision — that is where the speed and the memory savings come from — but the *reduction inside the matmul* is fp32. This is the same principle we will keep hitting: cast the bulk data to low precision for bandwidth and throughput, but do the summations in high precision so error does not accumulate. The matmul does it in silicon; you do it by hand for the gradient all-reduce and the optimizer.

That is only the compute half. The other half is memory and bandwidth. A bf16 tensor is exactly half the bytes of the fp32 version. That means:

- **Activations** — usually the single largest consumer of memory in a training step — halve. On a 7B model at sequence length 2048 that is the difference between fitting a micro-batch and OOMing.
- **Gradient communication** halves. When you all-reduce gradients across data-parallel ranks, the byte volume is what sets the time. A ring all-reduce moves $2(N-1)/N \cdot S$ bytes per GPU where $S$ is the gradient buffer size; cut $S$ in half by sending bf16 and the comms time halves too. (The collectives themselves are covered in [collectives from scratch](/blog/machine-learning/distributed-training/collectives-from-scratch); here we only care that fewer bytes means less time.)
- **Kernel memory bandwidth** halves. Most non-matmul kernels — layernorm, softmax, elementwise ops — are memory-bound, and they read and write half as many bytes in 16-bit.

Put together, 16-bit is the reason a training step that would take 400 ms in fp32 takes ~210 ms in bf16 on the same card. It is not a micro-optimization; it is the second-largest lever in the whole stack, behind only "which parallelism strategy did you pick." So the question is never *whether* to use it. The question is *which* 16-bit, and what you have to add so that going 16-bit does not quietly break the arithmetic. That is what the mixed-precision loop in Figure 1 exists to answer: compute in 16-bit for the speed, but keep an fp32 master weight and an fp32 gradient sum so the accuracy survives. Everything below is a justification of one of those boxes.

## Three floats and a wish: fp32, fp16, bf16

A floating-point number is three fields: a sign bit, an exponent, and a mantissa (also called the significand or fraction). The value is, for a normalized number,

$$x = (-1)^{s} \times (1.m) \times 2^{\,e - \text{bias}}$$

where $m$ is the mantissa bits interpreted as a binary fraction and $e$ is the exponent field. Two properties fall directly out of the field widths, and they are the entire story:

- The **exponent width** sets the **dynamic range** — how big and how small a number the format can represent at all. More exponent bits, wider range.
- The **mantissa width** sets the **precision** — how finely spaced two representable numbers are near a given magnitude. More mantissa bits, finer spacing.

These trade against each other because the total bit budget is fixed. Figure 2 lays out how each format spends its bits, and it is worth staring at, because the whole fp16-versus-bf16 drama is encoded in two columns.

![matrix comparing the sign exponent mantissa and range fields of fp32 fp16 bf16 and the two fp8 formats](/imgs/blogs/mixed-precision-at-scale-2.webp)

**fp32** is 1 sign, 8 exponent, 23 mantissa. Its 8-bit exponent (bias 127) gives a range from about $1.2\times 10^{-38}$ to $3.4\times 10^{38}$, and its 23-bit mantissa gives roughly 7 decimal digits of precision. This is the reference: everything else is measured as a loss relative to fp32.

**fp16** (IEEE half) is 1 sign, 5 exponent, 10 mantissa. It keeps a generous 10 mantissa bits — actually *more* effective precision than bf16 near 1.0 — but its 5-bit exponent (bias 15) is narrow. The largest finite fp16 is 65504. The smallest *normal* fp16 is $2^{-14} \approx 6.1\times 10^{-5}$. Below that it has subnormals down to $2^{-24} \approx 5.96\times 10^{-8}$, and below *that*, it is zero. That floor is the villain of this entire post.

**bf16** (bfloat16, Google's brain-float) is 1 sign, 8 exponent, 7 mantissa. The key move: it keeps fp32's *exact* 8-bit exponent, so it has fp32's *exact* dynamic range — the same $10^{\pm 38}$ span. It pays for that by dropping to 7 mantissa bits, so it has only 2–3 decimal digits of precision. A bf16 number is fp32 with the bottom 16 mantissa bits chopped off. That is literally the conversion.

Before we use these facts, let us make the two axes exact, because both underflow (a range failure) and swamping (a precision failure) fall straight out of one formula. For a normalized value in the binade $[2^{e-\text{bias}}, 2^{e-\text{bias}+1})$, the gap between two adjacent representable numbers — one *unit in the last place*, or ulp — is

$$\text{ulp} = 2^{\,e - \text{bias} - t}$$

where $t$ is the number of mantissa bits. Two consequences, one per axis. **Range** is set by the smallest and largest $e$: with a $b$-bit exponent the smallest normal magnitude is roughly $2^{1 - \text{bias}}$ and the largest is roughly $2^{2^{b} - 1 - \text{bias}}$, so more exponent bits push both ends out. **Precision** is set by $t$: near magnitude 1.0 (where $e = \text{bias}$) the spacing is exactly $2^{-t}$, so fp32 spaces values at $2^{-23} \approx 1.2\times 10^{-7}$, fp16 at $2^{-10} \approx 9.8\times 10^{-4}$, and bf16 at $2^{-7} \approx 7.8\times 10^{-3}$. Hold onto that last number — bf16's coarse $7.8\times 10^{-3}$ spacing near 1.0 is the entire reason master weights exist, two sections from now.

Now the derivation that explains the frozen run. Consider a gradient of magnitude $2^{-27}$ — small, but a perfectly ordinary value deep in a network where activations and gradients shrink layer by layer. In bf16, the exponent field can represent $2^{-27}$ directly (its range goes to $2^{-126}$), so the value is stored as roughly $2^{-27}$ with a couple digits of precision. Fine. In fp16, $2^{-27}$ is below the subnormal floor of $2^{-24}$ — there is no exponent bits configuration that reaches it — so it rounds to exactly `0.0`. The gradient does not get small; it *disappears*.

Multiply that across a whole layer's worth of small gradients and you get the pathology: the backward pass produces gradients that are mathematically nonzero but numerically zero in fp16, the optimizer receives all-zeros, the weights do not move, and the loss sits flat while the GPUs run at 98% doing arithmetic on zeros. That is exactly the `grad-norm 0.0000` from the intro. It is not a bug in the code. It is the 5-bit exponent doing precisely what 5 bits can do.

bf16 does not have this problem, full stop, because its 8-bit exponent reaches the same tiny magnitudes fp32 does. This is why **bf16 is the default for LLM training** and has been since the A100 made it fast: same range as fp32, so small gradients survive, so no rescue machinery is needed. The 7-bit mantissa is a real precision cost, but as we will see, the master-weight trick handles it. fp16's problem is range, and range is the one thing you cannot buy back cheaply — so fp16 needs a trick of its own, which is where we go next.

#### Worked example: reading a dtype off the hardware

You are handed a cluster and told to pick a training dtype. The single question that decides bf16 versus fp16 is: *does this GPU's Tensor Core support bf16?* Volta (V100) does not — it is fp16-only Tensor Cores, 125 TFLOP/s, no bf16 path. Ampere (A100), Hopper (H100), and Ada (L40S, RTX 40-series) all do. So the rule writes itself: on a V100 cluster you are on fp16 whether you like it or not, and you must add loss scaling. On anything A100 or newer you use bf16 and skip the scaler entirely. The dtype is not a preference; it is a property of the silicon, and reading it correctly saves you from either leaving the Tensor Core idle (fp32 on an A100) or fighting underflow you did not need to fight (fp16 on an A100).

## Loss scaling: teaching fp16 to keep small gradients

If your only problem with fp16 is that small gradients fall off the bottom of the range, the fix is almost embarrassingly direct: move them up. Before the backward pass, multiply the loss by a large constant $S$. By the chain rule, every gradient in the backward pass is then also multiplied by $S$ — a gradient that would have been $2^{-27}$ is now $2^{-27} \cdot 2^{15} = 2^{-12}$, comfortably inside fp16's representable range. Do the whole backward pass in fp16 with the inflated gradients, then, *before* the optimizer touches anything, divide the gradients back down by $S$ (unscale them) so the actual update magnitude is correct. The scale is a temporary lift that exists only to carry small numbers across the fp16 floor and is removed before it can affect the math. Figure 3 walks one step of this.

![timeline of a dynamic loss scaling step showing scale up backward unscale inf nan check and the step or skip decision](/imgs/blogs/mixed-precision-at-scale-3.webp)

There is one complication that turns a fixed constant into a small control system. Pick $S$ too small and the tiniest gradients still underflow; pick it too large and the *largest* gradients now overflow past 65504 to `inf`, which poisons everything downstream into `nan`. The safe range of $S$ shifts over training — early on, gradients are large and you want a modest scale; later they shrink and you want a bigger one. So production loss scaling is **dynamic**: start with a large $S$ (typically $2^{16}$), and after each backward:

1. Check the gradients for `inf` or `nan`.
2. If any are found, the scale was too high this step: **skip the optimizer step entirely** (do not apply a poisoned update) and **halve $S$**.
3. If the gradients are clean, apply the step, and if you have gone some number of consecutive clean steps (commonly 2000), **double $S$** to probe for more headroom.

This is a hill-climb on the scale factor: back off hard on overflow, creep up on success. It costs you the occasional skipped step early in training — a few wasted micro-batches while the scale settles — but it self-tunes to whatever the gradient magnitudes are doing without any manual babysitting. In PyTorch this entire dance is one object, `torch.cuda.amp.GradScaler`:

```python
import torch
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()                       # dynamic loss scaling, starts at 2**16

for step, (x, y) in enumerate(loader):
    x, y = x.cuda(), y.cuda()
    optimizer.zero_grad(set_to_none=True)

    with autocast(dtype=torch.float16):     # fp16 forward: matmuls on Tensor Cores
        logits = model(x)
        loss = loss_fn(logits, y)

    scaler.scale(loss).backward()           # loss * S, so grads come out * S

    scaler.unscale_(optimizer)              # grads / S, back to true scale
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip on true grads

    scaler.step(optimizer)                  # skips the step if grads held inf/nan
    scaler.update()                         # halve S on overflow, else creep up
```

Two details in that snippet trip people up. First, `unscale_` before `clip_grad_norm_`: gradient clipping only makes sense on the *true* gradients, so you must divide out $S$ before you clip, not after. If you clip the scaled gradients you are clipping to a threshold that is $S$ times too large and the clip does nothing. Second, `scaler.step` is not a normal `optimizer.step` — it inspects the gradients first and silently skips the update if it finds `inf`/`nan`, which is exactly the "skip step" branch of Figure 3. If you call `optimizer.step()` directly you defeat the whole mechanism and apply the poisoned update.

There is a third trap that only appears once you combine loss scaling with gradient accumulation — accumulating gradients over several micro-batches before one optimizer step, which is how you get a large effective batch on limited memory. The rule is: **call `scaler.scale(loss).backward()` on every micro-batch, but call `scaler.step` and `scaler.update` only once per optimizer step.** The scale factor $S$ must be the *same* across all the micro-batches that accumulate into one step — if $S$ changed mid-accumulation, the partial gradients would be scaled by different factors and their sum would be meaningless. `GradScaler` handles this correctly as long as you only call `update()` at the step boundary, not per micro-batch. And in DDP, wrap the non-final micro-batches in `model.no_sync()` so the expensive all-reduce fires only on the last one — the loss scaling and the `no_sync()` accumulation path compose cleanly, but only if `update()` stays outside the accumulation loop. Get this wrong and you see a run that trains but noticeably worse than the same effective batch without accumulation, which is a miserable bug to chase because nothing errors.

### Why bf16 does not need any of this

Everything in the last section — the scaler, the skipped steps, the dynamic hill-climb — exists to compensate for fp16's narrow exponent. bf16 has fp32's exponent, so there is no floor to carry gradients across, so there is nothing to scale. You run bf16 with a plain `optimizer.step()` and no `GradScaler` at all:

```python
for step, (x, y) in enumerate(loader):
    x, y = x.cuda(), y.cuda()
    optimizer.zero_grad(set_to_none=True)

    with autocast(dtype=torch.bfloat16):    # bf16 forward, no scaler anywhere
        logits = model(x)
        loss = loss_fn(logits, y)

    loss.backward()                         # grads are already in a good range
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

One thing `autocast` does that is easy to overlook: it does *not* cast every operation to 16-bit. It maintains an internal policy list. Matmuls, linear layers, and convolutions — the Tensor Core ops that benefit and are numerically robust — run in the autocast dtype. But operations that are numerically delicate in low precision are kept in fp32 automatically: softmax, layernorm and batchnorm, the loss function, and reductions like `sum`, `exp`, `log`, and `norm`. This is deliberate — a softmax over a long sequence in bf16 can lose too much precision in the exponentials and the normalizing sum, so autocast runs it in fp32 even inside a bf16 region. You get the matmul speedup where it is safe and fp32 stability where it matters, without annotating a single op yourself. The practical upshot: do not manually `.bfloat16()` your layernorms or your loss to "be consistent" — you would be overriding a correct default with a worse one. Let `autocast` decide; it already keeps the fragile ops in fp32, which is the same instinct as the fp32 accumulate, the fp32 reduce, and the fp32 master, applied one more time.

That difference — one code path with a stateful scaler and skipped steps, versus one plain path — is a real operational reason bf16 is preferred beyond the numerics. There is less to misconfigure. The `unscale_`-before-clip ordering bug does not exist because there is no unscale. The "my loss is fine but grad-norm is `nan` every few hundred steps" support ticket does not happen. When people say bf16 is "more robust" they mostly mean *there is less machinery to get wrong*.

#### Worked example: fp16 without and with the scaler

Take the 7B run from the intro on a V100 cluster (fp16-only, so we cannot dodge into bf16). Train for 500 steps two ways.

**Without loss scaling.** By step 40 the deepest layers' gradients have shrunk below $2^{-24}$. They flush to zero. The affected weights stop updating. Loss, which had been dropping from 8.0, stalls around 5.1 and stays there. grad-norm for the frozen layers reads `0.0`. Throughput is a healthy 41k tokens/s — the GPUs are busy — but the model is not learning. This is the silent failure: nothing crashes, the dashboard looks alive, and you have wasted every GPU-hour since step 40.

**With `GradScaler`.** Same run, `scaler.scale(loss).backward()`. The scale settles to $2^{17}$ within the first few hundred steps after a handful of skipped-step overflows early on. The $2^{-27}$ gradients are now stored as $2^{-10}$, well inside fp16, unscaled cleanly before the step. Loss drops past 5.1 and keeps going — 4.6, 4.1, 3.7 — on the same trajectory a bf16 run would follow. Throughput is 40.6k tokens/s, essentially identical; the scaler is nearly free. Figure 4 is this contrast on the single gradient that decides it.

![before and after diagram of one fp16 gradient underflowing to zero unscaled versus surviving after multiplying by two to the fifteenth](/imgs/blogs/mixed-precision-at-scale-4.webp)

The lesson is not "loss scaling is clever" — it is that on fp16 hardware it is not optional. The unscaled run and the scaled run differ by one wrapper, and one of them does not train at all.

## Master weights: the fp32 copy that makes updates stick

Loss scaling fixes fp16's range problem in the *backward* pass. There is a second, subtler numerics problem that afflicts *both* fp16 and bf16, on the *optimizer* side, and its fix is the fp32 **master weight** — the thing that actually makes "mixed precision" mixed.

Here is the failure. Suppose a weight has magnitude around 1.0 and the optimizer computes an update of magnitude $3\times 10^{-4}$ (a typical Adam step at learning rate $3\times 10^{-4}$). You want to do `w = w + update`. But if `w` is stored in bf16, the spacing between representable values near 1.0 is $2^{-7} \approx 7.8\times 10^{-3}$ — bf16's 7 mantissa bits. Your update of $3\times 10^{-4}$ is more than twenty times *smaller* than the gap to the next representable bf16 value. So `w + update` rounds right back to `w`. The update is real, it is correct, and it is completely swallowed by rounding. Do this every step and the weight never moves — the same frozen-loss symptom, arriving through a different door.

This is called **swamping**: a small number added to a larger one gets rounded away when the destination cannot represent the sum's precision. It is not about range this time — $3\times 10^{-4}$ is well within bf16's range — it is about the mantissa. And it hits bf16 *harder* than fp16 here, because bf16's 7 mantissa bits are coarser than fp16's 10. Range and precision are genuinely different axes, and this is the axis where bf16 is the weaker format.

The fix is to keep the weights in fp32 for the part of the step that needs precision — the accumulation — and use 16-bit only for the part that needs speed — the matmuls. Concretely, you keep a **master copy** of every weight in fp32. The optimizer states (Adam's first and second moments) also live in fp32. Each step:

1. Cast the fp32 master weights down to bf16 (a 16-bit *working* copy).
2. Run the forward and backward in bf16 on the Tensor Cores — fast.
3. Take the bf16 gradients, and apply the optimizer update to the **fp32 master**, not the bf16 copy. In fp32 the spacing near 1.0 is $2^{-23} \approx 1.2\times 10^{-7}$, so the $3\times 10^{-4}$ update lands with room to spare and *accumulates* correctly step after step.
4. Next step, cast the freshly-updated fp32 master back down to bf16 and repeat.

The tiny updates accumulate in the high-precision master over many steps and only *then*, once they have grown, show up in the bf16 working copy. That is the whole idea of mixed precision in one sentence: **store and update in fp32, compute in 16-bit.** The "mixed" is not fp16-versus-bf16; it is fp32-master-versus-16-bit-compute.

This is also where the mixed-precision story connects to the memory budget, because the master weight is not free. Figure 5 breaks down the per-parameter bytes for a mixed-precision Adam step.

![stack diagram showing sixteen bytes per parameter split into a four byte fp32 master two byte bf16 weight two byte bf16 grad and two four byte fp32 Adam moments](/imgs/blogs/mixed-precision-at-scale-5.webp)

Per parameter you are holding: the bf16 working weight (2 bytes), the bf16 gradient (2 bytes), the fp32 master weight (4 bytes), and Adam's fp32 first and second moments (4 bytes each). That is $2 + 2 + 4 + 4 + 4 = 16$ bytes per parameter. If you have read [the memory budget](/blog/machine-learning/distributed-training/the-memory-budget), you will recognize the famous $(2 + 2 + 12)\Psi$ formula — the "12" is exactly this trio of fp32 states: the master weight plus the two Adam moments, 4 bytes each. The fp32 master weight is not a separate line item you forgot; it is a third of the optimizer term that dominates large-model memory. This is why full sharding matters so much: for a 7B model, $16 \times 7\times 10^9 = 112$ GB of optimizer-and-weight state, which is why you shard it across ranks with [FSDP](/blog/machine-learning/distributed-training/fsdp-in-practice) rather than replicating all 112 GB on every GPU.

Here is the part that surprises people, and it is worth stating flatly because it corrects a common misconception. Line the two arrangements up side by side, per parameter:

| Tensor (per parameter) | Pure fp32 training | bf16 mixed precision |
| --- | --- | --- |
| Working weight (for compute) | 4 B | 2 B |
| Gradient | 4 B | 2 B |
| fp32 master weight | — (the weight *is* the master) | 4 B |
| Adam first moment (m) | 4 B | 4 B |
| Adam second moment (v) | 4 B | 4 B |
| **Total model state** | **16 B** | **16 B** |

They are *identical*. Mixed precision does **not** shrink your optimizer-and-weight state at all — the 2 bytes you save on the weight and 2 on the gradient are handed right back by the 4-byte fp32 master you had to add. Pure fp32 training keeps no separate master because the fp32 weight already is one. So where does mixed precision's memory win come from, if not here? Entirely from **activations** and **communication**. Activations — every intermediate tensor saved for the backward pass, usually the largest single memory consumer during a step — are stored in bf16 and halve. Gradient all-reduce buffers move bf16 and halve. Those are real, large savings. But if you were expecting bf16 to cut your optimizer state, it does not, and a plan that assumes it will is a plan that OOMs. The optimizer state is a *sharding* problem (FSDP/ZeRO), not a precision one — which is exactly why the two levers compose rather than substitute.

There is one clever way to skip the master weight and *still* not stall, worth knowing because it shows up in memory-tight optimizers: **stochastic rounding**. Ordinary (round-to-nearest) rounding of `w + update` always rounds a sub-ulp update to the same neighbor — usually back to `w` — so the update is lost every single time, deterministically. Stochastic rounding instead rounds up or down *at random*, with probability proportional to how close the true sum is to each neighbor. An update that is 10% of the way to the next bf16 value rounds up 10% of the time and stays put 90% of the time, so *in expectation* the weight moves by the right amount even though any single step either jumps a full ulp or does nothing. Over many steps the small updates accumulate statistically rather than being systematically discarded. This lets some low-memory optimizers keep weights in bf16 with no fp32 master — saving the 4 bytes — at the cost of extra gradient-estimate variance. It is a real technique (PyTorch exposes it in some fused optimizers; several 8-bit optimizers rely on it), but it is the exception. The default, robust answer is still the fp32 master; reach for stochastic rounding only when memory is so tight that the master's 4 bytes per parameter are the thing standing between you and fitting the run, and you have validated that the added variance does not hurt final loss.

The good news is that you almost never manage master weights by hand. Modern optimizers and wrappers keep the fp32 master for you. In FSDP the `MixedPrecision` policy handles the casts; in DeepSpeed ZeRO the fp32 master and moments live in the partitioned optimizer state; the `bitsandbytes` and `apex` fused optimizers keep an internal fp32 copy. When you write `autocast(dtype=torch.bfloat16)` and hand `model.parameters()` (which are fp32) to `torch.optim.AdamW`, you have *already* built the master-weight arrangement without thinking about it: the parameters stay fp32, autocast casts to bf16 only inside the compute region, and the optimizer updates the fp32 parameters. The trap is the opposite — someone calls `model.half()` or `model.bfloat16()`, which converts the *parameters themselves* to 16-bit permanently, destroying the fp32 master. Now the optimizer updates a bf16 weight directly, swamping kicks in, and the loss stalls. `model.half()` for training is almost always a bug; `autocast` is the correct tool because it leaves the master alone.

#### Worked example: where the fp32 master earns its 4 bytes

A 1.5B model, bf16 everywhere including the parameters (`model.bfloat16()`), Adam, learning rate $2\times 10^{-4}$ with a warmup. For the first few hundred steps, while gradients and the resulting updates are relatively large (>$10^{-2}$), it trains fine — the updates are big enough to clear bf16's spacing. Then warmup ends, the learning rate and the update magnitudes settle to their steady-state ~$10^{-4}$, and progress *slows to a crawl*: loss that was dropping by 0.05/step now drops by 0.002/step, then noise. The updates have shrunk below bf16's precision floor near the current weight magnitudes and are being rounded away. Switch to the standard arrangement — fp32 parameters, `autocast(bfloat16)` for compute — costing 4 extra bytes per parameter (6 GB for this model), and the same run trains smoothly through the whole schedule. Those 4 bytes are not overhead; they are the reason the last two-thirds of training does anything.

## Communication in low precision

There is one more place precision meets distributed training, and it is on the wire. When data-parallel ranks all-reduce their gradients, the dtype of that reduction is a separate choice from the compute dtype, and it is one of the more consequential knobs at scale.

The temptation is obvious: reduce gradients in bf16 and you halve the communication volume. A ring all-reduce moves $2(N-1)/N \cdot S$ bytes per GPU; if $S$ is bf16 instead of fp32, every gradient sync is twice as fast, and at 64 or 512 ranks where all-reduce can dominate the step, that is a large win. But an all-reduce is a *sum* across $N$ ranks, and summation is exactly the operation where low precision bites. Adding a small per-rank gradient contribution to a large running partial sum in bf16 — with only 7 mantissa bits — rounds the small contribution away (swamping again, now inside the collective). Across many ranks these dropped contributions compound into a gradient average that is subtly, systematically wrong. It does not crash. It produces a slightly-off update every step, and the run trains *almost* correctly and then diverges a few thousand steps in, for no reason visible in any single step.

So the standard, safe arrangement — the one FSDP's `MixedPrecision` policy expresses — separates the two dtypes:

```python
from torch.distributed.fsdp import MixedPrecision

policy = MixedPrecision(
    param_dtype=torch.bfloat16,     # cast params to bf16 for compute -> Tensor Cores
    reduce_dtype=torch.float32,     # reduce gradients in fp32 -> accurate sum
    buffer_dtype=torch.bfloat16,    # non-learnable buffers (e.g. norm stats) in bf16
)
```

`param_dtype=torch.bfloat16` gets you the compute speedup. `reduce_dtype=torch.float32` keeps the gradient sum accurate. You give up half the potential comms savings, but you keep the run stable, and stability is worth far more than a few percent of step time on a run that costs tens of thousands of dollars. Reduce in bf16 only when you have *measured* that your model tolerates it and you are genuinely comms-bound — and it is the first thing to revert when a large run starts drifting for no visible reason. (The FSDP-side details of this policy, including the `buffer_dtype` gotchas, are in [FSDP in practice](/blog/machine-learning/distributed-training/fsdp-in-practice).)

#### Worked example: does bf16 reduction actually help here?

A 7B model on 8 A100s over NVLink, gradient buffer ~14 GB in bf16. The ring all-reduce moves $2(N-1)/N \cdot S = 2 \cdot 7/8 \cdot 14 \approx 24.5$ GB per GPU; over NVLink3 at ~250 GB/s of achieved all-reduce bandwidth that is ~98 ms. Reducing in fp32 instead doubles $S$ to 28 GB and the time to ~196 ms — and if that all-reduce is fully overlapped behind the backward pass (which on a single NVLink node it usually is), the *extra 98 ms is hidden and costs you nothing*. So on one NVLink node, keep `reduce_dtype=torch.float32`; the accuracy is free. Now move to 8 nodes over InfiniBand HDR (~200 Gb/s ≈ 25 GB/s per link, far slower than NVLink), where the all-reduce no longer fully hides behind compute and becomes the step's critical path. There, halving $S$ with a bf16 reduction genuinely cuts step time — but only *there*, and only after you have confirmed the loss does not drift. The decision is not "bf16 reduce is faster"; it is "bf16 reduce is faster *only when the reduction is exposed on the critical path*, which is a multi-node-over-IB condition, not a single-node one." Measure which regime you are in before you trade accuracy for it.

### The four dtypes you are actually choosing

By now a pattern has emerged that is worth naming explicitly, because most mixed-precision confusion comes from collapsing four distinct choices into one word, "precision." A training step has four separate dtype decisions, and they should generally *not* all be the same:

1. **Storage dtype** — how the master weights and optimizer state are kept between steps. **fp32.** This is where tiny updates accumulate; it is the master weight. Never shrink it (that is swamping).
2. **Compute dtype** — what the matmuls' *inputs* are cast to. **bf16 (or fp16, or fp8).** This is where the Tensor Core speedup and the activation-memory savings come from. This is the one `autocast` sets.
3. **Accumulate dtype** — what the matmul sums its partial products in. **fp32**, in hardware, whether you asked or not. You do not control this and you would not want to.
4. **Communication dtype** — what gradients are reduced across ranks in. **fp32 by default, bf16 if measured-safe and comms-bound.** This is FSDP's `reduce_dtype`.

The healthy default is fp32 storage, bf16 compute, fp32 accumulate, fp32 reduce — three of the four in high precision, and only the one that touches the biggest, most bandwidth-hungry tensors (the matmul inputs and activations) in low precision. When someone says a run is "in bf16," ask *which* of these four they mean; the answer to "why did it diverge" is almost always that one of the other three got dragged down to 16-bit when it should not have been. `model.half()` is exactly the bug of collapsing storage down to compute. A bf16 all-reduce is collapsing communication down to compute. The whole art is keeping these four decoupled.

## Measuring the speedup honestly

Every number in this post — the 2x, the 1.3x, the tokens/s — is only as trustworthy as the way it was measured, and low-precision timing is unusually easy to get wrong. GPU kernels are asynchronous: `loss.backward()` returns to Python long before the GPU has finished the work, so a naive `time.time()` around it measures Python dispatch latency, not compute. You must synchronize.

```python
import torch, time

def timed_steps(model, batch, optimizer, scaler=None, dtype=torch.bfloat16,
                warmup=20, iters=100):
    x, y = batch
    # 1. WARM UP: first steps pay cuDNN autotuning, allocator growth, and
    #    (for fp16) the loss-scale settling. Never time these.
    for _ in range(warmup):
        _one_step(model, x, y, optimizer, scaler, dtype)

    torch.cuda.synchronize()          # 2. drain the queue before starting the clock
    t0 = time.perf_counter()
    for _ in range(iters):
        _one_step(model, x, y, optimizer, scaler, dtype)
    torch.cuda.synchronize()          # 3. drain again before stopping it
    dt = (time.perf_counter() - t0) / iters

    tokens = x.numel()                # tokens per step (batch * seq)
    return tokens / dt                # tokens/s, steady state
```

The three things that silently corrupt a precision benchmark, in order of how often they bite:

- **No warm-up.** The first 10–20 steps pay one-time costs: cuDNN/cuBLAS autotuning picks kernels, the caching allocator grows to its working size, and for fp16 the dynamic loss scale is still hunting for its value (with skipped steps that look like stalls). Time those and bf16 looks slower than it is. Always discard the warm-up.
- **No `synchronize()`.** Without the two `torch.cuda.synchronize()` calls you are timing the launch of the kernels, not their execution, and you will "measure" fp8 as infinitely fast. This is the single most common bogus-speedup bug.
- **The data-loader confound.** If the loader cannot keep the GPU fed, both bf16 and fp32 run at the loader's speed and the speedup vanishes — not because 16-bit did not help but because the GPU was starving either way. Time with data already resident on the GPU (as above) to isolate the compute, then separately confirm your real loader is not the bottleneck. And watch for **clock throttling**: a bf16 run at 2x the FLOPs draws more power and heat, and a thermally throttled card quietly clocks down, understating the win. Log `nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw` alongside your timings.

With that methodology, here is a representative before→after on named hardware — a 7B dense transformer, sequence length 2048, FSDP `FULL_SHARD`, on 8 H100 SXM. Treat the absolute tokens/s as approximate and the *ratios* as the real result:

| Compute dtype | Tensor Core peak (dense) | Step time vs bf16 | Peak mem/GPU | Loss scaling |
| --- | --- | --- | --- | --- |
| TF32 | ~495 TFLOP/s | ~2.0x slower | highest (fp32 activations) | none |
| fp16 | ~990 TFLOP/s | ~1.0x (matches bf16) | baseline | **required** (GradScaler) |
| bf16 | ~990 TFLOP/s | 1.0x (baseline) | baseline | not needed |
| fp8 (E4M3/E5M2) | ~1979 TFLOP/s | ~0.7–0.75x (1.3–1.4x faster) | ~= bf16 params | per-tensor |

Two honest reads of that table. First, bf16 and fp16 have the *same* Tensor Core peak and land at the same step time — the choice between them is entirely about stability (range and loss scaling), not speed. Second, fp8's ~1.3–1.4x end-to-end is well short of its 2x peak ratio, because only the big matmuls go faster while the memory-bound kernels, reductions, and optimizer do not — a perfect illustration of Amdahl's law on the precision axis.

## fp8: E4M3, E5M2, and the Transformer Engine

If 16-bit doubles Tensor Core throughput over fp32, 8-bit doubles it *again* over 16-bit on Hopper: ~1,979 fp8 TFLOP/s versus ~990 bf16 TFLOP/s on an H100 SXM. That is the entire reason fp8 exists. But 8 bits is a savage bit budget — there is no format that has both usable range and usable precision in 8 bits — so fp8 is not one format but two, and it is used far more surgically than bf16. Figure 6 is the shape of an fp8 training step.

![graph of an fp8 training step with delayed scaling feeding both the forward E4M3 and backward E5M2 matmuls which merge into fp32 accumulation reduction and master weight](/imgs/blogs/mixed-precision-at-scale-6.webp)

The two formats split the eight bits differently for two different jobs:

- **E4M3** — 4 exponent bits, 3 mantissa bits, max magnitude ~448. More mantissa, less range. Used for the **forward pass**: weights and activations, which are relatively well-behaved in magnitude and benefit from the extra precision bit.
- **E5M2** — 5 exponent bits, 2 mantissa bits, max magnitude ~57344. More range, less precision. Used for **gradients** in the backward pass, which have a wider dynamic range and need the exponent bits more than the mantissa (the same range argument that favored bf16, now inside 8 bits).

Neither format has anything like fp16's or bf16's precision, so fp8 cannot just be dropped in where you use bf16. Two things keep it stable, both visible in Figure 6. First, **scaling is per-tensor (or per-block), not global**. Each tensor gets its own scale factor chosen so that its values land in the sweet spot of E4M3's tiny range — this is loss scaling's idea generalized to every tensor. NVIDIA's **Transformer Engine** on Hopper does this with **delayed scaling**: it keeps a rolling history of the maximum absolute value (`amax`) seen in each tensor over recent steps and uses that history to pick the next step's scale, so it does not have to compute the max synchronously in the hot path. Second — and this is the part people miss — **only the matmul inputs are fp8**. The accumulation inside the Tensor Core is fp32. The gradient reduction across ranks is bf16 or fp32. The master weight and optimizer states are fp32. fp8 accelerates the two big matmuls (the forward and the backward-input product) and *nothing else*. Reductions, accumulations, softmaxes, layernorms, and the optimizer all stay in higher precision, because those are the operations where 2 or 3 mantissa bits would destroy the run.

In code with Transformer Engine, you swap `nn.Linear` for `te.Linear` and wrap the forward in an `fp8_autocast` context that carries the recipe:

```python
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format

fp8_recipe = DelayedScaling(
    fp8_format=Format.HYBRID,   # E4M3 in forward, E5M2 for gradients
    amax_history_len=16,        # rolling window of maxima per tensor
    amax_compute_algo="max",
)

model = te.TransformerLayer(hidden_size=4096, ffn_hidden_size=16384,
                            num_attention_heads=32)  # te layers, fp8-capable

for x, y in loader:
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        out = model(x)          # the big matmuls run in fp8, accumulate in fp32
        loss = loss_fn(out, y)
    loss.backward()             # gradients: E5M2; reduce + optimizer stay higher precision
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

Why *delayed* scaling, and not just compute each tensor's max and scale by it every step? Because computing an exact per-tensor `amax` requires a full reduction over the tensor *before* you can launch the matmul, which serializes a slow pass in front of the fast one and eats much of the fp8 win. Delayed scaling breaks that dependency: it assumes the tensor's magnitude does not jump wildly between adjacent steps (usually true in stable training), so it reuses a scale derived from the last several steps' maxima and updates the history asynchronously. The cost is a rare miss — if a tensor's magnitude spikes suddenly, the stale scale can overflow E4M3's max of 448 for one step — which is why the recipe also tracks whether overflows are happening and can fall back. For models where per-tensor scaling is too coarse (a single scale cannot cover the magnitude spread within one big weight matrix), newer recipes go **finer-grained**: per-block or per-tile scaling, where sub-blocks of a matrix each get their own scale. DeepSeek-V3's fp8 training used exactly this — per-tile 1x128 and per-block 128x128 scaling — precisely because one scale per tensor left too many values either overflowing or underflowing E4M3's narrow range. Finer scaling is more robust and more expensive to manage; it is the direction fp8 is maturing in.

Be honest about where fp8 pays and where it does not. Concretely, fp8 does **not** accelerate, and must **not** be applied to: the fp32 accumulation inside the matmul; the gradient all-reduce (bf16/fp32); the optimizer step and master weights (fp32); the softmax and layernorm reductions; and anything memory-bound rather than matmul-bound. It accelerates the two large GEMMs — the forward `activation x weight` and the backward `gradient x activation` — and that is the entire mechanism. It pays on the **largest matmuls** — big hidden dimensions, big batch — where the matmul dominates the step and doubling its rate moves the needle. It does *not* help memory-bound kernels (those are already not matmul-limited), it does *not* help small matmuls (the per-tensor scaling overhead eats the gain), and it is *finickier* than bf16: get a scale wrong and a tensor overflows E4M3's max of 448 into garbage, or underflows and loses all its bits. fp8 is newer and less forgiving, and validating that an fp8 run matches a bf16 baseline in final loss is real work you should budget for. It is not a free 2x — it is a 2x on the matmul-heavy fraction of the largest models, gated behind careful scaling and validation.

#### Worked example: fp8 on one big matmul

Take a single fully-connected layer from a large model: hidden 8192, FFN 28672, batch-times-sequence of 16384 tokens — the kind of matmul that dominates a step in a 70B-class model. On an H100 SXM, in bf16 this matmul runs near the card's ~990 TFLOP/s. In fp8 through Transformer Engine, with per-tensor delayed scaling, it runs near ~1,600 TFLOP/s achieved (you rarely hit the full 1,979 dense peak because of the scaling and cast overhead) — call it a 1.6x speedup *on that matmul*. Because such matmuls are 60–70% of the step in a large dense transformer, the end-to-end step speedup lands around 1.3–1.4x over bf16. The accuracy caveat is concrete: on a from-scratch pretraining run you must confirm the fp8 loss curve tracks the bf16 curve within noise for the first few thousand steps before you trust it for the full run. The 1.3x is real; so is the validation cost, and honest accounting includes both.

## Case studies: what real runs used

The numerics above are not theoretical — they map onto specific, published large-model runs.

**GPT-3 and early Megatron: fp16 with dynamic loss scaling.** Before bf16 Tensor Cores existed at scale (pre-A100), the large-model runs were fp16, and they used exactly the dynamic loss-scaling machinery described here — start at $2^{16}$, halve on overflow, skip the poisoned step. The `GradScaler`/`apex` `amp` recipe is a direct descendant of what those runs needed. If you are on V100s today, you are in this world.

**OPT-175B: the fp16 instability logbook.** Meta's OPT-175B was trained in fp16, and its publicly released training logbook is one of the best real documents on what fp16 at scale actually feels like: repeated loss spikes and divergences, manual restarts from earlier checkpoints, learning-rate and loss-scale adjustments to fight instability. It is the empirical case for why the field moved to bf16 — not because fp16 *cannot* train a 175B model, but because keeping it stable consumed engineer-weeks that bf16 would have given back.

**PaLM, LLaMA, and the bf16 default.** Once A100/TPU bf16 was fast, the large runs standardized on it. PaLM, the LLaMA family, and essentially every open LLM since train in bf16 precisely for the reason this post derives: fp32 range means no loss scaling, no skipped steps, no scale-factor babysitting. bf16 became the default not by fashion but because it removed an entire category of failure.

**DeepSeek-V3: fp8 at frontier scale.** The most prominent recent fp8-trained large model, DeepSeek-V3 (late 2024), demonstrated fp8 mixed-precision training on a 600B-parameter-class MoE with fine-grained scaling — per-tile and per-block quantization rather than a single per-tensor scale — precisely to keep E4M3's tiny range usable across the varying magnitudes inside a large model. It is the clearest published evidence that fp8 *can* train a frontier model, and also, in its detailed description of the scaling scheme required, the clearest evidence that fp8 is not a drop-in — the fine-grained scaling is the price of admission.

The through-line: the format follows the hardware and the risk tolerance. fp16 where you must, bf16 as the default the moment the hardware allows, fp8 at the frontier where the matmul savings justify the validation cost.

## When to reach for each (and when not to)

Precision is a lever with a cost, like every lever in this series. Here is the decision, made plainly. Figure 7 is the same call as a matrix.

![matrix comparing fp32 bf16 fp16 and fp8 across range precision speed loss scaling need and when to use each](/imgs/blogs/mixed-precision-at-scale-7.webp)

**Use bf16 by default.** On any A100, H100, L40S, or newer, bf16 is the right answer for training from scratch: full fp32 range so small gradients survive, ~2x Tensor Core throughput, half the activation memory and gradient bytes, and *no loss scaling* — a plain `optimizer.step()`. The only thing bf16 gives up is mantissa precision, and the fp32 master weight handles that. This is not a close call; it is the default for a reason.

**Use fp16 only when you must.** If the hardware is Volta (V100) or otherwise lacks bf16 Tensor Cores, or you are resuming a checkpoint that was trained in fp16, you are on fp16 — and then you *must* add `GradScaler` with dynamic loss scaling, and you must get the `unscale_`-before-clip ordering right. fp16 has one genuine advantage — 10 mantissa bits versus bf16's 7, so slightly finer precision — which occasionally matters for inference or specific numerically-sensitive layers, but for from-scratch training its narrow exponent makes it strictly more fragile.

**Use fp8 for the largest runs, with validation budget.** On H100/H200 (or newer fp8-capable silicon), for models large enough that big matmuls dominate the step, fp8 through Transformer Engine buys another ~1.3–1.4x end-to-end over bf16. Reach for it when you are training something big enough and long enough that a 1.3x is worth a validation effort, you have the hardware, and you can afford to confirm the fp8 loss tracks a bf16 baseline. Do *not* reach for it on small models, memory-bound workloads, or when you cannot spare the engineering to validate it — the 2x is on the matmul fraction only, and the finickiness is real.

**Never train with `model.half()` or `model.bfloat16()` on the parameters.** That destroys the fp32 master and reintroduces swamping. Use `autocast` (or FSDP `MixedPrecision`), which casts only inside the compute region and leaves the master fp32.

**Reduce gradients in fp32 unless you have measured otherwise.** `reduce_dtype=torch.float32` keeps the cross-rank sum accurate; bf16 reduction halves comms but risks slow drift at scale. It is the first knob to revert when a big run diverges for no visible reason.

The stress tests sharpen the recommendation. *At 64 GPUs where all-reduce dominates?* Keep `reduce_dtype=torch.float32` anyway — the drift from bf16 reduction gets worse with more ranks, not better, because more partial sums means more swamping. *On a tiny batch?* fp8's per-tensor scaling overhead is a bigger fraction of a small matmul, so the fp8 win shrinks or vanishes — stay bf16. *Resuming an fp16 checkpoint on an H100?* You can convert to bf16, but validate that the converted weights and optimizer state produce a matching loss for a few hundred steps before committing the run. *When the optimizer state will not fit?* The fp32 master and moments are 12 of the 16 bytes per parameter — that is a *sharding* problem for FSDP/ZeRO, not a precision problem, and dropping the master to bf16 to save memory just reintroduces swamping. Shard it; do not shrink it.

## Key takeaways

- **16-bit is the second-biggest lever after parallelism.** ~2x Tensor Core throughput, half the activation memory, half the gradient comms. Not optional at scale — but *which* 16-bit and *what machinery* you add decides whether it trains or silently freezes.
- **The formats trade range against precision.** Exponent bits buy range, mantissa bits buy precision, the bit budget is fixed. fp16 (5-bit exponent) has narrow range and underflows; bf16 (8-bit exponent, same as fp32) has full range and does not.
- **fp16's small gradients flush to zero below $2^{-24}$.** That is the frozen-loss, `grad-norm 0.0000` failure. It is the exponent width, not a bug.
- **Loss scaling rescues fp16:** multiply the loss by $S$ before backward to lift gradients into range, unscale before the step, and let dynamic scaling hill-climb $S$ (halve on `inf`/`nan`, creep up on clean steps). `GradScaler` is the whole thing; `unscale_` before you clip.
- **bf16 needs no loss scaling** because its exponent matches fp32 — one plain code path, less to misconfigure. This is a real reason it is the default, on top of the numerics.
- **Master weights are the "mixed" in mixed precision:** keep an fp32 copy of the weights (and fp32 Adam moments) so tiny $\sim 10^{-4}$ updates accumulate instead of being swamped by 16-bit rounding. `autocast` builds this for you; `model.half()` destroys it.
- **The fp32 master is 4 of the 12 optimizer bytes** in $(2+2+12)\Psi$ — shard it with FSDP/ZeRO, never shrink it.
- **Reduce gradients in fp32** to keep the cross-rank sum accurate; bf16 reduction halves comms but drifts, worse at more ranks.
- **fp8 doubles Tensor Core throughput again on H100** — E4M3 forward, E5M2 gradients, per-tensor delayed scaling, fp32 accumulation and reduction. It pays on the largest matmuls, is finickier than bf16, and needs a validation budget.
- **bf16 by default, fp16 only on hardware without bf16, fp8 for the largest models you can validate.** The format follows the silicon and the risk tolerance.

## Further reading

- [The memory budget](/blog/machine-learning/distributed-training/the-memory-budget) — the $(2+2+12)\Psi$ formula in full; the fp32 master weight is 4 of the 12 optimizer bytes derived here.
- [FSDP in practice](/blog/machine-learning/distributed-training/fsdp-in-practice) — the `MixedPrecision(param_dtype, reduce_dtype, buffer_dtype)` policy and the buffer-dtype gotchas, applied to sharded training.
- [Collectives from scratch](/blog/machine-learning/distributed-training/collectives-from-scratch) — the $2(N-1)/N \cdot S$ all-reduce byte volume that low-precision gradients halve.
- [Why distributed training](/blog/machine-learning/distributed-training/why-distributed-training) — the four walls; mixed precision is the lever for the run-too-slow and model-won't-fit walls at once.
- [The distributed training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) — the capstone decision and debugging checklist that ties precision into the whole scaling story.
- [Mixed-precision debugging: fp16 vs bf16](/blog/machine-learning/debugging-training/mixed-precision-debugging-fp16-vs-bf16) — the debugging companion for when a low-precision run NaNs, stalls, or drifts.
- Micikevicius et al., *Mixed Precision Training* (2018) — the original loss-scaling and fp32-master-weight paper; Micikevicius et al., *FP8 Formats for Deep Learning* (2022) — the E4M3/E5M2 specification. The NVIDIA [Transformer Engine documentation](https://docs.nvidia.com/deeplearning/transformer-engine/) for the delayed-scaling recipe, and the OPT-175B training logbook for the fp16-at-scale reality.

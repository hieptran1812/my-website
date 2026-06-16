---
title: "Hunting NaNs and Infs: A Systematic Method"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "A systematic method for localizing the exact operation that produces the first NaN in a training run, with the numerics behind each source and the guardrails that catch it at step 1 instead of step 4000."
tags:
  [
    "debugging",
    "model-training",
    "nan",
    "numerics",
    "mixed-precision",
    "finetuning",
    "deep-learning",
    "pytorch",
    "llm",
    "computer-vision",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/hunting-nans-and-infs-1.png"
---

You babysat the run for three hours. The loss curve was descending like a textbook example, the GPUs were pinned at ninety percent utilization, and you finally walked away to get coffee. When you came back, every panel on the dashboard had flatlined at the same horizontal value, the loss field read `nan`, and the model that had been a perfectly good ResNet at step 3,999 was, at step 4,001, a tensor of `nan` from the first convolution to the last linear layer. Nothing crashed. There was no stack trace. The run is technically still going, dutifully computing `nan` minus learning-rate times `nan` and writing `nan` to your checkpoint. You have a NaN at step 4,000, and you have no idea where it came from.

This is the most frustrating class of training bug because it is simultaneously catastrophic and silent. A NaN does not raise an exception in PyTorch by default; it propagates. One `nan` in one logit becomes one `nan` in the loss, which becomes a gradient full of `nan`, which the optimizer faithfully writes into every parameter on the next step, and now your entire model is dead. The poison spreads at the speed of one forward-backward pass, and by the time you notice the flat line on the dashboard, the original sin — the single operation that produced the first non-finite value — is thousands of steps in the past and buried under a model that is now uniformly `nan`. You cannot inspect a `nan` weight and learn where the NaN started any more than you can inspect a pile of ash and learn which match lit the fire.

![A dataflow graph showing five common operations each producing inf or NaN from a forbidden input, all feeding into one poisoned tensor that becomes a NaN loss at step 4000](/imgs/blogs/hunting-nans-and-infs-1.png)

The thesis of this post is that hunting a NaN is not luck or black magic — it is a **systematic bisection**, exactly like every other bug in this series. A NaN lives in the *numerics* place of the six places a training bug can hide — data, optimization, model code, numerics, systems, evaluation — and the numerics place has a small, finite, fully-understood set of sources. Every NaN and Inf in a neural network traces to one of about seven operations applied to a forbidden input: a logarithm of zero or a negative, a division by zero, an exponential that overflows, a square root of a negative, a `0 * inf`, a bad label or out-of-range index, or fp16 overflow and underflow. That is the entire list. Once you know the list, hunting becomes mechanical: you bisect by step to find *when* the first NaN appears, you bisect by layer to find *where*, and you read the offending operation directly off an anomaly backtrace or an `isfinite` assertion. Then you fix the math — and, more importantly, you install a guardrail so the next NaN announces itself at step 1 with the exact tensor named, instead of corrupting a run you discover dead at step 4,000.

By the end of this post you will be able to take any NaN-ing run — in vision, in an LLM finetune, in a speech model, in tabular gradient boosting's neural cousins — and localize the originating operation in minutes. You will know the numerics behind each source well enough to predict its signature before you instrument anything (a smooth-then-NaN curve is almost always overflow or `log(0)`; an immediate step-1 NaN is almost always a data or masking bug). You will have copy-and-run code for `torch.autograd.set_detect_anomaly`, for finite-checking forward hooks, for bisection-by-step, and for the numerically-stable loss formulations that make most of these bugs impossible in the first place. Let us start with where NaNs actually come from, because you cannot hunt what you do not understand.

## 1. What a NaN actually is, and why it spreads

Before we hunt, we need to be precise about the prey. A NaN — "Not a Number" — is a specific bit pattern in the IEEE 754 floating-point standard, the same standard that governs fp32, fp16, and (mostly) bf16. IEEE 754 reserves the maximum exponent with a non-zero mantissa to mean "this is not a real number," and it reserves the maximum exponent with a zero mantissa to mean $\pm\infty$. These are not error states the hardware traps on; they are *valid floating-point values* with defined arithmetic. That is the whole problem. When your GPU computes `0.0 / 0.0`, it does not halt; it returns a NaN and keeps going, because by the standard that is the correct behavior.

The arithmetic of these special values is what makes them spread. The rules are simple and merciless:

- Any arithmetic operation with a NaN operand returns NaN. `NaN + 1 = NaN`, `NaN * 0 = NaN`, `0 * NaN = NaN`, `max(NaN, 5) = NaN` in most implementations. There is no operation that "cleans" a NaN by accident.
- `inf` arithmetic produces `inf` until it meets a contradiction, at which point it produces NaN. `inf + 1 = inf`, `inf * 2 = inf`, but `inf - inf = NaN`, `inf / inf = NaN`, and `0 * inf = NaN`. This last one, `0 * inf = NaN`, is the bridge by which an infinity (often a "harmless" large logit) silently becomes a NaN.
- Every comparison with NaN is false, including `NaN == NaN`. This is the trick you use to *detect* a NaN: `x != x` is true if and only if `x` is NaN.

Now lift that to a tensor. Consider a matrix multiply, the single most common operation in a neural network. The output element $C_{ij} = \sum_k A_{ik} B_{kj}$ is a sum of products. If even one $A_{ik}$ is NaN, then one term in that sum is NaN, so the entire sum is NaN, so $C_{ij}$ is NaN. And $A_{ik}$ participates in computing $C_{ij}$ for *every* column $j$, so a single NaN in one entry of $A$ poisons an entire row of $C$. One matmul later, that row has spread to interact with more entries, and within two or three layers the entire activation tensor is NaN. This is not a bug in PyTorch; it is the arithmetic working exactly as specified.

![A dataflow graph tracing one NaN in a single logit through a matmul into all activations, then through the backward pass into the gradient and the optimizer step into every weight](/imgs/blogs/hunting-nans-and-infs-8.png)

The backward pass is even more efficient at spreading the poison. Gradients are computed by the chain rule, multiplying local Jacobians. If the loss is NaN, then $\partial L / \partial \theta$ is NaN for essentially every parameter $\theta$, because the NaN loss flows backward through every path. So a single NaN loss produces a *fully* NaN gradient in one backward pass. Then the optimizer applies the update $\theta \leftarrow \theta - \eta \cdot g$, and since $g$ is NaN, every parameter becomes $\theta - \eta \cdot \text{NaN} = \text{NaN}$. After exactly one optimizer step following the first NaN loss, your entire parameter tensor is NaN. The model is dead, and it stays dead — a NaN weight produces NaN activations forever, so the loss is NaN on every subsequent step.

This is why the dashboard flatlines. The loss is not "stuck"; it is literally the IEEE NaN value, and the plotting library either draws it at a fixed position or drops the point. It is also why you cannot debug from the corpse. By the time you see the flat line, you are typically a full step past the first NaN loss, which means every parameter is already NaN, which means the forward pass from now on is uniformly NaN and tells you nothing about which *original* operation misbehaved. **The entire art of hunting a NaN is getting to the scene of the crime before the poison spreads** — catching the first non-finite value at the operation that produced it, ideally at step 1 of a guarded run rather than step 4,000 of an unguarded one.

A note on Inf versus NaN, because they have different signatures and different fixes. An `inf` is a number that grew past the representable range — `exp` of a large value, a division by a tiny value, an overflow in fp16. It is "directional": it remembers whether it is $+\infty$ or $-\infty$, and `inf` arithmetic stays finite-ish (`inf + inf = inf`) until a contradiction. A `NaN` is fully undefined — it has no sign, no magnitude, and every operation with it returns NaN. In practice, most NaNs in training are *born as Inf and converted to NaN one operation later*: a logit overflows to `+inf`, then a `softmax` does `inf - inf = NaN` in its log-sum-exp, or an `inf` activation multiplies a `0` mask and gives `0 * inf = NaN`. When you hunt, finding an `inf` upstream of a `NaN` is finding the actual source; the `NaN` is just the symptom one hop later.

## 2. The seven sources, with the math for each

Here is the full taxonomy. I claimed there are about seven sources; let me make each one rigorous, because knowing the math lets you predict the signature and pick the right guardrail without instrumenting anything. The figure below is the lookup table we will fill in for the rest of the post: source, the one-line math reason, and the guardrail that catches it early.

![A matrix mapping six NaN sources to their one-line math cause and the single guardrail that catches each at step 1 instead of step 4000](/imgs/blogs/hunting-nans-and-infs-2.png)

### Source 1: log of zero or a negative

The logarithm diverges to $-\infty$ as its argument approaches zero from above: $\lim_{x \to 0^+} \log(x) = -\infty$. And $\log$ of a negative number is undefined over the reals, so it returns NaN. This is the single most common NaN source in classification, because cross-entropy *is* a logarithm of a probability. The cross-entropy loss for a true class $y$ is $-\log p_y$, where $p_y$ is the model's predicted probability of the correct class. If the model ever assigns exactly zero probability to the true class — which happens easily after a `softmax` saturates, or when you `clamp` a probability to `[0, 1]` and it lands on `0` — then $-\log(0) = +\infty$, and your loss is `inf`. One operation later, that `inf` participates in a mean or a backward pass and becomes NaN.

KL divergence has the same disease, twice over. The KL between distributions $P$ and $Q$ is $\sum_i p_i \log(p_i / q_i)$, which contains $\log p_i$ and $\log q_i$ and a division $p_i / q_i$. If any $q_i = 0$ where $p_i > 0$, the term is $p_i \log(p_i / 0) = p_i \cdot \infty = \infty$. KL is a frequent NaN source in distillation, in variational autoencoders, and in RLHF/DPO where you compute the KL between the policy and a reference model.

The fix is never to take the log of a raw probability. Use the **log-sum-exp** identity to compute log-probabilities directly from logits in a numerically stable way (Source 3 explains why this is stable), or add a small epsilon: $\log(p + \epsilon)$ with $\epsilon = 10^{-7}$ or so. PyTorch's `F.cross_entropy` and `F.log_softmax` do this internally, which is exactly why you should use them instead of a hand-rolled `torch.log(torch.softmax(...))`.

### Source 2: division by zero

The straightforward one: $x / 0 = \pm\infty$ for $x \neq 0$, and $0 / 0 = \text{NaN}$. In a neural network this hides inside every *normalization*. LayerNorm divides by $\sqrt{\text{Var}(x) + \epsilon}$; BatchNorm divides by $\sqrt{\text{running\_var} + \epsilon}$. The `+ epsilon` is there precisely to prevent division by zero — but only if the epsilon is large enough and is actually present. A common bug is normalizing a feature that happens to be constant across the batch (variance exactly zero) with an epsilon of zero or a too-small epsilon that fp16 rounds to zero. Cosine similarity divides by the product of norms; if either vector is the zero vector, you divide by zero. Attention's scaled dot-product divides by $\sqrt{d_k}$ — that one is safe because $d_k$ is a positive constant — but custom attention variants that normalize by a learned or data-dependent quantity are not.

The fix is an epsilon in the denominator that is large enough to survive your dtype. In fp16, the smallest normal positive number is about $6.1 \times 10^{-5}$, so an epsilon of `1e-8` literally *is zero* in fp16 and gives you no protection. Use `1e-5` for fp16-friendly code, or do the normalization in fp32 (autocast does this for LayerNorm and BatchNorm by default).

### Source 3: exp overflow and the unstable softmax

The exponential grows without bound: $\exp(z) \to \infty$ as $z \to \infty$. In fp32, $\exp(z)$ overflows to `inf` at around $z \approx 88.7$; in fp16 it overflows much earlier, around $z \approx 11.1$, because fp16's max value is only $65{,}504 \approx \exp(11.09)$. So an un-clamped logit of magnitude 12 — entirely plausible in a confident, late-training model, or in a finetune with a learning rate that briefly spiked — overflows fp16 the instant you exponentiate it.

The naive softmax is $\text{softmax}(z)_i = \exp(z_i) / \sum_j \exp(z_j)$. If any $z_j$ is large, $\exp(z_j)$ overflows to `inf`, and then you have `inf / inf = NaN`. The standard fix is the **max-subtraction trick**: subtract the row max $m = \max_j z_j$ before exponentiating.

$$\text{softmax}(z)_i = \frac{\exp(z_i - m)}{\sum_j \exp(z_j - m)}$$

This is mathematically identical (the $\exp(m)$ factor cancels top and bottom) but numerically safe: now the largest exponent argument is $z_i - m \le 0$, so $\exp(z_i - m) \le 1$ and cannot overflow. The smallest can still underflow to zero, but underflow to zero in the *numerator* is harmless as long as the denominator has at least one non-zero term (the max element gives $\exp(0) = 1$). The log-sum-exp used in `log_softmax` applies the same trick: $\log \sum_j \exp(z_j) = m + \log \sum_j \exp(z_j - m)$. This is the single most important numerical-stability identity in deep learning, and every framework's `softmax`, `log_softmax`, and `cross_entropy` implement it. Your job is to *use those functions* rather than reimplementing the unstable formula.

### Source 4: sqrt of a (tiny) negative

The square root is undefined for negatives over the reals: $\sqrt{x} = \text{NaN}$ for $x < 0$. You would think this never happens — variances and squared norms are non-negative by construction. But floating-point arithmetic does not respect that. A variance computed as $\mathbb{E}[x^2] - (\mathbb{E}[x])^2$ can come out *slightly negative* — like $-10^{-9}$ — due to catastrophic cancellation when the two large terms nearly cancel. Then $\sqrt{-10^{-9}} = \text{NaN}$. The same happens in the backward pass of `sqrt`: the gradient of $\sqrt{x}$ is $\frac{1}{2\sqrt{x}}$, which is `inf` at $x = 0$, so even a *legitimately zero* input to `sqrt` produces an `inf` gradient that becomes NaN downstream. This is why `torch.norm(zero_vector)` can give a clean `0` on the forward pass but a `NaN` gradient on the backward pass — a notorious trap.

The fix is `x.clamp_min(0)` before the `sqrt`, and an epsilon inside the `sqrt` to keep its gradient finite: `torch.sqrt(x.clamp_min(0) + eps)`. Or use the framework's `F.normalize` and norm functions, which guard this for you.

### Source 5: all-masked softmax (0 / 0 from attention)

This one deserves its own section because it is the most common NaN in transformer training and it is genuinely subtle. Attention masks out forbidden positions by adding $-\infty$ (or a very large negative number like `-1e9`) to the attention scores before the softmax, so that $\exp(\text{score}) = \exp(-\infty) = 0$ and the masked position gets zero attention weight. That works fine — *unless an entire row is masked*. If every position in a query's row is masked, then every score is $-\infty$, every $\exp$ is `0`, the denominator $\sum_j \exp(\cdot) = 0$, and the softmax computes $0 / 0 = \text{NaN}$ for every element of that row.

When does a whole row get masked? When a query position has no valid keys to attend to. This happens with **fully-padded sequences** in a batch (a sequence that is all padding tokens), with certain sliding-window or block-sparse masks at the boundaries, and with the combination of a causal mask and a padding mask where a padding query position is causally allowed to attend only to other padding positions that are themselves masked. The bug is invisible at small scale and appears the moment your batch contains one pathological sequence — which is why it often shows up not at step 1 but at the first batch that happens to include an empty sequence.

![A before-and-after diagram contrasting a fully-masked attention row that softmaxes to a 0 over 0 NaN against the fix that detects empty rows and skips the softmax to keep attention finite](/imgs/blogs/hunting-nans-and-infs-3.png)

The fix has two parts. First, prevent fully-padded sequences from entering the batch at all (a data-pipeline guard). Second, in the attention kernel, detect rows that are entirely masked and write a zero output for them instead of running the softmax — which is exactly what the fused `F.scaled_dot_product_attention` and FlashAttention do internally, another reason to use the library kernel rather than a hand-rolled `softmax(QK^T / sqrt(d) + mask) @ V`.

### Source 6: bad labels, targets, and indices

Not every NaN is born in the numerics; some are *fed in* from the data. Three flavors:

- **A NaN already in the data.** A tabular feature column with a `NaN` value, an image that decoded to NaN, an audio clip with a corrupt sample. The NaN enters at the input and propagates from there. This is a *data* bug masquerading as a numerics bug, and the tell is that it appears at the specific step that loads the bad row, reproducibly, regardless of learning rate.
- **An out-of-range class index.** `F.cross_entropy` and `F.nll_loss` index into the log-probability tensor with the label. If a label is `-1` (other than the `ignore_index` sentinel), or `C` when there are only `C` classes `0..C-1`, the gather either reads out of bounds (undefined, often a NaN or a crash) or indexes the wrong row. The classic version: your labels are `1..C` but the loss expects `0..C-1`, an off-by-one that silently corrupts every loss.
- **Targets out of the valid range.** A regression target of `inf`, a probability target outside `[0, 1]` fed to `binary_cross_entropy`, a normalized target that a bad scaler sent to `inf` by dividing by a zero standard deviation.

The fix is an assertion at the data boundary: check that inputs and labels are finite and in range *before* the forward pass, so a bad row fails loudly at the dataloader rather than silently at the loss.

### Source 7: fp16 overflow and underflow

Reduced precision deserves its own source because it *converts otherwise-safe computations into NaN sources by shrinking the representable range*. fp16 (half precision) has a 5-bit exponent, giving a maximum representable value of $65{,}504$ and a smallest normal positive value of about $6.1 \times 10^{-5}$. Anything larger than $65{,}504$ overflows to `inf`; anything smaller than the minimum underflows toward `0`. So a perfectly reasonable activation of magnitude $7 \times 10^4$ — which fp32 represents trivially — becomes `inf` in fp16, and then `inf * 0 = NaN` one operation later. Conversely, a gradient of magnitude $10^{-8}$ — which carries real signal — underflows to `0` in fp16, so it silently vanishes and the parameter stops learning (a different bug, the subject of the mixed-precision post, but the same root cause: range).

![A before-and-after diagram showing an fp16 activation past the 65504 ceiling rounding to inf and then NaN, fixed by bf16's wider range and a gradient scaler](/imgs/blogs/hunting-nans-and-infs-7.png)

The fixes are loss scaling (multiply the loss by a large factor so gradients land in fp16's representable range, then unscale before the optimizer step — what `torch.amp.GradScaler` automates) and switching to **bf16**, which trades mantissa bits for exponent bits: bf16 has an 8-bit exponent (the same range as fp32, up to about $3 \times 10^{38}$) but only a 7-bit mantissa. bf16 essentially cannot overflow at neural-network scales, which is why it has become the default for training large models. We will work this exact before-and-after in the case studies, and it cross-links to the dedicated [mixed-precision debugging post](/blog/machine-learning/debugging-training/mixed-precision-debugging-fp16-vs-bf16) and to the [sensitivity-analysis treatment of mixed precision](/blog/machine-learning/edge-ai/mixed-precision-and-sensitivity-analysis) on the efficiency side.

That is the whole list. Seven sources, each with a one-line math reason. Memorize them and a NaN stops being a mystery and becomes a multiple-choice question.

## 3. Read the signature before you instrument

Before writing a single line of diagnostic code, you can narrow the suspect from seven to two just by reading the *shape of the failure* — the same way you read a loss curve. The NaN's timing and its precursor signal are diagnostic.

**Step-1 NaN.** If the very first loss is NaN (or `inf`), the bug is almost never the numerics of optimization — there has been no optimization yet. A step-1 NaN points at the data or the model code: a NaN in an input, a bad label index, an all-masked sequence in the first batch, a zero-variance normalization on a constant feature, or a hand-rolled unstable loss that overflows on the very first forward pass. This is the *good* case, because it is reproducible and cheap to hunt: the same first batch fails the same way every time.

**Smooth-then-NaN.** If the loss descends cleanly for thousands of steps and *then* goes NaN, the bug is gradual numerical drift, and the suspects are overflow and `log(0)`. As training proceeds, the model gets more confident: logits grow in magnitude, which pushes `exp` toward overflow and pushes the softmax's top probability toward exactly `1.0` (so the others toward exactly `0.0`, so `log(0)` on a wrong-but-confident prediction). Or the learning rate, on a warmup-then-decay schedule, hit its peak and a single large update blew an activation past the fp16 ceiling. A smooth-then-NaN curve is a *numerics* story, not a data story — do not go hunting through your dataset.

**Spike-then-NaN.** If the loss spikes sharply (say from 2.0 to 40.0) and then goes NaN a few steps later, that is the loss-spike-into-divergence signature: a learning rate that is slightly too high, or a single pathological batch, produced an outsized update, which produced larger activations, which overflowed, which became NaN. This sits on the boundary between optimization and numerics and is covered in depth in the [loss spikes and divergence post](/blog/machine-learning/debugging-training/loss-spikes-and-divergence); the grad-norm overlay is what disambiguates it (a grad-norm spike just before the loss spike confirms it is optimization). It also overlaps heavily with [exploding and vanishing gradients](/blog/machine-learning/debugging-training/gradients-exploding-and-vanishing), since an exploding gradient is the most common path from a healthy run to an overflow.

**Intermittent NaN at irregular steps.** If the NaN appears at step 412 on one run and step 1,033 on a re-run with a different seed, and the steps do not correspond to a fixed data index, it is data-dependent through *augmentation* or *sampling* randomness, or it is a genuinely rare numerical edge case (an all-masked row that only appears when a particular short sequence lands in a batch). If the NaN appears at the *same* data index every time, it is a specific bad row — go look at that row.

The table below is the signature-to-suspect map. Reading it costs nothing and saves you from instrumenting the wrong place.

| Failure signature | Most likely suspect | First confirming test |
| --- | --- | --- |
| Loss NaN at step 1, reproducible | Data (bad row/label) or unstable loss | Assert-finite on inputs/labels; print the first batch |
| Smooth descent, then NaN | Overflow or `log(0)` from growing logits | Clamp logits; switch to `F.cross_entropy`; bf16 |
| Sharp spike, then NaN | LR too high / bad batch (optimization) | Grad-norm overlay; lower LR; clip; skip-batch |
| NaN at the same data index | A specific corrupt example | Look at that row; assert-finite at the dataloader |
| NaN only at fp16, clean at fp32 | fp16 overflow/underflow | bf16, or fp32 for the offending op |
| NaN only with a padding/attention mask | All-masked softmax row | Drop fully-padded sequences; use fused attention |
| NaN only on multi-GPU | A rank with bad data, or an all-reduce of inf | Single-GPU repro; per-rank assert |

The last row is worth a sentence: if a run is clean on one GPU and NaN on eight, the NaN is real on at least one rank, and DDP's `all_reduce` of gradients spreads it to all ranks (the average of anything with a NaN is NaN). Reproduce on a single GPU with each rank's data shard to find the offending shard. That bisection lives in the [DDP debugging track], but the NaN-hunting mechanics are identical.

## 4. The systematic hunt, part one: bisect by step

You have read the signature and you suspect numerics. Now you localize. The hunt has two axes — *when* (which step) and *where* (which operation) — and you bisect along both. Start with *when*, because pinning the step gives you a reproducible scene to investigate.

The naive approach is to add a NaN check after every step and let the run tell you the first step. That works but is slow if the NaN is at step 4,000 of a long run. The faster approach is **binary search over checkpoints**, which finds the first-NaN step in $\log_2(N)$ replays instead of one full run.

![A timeline showing binary search over training steps to localize the first NaN, narrowing from step 4000 through 2000, 3000, 3500 to step 3217, then saving the checkpoint just before to replay with hooks](/imgs/blogs/hunting-nans-and-infs-4.png)

The procedure: you know the NaN appears by step 4,000. Replay (or checkpoint) at step 2,000 — is the model still finite? If yes, the first NaN is between 2,000 and 4,000; check step 3,000. Still finite? Check 3,500. NaN there? The first NaN is between 3,000 and 3,500; check 3,250, then 3,200, narrowing until you pin it to, say, step 3,217. Now you save the checkpoint at step 3,216 — the last clean state — and replay just the single step 3,216 to 3,217 with full instrumentation. You have turned a 4,000-step needle-in-a-haystack into one fully-instrumented step. The prerequisite is that the run be *deterministic* (same seed, same data order, deterministic kernels) so the NaN reproduces at the same step on replay — which is exactly why [reproducibility and determinism](/blog/machine-learning/debugging-training/reproducibility-and-determinism-in-training) is a prerequisite for debugging at all. If your run is non-deterministic, fix that first or you will be hunting a moving target.

Here is the per-step finite check that gives you the exact first-NaN step on a single pass. It is cheap enough to leave on for the whole run:

```python
import torch

def first_nonfinite_param(model):
    # Returns (name, tensor) of the first parameter that is not all-finite.
    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            return name, p
    return None, None

for step, batch in enumerate(loader):
    optimizer.zero_grad(set_to_none=True)
    out = model(**batch)
    loss = out.loss

    # Catch a non-finite loss the instant it happens, before it spreads.
    if not torch.isfinite(loss):
        print(f"[NaN HUNT] non-finite loss {loss.item()} at step {step}")
        torch.save({"step": step, "batch": batch}, "nan_step.pt")
        raise FloatingPointError(f"loss became {loss.item()} at step {step}")

    loss.backward()

    # Catch a non-finite gradient before the optimizer writes it into the weights.
    bad = [n for n, p in model.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    if bad:
        print(f"[NaN HUNT] non-finite grad in {bad[:5]} at step {step}")
        raise FloatingPointError(f"grad became non-finite at step {step}")

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

Two things make this the right first move. It raises on the *loss* before the gradient is even computed, so you catch the NaN one full step before the optimizer would have spread it into the weights — meaning the model at the checkpoint just before is still clean and replayable. And it distinguishes a NaN that originates in the *forward* pass (loss already NaN) from one that originates in the *backward* pass (loss finite, but some gradient NaN) — which is the next fork in the hunt.

#### Worked example: bisecting a NaN at step 4,000

Concretely: a 12-layer transformer finetune, bf16 off (fp16 AMP on), `nan` loss discovered when the dashboard flatlined around step 4,000. I add the per-step `isfinite` check above and re-run with a fixed seed. It raises: `non-finite loss inf at step 3217`. Not NaN — `inf`. That single word narrows the suspect enormously: `inf`, not NaN, means *overflow*, not `0/0`. The loss is `inf` because the cross-entropy computed $-\log(p_y)$ where $p_y$ underflowed to exactly `0` (so $-\log 0 = +\infty$), and the underflow happened because a logit grew large enough that the fp16 softmax sent the competing classes to `0`. The signature (smooth-then-`inf` at fp16) plus the value (`inf` not NaN) plus the step (late, after the model got confident) is a textbook Source-1-via-Source-7. I have not even opened the model code yet and I already know the fix is "compute the loss in fp32 with `F.cross_entropy` and/or switch to bf16." The replay confirms it: at step 3,216 the max logit was $11.4$, and $\exp(11.4) = 89{,}000 > 65{,}504$, so the fp16 softmax overflowed. We will fix it in section 7.

## 5. The systematic hunt, part two: bisect by layer

You have the step. Now find the *operation*. There are two tools, and you reach for them in this order: `torch.autograd.set_detect_anomaly` for backward-pass NaNs, and finite-checking forward hooks for forward-pass NaNs. The tree below is the decision procedure.

![A decision tree routing a NaN at a known step through a forward-versus-backward split and a finite-inputs check to identify the one offending operation and its stable-formula fix](/imgs/blogs/hunting-nans-and-infs-5.png)

### Anomaly detection: the backward-pass backtrace

PyTorch's autograd anomaly mode is the single most powerful NaN-hunting tool, and almost nobody knows it exists. When you wrap your forward and backward in `torch.autograd.set_detect_anomaly(True)`, autograd does two things: it runs the forward pass with extra bookkeeping that records the Python stack trace where each operation was created, and on the backward pass it checks every gradient for NaN and, the instant it finds one, raises an error *that includes the forward-pass traceback of the operation whose backward produced the NaN*. In other words, it points a finger at the exact line of your model code that created the offending op.

```python
import torch

# Localize the op whose BACKWARD produces a NaN. Slow (2-5x) — use only to hunt.
torch.autograd.set_detect_anomaly(True)

out = model(**bad_batch)        # the batch saved from the bisection above
loss = out.loss
loss.backward()                 # raises a RuntimeError with the forward traceback
```

The error looks like this (abbreviated), and the gold is the second traceback:

```bash
RuntimeError: Function 'LogBackward0' returned nan values in its 0th output.

# ... and crucially, the forward-pass traceback it captured:
Traceback of forward call that caused the error:
  File "model.py", line 88, in forward
    loss = -torch.log(probs.gather(1, targets)).mean()
```

That backtrace says: the NaN's backward came from a `log`, on line 88, and the argument to that `log` is a gathered probability. That is `log(0)` — Source 1 — caught red-handed. You did not have to guess; the tool named the operation and the line. The cost is real: anomaly mode is two-to-five times slower because of the bookkeeping, and it raises on the first anomaly so it stops your run. So you do not leave it on; you turn it on for the *single bisected step* from section 4, get the backtrace, turn it off.

A subtlety: anomaly mode catches NaNs in the *backward* pass. Sometimes the forward pass is finite (the loss is a normal number) but the backward produces a NaN gradient — the canonical example is `sqrt(0)`, whose forward is a clean `0` but whose backward is $\frac{1}{2\sqrt{0}} = \infty$, or `log`/`div` whose forward is finite but whose local gradient blows up at the boundary. Anomaly mode is exactly the tool for these, because the symptom (NaN gradient, finite loss) is invisible to a forward-only check.

### Finite-checking forward hooks: localizing a forward-pass NaN

If the loss itself is non-finite, the NaN was born in the *forward* pass, and you want to know which layer's output first went non-finite. Forward hooks let you assert finiteness on every module's output without editing the model:

```python
import torch

def assert_finite_hook(name):
    def hook(module, inputs, output):
        out = output[0] if isinstance(output, tuple) else output
        if torch.is_tensor(out) and not torch.isfinite(out).all():
            n_bad = (~torch.isfinite(out)).sum().item()
            raise FloatingPointError(
                f"[{name}] produced {n_bad} non-finite values; "
                f"input finite={torch.isfinite(inputs[0]).all().item()}")
    return hook

# Register on every leaf module so the FIRST one to go non-finite fires.
handles = []
for name, module in model.named_modules():
    if len(list(module.children())) == 0:        # leaf modules only
        handles.append(module.register_forward_hook(assert_finite_hook(name)))

out = model(**bad_batch)        # raises at the first non-finite layer output

for h in handles:               # clean up when done hunting
    h.remove()
```

The error tells you *two* critical facts: which module produced the non-finite output, **and whether that module's input was already non-finite**. This second fact is the whole game. If the input was finite and the output is non-finite, *this module is the source* — it took good numbers and made a NaN, so it is an unstable op (a `log`, a `div`, a `softmax`, a `sqrt`). If the input was already non-finite, this module is just a victim — the source is *upstream*, and you walk back toward the input until you find the first module whose input is clean and output is dirty. That walk is the layer-bisection, and the `input finite=` field is what lets you do it in one pass instead of many.

#### Worked example: a NaN in the backward of a custom loss

A speech model finetune (CTC head, but the lesson generalizes) shows a finite loss for the first 50 steps, then `nan` gradients with the loss still printing a normal `4.3`. A forward-only finite check finds nothing — the forward is clean. So I reach for anomaly mode on the offending step. It raises: `Function 'SqrtBackward0' returned nan values`, with the forward traceback pointing at a custom feature-normalization line `feats = x / torch.sqrt(x.pow(2).sum(-1, keepdim=True))`. The forward of that `sqrt` is finite (the squared sum is a small positive number), but for one all-silence audio clip the squared sum is *exactly zero*, so the `sqrt` forward is `0` and its backward is $\frac{1}{2\sqrt{0}} = \infty$, which becomes NaN. The fix is `torch.sqrt(x.pow(2).sum(-1, keepdim=True).clamp_min(1e-12))` — or, better, `F.normalize(x, dim=-1)`, which guards this internally. Grad norm went from `nan` to `3.1` and the run finished. This is Source 4, and it is *only* findable with anomaly mode because the forward pass lies — it looks perfectly healthy.

## 6. Guardrails: catch it at step 1, not step 4,000

Hunting a NaN after the fact is a skill you should have, but the better engineering is to make the NaN announce itself the instant it is born — ideally at step 1 of a guarded run, with the exact tensor named — so you never lose a 4,000-step run again. The difference between a senior and a junior NaN experience is entirely in the guardrails. Here is the stack of them, cheapest and earliest first.

![A vertical stack of guardrails — input and label checks, a stable loss, logit clamping, finite hooks, and anomaly detection — that together catch a NaN at step 1 with the offending tensor named instead of step 4000](/imgs/blogs/hunting-nans-and-infs-6.png)

**Guardrail 1: check inputs and labels at the data boundary.** Before the forward pass, assert that inputs are finite and labels are in range. This catches Source 6 (bad data) at the dataloader, where the error message can name the offending row, instead of at the loss, where it is a context-free NaN.

```python
def check_batch(batch, num_classes):
    x, y = batch["input"], batch["labels"]
    assert torch.isfinite(x).all(), "non-finite values in inputs"
    # Allow the -100 ignore_index sentinel used by HF for masked positions.
    valid = (y == -100) | ((y >= 0) & (y < num_classes))
    assert valid.all(), f"label out of [0,{num_classes}): {y[~valid][:10].tolist()}"
```

**Guardrail 2: use the numerically stable loss formulations.** This is the highest-leverage single change in the entire post. Do not hand-roll `torch.log(torch.softmax(logits, -1))`; use `F.log_softmax` (max-subtraction inside) or, better, pass *raw logits* to `F.cross_entropy`, which fuses `log_softmax` and `nll_loss` and never materializes a probability you could take a `log(0)` of. The before-and-after is dramatic:

```python
import torch.nn.functional as F

# DANGEROUS: probs can be exactly 0, log(0) = -inf, then NaN.
probs = torch.softmax(logits, dim=-1)
loss_bad = -torch.log(probs.gather(1, targets.unsqueeze(1))).mean()

# STABLE: log_softmax uses the max-subtraction trick; never logs a raw 0.
loss_good = F.cross_entropy(logits, targets)          # pass LOGITS, not probs

# For binary: use the *_with_logits variant, never sigmoid-then-BCE.
loss_bce = F.binary_cross_entropy_with_logits(logits, targets.float())
```

The rule to memorize: **whenever a function name has a `_with_logits` or accepts logits directly, use that form** — `cross_entropy`, `binary_cross_entropy_with_logits`, `log_softmax`. The framework authors already solved the numerics; the only way to reintroduce the bug is to reimplement it yourself.

**Guardrail 3: clamp logits and other unbounded quantities.** Where a quantity can grow without bound and then overflow, clamp it. Clamping logits to $\pm 50$ before a custom softmax, clamping the KL ratio in DPO, clamping the standard deviation of a Gaussian policy to a floor — these are cheap and they cap the exponential before it overflows. `logits = logits.clamp(-50, 50)` costs nothing and makes Source 3 impossible.

**Guardrail 4: leave a lightweight finite check on, and a heavy one ready.** Keep the per-step loss/grad `isfinite` check from section 4 on for the whole run — it is microseconds and it converts a silent step-4,000 corruption into a loud step-3,217 stop with a saved batch. Keep the forward-hook and anomaly-mode snippets in a `--debug-nan` flag you can flip on for a single re-run when the cheap check fires.

The table below is the guardrail-to-source map — which guardrail neutralizes which source, and what it costs.

| Guardrail | Neutralizes (source) | Overhead | Leave on? |
| --- | --- | --- | --- |
| Assert inputs/labels finite & in range | 6 (bad data/labels) | microseconds | Yes |
| `F.cross_entropy` / `log_softmax` | 1, 3 (log0, exp overflow) | none (faster) | Yes, always |
| `_with_logits` BCE | 1, 3 | none | Yes, always |
| Epsilon in denominators / `clamp_min` before sqrt | 2, 4 (div0, sqrt-neg) | negligible | Yes |
| Clamp logits / ratios to a range | 3, 7 (overflow) | negligible | Yes |
| Drop fully-padded rows / fused attention | 5 (all-masked softmax) | negligible | Yes |
| Per-step `isfinite` on loss & grad | all (detection) | microseconds | Yes |
| Forward `isfinite` hooks | all (localization) | ~10-20% | Only when hunting |
| `set_detect_anomaly(True)` | backward NaNs | 2-5x | Only when hunting |
| Loss scaling (`GradScaler`) / bf16 | 7 (fp16 range) | none | Yes, for AMP |

Notice that almost everything in this table that *prevents* NaNs is free or nearly free and should be on permanently. Only the two *localization* tools (forward hooks, anomaly mode) are expensive, and those you flip on for a single re-run. The economics are overwhelming: a few microseconds per step versus a wasted multi-thousand-step run.

#### Worked example: from a wasted run to a step-1 catch

The cost case is stark. Suppose your run does 4,000 steps before the NaN, at roughly 1.5 seconds per step on a single A100 you rent at about \$2 per GPU-hour. Those 4,000 wasted steps are 100 minutes, or about \$3.30 of compute — for *one* failed attempt. If it takes you four attempts to find the bug by staring at the corpse, you have burned over \$13 and the better part of an afternoon. The guarded version stops at step 1 (or at whatever step the bad batch first appears), names the tensor, and costs you the few seconds to read the assertion. On a real multi-GPU run at \$30 to \$40 per GPU-hour across eight cards, a single 4,000-step wasted attempt is closer to \$25 to \$35, and the four-attempt hunt is a \$100-plus afternoon. The guardrails are not hygiene; they are the difference between a five-second fix and a hundred-dollar one.

## 7. The before-and-after: removing the step-3,217 NaN

Let us close the loop on the worked example from section 4 with a concrete, measured fix. The diagnosis: an fp16 finetune whose loss went `inf` at step 3,217 because the cross-entropy took $-\log$ of a probability that underflowed to `0` in fp16 after the model's max logit reached $11.4$ (and $\exp(11.4) \approx 89{,}000$ overflows the fp16 ceiling of $65{,}504$). The root cause is two sources stacked: Source 7 (fp16 range) enabling Source 1 (`log(0)`).

There are two independent fixes, and the lesson is to apply the cheaper one first and measure.

**Fix A — compute the loss stably and in fp32.** The loss was hand-rolled as `-(F.softmax(logits, -1).gather(...)).log().mean()`. Replacing it with `F.cross_entropy(logits.float(), targets)` does two things: it casts the logits to fp32 (so the softmax has the full fp32 range, overflowing only near $\exp(88.7)$, which a logit of 11.4 is nowhere near), and it uses the fused, max-subtracting, never-materializes-a-probability path. This is a one-line change.

**Fix B — switch the autocast dtype from fp16 to bf16.** `torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)` gives the whole forward pass the fp32-equivalent range, so even an un-clamped logit of magnitude 30 does not overflow. bf16 needs no `GradScaler` (its range is wide enough that gradients do not underflow the way they do in fp16), which also removes a moving part.

Here is the before-and-after, measured on the replayed window from step 3,216:

| Quantity | Before (fp16, hand-rolled loss) | After (Fix A: fp32 `cross_entropy`) | After (Fix B: bf16 autocast) |
| --- | --- | --- | --- |
| Max logit at step 3,216 | 11.4 | 11.4 | 11.4 |
| `exp(max logit)` in working dtype | `inf` (> 65,504) | finite (fp32) | finite (bf16) |
| Loss at step 3,217 | `inf` → NaN next step | 0.74, finite | 0.74, finite |
| Grad norm at step 3,217 | `nan` | 2.1 | 2.0 |
| Steps to completion (10k target) | dead at 3,217 | 10,000 clean | 10,000 clean |
| Throughput vs fp16 baseline | — | ~0.97x (fp32 loss only) | ~1.0x |

Both fixes work; the loss is identical to two decimals afterward because the math was always supposed to be the same — the NaN was purely a numerical artifact, not a modeling difference. In practice I apply *both*: the stable loss formulation as a permanent guardrail (it is free and it is correct regardless of dtype) and bf16 as the training precision (it removes a whole class of fp16-only NaNs at no throughput cost on Ampere and newer hardware). How would you confirm the fix honestly? Re-run from the same seed with the same data order and verify the run passes step 3,217 — the exact step that failed before — and then completes to 10,000 with the per-step `isfinite` check on and never firing. A fix that you cannot re-trigger the original failure to test is a fix you do not trust.

## 8. Stress-testing the diagnosis across modalities and regimes

A good NaN method survives a stress test: what changes when the modality, the precision, the batch size, or the parallelism changes? Walk through the regimes, because each one shifts which source is most likely.

**"What if it is data, not numerics?"** A NaN at the same data index every time, surviving a switch to fp32 and a stable loss, is not numerics — it is a corrupt row. The discriminating test: run the model in fp32 with `F.cross_entropy` and assert-finite on the inputs at the dataloader. If the assert fires, you have a NaN *in the data* (Source 6), and the fix is in the pipeline, not the math. This connects to [the input pipeline post](/blog/machine-learning/debugging-training/the-input-pipeline-is-lying-to-you): a NaN that survives every numerics fix is a data bug wearing a numerics costume.

**"What if it only fails at fp16?"** If fp32 and bf16 are clean and only fp16 NaNs, the bug is purely Source 7 — range, not formula. Do not hunt for an unstable op; there is not one. The fix is bf16 or loss scaling, and the diagnostic is a gradient histogram: if the gradients pile up against the fp16 underflow floor (around $6 \times 10^{-5}$) or overflow ceiling, the histogram shows it directly. This is the heart of the [mixed-precision post](/blog/machine-learning/debugging-training/mixed-precision-debugging-fp16-vs-bf16).

**"What if the batch is tiny?"** Small batches make the all-masked-softmax bug (Source 5) and the zero-variance-normalization bug (Source 2) *more* likely, because with a batch of 1 or 2 it is more probable that an entire batch is a single short or padded sequence, or that a feature is constant across the (tiny) batch. BatchNorm specifically divides by the batch variance, which is unstable or zero at batch size 1 — a NaN that appears only when the batch shrinks (the last partial batch, say) is pointing at a batch-statistic division. This overlaps with [initialization and normalization bugs](/blog/machine-learning/debugging-training/initialization-and-normalization-bugs).

**"What if it only fails on multi-GPU?"** If single-GPU is clean and eight-GPU NaNs, the NaN is born on one rank and DDP's gradient `all_reduce` spreads it to all (the mean of anything-with-a-NaN is NaN). Bisect by rank: run each rank's data shard alone on one GPU and find the shard that NaNs. Usually it is a bad row that only landed in one shard, or a sequence-length distribution on one rank that triggers the all-masked-softmax. The hunt is identical; you have just added a "which rank" axis before the "which step" axis.

**"What if it is an LLM finetune specifically?"** The LLM-specific NaN sources are the all-masked attention row (Source 5, from a fully-padded sequence or a packing boundary), the loss computed over a fully-masked label sequence where every position is `-100` (so the mean is `0/0`), and the reward/KL blow-up in RLHF/DPO (an exponential of a large log-ratio overflowing). The label-masking version is sneaky: if every label in a sequence is the `-100` ignore index, the cross-entropy averages over zero valid positions and divides by zero. Guard it by skipping sequences with no valid labels, which connects to [the loss masking bug](/blog/machine-learning/debugging-training/the-loss-masking-bug) and [loss function bugs](/blog/machine-learning/debugging-training/loss-function-bugs).

The unifying point across all five regimes: the *method* is invariant. Read the signature, bisect by step, bisect by layer, read the offending op off the backtrace or the finite-hook, fix the math, install the guardrail. Only the *prior* over which source is most likely shifts with the regime — and the signature table in section 3 encodes that prior.

## 9. The stable softmax and log-sum-exp, derived

I have told you several times to "use `F.cross_entropy`" and to "use the stable formulation," and you should take that as the operational advice. But to debug well you need to know *what* the framework does inside, because the day you write a custom loss — a focal loss, a label-smoothed cross-entropy, a contrastive objective, a DPO loss — you are the one who has to keep it stable, and "use the library function" is no longer an escape hatch. So here is the derivation, with the code, of the single most important numerical-stability trick in deep learning.

Start with the unstable softmax and trace where it dies. The softmax of a logit vector $z$ is $\sigma(z)_i = \exp(z_i) / \sum_j \exp(z_j)$. If the largest logit is $z_{\max} = 20$, then in fp16 you compute $\exp(20)$, which is about $4.85 \times 10^8$, far past the fp16 ceiling of $65{,}504$ — so it rounds to `inf`. Now your numerator and denominator both contain `inf`, and `inf / inf = NaN`. The softmax of a perfectly ordinary logit vector is NaN, in fp16, the instant any logit exceeds about 11. Even in fp32 the same death happens at $z_{\max} \approx 89$, which a large-model logit or an exploding-gradient run reaches easily.

The fix is an algebraic identity, not an approximation. For any constant $c$,

$$\sigma(z)_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)} = \frac{\exp(z_i - c)}{\sum_j \exp(z_j - c)}$$

because multiplying the top and bottom by $\exp(-c)$ changes nothing. Choose $c = z_{\max} = \max_j z_j$. Now every exponent argument $z_i - c \le 0$, so every $\exp(z_i - c) \in (0, 1]$ — no overflow is possible, ever, in any dtype. The largest term is exactly $\exp(0) = 1$ (the max element), so the denominator is at least 1 and never zero, which means no division by zero either. Underflow can still send small terms to `0`, but a `0` in the numerator just means that class gets probability `0`, which is correct, and the denominator is still $\ge 1$. The max-subtraction trick makes the softmax unconditionally safe.

For the *log*-probability — which is what cross-entropy actually needs — you go one step further with **log-sum-exp**. The log-probability of class $i$ is $\log \sigma(z)_i = z_i - \log \sum_j \exp(z_j)$, and that $\log \sum_j \exp(z_j)$ is the log-sum-exp (LSE). Applying the same shift:

$$\text{LSE}(z) = \log \sum_j \exp(z_j) = c + \log \sum_j \exp(z_j - c), \quad c = z_{\max}$$

Now the inner sum is safe (every term $\le 1$, at least one term $= 1$), so the `log` is taken of a number $\ge 1$ — never `log(0)`. The full stable log-softmax is $\log \sigma(z)_i = (z_i - c) - \log \sum_j \exp(z_j - c)$, which never overflows and never logs a zero. Cross-entropy is then just $-\log \sigma(z)_y$ for the true class $y$ — a direct, never-NaN computation that *never materializes a probability you could accidentally `log(0)`*. That is the whole reason `F.cross_entropy(logits, targets)` is safe and `-(softmax(logits).gather(...)).log()` is not: the former computes the loss through LSE and never forms a raw probability; the latter forms probabilities, lets one underflow to `0`, and then logs it.

Here is the stable softmax and log-softmax written out, so you can see there is no magic — and so you have a template when you must hand-roll a custom loss:

```python
import torch

def stable_softmax(z, dim=-1):
    # Subtract the max so the largest exponent argument is 0: no overflow.
    z = z - z.amax(dim=dim, keepdim=True)
    e = torch.exp(z)                       # every term in (0, 1], safe
    return e / e.sum(dim=dim, keepdim=True)  # denom >= 1, no div-by-zero

def stable_log_softmax(z, dim=-1):
    # log-sum-exp form: never logs a raw probability, never overflows.
    c = z.amax(dim=dim, keepdim=True)
    shifted = z - c
    lse = c + torch.log(torch.exp(shifted).sum(dim=dim, keepdim=True))
    return z - lse                          # log p_i = z_i - LSE(z)

def stable_cross_entropy(logits, targets):
    # Equivalent to F.cross_entropy: gather the true-class log-prob, negate.
    logp = stable_log_softmax(logits, dim=-1)
    return -logp.gather(1, targets.unsqueeze(1)).squeeze(1).mean()
```

When you write a custom objective, this is the pattern: subtract the max before any `exp`, and compute log-probabilities through LSE rather than `log(softmax(...))`. The same idea generalizes — a stable sigmoid uses $\sigma(x) = \exp(\min(x, 0)) / (1 + \exp(-|x|))$ to avoid `exp` of a large positive, and `F.binary_cross_entropy_with_logits` does exactly that internally. The meta-lesson: whenever you see `exp` followed by a `sum` and a `log`, the max-subtraction belongs in there, and if you cannot find it, you have found your NaN's source.

#### Worked example: a focal loss that NaNs at high gamma

A detection model uses a hand-rolled focal loss, $\text{FL} = -(1 - p_t)^\gamma \log p_t$, where $p_t$ is the predicted probability of the true class. With $\gamma = 2$ it trains fine; someone bumps $\gamma$ to 5 to focus harder on hard examples, and the loss NaNs at step 1. The author had written `p = softmax(logits); loss = -((1-p_t)**gamma) * torch.log(p_t)`, which forms the probability `p_t` explicitly. For a confidently-wrong prediction, `p_t` underflows to `0`, `log(0) = -inf`, and `(1-0)**5 = 1`, so the loss is `1 * -inf = -inf` → NaN. The grad is worse: the `log`'s gradient is $1/p_t$, which is `inf` at `p_t = 0`. The fix routes through `stable_log_softmax`: compute `logp_t = stable_log_softmax(logits).gather(...)`, recover `p_t = logp_t.exp()` (which is a safe `exp` of a non-positive number, never `inf`), and write `loss = -((1 - p_t)**gamma) * logp_t`. The `log` is never taken of a raw probability — it comes straight from the LSE — and the NaN is gone at every $\gamma$. Measured: at $\gamma = 5$ the loss went from `nan` at step 1 to a finite `0.41`, grad norm from `nan` to `1.8`, and the run trained to a mean average precision two points higher than the $\gamma = 2$ baseline, which was the whole point of raising $\gamma$. The NaN was never a modeling problem; it was a `log(0)` the stable formulation removes.

## 10. Make it fail small: a minimal NaN reproducer

The series' master tool is **make-it-fail-small** — shrink the failing run until the bug is the only thing left, then study it in isolation. For a NaN this is especially powerful because a NaN is *deterministic given the bad input*: the same logit vector that overflows in your 70-billion-parameter run overflows in a three-line script, and the three-line script runs in a millisecond and you can step through it in a debugger. The discipline is to extract the single offending operation and its actual input — both of which you now have from the bisection — and reproduce the NaN with nothing else around it.

Concretely, once the forward hook from section 5 has named the module and you have the saved `bad_batch`, you isolate that module and feed it the captured input:

```python
import torch

# From the hunt: the module name and the input tensor that triggered the NaN.
bad_module = dict(model.named_modules())["loss_head"]
bad_input = torch.load("nan_step.pt")["activation"]   # captured by the hook

with torch.no_grad():
    out = bad_module(bad_input)
    print("input  range:", bad_input.min().item(), bad_input.max().item())
    print("output finite:", torch.isfinite(out).all().item())

# Now shrink: find the SINGLE row of the input that produces the non-finite output.
for i in range(bad_input.shape[0]):
    o = bad_module(bad_input[i:i+1])
    if not torch.isfinite(o).all():
        print(f"row {i} is the culprit; its input max = {bad_input[i].max():.1f}")
        break
```

This collapses the search from "somewhere in a 4,000-step run" to "row 37 of one batch, fed to one module," which is a thing you can print, inspect, and reason about completely. Three things make this the right closing move of the hunt. It *confirms* the diagnosis — if the isolated op on the isolated input does not NaN, your hunt pointed at the wrong place and you should not have edited anything yet. It *characterizes* the trigger — you learn the exact input value that breaks it (a logit of 11.4, a zero-variance row, an all-masked sequence), which tells you precisely which guardrail to install and where to set its threshold. And it gives you a *regression test*: that captured input becomes a unit test that asserts the fixed module returns finite values, so the bug can never silently return. A NaN you have reproduced in a three-line script is a NaN you have beaten; a NaN you "fixed" by adding an epsilon and hoping is a NaN that will be back next week on a different batch.

The same shrink works at every level of the stack. If the bug is data, shrink to the single corrupt row and assert-finite catches it. If it is the all-masked softmax, shrink to the single all-padding sequence and the row-skip fix removes it. If it is fp16 overflow, shrink to the single op, run it once in fp16 (NaN) and once in bf16 (finite), and you have proven the dtype is the cause. Make-it-fail-small is not a separate technique from the bisection; it is the bisection taken to its limit, where the suspect space is one operation on one input, and at that point the bug has nowhere left to hide. This is the same discipline as [the overfit-a-single-batch test](/blog/machine-learning/debugging-training/the-overfit-a-single-batch-test) applied to numerics: make the unit of failure as small as it can be, and the cause becomes self-evident.

## 11. Case studies and real signatures

A few well-known patterns, named and cited, so you recognize them in the wild.

**Loss scaling in the Mixed Precision Training paper (Micikevicius et al., 2018).** The original AMP paper showed that fp16 training of many networks underflows small gradients to zero — the gradient histogram has a large mass below fp16's smallest normal value, $6.1 \times 10^{-5}$, which simply vanishes. Their fix, *loss scaling*, multiplies the loss by a large constant (e.g., $2^{15}$) before the backward pass so the gradients shift up into fp16's representable range, then divides them back out before the optimizer step. PyTorch's `torch.amp.GradScaler` automates this with a *dynamic* scale that it raises when no overflow occurs and halves when it detects an `inf`/`nan` gradient (skipping that step's update rather than letting the NaN through). The `GradScaler` skipping an optimizer step on an overflow is itself a NaN guardrail you get for free with AMP — and a reason fp16 AMP runs can survive transient overflows that an unscaled fp16 run would die on.

**The all-masked-attention NaN in transformer training.** This is folklore among people who train decoder-only models, and it has a precise cause: a batch that contains a fully-padded sequence (all pad tokens) produces a query row that can attend to nothing, so its softmax is `0/0`. The reason fused kernels like FlashAttention and `F.scaled_dot_product_attention` are recommended is partly that they handle this edge case correctly (writing zeros for fully-masked rows) whereas a naive `softmax(scores + mask)` does not. If you ever see a transformer NaN that appears only on certain batches and only when padding is involved, this is your first suspect.

**`sqrt` and `norm` gradient NaNs.** The PyTorch documentation and many GitHub issues note that `torch.norm`, `torch.std`, and `torch.sqrt` have `inf`/`nan` gradients at zero input, because the derivative of $\sqrt{x}$ is $\frac{1}{2\sqrt{x}}$. This is a recurring trap in normalization layers, in contrastive losses that normalize embeddings, and in any code that takes the norm of a vector that can be zero. The community fix is universally `clamp_min(eps)` before the `sqrt` or using `F.normalize`. It is the canonical "forward is clean, backward is NaN" bug, and the canonical reason to keep `set_detect_anomaly` in your toolkit.

**Cross-entropy on a hand-rolled softmax.** The most common beginner NaN, and one that still bites experts who reimplement a loss for a custom objective: computing `torch.log(torch.softmax(x))` instead of `F.log_softmax(x)`, or `softmax` then a manual `-log(p)`, reintroduces the overflow and `log(0)` that the library functions are specifically engineered to avoid. The fix is always the same — pass logits to the fused function — and it is the reason the framework provides `cross_entropy`, `log_softmax`, and the `_with_logits` family at all.

**The KL blow-up in RLHF and DPO.** Preference-tuning methods compute a log-ratio between the policy and a reference model, $\log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$, and the DPO loss exponentiates a scaled version of it inside a sigmoid. When the policy drifts far from the reference — exactly what happens when the learning rate is too high or the run is degenerating — that log-ratio grows large, and $\exp(\beta \cdot \text{ratio})$ overflows. The signature is a KL curve that climbs steeply just before the loss NaNs, which is why people training with DPO watch the reward and KL curves as obsessively as the loss. The guardrails are the standard ones lifted to this setting: clamp the log-ratio to a sane range before the exponential, keep the reference-model log-probs in fp32, and lower the learning rate when the KL climbs. This is a numerics symptom (overflow) of an optimization disease (the policy diverging), and it sits at the intersection of this post and [debugging RLHF, DPO, and preference tuning](/blog/machine-learning/debugging-training/debugging-rlhf-dpo-and-preference-tuning) — read the KL curve, then apply the clamp.

These five cover the overwhelming majority of NaNs I have seen in production training, across vision, language, speech, and preference tuning. Recognize the signature — `inf` versus NaN, the step it appears, the dtype that triggers it, the op the backtrace names — and you skip straight to the fix instead of staring at a flatlined dashboard wondering where the run went.

## 12. When this is (and isn't) a NaN-numerics bug

A decisive section, because misattributing a NaN sends you hunting in the wrong place for hours.

**It IS a numerics bug** when the failure is a literal `nan` or `inf` value, when it appears smoothly (clean then NaN) rather than at step 1, when it is dtype-sensitive (clean at fp32/bf16, NaN at fp16), and when an anomaly backtrace points at an unstable op (`log`, `div`, `exp`, `sqrt`, `softmax`). This is the home turf of this post: read the signature, bisect, read the op, fix the math, install the guardrail.

**It is NOT primarily a numerics bug — it is data —** when the NaN appears at the same data index every time, survives a switch to fp32 with `F.cross_entropy`, and an assert-finite at the dataloader fires. Then the math is innocent; you have a corrupt row, a bad label, or a target that a broken scaler sent to `inf`. Fix the pipeline.

**It is NOT a NaN bug at all — it is a plateau —** when people *say* "the loss is NaN" but mean "the loss is stuck." A loss flat at a constant value is not a NaN (check with `torch.isfinite`); it is a never-learning curve, which lives in the data/optimization/model places and is hunted with the overfit-one-batch test, not with anomaly mode. The discriminating check is one line: `print(torch.isfinite(loss))`. If it is `True`, stop reading this post and go read [the loss-curve diagnostic](/blog/machine-learning/debugging-training/reading-the-loss-curve-as-a-diagnostic) — a stuck-but-finite loss is a completely different animal.

**It is NOT purely numerics — it is optimization —** when a sharp loss *spike* precedes the NaN and a grad-norm spike precedes the loss spike. Then the proximate cause is an overflow, but the *root* cause is an exploding gradient from too-high a learning rate or a bad batch, and clamping logits will only delay the inevitable. Lower the LR, clip the gradient, or skip the bad batch — the [loss spikes post](/blog/machine-learning/debugging-training/loss-spikes-and-divergence) and [gradients exploding and vanishing post](/blog/machine-learning/debugging-training/gradients-exploding-and-vanishing) own this case. The NaN is a symptom; the optimization is the disease.

The meta-rule: a NaN is a numerics *event*, but its *root cause* can live in any of the six places. The hunt (signature, bisect, read the op) tells you which. Do not assume "NaN" means "add an epsilon" — sometimes it means "your learning rate is too high" or "row 8,412 is corrupt." The whole framing of [the bug taxonomy](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) applies: bisect to the right place before you touch code, and let [the capstone playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) be your checklist when you are deep in a 3 a.m. hunt.

## Key takeaways

- **A NaN spreads at the speed of one forward-backward pass.** One NaN in one logit becomes a NaN loss, a fully-NaN gradient, and — after one optimizer step — every parameter NaN. You must catch it at the *first* operation, because the corpse tells you nothing.
- **There are only seven sources.** `log(0)`/`log(neg)`, division by zero, `exp` overflow, `sqrt` of a negative, all-masked softmax (`0/0`), bad labels/indices, and fp16 overflow/underflow. A NaN is a multiple-choice question, not a mystery.
- **Read the signature before instrumenting.** Step-1 NaN = data or model code. Smooth-then-NaN = overflow or `log(0)` (numerics). Spike-then-NaN = optimization (LR/bad batch). Same-index NaN = a corrupt row. The timing narrows seven suspects to two for free.
- **Bisect by step, then by layer.** Binary-search the first-NaN step over checkpoints ($\log_2 N$ replays), save the clean checkpoint just before, then localize the op with `set_detect_anomaly(True)` for backward NaNs and `isfinite` forward hooks for forward NaNs.
- **`inf` upstream of a `NaN` is the real source.** Most NaNs are born as `inf` (overflow) and become NaN one op later via `inf - inf` or `0 * inf`. Find the `inf`.
- **Use the stable loss formulations — this is the highest-leverage fix.** Pass *logits* to `F.cross_entropy` and `F.binary_cross_entropy_with_logits`; never `torch.log(torch.softmax(...))`. The framework already solved the numerics; reimplementing it is how you reintroduce the bug.
- **Forward hooks tell you the source vs. the victim.** The `input finite=` field is the whole game: finite input + non-finite output means *this* op is the source; non-finite input means the source is upstream.
- **Anomaly mode finds the bugs forward checks cannot.** `sqrt(0)` and `log` have finite forwards but infinite backwards; `set_detect_anomaly(True)` catches the NaN gradient and prints the forward traceback of the offending op.
- **Guardrails are nearly free and turn step-4,000 corruption into a step-1 catch.** Assert-finite inputs/labels, stable losses, clamped logits, epsilon denominators, and a per-step `isfinite` check cost microseconds and save hundred-dollar runs.
- **A NaN is an event; its root cause is in one of the six places.** Sometimes "NaN" means "add an epsilon," sometimes "lower the LR," sometimes "row 8,412 is corrupt." The hunt tells you which — do not reach for an epsilon reflexively.

## Further reading

- **"Mixed Precision Training," Micikevicius et al., 2018 (arXiv:1710.03740).** The origin of loss scaling and the canonical analysis of fp16's representable range and gradient underflow. Read it before you debug any AMP run.
- **PyTorch Automatic Mixed Precision docs and the Autograd Anomaly Detection docs** (`torch.autograd.set_detect_anomaly`, `torch.amp.GradScaler`). The official reference for the two tools at the center of this post.
- **IEEE 754-2019, the floating-point standard.** The definitions of `inf`, `NaN`, and their arithmetic (`0*inf=NaN`, `inf-inf=NaN`, `NaN!=NaN`) are not PyTorch quirks; they are the standard, and understanding them is understanding why NaNs spread.
- **"FlashAttention," Dao et al., 2022, and the `F.scaled_dot_product_attention` docs.** Why fused attention kernels handle the all-masked-row edge case (and other numerics) correctly where a naive softmax does not.
- [A taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) — the master symptom-to-suspect-to-test decision tree that this NaN hunt instantiates in the numerics place.
- [The training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) — the capstone checklist for the 3 a.m. hunt, including the bisection method and the guardrails to build in from day one.
- [Mixed-precision debugging: fp16 vs. bf16](/blog/machine-learning/debugging-training/mixed-precision-debugging-fp16-vs-bf16) and [loss spikes and divergence](/blog/machine-learning/debugging-training/loss-spikes-and-divergence) — the two siblings a NaN most often turns out to be once you read its signature.

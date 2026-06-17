---
title: "Gradient Accumulation and Effective-Batch Bugs: When Accumulation Isn't a Bigger Batch"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Learn why gradient accumulation should be mathematically identical to a bigger batch, the exact bugs that break that equivalence, and the one unit test that proves your accumulated update matches the big-batch update to numerical precision."
tags:
  [
    "debugging",
    "model-training",
    "gradient-accumulation",
    "effective-batch-size",
    "distributed-training",
    "finetuning",
    "deep-learning",
    "pytorch",
    "optimization",
    "batch-normalization",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/gradient-accumulation-and-effective-batch-bugs-1.png"
---

A teammate sends you a plot on a Tuesday morning. They are finetuning a 7B language model, and they have done the responsible thing: their GPU only fits a micro-batch of 4 sequences, so instead of training on a tiny, noisy batch, they set `gradient_accumulation_steps=8` to get an effective batch of 32. The reasoning is textbook. Accumulate eight micro-batches' worth of gradients, then take one optimizer step, and you have the gradient of a batch of 32 — same as if you had a GPU eight times bigger. Except the run is broken. The loss starts at 2.4, drops for a few steps, then leaps to 7.9 around step 40 and prints `nan` shortly after. They have already swapped the optimizer, lowered the learning rate twice, and rewritten the data collator. None of it is the bug. The bug is that their training loop sums the eight micro-batch losses' gradients without dividing by eight, so the effective learning rate is eight times what they set, and an effective learning rate of `1.6e-4` on a pretrained 7B checkpoint is a guaranteed loss spike. You tell them to divide the loss by `accumulation_steps`. The loss drops smoothly to 0.7 and the model ships before lunch.

This is the central lie of gradient accumulation: it *looks* like a transparent memory trick, a way to pretend your GPU is bigger than it is, and most of the time it is exactly that. But "accumulation equals a bigger batch" is a precise mathematical claim with precise preconditions, and when one of those preconditions is violated, you do not get a crash — you get a run that is subtly or catastrophically wrong while every line of code looks reasonable. In the six-places framing this series is built on — a bug hides in **data, optimization, model code, numerics, systems, or evaluation** — accumulation bugs live mostly in *optimization* and *model code*, with a nasty branch into *systems* the moment you add multiple GPUs. They are insidious precisely because the symptom (a loss spike, a curve that does not match your single-GPU baseline, a finetune that forgot everything) points everywhere except the accumulation loop you wrote three weeks ago and stopped looking at.

![A workflow graph showing how four micro-batches feed an averaged-and-stepped optimizer update that matches a single big batch, with a danger branch where summed gradients make the learning rate four times too big](/imgs/blogs/gradient-accumulation-and-effective-batch-bugs-1.png)

This post does three things at once, because that is the contract of this series. The **science**: I will derive, in two lines of calculus, exactly why the gradient of a mean-loss over $N \times B$ examples equals the *average* (not the sum) of the $N$ micro-batch mean-gradients — so accumulation must average — and I will show precisely why BatchNorm breaks the equivalence no matter how carefully you average. The **practice**: runnable PyTorch code for the correct accumulation loop, the `no_sync()` pattern under `DistributedDataParallel`, and the way Hugging Face `Trainer` and `accelerate` handle all of this for you. And the **proof**: the definitive diagnostic — a unit test that compares the parameter update from one big batch against the update from $N$ accumulated micro-batches on the same data, which must match to numerical precision if your loop is correct, and a before-and-after where dividing the loss restores the expected loss curve. By the end you should be able to write an accumulation loop you trust, prove it equals a big batch, and recognize the six ways the equivalence breaks on sight.

A note before we start, because it is the single fact that prevents the most expensive class of accumulation bug: **gradient accumulation changes how many optimizer steps you take, and almost everything that "counts steps" — the learning-rate schedule, warmup, total-steps, logging cadence, checkpoint frequency — must count *optimizer* steps, not *micro*-steps.** A schedule that warms up over "the first 500 steps" will warm up four times too fast if it counts micro-steps at `accumulation_steps=4`, and your carefully tuned warmup evaporates. We will come back to this; it is the second-most-common accumulation bug after the normalization mistake, and it produces a curve that *almost* looks right, which makes it far harder to catch.

## 1. The symptom catalogue: how a broken accumulation loop looks

Before any theory, fix the signatures in your eye, because most of this post's value is recognizing them in two seconds from a plot. Accumulation bugs have a small number of distinct fingerprints, and they look different from each other.

The **loss spike that ends in NaN** is the normalization bug. You set what should be a safe learning rate, the loss drops for a handful of steps, and then it spikes — often in the first few optimizer steps, before the model has moved far — and either recovers into a higher plateau or runs off to `nan`. The tell is that lowering the learning rate by exactly your accumulation factor makes the problem vanish. If `accumulation_steps=8` and dividing your LR by 8 fixes it, you were summing gradients instead of averaging them: the loop was running at an 8× effective LR the whole time.

The **curve that is close but not right** is usually the schedule-counting bug. The run does not blow up; it just does not match your baseline. Your single-GPU, no-accumulation run reached 0.71 final loss, and your accumulated run reaches 0.78, or it reaches 0.71 but along a visibly different path — a warmup that ended too early, a learning rate that decayed before it should have. This one is dangerous because nothing screams "bug." You ship a slightly worse model and never know why.

The **train-looks-fine, eval-is-worse signature** can be BatchNorm. If your model has BatchNorm layers and you switched from a real batch of 128 to a micro-batch of 16 with `accumulation_steps=8`, your normalization statistics are now computed over 16 examples, not 128. They are noisier, the running statistics drift differently, and the model that "trained fine" generalizes worse — because for BatchNorm, accumulation was never a bigger batch.

The **8× GPUs, barely faster** signature is the DDP interaction. Your distributed accumulated run is correct numerically but crawls, because you are doing a full gradient all-reduce on every single micro-step instead of once per optimizer step. With `accumulation_steps=8`, that is 8× the communication you need, and on a bandwidth-bound model it can erase most of your scaling.

And the **off-by-one** signatures — stepping the optimizer every micro-step (so each micro-batch becomes its own tiny update, and accumulation does nothing), or forgetting to zero the gradients (so gradients pile up across windows unboundedly), or a wrong last partial window at epoch end — each produce their own flavor of "this should work but doesn't." We will dismantle all of them. Let us start with the one that matters most: why averaging, not summing, is the whole game.

## 2. Why accumulation exists: the activation-memory math

Before the bugs, it helps to know exactly what accumulation buys you, because the *reason* it works is the same reason its bugs are subtle. The whole point of gradient accumulation is **memory**, and to see what it saves, you have to know where training memory goes. A training step holds four kinds of tensors in GPU memory: the **parameters** (the weights), the **gradients** (one number per parameter), the **optimizer state** (for AdamW, two extra numbers per parameter — the first and second moment estimates), and the **activations** (the intermediate forward-pass tensors saved for the backward pass). The first three scale with the *model* size and are completely independent of the batch size. The fourth — activations — scales **linearly with the batch size**, because every example in the batch has its own forward-pass activations that must be kept around until the backward pass consumes them.

That linear dependence is the crux. For a transformer, the activation memory per step is roughly proportional to $\text{batch} \times \text{seq\_len} \times \text{layers} \times \text{hidden}$, and for large models at long sequence lengths, activations dominate total memory — often more than parameters, gradients, and optimizer state combined. So when your GPU runs out of memory at batch 32, it is almost always the *activations* that overflowed, not the model. Accumulation exploits exactly this: it processes the batch in $N$ small chunks, one at a time, and **only one chunk's activations are alive at once.** You run a micro-batch of 4, compute its loss, backward through it (which frees its activations as it goes), and the gradient — which is the same size as the parameters, batch-independent — accumulates into `.grad`. Then you do the next micro-batch. The peak activation memory is set by the *micro*-batch of 4, but the gradient you end up with is the big batch of 32's gradient. You traded time (8 sequential forward/backward passes instead of 1) for memory (1/8 the activation footprint).

This is why accumulation is "a bigger batch for free" — but the "free" is the trap. It is free *for the gradient*, because the gradient is a per-example sum that does not care whether the examples arrived together or in chunks. It is *not* free for anything that needs to see the whole batch at once inside a single forward pass — which is precisely the BatchNorm problem we will reach in section 8. The activation-memory math tells you both why accumulation is indispensable on memory-constrained hardware and, by the same token, exactly which operations it cannot help: anything that couples examples *within* a forward pass, because accumulation deliberately breaks the batch into separate forward passes to save the activation memory.

#### Worked example: the memory you actually save

Concretely, take a 1.3B-parameter model finetuned in mixed precision with AdamW. The model-size-dependent memory is fixed regardless of batch: parameters in bf16 are about 2.6 GB, gradients about 2.6 GB, and AdamW's two moment estimates plus an fp32 master copy of the weights add roughly $1.3\times10^9 \times (4 + 4 + 4) = 15.6$ GB — call the batch-independent total about 21 GB. Now the activations. Suppose at batch 32 and sequence length 2,048 the activations need about 40 GB. Total: 61 GB, which overflows an 80 GB card once you add fragmentation and workspace — or simply will not fit a 48 GB card at all. Drop to a micro-batch of 4 with `accumulation_steps=8`: activations fall to about $40 \times 4/32 = 5$ GB, total about 26 GB, which fits comfortably on a 48 GB card with headroom. You have not changed the model, the optimizer state, or the effective batch the gradient sees — you have only shrunk the live-activation footprint by 8×. That is the entire value proposition, and it is why accumulation is the default memory escape hatch before you reach for gradient checkpointing or sharding. (For the full accounting of where memory goes and how to read a memory snapshot, the out-of-memory debugging post in this series goes deep; accumulation is one lever in that toolbox.)

## 3. The science: why accumulation must average, not sum

Here is the entire mathematical content of gradient accumulation in two lines, and once you have it, every normalization bug becomes obvious.

Take a loss that is a **mean** over a batch — which is the default for nearly every loss function in PyTorch (`reduction='mean'`). For a batch $\mathcal{B}$ of size $|\mathcal{B}|$, with per-example loss $\ell(x_i; \theta)$ and parameters $\theta$, the batch loss is

$$
L_{\mathcal{B}}(\theta) = \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \ell(x_i; \theta).
$$

Now suppose we want the gradient of the loss over a *big* batch of size $N \cdot B$, formed by concatenating $N$ micro-batches $\mathcal{B}_1, \dots, \mathcal{B}_N$, each of size $B$. The big-batch loss is the mean over all $N \cdot B$ examples:

$$
L_{\text{big}}(\theta) = \frac{1}{N B} \sum_{k=1}^{N} \sum_{i \in \mathcal{B}_k} \ell(x_i; \theta).
$$

Gradient is linear, so the gradient of the big-batch loss is

$$
\nabla_\theta L_{\text{big}} = \frac{1}{N B} \sum_{k=1}^{N} \sum_{i \in \mathcal{B}_k} \nabla_\theta \ell(x_i; \theta) = \frac{1}{N} \sum_{k=1}^{N} \underbrace{\left( \frac{1}{B} \sum_{i \in \mathcal{B}_k} \nabla_\theta \ell(x_i; \theta) \right)}_{\nabla_\theta L_{\mathcal{B}_k}} = \frac{1}{N} \sum_{k=1}^{N} \nabla_\theta L_{\mathcal{B}_k}.
$$

Read that last equality carefully, because it is the whole post. **The gradient of the mean-loss over the big batch equals the *average* of the $N$ micro-batch mean-gradients.** Not the sum — the average. If you compute each micro-batch's mean-loss, call `.backward()` on it, and let PyTorch accumulate the gradients into `.grad` (which it does by adding, that is `+=`), then after $N$ micro-steps your `.grad` holds

$$
\sum_{k=1}^{N} \nabla_\theta L_{\mathcal{B}_k} = N \cdot \nabla_\theta L_{\text{big}},
$$

which is $N$ times too large. Take an SGD step with learning rate $\eta$ and you move by $-\eta \cdot N \cdot \nabla_\theta L_{\text{big}}$ — exactly as if you had used learning rate $N\eta$ on the correct big-batch gradient. That is the normalization bug, stated as an equation. **The effective learning rate is $N$ times what you set.**

There are two equivalent fixes, and you should pick one and be consistent. The first, and the one I recommend, is to **divide each micro-batch loss by $N$ before calling `.backward()`**:

$$
\text{loss}_k \leftarrow \frac{L_{\mathcal{B}_k}}{N}, \qquad \sum_{k=1}^{N} \nabla_\theta \left( \frac{L_{\mathcal{B}_k}}{N} \right) = \frac{1}{N}\sum_{k=1}^{N} \nabla_\theta L_{\mathcal{B}_k} = \nabla_\theta L_{\text{big}}.
$$

Now the accumulated `.grad` is exactly the big-batch gradient, and you can use the same learning rate you would use for the big batch. The second fix is to leave the loss alone (so you accumulate the sum, $N \cdot \nabla L_{\text{big}}$) and **divide the learning rate by $N$** instead. These are algebraically identical for plain SGD. They are *not* identical for adaptive optimizers, and that subtlety is important enough to deserve its own section — for now, internalize the safe default: **divide the loss by `accumulation_steps`, keep your learning rate as-is.**

![A two-column before-and-after figure contrasting summed gradients that make the effective learning rate four times too big against averaged gradients that match the big batch](/imgs/blogs/gradient-accumulation-and-effective-batch-bugs-2.png)

#### Worked example: the factor-of-N spike in numbers

Suppose you finetune with a target effective batch of 32, a micro-batch of 4, so `accumulation_steps=8`, and a learning rate of `2e-5` — a perfectly reasonable rate for a 7B finetune. Your loss is the default cross-entropy with `reduction='mean'`, so each micro-batch produces a mean-loss of order 2.0 and a mean-gradient of some norm, call it $g$. You call `.backward()` on each of the eight micro-batch losses without dividing, then `optimizer.step()`.

After eight micro-steps, `.grad` holds $\approx 8g$ instead of $g$. Your AdamW update, which for a fresh moment estimate moves roughly $\eta \cdot \text{sign}(\text{grad})$ in magnitude per coordinate early on (Adam's update is scale-normalized, which actually *masks* the pure factor-of-8 you would see under SGD — more on that below), still sees a gradient signal eight times larger feeding its first and second moments. Under SGD the update would be $-2\times10^{-5} \cdot 8g = -1.6\times10^{-4} \cdot g$ — an effective LR of `1.6e-4`, which on a pretrained checkpoint is firmly in spike territory. The loss drops from 2.4 to 2.1 for a few steps as the moment estimates warm up, then leaps to 7.9 as the over-large updates destroy the pretrained features, and `nan` follows once an activation overflows. Divide the loss by 8 and the effective LR is back to `2e-5`; the loss descends smoothly to 0.7. One division, an 8× change in the most important hyperparameter, and the difference between a wasted run and a shipped model.

The deeper lesson: the normalization bug is a **learning-rate bug in disguise**, which is why it presents with the classic too-high-LR signature (early spike, then divergence) and why people waste days on the optimizer and data when the real fix is one line in the loss computation. If you have read [the learning rate is almost always the problem](/blog/machine-learning/debugging-training/the-learning-rate-is-almost-always-the-problem), you already know that signature — accumulation gives you a *second* way to produce it without ever touching the `lr` argument.

## 4. The correct accumulation loop, line by line

Let us write the loop that is provably correct, then dissect every line, because each one is a place a bug hides. This is the bare PyTorch version; the HF `Trainer` and `accelerate` versions come later, and they exist precisely so you do not have to get this right by hand.

```python
import torch

ACCUM_STEPS = 8                      # micro-batches per optimizer step
model.train()
optimizer.zero_grad(set_to_none=True)

for step, batch in enumerate(dataloader):
    inputs, targets = batch
    inputs, targets = inputs.to(device), targets.to(device)

    outputs = model(inputs)
    loss = loss_fn(outputs, targets)        # reduction='mean' over the micro-batch
    loss = loss / ACCUM_STEPS               # <-- THE division that makes accum == big batch
    loss.backward()                         # grads ACCUMULATE into .grad (PyTorch does +=)

    if (step + 1) % ACCUM_STEPS == 0:       # only on the window boundary
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()                    # ONE optimizer step per window
        scheduler.step()                    # schedule counts OPTIMIZER steps
        optimizer.zero_grad(set_to_none=True)  # reset for the next window
```

Now the dissection. **`loss = loss / ACCUM_STEPS`** is the section-3 fix; without it you run at `ACCUM_STEPS`× the LR. Note it must come *before* `backward()` — dividing the loss scales the gradient by the same factor by linearity of differentiation. **`loss.backward()` every micro-step** is correct: this is how gradients accumulate. PyTorch's `backward()` adds the new gradient to whatever is already in `.grad`; it does not overwrite. That additive behavior is the entire mechanism of accumulation, and it is also why forgetting to zero is catastrophic.

**`if (step + 1) % ACCUM_STEPS == 0`** is the window boundary. Everything inside — clip, step, schedule, zero — happens *once per window*, not per micro-step. Use `(step + 1)` not `step` so the boundary lands after micro-batches 0–7 (the 8th), not after 0–6. **`clip_grad_norm_` goes here, after the full window accumulates**, so it clips the *complete* big-batch gradient. Clipping per micro-step would clip each partial gradient independently and change the math — a subtle bug we will revisit. **`scheduler.step()` here** is the optimizer-step-counting fix: the schedule advances once per optimizer step. **`optimizer.zero_grad(set_to_none=True)` after stepping** clears the accumulator for the next window. The `set_to_none=True` is a small memory and speed win; functionally it makes `.grad` `None` rather than a zero tensor.

![A vertical stack showing the five stages of a correct accumulation loop from forward and divide to step and zero, with a danger note that stepping every micro-step or never zeroing breaks it](/imgs/blogs/gradient-accumulation-and-effective-batch-bugs-6.png)

Three off-by-one bugs hide in this structure, and each one looks like reasonable code:

**Stepping every micro-step.** If you indent `optimizer.step()` and `zero_grad()` out of the `if` block so they run every iteration, you take eight tiny updates per window instead of one. Accumulation does nothing — you are just training with micro-batch 4 and learning rate `2e-5`, which is a noisier, smaller-batch run than you intended. The symptom is a noisier loss curve than your baseline and a schedule that runs `ACCUM_STEPS`× too fast. The fix is the `if` block.

**Never zeroing.** If you forget `zero_grad()` entirely, gradients accumulate across *every* window forever. After 10 windows your `.grad` is the sum of 80 micro-batch gradients, and your effective LR climbs without bound. The symptom is a loss that diverges progressively — fine at first, worse every window. The fix is to zero after each step.

**Zeroing every micro-step.** The mirror image: if you put `zero_grad()` inside the loop body but outside the `if`, you wipe the gradient before it can accumulate, so each window's update reflects only its *last* micro-batch. Accumulation is silently disabled again. The symptom is identical to stepping-every-micro-step from the optimizer's view. The fix is to zero only on the boundary.

## 5. The definitive diagnostic: prove accumulation equals a big batch

Every other check in this post is a heuristic. This one is a proof. If accumulation is correct, then the parameter update produced by **one optimizer step on a single big batch of $N \cdot B$ examples** must equal the update produced by **$N$ accumulated micro-batches of $B$ examples drawn from the same data**, to numerical precision. They are the same gradient by the section-3 derivation, so the same optimizer applied to the same starting parameters must land in the same place. Write that as a unit test and you never have to wonder whether your loop is right again.

```python
import copy
import torch

def assert_accumulation_equals_big_batch(model, loss_fn, big_batch, accum_steps, lr=1e-3, tol=1e-5):
    """Prove that N accumulated micro-batches == one big batch update, to numerical precision."""
    inputs, targets = big_batch
    micro = inputs.shape[0] // accum_steps
    assert inputs.shape[0] % accum_steps == 0, "big batch must split evenly into micro-batches"

    # ---- Path A: one optimizer step on the whole big batch ----
    model_a = copy.deepcopy(model)
    opt_a = torch.optim.SGD(model_a.parameters(), lr=lr)   # SGD: no moment state to confound
    opt_a.zero_grad(set_to_none=True)
    loss_a = loss_fn(model_a(inputs), targets)             # mean over N*B
    loss_a.backward()
    opt_a.step()

    # ---- Path B: N accumulated micro-batches, loss divided by accum_steps ----
    model_b = copy.deepcopy(model)
    opt_b = torch.optim.SGD(model_b.parameters(), lr=lr)
    opt_b.zero_grad(set_to_none=True)
    for k in range(accum_steps):
        sl = slice(k * micro, (k + 1) * micro)
        loss_k = loss_fn(model_b(inputs[sl]), targets[sl]) / accum_steps   # the divide
        loss_k.backward()
    opt_b.step()

    # ---- Compare the resulting parameters ----
    max_diff = 0.0
    for (na, pa), (nb, pb) in zip(model_a.named_parameters(), model_b.named_parameters()):
        d = (pa - pb).abs().max().item()
        max_diff = max(max_diff, d)
    assert max_diff < tol, f"accumulation != big batch, max param diff = {max_diff:.2e}"
    print(f"PASS: accumulation matches big batch (max param diff {max_diff:.2e})")
    return max_diff
```

A few deliberate design choices make this test sharp. **Use plain SGD, not Adam**, because SGD's update is a clean function of the gradient with no accumulated optimizer state — it isolates the gradient math, which is what you are testing. If you test under Adam you will see *near* equality but not to `1e-5`, for reasons we cover in section 7, and you will not know whether the residual is the optimizer or a real bug. **Use the same big batch split into micro-batches**, not random batches, so the only thing varying is the accumulation mechanism. **Put the model in `eval()` mode** if it contains BatchNorm or Dropout (add `model_a.eval(); model_b.eval()` before the forward passes), because those layers are stochastic or batch-dependent and will break the equivalence by design — we want to test the *gradient* path in isolation here, then test BatchNorm separately. **Set `tol` around `1e-5` for fp32**; under fp32 the two paths should agree to roughly single-precision rounding, and a difference of `1e-2` is a real bug while `3e-6` is just floating-point noise.

This test catches every normalization bug instantly. Run it on a buggy loop (no divide) and you will see `max param diff` around the magnitude of `(N-1)` times one SGD step — large and unmistakable. Run it on the correct loop and it passes at `1e-6`. Make this an assertion in your test suite and the most expensive accumulation bug in the field becomes a red CI light instead of a wasted week of compute.

#### Worked example: reading the test output

Concretely, run the test on a small MLP with `accum_steps=4`, `lr=1e-3`, fp32, BatchNorm off. The correct loop prints `PASS: accumulation matches big batch (max param diff 4.71e-07)` — that is single-precision noise, exactly what you want. Now delete the `/ accum_steps` line to simulate the summing bug and rerun: it prints `accumulation != big batch, max param diff = 3.10e-03`. That `3.1e-3` is approximately three SGD steps' worth of extra movement (because the summed gradient is 4× the averaged one, so path B moves 4× as far, and the difference from path A is the extra 3× on a `1e-3` step scaled by the gradient norm). The magnitude tells you the *flavor*: a diff near `(N-1)×` your step scale is the summing bug; a tiny but nonzero diff under Adam is optimizer state; a diff that only appears with BatchNorm on is the normalization-layer issue. The test does not just say "broken" — its number is a diagnostic.

## 6. The schedule bug: optimizer steps vs micro-steps

The normalization bug is loud. The schedule bug is quiet, which makes it the more dangerous of the two in production. Here is the mechanism. Your learning-rate schedule — warmup, cosine decay, linear decay, whatever — is parameterized by a *step count*: "warm up over the first 500 steps, then cosine-decay to zero over 10,000 total steps." The question is: 500 of *what*? The schedule should advance once per **optimizer step**, because an optimizer step is when the learning rate is actually applied. But if you call `scheduler.step()` once per **micro-step** — for example because you left it inside the loop body instead of inside the `if` block — then with `accumulation_steps=4` your warmup finishes in 125 optimizer steps instead of 500, and your "10,000-step" cosine schedule reaches zero LR in 2,500 optimizer steps and then trains the rest of the run at LR ≈ 0, learning nothing for three quarters of the run.

The math is trivial but the consequences are not. If `micro_steps = accum_steps × optimizer_steps`, then any schedule defined in optimizer-step units that is advanced per micro-step runs `accum_steps`× too fast. Warmup, total steps, decay length — all of it compresses by the accumulation factor. The symptom is the section-1 "close but not right" curve, the one a warmup that ends abruptly early, then a learning rate that decays sooner than your baseline, then a long tail of near-zero LR. The loss often still goes down — slowly — so nothing crashes; you just get a meaningfully worse model and a curve that does not line up with your single-GPU run.

![A timeline of one accumulation window showing three micro-steps that do not step, a final micro-step that steps and zeroes, and a danger versus success contrast between counting micro-steps and counting optimizer steps for the schedule](/imgs/blogs/gradient-accumulation-and-effective-batch-bugs-3.png)

The fix has two parts that must agree. First, **call `scheduler.step()` inside the optimizer-step boundary**, right after `optimizer.step()`, so it advances once per window — exactly as in the section-4 loop. Second, **compute your schedule's parameters in optimizer-step units.** If you want a 500-optimizer-step warmup, configure the warmup for 500, and make sure your total-steps is `len(dataloader) × epochs / accumulation_steps`, not `len(dataloader) × epochs`. This second part trips people up constantly: they correctly move `scheduler.step()` into the `if` block but still compute `num_training_steps = len(dataloader) * num_epochs`, which over-counts by `accumulation_steps` and makes the cosine schedule decay too *slowly* — the LR never reaches its scheduled minimum. Both halves matter.

```python
# Compute the schedule in OPTIMIZER-step units, not micro-step units.
num_update_steps_per_epoch = math.ceil(len(dataloader) / ACCUM_STEPS)
max_train_steps = num_epochs * num_update_steps_per_epoch    # OPTIMIZER steps

from transformers import get_cosine_schedule_with_warmup
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.03 * max_train_steps),   # 3% warmup, in OPTIMIZER steps
    num_training_steps=max_train_steps,             # OPTIMIZER steps
)
# ...and call scheduler.step() only inside the `if (step+1) % ACCUM_STEPS == 0` block.
```

The same "count optimizer steps" rule applies to **logging, checkpointing, and early stopping.** If you log loss every "100 steps" counting micro-steps, your logging cadence is `accum_steps`× denser than you think; if you checkpoint every "1000 steps," likewise. None of these are correctness bugs in the model, but they confuse your reading of the run, and a confused reading is how a real bug hides longer. The discipline: pick *optimizer steps* as your unit of time everywhere, and convert micro-step counts to it explicitly.

This is also why the `Trainer`'s and `accelerate`'s handling of accumulation is worth using — they define the global step as the optimizer step, count the schedule against it, and report progress in optimizer steps, so the whole class of "500 of what?" bugs disappears. More on that in section 10.

## 7. The adaptive-optimizer subtlety: why divide-the-loss beats divide-the-LR

I claimed in section 3 that dividing the loss by $N$ and dividing the learning rate by $N$ are algebraically identical. That is true for plain SGD. It is *false* for Adam, AdamW, RMSProp, and every other optimizer that adapts its step to the gradient's magnitude — and the difference is the reason "divide the loss" is the safer default.

Here is why. Adam does not use the raw gradient $g$; it uses $g / (\sqrt{v} + \epsilon)$, where $v$ is a running estimate of the squared gradient. To first order, when the gradient is consistent, $\sqrt{v} \approx |g|$, so the update is roughly $\eta \cdot \text{sign}(g)$ — **scale-invariant in the gradient.** Multiply every gradient by a constant $c$ and Adam's update barely changes, because $c g / \sqrt{c^2 v} = g/\sqrt{v}$. This has a startling consequence for accumulation: if you make the summing bug (gradients $N$× too large) but use Adam, the *step size* is largely unaffected, because Adam normalizes the scale away. The factor-of-$N$ does not show up as an $N$× learning rate the way it does under SGD.

That sounds like Adam protects you. It does not — it protects you *imperfectly*, in a way that is worse than a clean failure. The scale-invariance is only approximate: it breaks during warmup (when $v$ is biased toward zero), it breaks for the $\epsilon$ term (which is *not* scaled, so a larger gradient makes $\epsilon$ relatively negligible and changes the effective step on small-gradient coordinates), and it interacts with **decoupled weight decay** in AdamW (the decay term is independent of the gradient scale, so changing the gradient magnitude changes the *balance* between the gradient step and the decay step). The net effect is that the summing bug under AdamW produces a *subtly* mistuned run — not a clean $N$× spike, but a model that trains "fine" and lands a point or two worse, with weight decay effectively too weak. That is harder to detect than the SGD spike, not easier.

The contrast with the two fixes: **divide the loss by $N$** gives Adam the *correct* gradient ($\nabla L_{\text{big}}$), so its moment estimates, its $\epsilon$ scaling, and its weight-decay balance are all exactly what they would be for the big batch — the equivalence holds as well as it can under any optimizer. **Divide the LR by $N$** leaves Adam the $N$×-too-large gradient and tries to compensate with a smaller learning rate, but because Adam's step is scale-invariant, dividing the LR does *not* undo the gradient inflation the way it does under SGD — you end up with a smaller LR applied to a scale-normalized step, which is just a smaller LR, not a corrected gradient. So the two "fixes" are not equivalent under Adam, and only the divide-the-loss fix is actually correct.

| Aspect | Divide loss by N (recommended) | Divide LR by N | No fix (sum) |
| --- | --- | --- | --- |
| SGD effective LR | correct (1×) | correct (1×) | N× too big — spike/NaN |
| Adam gradient seen | correct $\nabla L_{\text{big}}$ | N× inflated | N× inflated |
| Adam moment estimates | match big batch | scaled, biased in warmup | scaled, biased in warmup |
| AdamW weight-decay balance | correct | distorted (decay too weak) | distorted (decay too weak) |
| Passes the section-5 SGD test | yes | yes (algebraically same) | no |
| Passes an Adam equivalence test | yes (to fp precision) | no | no |

The practical rule is one line: **always divide the loss by `accumulation_steps`; never compensate with the learning rate.** It is correct under every optimizer, it is what HF `Trainer` and `accelerate` do internally, and it keeps your learning rate meaning the same thing whether you accumulate or not — which matters enormously when you change `accumulation_steps` to fit a new GPU and expect the rest of your config to stay valid.

## 8. BatchNorm: the equivalence that cannot be saved

Everything so far has been fixable — divide the loss, count optimizer steps, and accumulation *is* a bigger batch for the gradient. But there is one place where the equivalence is mathematically impossible to recover, no matter how carefully you average: **BatchNorm.** Understanding exactly why is the most important conceptual payoff of this post, because it tells you when accumulation is and is not a valid substitute for a real big batch.

BatchNorm normalizes each feature using the **mean and variance computed across the batch dimension** of the current forward pass. For a feature with activations $\{a_i\}$ over a batch, BN computes $\mu = \frac{1}{m}\sum_i a_i$ and $\sigma^2 = \frac{1}{m}\sum_i (a_i - \mu)^2$ over the $m$ examples *in that forward pass*, then normalizes $\hat{a}_i = (a_i - \mu)/\sqrt{\sigma^2 + \epsilon}$. Here is the crux: **under accumulation, each forward pass only sees the micro-batch.** With `accumulation_steps=8` and a micro-batch of 16, every BN layer computes its statistics over **16 examples**, not the 128 that a real big batch would give it. You run eight separate forward passes, each normalized by its own 16-example statistics, and there is no point in the computation where BN ever sees all 128 examples together. Averaging the *gradients* afterward cannot fix this, because the normalization already happened, per micro-batch, inside each forward pass — the damage is baked into the activations before any gradient exists.

The numerical consequence: BN's statistics are noisier estimates of the true feature distribution at smaller $m$. The variance of the estimated mean scales as $\sigma^2/m$, so going from $m=128$ to $m=16$ makes your normalization statistics roughly $\sqrt{128/16} = \sqrt{8} \approx 2.8\times$ noisier per step. The running statistics (the exponential moving averages BN keeps for eval mode) are updated from these noisier per-micro-batch estimates, so they converge to a different, noisier place than they would with a true big batch. The model that "trained fine" under accumulation can generalize measurably worse at eval, and worse, the gap is invisible during training because BN uses *batch* statistics in `train()` mode and only switches to *running* statistics in `eval()` mode — the train-looks-fine, eval-is-worse signature from section 1.

![A four-row matrix comparing gradient, BatchNorm statistics, dropout masks, and memory across micro-size, accumulated, and big-batch columns, with a match column marking that only the gradient matches](/imgs/blogs/gradient-accumulation-and-effective-batch-bugs-4.png)

So what do you do? Three options, in order of preference depending on your situation:

**Switch to a normalization that does not depend on the batch.** `GroupNorm`, `LayerNorm`, and `InstanceNorm` all compute their statistics *within each example*, across feature or spatial dimensions, never across the batch. For these layers, **accumulation is exactly a bigger batch**, because each example's normalization is independent of how many other examples share the forward pass. This is one of the underappreciated reasons modern architectures (transformers with LayerNorm, ConvNeXt with LayerNorm, many detection backbones with GroupNorm) are friendlier to memory-constrained training than classic BatchNorm CNNs — accumulation just works for them. If you are designing for accumulation or small batches, prefer a per-example normalization.

**Use SyncBatchNorm if your "big batch" is really across GPUs.** `torch.nn.SyncBatchNorm` computes BN statistics across *all ranks* in a distributed group, so a batch sharded over 8 GPUs is normalized as one batch of $8 \times m$. But note carefully: **SyncBN synchronizes across GPUs, not across accumulation steps.** It does nothing for accumulation on a single GPU — there is no "other forward pass" to sync with, because the micro-steps are sequential, not parallel. SyncBN fixes the multi-GPU small-per-rank-batch problem; it does not fix the accumulation problem. This distinction confuses people constantly.

**Keep the micro-batch large enough for BN to be stable.** If you must use BatchNorm, the practical floor is around 16–32 examples per forward pass before the statistics get too noisy; below that, BN degrades regardless of accumulation. If your micro-batch is 4, BN is already shaky on its own, and accumulation will not save it. Increase the micro-batch (at the cost of accumulating fewer steps) or switch normalization.

The single sentence to remember: **accumulation is a bigger batch for everything that is computed per-example and summed, and is *not* a bigger batch for anything computed across the batch dimension.** Gradients are summed per-example, so they average correctly. BN statistics are computed across the batch, so they do not. Dropout, which we turn to next, is a milder version of the same theme.

## 9. Dropout and stochastic layers: a smaller, benign mismatch

BatchNorm breaks the equivalence hard. Dropout breaks it softly, and the difference is worth understanding so you know which mismatches to worry about and which to ignore.

Dropout works by sampling, on each forward pass, a random binary mask that zeros a fraction $p$ of activations (and scales the survivors by $1/(1-p)$ to keep the expected magnitude constant). The randomness is per-forward-pass. Now compare: a **single big batch of 128** gets *one* set of dropout masks applied across all 128 examples in one forward pass. Under **accumulation of 8 micro-batches of 16**, you run 8 forward passes, each drawing its *own independent* dropout mask. So the accumulated run uses 8 independent mask draws where the big-batch run used 1.

Is that a bug? No — it is a difference, and a benign one. Dropout is a regularizer whose effect is defined *in expectation* over the mask distribution; both setups sample from the same distribution, just with different granularity. If anything, the accumulated run sees slightly *more* mask diversity per optimizer step (8 draws vs 1), which is a marginally stronger, marginally different regularization. Neither is "correct"; they are two valid samples from the same stochastic training process. The practical consequence is that your section-5 equivalence test will **not** pass to `1e-5` if dropout is active and the two paths draw different masks — which is exactly why I told you to set the model to `eval()` (or seed the RNG identically per micro-batch) when running that test. The test is checking the deterministic gradient path; dropout's stochasticity is not a bug to fix, it is noise to control for during the test.

The one place dropout *does* bite is **reproducibility and the equivalence proof**, not model quality. If you want your accumulation run to be bit-for-bit reproducible, you must manage the RNG state carefully across micro-steps, exactly as you would across any forward passes — see [reproducibility and determinism in training](/blog/machine-learning/debugging-training/reproducibility-and-determinism-in-training) for the full treatment of seeding dataloader workers and CUDA ops. For the equivalence test specifically, the clean move is to run it in `eval()` mode with BN and dropout disabled, prove the *gradient math* is right, and then trust that the stochastic layers are sampling correctly from a separate, simpler check. Do not chase a `1e-5` match with dropout on — you will never get it, and it does not mean your loop is broken.

If you genuinely want to prove the equivalence *with* dropout active — to convince yourself the gradient path is right even through the stochastic layer — the trick is to force both paths to draw the *same* masks. Seed the RNG identically right before each forward pass so the big-batch pass and the corresponding concatenation of micro-passes sample the same dropout pattern. In practice that means snapshotting and restoring the RNG state around the micro-loop, which is fiddly enough that disabling dropout for the test is almost always the better choice:

```python
import torch

def forward_with_fixed_dropout(model, inputs, seed):
    # Force the same dropout mask sequence regardless of how the batch is chunked.
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    out = model(inputs)
    torch.set_rng_state(cpu_state)              # restore so the rest of the loop is unaffected
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    return out
```

Even this only works cleanly when the dropout masks line up example-for-example between the two chunkings, which they generally do not unless you also control the order of mask draws within the forward pass. The honest takeaway: do not fight dropout's stochasticity in the equivalence proof. Disable it, prove the deterministic gradient math, and verify dropout separately by checking that the *expected* loss over many seeds matches between the two setups (it will, because both sample the same distribution). That is a statistical check, not a bit-exact one, and it is the right tool for a stochastic layer.

The general principle, stated once more because it unifies sections 8 and 9: **batch-coupled determinism breaks the equivalence; per-forward-pass stochasticity merely changes which valid sample you draw.** BN couples examples through shared statistics computed across the batch, so it is a real, unfixable-by-averaging problem. Dropout couples nothing across examples — each activation's mask is independent — so its only effect under accumulation is a different RNG draw, which leaves the training distribution unchanged. When you audit a suspicious accumulated run, sort its differences from the baseline into these two bins: anything batch-coupled (BN, and any custom layer that reduces across the batch dimension) is a real equivalence break to design around; anything per-example-stochastic (dropout, stochastic depth, augmentation noise applied inside the model) is benign sampling variance you should not waste time chasing.

## 10. Doing it right with Hugging Face Trainer and accelerate

If you are using the Hugging Face stack, you should almost never write the section-4 loop by hand, because `Trainer` and `accelerate` implement it correctly — including the loss division, the optimizer-step-counting schedule, and the DDP `no_sync` optimization we will cover next. The bugs move from "did I write the loop right" to "did I configure it right," which is a smaller surface. Here is `TrainingArguments`:

```python
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="out",
    per_device_train_batch_size=4,         # the MICRO-batch per GPU
    gradient_accumulation_steps=8,         # accumulate 8 -> effective 32 per GPU
    learning_rate=2e-5,                    # the LR for the EFFECTIVE batch; do NOT pre-divide
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,                     # 3% of OPTIMIZER steps (Trainer counts correctly)
    num_train_epochs=3,
    logging_steps=10,                      # these are OPTIMIZER (global) steps
    bf16=True,                             # prefer bf16 over fp16 for finetuning stability
)
```

The key facts about `Trainer`'s behavior: it computes the **effective batch size** as `per_device_train_batch_size × gradient_accumulation_steps × num_gpus`, and that is the batch your learning rate is for — you do *not* divide the LR yourself, and you do *not* divide the loss yourself (the trainer scales the loss by `1/gradient_accumulation_steps` internally before backward). Its `global_step` counts **optimizer steps**, so `logging_steps`, `save_steps`, `eval_steps`, `warmup_steps`, and `max_steps` are all in optimizer-step units — the section-6 schedule bug cannot happen. And under DDP it uses `no_sync()` for the non-final micro-steps automatically. The one thing to get right is *interpreting* the effective batch: if you change `gradient_accumulation_steps` from 8 to 16 to fit a smaller GPU, your effective batch doubles, and you may want to revisit the LR — but the *mechanism* stays correct.

There is a historical wrinkle worth knowing. In `transformers` and `accelerate` versions before late 2024, there was a real, shipped **loss-averaging bug for token-level losses under accumulation**: the loss was divided by `gradient_accumulation_steps` (correct for the *batch* average) but token-level cross-entropy with variable-length sequences needs to be normalized by the *total number of non-padding tokens across the window*, not by a fixed accumulation count, because micro-batches with more tokens should contribute proportionally more. Dividing by a fixed $N$ instead weights each micro-batch equally regardless of its token count, which subtly mis-averages when sequence lengths vary. This was fixed by switching to a token-count-weighted reduction; if you are on an older version and finetuning with variable-length sequences and accumulation, upgrade or set the documented flag. It is a perfect example of how "divide by N" is only correct when each micro-batch contributes equally — for token-level losses with ragged lengths, the correct denominator is the token count, not the batch count. This connects to [the loss masking bug](/blog/machine-learning/debugging-training/the-loss-masking-bug), where `-100` masking and per-token normalization decide what the denominator actually is.

The `accelerate` version of manual accumulation is the cleanest hand-rolled option, because it handles the loss scaling, the `no_sync`, and the optimizer-step boundary through a context manager:

```python
from accelerate import Accelerator

accelerator = Accelerator(gradient_accumulation_steps=8)
model, optimizer, dataloader, scheduler = accelerator.prepare(
    model, optimizer, dataloader, scheduler
)

for batch in dataloader:
    with accelerator.accumulate(model):       # handles loss scaling + no_sync + step boundary
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)            # accelerate scales by 1/accum internally
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

Inside `with accelerator.accumulate(model)`, `accelerate` knows whether the current iteration is a window boundary; on non-boundary micro-steps it makes `optimizer.step()` and friends no-ops and wraps the backward in `no_sync()` under DDP, and on the boundary it actually steps. You write `optimizer.step()` every iteration and it does the right thing. This is the recommended pattern if you need a custom loop but do not want to maintain the off-by-one logic yourself.

## 11. DDP and accumulation: the no_sync optimization

Now the systems branch, which is pure performance, not correctness — but a 4–8× communication waste is worth fixing. Under `DistributedDataParallel`, the entire point of DDP is that it **all-reduces the gradients across all ranks** so every GPU ends up with the averaged gradient and they stay in lockstep. DDP triggers this all-reduce automatically during `backward()`, using hooks on the gradients. That is exactly what you want for a normal training step. It is exactly what you do *not* want on the non-final micro-steps of an accumulation window.

Here is the problem. With `accumulation_steps=8`, a naive DDP accumulation loop calls `backward()` eight times per window, and DDP all-reduces on *every one of those eight backwards.* But you only need the gradients synchronized once — right before `optimizer.step()`, on the final micro-step. The first seven all-reduces are pure waste: each rank is synchronizing a *partial* accumulated gradient that will be added to more anyway, and the only sync that matters is the last one over the complete window's gradient. On a model where communication is a meaningful fraction of step time (large models, slower interconnect), doing 8× the all-reduces can erase most of the benefit of accumulating in the first place — you saved activation memory but spent it all on the wire.

![A dataflow graph showing non-final micro-steps routed through no_sync to a single all-reduce at the final micro-step, with a danger branch where skipping no_sync causes four all-reduces and three wasted](/imgs/blogs/gradient-accumulation-and-effective-batch-bugs-5.png)

The fix is `DistributedDataParallel`'s `no_sync()` context manager, which **disables the gradient all-reduce for the operations inside it.** You wrap every micro-step *except the last one in the window* in `no_sync()`; on the final micro-step you let DDP sync normally. The result is one all-reduce per optimizer step instead of `accumulation_steps`, with byte-for-byte identical math — `no_sync` only changes *when* the sync happens, not the final synchronized value, because gradient accumulation is additive and DDP's all-reduce is linear, so syncing once at the end over the sum equals syncing each piece and summing.

```python
model = DistributedDataParallel(model, device_ids=[local_rank])

for step, batch in enumerate(dataloader):
    is_last_micro = (step + 1) % ACCUM_STEPS == 0
    # no_sync on every micro-step EXCEPT the last -> 1 all-reduce per window, not ACCUM_STEPS
    sync_context = model.no_sync() if not is_last_micro else nullcontext()
    with sync_context:
        loss = loss_fn(model(batch.x), batch.y) / ACCUM_STEPS
        loss.backward()                    # all-reduce happens ONLY when NOT in no_sync (last micro)

    if is_last_micro:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
```

Two cautions. First, **clip *after* the final sync**, as shown — you want to clip the fully-reduced, fully-accumulated big-batch gradient, not a partial or unsynced one. Second, **`no_sync` and `find_unused_parameters=True` interact badly**: if your model has parameters that do not receive a gradient on some micro-steps, DDP's unused-parameter detection runs per-backward and can conflict with deferred syncing — prefer structuring the model so all parameters always get a gradient, and reach for `static_graph=True` if your graph is fixed. The full treatment of DDP gradient-sync correctness, `find_unused_parameters`, and rank desync is its own topic; this series covers it in the distributed-training post on debugging DDP and multi-GPU runs, which is the companion to this one — accumulation and DDP are the two halves of "how do I get a big effective batch when one GPU isn't enough."

#### Worked example: the communication cost of skipping no_sync

Put numbers on it. Suppose a 1.3B-parameter model, fp32 gradients, so the gradient tensor is about $1.3\times10^9 \times 4 = 5.2$ GB. On a ring all-reduce across 8 GPUs, each rank sends and receives roughly $2(n-1)/n \approx 1.75\times$ the gradient size, so about 9.1 GB of traffic per all-reduce per rank. At, say, 50 GB/s effective interconnect bandwidth, one all-reduce takes about 0.18 seconds. With `accumulation_steps=8` and *no* `no_sync`, you pay that 8 times per optimizer step — about 1.45 seconds of pure communication per step. With `no_sync`, you pay it once — 0.18 seconds. If your compute per optimizer step is, say, 0.6 seconds, then without `no_sync` you are at $0.6 + 1.45 = 2.05$ s/step (29% compute-bound), and with `no_sync` you are at $0.6 + 0.18 = 0.78$ s/step (77% compute-bound) — a 2.6× throughput win for one context manager, at a GPU cost of perhaps several dollars per hour per GPU, which over a multi-day run is real money saved. This is the kind of "8× GPUs, barely faster" mystery that turns out to be a missing `no_sync`.

## 12. The last partial window and other epoch-boundary bugs

One more correctness trap, and it is the one people skip because it only bites at the seams. Your dataloader has, say, 1,003 batches and `accumulation_steps=8`. That is 125 full windows of 8 micro-batches (1,000 micro-batches) plus a **partial window of 3** at the end. What happens to those last 3 micro-batches?

If your loop steps only on `(step + 1) % ACCUM_STEPS == 0`, then after micro-batch 1,003 — which is `step=1002`, and `1003 % 8 = 3`, not 0 — **you never step.** The gradients from those last 3 micro-batches accumulate into `.grad` and then, at the start of the next epoch, you either (a) zero them away (losing 3 micro-batches of signal, usually harmless) or (b) if you forgot to zero at epoch start, *add them to the first window of the next epoch* (contaminating it with stale gradients, a real bug). Worse, those 3 micro-batches were divided by 8, so even if you did step on them, they would be normalized as if there were 8 — a partial window of 3 divided by 8 underweights them by $3/8$. There is no single "right" answer, but there are right *choices*:

The simplest correct behavior is to **flush a partial window at epoch end**: if the loop ends mid-window with accumulated gradients, take an optimizer step on what you have. To make the normalization correct for the partial window, divide each of its micro-losses by the *actual* number of micro-steps in that window (3), not by `ACCUM_STEPS` (8). Most people do not bother — they just drop the partial window by zeroing at epoch start — and for large datasets where a partial window is a tiny fraction of an epoch, dropping it is fine. The bug is *not making a choice*: letting partial-window gradients silently leak into the next epoch's first step. Always zero at epoch start, or always flush; never neither.

```python
for epoch in range(num_epochs):
    optimizer.zero_grad(set_to_none=True)   # guarantee a clean slate each epoch
    for step, batch in enumerate(dataloader):
        micro_in_window = (step % ACCUM_STEPS) + 1
        is_last_overall = (step + 1) == len(dataloader)
        is_boundary = ((step + 1) % ACCUM_STEPS == 0) or is_last_overall

        # divide by the ACTUAL window size so the last partial window is weighted correctly
        window_size = ACCUM_STEPS if not is_last_overall else (
            ACCUM_STEPS if (step + 1) % ACCUM_STEPS == 0 else (step % ACCUM_STEPS) + 1
        )
        loss = loss_fn(model(batch.x), batch.y) / window_size
        loss.backward()

        if is_boundary:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
```

This is exactly the kind of fiddly boundary logic that `accelerate`'s `accumulate` context manager handles for you (it detects the end of the dataloader and steps on the final partial window), which is a strong argument for using it. But if you hand-roll, the rule is: **zero at epoch start, and either flush the partial window with the correct denominator or deliberately drop it — just never let it bleed.** Drop-it is fine for big datasets; flush-it matters for small ones where a dropped window is a meaningful fraction of the data.

## 13. Bisecting an accumulation mismatch in practice

Let us put it together as a debugging narrative, the way you would actually work a real failing run, using the make-it-fail-small and read-the-instruments tools that are the spine of this series. The taxonomy post, [a taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs), gives the master decision tree; here is the accumulation-specific branch of it.

You have a run that does not match your baseline. Baseline = single GPU, batch 32, no accumulation, final loss 0.71. New run = single GPU, micro-batch 4, `accumulation_steps=8` (same effective 32), and it spikes / lands at 0.85 / whatever. The bisection:

**Step 1 — run the equivalence test (section 5).** This is the highest-leverage single action. Take one big batch of 32, run the `assert_accumulation_equals_big_batch` test under SGD with BN/dropout off. If it *fails* with a large param diff, your gradient math is wrong — almost certainly the missing loss division — and you are done; fix the divide and re-test. If it *passes*, the gradient path is correct and the bug is elsewhere. This single test splits "is it the gradient math or not" in one shot, which is the most important fork.

**Step 2 — if the test passed, suspect the schedule.** Plot the actual learning rate over optimizer steps for both runs. If the accumulated run's warmup ends `accumulation_steps`× too early, or its decay reaches zero too soon, your scheduler is counting micro-steps. Fix per section 6 and confirm the LR curves overlay.

**Step 3 — if the LR curves match too, suspect batch-coupled layers.** Does the model have BatchNorm? If yes, the micro-batch of 4 is starving BN of statistics (section 8) — switch to GroupNorm/LayerNorm, increase the micro-batch, or accept that this is not a bug in your loop but a fundamental limitation of BN under small batches. Check eval loss specifically, since BN's damage hides in `train()` mode.

**Step 4 — if it is multi-GPU and slow but correct,** it is the `no_sync` performance issue (section 11), not a correctness bug. Profile the step, see the all-reduce time, add `no_sync`.

![A decision tree that starts from an update mismatch and branches on whether parameters differ at step one into loss-normalization, schedule-counting, and BatchNorm suspects](/imgs/blogs/gradient-accumulation-and-effective-batch-bugs-7.png)

The instruments to read while bisecting: the **effective learning rate** (compute and log it — if it is `accumulation_steps`× your set value, you are summing), the **gradient norm right before `optimizer.step()`** (compare it to the baseline big-batch run's grad norm at the same point — they should be within noise if averaging is correct; if it is `accumulation_steps`× larger, you are summing), the **number of optimizer steps per epoch** (should be `len(dataloader) / accumulation_steps`, not `len(dataloader)`), and the **scheduler's reported step** (should advance once per window). Each of these is a one-line log that pins one suspect. The grad-norm comparison is especially clean: it directly measures whether you averaged or summed, before any optimizer state confounds it.

#### Worked example: the grad-norm tell

Concretely, instrument both runs to print `total_norm` from `clip_grad_norm_` (it returns the pre-clip norm) right before `optimizer.step()`. The baseline big-batch run reports a grad norm of about 1.8 in the early steps. The accumulated run, if correct, reports about 1.8 too — same gradient, same norm. If the accumulated run reports about 14.4, that is $8 \times 1.8$, and you have found the summing bug without even running the full equivalence test: the grad norm *is* `accumulation_steps`× too big, exactly as section 3 predicts, because you accumulated the sum instead of the average. This is the fastest possible confirmation — one number, read once, that distinguishes summing from averaging. After the fix (dividing the loss by 8), the accumulated grad norm drops to ~1.8 and tracks the baseline step for step.

## 14. Before and after: the evidence

Here is the concrete before→after for the running example — a 7B finetune, micro-batch 4, `accumulation_steps=8`, target effective batch 32, LR `2e-5`, AdamW, cosine schedule with 3% warmup over 500 optimizer steps, bf16. The buggy version summed gradients (no loss division) and counted the schedule in micro-steps.

| Instrument | Buggy run | After fix | What changed |
| --- | --- | --- | --- |
| Loss division | none (summed) | `loss / 8` before backward | gradient now averaged, not summed |
| Grad norm before step | ~15 (≈8×) | ~1.9 | matches the big-batch baseline |
| Effective LR | ~`1.6e-4` | `2e-5` | back to the rate you set |
| Schedule step unit | micro-steps | optimizer steps | warmup over 500 optim steps, not 62 |
| Loss curve | 2.4 → spike 7.9 → NaN | 2.4 → smooth → 0.71 | converges, matches baseline |
| Final eval loss | NaN / 1.9 | 0.71 | matches the big-batch run |
| Optimizer steps/epoch | (curve unstable) | `len(loader)/8` | correct step accounting |

![A before-and-after figure contrasting a buggy run that spikes from 2.1 to 8.5 with warmup ending 4x too early against a fixed run that descends smoothly from 2.3 to 0.7 with warmup over 500 optimizer steps](/imgs/blogs/gradient-accumulation-and-effective-batch-bugs-8.png)

How you would *confirm* this honestly, not just assert it: (1) run the section-5 equivalence test in CI — it goes from a large failing param diff to a `<1e-5` pass once you add the loss division; (2) overlay the accumulated run's loss curve on a true big-batch run (or the largest batch your GPU fits without accumulation) — they should be visually indistinguishable within run-to-run seed noise once the loop is correct; (3) log the effective LR and the pre-step grad norm and verify both match the baseline. The equivalence test is the proof; the overlaid curves and matching instruments are the corroboration. If the test passes and the curves overlay, accumulation *is* your bigger batch — for everything except BatchNorm, which you have already routed to GroupNorm or a larger micro-batch.

## 15. Case studies and real signatures

A few well-known patterns and real fixes, so you recognize them in the wild. Where I give a number I will say if it is approximate.

**The Hugging Face token-averaging fix (2024).** As noted in section 10, `transformers`/`accelerate` shipped a real loss-normalization bug for token-level losses under gradient accumulation: with variable-length sequences, dividing the loss by a fixed `gradient_accumulation_steps` weights each micro-batch equally regardless of how many real (non-padding) tokens it contains, so micro-batches with more tokens were under-weighted relative to a true big batch. The fix normalizes by the **total token count across the accumulation window** instead of by the step count, and the maintainers documented measurable loss-curve differences on language-model finetunes before and after. The general lesson is exact: "divide by N" is only correct when each micro-batch contributes equally; for per-token losses with ragged lengths the correct denominator is the token count, which is *not* proportional to the micro-batch count. If you finetune LLMs with accumulation on an older stack, this is a real, shipped bug to check for.

**The "8× GPUs, same wall-clock" all-reduce waste.** A common and unglamorous one: a team scales an accumulation loop to multi-GPU DDP, sees almost no speedup, and blames the model or the interconnect. The cause is the missing `no_sync` (section 11) — they are all-reducing on every micro-step. Profiling shows the all-reduce dominating the step; adding `no_sync` cuts communication by the accumulation factor and recovers most of the scaling. This is consistent with the worked example in section 11: a 2–3× throughput swing from one context manager, depending on the compute-to-communication ratio.

**BatchNorm CNNs that "won't benefit from accumulation."** Practitioners training classic BN-heavy detection or classification CNNs on memory-constrained GPUs often report that accumulation does not give them the accuracy of a real big batch — the gradient is right, but the model is a point or two worse. This is not a loop bug; it is the section-8 BatchNorm limitation, and it is *expected*. The standard fixes in the detection literature are exactly the ones in section 8: GroupNorm (used in several detection frameworks precisely for small-batch robustness), SyncBN across GPUs when the batch is sharded, or keeping the per-forward-pass batch large enough. The signature — train fine, eval worse, gradient math provably correct — is the giveaway that it is BN, not the loop.

**The schedule that decayed to zero LR halfway.** A finetune that "stops improving" at the midpoint of the run, with a loss that flatlines rather than spikes, is frequently a schedule counted in micro-steps (section 6): the cosine decay, configured for the micro-step count, reaches zero LR at the halfway point in optimizer steps, and the second half of training happens at LR ≈ 0, learning nothing. The tell is the LR curve hitting zero early; the fix is to count optimizer steps. People misdiagnose this as "the model converged" when it actually stopped learning because the learning rate vanished.

**The "I doubled accumulation and my model got worse" surprise.** A subtler one worth its own line, because it is *not* a bug — it is the linear-scaling rule biting. A practitioner who has a working config at micro-batch 4 with `accumulation_steps=8` (effective 32) hits a smaller GPU, bumps `accumulation_steps` to 16 to keep memory the same, and finds the model degrades. The loop is correct; the *effective batch doubled* from 32 to 64, and a larger batch wants a different (usually higher, up to a point) learning rate per the linear-scaling rule from Goyal et al.'s large-minibatch work, plus possibly more warmup. The fix is not in the accumulation loop at all — it is to treat the effective batch as the thing your LR is tuned for, and re-tune (or scale) the LR when you change `accumulation_steps`. The discriminator from a real bug: the equivalence test still passes and the gradient is correct; the issue is that you changed a hyperparameter (effective batch) without adjusting its partner (LR). This is exactly why I insisted on divide-the-loss over divide-the-LR — it keeps the LR meaning "the rate for the effective batch," so when you change accumulation, you change the effective batch knowingly, not the LR silently.

**The token-normalization mismatch, quantified.** To make the section-10 Hugging Face token-averaging issue concrete, take an accumulation window of 4 micro-batches whose non-padding token counts are 800, 1200, 400, and 1600 — a 4× spread, which is entirely normal when sequences vary in length and you pad to the longest in each micro-batch. A correct big-batch loss is the total cross-entropy summed over all 4,000 tokens, divided by 4,000. The fixed-denominator accumulation loss instead averages the four *per-micro-batch* mean losses with equal 1/4 weights, which up-weights the 400-token micro-batch (its tokens get 1/4 of the loss weight for only 10% of the tokens) and down-weights the 1,600-token one. The gradient you get is a different convex combination of per-token gradients than the big batch's, so it is genuinely a different (slightly mis-averaged) update. The magnitude of the error grows with the token-count spread across the window; it is negligible when sequences are uniform-length and meaningful when they are ragged. This is why the correct denominator for a per-token loss is the *token count over the window*, not the *micro-batch count* — and why fixed-length packing (which makes every micro-batch the same token count) sidesteps the problem entirely.

## 16. When this is (and isn't) your accumulation bug

A decisive section, because misattributing a symptom to accumulation wastes as much time as missing it.

**It probably IS accumulation if:** you recently introduced or changed `gradient_accumulation_steps`; the symptom is a loss spike or NaN that vanishes when you divide the LR by exactly the accumulation factor (that is the summing bug, full stop); your accumulated run does not match a no-accumulation baseline at the same effective batch; the grad-norm before stepping is `accumulation_steps`× your baseline's; or your warmup/decay ends `accumulation_steps`× too early. The equivalence test (section 5) is the arbiter — run it and stop guessing.

**It probably is NOT accumulation if:** the equivalence test *passes* and the LR curve overlays the baseline — then the gradient math and schedule are correct, and the bug is elsewhere (data, model code, numerics, eval). A loss spike that persists at *every* accumulation setting, including `accumulation_steps=1` (i.e., no accumulation at all), is not an accumulation bug — it is a plain too-high-LR or numerics problem; go to [the learning rate is almost always the problem](/blog/machine-learning/debugging-training/the-learning-rate-is-almost-always-the-problem) or the NaN-hunting post. A train-fine-eval-bad gap with *no* BatchNorm in the model is not the section-8 issue — look at overfitting, eval-set leakage, or a train/eval-mode bug. And a smooth-then-NaN curve with a *correct* equivalence test is numerics (fp16 underflow, an `exp` overflow), not accumulation — accumulation bugs show up as the *wrong magnitude* of a correct gradient, not as a fresh source of NaNs.

The clean discriminator: **accumulation bugs change the *scale* or *timing* of an otherwise-correct update; they do not invent new failure modes.** If you are seeing a brand-new kind of failure that has no analog in a normal training step, accumulation is probably not the cause. If you are seeing a normal failure (too-high LR, a schedule that ran too fast, a noisy BN) whose magnitude happens to be a factor of `accumulation_steps`, it very likely is. That factor-of-N fingerprint, and the equivalence test that confirms it, are what let you localize this class of bug in minutes. For the full symptom→suspect→test→fix tree across all bug classes, the [training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) capstone assembles the complete decision graph this post's branch plugs into.

## 17. Initialization, normalization, and the small-batch connection

One last connection, because it ties accumulation back to the broader normalization story. The reason BatchNorm breaks under accumulation is the *same* reason BatchNorm is fragile at small batch sizes in general: it depends on batch statistics, and small batches give noisy statistics. If you have read [initialization and normalization bugs](/blog/machine-learning/debugging-training/initialization-and-normalization-bugs), you know the running-stats trap — that BN's eval-mode behavior depends on EMA statistics accumulated during training, and that those statistics can be wrong even when training-mode loss looks great. Accumulation interacts with that trap directly: under accumulation, the EMA is updated from *micro-batch* statistics, so the running mean and variance converge to estimates appropriate for the micro size, not the effective size. When you then evaluate (eval mode uses running stats), the normalization is calibrated for batches of 4, and your effective-batch-32 model evaluates as if it were a batch-4 model — which is exactly the worse-eval signature.

This is the unifying view: **accumulation is a tool for getting a big *gradient* on a small-*memory* budget, and it succeeds completely for anything that is a per-example sum (the gradient itself) and fails for anything that is a batch-coupled statistic (BatchNorm's mean/variance).** Choose architectures and normalization that are per-example (LayerNorm, GroupNorm) and accumulation is a free, exact bigger batch. Stick with BatchNorm and accumulation is bigger-batch for the gradient only, with a per-example-noisy normalization that no amount of careful averaging can fix — because the normalization happens inside the forward pass, before any gradient exists, and the forward pass only ever sees the micro-batch. That single sentence is the whole post compressed: averaging fixes the gradient; nothing fixes the per-forward-pass statistic.

## Key takeaways

- **Accumulation must AVERAGE, not sum.** The gradient of the mean-loss over $N \cdot B$ examples is the *average* of the $N$ micro-batch mean-gradients. Since PyTorch's `backward()` adds gradients into `.grad`, you must divide the loss by `accumulation_steps` before `backward()`, or your effective learning rate is `accumulation_steps`× too big — a loss spike then NaN.
- **Divide the loss, never the LR.** Dividing the loss is correct under every optimizer; dividing the LR is only equivalent under SGD and is *wrong* under Adam/AdamW (whose step is scale-invariant in the gradient and whose weight-decay balance shifts), producing a subtly mistuned run that is harder to catch than a clean spike.
- **The equivalence test is the proof.** Compare the parameter update from one big batch against $N$ accumulated micro-batches on the same data, under SGD with BN/dropout off; they must match to `~1e-5` in fp32. A large diff is the summing bug. Put it in CI.
- **Schedules count optimizer steps, not micro-steps.** Warmup, total-steps, decay length, logging, and checkpointing must be in optimizer-step units. Counting micro-steps makes everything run `accumulation_steps`× too fast — a warmup that ends early and a decay that hits zero LR halfway through the run.
- **BatchNorm breaks the equivalence and cannot be fixed by averaging.** BN computes statistics across the batch dimension of each forward pass, which only sees the micro-batch. Use GroupNorm/LayerNorm (per-example, so accumulation is exact), SyncBN across GPUs (not across accumulation steps), or a large-enough micro-batch.
- **Dropout's mismatch is benign.** Independent masks per micro-step are a valid, slightly different sample from the same distribution — not a bug; it only means the equivalence test must run with dropout disabled or RNG controlled.
- **Use `no_sync()` under DDP.** Wrap non-final micro-steps in `model.no_sync()` to all-reduce once per optimizer step instead of `accumulation_steps` times — identical math, up to a `accumulation_steps`× cut in communication. The "8× GPUs, barely faster" mystery is usually a missing `no_sync`.
- **Step and zero only on the window boundary.** Stepping every micro-step disables accumulation; never zeroing makes gradients diverge across windows; zeroing every micro-step disables accumulation the other way. Zero at epoch start and either flush or deliberately drop the last partial window — never let it bleed.
- **Read the grad norm before stepping.** It is the fastest tell: a correct accumulated run's grad norm matches the big-batch baseline; an `accumulation_steps`× larger norm is the summing bug, confirmed in one logged number.

## Further reading

- PyTorch documentation, *Gradient Accumulation* and `torch.nn.parallel.DistributedDataParallel.no_sync` — the official mechanics of additive gradients and deferred all-reduce.
- Hugging Face Accelerate documentation, *Performing gradient accumulation* — the `Accelerator.accumulate` context manager and its handling of loss scaling, `no_sync`, and the optimizer-step boundary.
- Hugging Face Transformers, the 2024 gradient-accumulation loss-normalization fix for token-level losses — the token-count vs step-count denominator issue for variable-length sequences.
- Goyal et al., *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour* (2017) — the linear-scaling rule and warmup for large effective batches, the canonical reference for LR-vs-batch-size and why the schedule matters.
- Ioffe & Szegedy, *Batch Normalization* (2015), and Wu & He, *Group Normalization* (2018) — why BN depends on batch statistics and why GroupNorm is robust to small (and accumulated) batches.
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (AdamW, 2019) — why AdamW's decay is independent of the gradient scale, which is why divide-the-loss and divide-the-LR are not equivalent under it.
- Within this series: [a taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) for the master symptom→suspect→test→fix tree, [the training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) for the full capstone decision graph, [the learning rate is almost always the problem](/blog/machine-learning/debugging-training/the-learning-rate-is-almost-always-the-problem) since accumulation bugs masquerade as LR bugs, and [initialization and normalization bugs](/blog/machine-learning/debugging-training/initialization-and-normalization-bugs) for the BatchNorm running-stats trap that accumulation aggravates.

---
title: "The Training Debugging Playbook: A Field Manual from Symptom to Fix"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "The one page to keep open while debugging: the mindset, the six places a bug hides, the five-test preflight, the master symptom-to-fix table, the bisection procedure, and how to build a training system that surfaces its own bugs in minutes."
tags:
  [
    "debugging",
    "model-training",
    "finetuning",
    "deep-learning",
    "pytorch",
    "llm",
    "mixed-precision",
    "data-leakage",
    "distributed-training",
    "evaluation",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 60
image: "/imgs/blogs/the-training-debugging-playbook-1.png"
---

Here is the run that taught me to write this page. A 7B model, finetuned overnight on a curated instruction set. The loss curve was beautiful — a clean exponential decay from 2.1 down to 0.34, no spikes, no plateau. Validation perplexity dropped. Every dashboard was green. I shipped it. In production it behaved exactly like the base model: none of the new instructions stuck, the formatting we trained on never appeared, the refusals we wanted were gone. Roughly \$2,100 of GPU time produced a no-op. The cause took four minutes to find once I stopped trusting the loss curve and ran one test: the LoRA adapter was never in the optimizer's parameter group, so `print_trainable_parameters()` read `trainable params: 0`. The loss went down because the *base model* was already excellent at next-token prediction on English, and the "improvement" was the optimizer doing nothing to noise. Every instrument said healthy. The model learned nothing.

That experience is the whole reason this series exists, and this post is its hub. Training failures are *silent*. Unlike a web server that throws a stack trace, a broken training run produces plausible-looking numbers — a falling loss, a rising accuracy, a clean checkpoint — and the distance between "plausible" and "correct" is exactly where weeks of GPU time and engineer-time disappear. The single most expensive belief in machine learning is "the loss is going down, so it's working." A falling loss is not proof of correctness. It is proof that *some* scalar is being minimized, which is a much weaker statement than you want it to be.

This is the field manual. It is deliberately the longest, most cross-referenced post in the series, because it is the one I want open in a tab when a run is on fire. It ties the entire series into a single procedure you can run from the top: the mindset (bisect before you touch code; the scientific method applied to training), the six places a bug can hide and the cheapest test for each, the five-test preflight you run before any big run, the master symptom-to-suspect-to-test-to-fix table, the bisection procedure drawn as a decision tree, the finetuning-specific twists, how to build a system that surfaces its own bugs, and the anti-patterns that cost the most days. Figure 1 is the spine of everything below: a symptom at the top, six suspects in the middle, and the single cheapest test that confirms or clears each one.

![A decision tree that routes a training symptom through a fast-versus-stateful split down to six suspect places, each annotated with its cheapest confirming test such as overfit one batch or single-GPU repro](/imgs/blogs/the-training-debugging-playbook-1.png)

By the end you will be able to take any stalled, diverging, NaN-ing, or secretly-overfit run — in vision, LLM, speech, or tabular — and localize the bug to one of six places in minutes, name the test that confirms it, and know the direction of the fix. If you read one post in this series, read [the taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) for the conceptual frame and read this one for the procedure. Everything else is a branch of these two drawn in full.

## 1. The mindset: training failures are silent, so bisect before you touch code

Start with the property that makes ML debugging hard and reframe it as the property that makes it tractable. In ordinary software, a bug usually announces itself: the program crashes, the assertion fires, the test goes red. The feedback loop is tight and the failure is loud. In training, the feedback loop is long (a run is hours), the failure is quiet (numbers stay plausible), and the ground truth is hidden (you do not know what the model *should* have learned, only what you hoped). The result is that the default debugging move from software engineering — "stare at the code until you spot the bug" — fails catastrophically here, because the code often looks correct and the bug is in the *interaction* between code, data, numerics, and hardware.

The mental shift that changes everything is to stop debugging by inspection and start debugging by **bisection**. You do not guess which line is wrong. You run a small number of cheap tests that each cut the space of possible causes roughly in half, and you let the tests tell you where to look. This is not a metaphor borrowed loosely from `git bisect` — it is the same information-theoretic argument. If a bug could be in any of $N$ equally-likely places and each test partitions those places into two halves, then in the worst case you localize the bug in $\lceil \log_2 N \rceil$ tests. With six suspect places, that is at most three tests; in practice, because the symptom itself already biases the prior, it is usually one or two. The whole craft is choosing tests that *split the space evenly* and *cost almost nothing*.

It is worth making the information argument precise, because it explains *why* one test order is better than another. A test that splits the suspect set 3-versus-3 carries one full bit of information — it halves your uncertainty no matter the outcome. A test that splits it 5-versus-1 carries far less than a bit on average: if it usually comes back "the bug is in the big bucket of 5," you have barely narrowed anything. This is why overfit-one-batch leads the procedure: it splits the six places into roughly {data, evaluation, full-set schedule} versus {model code, loss, basic optimization}, a near-even 3-versus-3 cut, so whichever way it comes back you have killed half the suspects. A test that *can only confirm and never deny* — "let me lower the LR and see if it helps" — carries almost no information when it fails, because a no-change result is consistent with every other place still being the culprit. Prefer tests whose *both* outcomes are informative.

The contrast with ordinary software debugging is instructive and is the reason transferred habits fail here. In software you can often set a breakpoint and step until you see the wrong value, because the program is deterministic, fast, and observable at every line. A training run is none of those by default: it is stochastic (different every run unless you force a seed), slow (the "step" you want to inspect is hours away), and opaque (the quantity that is wrong — "the model didn't learn the concept" — is not a variable you can print). So the breakpoint-and-step instinct has to be replaced by the bisect-and-measure instinct. You do not step *through* the run; you run *small, cheap, discriminating experiments around* the run and triangulate. Everything in this playbook — the preflight, the table, the decision tree — is machinery for making those experiments cheap and their outcomes discriminating.

The second half of the mindset is the **scientific method**, compressed into a loop you run dozens of times a day:

1. **Observe the symptom precisely.** Not "it's not working" but "validation accuracy is stuck at 11.2% on a 9-class problem, which is exactly chance, while training loss falls smoothly from 2.20 to 1.95 and then flattens." Precision in the symptom is half the diagnosis. "Stuck at chance" points somewhere very different from "diverged after a spike."
2. **Form one hypothesis.** A single, falsifiable claim: "the labels are being masked, so the model is computing loss on nothing trainable." Not five hypotheses — one. If you have five, rank them and test the cheapest-to-confirm first.
3. **Run the cheapest test that would confirm or kill that hypothesis.** Decode one batch and print the label tensor. If it is all `-100`, you are done in thirty seconds. Notice the asymmetry: a *cheap* test that *strongly discriminates* is worth ten expensive tests that weakly hint.
4. **Fix one thing.** Exactly one. Change the masking. Nothing else.
5. **Re-measure.** Did the symptom move in the predicted direction and amount? If yes, you understand the bug. If no, your hypothesis was wrong — and now you have *new evidence*, not a wasted run.

The discipline that makes this loop work is **changing one thing at a time** and **measuring before and after**. The most common way engineers turn a one-hour bug into a three-day bug is by changing five things at once (LR, batch size, the model, the data, and the precision) and then being unable to attribute the result to any of them. If the run gets better, which change helped? If it gets worse, which change hurt? You have learned nothing and burned a run. The whole rest of this manual is in service of running this loop fast: the preflight makes it cheap, the table tells you which test discriminates, and the bisection tree tells you which fork to take first.

One more principle deserves its own sentence because it is violated constantly: **a falling loss is not proof of correctness.** The loss is one scalar summarizing a huge computation, and an enormous number of broken configurations still produce a smoothly falling loss — a model that overfits a leak, a model whose adapter does nothing while the frozen base predicts well, a model trained on the prompt instead of the answer, a model with a metric bug that reports the wrong thing. Treat the loss as a *necessary-but-not-sufficient* signal: if it is *not* falling you certainly have a bug; if it *is* falling you might still have a bug, and you need an independent check (overfit-one-batch, a decoded sample, an honest held-out metric) to believe in the run.

## 2. The six places a bug hides

The spine of this entire series is a claim that, once internalized, is freeing: **a training or finetuning bug hides in exactly one of six places — data, optimization, model code, numerics, systems, or evaluation — and you can bisect to the right one before touching code.** You do not have infinite suspects. You have six. Figure 2 stacks them, because they really do form a stack: data flows into model code, which is optimized, which runs on numerics, which runs on systems, and the whole thing is judged by evaluation. A wrong lower layer silently corrupts every layer above it while the loss keeps looking plausible.

![A vertical stack of the six layers a training bug can hide in, from data at the top through model code, optimization, numerics, and systems down to evaluation, each labeled with its characteristic failures](/imgs/blogs/the-training-debugging-playbook-2.png)

Here is each place, its characteristic *tell*, and the single cheapest test that confirms or clears it. This is the recap of the whole series; each one links to the post that draws it in full.

**Data.** The number-one source of bugs, and the one engineers blame last. The tells: a metric that is too good to be true (a leak), a model stuck at chance (the labels are wrong or masked), a sawtooth loss (the dataloader is repeating or mis-shuffling), or a great offline number that collapses in production (train-serve skew). The cheapest test is *look at your data*: print one batch, decode the inputs and the labels back to human-readable form, and check the per-class and per-feature distributions. Most data bugs are visible in the first decoded batch. See [data leakage, the silent killer](/blog/machine-learning/debugging-training/data-leakage-the-silent-killer) for the most expensive member of this family, and [the loss-masking bug](/blog/machine-learning/debugging-training/the-loss-masking-bug) for the one that quietly trains on nothing.

**Optimization.** The learning rate, the schedule, the optimizer state, gradient clipping. The tells: a loss that crawls (LR too low), a loss that spikes then diverges (LR too high), or a loss that plateaus far above where it should. The cheapest test is the *LR-range test* plus *per-layer grad norms*. As [the learning rate is almost always the problem](/blog/machine-learning/debugging-training/the-learning-rate-is-almost-always-the-problem) argues, an astonishing fraction of "the model won't train" reports are a learning rate off by 10x–100x, and in finetuning the LR is usually 10x–100x *too high* relative to the tiny step the pretrained weights can tolerate.

**Model code.** Shapes, masks, frozen parameters, broken gradient flow, the submodule that silently never updates. The tells: the model cannot even overfit one batch; a submodule's gradient is `None`; a mask leaks the future; an in-place op detaches the graph. The cheapest test is *overfit one batch* — if a correctly-wired model cannot drive loss to near-zero on two batches it can memorize, the bug is in the model or the optimization, not the data. See [your model isn't learning what you think](/blog/machine-learning/debugging-training/your-model-isnt-learning-what-you-think) and [attention and masking bugs](/blog/machine-learning/debugging-training/attention-and-masking-bugs). The most insidious member of this family is *silent broadcasting*: a shape mismatch that does not crash because PyTorch's broadcasting rules quietly expand a `[B, 1]` tensor against a `[B, T]` one and average over the wrong axis. The loss still computes, still falls, and is still wrong — the only way to catch it is to assert shapes explicitly or read the forward pass shape-by-shape. A bug that *crashes* is a gift; a bug that *broadcasts* is a thief.

**Numerics.** Floating-point underflow and overflow, `log(0)`, division by zero, `sqrt` of a negative, the fp16 gradient that rounds to zero. The tells: a smoothly-falling loss that suddenly goes to `NaN` or `Inf` at step N; gradients that are exactly zero in fp16; activations that blow up. The cheapest test is `torch.autograd.set_detect_anomaly(True)`, which points you at the exact op that first produced a non-finite value. See [hunting NaNs and Infs](/blog/machine-learning/debugging-training/hunting-nans-and-infs) and [loss spikes and divergence](/blog/machine-learning/debugging-training/loss-spikes-and-divergence).

**Systems.** Distributed training, gradient synchronization, gradient accumulation, out-of-memory, the idle GPU. The tells: the run is correct on one GPU and wrong on eight; "8x GPUs, same wall-clock"; an OOM that grows each step (a leak); a 31%-utilized GPU. The cheapest test is *reproduce on a single GPU*: if the bug vanishes on one device, it is a systems bug, full stop. See [debugging DDP and multi-GPU](/blog/machine-learning/debugging-training/debugging-ddp-and-multi-gpu) and [out-of-memory debugging](/blog/machine-learning/debugging-training/out-of-memory-debugging).

**Evaluation.** The metric itself is wrong, the eval set is leaked or skewed, the offline number does not predict the online number. The tells: a number that looks great offline and bad online; a 99% accuracy that is an artifact of class imbalance; an eval that disagrees with eyeballing samples. The cheapest test is *re-derive the metric by hand on ten examples* and *replay a real serving input through the eval path*. A metric bug can make a broken model look fixed and a fixed model look broken, so this place is where you check last but trust least. The classic traps are micro-versus-macro averaging (micro-F1 on an imbalanced set is dominated by the majority class and hides catastrophic minority-class failure), a threshold left at 0.5 on a calibrated-for-something-else model, and an eval set that quietly overlaps the training set. The deepest version is the *offline-online gap*: your held-out metric is excellent but production is poor, because the eval distribution does not match what users actually send — which is really a data/train-serve-skew story dressed up as an eval number. The capstone of this track is shipping alongside this post as your-metric-is-lying.

The reason this six-way split is worth memorizing is that it converts an unbounded search ("what's wrong with my run?") into a bounded one ("which of these six, and what's the test?"). The next sections turn that bounded search into a fixed procedure.

There is a deeper symmetry worth naming. Each of the six places fails by *corrupting a different part of the same pipeline*, and because the loss is computed at the very end of that pipeline, a corruption anywhere upstream produces a loss that still looks plausible. Data corruption feeds the model wrong inputs but the model still happily minimizes its loss on them. A model-code bug (a frozen submodule) leaves the rest of the network learning normally, so the loss still falls — just less than it should. A numerics bug stays invisible until a value crosses a representable boundary. A systems bug (wrong gradient averaging across ranks) scales the effective learning rate, which looks like a *slightly* different but still-falling curve. An evaluation bug never touches training at all — the run is fine and the *report* is wrong. This is the unifying reason the loss curve cannot be your only instrument: it is the single number furthest downstream of all six failure points, so it is the *last* place a bug shows up clearly and the *first* place it shows up misleadingly.

### The two master tools: make-it-fail-small and read-the-instruments

Underneath every test in this playbook are exactly two techniques, and recognizing them as the same two over and over is what lets you invent the right test for a bug you have never seen before.

The first is **make-it-fail-small**: shrink the run along whatever axis isolates the suspect, then see if the bug survives. Overfit *one batch* shrinks the data axis (does the model learn at all on a trivial dataset?). Run on *one GPU* shrinks the systems axis (does the bug need parallelism?). Run *one step* with `detect_anomaly` shrinks the time axis (which op first goes non-finite?). Train *one feature* shrinks the model axis (is a single feature enough to leak the target?). A *tiny model on a tiny config* shrinks everything at once (does the full pipeline even run end-to-end?). The art is choosing the axis whose shrinking *cleanly separates* the suspect you are testing from the others — exactly the even-split criterion from §1.

The second is **read-the-instruments**: instead of guessing what the run is doing, measure it. The gradient norm tells you whether learning is happening and at what scale. The parameter and update norms tell you whether steps are large enough to matter (a healthy update-to-parameter ratio sits around $10^{-3}$; far smaller and the model is barely moving, far larger and it is being torn apart). The activation histogram tells you whether a layer is dead (all zeros, post-ReLU) or saturated (piled at the tails of a sigmoid). The LR-at-step tells you the schedule is doing what you think. Throughput and GPU utilization tell you whether the hardware is the bottleneck. Each instrument *rules out* a class of bug by reading a number you would otherwise have guessed. The preflight in the next section is simply these two tools applied in cost order.

## 3. The first five tests: the preflight before any big run

Before you launch a run that will cost real money and real hours, you run a preflight. These five tests, in this order, cost a few minutes total and rule out the overwhelming majority of bugs *before* you have burned a single GPU-hour on a doomed run. I run them on every new model, every new dataset, and every config change that could plausibly break the pipeline. Figure 3 lays them out in cost order, because the order matters: each test is chosen to halve the suspect space before the next one runs.

![A left-to-right timeline of the five preflight tests in cost order, from full determinism through overfit one batch, print and decode a batch, log per-layer grad norms, and single-GPU reproduction, each labeled with what it splits](/imgs/blogs/the-training-debugging-playbook-3.png)

### Test 1 — set full determinism

You cannot debug what you cannot reproduce. If two runs of the same config produce different curves, you can never tell whether a change you made helped, hurt, or did nothing — the noise swamps the signal. So the very first thing is to make the run bit-reproducible (accepting the modest throughput cost; you turn it off once the bug is found). This rules in nothing about the bug directly, but it makes every subsequent test *trustworthy*. See [reproducibility and determinism in training](/blog/machine-learning/debugging-training/reproducibility-and-determinism-in-training) for the full treatment of why this is harder than `torch.manual_seed`.

```python
import os, random
import numpy as np
import torch

def set_full_determinism(seed: int = 0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    # cuBLAS reproducibility for matmuls used by many ops on GPU.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Force deterministic kernels; this will RAISE on any nondeterministic op,
    # which is exactly what you want during debugging.
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False     # benchmark picks fast, nondeterministic kernels
    torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    # Each DataLoader worker reseeds NumPy/random from torch's per-worker seed.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Pass to the DataLoader so augmentation/shuffle order is reproducible:
g = torch.Generator(); g.manual_seed(0)
# DataLoader(..., worker_init_fn=seed_worker, generator=g)
```

What it rules in/out: nothing about the bug yet, but if `use_deterministic_algorithms(True)` *raises*, you have just learned your run depends on a nondeterministic op — which is itself worth knowing, because that op can also be the source of multi-GPU divergence later.

### Test 2 — overfit one batch

This is the single highest-leverage test in all of training debugging, and the reason [the overfit-a-single-batch test](/blog/machine-learning/debugging-training/the-overfit-a-single-batch-test) has its own post. Take one small batch (or two), turn off all regularization and augmentation, and train *only on that batch* for a few hundred steps. A correctly-wired model with a reasonable LR will drive the loss to near-zero — it has the capacity to memorize a handful of examples, and if it cannot, something is preventing learning.

```python
def overfit_one_batch(model, batch, optimizer, loss_fn, steps=300, device="cuda"):
    model.train()
    batch = {k: v.to(device) for k, v in batch.items()}
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        out = model(**batch)
        loss = out.loss if hasattr(out, "loss") else loss_fn(out, batch["labels"])
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step == steps - 1:
            print(f"step {step:4d}  loss {loss.item():.4f}")
    # Decision rule: a healthy model reaches < 0.05 (or near-zero CE) here.
    return loss.item()
```

What it rules in/out. **Pass** (loss → ~0.01): your model can learn, your gradients flow, your loss is wired correctly. The bug is almost certainly in data, generalization, the schedule on the full set, or evaluation — *not* in the model code or basic optimization. That is half the suspect space gone in three minutes. **Fail** (loss stuck > 1.0): the bug is in the model code, the loss function, or the optimization — a frozen submodule, a detached graph, a wrong reduction, an LR that is far too low, a mask that hides the target. You now know to read the model, not the data.

### Test 3 — print and decode one batch

Look at exactly what the model sees. Not the dataset spec — the tensors at the model's input, decoded back to something human-readable, including the labels. This single habit catches more data bugs than any tool.

```python
batch = next(iter(train_loader))
print("input_ids shape:", batch["input_ids"].shape)
print("first example decoded:")
print(tokenizer.decode(batch["input_ids"][0]))      # is BOS/EOS right? double-BOS?
# Labels: what is actually being trained on?
labels = batch["labels"][0]
print("labels (first 20):", labels[:20].tolist())   # all -100? then you train on nothing
unmasked = labels[labels != -100]
print(f"unmasked label tokens: {unmasked.numel()} / {labels.numel()}")
print("unmasked decoded:", tokenizer.decode(unmasked[unmasked >= 0]))
# For vision: check the actual pixel range and channel order.
# print(images.min().item(), images.max().item(), images.shape)  # [0,1]? [0,255]? NCHW?
```

What it rules in/out: it confirms or clears the entire data layer. If the labels are all `-100`, you have the loss-masking bug and your overfit test would also have failed. If the inputs are normalized to `[0, 255]` when the model expects `[0, 1]`, or the channels are BGR when the model wants RGB, you see it immediately. If the decoded text has a double-BOS, the [tokenization bug](/blog/machine-learning/debugging-training/the-loss-masking-bug) is right there.

### Test 4 — log per-layer gradient norms

Read the instruments. The gradient norm per layer is the single most diagnostic signal in optimization, and a per-layer hook costs nothing. It tells you whether gradients are flowing, exploding, vanishing, or absent. As [instrumenting a training run](/blog/machine-learning/debugging-training/instrumenting-a-training-run-what-to-log) details, this one number rules out a huge swath of optimization and numerics bugs.

```python
def grad_norm_report(model, top_k=8):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is None:
            norms.append((name, None))          # None grad = not in graph / frozen by mistake
        else:
            norms.append((name, p.grad.detach().norm().item()))
    total = sum(n**2 for _, n in norms if n is not None) ** 0.5
    print(f"global grad norm: {total:.3e}")
    nones = [n for n, v in norms if v is None]
    if nones:
        print(f"  WARNING: {len(nones)} params have grad=None, e.g. {nones[:3]}")
    finite = [(n, v) for n, v in norms if v is not None]
    finite.sort(key=lambda x: x[1], reverse=True)
    print("  largest:", [(n, f'{v:.2e}') for n, v in finite[:top_k]])
    return total
# Call after loss.backward() and BEFORE optimizer.step().
```

What it rules in/out: a global grad norm of `1e4` that grows each step is an exploding-gradient / LR-too-high signature ([gradients exploding and vanishing](/blog/machine-learning/debugging-training/loss-spikes-and-divergence)). A norm of `1e-7` that does not move is vanishing gradients or fp16 underflow. A *subset* of parameters with `grad=None` is a frozen-by-mistake or detached-graph bug — the model code is wrong. A healthy run sits in a stable band (often `0.1`–`10`) and is clipped if it occasionally spikes.

### Test 5 — run on one GPU

If you are using multiple GPUs, reproduce the bug on a single one. This test exists purely to *split systems out of the search*. Distributed training adds gradient synchronization, data sharding, per-rank seeding, and BatchNorm-across-ranks — a whole class of bugs that simply cannot occur on one device.

```bash
# Multi-GPU launch (the run that's misbehaving):
torchrun --nproc_per_node=8 train.py --config base.yaml

# The diagnostic: same config, ONE process, ONE GPU.
CUDA_VISIBLE_DEVICES=0 python train.py --config base.yaml --world_size 1
```

What it rules in/out: if the bug **vanishes** on one GPU, it is a systems bug — gradient-sync, sharding, accumulation math, or rank desync ([DDP and multi-GPU](/blog/machine-learning/debugging-training/debugging-ddp-and-multi-gpu)). If it **persists**, systems is innocent; the bug is in the other five places, and you have saved yourself from chasing NCCL ghosts. This is the single test that most often ends a multi-day "it only fails in production-scale training" investigation.

#### Worked example: the preflight catches a no-op finetune in four minutes

A concrete run from the top. I am finetuning a `Qwen2-1.5B` base on a 12k-example instruction set with LoRA. Symptom: after one epoch, the model ignores every instruction in eval.

- **Test 1 (determinism):** set; no error. Good, the run is now repeatable.
- **Test 2 (overfit one batch):** loss starts at 2.31 and after 300 steps is... 2.27. It does **not** drop. A healthy LoRA finetune would crush one batch to near-zero. This *fails*, which indicts model code or optimization — not data.
- **Test 4 (grad norms):** I run the grad-norm report after a backward pass. The global norm is `0.000e+00` and the warning fires: *all* trainable params have `grad=None`. Nothing is in the graph.
- **The localization:** I add `model.print_trainable_parameters()`. It reads `trainable params: 0 || all params: 1,543,714,304 || trainable%: 0.0`. The `target_modules` in my `LoraConfig` was `["query", "value"]` but this architecture names them `["q_proj", "v_proj"]`, so the adapter matched nothing.
- **The fix:** correct `target_modules` to `["q_proj", "v_proj", "k_proj", "o_proj"]`. Re-run test 2: loss 2.31 → 0.04 in 300 steps. Re-run the real epoch: eval instruction-following jumps from 0% to working.

Elapsed: about four minutes for the diagnosis, versus the overnight run plus a morning of confusion it would otherwise have cost. This is the entire value proposition of the preflight: it moves the discovery of a class of catastrophic bugs from "after a full run" to "before any run."

## 4. The master table: symptom → suspect → test → fix

This is the lookup you keep open. Find your symptom in the left column, read across to the likeliest suspect place, the cheapest test that confirms it, and the direction of the fix. It is deliberately compressed; each row links to the post that draws it in full. Figure 4 is the same table as a matrix you can screenshot.

![A matrix mapping eight common training symptoms to their likeliest suspect place, the cheapest confirming test, and the fix direction, color-coded by severity](/imgs/blogs/the-training-debugging-playbook-4.png)

| Symptom | Likeliest suspect | Cheapest confirming test | Fix direction |
| --- | --- | --- | --- |
| **Stuck at chance** (acc ≈ 1/C, loss ≈ ln C) | Model code / optim | overfit one batch; grad-norm report | fix grad flow / mask; raise LR if too low |
| **Loss is NaN/Inf** at step N | Numerics | `detect_anomaly`; bisect by step | bf16; clip; guard `log(0)`, `/0`; loss-scale |
| **Spike then diverge** | Optim (LR too high) or a bad batch | LR-range test; inspect the batch at the spike | lower LR; warmup; clip; skip-batch |
| **Loss crawls / plateaus high** | Optim (LR too low) | LR-range test | raise LR; check schedule/warmup |
| **Sawtooth / periodic loss** | Data (dataloader) | print batches across an epoch | fix shuffle; drop_last; check workers |
| **Great train, bad val** | Model (eval mode) or eval | check `.eval()`; re-derive metric | call `model.eval()`; fix mask/leak |
| **Great offline, bad online** | Eval (train-serve skew) | replay a real serving input | match preprocessing/template/dtype |
| **99% val, too good** | Data leakage | dedup across splits; ablate features | GroupKFold / time split; drop leak |
| **OOM** (constant or growing) | Systems | `max_memory_allocated`; memory snapshot | grad-checkpoint; smaller batch; fix leak |
| **Differs on multi-GPU only** | Systems (DDP) | single-GPU repro | fix grad-sync / accum / SyncBN / seeds |
| **Loss jumps on resume** | Systems / optim state | resume-equivalence test | restore optimizer + scheduler + RNG |
| **Won't stop generating** | Data (EOS masked) | decode label ids near end | unmask EOS; fix chat template |
| **Submodule never updates** | Model code | grad-norm report (grad=None) | `requires_grad`; avoid detach/in-place |
| **Forgets base skills** (finetune) | Optim (LR too high) | drop LR to 1e-5; eval base task | lower LR; fewer epochs; mix in base data |

Two structural facts about this table are worth stating because they speed up real debugging.

First, the **"stuck at chance" row deserves a number you should memorize**. For a balanced $C$-class classification problem, a model that has learned nothing outputs the uniform distribution, and cross-entropy against the true label is $-\log(1/C) = \ln C$. So a 9-class problem stuck at chance sits at loss $\ln 9 \approx 2.20$; a 1000-class ImageNet model at chance sits at $\ln 1000 \approx 6.91$; a binary problem at chance sits at $\ln 2 \approx 0.69$. If your loss is parked at exactly $\ln C$ and not moving, the model is producing uniform outputs — it is not learning, and you should run overfit-one-batch immediately rather than waiting "to see if it picks up." This is a *quantitative* tell, not a vibe: $2.20$ on a 9-class task is not "a bit high," it is "exactly chance, the model is doing nothing."

Second, several symptoms are **near-duplicates that the cheapest test cleanly separates**. "Stuck at chance" and "loss crawls" look similar on a glance at a dashboard, but the first fails overfit-one-batch (model can't learn at all) and the second passes it (model can learn, the full-set LR is just too low). "Great train, bad val" has two very different causes — a forgotten `model.eval()` (BatchNorm/dropout active at inference) versus a real generalization gap from a leak or overfitting — and one cheap check (does calling `.eval()` fix it?) separates them. The table's value is not memorizing fixes; it is knowing which *single test* discriminates between the look-alikes.

Third, **the "cheapest test" column is the real intellectual content of the table** — the suspect and the fix are downstream of running it. Notice how often the cheapest test is one of a handful: overfit-one-batch, decode-a-batch, grad-norm-report, single-GPU-repro, detect_anomaly, replay-a-serving-input, dedup-across-splits. Six or seven tests cover the entire table, because each one cleanly partitions the six places. This is why the *preflight* in §3 and the *table* here are the same knowledge viewed two ways: the preflight is "run these discriminating tests proactively, in cost order"; the table is "given the symptom, here is which discriminating test to reach for first." Master the seven tests and their partitions, and you can regenerate the whole table from memory — and, more importantly, place a *new* symptom you have never seen onto it.

A final note on reading the table under uncertainty: the "likeliest suspect" column is a *prior*, not a verdict. When two places are plausible (the table lists "model or optim," "model or eval mode"), you do not have to pick — you run the test that separates them. "Model or optim" → overfit-one-batch separates them further only if you *also* watch the grad norm: if grad flows but loss is stuck, optimization (LR); if grad is `None` somewhere, model code. The table compresses; the tests disambiguate.

## 5. The bisection procedure as a decision tree

The preflight is what you run proactively; the bisection procedure is what you run when a specific run is already failing. It is the same logic — halve the suspect space with each test — but driven by the symptom. Figure 6 draws the central fork; here is the procedure in words.

![A branching decision graph in which the overfit-one-batch result forks the search into clearing the model and optimization on a pass or indicting the model code on a fail, both converging on a confirming test](/imgs/blogs/the-training-debugging-playbook-6.png)

The key insight is that a few tests are *information-dense* — each one cleanly partitions the six places — and you should always reach for those first. Here is the order, with the partition each test induces:

1. **Make it repeatable (fix the seed).** This is not a partitioning test; it is the precondition for every other test to be trustworthy. Do it first, always.
2. **Overfit one batch.** This is the master fork. It splits the six places into {data, evaluation, full-set optimization/schedule} on a *pass* versus {model code, loss function, basic optimization} on a *fail*. One test, half the space gone. There is no cheaper, more discriminating test in all of training debugging — which is exactly why it leads.
3. **Print and decode one batch.** Given a *pass* on overfit-one-batch (so the model is fine), this splits the remaining space: if the decoded inputs/labels are wrong, it is data; if they are right, the bug is in generalization (a leak you'll find by dedup) or evaluation (a metric you'll find by re-deriving).
4. **`detect_anomaly` / inspect numerics.** If the symptom is a `NaN` or `Inf` (rather than "wrong but finite"), this test localizes the *exact op* that first produced a non-finite value, splitting numerics out of the search in one shot. It is expensive (it slows the backward pass a lot), so you only reach for it when the symptom is non-finite.
5. **Single-GPU reproduction.** If the symptom only appears at scale, this splits systems out: vanishes on one GPU → systems; persists → one of the other five. Reach for it whenever the run touches more than one device.

Notice how the symptom changes which test is cheapest-to-discriminate, which is why the table in §4 pairs each symptom with *its* cheapest test rather than prescribing one universal order. A `NaN` symptom routes you to `detect_anomaly` before overfit-one-batch (a NaN run can't overfit anything). A "differs on multi-GPU" symptom routes you to single-GPU repro first (no point overfitting a batch if the bug is in NCCL). A "stuck at chance" or "won't learn" symptom routes you to overfit-one-batch first (the master fork). Internalize the *partition each test induces* and you will pick the right first test automatically.

The decision tree also tells you when to **stop blaming the wrong place**. If overfit-one-batch *passes*, stop reading the model code. The model can learn; the gradients flow; the loss is wired right. Engineers waste enormous time re-reading model code after overfit-one-batch has already cleared it, because reading code *feels* like progress. It is not — the test already told you the model is innocent. Conversely, if the bug *vanishes* on one GPU, stop reading the model and data; the logic is correct, and the bug is in how that logic is parallelized.

#### Worked example: a full bisection from "stuck at chance" to a fix

The real run, end to end, with numbers. A BERT-base finetune on a 5-class sentiment task. Symptom: after 2 epochs, validation accuracy is 19.8% (chance is 20%), and training loss falls from 1.61 (which is $\ln 5 \approx 1.609$, i.e. *exactly chance*) to 1.59 and then flatlines. Figure 5 captures the before/after of this bisection.

![A two-column before and after panel showing a vague stuck-at-chance finetune on the left and the localized loss-mask bug fixed to real learning on the right](/imgs/blogs/the-training-debugging-playbook-5.png)

- **Symptom precision.** Loss parked at $1.59 \approx \ln 5$. That is the quantitative tell: the model is outputting near-uniform logits. It is not "slow," it is "not learning at all." Prior shifts hard toward model code / optimization.
- **Test 1 (seed).** Set. Repeatable.
- **Test 2 (overfit one batch).** I grab 8 examples and train 300 steps. Loss: 1.61 → 1.60 → 1.60. It *fails*. A correctly-wired BERT-base crushes 8 examples to ~0.01 in well under 100 steps. This indicts model code, loss, or optimization — and clears data and evaluation (they cannot stop a model from memorizing 8 examples). Half the space gone.
- **Test 4 (grad norms).** I run the grad-norm report. Global norm is `0.000e+00`, and the warning fires: *every parameter* has `grad=None`. Nothing is in the graph at all.
- **Localization.** `grad=None` for *everything* means the loss is not connected to the parameters — either the loss is a constant, or the labels are masked out, or `requires_grad` is off everywhere. I print one batch (test 3): the `labels` tensor is `tensor([-100, -100, -100, ...])` for every example. The data collator was applying the *language-modeling* `-100` masking to a *classification* label column, so the classification labels were all turned into `ignore_index`. Cross-entropy on all-`-100` is a constant; its gradient is exactly zero; the model outputs uniform logits forever — loss sits at $\ln 5$.
- **Fix (one thing).** Stop the LM collator from touching the classification label; pass the integer class labels through untouched.
- **Re-measure.** Overfit-one-batch now: 1.61 → 0.008 in 80 steps (pass). Full run: validation accuracy 19.8% → 91.3% in 2 epochs; global grad norm now sits in a healthy `0.4`–`3.0` band. The symptom moved in the predicted direction *and* the predicted magnitude.

Total diagnosis time: about six minutes, dominated by waiting for 300 overfit steps to print. The bug had been "investigated" for half a day before this by reading the model code — which overfit-one-batch had already proven innocent in the first test. That is the cost of reading code instead of bisecting.

#### Worked example: a NaN at step 412, bisected to a guard

A second bisection, this time numerics. A vision model with a custom focal-loss implementation. Symptom: loss falls smoothly from 4.2 to 1.8 over 400 steps, then `NaN` at step 412, and every step after is `NaN`.

- **Symptom precision.** *Smooth then NaN* is the signature of numerics, not data. A data bug produces a *consistent* wrongness from step 1; a numerics bug produces a clean run that suddenly breaks when some value crosses a representable boundary. This routes me to `detect_anomaly`, not overfit-one-batch.
- **Make it repeatable.** Seed set; the NaN reproduces at step 412 every time. (If it did *not* reproduce, that itself would be a clue — a nondeterministic NaN points at a specific bad batch or a race.)
- **`detect_anomaly`.** I wrap the training step:

```python
torch.autograd.set_detect_anomaly(True)
# ... inside the step ...
out = model(images)
loss = focal_loss(out, targets)   # anomaly mode raises at the first NaN-producing op
loss.backward()
```

The traceback points at the `torch.log` call inside my focal loss. The `(1 - p_t)**gamma * log(p_t)` term computes `log(p_t)` where `p_t` is a softmax probability — and at step 412 a confident-correct prediction made `p_t` round to exactly `1.0` in fp16, while a confident-*wrong* prediction made some `p_t` round to `0.0`, and `log(0) = -inf`. The `-inf` times a finite weight is `-inf`; the backward pass propagates `NaN`.
- **The science.** This is the representable-range floor of half precision. fp16's smallest positive normal is $\approx 6.1\times10^{-5}$ and its smallest subnormal is $\approx 6\times10^{-8}$; a softmax probability of a strongly-wrong class can underflow below that and become exactly `0.0`, and `log(0.0)` is `-inf`. The bug was always latent; it only fired when the model got confident enough (around step 412) to push a probability past the floor. See [mixed-precision debugging](/blog/machine-learning/debugging-training/hunting-nans-and-infs) for the full fp16/bf16 range story.
- **Fix (one thing).** Clamp before the log: `log(p_t.clamp_min(1e-7))`, or better, compute the loss from log-probabilities directly with `F.log_softmax` (numerically stable by construction) instead of `log(softmax(x))`.
- **Re-measure.** The run goes from `NaN @ step 412` to a clean 5-epoch run; loss continues from 1.8 down to 0.31; grad norm stays finite throughout. Switching the run to bf16 (range $\approx 3.4\times10^{38}$, like fp32, at the cost of mantissa bits) would *also* have hidden this particular bug by giving `p_t` more headroom — but clamping the log is the correct fix, because it removes the latent `log(0)` regardless of precision.

Two different bisections, two different first tests, both localized in minutes — because the symptom precision (one stuck at $\ln C$, one smooth-then-NaN) chose the right discriminating test up front.

#### Worked example: "8x GPUs, same wall-clock and worse accuracy"

A third bisection, this time systems — the place engineers most often misdiagnose, because the symptom looks like a model or data problem. A team scales a working single-GPU training script to 8 GPUs with `DistributedDataParallel`. Two things go wrong at once: the run is barely faster than one GPU (about 1.3x, not the hoped-for 6–7x), and final validation accuracy is *worse* than the single-GPU baseline (88.1% vs 91.4%).

- **Symptom precision.** Two symptoms, and the discipline is to debug them *separately* rather than assume one cause. Symptom A: poor scaling (a throughput problem). Symptom B: worse accuracy (a correctness problem). Conflating them is the trap.
- **Single-GPU repro (the systems-splitter).** I run the exact config on one GPU. Accuracy is back to 91.4%. So symptom B *is* a systems bug — something about the parallelization corrupts learning. Half the search space (data, model, numerics, eval) is cleared in one run.
- **Localize symptom B.** The two classic DDP correctness bugs are (1) wrong gradient averaging — DDP all-reduces gradients with a *mean* across ranks, so the *effective* batch size is `per_gpu_batch x num_gpus`, which means the effective learning-rate-to-batch ratio changed when I scaled; and (2) BatchNorm computing statistics *per-rank* on a now-smaller per-GPU batch. Here, the per-GPU batch dropped from 256 to 32 when I split 256 across 8 GPUs, so BatchNorm's running statistics became noisy (small-batch BN is unstable), *and* the global batch effectively grew so the LR was now too small for it. The fix for the BN half is `SyncBatchNorm` (statistics computed across all ranks, restoring an effective BN batch of 256); the fix for the LR half is to scale the LR with the effective batch (the linear scaling rule: roughly, multiply the base LR by the ratio of effective-to-original batch, with warmup).
- **Localize symptom A (throughput).** Separately, I profile with the PyTorch profiler. GPU utilization is 34% — the GPUs are *starved*, waiting on the dataloader. The single-GPU script used 8 dataloader workers; the 8-GPU launch spawned 8 processes that *each* tried to use 8 workers on the same machine, oversubscribing the CPU and the disk, so data loading became the bottleneck. The fix: set workers per process so the total matches the host's capacity, and use `pin_memory` plus prefetching. See [the GPU is idle: throughput debugging](/blog/machine-learning/debugging-training/out-of-memory-debugging).
- **Re-measure.** With SyncBN + LR scaling + warmup, accuracy returns to 91.5% on 8 GPUs (matching the baseline). With the dataloader fix, utilization rises from 34% to 89% and wall-clock drops to about 6.2x faster than one GPU. Two symptoms, two bugs, both in systems, both found by *first* splitting systems out with a single-GPU repro and *then* attacking each symptom with its own instrument.

The meta-lesson across all three worked examples: the symptom chooses the first test (overfit-batch for "won't learn," detect_anomaly for "NaN," single-GPU for "multi-GPU-only"), the first test partitions the space, and you never touch a fix until a test has *localized* the bug to one place. Three runs that had collectively cost over a week of confusion before were each diagnosed in under ten minutes of disciplined bisection.

## 6. The finetuning addendum: the same six places, biting differently

Everything above applies to training from scratch and finetuning alike. But finetuning fails *differently*, and if you do not know the finetuning-specific tells you will run the right tests and misread them. The suspect map is unchanged — data, optimization, model code, numerics, systems, evaluation — but the characteristic failures shift. Figure 7 lays out the six places with their finetuning-specific failure, tell, and test.

![A matrix of the six bug places mapped to their finetuning-specific failure, the tell that gives it away, and the confirming test, covering chat-template skew, LR too high, LoRA no-op, dtype mismatch, resume LR jump, and preprocessing skew](/imgs/blogs/the-training-debugging-playbook-7.png)

**Optimization: the 100x-wrong learning rate.** The single most common finetuning bug. Pretraining LRs are often `1e-3`–`6e-4`; finetuning a pretrained model wants `1e-5`–`2e-5` for full finetuning and `1e-4`–`3e-4` for LoRA. People copy a pretraining config and finetune at `1e-3`, which takes a step so large it *destroys* the pretrained features. The tell is **catastrophic forgetting**: the model gets better at your task but worse at everything the base model could do — it forgets how to write code, or how to follow basic instructions. The test: drop the LR by 10x–100x and evaluate a *base-model* task alongside your task. If the base task recovers, the LR was too high. See [finetuning an LLM without breaking it](/blog/machine-learning/debugging-training/finetuning-an-llm-without-breaking-it).

The science of why finetuning wants a far smaller step is worth a paragraph, because it explains the *direction* of every finetuning LR fix. A pretrained model sits at a good point in weight space — a basin that encodes everything it learned from trillions of tokens. Finetuning data is small (thousands to millions of examples) compared to pretraining, so the finetuning gradient is *high-variance* and points in a direction that is good for your narrow task but not necessarily for the basin you are sitting in. A large step in that direction walks the weights *out of the good basin* before the model has seen enough finetuning data to find a new one that preserves the old capabilities. The size of the step is the LR times the gradient; since you cannot easily shrink the gradient, you shrink the LR. Quantitatively, the parameter-update norm per step is roughly $\eta \cdot \lVert g \rVert$, and you want that to be a small fraction (order $10^{-3}$ or less) of the parameter norm, so the model *adjusts* its representation rather than *overwriting* it. A pretraining LR applied to a pretrained model produces an update-to-parameter ratio orders of magnitude too large — which is precisely the mechanism of catastrophic forgetting. This is also why finetuning overfits in 1–3 epochs (the small data is memorized fast) and why warmup matters even more in finetuning (the first few large steps on a high-variance gradient do the most damage). The compute-optimal-scaling literature derives the from-scratch version of the same LR-versus-batch trade-off; here the lesson is narrower and sharper — when in doubt on a finetune, the LR is too high, not too low.

**Model code: the LoRA no-op.** The bug from this post's opening. The adapter is configured but never enters the computation graph — wrong `target_modules`, the base model not frozen properly, or the PEFT wrapper applied after the optimizer captured the parameters. The tell: the loss falls (the frozen base is already good) but the model does not change behavior. The test is one line, and it is non-negotiable before any LoRA run:

```python
from peft import get_peft_model, LoraConfig
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# Healthy:  trainable params: 4,194,304 || all params: 1,543,714,304 || trainable%: 0.27
# Broken:   trainable params: 0          || all params: 1,543,714,304 || trainable%: 0.0
```

If `trainable%` is `0.0`, your adapter matched nothing — fix `target_modules` to the architecture's real names (`q_proj`/`v_proj`/`k_proj`/`o_proj` for most Llama-family models). See [debugging LoRA and PEFT](/blog/machine-learning/debugging-training/finetuning-an-llm-without-breaking-it).

**Data: the chat-template / preprocessing skew.** In finetuning, the *format* matters as much as the content. If you train with one chat template and serve with another — different role tokens, a missing generation prompt, BOS added in training but not at inference — the model learns a distribution it never sees at serve time. The most visible symptom is **the model that won't stop generating**: if the EOS token was masked out of the labels (so the model was never trained to *emit* a stop), it runs to `max_length` every time. The test: decode one training batch and compare it byte-for-byte to a real serving prompt, and check that EOS appears *unmasked* in the labels. See [chat-template and formatting bugs](/blog/machine-learning/debugging-training/the-loss-masking-bug).

**Data: the loss-masking mistake.** Finetuning instruction data has a prompt and a completion. You want to train on the completion only (mask the prompt to `-100`), but it is easy to mask the *wrong* span — mask everything (train on nothing; this is the §3 worked example), or mask nothing (train on the prompt too, wasting capacity and teaching the model to generate prompts). The test is again to decode the unmasked label tokens and confirm they are exactly the completion. See [the loss-masking bug](/blog/machine-learning/debugging-training/the-loss-masking-bug).

**Numerics: dtype mismatch.** A LoRA adapter that stays in fp32 while the base is in bf16, or a frozen base loaded in int8/4-bit with an adapter that does not match, produces silent slowdowns or subtly wrong gradients. The test: print the dtype of the trainable parameters and confirm it matches the autocast policy.

**Evaluation: the regression you didn't measure.** Finetuning can improve your target metric while silently regressing a capability you care about. The discipline is to **always evaluate a held-out base-model task alongside the finetuning task**, so catastrophic forgetting shows up as a number, not a customer complaint. The capstone of the eval track, finetuning-pitfalls-across-modalities, collects these into a unified checklist across vision, LLM, speech, and tabular.

#### Worked example: a finetune that "works" but won't stop

A `Llama-3-8B` SFT on a customer-support dataset. Symptom: eval responses are correct *content* but every response runs to the 512-token limit, repeating and rambling at the end. Training loss looked perfect (0.41 final).

- **Symptom precision.** "Won't stop generating" is a known signature: the model never learned to emit EOS. That points at *data* (the labels), not the model.
- **Test 3 (decode one batch, focus on labels).** I decode the label tokens near the end of each example. The completion text is there and unmasked — good — but the final EOS token has label `-100`. The data formatting masked the EOS along with the padding, so the model was trained to produce the completion *and then was never penalized for failing to stop*.
- **The science.** Generation halts when the model samples EOS. If EOS is never an unmasked target, the cross-entropy never rewards producing it, so at inference the model assigns EOS low probability and keeps going until `max_length`. The loss looked perfect because predicting the *content* is what dominates the loss; the missing EOS is a tiny fraction of tokens and barely moves the average.
- **Fix (one thing).** Include the EOS token as an unmasked label at the end of each completion (and keep the padding masked).
- **Re-measure.** Average generation length drops from 512 (the cap) to a healthy 64–120 tokens; responses now end cleanly; the content quality is unchanged. The loss barely moved (0.41 → 0.43) — which is exactly why the loss curve never warned you. This is the canonical lesson of the series: the loss is a necessary but wildly insufficient check.

## 7. Build a debuggable system from day one

The fastest debugging is the debugging you never have to do, because the system surfaces its own bugs in minutes instead of hiding them for days. Everything above is reactive; this section is the proactive investment that makes the reactive part cheap. Figure 8 contrasts a silent-and-slow run with a loud-and-fast one.

![A two-column before and after comparing a silent slow run with no determinism that finds a NaN at step 4000 after three days against a seeded run with kill-on-NaN that finds the bug at step 1 in five minutes](/imgs/blogs/the-training-debugging-playbook-8.png)

**Determinism by default in CI.** Make every test and smoke run bit-reproducible. You cannot tell whether a fix helped if two runs of the same config diverge by noise. The §3 `set_full_determinism` is one function call; run it in every smoke test. The cost (a few percent throughput) is paid only in debugging configs, not production.

**Instrument everything, log the right things.** Log the global and per-layer gradient norm, the parameter and update norms, the grad-to-param ratio, the LR, activation/weight histograms, throughput, and data statistics — to W&B or TensorBoard. The point is not to look at all of it always; it is to have it *recorded* so that when a run goes wrong you can scrub back and see exactly which signal moved first. [Instrumenting a training run](/blog/machine-learning/debugging-training/instrumenting-a-training-run-what-to-log) is the full menu; the minimum is grad norm and LR on every step.

**Put the preflight in CI.** The five tests from §3 belong in your test suite, not just in your memory. A tiny CI job that runs determinism + overfit-one-batch on a 2-example fixture for every model architecture catches the LoRA no-op, the loss-mask bug, and the wrong-reduction bug *before* anyone launches a real run.

```python
def test_model_can_overfit_one_batch():
    set_full_determinism(0)
    model, batch, opt, loss_fn = build_tiny_fixture()
    final = overfit_one_batch(model, batch, opt, loss_fn, steps=200)
    assert final < 0.05, f"model failed to overfit one batch: loss={final:.3f}"

def test_labels_are_not_all_masked():
    batch = next(iter(build_train_loader()))
    labels = batch["labels"]
    unmasked = (labels != -100).sum().item()
    assert unmasked > 0, "every label is -100; training on nothing"
```

**Guardrails: kill on NaN.** Do not let a run burn hours after it has already gone non-finite. A two-line guard turns "NaN at step 4000, discovered the next morning" into "NaN at step 412, killed immediately with the step and batch logged for repro."

```python
loss = loss_fn(out, batch["labels"])
if not torch.isfinite(loss):
    # Log the offending step + batch, dump grad norms, then stop loudly.
    raise FloatingPointError(f"non-finite loss {loss.item()} at step {step}")
# Optional: a grad-norm tripwire to catch divergence before the NaN.
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9)
if total_norm > 1e3:
    print(f"WARNING: grad norm {total_norm:.1e} at step {step} (divergence risk)")
```

**The resume-equivalence test.** Checkpoint-and-resume is a notorious silent bug: people restore the model weights but forget the optimizer state, the LR scheduler position, or the RNG state, and the loss *jumps* on resume because Adam's moments reset and the schedule restarts. Bake an automated test: run 100 steps, checkpoint, resume, run 100 more; in parallel run 200 steps uninterrupted; assert the two losses at step 200 match to within floating-point tolerance. If they do not, your resume is not continuous. (This sibling on debugging checkpoint and resume is shipping with this wave.)

```python
def test_resume_is_continuous(tol=1e-4):
    set_full_determinism(0)
    cont_loss = run_steps(steps=200)                  # uninterrupted reference
    set_full_determinism(0)
    run_steps(steps=100, ckpt_out="ck.pt")            # save EVERYTHING: model+opt+sched+rng
    resumed_loss = run_steps(steps=100, ckpt_in="ck.pt")
    assert abs(cont_loss - resumed_loss) < tol, (
        f"resume not continuous: {cont_loss:.5f} vs {resumed_loss:.5f}")
```

**Small-config smoke tests before the big run.** Never launch the 8-GPU, 3-day run as the first execution of a new config. Run a 50-step, 1-GPU, tiny-model version first. It exercises the entire pipeline — data, collation, forward, loss, backward, step, checkpoint, eval — in two minutes, and it catches shape bugs, mask bugs, OOM, and config typos before they cost a fortune. The smoke test is the cheapest insurance in machine learning.

The compounding return here is real. A team that adopts determinism + a CI preflight + kill-on-NaN + the resume test + smoke tests does not just debug faster; it *creates far fewer bugs that reach a long run*, because the bugs are caught at the cheapest possible moment — in CI, on a fixture, in two minutes — instead of at the most expensive — after a multi-day distributed run.

#### Worked example: the loss that jumps on every resume

Why the resume-equivalence test earns its place. A long-running pretraining job checkpoints every 2,000 steps and resumes after preemptions. Symptom: every resume produces a visible *upward jump* in the loss — from 2.41 to 2.78 at the resume boundary — that then takes a few hundred steps to recover. Over a run with twenty preemptions, this wastes thousands of steps and looks alarming on the dashboard.

- **Symptom precision.** A loss *jump precisely at resume boundaries* is a near-certain resume-state bug. It is not data (data is unchanged), not numerics (no NaN), not the model (weights restored). The jump is the tell that *something stateful was not restored*.
- **The science of what jumps.** Adam keeps two exponential moving averages per parameter — the first moment $m$ and the second moment $v$ — and its effective per-parameter step is $\eta \cdot \hat m / (\sqrt{\hat v} + \epsilon)$. If you restore the weights but *reinitialize the optimizer* (`m = v = 0`), the first few post-resume steps run with a cold optimizer: the bias-correction term spikes, $\sqrt{\hat v}$ is tiny, and the effective step is briefly enormous — which knocks the loss up before Adam re-warms its moments. The LR scheduler is the second culprit: if it restarts from step 0 instead of resuming at step 2,000, the warmup replays and the LR is wrong for hundreds of steps. The RNG state is the third: if dropout and data shuffling reseed, the resumed run sees a different data order and different dropout masks, adding noise at the boundary.
- **The test that catches it.** The resume-equivalence test from above: 200 straight steps must equal 100 + checkpoint + resume 100, to floating-point tolerance. With the buggy checkpoint, the two diverge at step 100; with the fix, they match to `< 1e-4`.
- **Fix (all three pieces of state).** Save and restore the optimizer `state_dict`, the LR scheduler `state_dict`, *and* the RNG states (`torch.get_rng_state`, `torch.cuda.get_rng_state_all`, NumPy, Python `random`) — plus the GradScaler state if using AMP and the EMA weights if you keep them.
- **Re-measure.** The resume jump from 2.41→2.78 disappears; resumed loss continues at 2.41 seamlessly; the resume-equivalence test passes. Over the same twenty-preemption run, the thousands of wasted recovery steps are recovered. See the sibling on debugging checkpoint and resume (shipping this wave) for the full state inventory.

```python
def save_full_checkpoint(path, model, optimizer, scheduler, scaler, step):
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),     # Adam moments m, v
        "scheduler": scheduler.state_dict(),     # the LR schedule position
        "scaler": scaler.state_dict() if scaler else None,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }, path)
```

## 8. Anti-patterns: the things that waste days

Every senior engineer has a list of self-inflicted wounds. Here are the ones that cost the most, framed as the anti-pattern and its antidote.

**Changing five things at once.** You tweak the LR, the batch size, the augmentation, the model, and the precision, then re-run. The result is uninterpretable: if it improves, you cannot attribute the win; if it regresses, you cannot attribute the loss. *Antidote:* change exactly one variable, re-measure, attribute the delta, then change the next. The scientific method is not optional rigor — it is the only way to learn from a run.

**Trusting the loss curve alone.** The most expensive belief in ML. A falling loss is consistent with a leak, a no-op adapter, training on the prompt, a metric bug, and a model that won't stop generating. *Antidote:* always pair the loss with an independent check — overfit-one-batch, a decoded sample, an honest held-out metric, the per-layer grad norm. If the only evidence the run is healthy is the loss curve, you do not yet know the run is healthy.

**No determinism, so you can't tell if a fix helped.** You make a change, the run looks 0.3% better, and you ship it — but the run-to-run noise is 0.5%, so you have shipped noise. *Antidote:* fix the seed during debugging. A change is real only if it survives a deterministic before/after.

**Skipping overfit-one-batch.** People skip it because it "feels like cheating" or "obviously the model works." It is the single highest-leverage test, it takes three minutes, and it cleanly partitions the suspect space. Skipping it means you debug the whole pipeline when you could have eliminated half of it first. *Antidote:* run it before anything else when the symptom is "won't learn."

**Launching the big run before the smoke test.** The first execution of a new config should never be the 3-day, 8-GPU run. *Antidote:* a 50-step, 1-GPU smoke test on a tiny model first, every time. It is the cheapest bug-catcher you own.

**Reading code after a test has cleared it.** Overfit-one-batch passed, so the model is fine — but you keep re-reading the model because reading code *feels* productive. It is not; the test already told you the model is innocent. *Antidote:* trust your tests. When a test clears a place, stop searching there and move to the next partition.

**Reaching for the fix before the diagnosis.** "Let me just lower the LR and see." Sometimes it works, and you learn nothing about *why* — so the bug comes back in a different form next week. *Antidote:* diagnose first (what is the symptom, what is the hypothesis, what is the confirming test), fix second, re-measure third. A fix you do not understand is a bug you will meet again.

**Debugging the slow part instead of the broken part.** A run is 31% GPU-utilized and someone spends a day optimizing the model's matmuls — when the bottleneck is the dataloader starving the GPU. *Antidote:* profile before you optimize, just as you bisect before you fix. The instrument tells you where the time goes ([the GPU is idle](/blog/machine-learning/debugging-training/out-of-memory-debugging) covers the throughput case); guessing wastes the day.

## 9. Case studies and real signatures

Four patterns that recur in the literature and in practice, each a concrete instance of the framework above. These are the named bugs worth recognizing on sight.

**Label errors in benchmark test sets (confident learning).** Northcutt, Athalye, and Mueller's 2021 work on pervasive label errors used confident learning (the `cleanlab` library) to find label errors in the *test sets* of major benchmarks — on the order of a few percent of labels across ImageNet, MNIST, CIFAR, and others, with the widely-cited estimate of roughly 3.4% average across the ten datasets they studied. The signature: a model's "errors" cluster on examples that are actually mislabeled, and accuracy is capped not by the model but by the noise floor of the labels. The test: rank examples by the model's confident disagreement with the given label and *look at them*. The lesson for this playbook: when accuracy plateaus below expectation and overfit-one-batch passes, suspect the data — including the labels you are scoring against. See [garbage in: finding label noise](/blog/machine-learning/debugging-training/data-leakage-the-silent-killer).

**fp16 gradient underflow and loss scaling (Mixed Precision Training, Micikevicius et al., 2018).** The paper that made fp16 training practical identified the exact mechanism behind the §5 NaN worked example at the gradient level: many gradient magnitudes in fp16 fall below the smallest representable normal value ($\approx 6.1\times10^{-5}$) and round to zero, silently stalling learning. The fix — loss scaling — multiplies the loss by a large factor before the backward pass to shift gradients into the representable range, then unscales before the optimizer step. The signature: training in fp16 that quietly underperforms fp32, with a gradient histogram piled up at zero. The modern alternative, bf16, sidesteps this by keeping fp32's exponent range (so no underflow) at the cost of mantissa precision. The lesson: precision is a *numerics* bug location, and the gradient histogram is its instrument.

**The left-padding-breaks-generation bug (decoder-only LLMs).** A pattern every LLM practitioner hits once: a decoder-only model trained or batched with *right* padding generates fine one-at-a-time but produces garbage in a batch, because the padding tokens shift the position ids and the model attends to or starts from pad. The fix is left-padding for batched generation (so real tokens are flush-right and positions align) plus an attention mask that the model actually respects. The signature: single-example generation is correct, batched generation is wrong — a classic train/infer mismatch. The lesson: "works one at a time, breaks in a batch" is a systems/eval-path signature, not a model-capability signature. See [attention-mask and padding bugs for LLMs](/blog/machine-learning/debugging-training/attention-mask-and-padding-bugs-for-llms).

**Kaggle leakage post-mortems (target/temporal leakage).** Competition write-ups are a goldmine of leakage signatures: an ID column that correlates with the target because of how the data was assembled, a feature computed using future information, a row that appears in both train and test. The universal signature is a cross-validation or leaderboard score that is *too good* and that collapses on truly held-out data — exactly the "99% val, too good" row of the master table. The fix is split discipline: `GroupKFold` so related rows never straddle the split, `TimeSeriesSplit` so no fold trains on the future, and fitting all preprocessing *inside* the training fold. The lesson: when a metric is suspiciously high, the null hypothesis is a leak, not a breakthrough. See [data leakage, the silent killer](/blog/machine-learning/debugging-training/data-leakage-the-silent-killer).

**The preprocessing-before-split leak (tabular, the AUC that drops 0.2).** A pattern from countless data-science pipelines: you fit a `StandardScaler`, an imputer, or a target encoder on the *whole* dataset and *then* split into train and test. The test rows' statistics have now leaked into the transform that the model trains on, so cross-validation reports an optimistic AUC that collapses when the preprocessing is correctly fit *inside* each fold. The magnitude is real — a leak like target encoding fit on all data can inflate AUC by 0.1–0.2 absolute, turning an honest 0.78 into a fraudulent 0.95. The signature: a strong CV score that drops sharply when you wrap every transform in a scikit-learn `Pipeline` so it is fit only on the training fold. The fix is structural — `Pipeline` + `cross_val_score` so leakage is impossible by construction, never fit-then-split. The lesson for the playbook: when the cheapest test (re-run CV with preprocessing *inside* the fold) drops the score, the original number was a leak, not a model.

**The CTC inf-loss trap (speech).** A connectionist-temporal-classification model (the standard for many ASR systems) returns `inf` loss on some batches and the run never recovers. The mechanism is exact and worth knowing: CTC aligns an input sequence of length $T$ to a target of length $U$ by inserting blanks, and a valid alignment requires $T \geq U + (\text{number of repeated adjacent target tokens})$. If a downsampled audio clip is *shorter* than its transcript demands, there is *no valid alignment*, the probability of the target is zero, and the loss is $-\log 0 = +\infty$. The signature: `inf` loss correlated with short clips or long transcripts, not with anything in the model. The test: assert `input_lengths >= target_lengths` before the loss and log the offending examples. The fix: filter or pad so every input is long enough, or reduce the downsampling factor. The lesson: an `inf` that depends on the *data shape* rather than the *training step* is a structural numerics bug in the loss, found by checking the loss's preconditions — see [hunting NaNs and Infs](/blog/machine-learning/debugging-training/hunting-nans-and-infs).

The thread through all six: each is a specific value of (place, tell, test, fix). Label noise is (data, plateau-with-overfit-passing, rank-by-disagreement, clean/relabel). fp16 underflow is (numerics, gradient-histogram-at-zero, loss-scale-or-bf16). Left-padding is (model/eval-path, one-at-a-time-fine-batch-broken, left-pad-and-mask). Leakage is (data, too-good-score, group/time-split). Preprocessing-before-split is (data, score-drops-when-fit-in-fold, pipeline). CTC inf is (numerics, inf-on-short-clips, length-precondition-assert). The playbook does not memorize bugs; it gives you the coordinates to *locate* any new one. Once you can name the place, the tell, the test, and the fix for a bug, you have understood it — and you can recognize its cousin the next time it wears a different costume.

## 10. When this is — and isn't — your bug

A decisive section, because the fastest debugging is knowing where *not* to look. These are the load-bearing "if/then" calls that the framework licenses.

**If overfit-one-batch passes, it is not the model code.** The model can learn, the gradients flow, the loss is wired right. Stop reading the model. Look at data, generalization, the full-set schedule, or evaluation. The single most common time-sink in ML debugging is re-reading model code that a passing overfit test has already exonerated.

**A smooth-then-NaN curve is numerics, not data.** Data bugs are *consistent* from step 1 — wrong from the start. A clean run that suddenly goes non-finite at step N is a value crossing a representable boundary (a `log(0)`, an fp16 underflow, an `exp` overflow). Reach for `detect_anomaly`, not the dataloader.

**If the bug vanishes on one GPU, it is systems.** Single-GPU repro is the cleanest partition there is. Gone on one device → gradient-sync, sharding, accumulation math, or rank desync. Persists on one device → the logic is wrong everywhere; systems is innocent.

**A loss parked at exactly $\ln C$ is "learning nothing," not "learning slowly."** Uniform output is a *specific* failure (masked labels, dead gradients, a constant loss), not a too-low LR. Run overfit-one-batch; if it fails too, you are in model/loss, not optimization. A genuinely too-low LR *crawls* below $\ln C$ and *passes* overfit-one-batch.

**A too-good metric is a leak until proven otherwise.** 99% on a hard problem is the null hypothesis "leak," not "breakthrough." Dedup across splits, ablate suspicious features, replay a real serving input. The burden of proof is on the high number.

**If the loss curve is the only evidence, you have no evidence.** A falling loss is consistent with too many broken configurations to count. Demand an independent check before you believe a run is healthy.

And the inverse — when a symptom is *not* the bug you think: "the model degraded in production" is usually a *data* story (distribution shift, train-serve skew), not a model-weights story. "It only fails at scale" is usually *systems*, not the model. "Val is worse than train" is, far more often than overfitting, a forgotten `model.eval()` putting BatchNorm and dropout in the wrong mode. Knowing these default attributions saves you from chasing the dramatic explanation when the boring one is correct.

## 11. The printable checklist

Tape this to the wall. It is the whole playbook compressed to one column.

**Before any big run (the preflight):**

- [ ] Set full determinism (seed, deterministic algorithms, worker seeds, `CUBLAS_WORKSPACE_CONFIG`).
- [ ] Overfit one batch to near-zero loss. (Fails → model/loss/optim, not data.)
- [ ] Print and decode one batch — inputs *and* labels. (All `-100`? Wrong normalization? Double-BOS?)
- [ ] Log per-layer grad norms after the first backward. (Any `grad=None`? Norm `1e4` or `1e-7`?)
- [ ] Reproduce on a single GPU. (Bug gone → systems.)
- [ ] Run a 50-step, 1-GPU, tiny-model smoke test of the full pipeline.
- [ ] For finetuning: `print_trainable_parameters()` (0 → LoRA no-op); confirm LR ≈ 1e-5–3e-4; check EOS is unmasked.

**When a run is failing (the bisection):**

- [ ] State the symptom *precisely* (with the number: loss = $\ln C$? NaN at step N? acc = 1/C?).
- [ ] Form *one* falsifiable hypothesis.
- [ ] Run the *cheapest discriminating test* (overfit-batch for "won't learn"; `detect_anomaly` for NaN; single-GPU for multi-GPU-only).
- [ ] Localize to one of the six places; stop searching the cleared places.
- [ ] Fix *one* thing; re-measure; confirm the symptom moved in the predicted direction *and* magnitude.

**Build it debuggable:**

- [ ] Determinism + overfit-one-batch + "labels not all masked" assertions in CI.
- [ ] Kill-on-NaN guard; grad-norm tripwire.
- [ ] Resume-equivalence test (200 straight == 100 + resume 100).
- [ ] Log grad norm + LR every step to W&B/TensorBoard.

## 12. Key takeaways

- **Training failures are silent; a falling loss is not proof of correctness.** Always pair the loss with an independent check (overfit-one-batch, a decoded sample, an honest held-out metric). If the loss is your only evidence, you have no evidence.
- **A bug hides in exactly one of six places — data, optimization, model code, numerics, systems, evaluation — so bisect before you touch code.** Six suspects means at most $\lceil\log_2 6\rceil = 3$ discriminating tests to localize.
- **Overfit-one-batch is the master fork.** Pass clears model + optimization; fail indicts them. Run it first when the symptom is "won't learn," and stop reading model code once it passes.
- **Run the five-test preflight before any big run:** determinism, overfit-one-batch, print/decode a batch, per-layer grad norms, single-GPU repro. Minutes of testing save days of doomed runs.
- **A loss parked at $\ln C$ is "learning nothing," not "learning slowly."** Memorize the chance-loss numbers: $\ln 2 \approx 0.69$, $\ln 5 \approx 1.61$, $\ln 9 \approx 2.20$, $\ln 1000 \approx 6.91$.
- **Smooth-then-NaN is numerics; gone-on-one-GPU is systems; too-good-metric is a leak.** Let the symptom pick the cheapest discriminating test.
- **In finetuning, the LR is usually 100x too high and the LoRA adapter is often a no-op.** Drop the LR to ~1e-5 and run `print_trainable_parameters()` before you trust a finetune.
- **Change one thing at a time and measure before/after under a fixed seed.** A fix is real only if it survives a deterministic comparison.
- **Build the system debuggable from day one:** determinism, a CI preflight, kill-on-NaN, the resume-equivalence test, and a small-config smoke test. The cheapest bug is the one caught in CI on a fixture, not after a 3-day run.
- **Diagnose before you fix.** A fix you do not understand is a bug you will meet again.

## Further reading

- **Within this series — start here:** [A taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) (the conceptual frame) and [the overfit-a-single-batch test](/blog/machine-learning/debugging-training/the-overfit-a-single-batch-test) (the master fork).
- **The instruments:** [reading the loss curve as a diagnostic](/blog/machine-learning/debugging-training/reading-the-loss-curve-as-a-diagnostic), [instrumenting a training run — what to log](/blog/machine-learning/debugging-training/instrumenting-a-training-run-what-to-log), and [reproducibility and determinism in training](/blog/machine-learning/debugging-training/reproducibility-and-determinism-in-training).
- **By place:** [data leakage, the silent killer](/blog/machine-learning/debugging-training/data-leakage-the-silent-killer); [the learning rate is almost always the problem](/blog/machine-learning/debugging-training/the-learning-rate-is-almost-always-the-problem); [hunting NaNs and Infs](/blog/machine-learning/debugging-training/hunting-nans-and-infs); [loss spikes and divergence](/blog/machine-learning/debugging-training/loss-spikes-and-divergence); [your model isn't learning what you think](/blog/machine-learning/debugging-training/your-model-isnt-learning-what-you-think); [the loss-masking bug](/blog/machine-learning/debugging-training/the-loss-masking-bug); [debugging DDP and multi-GPU](/blog/machine-learning/debugging-training/debugging-ddp-and-multi-gpu); and [out-of-memory debugging](/blog/machine-learning/debugging-training/out-of-memory-debugging).
- **Confident learning / label noise:** Northcutt, Athalye, Mueller, "Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks" (2021), and the `cleanlab` library docs.
- **Mixed precision:** Micikevicius et al., "Mixed Precision Training" (ICLR 2018) — the loss-scaling mechanism and the fp16 representable-range argument.
- **PyTorch docs:** `torch.autograd.set_detect_anomaly`, `torch.use_deterministic_algorithms`, `torch.cuda.memory._record_memory_history`, and the reproducibility notes — the canonical references for the tools in the preflight.

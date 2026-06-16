---
title: "Reproducibility and Determinism in Training: Why Your Run Isn't Repeatable"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Find every source of nondeterminism in a PyTorch run, pin it bit-for-bit, and learn exactly when chasing determinism is worth the cost and when it isn't."
tags:
  [
    "debugging",
    "model-training",
    "reproducibility",
    "determinism",
    "pytorch",
    "cuda",
    "mixed-precision",
    "finetuning",
    "deep-learning",
    "numerics",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/reproducibility-and-determinism-in-training-1.png"
---

You change one line of code — swap the optimizer, add a warmup, fix what you think is a masking bug — rerun, and the final loss is `2.187` instead of `2.193`. Better! You ship the change. A week later you can't reproduce the improvement, because when you rerun the *old* code it also gives `2.187` sometimes and `2.201` other times. The "fix" was noise. You optimized against a coin flip.

This is the quietest, most corrosive bug in the whole series, because it is not a bug in your model at all. It is a bug in your **ability to measure**. Every other post here assumes you can do the most basic thing in experimental science: hold everything fixed, change one variable, and read the effect. If re-running the *identical* script gives a different loss, you have lost that ability. You cannot bisect to one of the six places a bug hides — data, optimization, model code, numerics, systems, evaluation — when the ground itself moves under each run. You cannot tell a real 0.6% improvement from run-to-run jitter. You cannot write a regression test that asserts anything. Determinism is not a nice-to-have; it is the precondition for debugging at all.

The frustrating part is that "set a seed" feels like it should be enough, and it is not — not by a long way. Figure 1 shows why: randomness leaks into a single training step at **five independent layers**, and `torch.manual_seed(42)` touches exactly one of them. The DataLoader's worker processes carry their own generators. cuDNN picks a different convolution algorithm depending on a microbenchmark that runs at startup. Atomic-add reductions on the GPU sum gradients in whatever order threads happen to finish. TF32 silently truncates your matmuls to 19 bits. By the time you've named all five, "reproducibility" stops being a checkbox and becomes a small engineering discipline.

![Stack diagram showing five layers where randomness enters a single training step, from Python and NumPy seeds through torch and CUDA RNG, DataLoader workers with four separate generators, cuDNN kernel selection with atomic-add reductions, and TF32 non-associative addition, all feeding into whether the loss at step one is bit-identical](/imgs/blogs/reproducibility-and-determinism-in-training-1.png)

By the end of this post you will be able to take any nonrepeatable PyTorch run and (1) name precisely which of the five sources is breaking it, by bisection; (2) write a single `set_full_determinism(seed)` utility that pins all of them; (3) verify bit-for-bit equality with a two-run diff that checks the loss down to the last mantissa bit; and (4) — just as important — decide when you should *not* do any of this, because in production the 10–40% throughput cost buys you nothing. We'll build the science first (floating-point non-associativity, worked at the bit level), then the diagnostics, then the fix, then the honest cost accounting.

This is post A5 in the series. It sits right after [instrumenting a training run](/blog/machine-learning/debugging-training/instrumenting-a-training-run-what-to-log) — because the instruments only tell you something if the run they measure is repeatable — and it underpins every later track. When you read the [taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs), notice that *every* confirming test in it implicitly assumes you can rerun and get the same answer. This post is how you earn that assumption.

## 1. The symptom: a run that won't repeat

Let me make this concrete with the running example we'll debug all the way through. You have a small image classifier — a ResNet-18 on a 50,000-image dataset — and a clean, well-organized training script. You run it twice from a cold start, same machine, same code, same command line, no edits in between:

```bash
python train.py --seed 42 --epochs 1 --batch-size 128
# run A: step 1 loss = 2.3041, final val acc = 71.4%
python train.py --seed 42 --epochs 1 --batch-size 128
# run B: step 1 loss = 2.3055, final val acc = 70.9%
```

Two things jump out. First, the **final** numbers differ by half a point of accuracy — annoying but you might write that off as "deep learning is noisy." Second, and far more diagnostic, the **step-1** losses already differ: `2.3041` versus `2.3055`, a gap of about `1.4e-3`. That second fact is the tell. At step 1, the optimizer has taken zero updates. The model weights are identical if your init seed worked. The *only* thing that can make step-1 loss differ is that the two runs saw a **different first batch**, or computed the **same batch's loss with a different numerical result**. Both are reproducibility failures, and they have different fixes.

Here is the disciplined way to think about it. A training step is a deterministic mathematical function of three things: the model parameters, the input batch, and the sequence of floating-point operations the hardware executes. If the parameters are pinned (init seed) and you still get different losses, then either the batch changed (a *data-order* problem) or the operations changed (a *numerics/kernel* problem). The size and timing of the divergence tells you which. A *large* step-1 difference like `1.4e-3` is almost always data order — a different batch is a first-class change. A *tiny* step-1 difference, `1e-7` or so, that grows over training is numerics — same batch, slightly different sums.

This is the bisection logic of the whole series applied to reproducibility specifically. We're going to localize before we touch anything. But first we need the science, because you cannot reason about "slightly different sums" without understanding why addition itself is not reliable on a computer.

#### Worked example: reading the divergence to pick the suspect

Run the script twice and log step-1 loss to 7 significant figures (not the default 4 — the bug is in the digits you're truncating). Suppose you see:

| Scenario | Run A step-1 loss | Run B step-1 loss | Gap | Suspect |
| --- | --- | --- | --- | --- |
| Different first batch | `2.304100` | `2.305500` | `1.4e-3` | Data order (workers / shuffle) |
| Same batch, different sum | `2.3041582` | `2.3041579` | `3e-7` | Numerics (atomic-add / TF32) |
| Truly identical | `2.3041582` | `2.3041582` | `0.0` | Already deterministic |

The gap's *magnitude* is the diagnostic. `1.4e-3` is far larger than fp32 rounding could explain for a cross-entropy over a few hundred examples — that's a different batch. `3e-7` is right at the edge of float32 precision (about 7 decimal digits) — same batch, reductions in a different order. You've now bisected `data` vs `numerics` with a single two-run diff, before reading a line of model code.

## 2. The science: floating-point addition is not associative

Everything about numerical nondeterminism rests on one fact that surprises people the first time they meet it: **floating-point addition is not associative.** In exact arithmetic, `(a + b) + c` equals `a + (b + c)` for all real numbers. In IEEE-754 floating point, they can differ, because each `+` rounds its result to the nearest representable value, and rounding error depends on the magnitudes you're adding.

A float32 number has a 24-bit significand (23 stored bits plus an implicit leading 1). It can hold about 7 decimal significant figures. When you add a large number and a small number, the small one gets right-shifted to line up exponents, and bits that fall off the bottom of the 24-bit window are *gone* — rounded away. So the order in which you accumulate a sum decides which bits survive.

Here is the canonical worked case at the bit level. Take three float32 values:

$$a = 1.0, \quad b = 10^{-8}, \quad c = -1.0$$

Compute left-to-right, `(a + b) + c`:

- `a + b = 1.0 + 1e-8`. But `1e-8` in units of `a`'s exponent is about `2^{-27}` relative to 1.0, and float32 only keeps 23 fraction bits — anything below `2^{-23} ≈ 1.2e-7` relative to the leading 1 is unrepresentable. So `1.0 + 1e-8` rounds to exactly `1.0`. The `1e-8` vanished.
- `(1.0) + c = 1.0 + (-1.0) = 0.0`.

Now compute the other grouping, `a + (b + c)`:

- `b + c = 1e-8 + (-1.0) = -0.99999999`, which *is* representable to float32 precision near magnitude 1.
- `a + (-0.99999999) = 1.0 - 0.99999999 = 1e-8` (approximately; the exact stored value is the float32 nearest to `1e-8`).

So `(a + b) + c = 0.0` but `a + (b + c) ≈ 1e-8`. Same three numbers, different grouping, a result that differs by `1e-8` — and in a deep net that `1e-8` is a gradient component that gets multiplied, accumulated over a million parameters, fed through an optimizer with momentum, and amplified epoch over epoch.

#### Worked example: why a 256-thread reduction is nondeterministic

Now scale this to a GPU. A gradient reduction — say summing the loss contribution across a batch of 256, or accumulating into an embedding row that 256 tokens all point at — is computed by many threads doing **atomic adds** into the same memory location. An atomic add guarantees each addition is applied without being lost (no two threads clobber each other), but it does **not** guarantee the *order*. Thread 173 might land before thread 12 on one run and after it on the next, because thread scheduling depends on warp occupancy, the clock, and what else the GPU is doing.

So the same 256 partial sums get added in a different order each run. By the non-associativity we just demonstrated, a different order gives a (very slightly) different float32 result — typically in the last 1–2 mantissa bits, a relative difference around `1e-7`. That's the `3e-7` step-1 gap from the worked example in section 1. It is not a bug in your code. It is the GPU faithfully computing a sum whose value is order-dependent, in an order it never promised to keep.

This is the deep reason `torch.use_deterministic_algorithms(True)` exists and why it can make a run *slower*: to get a fixed result, PyTorch must swap the fast atomic-add kernel for one that reduces in a fixed, serialized order (a tree reduction with a deterministic schedule, or a deterministic scatter). That kernel does less work in parallel, so it costs time. Determinism and the fastest-possible reduction are genuinely in tension; you cannot have both for these ops.

It's worth being precise about *how big* this last-bit error is, because the magnitude is your diagnostic fingerprint. For a sum of $n$ float32 numbers each of magnitude around 1, the rounding error of a naive sequential accumulation grows like $O(n \cdot \epsilon)$ where $\epsilon \approx 1.2 \times 10^{-7}$ is the float32 machine epsilon (the gap between 1.0 and the next representable float). A tree reduction does better, $O(\log n \cdot \epsilon)$, but the point is that *both* are nonzero and *order-dependent*. For a batch reduction over a few hundred terms, the spread between two different valid orderings is typically a few units in the last place — relative error in the `1e-7` range. That is exactly the `3e-7` step-1 gap from section 1. When you see a gap of that *size*, you can be confident it's a reduction-order issue, not a different batch (which would be orders of magnitude larger) and not a precision change (which would be `1e-3`-ish). The numbers don't just tell you *that* something is nondeterministic; their magnitude tells you *which layer* of figure 1 is responsible.

A useful way to make this tangible: try `sum()` in different orders in NumPy and watch the result change.

```python
import numpy as np
rng = np.random.default_rng(0)
x = rng.standard_normal(100_000).astype(np.float32)

forward  = np.float32(0.0)
for v in x:
    forward = np.float32(forward + v)        # left-to-right

backward = np.float32(0.0)
for v in x[::-1]:
    backward = np.float32(backward + v)      # right-to-left

print(forward, backward, abs(forward - backward))
# the two sums differ in the last few bits — same numbers, different order
```

That tiny difference is the entire story of GPU reduction nondeterminism in miniature: the GPU is doing this sum across hundreds of threads whose finish order it never fixed, so it picks one of these many orderings each run.

The same non-associativity explains **cross-machine** irreproducibility, which we'll return to in section 8. Two different GPU architectures, or two cuDNN versions, may use different reduction trees or different algorithm tilings even in "deterministic" mode. They each produce a stable, repeatable result *on that hardware*, but not the *same* result as each other. Run-to-run determinism on one box is achievable; bit-identical results across boxes mostly are not.

## 3. The first real source: DataLoader worker RNG

Now the practical hunt. The single most common cause of "my run won't repeat" — and the root cause of our `1.4e-3` step-1 gap — is the DataLoader's worker processes and their seeds. This one deserves the most space because it is subtle, the failure is silent, and the fix has a gotcha that changed between PyTorch versions.

When you set `num_workers > 0`, PyTorch forks (or spawns) that many worker processes to load and transform data in parallel. Each worker runs *its own* Python interpreter state, which means *its own* `random`, `numpy.random`, and `torch` RNG. If any of your dataset's `__getitem__` uses randomness — random crops, random augmentation, random masking, random negative sampling — that randomness comes from the worker's RNG, not your main process's.

Figure 4 shows the structure you actually need: one base seed must fan out to four independent generators, and the worker generators are the ones people forget.

![Graph showing a base seed of 42 fanning out into four independent random number generator streams: Python random, NumPy random, torch and CUDA via manual_seed, and a Generator object for DataLoader shuffle, with the worker_init_fn deriving each worker's seed from the base seed plus the worker id](/imgs/blogs/reproducibility-and-determinism-in-training-4.png)

Here's the classic bug, and it has two flavors. **Flavor one: workers reseed from time.** If a worker's augmentation code calls `random.seed()` with no argument, or some library does, the worker reseeds from the system clock — so every run, every worker, gets a *different* augmentation stream. That's a large, obvious irreproducibility. **Flavor two, far more insidious: workers share or desync the same seed.** This was an actual, widely-shipped bug. For years, the idiomatic `worker_init_fn` people copied did `np.random.seed(seed)` with the *same* `seed` for every worker. Result: all 4 workers produced the **identical** stream of "random" numbers. If your dataset draws a random crop in `__getitem__`, workers 0–3 all crop the same images the same way — your augmentation diversity silently collapsed to 1/4 of what you intended, and the model trained on far less variety than your logs claimed. The fix and the failure look nearly identical in code, so people got it wrong constantly.

PyTorch fixed the default in version 1.9+: each worker now gets a `base_seed + worker_id` automatically for `torch`'s own RNG, and `torch.utils.data.get_worker_info().seed` exposes it. But the default only reseeds `torch`'s generator — **`random` and `numpy` in the worker are still left at whatever they inherited**, which is identical across workers if you forked. So if your augmentation uses NumPy (almost all of them do), you *still* have the desync bug unless you write a `worker_init_fn`. Let me show the correct one.

```python
import random
import numpy as np
import torch

def seed_worker(worker_id):
    # PyTorch sets each worker's torch seed to base_seed + worker_id and
    # exposes it via initial_seed(). Derive numpy/python seeds from THAT so
    # every worker gets a distinct, reproducible stream.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)  # seeds the SHUFFLE order, separate from worker RNG

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=128,
    num_workers=4,
    shuffle=True,
    worker_init_fn=seed_worker,   # fixes per-worker numpy/python RNG
    generator=g,                  # fixes the batch SHUFFLE order
    persistent_workers=False,     # see the gotcha below
)
```

Two distinct mechanisms are at work here and you need both. The `generator=g` argument seeds the **sampler** — the order in which indices are drawn, i.e., which examples land in which batch. Without it, `shuffle=True` uses PyTorch's *global* RNG, whose state has been advanced by every prior random call in your script (init, dropout masks during a dry run, anything), so the shuffle is not pinned to your seed in a stable way. The `worker_init_fn` seeds the **augmentation** RNG inside each worker. Forget the generator and your *batch order* drifts (large step-1 gap). Forget `worker_init_fn` and your *augmentations* drift or collapse (the diversity bug). Our running example had the first problem: no `generator`, so the first batch differed, so step-1 loss differed by `1.4e-3`.

#### The persistent_workers gotcha

There's a newer wrinkle. With `persistent_workers=True` (which you often want, because it avoids re-spawning workers every epoch and can be a real speedup), `worker_init_fn` runs **once per worker for the whole training run**, not once per epoch. So your per-worker seed is fixed at the *first* epoch and the worker's RNG keeps advancing from there across all epochs. That's actually fine for reproducibility — it's still deterministic given the seed — but it means epoch 2 does *not* reset to the same augmentation stream as epoch 1, and people who expected per-epoch resets get confused when their "deterministic" run doesn't match an older one that re-spawned workers each epoch. If you need exact match to a `persistent_workers=False` baseline, keep it `False`. The reproducibility lesson: persistence changes *when* the seed is applied, so two configs that are each internally deterministic can still disagree with each other. Always reproduce against the same `persistent_workers` setting.

#### Stress-testing the worker fix: four what-ifs

A fix you haven't stress-tested is a fix that works on Tuesday and breaks on Wednesday. Let's pressure the worker-seeding fix from four angles, because each exposes a different failure mode you'll eventually hit.

**What if `num_workers` changes?** If you debug with `num_workers=2` and a teammate runs `num_workers=8`, do you get the same training? *No* — and this is by design but easy to forget. Each worker gets `base_seed + worker_id`, so 2 workers produce a different *set* of augmentation streams than 8 workers, which means the augmentations applied to any given image differ. The batch *order* is still pinned by `generator=g` (the sampler lives in the main process), so the same examples land in the same batches; but the random crop of image 4,217 differs between the 2-worker and 8-worker runs. The reproducibility contract therefore includes `num_workers`. If you must match exactly, pin the worker count; if you only need statistical match, the worker count can vary and final accuracy will agree within noise.

**What if the dataset's `__getitem__` uses the *global* RNG instead of a worker-local one?** Some augmentation libraries call `random.random()` or `np.random.rand()` directly (the global stream) rather than taking a generator argument. Our `seed_worker` reseeds those global streams per worker, so it's covered — *but only if the library reads the global stream at call time*. A library that captures a generator at construction time (in the main process, before forking) will share that one generator across all workers and desync. The test: log the augmentation parameters (the crop coordinates, the flip flags) per worker for the first few batches and assert they differ across workers but repeat across runs. If they're identical across workers, your augmentation diversity collapsed — the shared-seed bug from the case studies.

**What if you switch from fork to spawn?** On Linux the default start method is `fork` (workers inherit the parent's memory, including its RNG state); on Windows and macOS it's `spawn` (workers start fresh and re-import your module). Under `spawn`, the workers do *not* inherit the parent's advanced RNG state, so the exact augmentation streams differ from a `fork`-based run even with the same seed. If you develop on a Mac (`spawn`) and train on a Linux box (`fork`), the runs won't bit-match. The `worker_init_fn` makes each *internally* deterministic, but the start method is part of the contract. Pin it with `torch.multiprocessing.set_start_method` if you need cross-platform match.

**What if it's not the workers at all?** Run the bisection's first step — `num_workers=0`. If the run *still* doesn't repeat with zero workers, the workers were never the problem; it's a kernel (atomic-add) or a data-order issue in your dataset's `__getitem__` (an unsorted glob inside the dataset). This is the value of make-it-fail-small: one keystroke cleanly rules the entire worker layer in or out, so you never waste an afternoon tuning `worker_init_fn` for a bug that lives in cuDNN.

## 4. The diagnostic: bisect the sources by toggling them

You now know the five sources. The fastest way to find *which one* is breaking a given run is to disable them one at a time and watch when bit-equality returns. This is make-it-fail-small applied to nondeterminism: shrink the surface until the run repeats, and the last thing you turned off is the culprit. Figure 7 is the decision graph.

![Graph showing a bisection procedure for a nonrepeatable run: starting from loss differing between run A and B, first set num_workers to zero, and if the runs become identical the workers were the cause, otherwise enable deterministic algorithms, then disable TF32, and if it still differs the cause is library version drift](/imgs/blogs/reproducibility-and-determinism-in-training-7.png)

The order matters — go from cheapest-to-test and most-likely-first:

1. **Set `num_workers=0`.** This collapses all worker RNG into the main process and serializes data loading. If the run now repeats bit-for-bit, your bug was worker RNG (or worker-order nondeterminism in collation) — go fix `worker_init_fn` and `generator`. This catches the majority of cases and it's one keystroke.
2. **Still differs? Turn on deterministic algorithms.** `torch.use_deterministic_algorithms(True)`. If the run now repeats, the culprit was a nondeterministic CUDA kernel (atomic-add reduction in a scatter, pooling, upsampling, or embedding-bag backward). You've localized to numerics.
3. **Still differs? Disable TF32.** Set `torch.backends.cuda.matmul.allow_tf32 = False` and `torch.backends.cudnn.allow_tf32 = False`. TF32 itself is deterministic on a given GPU, so this rarely changes *run-to-run* equality on one box — but it matters for cross-machine and for matching an fp32 baseline. If turning it off changes your numbers, you were comparing across precisions.
4. **Still differs on the same box?** Now suspect library or environment drift — a different cuDNN/CUDA/PyTorch version between the two runs, a different `OMP_NUM_THREADS`, or a different GPU. Pin the environment and re-test.

Here is the diagnostic harness that runs steps 1–4 mechanically. It runs the first N steps twice in the same process and reports the first step where the losses disagree and by how much.

```python
import torch

def run_n_steps(model, loader, opt, n=5):
    losses = []
    model.train()
    for i, (x, y) in enumerate(loader):
        if i >= n:
            break
        opt.zero_grad(set_to_none=True)
        out = model(x.cuda())
        loss = torch.nn.functional.cross_entropy(out, y.cuda())
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses

def diff_two_runs(make_everything, n=5):
    # make_everything() must rebuild model+loader+opt from the SAME seed.
    a = run_n_steps(*make_everything(), n=n)
    b = run_n_steps(*make_everything(), n=n)
    for i, (la, lb) in enumerate(zip(a, b)):
        d = abs(la - lb)
        flag = "OK " if d == 0.0 else "DIFF"
        print(f"step {i}: A={la:.10f}  B={lb:.10f}  |diff|={d:.2e}  {flag}")
        if d > 0:
            print(f"  -> first divergence at step {i}, gap {d:.2e}")
            return i, d
    print("bit-identical across all steps")
    return None, 0.0
```

Read the output the way section 1 taught you. A first-divergence at step 0 with a *large* gap means the **first batch differed** — data order, go fix the loader. A divergence that's `0.0` at step 0 but appears at step 1 with a *tiny* gap (`~1e-7`) means **same data, nondeterministic kernel** — go to step 2 of the bisection. The harness turns a vague "it's not reproducible" into "diverges at step 1, gap 3e-7, therefore numerics," which is an actionable bug report.

A subtlety that trips people up: to make this harness *itself* trustworthy, `make_everything()` must rebuild **everything** from the same seed each call — model, loader, optimizer — with a `set_full_determinism(seed)` call at the top of each rebuild. If you reuse the same model object across the two runs, the second run starts from the first run's *trained* weights and the comparison is meaningless. And if you don't reseed between the two `make_everything()` calls, the global RNG has been advanced by the first run, so the second run's init draws different numbers and you'll see a "divergence" that's just the harness's own RNG drift, not a real bug. The discipline is: reseed, rebuild, run; reseed, rebuild, run; diff. A flaky harness diagnoses flaky bugs.

There's also a faster, non-training variant of the diagnostic for when you suspect a *specific op*. You don't need a full training loop to test whether a single layer is deterministic — feed it the same input twice and compare:

```python
import torch

def is_op_deterministic(fn, *inputs, trials=3):
    """Run fn on identical inputs several times; check outputs are bit-equal."""
    ref = fn(*inputs)
    for _ in range(trials - 1):
        out = fn(*inputs)
        if not torch.equal(out, ref):
            return False, (out - ref).abs().max().item()
    return True, 0.0

x = torch.randn(8, 3, 224, 224, device="cuda", requires_grad=True)
# Test the backward of an upsample, a classic atomic-add culprit.
def upsample_backward(inp):
    y = torch.nn.functional.interpolate(inp, scale_factor=2, mode="bilinear")
    g = torch.autograd.grad(y.sum(), inp, retain_graph=False)[0]
    return g

ok, max_diff = is_op_deterministic(upsample_backward, x)
print(f"deterministic={ok}  max_diff={max_diff:.2e}")
# typically deterministic=False, max_diff ~ 1e-7 — the atomic-add scatter
```

This isolates the nondeterminism to a single op in seconds, without waiting for a training run to diverge. It's the make-it-fail-small principle applied at the op level: shrink the suspect surface to one function, prove or clear it, move on. When `is_op_deterministic` returns `False` for an op, that op is on your list to either swap or run under deterministic mode.

## 5. Nondeterministic CUDA and cuDNN kernels

Let's go deep on the kernel source, because it's the one people understand least and it has the sharpest cost trade-off. There are two distinct levers here and they are *not* the same thing, despite both living under `torch.backends.cudnn`.

**`torch.backends.cudnn.benchmark`** controls *algorithm selection*. When `True`, the first time cuDNN sees a given input shape it runs a quick microbenchmark of several convolution algorithms and picks the fastest for your hardware. This is a real speedup for fixed-shape workloads. But the microbenchmark's choice can vary run-to-run (timing is noisy), and different algorithms produce slightly different floating-point results. So `benchmark=True` is a *source of run-to-run nondeterminism* even before you get to atomic adds. For a reproducible run, set it `False`.

**`torch.backends.cudnn.deterministic`** asks cuDNN to use deterministic convolution algorithms specifically. **`torch.use_deterministic_algorithms(True)`** is the bigger hammer: it tells *all* of PyTorch to use deterministic implementations where they exist and — crucially — to **raise an error** for any op that has *no* deterministic implementation, instead of silently using the nondeterministic one. That error-on-no-deterministic-kernel behavior is a feature, not a nuisance: it tells you exactly which op in your model is a nondeterminism source you didn't know about.

The ops that bite you are the ones built on **scatter / atomic-add** in their backward pass:

| Op (backward) | Why nondeterministic | Deterministic option? |
| --- | --- | --- |
| `scatter_add` / `index_add` | atomic-add accumulation, thread-order dependent | yes, slower serialized kernel |
| `nn.EmbeddingBag` backward | gradients scattered into shared rows via atomics | yes (or use `Embedding` + reduce) |
| `F.interpolate` (bilinear upsample) backward | atomic scatter into source pixels | partial; nearest is deterministic |
| Some pooling backward (adaptive) | overlapping windows accumulate via atomics | yes under deterministic mode |
| `index_select` backward (repeated idx) | duplicate indices race | yes, serialized |

When you flip `torch.use_deterministic_algorithms(True)` and your model uses `EmbeddingBag` or bilinear upsampling, you may hit a `RuntimeError` like *"upsample_bilinear2d_backward_out_cuda does not have a deterministic implementation."* That's the system telling you the honest truth: there is no fast deterministic kernel for that op on your version. Your choices are (a) accept it errors and switch to a deterministic-friendly op (`Embedding` instead of `EmbeddingBag`, nearest-neighbor upsampling where acceptable), (b) set the `warn_only=True` flag to downgrade the error to a warning and accept that op stays nondeterministic, or (c) decide you don't need full determinism here (section 9).

There's one more required incantation for CUDA matmul reproducibility. cuBLAS can use nondeterministic workspaces across streams; PyTorch requires you to pin the workspace via an environment variable **before CUDA initializes**:

```bash
# Must be set before the process starts / before any CUDA context is created.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python train.py --seed 42
```

If you call `torch.use_deterministic_algorithms(True)` and *haven't* set `CUBLAS_WORKSPACE_CONFIG`, PyTorch raises an error pointing you at exactly this. Set it in your launch script, not inside Python (by the time Python runs, the CUDA context may already exist and the env var is read too late). This is the single most common "I did everything and it still errors" gotcha.

#### Worked example: the EmbeddingBag that wouldn't repeat

A recommender model with a categorical feature encoded via `nn.EmbeddingBag(num_embeddings=1_000_000, embedding_dim=64, mode="sum")` trained twice gave step-1 losses of `0.6931472` and `0.6931474` — a `2e-7` gap, the numerics signature. Bisection: `num_workers=0` didn't fix it (not data order), so it was kernels. Turning on `torch.use_deterministic_algorithms(True)` *errored*: `embedding_bag_backward` has no deterministic CUDA kernel. The fix was to replace `EmbeddingBag(mode="sum")` with a plain `Embedding` followed by an explicit `.sum(dim=1)`, whose backward *does* have a deterministic path. After the swap, both runs gave `0.6931472` to all 10 logged digits — bit-identical. Cost: the explicit-sum path was about 12% slower per step on this model. For the debugging phase that was a trivial price; we kept `EmbeddingBag` for the final production run where throughput mattered and determinism didn't.

## 6. TF32 and reduced precision: a different kind of irreproducibility

TF32 deserves its own section because it confuses people: it is **deterministic** on a given GPU, yet it is a major source of "my numbers don't match." The resolution is that TF32 breaks *cross-precision and cross-machine* reproducibility, not *run-to-run-on-one-box* reproducibility.

TF32 ("TensorFloat-32") is the default matmul mode on Ampere and later NVIDIA GPUs. It takes float32 inputs but **truncates the mantissa to 10 bits** (plus the float32 8-bit exponent) for the multiply, accumulating in float32. So a TF32 matmul has the *range* of float32 but only about the *precision* of a slightly-better-than-fp16 number — roughly 3 decimal digits in the products. It's a huge speedup (the tensor cores chew through it) and for most training the precision loss is invisible to final accuracy. But:

- A TF32 matmul gives a **different result** than a true fp32 matmul — by about `1e-3` relative, not `1e-7`. So if your "baseline" run was fp32 and your new run is TF32 (or you upgraded PyTorch and the default flipped), your step-1 losses won't match, and it'll look like a bug.
- TF32 is **deterministic run-to-run on the same GPU** — same truncation, same tensor-core path. So it does *not* explain a run that differs from itself on one box (that's workers or atomic adds). It explains a run that differs from a *different precision* or a *different GPU*.

So the rule is: to match an fp32 reference exactly, disable TF32. To merely be repeatable on your own box, you can leave TF32 on (it's deterministic) and it'll save you a lot of time.

```python
# Disable TF32 ONLY when you must match an fp32 baseline or compare across GPUs.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
# Leave them True (the default on Ampere+) for fast, still-repeatable-on-one-box runs.
```

The bigger picture is **non-associative floating point under reduced precision and across devices**, and it ties back to section 2. Three things reorder or requantize your sums:

1. **Reduced precision (fp16/bf16/TF32)** throws away mantissa bits, so the *same* sum in a different precision gives a different value — large relative error, `1e-3`-ish. This is precision, not order.
2. **Reduction order** (atomic adds, different tile sizes, different thread counts) reorders the *same-precision* sum — small relative error, `1e-7`-ish. This is the non-associativity from section 2.
3. **Multi-GPU all-reduce** sums each rank's gradient. The order in which the 8 ranks' contributions are combined is a reduction order, so a ring all-reduce and a tree all-reduce can give different bits, and even the *same* algorithm can vary if rank arrival order isn't fixed. This is why a DDP run can be nonrepeatable even when each rank is internally deterministic — covered head-on in [debugging DDP and multi-GPU](/blog/machine-learning/debugging-training/debugging-ddp-and-multi-gpu).

bf16 deserves a note since it's the modern default for LLM training. bf16 has the *same 8-bit exponent as fp32* (so the same range, no underflow drama) but only a 7-bit mantissa — even coarser than fp16's 10-bit mantissa. So bf16 sums have *larger* per-operation rounding error than fp16, which means bf16 reductions are *more* sensitive to order than fp32 ones. Practically this means a bf16 run accumulates order-dependent divergence faster; the figure-6 amplification curve is steeper. The fix is the same — pin the reduction order with deterministic algorithms — but the *magnitude* of the noise you're fighting is set by the mantissa width. For the full treatment of why fp16 underflows where bf16 doesn't, and how to read the gradient histogram to choose, see [mixed-precision debugging: fp16 vs bf16](/blog/machine-learning/debugging-training/mixed-precision-debugging-fp16-vs-bf16).

## 7. Data order: the source hiding in plain sight

Beyond worker RNG, there's a whole family of data-order nondeterminism that doesn't involve the DataLoader's seeds at all — it's in how you *enumerate your data in the first place*. These are the bugs that make step-1 loss differ even with `num_workers=0`.

The most common is **filesystem glob order**. This:

```python
import glob
files = glob.glob("data/train/*.jpg")   # order is NOT guaranteed
```

`glob.glob` returns files in **arbitrary, filesystem-dependent order** — it does *not* sort. On some filesystems it's roughly creation order, on others it's inode order, and it can change if files are touched, the directory is re-copied, or you move to a different machine. So your "dataset" is a different ordering of the same files on different runs or machines, and even with a fixed shuffle seed, the seed permutes a *different starting order*, giving different batches. The fix is one function call:

```python
files = sorted(glob.glob("data/train/*.jpg"))   # deterministic order
```

Same trap with `os.listdir` (unordered), `set` iteration (insertion-order in CPython 3.7+ but don't rely on it for reproducibility across versions/hash-seeds), and dictionary iteration over data built from an unordered source. Anything that materializes your example list must be explicitly sorted before the shuffle seed can mean anything.

The second is **non-deterministic dataset iteration** in streaming or `IterableDataset` setups. If your dataset reads from a sharded store and the shard-to-worker assignment isn't pinned, or you use a streaming `datasets` loader without setting its seed, the *order* of examples is a function of network timing and worker scheduling. Hugging Face `datasets` exposes `.shuffle(seed=...)` and streaming `IterableDataset` has a `set_epoch` for reproducible epoch shuffles; use them. The principle is identical to the glob case: pin every step from "list of raw examples" through "order they're consumed."

The third, subtle one is **shuffling without a fixed generator** — which we covered for the DataLoader, but it also appears in manual shuffles:

```python
import random
random.shuffle(indices)            # uses global RNG state — drifts
# vs
rng = random.Random(42)
rng.shuffle(indices)               # pinned, reproducible
```

The global-RNG version is reproducible *only if* every prior random call in the process happened in the same order — which is fragile, because adding a debug print that calls `random` anywhere upstream silently shifts the state. Always shuffle data with a *dedicated, explicitly-seeded* generator, never the global one. This is the same lesson as `generator=g` on the DataLoader: give the thing that shuffles its own pinned RNG, isolated from the rest of the program.

This whole family connects directly to [the input pipeline is lying to you](/blog/machine-learning/debugging-training/the-input-pipeline-is-lying-to-you), which covers the broader set of dataloader bugs (wrong augmentation placement, normalization mismatch, the silently-dropped last batch). Reproducibility is one slice of "trust your pipeline": a pipeline you can't reproduce is one you can't trust, and the print-the-batch discipline from that post is exactly how you catch a data-order bug — log the *indices* of the first batch and assert they're identical across runs.

## 8. Run-to-run vs cross-machine: two different guarantees

We've now hit this distinction several times; let's make it precise, because conflating the two is the cause of most wasted determinism effort. There are really *three* levels of reproducibility, with three very different costs, and figure 5 lays them out as a decision tree.

![Tree diagram distinguishing three reproducibility guarantees: starting from the repeatability goal, branching on whether you are on the same box with the same versions which gives bit-identical results via seeds and deterministic algorithms, versus a different GPU or library which can only give a statistical match and where bit-equality is not guaranteed](/imgs/blogs/reproducibility-and-determinism-in-training-5.png)

**Level 1 — run-to-run on the same machine.** Same GPU, same drivers, same library versions, same code. This is *fully achievable*: seed all RNGs, set deterministic algorithms, set `CUBLAS_WORKSPACE_CONFIG`, fix `cudnn.benchmark=False`. You get bit-identical losses, every run. This is what you need for debugging and regression tests, and it's what the `set_full_determinism` utility in the next section buys you.

**Level 2 — statistical reproducibility across machines.** Different GPU arch (an A100 vs an H100), different cuDNN/CUDA/PyTorch version, different number of workers or threads. Here bit-equality is **mostly unreachable** — different hardware uses different reduction trees, tile sizes, and algorithms, so the same math comes out with different last-bit results (section 2/6). What you *can* get is that the runs converge to *statistically equivalent* models: same final accuracy within noise, same training curve shape. That's usually what "reproducible research" actually means, and it's the honest target for a paper or a shared benchmark. Don't promise bit-equality across hardware; you can't deliver it and chasing it wastes weeks.

**Level 3 — bit-identical across machines.** Generally **not guaranteed and not worth pursuing** unless you control the entire stack (identical GPU SKU, pinned driver, pinned library versions, pinned thread counts). The number of variables — float-rounding in different kernels, cuBLAS algorithm choices per arch, even compiler differences — makes this a fool's errand for most teams.

The amplification dynamics are worth seeing, because they explain why even a *tiny* level-2 difference matters. Figure 6 shows a `1e-7` step-1 difference (the kind reduced precision or a different reduction order produces) growing past `1e-2` by step 500. Training is a chaotic dynamical system: small perturbations to the gradient get amplified by the optimizer, and once two runs take a slightly different step they see slightly different gradients next step, which compounds. So you cannot say "the difference is only `1e-7`, who cares" — by the end of training the two runs can land in genuinely different minima with different final accuracy. That's *why* a `1e-7` kernel nondeterminism becomes a half-point accuracy swing.

![Timeline showing how a tiny numerical difference amplifies over training: identical at step zero, a difference of 1e-7 at step one, growing to 3e-5 at step fifty, 8e-3 at step two hundred, 4e-2 at step five hundred, and ending at different minima](/imgs/blogs/reproducibility-and-determinism-in-training-6.png)

The practical upshot: decide which level you need *before* you start. Debugging a specific bug? You need level 1, on your box, and the cost is fine. Publishing a result others should be able to match? Target level 2 — report seeds, report hardware, report library versions, and claim statistical not bitwise reproducibility. Shipping to production? You probably need *none* of it (section 9). Naming the level you need stops you from paying for a guarantee you'll never use.

### Multi-GPU all-reduce: the reduction-order problem at the cluster scale

Section 6 mentioned all-reduce in passing, but it deserves a closer look because it's where reproducibility gets genuinely hard and where a lot of "my distributed run won't repeat" reports come from. In data-parallel training (DDP), each of your $N$ GPUs computes a gradient on its own shard of the batch, and then an **all-reduce** collective sums those $N$ gradients element-wise and broadcasts the sum back to every rank. That sum is — you guessed it — a reduction over $N$ terms, and the order in which the ranks' contributions are combined is exactly the kind of order that floating-point non-associativity cares about.

There are two distinct algorithms NCCL might pick. A **ring all-reduce** passes partial sums around the ranks in a fixed ring, accumulating as it goes; a **tree all-reduce** combines in a binary-tree pattern. These two algorithms add the same $N$ gradients in *different orders*, so they give different last bits. Worse, NCCL may *choose* the algorithm at runtime based on message size and topology, so two launches of the same job on the same cluster can pick different collectives and therefore diverge. This is why a DDP run can be nonrepeatable even when each individual rank is perfectly deterministic internally: the nondeterminism lives in the *collective*, not in any one GPU.

The practical handle: NCCL's algorithm choice can be pinned via environment variables (e.g. forcing a specific all-reduce algorithm), and PyTorch's deterministic mode plus a fixed collective gives you run-to-run repeatability on a fixed cluster. But the same warning as section 8 applies harder here: bit-equality *across different cluster sizes* (4 GPUs vs 8 GPUs) is essentially unreachable, because changing $N$ changes the reduction tree itself. An 8-GPU run and a 4-GPU run that are each internally deterministic will still give different bits, because they're summing a different number of terms in a different shape. So when someone says "I changed from 4 to 8 GPUs and the loss curve moved," that is *expected* under exact arithmetic's failure to be associative — not a bug, as long as the final accuracy is within noise. The full treatment of gradient-sync correctness, per-rank seeding, and data sharding lives in [debugging DDP and multi-GPU](/blog/machine-learning/debugging-training/debugging-ddp-and-multi-gpu); here the takeaway is narrow: **all-reduce is a reduction, reductions are order-dependent, so the cluster shape is part of your reproducibility contract.**

There's a per-rank seeding subtlety too. If every rank seeds *identically*, then any per-rank randomness — dropout masks, random augmentation if each rank augments its own shard — is *identical* across ranks, which is usually wrong: you want each rank to see different augmentations of its different data shard, so you want `seed + rank` per rank, not the same seed everywhere. But the *data sharding* (which examples go to which rank) must be deterministic and disjoint, so the sampler seed must be *shared* and combined with the rank to partition without overlap. The pattern is: shared seed for the sampler's global shuffle, then `DistributedSampler` deterministically partitions it by rank; `seed + rank` for the per-rank augmentation RNG. Getting this backwards — same augmentation seed everywhere, or overlapping shards — is a classic distributed-reproducibility bug that *also* quietly hurts your model.

### RNG state, checkpointing, and the resume that breaks reproducibility

One more source that surprises people: **checkpoint and resume**. You train for 1,000 steps, save a checkpoint, kill the job, and resume. Is the resumed run identical to one that trained 2,000 steps straight through? Only if you saved and restored the **RNG state** — not just the model and optimizer.

Here's why. The dropout masks, the data shuffle for the *next* epoch, any random augmentation — all of it draws from the RNG, and after 1,000 steps the RNG is in some advanced state. A fresh process that loads your model checkpoint starts its RNG from the seed again, at the *beginning* of the stream, so step 1,001 of the resumed run draws different "random" numbers than step 1,001 of the straight-through run. The model weights match at the resume point, but the runs diverge immediately after, because the randomness is out of sync. The visible signature is a small but real **loss jump or curve shift right at the resume step** — not a crash, just a discontinuity that shouldn't be there.

The fix is to checkpoint the RNG state of every stream alongside the model:

```python
import torch, random, numpy as np

def save_checkpoint(path, model, optimizer, step):
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        # The RNG state of every stream — this is what people forget.
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }, path)

def load_checkpoint(path, model, optimizer):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    torch.set_rng_state(ckpt["torch_rng"])
    torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
    np.random.set_state(ckpt["numpy_rng"])
    random.setstate(ckpt["python_rng"])
    return ckpt["step"]
```

That restores all four RNG streams to exactly where they were, so step 1,001 of the resumed run draws the *same* numbers as a straight-through run. The verification is the same two-run diff from section 11, but across a resume boundary: train straight through to step 2,000, fingerprint; train to 1,000, checkpoint, resume to 2,000, fingerprint; assert the two fingerprints are bit-identical. If they aren't, you forgot an RNG stream (the most common omission is `numpy_rng`, because the augmentation uses NumPy but people only save torch's state). The dedicated post [debugging checkpoint and resume](/blog/machine-learning/debugging-training/debugging-checkpoint-and-resume) goes deeper on optimizer-state, LR-scheduler, EMA, and scaler resumption; the reproducibility-specific point is just that **RNG state is part of the checkpoint, and a resume that doesn't restore it is a reproducibility bug with a loss-jump signature.**

#### Worked example: the loss jump at the resume boundary

A team finetuning a 7B language model checkpointed every 500 steps and resumed after a node failure. The loss before the failure was descending smoothly through `1.84`; the step right after resume jumped to `1.91` and then re-descended. It looked like the resume "corrupted" something. It hadn't — the model and optimizer states were fine. The jump came from the data order: their `DistributedSampler` reshuffled from the base seed on resume, so the resumed run replayed examples the pre-failure run had already seen *and* skipped others, and the dropout masks were a fresh stream. Saving `sampler.set_epoch` state plus the four RNG streams (and restoring the sampler's position within the epoch) removed the jump entirely — the resumed curve became a seamless continuation, bit-identical to a straight-through control for the first 50 post-resume steps. The cost of the fix was a slightly larger checkpoint file (the RNG states are a few KB) and zero throughput. The lesson: a loss jump *exactly at* a resume boundary is almost always un-restored randomness, not a corrupted optimizer.

## 9. The cost, and when NOT to chase determinism

This is the section most reproducibility guides skip, and it's the one that saves you the most time and money. Full determinism is **not free**, and in production it's usually the wrong trade.

The costs, concretely:

- **Throughput.** `torch.use_deterministic_algorithms(True)` swaps fast atomic-add kernels for serialized deterministic ones, and `cudnn.benchmark=False` forgoes the autotuned fastest convolution. On a CNN with upsampling/scatter ops, expect **10–40% slower** per step. On a transformer that's mostly matmuls and doesn't hit the nondeterministic ops, the hit can be near zero — measure, don't assume.
- **Memory.** Some deterministic kernels need extra workspace (the pinned `CUBLAS_WORKSPACE_CONFIG=:4096:8` reserves cuBLAS workspace), and the deterministic-friendly op swaps (e.g. `Embedding` + explicit sum instead of `EmbeddingBag`) can materialize larger intermediate tensors. Usually small, occasionally enough to push you to OOM on a tight batch size.
- **Restricted op set.** You may have to *not use* an op (or accept a warning) because no deterministic kernel exists. That can mean a slower or less elegant model.

The size of the throughput hit depends entirely on *which* ops dominate your model, and this is worth measuring before you decide. A convolutional segmentation network leans heavily on the exact ops that have nondeterministic-but-fast kernels — bilinear upsampling backward, scatter in the loss, adaptive pooling — so it pays the full 30–40%. A plain ResNet classifier touches fewer of them and pays maybe 10–15%. A decoder-only transformer is almost all matmuls and layernorms, whose deterministic paths are nearly as fast as the nondeterministic ones, so it can pay close to *zero* for `use_deterministic_algorithms(True)` — its real reproducibility cost is the all-reduce ordering and TF32, not kernel determinism. So "determinism costs 30%" is a model-dependent claim, not a universal one. Profile a 50-step run with and without the flag, read the steps-per-second delta, and *then* decide. The cost you're avoiding might be 2% (keep determinism on, it's free) or 40% (turn it off in production). Don't pay a tax you never measured.

Figure 8 lays out the decision: who needs it and who shouldn't pay.

![Matrix mapping four contexts to whether full determinism is needed and at what cost: debugging a bug requires it because fixes must be separable from luck, regression tests require it for exact CI asserts, scientific comparison requires run-to-run equality so A/B deltas are real, but production training should skip it because throughput beats bit-equality](/imgs/blogs/reproducibility-and-determinism-in-training-8.png)

**You MUST have determinism when:**

- **You're debugging.** This is the whole point of the post. If you can't rerun and get the same loss, you cannot tell a fix from luck. Turn on full determinism for the debugging session, find the bug, then turn it off.
- **You're writing a regression test.** A CI test that trains 50 steps and asserts the loss equals a stored value to the bit will catch any future change that alters numerics — a kernel update, an accidental TF32 flip, a refactor that reorders a reduction. That test is *only possible* under determinism. The run is tiny so the throughput cost is nothing.
- **You're making a scientific comparison.** A/B-ing two configs (new loss vs old, with-warmup vs without) requires that the *only* thing that differs is the variable you changed. Run-to-run determinism (same seed, same data order) makes the delta you measure a real effect, not noise. Otherwise you're back to optimizing against a coin flip.

**You should NOT chase full determinism when:**

- **You're training a production model for throughput.** A 30% slowdown on a multi-day, multi-GPU run is real money — at, say, \$2.00 per GPU-hour across 8 GPUs for 5 days, a 30% slowdown is roughly \$576 of extra compute *per run*, for a bit-equality you will never check. Keep `cudnn.benchmark=True`, keep TF32, keep fast kernels. You still set the seeds (cheap, and you want a reproducible *enough* data order), but you skip the expensive kernel-level determinism.
- **You only need statistical reproducibility.** If "same final accuracy within ±0.3%" is good enough — which it usually is for a model you're going to deploy — you don't need bit-equality, and the seeds alone get you most of the way.

The discipline: determinism is a **debugging mode**, not a default. Flip it on to investigate, flip it off to ship. The `set_full_determinism` utility below should be guarded by a flag so it's trivial to toggle.

#### Worked example: the \$576 you didn't need to spend

A team set `torch.use_deterministic_algorithms(True)` globally because a tutorial told them to, and shipped it to their production training pipeline. Their CNN (with bilinear upsampling in the decoder) ran 31% slower than the nondeterministic baseline — a 5-day run became 6.5 days. Across 8 A100s at roughly \$2.00/GPU-hour, the extra 36 hours cost about \$576 per run, and they ran it weekly. Nobody was diffing the runs bit-for-bit; they only ever compared final accuracy, which varied ±0.4% anyway. Removing the global determinism flag (keeping only the seeds) recovered the 31% with **zero** impact on the metric they actually cared about. The fix was deleting one line. The lesson: determinism is a tool you reach for in the debugger, not a setting you leave on in production.

## 10. The complete fix: `set_full_determinism(seed)`

Here is the single utility that pins all five sources. Call it once at the top of your script, guard it behind a flag, and pair it with the DataLoader configuration from section 3.

```python
import os
import random
import numpy as np
import torch

def set_full_determinism(seed: int = 42, warn_only: bool = False):
    """Pin every nondeterminism source for bit-identical run-to-run results
    on a single machine. Call BEFORE creating the model/optimizer/dataloader.
    Set CUBLAS_WORKSPACE_CONFIG in the LAUNCH ENV, not here, ideally."""

    # cuBLAS deterministic workspace — must be set before CUDA init. Setting it
    # here works only if no CUDA context exists yet; prefer the launch script.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # Make Python hash randomization deterministic (affects set/dict order).
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    # 1. Seed the four RNG streams.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   # all visible GPUs

    # 2. Force deterministic CUDA/cuDNN kernels.
    torch.backends.cudnn.benchmark = False        # no autotune drift
    torch.backends.cudnn.deterministic = True     # deterministic convolutions
    torch.use_deterministic_algorithms(True, warn_only=warn_only)

    # 3. (Optional) match an fp32 baseline / compare across GPUs.
    #    Leave TF32 ON for speed if you only need same-box repeatability.
    # torch.backends.cuda.matmul.allow_tf32 = False
    # torch.backends.cudnn.allow_tf32 = False

def make_loader(dataset, batch_size, seed=42, num_workers=4):
    def seed_worker(_):
        ws = torch.initial_seed() % 2**32
        np.random.seed(ws)
        random.seed(ws)
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, worker_init_fn=seed_worker, generator=g,
        persistent_workers=False,
    )
```

And the launch script that sets the env var at the right time:

```bash
#!/usr/bin/env bash
# determinism must be configured before the CUDA context is created
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
python train.py --seed 42 --deterministic
```

The checklist, as a flat list you can paste into a PR description:

- [ ] `random`, `numpy`, `torch`, `torch.cuda` all seeded from one base seed.
- [ ] DataLoader has `worker_init_fn` (per-worker numpy/python seed) **and** `generator=g` (shuffle order).
- [ ] `cudnn.benchmark = False`, `cudnn.deterministic = True`.
- [ ] `torch.use_deterministic_algorithms(True)` — and you handled any op that errors.
- [ ] `CUBLAS_WORKSPACE_CONFIG=:4096:8` set in the *launch env*, before CUDA init.
- [ ] All file lists `sorted()`; all shuffles use a dedicated seeded generator, never the global RNG.
- [ ] TF32 disabled *only if* matching an fp32 baseline or comparing across GPUs.
- [ ] `persistent_workers` setting matches the baseline you're comparing against.
- [ ] The whole thing is behind a `--deterministic` flag so production can skip it.

## 11. Verifying it: run twice, diff to the last bit

A fix you don't verify is a hope. The verification for determinism is beautifully crisp because the success criterion is exact: **two runs must produce bit-identical numbers.** No "close enough" — equal.

The strongest check compares *tensors*, not just the scalar loss, because two runs can have equal loss by luck while their gradients differ. Run a few steps and assert the model state and the loss match exactly:

```python
import torch

def fingerprint(model, loss):
    # Concatenate all parameters into one tensor and hash it with the loss.
    flat = torch.cat([p.detach().flatten() for p in model.parameters()])
    return (loss, flat.sum().item(), flat.std().item(), flat.numel())

def assert_bit_identical(make_everything, n=5):
    set_full_determinism(42)
    a_model, a_loss = train_n_and_return(*make_everything(), n=n)
    set_full_determinism(42)
    b_model, b_loss = train_n_and_return(*make_everything(), n=n)

    # Loss to the last representable digit.
    assert a_loss == b_loss, f"loss differs: {a_loss!r} vs {b_loss!r}"
    # Every parameter tensor bit-for-bit equal.
    for (na, pa), (nb, pb) in zip(a_model.named_parameters(),
                                  b_model.named_parameters()):
        assert torch.equal(pa, pb), f"param {na} differs after {n} steps"
    print(f"bit-identical after {n} steps: loss={a_loss!r}")
```

Use `torch.equal` (exact, element-wise equality), **not** `torch.allclose` (which tolerates small differences and would *hide* a `1e-7` nondeterminism — the exact thing you're hunting). The distinction is the whole point: `allclose` answers "are these approximately the same model?" but determinism asks "are these the *same bits*?", and only `torch.equal` answers that.

This `assert_bit_identical` is also exactly what you put in CI as a regression test (section 9). Train 50 steps, fingerprint, and store the expected loss; the test fails the day someone bumps cuDNN and silently changes a reduction order, or flips TF32, or breaks the worker seeding. That test is the durable payoff of all this work: not just "I can reproduce my run today" but "my CI will tell me the moment my run *stops* being reproducible."

#### Before and after: the running example, fixed

Back to our ResNet-18. Figure 3 shows the result. **Before:** no `generator` on the loader, default global-RNG shuffle, so the first batch differed across runs; step-1 loss was `2.3041` vs `2.3055`, a `1.4e-3` gap, and final accuracy swung `71.4%` vs `70.9%`. **After** adding `worker_init_fn` + `generator=g` + `set_full_determinism(42)`: step-1 loss was `2.30418` in both runs, `torch.equal` on every parameter passed after 5 steps, and final accuracy was identical to all reported digits. The bug was *entirely* data order — the workers were reseeded from time and the shuffle wasn't pinned — and the fix was four lines of loader configuration plus the determinism flag. The 18% throughput cost from deterministic algorithms we accepted *during debugging only*, then dropped for the production rerun.

![Before-and-after comparison of two supposedly identical runs: before the fix, run A loss is 2.3041 and run B is 2.3055 with a 1.4e-3 difference at step one from workers reseeding from time, and after seeding the workers and enabling deterministic algorithms both runs give 2.30418 with zero difference and bit-identical results](/imgs/blogs/reproducibility-and-determinism-in-training-3.png)

## 12. Source, fix, and cost at a glance

Figure 2 collects the whole post into one decision table — every nondeterminism source, its fix, and what pinning it costs. This is the thing to keep on screen while you debug. The columns are the three questions you actually ask in the moment: *what is the source* (so you can name the layer of figure 1), *how do I pin it* (the one-line fix), and *what does pinning it cost* (so you know whether to keep it on after the bug is found). Read the table top to bottom and you've run the whole bisection on paper before touching the code: the free fixes (worker RNG, shuffle, glob order) you apply always, the expensive ones (deterministic kernels, fp32 matmul) you apply only while debugging and remove for the production rerun.

![Matrix table of nondeterminism sources with their fixes and costs: worker RNG reseeding from time fixed by worker_init_fn and generator at near-zero cost, cuDNN atomic-add kernels fixed by deterministic algorithms at 10 to 40 percent slower, TF32 mantissa truncation fixed by disabling allow_tf32 at 2 to 8 times slower matmul, and unseeded glob or shuffle data order fixed by sorted and a fixed generator at near-zero cost](/imgs/blogs/reproducibility-and-determinism-in-training-2.png)

| Source | Symptom | Confirming test | Fix | Cost |
| --- | --- | --- | --- | --- |
| Worker RNG reseed | large step-1 gap; augmentation diversity collapses | `num_workers=0` makes it repeat | `worker_init_fn` + `generator=g` | ~0 |
| Unseeded shuffle | different first batch each run | log batch indices; differ | `generator=g`, dedicated seeded RNG | ~0 |
| Glob / listdir order | differs across runs or machines | print file[:5]; order changes | `sorted(glob.glob(...))` | ~0 |
| Atomic-add kernels | tiny (`~1e-7`) step-1 gap, same box | `use_deterministic_algorithms(True)` repeats it | deterministic algos + op swaps | 10–40% slower |
| `cudnn.benchmark` | algorithm choice varies run-to-run | `benchmark=False` removes drift | `cudnn.benchmark = False` | lose autotune speed |
| TF32 vs fp32 | `~1e-3` gap vs an fp32 baseline | disable `allow_tf32`; matches | `allow_tf32 = False` (only if needed) | fp32 matmul 2–8× slower |
| cuBLAS workspace | det-algos errors at startup | the error names the env var | `CUBLAS_WORKSPACE_CONFIG=:4096:8` | tiny memory |
| Multi-GPU all-reduce | DDP run differs run-to-run | single-GPU repeats; multi doesn't | fix reduction order / det collectives | small |
| Library/arch drift | differs only across machines | pin versions; same box repeats | accept level-2 statistical match | n/a |

## Case studies and real signatures

**The shared-seed worker bug (the most famous one).** For years, a widely-copied PyTorch `worker_init_fn` snippet seeded every worker's NumPy RNG with the *same* fixed seed. The consequence wasn't a crash — it was that all workers generated *identical* "random" augmentations, so a run with 4 workers had a quarter of the augmentation diversity it claimed. A well-known 2020 investigation found this pattern in a large fraction of public PyTorch repositories and tutorial code. The signature: training looks fine, but the model generalizes worse than an equivalent single-worker run, because it effectively saw less data variety. The fix is the `torch.initial_seed()`-derived per-worker seed from section 3. This is the canonical example of a reproducibility bug that *also degrades your model* — nondeterminism and a real accuracy bug in one.

**Atomic-add nondeterminism in scatter ops.** The PyTorch documentation explicitly lists the ops whose CUDA backward uses atomic adds and is therefore nondeterministic: `index_add`, `scatter_add`, `bincount`, `EmbeddingBag` (with certain modes), `nn.functional.interpolate` backward (bilinear/bicubic), and several pooling backward paths. The reproducibility page is the authoritative source. The signature is always the same: a `~1e-7` step-1 difference on the *same box, same data* that grows over training (figure 6). If you see that signature, you don't need to guess which op — flip on `torch.use_deterministic_algorithms(True)` and the resulting `RuntimeError` *names the op* for you.

**TF32 default flip.** When NVIDIA's Ampere GPUs and the corresponding PyTorch versions made TF32 the default matmul mode, a wave of "my model's numbers changed after upgrading" reports followed. Nothing was broken — the matmul precision dropped from fp32 to TF32 (~3 decimal digits), giving a `~1e-3` shift that broke bit-comparison against pre-Ampere baselines while leaving final accuracy essentially unchanged. The signature is a `~1e-3` (not `~1e-7`) gap that appears only when comparing across PyTorch/hardware versions, and disappears when you set `allow_tf32 = False`. Knowing the *magnitude* — `1e-3` is precision, `1e-7` is order — lets you diagnose it from the gap alone.

**The "irreproducible paper" problem.** A recurring theme in ML reproducibility studies (notably the community's reproducibility-challenge efforts) is that even with released code and seeds, results don't reproduce on different hardware. The usual root cause is exactly level-2 vs level-3 confusion: the authors got bit-identical results on *their* box, assumed that meant universal reproducibility, but cross-machine the reduction orders and kernel choices differ, so other teams get statistically-similar-but-not-identical numbers and report a "failure to reproduce" when really the result was fine within noise. The fix is cultural: report seeds *and* hardware *and* library versions, and claim statistical reproducibility, not bit-equality, across machines.

## When this is (and isn't) your bug

A few decisive calls so you don't misdiagnose:

- **A large step-1 difference (`>1e-4`) is data order, not numerics.** No floating-point rounding produces a `1e-3` gap on a cross-entropy over a few hundred examples. Go straight to the loader: `generator`, `worker_init_fn`, `sorted()` on file lists. Don't waste time on deterministic algorithms.
- **A tiny step-1 difference (`~1e-7`) that grows is a nondeterministic kernel.** Same data, atomic-add reductions in a different order. `num_workers=0` won't fix it; `torch.use_deterministic_algorithms(True)` will. If it errors, the error names the op.
- **A `~1e-3` gap that only appears across versions/hardware is TF32 or precision, not a bug.** Disable `allow_tf32` to confirm. If accuracy is unchanged, there's nothing to fix — you were comparing across precisions.
- **If it only fails on multi-GPU, it's all-reduce order, not your model.** Confirm by running single-GPU (repeats) vs multi (doesn't). The fix lives in the distributed layer; see [debugging DDP and multi-GPU](/blog/machine-learning/debugging-training/debugging-ddp-and-multi-gpu).
- **If you can't reproduce a result across two *different* machines, that's expected, not a bug.** Target statistical (level-2) reproducibility there. Chasing bit-equality across hardware is a fool's errand.
- **If your run is already bit-identical and you're paying for determinism in production, that's a *cost* bug.** Turn it off. Determinism is for debugging, not throughput.

The connective tissue back to the series: this is the same bisection logic as everywhere else — read the *signature* (gap magnitude and timing), let it point at one of the sources, confirm with a single toggle, fix, verify. The [taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) treats "is your run even reproducible?" as the gate before any other diagnosis, and the [training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) capstone puts `set_full_determinism` at step zero of every investigation.

## Key takeaways

- **`torch.manual_seed` pins one of five sources.** You also need the worker RNG, the kernel selection, the reduction order, and the data order. One seed is not reproducibility.
- **Read the gap to pick the suspect.** Large step-1 gap (`>1e-4`) = data order. Tiny step-1 gap (`~1e-7`) that grows = nondeterministic kernel. `~1e-3` gap across versions = TF32/precision. The magnitude *is* the diagnosis.
- **Floating-point addition is not associative**, so reduction order changes the last bits — and chaos amplifies a `1e-7` step-1 difference into a different final minimum by step 500.
- **The worker bug is the most common one.** Seed each worker from `torch.initial_seed()`, and pass `generator=g` to pin the shuffle. The shared-seed version silently collapses augmentation diversity *and* breaks reproducibility.
- **`torch.use_deterministic_algorithms(True)` errors on purpose** when no deterministic kernel exists — that error names your nondeterminism source. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` in the launch env, before CUDA init.
- **`sorted()` your file lists.** `glob` and `listdir` return arbitrary order; an unseeded order means your shuffle seed permutes a different starting point every run.
- **Run-to-run on one box is achievable; cross-machine bit-equality usually isn't.** Name the level you need and target it; report seeds *and* hardware *and* versions for shareable results.
- **Determinism is a debugging mode, not a default.** It costs 10–40% throughput and sometimes memory. You MUST have it to debug, regression-test, or compare scientifically; you should turn it OFF in production where throughput beats bit-equality.
- **Verify with `torch.equal`, not `torch.allclose`.** The whole point is bit-equality; `allclose` would hide the `1e-7` you're hunting. Bake the two-run diff into CI so you learn the moment reproducibility breaks.

## Further reading

- **PyTorch Reproducibility docs** — the authoritative list of nondeterministic ops, the `use_deterministic_algorithms` / `cudnn.deterministic` / `CUBLAS_WORKSPACE_CONFIG` reference, and the DataLoader worker-seeding guidance. The single best primary source for this post.
- **PyTorch DataLoader docs — "Randomness in multi-process data loading"** — the official `worker_init_fn` / `generator` pattern and the per-worker seed behavior that changed in 1.9+.
- **Micikevicius et al., "Mixed Precision Training" (2018)** — the foundation for why reduced precision changes your sums and why loss scaling exists; essential background for the TF32/bf16 discussion here.
- **NVIDIA, "TensorFloat-32 in the A100 GPU" technical blog** — what TF32 truncates (10-bit mantissa) and why it's a `~1e-3` precision change, not a bug.
- **"Determinism in Deep Learning" (NVIDIA GTC talk, Duncan Riach)** — a deep practitioner treatment of GPU nondeterminism sources and the `tf-determinism` / framework-level fixes.
- **The shared-worker-seed investigation (2020 community analysis of PyTorch repos)** — documents how widespread the duplicated-augmentation bug was and why the naive `worker_init_fn` is wrong.
- Within this series: [a taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) (the decision tree this gates), [the training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) (the capstone), [the input pipeline is lying to you](/blog/machine-learning/debugging-training/the-input-pipeline-is-lying-to-you) (data-order bugs), [mixed-precision debugging: fp16 vs bf16](/blog/machine-learning/debugging-training/mixed-precision-debugging-fp16-vs-bf16) (precision and reductions), and [debugging DDP and multi-GPU](/blog/machine-learning/debugging-training/debugging-ddp-and-multi-gpu) (all-reduce order).

---
title: "Shape Bugs and Silent Broadcasting: The Errors That Don't Crash"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "The worst tensor-shape bugs don't throw an exception — they broadcast, quietly average the wrong thing, and let your loss fall toward a different objective; here is how to make them loud."
tags:
  [
    "debugging",
    "model-training",
    "pytorch",
    "einops",
    "broadcasting",
    "tensor-shapes",
    "finetuning",
    "deep-learning",
    "loss-functions",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/shape-bugs-and-silent-broadcasting-1.png"
---

A teammate once shipped me a regression model that "trained perfectly." The training loss fell from 31.2 to 8.4 over a few thousand steps, the curve was smooth, the GPU stayed busy, and the run finished without a single warning. Then we looked at the predictions on a held-out batch: the model output roughly the same number for every input, regardless of the features. It had not learned the task at all. The loss had genuinely gone down — the optimizer was diligently minimizing *something* — but that something was not the mean squared error we thought we were computing. Buried in the loss function was a one-character mistake: the predictions were shape `[B]` and the targets were shape `[B, 1]`, and subtracting them did not raise an error. It broadcast. The `[B]` predictions got stretched against the `[B, 1]` targets into a full `[B, B]` matrix of every prediction minus every target, and the `.mean()` averaged all `B^2` of those cross terms. The model was being trained to make every prediction close to the *average target across the whole batch*, which it can do by ignoring the inputs entirely. It "trained," it just trained toward the wrong objective.

This is the defining property of a shape bug in modern deep-learning frameworks: the dangerous ones do not crash. NumPy and PyTorch broadcasting was designed to be convenient, and it is — it lets you add a bias `[C]` to an activation `[B, C]` without writing a loop. But that same convenience means that when your tensors have the *wrong* shapes, the framework will very often find a way to make the operation "work" anyway, producing a result with a plausible shape and completely wrong contents. There is no exception, no NaN, no stack trace. The instruments you usually trust — a falling loss, a busy GPU, a clean log — all read green. The figure below lays out why: every check the framework performs is satisfied, the broadcast fires, and the run proceeds to optimize a different loss than the one you wrote.

![A vertical stack showing how a shape bug clears each check, broadcasts to a wrong shape, and still drives the loss down toward a wrong objective](/imgs/blogs/shape-bugs-and-silent-broadcasting-1.png)

By the end of this post you will be able to recognize the shape bugs that are silent, predict exactly *when* a mismatch will broadcast versus crash (the rules are simple and worth memorizing), and instrument your code so these bugs become loud — through exact-shape asserts at function boundaries, `einops` rearrange/reduce as self-documenting shape contracts, shape logging through a forward pass, and unit tests that pin the loss on tiny known inputs. We will work the `[B]`-versus-`[B, 1]` MSE case end to end with real numbers, build a "broadcast detector" that warns when an op stretches a tensor you did not expect it to, and tie the whole thing back to the master tool of this series: when you suspect a silent shape bug, you **make it fail small** — overfit one batch and watch whether the loss floor is suspiciously high.

In the language of the [training-debugging taxonomy](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs), this is a **model-code** bug — one of the six places a bug hides (data, optimization, model code, numerics, systems, evaluation). It is one of the hardest to localize precisely *because* it does not announce itself, so it gets its own discipline: never trust that a tensor has the shape you think it does; assert it.

## 1. The symptom: a loss that falls toward the wrong answer

Let us be precise about the failure mode, because it is unusually deceptive. Three categories of training outcome are easy to reason about. A *healthy* run gives honest positive feedback: the loss falls, the validation metric climbs, the predictions look right. A *crashed* run gives honest negative feedback: a shape mismatch that actually errors throws `RuntimeError: The size of tensor a (B) must match the size of tensor b (C)`, and you fix it in thirty seconds. The shape bug we care about is the fourth category — the one that produces *dishonest positive feedback*. The loss falls, so every dashboard says "working," but the thing being minimized is not your objective.

Here is the running example we will debug throughout this post. We are training a small multilayer perceptron to predict a continuous target — house prices, sensor readings, anything regression-flavored. The model has a final `nn.Linear(hidden, 1)` head, and we compute mean squared error against a target column. The training dashboard reads:

| Instrument | Reading | What it seems to say |
| --- | --- | --- |
| Train loss @ step 0 | 31.2 | "Reasonable starting point" |
| Train loss @ step 3000 | 8.4 | "It's learning well" |
| GPU utilization | 92% | "Compute is healthy" |
| Gradient norm | 3.1 | "Gradients are flowing" |
| Val R^2 | 0.02 | "...essentially zero?" |

Everything is green except the one metric that measures whether the model is actually useful. The validation coefficient of determination, $R^2$, is the fraction of target variance the model explains; a value near 0 means the model is no better than predicting the mean. So we have a loss that fell by nearly 4x and a model that explains nothing. A junior engineer reads this and reaches for the optimizer: lower the learning rate, add more capacity, train longer. A disciplined debugger reads it and asks a sharper question — *is the loss I am minimizing the loss I think I wrote?* — because if the objective is wrong, no amount of tuning the optimizer will help.

The tell is subtle but real: a loss that decreases while the task metric stays at chance is the signature of an objective mismatch, and the most common cause of an objective mismatch in PyTorch is a silent broadcast in the loss. The optimizer is not broken; it is doing exactly what you asked. You just asked for the wrong thing.

### 1.1 What "broadcast" actually means

Before we can hunt the bug, we need to be exact about the mechanism, because the rule is short and predictive. When you apply a binary elementwise operation — subtraction, multiplication, comparison — to two tensors of different shapes, the framework tries to *broadcast* them to a common shape by following two rules:

1. **Align trailing dimensions.** Line the two shapes up from the right. Pad the shorter one with implicit size-1 dimensions on the *left*.
2. **A size-1 dimension stretches to match.** In each aligned position, the sizes must be equal, or one of them must be 1. A size-1 dimension is "stretched" (repeated) to match the other. If neither is 1 and they differ, *then* you get an error.

That second clause is the whole problem. A dimension of size 1 is a wildcard: it will match anything by stretching. So shapes that are semantically incompatible — a `[B]` vector of predictions and a `[B, 1]` column of targets — are *numerically* compatible, because the framework pads `[B]` to `[1, B]`, sees a `[1, B]` against a `[B, 1]`, and stretches both size-1 axes into a `[B, B]` matrix. No error. A wrong answer with a plausible shape.

## 2. The science: exactly when a shape bug is silent

The single most useful thing you can carry from this post is a rule for predicting, *before you run the code*, whether a given shape mismatch will crash loudly or broadcast silently. Get this rule into your fingers and you will catch half of these bugs by reading the code.

### 2.1 The broadcasting algorithm, stated precisely

Take two tensors with shapes $\mathbf{a} = (a_n, a_{n-1}, \dots, a_1)$ and $\mathbf{b} = (b_m, b_{m-1}, \dots, b_1)$, written right-aligned (so $a_1, b_1$ are the trailing dimensions). The result of an elementwise op is defined when, for every aligned position $i$ (padding the shorter shape with 1s on the left), the following holds:

$$
a_i = b_i \quad \text{or} \quad a_i = 1 \quad \text{or} \quad b_i = 1.
$$

When it is defined, the output dimension at position $i$ is $\max(a_i, b_i)$, and any axis that was size 1 is repeated to fill. This is the entire rule. Two consequences follow, and they are the source of every silent shape bug:

**Consequence 1 — a size-1 axis is a silent wildcard.** Because $a_i = 1$ satisfies the condition against *any* $b_i$, a tensor with a trailing (or leading) size-1 dimension will broadcast against almost anything. A `[B, 1]` target will broadcast against `[B]`, `[B, C]`, `[1, C]`, and more. The places size-1 dimensions appear "for free" are exactly the dangerous ones: a label column loaded as `[N, 1]`, a per-sample weight reshaped to `[B, 1]`, the output of a `keepdim=True` reduction, a `[1, B]` from an accidental transpose.

**Consequence 2 — rank mismatch is filled with 1s, not flagged.** Because the shorter shape is left-padded with implicit 1s, a `[B]` and a `[B, 1]` are *not* treated as a rank error. They are aligned as `[1, B]` against `[B, 1]`, which is the perfect storm: both have a size-1 axis, so both stretch, and you get the outer-product `[B, B]`. The framework will never tell you that you mixed up a vector and a column vector; it will quietly compute every pairwise combination.

So the predictive rule is: **a mismatch is silent precisely when every disagreeing axis can be resolved by stretching a size-1 dimension; it crashes only when two axes both differ from each other and from 1.** That is why `[B]` vs `[C]` (with $B \ne C$, no size-1 axes) crashes — and `[B]` vs `[B, 1]` does not.

It is worth walking the alignment by hand once, because once you have traced it you will see it instantly in code. Take `preds` of shape `[4]` and `targets` of shape `[4, 1]`, and subtract. Step one is to right-align and left-pad the shorter shape: `preds` becomes `(1, 4)` and `targets` is `(4, 1)`. Step two checks each aligned position from the right. The trailing position pairs `4` (from preds) against `1` (from targets): not equal, but one is 1, so it is allowed, and the output size is $\max(4, 1) = 4$. The leading position pairs `1` (from preds, the pad) against `4` (from targets): again one is 1, allowed, output size $\max(1, 4) = 4$. Both positions resolved by stretching a size-1 axis, so the operation is legal and the result is `[4, 4]`. *Every single axis disagreement was patched by a size-1 stretch* — that is the silent-bug signature in its purest form. Now contrast `preds` `[4]` against `targets` `[5]`: right-aligned, the trailing position pairs `4` against `5`, neither is 1 and they differ, so the rule fails and PyTorch raises `RuntimeError: The size of tensor a (4) must match the size of tensor b (5)`. The difference between a three-day debug and a thirty-second fix is whether your shapes happened to leave a size-1 axis lying around for the framework to stretch.

### 2.2 Why the silent case still drives the loss down

It is worth understanding *why* the buggy run looks like it is working, because that is what makes you stop trusting the loss curve as proof of correctness. When the `[B]` predictions broadcast against `[B, 1]` targets into a `[B, B]` matrix, the loss becomes the mean over all pairs of $(p_i - t_j)^2$. Expand that:

$$
\mathcal{L}_{\text{bug}} = \frac{1}{B^2} \sum_{i}\sum_{j} (p_i - t_j)^2 = \frac{1}{B}\sum_i p_i^2 - \frac{2}{B^2}\Big(\sum_i p_i\Big)\Big(\sum_j t_j\Big) + \frac{1}{B}\sum_j t_j^2.
$$

The gradient of this with respect to a single prediction $p_k$ is

$$
\frac{\partial \mathcal{L}_{\text{bug}}}{\partial p_k} = \frac{2}{B} p_k - \frac{2}{B^2}\sum_j t_j \cdot \frac{\partial}{\partial p_k}\sum_i p_i \cdot \dots = \frac{2}{B}\Big(p_k - \bar{t}\Big),
$$

where $\bar{t} = \frac{1}{B}\sum_j t_j$ is the *mean target across the batch*. So the buggy loss pushes every prediction toward the batch-mean target, identically, regardless of that sample's input. The loss genuinely falls — collapsing all predictions to $\bar{t}$ reduces the cross-term spread — but the only thing the model learns is a constant. That is exactly the symptom: a falling loss, predictions that ignore the input, and $R^2 \approx 0$. The math predicts the symptom precisely, which is how you know you have the right diagnosis and not a coincidence.

This is the deep reason silent shape bugs are so corrosive: the optimizer is a faithful machine that minimizes whatever scalar you hand it. If the scalar is wrong, you do not get a crash; you get a confident, well-optimized solution to the wrong problem.

### 2.3 Why the loss floor is high, not zero

There is a second prediction the math makes that is worth drawing out, because it is the single most useful early-warning sign. A *correct* MSE on a learnable problem can reach a floor near zero — if the model has the capacity to fit the data, the residuals shrink and the loss follows. The *buggy* broadcast loss cannot. Look again at the expansion in Section 2.2: the buggy loss has a term $\frac{1}{B}\sum_j t_j^2$ that does not depend on the predictions at all. No matter how the model adjusts its outputs, that constant remains, so there is a floor below which the buggy loss physically cannot go. With targets drawn from a distribution of variance $\sigma_t^2$ and mean $\mu_t$, the best the model can do is set every $p_i = \mu_t$, which leaves a residual loss of approximately $\sigma_t^2$ (the variance of the targets). The buggy loss bottoms out at the *target variance*, which for most real datasets is a number well above zero.

That is why the running example's loss "plateaued at 8.4" rather than continuing to fall. It was not that training was slow; it was that the objective had a hard floor at roughly the target variance, and the model had already reached it by collapsing to the mean. When you see a loss decline briskly and then *stick* at a floor that is suspiciously close to the variance of your targets, that is the broadcast bug announcing itself in the only language it has — a refusal to go lower. The correct loss has no such floor on a fittable problem.

### 2.4 The size-1 trap in higher dimensions

The two-tensor case generalizes, and the higher-dimensional version is where production code gets bitten. Consider an attention score tensor of shape `[B, H, T, T]` (batch, heads, query positions, key positions) and a mask you intend to add. If the mask is `[B, 1, 1, T]` it broadcasts correctly across heads and queries — that is the standard, intended pattern. But if the mask is accidentally `[B, T]` (you forgot to add the head and query axes), the broadcasting rule left-pads it to `[1, 1, B, T]` and tries to align *that* against `[B, H, T, T]`. If $B = T$ it silently broadcasts into garbage; if $B \ne T$ it crashes. The same size-1 wildcard that makes the intended `[B, 1, 1, T]` mask convenient is what makes the accidental `[B, T]` mask dangerous. Every size-1 axis you introduce — through `unsqueeze`, `[:, None]`, `keepdim=True`, or a singleton dimension from a data loader — is a place where a future shape mismatch will be resolved silently rather than loudly. The discipline that follows is simple: treat every size-1 axis as a liability to be asserted, not a convenience to be trusted.

## 3. A field guide to the four silent shape bugs

Most silent shape bugs in practice are one of four patterns. Knowing their shape signatures lets you read code and spot them, and lets you write the asserts that catch them. The matrix below is the one I keep next to me when a loss "trains but predicts garbage."

![A matrix mapping four silent shape bug classes to the shape they create, their symptom, the confirming test, and the fix](/imgs/blogs/shape-bugs-and-silent-broadcasting-3.png)

### 3.1 The `[B]` vs `[B, 1]` loss (the outer-product loss)

This is the running example. Predictions are `[B]` (a Linear head with `out_features=1` followed by a `.squeeze()`, or a `[B, 1]` head you forgot to squeeze, against a `[B]` target — it goes wrong in either direction). The subtraction broadcasts to `[B, B]`, and the reduction averages $B^2$ cross terms. Signature: the loss is a *scalar* (so `loss.shape == ()` and nothing looks wrong), but its *value* is systematically off, and the model converges to a constant. The confirming test is two lines: assert the residual tensor has shape `[B]` before reduction, and assert the per-element loss matches what you expect on a tiny known input.

### 3.2 The `[B, 1]` vs `[1, B]` mix-up (the transpose outer product)

You meant to multiply a per-sample quantity of shape `[B, 1]` by another per-sample quantity, but one of them came out transposed as `[1, B]` — often because a `.T` or a `permute` slipped in, or because something was indexed in the wrong order. `[B, 1] * [1, B]` broadcasts to the `[B, B]` outer product. The figure below shows exactly how a column vector times a row vector stretches both size-1 axes into a full grid — the classic outer-product mistake that looks like elementwise multiplication.

![A grid showing a column vector times a row vector stretching both size-one axes into a full B by B outer product](/imgs/blogs/shape-bugs-and-silent-broadcasting-8.png)

### 3.3 The wrong-axis reduction

You call `.mean()` or `.sum()` with the wrong `dim`, or with no `dim` when you needed one. A `tensor.mean(dim=0)` over a `[B, C]` activation reduces over the *batch* instead of the *channels*, giving a `[C]` result that mixes samples together — and then if a later op broadcasts that `[C]` back against `[B, C]`, everything keeps running. Signature: the loss is oddly *constant* across batches (because you averaged away the per-sample variation), or a normalization statistic is computed over the wrong axis. This one rarely makes a `[B, B]` matrix; instead it silently collapses an axis you needed.

### 3.4 The per-sample weight applied the wrong way

You have a per-sample loss weight `w` of shape `[B]` and a per-element loss `[B, C]` (say, a sequence of token losses or a multi-output regression). You write `loss = (w * per_element).mean()`. But `[B] * [B, C]` aligns trailing dims as `[1, B]` against `[B, C]`, which broadcasts to `[B, B]` if $B = C$, or *crashes* if $B \ne C$ — so this one is sometimes loud and sometimes silent depending on whether your batch size equals your output dimension. When it is silent, the weights are applied across the wrong axis: sample 3's weight scales every sample's *column 3*. You wanted `w[:, None] * per_element` so the weight broadcasts down the columns. Signature: gradients are skewed in a way that correlates with the per-sample weights along the wrong axis.

### 3.5 Why these four pass every shape check

It is worth being explicit about *why* none of these four bugs is caught by the usual defenses, because that explains why you need the specific disciplines in the rest of this post. They pass the framework's broadcasting check by construction — that is the definition of "silent." They pass a casual `print(loss)` because the loss is a finite, plausible-looking scalar. They pass a NaN check because nothing overflows or divides by zero. They pass a gradient-flow check because gradients *do* flow — to the wrong objective. They even partially pass the overfit-one-batch test, because a broadcast loss can still be driven somewhat down. The only checks that catch them are the ones that look at the *shape of the intermediate tensors*: the residual before reduction, the per-element loss before weighting, the logits-and-labels alignment before cross-entropy. That is the throughline of every diagnostic in this post — stop trusting the scalar and start asserting the shapes that produce it.

A useful way to internalize the four is by their output shape signature. Bugs 3.1 and 3.2 both manufacture a `[B, B]` matrix where you expected a `[B]` vector — the outer-product family, recognizable because the residual or product tensor is suddenly square. Bug 3.3 *collapses* an axis you needed, producing a tensor of lower rank than intended — recognizable because a per-sample quantity becomes a single number or a per-feature quantity loses its batch dimension. Bug 3.4 *preserves* the shape but applies a quantity along the wrong axis — the sneakiest of the four, because the output shape is correct and only the *values* are wrong. The first three you can catch by asserting rank and shape; the fourth you catch by asserting the *intent* with einops named axes, which is why named-axis tooling earns its place in Section 6.

## 4. The diagnostic: shape-assert discipline

The cure for silent shape bugs is to make shapes *loud*. The framework will not check them for you, so you check them — at every function boundary, on the tensors that matter, with the exact shape spelled out. This is cheap, it is self-documenting, and it converts a class of bugs that cost you days into a class that costs you a single assertion error pointing at the exact line.

Here is the discipline, codified into a tiny helper that asserts exact shapes and gives you a readable error. The `*` wildcard lets you assert the parts of the shape you care about while leaving batch or sequence dimensions free.

```python
import torch

def assert_shape(t: torch.Tensor, expected: tuple, name: str = "tensor"):
    """Assert a tensor's shape, with '*' as a per-axis wildcard.

    assert_shape(x, ('*', 3, 224, 224), 'image')  # any batch, fixed CHW
    """
    actual = tuple(t.shape)
    if len(actual) != len(expected):
        raise AssertionError(
            f"{name}: rank {len(actual)} != {len(expected)}; "
            f"got {actual}, expected {expected}"
        )
    for i, (a, e) in enumerate(zip(actual, expected)):
        if e != "*" and a != e:
            raise AssertionError(
                f"{name}: axis {i} is {a}, expected {e}; "
                f"full shape {actual} vs {expected}"
            )
    return t
```

Now you wire it into the loss, which is where the most damaging shape bugs live. The key move is to assert the shape of the *residual before reduction*, not just the final scalar — because the scalar always looks fine.

```python
import torch
import torch.nn.functional as F

def mse_loss_checked(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    B = preds.shape[0]
    # Both must be exactly [B] -- this is the assertion that catches the bug.
    assert_shape(preds,   (B,), "preds")
    assert_shape(targets, (B,), "targets")
    resid = preds - targets
    assert_shape(resid, (B,), "resid")   # if this is [B, B], we broadcast!
    loss = (resid ** 2).mean()
    assert loss.dim() == 0, f"loss must be scalar, got shape {tuple(loss.shape)}"
    return loss
```

If `targets` arrives as `[B, 1]`, the very first `assert_shape(targets, (B,), ...)` fires with a precise message — rank 2 versus rank 1, got `(B, 1)`, expected `(B,)` — and you fix it at the source instead of discovering a dead model three days later. This is the single highest-value habit in this post: **assert exact shapes at function boundaries, especially in the loss.** A loss function is the one place where a wrong shape is guaranteed to be silent (the output is always a scalar), so it is the one place you must check the inputs.

### 4.1 A reusable assert-shapes decorator

Sprinkling `assert_shape` calls works, but for functions you call constantly — the loss, the model's forward, a collate function — a decorator that declares the contract once is cleaner and harder to forget. The decorator below takes a spec for the inputs and output, checks them on every call, and raises with the function name and argument name so the error points straight at the culprit. The `"B"` symbol binds to whatever the first call sees and must stay consistent across all tensors in that call, which is exactly what catches a `[B]` prediction paired with a `[B, 1]` target.

```python
import functools
import torch

def shapes(inputs: dict, output=None):
    """Decorator: declare input/output shapes with named symbolic axes.

    @shapes({"preds": ("B",), "targets": ("B",)}, output=())
    def mse(preds, targets): ...
    """
    def decorate(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            bound = dict(zip(fn.__code__.co_varnames, args))
            bound.update(kwargs)
            env = {}  # symbol -> concrete size, shared across all tensors
            def check(name, t, spec):
                actual = tuple(t.shape)
                if len(actual) != len(spec):
                    raise AssertionError(
                        f"{fn.__name__}: {name} rank {len(actual)} != {len(spec)} "
                        f"(got {actual}, spec {spec})")
                for i, (a, s) in enumerate(zip(actual, spec)):
                    if isinstance(s, int):
                        if a != s:
                            raise AssertionError(
                                f"{fn.__name__}: {name} axis {i} is {a}, want {s}")
                    else:  # symbolic axis like "B"
                        if s in env and env[s] != a:
                            raise AssertionError(
                                f"{fn.__name__}: {name} axis {i}={a} but {s}={env[s]} "
                                f"already (shape {actual})")
                        env[s] = a
            for name, spec in inputs.items():
                if name in bound and torch.is_tensor(bound[name]):
                    check(name, bound[name], spec)
            out = fn(*args, **kwargs)
            if output is not None and torch.is_tensor(out):
                check("<return>", out, output)
            return out
        return wrapped
    return decorate
```

Now the loss contract is declared in one place, and the `"B"` binding enforces that predictions and targets share a batch size *and rank*:

```python
@shapes({"preds": ("B",), "targets": ("B",)}, output=())
def mse_loss(preds, targets):
    return ((preds - targets) ** 2).mean()

# mse_loss(torch.randn(4), torch.randn(4))        # passes
# mse_loss(torch.randn(4), torch.randn(4, 1))     # raises: targets rank 2 != 1
```

The second call raises immediately — `targets rank 2 != 1` — before a single broadcast happens. The decorator pays for itself the first time it catches a `[B, 1]` target that slipped in from a refactored data loader, and it doubles as documentation: anyone reading `mse_loss` sees the exact shape contract in the decorator line. The symbolic `"B"` axis is the part that earns its keep — it does not just check that each tensor has the right rank, it checks that every tensor sharing the `"B"` symbol agrees on the concrete batch size, so a `[4]` prediction against a `[8]` target (a collation bug that duplicated the batch) is caught just as cleanly as the `[B, 1]` rank mismatch.

### 4.2 Printing shapes through a forward pass

When the bug is deeper than the loss — somewhere in the model's forward — the fastest localizer is to print the shape after every layer. You do not need a debugger; you need a tour of the shapes. The figure below traces a typical NLP forward pass and shows the exact moment a `[B, T, V]` logits tensor meets a `[B, T]` target tensor, where flattening saves you and forgetting to flatten broadcasts into nonsense.

![A dataflow graph tracing tensor shapes through embed, encoder, and head, branching into a correct flatten path and a silent broadcast path](/imgs/blogs/shape-bugs-and-silent-broadcasting-5.png)

A forward hook gives you this for free without editing the model:

```python
def register_shape_logger(model: torch.nn.Module):
    handles = []
    for name, module in model.named_modules():
        if name == "":           # skip the top-level container
            continue
        def hook(mod, inp, out, _name=name):
            def shp(x):
                return tuple(x.shape) if torch.is_tensor(x) else type(x).__name__
            in_shapes = [shp(x) for x in inp]
            out_shape = shp(out)
            print(f"{_name:<30} in={in_shapes} out={out_shape}")
        handles.append(module.register_forward_hook(hook))
    return handles  # call h.remove() for h in handles when done
```

Run one batch through the model with this attached and you get a complete shape trace. The bug almost always jumps out: a `[B, T, V]` where you expected `[B, V]` (you forgot to pool over the sequence), a `[B, 1]` that should be `[B]`, a `[B, C, H, W]` that became `[B, H, W, C]` after a stray `permute`. Reading the trace is faster than reasoning about the code, because the framework *tells you the truth* about what each layer produced.

### 4.3 Asserting shapes inside the forward, not just at the boundary

Boundary asserts catch the inputs and outputs of a function, but for a model with a long forward pass, you also want a few asserts *inside* it at the points where shape mistakes cluster: right after a reshape, right after a transpose or permute, and right before any reduction. These are the three operations that change rank or reorder axes, which is to say the three operations that manufacture silent broadcasts downstream. A reshape can quietly produce any shape whose element count matches (`x.view(B, -1)` will happily collapse the wrong axes); a permute can swap two dimensions you needed in a particular order; a reduction can eat the wrong axis. Asserting immediately after each one localizes the bug to a single line.

```python
import torch
import torch.nn as nn

class PooledClassifier(nn.Module):
    def __init__(self, d_model, n_classes, n_tokens):
        super().__init__()
        self.encoder = nn.TransformerEncoderLayer(d_model, nhead=4, batch_first=True)
        self.head = nn.Linear(d_model, n_classes)
        self.n_tokens = n_tokens

    def forward(self, x):                 # x: [B, T, d_model]
        B, T, D = x.shape
        h = self.encoder(x)
        assert h.shape == (B, T, D), f"encoder changed shape: {tuple(h.shape)}"
        pooled = h.mean(dim=1)            # reduce over tokens -> [B, D]
        assert pooled.shape == (B, D), f"pool gave {tuple(pooled.shape)}, want [B, D]"
        logits = self.head(pooled)        # [B, n_classes]
        assert logits.dim() == 2 and logits.shape[0] == B, \
            f"logits {tuple(logits.shape)} not [B, n_classes]"
        return logits
```

These asserts cost a few microseconds per forward pass and are worth leaving in during development; many teams gate them behind a `if self.debug:` flag so they vanish in production. The `pooled = h.mean(dim=1)` line is the one that bites in practice — `dim=1` reduces over tokens (correct), but `dim=0` would reduce over the batch (a wrong-axis reduction from Section 3.3 that yields a `[T, D]` tensor and then crashes or broadcasts at the head). The assert immediately after pins the intent: pooling must give `[B, D]`, and if it gives anything else the error names the exact line.

### 4.4 A diagnostic table: symptom to confirming test

When you arrive at a failing run cold, you want a lookup from what you observe to the single test that confirms or rules out a shape bug. This is the table I run down before touching code.

| What you observe | Likely shape bug | Confirming test (one line) | Fix |
| --- | --- | --- | --- |
| Loss falls, plateaus high, $R^2 \approx 0$ | `[B]` vs `[B, 1]` outer product | `assert resid.shape == (B,)` | `targets.view(-1)` |
| Predictions all near-identical | broadcast to batch-mean target | `print(preds.std())` near 0 | fix loss residual shape |
| Loss oddly constant across batches | wrong-axis reduction | `print(per_elem.shape)` after reduce | name axis with einops |
| Gradients skewed by sample index | per-sample weight on wrong axis | `assert (w * x).shape == x.shape` | `w[:, None] * x` |
| CE accepts but result is wrong | logits/labels not flattened | `print(logits.shape, labels.shape)` | `rearrange` flatten |
| Mask "works" but model peeks/leaks | mask broadcast on wrong axis | trace attention score shape | reshape mask to `[B,1,1,T]` |

Each row is a symptom-to-test-to-fix triple, and the tests are deliberately one line each — the whole point is that confirming a shape bug should be fast and decisive, not a multi-hour spelunk. The unifying move is always the same: stop looking at the loss value and look at the shape of the tensor that produced it.

## 5. Worked example: the MSE `[B]` vs `[B, 1]` bug with real numbers

Now let us nail the running example down to arithmetic, because seeing the numbers is what makes the bug unforgettable. The figure below walks the four-sample case left to right.

![A left-to-right timeline of a four-sample MSE computation comparing the buggy broadcast mean against the correct mean](/imgs/blogs/shape-bugs-and-silent-broadcasting-6.png)

#### Worked example: MSE with four samples

Take a batch of $B = 4$. The model predicts `preds = [2, 4, 6, 8]` and the true targets are `targets = [1, 3, 5, 7]`. The *intended* per-sample residuals are `preds - targets = [1, 1, 1, 1]`, every squared residual is 1, and the correct MSE is

$$
\mathcal{L}_{\text{true}} = \frac{1}{4}(1^2 + 1^2 + 1^2 + 1^2) = 1.0.
$$

Hold on — that gives 1.0, which is the clean case. The bug appears when the *shapes* are wrong. Suppose `preds` is `[4]` (a vector) but `targets` was loaded as `[4, 1]` (a column). Now `preds - targets` does not compute `[1, 1, 1, 1]`. It broadcasts: `preds` is padded to `[1, 4]`, `targets` is `[4, 1]`, and the result is the `[4, 4]` matrix whose entry $(i, j)$ is $t_i$... no — whose entry $(i, j)$ is `targets[i] - preds[j]` (or `preds[j] - targets[i]` depending on operand order). Concretely, with `preds = [2,4,6,8]` and `targets = [[1],[3],[5],[7]]`, the broadcast `preds - targets` is:

$$
\begin{bmatrix}
2-1 & 4-1 & 6-1 & 8-1\\
2-3 & 4-3 & 6-3 & 8-3\\
2-5 & 4-5 & 6-5 & 8-5\\
2-7 & 4-7 & 6-7 & 8-7
\end{bmatrix}
=
\begin{bmatrix}
1 & 3 & 5 & 7\\
-1 & 1 & 3 & 5\\
-3 & -1 & 1 & 3\\
-5 & -3 & -1 & 1
\end{bmatrix}.
$$

Square every entry and you get a matrix whose sum is $1+9+25+49 + 1+1+9+25 + 9+1+1+9 + 25+9+1+1 = 136$. The buggy mean is $136 / 16 = 8.5$. The *correct* MSE for this data — using the diagonal only, the matched pairs — is the mean of the diagonal squared residuals: $(1 + 1 + 1 + 1)/4 = 1.0$. But notice the more general lesson: if the predictions had genuinely been off (say `preds = [5, 5, 5, 5]` against `targets = [1, 3, 5, 7]`), the correct MSE would be $\frac{1}{4}((5-1)^2 + (5-3)^2 + (5-5)^2 + (5-7)^2) = \frac{1}{4}(16 + 4 + 0 + 4) = 6.0$, while the buggy broadcast version would compute a `[4, 4]` matrix mean of roughly $\frac{1}{16}\sum_{i,j}(5 - t_j)^2 = \frac{1}{4}\sum_j (5 - t_j)^2 = 6.0$ as well in *this* constant-prediction case — which is exactly why the bug hides: when the model predicts a constant, the buggy and correct losses can agree, so the optimizer's fastest path to a low buggy loss is to predict a constant. It learns the degenerate solution that the bug rewards.

The figure below contrasts the two computations side by side: the buggy path produces a `[B, B]` matrix and a scalar of 8.5, the fixed path produces a `[B]` vector and the true MSE.

![A before and after comparison contrasting the buggy preds minus targets broadcast to a B by B matrix against the fixed equal-length vector](/imgs/blogs/shape-bugs-and-silent-broadcasting-2.png)

#### Worked example: confirming it in a five-line script

You never have to reason about this in your head. Reproduce it:

```python
import torch

preds   = torch.tensor([5.0, 5.0, 5.0, 5.0])      # shape [4]
targets = torch.tensor([1.0, 3.0, 5.0, 7.0]).view(4, 1)  # shape [4, 1] -- the bug

bad  = ((preds - targets) ** 2).mean()             # broadcasts to [4, 4]
good = ((preds - targets.view(-1)) ** 2).mean()    # [4] - matched

print("residual shape (buggy):", (preds - targets).shape)   # torch.Size([4, 4])
print("buggy loss:", bad.item())                            # 6.0  (looks plausible!)
print("good  loss:", good.item())                           # 6.0  here, but...
print("residual shape (fixed):", (preds - targets.view(-1)).shape)  # torch.Size([4])
```

The output prints `torch.Size([4, 4])` for the buggy residual — that single line is the smoking gun. The loss *value* can look identical for a constant prediction, which is the trap; the *shape* of the residual never lies. This is why the discipline is to assert the residual shape, not to eyeball the loss value.

## 6. The fix: `einops` as a self-documenting shape contract

Asserts catch shape bugs after you write them. A better class of tool prevents you from writing them in the first place by forcing you to *name* every axis. That tool is `einops`. Its `rearrange`, `reduce`, and `repeat` functions take a pattern string in which you spell out the input and output axes by name, and the library checks that the named axes are consistent. A wrong axis becomes an immediate, readable error instead of a silent broadcast.

Compare a bare reduction to an einops reduction. The figure below makes the contrast concrete: bare ops guess the axis and any product fits, so a rank slip hides; named axes make the mismatch raise at the exact line.

![A before and after comparison contrasting bare mean and view operations that hide rank slips against named-axis einops reductions that raise errors](/imgs/blogs/shape-bugs-and-silent-broadcasting-7.png)

```python
import torch
from einops import rearrange, reduce

x = torch.randn(8, 16, 32)   # [batch, tokens, features]

# Bare ops: which axis did .mean(1) reduce? You have to remember.
pooled_bare = x.mean(1)              # [8, 32] -- but was 1 the right axis?

# einops: the axis is named, the intent is in the code.
pooled_named = reduce(x, "b t f -> b f", "mean")   # explicit: reduce tokens
assert pooled_named.shape == (8, 32)

# A wrong pattern is a loud error, not a silent broadcast:
# reduce(x, "b t -> b", "mean")   # raises: pattern has 2 axes, tensor has 3
```

The payoff is biggest in the place broadcasts hurt most — flattening logits and labels for a sequence loss. The bare version is `logits.view(-1, V)` and `labels.view(-1)`, and if you get one of them wrong the cross-entropy will sometimes broadcast or sometimes crash unpredictably. The einops version names the axes, so the `b` and `t` must be consistent between the two tensors:

```python
import torch
import torch.nn.functional as F
from einops import rearrange

logits = torch.randn(4, 10, 50)   # [B, T, V]
labels = torch.randint(0, 50, (4, 10))  # [B, T]

# einops makes the flatten a contract: same b, same t, V preserved.
flat_logits = rearrange(logits, "b t v -> (b t) v")  # [40, 50]
flat_labels = rearrange(labels, "b t -> (b t)")      # [40]
assert flat_logits.shape[0] == flat_labels.shape[0]  # 40 == 40

loss = F.cross_entropy(flat_logits, flat_labels)
```

If `labels` had been the wrong shape — say `[B, T, 1]` from a stray `keepdim` — the `rearrange(labels, "b t -> ...")` pattern would raise because the tensor has rank 3 and the pattern declares rank 2. The bug becomes a one-line error at the rearrange, naming exactly which tensor is wrong. This is what "self-documenting shape contract" means: the code *states* the shape it expects, and the library *enforces* it.

There is a related idea worth naming even though the tooling is still maturing: **named tensors**, where each axis carries a name (`"batch"`, `"channels"`) and operations refuse to align axes with mismatched names. PyTorch has experimental support via `tensor.refine_names("B", "C")`. The promise is that broadcasting `[B]` against `[B, 1]` would raise because the names do not line up, eliminating the wildcard problem at the source. In practice today, einops patterns give you most of the benefit with broad library support, so reach for einops first.

### 6.1 einops catches the bug by *naming* the axis

The reason einops is more than syntactic sugar is that it shifts shape checking from "do the element counts happen to match" to "do the *named* axes match." A bare `x.view(B, -1)` succeeds as long as the total number of elements is divisible by `B` — it does not care *which* axes you collapsed, so a tensor that should have been `[B, T, V]` and a tensor that became `[B, V, T]` after a stray transpose will *both* reshape to a `[B, T*V]` of the right size, and the bug propagates. An einops `rearrange(x, "b t v -> b (t v)")` declares that the input is `(batch, tokens, vocab)` in that order; if the actual tensor is `[B, V, T]`, the *values* still flow but at least the pattern documents the contract, and if the rank is wrong the pattern raises. For the reduction case the protection is stronger: `reduce(x, "b t f -> b f", "mean")` will only accept a rank-3 tensor and will only reduce the `t` axis, so the wrong-axis reduction bug from Section 3.3 becomes impossible to write — you cannot accidentally reduce `b` because you did not name `b` as the reduced axis.

The most valuable place to deploy this is the loss boundary, where the four silent bugs concentrate. Here is a reduction-aware multi-output loss that uses einops to make the per-sample-weight contract explicit, eliminating bug 3.4 by construction:

```python
import torch
import torch.nn.functional as F
from einops import reduce

def weighted_multi_output_mse(preds, targets, sample_w):
    """preds, targets: [B, K]; sample_w: [B] per-sample weights."""
    B, K = preds.shape
    assert targets.shape == (B, K), f"targets {tuple(targets.shape)} != [B, K]"
    assert sample_w.shape == (B,),  f"weights {tuple(sample_w.shape)} != [B]"
    per_element = (preds - targets) ** 2          # [B, K]
    per_sample  = reduce(per_element, "b k -> b", "mean")  # [B] -- named reduce
    weighted    = sample_w * per_sample           # [B] * [B] -> [B], safe
    return weighted.mean()
```

By reducing over `k` *first* with a named pattern, the per-sample loss is unambiguously `[B]`, so multiplying by the `[B]` weight is a clean elementwise op with no chance of broadcasting to `[B, B]`. The asserts at the top pin the input contract; the named `reduce` pins the reduction axis; the elementwise multiply is now provably safe. This is the pattern to copy whenever you weight a loss: reduce to a per-sample vector with a named axis, *then* weight, never the other way around.

## 7. The before→after evidence: a model that "trained" but predicted garbage

Let us make the payoff concrete and measurable, because "assert your shapes" is only convincing with a before-and-after. We take the running regression model, run it with the `[B]` vs `[B, 1]` bug, then fix the single line and rerun. Everything else — data, model, learning rate, seed — is held constant. Only the loss reduction shape changes.

| Instrument | Before (broadcast bug) | After (shapes fixed) | What changed |
| --- | --- | --- | --- |
| Train loss floor | 8.4 | 0.31 | Loss reaches a far lower, real floor |
| Val R^2 | 0.02 | 0.81 | Model now explains target variance |
| Prediction spread (std) | 0.04 | 1.12 | Predictions stop collapsing to a constant |
| Residual tensor shape | `[B, B]` | `[B]` | The smoking gun, before reduction |
| Overfit-one-batch loss | 1.9 (stuck) | 1e-4 | One batch now memorizes cleanly |

Read the table top to bottom and the story is unambiguous. Before the fix, the loss floor of 8.4 was a *high* floor — the model could not get below it because it was minimizing the wrong objective and the best it could do was predict the batch-mean target. The prediction standard deviation of 0.04 confirms the diagnosis from Section 2.2: the model collapsed to a near-constant output, exactly as the gradient analysis predicted. After the fix, the loss reaches 0.31 (a genuine floor for this problem), $R^2$ jumps to 0.81, and predictions regain real spread.

The most diagnostic row is the last one. The overfit-one-batch test — train on a single fixed batch until it memorizes — went from *stuck at 1.9* to *cratering to 1e-4*. This is the connection to the [overfit-a-single-batch test](/blog/machine-learning/debugging-training/the-overfit-a-single-batch-test): a silent broadcast often lets the model *partially* overfit (it can still drive the wrong objective somewhat down), so the overfit test does not always fail cleanly — it sometimes plateaus at a suspiciously high floor instead of going to zero. That high floor is itself the signal. When your overfit-one-batch loss bottoms out at 0.4 instead of 1e-4, do not assume the test "passed loosely" — suspect that the loss is computing the wrong thing, and add an exact-shape assert on the residual.

#### How to confirm this honestly

I want to be precise about what these numbers mean and how you would reproduce them, because fabricated precision is worse than honest approximation. The exact figures depend on the dataset, model, and seed; the ones above are representative of a small-MLP-on-tabular-regression setup and the *direction and order of magnitude* are what generalize, not the third decimal. The honest confirmation procedure is: (1) print the residual shape before reduction and verify it is `[B]`, not `[B, B]`; (2) check that the overfit-one-batch loss reaches ~1e-4 after the fix; (3) confirm prediction standard deviation is comparable to target standard deviation rather than near zero. Those three checks are deterministic and do not depend on the exact loss values, so they are the ones to trust.

## 8. Bisecting to a shape bug: the decision flow

When you face a real failing run, you do not jump straight to "it must be a broadcast." You bisect. The figure below is the decision flow I run when a loss falls but predictions are garbage.

![A decision tree starting from loss falls but predictions garbage, branching on overfit one batch and whether the loss is a scalar, ending at a broadcast bug and its fix](/imgs/blogs/shape-bugs-and-silent-broadcasting-4.png)

The flow has two branch points, and each one cleaves the space of suspects:

**Branch 1 — does one batch overfit?** Run the overfit-one-batch test. If the loss *stalls completely flat* (never moves), the problem is upstream of the loss shape: a frozen model, a detached graph, a learning rate of zero, or a data path that feeds constant inputs. That is a different post (`your-model-isnt-learning-what-you-think` covers gradient-flow and frozen-layer bugs). But if the loss falls *partway* and then plateaus at a high floor — say it drops from 2.3 to 0.4 and sticks — you are in shape-bug territory, because a broadcast lets the wrong objective decrease only so far.

**Branch 2 — is the loss the shape you expect, computed from residuals of the shape you expect?** This is where you assert. Add `assert resid.shape == (B,)` (or whatever the correct per-element shape is) right before the reduction. If the residual is `[B, B]`, you have found it. If the residual is the right shape but the loss is still wrong, the bug is elsewhere — in the model's forward (run the shape-logging hook from Section 4.1) or in the data (the targets themselves are wrong). The assert is decisive: it either fires and hands you the bug, or it passes and rules out the broadcast, sending you to the next suspect.

This is the bisection mindset of the whole series in miniature: do not guess and patch. Each test you run should *eliminate* a region of the suspect space. The overfit test splits "model/optimizer broken" from "objective wrong"; the shape assert splits "broadcast in the loss" from "everything downstream of a correct loss."

## 9. Building a broadcast detector

Asserts protect the boundaries you remember to guard. For the operations you forget — the stray multiply deep in a custom layer — you want a tripwire that fires whenever an op broadcasts in a way you did not intend. PyTorch's `__torch_function__` mechanism lets you wrap tensors and inspect every operation, but a lighter-weight approach catches the common case: a context manager that monkey-patches a few elementwise ops to warn when the operands have different ranks (the most reliable indicator of an unintended broadcast).

```python
import torch
import warnings
from contextlib import contextmanager

@contextmanager
def warn_on_broadcast():
    """Warn whenever a binary elementwise op broadcasts across different ranks."""
    originals = {}
    ops = ["add", "sub", "mul", "div"]

    def make_wrapper(name, orig):
        def wrapper(self, other, *args, **kwargs):
            if torch.is_tensor(other) and self.dim() != other.dim():
                warnings.warn(
                    f"{name}: rank mismatch {tuple(self.shape)} "
                    f"{name} {tuple(other.shape)} -> broadcast",
                    stacklevel=2,
                )
            return orig(self, other, *args, **kwargs)
        return wrapper

    for name in ops:
        orig = getattr(torch.Tensor, f"__{name}__")
        originals[name] = orig
        setattr(torch.Tensor, f"__{name}__", make_wrapper(name, orig))
    try:
        yield
    finally:
        for name, orig in originals.items():
            setattr(torch.Tensor, f"__{name}__", orig)
```

You wrap a single forward+loss pass in it during development:

```python
with warn_on_broadcast():
    preds = model(x)                 # [B]
    loss = ((preds - targets) ** 2).mean()   # targets is [B, 1] -> WARNS
```

The first time you run a forward pass under this detector, every accidental rank-crossing broadcast prints a warning with the exact shapes. You then either fix the shape or, for the deliberate broadcasts (adding a `[C]` bias to a `[B, C]` activation is *correct* and common), confirm it is intended. The detector is noisy by design — it warns on *every* rank mismatch, including the legitimate ones — so it is a development tool you run once to audit a model, not something you leave on in production. But it is unbeatable for catching the broadcast you did not know you wrote.

A more surgical version checks for the specific dangerous pattern: a size-1 axis stretching against a non-1 axis where you expected exact equality. That is harder to get right generically (size-1 broadcasts are often intended), which is why the boundary asserts and einops contracts remain the primary defense; the detector is the net underneath them.

In practice I run the broadcast detector exactly once per model — on the first forward-plus-loss pass of a new project, or right after a refactor that touched shapes — and then turn it off. The reason is that a healthy training step contains many *legitimate* rank-crossing broadcasts: adding a `[D]` bias to a `[B, T, D]` activation, applying a `[1, 1, T]` positional encoding, scaling by a scalar learning-rate-like factor. Leaving the warning on would bury the one bad broadcast under dozens of good ones. The one-shot audit is the right ergonomic: you read every warning once, classify each as intended or not, fix the bad ones at their boundary with an assert or an einops contract, and then rely on those permanent guards instead of the noisy global detector. The detector is a one-time smoke test, not a permanent alarm — it surfaces every rank-crossing broadcast once so the targeted asserts can hold the line from then on. This is the same division of labor as in software more broadly: a broad linter pass to surface issues, then narrow assertions and tests to lock the fixes in place permanently.

## 10. Unit-testing the loss on tiny known inputs

The most durable defense against silent shape bugs is a unit test that pins the loss to a hand-computed value on a tiny input. Shape bugs survive because the loss *looks* plausible; a test that checks the *exact* value on inputs where you know the answer makes the bug impossible to merge. This is the practice that turns a one-time fix into a permanent guarantee.

```python
import torch
import pytest

def test_mse_matches_hand_computed():
    preds   = torch.tensor([2.0, 4.0, 6.0, 8.0])      # [4]
    targets = torch.tensor([1.0, 3.0, 5.0, 7.0])      # [4]
    # Hand-computed: residuals [1,1,1,1], squared [1,1,1,1], mean = 1.0
    loss = mse_loss_checked(preds, targets)
    assert loss.dim() == 0, "loss must be a scalar"
    assert abs(loss.item() - 1.0) < 1e-6, f"expected 1.0, got {loss.item()}"

def test_mse_rejects_column_targets():
    preds   = torch.tensor([2.0, 4.0, 6.0, 8.0])      # [4]
    targets = torch.tensor([1.0, 3.0, 5.0, 7.0]).view(4, 1)  # [4, 1] -- the bug
    # The shape-checked loss must REFUSE this, not silently broadcast.
    with pytest.raises(AssertionError):
        mse_loss_checked(preds, targets)
```

Two tests, two guarantees. The first pins the *value*: on `[2,4,6,8]` vs `[1,3,5,7]` the MSE is exactly 1.0, and any broadcast bug would change that number, so the test fails loudly if someone reintroduces the bug. The second pins the *shape contract*: a `[B, 1]` target must raise, not broadcast. Together they make the silent bug a red CI run instead of a dead training run. This is the cheapest insurance in machine learning — a five-line test that protects against a class of bugs that costs days — and it is the practice I trust more than any runtime assert, because it runs on every commit whether or not anyone remembers to look.

#### Worked example: the test catches a refactor regression

Here is the payoff in practice. Suppose six months later someone refactors the data loader to return targets as `[B, 1]` "for consistency with the multi-output case." The training run would still execute, the loss would still fall, and the model would silently degrade — except `test_mse_rejects_column_targets` fails in CI the moment the loss function receives a `[B, 1]`, before the change ever merges. The engineer sees a precise assertion error pointing at the shape, adds a `.view(-1)` at the loss boundary, and the regression never reaches a GPU. That is the difference between a five-line test and a five-day debug.

## 11. Case studies and real signatures

These bugs are not hypothetical; they recur across every modality, and the experienced eye recognizes them by signature. Here are the patterns I have seen most often, organized by where they bite.

### 11.1 NLP: the `[B, T, V]` vs `[B, T]` cross-entropy flatten

The most common silent shape bug in language-model training is forgetting to flatten logits and labels before cross-entropy. `F.cross_entropy` expects logits `[N, V]` and targets `[N]`. If you pass `[B, T, V]` logits and `[B, T]` labels, PyTorch's cross-entropy *does* accept the higher-rank form (it treats dims after the first as spatial), so it does not crash — but the moment your shapes are *almost* right (a `[B, T, 1]` label from a stray `keepdim`, or a transposed `[B, V, T]` logits from a `permute` you added for a different op), you get either a silent broadcast or a wrong-axis loss. The fix is the einops flatten from Section 6, which names `b`, `t`, and `v` and makes the contract explicit. This bug ties directly to the [loss-function bugs](/blog/machine-learning/debugging-training/loss-function-bugs) post, where reduction and reshape mistakes are the central theme.

#### Worked example: the `[B, T, V]` vs `[B, T]` token loss

Take a tiny language-model batch: $B = 2$ sequences, $T = 3$ tokens each, vocabulary $V = 5$. The logits are `[2, 3, 5]` and the labels are `[2, 3]`. The *correct* cross-entropy treats each of the $B \cdot T = 6$ token positions as one classification example, so the loss is the mean over 6 per-token cross-entropies. Now suppose a refactor accidentally transposed the logits to `[2, 5, 3]` — someone added a `permute` to feed a convolution and forgot to undo it. PyTorch's `cross_entropy` interprets the second axis as the class axis, so it now reads 5 as the vocabulary and 3 as a spatial dimension, and it expects labels of shape `[2, 3]` — which you have. *Nothing crashes.* But the loss is now computed with the 5 logits per position treated as classes over the wrong axis: token position 0's label is scored against a mix of vocabulary entries and sequence positions. The numbers come out finite and plausible — a starting loss around $\ln(5) \approx 1.61$, falling steadily — yet the model is learning to predict a scrambled target. The detector is to print both shapes right before the loss: `logits [2, 5, 3]` against `labels [2, 3]` should immediately read as wrong, because the vocabulary axis (5) is not last. The einops flatten `rearrange(logits, "b t v -> (b t) v")` would have raised the moment the named `t` and `v` axes did not match the transposed tensor, turning a silent objective swap into a one-line error.

### 11.2 Computer vision: mask and logit shape mismatches in segmentation

Semantic segmentation is a shape minefield. The model outputs `[B, C, H, W]` logits (C classes per pixel), and the mask target is `[B, H, W]` (a class index per pixel). `F.cross_entropy` handles this correctly — but only if the shapes are exactly right. A common silent bug: the mask is loaded as `[B, 1, H, W]` (a channel dimension from the image loader), and a downstream `loss = criterion(logits, mask)` either broadcasts or computes over the wrong axis depending on the exact code path. Another: someone resizes the mask with bilinear interpolation (which is correct for images and *wrong* for integer masks, producing fractional class indices), and the loss silently consumes nonsense labels. The detector here is the shape-logging hook plus an assert that the mask is `[B, H, W]` integer-typed before the loss.

### 11.3 Multi-task heads: per-task weights applied across the wrong axis

A multi-task model produces a `[B, K]` tensor of per-task losses (K tasks), and you weight tasks with a `[K]` weight vector: `total = (task_weights * per_task_loss).mean()`. Here `[K] * [B, K]` broadcasts *correctly* (the `[K]` aligns with the trailing axis), which is the intended behavior. But the symmetric mistake — weighting *samples* with a `[B]` vector via `(sample_weights * per_task_loss)` — broadcasts `[B]` against `[B, K]` as `[1, B]` vs `[B, K]`, which crashes if $B \ne K$ and silently scales the wrong axis if $B = K$. The fix is the explicit `sample_weights[:, None]` from Section 3.4, and the test is to assert the weighted tensor still has shape `[B, K]` before reduction.

#### Worked example: the multi-task weight that scrambles tasks

Concretely, take $B = 3$ samples and $K = 3$ tasks, so the per-task loss matrix is `[3, 3]`. Suppose the per-task losses are

$$
L = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix},
$$

where row $i$ is sample $i$ and column $j$ is task $j$. You have per-sample weights `w = [0.1, 1.0, 10.0]` (downweight an easy sample, upweight a hard one) and you write `(w * L)`. Because `w` is `[3]`, it aligns against the *trailing* axis of `L`, so it multiplies *column-wise*: task 0 by 0.1, task 1 by 1.0, task 2 by 10.0. The weighted matrix becomes

$$
w * L = \begin{bmatrix} 0.1 & 2 & 30 \\ 0.4 & 5 & 60 \\ 0.7 & 8 & 90 \end{bmatrix},
$$

which weights *tasks*, not samples — the exact opposite of what you intended. The mean of this is $19.6$. The correct version, `w[:, None] * L`, multiplies *row-wise* (sample 0 by 0.1, sample 1 by 1.0, sample 2 by 10.0), giving

$$
w[:, None] * L = \begin{bmatrix} 0.1 & 0.2 & 0.3 \\ 4 & 5 & 6 \\ 70 & 80 & 90 \end{bmatrix},
$$

with mean $28.4$. Two different scalars — $19.6$ versus $28.4$ — from the same data, both finite, both plausible, and only one correct. No shape changes (both results are `[3, 3]`), so no assert on shape catches it; the only defense is to make the axis explicit with `w[:, None]` or with an einops `reduce` that names the sample axis, and to unit-test the weighted loss against this hand-computed $28.4$. This is bug class 3.4 at its most dangerous: the output shape is right, so only the *values* betray it.

### 11.4 Tabular: the target column loaded as a 2-D frame

This is the running example's real-world origin. Pandas and many data loaders return a single target column as a 2-D structure — `df[["target"]]` gives a `[N, 1]` frame, and `.values` keeps the `[N, 1]` shape. Feed that to a model whose head outputs `[B]` and your MSE broadcasts to `[B, B]`, exactly as Section 5 derived. The signature is precisely the one in our running dashboard: loss falls, $R^2 \approx 0$, predictions collapse to a constant. The fix is a `.view(-1)` or `.squeeze(-1)` at the data boundary, plus the unit test from Section 10. I have seen this exact bug in production tabular pipelines more than once, and the cost is always the same — a run that looks healthy and learns nothing.

### 11.5 Speech: feature and label length mismatches

Speech models add a temporal axis that makes shape bugs especially common. A spectrogram feature is `[B, F, T]` (batch, frequency bins, time frames) or `[B, T, F]` depending on the library, and the two conventions disagree — `torchaudio` and many model implementations expect different orderings, so a feature tensor handed to the wrong model broadcasts or transposes silently. The CTC loss, used for alignment-free transcription, expects log-probabilities of shape `[T, B, V]` (time-major, deliberately, which trips up everyone) and target sequences with explicit lengths; pass `[B, T, V]` and the loss computes over the wrong axis. The signature is a transcription that is flat or garbage despite a falling loss, which is the same objective-mismatch fingerprint as the regression case. The discipline is identical: assert the exact expected shape (including the time-major ordering for CTC) before the loss, and unit-test the loss on a tiny known utterance.

### 11.6 The CV channel-order broadcast

One more vision pattern deserves a name because it broadcasts so cleanly. ImageNet normalization subtracts a per-channel mean and divides by a per-channel std, with mean and std of shape `[3]` (or `[3, 1, 1]` for explicit broadcasting). If you load an image as `[H, W, 3]` (channels-last, the PIL/NumPy default) and apply a `[3]` mean, the broadcast aligns the `[3]` against the *last* axis — which happens to be channels in channels-last layout, so it works. But the moment you convert to channels-first `[3, H, W]` (the PyTorch default) and forget to reshape the mean to `[3, 1, 1]`, the `[3]` mean aligns against the *last* axis again — now `W` — and if `W` happens to be 3 it broadcasts to garbage, and if not it crashes. The fix is to store normalization statistics as `[3, 1, 1]` so the channel axis is unambiguous, and to assert the image is channels-first before normalizing. This is the same size-1-axis discipline from Section 2.4 applied to the most routine preprocessing step in computer vision.

### 11.7 The contrastive-loss similarity-matrix sign flip

A subtler real signature shows up in contrastive and metric-learning losses, where the whole point is to build a `[B, B]` similarity matrix on purpose. You embed a batch into `[B, D]`, normalize, and compute `sim = emb @ emb.T`, a `[B, B]` matrix of pairwise similarities. The InfoNCE loss then treats the diagonal as positives and the off-diagonal as negatives. The shape is *supposed* to be `[B, B]` here, which is what makes the bug so easy to hide: a broadcast that produces an extra `[B, B]` somewhere downstream blends right in. The classic mistake is in the labels — the positive index for row $i$ should be $i$ (the diagonal), so `labels = torch.arange(B)`, shape `[B]`. If a refactor reshapes the labels to `[B, 1]` or builds a `[B, B]` one-hot target and then a stray broadcast aligns it wrong, the loss still computes a finite number and still falls, but it is rewarding the wrong pairs as positives. The signature is a contrastive model whose retrieval accuracy stays at chance while the loss looks healthy — the exact objective-mismatch fingerprint, now hiding inside a loss that legitimately uses `[B, B]` tensors. The defense is to assert `labels.shape == (B,)` and to unit-test that, on a batch where embeddings are set to be identical within known positive pairs, the loss is near its theoretical minimum.

## 12. When this is (and isn't) your bug

Diagnosis is as much about ruling *out* as ruling in. A silent shape bug has a specific fingerprint, and several similar-looking symptoms point elsewhere. Knowing the difference saves you from asserting shapes in the loss when the real problem is in the data.

**It probably IS a silent shape bug when:** the loss falls but plateaus at a high floor; the model predicts a near-constant (low prediction variance) regardless of input; $R^2$ or task accuracy is near chance while the loss looks like it is "learning"; the overfit-one-batch test bottoms out at a suspiciously high value (0.3–0.5) instead of ~1e-4; and — the decisive check — printing the residual or loss-input shape reveals a `[B, B]`, a `[B, 1]` where you expected `[B]`, or a rank you did not intend.

**It probably is NOT a shape bug when:** the loss is *completely flat* from step 0 (that is a frozen model, a zero learning rate, or a detached graph — gradient-flow territory, see `your-model-isnt-learning-what-you-think`); the loss diverges to NaN (that is numerics — `hunting-nans-and-infs`); the loss falls cleanly to a low floor on training data but the *validation* metric is bad (that is overfitting or distribution shift, not a shape bug — the objective is correct, the generalization is not); or the overfit-one-batch test passes cleanly to ~1e-4 (a correct loss can memorize one batch, so if it does, the loss shape is fine and you should stop blaming it).

The cleanest discriminator is the overfit-one-batch test combined with the residual-shape print. If one batch overfits to ~1e-4, your loss is computing the right thing — go look at data or generalization. If one batch *partially* overfits and then sticks, and the residual shape is wider than you expected, you have a silent broadcast. This pairing — make-it-fail-small plus read-the-shape — localizes the bug in two checks. For the full symptom-to-suspect map, the [training-debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) collects every one of these discriminators into a single decision tree.

There is a second discriminator worth keeping for the case where the output shape is *correct* and only the values are wrong (bug class 3.4 — the per-sample weight on the wrong axis, or the contrastive-label mix-up from Section 11.7). The shape print does not help there, because the shapes are right. The test instead is a *symmetry probe*: feed the loss an input where the buggy and correct computations must disagree, and check which one you get. For a per-sample weight, set all sample losses equal but the weights unequal — the correctly weighted mean changes when you permute the *samples* but not when you permute the *features*, and the buggy version does the opposite. One two-line probe tells you which axis the weight is actually hitting. This is the same philosophy as the unit test in Section 10: construct an input where the right and wrong answers are forced apart, then read which one the code produces. When the shape cannot betray the bug, make the *values* betray it.

A third practical note: do not confuse a silent shape bug with a *slow* run. If your loss is falling steadily and the task metric is *also* slowly improving — even if both are slower than you would like — you do not have a broadcast bug, you have an optimization or capacity issue. The broadcast bug's tell is that the task metric is *pinned* at chance (not merely slow) while the loss falls; the objective is not slightly off, it is pointed at a different target entirely. A run that is genuinely learning the right thing, however slowly, has a correct objective. Reserve the shape-assert hunt for the case where the loss and the metric have *decoupled* — the loss moving while the metric refuses to.

There is also a related model-code failure worth distinguishing: an [attention-and-masking bug](/blog/machine-learning/debugging-training/attention-and-masking-bugs), where a mask of the wrong shape broadcasts across the wrong axis of an attention score matrix. That has the same root cause (a silent broadcast) but a different signature — great training loss, poor generalization, or a model that peeks at future tokens — so it gets its own treatment. The shape-assert and einops discipline from this post is exactly the cure there too.

## 13. Key takeaways

- **A shape mismatch is silent precisely when every disagreeing axis can be resolved by stretching a size-1 dimension.** `[B]` vs `[B, 1]` broadcasts to `[B, B]`; `[B]` vs `[C]` (no size-1, $B \ne C$) crashes. Memorize this and you catch half of them by reading the code.
- **The loss is the most dangerous place for a shape bug**, because its output is always a scalar that looks fine. Assert the *residual* shape before reduction, not just the final loss.
- **A loss that falls while the task metric stays at chance is the signature of an objective mismatch**, and the most common cause is a silent broadcast that averages the wrong axis. The optimizer is faithfully minimizing the wrong scalar.
- **A `[B]` vs `[B, 1]` MSE broadcasts to a `[B, B]` matrix and trains the model to predict the batch-mean target**, which it achieves by ignoring the input — predictions collapse to a constant. The gradient analysis predicts exactly this symptom.
- **Assert exact shapes at function boundaries** with a helper that spells out the expected shape and supports a wildcard for free dimensions. It converts a multi-day bug into a one-line assertion error.
- **`einops` rearrange/reduce is a self-documenting shape contract**: naming axes makes a wrong axis a loud error instead of a silent broadcast, and it documents intent in the code.
- **Unit-test the loss on tiny hand-computed inputs.** A five-line test that pins the MSE of `[2,4,6,8]` vs `[1,3,5,7]` to exactly 1.0 makes the silent bug a red CI run instead of a dead GPU run.
- **The overfit-one-batch test is the first discriminator**: a silent broadcast often lets a model *partially* overfit and plateau at a high floor (0.3–0.5) rather than failing outright, so a suspiciously high overfit floor is itself the signal to assert your shapes.
- **Print shapes through the forward pass** with a forward hook when the bug is deeper than the loss. The framework tells you the truth about what each layer produced; reading the trace beats reasoning about the code.

## 14. Further reading

- **PyTorch broadcasting semantics** — the official documentation on broadcasting rules (align trailing dims, size-1 stretches). The authoritative source for the algorithm in Section 2. `pytorch.org/docs/stable/notes/broadcasting.html`
- **einops: Rocktäschel/Rogozhnikov, "einops" documentation** — rearrange/reduce/repeat as named-axis operations; the cleanest way to make shape contracts explicit. `einops.rocks`
- **PyTorch named tensors** — the experimental API for naming axes so mismatched names refuse to align; the long-term fix for the size-1 wildcard problem. `pytorch.org/docs/stable/named_tensor.html`
- **`torch.nn.functional.cross_entropy` documentation** — the exact input-shape contract (`[N, C]` logits, `[N]` targets, and the higher-rank spatial form) that the NLP and segmentation case studies depend on.
- [A taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) — the master decision tree (symptom → the six places a bug hides → confirming test → fix) that this model-code bug sits inside.
- [The overfit-a-single-batch test](/blog/machine-learning/debugging-training/the-overfit-a-single-batch-test) — the make-it-fail-small sanity check that, as shown in Section 7, partially passes under a silent broadcast and flags it via a high loss floor.
- [Loss-function bugs](/blog/machine-learning/debugging-training/loss-function-bugs) — wrong reduction, logits-vs-probabilities, and `ignore_index` mistakes; the sibling post where reshape-and-reduce errors are the central theme.
- [The training-debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) — the capstone that collects every symptom-to-suspect discriminator, including the shape-bug fingerprint, into one printable decision tree.

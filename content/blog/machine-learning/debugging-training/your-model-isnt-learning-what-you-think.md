---
title: "Your Model Isn't Learning What You Think: Gradient-Flow Bugs"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Total loss moves, the dashboard looks alive, and yet one submodule never changes — learn the gradient-flow audit that names the dead parameter in 30 seconds and the one-line fixes that let it finally train."
tags:
  [
    "debugging",
    "model-training",
    "pytorch",
    "gradient-flow",
    "autograd",
    "finetuning",
    "deep-learning",
    "computer-vision",
    "llm",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/your-model-isnt-learning-what-you-think-1.png"
---

The loss was going down, so we shipped the run. Four GPUs, two days, a ResNet-50 backbone being finetuned for a 37-class fine-grained classification task. The training loss slid from 3.61 to 2.9 over the first few thousand steps, the GPUs sat at 94% utilization, and throughput was textbook. Everyone moved on. When validation accuracy came back at 41% — a respectable-sounding number that was, on closer inspection, *exactly* what you get if the new task head is a brick and only the last block of the backbone is doing anything — we went looking. It turned out the classification head had been constructed *after* the optimizer, so it lived in the model but not in any of the optimizer's parameter groups. Its weights never moved. Not once. The backbone learned just enough to push the loss down and make the run look alive, and the head — the one part that had to learn the actual 37 classes — was frozen in its random initialization for the entire run.

This is the most insidious class of training bug, because the instrument you trust most is lying to you. **Total loss is an aggregate.** It can fall steadily while a critical submodule contributes nothing, because the *other* parameters are absorbing the gradient and improving. The loss curve looks like learning. The GPU is busy. There is no crash, no NaN, no stack trace. And yet part of your model is silently not training. By the time you notice — if you notice — you have paid for the compute and lost the days.

![A vertical chain of five stages from a requires-grad leaf through the forward graph, a nonzero gradient, and an optimizer group to a parameter that actually moves, with any broken link freezing the param](/imgs/blogs/your-model-isnt-learning-what-you-think-1.png)

The figure above is the whole post in one picture. For a parameter to learn, it has to complete a chain: it must be a `requires_grad` leaf, it must lie on the path from the input to the loss, backward has to fill it with a nonzero gradient, and it has to sit inside an optimizer parameter group so that `step()` actually moves it. Break *any* link — freeze it by accident, `.detach()` the branch it lives on, construct it after the optimizer, never use its output in the loss — and that parameter quietly stops learning while everything around it carries on. The signature is always the same: **total loss moves, but this submodule's parameters never change; its gradient is `None` or its norm is `0`.**

By the end of this post you will have a reusable **gradient-flow audit**: a few lines that, after one `backward()`, walk `named_parameters()` and tell you exactly which parameters have `grad is None`, which have `grad.norm() == 0`, and which have `requires_grad = False`. You will have a **param-changed check** that snapshots a weight, runs one step, and proves whether it actually moved. You will know how to confirm every trainable parameter is in an optimizer group, how to read `loss.grad_fn` to see whether the graph was even built, and you will be able to take a stalled finetune — vision, LLM, or multi-task — and localize the dead component in under a minute. In the language of this series, gradient-flow bugs live in the **model-code** place of the [bug taxonomy](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs), and they are the textbook reason the [overfit-a-single-batch test](/blog/machine-learning/debugging-training/the-overfit-a-single-batch-test) fails: a model with a severed gradient path *cannot* drive loss to zero, because part of it never updates.

## 1. The symptom: an aggregate that hides a dead component

Let us be precise about what we are hunting, because the symptom is deliberately subtle. The dangerous failure is not the run that crashes — a crash is honest. It is the run that *runs*, consumes resources, moves its headline number, and is nonetheless not learning the thing you built it to learn.

Here is the running example we will debug throughout. A ResNet-50 pretrained on ImageNet, with the final fully-connected layer replaced by a fresh `nn.Linear(2048, 37)` for a fine-grained pets classification task. We finetune the whole thing — backbone plus new head — at a single learning rate. The dashboard, three thousand steps in, reads like this:

| Instrument | Reading | The naive interpretation |
| --- | --- | --- |
| Train loss @ step 3000 | 2.90 (from 3.61) | "It's learning." |
| GPU utilization | 94% | "Compute is healthy." |
| Throughput | 980 img/s | "Data pipeline is fine." |
| Global gradient norm | 4.2 | "Gradients are flowing." |
| Val accuracy @ epoch 5 | 41% | "...lower than the paper, but not chance?" |

Every signal except the last looks healthy, and even the last one is *plausible*. On 37 classes, pure chance is $1/37 \approx 2.7\%$, so 41% is clearly above chance — which is exactly what makes this bug so hard to catch. A model that is doing *some* learning but missing a key component does not collapse to chance; it lands at a frustrating intermediate number that you can rationalize as "needs more epochs" or "the LR is a touch low." The global gradient norm of 4.2 is the cruelest signal of all: it is a *sum over all parameters*, so the backbone's gradients alone keep it comfortably nonzero even when the head receives nothing.

The junior reflex is to turn knobs: raise the learning rate, add epochs, try a different scheduler. None of those touch the actual problem. The disciplined move is to stop trusting the aggregate and **interrogate the parameters individually**. Does *every* parameter that should be learning actually have a gradient? Does each one actually *move* across a step? That is a five-line question, and it cracks this case open immediately.

There is a deeper reason this bug class is so under-diagnosed: the tooling we reach for first is built around the aggregate. Loss curves, the global gradient norm that gets logged by default, accuracy on the val set — all of them are reductions over the whole model. None of them have the *resolution* to see that one tensor among a few hundred is dead. The shift in mindset this post is trying to install is to log and read *per-parameter* (or at least per-module) signals, because the bug lives at that resolution and is invisible above it. A model is not one thing that is "learning" or "not learning"; it is a few hundred tensors, each of which is independently either on the path to the loss or not, in the optimizer or not, moving or not. Once you see it that way, "is my model learning?" decomposes into a question you can answer mechanically for every parameter.

It also helps to know *why* the symptom lands where it does. A frozen head on a classification task does not produce chance-level accuracy; it produces a *capped* accuracy. The backbone keeps improving its features to be as separable as possible *under a fixed random linear projection*, and a random projection of good features is a surprisingly non-terrible classifier — often landing somewhere between chance and the real ceiling. That intermediate plateau is the tell. A run that is genuinely at chance screams "something is broken." A run stuck at a *plausible-but-disappointing* number whispers it, and that whisper is what costs people days.

### 1.1 Why "the loss went down" is not proof of learning

It is worth internalizing *why* a falling loss does not certify that any particular parameter is learning. Cross-entropy loss for a classifier is

$$
\mathcal{L} = -\frac{1}{B}\sum_{i=1}^{B} \log p_{y_i}, \qquad p = \mathrm{softmax}(z), \qquad z = W_{\text{head}} \, f_\theta(x) + b,
$$

where $f_\theta(x)$ is the backbone feature and $W_{\text{head}}$ is the head. The loss depends on *both* the backbone $\theta$ and the head $W_{\text{head}}$. Gradient descent will happily reduce $\mathcal{L}$ by improving the backbone features alone — making them more linearly separable under the *frozen, random* head — even if $W_{\text{head}}$ never changes. The features shift to accommodate a fixed random projection, and that is enough to move the loss a little. It is a smaller, capped improvement (a random head is a bad classifier), but it is real, and it is enough to fool the dashboard. The loss is the *sum* of contributions from every parameter; a falling sum tells you *something* is improving, never *what*.

The global gradient norm deserves the same skepticism, and for the same reason. What most training loops log is

$$
\|g\|_2 = \sqrt{\sum_{p} \|\nabla_p \mathcal{L}\|_2^2},
$$

a single number aggregating the gradient of every parameter. If the backbone has tens of millions of parameters each contributing a small gradient, and the head has a few thousand contributing *zero*, the sum is dominated entirely by the backbone. A head gradient of exactly zero changes $\|g\|_2$ by an amount too small to notice against the backbone's contribution. So the global gradient norm — the instrument people reach for to answer "are gradients flowing?" — is structurally incapable of detecting that *one* component's gradient is dead. It answers "are *most* gradients flowing?", which is a different and much less useful question. This is the mathematical reason the audit has to be *per-parameter*: any reduction over the model can be carried by the majority while a critical minority is silently zero.

## 2. The science: how autograd decides what gets a gradient

To debug gradient flow you have to know, mechanically, how PyTorch decides which tensors get gradients. It is not magic and it is not global; it is a set of local rules applied as the forward pass runs. Master these and most gradient-flow bugs become obvious on sight.

### 2.1 The graph is built on the forward pass, leaf by leaf

When you run a forward pass, every operation on a tensor that `requires_grad` records a node in a **dynamic computation graph**. Each resulting tensor gets a `grad_fn` — a backward function that knows how to compute the gradient of its inputs given the gradient of its output. A tensor with `requires_grad = True` and no `grad_fn` is a **leaf**: it is a parameter or an input you created directly, and it is the kind of tensor that *accumulates* a gradient into `.grad`. The rule that propagates `requires_grad` is simple and worth memorizing: an operation's output requires grad **if any of its inputs requires grad.** So a single `requires_grad = True` parameter anywhere upstream is enough to make the whole downstream chain part of the graph. Conversely, if *every* input to a subgraph has `requires_grad = False`, that entire subgraph is built without grad-tracking and contributes nothing to backward.

When you call `loss.backward()`, autograd starts at the loss (a scalar) and walks the graph *backward* along the recorded edges, applying the chain rule. It deposits the accumulated gradient into the `.grad` field of each leaf it can reach. The critical phrase is *each leaf it can reach.* If a leaf is not on any path from the loss — because the path was never recorded, or was cut — backward simply never visits it, and its `.grad` stays at whatever it was: `None` if it has never received a gradient.

The word *accumulates* in "accumulates a gradient into `.grad`" is load-bearing and worth a beat, because it is the reason `optimizer.zero_grad()` exists and the source of a related bug. PyTorch does not *overwrite* `.grad` on each backward; it *adds* to it. This is deliberate — it lets you accumulate gradients across several backward passes to simulate a larger batch — but it means that if you forget to zero the gradients between steps, every step's gradient piles on top of the last, and your effective update is the running sum of all gradients so far. That is not a gradient-flow bug in the "path is cut" sense, but it lives in the same family of "the gradient is not what you think it is" surprises, and it has its own signature: a gradient norm that *grows* monotonically across steps instead of fluctuating. The flip side — calling `zero_grad(set_to_none=True)`, the modern default — sets `.grad` back to `None` rather than to a zero tensor, which is slightly faster and is why a freshly-zeroed parameter shows `grad is None` *before* the next backward. The audit is meant to run *after* a backward, when `None` unambiguously means "no gradient arrived," not "I just zeroed it."

So there are exactly three ways a parameter ends up with no useful gradient, and they map one-to-one onto the bugs in this post:

1. **It is not a `requires_grad` leaf at all.** `requires_grad = False`, so autograd does not even track operations that consume it. Its `.grad` is `None` and stays `None`. (Frozen on purpose, or frozen by accident.)
2. **It is a `requires_grad` leaf, but no path from the loss reaches it.** The path was severed by `.detach()`, `.data`, `torch.no_grad()`, a numpy round-trip, or the submodule's output simply never enters the loss. Its `.grad` is `None`.
3. **It is on a path and gets a gradient, but the gradient is zero**, or the gradient never gets turned into a weight update because the parameter is not in the optimizer. The `.grad` exists (or is nonzero) but `step()` does nothing for it.

### 2.2 `requires_grad`, `is_leaf`, and `grad_fn` — the three flags you read

Three attributes tell you almost everything about a tensor's role in the graph. `requires_grad` says whether autograd tracks it. `is_leaf` says whether it is a parameter/input (a place gradients accumulate) versus an intermediate result. `grad_fn` is `None` for a leaf and set for an intermediate. The combination is diagnostic:

| `requires_grad` | `grad_fn` | `is_leaf` | What it is |
| --- | --- | --- | --- |
| `True` | `None` | `True` | A trainable parameter (a leaf). Should receive `.grad` after backward. |
| `True` | set | `False` | An intermediate result on the graph. `.grad` not populated by default. |
| `False` | `None` | `True` | A frozen parameter or a constant input. Never receives `.grad`. |
| `False` | `None` | `True` (after `.detach()`) | A tensor cut out of the graph. Everything upstream of it via this tensor is invisible to backward. |

The single most useful sanity check at the top of any debugging session is to print `loss.requires_grad` and `loss.grad_fn`. If `loss.grad_fn` is `None`, your loss is not even connected to the graph — you computed it under `torch.no_grad()`, or you turned a tensor into a Python float somewhere, and `backward()` will raise or do nothing. If `loss.grad_fn` is set, the graph exists; now the question is *which leaves it reaches.*

### 2.3 Where the path gets cut

The second class of bug — a `requires_grad` leaf that backward never reaches — is worth a picture, because it is the one people misdiagnose most. Consider a two-headed model: a backbone whose features feed both a classification head and an auxiliary head, with one of the branches accidentally detached.

![A dataflow graph where input feeds a backbone and a head, a detach on the backbone branch cuts the graph, and the backbone gradient is None while the head and loss still receive gradients](/imgs/blogs/your-model-isnt-learning-what-you-think-2.png)

The figure shows the mechanism precisely. Autograd only sends gradient back along edges it *recorded*. The `.detach()` call returns a new tensor that shares storage with the original but has `requires_grad = False` and no `grad_fn` — it is a fresh leaf, a dead end for backward. Everything *upstream* of that detach, reached only through the detached tensor, becomes invisible: backward arrives at the detach, finds no recorded edge continuing backward, and stops. The loss still has a `grad_fn`, the head still gets gradients, `backward()` runs without error — and the backbone's `.grad` is `None`. There is no warning. The graph is doing exactly what you told it to; you just told it to cut the wire.

The same severing happens with `.data` (the legacy way to grab a tensor's storage without grad-tracking — never use it in a forward path), with anything inside a `with torch.no_grad():` block, with `.item()` / `.tolist()` / `float()` that escape to Python, and with a numpy round-trip (`x.cpu().numpy()` then `torch.from_numpy(...)` — the new tensor is a fresh leaf with no history). In every case the symptom is identical: a `requires_grad` parameter upstream of the cut has `.grad is None` after backward.

### 2.4 A traced example: watching the graph break

It helps to see the cut happen on a tiny example you can run in a notebook, because the `grad_fn` chain is *visible* if you look. Build two parameters, run a forward, and inspect the chain:

```python
import torch

w1 = torch.randn(3, 3, requires_grad=True)   # a leaf: requires_grad True, grad_fn None
w2 = torch.randn(3, 3, requires_grad=True)
x  = torch.randn(3, 3)

h = x @ w1            # intermediate: grad_fn = MmBackward0, is_leaf False
y = h @ w2            # intermediate: grad_fn = MmBackward0
loss = y.sum()        # scalar: grad_fn = SumBackward0

print(w1.is_leaf, w1.grad_fn)   # True  None    -> a parameter
print(h.is_leaf,  h.grad_fn)    # False <MmBackward0>  -> on the graph
print(loss.grad_fn)             # <SumBackward0>  -> graph reaches the loss

loss.backward()
print(w1.grad is not None, w2.grad is not None)   # True True -> both reached
```

Both parameters receive a gradient because there is an unbroken `grad_fn` chain from `loss` back to each leaf. Now insert a cut and rerun the relevant lines:

```python
w1.grad = None; w2.grad = None
h = x @ w1
h = h.detach()        # CUT: h is now a fresh leaf, requires_grad False, grad_fn None
y = h @ w2
loss = y.sum()
loss.backward()
print(w1.grad)        # None  -> backward stopped at the detach, never reached w1
print(w2.grad is not None)   # True -> w2 is downstream of the cut, still fine
```

The cut is surgical: `w1`, which is *upstream* of the detach, gets `None`; `w2`, which is *downstream*, still gets a gradient. This is the entire mechanism behind every "this submodule won't learn" bug. The detach (or `.data`, or the numpy round-trip) does not raise, does not warn, and leaves the loss perfectly differentiable with respect to everything downstream of the cut. The only evidence is the `None` on the upstream leaf — which is exactly what the gradient-flow audit prints.

One subtlety worth keeping: `requires_grad` propagates *forward* (an output requires grad if any input does), but the gradient flows *backward* along the recorded `grad_fn` chain. A detach breaks the backward chain without changing the forward values at all, which is why your forward pass produces identical numbers and your model still "runs." The forward is fine; the backward is blind past the cut. That asymmetry — forward unaffected, backward severed — is what makes the bug survive every check that only looks at outputs.

### 2.5 Why `no_grad` and `detach` exist (and when they are correct)

It would be easy to read this section as "never use `detach` or `no_grad`," which is wrong — they are essential tools, and the bug is using them in the *wrong place*, not using them at all. `torch.no_grad()` is correct and necessary around evaluation, around metric computation, and around manual parameter updates (you do not want autograd tracking the optimizer's own arithmetic). `.detach()` is correct when you genuinely want to stop a gradient: detaching a target in a self-distillation loss so the student does not train the teacher, detaching the value baseline in some RL losses, detaching a tensor you are only logging. The skill is knowing that each of these is a *deliberate* gradient cut and confirming, with the audit, that you cut exactly the branch you meant to. A `detach` in a forward path that feeds the loss for parameters you *want* to train is the bug; the identical call around a target you *do not* want to train is correct engineering. The instrument does not judge intent — it just tells you which leaves got `None`, and you decide whether that matches what you wanted.

## 3. The diagnostic: the gradient-flow audit

Enough theory. Here is the tool. After one forward and one `backward()`, walk every parameter and report its gradient status. This is the single highest-value snippet in this post; paste it into any project and run it once.

```python
import torch

def grad_flow_audit(model, verbose=True):
    """After loss.backward(), report each parameter's gradient health.

    Categories:
      - frozen:      requires_grad is False (intentional or accidental)
      - no_grad:     requires_grad True but .grad is None (path cut / unused)
      - zero_grad:   .grad exists but its L2 norm is ~0 (dead path / dead neuron)
      - flowing:     .grad exists with a nonzero norm (healthy)
    """
    report = {"frozen": [], "no_grad": [], "zero_grad": [], "flowing": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            report["frozen"].append(name)
        elif p.grad is None:
            report["no_grad"].append(name)
        else:
            gnorm = p.grad.norm().item()
            if gnorm < 1e-12:
                report["zero_grad"].append((name, gnorm))
            else:
                report["flowing"].append((name, gnorm))

    if verbose:
        print(f"flowing : {len(report['flowing'])} params")
        print(f"frozen  : {len(report['frozen'])} params  {report['frozen'][:6]}")
        print(f"NO GRAD : {len(report['no_grad'])} params  {report['no_grad'][:6]}")
        print(f"ZERO    : {len(report['zero_grad'])} params  {[n for n,_ in report['zero_grad'][:6]]}")
        # The smoking gun: requires_grad True but no gradient arrived.
        for name in report["no_grad"]:
            print(f"  !! {name}: requires_grad=True but grad is None "
                  f"-> not on the path from the loss")
    return report
```

Run it after a single training step:

```python
model.train()
xb, yb = next(iter(train_loader))
optimizer.zero_grad(set_to_none=True)
loss = criterion(model(xb), yb)
loss.backward()                  # populate .grad
report = grad_flow_audit(model)  # read the instruments
```

On our broken pets run, the output is unambiguous:

```bash
flowing : 159 params
frozen  : 0 params  []
NO GRAD : 2 params  ['fc.weight', 'fc.bias']
ZERO    : 0 params  []
  !! fc.weight: requires_grad=True but grad is None -> not on the path from the loss
  !! fc.bias:   requires_grad=True but grad is None -> not on the path from the loss
```

The head (`fc.weight`, `fc.bias`) has `requires_grad = True` but `grad is None`. That single line rules out "the LR is too low" and "needs more epochs" instantly: the head is not on the path from the loss at all, or — as we will see — it is on the path but absent from the optimizer. Either way, you have gone from a vague "it's underperforming" to a named, two-parameter defect in five lines.

The four-way classification is the entire diagnostic surface, and reading which bucket a stalled parameter falls into routes you to the cause. The figure below crosses the two flags you read — `requires_grad` against gradient status — into the grid that sorts every parameter into healthy, frozen-on-purpose, or the one dead cell that is the bug.

![A three-by-three grid crossing requires-grad against gradient status, placing healthy params, a frozen-on-purpose cell, and the dead requires-grad-yet-grad-None cell that is the bug](/imgs/blogs/your-model-isnt-learning-what-you-think-7.png)

### 3.1 Confirm the parameter actually *moves*

`grad is None` is the clearest signal, but it is not the only failure mode. A parameter can have a perfectly good gradient and *still* never update — if it is not in the optimizer. The gradient flows, `.grad.norm()` is healthy, the audit says "flowing," and yet `step()` skips it because it is not in any `param_group`. To catch that, you need a second instrument that measures the *update*, not the gradient: snapshot the weight, run one optimizer step, and check whether it changed.

```python
import copy
import torch

@torch.no_grad()
def snapshot_params(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

@torch.no_grad()
def report_param_changes(model, before, atol=0.0):
    """Compare current params to a pre-step snapshot. delta==0 on a
    grad-bearing param means the optimizer never touched it."""
    print(f"{'param':40s} {'delta_norm':>12s} {'grad_norm':>12s}")
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        delta = (p - before[n]).norm().item()
        gnorm = p.grad.norm().item() if p.grad is not None else float("nan")
        flag = "  <-- DID NOT MOVE" if delta <= atol else ""
        print(f"{n:40s} {delta:12.3e} {gnorm:12.3e}{flag}")

# usage: bracket exactly one step
before = snapshot_params(model)
optimizer.zero_grad(set_to_none=True)
loss = criterion(model(xb), yb)
loss.backward()
optimizer.step()
report_param_changes(model, before)
```

The figure below brackets that single step from snapshot to comparison: snapshot the weight, run forward-backward-step, and compare. A nonzero delta proves the parameter learned; a zero delta on a parameter that *has* a gradient is the precise fingerprint of "the gradient flows but the optimizer skipped it."

![A left-to-right timeline of one optimizer step: snapshot the weight, forward and backward fill the gradient, step should move it, then a moved-versus-stuck comparison with a zero delta exposing a param missing from the optimizer](/imgs/blogs/your-model-isnt-learning-what-you-think-6.png)

This `delta == 0 with grad_norm > 0` case is the one the gradient-flow audit alone cannot see — which is why you run *both* checks. The audit catches "no gradient arrived"; the param-changed check catches "a gradient arrived but no update happened."

#### Worked example: the head built after the optimizer

Here is exactly how our pets bug arises in code, and exactly how the two instruments expose it. The original training script looked like this:

```python
model = torchvision.models.resnet50(weights="IMAGENET1K_V2")
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)   # (!) built here
model.fc = torch.nn.Linear(2048, 37)                          # head replaced AFTER
```

The bug is the order. `optimizer = AdamW(model.parameters(), ...)` captures the parameter objects that exist *at that moment* — including the original ImageNet `fc` with 1000 outputs. Then `model.fc = nn.Linear(2048, 37)` replaces the head with a *new* parameter object. The new `fc` has `requires_grad = True`, it *is* used in the loss, so it *does* get a gradient — but the optimizer is holding references to the old, now-orphaned head parameters. The new head is in the model but not in `optimizer.param_groups`. Run the two instruments and you see both signatures at once. The audit, if you call it before `step()`, shows `fc.weight` *flowing* (it gets a gradient), which is confusing until the param-changed check shows:

```bash
param                                      delta_norm    grad_norm
layer4.2.conv3.weight                       3.110e-03    1.840e+00
fc.weight                                   0.000e+00    9.210e-01  <-- DID NOT MOVE
fc.bias                                     0.000e+00    4.130e-01  <-- DID NOT MOVE
```

`fc` has a healthy gradient (`9.2e-01`) and a *zero* update. That is the unmistakable fingerprint of a parameter that is in the graph but not in the optimizer. The fix is to build the optimizer *after* the model is fully constructed:

```python
model = torchvision.models.resnet50(weights="IMAGENET1K_V2")
model.fc = torch.nn.Linear(2048, 37)                          # finish the model FIRST
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)   # then capture all params
```

We will look at the before→after numbers in Section 5. For now, note the discipline: *the optimizer must be the last thing you build, after every parameter the model will ever train already exists.*

### 3.2 Confirm every trainable parameter is in an optimizer group

Because "in the model but not in the optimizer" is such a common and silent bug, it deserves a dedicated assertion you can run once at startup — before you waste a single GPU-hour. The optimizer's parameter groups hold the exact tensor objects it will update; compare their identities (`id()`) against the model's trainable parameters.

```python
def assert_optimizer_covers_model(model, optimizer):
    """Every requires_grad param in the model must be in an optimizer group."""
    optim_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    missing = [(n, tuple(p.shape)) for n, p in model.named_parameters()
               if p.requires_grad and id(p) not in optim_ids]
    extra = sum(1 for pid in optim_ids
                if pid not in {id(p) for p in model.parameters()})
    if missing:
        print(f"MISSING from optimizer ({len(missing)} params):")
        for n, s in missing:
            print(f"  {n} {s}")
        raise AssertionError("trainable params not in any optimizer group")
    if extra:
        print(f"warning: optimizer holds {extra} params no longer in the model")
    print(f"OK: all {len(optim_ids)} optimizer params present; "
          f"{sum(p.requires_grad for p in model.parameters())} trainable")
```

On the buggy script this prints `MISSING from optimizer (2 params): fc.weight (37, 2048), fc.bias (37,)` and raises before training even starts. That assertion, run once, would have saved the original two-day run. It is cheap insurance: add it to your training harness right after you build the optimizer, alongside `print_trainable_parameters()` for PEFT models.

Two practical notes on this check. First, it compares by `id()`, not by name or shape, because that is the only thing that is actually true: the optimizer holds *references to specific tensor objects*, and a step updates those exact objects. A new `nn.Linear` with the same name and shape is a *different object* with a *different* `id`, which is precisely why "replace the head after building the optimizer" strands it — the names match, the shapes match, but the identity does not. Second, the `extra` count is worth logging even when nothing is missing: an optimizer holding parameters that are *no longer in the model* (the orphaned old head, in our case) is itself a smell. Those orphans get gradients of `None` every step, so most optimizers skip them harmlessly, but their presence tells you the model was mutated after the optimizer was built — exactly the situation that strands the new head.

#### Worked example: the optimizer built before unfreezing

The pets bug in Section 5 was "head built after the optimizer." Here is its close cousin, which bites people running a two-stage finetune: the optimizer built *before* unfreezing the backbone. The code reads like a careful, staged transfer-learning recipe:

```python
for p in model.parameters():
    p.requires_grad_(False)          # freeze the whole backbone
model.fc = nn.Linear(2048, 37)       # train only the head in stage 1
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)   # (!) built now

train(model, optimizer, stage1_loader, epochs=3)   # stage 1: head learns

for p in model.parameters():
    p.requires_grad_(True)           # stage 2: unfreeze backbone... or so we think
train(model, optimizer, stage2_loader, epochs=10)  # stage 2: nothing new learns
```

Stage 1 works: only the head is trainable, the optimizer was built over `model.parameters()`, and since the optimizer captures *all* parameter objects regardless of their `requires_grad` flag at construction time, the head is in a group and updates. The trap is stage 2. Flipping `requires_grad_(True)` on the backbone makes those parameters *receive gradients* again — but the optimizer's parameter groups were fixed at construction, and the AdamW state (the first and second moment buffers) only exists for the parameters it captured. In recent PyTorch the backbone parameters *are* in the optimizer (they were captured at build time, just with `requires_grad=False`), so they will start updating once unfrozen — but their optimizer state was initialized while frozen, and more importantly people who build the optimizer as `[p for p in model.parameters() if p.requires_grad]` (a common idiom to "only optimize trainable params") capture *only the head*, and then stage 2 genuinely updates nothing in the backbone. Run the param-changed check at the start of stage 2:

```bash
param                                      delta_norm    grad_norm
fc.weight                                   2.700e-03    8.100e-01
layer4.2.conv3.weight                       0.000e+00    1.510e+00  <-- DID NOT MOVE
layer3.5.conv2.weight                       0.000e+00    9.400e-01  <-- DID NOT MOVE
```

The backbone has healthy gradients (1.51, 0.94) and zero updates — the unmistakable "in the graph, not in the optimizer" fingerprint, exactly as in the pets case but caused by the optimizer-build idiom rather than head replacement. The fix is to *rebuild* the optimizer after unfreezing (Section 7.1 shows the two-LR-group version), and to run `assert_optimizer_covers_model` after the unfreeze so the gap fails loudly at the start of stage 2 instead of silently wasting ten epochs. The general rule this example reinforces: **any time you change which parameters are trainable, rebuild and re-assert the optimizer.** The flag and the optimizer are two separate facts, and changing one does not change the other.

### 3.3 Read `loss.grad_fn` and use `detect_anomaly` for the deep cuts

The two instruments above catch the common cases. For the subtle "the path is cut three modules deep inside the forward" case, two more tools earn their keep. First, the cheapest possible check: `print(loss.grad_fn)`. If it is `None`, the loss is not on the graph at all — you wrapped the forward in `torch.no_grad()`, or the loss came out of a `.item()`. If it is set, the graph exists and the cut is *inside* it. Second, for finding *where* inside, register backward hooks on intermediate tensors to see how far the gradient travels:

```python
# Trace how far backward gets: print the grad norm reaching each named module output.
def trace_backward(model, sample_modules):
    handles = []
    def make_hook(name):
        def hook(module, grad_input, grad_output):
            g = grad_output[0]
            norm = g.norm().item() if g is not None else None
            print(f"backward reached {name:30s} grad_out_norm={norm}")
        return hook
    for name, m in model.named_modules():
        if name in sample_modules:
            handles.append(m.register_full_backward_hook(make_hook(name)))
    return handles  # remember to handle.remove() afterwards
```

If `backward reached head` prints but `backward reached backbone` does not, the cut is between them — exactly the `.detach()` in Figure 2. `torch.autograd.set_detect_anomaly(True)` complements this: it does not directly flag a cut graph, but it raises with a stack trace the moment a NaN gradient or an in-place-corrupted tensor breaks backward, which is the other family of "the graph is wrong" bugs we cover in Section 6.

### 3.4 The per-module grad-norm table: a continuous instrument

The audit answers a yes/no question (did a gradient arrive?) at a single point in time. The richer instrument, the one you want logging *every* step of a real run, is the per-module gradient norm — a small table that shows the magnitude of gradient reaching each named module. It catches gradient-flow bugs (a module at exactly `0.0` while its neighbors are healthy) *and* the magnitude bugs they are often confused with (gradients that shrink ten-fold per layer toward the input — vanishing — versus a hard cut to zero). Here is a compact version you can log to TensorBoard or W&B:

```python
def per_module_grad_norms(model):
    """Sum of per-parameter grad norms, grouped by top-level module.
    A module at 0.0 while siblings are nonzero is a gradient-flow cut;
    a smooth decay toward early layers is vanishing, not a cut."""
    from collections import defaultdict
    norms = defaultdict(float)
    counts = defaultdict(int)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        top = name.split(".")[0]            # group by the first path component
        g = p.grad.norm().item() if p.grad is not None else 0.0
        norms[top] += g
        counts[top] += 1
    for mod in sorted(norms):
        print(f"{mod:20s} grad_norm_sum={norms[mod]:.3e}  ({counts[mod]} params)")
    return dict(norms)
```

On a healthy finetune this prints a row of comfortably-nonzero numbers; on our broken pets run it prints `fc  grad_norm_sum=0.000e+00  (2 params)` next to a backbone row reading `layer4  grad_norm_sum=1.840e+01`, and the contrast — one module at exactly zero while every other module is alive — is the visual signature of a *cut*, not a decay. The distinction matters because the fixes are completely different: a cut is graph surgery (re-enable grad, remove the detach), while a decay is an initialization/normalization problem. This table is the continuous companion to the one-shot audit; the deeper treatment of what to log every step lives in [instrumenting a training run](/blog/machine-learning/debugging-training/instrumenting-a-training-run-what-to-log).

A note on `register_full_backward_hook` versus the older `register_backward_hook`: use the *full* variant. The legacy hook had subtle, documented correctness issues with modules that have multiple inputs or in-place operations, and PyTorch deprecated it in favor of `register_full_backward_hook`, which reports the true gradients with respect to the module's inputs and outputs. When you are debugging gradient flow, the *last* thing you want is an instrument that itself reports the wrong gradient, so reach for the full-backward hook every time.

## 4. Bisecting a stalled run: the decision tree

You have the symptom (a submodule that does not learn while total loss falls) and the instruments (the audit, the param-changed check, the optimizer-coverage assert). Now the discipline: **bisect** before you fix. A stalled parameter splits three ways, and which way it splits tells you which broken link to repair.

![A decision tree branching from a non-moving parameter into grad-is-None, grad-norm-zero, and grad-present-but-no-update, each leading to its concrete cause and confirming check](/imgs/blogs/your-model-isnt-learning-what-you-think-5.png)

The tree above is the bisection. A parameter whose weight delta is zero is the root symptom. Run the audit and read which branch it falls into:

- **`grad is None`** — no path filled it in. Either it is not a `requires_grad` leaf (frozen, accidentally or otherwise), or its branch was cut (`.detach()`, `.data`, `no_grad`, numpy round-trip), or its output is simply never used in the loss term. Confirm by printing `p.requires_grad` (catches the freeze) and grepping the forward for `detach`/`numpy` (catches the cut).
- **`grad.norm() == 0`** — the path exists and a gradient arrived, but it is identically zero. This is rarely a wiring bug and usually a *learning* bug: a dead ReLU (every activation is negative, so the gradient is zero everywhere), a saturated sigmoid/tanh, or an input feature that is constant. The fix lives in initialization and learning rate, covered in [dead neurons and saturated activations](/blog/machine-learning/debugging-training/dead-neurons-and-saturated-activations), not in the graph.
- **`grad.norm() > 0` but `delta == 0`** — the gradient is fine; the *update* never happens. The parameter is not in `optimizer.param_groups` (built after the optimizer), or its learning rate in that group is zero, or you forgot to call `optimizer.step()`. Confirm with `assert_optimizer_covers_model`.

This three-way split is the whole game. It converts a fuzzy "my model underperforms" into "this parameter is in branch X, so the fix is Y." It is the model-code branch of the [bug taxonomy](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) decision tree, instantiated for gradient flow.

### 4.1 The matrix: break-cause × signature × check × fix

For the `grad is None` and `delta == 0` branches — the wiring bugs, as opposed to the dead-neuron learning bug — each cause leaves a distinct fingerprint. Memorize this table and you can name most gradient-flow bugs from the audit output alone.

![A matrix mapping five gradient-flow break causes to their audit signature, a confirming check, and the one-line fix for each](/imgs/blogs/your-model-isnt-learning-what-you-think-4.png)

| Break cause | Audit signature | Confirming check | Fix |
| --- | --- | --- | --- |
| `requires_grad = False` (frozen by accident) | `grad is None`, `requires_grad False`, is a leaf | `print(p.requires_grad)` for `named_parameters()` | `p.requires_grad_(True)` *before* building the optimizer |
| `.detach()` / `.data` / `no_grad()` in forward path | `grad is None` upstream of the cut, loss still has `grad_fn` | grep forward for `detach`, `.data`, `no_grad`, `.item` | remove the cut; use `.clone()` if you only needed a copy |
| Param created *after* the optimizer | `grad.norm() > 0`, `delta == 0` | `assert_optimizer_covers_model` finds it missing | build optimizer after the model, or `optimizer.add_param_group` |
| Submodule output never used in the loss | `grad is None`, `requires_grad True`, not on path | does the module's output appear in the loss expression? | add its output to the loss, or delete the dead branch (and `find_unused_parameters` for DDP) |
| numpy/CPU round-trip in forward | `grad is None`, graph reset at the round-trip | grep for `.numpy()`, `.cpu().tolist()`, `np.` on tensors | keep the forward in torch ops end to end |
| Reassigning a tensor instead of updating in place under `no_grad` | param does not move; new tensor not tracked by optimizer | check you used `p.data.copy_` / `with torch.no_grad(): p -= ...` not `p = p - ...` | mutate the existing parameter, do not rebind the name |

The last row deserves a word, because it is a subtle one people hit when they write a manual update loop. If you do `p = p - lr * p.grad` inside a custom step, you have *rebound the Python name* `p` to a brand-new tensor; the original parameter (still referenced by the model and the optimizer) is untouched, and your "update" is thrown away on the next iteration. The correct in-place mutation is `with torch.no_grad(): p -= lr * p.grad` (or `p.data.copy_(...)`), which modifies the existing tensor the optimizer and model both point at. Reassignment versus mutation is the same trap that bites people who think `model.fc = new_layer` *after* building the optimizer registers the new layer — it does not; it rebinds an attribute the optimizer never sees.

## 5. The before→after evidence

A diagnostic is only worth as much as the fix it leads to, and a fix is only credible with numbers. Here is the full before→after for the pets run, measured the way you would actually measure it: re-enable the head into the optimizer, rerun, and read the same instruments.

![A two-column before-after: before, the head is not in the optimizer with grad None and val at chance; after, the head is re-registered with a healthy gradient and val reaches a real number](/imgs/blogs/your-model-isnt-learning-what-you-think-3.png)

The figure summarizes the flip; the table gives the receipts:

| Instrument | Before (head not in optimizer) | After (head re-registered) | How measured |
| --- | --- | --- | --- |
| `fc.weight` grad norm | 0.92 (flowing) | 0.92 (flowing) | `grad_flow_audit` — unchanged; gradient was never the problem |
| `fc.weight` weight delta / step | 0.0 | 4.1e-3 | `report_param_changes` over one step |
| Optimizer coverage | 2 params missing | all params present | `assert_optimizer_covers_model` |
| Train loss @ step 3000 | 2.90 | 0.74 | same loader, same LR, same steps |
| Val accuracy @ epoch 5 | 41% | 89% | same val split, same eval code |

The decisive line is the second one: the gradient norm was *identical* before and after (0.92 both times), which is exactly why the gradient-flow audit alone would have *missed* this bug and called the head "flowing." Only the param-changed check — weight delta 0.0 → 4.1e-3 — exposed it. That is the lesson worth keeping: **run both instruments.** The audit catches "no gradient"; the delta check catches "gradient but no update."

#### Worked example: the LLM finetune with frozen embeddings

Now the same bug class in a different modality, to show the signature is universal. We are SFT-finetuning a small decoder-only language model with Hugging Face `transformers`. Someone, optimizing for memory, froze the embedding matrix to "save it from training" — a reasonable-sounding instinct — but the model *ties* its input embeddings and output (`lm_head`) weights. Freezing the embedding therefore also froze the output projection, and the model could no longer adjust *which token it predicts*, only the internal representations. The loss fell (the transformer blocks adapted) and then plateaued well above where it should have, with the model producing fluent-but-off-task text.

The freeze code:

```python
model = AutoModelForCausalLM.from_pretrained("...")
for p in model.get_input_embeddings().parameters():
    p.requires_grad_(False)   # "save memory" -> also freezes the tied lm_head
```

The audit makes the tie visible:

```bash
NO GRAD : 0 params  []
frozen  : 2 params  ['model.embed_tokens.weight', 'lm_head.weight']
```

Two frozen tensors, and the second one — `lm_head.weight` — is the one nobody meant to freeze. Because the weights are tied, `lm_head.weight` *is* `embed_tokens.weight` (the same storage), so freezing one freezes both. The fix is to unfreeze the embedding (accepting the memory cost) or, if you truly want frozen embeddings, to untie the head first. The before→after, measured on a held-out instruction set:

| Instrument | Before (tied weights frozen) | After (embeddings unfrozen) | How measured |
| --- | --- | --- | --- |
| Trainable params | 86% of total | 100% of total | `print_trainable_parameters()` |
| `lm_head.weight` in audit | `frozen` | `flowing`, norm 0.6 | `grad_flow_audit` |
| Eval loss (cross-entropy) | 1.81 | 1.34 | same eval split, same steps |
| Instruction-following score | 0.42 | 0.71 | held-out rubric, same judge |

The number that matters is the trainable-params line: 86% sounds like "almost everything is training," and that 14% is the most important 14% in a language model. Freezing the embeddings of a tied model is a textbook accidental-freeze, and the audit names it in one line.

## 6. The other graph corruptions: in-place ops and reassignment

So far the bugs have *cut* the graph (no gradient) or *bypassed* the optimizer (no update). There is a third family that *corrupts* the graph: in-place operations that overwrite a tensor autograd still needs for the backward pass. These are different because they often *do* raise — but with a cryptic message — and when they do not raise, they compute a *wrong* gradient silently.

### 6.1 In-place operations that break backward

Autograd's backward functions sometimes need the *value* of a forward tensor to compute gradients. For example, the backward of $y = \exp(x)$ needs $y$ itself, and the backward of many ops needs an input. If you overwrite that tensor in place after the forward pass, backward computes against corrupted data. PyTorch tracks a version counter on each tensor; if a tensor was modified in place after being saved for backward, `backward()` raises:

```bash
RuntimeError: one of the variables needed for gradient computation has been
modified by an inplace operation: [torch.FloatTensor [64, 128]], which is output 0
of ReluBackward0, is at version 2; expected version 1 instead.
```

The classic offenders are `x += y` (use `x = x + y`), `tensor[mask] = 0` on a graph tensor, `relu_(x)` / `add_(...)` and friends with the trailing underscore, and `x.clamp_(...)`. The fix is almost always to use the out-of-place version. To *find* which op is responsible, `torch.autograd.set_detect_anomaly(True)` is the right tool: it makes backward point at the exact forward operation that produced the corrupted tensor.

```python
import torch

torch.autograd.set_detect_anomaly(True)   # slow; use only while debugging

# Now the traceback on backward() includes the forward op that created
# the corrupted tensor, e.g. the offending `relu_(...)` line in your model.
loss = criterion(model(xb), yb)
loss.backward()   # raises with a forward-op traceback, not just a backward one
```

The subtler, scarier variant is the in-place op that does *not* raise — it modifies a tensor that backward did not happen to save, so the version check passes, but the math is now subtly wrong and the gradient is incorrect without any error. This is rare, but it is why the safe default in a forward pass is: **avoid in-place ops on anything that requires grad.** If you need to save memory, do it deliberately with gradient checkpointing, not with ad-hoc in-place tricks.

### 6.2 The submodule whose output is never used

A different way to get `grad is None` on a `requires_grad` parameter: build a submodule, run its forward, and then forget to use its output in the loss. This happens constantly with auxiliary heads ("we'll add the aux loss later") and with multi-task models under refactoring. The submodule's parameters are trainable and on no path to the loss, so they get `grad is None`. The audit flags them; the fix is either to wire the output into the loss or to delete the branch. Under `DistributedDataParallel` this same situation throws a louder error — `find_unused_parameters` — because DDP needs every parameter to participate in the backward to synchronize gradients; we cover that interaction in the distributed track.

#### Worked example: the multi-task model with a silent auxiliary head

A multi-task model with a main classification head and an auxiliary regression head, where the aux loss was commented out during an experiment and never restored:

```python
logits, aux = model(xb)             # model returns both heads
loss = ce(logits, y_class)          # main loss only
# loss = loss + 0.1 * mse(aux, y_reg)   # <-- aux term commented out
loss.backward()
```

The audit:

```bash
NO GRAD : 4 params  ['aux_head.0.weight', 'aux_head.0.bias',
                     'aux_head.2.weight', 'aux_head.2.bias']
  !! aux_head.0.weight: requires_grad=True but grad is None -> not on the path from the loss
```

Four `aux_head` parameters with `grad is None`, all `requires_grad = True`. They are computed in the forward (`aux` is returned) but never reach the loss. The aux head is dead weight — literally — burning forward compute and memory for nothing, and if you had *intended* to train it, the model is silently missing a task. The fix is one line: restore the aux loss term. The audit catches it before you ship a "multi-task" model that is quietly single-task.

## 7. Finetuning: freeze and unfreeze without stranding layers

Gradient-flow bugs are *especially* common in finetuning, because finetuning is precisely the regime where you deliberately freeze some parameters and train others — and getting the freeze/unfreeze/optimizer ordering wrong is the easiest way to train nothing, or to train the wrong thing.

### 7.1 The two classic ordering bugs

There are two failure modes that account for most finetuning gradient-flow bugs, and they are mirror images of each other:

1. **"I froze the backbone but also accidentally froze the head."** You loop over `model.parameters()` to freeze the backbone, but your loop catches the new head too (or you froze before adding the head). Result: nothing trains, loss is flat, the [overfit-one-batch test](/blog/machine-learning/debugging-training/the-overfit-a-single-batch-test) fails immediately. The audit shows *everything* in the `frozen` bucket.
2. **"I created the optimizer before unfreezing."** You build the optimizer over the frozen-backbone model (only the head is trainable), train stage one, then unfreeze the backbone for stage two — but the optimizer was built when the backbone had `requires_grad = False`, so even after you flip the flag, the backbone is not in `param_groups`. Stage two trains nothing new.

The figure below contrasts the second one, because it is the more insidious: the unfreeze *looks* like it worked (`requires_grad` is `True` again), but the optimizer was never told.

![A before-after of a frozen-backbone finetune: before, the optimizer is built before unfreezing so the backbone never enters its param groups; after, unfreeze first then build the optimizer with two LR groups so both stages learn](/imgs/blogs/your-model-isnt-learning-what-you-think-8.png)

The correct freeze-then-train recipe orders the operations so the optimizer always sees the final set of trainable parameters:

```python
# Stage 1: train only the new head on a frozen backbone.
for p in model.parameters():
    p.requires_grad_(False)        # freeze everything
model.fc = nn.Linear(2048, n_classes)   # new head is trainable by default
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=1e-3)
assert_optimizer_covers_model(model, optimizer)   # head present, backbone absent: correct
# ... train stage 1 ...

# Stage 2: unfreeze the backbone, REBUILD the optimizer with two LR groups.
for p in model.parameters():
    p.requires_grad_(True)
optimizer = torch.optim.AdamW([
    {"params": model.fc.parameters(),                          "lr": 1e-3},
    {"params": [p for n, p in model.named_parameters()
                if not n.startswith("fc")],                    "lr": 1e-5},
])
assert_optimizer_covers_model(model, optimizer)   # everything present now
```

Two things make this correct. First, the optimizer is rebuilt *after* the unfreeze, so every now-trainable parameter is in a group. Second, the backbone gets a much smaller LR (1e-5) than the head (1e-3): a pretrained backbone holds features you want to *nudge*, not overwrite, and a too-high LR on a pretrained backbone is its own destructive bug — it erases the very features you are finetuning to keep. The assertion after each rebuild is the cheap guarantee that the ordering is right.

### 7.2 PEFT and LoRA: confirm the adapter is in the graph

The same "is it actually training?" question is the central failure mode of parameter-efficient finetuning, where by design *almost everything* is frozen and only a tiny adapter trains. If the adapter is wired to the wrong `target_modules`, or the base model's `requires_grad` was not turned off, or gradient checkpointing interacts badly, you can run a LoRA finetune that updates *nothing* — the silent no-op. The PEFT-specific instruments are `print_trainable_parameters()` (which should show a small but *nonzero* trainable fraction, e.g. 0.2%) and the same gradient-flow audit restricted to the adapter parameters:

```python
from peft import LoraConfig, get_peft_model

peft_model = get_peft_model(base_model, LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"]))
peft_model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622

# after one backward, confirm the LoRA params actually receive gradients:
report = grad_flow_audit(peft_model, verbose=False)
lora_flowing = [n for n, _ in report["flowing"] if "lora" in n]
assert lora_flowing, "no LoRA params received a gradient -> adapter is a no-op"
print(f"{len(lora_flowing)} LoRA tensors flowing")
```

If `print_trainable_parameters()` reports `trainable%: 0.0000` or `lora_flowing` is empty, your adapter never entered the graph — the deep dive on that lives in [debugging LoRA and PEFT](/blog/machine-learning/debugging-training/debugging-lora-and-peft). The principle is identical to everything in this post: a parameter learns only if it is a `requires_grad` leaf, on the path to the loss, with a nonzero gradient, inside an optimizer group.

### 7.3 The gradient-checkpointing interaction that silently disables grads

There is one PEFT-plus-gradient-checkpointing interaction worth calling out by name, because it produces a perfect silent no-op and it surprises even experienced people. Gradient checkpointing saves activation memory by *not* storing intermediate activations during the forward pass and recomputing them during backward. For that recomputation to produce gradients for your adapter, the input to the checkpointed block has to require grad — and for a frozen base model with a LoRA adapter, the *embedding output* (the input to the first transformer block) comes from a frozen embedding and therefore has `requires_grad = False`. When the checkpointed block recomputes its forward, the whole recomputation runs without grad-tracking because its input did not require grad, and the adapter parameters inside never receive a gradient. The `print_trainable_parameters()` count looks fine — the adapter parameters *are* trainable leaves — but the audit shows them with `grad is None`, because the checkpoint recomputation severed the path.

The fix is a one-liner that Hugging Face provides exactly for this: call `model.enable_input_require_grads()` (or pass `gradient_checkpointing_kwargs={"use_reentrant": False}` on newer versions), which registers a forward hook that flips the embedding output to `requires_grad = True` so the checkpointed recomputation tracks gradients. The order also matters: enable gradient checkpointing and input-require-grads *before* wrapping with PEFT. The diagnostic is the same audit you already have:

```python
model.gradient_checkpointing_enable()
model.enable_input_require_grads()        # without this, LoRA grads are silently None
peft_model = get_peft_model(model, lora_config)

# confirm after one backward:
report = grad_flow_audit(peft_model, verbose=False)
assert [n for n, _ in report["flowing"] if "lora" in n], \
    "checkpointing severed the adapter path -> call enable_input_require_grads()"
```

This is gradient-flow debugging in its purest form: a memory optimization quietly changed *which tensors require grad*, the path was severed at the checkpoint boundary, and the only evidence is `grad is None` on parameters that `print_trainable_parameters()` swears are trainable. The audit is what reconciles the contradiction — "trainable but no gradient" — and points at the cause.

## 8. Case studies: gradient-flow bugs in the wild

These patterns are not hypothetical; they are the recurring shapes of real bug reports. Naming them helps you recognize them.

### 8.1 The tied-embedding freeze (Hugging Face)

Decoder-only language models routinely *tie* the input embedding and the output `lm_head` to the same weight matrix (a standard parameter-saving choice since the "Using the Output Embedding" work by Press and Wolf, 2017, and Inan et al., 2017). The practical consequence for debugging: `model.embed_tokens.weight` and `model.lm_head.weight` are the *same tensor*, so freezing one freezes both, and unfreezing one unfreezes both. The Worked example in Section 5 is exactly this. The tell is the audit showing `lm_head.weight` in the `frozen` bucket when you only meant to freeze the embedding. The defensive habit: when you freeze embeddings, print whether the head is tied (`model.config.tie_word_embeddings`) and run the audit to see what *else* you just froze.

### 8.2 The optimizer-before-head reorder (the canonical accidental freeze)

The pets example is the most common gradient-flow bug in transfer learning, and it is worth stating as a rule: **constructing the optimizer over `model.parameters()` and then mutating the model (replacing the head, adding a layer, unfreezing) strands the new or newly-trainable parameters.** It is silent because the stranded parameters often still get gradients — they are in the graph — so a gradient-only check passes. Only a param-changed check or an optimizer-coverage assert catches it. This is the single best argument for the discipline "build the optimizer last, and assert coverage."

### 8.3 The `find_unused_parameters` error under DDP

The "submodule output never used in the loss" bug is *silent* on a single GPU (the audit shows `grad is None`) but *loud* under `DistributedDataParallel`. DDP registers backward hooks on every parameter to all-reduce gradients across ranks; if a parameter never participates in backward, the all-reduce for it never fires, and DDP raises a `find_unused_parameters` error (or hangs). Setting `find_unused_parameters=True` suppresses the error but does not fix the underlying waste — those parameters are still dead weight, and you are paying a synchronization tax to discover it every step. The right fix is the same as on one GPU: wire the output into the loss or remove the branch. DDP just turns a silent single-GPU gradient-flow bug into a noisy multi-GPU one, which is one of the rare times distributed training does you a favor.

### 8.4 The `.detach()` that was meant to stop a *different* gradient

A common legitimate use of `.detach()` is to stop gradient flow into a target — for example, in a teacher-student setup or a target network in RL, you `.detach()` the teacher's output so the student's loss does not train the teacher. The bug arises when the detach is placed on the *wrong* branch, or when a copy-paste moves a teacher-style detach into the student's forward path. The audit then shows the student's upstream parameters with `grad is None`, and the confusing part is that `.detach()` was deliberate — just misplaced. The lesson: every `.detach()` in a forward path is a *deliberate* gradient cut; audit after adding one to confirm you cut exactly the branch you intended and no more.

### 8.5 The frozen layer that should have been trainable in a multi-task fanout

A more architectural variant shows up in multi-task and multi-encoder models, where different parameters are *meant* to train on different tasks and the freeze logic gets the boolean backwards. Picture a shared encoder feeding three task heads, where a config flag is supposed to freeze the encoder for two of the tasks and train it for the third. A single inverted condition — `if task in frozen_tasks: p.requires_grad_(True)` instead of `False` — and the freeze is exactly inverted: the encoder trains on the tasks where it should be frozen and is frozen on the one task that was supposed to adapt it. The headline loss looks fine because *something* is always training, and the per-task metrics are merely "a bit lower than expected" on the one task whose encoder is stuck. The audit, run per-task, makes the inversion obvious: the encoder shows up in the `frozen` bucket on exactly the task where you expected it to be `flowing`. The general principle is that freeze logic is *boolean* logic, and boolean logic is where off-by-a-`not` errors live; the audit is how you check that the booleans landed where you meant them, task by task. Whenever a freeze decision is computed (rather than hard-coded), audit the result rather than trusting the code that set the flags.

### 8.6 The `eval()`-mode forward that looks like a frozen layer

One last pattern, because it is a frequent *misdiagnosis* of a gradient-flow bug. If part of your model is accidentally in `eval()` mode during training — say a frozen-feature-extractor branch you called `.eval()` on to fix its BatchNorm statistics, and then forgot you also wanted *other* parameters in that branch to train — the BatchNorm layers stop updating their running statistics, which can *look* like "this branch isn't learning." It is not a gradient-flow bug: the trainable parameters in that branch still receive gradients and still update. The running mean and variance buffers are *not parameters* (they are buffers, updated by the forward pass in train mode, not by the optimizer), so `.eval()` freezing them is a mode bug, not a graph bug. The audit correctly shows the branch's parameters as `flowing`, which is the signal that redirects you: a flowing audit on a "not learning" branch means look at modes and buffers, not the graph. This is the boundary between this post and the train/eval-mode post, and the audit is the instrument that draws the line.

## 9. When this is (and isn't) a gradient-flow bug

A good diagnostic is as much about *ruling out* as ruling in. Here is when the "a submodule isn't learning" symptom points at gradient flow, and when it points somewhere else entirely.

**It is a gradient-flow bug when:** total loss falls but a *specific* submodule's parameters never change; the gradient-flow audit shows a `requires_grad = True` parameter with `grad is None`; or the param-changed check shows `delta == 0` on a parameter with a nonzero gradient. These three signatures are dispositive. The fix is in the wiring (re-enable grad, remove the detach, rebuild the optimizer), not in the hyperparameters.

**It is *not* a gradient-flow bug when:**

- **The loss is flat at chance for the *whole* model, not one submodule.** If *everything* fails to move, the overfit-one-batch test fails globally, and the suspect is the loss wiring, a learning rate of zero, or a data-path bug — not a single severed branch. Run the [overfit-a-single-batch test](/blog/machine-learning/debugging-training/the-overfit-a-single-batch-test) first; if it fails for all parameters, you are debugging the whole circuit, not one wire.
- **The gradient is present and nonzero but tiny, and shrinks with depth.** That is vanishing gradients — a numerical/optimization issue of magnitude, not a *cut* path. The audit shows `flowing`, just with small norms in early layers. That belongs to [gradients exploding and vanishing](/blog/machine-learning/debugging-training/gradients-exploding-and-vanishing), and the fix is initialization, normalization, or residual connections, not graph surgery.
- **`grad.norm() == 0` because the activations are dead.** A ReLU whose pre-activations are all negative produces a zero gradient through no fault of the graph — the path is intact, the signal is genuinely zero. That is a dead-neuron problem; revive it with better init or a lower LR, per [dead neurons and saturated activations](/blog/machine-learning/debugging-training/dead-neurons-and-saturated-activations).
- **The model learns in `train()` but fails in `eval()`.** That is a train/eval-mode bug (dropout, BatchNorm running stats), not gradient flow; the parameters *are* updating, the inference path just differs.
- **A parameter has a healthy gradient and a nonzero update, but the loss still does not fall.** Then the wiring is fine and the bug is in the *objective* — the loss is computed against the wrong target, the reduction is wrong (sum vs mean changes the effective LR by the batch size), or the labels are shifted. The parameters move; they just move toward the wrong place. That is a loss-function bug, not a gradient-flow bug, and the gradient-flow audit will (correctly) tell you everything is flowing — which is itself a useful signal: a clean audit *redirects* you away from the graph and toward the loss.
- **Only certain parameters move and the pattern matches a deliberate freeze schedule.** Discriminative or layer-wise learning rates, gradual unfreezing, and LoRA all *intentionally* leave many parameters frozen or slow. Before you call a frozen parameter a bug, confirm it was not frozen on purpose. The audit's `frozen` bucket is not automatically a problem list; it is a *did-you-mean-to* list. The bug is only the frozen parameter you did *not* intend to freeze.

The clean rule: gradient-flow bugs are about *whether a parameter updates at all*. If the parameter is updating but the *result* is wrong (too slow, diverging, fine-in-train-bad-in-eval, loss-not-falling), look elsewhere. The audit is the test that distinguishes the two — it tells you whether the gradient even arrived, and a *clean* audit is as informative as a dirty one: it clears the entire model-code/gradient-flow branch of the [bug taxonomy](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) and sends you to data, optimization, or evaluation instead. Spending five minutes to *rule out* gradient flow is rarely wasted, because it removes a whole branch of the search tree.

#### Worked example: bisecting "the new head won't learn"

Put it together on a fresh case. A colleague reports: "I added a second classification head to my model for a new label, but it stays at chance while the original head works fine." Bisect:

1. **Overfit one batch.** The original head's loss craters; the new head's loss stays flat. So it is *not* a global wiring or data bug — part of the model learns. This already points at a gradient-flow bug localized to the new head.
2. **Run the audit.** `new_head.weight: requires_grad=True but grad is None`. So the new head is a trainable leaf that gets no gradient. Branch: `grad is None`.
3. **Check the path.** Grep the loss expression. The loss is `ce(logits_old, y_old)` — the new head's logits are computed but never added to the loss. Confirmed: "submodule output never used."
4. **Fix.** `loss = ce(logits_old, y_old) + ce(logits_new, y_new)`. Re-run the audit: `new_head.weight` now `flowing`, norm 0.7. Overfit-one-batch now drives *both* heads to zero. Ship.

Total time from report to fix: about five minutes, because the audit collapsed "won't learn" into "gets no gradient" into "not in the loss." That is the entire value proposition of reading the instruments before turning knobs.

## 10. Building gradient-flow checks into your harness

The instruments in this post are most valuable when they run *automatically*, not only when you suspect a problem. Three checks, added once, catch the entire bug class at startup or in the first step:

```python
def gradient_flow_preflight(model, optimizer, sample_batch, criterion):
    """Run once before a real training run. Catches the whole bug class."""
    xb, yb = sample_batch
    model.train()

    # 1. The graph exists and the loss is connected.
    loss = criterion(model(xb), yb)
    assert loss.grad_fn is not None, "loss.grad_fn is None -> loss not on the graph"

    # 2. Every trainable param is in the optimizer.
    assert_optimizer_covers_model(model, optimizer)

    # 3. Every trainable param actually moves in one step.
    before = snapshot_params(model)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    audit = grad_flow_audit(model, verbose=False)
    assert not audit["no_grad"], f"params with grad None: {audit['no_grad']}"
    optimizer.step()
    stuck = [n for n, p in model.named_parameters()
             if p.requires_grad and (p - before[n]).norm().item() == 0.0]
    assert not stuck, f"params did not move after one step: {stuck}"
    print("gradient-flow preflight PASSED: all trainable params flow and move")
```

This is thirty seconds of compute that guards against two days of wasted GPU time. It belongs in every training harness, gated behind a `--preflight` flag you run before any long job. The instruments compose: `grad_flow_audit` for "did a gradient arrive," `report_param_changes` / the `stuck` check for "did the update happen," `assert_optimizer_covers_model` for "is it even in the optimizer," and `loss.grad_fn` for "is the graph built at all." Together they cover every link in the chain from Figure 1.

## 11. Key takeaways

- **Total loss is an aggregate; it can fall while a submodule learns nothing.** A falling loss curve is not proof that any particular parameter is updating. Trust per-parameter instruments, not the headline number.
- **A parameter learns only if it completes the whole chain:** `requires_grad` leaf → on the path from the loss → nonzero `.grad` → in an optimizer `param_group` → moved by `step()`. Break any link and it freezes silently.
- **Run two instruments, not one.** The gradient-flow audit catches `grad is None` (no gradient arrived). The param-changed check catches `delta == 0` with a healthy gradient (the optimizer skipped it). The pets bug was invisible to the first and obvious to the second.
- **`grad is None` on a `requires_grad = True` leaf means the path is cut** — by an accidental freeze, a `.detach()`/`.data`/`no_grad`, a numpy round-trip, or an output never used in the loss. Grep the forward pass for the cut.
- **Build the optimizer last.** Constructing it over `model.parameters()` and then replacing the head, adding a layer, or unfreezing strands the new parameters. Assert `assert_optimizer_covers_model` after every model mutation.
- **`grad.norm() == 0` (not `None`) is usually a learning bug, not a wiring bug** — dead ReLUs, saturated activations, constant features. The path is intact; the signal is genuinely zero. Look at init and LR, not the graph.
- **In-place ops corrupt the graph.** `x += y`, `relu_(x)`, `tensor[mask] = 0` on a graph tensor can break backward; `set_detect_anomaly(True)` points at the offending forward op. Prefer out-of-place ops in the forward path.
- **Reassign vs mutate:** `p = p - lr*g` throws the update away; `with torch.no_grad(): p -= lr*g` keeps it. The optimizer and model point at the *original* tensor object.
- **For finetuning, order is everything:** freeze → add/replace head → build optimizer → (later) unfreeze → *rebuild* optimizer with per-group LRs. For PEFT, `print_trainable_parameters()` must be nonzero and the audit must show adapter params flowing.
- **Distinguish "doesn't update" from "updates wrong."** Gradient-flow bugs stop a parameter from changing at all. If it changes but the result is bad (too slow, diverging, eval-only failure), the bug is elsewhere.
- **A clean audit is a result, not a non-result.** Confirming every parameter flows and moves clears the entire gradient-flow branch of the taxonomy and redirects you to data, optimization, or evaluation — that redirection is worth the five minutes it costs.
- **Make the checks automatic.** A thirty-second gradient-flow preflight — `loss.grad_fn` set, optimizer covers the model, every trainable param moves once — guards against two-day wasted runs and belongs in every training harness behind a flag.

## 12. Further reading

- **PyTorch Autograd mechanics** — the official docs on how the graph is built, `requires_grad` propagation, leaves, and `grad_fn`. The authoritative reference for everything in Section 2: `pytorch.org/docs/stable/notes/autograd.html`.
- **PyTorch `torch.autograd.set_detect_anomaly`** — the docs for anomaly detection, the right tool for in-place corruption and NaN-in-backward bugs (Section 6).
- **"Using the Output Embedding to Improve Language Models"**, Press and Wolf, 2017, and **"Tying Word Vectors and Word Classifiers"**, Inan, Khosravi, Socher, 2017 — the origin of weight tying that makes the frozen-embedding bug (Section 5, 8.1) possible.
- **Hugging Face PEFT documentation** — `LoraConfig`, `target_modules`, `print_trainable_parameters`, and the common no-op adapter pitfalls (Section 7.2).
- **A taxonomy of training and finetuning bugs** — [the master decision tree](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) that places gradient flow in the model-code branch and routes you between the six places a bug hides.
- **The overfit-a-single-batch test** — [the highest-leverage sanity check](/blog/machine-learning/debugging-training/the-overfit-a-single-batch-test); a severed gradient path is the textbook reason it fails.
- **Instrumenting a training run: what to log** — [the per-layer grad-norm and update-norm logging](/blog/machine-learning/debugging-training/instrumenting-a-training-run-what-to-log) that surfaces gradient-flow bugs continuously, not just at preflight.
- **Debugging LoRA and PEFT** — [the adapter-specific deep dive](/blog/machine-learning/debugging-training/debugging-lora-and-peft) on the silent no-op and `target_modules` mistakes.
- **The training debugging playbook** — [the capstone checklist](/blog/machine-learning/debugging-training/the-training-debugging-playbook) that folds the gradient-flow audit into the full symptom → suspect → test → fix workflow.

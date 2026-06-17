---
title: "Loss Function Bugs: Optimizing the Wrong Thing, Confidently"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "How to find the loss-function bug that makes your curve descend beautifully while the model learns the wrong objective, with hand-computed unit tests, a logits-versus-probabilities detector, and before-and-after evidence across vision, NLP, tabular, and multi-task models."
tags:
  [
    "debugging",
    "model-training",
    "loss-function",
    "cross-entropy",
    "pytorch",
    "finetuning",
    "deep-learning",
    "nlp",
    "computer-vision",
    "tabular",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 58
image: "/imgs/blogs/loss-function-bugs-1.png"
---

The run looked perfect. The loss started at 1.10, dropped to 0.95 in the first hundred steps, eased down to 0.7 by epoch two, and was sitting at 0.55 when you checked it before lunch. The curve was smooth, monotonic, the kind of descent you screenshot and paste into Slack with a thumbs-up. Then you ran the validation metric the team actually cares about — top-1 accuracy on the held-out set — and it read 38 percent. On a ten-class problem. The model was barely above the 10 percent you would get from guessing, and it had been "improving" for six hours. The loss had fallen by half and the accuracy had not moved off chance.

This is the signature of a loss-function bug, and it is the most psychologically dangerous bug in this entire series. Every other failure mode at least has the decency to *look* like a failure. A NaN flatlines the dashboard. An exploding gradient spikes the curve. A dead run plateaus visibly. But a loss bug lets the loss curve descend exactly as you expect, because the loss is descending — the optimizer is doing its job perfectly, faithfully minimizing the number you handed it. The problem is that the number you handed it is not the number you meant. You are optimizing the wrong objective, confidently, with a beautiful curve to prove it. The optimizer cannot tell you that your loss measures the wrong thing; it can only drive it down.

![A vertical stack showing data, model, loss, optimizer, and metric layers with the loss layer flagged as the place a bug optimizes the wrong objective while the curve still descends](/imgs/blogs/loss-function-bugs-1.png)

In the six-places framework that runs through this series — a bug hides in *data*, *optimization*, *model code*, *numerics*, *systems*, or *evaluation* — the loss function sits at a peculiar junction. It is the last thing the model produces and the first thing the optimizer consumes, the hinge between the forward pass and the gradient. A bug here is technically "model code," but its symptom masquerades as an optimization or even an evaluation problem: the loss moves, so optimization looks fine; the metric does not, so you go hunting in the eval pipeline. The unifying tell that pulls you back to the loss is exactly the one in the story above: **the loss decreases but the metric you care about does not track it**, or the model trains but is badly mis-calibrated, or the loss number is suspiciously, impossibly low. When loss and metric decouple, suspect the loss before you suspect anything else.

By the end of this post you will be able to take any run where the curve and the metric disagree and localize the loss bug in minutes. You will know the eight ways a loss function goes wrong — passing probabilities into a loss that wants logits, the wrong reduction silently rescaling your learning rate, padding tokens leaking into the average, the wrong target dtype or shape broadcasting into nonsense, label smoothing applied twice, a sign error that maximizes instead of minimizes, multi-task weights on incompatible scales, and a regression loss whose scale dwarfs your targets. More importantly, you will have the single highest-leverage check in the whole category: **unit-test the loss on tiny hand-computed inputs**. Compute the expected cross-entropy by hand, assert your loss returns it, and an entire class of bugs dies before you ever touch a GPU. Let us start with the most common and most insidious of them: the double softmax.

## 1. The double softmax: logits versus probabilities

The first thing you must know about every loss function in your framework is whether it expects **raw logits** or **probabilities**. This single distinction is the source of more loss bugs than any other, and it is invisible in the loss value — both versions return a finite, descending number. A *logit* is the raw, unbounded output of your model's final linear layer: any real number, positive or negative, before any squashing. A *probability* is what you get after applying `softmax` (for multi-class) or `sigmoid` (for binary): a number in $[0, 1]$ that sums to one across classes. The bug is feeding one where the other is expected.

PyTorch's API makes the trap easy to fall into because it is, sensibly, optimized for numerical stability. `torch.nn.CrossEntropyLoss` **expects raw logits** and applies `log_softmax` internally. It does this on purpose: computing `log(softmax(x))` directly would overflow for large logits and underflow for small probabilities, so the stable formulation fuses the softmax and the log into one operation using the log-sum-exp trick. The consequence is that if *you* apply `softmax` in your model's `forward` and *then* pass the result to `CrossEntropyLoss`, the softmax runs **twice**. This is the double softmax, and it is quietly catastrophic.

It is worth seeing exactly why the fused form is more stable than the naive one, because that stability is the entire reason the API is shaped the way it is — and understanding it makes the double-softmax bug feel inevitable rather than arbitrary. The naive computation of $\log p_y = \log \frac{e^{z_y}}{\sum_j e^{z_j}}$ exponentiates the logits first. If any logit is large — say $z = 90$, which is unremarkable in a deep network with no output normalization — then $e^{90} \approx 10^{39}$, which overflows fp32's maximum of about $3.4\times10^{38}$, and you get `inf` in the numerator and denominator, then `inf / inf = NaN`. The log-sum-exp trick rewrites the log-softmax as $\log p_y = z_y - \log\sum_j e^{z_j}$, and then stabilizes the sum by factoring out the maximum logit $m = \max_j z_j$: $\log\sum_j e^{z_j} = m + \log\sum_j e^{z_j - m}$. Every exponent $z_j - m$ is now $\le 0$, so every $e^{z_j - m} \in (0, 1]$, and the sum cannot overflow. This is what `log_softmax` does internally and what `CrossEntropyLoss` inherits. The lesson for debugging is twofold: first, you must never hand `CrossEntropyLoss` pre-softmaxed probabilities — you would be throwing away the very stability the loss was built to provide and re-softmaxing on top of it; second, a custom loss that computes `torch.log(torch.softmax(...))` by hand is a latent NaN waiting for a large logit, so reach for the fused `log_softmax` or `cross_entropy` instead. The double softmax is, in this light, the same mistake as the hand-rolled `log(softmax(...))` — both ignore that the framework already did the stable, correct thing for you.

![A decision tree from the model output asking whether it is raw logits or probabilities, branching to CrossEntropyLoss and BCEWithLogitsLoss on one side and NLLLoss or the double-softmax bug on the other](/imgs/blogs/loss-function-bugs-2.png)

### Why the double softmax kills the gradient

Here is the science, because the *why* tells you the signature. Softmax maps logits $z$ to probabilities $p$ via $p_i = e^{z_i} / \sum_j e^{z_j}$. The key property is that softmax is *contractive*: it squashes any input into the simplex $[0,1]^C$ with entries summing to one. Apply softmax once to logits, and you get a probability vector that might be sharp, like $[0.95, 0.03, 0.02]$. Apply softmax *again* to that probability vector, and you squash an already-bounded vector even closer to uniform, because the inputs are now all in $[0,1]$ and their differences are tiny. The second softmax of $[0.95, 0.03, 0.02]$ is approximately $[0.51, 0.25, 0.24]$ — nearly uniform.

This near-uniformity is the disaster. Cross-entropy loss is $L = -\log p_{y}$ where $y$ is the true class. With a correct, confident prediction the loss is small ($-\log 0.95 \approx 0.05$); with the double-softmax-flattened prediction it is $-\log 0.51 \approx 0.67$, much closer to the chance loss $-\log(1/C)$. For a 10-class problem chance loss is $-\log 0.1 \approx 2.30$, and a double-softmaxed model lives perpetually near there because the second softmax prevents the probabilities from ever becoming confident. The loss *can* still decrease a little — the logits can shift to make the post-double-softmax distribution slightly favor the right class — but the gradient is tiny because softmax's Jacobian near uniform is small. You get a loss that descends slowly to a floor around chance, and an accuracy that barely moves. Exactly the symptom we opened with.

The Jacobian point deserves a full derivation, because it is the quantitative heart of why the double softmax stalls training and not just why it caps accuracy. The gradient of cross-entropy with respect to the *logits* it is given is famously clean: $\partial L / \partial z_i = p_i - y_i$, the predicted probability minus the one-hot target. This is the single nicest fact about cross-entropy and it is exactly why we pair it with raw logits — the gradient is the residual, bounded in $[-1, 1]$, large when the prediction is wrong and small when it is right, never vanishing prematurely. Now insert a softmax in front. The loss is no longer a direct function of the logits $z$; it is a function of $q = \text{softmax}(z)$, and the gradient with respect to $z$ picks up the softmax Jacobian by the chain rule: $\partial L / \partial z = J_{\text{softmax}}^\top (p - y)$, where $J_{\text{softmax}}$ is the matrix with entries $\partial q_i / \partial z_j = q_i(\delta_{ij} - q_j)$. Near uniform, $q_i \approx 1/C$, so the diagonal entries are $\approx \frac{1}{C}(1 - \frac{1}{C})$ and the off-diagonals are $\approx -\frac{1}{C^2}$ — every entry of the Jacobian is order $1/C$ or smaller. The clean residual gradient $p - y$ gets multiplied by a matrix whose entries are tiny, and the gradient that reaches your logits is suppressed by a factor of roughly $1/C$. For 10 classes that is a 10× weaker gradient; for 1000 classes it is a 1000× weaker gradient. The double softmax does not just cap your ceiling at chance accuracy — it strangles the gradient that would let you climb toward that ceiling, which is why these runs look *almost* dead rather than merely capped.

![A before-and-after comparison contrasting a buggy double softmax with near-zero gradient and chance accuracy against the fixed logits-to-cross-entropy path with healthy gradient and high accuracy](/imgs/blogs/loss-function-bugs-3.png)

### Detecting it: read your model's last layer

The detector is mechanical. **Look at your model's `forward` and find the last operation before the output you pass to the loss.** If it is `F.softmax`, `torch.softmax`, `nn.Softmax`, `F.sigmoid`, `torch.sigmoid`, or `F.log_softmax`, and your loss is `CrossEntropyLoss` or `BCEWithLogitsLoss`, you have a double-activation bug. The pairing rules are short and worth memorizing:

| Loss | Expects | Applies internally | Pair with last layer |
| --- | --- | --- | --- |
| `nn.CrossEntropyLoss` | raw logits | `log_softmax` + NLL | linear (no activation) |
| `nn.BCEWithLogitsLoss` | raw logits | `sigmoid` (log-sum-exp stable) | linear (no activation) |
| `nn.NLLLoss` | log-probabilities | nothing | `log_softmax` |
| `nn.BCELoss` | probabilities in [0,1] | nothing | `sigmoid` |

Two stable, recommended pairings (`logits → CrossEntropyLoss`, `logits → BCEWithLogitsLoss`) and two manual pairings (`log_softmax → NLLLoss`, `sigmoid → BCELoss`). The manual pairings are not wrong, but they are fragile: `BCELoss` on a `sigmoid` output can produce `inf` when a probability rounds to exactly 0 or 1, which is precisely why `BCEWithLogitsLoss` exists. The single most common mistake is using a `sigmoid` or `softmax` in the model — often copied from a tutorial that used `NLLLoss` or `BCELoss` — and then switching the loss to the logits-expecting version without removing the activation.

Here is a runnable detector you can drop into any project. It introspects the model and the loss and flags the mismatch before you waste a single epoch:

```python
import torch
import torch.nn as nn

def detect_logits_vs_probs(model, loss_fn, sample_input):
    """Warn if the model's output looks like probabilities but the
    loss expects logits (or vice versa). Run once before training."""
    model.eval()
    with torch.no_grad():
        out = model(sample_input)

    # Heuristics: probabilities are bounded and (for softmax) rows sum to 1.
    in_unit_range = (out.min() >= 0.0) and (out.max() <= 1.0)
    rows_sum_to_one = torch.allclose(
        out.sum(dim=-1), torch.ones(out.shape[0]), atol=1e-3
    ) if out.dim() >= 2 else False

    expects_logits = isinstance(
        loss_fn, (nn.CrossEntropyLoss, nn.BCEWithLogitsLoss)
    )
    looks_like_probs = in_unit_range and (rows_sum_to_one or out.shape[-1] == 1)

    if expects_logits and looks_like_probs:
        raise RuntimeError(
            "Loss expects logits but model output looks like probabilities. "
            "Likely a softmax/sigmoid in forward() feeding a *WithLogits loss "
            "(double activation). Remove the final activation."
        )
    if isinstance(loss_fn, (nn.NLLLoss,)) and not in_unit_range:
        raise RuntimeError(
            "NLLLoss expects log-probabilities, but model output is unbounded "
            "(looks like raw logits). Add F.log_softmax before the loss."
        )
    print("Logits/probabilities pairing looks consistent.")
```

The range check is a heuristic, not a proof — a model can produce logits that happen to fall in $[0,1]$ early in training — but combined with "do the rows sum to one?" it catches the real cases reliably, because a genuine softmax output sums to one to numerical precision and raw logits essentially never do.

#### Worked example: the double softmax in numbers

Take a 3-class problem, one sample, true class 0. Suppose the model's final linear layer outputs logits $z = [2.0, 0.5, 0.1]$.

The correct path, `CrossEntropyLoss(z, 0)`, computes softmax once: $p = [0.659, 0.147, 0.099]$ after normalization (the values are $e^{2.0}, e^{0.5}, e^{0.1}$ normalized; rounding aside, $p_0 \approx 0.66$). The loss is $-\log 0.66 \approx 0.42$. A confident, correct prediction with a moderate loss and a strong gradient pushing $z_0$ higher.

The buggy path applies softmax in the model first, producing $p = [0.66, 0.15, 0.10]$ (approximately, summing to 1), then `CrossEntropyLoss(p, 0)` softmaxes *that*. Softmax of $[0.66, 0.15, 0.10]$ is roughly $[0.42, 0.26, 0.25]$ — the gaps collapsed because the inputs were already tiny. Now the loss is $-\log 0.42 \approx 0.87$, double the correct value, and crucially the prediction is barely above uniform. As training proceeds the best the model can do is push the pre-double-softmax probabilities toward $[1, 0, 0]$, whose second softmax is $[0.58, 0.21, 0.21]$ — still capped well below confident. The model is structurally prevented from ever being sure, the loss floors around $0.7$, and accuracy hovers near chance. The fix is one line: delete the `softmax` from `forward`. After the fix, on this exact example, the loss drops from $0.87$ to $0.42$ on step zero and continues falling toward zero as $z_0$ grows — the gradient is suddenly real.

This is also the place to cross-link the broader hunt: a double softmax that pushes logits to extremes can tip into the *numerics* place and produce `inf`/`NaN`, which is the subject of [hunting NaNs and infs](/blog/machine-learning/debugging-training/hunting-nans-and-infs). The `*WithLogits` losses exist precisely to keep that hunt from ever starting.

## 2. The NLLLoss and BCELoss traps

The logits-versus-probabilities axis has two siblings that bite from the opposite direction. `nn.NLLLoss` (negative log-likelihood) expects **log-probabilities**, not probabilities and not logits. If you feed it raw logits, it computes $-z_y$ instead of $-\log p_y$, which for positive logits is *negative loss* — your loss can go below zero and the optimizer happily maximizes the wrong quantity. If you feed it probabilities (forgetting the log), it computes $-p_y$, a loss bounded in $[-1, 0]$ that barely moves and never trains. The correct pairing is `F.log_softmax` in the model (or just before the loss) feeding `NLLLoss`; this is mathematically identical to `CrossEntropyLoss` on logits, and in fact `CrossEntropyLoss` *is* `LogSoftmax` followed by `NLLLoss` fused together.

`nn.BCELoss` versus `nn.BCEWithLogitsLoss` is the binary analog. `BCELoss` expects probabilities in $[0, 1]$ and computes $-[y \log p + (1-y)\log(1-p)]$ directly. The danger is the logarithm: if your `sigmoid` saturates to exactly $0.0$ or $1.0$ in fp16, $\log 0$ is $-\infty$ and the loss is `inf`. `BCEWithLogitsLoss` takes raw logits and uses the log-sum-exp-stable form $\max(z, 0) - z y + \log(1 + e^{-|z|})$, which never overflows. The bug is twofold: feeding logits to `BCELoss` (which clamps them as if they were probabilities and computes garbage), or applying `sigmoid` in the model and then using `BCEWithLogitsLoss` (the binary double-sigmoid, same flavor as the double softmax). Always prefer the `*WithLogits` variant and keep your model output as raw logits.

The stable-form derivation is short enough to walk through, and it shows precisely where `BCELoss` loses precision that `BCEWithLogitsLoss` keeps. Binary cross-entropy on a logit $z$ with label $y \in \{0,1\}$, where $p = \sigma(z) = 1/(1+e^{-z})$, is $\ell = -[y \log \sigma(z) + (1-y)\log(1-\sigma(z))]$. Substitute $\log \sigma(z) = -\log(1 + e^{-z})$ and $\log(1 - \sigma(z)) = -z - \log(1+e^{-z})$ and the whole thing collapses to $\ell = \max(z, 0) - z y + \log(1 + e^{-|z|})$. The crucial term is $\log(1 + e^{-|z|})$: because the exponent is $-|z| \le 0$, $e^{-|z|} \in (0, 1]$ always, so the `log` argument is in $[1, 2]$ and never underflows or overflows no matter how large $|z|$ grows. Contrast `BCELoss`, which is handed $p$ directly and must compute $\log p$: when $z = 30$, the true $p$ is $1 - 9\times10^{-14}$, which fp32 rounds to exactly $1.0$, so $\log(1 - p) = \log 0 = -\infty$, and a single confidently-correct-but-saturated prediction sends your loss to `inf`. PyTorch's `BCELoss` clamps its log output to $-100$ to avoid literal `-inf`, but that clamp is itself a silent distortion: the gradient is now wrong for any saturated example. The `*WithLogits` form never needs the clamp because it never forms $\log 0$. This is the same numerical-stability story as `CrossEntropyLoss` versus a hand-rolled `log(softmax(...))`, just in the binary case — and it is why "keep your output as logits and use the fused loss" is the single rule that prevents an entire family of `inf`/`NaN` bugs.

Here is a tiny reference that makes all four equivalences and traps explicit, the kind of thing worth pasting into a scratch notebook the first time you wire up a new model:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

logits = torch.tensor([[2.0, 0.5, 0.1]])      # raw model output
target = torch.tensor([0])                      # class index

ce = nn.CrossEntropyLoss()(logits, target)
nll = nn.NLLLoss()(F.log_softmax(logits, dim=-1), target)
print(ce.item(), nll.item())                    # identical: CE == NLL(log_softmax)

# Binary case, one logit per sample
blogits = torch.tensor([1.5, -0.7])
btarget = torch.tensor([1.0, 0.0])
bce_logits = nn.BCEWithLogitsLoss()(blogits, btarget)
bce_prob = nn.BCELoss()(torch.sigmoid(blogits), btarget)
print(bce_logits.item(), bce_prob.item())       # equal here, but BCE can overflow

# The overflow that BCEWithLogitsLoss avoids:
sat = torch.tensor([0.0, 1.0])                  # a saturated sigmoid output
print(nn.BCELoss()(sat, torch.tensor([1.0, 0.0])).item())  # inf on the log(0) terms
```

Run it once. Seeing `inf` come out of `BCELoss` on a saturated input, and *not* out of `BCEWithLogitsLoss`, is the kind of concrete demonstration that fixes the rule in your memory permanently.

#### Worked example: the binary classifier that NaN'd only in fp16

A team trained a binary fraud classifier with a `sigmoid` head and `nn.BCELoss`, and it worked in fp32 for months. They turned on automatic mixed precision (`torch.amp.autocast`) to speed up training, and the run started NaN-ing within a few hundred steps — but only sometimes, and never reproducibly. The instruments told a clean story once they looked: the loss would sit around 0.3, then a single step would read `inf`, then the next step the whole model was NaN. The confirming test was to log the maximum and minimum of the `sigmoid` output each step. In fp32, the sigmoid output stayed in roughly $[10^{-7}, 1 - 10^{-7}]$ — never exactly 0 or 1. In fp16, whose smallest positive normal is about $6.1\times10^{-5}$ and whose precision near 1.0 is coarse, a confident logit of $z \approx 12$ produced a sigmoid that rounded to *exactly* $1.0$. Then `BCELoss` computed $\log(1 - 1.0) = \log 0 = -\infty$ for the next negative example, and one `inf` loss became a fully-NaN model one optimizer step later. The numerics are exact: fp16 cannot represent $1 - 6\times10^{-6}$, so it snaps to $1.0$, and $\log(1-1.0)$ is the textbook `log(0)` NaN source. The fix was a single substitution — `nn.BCEWithLogitsLoss` on the raw logit instead of `nn.BCELoss` on the sigmoid — which uses the stable $\log(1 + e^{-|z|})$ form that never forms $\log 0$. After the swap, the fp16 run trained identically to the fp32 one, loss steady at 0.3 with no NaN at any step. This is the canonical "works in fp32, NaNs in fp16" loss bug, and it is one substitution away from gone; the deeper precision mechanics live in [hunting NaNs and infs](/blog/machine-learning/debugging-training/hunting-nans-and-infs).

## 3. Wrong reduction: how `sum` versus `mean` rescales your learning rate

The second great class of loss bug has nothing to do with activations and everything to do with a one-word argument: `reduction`. Every PyTorch loss takes `reduction='mean'` (the default), `reduction='sum'`, or `reduction='none'`. The default averages the per-element losses; `sum` adds them; `none` returns the per-element tensor. Choosing the wrong one does not produce a wrong *answer* in the sense of a NaN or a crash — it produces a wrong *gradient scale*, and that quietly rescales your effective learning rate by the batch size.

### The science: gradient scale and effective learning rate

Let $\ell_i$ be the loss on example $i$ in a batch of size $N$. With `mean`, the batch loss is $L = \frac{1}{N}\sum_i \ell_i$, so the gradient is $\nabla L = \frac{1}{N}\sum_i \nabla \ell_i$ — the *average* per-example gradient. With `sum`, $L = \sum_i \ell_i$ and $\nabla L = \sum_i \nabla \ell_i$ — the *sum*, which is $N$ times larger. The SGD update is $\theta \leftarrow \theta - \eta \nabla L$. So switching from `mean` to `sum` with the same learning rate $\eta$ is exactly equivalent to keeping `mean` and multiplying the learning rate by $N$:

$$\eta_{\text{effective}}^{\text{sum}} = N \cdot \eta_{\text{nominal}}$$

For a batch size of 32 and a configured learning rate of $3\times10^{-4}$, a stray `reduction='sum'` gives you an effective learning rate of $32 \times 3\times10^{-4} \approx 9.6\times10^{-3}$ — a 32-fold jump. That is squarely in loss-spike-and-divergence territory. And the failure is *batch-size-dependent*: change your batch size from 32 to 64 and the effective learning rate doubles again, so a config that "worked" on one GPU diverges when you scale to a bigger batch, which looks like a distributed or systems bug but is really a reduction bug interacting with batch size. This is why the learning rate that worked at one batch size mysteriously diverges at another — see [the learning rate is almost always the problem](/blog/machine-learning/debugging-training/the-learning-rate-is-almost-always-the-problem) for the LR side of this story.

![A before-and-after comparison showing reduction mean keeping gradient scale at one and effective learning rate as configured versus reduction sum amplifying gradient scale by the batch size and spiking the effective learning rate](/imgs/blogs/loss-function-bugs-4.png)

### The subtler version: per-token versus per-sequence averaging in NLP

Reduction has a second, harder form in sequence models. When you train a language model, the loss is computed over *tokens*, but a batch contains *sequences* of different lengths. There are two reasonable averages and they are not the same. **Per-token averaging** divides the total loss by the number of (non-pad) tokens, so every token contributes equally regardless of which sequence it is in. **Per-sequence averaging** first averages within each sequence, then averages across sequences, so a 5-token sequence and a 500-token sequence each contribute one "vote." These differ whenever sequence lengths vary, and they produce different gradients: per-sequence averaging up-weights short sequences. Most LLM training uses per-token averaging (it is what `CrossEntropyLoss` with flattened logits and `ignore_index` gives you), but if you accidentally average per-sequence — for example by calling `.mean()` over a `[batch, seq]` loss tensor that still has pad positions, or by averaging a list of per-sequence means — you change the objective and the model over-fits short examples. The fix is to flatten to `[batch*seq, vocab]`, mask pads with `ignore_index`, and let one `mean` divide by the true token count.

#### Worked example: the reduction that doubled the learning rate

A team finetuning a small transformer set `learning_rate=2e-4`, batch size 16, and the run trained fine. They refactored the training loop and, copying a snippet from a regression tutorial, wrote `loss = F.cross_entropy(logits, labels, reduction='sum')`. The next run spiked: loss climbed from 2.3 to 8.1 in forty steps, then went to `NaN`. The grad norm, which had sat around 1.5, was reading 24 on the first logged step. The instruments said "exploding gradient" and the team spent an afternoon adding gradient clipping, which masked the symptom but left the model under-trained.

The actual fix was one word. With `reduction='sum'` and batch size 16, the effective learning rate was $16 \times 2\times10^{-4} = 3.2\times10^{-3}$, sixteen times the intended value — a textbook too-high-LR divergence. The confirming test was a thirty-second experiment: log the gradient norm under `reduction='mean'` (got 1.5) and under `reduction='sum'` (got 24, almost exactly $16\times$). The ratio *being the batch size* is the fingerprint. They switched back to `mean`, removed the clipping crutch, and the run trained as before. The lesson: when grad norm is suspiciously large by a factor that equals your batch size, look at the reduction before you blame the optimizer.

### The gradient-accumulation version of the reduction bug

The reduction bug has a close cousin that bites teams scaling to larger effective batch sizes: **gradient accumulation with the wrong normalization**. The pattern is to run several micro-batches forward-and-backward, accumulating gradients, then step the optimizer once. If each micro-batch computes a `mean` loss and you accumulate over $K$ micro-batches without dividing, you have summed $K$ means, so your effective gradient is $K\times$ too large — the same effective-LR inflation as `reduction='sum'`, just spread across accumulation steps. The correct recipe is to divide each micro-batch loss by $K$ before `backward`, so the accumulated gradient equals the gradient of the mean over the full effective batch. The subtlety that catches people is that **a per-token mean does not commute with accumulation** when micro-batches have different token counts: averaging four `mean` losses gives each micro-batch equal weight regardless of how many real tokens it contains, which is not the same as a single mean over all tokens. To make accumulation exactly equal to a bigger batch, you must accumulate the *summed* token loss and divide by the *total* token count across all micro-batches, not average the per-micro-batch means. This is why the proof "accumulation $\equiv$ a bigger batch" only holds when the normalization is done over tokens, not over micro-batches — the same trap that the input-pipeline and accumulation posts return to.

There is also the `reduction='none'` case, which is not a bug but a tool. When you want per-example loss weighting — focal loss, importance weighting, hard-example mining, or just logging the loss distribution across a batch — you set `reduction='none'`, get the `[N]` (or `[N, T]`) tensor of per-element losses, apply your weights, and reduce *yourself*. The bug here is forgetting to reduce at all (you call `.backward()` on a non-scalar and PyTorch errors, which is at least loud), or reducing with `.sum()` when you meant `.mean()` and silently reintroducing the batch-size scaling. The discipline is the same: whatever you do with a `reduction='none'` tensor, make sure the final scalar you call `.backward()` on is the *mean over the units you intend to weight equally*, and unit-test that scalar against the built-in `mean` when the weights are all 1.

## 4. Padding and `ignore_index`: loss on tokens that should not count

Sequence models pad. Because a batch contains sequences of different lengths and tensors must be rectangular, you pad the short ones up to the longest with a `[PAD]` token. Those pad positions are not real data — they are filler — and they must not contribute to the loss. If they do, you are averaging your loss over a mix of real tokens and meaningless padding, which dilutes the signal and, worse, teaches the model to "predict" pad tokens that will never appear at inference.

### The mechanism and the magnitude

PyTorch's `CrossEntropyLoss` and `F.cross_entropy` take an `ignore_index` argument. Setting `ignore_index=-100` tells the loss to skip any position whose target label is `-100`: those positions contribute zero to both the loss and the gradient, and they are excluded from the denominator of the `mean`. The Hugging Face convention is exactly this — `-100` is the canonical "ignore" label, and `DataCollatorForLanguageModeling` sets pad-position labels to `-100` for you. The bug is forgetting to set it, so pad positions carry a real label (often `0`, the first vocab id, or the pad token id) and the loss tries to predict them.

The magnitude is easy to quantify and it explains the "loss too low, metric flat" signature. Suppose 70 percent of the positions in your batch are padding (common with high length variance and naive right-padding). Pad positions are trivially predictable — they are all the same token — so the model drives their per-token loss to near zero within a few steps. Your reported loss is then a weighted average: $0.3 \times (\text{real-token loss}) + 0.7 \times (\approx 0)$. If the real-token loss is $3.1$, your reported loss reads $\approx 0.93$ — looking like a well-trained model while the real-token loss has not moved. The curve descends (as the pad fraction is learned away), the perplexity on real text is flat, and you ship a model that learned to predict padding. This is the loss-side twin of the dedicated [loss masking bug](/blog/machine-learning/debugging-training/the-loss-masking-bug) post, which goes deeper on prompt-versus-completion masking and label shifting in causal LMs.

![A before-and-after comparison showing pad tokens counted in the loss diluting it to a low value with flat validation perplexity versus ignore_index set to negative one hundred giving an honest higher loss and falling perplexity](/imgs/blogs/loss-function-bugs-6.png)

### The masked-loss reference implementation

Here is a correct masked cross-entropy for a causal LM, written out explicitly so you can see exactly what `ignore_index` does and verify it against the built-in. The two should agree to numerical precision; if they do not, your masking is wrong.

```python
import torch
import torch.nn.functional as F

def masked_ce_reference(logits, labels, ignore_index=-100):
    """Explicit masked cross-entropy. logits: [B, T, V], labels: [B, T].
    Positions with label == ignore_index contribute nothing."""
    B, T, V = logits.shape
    logits = logits.reshape(-1, V)            # [B*T, V]
    labels = labels.reshape(-1)               # [B*T]
    mask = labels != ignore_index             # True where the token is real
    if mask.sum() == 0:
        raise ValueError("Every label is masked; loss has no real tokens.")
    # Clamp ignored labels to a valid index so gather doesn't error,
    # then zero them out via the mask.
    safe_labels = labels.clone()
    safe_labels[~mask] = 0
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(1, safe_labels.unsqueeze(1)).squeeze(1)  # [B*T]
    nll = nll[mask]                           # keep only real tokens
    return nll.mean()                         # divide by real-token count

# Verify against PyTorch's built-in:
torch.manual_seed(0)
logits = torch.randn(2, 5, 7)
labels = torch.randint(0, 7, (2, 5))
labels[0, 3:] = -100                          # pad the first sequence
labels[1, 4:] = -100
ref = masked_ce_reference(logits, labels)
builtin = F.cross_entropy(
    logits.reshape(-1, 7), labels.reshape(-1), ignore_index=-100
)
print(ref.item(), builtin.item())             # should match to ~1e-6
assert torch.allclose(ref, builtin, atol=1e-5)
```

The critical detail is the denominator. The reference divides by `mask.sum()` (the real-token count), and so does the built-in when `ignore_index` is set. If you instead divide by the total position count `B*T`, you get the diluted loss from the previous paragraph — a number that is too low by exactly the ratio of real tokens to total positions. A fast detector for "are my pads masked?" is to perturb a single pad position's logits and check that the loss does not change: if masking is correct, pad positions have zero gradient and zero loss contribution.

```python
# Detector: does the loss ignore pad positions?
logits2 = logits.clone()
logits2[0, 4, :] += 100.0       # wildly perturb a position whose label is -100
loss_a = F.cross_entropy(logits.reshape(-1,7), labels.reshape(-1), ignore_index=-100)
loss_b = F.cross_entropy(logits2.reshape(-1,7), labels.reshape(-1), ignore_index=-100)
assert torch.allclose(loss_a, loss_b), "Pad position changed the loss -> not masked!"
```

#### Worked example: the finetune that "trained" but learned nothing

A team finetuned a 1.3B-parameter decoder-only model on instruction data with a custom data collator. They right-padded every sequence to the batch maximum and built `labels` by copying `input_ids` — including the pad positions, whose label was the pad token id `0`. The training loss looked great: it started around 2.4 and fell smoothly to 0.6 over one epoch, a textbook curve. But every downstream eval was flat — the finetuned model scored within noise of the base model on the held-out instruction set. The loss said the model had learned a lot; the eval said it had learned nothing.

The bisection was quick once they distrusted the loss. The batches were heavily padded: instruction examples ranged from 12 to 480 tokens, so a batch padded to 480 was often 60-to-80 percent pad. With pad positions carrying a real label, the model learned the trivial task "predict token 0 after token 0" almost instantly, driving the per-pad-token loss to near zero. The reported loss was then $\approx 0.25 \times (\text{real loss}) + 0.75 \times (\approx 0)$, which is why 2.4 fell to 0.6 — most of the "improvement" was the pad fraction being learned away, not the instructions. The confirming test was the perturb-a-pad-position check from the code above: perturbing a pad logit *did* change the loss, proving the pads were not masked. The fix was to set pad-position labels to `-100` in the collator (or just use `DataCollatorForLanguageModeling`, which does it for you). After the fix the *real-token* loss was 3.1 at the start and fell to 1.4, the curve looked worse on paper but was now honest, and the downstream eval finally moved — exactly the before-and-after the figure above shows. The lesson is the recurring one: a finetune whose loss looks suspiciously low on padded data is almost always training on pad or prompt tokens, and the per-token-honest loss is *higher*, not lower. The completion-versus-prompt masking variant of this exact bug is the subject of [the loss masking bug](/blog/machine-learning/debugging-training/the-loss-masking-bug).

## 5. Target dtype, shape, and the silent broadcast

The fourth class of bug is the quietest, because it often does not error — it broadcasts. PyTorch's losses are particular about the dtype and shape of the target, and getting them wrong sometimes raises a clear exception and sometimes silently computes the wrong thing.

The dtype rule for `CrossEntropyLoss` is exact: the target is either **class indices** of dtype `long` with shape `[N]` (or `[N, d1, d2, ...]` for per-pixel tasks), *or* **class probabilities** of dtype `float` with shape `[N, C]` (the soft-label form added in newer PyTorch). If you pass a float index tensor where a long index tensor is expected, you get an error. But if you pass a `[N, C]` one-hot float tensor, `CrossEntropyLoss` interprets it as soft labels — which is *correct* if you meant one-hot, but a trap if you also applied label smoothing, because you will smooth labels that are already soft. The classic shape bug is target shape mismatch in `BCEWithLogitsLoss`: it expects the target to have the *same shape* as the input, and broadcasting rules can turn a `[N]` target against a `[N, 1]` input into an `[N, N]` comparison that computes $N^2$ loss terms instead of $N$. The loss is finite, the gradient is wrong, and nothing crashes. This is the same silent-broadcasting family covered in depth in [shape bugs and silent broadcasting](/blog/machine-learning/debugging-training/shape-bugs-and-silent-broadcasting); here the consequence lands specifically in the loss.

The defense is a shape-and-dtype assert at the top of your loss computation. It costs nothing and catches the bug at step 1:

```python
def assert_ce_targets(logits, target):
    """CrossEntropyLoss target sanity. logits: [N, C] or [N, C, d...]."""
    N, C = logits.shape[0], logits.shape[1]
    if target.dtype == torch.long:
        # Class-index form: target must NOT carry the class dimension.
        assert target.shape[0] == N, f"target batch {target.shape} != {N}"
        assert target.dim() == logits.dim() - 1, (
            f"index target should have one fewer dim than logits; "
            f"got target {tuple(target.shape)} vs logits {tuple(logits.shape)}"
        )
        assert int(target.max()) < C and int(target.min()) >= 0, (
            f"class index out of range [0,{C-1}]: "
            f"min={int(target.min())} max={int(target.max())}"
        )
    elif target.is_floating_point():
        # Soft-label form: target must match logits shape exactly.
        assert target.shape == logits.shape, (
            f"soft-label target {tuple(target.shape)} must match "
            f"logits {tuple(logits.shape)}"
        )
    else:
        raise TypeError(f"target dtype {target.dtype} is neither long nor float")
```

The `target.max() < C` check deserves special attention because it catches the **class-index off-by-one**, which is its own subtle bug. If your dataset labels classes `1..C` (one-indexed, common when labels come from a database or a one-indexed annotation tool) and you pass them straight to `CrossEntropyLoss`, which expects `0..C-1`, then class `C` is out of range. Sometimes this errors ("Target N is out of bounds"); but if you also sized your output layer to `C+1` classes "to be safe," it does *not* error — it trains a phantom class 0 that no example ever has, wastes capacity, and shifts every prediction. The model learns, the loss descends, and your class labels are all off by one at inference. The assert above turns this into an immediate, named failure.

#### Worked example: the one-hot that smoothed twice

A practitioner moved a multi-class classifier from one-hot targets to PyTorch's built-in `label_smoothing`. They left their data pipeline producing one-hot float targets of shape `[N, C]` and added `label_smoothing=0.1` to `CrossEntropyLoss`. The loss looked fine — it descended — but validation accuracy plateaued two points below the previous run, and the model's confidence histogram was visibly flatter than expected. The bug: their one-hot targets had *already* been built with smoothing in the data pipeline (a `0.9 / 0.011...` distribution rather than a hard `1.0 / 0.0`), and `CrossEntropyLoss(label_smoothing=0.1)` smoothed them *again*. The effective smoothing was roughly $0.1 + 0.1 - $ a cross term $\approx 0.19$, nearly double the intended $0.1$. Over-smoothing makes the optimal predicted distribution flatter, which caps confidence and costs a couple of accuracy points. The confirming test was a one-liner: print `target[0]` and check whether it is hard one-hot (`[0,1,0,...]`) or already soft (`[0.011, 0.9, 0.011, ...]`). It was soft. The fix was to either pass hard indices and let `CrossEntropyLoss` do all the smoothing, or keep the soft targets and set `label_smoothing=0.0`. Either way the accuracy recovered.

## 6. Label smoothing, sign errors, and maximizing by accident

Two more single-line bugs round out the "the loss optimizes the wrong scalar" family.

**Label smoothing applied twice**, the bug from the worked example above, deserves its own science note because the doubling is not exactly additive. Label smoothing with parameter $\epsilon$ replaces the hard target $[0,\dots,1,\dots,0]$ with $[\epsilon/C, \dots, 1-\epsilon+\epsilon/C, \dots, \epsilon/C]$ — it moves $\epsilon$ of the probability mass off the true class and spreads it uniformly. Apply smoothing $\epsilon_1$ in the pipeline and $\epsilon_2$ in the loss, and the true-class target becomes approximately $(1-\epsilon_1)(1-\epsilon_2)$, so the *effective* smoothing is $\epsilon_{\text{eff}} = 1 - (1-\epsilon_1)(1-\epsilon_2) = \epsilon_1 + \epsilon_2 - \epsilon_1\epsilon_2$. For $\epsilon_1 = \epsilon_2 = 0.1$ that is $0.19$, as quoted earlier. The signature of over-smoothing is a model that is systematically under-confident: its maximum softmax probability is capped well below 1 even on examples it gets right, and its calibration looks "too humble." The right $\epsilon$ is applied exactly once, and the standard value in the literature is $0.1$ from the Inception-v3 / "Rethinking the Inception Architecture" work that popularized it.

**The sign error** is the most embarrassing and the easiest to miss in custom losses. Standard losses are written to be *minimized*: cross-entropy is $-\sum y \log p$, and minimizing it maximizes the log-likelihood of the data. But when you write a custom loss — a contrastive term, a custom reward, a regularizer with a sign — it is alarmingly easy to drop or add a minus sign and hand the optimizer a quantity it should be *maximizing*. The optimizer dutifully minimizes it, which means it *worsens* the thing you wanted. The signature is unmistakable once you know it: **the loss goes up while the metric goes down**, or the loss decreases into large negative numbers without bound (minimizing $+\log p$ instead of $-\log p$ drives $\log p \to -\infty$, i.e. $p \to 0$, the wrong direction). A loss that trends negative and unbounded is almost always a sign error. The check is the same hand-computed unit test we keep returning to: feed the loss a *perfect* prediction and assert it returns *approximately zero* (or the known minimum); feed it a *worst-case* prediction and assert the loss is *larger*. If "perfect" gives a large or negative loss, your sign is wrong.

This connects to a broader truth the series keeps hammering: your loss and your metric must move in opposite-but-consistent directions (loss down, metric up). When they decouple — loss down, metric flat or down — you have either a loss bug (this post) or a metric bug, which is the subject of [your metric is lying](/blog/machine-learning/debugging-training/your-metric-is-lying). Distinguishing the two is itself a useful bisection: compute the metric *from the same predictions the loss sees*, on a tiny batch you understand, and check by hand which one is wrong.

## 7. Multi-task loss weighting and regression scale

The remaining two bugs are about *scale*, and they appear whenever your loss is a sum of terms with different natural magnitudes.

### Multi-task weighting on incompatible scales

A detection or multi-task model combines, say, a classification cross-entropy and a bounding-box regression loss: $L = L_{\text{cls}} + \lambda L_{\text{reg}}$. The trap is that the two terms can live on wildly different scales. Cross-entropy is on a log scale and typically sits between 0 and a few. A regression loss on raw pixel coordinates can be in the hundreds or thousands (a sum of squared errors over coordinates measured in pixels). If you naively sum them with $\lambda = 1$, the regression term dominates the gradient by orders of magnitude, the classification head is effectively starved, and your classifier never trains — even though the *total* loss descends nicely (because the dominant regression term descends). The science is just gradient arithmetic: $\nabla L = \nabla L_{\text{cls}} + \lambda \nabla L_{\text{reg}}$, and if $\|\nabla L_{\text{reg}}\| \gg \|\nabla L_{\text{cls}}\|$, the update direction is essentially $\nabla L_{\text{reg}}$ alone. The classification metric flatlines while the box metric improves and the total loss falls — the decoupling signature again, now between two heads.

![A branching graph showing a classification loss and a much larger regression loss summing into a total dominated by the regression term, starving the classification gradient until weighting balances the scales](/imgs/blogs/loss-function-bugs-7.png)

The fix is to **normalize the scales before you weight**. Either normalize each loss by its own running magnitude, choose $\lambda$ so the two terms are within an order of magnitude at initialization, or use uncertainty-based weighting (Kendall et al., learning per-task weights as homoscedastic uncertainty). The diagnostic is to **log the loss components separately**, never just the total. This is the single most important habit for multi-task training:

```python
# Always log multi-task loss components, not just the sum.
loss_cls = F.cross_entropy(cls_logits, cls_targets)
loss_reg = F.smooth_l1_loss(box_pred, box_targets)   # often huge on raw pixels
loss = loss_cls + lambda_reg * loss_reg

# These prints are the whole diagnostic:
print(f"cls={loss_cls.item():.3f}  reg={loss_reg.item():.3f}  "
      f"weighted_reg={lambda_reg*loss_reg.item():.3f}  total={loss.item():.3f}")
# If weighted_reg >> cls for many steps, the cls head is starved.
```

If `weighted_reg` is 1500 and `cls` is 0.7, your "multi-task" model is a single-task regression model with a vestigial classifier. Choose $\lambda$ (or normalize the box loss by image size) so the weighted terms are comparable.

#### Worked example: the detector whose classifier never trained

A team built a small object detector with the standard two-head loss: a classification cross-entropy over object categories and an L1 box-regression loss on un-normalized corner coordinates in pixels. They summed the two with $\lambda = 1$ and trained. The total loss fell from 1200 to 90 over the first epoch — a dramatic, satisfying drop. But the mean average precision was stuck: boxes were being placed roughly correctly, yet the class labels on those boxes were near random, so mAP hovered around 0.12. The team suspected the anchor assignment, the NMS, even the dataset, and spent two days there.

The diagnostic that ended it was the component log above. Printing `cls` and `reg` separately for ten steps showed `cls` pinned around 0.69 (exactly $\ln 2$ for their binarized "object vs not" head, i.e. the classifier was outputting chance) while `reg` fell from 1200 to 89. The total was 99.7 percent box loss. The classifier's gradient was being swamped: $\|\nabla L_{\text{reg}}\|$ was three orders of magnitude larger than $\|\nabla L_{\text{cls}}\|$, so the update direction was effectively pure box regression and the classification head received almost no learning signal. The confirming experiment was to zero the box loss for one run ($\lambda = 0$) and watch `cls` immediately start falling from 0.69 toward 0.2 — proof the classifier *could* learn when it was not being drowned. The fix was to normalize the box loss by the image diagonal (bringing `reg` from ~1000 to ~1.5) and set $\lambda = 1$, so both terms sat near 1 at initialization. After the fix, both heads trained, `cls` fell to 0.18, and mAP climbed from 0.12 to 0.61. The lesson generalizes to every multi-task and multi-loss setup, including segmentation (a per-pixel cross-entropy plus a Dice or boundary loss) and VAEs (a reconstruction term plus a KL term whose scales differ by orders of magnitude): **log every loss component, and balance the scales at initialization before you trust the total.**

### Regression loss scale versus target scale

The single-task version of the same problem: a regression loss whose scale is mismatched to the target scale. If your targets are house prices in the hundreds of thousands and you use mean-squared error, the per-example loss is in the billions, the gradients are enormous, and you must either use a microscopic learning rate or watch the run diverge. Conversely, if your targets are normalized to unit variance but you forgot and your loss is computed on raw values, the same mismatch appears. The fix is standard: **standardize regression targets** (subtract the mean, divide by the standard deviation) before computing the loss, and invert the transform at inference. The science is identical to the reduction bug — the gradient scale is proportional to the loss scale, so a loss measured in billions implies a gradient measured in billions, which the optimizer multiplies by your learning rate. The honest diagnostic is to print the loss value on the first batch: if a freshly initialized regression model reports a loss of $10^9$, your targets are unnormalized, full stop.

There is a subtler scale bug hiding inside the *choice* of regression loss, not just the target normalization. Mean-squared error has a gradient that is *linear* in the residual ($\partial \frac{1}{2}(\hat y - y)^2 / \partial \hat y = \hat y - y$), so a single large-residual outlier produces a proportionally large gradient that can dominate the batch update — one mislabeled target of $10^6$ in a sea of targets near 1 will yank the model toward it. Mean-absolute error (L1) has a constant-magnitude gradient ($\pm 1$), which is robust to outliers but non-smooth at zero and slow to converge near the optimum. The Huber / smooth-L1 loss is the common compromise: quadratic for small residuals (smooth, fast) and linear for large ones (outlier-robust), with a threshold $\delta$ that sets the crossover. The loss bug here is choosing the wrong member of this family for your noise profile and then blaming the optimizer when training is unstable: MSE on heavy-tailed targets gives you gradient spikes that look like exploding gradients but are actually a small number of outliers driving the loss; switching to Huber or L1 makes them vanish. The diagnostic is to plot the per-example loss distribution (`reduction='none'`, then a histogram): if a handful of examples carry most of the loss, your loss is outlier-dominated and either the targets have errors (a data bug — see the label-noise post) or you want a robust loss. This is one more instance of the unifying lesson: the loss you choose *defines* the objective, and an objective that over-weights outliers will optimize toward them confidently while your robust-error metric gets worse.

It is worth pausing on what all eight bugs share, because the shared structure is the real takeaway. In every case the loss is a *finite, descending number* — none of these bugs (except the fp16 `log 0`) crashes or NaNs by itself. That is precisely why they are dangerous: the optimizer's only job is to drive the loss down, and it does, faithfully and well. The bug is never in the optimization; it is in the *definition of the scalar* the optimizer was handed. A double softmax defines a flattened objective; a `sum` reduction defines an objective scaled by batch size; unmasked pads define an objective that includes predicting filler; a sign error defines an objective to be maximized; a scale mismatch defines an objective dominated by one term or one outlier. The optimizer cannot know any of this — it has no access to the metric you care about, only to the number you gave it. This is the deepest reason the diagnostic is always the same: do not watch the loss *descend* (it always will), watch whether the loss you defined is the loss you *meant*, by computing it on inputs small enough to verify by hand.

## 8. The one check that catches all of them: unit-test the loss

Every bug in this post — double softmax, wrong reduction, unmasked pads, off-by-one indices, double smoothing, sign errors, scale mismatches — is caught by the same discipline, and it is the single highest-leverage practice in loss debugging: **unit-test the loss on tiny, hand-computed inputs.** You compute the expected loss with a pencil, you assert your code returns it, and an entire taxonomy of bugs becomes impossible to ship.

![A left-to-right timeline of the unit-test-the-loss workflow, from a tiny hand-built input through a hand-computed cross-entropy value, an allclose assertion, a perfect-prediction check, a masking check, and a passing test](/imgs/blogs/loss-function-bugs-8.png)

The reason this works is that loss bugs are *deterministic functions of small inputs*. Unlike a data leak or a distributed desync, a loss function with three classes and two examples produces an exact number you can verify by hand. There is no randomness to average out, no scale to worry about — just $-\log p_y$ on numbers you control. If the hand-computed value and the code disagree, the loss is wrong; there is nothing else it could be.

Here is the canonical loss unit-test suite. It encodes the hand-computed values for the cases that catch the common bugs, and it is short enough to keep in every project:

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def test_cross_entropy_hand_computed():
    # Uniform logits over 3 classes -> loss must equal ln(3) = 1.0986.
    logits = torch.zeros(2, 3)                 # all-equal -> uniform softmax
    target = torch.tensor([0, 1])
    loss = F.cross_entropy(logits, target)
    assert math.isclose(loss.item(), math.log(3), abs_tol=1e-4), loss.item()

def test_perfect_prediction_near_zero():
    # A confident-correct logit should give near-zero loss.
    logits = torch.tensor([[10.0, 0.0, 0.0]])  # class 0 dominant
    target = torch.tensor([0])
    loss = F.cross_entropy(logits, target)
    assert loss.item() < 1e-3, loss.item()     # catches sign errors & double softmax

def test_worst_prediction_is_large():
    # Confidently wrong should give a large loss (> chance).
    logits = torch.tensor([[0.0, 0.0, 10.0]])  # predicts class 2
    target = torch.tensor([0])                  # truth is class 0
    loss = F.cross_entropy(logits, target)
    assert loss.item() > math.log(3), loss.item()

def test_reduction_sum_is_mean_times_n():
    logits = torch.randn(8, 5)
    target = torch.randint(0, 5, (8,))
    m = F.cross_entropy(logits, target, reduction='mean')
    s = F.cross_entropy(logits, target, reduction='sum')
    assert math.isclose(s.item(), m.item() * 8, rel_tol=1e-5)  # ratio == batch size

def test_ignore_index_drops_tokens():
    logits = torch.randn(4, 7)
    target = torch.tensor([3, -100, 5, -100])   # two ignored
    loss_all = F.cross_entropy(logits[[0, 2]], target[[0, 2]])
    loss_msk = F.cross_entropy(logits, target, ignore_index=-100)
    assert torch.allclose(loss_all, loss_msk, atol=1e-5)  # masked == real-only

for fn in [test_cross_entropy_hand_computed, test_perfect_prediction_near_zero,
           test_worst_prediction_is_large, test_reduction_sum_is_mean_times_n,
           test_ignore_index_drops_tokens]:
    fn()
    print(f"PASS  {fn.__name__}")
```

The first test — uniform logits must give $\ln C$ — is the most valuable single assertion in the suite. A freshly initialized classifier with small random weights produces near-uniform logits, so its very first loss should be approximately $\ln C$: $\ln 2 \approx 0.69$ for binary, $\ln 10 \approx 2.30$ for 10 classes, $\ln 1000 \approx 6.91$ for ImageNet. **If your initial loss is not near $\ln C$, something is wrong before step one.** An initial loss of $0.35$ on a 10-class problem (far below $\ln 10$) screams that your loss is too low — pads unmasked, or a metric measured instead of a loss. An initial loss of $4.6$ on a binary problem (far above $\ln 2$) screams a scale or sign problem. This one number, checked before training, is the cheapest sanity test in deep learning.

| Classes $C$ | Expected initial loss $\ln C$ | What a different value means |
| --- | --- | --- |
| 2 (binary) | 0.69 | $\gg 0.69$: scale/sign bug; $\ll 0.69$: leakage or unmasked |
| 10 | 2.30 | $\approx 0.4$: pads unmasked or double softmax floor |
| 1000 (ImageNet) | 6.91 | $\approx 2$: wrong reduction or soft-label mismatch |

The "perfect prediction → near-zero loss" and "worst prediction → large loss" pair is the sign-error detector. The "sum = mean × N" assertion is the reduction detector. The "ignore_index drops tokens" assertion is the masking detector. Five short tests, run in under a second, that between them catch the majority of loss bugs in this post.

## 9. The before-and-after evidence

Let me make the central claim concrete with measured-style numbers, because the whole point of this series is the before→after proof. The figure earlier in the post sketched the double-softmax case; here is the full instrument readout for the canonical fixes, the kind of table you would assemble from your own dashboard before and after a one-line change.

| Bug | Before (instrument reading) | After fix | The fix |
| --- | --- | --- | --- |
| Double softmax (10-class CV) | loss floors at 1.9, grad norm 0.02, acc 38% | loss → 0.3, grad norm 1.4, acc 92% | remove `softmax` from `forward` |
| `reduction='sum'` (LM finetune, bs 16) | grad norm 24, loss spikes 2.3→8.1→NaN | grad norm 1.5, loss 2.3→1.1 stable | `reduction='mean'` |
| Pads unmasked (LM, 70% pad) | loss 0.93, val PPL flat at 110 | loss 3.1, val PPL 110→34 | `ignore_index=-100` |
| Class off-by-one (segmentation) | mIoU 0.41, one phantom class | mIoU 0.67, classes aligned | shift labels to 0-indexed |
| Double label smoothing | acc plateau −2 pts, flat confidence | acc recovers, calibrated | smooth once, $\epsilon=0.1$ |
| Sign error (custom contrastive) | loss trends to −40, metric drops | loss → 0.2, metric recovers | negate the term |
| Multi-task imbalance | cls acc flat, reg good, total falls | both heads train | normalize/weight $\lambda$ |

Every row shares the same diagnostic story: an instrument that looked plausible-or-only-slightly-off (a smooth loss, a grad norm that could be "just a big batch") combined with a metric that refused to track it. The *confirming test* in each case is the hand-computed unit test or the component log, and the *fix* is one to three lines. The before→after numbers here are representative magnitudes for these well-known failure modes, not measurements from a single benchmark run; the *shapes* (loss floor near chance for double softmax, grad norm $\approx N\times$ for the reduction bug, loss diluted by the pad fraction) are exact and reproducible from the math in the preceding sections.

### How to measure this honestly

The discipline that makes these before→after numbers trustworthy is to **change exactly one thing and read the same instruments**. For the double softmax: log the gradient norm and the maximum softmax probability before and after removing the activation; the grad norm should jump by one to two orders of magnitude and the max probability should rise from $\approx 1/C$ toward 1. For the reduction bug: log grad norm under `mean` and `sum` on the *same batch* and confirm the ratio equals the batch size — this is a deterministic check, not a training run. For pads: perturb a pad logit and confirm the loss is unchanged (masked) or changed (not masked). For the off-by-one: print the confusion matrix and look for an empty row/column (the phantom class) or a consistent diagonal shift. None of these require a full training run; they are unit-scale checks that confirm the bug and its fix in seconds.

The mandatory matrix below is the lookup table for the whole post: each bug, the instrument signature that betrays it, the one-line unit test that confirms it, and the fix.

![A matrix mapping four loss bugs against their symptom, their hand-computed unit test, and their one-line fix](/imgs/blogs/loss-function-bugs-5.png)

## 10. Case studies and real signatures

These patterns are not hypothetical; they are among the most frequently reported issues in real codebases and forums. Here are five with their documented signatures.

**The `BCELoss` versus `BCEWithLogitsLoss` confusion (PyTorch forums, recurring).** Easily one of the most-asked PyTorch questions: a binary classifier with a `sigmoid` at the end of `forward` and `BCEWithLogitsLoss` as the criterion, producing a model that trains slowly and saturates early — the binary double-sigmoid. The PyTorch documentation explicitly warns that `BCEWithLogitsLoss` "combines a Sigmoid layer and the BCELoss in one single class" and is "more numerically stable than using a plain Sigmoid followed by a BCELoss." The fix, repeated thousands of times in answers, is to remove the `sigmoid` from the model. The numerical-stability point is real: `BCEWithLogitsLoss` uses the log-sum-exp trick to avoid the `log(0)` that a saturated `sigmoid` feeds to `BCELoss`.

**Hugging Face `-100` masking, the canonical convention.** Across the `transformers` library, `-100` is the universal "ignore this position" label. `DataCollatorForLanguageModeling` sets pad-position labels to `-100`, the `Trainer` passes them through, and `CrossEntropyLoss` (with its default `ignore_index=-100`) skips them. When people finetune with a *custom* collator and forget to set `-100`, they train on pad and prompt tokens — the dedicated [loss masking bug](/blog/machine-learning/debugging-training/the-loss-masking-bug) post documents the "training on the prompt" variant. The detector is the perturb-a-pad-position test above; the signature is a suspiciously low loss with flat downstream eval.

**Label smoothing from "Rethinking the Inception Architecture" (Szegedy et al., 2016).** Label smoothing with $\epsilon = 0.1$ was introduced and popularized in the Inception-v3 paper as a regularizer that improved top-1 and top-5 ImageNet error. It is now built into `CrossEntropyLoss` via `label_smoothing=`. The double-smoothing bug appears when teams migrate from a manual smoothing implementation in their data pipeline to the built-in argument without removing the manual version — exactly the worked example in section 5. The signature is systematic under-confidence and a small accuracy regression.

**The "loss goes down, accuracy stuck at chance" classifier (universal).** This is the double-softmax archetype and the most common "my model won't learn" post on every ML forum. A model with `nn.Softmax(dim=1)` (or `F.log_softmax` plus `CrossEntropyLoss` instead of `NLLLoss`) as its final layer, paired with `CrossEntropyLoss`, trains to a loss floor near $\ln C$ and an accuracy near chance. The fix — delete the final activation — turns a six-hour wasted run into a model that learns. The reliable detector is the initial-loss check: a model whose loss can never get meaningfully below $\ln C$ has its confidence structurally capped by a double activation.

**The segmentation `ignore_index` and class-index off-by-one (recurring in CV codebases).** Semantic segmentation is a per-pixel classification, so it inherits every loss bug above and adds its own. Two are endemic. First, the "void" or "unlabeled" class: many segmentation datasets reserve a label (often 255 or a dataset-specific id) for pixels that are ambiguous or outside the annotation, and that label must be passed as `ignore_index` so those pixels do not contribute to the loss; forget it and the model is penalized for "mis-predicting" pixels that have no ground truth, which floors the IoU and biases the model toward the background class. Second, the off-by-one: datasets that number classes `1..K` with `0` meaning background must be reconciled with a model that outputs `K` or `K+1` channels, and a mismatch produces a phantom class and a systematic label shift. The signature is a mean IoU that is depressed across the board with one class at near-zero IoU (the phantom), and the fix is to align the label space (0-indexed, with the void id routed to `ignore_index`). The dedicated segmentation-debugging post in this series and the IoU discussion go deeper; the loss-side rule is that a per-pixel cross-entropy must mask the void class and use a 0-indexed, contiguous label space, verified by the same hand-computed unit test on a tiny mask.

Each of these is an instance of the same master frame: a loss that descends while the metric does not, traced by bisection to the loss layer, confirmed by a tiny hand-computed test, and fixed in one line. For the full symptom→suspect→test→fix decision tree that places loss bugs among all the others, see [a taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs).

## 11. The bisection in action: a stalled run, debugged

Let me walk the full problem-solving narrative on the opening run — the 10-class classifier whose loss fell from 1.10 to 0.55 while accuracy sat at 38 percent — because the *process* matters more than any single fix.

**Symptom.** Loss descending, accuracy stuck near chance. The decoupling of loss and metric is the headline; it immediately points away from optimization (the loss *is* descending, so the optimizer works) and toward either the loss or the metric.

**Bisect: is it the loss or the metric?** Compute accuracy by hand on one batch of ten examples using `argmax(logits)`. If the hand-computed accuracy matches the reported 38 percent, the metric is correct and the loss is suspect. It matched. The metric is fine; the loss is lying. (If the hand-computed accuracy had been 90 percent while the dashboard said 38, the bug would be in the metric, and we would be in [your metric is lying](/blog/machine-learning/debugging-training/your-metric-is-lying) territory instead.)

**Read the instrument: initial loss.** Restart the model and read the loss at step 0. It was 1.10. For 10 classes, $\ln 10 \approx 2.30$. The initial loss is *less than half* of $\ln C$ — impossible for a correctly initialized classifier on balanced data, because a random classifier *must* be near $\ln C$. A sub-$\ln C$ initial loss means the loss is structurally too low: either the targets are wrong, or the probabilities are being flattened (which lowers the *effective* dynamic range of the loss). Combined with "accuracy is at chance," this points at the double softmax — a second softmax both flattens the predictions (chance accuracy) and compresses the loss range (sub-$\ln C$ readings as logits drift).

**Confirm with the unit test.** Run `test_perfect_prediction_near_zero` against the *model's* output path, not raw `F.cross_entropy`: build a logit that should be perfectly correct, push it through the model's final layers (including the suspect `softmax`), and through the loss. The "perfect" prediction gave a loss of 0.71 instead of near zero. A perfect prediction cannot give 0.71 unless the prediction is being flattened before scoring. Confirmed: double softmax.

**Fix.** One line: remove `nn.Softmax(dim=1)` from the model's `forward`; keep `CrossEntropyLoss` on the raw logits.

**After.** Initial loss rose to 2.31 ($\approx \ln 10$, exactly as it should for a random 10-class model — *higher* initial loss is the healthy sign here). Within an epoch the loss fell to 0.3 and accuracy climbed to 92 percent, now tracking the loss. Grad norm went from 0.02 to 1.4. The run that had "trained" uselessly for six hours converged in twenty minutes.

### Stress tests: what if it is not the double softmax?

The discipline is to keep bisecting if the first hypothesis fails. *What if the initial loss had been correct at $\ln C$ but accuracy was still stuck?* Then the loss is not flattening predictions, so suspect the data pipeline (labels shuffled relative to inputs — overfit one batch to check) or the reduction/masking. *What if the loss trended negative?* Sign error — run the perfect-prediction test, which would return a large negative number. *What if the loss spiked then NaN'd?* That is numerics/optimization, not a quiet loss bug — check the reduction (grad norm $\approx N\times$?) and then go to [hunting NaNs and infs](/blog/machine-learning/debugging-training/hunting-nans-and-infs). *What if it only fails at fp16?* A `BCELoss` on a saturated `sigmoid` can produce `inf` only in fp16 where the sigmoid rounds to exactly 1.0; switch to `BCEWithLogitsLoss`. *What if the batch is tiny?* A reduction bug's effective-LR inflation shrinks (smaller $N$), so a `sum`-reduction run that diverges at batch 64 might "work" at batch 4 — a misleading signal that the bug is gone when you have only changed its magnitude. Each branch is a different one of the six places, reached by reading one more instrument.

## When this is (and isn't) your bug

Be decisive about when to suspect the loss and when to look elsewhere, because chasing a loss bug that does not exist wastes as much time as missing one that does.

**It is a loss bug when:** the loss descends but the metric does not track it; the initial loss is far from $\ln C$ (too low *or* too high) on a balanced classification problem; the grad norm is larger than expected by a factor that equals the batch size (reduction); the loss is suspiciously low and your sequences are heavily padded (masking); the loss trends negative and unbounded (sign error); one head of a multi-task model never trains while the total loss falls (weighting); a fresh regression model reports a loss in the millions or billions (scale). These all point straight at the loss layer.

**It is *not* a loss bug when:** the loss is stuck flat from step 0 and the overfit-one-batch test *also* fails — that is a data-pipeline or gradient-flow problem ([shape bugs and silent broadcasting](/blog/machine-learning/debugging-training/shape-bugs-and-silent-broadcasting) or a frozen submodule), not the loss, because a correct loss on a tiny batch *will* drive to zero if the model and data are wired right. The loss is also not your bug when the curve is smooth and then suddenly NaNs — that is numerics (overflow, $\log 0$), best hunted with [hunting NaNs and infs](/blog/machine-learning/debugging-training/hunting-nans-and-infs). And it is not the loss when train metrics are excellent and only *eval* metrics are bad — that is overfitting, train/eval mode, or a metric/leakage bug in evaluation. The clean separator is the unit test: if your loss passes the hand-computed suite (initial loss $\ln C$, perfect→0, worst→large, sum=mean×N, ignore_index drops tokens), the loss is correct and the bug lives in one of the other five places. Stop blaming the loss and go bisect somewhere else.

There is a useful refinement of this rule that resolves the most common confusion, which is the boundary between a loss bug and an *optimization* bug, because both can present as "the loss moves but training is bad." The discriminator is the *initial* loss combined with the overfit-one-batch test. If the initial loss is at $\ln C$ and the model can overfit a single batch to near-zero loss, the loss function is almost certainly correct — overfitting one batch exercises the full forward-loss-backward path, and a broken loss (double softmax, sign error, wrong reduction-induced divergence) would prevent that batch from reaching zero. So a passing overfit-one-batch test is strong evidence the loss is fine and the problem is in optimization (learning rate, schedule, optimizer state) or generalization (data, regularization). Conversely, if the model *cannot* overfit one batch and the initial loss is wrong, the loss is the prime suspect, because there is no simpler explanation for a model failing the easiest possible learning task. This is why the series puts overfit-one-batch first: it cleanly separates the loss-and-model-code question ("can this thing learn at all?") from the data-and-optimization question ("does it learn the *right* thing on *real* data?"). When you are unsure which of the six places to blame, run overfit-one-batch and read the initial loss before you touch anything; together they rule the loss in or out in under a minute.

One more boundary worth drawing: a loss bug versus a *metric* bug. Both produce the loss-metric decoupling that is this post's headline symptom, and they are distinguished by computing both quantities from the *same* predictions on a tiny, hand-checked batch. Take ten examples you understand, run the model, and compute by hand both the loss (does it match `F.cross_entropy` on those logits and labels?) and the metric (does `argmax(logits)` actually match the labels you think it does?). If the hand loss disagrees with your code's loss, it is a loss bug; if the hand metric disagrees with your dashboard's metric, it is a metric bug ([your metric is lying](/blog/machine-learning/debugging-training/your-metric-is-lying)). If both agree with your code but disagree with *each other* — the loss is genuinely low and the metric is genuinely poor — then neither is buggy and you have a real modeling problem (the loss you chose is a poor proxy for the metric you care about, which is a modeling decision, not a bug). That last case is the rare honest one, and recognizing it stops you from hunting a bug that does not exist.

## Key takeaways

- **Loss down, metric flat is the master tell.** When the curve descends but the metric you care about does not track it, suspect the loss layer first, before optimization or evaluation.
- **Know whether your loss wants logits or probabilities.** `CrossEntropyLoss` and `BCEWithLogitsLoss` apply the activation internally and expect raw logits; a `softmax`/`sigmoid` in `forward` is the double-activation bug. `NLLLoss` wants log-probs, `BCELoss` wants probabilities.
- **Initial loss should be $\ln C$.** A random classifier's first loss is $\ln C$ (0.69 binary, 2.30 for 10 classes, 6.91 for ImageNet). Far from that — high or low — means a bug before step one.
- **`reduction='sum'` multiplies your learning rate by the batch size.** A grad norm that is large by exactly the factor $N$ is a reduction bug masquerading as an exploding gradient. Default to `mean`.
- **Mask pads with `ignore_index=-100`.** Unmasked padding dilutes the loss by the pad fraction; confirm masking by perturbing a pad logit and checking the loss does not change.
- **Check target dtype, shape, and indexing.** Long indices `[N]` versus float soft-labels `[N,C]`; class indices must be 0-indexed and `< C`; a `[N]` target against `[N,1]` logits silently broadcasts to $N^2$ terms.
- **Apply label smoothing exactly once.** Pipeline smoothing plus loss smoothing compounds to $\epsilon_1+\epsilon_2-\epsilon_1\epsilon_2$; the signature is systematic under-confidence.
- **A negative, unbounded loss is a sign error.** Test it: a perfect prediction must give near-zero loss; a worst-case prediction must give a larger one.
- **Log multi-task loss components separately.** A summed objective dominated by a large-scale term silently starves the other head; normalize scales before weighting.
- **Unit-test the loss on tiny hand-computed inputs.** The single highest-leverage check: compute $\ln C$, perfect→0, worst→large, sum=mean×N, and ignore_index by hand, and assert. It kills the whole class before a GPU runs.

## Further reading

- PyTorch documentation, [`torch.nn.CrossEntropyLoss`](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html), [`BCEWithLogitsLoss`](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html), and [`NLLLoss`](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html) — the authoritative reference on what each loss expects (logits vs log-probs vs probs), the internal `log_softmax`/`sigmoid`, `reduction`, `ignore_index`, and `label_smoothing`.
- Szegedy et al., "Rethinking the Inception Architecture for Computer Vision" (2016) — the origin of label smoothing with $\epsilon = 0.1$ and the regularization argument behind it.
- Kendall, Gal, and Cipolla, "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics" (2018) — principled multi-task loss weighting when terms live on different scales.
- Hugging Face `transformers` documentation on data collators and the `-100` masking convention — the canonical pattern for excluding pad and prompt tokens from the loss.
- [A taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) — the master symptom→suspect→test→fix decision tree that places loss bugs among the six places a bug hides.
- [The training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) — the capstone bisection method and the printable checklist that includes the loss unit-test suite.
- [The loss masking bug](/blog/machine-learning/debugging-training/the-loss-masking-bug) — the LLM-specific deep dive on `-100`, prompt-versus-completion masking, and label shifting that complements the `ignore_index` section here.
- [Your metric is lying](/blog/machine-learning/debugging-training/your-metric-is-lying) — the other half of the loss-metric decoupling: when the loss is right and the metric is the bug.

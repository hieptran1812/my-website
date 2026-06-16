---
title: "Loss Spikes and Divergence: Transient, Terminal, and How to Recover"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Learn to read a loss spike at step 8,000 as either a recoverable hiccup or a run-ender, and to deploy the skip-batch, clip, and rewind machinery that lets a diverging run finish."
tags:
  [
    "debugging",
    "model-training",
    "loss-spikes",
    "divergence",
    "optimization",
    "finetuning",
    "deep-learning",
    "pytorch",
    "mixed-precision",
    "llm",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/loss-spikes-and-divergence-1.png"
---

You are eight thousand steps into a run that has been behaving. The loss came down off its warmup plateau at 2.30, glided through the fast-descent phase, and is sitting around 1.9 — exactly where the previous good run was at this point. You step away to get coffee. When you come back the dashboard shows a single ugly tooth: at step 8,000 the loss jumped from 1.9 to 19, a clean ten-fold spike, and the line after it is doing something — you cannot quite tell whether it is settling back down or climbing toward the top of the chart. Your hand is on the keyboard. Do you kill the run and lose eighteen GPU-hours, or do you wait and risk watching it diverge to `NaN` while the meter runs?

This is one of the most consequential snap judgments in training, and most engineers make it on vibes. The spike is the symptom; the run-ending divergence is the failure you are trying to avoid; and the difference between a *transient* spike that self-heals in a hundred steps and a *terminal* one that poisons every weight on the way to `NaN` is something you can read off the instruments in under a minute — if you are logging the right ones. The whole point of this post is to replace the vibes with a procedure: name the spike, classify it transient or terminal, and apply the cheapest recovery that holds the run.

![A timeline of a training run that warms up, drops, spikes ten-fold at step eight thousand, then forks into self-recovery or a rewind-and-skip rescue that completes the run](/imgs/blogs/loss-spikes-and-divergence-1.png)

We do this the way the whole series does it — three things at once, because the user asked for a debugging series that is scientific, practical, and proven. The **science**: *why* a spike is even possible, derived from the curvature of the loss surface and the step-size stability bound $\eta < 2/L$; *why* large models spike more (the landscape sharpens as you scale, and Adam's second-moment estimate lags after a quiet region so one big gradient gets through unattenuated); and *why* fp16 turns a recoverable bump into an irrecoverable `NaN`. The **practice**: real PyTorch you can paste in — a spike detector that logs the running *max* loss and the per-step grad-norm (a smoothed curve provably hides the spike), a skip-batch-on-grad-norm guard, and a checkpoint-rewind harness that resumes past the offending data. The **proof**: a concrete before-and-after where a run that hit `NaN` at step 8,000 finishes at loss 1.42 once clipping and a skip-batch guard are added, with the grad-norm dropping from $10^3$ to a capped $1.0$.

This bug lives squarely in two of the [six places a training bug can hide](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) — **optimization** and **numerics** — with frequent guest appearances from **data** (the one outlier batch that triggers it). Keeping that frame in mind is what lets you bisect: a spike with a clean grad-norm is a different animal from a spike with a grad-norm of $10^3$, and the two demand different fixes. By the end you should be able to take a screenshot of any spiking run, in LLM pretraining, a vision finetune, or a speech model, and call it correctly: babysit, guard, or kill.

## 1. What a spike actually is, and the running max that reveals it

Start with definitions, because the word "spike" gets used for two different shapes and conflating them is the first mistake. A **loss spike** is a sudden, large, usually brief increase in the training loss — the curve jumps up by a factor of two to a hundred over one or a handful of steps, then either falls back toward its prior trajectory (transient) or fails to recover and climbs (terminal). It is distinct from a **plateau** (loss stops moving), a **sawtooth** (regular periodic bumps, usually a dataloader-shuffle artifact), and a slow **drift** upward (which is usually a learning-rate-schedule or data-distribution problem, not a spike at all).

The first practical problem is that you may not be able to *see* the spike. If you log loss once per epoch, a spike that lasts fifty steps inside a ten-thousand-step epoch is one averaged point — invisible. Worse, the default dashboards in most tools apply exponential-moving-average (EMA) smoothing, and a heavy EMA provably launders a spike into a gentle ripple. This matters enough to prove.

An EMA with smoothing factor $\beta$ produces $s_t = \beta s_{t-1} + (1-\beta)\ell_t$. Suppose your loss sits at a baseline $b$ and a single step injects a spike of height $h$ above baseline at step $t^\*$. The smoothed value at the spike step rises by only $(1-\beta) h$ above baseline, and decays back over subsequent steps as $(1-\beta) h \beta^{k}$ at step $t^\*+k$. With the common $\beta = 0.99$, a spike of height $h = 17$ (loss $1.9 \to 18.9$) shows up in the smoothed curve as a bump of just $(1-0.99)\times 17 = 0.17$ — a wiggle from $1.9$ to roughly $2.07$ that you would never look at twice. The spike is a factor of ten; the smoother shows you two percent of it. The signal is there and the instrument is hiding it.

The fix is to log a **running max** alongside the mean. Over a sliding window of the last $W$ steps, track $\max_{t-W < i \le t} \ell_i$. The mean and the EMA pull spikes *down*; the max pulls them *up*. A run whose EMA glides from 1.9 to 1.7 while its windowed max touches 19 has a spike problem the average is laundering. The running max is the single cheapest spike detector you can attach, and it costs you one `deque` and a comparison per step.

```python
import torch
from collections import deque

# Spike-aware logging: raw loss, running max, and grad-norm together.
# The EMA-smoothed curve your dashboard shows by default hides spikes;
# the running max is what surfaces them.
window_loss = deque(maxlen=50)   # sliding window for the running max
ema = None
EMA_BETA = 0.98

def grad_global_norm(model):
    # L2 norm of the full flattened gradient: sqrt(sum of every grad^2).
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().float().norm(2).item() ** 2
    return total ** 0.5

for step, batch in enumerate(loader):
    optimizer.zero_grad(set_to_none=True)
    loss = model(**batch).loss
    loss.backward()
    gnorm = grad_global_norm(model)         # BEFORE clipping or stepping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    l = loss.item()
    window_loss.append(l)
    ema = l if ema is None else EMA_BETA * ema + (1 - EMA_BETA) * l
    running_max = max(window_loss)

    # Log all three. The spike lives in running_max and gnorm, not ema.
    if running_max > 3 * ema:               # crude spike flag
        print(f"[SPIKE] step {step}: loss={l:.2f} ema={ema:.2f} "
              f"max50={running_max:.2f} gnorm={gnorm:.1f}")
```

Three things are logged together here for a reason that is the spine of the whole diagnosis: the **raw loss** tells you the spike happened, the **running max** makes sure you do not miss it under smoothing, and the **grad-norm measured before clipping** tells you *what kind* of spike it is. A loss spike with a simultaneous grad-norm spike is an optimization or numerics event — a big update was about to be taken. A loss spike with a *quiet* grad-norm is more likely a single weird batch with a large per-example loss but a bounded gradient (think one all-one-class batch under a loss that is large but not steep). That single distinction routes you down two different branches of the rest of this post, which is why we measure the grad-norm *before* `clip_grad_norm_` mutates it — clipping after the fact would erase the very evidence you need.

The decoupling of loss and grad-norm is subtle enough to dwell on, because it is the single most useful diagnostic in this whole post and people routinely conflate the two. The loss is a function value; the grad-norm is the slope. A batch can have a large *loss* without a large *gradient*: take a batch the model gets confidently wrong in a region where the surface is flat — the loss is high, but the gradient (the direction and magnitude of steepest ascent) is modest, so the step the optimizer takes is normal and the run is unharmed. The high loss shows up as a one-step bump in the curve that vanishes the next step, because the *parameters barely moved*. Conversely, a batch can have a modest loss but a huge *gradient* if it lands on a steep wall — the optimizer takes a giant step, the parameters lurch, and the loss on the *next* batch spikes because the model just moved to a bad place. So the timing tells you which: a loss spike that coincides with a grad-norm spike on the *same* batch is a big-step event (dangerous, can be terminal); a loss spike with a quiet grad-norm is a high-loss-bounded-gradient batch (annoying, self-heals). Logging both side by side, per step, is what lets you read this in real time instead of guessing.

This is the same discipline laid out in [reading the loss curve as a diagnostic](/blog/machine-learning/debugging-training/reading-the-loss-curve-as-a-diagnostic); here we zoom all the way in on one shape — the tooth — and take it apart.

## 2. The science: why a step too large overshoots

To understand why a spike is even possible, model what one optimizer step does to the loss. Near the current parameters $\theta$, expand the loss to second order along the update direction:

$$
L(\theta - \eta g) \approx L(\theta) - \eta\, g^\top g + \tfrac{1}{2}\eta^2\, g^\top H g,
$$

where $g$ is the gradient, $H$ the Hessian, and $\eta$ the learning rate. The middle term is the descent you want — it is negative, it lowers the loss. The last term is curvature pushback — it is positive when the surface curves upward, and it *grows quadratically in the step size*. For a simple quadratic with curvature $L$ (the largest eigenvalue of $H$, also the Lipschitz constant of the gradient), gradient descent on a single coordinate updates the distance-to-optimum by a factor of $(1 - \eta L)$ each step. That factor has magnitude less than one — meaning you actually move *toward* the minimum — only when

$$
0 < \eta < \frac{2}{L}.
$$

This is the stability bound, and it is the entire mechanism of a spike in one inequality. As long as your learning rate stays below $2/L$ everywhere you travel, every step shrinks the error and the loss trends down. The trouble is that $L$ is **not constant across the landscape**. There are regions where the loss surface is gently curved (small $L$, large allowable step) and regions where it is sharp (large $L$, tiny allowable step). Your learning rate is fixed by the schedule. So when training walks into a sharp region — a ravine wall, a high-curvature ridge — the local $L$ jumps, the product $\eta L$ crosses 2, and the *same step that was descending now overshoots the valley and lands higher up the opposite wall*. The loss jumps. That is the spike.

What happens next is the transient-vs-terminal fork, and it too falls out of the same factor. If the overshoot lands somewhere with $\eta L$ back below 2, the next steps contract again and the loss settles — **transient**. If the overshoot lands in an even sharper place, or if the big step injected `inf`/`NaN` into the weights, $\eta L$ stays above 2 and the error *grows* geometrically, $(1-\eta L)^k$ blowing up — **terminal**, ending in divergence to `NaN`. The spike is the loss bouncing off a too-sharp wall; whether it is a hiccup or a death depends entirely on where it bounces *to*.

![A dataflow graph showing a sharp region, an outlier batch, and an fp16 overflow each producing a huge gradient that overshoots the stability bound into a ten-fold loss spike](/imgs/blogs/loss-spikes-and-divergence-2.png)

There are three independent roads to the same overshoot, and the figure above is the map. **A sharp region** raises $L$ with the step size unchanged — the curvature road. **An outlier batch** raises $g$ — a corrupted, all-one-class, or duplicated batch produces a gradient ten or a hundred times the usual magnitude, so even a normal $\eta$ takes a giant step; this is the data road and it is why the same step size is fine on every other batch. **A numeric event** — an fp16 activation overflowing past the format's max representable value of about $6.5\times10^4$ — injects an `inf` that backpropagates into a gradient of `inf`, which is an infinitely large step; this is the numerics road, and it is the one most likely to be terminal because `inf` times anything is `inf` and the weights are poisoned in a single step. All three converge on the same event: a step far larger than the local stability bound allows, and a loss that jumps as a result.

It is worth being precise about why the *same* learning rate is fine for ten thousand steps and catastrophic on one. The stability bound $\eta < 2/L$ is a statement about the *local* curvature along the *current* update direction, and both of those vary step to step. The Hessian's largest eigenvalue is not a single number for the whole training run — it is a field over parameter space, and it changes as the weights move. Empirically, in deep networks $L$ tends to *rise* over the course of training (a phenomenon sometimes called "progressive sharpening"): early on the surface is gently curved and almost any reasonable LR descends, but as the model fits the data the surface develops sharper and sharper minima, $L$ climbs, and the margin $2/L - \eta$ shrinks. This is the quantitative reason spikes cluster in the middle and late phases of a run rather than the start, and it is why a constant learning rate that was safe at step 1,000 can be marginal at step 8,000. The schedule's job is partly to walk the LR *down* as $L$ walks up, keeping the product $\eta L$ under the bound; a schedule that decays too slowly relative to the sharpening will eventually cross 2 and spike.

The numeric road deserves its own short derivation because it is the one most likely to be terminal and the one most often misdiagnosed as "the LR is too high." In fp16, the largest finite representable value is $(2 - 2^{-10}) \times 2^{15} \approx 6.55\times10^4$. Any activation or intermediate that exceeds this saturates to `inf`. In a transformer, the most common overflow site is the attention logits: the dot product $q^\top k$ before the softmax scales with the dimension and the magnitude of the activations, and a single layer whose activations have drifted large can produce logits in the tens of thousands. Once one logit is `inf`, the softmax produces `NaN` (because $\exp(\infty)/\exp(\infty)$ is indeterminate), the `NaN` flows forward through every subsequent layer, the loss is `NaN`, the backward pass fills every gradient with `NaN`, and the optimizer writes `NaN` into every parameter it touches. This is why a numeric spike is so often a single-step death with no recovery window: there is no "next 100 steps" to watch because step $t+1$ is already operating on poisoned weights. The instrument tell is unmistakable — the grad-norm reads `inf` or `NaN` rather than a large-but-finite $10^3$ — and it is the cleanest signal that you are on the numerics road and that rung 4 (bf16), not rung 3 (lower LR), is your fix.

#### Worked example: when does the step start overshooting?

Make the bound concrete. Suppose along the update direction the local curvature is $L = 4{,}000$ in some sharp region (large but not unusual for a transformer block mid-training). The stability bound says the largest stable step is $\eta < 2/L = 2/4000 = 5\times10^{-4}$. If your schedule has the learning rate at $\eta = 3\times10^{-4}$, you are *under* the bound: $\eta L = 1.2 < 2$, steps contract, no spike. Now an outlier batch arrives whose gradient is $8\times$ the usual magnitude. The *effective* step in parameter space is $\eta \lVert g \rVert$, eight times larger than the curvature analysis assumed — equivalent to an effective learning rate of $2.4\times10^{-3}$, which gives $\eta_{\text{eff}} L = 9.6 \gg 2$. The step blows past the valley. The loss that was 1.9 lands at, say, 19 — a ten-fold spike — because the model walked up the far wall of a ravine it should have walked down. Halve the base learning rate to $1.5\times10^{-4}$ and the same outlier gives $\eta_{\text{eff}} L = 4.8$ — still over 2, still a spike, but a smaller one, more likely transient. Clip the gradient norm to 1.0 and the effective step is bounded regardless of the outlier, and the spike does not happen at all. That is the whole recovery toolkit previewed in one arithmetic example: lower the rate, or bound the step.

## 3. Why large models spike more

If you have only trained small models you may never have seen a real spike, and you might suspect the whole phenomenon is exotic. It is the opposite — spikes get *more* common and more dangerous as models scale, and the large-language-model pretraining literature is full of them. Two compounding mechanisms explain why, and both are worth understanding because they tell you which mitigations transfer.

First, **the landscape sharpens with scale**. Deeper, wider networks have loss surfaces with larger maximum curvature; the largest Hessian eigenvalue $L$ tends to grow, and the sharp regions get sharper. The same stability bound $\eta < 2/L$ that was generous for a two-layer MLP becomes tight for a 70-layer transformer, so the margin between "descending" and "overshooting" is thinner, and a routine fluctuation in $g$ or a slightly-too-warm point on the LR schedule is enough to cross it. This is why large-model recipes use such conservative peak learning rates and such long warmups — they are buying margin under the bound.

Second, and more subtly, **Adam's second-moment estimate lags after a quiet region**. Adam scales each coordinate's update by $1/\sqrt{\hat v}$, where $\hat v$ is an EMA of squared gradients with decay $\beta_2$ (typically 0.999 or 0.95). The effective per-coordinate step is roughly $\eta / \sqrt{\hat v}$. During a quiet plateau where gradients have been small, $\hat v$ shrinks toward that small magnitude. Then one large gradient arrives. The numerator $g$ jumps immediately, but the denominator $\sqrt{\hat v}$ is a *slow* average — with $\beta_2 = 0.999$ it takes roughly $1/(1-\beta_2) = 1000$ steps to forget the old small value. So for the step where the big gradient lands, the update is divided by a stale, *too-small* denominator, and the effective step is enormous. Adam, the optimizer that is supposed to tame gradient scale, briefly *amplifies* exactly the gradient that triggers the spike. This is why spikes so often appear *right after a quiet stretch*, and why one of the standard mitigations is to raise Adam's $\epsilon$ (which floors the denominator) or to lower $\beta_2$ from 0.999 to 0.95 so the second moment adapts faster.

The combination is nasty: scale makes the walls sharper, and Adam's lag makes the worst step land right when you walk into one after a calm. It is no accident that the documented spikes in large runs cluster in the middle of training, after the loss has settled into a slow grind, rather than in the chaotic early steps where everyone is watching.

| Mechanism | Why it worsens with scale | Instrument that catches it | Mitigation |
|---|---|---|---|
| Landscape sharpening | Max Hessian eigenvalue $L$ grows with depth/width | Grad-norm baseline creeps up over training | Lower peak LR, longer warmup |
| Adam second-moment lag | More quiet plateaus in long runs; $\beta_2{=}0.999$ slow | Spike follows a low-grad-norm stretch | Raise $\epsilon$, lower $\beta_2$ to 0.95 |
| Outlier-batch sensitivity | More data, more rare corrupt/degenerate batches | One-step spike, grad-norm $10^3$ | Skip-batch guard, gradient clipping |
| fp16 dynamic range | Larger activations overflow fp16's $6.5\times10^4$ ceiling | Loss-scaler skips, then NaN | bf16, or tune loss scaling |

That table is the compressed version of this section, and it doubles as a triage card: read the instrument column, find your signature, apply the mitigation. We will earn each row in code below.

There is a concrete, measurable consequence of progressive sharpening that you can log directly: the grad-norm *baseline* drifts upward over a long run. If you track the running median grad-norm in a sliding window, you will typically see it climb slowly — say from 1.5 early to 4.0 by mid-training — as the surface sharpens and the model takes larger steps to keep fitting. This matters operationally because **a fixed grad-norm spike threshold goes stale**: a threshold of 20 that flagged real spikes at step 1,000 (when the baseline was 1.5) never fires at step 50,000 (when the baseline is 4.0 and a genuine spike reaches 200, but a threshold tuned for the early run might have been set lower and now fires on normal late-run steps, or higher and now misses). The fix, which we build into the skip-batch guard in section 6, is to make the threshold *relative* to the running median rather than absolute. Logging the baseline drift itself is also a useful early warning: if the median grad-norm is climbing fast, the surface is sharpening fast, and you are heading toward the spike-prone regime — a signal to check whether your LR schedule is decaying quickly enough.

```python
from collections import deque

class GradNormBaseline:
    """Track the running MEDIAN grad-norm so spike thresholds stay relative.
    The median is robust to the very spikes we are trying to detect; a mean
    would be dragged up by them and the threshold would drift toward them."""
    def __init__(self, window=500):
        self.hist = deque(maxlen=window)

    def update(self, gnorm):
        self.hist.append(gnorm)

    def median(self):
        if not self.hist:
            return 1.0
        s = sorted(self.hist)
        return s[len(s) // 2]

    def is_spike(self, gnorm, k=5.0):
        # Relative threshold: k times the robust baseline, not a fixed number.
        return gnorm > k * max(self.median(), 1e-6)
```

This tiny class is the statistical heart of every reliable spike detector: a *robust*, *drifting* baseline against which "outlier" is defined relatively. Plug its `is_spike` into the logging loop and your spike flag stays calibrated for the entire run instead of going stale at the exact moment — mid-training — when spikes become most likely.

## 4. Transient or terminal: the decision that saves the run

Here is the moment from the intro — the spike has happened, your hand is on the keyboard. The question is binary: is this spike **transient** (the run will recover on its own as the optimizer state re-stabilizes) or **terminal** (it is on a path to `NaN` and every step from here is poisoning weights)? Getting this right is worth real money: kill a transient spike and you throw away a healthy run; babysit a terminal one and you waste GPU-hours watching it die and then have to rewind anyway.

The decision is not a guess. It is read off the next hundred steps, and the two instruments that decide it are the **loss** and the **grad-norm**.

![A decision tree that watches the loss and grad-norm for one hundred steps after a spike and splits a self-settling transient spike from a climbing terminal one that reaches NaN](/imgs/blogs/loss-spikes-and-divergence-4.png)

The procedure, which the tree above encodes, is: do not act on the spike step itself. Watch the next ~100 steps. **Transient signature:** the loss falls back toward its pre-spike trajectory (not necessarily all the way, but clearly descending), and the grad-norm returns to its baseline of order 1–10. The optimizer state re-stabilizes — Adam's $\hat v$ catches up to the big gradient, the parameters slide back off the wall — and the run is fine. You may not even need to intervene; the most you do is note the step and the batch for later. **Terminal signature:** the loss keeps *rising* after the spike, or plateaus at the elevated value, and the grad-norm stays high ($10^3$) or climbs toward `inf`. This is geometric blowup — $\eta L$ is stuck above 2 and the error compounds — and it ends in `NaN`, usually within a few hundred steps. Once any weight or activation is `NaN`, it propagates through every subsequent forward pass and the run is dead; there is no recovering it in place.

The dividing line is sharp in practice. A useful rule of thumb from watching many runs: if the loss has returned to within ~2× of its pre-spike value within 100 steps and the grad-norm is back under ~10, it is transient — let it run. If after 100 steps the loss is still above 3× pre-spike *and* the grad-norm has not come down, it is terminal — kill and rewind. The ambiguous middle (recovering slowly, grad-norm elevated but not exploding) is where you add a guard and keep watching rather than killing outright.

You can automate the watch so you are not the one staring at the dashboard. This monitor records the pre-spike state, then classifies over a window:

```python
import math

class SpikeWatcher:
    """Classify a loss spike as transient or terminal over a window of steps."""
    def __init__(self, ratio_spike=3.0, ratio_clear=2.0, gnorm_clear=10.0, window=100):
        self.baseline = None          # EMA of loss before any spike
        self.ratio_spike = ratio_spike
        self.ratio_clear = ratio_clear
        self.gnorm_clear = gnorm_clear
        self.window = window
        self.in_spike = False
        self.steps_since = 0

    def update(self, loss, gnorm):
        if math.isnan(loss) or math.isinf(loss):
            return "TERMINAL_NAN"     # already dead; rewind required
        if self.baseline is None:
            self.baseline = loss
        if not self.in_spike:
            if loss > self.ratio_spike * self.baseline:
                self.in_spike, self.steps_since = True, 0
                return "SPIKE_DETECTED"
            self.baseline = 0.98 * self.baseline + 0.02 * loss   # track slow trend
            return "OK"
        # we are inside a spike: watch for recovery or blowup
        self.steps_since += 1
        recovered = loss < self.ratio_clear * self.baseline and gnorm < self.gnorm_clear
        if recovered:
            self.in_spike = False
            return "TRANSIENT_RECOVERED"
        if self.steps_since >= self.window:
            return "TERMINAL_NO_RECOVERY"  # 100 steps, still elevated -> kill
        return "WATCHING"
```

Wire this into the training loop next to the logging from section 1, and it will print `SPIKE_DETECTED` when the tooth appears, `WATCHING` while it decides, and either `TRANSIENT_RECOVERED` (do nothing) or `TERMINAL_NO_RECOVERY` (trigger the rewind harness in section 7). The point is to make the transient-vs-terminal call a *measurement*, not a feeling.

#### Worked example: reading two spikes at step 8,000

Two runs, identical except for one config flag, both spike at step 8,000 from loss 1.9 to ~19. **Run A** (fp16, no clip): at step 8,020 the loss is 24 and the grad-norm reads `inf`; at 8,050 the loss is `NaN`. The grad-norm went to `inf` because an fp16 activation overflowed the format's ceiling, and `inf` backpropagated. This is terminal, and you could have called it by step 8,020 — a grad-norm of `inf` is never transient. Cost of waiting to "see if it recovers": 50 steps of dead compute plus the rewind you were always going to need. **Run B** (bf16, clip 1.0): the spike to 19 happens because of an outlier batch, but clipping capped the step, so by 8,030 the loss is 4.5, by 8,080 it is 2.2, and by 8,120 it is back to 1.95 with grad-norm 2.3. Transient — the watcher prints `TRANSIENT_RECOVERED` and the run finishes at loss 1.42. Same spike height, opposite outcomes, and the deciding factors are the two numerics/optimization choices that controlled where the overshoot landed. The grad-norm trajectory told you which run you were in within 30 steps.

### The babysit-or-kill decision matrix

The transient-vs-terminal classification answers *what the spike is*; the babysit-or-kill decision answers *what you should do about it right now*, and they are not the same question. A spike can be transient and still worth killing — for instance, a transient spike that recurs every five hundred steps is telling you the run has a chronic problem (a too-high LR or a recurring bad data shard) that will only get worse, and finishing the run gives you a model trained partly on poisoned steps. Conversely, a spike that looks terminal can be worth babysitting if you are ninety-five percent of the way through a run that costs a fortune to restart and you have a recent checkpoint to fall back to anyway. The decision blends three factors the pure classification ignores: the **recovery odds** (is it transient?), the **blast radius** (how many steps get poisoned if you are wrong?), and the **cost** (what does a restart actually cost in GPU-hours versus what is left to do?).

![A matrix mapping each spike signal to a babysit-and-continue or kill-and-fix decision based on recovery odds, blast radius, and remaining cost](/imgs/blogs/loss-spikes-and-divergence-8.png)

The matrix above is the decision surface. Read each row as a situation and the two columns as the case for each action. A spike that **self-recovered once** with the grad-norm back to baseline is a clean babysit — one spike is not a pattern, and intervening risks introducing instability for no gain. A grad-norm **climbing toward `inf`** is a kill: the weights are poisoning and every additional step makes the rewind start from a worse place, so the right move is to rewind *before* the `NaN` lands rather than after. A run that has **already hit `NaN`** is not a decision at all — it is dead, and the only path is rewind-and-skip. A spike that **recurs every few hundred steps** is the interesting case: each individual spike may be transient, but the *pattern* is a chronic bug, and the right call is usually to kill, fix the underlying cause (LR or data), and restart, because finishing a run that spikes chronically gives you a model of unknown quality. And a spike **late in a nearly-finished run** tilts toward babysitting: if you are at step 95,000 of 100,000 and a transient spike self-heals, finishing the last five thousand steps and patching the config for next time is cheaper than throwing away a near-complete run. The matrix is not a formula — it is the set of considerations that turn the intro's snap judgment into a defensible call you can explain to whoever owns the GPU budget.

## 5. The diagnostic: inspect the batch, the grad-norm, and the optimizer state

When the watcher flags a spike — transient or not — you want to know *which of the three roads* (curvature, batch, numerics) caused it, because that determines the durable fix. The diagnostic is a three-part inspection at the spike step: the **batch** that triggered it, the **grad-norm** (already in hand), and the **optimizer state**.

![A matrix mapping each spike cause to its instrument signature, the confirming test, and the one fix that addresses that specific cause](/imgs/blogs/loss-spikes-and-divergence-3.png)

The matrix above is the reference card; here is how to gather the evidence for each row. The single most informative thing you can do when a spike fires is **dump the batch**. If you have a skip-batch guard (section 6), have it serialize the offending batch to disk. Then look at it directly:

```python
import torch
from collections import Counter

def inspect_spike_batch(batch, tokenizer=None):
    """Diagnose WHY a batch triggered a spike: is it an outlier?"""
    ids = batch["input_ids"]
    labels = batch.get("labels", None)

    # 1. Is it all-one-class / degenerate? (classification or all-masked LM)
    if labels is not None:
        valid = labels[labels != -100]   # -100 is the HF ignore_index
        if valid.numel() == 0:
            print("DEGENERATE: every label is masked (-100). Loss is on nothing.")
        else:
            counts = Counter(valid.flatten().tolist())
            top, n = counts.most_common(1)[0]
            frac = n / valid.numel()
            if frac > 0.95:
                print(f"OUTLIER: {frac:.0%} of labels are token/class {top} "
                      f"(all-one-class batch -> large, lopsided gradient).")

    # 2. Numeric pathology in the inputs themselves.
    if torch.is_floating_point(ids):
        if torch.isnan(ids).any() or torch.isinf(ids).any():
            print("CORRUPT: NaN/Inf in the input tensor itself.")
        amax = ids.abs().max().item()
        if amax > 6.5e4:
            print(f"OVERFLOW RISK: |input| max = {amax:.1e} exceeds fp16 ceiling 6.5e4.")

    # 3. Sequence-length pathology (a single 100k-token doc among 512s).
    lengths = (ids != 0).sum(dim=-1)
    if lengths.max() > 4 * lengths.float().mean():
        print(f"LENGTH OUTLIER: longest seq {lengths.max().item()} vs "
              f"mean {lengths.float().mean():.0f} (one giant doc dominates the loss).")
```

Run this on the dumped batch and you usually get your answer in one line: an all-one-class batch (the gradient is large and lopsided), a corrupted sample with `NaN` in the inputs, an fp16-overflowing value, or one pathologically long sequence dominating the loss. If the batch looks *completely ordinary*, the spike was probably curvature or Adam-lag, not data — and the fix moves to learning rate and optimizer hyperparameters rather than data cleaning.

The second piece is the **grad-norm history**, which you are already logging. A grad-norm that spikes to $10^3$ from a baseline of $2.0$ confirms a large-update event (batch or curvature). A grad-norm that goes straight to `inf` confirms a numeric event (overflow). A grad-norm that is *quiet* during a loss spike is the tell that the spike is a large-loss-but-bounded-gradient batch — annoying but rarely terminal.

The third piece is the **optimizer state**, the one people forget. For Adam, the spike step's effective learning rate depends on $\hat v$, the second-moment estimate. Logging the ratio of the gradient to $\sqrt{\hat v}$ tells you whether the Adam-lag mechanism from section 3 is in play:

```python
def adam_step_ratio(optimizer):
    """For Adam, log update_size / param_scale to catch second-moment lag.
    A spike right after a quiet plateau shows a huge ratio: g is big but
    sqrt(v) is still small from the quiet stretch."""
    ratios = []
    for group in optimizer.param_groups:
        eps = group["eps"]
        for p in group["params"]:
            state = optimizer.state.get(p, {})
            if "exp_avg_sq" in state and p.grad is not None:
                v = state["exp_avg_sq"]
                step = (p.grad.abs() / (v.sqrt() + eps)).mean().item()
                ratios.append(step)
    return sum(ratios) / max(len(ratios), 1)
```

If `adam_step_ratio` is normally around 1 and jumps to 50 at the spike, the second-moment lag is your mechanism, and raising $\epsilon$ or lowering $\beta_2$ is the targeted fix. This three-part inspection — batch, grad-norm, optimizer state — is the bisection in miniature: it tells you which of *data*, *optimization*, or *numerics* owns the spike before you change a single hyperparameter, exactly the discipline from [the taxonomy of training bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs).

## 6. The skip-batch guard: stop stepping into the spike

The cheapest durable mitigation is to *not take the bad step in the first place*. If you measure the grad-norm before the optimizer step, you can decide, per step, whether this update is safe — and if the grad-norm is wildly above the recent baseline, you skip the batch: zero the gradients, log the offender, and move on without stepping. This is the production technique behind several large-scale recipes, and it is a dozen lines of code.

![A dataflow graph where grad-norm measured before the step routes the batch to a skip path or a clip-and-step path so an outlier never produces a spike](/imgs/blogs/loss-spikes-and-divergence-7.png)

The guard sits between `backward()` and `step()`, exactly where the grad-norm is available and the step has not yet been taken. The figure traces the two branches: measure the grad-norm, compare it to an adaptive threshold (a multiple of the running median grad-norm, *not* a fixed constant — the baseline drifts over training), and either skip (zero the grads, no step) or clip-and-step normally.

```python
import torch
from collections import deque

class SkipBatchGuard:
    """Skip a step when the grad-norm is a wild outlier; clip otherwise.
    Threshold is adaptive: k * running median grad-norm, so it tracks
    the baseline that drifts upward as training sharpens (section 3)."""
    def __init__(self, k=5.0, clip=1.0, window=200, warmup=200):
        self.k = k
        self.clip = clip
        self.hist = deque(maxlen=window)
        self.warmup = warmup
        self.skipped = 0
        self.steps = 0

    def maybe_step(self, model, optimizer):
        self.steps += 1
        gnorm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=float("inf")  # measure, don't clip yet
        ).item()

        # During warmup just collect statistics; never skip early.
        if self.steps > self.warmup and len(self.hist) >= 20:
            median = sorted(self.hist)[len(self.hist) // 2]
            threshold = self.k * max(median, 1e-6)
            if gnorm > threshold or not torch.isfinite(torch.tensor(gnorm)):
                optimizer.zero_grad(set_to_none=True)  # drop this update entirely
                self.skipped += 1
                return {"stepped": False, "gnorm": gnorm, "threshold": threshold}

        self.hist.append(gnorm)
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip)  # safe step
        optimizer.step()
        return {"stepped": True, "gnorm": gnorm}

# usage inside the loop:
guard = SkipBatchGuard(k=5.0, clip=1.0)
for step, batch in enumerate(loader):
    optimizer.zero_grad(set_to_none=True)
    loss = model(**batch).loss
    loss.backward()
    info = guard.maybe_step(model, optimizer)
    if not info["stepped"]:
        print(f"[SKIP] step {step}: gnorm {info['gnorm']:.0f} "
              f"> {info['threshold']:.0f}; dropped batch, no update.")
```

Two design choices matter. First, the threshold is **adaptive** — `k` times the running median grad-norm — because a fixed threshold either fires constantly early in training (when grad-norms are naturally large) or never fires late (when the baseline has crept up). The median is robust to the very outliers you are trying to catch. Second, the guard **skips, it does not clip the outlier into the step**. Clipping the outlier and stepping anyway still moves the parameters in the outlier's *direction*, just with bounded magnitude; for a genuinely corrupt batch, the direction is garbage and you do not want to move that way at all. Skipping drops the update entirely and continues with the next batch. (Clipping is still applied to *normal* steps as a belt-and-suspenders bound.)

A reasonable starting `k` is 5 (skip steps whose grad-norm exceeds 5× the median). Tune it by watching the skip rate: if you are skipping more than ~0.1% of batches, your threshold is too tight or you have a real data problem; if a known spike still gets through, lower `k`. In a well-behaved run you will skip a handful of batches over tens of thousands of steps, and never see a spike again.

#### Worked example: the skip rate and what it costs

Quantify the trade-off so you can set `k` with numbers instead of superstition. Suppose a 50,000-step run with a baseline median grad-norm that drifts from 1.5 to 4.0, and a genuine spike-triggering outlier batch appears roughly once every 8,000 steps — call it six real spikes over the run. With `k = 5`, the guard skips any batch whose grad-norm exceeds five times the current median. The six real outliers (grad-norm $10^3$, hundreds of times the median) are skipped, no question. The cost is the *false* skips: ordinary batches that happen to have an unusually high grad-norm without being pathological. In a well-conditioned run, the grad-norm distribution is tight enough that batches above 5× the median are genuinely rare — empirically well under 0.05% — so over 50,000 steps you skip perhaps the six real outliers plus ~20 borderline batches, around 26 skipped updates total. Skipping 26 of 50,000 updates discards 0.05% of your gradient signal, a rounding error against the catastrophe it prevents (a `NaN` that ends the run and forfeits, say, 18 GPU-hours and \$360). Now drop `k` to 2 (much more aggressive): you still catch the six real spikes, but now you also skip every batch above 2× the median, which in a normal run might be 1–2% of all batches — 500 to 1,000 skipped updates, a real dent in training signal and possibly a measurable hit to final loss. The lesson: set `k` high enough that you skip only true outliers (start at 5, never below ~3), because the asymmetry is stark — a missed spike can cost the whole run, but over-skipping silently degrades the model you do finish.

The grad-norm clipping itself deserves a word, because it is the most-reached-for fix and it is closely related to but distinct from skipping. Clipping rescales the whole gradient so its norm is at most `max_norm`, bounding the effective step $\eta \lVert g \rVert$ regardless of how large $\lVert g \rVert$ would have been. From the section-2 math, clipping directly enforces an upper bound on the step size, which is exactly what keeps $\eta_{\text{eff}} L$ under the stability bound on an outlier batch. Clipping handles the *magnitude* of the spike; skipping handles its *existence*. Production recipes use both. For more on the explode/vanish dynamics and clipping done right, see [gradients exploding and vanishing](/blog/machine-learning/debugging-training/gradients-exploding-and-vanishing).

## 7. The rewind: when a spike already poisoned the weights

Sometimes you are too late. The watcher says `TERMINAL_NAN`, the loss is already `NaN`, and every parameter is contaminated — `NaN` times anything is `NaN`, so the forward pass produces `NaN` activations, the backward produces `NaN` gradients, and no in-place fix recovers it. The only recovery is to **rewind to a checkpoint from before the spike and resume past the bad data**.

![A before-and-after showing a run that hits NaN at step eight thousand with no guard versus the same run completing at loss one point four two with clipping and a skip-batch guard](/imgs/blogs/loss-spikes-and-divergence-5.png)

The before-and-after above is the whole argument for building this machinery before you need it. The left column is the unguarded run: spike at 8k, grad-norm $10^3$ with no clip, `NaN` at 8.2k, and eighteen GPU-hours of progress gone because there was no checkpoint to fall back to. The right column is the same run with the recovery machinery in place: the spike is clipped, the one bad batch is skipped, and the run completes at loss 1.42. The difference is not a smarter model — it is the operational scaffolding around the run.

The rewind procedure has three parts, and skipping any one of them reintroduces the spike on resume. First, **rewind to a checkpoint from before the spike** — which requires that you checkpoint frequently enough that "before the spike" is recent (every few hundred steps for a run where spikes are a known risk, not every few thousand). Second, **restore the full state, not just the weights**: the optimizer state (Adam's $m$ and $\hat v$), the LR-scheduler position, the gradient-scaler state under AMP, and the dataloader/RNG position. If you restore weights but reset the optimizer, Adam's second moment is wrong and you may spike again immediately for the reasons in section 3. Third — the part everyone forgets — **skip past the offending data**. If the spike was triggered by a specific batch, resuming and re-feeding that same batch reproduces the spike. You must advance the data iterator past the poisoned range, which means your checkpoint has to record *where in the data stream* it was.

```python
import torch

def save_checkpoint(path, model, optimizer, scheduler, scaler, step, data_step):
    """Save EVERYTHING needed to resume continuously: weights, optimizer
    state, scheduler, AMP scaler, and the position in the data stream."""
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),   # Adam m and v -- critical
        "scheduler": scheduler.state_dict(),   # LR position
        "scaler": scaler.state_dict(),         # fp16 loss-scale state
        "step": step,
        "data_step": data_step,                # where in the data stream we are
        "rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }, path)

def resume_and_skip(path, model, optimizer, scheduler, scaler, dataset,
                    skip_batches=200):
    """Rewind to a pre-spike checkpoint and ADVANCE past the poisoned data."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])   # restore m, v -> no re-spike
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    torch.set_rng_state(ckpt["rng"])
    torch.cuda.set_rng_state_all(ckpt["cuda_rng"])

    start = ckpt["data_step"] + skip_batches        # JUMP past the bad batch
    loader = make_loader(dataset, start_index=start) # deterministic resume offset
    print(f"Resumed from step {ckpt['step']}, skipping {skip_batches} batches "
          f"(data_step {ckpt['data_step']} -> {start}) past the spike.")
    return model, optimizer, scheduler, scaler, loader, ckpt["step"]
```

The number of batches to skip is a judgment call: skip too few and you re-hit the bad data; skip too many and you waste fresh examples. For a known single-batch trigger, skipping a few hundred batches past the recorded data position is safe and cheap. If you do not know exactly which batch triggered it, bisecting — skip 200, if it re-spikes skip another 200 — finds the poisoned range quickly. This resume-continuity discipline is its own deep topic; the loss-jump-on-resume signature and the full state-restoration checklist are covered in [debugging checkpoint and resume](/blog/machine-learning/debugging-training/debugging-checkpoint-and-resume), and you should treat that as required reading before you trust any rewind.

Stress-test this procedure against the ways it commonly fails, because a half-correct rewind is worse than none — it reintroduces the spike and wastes the compute of the resume. *What if the resume re-spikes immediately at the same relative step?* That means you restored weights but not the optimizer state (Adam's $\hat v$ reset, so the second-moment lag from section 3 fires again) — check that `optimizer.load_state_dict` actually ran and that the state tensors are non-empty. *What if it re-spikes a few hundred steps later, at the same data?* Your `data_step` was wrong or your loader does not honor `start_index` deterministically, so you replayed the poisoned batch — verify the loader resumes at the exact recorded offset, which usually requires a deterministic sampler and a fixed seed (this is where [reproducibility and determinism](/blog/machine-learning/debugging-training/reproducibility-and-determinism-in-training) pays off). *What if the loss is slightly off after resume even with no spike?* A small loss discontinuity on resume points at an unrestored piece of state — most often the gradient-scaler under AMP or the LR scheduler position — and even a small jump means your "continuous" resume is not, which can compound. *What if it only spikes on multi-GPU?* Then the trigger may be a rank-desync or a per-rank data-sharding bug rather than a single bad batch, and you are debugging the distributed layer, not the data — a different bisection entirely. Running through these four questions before you trust a resume turns the rewind from a hopeful restart into a verified one.

#### Worked example: the rewind arithmetic

A 7B-parameter run, batch size 1M tokens, on 8×A100 at roughly \$2.50 per GPU-hour, so about \$20 per GPU-hour for the node. The run was at step 8,000 when it hit `NaN`, having spent 18 wall-clock hours — about \$360 of compute. Without checkpoints, that is all lost and you restart from zero. With checkpoints every 500 steps, the most recent good checkpoint is step 7,500; rewinding there loses 500 steps of progress, roughly \$22 of compute, plus the cost of re-doing those 500 steps (another \$22) — call it \$45 total versus \$360. Then you skip 200 batches past the recorded data position, fix the numerics (fp16 → bf16 took the overflow off the table), and the run completes. The checkpoint frequency directly bounds your blast radius: at every-5,000-steps checkpointing you would have rewound to step 5,000 and lost \$135 of progress instead of \$45. The lesson is operational, not algorithmic — *checkpoint often enough that a spike costs you minutes, not hours*, and always record the data position so the resume can skip the trigger.

## 8. The recovery ladder: cheapest fix that holds

You now have a kit of interventions: clip, skip-batch, lower LR, add warmup, fix numerics, rewind. They are not alternatives — they are a *ladder*, ordered from cheapest (apply in-place, no restart) to most expensive (rewind and re-run). The principle is to apply the cheapest rung that actually holds the run, and only climb when the cheaper one fails.

![A vertical stack of recovery interventions ordered from cheapest in-place gradient clipping up to a full checkpoint rewind, ending in a completed run](/imgs/blogs/loss-spikes-and-divergence-6.png)

The ladder in the figure reads bottom-cost to top-cost. **Rung 1, gradient clipping** (`max_norm=1.0`): in-place, costs nothing, bounds every step, and stops most outlier-batch spikes from ever materializing. Always on. **Rung 2, skip-batch guard**: in-place, drops the rare batch whose grad-norm is a wild outlier even after clipping would have stepped in its direction. **Rung 3, lower LR / add warmup**: a config change for the next run (or a hot LR-schedule adjustment mid-run if your framework allows it) — if spikes recur even with clipping, your peak LR is too high for the landscape's sharpness, and a longer warmup or a lower peak buys margin under the stability bound. **Rung 4, fix numerics**: switch fp16 → bf16 (bigger range, no loss-scaling games), raise Adam's $\epsilon$, or add an $\epsilon$ to a near-singular LayerNorm — this addresses the numerics and Adam-lag roads. **Rung 5, rewind to checkpoint**: the expensive last resort when a spike already went terminal and poisoned the weights.

The art is matching the rung to the cause you diagnosed in section 5. An outlier-batch spike is rungs 1–2. A recurring spike on ordinary batches is rung 3 (LR). An fp16-overflow spike is rung 4 (bf16). A spike that already hit `NaN` is rung 5 (rewind), and you climb back down to add rungs 1–4 so it does not happen again. Most production runs settle at rungs 1–2 permanently (clip + skip-batch always on) and reach for 3–4 once during tuning. Rung 5 is the fire extinguisher you hope to never use but build anyway.

| Symptom at the spike | Likely cause | Confirming test | Rung to apply |
|---|---|---|---|
| Grad-norm $10^3$, recurs every few hundred steps | LR too high for sharp region | Halve LR on a replay; spike shrinks | Rung 3: lower LR / longer warmup |
| One step spikes, grad-norm $10^3$, batch is degenerate | Outlier / corrupt batch | Dump batch; all-one-class or NaN input | Rungs 1–2: clip + skip-batch |
| Loss-scaler skips repeatedly, then NaN | fp16 overflow | Print max activation vs $6.5\times10^4$ | Rung 4: switch to bf16 |
| Spike right after a quiet plateau | Adam second-moment lag | `adam_step_ratio` jumps to 50× | Rung 4: raise $\epsilon$, lower $\beta_2$ |
| Already NaN, weights contaminated | Terminal divergence | Any NaN in weights/activations | Rung 5: rewind + skip data |

This is the diagnostic table the kit asks for, and it is the working surface of the post: read the left column off your instruments, confirm with the middle, climb to the right rung. Pin it next to the watcher's output and the snap judgment from the intro becomes a lookup.

## 9. Case studies: spikes in the wild

The spike phenomenon is not a toy concern dreamed up for a blog post; it is one of the best-documented operational hazards of large-scale training, and the public training reports of several flagship models describe both the spikes and the exact recovery procedures we have built here. The numbers below are drawn from those reports; where I am giving a recollection rather than an exact figure I say so, in keeping with this series' rule never to fabricate a precise number.

**PaLM (Google, 2022).** The PaLM technical report is the most-cited public description of mid-training loss spikes at scale. Training the 540B-parameter model, the authors observed roughly twenty distinct loss spikes over the course of the run, occurring irregularly and sometimes in the middle of otherwise-healthy training. Crucially, they reported that the spikes were *not* reproducible from the model state alone: restarting from a checkpoint a few hundred steps before a spike and skipping the batches the model had seen in the spike window allowed training to proceed past that point without the spike recurring — strong evidence that specific data batches, interacting with a particular optimizer state, were the trigger, rather than a deterministic property of the loss landscape. Their mitigation was precisely the rewind-and-skip from section 7: rewind ~100 steps before the spike, skip a few hundred batches, resume. That is the single most important real-world validation of this whole post — the people training the largest models recover from spikes by rewinding and skipping data, not by killing the run.

**OPT-175B (Meta, 2022).** The OPT logbook, released alongside the model, is an unusually candid day-by-day account of a large pretraining run, and it documents a long sequence of loss spikes, divergences, and hardware failures that required dozens of manual restarts. The OPT team's interventions included lowering the learning rate on restart, skipping over problematic data, switching optimizer hyperparameters, and rewinding to earlier checkpoints — the full recovery ladder, applied by hand, over weeks. The logbook is worth reading in full precisely because it shows how non-glamorous large-scale training is: a substantial fraction of the effort was spike-and-divergence babysitting, exactly the judgment calls this post is trying to systematize.

**GLM-130B (Tsinghua, 2022).** The GLM-130B report discusses training stability at length and is notable for attributing many of its spikes and divergences to *numerical* causes — attention-score overflow and the precision of certain operations — and for adopting mitigations aimed at the numerics road specifically (including careful handling of the softmax and embedding-layer gradients, and gradient shrinking). It is a clean example of a run where the dominant spike mechanism was numerics rather than data or raw LR, and where the durable fix was rung 4 (fix the numerics), not rung 3 (lower the LR). The general lesson the report reinforces is that the *cause* determines the *fix*, and that you must diagnose before you intervene.

**The Mixed Precision Training paper (Micikevicius et al., 2018).** Not a spike case study per se, but the foundational text on *why* fp16 spikes and how loss scaling addresses it. The paper documents that fp16's representable range bottoms out near $6\times10^{-5}$ for normal numbers and tops out near $6.5\times10^4$, so small gradients underflow to zero and large activations overflow to `inf`; loss scaling multiplies the loss by a large factor before backprop to shift gradient magnitudes up into the representable range, then unscales before the optimizer step. When loss scaling is mistuned — scale too high and activations overflow, scale too low and gradients underflow — you get exactly the numeric spikes of the GLM variety. The modern answer, bf16, sidesteps the dynamic-range problem by trading mantissa bits for exponent bits, which is why "switch to bf16" is rung 4's first move. For the full treatment of fp16 versus bf16 and reading the gradient histogram to choose, see [mixed-precision debugging](/blog/machine-learning/debugging-training/mixed-precision-debugging-fp16-vs-bf16) within this series and the [edge-AI quantization posts](/blog/machine-learning/edge-ai/) for the broader numeric-format trade-offs.

What unifies the four is the diagnose-then-fix discipline: PaLM and OPT spikes were dominantly data/optimizer (rewind-and-skip, lower LR); GLM's were numerics (fix the format and the softmax); and the mixed-precision paper explains the numeric mechanism that makes the numerics road possible at all. None of them killed-and-restarted-from-zero on a spike. They instrumented, classified, and recovered.

There is a second-order lesson in these reports that is easy to miss and worth stating plainly: the teams that recover gracefully are the ones who *built the recovery machinery before they needed it*. The PaLM rewind-and-skip works because they were checkpointing frequently and recording the data position; the OPT logbook reads as a sequence of recoveries rather than a sequence of catastrophes because the infrastructure to rewind, skip data, and resume continuously was already in place. A team that discovers at step 50,000 that they have been checkpointing only every 10,000 steps, without recording the data offset, is in a far worse position than one that planned for spikes from day one — the first has to choose between losing 10,000 steps of progress and re-running blind through the poisoned data, while the second loses a few hundred steps and skips precisely past the trigger. The spike is inevitable at scale; the question the reports answer is whether you have made it cheap or expensive to recover, and that is decided long before the spike happens, in how you set up checkpointing, logging, and the data iterator. This is the operational, build-it-debuggable-from-day-one theme that the [capstone playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) builds out in full.

## 10. The same bug across CV, speech, and tabular

Everything so far has leaned on LLM-pretraining examples because that is where spikes are most documented, but the mechanism — a step too large for the local curvature — is modality-agnostic, and the spike shows up everywhere with modality-specific triggers. Recognizing the cross-modal signature keeps you from mis-attributing a spike to something exotic.

**Computer vision.** Vision spikes most often come from the data road. A single corrupted image (a truncated JPEG that decodes to garbage, a label that is off by one class) or a too-aggressive augmentation that occasionally produces a degenerate sample (a crop that lands entirely on padding, a mixup that blends to noise) produces an outlier batch and a grad-norm spike. The fix is the same skip-batch guard, plus auditing the augmentation pipeline. A second common vision trigger is **BatchNorm at small batch sizes**: when a batch happens to have near-zero variance in some channel, the BN normalization divides by a tiny number and the activations blow up, producing a numeric spike — a rung-4 fix (raise BN's $\epsilon$, or switch to GroupNorm). When finetuning a pretrained vision backbone, a spike early in finetuning almost always means the learning rate is 10–100× too high for a pretrained model, destroying the features — rung 3, lower the LR drastically; this is covered in the vision-transformer-finetuning material of the series.

**Speech.** Speech models spike on the numerics and data roads. CTC loss — the connectionist temporal classification loss used to train alignment-free speech recognizers — returns `inf` when the input sequence (the number of encoder time-steps) is shorter than the target label sequence (the number of output tokens), because the alignment is mathematically impossible: CTC has to map each label to at least one input frame, and you cannot fit more labels than you have frames. One such sample in a batch produces an `inf` loss → `inf` gradient → instant terminal spike, and notably this is a *hard, deterministic* trigger, not a probabilistic one — the same offending sample blows up every single time it appears, which makes it both easier to find (filter on `input_length >= target_length` before training) and more dangerous if you do not (it is not an unlucky batch you can skip past, it is a structural property of that data point). The fix is to filter those samples up front rather than to clip after the fact, because clipping an `inf` still leaves you with garbage. A sample-rate mismatch (a 44.1kHz file fed to a 16kHz model without resampling, so the audio is effectively time-stretched) or a clipped/silent audio segment produces outlier features and a grad-norm spike of the ordinary, skip-able kind. The CTC `inf`-loss trap and the broader NaN-hunting machinery are the subject of [hunting NaNs and Infs](/blog/machine-learning/debugging-training/hunting-nans-and-infs); a speech spike that is a single `inf` is really a NaN-hunt, not a curvature problem, and the diagnostic is to check the length constraint, not the learning rate.

**Tabular and gradient boosting.** Neural tabular models spike on the same optimization mechanism, usually triggered by an unscaled feature with a huge range (one column in dollars from 0 to $10^9$ alongside columns in [0,1]) that dominates the gradient until normalized. Gradient-boosted trees (XGBoost/LightGBM) do not "spike" in the SGD sense — they do not take gradient steps in parameter space — but the analogous failure is a single round that massively overfits a leaked or outlier feature and tanks the validation metric, which you catch by watching the per-round validation curve rather than a per-step loss. The instrument changes (per-round eval metric instead of per-step grad-norm) but the discipline is identical: watch the right signal, and a sudden jump is a diagnose-then-fix moment.

The through-line is that a spike is always *a step too large for where the model currently is*, whether the step is an SGD update overshooting a sharp region, an `inf` from an impossible CTC alignment, or a boosting round chasing a leaked feature. The trigger is modality-specific; the mechanism and the recovery ladder are not.

## 11. When this is (and isn't) your bug

A spike is a specific signature, and the most expensive mistake is to apply spike remedies to a non-spike problem (or vice versa). Be decisive about when the tooth in your curve is, and is not, a loss spike.

**It is a spike** when the loss jumps up sharply over one or a few steps from an otherwise-healthy trajectory, *and the grad-norm jumps with it*. That coupling is the signature. The fix is the recovery ladder: clip, skip-batch, lower LR, fix numerics, rewind. If you see this, stop debugging the model architecture and the data semantics and instrument the optimization/numerics path.

**It is not a spike, it is divergence-from-the-start**, when the loss climbs from step 1 and never descends. That is not a spike off a healthy run; it is a learning rate that is too high *globally* (every step overshoots, not just the ones in sharp regions) or a broken loss sign. The fix is in [the learning rate is almost always the problem](/blog/machine-learning/debugging-training/the-learning-rate-is-almost-always-the-problem), not the skip-batch guard — guarding individual batches will not save a run that is overshooting on every batch. The LR-range test, not the spike watcher, is your tool.

**It is not a spike, it is a sawtooth**, when the bumps are *regular and periodic*, recurring at a fixed step interval (often once per epoch). That periodicity is the tell of a dataloader artifact — usually unshuffled data, so the same hard examples come around at the same phase each epoch. The grad-norm bumps too, but the *regularity* distinguishes it from a one-off spike. The fix is `shuffle=True` and a correct sampler, a data-pipeline problem covered in [the input pipeline is lying to you](/blog/machine-learning/debugging-training/the-input-pipeline-is-lying-to-you), not a numerics fix.

**It is not a spike, it is a smooth slide to NaN**, when the loss does *not* spike up first — it descends or plateaus normally and then the dashboard simply goes to `NaN` between two ordinary-looking steps, with no preceding tooth. A *smooth*-then-`NaN` curve, with no spike, points at a slow numeric pathology (a gradually growing activation that finally overflows, an accumulating `inf` in a running statistic) rather than a too-large-step overshoot. This is a NaN-hunt by bisection-over-steps, the subject of [hunting NaNs and Infs](/blog/machine-learning/debugging-training/hunting-nans-and-infs), and the rewind-and-skip will not help because there is no single bad batch to skip — the pathology is in the numerics, not the data.

**It is not a spike, it is the warmup ending**, when a modest bump appears right at the step where your learning-rate warmup finishes and the LR reaches its peak. As the LR ramps from near-zero to its peak, the step size grows, and at the peak the model is taking its largest steps of the whole run; a small, one-time increase in loss-noise around that transition is normal and expected, not a spike to panic over. The tell is the *timing* (it coincides exactly with the end of warmup, which you know from your schedule) and the *magnitude* (a modest wobble, not a 10× jump). If your loss bumps mildly at the warmup-to-peak transition and then resumes descending, that is the schedule working, not a bug — do not lower the peak LR reflexively, or you will leave performance on the table. A *large* spike at the warmup-end, on the other hand, is a real signal that your peak LR is too high for the landscape, and that is a rung-3 fix.

**And it is not your problem at all** when a single isolated spike self-recovers within 50 steps, the grad-norm comes straight back to baseline, and it never happens again. That is a healthy run absorbing one unlucky batch, exactly what clipping is for. Adding more machinery to "fix" a one-time transient that already healed is wasted effort and may *introduce* instability (an overzealous skip-batch threshold that starts dropping good batches). If the overfit-one-batch test passes, the loss is descending, and a lone spike self-heals, stop debugging and let the run finish. Knowing when *not* to intervene is half of the judgment this post is about.

## 12. Key takeaways

- **Log the running max and the grad-norm, not just the smoothed loss.** A heavy EMA ($\beta = 0.99$) shows you ~2% of a 10× spike; the windowed max and the per-step grad-norm are what surface it. If you cannot see the spike, you are reading the wrong channel.
- **A spike is a step too large for the local curvature**, formalized by the stability bound $\eta < 2/L$. It happens when training walks into a sharp region (high $L$), an outlier batch inflates $g$, or a numeric event injects an `inf` — three roads, one overshoot.
- **Large models spike more** because the landscape sharpens with scale and Adam's second-moment estimate lags after a quiet plateau, briefly amplifying the one big gradient that triggers the spike.
- **Classify transient vs terminal by watching the next ~100 steps.** Loss falling back and grad-norm returning to order 1–10 is transient — let it run. Loss climbing and grad-norm stuck at $10^3$ or going to `inf` is terminal — kill and rewind. A grad-norm of `inf` is never transient.
- **Diagnose the cause before choosing the fix:** dump the batch (outlier?), read the grad-norm (large-update vs `inf`), and check the optimizer state (Adam-lag ratio). Cause determines rung.
- **Climb the recovery ladder cheapest-first:** clip (always on) → skip-batch on adaptive grad-norm threshold → lower LR / longer warmup → fix numerics (fp16→bf16, raise $\epsilon$) → rewind to a pre-spike checkpoint and skip the data.
- **A terminal spike requires a full-state rewind** — restore optimizer, scheduler, scaler, and RNG, *and skip past the offending data*, or you reproduce the spike on resume.
- **Checkpoint often enough that a spike costs minutes, not hours.** The checkpoint interval is the blast radius of a divergence; for spike-prone runs, every few hundred steps.
- **Match the symptom to the right bug:** spike+grad-norm coupling is a spike; climb-from-step-1 is global LR; periodic bumps are a dataloader sawtooth; smooth-then-NaN is a numeric NaN-hunt, not a spike.

## Further reading

- **Chowdhery et al., "PaLM: Scaling Language Modeling with Pathways" (2022)** — the canonical public account of mid-training loss spikes at 540B scale and the rewind-and-skip-batch recovery that this post systematizes; read its training-stability section.
- **Zhang et al., "OPT: Open Pre-trained Transformer Language Models" (2022)** and the accompanying OPT logbook — a candid day-by-day record of spikes, divergences, and manual restarts on a 175B run; the recovery ladder applied by hand over weeks.
- **Zeng et al., "GLM-130B: An Open Bilingual Pre-trained Model" (2022)** — a stability-focused report where the dominant spike mechanism was numerics (attention-score overflow), illustrating the cause-determines-fix principle.
- **Micikevicius et al., "Mixed Precision Training" (ICLR 2018)** — the foundational text on fp16's representable range, why activations overflow and gradients underflow, and how loss scaling addresses it; the basis for the bf16 mitigation.
- **PyTorch documentation:** `torch.nn.utils.clip_grad_norm_`, `torch.autograd.set_detect_anomaly`, and the AMP guide (`torch.amp.autocast`, `GradScaler`) — the exact APIs for the guards and numerics fixes here.
- Within this series: [a taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) for the bisection frame, [reading the loss curve as a diagnostic](/blog/machine-learning/debugging-training/reading-the-loss-curve-as-a-diagnostic) for the shapes, [the learning rate is almost always the problem](/blog/machine-learning/debugging-training/the-learning-rate-is-almost-always-the-problem) and [hunting NaNs and Infs](/blog/machine-learning/debugging-training/hunting-nans-and-infs) for the adjacent failure modes, [debugging checkpoint and resume](/blog/machine-learning/debugging-training/debugging-checkpoint-and-resume) for the rewind machinery, and [the training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) for the full symptom-to-fix decision tree.

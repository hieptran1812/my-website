---
title: "Silent NaN at Scale: Hunting a Loss That Exploded Across 64 GPUs"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "A war story and a method: at step 12,000 the loss went to NaN on all 64 GPUs at once, and the last good checkpoint was two hours back. Why the all-reduce spreads one rank's NaN everywhere, the six causes to check, and the guardrails that turn a dead run into a skipped step."
tags:
  [
    "distributed-training",
    "multi-gpu",
    "numerical-stability",
    "nan-debugging",
    "mixed-precision",
    "pytorch",
    "nccl",
    "deep-learning",
    "ml-systems",
    "gpu",
  ]
category: "machine-learning"
subcategory: "Distributed Training"
author: "Hiep Tran"
featured: true
readTime: 36
---

The page came in at 3:11am. A 7B-parameter pretraining run on 64 H100s — eight nodes, eight GPUs each, chugging along at a healthy 118,000 tokens/second and 41% MFU for nine days — had frozen. Not slowed. Frozen. The loss dashboard, which had been sliding down a clean curve from 11.2 toward 2.4 for a week and a half, showed a single vertical line at step 12,000 and then a flat row of the three letters every ML engineer learns to dread: `NaN`, `NaN`, `NaN`. Every rank. All 64 GPUs. The same step. Simultaneously.

The first instinct — the wrong one — is to think one GPU broke. But the logs said something stranger: rank 0 reported NaN, rank 1 reported NaN, rank 37 reported NaN, rank 63 reported NaN, all at step 12,000, none at step 11,999. Sixty-four independent processes on eight physical machines connected by InfiniBand do not spontaneously agree on anything to the step unless something is forcing them to agree. Something was. The last checkpoint had been written at step 10,400, roughly two hours and \$1,100 of H100 time ago. The run was not just dead; it was dead *everywhere at once*, and the question that decides whether you lose two hours or two days is: **why did a numerical problem that must have started on one rank appear on all of them in the same instant?**

The answer is the gradient all-reduce, and it is the whole reason distributed NaNs behave differently from single-GPU NaNs. When you train with data parallelism, every rank computes gradients on its own micro-batch and then you *sum* those gradients across all ranks so that every rank applies the same averaged update. That sum is the contamination vector. In IEEE 754 floating point, `NaN + anything = NaN`. So the instant one rank contributes a single non-finite gradient element, the all-reduce output is NaN in that element on *every* rank, the optimizer writes NaN into the weights on every rank, the next forward pass reads NaN weights and produces a NaN loss on every rank, and from the dashboard it looks like all 64 GPUs failed at once. Figure 1 is that mechanism on one page — keep it in front of you for the rest of this post.

![diagram of one rank contributing a NaN gradient into a summing all-reduce that then writes NaN weights onto every rank](/imgs/blogs/silent-nan-at-scale-1.webp)

This is the fifth war story in the [Distributed Training in the Trenches](/blog/machine-learning/distributed-training/why-distributed-training) series. By the end of it you will be able to do four things fast, at 3am, half-awake: reason about *why* a NaN is global instead of local; run down the six causes that produce more than nine out of ten distributed NaNs; find the **origin rank** — the one that produced the NaN first — instead of drowning in 64 identical NaN logs; and install the guardrails that turn "the run is dead, restore the checkpoint, lose two hours" into "step 12,000 was skipped, the run kept going, here's the alert." We will hunt this specific NaN to ground — and it turns out to be a genuinely nasty one — but the method generalizes to every NaN you will ever chase across ranks.

## The distributed twist: why one NaN becomes everyone's NaN

Start with the arithmetic, because it is the entire reason a local bug becomes a global outage. A NaN — "not a number" — is a special floating-point bit pattern produced by operations that have no defined real answer: `0/0`, `inf - inf`, `inf * 0`, `sqrt(-1)`, `log(-1)`. Its defining and dangerous property is that it is **absorbing under arithmetic**: any operation with a NaN operand returns NaN. `NaN + 5 = NaN`. `NaN * 0 = NaN` (not 0). `0.5 * (NaN + finite) = NaN`. There is no arithmetic that recovers a real number from a NaN; once one enters a computation, it flows forward through every op that touches it until you explicitly detect and replace it.

Now overlay data-parallel training. Each of the 64 ranks holds a full replica of the model. Each computes a loss on its own slice of the batch and calls `.backward()`, producing a gradient tensor `g_i` on rank `i`. Before the optimizer steps, [Distributed Data Parallel](/blog/machine-learning/distributed-training/ddp-from-first-principles) runs an **all-reduce** — a collective that sums a tensor across all ranks and hands the identical summed result back to every rank. The averaged gradient every rank applies is

$$\bar{g} = \frac{1}{N} \sum_{i=0}^{N-1} g_i .$$

That sum is where the poison spreads. If a single element of a single rank's gradient — say element `k` of `g_37` — is NaN, then the sum $\sum_i g_i$ at position `k` is NaN, because NaN absorbs the 63 finite contributions. The all-reduce, faithfully doing its job, broadcasts that NaN back to all 64 ranks. Now `\bar{g}[k]` is NaN on every rank. The optimizer — SGD, Adam, whatever — computes `w[k] <- w[k] - lr * \bar{g}[k]`, and `finite - lr * NaN = NaN`, so `w[k]` becomes NaN on every rank. One weight element is now NaN across the entire cluster.

It does not stay one element for long. On the very next forward pass, that NaN weight multiplies into activations, the NaN spreads across the layer's outputs, the loss — a reduction over all outputs — becomes NaN, `.backward()` produces all-NaN gradients on every rank, and the all-reduce now has 64 NaN inputs instead of one. Within one or two steps the entire model is saturated with NaN on all 64 GPUs. That is why the dashboard shows a clean flat line of NaN starting at exactly step 12,000: the all-reduce synchronized the failure to a single step and then the absorbing property did the rest.

This is the single most important idea in distributed numerical debugging, so let me state it as a law: **the reduction that makes data parallelism correct is also what makes one rank's NaN into everyone's NaN.** The all-reduce is not broken. It is doing exactly what it is supposed to do — combine per-rank gradients into a consistent global gradient — and the same mechanism that keeps all 64 replicas in lockstep also keeps them in lockstep when one of them goes bad. You cannot debug a distributed NaN by staring at rank 0, because by the time rank 0 sees NaN, the all-reduce has already erased the evidence of which rank started it. Rank 0's NaN and rank 37's NaN are byte-identical — they are the *same* summed value. The whole game is finding the origin *before* the all-reduce averages it into anonymity.

### Why "check for NaN in the loss" is too late

The naive guardrail everyone writes first is `if torch.isnan(loss): ...` after the forward pass. It catches the fire, but only after the building is gone. By the time the loss is NaN, the all-reduce has run, the weights are already NaN on every rank, and the optimizer has already stepped. You cannot recover the pre-NaN weights from the current step — they were overwritten. The loss check tells you the run is dead; it does not save it. To actually save the run you have to intercept *between* the backward pass and the optimizer step, check the *gradient* for finiteness, and refuse to step if it is bad. We will build exactly that guard, but first we need to know what we are guarding against — the causes.

## A field guide to where NaNs come from

Over enough runs, the causes of a distributed NaN collapse into a short list. Figure 2 is that list as a lookup table: the symptom you see on the dashboard, the mechanical root cause, and the fix. Almost every NaN you hit at scale is one of these six.

![matrix mapping six NaN causes to their dashboard symptom their mechanical root cause and their fix](/imgs/blogs/silent-nan-at-scale-2.webp)

Here is the same catalog in prose, because each cause has a tell you can learn to recognize.

| Cause | What you see | Why it happens | The fix |
|---|---|---|---|
| **fp16 overflow** | grad-norm climbs then hits `inf`, then NaN a step later | an activation or gradient exceeds fp16's max (~65,504) and rounds to `inf`; then `inf - inf` or `inf * 0` makes NaN | dynamic loss scaling, or switch to bf16 |
| **Loss spike / bad batch** | loss jumps sharply for a step, gradients explode, NaN follows | a rare long or pathological sequence, or too-high LR, produces a huge gradient that overflows | `clip_grad_norm_`, skip-step, sequence-length filtering |
| **log(0) / divide-by-zero** | NaN appears inside one specific op, grad-norm otherwise normal | a custom loss takes `log` of a zero probability, or a normalization divides by a zero variance | add an epsilon (`1e-6`) inside the log/denominator |
| **Bad node (ECC bit flip)** | one rank's grad-norm spikes 6–8 orders of magnitude alone | faulty GPU memory flips a bit in a gradient; hardware, not math | detect the origin rank, evict the node, run a hardware test |
| **Attention softmax overflow** | NaN only at long context, correlated with sequence length | attention scores grow large without `1/sqrt(d)` scaling or masking, softmax overflows | scale scores by `1/sqrt(d_head)`, mask correctly, use a stable softmax |
| **Exploding / bad init** | NaN in the first handful of steps, never reaches step 100 | a parameter is initialized with too-large variance, or an uninitialized buffer, and blows up immediately | fix the init (e.g. `std=0.02`), check every buffer is initialized |

The reason this taxonomy is worth memorizing is that the *symptom* tells you which cause you are looking at before you have found the origin rank. A NaN in the first ten steps is an init or a code bug — it is not a bad node, because a bad node would have failed the previous run too. A NaN that only appears at long sequence length is attention or overflow, not `log(0)`. A NaN preceded by a smooth grad-norm ramp over hundreds of steps is a numerics problem (overflow, LR, data); a NaN preceded by *nothing* — a single rank spiking alone from a healthy baseline — is hardware until proven otherwise. That last distinction is the fork we will hang the whole investigation on later.

Notice that four of the six causes route through the same intermediate state: **an overflow to `inf`, which then becomes NaN.** fp16 overflow, the loss spike, attention softmax overflow, and exploding init all produce enormous values first, hit the representable ceiling of the float format, round to `inf`, and only *then* — when that `inf` meets a subtraction or a multiply-by-zero — turn into NaN. This matters enormously for detection, because `inf` has a precursor that NaN does not: the gradient *norm*, which grows continuously toward the ceiling before it crosses it. You can watch the norm climb and catch the problem a hundred steps before the NaN. That is the single most useful monitoring signal in the whole business, and it is next.

## Reading the grad-norm: catching the spike before the NaN

Here is the postmortem detail that turned our 3am mystery from "impossible" to "obvious." When we pulled the grad-norm history — the L2 norm of the full gradient vector, logged every step — the NaN at step 12,000 was not a bolt from the blue. The norm had been sitting around 0.4 for days. At step 11,800 it was 0.42. At step 11,950 it was 2.1. At 11,990 it was 55. At 11,998 it was 8,300. At 12,000 it was `inf`, and at 12,001 the loss was NaN. Figure 3 is that trajectory. The blowup took about two hundred steps, and every one of those steps was a step where a monitor could have paged us with "grad-norm is 55 and climbing" instead of "the run is dead."

![timeline of the gradient norm climbing from a stable baseline through rising values to infinity and then a NaN loss](/imgs/blogs/silent-nan-at-scale-3.webp)

Why does the norm ramp instead of jumping? Because the model is in a feedback loop with itself. A slightly-too-large update pushes the weights into a region where the loss landscape is sharper, which produces a slightly larger gradient next step, which produces a slightly larger update, and so on. It is a geometric progression: each step multiplies the norm by a factor a little above one. A geometric progression looks flat for a long time and then appears to explode, because $1.4^{n}$ is unremarkable at $n=10$ and enormous at $n=40$. By the time the human notices the loss ticked up, the norm is already three orders of magnitude into the runaway.

Make that precise, because the exponent is where the intuition lives. Let $r_t$ be the gradient norm at step $t$, and suppose the feedback loop has an effective per-step growth factor $\rho > 1$ — each unstable step multiplies the norm by roughly $\rho$. Then $r_{t} \approx r_0 \, \rho^{\,t}$, and the number of steps to cross the fp16 ceiling $C \approx 65{,}504$ from a baseline $r_0$ is

$$t_\text{blowup} \approx \frac{\log(C / r_0)}{\log \rho}.$$

Plug in our run: $r_0 = 0.4$, $C = 65{,}504$, so $\log(C/r_0) = \log(163{,}760) \approx 12.0$. If $\rho = 1.4$ (a norm growing 40% per step), then $t_\text{blowup} \approx 12.0 / \log(1.4) \approx 12.0 / 0.336 \approx 36$ steps. That matches the trajectory in figure 3 almost exactly — the runaway from a healthy 0.4 to `inf` took on the order of tens of steps, not one and not thousands. The formula also tells you the two knobs that buy time: a bigger ceiling $C$ (this is precisely what bf16's `3.4e38` gives you — it pushes $t_\text{blowup}$ from ~36 steps to hundreds, often enough to never actually overflow) and a smaller growth factor $\rho$ (this is what gradient clipping and a lower learning rate do — clip the norm back to 1.0 every step and $\rho$ can never exceed 1, so the progression cannot run away at all). The whole prevention stack is, in one sentence, *raise the ceiling and hold the growth factor at or below one.*

The mechanism that converts the runaway into a NaN is the float ceiling. fp16 (half precision) can represent numbers up to about 65,504. When the gradient runaway pushes a single element past that ceiling, it does not saturate — it rounds to `inf`. And `inf` is one arithmetic operation away from NaN. Adam, for instance, maintains a second-moment estimate `v = beta2 * v + (1 - beta2) * g^2`; if `g` is `inf`, then `g^2` is `inf`, `v` is `inf`, and the update `g / (sqrt(v) + eps)` is `inf / inf = NaN`. The overflow becomes a NaN inside the optimizer, and then the all-reduce on the *next* step spreads it. (This is the same numerics that the [mixed precision at scale](/blog/machine-learning/distributed-training/mixed-precision-at-scale) post covers from the format side; here we care about it as the thing that turns a big gradient into a global NaN.)

The practical consequence: **log the gradient norm every step and alert on it, because it is a leading indicator and the loss is a lagging one.** Here is the monitor. It keeps a rolling window, computes a robust baseline (median, not mean, so a single spike does not poison the baseline), and fires when the current norm exceeds the baseline by a large factor — long before `inf`.

```python
import torch
from collections import deque
import statistics

class GradNormMonitor:
    """Rolling grad-norm watchdog. Alerts when the norm departs its
    recent baseline by a large factor -- catches the ramp before overflow."""

    def __init__(self, window=200, spike_factor=8.0, hard_ceiling=1e4):
        self.history = deque(maxlen=window)
        self.spike_factor = spike_factor
        self.hard_ceiling = hard_ceiling

    def total_grad_norm(self, model):
        # Same L2 norm clip_grad_norm_ computes, but read-only.
        sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                sq += p.grad.detach().float().pow(2).sum().item()
        return sq ** 0.5

    def check(self, model, step):
        norm = self.total_grad_norm(model)
        alert = None
        if norm != norm or norm == float("inf"):     # NaN or inf
            alert = f"[step {step}] grad-norm is non-finite ({norm})"
        elif norm > self.hard_ceiling:
            alert = f"[step {step}] grad-norm {norm:.1f} over ceiling"
        elif len(self.history) >= 20:
            baseline = statistics.median(self.history)
            if baseline > 0 and norm > self.spike_factor * baseline:
                alert = (f"[step {step}] grad-norm {norm:.1f} is "
                         f"{norm / baseline:.1f}x baseline {baseline:.2f}")
        self.history.append(norm)
        return norm, alert
```

Wire it into the loop right after `backward()` and before the optimizer step, and page on `alert`. On our run this would have fired at step 11,970-ish, when the norm crossed eight times its baseline of 0.4 — about thirty steps and forty seconds before the `inf`. Not enough to hand-intervene, but more than enough for an *automatic* guardrail to skip the step. The point of the monitor is not usually to wake a human; it is to feed the skip-step guard we build next, and to leave a trail so that when you *do* wake up, the story is already written.

#### Worked example: the fp16 overflow trajectory

Put concrete numbers on it. Suppose you train in pure fp16 with a fixed loss scale of 65,536 (a common default). The loss scale multiplies the loss before backward so that small gradients do not underflow to zero in fp16; you divide it back out before the optimizer step. Now suppose a slightly hot batch produces a genuine gradient element of 1.5. Scaled, the gradient carried in fp16 is `1.5 * 65,536 = 98,304`, which is already past the fp16 ceiling of 65,504 — so it stores as `inf` *before* you ever get to unscale it. Unscaling `inf / 65,536` is still `inf`. Adam squares it, gets `inf`, and the update is NaN. The run dies, and the naive reading is "the data was bad." It was not; the *loss scale was too high for that gradient magnitude*.

The two fixes attack it from opposite ends. **Dynamic loss scaling** watches for exactly this: when it detects a non-finite gradient, it *skips the step* and halves the loss scale, so the next scaled gradient is `1.5 * 32,768 = 49,152`, which fits. PyTorch's `GradScaler` does this automatically — it is why fp16 training with `GradScaler` survives spikes that fixed-scale fp16 dies on. **bf16** attacks the ceiling directly: bfloat16 has the same 8-bit exponent as fp32, so its max is about `3.4e38`, and a gradient element of 98,304 is nowhere near it. bf16 trades mantissa precision for range, and range is exactly what overflow needs. This is why most large-model pretraining moved to bf16: it makes the whole overflow-to-NaN pathway nearly impossible, at the cost of a few bits of precision that a well-conditioned training run does not miss. If your NaN is preceded by a grad-norm ramp and you are in fp16, switching to bf16 or enabling dynamic scaling fixes it before you even find the origin rank.

## The guardrail: skip the step when gradients are not finite

Detection is half the job; the other half is *surviving* the detection. The correct behavior when a gradient is non-finite is not to crash and not to write NaN weights — it is to **skip the optimizer step**, zero the gradients, and continue from the same weights on the next batch. One skipped step out of twelve thousand costs you nothing; a run-ending NaN costs you two hours. Figure 4 is the before-and-after: an unguarded step versus a guarded one.

![before and after comparison of an unguarded optimizer step writing NaN weights versus a guarded step that checks finiteness and skips](/imgs/blogs/silent-nan-at-scale-4.webp)

The distributed subtlety — the thing that makes this harder than the single-GPU version — is that **all ranks must make the same skip decision.** If rank 37 sees a non-finite gradient and skips, but ranks 0–36 and 38–63 see finite gradients (because the NaN was only in rank 37's local gradient, *before* the all-reduce) and step, then you have a catastrophe: 63 ranks advance their weights and one does not, the replicas diverge, and the next all-reduce is summing gradients from models that no longer agree. A divergence bug is far nastier than a NaN — it is silent, it corrupts the model slowly, and it does not show up as `NaN` on the dashboard. So the finiteness decision itself has to be a collective: every rank computes a local "is my gradient finite" flag, you all-reduce those flags, and *every* rank skips if *any* rank's gradient was non-finite. Here is the guard.

```python
import torch
import torch.distributed as dist

def finite_grads_everywhere(model):
    """Returns True only if every gradient on every rank is finite.
    All ranks get the same answer, so they make the same skip decision."""
    local_ok = torch.ones(1, device="cuda")
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            local_ok.zero_()
            break
    # AND across ranks via MIN: if any rank is 0, everyone sees 0.
    dist.all_reduce(local_ok, op=dist.ReduceOp.MIN)
    return bool(local_ok.item())

def guarded_step(model, optimizer, scaler=None, max_norm=1.0):
    # 1. Clip first -- this alone tames most spikes (see below).
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    # 2. Collective finiteness check across ALL ranks.
    if not finite_grads_everywhere(model):
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.update()          # let GradScaler back off the loss scale
        return False, grad_norm      # skipped, weights unchanged everywhere

    # 3. Safe to step -- every rank agreed the grads are finite.
    if scaler is not None:
        scaler.step(optimizer); scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return True, grad_norm
```

Three things earn their place here. First, **`clip_grad_norm_` runs before the finiteness check**, because clipping is the cheap guard that prevents most spikes from ever reaching `inf`. It rescales the whole gradient so its L2 norm is at most `max_norm` (typically 1.0): if the raw norm is 8,300, clipping multiplies every element by `1.0 / 8,300`, and a 8,300-norm gradient becomes a 1.0-norm gradient — big, but finite and bounded, and the model survives the bad batch instead of exploding. Clipping does not help once a value is *already* `inf` (scaling `inf` is still `inf`), which is exactly why you also need the finiteness check as a second line. Second, **the check is a collective MIN over per-rank finite-flags**, so all 64 ranks reach the identical decision and skip or step together — no divergence. Third, **on a skip you still call `scaler.update()`** so that dynamic loss scaling backs the scale off; if you use bf16 without a scaler, drop that line.

Note the ordering relative to DDP. `clip_grad_norm_` and the finiteness check both need the *reduced* (all-reduced) gradients, so they must run *after* DDP's backward-pass all-reduce has completed — which, with standard DDP, is after `loss.backward()` returns. If you use gradient accumulation with `no_sync()`, clip and check on the final micro-batch of the accumulation window, after the sync. Get that ordering wrong and you clip local gradients that then get summed, which does not bound the sum — a subtle bug worth stating out loud.

With this guard installed, our step-12,000 NaN would have played out very differently: the grad-norm ramp would have been clipped down from 55 to 1.0 at step 11,990, likely preventing the overflow entirely; and if some element still overflowed, the finiteness check would have caught it, skipped one step across all 64 ranks in lockstep, and the run would have continued. The dashboard would show a single skipped step and an alert, not a wall of NaN. But — and this is the part that makes our specific war story interesting — a guard that skips the step does not tell you *why* the gradient was non-finite. It keeps the run alive. Finding the culprit is a separate hunt.

## Hunting the origin rank

Suppose the guard is not yet installed (it was not, that night), or suppose it *is* installed and keeps skipping steps and you need to know why. You have 64 identical NaN logs and you need the one rank that produced the NaN *first*, before the all-reduce laundered it into an anonymous global value. Figure 5 is the ladder we climb — from the cheapest always-on signal down to the hardware test that convicts a single node.

![vertical stack of debugging rungs from grad-norm monitoring down to a per-rank pre all-reduce norm log to a hardware test on the origin node](/imgs/blogs/silent-nan-at-scale-5.webp)

The key instrument is **per-rank, pre-all-reduce gradient-norm logging.** The insight is timing: the all-reduce makes all ranks identical, so you must measure each rank's gradient *before* it participates in the reduction. In DDP that is slightly awkward because DDP fuses the all-reduce into the backward pass via gradient hooks, so by the time `backward()` returns, the gradients are already reduced. The clean way to see local gradients is to register a hook that fires on each parameter's gradient *before* DDP's reduction, or — simpler for a one-off hunt — to temporarily wrap the model without DDP and run the same batch, or to log inside the autograd graph. For a production monitor, register a per-parameter hook and reduce a *scalar* (the local norm) with an op that preserves per-rank identity long enough to log it:

```python
import torch
import torch.distributed as dist

def log_local_grad_norm(model, step, log_fn):
    """Log THIS rank's gradient norm before the DDP all-reduce averages it.
    Run under model.no_sync() (or on a non-DDP replica) so grads stay local."""
    rank = dist.get_rank()
    local_sq = torch.zeros(1, device="cuda")
    for p in model.parameters():
        if p.grad is not None:
            local_sq += p.grad.detach().float().pow(2).sum()
    local_norm = local_sq.sqrt()

    # Gather every rank's local norm to rank 0 for a single ranked log line.
    world = dist.get_world_size()
    gathered = [torch.zeros(1, device="cuda") for _ in range(world)]
    dist.all_gather(gathered, local_norm)
    if rank == 0:
        norms = [g.item() for g in gathered]
        worst = max(range(world), key=lambda r: norms[r]
                    if norms[r] == norms[r] else float("inf"))
        log_fn(f"[step {step}] local grad-norms: "
               f"max=rank{worst}={norms[worst]:.3g}  "
               f"min={min(norms):.3g}  median={sorted(norms)[world//2]:.3g}")
```

Run this on the step where the NaN appears (or on every step during a hunt) and rank 0 prints a single line naming the worst rank. If sixty-three ranks report a local norm around 0.4 and one rank reports `inf` or a norm eight orders of magnitude larger, you have found the origin — the NaN was born on *that* rank, before the reduction, and everyone else inherited it. That is figure 7's whole story, which we will get to. If instead *all* ranks report a similarly elevated norm, the cause is not local to one rank — it is a shared numerics problem (overflow, LR, a bad batch that every rank happened to draw a version of), and you should be looking at the grad-norm ramp, not at hardware.

The second instrument is **`torch.autograd.set_detect_anomaly(True)`**, which finds the *operation* that first produced the NaN. It makes autograd check every backward op for non-finite outputs and raise a traceback pointing at the exact forward op whose backward went bad. It is slow — it roughly halves throughput because it validates every intermediate — so you never leave it on for a full run. You use it to *reproduce*: take the batch and step you know NaN'd, load the last good checkpoint, and rerun that batch with anomaly detection on. The traceback tells you it was, say, the backward of a `log` in your custom loss, or a `div` in a normalization, or a `softmax` at a long sequence.

```python
import torch

# Reproduction only -- never in a production run (roughly 2x slower).
torch.autograd.set_detect_anomaly(True)

# Load the last good checkpoint, feed the exact batch that NaN'd.
model.load_state_dict(torch.load("ckpt_step_10400.pt")["model"])
batch = dataset.get_batch(step=12000)     # the reproducible culprit batch
loss = model(**batch).loss
loss.backward()   # raises RuntimeError with a traceback at the first NaN op
```

The third instrument is **checkpoint bisection**: which batch, exactly, caused it? If the NaN is data-triggered, it is deterministic — the same batch at the same step produces the same NaN. Load the last good checkpoint, replay batches forward one at a time (or bisect the data-loader's index range), and watch the grad-norm. The batch where the norm first jumps is your culprit, and you can then inspect it: is it a pathologically long sequence? A document that is all one repeated token? A sample with a corrupt label? On more than one run, "the NaN" turned out to be a single 30,000-token document in a corpus tokenized without a length cap, producing an attention pattern that overflowed. The fix was a sequence-length filter, not a numerics change. Bisection is how you learn that.

## Software or hardware? The fork that decides everything

Now the two branches of the investigation split, and which branch you take is decided by *one observation*: did the grad-norm ramp up across many ranks over many steps, or did a single rank spike alone from a healthy baseline? Figure 6 is that decision tree.

![decision tree forking on whether the gradient norm ramped across ranks or one rank spiked alone leading to software or hardware fixes](/imgs/blogs/silent-nan-at-scale-6.webp)

If the norm **ramped over many steps** and the elevation was **shared across ranks**, it is a software numerics problem, and you already have the fixes: it is fp16 overflow (go bf16 or add dynamic scaling), or an LR-and-data spike (clip harder, filter sequences), or a `log(0)`/`div-by-0` in your own code (add an epsilon). These are reproducible, they will happen again on a rerun, and they are fixed by changing the math or the data, not the hardware. This is the common case, and it is the one every framework's stability tricks are designed for.

If instead a **single rank spiked alone**, from a baseline where every other rank was perfectly healthy, with **no ramp** — one step it is 0.4, the next it is 80 million on rank 37 and 0.4 everywhere else — then you are almost certainly looking at **hardware.** A bit flip in GPU memory (a failed ECC correction, or an uncorrectable multi-bit error) can turn a gradient element from `0.001` into a number with a corrupted exponent that reads as `1e30`, and there is no math reason for it because there is no math involved — a cosmic ray or a marginal DRAM cell flipped a bit. The tell is that it is *not reproducible from the checkpoint*: replay the exact same batch and the NaN does not come back, because the bit flip was a random hardware event, not a function of the data. That non-reproducibility is the signature. Software NaNs are deterministic; hardware NaNs are not.

When you suspect hardware, you stop debugging your model and start debugging the machine. Check the GPU's ECC error counters and the kernel log:

```bash
# Per-GPU ECC error counts -- volatile (since last reset) and aggregate.
nvidia-smi -q -d ECC | grep -A 20 "ECC Errors"

# Uncorrectable ECC or Xid errors show up in the kernel ring buffer.
dmesg -T | grep -iE "Xid|ECC|uncorrectable"

# DCGM's health check and diagnostic -- run on the suspect node.
dcgmi health -g 0 -c
dcgmi diag -r 3          # level 3 stress test, exercises memory + compute
```

A rising count of uncorrectable ECC errors, or an `Xid` error in `dmesg`, on the exact node that hosts rank 37, convicts the hardware. The fix is operational, not numerical: **cordon the node, evict it from the job, and restart from the last checkpoint on a healthy node.** In a job scheduler you drain that node and let the [elastic rendezvous](/blog/machine-learning/distributed-training/why-distributed-training) bring the run back up with a replacement. A single-bit ECC flip that the hardware *corrects* is fine and expected at scale (they happen constantly); an *uncorrectable* flip that reaches your gradient is the one that NaNs your run, and the counter distinguishes them.

## Worked example: the one-bad-node case

This is what our 3am run actually was — and it is worth walking through end to end, because it exercises every instrument above. After the guard-less NaN at step 12,000, we did three things in order. First, we loaded the step-10,400 checkpoint and replayed forward to step 12,000 with the per-rank grad-norm log turned on. Second, we watched the origin. Third, we checked the hardware. Figure 7 is the line that ended the mystery: the per-rank local gradient norms at step 12,000.

![grid of eight per-rank gradient norms where seven ranks sit near a small value and rank 37 reads eighty three million](/imgs/blogs/silent-nan-at-scale-7.webp)

Seven ranks in the window around rank 37 read local grad-norms between 0.38 and 0.42 — completely healthy, exactly where they had been for nine days. Rank 37 read `8.3e7` — eighty-three million. Eight orders of magnitude out, alone, with no ramp: at step 11,999 rank 37 had also been at 0.40. One step later it was 83 million, and one step after that the all-reduce had turned everyone's weights to NaN. That is not a numerics signature — a numerics blowup ramps and it is shared. This was a lone, instantaneous, single-rank spike. The decision tree in figure 6 pointed straight at hardware.

We replayed the exact same batch on rank 37's data from the checkpoint. No NaN. Non-reproducible — the hardware signature. Then we checked the node hosting rank 37 (node 4, local GPU 5):

```console
$ ssh node-04 'nvidia-smi -q -d ECC | grep -A6 "Aggregate"'
    Aggregate
        SRAM Correctable            : 41
        SRAM Uncorrectable          : 0
        DRAM Correctable            : 12043
        DRAM Uncorrectable          : 3
$ ssh node-04 'dmesg -T | grep -i Xid | tail -2'
[Wed Jul  2 03:04:11 2026] NVRM: Xid (PCI:0000:8d:00): 48, GPU memory DBE
[Wed Jul  2 03:04:11 2026] NVRM: Xid (PCI:0000:8d:00): 63, row remap pending
```

Three DRAM *uncorrectable* errors and an `Xid 48` — a double-bit error (DBE) in GPU memory — timestamped four minutes before our page, on the exact GPU running rank 37. That is the culprit. A DRAM cell went marginal, flipped two bits in a gradient buffer (past the point single-bit ECC can correct), and produced a gradient element with a garbage exponent reading tens of millions. The all-reduce summed it, NaN propagated, and 64 GPUs died for one bad DRAM cell on one card. We cordoned node-04, the scheduler brought up a spare, we restarted from step 10,400, and the run finished clean. Total loss: about two hours and a hardware RMA ticket. With the skip-step guard installed, it would have been one skipped step and an alert — the guard would not have *fixed* the bad node, but it would have kept the run alive long enough for the ECC counter to be the thing that paged us, at a civilized hour, instead of the dead loss.

The lesson that generalizes: **at 64 GPUs and beyond, hardware faults are not rare events, they are a rate.** A large cluster running for weeks will hit uncorrectable ECC errors, GPU falling off the bus, NVLink errors, and network flaps as a matter of statistics. Your training loop must treat a corrupt gradient from a bad node as a *routine* event to be survived — detected, skipped, and if it recurs on the same rank, escalated to eviction — not as an impossible one that takes down the run. The skip-step guard plus per-rank norm logging plus ECC monitoring is the minimum viable defense.

#### Worked example: the prevention checklist that would have saved the night

Concretely, here is the stack of guardrails, cheapest first, and what each one buys, sized against our incident:

1. **Gradient clipping (`clip_grad_norm_`, max 1.0).** Cost: one extra reduction per step, sub-millisecond. Catches: the loss-spike and LR-driven ramps — it would have flattened the 55 -> 8,300 climb before overflow. Does *not* catch a hardware `inf`.
2. **bf16 instead of fp16.** Cost: a few bits of mantissa precision; on a well-conditioned run, no measurable quality loss. Catches: the entire overflow-to-NaN pathway for real-valued gradients, because 83 million is nowhere near bf16's `3.4e38` ceiling. Worth noting: bf16 would *not* have saved us here, because a corrupted exponent bit can still produce a NaN pattern directly — but it removes four of the six causes.
3. **Skip-step on non-finite gradients (collective, all ranks agree).** Cost: near zero on healthy steps (one small all-reduce), one skipped step on a bad one. Catches: *everything* that reaches a non-finite gradient, including the hardware flip — this is the guard that would have kept our run alive.
4. **Per-rank grad-norm logging + alerting.** Cost: one `all_gather` of scalars per logged step. Catches: nothing by itself, but it is what lets you find the origin rank in seconds instead of hours.
5. **ECC / DCGM monitoring with per-node alerts.** Cost: a background daemon. Catches: the bad node *before* it flips a gradient, if you alert on rising uncorrectable counts.
6. **Sequence-length filtering + robust attention scaling.** Cost: a data-loader filter and a `1/sqrt(d)` you already have. Catches: the long-context softmax overflow that is a common hidden cause.

Install 1 through 4 on every run and you convert almost every NaN from a run-ending outage into a logged, skipped, attributed event. That is the whole payoff.

## The stable-softmax detail, because it bites at long context

One cause deserves its own note because it is subtle and increasingly common as context lengths grow: attention softmax overflow. Attention computes scores `S = Q K^T / sqrt(d_head)`, then `softmax(S)`, then `softmax(S) V`. The `softmax` of a row exponentiates every score: `exp(s_ij) / sum_j exp(s_ij)`. If any score `s_ij` is large — say 90 — then `exp(90)` overflows fp16 (whose ceiling is 65,504) *immediately*, because `exp(11.1)` already exceeds it. In fp32 it takes until about `exp(88)` to overflow, and in bf16 you have the range but you can still hit it. The classic guard is the **max-subtraction trick**: subtract the row max before exponentiating, `exp(s_ij - max_j s_ij)`, which is mathematically identical (the max cancels in the ratio) but keeps the largest exponent at `exp(0) = 1` and everything else below it, so nothing overflows. Every correct softmax implementation does this; FlashAttention does it in its online-softmax rescaling. The bugs come from *custom* attention that forgot it, or from an unmasked padding position that lets a score run unbounded, or from a `1/sqrt(d_head)` scale that got dropped in a refactor. If your NaN correlates with sequence length and lives inside attention, this is your cause, and the fix is to ensure the scale and the max-subtraction and the mask are all present.

```python
import torch
import torch.nn.functional as F

def stable_attention(q, k, v, mask=None):
    d_head = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / (d_head ** 0.5)   # DON'T drop this scale
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))  # mask BEFORE softmax
    # F.softmax subtracts the row max internally -- numerically stable.
    attn = F.softmax(scores, dim=-1)
    return attn @ v
```

The reason to prefer `F.softmax` (or `scaled_dot_product_attention`, or FlashAttention) over a hand-rolled `exp`-and-normalize is precisely that the library does the max-subtraction for you. The most common way people reintroduce this NaN is by writing their own softmax "for clarity" and dropping the stabilization. Do not.

## Case studies: how the big runs handled it

This is not a niche problem; it is the defining operational hazard of large-model pretraining, and the published training logs are candid about it.

**PaLM (540B, Google, 2022)** reported loss spikes roughly twenty times during training *despite* gradient clipping. Their diagnosis was instructive: the spikes were not caused by a single bad batch, because when they restarted from a checkpoint before the spike and skipped the batches around it, the spike did not recur — but restarting from the same checkpoint over the *same* batches *did* reproduce it. Their conclusion was that a spike arose from the specific combination of model state and a particular sequence of batches. Their mitigation was operational: restart from a checkpoint about a hundred steps before the spike and skip roughly two hundred to five hundred data batches. This is checkpoint bisection and batch-skipping as a production procedure, at the largest scale.

**OPT-175B (Meta, 2022)** published an unusually honest chronicle in their logbook: dozens of loss divergences, frequent manual restarts, lowering the learning rate to recover, switching to fresh data shards, and a steady drumbeat of hardware failures across the cluster. The takeaway they stated plainly is the one this post is built around — at that scale, instability and hardware faults are *routine*, and the run is a sequence of restarts managed by an on-call human as much as it is a training job. Their experience is why modern stacks bake in the skip-step and elastic-restart machinery.

**GLM-130B (Tsinghua, 2022)** documented persistent loss spikes and traced part of the instability to attention scores growing large enough to cause problems in their mixed-precision setup; their write-up discusses embedding-gradient shrinking and careful attention handling as stabilizers. It is a concrete instance of the attention-overflow cause being real at scale, not a textbook hypothetical.

The common thread across all three: **nobody at the frontier treats a NaN or a loss spike as a surprise.** They instrument for it, they clip, they monitor grad-norm, they checkpoint frequently enough that a restart costs minutes not days, and they have a documented procedure — skip these batches, lower this LR, evict this node — for when it happens. Frequent checkpointing is itself a NaN guardrail: the entire cost of a NaN is the distance back to the last good checkpoint, so a run that checkpoints every 250 steps loses ten minutes to a NaN that a run checkpointing every 5,000 steps loses three hours to. (Sharded and async checkpointing, which make frequent saves cheap, are their own topic in this series.)

## When to reach for this, and when not to

Not every guardrail belongs on every run, so here is the decisive version.

**Always install, on every distributed run:** gradient clipping, the collective skip-step guard, per-rank grad-norm logging, and frequent checkpoints. These cost almost nothing on healthy steps and they are the difference between a survivable NaN and a lost night. There is no run large enough to skip them and no run small enough that they hurt.

**Reach for bf16 over fp16** on any hardware that supports it (A100, H100, and newer). fp16 with `GradScaler` is a legitimate and well-supported path, but bf16 removes the whole overflow-to-NaN failure mode by construction, and on modern accelerators it is free. Stay on fp16 only if your hardware lacks bf16 Tensor Core support or you have a specific validated reason.

**Reach for `detect_anomaly` and checkpoint bisection** only when you are actively hunting a reproducible NaN — never leave anomaly detection on in production, it roughly halves throughput. They are reproduction tools, not monitoring tools.

**Reach for the hardware playbook** (ECC counters, `dcgmi diag`, node eviction) the moment you see a *lone rank spiking with no ramp and no reproducibility*. Do not waste hours re-checking your loss function for a bug that is a bad DRAM cell. The non-reproducibility and the single-rank signature are your cue to switch from debugging the model to debugging the machine.

**Do not** reach for exotic numerics interventions — custom gradient rescaling, per-layer loss scales, precision surgery — before you have clipping, skip-step, and bf16 in place. Ninety percent of NaNs are solved by those three plus finding the origin rank. Save the exotic stuff for the residual ten percent that survives the basics, and even then, suspect your data (a corrupt sample, an unfiltered long sequence) before you suspect the arithmetic.

## Key takeaways

- **The all-reduce is why a local NaN is a global outage.** Gradient all-reduce *sums* across ranks, and `NaN + anything = NaN`, so one rank's non-finite gradient becomes every rank's NaN weights in a single step. A distributed NaN looks simultaneous because the collective synchronizes the failure.
- **Find the origin rank, not the symptom.** By the time rank 0 sees NaN, the all-reduce has erased which rank started it. Log per-rank gradient norms *before* the reduction; the origin is the one rank that spiked first.
- **The grad-norm is a leading indicator; the loss is a lagging one.** Overflow-driven NaNs ramp geometrically over hundreds of steps before crossing the float ceiling. Monitor and alert on grad-norm and you catch the fire before it reaches NaN.
- **Skip the step, do not crash — and make all ranks skip together.** Check gradient finiteness after the all-reduce, all-reduce a finite-flag with MIN so every rank agrees, and skip the optimizer step in lockstep. A skipped step costs nothing; divergence from a partial skip costs everything.
- **Clip first, then check finiteness.** `clip_grad_norm_(..., 1.0)` bounds the spike before it overflows; the finiteness check catches whatever a hardware flip or a residual overflow slips past clipping.
- **Software NaNs ramp and reproduce; hardware NaNs spike alone and vanish on replay.** That one distinction — did it ramp across ranks, or spike on one rank with no ramp — forks the whole investigation between numerics fixes and node eviction.
- **bf16 removes four of the six causes.** Its fp32-range exponent makes overflow-to-NaN nearly impossible; prefer it over fp16 wherever the hardware supports it.
- **At scale, hardware faults are a rate, not an event.** A weeks-long 64-GPU run *will* hit uncorrectable ECC errors. Treat a corrupt gradient as routine: detect, skip, and evict on recurrence. Check ECC counters and `dmesg` when a lone rank spikes.
- **Frequent checkpoints are a NaN guardrail.** The cost of any NaN is the distance to the last good checkpoint. Checkpoint every few hundred steps and a NaN costs minutes, not hours.

## Further reading

- [Why distributed training](/blog/machine-learning/distributed-training/why-distributed-training) — the four walls and the series map; the frame this war story sits in.
- [DDP from first principles](/blog/machine-learning/distributed-training/ddp-from-first-principles) — how the gradient all-reduce actually works, the reduction that spreads the NaN.
- [Collectives from scratch](/blog/machine-learning/distributed-training/collectives-from-scratch) — all-reduce, all-gather, and reduce-scatter in detail, including the ring byte volume.
- [Mixed precision at scale](/blog/machine-learning/distributed-training/mixed-precision-at-scale) — bf16 vs fp16, loss scaling, and the numerics that turn a big gradient into an `inf`.
- [The straggler](/blog/machine-learning/distributed-training/the-straggler) — the sibling war story: one slow node instead of one bad node, and how to find it.
- [The distributed training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) — the capstone checklist that ties the guardrails here into the whole operational picture.
- [Mixed-precision debugging: fp16 vs bf16](/blog/machine-learning/debugging-training/mixed-precision-debugging-fp16-vs-bf16) — a deeper debugging drill on the precision formats behind the overflow cause.
- [Out-of-memory debugging](/blog/machine-learning/debugging-training/out-of-memory-debugging) — the other failure mode that pages you at 3am, and how activation memory interacts with it.
- Chowdhery et al., *PaLM: Scaling Language Modeling with Pathways* (2022) — the loss-spike-and-skip-batches procedure at 540B.
- Zhang et al., *OPT: Open Pre-trained Transformer Language Models* (2022) and the OPT training logbook — the candid chronicle of divergences, restarts, and hardware failures.

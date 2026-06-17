---
title: "Finetuning Pitfalls Across Modalities: A Unified Checklist for Vision, LLMs, Speech, and Tabular"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "The same finetuning bugs ruin a ViT, a Llama, a Whisper, and an XGBoost model — the 100x-too-high learning rate, the silent catastrophic forgetting, the preprocessing that doesn't match pretraining. Here is one unifying picture and one runnable checklist that catches all of them before you waste a run."
tags:
  [
    "debugging",
    "model-training",
    "finetuning",
    "transfer-learning",
    "computer-vision",
    "llm",
    "speech",
    "tabular",
    "learning-rate",
    "catastrophic-forgetting",
    "pytorch",
    "peft",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/finetuning-pitfalls-across-modalities-1.png"
---

A team finetunes a vision transformer on 4,000 product photos and the accuracy drops from the pretrained model's 81% to 34% — worse than chance-adjusted random. A second team finetunes Llama on 8,000 support tickets and the model now answers ticket questions beautifully but can no longer add two numbers or follow a simple instruction it handled fine before. A third team finetunes Whisper on call-center audio and the word error rate goes from 9% to 71%, as if it forgot how to transcribe speech entirely. A fourth team finetunes an XGBoost model on top of a learned feature encoder and gets 0.96 AUC in cross-validation and 0.74 in production.

Four teams, four modalities, four "the finetune broke" Slack threads. And here is the thing that took me years of losing weekends to see: **it is the same bug, four times.** Not a similar bug. The same root cause, wearing four costumes. In every one of those stories someone used a learning rate meant for training from scratch, or fed the pretrained model an input distribution it had never seen, or trained for so many epochs that the model memorized the small finetune set and bulldozed everything it knew. The modality changes the symptom and the API; it does not change the disease.

The reason this matters is economic as much as technical. Most teams will never pretrain a model — pretraining is the province of a handful of labs with eight-figure compute budgets. But *everyone* finetunes. Finetuning is how a pretrained model becomes your product, and it is where the overwhelming majority of practitioners actually touch training. So the finetuning failure modes are the ones you will hit again and again across your career, and learning to recognize them once — across all modalities at once — pays off every time you adapt a new model in a new domain. The eight pitfalls below are not exotic edge cases; they are the daily failure modes of applied ML, and the checklist that catches them is the most reusable debugging artifact in this entire series.

![Before and after a learning-rate fix shown side by side for a vision transformer, an LLM, and a speech model, where the from-scratch rate destroys all three and a smaller rate recovers all three](/imgs/blogs/finetuning-pitfalls-across-modalities-1.png)

This post is a cross-modal synthesis. It is the post I wish someone had handed me before I started shipping finetunes: one unifying thesis, eight recurring pitfalls, and a single checklist that works whether you are nudging a ResNet, a 7B language model, a wav2vec2 encoder, or a gradient-boosted tree on top of embeddings. The thesis fits on one line — **finetuning is "nudge a good model, don't retrain it."** Every pitfall in this post is a violation of that one line. By the end you will be able to take any broken finetune in any modality, run a five-minute preflight, and localize the bug to one of eight named failure modes before you touch a single hyperparameter.

We will keep the series' spine. A training bug hides in one of six places — **data, optimization, model code, numerics, systems, evaluation** — and you bisect to the right one before changing code, using two master tools: **make-it-fail-small** (overfit one batch, probe one held-out set) and **read the instruments** (loss curve, grad norm, a general-capability probe). Finetuning narrows the search dramatically, because a pretrained model is already known to work. If it stops working after you touch it, the bug is almost always in *how you touched it*. That is good news: it means the suspect list is short, and it is the same short list every time. For the broader frame, see [a taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs); this post is the cross-modal instantiation of it, and the [training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) is the capstone that ties every track together.

## 1. The unifying thesis: a pretrained model lives in a sharp basin

Before any code, you need the picture that predicts every pitfall in this post. Here it is.

A pretrained model is not a random point in weight space. It is a point at the bottom of a carefully carved valley in the loss landscape — a *basin* — where the features it has learned (edges and textures for a ViT, syntax and world facts for an LLM, phonetic structure for an ASR model) are encoded in the relative geometry of its weights. The pretraining run spent enormous compute (often hundreds of thousands of GPU-hours and trillions of tokens or billions of images) sliding the weights to the bottom of that basin. The basin is *sharp* along most directions: move the weights a little and the loss on the original pretraining distribution shoots up, which is to say the model forgets.

Finetuning's job is to move you to a *nearby* point — same basin, slightly shifted toward your task — without launching the weights out of the basin entirely. That is the whole game. And it immediately explains the central tension of finetuning, the **plasticity–stability tradeoff**: you need enough plasticity (weight movement) to learn the new task, but enough stability (weight conservatism) to keep the prior. Push too hard and you get plasticity at the cost of stability — the model learns the task and forgets everything else. Push too softly and you get stability at the cost of plasticity — the model keeps the prior but never learns the task.

Now the math that makes this concrete. The size of the weight step at one optimizer update is, to first order, $\Delta\theta \approx -\eta \, g$, where $\eta$ is the learning rate and $g$ is the gradient. The total distance you travel from the pretrained point over a run is roughly $\sum_t \eta \, \lVert g_t \rVert$, summed over all steps $t$. Two knobs control that distance directly: the **learning rate** $\eta$ and the **number of steps**. A from-scratch learning rate (say $10^{-3}$) is calibrated to move *random* weights a long way fast, because random weights are nowhere near a good solution and need to travel. Apply that same $\eta$ to weights that are *already* at the bottom of a sharp basin and the first few updates hurl you out of it. The gradient at the start of a finetune is large (the task head is random, so its loss is high), so $\eta \lVert g \rVert$ is large, and you can leave the basin in *the first ten steps* — before you have learned anything. That is why a too-high finetuning LR produces the signature loss spike followed by garbage: the model is not slowly degrading, it has been ejected from the basin almost immediately.

This single picture — pretrained weights at the bottom of a sharp basin, finetuning as a short conservative walk that must not leave it — is the generator of every pitfall below. Catastrophic forgetting is "you left the basin." The wrong LR is "your steps were too big." Too many epochs is "you took too many steps and wandered out." Layer-wise LR is "the basin is sharper in the lower layers, so step smaller there." Preprocessing mismatch is "you fed the model inputs from outside the basin's input distribution, so even step zero is broken." Hold this model in your head and the rest of the post is corollaries.

![A vertical stack of the six finetuning layers from input format down to evaluation, illustrating that each pitfall is a single layer where a too-aggressive change overwrites the pretrained prior](/imgs/blogs/finetuning-pitfalls-across-modalities-2.png)

It helps to make the "sharp basin" claim a little more rigorous, because the sharpness is not a metaphor — it is a measurable property of the loss surface. Expand the pretraining loss $\mathcal{L}_0$ around the pretrained point $\theta_0$ in a second-order Taylor series: $\mathcal{L}_0(\theta_0 + \delta) \approx \mathcal{L}_0(\theta_0) + \tfrac{1}{2}\,\delta^\top H \,\delta$, where the linear term vanishes because $\theta_0$ is (approximately) a minimum, and $H$ is the Hessian of the pretraining loss. The eigenvalues of $H$ are the curvatures along each direction. "Sharp basin" means $H$ has some large eigenvalues — move $\delta$ along a high-curvature eigenvector and $\mathcal{L}_0$ (the *pretraining* loss, i.e. the general capability) rises fast. Finetuning moves $\theta$ along directions chosen by *your task's* gradient, which is generally not aligned with the low-curvature directions of $H$, so your update has a component along the sharp directions and that component is what causes forgetting. This is exactly why EWC and its relatives work: they add a penalty $\tfrac{\lambda}{2}\sum_i F_i (\theta_i - \theta_{0,i})^2$ where $F_i$ is the diagonal of the Fisher information (a cheap proxy for $H$), penalizing movement most along the sharp directions. You do not need to implement EWC to benefit from the insight — the insight is that *some directions are cheap to move and some are expensive*, and a small LR plus few steps keeps your total movement small enough that even the expensive directions stay safe.

One more consequence worth stating up front, because it reframes the whole post: **finetuning failures are systematically different from from-scratch failures, and they are systematically the same across modalities.** A from-scratch run can fail because the architecture can't represent the function, because initialization collapsed the signal, because the data is too small to learn from at all. A finetune almost never fails for those reasons — the architecture is proven, the init is a working model, and even a few thousand examples are enough to adapt a model that already knows the domain. So the finetune failure space is *narrower* and *more stereotyped*. That is why one checklist can cover four modalities: the things that go wrong are the eight knobs you turned, not the open-ended space of things that can go wrong when you build a model from nothing.

## 2. Pitfall one: the wrong learning rate (the same mistake in every modality)

This is the single most common finetuning bug, and it is identical across modalities. The rule: **finetuning needs a learning rate roughly 10–100x smaller than training from scratch.** From-scratch ResNet training might use $10^{-1}$ with SGD or $10^{-3}$ with Adam; finetuning the same backbone wants $10^{-4}$ to $10^{-5}$. From-scratch transformer pretraining uses peak LRs around $10^{-3}$ to $3\times10^{-4}$; finetuning an LLM wants $10^{-5}$ to $2\times10^{-5}$. The mechanism is exactly section 1: a large LR makes $\eta\lVert g\rVert$ big enough to leave the basin in the first few steps.

The symptom is a *loss spike in the first 10–100 steps*, sometimes followed by a slow recovery to a much worse plateau, sometimes by NaN. Crucially, training loss can *look like it is going down* if you only glance at it after step 500 — the model partially re-learns a degraded solution on your tiny finetune set. That is why the LR bug fools people: the curve eventually descends, so it "looks like it is training." It is training, but from a wrecked starting point. For the full LR diagnosis playbook — the LR range test, warmup, the spike-vs-divergence distinction — see [the learning rate is almost always the problem](/blog/machine-learning/debugging-training/the-learning-rate-is-almost-always-the-problem); here we focus on the finetuning-specific version.

The diagnostic is a short **LR sweep with the basin as the reference**. Pick a tiny held-out probe of the *original* capability (more on this in section 3), and sweep LR over a log grid, watching both task loss and the probe. The right LR is the largest one where the probe does not collapse in the first few hundred steps.

```python
import torch, copy

# lr_grid spans from "definitely too small" to "from-scratch territory"
lr_grid = [1e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 5e-4, 1e-3]
base_state = copy.deepcopy(model.state_dict())

results = []
for lr in lr_grid:
    model.load_state_dict(base_state)          # reset to the pretrained basin
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for step, batch in enumerate(train_loader):
        loss = model(**batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        if step == 150:                         # short burn-in is enough
            break
    probe_acc = evaluate_general_probe(model)   # the held-out prior probe
    task_loss = loss.item()
    results.append((lr, task_loss, probe_acc))
    print(f"lr={lr:.0e}  task_loss={task_loss:.3f}  probe_acc={probe_acc:.3f}")
```

You are looking for the LR where task loss drops *and* `probe_acc` stays near baseline. Two implementation notes that people miss. First, **reset to the pretrained basin between LR trials** (`model.load_state_dict(base_state)`); if you sweep on the same model without resetting, each trial inherits the damage from the previous one and the sweep is meaningless. Second, **clip the gradient** (`clip_grad_norm_`) during the sweep so a single large early gradient from the random head does not detonate the run before you can read the signal — clipping turns the first-step spike from a catastrophe into a survivable bump, which is itself one of the standard finetuning guardrails.

There is a sharper, single-run version of this called the **LR range test** (the technique behind Leslie Smith's cyclical-LR work). Instead of a discrete grid, you start at a tiny LR and multiply it by a constant factor every step, plotting loss vs LR on a log axis. You will see three regimes: a flat region where the LR is too small to move loss, a steep descending region where the LR is in the productive range, and an explosion where the LR is too large and loss diverges. The good finetuning LR is roughly one order of magnitude below the LR at the explosion point — comfortably inside the descending region, with margin so a noisy batch doesn't tip you into the explosion. For a finetune you want to be *conservative* about that margin, because the cost of being too high (basin damage, which is partly irreversible) is much worse than the cost of being too low (a slightly slower run). For the full LR-range-test recipe and the warmup/schedule discussion, see [the learning rate is almost always the problem](/blog/machine-learning/debugging-training/the-learning-rate-is-almost-always-the-problem).

**Warmup deserves a sentence on mechanism, because it is finetuning's seatbelt.** Warmup ramps the LR from ~0 to its peak over the first few hundred steps. Per section 1, the most dangerous moment of a finetune is step 1, when the random head produces a huge gradient that, at full LR, would kick the body out of the basin. Warmup makes those first steps tiny, so the head's wild early gradients are scaled down to near-harmless, and by the time the LR reaches its peak the head has stabilized and its gradients are reasonable. This is also why warmup "hides init sins" — a bad head init that would blow up at full LR survives if warmup gives it time to settle. For a finetune, 3–6% of total steps as warmup is a safe default, and on very small datasets even a fixed 100–500-step warmup helps.

In practice the LR sweep table looks like this — and it looks the same shape in every modality.

| LR | Task loss @ step 150 | General probe (baseline 62%) | Verdict |
|---|---|---|---|
| 1e-6 | 1.41 | 62% | too small, barely learning |
| 1e-5 | 0.91 | 61% | healthy |
| 2e-5 | 0.74 | 60% | best — learns, keeps prior |
| 5e-5 | 0.68 | 55% | learning fast, prior eroding |
| 1e-4 | 0.59 | 41% | basin damage starting |
| 5e-4 | 3.2 (spiked) | 14% | ejected from basin |
| 1e-3 | NaN | 9% | dead |

#### Worked example: a ViT finetune that "wouldn't learn"

A vision transformer (`vit_base_patch16_224` from `timm`, pretrained on ImageNet) is finetuned on a 4,000-image product-defect dataset (12 classes). The engineer copies the from-scratch recipe: AdamW, LR $10^{-3}$, cosine schedule, 30 epochs. Validation accuracy after epoch 1 is 28%, climbs to 41% by epoch 10, and plateaus. The pretrained linear-probe baseline (frozen backbone, train only the head) was 79%. So the *full finetune is worse than just probing*. That is the giveaway: if full finetuning underperforms a linear probe, you have damaged the backbone.

The bisection: is it data, model, or optimization? Overfit one batch (32 images). With LR $10^{-3}$ the single-batch loss spikes to 6.1 then settles at 2.0 and *cannot* reach zero — a model that cannot memorize 32 images is being held back by optimization, not data. Drop the LR to $5\times10^{-5}$ for the body and keep $10^{-3}$ for the new head (layer-wise LR, section 6), and single-batch loss drops to 0.002 in 40 steps. Re-run the full finetune: validation accuracy hits 92% by epoch 8. The "data problem" was an LR problem, and the overfit-one-batch test localized it in two minutes. (See [the overfit a single batch test](/blog/machine-learning/debugging-training/the-overfit-a-single-batch-test) for why this is the highest-leverage check in the whole series.)

The numbers — 81% to 34% with the wrong LR, 81% to 92% with the right one — are the same story you would tell for an LLM (probe 62% to 9% vs 62% to 60%) or an ASR model (WER 9% to 71% vs 9% to 5.4%). Figure 1 above puts all three side by side: one knob, four modalities, identical shape.

## 3. Pitfall two: catastrophic forgetting (and how to actually see it)

Catastrophic forgetting is when narrow finetuning erases general capability. The model gets great at your task and worse at everything else — sometimes dramatically worse. It is the most *dangerous* finetuning bug precisely because **it is invisible in your training loss and usually invisible in your task validation set too.** Your task metric goes up; you ship; production breaks on the 80% of traffic that wasn't your narrow task.

The mechanism is section 1 again: forgetting is "you walked too far from the basin." Every gradient step optimizes *your* task's loss, and nothing in that objective rewards keeping the original capability. So the weights drift toward your task and away from the pretraining solution. The drift is proportional to total distance traveled, $\sum_t \eta \lVert g_t\rVert$ — which means the same two knobs (LR and steps) control forgetting and task learning *simultaneously*, and pushing either one harder buys task accuracy at the price of the prior. There is a real frontier here and you have to pick a point on it.

We can quantify the forgetting a single update causes. The change in the *pretraining* loss $\mathcal{L}_0$ from one finetuning step $\delta = -\eta g_{\text{task}}$ is, to first order, $\Delta \mathcal{L}_0 \approx \nabla \mathcal{L}_0 \cdot \delta = -\eta \,(\nabla\mathcal{L}_0 \cdot g_{\text{task}})$. If the task gradient $g_{\text{task}}$ points *against* the pretraining gradient $\nabla\mathcal{L}_0$ (a negative dot product), then $\Delta\mathcal{L}_0$ is positive — the pretraining loss *rises*, which is forgetting. And the size of that rise scales linearly with $\eta$. This is the formal statement of "lower the LR to forget less": halve the LR and you roughly halve the per-step forgetting. It also explains why **data mixing works**: if you mix general data into the batch, the effective gradient becomes $g = (1-\alpha)\,g_{\text{task}} + \alpha\,\nabla\mathcal{L}_0$ for mixing fraction $\alpha$, which pulls the update partly back along the pretraining gradient and shrinks the conflicting component. You are no longer optimizing purely against the prior; you are optimizing a weighted sum that includes it.

There is a subtlety worth calling out: forgetting is *not uniform across capabilities*. A finetune on support tickets does not forget all general ability equally — it tends to forget the capabilities whose representations overlap most with the directions your task gradient pushes. In practice that means narrow, stylistically distinctive finetune data (a single tone of voice, a rigid format) forgets *more* of general instruction-following than diverse data does, because the narrow data drives the weights consistently in one direction step after step. This is why "diversify the finetune data" is itself a forgetting mitigation: diverse data produces gradients that partly cancel, so the cumulative drift along any single direction is smaller even at the same LR and step count.

![A general probe scored alongside training loss reveals catastrophic forgetting that the smooth task loss curve completely hides](/imgs/blogs/finetuning-pitfalls-across-modalities-4.png)

**The detection trick is a held-out general probe.** Before you finetune, assemble a small, frozen set of examples that represent the *original* capability you want to keep — not your task. Score it at baseline, then re-score it at every checkpoint. The probe converts an invisible regression into a number you can put a threshold on. This is the single most valuable habit in cross-modal finetuning, and it is the same idea everywhere:

- **Vision**: a held-out slice of the original pretraining-style classes (e.g. 500 ImageNet val images) — does top-1 hold?
- **LLM**: 100–300 general questions (a slice of MMLU or a fixed instruction set) the base model handled — does accuracy hold?
- **Speech**: a held-out set from a *different* domain than your finetune data (read speech if you finetune on call-center audio) — does WER hold?
- **Tabular**: performance on the pre-finetune feature representation across the broader distribution, not just your target slice.

```python
# A modality-agnostic forgetting probe. score_fn(model) -> float (higher = better).
class ForgettingProbe:
    def __init__(self, model, score_fn, gate_pts=4.0):
        self.score_fn = score_fn
        self.baseline = score_fn(model)        # measure BEFORE any training
        self.gate = gate_pts                    # allowed drop, in points
        self.history = [self.baseline]
        print(f"probe baseline = {self.baseline:.3f}")

    def check(self, model, step):
        s = self.score_fn(model)
        self.history.append(s)
        drop = (self.baseline - s)
        flag = "  <-- FORGETTING" if drop > self.gate / 100 else ""
        print(f"step {step}: probe={s:.3f}  drop={drop*100:+.1f} pts{flag}")
        return drop <= self.gate / 100          # False -> trip early stop

# usage inside the training loop
probe = ForgettingProbe(model, score_fn=evaluate_general_probe, gate_pts=4.0)
for step, batch in enumerate(train_loader):
    train_step(model, batch, opt)
    if step % eval_every == 0:
        if not probe.check(model, step):
            print("early stop: probe dropped past gate")
            break
```

The **mitigations** are, again, the same in every modality, ordered from cheapest to most involved:

1. **Lower the LR and take fewer steps.** This directly shrinks $\sum_t \eta\lVert g_t\rVert$. The cheapest fix and often sufficient.
2. **Parameter-efficient finetuning (PEFT/LoRA).** Freeze the pretrained weights entirely and learn a low-rank update. The basin literally cannot move because $\theta$ is frozen; only the adapter changes. This is the strongest structural defense against forgetting, and it works for vision, LLMs, and speech. See [debugging LoRA and PEFT](/blog/machine-learning/debugging-training/debugging-lora-and-peft) for the no-op traps that make people *think* LoRA failed when it never entered the graph.
3. **Data mixing / replay.** Mix 10–30% general-distribution data back into your finetune batches so the objective itself rewards keeping the prior. This is the standard fix for LLM instruction-tuning forgetting and the rehearsal idea from continual-learning research.
4. **Regularize toward the base weights.** An L2 penalty $\lambda \lVert \theta - \theta_0\rVert^2$ toward the pretrained weights $\theta_0$ (or the Fisher-weighted version, EWC) explicitly tethers you to the basin.

#### Worked example: an LLM that learned support tickets and forgot arithmetic

An instruction-tuned 7B model is finetuned on 8,000 support-ticket Q&A pairs. LR $2\times10^{-5}$ (correct), but **5 epochs** and **no general probe**. Ticket-eval accuracy climbs from 51% to 89% — looks great, ship it. In production, complaints: the model now refuses to do basic reasoning, miscounts, and ignores formatting instructions it used to follow.

Post-mortem with a 200-question general probe (a fixed MMLU + arithmetic + instruction-following slice) scored from the saved checkpoints: baseline 62%, epoch 1 → 60%, epoch 2 → 57%, epoch 3 → 51%, epoch 5 → 44%. The model lost **18 points** of general capability while the ticket metric rose. The probe, had it been running, would have tripped the −4-point gate at epoch 2 and saved the run. The fix combined three of the mitigations: cap at 2 epochs, mix 20% general instruction data into the finetune set, and switch to LoRA (r=16, alpha=32 on `q_proj` and `v_proj`). Re-run: ticket eval 86% (−3 from the over-trained version, fine), general probe 59% (−3 from baseline, within gate). Shipped clean. This exact recipe — LR, epochs, forgetting probe, replay — is the subject of [finetuning an LLM without breaking it](/blog/machine-learning/debugging-training/finetuning-an-llm-without-breaking-it); the point here is that *the identical probe-and-replay pattern fixes the vision and speech versions too.*

The dollar cost of *not* having the probe is concrete. The over-trained run was 5 epochs on 8 A100s; at roughly \$2 per GPU-hour that is a few hundred dollars of compute, plus the far larger cost of shipping a regressed model and the days of debugging the production complaints. The probe-instrumented re-run was 2 epochs — *cheaper* in compute — and caught the problem before it ever reached production. The forgetting probe is the rare debugging tool that saves money on the same run it protects.

## 4. Pitfall three: frozen vs full vs PEFT (and the optimizer-before-unfreeze bug)

How much of the model should you actually train? There are three canonical answers, and choosing wrong wastes compute or accuracy. They sit on a spectrum of how much you let the weights move — which, per section 1, is the same axis as how much you risk leaving the basin.

![Decision tree routing the freeze choice to a linear probe, LoRA, or a full finetune based on dataset size and domain distance](/imgs/blogs/finetuning-pitfalls-across-modalities-5.png)

- **Linear probe** (freeze the body, train only a new head). Cheapest, fastest, *cannot* forget the body's features because they are frozen. Best when your data is small (< ~5k examples) and your domain is near the pretraining domain. The pretrained features are already good; you just need a classifier on top.
- **Full finetune** (train everything). Most plastic, highest ceiling, highest forgetting risk. Best when you have a lot of data (> ~50k) and/or a far domain where the features themselves need to adapt. Pair it with layer-wise LR (section 6) and a forgetting probe.
- **LoRA / PEFT** (freeze the body, learn a low-rank update injected into chosen layers). The sweet spot for mid-sized data or far domains where a probe is too weak but a full finetune forgets too much. You train ~0.1–1% of the parameters, the base weights never move (so forgetting is structurally bounded), and you can carry many adapters for one base model.

The decision rule is the tree in figure 5: small + near → probe; mid or far → LoRA; large + far → full finetune with layer-wise LR. The freeze schedule is a fourth option that interpolates: start frozen (train the head until it stabilizes), then progressively unfreeze from the top down. This warms up the random head before letting it backpropagate large early gradients into the pretrained body — which protects the basin during the most dangerous first steps.

**Now the bug that bites everyone who freezes layers: the optimizer-before-unfreeze ordering bug.** If you create the optimizer over `model.parameters()` *before* you unfreeze a layer (set `requires_grad=True` later), that layer is **not in the optimizer's parameter groups** and will never update — even though it now has gradients. The model silently trains only the originally-trainable params. Symptom: you "unfreeze the backbone at epoch 5" and nothing changes; the loss curve has no kink at the unfreeze step.

```python
# WRONG: optimizer captures only the params that were trainable at creation time.
for p in model.backbone.parameters():
    p.requires_grad = False
opt = torch.optim.AdamW(model.parameters(), lr=2e-5)   # head + (frozen) backbone
# ... epoch 5: unfreeze ...
for p in model.backbone.parameters():
    p.requires_grad = True                              # grads flow now...
# ...but opt never got these params. They will not update. Silent no-op.

# RIGHT (option A): create the optimizer AFTER unfreezing, or rebuild it.
for p in model.backbone.parameters():
    p.requires_grad = True
opt = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=2e-5
)

# RIGHT (option B): pass ALL params up front; freezing via requires_grad still
# zeroes their grads, so frozen params get no update even though they're in opt.
opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
# then toggle requires_grad over the schedule; opt already owns every param.
```

This is a gradient-flow bug, and the way you confirm it is the gradient-flow check from [your model isn't learning what you think](/blog/machine-learning/debugging-training/your-model-isnt-learning-what-you-think): after a backward pass, print which parameters actually received a gradient and which are in the optimizer's param groups, and look for the mismatch. The check is three lines and it catches the no-op instantly:

```python
loss.backward()
opt_params = {id(p) for grp in opt.param_groups for p in grp["params"]}
for name, p in model.named_parameters():
    if p.requires_grad and id(p) not in opt_params:
        print(f"!! {name}: requires_grad but NOT in optimizer -> will never update")
    if p.requires_grad and p.grad is None:
        print(f"?? {name}: trainable but grad is None -> not in the graph")
```

The same two messages — "trainable but not in optimizer" and "trainable but grad is None" — catch the optimizer-ordering bug *and* the LoRA-adapter-not-in-the-graph bug *and* the accidentally-detached-submodule bug. One check, many disguises.

#### Worked example: a LoRA finetune that "trained" but changed nothing

A team applies LoRA to a 7B model, runs it for 3 epochs, and the eval is identical to the base model — not worse, *identical*. Their first instinct is "the LR is too low." It is not. The trainable-but-no-grad check prints nothing alarming, but the *count* of trainable parameters is the tell:

```python
from peft import LoraConfig, get_peft_model
cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["query", "value"])  # BUG
peft_model = get_peft_model(base_model, cfg)
peft_model.print_trainable_parameters()
# trainable params: 0 || all params: 6,738,415,616 || trainable%: 0.0000
```

`trainable%: 0.0000` is the smoking gun: the `target_modules` names (`"query"`, `"value"`) do not match this architecture's module names (which are `"q_proj"`, `"v_proj"`). LoRA injected adapters into *nothing*, so there were no trainable parameters, so every step was a no-op regardless of LR. The fix is one line — correct the `target_modules` to the model's actual names — after which `print_trainable_parameters()` reads `trainable%: 0.062` and the loss finally moves. The lesson generalizes: **before debugging the LR of a PEFT run, confirm the adapter is actually in the graph** with `print_trainable_parameters()` and the no-grad check. A flat loss at *any* LR is a graph bug, not a hyperparameter. The full catalog of these silent no-ops — wrong `target_modules`, gradient-checkpointing interactions, dtype mismatches, merge/unmerge bugs — is in [debugging LoRA and PEFT](/blog/machine-learning/debugging-training/debugging-lora-and-peft).

One more structural point about *why LoRA is the default for forgetting-sensitive finetunes.* LoRA learns an update $\Delta W = BA$ where $A$ and $B$ are low-rank ($r \ll d$), added to a *frozen* base weight $W_0$, so the forward pass is $h = (W_0 + BA)x$. Because $W_0$ never receives a gradient, the pretrained features are *exactly* preserved in the base path; the adapter can only *add* a small correction. This is the basin picture made structural: you cannot leave the basin because the basin's coordinates ($W_0$) are frozen, and the most you can do is nudge with a rank-$r$ perturbation. The rank $r$ controls the budget — too small (r=2) underfits a far domain, too large (r=256) starts to behave like a full finetune and reintroduces forgetting risk. r=8–32 is the usual sweet spot, and `lora_alpha` scales the adapter's contribution (the effective scale is $\alpha/r$), so if you change $r$ you usually scale $\alpha$ with it to keep the contribution constant.

## 5. Pitfall four: discriminative / layer-wise learning rate

Here is a refinement of pitfall one that matters for full finetuning. A single global LR is a blunt instrument, because **different layers want different step sizes.** Lower layers of a pretrained model hold the most general, most transferable features (Gabor-like edge filters in vision, low-level acoustic features in speech, basic syntax in language) and you should barely touch them. Upper layers hold task-specific features and can move more. And the brand-new randomly-initialized head needs a *large* LR to catch up fast.

![Discriminative layer-wise learning rate assigns the smallest rate to the most general lower layers and the largest to the new head](/imgs/blogs/finetuning-pitfalls-across-modalities-6.png)

The rationale follows from the basin picture: the basin is *sharpest* (most sensitive to perturbation) in the lower, more-general layers, because those features are reused by everything above them — perturb an edge detector and every layer above it gets the wrong input. So step smaller there. The head, by contrast, is random — it is *not* in the basin at all — so it needs a large step to reach a good solution before it backpropagates damaging gradients into the body.

The standard implementation is **layer-wise LR decay**: assign the top layer a base LR and multiply by a decay factor $\gamma$ (typically 0.6–0.9) for each layer going down. Layer $\ell$ from the top gets $\eta_\ell = \eta_{\text{base}}\cdot \gamma^{\ell}$. This is the recipe ULMFiT introduced for NLP ("discriminative fine-tuning") and that ELECTRA/BERT finetuning adopted; the same idea drives `timm`'s `layer_decay` for vision transformers. It is genuinely cross-modal.

```python
def layerwise_lr_param_groups(model, base_lr=2e-5, decay=0.8, head_lr=1e-3):
    """Assign smaller LR to lower layers, large LR to the new head.
    Assumes model.encoder.layers is an ordered list, low -> high."""
    layers = list(model.encoder.layers)
    n = len(layers)
    groups = []
    # embeddings: the lowest, most general -> the smallest LR (or freeze).
    groups.append({"params": model.embeddings.parameters(),
                   "lr": base_lr * (decay ** (n + 1))})
    for i, layer in enumerate(layers):
        depth_from_top = n - 1 - i           # top layer -> 0
        groups.append({"params": layer.parameters(),
                       "lr": base_lr * (decay ** depth_from_top)})
    # the new randomly-initialized head: the LARGEST LR.
    groups.append({"params": model.head.parameters(), "lr": head_lr})
    return groups

opt = torch.optim.AdamW(layerwise_lr_param_groups(model), weight_decay=0.01)
for g in opt.param_groups:                    # sanity-print the LR ladder
    print(f"{len(list(g['params']))} tensors @ lr {g['lr']:.2e}")
```

With `base_lr=2e-5`, `decay=0.8`, and a 12-layer body, the head trains at $10^{-3}$, the top body layer at $2\times10^{-5}$, the bottom body layer at $2\times10^{-5}\cdot0.8^{11}\approx1.7\times10^{-6}$, and the embeddings at about $1.4\times10^{-6}$. That is a 700x ratio from head to embeddings — exactly matching the intuition that the head needs to move a lot and the embeddings almost not at all. In my experience this single change adds 1–3 points over a global LR on most full finetunes, and it sharply reduces forgetting because the general lower layers barely move.

There is real evidence for the "lower layers are general" premise, not just intuition. The classic transfer-learning study (Yosinski et al., 2014, "How transferable are features in deep neural networks?") showed that lower convolutional layers transfer across tasks with almost no loss, while upper layers are increasingly task-specific — feature generality decreases monotonically with depth. The same has been shown for transformers: probing studies find that lower transformer layers encode general, syntactic, low-level features and upper layers encode task-specific, semantic ones. So the layer-wise LR schedule is not a hack; it is matching the *step size to the generality of the feature* — touch the universally-useful low-level features as little as possible, adapt the task-specific upper features more freely. The decay factor $\gamma$ is your dial on this: $\gamma$ near 1.0 is almost a uniform LR (use it when your domain is far and even the low layers must adapt), $\gamma$ near 0.6 is aggressive protection of the lower layers (use it when your domain is near pretraining and you mainly need the head). A reasonable default is 0.75–0.85.

A common mistake to flag: people apply layer-wise decay but forget the embeddings, leaving them at the base LR. The embeddings (token embeddings for an LLM, patch+position embeddings for a ViT, the convolutional feature extractor for a speech model) are the *most* general part of the model and the most damaging to perturb — a shifted token embedding changes the meaning of a token everywhere it appears. Give the embeddings the *smallest* LR or freeze them outright. The `layerwise_lr_param_groups` helper above does this by putting the embeddings one decay step below even the bottom transformer layer.

## 6. Pitfall five: preprocessing and format must match the pretrained model

This is the most underestimated pitfall and the one with the most modality-specific disguises, even though it is one rule: **feed the pretrained model the input distribution it was trained on.** A pretrained model is a function tuned to a specific input statistics — a specific normalization, a specific tokenization, a specific sample rate, a specific encoding. Hand it inputs from a different distribution and you have moved *outside the basin's domain* before training even starts. Step zero is already broken, and no amount of finetuning fully recovers it because every gradient is computed on mis-scaled inputs.

This pitfall is *more* dangerous in finetuning than in from-scratch training, and the asymmetry is worth understanding. When you train from scratch, the model adapts to *whatever* input convention you give it — if you feed it BGR instead of RGB, the first conv layer just learns BGR filters; no harm done, because there is no pretrained expectation to violate. But a finetune inherits a model whose every weight was tuned for one specific convention. Violate it and you are asking a model optimized for RGB to interpret BGR — the channels are swapped relative to everything it learned. The model can *partially* re-adapt during finetuning (so the loss does go down, which is why this bug fools people), but it is fighting its own pretrained weights the whole way, and it never fully recovers the baseline. The tell is always the same: a finetune that lands far below its own linear-probe baseline despite a smoothly descending loss. When you see that, check the input convention *before* you touch the LR.

![Four modality-specific preprocessing mismatches, each a violation of the single rule that finetune inputs must match the pretrained input distribution](/imgs/blogs/finetuning-pitfalls-across-modalities-8.png)

The rule wears four costumes:

- **Vision — normalization and channel order.** The classic: a `timm`/`torchvision` model pretrained with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`, but your pipeline feeds raw `[0,1]` images (or worse, `[0,255]`, or BGR from OpenCV instead of RGB). The model's first conv expects normalized RGB; raw input is a different distribution. Symptom: accuracy stuck far below the linear-probe baseline, and the per-channel input histogram is centered at 0.5 instead of near 0. See [CV input pipeline bugs](/blog/machine-learning/debugging-training/cv-input-pipeline-bugs) for the full catalog (BGR/RGB, resize/interpolation, uint8 vs float). The one-line fix is to read the model's own expected transform.

```python
import timm
from timm.data import resolve_data_config, create_transform

model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=12)
# DO NOT hand-roll normalization. Ask the model what it was trained on.
cfg = resolve_data_config({}, model=model)
print(cfg)   # {'mean': (0.485,0.456,0.406), 'std': (0.229,0.224,0.225), ...}
transform = create_transform(**cfg, is_training=True)
# now your DataLoader uses exactly the pretrained input distribution.
```

- **LLM — tokenizer and chat template.** Two sub-bugs. First, the tokenizer must be the *checkpoint's own* tokenizer (a vocab mismatch silently maps tokens to the wrong ids). Second, and more insidious, the **chat template**: an instruction-tuned model was pretrained on a specific format (`<|im_start|>user ... <|im_end|>` or the Llama `[INST] ... [/INST]` form). If you finetune on raw `"question answer"` strings without the template, you are training a *different input distribution* than the one the model serves on, and at inference the templated input is out-of-distribution. Symptom: the model trains fine but ignores instructions or never emits the stop token at serving. The fix is to apply the tokenizer's own template in both training and inference.

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
msgs = [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
# Use the model's OWN template; do not hand-format strings.
text = tok.apply_chat_template(msgs, tokenize=False)
# At inference, add the generation prompt so the model knows it's its turn:
prompt = tok.apply_chat_template(
    [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
)
```

For the full set of these — left vs right padding, the double-BOS, loss masking, the "won't stop" bug — see [chat template and formatting bugs](/blog/machine-learning/debugging-training/chat-template-and-formatting-bugs) and [finetuning an LLM without breaking it](/blog/machine-learning/debugging-training/finetuning-an-llm-without-breaking-it).

- **Speech — feature extractor and sample rate.** Whisper expects 16 kHz log-mel spectrograms with specific `n_fft`/`hop_length`/`n_mels`. Feed it 44.1 kHz audio without resampling, or compute mels with different parameters, and every frame is mis-scaled in time and frequency. Symptom: WER explodes (often 5–10x worse) even though training loss looks reasonable. The fix is to use the model's bundled feature extractor, which encodes the exact pretraining parameters.

```python
from transformers import WhisperProcessor
import torchaudio

proc = WhisperProcessor.from_pretrained("openai/whisper-small")
wav, sr = torchaudio.load("clip.wav")
if sr != proc.feature_extractor.sampling_rate:            # 16000 for Whisper
    wav = torchaudio.functional.resample(wav, sr, proc.feature_extractor.sampling_rate)
feats = proc.feature_extractor(
    wav.squeeze().numpy(),
    sampling_rate=proc.feature_extractor.sampling_rate,
    return_tensors="pt",
).input_features      # exactly the mel parameters Whisper was pretrained with
```

This is the heart of [debugging ASR finetuning](/blog/machine-learning/debugging-training/debugging-asr-finetuning): the feature-extractor/sample-rate match is the speech version of vision normalization.

- **Tabular — fit encoders on train only.** When you finetune a model (or fit a downstream learner) on top of a learned feature encoder, the encoders/scalers/imputers must be **fit on training data only** and *applied* to validation and test. Fit a scaler or a target-mean encoder on the whole dataset and you have leaked test statistics into training — the tabular version of "the input distribution at serving differs from training." Symptom: validation/CV AUC is great and production AUC drops ~0.2. This is [tabular data leakage](/blog/machine-learning/debugging-training/tabular-data-leakage) and it is the same rule: the model must see, at serving, the exact transformation it saw in training, computed without peeking at the future.

The cross-modal preflight check for all four is the same: **print the actual statistics of one finetune batch and compare them to the pretrained model's expectation.** For vision, the per-channel mean/std should match the model's normalization. For LLMs, decode one tokenized example back to text and eyeball that the special tokens and template are present. For speech, the spectrogram shape and value range should match the feature extractor's. If the input distribution is wrong, *stop* — no hyperparameter will fix an out-of-distribution input.

## 7. Pitfall six: too few epochs underfit, too many overfit fast

From-scratch training often runs for many epochs over a large dataset. Finetuning is the opposite: the model already knows most of what it needs, and the finetune set is usually *small*, so the useful window is narrow. **Too few steps and the head never converges (underfit); too many and the model memorizes the small finetune set and forgets the prior (overfit + forgetting).** The right answer for most finetunes is 1–3 epochs, and on small data you can overfit within a *single* epoch.

![Timeline showing validation accuracy peaking by epoch two while the general probe falls steadily, the signature of over-finetuning on small data](/imgs/blogs/finetuning-pitfalls-across-modalities-7.png)

The mechanism is the same distance budget from section 1. On a small dataset, each example is revisited many times per epoch, so the gradient repeatedly points the same direction and the cumulative step $\sum_t \eta\lVert g_t\rVert$ grows fast — you leave the basin in a few epochs. And there is a statistical reason it overfits so quickly: with only a few thousand examples, the model has more than enough capacity to *memorize* them, driving train loss to near zero while learning nothing generalizable. The signature in figure 7 is unmistakable: task validation peaks around epoch 2 and then *plateaus or declines slightly*, while the general probe falls steadily the whole time. Train loss crashing toward zero is not success — past the second epoch it is the memorization-and-forgetting signature.

The diagnostic is to score **both** task validation and the forgetting probe at every epoch and stop at the epoch that maximizes task validation *subject to* the probe staying within gate. In code, that is an early-stopping rule with two conditions:

```python
best_task, best_state, patience, bad = -1, None, 2, 0
for epoch in range(max_epochs):
    train_one_epoch(model, train_loader, opt)
    task = evaluate_task(model, val_loader)        # higher is better
    prior = probe.check(model, step=f"epoch{epoch}")  # returns within-gate bool
    print(f"epoch {epoch}: task={task:.3f}  prior_ok={prior}")
    if task > best_task and prior:                 # only accept if prior held
        best_task, best_state, bad = task, copy.deepcopy(model.state_dict()), 0
    else:
        bad += 1
        if bad >= patience or not prior:
            print("stopping: task plateaued or prior breached")
            break
model.load_state_dict(best_state)                  # roll back to the best checkpoint
```

The key difference from a naive early stop is the `and prior` condition: a checkpoint is only "best" if it both improved the task *and* kept the prior. Without that condition you will happily save the over-trained epoch-5 checkpoint that has the highest task number and the worst forgetting.

#### Worked example: a Whisper finetune that got worse with more training

A speech team finetunes `whisper-small` on 12 hours of call-center audio. LR $10^{-5}$ (correct), feature extractor matched (correct), but they run **10 epochs** to "make sure it converges." In-domain WER: epoch 2 → 6.1%, epoch 5 → 5.4%, epoch 10 → 5.6%. So far it looks like 5 epochs was the sweet spot and 10 was harmless. But the *cross-domain* WER (a held-out set of read speech, the forgetting probe for ASR) tells the real story: baseline 9.0%, epoch 2 → 9.4%, epoch 5 → 12.1%, epoch 10 → **23.7%**. The model became a call-center specialist that can barely transcribe anything else — a textbook over-finetune. In-domain WER plateaued by epoch 5 while the model spent the next five epochs forgetting general speech. The fix: stop at epoch 3 (in-domain WER 5.7%, cross-domain WER 9.8%, both within tolerance), and add mild SpecAugment to slow memorization. This is the speech costume of the LLM ticket example and the vision over-finetune — *same timeline shape, different metric.* See [debugging ASR finetuning](/blog/machine-learning/debugging-training/debugging-asr-finetuning) for the WER-normalization traps that can also make this metric lie.

A note on the **freeze-then-unfreeze schedule** as an epoch-management tool. Instead of training everything for N epochs, you train the head for the first epoch (body frozen), then unfreeze and train everything at a small LR for one or two more. The first phase converges the random head *without* risking the body (the body has no gradients, so it cannot leave the basin), and the second phase makes a small, safe adaptation with a head that already produces reasonable gradients. This is the gradual-unfreezing idea from ULMFiT and it composes naturally with layer-wise LR — unfreeze top-down so the most task-specific layers adapt first. Here is a schedule that works across modalities:

```python
def progressive_unfreeze_schedule(model, total_epochs=3):
    """Train head first, then unfreeze top-down over the remaining epochs."""
    blocks = list(model.encoder.layers)          # low -> high
    for p in model.parameters():                  # start: freeze everything
        p.requires_grad = False
    for p in model.head.parameters():             # except the new head
        p.requires_grad = True
    yield rebuild_optimizer(model)                # epoch 0: head only

    n_unfreeze_per_epoch = max(1, len(blocks) // (total_epochs - 1))
    unfrozen = 0
    for epoch in range(1, total_epochs):
        for _ in range(n_unfreeze_per_epoch):     # unfreeze from the TOP down
            if unfrozen < len(blocks):
                for p in blocks[-(unfrozen + 1)].parameters():
                    p.requires_grad = True
                unfrozen += 1
        yield rebuild_optimizer(model)            # REBUILD opt each phase (sec. 4!)
```

The critical detail is `rebuild_optimizer(model)` each phase — the optimizer-before-unfreeze bug from section 4 will silently swallow every newly-unfrozen layer if you reuse a stale optimizer. The generator yields a fresh optimizer at each phase so the newly-trainable params actually enter the param groups.

## 8. Pitfall seven: the new head, its init, and the pooling choice

When you finetune for a new task you almost always attach a new output head — a new linear classifier for vision, a new sequence-classification head or LM head for an LLM, a new CTC/output projection for speech. Two bugs hide here.

**First, the head initialization.** The new head is random, so at step zero it produces high-loss, high-gradient outputs. If you let those large gradients backpropagate into the pretrained body at a normal LR, they *kick the body out of the basin in the first steps* — this is a major contributor to the LR pitfall. Two defenses: (a) initialize the head sensibly (small-variance init; for a classifier, zero-init the final bias so the initial prediction is uniform), and (b) use the freeze-then-unfreeze or layer-wise-LR pattern so the body sees small gradients while the head is still wild. The "warm up the head first" trick — train only the head for a few hundred steps, *then* unfreeze the body — directly addresses this and is cheap insurance.

**Second, the pooling choice — which must match how the backbone was pretrained.** A ViT pretrained with a CLS token expects you to read the CLS token for classification; pool the patch tokens with global average pooling instead and you are reading a representation the model never optimized as a summary. An LLM's sequence-classification head usually reads the *last* non-pad token's hidden state (for decoder-only models) — read the first token, or a pad token, and you read noise. A wav2vec2 model pools frame representations a specific way. The bug is silent: the model trains (the head adapts to whatever you give it) but underperforms, because you are summarizing the sequence in a way the pretraining never blessed.

```python
# LLM sequence classification: read the LAST real token, not a pad token.
# (Decoder-only models are causal; only the last position has seen the full input.)
def pooled_repr(hidden_states, attention_mask):
    # hidden_states: (B, T, H); attention_mask: (B, T) with 1 for real tokens
    seq_lens = attention_mask.sum(dim=1) - 1            # index of last real token
    batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return hidden_states[batch_idx, seq_lens]           # (B, H)
# A common bug: hidden_states[:, 0] (first token) or hidden_states[:, -1]
# (a PAD token under right-padding) -> the head trains on the wrong vector.
```

The cross-modal rule: **read the representation the backbone was pretrained to produce.** CLS for a CLS-token ViT, mean-pool for a mean-pool ViT, the last real token for a decoder-only LLM, the model's documented pooling for speech. Check the model card or the pretraining code; do not guess.

### The domain gap: how far you can ask a finetune to travel

Underlying the freeze/full/LoRA decision is one quantity that deserves its own paragraph: the **domain gap** between pretraining and your task. A near domain (finetuning an ImageNet ViT on more natural photos, an English LLM on English support text, a Whisper model on clean English audio) needs almost no movement — the features already fit, and a linear probe or a light LoRA gets you there. A far domain (finetuning a natural-image ViT on grayscale medical scans, an English LLM on legal code in another language, an English ASR model on a low-resource dialect) needs the features themselves to change, which means more plasticity: a full finetune, more epochs, a higher decay $\gamma$ so the lower layers can move too. The mistake in both directions is real: people full-finetune a near-domain task and get needless forgetting, or they linear-probe a far-domain task and underfit because the frozen features simply cannot represent the new domain.

The diagnostic for "how far is my domain?" is, again, the linear-probe baseline from the preflight. If a frozen-body linear probe already gets you most of the way to your target, the domain is near and you should *not* full-finetune. If the linear probe is far below target, the features need to adapt — the domain is far — and you should escalate to LoRA or a full finetune with a higher decay. The linear-probe number is doing double duty: it tells you the floor *and* it measures the domain gap. This is the cross-modal generalization of "look at your data before you train": measure the gap before you choose the tool.

## 9. Pitfall eight: evaluation that catches regression, not just train loss

Every pitfall above shares one enabling condition: an evaluation that only watches training loss (or only the narrow task metric) cannot *see* the failure. Train loss going down is consistent with the LR being wrong (it recovers a degraded solution), with forgetting (the prior is invisible to it), with over-finetuning (it crashes to zero precisely *because* you are memorizing), and with a preprocessing mismatch (the model adapts to the wrong inputs). **Train loss is the least informative instrument in finetuning.** You need an evaluation built to catch regression.

Why is train loss so untrustworthy in finetuning specifically? Because the model starts from a *good* solution, the absolute loss values are already low and the dynamic range is small — a finetune might move loss from 1.4 to 0.4, where a from-scratch run moves it from 9 to 0.4. A small, noisy movement in a narrow range carries almost no information about whether the model improved or degraded on anything you care about. From-scratch loss at least tells you "the model is learning *something*"; finetuning loss can go down while the model gets worse at everything except your narrow task. The instrument that *does* carry information is a held-out metric you chose to track the goal — and for finetuning you need two of them, one for the task and one for the prior.

The cross-modal evaluation harness has three components, and you want all three:

1. **A task validation set** that matches your deployment distribution — held out, never touched during tuning, scored with the metric you actually care about (and beware the metric bugs: micro vs macro, accuracy under imbalance, WER normalization — see [your metric is lying](/blog/machine-learning/debugging-training/the-input-pipeline-is-lying-to-you) territory and the metric track).
2. **A general-capability probe** (section 3) — the frozen prior set, scored every checkpoint, with a gate.
3. **An overfit-one-batch sanity check** at the start — proof that your loss, data, and optimization can drive a single batch to ~0. If it cannot, *stop and fix that first*; you have a model/optimization/loss bug, not a hyperparameter to tune.

Here is the consolidated preflight every finetune in every modality should run before the real training starts. It is five checks and it takes minutes.

```python
def finetune_preflight(model, batch, train_step_fn, score_general, score_task):
    report = {}
    # (1) Input distribution matches pretraining? (modality-specific check_fn)
    report["input_stats"] = describe_batch(batch)   # mean/std, token decode, mel shape
    # (2) General probe baseline recorded BEFORE training.
    report["probe_baseline"] = score_general(model)
    # (3) Linear-probe baseline: how good is a frozen-body head? (the floor to beat)
    report["linear_probe"] = quick_linear_probe(model, batch)
    # (4) Overfit one batch at the chosen LR: can loss -> ~0 in <100 steps?
    state = copy.deepcopy(model.state_dict())
    losses = [train_step_fn(model, batch) for _ in range(100)]
    report["overfit_one_batch"] = (losses[0], min(losses))
    model.load_state_dict(state)                    # restore; this was a probe
    # (5) LR sanity: does a single step keep the probe within gate?
    train_step_fn(model, batch)
    report["probe_after_one_step"] = score_general(model)
    model.load_state_dict(state)
    return report
```

If `overfit_one_batch` does not approach zero, your LR/loss/model is broken — do not proceed. If `probe_after_one_step` already dropped several points, your LR is too high — lower it. If `linear_probe` is close to your full-finetune target, you may not need a full finetune at all. This preflight is the cross-modal embodiment of the series' two master tools — make-it-fail-small and read-the-instruments — and it pays for itself the first time it catches a wrecked run at step 1 instead of step 4,000.

To make the bisection mechanical, here is the symptom → suspect → confirming test → fix table for finetuning. It is the finetuning-specific slice of the master taxonomy, and it is modality-agnostic — read the symptom, run the test, apply the fix, regardless of whether you are nudging a ViT, an LLM, or a Whisper.

| Symptom | Most likely suspect | Confirming test | Fix |
|---|---|---|---|
| Loss spikes in first 10–100 steps, then a worse plateau | LR too high (basin ejection) | overfit-one-batch fails to reach ~0; probe drops after one step | drop LR 10–100x; add warmup; clip grad |
| Full finetune *worse* than a linear probe | LR too high damaged the backbone | linear-probe baseline beats full finetune | lower body LR; use layer-wise LR |
| Task metric up, production/general worse | catastrophic forgetting | general probe drops past gate | fewer epochs; replay 10–30%; LoRA; lower LR |
| Train loss → ~0, val plateaus, prior falls | over-finetuning (too many epochs) | timeline: probe declines as loss crashes | cap at 1–3 epochs; early-stop on task+prior |
| Loss flat at *any* LR | adapter/layer not in graph | `print_trainable_parameters` = 0%; no-grad check | fix `target_modules`; rebuild optimizer after unfreeze |
| Accuracy stuck far below baseline from step 0 | input preprocessing mismatch | per-batch stats differ from model's expectation | use model's own transform / extractor / template |
| Model won't stop generating / ignores instructions | chat-template / EOS skew | decode a training example; check template + EOS | apply tokenizer's chat template; unmask EOS |
| Great CV, poor production (tabular) | preprocessing leakage | refit encoders inside CV folds; AUC drops | fit scaler/encoder on train fold only |

This table is the working artifact for the next time a finetune breaks. Match the symptom, run the one-line test, apply the fix. The point of the whole post is that this table has *one* column for the test and fix, not four — the diagnosis does not branch by modality.

## 10. The unified finetuning checklist

Everything above collapses into one table. The columns are modalities; the rows are the checklist steps; each cell is the concrete form that step takes in that modality. This is the artifact to pin above your desk. Print it, run down it before every finetune, and you will catch most of these pitfalls before they cost you a run.

![Matrix mapping each unified finetuning checklist step to its concrete form in vision, LLM, and speech](/imgs/blogs/finetuning-pitfalls-across-modalities-3.png)

| Step | Vision (ViT / ResNet) | LLM (decoder-only) | Speech (Whisper / wav2vec2) | Tabular (encoder + learner) |
|---|---|---|---|---|
| 1. Match the input format | ImageNet mean/std, RGB not BGR, model's own transform | tokenizer + chat template, right padding, no double-BOS | 16 kHz, model's feature extractor, matching mel params | fit scaler/encoder on train only, apply to val/test |
| 2. Learning rate (10–100x smaller) | body 1e-5–5e-5, head 1e-3 | 1e-5–2e-5 full, 1e-4–2e-4 LoRA | 1e-5 with 500-step warmup | n/a for trees; for NN-on-encoder, 1e-4 |
| 3. Layer-wise / discriminative LR | `layer_decay` 0.7–0.9 | decay 0.8, embeddings tiny/frozen | freeze CNN feature extractor, train transformer | freeze encoder, tune learner |
| 4. What to freeze (probe/full/LoRA) | probe if <5k, LoRA mid, full if large+far | LoRA r=16 on q,v default; full if large | freeze feature extractor; LoRA on attention | freeze encoder; train downstream learner |
| 5. Epochs (1–3, overfits fast) | 5–15 epochs small data, watch val | 1–3 epochs, often 1 | 2–5 epochs, SpecAugment to slow overfit | early-stop on a clean val fold |
| 6. New head init + pooling | zero-init bias; CLS vs mean to match backbone | last real token; head small-init | output projection per model card | n/a (learner is the head) |
| 7. Detect forgetting (general probe) | 500 held-out ImageNet val images | 200-Q MMLU + arithmetic + instruction slice | WER on a different-domain held-out set | AUC on the broad distribution, not just target |
| 8. Eval catches regression | top-1 + per-class recall, not just loss | task eval + probe + format check | WER with correct normalization | PR-AUC + calibration on a true holdout |

Notice how column-by-column the *rules* are identical and only the *nouns* change. Match the input. Use a small LR, smaller for lower layers. Freeze what you can. Don't over-train. Init the head, pool correctly. Probe for forgetting. Evaluate for regression. Eight steps, four modalities, one discipline.

The reason this collapses so cleanly is the basin picture one last time: all four modalities use the *same recipe* — a model pretrained to the bottom of a sharp loss basin, then nudged toward a downstream task — so they share the *same failure surface*. The architecture differs (convolutions vs attention vs CTC vs trees), the input differs (pixels vs tokens vs spectrograms vs feature vectors), the metric differs (top-1 vs perplexity vs WER vs AUC), but the *dynamics of adapting a pretrained model* are governed by the same handful of quantities — the learning rate, the number of steps, the per-layer step sizes, and the match between the input distribution at finetune time and at pretraining time. Master those four quantities once and you can debug a finetune in a modality you have never worked in, because the checklist transfers even when the domain knowledge does not. That portability is the entire payoff of treating finetuning as one problem instead of four.

## 11. Case studies and real signatures

These are well-documented patterns and representative results. Where a number is from a specific paper or benchmark I cite it; where it is a representative order of magnitude I say so.

**ULMFiT and discriminative fine-tuning (Howard & Ruder, 2018).** ULMFiT introduced the now-standard finetuning recipe for NLP: discriminative (layer-wise) learning rates, slanted triangular warmup, and gradual unfreezing, and showed it reduced error by 18–24% over training from scratch on several text-classification benchmarks. The reason it worked is exactly section 5's argument — lower layers carry general features and must be perturbed less. The same `layer_decay` idea was later standard in BERT and ViT finetuning. This is the cross-modal pattern's origin story.

**Catastrophic forgetting (McCloskey & Cohen, 1989; French, 1999).** The phenomenon — a network trained on a new task abruptly loses performance on an old one — was named and studied long before deep learning's current era, and continual-learning research (EWC, Kirkpatrick et al., 2017) formalized the fix as tethering weights to the prior solution (the Fisher-weighted L2 of section 3). The modern instruction-tuning version is the same effect: narrow SFT degrades general benchmarks unless you mix replay data or constrain the update. The mechanism is identical across vision, language, and speech; only the benchmark changes.

**The chat-template / formatting skew bug (well-known LLM finetuning failure).** A very common report: a finetuned chat model "won't stop generating" or "ignores instructions." The usual root cause is train-vs-inference template skew — the model was finetuned on raw strings (or a different template) than the one used at serving, so serving inputs are out of distribution, and/or the EOS token was never seen in the right place during training so the model never learned to stop. The fix is to apply the tokenizer's own chat template in both training and inference and ensure EOS is present and unmasked. This is the LLM costume of the section-5 preprocessing-mismatch rule.

**Tabular preprocessing leakage (a recurring Kaggle and production post-mortem).** A frequently-reported pattern: a model scores ~0.96 AUC in cross-validation and ~0.74 in production. The post-mortem almost always finds a scaler, imputer, or target encoder fit on the full dataset (or inside the wrong CV scope), leaking test statistics. The honest AUC after fixing the leak is the production number, and the "drop" of ~0.2 is the size of the leak. The same rule — the model must see at serving the exact transformation computed without peeking — is the tabular costume of preprocessing-match, and the [tabular data leakage](/blog/machine-learning/debugging-training/tabular-data-leakage) post is the deep dive.

**ASR feature-extractor mismatch (representative).** A common Whisper/wav2vec2 finetuning failure: WER that is 5–10x worse than the pretrained baseline despite a smoothly descending loss, traced to a sample-rate or mel-parameter mismatch (44.1 kHz audio fed without resampling, or mels computed with a different `n_fft`). Using the model's bundled feature extractor restores the baseline. The order of magnitude (an 8x WER blowup) is representative of these reports, not a single paper's figure.

**Linear-probe vs full-finetune (Kumar et al., 2022, "Fine-Tuning can Distort Pretrained Features").** This study made a subtle cross-modal point precise: full finetuning can *underperform* linear probing on out-of-distribution data, because the early large gradients from the random head distort the pretrained features before the head has stabilized — exactly the section-8 head-init argument. Their fix, "LP-FT" (linear-probe first, then finetune), is the freeze-then-unfreeze schedule: warm up the head while the body is frozen, *then* finetune. This is direct published evidence for two of our pitfalls at once (head init and freeze schedule), and it is why "a full finetune that's worse than a linear probe" is a named, expected signature rather than a fluke.

**The transfer-learning generality gradient (Yosinski et al., 2014).** Beyond motivating layer-wise LR (section 5), this work documented a second, subtler effect: when you copy and freeze middle layers and finetune the rest, performance can *drop* due to "fragile co-adaptation" — the frozen and trainable layers were co-tuned during pretraining and splitting them mid-network breaks the interaction. The practical lesson for finetuning is to freeze at *natural boundaries* (whole blocks, the embedding layer, the feature extractor) rather than splitting inside a tightly-coupled module, and it is why the freeze schedules in this post operate at the block level, never mid-block.

## 12. When this is (and isn't) your finetuning bug

Bisection means knowing when a symptom points *away* from finetuning recipe and toward something else. A few decisive calls:

- **If overfit-one-batch fails, stop blaming the LR.** A model that cannot drive a single batch to ~0 loss has a model-code, loss-function, or data-pipeline bug, not a hyperparameter problem. Fix the floor before tuning anything. (See [the overfit a single batch test](/blog/machine-learning/debugging-training/the-overfit-a-single-batch-test).)
- **A smooth-then-NaN curve is numerics, not recipe.** If loss is healthy for thousands of steps and then spikes to NaN, suspect fp16 underflow/overflow or a rare bad batch, not your LR or epochs. That is [hunting NaNs and infs](/blog/machine-learning/debugging-training/hunting-nans-and-infs) and [mixed precision debugging](/blog/machine-learning/debugging-training/mixed-precision-debugging-fp16-vs-bf16) territory.
- **If train and eval both look great but production is bad, suspect data, not training.** A clean finetune that fails in production is usually distribution shift or an eval set that doesn't match the real world — not a recipe bug. The forgetting probe and a production-matched holdout disambiguate.
- **If the LoRA loss never moves, it is a graph bug, not an LR bug.** A flat LoRA loss almost always means the adapter is not in the computation graph (wrong `target_modules`, gradient checkpointing interaction, a dtype mismatch) — confirm with the trainable-but-no-grad check from section 4, not by raising the LR. (See [debugging LoRA and PEFT](/blog/machine-learning/debugging-training/debugging-lora-and-peft).)
- **If a linear probe already matches your target, full finetuning is the wrong tool.** Don't pay the forgetting risk and compute for a full finetune when a frozen-body head gets you there. The linear-probe baseline is part of the preflight precisely to catch this.

The discipline is the same as the whole series: name the symptom, pick the suspect among the six places, run the cheap confirming test, *then* fix. Finetuning just narrows the suspect list, because a pretrained model already works — so the bug is overwhelmingly in how you touched it.

One more distinction that saves hours: **finetuning recipe bugs vs systems bugs.** Everything in this post is a recipe bug — a knob you set wrong (LR, epochs, freeze, preprocessing). But a finetune can also fail for *systems* reasons that look superficially similar: a multi-GPU run where gradients don't sync (loss looks fine on rank 0 but the model doesn't actually learn from all data), an out-of-memory error masked as a mysterious crash, a checkpoint-resume that jumps the loss because the optimizer state wasn't restored. The way to tell recipe from systems: a recipe bug reproduces on a *single* GPU with a *small* run, and a systems bug usually does not. So when you suspect a finetune is broken, first reproduce it on one GPU with a few hundred examples. If it still breaks, it is a recipe bug and this post applies. If it only breaks at scale or on resume, you are in systems territory — a different track of the series. The make-it-fail-small principle does double duty here: shrinking the run both speeds your debug loop *and* tells you which class of bug you have.

## 13. Key takeaways

- **Finetuning is "nudge a good model, don't retrain it."** Every pitfall in this post is a violation of that one line. Hold the basin picture — pretrained weights at the bottom of a sharp valley — and the fixes follow.
- **Use an LR 10–100x smaller than from-scratch.** A from-scratch LR ejects the model from the basin in the first ten steps. Symptom: an early loss spike that "recovers" to a worse plateau. The fix is the same in vision, LLMs, and speech.
- **Catastrophic forgetting is invisible in train loss — measure it with a frozen general probe.** Score the prior at every checkpoint, gate it (e.g. −4 points), and early-stop when it trips. This single habit is the highest-leverage cross-modal defense.
- **Match the input distribution to pretraining.** Normalization (vision), tokenizer + chat template (LLM), feature extractor + sample rate (speech), train-only encoders (tabular) — one rule, four costumes. A wrong input breaks step zero; no hyperparameter recovers it.
- **Choose freeze vs full vs LoRA by data size and domain distance.** Small + near → linear probe; mid or far → LoRA; large + far → full finetune with layer-wise LR. And beware the optimizer-before-unfreeze no-op.
- **Lower layers want a smaller LR; the new head wants a large one.** Layer-wise LR decay (γ ≈ 0.8) protects the general lower layers and is worth 1–3 points on most full finetunes.
- **1–3 epochs, not 30.** Small finetune sets overfit fast; train loss crashing to zero past epoch two is the memorization-and-forgetting signature, not success.
- **Run the five-check preflight before the real run:** input-stats match, probe baseline, linear-probe floor, overfit-one-batch, probe-after-one-step. It catches wrecked runs at step 1, not step 4,000.
- **Train loss is the least informative finetuning instrument.** Evaluate for *regression*: task validation + general probe + format/preprocessing check. If overfit-one-batch passes and the probe holds, you have a healthy finetune.

## 14. Further reading

- Howard, J. & Ruder, S. (2018). *Universal Language Model Fine-tuning for Text Classification (ULMFiT)* — the origin of discriminative (layer-wise) fine-tuning, slanted triangular warmup, and gradual unfreezing.
- Kirkpatrick, J. et al. (2017). *Overcoming catastrophic forgetting in neural networks (EWC)* — the Fisher-weighted regularization toward the prior; the formal version of "tether to the basin."
- McCloskey, M. & Cohen, N. (1989) and French, R. (1999) — the original characterizations of catastrophic forgetting and the stability–plasticity tradeoff.
- Hu, E. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models* — the parameter-efficient finetuning method that structurally bounds forgetting by freezing the base weights.
- Hugging Face `transformers`, `peft`, and `Trainer` documentation — chat templates (`apply_chat_template`), `LoraConfig`/`target_modules`, and the feature extractors / processors for vision and speech.
- `timm` documentation — `resolve_data_config`, `create_transform`, and `layer_decay` for vision-transformer finetuning.
- Within this series: [a taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) (the master decision tree), [the training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) (the capstone), [the learning rate is almost always the problem](/blog/machine-learning/debugging-training/the-learning-rate-is-almost-always-the-problem), [finetuning an LLM without breaking it](/blog/machine-learning/debugging-training/finetuning-an-llm-without-breaking-it), [debugging vision transformer finetuning](/blog/machine-learning/debugging-training/debugging-vision-transformer-finetuning), [debugging ASR finetuning](/blog/machine-learning/debugging-training/debugging-asr-finetuning), and [debugging LoRA and PEFT](/blog/machine-learning/debugging-training/debugging-lora-and-peft).

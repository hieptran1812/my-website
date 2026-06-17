---
title: "Debugging Vision Transformer Finetuning: Patches, Position Embeds, and the LR That Destroys Features"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Diagnose the failure set unique to finetuning a pretrained ViT, CLIP, or DINO backbone — the learning rate that wipes the representation in the first steps, the position-embed mismatch on a resolution change, and the preprocessing skew — and fix a finetune that collapsed to chance so it beats a linear probe."
tags:
  [
    "debugging",
    "model-training",
    "vision-transformer",
    "computer-vision",
    "finetuning",
    "deep-learning",
    "pytorch",
    "transfer-learning",
    "timm",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/debugging-vision-transformer-finetuning-1.png"
---

The run looked like it was working for about thirty seconds. I had a `vit_base_patch16_224` from `timm`, pretrained on ImageNet-21k, and I wanted to finetune it on a flowers dataset with 102 classes. I copied my from-scratch ResNet recipe almost verbatim: SGD, learning rate `5e-3`, cosine schedule, the works. The first few loss prints were encouraging — `4.62`, `4.51`, `4.58` — wobbling around but in the right ballpark for 102 classes (random guessing is `ln(102) ≈ 4.62`). By step 60 the loss had climbed back up and pinned itself at `4.62` and would not move. Validation accuracy: `1.0%`. For 102 balanced classes, chance is `0.98%`. My finetune of a model that already knew how to see had learned to guess.

The instinct is to blame the data — wrong labels, a broken `DataLoader`, classes that don't map. I spent an hour there and found nothing. The instinct after that is to blame the model code — a frozen layer, a detached graph, the head wired to the wrong feature. Also clean. The actual culprit was a single number I had carried over from a context where it was correct and into one where it was catastrophic: the learning rate. `5e-3` is a perfectly reasonable rate to train a ResNet from random initialization. Applied to a pretrained ViT, it is an explosive charge. In the first dozen optimizer steps it walked the weights so far out of the basin that the pretrained features lived in that the representation was gone — overwritten, not adapted. The model I was "finetuning" had, by step 60, less useful structure in its features than a frozen backbone with a logistic regression bolted on top. I had a state-of-the-art encoder and I had detonated it.

This post is about the failure set that is specific to finetuning a pretrained vision transformer — a ViT, a CLIP image tower, a DINO or DINOv2 backbone — as opposed to training one from scratch or finetuning a ResNet. Pretrained ViTs fail differently, and they fail *quietly*. The loss does not NaN. There is no shape error. The run completes. You just get a model that is worse than the thirty-line linear-probe baseline you could have written in five minutes. Figure 1 is the map: six handoffs between your input pipeline and your loss, each one a distinct bug class. We are going to walk down that stack — preprocessing skew, patchify and patch-size locks, the position-embedding interpolation you forgot on a resolution change, the backbone learning rate and layer-wise decay, the freeze schedule and the optimizer-before-unfreeze trap, and the CLS-versus-mean-pool head — and for each one I will give you the mechanism (why it happens), the diagnostic (runnable code that confirms it), and the before-after evidence (what the instruments read once it is fixed).

![Layered stack diagram of the six handoffs where a pretrained ViT finetune can break, from input preprocessing through patchify, position embeds, backbone learning rate, freeze schedule, and pooling head](/imgs/blogs/debugging-vision-transformer-finetuning-1.png)

By the end you will have a fixed reflex: when a ViT finetune collapses to chance, you do not start editing code. You run a frozen **linear probe** first to confirm the features are actually good, then you monitor **feature drift** — the cosine similarity of your live features to the frozen backbone over steps — to see whether your optimizer is adapting the representation or destroying it. That single instrument, plotted against step count, distinguishes almost every failure in this post from its healthy twin. It ties straight back to the series spine: a bug hides in one of six places — data, optimization, model code, numerics, systems, or evaluation — and you **bisect** to the right one before touching anything. For ViT finetuning, the linear probe *is* the bisection: it cleanly separates "the features are bad" (data or model) from "the finetune destroyed good features" (optimization).

## 1. The symptom: a finetune that collapses to chance

Let us be precise about what we are debugging, because "the finetune didn't work" covers at least four genuinely different failures, and they have different fixes. Across this post the running example is the flowers-102 finetune above: a `vit_base_patch16_224` backbone, 86M parameters, pretrained on ImageNet-21k, with a fresh 102-way linear head. Here are the four signatures you actually see in the wild.

- **Collapse to chance.** Loss climbs back to `ln(num_classes)` and stays; val accuracy sits at `1/num_classes`. This is almost always optimization — a learning rate too high for a pretrained backbone, which we cover in Sections 3 and 4. It is the loudest failure and the easiest to misattribute to data.
- **Worse than linear probe.** The finetune *trains* — loss falls, val accuracy is non-trivial — but it lands below where a frozen linear probe lands. The features got worse, not better. This is a milder version of the same optimization problem, or a resolution/pos-embed bug (Section 5), or a preprocessing skew (Section 6).
- **Trains but generalizes poorly.** Train accuracy climbs fast, val lags badly, and the gap opens within an epoch. ViTs overfit small datasets *fast* because they have no convolutional inductive bias to lean on; this is the few-epochs, strong-aug, layer-freezing story of Section 9.
- **Head learns, backbone is dead.** Loss falls a little then plateaus well above zero, and a grad-norm probe shows the backbone parameters have effectively zero gradient. This is a freeze-schedule or optimizer-construction bug (Section 7): you froze the wrong thing, or you built the optimizer before you unfroze.

The reason all four are worth separating is that the **first thing you do is the same for all of them**, and it tells you which one you have. You run a linear probe. If the probe is good and the finetune is at chance, you have a destruction problem (optimization). If the probe is *also* at chance, your features never worked — your preprocessing or pos-embed is wrong, and no learning rate will save you. Section 2 makes that probe concrete; the rest of the post is the branches of that decision.

It helps to map these four signatures onto the series' six-places frame explicitly, because doing so tells you which instrument to reach for. Collapse-to-chance and worse-than-probe-with-bad-drift are **optimization** (the LR and the schedule); the instrument is the feature-drift monitor. Trains-but-generalizes-poorly is **data and regularization** (too little data for the model's capacity); the instrument is the train-val gap curve. Head-learns-backbone-dead is **model code** (a freeze or optimizer-construction bug); the instrument is the per-parameter weight-delta probe. And the quiet under-the-probe deficit that survives an LR fix is **the input pipeline** — preprocessing skew or a missing pos-embed interpolation; the instrument is a diff against the model card and a `pos_embed.shape` assert. Four signatures, four places, four instruments — and the linear probe is the switch that routes you to the right one. That is the entire debugging method for this chapter compressed into a paragraph; the rest of the post is each branch in full.

One more framing note before we go deep. From-scratch training and finetuning a pretrained backbone are not the same activity at different scales — they are different optimization problems. From scratch, you are searching a high-dimensional landscape from a random point, and a large learning rate is your friend: it lets you cover ground and escape bad regions. Finetuning, you start *inside* a good basin that took 300 GPU-days and 14 million images to find, and your job is to nudge — to slide along the basin floor toward your task without leaving it. A large step is no longer covering ground; it is jumping out of the one place worth being. Everything in this post follows from that single asymmetry.

It is worth naming the failure mode formally because it has a name in the literature and naming it sharpens the debugging instinct: this is **catastrophic forgetting**. The pretrained weights encode a representation that is, for almost any downstream vision task, far more useful than anything your few thousand finetuning images could teach from scratch. When the optimizer takes steps large enough to overwrite that representation faster than your task can rebuild an equivalent one, you forget more than you learn, and the net effect on a small dataset is destruction. The reason it is *catastrophic* rather than gradual is the compounding-through-depth effect we will derive in Section 3: a perturbation to an early layer does not stay local, it propagates and amplifies through every block above it, so the representation degrades super-linearly in step size. A CNN finetune forgets too, but more gracefully, because its convolutional structure constrains how far the features can move per step. A ViT has no such constraint, which is precisely why it is both a stronger backbone and a more dangerous one to finetune carelessly.

## 2. Always run the linear probe first

Before you debug a finetune, prove the features are worth finetuning. The fastest way is a **linear probe**: freeze the entire backbone, extract features for your training set once, and fit a single linear classifier (logistic regression, or a one-layer `nn.Linear` trained for a few epochs) on top. The probe accuracy is your floor and your sanity check. A good ImageNet-21k ViT probes to something like 85-95% on a clean flowers dataset; if your probe lands there, the features are fine and any collapse is your finetune's fault. If your probe is at chance, stop — you have a feature-extraction bug (preprocessing, pos-embed, or the wrong pooled token), and finetuning will only paper over it.

![Tree diagram showing the linear probe as a bisection step that separates bad features from a bad finetune, with branches for learning rate, pooling, preprocessing, and position embeds](/imgs/blogs/debugging-vision-transformer-finetuning-6.png)

Here is the probe in `timm` + `sklearn`, the version I run before every ViT finetune. It is the cheapest insurance in computer vision.

```python
import torch
import timm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

device = "cuda"
# num_classes=0 makes timm return pooled features, not logits.
backbone = timm.create_model(
    "vit_base_patch16_224.augreg_in21k", pretrained=True, num_classes=0
).eval().to(device)

# Use the model's OWN preprocessing — this is non-negotiable (Section 6).
cfg = timm.data.resolve_model_data_config(backbone)
transform = timm.data.create_transform(**cfg, is_training=False)

@torch.no_grad()
def extract(loader):
    feats, labels = [], []
    for x, y in loader:
        f = backbone(x.to(device))          # [B, 768] pooled CLS feature
        feats.append(f.cpu())
        labels.append(y)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()

Xtr, ytr = extract(train_loader)
Xva, yva = extract(val_loader)

clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
print("linear-probe val acc:", accuracy_score(yva, clf.predict(Xva)))
# Healthy ImageNet-21k ViT on flowers-102: ~0.88
```

If this prints `0.88`, you now know two things with certainty. First, the features are excellent — better than 88% of a chance baseline of under 1%. Second, **any finetune that scores below 0.88 has made the features worse**, which is a finding, not a mystery. The probe converts a vague "my finetune is bad" into a precise "my finetune destroyed a representation that was already at 88%." That is the whole debugging move: turn an open question into a measured gap.

A subtle but important point: use `resolve_model_data_config` to get the *model's* expected resize, crop, mean, and std, and use the exact pooled feature the model returns. If you build your own transform with ImageNet mean/std on a CLIP tower, or pool the wrong token, the probe will under-read and you will misdiagnose good features as bad. The probe is only a valid baseline if its preprocessing matches the model card — which is exactly the bug in Section 6, so the probe and the preprocessing check reinforce each other.

Why is the linear probe the *right* baseline rather than, say, "what a from-scratch model gets"? Because it isolates exactly one variable: the quality of the pretrained features, holding everything downstream linear and trivial. A logistic regression on frozen features has no capacity to invent structure — it can only read linear combinations of the features the backbone already produces. So its accuracy is a direct, almost calibration-grade measurement of how linearly-separable your task is in the backbone's representation space. That makes it the perfect reference point for a finetune, whose entire job is to make the task *more* separable by adapting the features. If finetuning beats the probe, the adaptation helped. If it ties the probe, the adaptation did nothing useful (and you wasted compute and risk on something a frozen probe would have given you). If it loses to the probe, the adaptation actively hurt — it made the features less separable, which is the destruction signature. There is no other single number that so cleanly tells you whether your finetune is adding value, and it costs one pass over your data to compute.

#### Worked example: the probe that saved a week

A teammate had a DINOv2 ViT-S/14 finetune stuck at 4% on a 50-class plant-disease dataset and was three days into rewriting the augmentation pipeline. I asked for the linear-probe number. He did not have one. We ran it: the frozen probe scored **81%**. That single number relocated the bug instantly. The features were not just fine, they were excellent — the finetune was destroying them. Five minutes later we found the learning rate: `1e-3`, copied from a CNN recipe. We dropped it to `1e-5`, added layer-wise decay, and the finetune landed at **86%**, beating the probe by 5 points. Three days of augmentation work, all aimed at the wrong of the six places. The probe would have pointed at optimization in the first ten minutes.

## 3. The science: why a pretrained representation is a sharp basin

Why does one number — the learning rate — separate "adapts to 92%" from "collapses to chance"? The answer is geometric, and once you see it you will never again copy a from-scratch LR into a finetune.

A pretrained backbone sits at a point $\theta^\*$ in weight space that is a deep, relatively narrow minimum of the *pretraining* loss. The features it produces — the 768-dimensional vectors your head consumes — are a smooth function of $\theta$ near $\theta^\*$. When you finetune, gradient descent takes steps $\Delta\theta = -\eta\, g$ where $\eta$ is the learning rate and $g$ is the gradient of your *new* task loss. The question is how far one step moves the *features*, not just the weights.

To first order, the change in a feature vector $\phi(x)$ for input $x$ after one step is

$$\Delta \phi(x) \approx J_\phi(x)\, \Delta\theta = -\eta\, J_\phi(x)\, g,$$

where $J_\phi(x) = \partial \phi / \partial \theta$ is the Jacobian of the features with respect to the weights. The magnitude of the feature perturbation scales **linearly in $\eta$** and in the size of the Jacobian. A deep transformer has a large Jacobian norm — the features are a long composition of attention and MLP blocks, and small weight changes compound through depth. So a step that would be a gentle nudge for a shallow network is a violent shove for a 12-block ViT. This is the mechanism: the same $\eta$ produces a much larger feature displacement in a deep pretrained network than your CNN intuition expects.

Now stack a second effect on top. At the very start of finetuning, your randomly initialized head produces *garbage* logits, so the cross-entropy gradient flowing back into the backbone is large and essentially random — it points in no useful direction because the head has not yet learned to read the features. With a large $\eta$, you take a big step in a random direction, *out of the basin*, before the head ever produces a meaningful training signal. By the time the head could have provided useful gradients, the features it would have read are already gone. This is why the collapse is so fast — it happens in the first dozen steps, during the window when the head is still random. It is also why **warmup matters even more for finetuning than for from-scratch training**: warmup keeps $\eta$ tiny during exactly the window when the gradient is least trustworthy.

![Before-and-after diagram contrasting a high learning rate that steps out of the pretrained basin and collapses to chance against a small learning rate that stays in the basin and adapts the features](/imgs/blogs/debugging-vision-transformer-finetuning-2.png)

Put numbers on it. Suppose your pretrained backbone has weight norm on the order of $\|\theta\| \approx 50$ (typical for a ViT-B after pretraining), the basin's useful radius is on the order of a few percent of that, and the first-step gradient norm after a random head is $\|g\| \approx 30$. With $\eta = 5\times10^{-3}$ you move $\eta\|g\| \approx 0.15$ per step in weight space, but the *feature* displacement, amplified by the Jacobian through twelve blocks, is many times larger — enough to leave the basin within ten to twenty steps. With $\eta = 2\times10^{-5}$ you move $6\times10^{-4}$ per step: 250 times smaller, comfortably inside the basin, and the head gets time to start producing real gradients before the backbone has drifted. Same model, same data — the only thing that changed is whether your first twenty steps stayed in the place that took 300 GPU-days to find.

This is also the rigorous justification for **layer-wise learning-rate decay**, which we instrument in Section 4. The features at the *output* of the backbone are what your task cares about, and the late blocks are closest to them — they should adapt most. The early blocks compute low-level structure (edges, textures, patch statistics) that transfers almost unchanged across tasks; perturbing them is mostly downside, because their Jacobian to the output is enormous (their changes propagate through every later block). So the optimal step size *grows with depth*: tiny for `block 0`, larger for the head. A single global LR cannot express that; layer-wise decay can.

Let us make the compounding-through-depth claim concrete rather than hand-waved, because it is the load-bearing piece of the whole argument. A transformer feature is a composition $\phi = f_L \circ f_{L-1} \circ \cdots \circ f_1$ of $L$ blocks. By the chain rule, the sensitivity of the output to a perturbation in block $\ell$'s weights is the product of the Jacobians of every block *above* $\ell$:

$$\frac{\partial \phi}{\partial \theta_\ell} = \left(\prod_{k=\ell+1}^{L} \frac{\partial f_k}{\partial h_{k-1}}\right) \frac{\partial f_\ell}{\partial \theta_\ell},$$

where $h_{k-1}$ is the activation entering block $k$. If each block's Jacobian has spectral norm modestly above 1 — which residual connections encourage, because a residual block computes $h + g(h)$ and its Jacobian is $I + \partial g/\partial h$, biased toward expansion — then the product over $L - \ell$ blocks grows roughly geometrically in depth-from-the-output. A perturbation to `block 0` is amplified by all twelve blocks above it; a perturbation to the head is amplified by none. That product is exactly the factor by which the *same* weight step produces a larger *feature* step in early layers, and it is why a uniform LR is the wrong shape. The geometric decay factor `d` in layer-wise decay (we use 0.8) is an explicit attempt to cancel that geometric amplification: scale each layer's step down by `d` per level of depth so that the *feature* displacement per step is roughly equalized across layers. That is not a heuristic pulled from a config file — it is the inverse of the amplification the chain rule predicts.

There is a second, quieter reason the early layers should barely move, and it is statistical rather than geometric. The early blocks of a ViT have effectively *seen* far more data than the late blocks have been specialized on: they encode general patch statistics learned from millions of images across thousands of classes. Your finetuning set has, say, a few thousand images. The information content of your gradient signal about what the early layers *should* be is tiny compared to what pretraining already encoded; updating them aggressively is overwriting high-confidence pretrained knowledge with low-confidence task-specific noise. The late blocks and the head are where your task actually has something to say. Layer-wise decay is, in this light, a crude Bayesian prior: trust the pretrained early layers strongly (small steps), trust the pretrained late layers less (larger steps), trust the random head not at all (full base LR). Every part of the recipe falls out of one idea — match the step size to how much you actually know.

## 4. Diagnostic: measure feature drift and set up layer-wise LR decay

The mechanism gives us a direct instrument. If a too-high LR collapses a finetune by dragging the features out of the basin, then **measuring how far the features have moved from the frozen backbone, per step, will catch the collapse as it happens** — long before the val accuracy print confirms it. The right metric is cosine similarity (or its complement, drift) between the current features and the frozen-backbone features, averaged over a fixed probe batch.

![Branching graph diagram showing feature cosine similarity to the frozen backbone splitting by learning rate, with the high-LR path collapsing and the low-LR path holding](/imgs/blogs/debugging-vision-transformer-finetuning-3.png)

Here is the feature-drift monitor. Keep a frozen copy of the backbone (or just cache its features on a fixed probe batch once), then every N steps compute the cosine similarity of the live features to the frozen reference. A healthy finetune drifts slowly — cosine stays above ~0.8 for many steps. A collapsing finetune craters toward zero within the first hundred steps.

```python
import copy
import torch
import torch.nn.functional as F

# Freeze a reference snapshot of the backbone BEFORE training.
frozen = copy.deepcopy(model.backbone).eval()
for p in frozen.parameters():
    p.requires_grad_(False)

probe_x, _ = next(iter(val_loader))          # one fixed batch, reused every check
probe_x = probe_x.to(device)

@torch.no_grad()
def feature_drift(live_backbone):
    live_backbone.eval()
    f_live = live_backbone(probe_x)          # [B, D]
    f_ref  = frozen(probe_x)                 # [B, D]
    cos = F.cosine_similarity(f_live, f_ref, dim=1).mean().item()
    live_backbone.train()
    return cos                               # 1.0 = unchanged, ->0 = features destroyed

# In the training loop, every 25 steps:
if step % 25 == 0:
    cos = feature_drift(model.backbone)
    print(f"step {step:4d}  loss {loss.item():.3f}  feat_cos_to_frozen {cos:.3f}")
    if cos < 0.3 and step < 200:
        print("WARNING: features collapsing — LR is almost certainly too high")
```

When I ran this on the broken `5e-3` flowers finetune, the trace was unambiguous: `step 0 cos 1.000`, `step 25 cos 0.41`, `step 50 cos 0.18`, `step 75 cos 0.07`. The features were *gone* by step 50, which is exactly when val accuracy hit chance. With `2e-5` the trace read `step 25 cos 0.97`, `step 100 cos 0.91`, `step 500 cos 0.86` — a slow, controlled drift as the model adapted without forgetting. The drift monitor sees the bug 400 steps before the accuracy curve does, and it tells you the *cause* (features leaving the basin) rather than just the *symptom* (accuracy is bad).

A note on what "healthy drift" actually looks like, because the monitor is only useful if you know the threshold. You do *not* want cosine pinned at 1.0 — that means the backbone is not adapting at all, which is just an expensive linear probe. You want a slow, monotone-ish decline that levels off: starting at 1.0, dropping to perhaps 0.95 in the first hundred steps, and settling somewhere in the 0.80-0.90 range over the run as the late blocks specialize to your task. The danger sign is not a low absolute value, it is the *rate*: a cosine that falls below ~0.5 within the first hundred steps is a collapse in progress, almost always an LR problem, occasionally a bad-batch or numerics problem. A useful refinement is to track the drift *per block* rather than only at the output — compute the cosine of each block's output activations to the frozen reference. In a healthy layer-wise-decay run you will see the early blocks barely drift (cosine 0.97+) while the late blocks drift more (0.85), which is exactly the behavior the decay schedule is supposed to produce. If you see the *early* blocks drifting hard, your layer-wise decay is not actually taking effect — a parameter-grouping bug, which is the next thing to check.

It is worth being explicit about *why* cosine similarity is the right metric here rather than, say, the L2 distance between feature vectors. Feature magnitudes drift naturally during finetuning as the LayerNorm scales adjust, so an L2 distance conflates a benign magnitude change with a malignant directional change. What actually matters for a downstream linear head is the *direction* of the feature vector — the angle, not the length, because a linear classifier reads dot products. Cosine similarity isolates exactly the directional component, so it tracks the quantity your head cares about and ignores the magnitude wobble that is harmless. If you want a single scalar that correlates almost perfectly with "is my finetune destroying its usefulness," directional feature drift is it.

Now the fix the science predicted: **layer-wise LR decay** (also called discriminative or LLRD). Assign each parameter group a learning rate that decays geometrically with depth — the head gets the base LR, each block below it gets multiplied by a decay factor `d` (commonly 0.65-0.9), so the patch embedding and early blocks barely move.

![Timeline diagram of layer-wise learning-rate decay assigning a smaller rate to each deeper block, from the head at 2e-5 down to the patch embedding at 1.8e-6](/imgs/blogs/debugging-vision-transformer-finetuning-7.png)

```python
def layerwise_lr_groups(model, base_lr=2e-5, decay=0.8, weight_decay=0.05):
    """timm ViT: blocks are model.blocks[0..N-1]; patch_embed/cls/pos_embed are 'layer 0'."""
    num_layers = len(model.blocks) + 1          # +1 for the head
    groups = {}

    def layer_id(name):
        if name.startswith("patch_embed") or name in ("cls_token", "pos_embed"):
            return 0
        if name.startswith("blocks."):
            return int(name.split(".")[1]) + 1
        return num_layers                        # head / norm / classifier

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lid = layer_id(name)
        scale = decay ** (num_layers - lid)      # deepest layers ~ base_lr, early layers tiny
        # No weight decay on norms and biases / 1-D params.
        wd = 0.0 if (p.ndim == 1 or "bias" in name or "norm" in name) else weight_decay
        key = (lid, wd)
        groups.setdefault(key, {"params": [], "lr": base_lr * scale, "weight_decay": wd})
        groups[key]["params"].append(p)

    return list(groups.values())

optimizer = torch.optim.AdamW(layerwise_lr_groups(model.backbone), betas=(0.9, 0.999))
```

Two details in that snippet are bugs in their own right if you skip them, and they are the subject of Section 8: **no weight decay on norms and biases** (decaying a LayerNorm's gamma toward zero quietly degrades the pretrained representation), and **the optimizer must be built after you set `requires_grad`** so frozen params are not silently assigned a rate (Section 7). The full HuggingFace equivalent for a `ViTForImageClassification` follows the same structure — group parameters by `vit.encoder.layer.{i}` index, scale by `decay ** (num_layers - i)`, pass the groups to `AdamW` or to `TrainingArguments` via a custom optimizer.

Before we move down the stack, here is the whole diagnostic table in one picture: each of the four big ViT-finetuning knobs, its symptom, the one-line confirming test, and the fix. Print this on the wall next to your monitor — it is the symptom-to-fix lookup for almost every collapse in this post, and the next sections justify each row.

![Matrix diagram mapping each ViT finetuning knob — backbone learning rate, position embeds, preprocessing, and freeze plan — to its symptom, a confirming test, and the specific fix](/imgs/blogs/debugging-vision-transformer-finetuning-4.png)

## 5. Position-embedding interpolation on a resolution change

The second classic ViT-specific bug is silent and spatial: you pretrained (or downloaded a checkpoint trained) at 224×224, you finetune at 384×384 for higher accuracy, and you forget that the position embeddings do not automatically resize. A ViT has no convolutional translation equivariance baked in; it learns *where* each patch is entirely through an additive **position embedding** — one learned vector per patch slot. At 224 with 16×16 patches there are `(224/16)² = 14×14 = 196` patch slots plus one CLS token, so the position-embed table has 197 rows. At 384 there are `(384/16)² = 24×24 = 576` patches plus CLS — 577 rows. If you feed 577 patch positions into a 197-row position table, one of two things happens: the model errors on the shape mismatch (the lucky case, because at least you notice), or — if the code naively truncates or tiles — it silently uses the wrong positions and the spatial structure is scrambled. Accuracy lands well below the probe and you have no idea why.

![Three-by-three grid diagram of position-embedding interpolation reshaping a fourteen by fourteen grid to twenty-four by twenty-four via bicubic resize while preserving the CLS token](/imgs/blogs/debugging-vision-transformer-finetuning-5.png)

The fix is **position-embedding interpolation**, and the math is worth seeing because it tells you why `nearest` is wrong and `bicubic` is right. Strip off the CLS row, reshape the remaining 196 rows into a `14×14×D` grid (because the patches *are* a 2-D grid — slot order is row-major over the image), bicubic-resize that grid to `24×24×D`, flatten back to 576 rows, and prepend the CLS row again. You are treating the position table as a small image with `D` channels and resampling it. Bicubic (or bilinear) interpolation matters because position embeddings vary *smoothly* across the grid — adjacent patches have similar positions — so a smooth interpolant produces sensible embeddings for the new in-between slots. Nearest-neighbor would duplicate embeddings and break that smoothness, which is exactly why both the original ViT paper and DeiT use bicubic.

The crucial subtlety — the one that makes a naive "just resize the table" attempt fail — is that the reshape to a 2-D grid is **mandatory and must precede the interpolation**. The position table is stored as a flat `[N, D]` matrix, but the patches it indexes are laid out in 2-D, row-major: row 0 of the image is slots 0-13, row 1 is slots 14-27, and so on. If you interpolate the flat `[196, D]` table directly along the sequence axis — treating it as a 1-D signal of length 196 — you blend slot 13 (top-right corner) with slot 14 (start of the *next* row, left edge), which are spatially distant. You would be smearing the right edge of one image row into the left edge of the next, producing position embeddings that encode nonsense geometry. The whole point of reshaping to `14×14` first is to make the interpolation respect the true 2-D adjacency: slot 13 is interpolated against its actual neighbors above, below, and to the left, not against the wrap-around. This is why the `interpolate_pos_embed` function below reshapes to `(old, old, D)`, permutes to channels-first, and only *then* calls `F.interpolate` with a 2-D `size`. Getting the reshape wrong is a subtler version of the same bug, and it produces the same symptom: a quietly broken spatial map.

There is also a clean way to think about *how much* accuracy a missing interpolation costs, which helps you recognize it. When the position embeddings are wrong, the model can still use the *content* of each patch (the patch-embedding projection is unaffected), so it does not collapse to chance — it degrades to roughly the accuracy a bag-of-patches model would get, which on most fine-grained tasks is several to a dozen points below the spatially-aware model. That is the signature: not a collapse, but a stubborn, unexplained deficit relative to the lower-resolution run or the linear probe. A collapse is the LR; a deficit at higher resolution is the pos-embed.

```python
import torch
import torch.nn.functional as F

def interpolate_pos_embed(pos_embed, new_grid, has_cls=True, mode="bicubic"):
    """
    pos_embed: [1, 1 + Hp*Wp, D] (CLS + old grid)
    new_grid:  (Hn, Wn) target patch grid, e.g. (24, 24) for 384/16
    """
    D = pos_embed.shape[-1]
    cls = pos_embed[:, :1] if has_cls else None
    grid = pos_embed[:, 1:] if has_cls else pos_embed

    old = int(grid.shape[1] ** 0.5)
    assert old * old == grid.shape[1], "pos-embed grid is not square — check has_cls"

    grid = grid.reshape(1, old, old, D).permute(0, 3, 1, 2)        # [1, D, old, old]
    grid = F.interpolate(grid, size=new_grid, mode=mode, align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, new_grid[0] * new_grid[1], D)

    return torch.cat([cls, grid], dim=1) if has_cls else grid

# Sanity assert you should ALWAYS keep in the model:
# num position rows must equal num patches (+CLS), or spatial structure is broken.
n_patches = (img_size // patch_size) ** 2
assert model.pos_embed.shape[1] == n_patches + 1, (
    f"pos-embed has {model.pos_embed.shape[1]} rows but expects {n_patches + 1}"
)
```

The good news is that the major libraries do this for you *if you ask*. In `timm`, `timm.create_model("vit_base_patch16_224", pretrained=True, img_size=384)` interpolates the position embeddings automatically when you pass a non-default `img_size`. The bug appears when you load a state dict by hand, or change `img_size` after construction, or use a custom model that does not reimplement the interpolation. The diagnostic is the assert above — drop it in the forward pass once, and a resolution mismatch becomes a loud `AssertionError` at step 0 instead of a quiet 10-point accuracy loss at epoch 5.

A related trap lives one level down, in the **patch embedding** itself. The patchify step is a single strided convolution: a `Conv2d(3, D, kernel_size=patch_size, stride=patch_size)` that carves the image into non-overlapping `patch_size × patch_size` tiles and projects each to a `D`-dimensional vector. The patch size is *baked into the pretrained weights* — a `patch16` model has a `16×16` patch-embedding kernel, and you cannot run it as a `patch14` model without retraining that conv. The bug is loading a `patch16` checkpoint into a model configured with `patch_size=14` (or vice versa): either the conv-weight load fails on a shape mismatch (lucky), or, if you constructed a fresh conv and only partially loaded, the patch projection is random and the model is effectively un-pretrained at its very first layer. DINOv2, notably, uses `patch14`, so a copy-paste from a `patch16` recipe that hardcodes `224` as the image size gives you `224/14 = 16` patches per side and a grid that does not match a `patch16` model's `14`. The confirming test is to print `model.patch_embed.proj.weight.shape` and verify the kernel size equals your configured patch size, and that `img_size % patch_size == 0` so the image tiles evenly with no dropped border. A non-divisible resolution silently truncates a strip of the image, which is its own small, annoying accuracy leak.

#### Worked example: the 384 finetune that lost 11 points

I had a `vit_base_patch16` finetuning at 384 to squeeze out accuracy on a fine-grained bird dataset. At 224 it finetuned to 79%; at 384 it should have gained a couple of points (more resolution, more patches, finer detail). Instead it landed at **68%** — eleven points *worse*. No error, no NaN, just a quietly bad number. The feature-drift monitor was clean (cosine stayed at 0.9), so it was not an LR collapse. The probe at 384 was also bad — which pointed at feature extraction, not the finetune. I dumped `model.pos_embed.shape[1]` and got `197`, while `n_patches + 1 = 577`. The checkpoint loader had loaded the 224 position table and the model had silently broadcast/truncated it, so every patch beyond the first 196 was reading a garbage position. After running `interpolate_pos_embed` to resize the table to `24×24` and reloading, the probe jumped to its expected 82% and the finetune landed at **81%** — the gain we expected. The whole bug was a 197 where there should have been a 577, and the assert would have caught it in one line.

## 6. Preprocessing must match the model card

This is the same family as generic CV input-pipeline bugs — BGR vs RGB, wrong normalization, resize/crop mismatch — but with a pretrained-specific twist: the backbone was trained with a *specific* preprocessing, and it expects that exact preprocessing back. The features are only meaningful in the input distribution the model saw during pretraining. Feed it images normalized with the wrong mean and std, or resized with the wrong interpolation, and you are presenting out-of-distribution inputs to a model that has never been asked to be robust to them. The features degrade, the probe under-reads, and the finetune starts from a worse-than-necessary place.

The reason this matters more for a *pretrained* backbone than for a from-scratch model is subtle and worth stating. When you train from scratch, the first layer simply learns whatever normalization your pipeline happens to produce — the model adapts to your preprocessing, so a "wrong" but consistent normalization is mostly harmless. When you finetune, the patch-embedding conv and the early blocks are *already calibrated* to the pretraining distribution. They expect inputs whose per-channel statistics match what they saw. Hand them a distribution shifted by the wrong mean and std, and every downstream activation statistic is off; the LayerNorms, calibrated during pretraining, now normalize a distribution they were not tuned for, and the features land in a region of activation space the model is less competent in. With a small LR and a small finetuning set, the model cannot fully re-calibrate to your wrong preprocessing in the few epochs you train — so the damage persists into the final model. The fix is free and the cost of getting it wrong is silent, which is the worst combination: there is no error message, just a quietly worse number.

The trap that bites people specifically: **CLIP does not use ImageNet normalization.** The ImageNet mean/std that every CV tutorial hardcodes — `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` — is wrong for a CLIP image tower, which uses `mean=[0.481, 0.458, 0.408]`, `std=[0.269, 0.261, 0.276]`. The numbers look close, but the standard deviations differ by ~15-20%, which shifts the input distribution enough to cost you several points. CLIP also expects bicubic resize to 224 and a specific center crop. Many DINO and DINOv2 checkpoints use ImageNet stats; CLIP does not; SigLIP uses yet another normalization. **You cannot guess — you read the model card or, better, ask the library for the model's own config.**

```python
import timm
from transformers import AutoImageProcessor

# timm: the model carries its own data config. ALWAYS use it.
m = timm.create_model("vit_base_patch16_clip_224.openai", pretrained=True, num_classes=0)
cfg = timm.data.resolve_model_data_config(m)
print(cfg)
# {'input_size': (3, 224, 224), 'interpolation': 'bicubic',
#  'mean': (0.481, 0.458, 0.408), 'std': (0.269, 0.261, 0.276),
#  'crop_pct': 0.9, 'crop_mode': 'center'}  <-- CLIP stats, NOT ImageNet
train_tf = timm.data.create_transform(**cfg, is_training=True)
eval_tf  = timm.data.create_transform(**cfg, is_training=False)

# HuggingFace: the processor encodes the model's expected preprocessing. Use it verbatim.
proc = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
print(proc.image_mean, proc.image_std)   # again, CLIP stats
```

The diagnostic is a direct comparison: print the mean and std your transform actually applies and diff them against the model's config. A one-line confirming test — run the linear probe with your transform and with the model's own transform, and compare. If your transform's probe is several points lower, your preprocessing is skewed.

| Backbone family | Normalization mean | Normalization std | Resize interp | Common mistake |
| --- | --- | --- | --- | --- |
| ImageNet ViT (`augreg_in21k`) | `[0.485, 0.456, 0.406]` | `[0.229, 0.224, 0.225]` | bicubic | none if you use ImageNet stats |
| CLIP image tower | `[0.481, 0.458, 0.408]` | `[0.269, 0.261, 0.276]` | bicubic | hardcoding ImageNet stats |
| DINO / DINOv2 | `[0.485, 0.456, 0.406]` | `[0.229, 0.224, 0.225]` | bicubic | DINOv2 wants 14-patch, multiples of 14 |
| SigLIP | `[0.5, 0.5, 0.5]` | `[0.5, 0.5, 0.5]` | bilinear | using ImageNet stats |

Two more pretrained-specific preprocessing traps. First, **channel order**: if you load images with OpenCV you get BGR, and the pretrained model saw RGB — a silent ~3-8 point loss because the red and blue channels are swapped relative to what the patch-embedding conv learned. Second, **resize interpolation**: a model trained with bicubic resize and evaluated with bilinear or nearest sees subtly different patch statistics; it is a small effect but real, and it compounds with the others. The rule is the same in every case: do not reconstruct the preprocessing from memory, ask the library for the model's config and use it for both train and eval.

## 7. Freeze schedules and the optimizer-before-unfreeze bug

Finetuning gives you a dial between two extremes. At one end, **linear probing**: freeze the entire backbone, train only the head. Fast, robust, hard to break, and a great baseline — but it cannot adapt the features to your domain. At the other end, **full finetuning**: everything is trainable. Maximum capacity to adapt, maximum risk of destroying the representation (Section 3). In between live partial-freeze schedules — freeze the early blocks, train the late ones and the head — and staged schedules like **LP-FT** (linear-probe first to warm up the head, then unfreeze and finetune the whole thing at a small LR), which the literature shows preserves out-of-distribution robustness better than naive full finetuning.

The bugs here are mechanical and embarrassingly common. There are three.

**Freezing the wrong parts.** `for p in model.parameters(): p.requires_grad_(False)` freezes *everything*, including your new head, and then your loss does not move at all. Or you freeze by name with a substring match that catches more than you meant. The confirming test is one line: `print_trainable_parameters`.

```python
def print_trainable_parameters(model):
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"trainable {trainable:,} / {total:,}  ({100*trainable/total:.2f}%)")

# Linear probe should print ~0.01% trainable (just the head).
# Full finetune should print 100%. Anything unexpected is a freeze bug.
```

**The optimizer-before-unfreeze bug.** This one is genuinely subtle and I have shipped it more than once. You construct the optimizer over `model.parameters()`, *then* later unfreeze some layers (or, worse, you froze them after constructing the optimizer). PyTorch optimizers capture a reference to the parameter tensors at construction; whether a parameter's `requires_grad` is true or false at construct time does not retroactively change after you flip the flag, but more importantly, if you pass `filter(lambda p: p.requires_grad, model.parameters())` to the optimizer *before* unfreezing, the newly unfrozen parameters were never added to the optimizer and **will never receive an update** — their `.grad` is computed and then ignored, because no param group owns them. The model trains the head, the backbone gradients are non-zero in `.grad`, and yet the backbone weights never change. The signature is maddening: grad-norm probe shows backbone gradients exist, but a weight-delta probe shows the backbone weights are frozen in place.

```python
# BUG: optimizer built over only-currently-trainable params, BEFORE unfreezing.
for p in model.backbone.parameters():
    p.requires_grad_(False)
opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
# ... later, "unfreeze for full finetune" ...
for p in model.backbone.parameters():
    p.requires_grad_(True)          # too late — these params are not in `opt`'s groups

# FIX: set requires_grad to its FINAL value first, THEN build the optimizer.
for p in model.backbone.parameters():
    p.requires_grad_(True)
opt = torch.optim.AdamW(layerwise_lr_groups(model.backbone), lr=2e-5)
```

The weight-delta probe that catches it: snapshot a backbone weight before the step and after, and assert it actually moved when it should.

```python
w0 = model.backbone.blocks[0].mlp.fc1.weight.detach().clone()
# ... one optimizer.step() ...
delta = (model.backbone.blocks[0].mlp.fc1.weight - w0).norm().item()
print(f"block0 weight delta after step: {delta:.2e}")   # 0.0 = this layer is NOT updating
```

**Re-building the optimizer on staged unfreeze without restoring schedule state.** In an LP-FT schedule, when you unfreeze and build a new optimizer for stage two, the LR scheduler resets — if you are not careful you jump back to the warmup LR mid-run, which can re-spike the loss. The fix is to construct stage two's scheduler with the correct starting step, or to use a single optimizer with per-group LRs from the start and simply ramp the backbone groups from zero. This is the finetuning instance of the general checkpoint/resume schedule-state bug.

There is one more freeze-related subtlety that is easy to miss and produces a confusingly *partial* failure: **BatchNorm and the running statistics in models that have them**. Pure ViTs use LayerNorm, which is stateless at eval time, so this does not bite them — but hybrid backbones (a ConvNeXt or a CNN-stem ViT, or a CLIP ResNet tower) carry BatchNorm layers with running mean and variance buffers. If you "freeze" those layers by setting `requires_grad=False` on their parameters but leave the module in `train()` mode, the running statistics *keep updating* on your finetuning batches even though the affine parameters are frozen. Your "frozen" backbone is therefore not frozen — its normalization is silently drifting toward your small dataset's statistics, which on a small or non-representative finetuning set can degrade the features. Freezing a layer correctly means *both* `requires_grad=False` on its parameters *and* `.eval()` on the module so its buffers stop updating. For a pure-LayerNorm ViT you can ignore this; the moment a BatchNorm appears in the backbone, the freeze is a two-part operation and forgetting the `.eval()` half is a real bug. The diagnostic is to snapshot a running-mean buffer before and after an epoch and assert it did not move on the frozen layers.

A practical decision guide for *which* freeze schedule to reach for, since the right answer depends on your data size:

| Data size | Recommended schedule | Why |
| --- | --- | --- |
| < ~1k images | linear probe (freeze all) | too little data to safely adapt 86M params; the probe is robust |
| ~1k-10k images | LP-FT or freeze early blocks | warm the head first, then adapt late blocks at a small LR |
| ~10k-100k images | full finetune + layer-wise decay | enough signal to adapt safely if the LR is small and decayed |
| > ~100k images | full finetune, smaller decay | plenty of data; you can let earlier layers move more |

## 8. Tiny LR needs warmup, and weight decay on the wrong params

Two coupled issues that ride along with the small-LR finetune recipe, both with a clean mechanism.

**Warmup is not optional at finetuning LRs.** Section 3 explained why: at step 0 your head is random, the backbone gradient is large and meaningless, and any nonzero LR risks a step out of the basin during exactly the window when the signal is worst. Warmup — ramping the LR linearly from 0 to the target over the first few hundred steps — keeps the steps tiny until the head produces a useful gradient. It is cheap insurance and it is the difference between a finetune that occasionally spikes in the first epoch and one that is monotone. A short warmup (3-10% of total steps) plus cosine decay is the default that almost always works.

There is also an Adam-specific reason warmup helps that is independent of the random-head argument, and knowing it stops you from skipping warmup when you happen to be using a pretrained head. AdamW maintains running estimates of the first and second moments of the gradient, and at the start of training those estimates are based on almost no samples — they are noisy. The bias-correction terms partly compensate, but in the first few dozen steps the effective per-parameter step size is governed by a poorly-estimated second moment, which can produce a few erratically large updates. Warmup holds the global LR small during exactly the window when Adam's internal statistics are least reliable, so a noisy moment estimate cannot turn into a large weight step. The two arguments — random head and unwarmed Adam state — point the same way, and together they explain why essentially every published transformer finetuning recipe warms up. If you observe a loss spike in the first hundred steps that then recovers, the fix is almost always more warmup, not a lower peak LR.

```python
from transformers import get_cosine_schedule_with_warmup

total_steps = num_epochs * len(train_loader)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.05 * total_steps),   # ~5% warmup — keeps step-0 steps tiny
    num_training_steps=total_steps,
)
# Step the scheduler every optimizer step, not every epoch.
```

**Weight decay on the wrong parameters quietly corrupts the representation.** AdamW's weight decay pulls every parameter it touches toward zero. For the weight matrices of attention and MLP blocks, a little decay is healthy regularization. But applied to **LayerNorm gains (gamma), biases, and the CLS token**, decay is actively harmful: it pulls the pretrained LayerNorm scale toward zero, flattening the very normalization the model relies on, and it shrinks the learned CLS token. These are 1-D parameters that encode calibrated, pretrained structure, and decaying them is a slow corrosion of the representation that shows up as a finetune that trains fine but lands a couple of points low for no obvious reason. The fix is the standard "no-decay group": exclude all 1-D parameters (norms, biases) and the position/CLS embeddings from weight decay. The `layerwise_lr_groups` function in Section 4 already does this with the `wd = 0.0 if (p.ndim == 1 ...)` line — but if you let `AdamW(model.parameters(), weight_decay=0.05)` decay everything uniformly, you have this bug.

The mechanism is precise enough to derive. AdamW's decoupled weight decay applies, every step, the update $\theta \leftarrow \theta - \eta\lambda\theta$ in addition to the gradient step, where $\lambda$ is the weight-decay coefficient. For a parameter receiving little or no task gradient — which is exactly the case for a frozen-ish early-layer LayerNorm gamma whose gradient is tiny under layer-wise decay — the decay term dominates, and the parameter shrinks geometrically toward zero at rate $(1 - \eta\lambda)$ per step. Over a few thousand steps with $\eta\lambda$ on the order of $10^{-3}$ that is a multiplicative shrinkage of a few percent, which sounds small but is applied to the *scale* of a normalization the entire downstream network was calibrated against. A LayerNorm with gamma pulled 5% toward zero rescales every activation flowing through it, and because that rescaling compounds through depth, the cumulative distortion of the output features is much larger than 5%. This is the same compounding-through-depth effect from Section 3, now working against you through the normalization layers. The fix costs nothing and is unambiguously correct: never decay 1-D parameters of a pretrained model. Every strong finetuning recipe does this; the bug is only ever an omission.

To see whether you have the bug, audit your optimizer's param groups directly — print, for each group, how many parameters it holds and what weight decay it applies, and confirm that the norms and biases live in a zero-decay group.

```python
for i, g in enumerate(optimizer.param_groups):
    n = sum(p.numel() for p in g["params"])
    print(f"group {i}: {n:>10,} params | lr {g['lr']:.2e} | wd {g['weight_decay']}")
# You should see at least one group with wd=0.0 holding the 1-D params (norms/biases/CLS).
# If EVERY group has the same nonzero wd, you are decaying the norms — fix the grouping.
```

| Parameter type | Apply weight decay? | Why |
| --- | --- | --- |
| Attention / MLP weight matrices | yes (0.01-0.1) | healthy regularization on 2-D weights |
| LayerNorm gamma / beta | **no** | decaying the norm scale corrupts the pretrained representation |
| Biases (1-D) | **no** | biases encode offsets, not magnitudes to shrink |
| Position embeddings, CLS token | **no** | learned structure; shrinking it loses spatial/pooling info |
| New classification head | yes (small) | fresh params; light decay helps generalization |

## 9. CLS token versus mean pooling, and overfitting fast on small data

**Two final ViT-specific decisions.** First, which feature does your head read? A ViT produces one token per patch plus a CLS token; the standard ImageNet ViT pools the **CLS token** (the model was pretrained with a head on CLS, so CLS carries the global representation). But many self-supervised backbones — DINO, MAE, and others — were pretrained *without* a supervised CLS head, and their best global feature is the **mean over patch tokens** (mean-pooling), not CLS. If you bolt your head onto the wrong token, your features under-read: a DINO backbone read via CLS instead of mean-pool can lose several points because CLS was never trained to summarize. The confirming test is to probe both and compare.

```python
# Extract both pooled features and probe each — pick the one with the higher probe.
@torch.no_grad()
def pooled_features(backbone, x):
    tokens = backbone.forward_features(x)        # timm: [B, 1+N, D] (CLS + patches)
    cls    = tokens[:, 0]                         # CLS token
    mean   = tokens[:, 1:].mean(dim=1)            # mean over patch tokens
    return cls, mean
# Then run the Section-2 linear probe on `cls` and on `mean`; use the winner for your head.
```

Second, **ViTs overfit small datasets fast.** A ViT has far weaker built-in inductive bias than a CNN — no locality, no translation equivariance baked into the architecture — so on a few-thousand-image dataset it will memorize the training set in a handful of epochs if you let it. The signature is a train accuracy that races to 99% while val stalls, with the gap opening inside the first epoch. The fixes are the small-data finetuning toolkit: **few epochs** (1-3, with early stopping on val), **strong augmentation** (RandAugment, Mixup, CutMix, the augreg recipe the original ViT-21k models used), **layer freezing or layer-wise decay** so the early blocks stay near their pretrained values, and **weight decay on the right params** (Section 8). The diagnostic is the train-val gap curve: if it opens fast, you are overfitting, and the lever is regularization plus fewer epochs, not a bigger model.

The reason ViTs overfit faster than CNNs of similar size is worth understanding because it tells you which knob to turn. A convolution hard-codes two priors into the architecture: **locality** (a pixel's label depends mostly on its neighborhood) and **translation equivariance** (a cat is a cat wherever it appears). Those priors are not learned — they are structural, free, and correct for natural images, and they dramatically shrink the effective hypothesis space the model searches. A ViT bakes in none of that. Every spatial relationship it uses, it learned from data during pretraining. That is a strength at scale — given enough data, a ViT can learn priors a CNN cannot express, which is why ViTs win on large datasets — but it is a liability on small finetuning sets, where the model has the capacity to fit arbitrary patch-to-label mappings, including the spurious ones that happen to separate your few thousand training images. The practical consequence: ViT finetuning leans much harder on *augmentation and regularization* to substitute for the inductive bias the architecture lacks. The `augreg` in `vit_base_patch16_224.augreg_in21k` literally stands for "augmentation and regularization" — those models were pretrained with heavy augmentation precisely because the architecture needs it, and your finetune should keep that going.

A second small-data trap specific to the head: **the new classification head is the only randomly-initialized part, and it is where overfitting concentrates first**. With a frozen or slowly-moving backbone, the head can fit the training set on its own within a couple of epochs because it is a fresh linear (or small MLP) layer with no pretrained constraints. This is actually good news for diagnosis — if you see fast overfitting with a small backbone LR, the head is usually the culprit, and the levers are head dropout, label smoothing, and weight decay *on the head*. Label smoothing in particular (replacing the one-hot target with a softened distribution, e.g. 0.9 on the true class and 0.1/N elsewhere) is a cheap, reliable regularizer for the head that also calibrates the output probabilities. It is the first thing I add when a ViT finetune overfits and I cannot get more data.

#### Worked example: a ViT that overfit 4,000 images in two epochs

A `vit_base` finetune on a 4,000-image, 30-class medical-imaging dataset hit 99.8% train accuracy by the end of epoch 2 while val plateaued at 71%. The feature-drift monitor was healthy (cosine 0.88), the linear probe was 74%, so the features were fine and the backbone was not collapsing — this was pure overfitting, and the finetune had actually dropped *below* the probe by 3 points. I made three changes, all regularization: added RandAugment (`rand-m9-mstd0.5`) and random erasing to the train transform, added Mixup with `alpha=0.2`, and added label smoothing of 0.1 to the loss; I also cut training from 10 epochs to 3 with early stopping on val. Train accuracy fell to a healthier 91% (the model could no longer trivially memorize), and val *rose* to 81% — ten points, entirely from regularization, no architecture change and no extra data. The before-after is the textbook overfitting fix: when train and val diverge fast, you do not need a different model, you need to make the training task harder so the model cannot cheat. The lever was never the LR; it was the augmentation policy.

```python
# A compact, robust small-data ViT finetune config (timm + AdamW + LLRD).
import timm
model = timm.create_model("vit_base_patch16_224.augreg_in21k",
                          pretrained=True, num_classes=102, drop_path_rate=0.1)
cfg   = timm.data.resolve_model_data_config(model)
train_tf = timm.data.create_transform(**cfg, is_training=True,
                                       auto_augment="rand-m9-mstd0.5",  # RandAugment
                                       re_prob=0.25)                    # random erasing
optimizer = torch.optim.AdamW(layerwise_lr_groups(model, base_lr=2e-5, decay=0.8))
# + cosine schedule with 5% warmup, Mixup/CutMix collate, 3 epochs, early stop on val.
```

## 10. Before and after: putting it together on the collapsed flowers run

Here is the full before-after on the running example, the `vit_base_patch16_224` flowers-102 finetune that started this post at 1.0% accuracy. The fix was not one change — it was the stack of changes this post built, applied together, each justified by a measured signal.

![Before-and-after diagram showing a naive ViT finetune at one percent accuracy fixed by lowering the learning rate, adding layer-wise decay, and interpolating position embeds to reach ninety-two percent](/imgs/blogs/debugging-vision-transformer-finetuning-8.png)

| Signal | Before (naive finetune) | After (tuned finetune) | What confirmed it |
| --- | --- | --- | --- |
| Backbone LR | `5e-3` (from-scratch recipe) | `2e-5` + layer-wise decay 0.8 | feature-drift monitor |
| Feature cosine to frozen @ step 100 | `0.07` (collapsed) | `0.91` (controlled drift) | `feature_drift()` probe |
| Warmup | none | 5% linear warmup | loss spike in epoch 1 disappeared |
| Position embeds | loaded as-is | bicubic-interpolated to grid | `pos_embed.shape` assert |
| Weight decay | uniform 0.05 on all params | excluded norms/biases/CLS | no-decay param groups |
| Linear-probe baseline | (not run) | `0.88` | `sklearn` logistic regression |
| Val accuracy | `1.0%` (chance) | `92%` (beats probe by 4 pts) | held-out val set |

The headline result: a finetune that was *below chance-adjacent* — actively worse than doing nothing — became one that beats the strong linear-probe baseline. And critically, the diagnostic order was the point. The linear probe (Section 2) told us the features were good (88%), which ruled out data and feature-extraction bugs and pointed straight at optimization. The feature-drift monitor (Section 4) confirmed the features were being destroyed, not adapted, which is the signature of a too-high LR. The fix — drop LR 250×, add layer-wise decay, warmup, and the pos-embed interpolation — followed directly from the two instruments. We never guessed.

#### Worked example: CLIP finetune that trained but under-probed

A second, subtler case worth its own walkthrough. A CLIP ViT-B/16 image tower finetuned on a retail-product dataset *trained* — loss fell smoothly, val accuracy reached 74%. No collapse, no drama. But the linear probe came in at **79%**, five points *above* the finetune. The finetune was making the features worse, just slowly. Feature drift was healthy (cosine 0.9), so it was not an LR collapse. The tell was in the preprocessing: the team had used ImageNet normalization stats on a CLIP tower. The probe at 79% was itself under-reading, because *its* preprocessing was also wrong. We switched both the probe and the finetune to CLIP's own mean/std via `resolve_model_data_config` (Section 6). The probe jumped to **84%**, and the finetune — same LR, same schedule, only the normalization fixed — landed at **86%**, two points above the corrected probe. The bug was four floating-point numbers in a normalization transform, and it cost ten points across probe and finetune combined. The lesson: a finetune that under-probes is not always an LR problem — check that *both* the probe and the finetune are using the model's real preprocessing before you blame optimization.

## 11. When this is (and isn't) your bug

Be decisive about ruling ViT-finetuning bugs in and out, because the symptoms overlap with bugs that live elsewhere in the six places.

- **Collapse to chance in the first ~100 steps, with feature cosine cratering toward zero → it's the LR (optimization).** This is the destruction signature. Lower the LR, add warmup and layer-wise decay. If the cosine *stays high* and you are still at chance, it is **not** an LR collapse — look at the head wiring or preprocessing.
- **Finetune lands a few points below the linear probe, feature drift healthy → preprocessing or pos-embed.** Slow degradation rather than collapse points at OOD inputs (wrong normalization) or scrambled positions (un-interpolated pos-embed), not the optimizer. Diff your transform against the model card and assert `pos_embed.shape`.
- **Linear probe is ALSO at chance → it's not the finetune at all, it's feature extraction.** Stop tuning the optimizer. Your preprocessing is wrong, your pos-embed is wrong, or you are pooling the wrong token. No learning rate fixes a backbone that is being fed out-of-distribution inputs or read through the wrong feature.
- **Train accuracy races to 99%, val opens a gap in epoch 1 → overfitting, not a forgetting bug.** This is the small-data ViT story. The lever is regularization (strong aug, weight decay on the right params, fewer epochs, freezing), not a smaller LR. A smaller LR alone will not fix overfitting; it just overfits more slowly.
- **Backbone gradients exist but weights don't move → optimizer/freeze construction (model code).** The weight-delta probe is positive on the head and zero on the backbone. You built the optimizer before unfreezing, or the params were never added to a group. This is mechanical, not numerical.
- **A smooth loss that then NaNs → numerics, not a ViT-specific finetune bug.** That is a mixed-precision or initialization story; the ViT-specific bugs in this post degrade *quietly* and almost never NaN. If you see a NaN, you are in a different chapter.

The unifying rule: **the linear probe and the feature-drift monitor between them localize almost every ViT finetuning failure.** Probe good + drift bad = optimization. Probe good + drift good but under-probing = preprocessing/pos-embed. Probe bad = feature extraction, full stop. Run those two instruments before you change a line of code.

One honest caveat about the boundaries of this method. The linear probe and feature-drift monitor are powerful precisely because they assume the backbone is a *good* pretrained model being adapted to a related task. They are less informative in two situations. First, a **large domain gap**: if you are finetuning a natural-image ImageNet ViT onto, say, satellite imagery or medical scans, the linear probe may genuinely be poor *and* a careful finetune may genuinely help a lot — here a low probe does not mean "broken pipeline," it means "the pretrained features don't transfer well to this domain and you legitimately need to adapt them." You distinguish the two by checking the pipeline first (preprocessing, pos-embed, pooled token all correct?) and only then concluding the gap is real. Second, **very large finetuning datasets**: with hundreds of thousands of in-domain images, you can afford to move the backbone far, the feature-drift cosine will legitimately fall lower than the small-data regime, and a low-ish cosine is no longer alarming. The drift monitor's threshold is a function of how much data you have. In both cases the instruments still *work* — they just need their thresholds read in context rather than as absolutes. The method is a default, not a law.

## 12. Case studies and real signatures

A few named, well-documented patterns that match the mechanisms above, so you can recognize them in the wild.

**Position-embedding interpolation in the original ViT and DeiT.** The ViT paper (Dosovitskiy et al., 2021, "An Image Is Worth 16×16 Words") explicitly performs 2-D interpolation of the pretrained position embeddings when finetuning at higher resolution, and DeiT (Touvron et al., 2021) does the same with bicubic interpolation. This is not a workaround — it is the designed, expected procedure for any resolution change, and it is the reason `timm`'s `img_size` argument triggers automatic interpolation. The bug is not that interpolation is exotic; the bug is forgetting that it is *required* when you load a checkpoint by hand.

**LP-FT and the robustness of two-stage finetuning.** Kumar et al. (2022, "Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution") showed rigorously that naive full finetuning can *distort* pretrained features and underperform a linear probe on out-of-distribution data, and that warming up the head with a linear probe before unfreezing (LP-FT) preserves the pretrained features better. This is the published, peer-reviewed version of the destruction mechanism in Section 3 — the features really do get distorted by an aggressive finetune, and the fix is the same staged, small-step approach.

**Layer-wise LR decay in BEiT and the strong ViT recipes.** Layer-wise learning-rate decay is standard in the strongest ViT finetuning recipes (BEiT, MAE finetuning, and the `timm` reference scripts), with decay factors around 0.65-0.9. It is not a niche trick; it is part of why those recipes hit their headline numbers. The published default validates the Section-4 mechanism: deeper layers, larger steps; early layers, tiny steps.

**CLIP's normalization, repeatedly rediscovered.** The single most common pretrained-ViT preprocessing bug in practice is applying ImageNet normalization to a CLIP tower. CLIP (Radford et al., 2021) ships with its own normalization constants, and using ImageNet's instead is a small, silent distribution shift that costs a handful of accuracy points — a pattern that shows up over and over in finetuning post-mortems because the numbers are close enough to look right.

**The wrong pooled token on self-supervised backbones.** DINO and DINOv2 (Caron et al., 2021; Oquab et al., 2023) and MAE (He et al., 2022) are pretrained without a supervised CLS head, and their reference evaluations pool features differently from a supervised ImageNet ViT — DINOv2's strongest linear-probe results, for instance, concatenate the CLS token with the mean of the patch tokens. Bolting a head onto CLS alone for a DINO backbone, out of habit from supervised ViTs, leaves accuracy on the table because CLS was never trained to be the sole global summary. This is the published reason the Section-9 "probe both tokens" step exists: the right pooled feature is a property of *how the backbone was pretrained*, not a universal default, and the only safe move is to measure both and use the winner.

**Per-layer feature reuse and why early layers transfer.** A long line of transfer-learning work (going back to Yosinski et al., 2014, "How transferable are features in deep neural networks?") established empirically what Section 3 derives geometrically: early layers learn general features that transfer across tasks almost unchanged, while later layers are increasingly task-specific. That finding is the empirical backbone of layer-wise LR decay and of freeze schedules — it is *why* freezing or barely-moving the early blocks costs you almost nothing while protecting you from catastrophic forgetting. The mechanism (compounding Jacobians) and the empirics (measured per-layer transferability) agree, which is the comfortable position to debug from.

## 13. A reusable ViT-finetune debugging checklist

Run these in order. Each step either clears a class of bug or localizes it.

```bash
# 1. Linear probe FIRST. Freeze backbone, fit logistic regression on cached features.
#    Probe good  -> features fine, any collapse is the finetune's fault.
#    Probe bad   -> feature extraction is broken (preprocess / pos-embed / pool). STOP here.

# 2. Diff your preprocessing against the model card.
#    timm:  timm.data.resolve_model_data_config(model)
#    HF:    AutoImageProcessor.from_pretrained(...).image_mean / .image_std
#    CLIP != ImageNet stats. SigLIP != ImageNet stats.

# 3. Assert position-embed rows == num_patches + 1 for your input resolution.
#    Mismatch -> bicubic-interpolate the pos-embed grid (timm img_size= does it for you).

# 4. print_trainable_parameters(model). Linear probe ~0.01%, full FT 100%.
#    Build the optimizer AFTER setting final requires_grad (optimizer-before-unfreeze bug).

# 5. Set backbone LR to 1e-5..5e-5 (NOT 1e-3). Add layer-wise decay 0.65-0.9.
#    Add 3-10% linear warmup. Exclude norms/biases/CLS from weight decay.

# 6. Monitor feature drift: cosine(live features, frozen features) every N steps.
#    Cosine cratering toward 0 in first 100 steps -> LR still too high.

# 7. Probe CLS vs mean-pool; use whichever probes higher for your head.

# 8. Watch the train-val gap. Opens fast on small data -> regularize (aug, wd, freeze, fewer epochs).
```

## Key takeaways

- **Run the linear probe before you finetune.** A frozen backbone + logistic regression gives you the floor and the bisection: probe good means any collapse is optimization; probe bad means feature extraction is broken and no LR will save you.
- **A from-scratch LR destroys a pretrained backbone.** Finetune at `1e-5`-`5e-5`, not `1e-3`. The features are a sharp basin that took hundreds of GPU-days to find; a big step leaves it in the first dozen updates, while the head is still random.
- **Monitor feature drift, not just loss.** Cosine similarity of live features to the frozen backbone craters toward zero ~400 steps before val accuracy confirms a collapse, and it names the cause (features leaving the basin) instead of just the symptom.
- **Layer-wise LR decay is the right shape.** Deep blocks adapt most (they are closest to the features your task reads); early blocks barely move. Decay each block by 0.65-0.9 per depth.
- **Interpolate position embeddings on any resolution change.** 224→384 turns 197 position rows into 577; assert `pos_embed.shape[1] == num_patches + 1` and bicubic-resize the grid, or spatial structure silently breaks.
- **Match preprocessing to the model card, not your memory.** CLIP and SigLIP do not use ImageNet normalization; ask `resolve_model_data_config` or the HF processor for the model's real mean/std/interp.
- **Build the optimizer after setting `requires_grad`.** Optimizer-before-unfreeze leaves newly trainable params in no param group; their gradients are computed and ignored. The weight-delta probe catches it.
- **Exclude norms, biases, and CLS from weight decay, and use warmup.** Decaying 1-D pretrained params corrodes the representation; a tiny LR still needs warmup because the step-0 gradient (random head) is the least trustworthy of the run.
- **ViTs overfit small data fast.** A widening train-val gap in epoch 1 is overfitting, not forgetting — regularize (strong aug, fewer epochs, freezing), do not just lower the LR.

## Further reading

- Dosovitskiy et al., "An Image Is Worth 16×16 Words: Transformers for Image Recognition at Scale" (2021) — the original ViT, including resolution-change position-embedding interpolation.
- Touvron et al., "Training data-efficient image transformers & distillation through attention" (DeiT, 2021) — bicubic pos-embed interpolation and the small-data ViT training recipe.
- Kumar et al., "Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution" (2022) — the rigorous case for LP-FT and why aggressive finetuning destroys features.
- Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP, 2021) — the CLIP image tower and its specific normalization constants.
- Bao et al., "BEiT: BERT Pre-Training of Image Transformers" (2022) — layer-wise learning-rate decay as a standard finetuning ingredient.
- The `timm` documentation and training scripts (Ross Wightman) — `resolve_model_data_config`, automatic pos-embed interpolation via `img_size`, and reference layer-wise-decay recipes.
- Within this series: [a taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) for the symptom→suspect→test→fix decision tree, and [the training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) for the full bisection method.
- The LR mechanics here generalize: [the learning rate is almost always the problem](/blog/machine-learning/debugging-training/the-learning-rate-is-almost-always-the-problem) covers the LR-range test and warmup in depth, and [finetuning an LLM without breaking it](/blog/machine-learning/debugging-training/finetuning-an-llm-without-breaking-it) is the language-model twin of this post.
- The input side: [CV input pipeline bugs](/blog/machine-learning/debugging-training/cv-input-pipeline-bugs) for normalization and channel-order traps, and [finetuning pitfalls across modalities](/blog/machine-learning/debugging-training/finetuning-pitfalls-across-modalities) for the cross-modal synthesis.
- For the efficiency side of vision transformers, [efficient attention and vision transformers for the edge](/blog/machine-learning/edge-ai/efficient-attention-and-vision-transformers-for-edge) covers serving these backbones once they are finetuned.

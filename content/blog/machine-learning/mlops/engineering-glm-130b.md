---
title: "Engineering GLM-130B: Stability Tricks, the FP16 Bet, and INT4 on Four 3090s"
date: "2026-06-10"
publishDate: "2026-06-10"
description: "A deep dive into the GLM-130B technical report as an engineering document: the LayerNorm bake-off, embedding-gradient-shrink, the deliberate FP16-over-BF16 decision, 4x8 tensor-pipeline parallelism on 768 A100s, and the INT4 quantization that put a 130B model on four consumer GPUs."
tags:
  [
    "glm-130b",
    "training-stability",
    "mixed-precision",
    "fp16",
    "deepnorm",
    "quantization",
    "int4",
    "parallelism",
    "mlops",
    "large-language-model",
    "gpu",
  ]
category: "machine-learning"
subcategory: "MLOps"
author: "Hiep Tran"
featured: true
readTime: 50
---

There is a version of the GLM-130B story that is about a model, and it is the boring version. A 130-billion-parameter bilingual transformer, trained in 2022, competitive with GPT-3 — fine, the leaderboard has many of those now. The interesting version is that GLM-130B is one of the most honest *engineering* documents the field has produced. The [technical report](https://arxiv.org/abs/2210.02414) is not really about an architecture. It is about the dozens of ways a 100-billion-parameter training run tries to kill itself, and the specific, reproducible countermeasures that kept this one alive for sixty days on hardware the authors deliberately chose to be *modest*.

This is the third article in a [series reading the entire GLM lineage](/blog/machine-learning/large-language-model/glm-lineage-frontier-llm-technique). The [first article on the objective](/blog/machine-learning/large-language-model/autoregressive-blank-infilling-glm) covered what GLM *is*; this one is about what it took to *build* one at scale, on a budget, without a divergent loss curve. If you have ever babysat a large training run — refreshing a dashboard at 2 a.m., watching the loss for the twitch that means you've lost a day — this is the report you wish you'd read first.

I'm going to treat it the way I'd treat a postmortem: not "here is the final system" but "here is everything that almost went wrong, and the fix." Because that is where the transferable knowledge lives. Anyone can copy a hyperparameter table. The expensive knowledge is *which* failure modes are real at 100B and *which* countermeasure actually works.

## Why "scale is the whole game" is the wrong frame

The lazy reading of the 2022 large-model era is that it was about scale: more parameters, more tokens, more GPUs. GLM-130B's report is a sustained argument that, past a certain size, the binding constraint is not scale but *stability* — your ability to keep the run from diverging — and *accessibility* — your ability to run the result on hardware that exists. Here is the mismatch laid out:

| Assumption | The naive view | The reality in the GLM-130B report |
| --- | --- | --- |
| "The hard part is the architecture" | Pick the right blocks and scale | Three reasonable LayerNorm placements *all diverged*; the architecture was the easy part |
| "Just use BF16 like everyone" | BF16 avoids overflow, end of story | BF16 cost ~15% more memory and excluded V100-class GPUs; FP16 was chosen *despite* its overflow problem |
| "Loss spikes are bad luck" | Restart from a checkpoint and hope | Spikes were *predictable* — the embedding gradient norm rose several steps before each collapse |
| "Quantization is a deployment detail" | Quantize at the end if you must | INT4 was a design goal; GLM's weight distribution was the thing that made it work |
| "100B inference needs a cluster" | You need 8x A100 to serve it | INT4 put inference on 4x RTX 3090 — a workstation |

Every row is a section below. The through-line is that GLM-130B treated the *systems* problem as the actual research problem, and wrote down the negative results — the divergent runs, the rejected precisions — that most labs keep private.

> A hyperparameter table tells you what worked. A list of what *diverged* tells you what to never try. The second list is worth more, and almost nobody publishes it.

## The mental model: a stack of affordability

![A layered stack of the GLM-130B engineering choices, from the training objective at the bottom to single-server inference at the top](/imgs/blogs/engineering-glm-130b-1.png)

The stack above is the mental model for the whole report. Read it bottom-to-top as a chain of *affordability*: each layer exists to make the layer above it possible on the hardware the team actually had.

- The **objective** (`[MASK]` + `[gMASK]` blank infilling on 400B bilingual tokens) is inherited from [the GLM paper](/blog/machine-learning/large-language-model/autoregressive-blank-infilling-glm).
- The **architecture** (70 layers, 12,288 hidden, 96 heads, GeGLU, RoPE) is deliberately conventional, so that the novelty budget could be spent on the layers above.
- **Stability** (Post-LN + DeepNorm + embedding-gradient-shrink) is what keeps the run from diverging.
- **Precision** (FP16 forward/backward, FP32 softmax and optimizer) is the deliberate, contrarian bet.
- **Parallelism** (4-way tensor × 8-way pipeline × 24-way data) spreads the model over 768 A100s.
- **Quantization** (INT4 round-to-nearest) shrinks it back down.
- **Inference** (4× RTX 3090) is the payoff: a 130B model on a workstation.

The rest of this article is a tour up that stack, with the failure modes and fixes at each level. I'll spend the most time on stability and precision, because that is where the report spends its most hard-won pages.

## 1. The architecture, briefly

The senior rule of thumb that governs the architecture is: **spend your novelty budget on at most one thing.** GLM-130B's novelty is the objective and the engineering; the transformer itself is intentionally boring, and that restraint is a feature, because it means the team could attribute any instability to the parts they changed rather than a confounded pile of changes.

The configuration, worth memorizing as a reference for "what a stable dense 100B looks like":

| Spec | Value | Note |
| --- | --- | --- |
| Parameters | 130B | dense, bilingual EN+ZH |
| Layers | 70 | the "9×8−2" pipeline-balanced count |
| Hidden dim | 12,288 | |
| Attention heads | 96 | 128 dims/head |
| Sequence length | 2,048 | |
| Vocabulary | 150,000 | ~130k text + reserved image tokens |
| FFN | GLU with GeLU (GeGLU) | gated FFN |
| Positional encoding | RoPE | rotary |
| Objective | `[MASK]` 30% + `[gMASK]` 70% | short spans + long suffix |

Two of these choices were early adoptions that later became near-universal: **GeGLU** (a gated FFN variant) and **RoPE** (rotary positions). The one that looks strange — **70 layers** — is the first hint that engineering drove the architecture. 70 is not a round number; it is `9 × 8 − 2`. The pipeline has 8 stages, each holding 9 transformer layers, minus 2 to balance the load against the embedding-heavy first and last stages. We'll come back to why in the parallelism section. For now, notice the tell: when a model's depth is a slightly-off multiple of its pipeline width, the depth was chosen by the systems team, not the science team.

The objective split — `[MASK]` on 30% of sequences (short Poisson spans) and `[gMASK]` on 70% (a long generative suffix) — is the production form of the multi-task regime mixing from the original paper. The heavy weighting toward `[gMASK]` biases GLM-130B toward generation, which is the right call for a model meant to be a general assistant.

The two mask tokens are worth distinguishing precisely, because the split is a deliberate capability dial. `[MASK]` triggers the short-span blank infilling from the original paper — Poisson-length spans scattered through the text, reconstructed from bidirectional context. It builds the understanding muscle. `[gMASK]` masks a single *long suffix* of the document and asks the model to generate it, which is just left-to-right generation conditioned on a bidirectional prefix. It builds the generation muscle. At a 30/70 split, the model spends most of its training learning to generate long coherent text and a meaningful minority learning to fill local blanks — exactly the balance you want for an assistant that mostly *produces* text but needs to *understand* the prompt that conditions it. The choice of *where* on this dial to sit is one of the few genuinely tunable knobs in the recipe, and the lineage keeps tuning it: a more understanding-heavy product would push toward `[MASK]`, a pure generator toward `[gMASK]`. The fact that the split is reported as a clean 30/70 (rather than, say, 27/73) suggests it was chosen on principle rather than swept exhaustively — a reasonable default the team didn't have the GPU budget to over-optimize.

## 2. Stability: the part nobody else publishes

> **Senior rule of thumb:** at 100B+ parameters, your run does not fail because the math is wrong. It fails because a single layer's activations grew until a float overflowed, or a gradient spiked and never recovered. Stability *is* the research problem.

The report is unusually blunt that training stability was *the* decisive factor in whether GLM-130B succeeded at all. This is the section that justifies the whole article.

To appreciate why stability dominates at this scale, it helps to think about what a single divergence *costs*. At small scale, if a run diverges you restart it in an afternoon and you've lost an afternoon. At 130B on 768 GPUs, a divergence at day forty has cost you forty days of 768-GPU time — a sum of money that buys a small house — and you may not even know *why* it diverged, so the restart might diverge again. The asymmetry is brutal: a stable-but-slightly-suboptimal recipe that finishes is worth vastly more than a theoretically-optimal recipe that diverges at day forty. This reframes the whole engineering problem. You are not optimizing for the best possible model; you are optimizing for the best model *that reliably completes training*, and those are different objectives. The GLM team's choices — DeepNorm over the faster Pre-LN, FP16-with-fixes over hand-babysat instability — consistently favor *reliability* over peak performance, because at this scale reliability is where the money is. That is the unstated thesis of the stability section: at 100B+, the expected value of a technique is dominated by its effect on the probability of *finishing*, not its effect on the final loss.

### The LayerNorm bake-off

The placement of LayerNorm relative to the residual connection is one of those choices that looks cosmetic and is anything but. There are three standard options — Pre-LN (norm before the sublayer), Post-LN (norm after), and Sandwich-LN (norm on both sides) — and at small scale they all train fine, with minor differences. At 130B on FP16, the GLM team found that all three *diverged*.

![A matrix of the LayerNorm bake-off, showing Pre-LN, Post-LN, and Sandwich-LN all diverging while Post-LN with DeepNorm trains stably](/imgs/blogs/engineering-glm-130b-2.png)

The matrix above is the result of running the experiment nobody wants to run, because each cell costs days of GPU time. Pre-LN, Post-LN, and Sandwich-LN each diverged. Only **Post-LN with DeepNorm** produced the bottom row: a small, flat gradient norm and a stable run to 400B tokens. The report's own words are worth quoting because the negative result is the gift:

> Pre-LN, Post-LN, and Sandwich-LN all diverged; DeepNorm is the most stable one, with a small gradient norm that does not spike in early training.

DeepNorm is a specific recipe, not just "Post-LN." It rescales the residual branch and the initialization together. Concretely, the residual is `DeepNorm(x) = LayerNorm(α·x + Sublayer(x))` with `α = (2N)^½` for an `N`-layer network, and the FFN, value, and output projections are initialized with a gain of `(2N)^−½`. The intuition: as you stack more layers, the contribution of each new layer to the residual stream is *down-weighted* at init and the skip connection is *up-weighted*, so the value scale of the deep layers stays bounded instead of compounding. At 70 layers, the difference between bounded and compounding value scales is the difference between a run that survives and one that NaNs out in the first thousand steps.

It's worth understanding *why* each rejected variant fails, because the failure modes are instructive. **Pre-LN** (norm before the sublayer) is the modern default precisely because it's *easy* to train — the residual path is clean, so gradients flow without much LayerNorm interference. But that cleanliness has a cost: the residual stream's magnitude grows roughly linearly with depth, because every sublayer adds its output to an un-normalized running sum. At 70 layers in FP16, that growing magnitude eventually pushes activations toward the edge of the representable range, and the report found it diverged. **Post-LN** (norm after the residual add) keeps the stream's magnitude controlled — that's why the original transformer used it — but it makes the *gradient* path treacherous: the LayerNorm sits between every sublayer and the loss, and at depth the gradients passing back through 70 of them either vanish or explode depending on initialization. Vanilla Post-LN at 130B explodes. **Sandwich-LN** (norm on both sides) was an attempt to get the best of both and, in the GLM team's runs, got the instability of both instead. DeepNorm is Post-LN with the *scaling* that makes the gradient path survive: by up-weighting the skip connection (`α = (2N)^½`) and down-weighting the sublayer's initialization (`(2N)^−½`), it keeps both the forward magnitude bounded *and* the backward gradient well-conditioned. It is, in effect, a Post-LN whose constants were derived to not explode at depth.

Here is the practical version, which is a few lines on top of a standard transformer block:

```python
import math

def deepnorm_init(module, n_layers):
    """Scale the FFN, value, and output projections at init by (2N)^(-1/2)."""
    gain = (2 * n_layers) ** -0.5
    for name in ("ffn_in", "ffn_out", "v_proj", "out_proj"):
        torch.nn.init.xavier_normal_(getattr(module, name).weight, gain=gain)
        if getattr(module, name).bias is not None:
            torch.nn.init.zeros_(getattr(module, name).bias)   # all biases zero

def deepnorm_residual(x, sublayer_out, n_layers):
    """Post-LN with an up-weighted skip connection."""
    alpha = (2 * n_layers) ** 0.5
    return layer_norm(alpha * x + sublayer_out)
```

Notice the two halves work together: the `α`-scaled skip in the forward residual and the `(2N)^−½` init gain are a matched pair, derived so that the expected update to each layer's output is bounded independent of depth. Use one without the other and you lose the guarantee. This is why "DeepNorm" is a specific recipe and not just "Post-LN with a constant" — the constant on the residual and the constant on the init have to agree.

The transferable lesson is not "always use DeepNorm." It is: **normalization placement is a load-bearing decision at scale, and you cannot extrapolate from small-model behavior.** If the GLM team had tuned norm placement on a 1B model and frozen the choice, they would have shipped a divergent 130B. They re-ran the bake-off at scale, and it changed the answer.

### Embedding gradient shrink: the spike that predicts the future

The second failure mode is subtler and, to my mind, the most useful single idea in the report. Early in training, the GLM team observed that the gradient norm of the **embedding layer** ran orders of magnitude larger than every other layer — and, critically, that these embedding-gradient spikes *preceded* loss collapses by several steps.

![A graph of the FP16 stability stack: two failure modes (embedding spikes, attention overflow) each routed to a specific fix, converging on a stable run](/imgs/blogs/engineering-glm-130b-3.png)

The graph above traces both failure modes and their fixes. Follow the top path: `FP16 training → embedding gradient spikes → {detected by gradient-norm monitoring, mitigated by embedding-gradient-shrink} → stable`. The fix is one line:

```python
## Embedding Gradient Shrink (EGS): route most of the embedding's forward
## signal through a detached (no-grad) copy, so only `alpha` of the gradient
## flows back into the embedding table.
alpha = 0.1
word_embedding = word_embedding * alpha + word_embedding.detach() * (1 - alpha)
```

This expression is an identity in the forward pass — `α·e + (1−α)·e = e` — but because `.detach()` blocks gradient flow through the second term, the backward pass scales the embedding's gradient by `α`. At `α = 0.1`, only 10% of the gradient reaches the embedding table, which the report says "wipes out most spikes" with negligible runtime cost. Why does shrinking *just* the embedding gradient help? Because the embedding is where the spike originates — token-frequency imbalance and the discrete lookup make its gradient unusually heavy-tailed early in training, before the rest of the network has settled. Damp it there, at the source, and the spike never propagates into a collapse.

To see why the embedding is special, think about what its gradient looks like early in training. The embedding table has one row per vocabulary token, and a given token's row only receives gradient on the steps where that token *appears* in the batch. Frequent tokens (common words, punctuation) get updated almost every step; rare tokens get updated occasionally with a large accumulated signal. This produces a heavy-tailed gradient distribution — most rows quiet, a few rows occasionally enormous — that is much spikier than the gradient of a dense weight matrix where every element is touched by every example. Early in training, before the embeddings have organized themselves, those occasional enormous updates are exactly the spikes that precede collapse. Shrinking the embedding gradient by 10× tames the tail without meaningfully slowing the *frequent* tokens (whose many small updates still accumulate). It's a targeted dose of stability aimed precisely at the layer whose gradient statistics are pathological. A dense hidden layer wouldn't benefit from the same treatment — its gradient isn't heavy-tailed in the same way — which is why the fix is *embedding* gradient shrink and not "shrink everything."

But the deeper contribution is the *diagnostic*, not the fix. The embedding gradient norm is a **leading indicator** of divergence — it rises several steps before the loss does. That turns a postmortem statistic into a smoke alarm:

```python
## Use embedding gradient norm as an early-warning signal.
emb_grad_norm = word_embedding.grad.norm().item()
if emb_grad_norm > EARLY_WARNING_THRESHOLD:
    # intervene BEFORE the loss collapses: lower LR, skip the batch,
    # or roll back a few steps — you have a few steps of warning.
    logger.warning(f"embedding grad norm {emb_grad_norm:.1f} — divergence risk")
```

I have personally watched runs die from exactly this failure mode while staring at a loss curve that looked fine right up until it wasn't. The GLM-130B report is the document that tells you *where to look* so the loss curve isn't the first thing that tells you you're already dead. Instrument the embedding gradient norm, set a threshold, and you buy yourself the few steps of warning that let you intervene.

### Why these two fixes, together

It's worth being explicit that DeepNorm and embedding-gradient-shrink solve *different* problems and you need both. DeepNorm bounds the *forward* value scale of deep layers so activations don't overflow. EGS damps the *backward* gradient spike at the embedding so a heavy-tailed update doesn't blow up the optimizer state. One is a forward-pass fix, one is a backward-pass fix, and the FP16 regime (next section) is what makes both necessary in the first place. The report's stability stack is not one trick; it is a small, carefully chosen set, each member addressing a distinct way the run can die.

There's a useful way to think about this in terms of *where in the computational graph* each failure lives. A transformer training step is a forward pass (activations flow up), a backward pass (gradients flow down), and an optimizer step (states update). DeepNorm guards the forward pass — it keeps activations from overflowing as they accumulate through depth. EGS guards the backward pass — it keeps the embedding's gradient from spiking. FP32 softmax (next section) guards a specific *operation* in the forward pass — the one reduction whose dynamic range exceeds FP16. And FP32 optimizer states guard the *update* — they keep momentum and variance from accumulating rounding error over hundreds of thousands of steps. Four guards, four distinct locations in the step. When you're debugging your own divergence, this taxonomy is the first thing to reach for: *is the activation overflowing (forward), is a gradient spiking (backward), or is the optimizer state drifting (update)?* The answer tells you which of the four guards you're missing.

The honest caveat is that this stack was tuned for *this* model on *this* hardware. The exact `α = 0.1` for embedding-gradient-shrink, the `(2N)^½` DeepNorm constant — these are not universal physical constants, they're values the team found worked at 130B on A100s in FP16. Port the *structure* (guard each location) more confidently than the *constants* (re-tune them for your scale and precision). That said, the structure has held up remarkably well; the DeepNorm idea in particular shows up, evolved into QK-Norm, all the way up in [GLM-4.5](/blog/machine-learning/large-language-model/glm-lineage-frontier-llm-technique).

## 3. Precision: the FP16 bet

> **Senior rule of thumb:** the precision you train in is also a decision about *who gets to use the result*. GLM-130B's most contrarian choice was made on accessibility grounds, and it forced the team to invent fixes the field then got to keep.

By 2022, the safe choice for large-model training was BF16 — bfloat16, with the same exponent range as FP32, which makes the attention-logit-overflow problem largely disappear. GLM-130B deliberately chose **FP16** instead, and then had to solve every problem BF16 would have papered over.

![A before-after comparing BF16 (rejected) with FP16 (chosen), showing the memory and GPU-support tradeoffs and the fixes FP16 required](/imgs/blogs/engineering-glm-130b-4.png)

The before-after above lays out the bet. The case *for* BF16 is real: a wide exponent means attention scores don't overflow. But the GLM team weighed two costs against it. First, BF16 used roughly **15% more runtime GPU memory** in their setup than FP16 — at 130B, 15% of memory is the difference between fitting and not fitting on a given node count. Second, and more importantly for the project's goals, **BF16 was not supported on V100-class and older GPUs.** GLM-130B was meant to be an *open, accessible* model; choosing a precision that locked out everyone without an A100 would have defeated the purpose. So they chose FP16 — and inherited its central problem.

### The attention-overflow problem, and the FP32 patch

FP16 has a narrow exponent range, and as a transformer scales, the pre-softmax attention logits grow until they overflow that range, producing `inf`/`NaN` and killing the run. The standard 2022 alternatives were ugly: OPT-175B's team reportedly babysat the learning rate by hand through instability, and BLOOM used BF16 plus an embedding LayerNorm — which the GLM team found *hurt* zero-shot performance. GLM-130B's answer was cleaner: compute the **attention softmax in FP32** while keeping the rest of the forward pass in FP16.

```python
## Attention scores computed and softmaxed in FP32, then cast back to FP16.
## Only the numerically dangerous reduction runs in high precision.
scores = (q @ k.transpose(-1, -2)) * scale          # FP16 matmul
scores = scores.float()                             # up-cast for the reduction
attn = torch.softmax(scores, dim=-1)                # FP32 softmax — no overflow
out = (attn.to(q.dtype) @ v)                        # back to FP16 for the matmul
```

The cost is small: only the softmax reduction runs in FP32, not the expensive matmuls. The benefit is that the one numerically dangerous operation — the exponentiation and normalization over potentially-large logits — happens with FP32's headroom. Combined with DeepNorm bounding the logits' magnitude in the first place, this is what let GLM-130B stay in FP16 without the attention overflow that FP16 "should" have caused. The optimizer states are also kept in FP32 (standard mixed-precision practice), so the master weights accumulate without FP16 rounding error.

### The mixed-precision ledger

Putting the precision choices in one place, because the *pattern* is the lesson — **spend high precision only where the numerics are dangerous:**

| Component | Precision | Why |
| --- | --- | --- |
| Forward/backward matmuls | FP16 | cheap, memory-light, runs on V100 |
| Attention softmax | FP32 | the overflow-prone reduction |
| Optimizer states (momentum, variance) | FP32 | avoid accumulation error |
| Master weights | FP32 | avoid FP16 rounding on updates |

This per-component precision discipline is the same instinct that, three years later, shows up in GLM-4.5's RL stack as "BF16 training, FP8 rollout" — covered in the [lineage survey](/blog/machine-learning/large-language-model/glm-lineage-frontier-llm-technique). The venue changes; the principle ("precision is per-role, not global") does not.

### Dynamic loss scaling, the FP16 tax you can't skip

There's one more FP16-specific mechanism that the report assumes rather than belabors, and it's worth making explicit because it's the part most likely to bite a team trying to reproduce FP16 training: **dynamic loss scaling.** FP16 has not just a narrow *exponent* range (the overflow problem) but also a narrow *underflow* range — small gradients round to zero. For a 130B model, many gradients are small, and if they underflow to zero the corresponding weights simply stop learning. The fix is to multiply the loss by a large scale factor `S` before the backward pass (which scales all gradients up, away from the underflow floor), then divide the gradients by `S` before the optimizer step. The "dynamic" part is that you raise `S` when training is stable and halve it whenever a gradient overflows to `inf`:

```python
## Dynamic loss scaling: keep small FP16 gradients away from underflow.
scaled_loss = loss * scale
scaled_loss.backward()
if any_grad_is_inf_or_nan(model):
    scale = max(scale / 2, MIN_SCALE)   # overflow: back off, skip this step
else:
    grads = [p.grad / scale for p in model.parameters()]
    optimizer.step(grads)
    steps_since_overflow += 1
    if steps_since_overflow % 2000 == 0:
        scale = min(scale * 2, MAX_SCALE)  # stable for a while: push scale up
```

The interaction with GLM-130B's other stability machinery is subtle and important: when an embedding-gradient spike or an attention overflow produces an `inf`, the loss scaler catches it, halves the scale, and *skips the step*. So the loss scaler is itself a third line of defense — it converts a would-be-fatal overflow into a single skipped step. But it's a blunt one: if overflows happen *constantly* (because the run is genuinely unstable), the scaler thrashes its scale down and down and learning stalls. That's why DeepNorm and EGS matter even with a loss scaler present — they keep overflows *rare* enough that the scaler's occasional step-skip is a minor tax rather than a death spiral. The three mechanisms are complementary: DeepNorm and EGS make overflows rare; the loss scaler cleanly absorbs the few that remain.

### What the alternatives did, and why GLM avoided them

It sharpens the FP16 story to contrast it with how the other 2022 open giants handled the same numerics. OPT-175B, by the team's own released logbook, was babysat through instability largely by hand — lowering the learning rate, restarting from checkpoints, manually skipping bad data. That works, but it requires a human in the loop and doesn't generalize. BLOOM-176B used BF16 (avoiding the overflow problem at the memory and hardware cost discussed above) plus an extra LayerNorm right after the embedding — and the GLM team specifically reports that this embedding-LayerNorm approach *hurt zero-shot performance*, so they declined to use it. GLM-130B's contribution is a stability stack that is (a) automated (no manual babysitting), (b) FP16 (no BF16 memory/hardware cost), and (c) doesn't degrade the model (unlike the embedding-LayerNorm). That's the trifecta the DeepNorm + EGS + FP32-softmax + loss-scaler combination buys. For the modern mixed-precision landscape, the blog's [quantization tradeoffs](/blog/machine-learning/mlops/quantization-int8-fp16-int4-edge-tradeoffs) deep-dive is the companion.

## 4. Parallelism: spreading 130B over 768 GPUs

> **Senior rule of thumb:** at 100B scale the bottleneck is the bytes you move between GPUs, not the FLOPs you compute. Your parallelism topology is a bandwidth-budgeting decision.

A 130B model in FP16 is ~260 GB of weights, and the FP32 optimizer states add roughly another terabyte — vastly past a single 40 GB A100. GLM-130B composed three parallelism axes to fit and feed the model.

![A tree showing how GLM-130B composes 4-way tensor, 8-way pipeline, and 24-way data parallelism over 768 A100 GPUs](/imgs/blogs/engineering-glm-130b-5.png)

The tree above decomposes the topology. Each axis solves a different problem:

- **Tensor parallelism, 4-way.** Each layer's matmuls are sharded across 4 GPUs *within a node* (connected by fast NVLink), because tensor parallelism is communication-heavy and you want it on the fastest interconnect you have.
- **Pipeline parallelism, 8-way.** The 70 layers are split into 8 stages *across nodes*. This is where `70 = 9×8 − 2` comes from: 9 layers per stage, minus 2 to balance the embedding-heavy ends. Pipeline parallelism tolerates slower interconnect because it communicates only at stage boundaries.
- **Data parallelism, 24-way.** The whole `4×8 = 32`-GPU model is then replicated 24 times to consume the global batch of 4,224, giving `32 × 24 = 768` A100-40G GPUs total (96 DGX-A100 nodes).

The numbers that matter for calibration:

| Metric | Value |
| --- | --- |
| Hardware | 96× DGX-A100 (8×40 GB) = 768 A100-40G |
| Parallelism | 4 TP × 8 PP × 24 DP |
| Global batch | 4,224 |
| Hardware FLOP utilization (HFU) | 43.3% |
| Model FLOP utilization (MFU) | 32.5% |
| Wall-clock | ~60 days (May 6 – Jul 3, 2022) |
| Optimizer | AdamW, LR 1e-7 → 8e-5, cosine 10× decay |

A 32.5% model-FLOP utilization is the sobering number. It means that for every FLOP of useful math, roughly two are lost to communication, pipeline bubbles, and recomputation. That is not incompetence; it is the *physics* of training a model that doesn't fit on one device. The gap between HFU (43.3%) and MFU (32.5%) is the activation-recomputation overhead — the FLOPs you redo to save memory. For the modern version of these concerns, the blog's [multi-node training recipe](/blog/machine-learning/mlops/multi-node-llm-training-recipe-troubleshooting) walks the failure modes you hit in practice.

### Where the 67% of FLOPs goes

It's worth dwelling on that 32.5% MFU, because understanding where the other two-thirds of your compute goes is the difference between accepting a number and improving it. Three sinks dominate:

- **Pipeline bubbles.** With 8 pipeline stages, the first stage starts computing while stages 2–8 sit idle waiting for its output to arrive, and at the end of a batch the last stages drain while the first sit idle. This "bubble" of idle time scales with the number of stages relative to the number of micro-batches. The standard mitigation — which GLM-130B uses — is to split each batch into many micro-batches so the pipeline stays full most of the time, but the bubble never fully disappears. The `70 = 9×8 − 2` layer balancing is partly about minimizing this: if one stage held 12 layers and another held 6, the 12-layer stage would be the bottleneck and everyone would wait on it. Equal layers per stage means equal stage latency means a tighter pipeline.
- **Activation recomputation.** To fit 130B's activations in 40 GB GPUs, you can't store every intermediate tensor for the backward pass; you re-compute them. That's the gap between HFU (43.3%, counting recomputation as "useful" hardware work) and MFU (32.5%, counting only the FLOPs that advance training). Roughly a quarter of the "useful" hardware FLOPs are redone work — the memory-versus-compute trade that every large run makes.
- **Communication.** Tensor parallelism all-reduces activations across 4 GPUs *every layer*, which is why it lives on the fast NVLink within a node. Pipeline parallelism sends activations between stages across the slower inter-node fabric, but only at stage boundaries. Data parallelism all-reduces gradients once per step across all 24 replicas. Each of these is bytes-on-the-wire that isn't math, and at 130B the bytes are enormous.

The reason the topology is `4 TP × 8 PP × 24 DP` and not, say, `8 TP × 4 PP` is a direct consequence of this analysis: tensor parallelism's per-layer all-reduce is the most bandwidth-hungry, so you keep its degree small (4) and confined to the single fastest interconnect (intra-node NVLink). Pipeline parallelism's communication is cheap per step, so you can afford 8-way across nodes. Data parallelism's gradient all-reduce is once per step and overlaps with the backward pass, so it scales out widely (24). Every parallelism degree in the topology is a bandwidth decision, which is the whole point of the senior rule of thumb: at this scale you are budgeting bytes, not FLOPs.

### A note on ZeRO and what GLM-130B did not need

A reader steeped in modern training will ask: where is ZeRO / FSDP? The answer is that GLM-130B's tensor-plus-pipeline approach is an alternative to fully-sharded data parallelism, not a complement, and the team chose the 3D-parallelism route (tensor × pipeline × data) that Megatron-style training popularized. The FP32 optimizer states — the terabyte that doesn't fit on one GPU — are sharded across the data-parallel replicas in the standard way, so each replica holds only its slice of the optimizer state. The design point worth remembering is that there are multiple valid ways to fit a model that exceeds device memory, and the right one depends on your interconnect topology: 3D parallelism shines when you have fast intra-node links (for tensor parallelism) and a reasonable inter-node fabric (for pipeline and data), which is exactly the DGX-A100 cluster GLM-130B trained on.

## 5. Quantization: the closet supercomputer

> **Senior rule of thumb:** if you know at design time that you'll quantize, you can *shape the model* to quantize well — quantization-friendliness is a training-time property, not a deployment afterthought.

This is the part of the report that reads like a magic trick, and the reveal is a single property of GLM's weights.

Set the scene with the memory arithmetic, because it's what makes the trick necessary. A 130B model in FP16 is ~260 GB just for the weights. To run inference you need those weights plus the KV cache plus activation working memory. On 40 GB A100s that's a minimum of 8 GPUs, and those are data-center cards. The aspiration was to run on *consumer* hardware — RTX 3090s with 24 GB, the card a graduate student or a small startup actually owns. Four 3090s give you 96 GB total, which cannot hold 260 GB of FP16 weights. INT4 changes the arithmetic: at 4 bits the weights are ~65 GB, which *does* fit in 96 GB with room for the cache and activations. So the quantization isn't a nice-to-have optimization — it's the single thing standing between "needs a data-center node" and "runs on a workstation." That's why the report treats it as a design goal rather than a deployment afterthought, and why the narrow-weight-distribution property that makes 4-bit work is presented as a headline finding rather than an implementation note.

![A before-after explaining why GLM's narrow weight distribution tolerates INT4 while GPT-style weights are stuck at INT8](/imgs/blogs/engineering-glm-130b-6.png)

The before-after above is the whole story. GPT-style models have a relatively *wide* weight distribution, and 4-bit round-to-nearest rounding loses too much of it — they're typically stuck at INT8. GLM-130B's weights have a **much narrower distribution**, and the report states this plainly:

> GLMs tend to have much narrower weight distributions than similar-sized GPTs, so INT4 suffices where GPT-style models are limited to INT8.

Narrow distributions quantize cleanly because the quantization grid spends its 16 levels (4 bits) over a smaller range, so each level is finer. The result: GLM-130B became the **first 100B-scale model to ship INT4** with essentially no quality loss — LAMBADA fell 0.74% and MMLU actually *rose* 0.05%, both within noise.

### How the INT4 actually works

The quantization itself is almost anticlimactically simple, which is the point.

![A pipeline of INT4 weight-only quantization: FP16 weights round to INT4 with no calibration, then dequantize to FP16 at matmul time](/imgs/blogs/engineering-glm-130b-7.png)

The pipeline above is the full flow. It is **weight-only** quantization via **round-to-nearest (RTN)** with **no calibration data** — you don't need a representative dataset to tune the quantization, you just round. "Weight-only" is the crucial qualifier and it's worth understanding why it's the right call. In a transformer matmul, you multiply weights by activations. You *could* quantize both (so the matmul runs in integer arithmetic, which is faster), but activations are the problem children — they include those large, dynamic attention values, and quantizing them aggressively wrecks quality. Weights, by contrast, are static after training and (for GLM) narrowly distributed. So weight-only quantization takes the easy win — shrink the static, well-behaved weights to 4 bits for the *memory* savings — without touching the dangerous activations, which stay in FP16. You don't get the integer-matmul *speed* win, but you do get the *fit-on-small-GPUs* win, and for the goal here (run a 130B on consumer cards) memory was the binding constraint, not arithmetic throughput. It's the same "quantize the bounded thing, not the unbounded thing" logic as the FP32-softmax decision, applied at deployment instead of training. The weights are stored in INT4 (~65 GB instead of ~260 GB), and at matmul time they are dequantized back to FP16; the *activations* stay FP16 throughout. So the model computes in FP16 but *stores* in INT4, and the memory savings are what let it fit on small GPUs.

```python
## Per-channel INT4 round-to-nearest, weight-only, no calibration.
def quantize_int4_rtn(W):                      # W: [out, in] FP16 weights
    scale = W.abs().amax(dim=1, keepdim=True) / 7.0   # 4-bit signed: [-8, 7]
    q = torch.clamp(torch.round(W / scale), -8, 7).to(torch.int8)  # store as int4
    return q, scale

def dequantize(q, scale):
    return q.to(torch.float16) * scale         # back to FP16 at load/matmul time
```

The payoff is the headline: a 130B model that runs inference on **4× RTX 3090 (24 GB)** or **8× RTX 2080 Ti (11 GB)** — INT8 and full precision are also supported (full precision needs 8× A100-40G or 8× V100-32G). Built on SwissArmyTransformer with FasterTransformer kernels, inference was accelerated up to ~2.5×. A frontier-class model you could, in principle, host under a desk.

### Why per-channel, and why no calibration

Two details in that quantization code carry more weight than they look. The first is **per-channel** scaling: the `scale` is computed per output row of the weight matrix (`dim=1` reduction), not as a single number for the whole tensor. This matters because different output channels of a layer can have very different weight magnitudes; a single per-tensor scale would be dominated by the largest channel and would waste precision on all the smaller ones. Per-channel scaling gives each row its own quantization grid sized to its own range — a small cost in stored scale factors (one FP16 number per row) for a large gain in fidelity. It's the cheapest possible step up from naive per-tensor quantization and it's usually enough.

The second is **no calibration**. Modern quantizers like GPTQ and AWQ run a calibration pass: they feed a representative dataset through the model and use the resulting activation statistics to decide *which* weights to preserve most carefully (AWQ) or to solve a layer-wise reconstruction problem (GPTQ). These methods squeeze more quality out of aggressive quantization — but they need a calibration set, a tuning pass, and care that the calibration data matches deployment. GLM-130B needed *none* of that, because its weight distribution was narrow enough that dumb round-to-nearest already lands within noise. That's the real headline: **the model was quantization-friendly enough that the simplest quantizer sufficed.** When you can get away with RTN, you avoid an entire category of calibration-pipeline complexity and reproducibility risk. The senior move is to try RTN *first*, measure the quality drop, and only escalate to GPTQ/AWQ if the drop is unacceptable — many teams reach for the complex quantizer reflexively when their model would have quantized fine with rounding.

### The weight-distribution claim, examined

The load-bearing empirical claim — "GLMs have narrower weight distributions than similar-sized GPTs" — deserves a skeptical look, because the whole INT4 story rests on it. Why *would* GLM's weights be narrower? The report doesn't fully resolve the mechanism, but the plausible contributors are exactly the architecture choices from the stability section: DeepNorm bounds the value scale of every layer, which bounds the activations, which (through training) tends to bound the weights that produce them; and the careful initialization (the `(2N)^−½` gain) starts the weights small and the bounded dynamics keep them from spreading. In other words, *the same stability engineering that kept the FP16 run alive may also be what made the model quantize well.* That's a satisfying connection if it holds: it would mean the stability stack paid off twice — once at training time (no divergence) and once at deployment time (INT4 works). Even if the causal story is incomplete, the *practical* takeaway stands: weight distribution is a measurable property you should check early if you intend to quantize, and architectural choices upstream influence it. For the broader picture of where INT4, INT8, and FP8 each win today, and how the frontier has moved past 4 bits, see the blog's [quantization in LLM](/blog/machine-learning/large-language-model/quantization-in-llm) and [past the 4-bit wall](/blog/machine-learning/large-language-model/past-4-bit-wall-frontier-llm-quantization) deep-dives.

## 6. The data, and the instruction seasoning

The corpus is 400B tokens, balanced roughly 200B English and 200B Chinese — English from the Pile, Chinese from WudaoCorpora plus ~250 GB of crawled web text. The pipeline is the canonical one — exact and fuzzy deduplication, then quality filtering, then tokenization — but one detail is worth lifting because it predates a trend.

GLM-130B mixes **Multi-task Instruction Pretraining (MIP)** *into* pretraining: 5% of tokens come from 74 prompted datasets. This is "season the pretraining with a little instruction data," done in 2022, before instruction tuning was a standard post-training stage. The effect is a measurable bump in zero-shot ability without a separate alignment phase. The reusable idea: a small instruction fraction mixed into pretraining is cheap and helps, and you don't have to wait for post-training to start teaching the model to follow prompts.

The bilingual balance is itself an engineering decision with downstream consequences. A 50/50 English/Chinese split is not the "natural" ratio you'd get from scraping the web (English dominates available high-quality text), so achieving it required deliberately *up-sampling* Chinese and *capping* English. That choice is why GLM-130B dominates Chinese benchmarks while only matching the English-only giants on English — you get out, on each language, roughly what you put in. The lesson generalizes to any multilingual or multi-domain training: the corpus ratio is a lever you set, and the model's capability profile will mirror it. If you want a model strong in a low-resource language, you pay for it in the mixing ratio, not in hope.

The deduplication deserves a mention because it's the unglamorous step that quietly determines data quality. GLM-130B runs *both* exact and fuzzy (near-duplicate) deduplication. Exact dedup removes verbatim copies — the same document crawled twice. Fuzzy dedup removes near-copies — the same article with a different header, a forum post quoted across threads, boilerplate that recurs across a domain. Skipping fuzzy dedup is a classic way to silently waste training compute: the model sees the same near-content many times, over-fits the duplicated regions, and you pay full FLOPs to memorize boilerplate. The order matters too — dedup *before* quality filtering, so the filter operates on a deduplicated set and its statistics aren't skewed by repeated documents. This pipeline shape (dedup → filter → tokenize) is the canonical one the whole lineage carries forward, scaling to the 23-trillion-token corpus of GLM-4.5.

## 7. The training run as a case study

Everything above is mechanism; this section is what it looked like in practice, because the report is candid about the run itself.

![A timeline of the 60-day GLM-130B run, with the early loss spikes and the embedding-gradient-shrink intervention that kept it alive](/imgs/blogs/engineering-glm-130b-8.png)

The timeline above traces the two months. Training started May 6, 2022, on 96 DGX-A100 nodes. The early phase is where the danger lived: embedding-gradient spikes preceded loss collapses, and the team's response — `α = 0.1` embedding-gradient-shrink plus gradient-norm monitoring as an early-warning signal — is what carried the run through. After the early turbulence the run settled into a steady state at 43.3% HFU / 32.5% MFU and ran to completion on July 3, having consumed 400B tokens. Sixty days, one model, on hardware the team chose to be accessible rather than maximal.

The thing to internalize from the timeline is *where* the risk concentrates. It is not uniform across the run; it is front-loaded. The early steps, before the network's statistics have settled, are when the embedding gradient is heaviest-tailed and the value scales are least controlled. This is why the fixes target early training specifically — DeepNorm's "small gradient norm that does not spike in early training," EGS damping the early embedding spikes. If you survive the first few thousand steps, you have mostly survived. Budget your monitoring attention accordingly: watch the dashboards hardest at the start.

### Checkpointing and the cost of a restart

A two-month run on 768 GPUs is also a study in *failure recovery*, because over sixty days hardware will fail — a GPU throws an ECC error, a node drops off the fabric, a NCCL all-reduce hangs. The defense is frequent checkpointing, and the engineering tension is the checkpoint *interval*. Checkpoint too rarely and a crash costs you hours of recomputation back to the last save; checkpoint too often and the I/O of writing a terabyte-plus of FP32 optimizer state stalls training. The sweet spot is set by the expected time between failures: if a 768-GPU cluster loses a node every few hours on average, you want checkpoints frequent enough that a crash costs minutes, not hours. This is why large runs report wall-clock in *months* even when the raw compute would suggest weeks — the gap is restarts, checkpoint I/O, and the occasional manual intervention when a divergence-warning fires.

The interaction with the stability stack is worth naming: gradient-norm monitoring isn't only an early-warning for *divergence*, it's an input to the *restart decision*. When the embedding gradient norm crosses the threshold, you have a choice — let the loss scaler skip the step and hope, lower the learning rate, or roll back to the last checkpoint and resume past the bad data. Having the leading indicator means you can make that decision *deliberately* a few steps early, rather than discovering after the loss has already collapsed that you need to roll back twenty thousand steps. The economic value of the gradient-norm signal is measured in the GPU-days of recomputation it lets you avoid. On a 768-GPU run, a single avoided full-rollback is worth real money.

## 8. Results: did the engineering pay off?

The engineering only matters if the model is good, and on the 2022 open landscape it was.

![A matrix of GLM-130B versus GPT-3, OPT, and BLOOM across LAMBADA, MMLU, BIG-bench, and Chinese benchmarks](/imgs/blogs/engineering-glm-130b-9.png)

The matrix above is the scorecard against the contemporaneous open giants. GLM-130B leads on zero-shot **LAMBADA (80.2%** vs GPT-3's 76.2, OPT's 75.2, BLOOM's 67.2), on **MMLU 5-shot (44.8%** vs GPT-3's 43.9, BLOOM's 32.1), and posts a strong **BIG-bench-lite (13.31** vs GPT-3's 4.35). Where it genuinely dominates is Chinese: it beats ERNIE 3.0 Titan (a 260B model) by **+24.26% average on CLUE** and +12.75% on FewCLUE, because the English-only models simply weren't trained for it. The report is also honest about the limits: on *English* tasks specifically, GLM-130B shows no clear advantage over OPT-175B or BLOOM-176B. It is competitive, not dominant, in English — and dominant in Chinese, which is exactly what a bilingual model trained on a 50/50 split should be.

One more under-reported result: GLM-130B was *less biased and less toxic* than GPT-3 and OPT on the standard probes (CrowS-Pairs 65.8 vs GPT-3's 67.2; StereoSet ICAT 73.5 vs 60.8; lower RealToxicityPrompts toxicity), which the authors attribute to bilingual pretraining. Whether or not that mechanism is fully established, it's a reminder that data composition has downstream effects beyond raw capability.

### What "competitive with GPT-3" actually meant

It's worth being precise about the evaluation, because "competitive with GPT-3" is a phrase that hides a lot. The most credible third-party signal was HELM (Stanford's Holistic Evaluation of Language Models), which in late 2022 placed GLM-130B as a peer of GPT-3 davinci across a broad suite — not a cherry-picked subset. That breadth matters: a model can top one benchmark by overfitting to its format, but matching a strong baseline *across* a holistic suite is harder to game. The LAMBADA result (80.2% zero-shot) is the single cleanest data point, because LAMBADA — predicting the final word of a passage that requires the *whole* passage to disambiguate — directly rewards the bidirectional understanding that GLM's blank-infilling objective builds and that a pure causal model lacks. GLM-130B beating GPT-3, OPT, and BLOOM on LAMBADA isn't an accident of tuning; it's the objective showing up in exactly the place theory predicts it should.

The MMLU result (44.8% five-shot) is more of a "keeping pace" number than a dominating one — it's a knowledge-and-reasoning benchmark where raw scale and English data volume matter, and GLM-130B's even split with Chinese means it spent half its data budget on a language MMLU doesn't test. Read that way, matching GPT-3's MMLU *despite* spending half its tokens on Chinese is arguably the more impressive result: it's competitive on the English knowledge benchmark while *also* being the best Chinese model of its generation. You don't get both for free; the fact that GLM-130B got close to both is the dividend of the bilingual data choice plus an objective that extracts more signal per token.

The honest framing the report itself offers — no clear advantage over OPT-175B or BLOOM-176B on *English* specifically — is the kind of candor that makes the rest of the document trustworthy. A report that admits where it merely ties is a report you can believe when it claims where it wins. This is, in the end, why GLM-130B is worth reading three years after the model itself stopped being state of the art: it is a document you can *trust*, written by people who told you what diverged, what they rejected, and where they only tied. The model is a historical artifact now. The engineering judgment — and the habit of writing it down honestly — is not, and it is the part the rest of this series watches mature.

## War stories: ten engineering decisions worth stealing

Surveys flatten the texture. Here are the specific decisions, each a self-contained lesson, each the kind of thing you only learn by reading a report written by people who were in the room when the loss spiked.

### 1. The bake-off that had to be re-run at scale

The LayerNorm result — Pre-LN, Post-LN, Sandwich all diverging, only DeepNorm surviving — is expensive precisely because each data point is a multi-day run at 130B. The team could have tuned on a 1B proxy and frozen the choice; the proxy would have told them Pre-LN was fine. The lesson is uncomfortable: some decisions *cannot* be made on a small model, because the failure mode they guard against doesn't appear until scale. You have to spend the GPU-days to re-run the bake-off where it counts, and you have to be willing to have the answer change.

### 2. The α=0.1 that cost nothing

Embedding-gradient-shrink is the highest-leverage-per-line change in the report. One expression, an identity in the forward pass, scales the embedding gradient by 0.1 in the backward pass, and it "wipes out most spikes." The leverage comes from targeting the *source*: the team didn't clip all gradients or lower the global learning rate (blunt instruments that slow everything); they damped exactly the layer whose gradient was pathological. The general principle — *find the specific component generating the instability and fix it there, not globally* — is worth more than the specific trick. Contrast it with the reflexive fixes a stressed team reaches for at 3 a.m.: lower the global learning rate (slows the whole run to fix one layer), clip all gradients (distorts every update to tame one), restart from a checkpoint (loses progress, fixes nothing if the data is the problem). Those are sledgehammers. EGS is a scalpel — it knew *which* layer was sick and treated only that layer. The diagnostic work that lets you be surgical (here, noticing the embedding gradient was the outlier) is what separates a fix that costs nothing from a fix that costs throughput.

### 3. Gradient norm as a smoke alarm

The single most transferable monitoring idea in the report is that the embedding gradient norm rises *before* the loss collapses. Most teams watch the loss, which is a lagging indicator — by the time it spikes, you've already lost the steps. Watching the embedding gradient norm gives you a few steps of warning, which is enough to lower the LR, skip a bad batch, or roll back. Instrument the leading indicator, not just the lagging one. The general principle is one every reliability engineer knows in other domains: a good monitor measures the *cause* a little before the *symptom*, not the symptom itself. A disk-space alert that fires at 99% full is nearly useless; one that fires when the *fill rate* spikes gives you time to act. Loss is the 99%-full alert; embedding gradient norm is the fill-rate alert. The hard part is finding which upstream quantity leads your particular failure — for FP16 large-model training the GLM team found it's the embedding gradient, and they handed you that finding for free.

### 4. Choosing FP16 for who, not for what

The FP16-over-BF16 decision is the clearest example in the report of a constraint accepted for the *right reason*. The team didn't choose FP16 because it was technically superior — BF16 is easier. They chose it because BF16 would have locked out V100-class GPUs and cost 15% more memory, and an *open* model needs to run on the hardware open researchers have. That constraint forced them to invent the FP32-softmax + DeepNorm + EGS stack, which then became reusable knowledge. A constraint accepted for the right reason can be generative. This is a pattern worth internalizing beyond this one decision: the most productive constraints are the ones you choose for a principled reason and then refuse to relax, because they force you to solve the underlying problem instead of buying your way around it. A team with unlimited memory would have shrugged and used BF16, and the FP32-softmax + DeepNorm + EGS stack — knowledge the whole field now has — might never have been written down. Scarcity, accepted deliberately, is an engine of technique.

### 5. FP32 only where it's dangerous

The mixed-precision discipline — FP16 everywhere except the attention softmax, optimizer states, and master weights — is a model of surgical precision spending. The expensive matmuls stay cheap; only the numerically dangerous reduction pays for FP32. The anti-pattern is "train everything in FP32 to be safe," which doubles your memory and halves your throughput to guard against a problem that lives in one operation. Find the dangerous operation and pay only there. The way to find it is to ask, operation by operation, "what is the dynamic range of the numbers flowing through here?" Matmuls of bounded activations stay in range. The softmax — which exponentiates logits that can be large — does not. Accumulating a sum over hundreds of thousands of optimizer steps does not. Those are the operations that need headroom, and they're a small fraction of the compute. This same reasoning, applied at deployment time, is what makes INT4 *weight-only* quantization work: weights are bounded and quantize fine, but activations (which include those same large attention values) stay in FP16. Forward and backward, the principle is identical — *quantize the bounded things aggressively, keep the unbounded things in high precision.*

### 6. The layer count that betrays the pipeline

`70 = 9×8 − 2` is a small detail with a big lesson: at scale, your architecture numbers leak your systems constraints. The depth was chosen to balance an 8-stage pipeline, not for any modeling reason. When you see an odd layer count in a large-model report, look for the pipeline-stage count it's a near-multiple of — you'll usually find it, and it tells you the systems team had a seat at the architecture table. That collaboration is healthy; the alternative is a "clean" layer count that creates a pipeline imbalance and wastes 10% of your GPUs. The broader lesson is that at scale, the org chart leaks into the model. A model designed by researchers who never talk to the infra team will have round, "elegant" numbers that the pipeline chokes on; a model designed by infra people who never talk to research will be efficient and underpowered. The good large-model reports — and GLM-130B is one — are visibly the output of those two groups in the same room, and you can read the collaboration in the artifacts: a `9×8−2` layer count is a fingerprint of the systems team having had a say.

### 7. Designing the model to quantize

The narrow-weight-distribution insight is the deepest idea in the quantization section. GLM quantizes to INT4 *because of how it was trained*, not because of a clever post-hoc quantizer. The weight distribution is a property you can influence at training time — through the objective, the normalization, the initialization — and GLM's happened to be quantization-friendly. Treating "will this quantize well?" as a design question rather than a deployment question is what bought single-server inference. If you know your model will be quantized, measure its weight distribution early and treat narrowing it as a goal.

### 8. RTN with no calibration

The INT4 quantization needing *no calibration data* is an operational win that's easy to undervalue. Calibration-based quantizers (GPTQ, AWQ, and the like) need a representative dataset and a tuning pass; RTN just rounds. For a model with a narrow enough weight distribution, the simplest possible quantizer suffices, which means no calibration pipeline to build, no calibration-set bias to worry about, and trivial reproducibility. Reach for the simplest quantizer first and only escalate if the quality loss demands it.

### 9. Seasoning pretraining with instructions

The 5% Multi-task Instruction Pretraining is a 2022 move that anticipates a trend. Rather than treating instruction-following as a separate post-training phase, GLM-130B mixed a small fraction of prompted data into pretraining and got a zero-shot bump for nearly free. The reusable idea is that the boundary between "pretraining" and "instruction tuning" is softer than the standard pipeline implies — a little instruction data early is cheap insurance.

### 10. Publishing the negative results

The meta-decision that makes the whole report valuable is that the team *published what diverged*. The Pre-LN/Post-LN/Sandwich failures, the BLOOM-style embedding-norm that hurt zero-shot, the OPT-style manual LR babysitting they wanted to avoid — these negative results are the expensive knowledge, and most labs keep them private. A report that tells you what *not* to try is doing you a bigger favor than one that only shows the happy path.

### 11. Reserving vocabulary you don't use yet

A small foresight detail: the 150,000-token vocabulary reserves roughly 20,000 slots for *image* tokens the dense text model never used. The team built in headroom for the multimodal extension (the CogVLM line) they knew was coming, because expanding a tokenizer *after* training is painful — every reserved slot you didn't plan for means re-learning embeddings or awkward vocabulary surgery. The lesson is cheap and worth taking: if there's a real chance you'll add a modality or a special-token scheme later, reserve the vocabulary space at the start. Unused embedding rows cost almost nothing; a forced re-tokenization costs a retrain.

### 12. Owning the training stack

None of the unusual tricks in this report — embedding-gradient-shrink, the DeepNorm residual, the FP32-only-softmax — are things you can express as a config flag in someone else's framework. They required modifying the forward and backward passes directly, and GLM-130B could do that because it ran on the team's own library (SwissArmyTransformer), with FasterTransformer kernels for inference. There's a pattern across the field: the teams that ship genuinely novel stability or efficiency tricks almost always own their training stack, because the tricks live below the abstraction line that off-the-shelf frameworks expose. Owning the stack is expensive, but it's what buys you the freedom to implement an idea that doesn't fit anyone else's API. If your research depends on non-standard training internals, budget for owning the code that runs them.

## When to reach for the GLM-130B playbook

![A matrix of GLM-130B engineering techniques with the regime where each pays off and where it is overkill](/imgs/blogs/engineering-glm-130b-10.png)

The matrix above is the triage in one view: every technique in this article has a regime where it earns its complexity and a regime where it's overkill or actively wrong. The single most common mistake I see teams make with reports like this one is treating the techniques as universally good — bolting DeepNorm onto a 500M model, reaching for INT4 on a model whose weights are wide, building a 3D-parallelism harness for something that fits on two GPUs. Each of those is effort spent solving a problem you don't have. The skill is matching the technique to the regime, and the regime is usually set by two numbers: how big is the model, and how good is the hardware you're deploying to. Read the matrix as a function of those two and the rest of this section is the prose version.

**Reach for these techniques when:**

- **You're training above ~10B parameters on a fixed GPU budget.** The DeepNorm + embedding-gradient-shrink + FP32-softmax stack is battle-tested survival gear for exactly this regime. Steal the *gradient-norm monitoring* even if you train in BF16 — the early-warning discipline is precision-agnostic.
- **You need the result to run on modest hardware.** The FP16 bet and INT4 quantization are a coherent strategy for "frontier capability, accessible deployment." If your users have RTX 3090s rather than H100s, this is the playbook.
- **You know at design time that you'll quantize.** Shape the model for a narrow weight distribution and you may get INT4 nearly free. Measure the distribution early.
- **You're about to freeze a stability-critical choice on a small proxy.** Don't. Re-run the bake-off at (or near) your target scale for the choices — normalization, precision — that guard against scale-only failure modes.

**Skip or adapt when:**

- **You're below ~1B parameters.** Most of this machinery solves problems you don't have yet. Pre-LN will train fine; FP16 won't overflow; you don't need pipeline parallelism. The instruction-seasoning idea still applies; the stability stack doesn't.
- **You have unlimited H100s and BF16 is free.** The FP16 bet was made under accessibility and memory constraints. If those don't bind you, BF16 is the easier path and you should take it — but keep the gradient-norm monitoring.
- **Your model's weights aren't narrow.** INT4 RTN works *because* GLM's distribution is narrow. If yours isn't, you'll need calibration-based quantization (GPTQ/AWQ) or you'll be stuck at INT8 — measure before you assume.
- **You're chasing English-only benchmarks.** GLM-130B's edge was bilingual; on English alone it merely matched OPT/BLOOM. The engineering transfers; the specific capability advantage was a data-composition choice.
- **You're training a sparse MoE rather than a dense model.** Some of the dense-specific stability tuning (the exact DeepNorm constant, the embedding-gradient profile) shifts when most of your parameters are inactive per token. The *structure* (guard forward/backward/update separately) still applies, but MoE adds its own failure modes — router collapse, expert imbalance — that GLM-130B's dense report doesn't cover. For those, jump ahead in the series to the GLM-4.5 architecture article.

It's also worth saying what *doesn't* transfer and shouldn't be cargo-culted. The specific `α = 0.1` for embedding-gradient-shrink, the `8e-5` peak learning rate, the 400B token count, the 4×8×24 parallelism degrees — these are all fitted to GLM-130B's exact size, data, and cluster. Copy them onto a different model and you'll get a different (possibly worse) result. What you copy is the *shape* of the solution: monitor the leading indicator, guard each location in the training step, spend precision surgically, choose your precision for your users, and design the model for how you'll deploy it. Those are method; the constants are just one team's instantiation of the method.

There's a meta-point underneath the bullet lists, about *how to read a technical report at all*. Most readers mine reports for hyperparameters — the learning rate, the batch size, the layer count — and copy them. That's the least valuable thing in the document, because those numbers are tuned for a specific model on specific hardware and rarely transfer. The valuable content of GLM-130B is the *reasoning*: why FP16 over BF16 (accessibility), why DeepNorm over Pre-LN (it was the only one that didn't diverge), why per-channel RTN over GPTQ (the distribution was narrow enough), why gradient norm as a monitor (it leads the loss). Reasoning transfers; numbers don't. When you read a report like this, collect the *decisions and their justifications*, not the constants. The constants are a snapshot; the decisions are a method.

And the method here is unusually clean. GLM-130B is what it looks like when a team treats the *systems* problem as the *research* problem — when "will it diverge?" and "will it run on a 3090?" are first-class questions answered with the same rigor as "what's the architecture?". The result is a model that was competitive on capability and *radically* more accessible than its peers, and a report that hands you the negative results most labs would have buried. The next article in the series moves up the lineage to ChatGLM and GLM-4, where the hard problem shifts from "don't diverge" to "align the model and teach it to use tools." The failure modes change completely. But the habit that makes the GLM reports worth reading — write down what broke, and why the fix worked — stays constant, and it's the single most imitable thing about the whole lineage. Steal the habit before you steal any single technique; it is the one practice that keeps paying off no matter what you happen to be building, or at what scale you are building it.

## Further reading

- **Paper:** [GLM-130B: An Open Bilingual Pre-trained Model](https://arxiv.org/abs/2210.02414) (Zeng et al., ICLR 2023). [Code: THUDM/GLM-130B](https://github.com/THUDM/GLM-130B).
- **Series survey:** [The GLM Lineage: Five Years of Frontier-LLM Technique](/blog/machine-learning/large-language-model/glm-lineage-frontier-llm-technique).
- **Previous in series:** [Autoregressive Blank Infilling](/blog/machine-learning/large-language-model/autoregressive-blank-infilling-glm) — the objective GLM-130B scales.
- **Related on this blog:** [quantization tradeoffs (INT8/FP16/INT4)](/blog/machine-learning/mlops/quantization-int8-fp16-int4-edge-tradeoffs) and [multi-node LLM training](/blog/machine-learning/mlops/multi-node-llm-training-recipe-troubleshooting).

---
title: "Mixed Precision Training: Techniques, Tricks, and Troubleshooting for Production ML"
date: "2026-06-28"
publishDate: "2026-06-28"
description: "A complete guide to AMP, loss scaling, BF16 vs FP16, FP8, PyTorch patterns, and debugging every failure mode engineers hit in production training runs."
tags: ["mixed-precision", "training", "pytorch", "amp", "fp16", "bf16", "fp8", "gradient-scaling", "deep-learning", "mlops", "optimization"]
category: "machine-learning"
subcategory: "Training Techniques"
author: "Hiep Tran"
featured: true
readTime: 31
---

Training a 70B-parameter model in FP32 requires roughly 280 GB of GPU memory just for the weights alone — before activations, optimizer states, or gradients touch the budget. Mixed precision training cuts that in half while often *improving* throughput by 2–4x, but it introduces a class of subtle numerical failures that can silently corrupt your model, collapse your loss, or cause week-long training runs to diverge on step 47,000.

I have personally debugged all of those scenarios. This post is what I wish existed when I started: a ground-up explanation of how mixed precision actually works, why each design decision was made, every production tip and trick that matters, and a troubleshooting gallery covering the failures I see most often in real training jobs.

![The AMP training loop: forward and backward in FP16/BF16, optimizer step in FP32 master weights](/imgs/blogs/mixed-precision-training-1.webp)

The diagram above is the mental model: two precision worlds sharing one training loop. The forward and backward passes live in FP16 or BF16 — cheap, fast, memory-efficient. The optimizer step lives in FP32 — exact, stable, unhurried. A loss scaler bridges them by amplifying gradients before the backward and dividing afterward, keeping tiny values off the floor of FP16's representable range. Get this handshake right and you unlock near-FP32 accuracy at half the memory. Get it wrong and you spend days chasing NaNs that appear only on certain step counts, or a loss that trains fine for 10k steps and then quietly plateaus.

## 1. Numerical Formats Under the Hood

Before we touch any code, we need to understand exactly what FP32, FP16, and BF16 represent at the bit level. Most engineers treat them as "32-bit wide" vs "16-bit wide" and call it a day. That model is too coarse — the real distinction is *where* those bits go.

![FP32 vs FP16 vs BF16: bit-field layout showing exponent vs mantissa tradeoffs](/imgs/blogs/mixed-precision-training-2.webp)

A floating-point number is encoded as:

$$\text{value} = (-1)^s \times 2^{e - \text{bias}} \times (1 + \text{mantissa})$$

where $s$ is the sign bit, $e$ is the biased exponent, and the mantissa encodes the fractional part. The number of exponent bits determines the *dynamic range* — how far from zero the format can reach. The number of mantissa bits determines *relative precision* — how many significant figures you get within a given decade.

**FP32** uses 8 exponent bits + 23 mantissa bits. Dynamic range: roughly $10^{-38}$ to $10^{38}$. Relative precision: about 7 decimal digits. This is large enough to represent essentially every gradient, weight, and activation that appears in neural network training.

**FP16** was not designed for ML training — it was designed for graphics rendering, where you care about 16-bit colors and screen coordinates that fit comfortably in a narrow range. With only 5 exponent bits, its maximum representable value is 65504. Anything larger overflows to `Inf`. Anything smaller than about $6 \times 10^{-8}$ (the smallest subnormal) underflows to zero. Neural network gradients routinely live in the $10^{-9}$ to $10^{-4}$ range — squarely in FP16's danger zone.

**BF16** (Brain Float 16), introduced by Google and now standard on Ampere and later NVIDIA GPUs, makes a different tradeoff. It uses 8 exponent bits (same as FP32) and only 7 mantissa bits. Dynamic range is identical to FP32; precision is reduced from 7 to about 2 decimal digits. For training, this is the right tradeoff: gradients that would underflow in FP16 are representable in BF16, and the ~2-digit precision loss rarely affects model quality because gradient noise from stochastic mini-batches is already far larger than the rounding error.

**FP8** comes in two variants. E4M3 (4 exponent + 3 mantissa bits) maximizes precision in a narrow range — ideal for forward-pass activations, which tend to be well-behaved. E5M2 (5 exponent + 2 mantissa bits) maximizes range — better for backward-pass gradients, which can span a wider dynamic range. We will cover FP8 in depth in §6.

The practical upshot: **on Ampere+ GPUs, prefer BF16 as your default half-precision format**. FP16 is the right answer only when you're on older hardware (V100, T4) or when a specific layer needs the extra mantissa bits. On Hopper (H100+), FP8 is worth evaluating for large stable models.

## 2. The AMP Training Loop

Automatic Mixed Precision (AMP) does not simply cast everything to FP16. It maintains two copies of each parameter: a **FP16 weight copy** used in forward and backward passes, and a **FP32 master weight** updated by the optimizer. This split exists because matrix multiplications and convolutions run much faster in FP16 on tensor cores, while parameter updates — which accumulate tiny gradient signals across thousands of steps — need FP32 arithmetic to avoid drifting.

The loop runs as follows:

1. Cast the FP32 master weights to FP16 (or BF16) at the start of each step.
2. Run the forward pass entirely in FP16: activations, intermediate tensors, loss computation.
3. Multiply the scalar loss by a large **loss scale** $S$ (typically $2^{16}$). This magnifies the gradients before they get passed to the backward pass.
4. Run the backward pass. Gradients are computed in FP16 but scaled up by $S$, so what would have been a gradient of $10^{-7}$ is now $~6.5 \times 10^3$ — comfortably inside FP16's range.
5. **Unscale** the gradients by dividing by $S$, converting them to FP32.
6. Clip gradients (in FP32, against the master weight scale).
7. Check for `Inf` or `NaN`. If found, skip the optimizer step entirely.
8. Run the optimizer step on FP32 master weights with FP32 gradients.
9. Repeat.

The PyTorch implementation wraps this in two objects: `torch.autocast` (handles the cast in and out of lower precision) and `torch.cuda.amp.GradScaler` (handles the loss scaling, unscaling, inf-check, and conditional step). We will look at the exact code in §7.

### Why not just do everything in FP32?

Because tensor cores — the specialized matrix-multiply units that make modern GPUs fast — operate on FP16/BF16/FP8 inputs. On an A100, a GEMM in FP16 runs at 312 TFLOPS; the same GEMM in FP32 runs at 19.5 TFLOPS — a 16x difference. Memory bandwidth tells a similar story: FP16 weights are half the size, so twice as many fit in cache, and the model's memory footprint drops by roughly 40–50% in practice (the optimizer states in FP32 offset some of the saving).

### Why not just do everything in FP16?

Because accumulation error. Adam's first and second moments are running averages that accumulate over thousands of steps. In FP16, the precision floor means that gradient updates below ~0.001 of the current weight magnitude get rounded to zero and lost. Over many steps, this silent truncation shifts the parameter trajectory — the model trains, but to a slightly wrong place. The FP32 master weight eliminates this by keeping exact running sums.

## 3. Loss Scaling: Why and How

![Dynamic loss scaling pipeline: grow-on-health, halve-on-overflow](/imgs/blogs/mixed-precision-training-3.webp)

The FP16 format has a minimum positive normal value of $1.18 \times 10^{-38}$ for FP32, but only $6.1 \times 10^{-5}$ for FP16 normals and $5.96 \times 10^{-8}$ for subnormals. In practice, subnormals are extremely slow on most hardware (they drop out of hardware units into software emulation paths), so the effective floor for fast computation is closer to $6.1 \times 10^{-5}$.

Many weight gradients during training — especially early in training, after a learning rate warmup completes, or in layers with small initializations — live in the $10^{-7}$ to $10^{-5}$ range. Left unscaled, they underflow to zero in FP16. The optimizer then makes no update for those parameters, effectively freezing them. This often manifests as a loss that initially drops normally but then plateaus at a suboptimal value, which is one of the most insidious failure modes because it looks like a learning rate issue.

<figure class="blog-anim">
<svg viewBox="0 0 720 260" role="img" aria-label="Gradient magnitudes sweep from FP32 range into the FP16 underflow zone, then recover to the representable range after loss scaling is applied" style="width:100%;height:auto;max-width:820px">
<style>
.mp-axis{stroke:var(--border,#d1d5db);stroke-width:1.5;fill:none}
.mp-zone{opacity:.18}
.mp-fp32z{fill:#b2f2bb}
.mp-fp16z{fill:#ffc9c9}
.mp-lbl{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.mp-albl{font:700 13px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:start}
.mp-dot{fill:var(--accent,#6366f1)}
.mp-dot-dead{fill:#ffc9c9;stroke:#e03131;stroke-width:1.5}
.mp-dot-ok{fill:#b2f2bb;stroke:#2f9e44;stroke-width:1.5}
@keyframes mp-sweep{
  0%  {transform:translateX(0px);opacity:1}
  30% {transform:translateX(200px);opacity:1}
  45% {transform:translateX(340px);opacity:0.2}
  55% {transform:translateX(340px);opacity:0.2}
  70% {transform:translateX(340px);opacity:1}
  100%{transform:translateX(0px);opacity:1}
}
@keyframes mp-label-swap{
  0%,44%{opacity:1}45%,69%{opacity:0}70%,100%{opacity:1}
}
@keyframes mp-label-scale{
  0%,44%{opacity:0}45%,69%{opacity:1}70%,100%{opacity:0}
}
@keyframes mp-recovered{
  0%,69%{opacity:0}70%,100%{opacity:1}
}
.mp-g1{animation:mp-sweep 9s ease-in-out infinite}
.mp-g2{animation:mp-sweep 9s ease-in-out infinite;animation-delay:1.8s}
.mp-g3{animation:mp-sweep 9s ease-in-out infinite;animation-delay:3.6s}
.mp-lbl-noscale{animation:mp-label-swap 9s ease-in-out infinite}
.mp-lbl-scale{animation:mp-label-scale 9s ease-in-out infinite}
.mp-lbl-recover{animation:mp-recovered 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){
  .mp-g1,.mp-g2,.mp-g3{animation:none;transform:translateX(0)}
  .mp-lbl-noscale{animation:none;opacity:1}
  .mp-lbl-scale{animation:none;opacity:0}
  .mp-lbl-recover{animation:none;opacity:0}
}
</style>
<!-- number line background -->
<rect class="mp-zone mp-fp32z" x="60" y="80" width="200" height="60" rx="6"/>
<rect class="mp-zone mp-fp16z" x="340" y="80" width="120" height="60" rx="6"/>
<rect class="mp-zone mp-fp32z" x="500" y="80" width="160" height="60" rx="6"/>
<!-- axis -->
<line class="mp-axis" x1="40" y1="110" x2="680" y2="110"/>
<line class="mp-axis" x1="40" y1="105" x2="40" y2="115"/>
<line class="mp-axis" x1="260" y1="105" x2="260" y2="115"/>
<line class="mp-axis" x1="340" y1="105" x2="340" y2="115"/>
<line class="mp-axis" x1="460" y1="105" x2="460" y2="115"/>
<line class="mp-axis" x1="660" y1="105" x2="660" y2="115"/>
<!-- zone labels -->
<text class="mp-lbl" x="160" y="75">FP32 representable</text>
<text class="mp-lbl" x="400" y="75">FP16 underflow</text>
<text class="mp-lbl" x="580" y="75">FP16 representable</text>
<text class="mp-lbl" x="40"  y="130">1e-38</text>
<text class="mp-lbl" x="260" y="130">6e-8</text>
<text class="mp-lbl" x="340" y="130">↓ 0</text>
<text class="mp-lbl" x="460" y="130">6e-8</text>
<text class="mp-lbl" x="660" y="130">6.5e4</text>
<!-- animated gradient dots (without scaling → drift into underflow) -->
<circle class="mp-dot mp-g1" cx="80" cy="110" r="10"/>
<circle class="mp-dot mp-g2" cx="110" cy="110" r="10"/>
<circle class="mp-dot mp-g3" cx="140" cy="110" r="10"/>
<!-- status label: no scaling -->
<text class="mp-albl mp-lbl-noscale" x="60" y="200">Without loss scaling: gradients enter underflow zone → become 0 in FP16</text>
<!-- status label: with scaling applied -->
<text class="mp-albl mp-lbl-scale" x="60" y="200">Applying loss scale (×2¹⁶): gradients shift right into FP16 representable range</text>
<!-- status label: recovered -->
<text class="mp-albl mp-lbl-recover" x="60" y="200">Gradients recovered — unscale divides by 2¹⁶ after backward, restoring true values</text>
<!-- title row -->
<text style="font:700 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)" x="360" y="44" text-anchor="middle">Gradient magnitude on the number line</text>
</svg>
<figcaption>Without loss scaling, small FP32 gradients drift into the FP16 underflow zone and flush to zero. Multiplying the loss by a large scale (2¹⁶) shifts them right into the representable region; dividing after backward restores the true gradient values.</figcaption>
</figure>

### Static vs dynamic loss scaling

**Static loss scaling** uses a fixed multiplier chosen before training. It works if you know your gradient magnitudes in advance — you don't. Pick too small and gradients underflow; pick too large and gradients overflow to `Inf`. I have never used static scaling successfully outside a toy example.

**Dynamic loss scaling** solves this with a feedback controller. Start with a large scale (PyTorch defaults to $2^{16} = 65536$). After every successful step (no `Inf`/`NaN` in gradients), increment a patience counter. When the counter reaches `growth_interval` steps (default: 2000), multiply the scale by `growth_factor` (default: 2.0). If *any* step has `Inf` or `NaN` gradients, skip the optimizer update, divide the scale by `backoff_factor` (default: 0.5), and reset the patience counter to zero.

This self-tuning loop converges to the largest scale value that avoids overflow. In practice it stabilizes within the first few thousand steps and rarely changes after that unless the model enters a numerically difficult phase (e.g. a sudden spike in gradient norms due to a difficult batch).

### Choosing initial scale and patience parameters

The defaults are good for most models. The situations where I change them:

| Situation | Change |
|---|---|
| Frequent overflow early in training | Lower `init_scale` to $2^{14}$ or $2^{12}$ |
| Scale collapses to near 1 and stays there | Set `growth_interval` lower (e.g. 500) to recover faster |
| Very stable training, scale rarely changes | Increase `growth_interval` to 4000 to reduce overhead |
| Gradient accumulation with 16+ micro-steps | Lower `init_scale`; each micro-step multiplies overflow probability |

The overhead of dynamic scaling is negligible — one CPU-side check per step against the inf-tracker tensor, plus one `scaler.update()` call. Do not avoid it for performance reasons.

## 4. BF16 vs FP16: Choosing the Right Format

![Floating-point format comparison matrix: exponent bits, mantissa bits, dynamic range, loss scaling requirement](/imgs/blogs/mixed-precision-training-5.webp)

This is the single most impactful decision you make in a mixed-precision setup. The matrix above summarizes the tradeoffs, but the headline is: **BF16 is almost always the right answer on Ampere and later hardware**.

### Why BF16 wins on Ampere+

BF16's 8-bit exponent means its dynamic range is identical to FP32. A gradient of $10^{-37}$ is representable. This eliminates the need for loss scaling entirely — no `GradScaler`, no patience counter, no skipped steps, no inf-tracking tensor. Your training loop is simpler and slightly faster.

The cost is precision: 7 mantissa bits vs 10 for FP16 gives you about 2 significant decimal digits instead of 3. In practice this matters very little because:

1. Stochastic gradient descent with a mini-batch of 512+ tokens already has far more noise than 1 bit of mantissa.
2. The optimizer step runs in FP32 anyway, so accumulated rounding error doesn't compound across steps.
3. Empirically, BF16 and FP16 reach the same perplexity on language modeling tasks (within noise) on every large-scale training run I've seen.

### When FP16 might still win

- **Older hardware** (V100, T4, GTX series): BF16 tensor core support was introduced with Ampere (A100, A10G, RTX 3090). On V100 you get FP16 tensor cores only.
- **Models with activation outliers in the narrow range**: vision models with batch norm and high learning rates occasionally produce gradient norms that benefit from FP16's extra mantissa precision. Rare, but worth measuring.
- **TPU training**: TPU v2/v3 use BF16 natively; TPU v4 supports FP32. For cloud TPU workloads, BF16 is the default and usually correct choice regardless of the above.

### A/B testing format choice

To measure whether FP16 or BF16 hurts your specific model:

```python
import torch

def compare_precision_formats(model, data_loader, steps=200):
    """Run identical steps in FP16 and BF16, compare loss curves."""
    results = {}
    for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16)]:
        model_copy = copy.deepcopy(model).cuda()
        optimizer = torch.optim.AdamW(model_copy.parameters(), lr=1e-4)
        
        # fp16 needs a scaler; bf16 does not
        scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
        
        losses = []
        for i, batch in enumerate(data_loader):
            if i >= steps:
                break
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=dtype):
                out = model_copy(**batch)
                loss = out.loss
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model_copy.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        
        results[dtype_name] = losses
    return results
```

Run 200–500 steps and plot the loss curves. If they're within noise, go with BF16. If FP16 consistently reaches lower loss (which almost never happens in my experience), investigate whether a specific layer is the culprit before switching wholesale.

## 5. FP8 Training: The New Frontier

FP8 is the next step down — 8 bits total, split two ways depending on use case. NVIDIA's H100 and H800 introduce hardware-native FP8 tensor cores via the Transformer Engine library.

![FP8 dual-format training stack: Transformer Engine wrapping E4M3 forward and E5M2 backward GEMMs](/imgs/blogs/mixed-precision-training-6.webp)

### E4M3 and E5M2

**E4M3** (4 exponent, 3 mantissa bits): max value is 448, precision is roughly 1 significant digit. The narrow range is acceptable for forward activations because they are (in a well-trained model) close to zero. The extra mantissa bits matter for preserving subtle feature differences.

**E5M2** (5 exponent, 2 mantissa bits): max value is 57344, precision is less than 1 significant digit. The wider range is essential for backward gradients, which can span several decades depending on which layer they emerge from.

FP8 does not have automatic range management the way FP16/BF16 do. Each tensor requires a **per-tensor scale factor** — a FP32 scalar that maps the tensor's numerical range into E4M3 or E5M2's representable range. Transformer Engine tracks a rolling `amax` history (the maximum absolute value seen over the last N steps) and computes scales from that history. This is analogous to loss scaling but applied per-tensor rather than globally.

### Using Transformer Engine

```python
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

# configure FP8 recipe
fp8_recipe = DelayedScaling(
    margin=0,                   # buffer in exponent for safety
    interval=1,                 # update amax every step
    fp8_format=Format.HYBRID,   # E4M3 for fwd, E5M2 for bwd
    amax_history_len=16,        # rolling window for scale stability
    amax_compute_algo="max",
)

# replace nn.Linear with te.Linear — same API, FP8 kernel under the hood
model = te.Linear(in_features, out_features, bias=True)

# training loop
with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    output = model(input_tensor)
    loss = criterion(output, target)

loss.backward()
optimizer.step()
```

### When FP8 makes sense

FP8 is not a free upgrade. It requires:

1. **H100 / H800 / Ada Lovelace** hardware.
2. **Numerical stability in the model**: models with activation outliers (common in large LLMs trained without proper attention normalization) can see accuracy degradation. The LLaMA-2 architecture is FP8-stable; very deep networks with unusual activation scales may not be.
3. **The Transformer Engine library**: it handles per-tensor scaling, kernel dispatch, and the dequantization after each GEMM. Trying to implement FP8 without TE is significantly harder.

In practice, FP8 delivers 1.5–2x additional throughput over BF16 on H100 for transformer training, at roughly 0.1–0.3 perplexity points of degradation for large language models. For models above 7B parameters where H100s are the bottleneck, this is often worth accepting.

## 6. PyTorch AMP in Practice

![torch.autocast correct vs wrong placement: GradScaler.scale must be outside the autocast context](/imgs/blogs/mixed-precision-training-7.webp)

### The correct pattern

```python
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

model = MyTransformer().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scaler = GradScaler(
    init_scale=2**16,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000,
    enabled=True,  # set False for BF16 or debug runs
)

for epoch in range(num_epochs):
    for batch in data_loader:
        optimizer.zero_grad()
        
        # autocast wraps ONLY the forward pass
        with autocast(device_type="cuda", dtype=torch.float16):
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = nn.functional.cross_entropy(
                logits.view(-1, vocab_size),
                batch["labels"].view(-1),
            )
        
        # scaler.scale, backward, and step are OUTSIDE autocast
        scaler.scale(loss).backward()
        
        # unscale before clipping so clip operates on true gradient magnitudes
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # step only if no Inf/NaN found; scaler handles the skip internally
        scaler.step(optimizer)
        scaler.update()
        
        # log scale for monitoring
        if global_step % 100 == 0:
            print(f"step {global_step}, loss {loss.item():.4f}, "
                  f"scale {scaler.get_scale():.0f}")
        
        global_step += 1
```

### The common bug: scale inside autocast

The most frequent mistake I see in code reviews is placing `scaler.scale(loss).backward()` *inside* the `with autocast():` block. This looks fine — it runs without error — but it silently breaks loss scaling. Inside `autocast`, PyTorch has already changed the dtype context, so the scaled loss tensor may be processed through autocast rules rather than staying in the scale's intended dtype. The scaler's inf-detection can misfire, and in degenerate cases the scale collapses to 1.0 and stays there, meaning gradients go unscaled and the whole purpose of the scaler is defeated.

**Rule**: `autocast` wraps the forward pass only. Everything that touches the scaler — `scale()`, `backward()`, `unscale_()`, `step()`, `update()` — runs outside.

### BF16: no scaler needed

```python
# BF16 training — drop the scaler entirely
for batch in data_loader:
    optimizer.zero_grad()
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["labels"])
    # backward runs directly on the loss — no scale/unscale needed
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

This is the simplest correct training loop for modern hardware. If you're on A100 or newer, use this.

### Gradient accumulation with AMP

Gradient accumulation requires extra care. You must only call `scaler.step()` and `scaler.update()` on the final micro-step. Calling them after each micro-step means the scaler may skip a step on a micro-batch that has an `Inf` in its gradients even though the accumulated gradients across all micro-batches might be fine.

```python
accumulation_steps = 8
optimizer.zero_grad()

for i, batch in enumerate(data_loader):
    is_final_step = ((i + 1) % accumulation_steps == 0)
    
    with autocast(device_type="cuda", dtype=torch.float16):
        loss = model(**batch).loss / accumulation_steps  # normalize
    
    scaler.scale(loss).backward()
    
    if is_final_step:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

Note the division by `accumulation_steps` on the loss. Without this, your effective loss is `accumulation_steps` times larger than it would be with a single large batch, which means gradients are `accumulation_steps` times larger, which pushes the scaler toward constant overflow events.

### Checking which ops cast to FP16

PyTorch's autocast uses an allowlist to decide which ops run in FP16 and which stay in FP32. You can inspect this:

```python
import torch

# Check if an op is in the autocast whitelist
print(torch.cuda.amp.autocast_mode._amp_compatible_ops)

# Or trace what dtype a specific op uses
with autocast(device_type="cuda", dtype=torch.float16):
    x = torch.randn(1024, 1024, device="cuda")
    y = torch.randn(1024, 1024, device="cuda")
    z = torch.mm(x, y)
    print(f"matmul output dtype: {z.dtype}")  # float16

    # softmax stays in FP32 — it's on the FP32 allowlist
    s = torch.softmax(x, dim=-1)
    print(f"softmax output dtype: {s.dtype}")  # float32
```

Operations on the FP32 allowlist include: `softmax`, `log_softmax`, `cross_entropy`, layer normalization computations, and reductions. You generally do not need to override these.

## 7. Optimizer State and Master Weights

![Memory breakdown per parameter with AMP and Adam: 18 bytes total across FP16 copy, FP32 master, and optimizer moments](/imgs/blogs/mixed-precision-training-8.webp)

The memory arithmetic for AMP with Adam is worth knowing precisely. Per parameter:

| Buffer | Dtype | Bytes |
|---|---|---|
| FP16 weight copy (forward/backward) | FP16 | 2 |
| FP32 master weight | FP32 | 4 |
| Adam first moment $m$ | FP32 | 4 |
| Adam second moment $v$ | FP32 | 4 |
| Gradient buffer (after unscale) | FP32 | 4 |
| **Total** | | **18** |

For a 7B parameter model: $7 \times 10^9 \times 18 \approx 126$ GB. This is why a 7B model does not fit in a single 80 GB A100 in its full AMP-with-Adam state, even though the weights alone are only 14 GB in FP16.

### LoRA and PEFT interactions

When using LoRA (Low-Rank Adaptation), only the adapter matrices are trainable. The base model weights are frozen. This dramatically reduces the number of optimizer states. However, there are two common mistakes:

**Mistake 1**: Casting LoRA adapter weights to FP16 inside the optimizer. LoRA adapters are typically initialized with a very small scale (alpha/rank ratios around 0.01–0.1), meaning their gradients are small. If the optimizer maintains them in FP16, the updates are noisy. Keep LoRA adapter parameters in FP32 optimizer state.

```python
# Force LoRA parameters into a separate FP32 optimizer group
lora_params = [p for n, p in model.named_parameters() if "lora_" in n]
base_params = [p for n, p in model.named_parameters() if "lora_" not in n and p.requires_grad]

optimizer = torch.optim.AdamW([
    {"params": base_params, "lr": 0},       # frozen, zero lr
    {"params": lora_params, "lr": 3e-4},    # trainable
], weight_decay=0.01)
```

**Mistake 2**: Using `model.half()` before PEFT initialization. If you cast the base model to FP16 and then add LoRA adapters, the adapters inherit the FP16 dtype. Call PEFT's `get_peft_model()` before or after casting, but ensure LoRA adapter parameters themselves stay in FP32 by explicitly recasting:

```python
from peft import get_peft_model, LoraConfig

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, peft_config)

# ensure LoRA adapter params stay in FP32 for stable updates
for name, param in model.named_parameters():
    if "lora_" in name:
        param.data = param.data.float()
```

### Adam vs SGD memory comparison

For memory-constrained settings, SGD with momentum uses only 6 bytes/param with AMP: 2 (FP16 copy) + 4 (FP32 master) + 4 (momentum). No $v$ term. The 33% savings versus Adam can matter when you're right at the memory edge. In practice Adam converges faster and to better minima for most architectures, so only use SGD if you have a specific reason.

**AdaFactor** is even more aggressive: it approximates the second moment with a rank-1 outer product factorization, reducing $v$ from 4 B/param to roughly $\sqrt{N}$ total floats. Used in T5 training and still relevant for very large models on memory-constrained clusters.

## 8. Tips and Tricks

These are the micro-optimizations and patterns that make the difference between a clean training run and a debugging nightmare.

### Force specific layers to FP32

Not all ops play nicely with FP16. Three layers you should almost always run in FP32 regardless of your autocast setting:

**LayerNorm**: the normalization division is numerically sensitive. In FP16, the denominator (the standard deviation) can underflow to zero when activations are small, causing a division by zero or a very large normalized value. PyTorch's autocast allowlist already keeps `layer_norm` in FP32, but when you're using a custom LayerNorm implementation (e.g., Apex's `FusedLayerNorm` or a hand-rolled version), you must ensure FP32:

```python
class FP32LayerNorm(nn.LayerNorm):
    def forward(self, x):
        # cast input to FP32 for the norm, then cast back
        return super().forward(x.float()).to(x.dtype)
```

**Softmax in attention**: large query-key dot products in FP16 overflow to `Inf` before softmax normalizes them. This is the "attention softmax overflow" bug. The standard fix — already in Flash Attention and most modern attention implementations — is to compute the softmax in FP32:

```python
def attention_with_fp32_softmax(q, k, v, scale):
    # q, k, v arrive in FP16 from autocast
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    # cast to FP32 for numerically stable softmax
    weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(weights, v)
```

**Embedding layers**: embedding lookup produces integer-indexed float tensors. The gradients for embeddings are sparse and can have very high variance — one token may receive a gradient 100x larger than the average. Keep embeddings in FP32 if you see embedding gradient explosions.

### Gradient clipping order: unscale first

The single most common ordering bug I see: calling `clip_grad_norm_` before `scaler.unscale_()`. If you clip before unscaling, you're clipping scaled gradients. A clip of `max_norm=1.0` on gradients that are scaled by $2^{16}$ clips essentially nothing useful. Always:

```python
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
```

### Monitor your scale

Add `scaler.get_scale()` to your logging. A healthy training run shows a scale that stabilizes above $2^{10}$ (1024) and rarely changes after the first few thousand steps. Warning signs:

- Scale below 256: your gradients are frequently overflowing; investigate which layer.
- Scale collapsing toward 1: extremely frequent overflows; the scaler is essentially disabled. Check for layer-specific instability.
- Scale growing unboundedly (beyond $2^{30}$): your model is extremely numerically stable; nothing wrong, but worth noting.

### Controlling which ops cast

If a specific operation is causing numeric issues inside `autocast`, override its dtype locally:

```python
with autocast(device_type="cuda", dtype=torch.float16):
    x = encoder(inputs)                    # runs in FP16
    
    # force this matmul to FP32
    with autocast(device_type="cuda", enabled=False):
        problematic_output = special_op(x.float())
    
    output = decoder(problematic_output.half())  # back to FP16
```

### Mixed precision with `torch.compile`

With `torch.compile` (PyTorch 2.0+), the autocast context interacts with the compiled graph. The safe pattern:

```python
@torch.compile
def forward_fn(model, batch):
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        return model(**batch)

# or compile the model directly with fullgraph=True to trace through autocast
model = torch.compile(model, fullgraph=True, mode="reduce-overhead")
```

Note: `GradScaler` does not interact well with `torch.compile` in all versions. If you're using BF16 (no scaler needed), this is not an issue. For FP16 + GradScaler + torch.compile, test carefully and check for correctness regressions.

### Mixed precision across multiple GPUs

When using DDP (DistributedDataParallel), the gradient synchronization between GPUs happens in the gradient's current dtype. With AMP, gradients are in FP16 at the time of the `all-reduce`. This is usually fine — the scaling ensures they're in range — but watch for NCCL dtype mismatch errors if some parameters have been explicitly cast to FP32.

With FSDP (FullyShardedDataParallel), configure the `MixedPrecision` policy explicitly:

```python
from torch.distributed.fsdp import MixedPrecision
import torch

mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,        # shard in BF16
    reduce_dtype=torch.float32,         # all-reduce in FP32 for accuracy
    buffer_dtype=torch.bfloat16,       # non-parameter buffers
)
```

Using `reduce_dtype=torch.float32` avoids accumulation errors during gradient aggregation across many GPUs, at the cost of higher communication bandwidth. For large clusters (256+ GPUs), this tradeoff is often worth it.

## 9. Troubleshooting Gallery

![Mixed precision troubleshooting decision tree: NaN, slow convergence, loss spike, optimizer mismatch](/imgs/blogs/mixed-precision-training-9.webp)

### NaN gradients on step N

**Symptom**: training runs fine, then at a specific step (often not step 1), gradients become NaN. Loss goes to NaN or Inf.

**Root causes and diagnostics**:

```python
# add this hook to find which parameter first produces NaN
def nan_detector_hook(name):
    def hook(grad):
        if torch.any(torch.isnan(grad)):
            print(f"NaN gradient detected in parameter: {name}")
            print(f"grad stats: min={grad.min():.4f}, max={grad.max():.4f}")
        return grad
    return hook

for name, param in model.named_parameters():
    if param.requires_grad:
        param.register_hook(nan_detector_hook(name))
```

**Most likely causes in order**:
1. Loss scale too large: an individual gradient overflows to `Inf`, which propagates backward through the chain rule. The scaler should catch this and skip the step — verify `scaler.get_scale()` is decreasing.
2. `log(0)` in the loss: a softmax output of exactly 0 fed to cross-entropy. Add label smoothing or a small epsilon.
3. Softmax overflow: very large logits cause `exp(x) = Inf` before the softmax denominator can normalize. Use scaled dot-product attention and cap logit magnitude.
4. Embedding index out-of-range: a misconfigured dataset produces an input token ID larger than `vocab_size`. This is FP32-only territory but causes NaN that propagates to mixed precision.

### Scale collapsing to zero

**Symptom**: `scaler.get_scale()` prints 1.0 (or very close to it) after a few hundred steps. Loss trains but slowly.

**Diagnosis**: every step is overflowing and the backoff factor halves the scale faster than it can grow. The model has a layer producing Inf gradients on effectively every forward pass.

**Fix checklist**:
1. Check if the learning rate is too large. A 10x-too-large LR produces large activations, which cause softmax overflow, which cascades.
2. Identify the offending layer with the NaN hook above.
3. Ensure LayerNorm is running in FP32.
4. If using custom attention, ensure QK dot products are scaled by $1/\sqrt{d_k}$ and softmax runs in FP32.

### Loss trains fine but final accuracy is lower than FP32 baseline

**Symptom**: convergence behavior looks normal, but eval metrics (perplexity, accuracy, BLEU) lag the FP32 baseline by more than 0.5%.

**Root cause**: usually optimizer state precision. Check whether Adam's $m$ and $v$ tensors are being stored in FP16 (which can happen with certain deepspeed or custom optimizer configurations). Verify with:

```python
for g in optimizer.state.values():
    for k, v in g.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {v.dtype}")
```

Everything should be `torch.float32`. If `m` or `v` are FP16, your updates are noisy and the model will underfit.

### Distributed training: loss spikes at specific steps

**Symptom**: in DDP or FSDP training, the loss spikes at regular intervals (e.g., every 2000 steps) then recovers. Corresponds to the `growth_interval` of the scaler.

**Root cause**: each worker has an independent `GradScaler`. When the scale doubles (after `growth_interval` successful steps), there is a brief period of higher-magnitude scaled gradients before the system adjusts. In DDP, the gradient all-reduce happens *before* unscaling, so all workers see the same scaled gradient magnitude. This is generally fine.

The real bug version: different workers have different scales because they were initialized at different times or had different numbers of skipped steps. Gradients all-reduced across workers with different scales become incorrect. Fix by synchronizing the scaler across workers:

```python
# after initialization, sync the initial scale across ranks
scale_tensor = torch.tensor([scaler.get_scale()], device="cuda")
torch.distributed.broadcast(scale_tensor, src=0)
scaler._scale = scale_tensor
```

### FP8 instability: loss diverges after 1000+ steps

**Symptom**: FP8 training looks stable for the first several hundred steps, then diverges.

**Root cause**: `amax_history_len` too short. If the rolling window for scale estimation is smaller than the steps needed for the model to produce outlier activations (which often appear only after the model has partially learned useful representations), the scale can be temporarily wrong. Fix: increase `amax_history_len` from 16 to 32 or 64, and check that `margin` is at least 0 (positive margins add safety buffer in the exponent).

### Gradient accumulation: loss is NaN from step 1

**Symptom**: with gradient accumulation enabled, NaN appears immediately.

**Root cause**: calling `scaler.step()` and `scaler.update()` on every micro-step instead of every `accumulation_steps` micro-steps. If the first micro-batch happens to have an Inf gradient and `scaler.update()` is called, the scale halves before the remaining micro-batches have contributed. The accumulated gradient (partially from before the halve, partially from after) becomes internally inconsistent.

Fix: gate `scaler.step()`, `scaler.update()`, and `optimizer.zero_grad()` on `is_final_step` as shown in §6.

## 10. Best Practices Checklist

![Precision format decision flowchart: hardware generation determines format, then task sensitivity, then monitoring](/imgs/blogs/mixed-precision-training-10.webp)

### Decision flow

1. **H100 / H800?** → Try FP8 via Transformer Engine. If the model is numerically unstable under FP8, fall back to BF16.
2. **A100 / A10 / RTX 3000+?** → Use BF16. Drop the GradScaler. Verify BF16 tensor core utilization with `nsys profile` or `torch.profiler`.
3. **V100 / T4 / older?** → Use FP16 + dynamic GradScaler with default parameters. Monitor `scaler.get_scale()`.

### Layer-level checklist

| Layer | Recommended action |
|---|---|
| Embedding | FP32 if gradient variance is high; BF16 otherwise |
| LayerNorm | FP32 always (PyTorch's autocast handles this by default) |
| Attention (QK dot + softmax) | Softmax in FP32; QK in FP16/BF16 |
| FFN (GEMM-heavy) | FP16/BF16 — where you get the most speedup |
| Loss function | FP32 (PyTorch handles this by default) |
| Optimizer state (m, v) | FP32 always |

### Monitoring in production

Add these metrics to your training dashboard:

```python
class AMPMonitor:
    """Lightweight tracker for AMP health metrics."""
    
    def __init__(self, scaler, log_every=100):
        self.scaler = scaler
        self.log_every = log_every
        self._skipped_steps = 0
        self._total_steps = 0
    
    def step(self, global_step):
        scale = self.scaler.get_scale()
        self._total_steps += 1
        
        # detect whether the last step was skipped (scale decreased)
        if hasattr(self, "_prev_scale") and scale < self._prev_scale:
            self._skipped_steps += 1
        self._prev_scale = scale
        
        if global_step % self.log_every == 0:
            skip_rate = self._skipped_steps / max(self._total_steps, 1)
            print(f"[AMP] step={global_step} scale={scale:.0f} "
                  f"skip_rate={skip_rate:.3f}")
            
            # alert conditions
            if scale < 256:
                print("[AMP] WARNING: scale below 256 — check for layer overflow")
            if skip_rate > 0.05:
                print("[AMP] WARNING: >5% steps skipped — consider reducing LR")
```

### Profiling mixed precision utilization

Use `torch.profiler` to verify tensor cores are actually being used:

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("training_step"):
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(**batch).loss
        loss.backward()

# look for "aten::_ampere_bf16_gemm" or "aten::mm" with FP16 inputs
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

If you see `aten::mm` with FP32 inputs dominating the CUDA time, your autocast is not covering the expensive matmuls.

## 11. Case Studies

### Case Study 1: LLaMA-2 70B pretraining — BF16 on A100

Meta's LLaMA-2 technical report describes training in BF16 on 2048 A100s. The choice of BF16 over FP16 was deliberate: at this scale, a single skipped step from a loss scale overflow means one of 2048 workers falls behind and requires gradient synchronization recovery. BF16's overflow-free property eliminates this failure mode entirely.

The training used gradient clipping at `max_norm=1.0` applied in FP32 after accumulation. No `GradScaler` was used. One interesting detail: the model used SwiGLU activations, which involve pointwise multiplications of half-precision tensors. Without careful initialization, these multiplications can produce very large or small values early in training. LLaMA-2 used RMS norm (equivalent to LayerNorm without the centering step) which is simpler and less numerically sensitive than standard LayerNorm.

**Lesson**: for large-scale pretraining on Ampere, BF16 is the default. The zero-overhead property (no scaler, no skipped steps) dominates all other considerations.

### Case Study 2: Stable Diffusion XL fine-tuning — FP16 softmax overflow

During early fine-tuning experiments with SDXL (a 2.6B parameter U-Net with cross-attention), we encountered a specific failure: training loss would NaN around step 500–800. The scale was healthy (around $2^{15}$), which ruled out underflow.

Culprit: the cross-attention softmax in the U-Net's mid-block was receiving query-key products in the range of ±60,000 before scaling by $1/\sqrt{d}$. With $d=64$, the scale factor was $1/8 = 0.125$, leaving QK products around ±7,500 — which overflow FP16's max of 65504 only during rare batches with high-magnitude features.

Fix:

```python
# In the attention block's forward method:
def forward(self, q, k, v):
    scale = q.shape[-1] ** -0.5
    attn_scores = torch.einsum("bhid,bhjd->bhij", q, k) * scale
    # explicit FP32 cast for softmax stability
    attn_weights = torch.softmax(attn_scores.to(torch.float32), dim=-1)
    attn_weights = attn_weights.to(q.dtype)
    return torch.einsum("bhij,bhjd->bhid", attn_weights, v)
```

After this fix, FP16 training was stable through 10k steps with no loss spikes.

### Case Study 3: BERT fine-tuning on domain corpus — embedding layer NaN

Fine-tuning a BERT-base model on a small domain corpus (~50K documents) with FP16 produced NaN at step 23 reproducibly. The GradScaler's scale was $2^{16}$ and healthy.

Root cause: the domain corpus had a token (a numeric code) with index 30521 — exactly at `vocab_size - 1`. During a rare batch where this token appeared 47 times, the embedding gradient for that index accumulated 47 FP16 additions. FP16 can represent at most about 2048 as an integer without precision loss, and 47 × (each gradient) was pushing beyond FP16's representable range for sparse embedding gradients.

Fix: force the embedding layer to FP32:

```python
model.bert.embeddings.word_embeddings.weight.data = \
    model.bert.embeddings.word_embeddings.weight.data.float()
```

And register the embedding as a FP32 parameter excluded from autocast:

```python
def forward(self, input_ids):
    # bypass autocast for the embedding lookup
    with autocast(device_type="cuda", enabled=False):
        embeds = self.embeddings(input_ids.long())
    # re-enter FP16 for the encoder
    return self.encoder(embeds.half())
```

Training completed without NaN from step 24 onward.

### Case Study 4: GPT-NeoX 20B — loss scale collapse incident

During a 20B parameter GPT-NeoX training run (FP16, 256 A100s), the training loss jumped from 2.3 to 4.7 at step 12,840 and plateaued. The scaler logs showed scale at 1.0 from step 12,790 onward.

Root cause: a data preprocessing bug introduced a batch with all-zero attention masks for ~3% of sequences starting at step 12,780. With zero attention masks, softmax receives `−Inf` logits for all positions, the softmax output is undefined (NaN or 0/0), and the gradient of any downstream operation with respect to the attention logits is NaN. This pushed the scaler into rapid backoff: over 50 consecutive overflowing steps from 12,780–12,829, the scale halved 50 times, collapsing from $2^{16}$ to $2^{-34} \approx 5.8 \times 10^{-11}$.

At scale $5.8 \times 10^{-11}$, gradients were multiplied by essentially zero before backward, meaning the entire training from 12,790 onward made no useful updates. The loss plateau was not divergence — it was effectively frozen weights.

Lessons:
1. **Log the scale and alert when it drops below 256.** The collapse started at 12,780; an alert would have caught it within 10 minutes instead of 12 hours later when a human noticed the plateau.
2. **Validate attention masks in your dataloader.** Add an assertion: `assert mask.any(dim=-1).all()` (every sequence has at least one unmasked token).
3. **Add a step-skip-rate monitor.** A 100% skip rate for 50+ steps is unambiguous; even a 10% skip rate sustained for 100 steps warrants investigation.

### Case Study 5: Enabling FP8 for a 13B model on H100 — amax instability

Training a 13B decoder-only model with Transformer Engine's FP8 on 8× H100s: the first 500 steps looked normal, then loss spiked at step 502 and recovered, then again at 744 and recovered. After the third spike at step 1001, the loss did not recover.

Root cause: `amax_history_len=8` (the TE default at the time) was too short. The model's attention layers produced activation outliers with irregular periodicity — roughly every 300–500 steps, a batch would contain particularly long sequences that caused one or two attention heads to produce activations up to 10× larger than the rolling average. When this happened, the per-tensor scale was too small for the outlier, causing overflow to FP8's max value (448 for E4M3), which became the wrong activation for that batch, which corrupted the layer's output, which propagated to the loss.

Fix:

```python
fp8_recipe = DelayedScaling(
    margin=0,
    interval=1,
    fp8_format=Format.HYBRID,
    amax_history_len=32,    # extended from 8 to 32
    amax_compute_algo="max",  # use the maximum seen in the window
)
```

With `amax_history_len=32`, the rolling window captured the outlier events and the scale was sized conservatively enough to handle them. Training ran cleanly to completion.

**Lesson**: when adopting FP8 for a new model architecture, always start with `amax_history_len=32` and monitor per-layer amax values via TE's debug logging. Reduce only if memory pressure demands it.

## When to Use Mixed Precision — and When Not To

**Use mixed precision (BF16 or FP16) when:**
- You are training on any GPU with tensor core support (effectively everything from Volta onward).
- Your model architecture is standard: transformer blocks, LayerNorm, softmax attention, standard activations.
- Memory or throughput is a constraint (which is almost always).
- You are fine-tuning a pretrained model where the pretrained weights are already well-conditioned.

**Be cautious or use FP32 when:**
- Researching a new architecture where numerical properties are not yet characterized.
- Training a model with unusual activation distributions (custom normalizations, non-standard activations, extremely deep networks).
- Working with scientific computing targets where FP32 precision is a correctness requirement, not a nice-to-have.
- Debugging a correctness issue: always reproduce in FP32 first to eliminate precision as a confounding variable.

**Use FP8 when:**
- You have H100 hardware and throughput is the primary constraint.
- The model architecture is a standard transformer trained in BF16 for at least 10k steps (confirming stability) before switching to FP8.
- You are comfortable running amax profiling for the first 500–1000 steps to tune the recipe.

---

Related topics worth reading alongside this post:

- [Sequence packing for LLM fine-tuning](/blog/machine-learning/training-techniques/sequence-packing-llm-fine-tuning) — efficient token utilization in the training loop
- [Speeding up neural network training 4x by optimizing CPU-to-GPU data transfer](/blog/machine-learning/training-techniques/speeding-up-neural-network-training-4x-by-optimizing-cpu-to-gpu-data-transfer) — the data pipeline bottleneck that limits what precision changes can achieve
- [CUDA graph](/blog/machine-learning/deep-learning/cuda-graph) — reducing kernel launch overhead that compounds with AMP overhead
- [Fine-tuning tool-calling LLMs](/blog/machine-learning/training-techniques/fine-tuning-tool-calling-llms-when-how) — PEFT patterns in production where optimizer precision matters

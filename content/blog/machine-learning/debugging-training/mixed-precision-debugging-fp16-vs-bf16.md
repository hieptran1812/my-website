---
title: "Mixed-Precision Debugging: fp16 vs bf16 and the Bugs Between Them"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Learn why fp16 quietly underflows your gradients to zero and overflows your activations to inf, how loss scaling and bf16 fix it, and how to read the GradScaler to localize a numerics bug in minutes."
tags:
  [
    "debugging",
    "model-training",
    "mixed-precision",
    "fp16",
    "bf16",
    "numerics",
    "pytorch",
    "finetuning",
    "deep-learning",
    "llm",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/mixed-precision-debugging-fp16-vs-bf16-1.png"
---

Here is a run that wasted two GPU-days. A ResNet-50 finetune on a medical-imaging dataset, automatic mixed precision turned on because the docs promised a 2.5× speedup and it delivered. Loss fell from 2.31 to 0.94 over the first epoch, looked beautiful, and then at step 1,840 it printed `loss=nan` and never came back. The engineer who owned it did the natural thing: lowered the learning rate, added gradient clipping, shuffled the data, swapped the optimizer. Nothing helped. The NaN moved around — sometimes step 1,500, sometimes step 2,300 — but it always came. Three days in, someone flipped one flag, `fp16` to `bf16`, changed nothing else, and the run trained clean to convergence. No NaN. The bug was never the learning rate, never the data, never the optimizer. It was five bits.

This is the most common silent failure I see in modern training, and it lives in exactly one of the six places a bug can hide: **numerics**. Not data, not optimization, not model code, not systems, not evaluation. The whole reason mixed precision is dangerous is that it is *almost* invisible — your model is still mathematically the same model, your data is the same data, your hyperparameters are untouched. What changed is the set of numbers your hardware can represent, and a class of values your gradients routinely produce now rounds to zero or saturates to infinity. The arithmetic is the bug. Figure 1 lays the three formats side by side so you can see, bit for bit, where fp16 runs out of room.

![Bit layout of fp32 with 8 exponent and 23 mantissa bits, fp16 with 5 exponent and 10 mantissa bits, and bf16 with 8 exponent and 7 mantissa bits, showing fp16 has the narrowest dynamic range](/imgs/blogs/mixed-precision-debugging-fp16-vs-bf16-1.png)

By the end of this post you will be able to look at a NaN-ing or quietly-degrading mixed-precision run and localize the cause in minutes instead of days. You will know why fp16 underflows small gradients to zero at a hard floor near $6.1\times10^{-5}$ and overflows large activations to `inf` above $65{,}504$; why bf16 trades mantissa bits for the full fp32 exponent range and so almost never overflows but rounds more coarsely; what the fp32 master-weights copy is actually for; how loss scaling moves your gradients into the representable band and how the dynamic scaler chooses its factor; and which operations you must force back to fp32 no matter which format you pick. Most of all you will learn to **read the instruments** — the `GradScaler` scale over time, a gradient histogram, an autocast-dtype audit — so that the format bug confesses itself instead of hiding behind a misleading symptom. This is a numerics post in the debugging series; it slots directly under the taxonomy decision tree and feeds the capstone playbook.

## 1. The science: what a floating-point format can and cannot say

Every floating-point number is three fields: a sign bit, an exponent, and a mantissa (also called the significand or fraction). The value is approximately $(-1)^{\text{sign}} \times 1.\text{mantissa} \times 2^{\text{exponent} - \text{bias}}$. Two properties matter for training, and they are controlled by *different* fields, which is the entire reason fp16 and bf16 behave so differently.

The **exponent** controls **dynamic range** — how large and how small a number you can represent at all. More exponent bits means you can reach closer to zero (tiny gradients) and further toward infinity (large activations) before the format gives up. The **mantissa** controls **precision** — how finely you can resolve two nearby numbers. More mantissa bits means smaller relative rounding error per operation. Range is about *reach*; precision is about *resolution*. A format can be wide and coarse, or narrow and fine, and the choice between fp16 and bf16 is exactly that trade.

Here are the three formats that matter, with the numbers worth memorizing:

| Format | Sign | Exponent | Mantissa | Max finite | Min normal | Relative precision (eps) |
| --- | --- | --- | --- | --- | --- | --- |
| fp32 | 1 | 8 | 23 | $\approx 3.4\times10^{38}$ | $\approx 1.18\times10^{-38}$ | $\approx 1.2\times10^{-7}$ |
| fp16 | 1 | 5 | 10 | $65{,}504$ | $\approx 6.10\times10^{-5}$ | $\approx 9.8\times10^{-4}$ |
| bf16 | 1 | 8 | 7 | $\approx 3.4\times10^{38}$ | $\approx 1.18\times10^{-38}$ | $\approx 7.8\times10^{-3}$ |

Read that table the way a debugger reads instruments. fp16 spends 5 bits on exponent, which buys it a maximum of $65{,}504$ and a smallest normal number of about $6.1\times10^{-5}$. Above the max it overflows to `inf`; below the min-normal it slides into subnormals and then to zero (the absolute smallest subnormal is about $6\times10^{-8}$, but you lose mantissa precision the whole way down). bf16 spends 8 bits on exponent — the *same as fp32* — so it reaches $3.4\times10^{38}$ and down to $1.18\times10^{-38}$, an enormous range. bf16 pays for that range with only 7 mantissa bits, so its relative rounding error is about $7.8\times10^{-3}$: roughly one part in 128. fp16's 10 mantissa bits give it about one part in 1024, finer than bf16. And fp32's 23 mantissa bits give about one part in 8 million, which is why we keep a copy of the weights there.

One more reading of the table, because it is the crux of every choice in this post. Compare fp16 and bf16 directly: they are *the same size* — 16 bits each — but they spend those bits differently. fp16 puts 5 in the exponent and 10 in the mantissa: **narrow range, fine resolution.** bf16 puts 8 in the exponent and 7 in the mantissa: **wide range (identical to fp32), coarse resolution.** bf16 is, quite literally, fp32 with the bottom 16 mantissa bits chopped off — which is why converting fp32↔bf16 is nearly free and never overflows. The entire fp16-versus-bf16 debate reduces to one question about your training run: *do you have more of a range problem or a precision problem?* Training, with its tiny gradients and occasional large activations, is overwhelmingly a **range** problem — which is why bf16 wins for training. Inference, with bounded well-behaved values, is more of a **precision** problem — which is one reason fp16 remains popular for serving. Hold that framing; it predicts the right format almost every time.

Now make this concrete, because abstract bit-counts do not cause NaNs — specific values do.

#### Worked example: the gradient that underflows

Take a typical gradient deep in a network during finetuning. Say a particular weight's gradient is $g = 3.0\times10^{-6}$. That is a perfectly normal, useful gradient — small, but it should nudge the weight. In fp32 it is represented exactly enough. In fp16, the smallest *normal* number is $6.1\times10^{-5}$, which is about 20× larger than our gradient. Our $3.0\times10^{-6}$ falls into fp16's subnormal region, where it is represented with badly degraded precision, and many such values round straight to $0.0$. The optimizer then applies an update of zero. That weight does not learn. Multiply this across the many small gradients in a deep network and **the model silently stops improving in its lower layers** — not with a crash, but with a plateau. The loss curve looks like a stuck learner; the real cause is numerics. This is the underflow signature, and it is invisible unless you instrument for it.

#### Worked example: the activation that overflows

Now the other end. Suppose during a forward pass an attention score or an intermediate activation reaches $80{,}000$ — not unusual in an unnormalized residual stream or a layer whose scale crept up. fp16's maximum finite value is $65{,}504$. The value $80{,}000$ has nowhere to live; it becomes `inf`. The next operation that touches it (a subtraction of two infs, a multiply by zero) produces `NaN`, and from that step forward every gradient is `NaN`, every weight becomes `NaN`, and the run is dead. This is the overflow signature: a smooth loss curve that suddenly hits a wall. Critically, **bf16 would not overflow here** — $80{,}000$ is trivially inside bf16's range, which reaches $3.4\times10^{38}$. The exact same network, same data, same step, lives or dies on which five-versus-eight exponent bits you chose. That is the bug from the intro.

So the science gives us two falsifiable predictions before we touch any code:

1. fp16 will **underflow** gradients below roughly $6\times10^{-5}$ to zero (a plateau / vanishing signature) and **overflow** activations above $65{,}504$ to inf→NaN (a sudden-wall signature).
2. bf16, with fp32's exponent range, will essentially never overflow or underflow at the magnitudes training produces, but its 7-bit mantissa means **coarser rounding** — which bites precision-sensitive reductions, not stability.

Everything else in this post is making those two predictions observable and then acting on them.

One more number worth internalizing before we move on: the **gap to the next representable value** (the "ulp", unit in the last place) is not constant — it scales with the magnitude of the number, because floating point is logarithmically spaced. Near $1.0$, fp16's gap is about $9.8\times10^{-4}$, fp16 near $1024$ has a gap of about $1.0$, and fp16 near $32{,}768$ has a gap of about $32$. That last fact is startling and important: at the top of fp16's range, **consecutive representable integers are 32 apart** — you cannot represent $32{,}001$, only $32{,}000$ and $32{,}032$. So even *before* an activation overflows to inf at $65{,}504$, the values up there are being quantized into coarse buckets. A logit of $50{,}000$ and a logit of $50{,}016$ are the same fp16 number. This is why "it didn't NaN" is not the same as "it was accurate" — the high end of fp16 is a precision desert long before it is a cliff. bf16, with its wide-but-coarse design, has a gap near $1.0$ of about $7.8\times10^{-3}$ everywhere proportionally, so it is uniformly coarse rather than coarse-only-at-the-top. Keep that asymmetry in mind; it explains several of the bugs below.

### Why mixed precision is 2–3× faster (and why that's the whole point)

It is worth being precise about *where* the speedup comes from, because the magnitude of the win is exactly what makes the bug class worth tolerating. Two mechanisms.

**Tensor cores.** Modern NVIDIA GPUs (Volta onward) have dedicated matrix-multiply units — tensor cores — that perform fused multiply-accumulate on 16-bit inputs at a throughput several times higher than the fp32 ALUs. On an A100, fp16/bf16 tensor-core matmul peaks around 312 TFLOP/s versus about 19.5 TFLOP/s for fp32 on the regular cores — a 16× theoretical gap, of which real training captures roughly 2–3× end-to-end (the rest is lost to memory traffic, non-matmul ops, and the fp32 reductions we keep). Since transformers and CNNs are dominated by matmuls and convolutions, moving those to 16-bit is most of the run.

**Memory bandwidth and footprint.** A 16-bit tensor is half the bytes of a 32-bit tensor. Activations, the dominant memory consumer in deep networks, halve. That means you can fit a larger batch or a larger model in the same GPU, and — often more importantly — you move half as many bytes between HBM and the compute units. Many layers (normalization, element-wise ops, attention's softmax-times-values) are **memory-bound**, not compute-bound, so halving the bytes nearly halves their time. In the ResNet case study at the end, peak memory drops from 19.8 GB to 11.3 GB (a 43% cut) purely from 16-bit activations.

So the deal mixed precision offers is: **2–3× faster, ~40% less memory, for the price of a numerics bug class.** That is a great deal — which is why everyone takes it — provided you can debug the bug class in minutes. That is what the rest of this post is for.

## 2. The fp32 master copy and why mixed precision is "mixed"

A persistent confusion: if I train in fp16, are my weights fp16? In well-implemented automatic mixed precision (AMP), **no**. The optimizer keeps a **master copy of the weights in fp32**. The forward and backward passes are computed in the low-precision format (fp16 or bf16) for speed and memory, producing low-precision activations and gradients, but the *weight update itself* — $w \leftarrow w - \eta \cdot g$ — happens in fp32 against the master copy. That is the "mixed" in mixed precision, and figure 2 shows the loop.

![A training loop showing fp16 forward and backward passes producing low precision gradients, a loss scale multiply before backward and unscale after, and an fp32 master weight copy receiving the optimizer update each step](/imgs/blogs/mixed-precision-debugging-fp16-vs-bf16-2.png)

Why does the master copy matter? Because the *update* is often much smaller than the *weight*. Consider a weight $w = 1.0$ and a learning rate times gradient term $\eta g = 1\times10^{-4}$. In fp16 the value $1.0$ has a spacing to its next representable neighbor of about $1\times10^{-3}$ (that is fp16's eps near 1.0). An update of $1\times10^{-4}$ is *smaller than the gap between representable fp16 numbers near 1.0* — so $1.0 + 1\times10^{-4}$ rounds right back to $1.0$. The update vanishes not because the gradient underflowed, but because the *accumulation* lost it. This is the "swamping" problem: small updates added to larger weights disappear if both live in low precision. Keeping the master weights in fp32, whose spacing near 1.0 is about $1.2\times10^{-7}$, preserves those small updates. The "Mixed Precision Training" paper (Micikevicius et al., 2018) is explicit about this: the fp32 master copy is not optional polish, it is one of the three pillars (master weights, loss scaling, fp32 reductions) that make fp16 training match fp32 accuracy.

Let me make the swamping rule precise, because it predicts exactly when an update vanishes. A floating-point format with mantissa epsilon $\epsilon$ can resolve a change to a number $w$ only if that change is at least $\epsilon \cdot |w|$ — anything smaller rounds away. So the update $\eta g$ survives the accumulation $w \leftarrow w + \eta g$ only when

$$
|\eta g| \gtrsim \epsilon \cdot |w|.
$$

In fp16, $\epsilon \approx 9.8\times10^{-4}$. For a weight of magnitude $|w| = 1$, any update smaller than about $10^{-3}$ is lost. Typical late-training updates $\eta g$ are routinely $10^{-4}$ to $10^{-6}$ — one to three orders of magnitude *below* that threshold. So in pure fp16, late-training updates to order-one weights simply stop accumulating; the model freezes even though the gradients are fine and even though loss scaling rescued them from underflow. The freeze is in the *accumulation*, not the gradient. Move the accumulation to fp32, where $\epsilon \approx 1.2\times10^{-7}$, and the survival threshold for $|w| = 1$ drops to about $10^{-7}$ — now your $10^{-6}$ updates land. That is the entire justification for the fp32 master copy, and it is a one-line inequality you can check for your own weight and update magnitudes.

bf16 changes the calculus slightly but not the conclusion. Because bf16 shares fp32's exponent, you can in some setups train with bf16 weights and a bf16 optimizer state and still converge — but the 7-bit mantissa means the same swamping happens *sooner* (bf16's eps near 1.0 is about $7.8\times10^{-3}$, even coarser than fp16's). Plug $\epsilon = 7.8\times10^{-3}$ into the inequality: a bf16 order-one weight cannot absorb an update below about $8\times10^{-3}$, so bf16 freezes accumulation even more readily than fp16. So the master-fp32-copy discipline still earns its keep, and is arguably *more* important for bf16 than for fp16. The practical rule: **let AMP keep your master weights in fp32 and do not "optimize" that away** unless you have measured that a fully-low-precision optimizer matches your fp32 baseline. The savings are smaller than people think (optimizer states dominated by Adam's two moments are usually kept fp32 regardless), and the risk is a slow, invisible accuracy regression.

This is also why the AMP API in PyTorch has two pieces. `torch.amp.autocast` decides, op by op, which low-precision format to compute in during the forward and backward passes. `torch.amp.GradScaler` handles the loss-scaling dance that keeps fp16 gradients out of the underflow zone, then unscales them before the fp32 master update. autocast is about *speed and range during compute*; GradScaler is about *not losing small gradients*. They are separate concerns and you debug them separately.

## 3. Loss scaling: the central trick, derived

Loss scaling is the single idea that makes fp16 training work, and it is worth deriving rather than memorizing, because once you see *why* the factor is chosen the way it is, reading the scaler over time becomes second nature.

The problem, restated precisely: gradients during training tend to be small. Empirically, in many networks the gradient histogram has most of its mass between roughly $10^{-8}$ and $10^{-2}$, and a meaningful fraction sits below fp16's normal-range floor of $6.1\times10^{-5}$. Everything below that floor underflows toward zero. But here is the lever: **fp16 has plenty of room at the top** — gradients almost never approach $65{,}504$. So the representable band is *unbalanced*: empty at the top, crowded against the floor at the bottom. The fix is to **shift the whole gradient distribution up** into the empty room, do the fp16 work there, then shift it back down.

Why is the distribution shaped this way? Two reasons that are worth holding onto. First, **backpropagation multiplies**: a gradient deep in the network is a product of many Jacobian factors, each typically less than one in magnitude (weights initialized small, activations bounded by normalization), so the product shrinks with depth — the same mechanism behind vanishing gradients, here pushing mass toward the floor. Second, **the loss derivative is often small near a good region**: as the model improves, $\partial L / \partial \text{logit}$ shrinks, dragging the whole distribution down over training. Both effects mean the gradient distribution lives near the *bottom* of fp16's range and drifts lower as training proceeds — which is exactly why a *static* scale chosen early is wrong later, and why the dynamic scaler's slow upward creep (double every 2,000 clean steps) tracks the distribution down by keeping the scaled values pinned just under the overflow ceiling. The loss scale is, in effect, an automatic feedback controller keeping the gradient distribution centered in the representable band as that distribution moves.

Mechanically: multiply the loss by a scale factor $S$ before calling `backward()`. By the chain rule, every gradient is then multiplied by the same $S$:

$$
\frac{\partial (S \cdot L)}{\partial w} = S \cdot \frac{\partial L}{\partial w}.
$$

A gradient of $3\times10^{-6}$, which would underflow, becomes $S \cdot 3\times10^{-6}$. Pick $S = 2^{16} = 65{,}536$ and it becomes about $0.197$ — comfortably inside fp16's normal range, represented with full mantissa precision. The backward pass runs in fp16 with no underflow. Then, *before the optimizer step*, you **unscale**: divide the gradients by $S$ (in fp32) to recover the true gradient magnitude, and apply the fp32 master update. The net effect on the math is exactly nothing — you scaled up and scaled back down — but the round trip happened in the representable band instead of in the underflow zone. Figure 3 shows the distribution shift.

![A before and after comparison where unscaled fp16 gradients pile up below the underflow floor and round to zero, while loss scaled gradients shift up into the representable band and survive](/imgs/blogs/mixed-precision-debugging-fp16-vs-bf16-3.png)

How is $S$ chosen? Two regimes.

**Static loss scaling** picks one $S$ (often $2^{8}$ to $2^{16}$) and uses it for the whole run. It works if your gradient magnitudes are stable, but it is brittle: too small and you still underflow; too large and scaled gradients overflow to inf. You are guessing a constant for a quantity that changes over training.

**Dynamic loss scaling** — what PyTorch's `GradScaler` does by default — adapts $S$ automatically, and the algorithm is elegant enough to state in full:

- Start with a large $S$ (PyTorch default $2^{16} = 65{,}536$).
- Each step, scale the loss by $S$, backward, then **inspect the gradients for inf/NaN** (this is the unscale-and-check step). If the scale is too large, the scaled gradients overflowed and you will see inf/NaN.
- **If any gradient is inf/NaN:** the scale was too high. **Skip the optimizer step entirely** (do not corrupt the weights with garbage gradients) and **halve $S$** ($S \leftarrow S/2$).
- **If the gradients are finite:** apply the step. And if you have gone `growth_interval` steps (PyTorch default 2,000) with no inf/NaN, the scale might be unnecessarily conservative, so **double $S$** ($S \leftarrow S \times 2$) to push gradients even higher into the band.

The genius is the asymmetry: it backs off *fast* on the first sign of overflow (immediate halving, step skipped) and grows *slowly* (only after 2,000 clean steps). It rides the gradient distribution up to the edge of overflow without going over, automatically, with no tuning. And — this is the diagnostic gift — **the scale value over time is a readout of your gradient magnitudes**. A scale that climbs and stabilizes around, say, $2^{14}$ means "gradients are well-behaved, here is the headroom." A scale that keeps halving down to the floor means "gradients keep overflowing — something is producing huge values." We will read exactly that signal in section 5.

bf16 needs **no loss scaling at all**. Its exponent range matches fp32, so gradients of $3\times10^{-6}$ are represented natively without underflow, and activations of $80{,}000$ never overflow. This is bf16's headline advantage: you delete the entire GradScaler machinery and a whole class of bugs with it. You pay for it in the mantissa, which we will get to.

#### Worked example: choosing a static scale by hand

Suppose you are stuck on hardware where dynamic scaling is awkward and you must pick a static $S$. The principled procedure: run a handful of steps in fp32, collect the gradient histogram, and find the **maximum** absolute gradient $g_{\max}$ across all parameters. You want the largest scaled gradient to sit safely below fp16's overflow ceiling. Leave a margin — say you want $S \cdot g_{\max} \le 2^{14}$ (a factor of 4 below the $\approx 2^{16}$ ceiling, so a transient 4× spike won't overflow). If your measured $g_{\max} = 2.0$, then $S \le 2^{14}/2.0 = 2^{13} = 8{,}192$. Round down to a power of two: $S = 8{,}192$. Now check the bottom: the smallest gradient you care about, say $g_{\min} = 1\times10^{-7}$, becomes $S \cdot g_{\min} = 8{,}192 \times 10^{-7} \approx 8.2\times10^{-4}$, which is above fp16's $6.1\times10^{-5}$ floor — it survives. So $S = 8{,}192$ both avoids overflow at the top and rescues the small gradients at the bottom. The reason dynamic scaling is preferred is that $g_{\max}$ *grows* during a loss spike and *shrinks* as the model converges, so a static $S$ chosen at step 100 is wrong by step 10,000. Dynamic scaling re-derives this calculation every step, for free, by simply backing off whenever it sees an overflow.

## 4. The runnable AMP snippet, three formats

Here is the canonical mixed-precision training loop in modern PyTorch, written so you can copy it and run it. It shows all three modes — fp16 with the scaler, bf16 without, and the fp32 baseline — behind one flag, which is exactly the structure you want for debugging because you can bisect formats by changing one argument.

```python
import torch
from torch import nn

def make_amp_context(mode: str, device: str = "cuda"):
    """Return (autocast_dtype, use_scaler) for a given precision mode."""
    if mode == "fp16":
        return torch.float16, True       # fp16 NEEDS loss scaling
    if mode == "bf16":
        return torch.bfloat16, False     # bf16 needs NO loss scaling
    if mode == "fp32":
        return torch.float32, False      # baseline; autocast is a no-op
    raise ValueError(mode)

def train(model, loader, optimizer, mode="bf16", device="cuda"):
    autocast_dtype, use_scaler = make_amp_context(mode, device)
    # enabled=False makes GradScaler a transparent no-op for bf16/fp32
    scaler = torch.amp.GradScaler(device, enabled=use_scaler)

    model.train()
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        # autocast picks per-op precision; only forward+loss live inside it
        with torch.amp.autocast(device_type=device, dtype=autocast_dtype):
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)

        # fp16 path: scale loss -> backward -> (unscale+inf-check) -> step
        scaler.scale(loss).backward()

        # OPTIONAL but recommended: unscale before clipping so the clip
        # threshold is in true gradient units, not scaled units.
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)   # skips the step internally if grads are inf/nan
        scaler.update()          # adjusts the dynamic scale for next step

        if step % 50 == 0:
            print(f"step {step:5d} | loss {loss.item():.4f} | scale {scaler.get_scale():.0f}")
```

A few things to notice, because each is a bug people hit:

- `torch.amp.autocast` wraps **only the forward pass and the loss**, not the backward pass and not the optimizer step. autocast handles the backward dtype automatically; wrapping `backward()` or `step()` is a mistake.
- `scaler.scale(loss).backward()` is the loss-scale multiply. With `enabled=False` (bf16/fp32) it is a transparent pass-through, which is why the same loop serves all three modes.
- `scaler.unscale_(optimizer)` is called **before** `clip_grad_norm_` so that clipping sees true gradient magnitudes. Forgetting this means you clip *scaled* gradients and your `max_norm=1.0` is effectively `max_norm = 1.0 / S` — a silent, format-dependent clipping bug.
- `scaler.step(optimizer)` internally checks for inf/NaN and **skips the update** if found. That is why a NaN gradient does not immediately poison your weights under the scaler — the scaler protects you, then halves the scale. (Without the scaler, in raw fp16, the NaN goes straight into the weights.)
- `scaler.get_scale()` is your instrument. Logging it every 50 steps costs nothing and tells you the gradient story.

The one-line switch from a NaN-ing fp16 run to a stable bf16 run is literally `mode="fp16"` → `mode="bf16"`. That is the cheapest experiment in all of mixed-precision debugging and you should run it first whenever you see a numerics NaN.

### The same switch in the Hugging Face Trainer

You will rarely hand-write the loop above in production — you will be inside `transformers.Trainer` or `trl`'s `SFTTrainer`. The format switch is two booleans in `TrainingArguments`, and getting them right is the single most common AMP bug I see in finetuning code:

```python
from transformers import TrainingArguments

# fp16 path: Trainer wires up a GradScaler for you automatically.
args_fp16 = TrainingArguments(
    output_dir="out",
    fp16=True,            # enables fp16 autocast + an internal GradScaler
    bf16=False,
    # ... lr, batch size, etc.
)

# bf16 path: NO scaler is created (bf16 doesn't need one).
args_bf16 = TrainingArguments(
    output_dir="out",
    fp16=False,
    bf16=True,            # enables bf16 autocast, no GradScaler
)
```

Two rules that catch real bugs. First, **never set both `fp16=True` and `bf16=True`** — they are mutually exclusive autocast dtypes and the combination is a configuration error. Second, **match the format to your hardware**: setting `bf16=True` on a V100 (no hardware bf16) either errors or silently falls back to a slow software path; setting `fp16=True` on a model that overflows (some large LLMs) reproduces exactly the scale-collapse NaN from this post. The Trainer also exposes `torch_dtype` when you *load* the model — loading weights in bf16 (`torch_dtype=torch.bfloat16`) and then training with `bf16=True` is the standard large-model recipe, but loading in fp16 and training in bf16 (or vice versa) is a dtype-mismatch bug that produces subtle degradation. Keep the load dtype and the train dtype consistent.

To read the scaler when you are inside the Trainer (you do not own the loop), attach a callback:

```python
from transformers import TrainerCallback

class ScaleLogger(TrainerCallback):
    """Log the GradScaler value so you can see it stabilize or collapse."""
    def on_step_end(self, args, state, control, **kwargs):
        # Trainer keeps the scaler on the accelerator/trainer; access defensively.
        scaler = getattr(kwargs.get("trainer", None), "scaler", None)
        if scaler is not None and state.global_step % 50 == 0:
            print(f"step {state.global_step} | scale {scaler.get_scale():.0f}")
```

If you see that scale halving toward the floor inside a `SFTTrainer` run, you have the persistent-overflow signature, and the fix is the same one-liner: `fp16=False, bf16=True`.

## 5. Reading the GradScaler over time, the primary diagnostic

The dynamic loss scale is the most under-used instrument in mixed-precision training. People log loss and accuracy and never log the scale, then wonder why their fp16 run is unstable. Log the scale. Here is how to read it, with the shapes that mean different things, shown in figure 4 as a timeline.

![A timeline of the GradScaler value over training steps showing it climbing from the initial value to a stable plateau in a healthy run, contrasted with a run where the scale collapses by repeated halving toward the floor](/imgs/blogs/mixed-precision-debugging-fp16-vs-bf16-4.png)

**Healthy signature.** The scale starts at $2^{16}$, takes a few early halvings as the dynamic algorithm finds the ceiling (early gradients can be large during warmup), settles, then slowly climbs by doubling every 2,000 clean steps until it finds a stable plateau — often somewhere in $2^{12}$ to $2^{15}$. A scale that **stabilizes** means the algorithm found the headroom and your gradients live happily in the band. A few skipped steps early in training are normal and harmless.

**Persistent-overflow signature.** The scale keeps halving — $2^{16}, 2^{15}, 2^{14}, \dots$ — and never recovers, sometimes bottoming out near $1$ (PyTorch will not let it go below a minimum). **Constant halving means gradients are repeatedly overflowing to inf no matter how low you scale.** That is not a scaling problem you can tune away; it is a *real* exploding-gradient or bad-numerics problem — a too-high LR, a bad batch, a missing normalization, an activation that genuinely reaches $65{,}504$. The scaler is telling you the truth: your gradients are too big for fp16 even unscaled. When you see this, stop fiddling with the scaler and go find the explosion (cross-link the gradients-exploding-and-vanishing post). Frequently the right move is to switch to bf16, whose range absorbs the spikes.

Here is the code to log and interpret the scale, plus a per-layer gradient histogram to catch underflow that the scale alone will not show:

```python
import torch

@torch.no_grad()
def grad_diagnostics(model, scaler, step):
    scale = scaler.get_scale()
    # Collect TRUE (unscaled) grad magnitudes for the histogram.
    mags = []
    n_inf = n_zero = n_total = 0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        n_inf  += torch.isinf(g).sum().item() + torch.isnan(g).sum().item()
        n_zero += (g == 0).sum().item()
        n_total += g.numel()
        mags.append(g.abs().flatten())
    mags = torch.cat(mags)
    nonzero = mags[mags > 0]
    pctl = torch.quantile(nonzero, torch.tensor([0.01, 0.5, 0.99]).to(nonzero.device))
    print(
        f"step {step:5d} | scale 2^{int(round(torch.log2(torch.tensor(scale)).item()))} "
        f"| grad p1={pctl[0]:.2e} p50={pctl[1]:.2e} p99={pctl[2]:.2e} "
        f"| zero={100*n_zero/n_total:.1f}% inf/nan={n_inf}"
    )

# Call AFTER scaler.unscale_(optimizer) so grads are in TRUE units, BEFORE step.
```

Read the output like a doctor reads a chart. If `zero%` is climbing — say 40% of your gradient entries are exactly zero — you have **underflow**: a large fraction of gradients fell below the floor. The fix is more loss scaling (raise $S$) or bf16. If `inf/nan` is nonzero and the scale is collapsing, you have **overflow**: the fix is to find the explosion or switch to bf16. If `p1` (the 1st-percentile true gradient magnitude) sits below $6\times10^{-5}$ *and you are in fp16 without enough scale*, those gradients are dying. The histogram makes the section-1 prediction observable: it shows you the mass of the distribution relative to fp16's $6.1\times10^{-5}$ floor.

A subtlety that trips people up: this histogram is computed on **unscaled** gradients (note the comment — call it after `scaler.unscale_`). That matters because if you accidentally read the *scaled* gradients, every magnitude is inflated by $S = 65{,}536$, the `zero%` looks artificially low (loss scaling did its job and lifted everything off the floor), and you will wrongly conclude there is no underflow. The whole point of loss scaling is that the *fp16-stored* gradients are not underflowed; the question the histogram answers is whether the *true* gradients had mass below the floor that scaling rescued. So always unscale first. A clean way to think about it: the scale value $S$ and the `zero%` are two readings of the same phenomenon from opposite ends — $S$ tells you how far up you had to shift to avoid the floor, and `zero%` tells you how much still fell through. In a healthy fp16 run, $S$ is large enough that `zero%` stays low; when you cannot find an $S$ that keeps both the top (no inf) and the bottom (low `zero%`) happy at once, the gradient distribution is simply too wide for fp16's 5-bit exponent, and that is the precise, quantitative signal to switch to bf16. bf16's 8-bit exponent gives the distribution somewhere to live at both ends simultaneously, which is the whole reason it exists.

There is a second instrument worth wiring up: a **forward-activation range monitor** that catches the overflow at its *source* — the activation that exceeds $65{,}504$ — rather than downstream where it has already become a NaN. The scale trace tells you overflow is happening; this tells you *which layer's output* is too big, which is what you actually fix.

```python
import torch

def attach_activation_monitor(model, ceil=65504.0, near=0.5):
    """Forward hooks that flag any module whose fp16 output approaches the
    overflow ceiling. `near=0.5` flags outputs above 50% of the ceiling."""
    handles = []
    def hook(module, inp, out):
        if not isinstance(out, torch.Tensor) or not out.is_floating_point():
            return
        amax = out.detach().abs().max().item()
        if amax > near * ceil:
            pct = 100 * amax / ceil
            print(f"{module.__class__.__name__:24s} |out|max={amax:10.1f} "
                  f"({pct:5.1f}% of fp16 ceil)"
                  + ("  <-- WILL OVERFLOW" if amax >= ceil else ""))
    for m in model.modules():
        if len(list(m.children())) == 0:
            handles.append(m.register_forward_hook(hook))
    return handles  # call h.remove() for h in handles when done
```

Run this on a single forward pass under `autocast(dtype=torch.float16)`. A line like `Conv2d |out|max= 88000.0 (134.3% of fp16 ceil)  <-- WILL OVERFLOW` is your smoking gun: that layer's output exceeds fp16's range, and either you normalize it or you move to bf16. Combined, the scale trace, the gradient histogram, and the activation monitor give you the complete numerics picture — overflow (top of range), underflow (bottom of range), and the offending layer — from three short hooks.

#### Worked example: reading a real collapse

A 1.3B-parameter language-model finetune in fp16. Logs every 50 steps. The scale trace reads: `2^16, 2^16, 2^15, 2^15, 2^14, ... 2^3, 2^2, 2^1, 2^0` over about 600 steps, then sticks at the floor while loss flatlines at 6.2 (near the untrained entropy for the vocab). Diagnosis from the trace alone: persistent overflow — the scaler halved 16 times and still sees inf gradients, so the *unscaled* gradients are themselves overflowing. The `inf/nan` counter confirms dozens of inf entries per step concentrated in the final transformer block. Root cause turned out to be an un-normalized residual path whose activations reached $\sim 90{,}000$ in fp16. The fix that shipped: `mode="bf16"`. With bf16's range, those $90{,}000$ activations are ordinary numbers, no overflow, no scaler, and the loss resumed falling from 6.2 toward 1.8. Total debugging time once the scale was being logged: about ten minutes. Before logging the scale: the team had spent a day blaming the learning rate.

## 6. Which ops must stay in fp32, and the autocast audit

autocast does not run *everything* in low precision. It maintains an allow-list: matmuls and convolutions — the big, range-tolerant, compute-bound ops — run in fp16/bf16 because that is where the speed is. But a set of **precision-sensitive ops are forced back to fp32 automatically**, and understanding which and why is core to debugging "it's stable but the accuracy is slightly worse" complaints. The principle behind the allow-list is simple: an op is safe in low precision if it is *contraction-light and range-tolerant* (a matmul accumulates in fp32 internally on tensor cores even with 16-bit inputs, so its output is fine), and it is *unsafe* if it involves an unbounded function (exp, log) or a long accumulation (sum, mean, variance) where small errors compound. Once you internalize that split — bounded element-wise and matmul are fine, transcendental functions and long reductions are not — you can predict which custom op will misbehave before you even run the audit.

The ops that need fp32, and the reason:

- **Softmax and log-softmax.** Softmax computes $e^{x}$ and a sum. In fp16 the exponentials can overflow and the sum loses precision; the result feeds cross-entropy where small probability differences matter. PyTorch autocast runs softmax in fp32.
- **LayerNorm and BatchNorm.** Normalization computes a mean and a variance — a **reduction over many elements**. Reductions accumulate rounding error proportional to the number of terms; in bf16's 7-bit mantissa especially, a naive sum over 4,096 hidden units drifts. The mean and variance are computed in fp32.
- **The loss function.** Cross-entropy, especially with its log, is fp32. A `log(p)` for small `p` needs range and precision you do not have in fp16.
- **Large reductions / sums.** `tensor.sum()` over a big dimension, `mean()`, and similar accumulations. The error in a naive summation of $n$ terms grows like $O(n \cdot \epsilon)$, so a million-element bf16 sum can be off by a percent or more. Force these to fp32.
- **Exp, pow, and friends** in the loss or attention path — anything that can blow up range.

The summation-error claim deserves its derivation, because it is the single mechanism behind almost every bf16 "stable but worse" bug. When you sum $n$ numbers naively (a running accumulator, one add at a time), each addition introduces a relative rounding error of at most $\epsilon$. In the worst case these errors compound, and the bound on the total relative error of the sum is approximately

$$
\text{relative error} \lesssim (n-1)\,\epsilon.
$$

Now plug in numbers. Summing a hidden dimension of $n = 4096$ in bf16 ($\epsilon \approx 7.8\times10^{-3}$): the worst-case relative error is about $4095 \times 7.8\times10^{-3} \approx 32$ — that is *3200%*, total nonsense. The realistic (random-walk) error grows like $\sqrt{n}\,\epsilon \approx 64 \times 7.8\times10^{-3} \approx 0.5$, still a catastrophic 50%. In fp32 ($\epsilon \approx 1.2\times10^{-7}$) the same random-walk error is $64 \times 1.2\times10^{-7} \approx 8\times10^{-6}$ — negligible. *This* is why a LayerNorm or RMSNorm mean over thousands of channels must accumulate in fp32: a bf16 accumulator over the hidden dimension can be off by tens of percent, which silently shifts every normalized activation and costs you that mysterious half-point on the metric. (PyTorch's built-in norms and `sum` already accumulate in fp32 under autocast; the bug is in *custom* kernels that don't.) The fix in practice is either Kahan summation or — far simpler — just cast to fp32 before the reduction, which is what `autocast(enabled=False)` plus `.float()` does.

Figure 5 is the decision matrix: op class × whether it runs low-precision × the failure mode if you force it low × the fix.

![A matrix mapping operation classes like matmul and softmax and layernorm and loss against whether autocast keeps them low precision and the failure mode and the correct dtype to use](/imgs/blogs/mixed-precision-debugging-fp16-vs-bf16-5.png)

Most of the time autocast handles this for you. The bug appears when you **write a custom op outside autocast's allow-list**, or when you **disable autocast for a block and forget to restore fp32 for the reduction**, or when you do a manual reduction in your loss and leave it in the autocast dtype. The diagnostic is an **autocast-dtype audit**: register a forward hook that prints the output dtype of every leaf module, and look for a softmax, norm, or reduction that came out fp16/bf16 when it should be fp32.

```python
import torch

def audit_autocast_dtypes(model, sample_input, device="cuda",
                          dtype=torch.bfloat16):
    """Print the output dtype of every leaf module under autocast.
    Flags precision-sensitive ops that are NOT running in fp32."""
    SENSITIVE = ("softmax", "layernorm", "batchnorm", "groupnorm",
                 "rmsnorm", "crossentropy", "logsoftmax")
    handles = []

    def hook(module, inp, out):
        if not isinstance(out, torch.Tensor):
            return
        name = module.__class__.__name__.lower()
        is_sensitive = any(s in name for s in SENSITIVE)
        flag = ""
        if is_sensitive and out.dtype != torch.float32:
            flag = "  <-- SENSITIVE OP NOT IN fp32"
        print(f"{module.__class__.__name__:24s} -> {out.dtype}{flag}")

    for m in model.modules():
        if len(list(m.children())) == 0:        # leaf modules only
            handles.append(m.register_forward_hook(hook))

    model.eval()
    with torch.amp.autocast(device_type=device, dtype=dtype):
        model(sample_input.to(device))
    for h in handles:
        h.remove()
```

Run this once on a single batch. Every `LayerNorm -> torch.float32` is healthy; a `Softmax -> torch.bfloat16  <-- SENSITIVE OP NOT IN fp32` is your bug. The fix is to wrap the sensitive computation in `with torch.amp.autocast(enabled=False):` and cast its inputs to fp32 explicitly:

```python
# Inside a custom module's forward, force a reduction to fp32:
with torch.amp.autocast(device_type="cuda", enabled=False):
    x32 = x.float()
    mean = x32.mean(dim=-1, keepdim=True)
    var = x32.var(dim=-1, keepdim=True, unbiased=False)
    out = (x32 - mean) / torch.sqrt(var + self.eps)
return out.to(x.dtype)   # cast back to the autocast dtype for the next op
```

This is the bf16-specific failure mode in particular: bf16 almost never causes a NaN, so when a bf16 run is "stable but a half-point worse on the metric than the fp32 baseline," the culprit is nearly always a precision-sensitive reduction that drifted because of the 7-bit mantissa. The audit finds it.

Two more places the dtype leaks, both common enough to name. **Gradient checkpointing** recomputes activations in the backward pass; if the recomputation runs under a *different* autocast context than the original forward (a known footgun when you wrap checkpointed blocks by hand), the recomputed activations can be a different dtype than the cached ones, producing a subtle mismatch. The rule: the checkpointed function must run under the same autocast as the rest of the forward — `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` and let it inherit the ambient autocast. **Manual `.half()` or `.to(torch.float16)` calls** scattered in a model are the other leak: they force fp16 *outside* autocast's control, so autocast cannot promote a sensitive op back to fp32 because the tensor is already fp16. If you find `.half()` in a forward method, that is almost always a bug — let autocast manage the dtype, do not hard-cast. The audit hook above will surface both: a checkpointed block whose outputs flip dtype, or a sensitive op stuck in fp16 because an upstream `.half()` pinned it there.

## 7. fp16 vs bf16: the decision, and the bugs each owns

Now we can state the trade clearly, because we have the mechanism for both sides. We have seen all the pieces: the range floor and ceiling that fp16 crashes into (section 1), the master copy that protects small updates (section 2), the loss scaling that rescues fp16's underflow (section 3), the instruments that read the failure (section 5), and the reductions that must stay fp32 in either format (section 6). The decision is now mechanical rather than mystical. Figure 6 puts the formats head to head on the axes that decide a run.

![A before and after style comparison of fp16 needing loss scaling and risking overflow versus bf16 with wide range and no scaler but coarser mantissa precision](/imgs/blogs/mixed-precision-debugging-fp16-vs-bf16-6.png)

**Choose bf16 when you can.** On hardware that supports it (NVIDIA Ampere/Hopper and newer, TPUs natively, recent AMD), bf16 is the default for training large models for a simple reason: **it deletes the loss-scaling machinery and the entire overflow/underflow failure class.** No GradScaler, no skipped steps, no scale collapse, no $65{,}504$ ceiling to crash into. You trade away mantissa precision, but for the *stability* of training that almost never matters — gradient descent is robust to per-step rounding noise of one part in 128. This is why modern LLM pretraining (GPT-class, Llama-class) runs in bf16. The bf16 bug to watch is the precision-sensitive reduction from section 6, and the fix (fp32 reductions) is cheap.

**fp16 is the right tool when** your hardware predates bf16 support (Volta/Turing — the V100 and T4 do fp16 but not bf16 in hardware), or when you specifically need fp16's finer mantissa, or for *inference* where the values are bounded and the extra precision is free. fp16's bugs are the overflow NaN and the underflow plateau, and the fix is correct loss scaling (let dynamic GradScaler do its job) plus fp32 for the sensitive ops. fp16 training is entirely viable — it is how the original mixed-precision paper trained ImageNet and translation models to full accuracy — it just demands the scaler.

A practical note on **hardware**, because it decides the choice for you more often than theory does. bf16 in hardware arrived with NVIDIA Ampere (A100, RTX 30-series, 2020) and is present on everything since (Hopper H100, Ada, Blackwell), on Google TPUs from the start (bf16 is the TPU-native format), and on recent AMD Instinct (MI200+). fp16 tensor cores go back further, to Volta (V100, 2017) and Turing (T4, RTX 20-series). So the rule of thumb: **if you are on an A100/H100/TPU, default to bf16; if you are on a V100/T4, you only have fp16 and must run the scaler.** This is also why so much legacy training code is fp16-with-GradScaler — it was written for V100-era hardware — and why "just switch to bf16" sometimes is not an option and you genuinely have to debug the fp16 path. Check `torch.cuda.is_bf16_supported()` before you assume.

#### Worked example: the clip-before-unscale bug that silently disables clipping

A finetune with `clip_grad_norm_(model.parameters(), max_norm=1.0)` that was still occasionally diverging despite "having gradient clipping." The code clipped *before* `scaler.unscale_`. Walk the numbers: under loss scaling with $S = 2^{16} = 65{,}536$, the gradients in the graph are the *scaled* gradients, so their norm is roughly $65{,}536\times$ the true norm. A true gradient norm of $5.0$ appears as a scaled norm of $327{,}680$. The clip then compares $327{,}680$ against `max_norm=1.0` and scales everything down by a factor of $327{,}680$ — clipping far too aggressively, crushing the gradient to near nothing. Or, if you reason it the other way, your *intended* threshold of $1.0$ in true units is being applied as $1.0 / 65{,}536 \approx 1.5\times10^{-5}$ in true units after unscaling — effectively no useful clipping at the magnitude you wanted. Either way the clip is doing the wrong thing by a factor of the loss scale. The fix is the ordering in the section-4 snippet: `scaler.unscale_(optimizer)` first (bringing gradients to true units), *then* `clip_grad_norm_(..., 1.0)` (now a true threshold of 1.0), *then* `scaler.step`. After the fix, the clip fired correctly, the divergence stopped, and the grad norm sat at a healthy $\le 1.0$. The instrument that would have caught it instantly: print the grad norm immediately before and after `unscale_` — a 65,536× drop is the tell.

The diagnostic table you actually use at 2am:

| Symptom | Likely cause | Confirming test | Fix |
| --- | --- | --- | --- |
| Smooth loss, then sudden NaN | fp16 activation overflow > 65,504 → inf → NaN | Switch to bf16: does the NaN vanish? Log `inf/nan` count per layer | bf16, or normalize the offending activation; ensure scaler is on |
| Loss plateaus early, lower layers don't move | fp16 gradient underflow below 6.1e-5 → zero | Log `zero%` of grads; raise loss scale and watch it fall | More loss scaling, or bf16 |
| Scale halves to the floor and sticks | Persistent overflow: gradients explode even unscaled | Read `scaler.get_scale()` trace; it collapses, not stabilizes | Find the explosion (LR/init/norm); switch to bf16 |
| Stable but 0.3–0.8 pt worse than fp32 | bf16 coarse mantissa in a reduction (softmax/norm/sum) | autocast-dtype audit: a sensitive op outputs bf16 | Force that reduction to fp32 |
| NaN in raw fp16 with no scaler | No GradScaler → scaled grads overflow / no inf-check | Add `GradScaler`; or use bf16 | Use the scaler (or bf16) |
| `clip_grad_norm_` does nothing useful | Clipping scaled grads (forgot `unscale_`) | Print grad norm before/after `unscale_` | Call `scaler.unscale_()` before clipping |

## 8. The before→after evidence and a full bisection

Let me give you the concrete numbers, because "it got better" is not a debugging report. Here is the medical-imaging ResNet-50 from the intro, instrumented properly and bisected to root cause. Figure 7 is the bisection as a decision graph — the path from "NaN under AMP" to "fp16 overflow in the final block," with the confirming test on each edge.

![A decision graph that starts from a NaN under mixed precision and branches on whether the failure survives fp32 and whether the scale collapses or the zero percent climbs to reach overflow or underflow or a real math error](/imgs/blogs/mixed-precision-debugging-fp16-vs-bf16-7.png)

**The bisection.** Symptom: NaN at a wandering step under fp16 AMP. First, place the bug in one of six places. Make-it-fail-small: overfit a single batch. It overfits fine in fp32 (loss → 0.01 in 200 steps), which rules out **data**, **model code**, and **evaluation** — the model and pipeline are correct. The bug only appears under fp16 AMP, which points hard at **numerics**. Confirm by reading the instrument: log `scaler.get_scale()`. It collapses from $2^{16}$ toward the floor in the steps before each NaN — overflow signature, not underflow. Confirm the location: the per-layer `inf/nan` counter lights up in the final convolutional block, whose activations under fp16 reach the high tens of thousands. That is the whole diagnosis: numerics → fp16 overflow → final block. Two tests (overfit-one-batch in fp32, read the scale trace), no random hyperparameter changes.

The bisection graph generalizes beyond this one run, and it is the procedure to memorize. From a NaN-or-degradation under AMP, ask three questions in order. **(1) Does it survive fp32?** If yes, it is not a precision bug at all — it is a genuine math error (a `log(0)`, a divide-by-zero, a bad label) and you debug it like any other NaN. If no — it is precision-specific — proceed. **(2) Does the scale collapse, or does `zero%` climb?** A collapsing scale with `inf/nan` grads is **overflow**; a climbing `zero%` of gradients is **underflow**. **(3) Where?** The per-layer counter localizes overflow to a block; the per-layer histogram localizes underflow to the small-gradient layers (usually the lowest). Three questions, three instruments, root cause. The graph is small precisely because the decision is small once you read the right instruments.

**The fix and the measurements.** One flag, `fp16 → bf16`. Here is the instrument panel before and after:

| Metric | fp16 (buggy) | bf16 (fixed) | fp32 baseline |
| --- | --- | --- | --- |
| Outcome | NaN @ step ~1,840 | trains clean | trains clean |
| Final val accuracy | — (never finished) | 91.4% | 91.6% |
| `inf/nan` grads per step | 30–80 (final block) | 0 | 0 |
| GradScaler behavior | collapses to floor | n/a (no scaler) | n/a |
| Step time (A100) | 0.21 s | 0.20 s | 0.52 s |
| Peak memory | — | 11.3 GB | 19.8 GB |
| Speedup vs fp32 | — | 2.6× | 1.0× |

The accuracy delta between bf16 and fp32 is 0.2 points — inside run-to-run noise — confirming bf16 did not cost meaningful accuracy here. The speed and memory wins (2.6× faster, 43% less memory) are exactly why you wanted mixed precision in the first place. The NaN is gone because the activations that overflowed fp16's $65{,}504$ ceiling are ordinary numbers in bf16's range. This is the canonical mixed-precision debugging win: a one-flag change, justified by the science, verified by the instruments.

**The stress tests** — because a fix you cannot stress is a fix you do not understand:

- *What if it's data, not numerics?* Then overfit-one-batch would fail in fp32 too. It did not — fp32 is clean — so the bug is format-specific. A data bug does not care what precision you train in.
- *What if I have to stay on fp16 (V100, no bf16)?* Then keep dynamic loss scaling on, force the final block's reduction-heavy ops to fp32, and clip gradients (after `unscale_`). The scaler plus fp32-for-sensitive-ops recovers stability without bf16. Measured: scale stabilizes around $2^{11}$, no NaN, val accuracy 91.2%.
- *What if the batch is tiny?* Small batches mean noisier, occasionally larger gradients, so the dynamic scaler will skip more steps early. That is fine and self-correcting — a few skipped steps are not a bug. Only *persistent* collapse is.
- *What if it only fails on multi-GPU?* Mixed precision interacts with gradient all-reduce: under DDP the gradients are reduced across ranks, and if one rank produces an inf, the scaler's inf-check must see it consistently. PyTorch's `GradScaler` handles this, but a custom reduction can hide an inf from the check. The audit (count inf/nan *before* the all-reduce, per rank) localizes it. This is where a numerics bug masquerades as a systems bug.

## 9. Case studies and real signatures

These signatures recur across vision, language, and speech, because the bug lives in the *numbers*, not the modality. Figure 8 stacks the three pillars from the Mixed Precision Training paper that make low precision work, each annotated with the bug it prevents — read it top to bottom as the checklist a healthy AMP setup satisfies.

![A stack of the three mixed precision pillars showing fp32 master weights preventing swamped updates, loss scaling preventing gradient underflow, and fp32 reductions preventing precision drift, each layer feeding the next](/imgs/blogs/mixed-precision-debugging-fp16-vs-bf16-8.png)

**The Mixed Precision Training paper (Micikevicius et al., 2018).** The foundational reference. It introduced the three techniques this whole post rests on — the fp32 master copy of weights, loss scaling, and fp32 accumulation for reductions — and showed fp16 training matching fp32 accuracy across image classification (ImageNet), detection, language modeling, and translation, at roughly half the memory. The key empirical figure: gradient histograms with substantial mass below fp16's representable floor, which is precisely why loss scaling is necessary and not optional. If you read one source on this, read that paper; it is short and the histograms are worth the price of admission.

**bf16 as the LLM training default.** Large-model pretraining (the Llama family, many GPT-class models, PaLM on TPUs) standardized on bf16 rather than fp16 specifically to escape loss-scaling instability at scale. At billions of parameters and trillions of tokens, the cost of a scale collapse — wasted GPU-hours on skipped or NaN-ed steps — is enormous, and bf16's wide range makes the run boringly stable. The lesson generalized: when you have bf16 hardware and a big model, the default is bf16 and you stop thinking about scalers. The trade-off the field accepted is bf16's coarser mantissa, mitigated by fp32 reductions in the normalization and softmax layers. There is a deeper reason large models specifically favor bf16: the larger the model, the wider the *spread* of its activation and gradient magnitudes — attention logits, residual-stream norms, and the gradients flowing back through dozens of layers span more orders of magnitude than a small model's do. A wide-spread distribution is exactly what fp16's 5-bit exponent cannot contain (you cannot find a single loss scale that fits both ends), and exactly what bf16's 8-bit exponent handles effortlessly. So bf16's advantage *grows* with scale, which is why the format choice that is merely convenient for a small CNN becomes essentially mandatory for a 70B-parameter transformer. The mantissa cost, meanwhile, is amortized: averaged over billions of parameters and a long training run, the per-step rounding noise washes out, while a single overflow does not. Scale tilts the trade decisively toward range.

**The "stable but slightly worse" bf16 regression (a recurring pattern).** A frequently-reported signature: a model trained in bf16 converges fine but lands a fraction of a point below the fp32 baseline on a metric, and nobody can find a bug because nothing crashes. The cause is almost always a precision-sensitive reduction left in bf16 — a custom RMSNorm, a manual logit sum, an attention softmax in a hand-written kernel — drifting under the 7-bit mantissa. The autocast-dtype audit from section 6 finds it; forcing that one reduction to fp32 typically recovers the gap. This is the most common bf16 bug and the easiest to miss because there is no NaN to chase.

**The forgotten-scaler raw-fp16 NaN (a CV classic).** A vision finetune that someone "simplified" by deleting the `GradScaler` (it looked like boilerplate) and calling `loss.backward()` directly under `autocast(dtype=float16)`. Without the scaler there is no loss scaling (gradients underflow) *and* no inf-check skipping bad steps (a single overflow poisons the weights immediately). The signature is an almost-instant NaN, often within the first dozen steps. The fix is to put the scaler back — or switch to bf16, where no scaler is needed. The teachable point: under fp16, the GradScaler is not optional plumbing; it is load-bearing.

**The CTC `inf` loss that the scaler can't save (speech).** Connectionist Temporal Classification (CTC) — the loss behind many wav2vec2 and DeepSpeech-style ASR systems — computes the log-probability of all valid alignments between an input of length $T$ and a target of length $U$. When $T < U$ (the audio is shorter than the transcript), there is *no* valid alignment and CTC's loss is legitimately $+\infty$. Under fp16 this surfaces as `inf` loss, the GradScaler dutifully sees inf gradients, halves the scale, skips the step — and keeps doing so forever, because the inf is not a scaling problem, it is a *data* problem masquerading as one. The signature is a scale that collapses to the floor on specific batches (the short-audio ones), not all batches. The diagnostic that disambiguates: log which batches produce the inf; if it is always the batches with high `target_len / input_len`, the bug is data (resampling that shortened the audio, or a feature extractor with too large a hop length), not numerics. The scaler protected your weights but cannot fix a structurally-infinite loss. This is a clean example of the cross-link between mixed-precision symptoms and NaN-hunting: the instrument that catches it is the same per-step inf logger.

#### Worked example: an LLM finetune that degrades only in bf16

A 7B-parameter model finetuned in bf16. Trains clean, no NaN, loss curve textbook. But evaluation perplexity is 4.7 versus 4.4 for the fp32 baseline — a real, reproducible gap of 0.3 perplexity, roughly half a point of downstream accuracy, with nothing crashing. The bisection: it is **stable but worse**, which from the table points at a precision-sensitive reduction in bf16. Run the autocast-dtype audit on one batch. It reports the attention `Softmax -> torch.bfloat16` inside a custom fused-attention module that bypassed autocast's allow-list, and the final `RMSNorm -> torch.bfloat16` in a hand-written kernel. Both are reductions (softmax over the sequence, RMS over the hidden dim) running in 7-bit mantissa. The fix: wrap both in `with torch.amp.autocast(enabled=False)` and cast inputs to fp32, returning to bf16 after. Re-run: perplexity drops to 4.42, matching the fp32 baseline within noise, while keeping bf16's speed and memory everywhere else. Total cost of the fix: forcing two ops to fp32, a fraction of a percent of the compute. The lesson that spans modalities — vision softmax, speech CTC log-sum-exp, LLM attention softmax — is identical: **reductions belong in fp32, regardless of the modality or the autocast dtype around them.**

## 10. When this is (and isn't) your bug

Numerics bugs have a distinct fingerprint, and the discipline is to confirm the fingerprint before you blame the format — and equally to *stop* blaming the format when the fingerprint is absent.

**It probably IS a mixed-precision bug when:** the failure is **format-specific** — it appears under fp16/bf16 and vanishes in fp32 (the single most decisive test); the loss is **smooth then suddenly NaN** (overflow) rather than spiky-from-the-start; the `GradScaler` scale **collapses** or the `zero%` of gradients **climbs**; or a bf16 run is **stable but a fraction worse** than fp32 with a sensitive reduction in the low dtype. Switching `fp16 ↔ bf16 ↔ fp32` and watching the symptom move is the fastest bisection you have.

**It is probably NOT a mixed-precision bug when:** the run **NaNs in fp32 too** — then it is a real explosion (LR, init, a `log(0)` in your loss, a bad label), and the precision format is a bystander; the loss is **spiky and divergent from step one** at a high LR — that is optimization, not numerics (lower the LR first); **overfit-one-batch fails in fp32** — then the model or data pipeline is broken and no amount of dtype tuning will save it; or the metric is wrong but training is fine — that is an evaluation bug. A useful rule: **a smooth-then-NaN curve is numerics; a born-spiky curve is optimization; a NaN that survives the switch to fp32 is neither — it's a genuine math error in your code.** Bisect by format first; it is one flag and it tells you which of those three you are in.

A nuance worth stating because it confuses people: the **training** and **inference** precision questions are different, and a format that is wrong for one can be right for the other. fp16 inference is widely successful — serving a model in fp16 (or even int8) is standard — precisely because inference has no backward pass, so there are no tiny gradients to underflow and no loss scaling to manage; the forward activations are bounded and fp16's finer mantissa is a small win. The same model can be a nightmare to *train* in fp16 (gradient underflow, scale collapse) and perfectly happy to *serve* in fp16. So "we deploy in fp16 with no problems" is not evidence that fp16 *training* is fine — they are different regimes with different failure modes. If your training is unstable in fp16 but inference is clean, that is not a contradiction; it is the expected pattern, and the fix for the training side (bf16, or loss scaling) is independent of what you serve in. Keep the two questions separate and you will stop conflating a training numerics bug with a deployment one.

The other place people misattribute: blaming mixed precision for an **accuracy** problem that is actually a learning-rate or data problem. bf16 costs you essentially nothing in final accuracy on well-conditioned training; if your bf16 run is several points worse than fp32, suspect a reduction-in-low-precision bug (fixable) or, more likely, a different bug entirely (LR, data, eval) that you are pinning on the format because you changed the format recently. Recency is not causality. Confirm with the fp32 baseline.

A final piece of discipline that ties the whole post together: **keep an fp32 baseline run, always.** Mixed precision debugging is fundamentally a *differential* diagnosis — almost every confirming test in this post is "does the symptom move when I change the format?" That test only works if you have the fp32 reference to compare against. The fp32 run is slower and more memory-hungry, so people skip it; then when AMP misbehaves they have nothing to bisect against and are reduced to guessing. Run fp32 once, on a subset if you must, record its loss curve and final metric, and keep it. It is the single most valuable artifact for debugging a precision bug, because it answers the one question that splits the entire decision tree — is this the format, or is this real? — in a single comparison. The cost of one baseline run is trivial against the days a misattributed numerics bug can eat. Every confident mixed-precision debugger I know keeps that baseline; it is the reference signal against which the cheap one-flag bisections become decisive instead of suggestive.

## Key takeaways

- **Range is the exponent; precision is the mantissa.** fp16 (5 exp / 10 mant) has narrow range and finer precision; bf16 (8 exp / 7 mant) has fp32's wide range and coarser precision. Range bugs are crashes (NaN); precision bugs are quiet metric drift.
- **fp16 underflows gradients below $\approx 6.1\times10^{-5}$ to zero** (plateau signature) **and overflows activations above $65{,}504$ to inf→NaN** (sudden-wall signature). Both are predictable from the format, not the data.
- **Loss scaling shifts the gradient distribution up into fp16's empty top-of-range, then unscales before the fp32 master update.** Dynamic scaling backs off fast (halve on inf) and grows slow (double after 2,000 clean steps).
- **Read `scaler.get_scale()` over time.** A scale that **stabilizes** is healthy; a scale that **keeps halving to the floor** is a real exploding-gradient / overflow problem you cannot tune away — find the explosion or use bf16.
- **The fp32 master weight copy is load-bearing**, not optional: it preserves small updates that would be swamped when added to larger weights in low precision.
- **Force softmax, layernorm/batchnorm, the loss, and large reductions to fp32.** autocast does this for built-ins; your custom ops and manual reductions are where it leaks. The bf16 "stable but worse" bug is almost always a sensitive reduction left in bf16.
- **bf16 deletes the entire loss-scaling failure class.** If you have the hardware, bf16 is the default for training; its only watch-item is precision-sensitive reductions, fixed cheaply with fp32.
- **Bisect by format.** `fp16 ↔ bf16 ↔ fp32` is one flag. If the symptom survives fp32 it is not a precision bug; if it vanishes in bf16 it was an fp16 range bug. Two test runs localize it.
- **Unscale before you clip.** `scaler.unscale_(optimizer)` must precede `clip_grad_norm_`, or your clip threshold is silently divided by the scale.
- **Training and inference precision are separate questions.** fp16 serving working fine says nothing about whether fp16 *training* is stable — no backward pass means no underflow and no scaler to manage. Diagnose them independently.
- **Always keep an fp32 baseline run.** Mixed-precision debugging is a differential diagnosis; nearly every confirming test is "does the symptom move when I change the format?", which only works against a reference. The baseline is the cheapest insurance you can buy.

## Further reading

- Micikevicius et al., "Mixed Precision Training" (ICLR 2018) — the foundational paper: fp32 master weights, loss scaling, fp32 reductions, and the gradient histograms that justify them.
- PyTorch documentation, "Automatic Mixed Precision package — torch.amp" and the "Automatic Mixed Precision examples / recipe" — the authoritative API reference for `autocast` and `GradScaler`, including the op allow-list and the dynamic-scaling algorithm.
- NVIDIA, "Train With Mixed Precision" (Deep Learning Performance Guide) — practical guidance on fp16 vs bf16, tensor-core requirements, and which ops stay in fp32.
- Kalamkar et al., "A Study of BFLOAT16 for Deep Learning Training" (2019) — why bf16's range matches fp32 and why that makes loss scaling unnecessary.
- Within this series: the [taxonomy and decision tree of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) for placing a numerics bug among the six suspects; [hunting NaNs and Infs](/blog/machine-learning/debugging-training/hunting-nans-and-infs) for the by-step and by-layer NaN hunt; [gradients exploding and vanishing](/blog/machine-learning/debugging-training/gradients-exploding-and-vanishing) for the explosion behind a collapsing loss scale; [loss spikes and divergence](/blog/machine-learning/debugging-training/loss-spikes-and-divergence) for telling a transient spike from terminal divergence; and the [training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) capstone.
- For the related low-precision numerics of deployment, see [quantization from first principles](/blog/machine-learning/edge-ai/quantization-from-first-principles), which covers the integer-quantization analogue of the rounding and range trade-offs discussed here.

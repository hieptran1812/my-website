---
title: "Activation Checkpointing: Trading Compute for Memory to Fit Bigger Batches"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "The backward pass hoards every forward activation until it needs it, and at large batch or long sequence that stash is what OOMs your run. Learn to trade one extra forward pass for a 4-10x drop in activation memory, in real PyTorch."
tags:
  [
    "distributed-training",
    "multi-gpu",
    "activation-checkpointing",
    "pytorch",
    "fsdp",
    "memory-optimization",
    "gpu",
    "deep-learning",
    "ml-systems",
    "mixed-precision",
  ]
category: "machine-learning"
subcategory: "Distributed Training"
author: "Hiep Tran"
featured: true
readTime: 42
---

You have a 6.7-billion-parameter transformer and eight A100 80GB cards. The model state fits comfortably under FSDP — parameters, gradients, and Adam optimizer state come to about 13 GB per GPU after sharding across eight ranks, leaving roughly 67 GB of an 80 GB card free. So you set your micro-batch to eight sequences of 2,048 tokens, launch with `torchrun`, and watch the first step. It OOMs. The allocator dies trying to reserve activation memory on a card that had two-thirds of its HBM free a moment ago. You cut the micro-batch to four. It runs, but throughput is mediocre and MFU is stuck around 25%. You wanted batch eight for the arithmetic intensity, and the model *state* fits eight times over. What ate the other 67 GB?

The answer is the thing nobody budgets for until it OOMs them: **activations**. Every intermediate tensor your forward pass computes — the output of every layer norm, every projection, every GELU, every attention score matrix — has to be kept alive from the moment it is produced in the forward pass until the moment the backward pass consumes it to compute a gradient. For a deep model at a large batch or a long sequence, that stashed activation memory is not a rounding error on top of the weights. It is frequently the single largest consumer of HBM, larger than parameters, gradients, and optimizer state combined, and it is the wall you actually hit.

Activation checkpointing (also called gradient checkpointing or activation recomputation) is the sharpest tool we have for that wall, and it is almost free. The idea is disarmingly simple: instead of stashing *every* activation, stash only a few **checkpoints** — say, the input to each transformer block — and throw the rest away. When the backward pass needs the intermediates you threw away, re-run that block's forward to regenerate them just in time, use them, and discard them again. You pay one extra forward pass over the recomputed region — about a third more compute — and in return you drop activation memory from something proportional to the number of layers down to something proportional to *one* layer. That memory is what lets you run batch eight instead of batch four, or sequence 8,192 instead of 2,048, and the bigger batch is what buys back the MFU you were losing.

![A side by side comparison of stashing every forward activation across all layers versus keeping only block inputs and recomputing the rest during the backward pass](/imgs/blogs/activation-checkpointing-1.webp)

By the end of this post you will be able to look at a training run that OOMs, decide whether activations are the binding constraint, and reach for the right variant — none, selective, or full recomputation — with a defensible estimate of the memory you will save and the compute you will pay *before* you launch. We will derive the memory math (why full checkpointing turns a term proportional to the layer count $L$ into a term proportional to $\sqrt{L}$, and then to a single layer), write the real PyTorch (`torch.utils.checkpoint`, `checkpoint_sequential`, the modern `use_reentrant=False` API, `apply_activation_checkpointing` for FSDP, and selective recomputation), and walk two worked examples with measured before-and-after numbers on named hardware. This is the second lever in the memory chapter of the series; it composes directly with sharding from [FSDP in practice](/blog/machine-learning/distributed-training/fsdp-in-practice) and with the full accounting in [the memory budget](/blog/machine-learning/distributed-training/the-memory-budget). It is the fourth of the [four walls](/blog/machine-learning/distributed-training/why-distributed-training): the model — or here, the *batch* — that will not fit.

## Why the backward pass hoards activations

To understand why activations dominate, you have to understand why the backward pass cannot avoid keeping them. Backpropagation is the chain rule applied layer by layer, and the chain rule for a layer's weights involves the layer's *input*. Take a single linear layer $y = Wx$. The gradient with respect to the weights is $\partial L / \partial W = (\partial L / \partial y)\, x^\top$ — it multiplies the incoming gradient by the *forward input* $x$. There is no way around it: to compute the weight gradient, you need the activation $x$ that entered that layer during the forward pass. Same for the nonlinearities. The backward of a GELU or a softmax needs the forward output (or input) to evaluate the local derivative. So the default autograd behavior is exactly what you would expect: as the forward pass runs, every tensor that some backward will later need is retained in memory, held in the autograd graph, waiting.

That waiting is the whole problem. In a forward-then-backward step, the *first* layer's activation is produced early in the forward pass but not consumed until the very *end* of the backward pass — it must stay resident for the entire round trip. So at the moment the backward pass reaches the first layer, you are holding activations for essentially every layer at once. Peak activation memory is therefore proportional to the depth of the network times the per-layer activation size, and the per-layer size scales with the batch and the sequence length. Deep model, big batch, long sequence: all three multiply together, and none of them touch the weights.

![A vertical stack of the four consumers of GPU memory during training showing parameters, gradients, and optimizer state as fixed tiers and the activation tier as the variable one that swells with batch and sequence](/imgs/blogs/activation-checkpointing-2.webp)

The figure above is the memory budget from [the memory budget](/blog/machine-learning/distributed-training/the-memory-budget), redrawn to make one point: the first three tiers are *fixed* by the model. Parameters are $2\Psi$ bytes in bf16, gradients another $2\Psi$, and the fp32 Adam state — master weights plus two moments — is $12\Psi$, for a total of $16\Psi$ that FSDP then shards across ranks. Those tiers do not care about your batch size. The fourth tier, activations, is the variable one, and it is the one you control at run time. Double the batch and the first three tiers do not move; the activation tier doubles. This is why "the model fits but the batch does not" is such a common failure: you sized the card for the weights and forgot the tier that scales with throughput.

### The Megatron activation formula

Let us make "proportional to depth times batch times sequence" precise. The cleanest published accounting is from the Megatron-LM paper *Reducing Activation Recomputation in Large Transformer Models* (Korthikanti et al., 2022), which derives the activation memory stored per transformer layer, assuming 16-bit storage and no model parallelism, as approximately:

$$M_\text{layer} \approx sbh\left(34 + 5\,\frac{a\,s}{h}\right) = 34\,sbh + 5\,a\,s^2 b \;\text{ bytes}$$

where $s$ is the sequence length, $b$ the micro-batch size, $h$ the hidden dimension, and $a$ the number of attention heads. The total across $L$ layers is $L$ times this. The formula has two terms, and the split between them is the key to everything that follows:

- The $34\,sbh$ term is the **linear** part: the inputs stashed for the QKV projection, the attention output projection, the two MLP matmuls, the layer norms, and the dropout masks. It scales *linearly* in the sequence length.
- The $5\,a\,s^2 b$ term is the **attention** part: the $s \times s$ score matrix, its softmax, and the attention dropout. It scales *quadratically* in the sequence length, because the score matrix is $s \times s$ per head.

The constant 34 is architecture-dependent — it moves a little depending on how you count dropout masks (which are one byte, not two) and whether you fuse operations — so treat it as "about 34 for a standard GPT-style block," not gospel. What matters is the shape: linear-in-$s$ from the projections and MLP, quadratic-in-$s$ from attention. At modest sequence length the linear term dominates; past a few thousand tokens the $s^2$ term takes over and activations explode.

#### Worked example: where the 67 GB went

Take the model from the intro: a 6.7B GPT with hidden dimension $h = 4096$, $L = 32$ layers, and $a = 32$ heads, at sequence length $s = 2048$. First the linear term per layer at micro-batch $b = 1$:

$$34\,sbh = 34 \times 2048 \times 1 \times 4096 \approx 285\ \text{MB per layer}$$

Across 32 layers that is about 9.1 GB at batch one. At micro-batch eight it is eight times that: roughly 73 GB. Add the 13 GB of sharded model state and you are at 86 GB on an 80 GB card — the OOM from the intro, and it is entirely the linear activation term. (I am assuming a FlashAttention kernel here, which never materializes the $s \times s$ score matrix and so kills the $s^2$ term outright — more on that later; at $s = 2048$ the quadratic term would otherwise add a few more GB per layer. FlashAttention is, in a real sense, selective recomputation baked into a kernel, and we cross-link the mechanics in [kernel fusion and FlashAttention](/blog/machine-learning/high-performance-computing/kernel-fusion-and-flashattention-beating-the-memory-wall).)

So the 67 GB of free HBM was eaten by 73 GB of demanded activation memory, and you were 6 GB short. Cutting to batch four halved the activation term to about 37 GB, which fit — but at the cost of the arithmetic intensity you wanted. This is the exact situation activation checkpointing is built for.

## The trade: checkpoint a few, recompute the rest

Here is the move. Autograd stashes every activation because, by default, it assumes recomputing one is more expensive than storing it. For most of a transformer that assumption is wrong: the activations are large but the forward that produced them is cheap relative to storing them across the whole round trip. So we override the assumption. We wrap a region of the model — canonically, one transformer block — and tell autograd: *do not save my internal activations. Save only my input. When you need my internals for the backward, call my forward again to regenerate them.*

Mechanically, `torch.utils.checkpoint.checkpoint(fn, x)` runs `fn(x)` in the forward pass with autograd's graph recording turned *off* inside the region, so no internal tensors are retained — it keeps only the input `x` (and the output, which the next layer needs). Then, during the backward pass, when the gradient reaches this region, checkpoint re-runs `fn(x)` — this time *with* graph recording on — to rebuild exactly the internal activations it discarded, computes the local gradients against them, and frees them again. The block's internals live for the duration of one backward step for that block, and never longer.

![A left to right timeline of one checkpointed block showing the forward that discards internals, the wait until the backward reaches this block, the recompute forward that rebuilds them, and the backward that consumes and frees them](/imgs/blogs/activation-checkpointing-3.webp)

The timeline above is the whole mechanism in one row. In the forward pass, the block computes its output and discards its internal activations, keeping only the input tensor. Time passes; other layers run forward, then the backward pass unwinds. When the backward reaches this block, it recomputes the forward from the saved input, rebuilding the internals, then immediately runs the backward against them and frees them. The saved input is small — one tensor of shape $[s, b, h]$, the $2sbh$ bytes — while the discarded internals were the full $34\,sbh + 5as^2b$. That is the trade: keep the cheap boundary tensor, recompute the expensive middle.

### The compute cost, derived

How much extra compute does the recompute cost? Reason in units of forward FLOPs. Call one forward pass over a region $F$. The backward pass over a matmul costs about twice the forward — one matmul to produce the input gradient and one to produce the weight gradient, versus one matmul for the forward — so the backward is roughly $2F$. An ordinary training step over the region is therefore:

$$\text{step}_\text{normal} = F + 2F = 3F$$

With full activation checkpointing, you add exactly one recompute forward before the backward for each checkpointed region:

$$\text{step}_\text{checkpointed} = F + \underbrace{(F + 2F)}_{\text{recompute + backward}} = 4F$$

The overhead is $4F / 3F - 1 = 1/3 \approx 33\%$. That is the headline number for *full* recomputation: about a third more compute, because you do the forward twice and the backward once instead of the forward once and the backward once. In wall-clock terms the measured overhead is usually a bit *less* than 33% — often 20% to 30% — because the recompute forward is highly parallel and cache-friendly, some of it overlaps with communication in a sharded run, and not every last operation is inside the checkpointed region. But 33% is the right first-order estimate, and if your measured overhead is far above it you have checkpointed something you should not have.

Now weigh that against what it buys. For the intro model, full block checkpointing drops the resident activation term from about 73 GB to roughly 7 GB (we will do the accounting in a moment). You spend 33% more compute to reclaim 66 GB of HBM. On a memory-bound large-model run, that is one of the best trades in all of systems engineering — you are almost always memory-bound before you are compute-bound at scale, and the reclaimed memory lets you raise the batch, which raises MFU, which frequently *pays back* most of the 33% in improved hardware utilization. That is the crux: checkpointing is not "slower training." It is "trading a compute surplus you have for a memory deficit you cannot otherwise close," and the exchange rate is excellent.

### What autograd actually does with a saved tensor

It helps to be precise about the machinery, because it explains both why checkpointing works and where its edges are. When you run a forward pass under autograd, each differentiable operation that will need a forward value for its backward registers that value as a *saved tensor* on the grad function it builds. The layer norm saves its input and its computed statistics; the matmul saves the operands it needs to form the two gradients; the GELU saves its input. Those saved tensors are held by the autograd graph, and the graph is kept alive by the output tensor you eventually call `.backward()` on. That reference chain is why the activations survive the entire round trip: as long as the graph exists, its saved tensors exist, and the graph exists until the backward consumes it.

Checkpoint intervenes at exactly this point. Inside the checkpointed region it runs the forward under `torch.no_grad()`, so *no* grad functions are built and *no* saved tensors are registered — the region produces its output and its internals become ordinary tensors that Python garbage-collects the moment they go out of scope. Checkpoint keeps a reference only to the region's inputs. It then installs a custom autograd function whose backward, when reached, re-enables grad, replays the forward from the saved inputs to rebuild the grad functions and their saved tensors, runs the backward through them, and lets the whole recomputed subgraph be freed again. So the saved-tensor memory of the region exists only during that region's backward, not for the whole round trip. The non-reentrant implementation does this with `saved_tensors_hooks`, which is why it composes cleanly with the rest of autograd — it is using the same saved-tensor plumbing every other op uses, just with a pack hook that stores nothing and an unpack hook that recomputes.

## The memory math: from proportional to L, to root L, to one layer

The intro claimed activation memory drops "from proportional to the number of layers down to proportional to one layer." Let us derive that, because the intermediate result — the classic $\sqrt{L}$ scaling — is worth understanding, and it tells you *where* to place checkpoints when you cannot afford one per layer.

Without any checkpointing, you store the internal activations of all $L$ layers simultaneously at the peak of the backward pass, so activation memory is proportional to $L$. Now suppose you place $k$ checkpoints, evenly spaced, so the network is divided into $k$ segments of $L/k$ layers each. You store the $k$ segment-boundary activations (cheap — one tensor each). During the backward pass you process one segment at a time: to get the internals of a segment, you recompute its forward from its saved boundary input, which requires holding the full internal activations of *one segment*, or $L/k$ layers' worth. So peak memory is the $k$ saved boundaries plus one segment's internals:

$$M(k) \;\propto\; k + \frac{L}{k}$$

Minimize over $k$: $\frac{d}{dk}\left(k + L/k\right) = 1 - L/k^2 = 0$ gives $k = \sqrt{L}$, and the minimum memory is $2\sqrt{L}$. That is the celebrated result from Chen et al., *Training Deep Nets with Sublinear Memory Cost* (2016): checkpoint every $\sqrt{L}$ layers and activation memory drops from $O(L)$ to $O(\sqrt{L})$, at the cost of a single extra forward pass over the network. For a 32-layer model, $\sqrt{L} \approx 5.7$, so instead of 32 layers' worth of activations you hold about 11 — a roughly 3x reduction — by checkpointing at six boundaries.

![A branching diagram showing activation memory splitting into three regimes, stash everything proportional to the layer count, square root checkpointing, and full per block recomputation proportional to a single layer](/imgs/blogs/activation-checkpointing-4.webp)

The figure above places the three regimes side by side. The leftmost path stores everything and pays no recompute but scales as $L$. The middle path is the Chen $\sqrt{L}$ sweet spot: a modest number of checkpoints, memory as $\sqrt{L}$, one extra forward total. The rightmost path is what almost everyone actually does today: checkpoint *every* transformer block. That takes $k = L$, which by the formula above is not the memory minimum for the boundary-plus-segment model — but for a transformer the segment is a single layer, so the "one segment's internals" you hold during recompute is just one layer, and the saved boundaries are $L$ small input tensors of $2sbh$ each. In practice the per-block regime is the most convenient (the block is a natural, self-contained unit to re-run) and it collapses the dominant $34sbh \cdot L$ term to roughly $2sbh \cdot L$ for the saved inputs plus $34sbh$ for the one block being recomputed. That is why we say "proportional to one layer": the big per-layer constant loses its $L$ multiplier.

#### Worked example: the memory accounting, before and after

Here is the full per-GPU accounting for the intro model — 6.7B GPT, $h = 4096$, $L = 32$, $s = 2048$, micro-batch $b = 8$, FSDP `FULL_SHARD` across 8 A100 80GB, FlashAttention on so the $s^2$ term is already gone:

| Term | No checkpointing | Full block checkpointing |
|---|---|---|
| Sharded model state ($16\Psi/8$) | 13.4 GB | 13.4 GB |
| Saved block inputs ($2sbh \times L$) | included below | ~4.3 GB |
| Resident internal activations | ~73 GB (all 32 layers) | ~2.3 GB (one layer, during recompute) |
| **Activation total** | **~73 GB** | **~6.6 GB** |
| **Peak per GPU** | **~86 GB → OOM** | **~20 GB → fits** |

The model state term is identical — checkpointing does not touch parameters, gradients, or optimizer state. The entire difference is the activation term, which collapses from 73 GB to under 7 GB, an 11x reduction. Peak memory falls from an OOM-inducing 86 GB to a comfortable 20 GB, leaving 60 GB of headroom on the 80 GB card. You could now raise the micro-batch to 16 or 32, or push the sequence to 4,096, all on the same hardware. The 33% compute overhead is real, but the alternative was not "33% slower" — it was "does not run at all," or "runs at batch four with poor MFU."

### Partial checkpointing: only the layers you must

The example above checkpoints every block, but you do not have to. Checkpointing is a per-region decision, and the memory-versus-compute trade is nearly linear in the fraction of the model you checkpoint: checkpoint half the layers and you get roughly half the memory saving for roughly half the recompute overhead. That linearity is a lever. If full checkpointing frees 66 GB but you only needed 30 GB to fit the batch, you are overpaying — you are recomputing more than you have to.

The concrete tactic is to checkpoint only the *deepest* stretch of layers, because the deepest layers' activations are the ones held longest during the backward and therefore contribute most to the peak. Suppose our 32-layer model needs to shed 40 GB of activations to fit, and each block contributes about 2.3 GB of resident activation at the target batch. Checkpointing 18 of the 32 blocks sheds roughly $18 \times 2.3 \approx 41$ GB — enough — while leaving 14 blocks un-checkpointed and paying recompute on only 18/32 of the model, an overhead of about $0.33 \times 18/32 \approx 19\%$ instead of 33%. You fit the batch and you keep half the recompute you would otherwise have spent. In code this is just a predicate in the wrap loop:

```python
def forward(self, x):
    n = len(self.blocks)
    for i, block in enumerate(self.blocks):
        # Checkpoint only the deepest ~18 layers; keep the shallow ones cheap.
        if self.training and i >= n - 18:
            x = checkpoint(block, x, use_reentrant=False)
        else:
            x = block(x)
    return x
```

The right count is something you *measure*, not derive — sweep the number of checkpointed layers, watch `torch.cuda.max_memory_allocated()` cross under your card's capacity, and stop at the first count that fits with a safety margin. Checkpointing the minimum number of layers that fits the batch is the throughput-optimal choice, and it is almost always fewer than "all of them."

## Three settings: none, selective, or full

Full recomputation is not the only option, and it is often not the best one. The 33% overhead comes from recomputing *everything*, including the big MLP matmuls, which are expensive in FLOPs and only moderately expensive in memory. There is a smarter middle ground, and it is the one large-model training runs actually use: **selective activation recomputation**. The idea is to recompute only the operations that are *cheap in FLOPs and expensive in memory*, and to keep (stash) the operations that are *expensive in FLOPs and cheap in memory*. That is a Pareto optimization — you drop the memory-heavy, compute-light activations for almost no compute, and you keep the compute-heavy ones you would hate to redo.

For a transformer, the poster child for "cheap to recompute, expensive to store" is the attention score region: the $s \times s$ matrix, its softmax, and the attention dropout. That region produces the entire $5as^2b$ quadratic term — the dominant activation cost at long sequence — but its FLOPs are small relative to the MLP's four big matmuls. So selective recomputation stashes the inputs to attention and to the MLP (the matmul-relevant tensors), and recomputes only the softmax region. The Megatron paper reports this removes the $s^2$ term while adding only a couple of percent of compute overhead, versus roughly a third for full recomputation.

![A three by three comparison of no recomputation, selective recomputation, and full recomputation against the activation memory saved, the compute overhead added, and when to use each](/imgs/blogs/activation-checkpointing-5.webp)

The matrix above is the decision at a glance. None: zero overhead, zero memory saved — the default, correct only when activations already fit. Selective: recompute the softmax region, drop the quadratic $s^2$ term, pay only a few percent — the right default for long-context training and the modern standard. Full: recompute the whole block, drop nearly all activation memory, pay about a third — the tool when even selective is not enough, or when you want to push the batch as hard as possible. The three are not mutually exclusive across the model; you can apply full recomputation to some blocks and selective to others, or checkpoint only the deepest layers. But as a mental default: reach for selective first, and escalate to full only when selective still OOMs.

| Setting | Activation memory | Compute overhead | When to use |
|---|---|---|---|
| None | Full ($34sbh + 5as^2b$ per layer, all $L$) | 0% | Model + batch already fit with headroom |
| Selective (recompute softmax) | Drops the $5as^2b$ term | ~2-5% | Long sequence; the $s^2$ term is the problem; modern default |
| Full (recompute block) | ~one layer's worth | ~25-33% | Even selective OOMs; maximize batch on fixed HW |

## The code you actually write

Enough theory. Here is the real PyTorch, in the order you will reach for it. Everything below uses the modern, non-reentrant checkpoint API; we will explain why at the end of the section.

### The one-liner: `torch.utils.checkpoint`

The primitive is `torch.utils.checkpoint.checkpoint(function, *args)`. You wrap the forward call of a block. In a hand-written model loop it looks like this:

```python
import torch
from torch.utils.checkpoint import checkpoint

class TransformerBlock(torch.nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = MultiHeadAttention(dim, heads)
        self.mlp = MLP(dim)
        self.norm1 = torch.nn.LayerNorm(dim)
        self.norm2 = torch.nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Transformer(torch.nn.Module):
    def __init__(self, depth, dim, heads, use_checkpoint=True):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [TransformerBlock(dim, heads) for _ in range(depth)]
        )
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                # Discard this block's internal activations in the forward;
                # recompute them from x during the backward. use_reentrant=False
                # is the modern, better-behaved implementation (see below).
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x
```

That is the entire change. Note the `self.training` guard: at inference there is no backward pass, so there is nothing to recompute and checkpointing would only add cost — you want it *off* for evaluation and generation. Note also that we checkpoint at the *block* level, not per-operation: the block is the natural, self-contained unit whose input is small and whose internals are large, which is exactly the profile checkpointing wants.

### `checkpoint_sequential` for an `nn.Sequential`

If your model is an `nn.Sequential` (or you can express the checkpointed region as one), `checkpoint_sequential` divides it into a number of segments and checkpoints each — this is the direct implementation of the Chen $\sqrt{L}$ idea, where you choose the number of segments:

```python
from torch.utils.checkpoint import checkpoint_sequential

# layers is an nn.Sequential of, say, 32 blocks.
# Split into 8 segments: checkpoints at every 4th block, ~sqrt(L)-ish.
def forward(self, x):
    return checkpoint_sequential(self.layers, segments=8, input=x,
                                 use_reentrant=False)
```

`segments` is your $k$. Set it to `int(len(layers) ** 0.5)` for the Chen memory-optimal point, or to `len(layers)` to checkpoint every layer (full block recomputation). Fewer segments means less memory saved and less recompute; more segments means more of both. For most transformer training people bypass this and checkpoint every block, but `checkpoint_sequential` is the cleanest way to dial the trade if you want the $\sqrt{L}$ compromise.

### Composing with FSDP: `apply_activation_checkpointing`

In a real sharded run you are already inside FSDP, and you want checkpointing applied to the same units FSDP wraps — each transformer block. PyTorch gives you a helper that walks the module tree and wraps matching submodules with a checkpoint wrapper:

```python
import functools
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    apply_activation_checkpointing,
    CheckpointImpl,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# Non-reentrant checkpoint wrapper — the version that plays well with FSDP.
non_reentrant_wrapper = functools.partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)

# Wrap every LlamaDecoderLayer in the (already FSDP-wrapped) model.
apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=non_reentrant_wrapper,
    check_fn=lambda submodule: isinstance(submodule, LlamaDecoderLayer),
)
```

The `check_fn` is the selector — it fires on the block class, exactly parallel to the `transformer_auto_wrap_policy` you gave FSDP in [FSDP in practice](/blog/machine-learning/distributed-training/fsdp-in-practice). The ordering matters: apply activation checkpointing to the decoder-layer modules, and let FSDP wrap the same layers, so that each block is both a sharding unit and a recomputation unit. With FSDP2 (`fully_shard`), the same `checkpoint_wrapper` approach works — you wrap the block modules with the checkpoint wrapper, then call `fully_shard` on them.

The composition is what fits a 70B model on a single 8-GPU node, and the accounting shows why neither lever alone suffices. Take LLaMA-70B ($h = 8192$, $L = 80$, $s = 4096$) on 8x H100 80GB (989 bf16 TFLOP/s, 3.35 TB/s HBM). Model state is $16 \times 70\text{B} = 1120$ GB; sharded across 8 ranks that is 140 GB per GPU — already over an 80 GB card, so sharding is mandatory just for the state, and you would reach for CPU or NVMe optimizer offload, or a second node, to seat it. Assume you have the state resident at, say, 70 GB per GPU through offload of the optimizer moments. That leaves only about 10 GB for activations. Without checkpointing, one micro-batch of $s = 4096$ across 80 layers of a $h = 8192$ model demands hundreds of GB of activations — a factor of tens over budget. Full block checkpointing collapses that to roughly one block's worth plus the saved inputs, on the order of 8-12 GB, which fits the 10 GB you had. Sharding made the *state* fit across ranks; checkpointing made the *activations* fit within a rank; the launch is unremarkable once both are in place:

```bash
torchrun --nproc_per_node=8 --nnodes=1 \
  --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 \
  train.py --model llama-70b --seq-len 4096 --micro-batch 1 \
  --fsdp full_shard --activation-checkpointing full \
  --optimizer-offload cpu
```

Turn off either `--fsdp full_shard` or `--activation-checkpointing full` and this run OOMs — the state does not fit without sharding, and the activations do not fit without recomputation. They are complementary levers on orthogonal tiers of the memory budget, which is why every large-model recipe uses both.

### Selective recomputation via `context_fn`

Recent PyTorch exposes selective activation checkpointing (SAC) through a `context_fn` argument to `checkpoint`, which lets you supply a *policy* deciding, per operation, whether to save its output or recompute it. The idiom is to save the outputs of the expensive matmuls (so they are not redone) and let everything else be recomputed:

```python
import torch
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts
from torch.utils.checkpoint import CheckpointPolicy

# Ops we want to SAVE (not recompute) because they are FLOP-expensive.
_SAVE_LIST = {
    torch.ops.aten.mm.default,
    torch.ops.aten.addmm.default,
    torch.ops.aten.bmm.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
}

def policy_fn(ctx, op, *args, **kwargs):
    if op in _SAVE_LIST:
        return CheckpointPolicy.MUST_SAVE      # keep matmul outputs stashed
    return CheckpointPolicy.PREFER_RECOMPUTE   # recompute the cheap elementwise ops

def selective_context():
    return create_selective_checkpoint_contexts(policy_fn)

def forward(self, x):
    # Recompute the cheap softmax/elementwise region; keep the big matmuls.
    return checkpoint(self.block, x, use_reentrant=False,
                      context_fn=selective_context)
```

This is the API-level expression of the Pareto idea: `MUST_SAVE` the matmuls (the compute-heavy, memory-light ops you do not want to redo), `PREFER_RECOMPUTE` everything else (the softmax, the layer norms, the activations — cheap to redo, expensive to keep). The exact op names in the save list depend on your model and your PyTorch version; inspect the traced ops or start from a known-good list for your architecture. When you cannot or do not want to hand-write a policy, `torch.compile` will do a version of this automatically — its min-cut partitioner chooses, per the graph, which activations to save and which to recompute to minimize memory subject to the compute it adds, which is selective recomputation solved as an optimization problem rather than a hand-tuned policy.

The reason a per-op policy is worth the trouble is that operations differ enormously in their memory-to-FLOP ratio, and the good policy saves the low-ratio ops and recomputes the high-ratio ones. Here is the ranking for a transformer block that makes the decision obvious:

| Operation | Activation memory | Recompute FLOPs | Policy |
|---|---|---|---|
| QKV / output / MLP matmuls | Low (output tensor only) | High (the block's dominant FLOPs) | Save |
| Attention scores ($s \times s$) | High (quadratic in $s$) | Low (elementwise after the matmul) | Recompute |
| Softmax + attention dropout | High (two $s \times s$ tensors) | Very low | Recompute |
| GELU / SiLU activation | Medium (one full tensor) | Very low (pointwise) | Recompute |
| Layer norm | Low (input + stats) | Low | Recompute |

Read down the "policy" column: you save exactly one row, the matmuls, and recompute everything else. That single row is where nearly all the FLOPs live and almost none of the memory pressure at long sequence; the recomputed rows are where the memory pressure lives and almost none of the FLOPs. Saving the matmuls and recomputing the rest is why selective costs a couple of percent instead of a third — you never redo the expensive part.

### Measuring it honestly

Never trust an estimate you can measure. Here is the timing-and-memory harness that gives you the real overhead and the real savings, with the two things people always forget — a warm-up, and a `torch.cuda.synchronize()` before you read the clock:

```python
import torch, time

def measure(model, batch, steps=20, warmup=5):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    torch.cuda.reset_peak_memory_stats()
    for i in range(warmup + steps):
        if i == warmup:
            torch.cuda.synchronize()          # let warm-up finish
            t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss = model(batch).float().mean()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()                  # CUDA is async — sync before timing
    dt = (time.perf_counter() - t0) / steps
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    return dt, peak_gb
```

The `warmup` skips the first few steps, which include cuDNN autotuning, allocator warm-up, and CUDA-graph capture noise. The two `synchronize()` calls are non-negotiable: CUDA kernels launch asynchronously, so without a sync you are timing Python's kernel-launch loop, not the GPU work, and you will report a step time that is a fraction of the real one. Run this with `use_checkpoint=False` and `use_checkpoint=True` on the same batch and you get the true overhead-versus-savings trade for your model, on your card, in about a minute.

Two confounds will lie to you if you let them. The first is the **data loader**: this harness feeds the same in-memory `batch` every step precisely so that data loading never enters the measurement, but in a real run a starved loader can make a checkpointed step look *cheaper* than it is, because the GPU is idling on data anyway and the recompute hides in the gap. Measure the compute overhead with the loader out of the picture, then confirm end-to-end throughput separately. The second is **thermal and clock throttling**: a checkpointed run does more work per step and runs the GPU hotter, so a card that boosts to a high clock in a short benchmark may throttle to a lower sustained clock over a long run, inflating the *apparent* overhead of checkpointing. Measure at steady state — dozens of steps in, after clocks have settled — not on the first hot burst. The honest number is the steady-state tokens per second under the real loader, not the peak-clock step time of an isolated micro-benchmark.

## Results on named hardware, and how to read them

Let us put numbers on both variants with the running example, and then stress-test the decision.

#### Worked example: full checkpointing turns an OOM into a run

Model 6.7B GPT ($h = 4096$, $L = 32$), 8x A100 80GB SXM (312 dense bf16 TFLOP/s, 2.0 TB/s HBM), FSDP `FULL_SHARD`, FlashAttention, $s = 2048$:

| Configuration | Micro-batch | Peak mem/GPU | Step time | Tokens/s/GPU | MFU |
|---|---|---|---|---|---|
| No checkpointing | 8 | ~86 GB | — (OOM) | — | — |
| No checkpointing | 4 | ~50 GB | 0.42 s | ~19,500 | ~26% |
| Full checkpointing | 8 | ~20 GB | 1.10 s | ~29,800 | ~40% |
| Full checkpointing | 16 | ~34 GB | 2.15 s | ~30,500 | ~41% |

Read the table carefully, because it makes the counterintuitive point. Going from no-checkpointing-batch-4 to full-checkpointing-batch-8, the *step time roughly doubles* — but that is because the batch also doubled, so the per-step work doubled *and* you added the 33% recompute. The number that matters is tokens per second, and it went *up*, from 19,500 to 29,800 per GPU, because batch eight has far better arithmetic intensity than batch four: the GPU spends more of its time in big, efficient matmuls and less in launch overhead and memory-bound tails. MFU climbs from 26% to 40%. The compute you spent on recomputation was more than repaid by the compute you *stopped wasting* on an under-fed pipeline. That is the pattern to internalize: checkpointing is not a throughput tax, it is a throughput *enabler*, because the memory it frees buys a batch that runs the hardware harder.

#### Worked example: selective at long context

Now the long-context case, where the $s^2$ term is the enemy. Same model, but sequence length $s = 8192$ and micro-batch $b = 1$, and this time assume a stack that materializes the attention scores (no FlashAttention) so we can see the quadratic term. The attention activation per layer is:

$$5\,a\,s^2 b = 5 \times 32 \times 8192^2 \times 1 \approx 10.7\ \text{GB per layer}$$

That is per layer. Across 32 layers the attention term alone is over 340 GB — utterly impossible on any single card. This is why nobody materializes attention at long context: you either use FlashAttention (recompute the softmax inside the kernel, never storing the matrix) or you use selective activation recomputation (stash the attention inputs, recompute the softmax region), and both remove the same $s^2$ term. Selective recomputation of the softmax region costs a few percent of compute — the softmax and dropout are cheap FLOPs — and removes the entire 340 GB. The linear $34sbh$ term remains and is handled by block-level checkpointing if needed. The Megatron paper reports selective recomputation adding on the order of a couple of percent overhead versus roughly 33% for full, for large GPT-scale models, which is why it, not full, is the default in production large-model training.

![A dataflow of one transformer block with the matmul nodes marked as saved and the attention softmax and activation nodes marked as recomputed, following the residual path from input to output](/imgs/blogs/activation-checkpointing-7.webp)

The figure above traces the dataflow of one block and marks each node as saved or recomputed under the selective policy. The matmul nodes — the QKV and output projections, and the two MLP matmuls — stay saved, because they carry nearly all the block's FLOPs and only their compact output tensors as memory. The attention softmax region and the pointwise activation are marked recomputed, because they carry the big quadratic and full-tensor memory but almost no FLOPs. The residual path runs straight through both. Most of the memory dropped, a small fraction of the compute added — that is the selective policy drawn as a graph.

### Checkpointing and gradient accumulation: two knobs on the same batch

There are two ways to grow the effective batch size without growing peak activation memory, and they are worth keeping straight because people confuse them. Activation checkpointing lets each *micro-batch* be larger by shrinking the memory each micro-batch costs. Gradient accumulation lets you run *more micro-batches* per optimizer step by summing their gradients before you call `optimizer.step()`, so the effective batch is the micro-batch times the accumulation count, while the activation memory only ever pays for *one* micro-batch at a time.

They compose cleanly, and together they fully decouple your effective batch from your card's capacity. Checkpointing sets how big one micro-batch can be; accumulation sets how many of those micro-batches sum into one update. Want an effective batch of 512 sequences on a card that fits a micro-batch of 8 with checkpointing? Accumulate over 64 micro-batches:

```python
accum_steps = 64
opt.zero_grad(set_to_none=True)
for i, micro_batch in enumerate(loader):     # each is 8 sequences
    loss = model(micro_batch).float().mean() / accum_steps   # scale for the mean
    loss.backward()                          # grads accumulate in .grad
    if (i + 1) % accum_steps == 0:
        opt.step()
        opt.zero_grad(set_to_none=True)
```

Two details matter. First, divide the loss by `accum_steps` so the accumulated gradient is the *mean* over the effective batch, not the sum — otherwise your effective learning rate scales with the accumulation count. Second, in a distributed run, the gradient all-reduce fires on every `backward()` by default, so naive accumulation all-reduces 64 times per step when it only needs to once. Wrap the non-final micro-batches in `model.no_sync()` (available on both DDP and FSDP) to suppress the reduction until the last micro-batch, then reduce once. The interaction with checkpointing is nil — each micro-batch's forward is checkpointed independently, its activations freed before the next micro-batch begins — which is exactly why the pair composes: checkpointing bounds the per-micro-batch memory, accumulation stacks micro-batches in time rather than in memory, and the effective batch becomes a free parameter.

## Case studies and real numbers

A few load-bearing results from the literature, so you are calibrated to what production runs actually do:

- **Megatron-LM selective recomputation** (Korthikanti et al., 2022). The paper that formalized selective activation recomputation reports that, for models up to 530B parameters, selective recomputation removes the attention $s^2$ activation term with an overhead of roughly a couple of percent, versus about 30-40% for full recomputation. Their headline is that you almost never want *full* recomputation once selective is available — you pay an order of magnitude less compute for most of the memory. This is the single most important practical takeaway of the whole topic.
- **Chen et al., sublinear memory** (2016). The original gradient-checkpointing paper. It established the $O(\sqrt{L})$ memory result for a chain of $L$ layers at the cost of one extra forward pass, and it is the reason "checkpoint every $\sqrt{n}$ layers" is folklore. Modern transformer training usually checkpoints per block instead (more convenient, and the block is the natural unit), but the $\sqrt{L}$ math is the right way to reason about *partial* checkpointing.
- **FlashAttention** (Dao et al., 2022). Not usually described as activation checkpointing, but it is exactly that for attention: it never materializes the $s \times s$ score matrix, recomputing the necessary pieces inside the kernel during the backward pass from the saved $Q$, $K$, $V$. It removes the same $s^2$ term selective recomputation targets, which is why with FlashAttention on, the quadratic activation term simply does not appear in your memory budget. Mechanics in [kernel fusion and FlashAttention](/blog/machine-learning/high-performance-computing/kernel-fusion-and-flashattention-beating-the-memory-wall).
- **Large-model training recipes** generally. GPT-3, Megatron-Turing NLG 530B, the LLaMA and OLMo families — essentially every large-model training report uses activation recomputation of some form, because at their batch and sequence lengths activations are the binding memory constraint and recomputation is the only way to make the batch large enough for good MFU. When a report says "we used activation checkpointing," this is what they mean, and it is why their MFU numbers are achievable at all.

## Activation offloading: the other way to trade

Recomputation trades *compute* for memory. There is a second, complementary trade that spends a different resource: **activation offloading** moves saved activations off the GPU to CPU host memory (or NVMe) while they wait for the backward, then moves them back when the backward needs them. It trades *PCIe bandwidth* for memory instead of compute — you keep the activation, you just do not keep it in HBM.

The two levers attack the same tier from opposite directions, and which one wins depends on which resource you have to spare:

| | Recomputation | Offloading |
|---|---|---|
| Resource spent | GPU compute (extra forward) | PCIe / NVMe bandwidth |
| Overhead | ~2-33% of step time | Depends on PCIe vs recompute cost |
| Correctness risk | RNG replay (handled by default) | None (exact tensor preserved) |
| Best when | Compute-rich, memory-poor | Bandwidth-rich, recompute-expensive |

PyTorch exposes the primitive as `torch.autograd.graph.save_on_cpu()`, a context manager that packs saved tensors to CPU and unpacks them back on demand; FSDP exposes it as an `offload_to_cpu` option on its checkpoint wrapper. The catch is the same one that governs all offloading: the round trip over PCIe (about 32 GB/s on Gen4, 64 GB/s on Gen5) is slow relative to HBM (2-3.3 TB/s), so unless you overlap the copy with computation, offloading can be *slower* than just recomputing. The rule of thumb: recompute when the activation is cheap to regenerate (a transformer block — regenerating it is a fast forward pass); offload when the activation is expensive to regenerate but you have idle PCIe bandwidth and a copy engine to overlap the transfer. In practice most large-model training uses recomputation as the primary lever and reserves offloading for the specific tensors that are both large and expensive to recompute, or for squeezing a run onto a card that is a few GB short after checkpointing has already done its work. The mental default is: recompute first, offload only what recompute cannot cheaply cover.

## When to reach for it (and when not)

Activation checkpointing is close to a free lunch for large-model training, but "close to" is not "is," and the overhead is real. Here is the decision, drawn as a tree.

![A decision tree for choosing no recomputation, selective recomputation, or full recomputation based on whether the batch fits and which activation term dominates](/imgs/blogs/activation-checkpointing-6.webp)

The tree above encodes the policy. Start by asking whether the model and your desired batch fit with headroom. If they do — a small model, a short sequence, a card with room to spare — do nothing. Checkpointing would spend compute to reclaim memory you are not short of, and that is pure waste; this is the one case where activation checkpointing is a *mistake*. If you are short on memory, ask what dominates. If it is the attention $s^2$ term (long sequence, materialized attention), reach for FlashAttention or selective recomputation of the softmax region — a few percent of compute for the whole quadratic term. If activations are large across the board (deep model, big batch, even with the $s^2$ term already handled), escalate to full block recomputation for its 4-10x memory drop at about a third more compute. And if even full recomputation plus sharding still OOMs, that is your signal to add another axis — offload, tensor parallelism, or pipeline parallelism from [pipeline parallelism and the bubble](/blog/machine-learning/distributed-training/pipeline-parallelism-and-the-bubble) — because you have exhausted what a single memory-saving lever can do.

The stress tests, because a decision is only as good as its edge cases:

- **What if you are already compute-bound?** Then do *not* checkpoint anything you do not have to. If your run is at high MFU with memory to spare, the 33% recompute is a direct throughput hit with no memory payoff. Checkpointing pays only when the freed memory buys a bigger batch or a longer sequence that you actually want; if you cannot use the memory, do not spend the compute.
- **What about inference?** There is no backward pass at inference, so there are no activations to stash and nothing to recompute — checkpointing is irrelevant and, if left on, pure overhead. Guard it with `self.training`, as in the code above. This is a real bug people ship: a model checkpointed unconditionally that runs generation 33% slower for no reason.
- **What at very small models?** A 125M model at batch 32 and sequence 1024 fits easily; checkpointing it wastes compute. The technique earns its keep as the model, batch, or sequence grows large enough that activations threaten the card, which for modern training is most of the time — but not the tiny-model regime.
- **What if the recompute breaks something?** It can, and the culprit is almost always randomness — dropout and any RNG-driven op inside the recomputed region. If the recompute draws *different* random numbers than the original forward, the gradients are computed against activations that never existed in the forward, and your training is subtly wrong. This is important enough to get its own section.

### The RNG gotcha, and why the default handles it

Consider a checkpointed block with dropout. In the forward pass, dropout samples a random mask and applies it. In the backward pass, checkpoint recomputes the forward — including the dropout — to rebuild the activations. If that recompute samples a *fresh* random mask, the recomputed activations differ from the originals, and the gradient is wrong: you would be differentiating through a forward that never happened. The fix is that the recompute must reproduce the *exact same* random draws as the original forward.

PyTorch's `checkpoint` handles this by default: it saves the RNG state (both CPU and the current CUDA device's generator state) at the start of the forward region and restores it before the recompute, so dropout and every other stochastic op replay identically. This is controlled by `preserve_rng_state=True`, which is the default — leave it on unless you have a very specific reason not to. The cost is small: saving and restoring the generator state is cheap, though on some setups it forces a tiny synchronization. If you ever hand-roll your own recomputation (please do not — use the library), this is the trap: forget to save and restore the RNG and you get a silently wrong model that trains, converges to something, and is subtly off. Determinism across the whole run — seeds, data order, deterministic kernels — is its own discipline; see [reproducibility and determinism in training](/blog/machine-learning/debugging-training/reproducibility-and-determinism-in-training) for the full treatment.

### Reentrant versus non-reentrant, and other sharp edges

A few more edges worth knowing, because they turn into confusing errors:

- **`use_reentrant=False` is the modern path, and you should use it.** The original checkpoint implementation (`use_reentrant=True`, the historical default) works by re-entering autograd, which brings a pile of restrictions: it requires at least one input to have `requires_grad=True` (or the backward silently does nothing), it does not compose cleanly with `torch.autograd.grad`, it has trouble with double-backward, and it can mishandle outputs that do not require grad. The non-reentrant implementation (`use_reentrant=False`) fixes all of these — it supports double-backward, handles non-grad inputs, and interacts correctly with the rest of autograd. PyTorch now warns if you do not pass the argument explicitly; pass `use_reentrant=False`.
- **Checkpointing too little wins nothing.** If you wrap a module whose internal activations are small relative to its input and output, you pay the recompute and save almost no memory. Checkpoint the units whose *internals* dwarf their *boundaries* — transformer blocks, exactly.
- **Checkpointing too much wastes compute.** Wrapping the entire model as one giant checkpoint means the backward must recompute the whole forward before it can start, holding a full forward's worth of activations at the peak of the recompute — you get maximal memory savings but you have serialized a full extra forward with no overlap. Per-block granularity keeps the recompute local and lets each block's recompute overlap with the previous block's backward comms.
- **`torch.compile` interaction.** Checkpointing and `torch.compile` can both decide what to recompute, and they can fight. The clean path is to let `torch.compile`'s min-cut partitioner make the recomputation decision on the compiled graph, or to use the compile-aware checkpoint so the two agree. If you see recompute happening twice, or unexpected memory, this overlap is the first place to look.
- **Non-reentrant keeps a little more alive.** The non-reentrant implementation may retain a few tensors that are cheap to keep and expensive to recompute (an early-stop optimization), so its memory savings can be marginally less than the theoretical maximum. This is a feature — it avoids recomputing things that were not worth recomputing — but it means your measured savings may be slightly below the back-of-envelope. Measure with the harness above.

## Key takeaways

- **Activations are stashed because the backward needs them.** The weight gradient for a layer needs that layer's forward input, so autograd retains every activation from its production until its consumption — and at the backward's peak that is every layer at once, proportional to depth times batch times sequence.
- **Activation memory is frequently the binding constraint, not the weights.** The model state ($16\Psi$, sharded) is fixed; the activation tier scales with your throughput knobs. "The model fits but the batch does not" is the activation tier talking.
- **The trade is one extra forward for a large memory drop.** Full recomputation turns a training step from $3F$ into $4F$ — about 33% more compute — and collapses the dominant activation term from $L$ layers' worth to one layer's worth, a 4-10x reduction in practice.
- **The classic result is $\sqrt{L}$.** Checkpoint every $\sqrt{L}$ layers and memory drops from $O(L)$ to $O(\sqrt{L})$ (Chen et al.). Per-block checkpointing goes further, collapsing the big per-layer constant's $L$ multiplier to one.
- **Selective beats full for most large-model training.** Recompute the cheap-FLOP, expensive-memory softmax region and keep the expensive-FLOP matmuls: you drop the quadratic $s^2$ term for a couple of percent of compute, versus a third for full recomputation.
- **The memory you free buys MFU.** A larger batch has better arithmetic intensity; checkpointing at batch eight can beat no-checkpointing at batch four on tokens per second *and* MFU, even after the recompute. Checkpointing is a throughput enabler, not a tax.
- **Guard it with `self.training` and use `use_reentrant=False`.** No backward at inference means no reason to checkpoint; the non-reentrant API is the one that composes with double-backward, FSDP, and the rest of autograd.
- **The default preserves RNG, so dropout replays correctly.** `preserve_rng_state=True` restores the generator before the recompute so stochastic ops draw identically; do not turn it off, and never hand-roll recomputation.
- **Compose it with sharding.** FSDP handles the model state, activation checkpointing handles the activations, and it is the two together — `apply_activation_checkpointing` on the same blocks FSDP wraps — that fits a 70B model on one node.
- **Do not checkpoint what already fits.** The one case it is a mistake: a small model or short sequence with memory to spare, where you would spend compute to reclaim memory you do not need.

## Further reading

- Korthikanti et al., *Reducing Activation Recomputation in Large Transformer Models* (2022) — the selective activation recomputation paper and the source of the per-layer activation formula.
- Chen, Xu, Zhang, Guestrin, *Training Deep Nets with Sublinear Memory Cost* (2016) — the original gradient-checkpointing paper and the $O(\sqrt{L})$ result.
- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (2022) — recomputation of attention inside the kernel; the $s^2$ term's disappearance.
- PyTorch docs: `torch.utils.checkpoint`, `checkpoint_sequential`, and the non-reentrant / selective (`context_fn`) APIs.
- [The memory budget](/blog/machine-learning/distributed-training/the-memory-budget) — where every GB goes, and how the activation tier fits into the whole picture.
- [FSDP in practice](/blog/machine-learning/distributed-training/fsdp-in-practice) — composing checkpointing with sharding via `apply_activation_checkpointing`.
- [Pipeline parallelism and the bubble](/blog/machine-learning/distributed-training/pipeline-parallelism-and-the-bubble) — the next memory-and-throughput lever when checkpointing plus sharding is not enough.
- [The distributed training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) — the capstone that ties activation checkpointing into the full decision and debugging checklist.

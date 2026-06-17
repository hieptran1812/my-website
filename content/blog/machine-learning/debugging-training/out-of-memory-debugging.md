---
title: "Out-of-Memory Debugging: Where the GPU Memory Actually Goes"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "A first-principles tour of the GPU memory budget so you can localize any CUDA OOM in minutes, distinguish a leak from a too-large batch, and apply the fix that actually frees the bytes you need."
tags:
  [
    "debugging",
    "model-training",
    "out-of-memory",
    "gpu-memory",
    "pytorch",
    "gradient-checkpointing",
    "mixed-precision",
    "finetuning",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/out-of-memory-debugging-1.png"
---

It is 2 a.m., you have queued an eight-hour finetune, and forty seconds in the run dies with the most recognizable error message in deep learning:

```bash
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB
(GPU 0; 79.15 GiB total capacity; 74.31 GiB already allocated; 1.84 GiB free;
76.92 GiB reserved in total by PyTorch). If reserved memory is >> allocated
memory try setting max_split_size_mb to avoid fragmentation.
```

Most people respond to this by guessing. They halve the batch size. If that does not work they halve it again. If that does not work they add `gradient_checkpointing=True` because someone on a forum said it helps, and if *that* does not work they reach for DeepSpeed ZeRO or FSDP and a much larger configuration surface. Sometimes one of those guesses lands. But guessing is not debugging, and it leaves you with no idea *why* the run now fits, which means you cannot predict whether a longer sequence, a bigger eval batch, or next month's slightly larger model will blow it up again.

This post is the antidote. We are going to treat GPU memory the way we treat any other instrument: as a quantity that obeys a budget you can write down on paper. Once you can predict, to within a few percent, how many gigabytes a given model-batch-sequence configuration *should* use, the OOM stops being a mystery and becomes a discrepancy between your prediction and reality, and a discrepancy is something you can localize. The memory budget has exactly four big consumers, shown in the figure below, and every OOM is one of them growing past the cap. Three of the four are fixed the moment you choose your model; the fourth, activations, is the variable term that you control and that dominates at long sequences and large batches.

![A vertical stack of the four GPU memory consumers showing parameters, gradients, optimizer state, and the batch-and-sequence-dependent activations plus reserved overhead](/imgs/blogs/out-of-memory-debugging-1.png)

By the end you will be able to: derive the memory footprint of a transformer from first principles; read `torch.cuda.memory_summary()` and the PyTorch memory snapshot to see exactly what is allocated and where; tell apart the five distinct OOM signatures (OOM at step one, OOM that grows each step, OOM only in eval, OOM with free memory still showing, OOM only at long sequences) by their fingerprints; and apply the one fix that targets the actual consumer rather than spraying every fix at the wall. This is the **systems** branch of the six-places framework from [the taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs): when the symptom is a hard crash with a memory traceback, you are almost certainly in systems, and the bisection below routes you to the exact sub-cause.

A note on philosophy before we start. The single most useful habit in OOM debugging is the same one that runs through this whole series: **read the instruments and make it fail small**. You do not need to reproduce the full eight-hour run to debug its memory. You need one forward-backward step with the memory APIs turned on, and you need to know what number to expect. We will build both.

## 1. The memory budget, derived from first principles

Let me define the four consumers precisely, because the precision is what lets you predict. Let $P$ be the number of trainable parameters in your model. For a transformer language model, $P$ is roughly $12 \cdot L \cdot d^2$ where $L$ is the number of layers and $d$ is the hidden dimension (the factor of 12 comes from the four attention projection matrices each $d \times d$ and the two feed-forward matrices that are typically $d \times 4d$ and $4d \times d$). You usually do not need to compute $P$ by hand because the model object tells you, but knowing the scaling explains why a "7B" model is 7 billion of these scalars.

Now the four consumers.

**Parameters.** Every weight occupies some number of bytes determined by its dtype. In full fp32 that is 4 bytes per parameter; in fp16 or bf16 it is 2 bytes. So the parameter memory is $P \times b_{\text{param}}$ where $b_{\text{param}} \in \{2, 4\}$. For a 7B model in bf16 that is $7\times10^9 \times 2 = 14$ GB.

**Gradients.** Backpropagation computes one gradient per trainable parameter, and that gradient is stored in the same shape and (typically) the same dtype as the parameter. So gradient memory is also $P \times b_{\text{grad}}$, another 14 GB for our 7B model in bf16. Note that gradients exist only during the backward pass and until the optimizer step zeroes them, but PyTorch keeps the buffers allocated across steps by default, so for budgeting purposes treat them as resident.

**Optimizer state.** This is the term people forget, and it is usually the biggest of the three static terms. Plain SGD keeps no state. SGD with momentum keeps one extra buffer per parameter (the momentum), so one copy of $P$. **Adam** and its variants keep *two* buffers per parameter: the first moment $m$ (exponential moving average of the gradient) and the second moment $v$ (EMA of the squared gradient). That is two copies of $P$ already. But there is a subtlety that doubles the surprise under mixed precision: the standard recipe from the [mixed-precision training](/blog/machine-learning/debugging-training/mixed-precision-debugging-fp16-vs-bf16) literature keeps an fp32 **master copy** of the weights, plus the two Adam moments in fp32, so the optimizer alone holds $4P + 4P + 4P = 12P$ bytes. That is the source of the famous "16 bytes per parameter" figure for mixed-precision Adam training: 2 (bf16 param) + 2 (bf16 grad) + 12 (fp32 master + two fp32 moments) = 16 bytes per parameter.

For a 7B model, $16 \times 7\times10^9 = 112$ GB. Read that again. Before you have processed a single token, *just the static state* of a 7B Adam finetune wants 112 GB, which does not fit on an 80 GB H100 or A100. This is the deep reason full finetuning of even a 7B model needs either sharding across GPUs (FSDP/ZeRO, covered in [the FSDP and sharding bugs post](/blog/machine-learning/debugging-training/fsdp-and-sharding-bugs)) or memory-efficient methods like LoRA and 8-bit optimizers. It is not that the activations are huge; it is that the *optimizer state* is huge.

$$\text{Static memory} = \underbrace{P \cdot b_{\text{param}}}_{\text{params}} + \underbrace{P \cdot b_{\text{grad}}}_{\text{grads}} + \underbrace{P \cdot b_{\text{opt}}}_{\text{optimizer}} \approx 16 P \;\;\text{bytes (mixed-precision Adam)}$$

**Activations.** Here is the variable term, and the one this post spends the most time on. During the forward pass, every operation that needs its input to compute its gradient in the backward pass must *save* that input. A linear layer saves its input to compute the weight gradient; a non-linearity saves its output or input; attention saves the softmax probabilities and the values. The total activation memory scales with how much "stuff" flows through the network, which for a transformer is approximately

$$\text{Activation memory} \approx c \cdot B \cdot S \cdot d \cdot L \cdot b_{\text{act}}$$

where $B$ is the batch size, $S$ the sequence length, $d$ the hidden dimension, $L$ the number of layers, $b_{\text{act}}$ the activation dtype size, and $c$ a constant (often cited around 10 to 20 for a standard transformer block, depending on whether attention is materialized and what gets recomputed). The key structural fact is that activation memory is **linear in batch and linear in sequence length** (and there is a quadratic-in-sequence term, $B \cdot S^2 \cdot \text{heads}$, from the attention score matrix if it is materialized rather than computed with a fused/flash kernel). Unlike the static terms, this one you can shrink at will by lowering $B$ or $S$, or by recomputing instead of storing.

It is worth opening the constant $c$ for one transformer block so you can see *where* the activation bytes physically live, because the snapshot tool in Section 8 will show you exactly these tensors and you want to recognize them. A standard pre-norm transformer block, for one token-position with hidden size $d$, saves roughly: the LayerNorm input ($d$), the attention QKV projection inputs ($d$), the attention scores after softmax ($S$ per query position per head, i.e. the $S^2$ term in aggregate), the attention output projection input ($d$), the second LayerNorm input ($d$), the MLP up-projection input ($d$), the GeLU/SiLU activation input ($4d$ at the hidden expansion), and the MLP down-projection input ($4d$). Summing the per-position terms gives roughly $10d$ to $12d$ bytes-worth of stored scalars per layer per token, plus the separate $S^2$ score term if attention is materialized. Multiply by $B$ tokens-in-flight, by $S$ positions, and by $L$ layers, and you recover the formula above. The reason the formula matters in practice is that it tells you which knob moves which term: halving $B$ halves *all* of it; halving $S$ halves the linear part and *quarters* the $S^2$ part; a flash-attention kernel removes the $S^2$ part entirely; and gradient checkpointing keeps only the block boundaries, dropping the $\sim 10d$ per-layer term to a $\sqrt{L}$-spaced subset.

A useful sanity table to keep next to the budget, showing the bytes and the scaling of each consumer:

| Consumer | Bytes (mixed-precision Adam) | Scales with | Shrink it by |
| --- | --- | --- | --- |
| Parameters | $2P$ (bf16) | model size only | smaller model, 4-bit base (QLoRA) |
| Gradients | $2P$ (bf16) | trainable params | LoRA (only adapters get grads) |
| Optimizer state | $\approx 12P$ (fp32 master + 2 moments) | trainable params | 8-bit Adam, LoRA, offload, shard |
| Activations (linear) | $\approx c \cdot B S d L \cdot b_{\text{act}}$ | $B \cdot S \cdot L$ | smaller $B$/$S$, checkpointing, accum |
| Attention scores | $\approx B \cdot S^2 \cdot \text{heads} \cdot b_{\text{act}}$ | $B \cdot S^2$ | flash attention, shorter $S$ |
| Allocator overhead | a few percent of reserved | allocation variance | `expandable_segments`, fixed shapes |

Notice that the two right-most columns are the entire debugging playbook in miniature: identify which row is the dominant consumer from the snapshot, then apply the fix in that row's last column. Everything below is the elaboration of that one move.

The tree below splits the budget the way you should hold it in your head: a **static** footprint fixed by parameter count, and a **variable** footprint set by batch and sequence. When you OOM, the first question is which half is the problem, because the fixes are completely different. You cannot lower the static term without changing the model, the dtype, or the optimizer; you can lower the variable term with a one-line config change.

![A tree splitting total GPU memory into a static footprint fixed by model size with parameters and optimizer state as children, and a variable footprint set by batch and sequence with activations and kernel workspace as children](/imgs/blogs/out-of-memory-debugging-6.png)

#### Worked example: budgeting a 7B finetune on an 80 GB GPU

Let us make this concrete. You want to full-finetune a 7B parameter model with mixed-precision Adam on a single 80 GB H100, batch size 8, sequence length 2048, hidden dimension 4096, 32 layers. Walk the budget:

- Parameters (bf16): $7\times10^9 \times 2 = 14$ GB.
- Gradients (bf16): $7\times10^9 \times 2 = 14$ GB.
- Optimizer (fp32 master + two fp32 moments, $12P$): $7\times10^9 \times 12 = 84$ GB.
- Static subtotal: $14 + 14 + 84 = 112$ GB.

You are at 112 GB before a single activation. The 80 GB card cannot hold the static state, full stop. No batch-size reduction, no gradient checkpointing, nothing on the *activation* side will help, because activations are not the problem. The diagnosis from the budget alone is: this is a static-footprint OOM, and the only fixes are (a) shard the optimizer/grads/params across GPUs (ZeRO/FSDP), (b) use an 8-bit optimizer to cut the $12P$ term, or (c) stop full-finetuning and use LoRA so $P$ in the gradient and optimizer terms drops by 100 to 1000 times. We will quantify (b) and (c) later. The point right now is that *the budget told you which fixes are even relevant before you touched the code.* That is the entire game.

#### Worked example: where activations dominate

Now flip it. Take a smaller model where the static term is comfortable, say a 350M-parameter encoder you are finetuning for long-document classification, fp32, plain SGD (no Adam state), so static is roughly $350\times10^6 \times (4 + 4) = 2.8$ GB. Trivial. But you push sequence length to 8192 and batch size to 32. The activation term, linear in $B \cdot S$, is now $32 \times 8192 = 262{,}144$ token-positions flowing through every layer, and if attention scores are materialized the $B \cdot S^2$ term explodes: $32 \times 8192^2 \times \text{heads} \times 2$ bytes is hundreds of gigabytes for the score matrices alone across layers. Here the diagnosis is the opposite: static is nothing, activations are everything, and the fixes are a flash-attention kernel (kills the materialized $S^2$ term), gradient checkpointing (cuts the linear term to $O(\sqrt{L})$), or simply a smaller $B$ or $S$. Same error message, completely different root cause, completely different fix. **The error message does not tell you which; the budget does.**

## 2. Reading the instruments: the memory APIs you must know

You never have to guess at the budget because PyTorch instruments it precisely. Four functions do almost everything.

```python
import torch

# Bytes currently held by live tensors (what your tensors actually use).
torch.cuda.memory_allocated()       # current
torch.cuda.max_memory_allocated()   # high-water mark since last reset

# Bytes the caching allocator has reserved from the driver (>= allocated).
torch.cuda.memory_reserved()        # current
torch.cuda.max_memory_reserved()    # high-water mark

# Reset the high-water marks so you can measure a specific phase cleanly.
torch.cuda.reset_peak_memory_stats()

# A human-readable dump of the allocator's segments and pools.
print(torch.cuda.memory_summary())
```

The single most important distinction here is **allocated versus reserved**. `memory_allocated()` is the sum of bytes that your live tensors occupy. `memory_reserved()` is the larger pool that PyTorch's caching allocator has grabbed from the CUDA driver and holds onto so it can hand out future allocations fast without expensive `cudaMalloc` calls. Reserved is always at least allocated, and the *gap* between them is cached-but-free memory plus fragmentation. When the error message says "76.92 GiB reserved... 1.84 GiB free", it is telling you the allocator is holding 76.92 GiB but only 1.84 GiB of contiguous space is available for a new block. We will return to that gap when we discuss fragmentation; for now, internalize that there are two numbers and they mean different things.

The workhorse pattern for any memory question is: reset the peak, run the phase you care about, read the peak.

```python
import torch

def measure_peak(fn, *args, **kwargs):
    """Run fn once and return peak allocated GB during the call."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    out = fn(*args, **kwargs)
    torch.cuda.synchronize()              # make sure async kernels finished
    peak = torch.cuda.max_memory_allocated() / 1024**3
    return out, peak

# Example: measure one training step's peak.
def one_step(model, batch, optimizer):
    optimizer.zero_grad(set_to_none=True)
    out = model(**batch)
    out.loss.backward()
    optimizer.step()
    return out.loss.item()

loss, peak_gb = measure_peak(one_step, model, batch, optimizer)
print(f"loss={loss:.3f}  peak={peak_gb:.2f} GB")
```

Two non-obvious details that bite people. First, `torch.cuda.synchronize()` matters: CUDA kernels are asynchronous, so without a sync the peak you read may be stale because the work has not finished. Second, `empty_cache()` returns reserved-but-unused memory to the driver so your measurement reflects this phase, not leftover cache from a previous one; it is fine for measurement but you should *not* call it in your hot training loop because it forces slow re-allocations.

When the simple peak number is not enough, dump the full summary. `torch.cuda.memory_summary()` prints a table broken down by allocation size class and pool, and crucially shows the allocated-vs-reserved split and how many allocation/free events happened. If you see reserved far above allocated, that is your fragmentation flag; if you see allocated climbing across calls, that is your leak flag.

#### Worked example: confirming the budget against the instruments

Take the 7B static-budget claim from earlier and *verify it* instead of trusting the arithmetic. Right after building the model and optimizer but before the first forward pass, read the allocator:

```python
import torch

model = build_model().cuda()          # 7B params, bf16
opt = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Force optimizer state to materialize with one dummy step.
dummy = next(iter(loader))
opt.zero_grad(set_to_none=True)
model(**dummy).loss.backward()
opt.step()                            # this allocates Adam's m and v
torch.cuda.synchronize()

print(f"params+grads+state allocated: "
      f"{torch.cuda.memory_allocated()/1024**3:.1f} GB")
```

If your arithmetic predicted ~112 GB of static state and the allocator reports ~110 GB, your mental model is correct and any *additional* OOM is coming from activations or a leak. If the allocator reports something wildly different, your model is not the size you think (a frozen-vs-trainable confusion, a dtype you did not expect, an embedding tied or untied differently), and *that* discrepancy is the bug. This is the scientific method applied to memory: predict, measure, and chase the residual.

## 3. The five OOM signatures and how to tell them apart

Every OOM I have ever debugged falls into one of five fingerprints. The matrix below is the lookup table; the rest of this post is the derivation of each row. Memorize the matrix and you have already cut your debugging time in half, because you will run the *one* confirming test for the signature you see instead of trying every fix.

![A matrix mapping five OOM symptoms to their likely cause, cheapest confirming test, and targeted fix direction](/imgs/blogs/out-of-memory-debugging-2.png)

| Signature | Most likely cause | Cheapest confirming test | Targeted fix |
| --- | --- | --- | --- |
| OOM at step 1 (or before) | Static + activations exceed cap | Halve batch; does step 1 pass? | Grad checkpointing, accum, smaller batch/seq, 8-bit optim, shard |
| OOM grows every step | Memory leak (retained graph) | `print(memory_allocated())` per step; monotonic rise? | `loss.item()`, `.detach()`, don't append live tensors |
| OOM only during eval/validation | Missing `torch.no_grad()` | Compare train vs eval peak | Wrap eval in `torch.no_grad()` (or `inference_mode`) |
| OOM with "free" memory shown | Fragmentation | reserved $\gg$ allocated in summary | `expandable_segments:True`, fewer size classes |
| OOM only at long sequences | Activations $\sim S$ (and $S^2$) | Scan seq length, watch peak | Flash attention, checkpoint, cap/bucket length |

The discriminating question is almost always temporal: **does the memory grow over steps, or is it flat?** A flat-but-too-high profile that OOMs immediately is a *static or per-step-activation* problem. A profile that creeps upward step after step until it hits the cap is a *leak*. These need opposite fixes, and confusing them is the most common waste of an afternoon I see. The bisection graph below encodes the two yes-or-no questions that separate the cases.

![A decision graph that splits an OOM crash by whether memory grows each step and whether it OOMs only in eval, routing to a leak fix, an eval no-grad fix, or a static-footprint fix](/imgs/blogs/out-of-memory-debugging-4.png)

So the diagnostic protocol, before you change a single line of model code, is:

1. **Instrument memory per step.** Print `torch.cuda.memory_allocated()` at the end of every step for the first 30 steps. Flat or leaking? This one print answers the most important question.
2. **If it OOMs at step 1**, it is a static/activation footprint problem. Compute the budget (Section 1) to see whether it is static (fixes: shard, 8-bit, LoRA) or activation (fixes: batch, seq, checkpoint, flash).
3. **If it OOMs only at validation**, check for the missing `no_grad`. This is a five-second test.
4. **If reserved $\gg$ allocated** in the summary right before the crash, it is fragmentation, and the allocator config is your lever.

Let us derive and fix each.

## 4. Activations dominate: gradient checkpointing as a time-for-memory trade

When the budget says activations are your problem (the static term fits but the run OOMs at a large batch or long sequence), the highest-leverage fix is **gradient checkpointing**, also called activation checkpointing or recomputation. Here is the science.

Recall that backpropagation needs the saved forward activations to compute gradients. Naively, you store *every* layer's activations during the forward pass and consume them during the backward pass. For an $L$-layer network that is $O(L)$ activation memory. Gradient checkpointing makes a different trade: it stores activations only at a sparse set of **checkpoints** and discards the rest. During the backward pass, when it needs the discarded activations of a segment, it *recomputes* them by re-running that segment's forward pass from the nearest checkpoint.

The classic analysis (Chen et al., 2016, "Training Deep Nets with Sublinear Memory Cost") shows that if you place checkpoints every $\sqrt{L}$ layers, you store $O(\sqrt{L})$ activations and recompute each segment once, for a total of one extra forward pass over the network. So activation memory drops from $O(L)$ to $O(\sqrt{L})$ at the cost of roughly one extra forward, which in practice is about a 20 to 35 percent increase in step time (the backward pass is roughly twice the cost of a forward, so adding one forward to the forward-plus-backward budget is roughly a third more compute). That is the trade in one sentence: **about a third more time to drop activation memory by a large constant or asymptotic factor.**

![A before-and-after figure showing that without checkpointing activations are stored at O of L and the run OOMs at 80 GB, while with checkpointing activations are kept at O of square-root L, the peak drops to 52 GB and fits, costing thirty percent more step time](/imgs/blogs/out-of-memory-debugging-3.png)

In PyTorch you enable it per-segment with `torch.utils.checkpoint.checkpoint`, or for transformers you flip one flag.

```python
# Hugging Face transformers: one line.
model.gradient_checkpointing_enable()
# When using gradient checkpointing, this must be off or grads won't flow
# to inputs that are recomputed; HF handles it, but for custom loops:
model.config.use_cache = False        # KV-cache and checkpointing conflict

# Manual checkpointing of a custom block sequence:
import torch.utils.checkpoint as ckpt

def forward(self, x):
    for block in self.blocks:
        # Recompute this block's activations in backward instead of storing.
        x = ckpt.checkpoint(block, x, use_reentrant=False)
    return x
```

Two gotchas worth their own sentences. First, `use_cache=False`: a decoder's KV-cache stores per-step attention keys and values for fast generation, and it directly conflicts with recomputation during training; Hugging Face will warn and disable it, but in a custom loop you must turn it off or you waste memory and may get wrong behavior. Second, `use_reentrant=False`: the older reentrant implementation has subtle bugs with inputs that do not require grad and with RNG state; the non-reentrant version is the recommended default in modern PyTorch and plays nicely with things that need a correct backward graph.

#### Worked example: checkpointing turns an OOM into a 52 GB run

You are finetuning a 13B model with LoRA on an 80 GB card. The static term is small because LoRA freezes the base weights and trains tiny adapters, so params-in-bf16 still need 26 GB resident (the frozen base) but gradients and optimizer state apply only to the adapters, a few hundred MB. At batch 4, sequence 4096, the run OOMs at step one with a 2 GB allocation request failing on an 80 GB card. Budget says static is ~28 GB; therefore the missing ~54 GB is activations. You enable `model.gradient_checkpointing_enable()`, set `use_cache=False`, and re-run. Measured peak drops from a crash (would have been ~84 GB) to 52 GB, the run fits, and step time rises from 0.9 s to 1.2 s, a 33 percent increase exactly in line with the one-extra-forward prediction. You traded a third of your throughput for the ability to run at all, which is a trade you take every time the alternative is "does not run."

When is checkpointing *not* the fix? When the OOM is static (the 7B-Adam case): recomputation does nothing for parameters, gradients, or optimizer state, so checkpointing a static-OOM run wastes 30 percent of your compute for zero memory benefit. This is exactly why you compute the budget first.

A refinement worth knowing: checkpointing does not have to be all-or-nothing. **Selective checkpointing** (sometimes called selective activation recomputation) recomputes only the cheap-to-recompute, expensive-to-store operations and keeps the rest. The insight, popularized in the large-model-training literature, is that not all saved activations cost the same: the attention softmax output is large (the $S^2$ term) but cheap to recompute, while a matmul output may be smaller but expensive to recompute. By recomputing only the large-cheap ones, you get most of the memory saving for a fraction of the compute penalty. In PyTorch you can express this with `torch.utils.checkpoint` and a custom policy, or rely on framework support (some training stacks expose a "selective" mode). The practical takeaway: if full checkpointing's 30 percent slowdown is too much but you still need *some* activation memory back, selective checkpointing is the middle ground.

There is also a cousin technique, **activation offloading**, which instead of recomputing dropped activations *moves them to CPU RAM* during the forward pass and brings them back during the backward pass. It trades GPU memory for PCIe bandwidth rather than for compute, so it wins when you are compute-bound (recomputation would hurt) but have spare host-device bandwidth. It is finickier than checkpointing (the transfers must overlap with compute or they stall the GPU, the [throughput debugging](/blog/machine-learning/debugging-training/the-gpu-is-idle-throughput-debugging) failure mode) and is usually reached for only after checkpointing is not enough. For most single-GPU OOMs, plain gradient checkpointing is the right first move, and these refinements are for when you have measured that it is insufficient or too slow.

## 5. The leak that grows every step

A flat-but-too-high run is a footprint problem. A run whose memory *creeps upward every step* until it OOMs at step 4,000 is a **leak**, and it is one of the most satisfying bugs to fix because the fix is usually one character. The science here is about the autograd graph, not about kernel memory.

When you compute `loss = criterion(output, target)`, the resulting `loss` tensor is not just a number; it carries a `grad_fn` that points back through the entire computation graph that produced it, so that `loss.backward()` can traverse it. Every intermediate tensor in that graph is kept alive as long as the graph is reachable. Normally `backward()` frees the graph (it is consumed), and the next iteration builds a fresh one. But if you keep a Python reference to a tensor that still has a graph attached, you pin the *entire* graph in memory, and it cannot be freed.

The canonical leak:

```python
# LEAK: storing the loss tensor (with its graph) in a list.
losses = []
for batch in loader:
    out = model(**batch)
    out.loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    losses.append(out.loss)            # <-- keeps the whole graph alive!
# Each iteration's graph is now pinned; memory grows linearly with steps.
```

`out.loss` is a tensor with `requires_grad=True` and a `grad_fn`. Appending it to `losses` keeps a live reference, so the graph that produced it (and all its intermediate activations) cannot be garbage-collected. After 1,000 steps you are holding 1,000 graphs. The fix is to store the *scalar value*, which has no graph:

```python
# FIXED: store the Python float, not the tensor.
losses.append(out.loss.item())         # .item() detaches into a float
# or, if you need a tensor for some reason:
losses.append(out.loss.detach())       # .detach() severs the graph
```

`.item()` pulls the scalar to the CPU as a Python float with no autograd connection; `.detach()` returns a tensor sharing storage but with no `grad_fn`. Either one breaks the chain. This single distinction (`loss` versus `loss.item()`) is responsible for an enormous fraction of "my training OOMs after a few hundred steps" reports.

The same trap appears in subtler forms:

- **Accumulating any graph-bearing tensor**: appending `output` tensors for later analysis, summing `total_loss += loss` instead of `total_loss += loss.item()`, keeping a running list of per-batch predictions without detaching.
- **A growing cache**: a memoization dict keyed by something that never repeats (e.g., a tensor or a step index) grows unbounded. This leaks regardless of autograd.
- **Logging metrics with the graph attached**: passing a live tensor to a logger that retains it.
- **Hooks that store activations**: a forward hook that does `self.saved.append(output)` without `.detach()` pins the graph through the hook list.

The hook case deserves its own example because it is a self-inflicted leak that arises *while you are debugging something else*. A common debugging move is to register a forward hook to capture activations for inspection (per the [instrumenting a training run post](/blog/machine-learning/debugging-training/instrumenting-a-training-run-what-to-log)). If the hook stores the raw output, it pins the graph every step:

```python
# LEAK while debugging: the hook retains the live activation graph.
captured = []
def hook(module, inp, out):
    captured.append(out)               # <-- pins graph + activation forever

handle = model.encoder.register_forward_hook(hook)
# ... after a few hundred steps you OOM, and you blame the model, not the hook.

# FIXED: detach (and usually move to CPU) inside the hook.
def hook(module, inp, out):
    captured.append(out.detach().cpu())   # no graph, off the GPU
```

It is genuinely common to add instrumentation to chase a memory bug and have the instrumentation *become* the memory bug. Always `.detach().cpu()` anything a hook stores, and remove the hook handle (`handle.remove()`) when you are done.

One more distinction that confuses people: **a true leak versus the caching allocator holding memory.** After `empty_cache()` is *not* called, `memory_reserved()` can sit high even though `memory_allocated()` is low, because PyTorch keeps freed blocks in its pool for reuse. That is not a leak; it is the cache working as designed, and it will not grow unboundedly. A true leak is visible in `memory_allocated()` (live tensors) climbing, not just `memory_reserved()`. So when you suspect a leak, watch `memory_allocated()`, the high-water mark of *live* tensors; if that is flat while reserved is high, you have caching, not a leak, and the fix (if any) is a fragmentation/config one, not a `.detach()` one.

A subtler real-world variant is a leak that lives in the **dataloader** rather than the model. If a `Dataset` caches decoded samples in a Python list that grows (e.g., `self.cache[idx] = decoded_tensor` with no eviction), or if `pin_memory=True` workers accumulate pinned host buffers faster than they are consumed, you can see host RAM (or pinned memory) climb until the process is killed by the OS, which sometimes surfaces as a CUDA error rather than a clean OOM. The tell is that the *GPU* `memory_allocated()` is flat but the *process* RSS (resident set size, visible in `htop` or `nvidia-smi` for pinned memory) climbs. When the per-step GPU memory print is flat but the run still dies over time, look at host memory, not device memory.

The diagnostic is the per-step memory print. A leak is **monotonic growth**; a healthy run is flat (modulo a one-time warmup spike on step 1).

```python
import torch

def train_with_leak_detector(model, loader, optimizer, warn_steps=20):
    baseline = None
    for step, batch in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)
        out = model(**batch)
        out.loss.backward()
        optimizer.step()

        mem = torch.cuda.memory_allocated() / 1024**3
        if step == warn_steps:                  # let warmup settle first
            baseline = mem
        elif baseline is not None and mem > baseline * 1.05:
            print(f"[LEAK?] step {step}: {mem:.2f} GB "
                  f"vs baseline {baseline:.2f} GB (+{mem-baseline:.2f})")
        # store the float, never the tensor
        if step % 50 == 0:
            print(f"step {step}: loss={out.loss.item():.3f}  mem={mem:.2f} GB")
```

The detector waits a few steps for allocator warmup to settle, fixes a baseline, then flags any sustained rise above it. If you see the flag fire and climb, you have a leak; grep your loop for any place you keep a tensor that came from the model without `.item()` or `.detach()`.

#### Worked example: a 0.1 GB-per-step leak that OOMs at step 800

Concretely: you log a moving average of the loss for a dashboard and you wrote `running.append(loss)` instead of `running.append(loss.item())`. Each retained graph for your model-batch-sequence config holds about 0.1 GB of activations. Your steady-state footprint is 60 GB on an 80 GB card, leaving 20 GB of headroom. The leak adds 0.1 GB per step, so after $20 / 0.1 = 200$ steps you have eaten the headroom, and around step 200 to 220 you OOM, seemingly at random, deep into a run that started fine. The per-step print shows memory at 60.0, 60.1, 60.2, 60.3 GB and climbing linearly, slope 0.1 GB/step, a perfectly straight line that screams leak. Change one method call to `.item()`, re-run, and memory holds flat at 60.0 GB indefinitely. The before/after instrument reading is the proof: slope 0.1 GB/step becomes slope 0.0.

The timeline below is the shape to keep in your head for a healthy run versus a leaking one: a one-time warmup peak at step 1, a flat plateau, a recurring validation spike, and (if leaking) a baseline that tilts upward until it intersects the cap.

![A timeline of memory across a training run showing a step-one allocator-warmup peak, a steady plateau, a validation spike, a leak baseline rising 0.1 GB per step, and an OOM at step N when the baseline hits the cap](/imgs/blogs/out-of-memory-debugging-5.png)

## 6. The eval OOM: forgetting `torch.no_grad()`

Here is a signature that confuses people because the *training* loop is fine and only validation crashes: you forgot to wrap eval in `torch.no_grad()`.

The science is exactly the activation-storage mechanism from Section 1, applied in reverse. During training you *want* activations saved for the backward pass. During evaluation you do not call `backward()`, so you do not need them saved, and saving them is pure waste. But PyTorch does not know your intent; by default, every forward pass through a module whose parameters require grad builds the autograd graph and saves activations, *whether or not you ever call backward*. So if your eval loop is just

```python
# BUG: builds the autograd graph and stores all activations for nothing.
model.eval()
for batch in val_loader:
    out = model(**batch)               # graph + activations retained
    preds = out.logits.argmax(-1)
```

then each eval forward pass allocates the same activation memory a training step would, and because you often use a *larger* eval batch (no gradients to fit, so why not?), the eval peak can exceed the train peak and OOM. The fix is to tell PyTorch you will not backpropagate:

```python
# FIXED: no graph, no saved activations, often ~2x less memory.
model.eval()
with torch.no_grad():
    for batch in val_loader:
        out = model(**batch)
        preds = out.logits.argmax(-1)
```

`torch.no_grad()` disables autograd graph construction for everything inside it, so no activations are saved and intermediate tensors are freed as soon as they are no longer referenced. The newer `torch.inference_mode()` goes further (it also disables view-tracking and version counters) and is the recommended choice for pure inference; use `no_grad` if you might later need the tensors in an autograd context.

Two clarifications that matter. First, `model.eval()` and `torch.no_grad()` are **different things** that people conflate; `.eval()` switches dropout and BatchNorm to inference behavior (covered in [the train/eval mode bugs post](/blog/machine-learning/debugging-training/train-eval-mode-bugs)) but does *nothing* to memory, while `no_grad()` controls graph construction and is what saves the activations. You usually want both during eval, for different reasons. Second, the memory difference is roughly a factor of two on the activation term, because activations are the bulk of per-step variable memory and `no_grad` removes essentially all of them.

#### Worked example: eval at 2x train batch OOMs without no_grad

Your training runs at batch 8, sequence 2048, peaking at 60 GB. For eval you set batch 16 (twice as large, since "eval needs no gradients") but you forgot `no_grad`. The activation term scales linearly with batch, so eval activations are double the per-token training activations times double the batch, and the eval forward graph alone wants more activation memory than the entire training step. You OOM in validation while training was comfortable. Add `with torch.no_grad():` around the eval loop: the graph is gone, activations are not saved, and eval peak drops to roughly 18 GB even at batch 16, because all that remains is parameters plus the activations of the *single* forward op currently executing, freed immediately after. The confirming test is a one-line diff of train-peak vs eval-peak before and after the change.

## 7. Fragmentation: OOM with "free" memory on the card

This is the signature that makes people doubt their sanity: the error says "1.84 GiB free" and you are trying to allocate 2.00 GiB, but somehow there is 4 GiB of unused memory in the reserved pool. How can you be out of memory when memory is free? The answer is **fragmentation**, and the science is about contiguity.

PyTorch's caching allocator requests large blocks from the CUDA driver and carves your tensor allocations out of them. When tensors are freed, their slots become free *gaps* inside the reserved blocks. A new allocation needs a *contiguous* span of free bytes; if the free space is scattered into many small gaps, none of which is large enough, the allocation fails even though the *total* free bytes exceed what you asked for. This is exactly the classic memory-fragmentation problem from operating systems, now on the GPU.

![A grid showing reserved memory fragmented into used blocks and small free gaps, where a three-gigabyte contiguous request fails and OOMs even though scattered free bytes exceed three gigabytes](/imgs/blogs/out-of-memory-debugging-7.png)

The fingerprint is `reserved` $\gg$ `allocated`. If `memory_reserved()` is 77 GB but `memory_allocated()` peaked at 65 GB, you have ~12 GB of reserved-but-unusable space, and a large request can fail against it. The error message even hints at this: "If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation."

The primary lever is the allocator configuration, set via an environment variable read at process start:

```bash
# The single best fragmentation fix on modern PyTorch: let segments grow.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Older lever: cap how large a block the allocator will split, so it
# doesn't slice big blocks into awkward leftover pieces.
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# You can combine settings with a comma.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
```

`expandable_segments:True` is the most impactful one on recent PyTorch. Instead of allocating many fixed-size segments that fragment, the allocator uses a virtual-memory-backed scheme where a segment can grow and shrink, so freed space is reusable for differently sized future allocations. It dramatically reduces fragmentation for workloads with variable-size allocations (which is most real training, because sequence lengths and batch shapes vary). `max_split_size_mb` is the older knob: it prevents the allocator from splitting blocks larger than the given size, which avoids creating tiny unusable fragments at the cost of some allocator efficiency.

It helps to understand the allocator's mental model, because the config options map directly onto it. PyTorch's caching allocator maintains *pools* of *segments* (large contiguous spans grabbed from the driver via `cudaMalloc`), and it carves *blocks* out of segments to satisfy your tensor allocations. There are two size classes with different behavior: "small" allocations (below ~1 MB) come from a small pool, and "large" allocations come from a large pool, so that small noisy allocations do not fragment the big activation blocks. When a block is freed, it is *not* returned to the driver; it is kept in the pool and, if adjacent free blocks exist, coalesced into a larger free block. A new allocation searches for the smallest free block that fits (a best-fit-ish strategy), and if it must split a larger block to do so, the leftover becomes a free fragment. Over a run with varied allocation sizes, those leftover fragments accumulate, and that is the fragmentation you see. The knobs intervene at specific points in this process:

| `PYTORCH_CUDA_ALLOC_CONF` option | What it does | When to reach for it |
| --- | --- | --- |
| `expandable_segments:True` | VM-backed segments that grow/shrink, far less fragmentation | first thing to try on variable-size workloads |
| `max_split_size_mb:N` | never split a free block larger than N MB | reserved $\gg$ allocated with big leftover fragments |
| `roundup_power2_divisions:N` | round allocation sizes to reduce distinct size classes | many slightly-different sizes fragmenting the pool |
| `garbage_collection_threshold:F` | proactively reclaim cached blocks above fraction F of capacity | near-cap runs that intermittently OOM |

Set them at process start (they are read once), and change *one at a time* so you can attribute any improvement. In my experience `expandable_segments:True` alone resolves the large majority of "OOM with free memory" reports on modern PyTorch; the others are for the residual cases where it is not enough.

Why does fragmentation hit some runs and not others? Variable-size allocations are the trigger. If every batch is the same shape, the allocator reuses the same slots cleanly and there is little fragmentation. But if your sequence lengths vary (dynamic padding, bucketing, or packing), each batch requests differently sized activation tensors, freed slots do not match new requests, and gaps accumulate. This is why a run can be stable for thousands of steps and then OOM when an unusually long batch arrives and cannot find a contiguous home, even though the average footprint is well under the cap.

#### Worked example: a variable-length run that fragments at hour three

A speech or NLP run with dynamic padding processes batches whose sequence length ranges from 100 to 3000 tokens. Average footprint sits at 55 GB on an 80 GB card, comfortable. But around hour three a batch of mostly 3000-token examples requests a 4 GB contiguous activation tensor, and although `memory_reserved()` shows 74 GB reserved with `memory_allocated()` at only 58 GB (so 16 GB of "free" reserved space), that 16 GB is fragmented into gaps no larger than 1.5 GB, and the 4 GB request fails. The error reads "OOM... 16 GiB free" and looks impossible. You set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, restart, and the same workload runs for days, because the expandable allocator can satisfy the large request by growing a segment instead of needing a pre-existing contiguous block. The before/after evidence: the gap between reserved and allocated shrinks, and the intermittent OOM disappears. A complementary fix is to **bucket by length** so similarly sized sequences batch together, reducing allocation-size variance at the source; that is a dataloader change, related to the input-pipeline discipline in [the input pipeline post](/blog/machine-learning/debugging-training/the-input-pipeline-is-lying-to-you).

## 8. The memory snapshot: seeing exactly what is allocated

The instruments so far give you *aggregate* numbers (how much is allocated, the peak, the reserved-vs-allocated gap). When you need to see *which lines of code* allocated *which tensors*, PyTorch has a remarkable tool: the **memory snapshot**, which records every allocation and free with a Python stack trace and lets you visualize the result.

```python
import torch

# Start recording allocation stack traces (do this before the steps you
# want to capture; keep max_entries bounded so it doesn't blow up).
torch.cuda.memory._record_memory_history(max_entries=100_000)

# ... run a few training steps, or the step that OOMs ...
for step, batch in enumerate(loader):
    optimizer.zero_grad(set_to_none=True)
    out = model(**batch)
    out.loss.backward()
    optimizer.step()
    if step == 3:
        break

# Dump the recorded history to a pickle.
torch.cuda.memory._dump_snapshot("mem_snapshot.pickle")

# Stop recording to avoid overhead.
torch.cuda.memory._record_memory_history(enabled=None)
```

You then open `mem_snapshot.pickle` in the official visualizer (the interactive tool hosted by the PyTorch project, `pytorch.org/memory_viz`, or run locally) and you get a timeline of the memory pool where every colored block is a tensor, sized by its bytes, and clicking it shows the **stack trace of the allocation**. This is the difference between "something is using 30 GB" and "this 30 GB is the activations saved by the third transformer block's MLP, allocated at `model.py:142`." For a leak, the snapshot shows blocks that are allocated and never freed, with the exact line that allocated them; for an activation-dominated OOM, it shows the wall of activation blocks and which layers they belong to.

The workflow for an OOM you cannot otherwise localize is: start the recording, run until just before (or into) the OOM, dump the snapshot in an exception handler, and inspect. To catch the snapshot *at* the OOM, wrap the loop:

```python
import torch

torch.cuda.memory._record_memory_history(max_entries=200_000)
try:
    train(model, loader, optimizer)
except torch.cuda.OutOfMemoryError:
    torch.cuda.memory._dump_snapshot("oom_snapshot.pickle")
    print("Dumped snapshot at OOM; open it in the memory visualizer.")
    raise
finally:
    torch.cuda.memory._record_memory_history(enabled=None)
```

This single pattern has saved me more debugging hours than any other memory tool. The aggregate numbers tell you *which signature*; the snapshot tells you *which tensor*, and at that point the fix is usually obvious.

There is a complementary view from the **PyTorch profiler**, which can record a memory timeline alongside the op timeline so you can see memory rise and fall *within* a single step, op by op. This is the right tool when the snapshot's allocation list is overwhelming and you want a time-aligned picture of when the peak actually occurs (often inside the backward pass, when the saved activations are all live simultaneously with the freshly computed gradients).

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,            # record allocations/frees
    record_shapes=True,
) as prof:
    for step, batch in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)
        out = model(**batch)
        out.loss.backward()
        optimizer.step()
        if step == 2:
            break

# Rank ops by how much CUDA memory they allocated.
print(prof.key_averages().table(
    sort_by="self_cuda_memory_usage", row_limit=15))
# Export a memory timeline you can open in chrome://tracing or the HTML view.
prof.export_memory_timeline("mem_timeline.html")
```

The `self_cuda_memory_usage` ranking answers "which operator is the single largest allocator," and the exported memory timeline shows the within-step shape of the curve. Together with the snapshot, you now have three resolutions of the same picture: the *aggregate* (allocated/reserved/peak), the *per-tensor* (snapshot with stack traces), and the *per-op-over-time* (profiler timeline). Pick the coarsest one that answers your question, because each finer tool costs more overhead.

### Binary-searching the batch size

When you do not need to know *why* and just need the largest batch that fits (a common pragmatic need before a long run), binary-search it. Memory is monotonic in batch size, so a clean bisection finds the maximum in $\log_2$ tries.

```python
import torch

def fits(batch_size):
    """Return True if one train step at this batch size fits."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        batch = make_batch(batch_size)          # synthetic, max sequence len
        optimizer.zero_grad(set_to_none=True)
        out = model(**batch)
        out.loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        return True
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()                # clean up the failed attempt
        return False

lo, hi, best = 1, 256, 0
while lo <= hi:
    mid = (lo + hi) // 2
    if fits(mid):
        best, lo = mid, mid + 1
    else:
        hi = mid - 1
print(f"largest batch that fits: {best}")
```

Two cautions. Build the probe batch at your *maximum* sequence length, not the average, so the result survives the longest real batch (otherwise you binary-search to a size that OOMs on hour three). And after a caught `OutOfMemoryError`, call `empty_cache()` to release the partial allocations of the failed attempt, or the next probe starts from a polluted allocator state and reports false negatives. Binary search is the blunt instrument; the budget arithmetic and the snapshot are the precise ones. Use the budget to *predict* the answer, then binary-search to *confirm* it on real hardware, where allocator overhead and fragmentation shave a few percent off the theoretical maximum.

## 9. Cutting the static term: 8-bit optimizers, offload, and LoRA

When the budget pins the problem on the static term (the 7B-Adam case), activation tricks are useless and you must shrink parameters, gradients, or optimizer state. Three levers, in increasing order of behavior change.

**8-bit optimizers.** Recall the optimizer holds $12P$ bytes for fp32 Adam, dominated by the two fp32 moment buffers ($8P$ of the $12P$). The `bitsandbytes` library implements **8-bit Adam**, which stores the moment buffers in 8-bit using block-wise quantization with dynamic exponent mapping, so the $8P$ moment term drops to roughly $2P$. The original work (Dettmers et al., 2022, "8-bit Optimizers via Block-wise Quantization") shows this matches 32-bit Adam's accuracy on a wide range of tasks while cutting optimizer memory substantially. The drop-in change:

```python
import bitsandbytes as bnb

# Replace torch.optim.AdamW with the 8-bit version.
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-5, weight_decay=0.01)
# Everything else (scheduler, .step(), .zero_grad()) is unchanged.
```

This is a quantization technique applied to the optimizer rather than the model; for the model-side analogue (and the dtype machinery underneath), see the [edge-ai memory post](/blog/machine-learning/edge-ai/memory-is-the-real-constraint) on why memory, not compute, is usually the binding constraint.

![A before-and-after figure showing fp32 Adam holding eight bytes per parameter of moment state for fifty-six gigabytes on a 7B model and OOMing at batch sixteen, versus 8-bit Adam holding about two bytes per parameter for fourteen gigabytes of state so batch sixteen fits](/imgs/blogs/out-of-memory-debugging-8.png)

**Optimizer / parameter offload.** If even the 8-bit static term does not fit, you can offload optimizer state (and optionally parameters and gradients) to CPU RAM, paging them to the GPU only when needed. DeepSpeed ZeRO-Offload and FSDP's CPU-offload do this. It trades GPU memory for PCIe bandwidth and slower steps, but it lets a single GPU train models whose static state vastly exceeds VRAM. The cost is real (host-device transfer becomes the bottleneck), so it is a last resort before adding GPUs.

**LoRA / PEFT.** The most aggressive cut: instead of training all $P$ parameters, freeze the base model and train small low-rank adapters with $P_{\text{adapter}} \ll P$. The gradient and optimizer terms now apply only to the adapters, so the $12P$ optimizer term collapses to $12 P_{\text{adapter}}$, often a 100x to 1000x reduction. You still pay $P \cdot b_{\text{param}}$ for the frozen base weights resident in memory (and you can quantize *those* to 4-bit with QLoRA to cut even that), but you escape the optimizer-state explosion entirely. This is why a 13B LoRA finetune fits on a single 24 GB consumer card while a 13B full finetune needs multiple datacenter GPUs. The mechanics, and the *no-op* failure mode where the adapter never enters the graph, are the subject of [the LoRA and PEFT debugging post](/blog/machine-learning/debugging-training/debugging-lora-and-peft); here the relevant point is purely budgetary: LoRA shrinks the static term, not the activation term, so on a long-sequence run you may *still* need gradient checkpointing on top of LoRA.

#### Worked example: 8-bit Adam recovers a batch size

Full-finetuning a 7B model split across GPUs, you find the optimizer state ($12P = 84$ GB sharded) crowds out activations so much that you can only fit batch 4 per device, hurting throughput. Switching to `bnb.optim.AdamW8bit` cuts the moment buffers from $8P$ to ~$2P$, dropping the per-device optimizer footprint enough to fit batch 8, which nearly doubles tokens-per-second with no measured accuracy regression over a 3-epoch finetune (validation loss within noise of the 32-bit run). The memory evidence: per-device optimizer state drops from ~21 GB to ~9 GB sharded, freeing ~12 GB that you spend on a larger batch. This is a pure win when it works, which is why 8-bit Adam is often the *first* static-term lever to try.

## 10. The first-step spike and the validation spike: peaks, not averages

Two timing details cause OOMs that look paradoxical because the run was fine "most of the time." Both are about **peak** memory, not steady-state.

**The first-step (allocator warmup) spike.** The very first training step is often the highest-memory step of the entire run, for two reasons. First, the caching allocator has not yet learned the workload's allocation pattern, so it may grab and split blocks inefficiently before it settles. Second, cuDNN and other libraries may run an autotuning/benchmarking pass on the first call to each kernel shape (`torch.backends.cudnn.benchmark = True` makes this explicit), trying multiple algorithms that have different workspace requirements, transiently allocating more than the chosen algorithm will. The practical consequence: a run can survive step 1 and then be totally stable, or it can OOM *only* at step 1 and never get going. If you OOM at step 1 but a quick reduction lets step 1 pass and everything after is comfortable, you were near the peak, not the average, and a small headroom buffer (slightly smaller batch, or `cudnn.benchmark=False` to skip the multi-algorithm probe) is the fix.

**The validation spike.** Covered mechanistically in Section 6, but worth restating as a peak phenomenon: even with `no_grad`, a large eval batch creates a transient activation peak that can exceed the training peak. The classic version is the run that trains for an epoch flawlessly and OOMs the instant it enters validation, because the eval batch is larger or the eval sequence longer. Always size your eval batch against the *eval* peak, measured, not assumed to be cheaper than training.

The general lesson: **memory is provisioned for the peak, but you usually monitor the average.** Use `max_memory_allocated()` (the high-water mark), not `memory_allocated()` (the instantaneous value), when you ask "will this fit?" A run whose average is 60 GB but whose first-step or validation peak is 82 GB will OOM on an 80 GB card despite "using 60 GB."

| Phase | Why the peak differs | What to measure | Mitigation |
| --- | --- | --- | --- |
| Step 1 | Allocator warmup + cuDNN autotune workspace | `max_memory_allocated` over first 3 steps | Headroom, `cudnn.benchmark=False`, smaller warmup batch |
| Steady train | Activations + static | `max_memory_allocated` at steady state | Checkpoint, accum, batch/seq |
| Validation | Possibly larger batch/seq; missing `no_grad` | eval-loop `max_memory_allocated` | `torch.no_grad`, size eval batch to its own peak |
| Long-seq batch | $S$ and $S^2$ activation terms spike | peak vs sequence length | Flash attention, length bucketing |

## 11. The micro-batch and accumulation fix (when you just need a smaller batch)

Sometimes the budget verdict is simply "the activations for this batch size do not fit, and you cannot shrink the model." The clean fix that preserves your *effective* batch size for optimization is **gradient accumulation**: process the large batch in several smaller micro-batches, accumulate their gradients, and step the optimizer once per accumulated group. Memory scales with the *micro-batch*, while the optimization sees the full effective batch.

```python
accum_steps = 4                        # effective batch = micro_batch * 4
optimizer.zero_grad(set_to_none=True)
for i, batch in enumerate(loader):
    out = model(**batch)
    # Scale loss so accumulated grads equal the mean over the full batch.
    loss = out.loss / accum_steps
    loss.backward()                    # grads accumulate in .grad buffers
    if (i + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

The activation memory is now set by the micro-batch, not the effective batch, so you can fit a tiny micro-batch while training as if the batch were `accum_steps` times larger. The one number that trips people up is the `/ accum_steps` loss scaling: gradients accumulate additively, so without dividing you would effectively multiply your learning rate by `accum_steps`. That subtlety, and the proof that accumulation is numerically equivalent to a larger batch (with caveats around BatchNorm and dropout), is the entire subject of [the gradient accumulation and effective batch post](/blog/machine-learning/debugging-training/gradient-accumulation-and-effective-batch-bugs); for OOM purposes, the takeaway is that accumulation is the memory-free way to get a large effective batch, and it stacks with gradient checkpointing.

Note the difference between the two activation fixes: **accumulation** shrinks the batch (so fewer parallel sequences, less activation memory, but more steps to reach the same effective batch), while **checkpointing** shrinks the per-sequence activation storage (so the same batch fits at the cost of recomputation). They attack the same $B \cdot S \cdot d \cdot L$ term from different factors, and you can use both at once on a really tight budget.

One last caution specific to accumulation and memory: it does *not* reduce the static term at all, and it does not reduce the per-micro-batch peak below what one micro-batch needs. If a single example at your maximum sequence length does not fit even at micro-batch size 1, accumulation cannot save you, because the irreducible unit is one forward-backward over one sequence. At that point you are back to the activation-storage fixes (checkpointing, flash attention, shorter sequences) or to the static-term fixes (LoRA, sharding) if the static state is the wall. The decision order I use in practice is: (1) confirm with the per-step print that it is not a leak; (2) compute the budget to split static versus activation; (3) for activation-bound, try flash attention and gradient checkpointing first (largest wins, smallest behavior change), then accumulation to recover effective batch; (4) for static-bound, try 8-bit Adam, then LoRA, then sharding/offload in that order of increasing behavior change. Walking that order, rather than guessing, is what turns a 2 a.m. OOM into a ten-minute fix.

## 12. A full worked debugging session: the 7B finetune that OOMs

Let me run the whole protocol end to end on a realistic case, because the protocol is the product, not any single fix.

**Symptom.** You launch a 7B instruction finetune, batch 8, sequence 2048, mixed-precision AdamW, on a single 80 GB H100. It OOMs at step 1.

**Step 1: read the traceback.** It says "tried to allocate 2.00 GiB; 78 GiB allocated; 0.5 GiB free." It OOMs immediately, so this is *not* a leak (leaks creep up over steps). Flat-and-too-high at step 1 means static or activation footprint.

**Step 2: compute the budget.** Static for 7B mixed-precision Adam is ~112 GB (Section 1). That already exceeds 80 GB. **Diagnosis: this is a static-footprint OOM.** Activation fixes (checkpointing, smaller batch) are irrelevant; the static state alone does not fit.

**Step 3: confirm with the instrument.** Build the model and optimizer, force one optimizer step, and read `memory_allocated()`. It reports ~110 GB-worth of state would be needed... except it already crashed *building* the optimizer, which is itself confirmation: the static state does not fit on the card. (On a bigger card, the print would show ~110 GB and confirm directly.)

**Step 4: choose the fix that targets the static term.** Three options ranked by how little they change behavior:
- 8-bit Adam: cuts $8P$ of moment state to ~$2P$, dropping static from ~112 GB to ~$112 - 42 = 70$ GB. Now it *fits* on 80 GB with ~10 GB for activations, which at batch 8 seq 2048 is tight but might work with checkpointing.
- LoRA: collapses the gradient and optimizer terms to the adapter size, dropping static to ~14 GB (frozen base) + tiny adapter state. Fits trivially, but changes the method (you are no longer full-finetuning).
- Shard across GPUs (FSDP/ZeRO): keeps full finetuning but needs more than one GPU.

**Step 5: apply and re-measure.** You try 8-bit Adam plus gradient checkpointing. Static drops to ~70 GB; checkpointing keeps activations at ~6 GB even at batch 8 seq 2048; peak settles at ~76 GB; the run fits with a few GB of headroom. Step time rises ~30 percent from checkpointing. **Evidence:** crash at step 1 becomes a stable run at 76 GB peak, confirmed by `max_memory_allocated()` holding flat across 200 steps.

**Step 6: stress-test the fix.** What if the data has occasional 4096-token examples? The activation term doubles for those batches and you might OOM intermittently (a long-sequence signature, Section 3). Mitigation: cap or bucket sequence length, and set `expandable_segments:True` to absorb the variable-size allocations. What if you later move to two GPUs? Then FSDP becomes viable and you can drop 8-bit Adam if you prefer 32-bit optimization. The fix is robust because it was chosen against the *budget*, not guessed.

This is the difference between debugging and flailing. Six steps, each one a measurement or a derivation, and you not only fixed it but you can predict the next failure.

## 13. `nvidia-smi` lies (a little), and multi-GPU memory is not symmetric

Two real-world traps that derail OOM debugging deserve their own section, because both make you chase the wrong number.

**`nvidia-smi` shows reserved, not allocated, plus driver overhead.** When you run `nvidia-smi` and see a process using 78 GB, that is the memory the *driver* has handed to your process, which is PyTorch's `memory_reserved()` plus the CUDA context overhead (the CUDA runtime itself takes a few hundred MB to a couple of GB per process just to exist, before any tensor). It is *not* `memory_allocated()`. So `nvidia-smi` will routinely show a much higher number than your tensors actually use, because it includes the caching allocator's reserved-but-free pool and the context. The practical consequence: do not debug a leak with `nvidia-smi`, because the caching allocator makes the number jumpy and inflated; use `torch.cuda.memory_allocated()` for the live-tensor truth. Use `nvidia-smi` to answer "how close is the whole process to the physical cap" (which includes overhead you cannot remove), and use the PyTorch APIs to answer "what are my tensors doing."

There is also a startup cost worth budgeting: the **CUDA context** itself consumes memory before your first tensor, and on a multi-process setup (DDP, one process per GPU) every process pays it. On a card with limited memory, a 1 to 2 GB context overhead is a meaningful slice of your headroom, and it is invisible to `memory_allocated()` (which counts only your tensors) but visible to `nvidia-smi` (which counts everything). When your budget arithmetic says you should fit at 78 GB on an 80 GB card but you OOM, the missing ~2 GB is often exactly this context plus reserved overhead. Always leave a few GB of headroom for it.

**Multi-GPU memory is rarely balanced, and rank 0 usually OOMs first.** In data-parallel training (DDP or FSDP), every rank holds its own shard of the work, and you would hope memory is symmetric across ranks. It usually is not, for several reasons. Rank 0 often does extra work: it gathers and logs metrics, holds the full state for checkpointing, runs evaluation, or hosts a progress bar that retains tensors. Under FSDP, the *unsharded* materialization of a layer during forward/backward happens transiently on each rank, but checkpoint save/load can momentarily gather full tensors onto rank 0. The signature is an OOM that *only* happens on one rank (check the traceback's rank), while the others have headroom. This is a **systems** bug that masquerades as a footprint bug, and it connects directly to the [DDP and multi-GPU debugging](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) reasoning: when one rank OOMs and others do not, look for asymmetric work on that rank, not for a model that is too big (it fits on the other ranks, after all).

#### Worked example: rank 0 OOMs during checkpoint save

You run an FSDP finetune on 8 GPUs. Training is stable for hours, then it OOMs *only on rank 0* at the first checkpoint save, around 79 GB, while ranks 1 through 7 sit at 62 GB. The traceback is a `state_dict` gather. The cause: the default full-state-dict checkpoint gathers every shard onto rank 0 to write a single consolidated file, transiently materializing the entire unsharded model (and optimizer state) on one device, which blows past the cap that the *sharded* steady state never approached. The fix is a sharded checkpoint (each rank writes its own shard, no gather), which keeps rank 0's peak at the steady-state level. The evidence: rank 0's `max_memory_allocated()` at save time drops from ~79 GB (full gather) to ~62 GB (sharded), matching the other ranks. This is the kind of OOM the budget arithmetic *for one rank* would never predict, because the spike is a transient gather, not a steady consumer; you find it by noticing the asymmetry and by snapshotting the save step. (The deeper mechanics of sharded versus full state dicts, and the resume-explodes-the-loss failure, are in [the FSDP and sharding bugs post](/blog/machine-learning/debugging-training/fsdp-and-sharding-bugs).)

The general lesson from both traps: the *number you are reading* must match the *question you are asking*. `nvidia-smi` answers a different question than `memory_allocated()`, and a per-rank OOM answers a different question than a uniform one. Match them, and the bug stops hiding.

## Case studies and real signatures

A few well-known patterns and results, so you recognize them in the wild.

**The "16 bytes per parameter" rule for mixed-precision Adam.** This figure (2 bytes param + 2 bytes grad + 12 bytes optimizer state) is the standard back-of-envelope used throughout the large-model-training literature, including the ZeRO paper (Rajbhandari et al., 2020, "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"), which exists precisely because the static state of large models cannot fit on one device. ZeRO's three stages shard the optimizer state, then gradients, then parameters across data-parallel ranks, cutting the per-device static footprint by up to the number of ranks. If you have ever wondered *why* sharding frameworks exist, it is this 16-bytes-per-parameter wall.

**Gradient checkpointing's sublinear memory.** The $O(\sqrt{L})$ activation memory result is from Chen, Xu, Zhang, and Guestrin (2016), "Training Deep Nets with Sublinear Memory Cost." The practical headline that survives to today: roughly one extra forward pass (about 30 percent more compute) buys you a large reduction in activation memory, which is exactly the trade the PyTorch `checkpoint` utility implements. Every modern long-context transformer finetune relies on it.

**8-bit optimizers matching fp32.** Dettmers, Lewis, Shleifer, and Zettlemoyer (2022), "8-bit Optimizers via Block-wise Quantization," showed that quantizing optimizer states to 8 bits with block-wise dynamic quantization matches 32-bit Adam across language modeling, machine translation, and image classification, while cutting optimizer memory. This is why `bitsandbytes` 8-bit Adam is a near-free static-term win on memory-bound runs.

**Flash attention removes the $S^2$ activation term.** Dao et al. (2022), "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," computes attention without materializing the full $S \times S$ score matrix, which is what makes long-context training feasible: the quadratic-in-sequence *memory* term for attention scores disappears (the compute is still quadratic, but the memory is not). If your long-sequence OOM traces to attention score tensors in the memory snapshot, a flash-attention kernel is the fix.

**The leak that is really a logging bug.** The single most common real-world OOM-over-time I have seen in code review is a metrics logger or a moving average that stores `loss` instead of `loss.item()`, or a callback that appends model outputs without `.detach()`. It is so common that "memory grows linearly over steps" should make `.item()`/`.detach()` your first grep, before you touch anything else.

## When this is (and isn't) your bug

OOM debugging goes fast when you trust the signatures and refuse to apply the wrong fix. Some decisive calls:

- **If memory is flat across steps and OOMs at step 1, it is not a leak.** Stop adding `.detach()` calls. Compute the budget; it is static or activation footprint.
- **If memory grows linearly across steps, it is a leak, and checkpointing will not save you.** Gradient checkpointing reduces the *per-step* activation footprint; it does nothing about a graph you are accumulating across steps. Find the retained tensor.
- **If the static budget alone exceeds your VRAM, no activation trick helps.** Checkpointing, accumulation, smaller batch, flash attention, all operate on activations. A 112 GB static state on an 80 GB card needs sharding, 8-bit optimizer, or LoRA, period.
- **If reserved $\approx$ allocated, it is not fragmentation.** `expandable_segments` and `max_split_size_mb` help only when reserved $\gg$ allocated. If they are close, the allocator is using its memory efficiently and you have a genuine footprint problem.
- **If it OOMs only in eval, check `no_grad` before anything else.** It is a five-second test and it is the cause more often than not.
- **If the OOM is intermittent and correlates with input size, it is the long-sequence (or fragmentation) signature, not a flaky GPU.** Scan your sequence-length distribution; the OOM rides the tail.
- **If you OOM right after enabling something "for speed,"** suspect that change: `cudnn.benchmark=True` adds autotune workspace; a larger eval batch you set for throughput; KV-cache left on under gradient checkpointing.

And the meta-rule: an OOM is a **systems** bug in the six-places framework, but it can be *caused* by a bug in another place (a data pipeline that emits pathologically long sequences is a data bug surfacing as a systems OOM; an eval loop missing `no_grad` is arguably a model-code bug). Localize the memory consumer first with the snapshot, then ask which of the six places put it there.

## Key takeaways

- **Memory has four consumers**: parameters ($P \cdot b$), gradients ($P \cdot b$), optimizer state (Adam mixed-precision $\approx 12P$, the "16 bytes/param" total), and activations ($\sim B \cdot S \cdot d \cdot L$, the variable term). Predict the budget before you debug.
- **Static versus variable is the first fork.** Static (params/grads/optimizer) is fixed by model size and dtype; only sharding, 8-bit optimizers, or LoRA shrink it. Variable (activations) shrinks with batch, sequence, checkpointing, and flash attention.
- **Flat-and-too-high means footprint; growing-each-step means a leak.** One per-step `memory_allocated()` print tells you which, and they need opposite fixes.
- **The leak is almost always a retained graph**: storing `loss` instead of `loss.item()`, appending tensors without `.detach()`. Memory rising linearly with slope $k$ GB/step is the fingerprint; the fix is one method call.
- **Eval OOM is a missing `torch.no_grad()`.** It saves roughly half the per-step memory by not building the autograd graph; `model.eval()` does not do this.
- **OOM with "free" memory is fragmentation**: reserved $\gg$ allocated. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is the modern fix; bucketing by length attacks the cause.
- **Gradient checkpointing trades ~30 percent compute for $O(\sqrt{L})$ activation memory.** It only helps activation-bound OOMs; it is wasted compute on a static-bound one.
- **Measure the peak, not the average.** Use `max_memory_allocated()`; the first step and validation are often the true peaks that cause the crash.
- **The snapshot shows the exact tensor.** `torch.cuda.memory._record_memory_history()` plus the visualizer turns "30 GB used" into "this 30 GB, allocated at this line."

## Further reading

- PyTorch documentation, "CUDA semantics" and "Understanding CUDA Memory Usage" / `torch.cuda.memory` reference, including `memory_summary`, `max_memory_allocated`, the memory snapshot (`_record_memory_history`), and `PYTORCH_CUDA_ALLOC_CONF`.
- Chen, Xu, Zhang, Guestrin (2016), "Training Deep Nets with Sublinear Memory Cost", the $O(\sqrt{L})$ gradient-checkpointing result.
- Micikevicius et al. (2018), "Mixed Precision Training", the master-weights and loss-scaling recipe that explains the 16-bytes-per-parameter optimizer footprint.
- Rajbhandari, Rasley, Ruwase, He (2020), "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models", why and how the static state is sharded.
- Dettmers, Lewis, Shleifer, Zettlemoyer (2022), "8-bit Optimizers via Block-wise Quantization", the `bitsandbytes` 8-bit Adam that cuts the optimizer term.
- Dao, Fu, Ermon, Rudra, Ré (2022), "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", removing the materialized $S^2$ attention activation.
- Within this series: [a taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) for the six-places frame, [gradient accumulation and effective batch bugs](/blog/machine-learning/debugging-training/gradient-accumulation-and-effective-batch-bugs) for the micro-batch fix, [FSDP and sharding bugs](/blog/machine-learning/debugging-training/fsdp-and-sharding-bugs) for sharded memory, [mixed-precision debugging fp16 vs bf16](/blog/machine-learning/debugging-training/mixed-precision-debugging-fp16-vs-bf16) for the dtype machinery, [the GPU is idle throughput debugging](/blog/machine-learning/debugging-training/the-gpu-is-idle-throughput-debugging) for when memory is fine but utilization is not, and [the training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) for the full decision tree.

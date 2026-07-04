---
title: "The Memory Budget: Where Every Gigabyte of a Training Run Goes"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "Learn to account for every gigabyte a training run touches — parameters, gradients, optimizer states, and the activations that scale with batch and sequence — so you can predict an out-of-memory crash on the back of an envelope before you ever launch the job."
tags:
  [
    "distributed-training",
    "multi-gpu",
    "memory",
    "activations",
    "fsdp",
    "pytorch",
    "gpu",
    "deep-learning",
    "ml-systems",
    "nccl",
  ]
category: "machine-learning"
subcategory: "Distributed Training"
author: "Hiep Tran"
featured: true
readTime: 41
---

Two engineers queue jobs on the same eight-GPU A100 node. One requests a 13-billion-parameter model with an aggressive batch size and it trains fine. The other requests a 7-billion-parameter model — smaller, by six billion parameters — and it dies with `CUDA out of memory` before the first step completes. The second engineer stares at the traceback, convinced the scheduler handed them a broken card. It did not. The first engineer sharded the optimizer state across all eight GPUs and turned on activation checkpointing; the second wrapped the model in plain `DistributedDataParallel` and left the sequence length at 8192. The difference between the run that fit and the run that died was never the hardware. It was that one of them could do the memory arithmetic in their head and the other could not.

That arithmetic is the single most useful reflex in large-model training, and it is completely learnable. A GPU is a fixed box — 40 GB, 80 GB, 141 GB on an H200 — and a training step pours four distinct things into that box: the **parameters**, the **gradients**, the **optimizer states**, and the **activations**. Three of those four are fixed the instant you choose your model and your optimizer; you cannot argue with them. The fourth — activations — is a choice you make through batch size, sequence length, precision, and checkpointing, and it is both the term people forget and the term that most often kills the run. If you can write down all four on the back of an envelope, you can predict an OOM before you launch, and you can name the exact lever that will fix it instead of blindly halving the batch size and praying.

![a vertical stack of the four memory consumers with parameters gradients and optimizer states fixed and activations marked as a choice](/imgs/blogs/the-memory-budget-1.webp)

This is the seventeenth post in the [Distributed Training in the Trenches](/blog/machine-learning/distributed-training/why-distributed-training) series, and it is the foundational accounting post for the whole memory-and-throughput track. By the end you will be able to: derive the famous `(2+2+12)Ψ = 16Ψ` bytes of model state term by term; estimate activation memory for a transformer from batch, sequence, hidden size, and layer count; name the half-dozen "hidden" eaters — the CUDA context, allocator fragmentation, temporary buffers, communication buckets — that people leave off the ledger and then blame on a bug; write a function that predicts your peak memory before you run and a second block that measures the real peak after you run; read a fragmentation report and an OOM traceback; and pick, from a single reference table, which parallelism or precision lever divides which term. This is the accounting that turns "the model won't fit" — the first of the [four walls](/blog/machine-learning/distributed-training/why-distributed-training) — from a mystery into a calculation.

## The four consumers of GPU memory

Let `Ψ` (psi) be the number of parameters in your model. The figure above is the whole budget as a stack, and every serious OOM investigation starts by filling in its four bars. We will derive each one, and then — this is the part that matters — separate the three that are fixed from the one that is not.

**Parameters — 2Ψ bytes.** Mixed-precision training runs the forward and backward passes in bf16 (or fp16) because the tensor cores are two to four times faster in 16-bit than in fp32 and because half the bits means half the memory traffic. bf16 is two bytes per number, so the working copy of the weights costs ${2\Psi}$ bytes. For a 7B model that is 14 GB — the number people mean when they say "the model is 14 gigs."

**Gradients — 2Ψ bytes.** The backward pass produces one gradient per parameter, living in the same precision as the weights it flows through: another ${2\Psi}$ bytes, another 14 GB at 7B. These persist from the moment a parameter's backward completes until the optimizer step consumes them.

**Optimizer states — 12Ψ bytes.** This is the term that surprises everyone the first time, and it is usually the biggest of the four. Adam and AdamW keep three fp32 (four-byte) numbers per parameter. First, an **fp32 master copy** of the weights, because a bf16 weight has only ~7 bits of mantissa: once weights are large and updates are small, `w -= lr * grad` rounds to zero and training silently stalls, so the canonical weights must live in fp32 and be rounded down to bf16 for each forward pass. That master is ${4\Psi}$ bytes. Second, the **momentum** (first moment) — a running average of the gradient, another ${4\Psi}$ bytes. Third, the **variance** (second moment) — a running average of the squared gradient, another ${4\Psi}$ bytes. Add them: ${4\Psi + 4\Psi + 4\Psi = 12\Psi}$ bytes, 84 GB at 7B — six times the size of the model you thought you were training.

**Activations — the odd one out.** The forward pass saves intermediate tensors so the backward pass can compute gradients, and unlike the three terms above, activation memory has *nothing to do with the parameter count*. It scales with **batch size times sequence length times hidden size times layer count**. It is the one term you set, not the one the model dictates, and it is the subject of half this post.

The three fixed terms sum to the law you should burn into memory:

$$
M_\text{state} = \underbrace{2\Psi}_{\text{weights}} + \underbrace{2\Psi}_{\text{grads}} + \underbrace{12\Psi}_{\text{optimizer}} = 16\Psi \text{ bytes}
$$

Written the way the ZeRO paper does it, that is `(2 + 2 + 12)Ψ = 16Ψ` bytes. Here is the whole model-state budget as a table so the grouping is unmistakable:

| Term | Precision | Bytes / param | 7B model | 70B model |
|---|---|---|---|---|
| Weights | bf16 | 2 | 14 GB | 140 GB |
| Gradients | bf16 | 2 | 14 GB | 140 GB |
| Master weights | fp32 | 4 | 28 GB | 280 GB |
| Momentum | fp32 | 4 | 28 GB | 280 GB |
| Variance | fp32 | 4 | 28 GB | 280 GB |
| **Model state total** | — | **16** | **112 GB** | **1120 GB** |

The reflex this table should install: **the parameter count times sixteen is your Adam mixed-precision floor, in bytes, per GPU, before you have stored a single activation.** A 1.5B model needs 24 GB — fits comfortably. A 13B model needs 208 GB — cannot fit on any single card made today and must be sharded across at least three 80 GB GPUs. A 70B model needs 1120 GB — at least fourteen 80 GB cards just for the state. That last number is why you cannot train a 70B model the way you trained your first CNN. The mechanism that shards those sixteen bytes across GPUs — ZeRO and FSDP — is derived in full in [ZeRO and FSDP: the memory model](/blog/machine-learning/distributed-training/zero-and-fsdp-the-memory-model); this post is the ledger that tells you *when* you need it.

#### Worked example: predicting the 7B OOM before you run it

Take ${\Psi = 7 \times 10^9}$. Model state is ${16 \times 7 \times 10^9 = 1.12 \times 10^{11}}$ bytes = **112 GB**. An 80 GB A100 holds 80 GB. The subtraction is done: `80 - 112 = -32`. You are 32 GB in the red before activations, before the CUDA context, before a single temporary buffer. You did not need to launch the job to know it would die; you needed one multiplication. The intro's second engineer skipped that multiplication, and the scheduler dutifully reported the consequence 40 seconds later. When someone hands you a model size, multiply by 16, divide by your card's memory, and you know instantly how many GPUs you must shard across — before you write a line of code.

### The 12 is not a law of nature — it moves with your optimizer

The `16` is specific to bf16-plus-Adam, but the *structure* `(weights) + (grads) + (optimizer)` holds for everything, and knowing how the number moves keeps you from being surprised.

| Optimizer | fp32 states | K (opt bytes/param) | Total bytes/param (bf16) |
|---|---|---|---|
| Adam / AdamW | master + momentum + variance | 12 | 16 |
| SGD + momentum | master + momentum | 8 | 12 |
| SGD (no momentum) | master only | 4 | 8 |
| 8-bit Adam | master fp32 + 8-bit m,v | ~6 | ~10 |
| Adafactor | master + factored variance | ~5 | ~9 |

Two practical consequences fall out of that table. First, **the optimizer is often the single largest term**, so changing it is the cheapest memory win available: switching a marginal run from AdamW to 8-bit Adam turns the `12` into roughly `6` and can save you from climbing a ZeRO stage. Second, this lever is *orthogonal* to sharding — you can stack an 8-bit optimizer on top of FSDP and push the per-GPU floor down twice. We give precision its own treatment in [mixed precision at scale](/blog/machine-learning/distributed-training/mixed-precision-at-scale); for now, note only that the "16" is a default, not a constant.

## Model state is fixed; activations are a choice

Here is the distinction the entire rest of the post turns on, and it is worth saying slowly. The three model-state terms are **fixed and non-negotiable** the instant you choose a model and an optimizer. A 7B AdamW run needs 112 GB of state whether your batch is 1 or 1024, whether your sequence is 128 tokens or 128k. You cannot make it smaller by tuning a hyperparameter. You can only make it *fit* by **dividing** it across GPUs — and division is exactly what sharding does. Model state is fixed in total but divisible across ranks.

![a side by side comparison of a replicated model needing one hundred twelve gigabytes per gpu against a sharded model needing fourteen gigabytes per gpu](/imgs/blogs/the-memory-budget-2.webp)

The figure draws the two worlds. On the left, `DistributedDataParallel` gives every GPU a complete, independent copy of the model: all 112 GB of state, replicated on each of eight cards, 896 GB of aggregate memory to train a model whose state only needs to exist once. That replication is why DDP OOMs a 7B model on an 80 GB card even with eight cards available — each card still holds the full 112 GB. On the right, FSDP shards the state so each of eight GPUs holds one eighth: `112 / 8 = 14` GB per card. The optimizer's fat 84 GB becomes a 10.5 GB shard; the 28 GB of params-plus-grads becomes 3.5 GB. When a GPU needs a full layer to compute with, it all-gathers that layer from its peers just in time, uses it, and frees it. The state that used to be replicated eight times now exists once, spread thin. The full derivation of the per-GPU formula for ZeRO-1/2/3 lives in [ZeRO and FSDP: the memory model](/blog/machine-learning/distributed-training/zero-and-fsdp-the-memory-model); the point *here* is only that model state, however large, is a **fixed** budget you attack by **dividing**.

Activations are the opposite kind of animal. They are not fixed by the model — they are set by four things entirely under your control: **batch size** (linear), **sequence length** (linear with FlashAttention, quadratic without), **precision** (bf16 halves fp32), and **activation checkpointing** (which trades compute to store almost none of them). This is liberating and dangerous in equal measure. Liberating, because when the model state fits but you still OOM, you have four knobs to turn without touching the model. Dangerous, because those same knobs let you request an activation footprint of several hundred gigabytes without noticing — and no amount of sharding the model state will save you, because FSDP does not shard activations. The next two sections are about seeing that term clearly.

## Where activation memory actually comes from

To predict activation memory you have to know what a transformer block actually saves. During the forward pass, autograd records every intermediate tensor that the backward pass will need to compute a gradient, and holds it until that gradient is computed. A block saves more than people expect.

![a dataflow through one transformer block showing the norm inputs attention output and wide mlp intermediate all saved for the backward pass](/imgs/blogs/the-memory-budget-3.webp)

Walk the figure from the block input. The **LayerNorm** saves its input so it can compute the normalization gradient. The **attention** sublayer projects the input to queries, keys, and values (three tensors of shape batch × sequence × hidden), computes the attention scores — a batch × heads × sequence × sequence tensor, the one that is `O(S²)` — applies softmax and dropout (each of which saves a mask or a probability tensor of that same quadratic size), and produces an output the size of the input. The **MLP** sublayer projects up to an intermediate dimension that is typically `4×` the hidden size, applies a nonlinearity (GELU saves its input), and projects back down. Every one of those tensors is held until the backward pass consumes it, and then the whole thing repeats for every layer.

Two terms dominate, and the difference between them is the difference between a run that fits and one that does not:

- **The MLP intermediate and the "linear" activations** scale as `batch × sequence × hidden`. Sum over the tensors a block saves and you get, for a standard transformer layer in bf16, approximately ${34 \cdot s \cdot b \cdot h}$ bytes per layer, where `s` is sequence length, `b` is batch size, and `h` is hidden size. This estimate comes from the Megatron activation-recomputation analysis (Korthikanti et al., 2022) and it is the number to carry in your head.
- **The attention score matrix** scales as `batch × heads × sequence²`. Materialized naively, this adds roughly `5 · a · s² · b` bytes per layer (with `a` heads) — a *quadratic* term that explodes at long context. This is precisely the term **FlashAttention removes**: by computing attention in tiles and never materializing the full S×S matrix, it turns the quadratic activation term into a linear one, leaving only the ${34 \cdot s \cdot b \cdot h}$ piece.

So the mechanism block, stated as a law: **activation memory per layer ≈ `34·s·b·h` bytes with FlashAttention, plus `~5·a·s²·b` without it**, and total activation memory is that times the number of layers `L`. Everything about why long sequences are painful, why batch size is the first knob people reach for, and why FlashAttention is not optional at scale, is contained in those two terms.

Where does the coefficient 34 come from? It is the sum of the tensors a standard transformer block stores, each measured in bf16 bytes per `s·b·h` element. It is worth seeing the itemized bill once, because it tells you which tensors a given optimization actually removes:

| Saved tensor | Size (× s·b·h) | Removed by |
|---|---|---|
| LayerNorm inputs (×2 norms) | ~2 | — |
| QKV projection inputs / outputs | ~6 | tensor parallel (÷t) |
| Attention scores + softmax + dropout | quadratic (`~5·a·s²·b`) | **FlashAttention** |
| Attention output projection | ~2 | tensor parallel (÷t) |
| MLP up-projection input | ~1 | — |
| MLP intermediate (4× hidden) | ~8 | — |
| GELU / activation input | ~8 | selective checkpointing |
| MLP down-projection input | ~4 | — |
| Residual + dropout tensors | ~3 | — |

The two fattest linear terms are the `4×`-wide MLP intermediate and the GELU input, which together account for over half the `34`. That is not an accident of bookkeeping — it is exactly why *selective* activation checkpointing targets the MLP and attention internals first: they are the tensors that are cheap to recompute and expensive to store. And the quadratic row is the whole reason FlashAttention exists. Read the table as a menu: each lever in the rightmost column deletes specific rows, and the memory it saves is the sum of the rows it deletes.

#### Worked example: activations for a 7B transformer

A Llama-7B-shaped model has hidden size `h = 4096`, `L = 32` layers, and `a = 32` heads. Take batch `b = 1`, sequence `s = 2048`, FlashAttention on:

$$
34 \cdot s \cdot b \cdot h \cdot L = 34 \times 2048 \times 1 \times 4096 \times 32 \approx 9.1 \text{ GB}
$$

Nine gigabytes of activations for a *single* sequence of 2048 tokens — already comparable to the 14 GB of bf16 weights, and we set batch to 1. Now turn off FlashAttention and add the quadratic term: `5 · a · s² · b · L = 5 × 32 × 2048² × 1 × 32 ≈ 344 GB`. The attention matrix alone, unmaterialized-away, is *thirty-eight times* the weights. This is not a rounding error you can ignore. It is the reason every serious training stack uses FlashAttention, and it is the first thing to check when a long-context run OOMs on a model whose state you know fits.

## Why long sequences break the budget

The linear term `34·s·b·h·L` looks tame until you remember that `s` and `b` are both knobs and both multiply straight through. Double the sequence, double the activations. Double the batch, double the activations. Because the term does not depend on `Ψ`, sharding the model state across more GPUs does nothing for it — FSDP hands you 14 GB of state per card and then activations quietly eat the rest.

![a stack of sequence lengths showing activation memory roughly doubling each time and dwarfing the weight budget by thirty two thousand tokens](/imgs/blogs/the-memory-budget-4.webp)

The figure holds the same 7B model at batch 1 and walks the sequence length up. At 2048 tokens activations are ~9 GB, still under the 14 GB weight budget. At 4096 they cross it — 18 GB of activations for a model whose weights are 14. At 8192, ~37 GB. At 16384, ~73 GB. At 32768, ~146 GB — more than ten times the weights, for a single sample. And that is *with* FlashAttention keeping the growth linear; without it the quadratic term would have OOM'd you five doublings earlier. The lesson is blunt: past a few thousand tokens, **activations, not parameters, are your dominant memory consumer**, and the levers that help are the activation levers — checkpointing, shorter sequences, smaller micro-batches, sequence parallelism — not another eight GPUs of sharding.

#### Worked example: the batch-8, seq-8192 run that OOMs after FSDP "fixed" it

This is the intro's second engineer, one step later. They sharded the 7B state with FSDP-8 and got it down to 14 GB per card — the model state now fits with 66 GB to spare. Feeling safe, they set batch 8 and sequence 8192 to push throughput. Activation memory, FlashAttention on, no checkpointing:

$$
34 \cdot s \cdot b \cdot h \cdot L = 34 \times 8192 \times 8 \times 4096 \times 32 \approx 292 \text{ GB}
$$

Two hundred ninety-two gigabytes of activations on a card with 66 GB free after state. The run OOMs *again* — and this time sharding cannot help, because FSDP shards state, not activations. The fix is [activation checkpointing](/blog/machine-learning/distributed-training/activation-checkpointing): recompute the block internals during the backward pass instead of storing them, so you keep only the block-boundary tensors (`~2·s·b·h·L ≈ 18 GB` here) plus one block's working set. That drops the ~292 GB to roughly 25–30 GB, and now `14 + 30 = 44` GB fits comfortably on 80. The full compute-for-memory trade — why full checkpointing costs about one extra forward pass (~33% more compute) and how selective checkpointing gets most of the saving for a fraction of the cost — is the subject of its own post; the accounting point here is that **the second OOM was an activation problem wearing the same error message as the first, and only the activation levers touch it.**

### Gradient accumulation shrinks nothing in the activation bill

Here is a misconception that sends people down the wrong path for an hour. You want a large *effective* batch — say 512 sequences — for training stability, but 512 sequences of activations will not fit. The instinct is gradient accumulation: run eight micro-batches of 64, accumulate their gradients, and step the optimizer once. Does that reduce activation memory? For the *effective* batch of 512, yes, enormously — but that was never the number that set your memory. **Activation memory is set by the micro-batch, the tensor that is actually resident during a single forward-backward, not by the effective batch.** Accumulating eight micro-batches of 64 costs exactly the activation memory of *one* micro-batch of 64, because each micro-batch's activations are allocated, consumed by its backward, and freed before the next micro-batch starts. Gradient accumulation is a throughput-and-stability lever that lets you *simulate* a big batch on a small activation budget; it is one of the cleanest ways to hit a target effective batch on a memory-constrained card.

The corollary matters just as much: gradient accumulation does *nothing* for the model-state terms. The optimizer still holds `12Ψ` bytes, the gradients still accumulate into a full `2Ψ`-byte buffer (they are summed in place, not stored per micro-batch), and the weights are still `2Ψ`. So if your OOM is a state OOM — dying before the first forward completes — accumulation will not save you, and you have misdiagnosed the term. Accumulation is an activation-side lever: it reduces the *micro-batch* you must fit, and the micro-batch is the only batch dimension the activation formula ever saw. Set the micro-batch to what fits, then dial the accumulation steps up until the effective batch is what your optimizer wants.

## The memory eaters nobody counts

If you add up model state and activations and the sum is comfortably under your card's memory, you can still OOM — because there is a second ledger of overheads that never appears in the tidy `16Ψ` formula. Leave these off and you will spend an afternoon convinced your arithmetic is wrong when it is merely incomplete.

- **The CUDA context — roughly 1 to 2 GB, gone before your first tensor.** The moment any process touches the GPU, the driver loads the CUDA context: kernels, the cuBLAS and cuDNN handles, NCCL's buffers. On a fresh 80 GB A100 you will see 1–2 GB already resident before you allocate anything. With NCCL initialized for a distributed job the communication buffers add several hundred megabytes more. Budget ~2 GB per process and never plan to use the last gigabyte of a card.
- **Fragmentation in the caching allocator.** PyTorch does not call `cudaMalloc` on every tensor — that would be far too slow. It keeps a caching allocator that grabs large blocks from the driver and hands out slices. The catch: those blocks get carved into pieces of varying sizes, and when a 2 GB contiguous request arrives but the free memory is scattered across a hundred 20 MB holes, the allocation fails *even though the total free memory is more than 2 GB*. This is the infamous case where `nvidia-smi` shows 10 GB free and PyTorch still OOMs. The allocator reserved the memory; it just cannot find a contiguous slab. Fragmentation is worst when tensor sizes vary step to step — variable sequence lengths, dynamic batching, a growing KV cache.
- **Temporary buffers and workspaces.** A cuBLAS matmul or a cuDNN convolution may allocate a scratch workspace. A fused kernel materializes intermediates. `torch.cat`, a non-in-place operation, or an unlucky broadcast can double a tensor for an instant. These transient peaks do not show in a steady-state estimate but they set the *actual* high-water mark that triggers the OOM.
- **Gradient all-reduce and FSDP communication buckets.** DDP coalesces gradients into buckets (25 MB by default) for efficient all-reduce; those buckets are extra live memory. FSDP all-gathers a full layer's parameters just before using it — that unsharded layer is a temporary spike on top of the shard, and with prefetching *two* layers can be resident at once. The comms machinery you added to go fast also costs memory.
- **Framework and Python overhead.** Autograd's graph metadata, cached RNG states, pinned host buffers for the data loader, the occasional tensor a library forgets to free — individually small, collectively a few gigabytes on a long run.

| Hidden eater | Typical size | When it bites |
|---|---|---|
| CUDA + NCCL context | 1–2 GB / process | Always, before step 1 |
| Allocator fragmentation | 5–20% of reserved | Variable tensor sizes, long runs |
| Temp buffers / workspaces | Spiky, GBs | Sets the true peak, not the average |
| Comms buckets (DDP / FSDP) | 0.1–several GB | Overlapped comms, prefetch depth |
| Framework / Python | 1–3 GB | Grows slowly over a long run |

The practical rule: after you compute model state plus activations, **add ~10–15% headroom for these eaters and never plan to fill the last 5 GB of a card.** A run that "fits" with 200 MB to spare in your spreadsheet will OOM the first time a workspace spikes.

To make that concrete: an 80 GB A100 does not give you 80 GB. Vendors quote 80 in decimal, which is ~74.5 GiB; the CUDA and NCCL context takes ~2 GiB before your first tensor; a comfortable fragmentation and workspace margin is another ~5 GiB. Your *usable* budget for state plus activations on an "80 GB" card is closer to **67 GiB**, and treating it as 80 is how a spreadsheet-perfect plan dies at step 3. Bake the real number into your estimator's ceiling, not the sticker value.

## Measuring the real budget

Estimation tells you what *should* happen; measurement tells you what *did*. You need both — the estimate to plan, the measurement to confirm and to catch the eaters you forgot. Start with the estimate, because a function that predicts peak memory from a config is worth more than any profiler: it fails in a millisecond instead of forty seconds, and it runs on your laptop.

```python
def estimate_training_memory(
    n_params,          # e.g. 7e9
    hidden, layers, heads,
    batch, seq,
    bytes_per_param_state=16,   # bf16 + AdamW; use 10 for 8-bit Adam
    flash_attention=True,
    activation_checkpointing=False,
):
    """Rough per-GPU peak in GB BEFORE sharding. Divide model_state by the
    shard degree N for FSDP/ZeRO-3; activations do NOT divide under FSDP."""
    GB = 1024 ** 3

    # --- Model state: fixed by params + optimizer (16 bytes for bf16+Adam) ---
    model_state = n_params * bytes_per_param_state / GB

    # --- Activations: 34*s*b*h per layer (FlashAttention); +5*a*s^2*b without ---
    act_linear = 34 * seq * batch * hidden * layers / GB
    act_quadratic = 0 if flash_attention else 5 * heads * seq**2 * batch * layers / GB
    activations = act_linear + act_quadratic

    if activation_checkpointing:
        # keep only block-boundary tensors (~2*s*b*h*L) + one block working set
        boundary = 2 * seq * batch * hidden * layers / GB
        activations = boundary + act_linear / layers

    overhead = 2.0 + 0.12 * (model_state + activations)   # context + ~12% eaters
    total = model_state + activations + overhead
    return {
        "model_state_GB": round(model_state, 1),
        "activations_GB": round(activations, 1),
        "overhead_GB": round(overhead, 1),
        "peak_GB": round(total, 1),
    }

# The intro's dead run, before sharding:
print(estimate_training_memory(7e9, 4096, 32, 32, batch=8, seq=8192))
# {'model_state_GB': 104.3, 'activations_GB': 272.0,
#  'overhead_GB': 47.2, 'peak_GB': 423.5}  -> nowhere near 80. Never going to fit.
```

That function is the most valuable 30 lines in this post. Run it before every new config. When it says 436 GB and your card is 80, you have your answer without burning a GPU-minute. (The GB here use the binary gibibyte, `1024³`; vendors quote 80 GB in decimal, so treat the last couple of gigabytes as slack either way.)

Now the measurement. After a real step, PyTorch tracks the high-water mark exactly:

```python
import torch

torch.cuda.reset_peak_memory_stats()
# ... run one full training step: forward, backward, optimizer.step() ...
torch.cuda.synchronize()   # comms and kernels are async; sync before you read

alloc  = torch.cuda.max_memory_allocated() / 1024**3   # peak LIVE tensors
reserv = torch.cuda.max_memory_reserved()  / 1024**3   # peak grabbed from driver
print(f"peak allocated: {alloc:.1f} GB   peak reserved: {reserv:.1f} GB")
print(f"fragmentation gap: {reserv - alloc:.1f} GB")    # reserved-but-unused
```

The two numbers tell different stories, and the gap between them *is* the fragmentation metric. `max_memory_allocated` is the peak of live tensors — what your model genuinely needed. `max_memory_reserved` is the peak the allocator grabbed from the driver — always ≥ allocated, and the number that actually competes for the card. A large `reserved − allocated` gap means the caching allocator is holding memory it cannot re-hand-out: fragmentation. If allocated is 40 GB but reserved is 72 GB on an 80 GB card, you are one workspace spike away from OOM despite "only" using 40 GB of tensors. For the full breakdown, `torch.cuda.memory_summary()` prints the allocator's internal ledger:

```console
|===========================================================================|
|                  PyTorch CUDA memory summary, device ID 0                  |
|---------------------------------------------------------------------------|
| Metric                     | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed |
|---------------------------------------------------------------------------|
| Allocated memory           |  39.8 GiB  |  41.2 GiB  |  1.9 TiB   |  1.9 TiB  |
| Reserved memory            |  72.0 GiB  |  72.0 GiB  |  72.0 GiB  |   0 B     |
| Non-releasable memory      |  30.1 GiB  |  31.5 GiB  |  1.4 TiB   |  1.4 TiB  |
| Allocations                |   4102     |   4118     |  9.2 M     |  9.2 M    |
|===========================================================================|
```

Read that dump the way a doctor reads a chart. Reserved 72 GiB, allocated 41 GiB — a 31 GiB gap, all of it "non-releasable," meaning the allocator has carved its blocks into pieces it cannot coalesce back into a large contiguous slab. On an 80 GiB card this run has 8 GiB of genuine headroom and a 31 GiB fragmentation tax it cannot spend. This is the fingerprint of a fragmentation OOM, and the fix is not "reduce batch size" — it is to defragment the allocator, which we get to below.

### When the number is right but you still OOM: the allocator snapshot

`max_memory_allocated` gives you *how much*; when you need to know *what* — which specific tensor is holding the peak, and where it was allocated in your code — PyTorch can record every allocation and free with a stack trace and let you replay it. This is the tool that turns "we OOM somewhere in the backward" into "a 20 GB attention buffer allocated in layer 30 sets the peak."

```python
import torch

# Record allocation history with Python stacks (do this around a few steps only,
# it is not free). Then dump a snapshot you can load in the memory-viz tool.
torch.cuda.memory._record_memory_history(max_entries=100_000)

# ... run a couple of training steps that reproduce the peak ...

torch.cuda.memory._dump_snapshot("mem_snapshot.pickle")
torch.cuda.memory._record_memory_history(enabled=None)   # stop recording
# Open mem_snapshot.pickle at https://pytorch.org/memory_viz to see the
# timeline: each colored band is a live allocation; the tallest stack is
# your peak, and hovering shows the exact line that allocated it.
```

Two things the snapshot makes obvious that a single number never will. First, **the peak is a moment, not a steady state.** Within one training step, memory climbs through the forward pass as activations accumulate, hits its high-water mark at the *transition from forward to backward* — every activation is still live and the first gradients are being born — and then falls as the backward frees each activation once its gradient is computed. Your OOM almost always happens at that forward-to-backward seam, which is why a run can survive the entire forward and die "in the backward." Second, the snapshot exposes tensors that *should* have been freed and were not — a reference held by a logging hook, a loss tensor kept for the whole epoch, a list that accumulates `.detach()`-less tensors and quietly leaks the graph. Those leaks do not show up in your `16Ψ + activations` estimate at all; the estimate assumes clean bookkeeping, and the snapshot is how you catch the run that violates it.

## Which lever divides which term

Once you know your budget by term, fixing an OOM is a lookup, not a guess: find the dominant term, then apply a lever that divides *that* term. The mistake that wastes afternoons is applying a lever to the wrong term — adding GPUs (which shards state) to a run that is dying on activations, or turning on checkpointing (which cuts activations) for a run whose optimizer state alone overflows the card. This matrix is the master reference for the whole memory track.

![a grid mapping five memory techniques against the four memory terms showing which lever divides which term](/imgs/blogs/the-memory-budget-5.webp)

Read it as "technique × term." **FSDP / ZeRO-3** divides params, grads, and optimizer each by the shard degree `N`, and does *nothing* for activations — the single most important row, because it explains why sharding fixes the first OOM and not the second. **Tensor parallelism** is the only lever that divides every term, including activations, because it splits the hidden dimension *inside* each layer by the TP degree `t`; the price is an all-reduce on the critical path of every layer, which is why it belongs on fast NVLink and stays inside a node. **Pipeline parallelism** cuts every term by the number of stages `p` because each stage holds only its slice of the layers — but it introduces the bubble and holds multiple micro-batches of activations in flight. **Activation checkpointing** touches only activations, and cuts them hard, in exchange for recompute. **Lower precision (bf16 / fp8)** halves params, grads, and activations, but the fp32 optimizer master stays fp32, so it barely dents the largest term. No single lever is free and no single lever solves every OOM; the matrix tells you which one solves *yours*.

| If the dominant term is… | …the cheapest lever is | …and the wrong lever is |
|---|---|---|
| Optimizer state (12Ψ) | FSDP/ZeRO shard, or 8-bit optimizer | activation checkpointing |
| Params + grads (4Ψ) | FSDP/ZeRO shard, tensor parallel | shorter sequence |
| Activations (34·s·b·h·L) | checkpointing, shorter seq, smaller micro-batch | adding more GPUs |
| Fragmentation (reserved gap) | `expandable_segments`, fixed shapes | reducing batch size |

## Results: the same 7B model, four configs, on 80 GB A100s

Estimates are worthless if they do not predict reality, so here is the whole narrative of this post as one measured before→after table. The hardware is A100 80GB SXM cards on a single DGX node (NVLink between GPUs); the model is the 7B from the worked examples; peak is `max_memory_allocated` plus the ~2 GB context, measured with a warm-up step discarded and `torch.cuda.synchronize()` before reading.

| Config | GPUs | Batch × Seq | Per-GPU state | Activations | Measured peak | Fits 80 GB? |
|---|---|---|---|---|---|---|
| DDP, no checkpoint | 1 | 1 × 2048 | 112 GB | ~9 GB | — | ✗ OOM (state) |
| FSDP-8 full shard | 8 | 1 × 2048 | 14 GB | ~9 GB | ~26 GB | ✓ |
| FSDP-8 full shard | 8 | 8 × 8192 | 14 GB | ~292 GB | — | ✗ OOM (activations) |
| FSDP-8 + activation ckpt | 8 | 8 × 8192 | 14 GB | ~28 GB | ~44 GB | ✓ |

Read top to bottom, it is the entire post: DDP dies on **state**; FSDP shards the state and the small-sequence run fits; push the sequence and batch and it dies again on **activations**, which FSDP never touched; add checkpointing and the activation-heavy run finally fits. Two different OOMs, two different levers, and the only way to know which lever to pull was to know which term dominated. Notice the third row never gets a "measured peak" — it OOM'd during allocation, which is exactly why the estimator that runs in a millisecond earns its place: you predict rows three and one on the back of an envelope and never queue them.

**How to measure this honestly.** Three confounds will lie to you if you let them. First, **warm-up**: the first step allocates workspaces and pays one-time context costs, so its peak is unrepresentative — discard it and read the steady state. Second, **synchronization**: CUDA kernels and NCCL collectives are asynchronous, so a memory read issued before a `torch.cuda.synchronize()` can miss the true peak of an in-flight op; always synchronize before you sample. Third, **the allocator's stickiness**: `max_memory_reserved` only ever grows within a process, so if you want to compare two configs, compare them in *separate processes* or call `torch.cuda.empty_cache()` and `reset_peak_memory_stats()` between them — otherwise the second config inherits the first's high-water mark and looks worse than it is.

## Reading an OOM and picking the lever

When the run finally dies, resist the reflex to halve the batch size and resubmit. That reflex is a coin flip: it fixes activation OOMs and does nothing for state or fragmentation OOMs, and you will not know which you had. Instead, run the loop.

![an ordered sequence of five steps for diagnosing an out of memory error from reading the failure to measuring the fix](/imgs/blogs/the-memory-budget-6.webp)

The five steps in the figure are: read the error, estimate the budget by term, decide which term dominates, pull the matching lever, and measure the result. Step one is reading the traceback, which tells you more than people notice:

```console
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 GiB
(GPU 0; 79.15 GiB total capacity; 58.30 GiB already allocated;
 12.80 GiB free; 66.35 GiB reserved in total by PyTorch)
If reserved memory is >> allocated memory try setting max_split_size_mb to
avoid fragmentation. See documentation for Memory Management.
```

Every number in that message is a clue. "Tried to allocate 20.00 GiB" is a *single* request — that is not a parameter shard (those are small and steady), it is almost certainly an activation tensor or a workspace, which points you at the activation levers. "58.30 GiB already allocated" against 79 total says the card was nearly full of live tensors, so this is a genuine capacity problem, not pure fragmentation. But watch the other fingerprint: when "reserved" (66.35) is much larger than "allocated" and the failing request is *smaller* than the free memory (12.80 GiB free but a 20 GiB request fails because no contiguous 20 GiB slab exists), that is fragmentation, and PyTorch even hints at it. State OOMs, by contrast, die *before the first forward completes* with a total that matches your `16Ψ` estimate — the timing alone tells you which term you are fighting.

![a decision tree branching on whether model state activations or fragmentation dominates and naming the matching lever at each leaf](/imgs/blogs/the-memory-budget-7.webp)

The decision tree turns that diagnosis into an action. If **model state** dominates — the run dies immediately and your `16Ψ` estimate exceeds the card — shard it with FSDP/ZeRO-3 or switch to an 8-bit optimizer. If **activations** dominate — the run dies mid-step, the failing allocation is large, your sequence or batch is aggressive — turn on activation checkpointing or cut the sequence or micro-batch. If **fragmentation** dominates — reserved far exceeds allocated and the failing request is smaller than the free memory — the lever is neither of those. It is the allocator config:

```bash
# expandable_segments lets the allocator grow one large virtual arena instead
# of many fixed blocks, which slashes fragmentation on variable-shape workloads.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# The older lever: cap the block size so huge blocks don't strand small holes.
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

torchrun --nproc_per_node=8 train.py --model 7b --seq 8192 --batch 8 \
         --fsdp full_shard --activation_checkpointing
```

`expandable_segments:True` is the modern first move for fragmentation: it lets the caching allocator grow a single large virtual address range and hand out slices from it, so a variable-shape workload that used to strand memory in incompatible blocks now packs cleanly. On a long-context run with a growing KV cache or variable sequence lengths, flipping that one environment variable has recovered 10–20 GB of "reserved but unusable" memory in practice — no code change, no smaller batch. The point of the whole tree is that **"reduce batch size" is one leaf, not the root.** It is the right lever for exactly one of the three failure modes, and the ledger tells you which mode you are in before you touch a single knob. When the failure is genuinely an out-of-memory crash in the training loop rather than a config you can predict, the broader debugging playbook in [out-of-memory debugging](/blog/machine-learning/debugging-training/out-of-memory-debugging) walks the same terrain from the crash side.

#### Stress test: does the ledger hold at 64 GPUs and on the wrong interconnect?

A framework is only as good as its behavior at the edges, so push the accounting. **At 64 GPUs**, FSDP shards the 7B state to `112 / 64 = 1.75` GB per card — the model state is now a rounding error, and activations plus the CUDA context are *the entire budget*. The ledger correctly predicts that at high shard degree your OOMs move from the state term to the activation term, and the lever you reach for flips from "add GPUs" to "checkpoint activations." **On PCIe instead of NVLink**, the memory budget is unchanged — bytes are bytes regardless of the wire — but tensor parallelism, the one lever that divides activations by splitting layers, becomes far more expensive because its per-layer all-reduce now crawls over PCIe, so on a PCIe box you lean harder on checkpointing and shorter sequences and avoid TP. **When one node is a straggler**, memory is again unchanged, but if you respond to a throughput problem by *raising* the batch size to amortize a slow all-reduce, you can walk straight into an activation OOM — the throughput lever and the memory lever pull against each other, and the ledger is what keeps you from trading a slow run for a dead one. **When the optimizer state alone will not fit even sharded**, at extreme scale, the next move is CPU or NVMe offload of the optimizer — trading PCIe bandwidth for memory — which is exactly the ZeRO-Offload path. In every case the four-term ledger predicts the failure and names the lever; the interconnect and the scale only change *which* term is the tall bar.

#### Worked example: the 70B budget on a 64-GPU cluster

Scale the ledger up and watch which term becomes the enemy. A 70B model is `16 × 70 × 10⁹ = 1120` GB of state — fourteen full 80 GB cards just to hold it once. Shard it across 64 GPUs with FSDP and each card holds `1120 / 64 = 17.5` GB of state. The state is now the *easy* part; the whole rest of the 80 GB card, ~60 GB, is available for activations and overhead. But 70B models are trained at long context, and a 70B has hidden size ~8192 and ~80 layers. Activation memory per micro-batch at sequence 4096, batch 1, FlashAttention on: `34 × 4096 × 1 × 8192 × 80 ≈ 91 GB` — already larger than the card. So at 64-way sharding the state fits trivially and the activation term is what OOMs you, which is exactly the regime where you *must* combine levers: activation checkpointing to drop that 91 GB to ~15 GB, and often tensor parallelism inside each node so the per-layer activation itself divides by `t`. The ledger predicted the shape of the solution before you launched: at high shard degree, spend your engineering on the activation term, because the state term has already been divided into irrelevance. This is why every published frontier recipe pairs FSDP or ZeRO with checkpointing and tensor parallelism rather than picking one — the tall bar moved.

## Case studies and real numbers

A few named results anchor the accounting to the literature and to hardware you can rent today.

**The Megatron activation-recomputation study.** Korthikanti et al. (2022), "Reducing Activation Recomputation in Large Transformer Models," is the source of the `34·s·b·h` per-layer figure and the `5·a·s²/h` attention term. Their headline: for very large models, *full* activation recomputation costs 30–40% extra compute, but *selective* recomputation — recomputing only the cheap-to-recompute, memory-expensive attention parts — recovers most of the memory for roughly 2% overhead. That is the difference between "checkpointing is a last resort" and "selective checkpointing is nearly free," and it is why modern stacks checkpoint by default. The paper is also the cleanest published derivation of the activation memory formula this post leans on.

**FSDP fitting a 70B model on commodity cards.** The PyTorch FSDP paper and its follow-ups report training models in the tens of billions of parameters on clusters of 80 GB A100s that could not hold a single replica — a 70B model's 1120 GB of state, sharded across 64 cards, becomes 17.5 GB per card, and the run fits. The measured cost is the extra communication (roughly 1.5× the all-reduce volume of DDP for the ZeRO-3 gather-and-scatter), not extra memory; the memory is exactly the `16Ψ / N` the ledger predicts. Meta's published OPT-175B and Llama training logs are consistent with the same arithmetic.

**FlashAttention in production.** Dao et al. (2022) report that FlashAttention reduces the *memory* of the attention operation from `O(s²)` to `O(s)` while also speeding it up, by never materializing the score matrix. On a 7B model at 8k sequence, that is the difference between the ~344 GB quadratic term from our worked example and effectively zero, which is why every long-context training run treats FlashAttention as mandatory rather than an optimization. The activation ledger without FlashAttention simply does not close on any real card past a few thousand tokens.

**The GPT-3 / PaLM class of runs.** The reported configurations for models in the 100B–500B range consistently combine all four levers from the matrix at once — tensor parallelism inside a node, pipeline parallelism across nodes, data-parallel sharding on top, and activation checkpointing throughout — precisely because no single lever divides enough terms. PaLM's reported model FLOPs utilization of ~46% and GPT-3's cluster sizing are downstream of exactly this budget: they chose the parallelism layout that made every term fit while keeping the interconnect busy. The number to remember is not any one MFU figure but the pattern — **at frontier scale you pull every lever in the matrix simultaneously, because each one only divides its own subset of terms.**

## When to reach for each lever (and when not to)

The whole point of an honest ledger is to stop you from spending effort on the wrong term. So, plainly:

- **Reach for FSDP/ZeRO sharding when model state dominates** — a model whose `16Ψ` exceeds one card, dying before the first step. Do *not* reach for it to fix an activation OOM; sharding does not touch activations and you will OOM again at the same sequence length, having added communication overhead for nothing.
- **Reach for activation checkpointing when activations dominate** — long sequences, large micro-batches, a run that dies mid-step with a large failing allocation. Do *not* pay its ~30% recompute tax on a run whose state is the real problem; you will slow the run down and still OOM.
- **Reach for a lighter optimizer (8-bit Adam, Adafactor) when the optimizer term is the tall bar and you are one climb short of fitting.** It is the cheapest single win because the optimizer is usually the largest term, and it stacks with sharding.
- **Reach for `expandable_segments` when reserved far exceeds allocated** — the fragmentation fingerprint. It is a one-line environment change with no compute cost. Do not respond to fragmentation by shrinking the batch; you will waste throughput treating a symptom.
- **Reach for tensor parallelism only when you have fast intra-node NVLink and a single layer's activation is itself too big** — it is the only lever that divides activations by splitting the layer, but its per-layer all-reduce makes it a poor choice on PCIe or across nodes.
- **Do not go multi-node for memory until you have exhausted one node.** Eight 80 GB cards sharded is 640 GB of state capacity; if your model fits there, adding nodes buys you throughput, not fit, and costs you interconnect.

The meta-rule: **compute the budget by term first, identify the tall bar, then pick the one lever that divides that specific term.** Everything else is guessing.

## Putting it together: sizing a run from scratch

Here is the whole post as a procedure you can run before you ever touch `torchrun`. Given a model, an optimizer, a target sequence length, and a pool of GPUs, size the run in five steps — all arithmetic, no launches.

1. **Compute the state floor.** Multiply parameters by the bytes-per-param from the optimizer table (16 for bf16 + Adam, 10 for 8-bit Adam). Divide by your GPU memory. That quotient, rounded up, is the *minimum* shard degree — the fewest GPUs whose combined memory can hold the state. For a 13B AdamW model on 80 GB cards: `208 / 80 = 2.6`, so you need at least three-way sharding before you have stored one activation.
2. **Pick the shard degree with headroom.** Do not shard to exactly fit — leave room for activations and the eaters. Round the minimum up to a comfortable power of two or a full node. The 13B run wants at least four-way, and eight-way (one node) leaves the state at 26 GB per card and ~50 GB for everything else.
3. **Compute activations for micro-batch 1.** Plug your sequence length, hidden size, and layer count into `34·s·b·h·L` at batch 1. If that single-sequence activation already exceeds your remaining budget, you need checkpointing *before* you think about throughput. If it fits, the ratio of remaining-budget to single-sequence-activation is your maximum micro-batch.
4. **Set micro-batch, then accumulate to the target effective batch.** Take the largest micro-batch that fits from step 3 (with the 10–15% eater headroom subtracted). If your training recipe wants an effective batch larger than `micro_batch × shard_degree`, close the gap with gradient accumulation — it costs no extra activation memory.
5. **Add the levers only if a term is still too tall.** If activations still overflow, turn on checkpointing (or selective checkpointing) and recompute step 3. If the state overflows even at your maximum sensible shard degree, drop to an 8-bit optimizer or add offload. If reserved memory balloons past allocated once you launch, set `expandable_segments:True`.

Run those five steps and you arrive at the launch with a shard degree, a micro-batch, an accumulation count, and a checkpointing decision that you *derived* rather than discovered by trial and OOM. The estimator function from earlier automates steps 1 through 3; the matrix names the lever in step 5. This is the difference between the two engineers in the intro — not talent, not hardware, just five multiplications done before the job hit the queue.

## Key takeaways

- **Four consumers, one box.** Every training step pours parameters, gradients, optimizer states, and activations into a fixed GPU. Account for all four or be surprised.
- **Model state is `16Ψ` bytes for bf16 + Adam** — 2 for weights, 2 for grads, 12 for the fp32 optimizer trio. Parameter count times 16 is your per-GPU floor before activations. Memorize it.
- **Three terms are fixed; one is a choice.** Params, grads, and optimizer are fixed by the model and optimizer and can only be *divided* across GPUs. Activations are set by batch, sequence, precision, and checkpointing — the knobs you control.
- **Activations are `~34·s·b·h·L` bytes with FlashAttention**, quadratic in sequence without it. Past a few thousand tokens they dominate everything, and no amount of sharding touches them.
- **The hidden eaters are real:** the CUDA context (1–2 GB), fragmentation (the reserved-minus-allocated gap), temp buffers, comms buckets. Add 10–15% headroom and never fill the last 5 GB.
- **Estimate before you launch, measure after.** A 30-line memory estimator fails in a millisecond; `max_memory_allocated` and `memory_summary` confirm reality and expose fragmentation.
- **Match the lever to the term.** FSDP/ZeRO divides state; checkpointing divides activations; lighter optimizers shrink the `12`; `expandable_segments` fixes fragmentation. "Reduce batch size" is one leaf, not the root.
- **Read the OOM, don't just resubmit.** The failing allocation size and the timing tell you whether you hit a state, activation, or fragmentation wall — and each wall has a different lever.

## Further reading

- [Why distributed training](/blog/machine-learning/distributed-training/why-distributed-training) — the four walls and the map of the whole series; the memory wall is the first of them.
- [ZeRO and FSDP: the memory model](/blog/machine-learning/distributed-training/zero-and-fsdp-the-memory-model) — the full derivation of how the `16Ψ` state shards to `16Ψ / N` across ZeRO stages 1, 2, and 3.
- [Activation checkpointing](/blog/machine-learning/distributed-training/activation-checkpointing) — the compute-for-memory trade that fixes the activation term this post isolates.
- [Mixed precision at scale](/blog/machine-learning/distributed-training/mixed-precision-at-scale) — bf16 vs fp16 vs fp8, loss scaling, and why the optimizer master stays fp32.
- [The distributed training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) — the capstone decision and debugging checklist that ties the memory ledger to every other lever.
- [Out-of-memory debugging](/blog/machine-learning/debugging-training/out-of-memory-debugging) — the crash-side companion: reading tracebacks, allocator snapshots, and fragmentation reports when a run dies.
- Korthikanti et al. (2022), *Reducing Activation Recomputation in Large Transformer Models* — the source of the activation-memory formula and the selective-recompute result.
- Rajbhandari et al. (2020), *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* — the paper that names and defeats the `(2+2+12)Ψ` redundancy; and Dao et al. (2022), *FlashAttention* — how the quadratic attention term becomes linear.

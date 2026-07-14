---
title: "Anatomy of a Real Training Recipe: Every Knob, and Why"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "Walk a real Llama-style 6.7B config line by line on a 64-GPU cluster — parallelism, global batch, optimizer, schedule, precision, sharding, checkpoints — and learn what every knob buys, how it maps to memory and throughput, and exactly how it fails when you set it wrong."
tags:
  [
    "distributed-training",
    "multi-gpu",
    "multi-node",
    "fsdp",
    "pytorch",
    "adamw",
    "mfu",
    "mixed-precision",
    "gpu",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Distributed Training"
author: "Hiep Tran"
featured: true
readTime: 41
---

Someone hands you a config file. It is ninety lines of YAML with no comments, it trained a good 6.7-billion-parameter model last quarter, and your job is to adapt it — bigger context, a new data mix, a different cluster. You open it and there it is: `micro_batch_size: 2`, `lr: 3.0e-4`, `betas: [0.9, 0.95]`, `warmup_steps: 2000`, `sharding: FULL_SHARD`, `grad_clip: 1.0`. Every one of those numbers was chosen by someone who got paged when it was wrong. Change the wrong one and the loss NaNs at step one, or the run OOMs before the first optimizer step, or — the cruelest failure — it trains perfectly and quietly wastes a third of a 64-GPU reservation for three weeks.

This post takes one such recipe and reads it end to end, the way a principal engineer reads it: not as a list of magic numbers but as a stack of decisions, each one constraining the ones below it. We will train a nominal "7B" model — really about 6.7 billion parameters, the Llama-2-7B shape — on a 64-GPU cluster of eight-way H100 SXM nodes, and we will justify every knob against two budgets that never lie: the **memory** on each GPU, and the **throughput** of the whole job. By the end you will be able to open a strange config and know, for each line, what it fixes, where its cost lands, and how it fails.

![a vertical stack of seven recipe layers from launch at the top down to checkpoint at the bottom with each layer fixing the budget for the ones below it](/imgs/blogs/anatomy-of-a-real-training-recipe-1.webp)

The figure above is the whole recipe as a stack. Seven layers, top to bottom: the **launch** (how the processes start and find each other), the **parallelism** (how the model and data are split across 64 GPUs), the **batch** (how many tokens go into one optimizer step), the **optimizer** (AdamW and its clip), the **schedule** (warmup and cosine decay), the **precision** (bf16 plus activation checkpointing), and the **checkpoint** cadence. The order is not decorative. A choice at the top fixes the budget for everything beneath it: pick 64-way sharding and you have decided how much memory each GPU has left for activations, which decides your sequence length, which decides your tokens-per-step, which decides how many steps your schedule must cover. Read top-down and the recipe is a chain of consequences, not a pile of settings.

This is the thirty-ninth post in the [Distributed Training in the Trenches](/blog/machine-learning/distributed-training/why-distributed-training) series, and it is where the whole series gets assembled into one artifact. The [four walls](/blog/machine-learning/distributed-training/why-distributed-training) from the intro — the model won't fit, the data won't finish, the run is too slow, the cost is too high — are exactly the four things this config is fighting, and every knob is a lever against one of them. If you want the one-page decision-and-debug checklist instead of the full walkthrough, that is the [capstone playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook); this post is the annotated recipe that sits underneath it.

## The recipe, in one block

Here is the whole thing. Read it once; we will spend the rest of the post earning every line.

```yaml
# recipe.yaml — 6.7B decoder-only transformer, 64x H100 SXM (8 nodes x 8 GPUs)
model:
  name: llama-6.7b
  n_layers: 32
  hidden_size: 4096
  n_heads: 32
  vocab_size: 32000
  seq_length: 4096
  tie_embeddings: false

parallelism:
  strategy: fsdp             # full-shard data parallel (ZeRO-3 equivalent)
  data_parallel: 64          # world size = 8 nodes x 8 GPUs
  tensor_parallel: 1         # the model fits under FSDP; no TP needed
  pipeline_parallel: 1
  sharding: FULL_SHARD       # or HYBRID_SHARD for the two-axis mesh
  mixed_precision: bf16
  activation_checkpointing: full

batch:
  micro_batch_size: 2        # sequences per GPU per forward
  gradient_accumulation: 8   # micro-steps before one optimizer step
  # global batch = 2 x 8 x 64 = 1024 sequences = 4.19M tokens/step

optimizer:
  name: adamw
  lr: 3.0e-4
  betas: [0.9, 0.95]
  eps: 1.0e-8
  weight_decay: 0.1
  grad_clip: 1.0

schedule:
  type: cosine
  warmup_steps: 2000
  total_steps: 238000        # ~1T tokens / 4.19M tokens per step
  min_lr_ratio: 0.1          # floor = 3e-5

data:
  train_data: /data/tokenized/mix
  tokenizer: /data/tokenizer.model
  num_workers: 4
  prefetch_factor: 2

checkpoint:
  save_every: 1000
  keep_last: 3
  async_save: true
  dir: /checkpoints/llama-6.7b
```

Every number in that file traces to a budget. The table below is the same seven layers as the opening figure, but now with the question each layer answers and the wall it defends against. Keep it next to you; the sections that follow walk down it row by row.

| Layer | Config block | What it fixes | Wall it fights |
|---|---|---|---|
| Launch | `torchrun`/`srun` | How 64 processes start and rendezvous | Run too slow (start-up, placement) |
| Parallelism | `parallelism:` | How model state and data split across GPUs | Model won't fit |
| Batch | `batch:` | Tokens per optimizer step | Data won't finish / convergence |
| Optimizer | `optimizer:` | How gradients turn into weight updates | Run diverges (NaN, spikes) |
| Schedule | `schedule:` | Learning rate over time | Convergence speed and stability |
| Precision | `mixed_precision`, `activation_checkpointing` | Bytes per number, activation memory | Model won't fit / run too slow |
| Checkpoint | `checkpoint:` | Durability and restart cost | Run too slow (lost work on failure) |

## Layer 1: the launch and the device mesh

Nothing in the recipe runs until 64 processes — one per GPU — start on eight machines and agree on who is who. That agreement is the *rendezvous*, and it is the whole content of the launch command. Each process learns three numbers: its **global rank** (0–63, its unique id in the job), its **local rank** (0–7, which GPU on its own node), and the **world size** (64, how many peers exist). Get these wrong and the collectives that sync gradients will either hang forever or silently average the wrong tensors. This is exactly the ground covered in [launching on a SLURM cluster](/blog/machine-learning/distributed-training/launching-on-a-slurm-cluster), so here we keep it to the launch script and move on:

```bash
#!/bin/bash
#SBATCH --job-name=llama-6b7
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --time=21-00:00:00

export NCCL_DEBUG=WARN
export NCCL_IB_HCA=mlx5            # use the InfiniBand HCAs, not TCP fallback
export NCCL_SOCKET_IFNAME=eth0     # control-plane iface for the rendezvous
export OMP_NUM_THREADS=12

MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)

srun torchrun \
  --nnodes=8 \
  --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER:29500" \
  train.py --config recipe.yaml
```

The subtle knob here is not in the `torchrun` flags — it is `NCCL_IB_HCA=mlx5`. If NCCL cannot find the InfiniBand cards it will fall back to TCP over the management Ethernet, and your cross-node all-reduce will run at gigabytes-per-second instead of tens of gigabytes-per-second. The job still trains; it is just three times slower and nobody notices until the bill arrives. That single misconfiguration is the entire story of a later war story in this series, [multi-node slower than single-node](/blog/machine-learning/distributed-training/multinode-slower-than-single-node) — the launch is where it is born.

Once the 64 processes exist, they are not a flat list. They form a two-dimensional **device mesh**, and understanding that mesh is the difference between a recipe that scales and one that stalls.

![a two by four grid of GPUs where the top row shards a model along a fast intra node link and the bottom row syncs gradients across a slower cross node link](/imgs/blogs/anatomy-of-a-real-training-recipe-2.webp)

The grid above is the mesh (drawn four-wide for legibility; a real node is eight-wide). It has two axes because a cluster has two kinds of wire, and they differ by more than an order of magnitude:

- **Inside a node**, the eight GPUs talk over **NVLink / NVSwitch** — roughly 900 GB/s of aggregate bandwidth per GPU on H100 SXM. This is the fast axis.
- **Across nodes**, GPUs talk over **InfiniBand** — a typical H100 cluster runs multiple NDR links per node, but the *per-GPU* cross-node bandwidth you actually get for a collective is on the order of tens of GB/s, roughly 15–20× less than NVLink. This is the slow axis.

The recipe's `FULL_SHARD` strategy shards the model's parameters, gradients, and optimizer state evenly across all 64 GPUs (this is ZeRO-3 in DeepSpeed's vocabulary, and its memory math is the subject of [ZeRO and FSDP: the memory model](/blog/machine-learning/distributed-training/zero-and-fsdp-the-memory-model)). Full sharding is the simplest thing that fits, but it puts *both* the parameter all-gathers and the gradient reduce-scatters on whatever wire connects the shards — including the slow cross-node one. The refinement the mesh figure is really about is `HYBRID_SHARD`: shard the model **within** each node along the fast NVLink axis, and **replicate** it across nodes, so the only traffic that has to cross the slow InfiniBand axis is the gradient all-reduce between replicas. The heavy, latency-sensitive all-gather stays on NVLink; the once-per-step gradient sync is all that pays the InfiniBand tax.

| Mesh axis | Wire | Bandwidth (approx) | What rides it | Frequency |
|---|---|---|---|---|
| Intra-node (fast) | NVLink4 / NVSwitch | ~900 GB/s per GPU | Param all-gather, activation-time comms | Every layer |
| Inter-node (slow) | InfiniBand NDR | ~tens of GB/s per GPU | Gradient all-reduce between replicas | Once per step |

Why does splitting the traffic across two axes matter so much? Because the two collectives have wildly different tolerance for a slow wire, and that difference is provable, not aesthetic. A ring all-reduce of a gradient of size $S$ moves $2(N-1)/N \cdot S$ bytes across each GPU's link — approximately ${2S}$ for large $N$ (the full derivation is in [collectives from scratch](/blog/machine-learning/distributed-training/collectives-from-scratch)). Crucially, that gradient all-reduce happens **once per optimizer step** and, because DDP and FSDP fire it bucket-by-bucket as the backward pass completes, it **overlaps** with the compute it trails — the whole subject of [overlapping compute and communication](/blog/machine-learning/distributed-training/overlapping-compute-and-communication). A tensor-parallel all-reduce is the opposite on both counts: it happens **twice per layer** — 64 times per micro-step for a 32-layer model — and it sits **on the critical path**, because the next layer cannot start until the current layer's activation all-reduce finishes. It cannot hide behind compute. That is the comms-to-compute ratio that decides where each collective may live: the once-per-step, overlappable gradient sync can afford the slow inter-node wire; the per-layer, critical-path TP all-reduce cannot, and must stay on NVLink inside a node.

#### Worked example: is the cross-node gradient sync affordable?

Put numbers on it. The full bf16 gradient is $S \approx 13$ GB. A ring all-reduce moves $2 \times \frac{63}{64} \times 13 \approx 25.6$ GB across each GPU's link. On the fast intra-node axis (NVLink at ~900 GB/s) that is about 28 ms; on the slow inter-node axis (call it ~50 GB/s of effective per-GPU InfiniBand bandwidth) it is about 512 ms. The step time is 7.29 s, so even the pessimistic inter-node number is ~7% of the step — and most of it overlaps the backward pass, so the *exposed* cost is a couple of percent. That is why a once-per-step gradient sync survives the slow wire. Now contrast tensor parallelism: 64 critical-path all-reduces per micro-step, none of them overlappable, each stalling the pipeline until it completes. Run those over InfiniBand and the exposed comms would dwarf the compute — which is exactly why `tensor_parallel: 1` is the right call for a model that already fits under FSDP, and why the only reason to ever turn TP on is to keep it *inside* a node on NVLink.

This is the first place the recipe forces a real decision, and the decision framework is exactly [picking a parallelism strategy](/blog/machine-learning/distributed-training/picking-a-parallelism-strategy). For a 6.7B model the answer is easy: the model shards comfortably under FSDP, so we set `tensor_parallel: 1` and `pipeline_parallel: 1` and let data-parallel do all the work. Tensor and pipeline parallelism — the machinery of [3D parallelism](/blog/machine-learning/distributed-training/3d-parallelism) — only earn their considerable comms cost when a single layer's weights or activations no longer fit even after sharding, which for a 4096-hidden model on 80 GB cards they comfortably do. The general rule from the strategy post holds: do not reach for tensor parallelism until the model refuses to fit under pure data parallelism, and do not go multi-node until you have saturated one node.

## Layer 2: where the global batch comes from

Now the most misunderstood number in the file. The recipe never writes `global_batch_size` anywhere — and yet the global batch is the single most important convergence knob you own. It is not a setting; it is a *product* of three settings that live in different config blocks, and you have to multiply them in your head.

![three input knobs micro batch and gradient accumulation and data parallel degree merging into a single global batch that then combines with sequence length into tokens per step](/imgs/blogs/anatomy-of-a-real-training-recipe-3.webp)

The figure traces it. Three knobs fan into one:

$$
B_\text{global} = b_\text{micro} \times a_\text{accum} \times N_\text{DP}
$$

With the recipe's values, $b_\text{micro} = 2$ sequences per GPU per forward pass, $a_\text{accum} = 8$ micro-steps accumulated before we update, and $N_\text{DP} = 64$ data-parallel workers:

$$
B_\text{global} = 2 \times 8 \times 64 = 1024\ \text{sequences}.
$$

But sequences are not the currency the model optimizes over — **tokens** are. Multiply by the sequence length to get the quantity that actually controls the gradient's signal-to-noise ratio:

$$
T_\text{step} = B_\text{global} \times L = 1024 \times 4096 = 4{,}194{,}304 \approx 4.19\text{M tokens/step}.
$$

That ~4M-token global batch is not an accident — it is roughly the batch size the GPT-3 and Llama papers landed on for models in this size class, because it sits at the sweet spot where the gradient is stable enough to take a large step but not so large that you are wasting samples on a gradient you already know the direction of. The three knobs are *interchangeable* for convergence — the model only sees their product — but they are **not** interchangeable for the hardware:

- **Micro-batch** ($b_\text{micro}$) costs **activation memory**. Double it and you double the activations resident during the forward pass. It is capped by what fits on the GPU.
- **Grad-accumulation** ($a_\text{accum}$) costs **wall-clock, not memory**. Each micro-step is a full forward and backward with the gradients summed in place; eight of them take eight times as long but use the memory of one. It is how you reach a big global batch when memory won't let micro-batch grow.
- **Data-parallel degree** ($N_\text{DP}$) costs **GPUs and cross-GPU comms**. It is the only one that also buys you speed — more workers means more tokens per unit time.

#### Worked example: hitting a 4M-token batch three different ways

Suppose your target global batch is fixed at ~4M tokens and you have exactly 64 GPUs. You have freedom in how you split micro-batch against accumulation, and the choice is a pure memory-versus-time trade.

- **Config A — `micro=8, accum=2`:** activations for 8 sequences of 4096 tokens must fit. On an 80 GB H100 with activation checkpointing that is fine, and each step is only 2 accumulation passes, so wall-clock per step is lowest. This is what you pick when memory is plentiful.
- **Config B — `micro=2, accum=8`** (the recipe): activations for only 2 sequences resident, 8 accumulation passes. Slightly slower per step because there is less work to overlap comms against, but it leaves ~30 GB of headroom you can spend on a longer sequence or a fatter data-loader prefetch.
- **Config C — `micro=1, accum=16`:** the fallback when you bump the sequence to 8192 and activations for even 2 sequences no longer fit. It always reaches the same 4M-token batch; it just pays the most wall-clock.

All three train the *same model to the same place* — the loss curve is identical to within noise — because the optimizer only ever sees the 4M-token average gradient. The recipe chose B: enough micro-batch to keep the GPU busy, enough headroom to be safe. If you get this wrong in the other direction — push the global batch to 16M tokens to "go faster" — you do not go faster; you hit the large-batch wall where each step makes barely more progress than a 4M-token step did, and you have simply spent 4× the tokens to reach the same loss. That failure is why the [scaling a 7B LLM](/blog/machine-learning/distributed-training/scaling-a-7b-llm-1-to-64-gpus) war story treats "batch too big to converge" as the fourth and final wall.

## Layer 3: the optimizer

Five numbers, and each one is load-bearing:

```yaml
optimizer:
  name: adamw
  lr: 3.0e-4
  betas: [0.9, 0.95]
  eps: 1.0e-8
  weight_decay: 0.1
  grad_clip: 1.0
```

**AdamW** is the default for a reason: it adapts the step size per-parameter using running estimates of the gradient's first moment (the mean, controlled by $\beta_1$) and second moment (the uncentered variance, controlled by $\beta_2$). The recipe sets `betas: [0.9, 0.95]`, and that second value is a deliberate departure from Adam's textbook default of 0.999. A $\beta_2$ of 0.999 averages the variance estimate over roughly the last 1000 steps; at the scale of large-model pretraining, gradients shift regime faster than that, and a stale variance estimate under-scales a genuinely large gradient and lets a spike through. Dropping to **0.95** shortens that memory to ~20 steps, which makes the optimizer far more responsive to sudden changes in gradient magnitude — the single most common stability fix in large-model recipes, adopted by GPT-3, Llama, and essentially every recipe since.

`eps: 1.0e-8` is the floor added under the square-rooted second moment so you never divide by zero; it rarely needs touching, though some large-model recipes raise it to `1e-5` for extra bf16 stability. `weight_decay: 0.1` is the "W" in AdamW — a decoupled pull of every weight toward zero each step, which is the main regularizer in a pretraining run where you never see the same data twice. 0.1 is the community-standard value.

The one that saves runs is **`grad_clip: 1.0`**. Before every optimizer step, the recipe computes the global L2 norm of all gradients and, if it exceeds 1.0, scales the whole gradient vector down to norm 1.0. Pretraining loss curves are not smooth — a single unlucky batch (a run of repeated tokens, a corrupted document) can produce a gradient ten times the normal magnitude, and without clipping that one bad step blows the weights out of the good basin and the loss never recovers. Clipping to 1.0 caps the damage of any single step. Turn it off and, sooner or later, the loss explodes; this is not a maybe.

Under FSDP the clip is not a local operation — each rank holds only a shard of the gradients, so the global norm requires an all-reduce across ranks. PyTorch's `model.clip_grad_norm_` handles that for you, which is why you call the FSDP method and not `torch.nn.utils.clip_grad_norm_`:

```python
import math
from torch.optim import AdamW

optimizer = AdamW(
    model.parameters(),
    lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1,
)

for step in range(total_steps):
    optimizer.zero_grad(set_to_none=True)
    for micro in range(grad_accum):                 # 8 micro-steps
        batch = next(loader)                        # 2 seqs x 4096 tokens
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(batch).loss / grad_accum   # scale for accumulation
        loss.backward()                             # FSDP reduce-scatters grads
    model.clip_grad_norm_(max_norm=1.0)             # FSDP-aware global clip
    optimizer.step()
    scheduler.step()
```

Two details in that loop are easy to get wrong and expensive to debug. First, the `loss / grad_accum`: because we sum gradients over 8 micro-steps, we must divide each contribution by 8 or the effective learning rate is 8× too large and the run diverges at step one. Second, the clip happens **after** the accumulation loop and **before** `optimizer.step()` — clip inside the loop and you clip each micro-gradient separately, which is a different and weaker operation.

## Layer 4: the learning-rate schedule

The learning rate is not a constant. It follows a shape over the whole run, and the shape matters as much as the peak value.

![a left to right learning rate schedule that ramps from zero through a short warmup to a peak then decays on a cosine curve to a small floor](/imgs/blogs/anatomy-of-a-real-training-recipe-4.webp)

The timeline above is the recipe's schedule: `warmup_steps: 2000`, `type: cosine`, peak `lr: 3.0e-4`, floor `min_lr_ratio: 0.1` (so 3e-5), over `total_steps: 238000`. Three phases, each defending against a specific failure.

**Warmup** ramps the learning rate linearly from zero to the peak over the first 2000 steps. This exists because AdamW is *cold* at step zero: its second-moment estimate $v$ starts at zero, so the per-parameter step size $\eta / (\sqrt{v} + \epsilon)$ is enormous and wildly unstable for the first few dozen steps. Hit those steps with the full 3e-4 and you get the classic **NaN at step one**. Warmup gives Adam a couple of thousand steps to build a sane variance estimate before the learning rate is allowed to reach full strength. Two thousand steps is the community default for this scale; the caution node in the figure is what happens if you skip it.

**Cosine decay** takes the learning rate from its peak down along the right half of a cosine curve toward the floor. The intuition is that early training wants large steps to travel far across the loss landscape, and late training wants small steps to settle into a minimum without bouncing out. Cosine is the smooth interpolation between the two that has won empirically across almost every large-model recipe. The recipe decays to 10% of peak rather than to zero, because a nonzero floor keeps the model gently learning through the long tail instead of freezing.

The `total_steps: 238000` is not a free parameter — it is *derived* from the token budget and the batch. We are training on 1 trillion tokens (a defensible, roughly compute-optimal budget for a 6.7B model per the [Chinchilla scaling law](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling), which suggests ~20 tokens per parameter), so:

$$
\text{total\_steps} = \frac{T_\text{budget}}{T_\text{step}} = \frac{1 \times 10^{12}}{4.19 \times 10^{6}} \approx 238{,}600.
$$

This coupling is the single most common recipe bug when people adapt a config. If you change the global batch (say you got more GPUs and doubled $N_\text{DP}$) but forget to halve `total_steps`, the cosine schedule now decays over twice as many steps as you actually run, so the learning rate never reaches its floor, and the model finishes under-trained with the LR still high. The schedule length must always be recomputed from `token_budget / tokens_per_step`. A modern alternative worth knowing is the **WSD** schedule (warmup–stable–decay): warm up, hold the peak constant through the bulk of training, then decay sharply only at the very end. WSD's advantage is that the "stable" plateau is checkpoint-friendly — you can stop at any point, run the short decay, and get a usable model — which cosine, with its continuous decay, cannot offer.

## Layer 5: every knob, and how it fails

Before we drill into precision and memory, step back and look at the six load-bearing knobs together — because the most useful thing to know about a config is not what each line does but how each line *breaks*.

![a table of six core configuration knobs with their recipe value the behavior each governs and the specific failure that follows when each is set wrong](/imgs/blogs/anatomy-of-a-real-training-recipe-5.webp)

The matrix above is the debugging map for the whole recipe. Read it right-to-left when something goes wrong: a symptom in the third column points back to exactly one line in the first. This one-to-one property is what makes a good recipe debuggable — most failures have a single cause because most knobs govern a single behavior.

- **Global batch** governs gradient noise and the number of steps. Too big and the loss *stalls* — each step barely improves on the last, because you have driven the gradient noise so low that more samples add no information.
- **Peak LR + warmup** governs convergence speed. Too high, or warmup skipped, and you NaN in the first handful of steps as cold-Adam overshoots.
- **Grad clip** governs spike resistance. Off, and one bad batch's oversized gradient sends the loss to infinity.
- **bf16 + fp32 optimizer** governs the speed-versus-stability trade. Use pure fp16 instead and the gradients overflow fp16's narrow exponent range and NaN.
- **Activation checkpointing** governs activation memory. Off, and at sequence length 4096 the activations OOM the card before the backward pass finishes.
- **FSDP full shard** governs state memory per GPU. None, and the 107 GB of unsharded state OOMs an 80 GB card before step one.

Notice that four of the six failures are the *same symptom* — OOM or NaN — reached by four different roads. That is why the mapping matters: "the run NaN'd" is not a diagnosis, but "the run NaN'd and I'm using pure fp16" is. The debugging posts in this series — [debugging distributed jobs](/blog/machine-learning/distributed-training/debugging-distributed-jobs) and the mixed-precision material in [mixed precision at scale](/blog/machine-learning/distributed-training/mixed-precision-at-scale) — are essentially expansions of individual cells in this matrix.

## Layer 6: precision, checkpointing, and the memory budget

Now the layer that decides whether the run fits at all. Two knobs — `mixed_precision: bf16` and `activation_checkpointing: full` — plus the sharding strategy from Layer 1 together determine how many bytes sit on each GPU. To see why they are non-negotiable, do the memory arithmetic for our 6.7B model.

Let $\Psi = 6.7 \times 10^9$ be the parameter count. A mixed-precision AdamW step keeps several things resident, and the standard accounting (derived in full in [the memory budget](/blog/machine-learning/distributed-training/the-memory-budget)) is:

$$
M_\text{state} = \underbrace{2\Psi}_{\text{bf16 weights}} + \underbrace{2\Psi}_{\text{bf16 grads}} + \underbrace{12\Psi}_{\text{AdamW fp32 states}} = 16\Psi.
$$

The 12Ψ optimizer term is the one that surprises people: AdamW keeps an fp32 master copy of the weights (4Ψ), plus fp32 first-moment (4Ψ) and second-moment (4Ψ) estimates. Plug in the numbers:

$$
M_\text{state} = 16 \times 6.7 \times 10^9 = 1.07 \times 10^{11}\ \text{bytes} = 107\ \text{GB}.
$$

![a vertical memory budget for one GPU where the fp32 optimizer state is the tallest bar and sharding divides the whole stack down to under two gigabytes](/imgs/blogs/anatomy-of-a-real-training-recipe-6.webp)

The figure makes the shape of the problem obvious. The tallest bar by far is the **AdamW fp32 state at 80 GB** — bigger than the weights (13 GB) and gradients (13 GB) *combined*, and on its own already more than an 80 GB H100 can hold. Everyone quotes the weight count when they size a model; it is the optimizer that actually decides whether the run fits. You could delete the entire model from memory and the optimizer state alone would still overflow the card.

| Line item | Formula | Bytes for 6.7B | Full-shard per GPU (÷64) |
|---|---|---|---|
| Weights (bf16) | 2Ψ | 13 GB | 0.2 GB |
| Gradients (bf16) | 2Ψ | 13 GB | 0.2 GB |
| AdamW state (fp32) | 12Ψ | 80 GB | 1.3 GB |
| **Sharded state total** | 16Ψ | **107 GB** | **~1.7 GB** |
| Activations (checkpointed) | ∝ batch × seq | ~8 GB | ~8 GB (not sharded) |

This is where the recipe's three memory knobs earn their place, in order of leverage:

**`sharding: FULL_SHARD`** is the big one. FSDP splits the 107 GB of state evenly across all 64 GPUs, so each card holds only $107 / 64 \approx 1.7$ GB of state instead of the whole thing. This is the entire reason the run fits — it is the single knob that turns "impossible on any single 80 GB card" into "under 2 GB per card." The success node in the figure is that 1.7 GB shard.

**`mixed_precision: bf16`** halves the two largest tensors that move on the wire. Weights and gradients live in bf16 (2 bytes) instead of fp32 (4 bytes), which halves both the memory *and* the all-gather / reduce-scatter traffic — a throughput win as much as a memory one. The critical detail is *why bf16 and not fp16*: bf16 has the same 8-bit exponent as fp32, so it has fp32's dynamic range and simply cannot overflow the way fp16 does. That is why the recipe needs no loss scaler at all, and why the matrix's fourth row lists "pure fp16 → overflow NaN" as a failure mode. The optimizer states, meanwhile, stay in fp32 — reducing the second moment in bf16 loses too much precision on the small values and stalls convergence. This precision split is the whole subject of [mixed precision at scale](/blog/machine-learning/distributed-training/mixed-precision-at-scale).

**`activation_checkpointing: full`** attacks the one term that does not shard. Activations — everything the forward pass must save for the backward pass — scale with batch and sequence length, not with Ψ, and they sit fully on each GPU. At sequence 4096 with a 32-layer model, storing every layer's activations would run to tens of GB per card. Full checkpointing keeps only the input to each transformer block and *recomputes* the internal activations during the backward pass, trading roughly 30% more compute for a large drop in activation memory. That trade is exactly the topic of [activation checkpointing](/blog/machine-learning/distributed-training/activation-checkpointing); here it is what keeps activations down to the ~8 GB that leaves room for everything else.

Here is the code that expresses all three knobs, and this is the heart of the practical recipe — the FSDP wrap:

```python
import functools
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing, checkpoint_wrapper, CheckpointImpl,
)
from model import TransformerBlock   # your decoder block class

# bf16 for compute AND comms; fp32 kept only inside the optimizer step
bf16_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,     # gradient reduce-scatter in bf16
    buffer_dtype=torch.bfloat16,
)

# shard at the transformer-block granularity so each all-gather is one block
wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={TransformerBlock},
)

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,   # ZeRO-3 equivalent
    mixed_precision=bf16_policy,
    auto_wrap_policy=wrap_policy,
    device_id=torch.cuda.current_device(),
    limit_all_gathers=True,          # bound prefetch so peak memory stays flat
    use_orig_params=True,            # keep param groups / torch.compile happy
)

# recompute each block's internal activations during backward
apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=functools.partial(
        checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
    ),
    check_fn=lambda m: isinstance(m, TransformerBlock),
)
```

The `transformer_auto_wrap_policy` is the knob most people miss. It tells FSDP to make each transformer block its own sharded unit, so the runtime all-gathers one block's parameters, uses them, and frees them before gathering the next — keeping peak memory at roughly one block's worth of full parameters rather than the whole model's. Wrap too coarsely (the whole model as one unit) and you lose the memory savings; wrap too finely (every linear layer) and you drown in tiny all-gathers. Per-block is the sweet spot, and it is codified in [FSDP in practice](/blog/machine-learning/distributed-training/fsdp-in-practice). If you prefer DeepSpeed, the identical memory model is expressed as a ZeRO-3 config, and the [DeepSpeed ZeRO and 3D parallelism deep-dive](/blog/machine-learning/open-source-library/deepspeed-zero-3d-parallelism-deep-dive) covers it — the same 107 GB gets sharded the same way:

```json
{
  "train_micro_batch_size_per_gpu": 2,
  "gradient_accumulation_steps": 8,
  "bf16": { "enabled": true },
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6,
    "contiguous_gradients": true
  }
}
```

#### Worked example: will it fit on an 80 GB H100?

Do the fit calculation before you launch — it takes thirty seconds and saves an hour of watching a job OOM at step zero.

- **State, sharded:** $16\Psi / 64 = 107\,\text{GB} / 64 \approx 1.7$ GB.
- **Activations, checkpointed:** with `micro=2`, seq 4096, 32 layers, roughly 8 GB (checkpointing keeps only block inputs).
- **FSDP working buffers:** the all-gather of one block's full-precision parameters plus a prefetch of the next — on the order of a few GB with `limit_all_gathers=True`.
- **CUDA context + fragmentation + NCCL buffers:** call it 3–5 GB of overhead.

Sum: roughly 1.7 + 8 + 4 + 4 ≈ **18 GB per GPU** actively used, against an 80 GB budget. The training log at step 12000 will confirm it: `peak_mem 61.4 GB/80 GB` — the extra headroom over our 18 GB estimate is transient recompute and the largest all-gather. Sixty gigabytes leaves comfortable room, which is exactly what you want: a run at 78 GB/80 GB is one unlucky-length batch away from an OOM crash on hour 400. If your estimate comes out over ~70 GB, that is the signal to drop micro-batch to 1, enable a more aggressive sharding of activations, or shorten the sequence — *before* you launch, not after.

## Layer 7: data, tokenizer, and checkpoint cadence

Two remaining blocks, both about not wasting the expensive part of the run.

The **data** block points at a pre-tokenized, sharded dataset and a fixed tokenizer:

```yaml
data:
  train_data: /data/tokenized/mix
  tokenizer: /data/tokenizer.model
  num_workers: 4
  prefetch_factor: 2
```

Three things are load-bearing here. First, the data is **pre-tokenized** — you never tokenize on the fly in a pretraining run, because the CPU cannot keep 64 hungry H100s fed if it is also running a BPE tokenizer per batch. Second, the **data mix** is baked into `/data/tokenized/mix` — the ratio of web text to code to books to math, which for a general model is typically dominated by filtered web data with meaningful code and reference fractions. Get the mix wrong and no optimizer setting will save the model's downstream quality; the mix is a first-class recipe decision even though it is one line of config. Third, `num_workers: 4` and `prefetch_factor: 2` size the data-loader so the next batch is always ready before the GPU asks for it. This is the whole subject of [the data pipeline at scale](/blog/machine-learning/distributed-training/the-data-pipeline-at-scale): if the loader starves, your expensive MFU number collapses and no amount of GPU tuning helps, because the GPUs are sitting idle waiting for tokens.

The **checkpoint** block is insurance against the thing that *will* happen on a three-week, 64-GPU run: a node will fail.

```yaml
checkpoint:
  save_every: 1000
  keep_last: 3
  async_save: true
  dir: /checkpoints/llama-6.7b
```

`save_every: 1000` is a pure cost trade. Each step is ~7 seconds, so 1000 steps is about two hours; if a node dies, you lose at most two hours of the run's progress. Save more often and you lose less work per failure but spend more total time writing multi-hundred-gigabyte checkpoints; save less often and a failure at step 4999 throws away nearly five hours. Two hours is the standard compromise at this scale. `async_save: true` writes the checkpoint on a background thread so training does not stall for the minute-plus it takes to serialize a sharded checkpoint to storage — the mechanics of which are the subject of [distributed checkpointing](/blog/machine-learning/distributed-training/distributed-checkpointing). And the checkpoint must be **sharded**: each rank saves only its own 1.7 GB shard, in parallel, rather than gathering the whole 107 GB to rank 0 and writing it single-threaded. On a 64-GPU job the difference is a checkpoint that saves in seconds versus one that stalls the whole run for minutes every thousand steps.

## What the recipe buys at scale

Every choice above was made to serve two numbers: memory (does it fit) and throughput (how fast). We have shown it fits. Now measure the throughput honestly, and see what the whole recipe actually buys.

The north-star metric is **MFU — Model FLOPs Utilization** — the fraction of the GPUs' theoretical peak arithmetic that your model's math actually consumes. The denominator is the hardware's peak (H100 SXM ≈ 989 bf16 TFLOP/s, dense). The numerator uses the standard estimate that a transformer forward-plus-backward pass costs about $6\Psi$ FLOPs per token (derived in the [Chinchilla compute-optimal scaling](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) post):

$$
\text{MFU} = \frac{6\,\Psi \cdot (\text{tokens/s per GPU})}{\text{peak FLOP/s per GPU}}.
$$

Here is what an honest measurement looks like — a real log line from steady state, well past warmup, with `torch.cuda.synchronize()` before the timer so we are timing the GPU and not the Python launch:

```console
step 12000 | loss 2.184 | lr 2.94e-04 | grad_norm 0.61
  tokens/s (global) 574,880 | tokens/s/gpu 8,983
  MFU 37.2% | step_time 7.29s | peak_mem 61.4 GB/80 GB
  eta 19.8 days | tokens_seen 5.03e10 / 1.00e12
```

Check the MFU by hand: $6 \times 6.7\times10^9 \times 8983 = 3.61 \times 10^{14}$ FLOP/s per GPU, divided by $9.89 \times 10^{14}$ peak, gives 36.5% — matching the logged 37.2% (the small gap is the recompute from activation checkpointing, which does real FLOPs the $6\Psi$ estimate does not count). That number, 37%, is a genuinely good MFU for a sharded multi-node run; anything above ~35% at 64 GPUs on InfiniBand means the recipe is well-tuned and comms are overlapping properly.

![a two column comparison of the same recipe on eight GPUs versus sixty four GPUs showing throughput rising more than seven times while efficiency barely drops](/imgs/blogs/anatomy-of-a-real-training-recipe-7.webp)

The before-after figure is the payoff. Run the *identical recipe* on one node (8 GPUs, all-NVLink, no InfiniBand in the loop) and you get ~78,000 tokens/s at 40% MFU — higher MFU, because no gradient ever has to cross the slow inter-node wire. Scale to eight nodes (64 GPUs) and throughput rises to ~576,000 tokens/s, a **7.4× speedup for 8× the GPUs**, while MFU dips only from 40% to 37% — the 3-point drop is the InfiniBand tax on the gradient all-reduce, and 7.4/8 = 92% scaling efficiency is an excellent result for multi-node.

What that buys in the currency that matters:

$$
t_\text{1-node} = \frac{10^{12}}{78{,}000} \approx 1.28\times10^7\,\text{s} = 148\ \text{days};
\quad
t_\text{8-node} = \frac{10^{12}}{576{,}000} \approx 1.74\times10^6\,\text{s} = 20\ \text{days}.
$$

The same recipe, same config file, turns a **five-month run into three weeks**. That is the entire argument for multi-node in one comparison: you do not train faster per GPU (you train slightly slower per GPU), but you finish in a fraction of the wall-clock, and for a model with a deadline, wall-clock is the whole game.

#### Worked example: the GPU-hours and the dollar bill

Cost is just throughput times a rate, and it is the fourth wall.

- **GPU-hours:** 64 GPUs × 20.1 days × 24 h ≈ **30,900 GPU-hours** for the full 1T-token run.
- **Dollars:** at roughly \$2.50 per H100 GPU-hour (a typical negotiated cloud rate), that is about **\$77,000**; at \$3.00 it is about \$93,000. Call it \$75k–\$95k for the run.
- **Cost per token:** ~\$85k / 1e12 tokens ≈ **\$0.085 per million tokens** of training.

Now watch what a bad knob does to that bill. Suppose the `NCCL_IB_HCA` line from Layer 1 is missing and the job silently falls back to TCP: MFU drops from 37% to, say, 22%, throughput falls to ~340k tokens/s, the run stretches to ~34 days, and the bill grows to ~\$130,000. Same model, same final loss, **\$50,000 of pure waste** — and the only symptom was a throughput number nobody was watching. This is why the [cost and efficiency at scale](/blog/machine-learning/distributed-training/cost-and-efficiency-at-scale) post insists that MFU is not a vanity metric; it is the dollar meter. A recipe that trains correctly but at half the MFU is not "a bit slow" — it is a run that costs twice as much for the identical result.

### Measuring throughput without fooling yourself

That 37% number is easy to report and easy to get wrong. Five confounds turn a "measured" MFU into a fiction, and every one of them has bitten a real run:

- **Warm-up.** The first ~50 steps are a lie. The CUDA caching allocator is still settling, cuDNN and any autotuner are still picking kernels, and the GPU clocks are still ramping from idle. Discard the first 50–100 steps before you time anything; measure only steady state.
- **Asynchronous CUDA.** Kernel launches return immediately — the GPU runs behind the Python. If you time a step with `time.time()` and no `torch.cuda.synchronize()`, you are timing how fast Python enqueued the work, not how fast the GPU did it, and your tokens/s will look impossibly high. Synchronize before you start and stop the clock.
- **The median, not the mean.** Every thousandth step writes a checkpoint and every step that crosses a data-shard boundary reloads; those are outliers an order of magnitude slower. Take the median step time over a window of 100+ steps, or the mean will be dragged down by a handful of I/O stalls and understate your true throughput.
- **The data-loader confound.** If tokens/s is low, it is not always the GPU. A starved loader leaves the GPU idle waiting for the next batch, and the symptom — low throughput — looks identical to a slow kernel. The tell is GPU utilization: watch `nvidia-smi` or DCGM, and if utilization dips below ~95% during the step, the [data pipeline](/blog/machine-learning/distributed-training/the-data-pipeline-at-scale), not the model, is your wall. No optimizer knob fixes a hungry GPU.
- **Thermal and clock throttling.** A dense node running flat out for weeks can hit a thermal or power limit and drop its clocks, quietly shaving 5–10% off throughput. If one node is consistently slower than its peers, check its clocks with `nvidia-smi -q -d CLOCK` before you blame the network.

There is also a definitional trap worth naming: some reports quote **HFU (Hardware FLOPs Utilization)**, which counts the recompute FLOPs from activation checkpointing, and some quote **MFU**, which counts only the "useful" $6\Psi$ per token. HFU is always the larger number because recompute is real arithmetic the model didn't strictly need. Neither is wrong, but they are not comparable — when you cite a utilization figure, say which one it is, or you will "improve" your MFU by 10 points simply by switching which formula you print.

## The recipe under stress

A recipe that trains cleanly on the happy path is only half a recipe. The other half is knowing how it behaves when the cluster misbehaves — and at 64 GPUs for three weeks, it will. Here are the four stress cases that actually happen, and what in the config does (and does not) save you.

**One node becomes a straggler.** Data-parallel training is only as fast as its slowest worker: every step ends in a gradient all-reduce, and the all-reduce is a barrier — 63 GPUs sit idle waiting for the one node whose clocks throttled or whose NIC is flaking. Throughput collapses to the speed of the slowest rank, and the average GPU-utilization dashboard still reads high, because the fast GPUs are "busy" waiting. Nothing in `recipe.yaml` prevents this; the fix is observability, not configuration. Log per-rank step time, find the rank whose median is 15% above the pack, and evict its node from the reservation — the full diagnosis is [the straggler](/blog/machine-learning/distributed-training/the-straggler). The recipe's job is only to make the eviction cheap: because checkpoints are sharded and saved every 1000 steps, you lose at most two hours restarting on healthy nodes.

**The loss spikes right after a resume.** A node dies at step 40,000, you restart from the step-40,000 checkpoint, and the loss jumps by half a nat before recovering. This is almost never the model — it is checkpoint *incompleteness*. A correct resume must restore three things beyond the weights: the **optimizer state** (both AdamW moments, or the schedule's momentum is wrong), the **RNG state** (or dropout and data shuffling diverge from the original trajectory), and the **data-loader position** (or you replay data the model already saw this epoch, and re-seeing tokens spikes the loss). The recipe's `async_save: true` sharded checkpoint has to serialize all three, not just `model.state_dict()`. If your resume spikes, the checkpoint is missing one of them — that exact autopsy is [the loss spike after resume](/blog/machine-learning/distributed-training/the-loss-spike-after-resume), and the correct sharded-save mechanics are in [distributed checkpointing](/blog/machine-learning/distributed-training/distributed-checkpointing).

**You shrink the cluster for a debug run.** You want to reproduce a bug on 8 GPUs instead of 64, so you drop `data_parallel` to 8 and leave everything else alone. Now the global batch is $2 \times 8 \times 8 = 128$ sequences — a quarter of a million tokens instead of four million — and two things break silently. First, the learning-rate schedule was tuned for the 4M-token batch; at 0.5M tokens the same peak LR is effectively far more aggressive per token, and the debug run may diverge in ways the real run never would, sending you chasing a bug that does not exist. Second, `total_steps` is still 238,000, so the cosine barely decays over your short debug run. The rule: when you change the cluster, the batch changes, and the schedule must be recomputed from it. A debug run at a different batch is a *different recipe*, and it is why reproducing a large-scale bug at small scale is genuinely hard — the [determinism-across-ranks](/blog/machine-learning/distributed-training/determinism-across-ranks) discipline exists precisely to make small-scale repros trustworthy.

**You scale the model to 13B on the same cluster.** Double the parameters and the state doubles: $16\Psi = 16 \times 13\times10^9 \approx 208$ GB, which sharded 64 ways is 3.25 GB per GPU — still comfortable. But activations do not shard, and a 13B model has more layers and a wider hidden dimension, so activation memory climbs even with full checkpointing, and the FSDP all-gather of one block's parameters is now larger and more likely to collide with the compute it should overlap. The first lever is to keep FSDP full-shard but tighten `limit_all_gathers` and drop micro-batch to 1; if that is not enough, the model has outgrown pure data parallelism and it is time to add tensor parallelism inside each node — the [3D parallelism](/blog/machine-learning/distributed-training/3d-parallelism) path. The stress test tells you *when* the simple recipe stops being enough: the moment activations plus one block's full-precision parameters no longer leave headroom on the card, no amount of sharding the optimizer state will save you, because the optimizer state was never the term that grew.

The through-line across all four is the series' governing idea: the recipe encodes a set of assumptions — one healthy cluster, a fixed batch, a model that fits under FSDP — and each stress case is one assumption breaking. The config does not defend against broken assumptions; observability and the right next lever do. That is why the [capstone playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) pairs every recipe knob with the failure it invites and the tool that catches it.

## Case studies: real recipes in the wild

None of the numbers above are invented — they sit squarely in the range of published pretraining recipes for models in this class. A few real ones, with the knobs that matter:

- **Llama-2-7B** (Meta, 2023) trained on 2 trillion tokens with AdamW, betas (0.9, 0.95), weight decay 0.1, gradient clipping 1.0, a cosine schedule with 2000 warmup steps decaying to 10% of a 3e-4 peak, and a global batch of 4M tokens — which is exactly the recipe we have been reading, because it *is* the canonical recipe for this size. It ran on a cluster of A100 GPUs.
- **OLMo-7B** (AI2, 2024) is the fully-open reference: the same betas and clipping, a 4M-token batch, AdamW, trained on ~2.5T tokens of the Dolma mix, on 27 nodes of 8× A100. AI2 published the entire config and the training logs, and the reported throughput and MFU are in the range this post uses — OLMo is the recipe you can actually read end to end.
- **SmolLM / SmolLM2** (Hugging Face) apply the same skeleton at 1.7B and below, showing the recipe scales *down* cleanly: the parallelism gets simpler (the model fits on fewer GPUs, sometimes without any sharding), but the optimizer, the betas, the clip, and the warmup-plus-cosine schedule are unchanged. The recipe is a template, not a one-off.
- **GPT-3** (OpenAI, 2020) is where several of these defaults were popularized at scale: the gradual batch-size warmup, the cosine decay to 10% of peak, and the ~3e-4-class peak learning rates for models in the few-billion range. Later recipes standardized the betas-(0.9, 0.95) tweak that GPT-3 and its successors found necessary for stability.

| Recipe | Params | Tokens | Global batch | Peak LR | Warmup | Betas | Clip |
|---|---|---|---|---|---|---|---|
| Llama-2-7B | 6.7B | 2.0T | ~4M tok | 3.0e-4 | 2000 | 0.9, 0.95 | 1.0 |
| OLMo-7B | 6.9B | ~2.5T | ~4M tok | 3.0e-4 | ~2000 | 0.9, 0.95 | 1.0 |
| SmolLM2-1.7B | 1.7B | ~11T | ~2M tok | ~5e-4 | linear | 0.9, 0.95 | 1.0 |
| GPT-3 (6.7B) | 6.7B | 300B | ramped to ~3M | 1.2e-4 | ~375 tok-M | 0.9, 0.95 | 1.0 |

The striking thing across the table is how *little* varies. The optimizer, the betas, the clip norm, the warmup-then-cosine shape — these are effectively fixed across the whole industry for dense decoder pretraining. What varies is the token budget (which follows the compute you can afford), the exact peak LR (which scales gently with model size), and the parallelism plumbing (which follows the cluster). When you inherit a config, the load-bearing question is almost never "are the betas right" — they are 0.9/0.95 — but "is the global batch, the schedule length, and the sharding matched to *this* model on *this* cluster."

## When to reach for this recipe (and when not)

This exact recipe — FSDP full-shard, bf16, activation checkpointing, ~4M-token batch, AdamW with cosine — is the right default for **dense decoder-only models from roughly 1B to 13B parameters, on a cluster where each model replica fits under data-parallel sharding**. That covers the large majority of pretraining and continued-pretraining work. Reach for it as your starting point and change only what your situation forces you to change.

Change it when:

- **The model is much larger (30B+).** Once even a single layer's parameters or activations strain an 80 GB card after sharding, pure FSDP stops being enough and you compose in tensor parallelism inside each node and possibly pipeline parallelism across nodes — the full [3D parallelism](/blog/machine-learning/distributed-training/3d-parallelism) machinery. Do not add this complexity for a 7B model; it is pure overhead when FSDP already fits.
- **The context is very long (32k+).** Activations grow with sequence length, and past a point even full checkpointing does not save you; that is when sequence/context parallelism enters. For 4096, plain checkpointing is plenty.
- **The interconnect is weak.** On a cluster without InfiniBand — GPUs connected only over slow Ethernet or PCIe — the gradient all-reduce dominates and multi-node scaling collapses. The honest move there is to stay single-node, or switch to `HYBRID_SHARD` so the only cross-node traffic is the once-per-step gradient sync.
- **The model is small enough to fit unsharded (sub-1B on 80 GB).** Then FSDP's all-gather overhead is not worth it — plain [DDP](/blog/machine-learning/distributed-training/ddp-from-first-principles) with the model replicated on every GPU is faster, because it never has to gather parameters. Do not shard a model that fits.

The meta-rule, which is the spine of the whole series: **every knob is a cost, so only pay for the ones your situation forces.** Sharding costs comms; pay it only when the model won't fit. Tensor parallelism costs an all-reduce per layer; pay it only when sharding isn't enough. Multi-node costs the InfiniBand tax; pay it only when you have saturated one node. A recipe that reaches for the heavy machinery before it needs to is not "robust" — it is slow and expensive for no benefit.

## Key takeaways

- **A recipe is a stack, not a pile.** Seven layers — launch, parallelism, batch, optimizer, schedule, precision, checkpoint — and each one fixes the budget for the ones below it. Read it top-down as a chain of consequences.
- **The global batch is a product you compute, not a knob you set:** micro-batch × grad-accum × data-parallel degree, times sequence length for tokens. The three factors are interchangeable for convergence but cost memory, wall-clock, and GPUs respectively.
- **The optimizer state, not the weights, is the memory wall.** AdamW's fp32 state is 12Ψ — 80 GB for a 6.7B model, more than weights and gradients combined. FSDP exists to shard exactly that bar.
- **The device mesh has two axes because the cluster has two wires.** Keep the heavy, frequent all-gather on fast NVLink inside a node; let only the once-per-step gradient all-reduce cross the slow InfiniBand link between nodes. That split is why `HYBRID_SHARD` scales when `FULL_SHARD` stalls on a weak interconnect.
- **The data mix is a first-class decision hiding in one line.** `train_data` points at a fixed ratio of web, code, books, and math; no optimizer setting recovers a bad mix. Pre-tokenize it, and never tokenize on the fly with 64 GPUs waiting.
- **bf16 over fp16, always, for pretraining.** Same exponent as fp32 means no overflow and no loss scaler; the recipe needs neither. Keep the optimizer state in fp32.
- **Grad clip 1.0 and warmup 2000 are not optional.** Clipping caps the damage of any single bad batch; warmup protects the first steps from cold-Adam blowup. Turn either off and the run NaNs or explodes eventually — not maybe, eventually.
- **The schedule length is derived, not chosen:** `total_steps = token_budget / tokens_per_step`. Change the batch and forget to recompute it, and cosine decays over the wrong horizon and the model finishes under-trained.
- **Every knob has one failure mode.** OOM and NaN each have several roads in; the config-to-failure matrix is your first debugging move, turning "it crashed" into "it crashed *because of this line*."
- **MFU is the dollar meter.** A recipe that trains correctly at half the MFU costs twice as much for the identical model. At 64 GPUs on InfiniBand, ~37% MFU and ~92% scaling efficiency is the number to beat.
- **Do the fit calculation before you launch.** Thirty seconds of arithmetic — sharded state plus activations plus overhead against 80 GB — saves an hour of watching a job OOM at step zero.

## Further reading

- [Why distributed training](/blog/machine-learning/distributed-training/why-distributed-training) — the four walls that every knob in this recipe is fighting.
- [The distributed training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) — the capstone decision-and-debug checklist this recipe sits underneath.
- [The memory budget](/blog/machine-learning/distributed-training/the-memory-budget) — the full (2+2+12)Ψ derivation behind the memory figure.
- [FSDP in practice](/blog/machine-learning/distributed-training/fsdp-in-practice) — the wrap policy, sharding strategy, and mixed-precision knobs in depth.
- [Mixed precision at scale](/blog/machine-learning/distributed-training/mixed-precision-at-scale) — why bf16 and not fp16, and where fp32 survives.
- [Activation checkpointing](/blog/machine-learning/distributed-training/activation-checkpointing) — the compute-for-memory trade that keeps activations down.
- [Picking a parallelism strategy](/blog/machine-learning/distributed-training/picking-a-parallelism-strategy) and [3D parallelism](/blog/machine-learning/distributed-training/3d-parallelism) — when to go beyond FSDP.
- [Launching on a SLURM cluster](/blog/machine-learning/distributed-training/launching-on-a-slurm-cluster) and [the data pipeline at scale](/blog/machine-learning/distributed-training/the-data-pipeline-at-scale) — the launch and the loader that feed this recipe.
- The **Llama 2** and **OLMo** technical reports, and the **DeepSpeed ZeRO** and **PyTorch FSDP** documentation, for the published recipes and the sharding internals.

---
title: "Why Distributed Training: The Four Walls That Force You Off One GPU"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "The four hard limits that push every serious training job off a single GPU, the memory arithmetic that proves it, and the one-picture map of how the rest of this series fixes each wall."
tags:
  [
    "distributed-training",
    "multi-gpu",
    "fsdp",
    "ddp",
    "pytorch",
    "gpu-memory",
    "scaling-efficiency",
    "deep-learning",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Distributed Training"
author: "Hiep Tran"
featured: true
readTime: 41
---

You open a fresh notebook, load a 7-billion-parameter Transformer, wrap it in the Adam optimizer, call `.cuda()`, and start a training step. The GPU is an A100 with 80 GB of HBM2e — the biggest single accelerator most teams can get their hands on. The forward pass runs. The backward pass runs. And then, somewhere inside `optimizer.step()`, you get the message every ML engineer eventually meets:

```console
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 28.00 GiB
(GPU 0; 79.15 GiB total capacity; 74.62 GiB already allocated).
```

You did nothing wrong. The model is not exotic. The batch size is one. You simply asked a single 80 GB card to hold something that needs 112 GB before it stores a single activation, and it politely declined. That is the first wall, and it is not the only one. Even if the model *had* fit, you would soon meet three more: a training run that takes years on one card, a run that is too slow to iterate on, and a bill that makes your CFO wince. Distributed training exists because of these four walls, and this post is about seeing them clearly — with real arithmetic — before you write a single line of `torch.distributed` code.

This is the front door to a 40-post series, **Distributed Training in the Trenches**. The rest of the series is deep and specific: gradient all-reduce byte volumes, the pipeline bubble, FSDP sharding strategies, NCCL timeouts that hang a 64-GPU job at step 4,000. But none of that matters if you do not first understand *why* you are paying the complexity tax at all. So we start here, at the four walls, and we draw them.

![Four limits surrounding a single GPU each labeled with the constraint that forces a training job onto more hardware](/imgs/blogs/why-distributed-training-1.webp)

By the end of this post you will be able to do four concrete things. First, compute — on the back of an envelope — whether a given model will fit on a given GPU, and by how much it overflows. Second, estimate how long a training run will take and what it will cost, in GPU-hours and dollars, before you launch it. Third, read the naive scaling law well enough to know why eight GPUs almost never give you an eight-times speedup, and what number to actually expect. And fourth, look at any training problem and name which of the four walls you have hit, which points you at exactly which lever to pull. That last skill is the whole game, and it is what the map at the end of this post gives you.

One promise up front, because this series is opinionated about it: **distributed training is a cost, not a virtue.** If your model fits on one GPU and trains fast enough and cheaply enough, you should not distribute it. The complexity you take on — new failure modes, new debugging surface, new ways to silently corrupt a run — is real and it is not free. We distribute only when a wall forces us to. Naming the wall is how you justify the tax.

## 1. The four walls, named

A single GPU is a fixed box. It has a fixed amount of memory (80 GB on an A100 80GB SXM, 80 GB on an H100 SXM, less on everything cheaper) and a fixed peak compute rate (about 312 dense bf16 TFLOP/s on that A100, about 989 bf16 TFLOP/s on the H100). Everything you want to do has to fit inside that box and finish in a time you can tolerate at a price you can afford. When one of those constraints breaks, you are off the single GPU whether you like it or not. There are exactly four ways it breaks, and the figure above draws all four pushing outward from the one card in the middle.

**Wall 1 — the model won't fit.** The parameters, gradients, optimizer states, and activations all have to live in HBM at once. For a 7B model trained with mixed-precision Adam, the state alone is 112 GB, which does not fit in 80 GB. This is a hard, binary wall: you either fit or you crash before step one. It is the wall in the OOM message above, and it is the reason the largest models were *never* trainable on one device.

**Wall 2 — the data won't finish in time.** Even a model that fits has to see enough tokens to converge, and each token costs FLOPs. Multiply the tokens by the FLOPs-per-token, divide by your achieved compute rate, and you get wall-clock time. For a serious run this number comes out in years on a single card. A model you cannot finish is as useless as one that will not fit.

**Wall 3 — the run is too slow to iterate on.** This is the softer sibling of Wall 2 and it is about *you*, the engineer, not the model. Research and product work is a loop: change something, train, look at the curve, change again. If one loop takes three weeks, you get seventeen experiments a year. If it takes a day, you get three hundred. Speed is not a luxury here — it is the difference between a project that ships and one that dies of slow feedback.

**Wall 4 — it costs too much.** GPU-hours are money. A single run of a modern 7B model is on the order of tens of thousands of GPU-hours; at a couple of dollars an hour that is a five- or six-figure bill for *one* run, and you will do many. Worse, a badly scaled distributed run wastes those GPU-hours on communication overhead, so the cost wall and the efficiency of your distribution are the same conversation.

Here is the frame the whole series hangs on. Each wall has a symptom you can measure, a number that proves it, and a lever that knocks it down. Keep this table; every later post is a deep-dive on one of its rows.

| Wall | Symptom you see | The number that proves it | The lever |
|---|---|---|---|
| Won't fit | `CUDA out of memory` | State bytes > HBM bytes (112 GB > 80 GB) | Shard state: FSDP / ZeRO-3 (and TP/PP) |
| Won't finish | ETA in months or years | tokens × FLOPs/token ÷ achieved FLOP/s | Data parallelism: more GPUs, more tokens/s |
| Too slow | Iteration loop measured in weeks | wall-clock per experiment | More GPUs + overlap comms with compute |
| Too costly | GPU-hours × price is unaffordable | GPU-hours = wall-clock × N ÷ efficiency | Raise MFU and scaling efficiency |

Notice that the levers are not independent. More GPUs help Walls 2 and 3 directly, but they *cost* communication, which feeds Wall 4. Sharding helps Wall 1, but it adds its own collectives, which can slow you down and feed Wall 3. The entire discipline of distributed training is choosing, for your specific model on your specific cluster, the combination of levers that clears the walls you have hit without creating a new bottleneck somewhere else. The rest of this post takes the walls one at a time and gives each its arithmetic.

## 2. Wall 1: the memory arithmetic that decides if it fits

Before you can distribute a model you have to know why it does not fit, and "it's big" is not an answer you can act on. You need the byte-level budget. The good news is that the budget is simple enough to compute in your head once you know the four line items. Let me name them, and then draw the stack.

Call the number of parameters $\Psi$ (psi). For a 7B model, $\Psi = 7 \times 10^9$. In a standard mixed-precision Adam setup — which is what almost everyone trains with — you keep the following per-parameter state:

- **Parameters, in bf16:** 2 bytes each. That is $2\Psi$ bytes.
- **Gradients, in bf16:** 2 bytes each. Another $2\Psi$ bytes.
- **Optimizer states, in fp32:** Adam keeps three fp32 numbers per parameter — a full-precision master copy of the weight, the first moment (momentum) $m$, and the second moment (variance) $v$. Each is 4 bytes, so that is $12\Psi$ bytes.

Add them and you get the famous coefficient from the ZeRO paper:

$$
M_\text{states} = (2 + 2 + 12)\,\Psi = 16\,\Psi \ \text{bytes}.
$$

Sixteen bytes per parameter, before you have stored one activation. The figure below stacks those bytes up for a 7B model so you can see the optimizer, not the weights, is the elephant.

![Vertical stack of memory line items for a seven billion parameter model showing parameters gradients and optimizer states summing past the eighty gigabyte card limit](/imgs/blogs/why-distributed-training-2.webp)

Plug in $\Psi = 7 \times 10^9$. Parameters: 14 GB. Gradients: 14 GB. Optimizer states: 84 GB. Total: **112 GB**. The A100 has 80 GB. You are 32 GB over the line before activations, and activations are not small. This is the single most important calculation in this entire series, so let us make it a labeled worked example and then write the code that does it.

#### Worked example: does a 7B model fit on an 80 GB A100?

The optimizer states dominate, and this surprises people who think of a model as "its weights." In mixed-precision Adam the weights are the *smallest* of the three big items. The fp32 master copy alone (28 GB) is bigger than the bf16 weights and gradients combined. The two Adam moments add another 56 GB. So of the 112 GB, exactly 84 GB — three quarters — is optimizer overhead that exists only during training. At inference you drop the gradients and optimizer entirely and the same model needs about 14 GB, which is why a 7B model *serves* comfortably on one card but will not *train* on it. The training-time and inference-time memory footprints of the same model differ by 8×.

Now add activations. Activation memory scales with batch size, sequence length, hidden size, and number of layers, and — crucially — with whether you save intermediate tensors for the backward pass. For a 7B model (roughly 32 layers, hidden size 4096) at a 2,048-token sequence and a per-GPU micro-batch of 1, the saved activations without any recomputation land in the tens of GB. So the real budget is not 112 GB; it is 112 GB plus another 20–40 GB. The card is not 32 GB short — it is 50–70 GB short. There is no batch size small enough to save you, because the 112 GB of state is fixed the moment you construct the optimizer.

Here is the arithmetic as a function you can actually run and adapt. It is deliberately dependency-free so you can paste it into any environment.

```python
def training_memory_gb(num_params, bytes_param=2, bytes_grad=2, bytes_optim=12):
    """Peak *state* memory for mixed-precision Adam, in GB (activations extra).

    bytes_optim = 12 -> fp32 master(4) + Adam m(4) + Adam v(4).
    Pass bytes_optim=8 for optimizers that skip the fp32 master copy.
    """
    bytes_per_param = bytes_param + bytes_grad + bytes_optim
    total_bytes = num_params * bytes_per_param
    return total_bytes / (1024 ** 3)

for name, n in [("1.3B", 1.3e9), ("7B", 7e9), ("13B", 13e9), ("70B", 70e9)]:
    gb = training_memory_gb(n)
    fits = "fits" if gb < 80 else "OVERFLOWS 80 GB A100"
    print(f"{name:>5}: {gb:6.1f} GB of state -> {fits}")
```

Run it and you get the whole progression: 1.3B needs about 19 GB (fits comfortably), 7B needs 104–112 GB (overflows), 13B needs about 194 GB (overflows badly), and 70B needs over a *terabyte* of state. The point of the function is not the four numbers; it is that you now have a tool that answers "will this fit?" for any model, any optimizer, any precision, in one line, before you waste an hour discovering it the hard way. This same budget — expanded to include activations, and taught how to account for sharding — is the subject of the dedicated post [the memory budget](/blog/machine-learning/distributed-training/the-memory-budget), and the deep debugging playbook for when this exact wall bites in production lives at [out-of-memory debugging](/blog/machine-learning/debugging-training/out-of-memory-debugging).

### How sharding turns 112 GB into 14 GB

If the state does not fit on one GPU, the obvious move is to stop keeping a full copy of it on every GPU. That is the entire idea behind ZeRO (from DeepSpeed) and its PyTorch-native equivalent, FSDP (Fully Sharded Data Parallel). Instead of replicating all 112 GB on each of your eight GPUs, you *shard* it: GPU 0 owns one eighth of every parameter, gradient, and optimizer slice; GPU 1 owns the next eighth; and so on. When a layer needs its full weights for the forward pass, the GPUs briefly all-gather the shards, use them, and throw the gathered copy away. The figure below contrasts the two worlds.

![Comparison of replicating the full model state on one GPU which overflows versus sharding the same state across eight GPUs which fits](/imgs/blogs/why-distributed-training-3.webp)

The arithmetic is immediate. Shard 112 GB of state across 8 GPUs and each holds 14 GB. Suddenly there is 66 GB of headroom on each card for activations and the temporary gathered weights, and the model that could not exist on one GPU trains comfortably across eight. This is the mechanism that makes 13B, 70B, and 175B models trainable at all, and it is worth internalizing that it does *not* require a bigger GPU — it requires spreading a fixed amount of state over the GPUs you already have. The cost, as always, is communication: those all-gathers and reduce-scatters are bytes on the wire, and managing that cost is what the FSDP and ZeRO posts are about. For now the headline is what matters: **sharding converts a hard fit-or-die wall into a bandwidth problem, and bandwidth problems have knobs.**

There is a subtlety worth flagging so you are not surprised later. Sharding the *optimizer states* only (ZeRO stage 1) already saves the biggest 84 GB item. Sharding gradients too (stage 2) and finally parameters (stage 3, which is what FSDP's full-shard mode does) saves progressively more memory at the cost of progressively more communication. You do not always need the maximum. Picking the right stage for your model-to-GPU ratio is a real decision, covered in [ZeRO and FSDP: the memory model](/blog/machine-learning/distributed-training/zero-and-fsdp-the-memory-model). The reason to mention the gradient here is that Wall 1 is the wall that most often sends people to distributed training in the first place, and sharding is the most surgical lever for it.

## 3. Walls 2 and 3: throughput, wall-clock, and the naive scaling law

Suppose the model fits — either it is small enough, or you have sharded it. You are now facing Walls 2 and 3: will the run *finish*, and will it finish *soon enough to be useful*? Both are throughput questions, and throughput is where the intuition about "just add GPUs" quietly falls apart.

Start with the compute a training run needs. For a dense Transformer, a well-known approximation says a forward-and-backward pass costs about six FLOPs per parameter per token — roughly $2\Psi$ for the forward and $4\Psi$ for the backward. So the total compute to train on $D$ tokens is:

$$
C \approx 6\,\Psi\,D \ \text{FLOPs}.
$$

This is the same ${6ND}$ relationship that underlies compute-optimal scaling; if you want the derivation and its consequences for how many tokens you *should* train on, that is the subject of [Chinchilla compute-optimal scaling](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling). Here we just use it to size the wall.

#### Worked example: nine years on one GPU

Take a LLaMA-style 7B model trained on 1.0 trillion tokens. The compute is $C \approx 6 \times 7\times10^9 \times 1\times10^{12} = 4.2 \times 10^{22}$ FLOPs. Now divide by what one A100 actually delivers. Peak is 312 bf16 TFLOP/s, but you never get peak — real training runs land somewhere around 40% of it once you account for memory-bound operations, non-matmul work, and data loading. Call it 125 TFLOP/s achieved. Then:

$$
t_\text{1 GPU} = \frac{4.2 \times 10^{22}}{1.25 \times 10^{14}} \approx 3.36 \times 10^{8}\ \text{seconds} \approx 3{,}888\ \text{days} \approx 10.6\ \text{years}.
$$

Over a decade, on one card, *if* it could hold the model — which, from Section 2, it cannot. This is Wall 2 in its full absurdity. No amount of patience makes a single GPU finish a real pretraining run. And even the softer Wall 3 bites hard: a fine-tuning run that is one hundredth the size still takes over a month on one card, which is far too slow a loop to do research on.

The published numbers agree with this order of magnitude. The LLaMA paper reports that training LLaMA-7B took about 82,432 A100-80GB GPU-hours on 1.0T tokens. Divide 82,432 hours by the 24 hours in a day and you get roughly 3,435 days — about 9.4 years — on a single GPU. Meta did not wait 9.4 years; they spread the work across 2,048 A100s and finished in days. That division — a fixed pile of GPU-hours spread across many GPUs to collapse the wall-clock — is the essence of data parallelism and the answer to Walls 2 and 3.

### The naive scaling law, and why it is a lie

Here is the trap. If 82,432 GPU-hours on 1 GPU is 9.4 years, then surely 64 GPUs finish in $9.4 / 64 \approx 0.15$ years, about 54 days, and 2,048 GPUs finish in a couple of days. The first division assumes something that is almost never true: that $N$ GPUs give you exactly $N$ times the throughput. They do not. The honest law has a correction factor:

$$
\text{speedup}(N) = N \times \eta(N), \qquad \eta(N) = \frac{\text{throughput on } N \text{ GPUs}}{N \times \text{throughput on 1 GPU}}.
$$

$\eta$ is the **scaling efficiency**: the fraction of the ideal linear speedup you actually capture. At $\eta = 1$ you have perfect scaling; at $\eta = 0.5$ you are throwing away half of every GPU you added. The whole reason distributed training is hard — the reason this series is 40 posts and not one — is that $\eta$ drops as $N$ grows, because the GPUs have to talk to each other, and talking costs time that does not shrink when you add more talkers.

The Amdahl intuition makes the ceiling concrete. Suppose a fraction $s$ of each training step is inherently serial or is communication that cannot be hidden behind compute. Then no matter how many GPUs you throw at the parallel part, the serial part is a fixed tax, and:

$$
\text{speedup}(N) \le \frac{1}{s + \frac{1-s}{N}} \xrightarrow{N \to \infty} \frac{1}{s}.
$$

If just 5% of your step is non-overlapped communication, then $s = 0.05$ and your maximum possible speedup is $1/0.05 = 20\times$ — with a thousand GPUs, with a million GPUs, forever. This is why the *first* lever in distributed training is not "add GPUs"; it is "make $s$ as small as possible" by overlapping communication with computation so it hides behind work the GPU was doing anyway. That single idea — overlap — is so central it gets its own post, [overlapping compute and communication](/blog/machine-learning/distributed-training/overlapping-compute-and-communication). The figure below shows what the erosion looks like in practice as you climb from 1 to 64 GPUs.

![Grid of scaling results across one eight and sixty four GPUs showing tokens per second rising while efficiency falls from one hundred to seventy eight percent](/imgs/blogs/why-distributed-training-5.webp)

Those numbers are representative of a well-tuned run and worth reading carefully. One GPU is the baseline: 3,000 tokens/s, 100% efficiency by definition. Eight GPUs on a single node connected by NVLink reach 22,000 tokens/s — a 7.3× speedup, 92% efficiency. That last mile is lost to the gradient all-reduce that DDP does every step, even though NVLink is fast. Sixty-four GPUs across eight nodes, now talking over InfiniBand between nodes, reach 150,000 tokens/s — a 50× speedup, but only 78% efficiency. You added 8× more GPUs going from 8 to 64 and got only 6.8× more throughput, because inter-node bandwidth is roughly an order of magnitude below NVLink, and the all-reduce that was nearly free inside a node is now a visible cost. Here is the same story as a before-after table, which is the format you should demand from any scaling claim.

| GPUs | Interconnect | Tokens/s | Speedup | Scaling efficiency | Wall-clock (relative) |
|---|---|---|---|---|---|
| 1 | — | 3,000 | 1.0× | 100% | 1.00 |
| 8 | NVLink (1 node) | 22,000 | 7.3× | 92% | 0.14 |
| 64 | InfiniBand (8 nodes) | 150,000 | 50× | 78% | 0.02 |

The 78% is not a failure; for a 64-GPU multi-node run it is a solid result. The lesson is that you must *predict* and *measure* $\eta$, not assume it is 1. A team that budgets its cluster and its schedule assuming linear scaling will be 22% short on a 64-GPU run and much worse at larger scale — and they will not know why until the deadline slips. Diagnosing exactly where that efficiency leaks, and clawing it back, is the work of the debugging tracks in this series, especially [multi-node slower than single-node](/blog/machine-learning/distributed-training/multinode-slower-than-single-node).

One distinction sharpens all of this, and it is worth naming because people quietly conflate the two. **Strong scaling** holds the *total* problem fixed and adds GPUs to finish it faster: same global batch, more GPUs, so each GPU does less work per step and the fixed communication cost becomes a larger fraction of the step. Strong scaling is where efficiency erodes fastest, because you are shrinking the compute the comms hide behind. **Weak scaling** grows the problem *with* the GPUs — you add GPUs and raise the global batch proportionally, so each GPU keeps the same amount of compute per step and the comms stay hidden. Weak scaling holds efficiency far better, which is why large pretraining runs scale by *increasing the global batch* as they add hardware rather than slicing a fixed batch ever thinner. The practical consequence: if you add GPUs and keep the per-GPU batch constant, expect near-weak-scaling efficiency; if you add GPUs to a fixed global batch, expect the strong-scaling cliff. When someone quotes you a scaling number, the first question is always "strong or weak?" — because the same hardware gives very different curves depending on which one you ran.

## 4. Wall 4: the cost, in GPU-hours and dollars

The fourth wall is money, and it is downstream of everything above. The core identity is worth memorizing:

$$
\text{GPU-hours} = \frac{\text{wall-clock hours} \times N}{\text{(baked into }\eta)} = \frac{\text{total compute}}{\text{achieved FLOP/s per GPU}}.
$$

The cleanest way to think about it: GPU-hours is the total compute divided by the per-GPU rate you actually sustain. Scaling efficiency shows up because a lower $\eta$ means each GPU sustains a lower effective rate, so the *same* run consumes more GPU-hours. That is the direct financial link between your interconnect, your overlap, and your bill.

#### Worked example: what a 7B run costs, and what efficiency wastes

Take the LLaMA-7B figure again: about 82,432 A100-hours. Committed-use cloud pricing for an A100 80GB is in the neighborhood of \$2 per GPU-hour (on-demand is often \$3–4, and owned hardware amortizes lower). So one 7B run is roughly:

$$
82{,}432\ \text{GPU-hours} \times \$2/\text{GPU-hour} \approx \$165{,}000.
$$

At on-demand \$4/hour it is closer to \$330,000. For *one* model. Now watch what efficiency does to that. Those 82,432 GPU-hours already bake in whatever $\eta$ the run achieved. Suppose you could have run at 92% efficiency but a bad interconnect placement dropped you to 78%. The GPU-hours scale as $1/\eta$, so your run inflates by a factor of $0.92 / 0.78 \approx 1.18$ — an 18% penalty. On a \$165,000 run that is roughly \$30,000 of pure waste, burned on gradients waiting for a slow network, producing exactly the same model. Nobody signs off on a \$30,000 expense line called "misconfigured NCCL," but that is what a 14-point efficiency drop is. This is why the cost wall and the efficiency conversation are one conversation, and why [cost and efficiency at scale](/blog/machine-learning/distributed-training/cost-and-efficiency-at-scale) treats dollars per million tokens as a first-class metric.

The metric that ties compute, cost, and efficiency together is **MFU — Model FLOPs Utilization** — the fraction of the GPU's peak FLOP/s that your model's useful math actually consumes:

$$
\text{MFU} = \frac{6\,\Psi\,D / t}{N \times \text{peak FLOP/s per GPU}}.
$$

If you train a 7B model on 8 A100s and the run sustains, say, 140 TFLOP/s per GPU of *model* FLOPs, that is $140 / 312 \approx 45\%$ MFU. Every point of MFU is a point of your hardware budget you are not wasting. Published large runs live in a band you should calibrate against: the Megatron-LM paper reports sustaining about 52% of peak (roughly 163 TFLOP/s per A100) while training a 1-trillion-parameter model across 3,072 GPUs, and Google's PaLM reported about 46.2% MFU training a 540B model on 6,144 TPU v4 chips. If your run is at 15% MFU, you are leaving two-thirds of your very expensive cluster on the floor, and the fix is worth more than any new hardware you could buy.

## 5. The one-picture map: from four walls to four levers

You now have all four walls with their arithmetic. The payoff is that each wall maps cleanly onto a *lever* — a specific way of splitting the work — and each lever costs a specific kind of communication. This mapping is the skeleton of the entire series, and it is the single most useful diagram to keep in your head, so here it is as a matrix.

![Matrix mapping each of the four walls to its parallelism lever what it splits the communication it costs and the later post that covers it](/imgs/blogs/why-distributed-training-4.webp)

Read it row by row, because each row is a lever and each lever answers a wall.

**The "won't fit" wall → sharding (and, at the extreme, model parallelism).** When state exceeds HBM, you split the state. FSDP and ZeRO-3 shard the parameters, gradients, and optimizer across the data-parallel group, paying an all-gather (to reconstruct weights for compute) and a reduce-scatter (to reduce and re-shard gradients). When even a single layer's activations or a single matmul is too big for one GPU, you go further and split the *model itself*: **tensor parallelism** splits individual matrix multiplies across GPUs, and **pipeline parallelism** puts different layers on different GPUs. Those are the subjects of [tensor parallelism](/blog/machine-learning/distributed-training/tensor-parallelism-megatron) and [pipeline parallelism and the bubble](/blog/machine-learning/distributed-training/pipeline-parallelism-and-the-bubble), and composing all three — data, tensor, pipeline — into a device mesh is [3D parallelism](/blog/machine-learning/distributed-training/3d-parallelism).

**The "won't finish" wall → data parallelism.** When you have plenty of memory but too little throughput, replicate the model on every GPU, split the *batch* across them, and average the gradients with an all-reduce each step. This is the workhorse, DistributedDataParallel, and it is the first thing you should reach for. [DDP from first principles](/blog/machine-learning/distributed-training/ddp-from-first-principles) derives the all-reduce byte volume and the bucketing trick that hides it.

**The "too slow" wall → more GPUs plus overlap.** Raw GPU count buys throughput, but only if communication overlaps with computation. The all-reduce of layer $L$'s gradients can happen while the backward pass computes layer $L-1$'s, so the network cost hides behind math the GPU was doing anyway. Overlap is what turns a 3× speedup into a 7.3× speedup on the same eight GPUs.

**The "too costly" wall → efficiency, not hardware.** The cheapest GPU is the one you already have running at high MFU. Before you rent more, raise the utilization of what you have: bigger effective batch, better overlap, activation checkpointing to fit a larger micro-batch, mixed precision to move fewer bytes. Every lever in the series is scored, ultimately, by what it does to tokens/s per dollar.

Underneath all four levers sits the physical layer that decides what any of them cost: the **interconnect**. This is the one number that most surprises engineers coming from single-GPU work, so it deserves its own table. The bandwidth between two GPUs varies by more than an order of magnitude depending on how they are connected, and that ratio is exactly why "put the eight GPUs that talk most on the same node" is one of the highest-leverage decisions in the whole discipline.

| Link | Approx. bandwidth | Scope | Why it matters |
|---|---|---|---|
| NVLink 4 (H100) | ~900 GB/s aggregate per GPU | Within a node | All-reduce is nearly free; keep chatty GPUs here |
| NVLink 3 (A100) | ~600 GB/s aggregate per GPU | Within a node | Still fast enough to hide most DDP comms |
| PCIe Gen4 x16 | ~32 GB/s (bidirectional 64) | Within a node, no NVLink | An order of magnitude slower; comms stop hiding |
| InfiniBand HDR | ~25 GB/s per link (200 Gb/s) | Between nodes | The multi-node tax; where $\eta$ starts to fall |

The 900 GB/s of NVLink versus the 25 GB/s of a single InfiniBand link is a 36× gap, and it is the physical reason the 64-GPU run in Section 3 fell to 78% efficiency the moment it crossed node boundaries. The collectives that ride on top of this — all-reduce, all-gather, reduce-scatter, all-to-all — and their exact byte volumes are derived in [collectives from scratch](/blog/machine-learning/distributed-training/collectives-from-scratch), and the physics of the links themselves in [the interconnect physics](/blog/machine-learning/distributed-training/the-interconnect-physics). For the map, the takeaway is simple: **the lever you pull determines which collective you pay for, and the interconnect determines how much that collective costs.** Match the two badly and you get a 64-GPU job that runs slower than 8.

### The cost of talking, quantified

"It costs communication" is not something you can budget against, so let me turn it into a number, because the mechanism is clean and it is the number that decides whether adding GPUs helps or hurts. Take the all-reduce that DDP runs every step — it sums each rank's gradients and hands every rank the total. The efficient implementation is a **ring**: the $N$ GPUs form a logical ring, the gradient buffer of size $S$ bytes is chopped into $N$ chunks, and the algorithm runs $N-1$ reduce-scatter steps followed by $N-1$ all-gather steps, each step passing one chunk of $S/N$ bytes to a neighbor. Count the bytes each GPU sends over the whole operation:

$$
V_\text{ring} = 2 \cdot \frac{N-1}{N} \cdot S \ \text{bytes per GPU}.
$$

The factor of two is the two phases; the $(N-1)/N$ term approaches 1 as $N$ grows, so a ring all-reduce moves very close to ${2S}$ bytes per GPU *no matter how many GPUs participate*. That constant-with-$N$ property is exactly why ring all-reduce is the backbone of data-parallel training and why DDP scales at all — the per-GPU communication does not balloon as you add ranks. Divide those bytes by the interconnect bandwidth $B$ and you have the all-reduce time, which you then race against the backward-pass time to decide whether it hides.

#### Worked example: all-reducing a 7B gradient over NVLink versus InfiniBand

A 7B gradient buffer in bf16 is 14 GB, so $S = 14 \times 10^9$ bytes. On a single 8-GPU node over NVLink 3 at an effective 250 GB/s for the collective, the ring all-reduce moves $2 \times (7/8) \times 14 \approx 24.5$ GB per GPU, taking about $24.5 / 250 \approx 0.098$ seconds. If the backward pass takes roughly 0.3 seconds, that all-reduce hides comfortably behind it once DDP overlaps the two — the network is not your bottleneck, and this is why the 8-GPU run in Section 3 held 92% efficiency. Now run the *same* eight ranks split across two nodes, so the gradients cross a single InfiniBand HDR link at about 25 GB/s: the same 24.5 GB now takes roughly $24.5 / 25 \approx 0.98$ seconds — more than three times the 0.3-second backward pass it was supposed to hide behind. There is nothing left to overlap it with, so it stalls the GPUs and shows up directly as lost efficiency. Same code, same math, one changed cable, and communication went from free to dominant. That single ratio — comms time over compute time — is the quantity that governs every scaling decision in this series, and it is why the interconnect table above is not trivia but the first thing you check when scaling disappoints. The full derivation for every collective, and the tree-versus-ring trade-off at large $N$, is [collectives from scratch](/blog/machine-learning/distributed-training/collectives-from-scratch).

## 6. The smallest diff: from one GPU to many

Enough theory. What does it actually take to go from a single-GPU training loop to a distributed one? Less than you might fear, for the first and most important lever. Let me show the smallest honest diff, because seeing how little changes is the fastest way to stop being intimidated by `torch.distributed`.

Here is a stripped-down single-GPU training loop. Nothing distributed about it; this is what you already write.

```python
import torch
from torch.utils.data import DataLoader

def train_single_gpu(model, dataset, epochs=1, lr=3e-4):
    device = "cuda"
    model = model.to(device)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for batch in loader:
            inputs = batch["input_ids"].to(device)
            targets = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(inputs, labels=targets).loss
            loss.backward()
            optimizer.step()
    return model
```

Now the distributed version. The changes are small and they are all about three concepts you need to meet now, because they are the vocabulary of everything that follows. **World size** is the total number of processes (usually one per GPU) in the job. **Rank** is a process's global index, from 0 to world size minus 1. **Local rank** is its index *within its node*, which tells it which physical GPU to bind to. Every process runs the *same* script; they differ only by these integers, which the launcher injects as environment variables. Here is the diff.

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def train_ddp(build_model, dataset, epochs=1, lr=3e-4):
    # 1. Join the process group. NCCL is the GPU backend.
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)          # bind this process to its GPU
    device = torch.device("cuda", local_rank)

    # 2. Build the model on this GPU and wrap it. DDP handles the all-reduce.
    model = build_model().to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # 3. Shard the DATA so each rank sees a different slice (no overlap).
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, batch_size=8, sampler=sampler)

    for epoch in range(epochs):
        sampler.set_epoch(epoch)               # reshuffle consistently across ranks
        for batch in loader:
            inputs = batch["input_ids"].to(device)
            targets = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(inputs, labels=targets).loss
            loss.backward()                    # DDP all-reduces gradients here
            optimizer.step()

    dist.destroy_process_group()
```

That is the entire diff for data parallelism. Four additions: join the process group, bind to the right GPU, wrap the model in `DDP`, and give the `DataLoader` a `DistributedSampler` so each rank trains on a distinct slice of the data. Everything else — the forward, the loss, the backward, the optimizer step — is byte-for-byte identical to the single-GPU loop. The magic is hidden inside `loss.backward()`: DDP registers hooks on the gradients and, as each bucket of gradients finishes computing, kicks off an all-reduce to average it across all ranks, overlapping that communication with the rest of the backward pass. You get the average gradient, which is mathematically the gradient of the larger combined batch, and every rank applies the identical update so the replicas stay in sync. The *why* and the *gotchas* of that hook machinery are the whole of [DDP from first principles](/blog/machine-learning/distributed-training/ddp-from-first-principles); here the point is only that the diff is small.

You do not run this with `python train.py`. You run it with `torchrun`, which spawns one process per GPU and sets `RANK`, `LOCAL_RANK`, and `WORLD_SIZE` for each. On a single 8-GPU node:

```bash
torchrun --standalone --nproc_per_node=8 train.py
```

And across two 8-GPU nodes, using a rendezvous endpoint so the processes can find each other:

```bash
# On every node (node_rank differs per node), rdzv coordinates the 16 processes:
torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=node0.cluster:29500 \
  train.py
```

That is genuinely the smallest path from one GPU to sixteen. The end-to-end walkthrough — what each flag means, how rendezvous works, how to sanity-check that all ranks joined — is [your first multi-GPU run](/blog/machine-learning/distributed-training/your-first-multi-gpu-run), and when this launch inevitably misbehaves the first time, [debugging DDP and multi-GPU](/blog/machine-learning/debugging-training/debugging-ddp-and-multi-gpu) is the field guide.

One thing the diff does *not* solve: memory. DDP replicates the full model on every GPU, so if the model did not fit on one GPU, DDP does not help — you have eight copies of a thing that does not fit. That is the moment you reach past DDP to FSDP, whose diff is nearly as small but whose effect is to shard rather than replicate. As a preview, the FSDP wrap is a one-liner swap:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

# Replace  model = DDP(model, device_ids=[local_rank])  with:
model = FSDP(
    build_model().to(device),
    sharding_strategy=ShardingStrategy.FULL_SHARD,   # shard params, grads, optim
    device_id=local_rank,
)
```

Same loop, same launcher, one different wrapper — and now the 112 GB of state is spread across your GPUs instead of replicated. The details of the wrapping policy, the sharding strategies, and mixed precision are [FSDP in practice](/blog/machine-learning/distributed-training/fsdp-in-practice). The reason to show it here is to make the through-line concrete: the levers in this series are, at the code level, small wrappers around a loop you already know. The hard part was never the API. The hard part is knowing *which* wrapper, *when*, and *why* — which is exactly the judgment the four-walls map gives you.

## 7. Which lever do you actually pull?

Given a real model and a real cluster, how do you decide? The decision is not "use all the parallelism you can find." It is a short sequence of questions, and the answers route you to exactly one starting point. The tree below is the decision, and it starts — crucially — with permission to *not* distribute.

![Decision tree starting from whether the model fits on one GPU and whether it trains fast enough branching toward shipping DDP sharding or model parallelism](/imgs/blogs/why-distributed-training-6.webp)

Walk it top-down. **Does the model fit on one GPU, with its optimizer and activations?** Run the memory function from Section 2. If yes, ask the second question: **is it fast enough for your iteration loop and your deadline?** If that is also yes, you are done — ship it, do not distribute, and do not read the other 39 posts until you need them. This is the branch people skip, and skipping it is how teams end up debugging NCCL for a model that would have trained fine on one card in an afternoon.

If it fits but is too slow, you have hit Walls 2 or 3 only, and the answer is **DDP**: replicate, split the batch, all-reduce the gradients, add GPUs until you either finish in time or the scaling efficiency stops paying. Do not add anything more exotic than DDP while it is still buying you near-linear speedup; complexity you do not need is complexity that will page you at 3am.

If it does *not* fit, you have hit Wall 1, and the first answer is **shard it with FSDP or ZeRO-3**, which as we saw turns 112 GB into 14 GB per GPU on eight cards. Sharding plus data parallelism handles the enormous majority of models people actually train, up to and including 70B on a single well-connected node or two. Only when even sharding cannot make a single layer's matmul or a single micro-batch fit — or when the sharding communication itself becomes the bottleneck — do you reach for **tensor and pipeline parallelism**, and you reach for them last, because they are the most invasive to your model code and the most sensitive to interconnect topology. The full version of this decision, with thresholds by model size and cluster size, is [picking a parallelism strategy](/blog/machine-learning/distributed-training/picking-a-parallelism-strategy).

Here is the same judgment as a table, because you will want it as a reference when you are staring at a new model.

| Situation | Wall hit | First lever | Reach for next only if... |
|---|---|---|---|
| Fits + fast enough | none | **Do not distribute** | you actually measure a wall |
| Fits + too slow | 2 / 3 | **DDP** (data parallel) | DDP scaling efficiency drops off |
| Does not fit | 1 | **FSDP / ZeRO-3** (shard) | a single layer or matmul still won't fit |
| A layer/matmul won't fit | 1 (extreme) | **Tensor parallelism** | you also run out of layers per GPU |
| Too many layers for memory | 1 (extreme) | **Pipeline parallelism** | you need to compose all three (3D) |

The discipline in this table is the demotion of model parallelism to "last resort." A lot of writing about distributed training leads with tensor and pipeline parallelism because they are intellectually the most interesting. In production they are the *least* common, because most models either fit with sharding or are small enough for plain DDP. Leading with the exotic lever is how you overcomplicate a run that a one-line FSDP wrap would have solved.

## 8. Measuring honestly, so you do not fool yourself

Every number in this post — tokens/s, efficiency, MFU — is only as good as your measurement, and distributed measurement is full of ways to lie to yourself. Before we look at real-world case studies, you need a measurement discipline, because "it feels faster" is not a scaling result and neither is a single timed step. Here is the honest harness.

```python
import time
import torch

def measure_tokens_per_sec(step_fn, tokens_per_step, warmup=10, iters=50):
    """step_fn() runs one full training step (fwd+bwd+optimizer)."""
    # 1. Warm up: let cuDNN autotune, allocator settle, clocks spin up.
    for _ in range(warmup):
        step_fn()
    torch.cuda.synchronize()   # 2. GPU work is async; wait for it before timing.

    start = time.perf_counter()
    for _ in range(iters):
        step_fn()
    torch.cuda.synchronize()   # 3. Wait again before stopping the clock.
    elapsed = time.perf_counter() - start

    return (iters * tokens_per_step) / elapsed
```

Three lines carry the entire discipline, and each guards against a specific lie. The **warm-up** loop discards the first several steps, because the first step pays for cuDNN autotuning, allocator warm-up, and NCCL's own connection setup — timing it makes distribution look far worse than steady state. The two `torch.cuda.synchronize()` calls exist because CUDA is asynchronous: `step_fn()` returns as soon as the work is *queued*, not *done*, so without the sync you would time how fast Python enqueues kernels, which is meaningless. And you time **many iterations** and divide, because a single step is noisy — one data-loader stall or one clock throttle blows it up.

Beyond the harness, four confounds bite specifically in distributed runs, and every one of them has caught good engineers:

- **The data-loader confound.** If your `DataLoader` cannot keep the GPUs fed, you measure the loader, not the model, and adding GPUs makes it *worse* because now more GPUs starve. Use enough `num_workers`, `pin_memory=True`, and `prefetch_factor`, and check GPU utilization is near 100% before trusting any throughput number. This is [the data pipeline at scale](/blog/machine-learning/distributed-training/the-data-pipeline-at-scale).
- **Thermal and clock throttling.** A GPU that boosts to a high clock for the first minute and settles lower under sustained load will show a throughput that decays. Measure steady state, not the first burst.
- **Rank-0 bias.** It is tempting to print timing only from rank 0. But rank 0 may be the fastest rank; the *job* runs at the speed of the *slowest* rank — the straggler. Measure the max step time across ranks, not rank 0's. [The straggler](/blog/machine-learning/distributed-training/the-straggler) is an entire post because this confound routinely halves throughput invisibly.
- **The efficiency baseline.** Scaling efficiency is only meaningful against a *correct* single-GPU baseline at the same per-GPU batch size. Compare 8 GPUs at batch-8-per-GPU to 1 GPU at batch-8, not to 1 GPU at batch-64, or your efficiency number is fiction.

When you want to know *where* the time goes rather than just how much, you graduate to `torch.profiler` and Nsight Systems traces aligned across ranks — the subject of [profiling a distributed run](/blog/machine-learning/distributed-training/profiling-a-distributed-run). But the harness above, run correctly, is enough to catch the 90% case: a run whose "speedup" was really a warm-up artifact, or whose GPUs were starving the whole time.

## 9. Case studies: real numbers from real runs

Abstract laws are convincing only when the published record agrees with them. Here are four real results, each cited, each landing on one of the four walls.

**GPT-3 and the fit wall (Brown et al., 2020).** GPT-3 has 175 billion parameters. In fp16 the weights *alone* are $175\times10^9 \times 2 = 350$ GB — more than four A100 80GB cards just to hold the parameters, before gradients, before optimizer state, before a single activation. Including Adam's $16\Psi$ state, the training footprint is roughly 2.8 TB. There is no single GPU, and no single node, on which GPT-3 can be trained without splitting the model itself. It is the clearest possible illustration of Wall 1: some models were never a question of "which GPU," but of "how many, arranged how."

**Megatron-LM and sustained MFU at scale (Narayanan et al., 2021).** Training with combined tensor, pipeline, and data parallelism, Megatron-LM sustained about 52% of peak device throughput — roughly 163 TFLOP/s per A100 — while training a 1-trillion-parameter model across 3,072 GPUs, an aggregate of about 502 petaFLOP/s. The headline is not the trillion parameters; it is the 52%. Holding half of peak across three thousand GPUs is the result of relentlessly attacking the communication fraction $s$, and it is the number every large run is implicitly compared against.

**LLaMA and the GPU-hour ledger (Touvron et al., 2023).** LLaMA-7B: 1.0T tokens, about 82,432 A100-80GB GPU-hours, on a cluster of 2,048 A100s. That single line contains Walls 2, 3, and 4 at once — the 9.4-single-GPU-years we computed, collapsed to days by 2,048-way parallelism, at a cost you can now estimate in your head at roughly \$165,000 on committed pricing. The paper's throughput of around 380 tokens/s/GPU for the larger 65B model is exactly the kind of number you should be able to sanity-check against your own MFU math.

**ZeRO/DeepSpeed and FSDP breaking the fit wall (Rajbhandari et al., 2020).** ZeRO's central contribution is the observation this post's Section 2 rests on: the $(2+2+12)\Psi$ split means the optimizer state is the dominant term, and sharding it across the data-parallel group cuts per-GPU memory by up to the number of GPUs — without any model-parallel changes to the code. ZeRO and its PyTorch descendant FSDP are why a 13B model that needs 194 GB of state trains on a single 8-GPU node where no card holds more than 80 GB. The internals of the stages and offload are covered in the existing deep-dive on [DeepSpeed ZeRO and 3D parallelism](/blog/machine-learning/open-source-library/deepspeed-zero-3d-parallelism-deep-dive).

Four papers, four walls, and every one of the numbers is reconstructable from the arithmetic in this post. That is the point of building the intuition first: when you read a systems paper, you should be able to predict its headline numbers to an order of magnitude before you reach them, and be suspicious when you cannot. The reverse is also true and more useful day to day — when your own run reports a number that the arithmetic says is impossible, the arithmetic is right and something in your setup is lying to you. A "3× speedup on 8 GPUs" is not a scaling result, it is a bug report: either the loader is starving the GPUs, or the batch is too small to hide the all-reduce, or one rank is a straggler dragging the other seven. The four-walls math is not just a planning tool; it is the smell test you apply to every measurement, and learning to trust it over the dashboard is most of what separates an engineer who scales runs from one who merely launches them.

## 10. When to reach for this, and when not to

The honest counter to a 40-post series on distributed training is to say plainly when you should ignore all of it. Distribution is a tax, and the tax is not only the GPU-hours — it is the debugging surface, the new failure modes (hangs, stragglers, cross-rank NaNs, checkpoint corruption), the harder reproducibility, and the human time to operate a cluster. Do not pay it unless a wall makes you.

**Do not distribute when the model fits and trains fast enough.** A 1.3B model at 19 GB of state trains on a single A100 with room to spare. If your loop is fast enough to iterate on, a single-GPU run is *strictly simpler*: no NCCL, no rendezvous, no sampler, no cross-rank determinism to worry about. Reaching for DDP here buys you nothing and costs you a debugging surface.

**Do not add sharding when plain DDP fits and saturates the interconnect.** FSDP's all-gather and reduce-scatter are more communication than DDP's single all-reduce. If your model fits comfortably under DDP and NVLink is keeping up, sharding it will *slow you down* while solving a memory problem you do not have. Shard when you are memory-bound, not before.

**Do not go multi-node until you have saturated one node.** Crossing the node boundary drops your interconnect from 900 GB/s NVLink to 25 GB/s InfiniBand — a 36× cliff. Eight GPUs on one node at 92% efficiency will often out-throughput a poorly-placed 16-GPU two-node run. Fill the node, measure, and only then reach across the network. The failure mode of ignoring this is the classic [multi-node slower than single-node](/blog/machine-learning/distributed-training/multinode-slower-than-single-node) autopsy.

**Do not add tensor or pipeline parallelism speculatively.** They are the most invasive levers and the most topology-sensitive. Tensor parallelism inserts an all-reduce into every forward and backward pass and is only worth it inside a fast NVLink domain. Pipeline parallelism introduces the bubble — idle GPUs waiting for the pipeline to fill — that only amortizes with enough micro-batches. Reach for them when sharding plus data parallelism genuinely cannot fit or feed the model, and not one step sooner.

The unifying rule is the one we opened with: **name the wall, then pull the smallest lever that clears it.** Every technique in this series is introduced with the wall it answers and the cost it charges, so you can always make that trade deliberately rather than by cargo-cult. The full decision-and-debugging checklist that ties all of it together lives in the capstone, [the distributed training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook), and the map that gets you there — from the first OOM to a debugged 64-GPU run — is drawn below.

![Timeline of the distributed training journey from a model that will not fit through sharding data parallel scaling efficiency loss a failure and the final playbook](/imgs/blogs/why-distributed-training-7.webp)

That arc is the shape of the series and the shape of every real scaling project: you start where the model will not fit, you shard it so it does, you add GPUs so it finishes, you watch efficiency erode as you scale, you survive the failure that inevitably comes at step 4,000, and you end with a repeatable playbook scored in tokens/s, MFU, and dollars. Each stop on that timeline is a track in this series, and each is a wall from this post seen up close.

## Key takeaways

- **There are exactly four walls that force you off one GPU:** the model won't fit, the data won't finish, the run is too slow to iterate on, and it costs too much. Every distributed-training decision answers one of them.
- **Training memory is $16\Psi$ bytes for mixed-precision Adam** — $2\Psi$ params + $2\Psi$ grads + $12\Psi$ optimizer — which is 112 GB for a 7B model and overflows an 80 GB card before activations. The optimizer, not the weights, is the elephant.
- **Sharding turns a fit-or-die wall into a bandwidth problem:** FSDP/ZeRO-3 spreads that 112 GB over 8 GPUs to 14 GB each, and the cost is communication, which has knobs.
- **Training compute is about $6\Psi D$ FLOPs.** Divide by achieved FLOP/s to get wall-clock; a 7B run on 1T tokens is roughly 9.4 single-GPU-years, which only many GPUs collapse.
- **Never assume linear scaling.** Real speedup is $N \times \eta$, and $\eta$ falls as communication grows — 92% on 8 NVLink GPUs, 78% on 64 across InfiniBand. If 5% of your step is non-overlapped comms, your ceiling is 20× no matter how many GPUs you add.
- **The interconnect is the hidden variable.** NVLink is ~36× faster than a single InfiniBand link, which is why the same code runs at 92% efficiency inside a node and 78% across nodes.
- **Cost and efficiency are one conversation.** GPU-hours scale as $1/\eta$; a 14-point efficiency drop is a real five-figure line item producing the identical model. Track MFU as your north star.
- **The smallest diff to distribute is small:** `init_process_group`, bind the GPU, wrap in DDP, add a `DistributedSampler`, launch with `torchrun`. The hard part is judgment, not the API.
- **Measure honestly:** warm up, `torch.cuda.synchronize()` before and after timing, steady state, and time the *slowest* rank — never rank 0.
- **Distribute only when a wall forces you.** If it fits and trains fast enough, do not. Name the wall, pull the smallest lever, and demote model parallelism to the last resort.

## Further reading

- **DDP mechanics next:** [DDP from first principles](/blog/machine-learning/distributed-training/ddp-from-first-principles) and [your first multi-GPU run](/blog/machine-learning/distributed-training/your-first-multi-gpu-run) — the all-reduce and the `torchrun` launch, end to end.
- **The memory model:** [ZeRO and FSDP: the memory model](/blog/machine-learning/distributed-training/zero-and-fsdp-the-memory-model), [FSDP in practice](/blog/machine-learning/distributed-training/fsdp-in-practice), and the dedicated [the memory budget](/blog/machine-learning/distributed-training/the-memory-budget).
- **When it breaks:** [debugging DDP and multi-GPU](/blog/machine-learning/debugging-training/debugging-ddp-and-multi-gpu) and [out-of-memory debugging](/blog/machine-learning/debugging-training/out-of-memory-debugging) from the debugging-training pillar.
- **The whole map and the capstone:** [the map of parallelism](/blog/machine-learning/distributed-training/the-map-of-parallelism) and [the distributed training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook).
- **Scaling context:** [Chinchilla compute-optimal scaling](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) for why the token budget $D$ is what it is, and the [DeepSpeed ZeRO and 3D parallelism deep-dive](/blog/machine-learning/open-source-library/deepspeed-zero-3d-parallelism-deep-dive) for the sharding internals.
- **The seminal papers:** ZeRO (Rajbhandari et al., 2020), Megatron-LM (Shoeybi et al., 2019; Narayanan et al., 2021), GPipe (Huang et al., 2019), LLaMA (Touvron et al., 2023), and the PyTorch FSDP and NCCL documentation.

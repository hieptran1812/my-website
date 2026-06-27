---
title: "DeepSpeed ZeRO for RLHF: Fitting Four Models Into Your GPU Cluster"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "A first-principles walk through how ZeRO sharding lets you train a policy, reference, reward, and value model at once — with the memory math, the communication trade-offs, and runnable DeepSpeed configs for RLHF."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "rlhf",
    "deepspeed",
    "distributed-training",
    "llm-alignment",
    "machine-learning",
    "pytorch",
    "memory-optimization",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/deepspeed-zero-for-rlhf-1.png"
---

The first time I tried to run a real RLHF job on a single 8×A100 node, it died before the first gradient step. Not during training — during *model loading*. DeepSpeed printed a tidy `CUDA out of memory` and exited. I had a 7B policy I wanted to align, and I had assumed that if a 7B model trains fine with supervised fine-tuning on this box, then surely reinforcement learning from human feedback would too. That assumption was wrong by roughly a factor of four, and the factor of four is the entire reason this post exists.

RLHF is not one model. It is at minimum *four* models held in memory at the same moment: the **policy** you are training, a frozen **reference policy** that keeps the policy from drifting into gibberish, a **reward model** that scores the policy's outputs, and a **value head** (a critic) that estimates how good a state is so the policy gradient has a baseline. Each one is roughly the size of your base model. When you do the naive thing and replicate each of them on every GPU — which is what plain data-parallel training does — you are asking a single 80GB card to hold four full 7B models plus their optimizer states plus their activations. The arithmetic, which we will do carefully in a moment, lands somewhere around 320–480 GB *per GPU*. The card has 80. That is the cliff I walked off.

The figure below shows the cliff and the bridge across it. On the left is the naive approach: every GPU holds a complete copy of everything and overflows. On the right is what we are going to build: ZeRO-3 sharding, where the parameters, gradients, and optimizer states are sliced into N pieces and spread across all N GPUs, so each card only ever stores its 1/N slice. The same four-model RLHF job that needs 320 GB on one card needs about 50 GB per card when split across eight. It fits.

![Diagram contrasting naive RLHF replication that overflows an 80GB GPU against ZeRO-3 sharding that distributes one-eighth of every model state across eight GPUs so the job fits](/imgs/blogs/deepspeed-zero-for-rlhf-1.png)

By the end of this post you will be able to do three things. You will be able to *compute*, before you launch anything, how much GPU memory an RLHF job will need and how many cards it takes to fit — no more loading-time OOM surprises. You will understand *why* the ZeRO family of techniques reduces memory the way it does, derived from the structure of the Adam optimizer rather than asserted. And you will have *runnable* DeepSpeed configurations for a real RLHF setup, including the HybridEngine switch between generation and training and the ZeRO-3 + LoRA pattern that cuts the communication cost down to something sane. We will tie all of it back to the spine of this series: RLHF is just an RL loop where the environment is a reward model and the agent is a language model, and everything here is the systems plumbing that makes that loop physically possible on the hardware you actually have.

## 1. The memory problem in RLHF, counted byte by byte

Let me make the memory accounting concrete, because "four models is a lot" is not an engineering statement and you cannot provision a cluster from a vibe. We will count for a single 7B-parameter transformer and then multiply.

The first thing to internalize is that a parameter is not a fixed number of bytes. It depends on what *role* the bytes are playing. During training with the Adam optimizer in mixed precision — which is the standard recipe for fine-tuning large language models — every single parameter is stored several times over, in several precisions, for several purposes.

Here is the full breakdown for one parameter under the standard mixed-precision Adam recipe (this is the accounting from the original ZeRO paper, Rajbhandari et al. 2020):

- The model parameter itself, in BF16 (or FP16): **2 bytes**. This is the copy you do forward and backward passes with.
- The gradient for that parameter, in BF16: **2 bytes**. Produced during backprop.
- An FP32 master copy of the parameter: **4 bytes**. Mixed precision keeps a high-precision copy so that tiny updates don't get lost to rounding when added to the BF16 weight.
- The Adam first moment (the running mean of gradients, $m$), in FP32: **4 bytes**.
- The Adam second moment (the running mean of squared gradients, $v$), in FP32: **4 bytes**.

Add those up: $2 + 2 + 4 + 4 + 4 = 16$ bytes per parameter. People sometimes split this as "2 bytes of model + 2 bytes of gradients + 12 bytes of optimizer states", and the 12 bytes (the FP32 master copy plus the two FP32 moments) are exactly what ZeRO's stages attack. Hold onto that number: **the optimizer states are the dominant cost, three-quarters of the total, and they are pure redundancy in data-parallel training because every GPU keeps an identical copy.**

For a 7B model, $7 \times 10^9 \times 16$ bytes is **112 GB**. Just to *train* one 7B model with Adam, ignoring activations entirely, you need 112 GB of parameter/optimizer state. That already exceeds a single 80GB A100. The 14 GB figure people quote ("a 7B model in BF16 is 14 GB") is only the inference footprint — $7 \times 10^9 \times 2$ bytes. Training is 8× heavier than inference per model.

Now add **activation memory**. Activations are the intermediate tensors you must keep around between the forward and backward pass so you can compute gradients. Activation memory scales with batch size, sequence length, hidden size, and number of layers. For a 7B model with a modest batch and a few-thousand-token context, activations land in the tens of gigabytes — call it 20–40 GB depending on whether you use activation checkpointing (which trades compute to recompute activations rather than store them). So one 7B model *in training* lands at roughly **80–120 GB** all-in. One model. On one GPU. Already over the line.

It is worth being a little more precise about activations, because they are the one category ZeRO does *not* shard by default, and so they often become the binding constraint after you have applied ZeRO-3 to everything else. A rough model: for a transformer with $L$ layers, hidden size $h$, sequence length $s$, and (micro) batch size $b$, the activation memory without checkpointing is on the order of $L \cdot s \cdot b \cdot h \cdot c$ bytes, where $c$ is a per-layer constant capturing all the intermediate tensors a transformer block stashes (the attention scores, the QKV projections, the MLP intermediate which is typically $4h$ wide, the layernorm inputs). For a 7B model — $L \approx 32$, $h \approx 4096$ — with $s = 2048$ and $b = 4$, that constant pushes you into the 20–40 GB range quickly. The reason **activation checkpointing** (also called gradient checkpointing) matters so much in RLHF is that it changes the $L$ in that formula. Instead of storing every layer's activations, you store only the inputs at a handful of checkpoint boundaries and *recompute* the intermediate activations during the backward pass. That trades roughly a 33% increase in compute (one extra forward per checkpointed segment) for an order-of-magnitude reduction in activation memory — turning the activation term from $O(L)$ into $O(\sqrt{L})$ if you checkpoint optimally. In an RLHF job where memory is the binding constraint and you have compute headroom (the GPUs are often waiting on the generation phase anyway), activation checkpointing is nearly always on.

#### Worked example: the four-model RLHF footprint

Let's count a full PPO-style RLHF run on a 7B base, naive (every GPU holds full copies), per GPU:

| Model | Role | Precision needs | Memory |
| --- | --- | --- | --- |
| Policy | Trained with Adam | 16 bytes/param + activations | ~110 GB |
| Reference policy | Frozen, inference only | 2 bytes/param (BF16, no grads/opt) | ~14 GB |
| Reward model | Frozen, inference only | 2 bytes/param | ~14 GB |
| Value head + backbone | Trained with Adam (critic) | 16 bytes/param + activations | ~110 GB |
| **Total per GPU** | | | **~250 GB** |

That is the optimistic count where the reference and reward models are frozen and only need their inference footprint. If you naively wrap *all four* in a full training harness (which DeepSpeed-Chat early users did by accident), the reference and reward models also carry optimizer state they will never use, and you balloon to the 320–480 GB figure. Either way, you are far past 80 GB. This is the wall. Everything in this post is about getting under it without buying more, or fewer, GPUs than you need.

The key insight that unlocks the whole solution: in data-parallel training, all of that memory is *replicated*. If you have 8 GPUs, you are storing 8 identical copies of the optimizer states. That is 8× redundancy on the single most expensive component. ZeRO's name — Zero Redundancy Optimizer — is the entire thesis: eliminate the redundancy by giving each GPU only its share.

## 2. ZeRO in one picture: three stages of sharding

ZeRO (Zero Redundancy Optimizer) attacks redundancy in three escalating stages, each of which shards one more category of state across your N GPUs. The mental model to carry through the rest of this post is a stack: you start with full replication and peel off one redundant category at a time.

![Stacked diagram showing the five ZeRO stages from no sharding through optimizer, gradient, and parameter sharding to CPU and NVMe offload, with memory reduction factors at each level](/imgs/blogs/deepspeed-zero-for-rlhf-2.png)

Recall the three buckets from the last section: 2 bytes of parameters, 2 bytes of gradients, and 12 bytes of optimizer states, per parameter. The three ZeRO stages map exactly onto those buckets:

- **ZeRO Stage 1** shards the **optimizer states** (the 12 bytes). Each GPU stores only $\frac{1}{N}$ of the FP32 master weights and the Adam moments. Parameters and gradients are still fully replicated.
- **ZeRO Stage 2** additionally shards the **gradients** (the next 2 bytes). After backprop, each GPU keeps only the gradient slice for the parameters whose optimizer state it owns. Parameters are still fully replicated.
- **ZeRO Stage 3** additionally shards the **parameters** themselves (the last 2 bytes). Now *nothing* is fully replicated; every GPU holds a $\frac{1}{N}$ slice of everything, and full layers are reconstructed on demand via communication.

There is a clean way to express the memory reduction. Let $\Psi$ stand for the model's BF16 size — for a 7B model, $\Psi = 7 \times 10^9 \times 2 = 14$ GB. In those units the full mixed-precision Adam footprint is $8\Psi$, split as $1\Psi$ (params) $+ 1\Psi$ (grads) $+ 6\Psi$ (optimizer states); the optimizer term is $6\Psi$ rather than $1\Psi$ because its 12 bytes per parameter are $6\times$ the 2-byte BF16 weight. For $N$ GPUs:

$$
\text{ZeRO-1 per GPU} = 1\Psi + 1\Psi + \frac{6\Psi}{N}
$$

$$
\text{ZeRO-2 per GPU} = 1\Psi + \frac{1\Psi + 6\Psi}{N} = 1\Psi + \frac{7\Psi}{N}
$$

$$
\text{ZeRO-3 per GPU} = \frac{1\Psi + 1\Psi + 6\Psi}{N} = \frac{8\Psi}{N}
$$

Look at the asymptotics as $N$ grows large. ZeRO-1 floors out at $2\Psi$ (the un-sharded params and grads dominate). ZeRO-2 floors out at $1\Psi$. ZeRO-3 floors out at *zero* — per-GPU model-state memory goes to zero as you add GPUs, because every byte is shared. That is why people quote "ZeRO-3 reduces per-GPU memory by a factor of N": with enough GPUs the model state effectively disappears and you are bounded only by activations.

For the rule-of-thumb headline numbers people cite — "4×, 8×, 16×" — those come from a specific common case ($N = 64$ in the original paper's tables), but the formulas above are what you should actually plug your own $N$ into. The takeaways: ZeRO-1 is nearly free communication-wise and gives a big win because optimizer state is the fat part; ZeRO-3 gives the most memory savings but pays in communication, as we will see.

## 3. ZeRO-1 in detail: shard the optimizer states

Start with Stage 1 because it is the cleanest and because it shows the core trick that the later stages elaborate.

In ordinary data-parallel training, the lifecycle of a step is: every GPU has a full copy of the model; each GPU runs forward and backward on its own slice of the data batch, producing a full set of gradients; the GPUs **all-reduce** the gradients so everyone ends up with the *averaged* gradient; then every GPU independently runs the optimizer update using that averaged gradient and its own (identical) optimizer state. The waste is glaring: all $N$ GPUs run the *identical* optimizer computation on *identical* state to produce *identical* updated weights. That is $N$× redundant compute and, more importantly, $N$× redundant *memory* for the optimizer state.

ZeRO-1's move is to **partition the optimizer state** so that GPU $i$ owns only the slice for parameters $[\frac{i\Psi}{N}, \frac{(i+1)\Psi}{N})$. The gradient communication stays exactly the same — a standard all-reduce, so every GPU still ends up with the full averaged gradient. But now each GPU only runs the optimizer update on *its* slice of parameters, using *its* slice of optimizer state. After updating, each GPU has fresh values for only its slice of the FP32 master weights. To get the updated BF16 parameters back to full replication for the next forward pass, the GPUs do an **all-gather** of the updated parameter slices.

So the communication pattern for ZeRO-1 is: all-reduce on gradients (same as DDP) **plus** an all-gather on the updated parameters at the end of the step. That extra all-gather is the only added communication, and it is the same volume as the all-reduce's reduce half, so the communication cost is essentially unchanged from plain DDP. This is why ZeRO-1 is often described as "free": you get a 4× cut in optimizer-state memory at virtually no communication penalty.

It is worth pausing on what these collective operations actually are, because the rest of the post leans on them and they are the vocabulary of distributed training. An **all-reduce** combines a value from every GPU (summing the gradients, say) and leaves *every* GPU with the combined result — so after an all-reduce on gradients, all GPUs hold the identical averaged gradient. A **reduce-scatter** does the same combining but leaves each GPU with only *its slice* of the result — GPU $i$ ends up with the summed gradient for parameters $i$'s shard owns, and nothing else. An **all-gather** is the inverse of scatter: each GPU contributes its slice and ends up with the *concatenation* of everyone's slices — so after an all-gather of parameter shards, every GPU holds the full parameter tensor. The crucial identity, which the ZeRO stages exploit relentlessly, is that an **all-reduce is exactly a reduce-scatter followed by an all-gather**. Ring implementations of all-reduce literally do these two phases. This is why ZeRO-2 can replace the all-reduce with just its reduce-scatter half (stopping early, keeping only the slice) at *no extra communication* — it is not adding work, it is removing the all-gather half that DDP did not need anyway. Keep this identity in mind; it is the reason Stages 1 and 2 are "free" and Stage 3 is only 1.5×, not 3×.

#### Worked example: ZeRO-1 memory on 8 GPUs

Take the 7B policy, so $\Psi = 14$ GB (the BF16 size). Plain DDP needs the full $8\Psi = 112$ GB per GPU for model state. ZeRO-1 on $N = 8$ gives:

$$
1\Psi + 1\Psi + \frac{6\Psi}{8} = 2\Psi + 0.75\Psi = 2.75\Psi = 38.5 \text{ GB per GPU}
$$

We just went from 112 GB (impossible on an 80GB card) to 38.5 GB (comfortable, with room for activations). The optimizer-state term shrank from $6\Psi = 84$ GB to $0.75\Psi = 10.5$ GB. That one change — sharding the fattest, most redundant component — already makes the policy trainable. The catch is that for RLHF you have a *second* trainable model (the value/critic), so you would need this to hold for both, and you still pay $2\Psi = 28$ GB of un-sharded params+grads per trainable model. That is the limit ZeRO-2 and ZeRO-3 push past.

## 4. ZeRO-2 in detail: shard the gradients too

ZeRO-1 left the gradients fully replicated: after the all-reduce, every GPU holds the complete averaged gradient even though it will only update $\frac{1}{N}$ of the parameters. That is $2\Psi$ bytes of gradient that GPU $i$ mostly will not use. ZeRO-2 removes that waste.

The trick is to replace the all-reduce of gradients with a **reduce-scatter**. An all-reduce is logically a reduce-scatter followed by an all-gather: reduce-scatter sums the gradients but leaves each GPU with only its $\frac{1}{N}$ slice of the summed result, and the all-gather then broadcasts those slices so everyone has the full thing. ZeRO-2 simply *stops after the reduce-scatter*. Each GPU ends up holding only the gradient slice for the parameters whose optimizer state it owns. It then runs the optimizer update on that slice — and that's all it needs, because it owns exactly the matching optimizer state from Stage 1. Finally, an all-gather reconstructs the full updated parameters for the next forward pass, exactly as in Stage 1.

The beautiful thing is that the communication *volume* is unchanged. A reduce-scatter moves the same number of bytes as the reduce half of an all-reduce, and the closing all-gather is the same all-gather Stage 1 already did. So ZeRO-2, like ZeRO-1, costs essentially the same communication as plain DDP, yet it cuts the gradient memory by $N$×.

To plug numbers in without fumbling units, the cleanest trick is to anchor everything to the model's BF16 size. For a 7B model that is $G = 7 \times 10^9 \times 2 = 14$ GB. In those units the full Adam footprint of $16\Psi$ bytes becomes $8G$ (params $= 1G$, grads $= 1G$, optimizer states $= 6G$, because the 12 optimizer bytes are $6\times$ the 2-byte BF16 weight). Now the three stages on $N = 8$ GPUs evaluate cleanly:

- **ZeRO-1 (8 GPUs):** $1G + 1G + \frac{6G}{8} = 14 + 14 + 10.5 = 38.5$ GB.
- **ZeRO-2 (8 GPUs):** $1G + \frac{7G}{8} = 14 + 12.25 = 26.25$ GB.
- **ZeRO-3 (8 GPUs):** $\frac{8G}{8} = 14$ GB.

Those three numbers are the whole ballgame, and they are why I always recompute in BF16-size units — it removes the byte-counting that is so easy to get wrong when provisioning a cluster. ZeRO-2 brings the per-GPU model state for our 7B policy down to 26 GB; ZeRO-3 brings it down to a flat 14 GB — exactly $\frac{1}{8}$ of the 112 GB total, the textbook $\frac{1}{N}$ result.

## 5. ZeRO-3 in detail: shard the parameters

ZeRO-3 is the stage that makes RLHF on commodity clusters tractable, and it is also the one that fundamentally changes how the forward and backward passes execute. In Stages 1 and 2, every GPU still held a *full copy of the model parameters* — that was the un-shardable $2\Psi$ floor. ZeRO-3 removes it. Now each GPU stores only $\frac{1}{N}$ of every parameter tensor, all the time, and the full parameters of a layer only ever exist transiently, gathered just in time to compute that layer and freed immediately afterward.

Here is the forward pass under ZeRO-3, layer by layer:

![Graph diagram of a ZeRO-3 forward pass where eight GPU shards fan into an all-gather that reconstructs a full layer, the layer computes, then the parameters are freed back to one-eighth](/imgs/blogs/deepspeed-zero-for-rlhf-3.png)

For each layer, the GPUs **all-gather** the parameter shards belonging to that layer, so that — momentarily — every GPU has the full weights for that one layer. They compute the forward for that layer. Then they **immediately free** the gathered parameters, dropping back to holding only their $\frac{1}{N}$ shard. They move to the next layer and repeat. At no point is the whole model materialized; only one layer's worth of full parameters lives in memory at a time (plus the persistent shards).

The backward pass is the mirror image: for each layer, all-gather the parameters (needed to compute the gradient with respect to the inputs), compute the gradients, then **reduce-scatter** those gradients so each GPU keeps only the slice it owns, and free the gathered parameters. The optimizer step then runs locally on each GPU's $\frac{1}{N}$ slice, exactly as in Stage 2.

The communication cost is where ZeRO-3 earns its reputation. Each forward pass now requires an all-gather *per layer* (instead of zero parameter communication during forward in Stages 1–2). Each backward pass requires another all-gather per layer *plus* a reduce-scatter per layer. Roughly, ZeRO-3's total communication volume is about **1.5×** that of plain data parallelism (the all-gather in forward, plus all-gather and reduce-scatter in backward, versus a single all-reduce). That 50% overhead is the price of taking per-GPU model memory all the way down to $\frac{1}{N}$.

This is the central trade-off of ZeRO-3, and it dictates *when* it is the right choice. If your model is small and your network is slow, the per-layer all-gathers can dominate wall-clock time and you spend your training budget waiting on the interconnect. If your model is large enough that compute dominates, or your interconnect is fast (NVLink within a node, InfiniBand across nodes), the communication overlaps with compute and the 1.5× becomes nearly invisible. For RLHF on 7B-and-up models with a decent interconnect, ZeRO-3 is almost always the right call, because the memory pressure from four models is the binding constraint, not throughput.

The mechanism that makes the 1.5× nearly invisible in practice is **prefetching with communication-compute overlap**. While GPU $i$ is computing layer $\ell$, DeepSpeed has already issued the all-gather for layer $\ell+1$'s parameters on a separate CUDA stream. If the all-gather for the next layer finishes before the compute for the current layer does, the communication is entirely hidden — the GPU never stalls waiting for parameters. This is why the `stage3_prefetch_bucket_size` config knob (how far ahead to gather) and `overlap_comm` (whether to overlap at all) are the two settings that most determine ZeRO-3 throughput. When people complain "ZeRO-3 is slow", nine times out of ten it is because overlap is off, the prefetch bucket is too small to stay ahead of compute, or the interconnect genuinely cannot keep up (a small model on slow Ethernet). The theoretical 1.5× volume is real, but whether it costs you wall-clock time is entirely about whether it hides behind compute.

There is also a subtlety about *which* parameters get gathered and freed. ZeRO-3 does not blindly gather and free every single tensor — tensors below `stage3_param_persistence_threshold` (small things like layernorm scales and biases) are kept persistently un-sharded, because the communication overhead of sharding a 4096-element vector is not worth the handful of kilobytes saved. The big matrices — the attention projections, the MLP weights — are the ones that get the just-in-time treatment, because they are where the gigabytes live. This is a recurring theme in systems for large models: shard the fat tensors aggressively, leave the skinny ones alone.

```python
# ZeRO-3's just-in-time parameter gathering, conceptually.
# DeepSpeed does this automatically via forward/backward hooks;
# this is the mental model of what the hooks accomplish.

for layer in model.layers:
    full_params = all_gather(layer.sharded_params)   # transient: full layer in VRAM
    activations = layer.forward(activations, full_params)
    free(full_params)                                # back to 1/N immediately
# only one layer's full params ever co-resident with the shards
```

## 6. ZeRO-Infinity: spilling to CPU and NVMe

ZeRO-3 takes per-GPU model state to $\frac{1}{N}$, but $\frac{1}{N}$ of a *very* large model — or the activation memory, which ZeRO does not shard — can still overflow your GPUs. ZeRO-Infinity (Rajbhandari et al. 2021) extends the idea down the memory hierarchy: when GPU memory runs out, spill state to **CPU RAM**, and when CPU RAM runs out, spill to **NVMe SSD**.

The hierarchy is straightforward in principle. GPU HBM is fast (1.5–3 TB/s on an A100/H100) and scarce (40–80 GB). CPU DRAM is slower (tens to low-hundreds of GB/s over the PCIe/NVLink-C2C bus) but plentiful (hundreds of GB to terabytes). NVMe is slower still (single-digit GB/s per drive, more when striped) but effectively unbounded (terabytes). ZeRO-Infinity stages parameters and optimizer states across these tiers, prefetching the next layer's parameters up to the GPU while the current layer computes, so that — if your bandwidth math works out — the data movement hides behind compute.

The bandwidth math is the whole story for ZeRO-Infinity. For offload to be free, you need to move a layer's parameters up from CPU/NVMe to GPU faster than the GPU finishes computing the previous layer. If the layer compute takes 10 ms and pulling its parameters from NVMe takes 40 ms, you are NVMe-bound and your GPUs idle 75% of the time. The original paper engineered this carefully — partitioned NVMe access, overlapping, a dedicated DeepNVMe library — and demonstrated training a *trillion*-parameter model on a single DGX-2 node, something physically impossible to fit in GPU memory alone.

For RLHF specifically, ZeRO-Infinity is the escape hatch, not the default. You reach for it when even ZeRO-3 across your whole cluster cannot fit the four models — for instance, aligning a 70B policy on a node with only modest GPU count, where you offload the optimizer states (the fat $6\Psi$ term) to CPU RAM. **Offloading optimizer states to CPU is the highest-value, lowest-pain offload**, because optimizer state is the largest component and it is only touched once per step (during the optimizer update), so its bandwidth demand is low relative to parameters that are touched every layer. The DeepSpeed config knob is `offload_optimizer: {device: cpu}`, and it frequently turns an impossible job into a slow-but-feasible one.

There is a clean way to reason about whether a given offload will be worth it, and it comes down to the **arithmetic intensity** of the offloaded component — how much compute happens per byte moved across the slow bus. Optimizer states have the best ratio: they are read and written exactly once per training step, during the Adam update, and that update is itself a fair amount of arithmetic (the moment updates, the bias correction, the parameter step). So you move the optimizer state across PCIe once per step and get a whole optimizer update's worth of work out of it. Parameters are the worst ratio: under ZeRO-3 they are gathered for *every* layer of *every* forward and backward, so offloading them to CPU means hauling the full model across PCIe twice per step (once for forward, once for backward), and PCIe at ~25 GB/s versus HBM at ~2 TB/s is roughly 80× slower. That is why the standard escalation order is: first shard with ZeRO-3 (free, fast), then offload the optimizer to CPU (cheap, modest slowdown), then — only if you are still short — offload parameters (expensive, large slowdown), and finally offload to NVMe (the slowest tier, for when even CPU RAM is exhausted). A useful field heuristic: optimizer offload typically costs you 10–30% throughput; parameter offload can cost you 2–5×; NVMe parameter offload can cost you 5–10×. You climb that ladder only as far as feasibility forces you to, and not one rung further.

The other thing ZeRO-Infinity gives you is the ability to train a model whose *parameters alone* exceed the aggregate GPU memory of the cluster — the regime the paper's trillion-parameter demo lives in. For RLHF that is rarely the situation (your policy is usually a known, fixed size you chose deliberately), so the relevant ZeRO-Infinity use for alignment work is almost always optimizer offload to claw back the $6\Psi$ optimizer term when you are one or two cards short of fitting a large policy. It is the difference between "we cannot run this" and "we can run this, just slower", and for a one-off alignment run, slower-but-feasible usually beats waiting on more hardware.

## 7. Applying ZeRO to RLHF's four-model problem

Now we put the pieces together on the actual problem: four models, one cluster. Recall the spine of this series — the RL loop is an agent acting in an environment to maximize reward. In RLHF the agent is the policy LLM, the environment is "generate a completion and get it scored", the reward is the scalar from the reward model, and the policy update is PPO. The four models map onto that loop precisely, and each gets a different memory treatment depending on whether it learns.

![Graph diagram of RLHF's four models showing the ZeRO-3 trained policy branching to a reference policy and a value head, with a reward model and a KL penalty all merging into the PPO loss](/imgs/blogs/deepspeed-zero-for-rlhf-6.png)

Before the per-model accounting, the *why* behind the four models is worth a paragraph of theory, because it explains why you cannot just delete one to save memory. The PPO objective that RLHF optimizes is a clipped surrogate plus a KL penalty:

$$
\mathcal{L}(\theta) = \mathbb{E}_t\!\left[\min\!\big(r_t(\theta)\,\hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\,\hat{A}_t\big)\right] - \beta\, \mathrm{KL}\!\big(\pi_\theta \,\|\, \pi_{\text{ref}}\big)
$$

where $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$ is the probability ratio, $\hat{A}_t$ is the advantage estimate, and $\beta$ weights the KL penalty. Read off which model each term needs. The advantage $\hat{A}_t$ requires a **value function** $V(s_t)$ as its baseline — that is the value head/critic — because the advantage is $\hat{A}_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ and a good baseline is what keeps the policy-gradient variance from exploding. The reward that feeds $\hat{A}_t$ comes from the **reward model**. And the $\mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})$ term needs the **reference policy** $\pi_{\text{ref}}$ to measure against. Why is that KL term load-bearing? Because the reward model is an imperfect proxy: left unconstrained, the policy will find adversarial completions that score high under the reward model but are degenerate (repetition, exploits, gibberish that happens to trip the reward model's biases). The KL penalty pins the policy near the reference distribution, so it can only move into regions the reward model was actually trained to judge well. Remove the reference model and you remove the only thing stopping reward hacking. That is why all four models are non-negotiable, and why the memory problem is structural rather than a matter of sloppy engineering.

Here is the per-model treatment, which is the practical heart of the whole topic:

- **Policy (being trained):** ZeRO-3 *training* mode. It carries parameters, gradients, and full Adam optimizer state, all sharded $\frac{1}{N}$. This is the most expensive model and the one that most needs sharding. Its per-GPU footprint for a 7B model on 8 GPUs is the 14 GB we computed plus its share of activations.
- **Value head / critic:** in most modern RLHF implementations the value function **shares the policy's backbone** and adds only a small linear head that maps the final hidden state to a scalar value. This is the single most important memory optimization in the list — instead of a second full trainable model (another 110 GB naive), you add a value head that costs a rounding error. The head is trained jointly with the policy, so it lives inside the same ZeRO-3 partition.
- **Reference policy (frozen):** ZeRO-3 *inference* mode. It is never updated, so it carries *no* gradients and *no* optimizer states — just parameters, sharded $\frac{1}{N}$. Its only job is to produce log-probabilities for the KL penalty term that keeps the policy from drifting. Per-GPU on 8 GPUs for 7B: $\frac{14}{8} \approx 1.75$ GB of parameters.
- **Reward model (frozen):** ZeRO-3 *inference* mode, same as the reference. It scores completions to produce the reward signal. If your reward model is the same size as the policy, it adds another $\frac{14}{8} \approx 1.75$ GB per GPU; reward models are often *smaller* than the policy, which helps further.

#### Worked example: ZeRO-3 RLHF for 7B on 8×A100 80GB

Let's see if it fits and how much headroom we get. Per GPU:

| Component | Treatment | Per-GPU memory |
| --- | --- | --- |
| Policy params + grads + Adam (sharded) | ZeRO-3 training | 14 GB |
| Value head | Shares policy backbone | ~0.1 GB |
| Reference policy params (sharded) | ZeRO-3 inference | 1.75 GB |
| Reward model params (sharded) | ZeRO-3 inference | 1.75 GB |
| Activations (policy fwd/bwd, with checkpointing) | — | ~15 GB |
| Generation KV-cache (rollout phase) | — | ~8 GB |
| Fragmentation + DeepSpeed buffers | — | ~10 GB |
| **Total per GPU** | | **~50 GB** |

Fifty gigabytes per GPU on an 80 GB card. It fits, with about 30 GB of headroom for larger batch sizes or longer sequences. Compare that to the ~250 GB naive figure from Section 1 — ZeRO-3 turned an impossible job into one that runs comfortably on a single 8-GPU node. The headroom matters in RLHF specifically because the *generation* phase (sampling completions to score) needs a KV-cache, and that cache competes for the same VRAM as the training state. We will see how the HybridEngine manages that competition next.

How many GPUs do you actually need? The binding constraint for a frozen-reference, shared-value-head, 7B RLHF setup is roughly the activation + KV-cache + the policy's irreducible per-layer gathered parameters. On 80 GB cards, 8 GPUs is comfortable for 7B; for a 13B policy you want 8–16; for a 70B policy you are looking at 32–64 GPUs with ZeRO-3, possibly with optimizer offload to CPU to relieve pressure. Always run the memory estimator (Section 11) before committing the cluster.

## 8. DeepSpeed-Chat and the HybridEngine

There is a subtle systems problem hiding inside RLHF that ZeRO-3 alone does not solve, and DeepSpeed-Chat (released by Microsoft in 2023) is the framework that solved it. The problem is that RLHF alternates between two completely different computational modes within every single iteration.

First comes **generation** (the rollout phase): the policy autoregressively samples completions for a batch of prompts. This is *inference* — you want a KV-cache, you want fast token-by-token decoding, you want the parameters laid out for serving, and you do *not* want gradients or optimizer state in the way. Then comes **training** (the PPO update phase): you feed those completions back through the policy, reference, and reward models, compute the PPO loss, and backprop. This is *training* — you want gathered gradients, you want optimizer state, you want activation memory.

The naive way to run this is to use a vanilla ZeRO-3 model for both phases. But ZeRO-3's just-in-time per-layer all-gather is *terrible* for autoregressive generation: you would all-gather every layer's parameters for *every single token* you decode. Generating a 512-token completion means 512× the parameter-gathering communication of a single forward pass. RLHF spends the majority of its wall-clock time in generation, so this kills throughput.

DeepSpeed-Chat's **HybridEngine** is the fix. It transitions the policy between two modes within each iteration:

- For **generation**, it reorganizes the sharded ZeRO-3 parameters into a layout optimized for inference — gathering and pinning parameters so that token-by-token decoding does not pay the per-token all-gather tax, and enabling inference kernels and KV-caching. It is effectively a ZeRO-3-aware inference engine.
- For **training**, it transitions back to the standard ZeRO-3 partitioned training layout — re-sharding parameters, re-enabling gradient and optimizer-state management — so the PPO update runs with full memory efficiency.

The "generate-then-train switch" is the heartbeat of a DeepSpeed-Chat iteration: switch to inference layout, generate rollouts fast, switch to training layout, run the PPO update, repeat. The original DeepSpeed-Chat report claimed order-of-magnitude throughput improvements over earlier RLHF systems precisely because of this engine — for example, training an OPT-13B RLHF model in hours on a modest number of GPUs where naive setups took far longer or did not fit at all. The exact speedups depend on hardware and configuration, but the architectural lesson is durable: **RLHF needs a system that is excellent at both inference and training and can switch between them cheaply.**

There is a memory dimension to the switch that is easy to miss and that directly shapes the per-GPU budget from Section 7. During generation, the dominant *extra* memory cost is the **KV-cache**: the cached keys and values for every token generated so far, for every layer, so the model does not recompute attention over the whole prefix at each new token. KV-cache memory scales as $2 \cdot L \cdot s \cdot h \cdot b$ times the bytes-per-element — for a 7B model decoding a few hundred tokens for a batch of prompts, that is several gigabytes. Critically, this cache lives in the *same VRAM* that the training phase wants for activations and gathered parameters. The HybridEngine has to manage this hand-off: free the KV-cache and inference buffers before transitioning to the training layout, then free the training activations before the next generation phase. If you size your batch so that the *peak* of either phase fits, you are fine; if you size it so that only the average fits, you will OOM at the phase transition. This is why the Section 7 budget reserves a line for the generation KV-cache separately from the training activations — they peak at different moments, but you must provision for the larger peak.

DeepSpeed-Chat also formalizes RLHF as a **three-stage pipeline**, which is worth naming because the four-model problem only appears in the last stage. Stage one is supervised fine-tuning (SFT): ordinary fine-tuning of the base model on demonstration data — one trainable model, the simplest memory case. Stage two is reward-model training: fitting a model to predict human preference rankings — again essentially one trainable model (a classifier head on a base). Stage three is the PPO/RLHF step itself, and *this* is where all four models coexist and where the HybridEngine and ZeRO-3 earn their keep. So when people say "RLHF is expensive", they usually mean stage three specifically; stages one and two are standard fine-tuning jobs whose memory you already know how to estimate. Everything hard about the memory in this post is concentrated in that final PPO stage.

```python
# DeepSpeed-Chat style RLHF step, showing the generate/train phases.
# The HybridEngine transitions the policy layout between the two.

for batch_prompts in prompt_loader:
    # --- GENERATION PHASE (inference layout) ---
    # HybridEngine reorganizes ZeRO-3 params for fast decoding + KV cache
    seq = rlhf_engine.actor.generate(batch_prompts, max_new_tokens=256)

    # Score the completions with the frozen reward model (ZeRO-3 inference)
    reward = rlhf_engine.reward_model(seq).scores
    # Reference log-probs for the KL penalty (ZeRO-3 inference)
    ref_logprobs = rlhf_engine.ref_model(seq).log_probs

    # --- TRAINING PHASE (training layout) ---
    # HybridEngine transitions back to partitioned ZeRO-3 training
    actor_loss, critic_loss = rlhf_engine.train_rlhf(
        seq=seq, reward=reward, ref_logprobs=ref_logprobs
    )
    rlhf_engine.actor.step()   # ZeRO-3 optimizer step on 1/N params
    rlhf_engine.critic.step()
```

## 9. ZeRO-3 with LoRA: the high-leverage RLHF pattern

If there is one pattern that has done the most to make RLHF affordable for teams without a giant cluster, it is ZeRO-3 combined with **LoRA** (Low-Rank Adaptation). The combination attacks ZeRO-3's one weakness — communication — in a way that is almost too good.

Recall ZeRO-3's cost: it all-gathers every layer's *full* parameters on every forward and backward, because in standard fine-tuning all parameters are trainable and change every step, so they must be gathered, updated, and re-sharded. LoRA changes the premise. With LoRA, the giant base model is **frozen** — its weights never change — and you train only small low-rank adapter matrices $A$ and $B$ injected into each layer, where the update to a weight matrix $W$ is approximated as $\Delta W = BA$ with $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and rank $r \ll d, k$. The adapters are tiny — often well under 1% of the base model's parameter count.

Now combine the two: put the **frozen base model under ZeRO-3** (sharded $\frac{1}{N}$ so it fits in memory), and keep the **LoRA adapters replicated on every GPU** (they are small enough that replication costs almost nothing). The consequence is dramatic. The base parameters are frozen, so although they still must be *gathered* for the forward pass, they never have gradients reduce-scattered and never go through an optimizer update — the heaviest part of ZeRO-3's per-step communication, the gradient and parameter-update traffic, applies only to the *adapters*, which are tiny and replicated (so they need no gathering at all). You keep ZeRO-3's memory win on the base model while shedding most of its communication tax.

The practical effect on RLHF: with four models, putting the frozen base under ZeRO-3 for all of policy/reference/reward (they can even *share* the same frozen base weights, with different adapters for the policy and value head) collapses the memory footprint enormously, and because only the small adapters are trained and synchronized, throughput stays high. This is how people run 7B-and-up RLHF on a single 8-GPU node, or even on a handful of consumer-grade cards.

```python
# ZeRO-3 + LoRA for RLHF using TRL + PEFT + DeepSpeed.
# Base model is frozen and sharded under ZeRO-3; only LoRA adapters train.
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],   # inject adapters into attention
    task_type="CAUSAL_LM",
)

# Value head shares the backbone; PEFT freezes the base, adds adapters
policy = AutoModelForCausalLMWithValueHead.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    peft_config=lora_cfg,   # frozen base + trainable LoRA + value head
)

ppo_config = PPOConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    learning_rate=1.41e-5,
    batch_size=64, mini_batch_size=8,
    # DeepSpeed ZeRO-3 is configured via the accelerate/deepspeed launcher
)
trainer = PPOTrainer(config=ppo_config, model=policy, ref_model=None)
# ref_model=None tells TRL to derive the reference from the frozen base —
# no second full copy needed when the base is frozen.
```

Notice the `ref_model=None` trick: when the base is frozen and only adapters are trained, the "reference policy" is just the base model *without* the adapters applied. TRL can compute reference log-probs by temporarily disabling the adapters, so you do not need a fourth full model copy at all. The four-model problem partially collapses into a one-base-model-plus-adapters problem. That is the most elegant memory win in the entire RLHF toolkit.

Let me quantify just how small the trainable footprint becomes, because the numbers are striking. Consider one attention projection in a 7B model: a weight matrix $W \in \mathbb{R}^{4096 \times 4096}$, which is about 16.8M parameters. A full fine-tune trains all 16.8M with Adam, so $16$ bytes each. LoRA replaces the update with $\Delta W = BA$ where $B \in \mathbb{R}^{4096 \times r}$ and $A \in \mathbb{R}^{r \times 4096}$ at rank $r = 16$. The adapter parameter count is $2 \times 4096 \times 16 = 131{,}072$ — that is **128× fewer** trainable parameters for that matrix. Across the whole model, LoRA at rank 16 on the attention projections typically trains well under 1% of the base parameters. The consequence for ZeRO-3 communication is what matters: the gradient reduce-scatter and the optimizer-state all-gather, which in full fine-tuning move $O(\Psi)$ bytes per step, now move $O(\text{adapter size})$ bytes — a hundredth or less. The base model still gets gathered for the forward pass (you cannot skip that; the activations flow through the frozen weights), but the *trainable* traffic, the part that scales with how often you update, nearly vanishes.

#### Worked example: ZeRO-3 + LoRA trainable footprint for 7B RLHF

Take a 7B base, LoRA rank 16 on attention `q_proj` and `v_proj`, roughly 4.2M trainable adapter parameters (well under 0.1% of 7B). The trainable Adam state is $4.2 \times 10^6 \times 16 = 67$ MB — *megabytes*, replicated on every GPU at negligible cost. The frozen 7B base is sharded under ZeRO-3 at $\frac{14}{8} = 1.75$ GB per GPU. So the policy's *total* per-GPU footprint is about 1.75 GB of base shard + 0.07 GB of adapters + activations, versus the 14 GB of the full-fine-tune ZeRO-3 policy. And because the reference is just the base-without-adapters (`ref_model=None`) and the reward model can share the same frozen base with its own adapter, you can in the best case hold *one* sharded 7B base (1.75 GB per GPU) plus a few sets of tiny adapters and serve all of policy, reference, and reward from it. The four-model problem has collapsed to roughly one model's memory plus a rounding error. This is why a 7B RLHF run that "needs an 8-GPU node" with full fine-tuning can sometimes fit on two or even one card with ZeRO-3 + LoRA.

## 10. Practical configuration: ds_config.json for RLHF

Let's get concrete with the configuration you actually write. DeepSpeed is driven by a JSON config; here is a ZeRO-3 config tuned for RLHF, annotated.

```json
{
  "train_batch_size": 64,
  "train_micro_batch_size_per_gpu": 8,
  "gradient_accumulation_steps": 1,
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "allgather_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "gradient_clipping": 1.0,
  "wall_clock_breakdown": false
}
```

The parameters that matter most for RLHF and why:

- **`stage: 3`** — full parameter sharding, as discussed.
- **`bf16.enabled: true`** — BF16 mixed precision. Prefer BF16 over FP16 for RLHF: BF16's wider exponent range avoids the loss-scaling instabilities FP16 suffers, and reward/KL terms in PPO can have an awkward dynamic range. This is the single most common stability fix.
- **`overlap_comm: true`** — overlaps the gradient reduce-scatter / parameter all-gather communication with backward compute. This is what hides ZeRO-3's 1.5× communication behind compute; turning it off is a common cause of "why is my GPU at 40% utilization".
- **`reduce_scatter: true`** — uses reduce-scatter (Stage 2/3 behavior) rather than all-reduce for gradients.
- **`allgather_bucket_size` / `reduce_bucket_size`** — how many bytes to batch into a single communication call. Bigger buckets mean fewer, larger collective ops (better bandwidth utilization) but more transient memory and coarser overlap. `5e8` (500 MB) is a reasonable starting point; if you are tight on memory, shrink them; if you are communication-bound on a fast interconnect, grow them.
- **`stage3_prefetch_bucket_size`** — how much of the *next* layer's parameters to all-gather ahead of time so the gather overlaps with the current layer's compute. Critical for ZeRO-3 throughput.
- **`stage3_max_live_parameters`** — the cap on how many full (gathered) parameters may be co-resident in VRAM at once. This is your knob for the memory/throughput trade-off: higher allows more layers gathered ahead (faster, more memory), lower keeps memory tight (slower).
- **`stage3_param_persistence_threshold`** — parameter tensors smaller than this are kept *persistently* un-sharded (it is not worth the communication to shard tiny tensors like layernorm gains). Keeps small ops fast.
- **`gradient_clipping: 1.0`** — essential for PPO. RLHF gradients can spike when the policy finds a reward-hack; clipping the global gradient norm to 1.0 is standard and prevents a single bad batch from blowing up the run.

To add CPU offload of the optimizer (for when ZeRO-3 alone does not fit, e.g. a 70B policy on too few cards), add this block inside `zero_optimization`:

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": { "device": "cpu", "pin_memory": true },
    "offload_param": { "device": "cpu", "pin_memory": true }
  }
}
```

`pin_memory: true` uses pinned (page-locked) host memory so the CPU↔GPU transfers can use DMA and overlap with compute — without it, offload is much slower. Offload the optimizer *first* (biggest win, lowest bandwidth pressure); only offload parameters if you are still short, because parameter offload pays bandwidth on *every layer* of *every* forward and backward.

### Common mistakes and debugging

A short field guide to the failures I have actually hit:

- **OOM at model load, not training.** You are replicating something you think is sharded. Check that *all four* models are wrapped in DeepSpeed/ZeRO-3, not just the policy. The reference and reward models loaded with a plain `from_pretrained` outside the engine are fully replicated and will sink you.
- **GPU utilization stuck low (30–50%).** Communication is not overlapping. Confirm `overlap_comm: true`, increase `stage3_prefetch_bucket_size`, and check your interconnect — ZeRO-3 across nodes on slow Ethernet will be communication-bound no matter what.
- **Loss is NaN after a few steps.** Almost always FP16 loss-scaling fighting PPO's dynamic range. Switch to `bf16`. If you must use FP16, tune `loss_scale_window` and `initial_scale_power` down.
- **`stage3_gather_16bit_weights_on_model_save` matters at checkpoint time.** With sharded params, saving a normal HuggingFace checkpoint requires gathering the full weights; if this flag is off you will save shards that are annoying to reload for inference. Turn it on for RLHF where you want a deployable policy at the end.
- **Mismatched `train_batch_size`.** DeepSpeed enforces `train_batch_size = micro_batch_per_gpu × grad_accum × num_gpus`. If these do not multiply out, it errors at init. In RLHF the "batch" is rollout episodes, so be deliberate.
- **KL coefficient drifting or collapsing.** If your PPO KL term explodes, the policy is running away from the reference (often a sign the reward model is being hacked); if it collapses to zero, the policy is not learning. Many implementations use an *adaptive* KL controller that nudges $\beta$ up when measured KL exceeds a target and down when it falls below. Watch the measured KL, not just the loss — it is the single most informative RLHF diagnostic, and it tells you whether your reference model is doing its job.
- **Slow checkpointing under ZeRO-3.** Saving requires gathering the sharded weights to one rank (or to CPU), which for a large model is a heavy collective. If checkpoint writes are stalling training, save less frequently, or use DeepSpeed's sharded-checkpoint format during training and only consolidate to a HuggingFace checkpoint at the end (`zero_to_fp32.py` reconstructs the full weights from shards offline).

## 11. Memory profiling: estimate before you launch

The single best habit to build is estimating memory *before* you submit a multi-hour, multi-GPU job. DeepSpeed ships a small estimator that does exactly the arithmetic from Section 2 for you.

```python
# Estimate ZeRO-3 per-GPU memory before launching anything.
from deepspeed.runtime.zero.stage3 import (
    estimate_zero3_model_states_mem_needs_all_live,
)
from transformers import AutoModel

model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-hf")
estimate_zero3_model_states_mem_needs_all_live(
    model,
    num_gpus_per_node=8,
    num_nodes=1,
)
# Prints a table: per-GPU memory for ZeRO-3 with/without
# optimizer offload and parameter offload, so you can pick
# the configuration that fits before you ever allocate the cluster.
```

This prints a table of estimated per-GPU memory for the model *states* (params + grads + optimizer) under several offload configurations. It does **not** count activations or the KV-cache, so add your activation budget (use `wall_clock_breakdown` and the memory monitor during a short dry-run to measure it) and your generation KV-cache on top. There is also `estimate_zero2_model_states_mem_needs_all_live` for Stage 2.

During a run, DeepSpeed's memory monitoring will report a per-step breakdown if you enable it, and PyTorch's own `torch.cuda.memory_summary()` and `torch.cuda.max_memory_allocated()` are invaluable for catching fragmentation and finding the high-water mark. The discipline that has saved me the most cluster-hours: do a single-step dry run on the smallest config, read off the high-water mark, multiply by your safety factor (1.2–1.3× for fragmentation and the generation phase), and only then provision and launch the full job.

#### Worked example: ZeRO-1 vs ZeRO-3 communication overhead

Let me make the communication trade-off concrete with numbers, since "1.5× communication" is abstract until you cost it. Suppose a 7B model, $\Psi = 14$ GB in BF16, on 8 GPUs connected by NVLink at an effective 300 GB/s for collectives.

Plain DDP / ZeRO-1 / ZeRO-2 communication per step is dominated by one all-reduce of the gradients. An all-reduce of $\Psi$ bytes moves about $2\Psi$ bytes per GPU (reduce-scatter + all-gather halves): $2 \times 14 = 28$ GB. At 300 GB/s that is about **93 ms** of communication per step.

ZeRO-3 per step does: an all-gather of params in forward ($\Psi$), an all-gather of params in backward ($\Psi$), and a reduce-scatter of gradients in backward ($\Psi$) — roughly $3\Psi$ of one-directional volume, about $1.5\times$ the all-reduce's $2\Psi$. Concretely $\approx 1.5 \times 28 = 42$ GB equivalent, about **140 ms** per step. So ZeRO-3 adds roughly **50 ms** of communication per step versus ZeRO-1.

Whether that 50 ms matters depends on compute time. If a forward+backward step takes 800 ms of compute, the extra 50 ms (overlapped, so often much less in practice) is noise — take ZeRO-3 and the memory savings. If a step takes 60 ms of compute (tiny model, huge cluster), the communication now dominates and ZeRO-1 or ZeRO-2 wins. **This is the decision rule: ZeRO-3 when memory-bound and compute-heavy; lower stages when compute-light and communication would dominate.** For 7B-and-up RLHF, you are essentially always in the first regime.

## 12. Putting it together: a full ZeRO-3 RLHF launch

Here is an end-to-end launch using `accelerate` with a DeepSpeed ZeRO-3 config, which is the most common way to run TRL-based RLHF today. First the accelerate DeepSpeed config:

```yaml
# accelerate_zero3.yaml — ZeRO-3 for RLHF via accelerate + DeepSpeed
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
deepspeed_config:
  zero_stage: 3
  zero3_init_flag: true          # shard params at model init (avoids load-time OOM)
  zero3_save_16bit_model: true   # gather to a deployable checkpoint on save
  offload_optimizer_device: none # set to 'cpu' only if ZeRO-3 alone doesn't fit
  offload_param_device: none
  gradient_clipping: 1.0
  gradient_accumulation_steps: 1
mixed_precision: bf16
num_machines: 1
num_processes: 8                 # 8 GPUs on this node
```

The `zero3_init_flag: true` is quietly one of the most important lines: it shards parameters *as the model is constructed*, so you never materialize a full copy on a single rank during loading. Without it, rank 0 tries to build the whole 7B model before sharding and OOMs — the exact loading-time failure from this post's opening.

Then the launch and a sketch of the training loop:

```bash
# Launch RLHF across 8 GPUs with ZeRO-3
accelerate launch --config_file accelerate_zero3.yaml \
  train_rlhf_ppo.py \
  --base_model meta-llama/Llama-2-7b-hf \
  --reward_model my-org/llama2-7b-reward \
  --use_lora True \
  --batch_size 64 --mini_batch_size 8 \
  --max_new_tokens 256 --kl_coef 0.05
```

```python
# train_rlhf_ppo.py (sketch) — the RLHF loop under ZeRO-3 via TRL.
import torch
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(args.base_model)
# Policy + value head, frozen base + LoRA, sharded by accelerate/ZeRO-3
policy = AutoModelForCausalLMWithValueHead.from_pretrained(
    args.base_model, peft_config=lora_cfg
)
# Reward model, frozen, also placed under ZeRO-3 inference by accelerate
reward_model = AutoModelForSequenceClassification.from_pretrained(
    args.reward_model, num_labels=1
).eval()

ppo_trainer = PPOTrainer(
    config=PPOConfig(batch_size=args.batch_size,
                     mini_batch_size=args.mini_batch_size,
                     init_kl_coef=args.kl_coef),
    model=policy, ref_model=None, tokenizer=tokenizer,
)

for prompts in ppo_trainer.dataloader:
    query_tensors = [tokenizer.encode(p, return_tensors="pt")[0] for p in prompts]
    # GENERATION: sample completions (HybridEngine inference layout)
    response_tensors = ppo_trainer.generate(
        query_tensors, max_new_tokens=args.max_new_tokens, do_sample=True
    )
    texts = [tokenizer.decode(r) for r in response_tensors]
    # REWARD: score completions with the frozen reward model
    with torch.no_grad():
        rewards = [reward_model(tokenizer(t, return_tensors="pt").input_ids).logits[0]
                   for t in texts]
    # TRAIN: PPO update (HybridEngine training layout, ZeRO-3 optimizer step)
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, {"query": prompts, "response": texts}, rewards)
```

The pipeline of one ZeRO-3 training step — all-gather, forward, free, backward with reduce-scatter, local optimizer step, re-gather — is the engine ticking underneath that innocent-looking `ppo_trainer.step` call:

![Pipeline diagram of one ZeRO-3 training step from all-gathering parameters through forward, freeing params, backward with reduce-scatter gradients, the one-eighth optimizer step, and the updated shard](/imgs/blogs/deepspeed-zero-for-rlhf-4.png)

## 13. A taxonomy of the stages and how to choose

Everything above can be compressed into a comparison table and a decision tree. First the matrix — the stages side by side on the dimensions that decide the choice:

![Matrix comparing the four ZeRO stages across memory reduction, communication overhead, supported model size, and RLHF fit](/imgs/blogs/deepspeed-zero-for-rlhf-5.png)

| Stage | What it shards | Memory cut (large N) | Comm. overhead vs DDP | Best for |
| --- | --- | --- | --- | --- |
| ZeRO-1 | Optimizer states | up to 4× | ~1× (free) | Models that nearly fit; cheap first win |
| ZeRO-2 | + Gradients | up to 8× | ~1× (free) | 13B-class single-model training |
| ZeRO-3 | + Parameters | up to N× | ~1.5× | 70B+, **and four-model RLHF** |
| ZeRO-Infinity | + CPU/NVMe offload | beyond GPU sum | bandwidth-bound | Models too big for the whole cluster |

And the decision tree, which is how I actually pick on a new job:

![Decision tree for choosing a ZeRO stage based on whether the model fits on one GPU, its size, whether it is RLHF, whether LoRA is used, and whether it exceeds the cluster](/imgs/blogs/deepspeed-zero-for-rlhf-8.png)

The logic, in words: if the whole model trains on one GPU, you do not need sharding at all (or ZeRO-1 for a cheap optimizer-memory win). For a 13B–70B single model, ZeRO-2 is the sweet spot — most of the memory win, none of the communication tax. For 70B-plus, or for *any* RLHF (because of the four-model problem), reach for ZeRO-3. If you are using LoRA, ZeRO-3 with a frozen base is the highest-throughput option because only the tiny adapters move. And only when even ZeRO-3 across your whole cluster cannot fit do you escalate to ZeRO-Infinity's CPU/NVMe offload, accepting the bandwidth-bound slowdown as the price of feasibility.

## The history, briefly

It helps to see how these techniques arrived, because the order tells you what each was solving. The progression below traces ZeRO from a memory-sharding paper to the LoRA-aware RLHF systems people run today.

![Timeline of ZeRO's evolution from the 2020 staged-sharding paper through DeepSpeed, ZeRO-Infinity, ZeRO++, DeepSpeed-Chat, and ZeRO-3 with LoRA](/imgs/blogs/deepspeed-zero-for-rlhf-7.png)

The 2020 ZeRO paper introduced the three stages and showed that the optimizer-state redundancy in data parallelism was the thing to kill. DeepSpeed packaged it into a usable library the same year. ZeRO-Infinity (2021) added the memory hierarchy, pushing trainable model size past what GPU memory alone allowed. ZeRO++ (2023) attacked the communication cost directly with quantized weights for the all-gather and other tricks, important for ZeRO-3 on slower interconnects. DeepSpeed-Chat (2023) brought it all to bear on RLHF specifically with the HybridEngine. And the ZeRO-3 + LoRA pattern, which crystallized across the open-source community in 2023, made aligned 7B models reachable for small teams. Each step was a response to the previous one's binding constraint: first memory, then bigger memory, then communication, then RLHF's mode-switching, then RLHF's affordability.

## Case studies

**DeepSpeed-Chat training OPT-13B RLHF.** The DeepSpeed-Chat release (Yao et al., Microsoft, 2023) reported that its HybridEngine could run a full three-stage RLHF pipeline (supervised fine-tuning, reward modeling, and PPO) on models up to and beyond 13B parameters, with the PPO stage — the four-model stage — being the one their system specifically optimized. The headline claim was that the HybridEngine's mode-switching between generation and training gave large throughput gains over running RLHF on a ZeRO-3-for-everything baseline, turning multi-day RLHF runs into something measured in hours on a single node. The durable lesson is architectural, not the specific number: separating the inference-optimized generation phase from the training-optimized update phase is the key to RLHF throughput.

**InstructGPT and the scale of RLHF.** OpenAI's InstructGPT (Ouyang et al. 2022) is the canonical RLHF result — a 1.3B InstructGPT model was preferred by human raters over the 175B GPT-3 base, demonstrating that alignment via RLHF could make a model *100× smaller* feel better to use. The systems point relevant here: even at 175B, the RLHF pipeline still had to hold the policy, reference, reward, and value models simultaneously, which is precisely the four-model memory problem at extreme scale — solvable only with aggressive sharding and offload of exactly the kind this post describes.

**The 7B-on-a-single-node reality.** The most common case in practice today is a team aligning a 7B base (Llama-2-7B, Mistral-7B, Qwen-7B) on a single 8×A100 or 8×H100 node. As the Section 7 worked example shows, ZeRO-3 with a shared value head and frozen reference/reward models lands around 50 GB per GPU — comfortably within an 80 GB card. Add LoRA and the `ref_model=None` trick and the footprint drops further, often enough to run on smaller or fewer cards. This is the configuration most readers will actually launch, and the memory estimator from Section 11 will confirm it fits before you commit the cluster.

**ZeRO-Infinity training a trillion-parameter model on one node.** The ZeRO-Infinity paper (2021) demonstrated fine-tuning a model with over a trillion parameters on a single DGX-2 node by staging parameters and optimizer states across GPU, CPU, and NVMe — a configuration where the parameters alone vastly exceed the node's GPU memory. It is not an RLHF result, but it is the proof-of-concept for the memory hierarchy that the RLHF escape hatch relies on: it established that with careful prefetching and a fast-enough NVMe array, you can train models whose state lives mostly on disk and still keep the GPUs reasonably busy. For alignment work the lesson is narrower — you will almost never offload parameters to NVMe for a 7B–70B policy — but it is reassuring that the ladder has rungs all the way down when a job genuinely will not fit any other way.

## When to use this (and when not to)

ZeRO is a memory tool, and like all tools it has a wrong place to use it. Be honest about the regime you are in.

- **Use ZeRO-3 when you are memory-bound and compute-heavy.** This is essentially all of RLHF on 7B-and-up models. The four-model problem makes you memory-bound, and large transformers make each step compute-heavy enough to hide ZeRO-3's communication. This is the default and you should reach for it without much agonizing.
- **Use ZeRO-2, not ZeRO-3, for a single large model that fits with gradient sharding.** A 13B model fine-tuned alone (not RLHF) often fits comfortably under ZeRO-2, which has essentially no communication overhead. Do not pay ZeRO-3's 1.5× communication tax if ZeRO-2 already fits — that is a common over-engineering mistake.
- **Use ZeRO-1 or nothing when the model already fits.** If your model and optimizer state fit on one GPU, sharding only adds complexity. ZeRO-1 gives a cheap optimizer-memory win at near-zero communication cost; below that, plain DDP is simpler.
- **Do not use ZeRO-3 on a small model with a slow interconnect.** A 1B model sharded across 16 GPUs on slow Ethernet will spend most of its time in per-layer all-gathers. You will get worse throughput than just replicating the small model with DDP. Match the stage to the compute/communication balance.
- **Do not reach for ZeRO-Infinity offload until ZeRO-3 genuinely does not fit.** Offload is the feasibility escape hatch, not a throughput optimization. CPU/NVMe bandwidth is orders of magnitude below HBM, and parameter offload in particular can slow you 2–5×. Offload the optimizer first (cheapest), parameters only as a last resort.
- **Prefer ZeRO-3 + LoRA for RLHF whenever quality allows.** It keeps the memory win and sheds most of the communication tax and collapses the four-model problem toward a one-base-plus-adapters problem. If full fine-tuning is not strictly required, this is the most efficient RLHF configuration available.

## Key takeaways

- **RLHF is a four-model memory problem.** Policy, reference, reward, and value — each roughly base-model-sized — and the naive replicated footprint (hundreds of GB per GPU) is what makes RLHF "not fit" where supervised fine-tuning does.
- **Count memory in $\Psi$-units before you launch.** One model's BF16 size is $1\Psi$; full Adam training is $8\Psi$ ($1$ param $+ 1$ grad $+ 6$ optimizer). The optimizer states are three-quarters of training memory and pure redundancy in DDP.
- **The three ZeRO stages shard one bucket each:** Stage 1 the optimizer states ($4\times$ free), Stage 2 the gradients ($8\times$ free), Stage 3 the parameters ($N\times$, at ~1.5× communication). ZeRO-3 per-GPU model state is exactly $\frac{16\Psi}{N}$ bytes.
- **ZeRO-3's cost is per-layer all-gather communication.** Worth it when compute dominates (large models), painful when communication dominates (small models, slow networks).
- **For RLHF, freeze and inference-shard the reference and reward models, and share the value head with the policy backbone.** This is the difference between four trainable models and one-plus-a-head.
- **The HybridEngine's generate-then-train switch is what makes RLHF fast.** Vanilla ZeRO-3 is terrible at autoregressive generation; you need an engine that switches between inference and training layouts cheaply.
- **ZeRO-3 + LoRA is the highest-leverage RLHF pattern.** Frozen base under ZeRO-3, tiny replicated adapters trained — memory win kept, communication tax shed, four-model problem collapsed via `ref_model=None`.
- **Use `bf16`, `overlap_comm: true`, `gradient_clipping: 1.0`, and `zero3_init_flag: true`.** These four settings prevent the most common RLHF failures: NaN losses, low GPU utilization, gradient explosions, and load-time OOM.
- **Estimate, then dry-run, then provision.** `estimate_zero3_model_states_mem_needs_all_live` plus a single-step high-water-mark measurement with a 1.2–1.3× safety factor will tell you the GPU count before you waste cluster-hours.

This whole topic is plumbing for the RL loop at the heart of this series: the policy is the agent, the reward model is the environment, the KL term keeps the agent in-distribution, and ZeRO is what lets all of it physically coexist on the GPUs you have. For where this fits in the larger map of methods, see the unified taxonomy at `/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map` and the series capstone at `/blog/machine-learning/reinforcement-learning/the-reinforcement-learning-playbook`. For the PPO objective that the four models feed, the clipped surrogate and KL penalty are derived in the policy-gradient track, and the LoRA mechanics here connect to the parameter-efficient fine-tuning posts under `/blog/machine-learning/training-techniques`.

## Further reading

- Rajbhandari, Rasley, Ruwase, He, **"ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"** (2020) — the foundational paper with the stage definitions and the memory formulas used throughout this post.
- Rajbhandari et al., **"ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning"** (2021) — the CPU/NVMe offload hierarchy and the trillion-parameter-on-one-node demonstration.
- Yao et al., Microsoft, **"DeepSpeed-Chat: Easy, Fast and Affordable RLHF Training of ChatGPT-like Models"** (2023) — the HybridEngine and the generate-then-train switch for RLHF.
- Ouyang et al., **"Training language models to follow instructions with human feedback"** (InstructGPT, 2022) — the canonical RLHF result and the four-model pipeline at scale.
- Hu et al., **"LoRA: Low-Rank Adaptation of Large Language Models"** (2021) — the low-rank adapter method that pairs with ZeRO-3 to slash RLHF communication.
- Schulman et al., **"Proximal Policy Optimization Algorithms"** (2017) — the PPO objective whose four-model machinery this post makes fit.
- The DeepSpeed documentation's ZeRO configuration reference and the TRL library's PPO/DeepSpeed integration guides — the authoritative source for the config knobs and their current defaults.
- Within this series: the unified map `reinforcement-learning-a-unified-map` and the capstone `the-reinforcement-learning-playbook` for where RLHF sits among RL methods.

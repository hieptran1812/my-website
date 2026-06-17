---
title: "Why HPC is the bottleneck for modern AI: FLOPs, bytes, and watts"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "The compute wall, the memory-bandwidth wall, and the communication wall — and why every AI engineer must learn to reason in FLOPs, bytes, and MFU, not just layers and loss."
tags:
  [
    "high-performance-computing",
    "gpu",
    "mfu",
    "roofline",
    "memory-bandwidth",
    "distributed-training",
    "deep-learning",
    "ml-systems",
    "transformers",
    "cuda",
  ]
category: "machine-learning"
subcategory: "High Performance Computing"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/why-hpc-is-the-bottleneck-for-modern-ai-1.png"
---

A team I worked with had eight A100s in a single DGX box — roughly a quarter of a million dollars of silicon — and their 7-billion-parameter language model was training at a *crawl*. The loss curve looked fine. The code ran without errors. PyTorch reported the GPUs at "100% utilization" in `nvidia-smi`. And yet the run was projected to take five weeks. When we finally instrumented it properly, the verdict was brutal: the model was achieving **18% MFU** — Model FLOPs Utilization, the fraction of the hardware's peak arithmetic the job actually used. Those A100s were idle 82% of the time, in a very expensive way. The fix had nothing to do with the model architecture, the learning rate, or the dataset. It was a memory-bandwidth problem hiding behind a number nobody on the team had ever measured.

That is the uncomfortable truth this series is built around: **modern AI is bottlenecked by the machine, not by ideas.** A neural network, stripped of romance, is a *schedule of arithmetic over bytes moved across a memory hierarchy and a network*. Training a frontier model is, mechanically, the problem of keeping thousands of \$30,000 chips busy doing useful multiply-accumulates instead of waiting — waiting for data to arrive from memory, waiting for gradients to finish summing across the network, waiting for a kernel to launch. If you cannot reason about *where the time actually goes*, you cannot train or serve a large model efficiently, and you will burn money and weeks doing it slowly.

Most ML engineers were never taught this. We learn to think in layers, parameters, loss, and accuracy — the *math* of the model. But the machine underneath has its own physics: a warp of 32 threads that must march in lockstep, a memory hierarchy where the fast tier is a thousand times smaller than the slow one, an interconnect whose bandwidth sets a hard ceiling on how many GPUs you can usefully gang together. When your job is slow, the answer is almost never in the loss function. It is in one of **three walls** — the compute wall, the memory-bandwidth wall, and the communication wall — and the entire discipline of high-performance computing for AI is learning to *name which wall you are stuck behind and move the work to where the silicon is idle*.

![diagram of the three walls compute memory bandwidth and communication feeding into a measured MFU number and a fix step](/imgs/blogs/why-hpc-is-the-bottleneck-for-modern-ai-1.png)

This post is the spine of a 15-post series. By the end of it you will be able to do four things you probably cannot do crisply right now: (1) estimate how many FLOPs your model needs per token and compare that to your GPU's peak, (2) tell whether any given operation is *compute-bound* or *memory-bound* before you ever run it, (3) compute MFU and use it as the single north-star number for every optimization, and (4) read the map of *why* eight GPUs are often barely faster than one. We will set up the running example — train and serve a 1-to-7-billion-parameter Transformer on 1, then 8, then many GPUs — that every later post returns to. Then the rest of the series goes deep on each wall. Let's start by understanding why this problem exists at all.

## 1. FLOPs grew ~1000×, bandwidth only ~30× — the gap that defines the era

Here is the single most important graph in modern AI systems, and almost nobody plots it. Over the last decade, the peak *compute* of an NVIDIA datacenter GPU grew by roughly **three orders of magnitude**, while the *memory bandwidth* that feeds those compute units grew by barely **one and a half**. The cores got dramatically faster than the pipes that feed them. Everything painful about training and serving large models flows from that one divergence.

Let me define the two quantities precisely, because they are constantly confused and the difference is the whole point. A **FLOP** is a single floating-point operation — one multiply or one add. **FLOPs** (plural, with a lowercase *s*) is a *count*: how much arithmetic a workload requires, e.g. "a forward pass over this batch is 4.2 PetaFLOPs." **FLOP/s** (with a slash, "FLOPs per second") is a *rate*: how fast a chip can do arithmetic, e.g. "this GPU peaks at 312 TFLOP/s." A workload has FLOPs; hardware has FLOP/s. Time, in the best case, is `FLOPs ÷ FLOP/s`. Keep these straight and half the confusion in this field evaporates.

The other quantity is **memory bandwidth**, measured in bytes per second, usually TB/s. This is how fast the GPU can read and write its own on-board memory, which is a special high-speed type called **HBM** — High Bandwidth Memory, stacked DRAM dies sitting right next to the GPU die. Every number your kernel touches that is not already in a register or a tiny on-chip cache must travel over the HBM bus. Bandwidth, not compute, is what determines how fast you can *move* data.

Now the actual numbers, from NVIDIA's published specifications (these are vendor headline figures; real achievable numbers are lower, which is exactly the point of this series):

| GPU (year) | Peak dense FLOP/s | HBM bandwidth | FLOP per byte (balance) |
|---|---|---|---|
| V100 (2017) | 125 TFLOP/s (fp16) | 0.9 TB/s (HBM2) | ~139 |
| A100 80GB SXM (2020) | 312 TFLOP/s (bf16) | 2.0 TB/s (HBM2e) | ~156 |
| H100 SXM (2022) | 989 TFLOP/s (bf16) | 3.35 TB/s (HBM3) | ~295 |

Read the last column. The "balance point" of the chip — how many FLOPs it can do in the time it takes to read one byte — climbed from ~139 to ~295. That means each new generation demands you *reuse every byte you load more times* just to keep the compute units fed. A workload that was perfectly balanced on a V100 is *memory-starved* on an H100, because the H100's cores can chew through 295 FLOPs per byte but your operation only does, say, 60. The cores sit waiting for the next byte. This is not a bug; it is the deliberate trajectory of the hardware, and it means **the bottleneck for most deep-learning operations has shifted from compute to memory bandwidth.** A modern GPU is a starving giant.

It is worth tracing *why* the two curves diverged, because the cause is physics, not a vendor choice, and it will not reverse. From the V100 (2017) to the H100 (2022) — five years — peak dense bf16 throughput grew roughly **8×** (125 → 989 TFLOP/s), while HBM bandwidth grew only **~3.7×** (0.9 → 3.35 TB/s). Two distinct trends drive that. First, **compute** got cheaper per transistor through a mix of process shrinks *and* architectural specialization: the Tensor Core. A Tensor Core is a hardware matrix-multiply unit that performs a small dense matmul (e.g. a $4\times4$ tile) per instruction, so each clock retires dozens of multiply-accumulates instead of one. That is how NVIDIA bought an order of magnitude of FLOP/s without an order-of-magnitude transistor budget — they spent the transistors on units that do *only* the one operation deep learning needs most. Second, **bandwidth** is bottlenecked by physical pin count and the signaling rate of the wires between the GPU die and its stacked HBM. You cannot specialize your way out of moving a byte across a wire; HBM3 widened the bus and pushed the clock, but the gains are incremental because they are gated by I/O physics and power, not logic density.

Underneath both is the end of two free lunches the whole industry coasted on for decades. **Dennard scaling** — the observation that as transistors shrank, their power density stayed constant, so you could clock them faster every generation for free — broke down around 2005-2006. Once you cannot keep raising the clock without melting the chip, single-thread performance stops growing and the only way forward is *more parallelism*: more cores, wider SIMD, specialized units. That is exactly the GPU's bet, and it is why the FLOP/s curve keeps climbing — you add more parallel arithmetic units. But **Moore's law** (transistor count doubling every ~2 years) has also visibly slowed, and crucially, memory technology was never on Moore's curve in the first place. DRAM density and especially DRAM *bandwidth* have always improved far more slowly than logic — a gap memory architects have called the **"memory wall"** since Wulf and McKee named it in 1995. Deep learning simply dragged that decades-old CPU-era problem into the GPU era and made it the defining constraint of an entire industry. The cores race ahead on the back of specialization and parallelism; the pipes that feed them crawl forward on the back of I/O physics. Every architectural pattern in this series — fusion, tiling, FlashAttention, recomputation, sharding — is ultimately a response to that single, permanent divergence.

![before and after comparison showing V100 to H100 with peak FLOP rate growing about thirty times while HBM bandwidth grows about two times](/imgs/blogs/why-hpc-is-the-bottleneck-for-modern-ai-2.png)

There is a third axis the title promised: **watts**. A datacenter does not give you infinite GPUs; it gives you a power budget. An H100 draws up to 700 W; a rack is capped by what the building can cool. So the real currency of large-scale AI is not "GPUs" but **GPU-hours and the energy behind them**, which is why we eventually score everything in MFU and in dollars per result. When you double MFU, you have *halved* the GPU-hours, the electricity bill, and the carbon for the same trained model. That is why this single percentage matters more than any clever architectural tweak.

#### Worked example: how fast *should* a 7B model train?

Let me make the FLOPs-vs-FLOP/s idea concrete with the running example. A well-known rule of thumb (from the Kaplan and Chinchilla scaling-law work) is that a dense Transformer with $N$ parameters costs about **$6N$ FLOPs per training token** — $2N$ for the forward pass and $4N$ for the backward pass. Why $2N$ forward? Almost all the work is in matrix multiplies, and a matmul of a token's activation vector against the weights touches every parameter once in a multiply and once in an add — two FLOPs per parameter. The backward pass costs roughly twice the forward (you compute gradients with respect to both inputs and weights), giving $4N$, for $6N$ total.

So for a 7-billion-parameter model, $N = 7 \times 10^9$, training cost per token is:

$$C_\text{token} = 6N = 6 \times 7 \times 10^9 = 4.2 \times 10^{10} \text{ FLOPs/token.}$$

Now suppose we want to push **3,900 tokens per second** through one A100 (a number we will justify later). The required compute rate is:

$$\text{FLOP/s required} = 4.2 \times 10^{10} \times 3900 \approx 1.64 \times 10^{14} = 164 \text{ TFLOP/s.}$$

The A100 peaks at 312 bf16 TFLOP/s. So $164 / 312 \approx 52\%$ — that is the MFU you'd need to hit to reach 3,900 tokens/s. Flip it around: at the cold-start **18% MFU** my team saw, the achieved rate is only $0.18 \times 312 = 56$ TFLOP/s, which supports just $56 \times 10^{12} / 4.2 \times 10^{10} \approx 1{,}330$ tokens/s. Same hardware, same model — a 2.9× difference in throughput, and therefore in cost and calendar time, decided entirely by how well the job uses the machine. **This arithmetic is the job.** You will do versions of it in every post.

#### Worked example: one training step's FLOPs, bytes, and which wall it hits at batch 1 vs 256

The previous example treated the model as a single FLOP rate. Now let's open up *one training step* and account for both of its costs — the FLOPs it requires and the bytes it must move — because the ratio of those two is what decides which wall the step hits, and that ratio changes dramatically with batch size. This is the single most important calculation in the whole post, so I'll do it slowly.

Take the 7B model, sequence length $s = 2{,}048$, and process a batch of $b$ sequences in one step. The number of tokens per step is $T = b \times s$. The training FLOPs for the step, by the $6N$ rule, are:

$$\text{FLOPs}_\text{step} = 6N \times T = 6 \times 7\times10^9 \times (b \times 2048).$$

The *bytes that must move from HBM* are dominated, for a matmul-heavy step, by reading the weights at least once for the forward and once for the backward. The 7B model's weights in bf16 are $2N = 14\times10^9$ bytes $= 14$ GB. In the most weight-starved case (tiny batch, weights re-read each pass and not reused), the step reads roughly $2 \times 14 = 28$ GB of weight traffic, plus activation traffic that scales with the batch. Hold the weight term fixed and watch what the *arithmetic intensity* of the step — FLOPs per byte — does as $b$ grows:

**Batch $b = 1$** ($T = 2{,}048$ tokens). FLOPs $= 6 \times 7\times10^9 \times 2048 \approx 8.6\times10^{13}$. Weight bytes moved $\approx 2.8\times10^{10}$ (the 28 GB, which barely changes with such a small batch). Arithmetic intensity:

$$I_{b=1} = \frac{8.6\times10^{13}}{2.8\times10^{10}} \approx 3{,}070 \text{ FLOP/byte... per the dominant weight term.}$$

But that figure hides the real problem: at batch 1, every matmul is a *matrix-times-vector* (one token's activation against a big weight matrix), and a matrix-vector product reads the entire weight matrix to do only one multiply-accumulate per weight — its *own* intensity is barely above 2 FLOP/byte. The whole step is starved because each weight, dragged all the way from HBM, is used essentially once. Batch 1 is the textbook **memory-bandwidth-bound** regime: you pay the full 14 GB of weight reads to do a trivially small amount of math, so the cores idle. This is exactly why single-sequence inference (batch 1, autoregressive decoding) is memory-bound and why serving systems fight so hard to batch requests together.

**Batch $b = 256$** ($T = 524{,}288$ tokens). FLOPs $= 6 \times 7\times10^9 \times 524288 \approx 2.2\times10^{16}$ — a 256× jump, exactly as the batch grew. But the *weight* bytes are nearly unchanged: you still read each weight matrix about twice (forward, backward), because all 256 sequences in the batch reuse the *same* weights. The weight traffic is amortized across 256× more math. Now each weight loaded from HBM participates in 256 multiply-accumulates instead of one. The matmuls become matrix-times-*matrix* (a $256\times h$ activation block against the weights), whose arithmetic intensity scales with the batch dimension and easily clears the A100's ridge of 156 FLOP/byte. The step is now firmly **compute-bound** — the cores are busy, the weights are reused, and you are finally on the right roof.

The lesson is foundational and recurs in every later post: **batch size is the primary knob that moves an operation along the roofline.** Small batch starves the cores (memory-bound); large batch feeds them (compute-bound) by reusing each loaded weight across more tokens. This is *the* reason a larger batch lifted MFU from 31% to 50% in the running example, and it is why inference — where you often cannot batch — is a fundamentally harder bandwidth problem than training. The activation memory cost (Section 3) grows with batch in the opposite direction, so the practical art is finding the largest batch that still fits, which puts you as far up the compute roof as your 80 GB allows.

#### Worked example: the watts wall — GPU-hours, kWh, and dollars for a 7B run

The title promised a third axis, watts, and it deserves real numbers because power is what actually caps a datacenter and what ultimately gets billed. Take a standard training node: **8× H100 SXM at ~700 W per GPU**. The eight GPUs alone draw $8 \times 700 = 5{,}600$ W $= 5.6$ kW. A real node draws more once you count the CPUs, the HBM, the NVSwitch fabric, the NICs, fans, and power-supply losses — a commonly used rule of thumb is that the full node pulls roughly **1.4-1.5×** the bare GPU power, so call it about **8 kW** at the wall for the whole node under load. (Datacenter operators fold this overhead into a facility-wide **PUE**, power usage effectiveness; a modern facility runs ~1.1-1.2, adding cooling and distribution on top again.)

Now cost out a concrete run. Suppose the 7B model trains on 300 billion tokens at the tuned **50% MFU** on H100s. The total work is $6N \times \text{tokens} = 6 \times 7\times10^9 \times 3\times10^{11} = 1.26\times10^{22}$ FLOPs. At 50% of an H100's 989 TFLOP/s, each GPU delivers $0.50 \times 989\times10^{12} \approx 4.9\times10^{14}$ FLOP/s, so the run needs $1.26\times10^{22} / 4.9\times10^{14} \approx 2.6\times10^7$ GPU-seconds $\approx 7{,}140$ **GPU-hours**. On one 8-GPU node that is about $7{,}140 / 8 \approx 890$ wall-clock hours, or **~37 days** — and on, say, 16 such nodes (128 GPUs) it collapses to a bit over two days, network efficiency permitting.

Energy: at ~700 W per GPU, $7{,}140$ GPU-hours $\times 0.7$ kW $= 5{,}000$ **kWh** of GPU energy. Fold in node overhead and a 1.15 PUE and the facility draws roughly $5{,}000 \times 1.45 \times 1.15 \approx 8{,}300$ kWh at the meter. At an industrial rate of **\$0.10 per kWh**, the electricity alone is about **\$830** for GPU energy, ~\$1,250 wired-to-the-wall, ~\$1,440 with cooling — modest, because electricity is cheap relative to the silicon. The *capital/rental* cost dominates: at a representative **\$3 per H100 GPU-hour** on cloud, $7{,}140$ GPU-hours $\times \$3 = $ about **\$21,000** for this single 7B run. Notice that halving MFU would *double* every one of these figures — the GPU-hours, the kWh, the dollars, and the calendar — because the total FLOPs are fixed and time is FLOPs over achieved-rate. That is the watts wall in one sentence: **you do not buy GPUs, you buy GPU-hours and the power behind them, and MFU is the exchange rate.** All figures here are illustrative order-of-magnitude estimates; real prices vary widely by provider, contract, and region.

## 2. The three walls: the only three ways to be slow

Every slow AI workload — and I mean every one I have ever profiled — is stalled behind one of exactly three walls. Naming the wall is 80% of the fix, because each wall has a *different* set of levers, and pulling the wrong lever wastes days. Here is the taxonomy, with a one-line test for each.

**The compute wall.** You are limited by raw arithmetic throughput — the GPU's multiply-accumulate units are genuinely busy and there is simply a lot of math to do. Test: the operation has *high arithmetic intensity* (many FLOPs per byte loaded), the GPU is near its peak FLOP/s, and the only way to go faster is more or faster FLOP/s. A large dense matrix multiply on a Transformer's feed-forward layer is the canonical compute-bound op. **Good news: this is the wall you *want* to be stuck behind**, because it means the hardware is working. The levers are lower precision (bf16, fp8) to get more FLOP/s per cycle, and Tensor Cores, the specialized matmul units.

**The memory-bandwidth wall.** You are limited by how fast you can move bytes to and from HBM. The compute units are idle, *waiting* for data. Test: the operation has *low arithmetic intensity* — it reads a lot of data but does little math per byte. LayerNorm, softmax, activation functions, dropout, residual adds, and attention's intermediate tensors are all memory-bound. They touch huge tensors but do only a few FLOPs each. **This is the wall most deep-learning code is actually stuck behind**, and it is invisible in `nvidia-smi`, which is why the 18%-MFU job looked "100% utilized." The levers are *fewer trips to HBM*: kernel fusion (do several memory-bound ops in one pass while the data is on-chip), tiling (keep working data in fast SRAM), and IO-aware kernels like FlashAttention.

**The communication wall.** You are limited by moving data *between* GPUs — gradients summed across a data-parallel job, activations passed between pipeline stages, a tensor-parallel matmul split across devices. Test: a single GPU runs fine, but adding GPUs gives diminishing or negative returns; the profiler shows long gaps where the GPU waits on an `all-reduce` or a `send`/`recv`. **This is the wall that appears the moment you go multi-GPU**, and it gets worse as you cross node boundaries (NVLink inside a box is fast; InfiniBand between boxes is slower; PCIe is slower still). The levers are better collective algorithms (ring vs tree all-reduce), overlapping communication with computation, faster interconnects, and smarter parallelism strategies that move less data.

![graph of a training job branching into the compute wall the memory bandwidth wall and the communication wall and converging on measured MFU](/imgs/blogs/why-hpc-is-the-bottleneck-for-modern-ai-1.png)

Here is the three-wall taxonomy as a table you can keep open while you profile. The "tell" column is what the wall looks like in a profiler; the "lever" column is what actually moves the number, and the "later post" column is where the series goes deep on it.

| Wall | What limits you | Tell in the profiler | Levers | Series track |
|---|---|---|---|---|
| Compute | Peak FLOP/s — the math units are genuinely busy | A few large matmul kernels back-to-back, near-peak | bf16/fp8, Tensor Cores, less redundant math | B (precision) |
| Memory bandwidth | HBM bytes/s — cores wait on data | Many tiny kernels with gaps, low FLOP/s per kernel | fusion, tiling, FlashAttention, better access | B (kernels) |
| Communication | Bytes between GPUs — devices wait on each other | Long flat gaps during all-reduce / send / recv | overlap, ring vs tree, FSDP, faster interconnect | C (scale-out) |

The discipline is simple to state and hard to master: **profile to find the dominant wall, prove it with a number, pull the matching lever, then re-measure.** Never guess. I have watched brilliant engineers spend a week hand-writing a fused kernel for an operation that was already compute-bound and near peak — they pulled the memory lever on a compute problem and got nothing. And I have watched teams add tensor parallelism (a communication-heavy strategy) to a model that fit comfortably on one GPU, *slowing it down* because they added a wall that wasn't there before. The three-wall frame exists to stop exactly these mistakes.

A subtle but critical point: these walls are *per-operation* and *per-phase*, not per-job. A single training step can be compute-bound in the feed-forward matmuls, memory-bound in the LayerNorms and the attention softmax, and communication-bound in the gradient all-reduce — all in the same step. Optimizing means finding which *segments of the timeline* dominate and attacking those. A profiler timeline, which later posts cover in depth, is how you see this decomposition directly. The roofline model, which we teaser below, is how you predict it before you even run.

## 3. Where the 80 GB goes — the memory budget that decides if you can train at all

Before a model can be slow, it has to *fit*. The first thing the compute wall and the memory wall do together is decide whether your training job even starts, or whether it dies with the most familiar error in deep learning: `CUDA out of memory`. To reason about this you need to know where GPU memory goes, and the answer surprises most people: **the parameters are a small slice of the total.**

Let's account for it precisely with the 7B running example, training in mixed precision with the Adam optimizer (the standard recipe). For a model with $\Psi$ parameters, the memory budget breaks down like this:

- **Parameters** (the weights), stored in bf16 (2 bytes each): $2\Psi$ bytes.
- **Gradients**, one per parameter, also bf16: $2\Psi$ bytes.
- **Optimizer states.** Adam keeps a first moment ($m$) and second moment ($v$) per parameter, and the standard mixed-precision recipe *also* keeps an fp32 master copy of the weights for stable updates. All three in fp32 (4 bytes each): $4\Psi + 4\Psi + 4\Psi = 12\Psi$ bytes.
- **Activations.** The intermediate tensors saved during the forward pass so the backward pass can compute gradients. This depends on batch size, sequence length, and model depth — and it can be *enormous*, often rivaling everything else combined.

The famous accounting from the ZeRO paper sums the first three to $(2 + 2 + 12)\Psi = 16\Psi$ bytes for the model and optimizer state alone. For $\Psi = 7 \times 10^9$:

$$16 \times 7 \times 10^9 = 1.12 \times 10^{11} \text{ bytes} = 112 \text{ GB.}$$

That is **112 GB before a single activation is stored**, and an A100 has 80 GB. The 7B model *does not fit on one A100 in standard mixed-precision training* — not even close — and that is before activations, which for a reasonable batch and sequence length add tens of GB more. This is not an exotic edge case; it is the default situation for any model past a couple billion parameters, and it is why the entire Track C of this series exists (ZeRO, FSDP, activation checkpointing, offload, parallelism). The memory budget is the *first* constraint you hit, before throughput is even on the table.

![stacked bar showing GPU memory split into parameters gradients optimizer states and activations exceeding the eighty gigabyte A100 budget](/imgs/blogs/why-hpc-is-the-bottleneck-for-modern-ai-3.png)

#### Worked example: does a 1B model fit, and how big a batch?

Take the smaller end of the running example, a 1B-parameter model, $\Psi = 1 \times 10^9$. Model + optimizer state is $16\Psi = 16$ GB. That fits in an A100's 80 GB with 64 GB to spare for activations — comfortable. Now estimate activations. A rough per-layer activation memory for a Transformer is on the order of $s \cdot b \cdot h \cdot (\text{const})$ bytes, where $s$ is sequence length, $b$ is batch size, and $h$ is hidden size; the constant captures all the intermediate tensors (attention scores, FFN hidden, etc.). Without recomputation, activation memory grows *linearly with batch size*. So with 64 GB of headroom, you can push the batch until activations fill it — and the moment you cross 80 GB total, you OOM.

The practical lesson: when you OOM, you have three honest options, and they correspond to later posts — (1) reduce activation memory with **activation checkpointing** (recompute in the backward pass instead of storing, trading compute for memory), (2) shrink the optimizer footprint with **ZeRO/FSDP** (shard the $12\Psi$ states across GPUs), or (3) reduce per-GPU batch and use **gradient accumulation** to keep the effective batch large. Knowing the $2 + 2 + 12$ split tells you *which* knob has the most to give. For a 7B model, the optimizer states are the elephant ($84$ GB of the $112$), so sharding them (ZeRO-1/2/3) is the highest-leverage move. For a small model with a long sequence, activations dominate, so checkpointing wins. The budget math, not intuition, picks the lever.

Here is a tiny but genuinely useful snippet — a back-of-the-envelope memory estimator you can run before launching a job, to predict an OOM instead of discovering it after a 20-minute startup:

```python
def training_memory_gb(params_billion, optimizer="adam",
                       activation_gb=0.0, dtype_bytes=2):
    """Estimate training memory (GB) for a dense model in mixed precision.
    Returns model+grad+optimizer state, plus your activation estimate.
    Does NOT include framework/CUDA context overhead (~1-2 GB)."""
    psi = params_billion * 1e9
    params = dtype_bytes * psi            # bf16 weights
    grads = dtype_bytes * psi             # bf16 gradients
    if optimizer == "adam":
        opt = 12 * psi                    # fp32 m, v, master copy
    elif optimizer == "sgd":
        opt = 4 * psi                     # fp32 master copy only
    else:
        opt = 0
    total_bytes = params + grads + opt
    return {
        "params_gb": params / 1e9,
        "grads_gb": grads / 1e9,
        "optimizer_gb": opt / 1e9,
        "activations_gb": activation_gb,
        "total_gb": total_bytes / 1e9 + activation_gb,
    }

print(training_memory_gb(7, activation_gb=20))
# {'params_gb': 14.0, 'grads_gb': 14.0, 'optimizer_gb': 84.0,
#  'activations_gb': 20, 'total_gb': 132.0}  -> will NOT fit on 80 GB A100
```

Run that before you launch and you will never again be surprised by an OOM at step zero. The point of HPC thinking is to compute the answer *before* the cluster does.

## 4. The optimization loop: measure, name the wall, pull one lever, re-measure

Performance work is a *loop*, and treating it as anything else — a one-shot "make it fast" — is how teams waste weeks. The loop has four steps, and the discipline is to go around it one lever at a time, never changing two things at once, always re-measuring.

![timeline of the optimization loop with steps profile find the wall pick a lever and measure MFU](/imgs/blogs/why-hpc-is-the-bottleneck-for-modern-ai-4.png)

**Step 1 — Profile.** Get a real measurement of where time goes. The cardinal sin is optimizing from intuition; the human guess about the bottleneck is wrong more often than it is right. The tools (covered fully in the profiling post) are NVIDIA **Nsight Systems** (`nsys`) for a system-wide timeline that shows GPU gaps, host-vs-device overlap, and kernel durations; **Nsight Compute** (`ncu`) for a per-kernel roofline and occupancy; and PyTorch's built-in `torch.profiler`, which exports a Chrome/Perfetto trace you can scrub through visually.

**Step 2 — Find the wall.** From the profile, classify the dominant cost. Are the GPUs busy doing matmuls (compute), waiting on memory between many small kernels (memory bandwidth), or idle during an `all-reduce` or a `recv` (communication)? A useful tell: if your timeline is dominated by *many tiny kernels* with gaps between them, you are memory-bound and probably kernel-launch-bound; if it's a *few large matmul kernels* running back to back, you are compute-bound; if there are long flat stretches where the GPU does nothing, look at data loading or collectives.

**Step 3 — Pick one lever.** Match the lever to the wall (Sections 2 and the rest of the series enumerate them). Compute wall → lower precision, Tensor Cores. Memory wall → fusion, FlashAttention, better access patterns. Communication wall → overlap, better collectives, different parallelism. *One* lever, so you can attribute the change.

**Step 4 — Measure MFU.** Re-run, compute MFU (next section), and compare. Did the number move? If yes, keep it and go around again on the next-biggest cost. If no, *revert* — a change that doesn't move the number is complexity you don't want. The loop terminates when MFU is "good enough" for your hardware (40–55% is excellent for training; we'll calibrate this in the case studies) or when the marginal engineering cost exceeds the marginal speedup.

Here is the honest measurement harness that anchors Step 1 and Step 4. The single most common mistake in DIY benchmarking is forgetting that **CUDA is asynchronous** — when your Python line returns, the GPU work has only been *queued*, not finished. If you time it with `time.time()` you measure how long it took to *launch* the kernel, not run it, and you'll report numbers that are wildly, comically wrong. You must use CUDA events and synchronize. You must also warm up (the first iterations include compilation and allocator warmup) and time the *steady state*.

```python
import torch

def benchmark(fn, *args, warmup=10, iters=50):
    """Honest GPU timing. CUDA is async: you MUST use events + synchronize,
    not time.time(), or you measure launch latency, not execution time."""
    # Warm up: first calls include cuDNN autotuning, allocator growth, JIT.
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()  # drain the queue before we start the clock

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()  # wait for ALL queued work to actually finish

    ms_per_iter = start.elapsed_time(end) / iters  # elapsed_time is in ms
    return ms_per_iter

# Example: time one big matmul on the GPU.
a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
ms = benchmark(lambda: a @ b)
flops = 2 * 8192 ** 3                      # 2*M*N*K for an MxK @ KxN matmul
tflops = flops / (ms / 1e3) / 1e12
print(f"{ms:.3f} ms/iter, {tflops:.1f} TFLOP/s achieved")
```

That harness — warm up, `synchronize()` before *and* after, time with CUDA events, divide known FLOPs by measured seconds — is the foundation of every measurement in this series. Internalize it now. The `2 * M * N * K` formula for a matmul's FLOP count (each of the $M \times N$ output elements is a dot product of length $K$, which is $K$ multiplies and $K$ adds) is the other half: known FLOPs over measured time gives you achieved FLOP/s, and achieved over peak gives you MFU.

## 5. The roofline: a 30-second test for which wall you're behind

The roofline model is the single most useful mental model in this entire field, and it fits on the back of a napkin. It lets you predict — *before you run anything* — whether an operation will be compute-bound or memory-bound on a given chip. Once it clicks, you will never look at a kernel the same way again.

![graph of the roofline showing arithmetic intensity splitting operations into a memory bound region and a compute bound region at the ridge point](/imgs/blogs/why-hpc-is-the-bottleneck-for-modern-ai-5.png)

Here is the science. Every operation has two costs: the FLOPs it must do, and the bytes it must move. Their ratio is the operation's **arithmetic intensity**:

$$I = \frac{\text{FLOPs}}{\text{bytes moved from/to HBM}} \quad \left[\frac{\text{FLOP}}{\text{byte}}\right].$$

The hardware has two ceilings: a peak compute rate $P$ (FLOP/s) and a peak bandwidth $B$ (bytes/s). The maximum *attainable* performance for an op with intensity $I$ is whichever ceiling it hits first:

$$\text{Performance} = \min\left(P,\; I \times B\right) \quad [\text{FLOP/s}].$$

The two terms cross at the **ridge point**, the intensity where the chip transitions from memory-limited to compute-limited:

$$I_\text{ridge} = \frac{P}{B}.$$

For the A100: $P = 312 \times 10^{12}$ FLOP/s, $B = 2.0 \times 10^{12}$ bytes/s, so $I_\text{ridge} = 312/2.0 = 156$ FLOP/byte. (Note this is exactly the "balance point" from the spec table in Section 1 — it's the same quantity.) The rule is now mechanical: **if your operation's arithmetic intensity $I < 156$, it is memory-bound on an A100 — you will not reach peak FLOP/s no matter what, because you'll run out of bandwidth first. If $I > 156$, it's compute-bound.** And critically, the ridge moved *right* on the H100 (to ~295), so an operation that was compute-bound on an A100 can become memory-bound on an H100. The hardware trajectory pushes more and more operations below the ridge.

#### Worked example: a big GEMM vs a LayerNorm on an A100

Let's place two real Transformer operations on the roofline.

**A large GEMM (matrix multiply)**, say $M = N = K = 8192$ in bf16. FLOPs $= 2MNK = 2 \times 8192^3 \approx 1.1 \times 10^{12}$. Bytes moved: read two $8192 \times 8192$ bf16 matrices and write one, $3 \times 8192^2 \times 2 \approx 4.0 \times 10^8$ bytes. Arithmetic intensity:

$$I_\text{GEMM} = \frac{1.1 \times 10^{12}}{4.0 \times 10^8} \approx 2{,}730 \text{ FLOP/byte.}$$

That is *far* above the 156 ridge, so this GEMM is firmly **compute-bound**. It can in principle approach the A100's peak 312 TFLOP/s, and the only way to make it faster is more FLOP/s (lower precision, better Tensor Core utilization). Don't go hunting for a fused kernel here — there's nothing to fuse; you're already on the right roof.

**A LayerNorm** over a $b \times s \times h$ activation tensor, say a tensor of $T = 4{,}194{,}304$ elements in bf16. LayerNorm reads the tensor, computes a mean and variance (a handful of FLOPs per element), normalizes, and writes it back — roughly $\sim 10$ FLOPs per element but $2 \times 2 = 4$ bytes of HBM traffic per element (read in, write out, bf16):

$$I_\text{LN} = \frac{10 \times T}{4 \times T} = 2.5 \text{ FLOP/byte.}$$

That is *vastly* below 156, so LayerNorm is hopelessly **memory-bound**. It will run at $2.5 \times 2.0 \times 10^{12} = 5 \times 10^{12} = 5$ TFLOP/s — about **1.6% of peak** — and there is *nothing* you can do to raise its FLOP/s, because it's bandwidth-limited. The only win is to *do less HBM traffic*: fuse it into the neighboring operation so the data never round-trips to HBM (covered in the kernel-fusion post). This is why fusion and FlashAttention exist: the memory-bound operations are where the time leaks, and the only lever is fewer bytes moved. The roofline tells you this in one division, before you write a line of CUDA.

The roofline also explains the deepest source of low MFU: a Transformer is a *mix* of a few compute-bound GEMMs (the FFN and the QKV projections) and many memory-bound operations (LayerNorm, softmax, residuals, dropout, the attention score handling). If the memory-bound ops dominate wall-clock time, your *overall* MFU is dragged down even though the GEMMs themselves run near peak. The whole art of single-GPU optimization is shrinking the memory-bound fraction.

## 6. From 1 to 8 to many GPUs — where scaling efficiency leaks

The running example trains on 1, then 8, then many GPUs, and the transition exposes the third wall in full. Here is the brutal arithmetic of scaling: ideally, 8 GPUs would be 8× faster than 1, and 64 would be 64×. In reality they never are, and *understanding why* is the entire Track C of this series.

![graph of scaling from one GPU to eight on NVLink to sixty four over InfiniBand with a gradient all-reduce that leaks efficiency](/imgs/blogs/why-hpc-is-the-bottleneck-for-modern-ai-6.png)

Start with **one GPU**. There is no communication; you are limited by the compute and memory walls from the previous sections. Your job here is to maximize single-GPU MFU. This is where 80% of the practical wins live for most teams, and it's why Track B (precision, kernels, fusion, profiling) comes *before* the multi-GPU material. You should never scale out a job you haven't first made efficient on one device — you'll just multiply the waste by N.

Go to **8 GPUs in one node**. The simplest strategy is **data parallelism**: replicate the whole model on each GPU, give each a different slice of the batch, and after each backward pass *sum the gradients across all GPUs* so every replica updates identically. That summation is an **all-reduce** collective — every GPU contributes its gradient and every GPU receives the total. The gradients for a 7B model are $2\Psi = 14$ GB, and that volume has to traverse the interconnect every single step. Inside a DGX box, GPUs are linked by **NVLink/NVSwitch** at ~900 GB/s aggregate, which is fast enough that, if you overlap the all-reduce with the backward pass (which DDP does automatically), the communication can hide almost entirely behind computation. So 8 GPUs on NVLink often *do* get ~7× — scaling efficiency around 85–95%.

Now go to **64 GPUs across 8 nodes**. The GPUs *within* each node still talk over NVLink, but the nodes talk to each other over **InfiniBand** — a datacenter network, fast by network standards (200–400 Gb/s per link, i.e. ~25–50 GB/s) but **an order of magnitude slower than NVLink**. The all-reduce now has to cross these slower inter-node links, and the time it takes grows with the number of participants. This is where scaling efficiency falls off a cliff: the communication wall, hidden at 8 GPUs, dominates at 64. A job that got 90% efficiency on one node might get 60% across eight, meaning your 64 GPUs deliver the work of ~40. Every GPU you add past the point where communication dominates is partly wasted.

The science behind the all-reduce cost is worth previewing because it explains *why* the leak is fundamental, not a bug. A well-implemented **ring all-reduce** moves $2(N-1)/N \cdot S$ bytes per GPU, where $S$ is the size of the data (gradients) and $N$ is the number of GPUs. As $N$ grows, $2(N-1)/N \to 2$, so each GPU moves about $2S$ bytes no matter how many GPUs you have — the *per-GPU* volume is bounded, which is the clever part. But the *time* still depends on the slowest link in the ring, and once that ring spans slow inter-node links, the whole collective runs at inter-node speed. The collective-communication post derives this in full; the takeaway here is that **the interconnect topology sets a hard ceiling on useful parallelism**, and crossing a node boundary is the single biggest efficiency cliff in distributed training.

#### Worked example: gradient all-reduce volume for a 7B model

Concretely: a 7B model has $S = 2\Psi = 14$ GB of bf16 gradients. With ring all-reduce on $N = 8$ GPUs, each GPU moves $2(8-1)/8 \times 14 = 24.5$ GB per step. At NVLink's ~900 GB/s effective, that's $24.5 / 900 \approx 27$ ms of communication per step. If a training step's compute takes ~200 ms, the all-reduce is 27 ms that can mostly overlap — manageable. But move to inter-node InfiniBand at ~50 GB/s effective and that same 24.5 GB takes $24.5/50 \approx 490$ ms — now communication is *more than twice* the compute, and unless you overlap aggressively or shard differently, your GPUs sit idle waiting for gradients more than half the time. Same model, same math, a 18× difference in communication time purely from which wire the bytes travel on. **This is why the network is the new bottleneck**, and why posts on collectives, interconnects, and sharding (ZeRO/FSDP, which avoids replicating gradients at all) make up an entire track.

## 7. MFU: the one number that scores everything

If you take one habit from this entire series, make it this: **measure MFU, and treat it as your north-star metric.** MFU collapses "is my expensive hardware actually working?" into a single percentage that is comparable across models, hardware, and time. A 7B job at 18% MFU and a 70B job at 18% MFU are *both* wasting 82% of their silicon, and the number tells you so directly.

The definition is exactly what you'd guess from the FLOPs/FLOP-s distinction:

$$\text{MFU} = \frac{\text{achieved FLOP/s}}{\text{peak FLOP/s}} = \frac{(\text{model FLOPs per step}) / (\text{step time})}{\text{hardware peak FLOP/s} \times N_\text{GPUs}}.$$

The numerator is the *useful* arithmetic per second — the model's required FLOPs (using the $6N$-per-token estimate, or a more exact per-layer count) divided by how long a step actually takes. The denominator is the hardware's theoretical peak across all the GPUs you're using. MFU is the ratio, and it is honest in a way `nvidia-smi` "utilization" is not: that field tells you whether *a* kernel is running, not whether the kernel is doing *useful, near-peak* work. A 100%-utilized GPU running a memory-bound LayerNorm at 1.6% of peak is "100% utilized" and 1.6% MFU. MFU sees through that.

![matrix comparing a cold start run at eighteen percent MFU against a tuned run at fifty percent MFU on the same A100 hardware](/imgs/blogs/why-hpc-is-the-bottleneck-for-modern-ai-7.png)

A close cousin you'll also see is **HFU** (Hardware FLOPs Utilization), which counts *all* the FLOPs the hardware actually performed — including the redundant ones from activation checkpointing (where you recompute the forward pass during backward). HFU is always ≥ MFU. MFU counts only the *minimum necessary* model FLOPs and is the better measure of end-to-end efficiency, because recomputation is a memory-saving cost, not useful progress. When a paper quotes "MFU," it usually means the model-FLOPs version; be aware some report HFU and the gap can be 10+ points.

It helps to write the formula out fully so you can see every term you are accountable for. Spelled out for a dense Transformer training run:

$$\text{MFU} = \frac{6N \times (\text{tokens per step})}{(\text{step time in seconds}) \times P_\text{peak} \times N_\text{GPUs}},$$

where $N$ is parameter count, $P_\text{peak}$ is the per-GPU dense peak FLOP/s *at the precision you actually run* (use the bf16 number if you train in bf16, the fp8 number if you train in fp8 — mismatching these is a classic way to report a bogus MFU), and the $6N \times \text{tokens}$ numerator is the irreducible useful work. Everything that makes the *denominator* big without doing useful work in the numerator is what drags MFU down. There are five usual suspects, and naming them is half of any tuning session.

First, **small batch / matrix-vector regime.** As the batch-size worked example showed, a tiny per-GPU batch leaves the matmuls memory-bound — the cores idle waiting on weight reads, the step time balloons, and MFU craters. Raising the batch (or sequence length) until the GEMMs are compute-bound is almost always the first and biggest win. Second, **memory-bound non-matmul work** — the LayerNorms, softmaxes, residual adds, dropout, and activation functions that the roofline says run at single-digit percent of peak. If they aren't fused, they dominate wall-clock time and pull the *average* down even while the GEMMs run near peak; this is the gap that kernel fusion and FlashAttention close. Third, **fp32 instead of low precision.** Running matmuls in fp32 forfeits the Tensor Cores entirely (TF32/bf16/fp8 are where the headline FLOP/s live), so you are dividing useful work by a peak the hardware can't even reach in that mode — turning on AMP often doubles MFU in an afternoon. Fourth, a **starving data loader.** If the CPU input pipeline can't feed batches as fast as the GPU consumes them, the GPU sits idle between steps — pure dead time in the step-time denominator that `nvidia-smi` will still cheerfully report as busy. Fifth, on multi-GPU jobs, **exposed communication** — any portion of the gradient all-reduce or the tensor-parallel exchange that does *not* overlap with computation is GPU idle time that inflates the step and deflates MFU, which is why overlap and interconnect topology (Section 6) matter so much at scale.

This is why **30-50% MFU is "good" and ~50-55% is excellent**, not 90%+. Even with every lever pulled perfectly, a real Transformer carries an irreducible tax: the memory-bound operations the roofline guarantees, the optimizer step, the inevitable small bubbles in any pipeline, and the precision-conversion overhead. A workload that was *only* large GEMMs could in principle approach peak; a real model with attention, normalization, and an optimizer cannot. So the realistic target is not 100% — it is "have I closed the gap to my model's own roofline ceiling?", and the five drags above are the checklist for getting there.

Here is a runnable MFU calculator for the running example. Wire it into your training loop and print it every N steps:

```python
# Peak dense FLOP/s by GPU and dtype (NVIDIA spec sheets; dense, no sparsity).
PEAK_FLOPS = {
    ("A100", "bf16"): 312e12,
    ("A100", "fp16"): 312e12,
    ("H100", "bf16"): 989e12,   # SXM, dense bf16
    ("H100", "fp8"): 1979e12,   # SXM, dense fp8
    ("V100", "fp16"): 125e12,
}

def mfu(num_params, tokens_per_step, step_time_s,
        gpu="A100", dtype="bf16", num_gpus=1):
    """Model FLOPs Utilization for a dense Transformer.
    Uses the 6*N FLOPs-per-token training estimate."""
    model_flops_per_step = 6 * num_params * tokens_per_step
    achieved = model_flops_per_step / step_time_s          # FLOP/s
    peak = PEAK_FLOPS[(gpu, dtype)] * num_gpus
    return achieved / peak

# Cold-start run: 7B model, batch 8 x seq 2048 = 16384 tokens, 11.7 s/step.
print(mfu(7e9, 16384, 11.7, gpu="A100", num_gpus=1))   # ~0.18  -> 18% MFU
# After tuning (bf16 + fusion + bigger batch): same tokens in 4.2 s.
print(mfu(7e9, 16384, 4.2, gpu="A100", num_gpus=1))    # ~0.50  -> 50% MFU
```

Notice that *nothing about the model changed* between those two calls — same parameters, same tokens — only the step time dropped from 11.7 s to 4.2 s, and MFU jumped from 18% to 50%. That 2.8× came from the levers this series teaches: bf16 instead of fp32 (more FLOP/s on the Tensor Cores), kernel fusion (fewer HBM round-trips for the memory-bound ops), and a larger batch (better amortization of fixed launch overheads). Each lever moved the number; each was kept because it did.

#### Worked example: turning MFU into dollars and calendar time

MFU is not an academic score; it converts directly into money and weeks, and that conversion is what gets a performance project funded. Suppose you need to train the 7B model on **300 billion tokens** (a Chinchilla-style budget for that size). The total work is fixed by the model and the data: $6N \times \text{tokens} = 6 \times 7 \times 10^9 \times 3 \times 10^{11} = 1.26 \times 10^{22}$ FLOPs. The *time* to do it depends entirely on achieved FLOP/s, which is MFU times peak.

At the naive **18% MFU** on a single A100 (achieved $0.18 \times 312 \times 10^{12} = 56$ TFLOP/s), the run takes $1.26 \times 10^{22} / 56 \times 10^{12} \approx 2.25 \times 10^8$ seconds $\approx 2{,}600$ GPU-days. At a representative cloud rate of \$2 per A100 GPU-hour, that is $2{,}600 \times 24 \times \$2 = \$125{,}000$. Now apply the single-GPU levers to reach **50% MFU** (achieved 156 TFLOP/s): the same work takes $1.26 \times 10^{22} / 156 \times 10^{12} \approx 940$ GPU-days, or about \$45,000. The bf16-plus-fusion-plus-batch changes — a few days of engineering — saved roughly **\$80,000 and 1,660 GPU-days** on one training run, and the savings scale linearly with every run after. On a frontier-scale job with thousands of GPUs, the same percentage points are millions of dollars. That is why MFU is the number on the dashboard, and why "make MFU go up" is a fundable engineering objective rather than a vanity metric.

A caveat worth stating plainly: MFU has a ceiling below 100% even for perfect code, because real workloads contain irreducible non-matmul work — the memory-bound LayerNorms and softmaxes from Section 5, the optimizer step, the data movement. A model that is *all* large GEMMs could approach peak; a real Transformer with attention and normalization cannot. This is why ~50–55% is "excellent" rather than "leaving half on the table" — a chunk of the other half is the memory-bound tax the roofline guarantees. Knowing the realistic ceiling for *your* model (you can estimate it by summing the time each op would take at its own roofline limit) keeps you from chasing the last few points where there's nothing left to win.

You can read MFU directly off `nvidia-smi` for a sanity check on power and memory, but for the FLOP accounting you need either the calculator above or a profiler. Quick command-line checks every practitioner should know:

```bash
# Live view: GPU utilization%, memory used, power draw, temperature, clocks.
nvidia-smi dmon -s pucmt          # p=power u=util c=clocks m=mem t=temp

# One-shot: which processes hold memory, and how much.
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,power.draw \
           --format=csv

# System timeline trace (gaps, kernels, host<->device) for one training step:
nsys profile -t cuda,nvtx,osrt -o train_trace \
     python train.py --max-steps 5
```

The `nvidia-smi dmon` view tells you if you're power- or thermal-throttling (clocks dropping under load is a real and common cause of mysterious slowdowns). The `nsys` trace is what you open to *find the wall* in Step 2 of the loop. But the FLOP accounting — the actual MFU — comes from the calculator. That percentage is the number you optimize.

## 8. The running example, end to end: a before→after on named hardware

Let me tie Sections 1–7 together with the concrete before→after that this whole series is, in miniature. The setup: train a 7B-parameter Transformer, sequence length 2048, on a single A100 80GB SXM (312 bf16 TFLOP/s, 2.0 TB/s HBM2e), and then on 8 of them. The "before" is the naive PyTorch loop most people write first; the "after" is the same model with the single-GPU and multi-GPU levers applied. These numbers are representative of what the techniques deliver — treat the exact figures as illustrative of the *shape* of the win, calibrated against the public results in the next section, not as a benchmark of your specific model.

| Configuration | Precision | Step time | Achieved TFLOP/s | MFU | Tokens/s | Peak mem |
|---|---|---|---|---|---|---|
| 1×A100, naive | fp32 | 11.7 s | 56 | 18% | 1,400 | OOM-prone |
| 1×A100, +bf16 (AMP) | bf16 | 6.8 s | 96 | 31% | 2,400 | lower |
| 1×A100, +fusion +batch | bf16 | 4.2 s | 156 | 50% | 3,900 | tuned |
| 8×A100 DDP, NVLink | bf16 | 4.5 s | 1,180 (agg) | 47% | 29,100 | sharded |

Read the story in the table. Switching from fp32 to **bf16** mixed precision via AMP nearly doubled throughput, because bf16 matmuls run on Tensor Cores at far higher FLOP/s and the smaller tensors move fewer bytes — it attacked both the compute and memory walls at once. Adding **kernel fusion** (so the memory-bound LayerNorms, residuals, and activations stop round-tripping to HBM) plus a **larger batch** (amortizing launch overheads) lifted MFU to 50% — excellent for a single GPU. Then **8-way data parallelism over NVLink** scaled it to ~7.4× the tokens/s of one GPU (29,100 vs 3,900), holding 47% MFU — only a few points of efficiency lost to the gradient all-reduce, because NVLink is fast enough to hide it. The aggregate 1,180 TFLOP/s is 8 GPUs each near 156. That is what "thinking like an HPC engineer" buys: a 2.8× single-GPU win stacked on a 7.4× scaling win, for ~20× the throughput of the naive loop on the same hardware — which is the difference between a five-week run and a two-day run, and between a \$200,000 cloud bill and a \$10,000 one.

The "after" loop is not exotic. Here is the AMP (Automatic Mixed Precision) core that delivered the first jump — the single highest-leverage change most teams can make in an afternoon:

```python
import torch
from torch.cuda.amp import autocast, GradScaler

model = build_transformer().cuda()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
scaler = GradScaler()                 # only needed for fp16; bf16 can skip it

for step, (x, y) in enumerate(loader):
    x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
    opt.zero_grad(set_to_none=True)   # set_to_none frees grad memory faster

    # autocast runs matmuls/convs in bf16 on Tensor Cores, keeps reductions
    # (softmax, layernorm sums) in fp32 for stability -- the best of both.
    with autocast(dtype=torch.bfloat16):
        out = model(x)
        loss = loss_fn(out, y)

    # bf16 has fp32's exponent range, so it rarely under/overflows -> no
    # loss scaling needed. (fp16 would need scaler.scale(loss).backward().)
    loss.backward()
    opt.step()
```

And the multi-GPU launch that delivered the scaling win — `torchrun` plus a one-line `DistributedDataParallel` wrap, which overlaps the gradient all-reduce with the backward pass automatically:

```bash
# Launch 8 processes, one per GPU, on a single node. torchrun sets the
# RANK / LOCAL_RANK / WORLD_SIZE env vars the script reads below.
NCCL_DEBUG=INFO torchrun --standalone --nproc_per_node=8 train_ddp.py
```

```python
import os, torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend="nccl")          # NCCL = NVIDIA's collectives
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

model = build_transformer().cuda(local_rank)
model = DDP(model, device_ids=[local_rank])      # auto all-reduce of grads,
                                                 # overlapped with backward
# ... the AMP training loop from above runs unchanged on each rank ...
```

That is the whole running example: the same model, made ~20× faster by naming each wall and pulling the matching lever. Every later post in the series zooms into one of these levers and shows you how to push it further — fp8 on Hopper, FlashAttention for the attention block, FSDP to fit a 70B model, the right parallelism for your interconnect.

## Case studies / real numbers

Numbers I make up are worthless; here are real, cited figures so you can calibrate what "good" MFU looks like and trust the shape of the table above. Treat all of these as the reported figures from the cited sources, on the hardware and software of their time.

**GPT-3 / Megatron-LM training MFU.** NVIDIA's Megatron-LM scaling work reported training large GPT models at roughly **45–52% MFU** on A100 clusters using 3D parallelism (tensor + pipeline + data) — and that was considered a strong, hard-won result, not a baseline. The original GPT-3 training, by contrast, was estimated at a much lower effective utilization. The lesson: even the best-engineered frontier training runs leave roughly half the FLOPs on the floor, and getting from a naive ~18% cold start to a tuned ~50% is the entire job. (Source: Narayanan et al., *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*, 2021.)

**PaLM's 540B at high MFU.** Google's PaLM technical report stands out for reporting an unusually high **~46% MFU** (and ~57% HFU with their accounting) for a 540B-parameter model on TPU v4 pods — they emphasized that hitting this required careful parallelism and a compiler (the XLA stack) that fuses aggressively. The headline takeaway from the paper's own framing: high MFU at scale is achievable but is an engineering achievement, and most teams fall well short of it. (Source: Chowdhery et al., *PaLM*, 2022.)

**Llama 3: a frontier run at scale, with its MFU reported.** Meta's Llama 3 paper is unusually candid about the systems side, and the numbers anchor what "good" looks like on a real 16,000-GPU H100 cluster. They report sustained training throughput corresponding to roughly **38-43% MFU** (about 380-430 TFLOP/s per H100 in bf16 against the ~989 peak) for the 405B model on tens of thousands of GPUs — a figure that *fell* somewhat as they scaled past a few thousand GPUs, precisely because of the communication wall this post describes. Just as telling, they describe a hardware-reliability story most people never see: with 16,384 GPUs running for weeks, *hardware failures were frequent enough to interrupt training regularly*, and a meaningful slice of engineering went into checkpointing and fast recovery rather than raw FLOP/s. The takeaway calibrates the whole series: even a best-in-class 2024 frontier run lands around 40% MFU, scaling efficiency erodes as GPU count climbs, and at the largest scale *reliability* becomes a first-class HPC problem alongside throughput. (Source: Grattafiori et al., *The Llama 3 Herd of Models*, 2024; figures approximate and as reported.)

**The cost of a frontier run, order of magnitude.** Public estimates put GPT-3's training compute at roughly $3.1\times10^{23}$ FLOPs and its cost in the low **single-digit millions of dollars** of compute; GPT-4-class and Llama-3-405B-class runs are widely estimated (by third parties, not the labs) in the **tens of millions to ~\$100M+** range of compute alone, before salaries and failed runs. These figures are external estimates and should be treated as approximate, but the order of magnitude makes the point this series keeps hammering: at frontier scale, a *single* run is a capital expenditure, so every MFU point is six or seven figures. That is why the labs staff dedicated systems teams whose entire job is the arithmetic in this post.

**The "18% cold-start" is real and typical.** The 18%-MFU figure I opened with is not a strawman; it is roughly where an un-tuned PyTorch training loop lands — fp32 by default, no fusion, small batch, a data loader that stalls the GPU, attention that materializes the full $N \times N$ score matrix. Multiple practitioner write-ups and the MFU discussion in the PaLM and Megatron papers put the naive baseline in the ~15–25% range before any systems work. This is the gap the entire field of HPC-for-AI exists to close.

**FlashAttention: a memory-bound win, quantified.** Dao et al.'s FlashAttention reported **2–4× wall-clock speedups** on the attention block and, crucially, reduced attention's memory from $O(N^2)$ to $O(N)$ in sequence length, by never materializing the full score matrix in HBM — it tiles the computation to keep blocks in fast on-chip SRAM. This is the roofline lesson made real: attention's softmax is memory-bound, so the win came entirely from *fewer HBM trips*, exactly as the roofline predicts, not from more FLOP/s. (Source: Dao et al., *FlashAttention*, 2022, and *FlashAttention-2*, 2023.)

**ZeRO/FSDP: fitting models that don't fit.** Microsoft's ZeRO (the basis of DeepSpeed, and the idea behind PyTorch FSDP) showed you can train models far larger than one GPU's memory by sharding the $16\Psi$ of model+optimizer state across data-parallel ranks rather than replicating it — turning the Section 3 budget from a per-GPU cost into a per-cluster one. The reported result was training models with tens of billions of parameters on hardware where the naive replicated approach would OOM immediately, with modest communication overhead. (Source: Rajbhandari et al., *ZeRO*, 2020.) This is the entire premise of the memory-optimization post in Track C.

These five results bracket the series: the **roofline** (FlashAttention) and the **memory budget** (ZeRO) on one GPU, **MFU** as the score (Megatron, PaLM), and the **18% baseline** as the problem. If you remember nothing else: good training MFU is ~40–55%, naive is ~18%, and the gap is pure engineering.

## When to reach for HPC thinking (and when not to)

Every technique in this series is a *cost* — engineering time, code complexity, a new failure mode — and the mark of a senior engineer is knowing when *not* to pay it. Some honest guidance, since the rest of the series is about the techniques themselves.

**Reach for it when** the hardware bill is large enough that a 2× speedup pays for the engineering, when a run is long enough that calendar time matters, when you're memory-bound and a model won't fit, or when you're scaling past one node and seeing efficiency collapse. At frontier scale, HPC thinking is not optional — a 10-point MFU improvement on a large run is millions of dollars and weeks of calendar.

**Don't reach for it when** the model is small, fits comfortably on one GPU, and trains in minutes — the naive loop is fine, and your time is better spent on the data and the model. Don't hand-write a CUDA kernel for an operation that's already compute-bound and near peak; you'll spend a week to gain nothing (the roofline told you so). Don't add tensor or pipeline parallelism to a model that fits on one GPU where DDP already saturates NVLink — you'll add a communication wall that wasn't there and *slow the job down*. Don't chase fp8 if you're not on Hopper-class hardware or your model isn't numerically robust to it. And never optimize before you profile: the most expensive mistake in this field is a week spent making the wrong thing faster. **Measure first. The roofline and the profiler decide; intuition does not.**

There is also a sequencing rule that saves the most time: **fix the cheapest, highest-leverage walls first.** In practice that ordering is almost always (1) turn on bf16 mixed precision, (2) raise the batch size until you're compute-bound or out of memory, (3) fuse or replace the memory-bound kernels (FlashAttention for attention), (4) fix the data loader so it never starves the GPU, and only then (5) scale out across GPUs. Most teams invert this — they reach for more GPUs first, multiplying an 18%-MFU job across eight devices and calling it progress, when a single afternoon on steps 1 through 4 would have delivered the same throughput on one GPU at a fraction of the cost. Scaling out is the *last* lever, not the first, because it adds the hardest wall (communication) on top of the ones you haven't fixed yet.

The decisive rule that runs through every post: *find the bottleneck, prove it with a number, pull the matching lever, re-measure.* If you can't name the wall and quantify it, you are guessing, and guessing at this scale is how money disappears.

## What this series will teach: the four tracks

This is post 1 of 15. The series climbs from the silicon of a single GPU all the way to a multi-node cluster, and every post hits the same three notes: the *science* (why the hardware behaves this way), the *practice* (real runnable code in PyTorch, CUDA, Triton, NCCL, SLURM), and the *proof* (measured before→after on named hardware, scored by MFU). Here's the map.

![tree diagram of the series with four tracks the machine one GPU fast scale out and the cluster across fifteen posts](/imgs/blogs/why-hpc-is-the-bottleneck-for-modern-ai-8.png)

**Track A — The Machine: how a GPU actually computes (4 posts).** Why a GPU is not a fast CPU. [Inside the GPU: SMs, warps, and the SIMT execution model](/blog/machine-learning/high-performance-computing/inside-the-gpu-sms-warps-and-the-simt-execution-model) explains the 32-lane warp, latency hiding, occupancy, and Tensor Cores. [The memory hierarchy: registers, shared memory, and HBM](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm) gives the bandwidth and latency of every tier and why coalesced access and tiling matter. And [The roofline model: compute-bound vs memory-bound](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) turns Section 5's teaser into your everyday tool. This track builds the vocabulary the rest of the series speaks.

**Track B — Making one GPU fast: precision, kernels, profiling (4 posts).** This is where most practical wins live, and where you should start optimizing. It covers numerical formats and mixed precision (fp32, tf32, bf16, fp16, fp8) and *why* bf16 trains stably where fp16 needs loss scaling; the CUDA programming model with a first runnable kernel; kernel fusion and FlashAttention as the answer to the memory wall; and profiling GPU workloads to find the real bottleneck. Master this track and you'll routinely take an 18%-MFU job to 45%+ on a single GPU — before you've touched a second device.

**Track C — Scaling out: many GPUs as one machine (4 posts).** When the model is big enough or you need more throughput, you scale out — and the communication wall arrives. This track maps the parallelism strategies (data, tensor, pipeline, expert, and 3D parallelism) and the decision tree for choosing among them; collective communication and NCCL (the all-reduce math behind Section 6); interconnects (NVLink, NVSwitch, InfiniBand, RDMA) and why topology sets your ceiling; and memory optimization with ZeRO, FSDP, activation checkpointing, and offload — how to fit a 70B model that has no business fitting.

**Track D — The cluster and the playbook (3 posts).** The unglamorous, high-leverage final third: running on a cluster with SLURM, multi-node launch, and the data-pipeline bottleneck most teams ignore; inference at scale, which is a *different* HPC problem (throughput vs latency, continuous batching, the KV-cache, paged attention); and the capstone, [The HPC playbook for AI engineers](/blog/machine-learning/high-performance-computing/the-hpc-playbook-for-ai-engineers), which assembles every lever into one decision framework — profile, find the wall, pick the lever, measure MFU and dollars per result.

Throughout, we cross-link *out* to the existing series instead of re-deriving them: [memory is the real constraint](/blog/machine-learning/edge-ai/memory-is-the-real-constraint) from the edge-AI series for the on-device view of the same memory wall, [Chinchilla compute-optimal scaling](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) for *how much* compute a model should get, and [efficient LLM inference techniques](/blog/machine-learning/large-language-model/efficient-llm-inference-techniques) for the serving-side optimizations this series provides the hardware foundation for. This series owns the *systems and hardware* layer beneath model work; those series own the model and serving layers above it.

## Key takeaways

- **Modern AI is bottlenecked by the machine, not by ideas.** A model is a schedule of arithmetic over bytes moved across a memory hierarchy and a network; your job is to keep expensive chips busy.
- **There are exactly three walls: compute, memory bandwidth, and communication.** Naming which wall you're behind is 80% of the fix, because each has a different set of levers and the wrong lever wastes days.
- **FLOPs grew ~1000× while bandwidth grew ~30×**, so most deep-learning operations are *memory-bound*, not compute-bound — and the gap widens every GPU generation. The cores are a starving giant.
- **The roofline is your 30-second test.** Compute arithmetic intensity $I = \text{FLOPs}/\text{bytes}$; if it's below the ridge ($P/B$, ~156 for an A100) the op is memory-bound and the only lever is fewer HBM trips. Above it, the lever is more FLOP/s.
- **The memory budget decides if you can train at all.** Model + Adam state is $16\Psi$ bytes (a 7B model needs 112 GB before activations), which is why a 7B model doesn't fit on one 80 GB A100 and why sharding exists.
- **MFU is the north-star metric.** $\text{MFU} = \text{achieved FLOP/s} / \text{peak FLOP/s}$. A naive job is ~18%; a well-engineered one is ~40–55%. `nvidia-smi` "utilization" lies; MFU doesn't.
- **Scaling out adds the communication wall, and crossing a node boundary is the biggest efficiency cliff.** NVLink hides the all-reduce; InfiniBand often can't. The interconnect sets your ceiling on useful parallelism.
- **Performance work is a loop, not a one-shot.** Profile → find the wall → pull one lever → measure MFU → repeat. Never optimize from intuition, never change two things at once, always re-measure — and always make one GPU fast before you scale out.

## Further reading

- Narayanan et al., *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM* (2021) — tensor/pipeline/data parallelism and the MFU figures.
- Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* (2020) — the $16\Psi$ budget and how to shard it.
- Dao et al., *FlashAttention* (2022) and *FlashAttention-2* (2023) — the IO-aware, memory-bound win on attention.
- Chowdhery et al., *PaLM* (2022) — a 540B model and an unusually high reported MFU/HFU.
- Grattafiori et al., *The Llama 3 Herd of Models* (2024) — a 16K-GPU H100 run with reported MFU, scaling behavior, and a candid hardware-reliability account.
- Williams, Waterman, Patterson, *Roofline: An Insightful Visual Performance Model* (2009) — the original roofline paper.
- NVIDIA A100 and H100 architecture whitepapers — the peak FLOP/s and HBM bandwidth specs cited throughout.
- The PyTorch FSDP, AMP, and `torch.profiler` documentation; the NVIDIA NCCL and Nsight Systems/Compute docs — the practical toolchain.
- Within this series: [the HPC playbook for AI engineers](/blog/machine-learning/high-performance-computing/the-hpc-playbook-for-ai-engineers) (the capstone), [the roofline model](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound), [the memory hierarchy](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm), and [inside the GPU](/blog/machine-learning/high-performance-computing/inside-the-gpu-sms-warps-and-the-simt-execution-model).

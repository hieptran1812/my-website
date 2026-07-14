---
title: "Metrics that actually matter: utilization, occupancy, MFU, and the numbers that lie"
date: "2026-07-14"
publishDate: "2026-07-14"
description: "A field guide to the GPU metrics you already stare at — why utilization lies, what occupancy and MFU really mean, and which number to trust for each question about your AI service."
tags:
  [
    "performance-engineering",
    "gpu-optimization",
    "mfu",
    "occupancy",
    "profiling",
    "pytorch",
    "latency",
    "throughput",
    "memory",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Performance Engineering"
author: "Hiep Tran"
featured: true
readTime: 36
---

Two teams run the same model on the same A100. Both open a dashboard, point at a green line pinned near the top, and say the same sentence: "GPU is at 95% utilization, we're compute-bound, there's nothing left to squeeze." One of those teams is serving roughly eight times the throughput of the other, on identical hardware, for the same dollar-per-hour. Both are telling the truth about the number on the screen. Only one of them is telling the truth about the machine.

That gap — same headline metric, eightfold difference in real work — is the entire subject of this post. `nvidia-smi` reporting `utilization.gpu` at 100% does not mean the GPU is doing useful work. It means *a* kernel was running. A single tiny kernel, looping forever, launched every few microseconds, will hold that number at 100% while 94% of the silicon sits dark. The metric is not lying by malfunction; it is answering a different question than the one you think you asked.

This is the third post in the *Profiling & Optimizing AI Services* series, and it comes early on purpose. Before you profile anything, before you reach for Nsight or a Chrome trace, you have to know which numbers to believe. Every optimization you ever ship will be scored by a metric, and if that metric can't see the thing you changed, you will "improve" a service that got slower and "regress" one that got faster. The [series intro](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) lays out the four wastes — idle GPU, low occupancy, the bandwidth wall, redundant work — and the profile → hypothesize → fix → measure loop. This post arms the *measure* half of that loop. By the end you'll be able to look at any performance number on your dashboard and answer three questions cold: what does it actually measure, what does it hide, and what is the one question it can be trusted to answer?

![a five row table pairing each metric with what it measures what it hides and the one question it can be trusted to answer](/imgs/blogs/metrics-that-actually-matter-1.webp)

The table above is the whole post compressed into one figure, and we'll spend the rest of it earning every cell. Keep it in the corner of your eye: utilization measures "any kernel running" and is trustworthy only for spotting idle gaps; occupancy measures "active warps over max warps" and hides memory-bound stalls; MFU measures "useful FLOPs over peak FLOPs" and is the only honest end-to-end number; p99 measures the tail and hides both the mean and the cause; reserved memory measures what the allocator holds and hides how much is truly live. Five metrics, five different questions. The mistake that burns money is using one to answer another's question.

## The metric ladder: from coarse and lying to fine and honest

Performance metrics are not a flat menu you pick from by taste. They form a ladder. At the bottom sits the coarsest, most available, most quoted number — `nvidia-smi` utilization — and it is also the one that lies most freely. Each rung up costs more to read, requires a better tool, and tells you more of the truth, until you reach the top rung, Model FLOPs Utilization, which is the only metric that answers "how much of the machine I paid for am I actually using?"

![a vertical ladder of metrics rising from coarse nvidia-smi utilization at the bottom to honest model flops utilization at the top](/imgs/blogs/metrics-that-actually-matter-2.webp)

Read the ladder from the bottom. `nvidia-smi utilization.gpu` is free, always on, and tells you almost nothing about efficiency — only whether the GPU was ever busy. One rung up, SM occupancy tells you whether the streaming multiprocessors had enough warps queued to hide latency, but you need Nsight Compute to read it per-kernel. Above that, SM efficiency (and its cousin, "SM active" from DCGM) tells you what fraction of SMs were actually doing work averaged over time. Near the top, DCGM's profiling fields (SM active, tensor-pipe active, DRAM active) give you a live, low-overhead view of how busy the compute and memory pipes really are. And at the very top sits MFU: achieved FLOP/s divided by the hardware's peak FLOP/s, the single number that collapses every lower rung into one honest fraction.

Here's the mental discipline the ladder enforces, and it's worth stating as a rule: **a metric can only see problems at or below its own rung.** Utilization can see "the GPU went idle." It physically cannot see "the SMs were 6% full" — that information isn't in the number. If your problem lives three rungs up, no amount of staring at the bottom rung will reveal it. This is why teams get stuck: they optimize against utilization, drive it to 100%, and their throughput doesn't move, because the waste was never something utilization could measure.

We'll climb the ladder rung by rung. Utilization first, because it's the one on every dashboard and the one that lies to the most people.

## GPU utilization: the number everyone quotes and nobody understands

Let's define it precisely, because the precise definition is the whole trick. NVIDIA's driver samples the GPU on a fixed interval. `utilization.gpu` is **the percentage of the last sample period during which one or more kernels were executing on the device.** That is the entire definition. It is a duty-cycle over time — was the GPU busy or idle — and it says *nothing whatsoever* about how much of the GPU was busy.

Run the query yourself. This is the command you already know, with the fields that matter:

```bash
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv -l 1
```

On a service that is host-bound — Python can't launch kernels fast enough, so the GPU runs one small kernel, waits, runs another — you'll see something like this:

```console
utilization.gpu [%], utilization.memory [%], memory.used [MiB], memory.total [MiB]
100 %, 3 %, 41216 MiB, 81920 MiB
100 %, 4 %, 41216 MiB, 81920 MiB
100 %, 3 %, 41216 MiB, 81920 MiB
```

Utilization pinned at 100%. Case closed, compute-bound, go home. Except `utilization.memory` — the fraction of time the memory controller was reading or writing — is at 3%, which is your first clue that almost nothing is happening. The GPU is "busy" in the sense that at every sample tick, some kernel was mid-flight. It is *not* busy in the sense of doing a meaningful amount of arithmetic.

Why does a tiny kernel keep utilization at 100%? Because the sampler doesn't count SMs, it counts wall-clock windows. If a 5-microsecond kernel launches, the GPU launches the next one a few microseconds later, and so on, then across any 1-second sample window there is nearly always *a* kernel executing. The duty cycle is ~100%. The SMs are 94% idle the entire time, but the metric has no way to express that. It was never designed to.

### When utilization is still worth reading

Don't throw it away. Utilization is genuinely useful for exactly one thing: **spotting idle gaps.** If `utilization.gpu` drops to 30%, you have a real, unambiguous problem — the GPU is sitting idle 70% of the time, almost always because the host (Python, the dataloader, a synchronization point, an H2D copy) can't feed it. Low utilization is a true negative signal. It's a reliable smoke alarm. The lie is only in the other direction: **high utilization is not evidence of efficiency.** A 30% reading tells you something is wrong; a 100% reading tells you nothing about whether anything is right.

There's a sharper tool one step over. `nvidia-smi dmon` (or `dcgmi dmon`) streams per-second device counters, and DCGM exposes the profiling fields that the plain utilization number hides:

```bash
# DCGM profiling fields that actually see SM and tensor-core occupancy
dcgmi dmon -e 1002,1003,1004,1005 -d 1000
#   1002 = DCGM_FI_PROF_SM_ACTIVE      (fraction of SMs with a warp resident)
#   1003 = DCGM_FI_PROF_SM_OCCUPANCY   (resident warps / max warps)
#   1004 = DCGM_FI_PROF_PIPE_TENSOR_ACTIVE (tensor pipe busy fraction)
#   1005 = DCGM_FI_PROF_DRAM_ACTIVE    (memory pipe busy fraction)
```

```console
GPU   SMACT   SMOCC   TENSO   DRAMA
  0   0.061   0.048   0.004   0.031
  0   0.058   0.045   0.003   0.029
  0   0.063   0.050   0.005   0.033
```

There it is, in numbers. `SMACT` (SM active) is 0.06 — six percent of SM-time had a warp resident. `TENSO` (tensor-pipe active) is 0.004 — the tensor cores, the part of the A100 you actually pay for, are 0.4% busy. The same machine that `utilization.gpu` calls "100%" is, by the metric that counts silicon instead of wall-clock, running at six percent. That is the eightfold gap from the opening, made of exactly this discrepancy.

![two stacked columns showing identical one hundred percent utilization but eight percent flops on the left and forty-eight percent on the right](/imgs/blogs/metrics-that-actually-matter-3.webp)

Both columns of that figure read `util 100%`. The left one — a service dispatching thousands of tiny kernels from Python, the classic host-bound signature — has SM-active at 6% and, when you work the FLOPs, an MFU of 8%. The right one, the same model with its kernels fused and batched so the SMs stay packed, reads the identical `util 100%` but runs SM-active at 71% and MFU at 48%. If your dashboard only shows the top number, these two services are indistinguishable. One of them is quietly burning six of every seven dollars.

#### Worked example: catching a host-bound service that "looks fine"

You inherit an inference service on a single A100 80GB. The on-call dashboard shows GPU utilization steady at 96–100%, and the previous owner left a note: "GPU-bound, needs a bigger box." Throughput is 140 req/s and the finance spreadsheet says you're spending about \$2.20 per GPU-hour of on-demand capacity, so ~\$1,900/month.

You don't buy the bigger box. You run `dcgmi dmon` for thirty seconds and read SM-active at 0.07 and tensor-active at 0.005. That is not a compute-bound service; that is a service where the GPU is idle-in-disguise, waiting on the host between thousands of microscopic kernels. The metric that said "96–100%" was answering "was a kernel running?" — yes, always, one at a time. The metric that mattered, SM-active, said the silicon was 93% empty. The bigger box would have been 93% empty too, at twice the price. The real fix (CUDA graphs and fusion, covered later in the series) belongs to the [kernel-launch-overhead problem](/blog/machine-learning/performance-engineering/the-roofline-for-your-service); the *lesson* is that a single honest counter reframed a hardware purchase as a software bug.

## SM occupancy: necessary, but nowhere near sufficient

Climb to the next rung. To talk about occupancy we need three words of GPU vocabulary, defined once. A **streaming multiprocessor (SM)** is one of the ~108 independent processor blocks on an A100. A **warp** is a group of 32 threads that execute in lockstep — the smallest unit the SM schedules. Each SM can hold a fixed maximum number of resident warps (64 on an A100, since 2048 threads ÷ 32 = 64). **Occupancy** is the ratio you'd guess: the number of warps actually resident on an SM divided by that hardware maximum.

$$\text{occupancy} = \frac{\text{active warps per SM}}{\text{max warps per SM}}$$

Why does occupancy matter at all? Because the GPU hides latency by having *other* work ready. When one warp stalls waiting on a memory load (hundreds of cycles to reach HBM), the SM's scheduler instantly switches to another resident warp and keeps the math pipes fed. With plenty of warps resident, memory latency is hidden behind useful work. With too few, the SM stalls with nothing to do — the pipes go idle waiting for loads to return. Occupancy is your latency-hiding budget.

So what stops you from just having 64 warps resident always? Three independent resource limits, and the *smallest* one wins.

![three boxes for registers shared memory and block size all feeding into active warps per streaming multiprocessor which then divides by the hardware maximum](/imgs/blogs/metrics-that-actually-matter-4.webp)

The three caps fan into one number. First, **registers.** Each SM has a fixed register file (65,536 32-bit registers on an A100). If your kernel's compiler assigned 128 registers per thread, then one SM can hold at most 65,536 ÷ 128 = 512 threads = 16 warps. That alone caps occupancy at 16/64 = 25%, no matter how much else you do. Second, **shared memory.** Each SM has a fixed shared-memory budget (up to 164 KB on an A100). A kernel that requests 100 KB of shared memory per block can fit at most one block per SM, which may be far fewer warps than the register limit allows. Third, **block size.** Occupancy is granted in whole blocks; if your block is 256 threads (8 warps) and the register limit says 16 warps fit, you get two blocks — but if 3 blocks × 8 warps = 24 warps would fit under registers yet only 2 blocks fit under shared memory, you're capped at 16. The realized occupancy is the minimum across all three constraints. In the figure, registers cap you at 16 warps, so occupancy lands at 16/64 = 25% even though the hardware could hold 64.

You do not compute this by hand in production; you read it. Nsight Compute reports both theoretical and achieved occupancy per kernel:

```bash
# Profile one kernel to the metal: occupancy, memory throughput, stall reasons
ncu --set full -k "regex:.*flash.*|.*gemm.*" -c 10 python serve.py
```

Nsight Compute's *Occupancy* section will print the theoretical occupancy (what the launch config allows) and the *achieved* occupancy (what actually happened, averaged over the kernel's life), plus the limiter — "Block Limit Registers", "Block Limit Shared Mem", or "Block Limit Warps" — telling you *which* of the three caps bound you. That limiter field is the single most useful thing on the page: it names the resource to cut. We go deep on reading this in [the Nsight Compute kernel deep-dive](/blog/machine-learning/performance-engineering/nsight-compute-kernel-deep-dive); the GPU-architecture side of warps and SMs is covered in [inside the GPU: SMs, warps, and the SIMT model](/blog/machine-learning/high-performance-computing/inside-the-gpu-sms-warps-and-the-simt-execution-model).

### Achieved vs theoretical, and why high occupancy can still be a waste

Two traps live in occupancy. The first: **theoretical occupancy is a ceiling, not a measurement.** Your launch config might *allow* 75% occupancy, but if the grid is too small (few blocks), if there's a load imbalance, or if the kernel exits early on some warps, the *achieved* occupancy can be far lower. Always read achieved, not theoretical.

The second trap is the important one, and it's why occupancy sits in the *middle* of the ladder, not the top: **high occupancy is necessary but not sufficient for efficiency.** You can have 100% occupancy on a kernel that is completely memory-bound — every warp resident, every warp stalled on HBM, the math pipes idle the whole time because there simply isn't enough bandwidth to feed them. Occupancy says "the SM is full of warps." It does *not* say "those warps are doing arithmetic." A memory-bound elementwise kernel at 90% occupancy is still wasting the compute you paid for; the fix is fusion to cut HBM round-trips, not more warps. Whether your kernel is compute-bound or memory-bound is a roofline question, and it's exactly why [the roofline for your service](/blog/machine-learning/performance-engineering/the-roofline-for-your-service) is the companion to this post: occupancy tells you the SMs are full, the roofline tells you whether "full" is buying you FLOPs or just bandwidth stalls.

So occupancy is a real, honest, useful metric — one rung above utilization because it counts warps instead of wall-clock — but it's a *per-kernel* diagnostic, not a service-level score. To score the whole service, we need the top of the ladder.

## MFU: the only honest end-to-end number

Here is the number that doesn't lie: **Model FLOPs Utilization.** It's disarmingly simple.

$$\text{MFU} = \frac{\text{useful FLOP/s the model actually needs}}{\text{peak FLOP/s the hardware can do}}$$

The numerator counts only the floating-point operations your *model* mathematically requires — the ones you'd count from the architecture on paper, independent of how efficiently you ran them. The denominator is the chip's advertised peak for the precision you're using. Divide, and you get one fraction between 0 and 1 that answers the only question the CFO actually cares about: of the machine you're renting, what fraction of its arithmetic capacity is turned into model progress? Nothing about how many kernels launched, how full the SMs were, or whether utilization read 100% — just useful work over available work.

Because MFU is defined on *model* FLOPs (a paper quantity) rather than *executed* FLOPs, it's hardware-independent and implementation-honest. If you recompute activations to save memory (activation checkpointing), you execute *extra* FLOPs that MFU refuses to count — those get counted by a sibling metric, HFU (Hardware FLOPs Utilization), which we'll separate in a moment. MFU only ever rewards forward progress.

### Deriving the FLOP count for a dense Transformer

To compute MFU you need the numerator, and for a dense decoder-only Transformer there's a clean approximation worth deriving, because the derivation tells you why the constant is what it is.

Consider a model with $N$ parameters. Almost all of those parameters live in matrix multiplies (the attention projections and the MLP). A matrix multiply that uses a weight matrix contributes, per input token, one multiply and one add per weight element it touches — 2 FLOPs per parameter, per token. Summed across the whole network, the **forward pass costs about ${2N}$ FLOPs per token.** For $D$ tokens, that's ${2ND}$ FLOPs. This is the same $2N$-per-token rule you'll see in the scaling-laws literature, and it's accurate to within a few percent for models where the attention score computation is small relative to the weight matmuls.

The backward pass costs about twice the forward pass. Intuitively, backprop computes two gradients at each matmul — the gradient with respect to the input activations and the gradient with respect to the weights — each roughly as expensive as the forward matmul. So backward is about ${4N}$ FLOPs per token, and **training (forward + backward) costs about ${6N}$ FLOPs per token.** For inference, which is forward-only, you're back to ${2N}$ per token (prefill), plus the same ${2N}$ per generated token during decode.

Putting it together, for a training run:

$$\text{MFU}_\text{train} = \frac{6N \cdot (\text{tokens/s})}{P_\text{peak}}$$

and for forward-only inference, replace the 6 with a 2. That's the entire formula. The only inputs are your parameter count, your measured token throughput, and the chip's peak — all three of which you already have or can read in one line.

A word on the peaks, because using the wrong one is the most common MFU mistake. Peak FLOP/s depends on precision and on whether you count the sparsity-doubled marketing number. For dense (non-sparse) matmuls, the honest peaks are: **A100 SXM ≈ 312 bf16/fp16 TFLOP/s** (its dense tensor-core rate; the "624" you sometimes see is the 2:4-sparsity number, which does not apply to a normal matmul), and **H100 SXM ≈ 989 bf16 TFLOP/s** (dense). An L4 is about 242 fp16 TFLOP/s. Always divide by the *dense* peak for the *precision you actually ran*, or your MFU will look artificially low (wrong-precision peak) or impossibly high (sparsity peak).

### A small MFU calculator

You should compute MFU in the same script that benchmarks throughput, so the number is always fresh. Here's a calculator you can drop in:

```python
import torch, time

def measure_mfu(model, sample_batch, n_params, peak_flops,
                mode="train", warmup=10, iters=50):
    """Return achieved TFLOP/s and MFU for a dense transformer.

    n_params : parameter count N (exclude embeddings for a stricter number)
    peak_flops : dense peak for THIS precision (A100 bf16 = 312e12)
    mode : 'train' -> 6N per token, 'infer' -> 2N per token (forward only)
    """
    flops_per_token = 6 * n_params if mode == "train" else 2 * n_params
    tokens_per_batch = sample_batch["input_ids"].numel()

    # Warm up: first iters pay compile/allocator/cudnn-autotune costs.
    for _ in range(warmup):
        _run_step(model, sample_batch, mode)
    torch.cuda.synchronize()               # never time before this

    t0 = time.perf_counter()
    for _ in range(iters):
        _run_step(model, sample_batch, mode)
    torch.cuda.synchronize()               # wait for the GPU to finish
    dt = time.perf_counter() - t0

    tokens_per_s = tokens_per_batch * iters / dt
    achieved = flops_per_token * tokens_per_s
    return {
        "tokens_per_s": tokens_per_s,
        "achieved_tflops": achieved / 1e12,
        "mfu": achieved / peak_flops,
    }
```

The two `torch.cuda.synchronize()` calls are not decoration — they are the difference between a real number and a fantasy. CUDA kernel launches are asynchronous: `model(x)` returns to Python almost immediately, having only *queued* the work. If you stop your timer without synchronizing, you're timing how fast Python enqueued kernels, not how fast the GPU ran them, and you'll report a throughput that's 5–50× too high. Synchronize before starting the clock (so the warmup work is drained) and before stopping it (so the GPU has actually finished). This is the single most common way engineers fool themselves, and it's the foundation the whole series rests on — the dedicated post on [setting up a reproducible benchmark](/blog/machine-learning/performance-engineering/the-roofline-for-your-service) treats warm-up, CUDA events, and locked clocks in full.

Running it on a 7B model on an A100 might print:

```console
{'tokens_per_s': 3021.4, 'achieved_tflops': 126.9, 'mfu': 0.4067}
```

#### Worked example: MFU of a 7B model, A100 vs H100

Take a 7-billion-parameter model, $N = 7 \times 10^9$, training in bf16.

On an **A100** (peak 312 TFLOP/s dense bf16), you measure 3,000 tokens/s/GPU under your training loop. Plug in:

$$\text{MFU} = \frac{6 \cdot (7\times 10^9) \cdot 3000}{312 \times 10^{12}} = \frac{1.26 \times 10^{14}}{3.12 \times 10^{14}} = 0.404$$

40% MFU. You're turning 40% of the A100's arithmetic into training progress. That's a genuinely good number for a dense LLM — more on why in a moment.

Now move the *same model* to an **H100** (peak 989 TFLOP/s dense bf16). Because the H100 has roughly 3× the flops and faster memory, you measure ~9,500 tokens/s/GPU:

$$\text{MFU} = \frac{6 \cdot (7\times 10^9) \cdot 9500}{989 \times 10^{12}} = \frac{3.99 \times 10^{14}}{9.89 \times 10^{14}} = 0.403$$

Also 40% MFU — but at *three times the throughput.* This is exactly why MFU is the honest number: it's dimensionless. The H100 did 3× the work, and MFU correctly reports that both deployments are equally *efficient* (40% of their respective peaks) while raw throughput correctly reports the H100 is 3× *faster*. If you'd only looked at tokens/s, you'd have no idea whether the H100 was well-tuned or leaving half its silicon idle. MFU tells you it's tuned about as well as the A100 was — there's headroom on both, but neither is broken.

### Why 40–50% MFU is often "good"

A first reaction to "40% MFU" is disappointment — surely we should be at 90%? No. For dense LLM training, published state-of-the-art MFU lives in the 40–55% range, and there are structural reasons you never reach 100%:

- **Not everything is a big matmul.** Softmax, layernorm, activation functions, dropout, and the attention score/softmax path are memory-bound or low-arithmetic-intensity work. They consume wall-clock time while contributing few of the FLOPs the tensor cores are rated for. Time spent there is time the tensor cores idle.
- **Communication.** In any multi-GPU run, all-reduce and all-gather steps move gradients and shards over the network. Unless perfectly overlapped with compute (see [overlapping compute and communication](/blog/machine-learning/distributed-training/overlapping-compute-and-communication)), that time counts against MFU.
- **Pipeline bubbles, load imbalance, launch overhead, and warm-up** all eat into the fraction.

So MFU is a fraction with a real ceiling well below 1. The useful mental anchor: below ~25% MFU on a modern GPU, something is likely wrong (host-bound, tiny batches, unfused kernels) and worth investigating. In the 40–55% range you're in good company with published large-model training. Above ~60% is excellent and usually reflects heavy kernel fusion and near-perfect overlap.

### MFU vs HFU: don't double-count recomputation

One precision point that trips people up. **HFU (Hardware FLOPs Utilization)** counts *every* FLOP the hardware executed, including the redundant ones. The most common source of redundant FLOPs is activation checkpointing (gradient checkpointing): to save memory, you throw away activations in the forward pass and recompute them in the backward pass. Those recomputed forward passes are real FLOPs the hardware did — HFU counts them, MFU does not. So a checkpointed run has HFU higher than MFU (it did extra work), and the gap between them is exactly your recomputation overhead. MFU answers "how efficiently did I make model progress?"; HFU answers "how busy were the tensor cores?" For a serving or training scorecard, MFU is the one to publish, because it can't be gamed by doing more redundant work.

## Latency and throughput: the mean hides the tail

MFU is the efficiency scorecard, but a service also has a *latency* contract and a *throughput* target, and those two are in tension. Let's define them cleanly and then connect them with a law.

**Throughput** is work per unit time: requests/second or tokens/second. **Latency** is time per request. The trap in latency is summarizing it with a mean. Consider a service where 99 requests take 40 ms and one takes 2,000 ms (a periodic garbage-collection pause, a recompilation, a synchronizing checkpoint). The mean is (99·40 + 2000)/100 ≈ 60 ms, which sounds fine. But one in a hundred of your users waited two seconds, and if that user is a downstream service with a 200 ms timeout, it's now retrying, doubling your load, cascading. The mean *hid* the tail. This is why every latency SLO is written on percentiles, not means:

| Statistic | What it captures | What it hides | Use it for |
|---|---|---|---|
| mean | average time | the tail entirely; one 2 s stall vanishes into the average | rough capacity math only |
| p50 (median) | the typical request | the worst 50% | "what most users feel" |
| p99 | the 99th-percentile request | the mean; and *why* it's slow | SLO compliance, tail-stall hunting |
| p99.9 | the 1-in-1000 worst | everything below it | large-fanout services where one slow leaf stalls the whole response |

p99 is the number your SLO is written against, because in a service with fan-out — one user request that touches ten backends — the user's latency is the *max* of the ten, so even a 1% tail on each backend means roughly 1 − 0.99¹⁰ ≈ 10% of user requests hit a slow leaf. Tails compound. That's why you chase p99 and p99.9, not the mean.

### Little's Law ties concurrency, throughput, and latency

Here is the law that connects the two axes, and it's exact for any stable system, no assumptions about distributions:

$$L = \lambda W$$

$L$ is the average number of requests in the system (concurrency, in-flight requests), $\lambda$ is the throughput (arrival = completion rate in steady state), and $W$ is the average latency each request spends in the system. Three quantities, one equation, and knowing any two gives you the third.

Little's Law is the tool that keeps you honest about the latency-throughput tradeoff. You cannot independently pick all three. If your latency $W$ is fixed by your model and you want more throughput $\lambda$, you *must* increase concurrency $L$ — more requests in flight at once — which on a GPU means bigger batches. And bigger batches raise $W$. Round and round.

![a left to right sequence of growing batch sizes where throughput rises then flattens while the p99 latency keeps climbing](/imgs/blogs/metrics-that-actually-matter-5.webp)

The timeline shows the tradeoff as batch size grows on a fixed model. At batch 1, p99 is a snappy 20 ms but throughput is only 50 req/s — the GPU is mostly idle between requests, each one paying full latency for a nearly empty batch. Crank to batch 8 and 16 and throughput climbs steeply (320, then 520 req/s) while p99 creeps up (35, then 55 ms) — this is the sweet spot, where each larger batch amortizes fixed per-step overhead across more requests. Push to batch 64 and throughput is 780 req/s but p99 has grown to 95 ms; by batch 128, throughput has essentially saturated at 810 req/s (the SMs are full, there's no more compute to extract) while p99 has ballooned to 190 ms. Past the saturation knee you're paying pure latency for zero throughput gain. This is precisely the curve that [continuous batching and paged attention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention) exists to bend — by batching at the token level instead of the request level, so a request doesn't wait for its whole batch to finish.

#### Worked example: sizing concurrency with Little's Law

Your model's median serving latency is $W = 50$ ms = 0.05 s. Product wants $\lambda = 200$ req/s sustained. How many requests must be in flight?

$$L = \lambda W = 200 \times 0.05 = 10$$

You need 10 concurrent requests in flight, on average, to hit the target. If your batching scheduler only ever keeps 4 in flight, the math says you *cannot* reach 200 req/s no matter how fast the GPU is — you'll top out at $\lambda = L / W = 4 / 0.05 = 80$ req/s and the GPU will read low utilization because it's starved for concurrency. The fix isn't a faster kernel; it's a scheduler that holds more requests in flight. Little's Law told you the bottleneck was concurrency, not compute, before you profiled a single kernel. And the direction of causation matters: raising $L$ to 10 will raise $W$ above 50 ms (bigger effective batches), so you iterate — measure the new $W$, recompute the $L$ you need, and settle at the operating point where the p99 still meets your SLO.

## Memory: allocated, reserved, and why 60% used still OOMs

The last rung is memory, and it has its own special lie: the number that says "60% used" while your next allocation throws `CUDA out of memory`. Understanding it requires separating two quantities PyTorch tracks, because the caching allocator sits between your tensors and the driver.

When you free a tensor, PyTorch does *not* immediately hand the memory back to the CUDA driver. Returning memory to the driver is slow (it synchronizes), so PyTorch's caching allocator keeps the block in a pool to hand out again fast. This means two different numbers:

- **`torch.cuda.memory_allocated()`** — bytes currently backing live tensors. What you're *actually using*.
- **`torch.cuda.memory_reserved()`** — bytes the allocator has taken from the driver and is holding, live plus cached-and-free. What the *driver thinks you own*.

Reserved is always ≥ allocated, often by a lot. The gap is memory the allocator is caching for reuse — invisible to your tensors, unavailable to any other process, and counted by `nvidia-smi` as "used."

```python
import torch

def mem_report(tag=""):
    a = torch.cuda.memory_allocated() / 1e9
    r = torch.cuda.memory_reserved()  / 1e9
    p = torch.cuda.max_memory_allocated() / 1e9   # peak live, since last reset
    print(f"{tag:14s} allocated {a:6.2f} GB | reserved {r:6.2f} GB | peak {p:6.2f} GB")

mem_report("after load")
out = model(batch)            # activations spike allocated, then free
mem_report("after forward")
torch.cuda.reset_peak_memory_stats()
```

```console
after load     allocated  14.05 GB | reserved  15.20 GB | peak  14.05 GB
after forward  allocated  14.05 GB | reserved  48.30 GB | peak  36.11 GB
```

Read that second line carefully. After the forward pass, live tensors are back down to 14 GB (activations were freed), but reserved is 48 GB and the *peak* live was 36 GB. The allocator grabbed 48 GB from the driver to service the 36 GB transient spike, and it's holding all 48 even though only 14 is live now. `nvidia-smi` will report ~48 GB used. If someone reads "48 of 80 = 60%" and launches a second process expecting 32 GB free, they get about 32 GB minus fragmentation — and may OOM.

![a two row six cell layout of gpu memory split into live weights activations and kv cache above cached freed fragmented and truly free blocks below](/imgs/blogs/metrics-that-actually-matter-6.webp)

The figure breaks an 80 GB A100 into where the bytes really are. The top row is *live*, counted by `memory_allocated`: 14 GB of weights, 22 GB of activations, 12 GB of KV cache — 48 GB genuinely in use. The bottom row is the part `memory_allocated` doesn't show but the driver counts: 18 GB the allocator freed-but-cached, 9 GB lost to fragmentation (free blocks too small and scattered to satisfy a large contiguous request), and only 5 GB truly free. Reserved is ~75 GB, live is 48 GB, and the request that OOMs you needs a *contiguous* 8 GB that the 9 GB of scattered fragments can't supply. You are "60% used" and out of memory simultaneously.

#### Worked example: the OOM at 60% used

A training job runs fine for hours, then dies with `CUDA out of memory. Tried to allocate 8.00 GiB` — and the log line right above it says `reserved 49.2 GB, allocated 30.1 GB` on an 80 GB card. Thirty of eighty is 38% live; 49 of 80 is 61% reserved. Plenty of room, by both readings. So why?

Because the 8 GB allocation must be *contiguous*, and after hours of allocating and freeing tensors of varying sizes, the ~19 GB gap between reserved and the driver's view is chopped into fragments — a few hundred MB here, a gigabyte there — none of them 8 GB wide. The allocator can't coalesce them without returning memory to the driver and re-requesting (a slow sync it avoids), so it asks the driver for a fresh 8 GB block, the driver has only ~31 GB unreserved but fragmented at its level too, and the allocation fails. The metric ("61% used") was true and useless. The fix is to attack fragmentation directly — this is exactly what `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` addresses, and the full mechanism is the subject of [the CUDA caching allocator](/blog/machine-learning/performance-engineering/the-cuda-caching-allocator). For this post the lesson is narrower: **track reserved and peak, not just allocated, and never size headroom off `nvidia-smi`'s single "used" number.**

Here's the comparison to keep:

| Reading | API | Counts | Trust it for | The lie |
|---|---|---|---|---|
| allocated | `memory_allocated()` | live tensor bytes | current working set | ignores what the allocator holds |
| reserved | `memory_reserved()` | live + cached blocks | true footprint vs the card | can look alarmingly high and be fine |
| peak allocated | `max_memory_allocated()` | high-water live bytes | sizing batch/seq to fit | resets; must snapshot per phase |
| nvidia-smi "used" | driver query | reserved + other procs | is the card physically full | hides fragmentation; can OOM below it |

## The named-hardware scorecard: one service, three stories

Let's put utilization, SM-active, MFU, and throughput side by side on the *same service* — the host-bound inference service from earlier, before and after fusing kernels and enabling CUDA graphs — on a single A100 80GB. This is the table that should live on your review doc, because it's the one that shows the same box telling three different stories depending on which metric you read.

| Metric | Before (tiny-kernel loop) | After (fused + graphed) | What changed |
|---|---|---|---|
| `nvidia-smi` util | 100% | 100% | nothing — it lied both times |
| DCGM SM-active | 6% | 71% | the SMs went from empty to full |
| Tensor-pipe active | 0.4% | 39% | the tensor cores started working |
| MFU (forward) | 8% | 48% | 6× more useful FLOPs per second |
| Throughput | 140 req/s | 880 req/s | 6.3× |
| p99 latency | 210 ms | 48 ms | tail collapsed with the host stalls |
| Peak memory | 31 GB | 33 GB | graphs pin static buffers; small cost |
| Cost per 1M req | ~\$4.30 | ~\$0.68 | 6.3× cheaper for the same output |

The row that matters is the first one: `nvidia-smi util` read 100% *before and after.* If that were your only metric, you'd have concluded the "before" service was already maxed out and shipped a bigger, equally-empty box. Every honest metric — SM-active, tensor-active, MFU, throughput, p99, cost — tells the real story of a 6× win. The number on the wall-mounted dashboard was the one number blind to it.

## Case studies: published MFU, and what "good" really looks like

MFU isn't an internal curiosity — the large-model training reports publish it precisely because it's the one comparable, hardware-independent efficiency number. Treat all of these as approximate (rounding, differing FLOP-counting conventions, and hardware generations vary), but the pattern is robust.

| System | Params | Hardware | Reported MFU | Source (approx.) |
|---|---|---|---|---|
| GPT-3 175B | 175B | V100 | ~21% MFU | quoted in the PaLM paper's comparison |
| Megatron-LM GPT | 175B–1T | A100 | up to ~52% of peak | Narayanan et al., 2021 |
| PaLM 540B | 540B | TPU v4 | 46.2% MFU (57.8% HFU) | Chowdhery et al., 2022 |
| LLaMA 65B | 65B | A100 80GB | ~43% MFU (approx.) | Touvron et al., 2023 (from ~380 tok/s/GPU) |

A few things to read out of this. First, the PaLM paper is where MFU was crystallized as *the* metric, and it deliberately reported both MFU and HFU (46.2% vs 57.8%) so the ~11-point gap — the cost of rematerialization — was visible rather than hidden. Second, GPT-3's ~21% and PaLM's ~46% are *both* real, well-engineered runs; the difference is largely parallelism strategy, kernel fusion, and generation of hardware, not one team being twice as good as the other. Third, note the ceiling: nobody reports 85%. The 40–55% band is what a heavily optimized dense LLM training run looks like, which is why 40% on your own 7B run is a number to be pleased with, not ashamed of.

For inference specifically, published numbers are noisier because they depend heavily on batch size, sequence length, and whether you count prefill or decode — decode is famously memory-bound and low-MFU because each step does a tiny matmul against a large KV cache. If you want to ground your own inference numbers against a reference sweep on named hardware, the [LLM GPU benchmark](/blog/machine-learning/mlops/llm-gpu-benchmark) post walks a throughput/latency/MFU comparison across GPU types.

## torch.profiler: reading time when you can't read FLOPs directly

MFU needs a throughput number; sometimes you also need to know *where* the time went, and that's `torch.profiler`. It won't hand you occupancy (that's Nsight Compute's job) but it will attribute wall-clock and CUDA time to operators, which is how you find the kernel eating your MFU. Here's the idiomatic schedule-based invocation:

```python
import torch
from torch.profiler import profile, schedule, ProfilerActivity, record_function

sched = schedule(wait=1, warmup=1, active=3, repeat=1)  # skip 1, warm 1, record 3

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=sched,
    record_shapes=True,
    profile_memory=True,
    with_stack=False,
) as prof:
    for step, batch in enumerate(loader):
        with record_function("serve_step"):
            out = model(batch)
        prof.step()                       # advances the wait/warmup/active state

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=8))
```

The `schedule` matters: `wait=1` skips the first step (cold caches, allocator warm-up), `warmup=1` lets one step run without recording so autotuning settles, and `active=3` records three clean steps. Sorting by `cuda_time_total` puts the kernels that dominate GPU time on top. The output for a host-bound service is diagnostic on its own:

```console
-------------------------------  ------------  ------------  ------------  ------------
Name                               Self CUDA    CUDA total   # of Calls    CPU total
-------------------------------  ------------  ------------  ------------  ------------
aten::mm                            18.204ms      18.204ms         1536      42.118ms
aten::add                            6.881ms       6.881ms         1536      21.004ms
aten::native_layer_norm              4.552ms       9.221ms          768      19.887ms
aten::mul                            3.017ms       3.017ms         2304      14.552ms
void elementwise_kernel<...>         2.900ms       2.900ms         3072       9.100ms
-------------------------------  ------------  ------------  ------------  ------------
Self CUDA time total: 39.9ms
CPU time total: 118.4ms
```

Two numbers at the bottom tell the story before you read a single row: `Self CUDA time total: 39.9ms` versus `CPU time total: 118.4ms`. The CPU spent three times as long as the GPU. That is the host-bound signature — Python is the bottleneck, launching thousands of tiny kernels (`# of Calls` in the thousands, each op a few microseconds of CUDA time), and the GPU is waiting on the host between them. No MFU calculation needed; the ratio of CPU-total to CUDA-total already screams "you are launch-bound." That's the profile-read half of the loop; the fix (fewer, bigger kernels) is the fix half. And crucially, while this was happening, `nvidia-smi` read 100%.

## Which metric answers which question

You now have five metrics and a rule for each. The failure mode is reaching for the wrong one under pressure — staring at utilization during an OOM, or at p99 when the real problem is the GPU sitting idle. So route the symptom to the metric.

![a decision tree that branches from a single question through three symptom categories down to the single metric each one should read](/imgs/blogs/metrics-that-actually-matter-7.webp)

The tree routes three classes of symptom to their metric. If throughput is low or the GPU seems underused, first ask whether there are *idle gaps* — read `utilization` (and if it's low, you're host-bound or starved); if utilization is high but throughput is still bad, the SMs are busy-but-not-productive, so read `MFU` (and its neighbor, SM-active, to confirm the silicon is genuinely full of arithmetic). If a *latency SLO* is at risk, read `p99` and p99.9 — the mean and even p50 can look fine while the tail violates the contract. If the service is *crashing or OOMing*, read `reserved` and peak memory, not `nvidia-smi`'s "used", because fragmentation lives in the gap. Each leaf is a single metric, and the path to it is the symptom you started with. That's the whole discipline: never read a number that structurally cannot see your problem.

### The honest dashboard

Translate the tree into the board you actually build in Grafana. Most dashboards show one giant utilization gauge and nothing else, which is worse than useless because it invites the exact mistake this post is about. An honest board carries one panel per *question*:

- **Idle detection:** `utilization.gpu` *and* DCGM SM-active on the same axis. When they diverge (util high, SM-active low), you're host-bound — the divergence itself is the alert.
- **Efficiency:** rolling MFU, computed from throughput and your known parameter count. One line, one honest fraction, the number you defend in a review.
- **SLO:** p50, p99, p99.9 latency as separate lines — never a single "avg latency" panel. The gap between p50 and p99 is your tail-risk gauge.
- **Headroom:** `memory_reserved` and peak `memory_allocated` against card capacity, not `nvidia-smi` "used." This is the panel that predicts OOMs before they page you.
- **Cost:** throughput per dollar-hour (or per GPU), so an efficiency regression shows up as a cost line moving the wrong way.

Five panels, five questions, zero metrics pretending to answer a question they can't see. If you build only one thing from this post, build the util-vs-SM-active divergence panel — it's the single most revealing plot for an AI service, and it turns the opening story ("both at 95%") into an alert instead of a mystery.

## When to trust each metric — and when not to

Every metric is a tool with a blast radius. Here's the decisive version.

**Reach for utilization when** you suspect the GPU is *idle* — low utilization is a trustworthy, unambiguous "the host isn't feeding the device" signal. **Don't trust utilization when** it's high; high utilization is not evidence of anything except that kernels launched. Never conclude "compute-bound" from utilization alone.

**Reach for occupancy when** you're tuning a *specific kernel* and Nsight Compute says it's launch-config-limited — the limiter field names the resource to cut. **Don't chase occupancy when** the kernel is memory-bound; more warps won't help a kernel that's already bandwidth-saturated, and pushing occupancy can even hurt by increasing register pressure. Occupancy is necessary, not sufficient.

**Reach for MFU when** you want the one honest, comparable efficiency score for a service or a training run — it's the number to put in a review and to track over time. **Don't over-index on MFU when** you're memory-bound by design (LLM decode, tiny-batch inference); low MFU there is structural, not a bug, and the right lever is batching or a KV-cache optimization, not "make the matmul bigger."

**Reach for p99 when** you have a *latency SLO* — it's what the contract is written against and what fan-out amplifies. **Don't chase p99 before** you know it's a *stall* (a periodic GC/sync/recompile spike) and not just *load* (you're simply past your throughput knee, and the tail is the batching queue). Those have opposite fixes: a stall wants a code change; load wants more capacity or smaller batches.

**Reach for reserved/peak memory when** you're sizing batch and sequence length or hunting an OOM. **Don't size headroom off `nvidia-smi` "used"** — it hides the fragmentation that will OOM you below the number it shows.

## Key takeaways

- **`nvidia-smi utilization.gpu` measures "was a kernel running," not "how full the GPU was."** A single tiny kernel looping holds it at 100%. It's a reliable *idle* alarm and nothing more.
- **Climb the ladder.** Utilization → occupancy → SM-active → MFU, coarse-and-lying to fine-and-honest. A metric can only see problems at or below its own rung.
- **Occupancy is necessary, not sufficient.** Active-warps ÷ max-warps is bounded by the smallest of registers, shared memory, and block size. High occupancy on a memory-bound kernel is still wasted compute.
- **MFU is the only honest end-to-end number:** ${6N}$ FLOPs/token for training (${2N}$ for inference forward), times tokens/s, over the *dense* peak for *your* precision. 40–55% is state-of-the-art for dense LLM training; below ~25% something is wrong.
- **Always `torch.cuda.synchronize()` before and after timing.** Launches are async; timing without a sync measures Python, not the GPU, and inflates throughput 5–50×.
- **Latency lives in percentiles, not the mean.** p99 and p99.9 are what SLOs and fan-out amplify; the mean hides the two-second stall that's retrying your whole system.
- **Little's Law, $L = \lambda W$, is exact.** You can't pick concurrency, throughput, and latency independently — fix two and the third is determined. It often names your bottleneck before you profile.
- **Track reserved and peak memory, not just allocated.** The caching allocator holds more than is live; fragmentation OOMs you below the "used" number. Never size headroom off `nvidia-smi`.
- **Build the honest dashboard:** util-vs-SM-active divergence for idle, MFU for efficiency, p50/p99/p99.9 for the SLO, reserved/peak for headroom, throughput-per-dollar for cost. Five questions, five panels.

## Further reading

- [Why your AI service wastes CPU and GPU](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) — the series intro: the four wastes and the profile → fix → measure loop this post scores.
- [The roofline for your service](/blog/machine-learning/performance-engineering/the-roofline-for-your-service) — compute-bound vs memory-bound, and why occupancy alone can't tell you which.
- [Nsight Compute kernel deep-dive](/blog/machine-learning/performance-engineering/nsight-compute-kernel-deep-dive) — where you actually read achieved occupancy and the limiter that names the resource to cut.
- [The CUDA caching allocator](/blog/machine-learning/performance-engineering/the-cuda-caching-allocator) — allocated vs reserved, fragmentation, and the OOM-at-60% mechanism in full.
- [The performance engineering playbook](/blog/machine-learning/performance-engineering/the-performance-engineering-playbook) — the capstone decision tree tying every metric and fix in the series together.
- [Inside the GPU: SMs, warps, and the SIMT model](/blog/machine-learning/high-performance-computing/inside-the-gpu-sms-warps-and-the-simt-execution-model) — the architecture behind occupancy.
- [LLM GPU benchmark](/blog/machine-learning/mlops/llm-gpu-benchmark) — throughput, latency, and MFU compared across named GPU types.
- [PyTorch Profiler tutorial](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) and the [NVIDIA DCGM profiling fields reference](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html) — the primary sources for the tools above.

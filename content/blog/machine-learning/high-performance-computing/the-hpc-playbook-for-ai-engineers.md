---
title: "The HPC playbook for AI engineers: profile, find the wall, pick the lever, measure"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "The capstone decision framework that ties the whole series together — a single loop, a lever menu mapped to the three walls, a cost model in dollars, and MFU as the one number you chase from 18 percent to 50."
tags:
  [
    "high-performance-computing",
    "gpu",
    "mfu",
    "roofline",
    "distributed-training",
    "cost-model",
    "flash-attention",
    "fsdp",
    "deep-learning",
    "ml-systems",
    "transformers",
  ]
category: "machine-learning"
subcategory: "High Performance Computing"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/the-hpc-playbook-for-ai-engineers-1.png"
---

The first post in this series opened with a team whose eight A100s were training a 7-billion-parameter model at **18% MFU** — Model FLOPs Utilization, the fraction of the hardware's peak arithmetic the job actually used. A quarter of a million dollars of silicon, idle 82% of the time, and nobody had ever measured the one number that would have told them. We have now spent fourteen posts taking that machine apart: the warps and Tensor Cores inside a single SM, the memory hierarchy from registers down to HBM, the roofline that tells you whether an op is starved for FLOPs or for bandwidth, mixed precision and kernel fusion and FlashAttention on one GPU, then parallelism and NCCL collectives and NVLink and InfiniBand and ZeRO-FSDP across many GPUs, and finally the cluster and the serving box. This is the post where it all becomes one thing you can *do* on a Monday morning.

Because here is the trap. Each technique in this series is genuinely powerful, and each one has a section in the docs that promises a speedup. So the natural failure mode of a smart engineer is to reach for the *technique they read about most recently* — to add tensor parallelism because a blog post raved about it, to switch to fp8 because Hopper supports it, to enable activation checkpointing because someone said it saves memory. That is cargo-cult optimization, and it is how you end up *slower* than you started: tensor parallelism over PCIe is a disaster, fp8 on a memory-bound layer buys nothing, and checkpointing on a job that wasn't memory-constrained just burns 30% more compute for no reason. The levers are not interchangeable. **Each lever fixes exactly one wall, and if you apply it to the wrong wall you pay its cost and get none of its benefit.**

So the playbook is not a list of techniques. It is a *loop* and a *map*. The loop is four steps: **profile** to see where the time actually went, **find the wall** you are stuck behind (compute, memory bandwidth, or communication), **pick the lever** that targets that specific wall, and **measure** MFU and dollars per result to confirm the win — then repeat until MFU stops climbing. The map is which lever goes with which wall. Get the loop and the map into your hands and you stop guessing. You walk up to any slow training or serving job, run two profiler commands, read off the wall, pick from a short menu, and move the number. That is what a principal ML-systems engineer actually does, and it is almost mechanical once you see the structure.

![diagram of the master HPC loop profile then find the wall then pick the lever then measure MFU and dollars then repeat](/imgs/blogs/the-hpc-playbook-for-ai-engineers-1.png)

This post is the hub. It restates the unifying science — the one MFU formula and the one cost model that every prior post's lever moves a term of — gives you five runnable tools (an MFU calculator, a cost estimator, a decision-tree checklist, profiling one-liners, and a parallelism sizing calculator), walks a 7B run from 18% to 50% MFU lever by lever with the dollars at each step, and links back to the post that derives each lever in full. By the end you will have an operating manual, not a reading list. Let's build it.

## 1. The two equations that unify the whole series

Everything in this series — every wall, every lever, every result — collapses into two equations. If you internalize only these and nothing else from this post, you will be ahead of most teams. The first measures efficiency. The second turns efficiency into money.

### MFU: the one number that scores every optimization

Model FLOPs Utilization is the fraction of your GPU's peak floating-point throughput that your job actually achieves on *useful* model arithmetic. The numerator is the FLOPs your model genuinely needs; the denominator is what the chip could do at peak. The standard accounting, from the PaLM paper, is that a dense Transformer with $N$ parameters costs about $6N$ FLOPs per token of training — roughly $2N$ for the forward pass and $4N$ for the backward pass, which does about twice the work of the forward. So if your run processes $T$ tokens per second on hardware with peak throughput $P$ FLOP/s:

$$\text{MFU} = \frac{6 N \cdot T}{P}$$

That is the whole metric. An A100 in bf16 has a peak $P \approx 312$ TFLOP/s; an H100 SXM has $P \approx 989$ TFLOP/s (these are the dense, non-sparse tensor-core numbers from NVIDIA's datasheets). Plug in your parameter count $N$ and your measured tokens-per-second $T$ and you get a percentage. There is no judgment in it, no vibes — it is achieved-over-peak, and it falls between roughly 0.15 and 0.6 for real training runs. Below 0.2 you are leaving most of the machine on the floor; 0.4 to 0.5 is a well-tuned dense run; above 0.55 you are doing something special. The whole [intro to this series](/blog/machine-learning/high-performance-computing/why-hpc-is-the-bottleneck-for-modern-ai) is built around making this number the north star, and every post since has been a way to move it.

Why $6N$ and not the exact FLOP count of every LayerNorm and softmax? Because for a Transformer the matmuls dominate the FLOP count so thoroughly — the attention and MLP projections are $O(N)$ per token while the elementwise ops are a rounding error in *FLOPs* (though emphatically not in *time*, which is the whole point of the memory wall) — that $6N$ is accurate to within a few percent for models above a billion parameters. Some teams add the attention $O(s^2)$ term for long sequences; the canonical "$6N$" MFU ignores it, which makes the metric slightly pessimistic for long-context runs and is fine as a north star.

Let's actually derive the $6N$, because the factor is the load-bearing assumption of the whole metric and it's worth seeing where it comes from rather than taking it on faith. Consider a single linear layer that multiplies an input vector by a weight matrix with $N_\ell$ parameters. The forward pass is a matrix-vector product: for each of the $N_\ell$ weights you do one multiply and one add, so $2 N_\ell$ FLOPs per token. The backward pass computes two things — the gradient with respect to the layer's input (so the gradient can flow to earlier layers) and the gradient with respect to the weights (so the optimizer can update them) — and each of those is another matrix multiply of the same size, giving roughly $2 \times 2 N_\ell = 4 N_\ell$ FLOPs. Sum forward and backward: $2 N_\ell + 4 N_\ell = 6 N_\ell$ FLOPs per token for that layer. Sum over all layers and the per-token training cost is $6N$ where $N$ is the total parameter count. That is the entire derivation, and it's why the factor is exactly six and not five or seven — it's two for the forward, two for the input-gradient, two for the weight-gradient. There is a subtlety: this counts a multiply-accumulate as two FLOPs (one multiply, one add), which is the standard convention and matches how vendors quote peak FLOP/s, so the numerator and denominator use the same accounting and the ratio is honest.

There's a second flavor of utilization worth knowing so you don't get confused reading other people's numbers: **HFU**, Hardware FLOPs Utilization, which counts *all* the FLOPs the hardware executed, including the recomputation that activation checkpointing does. MFU counts only the *useful* model FLOPs ($6N$); HFU counts the extra recompute too. So if you enable activation checkpointing, HFU can be higher than MFU because the hardware is genuinely doing more arithmetic — it's just arithmetic you're doing twice to save memory. MFU is the number you want for cost, because cost is about *useful work per dollar*, and the recompute is overhead, not product. When a paper quotes "52% MFU" they almost always mean the useful-FLOPs version; when a vendor quotes a flattering "70% utilization" check whether they're counting recompute or even sparsity-doubled peak. Always ask: useful FLOPs over *dense* peak? That's the number that maps to money.

### The cost model: from MFU to dollars

MFU is a percentage, and percentages don't pay the cloud bill. So invert the equation. If you must process $T_\text{total}$ tokens, the useful work is $6 N \cdot T_\text{total}$ FLOPs, and the *wall-clock* GPU-hours it takes is that work divided by the *effective* throughput you actually sustain, which is $P \cdot \text{MFU}$:

$$\text{GPU-hours} = \frac{6 N \cdot T_\text{total}}{P \cdot \text{MFU} \cdot 3600}$$

and then the dollar cost is just GPU-hours times the rental rate:

$$\text{Cost} = \text{GPU-hours} \cdot (\text{\$ per GPU-hour})$$

Stare at that for a second, because it is the most important pair of equations in production ML systems. **MFU is in the denominator of the cost.** Double your MFU and you halve your GPU-hours and halve your dollar cost for the exact same model trained on the exact same tokens. The 18%-to-50% improvement that this post walks through is not a 32-point cosmetic gain — it is a **2.8× reduction in the training bill**. On a frontier run that is the difference between \$1M and \$360k. On your team's weekly experiments it is the difference between shipping on Friday and shipping the following Wednesday. Every lever in this series is, in the end, a way to push MFU up and therefore push that dollar number down.

A useful sanity check before you trust any of this: the cost equation has to be *dimensionally consistent*, and walking the units is the fastest way to catch an off-by-1000 error (which, in a field where everything is measured in tera- and peta- and you're constantly converting, happens constantly). The numerator $6 N \cdot T_\text{total}$ is FLOPs — dimensionless count times dimensionless count gives a pure FLOP count. The denominator $P \cdot \text{MFU} \cdot 3600$ is (FLOP/s) × (dimensionless) × (s/hour), which is FLOP/hour. Dividing FLOPs by FLOP/hour gives hours — GPU-hours. Then multiplying by dollars-per-GPU-hour gives dollars. Every time you compute one of these numbers, do that unit walk in your head; it catches the mistake where someone wrote peak FLOP/s in teraflops (e.g. `312` instead of `312e12`) and got an answer off by twelve orders of magnitude. The discipline of carrying units is not pedantry in HPC — it is the single most common source of "wait, that can't be right" moments, and the cure is free.

That is the unifying science. Profiling tells you which wall is holding MFU down; the lever you pick raises MFU; the cost equation converts the raised MFU into saved money. Notice what the two equations buy you politically as well as technically: when you walk into a planning meeting and say "this run will cost \$1.18M at our current efficiency, or \$425k if we spend a week on it," you have turned a vague engineering preference ("we should optimize") into a budget decision with a number attached, and budget decisions get made. MFU is the bridge between the silicon and the spreadsheet. Now let's make the walls and levers concrete.

## 2. The three walls and the levers that fix them

There are exactly three reasons your GPU is not doing useful math right now, and the profiler will tell you which one you're hitting. This is the central map of the whole series, and it is worth committing to memory because it converts a vague "my job is slow" into a precise, actionable diagnosis.

![diagram showing a profiler trace branching into three walls memory compute and communication each pointing to its own lever](/imgs/blogs/the-hpc-playbook-for-ai-engineers-2.png)

**The memory-bandwidth wall.** The GPU's compute units are idle because they are waiting for data to arrive from HBM. The diagnostic is *arithmetic intensity* $I$ — the FLOPs you do per byte you move — being lower than the GPU's ridge point. An A100 does 312 TFLOP/s of bf16 over 2.0 TB/s of HBM, so its ridge point is $312\text{e}12 / 2.0\text{e}12 \approx 156$ FLOP/byte; any op with $I$ below that is bandwidth-starved no matter how fast the cores are. LayerNorm, softmax, dropout, GELU, the optimizer step, the residual adds — almost every non-matmul op in a Transformer lives here. This is the single most common wall in deep learning, and the entire [roofline model post](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) is about reading it off. The levers that fix it: **kernel fusion** (do more FLOPs per HBM round-trip), **better precision** (move half the bytes in bf16 vs fp32), and **raising arithmetic intensity** by restructuring the computation — which is exactly what [FlashAttention](/blog/machine-learning/high-performance-computing/kernel-fusion-and-flashattention-beating-the-memory-wall) does by never writing the $N \times N$ attention matrix to HBM at all.

**The compute wall.** Now the cores *are* busy — you've fed them well, intensity is above the ridge — but the math itself is the limit. This is the *good* wall to be stuck behind, because it means you are using the silicon. Here the only way forward is faster arithmetic: **Tensor Cores** instead of CUDA cores (a 16× rate increase on matmul for the formats they support), **better kernels** (a well-written tiled GEMM versus a naive one), and dropping into [CUDA or Triton](/blog/machine-learning/high-performance-computing/cuda-programming-for-ai-engineers-threads-blocks-and-a-first-kernel) when the framework's kernels leave performance on the table. The physics of why a GPU is a throughput machine that loves dense regular matmul is in the [inside-the-GPU post](/blog/machine-learning/high-performance-computing/inside-the-gpu-sms-warps-and-the-simt-execution-model) on SMs, warps, and SIMT.

**The communication wall.** This one only appears when you have more than one GPU. The cores are starved because gradients are being summed across the network, or activations are being shipped between pipeline stages, or a tensor-parallel layer is doing an all-reduce mid-forward. The diagnostic is a profiler timeline full of NCCL kernels with idle compute gaps around them. The levers: the **right parallelism for your interconnect** ([data, tensor, pipeline, or expert](/blog/machine-learning/high-performance-computing/parallelism-strategies-data-tensor-pipeline-and-expert)), **overlapping** communication with computation so the network runs in the shadow of the math, and **sharding** so each GPU holds less and communicates the difference. The math of how much a ring all-reduce moves — $2(N-1)/N \cdot S$ bytes per GPU — is derived in the [collectives post](/blog/machine-learning/high-performance-computing/collective-communication-and-nccl-all-reduce-from-scratch), and why the *physical link* (NVLink at 900 GB/s versus InfiniBand at 400 Gb/s versus PCIe) sets your ceiling is in the [interconnects post](/blog/machine-learning/high-performance-computing/interconnects-nvlink-nvswitch-infiniband-and-rdma).

There is a fourth state that is not really a wall but masquerades as one: **GPU-starved**, where the GPU is idle not because of any of the three walls but because the *data pipeline* can't feed it. The profiler shows the GPU waiting at the start of every step while the CPU and disk scramble to assemble a batch. This is heartbreakingly common and is covered in the [cluster and data-pipeline post](/blog/machine-learning/high-performance-computing/running-on-a-cluster-slurm-multi-node-launch-and-data-pipelines). The fix is never a fancier kernel — it is `num_workers`, `pin_memory`, `prefetch_factor`, and a sharded dataset.

Why does naming the wall matter so much? Because the walls are *mutually exclusive at any instant* but *sequential over the optimization*. At any single moment, your job is bottlenecked by exactly one thing — the GPU is either waiting on HBM, or saturating the cores, or waiting on the network, or waiting on data. You cannot be limited by two walls simultaneously; one is always the binding constraint. That's what makes the loop work: you find the *current* binding constraint, remove it, and a *different* wall becomes binding. Fix the memory wall and you might expose a communication wall that was hiding behind it. This is exactly why you re-profile after every lever — the wall *moves*. A team that applies five levers at once and measures only at the end has no idea which lever helped, which hurt, and which wall they're now behind. One lever, one re-profile: that discipline is the difference between a measured climb and a random walk.

The walls also have a natural *priority order* that the checklist in section 5 encodes. GPU-starvation trumps everything because a kernel can't help a GPU with no data. Free precision wins come next because they're one line and help two walls. Capacity walls (you don't fit) must be solved before throughput walls because a job that OOMs has zero throughput. And communication walls only exist multi-GPU, so they're irrelevant until you scale out. This ordering is not arbitrary aesthetics — it's the cost-benefit gradient of the levers, sorted so you always pull the cheapest high-impact lever available.

Here is the map as a table, because you will refer back to it constantly:

| Profiler symptom | Wall | The lever | Derived in |
|---|---|---|---|
| Low intensity, HBM-bound kernels | Memory bandwidth | Fuse, bf16, raise intensity | roofline, kernel-fusion, mixed-precision |
| Cores busy, near peak FLOP/s | Compute | Tensor Cores, better kernels | inside-the-GPU, cuda-programming |
| NCCL kernels with compute gaps | Communication | Right parallelism, overlap, shard | parallelism, collectives, interconnects, FSDP |
| GPU waits at step start | GPU-starved | Data pipeline: workers, prefetch | cluster/SLURM |
| KV-cache OOM, low decode throughput | Serving-specific | Batching, paged KV-cache, TP serving | inference-at-scale |

The diagnosis is the hard part and the profiler does it for you. The cure is then a lookup. Let's see the diagnosis in code.

## 3. Profile first: the two commands that find the wall

The cardinal rule of this entire discipline, repeated in the [profiling post](/blog/machine-learning/high-performance-computing/profiling-gpu-workloads-finding-the-real-bottleneck), is *measure, don't guess*. Human intuition about where time goes in a GPU program is famously, catastrophically wrong — the thing you "know" is slow is usually fine, and the bottleneck is a kernel you never thought about. So before you touch a single lever, you profile. Two tools cover almost everything.

The first is **Nsight Systems** (`nsys`), which gives you the *timeline*: the whole job laid out as a Gantt chart of CUDA kernels, memory copies, and NCCL collectives, with the host (CPU) and device (GPU) on separate rows so you can instantly see whether the GPU is waiting on the CPU. The one-liner:

```bash
# Capture a system-wide timeline of a few training steps.
# -t cuda,nvtx,osrt traces CUDA kernels, your nvtx ranges, and OS runtime.
# --gpu-metrics-device captures SM occupancy and HBM throughput.
nsys profile -t cuda,nvtx,osrt \
  --gpu-metrics-device=0 \
  --capture-range=cudaProfilerApi \
  -o train_profile \
  python train.py --steps 20
```

You open `train_profile.nsys-rep` in the Nsight Systems GUI and look for three things: a GPU row full of *gaps* (GPU-starved or comm-bound), long *NCCL* kernels with compute idle around them (comm wall), or a timeline dominated by a *single elementwise kernel family* like LayerNorm or elementwise-add (memory wall). The shape of the timeline tells you the wall in about ten seconds once you know what you're looking at.

The second is **`torch.profiler`**, which lives inside PyTorch and gives you a per-operator breakdown with self-CUDA-time, plus a Chrome/Perfetto trace. You don't need to leave Python:

```python
import torch
from torch.profiler import profile, ProfilerActivity, schedule

# Profile steady-state steps only: skip warm-up, then record a few.
prof_schedule = schedule(wait=1, warmup=2, active=3, repeat=1)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=prof_schedule,
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(loader):
        loss = train_step(model, batch)   # your normal step
        prof.step()
        if step >= 6:
            break

# The killer table: ops sorted by time actually spent on the GPU.
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
prof.export_chrome_trace("trace.json")   # open in chrome://tracing or Perfetto
```

That `key_averages().table(...)` output is the single most useful thing in your toolkit. If `aten::layer_norm` and `aten::add` and `aten::mul` dominate self-CUDA-time, you are memory-bound and the lever is fusion. If `aten::mm` / `aten::bmm` (the matmuls) dominate *and* their achieved FLOP/s is near peak, you are compute-bound and doing well. If `nccl:all_reduce` dominates, you are comm-bound. If the GPU sits idle between steps and the time is in the dataloader, you are GPU-starved. **One table, four diagnoses.** Everything downstream is a lookup against the map in section 2.

A note on honest measurement, because it bites everyone. GPU work is asynchronous: a Python line that "launches" a kernel returns immediately while the kernel runs later. So `time.time()` around a forward pass measures *launch* time, not *execution* time, and reports nonsense. You must `torch.cuda.synchronize()` before you read the clock, and you must throw away the first few iterations (warm-up: cuDNN autotuning, allocator warm-up, clocks spinning up) and measure steady state. Get this wrong and every number you produce is fiction. The profiler handles synchronization for you, which is one more reason to use it instead of hand-rolled timers.

## 4. The MFU and cost calculators: turning step time into a verdict

Once the profiler tells you the wall, you need to *quantify* where you are so you can tell whether a lever helped. That means computing MFU from a measured step time, and converting MFU into dollars. Here are both as small, runnable functions you can drop into any training script.

```python
def mfu_from_step_time(
    n_params: float,        # model parameter count, e.g. 7e9
    tokens_per_step: int,   # global_batch_size * seq_len
    step_time_s: float,     # measured steady-state seconds per optimizer step
    n_gpus: int,            # GPUs participating
    peak_flops_per_gpu: float,  # e.g. 312e12 for A100 bf16, 989e12 for H100
) -> float:
    """Model FLOPs Utilization = useful FLOP/s achieved / aggregate peak FLOP/s."""
    # 6 * N FLOPs per token (2N forward + 4N backward), the PaLM accounting.
    useful_flops = 6.0 * n_params * tokens_per_step
    achieved_flops_per_s = useful_flops / step_time_s
    aggregate_peak = peak_flops_per_gpu * n_gpus
    return achieved_flops_per_s / aggregate_peak


# Worked: 7B model, 8x A100, global batch 256 seqs of 2048 tokens, 1.9 s/step.
tokens = 256 * 2048                       # 524,288 tokens per step
mfu = mfu_from_step_time(7e9, tokens, 1.9, 8, 312e12)
print(f"MFU = {mfu:.1%}")                 # MFU = 18.6%
```

That `18.6%` is the verdict. It is the same 18% we opened the series with, and now you can compute it for any run in one function call. The beauty is that you can call this *every step* and log it to TensorBoard, so MFU becomes a live dashboard metric, not a one-off measurement. If MFU drops mid-run, something regressed — a checkpoint stall, a thermal throttle, a straggler GPU.

Now the cost side. Given a target token budget, how many GPU-hours and dollars will this run cost at the current MFU, and how much would a better MFU save?

```python
def training_cost(
    n_params: float,
    total_tokens: float,        # e.g. 1e12 for a 1T-token run
    mfu: float,                 # current or target MFU, e.g. 0.18 or 0.50
    peak_flops_per_gpu: float,  # 312e12 A100 bf16
    dollars_per_gpu_hour: float,  # e.g. 2.0 for a rented A100
) -> tuple[float, float]:
    """Return (gpu_hours, dollars) for a dense run at the given MFU."""
    useful_flops = 6.0 * n_params * total_tokens
    effective_flops_per_s = peak_flops_per_gpu * mfu
    gpu_hours = useful_flops / (effective_flops_per_s * 3600.0)
    dollars = gpu_hours * dollars_per_gpu_hour
    return gpu_hours, dollars


for label, mfu in [("baseline", 0.18), ("tuned", 0.50)]:
    hrs, usd = training_cost(7e9, 1e12, mfu, 312e12, 2.0)
    print(f"{label:9s}  MFU {mfu:.0%}  {hrs:8,.0f} GPU-hrs  ${usd:12,.0f}")
# baseline   MFU 18%   590,278 GPU-hrs  $  1,180,556   (approximate)
# tuned      MFU 50%   212,500 GPU-hrs  $    425,000   (approximate)
```

There it is in dollars: the same 7B model on the same trillion tokens costs about **\$1.18M at 18% MFU and about \$425k at 50%** — a saving of roughly **\$755k from optimization alone**, no architecture change. (These are approximate; real costs depend on your exact rate, checkpointing overhead, and restarts. The point is the *ratio*, which is exact: $0.50 / 0.18 = 2.78\times$.) This is why MFU is not an academic metric. It is the lever on your budget.

#### Worked example: is it worth a week of engineering to go from 18% to 35% MFU?

You have a 7B run that will consume 1T tokens at 18% MFU on rented A100s at \$2/GPU-hour. A senior engineer estimates one week of work to reach 35% MFU (bf16 + FlashAttention + a dataloader fix). Is it worth it?

At 18%: from the calculator, \$1,180,556. At 35%: $6 \cdot 7\text{e}9 \cdot 1\text{e}12 / (312\text{e}12 \cdot 0.35 \cdot 3600) \approx 303{,}600$ GPU-hours, so about **\$607,000**. The optimization saves roughly **\$573,000** on this single run. A week of one engineer's time — call it \$5,000 fully loaded — returns about *115×*. And that's before you count every *future* run that inherits the same 35% MFU codebase. The decision is not close. This is the normal economics of HPC work, and it is why "we don't have time to profile" is almost always false economy: profiling is the cheapest hour you will spend on the project.

### Measuring honestly: the traps that make every number a lie

Before you trust an MFU number — yours or anyone else's — you have to know how it was measured, because the easiest way to "improve" MFU is to measure it wrong. There are five traps, and every one of them has burned a team I've worked with.

**The async trap.** As noted, CUDA kernels launch asynchronously: the Python line returns before the GPU finishes. So timing a forward pass with `time.time()` measures *launch* latency, often microseconds, not the milliseconds of actual execution — and your "MFU" comes out absurdly high. You must `torch.cuda.synchronize()` before reading the clock. The correct timing harness uses CUDA events, which timestamp on the GPU's own timeline:

```python
import torch

def time_step(fn, n_warmup=5, n_iters=20):
    # Warm-up: cuDNN autotune, allocator caching, clocks spin up.
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        fn()
    end.record()
    torch.cuda.synchronize()          # wait for all GPU work to finish
    return start.elapsed_time(end) / n_iters / 1000.0   # ms -> s per step
```

**The warm-up trap.** The first few iterations are pathologically slow: cuDNN is benchmarking algorithms, the CUDA caching allocator is carving up memory, and the GPU clocks haven't ramped to their boost frequency. Including those in your average drags MFU down and makes a fast run look slow. Always discard the first 5–10 iterations and measure steady state, as the harness above does.

**The thermal/clock trap.** A GPU under sustained load throttles. An A100 boosts to a high clock for a few seconds, then settles to a lower sustained clock as it heats up — so a 10-iteration benchmark and a 10,000-iteration training run can report different throughput for the *same code*. Measure over enough iterations to reach thermal steady state (tens of seconds), or your benchmark will lie optimistically. This is also why `nvidia-smi`'s clock readout matters: a run pinned at a low clock is being throttled, and no kernel will fix that — it's a cooling or power-cap problem.

**The dataloader-confound trap.** If you time end-to-end step time and the dataloader is the bottleneck, your "GPU MFU" is actually measuring your *disk*. To isolate the GPU's true efficiency, time a step on a single cached batch in a loop (no dataloader), which tells you the ceiling; then compare to the real end-to-end step time. The gap is exactly the GPU-starvation tax. This decomposition is what tells you whether to reach for a kernel lever or a pipeline lever.

**The peak-FLOP/s trap.** Vendors quote multiple peak numbers, and using the wrong denominator silently changes your MFU by 2×. NVIDIA quotes both a *dense* and a *sparse* (2:4 structured sparsity) bf16 figure, and the sparse number is double — an H100's "1979 TFLOP/s" is the sparse figure; the dense number you should use for a normal training run is ~989 TFLOP/s. Use the wrong one and your MFU halves on paper for no real reason. Always anchor the denominator to the *dense* peak for the *exact dtype* your matmuls run in.

Get all five right and your MFU number is trustworthy — it actually reflects the silicon and it actually maps to dollars. Get any one wrong and you'll spend a week chasing a phantom or, worse, ship a "win" that was a measurement artifact.

## 5. The decision tree, as a runnable checklist

Now we make the loop executable. Here is the profile-find-wall-pick-lever logic as Python you could literally run against profiler outputs. It is "pseudocode-but-runnable" — every branch is a real decision you make, encoded as a function that takes a few measured numbers and returns the lever to pull.

```python
from dataclasses import dataclass

@dataclass
class ProfileReport:
    mfu: float                  # current MFU, e.g. 0.18
    n_gpus: int                 # 1, 8, 64 ...
    top_op_is_matmul: bool      # does aten::mm/bmm dominate self-CUDA time?
    matmul_near_peak: bool      # is matmul achieving > ~70% of peak FLOP/s?
    nccl_frac: float            # fraction of step time in NCCL collectives
    gpu_idle_at_step_start: float  # fraction of step the GPU waits for data
    using_bf16: bool
    fits_in_memory: bool        # does the model + optimizer + acts fit?


def pick_lever(r: ProfileReport) -> str:
    # 0. If MFU is already great, stop optimizing and ship.
    if r.mfu >= 0.50:
        return "Good MFU. Stop. Spend effort elsewhere (data, eval)."

    # 1. GPU-starved beats every other wall: a fast kernel can't help a
    #    GPU that has no data. Fix the pipeline FIRST.
    if r.gpu_idle_at_step_start > 0.15:
        return "GPU-STARVED -> data pipeline: num_workers, prefetch, pin_memory, sharded dataset."

    # 2. Communication wall (only possible multi-GPU).
    if r.n_gpus > 1 and r.nccl_frac > 0.20:
        return ("COMM-BOUND -> overlap comm with compute (DDP buckets/FSDP), "
                "check NVLink vs PCIe vs IB, pick parallelism for the interconnect.")

    # 3. Free precision win: if still in fp32, take bf16 before anything else.
    if not r.using_bf16:
        return "MEMORY-BOUND (precision) -> enable bf16 AMP autocast. Nearly free 1.3-2x."

    # 4. Memory-bound elementwise: fusion / FlashAttention.
    if not r.top_op_is_matmul:
        return ("MEMORY-BOUND (elementwise) -> kernel fusion (torch.compile), "
                "FlashAttention for the attention block.")

    # 5. Compute-bound but matmul below peak: better kernels / Tensor Cores.
    if r.top_op_is_matmul and not r.matmul_near_peak:
        return "COMPUTE-BOUND (under peak) -> ensure Tensor Core path, tune kernels (Triton/cuBLAS)."

    # 6. Doesn't fit: shard.
    if not r.fits_in_memory:
        return "MEMORY-CAPACITY -> ZeRO/FSDP sharding, activation checkpointing, offload."

    # 7. Compute-bound near peak: you've won this GPU. Scale out.
    return "COMPUTE-BOUND (near peak) -> you are efficient. Scale out with data parallelism."


# Our opening case: 18% MFU, fp32, 8 GPUs, dataloader stalling.
report = ProfileReport(
    mfu=0.18, n_gpus=8, top_op_is_matmul=False, matmul_near_peak=False,
    nccl_frac=0.10, gpu_idle_at_step_start=0.22, using_bf16=False, fits_in_memory=True,
)
print(pick_lever(report))
# GPU-STARVED -> data pipeline: num_workers, prefetch, pin_memory, sharded dataset.
```

Notice the *ordering* in that function — it is not arbitrary. You check GPU-starvation first because no kernel optimization helps a GPU with nothing to chew on; a fused FlashAttention kernel running on stale data is still 0% useful. You take the free bf16 win before fiddly fusion because it's one line and helps both the memory and compute walls. You only shard when you actually don't fit, because sharding adds communication you don't want if the model fits comfortably. And you stop at 50% MFU because past there the marginal engineering hour is better spent on data quality or evaluation than on squeezing the last few points. **The order encodes the cost-benefit of each lever.** This checklist is the playbook compressed into thirty lines.

## 6. The optimization loop in motion: 18% to 50% on a 7B run

Let's run the loop for real on our spine example — a 7B dense Transformer on a single 8×A100 DGX node — and watch MFU climb lever by lever. Each lap of the loop is: profile, read the wall, pull the one lever the wall calls for, re-measure. This is the heart of the playbook, so we'll go slowly and show the dollars at every step.

![timeline showing MFU climbing from 18 percent baseline through bf16 FlashAttention dataloader fix and FSDP overlap to 50 percent](/imgs/blogs/the-hpc-playbook-for-ai-engineers-3.png)

**Lap 0 — baseline: fp32, naive attention, default dataloader. MFU 18%.** We profile with `nsys`. The timeline shows the GPU idle ~22% of every step waiting at the step boundary, and the dominant kernels are elementwise (LayerNorm, GELU, the attention softmax materializing a big matrix) — and everything is in fp32. The checklist fires GPU-starvation first.

**Lap 1 — fix the data loader.** The model was waiting on a single-process dataloader reading individual files. We shard the dataset, set `num_workers=8`, `pin_memory=True`, `prefetch_factor=4`. The GPU stops waiting at the step boundary.

```python
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=micro_batch,
    num_workers=8,          # parallel CPU workers assembling batches
    pin_memory=True,        # page-locked host memory -> faster H2D copy
    prefetch_factor=4,      # each worker stages 4 batches ahead
    persistent_workers=True,
    drop_last=True,
)
```

Re-profile: the step-start gap is gone, GPU idle drops below 3%. MFU climbs from 18% to about **24%**. This is the unglamorous fix the [cluster post](/blog/machine-learning/high-performance-computing/running-on-a-cluster-slurm-multi-node-launch-and-data-pipelines) calls the 30% most teams leave on the table, and it cost us four lines.

**Lap 2 — bf16 mixed precision.** Profile again: now the GPU is busy but everything is fp32, moving twice the bytes it needs to and not touching the Tensor Cores' fast bf16 path. The checklist says take the free precision win. We wrap the forward in `autocast`. bf16 needs no loss scaling (its 8-bit exponent matches fp32's dynamic range — the [mixed-precision post](/blog/machine-learning/high-performance-computing/numerical-formats-and-mixed-precision-fp32-tf32-bf16-fp16-fp8) derives exactly why fp16 would need a `GradScaler` here and bf16 does not):

```python
for batch in loader:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(batch).loss          # matmuls run in bf16 on Tensor Cores
    loss.backward()                       # bf16: no GradScaler needed
    optimizer.step()
```

Half the bytes moved on the memory-bound ops, and the matmuls now hit the bf16 Tensor Core path at 312 TFLOP/s instead of the fp32 path at ~19.5 TFLOP/s. MFU jumps from 24% to about **32%**.

**Lap 3 — FlashAttention.** Re-profile. The matmuls are healthy, but the attention block is still a memory-bandwidth pig: it materializes the $s \times s$ scores matrix in HBM, runs softmax over it, writes it back, reads it again. For a 2048-token sequence that's a lot of HBM traffic for a low-intensity op. We swap in FlashAttention, which tiles the computation and never writes the full scores matrix to HBM — the [kernel-fusion post](/blog/machine-learning/high-performance-computing/kernel-fusion-and-flashattention-beating-the-memory-wall) derives the HBM-traffic reduction in detail.

```python
import torch.nn.functional as F
# PyTorch's scaled_dot_product_attention dispatches to a FlashAttention
# kernel when shapes/dtypes qualify. is_causal handles the mask for free.
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

MFU climbs from 32% to about **38%**, and as a bonus, peak activation memory drops because we no longer store the big scores matrix — which matters for the next lap.

**Lap 4 — overlap and shard with FSDP.** Profile once more. We are now multi-GPU bound: the gradient all-reduce after `backward()` shows up as a NCCL block on the timeline with the GPUs idle around it. The optimizer states (Adam's two moments at fp32 = 8 bytes/param, plus the fp32 master copy) also crowd the 80 GB. We switch from plain DDP to **FSDP**, which shards parameters, gradients, and optimizer states across the 8 GPUs and *overlaps* the all-gather of the next layer's weights with the current layer's compute — the [ZeRO-FSDP post](/blog/machine-learning/high-performance-computing/memory-optimization-zero-fsdp-activation-checkpointing-and-offload) walks the `(2+2+12)\Psi` memory split and the overlap mechanism.

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision

model = FSDP(
    model,
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,    # reduce-scatter grads in bf16: half the comm bytes
    ),
    # FSDP overlaps the next layer's all-gather with this layer's compute,
    # so communication runs in the shadow of computation.
)
```

The NCCL time now overlaps with compute instead of blocking it, and the freed memory lets us grow the micro-batch (raising arithmetic intensity, which feeds the Tensor Cores better still). MFU reaches about **50%**.

Here is the whole arc as a scorecard, with the dollar cost of a 1T-token run at each MFU (rented A100 at \$2/GPU-hour, from the cost calculator):

| Lap | Lever | Wall fixed | MFU | Cost for 1T tokens |
|---|---|---|---|---|
| 0 | baseline fp32 | — | 18% | ~\$1,180,000 |
| 1 | dataloader fix | GPU-starved | 24% | ~\$885,000 |
| 2 | bf16 AMP | memory + compute | 32% | ~\$664,000 |
| 3 | FlashAttention | memory | 38% | ~\$559,000 |
| 4 | FSDP overlap + shard | comm + memory | 50% | ~\$425,000 |

Four laps, five levers, and the run got **2.8× cheaper** — roughly **\$755,000 saved** on this single training run, every number traceable to a specific lever and a specific wall. None of it required a new GPU, a different model, or a heroic kernel. It required the loop: profile, find the wall, pick the lever, measure. That is the whole playbook, and you just watched it work.

#### Worked example: where did the 32 points of MFU come from?

Decompose the 18% → 50% gain so you see that the levers *compose* rather than overlap. Think of MFU as a product of efficiencies (this is the abstraction the loop relies on): $\text{MFU} \approx \eta_\text{feed} \cdot \eta_\text{precision} \cdot \eta_\text{kernel} \cdot \eta_\text{comm}$, where each factor is "how much of the available throughput this dimension preserves." At baseline, the dataloader stall caps $\eta_\text{feed}$ at about 0.78 (22% idle), fp32 caps $\eta_\text{precision}$ at roughly 0.55 (fp32 path is far slower than the bf16 Tensor Core path), naive attention caps $\eta_\text{kernel}$ at about 0.75, and single-node DDP keeps $\eta_\text{comm}$ near 0.9. Multiply through with a base hardware efficiency and you land near 18%. Each lever raises one factor toward 1.0: the dataloader fix takes $\eta_\text{feed}$ to ~0.97, bf16 takes $\eta_\text{precision}$ to ~0.95, FlashAttention takes $\eta_\text{kernel}$ to ~0.95, FSDP overlap takes $\eta_\text{comm}$ to ~0.95. Because they multiply, fixing four separate 0.75–0.9 factors compounds into a 2.8× total. This is *why* you fix one wall at a time and re-profile: each lever targets a different factor, and a lever aimed at an already-healthy factor buys nothing. The 50% ceiling is roughly where the remaining inefficiency is irreducible overhead (kernel launch, the last bit of comm, non-matmul FLOPs) that no single lever removes cheaply.

## 7. The lever menu in full: technique × wall × typical gain

You've now seen the loop run once. Here is the complete menu you draw from on every lap, with the wall each lever fixes and the typical measured gain from the literature and from real runs. Treat this as the reference card you keep open while optimizing.

![matrix of optimization levers showing the wall each fixes its effect and the typical measured gain](/imgs/blogs/the-hpc-playbook-for-ai-engineers-4.png)

The table below is the same information in prose-friendly form. The gains are *typical* and *workload-dependent* — mark them all as approximate, because the exact number depends on your shapes, sequence length, and hardware. They are the right *order of magnitude*, sourced where possible.

| Lever | Wall it fixes | What it moves | Typical gain (approx) |
|---|---|---|---|
| bf16 / tf32 AMP | memory + compute | half the bytes; Tensor Core matmul | 1.3–2× throughput |
| Kernel fusion (`torch.compile`) | memory | fewer HBM round-trips on pointwise chains | 1.2–2× on memory-bound blocks |
| FlashAttention | memory | never materialize the s×s scores matrix | 2–4× attention, big memory saving |
| Tensor Cores / better GEMM | compute | 16× matmul rate vs CUDA cores | up to ~10× on the matmul itself |
| Data parallelism (DDP) | comm | replicate, all-reduce gradients | near-linear scaling on NVLink |
| Tensor parallelism | comm + capacity | split one matmul across GPUs | fits layers too big for one GPU |
| Pipeline parallelism | comm + capacity | split layers into stages | fits very deep models; bubble cost |
| ZeRO / FSDP | memory capacity | shard params, grads, optimizer states | fits 13B–70B on commodity nodes |
| Activation checkpointing | memory capacity | recompute instead of store activations | ~30% more compute, big memory cut |
| Fix the data loader | GPU-starved | overlap I/O with compute | up to ~1.5× when starved |
| Continuous batching + paged KV | serving | pack the decode batch, page the cache | several× serving throughput |

Two things to internalize from this table. First, **a lever can fix more than one wall** — bf16 helps both memory (fewer bytes) and compute (Tensor Cores), tensor parallelism helps both communication patterns and capacity. That's good; the multi-wall levers are usually the first ones to reach for. Second, **some levers cost compute to save memory** — activation checkpointing trades ~30% extra FLOPs for a large activation-memory reduction. That is *only* worth it when memory capacity is the wall (you don't fit) and you have compute to spare; applying it to a compute-bound run makes everything slower. The lever's *cost* is as important as its benefit, which is why the playbook always names the wall first.

The serving row points at a genuinely different problem. Training is throughput-bound and you optimize tokens-per-second; serving is latency-*and*-throughput bound, and the dominant cost is the memory-bound autoregressive decode plus the KV-cache that grows with batch × sequence length. The [inference-at-scale post](/blog/machine-learning/high-performance-computing/inference-at-scale-batching-kv-cache-and-tensor-parallel-serving) covers continuous batching and paged attention; the headline result there is vLLM's PagedAttention reporting up to ~24× higher throughput than naive HuggingFace serving by eliminating KV-cache fragmentation. Same philosophy — find the wall (here, KV-cache memory and decode bandwidth), pick the lever (paging and batching) — different problem shape.

One more structural point about the menu: the levers are not independent — some *unlock* others. Cut activation memory with FlashAttention and you can grow the batch, which raises arithmetic intensity, which feeds the Tensor Cores better — a memory lever that pays a compute dividend. Shard with FSDP and you free memory that lets you grow the micro-batch, again raising intensity. This is why the *order* you pull levers matters beyond just the priority of the walls: a memory lever early can create headroom that makes a later throughput lever possible. The 7B arc in section 6 is built this way on purpose — the FlashAttention step in lap 3 frees the activation memory that lets the FSDP step in lap 4 grow the batch and push intensity up. When you plan a multi-lever optimization, sketch this dependency chain first: which levers create headroom, which consume it, and in what order they compound. A good optimization plan reads like a build dependency graph, not a checklist of independent items.

## 8. The scaling ladder: what changes at 1, 8, and many GPUs

The single most expensive mistake in distributed training is picking the wrong parallelism for your interconnect, so the playbook needs an explicit model of how the walls *shift* as you scale. The key insight: **every rung of the scaling ladder adds a slower communication link, and the slower link becomes the new wall.** What was a compute problem on one GPU becomes a bandwidth problem across a node and a network problem across nodes.

![diagram of the scaling ladder from one GPU on HBM to eight on NVLink to many on InfiniBand with the lever at each rung](/imgs/blogs/the-hpc-playbook-for-ai-engineers-5.png)

**One GPU.** The only links are on-chip: registers, shared memory, L2, and HBM at ~2–3.3 TB/s. There is no network, so the communication wall doesn't exist. Your levers are all single-GPU: precision, fusion, FlashAttention, better kernels. This is where you should get MFU as high as possible *before* scaling out, because every inefficiency you carry here gets multiplied across every GPU you add. A 30% MFU single-GPU baseline scaled to 64 GPUs is 64 GPUs running at 30%; fix it to 50% first and you've effectively bought ~21 GPUs for free.

**Eight GPUs in a node (NVLink/NVSwitch).** Now there's an intra-node fabric. On a DGX, NVSwitch gives every GPU ~900 GB/s of all-to-all bandwidth — fast enough that *tensor parallelism*, which does an all-reduce in the middle of every forward and backward, is viable. Data-parallel all-reduce of gradients is comfortably hidden. The wall here is usually still memory or compute, with communication overlapping nicely. The lever menu: DDP or FSDP for data parallelism, tensor parallelism for layers too big to fit, both riding the fast NVLink fabric. The [interconnects post](/blog/machine-learning/high-performance-computing/interconnects-nvlink-nvswitch-infiniband-and-rdma) is explicit that *tensor parallelism should not cross the NVLink boundary* — its frequent, latency-sensitive collectives die over the slower inter-node link.

**Many GPUs across nodes (InfiniBand + RDMA).** Now you've crossed the network. InfiniBand at 400 Gb/s (= 50 GB/s) per port with RDMA and GPUDirect is fast for a *network* but ~18× slower than NVLink. The communication wall now dominates your scaling decisions. The rule that falls out: **keep the chatty parallelism inside the node and the sparse parallelism across nodes.** Tensor parallelism (chatty, every layer) stays within the 8-GPU NVLink island; pipeline parallelism (sparse, once per micro-batch boundary) and data parallelism (once per step, and overlappable) span the InfiniBand fabric. This is exactly the 3D-parallelism layout that Megatron-LM uses to train models across thousands of GPUs, and it is why "scaling efficiency falls off a cliff over the wrong link" — put tensor parallelism on InfiniBand and your 64-GPU run can be *slower* than your 8-GPU run.

Here is a sizing calculator that encodes this — given your model, your GPU memory, and your interconnect topology, it suggests a parallelism layout:

```python
def suggest_parallelism(
    n_params: float,            # 7e9, 70e9, ...
    bytes_per_param_state: int, # bf16 params+grads + fp32 Adam states ~ 18 B/param for ZeRO-1
    gpu_mem_gb: float,          # 80 for A100/H100 80GB
    gpus_per_node: int,         # 8 on a DGX
    total_gpus: int,
    has_nvlink: bool,
    has_infiniband: bool,
) -> dict:
    state_gb = n_params * bytes_per_param_state / 1e9
    # Can one GPU hold the full sharded-nothing state? (DDP replicates fully.)
    fits_one_gpu = state_gb < gpu_mem_gb * 0.7   # leave 30% for activations
    plan = {}

    if fits_one_gpu:
        plan["primary"] = "DDP (data parallel) — model fits, just replicate."
        plan["shard"]   = "none needed; add FSDP only if activations are tight."
    else:
        # Need to shard. FSDP shards across all GPUs.
        plan["shard"] = "FSDP / ZeRO-3 — shard params+grads+optimizer states."
        # Does a single LAYER's matmul exceed one GPU? Then tensor-parallel it,
        # but ONLY within the NVLink island.
        if has_nvlink:
            plan["tensor_parallel"] = f"TP within node (<= {gpus_per_node} GPUs, on NVLink)."
        else:
            plan["tensor_parallel"] = "AVOID TP — no NVLink; TP over PCIe/IB will stall."

    # Across nodes, prefer pipeline + data parallel (sparse comms).
    if total_gpus > gpus_per_node:
        if has_infiniband:
            plan["across_nodes"] = "Pipeline + data parallel across nodes (sparse, overlappable comms)."
        else:
            plan["across_nodes"] = "WARNING: multi-node without IB — expect poor scaling efficiency."
    return plan


print(suggest_parallelism(70e9, 18, 80, 8, 64, has_nvlink=True, has_infiniband=True))
# {'shard': 'FSDP / ZeRO-3 ...',
#  'tensor_parallel': 'TP within node (<= 8 GPUs, on NVLink).',
#  'across_nodes': 'Pipeline + data parallel across nodes ...'}
```

The logic mirrors the [parallelism post](/blog/machine-learning/high-performance-computing/parallelism-strategies-data-tensor-pipeline-and-expert)'s decision tree: if it fits, just DDP; if it doesn't, shard with FSDP and keep tensor parallelism inside the NVLink island; span nodes with the sparse, overlappable parallelisms. The calculator is the playbook's "which parallelism" answer, encoded.

#### Worked example: 70B on 64 H100s — does the layout matter?

You're training a 70B model on 64 H100 80GB GPUs (8 nodes × 8 GPUs, NVLink within node, InfiniBand between). The state is roughly $70\text{e}9 \times 18 \text{ B} = 1{,}260$ GB — far past one GPU's 80 GB, so you must shard. Two layouts:

**Layout A (naive):** tensor-parallel degree 8 *across* two nodes (crossing InfiniBand), data-parallel the rest. Tensor parallelism does an all-reduce twice per layer; on a 70B model with ~80 layers that's ~160 all-reduces per forward, each now crossing the 50 GB/s InfiniBand link instead of the 900 GB/s NVLink. Scaling efficiency collapses — you might see the 64-GPU run sustain only ~25% MFU, *worse per-GPU* than 8 GPUs.

**Layout B (playbook):** tensor-parallel degree 8 *within* each node (on NVLink), FSDP to shard across the data-parallel dimension, pipeline across node groups if needed. The chatty TP collectives stay on the 900 GB/s fabric; only the sparse FSDP reduce-scatter and the once-per-step data-parallel sync cross InfiniBand, and both overlap with compute. This is the Megatron/DeepSpeed layout, and it sustains roughly **40–48% MFU at this scale** in published runs. Same hardware, same model — the *only* difference is matching the parallelism to the link, and it is a ~2× difference in cost. This single decision is worth more than every kernel you'll ever hand-tune.

## 9. The scorecard: MFU as the north star, dollars as the verdict

A playbook needs a scoreboard, and ours has exactly two numbers: MFU (efficiency) and dollars (the verdict). Everything else — GB/s, µs per kernel, scaling efficiency — is *diagnostic*, useful for finding the wall, but MFU and cost are what you report up and track over the life of a project.

![matrix scorecard relating MFU to GPU-hours and dollars per result for a one trillion token run](/imgs/blogs/the-hpc-playbook-for-ai-engineers-6.png)

Here is the scorecard for our 7B run on a 1T-token budget, plus a frontier reference point. The frontier numbers anchor your expectations: PaLM 540B reported ~46% MFU, GPT-3-scale and LLaMA runs land in the 30–55% range depending on hardware and scale, and Megatron-LM's published tensor+pipeline runs hit the high 40s to low 50s. So a well-tuned dense run at ~50% MFU is genuinely *at the frontier of what's achievable* for dense Transformers — there is no secret 90% hiding from you.

| Configuration | MFU | GPU-hours (1T tok, A100) | Cost at \$2/GPU-hr |
|---|---|---|---|
| Baseline (fp32, naive) | 18% | ~590,000 | ~\$1,180,000 |
| bf16 + FlashAttention | 35% | ~304,000 | ~\$607,000 |
| Fully tuned (this playbook) | 50% | ~213,000 | ~\$425,000 |
| PaLM 540B (reference) | ~46% | — | frontier bar |

For serving, the scorecard's units change but the philosophy doesn't: the north-star efficiency metric becomes **\$/million-tokens** (or tokens-per-second-per-GPU at a latency SLA), and the wall is usually KV-cache memory and decode bandwidth rather than training FLOPs. You still profile, still find the wall, still pick the lever (continuous batching, paged KV, tensor-parallel serving), still measure the cost per million tokens. The cost equation is the same shape — useful work over effective throughput times the rate — just with serving's definition of "useful work."

The discipline is: **pick one efficiency number for your problem (MFU for training, \$/million-tokens for serving), put it on a live dashboard, and never run a job without watching it.** A run whose MFU silently dropped from 45% to 30% halfway through — because a node started throttling, or a straggler appeared, or a checkpoint write stalled the all-reduce — is a run quietly costing you 50% more money, and without the dashboard you'd never know. The scorecard is not paperwork; it is the smoke detector.

## 10. The anti-patterns and their one-line fixes

Most slow jobs are slow for the same five reasons, and each has a fix the profiler points you straight to. If you remember nothing else, remember these — they account for the overwhelming majority of the "my expensive cluster is somehow slow" tickets.

![before and after diagram of five training anti-patterns on the left and their matching fixes on the right](/imgs/blogs/the-hpc-playbook-for-ai-engineers-7.png)

**Anti-pattern 1: fp32 by default.** PyTorch defaults to fp32, which moves twice the bytes and skips the bf16 Tensor Core path entirely. The fix is one context manager — `torch.autocast("cuda", dtype=torch.bfloat16)` — and on modern hardware it is nearly free accuracy-wise for the forward/backward (keep the optimizer master weights in fp32). This single line is often a 1.3–2× win and it's the first thing to check on any untuned run.

**Anti-pattern 2: tiny batches.** A small batch means low arithmetic intensity — the matmuls are too small to amortize the cost of reading weights from HBM, so you sit on the memory-bound side of the roofline with the Tensor Cores half-idle. The fix is to grow the batch until the GPU saturates, and if memory won't allow it, use **gradient accumulation** to get a large *effective* batch from small micro-batches. The [roofline post](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) shows exactly how intensity rises with batch size and pushes the op rightward toward the compute-bound regime.

**Anti-pattern 3: ignoring the data loader.** The most common cause of a GPU sitting at "100% utilization" in `nvidia-smi` while doing almost nothing useful — `nvidia-smi`'s utilization counts "a kernel was resident," not "the kernel did useful work." A single-process loader reading small files starves the GPU between steps. The fix: `num_workers`, `pin_memory`, `prefetch_factor`, a sharded/columnar dataset, and an NVMe cache. This is the cheapest large win in the whole playbook and the most overlooked.

**Anti-pattern 4: wrong parallelism for the interconnect.** Putting tensor parallelism over PCIe or across InfiniBand, as the section 8 worked example showed, can make a bigger cluster *slower* than a smaller one. The fix: keep chatty collectives (tensor parallel) inside the NVLink island, span nodes with sparse, overlappable parallelisms (pipeline, data). Match the parallelism to the link.

**Anti-pattern 5: never profiling.** The meta-anti-pattern that causes all the others. Optimizing without profiling is optimizing the wrong thing — you "know" the matmul is slow and spend a week on a custom kernel while the real bottleneck was a dataloader you never looked at. The fix is the loop itself: `nsys` and `torch.profiler` first, *always*, before touching a single lever. The cheapest hour on any HPC project is the first profiling hour.

| Anti-pattern | Symptom in the profiler | One-line fix |
|---|---|---|
| fp32 by default | fp32 kernels, 2× HBM traffic | `autocast(dtype=bfloat16)` |
| Tiny batches | matmuls below peak, low intensity | grow batch / gradient accumulation |
| Ignored data loader | GPU idle at step start | `num_workers`, `pin_memory`, `prefetch_factor` |
| Wrong parallelism for link | NCCL stalls, sub-linear scaling | TP on NVLink, pipeline/DP across IB |
| Never profiling | optimizing the wrong kernel | run `nsys` / `torch.profiler` first |

There is a sixth, subtler anti-pattern worth naming: **optimizing past the point of diminishing returns.** Once you're at 50% MFU on a dense run, you are near the frontier, and the next five points of MFU might cost a month of kernel engineering. That month is almost always better spent on data quality, a better evaluation harness, or simply starting the run. Know when to stop — the checklist's first branch (`if mfu >= 0.50: stop`) is there for a reason. A playbook that doesn't tell you when to *quit* optimizing is incomplete.

## 11. Case studies / real numbers

The levers in this playbook are not theoretical — they are the techniques that built every frontier model, and the published numbers anchor what "good" looks like. Treat all figures as approximate and as reported by their sources; they vary with hardware, scale, and the exact configuration.

**PaLM 540B — the MFU benchmark.** Google's PaLM paper reported training 540B parameters at roughly **46% MFU** on 6,144 TPU v4 chips, which at the time was a landmark efficiency for a model that scale. The number matters because it sets the bar: ~46% on a frontier-scale model means a well-tuned dense Transformer run lives in the mid-40s to low-50s, and if your run is at 18% you have enormous headroom, while if you're at 48% you're essentially done. PaLM's efficiency came from exactly the levers in this playbook — careful parallelism layout, attention/MLP fusion, and high arithmetic intensity from large batches.

**Megatron-LM — parallelism done right.** NVIDIA's Megatron-LM work demonstrated training models up to 1 trillion parameters by combining tensor, pipeline, and data parallelism (3D parallelism) across thousands of GPUs, sustaining roughly **50%+ of peak** on the largest configurations. The published scaling studies are the empirical proof of section 8's rule: tensor parallelism kept within the NVLink island, pipeline and data parallelism spanning the InfiniBand fabric. When you read that Megatron hits ~52% MFU at 1T parameters while a naive layout collapses, you are reading the dollar value of matching parallelism to the interconnect.

**FlashAttention — the memory-wall lever, measured.** The FlashAttention papers reported **2–4× speedups** on the attention block and, crucially, a reduction in memory from $O(s^2)$ to $O(s)$ by never materializing the scores matrix — enabling much longer context lengths at fixed memory. This is the canonical memory-wall win: same FLOPs, dramatically fewer HBM round-trips, by restructuring the computation to raise arithmetic intensity. FlashAttention-2 and -3 pushed further, with -3 reaching a large fraction of H100 peak by exploiting the newer Tensor Cores and asynchrony.

**ZeRO / FSDP — fitting the unfittable.** Microsoft's ZeRO (the foundation of DeepSpeed and the model behind PyTorch FSDP) showed that sharding parameters, gradients, and optimizer states across data-parallel ranks lets you fit models *far* larger than a single GPU's memory — the published work fit 13B and then 100B+ parameter models on commodity GPU counts that could never hold the full fp32 optimizer state. The memory split $(2 + 2 + 12)\Psi$ — 2 bytes bf16 params, 2 bytes bf16 grads, 12 bytes of fp32 Adam states per parameter — is exactly what ZeRO-3/FSDP shards, turning a model that OOMs into one that fits with near-DDP throughput thanks to comm/compute overlap.

**vLLM PagedAttention — the serving lever.** On the inference side, vLLM's PagedAttention reported up to **~24× higher throughput** than naive HuggingFace Transformers serving by paging the KV-cache like virtual memory, eliminating fragmentation, and enabling continuous batching. It's the serving analogue of FlashAttention: find the wall (KV-cache memory and decode bandwidth), restructure the data layout, and unlock multiples of throughput — and therefore cut \$/million-tokens by a similar factor.

**LLaMA and the open-model MFU range.** Meta's LLaMA and LLaMA-2 technical reports describe training on large A100 clusters at MFU figures in the roughly 30–50% range depending on model size and sequence length, using exactly the playbook's levers: bf16, FlashAttention, fused kernels, and a 3D-parallel layout matched to the interconnect. The 65B/70B runs are particularly instructive because they sit right at the boundary where a single node can't hold the optimizer state, forcing sharding — which is the real-world version of section 8's sizing calculator firing the "FSDP / ZeRO-3" branch. When you read that a 70B model trained at ~45% MFU on a few thousand A100s, you are reading the end state of the loop in this post applied at scale: every wall found, every lever pulled, the result measured in MFU and quoted in the paper.

The thread through all five: each one *found a specific wall and applied the specific lever for it*, then *measured the win in MFU or throughput or dollars*. None of them is magic. Every one is the loop, run by people who profiled first. And notice the convergence: independent teams at Google, NVIDIA, Microsoft, and Meta, working on different hardware with different models, all land in the same ~45–52% MFU band for well-tuned dense training. That convergence is not a coincidence — it's the empirical signature of the same physical ceiling. The non-matmul FLOPs, the irreducible communication, the kernel-launch overhead, and the memory-bound tails of even a fused model add up to roughly the same ~50% efficiency loss everywhere, which is why ~50% MFU is the honest target and ~90% is a fantasy. Knowing where the real ceiling sits is itself a lever: it tells you when to stop spending engineering hours and start spending GPU-hours.

## 12. When to reach for this (and when not to)

The playbook is a loop, but the most senior thing in it is knowing when to *stop pulling levers*. Optimization is never free — every lever costs engineering time, adds code complexity, and sometimes trades one resource for another. Here is the decisive guidance.

**Always profile first — there is no exception.** The one rule that never has a "but." Even if you're "sure" you know the bottleneck, spend the profiling hour. It's the cheapest insurance in the discipline and it's wrong less than half the time, which is exactly when it saves you a wasted week.

**Take the free wins immediately: bf16 and the dataloader.** On any untuned run, `autocast(bfloat16)` and a properly configured `DataLoader` are nearly free and almost always help. Do them before you profile deeply — they're cheap enough that the cost-benefit never loses.

**Reach for FlashAttention whenever you have attention and any real sequence length.** It is a drop-in via `scaled_dot_product_attention`, it both speeds up and saves memory, and there is essentially no downside. This is a default, not a tuning decision.

**Add sharding (FSDP/ZeRO) only when you don't fit.** If the model, gradients, and optimizer states fit comfortably with DDP and DDP saturates your NVLink, *do not* add FSDP — you'd pay extra all-gather communication for no benefit. Sharding is a capacity lever; reach for it when capacity is the wall, not before.

**Add tensor parallelism only when a single layer won't fit and you have NVLink.** TP is the chattiest parallelism and the most fragile — it pays only when a layer is too big for one GPU *and* you're on a fast intra-node fabric. Over PCIe or InfiniBand it's a trap. If the whole model fits and DDP is fine, TP is pure overhead.

**Add pipeline parallelism only past several stages.** The pipeline bubble — the $(p-1)/(m+p-1)$ fraction of idle time where $p$ is stages and $m$ is micro-batches — means pipelining only pays when you have enough stages and micro-batches to keep the bubble small. For a model that fits with data parallelism, pipelining just adds bubble.

**Don't chase a custom kernel when you're compute-bound and near peak.** If the profiler says your matmuls are at 80% of peak FLOP/s, you are *done* with that op — a hand-written kernel will, at best, claw back a few percent for weeks of work. Spend the time scaling out or improving the data instead.

**Stop optimizing at ~50% MFU on a dense run.** You're at the frontier. The marginal engineering hour now returns more invested in data, evaluation, or just launching. The checklist's first branch encodes this; respect it.

The meta-rule: **the wall picks the lever, and the cost picks whether to pull it.** A lever that fixes your wall but costs more than it saves is still the wrong move. Always run the cost calculator before committing engineering weeks — a week that saves \$573k is obvious; a week that saves \$3k is not, and the only way to tell them apart is the arithmetic in section 4.

## Key takeaways

- **The loop is the whole job: profile → find the wall → pick the lever → measure MFU and dollars → repeat.** Optimization is not a list of techniques; it is this loop run until MFU plateaus. Every prior post in this series is a lever in step three.
- **There are exactly three walls** — compute, memory bandwidth, communication — plus a GPU-starved state and a serving-specific case. The profiler tells you which one you're behind in about ten seconds; the wall then narrows your lever choice to a short menu.
- **MFU is the one number that scores everything**: $\text{MFU} = 6N \cdot T / P$. It's achieved-over-peak, it falls between ~0.15 and ~0.6 for real runs, and ~0.5 is the frontier for dense Transformers.
- **MFU is in the denominator of the cost.** $\text{GPU-hours} = 6N \cdot T_\text{total} / (P \cdot \text{MFU} \cdot 3600)$. Double MFU, halve the bill. Our 7B example went 18% → 50% MFU and 2.8× cheaper — about \$755k saved on one run — with zero model changes.
- **Each lever fixes one wall; applied to the wrong wall it pays its cost and buys nothing.** bf16 and fusion fix memory; Tensor Cores fix compute; the right parallelism + overlap + sharding fix communication; the dataloader fix feeds a starved GPU.
- **The scaling ladder shifts the wall**: one GPU is compute/memory, eight on NVLink adds intra-node bandwidth, many on InfiniBand makes communication dominate. Keep chatty parallelism (TP) on NVLink, span nodes with sparse parallelism (pipeline, DP).
- **Five anti-patterns cause most slow jobs**: fp32 by default, tiny batches, ignored dataloaders, wrong parallelism for the link, and never profiling. Each has a one-line fix the profiler points you to.
- **Know when to stop.** At ~50% MFU you're at the frontier; the next hour is better spent on data or evaluation. The wall picks the lever, and the cost model picks whether the lever is worth pulling.

## Further reading

- Chowdhery et al., *PaLM: Scaling Language Modeling with Pathways* — the MFU accounting and the ~46% benchmark that anchors "good."
- Shoeybi et al. and Narayanan et al., *Megatron-LM* — 3D parallelism and the published scaling efficiency that proves the interconnect-matching rule.
- Dao et al., *FlashAttention* (and -2, -3) — the IO-aware attention kernel and its 2–4× / memory-saving measurements.
- Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* — the sharding math behind DeepSpeed and FSDP.
- Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM) — the serving-side lever and its throughput numbers.
- The NVIDIA A100 and H100 architecture whitepapers — the peak FLOP/s and HBM bandwidth specs every roofline and MFU computation depends on.
- The PyTorch FSDP and `torch.profiler` docs, and the NCCL and Nsight Systems/Compute documentation — the toolchain this playbook runs on.
- Within this series, the full map of where each lever is derived:

![tree showing the full HPC series map of four tracks and fifteen posts from the machine to one fast GPU to many GPUs to the cluster](/imgs/blogs/the-hpc-playbook-for-ai-engineers-8.png)

This capstone is the hub of a 15-post series. Track A — the machine — is [why HPC is the bottleneck](/blog/machine-learning/high-performance-computing/why-hpc-is-the-bottleneck-for-modern-ai), [inside the GPU](/blog/machine-learning/high-performance-computing/inside-the-gpu-sms-warps-and-the-simt-execution-model), [the memory hierarchy](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm), and [the roofline model](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound). Track B — one fast GPU — is [numerical formats and mixed precision](/blog/machine-learning/high-performance-computing/numerical-formats-and-mixed-precision-fp32-tf32-bf16-fp16-fp8), [CUDA programming](/blog/machine-learning/high-performance-computing/cuda-programming-for-ai-engineers-threads-blocks-and-a-first-kernel), [kernel fusion and FlashAttention](/blog/machine-learning/high-performance-computing/kernel-fusion-and-flashattention-beating-the-memory-wall), and [profiling GPU workloads](/blog/machine-learning/high-performance-computing/profiling-gpu-workloads-finding-the-real-bottleneck). Track C — many GPUs — is [parallelism strategies](/blog/machine-learning/high-performance-computing/parallelism-strategies-data-tensor-pipeline-and-expert), [collective communication and NCCL](/blog/machine-learning/high-performance-computing/collective-communication-and-nccl-all-reduce-from-scratch), [interconnects](/blog/machine-learning/high-performance-computing/interconnects-nvlink-nvswitch-infiniband-and-rdma), and [memory optimization with ZeRO and FSDP](/blog/machine-learning/high-performance-computing/memory-optimization-zero-fsdp-activation-checkpointing-and-offload). Track D — the cluster and this playbook — is [running on a cluster with SLURM](/blog/machine-learning/high-performance-computing/running-on-a-cluster-slurm-multi-node-launch-and-data-pipelines), [inference at scale](/blog/machine-learning/high-performance-computing/inference-at-scale-batching-kv-cache-and-tensor-parallel-serving), and this post. For the layers above this one, the [edge-AI inference-runtimes comparison](/blog/machine-learning/edge-ai/inference-runtimes-compared) covers the on-device serving stacks, and the [Chinchilla compute-optimal scaling post](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) tells you how many tokens the cost model in section 1 should be budgeted for in the first place.

You now have the operating manual. Walk up to any slow training or serving job, run `nsys` and `torch.profiler`, read the wall off the timeline, pick the one lever from the menu, and measure MFU and dollars. Then do it again. That loop — not any single trick — is what separates a quarter-million dollars of idle silicon from a cluster running at the frontier of what the hardware can do.

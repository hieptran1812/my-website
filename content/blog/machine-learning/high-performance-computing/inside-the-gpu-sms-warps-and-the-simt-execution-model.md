---
title: "Inside the GPU: SMs, Warps, and the SIMT Execution Model"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "A working tour of the silicon you call with .cuda() — streaming multiprocessors, 32-lane warps, latency hiding, occupancy, Tensor Cores, and the divergence tax — so you can finally predict why throughput hardware loves dense, regular work."
tags:
  [
    "high-performance-computing",
    "gpu",
    "cuda",
    "simt",
    "tensor-cores",
    "warp-scheduling",
    "occupancy",
    "deep-learning",
    "ml-systems",
    "a100",
  ]
category: "machine-learning"
subcategory: "High Performance Computing"
author: "Hiep Tran"
featured: true
readTime: 59
image: "/imgs/blogs/inside-the-gpu-sms-warps-and-the-simt-execution-model-1.png"
---

You have called `.cuda()` a thousand times. You have watched `nvidia-smi` flicker to 100% and felt good about it. And then one day you profiled a training run, saw 18% MFU — model FLOPs utilization, the fraction of the GPU's advertised math throughput your model actually used — and realized that the green bar in `nvidia-smi` was lying to you. The GPU was *busy*. It was just not doing your matmuls. It was waiting on memory, replaying a branchy kernel lane by lane, or running a kernel so small that launch overhead dwarfed the work.

To fix any of that you have to stop treating the GPU as a magic box that makes tensors fast and start treating it as what it is: a very particular kind of machine with a very particular set of tastes. **A GPU is not a fast CPU.** It is a throughput engine built from many small, simple, latency-tolerant lanes, and it is fast *only* when you feed it dense, regular, predictable arithmetic. The same chip that does 312 trillion bf16 multiply-adds per second on a big matmul will crawl on a reduction, choke on a branchy kernel, and idle on a workload that doesn't have enough parallel work to hide its own memory latency.

This post is the hardware tour underneath that fact. We will open up an NVIDIA A100 — a real, named chip with public specs in the [A100 whitepaper](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf) — and walk down through it: the **streaming multiprocessors (SMs)** that are the real compute units, the **CUDA cores** inside them, the **warp** of 32 threads that is the true unit of execution, the **SIMT** (single instruction, multiple threads) model that ties those threads together, the **warp scheduler** that hides memory latency by swapping warps instead of by building big caches, the **occupancy** math that decides how many warps an SM can keep in flight, the **Tensor Cores** that make matmul special, and **warp divergence**, the tax you pay when your code branches. Figure 1 is the map of the chip we are about to take apart.

![layered diagram of an A100 GPU die showing GPCs, streaming multiprocessors, Tensor Cores, the register file and shared memory](/imgs/blogs/inside-the-gpu-sms-warps-and-the-simt-execution-model-1.png)

By the end you will be able to look at a kernel and predict, before you ever profile it, whether the GPU will love it or starve on it — and you will understand *why* the whole game of high-performance AI is to reshape your computation into dense, regular, predictable work. This is the second post in the series; the [intro](/blog/machine-learning/high-performance-computing/why-hpc-is-the-bottleneck-for-modern-ai) lays out the three walls (compute, memory bandwidth, communication), and the [capstone playbook](/blog/machine-learning/high-performance-computing/the-hpc-playbook-for-ai-engineers) puts every lever together. Here we go down to the silicon, because you cannot reason about the walls without first understanding the machine that hits them.

Throughout, we keep one running example: a single matrix multiply — a **GEMM** (general matrix-matrix multiply) — of the kind that dominates a Transformer. Every attention projection ($Q$, $K$, $V$, and the output), every feed-forward layer, every logit projection is a GEMM. When people say "a large language model is mostly matmul," they mean it literally: for a model like a 7B-parameter LLM, well over 95% of the floating-point operations live in a handful of GEMM shapes. So if we understand how one GEMM maps onto this chip, we understand where the time goes for the whole model.

## A GPU is not a fast CPU: two philosophies of silicon

Start with the design question every chip architect answers, because it explains everything downstream. You have a fixed transistor budget. Do you spend it making *one* stream of instructions finish as fast as possible, or making *many* streams of instructions finish a lot of total work per second? Those are two different machines, and you cannot have both with the same transistors.

A CPU answers "make one stream fast." It pours transistors into latency-hiding machinery for a *single* thread: deep out-of-order pipelines that reorder instructions around stalls, branch predictors that guess which way an `if` goes so the pipeline never empties, speculative execution that runs ahead before it knows the answer, and — above all — *enormous caches*. A modern server CPU might carry 30 to 50 MB of L3 cache, all dedicated to keeping the working set close so a load doesn't have to walk out to DRAM and wait ~200 nanoseconds. The CPU's whole strategy is: *avoid the wait.* Predict, cache, reorder, speculate — do whatever it takes so the one precious thread never stalls.

A GPU answers "make many streams do a lot of total work." It spends almost none of its transistors on latency tricks for any individual thread. There is no big branch predictor per lane, no deep out-of-order window, and the caches are tiny by CPU standards. Instead it spends the transistor budget on *arithmetic lanes* — thousands of them — and on the machinery to keep an enormous pool of threads in flight so that when one thread stalls on memory, there is always another ready to run. The GPU's strategy is the opposite of the CPU's: *don't avoid the wait — hide it behind other work.* Figure 2 puts the two philosophies side by side.

![before and after comparison of a latency-optimized CPU with fat cores and big caches versus a throughput-optimized GPU with thousands of lanes](/imgs/blogs/inside-the-gpu-sms-warps-and-the-simt-execution-model-2.png)

Put numbers on it. A high-end server CPU might field 64 cores at, say, a couple of FP64 TFLOP/s — trillion floating-point operations per second — and it will demolish the GPU on a single-threaded, branchy, pointer-chasing workload like parsing JSON or walking a linked list. An A100 fields **108 SMs with 64 FP32 cores each — 6,912 FP32 lanes** — and on a big dense matmul in bf16 it does **312 TFLOP/s** through its Tensor Cores, more than a hundred times the CPU's general throughput. But hand that same A100 a workload with no parallelism — one thread chasing pointers — and it is *slower* than the CPU, because a single GPU lane is a simple, in-order, modestly-clocked thing with no tricks to make it fast on its own.

This is the first law of using a GPU well, and it sounds almost too simple: **the GPU is only fast when you give it a mountain of independent, identical work.** Everything in the rest of this post — warps, occupancy, Tensor Cores, divergence — is a consequence of that one design choice. The chip was built to amortize latency over parallelism, so the moment your work isn't parallel and regular, the design works against you. If you want the formal version of "compute grew faster than bandwidth, so most work is memory-bound," that is the [roofline model](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound); here we are establishing *why* the silicon has the shape it does.

## The streaming multiprocessor: the real unit of compute

When you say "GPU," you are talking about a package that contains, on the A100, a single big die (the GA100) carved into 108 **streaming multiprocessors**. The SM is the unit that matters. The GPU is, to a first approximation, just 108 copies of the SM plus a memory system to feed them and a scheduler to hand them work. If you understand one SM, you understand the chip; the other 107 are doing the same thing on different data.

So what is inside one SM? On the A100, each SM contains:

- **64 FP32 CUDA cores** — the scalar arithmetic lanes that do an FP32 multiply-add per clock. ("CUDA core" is NVIDIA's marketing name for one of these lanes; a core is *not* a CPU core — it is closer to one lane of a SIMD unit.)
- **32 FP64 cores** — for double precision, which AI workloads rarely touch.
- **4 Tensor Cores** — specialized matrix-multiply units we will get to. Across 108 SMs that is **432 Tensor Cores**, the source of the 312 bf16 TFLOP/s headline.
- **4 warp schedulers**, each with its own dispatch unit — the brains that pick which warp runs next, one of the most important parts of the chip and almost invisible from software.
- **A 256 KB register file** — that is **65,536 32-bit registers** per SM, a startlingly large number that turns out to be the single most important resource budget on the chip.
- **192 KB of combined L1 cache and shared memory** — a small, fast, software-managed on-chip scratchpad we cover in depth in [the memory hierarchy post](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm).

The numbers in that list — 64 cores, 4 Tensor Cores, 4 schedulers, 65,536 registers — are not trivia. They are the *constraints* that decide how fast your kernel runs, and we will use every one of them before we are done. Notice already the strangeness: 65,536 registers but only 64 FP32 cores. The register file is vastly oversized relative to the cores. That is not a mistake. It is the latency-hiding strategy made physical, and we will see exactly why in a moment.

A quick sanity check you can run yourself. CUDA ships a sample called `deviceQuery`; running it on an A100 prints exactly these structural facts:

```bash
$ ./deviceQuery
Device 0: "NVIDIA A100-SXM4-80GB"
  CUDA Capability Major/Minor version number:    8.0
  Total amount of global memory:                 81251 MBytes
  (108) Multiprocessors, (064) CUDA Cores/MP:    6912 CUDA Cores
  GPU Max Clock rate:                            1410 MHz
  Total amount of shared memory per block:       49152 bytes
  Total number of registers per block:           65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
```

Read that output as a spec sheet for the machine you are about to program. 108 SMs ("Multiprocessors"), 64 CUDA cores each, warp size 32, 2048 threads (= 64 warps) max resident per SM, 65,536 registers per block. Those last three lines — warp size 32, 64 warps per SM, 65,536 registers — are the entire occupancy story, and we will derive it from exactly these numbers later in the post.

## The warp: 32 threads that move as one

Here is the idea that, once it clicks, makes the GPU make sense. **The hardware does not schedule threads one at a time. It schedules them in groups of 32 called warps, and all 32 threads in a warp execute the same instruction at the same time on different data.** That is the **SIMT** model — single instruction, multiple threads — and it is the beating heart of how a GPU works.

The name "warp" comes from weaving: a warp is the set of parallel threads stretched on a loom. NVIDIA borrowed it for the same reason — a warp is a bundle of parallel threads (32 of them, always, on every NVIDIA GPU to date) that move together. When the warp scheduler picks a warp to run, it fetches *one* instruction and broadcasts it to all 32 lanes. Lane 0 runs that instruction on its data, lane 1 runs the same instruction on its data, and so on through lane 31, all in the same clock cycle. Figure 3 shows the picture: one instruction up top, 32 lanes below, all executing it in lockstep.

![grid diagram of a warp showing one scheduler instruction broadcast across 32 thread lanes executing the same multiply-add](/imgs/blogs/inside-the-gpu-sms-warps-and-the-simt-execution-model-3.png)

Why 32 and not 64 or 16? It is a hardware design point — a balance between the cost of the control logic (fetch, decode, schedule) that gets *amortized* across the warp, and the granularity loss you suffer when you have fewer than a full warp of useful work. NVIDIA has used 32 for over fifteen years; AMD GPUs historically used 64 (they call it a "wavefront") and recent ones support 32. The exact number matters less than the consequence: **the GPU thinks in units of 32, so your code should too.** Launch 100 threads and the hardware still runs 4 warps — the last one with 28 of its 32 lanes masked off, doing nothing. Launch 1,024 threads and you get exactly 32 full warps, no waste.

SIMT is *almost* the same idea as SIMD (single instruction, multiple data) on a CPU — one instruction over a vector of data — but with a crucial software difference. In SIMD you, the programmer, explicitly load a vector register and call a vector instruction; the vectorization is visible in your code. In SIMT you write what looks like ordinary scalar code for a single thread — `int i = threadIdx.x; c[i] = a[i] * b[i];` — and the hardware *automatically* groups 32 of those scalar threads into a warp and runs them as a vector. The programming model is "many scalar threads"; the execution model is "vectors of 32." That gap between how you write it and how it runs is the source of both the GPU's ease of use and its sharpest performance traps. The single biggest trap — what happens when the 32 threads in a warp want to do *different* things — is warp divergence, and we will spend a whole section on it.

A first kernel makes this concrete. Here is the GPU's "hello world," a vector add, written in CUDA C:

```cuda
// Each thread computes ONE element of c = a + b.
// The hardware will group every 32 consecutive threads into a warp
// and run them in SIMT lockstep.
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    // Global index of THIS thread across the whole grid.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {                 // guard the tail; some threads in the
        c[i] = a[i] + b[i];      // last block have i >= n and must skip
    }
}

// Launch: 1,000,000 elements, 256 threads per block.
int n = 1'000'000;
int threads = 256;                          // = 8 warps per block
int blocks  = (n + threads - 1) / threads;  // = 3907 blocks
vector_add<<<blocks, threads>>>(a, b, c, n);
```

Three things to notice, because they recur in every CUDA kernel ever written. First, `threadIdx.x` and `blockIdx.x` are the coordinates the hardware hands each thread so it can figure out *which* element it owns — this is how a single piece of code becomes a million parallel computations. Second, `blockDim.x` is the threads-per-block you chose at launch (256 here = 8 warps). Third, the `if (i < n)` guard: because we launch in whole blocks and `n` is not a multiple of 256, the last block has threads with `i >= n` that must do nothing. That guard is the cheapest, most benign form of divergence — a handful of masked lanes in exactly one warp of the whole grid — and it does not hurt. We will see the malignant kind shortly.

## Latency hiding: why the GPU swaps warps instead of caching

Now we can answer the puzzle from earlier: why does one SM have a 256 KB register file — room for 65,536 registers — when it only has 64 FP32 cores? Because the register file's job is not to feed the 64 cores. Its job is to hold the live state of *many warps at once* so the SM can switch between them instantly.

Here is the problem the GPU has to solve. A load from HBM — the high-bandwidth memory, the GPU's main DRAM — takes on the order of **400 to 800 clock cycles** to come back. An arithmetic instruction takes a handful of cycles. So whenever a warp issues a load and then needs the result, it stalls for hundreds of cycles. A CPU would hide that stall with out-of-order execution and a big cache: reorder independent instructions to fill the gap, and try to have the data in cache so the load is fast in the first place. The GPU does neither. It has no big per-lane reorder window and only tiny caches. So how does it not just sit idle for 400 cycles on every memory access?

It hides the latency with *other warps.* This is the single most important mechanism on the chip, so it is worth stating precisely. Each SM keeps many warps **resident** at once — up to **64 warps** (2,048 threads) on an A100. The state of every resident warp — its registers, its program counter — lives on-chip permanently while the warp is active. That is what the giant register file is *for*: not to feed 64 cores from 64 registers, but to hold the live context of dozens of warps so that switching between them costs *zero cycles.* There is no save-and-restore, no context-switch overhead like an operating system pays. The warp's registers are already sitting in the file; the scheduler just changes which warp it issues from on the very next cycle.

So when Warp A issues a load and stalls for 400 cycles waiting on HBM, the scheduler does not wait. On the next cycle it issues an instruction from Warp B, which is ready. The cycle after, from Warp C. And so on. If the SM has enough ready warps, by the time it has cycled through all of them, Warp A's data has arrived and it is ready again. The 400-cycle stall has been completely *hidden* behind the useful work of the other warps — the SM never idled. Figure 4 shows the timeline: one warp stalls, the scheduler keeps the pipeline full from the others, and the stalled warp resumes once its data lands.

![timeline showing a warp scheduler swapping in ready warps each cycle while one warp waits on a long HBM memory load](/imgs/blogs/inside-the-gpu-sms-warps-and-the-simt-execution-model-4.png)

This is *why* the GPU loves a mountain of parallel work. The latency-hiding only functions if there are enough independent warps to fill the stall. Too few warps, and the SM runs out of ready work, stalls, and idles — the very thing the CPU's caches and out-of-order machinery exist to prevent, except the GPU chose not to build those, betting instead on parallelism. When you hear "the GPU hides latency with occupancy," this is the mechanism. The scheduler swaps warps; the register file makes the swap free; parallelism supplies the warps to swap to.

### The science: a Little's-law argument for how many warps you need

We can make "enough warps" quantitative, and it is one of the most useful back-of-the-envelope calculations in GPU programming. The question is: how many warps must an SM keep in flight to fully hide memory latency? This is exactly **Little's law** from queueing theory — the number of in-flight items equals the rate at which they arrive times how long each takes — applied to instructions.

Let $L$ be the latency we need to hide (cycles), and let each warp supply some amount of independent work between the stalls. In the simplest model, if a warp issues one instruction and then must wait $L$ cycles for it, and the SM can issue one instruction per cycle per scheduler, then to keep that scheduler busy for all $L$ cycles you need about $L$ *instructions* in flight from other warps. If each warp contributes one independent instruction at a time, you need on the order of

$$ W_{\text{needed}} \approx \frac{\text{latency to hide}}{\text{work issued per warp per turn}} . $$

Plug in numbers. Suppose memory latency is $L \approx 400$ cycles and each warp, when picked, issues one instruction before stalling again. Then you need roughly 400 instruction-slots of other work to fill the gap. If the SM has 4 schedulers each issuing one instruction per cycle, the SM as a whole consumes 4 instructions per cycle, so it burns through $4 \times 400 = 1600$ instruction-slots during one warp's stall. With each warp supplying a steady trickle of independent instructions, you need many tens of warps resident to never run dry — which is exactly why the hardware ceiling is **64 warps per SM** and why low occupancy hurts. The hardware was sized so that a fully occupied SM has just enough warps to hide HBM latency. Drop to 8 or 12 active warps and you no longer have enough independent work to cover a 400-cycle stall: the SM idles, and your kernel runs at a fraction of peak even though every individual operation is "correct."

#### Worked example: how many warps to hide a memory stall

Take a kernel whose every thread does one load, then a few arithmetic operations on the result, in a loop. Say the memory latency is $L = 480$ cycles and, thanks to instruction-level parallelism within a thread, each warp can keep about 4 independent instructions issuing before it must wait on the load. The SM has 4 schedulers issuing 1 instruction/cycle each, so it wants to dispatch $4 \times 480 = 1920$ instructions during the stall window. Each warp supplies 4 instructions per "turn," and a warp comes back around roughly every (number of warps) cycles. The fixed-point of that is approximately $W \approx L / (\text{instructions per warp per turn}) = 480 / 4 = 120$ warp-issue opportunities to fill — far more than one warp can supply, so you need a large fraction of the 64-warp ceiling resident. In practice you measure this: a memory-bound kernel that hits, say, 25% occupancy (16 of 64 warps) and is *latency-bound* will often speed up 1.5 to 2 times just by raising occupancy to 50%, because more resident warps means more independent loads in flight to hide each other's latency. That is the entire reason occupancy is a knob you tune. We will now make occupancy itself precise.

### Eligible vs stalled warps: what the scheduler actually sees each cycle

The Little's-law argument tells you *how many* warps you need; the next level of detail is *what the scheduler does with them on any given cycle*, and this is the view a profiler hands you when a kernel is slow for no obvious reason. At any instant, every warp resident on an SM is in one of three states. It is **selected** — the scheduler picked it this cycle and issued one of its instructions. It is **eligible** — its next instruction's operands are all ready and it *could* be issued this cycle, but the scheduler picked a different warp (only one issue per scheduler per cycle). Or it is **stalled** — its next instruction cannot issue yet because it is waiting on something: a memory load that has not returned, a previous arithmetic result not yet written back, a barrier (`__syncthreads()`) that other warps have not reached, or a pipeline that is busy.

Each of the A100's 4 warp schedulers owns a disjoint subset of the SM's warps (16 of the 64 slots each) and, on every clock, scans *its* warps for an eligible one and issues a single instruction from it. That is why there are 4 schedulers and not 1: with one scheduler the SM could issue at most one instruction per cycle, leaving most of the 64-wide FP32 datapath, the load/store units, and the Tensor Cores idle. Four independent schedulers let the SM issue up to 4 instructions per cycle — enough to keep the FP32 cores, the special-function units, and the Tensor Cores all fed in parallel from different warps. The arithmetic is clean: 64 FP32 cores divided across 4 schedulers is 16 lanes each, which is exactly half a warp, so a scheduler issues a 32-wide warp instruction to its 16 FP32 cores over two cycles — a deliberate design point that keeps the issue logic simple while the datapath stays busy.

The crucial insight for performance is the gap between "eligible" and "selected." If on most cycles a scheduler has *zero* eligible warps — every warp it owns is stalled — then the scheduler issues nothing that cycle, the SM does no work, and you are *latency-bound*: you simply do not have enough independent work in flight to cover the stalls. If, conversely, a scheduler almost always has *several* eligible warps (it issues every cycle and warps queue up waiting their turn), then you are *not* latency-bound — adding more occupancy will not help, because the scheduler is already issuing every cycle and the bottleneck is elsewhere (the math pipeline throughput, or a structural hazard). This single distinction — "are my schedulers starved for eligible warps, or are they saturated?" — decides whether raising occupancy will help at all, and you read it straight off the profiler.

#### Worked example: reading the stall-reason breakdown in `ncu`

When you run `ncu --section WarpStateStats`, the profiler reports, for the average warp, how many cycles it spent in each stall reason between issued instructions — the "warp state" histogram. This is the single most diagnostic table in GPU performance work, so it is worth knowing the common reasons by name and what each one tells you to do:

```bash
ncu --section WarpStateStats --section SchedulerStats \
    --kernel-name regex:"attn|gemm" python train.py

# Representative lines from a memory-bound kernel's report:
#   Warp Cycles Per Issued Instruction      18.4  cycles
#   Issued Warp Per Scheduler                0.31         <- < 1.0 means starved
#   Eligible Warps Per Scheduler             0.42         <- rarely >1 -> latency-bound
#   Stall Long Scoreboard                    11.2  cycles <- waiting on HBM loads
#   Stall Wait                                2.1  cycles <- waiting on math latency
#   Stall Barrier                             1.4  cycles <- waiting at __syncthreads()
#   Stall MIO Throttle                        0.9  cycles <- shared-mem/SFU queue full
```

Read that report top to bottom and it tells a complete story. **Issued Warp Per Scheduler** of 0.31 means each scheduler issued an instruction on only 31% of cycles — it was idle the other 69%, so the SM is badly underutilized. **Eligible Warps Per Scheduler** of 0.42 confirms *why*: on most cycles the scheduler had less than one eligible warp to choose from, so it had nothing to issue. The dominant stall reason, **Long Scoreboard** at 11.2 cycles, names the culprit: warps are stalled waiting on global-memory (HBM) loads whose results have not arrived. This is the textbook signature of a latency-bound, low-occupancy memory kernel — and the fix is exactly the one Little's law predicted: get more warps in flight (raise occupancy by cutting register or shared-memory pressure) so there are always eligible warps to hide the long-scoreboard stalls. If instead the dominant reason were **Stall Wait** (math-pipeline latency) with eligible warps already above 1, more occupancy would not help and you would look at instruction mix instead. Same report, opposite prescription — which is the whole reason you read the breakdown rather than guessing.

## Occupancy: how many warps you actually get, and why

**Occupancy** is the ratio of *active warps per SM* to the *maximum warps per SM the hardware allows* — on the A100, active warps divided by 64. It is the single most-quoted GPU tuning number, and it is widely misunderstood, so let us be precise: occupancy is not a goal in itself, it is the *supply of warps available to hide latency.* High occupancy gives the scheduler more ready warps to swap to; that is its only job. You do not need 100% occupancy to go fast (a compute-bound kernel near peak may run great at 50%), but if a kernel is latency-bound, low occupancy is usually why.

The crucial fact is that **you rarely get all 64 warps.** Each resident warp consumes finite SM resources, and you run out of *something* before you fill all 64 slots. There are four limiters, and your actual occupancy is the *minimum* across all of them — the scarcest resource wins. Figure 5 lays them out as a resource-by-limit table.

![matrix showing how registers per thread, shared memory per block, and warp slots each cap occupancy on an A100 streaming multiprocessor](/imgs/blogs/inside-the-gpu-sms-warps-and-the-simt-execution-model-5.png)

The four limiters:

1. **The hard warp-slot ceiling.** The SM can hold at most 64 warps (2,048 threads) no matter what. This is the cap occupancy is measured against.
2. **Registers per thread.** The SM has 65,536 registers total. If your kernel uses $R$ registers per thread, then the SM can host at most $65536 / R$ threads. Divide by 32 to get warps.
3. **Shared memory per block.** The SM has ~164 KB of shared memory usable per block on the A100 (out of 192 KB combined L1/shared). If each block requests $S$ bytes of shared memory, at most $\lfloor 164\,\text{KB} / S \rfloor$ blocks fit, and each block contributes its warps.
4. **Block-count and thread-per-block limits.** There are caps on blocks per SM (32 on the A100) and threads per block (1,024); these rarely bind first but can.

The one that bites most often in real AI kernels is **registers**, and it is worth burning the arithmetic into memory because it is the most common reason a kernel mysteriously won't hit full occupancy.

### The science: deriving occupancy from the register budget

The register equation is just integer division against a fixed budget. The SM has $65{,}536$ 32-bit registers. A warp is 32 threads, so a warp's register footprint is $32 \times R$ where $R$ is registers per thread. The maximum number of warps the register file can host is therefore

$$ W_{\text{reg}} = \left\lfloor \frac{65536}{32 \cdot R} \right\rfloor = \left\lfloor \frac{2048}{R} \right\rfloor . $$

And occupancy from registers is $\min(W_{\text{reg}}, 64) / 64$. Now read off the cliff. If your kernel uses $R = 32$ registers per thread, $W_{\text{reg}} = \lfloor 2048/32 \rfloor = 64$ warps — full occupancy, registers don't bind. Push to $R = 64$ registers and $W_{\text{reg}} = \lfloor 2048/64 \rfloor = 32$ warps — you just halved your maximum occupancy to 50%, because each thread now hogs twice the register file. At $R = 128$ registers (common in a heavily-unrolled or high-precision kernel) you get $W_{\text{reg}} = 16$ warps, 25% occupancy. The register count per thread is set by the compiler based on how much state your kernel keeps live; a fatter kernel with more live variables uses more registers and gets fewer warps.

This is the central tension of kernel tuning and it is genuinely a trade-off, not a free lunch: more registers per thread can make each *thread* faster (more values kept in fast registers, fewer spills to slow memory), but it lowers occupancy, which can leave the SM with too few warps to hide latency. The right answer depends entirely on whether the kernel is latency-bound (favor more warps) or already saturating the math units (favor faster threads). You do not guess this — you measure it.

#### Worked example: occupancy of a Transformer feed-forward kernel

Take the feed-forward block of a Transformer, the part that does $\text{GELU}(xW_1)W_2$. Suppose you write (or the compiler generates) a fused kernel for the elementwise GELU activation that uses $R = 40$ registers per thread and requests $S = 8$ KB of shared memory per block, with 256 threads (8 warps) per block. Walk the four limiters on an A100:

- **Warp slots:** 64 warps. (The ceiling.)
- **Registers:** $W_{\text{reg}} = \lfloor 2048 / 40 \rfloor = 51$ warps. Round down to a whole number of blocks: each block is 8 warps, so $\lfloor 51 / 8 \rfloor = 6$ blocks = 48 warps.
- **Shared memory:** $\lfloor 164\,\text{KB} / 8\,\text{KB} \rfloor = 20$ blocks — way more than enough, not binding.
- **Blocks per SM:** 32 max — not binding at 6 blocks.

The minimum is the register limit: **48 active warps out of 64 = 75% occupancy.** That is fine for a memory-bound elementwise kernel — 48 warps supply plenty of independent loads to hide HBM latency. But suppose a code change (say, switching to a higher-precision accumulation path) pushes register usage to $R = 72$. Now $W_{\text{reg}} = \lfloor 2048/72 \rfloor = 28$ warps, $\lfloor 28/8 \rfloor = 3$ blocks = 24 warps = **37.5% occupancy.** If the kernel was latency-bound, that change just slowed it down even though it made each thread "more accurate," and the only way you'd know is by profiling occupancy before and after. The general lesson: occupancy is an emergent property of your resource usage, and a one-line code change can move it dramatically.

You query the achieved occupancy of a real kernel with NVIDIA's profiler, Nsight Compute (`ncu`):

```bash
# Profile one kernel and dump the occupancy + launch stats. --set full
# runs the full section set (roofline, occupancy, memory). Here we ask
# only for the occupancy and launch-configuration sections to keep it fast.
ncu --section Occupancy --section LaunchStats \
    --kernel-name regex:"ffn|gelu" \
    python train.py

# Relevant lines in the report:
#   Achieved Occupancy            48.2  %     <- what you actually got
#   Theoretical Occupancy         75.0  %     <- the resource-limited ceiling
#   Registers Per Thread          40         <- why it's capped at 75
#   Block Limit Registers         6  block    <- the binding limiter
```

Read those four lines together and you have the whole story: theoretical occupancy is what the resource math allows (here 75%, register-limited as we derived), and achieved occupancy is what the kernel actually averaged at runtime (48%, lower because of tail effects and uneven warp lifetimes). When achieved is far below theoretical, you have a load-balance or launch-configuration problem; when theoretical itself is low, you have a resource problem — too many registers or too much shared memory per thread. We go much deeper on reading `ncu` and `nsys` traces in the [profiling post](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound); here the point is just that occupancy is a *derivable, measurable* quantity, not a vibe.

### Register pressure and spilling: the cliff below the occupancy cliff

There is a second, sharper failure mode hiding inside the register budget, and it catches people who fix occupancy but not its cause. The register count per thread is not unbounded: the hardware caps a single thread at **255 registers** on the A100, and the compiler tries to keep a kernel's usage well below that to preserve occupancy. But when a kernel keeps a lot of state live at once — a deeply unrolled loop, a big accumulator tile, many temporaries that all overlap in lifetime — the compiler can run out of registers to assign. When that happens it **spills**: it picks some live values to evict from registers and stash in "local memory." The name is a trap. *Local memory is not on-chip.* Despite the friendly name, local memory is a per-thread region that physically lives in **HBM** (cached through L1/L2, but backed by DRAM). So a register spill turns what should have been a 1-cycle register access into a potential several-hundred-cycle round trip to device memory, on a value the kernel touches repeatedly. Spilling is one of the few things that can make a kernel *slower* than a naive version while looking innocent in the source.

The compiler will tell you about spills if you ask. Compile with `-Xptxas -v` (or read the `ptxas` info that `nvcc`/Triton emit) and you get a per-kernel line like this:

```bash
nvcc -Xptxas -v -arch=sm_80 my_kernel.cu
# ptxas info: Compiling entry function '_Z10my_kernelPfS_i' for 'sm_80'
# ptxas info: Used 168 registers, 384 bytes cmem[0], 96 bytes lmem  <- lmem != 0!
#                                                       ^^^^^^^^^^^
#       96 bytes of local memory per thread = SPILLS. Each spilled load/store
#       is an HBM round trip. Either cut live state or cap registers and
#       accept lower per-thread speed in exchange for no spills.
```

The non-zero `lmem` (local memory) figure is the red flag: any local memory means the compiler spilled. You have three levers. You can *reduce live state* (shorten unroll factors, recompute cheap values instead of holding them, tile smaller) so the kernel naturally fits in registers. You can *cap registers* with the `__launch_bounds__` qualifier or `-maxrregcount`, which forces the compiler to stay under a register ceiling — but be careful, because if the live state genuinely does not fit, capping registers just *causes* spills to make room, trading one problem for the other. Or you can decide the per-thread speed is worth it and accept the lower occupancy. The right call is, as always, measured: a kernel that spills but keeps high occupancy may still beat a spill-free version at lower occupancy, or it may not — you compare wall-clock times, you do not theorize.

#### Worked example: max occupancy at 32, 64, and 128 registers per thread

Walk the register cliff with three concrete numbers, because seeing it as a table makes the trade-off impossible to forget. The SM has 65,536 registers and a 64-warp ceiling; a thread uses $R$ registers, a warp is 32 threads, so the register-limited warp count is $W_{\text{reg}} = \lfloor 2048 / R \rfloor$, capped at 64:

| Registers per thread $R$ | Warps the register file allows $\lfloor 2048/R \rfloor$ | Max occupancy (of 64) | What it means |
|---|---|---|---|
| 32 | 64 | 100% | Registers do not bind; the warp-slot ceiling is the only limit. |
| 64 | 32 | 50% | Each thread hogs twice the file; occupancy halves. |
| 128 | 16 | 25% | A heavy, unrolled, or high-precision kernel; only 16 warps to hide latency. |
| 255 (hardware max) | 8 | 12.5% | The absolute floor — a single thread monopolizing registers. |

The shape of the table is the lesson: occupancy falls roughly as $1/R$, and it falls in *steps* because warps must round down to whole blocks. A kernel sitting at $R = 64$ (50% occupancy) that adds just a few live variables and tips to $R = 72$ drops to $\lfloor 2048/72 \rfloor = 28$ warps — and if its block size is 8 warps, that rounds to 3 blocks = 24 warps = **37.5%**, a visible cliff from a one-line change. Worse, if instead of accepting fewer warps the compiler is told to hold the line at 64 registers but the kernel genuinely needs 72, it spills 8 registers' worth of state to local memory (HBM) — now you have 50% occupancy *and* spill traffic, the worst of both. This is why register pressure is the quiet killer of fused kernels: the moment you fuse more work into one kernel to save memory traffic, you raise live state, and live state is registers, and registers are occupancy. The art of writing a fast fused kernel is keeping it under the register cliff while still doing enough work per launch to amortize the launch — a balance you can only find by reading the `ptxas` register count and the achieved occupancy together.

## Tensor Cores: why matmul is special

Everything so far — warps, latency hiding, occupancy — applies to *general* GPU code running on the CUDA cores. But the headline number on a modern GPU, the 312 bf16 TFLOP/s on the A100, does not come from the CUDA cores at all. It comes from the **Tensor Cores**, and understanding them is understanding why deep learning got fast.

A CUDA core does a **FMA** — a fused multiply-add, $d = a \times b + c$ — one scalar operation per lane per clock. That is the atom of general arithmetic. A Tensor Core does something categorically bigger: a **MMA**, a matrix multiply-accumulate, $D = A \times B + C$ where $A$, $B$, $C$, $D$ are small *matrices* (a tile of, say, $16 \times 16$), all in a single instruction. Instead of one multiply-add, a Tensor Core performs a whole tile's worth of multiply-adds — hundreds of them — per instruction. Figure 6 puts the scalar FMA and the matrix MMA side by side.

![before and after comparison of a CUDA core doing one scalar fused multiply-add versus a Tensor Core doing a full matrix tile multiply-accumulate](/imgs/blogs/inside-the-gpu-sms-warps-and-the-simt-execution-model-6.png)

Why does this win so enormously? Because a matrix multiply has *colossal arithmetic reuse* baked into its structure, and dedicated hardware can exploit it where scalar cores cannot. When you compute a tile of $C = A \times B$, every element of $A$ gets multiplied by a whole row of $B$, and every element of $B$ by a whole column of $A$ — each input value is reused across many multiply-adds. A scalar FMA pipeline has to re-fetch operands and pay instruction-issue overhead for *every single* multiply-add. A Tensor Core is built as a small systolic-style array that loads the tile once and streams the multiply-adds through dedicated wiring, amortizing all that overhead across the whole tile. The result on the A100: the FP32 CUDA cores deliver about 19.5 TFLOP/s for general FP32 math, while the Tensor Cores deliver **312 TFLOP/s in bf16** — roughly **16 times** the throughput on exactly the dense regular matmul that dominates a Transformer.

That ratio is the whole reason mixed precision and Tensor Cores took over training. But the Tensor Core only fires under specific conditions, and missing them is one of the most common ways AI engineers silently leave a 10-times speedup on the floor.

### The science: counting Tensor Core FLOPs

Let us verify the 312 TFLOP/s figure is internally consistent, because doing the arithmetic teaches you where the number comes from. A Tensor Core's throughput is (operations per instruction) times (instructions per cycle) times (clock) times (number of Tensor Cores). On the A100, the third-generation Tensor Core does a $16 \times 8 \times 16$ MMA shape for bf16 in hardware terms; the useful way to think about it is total multiply-adds per cycle. The A100 has 432 Tensor Cores running at about 1.41 GHz. Each multiply-add is 2 floating-point operations (one multiply, one add). The published bf16 peak is

$$ \text{TFLOP/s} = (\text{MACs per TC per cycle}) \times 432 \times 1.41 \times 10^9 \times 2 . $$

Working backward from the published 312 TFLOP/s: $312 \times 10^{12} / (432 \times 1.41 \times 10^9 \times 2) \approx 256$ multiply-adds per Tensor Core per cycle. That is the order of a $16 \times 16$ tile's worth of work per core per cycle — consistent with the architecture. The exact per-shape decomposition is in the [A100 whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf); the point of the back-of-envelope is that **312 TFLOP/s is 432 Tensor Cores each doing on the order of 256 multiply-adds per cycle at 1.4 GHz**, and that is roughly 16 times what the 64-per-SM FP32 cores can do. The number is not magic; it is dense-matmul hardware multiplied out.

### The MMA shapes and precision ladder: what the instruction actually computes

A Tensor Core is not invoked one element at a time; it is invoked through a **warp-level MMA instruction** (in PTX, `mma.sync` and, since the Ampere generation, the asynchronous `wmma`/`ldmatrix` family; on Hopper, the warpgroup-level `wgmma`). The defining feature is that the *whole warp* cooperates to feed one matrix multiply-accumulate. The threads of the warp jointly hold the input tiles in their registers (laid out in a specific, hardware-mandated pattern), the Tensor Core consumes them, and the result tile is written back across the warp's registers. This is why you almost never write `mma.sync` by hand — getting the register layout right is fiddly — and instead let cuBLAS, CUTLASS, or Triton emit it.

The shapes are fixed, small, and named as $M \times N \times K$ — the output tile is $M \times N$ and the contracted dimension is $K$. On the A100 the third-generation Tensor Core exposes shapes like $16 \times 8 \times 16$ and $16 \times 8 \times 8$ for fp16/bf16, with the larger contraction depths for the lower-precision modes. On the H100, `wgmma` works at the *warpgroup* level (4 warps = 128 threads cooperating) and supports much wider shapes like $64 \times 256 \times 16$, which is part of how Hopper extracts more throughput per instruction — fewer, fatter instructions mean less issue overhead per multiply-add. The precise legal shapes per precision are tabulated in the PTX ISA documentation; the practical fact to remember is that they are *always multiples of 8 in every dimension*, and usually 16.

The precision ladder is the other half of the story, because the same Tensor Core does different things at different widths, and throughput roughly doubles each rung you descend:

- **TF32** — a 19-bit format with fp32's 8-bit exponent but only a 10-bit mantissa. It runs fp32-*shaped* matmuls on the Tensor Cores at ~**156 TFLOP/s** on the A100, about 8x true FP32, with accuracy good enough for training in practice. It is the "free" path: flip a flag, change no tensors.
- **fp16 / bf16** — the 16-bit training formats, ~**312 TFLOP/s** on the A100. bf16 keeps fp32's exponent range (fewer overflow headaches) at the cost of mantissa bits; fp16 has more mantissa but a narrow range that often needs loss scaling. Both run the same Tensor Core path at the same peak.
- **int8** — ~**624 TOP/s** on the A100, the inference-quantization path. Same silicon, narrower operands, double the throughput again.
- **fp8** — *not* on the A100; introduced on the H100's fourth-generation Tensor Cores (E4M3 and E5M2 variants), reaching ~**1,979 TFLOP/s** dense. This is the rung the Transformer Engine climbs to during H100 training.

Each step down the ladder roughly doubles throughput because the Tensor Core can pack more, narrower multiply-adds through the same wiring per cycle. That doubling is the entire economic argument for quantization and low-precision training: the math units are *built* to reward you for asking for fewer bits.

### Structured sparsity: the 2:4 pattern that doubles throughput again

There is one more Tensor Core feature that is easy to miss and worth a paragraph because it is a literal free 2x. The Ampere (and later) Tensor Core supports **2:4 structured sparsity**: if you prune the weight matrix so that in every contiguous group of 4 values, *at most 2 are non-zero*, the hardware can skip the zeros and run the matmul at **double the dense rate** — the 312 bf16 TFLOP/s headline becomes **624 TFLOP/s** for a 2:4-sparse weight. The "structured" part is what makes it hardware-friendly: a general sparse matrix has irregular non-zero locations that wreck the dense, regular access pattern the Tensor Core needs, but a *fixed* 2-of-4 pattern can be encoded with a small metadata index per group, and the Tensor Core's datapath is built to consume exactly that. The matrix is stored compressed (the 2 non-zeros plus a 2-bit index of where they sat among the 4), so you also halve the memory footprint and bandwidth for that operand.

The catch — and the reason it is not universal — is that 2:4 is a real constraint on the weights, so you must prune and usually fine-tune the model to recover accuracy under that pattern; it is a training-time decision, not a runtime flag. When it works, it stacks with low precision: a 2:4-sparse int8 matmul on the A100 reaches the **624 TOP/s** dense-int8 rate doubled again. It is the cleanest example in the whole chip of the central thesis — *the hardware pays you, in throughput, for making your work more regular* — because all you did was agree to a regular sparsity structure and the silicon handed back a 2x.

### Why the shapes must be multiples of 8 (and ideally 16 or 64)

This now explains the alignment rule that bites people, and it is not arbitrary. Because the Tensor Core consumes fixed tiles whose every dimension is a multiple of 8 (often 16), a GEMM whose $M$, $N$, or $K$ is *not* a multiple of that tile size cannot fill its boundary tiles. The kernel has two bad options at the ragged edge: pad the partial tile with zeros (computing multiply-adds you then throw away — wasted FLOPs and wasted lanes), or fall back to a slower non-Tensor-Core path for the remainder. Either way, the closer your dimension is to *just over* a tile boundary, the more you waste: a $K$ of 4097 is nearly as expensive as 4104 or 4112 because it forces a whole extra tile that is almost entirely padding. This is why a hidden size of 4096 (a clean multiple of 64) is friendly and a vocabulary of 50,257 is hostile, and why the standard fix is to pad odd dimensions up to the next multiple of 64 — which we demonstrate next.

### Practical: how to actually hit the Tensor Core path in PyTorch

This is the part that bites people. You can have an A100, write a perfectly correct matmul in PyTorch, and run it entirely on the slow FP32 CUDA cores at ~19 TFLOP/s while the 312-TFLOP/s Tensor Cores sit idle — because you did not meet the conditions. The Tensor Cores fire only when:

1. **The data type is right.** Tensor Cores accelerate fp16, bf16, tf32, int8, and (on Hopper) fp8 — *not* plain fp32. If your tensors are fp32 and you have not enabled TF32 or autocast, you get the CUDA-core path. The fix is mixed precision; details and the numerics of why bf16 is the safe default are in [the precision post](/blog/machine-learning/high-performance-computing/numerical-formats-and-mixed-precision-fp32-tf32-bf16-fp16-fp8), and the int8 path is exactly what [quantization](/blog/machine-learning/edge-ai/quantization-from-first-principles) targets to squeeze more matmul throughput out of the same silicon.
2. **The shapes are aligned.** Tensor Cores work on tiles. If your matrix dimensions are not multiples of 8 (for fp16/bf16) — ideally 16 or 64 — the kernel falls back to a slower path or wastes lanes padding the ragged edge. A hidden size of 768 (divisible by 8) is friendly; a vocabulary of 50,257 (a prime-ish odd number) is not, and the final logit GEMM pays for it.

Here is the difference in code:

```python
import torch

# A Transformer-ish GEMM: (batch*seq, hidden) @ (hidden, 4*hidden)
M, K, N = 8192, 4096, 16384

# --- SLOW PATH: fp32 inputs, no TF32 -> runs on FP32 CUDA cores (~19 TFLOP/s)
torch.backends.cuda.matmul.allow_tf32 = False
a32 = torch.randn(M, K, device="cuda", dtype=torch.float32)
b32 = torch.randn(K, N, device="cuda", dtype=torch.float32)
c32 = a32 @ b32        # FP32 cores only; Tensor Cores idle

# --- FAST PATH: bf16 inputs -> runs on Tensor Cores (~312 TFLOP/s peak)
a16 = a32.to(torch.bfloat16)
b16 = b32.to(torch.bfloat16)
c16 = a16 @ b16        # Tensor Core MMA path

# --- ALSO FAST, no code change to tensors: enable TF32 for fp32 matmuls.
# TF32 runs fp32-shaped matmuls on the Tensor Cores at reduced mantissa,
# ~8x faster than true FP32 with negligible accuracy loss for training.
torch.backends.cuda.matmul.allow_tf32 = True
c_tf32 = a32 @ b32     # now uses Tensor Cores via TF32
```

And the shape rule, demonstrated:

```python
# Tensor Cores want dimensions divisible by 8 (bf16/fp16). A vocab of
# 50257 is hostile; padding it to 50304 (= 8 * 6288) restores the fast path.
import torch

hidden, vocab = 4096, 50257
# Pad the vocab dimension up to a multiple of 8 (here also a multiple of 64).
vocab_padded = (vocab + 63) // 64 * 64          # 50304
logits_w = torch.randn(hidden, vocab_padded, device="cuda", dtype=torch.bfloat16)

x = torch.randn(8192, hidden, device="cuda", dtype=torch.bfloat16)
logits = x @ logits_w                            # clean Tensor Core tiles
# Slice off the padding for the loss; the extra columns are never trained.
logits = logits[:, :vocab]
```

That `50257 -> 50304` padding is exactly the trick the GPT-NeoX and nanoGPT codebases use, and it is worth a measurable few-percent end-to-end speedup on the final projection for free. The general principle — **shape your tensors so the dense regular matmul tiles cleanly onto the Tensor Cores** — is the single highest-leverage thing you can do to make a Transformer fast on one GPU, and it falls straight out of understanding what the silicon wants.

## Warp divergence: the tax on branchy code

We have built up to the GPU's sharpest trap. Recall the SIMT contract: all 32 threads of a warp execute *one* instruction together. That contract is beautiful when all 32 threads want to do the same thing. It becomes a tax the moment they want to do *different* things — when your code branches on data, and some lanes take the `if` while others take the `else`. This is **warp divergence**, and it is the reason branchy code is slow on a GPU even when it would be fine on a CPU.

Here is the mechanism, stated exactly. When a warp hits a data-dependent branch — say `if (x[i] > 0)` where some lanes have positive `x[i]` and some don't — the hardware cannot run two different instructions in the same warp at the same time. It only has one instruction fetch per warp per cycle. So it does the only thing it can: it runs the `if` path with the lanes that took it *active* and the others *masked off* (idle, producing no work), then runs the `else` path with the masks flipped. The two paths are executed **serially**, one after the other, and on each pass some lanes sit idle. The warp's effective throughput drops by the fraction of lanes masked. Figure 7 contrasts a convergent warp (all 32 lanes busy, one pass) with a divergent one (16 active per pass, two serial passes).

![before and after comparison of a convergent warp running at full lane utilization versus a divergent warp serializing two branch paths at half utilization](/imgs/blogs/inside-the-gpu-sms-warps-and-the-simt-execution-model-7.png)

The cost is bounded but real. A simple two-way branch where the warp splits in half costs up to **2 times** (two serial passes, half the lanes idle on each). A branch with $k$ distinct paths taken within one warp can cost up to $k$ times. The worst case is a `switch` or a long `if/else if` chain where every lane in the warp takes a different arm — the warp serializes into many passes and your 32-wide vector machine degenerates into something close to scalar.

But — and this is the subtlety that separates people who understand the hardware from people who fear it — **divergence is only a tax when threads in the *same warp* diverge.** If all 32 threads of a warp take the same branch, there is *no* divergence and *no* penalty, even though the code contains an `if`. The branch resolved the same way for the whole warp, so the warp just runs one path at full width. This is why the `if (i < n)` tail guard in our vector-add kernel was free: it diverges in at most one warp of the entire grid (the last one), and even there only a few lanes are masked once.

The practical rule that falls out: **structure data so that threads which branch the same way live in the same warp.** Divergence that aligns to warp boundaries (warp 0 all takes path A, warp 1 all takes path B) is free; divergence *within* a warp (lane 0 takes A, lane 1 takes B) is the tax. This is why, for example, sorting or bucketing data by its branch condition before a kernel can turn a divergent kernel into a convergent one.

Here is a kernel that deliberately diverges, so you can see the shape of the problem:

```cuda
// BAD: data-dependent branch that splits warps. If x[i] alternates sign
// across neighboring lanes, every warp diverges and serializes both arms.
__global__ void divergent_relu_special(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (x[i] > 0.0f) {
        // expensive path A: some transcendental work
        y[i] = expf(x[i]) - 1.0f;     // taken by SOME lanes in the warp
    } else {
        // expensive path B: different transcendental work
        y[i] = -logf(1.0f - x[i]);    // taken by the OTHER lanes
    }
    // Within a warp where signs are mixed: BOTH paths run, serially.
}

// BETTER: branchless / predicated form. The compiler computes both cheap
// sides and selects, so there is one instruction stream and no serialized
// passes. Only worth it when both sides are cheap; otherwise reduce the
// divergence by reorganizing data so a warp's lanes agree on the branch.
__global__ void predicated_form(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xi = x[i];
    float a = expf(xi) - 1.0f;
    float b = -logf(1.0f - xi);
    y[i] = (xi > 0.0f) ? a : b;   // select, not branch -> no warp divergence
}
```

The predicated version is not always a win — it computes *both* sides for every lane, so it only pays off when each side is cheap. The deeper fix for expensive divergent work is data layout: arrange the input so that a whole warp's 32 lanes resolve the branch the same way. Either way, the engineering instinct you want is: **a branch on data, inside a warp, is a performance bug until proven otherwise.**

#### Worked example: measuring the divergence penalty in microseconds

Let us put a microsecond number on it, the way you would in a microbenchmark, so the tax stops being abstract. Take 64 million elements and a kernel where every thread does the same chunk of transcendental work, but in two variants: a *convergent* version where the branch resolves identically for all 32 lanes of every warp (we arrange the input so signs are blocked by warp), and a *divergent* version where the sign alternates lane-by-lane so every warp splits 16/16. On an A100 you would time them with CUDA events, warming up and synchronizing properly:

```python
import torch

n = 64 * 1024 * 1024
x = torch.randn(n, device="cuda")

def timed(fn, x, iters=50):
    # Warm up so the kernel is compiled/cached and clocks are up.
    for _ in range(10):
        fn(x)
    torch.cuda.synchronize()                 # finish all pending work
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(x)
    end.record()
    torch.cuda.synchronize()                 # MUST sync before reading time
    return start.elapsed_time(end) / iters   # ms per call

# Convergent: same op for all lanes. Divergent: branchy elementwise that
# splits warps. (Custom CUDA/Triton kernels would make the split explicit.)
```

The numbers you would observe on an A100 for a kernel like this are in the table below — and they are representative, not a single canonical published figure, so treat them as approximate order-of-magnitude results from the kind of microbenchmark you would run yourself:

| Variant | Lane utilization | Passes per warp | Kernel time (approx) | Slowdown |
|---|---|---|---|---|
| Convergent (warp-aligned branch) | 32 / 32 | 1 | ~190 µs | 1.0x baseline |
| Divergent (16/16 split per warp) | 16 / 32 | 2 | ~360 µs | ~1.9x |
| Fully divergent (worst case) | ~1 / 32 effective | up to 32 | many times slower | up to ~kx |

The 16/16 split costs essentially the predicted ~2 times. The key takeaway is that the penalty is *bounded by the number of distinct paths within a warp* and is paid in extra serial passes — so it is real, measurable in microseconds, and entirely avoidable by aligning branches to warp boundaries or going branchless. This is the concrete face of "throughput hardware hates irregular work."

## The execution hierarchy: grid, block, warp, thread

We have met the pieces; now assemble the mental model of how a kernel launch becomes work on the silicon, because this is the framework you reason inside every time you choose a launch configuration. CUDA exposes a four-level hierarchy, and each level maps onto the hardware in a specific way. Figure 8 shows the mapping as a dataflow from the grid down to individual threads.

![graph showing a CUDA grid splitting into thread blocks that schedule onto SMs and run as warps of 32 threads](/imgs/blogs/inside-the-gpu-sms-warps-and-the-simt-execution-model-8.png)

From the top:

- **Grid.** When you launch a kernel (`my_kernel<<<blocks, threads>>>(...)`), you launch a *grid* of thread blocks — the whole batch of parallel work. A grid for a big GEMM might be thousands of blocks. The grid is a software concept; it is the total problem.
- **Block (thread block).** The grid is divided into blocks of up to 1,024 threads. A block is the unit of *scheduling and resource allocation*: the hardware assigns a whole block to a single SM, and the block's threads can cooperate via shared memory and synchronize with `__syncthreads()`. Crucially, all threads in a block run on *one* SM and stay there for the block's lifetime. The grid's blocks are distributed across the 108 SMs.
- **Warp.** Inside an SM, each block's threads are automatically partitioned into warps of 32 (in order: threads 0–31 are warp 0, 32–63 are warp 1, and so on). The warp is the unit of *execution and scheduling* — the warp scheduler issues instructions one warp at a time, and the warp is what gets swapped to hide latency.
- **Thread.** The leaf: one lane, with its own registers and its own `threadIdx`. A thread is the unit your code is written from the perspective of, but it never runs alone — it always runs as one of 32 lanes in a warp.

The reason this hierarchy matters for performance is that **your launch configuration — how many threads per block — sets your occupancy and your warp alignment**, both of which we have now seen decide speed. Choose a block size that is a multiple of 32 (always) and large enough to give the SM enough warps to hide latency, but not so large that registers or shared memory cap you below full occupancy. A block of 128 or 256 threads (4 or 8 warps) is the common sweet spot for a reason: it tiles cleanly into the 64-warp ceiling and gives the scheduler enough warps to swap. A block of 100 threads is a bug — it rounds up to 4 warps with 28 lanes wasted in the last one. A block of 1,024 threads (32 warps) means only two blocks fit per SM, which can hurt if one block stalls.

Tie it back to the Transformer GEMM that has been our running example. When PyTorch (via cuBLAS) runs $Q = xW_Q$, it launches a grid where each block computes one *tile* of the output matrix. Each block loads a tile of $x$ and a tile of $W_Q$ into shared memory (the on-chip scratchpad), then the warps in the block march those tiles through the Tensor Cores with MMA instructions, accumulating the output tile. The block size, tile shape, and shared-memory usage are all tuned (by cuBLAS, or by you in a [fused kernel](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm)) so that the matmul tiles map cleanly onto the Tensor Cores at high occupancy with no divergence. That is the whole machine working in concert: a grid of blocks, distributed to SMs, running as warps, feeding dense regular tiles to Tensor Cores. Every word in this post is a piece of that sentence.

#### Worked example: mapping a 4096×4096×4096 GEMM onto the SMs

Make the mapping fully concrete with one shape. Take a square GEMM $C = A \times B$ with $M = N = K = 4096$ — a plausible attention-projection or MLP sub-block for a mid-size model. We want to see how this single matmul becomes thousands of warps spread across 108 SMs. A typical cuBLAS tiling for a GEMM this size uses an output tile of, say, **128×128** per thread block (the exact tile is heuristic-chosen and shape-dependent, but 128×128 is a common, representative choice). Each block is responsible for computing one 128×128 patch of the 4096×4096 output $C$.

How many blocks is that? The output is $4096 \times 4096$, tiled into $128 \times 128$ patches, so the grid is $(4096/128) \times (4096/128) = 32 \times 32 = $ **1,024 thread blocks**. Each block walks the shared $K = 4096$ dimension in steps of its $K$-tile (say 32 or 64 at a time), loading a strip of $A$ and a strip of $B$ into shared memory each step and accumulating into its 128×128 output tile in registers — the classic tiled-GEMM inner loop. Inside one block, the 128×128 output tile is divided among the block's warps: with, say, 8 warps per block (256 threads), each warp owns a 32×64-ish sub-tile and drives the Tensor Core MMAs for it. So the hierarchy is: **1 GEMM → 1,024 blocks → ~8 warps each → 32 lanes each**, and at the leaf, those lanes cooperatively feed 16×8×16 MMA tiles to the Tensor Cores.

Now the occupancy and scheduling picture. With 1,024 blocks and only 108 SMs, each SM gets roughly $1024 / 108 \approx 9$–10 blocks to chew through over the kernel's life. But they do not all run at once: a 128×128 fp16 tile with double-buffered shared memory might use most of an SM's ~164 KB shared-memory budget for 1–2 *resident* blocks at a time, so the SM holds a couple of blocks concurrently (enough warps to hide HBM latency while loading the next $K$-strip) and streams through its ~9–10 assigned blocks in waves. The total work is $2 \times 4096^3 \approx 1.37 \times 10^{11}$ FLOPs (the factor of 2 is multiply plus add); on an A100 at ~250 achieved bf16 TFLOP/s that is about **0.55 milliseconds** of compute — and because the shape is a clean multiple of 128 in every dimension, every tile is full, no lanes are wasted on ragged edges, and the GEMM runs near peak. Change $K$ to 4097 and one extra, almost-empty tile-strip appears along the contraction for *every* output block, which is exactly the alignment tax the Tensor Core section warned about. This is the entire chip in one example: a dense regular matmul fragments into a grid of full tiles, distributes across all 108 SMs, runs as high-occupancy warps with no divergence, and feeds the Tensor Cores their preferred dense work — the near-perfect customer.

## Putting it together: why throughput hardware loves dense, regular work

Step back and look at what the hardware has been telling us. Four design choices, each a consequence of "spend transistors on throughput, not latency," and each with a corresponding rule for how to feed the chip:

1. **It executes in warps of 32 (SIMT).** So your work should come in big, uniform batches where 32 adjacent threads do the same thing. *Irregular, per-element-different work wastes lanes.*
2. **It hides latency with many resident warps, not big caches.** So your kernel needs enough parallel work (occupancy) to fill the memory-stall windows. *Too little parallelism and the SM idles.*
3. **It has dedicated matrix hardware (Tensor Cores) that is ~16 times faster on dense matmul.** So your computation should be shaped as big, aligned matmuls in the right precision. *Scalar or ragged work runs on the slow cores.*
4. **It penalizes data-dependent branches within a warp (divergence).** So your control flow should be uniform across a warp. *Branchy code serializes lanes.*

Read those four rules together and they spell one word: **regularity.** A GPU is a machine that converts *dense, regular, predictable, parallel arithmetic* into enormous throughput, and converts *sparse, irregular, branchy, serial work* into disappointment. This is not a flaw to be patched; it is the deal. The entire discipline of high-performance AI engineering — mixed precision, kernel fusion, FlashAttention, big batches, tensor parallelism — is the art of reshaping a model's computation into the dense regular form the silicon was built to devour. And the reason Transformers, of all architectures, scaled to trillions of parameters is partly that their core operation is the GEMM: the single most dense, regular, Tensor-Core-friendly operation there is. The architecture and the hardware co-evolved. A Transformer is, from the silicon's point of view, a near-perfect customer.

The connection to the [roofline model](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) is direct: a big GEMM has high *arithmetic intensity* (many FLOPs per byte loaded), so it can actually reach the Tensor Cores' compute peak, while a LayerNorm or a softmax reduction has low intensity and is bounded by memory bandwidth no matter how many warps you throw at it. The hardware tour in this post is the *why* behind the roofline's *what*: dense regular matmul is compute-bound and fast because it maps onto the Tensor Cores at high occupancy with no divergence; everything else is usually memory-bound and the game becomes moving fewer bytes, which is the [memory hierarchy](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm) story.

## Case studies and real numbers

Abstract rules are cheap; let us anchor them in named hardware and reproducible figures. All of these trace to vendor specs or standard microbenchmarks; where a figure is representative rather than an exact published constant, it is marked approximate.

**A100 SM and Tensor Core counts (vendor spec).** The NVIDIA A100 (GA100, SXM4) has **108 SMs**, each with **64 FP32 CUDA cores** (6,912 total), **4 third-generation Tensor Cores** (432 total), **4 warp schedulers**, a **256 KB register file** (65,536 registers), and up to **192 KB combined L1/shared memory**. Peak bf16/fp16 Tensor Core throughput is **312 TFLOP/s** (624 with structured sparsity), peak TF32 is **156 TFLOP/s**, and peak FP32 on the CUDA cores is **19.5 TFLOP/s**. HBM2e bandwidth is **~2.0 TB/s**. These are from the [A100 architecture whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf) and the [datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf). The 312-vs-19.5 ratio is the **16x** Tensor-Core advantage we derived, and it is the empirical reason mixed precision is non-negotiable for training.

**H100 raises every number (vendor spec).** The successor H100 (GH100, SXM5) has **132 SMs**, fourth-generation Tensor Cores with a **Transformer Engine** and native **fp8**, roughly **989 bf16 TFLOP/s** and **~1,979 fp8 TFLOP/s** (dense), and **~3.35 TB/s** HBM3 bandwidth — see the [Hopper whitepaper](https://resources.nvidia.com/en-us-tensor-core). The structural story is identical to the A100's; the SM is still the unit, warps are still 32, latency is still hidden by occupancy. The H100 just has more SMs, faster Tensor Cores, a new fp8 path, and more bandwidth. Everything you learned here transfers; only the constants change. (If you are choosing between them for serving, the [GPU-selection post](/blog/machine-learning/large-language-model/choosing-gpu-for-llm-serving-cost-throughput-latency) walks the cost-throughput-latency trade-off.)

**The matmul on the right vs wrong path (reproducible microbenchmark).** Run a large square GEMM ($M=N=K=8192$) on an A100 three ways and you can watch the hardware lessons play out in achieved TFLOP/s:

| Path | Precision | Hardware used | Achieved TFLOP/s (approx) | Fraction of relevant peak |
|---|---|---|---|---|
| `a @ b`, TF32 off | fp32 | FP32 CUDA cores | ~17 TFLOP/s | ~87% of 19.5 FP32 peak |
| `a @ b`, TF32 on | tf32 | Tensor Cores | ~115 TFLOP/s | ~74% of 156 TF32 peak |
| `a @ b`, bf16 | bf16 | Tensor Cores | ~250 TFLOP/s | ~80% of 312 bf16 peak |

The exact achieved numbers depend on cuBLAS version, clocks, and shape, so treat them as approximate — but the *structure* is robust and reproducible: flipping the precision flag moves the same matmul from ~17 to ~250 TFLOP/s, an order of magnitude, with zero change to the math you wrote. That gap is the difference between using the Tensor Cores and not. A well-tuned big GEMM reaches **70–80% of the Tensor Cores' peak** (its MFU is high) precisely because it is dense, regular, well-aligned work at high occupancy with no divergence — the four hardware rules all satisfied at once.

**The divergence microbenchmark (reproducible).** As in the worked example, a kernel that splits each warp 16/16 on a data-dependent branch runs about **1.9x** slower than the same arithmetic with the branch aligned to warp boundaries — measured in µs with CUDA events, warm-up and `synchronize()` done properly. A fully divergent worst case (every lane a different path) can be several times slower still. This is the µs-scale, reproducible face of the SIMT tax. NVIDIA's own [Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) documents the same effect and recommends warp-aligned branching for exactly this reason.

**FlashAttention as the lesson applied (published result).** The clearest real-world payoff of "shape work to the silicon" is [FlashAttention](https://arxiv.org/abs/2205.14135): by tiling attention so the $N \times N$ score matrix is never materialized in HBM and the whole computation stays as dense Tensor-Core matmuls fused with the softmax in on-chip memory, it achieves 2–4x speedups and large memory savings over a naive attention implementation — not by changing the math, but by reshaping it into the dense, regular, high-occupancy, Tensor-Core-friendly, fusion-friendly form this whole post has been describing. It is the single best case study of why understanding the SM, the warp, occupancy, and the Tensor Core pays off. (The fusion mechanics are the subject of [the kernel-fusion post](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound).)

## When this matters (and when to stop caring)

You do not need to think about warps and occupancy every time you call `model(x)`. Most of the time, cuBLAS and cuDNN and PyTorch's own kernels already do the warp-aligned, high-occupancy, Tensor-Core-tiled thing for you, and you should let them. So here is the honest guide to when this knowledge earns its keep and when it is a distraction.

**Reach for this understanding when:** your profiler says a kernel is latency-bound and you suspect low occupancy (check `ncu` achieved occupancy); your matmuls are running far below peak TFLOP/s and you suspect they are on the FP32 path or have unaligned shapes (check the precision and pad the dims); you are writing a *custom* CUDA or Triton kernel and need to choose a block size; you have a branchy, data-dependent kernel that is slow and you suspect divergence; or you are choosing model dimensions (hidden size, vocab size, number of heads) and want them Tensor-Core-friendly from the start. In all of these, the SM-warp-occupancy-Tensor-Core model is the lens that turns "it's slow" into "it's slow *because* X, and here is the fix."

**Do not reach for it when:** you are already near peak TFLOP/s on your dominant kernels (an MFU of 50%+ on a big model means the matmuls are fine — go optimize the data loader or the communication instead); the model fits and trains and you have not profiled (premature kernel tuning is a classic waste — measure first); or you would be hand-writing a CUDA kernel to replace something cuBLAS already does well (you will almost never beat cuBLAS on a standard GEMM; spend the effort on *fusion* of the surrounding ops instead). The fastest path is usually to keep the matmuls in the framework's tuned kernels and spend your attention on the *shape* of the computation — precision, alignment, batch size, fusion — which is exactly where understanding the hardware, rather than rewriting it, pays off.

The meta-rule, consistent with the whole series: **profile to find the wall, then use the hardware model to explain it and pick the lever.** This post gives you the model for the compute wall on a single GPU. The [memory hierarchy post](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm) gives you the model for the bandwidth wall, the [roofline post](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) ties them together into the single most useful diagram in the field, and the [capstone](/blog/machine-learning/high-performance-computing/the-hpc-playbook-for-ai-engineers) turns the whole thing into a decision procedure.

## Key takeaways

- **A GPU is a throughput machine, not a fast CPU.** It spends transistors on thousands of simple parallel lanes and on hiding latency with parallelism, not on making one thread fast with big caches and out-of-order tricks. It is fast only on dense, regular, parallel work.
- **The SM is the real unit of compute.** An A100 is 108 SMs, each with 64 FP32 cores, 4 Tensor Cores, 4 warp schedulers, 65,536 registers, and ~192 KB of shared memory. Understand one SM and you understand the chip.
- **The warp of 32 threads is the unit of execution.** Under SIMT, all 32 lanes run one instruction together. The GPU thinks in 32s, so your launch configs and data layout should too.
- **Latency is hidden by swapping warps, not by caching.** The giant register file holds many resident warps so the scheduler can switch in zero cycles when one stalls on HBM. This only works if you supply enough warps — which is what occupancy measures.
- **Occupancy is the minimum across resource limits**, and registers usually bind first: $W_{\text{reg}} = \lfloor 2048 / R \rfloor$ warps for $R$ registers per thread. It is a derivable, measurable number (`ncu`), not a vibe, and it is the supply of warps available to hide latency.
- **Tensor Cores make matmul special.** They do a whole matrix tile per MMA instruction, ~16x the throughput of scalar FP32 FMA — but only on the right precision (fp16/bf16/tf32) with shapes aligned to multiples of 8. Hitting that path is the highest-leverage single-GPU optimization for a Transformer.
- **Warp divergence is a tax.** A data-dependent branch that splits a warp serializes the paths and masks lanes, costing up to the number of distinct paths. Align branches to warp boundaries or go branchless; divergence *within* a warp is the bug, divergence *across* warps is free.
- **Throughput hardware loves regularity.** Dense, aligned, branch-free, high-occupancy matmul is what the silicon was built for — which is exactly why the GEMM-dominated Transformer maps onto it so well, and why the whole craft of HPC for AI is reshaping computation into that form.

## Further reading

- [NVIDIA A100 (Ampere) Architecture Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf) — the source for SM counts, Tensor Core math, register file, and the 312 TFLOP/s figure.
- [NVIDIA H100 (Hopper) Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core) — the successor's SM, fp8 Transformer Engine, and bandwidth numbers.
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — the canonical reference for the grid/block/warp/thread model, SIMT, and divergence.
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) — warp-aligned branching, occupancy, and memory-coalescing guidance.
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — the canonical case study of reshaping work to the silicon.
- Series intro: [Why HPC Is the Bottleneck for Modern AI](/blog/machine-learning/high-performance-computing/why-hpc-is-the-bottleneck-for-modern-ai) — the three walls and the spine of this series.
- Series capstone: [The HPC Playbook for AI Engineers](/blog/machine-learning/high-performance-computing/the-hpc-playbook-for-ai-engineers) — profile, find the wall, pick the lever, measure the win.
- Next in series: [The Roofline Model: Compute-Bound vs Memory-Bound](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) — the diagram that tells you whether more FLOP/s or more bandwidth will help.

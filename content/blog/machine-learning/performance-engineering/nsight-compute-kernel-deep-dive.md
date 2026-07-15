---
title: "Nsight Compute: Taking One Kernel Down to the Metal"
date: "2026-07-14"
publishDate: "2026-07-14"
description: "Nsight Systems tells you which kernel is the wall; Nsight Compute tells you why. Read the Speed-of-Light section, occupancy limiters, and warp-stall reasons to convert one slow kernel into one specific fix."
tags:
  [
    "performance-engineering",
    "gpu-optimization",
    "nsight",
    "cuda",
    "profiling",
    "pytorch",
    "occupancy",
    "memory",
    "inference",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Performance Engineering"
author: "Hiep Tran"
featured: true
readTime: 42
---

Here is the moment this post is about. You ran Nsight Systems over your inference service, you stared at the timeline, and one kernel — a softmax over the attention scores — lit up as 60% of GPU time. That is a real finding: `nsys` told you *which* kernel is the wall. But it did not tell you what to do next. Do you launch more threads? Rewrite it in Triton? Buy a faster GPU? Batch harder? Every one of those is expensive, and at least three of them are the wrong move. The trace that told you *where* the time went is silent about *why* it went there, and "why" is the only thing that picks the fix.

Nsight Compute — `ncu` — is the tool that answers "why." It is the deepest-zoom instrument in this series: where `nsys` looks at the whole system timeline across CPU, GPU, and copies, `ncu` clamps onto a *single kernel launch* and reads the hardware performance counters for that one kernel — how close it got to the compute roof, how close to the memory roof, how many of its warps were actually running versus stalled, what they stalled *on*, and how efficiently it touched the cache hierarchy. When you point `ncu` at that softmax, it comes back with a verdict in two lines: it is **memory-bound**, running at 31% of peak HBM bandwidth, with 68% of its warp stalls attributed to `Long Scoreboard` — the counter that means "the warp is parked waiting for a load to come back from memory." That verdict rules out three of your four candidate fixes and rules in exactly one: cut the memory traffic. Fuse the kernel so the scores never leave the chip. More threads will not help a kernel whose threads are already standing around waiting for bytes.

Figure 1 is that whole reading in one picture — the before state `ncu` measured, and the after state once the fix the verdict implied was applied. The rest of this post is how to produce that reading yourself, section by section, for any kernel in your service.

![a two state comparison of one softmax kernel showing the memory bound stalled version on the left and the fused version on the right with higher memory throughput and collapsed stalls](/imgs/blogs/nsight-compute-kernel-deep-dive-1.webp)

This is the ninth post in the [Profiling & Optimizing AI Services](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) series, and it sits at the bottom of the tool ladder the series climbs down: `torch.profiler` and the Chrome trace show you the shape of a step, [Nsight Systems](/blog/machine-learning/performance-engineering/nsight-systems-for-ai-services) shows you the system timeline and names the wall kernel, and `ncu` takes that one named kernel apart. It leans hard on two earlier posts and repays them: the [roofline post](/blog/machine-learning/performance-engineering/the-roofline-for-your-service) established the compute-bound-versus-memory-bound dichotomy that the Speed-of-Light section reads off directly, and the [metrics post](/blog/machine-learning/performance-engineering/metrics-that-actually-matter) established why "GPU util 100%" is a lie that `ncu`'s occupancy and throughput numbers replace with the truth. We keep the series' running spine — a Transformer inference service on one A100 or H100 — and score every fix the way the series always does: percent of the compute roof, percent of the memory roof, achieved occupancy, achieved GB/s, kernel duration, and the before-versus-after that proves the fix moved a number.

## What ncu measures, and why it is a different kind of tool

Before any command, it is worth being precise about the boundary between `nsys` and `ncu`, because using the wrong one wastes hours. **Nsight Systems is a timeline profiler**: it samples the whole process — CPU threads, CUDA API calls, kernel executions, memory copies, NVTX ranges — at low overhead, and shows you their durations and overlaps laid out in time. Its job is *attribution across the system*: which kernel, which copy, which host stall is eating the wall clock. It answers "where."

**Nsight Compute is a kernel profiler**: it stops the world around one kernel launch, reads the streaming-multiprocessor and memory-subsystem performance counters for that launch, and reports metrics that have no meaning at the timeline level — achieved occupancy, warp issue efficiency, per-reason stall breakdowns, sector-level memory efficiency, cache hit rates, the kernel's exact position on the roofline. Its job is *analysis within one kernel*: given that this kernel is the wall, what inside it is the constraint. It answers "why."

The two are a pipeline, and the order is not negotiable. You do not open `ncu` first, because `ncu` profiles one kernel at crippling overhead and you would have no idea which of your thousands of kernels to point it at. You run `nsys`, you find the one kernel that dominates, and *then* you spend `ncu`'s expensive, precise attention on that one. Running `ncu` before `nsys` is like sending a kernel to an electron microscope before you have decided which kernel matters — you will produce a beautiful, useless image of the wrong thing. This is the same discipline the [HPC profiling post](/blog/machine-learning/high-performance-computing/profiling-gpu-workloads-finding-the-real-bottleneck) frames as "find the bottleneck before you optimize"; `ncu` is the second half of that sentence, applied to a single kernel.

One more framing that pays off throughout. Everything `ncu` reports is a comparison of *achieved* against *theoretical peak* for the specific GPU you profiled on. Achieved occupancy is measured warps against the hardware's maximum warps. Memory throughput is measured GB/s against the datasheet HBM bandwidth. Compute throughput is measured instruction issue against the SM's peak issue rate. This is why `ncu` reports are honest where `nvidia-smi` lies: `nvidia-smi` reports that *a* kernel was resident (utilization), while `ncu` reports what *fraction of the hardware's capability* that kernel actually used. A kernel can be 100% "utilized" and 4% of the compute roof at the same time, and only one of those two numbers tells you there is money on the table.

## Profiling one kernel without profiling the app: the command and the replay tax

The core invocation is short, and every flag on it exists to solve the "which kernel, how many times" problem. Here is the one I reach for first, pointed at the softmax from the intro.

```bash
# Profile up to 10 launches of any kernel whose mangled name matches
# "softmax", collect the full section set, write a report file.
#   --set full        every section (SOL, occupancy, memory, warp state, source)
#   -k regex:softmax  only kernels whose name matches this regex
#   -c 10             --launch-count: profile at most 10 matching launches
#   -s 50             --launch-skip: skip the first 50 matches (warm-up)
#   -o softmax_rep    write softmax_rep.ncu-rep (open later, share, diff)
ncu --set full \
    -k "regex:softmax" \
    -s 50 -c 10 \
    -o softmax_rep \
    python serve_one_request.py
```

Read the flags as a filter narrowing from "every kernel in the program" down to "ten steady-state launches of the one kernel I care about." Without `-k`, `ncu` profiles *every* kernel the program launches, which for a Transformer forward pass is hundreds of distinct kernels and thousands of launches, each replayed many times — the run would take hours and drown you in reports. The `-k "regex:softmax"` restricts profiling to launches whose (demangled) kernel name matches the regex; you can match on function name, template arguments, anything in the symbol. The `-s 50` (`--launch-skip`) skips the first fifty matching launches so you measure *steady state*, not the first call that pays for cuDNN algorithm selection, autotuning, and allocator warm-up — the exact same warm-up discipline the [benchmark post](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) insists on for wall-clock timing. The `-c 10` (`--launch-count`) then profiles ten launches and lets you see the spread; a kernel whose duration varies wildly across launches is telling you something (shape variation, cache state) before you have read a single counter.

For a first pass you often do not want `--set full`. The section set is a real trade-off between how much you learn and how long you wait, and it maps directly onto the replay cost we are about to derive:

| `--set` | Sections collected | Passes (rough) | When to use |
|---|---|---|---|
| `basic` (default) | SOL, launch stats, occupancy | few | first look — "am I memory- or compute-bound?" |
| `roofline` | + the roofline chart | moderate | placing the kernel on the roofline |
| `detailed` | + memory workload, warp state | many | after SOL, to find the specific cause |
| `full` | everything incl. source counters | many (dozens) | the deep dive on one kernel you will rewrite |

The reason `--set full` is slow is not incidental; it is the single most important operational fact about `ncu`, and it is why you can never run this tool in production. **`ncu` collects its metrics by replaying the kernel.**

### The mechanism: why one kernel becomes thirty runs

A GPU's streaming multiprocessors have a fixed, small number of hardware performance-monitor counters — physical registers that can be wired to count events like "sectors fetched from L2" or "cycles a warp was stalled on a long scoreboard." There are far more metrics you might want than there are counter slots to hold them simultaneously. So `ncu` cannot collect everything in one run. Instead it partitions the requested metrics into groups that *do* fit the available counters, then runs the kernel once per group — a **pass** — reconfiguring the counters between passes and re-executing the identical kernel each time. To keep the passes comparable it also flushes the GPU caches before each replay, so every pass starts from the same cold state. Finally it stitches the per-pass counter values into one coherent report.

The cost is straightforward to write down. If a report needs $N_\text{passes}$ groups, and each replay costs the kernel's own execution time plus a cache flush plus fixed per-pass overhead, then the wall time to profile one kernel is approximately

$$t_\text{profile} \approx N_\text{passes} \times \big(t_\text{kernel} + t_\text{flush} + t_\text{fixed}\big) + t_\text{setup}.$$

For `--set full`, $N_\text{passes}$ is routinely a couple dozen or more, and $t_\text{flush} + t_\text{fixed}$ per pass dwarfs a microsecond-scale kernel. A softmax that runs in 41 µs in production can take tens of milliseconds to profile under `--set full` — a slowdown of hundreds to a thousand times *for that kernel*, and the surrounding process is serialized and instrumented on top. Figure 2 is that replay loop drawn out: one launch becomes many passes, each collecting a different slice of counters, before the report is assembled.

![a left to right sequence showing one kernel launch expanded into many replay passes each collecting a different counter group before the results are stitched into one report](/imgs/blogs/nsight-compute-kernel-deep-dive-2.webp)

Three consequences follow, and all three are operational rules, not trivia. **First: profile one representative launch, never production.** The replay overhead and the cache flushing perturb timing so severely that any latency number from a running `ncu` session is meaningless as a service latency — the counters are trustworthy (that is the whole point), but the wall clock is not. You profile offline, on a representative input, to learn the *shape* of the kernel's behavior, then you apply the fix and re-measure end-to-end latency with `nsys` or CUDA events. **Second: the flush means `ncu` shows you cold-cache behavior by default**, which can make a kernel look more memory-bound than it is in a hot-cache production loop; when cache residency matters, `ncu` has an `--cache-control none` option to leave the caches alone, at the cost of pass-to-pass noise. **Third: minimize the metric set.** Every metric you do not ask for is passes you do not pay. Start with `--set basic` to get the SOL verdict in a handful of passes, and only escalate to `--set full` on the one kernel whose SOL says it is worth the deep dive.

Once the report exists, you read it either in the GUI or from the CLI. The GUI is genuinely better for the memory chart and the source-counter view:

```bash
ncu-ui softmax_rep.ncu-rep         # open the full graphical report

# Or read specific sections from the terminal, no display needed:
ncu --import softmax_rep.ncu-rep --page details \
    --section SpeedOfLight \
    --section Occupancy \
    --section WarpStateStats
```

`--import` re-reads a saved report without re-profiling — so you profile once on the machine with the GPU, save the `.ncu-rep`, and analyze it anywhere, diff it against yesterday's report, or attach it to a ticket. This is the workflow that makes `ncu` a *record* and not just a live tool: the report is an artifact you keep.

## Reading the report top-down: the section ladder

An `ncu` report is not a flat dump; it is a ladder from coarse to fine, and the discipline that keeps a deep dive from turning into an aimless counter-hunt is to read it top-down and *stop as soon as the ladder tells you the answer*. Figure 3 is the ladder itself, the order you read the sections in.

![a vertical stack of report sections from the speed of light verdict at the top down through workload detail occupancy warp state and per line source counters at the bottom](/imgs/blogs/nsight-compute-kernel-deep-dive-3.webp)

The top rung is the **Speed-of-Light (SOL)** section, and for most investigations it is the only rung you need to reach a decision. It reports two numbers that summarize the kernel's relationship to the hardware: **Compute (SM) Throughput %** and **Memory Throughput %**, each the achieved rate as a percentage of that subsystem's theoretical peak. Below SOL sit the **Compute Workload Analysis** and **Memory Workload Analysis** sections, which break the two SOL bars down into their constituents — which pipelines, which cache levels. Below those, **Occupancy** tells you how full the SMs were and what limited them. Below that, **Warp State Statistics** tells you, of the cycles warps spent not issuing, *why* — the stall breakdown. And at the very bottom, **Source Counters** attributes stalls and memory inefficiency to individual lines of your kernel source or SASS. You descend only as far as you need: SOL names the bound, the workload/occupancy sections localize it, and the warp-state and source sections pin it to a cause you can edit.

### The Speed-of-Light section: the two-line verdict

Here is the top of a real report for the intro's memory-bound softmax, read from the terminal. This is the output you actually stare at.

```console
  void softmax_warp<float, 128>(float*, const float*, int, int)
  Section: GPU Speed Of Light Throughput
  ---------------------------------------------------------------------
    DRAM Frequency            cycle/nsecond          1.51
    SM Frequency              cycle/nsecond          1.28
    Elapsed Cycles                    cycle        52,800
    Memory Throughput                     %         31.04   <-- the wall?
    Compute (SM) Throughput               %          6.20   <-- math idle
    L1/TEX Cache Throughput               %         44.9
    L2 Cache Throughput                   %         38.2
    DRAM Throughput                       %         31.04
    Duration                        usecond         41.28
    Achieved Occupancy                    %         33.1
    Achieved Active Warps Per SM           warp     21.2
  ---------------------------------------------------------------------
```

Read the two throughput lines and you have the first fork. Compute is 6%, Memory is 31%. Neither is near 100%, which already rules out "this kernel is at a hardware roof and there is nothing to do." Compute is negligible, so this is not a compute-bound kernel — no amount of tensor-core work or precision change touches it. Memory is the higher of the two at 31%, which points at the memory subsystem, but *31% is not a wall* — a memory-bound kernel that were truly saturating bandwidth would read 85–95% here. So the honest reading of this SOL section is: *memory-leaning, but not memory-saturated; something is preventing this kernel from either doing math or moving bytes at capacity.* That "something" is what the lower sections exist to name, and the occupancy number already gives a hint: 33% achieved occupancy means the SMs are two-thirds empty.

This is exactly the "both throughputs low" case that the [roofline post](/blog/machine-learning/performance-engineering/the-roofline-for-your-service) flagged as *not a roofline problem*. The roofline tells you which resource bounds a kernel that is *on a roof*; when both SOL bars are low, the kernel is on neither roof, and the roofline hands off to precisely this tool. The fix is not "buy bandwidth" or "buy FLOP/s" — it is to find out why the kernel is stalling below both ceilings. So we descend the ladder. The next stop is occupancy, because a two-thirds-empty SM is the first thing worth explaining.

Before we do, the general SOL decision — the one you apply to every kernel, not just this one — is worth stating as the fork it is, and it is drawn in Figure 6 later when we assemble the full loop. For now, the three outcomes: **Compute high, Memory low** means compute-bound, and you descend into Compute Workload Analysis to check tensor-core utilization and pipe pressure. **Memory high, Compute low** means memory-bound, and you descend into Memory Workload Analysis to check coalescing and hit rates. **Both low** — our case — means the kernel is starved below both roofs, and you descend into Occupancy and Warp State to find the stall. The SOL section is a two-line triage that tells you which lower section to open; that is its entire job, and it does it in the first screen of the report.

## Occupancy: achieved versus theoretical, and the limiter

**Occupancy** is the fraction of the GPU's warp slots that are actually filled with resident warps. A warp is the hardware's unit of scheduling — 32 threads that execute together in lockstep under the SIMT model the [SM-internals post](/blog/machine-learning/high-performance-computing/inside-the-gpu-sms-warps-and-the-simt-execution-model) works out in full. Each SM has a hard ceiling on how many warps can be *resident* at once (64 on both A100 and H100, i.e. 2048 threads), and occupancy is how many of those slots your kernel fills:

$$\text{occupancy} = \frac{\text{active warps per SM}}{\text{max warps per SM}}.$$

`ncu` reports two flavors. **Theoretical occupancy** is the ceiling imposed by your kernel's *resource footprint* — the most warps that *could* be resident given how many registers each thread uses, how much shared memory each block allocates, and how many threads are in a block. **Achieved occupancy** is what actually happened, measured by sampling how many warps were resident over the kernel's run. The gap between them is itself diagnostic: theoretical 50% but achieved 33% (as in our softmax) means the kernel *could* have filled half the SM but did not, usually because there was not enough work — too few blocks to go around, or a tail where most blocks have finished and a few stragglers run alone.

Why does occupancy matter at all? Because it is the GPU's mechanism for *hiding latency*. When one warp issues a load and stalls waiting for memory, the SM's schedulers switch to another resident warp that is ready to run. With 64 resident warps, a memory latency of hundreds of cycles can be completely hidden behind other warps' work. With 21 resident warps (our 33%), there are far fewer ready warps to switch to, so when the active warps all stall on memory, the SM has nothing to run and sits idle — which is exactly why a kernel can be at 31% memory throughput and 6% compute throughput simultaneously: it is neither moving bytes nor doing math because its handful of warps are all blocked at the same time and there is no one to cover for them.

### The mechanism: what caps occupancy

Theoretical occupancy is the *minimum* over three independent per-block budgets, each of which caps how many blocks fit on an SM, which then caps the warps. This min is worth deriving because it is the single most common thing `ncu` reveals that a developer did not know about their own kernel. Figure 4 is the min in one picture.

![three independent per block resource budgets registers shared memory and block size each producing a blocks per sm ceiling that merge into the smallest one which sets active warps and occupancy](/imgs/blogs/nsight-compute-kernel-deep-dive-4.webp)

Take an A100, which has 65,536 32-bit registers per SM, up to 164 KB of shared memory per SM, a cap of 32 resident blocks per SM, and the 64-warp ceiling. Suppose a kernel launches with 256 threads per block (8 warps) and the compiler assigned it 64 registers per thread. Work the three budgets:

- **Register budget.** Each block needs $256 \times 64 = 16{,}384$ registers. The SM has 65,536, so at most $65{,}536 / 16{,}384 = 4$ blocks fit. That is $4 \times 8 = 32$ warps.
- **Shared-memory budget.** Say the kernel uses 16 KB of shared memory per block. Then $164 / 16 \approx 10$ blocks fit — not the limiter.
- **Block/warp budget.** The 64-warp ceiling allows $64 / 8 = 8$ blocks; the 32-block cap is looser still — not the limiter.

The minimum is 4 blocks, from registers. Four blocks is 32 warps, which is $32 / 64 = 50\%$ theoretical occupancy. The kernel is register-limited, and no change to block size or shared memory will lift it — only cutting register pressure (which the compiler can be pushed toward with `__launch_bounds__` or `-maxrregcount`, at the risk of spilling registers to local memory, which is its own memory-traffic problem) or accepting 50%. `ncu` reports this directly in the Occupancy section as the block limits, so you never have to do this arithmetic by hand:

```console
  Section: Occupancy
  ---------------------------------------------------------------------
    Block Limit SM                    block           32
    Block Limit Registers             block            4   <-- the limiter
    Block Limit Shared Mem            block           10
    Block Limit Warps                 block            8
    Theoretical Active Warps per SM   warp            32
    Theoretical Occupancy                 %           50
    Achieved Occupancy                    %           33.1
    Achieved Active Warps Per SM       warp           21.2
    Registers Per Thread           register/thread     64
  ---------------------------------------------------------------------
```

`Block Limit Registers` is 4, lower than all the others — registers are the limiter, exactly as the derivation said. The `Registers Per Thread` line (64) is the knob. And the gap between 50% theoretical and 33% achieved says that even the 4-blocks-per-SM ceiling was not consistently reached, pointing at a launch that did not have enough total blocks to keep every SM busy.

### Why raising occupancy sometimes does nothing

Here is the trap that burns hours, and it is the reason occupancy is a *diagnostic* and not a *goal*. Higher occupancy helps only when the kernel is **latency-bound** — when its warps stall on latency that more resident warps could hide. If a kernel is already **bandwidth-saturated** — moving bytes at 90% of HBM peak — then adding warps does nothing, because the memory system is the bottleneck and more warps just wait in a longer line for the same saturated bus. Chasing occupancy on a bandwidth-bound kernel is pure wasted effort; you will drive occupancy from 40% to 80% and watch the duration not move, because the bytes still have to cross the same wire at the same rate.

So occupancy is only actionable in combination with the SOL and warp-state readings. Low occupancy *and* both SOL throughputs low *and* stalls dominated by latency reasons — that is a latency-bound, under-occupied kernel where raising occupancy will help. Low occupancy *and* memory throughput at 90% — that is a saturated kernel where occupancy is a red herring and the only fix is fewer bytes. The occupancy number alone cannot distinguish these; you have to read what the warps were stalled on, which is the next rung down.

#### Worked example: the occupancy that mattered

A LayerNorm kernel in a training step on an A100 profiled at 28% achieved occupancy, both SOL throughputs under 35%, and warp stalls dominated by `Long Scoreboard`. The Occupancy section showed `Block Limit Registers` at 3 with 80 registers per thread — register-limited. Because the stalls were latency (long scoreboard = waiting on memory), and memory throughput was only 33% (not saturated), this was the actionable case: the kernel was latency-bound and under-occupied. Recompiling with `__launch_bounds__(256, 4)` pushed the compiler to 56 registers per thread, lifting `Block Limit Registers` to 4 and theoretical occupancy to 50%; achieved occupancy rose to 47%, memory throughput climbed to 58% as more warps' loads overlapped, and the kernel's duration dropped from 22 µs to 15 µs. Occupancy was the right lever *because* the stall reason and the unsaturated memory said so.

#### Worked example: the occupancy that did not

A fused attention kernel on the same A100 profiled at 41% occupancy, and the reflex was "raise it." But its SOL memory throughput was 91% — it was reading HBM at near-peak. The Warp State section confirmed the stalls were `Long Scoreboard`, but the *memory system was already saturated*, so those stalls were the physics of a bandwidth-bound kernel, not a latency-hiding failure. A day spent shrinking register pressure to lift occupancy from 41% to 74% moved the duration by under 2%: the bytes were the wall, not the warp count. The correct read was that this kernel was *already optimal for what it was*, and the only lever left was to move fewer bytes — which meant algorithm change (a different tiling), not occupancy tuning. Same occupancy number, opposite conclusion, and the difference was one line in the memory section.

## Warp state statistics: why the warps aren't running

If occupancy tells you how many warps were resident, the **Warp State Statistics** section tells you what those warps were *doing* — specifically, of every cycle a warp spent not issuing an instruction, why it could not issue. This is the most mechanistically revealing section in the entire tool, because each stall reason names a different physical bottleneck, and each bottleneck names a different fix. Figure 5 is the map from reason to fix.

![a table pairing each warp stall reason with what it physically means and the fix it implies covering long scoreboard mio throttle barrier wait and not selected](/imgs/blogs/nsight-compute-kernel-deep-dive-5.webp)

First, the mechanism, because "stall" is doing a lot of work in that sentence. On each cycle, an SM's warp scheduler looks at its resident warps and picks one that is *eligible* — one whose next instruction has all its operands ready and whose target pipeline is free — and issues that instruction. A warp is **stalled** on a cycle when it is *not* eligible: its next instruction is waiting on something. `ncu` samples which warp is at the head of each scheduler and, when it is stalled, records *why*, then aggregates those samples into a breakdown. The headline metric is **Warp Cycles Per Issued Instruction** — average cycles a warp waits between issuing instructions — and the stall reasons decompose it. Here is the section for our softmax:

```console
  Section: Warp State Statistics
  ---------------------------------------------------------------------
    Warp Cycles Per Issued Instruction   cycle         38.6
    Warp Cycles Per Executed Instruction cycle         39.1
    Avg. Active Threads Per Warp                        29.4
    ---- Warp Stall Reasons (cycles per instruction) ----
    Stall Long Scoreboard                cycle         26.2   <-- 68%
    Stall Wait                           cycle          5.1
    Stall Not Selected                   cycle          3.0
    Stall MIO Throttle                   cycle          2.4
    Stall Barrier                        cycle          1.1
    Stall Math Pipe Throttle             cycle          0.4
  ---------------------------------------------------------------------
```

Of 38.6 cycles between issued instructions, 26.2 — 68% — are `Long Scoreboard`. That is the number from the intro, and now we can say exactly what it means. A **scoreboard** is the hardware bookkeeping that tracks outstanding dependencies for a warp. A **long scoreboard** dependency is one on an L1TEX memory operation — a global, local, surface, or texture access that has to go out to L2 and possibly DRAM. `Stall Long Scoreboard` therefore means, physically: *the warp issued a load, and it is parked waiting for that load's data to come back from the memory hierarchy before it can use it.* Two-thirds of this softmax's warp-cycles are spent waiting on memory latency. That, combined with the low occupancy (few warps to hide behind) and the unsaturated 31% memory throughput (the bus is not the wall — the *latency*, uncovered by enough warps, is), completes the diagnosis: this kernel is **latency-bound on memory**, and it is latency-bound because it reads a lot from HBM and does not have enough parallelism to hide the round-trips.

Now the fix follows from the reason, not the duration. Here is the map every kernel deep-dive comes back to:

| Dominant stall | Physical meaning | The fix it implies |
|---|---|---|
| Long Scoreboard | Warp waiting on a global/local memory load (L2/DRAM latency) | Cut memory traffic (fuse), improve locality (raise hit rate), add ILP (more independent loads in flight), or raise occupancy to hide latency |
| Short Scoreboard | Waiting on a shared-memory / MIO dependency | Shorten shared-memory dependency chains; rethink the smem access pattern |
| MIO Throttle | The memory-IO instruction queue is full (too many shared / special-function / branch ops) | Reduce shared-memory or SFU instruction pressure |
| LG Throttle | The local/global instruction queue is full (too many memory instructions issued) | Batch or vectorize memory accesses; fewer, wider loads |
| Wait | Fixed-latency execution dependency (e.g. waiting on an FMA result) | Increase instruction-level parallelism; unroll to interleave independent work |
| Barrier | Warps waiting at `__syncthreads()` for laggards | Balance work across threads; reduce synchronization |
| Math Pipe Throttle | The math pipeline is oversubscribed | This is compute-bound — you have found the good problem |
| Not Selected | The warp was eligible but the scheduler picked another | Healthy: it means you have *plenty* of parallelism; no action |

That last row is the one that surprises people. `Stall Not Selected` is not a problem — it means so many warps were ready that the scheduler had to pick among them, which is the signature of a well-fed, high-occupancy kernel. A kernel dominated by `Not Selected` and `Selected` (issuing) with few latency stalls is *healthy*; you are done. The stalls you chase are `Long Scoreboard` (memory latency), the throttles (queue pressure), and `Barrier` (sync imbalance). Reading this section is what converts "the kernel is slow" into "the kernel is slow *because* its warps wait on HBM loads it cannot hide," which is a sentence you can act on.

### Thread divergence: when a warp runs at a fraction of its width

There is a second waste hiding in the same section, and it is easy to miss because it does not show up as a stall: **`Avg. Active Threads Per Warp`**, which read 29.4 in the softmax report. A warp is 32 threads executing in lockstep, and the only way it can do 32 threads' worth of work per instruction is if all 32 take the same branch. When threads within a warp diverge — some take the `if`, some take the `else` — the hardware executes both paths serially, masking off the inactive threads on each, and the warp's effective width drops. `Avg. Active Threads Per Warp` is the measured mean of that width. At 32 you have no divergence; at 29.4 you are losing about 8% of your execution slots to threads sitting idle behind a branch; at 16 you are running a warp at half width, doing two passes to accomplish what one full-width warp could. This does not appear as a stall — the warp is *issuing*, just at reduced width — so it hides from the stall breakdown and only surfaces here and in the Source Counters' per-line predication numbers.

The mechanism matters because the fix is different from anything above: divergence is not a memory or occupancy problem, it is a *control-flow* problem, and the remedies are structural. Sort or bucket the data so threads in a warp take the same path (the classic fix for a kernel that branches on sequence length or on a ragged batch), replace data-dependent branches with predication or arithmetic (`select` instead of `if`), or restructure the loop so the divergent region is as small as possible. In a Transformer service the usual culprit is a mask or a variable-length branch — attention over padded sequences where some threads process real tokens and some process padding, or a sampling kernel that branches per token. When `Avg. Active Threads Per Warp` is well below 32 and the kernel is compute-leaning, divergence is the waste, and no amount of memory tuning will find it because it is not in the bytes — it is in the branches. It is one more counter that turns "slow" into a specific, editable cause.

### Source counters: pinning the stall to a line

For the deepest zoom, the **Source Counters** section (best in `ncu-ui`, where it overlays counters on your CUDA-C source and the generated SASS side by side) attributes the warp stalls and memory inefficiencies to *individual instructions*. The columns that matter are the per-line **warp stall sampling** (which source line accumulated the most `Long Scoreboard` samples — i.e. which load the warps wait on) and **L2 Theoretical Sectors Global Excessive** (which access moves more sectors than it needs — the coalescing problem we come to next). This is where "the kernel is memory-latency-bound" becomes "line 47, the `scores[idx]` gather, is where 71% of the long-scoreboard stalls land," which is a fix you can write. You reach this rung only after SOL, occupancy, and warp-state have told you it is worth it — but when you are about to rewrite a kernel, per-line attribution is what tells you *which* line.

## The read-to-fix loop, end to end

Now assemble the sections into the loop you actually run. The whole point of reading top-down is that the SOL fork routes you to exactly one lower section and one class of fix; you do not read every counter, you read the ones the fork sends you to. Figure 6 is that routing as a decision tree.

![a decision tree rooted at the speed of light throughput bars branching into memory bound compute bound and neither roof each leading to the specific section to read and the class of fix](/imgs/blogs/nsight-compute-kernel-deep-dive-6.webp)

Read the tree as: the root is "read the two SOL bars," and which bar is high sends you down one branch. **Memory high, compute low** sends you to Memory Workload Analysis to check coalescing and hit rates, and the fix class is *cut or streamline bytes* — fuse, coalesce, or lower precision. **Compute high, memory low** sends you to Compute Workload Analysis to check tensor-core and pipe utilization, and the fix class is *better precision or tiling*. **Both low** sends you to Occupancy and Warp State to find the stall, and the fix class is *more parallelism* — raise occupancy, add ILP, or (if the stalls are launch-driven) cut launch overhead with the CUDA-graph techniques from later in the series. The tree is the discipline: it stops you from optimizing occupancy on a bandwidth-bound kernel or buying bandwidth for a launch-bound one, which are the two most expensive mistakes in kernel work.

Now walk the two canonical cases all the way through, because the loop only earns its keep when you see it produce a measured win.

#### Worked example: the memory-bound softmax, fused

This is the intro's kernel, followed to its conclusion. `nsys` named `softmax_warp` as 60% of GPU time in the attention block. `ncu --set full` on ten steady-state launches reported: SOL Memory 31%, Compute 6%; Occupancy achieved 33%, register-limited theoretical 50%; Warp State dominated by `Long Scoreboard` at 68%. The tree's "both low" branch plus latency stalls plus unsaturated memory said: latency-bound on memory, too few warps to hide the loads, *and* — reading the Memory Workload section — the softmax was making three separate HBM passes over the score matrix (read to find the max, read again to exponentiate and sum, read a third time to normalize), which is a lot of avoidable traffic.

The fix is the classic memory-bound move: **fuse the passes** so the scores are read from HBM once, kept in registers and shared memory across the max/exp/sum/normalize steps, and written once — the same online-softmax trick that makes FlashAttention work, described in the [kernel-fusion post](/blog/machine-learning/high-performance-computing/kernel-fusion-and-flashattention-beating-the-memory-wall). Fewer HBM round-trips means fewer long-scoreboard stalls (less latency to hide) and more bytes-per-launch of useful work. Here is the before-and-after `ncu` read, the measurement that proves the fix.

| Metric (A100 80GB, attention softmax) | Before (unfused) | After (fused) |
|---|---|---|
| SOL Memory Throughput % | 31% | 78% |
| SOL Compute Throughput % | 6% | 9% |
| Long Scoreboard (% of warp cycles) | 68% | 22% |
| Achieved occupancy % | 33% | 48% |
| Achieved DRAM bandwidth | 0.62 TB/s | 1.56 TB/s |
| Kernel duration | 41.3 µs | 17.1 µs |
| HBM passes over scores | 3 | 1 |

The duration fell 2.4×, and every counter explains why: memory throughput jumped because the kernel now moves its bytes in one saturating pass instead of three latency-exposed ones; the long-scoreboard fraction collapsed because there is far less memory latency left to hide; occupancy rose as a side effect of the simpler register footprint. Note that compute throughput barely moved — this was never a compute problem, and no amount of tensor-core tuning would have touched it. The verdict picked the fix, and the fix moved exactly the counters the verdict predicted. That is the loop working. (These are representative numbers from this class of fix, not a single published benchmark; the point is the *shape* — memory% up, stalls down, duration down — which is what you will see when you fuse a multi-pass memory-bound kernel.)

#### Worked example: the compute-bound GEMM

The opposite case, so you have both in hand. A batched GEMM in the feed-forward block on an H100 profiled at SOL Compute 61%, Memory 30% — the mirror image of the softmax. The tree's "compute high" branch sends you to Compute Workload Analysis, where the telling number was **Tensor (Tensor Core) pipe utilization: 58%** — the tensor cores were the busiest pipe but not saturated, and the FMA pipe was also carrying load, which meant not all the math was going through the tensor cores. Warp State showed `Math Pipe Throttle` and `Wait` as the top stalls, the compute-bound signature: warps waiting on the math pipeline and on execution dependencies, not on memory.

For a compute-bound kernel the levers are precision and tiling, not fusion. The GEMM was running in bf16; the model tolerated fp8 for this layer, and switching the matmul to fp8 on the H100 (whose fp8 tensor-core rate is double bf16) plus letting the kernel autotuner pick a tile shape that kept more of the work on the tensor cores moved the numbers as a compute-bound fix should:

| Metric (H100 SXM, FFN GEMM) | Before (bf16) | After (fp8 + retiled) |
|---|---|---|
| SOL Compute Throughput % | 61% | 84% |
| SOL Memory Throughput % | 30% | 41% |
| Tensor pipe utilization % | 58% | 82% |
| Dominant stall | Math Pipe Throttle | Not Selected |
| Kernel duration | 88 µs | 49 µs |

Duration fell 1.8×, compute throughput rose toward the roof, and — the tell that you are near the ceiling — the dominant stall flipped to `Not Selected`, meaning the kernel now has so much ready work that the scheduler is choosing among eligible warps. When your top stall becomes `Not Selected`, stop: the kernel is compute-bound and near saturated, and further tuning has sharply diminishing returns. Contrast this end to end with the softmax: same tool, same top-down read, opposite bound, opposite fix, and in both cases the counters chose the fix before a line of code changed.

### Trusting the counters but not the clock: re-measuring the win

There is a discipline that separates a real win from a wishful one, and it comes straight out of the replay tax. Because `ncu` replays the kernel dozens of times and flushes caches between passes, **the kernel duration `ncu` reports is measured under near-cold-cache conditions and is not the duration the kernel has in your running service.** The 41 µs and 17 µs in the softmax table are honest *relative to each other* — same tool, same cold-cache methodology before and after, so the ratio is trustworthy — but neither is the number your production loop sees, where the score matrix may be warm in L2 from the previous layer. `ncu`'s counters (which fraction of the roof, which stall reason, how many sectors) are the ground truth you profile for; `ncu`'s wall clock is a lab measurement, not a field one.

So the loop does not end at the `ncu` report. After the fix, you re-measure the win *end to end, in the real execution context*, with the timing discipline the [benchmark post](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) lays out: warm up until clocks and caches are steady, lock the GPU clocks with `nvidia-smi -lgc` so a thermal downclock does not masquerade as a regression, `torch.cuda.synchronize()` before and after (or bracket with CUDA events) so you time the kernel and not the async launch, and run enough iterations to get a stable p50. Then confirm the *step-level* improvement with `nsys`, which measures the kernel in situ at low overhead with warm caches — the real duration, in the real pipeline, next to the real neighbors. The chain is: `nsys` finds the wall → `ncu` explains it and you fix it → `nsys` and CUDA events confirm the wall moved. If `ncu` says the memory throughput doubled but the end-to-end step time did not budge, the honest conclusion is that this kernel was not on your critical path the way you thought — perhaps it overlapped with something, perhaps a different kernel is now the wall — and you go back to the timeline. The counters tell you the kernel got better; only the end-to-end re-measure tells you the *service* got better, and those are not the same claim.

## Memory workload analysis: coalescing, sectors, and hit rates

When the SOL fork sends you down the memory branch, the **Memory Workload Analysis** section is where you find the *specific* memory inefficiency. It reports the traffic and hit rate at each level of the hierarchy — L1/TEX, L2, DRAM — and, most usefully, the **sector efficiency** that reveals uncoalesced access. Figure 7 is how to read it: four counters, each with a healthy value and a wasteful one.

![a table of four memory counters l1 hit rate l2 hit rate dram throughput and sectors per request each shown with its healthy value and its wasteful value](/imgs/blogs/nsight-compute-kernel-deep-dive-7.webp)

The mechanism to understand is **coalescing**, because it is the most common and most invisible memory waste. The GPU moves global memory in **sectors** of 32 bytes. When the 32 threads of a warp each load a 4-byte value, that is 128 bytes of *useful* data. If those 32 addresses are contiguous and aligned — a coalesced access — they fall into exactly four 32-byte sectors, and the hardware moves four sectors to satisfy the warp: maximum efficiency. But if the 32 threads read *scattered* addresses — say a gather through an index array, or a strided access with the wrong stride — each thread's 4 bytes may land in a *different* 32-byte sector, so the hardware moves up to 32 sectors (1024 bytes) to deliver the same 128 useful bytes. You paid for 1024 bytes of bandwidth and used 128: **8× waste**, and the kernel looks busy on the memory bus while accomplishing an eighth of the work. The efficiency is

$$\text{sector efficiency} = \frac{\text{useful bytes}}{\text{bytes moved}} = \frac{\text{sectors needed if coalesced}}{\text{sectors actually fetched}}.$$

`ncu` reports this as sectors-per-request, and the gap from the ideal is the coalescing loss. Here is a memory section for an uncoalesced gather kernel:

```console
  Section: Memory Workload Analysis
  ---------------------------------------------------------------------
    Memory Throughput               Gbyte/s        1902.0
    L1/TEX Hit Rate                       %          12.4    <-- no reuse
    L2 Hit Rate                           %          24.8    <-- thrashing
    Mem Busy                              %          61.0
    Max Bandwidth                         %          31.0
    L2 Theoretical Sectors
      Global Excessive                    %         680.0    <-- ~7x waste
    Sectors Per Request (global load)                31.6    <-- scattered
  ---------------------------------------------------------------------
```

Read it: 31.6 sectors per request against an ideal of ~4 is a scatter — the kernel is fetching roughly eight times the sectors it needs, and the "Global Excessive" line names it directly at 680% (about 7× the necessary sectors). The L1 and L2 hit rates are floor-scraping because scattered access has no locality to reuse. And here is the subtlety that ties back to SOL: `Memory Throughput` reads 1902 GB/s — near the A100's 2.0 TB/s peak! — but `Max Bandwidth` (the useful fraction) is only 31%. The bus is *saturated moving mostly waste*. A naive reading of "95% of peak bandwidth" would conclude the kernel is optimal; the sectors-per-request line reveals that seven-eighths of that bandwidth is being thrown away on sectors the kernel does not use. This is the counter that stops you from declaring victory on a kernel that is actually broken.

#### Worked example: fixing the coalescing waste

The gather kernel above was reading a value per token through a permutation index — `x[perm[i]]` — which scatters. The fix was to change the data layout so the access became contiguous: pre-gathering the permutation once into a contiguous buffer (paying one coalesced scatter up front) so the hot kernel read sequentially. After the layout change, sectors-per-request dropped from 31.6 to 4.1, `Global Excessive` from 680% to 4%, L2 hit rate rose from 25% to 71% (the working set now had locality), and the kernel's duration fell from 54 µs to 19 µs on the A100 — a 2.8× win from moving the *same bytes* efficiently instead of scattered. The SOL memory throughput barely changed (it was already "high" — but now it is high on useful bytes), which is the final lesson of this section: **throughput percentage alone can lie; sector efficiency tells you whether the throughput is useful.** This is the same distinction the [memory-hierarchy post](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm) makes between moving bytes and moving *needed* bytes, seen through the counter that measures it.

## Case studies and real numbers

The read-to-fix loop is not a toy; it is the documented method behind several well-known kernel optimizations. A few, framed honestly about what is precise and what is representative.

**FlashAttention as a warp-stall fix.** The FlashAttention line of work (Dao et al., 2022 and after) is the canonical `ncu`-guided memory-wall optimization, and the counters tell the story cleanly: naive attention's softmax and score materialization are `Long Scoreboard`-dominated, memory-bound kernels making multiple HBM passes over the $N \times N$ scores. Tiling the computation to keep scores in on-chip SRAM deletes those passes, which is exactly the softmax fusion worked above at production scale. The published wins are wall-clock speedups that *grow with sequence length* — the signature of removing $O(N^2)$ traffic — and under `ncu` they show up as collapsed long-scoreboard fractions and memory throughput moving from a latency-exposed low number to a saturating high one. The FLOPs never changed; the byte traffic and the stalls did.

**Tensor-core utilization as the compute-bound lever.** Every high-performance GEMM library — cuBLAS, CUTLASS, the Inductor-generated matmuls behind `torch.compile` — is tuned by reading the Compute Workload Analysis section's tensor-pipe utilization and iterating tile shapes until the tensor cores saturate. NVIDIA's own CUTLASS profiling workflow is precisely this: profile a GEMM under `ncu`, read the pipe utilization and the stall reasons, adjust the tile and stage counts, re-profile. The fp8-and-retile win in the second worked example is a small instance of the same loop the library authors run at scale, and the "stall flips to Not Selected near the roof" signature is the standard stopping criterion.

**Occupancy limiters in real kernels.** The register-limited occupancy story is one of the most common findings `ncu` surfaces, because register pressure is invisible in source code — the compiler decides it, and a developer rarely knows their kernel uses 80 registers per thread until the Occupancy section says `Block Limit Registers` is the constraint. The standard remedies (`__launch_bounds__`, `-maxrregcount`, restructuring to reduce live values) are textbook precisely because the diagnosis is so reliably wrong-guessed without the tool. The important honesty from the two worked examples: the fix helps only when the kernel is latency-bound and unsaturated, and `ncu`'s warp-state and memory sections are what tell you which case you are in.

**The `ncu` roofline chart as ground truth.** For placing a kernel on the roofline — the analysis the [roofline post](/blog/machine-learning/performance-engineering/the-roofline-for-your-service) runs by hand — `ncu --set roofline` draws the dot from measured counters, capturing the *achieved* intensity, which is often below the ideal because real kernels re-read operands and move more bytes than the napkin math assumes. Use the hand calculation for the ceiling and the `ncu` chart for the truth; the gap between them is the kernel-optimization headroom.

## When to reach for ncu (and when not to)

`ncu` is the most powerful and the most expensive tool in the profiling kit, and reaching for it at the wrong moment wastes the most time. The decisive rules:

**Reach for it when** `nsys` (or the Chrome trace) has already named one kernel as a meaningful fraction of GPU time and you are about to invest in changing that kernel — fusing it, rewriting it in Triton or CUDA, changing its precision, or tuning its launch configuration. `ncu` is what converts "this kernel is slow" into "this kernel is slow *because* X," and X is what tells you whether the investment will pay. It is also the right tool when a kernel's *throughput number lies* — the coalescing case, where 95% of peak bandwidth is 85% waste — because sector efficiency is a counter no timeline tool can see.

**Do not reach for it before `nsys`.** Profiling every kernel at replay overhead to find the wall is backwards and slow; find the wall on the timeline first, then spend `ncu` on the one kernel that matters. **Do not run it in or near production** — the replay overhead and cache flushing make its wall-clock meaningless as a latency and perturb everything around it; it is an offline analysis tool on a representative input, full stop. **Do not chase occupancy on a bandwidth-saturated kernel** — if memory throughput is at 90% and stalls are the physics of a saturated bus, more warps do nothing, and the only lever is fewer bytes. **Do not deep-dive a kernel that is 3% of your time** — `ncu` will happily tell you why a negligible kernel is slow, and fixing it changes nothing; the [Amdahl ceiling](/blog/machine-learning/performance-engineering/metrics-that-actually-matter) caps your win at the fraction the kernel occupies, and `nsys` is what tells you that fraction. And **do not deep-dive a host-bound service** — if your GPU timeline is full of gaps because the CPU cannot launch kernels fast enough, `ncu` will report perfectly healthy kernels while you optimize the wrong layer entirely; that is the [kernel-launch-overhead problem](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu), and the fix is CUDA graphs, not kernel tuning.

The meta-rule: `ncu` answers "why is *this kernel* slow" with total authority, and answers nothing else. Use `nsys` to decide *which* kernel — and whether a kernel is even the problem — and only then bring `ncu` down to the metal on the one that is.

## Key takeaways

- **`nsys` says which kernel; `ncu` says why.** They are a pipeline in a fixed order: find the wall on the system timeline, then bring `ncu`'s expensive, precise attention down onto that one kernel. Never the reverse.
- **The Speed-of-Light section is a two-line triage.** Compute Throughput % and Memory Throughput % name the bound at a glance: memory high → memory-bound (fuse/coalesce), compute high → compute-bound (precision/tiling), both low → neither roof (occupancy/warp-state/launch). Read this fork before touching code.
- **`ncu` profiles by replay, so it is slow and offline-only.** It runs the kernel once per counter group — dozens of passes for `--set full` — flushing caches between passes. The counters are trustworthy; the wall clock is not. Profile one representative launch with `-k`, `-s`, `-c`; never production.
- **Occupancy is a diagnostic, not a goal.** It is capped by the min of the register, shared-memory, and block-size budgets — `ncu` names the limiter directly. Raising it helps a *latency-bound* kernel with warps to hide behind, and does nothing for a *bandwidth-saturated* one. The warp-state section tells you which you have.
- **The dominant warp stall reason names the fix.** `Long Scoreboard` = waiting on HBM (fuse, add ILP, raise occupancy); throttles = queue pressure; `Barrier` = sync imbalance; `Not Selected` = healthy, you are done. The stall reason, not the duration, chooses the optimization.
- **Throughput percentage can lie; sector efficiency tells the truth.** A kernel at 95% of peak bandwidth moving 32 scattered sectors per request is 85% waste. `Sectors Per Request` and `Global Excessive` reveal uncoalesced access that a raw GB/s number hides.
- **The fix follows the counter, mechanically, and re-measuring proves it.** Every worked win here moved exactly the counters its verdict predicted — memory% up and stalls down for the fused softmax, tensor-pipe up and stall flipped to Not Selected for the fp8 GEMM. If the numbers do not move the way the diagnosis said, the diagnosis was wrong.
- **Match the tool to the layer.** `ncu` answers "why is this kernel slow" and nothing else. Use it only after `nsys` proves a kernel is the wall and worth the rewrite — and not at all for host-bound services, negligible kernels, or production latency.

## Further reading

- [Nsight Systems for AI Services](/blog/machine-learning/performance-engineering/nsight-systems-for-ai-services) — the tool one rung up: the system timeline that names the wall kernel you bring `ncu` down onto. Run it first, every time.
- [The Roofline for Your Service](/blog/machine-learning/performance-engineering/the-roofline-for-your-service) — the compute-bound-versus-memory-bound model the Speed-of-Light section reads off directly, and where the "both throughputs low" case hands off to this tool.
- [Metrics That Actually Matter](/blog/machine-learning/performance-engineering/metrics-that-actually-matter) — why `nvidia-smi` utilization lies and occupancy is the truth, and why Amdahl caps the win from any one kernel.
- [Inside the GPU: SMs, Warps, and the SIMT Execution Model](/blog/machine-learning/high-performance-computing/inside-the-gpu-sms-warps-and-the-simt-execution-model) — what a warp, an SM, and occupancy physically are, worked from first principles.
- [The Memory Hierarchy: Registers, Shared Memory, and HBM](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm) — the sectors, cache levels, and coalescing that the Memory Workload Analysis section measures.
- [Kernel Fusion and FlashAttention: Beating the Memory Wall](/blog/machine-learning/high-performance-computing/kernel-fusion-and-flashattention-beating-the-memory-wall) — the canonical fix for a `Long Scoreboard`-dominated, multi-pass memory-bound kernel like the softmax worked here.
- [Why Your AI Service Wastes CPU and GPU](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) — the series intro and the four-wastes frame that positions the launch-overhead case `ncu` cannot see.
- [The Performance Engineering Playbook](/blog/machine-learning/performance-engineering/the-performance-engineering-playbook) — the capstone decision tree that ties `ncu`'s kernel-level verdict into the full symptom → tool → fix flow.
- NVIDIA Nsight Compute documentation — the Kernel Profiling Guide (Speed-of-Light, Occupancy, Warp State Statistics, Memory Workload Analysis, Source Counters), the `--set`/`--section`/`--metrics` reference, and the replay-mode and cache-control options.

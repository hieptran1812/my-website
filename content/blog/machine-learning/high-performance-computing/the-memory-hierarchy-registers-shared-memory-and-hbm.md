---
title: "The memory hierarchy: registers, shared memory, and HBM"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Why most deep-learning ops are memory-bound, how coalescing and bank conflicts decide your bandwidth, and how tiling turns an HBM-starved kernel into a compute-bound one on an A100."
tags:
  [
    "high-performance-computing",
    "gpu",
    "memory-bandwidth",
    "hbm",
    "shared-memory",
    "coalescing",
    "tiling",
    "flashattention",
    "cuda",
    "deep-learning",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "High Performance Computing"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/the-memory-hierarchy-registers-shared-memory-and-hbm-1.png"
---

A while back I watched an engineer spend two days trying to make a custom LayerNorm kernel faster. He had fused the mean, variance, normalize, and affine-transform into a single CUDA kernel — textbook stuff — and on paper it should have flown. The math was trivial: a handful of floating-point operations per element. On an A100 that can do **312 trillion bf16 floating-point operations per second**, a LayerNorm over a few million elements should finish before you can blink. It did not. It ran at a fraction of the speed he expected, and no amount of unrolling loops or tweaking thread counts moved the needle.

The problem was never the arithmetic. LayerNorm reads every element from memory, does a tiny bit of math, and writes every element back. The kernel was not waiting on the math units — it was waiting on **memory**. The GPU's compute cores were sitting idle, starved, while data trickled in from off-chip memory at a rate that, however impressive on a spec sheet, is the actual ceiling for that operation. He was trying to optimize the one thing that was already free, and ignoring the thing that was costing him everything.

This is the single most important idea in GPU performance, and it is the one most AI engineers never internalize: **for the majority of the operations in a neural network, the bottleneck is not how fast the chip can compute — it is how fast it can move bytes through the memory hierarchy.** The phrase you will hear from people who run large models is some version of *"the model doesn't fit"* or *"the model doesn't feed fast enough."* Those two failures — capacity and bandwidth — are both stories about memory, and they are the real constraints behind almost every slow training run and every out-of-memory crash you will ever debug.

By the end of this post you will be able to look at any deep-learning operation and answer three questions with numbers, not vibes: *Is this op limited by compute or by memory? How many bytes does it actually move across the slowest link it touches? And can I restructure it — through coalescing, avoiding bank conflicts, or tiling — to move fewer bytes and run closer to the hardware's real limit?* That skill is what separates an engineer who writes a kernel that runs at 8% of peak from one who writes a kernel that runs at 80%. The figure below is the map we will be navigating the whole way down.

![A vertical hierarchy of GPU memory tiers from registers at twenty terabytes per second down to NVMe storage at five gigabytes per second](/imgs/blogs/the-memory-hierarchy-registers-shared-memory-and-hbm-1.png)

This post is the second pillar of the series, after [why HPC is the bottleneck for modern AI](/blog/machine-learning/high-performance-computing/why-hpc-is-the-bottleneck-for-modern-ai). That post named the three walls — compute, memory bandwidth, and communication. This one cracks open the **memory-bandwidth wall** and shows you the physical structure behind it: the tiers, the rules for moving bytes through them, and the one technique (tiling) that lets you climb off the wall.

## 1. The hierarchy: six tiers, orders of magnitude apart

Let me define the terms before we use them, because the whole post rests on three words an AI engineer who has only ever called `.cuda()` may never have had to think about.

A **register** is the fastest, smallest piece of storage on the chip — a single 32-bit slot that one thread reads and writes with essentially zero latency, the way a CPU core uses its own registers. Each Streaming Multiprocessor (SM, the GPU's equivalent of a CPU core cluster) on an A100 has a register file of 256 KB, split among the threads running on it.

**SRAM** — static RAM — is the fast on-chip memory built from transistors, the same technology as a CPU's L1/L2 cache. On a GPU, SRAM shows up as the **shared memory** (a programmer-managed scratchpad, 192 KB per SM on the A100, which doubles as the L1 cache) and the **L2 cache** (40 MB, shared across the whole chip). SRAM is fast and tiny and expensive per byte.

**HBM** — High Bandwidth Memory — is the large off-chip DRAM stacked next to the GPU die, connected by an extremely wide bus. This is the "80 GB" in "A100 80GB." It is what we usually mean by *global memory* or *device memory*: when you do `x = torch.randn(4096, 4096, device='cuda')`, that tensor lives in HBM. It is enormous compared to SRAM and slow compared to SRAM, and the entire game of GPU optimization is moving data *out* of HBM as few times as possible.

Below HBM sit two more tiers that matter when a model is too big for one GPU: **host DRAM** (the CPU's main memory, reached across the PCIe bus) and **NVMe** (solid-state storage). Both are vast and both are glacially slow by GPU standards.

Here is the shape of the A100's hierarchy, top (fast, small) to bottom (slow, big):

| Tier | Capacity (A100 80GB) | Bandwidth | Latency | Scope |
| --- | --- | --- | --- | --- |
| Registers | 256 KB per SM | ~20 TB/s | ~1 cycle | per-thread |
| Shared mem / L1 (SRAM) | 192 KB per SM | ~19 TB/s | ~30 cycles | per-thread-block |
| L2 cache (SRAM) | 40 MB total | ~5 TB/s | ~200 cycles | whole chip |
| HBM2e (global) | 80 GB | 2.0 TB/s | 400–800 cycles | whole chip |
| Host DRAM (PCIe Gen4) | 1–2 TB | ~25–64 GB/s | ~1 µs | host |
| NVMe SSD | TB-scale | ~5 GB/s | ~100 µs | host |

![A table of GPU memory tiers showing size bandwidth and latency from registers down to NVMe on an A100](/imgs/blogs/the-memory-hierarchy-registers-shared-memory-and-hbm-2.png)

The numbers in the SRAM rows are approximate aggregate figures across all 108 SMs of the A100; treat the per-SM SRAM bandwidth and the cycle-latency figures as order-of-magnitude estimates from NVIDIA's architecture disclosures and independent microbenchmarks rather than guaranteed spec values. The two figures that are hard spec numbers — and the two you should burn into memory — are **HBM at 2.0 TB/s** and the **80 GB** capacity, both from the NVIDIA A100 Tensor Core GPU architecture whitepaper. The L2 is **40 MB**, and the combined L1/shared memory is **192 KB per SM**, also from the whitepaper.

The crucial thing is not any single number — it is the **ratio between tiers**. Look at the bandwidth column: registers to shared memory is roughly flat, shared memory to L2 is a ~4× drop, L2 to HBM is another ~2.5× drop, and HBM to PCIe is a **30–80× cliff**. The latency column is even more violent: a register read is ~1 cycle; an HBM read can be **400–800 cycles**. That is the difference between "instant" and "the core does nothing useful for the next several hundred clock ticks unless it has other work to overlap."

That last clause is the GPU's escape hatch and the reason any of this works at all. A GPU hides memory latency through massive **parallelism**: while one warp (a group of 32 threads executing in lockstep, the SIMT unit) is stalled waiting on HBM, the SM switches to another warp that has its data ready. With enough warps in flight — enough **occupancy** — the math units stay fed even though every individual memory access is slow. But latency-hiding only works if there is enough *bandwidth* to keep all those warps supplied. When bandwidth runs out, no amount of occupancy saves you. That is the wall.

It is worth dwelling on why latency-hiding and bandwidth are *different* problems, because conflating them is the source of half the confused GPU debugging I have watched. **Latency** is how long one access takes from issue to data-in-hand — 400-800 cycles for HBM. **Bandwidth** is how many bytes per second the link can sustain when fully pipelined. A GPU papers over *latency* by keeping hundreds of accesses in flight at once: it does not wait for one HBM read to return before issuing the next; it issues thousands, and the link streams the results back. This is why a single thread on a GPU is *slow* (it eats the full latency on every miss) but ten thousand threads are *fast* (their latencies overlap and the link runs flat-out). But once the link is saturated — once you are asking for bytes faster than 2.0 TB/s — adding more warps does nothing, because the bottleneck is no longer "this access is slow to start" but "the pipe is full." That saturated-pipe condition is what we mean by *memory-bound*, and it is a hard physical ceiling, not a scheduling problem you can tune your way out of.

There is one more structural fact worth stating now because it governs everything downstream: the fast tiers are **small and private**, the slow tiers are **large and shared**. Registers are private to one thread (256 KB per SM split across maybe 2,048 resident threads is only ~128 bytes of registers per thread). Shared memory is private to one thread block (192 KB per SM, shared by the few hundred threads of a block). L2 and HBM are global, visible to every SM. This is not an accident — it is the only way to make the fast tiers fast. A memory that every one of the GPU's ~7,000 simultaneous threads can write to cannot also be a single-cycle SRAM bank; physics forbids it. So the hierarchy is also a **scope** hierarchy, and the art of GPU programming is partly the art of demoting data to the smallest scope that still lets the threads that need it cooperate. Tiling, which we reach in §6, is exactly that move: take data that lives in global HBM and promote a working slice of it into block-private shared memory where a few hundred threads can hammer it cheaply.

### Inside the bottom of the chip: what HBM actually is

It is worth opening the box on that 2.0 TB/s number, because the physical structure of HBM explains both why it is so fast *and* why you so rarely see the full 2.0 TB/s in practice. HBM is not one flat pool of DRAM. On the A100 it is a set of memory **stacks** — vertical towers of DRAM dies bonded on top of one another and connected to the GPU die through thousands of microscopic vertical wires called through-silicon vias, all sitting on a shared silicon interposer beside the compute die. The A100 80GB carries five such stacks, and the reason the bus is so wide is exactly this stacking: instead of a narrow, fast bus like a CPU's DDR channel, HBM trades clock speed for an enormous number of parallel wires. Each stack exposes multiple independent **channels**, each channel is a separate data path with its own command stream, and the aggregate of all those channels across all the stacks is what sums to 2.0 TB/s. The bandwidth is fundamentally a *width* story, not a *speed* story — the individual DRAM cells are not especially fast, there are simply thousands of them being read in parallel.

Within a channel, the DRAM is further divided into **banks**, and each bank has a row buffer — a long strip of sense amplifiers that holds one "open" row of the array. This is where the gap between peak and achievable bandwidth is born. To read any byte, the bank must first *activate* the row containing it, copying that whole row (typically a couple of kilobytes) into the row buffer; only then can the column be read out. If your next access lands in the *same* open row, it is nearly free — a column read with no activation. If it lands in a *different* row of the same bank, the controller must close the current row (precharge) and activate the new one, a multi-cycle penalty. So HBM strongly rewards access patterns that stream sequentially through an open row and punishes ones that hop between rows of the same bank. This is the deep physical reason behind everything in §3: a coalesced, sequential access marches down an open row and runs near peak; a scattered access keeps activating fresh rows and pays the activation tax over and over.

HBM also moves data in a fixed minimum quantum called the **burst**. A single DRAM read does not hand back one byte; it hands back a burst of consecutive bytes — on HBM2e the burst granularity is 32 bytes per access. Ask for 4 bytes at a random address and the device still transfers a 32-byte burst; the other 28 bytes are fetched and discarded. This is the same waste as a broken coalesce, one level lower in the stack: random, fine-grained access can throw away the large majority of every burst, so a kernel that *looks* like it only reads a few hundred megabytes can saturate the bus moving gigabytes of bursts that were mostly thrown away. Sequential access, by contrast, uses every byte of every burst.

Put those three facts together — channels, row buffers, and bursts — and the headline follows: **the 2.0 TB/s figure is a peak you reach only with long, sequential, well-distributed access; real kernels land at roughly 80–90% of it even when they are doing everything right.** That 10–20% haircut comes from refresh cycles (DRAM cells leak and must be periodically rewritten), row-activation overhead at the edges of your access stream, and the read-to-write turnaround on the bus when an op both reads and writes. This is why the honest microbenchmark in §4 tops out near 1.76–1.81 TB/s rather than a clean 2.0: that ~88% is not a flaw in the kernel, it is the achievable ceiling, and treating ~85–90% of peak as "done" is the single most useful calibration you can carry into a profiling session. Chasing the last 10–15% toward the spec-sheet peak is usually chasing a number the hardware will not give you.

If on-device capacity and the "model doesn't fit" failure is what keeps you up at night, the edge-AI companion piece [memory is the real constraint](/blog/machine-learning/edge-ai/memory-is-the-real-constraint) walks the same hierarchy from the deployment side. Here we are after speed, not just fit.

## 2. Memory-bound vs compute-bound: the number that decides everything

The way to predict whether an operation is limited by compute or by memory is to compute its **arithmetic intensity** — the ratio of arithmetic work to bytes moved:

$$I = \frac{\text{FLOPs}}{\text{bytes moved through the bottleneck tier}}$$

Arithmetic intensity has units of FLOPs per byte. It is the single most useful number in GPU performance, because it tells you which ceiling you are going to hit. Every machine has two ceilings: a **compute ceiling** (peak FLOP/s — for the A100, 312 TFLOP/s of bf16 with Tensor Cores) and a **memory ceiling** (peak bandwidth — 2.0 TB/s of HBM). Divide them and you get the machine's **ridge point**, the arithmetic intensity at which the two ceilings cross:

$$I_\text{ridge} = \frac{312 \times 10^{12}\ \text{FLOP/s}}{2.0 \times 10^{12}\ \text{B/s}} = 156\ \text{FLOP/byte}$$

If your operation's arithmetic intensity is **below** ~156 FLOP/byte, the A100 will finish the math long before the bytes arrive, and you are **memory-bound** — your speed is `bandwidth × intensity`, full stop. If it is **above** ~156, the bytes arrive faster than the cores can chew them, and you are **compute-bound** — your speed is the FLOP ceiling. (In practice the bf16 ridge sits a little lower if you reckon against an achievable rather than peak bandwidth, and FlashAttention's authors cite a working figure near 13 FLOP/byte for fp32 against the same machine; the exact crossover depends on the precision and the achievable bandwidth you measure, but the structure is identical.) This is the **roofline model**, and the next post in the series, [the roofline model: compute-bound vs memory-bound](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound), does nothing but this idea in full. Here we only need the punchline: *most deep-learning ops have an arithmetic intensity far below the ridge, so most deep-learning ops are memory-bound.*

Let me make that concrete with the running Transformer example. Take a single **LayerNorm** over an activation tensor of $N$ elements in bf16 (2 bytes each). To normalize, the kernel must read all $N$ elements, compute a mean and variance (a couple of passes, but you can fuse them into roughly one), and write all $N$ elements back. So:

- **Bytes moved** ≈ read $2N$ bytes + write $2N$ bytes = $4N$ bytes.
- **FLOPs** ≈ a small constant per element — subtract the mean, divide by the standard deviation, scale, shift: call it ~10 FLOPs per element, so $10N$ FLOPs.

$$I_\text{LayerNorm} = \frac{10N}{4N} = 2.5\ \text{FLOP/byte}$$

That is **2.5 FLOP/byte against a ridge of 156**. LayerNorm is not close to compute-bound; it is *catastrophically* memory-bound. The cores will be idle ~98% of the time. The matrix below puts a handful of ops on this axis so you can see the pattern.

![A grid placing arithmetic intensity on the roofline with elementwise ops pinned to the HBM ceiling and a large matmul reaching the compute ceiling](/imgs/blogs/the-memory-hierarchy-registers-shared-memory-and-hbm-7.png)

Now contrast with a **matrix multiply**, the operation that actually fills GPU FLOP charts. For $C = A \times B$ where all three are $n \times n$:

- **FLOPs** = $2n^3$ (each of the $n^2$ outputs is a dot product of length $n$, costing $2n$ FLOPs).
- **Bytes moved** (if done naively, reading the operands once each and writing the output once) = $3n^2 \times 2$ bytes in bf16 = $6n^2$.

$$I_\text{matmul} = \frac{2n^3}{6n^2} = \frac{n}{3}\ \text{FLOP/byte}$$

For a $4096 \times 4096$ matmul, that is $4096 / 3 \approx 1365$ FLOP/byte — **well above the ridge**, firmly compute-bound. *This is why GPUs were built for matmuls.* The arithmetic intensity grows linearly with the matrix dimension: the bigger the matmul, the more compute you extract per byte you move. Small matmuls, like the ones in a tiny attention head, can fall back below the ridge and become memory-bound again — which is exactly the regime where fusion and tiling earn their keep.

Step back and look at what just happened, because it is the central lesson of the whole post. *The same operation — matrix multiply — can be compute-bound or memory-bound depending purely on how you move its bytes.* The $I = n/3$ figure above assumed each operand is read from HBM exactly once. The naive implementation in §6 reads each operand $n$ times, which drags the intensity back down to ~0.5 FLOP/byte and makes the *same matmul* memory-bound. So arithmetic intensity is not a fixed property of an algorithm; it is a property of the *implementation's data movement*. This is liberating: it means the bottleneck is something you can *engineer*. You do not get to change how many FLOPs a matmul requires, but you absolutely get to change how many bytes it drags across HBM — and that ratio is what decides your fate against the roofline.

Run the same calculation across a Transformer's forward pass and a pattern jumps out. The big GEMMs — the QKV projection, the attention output projection, the two MLP matmuls — are large and compute-bound. Everything *between* them — the LayerNorms, the residual adds, the activation functions (GELU/SiLU), the softmax, the dropout — is elementwise or near-elementwise, with arithmetic intensity in the single digits, and therefore memory-bound. A naive Transformer thus alternates between compute-bound matmuls and a swarm of memory-bound "glue" ops, and those glue ops can eat a startling fraction of the wall-clock time precisely because each one makes a full HBM round-trip to do almost no math. This is the structural reason **kernel fusion** is so valuable for Transformers: fuse the bias-add, activation, and dropout into the matmul's epilogue, or fuse the LayerNorm into one pass, and you delete entire HBM round-trips that were pure memory-wall tax. The intensity of the *fused* region rises because you amortize the bytes over more of the math.

#### Worked example: a LayerNorm over a 4096×4096 tensor, and why more FLOP/s is useless

Let me put real numbers on the opening story. Take a LayerNorm over an activation tensor of shape $4096 \times 4096$ in bf16 — about 16.8 million elements, the size of one Transformer layer's activations at a healthy batch. A well-fused LayerNorm reads every element once (a single pass that computes mean, variance, and the normalized output, using a numerically stable streaming formulation), and writes every element once.

- **Bytes moved** $= 2 \times (4096 \times 4096) \times 2\ \text{bytes} = 2 \times 16.78\text{M} \times 2 \approx 67$ MB (read $33.5$ MB + write $33.5$ MB).
- **FLOPs** $\approx 10$ per element (subtract mean, multiply by inverse standard deviation, scale, shift, plus the reduction arithmetic) $\times 16.78\text{M} \approx 1.7 \times 10^8$ FLOPs.

At the A100's bf16 compute ceiling of $312 \times 10^{12}$ FLOP/s, that $1.7 \times 10^8$ FLOPs would finish in **~0.5 microseconds** if compute were the limit. But the bytes tell a different story: moving 67 MB at the *achievable* ~1.8 TB/s (88% of the 2.0 TB/s peak) takes $67 \times 10^6\ \text{B} / 1.8 \times 10^{12}\ \text{B/s} \approx 37$ microseconds. The kernel is therefore **~70× slower than its own arithmetic would allow**, and it spends ~99% of its life waiting on HBM. A perfectly written LayerNorm here runs at roughly the achievable bandwidth — call it **~1.8 TB/s, about 88% of the 2.0 TB/s peak** — and that is the *best case*, the number you should be thrilled to hit.

Here is the part that catches people. Suppose NVIDIA shipped a chip with *double* the FLOP/s — 624 TFLOP/s — and identical 2.0 TB/s HBM. How much faster does this LayerNorm run? **Not at all.** The 0.5-microsecond compute window shrinks to 0.25, but the 37-microsecond memory transfer is untouched, and 37 dominates 0.25 exactly as before. The op is bottlenecked on a link the extra FLOP/s never touch. This is the whole reason arithmetic intensity, not FLOP/s, is the number that predicts speed for memory-bound ops: below the ridge, your runtime is $\text{bytes} / \text{bandwidth}$, and the FLOP ceiling is a spec you are nowhere near using. The *only* levers that help a memory-bound op are the ones that move fewer bytes — fuse it into a neighbor so the intermediate never round-trips HBM (§2's fusion example), or, if it had reuse, tile it (it does not; LayerNorm touches each byte once, so there is nothing to tile and fusion is the entire game). That is the lesson the engineer in the opening story learned the slow way: he was buying speed on the one axis the hardware was already giving him for free.

#### Worked example: counting HBM traffic for a softmax row

Take the softmax inside attention. For a sequence of length $S = 2048$, attention produces an $S \times S$ score matrix; softmax runs over each of the $S$ rows. Consider one row of length $S = 2048$ in bf16.

- A naive softmax makes **three passes** over the row: one to find the max (for numerical stability), one to exponentiate and sum, one to divide. Each pass reads the row from HBM if the row does not fit in registers/SRAM. Plus one write of the result.
- **Bytes** ≈ $3 \times (2 \times 2048)$ read + $(2 \times 2048)$ write = $12288 + 4096 = 16384$ bytes per row.
- **FLOPs** ≈ a handful per element (compare, exp, add, divide), call it ~5 FLOPs × 2048 ≈ 10,240 FLOPs per row.

$$I_\text{softmax,naive} = \frac{10{,}240}{16{,}384} \approx 0.63\ \text{FLOP/byte}$$

Memory-bound by a factor of ~250 against the ridge. Notice the lever already: the three passes triple the HBM reads. If you can keep the row resident in fast memory and make a **single fused pass**, you cut the read traffic to $1\times$, roughly halving total bytes. That is the entire reason fused softmax and the online-softmax trick in FlashAttention exist — and we will return to it as the case study.

#### Worked example: the HBM trip count of an unfused vs fused activation

Take a slice of a Transformer MLP: a matmul output $Y$ of $N$ bf16 elements, followed by a bias add, then a GELU, then a residual add. Implemented as four separate PyTorch ops, each op makes its own HBM round-trip:

- `Y + bias` reads $Y$ ($2N$), reads bias (small, ignore), writes ($2N$) → $4N$ bytes.
- `gelu(...)` reads ($2N$), writes ($2N$) → $4N$ bytes.
- `... + residual` reads ($2N$), reads residual ($2N$), writes ($2N$) → $6N$ bytes.

Total HBM traffic for the three glue ops: **$\approx 14N$ bytes**, for an amount of arithmetic that is a small constant per element — call it ~15 FLOPs × $N$. Arithmetic intensity ≈ $15N / 14N \approx 1.07$ FLOP/byte: deeply memory-bound, three full HBM round-trips.

Now **fuse** all three into one kernel (or let `torch.compile` do it). The fused kernel reads $Y$ once, reads the residual once, does the bias-add, GELU, and residual-add in registers, and writes the result once: $2N + 2N + 2N = 6N$ bytes. We cut HBM traffic from $14N$ to $6N$ — a **~2.3× reduction** — by deleting the intermediate writes-then-reads that existed only to hand data between kernels. The arithmetic is byte-for-byte identical; we just stopped bouncing the intermediates off HBM. *This is the bytes-moved model paying for itself: count the round-trips, delete the ones that exist only as kernel boundaries.* Tiling (§6) attacks reuse within one op; fusion attacks the round-trips between ops. Together they are the two halves of beating the memory wall.

## 3. Coalesced vs strided: how a warp talks to HBM

Knowing an op is memory-bound tells you the bytes you *must* move set the floor. But there is a second, sneakier problem: a sloppy access pattern can make the GPU move **far more** bytes than the floor demands. To see why, you have to understand how a warp physically reads HBM.

When a warp's 32 threads each issue a load, the GPU's memory subsystem does not service 32 independent reads. It services **transactions** — fixed-size chunks, typically **128 bytes**, aligned to 128-byte boundaries. If the 32 threads in a warp read 32 *contiguous* 4-byte words (a 128-byte span), the hardware fuses them into **one 128-byte transaction**. Every byte fetched is a byte used. This is **coalesced** access — "coalesced" meaning the per-thread requests merge into the minimum number of transactions.

Now suppose those 32 threads read with a stride — say each reads a 4-byte word but the words are 128 bytes apart (a column of a row-major matrix). Now each thread's word lands in a *different* 128-byte chunk. The hardware must issue **32 separate 128-byte transactions** to satisfy the warp. It fetches $32 \times 128 = 4096$ bytes to deliver the $32 \times 4 = 128$ bytes the warp actually wanted. **Bandwidth efficiency: 128 / 4096 = 3%.** You are paying full price for 4 KB to use 128 B. This is **strided** (or scattered) access, and it is the most common reason a kernel that "should" be memory-bound runs at a tenth of HBM bandwidth.

![A two-column comparison of a strided warp issuing many wasteful transactions versus a coalesced warp served by a single aligned transaction](/imgs/blogs/the-memory-hierarchy-registers-shared-memory-and-hbm-3.png)

The fix is almost always **lay your data out so consecutive threads touch consecutive addresses.** For a 2D tensor, that means the dimension you parallelize across threads should be the *innermost* (contiguous) dimension in memory. In PyTorch terms: a row-major (C-contiguous) tensor read along its last dimension coalesces; read along its first dimension and you stride. This is why `tensor.contiguous()` after a transpose sometimes makes a downstream kernel several times faster — you have relaid the bytes so the warp can coalesce.

A subtlety worth nailing down: coalescing is not all-or-nothing. The hardware tracks accesses at the granularity of a **sector** — a 32-byte chunk of a 128-byte cache line, so four sectors per line. When a warp's reads are perfectly contiguous and aligned, the four sectors of one 128-byte line are all *useful*, and the metric `ncu` reports — *sectors per request* — sits near its ideal of 4 sectors per warp-request for 32-bit loads. When the reads scatter, each warp-request drags in sectors of which only a fraction carry wanted bytes, and the sectors-per-request count climbs toward 32. That single metric is the most direct measurement of how badly you have broken coalescing, and it is why §4 leans on it. *Alignment* matters too: even contiguous reads that start at a non-128-byte-aligned offset can straddle two cache lines and cost an extra transaction. The allocator usually aligns the base of a tensor, but a sliced view (`x[1:]`) can land mid-line — another reason a stray `contiguous()` sometimes pays for itself.

The same coalescing logic applies to **writes**, and writes have an extra trap: a partial write to a cache line forces a read-modify-write if the hardware has to preserve the untouched bytes. Scattered or misaligned stores are therefore even costlier than scattered loads in some patterns. The practical upshot is identical to the read case — make the thread index the innermost dimension — but it is worth knowing that the write side of an op contributes to the bytes-moved total and to the coalescing penalty, not just the read side.

Here is a deliberately minimal CUDA kernel that demonstrates both patterns. The only difference between the two is the index arithmetic.

```cuda
// Each thread copies one float. The launch covers n floats.
// Coalesced: thread i reads element i  -> consecutive threads, consecutive addrs.
__global__ void copy_coalesced(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}

// Strided: thread i reads element (i * STRIDE) % n -> consecutive threads jump
// STRIDE floats apart, so each warp scatters across many 128-B transactions.
__global__ void copy_strided(const float* in, float* out, int n, int stride) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) {
        int i = (long long)t * stride % n;   // big stride -> scattered reads
        out[t] = in[i];
    }
}
```

The `copy_coalesced` kernel will run near HBM peak. The `copy_strided` kernel, with a large stride, can run **5–10× slower** despite issuing the same number of `if (i < n)` instructions, because each warp's reads explode into many transactions. The arithmetic is identical; only the byte movement changed.

#### Worked example: effective bandwidth, coalesced vs strided, on an A100

Suppose you copy a 256 MB float array (`n = 64M` floats) on an A100 (peak HBM 2.0 TB/s). The kernel reads 256 MB and writes 256 MB, so it must move **512 MB** of useful traffic.

- **Coalesced**: measured time ≈ 290 µs. Effective bandwidth = $512 \times 10^6\ \text{B} / 290 \times 10^{-6}\ \text{s} \approx 1.77\ \text{TB/s}$, i.e. **~88% of peak** — about as good as a real copy gets after accounting for read+write turnaround.
- **Strided** (stride 32): the *useful* traffic is still 512 MB, but the *moved* traffic balloons ~16–32× because most of each 128-B transaction is wasted. Measured time ≈ 3.2 ms. Effective *useful* bandwidth = $512 \times 10^6 / 3.2 \times 10^{-3} \approx 160\ \text{GB/s}$, **~8% of peak**.

These timing figures are representative of what you see on an A100; mark them approximate, but the **~11× gap** between them is the real, reproducible cost of breaking coalescing. The formula you used — `effective bandwidth = bytes ÷ time` — is the single most important measurement in this whole post, and the next section shows how to measure it honestly in PyTorch.

## 4. Measuring bandwidth honestly: a PyTorch microbenchmark

You cannot optimize what you do not measure, and GPU timing has three traps that fool almost everyone the first time: the **first call is slow** (kernel compilation, cache cold, allocator warm-up), the GPU runs **asynchronously** (Python returns before the kernel finishes, so wall-clock time around a call is meaningless), and **clocks throttle** under thermal load. The honest recipe is: warm up, use CUDA events, synchronize, and average over many steady-state iterations.

```python
import torch

def bandwidth_gbps(fn, bytes_moved, n_warmup=20, n_iter=100):
    # Warm up: trigger lazy init, JIT, allocator, and let clocks ramp.
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_iter):
        fn()
    end.record()
    torch.cuda.synchronize()          # block until the GPU is actually done

    ms_total = start.elapsed_time(end)  # milliseconds, GPU-measured
    ms_per   = ms_total / n_iter
    gbps     = bytes_moved / (ms_per * 1e-3) / 1e9
    return gbps, ms_per

n = 64 * 1024 * 1024                    # 64M floats = 256 MB
x = torch.randn(n, device="cuda", dtype=torch.float32)
y = torch.empty_like(x)

# Copy: read 256 MB + write 256 MB = 512 MB moved.
copy_bytes = 2 * x.numel() * x.element_size()
gbps, ms = bandwidth_gbps(lambda: y.copy_(x), copy_bytes)
print(f"copy : {gbps:7.1f} GB/s   {ms:.3f} ms/iter")

# Add: read x, read y, write z = 3 * 256 MB = 768 MB moved.
z = torch.empty_like(x)
add_bytes = 3 * x.numel() * x.element_size()
gbps, ms = bandwidth_gbps(lambda: torch.add(x, y, out=z), add_bytes)
print(f"add  : {gbps:7.1f} GB/s   {ms:.3f} ms/iter")
```

On an A100 you should see something close to this — and if you do not, that *is* the bug you are hunting:

```bash
copy :  1760.0 GB/s   0.291 ms/iter
add  :  1810.0 GB/s   0.424 ms/iter
```

(Treat the exact GB/s as approximate; they depend on your driver, clocks, and array size.) Both numbers sit around **88–91% of the 2.0 TB/s peak**, which tells you these elementwise kernels are *bandwidth-bound and already near-optimal* — there is nothing to gain from a fancier kernel, because you are bounded by physics, not by your code. That is a genuinely valuable thing to know before you sink a day into "optimizing" a kernel that is already at the wall.

The reason `copy` moves $2\times$ the array and `add` moves $3\times$ is the **bytes-moved model** from §2: count every input you read and every output you write, in bytes. `y.copy_(x)` reads `x` and writes `y` → 2 arrays. `z = x + y` reads `x`, reads `y`, writes `z` → 3 arrays. Get that count right and `bytes ÷ time` gives you the true effective bandwidth; get it wrong and your GB/s is off by a constant factor and you will chase a phantom.

To confirm the diagnosis at the hardware level, profile the kernel with **Nsight Compute** and read the memory workload analysis. The relevant invocation and the section to read:

```bash
# Profile one kernel with the full set, including the memory workload section.
ncu --set full --section MemoryWorkloadAnalysis \
    --launch-skip 20 --launch-count 1 \
    python bench.py
```

In the report, the line that ends the argument is **"Memory Throughput"** expressed as a percent of peak (often shown as *DRAM Throughput* / *% of peak*). If an elementwise kernel reports ~90% DRAM throughput, it is memory-bound and done — move on. If it reports 30% DRAM throughput *and* low Compute throughput, you have an *uncoalesced* or *occupancy* problem and the bytes are being wasted; `ncu` will also report the **"L2 Hit Rate"** and the number of **sectors per request** (the tell for coalescing — close to 4 sectors/request for 128-B coalesced reads, far higher for scattered ones).

## 5. Shared memory and bank conflicts: the on-chip speed trap

Once you have decided to keep data on-chip to avoid HBM round-trips, you stage it in **shared memory** — the programmer-managed SRAM scratchpad, declared in CUDA with `__shared__`. Shared memory is ~100× lower latency than HBM and the staging ground for every fast matmul and attention kernel. But it has its own performance trap, and it is the second access-pattern bug after coalescing.

Shared memory is physically divided into **32 banks** — 32 independent memory modules, one per lane of a warp. The banks are interleaved by word: address 0 is in bank 0, address 1 in bank 1, …, address 31 in bank 31, address 32 wraps back to bank 0, and so on. The hardware can service **one access per bank per cycle**. So if all 32 lanes of a warp hit 32 *different* banks, the whole warp is served in a single cycle — full bandwidth. This is **conflict-free** access.

A **bank conflict** happens when two or more lanes in the same warp address *different words in the same bank*. The bank can only serve one at a time, so the hardware **serializes** the conflicting accesses: a 2-way conflict takes 2 cycles, an $k$-way conflict takes $k$ cycles, and bandwidth drops by that factor. The classic trigger is striding through a 2D shared array whose row width is a multiple of 32: column accesses then all land in the same bank.

![A two-column comparison of a two-way bank conflict serializing a warp versus a padded conflict-free layout served in one cycle](/imgs/blogs/the-memory-hierarchy-registers-shared-memory-and-hbm-4.png)

The fix is a one-word trick that looks like a waste of memory and is one of the best deals in GPU programming: **pad the row by one element.** If you declare a tile as `tile[32][33]` instead of `tile[32][32]`, every column access now lands in a *different* bank, because the extra padding column shifts the bank mapping by one each row. You spend 32 floats of shared memory to eliminate the conflicts entirely.

```cuda
// 32x32 tile, padded to 33 cols to break column-access bank conflicts.
__shared__ float tile[32][33];   // the '33' is the anti-conflict pad

// thread (ty, tx) within a 32x32 block:
tile[ty][tx] = global_in[...];   // row write: lanes vary tx -> 32 banks, no conflict
__syncthreads();
float v = tile[tx][ty];          // column read: WITHOUT the pad this is a 32-way
                                 // conflict (32x slower); WITH the pad it is conflict-free
```

Bank conflicts are subtle because the kernel is *correct* either way — it produces the right answer, it just runs up to 32× slower on the shared-memory accesses. The way you catch them is, again, the profiler: `ncu` reports a **"Shared Memory Bank Conflicts"** metric (bank conflicts per shared load/store). Nonzero is a smell; high is a bug. This is the on-chip mirror of coalescing: coalescing is about hitting the minimum number of *HBM transactions*, bank-conflict-avoidance is about hitting the maximum number of *distinct shared-memory banks*. Both are "lay your data out so the parallel hardware units don't collide."

## 6. Tiling: turning a memory-bound op into a compute-bound one

Now the payoff. We have established that matmul *can* be compute-bound (its intensity grows with size), but the *naive* implementation throws that away. Consider computing $C = A \times B$ for $n \times n$ matrices, one output element per thread, reading operands straight from HBM:

To compute one output $C[i][j]$, a thread reads row $i$ of $A$ ($n$ elements) and column $j$ of $B$ ($n$ elements) from HBM. There are $n^2$ outputs, so naively the kernel reads $n^2 \times 2n = 2n^3$ elements from HBM. The matmul only *needs* $2n^2$ input elements to exist — so the naive kernel re-reads every element from HBM **$n$ times**. For $n = 4096$ that is reading each operand **4096 times** from the slowest tier you touch. The arithmetic intensity collapses back to roughly $2n^3 / (2n^3 \times 2\ \text{bytes}) = 0.5$ FLOP/byte — memory-bound, the cores idle, TFLOP/s in the single digits.

**Tiling** (also called *blocking*) fixes this. The idea: instead of computing one output at a time, compute a **tile** of outputs — say a $64 \times 64$ block — cooperatively with a whole thread block. To do that, the block loads a $64 \times 64$ tile of $A$ and a $64 \times 64$ tile of $B$ from HBM **into shared memory once**, then every thread in the block reuses those tiles for many multiply-accumulates before fetching the next pair of tiles. The data goes to HBM once and gets reused dozens of times from SRAM.

![A two-column comparison of an untiled matmul re-reading HBM for every operation versus a tiled kernel loading a tile into shared memory once and reusing it](/imgs/blogs/the-memory-hierarchy-registers-shared-memory-and-hbm-5.png)

How much does this save? For a tile of size $T \times T$, each element loaded into shared memory is reused $T$ times (once per output column it contributes to within the tile). So the HBM reads drop by a factor of $T$. Going from the naive $2n^3$ HBM element reads to the tiled $2n^3 / T$ reads, with $T = 64$, is a **64× reduction in HBM traffic.** The arithmetic intensity rises by the same factor — from ~0.5 to ~32 FLOP/byte for $T=64$, and higher with register-level reuse on top — and the kernel crosses from memory-bound into compute-bound territory. *This is the single most important algorithmic technique in all of GPU computing, and it is the engine inside fast GEMM, convolution, and FlashAttention.*

Here is a complete (if compact) tiled matmul using `__shared__`. Read it as the canonical pattern; production kernels add register tiling and double-buffering on top, but the skeleton is exactly this.

```cuda
#define TILE 32

__global__ void matmul_tiled(const float* A, const float* B, float* C, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    // March across the K dimension one tile at a time.
    for (int k0 = 0; k0 < n; k0 += TILE) {
        // Cooperatively stage one TILE x TILE block of A and of B into SRAM.
        // These global reads are COALESCED: threadIdx.x is the innermost index.
        As[threadIdx.y][threadIdx.x] = A[row * n + (k0 + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(k0 + threadIdx.y) * n + col];
        __syncthreads();                 // tiles are now resident in shared memory

        // Reuse the staged tiles TILE times each, all from fast SRAM, no HBM.
        for (int k = 0; k < TILE; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();                 // done with these tiles; safe to overwrite
    }
    C[row * n + col] = acc;              // one coalesced write per output
}
```

Trace the byte movement: each element of $A$ and $B$ is read from HBM exactly once per tile-march step, staged into `As`/`Bs`, and then reused `TILE = 32` times in the inner loop — all those 32 reuses hit shared memory, not HBM. The two `__syncthreads()` are load-bearing: the first guarantees every thread has finished writing the shared tiles before anyone reads them; the second guarantees everyone is done reading before the next iteration overwrites them. Drop either and you get a subtle race that produces wrong numbers intermittently.

Production GEMM kernels stack two more levels of reuse on top of this shared-memory tiling, and it is worth knowing they exist because they are why cuBLAS is uncatchable by hand. The first is **register tiling**: instead of one output per thread, each thread computes a small $m \times n$ micro-tile (say $8 \times 8 = 64$ outputs) held entirely in *registers*, the fastest tier of all. Now each value loaded from shared memory feeds not one but $m$ or $n$ multiply-accumulates, so the reuse factor multiplies again — shared-memory traffic drops by the register-tile dimension on top of the HBM drop from the shared tile. The second is **double-buffering** (a.k.a. software pipelining): while the cores chew on the current pair of tiles, the kernel issues the HBM loads for the *next* pair into a second shared-memory buffer, so the long HBM latency overlaps with compute instead of stalling between tiles. The net effect of these three nested levels — HBM→shared, shared→register, plus prefetch overlap — is a kernel that reads each operand from HBM once, reuses it dozens of times across two on-chip tiers, and never idles waiting for the next tile. That is the full anatomy of a fast GEMM, and it is all memory-hierarchy choreography.

The choice of tile size $T$ is itself a memory-hierarchy tradeoff and a nice illustration of why these numbers matter. Bigger tiles mean more reuse (the HBM traffic drops as $1/T$) but cost more shared memory ($2T^2$ floats for the two tiles), and shared memory is only 192 KB per SM. Allocate too much per block and you can fit fewer blocks per SM, which lowers occupancy and hurts latency-hiding. So tile size is tuned against the *shared-memory budget* and the *occupancy* it permits — a $128 \times 128$ tile gives glorious reuse but may starve occupancy; a $16 \times 16$ tile keeps occupancy high but leaves reuse on the table. There is no universal best; cuBLAS ships dozens of kernels and picks per problem shape and GPU. The lesson for you is that "how big a tile" is not a free parameter — it is a negotiation between three tiers of the hierarchy at once.

![A grid of output tiles each fed by one row tile of A and one column tile of B staged in shared memory](/imgs/blogs/the-memory-hierarchy-registers-shared-memory-and-hbm-8.png)

#### Worked example: untiled vs tiled GEMM TFLOP/s on an A100

Take $n = 4096$, fp32, on an A100. Total work = $2n^3 = 2 \times 4096^3 \approx 1.37 \times 10^{11}$ FLOPs.

- **Naive (one output per thread, HBM reads)**: re-reads operands $n = 4096$ times → HBM traffic ≈ $2n^3 \times 4\ \text{bytes} \approx 550\ \text{GB}$ moved. At 2.0 TB/s that floor alone is ~275 ms, before any compute. Measured throughput lands around **0.5–2 TFLOP/s** — under **1% of peak**. The kernel is pinned to the HBM wall.
- **Tiled ($T = 32$, shared memory)**: HBM traffic drops ~32× to ~17 GB, the kernel becomes compute-bound on the CUDA cores, and a clean tiled fp32 kernel reaches **single-digit-to-low-teens TFLOP/s** on the FP32 path. Switch the math to the **Tensor Cores** (what cuBLAS/`torch.matmul` actually do, in TF32/bf16) and the same operation reaches **well over 100 TFLOP/s**, approaching the 312 TFLOP/s bf16 peak for large enough matrices.

The headline: **tiling cut HBM traffic by ~32× and moved the kernel from <1% of peak to compute-bound**, and that is *before* the Tensor-Core speedup that the library kernels layer on top. Mark the exact TFLOP/s as approximate and hardware/library-dependent; the structural result — a 32× traffic cut and a regime change from memory-bound to compute-bound — is the reproducible, load-bearing fact. This is precisely why you should almost never hand-write a matmul: `torch.matmul` dispatches to cuBLAS, which is tiled, register-blocked, double-buffered, and Tensor-Core-aware far beyond the teaching skeleton above. You write tiled kernels to understand them, and to fuse them with surrounding ops the library can't.

#### Worked example: HBM bytes for a 4096-cubed GEMM, untiled vs a 128×128 tile

Let me do the full byte accounting for a square GEMM $C = A \times B$ where all three matrices are $4096 \times 4096$, in bf16 (2 bytes per element), and contrast the untiled traffic with a $128 \times 128$ output tile. This is the calculation to internalize, because it is the exact shape that makes a Transformer MLP matmul fast or slow.

First, the irreducible facts. The output is $n^2 = 4096^2 \approx 16.8$ million elements; each is a dot product of length $n = 4096$, so the total arithmetic is $2n^3 = 2 \times 4096^3 \approx 1.37 \times 10^{11}$ FLOPs no matter how you implement it. The *minimum* possible HBM traffic is to read each operand once and write the output once: $3n^2$ elements $\times 2$ bytes $= 3 \times 16.8\text{M} \times 2 \approx 101$ MB. That floor never changes; the only variable is how many times you re-read the operands above that floor.

**Untiled (one output per thread).** Each output element reads a full row of $A$ ($n$ elements) and a full column of $B$ ($n$ elements) straight from HBM, and nothing is reused across threads. So HBM operand reads $= n^2 \times 2n = 2n^3$ elements $= 2 \times 4096^3 \times 2$ bytes $\approx 275$ GB of reads, plus the ~34 MB output write, call it **~275 GB moved**. Against the 101 MB minimum, that is re-reading each operand about **4096 times** — the kernel drags every input across the slowest tier it touches once per output column it feeds. The arithmetic intensity is $1.37 \times 10^{11}\ \text{FLOP} / 275 \times 10^9\ \text{B} \approx 0.5$ FLOP/byte, far below the 156 ridge: catastrophically memory-bound. At 2.0 TB/s, just *moving* 275 GB takes ~140 ms before a single useful FLOP is credited.

**Tiled with a $128 \times 128$ output tile.** Now a whole thread block owns a $128 \times 128$ block of $C$ and marches across the $K$ dimension in $128$-wide strips, staging a $128 \times 128$ tile of $A$ and of $B$ into shared memory at each step. The key counting fact: each element loaded into the shared tile is reused once for every output column in the tile it contributes to — a **reuse factor of $T = 128$**. So the HBM operand reads drop by $128\times$, from $2n^3$ down to $2n^3 / T$ elements $= 275\ \text{GB} / 128 \approx 2.15$ GB. The arithmetic intensity rises by the same $128\times$, from ~0.5 to about **$64$ FLOP/byte** ($0.5 \times 128 = 64$). That is still a hair under the 156 ridge on raw bytes alone — which is precisely why production kernels add the second tier of reuse (register tiling, §6) that lifts the effective intensity above the ridge and pins the kernel to the compute ceiling. The structural move is unmistakable, though: **a single level of $128 \times 128$ tiling cut HBM traffic ~128×, from 275 GB to ~2.15 GB, and lifted arithmetic intensity from 0.5 to ~64 FLOP/byte — dragging the op from deep in memory-bound territory right up to the edge of compute-bound.** Layer the register micro-tile on top and you cross the ridge; that is the whole reason cuBLAS hits >100 TFLOP/s on this exact shape. (Mark the absolute GB and FLOP/byte figures as the clean arithmetic they are; real kernels carry a little overhead, but the ~128× ratio is exact and load-bearing.)

### Hiding the load behind the math: `cp.async` and the software pipeline

The tiled kernel above has a hidden stall built into it. Look at the structure: load a tile into shared memory, `__syncthreads()`, compute on it, `__syncthreads()`, load the next tile. While the load is in flight — hundreds of cycles of HBM latency — the cores have nothing to do, because the data for the current step is not yet resident and the previous step's data has already been consumed. The naive tiled loop *serializes* load and compute, and on a memory-bound-ish problem that serialization can eat a large slice of the runtime. The cure is to **overlap** the HBM load of the next tile with the compute on the current one, so the long latency is hidden behind useful math rather than exposed as a stall. That is what double-buffering, mentioned earlier, achieves in principle — and on pre-Ampere hardware you implemented it by hand by loading into registers and then copying register-to-shared, a clumsy round trip that burned registers and still went through the core's load/store pipeline.

Ampere (the A100's generation) added a hardware instruction that makes this clean: **`cp.async`**, an asynchronous copy that streams data directly from global HBM into shared memory *without* staging it through registers and *without* blocking the issuing thread. You issue a batch of `cp.async` copies for the next tile, keep computing on the current tile, and later wait on a commit-group barrier to be sure the new tile has landed before you read it. Because the copy bypasses the register file and the L1 path, it frees registers (helping occupancy) and, crucially, lets the memory system work the HBM load in the background while the Tensor Cores grind. The result is a **software pipeline**: at any instant the kernel is computing on tile $k$ while `cp.async` is fetching tile $k+1$, so HBM latency is paid once at the start and then perfectly overlapped for the rest of the march. This is the mechanism behind the double-buffered GEMM and behind every fast Ampere attention kernel; it is the difference between a tiled kernel that exposes its HBM latency and one that buries it.

```cuda
// Ampere: stream the NEXT tile from HBM to shared memory asynchronously,
// then compute on the CURRENT tile while that load is in flight.
__pipeline_memcpy_async(&As_next[ty][tx], &A[/* next tile addr */], sizeof(float));
__pipeline_commit();                 // mark this batch of async copies
// ... do all the multiply-accumulates on the CURRENT shared tile here ...
__pipeline_wait_prior(0);            // now ensure the next tile has arrived
__syncthreads();                     // ...before any thread reads it
```

Hopper (the H100's generation) pushes this idea further with the **Tensor Memory Accelerator (TMA)**, a dedicated hardware unit that copies whole multidimensional tiles between HBM and shared memory from a single descriptor, computing all the addresses in hardware and freeing the threads entirely from the bookkeeping of staging tiles. The principle is identical — overlap the bulk HBM transfer with compute — but TMA moves the address generation off the cores and handles bigger, structured transfers in one shot, which is part of why H100 GEMM and attention kernels feed the Tensor Cores so much more efficiently. The throughline across `cp.async` and TMA is the same memory-hierarchy discipline this whole post preaches: *get the bytes moving early and in the background, so the slow link is never the thing the cores are waiting on.*

### When L2 saves you: residency control on the A100

There is a tier we have mostly skipped — the **L2 cache**, 40 MB and shared across the whole chip — and it deserves a word because it is the one tier you can sometimes steer explicitly. By default L2 is a transparent, hardware-managed cache: data you read from HBM lands there, and if another SM (or the same one later) reads the same address, the request is served from L2 at ~5 TB/s instead of crossing to HBM at 2.0 TB/s. For a workload whose hot data fits inside 40 MB, L2 is a free 2.5× bandwidth multiplier on the reused bytes — no code changes required. The catch is that 40 MB is small relative to the tensors in a large model, so for a single big GEMM the operands blow straight through L2 and it does little; L2 helps most when *many* small accesses share a *small* hot working set.

Ampere added a knob for exactly that case: **L2 residency control**, the ability to *set aside* a portion of the 40 MB L2 as a **persisting** region for data you mark as high-reuse, so the hardware preferentially keeps it resident instead of evicting it under streaming pressure. You carve out a set-aside window (`cudaDeviceSetLimit` with `cudaLimitPersistingL2CacheSize`) and tag an address range with an access policy so its reads are treated as *persisting* rather than *streaming*. The canonical win is a kernel that repeatedly reads a modest, fixed table — an embedding lookup table, a small set of weights reused across a batch, or a value reread every iteration of an inner loop — alongside a flood of one-shot streaming data. Without residency control the streaming flood evicts the hot table from L2 and every reuse pays the full HBM trip; with it, the table stays pinned in the persisting window and its rereads hit L2 at ~5 TB/s. It is the same idea as tiling — keep the reused bytes in a faster tier — but applied to the one cache tier you would otherwise have no control over, and it is worth reaching for when the profiler shows a low L2 hit rate on data you *know* is reused but is being evicted by streaming traffic.

## 7. The energy angle: why moving bytes is the real cost

There is a second reason memory dominates that has nothing to do with time and everything to do with **watts** — and it is the deeper "why" behind the whole hierarchy. The energy cost of an operation is wildly dominated by *data movement*, not arithmetic. A floating-point multiply-add costs on the order of **~1 picojoule**. Reading the operands for that FLOP from off-chip DRAM costs on the order of **~200–600 picojoules** — roughly **100–600× more energy than the computation it feeds.**

![A table comparing the picojoule energy of a floating-point operation against reads from each memory tier up to several hundred times more for DRAM](/imgs/blogs/the-memory-hierarchy-registers-shared-memory-and-hbm-6.png)

These figures are the well-known order-of-magnitude estimates from the computer-architecture literature (Horowitz and others; the exact pJ values shift with process node and voltage, so treat them as approximate). But the ordering is rock-solid and it is the physical reason the memory hierarchy exists at all: **the closer to the compute units you keep your data, the cheaper it is in both time and energy.** A register access is nearly free; an SRAM access is cheap; an HBM access is expensive; a PCIe access is ruinous. Every tier you avoid touching is a tier of latency *and* energy you save.

This reframes tiling and fusion as **energy** optimizations, not just speed ones. When tiling cuts HBM traffic 32×, it is not only making the kernel faster — it is cutting the dominant energy term of the operation by roughly the same factor. At datacenter scale, where a training run is metered in megawatt-hours and dollars, the engineer who moves fewer bytes is the engineer with the smaller power bill. The memory wall is simultaneously a *time* wall, a *capacity* wall, and an *energy* wall, and all three push you toward the same discipline: keep data on-chip, move it as few times as you can.

| Access | Approx energy | Relative to one FLOP |
| --- | --- | --- |
| fp16 multiply-add | ~1 pJ | 1× (baseline) |
| Register read | ~0.1 pJ | ~0.1× |
| SRAM / L1 read | ~5 pJ | ~5× |
| L2 read | ~25 pJ | ~25× |
| HBM / DRAM read | ~200–600 pJ | ~100–600× |

## 8. Why the same logic governs the bigger tiers: PCIe and NVMe

Everything above plays out again, one level down, the moment a model stops fitting in HBM. The two tiers below HBM — host DRAM over PCIe (~25–64 GB/s on Gen4/Gen5) and NVMe (~5 GB/s) — are **30–400× slower than HBM**, and the same arithmetic-intensity logic decides whether crossing them kills you.

This is the regime of **offloading**: techniques like ZeRO-Offload and FSDP's CPU offload park optimizer states, gradients, or even parameters in host DRAM and stream them to the GPU as needed. It works *only* when the arithmetic done per byte fetched across PCIe is high enough to hide the transfer behind compute — exactly the roofline argument, but with the bottleneck tier now being the **25 GB/s PCIe link** instead of the 2.0 TB/s HBM. If you offload an op with low arithmetic intensity, you simply move the wall from "2 TB/s" to "25 GB/s" and make everything ~80× worse. The discipline is identical; only the numbers change.

The same is true at the very bottom for **data loading from NVMe**: streaming a training dataset off disk at ~5 GB/s can starve a GPU that wants to consume tens of GB/s of tensors, which is why production pipelines prefetch, pin memory, and overlap I/O with compute (`DataLoader(num_workers=8, pin_memory=True, prefetch_factor=4)`). *Pinned* host memory — page-locked so the GPU's DMA engine can read it directly — is the host-side analogue of coalescing: it lets the PCIe transfer run at full bandwidth instead of stalling on pageable-memory copies.

```python
# Overlap PCIe transfers with compute by prefetching pinned, non-blocking copies.
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    num_workers=8,          # CPU workers stage batches off NVMe in parallel
    pin_memory=True,        # page-locked staging buffer -> full PCIe bandwidth
    prefetch_factor=4,      # keep the pipe full ahead of the GPU
)

for x, y in loader:
    # non_blocking=True lets the H2D copy overlap with the previous step's compute,
    # but ONLY works because pin_memory gave us page-locked source buffers.
    x = x.cuda(non_blocking=True)
    y = y.cuda(non_blocking=True)
    train_step(x, y)
```

There is a real engineering decision buried here that catches teams off guard. When a 70B model in bf16 (~140 GB of weights) will not fit in a single 80 GB A100, you have three options, and they are *all* memory-hierarchy choices: shard the model across multiple GPUs so each holds a slice in HBM (fast, costs more GPUs), offload some of it to host DRAM over PCIe (fits on fewer GPUs, but every offloaded byte crosses the 25 GB/s link and you had better have the arithmetic intensity to hide it), or quantize the weights to 4-bit so 140 GB becomes ~35 GB and the model fits in one GPU's HBM again (cheapest, costs accuracy). Notice that *every* one of these is a statement about which tier the bytes live in and how fast you can reach them. The "right" answer depends entirely on the numbers — the model size, the available HBM, the PCIe bandwidth, the arithmetic intensity of the offloaded ops — which is exactly the kind of back-of-envelope the bytes-moved model lets you do in your head before you spend a dollar on hardware.

The unifying principle — and the reason this one post buys you intuition across the whole stack — is that **the hierarchy is fractal**: registers→SRAM→HBM and HBM→DRAM→NVMe are the same story at different scales, and at every scale the winning move is to maximize the work you do per byte you drag across the slowest link you are forced to touch.

## 9. The serving angle: the KV-cache lives on this hierarchy too

This is not only a training story. When you *serve* an LLM, the dominant memory object is the **KV-cache** — the stored keys and values for every token generated so far, kept in HBM so the model doesn't recompute attention over the whole prefix at each decode step. The KV-cache is a pure memory-hierarchy problem wearing a serving costume.

Decode is memory-bound for exactly the reason §2 predicts: generating one token reads the entire model's weights *and* the entire KV-cache from HBM to do a tiny amount of arithmetic (one token's worth). The arithmetic intensity of autoregressive decode is *low*, so decode throughput is set by **HBM bandwidth**, not by FLOP/s — which is why a serving GPU is chosen for its bandwidth and HBM capacity as much as its FLOP rating. And the KV-cache competes with the weights for the same 80 GB of HBM, so capacity ("the model + cache doesn't fit") becomes the binding constraint as batch size and context length grow. The whole discipline of paged attention and KV-cache management — covered in [KV-cache optimization and management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management) — is memory-hierarchy engineering: pack the cache densely in HBM, avoid fragmentation, and minimize the bytes re-read per decode step. Same hierarchy, same rules, different workload.

The lever that makes serving economical falls straight out of this analysis. Because decode reads the *full weight matrix* per token regardless of how many sequences you batch, you can amortize that fixed HBM cost by **batching** more requests: with one sequence you read all the weights to produce one token (terrible intensity); with 256 sequences you read the same weights once to produce 256 tokens (256× the arithmetic per byte of weights moved). Batching pushes decode's arithmetic intensity up and *toward* the compute ceiling — the exact same roofline move as tiling, applied across requests instead of across a matrix. The catch is capacity: every additional sequence adds its own KV-cache to the 80 GB HBM budget, so the maximum batch you can run is set by *how much HBM the cache eats*, which is why packing the cache densely (paged attention) directly buys you a bigger batch and therefore higher throughput. The whole serving stack is a negotiation between HBM *bandwidth* (which wants big batches) and HBM *capacity* (which limits them) — two faces of the same wall this entire post has been about. An engineer who internalizes the memory hierarchy reads a serving spec sheet completely differently: HBM GB/s and HBM GB stop being trivia and become the two numbers that set your tokens-per-second and your cost-per-million-tokens.

## Case studies / real numbers

Three measured results from the literature and from a benchmark you can reproduce, each illustrating one face of the memory wall.

**FlashAttention: tiling attention to avoid materializing the $N \times N$ matrix.** Standard attention computes the full $S \times S$ score matrix, writes it to HBM, reads it back for softmax, writes the softmax result, reads it again for the value multiply — an $O(S^2)$ blizzard of HBM traffic for a sequence of length $S$. FlashAttention (Dao et al., 2022) restructures the computation to **tile** over the sequence and keep the working blocks in SRAM, using the online-softmax trick to never materialize the full score matrix in HBM at all. The result, straight from the paper: attention becomes **memory-IO-bound-aware**, HBM traffic drops from $O(S^2)$ to $O(S^2 / M)$ where $M$ is the SRAM tile size, and end-to-end attention runs **2–4× faster** with a **memory footprint linear in $S$ instead of quadratic**, enabling far longer contexts. It is exactly the tiling argument from §6 applied to attention: load tiles into SRAM once, reuse, and the bytes you never moved are the bytes you never paid for. The follow-on [kernel fusion and FlashAttention: beating the memory wall](/blog/machine-learning/high-performance-computing/kernel-fusion-and-flashattention-beating-the-memory-wall) takes this apart in full.

**A bandwidth-bound elementwise kernel near peak.** The microbenchmark in §4 is itself a case study: a simple `copy`/`add` on an A100 reaches **~1.76–1.81 TB/s, roughly 88–91% of the 2.0 TB/s HBM peak.** That is the *ceiling* for any elementwise op on that machine. The lesson is diagnostic: if your fused activation kernel hits ~90% of peak DRAM throughput in `ncu`, it is *done* — you are bounded by HBM physics and there is no kernel cleverness left to extract. Knowing you are at the wall is as valuable as climbing off it, because it stops you wasting a day on a problem the hardware has already solved against you.

**Tiled GEMM, naive vs library.** The §6 worked example: a naive one-thread-per-output fp32 matmul at $n=4096$ runs at **under 1% of peak** because it re-reads operands ~4096× from HBM. A tiled shared-memory kernel cuts that HBM traffic ~32× and becomes compute-bound; cuBLAS (what `torch.matmul` calls) layers register tiling, double-buffering, and Tensor Cores on top and reaches **well over 100 TFLOP/s** for large matrices, approaching the **312 TFLOP/s bf16 peak** on the A100 (NVIDIA A100 architecture whitepaper). The entire ~100× gap between the naive and the library kernel is *memory-hierarchy engineering* — coalescing, tiling, bank-conflict avoidance, reuse — not arithmetic cleverness.

## When to reach for this (and when not to)

Every technique here is a cost — in code complexity, in development time, in the risk of a subtle race. Spend them where the bytes are, not everywhere.

- **Always run the microbenchmark and the profiler first.** Before you optimize anything, measure the effective bandwidth (`bytes ÷ time`) and read `ncu`'s memory-throughput percent. If a kernel is at ~90% of HBM peak, it is bandwidth-bound and *done* — no kernel rewrite will help. If it is at 30% with low compute throughput, *now* you have a coalescing or occupancy bug worth chasing.
- **Fix coalescing before anything fancy.** It is the highest-leverage, lowest-effort fix: relaying data so consecutive threads touch consecutive addresses can be a 5–10× win for one `contiguous()` call or one index swap. Do this before you even think about shared memory.
- **Reach for tiling/shared memory when an op has reuse and is memory-bound.** Matmul, convolution, and attention all reuse each input many times — tiling pays enormously. A pure elementwise op (add, GELU, LayerNorm) has *no reuse*; tiling it buys nothing because every byte is touched exactly once. For those ops, the win is **fusion** (do several elementwise ops in one HBM pass), not tiling.
- **Don't hand-write a matmul.** `torch.matmul`/cuBLAS will beat your hand-tiled kernel by a wide margin. Write a tiled kernel to *understand* the pattern and to *fuse* matmul with neighbors that the library can't, never to replace the library's GEMM.
- **Worry about bank conflicts only once you're already in shared memory** and the profiler shows nonzero conflicts. The `[33]` padding trick is cheap; reach for it when `ncu`'s shared-conflict metric is hot, not prophylactically everywhere.
- **Offloading to PCIe/NVMe only when you can't fit and the arithmetic intensity per offloaded byte is high.** Offloading a low-intensity op just moves the wall from 2 TB/s to 25 GB/s and makes it ~80× worse. It is a capacity tool, not a speed tool.

## Key takeaways

- The GPU memory hierarchy spans **six orders of magnitude in bandwidth and latency** — registers (~20 TB/s, ~1 cycle) to NVMe (~5 GB/s). The A100's binding numbers: **2.0 TB/s HBM, 80 GB, 40 MB L2, 192 KB L1/shared per SM.**
- **Arithmetic intensity** $= \text{FLOPs}/\text{bytes}$ decides everything. Below the machine's ridge point (~156 FLOP/byte on the A100) you are **memory-bound**; above it, **compute-bound**. Most DL ops (LayerNorm ~2.5, softmax ~0.6) are far below the ridge — *memory-bound*.
- Count bytes with the **read-inputs + write-outputs** model and measure **effective bandwidth = bytes ÷ time**. That one ratio tells you if you are at the wall.
- **Coalescing** turns a warp's 32 reads into one 128-byte transaction; breaking it can waste >90% of fetched bytes and cost 5–10×. Lay data out so consecutive threads touch consecutive addresses.
- **Bank conflicts** serialize shared-memory access up to 32×; the `[N+1]` padding trick fixes the common column-stride case for the price of one column.
- **Tiling** loads a tile into SRAM once and reuses it $T$ times, cutting HBM traffic by ~$T$× and turning a memory-bound matmul into a compute-bound one. It is the engine inside fast GEMM and **FlashAttention** (2–4× faster, linear memory).
- **Data movement dominates energy**: an HBM read costs ~100–600× a FLOP in picojoules. Moving fewer bytes is faster *and* cheaper.
- Always **measure first** (microbenchmark + `ncu`). A kernel at ~90% of HBM peak is finished; one at 30% has a coalescing or occupancy bug.

## Further reading

- **NVIDIA A100 Tensor Core GPU Architecture** (whitepaper) — the source for 2.0 TB/s HBM2e, 80 GB, 40 MB L2, 192 KB combined L1/shared per SM, and 312 bf16 TFLOP/s.
- **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** (Dao, Fu, Ermon, Rudra, Ré, 2022) — the canonical demonstration of SRAM tiling to cut HBM traffic in attention.
- **CUDA C++ Programming Guide and Best Practices Guide** (NVIDIA) — the authoritative reference on coalescing, shared-memory banks, and occupancy.
- **Nsight Compute documentation** — the Memory Workload Analysis section, sectors-per-request, and bank-conflict metrics used above.
- **Computing's Energy Problem** (Horowitz, ISSCC 2014) — the order-of-magnitude energy figures for FLOP vs SRAM vs DRAM access.
- Within this series: [why HPC is the bottleneck for modern AI](/blog/machine-learning/high-performance-computing/why-hpc-is-the-bottleneck-for-modern-ai), [the roofline model: compute-bound vs memory-bound](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound), [kernel fusion and FlashAttention: beating the memory wall](/blog/machine-learning/high-performance-computing/kernel-fusion-and-flashattention-beating-the-memory-wall), and the capstone [the HPC playbook for AI engineers](/blog/machine-learning/high-performance-computing/the-hpc-playbook-for-ai-engineers).
- Going deeper on deployment-side memory and serving: [memory is the real constraint](/blog/machine-learning/edge-ai/memory-is-the-real-constraint) and [KV-cache optimization and management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management).

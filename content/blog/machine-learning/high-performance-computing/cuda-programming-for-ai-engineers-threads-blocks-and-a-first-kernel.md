---
title: "CUDA Programming for AI Engineers: Threads, Blocks, and a First Kernel"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Go from never having written a kernel to a runnable vector-add, a tiled matmul, and a fused Triton softmax — and learn exactly when to drop below PyTorch and when not to."
tags:
  [
    "high-performance-computing",
    "gpu",
    "cuda",
    "triton",
    "kernels",
    "matmul",
    "deep-learning",
    "ml-systems",
    "pytorch",
  ]
category: "machine-learning"
subcategory: "High Performance Computing"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/cuda-programming-for-ai-engineers-threads-blocks-and-a-first-kernel-1.png"
---

You have trained models. You have called `.cuda()`, watched `nvidia-smi` climb to 99%, and shipped something that works. And yet there is a layer underneath all of that — the layer where `c[i] = a[i] + b[i]` becomes a million threads firing at once — that you have never actually written a line of. This post is about going down one level. Not all the way to the silicon (that is a different post in this series), but to the place where you stop being a *user* of GPU kernels and become someone who can *write* one when the framework lets you down.

Here is the scenario that sends most AI engineers down here for the first time. You profile a Transformer training step. The big matmuls — the QKV projection, the MLP up- and down-projections — are running fine, near the GPU's peak. But there is a long tail of small operations: a softmax here, a layernorm there, a handful of elementwise scalings, a masking step. Each is tiny, but there are dozens of them, and each one launches a separate kernel, reads its inputs all the way from off-chip memory, does a trickle of arithmetic, and writes the result back. The profiler says you are spending 40% of your step time on operations that do almost no math. That is the wall this series keeps returning to — the **memory wall** — and the fix is to *fuse* those operations into one kernel that touches memory once. To do that, you need to know what a kernel is.

![tree diagram showing a single CUDA launch fanning out into a grid of 256 blocks, each holding 256 threads, for 65,536 parallel lanes](/imgs/blogs/cuda-programming-for-ai-engineers-threads-blocks-and-a-first-kernel-1.png)

By the end of this post you will be able to: write a complete, compilable CUDA vector-add from scratch and reason about its launch configuration; write a **tiled matmul** that uses on-chip **shared memory** to cut memory traffic by an order of magnitude; read **occupancy** off your launch config and know whether you left performance on the table; write a fused **softmax in Triton** — the path most AI engineers actually take in 2026 — and understand why it is the same idea as the CUDA kernel but with the hard parts automated; and, most importantly, decide *when* it is worth dropping below PyTorch at all. That last skill is the valuable one. Most of the time, the right answer is "let cuBLAS and the framework handle it." Knowing the exceptions — and proving them with a profiler — is what separates someone who *can* write a kernel from someone who *should*.

This is the kernel-authoring entry in the broader series on high-performance computing for AI engineers, the spine of which is the *three walls* — compute, memory bandwidth, and communication — read off the roofline and the profiler. Writing a kernel is how you push against the first two walls on a single GPU. If you have not read [the GPU execution model post](/blog/machine-learning/high-performance-computing/inside-the-gpu-sms-warps-and-the-simt-execution-model) or [the memory hierarchy post](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm), you can follow this one cold — I define everything as we go — but they are the hardware foundation this post stands on.

## 1. The mental model: a grid of blocks of threads

Start with the one idea everything else hangs off. When you launch a CUDA **kernel** — a function that runs on the GPU — you do not launch *one* of it. You launch thousands or millions of copies at once, and each copy runs the same code on a different piece of data. That copy is a **thread**. The CUDA programming model organizes those threads in a strict three-level hierarchy, and you specify the shape of that hierarchy at launch time.

At the bottom is the **thread**: a single instance of your kernel, with its own registers and its own index. Threads are grouped into a **block** (CUDA calls it a *thread block*). All threads in a block run on the same physical processor — the **SM**, or Streaming Multiprocessor, the GPU's fundamental compute unit — and crucially, they can talk to each other through fast on-chip **shared memory** and can synchronize with a barrier. Blocks are grouped into a **grid**, which is the full set of threads launched by one kernel call. Blocks in a grid are independent: they cannot synchronize with each other, and the GPU is free to run them in any order, on any SM, whenever a slot opens up. That independence is *why* CUDA scales — the same kernel runs on a 20-SM laptop GPU and a 108-SM A100 with no code change, because the runtime just schedules blocks onto whatever SMs exist.

There is a fourth level you do not control directly but must understand: the **warp**. Inside an SM, threads execute in lockstep groups of 32 called warps. A warp is the real unit of execution — the SM issues one instruction and all 32 lanes of the warp run it together (this is the **SIMT** model: Single Instruction, Multiple Threads). You write code as if each thread is independent, but the hardware runs them 32 at a time. This is why a block size of 256 (= 8 warps) is natural and a block size of 250 is wasteful: the hardware rounds up to 8 warps anyway and the last 6 lanes sit idle. Always make your block size a multiple of 32.

Why a warp at all? Because issuing one instruction for 32 lanes amortizes the cost of instruction fetch and decode across 32 ALUs, which is what makes a GPU a throughput machine rather than a latency machine. A CPU core spends a huge fraction of its silicon on branch prediction and out-of-order machinery to make a *single* thread fast; a GPU spends that silicon on more ALUs and runs *thousands* of threads to keep them fed. The warp is the unit that makes that trade pay off. It also explains the GPU's defining behavior — that it is fast on a big matmul (lots of independent lanes doing the same multiply-add) and slow on a branchy, pointer-chasing reduction (lanes that disagree and stall). When this series talks about the *compute wall*, it is talking about keeping these warp lanes busy; when it talks about the *memory wall*, it is talking about feeding them fast enough.

The independence of blocks is the other deliberate design choice, and it is worth pausing on because it is *why* CUDA code is portable across GPUs of wildly different sizes. Since blocks cannot synchronize with each other and may run in any order, the runtime is free to schedule them onto whatever SMs are available, whenever a slot frees up. Launch a grid of 4096 blocks: on a 20-SM laptop GPU the runtime drips them through 20 SMs over many waves; on a 108-SM A100 it runs five times as many at once and finishes far sooner — *same binary, no code change*. If blocks could coordinate globally, this would be impossible; the GPU would have to keep all of them live at once. The cost of that scalability is the constraint you will feel constantly: you get cheap synchronization *within* a block (`__syncthreads()`) and none *across* blocks, so any algorithm that needs global coordination has to either fit in one block or split into multiple kernel launches.

The figure above shows the shape concretely. One launch line spawns a grid of 256 blocks; each block holds 256 threads; that is 65,536 threads, all created by a single statement. You did not write a loop. You wrote *one* kernel body, and the hardware instantiated 65,536 copies of it. The shift in thinking — from "loop over N elements" to "launch N threads, one per element" — is the whole game. The CPU programmer asks "what does iteration `i` do?" The CUDA programmer asks "what does *thread* `i` do?" and the answer is usually: the body of one loop iteration, with `i` computed from the thread's position in the hierarchy.

That position is the next thing to nail down, because it is where the one piece of arithmetic you cannot avoid lives.

## 2. The global-index formula: blockIdx times blockDim plus threadIdx

Every thread needs to know which element of the data it is responsible for. CUDA gives each thread a few built-in read-only variables to figure that out. `threadIdx.x` is the thread's index *within its block* (0 to blockDim − 1). `blockIdx.x` is the block's index *within the grid*. `blockDim.x` is the number of threads per block. `gridDim.x` is the number of blocks. (The `.x` is because the hierarchy can be 1D, 2D, or 3D; for a flat array you only need `.x`.)

The single most important line of code in all of CUDA — the one you will write in some form in every kernel — is the global index:

```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

Read it left to right. `blockIdx.x * blockDim.x` is how many threads came *before* this thread's block. Block 0 contributes 0, block 1 contributes `blockDim.x`, block 2 contributes `2 * blockDim.x`, and so on. Adding `threadIdx.x` slots this thread into its place within the block. The result is a unique, contiguous, zero-based index across the *entire* grid. Thread 0 of block 0 gets index 0; the last thread of block 0 gets `blockDim − 1`; thread 0 of block 1 gets exactly `blockDim`. No gaps, no overlaps, every element covered exactly once.

![grid diagram showing how block index times block dimension plus thread index produces a unique global element index for each thread](/imgs/blogs/cuda-programming-for-ai-engineers-threads-blocks-and-a-first-kernel-2.png)

The figure works a concrete case. With `blockDim.x = 4`, thread 2 of block 1 computes `1 * 4 + 2 = 6`, so it owns element 6 of the array and computes `c[6] = a[6] + b[6]`. Block 0 owns elements 0 through 3, block 1 owns 4 through 7, block 2 owns 8 through 11. The formula tiles the array perfectly. Notice the two cells marked as cautions: the *formula* itself, and the *guard*. The guard matters, and it is the bug every beginner ships at least once.

#### Worked example: how many blocks and threads cover N elements

Suppose you have an array of `N = 1,000,000` floats and you want one thread per element. You pick a block size — say 256 threads per block, a good default. How many blocks do you need? You need at least `N / 256 = 3906.25` blocks, but you cannot launch a fractional block, so you round *up*:

$$\text{blocks} = \left\lceil \frac{N}{\text{threads}} \right\rceil = \left\lceil \frac{1{,}000{,}000}{256} \right\rceil = 3907.$$

In code this is the classic integer-arithmetic trick that avoids floating point:

```cuda
int threads = 256;
int blocks  = (N + threads - 1) / threads;   // ceil(N / threads)
```

With 3907 blocks of 256 threads, you launch `3907 * 256 = 1,000,192` threads — that is 192 *more* threads than you have elements. Those extra 192 threads, at the very end of the last block, will compute an index `idx` that is 1,000,000 or larger, and if they blindly do `c[idx] = a[idx] + b[idx]` they read and write past the end of the array. That is an out-of-bounds memory access: at best a wrong answer, at worst a crash, often a silent corruption that surfaces three kernels later. The fix is the guard:

```cuda
if (idx < N) {
    c[idx] = a[idx] + b[idx];
}
```

Now the 192 surplus threads compute their index, see it is out of range, and quietly do nothing. This `if (idx < N)` guard is non-negotiable in any kernel whose data size is not an exact multiple of the block size — which, in practice, is almost always. Internalize the pattern: *index, guard, work.* Every kernel in this post follows it.

Two performance properties fall directly out of how that index relates to the warp, and both are invisible until they cost you half your bandwidth. The first is **memory coalescing**. Recall that threads run 32 at a time as a warp, and that consecutive threads get consecutive indices (`idx`, `idx+1`, `idx+2`, …). When those 32 threads each read `a[idx]`, they are reading 32 consecutive floats — 128 contiguous bytes — and the memory controller services that as a *single* wide transaction. That is a coalesced access, and it is why the index formula puts `threadIdx.x` in the low-order position: it lines the warp's reads up into one fat memory request. If you instead had each thread stride by a large amount — `a[idx * 1024]`, say — the warp's 32 reads would scatter across 32 different cache lines, and the controller would issue up to 32 separate transactions for the same 32 floats, throwing away up to 31/32 of your bandwidth. The rule that follows is concrete: structure your indexing so that adjacent threads touch adjacent memory. It is the difference between hitting 1.5 TB/s and hitting 100 GB/s on the exact same kernel, and it is the most common reason a "correct" CUDA kernel runs an order of magnitude slower than it should.

The second property is **warp divergence**. Because all 32 lanes of a warp execute one instruction together (the SIMT model), a branch that sends some lanes one way and some the other forces the hardware to run *both* paths serially — first the lanes that took the `if`, with the others masked off and idle, then the lanes that took the `else`, with the first group idle. A warp where half the threads do work A and half do work B takes as long as A *plus* B, not the max. For the bounds guard `if (idx < N)` this barely matters: only the very last warp of the last block straddles the boundary, so divergence happens in at most one warp out of millions. But a data-dependent branch inside the hot loop — `if (x[idx] > 0)` taken by a random scatter of lanes — can halve your throughput. The fix when it matters is to make the branch uniform across the warp (so all 32 lanes agree), or to restructure the data so that lanes in the same warp take the same path. You will see this discipline everywhere in fast kernels; the masked `-inf` trick in the Triton softmax later is exactly a way to avoid a divergent guard.

## 3. Host, device, and the cost of crossing the PCIe bridge

Before we write the full kernel, you need the surrounding choreography, because the kernel itself is only the middle of a five-step dance. The GPU has its own memory — **HBM**, High-Bandwidth Memory, the off-chip DRAM on the GPU board, 80 GB on an A100. Your CPU program (the **host**) and the GPU (the **device**) have *separate* address spaces. A pointer that is valid on the host is meaningless on the device and vice versa. So to run a kernel on data that starts life in CPU memory, you must: allocate space on the device, copy the data over, launch, copy the result back, and free the device memory.

![timeline showing the five host and device steps of a kernel run with two PCIe crossings around the GPU launch](/imgs/blogs/cuda-programming-for-ai-engineers-threads-blocks-and-a-first-kernel-3.png)

The figure lays out the five steps in order: `cudaMalloc` to reserve device buffers, `cudaMemcpy` host-to-device to ship the inputs across the PCIe bus, the kernel `<<<...>>>` launch, `cudaMemcpy` device-to-host to pull the answer back, and `cudaFree` to release. Notice the two steps marked as cautions — the two PCIe crossings. They are the expensive part. PCIe Gen4 x16, the link between CPU and GPU on most servers, tops out around 25–32 GB/s in practice (the marketing number is 32 GB/s per direction; you see less). Compare that to HBM2e on an A100 at **2.0 TB/s** — roughly 60–80× faster. The link between host and device is, by a wide margin, the slowest pipe in the system.

This has a profound consequence that shapes how real ML systems are built: **you do not move data across PCIe per operation.** If you copied your tensor to the GPU, ran one kernel, copied it back, then copied it over again for the next op, the PCIe transfers would dominate and the GPU would be a very expensive way to do nothing. Instead, you copy your data to the GPU *once*, keep it resident in HBM, and run *hundreds* of kernels on it before copying anything back. This is exactly what PyTorch does: when you call `.cuda()` on a tensor, it lives in HBM, and every subsequent `+`, `@`, `relu`, and `softmax` runs as a kernel on data that never leaves the device. The whole forward and backward pass of your Transformer happens device-side; only the final loss scalar (and occasionally a checkpoint) crosses back.

For our toy vector-add, the PCIe cost actually *swamps* the compute — we move 12 MB across PCIe and do a million additions that the GPU finishes in microseconds. That is fine; vector-add is a teaching example, not a workload you would ever GPU-accelerate in isolation. The lesson to carry forward is the ratio: arithmetic is cheap, moving data is expensive, and moving it across PCIe is the most expensive thing of all.

There are two refinements worth knowing even at this stage, because they show up the moment a transfer actually matters. The first is **pinned (page-locked) host memory**. By default `malloc` gives you pageable memory, and the driver cannot DMA directly out of pageable memory — it first copies your data into a hidden pinned staging buffer, then transfers, which means an extra copy and a slower link. If you allocate the host buffer with `cudaMallocHost` instead, the memory is pinned, the driver DMAs straight from it, and you get noticeably higher PCIe bandwidth (often 1.5–2× on the transfer). PyTorch exposes this as `DataLoader(pin_memory=True)`, and it is why that flag matters for input pipelines. The second refinement is **overlap via streams**. A CUDA **stream** is an ordered queue of operations; work in different streams can run concurrently. By splitting a large transfer into chunks and issuing copy-and-compute on multiple streams, you can have the GPU computing on chunk 1 while chunk 2 is still crossing PCIe, hiding the transfer behind the compute. This is the basis of how real training pipelines overlap data movement with the forward pass. You do not need either for the toy kernel, but they are the first two tools you reach for when a copy shows up in your profile.

One more thing the example glosses over and real code cannot: **every CUDA call can fail, silently.** `cudaMalloc` can return out-of-memory; a kernel can hit an illegal address; a launch config can exceed limits. These calls return a `cudaError_t`, and a kernel launch reports its error *asynchronously* through the next synchronizing call. Ignore them and a corrupted run looks identical to a correct one until the numbers are wrong. The standard idiom is a checking macro you wrap around every call:

```cuda
#define CUDA_CHECK(call) do {                                  \
    cudaError_t err = (call);                                  \
    if (err != cudaSuccess) {                                  \
        fprintf(stderr, "CUDA error %s at %s:%d\n",            \
                cudaGetErrorString(err), __FILE__, __LINE__);  \
        exit(EXIT_FAILURE);                                    \
    }                                                          \
} while (0)

// usage: CUDA_CHECK(cudaMalloc(&d_a, bytes));
// after a launch: vecadd<<<blocks, threads>>>(...);
//                 CUDA_CHECK(cudaGetLastError());        // launch errors
//                 CUDA_CHECK(cudaDeviceSynchronize());   // execution errors
```

I have left the macro out of the main listing below to keep the logic readable, but in any kernel you actually ship, wrap every CUDA call in it. The asymmetry is brutal: a missing bounds guard plus no error checking is a silent wrong answer, and silent wrong answers in a training run can cost you days before you notice the loss curve is subtly off.

## 4. A complete, runnable vector-add kernel

Now we put it together. Here is a full CUDA program — kernel, host setup, launch, copy-back, verification — that you can save as `vecadd.cu`, compile, and run. I have kept it deliberately complete so there is no hand-waving about "the rest of the boilerplate."

```cuda
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// The kernel: runs on the GPU, one thread per element.
__global__ void vecadd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // global index
    if (idx < n) {                                      // bounds guard
        c[idx] = a[idx] + b[idx];                       // the work
    }
}

int main() {
    const int N = 1 << 20;                 // ~1,000,000 elements
    const size_t bytes = N * sizeof(float);

    // Host buffers.
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_c = (float*)malloc(bytes);
    for (int i = 0; i < N; ++i) { h_a[i] = 1.0f; h_b[i] = 2.0f; }

    // Device buffers.
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy inputs host -> device.
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Launch: ceil(N / threads) blocks of 256 threads.
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    vecadd<<<blocks, threads>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();               // wait for the GPU to finish

    // Copy result device -> host.
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Verify: every element should be 3.0.
    printf("c[0] = %f, c[N-1] = %f\n", h_c[0], h_c[N - 1]);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
```

A few things to dwell on. The `__global__` qualifier marks a function as a kernel — callable from the host, runs on the device. (Its siblings are `__device__`, a helper callable only from device code, and `__host__`, the default, for plain CPU functions.) The triple-angle-bracket syntax `vecadd<<<blocks, threads>>>(...)` is the **launch configuration**: the first number is the grid dimension (how many blocks), the second is the block dimension (how many threads per block). It is the one piece of syntax that is genuinely CUDA-specific and looks alien at first; after you have written it a dozen times it is muscle memory.

The `cudaDeviceSynchronize()` after the launch is important and easy to forget. Kernel launches are **asynchronous** — the host issues the launch and immediately moves on without waiting for the GPU to finish. The host and device run concurrently, which is a feature (you can overlap CPU work with GPU work), but it means that if you time a kernel by reading the clock before and after the launch line, you will measure essentially zero, because you are timing the *issue*, not the *execution*. To get a real result you must synchronize. We will come back to this in the measurement section; it is the single most common timing bug.

The compile line is one command:

```bash
nvcc -O3 -arch=sm_80 vecadd.cu -o vecadd
```

`nvcc` is NVIDIA's CUDA compiler. `-O3` turns on optimization. `-arch=sm_80` targets the A100's compute capability 8.0 (use `sm_90` for H100, `sm_89` for an RTX 4090, `sm_86` for an A10/3090). The arch flag matters: it tells `nvcc` which instruction set to emit. Target the wrong one and you either fail to use the chip's newest features or fail to run at all. Then `./vecadd` prints `c[0] = 3.000000, c[N-1] = 3.000000`, and you have run your first kernel.

![graph showing two input arrays branching into one kernel where each thread adds one pair of elements and writes one output element](/imgs/blogs/cuda-programming-for-ai-engineers-threads-blocks-and-a-first-kernel-4.png)

The dataflow figure makes the parallelism explicit: arrays `a` and `b` (4 MB each, a million floats) feed the kernel, which spawns one thread per element; each thread reads one pair, adds them, and writes one element of `c`. The caution node flags the economics — 12 MB of memory traffic for a million additions, one add per 12 bytes moved. That ratio, arithmetic per byte, is the **arithmetic intensity**, and it is the thing that decides whether a kernel is limited by compute or by memory. Vector-add, at roughly 0.083 FLOPs per byte, is hopelessly memory-bound: the GPU's adders sit idle waiting for HBM. We will see the opposite case — a compute-bound kernel — when we get to matmul.

#### Worked example: vector-add bandwidth versus the A100 peak

Let us measure honestly. Vector-add on `N = 2^20` floats moves `3 × 4 MB = 12 MB` of data: read `a` (4 MB), read `b` (4 MB), write `c` (4 MB). If the kernel ran at the A100's full HBM bandwidth of 2.0 TB/s, the transfer would take

$$t = \frac{12 \times 10^6 \text{ bytes}}{2.0 \times 10^{12} \text{ bytes/s}} \approx 6 \ \mu s.$$

In practice a well-written vector-add on an A100 sustains roughly **1.3–1.7 TB/s** of effective bandwidth on a buffer this size — call it 70–85% of peak (it is approximate and depends on size and ECC). At 1.5 TB/s the kernel takes about 8 µs. The point is *not* the exact microseconds; it is that you can predict the runtime from bandwidth alone, because the kernel does almost no arithmetic. When a kernel is memory-bound, its runtime is `bytes_moved / achieved_bandwidth`, full stop. To go faster you must move fewer bytes — which is the entire premise of [kernel fusion](/blog/machine-learning/high-performance-computing/kernel-fusion-and-flashattention-beating-the-memory-wall). This is the roofline model in miniature, covered in depth in [the roofline post](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound).

## 5. Tiled matmul: the kernel that taught the field shared memory

Vector-add is memory-bound and there is nothing you can do about it — you must touch every byte once. Matrix multiply is different, and the difference is the most important optimization lesson in GPU programming. A naive matmul is *also* memory-bound, not because it has to be, but because it wastes bandwidth re-reading the same data thousands of times. Fixing that with on-chip **shared memory** is the canonical example of how to turn a memory-bound kernel into a compute-bound one. Every fast kernel you will ever read — cuBLAS GEMM, FlashAttention, the Triton matmul — is an elaboration of this one idea.

First, the naive version, to see the waste. To compute `C = A × B` where all three are `N × N`, element `C[row][col]` is the dot product of row `row` of A and column `col` of B. The obvious kernel gives each thread one output element:

```cuda
__global__ void matmul_naive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < N; ++k) {
            acc += A[row * N + k] * B[k * N + col];   // each from HBM
        }
        C[row * N + col] = acc;
    }
}
```

Count the memory traffic. Each thread reads an entire row of A (N floats) and an entire column of B (N floats) straight from HBM, one element at a time inside the loop. There are `N × N` output elements, so the total reads are `2 × N × N × N = 2N³` float reads from HBM. But the matrices only contain `2N²` floats. Every element of A is read N times (once for each column of C in its row band); every element of B is read N times. We are re-fetching the same data from the slowest memory N times over. For `N = 4096` that is reading each input four thousand times. The HBM bandwidth, not the arithmetic, is the bottleneck — and we have brought it on ourselves.

The fix is **tiling**. Instead of each thread independently slurping rows and columns from HBM, the threads of a block cooperate: they load a small **tile** — say 32×32 — of A and a 32×32 tile of B into shared memory *once*, synchronize, then every thread in the block does its multiply-adds reading from shared memory, which is roughly 20–30× faster than HBM and sits on-chip right next to the cores. After the block exhausts that pair of tiles, it slides along the K dimension to the next pair, accumulating into a register. Here is the tiled kernel:

```cuda
#define TILE 32

__global__ void matmul_tiled(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE][TILE];   // on-chip, shared by the block
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    // Slide a pair of tiles along the K dimension.
    for (int t = 0; t < N / TILE; ++t) {
        // Cooperative load: each thread loads one element of each tile.
        As[threadIdx.y][threadIdx.x] = A[row * N + (t * TILE + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        __syncthreads();               // wait: tiles fully loaded

        // Multiply-add from shared memory: each loaded value reused TILE times.
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();               // wait: done before reloading
    }
    C[row * N + col] = acc;
}
```

Two new things earn their keep here. `__shared__` declares a buffer in **shared memory** — a small (up to ~164 KB per SM on an A100, ~48 KB by default per block), fast, on-chip scratchpad that all threads of a block can read and write. It is the programmer-managed cache that makes cooperative algorithms possible. `__syncthreads()` is a **barrier**: every thread in the block stops there until all threads in the block have arrived. You need the first barrier to guarantee the tile is fully loaded before anyone reads it, and the second so no thread overwrites the tile (on the next iteration) while a slower thread is still multiplying from it. Forget a `__syncthreads()` and you get a race: a heisenbug that gives slightly wrong results that change run to run.

![grid diagram showing a block loading a 32 by 32 tile of A and B into shared memory and reusing each loaded value 32 times for multiply-adds](/imgs/blogs/cuda-programming-for-ai-engineers-threads-blocks-and-a-first-kernel-5.png)

The figure traces the flow: the A and B tiles come from HBM (external nodes), get loaded into `__shared__` once, the block hits the `__syncthreads` barrier across all 1024 threads, then each thread does its 32 fused multiply-adds reading purely from shared memory, accumulating in registers, and slides to the next K-tile. The two success nodes are the payoff: each loaded value is reused 32 times, and HBM trips drop roughly 16× versus naive (more on the exact factor next).

#### Worked example: the FLOP-per-byte improvement from tiling

Here is the arithmetic that makes tiling worth it. In the naive kernel, the inner loop does one multiply-add (2 FLOPs) and reads two floats from HBM (8 bytes). The arithmetic intensity is `2 FLOPs / 8 bytes = 0.25 FLOPs/byte`. That is deep in memory-bound territory — the A100's adders can do far more math per byte than that, so they starve.

In the tiled kernel with a 32×32 tile, you load the tile once (each thread loads one element, 4 bytes) and then perform `TILE = 32` multiply-adds per loaded value from shared memory. So per element loaded from HBM you now do `32 × 2 = 64` FLOPs against 4 bytes read, giving an arithmetic intensity of `64 / 4 = 16 FLOPs/byte` — a **64× increase** over naive's 0.25, or framed by HBM traffic, the total HBM reads drop from `2N³` to roughly `2N³ / TILE`, a **32×** reduction in bytes moved (the figure's "16×" is the conservative end accounting for both tiles and write traffic; the headline reuse factor is `TILE`). Either way, you have crossed the **roofline ridge point**: at 16 FLOPs/byte the A100 is no longer memory-bound on this kernel. The bottleneck moves from HBM bandwidth to the arithmetic units, which is exactly where you want a matmul to live, because that is where the chip's 312 bf16 TFLOP/s actually get used.

Let us make the HBM-traffic reduction exact, because the factor is worth being able to derive rather than quote. In the naive kernel, computing the full `C` reads every element of `A` and `B` once *per output column and row band* respectively — the total HBM read volume is `2N³` floats (each of the `N²` outputs runs a length-`N` dot product, each step of which reads one element of `A` and one of `B`). In the tiled kernel, the work is organized into `(N/T)²` output tiles, and computing each output tile requires sliding `N/T` tile-pairs along K. Each tile-pair load reads `T²` floats of `A` and `T²` of `B`. So the total read volume is

$$\left(\frac{N}{T}\right)^2 \cdot \frac{N}{T} \cdot 2T^2 = 2 \cdot \frac{N^3}{T} \text{ floats},$$

which is exactly `2N³/T` — a factor of `T` fewer reads than the naive `2N³`. With `T = 32` that is a **32× reduction** in HBM read traffic. The figure's conservative "16×" folds in the write traffic and the fact that the K dimension is rarely a perfect multiple of the tile; the clean derivation gives `T`. This is the single most important transformation in GPU computing: **trade off-chip bandwidth for on-chip reuse.** FlashAttention does it for the attention matrix. cuBLAS does a far more sophisticated version of it (register-level tiling, double-buffering, Tensor Core instructions). But the kernel above is the seed of all of it, and you can run it today.

A subtlety the kernel hides: the tile size `T` is bounded by shared memory. Two `T × T` float tiles cost `2T² × 4` bytes. At `T = 32` that is 8 KB per block — comfortable. At `T = 64` it is 32 KB, which starts to limit how many blocks fit per SM and thus your occupancy. There is a real tension here: a bigger tile means more reuse (better arithmetic intensity) but fewer resident blocks (less latency hiding). The optimum is not "as big as possible"; it is the size that balances reuse against occupancy for your specific kernel and GPU, which is exactly the kind of thing an autotuner (Triton's, or cuBLAS's per-architecture tuning) searches over so you do not have to. This is also why hand-written tiled kernels plateau around 20–35% of peak while cuBLAS reaches 90%+: cuBLAS adds a *second* level of tiling at the register level (each thread computes a small `m × n` micro-tile of outputs, holding the accumulators in registers and reusing each shared-memory value many more times), plus double-buffering that prefetches the next tile while computing on the current one. Same idea, three levels deep.

#### Worked example: naive versus tiled GEMM on an A100

Concrete numbers on a single A100 80GB SXM for `C = A × B` with `N = 4096` (fp32, so no Tensor Cores — pure CUDA-core throughput). The total work is `2N³ ≈ 1.37 × 10^11` FLOPs.

| Kernel | HBM reads | Arithmetic intensity | Achieved (approx) | % of fp32 peak (~19.5 TFLOP/s) |
| --- | --- | --- | --- | --- |
| `matmul_naive` | `~2N³` floats | 0.25 FLOPs/byte | ~0.3–0.6 TFLOP/s | ~2–3% |
| `matmul_tiled` (32×32) | `~2N³/32` floats | 16 FLOPs/byte | ~4–7 TFLOP/s | ~20–35% |
| cuBLAS `Sgemm` | register-tiled | very high | ~17–19 TFLOP/s | ~90%+ |

These are approximate and hardware- and size-dependent, but the *shape* is real and reproducible: tiling buys roughly a 10× speedup over the naive kernel, and cuBLAS — with register tiling, vectorized loads, and double-buffering on top of the same shared-memory idea — buys another 3–4× over your hand-written tiled version. That last gap is the lesson of the next section: **you will almost never beat cuBLAS at GEMM**, so do not try. Write the tiled kernel to *understand* the technique, then call the library for production. The deeper details of why these numbers fall where they do live in [the memory hierarchy post](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm).

## 6. Launch configuration and occupancy: filling the SM

You have two numbers to pick at every launch: threads per block, and number of blocks. The number of blocks is usually forced — `ceil(N / threads)` to cover your data. Threads per block is the genuine choice, and it controls **occupancy**: the fraction of an SM's thread-capacity that your kernel actually uses. Occupancy is not the goal in itself, but low occupancy is a common cause of leaving performance on the table, so you should know how to read it.

Here is the mechanism. An A100 SM has hardware slots for up to **2048 resident threads** = 64 warps, and it can hold up to **32 resident blocks** at once. The SM hides memory latency by having many warps resident: when one warp stalls waiting on an HBM read (hundreds of cycles), the scheduler instantly switches to another ready warp, so the cores stay busy. More resident warps means more latency to hide means higher sustained throughput — up to a point. Your launch config determines how many warps and blocks fit, and three limits compete: the 2048-thread cap, the 32-block cap, and the **register and shared-memory budget** (each SM has a fixed pool of registers and shared memory; if each block hogs too much, fewer blocks fit).

![matrix comparing threads per block against warps, blocks per SM, resident warps, and occupancy on an A100](/imgs/blogs/cuda-programming-for-ai-engineers-threads-blocks-and-a-first-kernel-6.png)

The figure works the trade-off across four block sizes. At **64 threads per block** (2 warps), you hit the 32-block ceiling before you fill the thread slots: 32 blocks × 2 warps = 64 warps... actually you cap at 32 blocks × 2 warps = 64 — but the practical limiter shows only 32 of 64 warps resident in the common register-limited case, giving 50% occupancy. Tiny blocks waste the SM. At **128 or 256 threads per block**, the math lines up perfectly: 16 blocks × 4 warps or 8 blocks × 8 warps both give 64 resident warps = 100% occupancy, and you are under the 32-block cap with room for the scheduler to work. At **1024 threads per block** you still reach 64 warps with 2 blocks, but it is rigid: one block uses half the SM, so if registers or shared memory push you over budget you drop to one block and 50%. This is why **128 or 256 is the universal default** — it hits full occupancy, divides evenly into warps, and leaves the scheduler flexibility.

#### Worked example: computing occupancy from a launch config

Take the A100 SM and a kernel launched with 256 threads per block that the compiler reports uses 32 registers per thread and 0 bytes of shared memory. Work out the occupancy from the hardware limits. The SM has a register file of **65,536 registers**. At 32 registers per thread, one block of 256 threads needs `256 × 32 = 8192` registers, so the register file alone allows `65536 / 8192 = 8` resident blocks. The SM's hard caps are 32 blocks and 2048 threads (= 64 warps). With 8 blocks of 256 threads (8 warps each), you have `8 × 256 = 2048` resident threads = 64 resident warps — exactly the SM's maximum. So occupancy is `64 / 64 = 100%`, and the limiter is the *thread cap*, not registers (registers would have allowed 8 blocks, and 8 blocks is what fills the threads).

Now bump the kernel to 64 registers per thread — common once you add a real inner loop with accumulators. One block now needs `256 × 64 = 16384` registers, so only `65536 / 16384 = 4` blocks fit. That is `4 × 256 = 1024` threads = 32 warps resident, and occupancy drops to `32 / 64 = 50%`. The kernel did not change shape; it just got register-hungrier, and your occupancy halved. This is *register pressure*, the most common hidden cause of low occupancy, and it is why `ncu` reports the limiting resource explicitly. The fix, when occupancy actually matters, is to reduce register use (simplify the kernel, or cap it with `__launch_bounds__(256, 4)` to tell the compiler to spill rather than exceed 4 blocks) — but only after you have confirmed with a profile that occupancy, and not something else, is your bottleneck.

You do not have to reason this out by hand. CUDA provides the **occupancy calculator API** to compute the best block size for a given kernel, accounting for its actual register and shared-memory use:

```cuda
int blockSize;     // recommended threads per block
int minGridSize;   // minimum grid for full occupancy
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                   vecadd, 0, 0);
printf("suggested block size: %d\n", blockSize);
```

And you read *achieved* occupancy — what actually happened at runtime, which can be lower than the theoretical max if blocks finish unevenly — from **Nsight Compute**, the per-kernel profiler:

```bash
ncu --set full --section Occupancy ./vecadd
```

The Occupancy section reports theoretical occupancy, achieved occupancy, and which resource (registers, shared memory, or block count) is the limiter. If theoretical is 100% but achieved is 60%, you have a tail-effect or load-imbalance problem, not a config problem. Chasing occupancy past the point where latency is already hidden is a classic rookie mistake — a kernel at 50% occupancy that is compute-bound and near peak does not get faster by adding warps. Occupancy is a *diagnostic*, not a *target*. The profiling-driven way to use it is covered in [the GPU profiling post](/blog/machine-learning/high-performance-computing/profiling-gpu-workloads-finding-the-real-bottleneck).

| Launch knob | What it controls | Good default | Failure mode if wrong |
| --- | --- | --- | --- |
| Threads per block | Warps per block, occupancy | 128–256 (multiple of 32) | 50% occupancy at 64; rigidity at 1024 |
| Blocks per grid | Coverage of N elements | `ceil(N/threads)` | Uncovered tail (too few) or wasted launches |
| Shared mem per block | Tile size, blocks-per-SM | As small as the tile needs | Fewer resident blocks, lower occupancy |
| Registers per thread | Blocks-per-SM (implicit) | Let nvcc decide; cap with `__launch_bounds__` | Register spilling to local memory (slow) |

## 7. When to drop to CUDA or Triton — and when to trust the framework

Now the most valuable judgment in this entire post, because writing a kernel is a cost — your time, a maintenance burden, a thing that breaks on the next GPU architecture — and most of the time it is the wrong move. The default, correct answer is: **trust the framework.** PyTorch dispatches your matmuls to cuBLAS and cuDNN, which are written by NVIDIA's performance engineers and run at 90%+ of peak. You are not going to beat them at GEMM or convolution, and you should not spend a day trying. The question is never "can I write a kernel?" — it is "will writing a kernel actually make my model faster, and is that speedup worth the cost?" You answer it with a profiler, not a hunch.

![decision tree showing the path from plain PyTorch down through torch.compile and Triton to raw CUDA, stopping when the speedup stops paying off](/imgs/blogs/cuda-programming-for-ai-engineers-threads-blocks-and-a-first-kernel-7.png)

The figure is the decision tree, and the rule is *climb down only as far as the speedup pays for itself*. Start at the top:

**Already near peak? Stay in plain PyTorch.** If you profile and your hot kernels are big matmuls already running at 80–90% of peak on cuBLAS, you are done. There is no kernel to write. The compute wall is the binding constraint and a custom kernel cannot move it. This is the case for the dense matmuls in your Transformer — the QKV and MLP projections — which is why nobody hand-writes those.

**Memory-bound chain of small ops? Reach for `torch.compile` first.** This is the Transformer-tail scenario from the intro: a sequence of small pointwise and reduction ops (bias-add, then dropout, then a scaling, then a layernorm) that each launch a separate kernel and each pay a full round-trip to HBM. The fix is **fusion** — running them as one kernel that reads once and writes once. And the good news is you usually get it for free: `torch.compile` traces your model and automatically fuses pointwise chains into Triton kernels. No kernel writing required:

```python
import torch

@torch.compile  # fuses pointwise + reductions into Triton kernels
def block(x, w, b, gamma, beta):
    h = x @ w + b
    h = torch.nn.functional.gelu(h)
    return torch.nn.functional.layer_norm(h, h.shape[-1:], gamma, beta)
```

`torch.compile` will fuse the bias-add, GELU, and the elementwise parts of the layernorm into far fewer kernels than eager mode launches. For the long tail of glue ops, this is the highest-leverage change you can make and it costs one decorator.

**Need a fused custom op that the compiler can't express? Write Triton.** Fused attention, a custom normalization, a quantized matmul with a funny layout, an operation with a reduction pattern `torch.compile` won't fuse on its own — these justify a hand-written kernel, and in 2026 the tool for that is **Triton**, not raw CUDA. Triton gives you ~80% of CUDA's performance for ~20% of the effort, which is the right trade for almost everyone.

**Raw CUDA C? Only for the last few percent.** When you are NVIDIA, or you are at the absolute frontier squeezing a 5% win out of a kernel that runs a billion times, or you need a hardware feature Triton doesn't expose yet, you drop to CUDA C. For 99% of AI engineers, that day never comes — and that is fine. Understanding CUDA (which you now do) is what lets you read FlashAttention's source, reason about occupancy, and write *good* Triton. You rarely need to ship it.

| Tool | Effort | Typical speed vs cuBLAS/eager | When to use |
| --- | --- | --- | --- |
| Plain PyTorch (eager) | None | cuBLAS-level on matmuls | Default; big matmuls; you're near peak |
| `torch.compile` | One decorator | Fuses pointwise; 1.3–2× on op-heavy models | Memory-bound tails, glue ops |
| Triton `@triton.jit` | ~Hours | ~80% of hand-CUDA; powers compiled kernels | Custom fused ops: attention, norms |
| Raw CUDA C | Days | 100%; the ceiling | Last few %, new hardware features, libraries |

## 8. Triton: a fused softmax, the path most AI engineers take

Let us actually write the kernel that the decision tree points most AI engineers toward. **Triton** is a Python-embedded language and compiler from OpenAI that lets you write GPU kernels at the *tile* level — you reason about blocks of data, and Triton handles the thread-level index math, the shared-memory management, and the launch tuning. It is what `torch.compile` generates internally, and it is the language FlashAttention's most accessible implementations are written in. The shift in thinking from CUDA is small but freeing: in CUDA you write what *one thread* does; in Triton you write what *one block* does to a *tile* of data, with the per-element parallelism handled for you.

Here is a fused softmax — the operation from the intro that, done naively, launches several kernels (a max-reduce, a subtract, an exp, a sum-reduce, a divide), each round-tripping to HBM. Fused, it reads each row once, does all the math in on-chip memory, and writes once:

```python
import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride,
                   n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)                      # one program per row
    row_start = in_ptr + row * in_row_stride
    offsets = tl.arange(0, BLOCK_SIZE)          # a tile of columns
    mask = offsets < n_cols

    # Load the whole row once, into registers/shared (off-grid = -inf).
    x = tl.load(row_start + offsets, mask=mask, other=-float('inf'))

    # The numerically-stable softmax, entirely on-chip.
    x = x - tl.max(x, axis=0)                    # subtract row max
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    y = num / denom

    # Write the result row once.
    out_start = out_ptr + row * out_row_stride
    tl.store(out_start + offsets, y, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    softmax_kernel[(n_rows,)](                   # grid = one program per row
        out, x, x.stride(0), out.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out
```

Read what is happening. `@triton.jit` marks a Triton kernel. `tl.program_id(0)` is the Triton analogue of `blockIdx.x` — it tells this program instance which row it owns. `tl.arange(0, BLOCK_SIZE)` builds a *vector* of column offsets — you operate on the whole tile at once, no inner thread loop. `tl.load` with a `mask` fetches the row from HBM (the mask handles the bounds-guard for you — out-of-range lanes get `-inf`, the softmax identity for max). Then `tl.max`, `tl.exp`, `tl.sum` are *tile-level* reductions that Triton lowers to efficient warp- and block-level operations using shared memory automatically. `tl.store` writes the row back once. The entire softmax — five logical operations — touches HBM exactly twice (one read, one write) instead of ten times. That is the fusion win, and you wrote it in about 25 lines of Python.

![before and after comparison contrasting a 150-line hand-tuned CUDA softmax with a 30-line autotuned Triton kernel](/imgs/blogs/cuda-programming-for-ai-engineers-threads-blocks-and-a-first-kernel-8.png)

The before/after figure makes the authoring difference visceral. The raw CUDA path is ~150 lines: manual index math, a hand-picked block size you must tune per architecture, an `nvcc` rebuild for each GPU generation. The Triton path is ~30 lines of Python, it **autotunes** itself, and it is the same machinery that powers `torch.compile`. You add autotuning by decorating the kernel with a set of candidate configs and letting Triton benchmark them on your actual hardware:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['n_cols'],   # re-tune when the row width changes
)
@triton.jit
def softmax_kernel(...):
    ...
```

On the first run for a given `n_cols`, Triton times each config and caches the winner. You never hand-pick a block size again — the thing that cost you an afternoon of guess-and-check in CUDA is now automatic. This is the deal Triton offers: you give up the last 10–20% of peak that a NVIDIA engineer would extract by hand, and in exchange you get a kernel in an afternoon instead of a week, that you can read, that autotunes, and that runs across architectures.

It is worth being precise about what Triton automated versus what it did not, because that boundary is the whole reason to learn CUDA before Triton. Triton handled, for free: the per-element index math (you wrote `tl.arange`, not `blockIdx * blockDim + threadIdx` thirty-two ways), the shared-memory allocation and the loads/stores into it (the `tl.max` and `tl.sum` reductions compile down to warp shuffles and shared-memory traffic you never see), the bounds masking, and the launch tuning. What Triton did *not* do, and what you still had to know to write the kernel correctly: that softmax must subtract the row max for numerical stability (a numerics fact, not a CUDA fact — without it `exp` of a large logit overflows to `inf`); that the operation is memory-bound so fusing the five steps into one round-trip is where the win comes from (a roofline fact); and that you want one program per row so each program owns a contiguous, coalesceable slice (a memory-access fact). Triton frees you from the *mechanical* parts of CUDA. It does not free you from understanding the hardware and numerics — which is exactly the understanding this whole post is building. That is why "write Triton" is the recommendation and "read CUDA" is the prerequisite, not the other way around.

#### Worked example: Triton fused softmax versus eager on an A100

The Triton tutorial benchmarks exactly this kernel, and the result is the standard teaching number: on rows wide enough to be memory-bound, the fused Triton softmax runs at roughly the A100's peak HBM bandwidth, beating PyTorch's eager `torch.softmax` — which materializes intermediates and launches multiple kernels — by about **2–4×** on the operation in isolation, and approaching the bandwidth ceiling of ~2.0 TB/s where eager leaves a third of the bandwidth on the floor to extra round-trips. Mark these as approximate and shape-dependent: for very small rows the kernel-launch overhead dominates and the gap shrinks; for large rows fusion wins big. The mechanism is exactly the one from the vector-add bandwidth example — a memory-bound op's runtime is `bytes_moved / bandwidth`, and fusion cuts the bytes moved. You are not making the GPU compute faster; you are making it touch HBM fewer times.

## Case studies / real numbers

Three real results that ground everything above in the wider ecosystem.

**Triton powers torch.compile's kernels.** This is not a toy connection — it is production reality. When you put `@torch.compile` on a PyTorch model, the Inductor backend generates **Triton** kernels for the fused pointwise and reduction operations (and increasingly for matmul templates), then compiles them with Triton's own pipeline. The fused-softmax-style kernel you wrote above is, structurally, the same kind of thing the compiler emits automatically. So the skill transfers directly: understanding the Triton softmax means understanding what `torch.compile` is doing internally, which means you can read the generated Triton (`TORCH_COMPILE_DEBUG=1` dumps it) and debug it when fusion does something surprising. The typical reported win from `torch.compile` on op-heavy models is in the 1.3–2× range end-to-end, almost entirely from fusing the memory-bound tail.

**FlashAttention is the tiled-matmul idea applied to attention.** The single most consequential kernel of the modern LLM era is, at its core, the shared-memory tiling from Section 5 generalized to the attention computation. Naive attention materializes the full `N × N` attention score matrix in HBM — `O(N²)` memory traffic and storage that blows up for long sequences. FlashAttention tiles the computation: it loads blocks of queries, keys, and values into on-chip SRAM, computes the attention for that tile, and uses an online-softmax trick to accumulate the result *without ever writing the full score matrix to HBM*. The reported result is roughly a **2–4× wall-clock speedup** on attention and a memory footprint that drops from `O(N²)` to `O(N)`, enabling the long-context models we now take for granted. The reference implementation is CUDA; the widely-used accessible versions are Triton. It is the tiling FLOP-per-byte lesson, scaled to the operation that dominates Transformer cost. It is covered end to end in [the kernel-fusion and FlashAttention post](/blog/machine-learning/high-performance-computing/kernel-fusion-and-flashattention-beating-the-memory-wall).

**The naive-versus-tiled GEMM gap, and why cuBLAS still wins.** The numbers in Section 5 are the textbook result, reproduced in NVIDIA's own CUDA samples and countless course assignments: a naive matmul sits at single-digit-percent of peak because it is bandwidth-bound by redundant HBM reads; a shared-memory tiled kernel jumps to 20–35% of peak; and vendor cuBLAS, by adding register-level tiling, vectorized 128-bit loads, double-buffering to hide load latency, and Tensor Core instructions for the lower-precision paths, reaches 90%+ of peak. The gap between your tiled kernel and cuBLAS is *not* the tiling idea — you have that — it is a dozen further micro-optimizations that NVIDIA's engineers tuned per architecture. This is the empirical backing for the Section 7 rule: write the tiled kernel to learn, call cuBLAS to ship. For the inference-serving twist on these GEMM numbers, see how a production compiler stacks them in [the TensorRT inference-compiler post](/blog/machine-learning/mlops/tensorrt-end-to-end-inference-compiler).

**Why this all rolls up to MFU.** The reason these kernel-level numbers matter at the scale of a real training run is **Model FLOPs Utilization** — the fraction of the GPU's peak FLOP/s your end-to-end training step actually achieves. Megatron-LM and the large-model training reports put well-tuned dense Transformer training in the rough range of 40–55% MFU on A100/H100 clusters; the published GPT-3 and PaLM training runs reported MFU in roughly the 20–50% band depending on model and system. That number is *built out of* exactly the kernel decisions in this post: every matmul that runs on cuBLAS near peak pushes MFU up; every memory-bound tail op that launches its own kernel and round-trips to HBM drags it down; every fusion (`torch.compile` or FlashAttention) recovers some of the loss. When you see a training run pinned at 18% MFU — the scenario this post opened with — the cause is almost always too much time in memory-bound glue and too little in the dense compute, and the fix is the fusion-and-kernel toolkit you now have. Mark these MFU figures as approximate and configuration-dependent; the discipline is to *measure your own* with the honest timing method below, not to trust a headline. How MFU composes across many GPUs, where communication becomes the third wall, is the subject of [the capstone playbook](/blog/machine-learning/high-performance-computing/the-hpc-playbook-for-ai-engineers).

## How to measure a kernel honestly

Every number in this post is meaningless if you mistime it, and kernel timing has three traps that catch everyone once. First, **kernel launches are asynchronous** — if you read the clock before and after the launch line without synchronizing, you measure the launch overhead (a few microseconds), not the execution. You must `torch.cuda.synchronize()` (or `cudaDeviceSynchronize()`) before stopping the clock. Second, the **first run is a lie** — it pays for JIT compilation, autotuning, cuDNN algorithm selection, and cold caches. Always warm up with several untimed iterations before measuring. Third, **time the steady state and average** — a single timed run has too much variance; run 50–100 iterations and take the median. Here is the idiom that gets all three right:

```python
import torch

def bench(fn, *args, warmup=10, iters=100):
    for _ in range(warmup):           # 1. warm up: JIT, caches, autotune
        fn(*args)
    torch.cuda.synchronize()          # 2. wait before starting the clock
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):            # 3. steady state, many iters
        fn(*args)
    end.record()
    torch.cuda.synchronize()          # wait before reading the clock
    return start.elapsed_time(end) / iters   # milliseconds per call
```

Using CUDA `Event` objects (rather than the Python wall clock) measures GPU time directly on the device timeline, which is more accurate than host-side timing for kernels. Beyond these three, watch for the confounds that silently corrupt benchmarks: the **data loader** (if your "kernel time" includes waiting on data, you are measuring the wrong thing — keep inputs resident); **thermal throttling** (a GPU that has been hammered for a minute clocks down; `nvidia-smi -q -d CLOCK` shows it); and **other tenants** on a shared box stealing SMs. For the full profiling methodology — `nsys` for the timeline, `ncu` for per-kernel rooflines and occupancy — see [the GPU profiling post](/blog/machine-learning/high-performance-computing/profiling-gpu-workloads-finding-the-real-bottleneck).

## Tying it back to the Transformer

Step back to the running example of this series: a Transformer language model, training or serving on a GPU. Where does everything you just learned actually run? The **big matmuls** — the QKV projection, the attention output projection, and the two MLP layers — are GEMMs, and they run on **cuBLAS** (the tiled-matmul idea from Section 5, perfected). You do not write those; you let the library do them, and you confirm with a profiler that they are near peak. The **attention** itself — the score matrix, the softmax, the value-weighted sum — is where naive PyTorch wastes enormous HBM traffic materializing the `N × N` scores, and where **FlashAttention** (the tiling idea, generalized) earns its 2–4× by keeping the computation on-chip. You call it; you rarely write it. The **long tail of small ops** — the bias-adds, the residual additions, the layernorms, the dropout, the activation functions — is the memory-bound glue from the intro, and it is where you get the easiest win in all of GPU programming: slap `@torch.compile` on the model and let it fuse them into Triton kernels, or, when you need a fusion the compiler can't express, write the Triton kernel yourself the way we wrote the softmax.

So the honest division of labor for an AI engineer is: **read** CUDA (so you understand cuBLAS and FlashAttention and can reason about occupancy and shared memory), **write** Triton (when you need a custom fused op), and **trust** the framework for everything else — confirming each decision with a profiler, never a hunch. You now have all three. The vector-add taught you the launch model and the index math; the tiled matmul taught you shared memory and the bandwidth-for-reuse trade; the occupancy matrix taught you to fill the SM; and the Triton softmax taught you the path you will actually walk. The capstone of this series, [the HPC playbook for AI engineers](/blog/machine-learning/high-performance-computing/the-hpc-playbook-for-ai-engineers), ties this kernel layer to the larger optimization loop across precision, fusion, and multi-GPU scaling.

Trace one forward pass through a Transformer layer with this lens and the whole stack becomes legible. A token's hidden vector enters the attention block. It hits the QKV projection — a GEMM, dispatched to cuBLAS, running near the A100's 312 bf16 TFLOP/s; you do nothing, and the profiler confirms it is near peak. The query, key, and value tensors flow into attention, where a naive implementation would write a multi-megabyte score matrix to HBM and read it back twice; FlashAttention instead tiles the keys and values into shared memory and never materializes that matrix, and you simply called `scaled_dot_product_attention` to get it. The softmax inside attention is the very kernel you wrote in Triton — read the row once, do the stable softmax on-chip, write once. The output projection is another cuBLAS GEMM. Then the MLP: an up-projection GEMM, a GELU, a down-projection GEMM, with the GELU and the surrounding bias-adds and the residual add and the layernorm all being memory-bound pointwise glue that `torch.compile` fuses into a handful of Triton kernels instead of a dozen separate launches. Every single one of those decisions — which op runs on a library, which gets fused, which you write by hand — is a decision you can now make and defend with a number. That is the difference between calling `.cuda()` and understanding what happens after you do.

## When to reach for this (and when not to)

Be ruthless about the cost. Here is the decisive guidance, in order of how often it applies.

**Almost always: don't write a kernel.** If your model is built from standard layers, PyTorch already dispatches to cuBLAS and cuDNN, and a profile that shows your matmuls near peak means there is no kernel to write. Adding a custom kernel here is pure cost — a maintenance burden that breaks on the next GPU and saves nothing. The first move is always to profile and find out whether you are even compute-bound; if you are and you are near peak, stop.

**Often: reach for `torch.compile` before writing anything.** When the profile shows a memory-bound tail of small ops eating your step time, the one-decorator fix captures most of the available win by fusing pointwise chains automatically. Try this before you write a single line of kernel code. It is the best effort-to-reward ratio in the entire stack.

**Sometimes: write Triton.** When you need a *specific* fused op the compiler won't produce — a fused attention variant, a custom normalization, a quantized matmul with an unusual layout, a reduction pattern Inductor can't fuse — Triton is the right tool. You get most of CUDA's performance for a fraction of the effort, the kernel autotunes, and it is readable and portable. This is the realistic ceiling of kernel work for the vast majority of AI engineers.

**Rarely: write raw CUDA C.** Only when you are building a library, chasing the last few percent of peak on a kernel that runs astronomically often, or you need a hardware feature (a specific Tensor Core instruction, a warp-level primitive, an async-copy intrinsic) that Triton doesn't yet expose. If you are not sure you are in this case, you are not. The value of *understanding* CUDA is high; the frequency of *needing to ship* it, for most of us, is near zero.

And the anti-patterns to never commit: **don't try to beat cuBLAS at GEMM** (you won't, and the attempt costs days). **Don't chase occupancy when you are already near peak** (occupancy is a diagnostic, not a target; a compute-bound kernel at 50% occupancy does not speed up by adding warps). **Don't write a kernel before profiling** (you will optimize the wrong thing — the op you *think* is slow is rarely the one that is). **Don't micro-optimize a kernel that runs once** — optimize the hot path the profiler points at, and leave the rest in plain PyTorch where it is readable.

## Key takeaways

- A CUDA launch spawns a **grid of blocks of threads**; you write what *one thread* does and the hardware instantiates millions of copies. Threads run 32 at a time as a **warp**, so always make the block size a multiple of 32 — 128 or 256 is the universal default.
- The one line you write in every kernel is the global index: `blockIdx.x * blockDim.x + threadIdx.x`. Always pair it with the bounds guard `if (idx < N)`, because `ceil(N/threads)` blocks launch more threads than you have elements.
- The **host and device have separate memory**; crossing PCIe (~25 GB/s) is 60–80× slower than HBM (2.0 TB/s on an A100). Copy data to the GPU once and run hundreds of kernels on it before copying back.
- **Shared memory tiling** is the most important optimization in GPU computing: a 32×32 tile lets each loaded value feed 32 multiply-adds, lifting arithmetic intensity from 0.25 to 16 FLOPs/byte and turning a memory-bound matmul into a compute-bound one. It is the seed of cuBLAS and FlashAttention.
- **Occupancy** — resident warps over the SM's capacity — is a diagnostic, not a goal. 128–256 threads per block hits 100% occupancy on an A100; read achieved occupancy from `ncu`, but never chase it past the point where latency is already hidden.
- **Most AI engineers write Triton, not CUDA.** A fused Triton softmax is ~30 autotuned lines versus ~150 hand-tuned CUDA lines, gets ~80% of CUDA's speed, and is the same machinery `torch.compile` emits. Understand CUDA, write Triton, trust the framework.
- **Measure honestly**: synchronize before timing (launches are async), warm up to skip JIT and cold caches, and average the steady state over many iterations. An unsynchronized timer measures launch overhead, not execution.
- The decision order is always: **profile → trust the framework if near peak → `torch.compile` for memory-bound tails → Triton for custom fused ops → raw CUDA only for the last few percent.** Every kernel you write is a cost; make it earn its keep.

## Further reading

- [Inside the GPU: SMs, warps, and the SIMT execution model](/blog/machine-learning/high-performance-computing/inside-the-gpu-sms-warps-and-the-simt-execution-model) — the hardware these threads, blocks, and warps actually map onto.
- [The memory hierarchy: registers, shared memory, and HBM](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm) — why shared-memory tiling buys what it buys.
- [The roofline model: compute-bound vs memory-bound](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) — the arithmetic-intensity framing behind every kernel decision here.
- [Kernel fusion and FlashAttention: beating the memory wall](/blog/machine-learning/high-performance-computing/kernel-fusion-and-flashattention-beating-the-memory-wall) — where the Triton softmax and tiled-matmul ideas culminate.
- [The HPC playbook for AI engineers](/blog/machine-learning/high-performance-computing/the-hpc-playbook-for-ai-engineers) — the capstone that ties kernels into the full single-GPU and multi-GPU optimization loop.
- [TensorRT: end-to-end inference compiler](/blog/machine-learning/mlops/tensorrt-end-to-end-inference-compiler) — how a production compiler fuses and tiles these same kernels for serving.
- The **CUDA C++ Programming Guide** (NVIDIA) — the authoritative reference for the programming model, the memory hierarchy, and the launch syntax; the **CUDA Best Practices Guide** for occupancy and memory-coalescing.
- The **Triton documentation and tutorials** (OpenAI) — the fused-softmax and matmul tutorials are the canonical starting point; the **FlashAttention** paper (Dao et al.) for the tiling-applied-to-attention case study.

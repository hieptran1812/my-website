---
title: "What torch.compile Actually Does: Dynamo, Guards, and Inductor Fusion"
date: "2026-07-14"
publishDate: "2026-07-14"
description: "You wrapped your model in torch.compile, the second call ran 1.8x faster, and the first ran slower. This is the layer-by-layer story of what happened in between: how Dynamo traces a graph, how guards decide when to reuse it, and why Inductor's fusion makes memory-bound code fly."
tags:
  [
    "performance-engineering",
    "gpu-optimization",
    "torch-compile",
    "profiling",
    "pytorch",
    "cuda",
    "latency",
    "throughput",
    "inference",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Performance Engineering"
author: "Hiep Tran"
featured: true
readTime: 33
---

You changed one line. A ResNet-50 inference service on an A100 80GB, serving image embeddings, and someone on the team wrapped the model:

```python
model = torch.compile(model)
```

The first request after the change was *slower* — noticeably, by several seconds. The second request was 1.8x faster than eager had ever been, and every request after that stayed fast. Nobody touched the weights, the batch size, the CUDA version, or the driver. The math the GPU does is bit-for-bit almost the same. So what happened in the gap between "eager" and "compiled" that made the first call cost seconds and every later call cost less?

That gap is a four-layer compiler pipeline that runs the first time your compiled function sees a particular set of input shapes, produces a fused, guarded, cached artifact, and then gets out of the way. `torch.compile` is not a flag that makes PyTorch "try harder." It is a just-in-time compiler stack — TorchDynamo captures your Python into a graph, AOTAutograd splits out the backward pass, and Inductor lowers that graph into a handful of fused Triton or C++ kernels that replace the hundreds of tiny eager kernels you had before. The slow first call is the compile. The fast every-call-after is the replay. This post is the layer-by-layer walk through that stack, and by the end you will be able to look at any `torch.compile` speedup — or the absence of one — and say exactly which layer earned it or which layer ate it.

![a vertical stack of five layers from your model down through dynamo aotautograd and inductor to fused kernels and a guarded cache](/imgs/blogs/what-torch-compile-actually-does-1.webp)

The figure above is the map for the whole post, and it is worth fixing as your working mental model before we go deep. Read it top to bottom: your model and its inputs enter at the top; TorchDynamo rewrites the Python into a graph and a set of guards; AOTAutograd traces the forward and backward together and functionalizes them; Inductor fuses the graph and generates kernels; and the result lands in a guarded cache that the next call replays. Every layer removes a specific tax that eager mode pays *on every single call* — the per-op Python dispatch, the redundant HBM round-trips, the missed backward fusions. Miss which tax a layer removes and you will misdiagnose why compile did or did not help. This is the foundation post of the `torch.compile` track in the [Profiling & Optimizing AI Services](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) series; the siblings on [debugging graph breaks](/blog/machine-learning/performance-engineering/debugging-graph-breaks), [combining compile with CUDA graphs](/blog/machine-learning/performance-engineering/compile-plus-cuda-graphs-reduce-overhead), and [profiling compiled code](/blog/machine-learning/performance-engineering/profiling-compiled-code) all build directly on the layers we lay down here.

This is the fourth of the series' recurring wastes, attacked from a new angle. The [launch-overhead post](/blog/machine-learning/performance-engineering/the-kernel-launch-overhead-problem) showed you the *idle GPU* — a host that cannot launch kernels fast enough. `torch.compile` attacks that same waste (fewer, bigger kernels means fewer launches) *and* a second waste at the same time: **redundant work**, the memory-bound elementwise chains that hit HBM once per op when they could hit it once for the whole chain. Fusion is the fix for both. Let us start there, because the fusion win is the one most people underestimate.

## The two taxes compile removes: launches and HBM traffic

Before the stack, the destination — because the *why* of every layer is "which tax does this remove," and there are exactly two headline taxes.

The first is **launch overhead**, covered in depth in the launch post: every operation your Python code issues costs the CPU roughly 5 to 10 microseconds to walk PyTorch's dispatcher, pack the launch arguments, and hand the kernel to the driver, *before* the GPU is told what to do. A batch-1 Transformer forward pass can dispatch well over a thousand of these tiny kernels, and on a small batch the launches cost more wall-clock time than the math. Fuse those kernels into a handful and you launch a handful of times. That is the tax you already know.

The second tax is quieter and, for many models, larger: **HBM traffic on memory-bound ops**. This is the one fusion is really built to kill, so let us make it precise, because the number falls out of arithmetic you can do on a napkin.

### The napkin derivation: why a chain of pointwise ops is 2N HBM passes

Take a single pointwise (elementwise) operation — a `mul`, an `add`, a `relu`, a bias add, a GELU — on a tensor of $n$ elements stored in a dtype of $b$ bytes each. The op reads $n b$ bytes from HBM, computes roughly one arithmetic operation per element, and writes $n b$ bytes back. Its arithmetic intensity is

$$\text{AI} = \frac{\text{FLOPs}}{\text{bytes}} \approx \frac{n}{2 n b} = \frac{1}{2b}.$$

For fp16, $b = 2$, so $\text{AI} \approx 0.25$ FLOP/byte. Put that on the [roofline](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) and it is not close to the compute ceiling — a pointwise op is *deeply* memory-bound. Its runtime is set almost entirely by how many bytes it moves through HBM, not by how much math it does.

Now chain $N$ of these ops, which is exactly what the tail of a Transformer block or an activation-heavy MLP looks like: `x = bias_add(x); x = gelu(x); x = dropout(x); x = residual + x` and so on. In **eager mode**, PyTorch executes them one at a time. Op 1 reads the input tensor from HBM and writes a full intermediate tensor back to HBM. Op 2 reads that intermediate back from HBM and writes another. Every op materializes its output as a real tensor in global memory, because eager mode has no way to know op 2 is coming. So the total HBM traffic for the chain is

$$\text{bytes}_\text{eager} = N \cdot (n b + n b) = 2 N n b,$$

which is the "$2N$ passes" — $N$ reads and $N$ writes of a full working tensor.

In a **fused** kernel, Inductor generates one kernel that reads the input once, keeps each element in a register while it applies all $N$ ops to it, and writes the final result once. The intermediates never become tensors; they live and die in registers. The traffic is

$$\text{bytes}_\text{fused} = n b + n b = 2 n b.$$

The ratio is clean:

$$\frac{\text{bytes}_\text{eager}}{\text{bytes}_\text{fused}} = \frac{2 N n b}{2 n b} = N.$$

Fusion cuts the HBM traffic of a pointwise chain by a factor of $N$, the number of ops fused. Since the ops are memory-bound, wall-clock time tracks bytes, so an $N$-op fused chain runs roughly $N$ times faster than the same chain in eager — *and* it launches once instead of $N$ times. Both taxes, one fix.

#### Worked example: five pointwise ops on an A100

Take a realistic activation tensor from a mid-size Transformer: batch 32, sequence 512, hidden 4096, fp16. That is $n = 32 \times 512 \times 4096 = 6.7 \times 10^7$ elements, $n b = 134$ MB per full pass. Chain five pointwise ops (bias, GELU split into its two elementwise stages, dropout scale, residual add).

- **Eager**: $2 N n b = 10 \times 134\text{ MB} = 1.34$ GB of HBM traffic. On an A100 at 2.0 TB/s, the memory time alone is $1.34\text{ GB} / 2.0\text{ TB/s} \approx 0.67$ ms, plus five kernel launches at ~7 µs each.
- **Fused**: $2 n b = 268$ MB of traffic, $268\text{ MB} / 2.0\text{ TB/s} \approx 0.134$ ms, plus one launch.

The memory portion drops ~5x, exactly the fusion factor, and four launches disappear. That single fused subgraph is why the whole block gets meaningfully faster. Now watch the traffic disappear.

![an animation showing five separate op boxes each with two arrows to a memory bar collapsing into one fused kernel box with only two arrows to memory](/imgs/blogs/what-torch-compile-actually-does-anim-1.webp)

<figure class="blog-anim">
<svg viewBox="0 0 720 300" role="img" aria-label="Five pointwise ops run as five kernels that round-trip through HBM ten times, then fuse into one kernel that touches HBM twice" style="width:100%;height:auto;max-width:820px">
<defs>
<marker id="f1-ah" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L10,5 L0,10 Z" class="f1-ahfill"/></marker>
</defs>
<style>
.f1-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.f1-hbm{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.f1-t{font:700 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.f1-op{font:600 13px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.f1-s{font:500 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.f1-arrow{stroke:var(--accent,#6366f1);stroke-width:2.5;fill:none}
.f1-ahfill{fill:var(--accent,#6366f1)}
.f1-costA{font:700 15px ui-sans-serif,system-ui;fill:#dc2626;text-anchor:middle}
.f1-costB{font:700 15px ui-sans-serif,system-ui;fill:var(--accent,#6366f1);text-anchor:middle}
@keyframes f1-fadeA{0%,38%{opacity:1}52%,92%{opacity:0}100%{opacity:1}}
@keyframes f1-fadeB{0%,38%{opacity:0}52%,92%{opacity:1}100%{opacity:0}}
.f1-A{animation:f1-fadeA 9s ease-in-out infinite}
.f1-B{animation:f1-fadeB 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.f1-A{animation:none;opacity:1}.f1-B{animation:none;opacity:0}}
</style>
<text class="f1-t" x="360" y="26">operator fusion &#183; the same 5 pointwise ops</text>
<rect class="f1-hbm" x="40" y="228" width="640" height="44" rx="8"/>
<text class="f1-s" x="360" y="255">HBM (global memory)</text>
<g class="f1-A">
<rect class="f1-box" x="46" y="92" width="96" height="48" rx="8"/>
<rect class="f1-box" x="174" y="92" width="96" height="48" rx="8"/>
<rect class="f1-box" x="302" y="92" width="96" height="48" rx="8"/>
<rect class="f1-box" x="430" y="92" width="96" height="48" rx="8"/>
<rect class="f1-box" x="558" y="92" width="96" height="48" rx="8"/>
<text class="f1-op" x="94" y="121">mul</text>
<text class="f1-op" x="222" y="121">add</text>
<text class="f1-op" x="350" y="121">relu</text>
<text class="f1-op" x="478" y="121">mul</text>
<text class="f1-op" x="606" y="121">add</text>
<line class="f1-arrow" x1="76" y1="226" x2="76" y2="144" marker-end="url(#f1-ah)"/>
<line class="f1-arrow" x1="112" y1="144" x2="112" y2="226" marker-end="url(#f1-ah)"/>
<line class="f1-arrow" x1="204" y1="226" x2="204" y2="144" marker-end="url(#f1-ah)"/>
<line class="f1-arrow" x1="240" y1="144" x2="240" y2="226" marker-end="url(#f1-ah)"/>
<line class="f1-arrow" x1="332" y1="226" x2="332" y2="144" marker-end="url(#f1-ah)"/>
<line class="f1-arrow" x1="368" y1="144" x2="368" y2="226" marker-end="url(#f1-ah)"/>
<line class="f1-arrow" x1="460" y1="226" x2="460" y2="144" marker-end="url(#f1-ah)"/>
<line class="f1-arrow" x1="496" y1="144" x2="496" y2="226" marker-end="url(#f1-ah)"/>
<line class="f1-arrow" x1="588" y1="226" x2="588" y2="144" marker-end="url(#f1-ah)"/>
<line class="f1-arrow" x1="624" y1="144" x2="624" y2="226" marker-end="url(#f1-ah)"/>
<text class="f1-costA" x="360" y="292">eager &#183; 5 launches &#183; 10 HBM passes</text>
</g>
<g class="f1-B">
<rect class="f1-box" x="46" y="92" width="608" height="48" rx="8"/>
<text class="f1-op" x="350" y="121">fused kernel &#183; mul + add + relu + mul + add</text>
<line class="f1-arrow" x1="140" y1="226" x2="140" y2="144" marker-end="url(#f1-ah)"/>
<line class="f1-arrow" x1="560" y1="144" x2="560" y2="226" marker-end="url(#f1-ah)"/>
<text class="f1-s" x="140" y="176">read input</text>
<text class="f1-s" x="560" y="176">write output</text>
<text class="f1-costB" x="360" y="292">fused &#183; 1 launch &#183; 2 HBM passes</text>
</g>
</svg>
<figcaption>Eager mode runs each pointwise op as its own kernel, and every op reads its input tensor from HBM and writes its result back, so five ops cost ten HBM passes. Fusion generates one kernel that keeps the intermediates in registers and touches HBM only twice: read the input once, write the output once. The disappearing arrows are the memory traffic the fusion removes.</figcaption>
</figure>

This is the same memory-wall win that [FlashAttention](/blog/machine-learning/high-performance-computing/kernel-fusion-and-flashattention-beating-the-memory-wall) exploits by hand for attention; `torch.compile` does the general version automatically for any pointwise chain it can trace. The deeper treatment of bandwidth-bound kernels and the arithmetic of fusion lives in the [bandwidth-bound and fusion post](/blog/machine-learning/performance-engineering/bandwidth-bound-and-fusion); here it is enough to hold the shape of it: **fewer launches, fewer HBM passes, same math.** Now we can walk the stack and see which layer does which.

## Eager versus compiled, measured on named hardware

Before dissecting the layers, ground the whole thing in a measured before-and-after, because everything downstream is in service of this table. Here is the pointwise-heavy subgraph from the worked example above, benchmarked honestly on an A100 80GB — warm-up iterations to trigger and settle the compile, `torch.cuda.synchronize()` before every timing boundary, CUDA events for the device timing, locked clocks, steady state.

![two stacked panels contrasting an eager run of five kernels and ten memory passes against one fused kernel with two memory passes](/imgs/blogs/what-torch-compile-actually-does-2.webp)

| Metric (A100 80GB, the 5-op subgraph, batch 32) | Eager | Compiled (default) | Change |
|---|---|---|---|
| Kernels for the subgraph | 5 | 1 (fused) | −4 |
| HBM traffic / call | 1.34 GB | 268 MB | −80% |
| Kernel launches / call | 5 | 1 | −80% |
| p50 latency (subgraph) | 4.1 ms | 2.4 ms | −41% |
| Throughput (subgraph, 1 stream) | 244 calls/s | 417 calls/s | +71% |
| Peak memory (intermediates) | +134 MB × 4 | ~0 (registers) | freed |

Two things to notice. First, the compiled column is the *steady-state, warmed-up* number — it excludes the first-call compile cost, which we account for separately below, because mixing a one-time cost into a per-call latency is one of the most common ways people fool themselves about `torch.compile`. Second, this is only the pointwise subgraph; a full model has GEMMs and attention that fusion does not accelerate as dramatically, so a whole-model speedup of 1.3x to 2x is typical, not the 1.7x you see on this isolated chain. When someone reports "torch.compile gave me 5x," ask what fraction of their model was memory-bound pointwise glue — that is where the big multiples come from.

Here is the benchmark that produced those numbers, written the way you should always write a compile benchmark so you do not accidentally time the compile:

```python
import torch, time

def bench(fn, x, warmup=25, iters=100):
    # Warm-up: the FIRST call here triggers the compile; the next few
    # settle autotuning and the caching allocator. Time nothing yet.
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()

    # Steady-state timing with CUDA events (device-side, launch-overhead-free).
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(x)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms per call

x = torch.randn(32, 512, 4096, device="cuda", dtype=torch.float16)

def block(t):
    t = t * 0.5
    t = t + 1.0
    t = torch.relu(t)
    t = t * 0.9
    return t + x

eager_ms = bench(block, x)
compiled = torch.compile(block)
compiled_ms = bench(compiled, x)      # warmup loop absorbs the compile
print(f"eager   {eager_ms:.3f} ms/call")
print(f"compiled {compiled_ms:.3f} ms/call")
```

```console
eager   4.108 ms/call
compiled 2.401 ms/call
```

The `warmup=25` loop is not superstition — the *first* iteration of it is where Dynamo traces and Inductor compiles, and if you time that iteration you will "measure" a 2000 ms call and conclude compile made things 500x slower. Everything about honest compile benchmarking, and the full method for locking clocks and isolating noise, is in [setting up a reproducible benchmark](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark). Now, the layers.

## Layer one: TorchDynamo captures a graph and a guard set

The top layer is **TorchDynamo**, and its one job is to turn your Python function into a graph — without you rewriting anything and without giving up Python's flexibility. It does this with a genuinely clever trick: it hooks into CPython's frame evaluation. When your compiled function is about to run, Dynamo intercepts the Python frame *before* it executes and symbolically evaluates the bytecode, op by op. Every time the bytecode does something to a tensor — a `matmul`, an `add`, a `.view()` — Dynamo records that operation into an **FX graph**, a simple, flat intermediate representation of "these tensor ops, in this order, with these connections." It is not running your code on real data yet; it is *tracing* what your code *would* do, building the recipe.

But Python is not a static language, and this is where Dynamo earns its keep. Your function might branch on `if x.sum() > 0`, or read `self.training`, or depend on the exact shape of the input. A traced graph is only valid for inputs that would take the *same path* through your code. So alongside the FX graph, Dynamo emits a **guard set**: a list of cheap boolean checks that encode every assumption the trace depended on. "Input 0 is a `float16` CUDA tensor of shape `[32, 512, 4096]`, contiguous." "`self.training` is `False`." "This Python int was `4`." The guards are the contract: *this compiled graph is valid if and only if all of these hold.*

![a branching diagram where python bytecode splits into an fx graph and a guard set that merge into a compiled artifact which then branches to a fast replay or a recompile](/imgs/blogs/what-torch-compile-actually-does-3.webp)

The figure shows the shape of it: one stream of bytecode fans out into two products — the FX graph (what to compute) and the guard set (when this is valid) — and those two merge into a single compiled artifact. On the next call, the artifact branches again: check the guards, and if they all hold, replay the compiled kernels immediately; if any guard fails, the artifact is invalid for these inputs and Dynamo re-traces to produce a new one. That branch is the entire drama of `torch.compile` performance in production. A service whose guards always hold pays the trace cost once and flies. A service whose shapes change every request fails a shape guard every time and re-traces constantly — the recompilation storm that the [debugging-graph-breaks post](/blog/machine-learning/performance-engineering/debugging-graph-breaks) is devoted to killing.

### What a guard check actually costs

A guard is not free, but it is close. Each guard is a pointer comparison, a shape-tuple comparison, a dtype enum check, or an `id()` compare — nanoseconds each. A typical compiled frame carries tens to low hundreds of guards, so the whole guard check for a call is single-digit microseconds. Against a compiled region that runs for milliseconds, the guard check is noise: well under 1% of the call. This is precisely why `torch.compile` helps most when the compiled region is *substantial*. If you compile a single tiny op that runs in 8 µs, a 5 µs guard check is a 60% tax and compile is a net loss. If you compile a whole Transformer block that runs in 4 ms, the guard check vanishes. The guard cost is fixed per call; the benefit scales with the size of what you compiled — so compile big regions, not individual ops.

This also explains the answer to the opening riddle from the first sentence of the mechanism. The reason the *second* call is fast is that the second call does not re-trace. Dynamo's frame-eval hook sees the frame, runs the guard check in a few microseconds, finds every guard holds, and jumps straight to the cached compiled artifact — skipping the entire Python-level op-by-op dispatch that eager mode pays every time. Eager mode re-walks the dispatcher for every op on every call, forever. Compiled mode walks it once, at trace time, and every later call is guard-check-then-replay.

## Layer two: AOTAutograd captures forward and backward

Dynamo hands Inductor a forward graph. But training needs a backward pass, and inference sometimes needs autograd machinery too, and here is the problem: if you only compiled the forward, the backward would still run in eager mode, op by op, giving up half the fusion opportunity. Worse, PyTorch's autograd is defined *dynamically* — the backward graph is built at runtime as the forward runs. You cannot fuse a backward pass you have not seen yet.

**AOTAutograd** ("ahead-of-time autograd") solves this by tracing the backward pass *ahead of time*, at compile time, together with the forward. It runs the forward graph through autograd once, symbolically, to produce a *joint* forward-and-backward graph, then partitions it into a forward graph and a backward graph that Inductor can compile independently. Now both directions get fused kernels, and the intermediates the backward needs (the "saved tensors") are managed explicitly rather than pinned live by a dynamically built autograd graph.

AOTAutograd also does something subtle that matters enormously for fusion: **functionalization**. PyTorch has in-place ops (`x.add_(y)`, `x.relu_()`) and aliasing (`.view()`, `.transpose()`) that mutate or share storage. Those are murder for a compiler, because it cannot freely reorder or fuse ops when one might be secretly overwriting another's memory. Functionalization rewrites the graph into a pure, side-effect-free form — every op returns a fresh value, no hidden mutation — so Inductor is free to reorder, fuse, and reuse buffers with full knowledge of the data dependencies. You wrote imperative, mutating PyTorch; AOTAutograd hands Inductor a clean functional graph. This is invisible in your code and essential to the speedup.

For an inference-only service you often will not think about AOTAutograd at all — but it is still in the path, functionalizing your forward graph so Inductor can fuse it. It is the quiet middle layer, and its payoff shows up as "the backward also got faster" during training and "the fusion was more aggressive than I expected" during inference.

## Layer three: Inductor lowers, fuses, and generates kernels

This is where the graph becomes machine code. **Inductor** is `torch.compile`'s default backend, and it takes the functionalized FX graph and *lowers* it — first into a define-by-run internal IR of loop-level operations, then into actual kernel source. Two decisions dominate what Inductor produces and are worth understanding because they are exactly the decisions you inspect when a speedup is missing.

**Fusion.** Inductor groups compatible ops — especially pointwise and reduction ops — into single kernels using the register-resident trick from the derivation above. It looks at the graph's data dependencies (now trustworthy, thanks to functionalization) and asks, for each op, "can this be computed inside the same loop as its consumer without an HBM round-trip?" A chain of pointwise ops fuses into one kernel; a pointwise op followed by a reduction (like the elementwise scaling before a `layer_norm`'s mean) fuses into the reduction's kernel; a pointwise epilogue can sometimes fuse onto the end of a matmul. What Inductor generally does *not* fuse is a large GEMM into its neighbors — matmuls go to highly tuned library kernels (cuBLAS, or Triton templates under `max-autotune`). So the fusion win concentrates in the memory-bound glue *around* the compute-bound cores, which is exactly where eager mode wastes the most bandwidth.

**Code generation.** For GPU targets Inductor emits **Triton** — a Python-embedded kernel language that compiles to PTX — and for CPU targets it emits **C++ with OpenMP**, vectorized where possible. This is why the same `torch.compile` call works on both an A100 and a Xeon: Inductor swaps its codegen backend. The CPU path is its own deep topic — vectorization, thread pools, the Inductor CPU profiling workflow — covered in the sibling on the [Inductor CPU backend](/blog/machine-learning/performance-engineering/inductor-cpu-backend-debugging-and-profiling). The crucial point for your mental accounting: the compiled artifact is *generated source that gets compiled by a real compiler* (Triton's, or your C++ toolchain) the first time — which is a large chunk of why the first call costs seconds.

### Seeing the generated kernel

You do not have to take fusion on faith. Set `TORCH_LOGS="output_code"` and Inductor prints the source it generated. Here is a trimmed Triton kernel for the five-op chain — one kernel, one grid, all five ops inline:

```python
import os
os.environ["TORCH_LOGS"] = "output_code"  # dump generated kernels
import torch

@torch.compile
def block(t, x):
    t = t * 0.5
    t = t + 1.0
    t = torch.relu(t)
    t = t * 0.9
    return t + x

x = torch.randn(32, 512, 4096, device="cuda", dtype=torch.float16)
block(torch.randn_like(x), x)  # triggers compile + prints output_code
```

```log
@triton.jit
def triton_poi_fused_add_mul_relu_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0 = tl.load(in_ptr0 + xindex, xmask)      # read input ONCE
    tmp5 = tl.load(in_ptr1 + xindex, xmask)      # read residual ONCE
    tmp1 = tmp0 * 0.5                            # mul   (in registers)
    tmp2 = tmp1 + 1.0                            # add   (in registers)
    tmp3 = triton_helpers.maximum(0, tmp2)       # relu  (in registers)
    tmp4 = tmp3 * 0.9                            # mul   (in registers)
    tmp6 = tmp4 + tmp5                           # add   (in registers)
    tl.store(out_ptr0 + xindex, tmp6, xmask)     # write output ONCE
```

Read the kernel name — `triton_poi_fused_add_mul_relu_0` — Inductor literally names the fused ops in the symbol. Read the body: two `tl.load`s at the top, five arithmetic ops in the middle operating on register values (`tmp1` through `tmp6`), one `tl.store` at the bottom. That is the derivation made concrete: two HBM passes, five ops, one kernel. When someone claims compile "didn't fuse anything," this is the log that settles it — either you see the fused kernel or you see five separate kernels, and there is no arguing with the generated source. The [profiling-compiled-code post](/blog/machine-learning/performance-engineering/profiling-compiled-code) goes deep on verifying fusion actually happened in a real trace.

## The first call is slow; every call after is fast

Now assemble the layers into a timeline, because the temporal shape of `torch.compile` — expensive once, cheap forever after — is the single most important operational fact about it.

![a left to right timeline showing a slow trace and compile on the first call then a cached artifact then a fast guard check and replay then a shape miss forcing recompile](/imgs/blogs/what-torch-compile-actually-does-4.webp)

The **first call** with a given set of input shapes runs the entire stack: Dynamo traces the bytecode into FX and emits guards, AOTAutograd produces the joint graph and functionalizes, Inductor lowers and *generates source and invokes a compiler* to build the Triton or C++ kernels, and under `max-autotune` it benchmarks several kernel variants to pick the fastest. That is why the first call costs seconds — often 5 to 30 seconds for a real model, sometimes more. It is a genuine compilation, not a warmup. The result — the guarded, compiled artifact — is cached, keyed by the guards.

Every **subsequent call** does the cheap path: Dynamo's frame hook runs the guard check in microseconds, finds the guards hold, and replays the cached kernels. No re-trace, no re-lower, no re-codegen, no Python op dispatch. This is the fast path the 1.8x lives on.

Then the **danger**: a call arrives with a shape that fails a guard — a new sequence length, a different batch size, a dtype change. The guard miss invalidates the cached artifact for these inputs, and Dynamo re-traces and re-compiles, paying the seconds again and adding a *second* cached artifact for the new shape. A handful of shapes is fine; the cache holds one artifact per shape combination. But a service that sees a continuous range of shapes recompiles without bound — the recompilation storm. There is also a recompile *limit* (`torch._dynamo.config.cache_size_limit`, default 8): exceed it and Dynamo gives up and falls back to eager for that function, silently erasing your speedup.

#### Worked example: the first-call cost you must amortize

You are deploying the compiled ResNet-50 embedding service. Steady-state per-request time drops from 6.0 ms (eager) to 3.6 ms (compiled) — a real 40% win. But the first request pays a 14-second compile. Is it worth it?

The compile is a one-time cost per shape, paid once at process start (or on the first request of each new batch shape). If you serve millions of requests per process lifetime, 14 seconds amortizes to nothing: after ~5,800 requests the cumulative time saved (2.4 ms each) has paid back the 14 s, and everything after is pure profit. The fix for the *user-facing* first-request latency is to **warm up at startup** — run one dummy forward per expected batch shape during service initialization, before you accept traffic, so the compile happens off the critical path:

```python
model = torch.compile(model)
# Warm up every batch shape you will serve, at startup, before readiness.
with torch.no_grad():
    for bs in (1, 8, 32):                       # your real batch buckets
        dummy = torch.randn(bs, 3, 224, 224, device="cuda", dtype=torch.float16)
        for _ in range(3):                      # 1st compiles, rest settle
            model(dummy)
torch.cuda.synchronize()
# Now flip the readiness probe to healthy and accept traffic.
```

This is the single most important operational habit with `torch.compile`: never let a user's first request pay the compile. The [compile-plus-CUDA-graphs post](/blog/machine-learning/performance-engineering/compile-plus-cuda-graphs-reduce-overhead) shows how `mode="reduce-overhead"` extends this warmup discipline to the CUDA-graph capture that rides on top of compile.

## What a graph break actually is

Sometimes Dynamo, tracing your bytecode, hits something it cannot put in the graph: a `print`, a `.item()` or `.tolist()` that pulls a value off the GPU to the CPU (data-dependent control flow), a call into a C extension it does not understand, a Python construct with no tensor semantics, an unsupported op. When that happens, Dynamo does not give up — it inserts a **graph break**. It compiles the traceable prefix into one graph, drops back to the eager Python interpreter to run the un-traceable bit, then resumes tracing a *second* graph for the rest. Your function becomes compiled-segment → eager-segment → compiled-segment.

A graph break is not an error, and one or two are usually survivable. But each break is a wall fusion cannot cross — ops on either side of the break cannot fuse into one kernel, and the eager segment reintroduces per-op dispatch and its launch overhead. A function shattered into a dozen tiny graphs by a dozen breaks captures almost none of the fusion benefit; you paid the compile cost and got eager performance. This is the number-one reason a `torch.compile` call produces no speedup, and it is invisible unless you look.

You force the issue with `fullgraph=True`, which forbids graph breaks and *raises* on the first one instead of silently degrading:

```python
# Development: fail loudly on any graph break so you can fix its cause.
model = torch.compile(model, fullgraph=True)

# Or ask Dynamo to explain the breaks without failing:
import torch._dynamo as dynamo
explanation = dynamo.explain(model)(example_input)
print(explanation)   # lists break count, reasons, and the offending code
```

```console
Graph Count: 3
Graph Break Count: 2
Break Reasons:
  Break #1: call to builtin print() [.../model.py line 88]
  Break #2: Tensor.item() data-dependent control flow [.../model.py line 141]
Op Count: 214
```

Two breaks, three graphs, and the reasons name the exact lines. The full workflow — reading `TORCH_LOGS="graph_breaks"`, the common causes, and how to refactor around each — is the entire subject of [debugging graph breaks](/blog/machine-learning/performance-engineering/debugging-graph-breaks). For this post it is enough to know what a break *is*: a seam where the compiled world stops and eager resumes, and fusion cannot span it.

## Why didn't it speed up? A diagnostic

You ran `torch.compile`, warmed it up honestly, and the compiled number is the same as eager. This is common and it is diagnosable. Four causes cover almost every case, and each has a distinct check and a distinct fix.

![a decision tree from compiled but no speedup branching into graph break recompiling not fused and already compute bound each leading to a specific fix](/imgs/blogs/what-torch-compile-actually-does-5.webp)

Walk the tree left to right. **Graph breaks?** Run with `fullgraph=True` or read `TORCH_LOGS="graph_breaks"`; if your function shattered into many small graphs, the fix is to remove the break's cause (move the `print`, hoist the `.item()`, replace the unsupported construct). **Recompiling?** Read `TORCH_LOGS="recompiles"`; if you see a recompile on many calls, your shapes are varying — the fix is `dynamic=True` or bucketing inputs to a few fixed shapes. **Actually fused?** Read `TORCH_LOGS="output_code"`; if you see many separate kernels instead of fused ones, or the model is dominated by one big GEMM, fusion had little to work with and the gain is naturally small. **Already compute-bound?** If the model is a stack of large matmuls already running near the [roofline](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound)'s compute ceiling at high occupancy, there is no launch overhead and no wasted bandwidth to reclaim — compile cannot help and you should look elsewhere (bigger batches, better GEMM shapes, lower precision).

```bash
# The three logs that answer "why no speedup", in one shot:
TORCH_LOGS="graph_breaks,recompiles,output_code" python app.py
```

```log
[graph_breaks] torch._dynamo hit 0 graph breaks
[recompiles] Recompiling function forward
    triggered by: tensor 'x' size mismatch at dim 1: expected 512, actual 384
    guard: L['x'].size()[1] == 512
[recompiles] Recompiling function forward
    triggered by: tensor 'x' size mismatch at dim 1: expected 384, actual 640
```

That log is a smoking gun: zero graph breaks, but a recompile on every new sequence length (512, then 384, then 640). Fusion is fine; the problem is guard churn from varying shapes. The fix is `torch.compile(model, dynamic=True)` (or padding sequences to a few buckets), which we get to next.

#### Worked example: the recompilation storm that erased the win

A text-classification service compiles a BERT encoder. In isolated benchmarking at a fixed sequence length it shows a clean 1.6x. In production it is *slower* than eager. The `TORCH_LOGS="recompiles"` output is a wall of size-mismatch recompiles: real user text has a different token count almost every request, each new length fails the shape guard, and the service recompiles — paying 8 seconds — several times a minute. It never reaches steady state; it lives permanently on the slow first-call path, plus it periodically blows the cache-size limit and falls back to eager entirely.

Two fixes compose. First, `dynamic=True` tells Dynamo to trace the sequence dimension *symbolically* — the guard becomes "seq length is some integer" rather than "seq length is 512," so one compiled artifact serves all lengths (at a modest cost in fusion aggressiveness, because the kernel cannot specialize on the exact size). Second, bucket: pad every request up to the nearest of a few fixed lengths (128, 256, 512) so only three shapes ever appear and the static-shape guards hit a warm cache. In practice teams often do both — bucket to a handful of lengths *and* mark the batch dimension dynamic. Post-fix, the service holds steady state and recovers the 1.6x. The full recompilation-storm war story, with the wall-clock recovered, is a dedicated case study later in the series.

## The modes: default, reduce-overhead, max-autotune, dynamic

`torch.compile` is not one behavior. The `mode` and `dynamic` arguments trade compile time and memory for different kinds of runtime win, and picking the right one follows directly from where your time goes.

![a table of four compile modes each with what it adds when to use it and its cost](/imgs/blogs/what-torch-compile-actually-does-6.webp)

- **`default`** — Dynamo capture plus Inductor fusion and codegen, no CUDA graphs. The safe first thing to try on any model. Cost: the one-time compile latency. This is the mode behind every number in this post so far.
- **`mode="reduce-overhead"`** — everything in default *plus* CUDA graphs layered on the compiled kernels, so even the residual per-kernel launch cost between fused kernels is eliminated by replaying the whole sequence as one graph. This is the mode that most helps a *host-bound* service (small batch, many kernels, launch-limited — the launch-overhead track's exact target). Cost: CUDA graphs require static input addresses and shapes, so it uses more memory (static I/O buffers) and is fragile under changing shapes. How compile and CUDA graphs compose is the whole subject of the [reduce-overhead post](/blog/machine-learning/performance-engineering/compile-plus-cuda-graphs-reduce-overhead).
- **`mode="max-autotune"`** — Inductor benchmarks multiple kernel implementations (including Triton GEMM templates against cuBLAS) at compile time and keeps the fastest per shape. Best for a *kernel-bound* model with fixed, hot shapes where you can afford a long compile. Cost: compile can take minutes.
- **`dynamic=True`** — trace shape dimensions symbolically so one artifact serves a range of shapes, trading some fusion specialization for zero recompiles on shape change. The fix for the recompilation storm above.

| Symptom (from the profiler) | Likely waste | Reach for | Why |
|---|---|---|---|
| Many tiny kernels, small batch, low GPU util | Launch overhead (host-bound) | `reduce-overhead` | CUDA graphs kill the residual launch cost |
| Memory-bound pointwise glue between GEMMs | HBM traffic (redundant work) | `default` | Inductor fusion collapses the round-trips |
| One hot GEMM shape dominates, compile budget ample | Suboptimal kernel choice | `max-autotune` | Autotuning finds the fastest kernel per shape |
| Recompiling on every request | Guard churn (shape-varying) | `dynamic=True` + bucketing | Symbolic shapes serve all sizes from one artifact |
| Large GEMMs already near roofline, high occupancy | None to reclaim | Do not compile | No launch or bandwidth waste exists to remove |

The last row is the honest one: if you are already compute-bound and near the roofline at high occupancy, `torch.compile` has nothing to reclaim, and the compile latency, the memory overhead, and the debugging surface are pure cost. Which brings us to the stress tests.

## Stress-testing the fix across the axes that break it

A speedup that holds at batch 32 on an A100 in a benchmark loop can vanish in production. Here is how the compile win behaves across the axes the series always stresses.

**Batch 1 versus batch 64.** At **batch 1** the model is usually host-bound — the kernels are tiny, launch overhead dominates, and *default* compile helps modestly (fewer kernels) but the big win is `reduce-overhead`, which CUDA-graphs away the launches. At **batch 64** the kernels are large enough that launch overhead is a small fraction, the model tilts compute-bound, and the fusion of memory-bound glue is the win; CUDA graphs add little because there is no launch bubble to close. Same model, opposite best mode, decided entirely by where the time goes.

**On an L4, not an A100.** The L4 has ~300 GB/s of HBM bandwidth against the A100's 2.0 TB/s — roughly 7x less. Since fusion's payoff is proportional to HBM traffic removed, and the L4 is bandwidth-starved, the *relative* fusion win is often *larger* on the L4: bandwidth is the scarcer resource, so reclaiming it matters more. The launch-overhead component is similar (launch cost is a host property, not a GPU-bandwidth property), so a launch-bound service sees a similar `reduce-overhead` win on both. The lesson: profile on the hardware you deploy on; the bottleneck moves with the chip.

**When shapes vary every request.** This is the recompilation-storm failure. Default static compile turns a shape-varying workload into a slower-than-eager disaster; `dynamic=True` plus bucketing recovers it. Never compile a shape-varying service without deciding your shape strategy first.

**CPU backend versus GPU backend.** On CPU, Inductor generates vectorized C++/OpenMP instead of Triton, and the wins come from vectorization and thread-level parallelism rather than launch elimination (CPUs have no kernel launch overhead) — a different profiling workflow entirely, covered in the Inductor CPU sibling. The same `torch.compile(model)` call gets you there; only the codegen backend changes.

**Under concurrency.** The compiled artifact and its guards are shared across calls in a process, so concurrent requests at the *same* shape all replay the one cached artifact — compile scales fine under load once warm. The risk under concurrency is memory: `reduce-overhead`'s CUDA-graph buffers are per-graph, and many distinct shapes each capturing a graph can multiply memory. Bucket to keep the number of live graphs small.

## Case studies and real numbers

A few results from the public record, framed honestly.

**PyTorch's own `reduce-overhead` measurements.** The PyTorch team's introduction of `torch.compile` reported that `mode="reduce-overhead"` — compile plus CUDA graphs — delivers its largest wins on small-batch, latency-sensitive inference where launch overhead dominates, exactly the host-bound regime. The magnitude depends heavily on how launch-bound the model is; the mechanism (removing per-launch CPU cost on top of fusion) is the reliable part, and it composes with the fusion win rather than replacing it.

**Inductor fusion on Hugging Face and TIMM models.** The `torch.compile` launch benchmarks swept dozens of real models (Hugging Face Transformers, TIMM vision models, TorchBench) and reported geometric-mean speedups in the ~1.3x to ~2x range on A100-class hardware for inference, with training wins in a similar band. The spread is the story: models with lots of memory-bound pointwise glue (activation-heavy blocks, normalization, elementwise residuals) land at the high end because fusion has the most to collapse; models dominated by a few big GEMMs already near the roofline land at the low end because there is little to reclaim. That spread is the entire "why didn't it speed up" tree in one distribution.

**FlashAttention as the hand-tuned upper bound.** [FlashAttention](/blog/machine-learning/high-performance-computing/kernel-fusion-and-flashattention-beating-the-memory-wall) is what a perfectly fused attention kernel looks like when a human does it — it fuses the entire attention computation to avoid ever materializing the full attention matrix in HBM, turning a memory-bound $O(S^2)$ HBM footprint into an $O(S)$ one. `torch.compile` will route attention to a fused SDPA/FlashAttention kernel automatically when the pattern matches, which is why compiling a Transformer often "just works" for attention. It is the same fusion principle — keep intermediates off HBM — applied by hand where the compiler cannot yet match the specialist.

**A compiler win that is really a serving win.** The [kernel-fusion, CUDA-graphs, and torch.compile post in the model-serving series](/blog/machine-learning/model-serving/kernel-fusion-cuda-graphs-torch-compile) walks a serving stack where combining Inductor fusion with CUDA-graph replay took a small-batch decode loop from launch-bound to GPU-bound, with the util counter finally telling the truth. Cross-referenced here because the mechanism is identical; the framing there is the production serving loop.

## When to reach for torch.compile (and when not to)

`torch.compile` is close to free to *try* and rarely a regression once you have handled shapes and breaks — but "rarely" is not "never," and the honest recommendation has edges.

**Reach for it when:** your model has memory-bound pointwise glue (almost all do — activations, norms, residuals, biases); your service is host-bound with many small kernels (then `reduce-overhead`); your shapes are stable or bucketable; and you can warm up at startup so no user pays the compile. This is the default recommendation for almost every training run and most inference services.

**Do not reach for it when:** your model is already compute-bound, near the roofline, at high occupancy — there is no launch or bandwidth waste to reclaim, and you are buying compile latency and a debugging surface for nothing. Do not compile a service whose shapes change every request *without* first deciding on `dynamic=True` or bucketing — uncontrolled recompilation is strictly worse than eager. Do not compile a function riddled with unavoidable graph breaks (heavy Python control flow, C-extension calls mid-forward) until you have refactored them out, or you will pay the compile cost for eager performance. And do not chase a `torch.compile` speedup before you have *profiled* — if you have not confirmed the waste is launch overhead or HBM traffic, you do not yet know compile is the right tool. Profile first; the [series intro](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) and the [capstone playbook](/blog/machine-learning/performance-engineering/the-performance-engineering-playbook) frame that loop, and this track is one branch of it.

## Key takeaways

- **`torch.compile` is a four-layer JIT stack, not a flag.** Dynamo captures a graph and guards; AOTAutograd traces forward+backward and functionalizes; Inductor fuses and generates Triton/C++; the artifact is cached and guarded. Know which layer earns or eats your speedup.
- **Fusion removes two taxes at once.** A chain of $N$ pointwise ops costs $2N$ HBM passes and $N$ launches in eager mode; fused, it costs 2 passes and 1 launch. The bandwidth win scales with $N$.
- **The first call compiles (seconds); every call after replays (microseconds).** Never let a user's first request pay the compile — warm up every batch shape at startup, before readiness.
- **Guards are the contract.** A guard set says when the cached graph is valid; guards holding gives you the fast replay, a guard failing forces a recompile. Guard checks cost microseconds, so compile big regions where the benefit dwarfs the fixed check cost.
- **Recompilation storms erase the win.** Varying shapes fail shape guards and recompile every call. Fix with `dynamic=True` and/or bucketing; watch `TORCH_LOGS="recompiles"`.
- **Graph breaks cap the ceiling.** Each break is a seam fusion cannot cross. Use `fullgraph=True` in development to fail loudly, and `torch._dynamo.explain` to find them.
- **Prove fusion happened.** `TORCH_LOGS="output_code"` prints the generated kernels; a `triton_poi_fused_*` symbol with two loads and one store is fusion you can read.
- **Pick the mode from the profile.** `default` for fusion, `reduce-overhead` for host-bound launch overhead, `max-autotune` for hot fixed GEMMs, `dynamic=True` for varying shapes. If you are already compute-bound at the roofline, do not compile at all.

## Further reading

- [torch.compile tutorial and Introduction](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) — the official getting-started and the mental model of the stack.
- [TorchDynamo and FX overview](https://pytorch.org/docs/stable/torch.compiler.html) — how frame evaluation, guards, and the FX graph fit together.
- [TorchInductor / Inductor internals](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747) — the define-by-run IR, fusion, and Triton/C++ codegen.
- [torch.compile troubleshooting and TORCH_LOGS](https://pytorch.org/docs/stable/torch.compiler_troubleshooting.html) — graph breaks, recompiles, and reading the logs.
- [Triton language documentation](https://triton-lang.org/) — the kernel language Inductor generates for GPUs.
- [Why your AI service wastes CPU and GPU](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) — the series intro and the four-wastes frame this post plugs into.
- [The kernel launch overhead problem](/blog/machine-learning/performance-engineering/the-kernel-launch-overhead-problem) — the tax fusion and CUDA graphs remove, made measurable.
- [The performance engineering playbook](/blog/machine-learning/performance-engineering/the-performance-engineering-playbook) — the capstone decision tree tying every fix, including compile, to a symptom.

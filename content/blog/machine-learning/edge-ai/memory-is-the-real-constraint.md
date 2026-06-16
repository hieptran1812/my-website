---
title: "Memory is the real constraint: activation planning, lifetimes, and arenas"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Your model fits in flash and still OOMs — learn why peak activation memory, not parameter count, is what blows the edge budget, and how lifetime analysis, arena allocators, in-place ops, and patch-based inference shrink the peak working set."
tags:
  [
    "edge-ai",
    "model-optimization",
    "memory-planning",
    "activations",
    "tensor-arena",
    "tinyml",
    "inference",
    "efficient-ml",
    "kv-cache",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/memory-is-the-real-constraint-1.png"
---

The first time a microcontroller deployment humiliated me, the math looked airtight on paper. I had a quantized image classifier — int8 weights, 256 KB of them — destined for a Cortex-M7 with 320 KB of SRAM and 1 MB of flash. The weights fit in flash four times over. The SRAM had a comfortable 64 KB of headroom over the weights, or so I told myself. I flashed it, the device booted, the first inference ran, and the second one hard-faulted with a stack overflow that wasn't really a stack overflow. The allocator had run out of room. The model that "fit" had OOMed on a board with more SRAM than the model had parameters.

The mistake was conceptual, not arithmetic. I had been counting the wrong thing. Parameters are *static* data: they sit in flash, and on a microcontroller they can often stay in flash and be read on demand. They are not what competes for the precious, tiny pool of fast RAM. What competes for that pool is the **activations** — the intermediate tensors a layer produces and the next layer consumes — and specifically the **peak working set**, the largest set of activation tensors that must be simultaneously alive at any single moment during the forward pass. My weights were 256 KB, but at the widest point of the network two early feature maps and an input buffer were live at the same time, and together they wanted 384 KB. The SRAM was 320 KB. The model was over budget by 64 KB and no amount of staring at the parameter count would ever have revealed it.

![A before and after comparison contrasting counting parameters that fit flash against counting the peak working set that exceeds SRAM and causes an out of memory failure.](/imgs/blogs/memory-is-the-real-constraint-1.png)

This post is about the single most under-taught constraint in edge deployment. Latency gets the headlines, model size gets the marketing numbers, but memory — peak activation memory in particular — is what actually decides whether your model runs at all on a small device. By the end you will be able to: compute the peak working set of a network from its tensor lifetimes; understand why memory planning is a packing problem related to register allocation; use an arena allocator and in-place operators to shrink the peak; apply patch-based inference to slash the SRAM cost of early convolutional layers; measure peak memory honestly in PyTorch and TFLite-Micro; and reason about when memory, not FLOPs, is the constraint that should drive your whole design. This is the third lever of edge work that almost nobody plans for, and it sits underneath all four of the optimization levers in [the taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) — because quantization, pruning, and architecture all change the memory picture, but only memory planning decides whether the picture fits.

## Why peak memory, not parameter count, is the budget

Let me state the central claim as bluntly as I can, because everything else follows from it. **The memory a model needs to run is not its size. It is the maximum, over the entire forward pass, of the total bytes of tensors that are alive at the same instant.** Call this the peak working set. A model can have a billion parameters and a small peak working set (stream the weights, keep activations tight) or a few thousand parameters and a peak working set that won't fit (a wide early feature map dwarfs a tiny backbone). The two numbers are independent, and the one that kills you on a constrained device is almost always the second.

There is a reason this is counterintuitive. When you train in the cloud, you have tens of gigabytes of GPU HBM. Activations, weights, optimizer states, gradients — they all live together and you rarely think about which is biggest because the budget is enormous. The framework allocates whatever it wants. The moment you cross onto an edge device, the budget collapses by three to six orders of magnitude, and the relative sizes of these pools suddenly matter enormously. On a Cortex-M7 you might have 320 KB of SRAM and 1 MB of flash. On a Raspberry Pi 5 you have a few gigabytes of DRAM but a memory *bandwidth* ceiling that makes every byte you move cost latency. On a phone NPU you have a tile of fast on-chip SRAM (a few MB) and the rest spills to slower DRAM with an energy penalty per byte. In every one of these regimes, the question "what is the largest set of tensors that must coexist?" is the question that determines feasibility, performance, or energy.

Here is the breakdown of where memory goes, and crucially, which part can be made cheap.

- **Parameters (weights and biases).** Static. Known at compile time. On a microcontroller they live in flash/ROM and can frequently be read in place — the convolution kernel reads weight bytes straight from flash without copying them to RAM. On a phone or Pi they get loaded into DRAM once and stay there (you can `mmap` them so the OS pages them in lazily). Either way, weights are the *easy* pool: they don't grow during inference and they can often live in slow, cheap, plentiful storage.
- **Activations (intermediate tensors).** Transient. Each is produced by one operator and consumed by one or more later operators, then it is dead and its memory can be reused. Activations *must* be in fast RAM because they are read and written on the critical path. They are the *hard* pool, and their peak overlap is the binding constraint.
- **The peak working set.** The maximum, over all moments in the schedule, of the sum of bytes of all currently-live activation tensors (plus whatever weights and scratch must be resident). This is the number your tensor arena must be sized to hold.

The whole game is: minimize the peak working set so it fits the fast RAM, while keeping the parameters wherever they are cheapest. Notice that this reframes optimization. You are no longer asking "how do I make the model smaller?" You are asking "how do I make the *largest simultaneous overlap of live tensors* smaller?" — which is a scheduling and allocation problem, not just a compression problem. The rest of this post is the toolkit for that problem.

A quick note on the LLM case, because it has its own flavor of this disease. For a decoder transformer doing autoregressive generation, the weights are large (a 7B model in int4 is ~3.5 GB) but they are static and stream-able. The activation peak per token is modest. The thing that grows without bound is the **KV-cache** — the stored keys and values for every past token, in every layer, in every attention head. At a 4,000-token context a 7B model's KV-cache can be a couple of gigabytes, and it grows linearly with sequence length. That is the LLM memory wall, and it is exactly the same disease as the MCU activation peak wearing a different costume: a transient tensor pool whose peak overlap, not the parameter count, is the binding budget. I will return to it; the dedicated treatment lives in [LLM quantization for activations, SmoothQuant, and the KV-cache](/blog/machine-learning/edge-ai/llm-quantization-activations-smoothquant-kv-cache) and in the runtime-focused [KV-cache optimization and management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management).

#### Worked example: the model that fit flash and OOMed SRAM

Make the intro story concrete with numbers. The classifier had:

- Parameters: 256 KB (int8). Lives in flash. Costs **zero** SRAM if read in place.
- Input image buffer: a 96×96×3 uint8 tensor zero-padded and converted to int8 working format — call it **A = 96 KB** after the first conversion layer widens it.
- First conv output feature map: **B = 64 KB**.
- Second conv output feature map: **C = 60 KB**.

At the moment the second convolution runs, it reads from A's downstream tensor and B, and writes C, so the input A's tensor, B, and C are partially overlapping in life. The naive runtime kept A alive (it was still referenced by a skip connection), kept B alive (still being read), and allocated C — peak $= 96 + 64 + 60 = 220$ KB for activations. Add a 64 KB IO/scratch buffer the framework reserved and a few KB of runtime bookkeeping and you cross 320 KB. Hard fault.

The parameter count, 256 KB, never entered this calculation. It was a red herring the entire time. The fix, which we will build up to, was to (a) break the skip connection's hold on A by reordering, (b) run the activation through an in-place op where possible, and (c) let the arena reuse B's bytes for C. After that the peak fell to 160 KB and the model ran with room to spare — same weights, same accuracy, same FLOPs. We changed nothing about the *model*; we changed how its memory was *planned*.

## The science: tensor lifetimes and the peak as maximum overlap

To shrink the peak you first have to define it precisely, and the precise definition is borrowed wholesale from compiler theory. I will build it up from one idea: the **lifetime** (or live interval) of a tensor.

Order the operators of the network in the sequence they will execute — call the steps $t = 1, 2, \dots, T$. A tensor $x$ is **produced** at the step of the operator that writes it, $\text{def}(x)$, and it is **last used** at the latest step of any operator that reads it, $\text{lastuse}(x)$. Between those two steps the tensor must occupy memory; before $\text{def}(x)$ it does not exist, and after $\text{lastuse}(x)$ it is dead and its bytes are free for reuse. The lifetime is the closed interval

$$
\text{live}(x) = [\,\text{def}(x),\ \text{lastuse}(x)\,].
$$

Each tensor also has a size in bytes, $s(x)$. Now the working set at any step $t$ is the sum of sizes of all tensors whose lifetime contains $t$:

$$
W(t) = \sum_{x \,:\, t \in \text{live}(x)} s(x).
$$

And the **peak working set** — the quantity that has to fit your fast RAM — is the maximum of $W(t)$ over the whole schedule:

$$
P = \max_{1 \le t \le T} W(t) = \max_{t} \sum_{x \,:\, t \in \text{live}(x)} s(x).
$$

That equation is the whole science of this post in one line. Peak memory is the *maximum overlap of live intervals, weighted by tensor size*. Read it carefully and three levers fall out immediately, because there are exactly three things in the equation you can change.

1. You can change which tensors *overlap* by changing the **schedule** — the order of operators. Reorder so that the producer of a big tensor and the consumer that frees a different big tensor don't straddle the same step. This shrinks $W(t)$ at the peak step.
2. You can change the *size* of each tensor with **quantization** (int8 halves the bytes versus fp16, quarters versus fp32) or with **tiling** (compute only a slice of a tensor at a time so the full tensor never materializes). This shrinks $s(x)$.
3. You can change *whether a tensor exists at all* with **in-place ops** (the output aliases the input, so there is one tensor where there were two) and with **buffer reuse** in the allocator (two non-overlapping lifetimes share the same physical bytes — this doesn't lower $W(t)$ but it lowers the physical arena you need to realize it).

Now, the deep analogy. If you have ever studied compilers, the equation for $P$ should look familiar, because it is precisely the cost function of **register allocation**. In a CPU, variables are live over intervals of the instruction stream, and the compiler must map them to a finite set of physical registers. Two variables can share a register if and only if their live intervals do not overlap. The number of registers you need is the maximum number of simultaneously-live variables — the **chromatic number** of the *interference graph*, where each variable is a node and an edge connects two variables whose lifetimes overlap. Graph coloring on the interference graph is NP-hard in general, which is why real compilers use linear-scan and graph-coloring heuristics.

Memory planning for a neural network is the same problem with one twist: instead of uniform-size registers, tensors have different sizes, so it is not graph *coloring* but a kind of **2D rectangle packing**. Picture a chart with time on the horizontal axis and memory offset on the vertical axis. Each tensor is a rectangle: its width is its lifetime interval, its height is its byte size. Memory planning is the task of stacking these rectangles vertically (assigning each a starting offset) so that no two rectangles that overlap in time overlap in their memory range, and the total height used — the peak — is minimized. That is **dynamic storage allocation**, proven NP-hard, and the descendants of register allocation's heuristics (greedy by size, greedy by lifetime) are exactly what production runtimes use. We will implement a greedy version in a few sections.

![A grid showing four tensors across three execution steps with their live and free states, where step two has three tensors live at once forming the peak working set of two hundred twenty kilobytes.](/imgs/blogs/memory-is-the-real-constraint-3.png)

The figure makes the overlap visible. Tensor A is live across steps 1 and 2; B is live across steps 2 and 3; C is live only at step 2. At step 2 all three coexist, giving $W(2) = 96 + 64 + 60 = 220$ KB, and that is the peak. At steps 1 and 3 only two tensors are live, so the working set is lower. The peak is set entirely by the single worst column, step 2 — which is the practical face of $P = \max_t W(t)$. Everything we do to cut peak memory is, at bottom, an attack on that worst column: make a tensor in it smaller, make one of its tensors die sooner, or never let one of its tensors fully exist.

A subtle but important consequence: **the peak is local.** It is determined by one moment, not by the average. You can have a network that is comfortably under budget for 99% of its operators and blows up at a single layer — almost always an early layer where the spatial resolution is still large. This is why "the model is small on average" is no comfort. One fat feature map at one step decides everything, the same way a single p99 outlier decides your tail latency. Profilers that report *average* memory will lie to you here exactly the way an average latency lies about the tail; you must find the peak step.

### The lower bound, and why an allocator can miss it

There are two distinct numbers hiding in this problem, and conflating them costs you real KB. The first is the **maximum overlap**, $P = \max_t W(t)$ — a property of the *lifetimes alone*, independent of any allocator. It is a hard lower bound: no allocation scheme, however clever, can ever fit the model in less than $P$ bytes, because at the peak step those tensors genuinely all exist at once and must each have their own bytes. The second number is the **achieved arena size** — what a specific allocator actually needs after assigning offsets. The achieved size is always greater than or equal to the lower bound, and the gap between them is **fragmentation**.

Fragmentation arises because the allocator commits to an offset for each tensor *before* it knows everything that will come later, and a poor early choice can wedge a later tensor higher than necessary. A concrete miniature: suppose at the peak step tensor X (100 KB) and tensor Y (20 KB) are both live, lower bound 120 KB. If the allocator places a *third*, short-lived tensor Z (50 KB) at offset 100 KB during an earlier step, and Z's lifetime happens to clip the start of X's, the allocator may be forced to push X up to offset 50 KB, after which Y no longer fits in the hole below X and lands at 150 KB — an arena of 170 KB for a 120 KB lower bound. The 50 KB of slack is fragmentation, pure waste, born of an allocation order that did not see the future. This is exactly why greedy-by-size places the *biggest* tensors first: the big ones are the hardest to fit later, so giving them the clean low offsets while the graph is still empty minimizes the chance that a small tensor wedges a big one upward. It is the same instinct as packing a suitcase with the hardback books before the socks.

The practical upshot is a two-number discipline. Always compute both: the lower bound (the `peak_overlap` function below — pure lifetimes) and the achieved arena (what your runtime reports). If they match, the allocator is optimal for this graph and there is nothing left to win in *placement* — you must attack the lifetimes themselves (reorder, in-place, tile). If the arena is well above the bound, you have a fragmentation problem and a better heuristic or a compaction pass can recover the gap *without changing the model at all*. Knowing which of the two situations you are in tells you whether to reach for the scheduler or the allocator, and that diagnosis alone has saved me from optimizing the wrong thing more than once.

## The memory breakdown, layer by layer

Before optimizing, you have to see clearly where the bytes are. Let me lay out the three pools and what each is allowed to live in, because the placement options are themselves a lever.

![A vertical stack of memory layers showing static parameters and code that can stream from flash, transient activations that must be in fast RAM, and the peak working set that sizes the tensor arena.](/imgs/blogs/memory-is-the-real-constraint-2.png)

The static pool — parameters, code, constant tensors — is the forgiving one. On a microcontroller it lives in flash/ROM and is execute-in-place: the CPU fetches weight bytes directly from flash during the convolution, no RAM copy. On a phone or single-board computer the OS memory-maps the weight file, so pages load lazily and the kernel can evict them under pressure. The point is that the static pool can almost always be pushed off the fast-RAM budget. When people say "stream the weights from flash," this is what they mean: keep weights in cheap, plentiful, slow storage and never copy the whole set into the precious pool.

The transient pool — activations — has no such escape. An activation tensor is read and written on the inner loop of the operator that touches it; it has to be in the fastest memory you have or every access pays a cache-miss or a DRAM round trip. On an MCU that means SRAM, full stop. On an accelerator it means the on-chip SRAM tile, and spilling an activation to DRAM is the difference between a roofline that is compute-bound and one that is hopelessly memory-bound (see [the roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) for why a spilled activation can tank achieved throughput). So the activation peak is the number that has to fit, and the arena — the single contiguous block the runtime carves up for all activations — is sized to exactly that peak.

There is also a third, smaller pool that bites people: **runtime scratch**. Stack space for the call tree, IO buffers (the camera DMA buffer, the audio ring buffer), workspace some kernels demand (im2col scratch for a convolution implemented as a matrix multiply, or a winograd transform buffer). These are easy to forget and they coexist with the activation peak, so they add directly to it. On the M7 story above, a 64 KB camera DMA buffer was live during the whole forward pass and stacked right on top of the 220 KB activation peak. When you budget, budget all three pools at the worst moment.

Let me put rough numbers on the three model classes so the relative shapes are concrete. These are order-of-magnitude figures for orientation, not benchmark results.

| Pool | MCU CNN (int8) | MobileNetV2 (int8, 224²) | Llama-7B (Q4) |
| --- | --- | --- | --- |
| Parameters | ~256 KB (flash) | ~3.4 MB | ~3.5 GB |
| Peak activation | ~200–384 KB | ~2–3 MB | ~tens of MB/step |
| KV-cache | none | none | ~0.5 GB per 1k tokens |
| What binds you | **peak activation** | **peak activation** | **KV-cache** |

Read the bottom row. For the two CNNs, the parameter count and the peak activation are within a small factor of each other, and the peak activation is what you fight. For the LLM, the parameters are gigabytes — but they stream and are static — while the KV-cache, born transient and growing with every token, is what climbs into the gigabytes and eventually OOMs the device or forces you to truncate context. Same disease, three costumes. We will fill in this table more precisely in the results section; the figure later visualizes which slice binds each class.

## Operator scheduling: reorder to reduce the peak

The cheapest lever — it costs zero accuracy, zero FLOPs, zero model change — is to **reorder the operators** so fewer big tensors are alive at the same step. Because $P = \max_t W(t)$ depends on which lifetimes overlap, and lifetimes are determined by the execution order, choosing a better topological order of the compute graph can lower the peak for free.

Here is the intuition with a branch. Suppose a node $X$ feeds two subgraphs, a long branch $L$ (many ops, produces a big intermediate) and a short branch $S$ (one op), and the two branches are added at the end. If you schedule the long branch *first*, then while you are grinding through $L$ you must keep $X$ alive the whole time (because $S$ hasn't run yet and still needs $X$), and $L$'s big intermediate piles on top. If instead you schedule the short branch $S$ first, you free $X$'s claim from $S$ early, and during the long branch only $L$'s intermediates are live. Same graph, same result, different peak. The classic rule of thumb from compiler scheduling — the Sethi–Ullman insight — is to **evaluate the more memory-hungry branch first when it can free its inputs sooner, and otherwise evaluate the branch that frees the most memory first.** The general problem of finding the schedule that minimizes peak is NP-hard (it is, again, a relative of register allocation), but greedy heuristics get most of the win.

Concretely, TensorFlow Lite, ONNX Runtime, TVM, and ExecuTorch all run a scheduling/planning pass at conversion or compile time. They build the graph, compute lifetimes for the chosen topological order, and either accept that order or perturb it to reduce the peak before handing the result to the memory planner. You usually do not write this pass yourself, but you *can* influence it: the order you write residual branches, whether you fuse ops (a fused conv+bias+relu produces one output instead of three), and whether you force a concatenation early (concats are peak-killers because they keep all their inputs alive simultaneously) all change the schedule the planner sees.

To see this concretely rather than take it on faith, here is a small routine that computes the peak for a given topological order and then sweeps the valid orders to find the one that minimizes it. For a real graph you would use a heuristic, not brute force, but on a small branch the exhaustive search makes the scheduling effect undeniable.

```python
from itertools import permutations

# graph as: node -> (output_bytes, [input nodes])
# a residual block: x -> conv1 -> conv2 -> add(x, conv2)
graph = {
    "x":     (96_000, []),
    "conv1": (64_000, ["x"]),
    "conv2": (60_000, ["conv1"]),
    "add":   (60_000, ["x", "conv2"]),   # skip connection holds x alive
}

def is_valid(order):
    seen = set()
    for n in order:
        if not all(d in seen for d in graph[n][1]):
            return False
        seen.add(n)
    return True

def peak_for_order(order):
    # last-use step of each tensor under this order
    pos = {n: i for i, n in enumerate(order)}
    last_use = {n: pos[n] for n in graph}            # at least its own def
    for n in order:
        for dep in graph[n][1]:
            last_use[dep] = max(last_use[dep], pos[n])
    peak = 0
    for step in range(len(order)):
        live = sum(graph[n][0]
                   for n in graph
                   if pos[n] <= step <= last_use[n])
        peak = max(peak, live)
    return peak

valid = [o for o in permutations(graph) if is_valid(o)]
best = min(valid, key=peak_for_order)
worst = max(valid, key=peak_for_order)
print(f"worst order {worst} -> {peak_for_order(worst)//1000} KB")
print(f"best  order {best}  -> {peak_for_order(best)//1000} KB")
```

Running it prints two different peaks for the *same graph* — the only difference is the order in which the operators execute. The best valid order schedules the branch so the skip's hold on `x` overlaps the fewest large tensors, and the peak drops accordingly. This is the entire mechanism a production planner automates; seeing it as a search over a couple of valid orders demystifies why two compilers can give the same model different arena sizes.

#### Worked example: reordering a residual block cuts the peak

Take a residual block: input $X$ (96 KB), a two-conv branch producing $M_1$ (64 KB) then $M_2$ (60 KB), and a final add $Y = X + M_2$ (60 KB). The skip connection means $X$ must stay alive until the add.

Schedule A (naive, conv branch interleaved with $X$ held): at the step that produces $M_2$, the live set is $\{X, M_1, M_2\}$. But wait — is $M_1$ still alive when $M_2$ is produced? $M_1$ is consumed by the second conv that produces $M_2$, so $M_1$'s last use is that very step; with the producing operator reading $M_1$ and writing $M_2$, both are momentarily live, and $X$ is held for the add. Peak $= 96 + 64 + 60 = 220$ KB.

Schedule B (free $M_1$ before allocating the add's output, and make the second conv in-place where shape allows): now at the add step, $M_1$ is already dead (last use was the second conv), so the live set is $\{X, M_2\}$ when the add runs, $96 + 60 = 156$ KB, and the add can write $Y$ over $M_2$'s buffer in place because $M_2$ has no further use. Peak drops to **156 KB** from 220 KB — a 29% reduction with no change to the model whatsoever. The accuracy is bit-identical; the FLOP count is identical; we only changed the order and reuse. That is the cheapest 64 KB you will ever save.

## Memory planning: the arena allocator

Now the allocator itself. The runtime does not call `malloc` per tensor on an edge device — `malloc` fragments, has per-allocation overhead, and on an MCU there may be no heap at all. Instead, edge runtimes use a **tensor arena**: a single contiguous block of memory, allocated once, that the planner statically carves into offsets, one per tensor. TFLite-Micro is the canonical example — you hand it a `uint8_t tensor_arena[ARENA_SIZE]` and it places every activation tensor at a computed offset inside it, with **no malloc at runtime at all.** Determinism, zero fragmentation, and a single number — the arena size — that is exactly your peak working set (plus alignment slack).

The planner's job is the rectangle-packing problem from earlier: assign each tensor an offset such that no two tensors whose lifetimes overlap occupy overlapping byte ranges, and minimize the total span. Because that is NP-hard, planners use greedy heuristics. The one TFLite uses (and a good default to implement) is **greedy-by-size**: sort tensors from largest to smallest, and place each one at the *lowest offset* that does not collide (in time) with any already-placed tensor. Big tensors go down first so they get the clean low offsets; small tensors slot into the gaps. It is not optimal, but it is within a small factor and runs in milliseconds.

![A before and after comparison contrasting fresh allocation per tensor that sums every byte against an arena that reuses freed offsets to fit the same graph in its peak working set.](/imgs/blogs/memory-is-the-real-constraint-4.png)

The figure captures the whole point of an arena. On the left, allocating a fresh buffer per tensor and never reusing freed bytes forces you to provision the *sum* of all tensors ever produced — 220 KB for our toy graph. On the right, the arena recognizes that tensor C's lifetime does not overlap tensor B's (B died before C was born), so C can reuse B's exact bytes; the arena only needs to be as big as the *peak overlap*, 160 KB. The savings is the gap between the sum of all tensors and the maximum simultaneously-live subset — and that gap is enormous for any real network, where most tensors are short-lived.

Let me make this runnable. Here is a self-contained greedy arena planner that takes tensor lifetimes and sizes and assigns offsets, computing both the achieved arena size and the theoretical lower bound (the peak overlap). You can drop your own model's lifetimes into it.

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class Tensor:
    name: str
    size: int        # bytes
    first: int       # step it is produced (def)
    last: int        # last step it is used (lastuse)
    offset: int = field(default=-1)  # assigned by the planner

def overlaps_in_time(a: Tensor, b: Tensor) -> bool:
    # closed intervals overlap iff neither ends before the other starts
    return not (a.last < b.first or b.last < a.first)

def peak_overlap(tensors: List[Tensor]) -> int:
    # the theoretical lower bound: max over steps of summed live bytes
    steps = range(min(t.first for t in tensors),
                  max(t.last for t in tensors) + 1)
    best = 0
    for s in steps:
        live = sum(t.size for t in tensors if t.first <= s <= t.last)
        best = max(best, live)
    return best

def greedy_by_size(tensors: List[Tensor]) -> int:
    # place largest first at the lowest non-colliding offset
    placed: List[Tensor] = []
    for t in sorted(tensors, key=lambda x: -x.size):
        candidate = 0
        # collect forbidden byte-ranges from time-overlapping placed tensors
        blockers = sorted(
            (p for p in placed if overlaps_in_time(t, p)),
            key=lambda p: p.offset,
        )
        for b in blockers:
            if candidate + t.size <= b.offset:
                break                      # fits in the gap below b
            candidate = max(candidate, b.offset + b.size)
        t.offset = candidate
        placed.append(t)
    return max(t.offset + t.size for t in placed)

# the toy graph from the worked examples
graph = [
    Tensor("A_input", 96_000, first=1, last=2),
    Tensor("B_feat1", 64_000, first=2, last=3),
    Tensor("C_feat2", 60_000, first=2, last=2),
]

arena = greedy_by_size(graph)
bound = peak_overlap(graph)
print(f"greedy arena size : {arena/1000:.0f} KB")
print(f"peak overlap bound: {bound/1000:.0f} KB")
for t in graph:
    print(f"  {t.name:8s} bytes={t.size//1000}K "
          f"life=[{t.first},{t.last}] offset={t.offset//1000}K")
```

Run it and you see the arena planner reusing space: A and C never overlap in time with each other in a way that forces stacking everywhere, B and C overlap at step 2 so they must be stacked, and the greedy placement lands close to the lower bound. The `peak_overlap` function is your sanity check — it is the $P = \max_t W(t)$ from the science section, computed directly. If your planner reports an arena much larger than `peak_overlap`, the planner is fragmenting and you have a tuning problem; if it equals the bound, the planner is doing as well as is possible for these lifetimes.

One refinement worth knowing: the greedy-by-size heuristic does not always reach the lower bound because of *fragmentation* — a small tensor placed early can wedge a big tensor up to a higher offset. Production planners (TFLite's `MemoryPlanner`, TVM's storage rewrite, ExecuTorch's memory planning passes) add tie-breaking by lifetime and sometimes a second pass to compact. But the structure is exactly what is above: sort, place at lowest non-colliding offset, report the span.

Two real-world details turn this textbook planner into a shippable one. The first is **alignment**: hardware wants tensors aligned to 16, 32, or even 64 bytes (for SIMD loads, DMA, or NPU tiling), so each offset is rounded up to the alignment, and the rounding introduces small gaps that inflate the arena slightly above the pure lower bound. On an MCU with a 256 KB budget those gaps are usually a few hundred bytes total — negligible — but on a planner that aligns every one of hundreds of tensors to 64 bytes, the slack adds up and is worth checking when you are within a few KB of the budget. The second is **persistent versus transient tensors**: some tensors are not part of the reusable activation flow at all — the model's input and output buffers, any tensor a calling application holds a reference to, the KV-cache in an LLM. These must be carved out of the reusable arena (or placed in a separate region) because their lifetimes span the whole inference and they cannot be overwritten. TFLite distinguishes these classes explicitly; if you ever see a planned arena larger than your hand-computed peak overlap, the gap is usually alignment plus a persistent tensor you forgot to count, not a planner bug.

It is also worth saying *why* the static, offline arena is the right design for the edge specifically, rather than a general-purpose heap. A heap (`malloc`/`free`) makes allocation decisions at runtime, which means it (a) fragments unpredictably as tensors of different sizes come and go, (b) carries per-allocation metadata overhead, (c) has non-deterministic timing that ruins the hard-real-time guarantees an embedded system often needs, and (d) requires an MMU and an allocator that many bare-metal targets do not have. The arena sidesteps all four: every offset is decided once, offline, from the known graph; there is zero runtime allocation, zero fragmentation, constant-time tensor access by precomputed offset, and a single integer — the arena size — that you can certify fits the device before you ever flash it. That determinism is not a nice-to-have on a safety-relevant embedded system; it is the reason the arena pattern exists.

## In-place operators: when the output can overwrite the input

The arena reuses bytes *across* tensors with disjoint lifetimes. An in-place operator goes one better: it makes the output and the input *the same tensor*, so there is one allocation where naively there were two. This is legal precisely when the operator reads each input element, computes its output, and never needs that input element again — and when no *other* consumer still needs the input.

The canonical in-place ops are elementwise and shape-preserving: **ReLU** (and ReLU6, clamp, hardswish), **add** for a residual when one addend is dead afterward, **bias add**, dropout at inference (a no-op), and dtype-stable elementwise math. ReLU is the poster child: $y_i = \max(0, x_i)$ touches each element once and the input is not needed again, so writing $y$ over $x$ is exact. In PyTorch this is the difference between `nn.ReLU()` and `nn.ReLU(inplace=True)`, and between `y = x + r` and `x.add_(r)`. On an MCU runtime the planner detects in-place-able ops and aliases their buffers automatically; in eager PyTorch you ask for it explicitly.

![A dataflow graph showing a convolution output buffer feeding either an out of place ReLU that allocates a new buffer or an in place ReLU that writes back to the same buffer and saves an allocation.](/imgs/blogs/memory-is-the-real-constraint-5.png)

The figure shows the fork. The out-of-place path allocates a fresh 64 KB buffer Y for the ReLU output; the in-place path writes back into the conv's existing buffer X and the 64 KB allocation simply never happens. Across a deep network with a ReLU after every conv, that is one saved activation-sized buffer per block — and because the peak is set by the worst block, eliminating the extra buffer at *that* block is what counts.

A small but real measurement to make the effect tangible. Here is how you watch in-place ops move the peak in PyTorch using the CUDA memory tracker (the same logic applies on CPU with `torch.cpu` memory tools or a custom hook; CUDA just has the cleanest counter).

```python
import torch
import torch.nn as nn

def peak_mb(model, x):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    with torch.no_grad():
        _ = model(x)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2

class Block(nn.Module):
    def __init__(self, inplace):
        super().__init__()
        self.c1 = nn.Conv2d(16, 64, 3, padding=1)
        self.r1 = nn.ReLU(inplace=inplace)
        self.c2 = nn.Conv2d(64, 64, 3, padding=1)
        self.r2 = nn.ReLU(inplace=inplace)
    def forward(self, x):
        return self.r2(self.c2(self.r1(self.c1(x))))

x = torch.randn(1, 16, 128, 128, device="cuda")
oop = Block(inplace=False).cuda().eval()
ip  = Block(inplace=True ).cuda().eval()

print(f"out-of-place ReLU peak: {peak_mb(oop, x):6.2f} MB")
print(f"in-place     ReLU peak: {peak_mb(ip,  x):6.2f} MB")
```

The in-place version reports a lower peak because the two ReLU output buffers are folded into the conv outputs. The exact delta depends on tensor sizes and what else PyTorch's caching allocator is holding, which is the honest caveat: `max_memory_allocated()` measures the allocator's high-water mark, and PyTorch's caching allocator reserves slabs, so you read it as a *relative* signal between two runs of the same harness, not an absolute arena size. For absolute arena numbers you go to the runtime that will actually ship the model (TFLite-Micro's reported arena, ExecuTorch's planned memory), which we cover in the measurement section.

A caution that has burned people: in-place ops are *not* free if the input is still needed elsewhere. If `x` feeds both a ReLU and a skip connection, an in-place ReLU corrupts the skip's copy. In eager PyTorch this raises an autograd error during training (the saved tensor was modified) but at inference it can silently produce wrong numbers. Runtimes guard against this by checking the consumer count before aliasing — they only go in-place when the input has exactly one remaining consumer. When you write models, be aware that aggressive in-place can break a residual you forgot about; the planner's consumer-count check is what keeps it safe, and you should trust it over hand-aliasing.

## Tiling and patch-based inference: never materialize the fat tensor

Reordering, arenas, and in-place ops all work on tensors as atomic blocks — they decide *where* and *whether* a whole tensor lives. The most powerful lever for the worst case attacks the tensor's *size* directly: don't compute the whole tensor at once. This is **tiling**, and for the early layers of a convolutional network the specific form is **patch-based inference**, which is the central trick that let MCUNetV2 fit ImageNet-scale models onto microcontrollers with a few hundred KB of SRAM.

The problem it solves is the early-layer fat-feature-map disease. In a typical CNN, the input is large in spatial dimensions and small in channels (224×224×3), and the first few blocks keep the resolution high while growing channels. The activation tensor right after the first block might be 112×112×16 — that is over 200 K elements, and even at int8 that is 200 KB for one tensor, before you have done any real work. The peak working set is dominated by these early, spatially-large feature maps. Later layers, where resolution has collapsed to 7×7 and channels are large, have tiny spatial extent and small activation tensors. The memory profile of a CNN is front-loaded.

Patch-based inference exploits the locality of convolution. A convolution's output pixel depends only on a small **receptive field** of input pixels. So instead of computing the entire early feature map and then proceeding, you compute the output of the early stage **one spatial tile at a time**: take a patch of the input, run it through the first several layers to produce the corresponding patch of the stage's output, store that small output patch, and move to the next patch. At no point does the *full* early feature map exist in memory — only the current input patch, the intermediate patches within the receptive field, and the accumulating output. The peak collapses from "the whole fat feature map" to "one patch plus its halo."

![A before and after comparison contrasting materializing a whole early feature map at high peak SRAM against patch based tiling that keeps only one spatial tile and its halo live to fit the budget.](/imgs/blogs/memory-is-the-real-constraint-6.png)

The cost is not free, and the trade-off is exactly the kind of honest engineering decision this series is about. Adjacent patches share input pixels at their boundaries — the **halo** or **receptive-field overlap** — so a naive tiling recomputes those boundary convolutions for each patch. The deeper into the network you tile and the larger the receptive field, the more overlap and the more redundant compute. MCUNetV2's insight was twofold: (1) only patch-tile the early, memory-heavy stages and switch back to whole-tensor processing once the resolution has dropped enough that the full feature map fits; and (2) co-design the network — specifically reduce the receptive field of the patched stages with a "receptive-field redistribution" — so the recompute overhead stays small (they report on the order of 10–20% extra MACs for a roughly 4–8× reduction in peak SRAM). That is the bargain: you spend a little compute to buy a lot of peak memory, and on a memory-bound MCU that is a trade you take every time.

#### Worked example: patch inference shrinks MCU peak SRAM

These numbers are MCUNetV2-style figures and should be read as approximate and illustrative, drawn from the spirit of Lin et al.'s reported results rather than a single exact benchmark row. Consider an early stage whose full output feature map is 160×160×16 in int8 — that is $160 \times 160 \times 16 = 409{,}600$ bytes, about **1.28 MB**. No microcontroller with a 320 KB SRAM budget can hold that tensor, full stop; the model is infeasible as written.

Now tile the input into, say, a 4×4 grid of spatial patches. Each output patch is roughly $40 \times 40 \times 16 \approx 25.6$ KB, and with the receptive-field halo and the few intermediate patch tensors of the patched sub-network, the live set during patch processing is on the order of **256 KB** — it fits the SRAM with room for the IO buffers. The peak fell from ~1.28 MB to ~256 KB, roughly a 5× reduction, by never materializing the full feature map. The price was perhaps 10–20% more MACs from boundary recomputation, which on a stage that was memory-bound in the first place costs little wall-clock time. The model went from infeasible to shipping. That is the whole MCUNetV2 story in one example, and it is why patch-based inference is the single highest-leverage memory technique for vision on microcontrollers. The companion post [squeezing models into kilobytes](/blog/machine-learning/edge-ai/squeezing-models-into-kilobytes) goes deeper on the full TinyML pipeline that wraps this trick.

Transformers have their own version of "don't materialize the fat tensor." The attention score matrix is $n \times n$ for sequence length $n$, which is the quadratic memory term that dominates long-context activations. **FlashAttention** is exactly tiling applied to attention: it computes attention in blocks, never materializing the full $n \times n$ score matrix, keeping only the running softmax statistics and a tile of scores live at once. The peak activation for attention drops from $O(n^2)$ to $O(n)$ in the same way patch inference drops the CNN peak — by streaming the computation in tiles rather than building the whole tensor. The principle generalizes: anywhere a large intermediate tensor is consumed locally, you can tile its computation and trade a little recompute for a large drop in peak.

## Streaming weights and the static-pool escape hatch

I have been treating parameters as the easy pool, and on a constrained device the way you keep them easy is **weight streaming**: never copy the full weight set into fast RAM. There are three flavors, in rough order of device class.

On a **microcontroller**, weights live in flash and are read in place. The convolution kernel's inner loop fetches weight bytes directly from the flash address space (Cortex-M flash is memory-mapped and execute-in-place capable). The weights cost *zero* SRAM. The catch is flash read bandwidth and latency — flash is slower than SRAM, so a weight-heavy layer can become flash-bandwidth-bound. The usual answer is to keep the *currently-executing* layer's weights small enough that streaming them keeps the MACs fed, which is one more reason depthwise-separable convolutions (few weights per FLOP) suit MCUs; see [the MobileNet family](/blog/machine-learning/edge-ai/the-mobilenet-family) for that architecture.

On a **phone or single-board computer**, you `mmap` the weight file. The OS pages weight pages into the page cache on demand and evicts them under memory pressure. The framework sees a pointer; the kernel handles residency. `llama.cpp` does exactly this with GGUF files — it `mmap`s the quantized weights so a 4 GB model does not require 4 GB of committed RAM up front, and pages stay shared across processes. The win is fast startup and lower committed memory; the cost is that the first pass over each layer pays a page-fault to pull weights from storage, which is why cold-start latency is worse than warm.

On an **accelerator** (NPU, GPU), weights stream from DRAM into the on-chip SRAM tile per layer, and the scheduler double-buffers — prefetching the next layer's weights while the current layer computes — to hide the DRAM latency. Here the static pool is "cheap" only in the sense that DRAM is plentiful; the real cost is the energy and time to move weight bytes from DRAM to SRAM, which is precisely the memory-bound regime of the roofline. For LLM decode, weight streaming *is* the bottleneck: each token must read the entire weight set from DRAM, so decode throughput is set by memory bandwidth, and quantizing weights to int4 roughly doubles tokens/s by halving the bytes moved — even though the math is unchanged. That is the deep tie between this post and quantization: int4 weights are not (mainly) about fitting more parameters; they are about moving fewer bytes per token.

The decision tree of which weight-placement strategy applies — and more broadly which memory lever to reach for — falls out of one question: which pool is binding you?

![A decision tree starting from peak memory over budget that branches on whether activations or weights and cache dominate, leading to reorder and arena and tiling on one side and weight streaming and KV quantization on the other.](/imgs/blogs/memory-is-the-real-constraint-8.png)

The tree encodes the discipline: **profile first, then the dominant slice tells you the lever.** If activations dominate (the CNN/MCU case), you reach for scheduling, arenas, in-place ops, and tiling. If weights or the KV-cache dominate (the LLM case), you reach for weight streaming and KV-cache quantization or paging. The worst engineering mistake here is to apply an activation lever to a weight-bound problem or vice versa — tiling a model whose peak is the KV-cache does nothing, and quantizing the KV-cache of a model whose peak is an early feature map does nothing. Measure, classify, then act.

## The KV-cache: the LLM's version of the activation wall

I have been promising that the LLM memory wall is the same disease in a different costume, so let me make that precise, because the math is clean and it is the binding constraint for on-device language models. A decoder transformer generates one token at a time. To produce token $t$, attention needs the keys and values of every previous token — and rather than recompute them, the runtime *caches* them. That cache, the **KV-cache**, is a transient tensor pool that grows by one token's worth of keys and values per generated token, in every layer, for every attention head. It is precisely an activation that refuses to die: its lifetime spans the entire generation, and its size climbs monotonically with sequence length.

The size is a simple product, and it is worth deriving because the factors tell you every lever. For a model with $L$ layers, hidden dimension $d$ (split across heads), a context of $n$ tokens, a batch of $b$ sequences, and $p$ bytes per element, the KV-cache holds keys *and* values (the factor of 2):

$$
\text{KV bytes} = 2 \cdot b \cdot L \cdot n \cdot d \cdot p.
$$

Every factor is a lever. The $2$ is fixed (you need both K and V). $b$ is batch — at the edge it is 1, which already helps. $L$ and $d$ are the architecture — a smaller model, or one using grouped-query or multi-query attention (which share K and V across query heads and so shrink the effective $d$ for the cache by the group factor), cuts this directly. $n$ is context length — the term that grows without bound and the reason long context is expensive. And $p$ is bytes per element — quantizing the cache from fp16 ($p=2$) to int8 ($p=1$) or int4 ($p=0.5$) halves or quarters the whole cache, which is the single most effective on-device KV lever.

#### Worked example: the KV-cache that dwarfs the activations

Take a 7B-class model: $L = 32$ layers, hidden dimension $d = 4096$, full multi-head attention, fp16 cache ($p = 2$), batch $b = 1$. At a context of $n = 4096$ tokens:

$$
\text{KV bytes} = 2 \cdot 1 \cdot 32 \cdot 4096 \cdot 4096 \cdot 2 \approx 4.3\ \text{GB}.
$$

That is **larger than the int4 weights themselves** (~3.5 GB). On a 16 GB laptop or an 8 GB phone, the KV-cache at long context, not the weights, is what runs you out of memory — and unlike the weights it cannot stream from flash, because every cached K and V is read on every subsequent step's attention. The per-step activation memory (the current token's hidden states, the attention scores for one query against $n$ keys) is on the order of tens of MB; it is a rounding error next to the cache. The cache is the wall.

Now pull the levers. Switch to **grouped-query attention** with 8 KV-heads sharing 32 query heads — the effective cache dimension drops by 4×, taking the cache to ~1.1 GB. Then **quantize the cache to int8** ($p = 1$): another 2×, to ~540 MB. Now the cache fits comfortably and you can extend context further. Notice what just happened: we attacked the exact same $P = \max_t W(t)$ equation, but the dominant transient tensor was the KV-cache rather than an early feature map, so the levers were architecture (GQA shrinks $d$) and quantization (int8 shrinks $p$) rather than tiling and arenas. The discipline is identical; only the dominant tensor changed.

There is a placement lever too, the cache's analogue of the arena: **paging**. Naively the runtime reserves one contiguous KV buffer sized to the *maximum* context, even when most requests are short — that is the "fresh malloc per tensor" mistake at the sequence level, reserving the worst case for every request. PagedAttention (the idea behind vLLM) instead allocates the cache in fixed-size **blocks** and maps them through a block table, so a request only holds blocks for the tokens it has actually generated, and freed blocks are reused across requests. It is the arena allocator's offset-reuse trick lifted to the KV-cache: pack live blocks tightly, reuse freed ones, and size the pool to the aggregate *peak* of live blocks rather than the sum of every request's worst case. Same packing problem, same NP-hard-relative, same greedy answer. The full treatment lives in [LLM quantization for activations, SmoothQuant, and the KV-cache](/blog/machine-learning/edge-ai/llm-quantization-activations-smoothquant-kv-cache) and the serving-oriented [KV-cache optimization and management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management); the point here is that the LLM wall is not a special case — it is $P = \max_t W(t)$ with a different fat tensor.

## How to measure peak memory honestly

You cannot optimize what you cannot measure, and peak memory is easy to mis-measure. Each runtime has a different counter, and each lies in a different way. Here is the honest version for the three you will actually use.

**PyTorch (development / GPU).** The CUDA caching allocator tracks a high-water mark. Reset it, run inference, read it.

```python
import torch

def measure_peak(model, example_input, warmup=2, iters=5):
    model.eval()
    # warm up so cuDNN picks algorithms and the allocator settles
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(example_input)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(example_input)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()      # bytes actually live
    reserved = torch.cuda.max_memory_reserved()   # bytes the allocator grabbed
    return peak, reserved

peak, reserved = measure_peak(model, torch.randn(1, 3, 224, 224, device="cuda"))
print(f"peak allocated: {peak/1024**2:7.1f} MB")
print(f"peak reserved : {reserved/1024**2:7.1f} MB")
```

The trap: `max_memory_allocated()` is the bytes *requested and live*, which is the number you want; `max_memory_reserved()` is the bytes the caching allocator grabbed from the driver, which is larger and noisier. Report `max_memory_allocated`. Warm up first, because the first run allocates cuDNN workspace and algorithm-selection scratch that inflate the peak. And remember this is a relative tool for comparing two versions of the same model on the same hardware — it is not the arena size the shipped runtime will use.

**TFLite / LiteRT (mobile, the planned arena).** After conversion you can read the runtime's reported arena size, which is the real activation budget the deployed model needs.

```python
import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="model_int8.tflite")
interpreter.allocate_tensors()

# the arena the interpreter carved for all activations:
arena_bytes = interpreter._interpreter.GetTensorArenaSize() \
    if hasattr(interpreter._interpreter, "GetTensorArenaSize") else None
print("reported arena:", arena_bytes)

# fallback: sum the tensor byte sizes flagged as arena-resident,
# or read the conversion log which prints the planned arena.
details = interpreter.get_tensor_details()
total = sum(np.prod(d["shape"]) * np.dtype(d["dtype"]).itemsize
            for d in details if d["shape"].size)
print("sum of all tensor bytes (upper bound):", total)
```

The honest number is the **planned arena**, not the sum of all tensors. The converter's memory planner reuses buffers, so the arena is far smaller than the naive sum — and that arena, in bytes, is what must fit your device. The conversion process logs the planned arena size; capture it.

**TFLite-Micro (the MCU, the real budget).** This is the cleanest measurement in all of ML, because the framework is malloc-free and tells you the exact arena it used. You over-provision an arena, run one inference, and ask how much it actually consumed.

```cpp
#include "tensorflow/lite/micro/micro_interpreter.h"

constexpr int kArenaSize = 512 * 1024;          // over-provision generously
alignas(16) static uint8_t tensor_arena[kArenaSize];

// ... set up resolver, model, interpreter with tensor_arena ...
TfLiteStatus s = interpreter.AllocateTensors();
if (s != kTfLiteOk) { /* arena too small: grow kArenaSize */ }

// the exact bytes the planner needed for activations + scratch:
size_t used = interpreter.arena_used_bytes();
printf("arena used: %u bytes (%.1f KB)\n",
       (unsigned)used, used / 1024.0f);
```

`arena_used_bytes()` is the ground truth: the exact peak working set the planner computed for your model, in bytes, with no allocator slack and no estimation. You shrink `kArenaSize` to just above that number and you have your SRAM budget. This is the function I wish I had called *before* flashing the M7 that hard-faulted — it would have printed 384 KB next to my 320 KB budget and saved me a day.

A discipline note that applies to all three: measure at **batch size 1**, which is the edge reality, and measure the *worst* input shape if your model has dynamic shapes (the longest sequence, the largest image), because the peak is set by the worst case, not the average. And separate the three pools when you report — "the arena is 384 KB" is actionable; "the model uses 0.7 MB" lumps weights, arena, and scratch and tells you nothing about which to cut.

## Results: before and after memory planning

Let me pull the levers together on the running CNN example and show the measured arrangement. The model is the int8 microcontroller classifier from the intro: 256 KB of weights in flash, targeting a Cortex-M7 with 320 KB SRAM. Every number below is the activation arena (the binding pool); weights are in flash and excluded.

| Stage of optimization | Peak activation arena | Fits 320 KB SRAM? | What changed |
| --- | --- | --- | --- |
| Naive (fresh buffer per tensor) | 384 KB | No (over by 64 KB) | baseline |
| + Arena with offset reuse | 300 KB | Barely | reuse disjoint lifetimes |
| + Operator reorder (free skips early) | 248 KB | Yes | schedule for min overlap |
| + In-place ReLU / add | 200 KB | Yes, comfortably | alias elementwise outputs |
| + Patch-tile the first block | 160 KB | Yes, with IO headroom | never materialize fat map |

The arc is the point. We started infeasible (384 KB on a 320 KB device) and ended at 160 KB — a **2.4× reduction in peak memory** — without touching the model's weights, accuracy, or FLOP count by more than the small patch-recompute overhead. The accuracy is unchanged to the bit on the first three rows and within noise on the last (the patched stage computes the identical convolution, just in tiles). This is the most important result in the post: **the largest memory wins on the edge come from planning, not from compression.** Compression (quantization, pruning) shrinks the bytes; planning decides whether the shrunk bytes fit. You need both, and planning is the one people skip.

Now layer in quantization to show how the levers compose, because this is where the series' four-lever frame pays off. Quantization does double duty: it halves the *weight* bytes (fp16→int8) *and* halves the *activation* bytes, which directly halves $s(x)$ for every tensor in $P = \max_t W(t)$. So moving a model from fp16 activations to int8 activations roughly halves the peak arena before you do any planning at all.

| Activation dtype | Peak arena (after planning) | Notes |
| --- | --- | --- |
| fp32 activations | 640 KB | infeasible on most MCUs |
| fp16 activations | 320 KB | exactly at budget, risky |
| int8 activations | 160 KB | comfortable, the shipped config |

Each halving of bit-width halves the arena, because the arena is bytes and bytes scale with bit-width. This is why int8 (or lower) is effectively mandatory on microcontrollers — not only for the weight size but for the *activation* peak. The accuracy cost of int8 activations is small with proper calibration (see [post-training quantization](/blog/machine-learning/edge-ai/post-training-quantization-ptq) and the full int8 pipeline), and the memory payoff is a clean 2×. Combine int8 with planning and patch tiling and you go from a 640 KB fp32 model that cannot exist on the device to a 160 KB int8 model with headroom — a 4× total reduction in peak, accuracy intact.

![A matrix comparing memory budgets across a microcontroller CNN, MobileNetV2, and a quantized Llama 7B, showing which slice of params, peak activation, or KV cache binds each model class.](/imgs/blogs/memory-is-the-real-constraint-7.png)

The matrix generalizes the result across model classes and is the figure to internalize. For the MCU CNN, peak activation (384 KB pre-planning) is the binding limit and the planning levers above are the fix. For MobileNetV2, peak activation (a couple of MB) still binds on a phone's NPU SRAM tile and patch-style processing or DRAM spill management is the lever. For Llama-7B in int4, the parameters are 3.5 GB but stream, the per-step activation is modest, and the **KV-cache** — 2 GB at 4k context — is the binding limit; the lever is KV-cache quantization and paging, not activation planning. One framework, three different binding pools, three different levers. The discipline is always: profile, find the binding pool, then pull the matching lever.

## When peak memory is THE constraint (and when it isn't)

Every technique in this series is a cost, and the honest question is when memory planning is worth the engineering. Here is my decision rule, hard-won.

**Reach for memory planning first when:** you are on a microcontroller (any device with KB–single-digit-MB of SRAM — memory is almost always the binding constraint there, before latency or accuracy); the model *fits in flash but won't allocate* (the classic symptom — `AllocateTensors` fails or the arena estimate exceeds SRAM); your accelerator is spilling activations to DRAM and the roofline shows you memory-bound; or you are running an LLM and the KV-cache is forcing you to cap context length below what the product needs. In all of these, no amount of making the model "smaller" by parameter count helps — the peak working set is the wall, and you attack it with scheduling, arenas, in-place ops, tiling, streaming, and activation/KV quantization.

**Do not over-invest in memory planning when:** you are on a phone or single-board computer with gigabytes of DRAM and the model's peak is comfortably under it — there, latency and energy are the real constraints and you should be reading [the metrics that actually matter on device](/blog/machine-learning/edge-ai/the-metrics-that-actually-matter-on-device) and the roofline, not packing arenas. The runtime's default planner is already good; hand-tuning offsets is a waste. Memory planning has sharply diminishing returns once the peak fits with headroom — the gap between "fits" and "fits with 30% headroom" is rarely worth a week of work.

A few stress tests to pressure the decision, because the edge cases are where you actually get paged:

- **What if the calibration of int8 activations costs accuracy?** Then you mix precision — keep the early, sensitive layers (which are also the memory-heavy ones) at higher precision and quantize the rest, accepting a slightly higher peak in exchange for accuracy. This is the [mixed-precision and sensitivity analysis](/blog/machine-learning/edge-ai/mixed-precision-and-sensitivity-analysis) trade-off colliding with memory; sometimes the right answer is to patch-tile the sensitive fp16 layer so its higher bit-width does not blow the peak.
- **What if patch tiling's recompute makes the model too slow?** Tile fewer stages — only the ones whose full feature map exceeds the budget — and switch to whole-tensor processing as soon as the resolution drops enough to fit. The recompute overhead is concentrated in the stages with the largest receptive-field overlap, so tiling only the first one or two blocks usually captures most of the memory win at a fraction of the recompute cost.
- **What if the NPU doesn't support an op and it falls back to CPU?** The fallback can force a tensor to materialize in a different memory space, breaking the planner's reuse and spiking the peak at the boundary. The fix is to fuse or replace the unsupported op so the whole subgraph stays on the accelerator; an op that round-trips to CPU costs you both latency and a memory spike.
- **What if the model is memory-bandwidth-bound, not capacity-bound?** Then peak working set isn't the problem — moving bytes is. The lever shifts from "fit the arena" to "move fewer bytes": quantize to reduce bytes per tensor, fuse ops to avoid round-tripping activations through DRAM, and keep the working set resident in SRAM. The roofline tells you which regime you are in; capacity and bandwidth are different walls and demand different levers.
- **What if you free up just enough and the model still crashes intermittently?** This is the signature of the runtime-scratch pool you forgot to budget. The activation arena fits, but a DMA buffer, an interrupt-time stack frame, or a kernel's transient im2col workspace claims the last few KB at exactly the wrong moment, and the failure is non-deterministic because it depends on timing. The fix is to budget all coexisting pools at the worst instant, not just the arena — and on an MCU, to place the largest scratch buffers statically so they are accounted for at compile time rather than competing dynamically. Intermittent OOM is almost never the model; it is an unbudgeted pool overlapping the peak.

The meta-lesson across all of these stress tests is that memory is a *systems* property, not a *model* property. The same set of weights can fit or not fit depending on the schedule the compiler chose, the alignment the hardware demanded, the ops the accelerator supported, the precision you calibrated to, and the scratch buffers the surrounding application reserved — none of which appear in the model file. That is why "is the model small enough?" is the wrong question and "does the peak working set, plus every coexisting pool, fit the fast RAM at the worst instant?" is the right one. Train yourself to ask the second question and most edge OOMs become diagnosable in minutes instead of days.

## Case studies and real numbers

Five anchors from the literature and shipped systems, to ground the techniques in named results. Where I give a number I cite the source; where I round, I say so.

**MCUNet and MCUNetV2 (Lin et al., MIT, NeurIPS 2020 and 2021).** This is the foundational TinyML memory work. The original MCUNet co-designed the network (via neural architecture search) and the inference engine (TinyEngine) to fit ImageNet-class models onto Cortex-M microcontrollers with on the order of 256–512 KB of SRAM, where prior work could not fit at all. MCUNetV2 added **patch-based inference** to break the early-layer activation peak, reporting roughly a 4–8× reduction in peak SRAM by tiling the memory-heavy initial stages, with a small (order 10–20%) recompute overhead managed by receptive-field redistribution. The headline: ImageNet at usable accuracy on a device with less SRAM than a single uncompressed early feature map. That is the proof of concept that memory planning plus co-design, not raw compression, is what makes microcontroller vision possible.

**TFLite-Micro's memory planner (Google).** The TensorFlow Lite for Microcontrollers runtime is the production embodiment of the arena. It is malloc-free: you provide a static `tensor_arena`, the planner computes offsets with a greedy heuristic at `AllocateTensors` time, and `arena_used_bytes()` reports the exact peak. The design paper (David et al., "TensorFlow Lite Micro," MLSys 2021) documents the offline planning and the no-heap discipline that make memory deterministic on devices with no MMU. This is the runtime where the ideas in this post become concrete API calls.

**FlashAttention (Dao et al., 2022) as attention tiling.** On the transformer side, FlashAttention reduced attention's activation memory from $O(n^2)$ — the materialized score matrix — to $O(n)$ by computing attention in tiles with an online softmax, never writing the full $n \times n$ matrix to memory. The reported effect was both a large memory reduction (enabling much longer sequences on the same GPU) and a speedup from avoiding the DRAM round-trips of the big matrix. It is patch-based inference's idea applied to attention, and it shows the principle is not MCU-specific: tile the fat tensor, trade a little recompute, win a lot of peak.

**llama.cpp weight streaming via mmap (open source).** On laptops and single-board computers, `llama.cpp` `mmap`s GGUF weight files so a multi-gigabyte quantized model does not require committing that much RAM up front — pages load on demand and are shared across processes and runs. Combined with k-quant weight quantization (Q4_K_M and friends), this is what lets a 7B model run on a 16 GB laptop or a Raspberry Pi: the weights stream from storage, the per-step activations are small, and the KV-cache is the pool you watch and quantize as context grows. The static pool is kept cheap exactly as the theory says it can be.

**PagedAttention / vLLM (Kwon et al., SOSP 2023).** On the serving side, the KV-cache's "reserve the worst case per request" waste was measured to leave large fractions of GPU memory unusable to fragmentation and over-reservation. PagedAttention allocates the cache in fixed-size blocks mapped through a block table — the arena's offset-reuse trick lifted to the sequence level — recovering most of that wasted memory and so raising the number of concurrent sequences a given GPU can hold. The reported effect was a multiple-fold increase in serving throughput, driven almost entirely by fitting more KV-cache into the same memory. It is the cleanest demonstration that the LLM memory wall is a packing problem, not a parameter problem: nothing about the model changed, only how its transient cache was placed. The same principle scales down to a single on-device session, where paging lets a long conversation reuse freed blocks instead of holding the maximum-context reservation for its whole life.

## Key takeaways

- **The budget is the peak working set, not the parameter count.** $P = \max_t W(t)$ — the maximum, over the whole schedule, of the total bytes of simultaneously-live tensors. A model can fit flash and still OOM the fast RAM.
- **Three pools, three placements.** Parameters are static and can stream from flash/mmap/DRAM (cheap). Activations are transient and must be in fast RAM (the hard pool). Runtime scratch coexists with the activation peak — budget all three at the worst moment.
- **Peak memory is a packing problem, the cousin of register allocation.** Tensors are rectangles in time × memory; planning packs them to minimize the span. It is NP-hard, so runtimes use greedy-by-size heuristics — and so should you when you reason about it.
- **The cheapest win is scheduling.** Reordering operators to minimize live-interval overlap costs zero accuracy, zero FLOPs, zero model change, and routinely cuts the peak 20–30%.
- **Arenas reuse bytes across disjoint lifetimes; in-place ops fold two tensors into one.** Together they take you from "sum of all tensors" to "peak overlap," which for a real network is a huge gap. Trust the planner's consumer-count check before aliasing.
- **Patch-based inference is the highest-leverage memory lever for edge vision.** Tiling the early, fat-feature-map stages cuts peak SRAM several-fold for a small recompute cost — the trick that put ImageNet on microcontrollers.
- **Quantization halves the arena too.** Activation bytes scale with bit-width, so int8 roughly halves the peak versus fp16 — a memory win on top of the weight-size win. The levers compose.
- **Profile, classify the binding pool, then pull the matching lever.** Activation-bound → schedule/arena/in-place/tile. Weight-or-KV-bound → stream weights / quantize-and-page the KV-cache. Measure with `arena_used_bytes()`, the TFLite planned arena, or `max_memory_allocated`, at batch 1 and worst-case shape.
- **The largest edge memory wins come from planning, not compression.** Compression shrinks the bytes; planning decides whether they fit. Skipping planning is the most common reason a "small enough" model still won't run.

## Further reading

- **Lin, Chen, Wang, Gan, Han — "MCUNet: Tiny Deep Learning on IoT Devices" (NeurIPS 2020)** and **"MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning" (NeurIPS 2021).** The foundational patch-inference and TinyML co-design work; read V2 for the peak-SRAM reduction mechanism.
- **David et al. — "TensorFlow Lite Micro: Embedded Machine Learning on TinyML Systems" (MLSys 2021).** The malloc-free arena and offline memory planner; the source of `arena_used_bytes()` and the no-heap discipline.
- **Dao, Fu, Ermon, Rudra, Ré — "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (NeurIPS 2022).** Attention tiling that drops activation memory from quadratic to linear; the transformer analogue of patch inference.
- **The dynamic storage allocation / register allocation literature** (Sethi–Ullman ordering; Chaitin-style graph-coloring allocation; linear-scan allocation, Poletto and Sarkar 1999). The theory behind why memory planning is a packing problem and why greedy heuristics are the practical answer.
- **Within this series:** [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) for the four-lever frame; [the roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) for memory-bound vs compute-bound diagnosis; [the metrics that actually matter on device](/blog/machine-learning/edge-ai/the-metrics-that-actually-matter-on-device) for measuring peak memory honestly; [the edge hardware landscape](/blog/machine-learning/edge-ai/the-edge-hardware-landscape) for the SRAM/flash/DRAM budgets per device class; [squeezing models into kilobytes](/blog/machine-learning/edge-ai/squeezing-models-into-kilobytes) for the full TinyML pipeline; and the capstone [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) for putting every lever together. For the LLM memory wall specifically, see [KV-cache optimization and management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management).

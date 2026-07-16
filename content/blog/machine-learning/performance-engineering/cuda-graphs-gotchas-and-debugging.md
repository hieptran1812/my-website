---
title: "CUDA graphs gotchas: dynamic shapes, stale pointers, and why your replay produces garbage"
date: "2026-07-14"
publishDate: "2026-07-14"
description: "The honest failure-mode guide to CUDA graphs — the five ways replay silently corrupts your output, why the failures don't crash, and the profiler-and-allclose workflow to diagnose every one before you ship."
tags:
  [
    "performance-engineering",
    "gpu-optimization",
    "cuda-graphs",
    "cuda",
    "profiling",
    "pytorch",
    "inference",
    "latency",
    "ml-systems",
    "debugging",
  ]
category: "machine-learning"
subcategory: "Performance Engineering"
author: "Hiep Tran"
featured: true
readTime: 41
---

You graph your model. The trace goes from a forest of tiny kernels with idle gaps between them to a single fat `cudaGraphLaunch`, host overhead per step falls from six milliseconds to under two, throughput jumps 3x, and you are ready to write the celebratory Slack message. Then someone on the eval team pings you: the graphed service is returning slightly wrong answers. Not crashes. Not NaNs. Wrong. The classifier that was 94% accurate is now 71%. The decoder emits fluent-looking text that has quietly drifted off the prompt. And the worst part — it was fine in the smoke test, because the smoke test happened to reuse the exact tensor you captured on.

This is the defining characteristic of CUDA graphs, and the reason they deserve a whole post on failure modes: **they fail by corrupting, not by crashing.** A normal bug throws an exception, and the exception points at the line. A graph bug produces plausible numbers that are subtly, sometimes catastrophically wrong, and it does so *at full speed*, because the whole point of a graph is that it stops asking questions and just replays a fixed sequence of kernels. There is no bounds check, no shape check, no "is this the tensor you meant" check at replay time. The graph does exactly what you recorded, byte for byte, and if the world changed underneath it, the graph does not notice.

![a table of five cuda graph failure classes with columns for symptom root cause and fix showing four of the five corrupt output silently](/imgs/blogs/cuda-graphs-gotchas-and-debugging-1.webp)

This post is the debugging companion to the two siblings that taught you the API — [CUDA graphs from first principles](/blog/machine-learning/performance-engineering/cuda-graphs-from-first-principles) (capture versus replay, the recorded kernel DAG) and [CUDA graphs in PyTorch](/blog/machine-learning/performance-engineering/cuda-graphs-in-pytorch) (`torch.cuda.graph`, `make_graphed_callables`, static I/O). You know how to capture. This post is everything that will break *after* you do, laid out as five failure classes, each with a symptom, a mechanism, a repro, and a fix — plus the one test that catches all of them before a wrong number ever reaches a user. By the end you will have a correctness gate you can drop into CI and a decision tree that turns "my graph is wrong" from a night of confusion into a two-minute lookup. This is the honest version of the CUDA-graphs story, and it belongs to the same profile → hypothesize → fix → measure loop that runs through the whole [performance engineering series](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu).

## The five ways a graph betrays you

Before we go deep, here is the whole failure surface on one page, because the single most useful thing you can do when a graph misbehaves is recognize *which* of a small, fixed set of failures you are looking at. There are five, and they are shown in the figure above.

1. **Dynamic shapes.** Capture froze one shape. Feed a different shape and replay reads or writes the wrong buffer extent — garbage if you are lucky enough to stay in mapped memory, an illegal access if you run off the end.
2. **Stale or changed pointers.** Replay launches the *exact captured addresses*. If you allocate a fresh input tensor each step instead of copying into the static buffer the graph captured, replay reads the old address and you get stale or garbage output.
3. **CPU-side control flow.** A data-dependent branch (`if x.item() > 0`) inside the captured region is a host decision plus a forced device-to-host sync. Capture either aborts or, worse, freezes whichever branch happened to be taken during capture, and replays that branch forever.
4. **Illegal sync / disallowed ops during capture.** `.item()`, `.cpu()`, `print(tensor)`, a device-to-host `cudaMemcpy`, most raw allocations — any of these while a stream is capturing throws "operation not permitted when stream is capturing." This one is *loud*, which makes it the friendliest failure in the set.
5. **RNG and stateful ops.** A random op inside the region needs graph-safe RNG state. If it does not get it, every replay draws the *same* "random" numbers, because the philox offset was baked in at capture and never advances.

Notice the asymmetry: four of the five corrupt output silently. Only class 4 crashes. That is exactly backwards from what your debugging instincts expect — you are trained to fear the loud failures, but with graphs the loud one is the easy one and the quiet ones are what page you at 2am. Hold that thought; it is the reason the correctness gate later in this post is not optional.

## Why replay corrupts instead of crashing

To debug any of these you need a precise mental picture of what a replay actually *is*, because every failure class is a violation of the same invariant. Let me make the invariant exact.

When you capture a region, CUDA records the stream's work as a directed acyclic graph of nodes. Most nodes are kernel launches. Crucially, each kernel-launch node stores its arguments *by value at capture time*: the raw device pointers of every input and output tensor, the grid and block dimensions computed from the captured shapes, and any scalar launch parameters. Call the captured graph a fixed list of launches $L = [\ell_1, \ell_2, \ldots, \ell_K]$, where each $\ell_i$ carries a frozen tuple $(\text{ptr}_i, \text{grid}_i, \text{scalar}_i)$. A replay is nothing more than re-issuing that exact list to the GPU in one host call. No Python runs. No shapes are recomputed. No addresses are looked up. The launch arguments are whatever they were when you pressed record.

From this, the correctness condition falls right out. Replay is correct if and only if, at replay time, for every launch $\ell_i$:

$$\text{addr}_\text{replay}(\ell_i) = \text{addr}_\text{capture}(\ell_i) \quad\text{and}\quad \text{extent}_\text{replay}(\ell_i) = \text{extent}_\text{capture}(\ell_i)$$

and the *bytes* living at each captured input address hold the data you intend for this step. That is the entire contract. Every failure class is a way of breaking one clause:

- **Stale pointer** breaks the first clause when you hand the graph a new tensor at a new address — the graph still reads $\text{ptr}_i$ from capture, so it reads the wrong buffer. Or it breaks the "bytes hold the intended data" clause when the address is right but you forgot to refill it.
- **Dynamic shape** breaks the extent clause: $\text{grid}_i$ was sized for the captured shape, so a bigger input under-reads (misses data, garbage) and a launch writing a bigger output over-runs the captured buffer (illegal access).
- **Control flow** breaks the "the list $L$ is the right list" assumption: only one branch's kernels got recorded, so replay always runs that branch regardless of the data.
- **RNG** breaks the intended-data clause on the RNG state buffer: the philox counter was frozen, so the same numbers come out every time.

It is worth being explicit about *why we tolerate* a construct this fragile, because the payoff is exactly what makes the invariant worth respecting. An eager step launches every kernel from the host, so its per-step host cost is $C_\text{eager} = K \cdot t_\text{launch}$, where $t_\text{launch}$ is the 5-to-10 microseconds of CPU-side driver work each `cudaLaunchKernel` costs. For a step with $K = 900$ tiny kernels — a small transformer forward — that is 4.5 to 9 milliseconds of pure launch overhead, and if each kernel runs in only a few microseconds on the GPU, the device sits idle most of that time waiting for Python to feed it. A replay collapses all $K$ launches into a single `cudaGraphLaunch`, so $C_\text{graph} \approx t_\text{launch}$ regardless of $K$. The host cost drops by a factor of $K$. That is the entire prize, and it is enormous precisely when the step is a long chain of small kernels — the host-bound regime. The fragility is the price of that collapse: to launch $K$ kernels in one call, the driver must have every launch's arguments baked in ahead of time, which is exactly why the pointers and grids are frozen. You cannot have the speed without the frozen arguments, and you cannot have the frozen arguments without the invariant. Respecting the invariant is not defensive paranoia; it is the contract that buys the 900x reduction in launch count.

Now, *why silent?* Because a kernel launch on the GPU does not validate its pointer arguments against a shape or a type or an owner. It receives an address and a grid, and it dutifully reads `grid × block` threads' worth of data from that address and writes wherever it was told. If the address is mapped (which it usually is — the allocator caches freed blocks and hands them back out, so a "freed" buffer's memory is very much still there), the read succeeds and returns *whatever bytes happen to be there now*. No page fault, no exception, just wrong numbers flowing downstream. You only get a loud `illegal memory access` when a launch's grid runs the read or write past the end of a mapped allocation into an unmapped page — which is why dynamic-shape overruns sometimes crash and stale pointers almost never do. The GPU is a very fast, very literal machine executing a recording. It has no way to know the recording no longer matches reality.

Everything below is a concrete instance of breaking one of those clauses, and a concrete way to detect and fix it.

## Failure class one: illegal ops abort capture (the loud, friendly one)

Start with the friendly failure, because it teaches the capture model and it is the only one that announces itself. Certain operations are simply forbidden while a stream is capturing, and hitting one throws immediately.

![a vertical stack of five forbidden operations during capture ending in a capture abort error because each forces a host sync or an untracked allocation](/imgs/blogs/cuda-graphs-gotchas-and-debugging-2.webp)

The forbidden set all share one trait: each forces the GPU to talk back to the host *right now*, or allocates memory outside the graph's pool. Capture is a recording of asynchronous GPU work; an operation that requires a synchronous answer from the device cannot be recorded as a replayable node, so CUDA refuses it. The usual offenders in PyTorch code:

- `tensor.item()` and `float(tensor)` / `int(tensor)` — pull one scalar back to the CPU, which is a device-to-host copy plus a sync.
- `tensor.cpu()`, `tensor.numpy()`, `tensor.tolist()` — copy the whole tensor to host memory.
- `print(tensor)` — reads values, so it syncs; a stray debug print is the classic capture-killer.
- an explicit `torch.cuda.synchronize()` or a device-to-host `cudaMemcpy`.
- most raw allocations that bypass the caching allocator's graph pool.

Here is what it looks like when you leave a `.item()` inside the region, and — importantly — how to catch the error instead of letting it kill the process:

```python
import torch

model = build_model().cuda().eval()
static_input = torch.randn(8, 512, 768, device="cuda")

g = torch.cuda.CUDAGraph()

# Warm up on a side stream first (required before capture).
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        _ = model(static_input)
torch.cuda.current_stream().wait_stream(s)

try:
    with torch.cuda.graph(g):
        out = model(static_input)
        # BUG: a data-dependent guard sneaks a sync into the captured region.
        if out.abs().max().item() > 100.0:   # <-- .item() forces a D2H sync
            out = out.clamp(-100, 100)
except RuntimeError as e:
    print("CAPTURE FAILED:", e)
    # Re-raise only after logging; do NOT swallow silently in a real service.
    raise
```

The console output is unambiguous, which is the whole reason this class is the easy one:

```console
CAPTURE FAILED: CUDA error: operation not permitted when stream is capturing
CUDA kernel errors might be asynchronously reported at some other API call, so the
stacktrace below might be incorrect. For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

RuntimeError: operation not permitted when stream is capturing
```

The fix is to remove the sync from the region. The clamp is fine — it is a pure GPU op — but the *decision* to clamp based on `.item()` is not. Either clamp unconditionally (do the work every time; on a GPU a branch you always take is cheaper than a sync anyway), or move the decision to the host outside the graph. We will return to the control-flow version of this problem in class four. For now the lesson is mechanical: **wrap your capture in `try/except RuntimeError`, log the message, and treat "operation not permitted when stream is capturing" as a signpost pointing at a hidden sync.** Grep the captured region for `.item()`, `.cpu()`, `.numpy()`, `.tolist()`, and `print(` before you even run it.

One subtlety that trips people up: the error may surface a few lines *after* the real culprit, because CUDA reports capture errors lazily at the next API call. If the traceback points at an innocent-looking op, set `CUDA_LAUNCH_BLOCKING=1` (more on that later) to make the report line up with the offending call.

## Failure class two: stale pointers (the silent one that eats your accuracy)

This is the failure that produced the wrong-accuracy story in the intro, and it is the most common CUDA-graph bug in the wild because the code that causes it looks completely reasonable.

![a broken graph that allocates a fresh tensor each step so replay reads the stale captured address versus a fixed graph that copies into a static buffer](/imgs/blogs/cuda-graphs-gotchas-and-debugging-3.webp)

Recall the invariant: replay reads from the captured input address $\text{ptr}_i$. The captured address belongs to whatever tensor you passed *during capture* — the `static_input` above. If, at inference time, you do the natural thing and build a fresh input tensor per request, that tensor lives at a *different* address, and the graph never sees it. Replay reads the old address, which either still holds the capture-time data (stale answer) or has been reallocated to something else (garbage). Here is the bug in its most seductive form:

```python
# WRONG — looks correct, silently reads stale memory on replay.
g = torch.cuda.CUDAGraph()
static_input = torch.randn(8, 512, 768, device="cuda")

# ... warmup on side stream, then capture ...
with torch.cuda.graph(g):
    static_output = model(static_input)

def infer(new_batch):            # new_batch is a fresh tensor at a NEW address
    g.replay()                   # replay still reads static_input's OLD address
    return static_output.clone() # returns the answer for the CAPTURED input!
```

`infer` ignores `new_batch` entirely. The graph has no idea `new_batch` exists — it was never told the address. Every call returns the output computed for `static_input` at capture time. If your smoke test happens to pass the same values you captured on, it passes. The moment real traffic arrives, every response is the answer to a question nobody asked. That is exactly the "94% to 71%" collapse from the intro: the model is running perfectly, on the wrong inputs.

The fix is the single most important pattern in all of CUDA-graph usage: **fill the static input buffer in place, then replay.**

```python
# CORRECT — copy new data INTO the captured buffer, then replay.
def infer(new_batch):
    static_input.copy_(new_batch)   # write new data at the SAME address
    g.replay()                      # now replay reads live data
    return static_output.clone()    # correct answer for new_batch
```

`copy_` writes `new_batch`'s bytes into `static_input`'s existing memory, so the address the graph captured now holds the data you want. The graph is none the wiser and none the slower — it still just replays. The `clone()` on the way out matters too: `static_output` is overwritten by the *next* replay, so if you hand out a reference to it without cloning, a concurrent or subsequent request will stomp your result. Copy in, clone out. That is the whole discipline.

Let me show what the failure looks like on a real diff, because seeing it once inoculates you. Capture on one input, replay, then swap in a new input *without* the `copy_`, and compare against eager:

```console
# eager reference on the NEW input
eager  out[:5] = tensor([ 0.7132, -1.2043,  0.5561,  2.1099, -0.4410], device='cuda:0')

# first graph replay, static buffer still holds capture data -> stale
graph  out[:5] = tensor([-0.8817,  1.2210, -0.0114,  0.5532,  1.7784], device='cuda:0')

torch.allclose(eager, graphed, atol=1e-3) = False
max abs diff = 0.4312
```

The numbers are the same *magnitude* as real activations — no NaN, no inf, nothing that a naive sanity check would flag. Only a direct comparison against the eager path reveals it. Which brings us to the single most valuable habit in this entire post.

#### Worked example: the copy_ fix on an A100

A production intent classifier — a small encoder, batch 8, sequence 512 — was host-bound at **31% GPU utilization** on an A100 80GB SXM, launching roughly 900 tiny kernels per forward pass. Graphing it was the obvious win, and the first version shipped with the stale-pointer bug: it built a fresh input tensor per batch and called `g.replay()`. Latency was gorgeous and accuracy was quietly wrong.

The table below is the honest before-and-after, and it includes the *broken* graph on purpose, because "fast and wrong" is a state you must be able to recognize and rule out.

| Variant | Correct? | GPU util | Host overhead/step | p50 latency | Throughput | Kernels launched/step |
| --- | --- | --- | --- | --- | --- | --- |
| Eager | yes | 31% | 6.4 ms | 6.8 ms | 147 req/s | ~900 |
| Broken graph (new tensor, no `copy_`) | **no** | 88% | 0.2 ms | 1.9 ms | 526 req/s | 1 (replay) |
| Fixed graph (`copy_` into static buffer) | yes | 86% | 0.2 ms | 1.9 ms | 521 req/s | 1 (replay) |

The broken and fixed graphs are *identical* on every performance axis — same utilization, same latency, same throughput — and differ only in the one column that matters. Speed cannot tell them apart; only correctness can. The fixed graph delivered the real win the eager path was leaving on the table: host overhead per step fell from 6.4 ms to 0.2 ms, latency dropped 3.6x, and utilization climbed from 31% to 86% because the GPU stopped waiting on Python to launch the next kernel. (For the mechanics of *why* host overhead was the bottleneck, see [the kernel launch overhead problem](/blog/machine-learning/performance-engineering/the-kernel-launch-overhead-problem).)

## The one test that catches all of it: eager versus graphed

If you take one thing from this post, take this: **before you trust a graph, run one fixed input through both the eager model and the graph replay and assert they agree.** This single gate catches stale pointers, dynamic-shape reads, frozen branches, and RNG staleness — every silent class at once — because all of them manifest as "the graph's output differs from eager's on the same input."

![a dataflow where one fixed input feeds both an eager forward and a graph replay then merges at an allclose check that branches to pass or fail](/imgs/blogs/cuda-graphs-gotchas-and-debugging-4.webp)

The gate is a dozen lines and it belongs in your CI, not just your notebook:

```python
import torch

def assert_graph_matches_eager(eager_model, graph_runner, sample_input, atol=1e-3, rtol=1e-3):
    """Gate a graphed model against its eager reference on a fixed input.
    graph_runner(x) must copy x into the static buffer and replay.
    """
    eager_model.eval()
    with torch.no_grad():
        ref = eager_model(sample_input).float()
        got = graph_runner(sample_input).float()

    torch.cuda.synchronize()  # make sure both paths finished before comparing
    if not torch.allclose(ref, got, atol=atol, rtol=rtol):
        diff = (ref - got).abs()
        raise AssertionError(
            f"GRAPH MISMATCH: max|diff|={diff.max().item():.4g} "
            f"mean|diff|={diff.mean().item():.4g} "
            f"(atol={atol}, rtol={rtol}) — do NOT ship this graph"
        )
    print(f"graph OK: max|diff|={ (ref-got).abs().max().item():.2g }")
```

Two things make this gate trustworthy. First, the `torch.cuda.synchronize()` before comparing — both the eager and graphed paths are asynchronous, and comparing before the GPU has finished would race. Second, the tolerance. A graph is not bit-identical to eager even when it is *correct*, because kernel fusion, different cuBLAS algorithm selection, and accumulation order all shift the last few bits. A correct graph typically lands at `max|diff|` around 1e-6 to 1e-5 in fp32 and 1e-3 to 1e-2 in bf16; a broken one lands at 0.1 or 1.0 or NaN. The gap between "correct graph noise" and "broken graph" is several orders of magnitude, so a loose `atol=1e-3` (fp32) or `atol=1e-2` (bf16) cleanly separates them without false alarms. When this gate fires, you have a bug; when it passes, you can ship.

Run it on more than one input, too. A stale-pointer bug that happens to capture on your test input will pass the gate if you reuse that exact input — so feed the gate a *different* random input than you captured on. That is the whole trick to catching class two: capture on input A, gate on input B.

Set the tolerance to the dtype, not to a habit. In fp32 a correct graph lands near 1e-6, so `atol=1e-4` leaves four orders of magnitude of headroom below the smallest real bug. In bf16, whose mantissa gives only about three decimal digits, a correct graph can differ from eager by 1e-2 or more purely from fusion and accumulation-order changes, so `atol=1e-2, rtol=1e-2` is right and a tight fp32 tolerance would raise constant false alarms. Match the gate to the precision the model actually runs in, and when the model is mixed-precision, compare in the output dtype and cast both sides to fp32 before the diff so the subtraction itself does not lose bits. A gate that false-alarms gets disabled, and a disabled gate catches nothing — so tune it once, correctly, and leave it on.

```console
graph OK: max|diff|=3.1e-06        # correct fp32 graph, ship it
```

```console
AssertionError: GRAPH MISMATCH: max|diff|=0.4312 mean|diff|=0.0873
(atol=0.001, rtol=0.001) — do NOT ship this graph
```

This gate is cheap — one eager forward and one replay — and it is the difference between finding the bug in CI and finding it in an incident review. Wire it in.

## Failure class three: dynamic shapes

Now the class that most often turns a silent corruption into a loud crash, which at least is honest of it. A graph captures one shape. Its kernels' grid dimensions were computed from that shape. Feed a different shape and every extent assumption is wrong.

![a two row grid where three variable length requests each pad up to the nearest bucket and each bucket owns one fixed shape graph](/imgs/blogs/cuda-graphs-gotchas-and-debugging-5.webp)

Consider what happens concretely when you capture on sequence length 512 and replay on sequence length 640. The attention kernel's grid was launched for 512 positions. On the longer input, those threads cover only the first 512 tokens — the last 128 are never processed, so their outputs are whatever was in the buffer before (garbage), and the tokens that *were* processed may read past the 512-shaped buffer into adjacent memory. If the over-read stays inside a mapped allocation you get silent garbage; if it runs off the end you get:

```console
RuntimeError: CUDA error: an illegal memory access was encountered
```

You cannot make one graph handle arbitrary shapes — that is fundamental, not a PyTorch limitation. A graph is a recording of specific-sized launches. The only correct strategies are to make the shape *not* vary, or to keep *one graph per shape*:

1. **Pad to a fixed size.** Always run at the maximum sequence length (or the maximum batch size), padding shorter inputs and masking the pad. One graph, one shape, always legal. You pay for the wasted compute on the pad tokens, which is fine when your inputs cluster near the max and terrible when they range from 10 to 4096.
2. **Bucket the shapes.** Pick a handful of sizes — 128, 256, 512 — capture one graph per bucket, and pad each request up to its nearest bucket. This is the grid in the figure above: a request of length 100 pads to 128 and uses the 128-graph; a request of length 400 pads to 512 and uses the 512-graph. You trade a little padding waste for a lot less than pad-to-max, at the cost of holding several graphs in memory.

Here is the bucketing pattern as a reusable dispatcher:

```python
import torch

BUCKETS = [128, 256, 512, 1024]  # ascending; pick to cover your traffic

class BucketedGraphRunner:
    def __init__(self, model, hidden, dtype=torch.bfloat16):
        self.model = model.eval()
        self.graphs = {}         # bucket -> (CUDAGraph, static_in, static_out)
        for b in BUCKETS:
            static_in = torch.zeros(8, b, hidden, device="cuda", dtype=dtype)
            # warmup on a side stream (omitted for brevity), then capture:
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self.model(static_in)
            torch.cuda.current_stream().wait_stream(s)
            with torch.cuda.graph(g):
                static_out = self.model(static_in)
            self.graphs[b] = (g, static_in, static_out)

    def _bucket_for(self, seqlen):
        for b in BUCKETS:
            if seqlen <= b:
                return b
        raise ValueError(f"seqlen {seqlen} exceeds largest bucket {BUCKETS[-1]}")

    def __call__(self, x):                       # x: (8, seqlen, hidden)
        seqlen = x.shape[1]
        b = self._bucket_for(seqlen)
        g, static_in, static_out = self.graphs[b]
        static_in.zero_()                        # clear stale pad region
        static_in[:, :seqlen].copy_(x)           # copy real tokens, rest is pad
        g.replay()
        return static_out[:, :seqlen].clone()    # slice back to real length
```

Three details carry the correctness. `static_in.zero_()` clears the pad region so leftover data from a previous request cannot leak into the current one through the mask's cracks. The `copy_` into `[:, :seqlen]` is the same static-buffer discipline from class two, now applied to a padded buffer. And the output slice `[:, :seqlen]` throws away the pad positions on the way out. Skip any of the three and you reintroduce a silent bug.

#### Worked example: variable sequence length on an L4

A summarization service on an L4 (24 GB, ~242 fp16 TFLOP/s, 300 GB/s) received requests from 40 to 900 tokens. The first graphed attempt captured at 512 and crashed on the first 900-token request with `illegal memory access` — the loud, honest version of the dynamic-shape failure. The naive fix, pad-everything-to-1024, worked but wasted enormous compute: the median request was 180 tokens, so it was doing 5.7x the necessary attention work and actually ran *slower* than eager on short inputs.

Bucketing at `[128, 256, 512, 1024]` fixed both. Below is the measured comparison; every number is one you would read off `torch.profiler` and `nvidia-smi dmon`.

| Strategy | Correct on 900 tok? | Median req compute waste | p50 latency (180 tok) | p50 latency (900 tok) | Graph memory |
| --- | --- | --- | --- | --- | --- |
| Single graph @512 | **crashes** | — | — | crash | 1 graph |
| Pad-to-max @1024 | yes | 5.7x on median | 14.2 ms | 12.9 ms | 1 graph |
| Bucketed [128..1024] | yes | ~1.4x on median | 4.1 ms | 12.9 ms | 4 graphs |

Bucketing recovered a 3.5x latency win on the common short requests that pad-to-max had thrown away, while still serving the long tail correctly, at the cost of holding four graphs resident. That memory cost is real — four captured graphs plus their static buffers — and it is the trade you accept for not wasting compute on padding. On a memory-tight box you cap the number of buckets; on a memory-rich A100 you can afford more, finer buckets and less padding waste.

The stress test that matters here: what happens on a request *longer* than your largest bucket? The dispatcher above raises rather than silently truncating — because silent truncation is exactly the class-three failure we are trying to avoid. Decide the policy explicitly: reject, or fall back to the eager path for the over-long tail. Never let an out-of-range shape reach `replay()`.

## Failure class four: CPU control flow and data-dependent branches

Class one showed that `.item()` inside capture throws. The deeper problem is the *branch* it was feeding. A graph records one straight-line sequence of GPU work. It cannot record "if the logits exceed a threshold, run this kernel, otherwise that one," because the branch is a decision made on the host, from a value that only exists after the GPU computes it — and reading that value back is exactly the forbidden sync.

There are two ways this bites. If the branch condition reads a device value (`if x.max().item() > 0`), capture aborts loudly, as in class one. But if the branch condition reads a *host* value that was fixed at capture time — say a Python flag, or a shape, or a config setting — capture *succeeds*, silently recording only the branch that was true during capture, and replays that branch forever. That is the nastier version: no error, just a graph that has hard-coded a decision that was supposed to be dynamic.

```python
# DANGEROUS — the branch is frozen at capture time.
use_extra_refine = compute_flag()   # host-side bool, True during capture

with torch.cuda.graph(g):
    out = model(static_input)
    if use_extra_refine:             # captured as ALWAYS-True; the else path
        out = refine(out)            # is never recorded and never replays
```

If `use_extra_refine` is `True` when you capture, the graph *always* refines, even on later steps where the flag is `False`. The `else` path simply does not exist in the recording. This passes the smoke test and fails in production the first time the flag flips — and it fails silently, because there is no error, just the wrong computation.

The fix is to **hoist all control flow out of the captured region.** Decide the branch on the host, and either keep separate graphs per branch or run the branch-specific part eagerly:

```python
# CORRECT — one graph per branch, host picks which to replay.
graph_plain  = capture_graph(model)                 # no refine
graph_refine = capture_graph(lambda x: refine(model(x)))

def infer(x, want_refine):
    static_input.copy_(x)
    (graph_refine if want_refine else graph_plain).replay()
    return static_output.clone()
```

The decision `want_refine` lives on the host, outside any graph, exactly where a CPU decision belongs. Each graph is a pure, branch-free recording. This is the same "bucket the variation" pattern as dynamic shapes — you enumerate the finite set of distinct straight-line computations and capture one graph for each. When the set of branches is small and known, this is clean. When it is large or unbounded (a loop whose trip count depends on the data, like autoregressive decode with early stopping), a single graph cannot express it — you graph the *inner* fixed-shape step and keep the loop and its stopping condition on the host, replaying the step graph each iteration. That is precisely the pattern the [serving-loop sibling](/blog/machine-learning/performance-engineering/cuda-graphs-in-a-serving-loop) builds on for decode.

The autoregressive-decode case is worth spelling out because it is where control flow and graphs most often collide in serving. A generation loop runs a step, samples a token, and stops when it emits an end-of-sequence token or hits a length cap — the stopping condition is a data-dependent branch on a value the GPU just produced. You cannot capture the whole loop as one graph, because the trip count is not known at capture time and the EOS check is a device-to-host read. What you *can* do is capture the single decode step — one token in, one token out, fixed KV-cache shape — as a graph, and run the loop, the sampling, and the EOS check on the host, replaying the step graph once per generated token. The step is where the launch overhead lives (it repeats hundreds of times), so graphing just the step captures nearly all the benefit while leaving the branch where it belongs.

The stress test for this pattern: what happens when the EOS check is *also* pulled into the graph by an over-eager refactor? The `.item()` on the sampled token aborts capture (class one, loud) — which is the good outcome, because it stops you before you ship. The dangerous version is a refactor that hard-codes "always generate the full 256 tokens" to avoid the sync, freezing a fixed trip count into the loop. That captures fine and runs fast, and every response is padded to 256 tokens regardless of when the model wanted to stop — a silent quality and cost regression. The fix is the same rule below: the stopping decision is a fork, and forks stay on the host.

The rule to internalize: **a graph is a straight line. Every fork in the road stays on the CPU.**

## Failure class five: RNG and stateful ops

The last silent class is the subtlest, because the code has no obvious sync and no obvious shape problem — it just quietly stops being random. Any op that consumes RNG state — dropout, `torch.randn`, sampling — advances a philox counter on the GPU. During capture, the *current* counter value is baked into the kernel's arguments. On replay, that same value is re-issued, so every replay produces the *identical* "random" draw. For training with dropout inside a graphed step, this means every step sees the same dropout mask; for a sampling decoder, every request draws the same tokens.

PyTorch handles this correctly *if* you let it. `torch.cuda.graph` and `make_graphed_callables` register the default CUDA generator with the graph and capture the RNG state as a *pointer to a live counter* rather than a frozen value, so each replay advances the offset and produces fresh draws. The failure mode appears when you use a generator PyTorch cannot see into — a manually constructed `torch.Generator`, CPU-side RNG feeding a `.cuda()` copy, or a custom kernel that reads its own seed. Those get frozen.

The diagnostic is delightfully direct: replay twice and check the outputs *differ*.

```python
import torch

def assert_rng_advances(graph_runner, static_input, n=3):
    """A capture-safe RNG region should produce DIFFERENT output across replays."""
    outs = []
    for _ in range(n):
        graph_runner(static_input)            # replay, RNG should advance
        outs.append(static_output.clone())
    # consecutive replays must differ if RNG is graph-safe
    for i in range(1, n):
        if torch.equal(outs[i], outs[i - 1]):
            raise AssertionError(
                f"RNG FROZEN: replay {i} identical to replay {i-1} — "
                f"random op inside graph is not capture-safe"
            )
    print("RNG advances across replays: OK")
```

Note the *inversion* from the correctness gate: for a deterministic region you assert eager and graphed *agree*; for an RNG region you assert consecutive replays *disagree*. Same input, same graph, different output — that is what healthy graph-safe RNG looks like. If the outputs are byte-identical across replays, your RNG is frozen and every "random" decision in production is a copy of whatever you rolled at capture time.

```console
RNG FROZEN: replay 2 identical to replay 1 — random op inside graph is not capture-safe
```

The fix is to use the default CUDA generator and let PyTorch's graph machinery register it — which it does automatically inside `torch.cuda.graph` and `make_graphed_callables` on modern PyTorch. If you seed per-request for reproducibility, seed the default generator *before* the region and let the graph advance it, rather than constructing a private generator the graph cannot track. When in doubt, run `assert_rng_advances` — it is the RNG counterpart to the allclose gate and just as cheap.

#### Worked example: dropout frozen in a training step

The RNG failure is easiest to see in training, where dropout runs inside the step and a frozen mask does measurable damage. A vision model was graphed with `make_graphed_callables` to cut launch overhead on an A100, with dropout at 0.1 inside the captured forward. An early version constructed a private `torch.Generator` to seed dropout per step for "reproducibility" — a generator the graph could not track. The result was a step that trained with the *same* dropout mask on every iteration, which is closer to training with no dropout at all than with real dropout: the network overfit the fixed mask and generalized worse.

The numbers told the story cleanly. The tell that the RNG was frozen was that two consecutive graphed steps on the same batch produced byte-identical loss — `assert_rng_advances` failed on step 2. Once the code dropped the private generator and let `make_graphed_callables` register the default CUDA generator, consecutive replays diverged as they should and validation accuracy recovered to the eager baseline.

| Variant | RNG advances? | Consecutive-step loss | Final val accuracy |
| --- | --- | --- | --- |
| Eager | yes | differ | 78.3% |
| Graph, private generator | **no** | identical | 74.1% |
| Graph, default generator | yes | differ | 78.2% |

The 4.2-point accuracy gap is not a rounding error — it is the entire cost of a frozen dropout mask, and it would have been invisible without the consecutive-replay check, because the loss curve looked *smooth and plausible* the whole way down. It just converged to a worse place. This is the training-side face of the same lesson: a graph's silent failures degrade quality without ever raising an error, and only an explicit gate catches them.

## Failure class six: allocator interactions

The sixth class is less a distinct bug than a family of ways the caching allocator and the graph step on each other, and it is worth a short section because it foreshadows the memory track of this series. Two interactions matter.

First, **allocations during capture must go through the graph's private memory pool.** PyTorch's caching allocator knows about graph capture and routes allocations made during capture into a pool that is reserved for the graph's lifetime — that is why the static buffers a graph captures are not handed back out to unrelated code. But if you capture inside a context where the allocator's pool assumptions are violated (for instance, calling `torch.cuda.empty_cache()` between capture and replay, which can release blocks the graph still references, or capturing two graphs that share a pool and then freeing one), you can end up with a graph whose captured addresses have been reclaimed. Replay then reads reallocated memory — back to garbage. The rule: **do not `empty_cache()` a graph's memory out from under it, and keep each graph's static tensors alive for as long as the graph can replay.**

A concrete version of the first interaction: a service captured a graph at startup, then a periodic background task called `torch.cuda.empty_cache()` every few minutes to keep reserved memory low. Most of the time nothing happened, because the graph's blocks were still referenced and the allocator could not release them. But after a config reload rebuilt part of the pipeline and dropped the last Python reference to a static buffer, the next `empty_cache()` released that block, and the following replay read reallocated memory — intermittent garbage that appeared only minutes after a reload and was maddening to reproduce. The allclose gate would have caught it at deploy if it had run *after* the reload path, which is the lesson: gate correctness on the code path that actually runs in production, not only at startup. The fix was to keep an explicit reference to every static buffer for the graph's whole lifetime and to stop clearing the cache underneath live graphs.

Second, **`expandable_segments` changes how the allocator lays out memory**, and its interaction with graph pools has been a source of subtle issues across PyTorch versions. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` lets the allocator grow segments instead of pre-splitting them, which reduces fragmentation for dynamic workloads — but graph capture pins specific addresses, so the two features have historically needed care to compose. If you graph *and* set `expandable_segments`, add the allclose gate to your CI and re-run it after any PyTorch upgrade, because this is exactly the kind of interaction that a point release can shift. The full story of allocated-versus-reserved memory, fragmentation, and these config flags lives in [the CUDA caching allocator](/blog/machine-learning/performance-engineering/the-cuda-caching-allocator); for graphs, the operational takeaway is narrow: keep the graph's buffers alive, do not clear the cache underneath it, and gate correctness after any allocator config change.

## The debugging loop, step by step

When a graph misbehaves and you do not yet know which class you are in, run this loop. It is the same profile → hypothesize → fix → measure loop as the rest of the series, specialized for graphs, and it moves left to right from catching the loud errors to catching the silent ones.

![a left to right sequence from wrapping capture in try except through reading the cuda error running allclose bisecting the region re-capturing and gating in ci](/imgs/blogs/cuda-graphs-gotchas-and-debugging-6.webp)

**Step one: wrap capture in `try/except` and read the CUDA error.** This immediately separates class four/one (loud, "operation not permitted") from the silent classes. If capture throws, you have a sync or a disallowed op — grep the region for `.item()`, `.cpu()`, `print(`, `synchronize`. If capture succeeds, the bug is silent and you proceed.

**Step two: run the allclose gate on a *different* input than you captured on.** This is the universal detector. If eager and graphed disagree, you have a silent corruption; the magnitude and pattern of the diff hints at the class (a constant stale answer regardless of input points at class two; a crash or garbage only on certain shapes points at class three; identical-across-replays points at class five).

**Step three: turn on `CUDA_LAUNCH_BLOCKING=1`.** Normally kernel launches are asynchronous and errors surface at some later, unrelated API call, so the traceback lies. Blocking mode makes every launch synchronous, so an error is reported at the exact offending kernel:

```bash
# Make CUDA errors point at the real kernel instead of a later API call.
CUDA_LAUNCH_BLOCKING=1 python serve.py

# Force device-side assertions (bounds checks) to catch illegal accesses precisely.
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python serve.py
```

`CUDA_LAUNCH_BLOCKING=1` costs you all the async overlap — it will be slow — so use it only while debugging, never in production. But for a dynamic-shape illegal access, it is the difference between a traceback that points at the attention kernel and one that points at some innocent op three lines later.

**Step four: bisect the region.** If the gate fails but you cannot see why, shrink the captured region. Graph only the first half of the model; gate it. If that half passes, the bug is in the second half; if it fails, the bug is in the first. Halve again. In a handful of steps you localize the failure to one block — usually the one doing something shape-dependent, allocating, or reading RNG. This is ordinary bisection, and it works because a graph is composable: you can capture any contiguous sub-region and compare it to eager independently.

**Step five: re-capture with the fix and gate in CI.** Apply the class-specific fix — static buffers, bucketing, hoisted branch, capture-safe RNG — re-capture, and add the allclose gate as a required CI check so the bug cannot come back on the next refactor. Confirm on the trace that replay is actually one `cudaGraphLaunch` and not a silent fall-back to eager (a graph that failed to capture and got skipped will be correct but slow — the gate passes, the speed does not, so check both).

The whole loop is maybe thirty minutes the first time and five minutes once it is muscle memory. The key insight is the *order*: catch the loud errors first (cheap, unambiguous), then the silent ones with the gate (cheap, universal), then localize with bisection (a few iterations). You never stare at a graph guessing.

### Confirming the graph actually replayed

There is a seventh failure that is not on the taxonomy because it is not a *correctness* failure — it is a silent *performance* failure, and it hides right where you stop looking. If capture threw and your code caught the exception and fell back to eager, the allclose gate passes (eager matches eager) but you get none of the speed. You "shipped a graph" that is quietly running the slow path. The gate cannot catch this because the output is correct. Only the trace can.

So the last check in the loop is to confirm the replay is really one graph launch and not a fan-out of individual kernels. Profile a step and look at the CUDA API row: a real replay is a single `cudaGraphLaunch`; a fallen-back eager step is hundreds of `cudaLaunchKernel` calls with the familiar host-bound gaps. In an Nsight Systems timeline or a `torch.profiler` Chrome trace it is unmistakable:

```console
# a real graph replay — one launch, then the GPU runs uninterrupted
cudaGraphLaunch            1     0.9 ms   <- the whole step, one host call
    (912 kernels execute on the GPU with no host gaps)

# a silent fall-back to eager — the graph never captured
cudaLaunchKernel         912     6.1 ms   <- back to per-kernel launches
    (idle gaps between kernels while Python refills the queue)
```

If you see the second pattern when you expected the first, capture silently failed and your fall-back caught it. Read your capture logs (this is why step one wraps capture in `try/except` and *logs* rather than swallowing), fix the capture error, and re-verify the trace. The rule: **gate correctness with allclose, but gate speed with the trace.** A graph that is correct and slow is a graph that never captured. For the mechanics of reading these traces, the [Chrome-trace](/blog/machine-learning/performance-engineering/reading-a-chrome-trace) and [Nsight Systems](/blog/machine-learning/performance-engineering/nsight-systems-for-ai-services) siblings go lane by lane.

## The decision tree: symptom to failure to fix

Once you have run the loop a few times, the diagnosis collapses into a lookup. The tree below is the compressed version — start from what the graph did wrong and follow the branch.

![a decision tree from a graph that is wrong or will not capture through four yes no questions each ending in one canonical fix](/imgs/blogs/cuda-graphs-gotchas-and-debugging-7.webp)

Read it top-down. Did the shape vary between capture and replay? Bucket and pad — one graph per bucket. Are you allocating a new tensor each step instead of copying into the captured buffer? Static buffer plus `copy_`. Is there an `.item()`, a `print(tensor)`, or a data-dependent branch inside the region? Hoist the sync and the decision out of the graph. Is there a random op that keeps producing the same numbers? Move to capture-safe RNG. Four questions, four fixes, and the allclose gate underneath all of them as the detector that told you something was wrong in the first place.

Here is the same tree as a table you can paste into a runbook, with the diagnostic that confirms each class:

| Failure class | Symptom | Confirming diagnostic | Fix |
| --- | --- | --- | --- |
| Illegal sync | Capture throws "operation not permitted" | `try/except` at capture prints the error | Remove `.item()`/`.cpu()`/`print` from region |
| Stale pointer | Silent wrong numbers, same answer regardless of input | allclose fails; diff constant across inputs | Static buffer + `copy_` in, `clone` out |
| Dynamic shape | Garbage or `illegal memory access` on new shapes | Crash under `CUDA_LAUNCH_BLOCKING=1`; fails only on off-shape inputs | Pad to fixed, or bucket per shape |
| CPU control flow | Wrong branch always taken, no error | allclose fails when the branch should differ | Hoist branch to host, one graph per path |
| RNG frozen | Identical output across replays | `assert_rng_advances` fails | Default CUDA generator, capture-safe RNG |

## Case studies and real numbers

A few grounded results, so the numbers above are not the only ones you see. Where I cite a figure I have tried to keep it to what the primary source reports; treat the ratios as representative, not universal.

**PyTorch `mode="reduce-overhead"`.** `torch.compile(model, mode="reduce-overhead")` composes Inductor's fused kernels with CUDA graphs automatically, and the PyTorch team has reported meaningful latency reductions on small-batch inference where launch overhead dominates — the wins are largest exactly where a workload is host-bound, and negligible where it is already compute-bound. The documentation is explicit about the two constraints that come straight from this post: the mode requires static input addresses, and it warns that holding references to graph outputs across iterations can corrupt results, because the next replay overwrites the static output buffer. That warning *is* the class-two stale-pointer failure, surfaced by the framework — which is a good reminder that even the automatic path does not repeal the invariant, it just manages the static buffers for you. When `reduce-overhead` gives wrong numbers, the first suspect is a retained output reference; `clone()` it.

**The NVIDIA/PyTorch CUDA-graphs integration.** The original "Accelerating PyTorch with CUDA Graphs" work reported end-to-end training speedups in the range of roughly 1.1x to 1.5x on models bottlenecked by CPU launch overhead, and its constraints section reads like the table above: shapes and addresses must be static across replays, and control flow and disallowed syncs must stay out of the captured region. The engineering effort in that integration was almost entirely about *managing static buffers and RNG state safely* — evidence that the failure classes here are not exotic edge cases but the central difficulty of using graphs at all.

**Serving stacks with bucketed graphs.** Production LLM serving systems that use CUDA graphs (for the decode step, where the same fixed-shape kernel runs thousands of times) universally adopt the two patterns from this post: a static KV-cache and input buffers written in place, and a set of graphs captured per batch-size bucket, with a fall-back to eager for shapes outside the bucket set. The reported decode-latency improvements from graphing are consistent with the host-overhead elimination we measured on the A100 above — single-digit-millisecond per-step savings that compound over long generations. The [model-serving companion on kernel fusion, CUDA graphs, and torch.compile](/blog/machine-learning/model-serving/kernel-fusion-cuda-graphs-torch-compile) walks the serving-side integration in detail.

The through-line of all three: nobody who ships CUDA graphs at scale avoids these failures. They *manage* them, with static buffers, bucketing, and a correctness gate. The techniques in this post are not defensive extras; they are the price of admission.

## When to reach for graphs, and when not

Graphs are a sharp tool. Here is the honest guidance on when the sharpness is worth it.

**Reach for CUDA graphs when** your service is host-bound — the GPU timeline is full of idle gaps between short kernels while the CPU scrambles to launch the next op (the [30%-utilization signature](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu)), your shapes are fixed or fall into a small set of buckets, and your per-step work is a stable straight line. Small-batch inference and autoregressive decode are the canonical wins, because launch overhead is a large fraction of a short step and the step repeats identically thousands of times.

**Do not reach for CUDA graphs when** any of the following holds, because the cost will exceed the benefit:

| Situation | Why graphs hurt | Do this instead |
| --- | --- | --- |
| Already compute-bound at high occupancy | No launch overhead to remove; graphs add complexity for ~0% gain | Profile the kernel; look at fusion or a better algorithm |
| Shapes vary continuously with no natural buckets | Every shape needs its own graph or wasteful padding | `torch.compile` with dynamic shapes, or stay eager |
| The region has genuine data-dependent control flow | Cannot be captured as a straight line | Graph the fixed inner step, keep the loop on the host |
| Rapid model iteration / research code | Capture discipline and the gate are overhead you do not need yet | Ship eager; graph once the model and shapes stabilize |
| You have no correctness gate in CI | Silent corruption will reach users | Add the allclose gate *first*, then graph |

The last row is the one I want to leave you on. The single most common way CUDA graphs go wrong is not any individual failure class — it is shipping a graph with *no correctness gate at all*, so that when a refactor reintroduces a stale pointer six months later, nothing catches it until an eval regression or a user complaint. Graphs are safe to use in production precisely to the degree that you can prove, on every deploy, that the graphed output matches eager. Build the gate before you build the graph.

## Key takeaways

- **Graphs fail by corrupting, not crashing.** Four of the five failure classes produce plausible-looking wrong numbers at full speed. Your normal "wait for the exception" instinct will not save you.
- **A replay is a fixed list of launches with frozen pointer and grid arguments.** Correctness requires the live memory at each captured address to hold the data you intend, at the extent captured. Every failure is a violation of that one invariant.
- **The stale-pointer fix is the whole discipline: copy in, clone out.** `static_input.copy_(new_data)` before replay; `static_output.clone()` after. New tensors are invisible to the graph.
- **One graph, one shape.** Pad to a fixed size or bucket per shape. A different shape reads the wrong extent — garbage if you are lucky, `illegal memory access` if you are not.
- **Every fork stays on the CPU.** Data-dependent branches and `.item()` guards cannot live inside a captured region. Hoist them out; keep one graph per straight-line path.
- **RNG must be capture-safe, and the test is inverted:** consecutive replays of a random region must *differ*. Identical output across replays means the philox offset is frozen.
- **The allclose gate catches all silent classes at once.** Run one fixed input — different from the capture input — through eager and graphed, assert they agree within a loose tolerance, and put it in CI. Build it before the graph.
- **Debug in order: loud errors first (`try/except` at capture), then silent ones (the gate), then localize (bisect the region).** Use `CUDA_LAUNCH_BLOCKING=1` to make async errors point at the real kernel.

## Further reading

- [CUDA graphs from first principles](/blog/machine-learning/performance-engineering/cuda-graphs-from-first-principles) — capture versus replay and the recorded kernel DAG this whole post debugs.
- [CUDA graphs in PyTorch](/blog/machine-learning/performance-engineering/cuda-graphs-in-pytorch) — `torch.cuda.graph`, `make_graphed_callables`, and the static-I/O API these fixes build on.
- [CUDA graphs in a serving loop](/blog/machine-learning/performance-engineering/cuda-graphs-in-a-serving-loop) — batch-size bucketing and per-shape graphs applied to a real inference service.
- [The CUDA caching allocator](/blog/machine-learning/performance-engineering/the-cuda-caching-allocator) — allocated versus reserved, fragmentation, and the `expandable_segments` interaction foreshadowed above.
- [The kernel launch overhead problem](/blog/machine-learning/performance-engineering/the-kernel-launch-overhead-problem) — why host overhead is the bottleneck graphs remove, and how to measure it.
- [Why your AI service wastes CPU and GPU](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) — the four wastes and the profile-to-fix loop this post specializes.
- [The performance engineering playbook](/blog/machine-learning/performance-engineering/the-performance-engineering-playbook) — the capstone decision tree tying every fix in the series together.
- [Kernel fusion, CUDA graphs, and torch.compile for serving](/blog/machine-learning/model-serving/kernel-fusion-cuda-graphs-torch-compile) — how graphs compose with compilation in a production serving stack.
- PyTorch docs — the CUDA Graphs section of the CUDA semantics guide and the `torch.cuda.graph` / `make_graphed_callables` reference, plus the "Accelerating PyTorch with CUDA Graphs" engineering blog for the constraints and reported speedups cited above.

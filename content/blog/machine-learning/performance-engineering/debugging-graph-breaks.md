---
title: "Debugging Graph Breaks: Why torch.compile Isn't Making Your Model Faster"
date: "2026-07-14"
publishDate: "2026-07-14"
description: "You wrapped your model in torch.compile and got the same speed. This is how to find the graph breaks and recompilation storms that ate your speedup, and how to fix each one."
tags:
  [
    "performance-engineering",
    "gpu-optimization",
    "torch-compile",
    "pytorch",
    "profiling",
    "graph-breaks",
    "recompilation",
    "cuda",
    "inference",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Performance Engineering"
author: "Hiep Tran"
featured: true
readTime: 34
---

You did the thing everyone tells you to do. You wrapped your model in one line — `model = torch.compile(model)` — expected the 1.5x to 2x that the blog posts promise, ran the benchmark, and got... 12.1 ms per forward instead of 12.4 ms. A 2.4% "speedup" that is inside the noise. Or worse: the first request took 4 seconds and every request after it was somehow *slower* than eager. The compiler ran. It reported no errors. And it did essentially nothing.

This is the single most common way `torch.compile` disappoints, and it is almost never because the compiler is broken. It is because your model was silently shattered into dozens of little eager-mode islands connected by uncompiled Python — and each seam between them threw away exactly the thing you were paying the compiler for. Forty graph breaks turned one fusable graph into forty-one fragments, and the fusion and dispatch savings evaporated across every seam. Or a varying input shape made the compiler recompile the whole model on every single batch, so you paid a three-second compile tax over and over while the actual inference stayed eager.

![a fused single compiled region reaching a 1.57 times speedup next to the same model split by forty breaks reaching only 1.06 times](/imgs/blogs/debugging-graph-breaks-1.webp)

The figure above is the whole problem in one picture. On the right, one fused region: the model compiles into a single graph, kernels fuse end to end, and on an A100 you keep the 1.57x. On the left, the *same model*, same compile call, same hardware — but forty graph breaks chop it into eager islands, and the measured win collapses to 1.06x. Nothing in the API told you which one you got. `torch.compile` succeeds either way. This post is about telling the two apart, finding every seam, and closing it.

By the end you will be able to: count your graph breaks and read the exact line that caused each one; understand *mechanically* why a break costs you fusion and re-incurs dispatch overhead; recognize a recompilation storm in the logs and know which of four fixes to reach for; and run a repeatable dev-to-prod workflow that surfaces breaks as hard errors, fixes them, confirms the speedup returned, and then guards production against the shapes that would recompile. This is a debugging post. It assumes you already know roughly what the stack does (Dynamo traces, Inductor codegens, guards protect the compiled artifact); if that is fuzzy, read [what torch.compile actually does](/blog/machine-learning/performance-engineering/what-torch-compile-actually-does) first, then come back. It fits into the series' recurring loop — profile, hypothesize, fix, re-measure — that we lay out in [why your AI service wastes CPU and GPU](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu).

## What a graph break actually costs you

Before we chase down breaks, we have to be honest about *why* they hurt, because the answer decides which ones matter. A graph break is not an error. It is Dynamo's fallback when it hits Python it cannot trace: a data-dependent branch, a call into a C extension it has no rule for, a `print` of a tensor, anything with a side effect it cannot reason about. When that happens, Dynamo does three things in sequence. It finalizes the graph it was building up to that point and hands it to Inductor to compile. It runs the offending operation in plain eager Python. Then it starts tracing a *fresh* graph on the far side of the break. Your one model becomes N compiled regions plus N-1 eager seams.

That sounds harmless — "it still runs the fast part fast" — but two concrete costs hide in every seam, and both are measurable.

**Cost one: fusion cannot cross the boundary.** Inductor's biggest lever is fusion — collapsing a chain of elementwise and reduction ops into a single kernel so intermediate tensors never leave the GPU's on-chip memory. A `LayerNorm` followed by a bias-add followed by a GELU, in eager mode, is three kernels writing three full activation tensors out to HBM and reading them back. Fused, it is one kernel that reads the input once, keeps everything in registers and shared memory, and writes the output once. But Inductor can only fuse ops that live in the *same* graph. A break in the middle forces the tensor at the boundary to be fully materialized to HBM so eager Python can touch it, and it forces the far side to re-read it. Every break you insert is at least one extra HBM round trip on the boundary tensor, plus the fusion opportunities on both sides that will never happen.

We can put a number on the HBM tax. If the boundary tensor is $B$ bytes and HBM bandwidth is $BW$, the break costs you at minimum

$$t_\text{HBM} = \frac{2B}{BW}$$

for the forced write-then-read. For a batch-16, seq-128, hidden-768 bf16 activation that is $16 \times 128 \times 768 \times 2 \approx 3.1$ MB; at the A100's 2.0 TB/s that single round trip is only about 3 µs. Small — but that is *one* tensor at *one* seam. The real damage is the fusion you lose: the three-kernel `LayerNorm`+bias+GELU that should have been one kernel now stays three, and if there are forty seams in a model with hundreds of fusable ops, the fused-kernel count that should have dropped from 430 to 95 instead stays near 360.

**Cost two: dispatch overhead comes back at every seam.** The reason `torch.compile` helps host-bound models at all is that it replaces hundreds of individual Python-level operator dispatches — each one a trip through the dispatcher, autograd, and the CUDA launch machinery — with a handful of fused kernel launches. Every graph break re-inserts a Dynamo boundary: Python re-enters the interpreter, Dynamo checks the guards on the *next* region to decide whether its cached compiled artifact is still valid, and only then launches. Guard evaluation is cheap per region but not free, and the eager op at the seam pays the full un-fused dispatch cost. If your model was host-bound to begin with — the classic "GPU util shows 100% but the timeline is full of gaps" service — reintroducing forty dispatch boundaries can erase most of what compilation bought you.

Put those together into a back-of-the-envelope model. Say the fully fused graph runs the GPU work in time $T_g$, and each break adds a fixed overhead $c$ (materialization + lost fusion + re-dispatch on that seam). With $B$ breaks:

$$T_\text{broken} \approx T_g + B \cdot c$$

This is an Amdahl argument in disguise. Compilation can only speed up the fraction of work $p$ that lands inside fused graphs; the eager seams are the un-accelerated $(1-p)$. If breaks push a third of your op-time back into eager islands, your ceiling drops from a 1.57x whole-model win to

$$S = \frac{1}{(1-p) + p/s}$$

with $p \approx 0.67$ and a per-region kernel speedup $s \approx 2$, giving $S \approx 1.2$ at best — and in practice the extra HBM trips and re-dispatch drag it down to the 1.06x we measured. That is the arithmetic behind the left column of figure 1. The lesson is blunt: **a few breaks in cold code cost nothing; a break inside the hot repeated block costs almost everything.** Location matters more than count.

#### Worked example: the sampling loop that ate the speedup

A team compiled a small GPT-style decoder for a generation service on an A100 80GB SXM. Eager decode was host-bound — the classic empty-timeline signature — so `torch.compile` should have been a slam dunk. It gave them 1.04x. The culprit was four lines in the per-step sampling logic:

```python
def sample_step(logits):
    probs = torch.softmax(logits[:, -1, :], dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    if next_token.item() == eos_token_id:   # <-- graph break every step
        return None
    return next_token
```

The `next_token.item()` pulls a scalar off the GPU to compare it in Python. Dynamo cannot trace a branch whose condition is a device value it does not have, so it breaks — *inside the loop that runs once per generated token*. For a 256-token generation, that is 256 graph breaks per request, each forcing a device-to-host sync (which also serializes the GPU behind the CPU). The fix kept the comparison on the GPU and moved the host check out of the traced region:

```python
def sample_step(logits):
    probs = torch.softmax(logits[:, -1, :], dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    is_eos = (next_token == eos_token_id)     # stays a GPU tensor, no break
    return next_token, is_eos
# the .item() / early-return lives in the OUTER python loop, outside compile
```

The compiled region no longer contained a data-dependent `.item()`, so the per-step break disappeared. Measured on the A100: p50 per-step latency went from 6.2 ms (eager) and 5.9 ms (broken compile) to 3.4 ms (fixed compile) — the 1.04x became 1.82x once the seam was gone. Same compile call. The only change was where the host-value comparison lived.

## Watching a single break in slow motion

To fix breaks reliably you need to see what one *does*, not just that it happened. The figure below traces a single `.item()` break through Dynamo.

![a compiled region A that forks into a finalized graph and a full dispatch eager island which merge into a re-entered region B with fusion broken across the seam](/imgs/blogs/debugging-graph-breaks-2.webp)

Follow it left to right. Region A is trucking along — eight ops traced and fused into one graph. Then it hits the `.item()` call, which needs a host value Dynamo does not have. That single call forks the flow: on one path, graph A is finalized, compiled, launched, and its guards recorded; on the other, the `.item()` itself runs as a full eager dispatch, syncing the GPU to hand Python the scalar. The two paths merge back when Dynamo re-enters and begins tracing region B from scratch, re-installing a fresh set of guards. The critical detail is the node at the bottom: the fusion that *would have spanned* the ops in A and the ops in B is now impossible, so you eat an extra HBM trip and lose the fused kernel — and for a model with forty such calls you end up with forty-one regions per forward pass. This is the mechanism. Everything else in this post is either finding these forks or removing the thing that causes them.

Two properties of the mechanism drive the whole debugging strategy. First, **the break is always caused by a specific Python operation on a specific line** — there is no such thing as a break "somewhere in the model." That is why the tools below can name the line. Second, **a break early in cold setup code is free; the same break inside a per-layer or per-token loop multiplies by the loop count.** When you triage, sort by hotness, not by count.

## Finding your breaks: three tools, coarse to fine

You have three instruments, and they form a natural progression from "how bad is it" to "which exact line." Use them in that order.

### torch._dynamo.explain — the break census

Start with a census. `torch._dynamo.explain` runs Dynamo over your function without requiring a full compile-and-run, and hands back a structured report: how many graphs it produced, how many breaks, and the reason for each.

```python
import torch
import torch._dynamo

model = build_encoder().cuda().eval()
example = torch.randn(16, 128, 768, device="cuda", dtype=torch.bfloat16)

explanation = torch._dynamo.explain(model)(example)
print(explanation)
```

The output tells you immediately whether you have a fragmentation problem:

```console
Graph Count: 41
Graph Break Count: 40
Op Count: 512
Break Reasons:
  Break Reason 1:
    Reason: Tensor.item
    User Stack:
      <FrameSummary file model.py, line 88, in forward>
  Break Reason 2:
    Reason: builtin: print [<class 'torch._dynamo.variables.tensor.TensorVariable'>]
    User Stack:
      <FrameSummary file model.py, line 94, in forward>
  Break Reason 3:
    Reason: call_function BuiltinVariable(len) [TensorVariable] {}
    User Stack:
      <FrameSummary file model.py, line 101, in _route>
  ... (37 more)
Ops per Graph: [14, 12, 13, ...]
```

Graph Count 41, Break Count 40: the model is in forty-one pieces. The Break Reasons list is your work queue — each entry names the reason (`Tensor.item`, `print`, `len` on a tensor) and the source line. Note the "Ops per Graph" list: lots of tiny graphs with a dozen ops each is the fragmentation signature. A healthy compile of this model would print `Graph Count: 1, Graph Break Count: 0`.

`explain` is the right first move because it is non-fatal — it surveys the whole model in one pass and gives you the full list, whereas the next tool stops at the first break.

### TORCH_LOGS — watching breaks and recompiles at runtime

`explain` shows you a static census. To see what actually happens *during a real run* — including recompiles, which `explain` does not surface — turn on the logging subsystem. It is an environment variable, no code changes:

```bash
TORCH_LOGS="graph_breaks,recompiles" python serve.py
```

Graph breaks print as they occur, with the reason and the user code line:

```log
[rank0] torch._dynamo hit a graph break:
  Graph break in user code at model.py:88
  Reason: Tensor.item() called on a tensor with unknown value; this forces
          a graph break because the result is data-dependent.
  User code traceback:
    File "model.py", line 88, in forward
      if attention_mask.sum().item() > 0:
```

And this is also where you catch the *other* failure mode — recompiles — which no break count will reveal:

```log
[rank0] torch._dynamo.recompile: recompiling forward because:
  triggered by the following guard failure(s):
  - tensor 'L['x']' size mismatch at index 1. expected 128, actual 256
```

That last line is the tell for a recompilation storm, which we come back to in its own section. For now, know that `TORCH_LOGS` is how you watch both failure modes in production traffic without instrumenting the code. You can add `inductor` to the list (`TORCH_LOGS="graph_breaks,recompiles,inductor"`) when you want to see the codegen too, but for break-hunting the two above are what you want.

### fullgraph=True — turn the next break into a hard error

The census and the logs are diagnostic. When you are ready to *fix*, flip the switch that makes the compiler refuse to break at all:

```python
model = torch.compile(model, fullgraph=True)
out = model(example)   # raises on the FIRST break, with the exact line
```

With `fullgraph=True`, the very first graph break becomes an exception instead of a silent fallback:

```console
torch._dynamo.exc.Unsupported: Tensor.item

from user code:
   File "model.py", line 88, in forward
     if attention_mask.sum().item() > 0:

Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information
```

This is the single most useful flag in the whole workflow, and the reason is psychological as much as technical: a silent fallback is easy to ignore, but an exception with a line number is a bug you *have* to fix. The routine is: compile with `fullgraph=True` in development, run, get the error, fix that line, run again, get the next error, repeat until it runs clean. Each fix removes one seam. When it finally runs without raising, you have zero breaks and one graph — guaranteed, not hoped for.

A comparison of the three, because using the wrong one wastes time:

| Tool | What it gives you | Fatal? | Sees recompiles? | Best for |
|---|---|---|---|---|
| `torch._dynamo.explain` | Full break census, all reasons + lines, in one pass | No | No | First look: how fragmented am I? |
| `TORCH_LOGS="graph_breaks,recompiles"` | Live breaks and recompiles as they happen | No | **Yes** | Production traffic; catching recompile storms |
| `torch.compile(fullgraph=True)` | Hard error on the first break, exact line | Yes | No | Fixing: forces you to close every seam |

Before we start fixing, one more triage step. A compiled model that is no faster is not always a break problem — and reaching for `fullgraph=True` when the real issue is a compute-bound kernel is a waste of an afternoon.

## Is it even a break problem? A 30-second triage

![a decision tree splitting a compiled model with no speedup into three branches for graph breaks recompiling and a compute bound kernel each with its own tool and fix](/imgs/blogs/debugging-graph-breaks-3.webp)

Split the symptom before you touch code. "Compiled but no faster" is one of exactly three failures, and each has a different tool and a different fix — the tree above is the map.

1. **Too many graph breaks.** Signature: `explain` reports a high Graph Break Count; the timeline still looks like eager mode with all its little kernels and gaps. Tool: `torch._dynamo.explain`, then `fullgraph=True`. Fix: close the seams (the next section). This is the most common case by far.

2. **Constant recompiles.** Signature: the *first* request to each new shape is slow (seconds), steady-state is fine until a new shape arrives, and `TORCH_LOGS=recompiles` is chattering. Tool: the recompiles log. Fix: dynamic shapes and bucketing (two sections down). Sneaky because break count can be zero — the graph is perfect, you are just rebuilding it constantly.

3. **Genuinely compute-bound.** Signature: `explain` shows one graph and zero breaks, the recompiles log is quiet, and Nsight Compute shows the dominant kernel already at high occupancy and near the Speed-of-Light memory or compute roofline. Fix: none from `torch.compile` — the compiler already did its job; the kernel is simply near the hardware limit. Reaching for compile tricks here is chasing a speedup that physics will not give you. This is where you cross-link out to real kernel work like [kernel fusion, CUDA graphs, and torch.compile in serving](/blog/machine-learning/model-serving/kernel-fusion-cuda-graphs-torch-compile) or accept the number.

The thirty-second version: run `explain` (breaks?), glance at the recompiles log (churning?), and if both are clean, profile the hot kernel before blaming the compiler. Only branch 1 and branch 2 are "torch.compile isn't helping" problems; branch 3 is "torch.compile already helped and this is as fast as it gets."

## The five causes of a graph break, and the fix for each

Almost every break you will ever see traces to one of five patterns. The matrix below is the cheat sheet; the prose after it explains the mechanism and the fix for each row so the table is not just a lookup.

![a table of five graph break causes paired with fixes covering data dependent flow unsupported ops printing tensors device to host copies and dynamic python](/imgs/blogs/debugging-graph-breaks-4.webp)

**Data-dependent control flow.** This is the big one: `.item()`, `if tensor:`, `bool(tensor)`, `tensor.tolist()` used in a branch, `while some_tensor_condition:`. Any Python control-flow decision made on a value that lives on the GPU forces Dynamo to break, because it would have to know the value to trace which branch runs, and the value is on the device. Three fixes, in order of preference. Best: **make the operation branchless** — replace `if mask.any(): x = x + bias` with an unconditional masked update `x = torch.where(mask, x + bias, x)`, so there is no branch to trace. Next: **use the structured control-flow ops** `torch.cond` and `torch.while_loop`, which are traceable primitives that keep the condition on the GPU:

```python
# breaks: python branch on a device value
if logits.argmax(-1).item() == stop_id:
    ...

# traceable: keep the decision on-device with torch.cond
def true_fn(x): return x * 0.0
def false_fn(x): return x
pred = (logits.argmax(-1) == stop_id).squeeze()
out = torch.cond(pred, true_fn, false_fn, (hidden,))
```

Last resort: **move the host check outside the compiled region** entirely, as in the sampling-loop worked example — compile the per-step tensor math, do the `.item()` early-stop in the outer Python loop.

**Unsupported op / C-extension call.** A custom op implemented in a C extension, or a third-party function Dynamo has no tracing rule for, breaks because there is nothing to trace *through*. The clean fix is to **register the op** with `torch.library` (formerly a custom-op registration) so Dynamo treats it as an opaque but traceable node with a fake/meta implementation for shape propagation. When that is too much work for a cold-path op, **move the call out of the hot path** so the break lands in setup code where it costs nothing.

**Printing or logging a tensor.** `print(x)`, `logging.info(f"{x}")`, or a debug `assert x.mean() > 0` all break, because reading a tensor's value forces a device sync and a side effect Dynamo will not fold into a graph. Fix: **drop tensor prints from the hot path.** If you must log, log a scalar you already have on the host, or gate the print behind a flag that is off in the compiled path. This is the easiest category to fix and the easiest to leave in by accident — a stray debug `print` from three weeks ago is a classic silent-speedup-killer.

**`.cpu()` / `.numpy()` / `.tolist()` — device-to-host copies.** Any explicit move off the GPU breaks the graph and, worse, injects a synchronizing D2H copy. Fix: **keep the tensor on the device** and defer the single unavoidable copy to the very end of the forward, outside the compiled region. Do not `.cpu()` an intermediate to do some NumPy math and then push it back — either express the math in torch on-device, or restructure so the boundary crossing happens once at the end.

**Dynamic Python structure.** `*args`/`**kwargs` splat with varying arity, closures over changing free variables, a Python `dict` whose keys vary at runtime, iterating a list whose length is data-dependent. Dynamo needs a static structure to trace. Fix: **flatten the inputs into a fixed signature** — pass a fixed set of named tensors instead of a variable-length container, and hoist any Python-structural logic out of the traced function.

The unifying principle across all five: **a graph break is Python touching a Python-level value where Dynamo needs to stay in tensor-land.** Every fix either keeps the value on the GPU (`torch.where`, `torch.cond`, `is_eos` tensor) or moves the untraceable Python work out of the hot compiled region. Keep that principle and you can fix a break you have never seen before.

#### Worked example: closing forty breaks on an encoder

Back to the model from figure 1 — a Transformer encoder inference service, batch 16, seq 128, hidden 768, bf16, A100 80GB SXM. `explain` reported Graph Count 41, Break Count 40. The forty broke down into three real bugs, each repeated across the twelve layers:

- A `print(f"layer {i} norm: {x.norm().item()}")` left in from debugging — twelve breaks (one per layer). Deleted. Twelve gone.
- A routing helper that did `if x.abs().max().item() > threshold: x = clamp(x)` — twelve breaks. Rewritten as `x = torch.where(x.abs().max() > threshold, clamp(x), x)`... except `max()` is a reduction so we used `x = torch.clamp(x, -threshold, threshold)` unconditionally, which was the intent anyway. Twelve gone.
- An attention-mask check `if attention_mask.sum().item() > 0:` in each layer — twelve breaks. The mask is always non-empty in this service, so the guard was dead code; deleted. Twelve gone. (The remaining four were a `.tolist()` in a position-encoding helper, hoisted out to run once at model build.)

After the three fixes, `explain` printed `Graph Count: 1, Graph Break Count: 0`. The measured result on the A100:

| Configuration | p50 (ms) | Kernel count | Compiled regions | Speedup vs eager |
|---|---|---|---|---|
| Eager | 12.4 | 431 | — | 1.00x |
| Compiled, 40 breaks | 11.7 | 358 | 41 | 1.06x |
| Compiled, `fullgraph=True`, 0 breaks | 7.9 | 96 | 1 | **1.57x** |

The kernel count is the smoking gun: 431 eager kernels barely dropped to 358 with forty breaks (fusion could only happen *within* each tiny region), then collapsed to 96 once the whole model was one graph and Inductor could fuse across what used to be seams. That 431→96 is the fusion you were paying for; the breaks were preventing it. Note the broken-compile row bought almost nothing (1.06x) despite "compiling successfully" — that is figure 1's left column, measured.

## Recompilation storms: when the graph is perfect and you are still slow

Now the sneakier failure. Suppose you fixed every break — `explain` says one graph, zero breaks — and the service *still* has ugly latency, specifically a p99 that periodically spikes into the *seconds* while p50 is fine. That is not a break. That is a recompilation storm.

Here is the mechanism. When Dynamo compiles a graph, it does not compile "the model" — it compiles a specialization guarded by a set of runtime assumptions: this input has this shape, this dtype, this device, this stride. Those assumptions are the **guards**. On every call, Dynamo evaluates the guards; if they all pass, it dispatches the cached compiled artifact (fast). If a guard fails — say the sequence length is now 256 when the compiled graph assumed 128 — Dynamo throws away nothing but *recompiles a new specialization* for the new shape, which takes seconds, and caches that one too. By default, shapes are treated as **static**: the very first shape it sees gets baked in as a constant, and any different shape fails the guard and triggers a fresh multi-second compile.

For a service whose inputs are always the same shape, this is fine — one compile at warmup, then pure cache hits forever. For a service whose input shape *varies per request* — variable sequence lengths, variable batch sizes from dynamic batching, variable image resolutions — it is a disaster. The figure walks through it.

![a timeline showing a service recompiling on each new input shape at about three and a half seconds each until the cache limit of eight is hit on the ninth shape then silently falling back to eager](/imgs/blogs/debugging-graph-breaks-5.webp)

Shape A (seq 128) arrives: compile, 3.5 s. Shape B (seq 256): guard fails, recompile, 3.4 s. Shape C (seq 384): recompile again, 3.6 s. Each distinct shape you have never seen pays the full compile tax as a latency spike on the unlucky request that first sees it. By the time eight distinct shapes have gone through, you have paid eight multi-second stalls. Then the ninth shape hits a limit you probably did not know existed — and this is the part that turns a bad situation into a silent one.

### The cache_size_limit trap

Dynamo will not recompile the same function forever. There is a config knob, `torch._dynamo.config.cache_size_limit`, that defaults to **8**. Once a single guarded function (frame) has accumulated more than that many compiled specializations, Dynamo gives up on compiling it and **silently falls back to running it in eager mode — permanently, for the rest of the process.** You get one warning in the logs and then nothing; the service just quietly runs uncompiled from then on.

```log
[rank0] torch._dynamo hit config.cache_size_limit (8)
   function: 'forward' (model.py:52)
   last reason: tensor 'L['input_ids']' size mismatch at index 1. expected 384, actual 512
   To log all recompilation reasons, use TORCH_LOGS="recompiles".
```

This is the trap in the right half of figure 5. The engineer sees the first eight requests to new shapes stall, assumes "compile warmup, it'll settle down," and it *appears* to settle — because after the ninth shape it stops recompiling. But it stopped by *giving up*: every request now runs eager, so the p99 that was spiking to 3,500 ms on recompiles is now a steady, un-accelerated eager latency, and the 1.57x you fixed all those breaks to earn is gone. The service is slower than it looks like it should be, and nothing is erroring. You only find it by seeing that one `cache_size_limit` warning scroll by, or by noticing `explain`/`nvidia-smi` disagreeing with your expectation.

We can bound the storm's cost. If you see $N$ distinct shapes and each recompile costs $t_c$, the total wasted compile time is $N \cdot t_c$ — but only up to the cap. Beyond `cache_size_limit = L`, you stop paying compile time and start paying the eager penalty on *every* request forever:

$$t_\text{wasted} = \min(N, L) \cdot t_c \;+\; (\text{requests after cap}) \cdot (t_\text{eager} - t_\text{compiled})$$

The first term is bounded and one-time; the second term is unbounded and grows with traffic. That is why the fallback is worse than the storm: the storm ends, the fallback does not.

#### Worked example: twenty shapes and a silent fallback

A summarization service on an A100 accepted arbitrary input lengths and padded to the next multiple of 64, so it saw sequence lengths of 128, 192, 256, 320, ... up to 20 distinct buckets in production traffic. It was compiled with default settings, zero graph breaks — a clean graph. Observed behavior over the first few minutes of traffic:

- Requests 1 through 8 (first sighting of each of 8 shapes): p99 spiked to 3.3–3.7 s each, one spike per new shape.
- Request that first saw the 9th distinct shape: the `cache_size_limit (8)` warning fired.
- Every request after that: ran eager. Steady-state p50 climbed from the expected 7.9 ms (compiled) back to 12.4 ms (eager), and stayed there.

The team's dashboard showed p99 "recovering" after the initial spikes and declared victory — while the service quietly ran 1.57x slower than it should for the rest of the deployment. The fix is the next section. The diagnosis was one grep for `cache_size_limit` in the logs.

## Four ways to stop the storm

You have four levers, and the right one depends entirely on *how many distinct shapes you actually see* and how latency-critical the path is. The matrix lays out the trade-off.

![a table of four recompilation fixes dynamic equals true mark dynamic bucket and pad and raising the cache limit each paired with when to use it](/imgs/blogs/debugging-graph-breaks-6.webp)

**`dynamic=True` — one symbolic graph for all shapes.** Tell the compiler up front to treat shapes as symbolic rather than baking them in:

```python
model = torch.compile(model, dynamic=True)
```

Dynamo compiles a single graph parameterized by symbolic shape variables (`s0`, `s1`, ...) with guards that only assert relationships (`s0 >= 0`), not concrete values. One compile, no recompiles across shapes. The cost: the symbolic graph is slightly less specialized, so per-request latency is a touch higher than a perfectly static-shape graph would be, and some kernels that need a concrete size still specialize. For a service with genuinely unpredictable shapes, this is the default answer.

**`torch._dynamo.mark_dynamic` — make one dimension symbolic, keep the rest static.** When only *one* dimension varies (sequence length varies, but batch, hidden, and heads are fixed), marking just that dimension dynamic gives you most of the static-graph speed with no recompiles on the varying axis:

```python
import torch._dynamo

x = torch.randint(0, vocab, (16, seq_len), device="cuda")
torch._dynamo.mark_dynamic(x, 1)      # dim 1 (sequence) is symbolic
model = torch.compile(model)          # leave dynamic="auto" (the default)
out = model(x)                        # recompiles on batch change, not seq change
```

This is more surgical than blanket `dynamic=True`: the compiler specializes everything it can and only generalizes the one axis you told it to. Use it when you *know* which dimension is the troublemaker.

**Bucket and pad — a fixed set of compiled shapes.** When you have a few hot sizes and latency matters more than memory, pad every request up to the nearest of a small set of buckets (say seq 128, 256, 512) and let the compiler produce one static-shape graph per bucket. You pay compile time once per bucket at warmup and then get maximally-specialized static graphs forever, with no runtime symbolic overhead. The cost is wasted compute on the padding and more warmup compiles. This is what production LLM serving stacks do; it composes with continuous batching (see [continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention) for how the batching layer picks buckets).

**Raise `cache_size_limit` — last resort.** If your shape set is genuinely finite and small-ish (say a dozen), and you do not want dynamic or bucketing, you can lift the cap so Dynamo stops falling back to eager:

```python
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64   # default is 8
```

But understand what this does and does not fix. It stops the *silent eager fallback* — good, no more mystery slowdown. It does **not** stop the recompiles; you still pay a compile stall on every new shape, you just allow more of them to be cached. If your shape set is actually unbounded, raising the limit only delays the pain and burns memory on dozens of cached graphs. Reach for it only when you know the shape set is small and static and you have ruled out the other three.

| Fix | Recompiles? | Per-request speed | Warmup cost | When to use |
|---|---|---|---|---|
| `dynamic=True` | None (one graph) | Good | One compile | Shapes vary a lot, any dimension |
| `mark_dynamic(dim)` | Only on other dims | Very good | One compile | Exactly one dimension varies |
| Bucket + pad | None (finite set) | Best (static graphs) | N compiles | Few hot sizes, latency-critical |
| Raise `cache_size_limit` | **Still recompiles** | Good once warm | N compiles | Small finite shape set, last resort |

Which one for the twenty-shape summarization service above? The shape set was effectively unbounded (arbitrary input lengths), and only the sequence dimension varied. `mark_dynamic` on the sequence axis was the fit: one compile, recompiles only if batch changed, and it recovered steady-state p50 to 8.5 ms — slightly above the 7.9 ms a perfectly static graph would hit, but with *zero* recompiles and no fallback. Bucketing would have shaved that last 0.6 ms at the cost of managing bucket sizes and padding waste; for this service the simplicity of `mark_dynamic` won.

## Measuring this honestly

Every number in this post came from a harness, and if you measure carelessly you will fool yourself in both directions — declaring a win that is warmup noise, or missing a regression that hides behind the first-call compile cost. The compile-specific traps:

**The first call includes compile time.** `torch.compile` is lazy: it compiles on the first call to each shape. If you time the first call, you are timing minutes-of-your-life compile latency, not inference. Always warm up — call the compiled model several times on the exact shape you will measure, *before* timing — and if shapes vary, warm up every shape you care about.

**Synchronize before you read the clock.** GPU work is asynchronous; the Python call returns before the kernels finish. Time with CUDA events (which record on the stream) or call `torch.cuda.synchronize()` before stopping a wall-clock timer, or every "speedup" you see is just the CPU racing ahead of the GPU.

**Separate compile time from runtime.** Report them as two numbers. A model that compiles in 40 s but then runs 1.57x faster is a great deal for a long-lived service and a terrible one for a script that runs once. Say which regime you are in.

```python
import torch

model = torch.compile(model, fullgraph=True)
x = torch.randn(16, 128, 768, device="cuda", dtype=torch.bfloat16)

# 1. warm up: triggers compile, then a few steady-state iters
for _ in range(10):
    model(x)
torch.cuda.synchronize()

# 2. time steady state with CUDA events
start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
iters = 100
start.record()
for _ in range(iters):
    model(x)
end.record()
torch.cuda.synchronize()
print(f"p50 per-forward: {start.elapsed_time(end) / iters:.2f} ms")
```

And crucially: **re-run `explain` after every fix to confirm the break is actually gone.** Do not assume a code change closed a seam — verify Graph Break Count dropped. The whole loop is only trustworthy if you re-measure. For deeper trace-reading of the compiled kernels — confirming fusion actually happened, not just that breaks disappeared — that is a companion skill covered in [profiling compiled code](/blog/machine-learning/performance-engineering/profiling-compiled-code).

## The workflow: dev to prod

Putting it all together into a repeatable procedure. The steps below are the reliable order — surface breaks as hard errors in dev, fix each named line, confirm the speedup came back, *then* guard production against the shapes that would recompile.

![a six step vertical workflow from compiling with fullgraph in dev through reading the named error and fixing each break to confirming the speedup and enabling dynamic shapes in production](/imgs/blogs/debugging-graph-breaks-7.webp)

1. **Compile in dev with `fullgraph=True`.** This converts every silent break into a hard error you cannot ignore. Do this in a dev run on a representative input, not in production.
2. **Read the error — it names the line.** The `Unsupported` exception points at the exact source line. No guessing.
3. **Fix the break using the cause table.** Match the reason to one of the five patterns and apply its fix — keep the value on the GPU, or move the Python work out of the hot region.
4. **Re-run `torch._dynamo.explain` to confirm.** You want to see `Graph Count: 1, Graph Break Count: 0`. Verify, do not assume.
5. **Confirm the speedup returned.** Benchmark against eager with the honest harness. On the encoder that was the 1.06x → 1.57x jump. If the number did not move, you were in branch 3 of the triage tree (compute-bound) and compile was never going to help.
6. **In production, switch on dynamic shapes and bucket.** Drop `fullgraph=True` (or keep it if you are confident) and add `dynamic=True` / `mark_dynamic` / bucketing so the shape variability of real traffic does not trigger the recompilation storm. Set `cache_size_limit` appropriately and watch the recompiles log.

The reason `fullgraph=True` belongs in dev and dynamic shapes belong in prod is that they solve opposite problems. `fullgraph=True` is *strict* — it refuses to run if anything would break, which is exactly what you want while hunting seams, and exactly what you do *not* want in production where an unexpected input should degrade gracefully rather than crash. Dynamic shapes are *permissive* — they handle the messy variety of real traffic without recompiling. Dev is where you want maximum strictness; prod is where you want maximum robustness.

## Case studies and real numbers

A few results from the field and the docs, so the numbers above are anchored to something real.

**PyTorch's own `reduce-overhead` benchmarks.** The `torch.compile` tutorial and the PyTorch team's blog posts report that on a clean graph (no breaks), typical inference speedups on an A100 land in the 1.3x–2.2x range for Transformer and vision models, with the win coming almost entirely from fusion and reduced dispatch. The corollary the docs are explicit about: those numbers assume `fullgraph=True` compiles cleanly. A model that graph-breaks does not get them — which is exactly the gap between figure 1's two columns. Treat the specific 1.57x here as representative of a mid-size encoder, not a universal constant; your model's number depends on how host-bound it was and how much fusion Inductor finds.

**The HuggingFace `generate` graph-break saga.** Text generation was for a long time a canonical graph-break trap, because the decode loop is full of data-dependent stopping criteria (`if all sequences hit EOS: break`) and Python-side KV-cache bookkeeping — every one a break inside the hottest loop in the model. The fix that made compiled generation actually fast was the same principle as our sampling worked example: keep the stopping condition as a GPU tensor, compile the per-step forward, and run the host-side loop control outside the traced region. It is the clearest real-world instance of "location matters more than count" — a *single* break, but inside the per-token loop, so it multiplied by the generation length.

**The `cache_size_limit` fallback in serving.** Multiple production write-ups of LLM and vision serving describe the exact silent-fallback failure in figure 5: a service compiled with default settings, variable input shapes, and a mysterious steady-state slowdown that turned out to be Dynamo hitting the cache limit and running eager. The resolution in every case was one of the four fixes above — most often `mark_dynamic` on the sequence axis or bucketing to a fixed set of lengths. If you take one operational habit from this post, make it *grep your logs for `cache_size_limit`*; it is the difference between a service running at its compiled speed and one silently running at eager speed. This composes with the broader recompilation deep-dive in [the torch.compile recompilation storm](/blog/machine-learning/performance-engineering/the-torch-compile-recompilation-storm).

## When to reach for this (and when not to)

`torch.compile` is not free and not always the answer. Be honest about when it is worth the debugging effort.

- **Reach for break-hunting when** `explain` shows a high break count *and* the model was host-bound (empty-timeline signature) — that is the case where closing seams reliably pays off, because fusion and dispatch reduction are exactly what a host-bound model needs.
- **Reach for dynamic/bucketing when** your break count is already zero but the recompiles log is chattering or you see the `cache_size_limit` warning. Fixing breaks here does nothing; the graph is fine, the shapes are the problem.
- **Do not chase breaks when** `explain` shows one clean graph and Nsight Compute says the dominant kernel is already near the roofline. You are compute-bound; the compiler already won. Spend the afternoon on a better kernel or a bigger batch, not on a speedup that does not exist.
- **Do not `torch.compile` at all when** the model runs once (a one-shot script), because you will pay tens of seconds of compile time to save milliseconds of runtime — a net loss. Compile is for long-lived services and training loops that amortize the compile cost over many steps.
- **Do not deploy `fullgraph=True` to production** unless you are certain every possible input traces cleanly — a break on an unexpected input becomes a hard crash for a real user instead of a graceful eager fallback. Use it to *find* breaks in dev; use graceful compile in prod.

## Key takeaways

- A graph break is not an error — it is Dynamo splitting your model into a compiled region, an eager op, and a fresh compiled region. `torch.compile` "succeeding" tells you nothing about whether you got the speedup.
- A break costs you two concrete things: fusion cannot cross the seam (extra HBM round trips, more kernels), and dispatch overhead comes back at every boundary. Location beats count — a break in the per-token loop multiplies by the sequence length.
- Find breaks coarse-to-fine: `torch._dynamo.explain` for the census, `TORCH_LOGS="graph_breaks,recompiles"` for live traffic, `torch.compile(fullgraph=True)` to turn the next break into a hard error naming the exact line.
- Triage first: "compiled but no faster" is breaks, recompiles, or genuinely compute-bound — three different tools, three different fixes. Do not fix breaks on a compute-bound kernel.
- Almost every break is one of five causes, and every fix follows one principle: keep the value on the GPU (`torch.where`, `torch.cond`), or move the untraceable Python out of the hot region.
- A recompilation storm has zero breaks and a perfect graph — it is guard failures on varying shapes triggering seconds-long recompiles. `TORCH_LOGS=recompiles` is how you see it.
- The `cache_size_limit` (default 8) trap is worse than the storm: past the cap, Dynamo silently falls back to eager *forever*, and your service runs uncompiled with no error. Grep your logs for that one warning.
- Stop the storm with `dynamic=True` (shapes vary widely), `mark_dynamic` (one dimension varies), bucket-and-pad (few hot sizes, latency-critical), or — last resort, small finite shape set — raising the cache limit.
- Measure honestly: warm up to exclude compile time, synchronize before timing, report compile time and runtime separately, and re-run `explain` after every fix to confirm the break actually closed.
- The workflow: `fullgraph=True` in dev to surface every break → fix each named line → confirm the speedup returned → dynamic shapes and bucketing in prod. Strict in dev, robust in prod.

## Further reading

- [what torch.compile actually does](/blog/machine-learning/performance-engineering/what-torch-compile-actually-does) — the Dynamo/guards/Inductor stack this post assumes.
- [profiling compiled code](/blog/machine-learning/performance-engineering/profiling-compiled-code) — reading a trace of compiled kernels to confirm fusion actually happened.
- [compile plus CUDA graphs to reduce overhead](/blog/machine-learning/performance-engineering/compile-plus-cuda-graphs-reduce-overhead) — how `mode="reduce-overhead"` composes compile with CUDA graphs once your graph is clean.
- [the torch.compile recompilation storm](/blog/machine-learning/performance-engineering/the-torch-compile-recompilation-storm) — the full war-story deep-dive on recompiles and dynamic shapes.
- [why your AI service wastes CPU and GPU](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) — the four wastes and the profile → hypothesize → fix → measure loop this fits into.
- [the performance engineering playbook](/blog/machine-learning/performance-engineering/the-performance-engineering-playbook) — the capstone decision tree tying every fix in the series together.
- [kernel fusion, CUDA graphs, and torch.compile in serving](/blog/machine-learning/model-serving/kernel-fusion-cuda-graphs-torch-compile) — how these fixes land in a production serving stack.
- PyTorch docs: the `torch.compile` tutorial, the `torch._dynamo` troubleshooting guide, and the dynamic-shapes documentation are the primary sources for the flags and behaviors above.

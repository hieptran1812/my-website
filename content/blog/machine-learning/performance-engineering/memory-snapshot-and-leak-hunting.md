---
title: "Memory snapshot and leak hunting: finding the tensor that never frees"
date: "2026-07-14"
publishDate: "2026-07-14"
description: "A tool-first guide to torch.cuda.memory._record_memory_history and the memory_viz viewer — how to see the allocation timeline, click any live block to its stack trace, and find the exact line of code that leaked the tensor that OOMs your service."
tags:
  [
    "performance-engineering",
    "gpu-optimization",
    "memory",
    "memory-leak",
    "profiling",
    "pytorch",
    "cuda",
    "inference",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Performance Engineering"
author: "Hiep Tran"
featured: true
readTime: 41
---

There is a particular kind of page that ruins a week. The training job has been running fine for hours. Loss is going down, the dashboards are green, everyone has gone home. Then at 03:14 the job dies with `CUDA out of memory. Tried to allocate 2.00 GiB`. You restart it, it dies again five hours later. You restart it, it dies again. The GPU has 80 GB, your model needs maybe 40, and yet every few hours the number creeps up until it hits the ceiling and the whole thing falls over. `nvidia-smi` confirms the number is climbing — 42 GB, then 51, then 63 — but it will not tell you *why*. It shows you the symptom with the confidence of a thermometer and the diagnostic value of one.

This is a leak, and a leak is the most frustrating waste in the whole performance catalogue because it is invisible in exactly the tool everyone reaches for first. `nvidia-smi` and `torch.cuda.memory_allocated()` are counters. They tell you memory went up. They do not tell you which of the ten thousand tensors your code allocated last second is the one that should have been freed and was not. You can stare at `62914560000` bytes for an hour and learn nothing. The bytes do not carry a return address.

The memory snapshot does. `torch.cuda.memory._record_memory_history()` turns on a recorder that logs every single allocation and every single free, and — this is the part that changes your life — captures the Python stack trace at the moment of each one. Dump that history to a file, drag the file onto [pytorch.org/memory_viz](https://pytorch.org/memory_viz), and you get an interactive timeline of memory over time where every colored band is one live allocation. A leak is the band that opens and never closes. Click it, and the viewer shows you the exact line of your code that allocated it. The counter said "memory is going up." The snapshot says "line 88 of `train.py` appended a tensor to `self.history` and never let it go." One of those is a diagnosis.

![a leaking service whose memory climbs from eighteen gigabytes to eighty over six hours and crashes, beside the same service holding a flat eighteen gigabytes for a full day after the fix](/imgs/blogs/memory-snapshot-and-leak-hunting-1.webp)

By the end of this post you will be able to take a service that OOMs every six hours, record its allocation history over a couple of hundred steps, open the snapshot, find the band that never frees, walk its stack to the offending line, and confirm the fix by watching the baseline go flat — the difference drawn in the figure above. This is the tool post for GPU memory debugging. Its sibling, [the CUDA caching allocator](/blog/machine-learning/performance-engineering/the-cuda-caching-allocator), explains the machinery underneath — *allocated* versus *reserved*, why you OOM at 60% used, how fragmentation happens. Here we do not explain the allocator; we hunt the leak inside it. If you have not read that post, the one thing to carry over is this: PyTorch caches freed memory rather than returning it to the driver, so *reserved* (what PyTorch holds) is always at least *allocated* (what your live tensors use), and a leak is growth in *allocated* — the live set — not merely in *reserved*. Keep that distinction; it is the whole diagnostic split later.

This post belongs to the [performance engineering playbook](/blog/machine-learning/performance-engineering/the-performance-engineering-playbook), whose recurring loop is *profile → read the trace → hypothesize the cause → apply one fix → re-measure*. A leak hunt is that loop applied to memory: the snapshot is the profiler, the never-freed band is the reading, the retained reference is the cause, `.item()` is the fix, and a flat baseline over 24 hours is the re-measure.

## The symptom: a number that only goes up

Let us be precise about what you actually observe, because the shape of the growth is your first and cheapest clue. You have a service — say a Transformer running inference behind an HTTP handler, or a training loop grinding through batches — and you are watching memory with the bluntest possible instrument:

```bash
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv -l 60
```

```console
memory.used [MiB], utilization.gpu [%]
18432 MiB, 71 %
18604 MiB, 68 %
18776 MiB, 73 %
18948 MiB, 70 %
19120 MiB, 72 %
```

Every minute, roughly 170 MiB more — about 2.9 MiB per second. It never comes back down. Utilization is fine — the GPU is doing real work — but the memory floor rises like a tide. Extrapolate: an 80 GB card that starts at 18 GB has about 62 GB, or roughly 63,500 MiB, of headroom, and at 2.9 MiB per second that headroom lasts about `63500 / 2.9 ≈ 21600` seconds — six hours — before it hits the wall. That matches the pager exactly. The interval is not random; it is the free headroom divided by the leak rate, and both are numbers you can read.

`nvidia-smi` reports the whole process's memory, which includes the CUDA context, cuDNN/cuBLAS workspaces, and PyTorch's reserved cache — so it is noisy and coarse. Tighten the instrument to PyTorch's own counters:

```python
import torch

def mem(tag):
    a = torch.cuda.memory_allocated() / 1e9
    r = torch.cuda.memory_reserved()  / 1e9
    print(f"[{tag}] allocated={a:6.2f} GB   reserved={r:6.2f} GB")

# inside the loop, once every 100 steps
mem(f"step {step}")
```

```console
[step 0]     allocated=  2.14 GB   reserved=  2.38 GB
[step 100]   allocated=  2.86 GB   reserved=  3.12 GB
[step 200]   allocated=  3.58 GB   reserved=  3.88 GB
[step 300]   allocated=  4.30 GB   reserved=  4.62 GB
[step 1000]  allocated=  9.34 GB   reserved=  9.62 GB
```

Now the signature is unambiguous. `allocated` — the sum of your *live* tensor bytes — is rising by about 7.2 MB per step, dead linear, and `reserved` tracks it about 300 MB above. This is the fingerprint of a leak, and specifically the fingerprint that separates a leak from its two impostors:

- A **leak** grows `allocated` monotonically. The live set genuinely gets bigger every step because something keeps a reference to a tensor that should have died. The floor rises forever.
- **Fragmentation** grows `reserved` while `allocated` sawtooths — up on the forward pass, back down on the backward, up, down — and you see `cudaMalloc retries` climb in the memory summary. The live set is stable but the allocator cannot pack it into the blocks it holds. That is the caching allocator's problem, covered in [the CUDA caching allocator](/blog/machine-learning/performance-engineering/the-cuda-caching-allocator).
- **Expected warming** grows `allocated` for the first hundred or so steps — the first time it sees the largest batch shape, the first time an optimizer materializes its state — then plateaus. Growth that stops is not a leak. Growth that never stops is.

It helps to lay the three side by side, because the same rising curve on `nvidia-smi` can be any of them and the fix is completely different for each:

| Cause | `allocated` | `reserved` | `cudaMalloc retries` | Where growth stops | Fix lives in |
|---|---|---|---|---|---|
| **Leak** | rises every step | tracks allocated | 0 | never | your code (drop the reference) |
| **Fragmentation** | flat / sawtooth | high, pulls away | climbs | never (but live set stable) | `PYTORCH_CUDA_ALLOC_CONF` |
| **Warming** | rises then plateaus | rises then plateaus | 0 | after ~100–200 steps | nothing — it is expected |

The only row that is a leak is the first, and the only tool that confirms it is the first *and* pins it to a line of code is the snapshot. The linear, unbounded rise in `allocated` is what sends us there. A counter told us *that* memory leaks. It cannot tell us *what* leaks. For that we need the recorder.

## What the memory snapshot actually records

Here is the mechanism, because you will trust the tool more once you know it is not magic. PyTorch's caching allocator is the single chokepoint through which every CUDA tensor's memory passes — `torch.empty`, the output of every kernel, every intermediate. When you call `torch.cuda.memory._record_memory_history()`, you install a lightweight recorder on that chokepoint. From that moment, every allocation and every free event is intercepted before it returns, tagged with metadata, and appended to a ring buffer.

![a dataflow where the allocator hooks fan every alloc and free event into a stack capture step, both merge into a hundred thousand entry ring buffer, which is dumped to a pickle and opened in the viewer](/imgs/blogs/memory-snapshot-and-leak-hunting-3.webp)

The figure traces one event. The allocator hook sees a call — either an `alloc` (a new block handed out, tagged with its size and device address) or a `free` (a block returned, tagged with the address released). Both kinds flow into the same step: the recorder walks the current call stack and captures it — Python frames and, if you ask, the C++ frames underneath. That stack is the return address the raw counter never had. Both event streams then merge into one time-ordered ring buffer holding up to `max_entries` events. When you call `_dump_snapshot()`, PyTorch serializes that buffer — every event, every size, every captured stack, plus the allocator's current segment layout — into a single pickle file. The [memory_viz](https://pytorch.org/memory_viz) page is a pure client-side reader for that pickle; nothing is uploaded, it all runs in your browser.

The cost of recording is real but modest: capturing a Python stack on every allocation is not free, and on a hot loop that allocates thousands of tensors per step you will see maybe 5–15% slowdown while recording is on. That is fine — you record for a couple of hundred steps to reproduce the growth, then turn it off. You do not ship with it on. The `max_entries` cap is a ring buffer precisely so that a long run does not grow the history unbounded; older events roll off. For leak hunting you want enough entries to cover the window where the growth is visible, which is why 100,000 is the usual starting number.

Two options on the recorder are worth knowing from the start. `context` controls how much stack detail is kept per event (`"all"` keeps allocation and free stacks; you almost always want the allocation stack). `stacks` chooses `"python"` or `"all"` (Python-only is lighter and usually enough; `"all"` adds the C++ frames, useful when the leak is inside a fused kernel or a custom op). We will turn them up when Python frames alone do not name the culprit.

One subtlety about the x-axis: the timeline counts *allocation events*, not milliseconds. That is deliberate and it is what makes the plot readable. Wall-clock time compresses the interesting part — a leak that adds one band per step is spread evenly across steps whether each step takes 40 ms or 400 ms — and it lets a slow warmup dominate the frame. Counting events puts every allocation on equal footing, so a staircase of retained bands reads as a staircase regardless of how fast the loop runs. If you need to correlate a band back to wall-clock time, the underlying event carries a timestamp; the plot just does not use it as the horizontal scale.

There are also two ways to get the history out. `_dump_snapshot(path)` serializes to a pickle on disk — the one you drag onto the viewer. `torch.cuda.memory._snapshot()` returns the same structure as an in-process Python dict, which is what you want when you would rather assert on it in code than eyeball a plot — for example, counting how many live blocks share a given allocation frame and failing a test if that count grows. Same data, two doors: the pickle for the human, the dict for the machine.

Here is the honest cost table for turning recording on, so you can decide how long to leave it running:

| Setting | What it captures | Overhead | Use it for |
|---|---|---|---|
| `stacks="python"` | Python frames per event | ~5–10% | almost every leak hunt |
| `stacks="all"` | Python + C++ frames | ~10–20% | leaks inside kernels / custom ops |
| `context="all"` | alloc *and* free stacks | small extra memory | when you need to see what freed a block early |
| recording off | nothing | 0% | steady-state serving and all speed benchmarks |

The overhead is per allocation, so a loop that allocates thousands of tiny tensors per step pays more than one that allocates a few big ones. This is why the discipline is *record for a bounded window, then turn it off*: you need the history only long enough to catch the growth, and you never want the stack-capture tax in a latency benchmark or a production hot path.

### The code that turns it on

The whole recording API is three calls. Turn it on before the region you want to observe, run enough steps that the leak is visible, dump, and turn it off:

```python
import torch

# 1. Start recording. max_entries caps the ring buffer of alloc/free events.
torch.cuda.memory._record_memory_history(
    max_entries=100_000,
    stacks="python",     # "all" also captures C++ frames
    context="all",       # keep both alloc and free stacks
)

# 2. Reproduce the growth. Run enough steps that the leak is obvious
#    in the timeline — a few hundred is plenty if it leaks every step.
for step, (x, y) in enumerate(loader):
    train_step(model, optimizer, criterion, x, y)
    if step >= 300:
        break

# 3. Dump the recorded history to a file, then stop recording.
torch.cuda.memory._dump_snapshot("snap.pickle")
torch.cuda.memory._record_memory_history(enabled=None)   # None disables it
```

That is the entire tool. `snap.pickle` might be a few megabytes to a couple hundred, depending on how many events you captured. You drag it onto [pytorch.org/memory_viz](https://pytorch.org/memory_viz) and the timeline appears. In a headless training job, wrap the dump so that it fires on the way to an OOM as well, so a real crash still leaves you a snapshot to read:

```python
import torch

def install_oom_snapshot(path="oom_snap.pickle"):
    torch.cuda.memory._record_memory_history(max_entries=200_000)

    def _on_oom(device, alloc, device_alloc, device_free):
        # Called by the allocator right before it raises OOM.
        torch.cuda.memory._dump_snapshot(path)
        return False   # do not attempt to retry; let the OOM propagate

    torch._C._cuda_attachOutOfMemoryObserver(_on_oom)
```

Now even the 03:14 crash you never saw dumps a snapshot as it dies, and you read the leak in the morning from the exact state that killed it.

## The leak-hunt loop: seven moves, every time

A leak hunt is not a stroke of insight; it is a fixed procedure you run the same way every time. The figure lays out the seven moves.

![a seven step ordered loop showing record history, run steps, dump snapshot, open the viewer, find the band that never frees, click to the stack, then fix and rerun to a flat baseline](/imgs/blogs/memory-snapshot-and-leak-hunting-2.webp)

1. **Record history.** Turn on the recorder with a `max_entries` large enough to cover your reproduction window.
2. **Run N steps.** Let the service take enough traffic, or the training loop take enough batches, that the growth is unmistakable — a few hundred steps if it leaks every step, more if it leaks per request or per epoch.
3. **Dump the snapshot.** Serialize the history to `snap.pickle`.
4. **Open memory_viz.** Drag the pickle onto the page. The memory-over-time plot renders.
5. **Find the band that never frees.** In the plot, most bands are short — they open and close within a step. The leak is the band, or the stack of bands, that opens and stays open, pushing the baseline up.
6. **Click to the stack.** Selecting a leaked block shows the captured allocation stack — the exact file, function, and line that allocated it.
7. **Fix and re-run.** Remove the retention, turn recording back on, run the same N steps, and confirm the baseline is now flat.

The discipline matters because leaks lie to intuition. The line that *allocates* the leaked tensor is often innocent — it is a perfectly normal `model(x)` — and the bug is somewhere else entirely holding a reference to its output. The snapshot cuts through that: it shows you what is *still alive*, and the stack tells you where it was born, and from the birthplace you can usually find the crib it never left. You do not guess. You read.

## Reading the viewer

Open `snap.pickle` and you get two views that answer two different questions. Do not confuse them; people waste hours reading the wrong one.

The **Active Memory Timeline** is the one you want for a leak. The x-axis is time (in allocation events, not wall-clock — the tool counts events). The y-axis is total live bytes. Every allocation is a colored horizontal band that starts when the block is allocated and ends when it is freed. A healthy step is a flurry of bands that open on the forward pass and all close on the backward pass, so the total returns to the same baseline every iteration. A leak is a band — or a new band every step — that opens and does not close, so the envelope of the whole plot ramps upward. You are looking for the staircase: a floor that rises one step at a time and never descends.

The **Allocator State History** answers a different question — *is the allocator fragmented?* — by drawing the reserved segments and which parts of each are in use versus free. That is the view for the fragmentation story in [the CUDA caching allocator](/blog/machine-learning/performance-engineering/the-cuda-caching-allocator): lots of reserved segments each with a little live data wedged in, so you cannot fit a big new block despite plenty of free bytes. For a leak, you live in the Active Memory Timeline.

When you click a band, the viewer shows its **allocation stack** — the frames captured at the moment that block was allocated. This is where the counter's bytes finally get their return address. You read the stack top-down: the deepest frame is usually a PyTorch internal (`empty`, `_to_copy`, a kernel), and a few frames up is *your* code — the line in your model, your handler, your training loop that triggered the allocation. That line, plus the fact that the block is still alive N steps later, is the diagnosis.

A few navigation habits make the viewer fast. Zoom the timeline to the tail of your reproduction window, where the leaked bands are thickest and easiest to click — early bands are buried under the warmup. Bands that share an allocation stack render in the same color, so a leak that adds one identical band per step shows up as a fat wedge of one color climbing the right edge; that color-clustering is often the first thing your eye catches before you have clicked anything. When several stacks are involved, hover to read the top application frame of each and group them mentally by call site — a real leak usually funnels through one or two lines, not fifty. And when the plot is too busy to read, shorten the reproduction: 100 steps of a per-step leak is enough to draw a clear staircase, and a smaller pickle opens faster and clicks cleaner than a 200 MB one.

### The categories a healthy snapshot shows

Before you can spot the anomalous layer you need to know the normal ones. A training step's live memory decomposes into four stable categories, and the snapshot colors them so you can tell an extra one apart.

![a stack of memory categories showing parameters, gradients, optimizer state, and activations as stable layers with a fifth retained layer growing every step on top](/imgs/blogs/memory-snapshot-and-leak-hunting-5.webp)

- **Parameters** — the model weights. Fixed for the whole run. On a mid-size model, a few gigabytes. This band is allocated once at load and never moves.
- **Gradients** — one buffer per parameter, same total size as the parameters in the common case. Allocated on the first backward, reused every step. Stable.
- **Optimizer state** — Adam keeps two moments (`m` and `v`) per parameter, so roughly twice the parameter bytes. Materialized on the first `optimizer.step()`, then stable. This is why memory jumps once after the first step and then plateaus — expected warming, not a leak.
- **Activations** — the intermediates saved by the forward pass so the backward pass can compute gradients. This is the band that *should* breathe: it inflates on the forward and deflates on the backward, every single step. Its peak sets your batch-size ceiling.

Those four are your baseline. A leak is a **fifth** layer the snapshot draws on top — retained memory that grows step over step and never frees. The whole art of reading the snapshot is recognizing that fifth layer against the four that belong there. If `allocated` is climbing and the activation band still breathes normally, the growth is the retained layer, and the retained layer has a stack.

## Why a retained tensor holds its whole graph

This is the mechanism that makes the most common training leak so brutal, and it is worth deriving because it explains why the growth rate is what it is. When you compute a loss with autograd, the loss tensor is not just a number. It carries a `.grad_fn` — a pointer to the node at the top of the backward graph. That node points to its inputs' `grad_fn`s, and so on, all the way back to the parameters. The graph is a DAG of every operation that produced the loss.

Here is the sharp part: many of those backward nodes **save tensors** they will need to compute gradients. A matmul's backward needs the inputs to the matmul. A `ReLU`'s backward needs its output mask. Softmax, layernorm, attention — they all stash activations for the backward pass. Those saved tensors are held *by the graph*. As long as anything holds a reference to the loss (or to any tensor in the graph), the whole graph stays alive, and the whole graph *pins every saved activation in GPU memory*.

Normally this is fine, because the reference dies at the end of the step. Python's reference counting frees `loss` when the loop variable is reassigned on the next iteration; freeing `loss` drops the last reference to its `grad_fn`; the graph is collected; every saved activation is freed back to the allocator. That is the activation band breathing out.

But keep one reference — append `loss` to a list, stash `output` in a dict, add `loss` to a running total without detaching — and the graph cannot die. Every saved activation from that step stays live. Do it every step and you pin one full forward pass's worth of activations per step, forever. If a step's saved activations are $A$ bytes, then after $n$ leaked steps the retained memory is

$$M_\text{leaked}(n) = n \cdot A$$

which is exactly the dead-straight linear climb you saw in the counter. The leak rate *is* the per-step activation footprint of the retained subgraph. On the earlier trace, 7.2 MB/step means the retained graph pins about 7 MB of activations per step — a small model, or a small retained subgraph of a big one. On a large Transformer training run, one retained `loss` can pin hundreds of megabytes per step, and you OOM in minutes, not hours.

This is why the fix is almost always to strip the graph off the tensor before you keep it. `loss.item()` pulls the scalar to the host as a plain Python float — no `grad_fn`, no graph, nothing pinned. `loss.detach()` gives you a tensor that shares the data but carries no `grad_fn`, so the graph behind it is free to die. Either one lets the activations breathe out. The bug and the fix differ by one method call, and the difference is $n \cdot A$ bytes.

There is a second-order version worth naming because it surprises people who *do* detach. Gradient accumulation — running several forward/backward passes before one `optimizer.step()` — is legitimate and does not leak, because each backward frees its own graph as it goes. But if you accumulate the *loss tensors* across the micro-batches to average them for logging, and you forget to detach, you rebuild the same $n \cdot A$ pin, now per accumulation window rather than per step. The rule survives intact: the moment a graphed tensor outlives the backward pass that would have freed it, you are pinning activations. Detaching at the boundary — `running += loss.detach()` inside the window, `running.item()` at the end — keeps accumulation legitimate and the memory flat.

## Leak patterns and their snapshot signatures

Five patterns account for the overwhelming majority of GPU leaks I have chased. Each leaves a recognizable signature in the snapshot, and each has a one-line fix. The figure is the lookup table; the sections below are the details.

![a table mapping five leak patterns to their snapshot signature and a one line fix, covering graph holding lists, missing zero grad, unevicted caches, growing key value buffers, and loss accumulation](/imgs/blogs/memory-snapshot-and-leak-hunting-4.webp)

### Pattern 1: accumulating graph-holding tensors

The classic. You want to log the loss curve, so you keep the losses:

```python
losses = []
for step, (x, y) in enumerate(loader):
    out  = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    losses.append(loss)          # BUG: loss still carries its grad_fn
```

`loss` carries `.grad_fn`, so appending it pins that step's graph and all its saved activations. **Snapshot signature:** a new retained band appears every step, all with the same allocation stack pointing into your model's forward — because each is a fresh forward pass's activations, kept alive. The Active Memory Timeline shows a perfect staircase. **Fix:** keep the number, not the graph.

```python
    losses.append(loss.item())     # host float, no graph — or loss.detach()
```

`.item()` also synchronizes the stream (it has to, to read the value on the host), so in a very hot loop prefer accumulating `loss.detach()` on device and calling `.item()` once at logging time. Both kill the leak; `.detach()` avoids the per-step sync.

#### Worked example: the 7 MB-per-step staircase

A ResNet-50 training loop on an **A100 80GB SXM** started at 2.14 GB allocated and climbed 7.2 MB per step, dead linear, OOMing after about 10,000 steps — roughly every 6.2 hours at the loop's step rate. Recording history for 300 steps and opening the snapshot showed the staircase immediately: 300 retained bands, each about 7 MB, each with an identical stack ending in `criterion(out, y)` and `losses.append(loss)`. Changing `append(loss)` to `append(loss.item())` flattened it: allocated held at 2.14 GB for 20,000 steps with no growth. Memory growth per 1000 steps went from 7.2 GB to 0.00 GB. The fix was seven characters.

### Pattern 2: gradients that never zero

You forgot `optimizer.zero_grad()`, or you call it but PyTorch keeps the buffers around:

```python
for x, y in loader:
    loss = criterion(model(x), y)
    loss.backward()      # gradients ACCUMULATE into .grad buffers
    optimizer.step()
    # no zero_grad(): grads keep summing, buffers stay allocated
```

Missing `zero_grad` is more often a correctness bug than a leak — gradients accumulate and your training diverges — but it does keep the gradient buffers allocated. The subtler memory point is `zero_grad()` versus `zero_grad(set_to_none=True)`. The default zeros the buffers in place, keeping them allocated. `set_to_none=True` releases them, so between backward passes the gradient memory can be reused. **Snapshot signature:** a persistent gradient band that never releases, sized to the parameter count. **Fix:** `optimizer.zero_grad(set_to_none=True)`, which has been the default since PyTorch 2.0 but is worth asserting in older code.

### Pattern 3: an unbounded cache

Someone added a cache to skip recomputation, and the cache never evicts:

```python
class Embedder:
    def __init__(self, model):
        self.model = model
        self.cache = {}                     # BUG: grows without bound

    def embed(self, key, x):
        if key not in self.cache:
            self.cache[key] = self.model(x)  # GPU tensor stored forever
        return self.cache[key]
```

Every distinct `key` stores a GPU tensor that lives as long as the object. In a service with unbounded key cardinality — request IDs, user IDs, timestamps — the dict grows forever. **Snapshot signature:** a steadily growing set of retained bands with the same allocation stack (`self.model(x)`), but the growth rate tracks *request diversity*, not steps — so it climbs faster under real traffic than in your synthetic loop. That is what makes it evade testing. **Fix:** cap the cache and evict, and store the tensor detached (and often on CPU):

```python
from functools import lru_cache            # or a manual LRU with a max size

# For tensors, a manual bounded dict is clearer than lru_cache:
from collections import OrderedDict

class Embedder:
    def __init__(self, model, cap=1024):
        self.model, self.cap = model, cap
        self.cache = OrderedDict()

    def embed(self, key, x):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        v = self.model(x).detach()          # no graph pinned
        self.cache[key] = v
        if len(self.cache) > self.cap:
            self.cache.popitem(last=False)  # evict oldest → frees its GPU tensor
        return v
```

### Pattern 4: a growing KV cache or metrics buffer

In LLM serving, the KV cache holds the attention keys and values for every token of every in-flight sequence. If a finished request's blocks are not released, its KV band never frees. **Snapshot signature:** one large retained band per request, freed when the request completes — unless the completion path has a bug, in which case the bands accumulate and the pattern looks exactly like Pattern 3 but with much larger blocks. A related, sneakier version is a **metrics buffer**: you append per-request latency or logits tensors to a list for a percentile computation and never clear it, and because they are small you do not notice until hour twelve. **Fix:** free KV blocks on request completion (paged attention systems do this by design — see [continuous batching and paged attention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention)), and for metrics, keep host scalars, not device tensors: `latencies.append(t.item())`, never `latencies.append(t)`.

### Pattern 5: `loss += loss` across steps

The most famous one-liner leak:

```python
total_loss = 0
for x, y in loader:
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    total_loss += loss           # BUG: total_loss becomes a live graph
```

The first iteration, `total_loss` is `0` and `total_loss += loss` makes it a tensor with a `grad_fn`. The second iteration adds another graphed tensor, and now `total_loss`'s graph references *both* steps' subgraphs. By step $n$, `total_loss` pins the saved activations of all $n$ steps. **Snapshot signature:** the same staircase as Pattern 1, and the stack points at the `total_loss += loss` line. **Fix:** add the detached scalar.

```python
    total_loss += loss.item()    # accumulate the float, not the graph
```

The bug and the fix are `loss` versus `loss.item()`. It is worth internalizing the rule that generates all five fixes: **never keep a tensor that carries a `grad_fn` past the step that produced it.** If you want the value, take `.item()`. If you want the tensor without the graph, take `.detach()`. If you want it off the GPU, take `.detach().cpu()`. The snapshot exists to catch the times you forgot.

## Confirming the fix by reading memory_stats

The snapshot finds the leak; `torch.cuda.memory_summary()` confirms it is gone. Print it before and after your window, and read the `Cur Usage` column for `Allocated memory`. Before the fix, over 1000 steps:

```console
|===========================================================================|
|                  PyTorch CUDA memory summary, device ID 0                  |
|---------------------------------------------------------------------------|
|            CUDA OOMs: 0            |        cudaMalloc retries: 0          |
|===========================================================================|
|        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed  |
|---------------------------------------------------------------------------|
| Allocated memory      |   9342 MiB |   9350 MiB |   1284 GiB |   1274 GiB |
| Reserved memory       |   9856 MiB |   9856 MiB |   9856 MiB |      0 B   |
| Active memory         |   9342 MiB |   9350 MiB |   1284 GiB |   1274 GiB |
| Active allocs         |     91834  |     91850  |          - |          - |
|===========================================================================|
```

Two numbers indict the leak. `Allocated memory` at 9342 MiB after starting at ~2100 MiB is the climb. `Active allocs` at 91,834 — ninety-one thousand live allocations — is absurd for a model that should hold a few thousand tensors at a time. That count is the retained bands, one per leaked step, and it is often the fastest tell: a healthy loop holds a roughly constant number of active allocations, and a leak makes the count rise in lockstep with the steps. After the fix, the same window:

```console
|        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed  |
|---------------------------------------------------------------------------|
| Allocated memory      |   2148 MiB |   6402 MiB |   1291 GiB |   1289 GiB |
| Active memory         |   2148 MiB |   6402 MiB |   1291 GiB |   1289 GiB |
| Active allocs         |      3187  |      4021  |          - |          - |
|===========================================================================|
```

`Allocated` back to 2148 MiB and holding. `Active allocs` at 3187 and stable. Note `Peak Usage` of 6402 MiB — that is the activation band breathing out to its normal peak on the forward pass, which is *supposed* to happen; peak memory and a leak are different things. `Tot Alloc` and `Tot Freed` are both enormous and nearly equal, which is the signature of health: the loop allocates and frees an enormous cumulative volume, but the live set stays small. A leak is precisely `Tot Alloc − Tot Freed` growing without bound.

For a scriptable check that does not require eyeballing a table, read the stats dict directly and assert on the trend:

```python
import torch

def allocated_gb():
    return torch.cuda.memory_allocated() / 1e9

base = allocated_gb()
for step in range(2000):
    train_step(...)
    if step % 500 == 499:
        grew = allocated_gb() - base
        # after warmup, allocated should not drift up
        assert grew < 0.10, f"leak: +{grew:.2f} GB by step {step}"
```

Drop that assertion into CI against a short run and a leak becomes a failing test instead of a 03:14 page. That is the highest-leverage thing in this post: the snapshot finds the leak once; a 20-line assertion keeps it found.

#### Worked example: the innocent allocating line

A team had an **H100 SXM** training run that OOMed after about 4 hours, allocated climbing 210 MB per step — a large per-step footprint, so a big retained subgraph. They installed the OOM observer, let the nightly run crash, and opened `oom_snap.pickle` in the morning. The Active Memory Timeline showed the staircase, and clicking the top leaked band gave a stack whose innermost application frame was `model(x)` — the forward pass. Their first instinct was that the forward was allocating too much, which is nonsense; the forward *should* allocate that much, and free it on the backward.

The trick was to read one frame further up. Above `model(x)` was `evaluate_and_log()`, a helper called every step that computed a validation metric and did `self.metric_history.append(val_out)` — where `val_out` was a graph-carrying tensor from a *second* forward pass under no `no_grad()`. The allocating line (`model(x)`) was innocent; the *retaining* line (`self.metric_history.append(val_out)`) was three frames up and in a different function. This is the whole reason you read the stack rather than guessing from the symptom: the block that leaks is born in normal code and orphaned somewhere else entirely. Wrapping the metric forward in `torch.inference_mode()` and appending `val_out.item()` dropped the per-step growth from 210 MB to 0 and turned the 4-hour crash into a clean multi-day run. Growth per 1000 steps: 210 GB → 0.00 GB.

## Leak, fragmentation, or expected? — the decision that comes first

Before you spend an afternoon in the snapshot, spend thirty seconds deciding whether you even have a leak. Three different things make memory climb, and only one of them is a leak. Get this wrong and you will hunt a phantom in the Active Memory Timeline while the real problem is fragmentation in the Allocator State History.

![a decision tree splitting climbing memory into leak when allocated rises every step, fragmentation when reserved is high with allocator retries, and expected warming when growth stops after the first steps](/imgs/blogs/memory-snapshot-and-leak-hunting-6.webp)

The tree is the triage. Start from "memory is climbing" and ask, in order:

- **Does `allocated` rise every step, monotonically?** If yes, it is a **leak** — the live set is genuinely growing, something retains what should die, and the snapshot's Active Memory Timeline will show the staircase. Go hunt the band.
- **Is `reserved` high and pulling away from `allocated`, with `cudaMalloc retries` climbing in the summary?** That is **fragmentation** — the live set is stable but the allocator holds many reserved segments it cannot pack a new request into. The fix is not in your code; it is `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` or `max_split_size_mb`, and the whole diagnosis is [the CUDA caching allocator](/blog/machine-learning/performance-engineering/the-cuda-caching-allocator)'s subject. The snapshot's Allocator State History is the view for it.
- **Does growth stop after the first hundred or two hundred steps?** That is **expected warming** — the first large batch shape, the optimizer materializing Adam's moments, cuDNN's autotuner caching workspaces. A plateau is not a leak. Only re-open the snapshot if the growth resumes.

The two impostors matter because they are common and they masquerade. Fragmentation makes `nvidia-smi` climb (reserved grows) while your live tensors are fine — chase it as a leak and you will find no retained band, because there is none. Warming makes the first minute look alarming and then settles. The counter cannot tell these three apart. The snapshot can: a leak is bands that never close, fragmentation is reserved segments with holes, and warming is a step that happens once. This is exactly why the two views exist.

### Reading a band: leak versus normal

Zoom into a single allocation and the difference is stark, which is the whole reason the timeline view works. The figure contrasts the two shapes.

![a comparison of a leaked block that opens once and never closes while its floor keeps rising against a normal block that opens on the forward pass and frees on the backward, keeping the baseline flat](/imgs/blogs/memory-snapshot-and-leak-hunting-7.webp)

A **normal band** opens on the forward pass — say +4.2 GB of activations — and closes on the backward pass when the graph is collected, returning those bytes to the pool. On the timeline it is a bump: up, then back down to the same baseline. Every step draws the same bump. The floor is flat.

A **leak band** opens once and never closes. On the timeline it is a step in a staircase: +1.8 GB that stays, so the *next* step's normal bump now sits on top of a higher floor, and the step after that on a higher floor still. By step 6000 the floor is at 38 GB and rising, and the OOM is a matter of arithmetic. The normal band and the leak band can have the *same size and the same allocation stack* — the difference is entirely whether the reference dies. That is why you cannot find a leak by looking at what allocates; you find it by looking at what fails to free. The Active Memory Timeline is the only view that shows failure-to-free directly, as a band whose right edge runs off the end of the plot.

## The before → after on named hardware

Here is the full measurement for the ResNet-50 training leak (Pattern 1) and, for contrast, an LLM inference cache leak (Pattern 3) reproduced on an **L4 24GB**. The A100 has room to hide a leak for hours; the L4, with a quarter of the memory, OOMs in minutes — smaller cards make leaks more urgent, not less.

| Metric | A100 80GB, before | A100 80GB, after | L4 24GB, before | L4 24GB, after |
|---|---|---|---|---|
| Allocated at start | 2.14 GB | 2.14 GB | 3.1 GB | 3.1 GB |
| Growth per 1000 steps | 7.2 GB | 0.00 GB | 9.6 GB | 0.00 GB |
| Active allocs (steady) | 92k and rising | 3.2k stable | 41k and rising | 2.4k stable |
| OOM interval | every 6.2 h | none in 24 h | every 34 min | none in 24 h |
| Peak allocated (steady) | climbs to 80 GB | 6.4 GB flat | climbs to 24 GB | 8.9 GB flat |
| Throughput impact | degrades near OOM | full | degrades near OOM | full |

Two honest notes on how to measure this without fooling yourself. First, **turn recording off before you measure throughput** — the stack capture costs 5–15% and will muddy a before/after speed comparison; measure memory with recording on, measure speed with it off. Second, **the growth-per-1000-steps number is the one to trust**, not the instantaneous `allocated`, because instantaneous allocated bounces with the activation band. Fit a line to `allocated` sampled every 100 steps after warmup; the slope is the leak rate, and a fixed leak has a slope of zero. "Peak allocated climbs to 80 GB" on the A100 is the same leak as "climbs to 24 GB" on the L4 — the slope is the same story, the ceiling just differs. Report the slope.

#### Worked example: the leak that only showed under real traffic

A recommendation service passed every test. In the load test — synthetic requests, 50 distinct user IDs replayed — memory was flat. In production it OOMed every 40 minutes. The difference was Pattern 3: a per-user embedding cache keyed on user ID, with no eviction. The synthetic test replayed 50 users, so the cache filled to 50 entries and stopped growing; production saw millions of distinct users, so the cache grew without bound. `nvidia-smi` showed the climb; it could not show that the climb tracked *user diversity* rather than *request count*.

Recording history in production for 200 requests and opening the snapshot showed a steadily rising set of retained bands, all with the same stack ending at `self.cache[uid] = self.model(x)`. The Active Memory Timeline made the tracking obvious: bands appeared at the rate of *new* users, not total requests. Capping the cache at 100k entries with LRU eviction and storing `.detach().cpu()` tensors flattened it — peak allocated dropped from a rising 24 GB to a flat 8.9 GB on the L4, and the 40-minute OOM became a 24-hour clean run. The lesson: **reproduce the growth under the traffic distribution that triggers it.** A leak keyed on cardinality is invisible to a low-cardinality test, and only the snapshot showed that the growth rate had nothing to do with request count.

## Stress-testing the diagnosis

A fix is not done until you have tried to break it. Here are the edges that matter for leaks.

**Does the leak reproduce at batch 1?** A per-step leak (Patterns 1, 2, 5) scales with batch size — bigger batches pin bigger activations — but it still leaks at batch 1, just slower. A per-request leak (Patterns 3, 4) does not care about batch size at all; it tracks request or key count. If your leak vanishes at batch 1, suspect an activation-pinning graph leak. If it is unchanged, suspect a cache or KV leak. The snapshot's growth rate tells you which knob it responds to.

**Does it leak in eval mode?** This one catches people. Inference under `model.eval()` still builds autograd graphs unless you wrap it in `torch.no_grad()` — `eval()` only changes dropout and batchnorm behavior; it does not stop grad tracking. So an inference service that appends outputs to a buffer *and* forgot `torch.no_grad()` leaks the full graph exactly like training:

```python
# BUG: eval() does not disable autograd. Every output pins a graph.
model.eval()
results = []
for x in requests:
    results.append(model(x))          # graph-carrying tensors accumulate
```

```python
# FIX: no_grad() strips grad_fn at the source, and store host/detached values.
model.eval()
results = []
with torch.no_grad():
    for x in requests:
        results.append(model(x).cpu())  # no graph to pin; off the GPU too
```

**Snapshot signature in eval mode:** retained bands with allocation stacks pointing into your model's forward, growing per request, *even though you never call `.backward()`*. That last detail — a leak with no backward pass in sight — is the tell that autograd is on when it should be off. The fix is `torch.no_grad()` (or `torch.inference_mode()`, which is stricter and slightly faster), and it eliminates the saved activations entirely, so the leak cannot form.

**Is it a leak or just fragmentation under concurrency?** Run 50 concurrent requests and memory climbs — is that a leak? Open the snapshot. If the Active Memory Timeline shows bands that free as requests complete, and the climb is in `reserved` not `allocated`, it is fragmentation from many differently-sized concurrent allocations, and the fix is `expandable_segments:True`, not a code change. If the bands never free, it is a real leak in the completion path. The two look identical on `nvidia-smi` and completely different in the snapshot — which is the entire argument for owning this tool.

**What if the stack points into a library, not your code?** Turn on C++ frames with `stacks="all"` and increase `context`. Sometimes the allocating frame is inside a fused kernel or a third-party op, and the Python-only stack stops at your call into the library. The C++ frames show which internal op holds the block, and the Python frames a few levels up still show your call site. If even that is opaque, bisect: disable half the model's caching/logging paths and see which half stops the growth.

**Does a forward hook or DDP hold a reference you forgot?** Two framework features quietly retain tensors. A `register_forward_hook` that stashes activations for later inspection keeps every activation it captures — if the hook appends to a list and you never remove the hook, that list is a leak with a stack pointing into the hook. And `DistributedDataParallel` holds gradient buckets; if you keep references to `.grad` tensors across steps (for gradient logging, say) you pin those buckets. The snapshot signature is the same retained-band staircase; the fix is to remove the hook when done (`handle.remove()`) and to log `.grad.norm().item()` rather than stash the gradient tensor itself. When the allocating stack lands in `torch/nn/modules/module.py` or the DDP reducer, suspect a hook or a bucket reference in your own code higher up the stack.

## Instrumenting a long-running service

For a service that leaks over days, not minutes, you cannot sit and watch. Wire the recorder into the process so it captures itself. Two patterns cover almost everything.

The first is the **OOM observer** shown earlier: attach it once at startup and every crash leaves a snapshot from the exact state that killed the process. That is your safety net for the leak you have not caught yet.

The second is a **rotating periodic snapshotter** for the slow drift you *are* hunting. Dump a snapshot every N minutes, keep the last few, and compare the newest against the oldest to read the trend without keeping the recorder on forever:

```python
import torch, time, threading, os

class LeakWatch:
    """Dump a rotating memory snapshot every `period` seconds."""
    def __init__(self, period=900, keep=4, out="/var/log/memsnap"):
        self.period, self.keep, self.out = period, keep, out
        os.makedirs(out, exist_ok=True)
        torch.cuda.memory._record_memory_history(max_entries=200_000)
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        i = 0
        while True:
            time.sleep(self.period)
            path = os.path.join(self.out, f"snap_{i % self.keep}.pickle")
            torch.cuda.memory._dump_snapshot(path)
            alloc = torch.cuda.memory_allocated() / 1e9
            print(f"[leakwatch] {path}  allocated={alloc:.2f} GB")
            i += 1
```

Now the on-call engineer has a snapshot from 15 minutes ago and one from an hour ago; opening both and comparing the floor of the Active Memory Timeline shows the drift, and the newer snapshot's bands carry the stacks. The cost is the recording overhead, which is why this is a hunting tool you enable when you *know* you are leaking — not a permanent fixture. Once the leak is fixed and the CI assertion is in place, take it back out.

## Case studies and real numbers

**PyTorch's own "Understanding GPU Memory" walkthrough.** The PyTorch team's blog post introducing `_record_memory_history` and the memory_viz viewer demonstrates the workflow on a ResNet-50 reference and on a deliberately planted reference-cycle leak, showing the retained blocks in the Active Memory Timeline and clicking through to the allocation stack. The headline: the snapshot turned "memory is going up" into a named line of code, and the same tool distinguishes a genuine leak from a normal activation peak. The [PyTorch memory docs](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html) are the authoritative reference for the API and the viewer.

**The `loss += loss` bug in the wild.** This exact pattern has appeared in countless training scripts and framework issue trackers — a running loss or metric accumulated without `.item()`, retaining the graph across the whole epoch. The symptom is always the same linear climb, and the snapshot always resolves it to the accumulation line in seconds. It is the single most common training-loop leak, which is why "accumulate the float, not the tensor" is worth memorizing as a rule rather than rediscovering as a bug.

**Fragmentation masquerading as a leak.** A widely reported class of OOMs — memory climbing in `reserved` while `allocated` is flat, `cudaMalloc retries` in the summary — has nothing to do with retained references and everything to do with variable sequence lengths fragmenting the allocator's segments. Teams have chased these as leaks for days before the snapshot's Allocator State History showed the truth: stable live set, fragmented reserve. The fix was `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and the story lives in [the CUDA caching allocator](/blog/machine-learning/performance-engineering/the-cuda-caching-allocator). The case study's value is negative: it is the leak that was not a leak, and only the two-view snapshot could tell the difference.

**The slow inference leak that OOMed every six hours.** The forward-looking case study in this series — [the memory leak that OOMed every 6 hours](/blog/machine-learning/performance-engineering/the-memory-leak-that-oomed-every-6-hours) — is exactly the shape we have been dissecting: a service holding tensors it should have released, found by recording history over a reproduction window and reading the never-freed band. That post is the full war story end to end; this one is the tool it uses.

## When to reach for the snapshot (and when not)

The memory snapshot is not free and not always the right first move. Reach for it when:

- **`allocated` grows monotonically** and you have confirmed it is not warming — that is a leak, and the snapshot is the only tool that names it.
- **You OOM at a memory level well below the card's capacity and cannot explain why** — the snapshot's Allocator State History shows whether it is fragmentation.
- **A service degrades slowly over hours or days** — record on an OOM observer so the crash itself dumps a snapshot you can read.

Do **not** reach for it when:

- **The number is flat and you just want to reduce peak memory** — that is not a leak; use activation checkpointing, a smaller batch, or mixed precision, and read peak from `max_memory_allocated()`. The snapshot is for growth, not for level.
- **You are in a tight production hot path and cannot afford 5–15% overhead continuously** — record for a bounded reproduction window, then turn it off. Do not leave `_record_memory_history` on in steady-state serving.
- **The growth stops on its own** — it was warming. Confirm the plateau and move on; do not pathologize expected one-time allocations.
- **You have not yet split leak from fragmentation from warming** — do that triage first (the decision tree above). The snapshot answers "which block leaks"; it is wasted effort if the real answer is "the allocator is fragmented," which a one-line env var fixes.

The honest framing: the snapshot is a scalpel for the specific disease of retained references. It is overkill for peak-memory tuning and the wrong tool for fragmentation's *fix* (though the right tool for its *diagnosis*). Match the instrument to the symptom.

## Key takeaways

- **A counter tells you memory grew; the snapshot tells you what grew and where it was born.** `nvidia-smi` and `memory_allocated()` have no return address; `_record_memory_history()` captures the stack of every alloc and free.
- **A leak is monotonic growth in `allocated`.** Fragmentation grows `reserved` with `cudaMalloc retries`; warming grows then plateaus. Split the three before you hunt.
- **The Active Memory Timeline shows failure-to-free directly** — a band that opens and never closes, raising the floor of every subsequent step. Click it for the allocation stack.
- **A retained tensor pins its whole autograd graph**, and the graph pins every saved activation, so one kept `loss` leaks $n \cdot A$ bytes over $n$ steps — a dead-straight linear climb whose slope is the per-step activation footprint.
- **The five common leaks each have a signature and a one-line fix**: `.item()` or `.detach()` for graph-holding lists and `loss += loss`; `set_to_none=True` for gradients; a bounded LRU for caches; free-on-completion for KV; and never store a device tensor when a host scalar will do.
- **`eval()` does not disable autograd.** An inference leak with no backward pass in sight means you forgot `torch.no_grad()` / `torch.inference_mode()`.
- **Reproduce under the traffic that triggers the leak.** A leak keyed on cardinality is invisible to a low-cardinality test; the snapshot's growth rate reveals which knob — batch size, step count, or key diversity — it responds to.
- **Confirm the fix with a slope, not a level.** Fit a line to `allocated` after warmup; a fixed leak has slope zero. Then bolt a `+0.10 GB` assertion into CI so the leak stays fixed.

## Further reading

- [Understanding CUDA Memory Usage and the memory snapshot](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html) — the authoritative PyTorch docs for `_record_memory_history`, `_dump_snapshot`, and the memory_viz viewer.
- [pytorch.org/memory_viz](https://pytorch.org/memory_viz) — the interactive, client-side viewer you drag your `snap.pickle` onto.
- [The CUDA caching allocator](/blog/machine-learning/performance-engineering/the-cuda-caching-allocator) — allocated vs reserved, fragmentation, OOM at 60% used, and `PYTORCH_CUDA_ALLOC_CONF`. The sibling that explains the machinery you are hunting inside.
- [Why your AI service wastes CPU and GPU](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) — the four wastes and the profile → fix → measure loop this leak hunt instantiates.
- [The performance engineering playbook](/blog/machine-learning/performance-engineering/the-performance-engineering-playbook) — the capstone decision tree from symptom to profiler to fix.
- [Metrics that actually matter](/blog/machine-learning/performance-engineering/metrics-that-actually-matter) — allocated vs reserved among the numbers that lie, and which to trust for each question.
- [The memory hierarchy: registers, shared memory, and HBM](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm) — where the bytes a leak consumes actually live, and why HBM is the scarce resource you are protecting.

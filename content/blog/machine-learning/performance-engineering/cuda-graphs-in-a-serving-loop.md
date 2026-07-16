---
title: "CUDA graphs in a serving loop: bucketing batches and living with dynamic traffic"
date: "2026-07-14"
publishDate: "2026-07-14"
description: "Production traffic is dynamic; CUDA graphs demand static shapes. Learn the bucketing discipline that resolves the tension and turns a launch-bound service at 30% GPU utilization into a replay-bound one at 85%."
tags:
  [
    "performance-engineering",
    "gpu-optimization",
    "cuda-graphs",
    "cuda",
    "inference",
    "latency",
    "throughput",
    "pytorch",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Performance Engineering"
author: "Hiep Tran"
featured: true
readTime: 42
---

The service looked healthy on every dashboard except the one that mattered. An 8-billion-parameter decoder running a guardrail-and-routing job — it reads a request, generates a short structured verdict of ten to twenty tokens, and returns it — was sitting on a single A100 80GB SXM at 30% GPU utilization while the on-call graphs stayed green. Throughput was pinned at 180 requests per second, p99 latency at 180 ms, and every attempt to push more load just grew the queue without moving the GPU past a third busy. The team's instinct was the usual one: the GPU is the expensive part, it's only a third used, so we must be starved for requests. We were not. We had plenty of load. The GPU was starved for *work it could start*, because the CPU could not hand it kernels fast enough.

A single decode step of that model launched roughly 1,800 CUDA kernels — every layer's projections, its attention, its two matmuls, its norms, its residual adds, its activation, plus the sampling at the end. Each launch costs the CPU somewhere between five and ten microseconds of pure driver-side overhead before the GPU sees anything. Multiply five microseconds by 1,800 kernels and the host spends about nine milliseconds per step just *asking* for work, while the GPU needs only about three milliseconds to actually do it. The host cannot run far enough ahead to hide that, so the GPU finishes its three milliseconds of real work and then sits idle for six more, waiting for Python and the driver to catch up. That is the launch-bound signature, and it is the single most common reason a healthy-looking inference service wastes two-thirds of the most expensive silicon in the building.

![a two column comparison of a launch bound service at thirty percent utilization against a graphed service at eighty five percent utilization](/imgs/blogs/cuda-graphs-in-a-serving-loop-1.webp)

The fix for launch overhead is CUDA graphs: capture the whole decode step once as a recorded kernel graph, then replay it with a single launch instead of 1,800. The sibling posts in this series build that machinery from the ground up — [CUDA graphs from first principles](/blog/machine-learning/performance-engineering/cuda-graphs-from-first-principles) explains capture versus replay, and [CUDA graphs in PyTorch](/blog/machine-learning/performance-engineering/cuda-graphs-in-pytorch) wires `torch.cuda.graph` into a forward pass. But there is a catch that stops most engineers cold the first time they try it on a real service, and it is the whole subject of this post. **A captured graph is frozen. It replays the exact kernels on the exact tensor shapes at the exact memory addresses it recorded. Production traffic is the opposite of frozen — batch sizes vary request to request, sequence lengths vary token to token.** A graph captured for batch 8 and sequence length 256 is useless for a batch of 3 at length 200. So how do you apply a technique that demands static shapes to a service whose defining property is dynamic shapes?

The answer is *bucketing*, and by the end of this post you will be able to design a bucket set from your own traffic histogram, capture one graph per bucket at startup, route each live request to the smallest bucket that fits, pad it, replay, and account honestly for the three costs this buys you — memory, padding waste, and startup latency. This is where the launch-overhead track lands: a full host-bound-to-GPU-bound transformation on a real service, with the numbers, framed by the same four wastes and the profile → hypothesize → fix → measure loop from the [series intro](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu).

## The tension, stated precisely: why graphs need static shapes

Before the fix, be exact about the constraint, because everything downstream is a workaround for it. When you capture a CUDA graph you are not recording your Python code. You are recording the *sequence of GPU operations* that your Python code happened to issue on one particular run: this kernel, then that kernel, reading from this address, writing to that one, with these launch dimensions. Capture walks your forward pass once with recording turned on and freezes the resulting kernel DAG. Replay re-issues that exact DAG in one driver call.

That is why replay is fast — there is nothing left for the CPU to decide, so the per-op launch overhead vanishes — and it is also why replay is rigid. The recorded kernels have their grid and block dimensions baked in from the shapes they saw at capture. The recorded memory operations point at the specific addresses of the tensors that existed at capture time. Feed the graph a batch of a different size and the kernels would need different launch dimensions, which the recording does not have. Point it at a freshly allocated input tensor and the recorded reads would fetch from the wrong address. Neither is allowed. A captured graph has exactly two hard requirements, and violating either is the source of nearly every "my graph produced garbage" bug catalogued in the [gotchas-and-debugging sibling](/blog/machine-learning/performance-engineering/cuda-graphs-gotchas-and-debugging):

1. **Static shapes.** Every tensor in the captured region must have the same shape on replay as at capture. No data-dependent sizes, no varying batch, no growing sequence length.
2. **Static addresses.** Inputs and outputs live in fixed *static* buffers. You do not pass a new tensor to a graph; you copy your data *into* the graph's static input buffer, replay, and read the result *out* of its static output buffer.

Dynamic traffic collides with requirement 1 on two axes at once. **Batch size** varies because a serving scheduler assembles whatever requests happen to be waiting — sometimes 1, sometimes 12, rarely the same number twice. **Sequence length** varies because a decode step attends over a KV cache that grows by one token every step and differs across concurrent requests. A single frozen graph can serve exactly one (batch, sequence) shape. Production hands you thousands of shapes. That is the tension in one sentence: the technique that removes your bottleneck refuses to run on the input your service actually receives.

### Why the decode step is the graph-friendly part

Here is the observation that makes the whole thing tractable, and it is worth internalizing because it decides *what* you graph. An autoregressive generation request has two phases with very different shape behavior. **Prefill** ingests the whole prompt at once: its sequence dimension is the prompt length, which is genuinely different on every request and can be anything from 5 to 5,000 tokens. Prefill is a shape nightmare and, mercifully, it is usually compute-bound (a big matmul over the whole prompt), so it does not suffer much from launch overhead and rarely needs graphing. **Decode** generates one token at a time: on every step the batch processes exactly one new token per sequence, so the *per-step* tensor shapes are fixed once you fix the batch size and round the KV-cache length up to a bucket. Decode is memory-bound and made of many tiny kernels — the exact profile that launch overhead destroys.

So the decode step is simultaneously the part that hurts most from launch overhead *and* the part whose per-step shape is most nearly static. That is not a coincidence you have to be lucky to exploit; it is structural. You graph the decode step, bucketed by (batch, KV length), and you leave prefill eager or handle it with a separate coarse bucket set. Every production LLM serving stack that uses CUDA graphs — vLLM, TensorRT-LLM, SGLang — does exactly this. The rest of this post is the discipline for doing it yourself and knowing what it costs.

## Route, pad, replay: the bucketing strategy

The strategy is three moves, and once you see the shape of it the code writes itself. You choose a finite set of *buckets* — supported (batch, sequence) shapes, for example batch in {1, 2, 4, 8, 16} crossed with sequence padded to {128, 256, 512}. At startup you capture one CUDA graph per bucket. At request time you route each incoming shape to the smallest bucket that fits, pad the real data up into that bucket's static input buffer, replay the bucket's graph, and slice the real output back out.

![a routing diagram where a request is sent to the smallest resident bucket graph that fits then padded and replayed to produce a response](/imgs/blogs/cuda-graphs-in-a-serving-loop-2.webp)

Walk the figure with a concrete request: a batch of 3 sequences whose KV length has reached 200. There is no bucket for (3, 200) and there never will be — the whole point is to *not* have a graph per exact shape. The router finds the smallest bucket that dominates it on both axes: batch 3 rounds up to the batch-4 tier, length 200 rounds up to the 256 tier, so the request lands in bucket B4/S256. The router copies the three real sequences into the top three rows of that bucket's static input buffer (row four is padding, its attention masked off so it contributes nothing to the real rows), replays the B4/S256 graph in a single launch, and reads the three real output rows back. The other resident buckets — B8/S256, B16/S512, and the rest — sit idle in memory, ready for whatever shape the next request brings. One request touches exactly one graph; the fan-out in the figure is the router *choosing* among resident graphs, not running all of them.

The payoff, per step, is stark. The eager path issued about 1,800 launches and spent nine milliseconds of host time doing it. The replay path issues one launch — `cudaGraphLaunch` — and the host is done in well under a millisecond, free to prepare the next step or serve another stream while the GPU works. The three milliseconds of real GPU work is unchanged; graphs do not make kernels faster, they make the *asking* free. That is the entire mechanism, and it is why the utilization number moves so far: you removed the six milliseconds of idle, not the three of work.

Three properties of this design deserve to be stated before we cost them out, because they are where the engineering judgment lives:

- **The bucket set is finite and pre-chosen.** You do not capture a graph on demand for each new shape — capture is expensive (a full warmup-and-record pass) and you would thrash. You decide the buckets up front from your traffic and capture them all once.
- **Padding is the price of the frozen shape.** A request of batch 3 in a batch-4 bucket wastes one row's worth of compute. A sequence of length 200 in a 256-length bucket wastes attention work over 56 padded positions. Bucketing converts variance into a fixed, bounded overhead.
- **Coverage is never 100%.** Some shape will exceed your largest bucket — a batch of 20 when your top bucket is 16, a prompt whose KV blows past 512. You need a fallback path (split the batch, or run eager), and you need to know how often it fires.

## The mechanism: coverage, padding waste, and the memory bill

This is the part that turns bucketing from a vibe into an engineering decision. Three quantities govern every choice you make, and they pull against each other. Deriving them is what lets you pick buckets on purpose instead of by superstition.

**Launch cost, the thing we are removing.** Recall the launch-overhead law from the [kernel-launch-overhead post](/blog/machine-learning/performance-engineering/the-kernel-launch-overhead-problem): the host time spent launching is

$$t_\text{host} = N_\text{kernels} \times t_\text{launch}$$

With $N_\text{kernels} = 1800$ and $t_\text{launch} \approx 5\ \mu s$, that is $t_\text{host} \approx 9\ \text{ms}$. The GPU work per step is $t_\text{gpu} \approx 3.1\ \text{ms}$. Because the launch queue drains faster than Python refills it, the two do not overlap and the step is host-bound. GPU utilization is the fraction of wall-clock the device is actually busy:

$$\text{util} = \frac{t_\text{gpu}}{\max(t_\text{host},\, t_\text{gpu})} = \frac{3.1}{9.0} \approx 0.34$$

which is the 30%-ish we measured. Replay makes $t_\text{host}$ negligible (one launch, plus a little Python between steps and the sampling copy), so $\max(t_\text{host}, t_\text{gpu})$ collapses to roughly $t_\text{gpu}$ and utilization climbs toward the mid-eighties — not to 100%, because sampling, the scheduler's per-step Python, and the new-token copy still cost something the graph did not capture. Amdahl's law is honest about the ceiling: if graphs remove the fraction $p$ of a step that was launch overhead, the best speedup is

$$S = \frac{1}{(1-p) + p/s}$$

with $s$ the speedup on the launch portion (effectively enormous — 1,800 launches become 1). With $p = 9.0/12.1 \approx 0.74$ of the serial step being launch, the step time falls from about 12.1 ms toward 3.2 ms, a 3.8× step speedup. That is the whole prize, and it is bounded by how much of your step was launch to begin with — which is exactly why graphs help a launch-bound service enormously and a compute-bound one barely at all.

**Padding waste, the price we pay.** When a request of batch $b$ lands in a bucket of batch $B \ge b$, the graph still computes all $B$ rows; the extra $B - b$ rows are wasted work. The batch padding fraction is

$$w_\text{batch} = \frac{B - b}{B}$$

Sequence padding is similar but weighted by which kernels scale with length. In a decode step, the attention kernels read a KV cache of the bucket's padded length $S$ while only $s$ positions are real, so their wasted fraction is $(S - s)/S$; the projections and MLP scale with batch, not length, so they pay only the batch padding. A defensible single-number model for the wasted compute in a decode bucket is

$$w \approx \alpha \cdot \frac{B - b}{B} + (1 - \alpha) \cdot \frac{S - s}{S}$$

where $\alpha$ is the share of step FLOPs that scale with batch (the projections and MLP, typically the large majority) versus attention over the KV. For a request of (3, 200) in bucket (4, 256) with $\alpha \approx 0.8$: $w \approx 0.8 \cdot 0.25 + 0.2 \cdot 0.22 \approx 0.24$. About a quarter of that step's compute is padding. Whether that is acceptable depends entirely on how far the average request sits below its bucket — which the histogram tells you.

**Coverage, the requests we can serve at all.** With a largest bucket of batch $B_\max$ and length $S_\max$, the fraction of traffic a graph can serve directly is

$$\text{coverage} = P\big(b \le B_\max \ \wedge\ s \le S_\max\big)$$

Everything outside falls to the fallback path. If 3% of requests exceed length 512, then 3% of traffic runs eager (slow) or split (extra requests), and your p99 will carry that tail unless you size $S_\max$ to cover it. Coverage is a business decision dressed as a number: you are choosing how much of the long tail to pay graph memory for.

These three — the launch we remove, the padding we add, the coverage we buy — are the entire trade space. Coarser buckets: less memory, less startup time, more padding waste, coverage holes. Finer buckets: less padding, tighter coverage, more memory, longer startup. There is no free axis; there is only the point on the curve your traffic justifies.

## Choosing bucket granularity from the histogram

Now make the trade-off concrete. The knob is *how many buckets*, and it moves memory, padding waste, coverage, and startup time all at once, in the directions the mechanism predicts.

![a comparison matrix of coarse medium and fine bucket sets across memory padding coverage and startup cost](/imgs/blogs/cuda-graphs-in-a-serving-loop-3.webp)

Read the matrix as a curve you are choosing a point on. A **coarse** set — say three buckets — is cheap: 1.8 GB of graph memory, a 4-second startup, but it pads aggressively (a batch-5 request forced into a batch-16 bucket wastes 69% of its rows) and it leaves a coverage hole wherever the tail lives. A **fine** set — twenty buckets — barely pads (3% waste) and covers 99% of traffic, but it costs 11 GB of resident graph memory on top of your weights and takes 30 seconds to capture at every deploy. The **medium** set of eight buckets is usually the right answer for a real service: 4.5 GB, 12% average padding, 97% coverage, a 12-second startup. But "usually" is not "always," and the only way to place the point correctly is to look at your traffic.

The discipline is to pick buckets from a histogram, not from round numbers. Log every request's (batch, KV-length) for a representative day, build the 2-D histogram, and place bucket boundaries where mass accumulates so that most requests land *just* under a boundary (little padding) and few land far below one (much padding). Buckets should be dense where traffic is dense and sparse in the tail.

Here is the sizing done as code, not hand-waving. This reads a day of shape logs and proposes batch and sequence boundaries that keep average padding under a target:

```python
import numpy as np

# shapes: array of (batch, kv_len) sampled from one representative day
shapes = np.load("request_shapes.npy")          # e.g. (2_000_000, 2)
batches, lengths = shapes[:, 0], shapes[:, 1]

def padding_waste(boundaries, values, alpha_axis):
    """Mean fractional waste if each value rounds up to the next boundary."""
    boundaries = np.sort(boundaries)
    idx = np.searchsorted(boundaries, values, side="left")
    covered = idx < len(boundaries)
    bucket = boundaries[np.clip(idx, 0, len(boundaries) - 1)]
    waste = np.where(covered, (bucket - values) / bucket, np.nan)
    coverage = covered.mean()
    return np.nanmean(waste), coverage

# candidate tiers chosen to sit just above dense regions of the histogram
batch_tiers = [1, 2, 4, 8, 16]
seq_tiers   = [128, 256, 512]

wb, cb = padding_waste(np.array(batch_tiers), batches, alpha_axis="batch")
ws, cs = padding_waste(np.array(seq_tiers),   lengths, alpha_axis="seq")
print(f"batch: mean pad {wb:.1%}, coverage {cb:.1%}")
print(f"seq:   mean pad {ws:.1%}, coverage {cs:.1%}")
print(f"buckets = {len(batch_tiers) * len(seq_tiers)}  "
      f"(each captured graph costs ~{4.5/8:.2f} GB at this model size)")
```

```console
batch: mean pad 11.4%, coverage 99.7%
seq:   mean pad 9.8%, coverage 96.9%
buckets = 15  (each captured graph costs ~0.56 GB at this model size)
```

The output tells you three actionable things at once: the padding you will eat (about 11% on batch, 10% on sequence), the coverage you will get (batch is nearly total, sequence leaves a 3% tail beyond 512 that needs a fallback), and the memory bill (15 buckets at ~0.56 GB each is 8.4 GB — probably too many; you would drop the rarely-hit corners like B16/S512 down to a shared coarse bucket and land near eight). This is the entire granularity decision, made from data in ten lines.

#### Worked example: sizing buckets for a routing model

A guardrail service on one A100 80GB logs a day of traffic. The batch histogram is heavily bimodal — 60% of steps are batch 1 or 2 (interactive single requests), 35% are batch 4 to 8 (small dynamic batches), 5% are batch 9 to 16 (burst-assembled). The KV-length histogram is right-skewed: median 140, 90th percentile 300, 99th percentile 470, with a thin tail to 900.

Reading the mechanism off those numbers: batch tiers {1, 2, 4, 8, 16} put almost every request just under a boundary, giving the 11% batch padding above. Sequence tiers {128, 256, 512} cover the 99th percentile but not the thin tail past 512 — the 0.4% of steps beyond 512 will hit the fallback, which is acceptable for a p99 target but would wreck a p99.9 target, so if the SLO were tighter you would add a 1024 tier and eat the extra 0.9 GB. Crossing the tiers gives 15 candidate buckets; pruning the three long-sequence/large-batch corners that together see under 1% of traffic drops it to eight buckets at 4.5 GB — the medium column of the matrix, chosen on purpose. Average wasted compute lands near 12%, which at 30%-to-85% utilization is a rounding error against the launch overhead you just deleted. That is the trade made explicitly: pay 12% padding and 4.5 GB to delete 74% of your step time.

## Warmup: capturing every bucket at startup

Buckets chosen, you capture them — all of them, once, before the service takes traffic. Capture is not free and it is not instant, so it belongs at startup where it is a one-time cost, never on the request path where it would spike a tail.

![a startup timeline that loads weights once then captures each bucket graph in turn until all graphs are resident and the service serves traffic](/imgs/blogs/cuda-graphs-in-a-serving-loop-4.webp)

The timeline shows the shape of startup: load weights into HBM once (16 GB for the 8B model), then for each bucket run a few warmup iterations and record the graph, accumulating a memory pool per bucket, until all eight are resident and the service flips to serving. Two details in that sequence are load-bearing and easy to get wrong.

First, **every bucket needs warmup iterations before capture**, not just the first. Capture records whatever kernels run during the recorded pass, so anything that would run lazily on a real first call — cuBLAS or cuDNN autotuning picking an algorithm, the caching allocator growing a segment, a one-time workspace allocation — must have already happened, or it gets baked into the graph or, worse, tries to allocate *during* capture (which is illegal and aborts the capture). The convention is three to eleven warmup iterations on a side stream per bucket, then capture. Second, **capture allocates**. Each graph's memory pool holds the transient activations the recorded kernels write to, and those are captured at fixed addresses. That is where the per-bucket 0.4 to 0.9 GB in the timeline comes from, and it is why startup memory climbs bucket by bucket.

Here is the capture loop, using the low-level `torch.cuda.graph` context manager so the static-buffer discipline is explicit:

```python
import torch

class BucketGraph:
    """One captured decode-step graph for a fixed (batch, seq) shape."""
    def __init__(self, model, batch, seq, device="cuda", dtype=torch.bfloat16):
        self.batch, self.seq = batch, seq
        # Static input/output buffers: the graph reads/writes THESE addresses.
        self.static_tokens = torch.zeros(batch, 1, dtype=torch.long, device=device)
        self.static_kv_len = torch.full((batch,), seq, dtype=torch.long, device=device)
        self.static_out    = torch.zeros(batch, model.vocab_size, dtype=dtype, device=device)
        self.graph = torch.cuda.CUDAGraph()
        self._capture(model)

    def _capture(self, model):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            # Warm up on a side stream: force autotuning + allocator growth
            # so nothing lazy happens during capture.
            for _ in range(5):
                _ = model.decode_step(self.static_tokens, self.static_kv_len)
        torch.cuda.current_stream().wait_stream(s)

        with torch.cuda.graph(self.graph):          # begin recording
            self.static_out = model.decode_step(self.static_tokens, self.static_kv_len)

    def replay(self, tokens, kv_len):
        # Copy real data INTO the static input, replay, read OUT of static output.
        self.static_tokens[: tokens.shape[0]].copy_(tokens)
        self.static_kv_len[: kv_len.shape[0]].copy_(kv_len)
        self.graph.replay()                          # ONE launch replaces ~1800
        return self.static_out
```

And the startup loop that captures the whole set, logging the cost as it goes:

```python
import time, torch

BUCKETS = [(1,128),(1,256),(2,256),(4,256),(4,512),(8,256),(8,512),(16,512)]

def warmup_all(model):
    registry, t0 = {}, time.time()
    base = torch.cuda.memory_allocated() / 1e9
    for (b, s) in BUCKETS:
        t = time.time()
        registry[(b, s)] = BucketGraph(model, b, s)
        torch.cuda.synchronize()
        used = torch.cuda.memory_allocated() / 1e9
        print(f"captured B{b}/S{s:<4} in {time.time()-t:4.1f}s  "
              f"resident {used-base:5.2f} GB  total {time.time()-t0:5.1f}s")
    return registry
```

```console
captured B1/S128   in  1.9s  resident  0.31 GB  total   1.9s
captured B1/S256   in  1.1s  resident  0.58 GB  total   3.0s
captured B2/S256   in  1.0s  resident  0.94 GB  total   4.0s
captured B4/S256   in  1.2s  resident  1.63 GB  total   5.2s
captured B4/S512   in  1.4s  resident  2.41 GB  total   6.6s
captured B8/S256   in  1.3s  resident  3.02 GB  total   7.9s
captured B8/S512   in  1.6s  resident  3.98 GB  total   9.5s
captured B16/S512  in  2.1s  resident  4.51 GB  total  11.6s
```

Eight buckets, 4.5 GB of graph memory, 11.6 seconds of startup — the medium column, materialized. That startup cost is real and it matters operationally: it is added cold-start latency on every deploy and every autoscale event, so a service that scales pods up and down under bursty load pays it repeatedly. The mitigations are to keep pods warm (do not scale to zero if cold start is 12 seconds and your burst rises in 5), to capture buckets lazily in priority order (capture the hot B4/S256 first so you can serve the common case in 5 seconds while the rare corners finish in the background), or to snapshot and reuse the allocator state. What you must not do is move capture onto the request path.

### Sharing the pool to cut the bill

The 4.5 GB is not fixed by physics; a chunk of it is the per-bucket activation pools, and those can *share* memory because no two buckets replay at the same instant on the same stream. PyTorch exposes this: pass a shared pool handle so later captures reuse the same underlying segments instead of each grabbing its own.

```python
# First capture creates the pool; subsequent captures reuse it.
shared = None
for (b, s) in BUCKETS:
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=shared):
        out = model.decode_step(static_in[(b, s)], static_len[(b, s)])
    if shared is None:
        shared = g.pool()          # reuse this pool for every later bucket
```

Sharing the pool trades a little safety (you must never replay two pooled graphs concurrently on overlapping memory) for a real memory saving — on this service it took the eight-bucket footprint from 4.5 GB down to about 2.9 GB, because the transient activation scratch, the largest part of each pool, overlapped cleanly. `make_graphed_callables` does this pooling for you automatically across the callables you hand it, which is why it is the higher-level tool of choice when you are graphing several shapes of the same module; the [PyTorch-integration sibling](/blog/machine-learning/performance-engineering/cuda-graphs-in-pytorch) covers its ergonomics.

## The bucket table: what stays resident

After warmup, the service holds a table of captured graphs, one per supported shape, all resident in HBM. This table is the object the router indexes into on every request, so it is worth seeing as a table.

![a three by three grid of captured graphs indexed by batch tier and sequence tier with one hot center bucket](/imgs/blogs/cuda-graphs-in-a-serving-loop-5.webp)

The grid crosses three batch tiers with three sequence tiers into nine cells, each a captured graph with a fixed shape (a real deployment prunes the cold corners, as we did to reach eight, but the full grid shows the structure). Two things about this table drive real behavior. The **hot cell** — here B4/S256 — takes the large majority of traffic, because that is where the histogram's mass sits; most of your replays hit one or two buckets, and the rest exist for coverage and tail control. The **long-sequence column** (S512) is expensive per graph (bigger KV, bigger activation pool) and rarely hit, which is exactly the kind of cell you scrutinize when memory is tight: it costs the most and earns the least, so it is the first candidate to demote into a shared coarse bucket or drop entirely if its traffic can tolerate the fallback.

Keeping the whole table resident is the point — you never re-capture, you only replay — but it means the table's memory is a permanent tax you carry for the life of the process, on top of weights and KV cache. That is the bill we account for two sections down.

## Routing a request: which bucket, pad, or reject

The request path is the router: classify the shape, pick a bucket or a fallback, pad, replay, unpad. It is a small amount of code and it must be fast, because it runs on every request and any Python you add here is host overhead you just worked to remove.

![a decision tree that routes a request by whether its shape is in range then whether it hits a bucket exactly or needs padding with an out of range fallback](/imgs/blogs/cuda-graphs-in-a-serving-loop-6.webp)

The tree is the whole routing logic. First question: is the shape in range — batch at most 16, sequence at most 512? If yes, round batch and sequence up to the smallest tiers that fit and replay that bucket; the request either hits a bucket exactly (no padding) or pads up to it (bounded waste). If no — a batch of 20, or a KV that blew past 512 — it takes the fallback: split the oversized batch into two in-range replays, or run a single eager (un-graphed) step for the rare long sequence. The fallback is slow by design; it exists so that a shape you did not plan for degrades gracefully instead of crashing.

In code, the router is a lookup plus a copy plus a replay:

```python
import bisect, torch

BATCH_TIERS = [1, 2, 4, 8, 16]
SEQ_TIERS   = [128, 256, 512]

class GraphRouter:
    def __init__(self, registry):
        self.registry = registry               # {(batch, seq): BucketGraph}

    def _bucket_for(self, batch, kv_len):
        bi = bisect.bisect_left(BATCH_TIERS, batch)
        si = bisect.bisect_left(SEQ_TIERS, kv_len)
        if bi == len(BATCH_TIERS) or si == len(SEQ_TIERS):
            return None                        # out of range -> fallback
        return (BATCH_TIERS[bi], SEQ_TIERS[si])

    def step(self, tokens, kv_len):
        b, s = tokens.shape[0], int(kv_len.max())
        key = self._bucket_for(b, s)
        if key is None:
            return self._fallback(tokens, kv_len)     # split or eager
        graph = self.registry[key]
        out = graph.replay(tokens, kv_len)            # pad happens inside replay()
        return out[:b]                                # unpad: real rows only

    def _fallback(self, tokens, kv_len):
        b = tokens.shape[0]
        if b > BATCH_TIERS[-1]:                       # split oversized batch
            half = b // 2
            top = self.step(tokens[:half], kv_len[:half])
            bot = self.step(tokens[half:], kv_len[half:])
            return torch.cat([top, bot], dim=0)
        return self.model.decode_step(tokens, kv_len) # eager for long sequence
```

The `[:b]` unpad on the way out is the counterpart to the pad-in inside `replay()`: the graph always computes `B` rows and `S` positions, and you keep only the `b` real rows. The padded rows produced real numbers — the graph did not know they were fake — so you must slice them off, and you must have masked their attention during capture so they never influenced the real rows. Getting that mask wrong is a classic silent-garbage bug: the output looks plausible but a real sequence was subtly contaminated by a padding row. The [gotchas sibling](/blog/machine-learning/performance-engineering/cuda-graphs-gotchas-and-debugging) treats this failure mode in depth; the rule to carry is that padding must be *provably inert*, verified once with a test that replays a bucket with garbage in the padding rows and asserts the real outputs are bit-identical to the un-padded eager result.

#### Worked example: a batch of 3 through the router

Trace (3, 200) end to end with numbers. `_bucket_for(3, 200)`: `bisect_left([1,2,4,8,16], 3)` returns index 2 → batch tier 4; `bisect_left([128,256,512], 200)` returns index 1 → sequence tier 256. Bucket key (4, 256), which is resident. `replay` copies the three real token rows into `static_tokens[:3]`, leaves row 3 as its zero padding, sets `static_kv_len` to 256 for all four rows (attention masks the 56 padded positions and the one padded row). One `cudaGraphLaunch` fires the recorded ~1,800-kernel DAG. The router slices `out[:3]` and returns three logits rows. Wasted compute: one padded batch row out of four (25% on the batch-scaling kernels) plus 56 masked positions out of 256 (22% on attention), blended to about 24% overall — the padding tax, paid to buy a single-launch step. Host time on the request path: the two `copy_` calls and the launch, well under a millisecond, versus the nine milliseconds the eager path spent launching. That is the trade in one request.

## What N graphs actually cost in memory

Everything above bought speed with memory, so account for the memory precisely. It is the cost most teams underestimate, and it is the one that OOMs you at deploy if you skip the arithmetic.

![a layered memory stack showing shared weights plus per bucket static tensors and pools multiplied across buckets to a total footprint](/imgs/blogs/cuda-graphs-in-a-serving-loop-7.webp)

The stack shows where the bytes go. **Weights** are shared across every graph — all buckets replay the same 16 GB of parameters, so weights are paid once no matter how many buckets you capture. That is the good news, and it is why bucketing is even affordable. The **per-bucket cost** is two things: the static input and output tensors (small — a token buffer, a length buffer, a logits buffer), and the captured memory pool holding the step's transient activations (the large part). Total footprint is

$$M_\text{total} = M_\text{weights} + \sum_{i=1}^{N} \big(M_\text{io}^{(i)} + M_\text{pool}^{(i)}\big)$$

The pool term scales with the bucket's batch and sequence, so the big buckets dominate the sum — B16/S512's pool dwarfs B1/S128's. This is why the granularity decision is really a memory decision: doubling the bucket count roughly doubles the second term while leaving the first untouched, so at some point you are spending a gigabyte per deploy to shave a few percent of padding off a bucket that sees 0.3% of traffic. On this service: 16 GB weights + about 4.5 GB across eight independent pools, or 2.9 GB if the pools are shared. Add the KV cache (paged, shared across the service, sized separately) and you are budgeting toward 24 to 26 GB of the A100's 80 GB — comfortable, but only because we counted.

The failure mode when you do not count is memory pressure from too many graphs, and it is nastier than a clean OOM. Because each pool is captured at fixed addresses, graph memory cannot be reclaimed or compacted while the graphs are resident; it fragments the allocator's view of free space. A service that captures 20 fine buckets can find itself unable to grow its KV cache under load even though `nvidia-smi` shows free memory, because the free memory is in holes between pinned graph pools. The [caching-allocator behavior](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) that makes this bite is covered elsewhere in the series; the operational rule is to treat resident graph memory as a hard, upfront reservation you subtract from your KV budget before you decide how many concurrent requests you can hold.

## The full before → after, measured

Now the payoff, measured the way the series insists: warm up, synchronize, time steady state, read the profiler, and report named-hardware numbers you could reproduce. The service is the 8B guardrail decoder on one A100 80GB SXM (312 dense bf16 TFLOP/s, 2.0 TB/s HBM2e, per NVIDIA's A100 datasheet), eight buckets, requests generating a 16-token verdict.

Start by confirming the diagnosis, because you never optimize on a guess. Here is `torch.profiler` on the eager decode loop, the run that told us it was launch-bound:

```python
import torch
from torch.profiler import profile, ProfilerActivity, schedule

sched = schedule(wait=1, warmup=3, active=5, repeat=1)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             schedule=sched, record_shapes=True) as prof:
    for _ in range(9):
        for _ in range(16):                 # 16 decode steps = one request
            model.decode_step(tokens, kv_len)
        prof.step()
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=6))
```

```console
-----------------------------  ------------  ------------  ------------  ------------
Name                             CPU total %   CPU total    CUDA total   # of Calls
-----------------------------  ------------  ------------  ------------  ------------
cudaLaunchKernel                     71.4%       142.8ms         --          28800
aten::layer_norm                      6.1%        12.2ms       2.9ms          3200
aten::matmul                          5.8%        11.6ms      38.4ms          6400
aten::add                             4.2%         8.4ms       1.1ms          9600
cudaStreamSynchronize                 3.1%         6.2ms         --             16
aten::softmax                         2.0%         4.0ms       3.7ms          1600
-----------------------------  ------------  ------------  ------------  ------------
Self CPU time total: 200.1ms   Self CUDA time total: 49.6ms
```

The table is a confession. `cudaLaunchKernel` is 71% of CPU time — the host is spending its life launching, 28,800 launches across the profiled window, which is the 1,800 per step we predicted times sixteen steps. Total CPU time (200 ms) dwarfs total CUDA time (49.6 ms) by 4×: the host is the wall, the GPU is idle waiting. This is the launch-bound signature named exactly, and it is the go-signal for graphs. (If instead CUDA time had dominated and `cudaLaunchKernel` were a slim fraction, you would be compute-bound and graphs would be the wrong tool — a distinction the "when not to" section returns to.)

Apply the bucketed graphs, re-profile the same window, and the confession flips:

```console
-----------------------------  ------------  ------------  ------------  ------------
Name                             CPU total %   CPU total    CUDA total   # of Calls
-----------------------------  ------------  ------------  ------------  ------------
cudaGraphLaunch                      18.9%         9.6ms         --             16
aten::copy_ (into static in)         12.4%         6.3ms       0.4ms            32
aten::argmax (sampling)               9.1%         4.6ms       1.8ms            16
cudaStreamSynchronize                 7.7%         3.9ms         --             16
-----------------------------  ------------  ------------  ------------  ------------
Self CPU time total:  50.8ms   Self CUDA time total: 48.9ms
```

`cudaLaunchKernel` is gone from the top — the 1,800 per-step launches are inside the graph now, invisible to the host, replaced by 16 `cudaGraphLaunch` calls (one per step). CPU time fell from 200 ms to 51 ms and now roughly matches CUDA time (48.9 ms), which is what "GPU-bound" looks like: the host keeps up, the device sets the pace. The CUDA total barely moved (49.6 → 48.9 ms) because *the kernels did the same work* — graphs changed who waited on whom, not what the GPU computed.

One caution on how these numbers were produced, because a launch-bound service is exactly where naive timing lies to you. You must warm up before you time — the first replay of a bucket still pays cuBLAS autotuning and allocator growth, so a cold measurement of the graphed path looks *slower* than steady state and a cold measurement of the eager path looks faster than it really is once the queue fills. The honest recipe is the one the [reproducible-benchmark post](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) lays out: run 20 steady-state steps to warm caches and let the launch queue reach its true depth, then time a fixed window with CUDA events (`torch.cuda.Event(enable_timing=True)` around the region, `torch.cuda.synchronize()` before reading the elapsed time) rather than `time.time()`, which on the eager path would measure only how fast Python *enqueued* work, not how fast the GPU *finished* it. Lock the clocks if you can (`nvidia-smi -lgc`), because a launch-bound service runs cool and boosts differently than a GPU-bound one, and the boost difference alone can swing your step time 10%. The utilization and throughput numbers in the table are steady-state, event-timed, clock-locked; anything less and you are measuring your benchmark harness, not your service.

`nvidia-smi dmon` across the switch shows the same story from the hardware's side:

```console
# gpu   sm   mem   enc   dec   ...        <- BEFORE (eager)
    0   31    18     0     0
    0   28    17     0     0
    0   33    19     0     0
# gpu   sm   mem   enc   dec   ...        <- AFTER (bucketed graphs)
    0   84    52     0     0
    0   86    53     0     0
    0   85    51     0     0
```

SM activity climbs from the low thirties to the mid-eighties — the utilization number from the dashboard, now earned. The full before/after on named hardware:

| Metric (A100 80GB SXM) | Before: eager | After: bucketed graphs | Change |
|---|---|---|---|
| GPU utilization | 30% | 85% | +55 pts |
| SM occupancy (decode) | ~22% | ~22% | unchanged |
| Kernel launches / step | ~1,800 | 1 | 1,800× fewer |
| Host overhead / step | 9.0 ms | 0.15 ms | 60× less |
| GPU time / step | 3.1 ms | 3.1 ms | unchanged |
| Step time | ~9.4 ms | ~3.2 ms | 2.9× faster |
| p50 latency / request | 95 ms | 42 ms | 2.3× |
| p99 latency / request | 180 ms | 70 ms | 2.6× |
| Throughput | 180 req/s | 520 req/s | 2.9× |
| Graph memory (8 buckets) | 0 GB | 4.5 GB (2.9 shared) | +4.5 GB |
| Cost per million requests | ~\$5.7 | ~\$2.0 | 2.9× cheaper |

Two honesty notes the table makes explicit. **Occupancy did not change** — graphs do not touch how well each kernel fills the SMs; they remove the gaps *between* kernels. A memory-bound decode step has low occupancy before and after, because that is a property of the kernels, not the launch path. If you want occupancy up you need better kernels (fusion, bigger tiles), a different fix entirely. **GPU time per step did not change** either, for the same reason. Every win in the table traces to deleting host overhead, and the utilization and throughput moved precisely as much as the host overhead was inflating the step. The cost-per-request fell because you are renting the same A100 at the same roughly \$3–4 per hour but serving 2.9× the requests through it.

#### Worked example: reading the two profiles side by side

Put the two `cudaLaunchKernel` lines next to each other, because that single row is the whole diagnosis and the whole verification. Before: 71% of CPU, 28,800 calls, and a total CPU time (200 ms) four times the CUDA time (49.6 ms) — an unambiguous launch-bound service. After: the row is absent from the top six, replaced by 16 `cudaGraphLaunch` calls, with CPU time (51 ms) now within 4% of CUDA time (49 ms). You did not have to trust the utilization dashboard or the throughput number; the profiler *shows* the mechanism changing — 28,800 launches becoming 16 replays — which is the difference between "the number went up" and "I know why the number went up." That is the profile → hypothesize → fix → re-measure loop closing on itself, and it is the only evidence that survives a code review.

## Coexisting with continuous batching

A real LLM server does not process one fixed batch; a continuous-batching scheduler assembles a new batch every step from whatever requests are in flight, admitting new ones and retiring finished ones. That sounds fatal to graphs — if the batch changes every step, how can the shape be static? The resolution is the same bucketing, applied at the scheduler boundary, and it is worth being precise because it is how production stacks actually run.

The scheduler's job each step is to decide *which* requests run; once it has decided, the batch for that step has a concrete size, and that size is what you bucket. The scheduler assembles the running set, rounds the count up to the nearest batch bucket (padding the batch with inert slots, exactly as we padded batch 3 to 4), rounds the max KV length in the batch up to the nearest sequence bucket, and replays that bucket's graph. The next step may have a different count — a request finished, two were admitted — so it may route to a different bucket, but each individual step is a static-shape replay. The graph is captured per *step shape*, not per *request*, and continuous batching only changes which step shape you land on, which is precisely what the router already handles.

This is why the two techniques compose rather than conflict, and it is the design in vLLM and TensorRT-LLM: continuous batching maximizes how *full* each step's batch is (throughput), and CUDA graphs make each step's launch *free* (utilization). The [continuous-batching and PagedAttention post](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention) covers the scheduler side of this in depth, and the [kernel-fusion, CUDA-graphs, and torch.compile serving post](/blog/machine-learning/model-serving/kernel-fusion-cuda-graphs-torch-compile) covers how the graphed step slots into the larger serving stack. The one interaction to watch: PagedAttention's KV cache lives in a pool the graph reads through fixed block-table pointers, so the block table must itself be a static tensor you copy the current step's block indices into before replay — another instance of the copy-into-static-buffer discipline, applied to the KV addressing rather than the tokens.

## Stress tests and failure modes

A fix you have not stress-tested is a fix you do not understand. Push the bucketed service into its corners and watch what breaks, because these are the pages you will get at 3 a.m.

**A batch size with no bucket.** A burst assembles a batch of 20; the largest bucket is 16. `_bucket_for` returns `None`, the fallback splits it into two batches of 10, each of which routes to the batch-16 bucket (padding 10 to 16, 37% batch waste each) and replays. Latency for that request roughly doubles versus a single replay, and its padding waste is high, but it *serves* — no crash, a bounded slowdown, and it shows up in the trace as two replays where the others have one. The alternative, adding a batch-32 bucket, costs another large pool for a shape that appears in under 1% of steps; splitting is usually the right call. Measure how often the split fires; if it climbs past a few percent your scheduler is assembling batches your bucket set does not cover, and you re-fit the buckets.

**A burst of one shape.** Traffic suddenly concentrates on a single shape — every request is batch 1, length 130 (an interactive-only spike). Every step routes to B1/S256 and replays the same graph. This is the *good* case: the hot bucket is exactly what graphs excel at, the other seven buckets sit idle in memory but cost nothing to hold, and utilization is highest here because there is zero routing variance. The only waste is the memory of the seven unused buckets, which you already paid for at startup. A concentrated burst is not a failure mode for bucketing; it is the mode bucketing was built for.

**Memory pressure from too many graphs.** Someone, chasing the last few percent of padding, grows the bucket set from 8 to 24. Startup latency triples to 35 seconds, resident graph memory climbs past 12 GB, and under load the KV cache cannot grow into the fragmented free space between graph pools — the service starts rejecting requests at 60% reported memory use, the fragmentation failure. The fix is to walk it back: prune buckets by traffic share (drop every cell under 1% into a shared coarse bucket), share the memory pool across captures, and re-measure that padding barely rose. This is the direct cost of over-fine bucketing, and it is why the granularity matrix has memory in red on the fine row.

**A shape outside all buckets.** A single request arrives with a 900-token KV, past the 512 ceiling. It routes to the eager fallback: a full un-graphed decode step, ~1,800 launches, ~9 ms of host overhead — slow, but correct, for one step. If such requests are truly rare (the 0.4% tail from the histogram), the p99 absorbs it and you move on. If they are not rare — if the tail is fatter than the histogram suggested because traffic shifted — your p99 degrades and the signal is a rising count of eager-fallback steps in your metrics, which is your cue to add a 1024 sequence bucket and pay its memory. The lesson that generalizes: **the fallback path is your early-warning system.** Instrument it, alert on its rate, and let it tell you when your bucket set has drifted out of sync with your traffic.

## Case studies and real numbers

Three published results ground the numbers above in systems you can go read.

**vLLM's CUDA-graph decode.** vLLM captures CUDA graphs for the decode phase across a set of batch-size buckets (a `capture_sizes` list, defaulting to a spread like 1, 2, 4, 8 and then 16 up through a configured maximum), replaying the graph whose captured batch is the smallest that covers the current running batch — the exact route-and-pad design above, in production. Its documentation and issue history are candid about the trade this post derives: graph capture adds startup time and a fixed memory cost per captured size, `enforce_eager=True` disables graphs entirely for debugging or when memory is tight, and the capture sizes are tunable precisely because the right set depends on your traffic. The reported effect is the launch-overhead removal we measured — decode goes from host-bound to GPU-bound on small batches, where launch overhead dominates most.

**TensorRT-LLM and PyTorch's `reduce-overhead` mode.** TensorRT-LLM similarly uses CUDA graphs for the generation loop and bucketed batch sizes, and PyTorch's own `torch.compile(mode="reduce-overhead")` is, under the description in the PyTorch docs, "compile plus CUDA graphs" — Inductor generates fused kernels and then wraps the compiled region in a CUDA graph to erase the residual launch overhead. The PyTorch team's published `reduce-overhead` benchmarks show the largest wins on exactly the workloads this post targets: many small kernels where per-launch CPU cost, not GPU compute, is the wall. The later torch.compile track in this series covers how that mode composes graphs with fusion, and the [model-serving post on kernel fusion, CUDA graphs, and torch.compile](/blog/machine-learning/model-serving/kernel-fusion-cuda-graphs-torch-compile) shows it inside a full serving stack.

**The launch-overhead order of magnitude.** The headline numbers in this post rest on one measured constant: a CUDA kernel launch costs roughly 5–10 µs of CPU-side overhead, a figure consistent across NVIDIA's own profiling guidance and years of practitioner measurement. That constant is what makes 1,800 tiny kernels a 9 ms wall and one replay a rounding error. It is also why the technique's benefit is so predictable: multiply your per-step kernel count by ~5 µs, compare to your per-step GPU time, and you know before writing a line of capture code whether graphs will move your utilization or not.

## When to reach for this, and when not to

CUDA graphs in a serving loop are a specific fix for a specific waste. Reach for them when the profile shows a **launch-bound** service — `cudaLaunchKernel` dominating CPU time, CPU total far exceeding CUDA total, a decode step made of hundreds or thousands of small kernels, GPU utilization low while load is high. That is the case where deleting per-op launch overhead moves everything. And reach for bucketing specifically when your traffic is dynamic but *clusters* — a handful of common shapes carrying most requests, a thin tail you can fall back on.

Do not reach for them when any of the following holds, because you will spend the memory and startup cost for little or no return:

| Situation | Why graphs won't help | What to do instead |
|---|---|---|
| Compute-bound service | The wall is GPU time, not launch; graphs delete launch, which is already tiny | Optimize kernels: fusion, better tiling, bigger batches |
| Already at 85%+ util | Little host overhead left to remove; Amdahl caps the win near zero | Leave it; chase a different waste |
| Highly variable shapes, long tail | Coverage needs too many buckets; memory and startup explode | Bucket only the dense core; eager fallback for the tail, or use `torch.compile(dynamic=True)` |
| Rapidly-changing model / dev loop | Every code change invalidates every captured graph; re-capture cost dominates | Stay eager until the model stabilizes; graph at deploy |
| Prefill-dominated workload | Prefill shapes are genuinely arbitrary and prefill is compute-bound anyway | Graph decode only; leave prefill eager |

The decision is honest arithmetic, not enthusiasm. Multiply kernel count by launch cost, compare to GPU time. If launch is a small fraction of your step, graphs are a solution to a problem you do not have — and the memory, startup latency, padding waste, and capture-fragility they cost are pure downside. This is the same discipline the whole series preaches and the [capstone playbook](/blog/machine-learning/performance-engineering/the-performance-engineering-playbook) systematizes: name the waste from the profile first, then apply the fix that kills *that* waste, then re-measure. A service that is already GPU-bound does not need graphs; a service drowning in tiny launches needs almost nothing else. The full end-to-end transformation — a service taken from 30% to 85% by exactly this route — is the subject of the [war-story case study](/blog/machine-learning/performance-engineering/the-service-at-30-percent-gpu-util) later in the series, which stacks graphing on top of the other fixes.

Compare the three tools you might reach for on a launch-bound service, because they are not interchangeable:

| Approach | Kills launch overhead? | Handles dynamic shapes? | Extra cost |
|---|---|---|---|
| No graphs (eager) | No | Yes (natively) | none; but stays host-bound |
| Bucketed CUDA graphs | Yes (1,800 → 1) | Only via bucketing + padding | graph memory + startup + padding waste |
| `torch.compile(dynamic=True)` | Partially (fewer, fused kernels) | Yes, with guards + recompiles | compile time; still has some launch overhead |
| `compile(mode="reduce-overhead")` | Yes (compile + graphs) | Via dynamic bucketing internally | compile time + graph memory + shape guards |

The compiled `reduce-overhead` path is often the pragmatic default now: it fuses kernels (fewer to launch in the first place) *and* graphs them (the rest launch free), and it manages the bucketing and static buffers for you. Hand-rolled bucketed graphs are for when you need explicit control over exactly which shapes are captured and how memory is shared — a large-scale serving stack tuning its bucket set against a known traffic histogram, which is exactly the case this post has been walking.

## Key takeaways

- **Launch-bound is a measurable diagnosis, not a guess.** `cudaLaunchKernel` dominating CPU time with CPU total far above CUDA total is the signature; confirm it in the profiler before you reach for graphs, and confirm the launches became replays after.
- **Graph the decode step, not the whole request.** Decode's per-step shape is nearly static once batch and KV length are bucketed, and decode is where launch overhead bites; prefill is arbitrary-shaped and compute-bound, so leave it eager.
- **Bucketing converts shape variance into bounded padding.** Pre-capture one graph per (batch, sequence) bucket, route each request to the smallest bucket that fits, pad in, replay, unpad out.
- **Pick buckets from your traffic histogram, not round numbers.** Place boundaries just above dense regions so most requests pad little; coarse buckets waste compute, fine buckets waste memory and startup time — choose the point your traffic justifies.
- **Graphs remove launch overhead, nothing else.** Occupancy and per-kernel GPU time do not change; the entire win is deleting the gaps between kernels. A compute-bound service gains almost nothing.
- **Count the memory before you deploy.** Weights are shared across buckets, but each bucket adds static I/O plus a captured pool; too many buckets fragment HBM and starve the KV cache at deploy. Share the pool to cut the bill.
- **Warmup and capture belong at startup, never on the request path.** Capture is seconds of cost and it allocates; pay it once, keep the graphs resident, and mitigate cold-start latency with warm pods or priority capture.
- **Instrument the fallback.** The rate of out-of-range and split requests is your early warning that your bucket set has drifted from your traffic — alert on it.

## Further reading

- [Why your AI service wastes CPU and GPU](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) — the series intro: the four wastes and the profile → hypothesize → fix → measure loop this post lives inside.
- [The kernel-launch-overhead problem](/blog/machine-learning/performance-engineering/the-kernel-launch-overhead-problem) — where the `N_kernels × t_launch` law and the launch-bound signature come from.
- [CUDA graphs in PyTorch](/blog/machine-learning/performance-engineering/cuda-graphs-in-pytorch) — the API mechanics: `torch.cuda.graph`, `make_graphed_callables`, static buffers, graph pools.
- [CUDA graphs: gotchas and debugging](/blog/machine-learning/performance-engineering/cuda-graphs-gotchas-and-debugging) — dynamic-shape breaks, allocator interactions, and the padding-contamination "garbage output" bug.
- [The service at 30% GPU utilization](/blog/machine-learning/performance-engineering/the-service-at-30-percent-gpu-util) — the full war-story case study that stacks graphing with the other fixes end to end.
- [Continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention) — the scheduler side of how graphed decode steps compose with dynamic batching.
- [Kernel fusion, CUDA graphs, and torch.compile for serving](/blog/machine-learning/model-serving/kernel-fusion-cuda-graphs-torch-compile) — how the graphed step slots into a full serving stack.
- [The performance engineering playbook](/blog/machine-learning/performance-engineering/the-performance-engineering-playbook) — the capstone decision tree: symptom → profiler → cause → fix.
- PyTorch CUDA Graphs docs and the `torch.compile(mode="reduce-overhead")` tutorial — the primary sources for capture/replay semantics and the compile-plus-graphs mode.

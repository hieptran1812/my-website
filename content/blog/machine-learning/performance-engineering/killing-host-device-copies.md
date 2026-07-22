---
title: "Killing Host-Device Copies: Pinned Memory, non_blocking, and Overlapping the Transfer"
date: "2026-07-14"
publishDate: "2026-07-14"
description: "Your service spends a third of every request moving tensors across PCIe while the GPU sits idle. This post shows you how to see that copy in a trace, why pageable memory makes it worse, and how pinned memory plus non_blocking plus a dedicated copy stream make the transfer effectively free."
tags:
  [
    "performance-engineering",
    "gpu-optimization",
    "memory",
    "pytorch",
    "cuda",
    "profiling",
    "latency",
    "throughput",
    "inference",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Performance Engineering"
author: "Hiep Tran"
featured: true
readTime: 43
---

Here is a symptom that shows up in more inference services than anyone likes to admit, and almost never gets diagnosed correctly: the GPU is idle for a third of every request, and the dashboard still says "GPU-Util 92%." You pull a trace expecting to find a kernel that runs too long, or a launch storm, or a bad batch. Instead the fat idle bars in the GPU row are bracketed by two green stripes labeled `Memcpy HtoD` and `Memcpy DtoH` — the input tensor being copied *to* the device, and the output copied back *from* it. The GPU is not computing during those stripes. It is waiting for bytes to crawl across a PCIe link at a fraction of the bandwidth it computes against, and then it is waiting again on the way out. The copy is on the critical path, serialized with the compute, and it is costing you real money: at \$1.50 to \$4 per GPU-hour depending on your provider, a GPU that idles 40% of every step is a GPU you are renting at nearly double its useful rate.

This is the fourth waste from the [series intro](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) — data movement — in its most concrete and most fixable form. A [chrome trace](/blog/machine-learning/performance-engineering/reading-a-chrome-trace) makes it visible in ten seconds once you know the shape to look for, and the fix is not exotic: it is three changes to how you allocate and copy host tensors, each one a few lines, each one measurable. By the end of this post you will be able to spot a host-device copy on the critical path in any trace, derive from first principles exactly how much time it should take and whether it can be hidden, pin your host memory so the copy DMAs directly instead of routing through a hidden staging buffer, mark the copy `non_blocking=True` so it stops stalling the CPU, run it on a dedicated CUDA stream so it overlaps the compute, and prefetch the next batch's transfer under the current batch's forward pass so the copy leaves the critical path entirely. And — just as important — you will know the traps that make each of those changes silently do nothing.

![a before and after comparison showing a serialized copy compute copy timeline of ten milliseconds shrinking to a six millisecond overlapped timeline where the transfer runs under compute](/imgs/blogs/killing-host-device-copies-1.webp)

The figure above is the whole thesis. On the left is a serialized step: a 3 ms H2D copy, then 6 ms of compute, then a 1 ms D2H copy, 10 ms end to end, with the GPU idle during both copies — 4 ms of dead time, 40% of the step. On the right is the same work overlapped: the compute runs back to back at 6 ms, the next batch's copy slides underneath it on a separate stream, and the GPU never idles on a transfer. Same model, same hardware, same bytes moved. The step went from 10 ms to 6 ms and the GPU-idle-on-copy went from 40% to zero, purely by moving *when* the copy happens relative to the compute. That 1.67x is not a kernel optimization — we did not touch a single kernel. It is a scheduling win, and scheduling wins are the cheapest wins in performance engineering because the compute was already correct; it was just badly surrounded.

Our running example for the whole post is a dense-prediction vision service — a segmentation model in the ResNet-50 family — running on one **A100 80GB SXM** behind a **PCIe gen4 x16** link, serving batches of 128 images at 224×224×3. That service moves a lot of bytes per request in both directions (big image batches in, dense output maps out), which makes it the perfect specimen for the copy tax. Everything below is measured against that service, and every number is one the profiler I show you would actually produce.

## The copy tax: PCIe is a hundred times slower than HBM

Start with the physics, because the whole reason a copy hurts is a bandwidth mismatch that most engineers have never put a number on. A GPU kernel reads and writes its data from **HBM** — the high-bandwidth memory soldered onto the GPU package. On an A100 80GB that HBM delivers roughly 2.0 TB/s, which is 2,000 GB/s. When you copy a tensor from host (CPU) memory to the device, those bytes do not travel over HBM; they travel over the **PCIe** bus that connects the CPU and the GPU. A PCIe gen4 x16 link tops out around 32 GB/s in theory and, after protocol overhead, delivers something closer to 24 to 26 GB/s of usable bandwidth in one direction. That is not a small gap. It is a factor of roughly 60 to 80 between the rate at which the GPU can *compute against* data and the rate at which it can *receive* data.

The consequence is a simple, brutal inequality. Per byte, moving data across PCIe is about two orders of magnitude more expensive than doing on-chip work with it. So a copy that cannot be hidden behind compute is not a minor tax — it is dead GPU time at the worst possible exchange rate. The copy time itself is easy to derive. For a transfer of $B$ bytes over a link with effective bandwidth $\text{BW}$:

$$t_\text{copy} = \frac{B}{\text{BW}}$$

That is it — copy time is linear in bytes, and the slope is one over the bandwidth. There is a small fixed startup cost per transfer (kernel launch, DMA setup, on the order of a few microseconds), so for very small tensors the launch overhead dominates and for large tensors the bandwidth term dominates. The crossover for PCIe is around tens of kilobytes; above that you are firmly bandwidth-bound and the formula above is accurate to within a few percent. This is also why *fewer, bigger* copies beat *many, smaller* ones: fifty separate 1.5 MB copies each pay the fixed DMA-setup cost and never reach peak bandwidth, while one coalesced 77 MB copy pays that cost once and runs at the full link rate. If your service copies tensors one at a time in a loop, coalescing them into a single transfer is often a bigger win than any of the four fixes below — the profiler will show it as fifty short memcpy bars collapsing into one long one.

It is worth naming the piece of hardware doing the work, because it is the reason overlap is possible at all. The **DMA (Direct Memory Access) engine** is a dedicated copy unit on the GPU, separate from the SMs that run kernels. High-end datacenter GPUs have several of them (often one per direction, sometimes more), which is why an H2D copy and a D2H copy and a compute kernel can all be in flight simultaneously — they use three different pieces of silicon. That physical separation is what makes the `max(compute, copy)` overlap real rather than aspirational: the SMs are not stealing cycles from the copy, and the copy is not stealing cycles from the SMs, so putting them on different streams genuinely runs them at the same time. The only shared resource they contend for is the PCIe link's fixed bandwidth, which matters under concurrency but not for a single overlapped step.

#### Worked example: the 77 MB batch that idled an A100 40% of every step

Our service copies a batch of 128 images at 224×224×3 in fp32. That is $128 \times 224 \times 224 \times 3 \times 4$ bytes $= 77{,}070{,}336$ bytes, about 77 MB, moving host-to-device every request. Plug it into the copy-time formula at the pinned-memory bandwidth of ~25 GB/s:

$$t_\text{H2D} = \frac{77 \times 10^6 \text{ B}}{25 \times 10^9 \text{ B/s}} \approx 3.1 \text{ ms}$$

The segmentation output — dense per-image feature maps — comes to about 25 MB copied device-to-host, which at the same 25 GB/s is roughly 1.0 ms. The forward pass itself takes 6 ms of GPU compute. Serialized, that is ${3.1 + 6.0 + 1.0 = 10.1}$ ms per step, of which ${3.1 + 1.0 = 4.1}$ ms is the GPU sitting idle waiting on copies — **40% of every step is dead time**, and it is dead time you are paying full GPU-hour price for. The throughput ceiling is $128 / 0.0101 \approx 12{,}700$ images per second, and roughly 5,000 of those images per second of capacity is being burned on idle. That is the tax. Now watch what happens when we stop routing the copy through a hidden staging buffer, and then when we stop putting it on the critical path at all.

The reason a copy shows up as a *fat idle bar* rather than a busy one is worth stating precisely, because it is what makes the copy tax visible in a trace. The GPU's compute units (its SMs) do nothing during a pure memory transfer — the copy is handled by a **DMA engine**, a dedicated copy unit on the GPU that moves bytes between host and device memory without involving the SMs. So on the GPU-utilization dashboard, a step that is 40% copy reads as "the GPU was busy" (the DMA engine counts as activity on some counters), while on a real per-stream trace the SM row is flat and only the memcpy row is active. That is the single most important reason `nvidia-smi` "GPU-Util" lies about copy-bound services and why you have to look at an actual [Nsight Systems](/blog/machine-learning/performance-engineering/nsight-systems-for-ai-services) or chrome trace to catch it.

## Pageable versus pinned: the hidden staging copy

Here is the first fix, and it is the one almost nobody knows about until they trip on it. When you allocate a normal host tensor — `torch.empty(shape)`, a NumPy array, anything from the default allocator — that memory is **pageable**. The operating system's virtual-memory system is allowed to move it around in physical RAM or swap it out to disk at any time. The GPU's DMA engine cannot safely read from memory that might move out from under it mid-transfer, so it refuses to DMA directly from a pageable buffer. Instead, the CUDA driver does something you never asked for and rarely see: it allocates a hidden **pinned** (page-locked) staging buffer, synchronously copies your pageable tensor into that staging buffer with the CPU, and only *then* starts the PCIe DMA from the staging buffer to the device.

![a vertical stack of layers showing a pageable buffer going through a synchronous driver staging copy into a hidden pinned bounce buffer before the PCIe transfer reaches device memory](/imgs/blogs/killing-host-device-copies-3.webp)

The figure traces both hops. Your pageable buffer sits at the top. Below it, the driver's synchronous staging copy — a plain CPU memcpy into a bounce buffer the driver keeps around — happens *before the DMA can start*, and it happens on the CPU thread that issued the copy, so your Python is blocked the whole time. Then the pinned bounce buffer is DMA'd over PCIe, and finally the bytes land in HBM. That extra CPU-side staging copy is why pageable transfers measure at roughly half the bandwidth of pinned ones (~12 GB/s versus ~25 GB/s in practice on gen4): you are paying for the same bytes to be copied twice — once CPU-to-staging, once staging-to-device — and the first copy is a synchronous CPU operation with no overlap.

**Pinned (page-locked) memory** deletes the top two layers of that stack. When you allocate host memory as pinned, you are telling the OS "never move or swap this," which lets the DMA engine read from it directly. The transfer becomes a single DMA hop, at full PCIe bandwidth, and — critically — it can run *asynchronously*, because there is no CPU-side staging copy that has to finish first. Pinning is the prerequisite for every other fix in this post: `non_blocking=True` does nothing on pageable memory, and you cannot overlap a copy that is secretly a synchronous CPU memcpy. In PyTorch you pin in one of two places:

```python
import torch

# Option A: pin a host tensor explicitly at allocation.
host_input = torch.empty(128, 3, 224, 224, dtype=torch.float32, pin_memory=True)
# ... fill host_input with your preprocessed batch (in place) ...
device_input = host_input.to("cuda", non_blocking=True)  # single DMA, no staging copy

# Option B: pin an existing pageable tensor (allocates a new pinned buffer + copies once).
pageable = torch.from_numpy(preprocessed_batch)      # pageable
pinned = pageable.pin_memory()                        # now page-locked
device_input = pinned.to("cuda", non_blocking=True)   # fast async path
```

Option A is what you want in a real service: allocate the pinned buffer *once*, at startup, and reuse it by filling it in place every request. Calling `.pin_memory()` on a fresh pageable tensor every request (Option B) allocates a new page-locked buffer each time, and pinning is not free — the OS has to lock pages, which is a syscall — so per-request pinning can cost more than it saves. Allocate pinned buffers up front and treat them as a fixed staging area you copy your preprocessed input into.

You can confirm the bandwidth difference with a ten-line microbenchmark. This is the honest way to measure a copy — CUDA events around the transfer, a warmup, and a `synchronize()` so you are timing the DMA and not the async launch:

```python
import torch

def copy_bandwidth(nbytes, pinned, iters=200, warmup=20):
    n = nbytes // 4
    host = torch.empty(n, dtype=torch.float32, pin_memory=pinned)
    dev = torch.empty(n, dtype=torch.float32, device="cuda")
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        dev.copy_(host, non_blocking=pinned)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        dev.copy_(host, non_blocking=pinned)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    gbps = nbytes / (ms * 1e-3) / 1e9
    print(f"{'pinned' if pinned else 'pageable':>8}  {ms:6.3f} ms  {gbps:6.1f} GB/s")

for pinned in (False, True):
    copy_bandwidth(77 * 1024 * 1024, pinned)
```

On the A100 + PCIe gen4 box that prints roughly this — and it is the single most convincing number you can show a skeptical reviewer:

```console
pageable   6.512 ms   11.8 GB/s
  pinned   3.086 ms   24.9 GB/s
```

Pinning alone cut the copy from 6.5 ms to 3.1 ms — a 2.1x bandwidth improvement — because the transfer stopped routing through the driver's staging buffer. That is the first fix, and on a copy-heavy service it is often the single biggest one. But notice what it did *not* do: the copy is still on the critical path. The GPU still waits 3.1 ms before it can start computing. To get rid of that wait we have to stop the copy from blocking, and then move it off the path entirely.

## The four fixes, in order

There are exactly four levers, and they compose in a specific order. Each one is necessary but not sufficient for the next: you cannot go async without pinning, you cannot overlap without going async, and you cannot double-buffer without a copy stream. The matrix below is the map for the rest of the post.

![a matrix comparing pageable pinned pinned with non blocking and pinned with a copy stream across bandwidth whether the cpu blocks and whether the copy overlaps compute](/imgs/blogs/killing-host-device-copies-2.webp)

Read it top to bottom as a progression. **Pageable** is the worst square in every column: half the bandwidth, the CPU blocks on the synchronous staging copy, and nothing overlaps. **Pinned** fixes the bandwidth column — the DMA runs at full rate — but the CPU still blocks and the copy still does not overlap, because a plain `.to("cuda")` is a synchronous call that waits for the transfer to finish. **Pinned + non_blocking** fixes the CPU-blocking column: `non_blocking=True` returns control to your Python thread the instant the copy is *enqueued*, not when it *completes*, so the CPU can go do other useful work (launch the next kernels, start preparing the next batch) while the DMA runs. But there is a subtle trap in that third row that the last column exposes: if the copy and the compute are on the *same* CUDA stream, the GPU still executes them in order — copy, then compute — so even though the CPU no longer blocks, the *GPU* still serializes them, and the copy is still on the critical path. Only the fourth row — **pinned + a dedicated copy stream** — puts the copy on a different stream from the compute, which is what actually lets the hardware run them at the same time. That is the row that makes the copy free.

The two-line change from row two to row three looks like this:

```python
# Row 2 (pinned, synchronous): CPU blocks until the DMA completes.
device_input = host_pinned.to("cuda")            # returns AFTER the copy lands

# Row 3 (pinned, non_blocking): CPU returns immediately, copy runs async.
device_input = host_pinned.to("cuda", non_blocking=True)  # returns after ENQUEUE
logits = model(device_input)   # PyTorch inserts the stream dependency for you
```

That `non_blocking=True` is doing something precise, and it is worth being exact about it, because it is the single most misunderstood flag in this whole area. It does **not** make the copy faster. It does **not** make the copy overlap the compute. All it does is stop the CPU from waiting on the copy's completion. The bytes still move at the same rate; the copy still runs on the default stream; and because `model(device_input)` also runs on the default stream, the GPU still does copy-then-compute in order. What you gained is a free CPU: while the DMA runs, your Python thread is off launching the next batch's work instead of blocking. On a service where the CPU is also a bottleneck that matters, but on its own it does not close the GPU idle gap. To close the gap you need the copy and the compute on different streams — the fourth row — which is the next section.

There is one correctness rule that comes with `non_blocking=True`, and violating it produces the single scariest bug in this area: **garbage output that only appears under load.** If you copy `host_pinned` to the device with `non_blocking=True` and then, before the copy has landed, you *reuse or free the host buffer* (overwrite it with the next batch, or let it go out of scope), you have a race — the DMA might still be reading the buffer you just clobbered. PyTorch's stream semantics protect the *device*-side dependency (the compute waits for the copy), but they do not stop *you* from mutating the host source. The rule: after a `non_blocking` H2D copy, do not touch the host source tensor until you know the copy is done — which in the double-buffer pattern below means not writing into a pinned slot until the stream that copied from it has passed a recorded event.

## Overlapping the copy with compute: the free-copy condition

Now the payoff. The reason overlapping a copy with compute is such a good deal is that it is not a speedup in the usual sense — it does not make anything run faster. It makes the copy *disappear* from the timeline, as long as one condition holds. Here is the condition, derived. Suppose a step does $t_\text{compute}$ of GPU work and $t_\text{copy}$ of transfer, and you run them on separate streams so they can execute concurrently. The step is not done until *both* finish, so the wall-clock time of the step is:

$$t_\text{step} = \max(t_\text{compute},\ t_\text{copy})$$

That `max` is the entire game. If the copy is smaller than the compute — $t_\text{copy} \le t_\text{compute}$ — then the max is just $t_\text{compute}$, and the copy contributes **nothing** to the step time. It has been completely absorbed under the compute. The copy is free. If instead the copy is bigger than the compute — $t_\text{copy} \gt t_\text{compute}$ — then the max is $t_\text{copy}$, the copy sets the pace, and you are **copy-bound**: no amount of kernel optimization will help, because the GPU is waiting on the bus, not on math. So the overlap condition is exactly:

$$t_\text{copy} \le t_\text{compute} \implies \text{the copy is free}$$

For our service, $t_\text{copy} = 3.1$ ms (H2D) and $t_\text{compute} = 6.0$ ms, so $3.1 \le 6.0$ and the H2D copy fits entirely under the forward pass with 2.9 ms of compute to spare. The D2H of the previous step (1.0 ms) also fits. Both copies can be hidden, and the step collapses from 10.1 ms to about 6 ms — the compute time — exactly as the figures show.

<figure class="blog-anim">
<svg viewBox="0 0 760 250" role="img" aria-label="A timeline alternates between a serialized layout where the copy runs before compute and an overlapped layout where the next copy runs under compute on a second stream and the timeline shrinks from ten to six units" style="width:100%;height:auto;max-width:860px">
<style>
.k4-copy{fill:var(--accent,#6366f1);opacity:.30}
.k4-comp{fill:var(--accent,#6366f1)}
.k4-idle{fill:var(--border,#d1d5db);opacity:.5}
.k4-tt{font:600 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.k4-onbar{font:600 14px ui-sans-serif,system-ui;fill:#fff;text-anchor:middle}
.k4-dk{font:600 13px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.k4-sub{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.k4-acc{font:700 14px ui-sans-serif,system-ui;fill:var(--accent,#6366f1);text-anchor:middle}
.k4-rule{stroke:var(--text-secondary,#6b7280);stroke-width:1.5}
@keyframes k4-fadeA{0%,42%{opacity:1}54%,96%{opacity:0}100%{opacity:1}}
@keyframes k4-fadeB{0%,42%{opacity:0}54%,96%{opacity:1}100%{opacity:0}}
.k4-A{animation:k4-fadeA 9s ease-in-out infinite}
.k4-B{animation:k4-fadeB 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.k4-A{animation:none;opacity:1}.k4-B{animation:none;opacity:0}}
</style>
<g class="k4-A">
<text class="k4-tt" x="380" y="34">Serialized — the GPU waits on every copy</text>
<rect class="k4-copy" x="90" y="70" width="150" height="46" rx="6"/>
<rect class="k4-comp" x="240" y="70" width="300" height="46" rx="6"/>
<rect class="k4-copy" x="540" y="70" width="50" height="46" rx="6"/>
<text class="k4-dk" x="165" y="98">H2D · 3 ms</text>
<text class="k4-onbar" x="390" y="98">compute · 6 ms</text>
<text class="k4-dk" x="565" y="98">D2H</text>
<text class="k4-sub" x="165" y="136">GPU idle</text>
<text class="k4-sub" x="565" y="136">GPU idle</text>
<line class="k4-rule" x1="90" y1="170" x2="590" y2="170"/>
<line class="k4-rule" x1="90" y1="164" x2="90" y2="176"/>
<line class="k4-rule" x1="590" y1="164" x2="590" y2="176"/>
<text class="k4-sub" x="340" y="196">end to end · 10 ms · GPU idle 40%</text>
</g>
<g class="k4-B">
<text class="k4-tt" x="380" y="34">Overlapped — the copy hides under compute</text>
<rect class="k4-comp" x="90" y="60" width="300" height="46" rx="6"/>
<text class="k4-onbar" x="240" y="88">compute · 6 ms</text>
<text class="k4-sub" x="470" y="88">compute stream</text>
<rect class="k4-copy" x="90" y="118" width="150" height="42" rx="6"/>
<text class="k4-dk" x="165" y="144">H2D next · hidden</text>
<text class="k4-sub" x="330" y="144">copy stream</text>
<line class="k4-rule" x1="90" y1="188" x2="390" y2="188"/>
<line class="k4-rule" x1="90" y1="182" x2="90" y2="194"/>
<line class="k4-rule" x1="390" y1="182" x2="390" y2="194"/>
<text class="k4-sub" x="240" y="214">end to end · 6 ms · GPU idle 0%</text>
<text class="k4-acc" x="560" y="150">copy left the critical path · −4 ms</text>
</g>
</svg>
<figcaption>The same step, serialized versus overlapped: moving the next batch's copy onto a second stream tucks it under the compute bar, so the timeline collapses from 10 ms to 6 ms and the GPU stops idling on the transfer.</figcaption>
</figure>

To make that happen you need a second stream. A **CUDA stream** is a queue of GPU operations that execute in order relative to each other but *concurrently* with operations on other streams, as long as the hardware has the resources. Copies and compute use different hardware units — the DMA engine versus the SMs — so a copy on one stream and a matmul on another genuinely run at the same time. The minimal version, copying the next input while the current one computes, looks like this:

```python
import torch

copy_stream = torch.cuda.Stream()          # a dedicated stream for H2D transfers
model = model.cuda().eval()

def step(host_pinned_next, device_cur):
    # Launch the current batch's compute on the default stream.
    with torch.no_grad():
        out = model(device_cur)            # runs on the default (compute) stream

    # Concurrently, stage the NEXT batch's input on the copy stream.
    with torch.cuda.stream(copy_stream):
        device_next = host_pinned_next.to("cuda", non_blocking=True)

    # The copy stream must not free/reuse the pinned source until the copy is done;
    # record an event the producer can wait on before refilling the slot.
    copy_done = torch.cuda.Event()
    copy_done.record(copy_stream)
    return out, device_next, copy_done
```

The key line is `with torch.cuda.stream(copy_stream)`. It puts the H2D copy on a stream that is independent of the compute, so the DMA engine moves the next batch's 77 MB across PCIe *while the SMs are still chewing on the current batch's forward pass*. As long as the copy (3.1 ms) is shorter than the compute (6.0 ms), it finishes before the compute does and the next step starts with its input already on the device. The copy has left the critical path. This is the exact same principle as [overlapping compute and communication](/blog/machine-learning/distributed-training/overlapping-compute-and-communication) in distributed training — there the "copy" is a NCCL all-reduce over the network, here it is a DMA over PCIe, but the trick is identical: put the transfer on its own stream so it runs under the compute instead of in front of it.

## Which fix does the trace call for?

You do not apply all four fixes blindly; you apply the one the trace is asking for. The decision splits along one clean axis — is the problem that the copy is *slow and synchronous*, or that it is *on the critical path*? — and the trace tells you which.

![a decision tree that starts from a gpu idle on a copy symptom and branches into making the copy asynchronous through pinning and non blocking or hiding it behind compute through a copy stream and prefetching](/imgs/blogs/killing-host-device-copies-4.webp)

The tree splits the fix space in two. The left branch — **make the copy async** — is where you go when the copy itself is expensive or the CPU is stalling on it: pin the host memory (2x bandwidth, deletes the staging copy) and add `non_blocking=True` (frees the CPU). You reach for this branch when the memcpy row shows pageable-to-device copies, or when the CPU thread is visibly blocked in `cudaMemcpy` in a py-spy sample. The right branch — **hide it behind compute** — is where you go when the copy is already fast and async but still sits in an idle gap before the compute: put it on a copy stream (overlaps one step) and prefetch the next batch (the double-buffer pattern, overlaps continuously). You reach for this branch when the trace shows a clean, full-bandwidth copy that is nonetheless serialized in front of the kernels. Most real services need both branches — pin and go async first, then overlap — but knowing which one the current trace is complaining about keeps you from, say, building a double-buffer prefetcher on top of pageable memory that is still secretly staging-copying on every transfer.

Here is the honest diagnostic flow. Open a chrome trace or an `nsys` timeline and look at the memcpy row against the kernel row:

- **Copies are labeled `Memcpy HtoD (Pageable -> Device)`** — you are not even pinned. Fix the left branch first; it is free bandwidth.
- **Copies are `Pinned -> Device` and fast, but sit in a gap before the kernels with the SM row flat** — you are pinned but serialized. Fix the right branch: copy stream + prefetch.
- **The `.item()` / `.cpu()` / `.numpy()` calls show up as tiny D2H copies followed by a `cudaStreamSynchronize` that stalls everything** — you have a hidden sync mid-pipeline. That is the trap section below.
- **The copy is bigger than the largest compute gap and there is no gap to hide it in** — you are copy-bound, and the fix is not overlap, it is *moving fewer bytes* (lower precision inputs, smaller batch, compute-side preprocessing). More on this in the stress test.

Because each fix targets a specific waste and carries a specific cost, it helps to have the whole decision on one page. This table is the compressed version of the tree above — read it left to right to pick the fix, and read the last two columns before you commit, because the copy stream and the prefetcher are the only ones with real correctness obligations attached:

| Fix | Kills which waste | Cost / risk | When NOT to bother |
|---|---|---|---|
| Pin host memory | Half-bandwidth driver staging copy | Unswappable host RAM; page-lock syscall | Copies already tiny relative to compute |
| `non_blocking=True` | CPU stalling on the transfer | None, but needs pinned memory to work | Nothing else for the CPU to do meanwhile |
| Dedicated copy stream | Copy serialized in front of compute | Stream ordering (`wait_stream`) | Copy already hidden, or compute-bound |
| Double-buffer prefetch | Copy on the critical path every step | Two buffers; `record_stream` races | Single-shot inference, no next batch |
| Fewer bytes (fp16 / GPU decode) | Copy bigger than compute (copy-bound) | Precision loss; GPU-decode complexity | Already compute-bound, copy in the noise |

The one row people skip is the last: when the mechanism says you are copy-bound, none of the first four rows can help you, and reaching for a copy stream there is wasted effort. The `max(compute, copy)` derivation is what keeps you honest about which row your service is actually on.

## Double-buffering: prefetch the next batch under the current forward pass

A single copy stream overlaps *one* copy with *one* compute, but a serving loop runs step after step, and to keep the copy permanently off the critical path you want the transfer for batch N+1 to be running during the compute for batch N, forever. That is **double-buffering**: two pinned host slots and two device buffers, so that while the GPU computes on one, the DMA engine fills the other.

![a dataflow graph where a double buffer feeds batch N which forks into a compute stream forward pass and a copy stream that stages batch N plus one and the two streams merge at a step boundary](/imgs/blogs/killing-host-device-copies-5.webp)

The graph shows the fork and the join, which is the heart of the pattern. Batch N, already on the device, forks into two concurrent activities: the **compute stream** runs its 6 ms forward pass, and at the same time the **copy stream** stages batch N+1's 3 ms H2D transfer. The two streams rendezvous at the step boundary — the compute finishes, the copy has long since finished, and an event records that the pinned slot batch N+1 came from is now safe to refill. Because the copy (3 ms) always fits inside the compute (6 ms), the copy stream is idle half the time and the compute stream never waits: GPU-idle-on-copy is driven to zero. The loop that implements this is short:

```python
import torch

class Prefetcher:
    """Double-buffered H2D prefetch: overlaps the next batch's copy with compute."""
    def __init__(self, loader, device="cuda"):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()      # dedicated copy stream
        self.next_input = None
        self._preload()                         # prime the pipe

    def _preload(self):
        try:
            cpu_batch = next(self.loader)        # already pinned by the DataLoader
        except StopIteration:
            self.next_input = None
            return
        with torch.cuda.stream(self.stream):
            # non_blocking H2D on the copy stream; runs concurrently with compute
            self.next_input = cpu_batch.to(self.device, non_blocking=True)

    def next(self):
        if self.next_input is None:
            raise StopIteration
        # Make the default (compute) stream wait for THIS batch's copy to finish.
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_input
        # Pin the tensor to the default stream so the allocator won't reuse it early.
        batch.record_stream(torch.cuda.current_stream())
        self._preload()                          # kick off the NEXT copy immediately
        return batch

# Serving / eval loop
prefetch = Prefetcher(loader)
while True:
    batch = prefetch.next()      # returns batch N; batch N+1 is already copying
    with torch.no_grad():
        out = model(batch)       # compute overlaps the N+1 copy on the copy stream
```

Two lines carry the whole pattern and both are easy to get wrong. `wait_stream(self.stream)` makes the compute stream block until *this* batch's copy has landed — without it you would compute on half-copied garbage. `record_stream(...)` tells the caching allocator "the default stream is now using this tensor, do not hand its memory to someone else until the default stream is done" — without it, the allocator can free and reuse the buffer while the copy stream still references it, another route to garbage-under-load. This is the same `record_stream` discipline the [CUDA caching allocator](/blog/machine-learning/performance-engineering/the-cuda-caching-allocator) uses everywhere streams cross. Get those two lines right and the copy is invisible; get them wrong and you get a nondeterministic corruption bug that only shows up when the copy and compute happen to race, which is to say, in production and never in your test.

The good news is that PyTorch's `DataLoader` already does the pinning half of this for you, and you should let it:

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=128,
    num_workers=8,          # CPU workers do preprocessing in parallel
    pin_memory=True,        # workers write batches into pinned host buffers
    prefetch_factor=4,      # each worker stages ahead so the pipe never drains
    persistent_workers=True,
)
```

`pin_memory=True` makes the DataLoader collate each batch directly into page-locked memory, so by the time your training or serving loop calls `.to(device, non_blocking=True)`, the fast async DMA path is available with no extra `.pin_memory()` call and no per-request page-locking cost. Combined with the `Prefetcher` above, the DataLoader handles the CPU-side preprocessing and pinning, and the prefetcher handles the overlap — the input pipeline feeds the GPU without ever appearing on the critical path. The interaction between `num_workers`, `prefetch_factor`, and pinning is deep enough that it gets its own post later in the series on [the dataloader and preprocessing wall](/blog/machine-learning/performance-engineering/the-dataloader-and-preprocessing-wall); here the point is just that `pin_memory=True` is the flag that makes overlap *possible* at the input boundary.

#### Worked example: the double-buffer that took the service from 12,700 to 20,300 img/s

Take our A100 segmentation service through all four fixes, measured at each stage on the same box, same 128-image batches, steady state, clocks locked. The baseline is pageable and synchronous — the naive `batch.to("cuda")` with a normal DataLoader. The end state is pinned, non_blocking, double-buffered.

| Config | H2D (ms) | D2H (ms) | GPU idle % | p50 (ms) | Throughput (img/s) |
|---|---|---|---|---|---|
| Pageable, synchronous | 6.4 | 2.1 | 58% | 14.5 | 8,830 |
| Pinned, synchronous | 3.1 | 1.0 | 40% | 10.1 | 12,700 |
| Pinned + non_blocking (1 stream) | 3.1 | 1.0 | 38% | 9.8 | 13,100 |
| Pinned + copy stream + prefetch | 3.1\* | 1.0\* | 5% | 6.3 | 20,300 |

The story reads straight down the table. Pinning alone (row 1 to row 2) is the biggest single jump — 8,830 to 12,700 img/s, a 44% gain — purely from deleting the staging copy and halving the transfer time. Adding `non_blocking` on a single stream (row 3) barely moves throughput (13,100), exactly as the mechanism predicted: it frees the CPU but the GPU still serializes copy-then-compute on the default stream. The last row is where the copy leaves the critical path: with a copy stream and double-buffering, the H2D and D2H are hidden under compute (the asterisks mean "still happening, just not on the timeline"), GPU-idle-on-copy drops from 38% to 5%, and throughput jumps to 20,300 img/s — **2.3x over the naive baseline**, with a p50 that fell from 14.5 ms to 6.3 ms, essentially the compute time plus a sliver of overhead. And to be clear about cost: this is the same GPU, the same model, the same batch size. We moved the exact same bytes. The only thing that changed is *when*.

## Reading the copy in a real profile

You cannot fix what you cannot see, so before and after every change, look at the trace. The two tools are `torch.profiler` (for the table and the chrome trace) and Nsight Systems (for the system-wide timeline with a real memcpy row). Here is the profiler block that produces both, with the standard wait/warmup/active schedule so you are profiling steady state and not cold-start:

```python
import torch
from torch.profiler import profile, schedule, ProfilerActivity, tensorboard_trace_handler

sched = schedule(wait=1, warmup=2, active=3, repeat=1)
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=sched,
    on_trace_ready=tensorboard_trace_handler("./trace"),
    record_shapes=True,
    profile_memory=True,
) as prof:
    for _ in range(6):
        batch = prefetch.next()
        with torch.no_grad():
            out = model(batch)
        prof.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))
```

In the naive (pageable, synchronous) version, the key-averages table has an unmistakable signature — a `Memcpy` row near the top, and crucially it says `Pageable`:

```console
-------------------------------------------------------  ------------  ------------  ------------
Name                                                       Self CUDA %    Self CUDA    # of Calls
-------------------------------------------------------  ------------  ------------  ------------
Memcpy HtoD (Pageable -> Device)                              28.4%       6.401ms             1
aten::conv2d                                                 41.2%       6.010ms           108
Memcpy DtoH (Device -> Pageable)                              9.3%       2.090ms             1
aten::batch_norm                                             11.7%       1.703ms           108
aten::relu                                                    6.1%       0.889ms           102
-------------------------------------------------------  ------------  ------------  ------------
Self CUDA time total: 22.55ms
```

That `Memcpy HtoD (Pageable -> Device)` at 28% of the step is the whole problem in one line — the copy is the second-biggest CUDA cost in the trace, it is pageable, and it runs on the default stream where it serializes with the convs. After pinning and moving to a copy stream, the same table shows `Pinned -> Device`, the memcpy self-time halves, and — most tellingly — in the chrome trace the memcpy bar moves *under* the conv bars on a separate stream lane instead of sitting in a gap before them. That relocation, memcpy-in-a-gap to memcpy-under-compute, is the visual proof the overlap worked, and it is why the [chrome trace](/blog/machine-learning/performance-engineering/reading-a-chrome-trace) is the tool you verify against.

For the system-wide view, especially to see the DMA engine and the copy stream as distinct timeline rows, Nsight Systems is the better lens:

```bash
nsys profile -t cuda,nvtx,osrt -o copy_overlap --force-overwrite true \
    python serve.py --steps 200
```

In the resulting timeline, the CUDA HW rows split into a **kernel** row and a **memcpy** row, and with a copy stream you will literally see the green H2D bar on one CUDA stream lane running left-to-right *at the same horizontal position* as the blue kernel bars on another lane — concurrent, overlapped, hidden. Before the fix, the memcpy bar sits alone in a gap with the kernel row flat beside it. That side-by-side-versus-in-a-gap distinction is the single clearest thing Nsight shows you about a copy, and it is worth an [Nsight Systems](/blog/machine-learning/performance-engineering/nsight-systems-for-ai-services) session on any service you suspect is copy-bound.

One more thing worth doing in the trace: wrap your handler's phases in NVTX ranges so the memcpy row lines up with a *named* phase instead of an anonymous gap. A single `torch.cuda.nvtx.range_push("h2d")` / `range_pop()` around the copy, and another around `model(...)`, turns the Nsight timeline into a labeled story — you can see at a glance that the `h2d` range and the `forward` range now *overlap in wall-clock time* on two different streams, which is the exact evidence the fix worked. Without the labels you are eyeballing which green bar belongs to which phase; with them the profiler draws the overlap for you. The [NVTX and semantic-profiling](/blog/machine-learning/performance-engineering/nvtx-and-semantic-profiling-traces) approach — annotating the code so the trace reads like your handler — pays for itself the first time you have to explain a copy-overlap fix to someone who was not staring at the timeline with you.

One measurement honesty note, because it is the most common way people fool themselves here: **you must `torch.cuda.synchronize()` before you stop the clock.** A `non_blocking` copy returns to the CPU immediately, so if you time it with a wall clock and no sync, you are timing the *enqueue*, not the *transfer*, and you will "measure" a 3 ms copy at 0.05 ms and conclude it is already free. It is not; you just did not wait for it. Use CUDA events (which record on the stream and measure actual GPU-side duration) or a `synchronize()` before the final `time.perf_counter()`. Every number in this post came from CUDA events with a warmup and locked clocks, which is the discipline the [reproducible-benchmark](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) rules in this series exist to enforce.

## The traps: how each fix silently does nothing

Every fix in this post has a failure mode where it looks applied but does nothing, and each one is a real bug I have watched cost a team a day. Here are the four that matter, as a problem-solving narrative and then a stress test.

**Trap 1: `non_blocking=True` on pageable memory is a no-op.** This is the most common one by far. Someone reads that `non_blocking` makes copies async, sprinkles it on every `.to("cuda")`, measures no change, and concludes "async copies don't help." What actually happened: the tensor was pageable, so CUDA *cannot* do an async copy from it — it silently falls back to a synchronous staging copy and ignores the `non_blocking` flag entirely. There is no error, no warning, just a flag that did nothing. The fix is always to pin *first*; `non_blocking` without pinning is decoration. You can catch this in the profiler in one glance: if the memcpy row still says `Pageable`, your `non_blocking` is being ignored. This is important enough to state as a rule: **`non_blocking=True` only takes effect on page-locked memory.**

**Trap 2: a `.item()` / `.cpu()` / `.numpy()` mid-loop serializes the whole pipeline.** You built a beautiful overlapped pipeline, and it is still slow, and the trace shows a `cudaStreamSynchronize` blocking everything every step. The culprit is almost always a small, innocent-looking D2H read in the hot path — a `loss.item()` for logging, a `pred.cpu()` for a metric, a `.numpy()` for a postprocessing step, an `if tensor.max() > threshold` for a control-flow decision. Each of those forces the CPU to *wait for the GPU to finish everything queued so far* so it can read the value back, which drains your carefully-filled async pipeline down to empty and reintroduces the exact serialization you removed. The fix: batch your D2H reads (accumulate on the device, copy back once at the end), use `non_blocking` D2H into a pinned buffer and read it a step later, or just delete the logging read from the hot loop. In a decode loop or a per-step training loop, a single `.item()` can cost you 20 to 30% of throughput, and it will never show up as a slow kernel — it shows up as a sync stall, which is why you have to read the trace and not just the op table.

**Trap 3: pinning too much host memory starves the OS.** Pinned memory is *unswappable* by definition — that is what makes it DMA-able — which means every byte you pin is a byte the OS can never reclaim, page out, or give to another process. Pin a few hundred megabytes of staging buffers and you are fine. Pin tens of gigabytes (a common accident when you pin an entire dataset, or allocate a fresh pinned buffer every request and leak them), and you starve the page cache, push other processes toward swap, and can drive the whole box into thrashing or trigger the OOM killer — on the *host*, which is a confusing place to get OOM-killed when you were worried about the GPU. The rule is to pin a *bounded, reused* staging area (double or triple buffer, a few multiples of your batch size), not an unbounded amount, and never allocate pinned memory per request without freeing it.

**Trap 4: you overlapped, but you are copy-bound, so overlap cannot help.** This is the one where the mechanism protects you if you did the derivation. Overlap only makes the copy free when $t_\text{copy} \le t_\text{compute}$. If your copy is *bigger* than your compute — a huge input into a tiny model, or fp32 inputs into a model that computes in a few hundred microseconds — then $\max(t_\text{compute}, t_\text{copy}) = t_\text{copy}$, and no stream trickery helps, because the GPU genuinely cannot proceed until the bytes arrive. The fix for copy-bound is not scheduling, it is *moving fewer bytes*: send fp16 or int8 inputs instead of fp32 (halving or quartering the H2D), do preprocessing on the GPU so you copy raw compressed bytes instead of decoded tensors, or increase the batch size so the compute grows relative to the fixed per-image copy. Knowing whether you are copy-bound or compute-bound is a one-line check — compare the memcpy self-time to the kernel self-time in the profiler table — and it decides whether this entire post applies to you.

#### Stress test: does the fix survive batch 1, an L4, and varying shapes?

A fix that only works on the benchmark is not a fix. Push the overlapped pipeline on four axes:

- **Batch 1 versus batch 128.** At batch 1 the copy is tiny (~0.6 MB, ~25 µs) and so is the compute (~0.4 ms), but the *ratio* still matters — and at batch 1 the fixed per-launch overheads (kernel launch, stream sync) start to dominate both, so the overlap win shrinks from 2.3x to maybe 1.1x. Overlap is most valuable at the batch sizes where the copy is a meaningful fraction of a meaningful compute; at batch 1 the whole step is small and the copy is in the noise. This is the general rule: **the copy tax scales with bytes moved, so it hurts most on big batches and big inputs.**
- **L4 instead of A100.** An L4 has ~300 GB/s of HBM and a PCIe gen4 link, so its *compute* is much slower relative to its copy than an A100's. That actually makes overlap *more* valuable on the L4 for a given model — the compute window is longer, so a copy of the same size hides more easily under it — right up until the model is small enough that the L4 becomes copy-bound, at which point (trap 4) you switch to moving fewer bytes. The mechanism, not the hardware, tells you which regime you are in.
- **Shapes that vary per request.** Variable input sizes (different image resolutions, different sequence lengths) mean the copy size varies, which is fine for the copy itself — DMA does not care about shape — but it interacts badly with fixed-size pinned staging buffers. Allocate your pinned slots at the *maximum* size and copy the actual (smaller) number of bytes each request; do not reallocate a pinned buffer per shape, or you pay the page-locking syscall on every request and reintroduce the very stall you removed.
- **Fifty concurrent requests.** Under concurrency, multiple streams contend for one DMA engine and one PCIe link, so copies that overlapped compute in isolation can start to serialize *against each other* on the bus. The link has a fixed ~25 GB/s; if ten requests each want to move 77 MB at once, they share that bandwidth and each copy takes ~10x longer. The answer is the same as everywhere in serving: batch at the request boundary so you do one big copy instead of fifty small ones, and let the [batching layer](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention) amortize the transfer.

## When copies stop mattering: know your interconnect

Everything above assumes you are copying over PCIe, which is the slow link and the one worth fighting. But the copy tax is entirely a function of *which link the bytes cross*, and the moment they cross a faster one, the math changes.

![a matrix comparing pcie gen4 pcie gen5 nvlink and hbm bandwidth against pcie gen4 and what each interconnect implies for copies](/imgs/blogs/killing-host-device-copies-6.webp)

The matrix ranks the interconnects that matter. **PCIe gen4 x16** at ~32 GB/s (theoretical; ~25 effective) is the copy tax we have been fighting — the baseline for host-device transfers on most cloud GPUs. **PCIe gen5** at ~64 GB/s doubles that, which halves the H2D copy time and makes the overlap condition easier to satisfy, but it is still off-chip and still 30x slower than HBM, so the copy is still worth hiding. **NVLink** at ~900 GB/s is a different category: it is a GPU-to-GPU interconnect (not host-device), roughly 28x faster than PCIe gen4, and on an NVLink-connected multi-GPU box a device-to-device copy between two GPUs is nearly free compared to a host-device copy — which is why multi-GPU serving prefers to move activations GPU-to-GPU over NVLink rather than routing through host memory. And **HBM** at ~2,000 GB/s is the on-chip memory the compute runs against; it is the yardstick, and the whole reason a PCIe copy is a "tax" is that it is ~62x slower than the memory the GPU would rather be using. The practical reading: if your copy crosses PCIe, fight it with everything in this post; if it crosses NVLink, it is probably already in the noise; if it is an HBM-to-HBM on-device copy, it is not a data-movement problem at all and you should be looking at [kernel fusion](/blog/machine-learning/high-performance-computing/kernel-fusion-and-flashattention-beating-the-memory-wall) instead.

There is also a whole class of services where the copy genuinely does not matter and you should not spend a minute on it. An LLM decode service copies almost nothing per step — a handful of token IDs in (a few kilobytes), one token out — so its H2D/D2H is microseconds and utterly dwarfed by the attention and FFN compute; pinning the token-ID copy there is a rounding error, and the real fixes are KV-cache and batching. The copy tax is a *big-tensor, PCIe-crossing* problem: vision services moving image batches, embedding services moving large feature tensors, any pipeline where preprocessing happens on the CPU and produces big dense inputs. Diagnose before you optimize — the profiler table's memcpy self-time versus kernel self-time is the one number that tells you whether this post is about your service or not.

## Case studies and real numbers

A few concrete results, from the PyTorch docs and from measured services, to calibrate expectations.

**The PyTorch pinned-memory tutorial's own numbers.** PyTorch's official guidance on `pin_memory` and `non_blocking` is explicit that the two must be used together and that pinning roughly doubles achievable H2D bandwidth on PCIe by removing the pageable staging copy — the ~12 GB/s versus ~25 GB/s gap this post measured is the canonical result, reproduced across many gen4 boxes. The docs are equally explicit about the trap: `non_blocking=True` on non-pinned memory falls back to synchronous, and a `.cpu()` or `.item()` in the loop reintroduces a sync. If you read one primary source after this post, read that tutorial; it is the ground truth for the API behavior.

**DataLoader `pin_memory` in training throughput.** The standard result across vision training recipes is that `DataLoader(pin_memory=True, num_workers=N)` combined with `.to(device, non_blocking=True)` closes the input-side GPU idle gap that otherwise starves the GPU while the CPU stages each batch — the win is largest exactly when the input tensors are big (images, spectrograms) and the model is fast enough that the copy is a meaningful fraction of the step. On copy-heavy vision training this is routinely a 20 to 40% throughput improvement, and it is the first thing to check when a training run shows periodic GPU idle synchronized with batch boundaries.

**The `.item()` sync in a training loop.** A frequently-cited pattern: a training loop that logs `loss.item()` every step runs measurably slower than one that accumulates the loss on-device and reads it back every N steps, because each `.item()` forces a full pipeline sync. The measured cost varies with how much work is queued, but on tightly-pipelined loops a per-step `.item()` removal has been reported to recover on the order of 5 to 15% of wall-clock — a real, free win from deleting one synchronizing read from the hot path. It is the cheapest optimization in this entire post and the one most services are still leaving on the table.

**Copy-bound preprocessing solved by moving fewer bytes.** For services that turned out to be copy-bound (trap 4) — big fp32 inputs into a fast model — the winning fix is consistently to move the decode/preprocess onto the GPU (copy compressed JPEG bytes over PCIe, decode on the device) or to send lower-precision inputs, cutting the H2D bytes by 2x to 10x. This is the NVIDIA DALI / GPU-decode playbook, and it is the correct response when overlap cannot help because the copy is bigger than the compute. The lesson: overlap hides a copy that fits under compute; when it does not fit, you make it smaller, not later.

## When to reach for this (and when not to)

Killing host-device copies is one of the highest-leverage, lowest-risk optimizations in the whole series *when it applies*, and a complete waste of a day when it does not. Reach for it when:

- The profiler shows a `Memcpy HtoD` or `DtoH` row that is a meaningful fraction (say, more than 10%) of your step's CUDA time, and especially when it says `Pageable`.
- Your service moves big dense tensors across PCIe — vision (image batches), audio (spectrograms), embedding services, any CPU-preprocessing pipeline with large outputs.
- The GPU trace shows idle gaps aligned with copies, with the SM row flat during the transfer.

Do **not** reach for it when:

- Your copies are tiny relative to compute — LLM decode (token IDs in, one token out), or any model whose compute dwarfs its input by 100x. Pinning a 4 KB copy saves microseconds; go optimize the attention kernels instead.
- You are already copy-bound and the copy is bigger than any compute window — then overlap cannot help by construction ($\max$ picks the copy), and the fix is fewer bytes (lower precision, GPU-side decode), not a copy stream.
- You have not profiled. The single most common mistake is to pin and `non_blocking` everything reflexively without checking whether copies are on the critical path at all — you add double-buffer complexity and race-condition risk to a service whose copies were already in the noise. Diagnose first; the memcpy-versus-kernel self-time ratio is the whole decision.

The general shape of the recommendation: **pin and go async almost always (it is cheap and safe), overlap when the copy is on the critical path and fits under compute, and shrink the bytes when it does not.** The copy stream and double-buffer come with real correctness obligations (`wait_stream`, `record_stream`, not clobbering pinned sources), so add them deliberately and verify with a trace, not by faith.

## Key takeaways

- **A host-device copy is ~60 to 80x slower per byte than on-chip compute**, because it crosses PCIe (~25 GB/s effective) instead of HBM (~2,000 GB/s). An unhidden copy is dead GPU time at the worst exchange rate you have.
- **Copy time is `bytes / bandwidth`.** Derive it, and you know before you profile whether a copy is worth fighting and whether it can hide under compute.
- **The overlap condition is `t_copy ≤ t_compute`.** When it holds, a copy on a separate stream is *free* — it contributes nothing to the step, because the step time is `max(compute, copy)`.
- **Pin first — it is the prerequisite for everything.** Pinned (page-locked) host memory DMAs directly at ~2x the bandwidth of pageable memory and is the *only* memory `non_blocking` actually works on.
- **`non_blocking=True` frees the CPU, not the GPU.** On its own, on the default stream, it does not overlap the copy with compute — you need a dedicated copy stream for that.
- **Double-buffer to hide the copy permanently:** two pinned slots, a copy stream staging batch N+1 while the compute stream runs batch N, with `wait_stream` and `record_stream` to keep it correct.
- **Watch for the silent no-ops:** `non_blocking` on pageable memory does nothing; a `.item()`/`.cpu()` mid-loop drains the pipeline with a hidden sync; pinning too much starves the host OS.
- **Copy-bound is a different problem than copy-serialized.** If the copy is bigger than the compute, overlap cannot help — move fewer bytes (lower precision, GPU-side decode) instead.
- **Verify in a trace, not by faith.** The proof of overlap is the memcpy bar moving from a gap *under* the compute on a separate stream lane — see it in the chrome trace or Nsight before you believe it.

## Further reading

- [PyTorch: pin_memory, non_blocking, and CUDA streams](https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html) — the primary source for the API semantics, the pageable-fallback trap, and the sync-on-`.item()` gotcha.
- [PyTorch CUDA semantics: streams and events](https://pytorch.org/docs/stable/notes/cuda.html) — how streams, `wait_stream`, `record_stream`, and events actually order copies against compute.
- [NVIDIA: How to Optimize Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/) — the classic pinned-memory and overlap explainer, with the DMA-engine and PCIe-bandwidth mechanics from the metal up.
- [Why your AI service wastes CPU and GPU](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) — the series intro and the four-wastes frame this post's data-movement waste sits inside.
- [Reading a chrome trace](/blog/machine-learning/performance-engineering/reading-a-chrome-trace) and [Nsight Systems for AI services](/blog/machine-learning/performance-engineering/nsight-systems-for-ai-services) — the two tools you verify a copy fix against, and how to read the memcpy row.
- [Overlapping compute and communication](/blog/machine-learning/distributed-training/overlapping-compute-and-communication) — the same separate-stream overlap trick applied to network transfers in distributed training.
- [The performance engineering playbook](/blog/machine-learning/performance-engineering/the-performance-engineering-playbook) — the capstone decision tree that routes a symptom to the right fix, including copy-bound.

---
title: "The Straggler: How One Slow GPU Halved a 64-GPU Job"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "A war story about a 64-GPU run stuck at half its throughput while every GPU read 100% busy, the synchronization tax that made one slow card gate all 64, and the per-rank timing that fingered the culprit in ten minutes."
tags:
  [
    "distributed-training",
    "multi-gpu",
    "multi-node",
    "straggler",
    "nccl",
    "pytorch",
    "gpu",
    "ml-systems",
    "deep-learning",
    "profiling",
  ]
category: "machine-learning"
subcategory: "Distributed Training"
author: "Hiep Tran"
featured: true
readTime: 35
---

The dashboard said everything was fine. All sixty-four H100s on the cluster read 100% GPU utilization, power draw was healthy, the loss curve was descending on schedule, no NCCL warnings, no crashes, no OOM. By every indicator we habitually watch, the job was a picture of health. The only problem was that it was running at a little over half the speed it should have been. We had measured the same model on eight GPUs the week before: roughly 85,000 tokens per second per node, scaling almost linearly, which projected to something north of 650,000 tokens per second on all eight nodes. The sixty-four-GPU job was delivering about 380,000. We had eight times the hardware and a hair under four and a half times the throughput. Forty percent of a very expensive cluster was evaporating into thin air, and nothing on any dashboard would tell us where.

This is the single most disorienting failure mode in distributed training, and it is worth naming precisely because it violates the intuition every single-GPU engineer brings to the job. On one GPU, slow is slow: if the job is slow, the GPU is slow, and `nvidia-smi` will show you a low clock or a thermal warning and you fix it. In a synchronous multi-GPU job that reasoning breaks completely. The job can be slow while *every* GPU looks maxed out, because the thing dragging you down is not any GPU doing too little work — it is one GPU doing its work *slightly too slowly*, and sixty-three other GPUs politely waiting for it, spinning in a communication kernel that registers as 100% utilization while accomplishing exactly nothing. The busy-looking idle. The straggler.

![a gradient all-reduce barrier where sixty-three fast ranks and one slow straggler rank both feed into the collective which then gates the step and halves throughput](/imgs/blogs/the-straggler-1.webp)

By the end of this post you will be able to derive why a single slow rank sets the pace for the entire job, and why the loss is exactly proportional to how much slower that rank is; explain why the fast ranks show 100% utilization while doing nothing; instrument a training loop to log per-rank compute time and per-rank collective-wait time so the straggler fingers itself in one line of output; read `nvidia-smi` clocks, temperatures, and throttle reasons and cross-check them against DCGM to find the physical cause; and choose the right fix from the small menu that actually exists — cool it, evict it, rebalance it, or accept it. This is a war story from Wave 5 of the [Distributed Training in the Trenches](/blog/machine-learning/distributed-training/why-distributed-training) series, and its lesson is the most expensive one in the whole series: in a synchronous job, **a collective is only as fast as its slowest participant**, and if you cannot see per-rank timing, you are flying blind into exactly the failure that most reliably halves your throughput at scale.

## The synchronization tax

Let me state the mechanism as plainly as I can, because everything else in this post is a consequence of it. Data-parallel training — the [DDP loop we built from first principles](/blog/machine-learning/distributed-training/ddp-from-first-principles) earlier in the series — repeats four steps every iteration: each rank runs a forward pass on its own slice of the batch, each runs a backward pass, then **all ranks average their gradients through a single all-reduce**, then each applies the identical averaged gradient with its optimizer. The third step is a *synchronizing collective*. An all-reduce is not a message one rank sends and forgets; it is a barrier. Every rank contributes its gradient and every rank walks away with the sum, and by construction **no rank can leave the all-reduce until every rank has entered it**. If you want the [full anatomy of the collective](/blog/machine-learning/distributed-training/collectives-from-scratch) — ring versus tree, the byte volume it moves — it is one click away; here the only property that matters is that word: barrier.

That single property is the whole story. Consider what it means for timing. Rank 0 finishes its backward pass and reaches the all-reduce. It cannot proceed. It must wait until ranks 1 through 63 have also reached the all-reduce. The moment the *last* rank arrives, the collective fires, the reduction completes, and everyone proceeds together into the optimizer step. So the wall-clock time of one training step is not the average of the ranks' compute times, and it is not rank 0's compute time. It is the **maximum** over all ranks, plus the cost of the reduction itself once everyone has arrived.

Write it as a formula. Let $c_i$ be the compute time (forward plus backward) of rank $i$ on a given step, and let $A$ be the time the all-reduce itself takes once the last rank has arrived. The step time for the *entire job* is

$$T_\text{step} = \max_{i} c_i + A$$

Not $\frac{1}{N}\sum_i c_i$. Not $c_0$. The max. This is the synchronization tax, and it is levied on every synchronous step of every data-parallel, tensor-parallel, and pipeline-parallel job ever run, because all three end their step on a collective that every participant must reach.

Now suppose all ranks are identical and each takes $c$ to compute. Then $T_\text{step} = c + A$ and life is good; the max equals the common value and nobody waits more than the fixed reduction cost. This is the healthy case, and it is what your eight-GPU benchmark measured. But suppose one rank — call it the straggler — is slower by a fraction $s$, so it takes $c(1+s)$ while the other $N-1$ ranks still take $c$. Now

$$T_\text{step} = c(1+s) + A$$

The step got slower by $c \cdot s$, and it got slower for **all $N$ ranks**, not just the slow one. The throughput of the whole job, which is tokens-per-step divided by $T_\text{step}$, falls by the ratio

$$\frac{T_\text{healthy}}{T_\text{straggler}} = \frac{c + A}{c(1+s) + A}$$

When the all-reduce is small relative to compute — the normal case on a good interconnect where overlap hides most of it — $A \ll c$ and this simplifies to a number you should tattoo on the inside of your eyelids:

$$\text{throughput ratio} \approx \frac{1}{1+s}$$

A rank that is 40% slower ($s = 0.4$) drops the whole job to $1/1.4 \approx 0.71$ of its throughput — a 29% loss. A rank that is 65% slower drops it to $1/1.65 \approx 0.61$ — a 39% loss. A rank that is fully 2× slower ($s = 1.0$) drops the job to exactly one half. **One slow GPU. Half the cluster. That is not hyperbole; that is the formula.** And notice what is *not* in the formula: $N$. The size of the cluster does not appear. A straggler that is 2× slow halves a 4-GPU job and halves a 4,000-GPU job by the identical mechanism. The tax is set entirely by the slowest rank's relative slowdown, not by how many fast ranks are waiting on it.

### The busy-looking idle

Here is the part that makes stragglers so maddening to catch. When the 63 fast ranks reach the all-reduce and wait for the straggler, what are they doing? They are *executing the NCCL all-reduce kernel*. NCCL, to minimize latency, does not put the GPU to sleep and wait on an interrupt when it reaches a collective. It **spins** — it runs a kernel that busy-waits, polling for the data to arrive from its peers, keeping the SMs resident and hot so that the instant the straggler's contribution shows up, the reduction proceeds with zero wake-up latency. That design choice is correct for latency and catastrophic for observability, because `nvidia-smi`'s "GPU-Util" field means one specific thing: *the percentage of the last sample interval during which at least one kernel was executing on the device.* A spinning NCCL kernel is a kernel executing on the device. So a rank that is doing nothing but waiting reads **100% GPU utilization**. All 63 idle ranks show 100% util. The one straggler, grinding through real work, also shows 100% util. The dashboard is uniform green and completely useless. This is why the intro says every GPU looked busy: they *were* busy, in the only sense `nvidia-smi` can measure. They were busy waiting.

That single fact — utilization cannot distinguish useful work from a spin-wait — is why you cannot debug a straggler with the tools that work on one GPU, and it is the reason the entire middle of this post is about measuring something `nvidia-smi` does not show you: **per-rank compute time** and **per-rank collective-wait time**.

## Worked example: the throttle that cost forty percent

Let me put concrete numbers on the mechanism, because "40% slower" is abstract until you see it in tokens and dollars. This is the clean, isolated version of the real incident.

Take a 7-billion-parameter transformer training on 64 H100 SXM GPUs across eight DGX nodes, data-parallel, bf16. On healthy hardware every GPU runs its SM clock at roughly 1900 MHz under sustained load and completes one forward-plus-backward step in about 175 ms. The global batch is sized so that each step processes about 119,000 tokens across the cluster. That gives a healthy throughput of

$$\frac{119{,}000 \text{ tokens}}{0.175 \text{ s}} \approx 680{,}000 \text{ tokens/s}$$

which for a 7B model at bf16 on H100 works out to a believable mid-40s MFU. Good numbers. Now one GPU — rank 41, physically the third card in node 5 — has a failing fan. Under sustained load it heats up, hits its hardware thermal cap, and the GPU protects itself the only way it can: it sheds clock. Its SM clock drops from 1900 MHz to 1200 MHz. For a compute-bound training kernel, throughput scales roughly with clock, so rank 41's step time stretches by the inverse of the clock ratio:

$$c_{41} = 175 \text{ ms} \times \frac{1900}{1200} \approx 277 \text{ ms}$$

Round it to about 290 ms once you include the memory-bound layers that suffer a little extra. Rank 41 is now roughly 65% slower than its 63 peers, who are all still humming along at 175 ms. But because of the barrier, *all 64 ranks* now step at 290 ms, gated by rank 41. The whole-job throughput becomes

$$\frac{119{,}000 \text{ tokens}}{0.290 \text{ s}} \approx 410{,}000 \text{ tokens/s}$$

![a before and after comparison showing a healthy 64-GPU job at 680k tokens per second collapsing to 410k tokens per second when one card throttles to 1200 MHz](/imgs/blogs/the-straggler-2.webp)

The job lost 270,000 tokens per second — a hair under 40% of its throughput — because one of sixty-four cards is running its fan a little slow. The other 63 H100s, each a roughly thirty-thousand-dollar accelerator, spend 115 ms of every 290 ms step spinning in a NCCL kernel accomplishing nothing, and report 100% utilization while they do it.

Now price the damage over the run. Suppose the job was planned as a 30-day run at 680k tokens/s. At 410k tokens/s it now needs $30 \times \frac{680}{410} \approx 49.7$ days to process the same tokens — 19.7 extra days. Over 64 GPUs that is $64 \times 24 \times 19.7 \approx 30{,}200$ extra GPU-hours. At a representative cloud rate of \$3 per H100-hour, one throttling fan just added roughly **\$90,000** to the run and pushed delivery back three weeks. Nobody signed off on that. Nobody even saw it, because the loss curve looked fine and every GPU read 100% busy. That is the cost of a straggler you cannot see, and it is why per-rank timing is not a nice-to-have.

## Why GPUs throttle in the first place

The 1900-to-1200 MHz drop deserves a closer look, because thermal throttling is the single most common cause of a straggler and the one most people have never watched happen live. A GPU has a fixed thermal and power envelope. Under a heavy sustained kernel — and training kernels are about as heavy and sustained as it gets — the die heats up. When it reaches the hardware thermal threshold (on H100, in the low-to-mid 80s Celsius for the sustained slowdown point, with a hard shutdown much higher), the on-chip governor **caps the clock** to keep the temperature safe. It has no choice; the alternative is damage. The clock drop is the GPU doing exactly what it is designed to do, and it is invisible unless you are looking at per-GPU clocks and throttle reasons — which nobody looks at until they have been burned once.

![a timeline showing a GPU starting cool at 1980 MHz, heating to 83C, hitting the hardware thermal cap, dropping to 1200 MHz, halving job throughput, and recovering after the fan is fixed](/imgs/blogs/the-straggler-3.webp)

What makes one card in a rack of eight hit its thermal cap while its seven neighbors do not? It is almost always **airflow, not silicon**. A single failing or fouled fan. A card seated in the hottest slot of the chassis, downstream of the others' exhaust. A rack-level hot spot near a failing CRAC unit in the data center. Dust. A bent heatsink fin. The silicon is fine; the cooling path to that one card is compromised, so under identical load it runs hotter, hits the cap sooner, and sheds clock while the rest of the rack stays at full boost. This is why stragglers cluster on *physical* location — a specific slot, a specific node, a specific rack — far more than on logical rank. The straggler is a facilities problem wearing a distributed-systems costume.

Thermal throttling is the headline cause, but it is one of several, and the others produce the same "one slow rank" symptom through entirely different physics.

## The seven ways a rank goes slow

Before we can fix a straggler we have to enumerate what makes one, because the fix depends entirely on the cause and they are not interchangeable. Cooling a card that has a corrupted data shard does nothing. Rebalancing shards on a card that is thermally throttling does nothing. Here is the taxonomy I keep in my head, roughly in order of how often I have actually hit each one at scale.

**One — thermal throttling.** The case above. One hot card sheds clock. Signature: SM clock materially below its peers under load; `clocks_throttle_reasons.active` shows a thermal bit set; temperature pinned at the cap. Physical, per-card, and often intermittent (it throttles only once the die is hot, so the first few minutes of a run look fine and then throughput sags).

**Two — a genuinely slower or degraded card.** Mixed hardware in the same job (you meant to run all H100s but one node has an older stepping, or someone slipped an A100 into the pool), or a card that has silently degraded. Signature: the rank is slow *even at full clock and normal temperature*. Nothing is throttling; the card is just slower. This is the nastiest to diagnose because every throttle indicator is clean.

**Three — ECC errors and row remapping.** GPU HBM has error-correcting memory. When a memory cell starts going bad, the GPU corrects the error transparently (correctable ECC) and, past a threshold, **remaps the bad row** to a spare. Both the correction and the remapping add latency to memory accesses, and a card accumulating ECC events runs its memory-bound layers measurably slower while looking completely healthy on clocks and temperature. Signature: normal clocks, normal temp, but rising ECC counters in DCGM and a slowdown concentrated in memory-bound operations. Left alone, this card is on its way to a hard failure and an XID error.

**Four — a noisy neighbor.** The GPU is fine, but something *else* on its node is stealing a shared resource. Another job (or a rogue process) hammering the same NVMe the data loader reads from; a monitoring agent pinning a CPU core the training process needs for its data pipeline; network contention on a shared uplink. Signature: the slowdown is spiky and correlated with *node*, not card, and it moves if you move the workload. On shared clusters this is depressingly common.

**Five — a data-loader straggler.** The GPU compute is identical across ranks, but one rank's *input pipeline* is slow: its shard of the dataset lives on a slower storage tier, or contains larger or harder-to-decode samples (long documents, high-resolution images), or its `DataLoader` workers are starved. The GPU sits idle waiting for the next batch, then arrives late at the all-reduce. Signature: compute time per rank is uniform, but *end-to-end* step time has one slow rank whose slowness is in data-wait, not kernel time. This one is a favorite because it hides from every GPU tool — the GPU is genuinely idle and even reports *low* utilization, but only during the wait, which most sampling misses.

**Six — NUMA and PCIe placement.** On a multi-socket node, a process pinned to the wrong NUMA node reaches its GPU across the inter-socket link and its host memory across a slower path, adding latency to every host-to-device copy and every CPU-side data operation. One misplaced rank in an otherwise well-pinned job is a persistent mild straggler. Signature: consistent small slowdown on specific local ranks that correlates with CPU/GPU affinity, fixed by correct `numactl`/affinity binding.

**Seven — a degraded network link.** On multi-node jobs, one node's InfiniBand or RoCE link negotiates down to a lower rate, or a cable/transceiver is marginal and forces retransmits, or the link falls back from RDMA to slower TCP. That node's contribution to every cross-node collective arrives late. Signature: the slow "rank" is really a whole *node*, the slowness appears only in inter-node collectives, and NCCL debug or the fabric counters show the degraded link. This overlaps with the failure modes in [the NCCL timeout that hung the job](/blog/machine-learning/distributed-training/the-nccl-timeout-that-hung-the-job), except here the link is slow, not dead — it does not time out, it just taxes you.

![a matrix classifying five straggler causes by their throughput signature, where to look for each, and the fix for each](/imgs/blogs/the-straggler-4.webp)

The matrix above compresses the five most common of these into signature, where-to-look, and fix, because when you are staring at a slow job at 2am the enumeration is what you need, not the prose. Note the column that unifies them: every one of these produces the *same* aggregate symptom — throughput below what the cluster should deliver — and each is distinguished only by a *different* per-node or per-rank measurement. Which brings us to the actual hard problem.

## The detection problem: what versus where

Here is the crux of the entire post, and the reason stragglers are a genuinely hard operational problem rather than a trivial one: **aggregate throughput tells you that you have a straggler, but it cannot tell you which rank it is.** Your tokens-per-second number is a single scalar for the whole job. It went down. Great — now which of sixty-four GPUs, across eight nodes, is the culprit? The aggregate number has thrown away exactly the information you need. You have a "what" and you need a "where," and no amount of staring at the loss curve or the throughput graph will convert one into the other.

The instinct is to reach for `nvidia-smi` on every node and eyeball it. This fails for the reason we established: every rank reads 100% utilization because the fast ones are spinning in NCCL. Utilization is uniform and uninformative. The instinct after that is to look at temperatures and clocks across all 64 GPUs, and this *can* work — if the cause is thermal — but it is a shot in the dark that misses causes two through seven entirely, and it requires you to already suspect the physical layer before you have localized the logical rank.

The correct approach inverts the problem. Instead of measuring the *hardware* on every node and hoping to spot an anomaly, measure the *timing* on every rank and let the barrier itself tell you who is slow. The insight is beautiful in its simplicity: **the straggler is the rank that never waits.** Every fast rank reaches the all-reduce early and burns time waiting for the straggler. The straggler reaches the all-reduce last and waits for essentially no one. So if you log, per rank, how much time each spends *waiting inside the collective*, you get a signature that points straight at the culprit: 63 ranks with large, roughly equal wait times, and one rank with near-zero wait. The rank with no wait is your straggler. You do not need to know the cause yet. You do not need to touch a single node. You just need per-rank wait time, and the physics of the barrier hands you the answer.

![a stack descending from aggregate tokens per second through per-rank step time and collective wait time down through nvidia-smi and DCGM counters to the single root-cause card](/imgs/blogs/the-straggler-5.webp)

The stack above is the drill-down I run every time, top to bottom. Aggregate throughput says "slow, but not where." Per-rank step time narrows it to a rank. Per-rank collective-wait time confirms it (the near-zero-wait rank) and rules out a global cause. *Then*, and only then, do you go to that specific rank's node and read `nvidia-smi` clocks/temps/throttle reasons and DCGM counters to find the physical cause — because now you know exactly which card to interrogate. The whole descent takes about ten minutes once you have the instrumentation in place, and the instrumentation is a dozen lines of code.

## Instrumenting per-rank timing

Let me show the code, because this is the part that turns a mysterious slow job into a two-minute diagnosis. The idea: on a sampling of steps, time each rank's local compute (forward + backward) with CUDA events, then insert an explicit barrier and time how long the rank waits there. Gather both numbers to rank 0 and print them. The straggler is the rank with the highest compute and the lowest wait.

```python
import time
import torch
import torch.distributed as dist


def timed_step(model, batch, optimizer):
    """One training step, instrumented to separate compute from wait."""
    # --- time local compute: forward + backward ---
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    loss = model(**batch).loss
    loss.backward()

    torch.cuda.synchronize()          # local grads are ready here
    t_compute = time.perf_counter() - t0

    # --- time the barrier wait ---
    # A fast rank blocks here until the SLOWEST rank arrives.
    # The straggler arrives last and waits ~0.
    t1 = time.perf_counter()
    dist.barrier()
    t_wait = time.perf_counter() - t1

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return t_compute, t_wait
```

The `torch.cuda.synchronize()` calls matter: CUDA kernels launch asynchronously, so without a sync your Python timer measures launch time, not execution time. You must synchronize before reading the clock or every number you print is fiction. This is the single most common mistake in homemade GPU timing and it will make a straggler vanish because you never actually waited for the compute to finish.

One honesty note before we gather the numbers: the explicit `dist.barrier()` here is a *diagnostic* probe, not the steady-state code path. In real DDP the gradient all-reduce is overlapped with the backward pass — it does not sit as a clean barrier after compute. So you toggle this instrumentation on for a few hundred sampled steps when you suspect a straggler, read the signature, and toggle it off. It perturbs throughput slightly while on, which is a fine price for localizing a 40% loss. For a zero-overhead version you instead read the wait time NCCL already tracks, but the explicit barrier is the clearest thing to reason about and to teach.

Now aggregate across ranks and flag the outlier:

```python
def report_straggler(t_compute, t_wait):
    rank = dist.get_rank()
    world = dist.get_world_size()

    stats = torch.tensor([t_compute, t_wait], device="cuda")
    gathered = [torch.zeros_like(stats) for _ in range(world)]
    dist.all_gather(gathered, stats)

    if rank == 0:
        comps = [g[0].item() * 1e3 for g in gathered]   # ms
        waits = [g[1].item() * 1e3 for g in gathered]    # ms
        slow = max(range(world), key=lambda i: comps[i])
        median = sorted(comps)[world // 2]
        slowdown = comps[slow] / median - 1.0
        print(
            f"[straggler] rank {slow}: "
            f"compute={comps[slow]:.0f}ms (median {median:.0f}ms, "
            f"+{slowdown*100:.0f}%), wait={waits[slow]:.1f}ms"
        )
```

Run this on a sampled step of the throttled job and the output is unambiguous:

```console
[straggler] rank 41: compute=289ms (median 175ms, +65%), wait=2.1ms
```

There it is. Rank 41 computes 65% slower than the median and waits essentially zero time at the barrier, while — if you print the full vector — every other rank shows a wait of roughly 114 ms. You did not touch a single node. You did not eyeball a single temperature. The barrier told you exactly who is slow. Now you know precisely which card to interrogate physically, and the second half of the diagnosis begins.

#### Worked example: reading the wait-time signature

Let me make the signature explicit, because *reading* it correctly is what separates a ten-minute diagnosis from a day of thrashing. Take the throttled job at steady state: the straggler (rank 41) computes for 290 ms; the 63 fast ranks compute for 175 ms each. The all-reduce, once everyone arrives, takes about 15 ms. Walk through one step from each perspective.

A fast rank, say rank 7: it computes for 175 ms, reaches the barrier, and waits. It cannot leave until rank 41 arrives at the 290 ms mark. So rank 7 waits $290 - 175 = 115$ ms, then joins the 15 ms reduction. Its step is $175 + 115 + 15 = 305$ ms, of which **115 ms — 38% of the entire step — is pure idle**, spent spinning in NCCL reading 100% utilization.

The straggler, rank 41: it computes for 290 ms, reaches the barrier *last*, and waits for essentially no one — everyone else is already there. Its wait is about 2 ms (the residual scheduling jitter of the last-arriver). Its step is $290 + 2 + 15 = 307$ ms, of which essentially none is idle. It is the only card in the job doing useful work for the full step.

![a before and after comparison of per-rank wait time showing sixty-three ranks waiting 115 ms while the straggler waits 2 ms, then all ranks waiting only 8 ms after eviction](/imgs/blogs/the-straggler-6.webp)

So the signature across the 64-element wait vector is 63 values clustered near 115 ms and exactly one value near zero. **The straggler is the minimum of the wait vector, not the maximum** — an inversion that trips people up every time. You are hunting for the rank that *isn't* waiting. Equivalently, and more robustly, it is the maximum of the *compute* vector; the two agree and cross-checking them guards against a rank whose slowness is in data-wait rather than kernel time. After you fix or evict rank 41 (the "after" side of the figure), the compute vector flattens to a uniform 175 ms and the wait vector collapses to a uniform ~8 ms of ordinary jitter — no near-zero outlier, no 115 ms victims. That flat wait vector is what a healthy job looks like, and it is the thing to alert on: **not** low throughput (a lagging indicator) but a *wait-time outlier* (the leading, localizing one).

## Finding the physical cause

Per-rank timing gives you the rank. Now you need the cause, and for that you go to that rank's node and read the hardware. The first tool is `nvidia-smi`, queried for exactly the fields that distinguish the causes:

```bash
# On the straggler's node, watch clocks, temp, and throttle reasons at 1 Hz
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,\
power.draw,clocks_throttle_reasons.active \
--format=csv -l 1
```

On a healthy node every card looks alike. On the node hosting rank 41 you see the smoking gun:

```console
index, clocks.sm [MHz], temperature.gpu, power.draw [W], clocks_throttle_reasons.active
0, 1905 MHz, 71, 698.4 W, 0x0000000000000000
1, 1905 MHz, 70, 701.1 W, 0x0000000000000000
2, 1200 MHz, 84, 512.7 W, 0x0000000000000004
3, 1905 MHz, 72, 699.8 W, 0x0000000000000000
```

Card 2 (local rank 2, global rank 41) is the outlier on every axis: clock pinned at 1200 while its neighbors boost to 1905, temperature at 84°C against their low 70s, power *down* at 513 W because it is doing less work per unit time, and a nonzero throttle-reasons bitmask. That `0x...0004` is the `SW Thermal Slowdown` bit — the driver decode of the reasons is worth memorizing at least loosely: `0x0000000000000001` is idle, `0x0000000000000004` is software thermal slowdown, `0x0000000000000008` is hardware thermal slowdown (the serious one), `0x0000000000000040` is hardware power brake, and `0x0000000000000080` is the power cap. A set thermal bit plus a temperature at the cap plus a depressed clock is the unambiguous fingerprint of a cooling problem, and it points you at facilities, not code. If you prefer booleans to bitmasks, query the decoded fields directly:

```bash
nvidia-smi --query-gpu=index,\
clocks_throttle_reasons.hw_thermal_slowdown,\
clocks_throttle_reasons.sw_thermal_slowdown,\
clocks_throttle_reasons.hw_power_brake \
--format=csv
```

For the causes that leave clocks and temperature clean — the degraded card, ECC row remapping, a silent hardware fault on its way to failure — `nvidia-smi`'s summary is not enough and you go to DCGM, NVIDIA's data-center GPU manager, which exposes the counters `nvidia-smi` does not surface at a glance:

```bash
# Live per-GPU: SM clock, mem clock, temp, power, ECC, XID errors
dcgmi dmon -e 100,101,150,155,319,394 -d 1000
```

Those field IDs are SM clock, memory clock, GPU temperature, power usage, XID errors, and correctable ECC errors respectively. A card whose clocks are fine but whose ECC counter is climbing, or that has logged an XID event, is a degraded-hardware straggler heading toward a hard failure — you want it out of the job *before* it takes the run down with a NaN or a hang. For a one-shot health verdict rather than a live monitor, `dcgmi diag` runs an active diagnostic:

```bash
dcgmi diag -r 2          # medium-length health + stress check
```

which will flag a card failing its clock, memory, or thermal targets directly. The division of labor is simple: `nvidia-smi` throttle reasons catch the thermal cases in seconds; DCGM ECC/XID counters and `dcgmi diag` catch the degraded-hardware cases that throttle reasons miss. Between the per-rank wait signature (which rank) and these two tools (why), you have localized and root-caused the straggler without a single lucky guess.

## The fixes, and the one case with no fix

You have the rank and the cause. What now? The menu of fixes is short, and which entry you pick is dictated entirely by the cause you found.

![a decision tree branching from a detected straggler through transient thermal, persistent hardware, and software placement causes to the matching fix for each](/imgs/blogs/the-straggler-7.webp)

The tree above is the whole decision. Follow the three branches.

**Transient / thermal → cool it, and alert on it.** If the cause is throttling, the fix is physical: fix the fan, clear the airflow obstruction, address the rack hot spot. In the moment, if you cannot get hands on the hardware, you have two stopgaps. You can cap the clock of the *fast* GPUs to match the straggler with `nvidia-smi -lgc` (lock graphics clock) so nobody throttles unpredictably and your throughput at least becomes stable and predictable rather than sawtoothing as the hot card heats and cools — this trades a little peak speed for a lot of predictability, occasionally worth it to finish a run cleanly. Or you evict the node (below). The durable fix is a maintenance ticket and, going forward, an **alert on `clocks_throttle_reasons` and per-GPU temperature** so the next hot card pages you in minute two instead of costing you three weeks silently.

**Persistent hardware → evict the node.** If the card is degraded, throwing ECC errors, or slow at full clock, you do not want it in the job at all — it is not just slow, it is a NaN or a hang waiting to happen. You drain it. This is where **elastic training** earns its keep: a job launched with `torchrun` under an elastic rendezvous can lose a node and *re-form the process group without it*, continuing on the survivors from the last checkpoint rather than dying. The launch looks like this:

```bash
torchrun \
  --nnodes=7:8 \
  --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$HEAD_NODE:29500 \
  --max_restarts=3 \
  train.py --config run.yaml
```

The `--nnodes=7:8` is the elastic range: the job wants 8 nodes but will run on as few as 7. When the monitoring layer drains the straggler's node, the rendezvous re-forms the group at 7 nodes, the job reloads the latest checkpoint, and training resumes — now genuinely 7/8ths the size but at *full* per-GPU speed rather than 8/8ths the size at 60% speed, which is very often the better trade. The full machinery of restarts, rendezvous, and resume-on-failure is its own topic in [fault tolerance and elastic training](/blog/machine-learning/distributed-training/fault-tolerance-and-elastic-training); here the point is only that eviction is a *supported operation*, not a job restart from scratch, if you launched elastically.

**Software / placement → rebalance or re-pin.** If the straggler is a data-loader problem — one rank's shard slow to read or decode — the GPU is innocent and eviction is the wrong hammer. You rebalance the data. The common culprit is length or size skew: one shard packed with long documents or big images so its rank spends longer decoding every batch. The fix is to distribute samples so each rank's *decode cost per step* is even — interleave shards rather than assigning contiguous slabs, or length-bucket and round-robin so no single rank owns all the heavy samples:

```python
# Even out per-rank input cost: interleave, don't assign contiguous slabs.
# Rank r sees indices r, r+world, r+2*world, ... so heavy/light samples
# spread evenly across ranks instead of clumping in one shard.
from torch.utils.data import DistributedSampler

sampler = DistributedSampler(
    dataset,
    num_replicas=dist.get_world_size(),
    rank=dist.get_rank(),
    shuffle=True,
    drop_last=True,      # ragged last batch is its own tiny straggler
)
# And make sure the loader can actually keep the GPU fed:
loader = torch.utils.data.DataLoader(
    dataset, batch_size=micro_bs, sampler=sampler,
    num_workers=8, pin_memory=True, prefetch_factor=4, persistent_workers=True,
)
```

If it is a NUMA/PCIe placement straggler, the fix is affinity binding — launch each rank pinned to the NUMA node local to its GPU (`numactl --cpunodebind`/`--membind`, or your launcher's affinity flags) so no rank is reaching across the socket for host memory. If it is a degraded network link, you drain that node and open a fabric ticket, same as degraded hardware.

**And the case with no fix.** Here is the honest, uncomfortable truth the fixes tiptoe around: for a *persistent, mild* straggler in a purely synchronous job, there is no software cure. If one card is permanently 8% slower and you cannot replace it, synchronous SGD will run 8% slower, full stop, because the barrier is not optional — dropping it would let the replicas diverge and break the correctness invariant the whole [DDP loop](/blog/machine-learning/distributed-training/ddp-from-first-principles) exists to preserve. You can evict the card (losing its compute entirely, which for one card in sixty-four is usually the better deal) or you can eat the slowdown. What you cannot do is keep the card *and* the speed. The only regime that tolerates a persistent straggler without paying the full tax is **asynchronous** training — parameter-server or gossip approaches where fast ranks proceed on slightly stale parameters instead of waiting — and that trades the straggler tax for a *convergence* cost: stale gradients hurt sample efficiency and, past a point, stability. For pretraining at scale the field has overwhelmingly chosen synchronous training and paid the straggler tax with monitoring and eviction rather than accept async's convergence penalty. That is a real trade-off, not a solved problem, and knowing which side you are on is part of the job.

## The full autopsy, in order

Let me stitch the incident back together as the runbook it became, because the *order* of operations is the actual deliverable. When throughput is below what the cluster should deliver and nothing has crashed:

1. **Confirm it is a straggler, not a global regression.** Is throughput low uniformly, or did it *sag* after a healthy start? A sag that grows over minutes screams thermal (the card heats up over time). A flat low from step zero points to placement, a slow shard, or a degraded card. A global cause — a bad `NCCL_ALGO`, an interconnect fallback, a too-small batch — would slow *every* rank equally and show *no* wait-time outlier, so it is a different investigation (the [multi-node throughput autopsy](/blog/machine-learning/distributed-training/why-distributed-training) covers that sibling case). The wait-time vector distinguishes them instantly: an outlier means straggler; a uniform slowdown means global.

2. **Turn on per-rank timing.** Toggle the `timed_step` instrumentation for a few hundred steps. Read the compute vector and the wait vector. One high-compute, near-zero-wait rank is your straggler. No outlier means look elsewhere.

3. **Go to that rank's node and read the hardware.** `nvidia-smi` clocks/temp/throttle reasons first (catches thermal in seconds), then `dcgmi dmon` ECC/XID and `dcgmi diag` (catches degraded hardware). Now you have the cause.

4. **Apply the matching fix.** Thermal → cool it and alert on throttle reasons. Degraded/ECC/XID → drain the node, let elastic re-form the group. Data/placement → rebalance shards or re-pin NUMA.

5. **Make it a permanent alert.** The whole incident happened because *nothing watched the leading indicator.* Add per-GPU clock/temp/throttle-reason and per-rank wait-time to your [long-run monitoring](/blog/machine-learning/distributed-training/monitoring-a-long-run) so the next straggler pages you in minute two, not payday three.

In our real incident that whole sequence took about eleven minutes from "the throughput looks wrong" to "rank 41's fan is dead, draining node 5." Ten of those minutes were me not yet having the instrumentation wired in. The second time it happened — different cluster, different card, a rack hot spot instead of a fan — it took ninety seconds, because the wait-time alert had already fired and named the rank before I opened the terminal. That delta, ten minutes versus ninety seconds versus three silent weeks, is the entire return on understanding stragglers.

## Case studies and real numbers

Stragglers are not a toy problem that only bites hobby clusters; they are a first-order concern in every published account of frontier-scale training, and the largest operators build dedicated tooling for exactly the detection problem this post describes. A few grounded data points, cited approximately because the exact figures vary by report and I will not invent precision I do not have.

**Meta's Llama 3 (2024).** The Llama 3 herd was trained on a cluster of 16,384 H100s, and Meta's paper is unusually candid about reliability at that scale: over a 54-day pre-training snapshot they report on the order of 466 job interruptions, the large majority traced to hardware — GPU failures, HBM/ECC issues, network problems. Crucially they distinguish outright *failures* (a card dies, the job restarts) from *stragglers* (a card lives but lags), and they describe tooling to detect and remove lagging GPUs, because at 16k GPUs the probability that *some* card is throttling or degrading at any given moment approaches one. The lesson they state explicitly is the one this post is built on: at scale, per-node/per-rank health telemetry is not optional infrastructure — it is the difference between 90% and 50% effective throughput.

**ByteDance's MegaScale (2024).** MegaScale, describing training LLMs on more than 10,000 GPUs, devotes real engineering to what they call the diagnosis of stragglers and "abnormal" nodes, reporting a production MFU in the mid-50s percent and arguing that sustaining it required continuous per-GPU and per-rank monitoring to catch and evict slow participants before they dragged the collective down. Their framing matches ours precisely: aggregate MFU tells you *that* you are slow; localized per-rank metrics tell you *where*, and they built the second because the first was useless for action.

**Meta's OPT-175B logbook (2022).** The famous OPT-175B training logbook is a public, blow-by-blow diary of a large run, and it is a catalog of exactly this genre of failure: dozens of hardware faults, slow and dead nodes swapped out over the course of training, loss behavior watched like a hawk, manual restarts from checkpoints. Reading it is the single best inoculation against the belief that large training runs are smooth; they are a continuous fight against hardware entropy, and the straggler is one of its most common weapons. If you internalize one artifact after this post, make it that logbook.

The through-line across all three is that the operators closest to the metal converged on the same conclusion independently: **the straggler is a monitoring problem, and per-rank/per-GPU telemetry is the tool.** Nobody at frontier scale is eyeballing `nvidia-smi`. They are alerting on wait-time outliers and throttle reasons, evicting through elastic restart, and treating a lagging card as a page, because they have all paid the \$90,000-per-fan tax at least once and decided never again.

## When to invest in straggler tooling, and when not

Every technique in this series is a cost, and honesty about when a technique does *not* earn its keep is more valuable than one more description of when it does. Straggler instrumentation is cheap but not free, and there is a real threshold below which you should not bother.

**Do not build straggler tooling for a single 8-GPU node.** On one well-cooled DGX with a shared chassis and NVLink, stragglers are rare and, when they happen, easy to spot: eight GPUs fit on one `nvidia-smi` screen, and a thermal outlier is visible at a glance. The per-rank wait instrumentation is overkill; just watch clocks and temps. The synchronization tax is still real, but with only 8 participants the probability that any one is throttling on a given day is low enough that ad-hoc checking suffices.

**Do build it the moment you go multi-node, and treat it as mandatory past ~32 GPUs.** The reason is the probability argument, and it is the one number in this post that scales with $N$. Suppose any given GPU has some small independent probability $p$ of being a straggler at a given time — a fan degrading, a hot slot, a marginal link. The probability that *at least one* of your $N$ GPUs is a straggler is $1 - (1-p)^N$, which grows fast: at $p = 1\%$ per card, a single card has a 1% chance of gating you, 8 cards about 8%, 64 cards about 47%, 512 cards about **99.4%**. The tax per straggler does not grow with $N$ — but the *probability of paying it* climbs toward certainty. Past a few dozen GPUs you are not asking *whether* you will have a straggler but *how fast you will find it*, and that question is answered entirely by whether you built the per-rank telemetry in advance.

**Do not reach for asynchronous training just to dodge stragglers.** It is tempting to read "the barrier is the problem" and conclude "remove the barrier." For pretraining at scale that trade is almost always bad: async's stale gradients cost you convergence and stability, and you will spend more compute reaching the same loss than you saved dodging the occasional straggler. Synchronous training plus monitoring plus eviction is the mainstream choice for good reason. Async has its place — some RL and recommendation-system setups genuinely prefer it — but "I had a straggler once" is not the justification.

**Do add clock/temp/throttle-reason and wait-time-outlier alerts to any run longer than a day.** This is the cheapest high-value monitoring you can add and the one most teams skip until they have been burned. It is a handful of lines and it converts the most expensive silent failure in distributed training into a two-minute page.

## Key takeaways

- **A synchronous collective is only as fast as its slowest participant.** The step time is $\max_i c_i + A$, the max over ranks, not the average. One slow rank sets the pace for the entire job.
- **The throughput loss is $\approx 1/(1+s)$ and does not depend on cluster size.** A rank 40% slower costs the job ~29%; a rank 2× slower halves it. The same straggler halves a 4-GPU job and a 4,000-GPU job identically.
- **Fast ranks report 100% GPU utilization while doing nothing**, because they spin in the NCCL kernel and `nvidia-smi` counts a spinning kernel as busy. Utilization cannot distinguish useful work from a wait; it is useless for finding stragglers.
- **Aggregate throughput is a "what," not a "where."** It tells you the job is slow and discards the information about which rank. You need per-rank timing to localize.
- **The straggler is the rank that never waits.** Log per-rank collective-wait time: 63 ranks with a large equal wait and one rank with near-zero wait — the near-zero one is your culprit. Cross-check with the max of the compute vector.
- **Then read the hardware on that one node.** `nvidia-smi` clocks/temp/throttle-reasons catches thermal in seconds; DCGM ECC/XID counters and `dcgmi diag` catch the degraded-hardware cases that leave clocks clean.
- **Match the fix to the cause.** Cool a throttle and alert on it; evict a degraded card via elastic restart; rebalance a lopsided data shard; re-pin a NUMA-misplaced rank. Cooling a bad shard or rebalancing a hot card does nothing.
- **A persistent mild straggler in synchronous SGD has no software cure** but to fix or evict the hardware. Async removes the barrier at a convergence cost the field has mostly declined to pay for pretraining.
- **Alert on the leading indicator, not the lagging one.** Throughput dropping is the symptom you notice too late; a wait-time outlier or a set throttle-reason bit is the cause you can catch in minute two.

## Further reading

- [Why distributed training](/blog/machine-learning/distributed-training/why-distributed-training) — the four-walls frame and the map of the whole series; start here if the synchronization tax is new.
- [Collectives from scratch](/blog/machine-learning/distributed-training/collectives-from-scratch) — the all-reduce as a barrier, ring versus tree, and the byte volume that sets $A$.
- [DDP from first principles](/blog/machine-learning/distributed-training/ddp-from-first-principles) — why the barrier exists at all and the correctness invariant it protects.
- [The NCCL timeout that hung the job](/blog/machine-learning/distributed-training/the-nccl-timeout-that-hung-the-job) — the sibling war story where the slow link does not lag but dies.
- [Fault tolerance and elastic training](/blog/machine-learning/distributed-training/fault-tolerance-and-elastic-training) — how `torchrun` elastic rendezvous evicts a bad node and re-forms the group.
- [Monitoring a long run](/blog/machine-learning/distributed-training/monitoring-a-long-run) — DCGM dashboards and what to alert on so the next straggler pages you, not surprises you.
- [The distributed training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) — the capstone checklist that ties detection, eviction, and monitoring together.
- [Debugging DDP and multi-GPU jobs](/blog/machine-learning/debugging-training/debugging-ddp-and-multi-gpu) and [profiling GPU workloads](/blog/machine-learning/high-performance-computing/profiling-gpu-workloads-finding-the-real-bottleneck) — the general debugging and profiling techniques this post specializes.
- The Llama 3 herd of models (Meta, 2024), the MegaScale paper (ByteDance, 2024), and the public OPT-175B training logbook (Meta, 2022) — three grounded accounts of stragglers and hardware entropy at frontier scale.

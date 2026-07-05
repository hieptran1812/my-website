---
title: "Throughput Regressions: When Yesterday's Job Was Faster and Nobody Changed Anything"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "How to measure training throughput and MFU reliably enough to tell a real regression from noise, catalog the eight confounds that quietly move tokens per second, and bisect a genuine slowdown in the right order — hardware and data before code."
tags:
  [
    "distributed-training",
    "multi-gpu",
    "multi-node",
    "mfu",
    "throughput",
    "profiling",
    "nccl",
    "pytorch",
    "gpu",
    "ml-systems",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Distributed Training"
author: "Hiep Tran"
featured: true
readTime: 41
---

Monday morning the run was doing 84,700 tokens per second. Tuesday morning, the exact same job — same git commit, same config, same cluster, same nodes as far as the scheduler could tell — was doing 59,000. Nobody had merged anything. Nobody had touched the launch script. The loss curve was still descending, no NCCL warnings, no crashes, no out-of-memory. And yet a third of the throughput had simply evaporated between one day and the next. Somebody opened a bug titled "training got slower??" with two question marks, which is the correct number of question marks, because this is one of the most genuinely confusing failure modes in distributed training: the regression that appears with no change to explain it.

The instinct in that moment is to reach for `git bisect`, and it is exactly the wrong first move. Throughput — tokens per second, and its hardware-normalized cousin MFU — is the single best number for telling you whether a distributed run is healthy. It rolls compute, communication, memory, and the data pipeline into one scalar you can watch on one dashboard. But that same number is *noisy*, and it has a long list of confounds that move it without any code changing at all: a hot rack throttling two nodes, a cold storage cache on a fresh data shard, NCCL silently falling back from InfiniBand to TCP sockets, a driver update that changed which matmul kernel gets picked, or — maddeningly — nothing wrong at the machine at all, just longer sequences in today's data making the *steps* slower while the *tokens* are exactly as fast as before. A "regression" is very often not what it seems.

![a decision tree that starts from a throughput drop and branches into hardware, data, interconnect, and software families of cause each ending in a concrete detectable signal](/imgs/blogs/throughput-regressions-1.webp)

The tree above is the map for the whole post, and it is the thing I wish I could hand every engineer before they open the bisect. By the end you will be able to compute MFU from first principles and know why it, not raw tokens per second, is the number to track; know what MFU is *good* at each scale so you can tell "healthy 38%" from "broken 38%"; run down the eight confounds that move throughput, each with the one signal that detects it; measure tokens per second honestly enough that the number stops lying to you; and bisect a genuine regression in the order that finds it fastest — hardware and data before code, because those are cheaper to check and more often the culprit. This is the throughput chapter of the [Distributed Training in the Trenches](/blog/machine-learning/distributed-training/why-distributed-training) series, and it sits squarely on the third of the four walls the series is built around: the run is too slow. The difference between an engineer who chases ghosts for two days and one who fixes it before lunch is almost entirely about knowing which of these boxes to open first.

## Throughput is the one number — and it lies

Every distributed training run is scored, ultimately, on two questions: is it converging, and how fast is it going? The loss curve answers the first. Throughput answers the second, and it is the number that decides your wall-clock time to a finished model and, because GPUs are rented by the hour, your bill. If you train a fixed number of tokens — and after [Chinchilla](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) most of us size the token budget up front — then tokens per second is inversely proportional to both time-to-done and dollars-to-done. Double your throughput and you halve your training bill. That is why it is the north-star metric of the entire discipline: it is the one scalar that ties compute efficiency, communication overhead, memory pressure, and data-pipeline health together into a single line you can put on a dashboard and watch.

But there is a trap hiding in "tokens per second," and it is the first thing to get straight because half of all phantom regressions live here. There are two throughput numbers you could report, and they are not the same:

- **Steps per second** — how many optimizer steps complete per wall-clock second. Easy to measure (just count iterations), and completely dependent on how much work is in a step.
- **Tokens per second** — how many training tokens flow through the model per wall-clock second. This is the one that actually maps to progress and to cost, because the model learns from tokens, not from steps.

They differ by the number of tokens in a step, and that number is *not constant* across a run. If your batches are variable-length and you pack or pad to a per-batch maximum, then the token count per step drifts as the data drifts. A shard of long documents makes each step heavier — fewer steps per second — while the tokens per second, the thing you actually care about, may be dead flat. If you are watching steps per second, that data shift reads as a 30% regression. If you are watching tokens per second, it reads as nothing, because nothing happened. **Measure tokens per second, never steps per second**, and you have already immunized yourself against one of the two worked examples at the end of this post.

Even tokens per second, measured correctly, is noisy. Boost clocks wander with temperature and power. The first steps after a checkpoint resume are slow while caches warm. A shared filesystem has a noisy neighbor. So you cannot treat a single number from a single interval as ground truth. The whole game is turning a noisy, confounded signal into a measurement stable enough that a *real* change stands out from the jitter — and then knowing the catalog of things that produce the jitter.

## What MFU actually is

Raw tokens per second has one more weakness: it is not comparable across hardware, model size, or even sequence length. Is 84,700 tokens per second good? You cannot answer that without knowing the model, the GPUs, and how many of them. A 1B model on eight H100s *should* do far more than 84,700; a 70B model on the same hardware would kill for it. To compare across all of that you need to normalize by the hardware's theoretical ceiling, and that normalized fraction is **MFU — Model FLOPs Utilization**.

The definition is simple and worth stating exactly, because people conflate two different versions of it:

$$\text{MFU} = \frac{\text{achieved model FLOP/s}}{\text{peak hardware FLOP/s}}$$

"Model FLOPs" means the floating-point operations the *model's math* requires — the matmuls of the forward and backward pass — and pointedly does **not** count any extra FLOPs you spend on activation recomputation, on redundant work, or on communication. It is the useful arithmetic. Peak hardware FLOP/s is the vendor's headline dense number for your dtype: for an H100 SXM in bf16 that is roughly 989 TFLOP/s per GPU; for an A100 80GB, roughly 312 TFLOP/s. So MFU asks: of the arithmetic the hardware *could* do, what fraction did the model's real math actually use?

The reason this is computable at all — the reason you do not need a profiler to get it — is a clean approximation for how many FLOPs a transformer costs per token. For a dense transformer with $N$ parameters, one forward pass costs about ${2N}$ FLOPs per token (every parameter participates in one multiply and one add), and the backward pass costs about twice that, roughly ${4N}$, because you compute gradients with respect to both the inputs and the weights. Add them:

$$\text{FLOPs per token (fwd + bwd)} \approx 6N$$

This `6N` rule — from the [scaling-laws literature](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) and used in the PaLM and Megatron reports — is the whole trick. It ignores attention's quadratic term (fine for moderate sequence lengths relative to model width) and normalization and activation costs (small), but it is accurate to a few percent for the models most people train, and it means you can compute achieved FLOP/s from a number you already have: tokens per second. Multiply throughput by the per-token cost, divide by the cluster's peak, and MFU falls out:

$$\text{MFU} = \frac{6 N \cdot (\text{tokens/s})}{G \cdot P_\text{peak}}$$

where $G$ is the number of GPUs and $P_\text{peak}$ is the per-GPU peak FLOP/s. That is the entire derivation, and the figure below walks the same arithmetic as a ladder you can read top to bottom.

![a vertical ladder that turns measured tokens per second into MFU by multiplying by six N flops per token then dividing by the aggregate peak of eight GPUs](/imgs/blogs/throughput-regressions-2.webp)

Let me put the running example through it. A 7B-parameter model ($N = 7 \times 10^9$) on eight H100 SXM GPUs, measured at 84,700 tokens per second:

$$\text{achieved} = 6 \times (7\times 10^9) \times 84{,}700 \approx 3.56 \times 10^{15} \text{ FLOP/s} = 3{,}557 \text{ TFLOP/s}$$
$$\text{peak} = 8 \times 989 \text{ TFLOP/s} = 7{,}912 \text{ TFLOP/s}$$
$$\text{MFU} = \frac{3{,}557}{7{,}912} \approx 0.45 = 45\%$$

Forty-five percent. Now that number is portable: it means the same thing on A100s, on H100s, at eight GPUs or 64. When throughput drops from 84,700 to 59,000, MFU drops from 45% to 31%, and *that* is the number to put on the incident — not the raw tokens, which nobody can sanity-check without also knowing the hardware.

Here is the MFU calculator I paste into every training repo. It is deliberately boring; the only subtlety is being honest about `N`.

```python
def transformer_flops_per_token(num_params: int, seq_len: int = None,
                                hidden: int = None, layers: int = None) -> float:
    """Forward+backward FLOPs per token. The 6N term dominates; the
    attention term (12 * layers * hidden * seq_len) matters only for
    long context. Pass seq_len/hidden/layers to include it."""
    flops = 6 * num_params
    if seq_len and hidden and layers:
        flops += 12 * layers * hidden * seq_len  # attention score+context
    return flops

def mfu(tokens_per_sec: float, num_params: int, num_gpus: int,
        peak_flops_per_gpu: float, **attn) -> float:
    achieved = transformer_flops_per_token(num_params, **attn) * tokens_per_sec
    peak = num_gpus * peak_flops_per_gpu
    return achieved / peak

# H100 SXM bf16 dense peak ~= 989 TFLOP/s
H100_BF16 = 989e12
print(mfu(84_700, 7e9, 8, H100_BF16))   # -> 0.4496  (45%)
print(mfu(59_000, 7e9, 8, H100_BF16))   # -> 0.3132  (31%)
```

### MFU versus HFU — the one distinction that trips people up

There is a sibling metric, **HFU — Hardware FLOPs Utilization** — and confusing the two produces its own phantom regression. MFU counts only the model's essential math. HFU counts *every* FLOP the hardware actually executed, including the extra forward passes that [activation checkpointing](/blog/machine-learning/distributed-training/activation-checkpointing) does to save memory. If you enable full activation recomputation, you redo the forward pass during backward, which adds roughly another `2N` per token — so HFU can be about 1.33× your MFU. The hardware is genuinely busier; the *model* is not learning any faster per token. If someone turns on activation checkpointing to fit a bigger batch and your dashboard tracks HFU, you will see utilization *rise* and conclude you got faster, when your tokens per second may have fallen. Track MFU for "am I making progress efficiently," and treat HFU as a separate diagnostic for "how much recomputation am I paying." Never mix them on the same axis.

## What good MFU looks like

MFU is only useful if you know what number is healthy, and the honest answer is that it *depends on scale*, which is exactly why a single global target ("we should be at 50%") causes bad decisions. The wider the job, the more of every step is spent on communication and waiting for the slowest rank, so the achievable MFU falls as GPUs are added even when nothing is wrong.

![a comparison table of MFU targets across four cluster scales listing the typical range the good threshold and the dominant limiter at each scale](/imgs/blogs/throughput-regressions-3.webp)

The table is a rule of thumb, not a spec sheet, and the ranges are for well-tuned dense transformer training in bf16:

| Scale | Typical MFU | "Good" MFU | What limits it |
|---|---|---|---|
| Single GPU | 40–55% | >55% | Kernel efficiency, memory bandwidth |
| 8-GPU node (NVLink) | 40–50% | >50% | All-reduce overlap with backward |
| 64-GPU (InfiniBand) | 35–45% | >45% | Inter-node comms, stragglers |
| 512+ GPU (multi-node) | 30–40% | >40% | Network bandwidth, pipeline bubble |

Two things fall out of this table immediately. First, **the same MFU is a different verdict at different scale**: 38% on one node is a problem worth a day of investigation; 38% at 512 GPUs is a perfectly healthy run you should leave alone. If your alerting fires on an absolute MFU threshold, it will cry wolf on your biggest, most expensive jobs and stay silent on your small ones. Alert on a *change* from that job's own established baseline, not on an absolute number.

Second, the reason the numbers slope downward is the same mechanism that drives the whole series: communication is not free, and the fraction of the step spent on it grows with the world size. On one node over NVLink at roughly 900 GB/s aggregate per GPU, the gradient all-reduce is small relative to compute and overlaps almost entirely with the backward pass — so you keep most of your single-GPU MFU. Cross to InfiniBand between nodes at roughly 200 Gb/s per link and the same all-reduce is an order of magnitude slower per byte; whether it still hides depends entirely on whether your [compute-communication overlap](/blog/machine-learning/distributed-training/overlapping-compute-and-communication) is working. When multi-node MFU craters far below the table, overlap has usually broken — and that is its own war story, told in [why multi-node ran slower than single-node](/blog/machine-learning/distributed-training/multinode-slower-than-single-node).

## The confounds that move throughput

This is the heart of the post. A throughput number is the output of a long chain — GPU clock, thermal state, data loader, interconnect, kernel selection — and a wobble anywhere in that chain moves the number. The figure traces how those inputs flow through two merge points, compute time and step wall time, before they surface in the one scalar you watch.

![a dataflow graph in which clock and thermal state merge into compute time then loader and interconnect delays merge with it into step wall time which produces measured tokens per second](/imgs/blogs/throughput-regressions-4.webp)

The key structural insight the figure encodes: **almost everything reaches you through the step wall time, and the step wall time is a maximum over ranks, not an average.** (That "max over ranks" is the synchronization tax derived in full in [the straggler](/blog/machine-learning/distributed-training/the-straggler); here it is enough to know that one slow rank sets the pace for all of them.) So a confound that hits *one* node — a single hot rack, a single node's bad NIC — degrades the *whole job's* throughput. That is what makes these so slippery: the cause is local, the symptom is global, and the dashboard averages over ranks and hides it. Here is the catalog. For each, the mechanism, and — the part that matters — the one signal that detects it.

### 1. Thermal throttling

A GPU has a maximum junction temperature. As it approaches that limit, the driver reduces the clock to shed heat, and since transformer training is compute-bound, throughput falls roughly in proportion to the clock. An H100 running its SM clock at 1,980 MHz that throttles to 1,350 MHz has lost about a third of its compute, and because of the max-over-ranks rule, if that GPU is in your job, the *whole job* loses that third. Thermal problems are often *rack-local* — a failed fan, a hot aisle, one node near a wall — so they hit a subset of nodes, produce a straggler, and appear overnight when the datacenter warms up or when a neighboring tenant's job heats the shared cooling. Detection: read per-GPU SM clocks and the throttle-reason bitmask. If any GPU reports an active thermal throttle reason, you are done — that is your regression. Cross-linked in full to [the straggler](/blog/machine-learning/distributed-training/the-straggler), because a thermal throttle is the single most common way a straggler is born.

### 2. The data loader

The GPU can only train as fast as it is fed. If the input pipeline cannot deliver the next batch before the current one finishes, the GPU stalls at the top of the step waiting for data, and throughput drops with zero indication that anything is wrong with the model or the hardware. The insidious version is intermittent: a fresh data shard whose files are on cold storage, a caching layer that missed, a new epoch that re-shuffles and forces re-reads, a noisy neighbor saturating the shared NAS. The GPU-side symptom is a stall at the *start* of each step; the loader-side symptom is that the `DataLoader` iterator is blocking. Detection: time the fetch. Wrap `next(iterator)` and measure how long it blocks; if that time is a non-trivial fraction of your step time, the loader is your bottleneck. This is common enough and deep enough to have its own chapter — [the data pipeline at scale](/blog/machine-learning/distributed-training/the-data-pipeline-at-scale) — but the detection is one timer.

### 3. A silent interconnect fallback

NCCL picks a transport at initialization. On a healthy node it uses NVLink between GPUs and InfiniBand (with GPUDirect RDMA) between nodes. But if the InfiniBand fabric is misconfigured, a NIC is down, `NCCL_IB_HCA` points at the wrong device, or a firmware mismatch disables RDMA, NCCL will *silently fall back to TCP sockets over the ethernet management network*. It does not crash. It does not warn at the default log level. It just runs your all-reduce over a link that is ten to fifty times slower, and your multi-node job's throughput collapses while single-node runs stay fine. This is the classic "multi-node is slower than single-node" autopsy. Detection: run once with `NCCL_DEBUG=INFO` and read which transport it chose — the log literally prints `via NET/IB` versus `via NET/Socket`. Full treatment in [multi-node slower than single-node](/blog/machine-learning/distributed-training/multinode-slower-than-single-node).

### 4. A noisy neighbor

On shared infrastructure — a shared parallel filesystem, a shared ethernet fabric, even shared NVSwitch bandwidth on a multi-tenant node — another team's job can consume the resource you depend on. Your storage reads slow down because someone is checkpointing a 70B model to the same Lustre mount; your gradient all-reduce slows because another job is saturating the shared spine switch. Nothing changed in your code or your nodes; the *environment around them* changed. This is the hardest confound to prove because the evidence lives on machines you do not own. Detection: correlate your throughput dips against the cluster's shared-resource metrics (filesystem IOPS, switch utilization) if you have access, or against the simple fact that the dip is not reproducible in isolation — reserve the nodes exclusively and the regression vanishes.

### 5. Clock and power variation

Even with no thermal throttle, boost clocks are not constant. A GPU under a power cap, or one whose neighbors on the same board are drawing hard, will boost less. Two GPUs of the same model can differ by a few percent in sustained clock under identical load — silicon lottery plus board-level power sharing. This is usually small (a couple of percent of throughput), but it sets a *noise floor* on your measurement: if you benchmark on unpinned clocks, run-to-run variation of several percent is expected and does not indicate a regression. Detection: read the SM clock; if it is below the base clock or fluctuating, power/clock variation is contributing. The fix for *measurement* is to pin the clock (below); the fix for *production* is usually to accept the couple of percent.

### 6. A different data distribution

This is the confound that is not a bug at all, and the one that fools the most people. If your data is variable-length and this batch (or this shard, or this curriculum phase) has longer sequences than yesterday's, each step does more compute and takes longer — so *steps per second* falls. But you did more work: more tokens went through. Tokens per second, and therefore MFU, may be *identical*. There is no regression; there is a measurement artifact created by watching the wrong metric. There is even a second-order effect: attention is quadratic in sequence length, so very long sequences can push MFU slightly *up* (more useful FLOPs per token of overhead). Detection: log the tokens-per-step and the sequence-length histogram alongside throughput. If steps per second fell but tokens per second held, close the bug. This is worked example two.

### 7. A library or driver update

Someone rebuilt the container. The base image pulled a new PyTorch, which shipped a new cuDNN, which changed the heuristic that picks which convolution or matmul kernel runs — and the new pick is faster on some shapes and slower on yours. Or a new NCCL changed the default algorithm (Ring vs Tree) at your message size. Or a driver update changed the default persistence or power policy. None of this is in *your* git history, so `git bisect` on your repo will never find it; the change is in the environment, not the code. Detection: diff the environment. Capture the versions of torch, CUDA, cuDNN, NCCL, and the driver on every run and compare the slow run to the fast one. A one-line version delta is often the entire root cause.

### 8. Warm-up not excluded

The most self-inflicted confound. The first several steps of any run — or the first steps after a resume — are slow: CUDA is JIT-compiling kernels, cuDNN is running its autotuner to pick algorithms, allocator caches are cold, the data loader's prefetch queue is empty, and NCCL is establishing its communication rings. A benchmark that includes these steps in its average reports a throughput far below steady state, and if two runs happen to include different numbers of warm-up steps, you get a phantom regression between them. Detection: none needed — just *always exclude the warm-up*. Drop the first 10–50 steps from every timing average, unconditionally.

### The config changes that masquerade as regressions

There is a ninth cause that deserves its own note because it is technically a change you made, but never the change you think you made: an innocent-looking config edit that quietly moved the comms-to-compute ratio. Distributed training throughput is not just about how fast each GPU computes; it is about what fraction of the step is compute versus communication, and several knobs move that fraction without touching a line of model code.

The clearest example is the global batch size and its relationship to gradient accumulation. Say you halve the micro-batch size to fit a longer sequence, and to keep the global batch constant you double the gradient-accumulation steps. The model math is unchanged and the loss curve is identical — but you now run twice as many forward/backward passes per optimizer step, and on a data-parallel job the gradient all-reduce fires once per optimizer step regardless. That is *good* for the comms-to-compute ratio (more compute per all-reduce), so MFU may tick *up*. Conversely, shrinking the accumulation steps means the all-reduce fires more often relative to compute, and on a slow interconnect MFU falls. Someone "just tuning the batch config" can move throughput several percent in either direction and call it a regression when it is a direct, predictable consequence of the comms-to-compute arithmetic.

The other frequent offenders in this family: turning on activation checkpointing (adds recompute FLOPs, lowers MFU but may enable a larger batch that raises it — a net effect you must actually measure); changing the sharding strategy on [FSDP](/blog/machine-learning/distributed-training/fsdp-in-practice) from `SHARD_GRAD_OP` to `FULL_SHARD` (more all-gathers, more comms); and toggling `torch.compile` (a slow first-step compile that, if not excluded as warm-up, tanks a short benchmark). The detection is the environment and config diff from step 3 of the bisection loop, extended to the *training config*, not just library versions. Capture the effective global batch, micro-batch, accumulation steps, sharding strategy, and checkpointing flag in your run fingerprint, and diff those too. A config change is invisible to a code bisect exactly as a driver bump is — it lives in the launch command, not the source tree.

That is the catalog: thermal, loader, interconnect fallback, noisy neighbor, clock variation, data distribution, library/driver, warm-up, and the config change that moved the comms-to-compute ratio. Notice how few of them are code. Notice how many are detectable with a single signal you are probably not logging. The rest of the post is about logging those signals and reading them in the right order.

## Measuring throughput reliably

Before you can diagnose a regression you have to trust the number, and the default way most people measure throughput is wrong in at least three ways at once. The figure contrasts the naive protocol with the honest one.

![a before and after comparison contrasting naive timing that includes warm-up and measures steps per second against honest timing that excludes warm-up meters tokens per second and pins clocks](/imgs/blogs/throughput-regressions-5.webp)

The honest protocol has six rules, and each one closes off a confound from the catalog:

1. **Exclude warm-up.** Drop the first 10–50 steps. Closes confound 8.
2. **Synchronize before you stop the clock.** CUDA is asynchronous: `loss.backward()` returns before the GPU has finished. Timing without a `torch.cuda.synchronize()` measures Python-launch time, not compute time, and is meaningless. This is the single most common timing bug in the ecosystem.
3. **Measure tokens per second, not steps per second.** Count real tokens through the model, so variable sequence length cannot fool you. Closes confound 6.
4. **Average over steady state, and long enough.** One step is pure noise. Average over enough steps (a few hundred) that clock jitter and loader hiccups wash out. Closes confounds 4 and 5.
5. **Pin the clocks when benchmarking.** For a controlled A/B, lock the SM clock with `nvidia-smi -lgc` so clock variation cannot contaminate the comparison. Closes confound 5 for measurement.
6. **Compare like-for-like.** Same batch size, same sequence length, same hardware, same node count. A throughput comparison across different anything is not a comparison.

Here is the steady-state tokens-per-second meter I actually use. It excludes warm-up, synchronizes correctly, and reports tokens per second so sequence length cannot lie to it.

```python
import torch, time

class ThroughputMeter:
    def __init__(self, warmup_steps: int = 20):
        self.warmup = warmup_steps
        self.step = 0
        self.tokens = 0
        self.t0 = None

    def start_step(self):
        # First real (post-warmup) step: sync, then start the clock.
        if self.step == self.warmup:
            torch.cuda.synchronize()
            self.t0 = time.perf_counter()

    def end_step(self, tokens_this_step: int):
        if self.step >= self.warmup:
            self.tokens += tokens_this_step
        self.step += 1

    def tokens_per_sec(self) -> float:
        torch.cuda.synchronize()           # GPU work must be done before we read the clock
        elapsed = time.perf_counter() - self.t0
        return self.tokens / max(elapsed, 1e-9)

# In the loop:
meter = ThroughputMeter(warmup_steps=20)
for batch in loader:
    meter.start_step()
    loss = model(batch).loss
    loss.backward()
    optimizer.step(); optimizer.zero_grad()
    meter.end_step(tokens_this_step=batch["input_ids"].numel())  # real tokens, not steps
if rank == 0 and meter.step % 200 == 0:
    print(f"steady-state throughput: {meter.tokens_per_sec():,.0f} tok/s")
```

Two details earn their keep. The `torch.cuda.synchronize()` inside `tokens_per_sec()` is not optional — without it you are timing kernel *launches*, which are microseconds, and your reported throughput will be fiction. And `batch["input_ids"].numel()` counts the actual tokens in the batch, padding included; if you pack sequences or want to exclude padding, count the non-pad tokens instead, but be consistent so the metric is comparable to itself over time.

### Reading per-GPU clocks and throttle reasons

The first thing to check on any suspected regression is whether the hardware is throttling, and `nvidia-smi` will tell you in one query. This is the command that ends most thermal investigations in under a minute:

```bash
# Per-GPU: SM clock, temperature, and the active throttle-reason bitmask.
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,\
clocks_throttle_reasons.active,clocks_throttle_reasons.hw_thermal_slowdown,\
clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.hw_power_brake_slowdown \
  --format=csv
```

A healthy GPU shows an active-reasons value of `0x0000000000000001` (that bit means "GPU idle" or "clocks are as requested" — no throttle). Any of the thermal or power bits set is your smoking gun. On a real cluster you run this across every node — through `pdsh`, `srun`, or your orchestrator — and look for the outliers:

```bash
# Fan out across a SLURM allocation and flag any GPU below the expected clock.
srun --nodes=8 --ntasks-per-node=1 bash -c '
  hostname
  nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active \
    --format=csv,noheader'
```

When two nodes come back at 1,350 MHz and 86°C while the other six sit at 1,980 MHz and 61°C, you have found your regression without touching the code. That is worked example one.

### Pinning clocks for a clean benchmark

When you need to *compare* two configurations — the suspected-slow build against the known-good one — clock variation is a confound you must eliminate, and you do that by locking the clock so both runs execute at exactly the same frequency:

```bash
# Enable persistence mode, then lock the SM clock to a fixed value for benchmarking.
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc 1980,1980     # lock SM clock to 1980 MHz (min,max the same)
# ... run both A and B benchmarks here, now clock-invariant ...
sudo nvidia-smi -rgc               # reset to default boost behavior when done
```

Pin the clock, run A, run B, and any difference that survives is real — not silicon lottery, not thermal drift. Do **not** leave clocks pinned in production, though: you would forfeit the free boost headroom that a healthy, cool GPU normally enjoys. Pinning is a measurement tool, not an operating mode.

### A rolling-MFU logger and dashboard

The single highest-leverage thing you can do to make regressions *legible* is to log MFU continuously, so a drop is a visible step in a time series rather than a surprise someone notices two days late. This wraps the meter above and emits MFU on a rolling window:

```python
import torch, time, collections

class RollingMFU:
    def __init__(self, num_params, num_gpus, peak_flops_per_gpu, window=50):
        self.per_token = 6 * num_params
        self.peak = num_gpus * peak_flops_per_gpu
        self.buf = collections.deque(maxlen=window)   # (tokens, seconds) per step

    def update(self, tokens_this_step, seconds_this_step):
        self.buf.append((tokens_this_step, seconds_this_step))

    def mfu(self):
        tok = sum(t for t, _ in self.buf)
        sec = sum(s for _, s in self.buf)
        achieved = self.per_token * (tok / max(sec, 1e-9))
        return achieved / self.peak

roller = RollingMFU(7e9, 8, 989e12, window=50)
t_prev = time.perf_counter()
for step, batch in enumerate(loader):
    loss = model(batch).loss; loss.backward()
    optimizer.step(); optimizer.zero_grad()
    torch.cuda.synchronize()
    now = time.perf_counter()
    roller.update(batch["input_ids"].numel(), now - t_prev)
    t_prev = now
    if rank == 0 and step > 20 and step % 50 == 0:
        m = roller.mfu()
        print(f"step {step}  rolling MFU {m:5.1%}")
        # emit to your metrics backend; alert on a sustained drop vs baseline
```

Ship the rolling MFU (and the raw tokens per second, and the tokens-per-step) to whatever you use for dashboards. The alert that actually works is not "MFU below 40%" — that cries wolf at scale — but "MFU dropped more than 15% below this job's own 200-step trailing baseline for more than 200 steps." A per-GPU throttle-reason alarm belongs right next to it. Monitoring a long run well is its own discipline; the point here is that a regression you can *see* on a time series is a regression you fix in an hour instead of a day.

### An environment-diff script

Confound 7 — the library or driver update — is invisible to your repo's history, so you have to capture the environment explicitly on every run and diff it when a regression appears. This dumps the versions that actually move kernel selection and comms:

```python
import torch, json, subprocess

def env_fingerprint() -> dict:
    fp = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nccl": ".".join(map(str, torch.cuda.nccl.version())),
        "gpu": torch.cuda.get_device_name(0),
    }
    try:  # driver version straight from nvidia-smi
        fp["driver"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version",
             "--format=csv,noheader"], text=True).splitlines()[0].strip()
    except Exception:
        fp["driver"] = "unknown"
    return fp

if rank == 0:
    print("ENV_FINGERPRINT " + json.dumps(env_fingerprint()))
    # e.g. {"torch":"2.4.1","cuda_runtime":"12.4","cudnn":90100,
    #       "nccl":"2.21.5","gpu":"NVIDIA H100 80GB HBM3","driver":"550.90.07"}
```

Log this line on every run. When a regression appears, `diff` the fingerprint of the slow run against the last-known-fast run. A single changed field — `nccl` went from 2.20 to 2.21, or `driver` bumped — is frequently the whole answer, and it is an answer `git bisect` on your own code will never, ever find.

### Timing the data loader

The loader confound is the one people most often miss because it looks like a slow model rather than a slow pipeline. The detection is a two-line change: measure how long the training loop blocks waiting for the next batch, and compare it to the compute time. If the fetch is a meaningful fraction of the step, the GPU is starving and no amount of model optimization will help.

```python
import torch, time

fetch_time = compute_time = 0.0
it = iter(loader)
for step in range(num_steps):
    t0 = time.perf_counter()
    batch = next(it)                    # blocks iff the loader can't keep up
    t1 = time.perf_counter()
    loss = model(batch).loss; loss.backward()
    optimizer.step(); optimizer.zero_grad()
    torch.cuda.synchronize()            # so compute_time is real GPU time
    t2 = time.perf_counter()
    if step >= 20:                      # skip warm-up
        fetch_time  += (t1 - t0)
        compute_time += (t2 - t1)

frac = fetch_time / (fetch_time + compute_time)
print(f"data-loader stall fraction: {frac:.1%}")   # >5% means the loader is your bottleneck
```

On a healthy pipeline this fraction is a percent or two — the loader's background workers stay ahead of the GPU and `next()` returns instantly from a prefilled queue. When a fresh shard lands on cold storage or a cache misses, the fraction jumps to 20% or 40% and *that* is your regression, cleanly attributed to the input pipeline rather than the model. If it is high, the levers are in [the data pipeline at scale](/blog/machine-learning/distributed-training/the-data-pipeline-at-scale): more `num_workers`, a larger `prefetch_factor`, `pin_memory`, and warming the storage cache before the run.

### Confirming the interconnect transport

For any multi-node regression, prove which transport NCCL chose before assuming anything about the model. A single run with `NCCL_DEBUG=INFO` prints the transport per connection, and the difference between a healthy and a fallen-back fabric is one word in the log:

```bash
NCCL_DEBUG=INFO torchrun --nnodes=8 --nproc_per_node=8 \
  --rdzv_backend=c10d --rdzv_endpoint=$MASTER:29500 train.py 2>&1 | grep -E "NET/|Channel"
```

Healthy — RDMA over InfiniBand:

```console
NCCL INFO Channel 00 : 0[0] -> 8[0] [receive] via NET/IB/0/GDRDMA
NCCL INFO Channel 00 : 8[0] -> 0[0] [send]    via NET/IB/0/GDRDMA
```

Fallen back — TCP sockets over the management network, ten to fifty times slower per byte:

```console
NCCL INFO Channel 00 : 0[0] -> 8[0] [receive] via NET/Socket/0
NCCL INFO Channel 00 : 8[0] -> 0[0] [send]    via NET/Socket/0
```

If you see `NET/Socket` where you expected `NET/IB`, the whole multi-node slowdown is explained, and the fix is in the fabric config — `NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`, GPUDirect, the NIC — not in your training loop. The full autopsy is in [multi-node slower than single-node](/blog/machine-learning/distributed-training/multinode-slower-than-single-node).

### Establishing the noise floor

Every number above is only actionable relative to how noisy your measurement is, so the first thing to do on a new setup — before you ever declare a regression — is to *measure the noise*. Run the same benchmark five times, back to back, with clocks pinned, and compute the coefficient of variation:

```python
import statistics
runs = [bench_tokens_per_sec() for _ in range(5)]   # pinned clocks, warm-up excluded
mean = statistics.mean(runs)
cv = statistics.pstdev(runs) / mean
print(f"tok/s: {mean:,.0f}  noise floor (CV): {cv:.1%}")
# pinned clocks: expect CV ~1-2%.  unpinned: 3-6% is normal.
```

That coefficient of variation is your detection threshold. If pinned-clock CV is 1.5%, a sustained 4% drop is real and worth chasing; a 1% wobble is noise. If you skip this step, you will either chase phantom 2% regressions forever or miss a real 3% one under 5% unpinned jitter. **You cannot detect a regression smaller than your noise floor** — which is the entire argument for pinning clocks and averaging over hundreds of steps before comparing anything.

## Bisecting a real regression

Sometimes it *is* real — the number genuinely dropped and stayed down. Now you need a systematic loop, and the crucial thing is the *order*: check the cheap, common causes before the expensive, rare ones. Hardware and data before code. The figure lays out the loop as an ordered sequence.

![an ordered timeline of regression diagnosis steps beginning with confirming the drop in tokens per second and ending with confirming the fix restored MFU](/imgs/blogs/throughput-regressions-6.webp)

Walk it in order and you will almost never reach the expensive step:

1. **Confirm the drop is real — and in tokens per second.** Re-measure with the honest protocol: warm-up excluded, synchronized, tokens per second, steady state. Half of all reported regressions die here because they were steps per second moving with sequence length, or a warm-up artifact. If tokens per second is genuinely down and stable, proceed.

2. **Read per-GPU clocks and throttle reasons.** The one-line `nvidia-smi` query across all nodes. Thermal and clock throttling are the most common *real* cause of an overnight regression and the cheapest to check. If a subset of GPUs is throttling, you are done — go cool them.

3. **Diff the environment.** Compare the version fingerprint of the slow run against the fast one: torch, CUDA, cuDNN, NCCL, driver. A container rebuild is invisible to your code history and a frequent culprit. If a version moved, you have a strong suspect to A/B.

4. **Check for a data-distribution shift.** Log the sequence-length histogram and tokens-per-step for the slow window versus the fast one. If the data got heavier (longer sequences, a new shard), your steps slowed but your tokens may be fine — reconfirm on tokens per second and likely close the bug.

5. **Check the interconnect (if multi-node).** Run once with `NCCL_DEBUG=INFO` and confirm it is still choosing InfiniBand, not sockets. A silent fabric fallback halves multi-node throughput and shows nothing at the default log level.

6. **Only now, `git bisect` the code.** If hardware, environment, and data are all clean and the interconnect is healthy, *then* the cause is in your code, and you bisect it. Because tokens per second is a continuous number rather than a pass/fail, you script the bisection with a threshold:

```bash
# git bisect with a throughput gate. bench.sh runs a short pinned-clock
# benchmark and exits 0 if steady-state tok/s >= threshold, else 1.
git bisect start
git bisect bad HEAD                 # today: slow
git bisect good v1.4.0              # last release: known fast
git bisect run bash bench.sh 80000  # threshold: 80k tok/s

# bench.sh (sketch):
#   nvidia-smi -lgc 1980,1980
#   tps=$(python bench_throughput.py --steps 300 --warmup 30 --report tps)
#   nvidia-smi -rgc
#   awk -v t="$tps" -v thr="$1" 'BEGIN{exit !(t>=thr)}'
```

Bisecting on a noisy metric works only if you pin the clocks and average over enough steps that the noise floor is well below your threshold margin — which is exactly why the measurement discipline from the previous section is a prerequisite, not an afterthought. If your run-to-run noise is 10% and the regression is 8%, no bisection can find it; you must drive the noise down first.

7. **A/B the suspected change, then confirm the fix.** Whether the suspect came from the environment diff or the bisect, isolate it: run the old and new versions back-to-back on pinned clocks with identical data, and confirm the throughput difference reproduces and then disappears when you revert. Then re-measure MFU end to end and confirm it is back to baseline. A fix you did not measure is a hypothesis, not a fix.

#### Worked example: the "regression" that was two hot nodes

Here is the Monday-to-Tuesday drop from the intro, resolved. A 7B model on 64 H100 SXM GPUs across eight DGX nodes, data-parallel, bf16. Monday's steady-state throughput, measured honestly: 84,700 tokens per second, MFU 45%. Tuesday: 59,000 tokens per second, MFU 31%. Same commit, same config.

Step 1, confirm: re-measured with warm-up excluded and on tokens per second — genuinely down, and stable across 300 steps. Real regression.

Step 2, clocks. The one-line query fanned across all eight nodes came back with six nodes at 1,980 MHz and 60–63°C, and **two nodes at 1,350 MHz and 84–87°C, with `sw_thermal_slowdown` active**. There it was. Two nodes had crossed into thermal throttling — a facilities issue, a partially failed fan tray in one rack that the overnight datacenter warm-up had pushed over the edge.

The arithmetic confirms it end to end. Those two nodes (16 of the 64 GPUs) dropped to about $1350/1980 \approx 0.68$ of their compute. But because of the max-over-ranks synchronization tax, the *whole 64-GPU job* runs at the pace of its slowest rank, so the entire job dropped to roughly 68% of baseline: $0.68 \times 84{,}700 \approx 57{,}600$ tokens per second — within measurement noise of the observed 59,000. One local thermal fault, one third of a 64-GPU cluster's throughput, no code involved. Total diagnosis time once the clock query existed: about four minutes. The fix was facilities cooling the rack; the *prevention* was adding a per-GPU throttle-reason alarm so the next occurrence pages in minutes instead of being discovered by a human squinting at tokens per second the next morning.

#### Worked example: the "regression" that was longer sequences

A different team, a different Tuesday. A model reported a 30% throughput regression after a data refresh — steps per second had fallen from 0.30 to 0.21, a textbook-looking 30% drop, and someone had already started bisecting a data-loader refactor merged the same week.

Step 1, confirm — and this is where it ended. The batch size was fixed at 256 *sequences*, but the new data shard had longer documents: the mean sequence length had risen from about 1,600 tokens to about 2,300. Tokens per step therefore rose from $256 \times 1{,}600 = 409{,}600$ to $256 \times 2{,}300 = 588{,}800$. Compute per step rose proportionally, so of course steps per second fell. But tokens per second:

$$\text{before: } 0.30 \times 409{,}600 \approx 122{,}900 \text{ tok/s}$$
$$\text{after: } 0.21 \times 588{,}800 \approx 123{,}600 \text{ tok/s}$$

Flat. MFU, computed from tokens per second, was identical at both — the model was exactly as efficient as before. There was no regression. There was a team watching steps per second while their data got heavier. The "fix" was to change the dashboard to plot tokens per second and MFU instead of steps per second, and to log the sequence-length histogram next to throughput so the next data shift is self-explanatory rather than alarming. The data-loader refactor was entirely innocent; a day of bisection was averted by one honest re-measurement.

The two cases together are the whole lesson, and the table below sets them side by side — one a genuine hardware fault, one a pure measurement artifact, both first mistaken for a bad code push.

![a comparison matrix placing the thermal throttling case and the longer sequences case side by side across symptom wrong guess real cause fix and the alert to add](/imgs/blogs/throughput-regressions-7.webp)

## Stress-testing the metric across scales

MFU is a good north star, but it is not equally informative at every scale and configuration, and knowing where it gets slippery keeps you from misreading it. Walk it through the hard cases the way you would stress-test any decision.

**At 64 GPUs when all-reduce dominates.** On a wide data-parallel job over InfiniBand, a growing share of each step is the gradient all-reduce. If overlap is working, the comms hide behind the backward pass and MFU stays in the high 30s; if it is not, MFU falls and the *cause is communication, not compute*. The trap is reading a scale-driven MFU decline as a regression. It is not — it is the table's expected slope. The tell that it is a *real* regression on top of the expected slope is a drop below that job's own established baseline, which is why relative alerting beats absolute thresholds every time. When comms genuinely dominate, the fix lives in [overlapping compute and communication](/blog/machine-learning/distributed-training/overlapping-compute-and-communication), not in chasing kernels.

**On PCIe instead of NVLink.** Move the same eight-GPU job from an NVLink node (roughly 900 GB/s aggregate per GPU) to a PCIe Gen4 box (roughly 32 GB/s per direction, shared) and the all-reduce is more than an order of magnitude slower per byte. MFU can fall from 45% to the low 20s with zero code change — and this is a *real* throughput difference, correctly reported by MFU, that you would completely miss if you only compared raw tokens per second without knowing the interconnect. MFU earns its keep here: it makes the interconnect's cost visible as a number.

**When the batch is tiny.** Shrink the per-GPU batch toward one and MFU collapses, because the GPU becomes launch-bound and memory-bandwidth-bound rather than compute-bound — there is not enough arithmetic per kernel to feed the tensor cores, and the fixed per-step overheads (kernel launches, the all-reduce latency floor) dominate. This is not a regression either; it is the arithmetic-intensity floor. If someone reduces the batch to fit memory and MFU tanks, the diagnosis is "batch too small to saturate the hardware," and the fix is gradient accumulation or a memory technique that lets the batch grow, not a hunt for a slow kernel.

**When a straggler is present.** MFU is computed from the *whole job's* tokens per second, so a single throttling rank drags the reported MFU down for all of them — exactly worked example one. This is a feature: MFU faithfully reports the job's real efficiency including the tax of its slowest member. But it is also why MFU alone cannot tell you *which* rank is slow. When MFU drops and the cause is a straggler, you need per-rank timing to localize it, which is the whole subject of [the straggler](/blog/machine-learning/distributed-training/the-straggler). MFU says "the job is slow"; per-rank timing says "rank 41 is slow."

**When the model is pipeline-parallel.** With pipeline parallelism, the pipeline bubble — the idle time at the start and end of each micro-batch schedule — shows up as lower MFU, and the bubble fraction depends on the number of micro-batches. Add micro-batches and the bubble shrinks and MFU rises; a config change that reduced the micro-batch count will *look* like a regression while being a direct consequence of the bubble formula. The same discipline applies: capture the pipeline config in the fingerprint so the cause is legible.

The through-line across all five cases: MFU is an honest number, but "MFU fell" is a symptom with a scale-and-config-dependent set of causes, and the diagnosis always routes back to the confound tree and the per-rank, per-config visibility that lets you tell an expected slope from a genuine fault. When the number says "slow" but not "where," that is the handoff to [profiling a distributed run](/blog/machine-learning/distributed-training/profiling-a-distributed-run) — the profiler across ranks shows you which kernel, which collective, which stall is eating the step that MFU only summarizes.

## Case studies and real numbers

MFU targets are not aspirational hand-waving; the large public training reports state theirs, and calibrating against them keeps your own expectations honest. A few reference points, cited as reported and approximate — treat them as order-of-magnitude anchors, not guarantees for your hardware.

| System (source) | Reported MFU | Hardware / scale |
|---|---|---|
| GPT-3 175B (as tabulated in the PaLM paper) | ~21% | reported baseline |
| Gopher 280B (PaLM paper table) | ~33% | — |
| Megatron-Turing NLG 530B (PaLM paper table) | ~30% | A100 cluster |
| PaLM 540B (Chowdhery et al., 2022) | ~46% | 6,144 TPU v4 |
| Megatron-LM (Narayanan et al., 2021) | ~52% of peak | ~163 TFLOP/s per A100 (of 312) |

The lesson in that table is not any single number but the *spread and the trend*: MFU in the 30–50% band is the norm for large-scale training, the very best tuned runs reach the low 50s, and — visible across the GPT-3-to-PaLM progression — a large part of the gains over time came not from faster chips but from raising MFU through better parallelism, better overlap, and better kernels. When your run posts 45% MFU at eight GPUs, you are in excellent company; when it posts 45% at 512 GPUs, you are doing better than most published systems. Two more concrete anchors worth internalizing: the Megatron-LM report's headline was roughly 163 achieved TFLOP/s per A100 against a 312 dense peak — that is where the ~52% comes from, and it is about the ceiling for dense transformer training on that generation. And the reason PaLM introduced MFU as a metric at all was precisely this post's thesis: raw hardware-FLOPs numbers were being gamed by counting recomputation, so they defined a *model*-FLOPs metric that could not be inflated by doing redundant work. MFU exists because throughput lies, and someone got tired of it.

There is also a published-postmortem genre worth reading for pattern-matching: the large open training runs (OPT-175B's logbook is the famous one) are full of exactly these throughput incidents — hardware faults taking nodes down, mysterious slowdowns traced to specific machines, restarts that came back slower until a bad node was evicted. The recurring shape is always the same: a global symptom, a local cause, and a diagnosis that hinged on per-rank or per-GPU visibility. If you take one operational habit from this whole post, make it *per-GPU* logging, because the average across ranks is exactly the view that hides every one of these.

## When to reach for this (and when not to)

Not every throughput wobble deserves an investigation, and knowing when to *not* chase one is as valuable as the diagnosis loop.

- **Chase it** when the drop is sustained (hundreds of steps, not a spike), confirmed on tokens per second (not steps per second), and larger than your measured noise floor. That combination means something real changed.
- **Do not chase it** when the "drop" is within run-to-run variance. If your unpinned-clock noise floor is 3–5% and MFU wobbled 4%, that is silicon lottery and boost-clock drift, not a regression. Pinning clocks and re-measuring is the whole investigation — if the gap vanishes, there was no gap.
- **Do not chase it in steps per second at all.** If steps per second fell but tokens per second held, the data got heavier and nothing is wrong. Fix the dashboard, not the code.
- **Do not micro-optimize MFU that is already at target for your scale.** Grinding from a healthy 42% to 44% at 512 GPUs is rarely worth an engineer-week; a genuine regression from 45% to 31% always is. Know your scale's target from the table so you can tell "leave it alone" from "all hands."
- **Prevent instead of diagnose** where you can. The three cheap preventions — always exclude warm-up and meter tokens per second; log a rolling MFU with a *relative* alert; add a per-GPU throttle-reason alarm — turn most of the confounds in this post from two-day mysteries into automatic pages. The instrumentation is a hundred lines. The alternative is discovering a third of your cluster was throttling for a week because nobody was looking at per-GPU clocks.

## Key takeaways

- **Track MFU, not raw tokens per second, as the health metric** — it is the only throughput number comparable across hardware, model size, and scale. Compute it with $\text{MFU} = 6N \cdot (\text{tokens/s}) / (G \cdot P_\text{peak})$.
- **Measure tokens per second, never steps per second.** Variable sequence length makes steps per second lie; tokens per second is robust. This one change kills the single most common phantom regression.
- **A regression is usually not a code bug.** It is far more often a hot rack, a cold data shard, a silent NCCL socket fallback, a container rebuild, or a warm-up artifact. Check hardware and data before you `git bisect`.
- **The step wall time is a max over ranks, so a local fault is a global symptom.** One throttling node halves a 64-GPU job. Per-GPU logging is the only view that exposes it; the rank-average hides it.
- **Honest timing has six rules:** exclude warm-up, `torch.cuda.synchronize()` before reading the clock, meter tokens per second, average over steady state, pin clocks for A/B, compare like-for-like.
- **What is "good" MFU depends on scale** — above 50% on one node, high-30s at 512 GPUs. Alert on a *relative* drop from a job's own baseline, never on an absolute threshold.
- **Bisect in order: hardware, environment, data, then code.** The `nvidia-smi` throttle-reason query and the environment version diff each end most investigations in minutes, before any bisection.
- **A fix you did not re-measure is a hypothesis.** Confirm the regression reproduces, disappears on revert, and that MFU is back to baseline.

## Further reading

- [Why distributed training](/blog/machine-learning/distributed-training/why-distributed-training) — the series intro and the four-walls frame throughput sits inside.
- [The distributed training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) — the capstone decision-and-debugging checklist this post feeds into.
- [The straggler: how one slow GPU halved a 64-GPU job](/blog/machine-learning/distributed-training/the-straggler) — the synchronization-tax derivation behind thermal throttling as a throughput killer.
- [The data pipeline at scale](/blog/machine-learning/distributed-training/the-data-pipeline-at-scale) — the loader confound in full: sharded datasets, `num_workers`, prefetch, and not starving the GPU.
- [Multi-node slower than single-node](/blog/machine-learning/distributed-training/multinode-slower-than-single-node) — the silent interconnect fallback and how to catch it with `NCCL_DEBUG=INFO`.
- [Overlapping compute and communication](/blog/machine-learning/distributed-training/overlapping-compute-and-communication) — why MFU falls with scale and what keeps the all-reduce hidden behind the backward pass.
- [Profiling a distributed run](/blog/machine-learning/distributed-training/profiling-a-distributed-run) — when the throughput number says "slow" but not "where," reach for the profiler across ranks.
- [Scaling a 7B LLM from 1 to 64 GPUs](/blog/machine-learning/distributed-training/scaling-a-7b-llm-1-to-64-gpus) — the full MFU-at-each-step journey this post's measurement discipline was built for.
- Chowdhery et al., *PaLM* (2022) — the paper that defined MFU and tabulated the reported values above. Narayanan et al., *Megatron-LM* (2021) — the ~52%-of-peak large-scale reference.

---
title: "Debugging Distributed Jobs: Where to Even Look When 64 GPUs Go Quiet"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "A field guide to debugging distributed training: classify the failure, pull a stack trace from every rank, and find the one process that is not where the others are."
tags:
  [
    "distributed-training",
    "multi-gpu",
    "multi-node",
    "debugging",
    "nccl",
    "pytorch",
    "py-spy",
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

It is 2:40 a.m. Your 64-GPU run — eight nodes, eight H100s each — was at step 12,300 and climbing. Now the loss curve on your dashboard has flatlined. Not spiking, not NaN. Just... stopped, the last point twenty minutes old. You SSH into the head node, tail the log, and it ends mid-sentence on a perfectly normal line: `step 12,300 | loss 2.41 | 41,000 tok/s`. No traceback. No `CUDA out of memory`. No exit code. `nvidia-smi` on the head node says all eight GPUs are pinned at 100% utilization. The job is, by every surface measure, working as hard as it possibly can. And it is doing absolutely nothing.

This is the moment that separates single-GPU debugging from distributed debugging. On one GPU, a broken program crashes, and the crash carries a stack trace that points at the line. Here you have 64 processes spread across 8 machines, the failure is a *silence* rather than a crash, the symptom (a flat loss on rank 0) is almost certainly not where the cause lives, and the one log you are staring at — rank 0's, because that is the one everyone prints — is statistically the *least* likely to contain the answer. If you approach this like a single-process bug, you will lose the night.

The good news: distributed failures are not infinitely varied. They come in four flavors, each with a different tell-tale signal and a different first move. The entire discipline of debugging a distributed job is (1) figuring out *which* of the four you have, fast, and (2) collecting state from *every* rank so that the outlier — the one process that is not where the others are — reveals itself. This post is the meta-method: the triage runbook, the tools, and the mental model that turns "64 GPUs went quiet" from a panic into a ten-minute diagnosis. The deep-dives on individual failures live in sibling posts; this is the map that tells you which one to open.

![a tree splitting a misbehaving distributed job into hang, crash, silently wrong, and slow with each tell-tale signal](/imgs/blogs/debugging-distributed-jobs-1.webp)

If you are new to the vocabulary — *rank* (the global index of a process, 0 to world_size−1), *world size* (total number of processes), *collective* (an operation like all-reduce that every rank must call together), *watchdog* (a background thread that kills the job if a collective takes too long) — start with [why distributed training exists at all](/blog/machine-learning/distributed-training/why-distributed-training) and come back. Everything below assumes you have a job that launches and mostly runs, and now, sometimes, does not.

## Why distributed debugging is a different sport

Before the runbook, it is worth being precise about *why* the usual instincts fail. Four properties of distributed systems break the single-process debugging loop, and every technique in this post exists to counter one of them.

**The state is spread across N processes on M machines.** On one GPU, the entire state of your program — the Python stack, the tensors, the optimizer — lives in one address space you can inspect with one debugger. At 64 GPUs, the "state of the program" is the *union* of 64 independent Python interpreters, each with its own stack, its own CUDA context, its own view of the world. There is no single place to attach a debugger and see everything. A bug is often a *disagreement* between ranks — rank 37 thinks it is on step 12,301 while everyone else is on 12,300 — and a disagreement is invisible from inside any single process. You have to gather state from all of them and diff.

**Failures are often silent — a hang, not a crash.** A crash is a gift: it stops the program at the fault and hands you a stack trace. The characteristic distributed failure is a *hang*, where a collective never completes and every rank sits politely blocked inside NCCL waiting for a peer that will never arrive. Nothing crashes. Nothing prints. The GPUs may even show 100% utilization because a stuck NCCL kernel spins. Silence is the symptom, and silence carries no information — you have to go extract it.

**The symptom appears far from the cause.** Because a collective is a barrier — every rank must arrive before any rank proceeds — one misbehaving rank freezes the entire job. But the *visible* symptom (the whole cluster is idle) is global and uniform; it points nowhere. The cause is local and specific: rank 37 hit a CUDA error, or took a data-dependent branch no one else took, or its NIC dropped a packet. The distance between "everything is frozen" and "rank 37's third attention layer" is the entire debugging problem.

![a merge diagram showing one stalled rank and the other ranks all blocking at the same all-reduce until the whole job freezes](/imgs/blogs/debugging-distributed-jobs-2.webp)

**Rank-0 bias: you log from rank 0, but the bug is on rank 37.** By deep convention, distributed training scripts guard almost all logging with `if rank == 0:` — otherwise every line prints 64 times and the log is unreadable. That convention is correct for normal operation and a trap during a failure. If a hang is equally likely to originate on any rank, the probability that rank 0 — the one whose log you can see — is the culprit is ${1/N}$. At a world size of 64, that is under 2%. Put the other way: **98% of the time, the informative rank is one you are not looking at.** Chasing rank 0's log during a hang is, quite literally, looking under the streetlight.

Keep these four in mind, because the runbook below is nothing but their direct countermeasures: gather state from *all* ranks (counters the spread), get a stack trace out of a process that will not crash on its own (counters the silence), find the outlier (counters symptom-far-from-cause), and never trust rank 0's log alone (counters rank-0 bias).

## The four failure classes, and why each needs a different move

Every distributed training failure I have chased reduces to one of four classes. Getting the class right in the first ten minutes is 80% of the work, because each class rules out different causes and calls for a different tool. Guess wrong — treat a slow run as a hang, or a silently-wrong run as fine — and you burn hours.

| Class | What you see | Root cause family | First move |
|---|---|---|---|
| **Hang** | Loss log stops; no traceback; GPUs at 100% or 0% util forever | A collective mismatch, a data-dependent divergence, a dead NIC, a deadlock | All-rank stack traces; the flight recorder |
| **Crash** | One rank prints a traceback and exits; others hang or die | CUDA error, OOM, assertion, an uncaught exception on one rank | Read *all* logs; find the first non-zero exit and its traceback |
| **Silently wrong** | Job runs to completion but the loss is wrong / worse than one GPU | A reduction bug, wrong gradient scaling, a broken shard, non-determinism | Single-GPU reference; param-hash agreement checks |
| **Slow** | Runs fine, but at a fraction of expected tokens/s | A straggler, a comms wall, a data-loader stall, thermal throttling | Profile every rank; compare per-rank step times |

Two things about this table earn emphasis. First, **hang and crash often masquerade as each other.** A crash on rank 37 (say, a CUDA assert) kills that process — but ranks 0–36 and 38–63 do not know rank 37 died. They are still blocked in the next all-reduce, waiting for a peer that is now a corpse. So a *crash on one rank presents as a hang on all the others*. If you only look at the survivors, you see a hang; the crash is buried in a log you have to go find. This is the single most common misclassification, and it is why "read all logs" appears twice in the table.

Second, **silently wrong is the scariest**, because it does not announce itself. A hang wastes a night; a silently-wrong run wastes a week and a training budget, and you only discover it when the model is bad at eval. It is the one class where "the job finished successfully" is not evidence of correctness. We will spend real time on the technique that catches it — a single-GPU reference — because nothing else will.

The rest of this post walks the classes in the order you will actually confront them: triage first (which class?), then the class-specific techniques, then the two failures worth a full worked example.

## The first ten minutes: a triage runbook

When the loss flatlines, resist the urge to read code. You do not yet know *which* of the four classes you have, and every class points at different code. Spend the first ten minutes gathering signals, in a fixed order, so that by minute ten you have a classification and a suspect rank.

![a timeline of the first ten minutes of triage from nvidia-smi across nodes to classifying the failure](/imgs/blogs/debugging-distributed-jobs-3.webp)

**Minute 0 — `nvidia-smi` across every node, not just the head.** GPU utilization is your first fork. Three readings, three meanings:

- **100% util, frozen** on all GPUs: a stuck kernel — very often a NCCL collective spinning on a peer that will not arrive. This is a *hang*. (Utilization counts "a kernel is resident," not "useful work is happening," so a spinning NCCL kernel reads as 100%.)
- **0% util** on all GPUs: everyone is blocked on the host side (a Python lock, a barrier, a data loader that stalled, a broken rendezvous). Also a hang, but a *host-side* one — different suspects.
- **A mix — one node at 0%, the rest at 100%**: the 0% node is your straggler or your dead rank. You have already localized the problem to a node in sixty seconds.

Do it on all nodes at once. On SLURM:

```bash
# Fan nvidia-smi out to every node in the allocation, tag each line with the host.
srun --overlap -N "$SLURM_NNODES" --ntasks-per-node=1 \
  bash -c 'echo "=== $(hostname) ==="; nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw --format=csv,noheader'
```

```console
=== node-03 ===
0, 100 %, 79214 MiB, 698.44 W
1, 100 %, 79210 MiB, 701.10 W
...
=== node-07 ===
0, 0 %, 78980 MiB, 71.30 W    <-- the odd node: idle, cold, memory still held
1, 0 %, 78975 MiB, 70.98 W
```

That `node-07` at 0% util and 71 W (idle power) while every other node burns 700 W is the entire diagnosis in one screen. The rest of triage is confirming it.

**Minute 2 — count the live ranks.** A hang and a partial crash look identical from the loss curve. The difference is whether all 64 processes are still alive. Count them:

```bash
srun --overlap -N "$SLURM_NNODES" --ntasks-per-node=1 \
  bash -c 'echo "$(hostname): $(pgrep -c -f train.py) python procs"'
```

If every node reports 8, it is a true hang (everyone alive, everyone stuck). If one node reports 7, a process died — a *crash* that presents as a hang. That missing process is your suspect, and its log (not rank 0's) has the traceback.

**Minute 4 — pull a stack trace from every rank.** This is the heart of the method and the single highest-value action in distributed debugging. A hung process will not crash on its own, so you have to go in and ask it where it is. `py-spy dump` reads the stack of a running Python process without stopping it and without any cooperation from the program — no signal handler, no pre-instrumentation:

```bash
# Dump the Python (and native) stack of every training process on this node.
for pid in $(pgrep -f train.py); do
  echo "===== pid $pid ====="
  py-spy dump --pid "$pid" --native
done
```

Fan that across all nodes and collect the output. What you are looking for is not any single stack — it is the *outlier*. Sixty-three ranks will show a stack ending in the same place:

```console
Thread 0x7f... (idle): "MainThread"
    all_reduce (torch/distributed/distributed_c10d.py:2050)
    _allreduce_grads (train.py:214)
    backward (train.py:198)
    train_step (train.py:176)
```

All sixty-three are blocked in `all_reduce` at `train.py:214`. And one rank — rank 37 — is somewhere else entirely:

```console
Thread 0x7f... (active): "MainThread"
    _sample_batch (data.py:88)
    __next__ (data.py:61)
    train_step (train.py:170)
```

Rank 37 is still in the data loader at `train.py:170` — it never reached the all-reduce. **The rank that is not where the others are is your culprit.** Everyone else is waiting for rank 37 at the barrier; rank 37 is stuck fetching a batch. Now you have a rank *and* a line, and you have spent four minutes.

**Minute 8 — read ALL the logs, aggregated.** Even with a stack trace, read every rank's log, because the *first* error in time is the cause and everything after is a cascade. Aggregate them so you can sort by timestamp and by rank:

```bash
# Merge per-rank logs, prefix each line with its rank, sort by time.
for f in logs/rank_*.log; do
  r=$(echo "$f" | grep -oE '[0-9]+')
  sed "s/^/[rank $r] /" "$f"
done | sort -k2 > logs/merged.log
tail -n 50 logs/merged.log
```

The first line with `Traceback` or `NCCL WARN` or `CUDA error`, sorted by wall-clock time, is where the failure began. If rank 37's log shows `CUDA error: an illegal memory access` five seconds before everyone else went quiet, that is your cause and the hang is a cascade.

**Minute 10 — classify.** By now you know: all ranks alive or not (hang vs crash), util 100% or 0% (kernel-stuck vs host-stuck), which rank is the outlier, and what its log says. You have a class and a suspect. Now — and only now — you open the code, or the class-specific deep-dive: [the NCCL timeout that hung the job](/blog/machine-learning/distributed-training/the-nccl-timeout-that-hung-the-job) for a collective mismatch, [the straggler](/blog/machine-learning/distributed-training/the-straggler) for a slow node, [silent NaN at scale](/blog/machine-learning/distributed-training/silent-nan-at-scale) for a corrupted reduction.

The payoff of running this fixed sequence, rather than improvising, is measurable. On the teams I have worked with, the mean time from "the job is stuck" to "I know the class and the suspect rank" dropped from roughly an hour of unstructured log-reading to under ten minutes once the runbook — and the per-rank logging that feeds it — was in place. Those numbers are approximate and depend on cluster size and how much instrumentation you wired in advance, but the shape is robust: the improvement comes almost entirely from *not* starting with rank 0's log and *not* guessing at code before you have classified. The runbook is cheap to internalize precisely because it is mechanical — the same six checks in the same order every time, so at 3 a.m. you execute rather than reason. And because it is mechanical, it is teachable: a new engineer on their first on-call can run it as a checklist and reach the same suspect rank you would, which is worth more than any single clever diagnosis.

## Hung or healthy? Reading the vital signs

The triage runbook assumes you already know the job is stuck. But a large run has slow steps for legitimate reasons — a checkpoint save, an eval pass, a data shard boundary — and you do not want to page yourself for a 90-second checkpoint. So it helps to have a crisp picture of what a *healthy* step looks like, so a hung one is unmistakable.

![a two-column comparison of a healthy training step versus a hung step across utilization, collective position, and loss logging](/imgs/blogs/debugging-distributed-jobs-4.webp)

A healthy step has three signatures. GPU utilization *cycles* — it dips during the data-loader wait and the optimizer step and peaks during the matmuls, so a `watch -n1 nvidia-smi` shows numbers moving. All ranks are at (or near) the *same* collective at any instant, because they run the same code in lockstep. And the loss log *ticks down* on a steady cadence. A hung step inverts all three: utilization is *pinned* flat (100% on a spinning kernel, or 0% on a host block), one rank is at a *different* operation than the rest, and the loss log has *stopped*. Two out of three is enough to call it.

### The mechanism: why one rank hangs all of them

The reason a single stuck rank freezes 64 GPUs is worth deriving, because it explains why every technique here is about *finding the one* rather than *watching the many*. A collective — all-reduce, all-gather, broadcast — is a synchronization barrier with a data-exchange attached. Formally, a collective call on rank $r$ returns only when **all** ranks in the process group have called the *matching* collective: same operation, same dtype, same shape, same element count, same root (for rooted ops like broadcast). Miss any of those and the ranks that did call it block, waiting for the ones that did not.

Let $W$ be the world size. The completion condition for a collective is a logical AND across all $W$ ranks:

$$\text{collective completes} \iff \bigwedge_{r=0}^{W-1} \big(\text{rank } r \text{ called the matching op}\big)$$

A single false term — one rank that took a different branch, called a different collective, or died — makes the whole conjunction false, and every other rank blocks indefinitely. There is no timeout at the semantic level; the operation is simply *not done*. The only thing that eventually breaks the deadlock is the **watchdog**: a background thread in PyTorch's NCCL process group that tracks how long each collective has been outstanding and aborts the process group if one exceeds a timeout. That timeout defaults to 10 minutes (600 seconds) in recent PyTorch and is set via the `timeout` argument to `init_process_group` — which is exactly why a hang shows up as a 10-minute gap in the log before, sometimes, a `Watchdog caught collective operation timeout` error.

This AND-across-ranks structure is the whole reason for the rank-0-bias math from earlier. The failure is one false term out of $W$; the probability it is the term you are watching is ${1/W}$. **The mechanism forces the method: you must look at all ranks, because the informative one is, with probability $(W-1)/W$, not the one you log.**

### Deadlock: the same collectives in a different order

There is a subtler cousin of the missing-rank hang that trips people who "did everything right": every rank calls exactly the same collectives, but in a *different order*. NCCL matches collectives by the order in which each rank issues them, not by name. So if rank 0 calls `all_reduce(A)` then `all_reduce(B)`, while rank 1 calls `all_reduce(B)` then `all_reduce(A)` — perhaps because a conditional reordered them — rank 0's first all-reduce is waiting to pair with rank 1's first all-reduce, but rank 1's first is a *different* tensor. Both ranks are inside a collective; neither is missing; every rank is "participating." And yet they deadlock, each waiting for the other to issue the operation it already moved past.

This is the failure mode where all-rank stack traces are *least* helpful, because every rank shows a perfectly reasonable stack inside `all_reduce` — just a *different* all-reduce. It is exactly what the flight recorder was built for: its per-rank sequence numbers reveal that rank 0's collective #101 and rank 1's collective #101 have different shapes or tensor fingerprints, which no stack trace would show. The usual culprits are a data-dependent branch that only some ranks take (an `if loss.isnan(): extra_reduce()` that fires on one rank), an unguarded logging all-reduce inside a rank-conditional block, or gradient reductions on parameters that only exist on some ranks. The discipline that prevents it: **every collective must be reached by every rank on every iteration, unconditionally.** If a collective can be skipped by a branch, it will eventually be skipped by only *some* ranks, and you will spend a night on it.

### The countermeasure: make the hang self-report

Ten minutes of manual `py-spy` per incident is fine once. On a run that hangs intermittently, automate it. Two mechanisms make a hung process report its own stack without you SSH-ing in:

```python
# In your training entrypoint, before init_process_group.
import faulthandler, signal, os

# 1) On-demand: `kill -USR1 <pid>` prints all Python thread stacks to stderr.
faulthandler.register(signal.SIGUSR1, all_threads=True)

# 2) Watchdog: if any step takes longer than 600s, dump all thread stacks
#    automatically, then repeat every 600s while still hung.
faulthandler.dump_traceback_later(600, repeat=True, exit=False)
```

`faulthandler` is in the standard library, costs nothing when idle, and works even when the GIL is held by a stuck thread (it uses a separate OS-level timer). Combined with the NCCL flight recorder (next section), a well-instrumented job dumps enough state *at the moment of the hang* that you rarely need to attach live.

## The rank-0 bias trap and a logging strategy that survives it

We keep returning to rank-0 bias because it is the mistake even experienced engineers make at 3 a.m. The fix is a logging discipline you set up *before* the failure, not during it. Three rules.

**Rule 1 — put the rank in every line.** If a log line does not say which rank emitted it, it is nearly useless during a distributed failure, because the whole game is attributing behavior to a specific rank. Bake the rank into the log format once:

```python
import logging, os

rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))

logging.basicConfig(
    format=f"%(asctime)s [rank {rank:>3} lr {local_rank}] %(levelname)s %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("train")
```

**Rule 2 — write per-rank log files, then aggregate.** Interleaving 64 ranks onto one stdout is unreadable, and truncation loses the rank that matters. Give each rank its own file and aggregate at read time (the `sort -k2` merge from the runbook). Under `torchrun`, you get this for free with `--redirect` and `--tee`:

```bash
torchrun \
  --nnodes=8 --nproc_per_node=8 \
  --rdzv_backend=c10d --rdzv_endpoint="$HEAD_NODE:29500" \
  --redirects=3 --tee=3 \
  --log_dir=logs/ \
  train.py --config prod.yaml
```

`--redirects=3` sends each worker's stdout and stderr to a file under `--log_dir`; `--tee=3` also echoes rank 0 to your console so you keep a live view without losing the others. Now the moment a rank dies, its traceback is on disk in `logs/<rank>/stderr.log` even if your terminal scrolled past it.

**Rule 3 — log the outlier, not the average.** The instinct is to log mean throughput or mean loss across ranks. That hides the exact thing you need: the *worst* rank. A straggler is invisible in the mean and obvious in the max. So periodically reduce the *extremes* and log them:

```python
import torch, torch.distributed as dist

def log_step_time_stats(step_time_s: float):
    """All-reduce min/max/mean of per-rank step time; rank 0 logs the spread."""
    t = torch.tensor([step_time_s], device="cuda")
    tmin = t.clone(); dist.all_reduce(tmin, op=dist.ReduceOp.MIN)
    tmax = t.clone(); dist.all_reduce(tmax, op=dist.ReduceOp.MAX)
    tsum = t.clone(); dist.all_reduce(tsum, op=dist.ReduceOp.SUM)
    mean = tsum.item() / dist.get_world_size()
    if dist.get_rank() == 0:
        skew = tmax.item() / max(tmin.item(), 1e-9)
        log.info(f"step_time min={tmin.item():.3f}s max={tmax.item():.3f}s "
                 f"mean={mean:.3f}s skew={skew:.2f}x")
```

If `skew` is 1.05x, your ranks are balanced. If it is 1.8x, one rank is 80% slower than the fastest and is capping the whole job — a straggler, and you have caught it from a one-line metric instead of a profiler session. This single log line has saved me more debugging hours than any tool, because it turns "the run feels slow" into a number that trips an alert.

## The core techniques: an escalation ladder

With triage done and logging in place, here is the toolkit — six techniques, ordered cheapest-first. Start at the top; only climb when the rung you are on fails to explain the symptom. Most bugs surrender by rung three.

![a vertical ladder of debugging techniques from reproduce smaller up to a single-GPU reference run](/imgs/blogs/debugging-distributed-jobs-5.webp)

### 1. Reproduce smaller

Before you debug at 64 GPUs, try to reproduce at 2. A bug that reproduces on two GPUs on one node is a hundred times cheaper to iterate on: faster to launch, cheaper to hold idle, and free of the multi-node interconnect as a variable. Keep a `debug.yaml` that shrinks every dimension while keeping the code path identical:

```bash
# Same code, two GPUs, one node, tiny model, a handful of steps.
torchrun --standalone --nproc_per_node=2 train.py \
  --config debug.yaml   # hidden=256, layers=2, seq_len=128, max_steps=50
```

Crucially, a bug's *reproducibility at small scale tells you its class*. If it reproduces at 2 GPUs, it is a logic bug in your parallel code — a wrong reduction, a mis-sharded tensor — and you can chase it on a laptop-sized job. If it *only* appears at 64 GPUs or only across nodes, it is a scale/environment bug — a NIC, a topology issue, a NCCL algorithm that only kicks in above some size, a race that needs enough ranks to lose. That fork alone saves you from grinding on the wrong hypothesis. For the DDP-specific version of "it works on one GPU but not two," the [debugging DDP and multi-GPU](/blog/machine-learning/debugging-training/debugging-ddp-and-multi-gpu) guide has the pattern catalog.

### 2. Bisect: which rank, which step, which layer

Bisection is the oldest trick in debugging and it maps cleanly onto three distributed axes:

- **Which rank?** Binary-search the world size. If it hangs at 64 but not 2, try 8, then 16, then 32 — the threshold where it starts often names the cause (a topology change, an algorithm switch, a NIC that only participates above a node count).
- **Which step?** If it fails at step 12,300 every time, it is data-dependent — a specific batch, a shard boundary, an eval trigger. If it fails at a *random* step, it is a race or a hardware flake. Deterministic-step failures are gold; make the run deterministic (fixed seed, `torch.use_deterministic_algorithms(True)`) and it will fail at the same step every time, which lets you iterate.
- **Which layer?** For a silently-wrong or NaN result, bisect the model. Register forward hooks that check each layer's output for NaN/Inf and log the first offender's name. The [silent NaN at scale](/blog/machine-learning/distributed-training/silent-nan-at-scale) post has the full hook harness.

### 3. All-rank stack traces

Covered in the runbook, but it earns its own rung because it is the technique. Two flavors: `py-spy dump` (external, needs nothing from your program, best for a live hang you did not anticipate) and `faulthandler` (internal, you wire it in advance, best for a hang you want to auto-capture). A third, `TORCH_DISTRIBUTED_DEBUG=DETAIL`, is worth setting on any run you expect to debug:

```bash
TORCH_DISTRIBUTED_DEBUG=DETAIL \
TORCH_SHOW_CPP_STACKTRACES=1 \
  torchrun --nnodes=8 --nproc_per_node=8 ... train.py
```

`TORCH_DISTRIBUTED_DEBUG=DETAIL` makes PyTorch check, at every collective, that all ranks agree on the tensor shapes and the sequence of operations — and it *raises with a clear message* when they do not, converting a silent hang into a loud, attributable error like `Detected mismatch between collectives on ranks`. It adds overhead, so you would not leave it on for a production run, but it is the fastest way to catch a collective mismatch (the classic cause of a hang where one rank calls all-reduce with a different shape than the rest).

### 4. The flight recorder for collectives

When a hang happens with no clear outlier — everyone is blocked in the *same* all-reduce, so `py-spy` shows all 64 in the identical spot — you need to know *which* collective, in program order, they disagree about. PyTorch's NCCL **flight recorder** (sometimes "Flight Recorder") keeps a ring buffer of the last N collective operations per rank, with their sequence numbers, shapes, and completion state. When the watchdog fires, it dumps the buffer. Turn it on with environment variables:

```bash
# Keep the last 2000 collective records per rank; dump them when a collective times out.
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/shared/flight/rank_trace_

torchrun --nnodes=8 --nproc_per_node=8 ... train.py
```

On a hang, each rank writes its trace to `/shared/flight/rank_trace_<rank>`. Now you can compare *sequence numbers*: if 63 ranks completed all-reduce #48,201 and are blocked entering #48,202, but rank 37's last *started* collective is #48,201 (never completed), rank 37 is where the divergence is — even though all 64 stacks looked identical. The flight recorder answers the question `py-spy` cannot: not "where is each rank in the Python code" but "which collective, by global sequence number, did they stop agreeing on." Recent PyTorch ships an analyzer (`torch/utils/flight_recorder`, invoked as an `fr_trace` script) that ingests all ranks' dumps and prints the mismatch for you. The [NCCL debugging deep-dive](/blog/machine-learning/distributed-training/nccl-debugging-deep-dive) sibling reads a real dump end to end.

### 5. Assert that all ranks agree

The defense against *silently wrong* is to check the invariants that must hold and are easy to violate. Under data parallelism, two invariants are load-bearing: **model parameters must be identical on every rank** (after the initial broadcast and after every optimizer step, since gradients were all-reduced), and **input batches must differ across ranks** (each rank processes a different slice, or you are wasting the parallelism and effectively training on a smaller dataset). Both are one `all_reduce` away from verifiable:

```python
import torch, torch.distributed as dist

def param_hash(model) -> torch.Tensor:
    """A cheap order-sensitive fingerprint of all parameters, as one scalar."""
    h = torch.zeros(1, dtype=torch.float64, device="cuda")
    for i, p in enumerate(model.parameters()):
        # weight each param by its index so permutations don't collide
        h += (i + 1) * p.detach().double().sum()
    return h

def assert_params_identical(model):
    """Every rank must hold the same weights. Fails loudly if a shard drifted."""
    local = param_hash(model)
    gathered = [torch.zeros_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    ref = gathered[0]
    for r, g in enumerate(gathered):
        if not torch.allclose(g, ref, rtol=0, atol=1e-6):
            raise RuntimeError(
                f"PARAM DRIFT: rank {r} hash {g.item():.6e} != rank 0 {ref.item():.6e}")

def assert_inputs_differ(input_ids):
    """Adjacent ranks must see different data, or DP is a no-op."""
    local = input_ids.double().sum().reshape(1)
    gathered = [torch.zeros_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    if len({round(g.item(), 3) for g in gathered}) == 1:
        raise RuntimeError("ALL RANKS GOT THE SAME BATCH: DistributedSampler misconfigured")
```

Call `assert_params_identical(model)` once after wrapping in DDP/FSDP and again every few hundred steps. If it ever fires, you have caught a divergence — a rank whose gradients did not get all-reduced, an optimizer that stepped on unsynchronized state, a checkpoint that restored one shard wrong — *at the step it happened*, not a week later at eval. Call `assert_inputs_differ` once at startup; a stunning number of "why is my big run no better than one GPU" mysteries are a `DistributedSampler` that was never given `set_epoch` or was constructed without `num_replicas`/`rank`, so all 64 ranks train on rank 0's shard.

### 6. The single-GPU reference run

The final rung, and the only reliable catch for *silently wrong*: run the identical model and the identical batch on a single GPU, and compare outputs against the distributed run to a tight tolerance.

![a branching diagram feeding one frozen batch into a single-GPU run and the cluster run then comparing their logits](/imgs/blogs/debugging-distributed-jobs-7.webp)

The logic: distributed parallelism is supposed to be *numerically equivalent* to the single-GPU computation (up to floating-point reordering, which is small and bounded). So if you freeze the inputs and the seed, run one forward pass on one GPU to get reference logits, and run the same frozen batch through the distributed model, the two logit tensors must match to roughly floating-point tolerance. If they match, your parallel plumbing is correct and any badness is in the data or the hyperparameters. If they *diverge* — beyond what fp reordering explains — you have a real reduction or sharding bug, and the *size* and *location* of the divergence localizes it (diverges only in the last layer? your loss reduction; diverges everywhere? your gradient all-reduce or a wrong shard).

```python
# Freeze one batch and one seed; capture single-GPU reference logits offline.
# ref_logits.pt was produced by running this batch on 1 GPU with the same init.
def check_against_reference(model, frozen_batch, ref_logits, atol=1e-3, rtol=1e-3):
    model.eval()
    with torch.no_grad():
        logits = model(frozen_batch["input_ids"].cuda()).float().cpu()
    if not torch.allclose(logits, ref_logits, atol=atol, rtol=rtol):
        diff = (logits - ref_logits).abs()
        raise RuntimeError(
            f"REFERENCE MISMATCH: max|Δ|={diff.max():.3e} at index {diff.argmax().item()} "
            f"(atol={atol}); parallel path is not numerically equivalent to 1 GPU")
    model.train()
```

Yes, it costs a single-GPU run to produce the reference. That cost is trivial next to discovering at the end of a week-long run that your loss was subtly wrong the whole time. For any new parallelism configuration — a new sharding policy, a custom collective, a hand-rolled tensor-parallel layer — the single-GPU reference is the acceptance test you run before you trust it at scale.

## The distributed-specific toolbox

Most of the tools above are worth naming in one place, with the exact knob that turns them on, because half of distributed debugging is knowing that the tool exists.

| Tool / knob | What it does | When to reach for it |
|---|---|---|
| `py-spy dump --pid` | Prints a live process's Python + native stack without stopping it | A hang, right now, no prior instrumentation |
| `faulthandler` (SIGUSR1 / `dump_traceback_later`) | Process dumps its own thread stacks on signal or on a timer | Auto-capture a hang you expect to recur |
| Flight recorder (`TORCH_NCCL_TRACE_BUFFER_SIZE`) | Ring buffer of recent collectives, dumped on timeout | A hang where all ranks look identical — find the mismatched collective by seq number |
| `NCCL_DEBUG=INFO` / `WARN` | NCCL logs its rings, transports, and errors | Comms-layer failures: transport fallback, IB errors, unexpected algorithm |
| `TORCH_DISTRIBUTED_DEBUG=DETAIL` | Validates shape/op agreement across ranks every collective | Catch a collective mismatch as a loud error instead of a hang |
| `dist.monitored_barrier()` | A barrier that reports *which* ranks failed to arrive | A liveness probe to name the dead/stuck rank |
| `torch.profiler` / Nsight Systems | Per-rank timelines of compute and comms | A *slow* run — find the wall (comms-bound? loader-bound?) |

Two of these deserve a code snippet. First, a **liveness probe** built on `monitored_barrier`. A plain `dist.barrier()` on the NCCL backend will *itself* hang if a rank is missing — useless for diagnosis. But `monitored_barrier` on a side **gloo** (CPU) group has a timeout and, on failure, raises an error that *names the ranks that did not check in*:

```python
import datetime, torch.distributed as dist

# At init: create a CPU (gloo) group alongside your NCCL group, just for probing.
gloo_group = dist.new_group(backend="gloo")

def liveness_probe(timeout_s: int = 60):
    """Returns cleanly if every rank arrives; raises naming the missing ranks otherwise."""
    try:
        dist.monitored_barrier(
            group=gloo_group,
            timeout=datetime.timedelta(seconds=timeout_s),
            wait_all_ranks=True,   # report *all* stragglers, not just the first
        )
    except RuntimeError as e:
        # e lists the ranks that failed to check in — exactly the suspects.
        log.error(f"LIVENESS FAILED: {e}")
        raise
```

Drop `liveness_probe()` at the top of your training loop behind a "every N steps" guard, and a hang converts itself into an error message naming rank 37 within 60 seconds, instead of a 10-minute watchdog silence. Second, the **NCCL comms log** — set `NCCL_DEBUG=INFO` and read the ring construction and transport choices; a multi-node run that quietly fell back from InfiniBand to TCP sockets (a 10x bandwidth cliff) announces it here, and it is the first thing to check when [multi-node is slower than single-node](/blog/machine-learning/distributed-training/multinode-slower-than-single-node). The [NCCL debugging deep-dive](/blog/machine-learning/distributed-training/nccl-debugging-deep-dive) reads a full `NCCL_DEBUG=INFO` dump line by line; [profiling a distributed run](/blog/machine-learning/distributed-training/profiling-a-distributed-run) covers the `torch.profiler` and Nsight side for the *slow* class.

## When it's a crash: the exit code and the first traceback

The *crash* class deserves its own treatment, because a crash is where the rank-0 trap bites hardest and where `torchrun` does more for you than people realize. Recall the pattern: one rank hits a real error — a CUDA fault, an OOM, an assertion — and dies, while every other rank keeps running until it blocks in the next collective waiting for the dead one. On your terminal (rank 0) you see a *hang*. The traceback that explains everything is sitting in the dead rank's stderr, which you will only find if you were logging per-rank.

`torchrun` (the elastic launcher) is built to surface this for you, but only if you let it. When a worker process exits non-zero, the elastic agent detects it, tears down the other workers on that node, and prints an **error summary** that names *which* rank failed with *what* exit code and, if you decorated your entrypoint, the actual Python traceback. The decoration is a one-line change:

```python
from torch.distributed.elastic.multiprocessing.errors import record

@record
def main():
    setup_distributed()
    train()

if __name__ == "__main__":
    main()
```

The `@record` decorator catches the exception on the failing worker, writes it to a JSON file the agent reads, and hoists it into the launcher's error summary — so instead of a bare `exited with code 1`, you get the child rank's traceback attributed to its global rank. Without it, `torchrun` can tell you rank 37 died with exit code 1 but not *why*, and you are back to grepping per-rank logs.

Read the **exit code** itself, because it classifies the crash before you read a single line of Python:

| Exit code | Signal / meaning | Usual cause |
|---|---|---|
| `1` | Uncaught Python exception | Assertion, shape error, a genuine bug — read the traceback |
| `137` | `SIGKILL` (128 + 9) | The OS OOM-killer reaped the process — host RAM exhausted, often the data loader |
| `139` | `SIGSEGV` (128 + 11) | A native segfault — a bad CUDA op, a driver/NCCL crash, corrupted memory |
| `134` | `SIGABRT` (128 + 6) | A C++ `abort()` — often a NCCL or CUDA runtime assertion |
| `-6` / `-9` | Negated signal (some launchers) | Same as 134 / 137 above, reported as a negative number |

An exit code of 137 sends you to look at *host* memory and the data loader, not the GPU; a 139 sends you to the native layer (CUDA, NCCL, a custom kernel), where a Python traceback would not have helped anyway and `TORCH_SHOW_CPP_STACKTRACES=1` will. The single most important rule for a crash: **find the *first* worker to fail in wall-clock time.** The elastic agent may report several failures, but the later ones are almost always cascades — a rank that died with `NCCL error: remote process exited` because its *peer* crashed first. Sort the failures by timestamp; the earliest non-cascade traceback is the cause. Everything after it is the blast radius. For the OOM case specifically — the most common crash in practice — the [out-of-memory debugging](/blog/machine-learning/debugging-training/out-of-memory-debugging) patterns tell you whether it is a genuine capacity problem or a fragmentation/leak one.

## Symptom to technique: the decision table

Put the four classes and their techniques together and you get the on-call decision table — the thing to pin above your desk. Classify the symptom, read across, execute.

![a table mapping each failure class to its first move, key tool, and what the tool reveals](/imgs/blogs/debugging-distributed-jobs-6.webp)

Read it as a flowchart. Start at *is the job making progress?* If no and everyone is alive, it is a **hang** → `py-spy` all ranks, then the flight recorder if the stacks are identical. If no and someone is missing, it is a **crash** → read all logs, find the first traceback by timestamp. If yes but the result is wrong, it is **silently wrong** → single-GPU reference and param-hash agreement. If yes but slow, it is **slow** → profile every rank and compare step times, looking for the straggler or the comms wall. Every branch names a first move and a tool; none of them is "read rank 0's log and hope."

The value of the table is that it *pre-commits* you to a move before the adrenaline hits. At 3 a.m., you do not want to be reasoning from first principles about whether to profile or to `py-spy`; you want to classify in one glance and execute a rehearsed action.

#### Worked example: a hang triaged to one rank in five minutes

A 32-GPU run (4 nodes × 8 A100s) training a 7B model hangs at step 8,410. The loss log stops. Here is the actual sequence, timed.

*Minute 0.* `srun` fans `nvidia-smi` across all four nodes. All 32 GPUs show 100% utilization, ~400 W. So: everyone is busy on a stuck kernel — a *kernel-side hang*, most likely NCCL. Not a host-side stall, not a dead process (they would be at 0%/70 W).

*Minute 1.* Count live processes: every node reports 8. All 32 ranks alive. So it is a true hang, not a crash-masquerading-as-hang. Suspects narrow to collective mismatch or data-dependent divergence.

*Minute 2.* `py-spy dump` across all ranks. Thirty-one ranks show the same stack: blocked in `all_reduce` inside the gradient reduction at `train.py:214`. One rank — rank 19 — shows a different stack: it is inside `model.generate()` at `eval.py:44`. The bug is now obvious in hindsight: an eval hook fired on rank 19 only (a `if rank == 19` left in from a debugging session weeks ago, meant to be `if rank == 0`), so rank 19 went off to run generation while the other 31 entered the training all-reduce. Thirty-one ranks wait at a barrier rank 19 will not reach until it finishes generating — which it cannot, because generation *also* uses collectives that the other ranks are not participating in. Classic deadlock: two groups each waiting for the other.

*Minute 5.* Fix is a one-character change (`19` → `0`), confirmed by grepping for stray rank literals. Total time from page to root cause: five minutes, almost all of it mechanical. Notice what did *not* happen: no reading of rank 0's log (rank 0 was a victim, blocked in all-reduce like the other 30 innocents), no guessing, no reproduce-at-scale. The all-rank stack diff did the entire job. **The rank that is not where the others are was, once again, the whole answer.**

#### Worked example: a silently-wrong run caught by the reference

A team migrates a training script from DDP to FSDP to fit a larger model. The FSDP run launches, runs, checkpoints, finishes — everything green. But eval perplexity is 8% worse than the DDP baseline on the identical data and hyperparameters. No crash, no NaN, no hang. This is the scary class: *silently wrong*, and "the run succeeded" is not evidence of anything.

The debugging move is the single-GPU reference. They freeze one batch and one seed, run it through a 1-GPU forward pass with the pre-migration code to get `ref_logits.pt`, then run the same frozen batch through the FSDP model with `check_against_reference`. It fires immediately: `max|Δ|=3.1e-2`, far beyond the `1e-3` tolerance that fp reordering would explain, and the largest deviation is concentrated in the parameters of the normalization layers.

That localization is the clue. FSDP shards parameters, and the mixed-precision config was set to keep the *compute* in bf16 but — through a misread flag — was also casting the normalization layers' parameters to bf16 for the *reduction*, where DDP had kept them in fp32. LayerNorm/RMSNorm weights are precision-sensitive; the low-precision reduction was quietly degrading them. The single-GPU reference caught in one forward pass what a week of training and an 8% eval regression had failed to make legible. The fix — an explicit `MixedPrecision(reduce_dtype=torch.float32)` on the norm layers — brought the reference diff back under `1e-3` and closed the eval gap. Cost of the diagnosis: one single-GPU forward pass. Cost of not having the reference: a week and a training budget, discovered late. This is why the single-GPU reference is the acceptance test for *any* new parallelism configuration. The FSDP-specific version of this precision trap is covered in [FSDP in practice](/blog/machine-learning/distributed-training/fsdp-in-practice).

## Measuring the thing you are debugging

Two of the four classes — *slow* and *silently wrong* — are only visible if you measure honestly, and distributed measurement has confounds that will fool you into chasing ghosts. If you take one measurement discipline from this post, take this: **synchronize before you time, and time steady state, per rank.**

```python
import torch, time, torch.distributed as dist

def timed_steps(model, batches, warmup=10, measure=50):
    times = []
    for i, batch in enumerate(batches[: warmup + measure]):
        torch.cuda.synchronize()          # wait for prior GPU work to actually finish
        t0 = time.perf_counter()
        loss = train_step(model, batch)   # forward + backward + optimizer + collectives
        torch.cuda.synchronize()          # wait for THIS step's GPU work to finish
        if i >= warmup:                    # discard warm-up (allocator, cuDNN autotune, NCCL warmup)
            times.append(time.perf_counter() - t0)
    local_med = torch.tensor([sorted(times)[len(times)//2]], device="cuda")
    # find the SLOWEST rank — that is the one setting the pace of the whole job
    dist.all_reduce(local_med, op=dist.ReduceOp.MAX)
    if dist.get_rank() == 0:
        log.info(f"slowest-rank median step: {local_med.item():.3f}s")
```

Four confounds that this guards against, and that catch people constantly. **The async trap:** CUDA kernels are asynchronous, so without `torch.cuda.synchronize()` you time the *launch*, not the *execution*, and get absurdly fast (wrong) numbers. **Warm-up:** the first several steps pay one-time costs — the CUDA caching allocator growing, cuDNN autotuning, NCCL establishing its rings — so timing them makes everything look slow. **The data-loader confound:** if the loader can't keep up, your "GPU is slow" is actually "GPU is starved"; you can only tell them apart by checking whether GPU util dips between steps. And **the mean-hides-the-straggler trap:** the job runs at the pace of its *slowest* rank, so you must reduce with `MAX`, not average — the mean will tell you everything is fine while one rank quietly halves your throughput.

#### Worked example: measuring a suspected regression

A run that did 41,000 tok/s last month now does 33,000 — a 20% throughput regression, no code change. Is it real? First, measure with the discipline above: warm-up discarded, synchronized, per-rank max. The slowest-rank median step is 0.62s versus a historical 0.50s — the regression is real and it is one rank. `nvidia-smi` on that rank shows its clock throttled to 1,200 MHz against 1,410 MHz elsewhere: thermal throttling on a node with a failing fan. The "regression" was hardware, not software, and the honest measurement — per-rank, not averaged — pointed straight at it. Averaging would have shown 0.53s, an ambiguous 6% that you might have dismissed. The [throughput-regressions](/blog/machine-learning/distributed-training/profiling-a-distributed-run) workflow chases the software-side confounds when the hardware checks out.

## Case studies and real numbers

A few grounding facts from the public record, because the method above did not emerge from nowhere.

**The OPT-175B logbook.** Meta published the raw training logbook for OPT-175B, and it is the best public document on what large-run debugging actually feels like: dozens of hardware failures, loss spikes, NaN episodes, and manual restarts over the course of the run. The recurring pattern in that logbook is exactly the one this post formalizes — a failure appears as a hang or a spike, the team collects state across ranks, isolates a bad node or a bad step, and restarts from a checkpoint. It is worth reading precisely because it strips away the illusion that large runs are smooth.

**The flight recorder exists because of hangs.** PyTorch's NCCL flight recorder (the `TORCH_NCCL_TRACE_BUFFER_SIZE` machinery) was built by the PyTorch distributed team specifically to debug hangs at scale, where attaching to hundreds of processes by hand is infeasible. Its whole design — a per-rank ring buffer of recent collectives, dumped on timeout, with sequence numbers so you can find the divergence — encodes the lesson that the informative data is *which collective the ranks stopped agreeing on*, gathered from all of them at the moment of failure.

**Watchdog timeouts are the standard hang signature.** The default NCCL watchdog timeout in recent PyTorch is 10 minutes; when a collective exceeds it, the process group aborts and, ideally, dumps the flight recorder. Teams running large jobs routinely tune this — shorter to fail fast in development, longer to survive legitimately slow steps like large checkpoint saves — via the `timeout` argument to `init_process_group`. The 10-minute gap between the last log line and the watchdog error is *the* diagnostic fingerprint of a collective hang, and recognizing it saves you from suspecting a crash.

**NCCL transport fallback is a common silent slowdown.** On multi-node runs, a misconfigured `NCCL_SOCKET_IFNAME` or a down InfiniBand link causes NCCL to fall back from RDMA (InfiniBand HDR ≈ 200 Gb/s) to TCP over Ethernet, which can be an order of magnitude slower. This shows up as the *slow* class, is invisible in the loss curve, and is announced plainly in the `NCCL_DEBUG=INFO` log — which is why "read the NCCL log" is the first move when multi-node underperforms. (These are approximate line-rate figures; achieved bandwidth is lower and depends on message size and topology.)

## When to reach for this — and when not

Debugging is a cost, and not every wobble deserves the full runbook. Some judgment on when to escalate.

**Don't page yourself for a slow step that is a checkpoint or an eval.** Before declaring a hang, confirm the "stopped" log is not just a legitimately long operation. A 30-second gap during a known checkpoint save is not a hang. Set the watchdog timeout comfortably above your longest legitimate step so it does not fire on a checkpoint.

**Don't debug at 64 GPUs what reproduces at 2.** The single highest-leverage decision in distributed debugging is scale. If the bug reproduces small, every iteration is a hundred times cheaper. Spend five minutes trying to reproduce at 2 GPUs before you spend an hour iterating at 64. Only when the bug is genuinely scale-dependent (a NIC, a topology, a race that needs many ranks) do you pay for debugging at scale.

**Don't reach for the profiler on a hang.** The profiler is the tool for the *slow* class, not the hung one — a hung job produces no steady state to profile, and the profiler itself may hang trying to synchronize. For a hang, the tools are stack traces and the flight recorder. Matching the tool to the class is the whole point of the decision table.

**Don't skip the single-GPU reference on a new parallelism config.** The one time it is *not* optional is when you have changed how parallelism works — new sharding, a custom collective, hand-rolled tensor parallelism. That is exactly when *silently wrong* is most likely and least visible. Pay the one single-GPU run; it is the cheapest insurance in the entire pipeline.

**Do invest in logging and liveness probes on any run you will run more than once.** The rank-aware logging, the outlier-not-average metric, the `monitored_barrier` liveness probe — these cost an afternoon to set up and pay for themselves the first time a run hangs at 3 a.m. and tells you *which rank* before you have finished pouring coffee. A run you launch once can be debugged reactively; a run you launch weekly deserves the instrumentation up front.

## Key takeaways

- **Classify before you dig.** Every distributed failure is a hang, a crash, a silently-wrong, or a slow. Naming the class in the first ten minutes rules out three families of cause and picks your tool. Guessing wrong costs hours.
- **A crash on one rank looks like a hang on all the others.** Count live processes early: all alive is a true hang; one missing is a crash whose traceback is in a log you have to go find, not rank 0's.
- **The informative rank is almost never rank 0.** With probability $(W-1)/W$, the failing rank is one you do not log. Gather stack traces from *every* rank and find the outlier — the one process that is not where the others are.
- **A collective is a barrier; one stuck rank freezes all of them.** The symptom (whole job idle) is global and points nowhere; the cause is local and specific. The distance between them is the debugging problem, and all-rank collection is how you close it.
- **`py-spy dump` for the live hang, `faulthandler` for the recurring one, the flight recorder when all stacks look identical.** Match the tool to what you have. When 64 stacks are the same, the flight recorder's sequence numbers show which collective they stopped agreeing on.
- **Assert your invariants: params identical across ranks, inputs different across ranks.** These are one `all_reduce` away from checkable and catch silently-wrong bugs at the step they happen instead of a week later at eval.
- **The single-GPU reference is the only reliable catch for silently-wrong.** Freeze a batch and a seed, compare distributed logits against a one-GPU run to tight tolerance. Non-optional for any new parallelism configuration.
- **Measure per-rank max, synchronized, in steady state.** The job runs at the pace of its slowest rank; the mean hides the straggler, unsynchronized timing measures kernel launches, and warm-up steps lie.
- **Reproduce smaller first.** A bug that reproduces at 2 GPUs is a hundred times cheaper to fix than one you chase at 64 — and whether it reproduces small tells you its class.

Fold all of this into the [distributed training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook), which threads the debugging checklist together with the decision framework for the whole series. The next time 64 GPUs go quiet, you will not be staring at rank 0 hoping. You will run the ten-minute runbook, diff the stacks, and find the one process that is not where the others are.

## Further reading

- [Why distributed training](/blog/machine-learning/distributed-training/why-distributed-training) — the four walls and the series map; the vocabulary this post assumes.
- [The NCCL timeout that hung the job](/blog/machine-learning/distributed-training/the-nccl-timeout-that-hung-the-job) — a full collective-mismatch war story and how to read the flight recorder.
- [The straggler](/blog/machine-learning/distributed-training/the-straggler) — the *slow* class in depth: finding and evicting the one node that halves throughput.
- [Silent NaN at scale](/blog/machine-learning/distributed-training/silent-nan-at-scale) — bisecting a NaN across ranks with forward hooks.
- [NCCL debugging deep-dive](/blog/machine-learning/distributed-training/nccl-debugging-deep-dive) — reading a real `NCCL_DEBUG=INFO` dump and the flight-recorder analyzer line by line.
- [Profiling a distributed run](/blog/machine-learning/distributed-training/profiling-a-distributed-run) — `torch.profiler` and Nsight Systems across ranks for the *slow* class.
- [Debugging DDP and multi-GPU](/blog/machine-learning/debugging-training/debugging-ddp-and-multi-gpu) — the single-process-to-DDP bug catalog behind "works on one GPU, not two."
- [The distributed training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) — the capstone decision-and-debugging checklist for the whole series.
- PyTorch distributed docs on the NCCL flight recorder (`TORCH_NCCL_TRACE_BUFFER_SIZE`) and `TORCH_DISTRIBUTED_DEBUG`; the `py-spy` and `faulthandler` documentation; Meta's public OPT-175B training logbook for what large-run debugging looks like in practice.

---
title: "The NCCL Timeout That Hung the Whole Job: Debugging a Collective Deadlock"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "A war story: a 64-GPU run freezes at step 4,000 with no crash and no error, then dies ten minutes later with a NCCL watchdog timeout. Here is the rendezvous mental model that explains why a collective hangs instead of crashing, the full catalog of causes, and the exact toolkit — flight recorder, py-spy across ranks, a short debug timeout — that finds the one rank that never showed up."
tags:
  [
    "distributed-training",
    "multi-gpu",
    "multi-node",
    "nccl",
    "ddp",
    "pytorch",
    "debugging",
    "deep-learning",
    "ml-systems",
    "gpu",
  ]
category: "machine-learning"
subcategory: "Distributed Training"
author: "Hiep Tran"
featured: true
readTime: 37
---

The page came in at 3:11 a.m.: "training job idle — no progress for 30 minutes." I pulled up the run. Sixty-four H100s across eight nodes, all reporting `99%` GPU utilization in `nvidia-smi`. Power draw near the cap. Fans howling. And the loss log — which had been printing a clean line every few seconds for six hours — stopped dead at step 4,000. No exception. No stack trace. No `CUDA error`. No `RuntimeError`. Nothing in stderr at all. Every process was alive, every GPU was busy, and the job was making exactly zero forward progress. It was, by every dashboard we had, *perfectly healthy* and *completely frozen* at the same time.

Then, at 3:21 a.m. — precisely ten minutes after the last logged step — the job finally spoke. One rank printed a wall of text that began with `Watchdog caught collective operation timeout`, the whole thing tore down, and `torchrun` reported a non-zero exit. The GPUs went quiet. Six hours of a 64-GPU run, gone, and the only clue was a timeout that fired ten minutes *after* the actual failure had already happened.

If you have trained anything on more than one GPU for more than a few days, you have met this exact failure, and it is the single most common way a distributed training job dies. Not a crash — a *hang*. The reason it hangs instead of crashing, and the reason the error message points at the wrong rank, both come from one fact about how GPUs talk to each other: a collective operation like an all-reduce is a **rendezvous**. Every rank has to show up, call the same collective, with the same shape, in the same order, or the ranks that did show up wait forever for the ones that did not. Figure 1 is that picture — seven ranks standing at the all-reduce, one rank that wandered off, and the deadlock that follows.

![diagram showing seven GPU ranks entering an all-reduce and blocking while one rank takes a divergent code path and never posts the collective, leading to a watchdog timeout](/imgs/blogs/the-nccl-timeout-that-hung-the-job-1.webp)

By the end of this post you will be able to: explain precisely why a collective hangs instead of crashing, and why the process that *prints* the timeout is a victim, not the culprit; recognize all six common ways a group deadlocks from their tell-tale signs; reproduce the hang in ten lines and fix it; wire up the debugging toolkit that turns a ten-minute blind wait into a sixty-second, single-rank diagnosis — `NCCL_DEBUG=INFO`, the NCCL flight recorder, `py-spy` across every rank, `gdb` for the C++ stack, and a short debug timeout; and lock the failure out for good with a prevention checklist. This is the "what breaks and how you debug it" wall of the [Distributed Training in the Trenches](/blog/machine-learning/distributed-training/why-distributed-training) series — the failure mode that costs more wall-clock than any other — and it builds directly on the collectives we assembled in [collectives from scratch](/blog/machine-learning/distributed-training/collectives-from-scratch) and the DDP mechanics in [DDP internals and gotchas](/blog/machine-learning/distributed-training/ddp-internals-and-gotchas).

## Why a collective hangs instead of crashing

Single-GPU training cannot hang like this. There is one process, one stream of instructions; if something goes wrong it raises, and you get a traceback. The moment you launch with `torchrun --nproc_per_node 8` you have eight processes that have to agree with each other several times per step, and agreement is enforced by *blocking*. That blocking is the whole story.

Recall what a collective actually is. When DDP finishes a bucket of gradients and calls `dist.all_reduce`, that call does two things. On the CPU side it enqueues a NCCL kernel onto the GPU's CUDA stream and returns almost immediately — the Python `all_reduce` call is *asynchronous*, it does not wait for the reduction to finish. On the GPU side, the NCCL kernel starts running, and its first job is to **synchronize with the same kernel on every other rank in the process group**. A ring all-reduce, which we derived byte-for-byte in [collectives from scratch](/blog/machine-learning/distributed-training/collectives-from-scratch), passes each rank's shard around a ring of `N` GPUs; rank `i` cannot send its chunk to rank `i+1` until rank `i+1`'s kernel is running and waiting to receive. The kernel spins on the GPU, polling a flag in the peer's memory, waiting for the handshake.

Now suppose one rank never launches that kernel — it took a different branch, it ran out of data, it is still stuck in the data loader. The other seven ranks' kernels are already running on their GPUs, spinning, polling a flag that will never be set. That spin is a busy-wait: the streaming multiprocessors are executing the wait loop at full tilt, which is exactly why `nvidia-smi` reports `99%` utilization on a job that is doing no useful work at all. The GPUs are not computing your model. They are computing "has my neighbor arrived yet? no. has my neighbor arrived yet? no." forever.

This is why you get a hang and not a crash. Nothing has *errored*. Every rank that reached the collective is doing exactly what it was told: wait for the group. There is no exception to raise because from each waiting rank's point of view, the operation is simply taking a long time. The one rank that diverged is not in a collective at all — it is off doing something else, or blocked in Python, so *it* has no error to raise either. Eight processes, all internally consistent, collectively deadlocked. The mechanism is worth stating as a mental model you can carry: **a collective is a barrier that only lifts when all `N` ranks post a matching operation; if even one rank posts a different operation, or no operation, the barrier never lifts.**

There is a second-order consequence that trips up everyone the first time. Collectives are matched **positionally**, by the order in which each rank issues them, not by name. NCCL assigns each collective on a process group a monotonically increasing sequence number. The all-reduce that all ranks post right after backward on step 4,001 is sequence number 4,001 on every rank — *as long as every rank has posted exactly the same collectives in the same order up to that point*. If rank 3 skipped one all-reduce a thousand steps ago and nobody noticed (because the shapes happened to line up and the numbers were close enough), then from step 1,001 onward rank 3's sequence numbers are off by one from everyone else's, and every subsequent collective is silently reducing *mismatched* tensors. That is the truly evil version of this bug, and we will come back to it. For now, hold the simple picture: a collective needs all `N` ranks, and one no-show hangs the rest.

## Reading the watchdog: what the timeout is actually telling you

For a long time, the default behavior of a NCCL deadlock was to hang *forever* — the job would sit there burning GPUs until a human noticed or the scheduler's wall-clock limit killed it. Modern PyTorch protects you from the infinite version with a **watchdog thread**. Every process group backed by NCCL spawns a background thread that periodically checks each in-flight collective against a timeout. When a collective has been outstanding longer than the timeout, the watchdog gives up and takes action.

Two environment variables control this, and both have been renamed recently, which is a frequent source of confusion:

```bash
# Modern names (PyTorch >= 2.2). The older NCCL_-prefixed names still work.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1   # watchdog aborts the PG on timeout
export TORCH_NCCL_BLOCKING_WAIT=0          # do not busy-block the CPU thread
```

With `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` (the default in recent releases), the watchdog does not just print — it **aborts the communicator and tears the process down** when a collective times out. This is the behavior you want. It converts an eternal hang into a crash-after-a-delay, so at least the job dies and your scheduler can restart it or your alerting can fire, instead of silently wasting a node for eight hours. If you have ever inherited a codebase that sets this to `0`, delete that line; a job that hangs forever is strictly worse than one that dies loudly.

The timeout itself is set when you create the process group. The default for NCCL is **10 minutes** in recent PyTorch (it was 30 minutes in older releases). That is the ten-minute gap between our last logged step and the death of the job — the watchdog was doing its job, it just does it on a ten-minute fuse by design, because legitimate collectives (the first all-gather of a huge sharded parameter, a barrier around a slow checkpoint save) can genuinely take minutes and you do not want the watchdog to kill those.

Here is the actual message, lightly trimmed, from that 3:21 a.m. teardown:

```log
[rank1]:[E ProcessGroupNCCL.cpp:563] [Rank 1] Watchdog caught collective operation timeout:
WorkNCCL(SeqNum=4001, OpType=ALLREDUCE, NumelIn=26214400, NumelOut=26214400, Timeout(ms)=600000)
ran for 600089 milliseconds before timing out.
[rank6]:[E ProcessGroupNCCL.cpp:563] [Rank 6] Watchdog caught collective operation timeout:
WorkNCCL(SeqNum=4001, OpType=ALLREDUCE, NumelIn=26214400, NumelOut=26214400, Timeout(ms)=600000)
ran for 600091 milliseconds before timing out.
[rank0]:[E ProcessGroupNCCL.cpp:563] [Rank 0] Watchdog caught collective operation timeout:
WorkNCCL(SeqNum=4001, OpType=ALLREDUCE, ...) ran for 600088 milliseconds before timing out.
```

Read this carefully, because it contains the single most important clue and the single most common misread. It tells you: the stuck operation is `ALLREDUCE`, sequence number `4001`, reducing `26,214,400` elements (a `100 MB` bf16 bucket), and it ran for `600089 ms` — ten minutes — before the watchdog killed it. Useful. But now count the ranks that printed it: `1`, `6`, `0`, and if you scroll, `2`, `4`, `5`, `7`. **Seven ranks printed the timeout. Rank 3 did not.** The message names the ranks that *entered* the collective and *waited* — the victims. The rank that never printed a timeout is the one that never entered the collective at all, so its watchdog had nothing to time out. **The rank missing from the timeout log is your prime suspect.** That inversion — the error names everyone except the culprit — is why this bug feels so disorienting, and why staring at the loudest rank's traceback wastes hours. Figure 2 lays out the full sequence in wall-clock time so you can see how far apart the failure and the symptom really are.

![timeline showing a job logging normally at step 4000 then rank 3 hitting end of data at step 4001 while seven ranks block in the all-reduce and burn full utilization until the watchdog fires after 600 seconds and aborts](/imgs/blogs/the-nccl-timeout-that-hung-the-job-2.webp)

### The mechanism: how expensive is a hang, and how likely is one

Two quick pieces of arithmetic make the stakes and the odds concrete, and both are worth internalizing.

**The cost of a single hang.** When one rank no-shows, every other rank spins until the watchdog fires. With `N` GPUs and a timeout `T`, the wasted GPU-time of one incident is at minimum

$$W = N \cdot T.$$

For our run, $N = 64$ and $T = 600\text{ s} = \tfrac{1}{6}\,\text{h}$, so one hang burns $64 \cdot \tfrac{1}{6} \approx 10.7$ GPU-hours *just waiting for the watchdog* — before you count the six hours of progress you lost, and before the job auto-restarts and re-hangs at the same step, and the next, and the next. At a rented rate around `\$3` per H100-hour, that is `\$32` of pure timeout per incident, and a restart loop overnight quietly turns into hundreds of dollars of nothing. The single most valuable thing you can do while debugging is *shorten `T`* so each failed attempt costs seconds, not minutes — more on that below.

**The probability a long run hits one.** Suppose your code has a small, data-dependent chance `p` of diverging across ranks on any given step — a branch that fires on some ranks and not others, a shard that occasionally comes up one batch short. The run survives a step with probability $1-p$, so the probability it survives all `S` steps and the probability it hangs at least once are

$$P_\text{survive}(S) = (1-p)^S, \qquad P_\text{hang}(S) = 1 - (1-p)^S.$$

This is the geometric-distribution tail, and it is merciless for long runs. Take a per-step divergence probability of $p = 10^{-4}$ — one step in ten thousand, which *feels* negligible. Over a $S = 50{,}000$-step run, $P_\text{hang} = 1 - (1 - 10^{-4})^{50000} \approx 1 - e^{-5} \approx 0.993$. A one-in-ten-thousand bug is a **99.3% chance of a hang** before your run finishes. This is why "it worked in the smoke test" tells you almost nothing; the smoke test ran 200 steps and had a 2% chance of tripping the bug. Rare per-step divergence is *certain* divergence at scale, and that is the mathematical reason collective hangs dominate the failure logs of every large run.

## The catalog of causes

Every collective deadlock is the same abstract event — ranks disagree about which collective to post next — but it arrives through a handful of concrete doors. Figure 3 is the field guide: six common causes, the tell-tale sign that distinguishes each, and the fix. I keep a version of this taped next to the monitors.

![matrix listing six causes of a collective deadlock with how each makes ranks diverge, the tell-tale sign that identifies it, and the specific fix for each one](/imgs/blogs/the-nccl-timeout-that-hung-the-job-3.webp)

Walk through them, because the *tell-tale sign* column is what lets you skip straight to the right fix instead of guessing.

**1. Data-dependent control flow.** Somewhere you have a collective — an `all_reduce`, a `barrier`, a metric `all_gather` — inside an `if` whose condition depends on the data, and the data differs across ranks. Classic version: "only log the gradient norm when it exceeds a threshold," where computing the norm involves an `all_reduce`. On steps where rank 3's local gradient is large and everyone else's is small, only rank 3 posts the reduction. *Tell-tale:* the hang is intermittent and data-dependent — it survives restarts to different steps. *Fix:* never put a collective under a rank-varying condition; compute the condition, `all_reduce` it to a group-wide decision, then branch on the agreed value.

**2. Uneven data sharding.** Each rank reads its own shard of the dataset. If the shards are not exactly equal in the number of *batches*, the ranks with fewer batches exhaust their iterator first, exit the training loop, and stop posting collectives, while the ranks with more batches keep going and block. *Tell-tale:* the hang is at a *fixed, late* step — the same step every restart — near the end of an epoch. *Fix:* `drop_last=True` on the sampler or loader so every rank runs the same integer number of batches, and shard so counts are even. This is the worked example below.

**3. A swallowed exception on one rank.** One rank hits a real error inside the step — a CUDA OOM on a slightly larger batch, a `nan` assertion, a corrupt sample — and a well-meaning `try/except` catches it and `continue`s. That rank skips the rest of the step, including the gradient all-reduce; the others block. *Tell-tale:* one rank's log shows a caught warning right before the group goes silent. *Fix:* do not swallow exceptions in the training step; or if you must, `all_reduce` an error flag so *all* ranks agree to skip the step together.

**4. An unused parameter under DDP.** A parameter produces no gradient on some ranks — a task head used only for certain examples, a branch of the model not taken — so under DDP's default `find_unused_parameters=False`, the bucket holding that parameter never completes on those ranks, and its all-reduce never launches, while on the ranks that did use it the all-reduce does launch and then waits. *Tell-tale:* the hang is at *step 1*, immediately, and reproduces every time. *Fix:* `find_unused_parameters=True` (it costs a graph traversal each step) or restructure so the parameter always participates. This is the mechanism we dissected in [DDP internals and gotchas](/blog/machine-learning/distributed-training/ddp-internals-and-gotchas); the second worked example below puts numbers on it.

**5. A collective inside `if rank == 0`.** Someone wraps evaluation, checkpointing, or logging in `if dist.get_rank() == 0:` — correctly, for the file I/O — but leaves a collective inside the block. Now only rank 0 posts it. *Tell-tale:* the hang is right at an eval or save boundary, at a round step number. *Fix:* keep only the *non-collective* I/O under the rank guard; call any collective (a `barrier`, a metric reduction, a broadcast of the loaded state) on *every* rank.

**6. A straggler that only looks like a hang.** One node is genuinely slow — a thermal-throttled GPU, a NIC that fell back to TCP, a noisy neighbor on a shared host — so the group *is* making progress, just crawling, and every fast rank spends most of each step blocked in the all-reduce waiting for the slow one. If it is slow enough, it looks frozen. *Tell-tale:* it *does* progress, just at a fraction of the expected tokens/s, and the same rank is always last. *Fix:* find and evict the slow node. This one is a different animal, covered in depth in the sibling war story [the straggler](/blog/machine-learning/distributed-training/the-straggler); the triage tree in the next section keeps you from confusing the two.

## Triage: is it a hang, a crash, or a straggler?

Before any deep tooling, spend thirty seconds separating the three failures that all present as "the job stopped moving," because they need completely different first moves. The discriminator is cheap and lives in `nvidia-smi` and your throughput metric. Figure 4 is the decision I run first, every time.

![decision tree that starts from a job not progressing and branches on GPU utilization to distinguish a collective deadlock at full utilization from a crashed rank at zero utilization from a straggler with one slow rank](/imgs/blogs/the-nccl-timeout-that-hung-the-job-4.webp)

- **Utilization pinned at `~100%`, zero progress.** This is the collective deadlock — ranks busy-waiting inside a NCCL kernel. It is what this whole post is about. Go to the toolkit below.
- **Utilization at `~0%`, zero progress.** A rank crashed, or a host is stuck outside CUDA — a process died and the others are blocked in a collective waiting for a rank that no longer exists, or everyone is blocked on the CPU side (a distributed data loader deadlock, a filesystem hang on a checkpoint read). Check whether all processes are still alive first (`pgrep -f train.py` on each node); a missing PID is your answer.
- **Utilization uneven, *some* progress.** Not a deadlock at all — a straggler. Your throughput metric is nonzero but low, and one rank is consistently the last to arrive. That is a performance problem, not a correctness one; the fix is placement and eviction, not the collective-matching logic here.

The `100%`-versus-`0%` split is the highest-value five seconds in distributed debugging. It immediately tells you whether you are looking for a rank that is *stuck inside* a collective (hot) or a rank that *died and left* the collective (cold), and those lead to opposite investigations.

## The diagnosis: finding the one rank that isn't there

Back to our 3:21 a.m. incident. The triage tree said `100%` utilization, so: collective deadlock, and the watchdog log named ranks 0, 1, 2, 4, 5, 6, 7 — everyone but rank 3. That is already a strong hypothesis (rank 3 is the no-show), but "the rank that didn't print" is a heuristic, not proof, and on a bad night several ranks fail to print for unrelated reasons. To *prove* which rank diverged, you go look at what every rank's Python interpreter is actually doing, and the tool for that is `py-spy`, which samples the call stack of a running process without stopping it or needing it to cooperate.

The move is to dump the stack of every training process on every node and diff them. On a single node:

```bash
# Dump the Python stack of every rank on this node, no restart needed.
for pid in $(pgrep -f train.py); do
  echo "===== PID $pid ====="
  py-spy dump --pid "$pid" --native   # --native descends into C/CUDA frames
done
```

`--native` is essential here: without it you see the Python frames and the deepest one is just `all_reduce`, which every rank shares. With `--native` you descend into the C++ and CUDA frames and can tell a rank that is *inside the NCCL wait* from a rank that is *somewhere else entirely*. For multi-node, wrap it in an `ssh` fan-out (or `srun` / `pdsh`) so you collect all 64 stacks:

```bash
# Fan the dump across all nodes in the SLURM allocation.
srun --overlap --ntasks-per-node=1 bash -c '
  for pid in $(pgrep -f train.py); do
    echo "===== $(hostname) PID $pid ====="
    py-spy dump --pid "$pid" --native
  done
' > /tmp/allstacks.txt 2>&1
```

Now diff the stacks. Seven of them bottom out identically — the tell of a healthy-but-waiting rank inside a collective:

```console
===== node-0 PID 11841 =====      (ranks 0,1,2,4,5,6,7 all look like this)
Thread 0x7f... (active): "MainThread"
    all_reduce (torch/distributed/distributed_c10d.py:2050)
    _allreduce_grads (torch/nn/parallel/distributed.py:1102)
    backward (torch/autograd/__init__.py:200)
    train_step (train.py:214)
    <module> (train.py:301)
  native frames:
    c10d::ProcessGroupNCCL::WorkNCCL::synchronizeInternal
    ncclAllReduce ... cudaStreamSynchronize   <-- blocked in the NCCL wait
```

And one of them — rank 3 — is nowhere near a collective:

```console
===== node-0 PID 11844 =====      (rank 3, the culprit)
Thread 0x7f... (active): "MainThread"
    __next__ (torch/utils/data/dataloader.py:631)
    _next_data (torch/utils/data/dataloader.py:1345)
    train_step (train.py:207)          <-- still trying to fetch a batch
    <module> (train.py:301)
```

There it is, in black and white. While seven ranks are blocked in `all_reduce`, rank 3 is sitting in `dataloader.__next__`, still trying to pull a batch that will never come — its shard is exhausted and the iterator is hung (or about to raise `StopIteration` and leave the loop). That single stack diff is the entire diagnosis. Figure 5 is the before-and-after of exactly this: a healthy step where all eight ranks are in the same collective at the same sequence number, versus the hung step where seven are in the collective and one is off in the loader.

![before and after comparison contrasting a healthy step with all eight ranks inside the all-reduce at the same sequence number against a hung step with seven ranks in the all-reduce and one rank stuck in the data loader for over ten minutes](/imgs/blogs/the-nccl-timeout-that-hung-the-job-5.webp)

If `py-spy` is not installed on the nodes and you cannot install it (locked-down cluster), `gdb` gives you the C++ stack of the same processes and shows the NCCL wait directly:

```bash
# C++ stacks for a rank; shows the ProcessGroupNCCL wait or the odd rank out.
gdb -p "$pid" -batch -ex "thread apply all bt" 2>/dev/null | grep -A2 -i "nccl\|dataloader\|recv"
```

There is an even cleaner way to get the same answer without diffing stacks at all, and it is the tool you should reach for first at scale: the **NCCL flight recorder**.

## The debugging toolkit: an escalation ladder

The stack-diff above works but it does not scale gracefully — fanning `py-spy` across 512 ranks and eyeballing the odd one out is slow. There is an ordered toolkit here, from the cheapest signal to the most invasive, and you climb it only as far as you need to. Figure 6 is that ladder.

![stacked ladder of hang debugging tools from the cheapest watchdog log at the top through NCCL debug info and the flight recorder to per-rank py-spy dumps and gdb at the most invasive bottom](/imgs/blogs/the-nccl-timeout-that-hung-the-job-6.webp)

**Rung 1 — the watchdog log.** Free, already there. Which ranks printed the timeout, and which did not? The missing rank is your first suspect, as we saw. If the log is enough to name the culprit, stop here.

**Rung 2 — `NCCL_DEBUG=INFO`.** Set it and NCCL prints, per rank, the rings it built, the transports it chose (NVLink, IB, or — alarmingly — a socket fallback), and it logs each collective. Grepping for the last collective each rank *entered* often shows the culprit is one sequence number behind. It is verbose; pipe it per rank. We go deep on reading these logs in [the NCCL debugging deep dive](/blog/machine-learning/distributed-training/nccl-debugging-deep-dive).

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL   # rings + collective traces, not everything
export NCCL_DEBUG_FILE=/logs/nccl.%h.%p.log   # one file per host/pid, or it interleaves
```

**Rung 3 — the flight recorder.** This is the one to reach for at scale, and it is the best thing to land in PyTorch distributed in years. The flight recorder keeps a ring buffer, per rank, of the last `K` collectives — their sequence number, op type, input/output sizes, and crucially their *state*: `scheduled`, `started`, or `completed`. When the watchdog fires it can dump every rank's buffer to disk. You then load all 64 dumps and look for the rank whose highest sequence number is *lower* than everyone else's — that rank never even scheduled the collective the others are stuck on. Turn it on:

```bash
# Enable the flight recorder and dump it automatically on a timeout.
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000          # ring buffer of last 2000 collectives/rank
export TORCH_NCCL_DUMP_ON_TIMEOUT=1               # write the buffer when the watchdog fires
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/logs/nccl_trace   # dump path prefix, one per rank
```

And a small reader that finds the culprit for you instead of by eye:

```python
# find_culprit.py — load per-rank flight-recorder dumps, find the rank that
# never posted the collective the majority is stuck on.
import glob, pickle
from collections import Counter

last_started = {}   # rank -> highest seq_id that reached "started"
stuck_on     = {}   # rank -> (seq_id, op) of the collective it is waiting in

for path in glob.glob("/logs/nccl_trace*"):
    with open(path, "rb") as f:
        trace = pickle.load(f)          # {'rank': r, 'entries': [ ... ]}
    rank = trace["rank"]
    entries = trace["entries"]
    last_started[rank] = max((e["seq_id"] for e in entries
                              if e["state"] in ("started", "completed")), default=-1)
    # the collective this rank is blocked in = started but never completed
    pending = [e for e in entries if e["state"] == "started"]
    if pending:
        e = max(pending, key=lambda e: e["seq_id"])
        stuck_on[rank] = (e["seq_id"], e["profiling_name"])

# The group is stuck on the seq_id the majority reached. The culprit is the
# rank whose last_started is behind that number: it never posted the op.
target = Counter(s for s, _ in stuck_on.values()).most_common(1)[0][0]
print(f"group is blocked on collective seq_id={target}")
for rank in sorted(last_started):
    tag = "  <-- CULPRIT (never posted it)" if last_started[rank] < target else ""
    print(f"  rank {rank:>3}: last_started seq_id={last_started[rank]}{tag}")
```

Run it and the answer is unambiguous — the culprit rank's `last_started` is `4000` while everyone else reached `4001`:

```console
group is blocked on collective seq_id=4001
  rank   0: last_started seq_id=4001
  rank   1: last_started seq_id=4001
  rank   2: last_started seq_id=4001
  rank   3: last_started seq_id=4000  <-- CULPRIT (never posted it)
  rank   4: last_started seq_id=4001
  ...
  rank   7: last_started seq_id=4001
```

That is the whole diagnosis, produced mechanically, in the seconds it takes the dumps to write — no stack diffing, no guessing, scales to thousands of ranks. (Field names in the dump vary a little across PyTorch versions — inspect one dump with `pickle` and adjust `seq_id` / `profiling_name` / `state` to match; the *idea* — find the rank whose max sequence number is behind — is stable.)

**Rung 4 — `py-spy --dump` across all ranks.** The stack diff we did above. Reach for it when the flight recorder is not available (older PyTorch) or when the culprit is stuck somewhere the recorder cannot see (Python-side, in the data loader, in user code) and you need the exact line.

**Rung 5 — `gdb thread apply all bt`.** The C++ stacks, when you suspect the hang is below Python — inside NCCL itself, a driver issue, a genuine transport wedge. Slowest and most invasive; the bottom of the ladder for a reason.

There is one more piece of the toolkit that is not a rung but a *setting*, and it is the highest-leverage change on this whole list: **shorten the timeout while you debug.** The default ten minutes exists to protect legitimately slow collectives in production. While you are actively reproducing a hang, ten minutes per attempt is agony. Drop it to sixty seconds — or even ten — so each failed attempt fails *fast*:

```python
import torch.distributed as dist
from datetime import timedelta

# Short timeout for a debug session: fail in 60 s, not 10 min.
dist.init_process_group(
    backend="nccl",
    timeout=timedelta(seconds=60),      # default is 10 min; crank it down to iterate
)
# Remember to set it back (or remove it) before the real run — a legit
# first-step all-gather of a large sharded param can exceed 60 s.
```

This single line is what turns the `\$32`-per-attempt, ten-minute debug loop into a `\$3`, sixty-second one. Set it, reproduce, fix, and *put it back* before the production run — because the same short fuse that helps you iterate will murder a legitimate multi-minute collective in production.

## Reproducing and fixing the hang

You cannot fix what you cannot reproduce, and the good news is that this bug reproduces trivially. Here is a fifteen-line program that deadlocks a group by making one rank post an extra collective. Save it and run it with `torchrun --nproc_per_node 4 hang.py`:

```python
# hang.py — deterministically deadlocks the group on rank 1.
import os, torch, torch.distributed as dist
from datetime import timedelta

dist.init_process_group("nccl", timeout=timedelta(seconds=30))  # short fuse
rank = dist.get_rank()
torch.cuda.set_device(rank % torch.cuda.device_count())

x = torch.ones(1024, device="cuda") * rank
for step in range(10):
    dist.all_reduce(x)                      # collective #1: all ranks post it
    if rank == 1 and step == 3:             # data-dependent divergence
        # rank 1 posts a SECOND all-reduce nobody else does -> deadlock
        dist.all_reduce(x)                  # collective #2: only rank 1 posts it
    if rank == 0:
        print(f"step {step} done", flush=True)
dist.destroy_process_group()
```

Run it and you will see `step 0 done`, `step 1 done`, `step 2 done`, then silence, then thirty seconds later the watchdog kills it. From step 3 onward, rank 1 is one collective ahead of everyone else: its second `all_reduce` on step 3 matches rank 0's *step-4* `all_reduce`, so they reduce mismatched data for one step and then rank 1 runs out of collectives while the others wait for its (never-posted) step-9 reduction. This is the pure form of the bug — a collective under a rank-varying condition.

The fix for control-flow divergence is a discipline, not a flag: **any decision that gates a collective must be made identically on every rank.** If the decision depends on data, reduce the decision *first*, then branch on the agreed result:

```python
# fix.py — decide once, agree everywhere, THEN branch.
import torch, torch.distributed as dist

def should_do_expensive_thing(local_signal: torch.Tensor) -> bool:
    # local_signal differs per rank; agree on a single group-wide decision.
    flag = (local_signal.norm() > 100.0).to(torch.int32).cuda()
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)   # 1 on ALL ranks if ANY rank tripped
    return bool(flag.item())

for step in range(10):
    dist.all_reduce(grads)                        # the normal, always-posted collective
    if should_do_expensive_thing(grads):          # SAME answer on every rank now
        dist.all_reduce(extra_metric)             # so all ranks post it, or none do
```

The pattern generalizes to the whole catalog. **Uneven shards:** make every rank run the same integer number of batches. **Swallowed exceptions:** turn a local error into a group decision by all-reducing an error flag so all ranks skip the step together. **Rank-0 collectives:** move the collective out of the `if rank == 0` block. The principle underneath every fix is the same one from Figure 1: keep all `N` ranks on the same sequence of collectives, always.

## Worked examples

#### Worked example: the uneven-shard hang

This is the one that hit us at 3:11 a.m., and the arithmetic is worth doing because it explains the "fixed, late step" tell-tale exactly.

Setup: a 64-GPU run, but focus on one 8-GPU node to keep the numbers legible. We stream a corpus split into shard files, assigned round-robin: rank `r` reads shards `r, r+8, r+16, …`. The corpus has an odd number of shards, so it does not divide evenly. Concretely, ranks 0–3 each end up with one extra shard compared to ranks 4–7. Each shard holds `8,000` samples; the per-rank batch size is `16`, so each shard is `8000 / 16 = 500` batches. That means:

- Ranks 0–3: an extra shard → `4,500` batches each.
- Ranks 4–7: one fewer shard → `4,000` batches each.

All eight ranks log identically through step 4,000 — nothing is wrong yet. On the attempt to fetch batch 4,001, ranks 4–7 hit `StopIteration`, their `for batch in loader` loop ends, they leave the training loop, and they stop posting collectives. Ranks 0–3 fetch their 4,001st batch, run forward and backward, and DDP posts the gradient all-reduce — which now waits for ranks 4–7 that have already left the building. Deadlock at **step 4,000**, every single restart, because the shard split is deterministic. That is the signature: a *fixed, late* step, reproducible to the exact number. Compare it to the control-flow hang, which wanders to a different step each restart because it is data-dependent.

The fix is two lines and a mental model of "make every rank do the same count":

```python
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                             shuffle=True, drop_last=True)   # <-- equalize batch counts
loader = DataLoader(dataset, batch_size=16, sampler=sampler,
                    drop_last=True,                          # <-- and drop the ragged tail
                    num_workers=4, pin_memory=True)

for epoch in range(num_epochs):
    sampler.set_epoch(epoch)     # reshuffle identically across ranks each epoch
    for batch in loader:
        train_step(batch)        # now every rank runs the SAME number of steps
```

`DistributedSampler(drop_last=True)` truncates the dataset to a length divisible by `world_size` and hands each rank an equal count; `DataLoader(drop_last=True)` drops any final partial batch. With both, all eight ranks run exactly `4,000` steps and finish the epoch together — no rank is ever left waiting for a rank that has already quit. If you use a custom `IterableDataset` and shard by file (as many streaming pipelines do), `DistributedSampler` does not apply, and you must equalize batch counts yourself: pad the short shards, or clamp every rank to the global minimum number of batches, or all-reduce a "keep going?" flag each step and stop the whole group the moment *any* rank runs dry. The general debugging of this class lives in the [DDP internals](/blog/machine-learning/distributed-training/ddp-internals-and-gotchas) post and, from the bug-taxonomy side, in [debugging DDP and multi-GPU](/blog/machine-learning/debugging-training/debugging-ddp-and-multi-gpu).

#### Worked example: the find_unused_parameters hang

The other classic, and the one with the cleanest step-1 signature. Say your model has an auxiliary task head — a coherence classifier used only when a batch contains a certain example type:

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()                 # always runs
        self.lm_head = nn.Linear(1024, vocab)      # always runs
        self.coh_head = nn.Linear(1024, 1024)      # 1024*1024 + 1024 = 1,049,600 params

    def forward(self, x, has_coherence_labels):
        h = self.backbone(x)
        logits = self.lm_head(h)
        if has_coherence_labels:                   # <-- data-dependent: differs per rank
            aux = self.coh_head(h[:, 0])           # coh_head participates only sometimes
            return logits, aux
        return logits, None
```

Now wrap it in DDP with the default `find_unused_parameters=False`. On a step where rank 3's micro-batch happens to contain coherence examples but the other ranks' do not, `coh_head` produces gradients on rank 3 and *no* gradients on ranks 0–2 and 4–7. DDP packs `coh_head`'s `1,049,600` parameters into a bucket. On rank 3, those parameters' autograd hooks fire, the bucket's pending counter reaches zero, and DDP launches that bucket's all-reduce. On the other seven ranks, the hooks *never fire* — no gradient, no hook — so the bucket's counter never reaches zero, and DDP *never launches that bucket's all-reduce*. Rank 3 is now waiting in an all-reduce that the other seven will never post. Deadlock, at the first step where the data diverges — often step 1 — and it reproduces reliably. That is the tell: **immediate hang, right at the start.**

Two fixes, with a real trade-off between them:

```python
# Fix A: tell DDP to expect unused params. Correct, but costs a full autograd-graph
# traversal every step to find which params were used -> a few percent throughput.
model = DDP(model, device_ids=[rank], find_unused_parameters=True)

# Fix B (cheaper): make the head ALWAYS participate, so no param is ever unused.
def forward(self, x, has_coherence_labels):
    h = self.backbone(x)
    logits = self.lm_head(h)
    aux = self.coh_head(h[:, 0])                 # always compute it
    if not has_coherence_labels:
        aux = aux * 0.0                          # ...but zero its loss contribution
    return logits, aux
```

Fix A is a one-liner but pays a per-step tax forever; Fix B keeps the fast path (`find_unused_parameters=False`, no traversal) at the cost of a wasted matmul on the head. On a large model where the head is tiny relative to the backbone, Fix B is usually the right call. And a warning that bites people: if you set `find_unused_parameters=True` *and* `static_graph=True`, DDP assumes the set of used parameters is fixed after the first iteration — if which parameters are unused actually *varies across steps or across ranks*, `static_graph` re-introduces the deadlock it was supposed to prevent. When the graph genuinely changes step to step, use `find_unused_parameters=True` alone, not with `static_graph`.

## Measuring the cost: how a hang burns GPU-hours, and how tooling stops it

The reason this failure mode deserves a whole post is not that it is hard to fix once found — most fixes are two lines — but that it is *catastrophically expensive to find* without the right setup, and cheap with it. Here is the honest before/after from moving one team's 64-GPU H100 SXM runs (8 nodes × 8 H100, NVLink within a node, InfiniBand HDR between) from "default settings and log-staring" to "flight recorder plus a short debug timeout."

| Metric | Before (default) | After (flight recorder + short debug timeout) |
|---|---|---|
| Time to *detect* a hang | 10 min (watchdog) or hours (no watchdog abort) | 60 s (short debug timeout) |
| Time to *root-cause* which rank | Hours of log/stack staring | ~10 min (`find_culprit.py` on the dump) |
| GPU-hours wasted per incident | ~200 (overnight restart loop on 64 GPUs) | ~2 (fail-fast, fix, resume) |
| Cost per incident at `\$3`/GPU-h | ~`\$600` | ~`\$6` |
| Incidents that reach a human at 3 a.m. | Most of them | Almost none (fails fast, auto-restart succeeds after fix) |

The `~200` GPU-hours "before" number is not a hang sitting for 200 hours — it is the *restart loop*. With `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`, the job dies after ten minutes, the scheduler dutifully restarts it, it re-hangs at the *same deterministic step* ten minutes later, and this repeats all night until someone wakes up. Sixty-four GPUs × three wasted hours ≈ 192 GPU-hours of a job that could never have succeeded, because the bug is deterministic. The single change that collapses this is making each failed attempt cost sixty seconds instead of ten minutes, plus a dump that names the culprit mechanically so the fix lands on the first human look.

**How to measure this honestly, so you do not fool yourself.** A few confounds specific to hangs:

- **Do not time "to first hang" and call it MTBF.** A deterministic hang (uneven shard, unused param) reproduces at the *same* step every time; its "time between failures" is a property of the bug, not of reliability. Only *non-deterministic* divergence (data-dependent branches) has a meaningful failure-rate, and you estimate its per-step `p` by counting hangs over total steps across many runs, not from one run.
- **Separate detection latency from root-cause latency.** They have different fixes: detection latency is the *timeout* setting; root-cause latency is the *tooling*. Shortening the timeout does nothing for root-cause time, and the flight recorder does nothing for detection time. Measure and attack them separately.
- **Warm up before you trust the flight recorder's timing fields.** The first few steps include CUDA context creation, cuDNN autotuning, and NCCL ring setup; a collective that takes seconds on step 0 and microseconds on step 100 is normal and is not a straggler. Read steady-state, not cold-start.

Figure 7 is that before/after as a picture, because the gap is the entire argument for spending an afternoon wiring up the tooling before you need it.

![before and after comparison showing a hang costing ten minutes to detect and roughly two hundred GPU-hours per incident without tooling versus sixty seconds to detect and about two GPU-hours per incident with a flight recorder and a short debug timeout](/imgs/blogs/the-nccl-timeout-that-hung-the-job-7.webp)

## Case studies and real numbers

Collective hangs are not an exotic corner case; they are the dominant operational reality of large training runs, documented candidly in the field.

- **The OPT-175B training logbook (Meta AI, 2022).** The most honest public artifact about what large-scale training actually feels like. Zhang et al. released the day-by-day notes from training a 175B model on 992 A100s, and the log is a near-continuous stream of hardware failures, hangs, and restarts — dozens of manual restarts over the run, many traced to a node or NCCL issue that wedged the group. The lesson the OPT team drew, and stated plainly, is that at that scale *something is always failing*, and the run's success depends less on never hanging than on *detecting and recovering fast*. That is precisely the detection-latency-versus-root-cause-latency split above.
- **The BLOOM-176B run (BigScience, 2022).** Trained on 384 A100 80GB GPUs on the Jean Zay cluster, the BigScience engineering notes describe hardware and communication failures as routine and document leaning on `torchrun`'s elastic restart plus careful checkpointing to keep a multi-month run alive across them. Their operational posture — assume collectives will wedge, checkpoint often, restart automatically — is the industrialized version of this post's advice.
- **PyTorch's NCCL flight recorder (2024).** The flight recorder landed in PyTorch precisely because stack-diffing `py-spy` across thousands of ranks did not scale. The PyTorch distributed team's design writeup frames it exactly as we used it here: a per-rank ring buffer of recent collectives with their state, dumped on timeout, so you can find the rank whose sequence number is behind. It exists because this failure was common enough, and expensive enough, to justify building first-class tooling for it.
- **NCCL's own timeout and health-check guidance (NVIDIA).** NVIDIA's NCCL documentation and the surrounding tooling (`NCCL_DEBUG`, the async error handling, IB health checks) are built around the assumption that a stuck collective is a *when*, not an *if*, on a large fabric — a slow or flapping InfiniBand link silently degrades an all-reduce until it looks like a hang. The interconnect physics behind why a single bad link stalls the whole ring is in [the interconnect physics](/blog/machine-learning/distributed-training/the-interconnect-physics).

The through-line of all four: nobody at scale prevents every hang. They *instrument* for it, so the mean time to *diagnosis* is minutes, and the run survives.

## When to reach for a short timeout, and when not

Every knob here is a trade-off, and the timeout is the sharpest one — the same setting that saves you while debugging will sabotage you in production if you leave it wrong.

| Situation | Timeout to use | Why |
|---|---|---|
| Actively reproducing a hang | 30–60 s | Fail fast; each attempt costs seconds, not ten minutes |
| Steady-state production run | 10–30 min (default range) | Absorb legitimately slow collectives without false kills |
| First step loads a huge sharded checkpoint | Longer, or a separate group | The initial all-gather / broadcast can genuinely take minutes |
| A `barrier` around a slow distributed save | Longer, on that call | Checkpoint I/O can exceed a tight timeout and false-trip |

The mistakes are symmetric. **Too short in production** turns a legitimately slow collective — the first-step all-gather of a 70B sharded model, a barrier wrapping a multi-gigabyte checkpoint write to a slow filesystem — into a spurious watchdog kill, and now your tooling is *causing* the failures. **Too long while debugging** wastes ten minutes per iteration and makes you hate your life. Match the timeout to the phase: short and aggressive while you hunt, generous and forgiving once you ship.

And know when the answer is *not* the timeout at all. If `nvidia-smi` shows uneven utilization and the job *is* progressing, you do not have a deadlock and no timeout setting will help — you have a straggler, and the fix is placement and eviction, covered in [the straggler](/blog/machine-learning/distributed-training/the-straggler). If utilization is at `0%` and processes have died, you have a crash, not a hang, and you want the traceback from the rank that *died*, not the timeout from the ranks that *waited*. The triage tree in Figure 4 is there so you spend your ten-minute fuse on the right problem. The broader "where do I even look first" discipline is the subject of [debugging distributed jobs](/blog/machine-learning/distributed-training/debugging-distributed-jobs), and the deeper NCCL-log reading is in [the NCCL debugging deep dive](/blog/machine-learning/distributed-training/nccl-debugging-deep-dive).

## The prevention checklist

You cannot eliminate hardware-induced hangs, but you can eliminate every *logic-induced* one, and they are the majority. Wire these into the template every job starts from:

- **Same code path on every rank.** No collective under a rank-varying or data-varying `if`. If a decision gates a collective, all-reduce the decision first and branch on the agreed value.
- **`drop_last=True` and even shards.** Every rank runs the identical integer number of steps per epoch. For custom `IterableDataset` sharding, clamp to the global-minimum batch count or all-reduce a "keep going" flag.
- **Propagate exceptions, do not swallow them.** A `try/except` in the training step that `continue`s past a collective is a latent deadlock. Turn local errors into a group decision to skip the step together.
- **No collective inside `if rank == 0`.** Only file I/O goes under the rank guard. Barriers, metric reductions, and state broadcasts run on all ranks.
- **Turn on `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`.** A hang that becomes a crash after the timeout is strictly better than one that runs forever.
- **Enable the flight recorder in production.** `TORCH_NCCL_TRACE_BUFFER_SIZE=2000` and `TORCH_NCCL_DUMP_ON_TIMEOUT=1` cost almost nothing and turn a future 3 a.m. mystery into a mechanical `find_culprit.py` run.
- **Use a short debug timeout while iterating,** and set it back before the real run.
- **Assert your ranks agree.** Periodically all-reduce a hash of the parameters (or the step count) so a silent sequence-number drift surfaces as a mismatch instead of a hang a thousand steps later.

## Key takeaways

- A collective is a rendezvous: it completes only when all `N` ranks post a matching operation. One no-show blocks the rest, which is why the failure is a **hang, not a crash** — nothing errored, everyone is just waiting.
- The busy-wait inside the NCCL kernel is why a deadlocked job reports **`~100%` GPU utilization** while making zero progress. Utilization is not progress.
- The watchdog timeout names the **victims** — the ranks that entered the collective and waited. The **culprit is the rank that did *not* print**, because it never entered a collective to time out.
- Triage on utilization first: `~100%` = deadlock, `~0%` = crash/dead rank, uneven-with-progress = straggler. Three different failures, three different first moves.
- The **flight recorder** (`TORCH_NCCL_TRACE_BUFFER_SIZE` + `DUMP_ON_TIMEOUT`) is the highest-leverage tool: it finds the rank whose sequence number is behind, mechanically, at any scale. `py-spy --native` across ranks is the fallback that pins the exact line.
- The catalog is small: data-dependent branches, uneven shards, swallowed exceptions, unused DDP parameters, rank-0 collectives, and stragglers. Each has a tell-tale sign and a two-line fix.
- Rare per-step divergence is *certain* divergence at scale: a $10^{-4}$ per-step chance is a 99% hang over 50,000 steps. "It passed the smoke test" proves almost nothing.
- **Shorten the timeout while debugging** (60 s) and lengthen it in production (10–30 min). The single setting that turns a `\$600` incident into a `\$6` one.
- You do not prevent every hang — you *instrument* so diagnosis takes minutes. That is what separates a run that finishes from one that dies at 3 a.m.

## Further reading

- [Why distributed training](/blog/machine-learning/distributed-training/why-distributed-training) — the four walls and the map of the whole series; this post is the "what breaks" wall.
- [Collectives from scratch](/blog/machine-learning/distributed-training/collectives-from-scratch) — the ring all-reduce and the `2(N-1)/N·S` byte volume that this rendezvous is built on.
- [DDP internals and gotchas](/blog/machine-learning/distributed-training/ddp-internals-and-gotchas) — buckets, autograd hooks, and the unused-parameter hang in full mechanical detail.
- [Your first multi-GPU run](/blog/machine-learning/distributed-training/your-first-multi-gpu-run) — ranks, world size, and the `torchrun` launch these processes come from.
- [The straggler](/blog/machine-learning/distributed-training/the-straggler) — the sibling war story for the "looks like a hang but is really slow" case.
- [Debugging distributed jobs](/blog/machine-learning/distributed-training/debugging-distributed-jobs) and [the NCCL debugging deep dive](/blog/machine-learning/distributed-training/nccl-debugging-deep-dive) — the first-ten-minutes discipline and how to read `NCCL_DEBUG=INFO` logs line by line.
- [Debugging DDP and multi-GPU](/blog/machine-learning/debugging-training/debugging-ddp-and-multi-gpu) — the broader bug taxonomy from the debugging-training pillar.
- [The distributed training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) — the capstone decision-and-debugging checklist that ties the whole series together.
- **PyTorch docs:** `torch.distributed` (process groups, timeouts, `init_process_group`), the DDP notes on `find_unused_parameters` and `static_graph`, and the NCCL flight recorder / c10d debugging guide.
- **NCCL docs (NVIDIA):** environment variables (`NCCL_DEBUG`, `NCCL_DEBUG_SUBSYS`, `NCCL_IB_*`, `NCCL_P2P_*`) and the collective-operation reference.
- **The OPT-175B logbook** (Zhang et al., 2022) and the **BigScience BLOOM engineering notes** (2022) — candid field accounts of hangs, restarts, and recovery at scale.

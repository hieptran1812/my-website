---
title: "Fault Tolerance and Elastic Training: Finishing a Run When Hardware Won't Cooperate"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "Why a multi-week run on thousands of GPUs is guaranteed to be interrupted, the MTBF and checkpoint-interval math that decides how much work you lose, and how elastic rendezvous lets a job shrink onto its survivors and keep going instead of restarting from scratch."
tags:
  [
    "distributed-training",
    "multi-gpu",
    "multi-node",
    "fault-tolerance",
    "elastic-training",
    "torchrun",
    "checkpointing",
    "pytorch",
    "nccl",
    "slurm",
    "ml-systems",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Distributed Training"
author: "Hiep Tran"
featured: true
readTime: 42
---

At two in the morning, roughly nine days into a two-week pretraining run on 512 H100 GPUs, the job died. Not gracefully. One rank threw a CUDA error — `Xid 79`, a GPU that had fallen off the PCIe bus — and because every other rank was waiting on that rank inside an all-reduce, the whole collective hung until the NCCL watchdog fired thirty minutes later and tore the process group down. The scheduler dutifully marked the job failed and moved on. When I woke up, 511 perfectly healthy GPUs had been sitting idle for six hours, the run was frozen at the last checkpoint from ninety minutes before the crash, and a queue of other people's jobs had eaten the slot. We had lost the six idle hours, the ninety minutes of training since the last save, and another forty minutes of queue wait to get the nodes back — all because a single fifteen-thousand-dollar card, one of five hundred and twelve, decided to leave the bus.

This is the failure mode that separates people who have run one job on eight GPUs from people who have shepherded a real run across thousands of GPUs for weeks. On one node, hardware failure is a rare accident you can mostly ignore. At scale it is not an accident and it is not rare — it is a **schedule**. Every extra GPU you add is another component that can die, and past a certain size the arithmetic guarantees that *something* will fail before your run finishes. The only question that matters is whether the run survives the failure or dies with it. A run that dies with every failure never finishes; a run that survives every failure finishes on time even though the hardware fought it the whole way.

![a healthy training run hitting a hardware fault and branching into a rigid restart from the last checkpoint versus an elastic reform on the surviving nodes that both finish the run](/imgs/blogs/fault-tolerance-and-elastic-training-1.webp)

By the end of this post you will be able to derive the job-level mean-time-between-failures from a single component's MTBF and see why it collapses as you scale; enumerate the failure modes that actually take a node down — a GPU off the bus, an uncorrectable ECC error, a kernel panic, a network partition, a straggler that became a corpse, a preemption — and recognize each from its signal; compute the mathematically optimal checkpoint interval given your MTBF and save cost, and stop guessing; and stand up an **elastic** training job with `torchrun` that catches a node failure, re-forms the process group on the survivors, reshards its state, and keeps going without a full restart from scratch. This is Track G of the [Distributed Training in the Trenches](/blog/machine-learning/distributed-training/why-distributed-training) series — the reliability track — and its thesis is blunt: at scale, fault tolerance is not a feature you add if you have time. It is the difference between a run that finishes and a run that does not.

## Failure is a schedule, not a risk

Let me start with the mechanism, because everything else in this post is a consequence of one piece of arithmetic that too few people actually work out. The claim is that on a large enough cluster, over a long enough run, the probability that some component fails during the run approaches one. This is not hand-waving. It falls directly out of how independent failures compose.

Model each GPU (or, more usefully, each node) as failing randomly over time at some constant average rate. If a single node has a mean-time-between-failures of $M_1$ — say it fails on average once every $M_1$ hours — then its failure rate is $\lambda = 1 / M_1$ failures per hour. Now put $N$ of these nodes in a cluster and run one synchronous job across all of them. The job is dead the instant *any* node dies, because a synchronous collective cannot complete without every participant. So the job's failure process is the **superposition** of $N$ independent failure processes. And the foundational fact about superposing independent random failure streams is that their rates add: the combined rate is

$$\lambda_\text{job} = N \lambda = \frac{N}{M_1}$$

which means the job's mean-time-between-failures is

$$M_N = \frac{1}{\lambda_\text{job}} = \frac{M_1}{N}$$

There it is. **The job MTBF is the single-node MTBF divided by the number of nodes.** Double the cluster, halve the time between failures. This is the central, non-negotiable fact of training at scale, and it is why a technique that is optional at eight GPUs is mandatory at eight thousand.

![a ladder of cluster sizes from eight to sixteen thousand GPUs showing the job mean-time-between-failures shrinking from months to hours as node count grows](/imgs/blogs/fault-tolerance-and-elastic-training-2.webp)

Put numbers on it. What is a realistic per-GPU MTBF? The most useful public anchor is Meta's Llama 3 training report: over a 54-day pretraining run on 16,384 H100 GPUs, they logged 466 job interruptions, 419 of them unexpected, and roughly 78% of those traced to hardware — failing GPUs, HBM, cables, switches, and the like. That is one interruption roughly every three hours across the whole cluster. Invert the arithmetic: if the job MTBF is about 3 hours on 16,384 GPUs, then the implied per-GPU MTBF is around $3 \times 16{,}384 \approx 50{,}000$ GPU-hours, which is a bit under six years per individual GPU. That is a perfectly healthy per-card reliability number — GPUs are not flaky in isolation. It is the *multiplication by sixteen thousand* that turns "once every six years per card" into "once every three hours for the run."

Now walk that per-GPU MTBF of roughly 50,000 GPU-hours down the scale ladder and watch the job MTBF fall:

| Cluster | GPUs | Job MTBF ≈ 50,000 / N | Failures in a 2-week (336 h) run |
| --- | --- | --- | --- |
| One node | 8 | ~6,250 h (~8.6 months) | ~0.05 (probably none) |
| Small pod | 64 | ~780 h (~33 days) | ~0.4 |
| Mid cluster | 512 | ~98 h (~4 days) | ~3.4 |
| Large cluster | 4,096 | ~12 h | ~28 |
| Frontier | 16,384 | ~3 h | ~110 |

On one node, you can run for two weeks and reasonably expect to never see a hardware failure — checkpointing is insurance you probably won't cash. On 512 GPUs, you should *plan* on three or four interruptions per run. On 16,384 GPUs you will be interrupted more than a hundred times, which means a fault every few hours for the entire run, which means your recovery path is not an exception handler you write and forget — it is the **hot path**, exercised over and over, and every minute it wastes gets multiplied by a hundred.

There is a second way to state the same fact that is sometimes more visceral. The probability of *at least one* failure during a run of length $T$ on $N$ nodes is

$$P(\text{fail}) = 1 - e^{-N \lambda T} = 1 - e^{-T / M_N}$$

For the 512-GPU, two-week case, $T / M_N = 336 / 98 \approx 3.4$, so $P(\text{fail}) = 1 - e^{-3.4} \approx 0.967$. There is a 97% chance you get interrupted at least once. For the 16,384-GPU case the exponent is around 110 and the probability rounds to 1.0000 — it is not "likely," it is a certainty with more nines than you can measure. **You are not deciding whether to handle failure. Failure already decided for you.**

### Why this changes the whole engineering posture

Once you internalize $M_N = M_1 / N$, a lot of decisions that felt like taste become forced. You cannot run a frontier-scale job and hope; the hope has a half-life of three hours. You cannot treat checkpointing as a nice-to-have; it is the only thing standing between a fault and starting over. And you cannot treat a full job restart — kill everything, requeue, reinitialize, reload — as cheap, because you are going to pay it dozens of times and the fixed costs (scheduler latency, process-group init, NCCL handshake, checkpoint read) do not shrink with better luck. The rest of this post is about driving those two costs — the *work you lose* per failure and the *time you waste* recovering — as close to zero as the hardware allows.

## The failure zoo: what actually takes a node down

"A node failed" is too coarse to act on. Different failures emit different signals, are visible to different parts of the stack, and demand different responses — some are recoverable in place after a reboot, some mean a card is dead and must be evicted, and some are not failures at all but *planned* interruptions you should have caught with a signal handler. If you cannot tell them apart, you cannot automate recovery, and un-automated recovery at a fault-every-three-hours cadence means a human awake at all times, which is not a plan.

![a table of failure modes from a GPU falling off the bus to preemption with the signal each one shows whether it is recoverable in place and the response it demands](/imgs/blogs/fault-tolerance-and-elastic-training-3.webp)

Here is the zoo, in the rough order you will meet them, each with the tell that identifies it:

- **A GPU falls off the bus.** The card stops responding to the driver mid-run. You see an `Xid 79` (GPU has fallen off the bus) or `Xid 48` (double-bit ECC leading to fallout) in `dmesg` and the kernel log, and CUDA calls on that device start returning errors. The card is gone for this run; the node needs the GPU reset or, often, the whole node drained and the card RMA'd. Not recoverable in place.
- **An uncorrectable ECC error.** HBM detected a memory error it could not correct — an `Xid 63` or `Xid 64`, a double-bit flip. Single-bit errors are corrected silently and only logged; double-bit errors corrupt data and the driver takes the affected memory (or the whole GPU) offline. The node should be retired and the row of memory retested. Not recoverable in place; suspect the card.
- **A node crash or reboot.** The whole box goes down — a kernel panic, a thermal shutdown, a power event, a BMC watchdog reset. The rank simply vanishes; from the other ranks' perspective, a peer stopped answering. After the node reboots and passes health checks it can rejoin, so this is recoverable — but not *quickly*, and not without reforming the group.
- **A network partition.** An InfiniBand link flaps, a switch reboots, a cable degrades, or the NIC drops. One or more ranks become unreachable even though their GPUs are fine. If the link comes back the ranks can be reintegrated; if it doesn't, they must be evicted like a dead node. The tell is NCCL connection errors and unreachable-peer messages, not a CUDA fault.
- **A straggler that became a corpse.** This is the nastiest, because it starts as the *slow-node* problem from [the straggler post](/blog/machine-learning/distributed-training/the-straggler) and ends as a failure. A rank stops making progress — deadlocked, wedged in a driver bug, spinning on hung I/O — but does not crash. Every other rank blocks in the collective waiting for it. Nothing looks broken until the NCCL watchdog times out (default 30 minutes; `TORCH_NCCL_TIMEOUT`) and aborts the communicator. You lose thirty minutes to a corpse before the system even admits something is wrong.
- **A preemption.** Not a failure at all: the scheduler *reclaims* your nodes — a spot-instance reclaim, a higher-priority job on a shared cluster, a maintenance drain. You get warning: a `SIGTERM`, or on SLURM a `SIGUSR1` a configurable number of seconds before the hard kill. This is the *only* failure you get advance notice of, and if you ignore the notice you throw away a free, clean checkpoint.
- **An out-of-memory error.** Usually deterministic and your own fault (a batch too large, a memory leak, an activation you forgot to checkpoint), but it can also be triggered by fragmentation that only bites after hours, or by a resharding event that lands more parameters on one rank. It kills one rank, which kills the collective. This one you fix in code, not with fault tolerance — but the job still has to survive the crash long enough for you to see it.

What these actually look like in the logs is worth seeing once, because pattern-matching the signal is how you route the response without a human reading every line. A GPU-off-the-bus fault and the NCCL abort it triggers land in the kernel log and the training stderr together:

```log
# dmesg / kernel log on the failing node
NVRM: Xid (PCI:0000:8a:00): 79, pid=..., GPU has fallen off the bus.
NVRM: GPU 0000:8a:00.0: GPU has fallen off the bus.

# training stderr, moments later, on every OTHER rank
[rank39]: NCCL WARN Cuda failure 'an illegal memory access was encountered'
[rank39]: work queue not empty; aborting NCCL communicator
[rank12] Watchdog caught collective operation timeout:
         WorkNCCL(OpType=ALLREDUCE, Timeout(ms)=600000) ran for 600011 ms
         before timing out.
torch.distributed.DistBackendError: NCCL communicator was aborted on rank 12.
```

The first two lines name the culprit precisely — device `8a:00`, `Xid 79`, off the bus — so an automated handler can extract the PCI address, mark that GPU's node for eviction, and trigger a reform on the rest. The `Watchdog caught collective operation timeout` line is the *survivors* reporting that they waited the full timeout on a rank that will never answer; if that timeout is still the 30-minute default, those 63 ranks just burned half an hour spinning before the system admitted the truth. This is the entire argument for tuning `TORCH_NCCL_TIMEOUT` down and leaning on async error handling: you want the abort to fire off the `Cuda failure`, not off the watchdog thirty minutes later.

The single most important line in that list is the distinction between the straggler-that-died and everything else. Most failures announce themselves loudly and fast — a CUDA error, a vanished rank, a connection reset. The hung rank announces itself only by *silence*, and silence in a synchronous job is indistinguishable from "still working" until a timeout expires. That is why the NCCL watchdog exists, why its timeout is a parameter you should tune down from 30 minutes if your steps are short, and why "detect the failure" is a genuine engineering problem and not a freebie. Half the value of a good fault-tolerance setup is *shortening the time between when a node dies and when the system admits it*.

## The baseline: checkpoint and restart

The oldest, simplest, and still most important fault-tolerance mechanism is **checkpoint and restart**: periodically write the full training state to durable storage, and on any failure, kill the job, requeue it, and reload the last checkpoint. Everything fancier — elastic reform, hot spares, in-memory redundancy — is an *optimization on top of* checkpointing, not a replacement for it. If your checkpoints are wrong, nothing else can save you; if they are right, you always have a floor to fall back to. This post assumes you have the checkpoint itself under control — the [distributed-checkpointing post](/blog/machine-learning/distributed-training/distributed-checkpointing) covers how to save a sharded state correctly and quickly, and [the loss-spike-after-resume post](/blog/machine-learning/distributed-training/the-loss-spike-after-resume) covers the ways a resume silently drops state and corrupts the run. Here we care about one thing: given that you *can* save and restore correctly, **how often should you, and what does a restart cost?**

![an ordered sequence from a checkpoint write through a node death to the scheduler requeue reload and the redo of the steps lost since the last save](/imgs/blogs/fault-tolerance-and-elastic-training-4.webp)

The cost of a single restart has two parts. The first is **wasted work**: everything computed since the last checkpoint is gone, so if you checkpoint every $\tau$ seconds of useful compute and a failure strikes at a uniformly random moment within an interval, you lose on average half an interval, $\tau / 2$, of recomputation. The second is **fixed recovery latency** $R$: the scheduler must notice the job died, requeue it, wait for nodes, spin up the processes, re-establish the NCCL communicators, and read the checkpoint back from storage. On a busy shared cluster $R$ can be dominated by queue wait and run to tens of minutes; on a dedicated cluster with a warm restart it can be a couple of minutes. Either way, the total wall-clock lost per failure is roughly $\tau/2 + R$.

That immediately tells you the tension. If you checkpoint *very* often, $\tau$ is tiny and you lose almost no work per failure — but you spend a large fraction of every hour writing checkpoints, and that write cost is pure overhead you pay whether or not you ever fail. If you checkpoint *rarely*, the write overhead vanishes but each failure redoes a huge span of steps. Somewhere between "every step" and "never" is an interval that minimizes total waste. It is not a matter of taste; it has a closed-form answer.

### Deriving the optimal checkpoint interval

Let $C$ be the wall-clock cost of writing one checkpoint (seconds), and $M$ the job MTBF (seconds) from the section above. We checkpoint every $\tau$ seconds of useful compute. Consider the overhead per unit of useful work, and add up the two sources of waste.

The **checkpoint overhead** is straightforward: in every interval of length $\tau$ you pay $C$ seconds of writing, so the overhead fraction is $C / \tau$.

The **rework overhead** comes from failures. Failures arrive at rate ${1/M}$. When one strikes, you lose on average half the current interval, $\tau/2$, of recomputed work. So the expected rework per unit time is (failures per unit time) times (work lost per failure), which is $\frac{1}{M} \cdot \frac{\tau}{2} = \frac{\tau}{2M}$.

Total overhead as a function of the interval:

$$W(\tau) = \frac{C}{\tau} + \frac{\tau}{2M}$$

The first term falls as $\tau$ grows; the second rises. Minimize by setting the derivative to zero:

$$\frac{dW}{d\tau} = -\frac{C}{\tau^2} + \frac{1}{2M} = 0 \quad\Longrightarrow\quad \tau^2 = 2CM \quad\Longrightarrow\quad \boxed{\tau_\text{opt} = \sqrt{2CM}}$$

This is **Young's formula** (refined by Daly), and it is one of the genuinely useful closed-form results in systems engineering. The optimal checkpoint interval grows as the geometric mean of your save cost and your time between failures. Cheaper checkpoints or more frequent failures pull the interval down; expensive checkpoints or rock-solid hardware push it up. Plug $\tau_\text{opt}$ back in and the *minimum* overhead is beautifully symmetric — the two terms become equal:

$$W(\tau_\text{opt}) = \frac{C}{\sqrt{2CM}} + \frac{\sqrt{2CM}}{2M} = \sqrt{\frac{C}{2M}} + \sqrt{\frac{C}{2M}} = \sqrt{\frac{2C}{M}}$$

The minimum achievable overhead is $\sqrt{2C/M}$. This is the number that tells you whether checkpointing is a rounding error or a real tax. If $C$ is small relative to $M$ — cheap saves, reliable hardware — the overhead is negligible. If $C$ creeps up toward $M$ — a slow, blocking, full-state save on a cluster that fails every few hours — the overhead becomes punishing, which is exactly why the [distributed-checkpointing post](/blog/machine-learning/distributed-training/distributed-checkpointing) obsesses over making $C$ small with sharded and asynchronous saves. **Cutting $C$ doesn't just save you the write time; it lets you save more often, which shrinks rework, which is a double win.**

![two checkpoint cadences contrasted where a too-rare interval redoes thousands of steps on failure while the tuned square-root interval balances save cost against rework](/imgs/blogs/fault-tolerance-and-elastic-training-5.webp)

One caveat the clean formula hides: it assumes the recovery latency $R$ is a fixed per-failure cost that does not depend on $\tau$, which is true, so $R$ shifts the total waste up by $R/M$ per unit time but does *not* move $\tau_\text{opt}$. It also assumes failures are memoryless (a constant rate), which is a decent approximation but breaks during "bad node" clusters where one flaky component fails repeatedly — in practice you evict such nodes rather than model them. And it assumes you *can* checkpoint at an arbitrary interval; if your save is blocking and takes 45 seconds, you obviously cannot checkpoint every 10 seconds. Async checkpointing relaxes that. But as a default, compute $\sqrt{2CM}$, round it to a convenient step count, and you will be within a few percent of optimal instead of guessing.

#### Worked example: three failures in a two-week run

Take the concrete run from the intro: 512 H100 GPUs across 64 nodes, a 7-billion-parameter transformer, bf16, data-parallel with FSDP, targeting a two-week (336-hour) run. From the MTBF ladder, the job MTBF is about $M = 98$ hours $= 352{,}800$ seconds, so we expect $336/98 \approx 3.4$ failures over the run — call it three or four interruptions.

Now size the checkpoint interval. Suppose a sharded checkpoint write costs $C = 45$ seconds of wall-clock (each rank writes its shard in parallel to a fast parallel filesystem; this is a *blocking* save for the moment). The step time is about 3 seconds. Then:

$$\tau_\text{opt} = \sqrt{2 \cdot 45 \cdot 352{,}800} = \sqrt{31{,}752{,}000} \approx 5{,}635 \text{ s} \approx 94 \text{ min} \approx 1{,}878 \text{ steps}$$

So checkpoint every ~1,900 steps. The minimum overhead is

$$W(\tau_\text{opt}) = \sqrt{\frac{2 \cdot 45}{352{,}800}} = \sqrt{0.000255} \approx 0.016 = 1.6\%$$

Split evenly: 0.8% of your time writing checkpoints, 0.8% redoing lost work. Over a 336-hour run that is about 5.4 hours of total overhead — a couple of hours of checkpoint writes plus a couple of hours of rework spread across three or four failures. Add the fixed recovery latency: if each *rigid restart* costs $R \approx 15$ minutes (a lucky warm requeue) to maybe 60+ minutes (queue wait on a busy cluster), then 3.4 failures cost another one to three-plus hours. Total: your two-week run stretches by roughly 2–4%, and none of it required you to be awake.

Contrast the guesses. Checkpoint every 100 steps (~5 minutes) "to be safe" and your write overhead alone is $45 / 300 = 15\%$ — you would burn an entire extra day of the run writing checkpoints to protect against a rework cost that was already under 1%. Checkpoint every 10,000 steps (~8.3 hours) "to save I/O" and each failure now redoes up to 10,000 steps, averaging ~4 hours of rework per failure, ~14 hours over the run — worse than the write-heavy extreme. The tuned interval beats both by a wide margin, and you got it from one square root. **This is the single highest-leverage number in your run config, and most people pick it by vibe.**

## Elastic training: reform on the survivors

Checkpoint-and-restart has a structural weakness the math above makes obvious: the fixed recovery latency $R$. Every failure pays the full cost of tearing the whole job down and standing it back up — scheduler requeue, wait for nodes, reinitialize all ranks, re-handshake NCCL, reread the checkpoint — *even though only one node out of sixty-four actually failed*. The other sixty-three nodes were healthy the entire time. You killed 511 working GPUs to deal with the loss of one. On a cluster that fails every three hours, paying a 15-to-60-minute full-restart tax more than a hundred times is the dominant cost of the whole run.

Elastic training attacks exactly that waste. Instead of the scheduler killing the job on failure, the *job itself* detects the failure, throws away the dead ranks, **re-forms its process group on the surviving nodes**, reshards its state onto the new (smaller) world, and continues — no scheduler round-trip, no queue wait, no cold start. You still reload from the last checkpoint to get consistent state, but you skip the entire external restart machinery. The recovery latency drops from tens of minutes to a couple of minutes, and the 511 healthy GPUs never stop being yours.

![a sixty-four node run losing one node to a fault then re-forming the process group on the surviving sixty-three nodes resharding state and resuming without a full restart](/imgs/blogs/fault-tolerance-and-elastic-training-6.webp)

In PyTorch this is **`torchelastic`**, now the default engine behind `torchrun`. The core idea is a **rendezvous**: a coordination service that all workers register with, that agrees on the current membership (the world size and each worker's rank), and that can *re-run the agreement* when membership changes. When a worker dies, the agent processes on the surviving nodes detect it (the local `torchrun` agent monitors its child processes and the rendezvous heartbeats), the rendezvous is re-triggered, a new membership is agreed among whoever is still alive, and every surviving worker is restarted with fresh `RANK` and `WORLD_SIZE` values reflecting the new, smaller group. Your training script re-runs from its top — which is why the script *must* be written to reload from checkpoint on startup rather than assuming it always begins at step zero.

### The rendezvous protocol

The rendezvous is the heart of elasticity, so it is worth being precise about what it guarantees. You configure it with a **backend** and an **endpoint**. The two common backends are `c10d` (a lightweight TCP store hosted by the rendezvous, no external dependency — the modern default) and `etcd` (a separate distributed key-value store, more robust for very large or long-lived deployments but one more thing to operate). You give it an `rdzv_id` (a unique name for this job, so restarts of the same job find each other and unrelated jobs don't collide) and an `rdzv_endpoint` (host:port of the node hosting the rendezvous — typically node 0).

The protocol runs in rounds. Workers arriving at a rendezvous wait until enough of them have gathered — at least `min_nodes` — within a timeout window, then the round *closes* and the gathered set becomes the active membership: each worker gets a rank, the world size is fixed, and training proceeds. If a worker dies mid-run, the next round opens: survivors re-enter the rendezvous, and as long as at least `min_nodes` are present, a new membership forms and training resumes at the new size. If the survivor count drops below `min_nodes`, the rendezvous cannot close and the job waits (or fails, per your restart policy) until enough nodes — perhaps replacements the scheduler brings up — arrive to satisfy the minimum. This is what `--nnodes=MIN:MAX` expresses: the job is allowed to run on anywhere from `MIN` to `MAX` nodes, and it dynamically re-forms within that band as nodes come and go.

The `--max_restarts` parameter caps how many times the agent will re-form and restart the workers before giving up — a safety valve so a genuinely broken job (say, a deterministic OOM that recurs on every restart) doesn't thrash forever. Set it high enough to absorb your expected failure count with headroom (if you expect 4 failures, `--max_restarts=8` or more), but finite, so a crash-loop eventually surfaces to a human.

### Making the training loop restart-safe

Elasticity only works if your training script can be *restarted from scratch and resume correctly*, because that is exactly what torchelastic does to every surviving worker on a reform: it re-executes your script top-to-bottom with new rank assignments. A script that assumes it always starts at step zero will happily throw away days of progress. A restart-safe loop does four things on startup: finds the latest checkpoint, loads full state (model, optimizer, scheduler, RNG, step counter, data position), reinitializes the process group at the *current* world size, and resumes the loop from the saved step. Here is the skeleton, with the elastic-specific parts called out:

```python
import os
import glob
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

def setup_distributed():
    # torchrun/torchelastic sets these; on a reform they reflect the NEW world.
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size(), local_rank

def find_latest_checkpoint(ckpt_dir):
    ckpts = glob.glob(os.path.join(ckpt_dir, "step_*"))
    if not ckpts:
        return None
    # step_0000123 -> 123; pick the highest completed step.
    return max(ckpts, key=lambda p: int(p.split("_")[-1]))

def load_full_state(path, model, optimizer, scheduler, map_location):
    state = torch.load(os.path.join(path, "train_state.pt"),
                       map_location=map_location)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    # Restore RNG so data order and dropout match — see the loss-spike post.
    torch.set_rng_state(state["cpu_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"])
    return state["step"], state["epoch"], state["samples_seen"]

def main(cfg):
    rank, world_size, local_rank = setup_distributed()
    model = build_model(cfg).cuda(local_rank)
    model = FSDP(model, device_id=local_rank)          # resharded to new world
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    start_step, epoch, samples_seen = 0, 0, 0
    latest = find_latest_checkpoint(cfg.ckpt_dir)
    if latest is not None:
        start_step, epoch, samples_seen = load_full_state(
            latest, model, optimizer, scheduler,
            map_location=f"cuda:{local_rank}")
        if rank == 0:
            print(f"[resume] loaded {latest}: resuming at step {start_step}")
    else:
        if rank == 0:
            print("[resume] no checkpoint found, starting fresh at step 0")

    # A resumable sampler that skips already-consumed samples this epoch.
    loader = build_resumable_loader(cfg, world_size, rank,
                                    epoch=epoch, skip=samples_seen)

    for step in range(start_step, cfg.max_steps):
        batch = next(loader)
        loss = train_step(model, optimizer, scheduler, batch)
        if step % cfg.ckpt_every == 0 and step > start_step:
            save_full_state(cfg.ckpt_dir, step, epoch, samples_seen,
                            model, optimizer, scheduler)
    dist.destroy_process_group()
```

Three things in that code are load-bearing and routinely botched. First, `init_process_group` reads the *current* `RANK`/`WORLD_SIZE` from the environment torchelastic just set, so after a reform from 64 to 63 nodes it forms a 63-node group automatically — you do not hardcode the world size anywhere. Second, wrapping the model in `FSDP` *after* loading means the parameters reshard onto the new world size on the spot; this is the [state resharding](/blog/machine-learning/distributed-training/distributed-checkpointing) that lets you resume on a different node count than you saved on, and it is the whole reason elastic-plus-sharded works. Third, the loop starts at `start_step`, not zero, and the data loader skips `samples_seen` so you do not re-feed the model data it already trained on — get this wrong and you get the plateau-and-repeat failure described in [the loss-spike post](/blog/machine-learning/distributed-training/the-loss-spike-after-resume). Note also `step > start_step` on the save condition: without it, a job that resumes at a checkpoint step immediately re-saves the same step, which is harmless but wasteful, and worse, can race with the checkpoint you just loaded.

### The startup health probe

Elastic reform is only as good as your ability to *detect* that a node is bad, and the cheapest place to catch a sick node is *before* it joins the collective and poisons the group. A startup health probe runs a tiny distributed self-test the moment the group forms: every rank does a small all-reduce and a small matmul, times them, and any rank that is dramatically slow — or that errors outright — is flagged before real training begins. This catches the card that comes up in a throttled state, the node with a degraded NVLink, and the rank whose NCCL can't reach its peers, turning a mid-run hang hours later into a fast, clean startup abort.

```python
import time
import torch
import torch.distributed as dist

def health_probe(local_rank, warn_factor=3.0, size=(8192, 8192)):
    """All-reduce + matmul self-test. Flags a rank far slower than the median."""
    dev = torch.device(f"cuda:{local_rank}")
    x = torch.randn(*size, device=dev, dtype=torch.bfloat16)

    # 1) Compute probe: a big matmul. A throttled/sick GPU is much slower.
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    for _ in range(10):
        x = (x @ x) * 0.5 + 1e-3
    torch.cuda.synchronize(dev)
    compute_ms = (time.perf_counter() - t0) * 1e3

    # 2) Comms probe: all-reduce a healthy tensor and time it.
    probe = torch.ones(1 << 24, device=dev, dtype=torch.bfloat16)  # 32 MB
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    dist.all_reduce(probe)
    torch.cuda.synchronize(dev)
    comms_ms = (time.perf_counter() - t0) * 1e3

    # 3) Gather every rank's timing and compare to the median.
    local = torch.tensor([compute_ms, comms_ms], device=dev)
    world = dist.get_world_size()
    gathered = [torch.zeros_like(local) for _ in range(world)]
    dist.all_gather(gathered, local)
    all_compute = torch.stack([g[0] for g in gathered])
    median = all_compute.median()

    if compute_ms > warn_factor * median.item():
        print(f"[health] rank {dist.get_rank()} SLOW: "
              f"{compute_ms:.1f} ms vs median {median.item():.1f} ms")
        # Non-zero exit -> torchelastic treats this worker as failed and,
        # if the node is truly bad, the scheduler drains it before reform.
        raise RuntimeError(f"rank {dist.get_rank()} failed health probe")

    if dist.get_rank() == 0:
        print(f"[health] OK: median compute {median.item():.1f} ms, "
              f"all-reduce {comms_ms:.1f} ms across {world} ranks")
```

The comparison-to-the-median trick matters: you do not know the absolute "correct" matmul time on arbitrary hardware, but you *do* know that in a homogeneous cluster every rank should be within a small factor of every other rank, so an outlier fingers itself. This is the same per-rank-timing logic that catches a live [straggler](/blog/machine-learning/distributed-training/the-straggler) mid-run, run once at startup as a gate. Pair it with a full hardware diagnostic — `dcgmi diag -r 3` runs NVIDIA's stress-and-check suite and will fail a card with marginal HBM or a bad NVLink that a quick matmul might miss — as a slower, deeper gate you run when a node first enters the pool rather than on every job start.

### Catching a preemption: the SIGTERM handler

Preemption is the one interruption you get warned about, and throwing away that warning is pure waste. On a spot instance you get a `SIGTERM` some seconds before termination; on SLURM you can request a `SIGUSR1` a fixed lead time before the job's time limit or a preemption (`--signal=B:USR1@120` sends it 120 seconds early). A signal handler that checkpoints on receipt turns a violent kill into a clean, planned save — you lose zero steps, not half an interval.

```python
import signal
import torch.distributed as dist

class PreemptionHandler:
    """Sets a flag on SIGTERM/SIGUSR1 so the loop checkpoints at a safe point."""
    def __init__(self):
        self.should_stop = False
        signal.signal(signal.SIGTERM, self._catch)
        signal.signal(signal.SIGUSR1, self._catch)  # SLURM preemption signal

    def _catch(self, signum, frame):
        # Do NOT checkpoint inside the handler — you may be mid-collective.
        # Just raise a flag; the training loop acts on it at a clean boundary.
        self.should_stop = True

def train_with_preemption(cfg, model, optimizer, scheduler, loader,
                          start_step, rank):
    handler = PreemptionHandler()
    for step in range(start_step, cfg.max_steps):
        batch = next(loader)
        train_step(model, optimizer, scheduler, batch)

        # Agree across ranks whether to stop, so everyone saves together.
        stop = torch.tensor([1 if handler.should_stop else 0], device="cuda")
        dist.all_reduce(stop)  # any rank flagged -> all stop
        if stop.item() > 0:
            if rank == 0:
                print(f"[preempt] signal at step {step}, checkpointing")
            save_full_state(cfg.ckpt_dir, step, ..., model,
                            optimizer, scheduler)
            dist.barrier()
            # On SLURM, requeue ourselves so we resume when rescheduled.
            if rank == 0:
                os.system(f"scontrol requeue {os.environ['SLURM_JOB_ID']}")
            return
```

Two subtleties that bite people. First, **never checkpoint inside the signal handler itself.** The signal can arrive while you are mid-collective or holding a CUDA stream, and saving from there can deadlock or corrupt. Set a flag; act on it at a clean loop boundary. Second, **the decision to stop must be collective.** The scheduler may signal only rank 0, or signal ranks at slightly different times, so you `all_reduce` the stop flag and everyone agrees to checkpoint on the same step — otherwise one rank saves and barriers while the others charge ahead, and the group hangs. The lead time you request (`@120`) must comfortably exceed your checkpoint cost $C$ plus a step, or the kill lands before the save finishes and the whole exercise was pointless.

#### Worked example: losing one node of sixty-four

Now the payoff, quantified. Same 512-GPU / 64-node run. At step 40,200, node 12 throws an `Xid 79` — a GPU fell off the bus. Compare the two recovery paths on the same failure.

**Rigid restart (checkpoint-and-restart, no elasticity).** NCCL blocks on the dead rank. If you have not tuned the watchdog, you wait up to the default 30-minute `TORCH_NCCL_TIMEOUT` before the communicator aborts and the job exits — 30 minutes gone to a corpse. The scheduler marks the job failed and requeues it. On a busy shared cluster the requeue waits for 64 healthy nodes to free up; if the cluster is full, that is minutes to hours. Say you are lucky and it is 10 minutes. Processes reinitialize, NCCL re-handshakes across 64 nodes (~1–2 minutes), and you read back the last checkpoint (step 39,000, since you save every ~1,900 steps and 40,200 rounds down to a 39,000-ish checkpoint), then redo ~1,200 steps at 3 s each — 60 minutes of rework. Total on a *good* day: 30 (timeout) + 10 (queue) + 2 (init) + 60 (rework) ≈ **100 minutes**, and it can easily be triple that if the queue is contended. Plus you are now running on 64 nodes only *if* a replacement for the dead card is available; if not, you wait for hardware.

**Elastic reform (torchelastic, `--nnodes=60:64`).** With a tuned `TORCH_NCCL_TIMEOUT` of, say, 5 minutes (short steps, so a real hang shows fast) — or better, with NCCL async error handling aborting the communicator the moment the dead rank's connection resets rather than on timeout — the surviving 63 nodes catch the failure in well under a minute. The `torchrun` agents detect the missing worker, re-enter the rendezvous, and agree a new 63-node membership in seconds. Each survivor's script re-executes, reloads the step-39,000 checkpoint, FSDP reshards the state across 63 instead of 64 ranks, and training resumes. You still redo the ~1,200 steps since the last checkpoint — elasticity does not save you the rework, only the restart machinery — but at 63/64 = 98.4% of the throughput. Total: ~2 minutes to detect and reform + ~60 minutes of rework at slightly reduced speed ≈ **62 minutes**, with *no queue wait* and no dependence on a replacement node being ready. When a replacement node does come up, it joins at the next rendezvous and you are back to 64.

| Metric (per failure) | Rigid restart | Elastic reform |
| --- | --- | --- |
| Detect the failure | up to 30 min (default timeout) | < 1 min (async abort / short timeout) |
| Get nodes back | queue wait: 10–120+ min | 0 (keep the survivors) |
| Re-init + NCCL handshake | ~2 min (all 64 nodes cold) | ~seconds (survivors stay up) |
| Rework since checkpoint | ~60 min (both pay this) | ~60 min (at 98.4% speed) |
| Node-count dependence | blocked until 64 available | runs on 63, backfills later |
| **Typical total** | **~100 min (up to 3×)** | **~62 min, no queue** |

Multiply by the 3.4 expected failures over the run and the difference is roughly two-plus hours saved on a good week and most of a *day* saved on a contended cluster — for the price of a rendezvous flag and a restart-safe loop you needed for correctness anyway. The rework term dominates both columns, which is the real lesson: **elasticity removes the fixed restart tax, but only good checkpoint tuning removes the rework, so you need both.** Elastic training without a tuned checkpoint interval still bleeds an hour of redo per failure; a tuned interval without elasticity still bleeds the queue-and-restart tax. They compose.

## The operational patterns: fail-fast versus fail-in-place

Behind the two worked examples sits a genuine architectural choice, and naming it clearly resolves most of the confusion around fault tolerance. There are two philosophies for what to do when a node goes bad, and they trade differently.

**Fail-fast (rigid restart).** On any failure, kill the whole job and restart it from the last checkpoint on a fresh, full-size allocation. Simple, robust, and dead easy to reason about — the world size is constant, the code never has to handle a changing membership, and every restart is identical to a cold start. The cost is the full restart tax on every failure, and the requirement that a full-size allocation be available before you can proceed. This is the right default for small-to-mid jobs, for jobs on dedicated clusters where restart latency is low, and for any code that isn't written to reshard.

**Fail-in-place (elastic).** On a failure, keep the survivors alive, re-form on them, and continue at a reduced size, backfilling replacements as they arrive. Recovers in minutes instead of tens of minutes, keeps the healthy hardware working, and tolerates a cluster that can't always give you a full allocation. The cost is complexity: the code must reshard state across a changing world size, the global batch size changes when the node count changes (so your effective learning-rate schedule and token budget shift unless you compensate — often by adjusting gradient accumulation to hold the global batch constant), and the reduced-size window runs slower until you backfill. Worth it at scale, where the failure cadence makes the restart tax the dominant cost.

![a decision tree that routes an unhealthy node event into elastic reform rigid restart drain and hot-spare swap or accepting a slowdown depending on whether the node is dead or merely sick](/imgs/blogs/fault-tolerance-and-elastic-training-7.webp)

Layered on top of both are the operational patterns that keep the failure rate itself down and the recovery smooth:

- **Health checks at the door.** Run the startup probe on every job start and a deeper `dcgmi diag` when a node first enters the pool. A node that fails a health check never joins a run, so it never fails one. This converts a mid-run hang (expensive) into a pre-run rejection (free). It is the single highest-value operational habit.
- **Drain and evict.** When a node shows warning signs — correctable-ECC counts climbing, a throttling GPU, an NVLink degrade, repeated Xid warnings short of a hard fault — take it out of the pool *proactively* at the next checkpoint boundary rather than waiting for it to fail the collective. A [straggler](/blog/machine-learning/distributed-training/the-straggler) you evict on your schedule costs one clean reform; a straggler that dies mid-collective costs a timeout.
- **Hot spares.** Keep a few idle nodes in the allocation beyond `min_nodes`. When one dies, a spare is already warm and can join the next rendezvous immediately, so you reform back to full size in seconds instead of waiting for the scheduler to allocate fresh hardware. On a 64-node run, allocating 66 nodes and running on 64 with 2 spares costs ~3% of your hardware and buys near-instant backfill — usually a good trade at scale.
- **Automatic restart with backoff.** Wrap the whole thing so that on a reform failure the agent retries with exponential backoff (and a cap via `--max_restarts`), rather than either giving up on the first hiccup or thrashing in a tight crash-loop. A transient network blip should self-heal; a deterministic OOM should surface to a human after a few tries.

| Pattern | What it prevents | Cost | Reach for it when |
| --- | --- | --- | --- |
| Startup health probe | Sick node joins and hangs the run | Seconds per job start | Always |
| `dcgmi diag` gate | Marginal card enters the pool | Minutes per node onboard | Node first joins pool |
| Drain and evict | Straggler-that-dies mid-collective | One clean reform | Warning signs appear |
| Hot spares | Waiting on the scheduler to backfill | ~3% idle hardware | Failure cadence is high |
| Restart with backoff | Crash-loop or premature give-up | A few wasted retries | Any long unattended run |

## The launch: torchrun elastic and SLURM requeue

Wiring it together is mostly a matter of the launch command and one SLURM directive. The elastic `torchrun` invocation, run on every node (SLURM's `srun` fans it out for you), looks like this:

```bash
# Head node hostname is where the c10d rendezvous lives.
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)

torchrun \
  --nnodes=60:64 \                 # min:max nodes -> elastic band
  --nproc_per_node=8 \             # 8 GPUs per node
  --max_restarts=8 \               # cap reforms before giving up
  --rdzv_backend=c10d \            # built-in TCP rendezvous, no etcd
  --rdzv_id="pretrain_7b_run42" \  # unique; restarts of this job find each other
  --rdzv_endpoint="${HEAD_NODE}:29400" \
  --rdzv_conf="join_timeout=600" \ # wait up to 10 min to gather min_nodes
  train.py --config configs/7b_fsdp.yaml
```

The `--nnodes=60:64` is the whole elasticity story in one flag: the job forms as soon as 60 nodes are present, runs on as many as 64, and re-forms within that band as nodes die and rejoin. Set `min_nodes` by asking "what is the smallest configuration I am willing to make progress on?" — too high and a couple of failures drop you below the floor and stall the job; too low and you might limp along on half the cluster, which for a large run means a badly shrunken global batch and possibly a training-dynamics change. A common choice is a floor 5–10% below full size, enough to absorb a handful of failures without stalling.

On SLURM, three directives turn a fragile job into a self-healing one, and the requeue directive is the one people forget:

```bash
#!/bin/bash
#SBATCH --job-name=pretrain_7b
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1          # one torchrun agent per node
#SBATCH --gpus-per-node=8
#SBATCH --time=48:00:00
#SBATCH --requeue                    # let the scheduler requeue on node failure
#SBATCH --signal=B:USR1@120          # SIGUSR1 to the batch step 120s before kill
#SBATCH --open-mode=append           # keep logs across requeues

# Resume-aware: the training script reloads from the latest checkpoint on start,
# so a requeued job continues instead of restarting from step 0.
srun torchrun \
  --nnodes=60:64 --nproc_per_node=8 --max_restarts=8 \
  --rdzv_backend=c10d --rdzv_id="${SLURM_JOB_ID}" \
  --rdzv_endpoint="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1):29400" \
  train.py --config configs/7b_fsdp.yaml
```

There are two layers of fault tolerance stacked here, and it helps to see them as distinct. Torchelastic handles the *in-job* case: a node dies, the survivors reform, no scheduler involvement. `--requeue` plus the `SIGUSR1` handler handles the *whole-job* case: the job is preempted or drops below `min_nodes` and SLURM puts it back in the queue, and because your script reloads from checkpoint on startup, the requeued job resumes rather than restarting. The [SLURM-cluster post](/blog/machine-learning/distributed-training/launching-on-a-slurm-cluster) goes deep on the `sbatch`/`srun` mechanics, node placement, and rendezvous plumbing; here the point is that the two layers compose — elastic reform for the common single-node failure, requeue-and-resume for the rare whole-job loss.

A practical NCCL note that saves you a 30-minute timeout on every failure: set the environment so a dead peer aborts the communicator promptly instead of hanging until the watchdog fires. `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` (on by default in recent PyTorch) makes a NCCL error tear down the communicator and raise, rather than deadlock; and if your steps are short, drop `TORCH_NCCL_TIMEOUT` (formerly `NCCL_TIMEOUT`) from the 30-minute default to something like 300–600 seconds so a genuinely hung rank is declared dead in minutes, not half an hour. The tradeoff is that too-short a timeout can false-positive on a legitimately slow step (a giant checkpoint, a slow first data batch), so set it comfortably above your slowest expected step, not at your median.

## Case studies: what the big runs actually reported

The mechanisms above are not theoretical; the large public training runs all lived this, and their postmortems are the best evidence that the failure-is-a-schedule framing is correct.

**Meta OPT-175B (2022).** The OPT logbook is the most honest public document about what a large run actually feels like, and it is worth reading in full. Training 175B parameters on 992 A100 GPUs (124 nodes) over roughly two months, the team logged *over 100 manual restarts* and a long tail of hardware failures — dead GPUs, ECC errors, and the recurring pain of nodes that had to be cordoned and replaced. Crucially, much of their recovery was *manual*: notice the job died, diagnose, cordon the bad node, restart from checkpoint. The lesson the field took from OPT is precisely why elastic tooling matters — at that failure cadence, manual restart is a full-time on-call job, and automating detect-reform-resume is the difference between shipping and burning out.

**Meta Llama 3 (2024).** The Llama 3 report quantified it cleanly, which is why I anchored the MTBF math on it: 466 interruptions over 54 days on 16,384 H100 GPUs, 419 unexpected, ~78% hardware-caused — about one interruption every three hours. They reported achieving over 90% *effective* training time despite that cadence, which is only possible with heavy automation: fast failure detection, automated recovery, and enough checkpoint hygiene that each interruption cost minutes, not hours. That 90%-effective number is the target the whole reliability track is aiming at, and it is a direct function of driving both the per-failure work loss (checkpoint tuning) and the recovery latency (elastic reform, health checks, hot spares) toward zero.

**BLOOM / BigScience (2022).** The 176B BLOOM run on the Jean Zay cluster (384 A100 GPUs) documented its own steady drip of hardware failures and the operational machinery — SLURM requeue, checkpoint-and-resume, careful node health tracking — needed to push a multi-month run to completion on a shared academic cluster where full allocations were not always instantly available. It is a good counterpoint to the hyperscaler runs: the same failure arithmetic, a smaller cluster, and a heavier reliance on requeue-and-resume because elastic tooling was less mature at the time.

The through-line across all three: nobody at scale got a clean run, everybody planned for interruption, and the ones who reported high effective-training-time percentages did so by making recovery *fast and automatic*, not by getting lucky with the hardware. The MTBF math guarantees you will not get lucky.

## When to reach for this (and when not to)

Fault tolerance is a cost like every technique in this series, and the honest answer to "how much should I invest" depends entirely on where you sit on the MTBF ladder.

**Don't bother with elastic training on a single node.** At 8 GPUs your job MTBF is months; you will almost certainly finish a two-week run without a single hardware failure. Checkpoint every so often as cheap insurance, requeue on the rare crash, and move on. The complexity of resharding across a changing world size buys you nothing when the world size never changes. Fail-fast is correct here.

**Do tune your checkpoint interval as soon as your run is long enough to fail.** This is the cheapest high-value thing in the whole post: compute $\sqrt{2CM}$, round to a step count, done. It costs you one square root and saves you from both the checkpoint-every-100-steps overhead disaster and the checkpoint-every-10000-steps rework disaster. Do this at 64 GPUs and above without hesitation.

**Reach for elastic training when the restart tax starts dominating** — roughly when you are at hundreds of GPUs on a shared or spot-heavy cluster, or any time your job MTBF drops below a day and your restart latency is more than a few minutes. That is the regime where paying the full cold-restart cost on every one of many failures is the largest single line item in your run's wall-clock. Below that, the added complexity of a restart-safe, reshardable loop may not pay for itself — though note that you need most of that machinery for *correct checkpointing anyway*, so the marginal cost of going elastic is smaller than it looks.

**Always run a startup health probe once you are multi-node.** It is nearly free and it converts the most expensive failure — a sick node that joins and hangs the collective hours in — into the cheapest — a rejection at the door. There is no cluster size above one node where this is not worth it.

**Don't over-rotate on exotic redundancy** — in-memory replicated checkpoints, redundant compute, custom RDMA-based state transfer — until you have exhausted the boring wins. Tuned checkpoint interval, fast sharded async saves, elastic reform, health checks, and hot spares get you to the 90%-effective range that the frontier runs report. The exotic stuff is for the last few percent at the very largest scale, and it is a lot of engineering for that last increment.

## Key takeaways

- **Job MTBF is single-node MTBF divided by N.** Failures superpose, so their rates add; double the cluster, halve the time between failures. At frontier scale the probability of interruption during a run is not high, it is one.
- **At scale, recovery is the hot path, not the exception handler.** A fault every few hours over weeks means your detect-reform-resume code runs a hundred-plus times, and every wasted minute is multiplied by that count.
- **The optimal checkpoint interval is $\sqrt{2CM}$**, and the minimum overhead it achieves is $\sqrt{2C/M}$. Compute it; do not guess. Too-frequent wastes time writing, too-rare wastes time redoing, and the square root sits at the bottom of the U.
- **Cutting checkpoint cost $C$ is a double win:** less time writing *and* a smaller optimal interval, which means less rework per failure. This is why sharded, async checkpointing matters so much.
- **Elastic training removes the restart tax, not the rework.** It skips the scheduler round-trip, the queue wait, and the cold start by re-forming on the survivors — but you still redo the steps since the last checkpoint. You need tuned checkpointing *and* elasticity; they compose.
- **The rendezvous with `--nnodes=MIN:MAX` is the whole mechanism.** The job forms at `MIN`, runs up to `MAX`, and re-forms within the band as nodes die and rejoin, resharding state onto the new world size each time.
- **Your training loop must be restart-safe or elasticity is a lie.** Load the latest checkpoint on startup, resume from the saved step, skip consumed data, and let the process group form at the current world size. torchelastic re-runs your script from the top on every reform.
- **Detect fast.** A hung rank hides in silence until a timeout fires; tune `TORCH_NCCL_TIMEOUT` down and rely on async error handling so a dead peer aborts the communicator in minutes, not thirty.
- **Health-check at the door and keep hot spares.** A sick node caught at startup is free to reject; the same node caught mid-collective costs a timeout. Spares turn a scheduler wait into an instant backfill.
- **Distinguish dead from sick.** A dead node forces a reform or restart; a sick-but-alive node is best drained and replaced on your schedule before it fails the collective on the cluster's schedule.

## Further reading

- [Why distributed training](/blog/machine-learning/distributed-training/why-distributed-training) — the four walls and the series map; fault tolerance is the wall you hit when the run is long enough that the hardware gets a vote.
- [Distributed checkpointing](/blog/machine-learning/distributed-training/distributed-checkpointing) — sharded and async saves, and the resharding that lets you resume on a different node count; the mechanism this post depends on to make $C$ small and elastic reform possible.
- [The loss spike after resume](/blog/machine-learning/distributed-training/the-loss-spike-after-resume) — the ways a resume silently drops RNG, optimizer, or data-position state and corrupts the run; the correctness half of restart-safety.
- [The straggler](/blog/machine-learning/distributed-training/the-straggler) — the slow-node problem that, un-evicted, becomes the corpse that hangs the collective; the per-rank timing that the startup health probe reuses.
- [Launching on a SLURM cluster](/blog/machine-learning/distributed-training/launching-on-a-slurm-cluster) — the `sbatch`/`srun`/rendezvous plumbing, node placement, and requeue mechanics that this post's launch section assumes.
- [Monitoring a long run](/blog/machine-learning/distributed-training/monitoring-a-long-run) — the dashboards and alerts that tell you a node is going bad *before* it fails the collective, feeding the drain-and-evict pattern.
- [The distributed training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) — the capstone checklist that ties reliability into the whole decision-and-debugging flow.
- Young, "A first order approximation to the optimum checkpoint interval" (1974) and Daly's refinement (2006) — the original derivations of the $\sqrt{2CM}$ result.
- The PyTorch [torchelastic / `torchrun` docs](https://pytorch.org/docs/stable/elastic/run.html) and the Meta OPT-175B logbook and Llama 3 training report — the primary sources for the rendezvous protocol and the real-world failure cadences cited above.

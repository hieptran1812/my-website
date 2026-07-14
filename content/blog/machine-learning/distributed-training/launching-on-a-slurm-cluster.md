---
title: "Launching on a SLURM Cluster: From sbatch to a Running Multi-Node Job"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "Your code works on 8 GPUs. Now it has to run on 64 across a shared cluster you do not own. This is the end-to-end recipe: how SLURM allocates nodes, how torchrun finds its peers across them, how the rendezvous works, and the dozen small config mistakes that make a multi-node job hang before it even starts."
tags:
  [
    "distributed-training",
    "multi-gpu",
    "multi-node",
    "slurm",
    "torchrun",
    "nccl",
    "pytorch",
    "sbatch",
    "deep-learning",
    "ml-systems",
    "gpu",
  ]
category: "machine-learning"
subcategory: "Distributed Training"
author: "Hiep Tran"
featured: true
readTime: 39
---

There is a specific kind of humbling that only happens on a shared cluster. Your training loop is clean. It ran on your workstation's 8 GPUs at a healthy 48 percent MFU. You wrote the sbatch script, you submitted it to 64 GPUs across eight nodes, and then you watched `squeue` flip your job from `PD` to `R`, opened the log, and saw... nothing. No stack trace. No CUDA error. Just a cursor blinking after "initializing process group" for thirty minutes, until NCCL gave up with a watchdog timeout and the whole allocation died — 64 GPUs, eight nodes, thirty minutes of your quota, burned before a single gradient moved.

That gap — between "my code works on one node" and "it runs across a cluster I do not own" — is where most ML engineers get stuck. Not because the distributed algorithm is hard (you already understand [data parallelism](/blog/machine-learning/distributed-training/ddp-from-first-principles) and [how ranks find each other](/blog/machine-learning/distributed-training/your-first-multi-gpu-run) on a single box), but because the *launch* is a different skill. SLURM has to allocate the right nodes. `srun` has to place your tasks on them. `torchrun` has to bootstrap a process group that spans machines it has never seen. And a dozen environment variables — the network interface, the InfiniBand device, the CPU thread count, the master address — all have to be right, or the job hangs, silently, before step one.

This post is the recipe. By the end you will be able to read a `#SBATCH` header and know exactly what it reserves, choose between the two ways to launch (SLURM as the launcher versus SLURM plus `torchrun`), derive a rendezvous endpoint that all your nodes agree on, set the handful of NCCL and OpenMP variables that decide hang-or-run, and write one complete, annotated sbatch script you can copy and adapt to any cluster. We will finish by diagnosing two real failures — the thirty-minute hang above, and a job that ran at half speed because its dataloader was starved — down to their one-line fixes.

This is the operational glue in the series' spine: the four walls (a model too big to fit, data too big to finish, a run too slow, cost too high) force you to *scale out*, and scaling out means someone has to launch the job on real hardware. That someone is you, at 2am, staring at a `PENDING` queue. Let us make sure the job actually starts.

![Diagram mapping SLURM concepts to distributed ranks: one sbatch job fans out into an allocation of two nodes, srun places eight tasks per node, and those tasks become the global and local ranks](/imgs/blogs/launching-on-a-slurm-cluster-1.webp)

## SLURM in one page: the mental model

SLURM (Simple Linux Utility for Resource Management) is the scheduler that owns the cluster. You do not `ssh` into nodes and run `python train.py` — you *ask* SLURM for resources, it puts your request in a queue, and when the resources free up it hands you an **allocation**: a set of nodes reserved exclusively for your job for a bounded time. Everything else is bookkeeping around that one idea.

There are exactly five commands you need on day one:

- `sbatch script.sh` — submit a batch job. SLURM reads the `#SBATCH` directives at the top of the script, queues the request, and runs the script *on the first allocated node* once resources are granted. This is how you launch training.
- `srun ...` — run a command as one or more **tasks** across the allocation. Inside an sbatch script, `srun` is what actually fans your program out onto every node. On the command line it can also grab a quick interactive allocation.
- `squeue -u $USER` — list your jobs and their state: `PD` (pending, waiting for resources), `R` (running), `CG` (completing).
- `scancel <jobid>` — kill a job.
- `sacct -j <jobid>` — after the fact, query what the job actually used: elapsed time, exit code, max memory, per-step breakdown. This is your black-box recorder.

The mental model that makes the rest click is the mapping in the figure above: **an sbatch job becomes an allocation of nodes, `srun` places tasks onto those nodes, and each task is a rank.** That is the whole translation between SLURM's vocabulary and PyTorch's. Once you see that a "task" is just a process that will become a rank, and that SLURM already knows the node list and can number your tasks globally, the multi-node launch stops being mysterious.

### The directives that reserve your hardware

The `#SBATCH` lines at the top of a script are the request. These are the ones that matter for ML, and getting them consistent with each other is half the battle:

| Directive | What it reserves | ML meaning |
|---|---|---|
| `--nodes=2` | How many machines | The number of physical nodes |
| `--ntasks-per-node=8` | Tasks (processes) per node | Usually = GPUs per node (one task per GPU) |
| `--gpus-per-node=8` | GPUs per node | The whole point; must match your GPU count |
| `--cpus-per-task=8` | CPU cores per task | Cores each rank's dataloader gets |
| `--time=24:00:00` | Wall-clock limit | Hard kill at the limit; checkpoint before it |
| `--partition=gpu` | Which queue/pool | The node type you are allowed to use |
| `--mem=0` | Host RAM per node | `0` means "all of the node's RAM" |
| `--exclusive` | No node sharing | Reserve whole nodes, not slices |

The trap is inconsistency. If you set `--ntasks-per-node=8` but `--gpus-per-node=4`, you have asked SLURM for eight processes fighting over four GPUs — a guaranteed crash or a silent doubling-up where two ranks land on the same device. The rule to internalize: **for one-GPU-per-rank data parallelism, `ntasks-per-node` equals `gpus-per-node`, and `cpus-per-task` equals the node's physical cores divided by that number.** On a 64-core, 8-GPU node, that is 8 tasks and 8 cores per task. We will see later why the `cpus-per-task` line is not optional bookkeeping — it is what stops your loader from starving.

### gres, partitions, and QOS: the cluster-specific layer

Two things about the header vary by cluster and will trip you the first time on a new site. First, older or more conservative SLURM installs do not accept `--gpus-per-node` and instead want the *generic resource* syntax: `--gres=gpu:8`, or `--gres=gpu:a100:8` to demand a specific GPU model. Functionally they request the same thing; which one your cluster accepts is set by the admins, and `sinfo -o "%P %G"` shows you the gres each partition offers. When a job sits `PENDING` forever with reason `(Resources)` or `(QOSMaxGRESPerJob)`, the gres string is the first thing to check.

Second, `--partition` and `--qos` gate what you are *allowed* to ask for. A partition is a pool of nodes (`gpu`, `gpu-preempt`, `debug`); a QOS (quality of service) is a policy layered on top that caps how many GPUs or how much wall-clock your job or account may hold. A common surprise: your job pends indefinitely not because the cluster is full but because your QOS caps you at, say, 32 GPUs and you asked for 64. `sacctmgr show qos format=Name,MaxTRESPU,MaxWall` prints those limits, and `squeue --start -j <jobid>` will show a reason like `(QOSMaxGRESPerUser)` when a policy, not capacity, is what blocks you. Reading these two lines before you submit saves the "why won't my job start" hour that every cluster newcomer spends at least once.

### The environment variables SLURM hands you

When `srun` launches your tasks, it injects environment variables into each one. These are the bridge from SLURM's allocation to your code. The essential ones:

- `SLURM_JOB_NODELIST` — the compressed list of allocated nodes, like `gpu-[001-002]`. You expand it with `scontrol show hostnames`.
- `SLURM_NNODES` — number of nodes (here, 2).
- `SLURM_NODEID` — which node this task is on, 0-indexed (0 or 1).
- `SLURM_PROCID` — the **global task index** across the whole allocation. For 16 tasks this runs 0 to 15. This is your global rank.
- `SLURM_LOCALID` — the task index *within its node*, 0 to 7. This is your local rank — the GPU index on that machine.
- `SLURM_NTASKS` — total tasks across all nodes (16). This is your world size.

Read those definitions again and notice: SLURM has *already computed* everything `torch.distributed` needs. World size is `SLURM_NTASKS`. Global rank is `SLURM_PROCID`. Local rank is `SLURM_LOCALID`. The only thing SLURM does not give you for free is the rendezvous endpoint — the address and port where the ranks first shake hands — and deriving that correctly is the single most common place multi-node launches fail. We will spend a whole section on it.

The arithmetic that ties it together is trivial but worth stating as a law, because violating it is the root of half of all launch bugs:

$$W = N_\text{nodes} \times \text{ntasks-per-node}$$

World size $W$ is nodes times tasks-per-node, and every rank index in $[0, W)$ must map to exactly one GPU. If your `torchrun`/`srun` configuration produces a different count than your `#SBATCH` reserved, some ranks will have no GPU (and hang on the first collective) or two ranks will share one (and NaN or deadlock). Keep the arithmetic consistent end to end and most launch failures never happen.

## The two ways to launch

Here is the fork in the road, and it confuses everyone the first time because both patterns are correct and you will see both in real codebases. The question is: **who is the launcher — SLURM, or `torchrun`?**

![Before and after comparison of two launch patterns: srun spawning one task per GPU and reading rank from SLURM, versus srun placing one task per node and handing each node to torchrun with a c10d rendezvous](/imgs/blogs/launching-on-a-slurm-cluster-2.webp)

### Pattern 1: SLURM is the launcher (srun per GPU)

In the first pattern, you tell `srun` to start **one task per GPU** directly. With `--ntasks-per-node=8` on two nodes, `srun python train.py` launches 16 processes, and SLURM numbers them for you. Each process is a rank. Your Python reads the rank straight out of the SLURM environment and initializes the process group — no `torchrun` involved at all. SLURM *is* the elastic launcher.

The Python side looks like this — a small shim that translates SLURM's variables into what `torch.distributed` expects:

```python
import os
import torch
import torch.distributed as dist


def init_distributed_from_slurm():
    # SLURM has already numbered every task globally and per-node.
    rank = int(os.environ["SLURM_PROCID"])          # global rank, 0..W-1
    local_rank = int(os.environ["SLURM_LOCALID"])   # GPU index on this node
    world_size = int(os.environ["SLURM_NTASKS"])    # total ranks

    # torch.distributed reads these four; we bridge SLURM -> torch.
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    # MASTER_ADDR / MASTER_PORT are set once in the sbatch script (next section).

    torch.cuda.set_device(local_rank)               # pin THIS rank to its GPU
    dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size
```

And the launch, inside the sbatch script, is simply:

```bash
srun python train.py
```

`srun` spawns 16 copies of `train.py`, one per GPU, each with its own `SLURM_PROCID`. Clean, and there is no second launcher to reason about. The cost: SLURM is now your only process supervisor. If one rank dies, SLURM's default behavior is to tear down the step; there is no per-node agent that can locally restart a failed worker. For a job you expect to run to completion in one shot, that is fine.

### Pattern 2: SLURM plus torchrun (srun per node)

In the second pattern — the modern default for large runs — you tell `srun` to start **one task per node**, and that single per-node task runs `torchrun`, which then spawns the eight per-GPU worker processes itself. SLURM's only job is to *place one launcher on each node and give it the node list*. `torchrun` owns everything below that: it spawns workers, assigns local ranks, forms the rendezvous, and — crucially — monitors its workers and can restart them on the same node without SLURM tearing down the whole job.

The launch changes shape. Now `srun` runs two tasks (one per node), and each runs `torchrun` with a rendezvous configuration:

```bash
srun --ntasks-per-node=1 torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc-per-node=8 \
    --rdzv-backend=c10d \
    --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --rdzv-id="$SLURM_JOB_ID" \
    train.py
```

Notice what disappeared from the Python side: you no longer read `SLURM_PROCID`. `torchrun` sets `RANK`, `LOCAL_RANK`, and `WORLD_SIZE` itself, exactly as it does on a single node. Your training script becomes *identical* to the single-node version — that is the real payoff. The same `train.py` runs on your workstation under `torchrun --standalone` and on the cluster under `srun ... torchrun`, with no SLURM-specific code inside it.

Here is the decision, distilled:

| | srun per GPU (Pattern 1) | srun + torchrun (Pattern 2) |
|---|---|---|
| `--ntasks-per-node` | = GPUs (e.g. 8) | 1 |
| Who spawns workers | SLURM | torchrun, per node |
| Rank source in code | `SLURM_PROCID` | `torchrun` sets `RANK` |
| Per-node restart | No (SLURM tears down) | Yes (torchrun restarts locally) |
| Training script | SLURM-aware shim | Identical to single-node |
| Best for | short, one-shot runs | long runs, elastic, [fault tolerance](/blog/machine-learning/distributed-training/fault-tolerance-and-elastic-training) |

For anything that runs long enough to hit a hardware failure — which at 64 GPUs is *most* real training — Pattern 2 is what you want, because it composes with `torchrun`'s elastic restart machinery. The rest of this post uses Pattern 2, and the one piece it demands that Pattern 1 mostly hides is the rendezvous.

## The rendezvous: how nodes find each other

This is the part that hangs. When `torchrun` starts on two separate machines, neither knows the other exists. They have to *meet* — agree on a leader, exchange addresses, count heads until all 16 workers are present — before NCCL can build a communicator. That meeting is the **rendezvous**, and it happens through a small key-value store hosted by one node, at an address and port every other node must be told about.

![Diagram showing two torchrun agents on separate nodes both connecting to one c10d store bound to the master address, which then forms the NCCL process group and reaches the first training step](/imgs/blogs/launching-on-a-slurm-cluster-3.webp)

Three things must line up, and the figure shows why: a **master address** (the host running the store), a **master port** (where it listens), and a **rendezvous id** (so two unrelated jobs on the same node do not collide). Get the address wrong and you get the thirty-minute hang from the intro — because each node stands up its *own* store, waits for peers who are looking at a different address, and never reaches the head count. NCCL's watchdog eventually fires, but by then your quota is gone.

### Deriving the master address correctly

The master must be a real, resolvable hostname of a node *in this allocation* — and every node must derive the *same* one. The canonical trick is to take the first name out of the node list. `scontrol show hostnames` expands `SLURM_JOB_NODELIST` (which is compressed, like `gpu-[001-002]`) into one hostname per line, and you take the first:

```bash
# Runs once, on the sbatch script's node, before srun:
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
```

Because the sbatch script runs on the first allocated node and `export` propagates through `srun` to every task, all 16 workers now agree: the store lives on `MASTER_ADDR`, port `MASTER_PORT`. Node 0 hosts it; node 1 connects to it. The head count reaches 16, the c10d store releases everyone, and NCCL builds its communicator.

Why `scontrol show hostnames | head -1` and not `hostname`? Because `hostname` gives you *the node the command ran on* — which is different on each node. If every node exports its own `hostname` as `MASTER_ADDR`, you get exactly the split-brain in the intro: two stores, no meeting, hang. `scontrol show hostnames | head -1` returns the *same* first name no matter which node evaluates it, because it reads the shared allocation, not the local machine. That single distinction is the most valuable line in this whole post.

A note on the port: pick something in the ephemeral range (say 29500) and make sure it is not already held by another of your jobs on the same node. If you run several jobs per node, derive the port from the job id — `MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))` — so two allocations never fight over the same socket. And `--rdzv-id=$SLURM_JOB_ID` gives the rendezvous a unique name per job, which matters when the elastic backend reuses stores.

### The c10d backend

`--rdzv-backend=c10d` tells `torchrun` to use PyTorch's built-in TCP store for the rendezvous — no external `etcd` cluster to run, no extra dependency. The first worker to arrive at `MASTER_ADDR:MASTER_PORT` becomes the store host; everyone else connects as clients. This is the right default for SLURM: SLURM already guarantees the nodes are up and networked, so you do not need a separate highly-available store. Reserve `etcd` for the rare case where you want a rendezvous that survives the master node itself failing — which, on SLURM, is unusual because losing the master node usually means losing the allocation anyway.

### The static rendezvous, and why c10d beat it

Before c10d elastic rendezvous existed, the standard way to launch multi-node was **static**: you told each node its own rank explicitly. In `torchrun` terms that is `--node-rank=$SLURM_NODEID` with `--master-addr` and `--master-port` passed directly, and every node's index is fixed at launch. You still see this in older recipes:

```bash
srun --ntasks-per-node=1 torchrun \
    --nnodes="$SLURM_NNODES" \
    --node-rank="$SLURM_NODEID" \
    --nproc-per-node=8 \
    --master-addr="$MASTER_ADDR" \
    --master-port="$MASTER_PORT" \
    train.py
```

This works, and it maps cleanly onto SLURM because `SLURM_NODEID` is exactly the node index each `srun` task needs. The reason the c10d form replaced it for long runs is elasticity: with a static rendezvous, node ranks are baked in, so if a worker dies there is no re-forming — the whole job must be relaunched from scratch. With `--rdzv-backend=c10d`, the rendezvous is a live membership protocol that can *re-form* when a node drops and rejoins, which is the foundation the [elastic and fault-tolerant](/blog/machine-learning/distributed-training/fault-tolerance-and-elastic-training) restart machinery builds on. For a short job the static form is marginally simpler; for anything long enough to hit a failure, c10d is what lets a single dead worker not cost you the run. The rest of this post uses c10d for that reason.

## The env vars that decide hang or run

You have the launch pattern and the rendezvous. Now the quiet killers: a small set of environment variables that decide whether NCCL uses the fast fabric or a dead one, and whether your CPUs feed the GPUs or thrash. These do not throw errors. They hang, or they halve your throughput, and you spend a day finding them. The matrix below is the reference; keep it next to your sbatch script.

![Matrix of five critical environment variables showing what each sets, its good value, and its failure signature when set wrong, covering the master address, NCCL interface and InfiniBand device, OpenMP threads, and CUDA visible devices](/imgs/blogs/launching-on-a-slurm-cluster-4.webp)

### The fabric: NCCL_SOCKET_IFNAME and NCCL_IB_HCA

NCCL has to choose a network interface for its control plane and a device for its data plane. On a clean node these auto-detect fine. On a real cluster, nodes often have half a dozen interfaces — the management NIC, a Docker bridge, a loopback, and the actual fast fabric — and NCCL's auto-detection sometimes picks the wrong one. The classic symptom is a job that *rendezvouses fine* (the c10d store is over plain TCP) but then hangs on the first `all_reduce`, because NCCL chose `docker0` for cross-node traffic and there is no route between nodes on that interface.

Two variables fix it:

- `NCCL_SOCKET_IFNAME=ib0` (or `eth0`, or whatever your fast NIC is called) — pins NCCL's socket-based control traffic to the right interface. You can list candidates with `ip -o link show`.
- `NCCL_IB_HCA=mlx5_0` (or `mlx5` to match all Mellanox HCAs) — points NCCL at the InfiniBand host channel adapter for RDMA data transfer. Without it, NCCL may fall back from InfiniBand to TCP-over-Ethernet and quietly run 10x slower — the exact failure dissected in [multinode slower than single node](/blog/machine-learning/distributed-training/multinode-slower-than-single-node).

Set `NCCL_DEBUG=INFO` for the first launch on any new cluster. It prints, per rank, which interface and which HCA NCCL selected and whether it is using RDMA or falling back to sockets. Read those lines once and you will never guess again:

```log
gpu-001:0:142 [0] NCCL INFO NET/IB : Using [0]mlx5_0:1/IB [RO]; OOB ib0:10.0.0.1<0>
gpu-001:0:142 [0] NCCL INFO Using network IB
gpu-002:0:139 [0] NCCL INFO Channel 00/0 : 0[0] -> 8[0] via P2P/IB
```

`Using network IB` is what you want. If you see `Using network Socket`, NCCL never found the InfiniBand path, and your cross-node bandwidth just collapsed from ~200 Gb/s to whatever your Ethernet does.

### The CPU budget: OMP_NUM_THREADS and the oversubscription trap

This is the one that gets everyone, and it is worth deriving properly because the fix is one line but the reasoning is what saves you next time. Here is the mechanism.

Every rank on a node is a full Python process. Each spawns dataloader workers, and every process — plus NumPy, plus PyTorch's CPU-side ops, plus the tokenizer — will, by default, launch OpenMP threads *up to the number of cores it can see*. On a node with 8 ranks, if each process thinks it owns all 64 cores, they collectively demand:

$$\text{threads}_\text{node} \approx R \times (1 + w) \times \tau$$

where $R$ is ranks per node, $w$ is dataloader workers per rank, and $\tau$ is OpenMP threads per process. This must stay under the physical core count $C$, or the OS spends its time context-switching instead of computing. With $R = 8$, $\tau$ defaulting to $C = 64$, you get on the order of 512+ threads fighting over 64 cores — roughly 8x oversubscription. The symptom is brutal and confusing: GPU utilization sawtooths, every step waits on the dataloader's collate, and your MFU drops by a third or more, while `nvidia-smi` shows the GPUs *idle*, not busy. It looks like a GPU problem. It is a CPU thrash problem.

The fix is to give each process its fair share:

```bash
# Each rank gets cpus-per-task cores; cap OMP to that.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK   # e.g. 8 on a 64-core, 8-GPU node
```

Now each process spawns at most 8 OpenMP threads, $8 \times 8 = 64$ matches the cores, and the loader keeps up. This is exactly the failure mode dissected at length in [the data pipeline at scale](/blog/machine-learning/distributed-training/the-data-pipeline-at-scale) — the CPU side starving the GPU side — and on a cluster the default that triggers it is *unset `OMP_NUM_THREADS`*. Set it every time. It costs nothing and it is the single most common cause of a "why is my cluster job slower than my workstation" ticket.

### CUDA_VISIBLE_DEVICES: usually, do nothing

The instinct on a new setup is to set `CUDA_VISIBLE_DEVICES` to control which GPU each rank uses. On SLURM, **do not** — SLURM's GPU cgroup already scopes each task to its allocated device(s), and `torch.cuda.set_device(local_rank)` picks the right one within that scope. If you manually export `CUDA_VISIBLE_DEVICES=0` in the sbatch script, it propagates to every task, and now all 8 ranks on a node see only "GPU 0" — they pile onto one device, and you get an out-of-memory crash or a silent 8x slowdown. Leave it unset and let SLURM and `set_device` cooperate. The one exception is debugging a single-GPU repro, where you deliberately want to hide the rest.

### Binding for NUMA locality

The last knob is CPU-to-GPU affinity. On a two-socket node, GPUs are wired closer to one socket's cores and memory than the other's. If a rank's process runs on the far socket, every host-to-device copy crosses the inter-socket link and adds latency. `srun --cpu-bind=cores` (or the more explicit `--cpu-bind=verbose,cores`) pins each task to a contiguous core set, and combined with `--gpus-per-task=1` SLURM will place each rank's cores near its GPU. This is a single-digit-percent throughput win in most cases and a bigger one for data-heavy pipelines — worth the one line, not worth agonizing over on day one.

## The full sbatch script

Here is the centerpiece: one complete, annotated script that puts every piece together. This is the thing you copy, change three values in, and submit. It uses Pattern 2 (srun + torchrun), derives the master address the safe way, sets the NCCL and OMP variables, and lays out logging so you can find the one node that misbehaves.

```bash
#!/bin/bash
#SBATCH --job-name=llm-7b-pretrain
#SBATCH --nodes=2                       # 2 machines
#SBATCH --ntasks-per-node=1             # ONE srun task per node -> runs torchrun
#SBATCH --gpus-per-node=8               # 8 GPUs each -> 16 GPUs total
#SBATCH --cpus-per-task=64              # all cores to the single per-node task
#SBATCH --mem=0                         # all host RAM on the node
#SBATCH --time=24:00:00                 # hard wall-clock limit
#SBATCH --partition=gpu                 # the GPU queue on this cluster
#SBATCH --exclusive                     # do not share nodes with other jobs
#SBATCH --signal=B:SIGTERM@120          # send SIGTERM 120s before the kill
#SBATCH --output=logs/%x-%j/node-%N.out # per-node log: jobname-jobid/node-host.out
#SBATCH --error=logs/%x-%j/node-%N.err

set -euo pipefail

# --- 1. Rendezvous: derive ONE master addr all nodes agree on ---------------
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))   # per-job port, no clashes

# --- 2. NCCL fabric: pin the fast interface + InfiniBand HCA ----------------
export NCCL_SOCKET_IFNAME=ib0           # fast NIC for control traffic
export NCCL_IB_HCA=mlx5                 # match all Mellanox HCAs -> RDMA
export NCCL_DEBUG=WARN                  # INFO on a new cluster; WARN once it works

# --- 3. CPU threads: stop the loader from oversubscribing -------------------
# 8 GPUs share 64 cores -> 8 cores per rank -> cap OMP to that.
export OMP_NUM_THREADS=8

# --- 4. Activate the environment (baked into the image or a venv) -----------
source /shared/envs/train/bin/activate

# --- 5. Launch: srun places one torchrun per node; torchrun spawns 8 ranks --
srun --ntasks-per-node=1 --cpu-bind=cores \
  torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc-per-node=8 \
    --rdzv-backend=c10d \
    --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --rdzv-id="$SLURM_JOB_ID" \
    train.py \
      --config configs/llm-7b.yaml \
      --output-dir /shared/ckpts/$SLURM_JOB_ID
```

Read it top to bottom the way the machine does, and it maps cleanly onto the stack in the figure below. The `#SBATCH` block is the *request* — SLURM parses it, queues you, and when granted, runs everything below on the first node. The `export` lines set the shared context that `srun` will carry to every node. The `srun` at the bottom is the *fan-out*: it starts one `torchrun` per node, and each `torchrun` becomes the per-node launcher that spawns the eight ranks.

![Layered stack diagram of one node showing SLURM allocation on top handing down to srun, then torchrun spawning eight processes, then python running the DDP ranks, then NCCL and the InfiniBand fabric at the bottom](/imgs/blogs/launching-on-a-slurm-cluster-5.webp)

The stack view is the debugging map. Each layer owns exactly one job, so when init fails you can localize it: a bad `#SBATCH` request never leaves the SLURM layer (wrong node count, denied partition); a rendezvous hang lives at the `torchrun` layer (master address); an `all_reduce` hang lives at the NCCL layer (wrong interface); a slow-but-running job usually lives at the Python/loader layer (OMP oversubscription). When someone pastes you a hang, the first question is always "which layer?" — and the stack tells you where to look.

Two details in the script pay for themselves. The `--output=logs/%x-%j/node-%N.out` pattern writes a *separate log per node* (`%N` is the node hostname, `%x` the job name, `%j` the job id). When a job hangs, you almost never have a symmetric failure — one node's log will look different, and per-node logs let you spot the odd one out in seconds instead of grepping a 16-rank interleaved mess. The `--signal=B:SIGTERM@120` directive is the other: it tells SLURM to send your job a `SIGTERM` 120 seconds *before* the hard time-limit kill, giving your code a grace window to checkpoint. Which brings us to the lifecycle.

### Test interactively before you batch

The slowest way to debug a launch is to fix one line, `sbatch`, wait ten minutes in the queue, read the log, and repeat. Do not do that. Grab an *interactive* allocation and iterate in seconds. `salloc` reserves nodes and drops you into a shell that holds them:

```bash
# Reserve 2 nodes x 8 GPUs for an hour, interactively:
salloc --nodes=2 --gpus-per-node=8 --ntasks-per-node=1 \
       --cpus-per-task=64 --time=1:00:00 --partition=gpu

# Now you are in a shell WITH the allocation. Test the rendezvous derivation:
scontrol show hostnames "$SLURM_JOB_NODELIST"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
srun bash -c 'echo "$(hostname) -> MASTER_ADDR=$MASTER_ADDR"'
```

That last line — running a trivial command across the allocation and printing what each node *sees* — is the single most useful debugging move on a cluster. It confirms in one second whether every node agrees on the master address, which interface `ip -o link` reports, and what `nvidia-smi -L` enumerates per node, before you have spent a minute of real GPU time. Once the two-line smoke test passes interactively, promote the same commands into the sbatch script and submit with confidence. A `srun --pty --nodes=1 --gpus-per-node=8 bash` gets you an interactive shell *on a compute node* for the same purpose when you only need one box.

### Where the environment comes from

The script's `source /shared/envs/train/bin/activate` line hides a real decision: how does the *same* Python environment reach every node? On a shared cluster the workers do not share your login node's filesystem for pip packages unless someone arranged it. Three common answers, in rough order of reproducibility:

- **A shared filesystem venv or conda env**, as above — simple, works when `/shared` is mounted on every compute node, but fragile if a node has a different CUDA driver than the one your wheels were built against.
- **`module load`** — many HPC sites expose curated software stacks (`module load cuda/12.4 python/3.11`) that pin driver-matched builds per node. Use these for the CUDA toolkit and MPI even if your Python comes from a venv.
- **A container** via `enroot`/`pyxis` (`srun --container-image=...`) or Apptainer/Singularity — the most reproducible option, because the image carries the exact CUDA, NCCL, and Python versions and the node only supplies the driver and the fabric. On clusters where node images drift, this is the difference between "works today" and "works next month". The tradeoff is one more layer to get right — the container must expose the InfiniBand devices (`--container-mounts` for `/dev/infiniband`) or NCCL falls back to TCP inside the container, reproducing the fabric failure from the env-var section.

Whichever you pick, pin it. A launch that pulls "latest" of anything is a launch that will mysteriously break the week a base image updates.

## The lifecycle of a job, and operating it

A production training job is not fire-and-forget. It waits in a queue, wins an allocation, trains until it bumps the time limit, and — if you built it right — checkpoints itself and requeues to continue. The timeline below is the shape of a well-behaved long run.

![Timeline of a sbatch job from submission through queue, allocation, rendezvous, steady-state training, a SIGTERM grace window before the time limit, a checkpoint flush, and an automatic requeue to resume](/imgs/blogs/launching-on-a-slurm-cluster-6.webp)

### Checkpoint before the time limit

Clusters cap wall-clock time — 24 hours is common — but training a 7B model takes days. So you run in segments: train up to the limit, checkpoint, resume in the next allocation. The danger is the *hard* kill: if SLURM `SIGKILL`s you at exactly 24:00:00, you lose everything since the last periodic checkpoint. The `--signal=B:SIGTERM@120` directive plus a signal handler closes that gap. The handler catches the `SIGTERM`, sets a flag, and the training loop checkpoints at the next safe step and exits cleanly:

```python
import signal
import torch.distributed as dist

_should_stop = False


def _on_sigterm(signum, frame):
    global _should_stop
    _should_stop = True   # do NOT checkpoint inside the handler; just flag it.


signal.signal(signal.SIGTERM, _on_sigterm)

for step, batch in enumerate(loader):
    loss = train_step(model, batch)
    if _should_stop or step % save_every == 0:
        if dist.get_rank() == 0:
            print(f"checkpointing at step {step}", flush=True)
        save_checkpoint(model, optimizer, step)   # all ranks; sharded save
        if _should_stop:
            dist.barrier()   # let every rank finish its shard
            break
```

The 120-second grace window has to be larger than your checkpoint-write time — for a sharded 7B checkpoint over a shared filesystem, tens of seconds is typical, so 120s is comfortable. If you use [distributed checkpointing](/blog/machine-learning/distributed-training/fault-tolerance-and-elastic-training) with async save, the flush is even faster. Pair this with `--requeue` (or `sbatch --requeue`) and SLURM will automatically resubmit the job after it exits, so a single `sbatch` becomes a self-continuing run that survives time limits and preemption.

### Job arrays for sweeps

When you need to run the same script across a grid of hyperparameters, do not write 20 sbatch files. Use a **job array**: one script, `N` copies, each with a distinct `SLURM_ARRAY_TASK_ID` you index into a config list:

```bash
#!/bin/bash
#SBATCH --job-name=lr-sweep
#SBATCH --array=0-7            # 8 array tasks, ids 0..7
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=6:00:00

LRS=(1e-4 2e-4 3e-4 5e-4 8e-4 1e-3 2e-3 3e-3)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}

srun torchrun --standalone --nproc-per-node=8 \
  train.py --lr "$LR" --output-dir /shared/sweep/lr-$LR
```

SLURM schedules the eight array tasks independently as capacity frees up, so a sweep drains through the queue without you babysitting it. Each is a normal single-node job here, but arrays compose with multi-node too.

### Reading what actually happened

After a job — running or finished — `sacct` is your source of truth. It reports what the scheduler *observed*, which is often more honest than your own logs:

```bash
sacct -j 184213 --format=JobID,JobName,State,Elapsed,MaxRSS,MaxVMSize,ExitCode,NNodes,ReqTRES%40
```

The columns that matter for a training job: `State` (did it `COMPLETED`, `TIMEOUT`, or `FAILED`?), `Elapsed` (real wall-clock, to reconcile against your step count), `MaxRSS` (peak host RAM per task — if this is near the node limit, you are one bad batch from an OOM-kill of your Python process, not a CUDA OOM), and `ExitCode`. When a job dies mysteriously, `sacct` telling you `State=OUT_OF_MEMORY` or `TIMEOUT` instantly rules out half the hypotheses. Live, `squeue --start` estimates when a pending job will run, and `scontrol show job <jobid>` dumps the full allocation detail including the actual node list you got.

## Worked examples

Enough mechanism. Here are two real failures, the way they present, and the reasoning to the fix. Both are launches that *look* like they should work and do not, and both are one line away from working.

#### Worked example: the 2-node job that hangs at init

The setup: a 7B model, 2 nodes, 8 A100 80GB per node, 16 GPUs total, InfiniBand HDR fabric. Someone adapted a single-node script and, wanting to be "explicit", set the master address inside the launched program using Python's `socket.gethostname()`. The job allocates in seconds, both nodes go `R`, and then the logs stop after `Added key: store_based_barrier_key:1 to store for rank: 0` on *each* node. Thirty minutes later, NCCL's watchdog fires:

```log
[rank0]: torch.distributed.DistBackendError: NCCL error: ...
[rank0]: Watchdog caught collective operation timeout:
[rank0]: WorkNCCL(SeqNum=1, OpType=ALLREDUCE) ran for 1800000 milliseconds
[rank0]:   before timing out.
[E] Rendezvous timed out: not all 16 members joined within 1800s
```

Reason step by step. The rendezvous timed out with fewer than 16 members — so the nodes never met. If NCCL had chosen a bad *interface*, the rendezvous (plain TCP) would still form and the hang would come later, on the first `all_reduce`. But this hangs at the rendezvous itself, before any collective. That points at the master address. Print it per node — `srun bash -c 'echo $(hostname) sees MASTER_ADDR=$MASTER_ADDR'` — and the smoking gun appears:

```console
gpu-001 sees MASTER_ADDR=gpu-001
gpu-002 sees MASTER_ADDR=gpu-002
```

Each node used its own hostname. Node 0 stood up a store on `gpu-001` and waited for peers; node 1 stood up a *different* store on `gpu-002` and waited for peers. Two stores, no meeting, hang — exactly the split-brain the rendezvous figure warns about. The fix is one line, moved to the sbatch script so it evaluates *once* and propagates identically:

```bash
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
```

Now both nodes derive `gpu-001` (the first name in the shared allocation, regardless of which node evaluates it), both connect to the one store on `gpu-001`, the head count reaches 16, and the first step runs in about 12 seconds instead of never. This is the before-and-after below, and it is the same class of hang covered in [the NCCL timeout war story](/blog/machine-learning/distributed-training/the-nccl-timeout-that-hung-the-job) — a mismatch that never surfaces as a clean error, only as a wait.

![Before and after comparison showing each node exporting its own hostname as master leading to two stores that never meet and a thirty-minute hang, versus deriving one master from scontrol so all sixteen ranks join one store and the first step runs in twelve seconds](/imgs/blogs/launching-on-a-slurm-cluster-7.webp)

#### Worked example: the job that ran at half speed

Same cluster, different failure. This time the job *runs* — no hang, loss goes down, everything looks healthy — but it trains at roughly 60 percent of the throughput the same code hit on the workstation. The profile is the tell: GPU utilization sawtooths between 95 percent and 20 percent on a period that matches the batch time, and `nvidia-smi dmon` shows the SMs going idle between steps. The GPUs are *waiting*.

Waiting for what? Not for gradients — an `all_reduce` stall would show as a plateau, not a sawtooth, and NCCL debug shows healthy IB. Waiting for *data*. The dataloader cannot assemble the next batch in time, so the GPUs starve between steps. Now ask why the loader is slow on the cluster but not the workstation. Same code, same dataset. The difference is the node: 64 cores, 8 ranks, and — the missing line — `OMP_NUM_THREADS` unset. Each of the 8 rank processes, plus each dataloader worker, spawns OpenMP threads up to all 64 cores. Do the arithmetic from the mechanism section: 8 ranks each grabbing 64 threads is ~8x oversubscription, the OS thrashes on context switches, and the CPU-bound collate that feeds each batch takes far longer than the GPU step it should overlap with. On the workstation the ratio happened to be fine; on the cluster's fatter node it tips into thrash.

The fix is the one line from the env-var section:

```bash
export OMP_NUM_THREADS=8   # = cpus-per-task; 8 ranks x 8 = 64 cores, no oversub
```

Throughput jumps back to workstation levels — the loader now keeps pace, the sawtooth flattens to a steady 90-plus percent utilization, and the run's tokens/s recovers by the ~40 percent it had lost. This is the cluster incarnation of the starvation dissected in [the data pipeline at scale](/blog/machine-learning/distributed-training/the-data-pipeline-at-scale): the GPU was never the bottleneck; the CPU-side pipeline was, and the trigger was a default that only bites on a many-core shared node.

## Measuring it honestly

Both fixes above are throughput claims, so it is worth stating how you *measure* one without fooling yourself — because a naive timer will lie to you on a cluster more than anywhere else.

- **Warm up, then time steady state.** The first 10-20 steps include CUDA context creation, NCCL communicator setup, autotuner passes, and filesystem caching. Discard them. Time steps 50-150, not 0-100.
- **Synchronize before you read the clock.** `torch.cuda.synchronize()` before and after the timed region — GPU work is asynchronous, and without it you are timing kernel *launches*, not kernel *execution*.
- **Separate the loader confound.** To prove a slowdown is comms and not data, replace the dataloader with a single cached batch you reuse every step. If throughput jumps, the loader was the wall (the second worked example); if it does not, look at NCCL and placement.
- **Watch for thermal and clock throttling.** On a packed node, sustained load lowers the GPU boost clock. `nvidia-smi -q -d CLOCK` and `PERFORMANCE` will show throttle reasons. A 5 percent throughput drift over an hour is often clocks, not your code.
- **Report per-GPU, not aggregate.** Tokens/s *per GPU* at N GPUs, compared to 1 GPU, is your scaling efficiency. Aggregate tokens/s always goes up with more GPUs; efficiency is what tells you the launch is healthy.

Here is the before/after for the two examples on the named hardware, framed the way you would put it in a run report. The numbers are representative of a 7B-class model on this class of node; treat them as order-of-magnitude, not a benchmark:

| Scenario | Hardware | Init time | Tokens/s per GPU | 16-GPU efficiency |
|---|---|---|---|---|
| Wrong `MASTER_ADDR` | 2x8 A100 80GB, IB HDR | never (30 min timeout) | 0 | job dies |
| Fixed rendezvous | 2x8 A100 80GB, IB HDR | ~12 s | ~12,000 | ~92% |
| `OMP_NUM_THREADS` unset | 2x8 A100 80GB, IB HDR | ~12 s | ~7,200 | ~55% |
| `OMP_NUM_THREADS=8` | 2x8 A100 80GB, IB HDR | ~12 s | ~12,000 | ~92% |

The shape of that table is the whole lesson: two one-line environment fixes are the difference between a dead job, a half-speed job, and a job at 92 percent scaling efficiency. Nothing in the model changed.

## Stress-testing the launch

The recipe above works cleanly at 2 nodes. The engineering question is whether it holds when you change the variables that break things. Reason through the four that matter.

**What happens at 64 GPUs (8 nodes)?** The rendezvous cost grows — 64 workers all connecting to one c10d store instead of 16 — but it is still seconds, not minutes, because the store handshake is cheap relative to the NCCL communicator build that follows. What does grow is the *tail sensitivity*: with 8 nodes, the probability that *one* node has a degraded NIC, a stuck process from a previous job, or a thermal issue is eight times higher than with one node. The rendezvous will wait for the slowest member to join, so a single sick node stalls all 64 GPUs at init. This is why the per-node log (`%N`) matters more the larger you scale — at 8 nodes you *will* eventually have an asymmetric failure, and your only fast path to the culprit is comparing per-node logs. The launch recipe does not change; your *observability* has to scale with it.

**What happens on PCIe instead of NVLink?** Within a node, if the GPUs talk over PCIe rather than NVLink, intra-node `all_reduce` bandwidth drops from NVLink's hundreds of GB/s to PCIe's ~25-30 GB/s, and the *launch* is unaffected but the *efficiency* falls. The lever that helps is not a SLURM flag — it is making sure NCCL uses the best available intra-node path, which it does automatically, and confirming with `NCCL_DEBUG=INFO` that it reports `via P2P` rather than `via SHM` (shared-memory host bounce) or `via SYS`. If you see host-memory staging where you expected P2P, a PCIe ACS (access control services) BIOS setting is often blocking peer-to-peer, and that is a cluster-admin fix, not a script one.

**What happens when the batch is tiny?** Small per-GPU batches make each step's compute short, so the fixed costs — the `all_reduce` latency, the dataloader collate, the Python overhead — dominate a larger fraction of the step. On a launch that was fine at batch 8 per GPU, dropping to batch 1 can tank efficiency to 40 percent because comms and launch overhead no longer hide behind compute. The fix is not in the launcher; it is gradient accumulation to restore a healthy compute-to-comms ratio, covered in [DDP from first principles](/blog/machine-learning/distributed-training/ddp-from-first-principles). But it shows up first as "my cluster launch scales badly", so it is worth ruling in.

**What happens when one node is a straggler?** A collective is only as fast as its slowest participant — every `all_reduce` is a barrier. If one node runs at 80 percent of the others' clock (thermal throttling, a noisy neighbor on a non-`--exclusive` allocation, a failing NIC), the *whole job* runs at that node's pace, and your 92 percent efficiency quietly becomes 74 percent. `--exclusive` in the sbatch header removes the noisy-neighbor cause by refusing to share nodes. The rest — finding and evicting a genuinely slow node — is its own discipline, dissected in [the straggler](/blog/machine-learning/distributed-training/the-straggler); the launch-time defense is `--exclusive` plus per-node throughput logging so a straggler shows up as one node's tokens/s lagging the pack.

The through-line of all four: the *launch recipe* is stable, but its *efficiency* is exposed to hardware and configuration you do not fully control. Build the observability — per-node logs, `NCCL_DEBUG` on first launch, per-GPU throughput — so that when efficiency drops you can localize which of these four it is in minutes.

## Case studies and real numbers

SLURM plus `srun` plus `torchrun` (or its predecessors) is the launch path behind most of the large open training runs, and the postmortems are consistent about where the pain is.

**BigScience BLOOM (176B, Jean Zay cluster).** The BLOOM training was run on 384 A100 80GB GPUs (48 nodes of 8) under SLURM, using Megatron-DeepSpeed. Their engineering notes are candid that a large share of early effort went not into the model but into the *launch and reliability* layer — deriving the rendezvous, pinning NCCL to the right InfiniBand HCAs, and building the `--requeue` plus checkpoint-on-signal loop so a multi-month run could survive node failures and the cluster's wall-clock limits. The training log they published reads as much like an ops diary as an ML one: hardware failures, requeues, and NCCL settings dominate.

**Meta OPT-175B.** The OPT logbook is the canonical "training at scale is an operations problem" document. Running on 992 A100 80GB GPUs under SLURM, the team recorded dozens of restarts driven by hardware failures and NCCL errors; their standard operating procedure was to detect a hang, cancel with `scancel`, and requeue from the last checkpoint — the exact loop this post's signal-handler and `--requeue` pattern automates. The lesson they draw explicitly: at this scale, the launcher and the restart machinery are first-class parts of the training system, not glue.

**Meta Llama and the Grand Teton fleet.** Meta's published accounts of training the Llama family describe running on the order of 16,000 H100 GPUs, and they are explicit that at that scale the launcher and the fault-tolerance layer are *the* engineering problem: with tens of thousands of GPUs, some hardware fails roughly every few hours, so a run that cannot detect a failed node, evict it, and restart from a checkpoint simply never finishes. Their tooling around SLURM-style scheduling — health-checking nodes before admitting them to the allocation, draining suspect nodes, and requeueing — is the industrial version of the `--exclusive` plus per-node-log plus checkpoint-on-signal hygiene in this post. The lesson generalizes down: the same discipline that keeps a 16,000-GPU run alive is what keeps your 64-GPU run from dying on the first bad node.

**The InfiniBand fallback tax.** NCCL's own documentation and countless cluster postmortems converge on the same number: when NCCL fails to find the InfiniBand path and falls back to TCP-over-Ethernet, cross-node collective bandwidth drops roughly an order of magnitude — from the ~150-200 Gb/s an HDR fabric delivers to the ~10-25 Gb/s of a commodity Ethernet control NIC. On a communication-bound run that can more than halve total throughput, and it presents as "multi-node is slower than we expected" with no error at all — which is why `NCCL_DEBUG=INFO` confirming `Using network IB` on the first launch is non-negotiable. This is the same fallback autopsied in [multinode slower than single node](/blog/machine-learning/distributed-training/multinode-slower-than-single-node).

## When to reach for a SLURM cluster (and when not)

Every layer of this machinery is a cost, so be honest about when it pays.

- **Do not go multi-node until you have saturated one node.** If your model fits on 8 GPUs and DDP already saturates NVLink at high efficiency, adding a second node adds InfiniBand hops that only *lower* your per-GPU efficiency unless the extra scale earns it. Prove single-node efficiency first; the [scaling-a-7B journey](/blog/machine-learning/distributed-training/scaling-a-7b-llm-1-to-64-gpus) walks the full ladder.
- **Use Pattern 1 (srun per GPU) for short, one-shot jobs.** If the job finishes in an hour and you do not need per-node restart, skipping `torchrun` removes a layer to reason about. Reach for Pattern 2 the moment the run is long enough to hit a hardware failure.
- **Do not hand-tune NCCL variables you have not measured.** Set `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`, and `OMP_NUM_THREADS` — those are load-bearing. Leave `NCCL_ALGO`, `NCCL_P2P_DISABLE`, and the rest at their defaults unless `NCCL_DEBUG` or a profile *shows* you a problem. Speculative env-var tuning is how people make working jobs slower.
- **Prefer the scheduler's own tools over shell hacks.** Job arrays for sweeps, `--requeue` for continuation, `--signal` for graceful checkpoint, `--dependency` for pipelines. Every custom bash loop you write to imitate these is a thing that breaks at 2am.
- **If you are on one workstation with 8 GPUs and no scheduler, you do not need any of this.** `torchrun --standalone --nproc-per-node=8 train.py` is the whole launch. SLURM earns its complexity only when the cluster is shared and multi-node — which is precisely when the four walls have pushed you past a single box.

## Key takeaways

- **SLURM allocates; `srun` places; the task is the rank.** Once you see that `SLURM_PROCID` is global rank, `SLURM_LOCALID` is local rank, and `SLURM_NTASKS` is world size, the launch is a translation, not a mystery.
- **Keep the arithmetic consistent.** World size equals nodes times tasks-per-node, `ntasks-per-node` equals `gpus-per-node`, `cpus-per-task` equals cores divided by GPUs. Break any of these and ranks lose GPUs or share them.
- **Pattern 2 (srun + torchrun) is the modern default** because it makes your training script identical to the single-node version and gives each node a local supervisor that composes with elastic restart.
- **Derive `MASTER_ADDR` once, with `scontrol show hostnames | head -1`.** Never `hostname` per node. This one line prevents the most common multi-node hang.
- **Set `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`, and `OMP_NUM_THREADS` every time.** The first two keep NCCL on the fast fabric; the third stops the loader from starving the GPUs. Confirm with `NCCL_DEBUG=INFO` on any new cluster.
- **Leave `CUDA_VISIBLE_DEVICES` to SLURM.** Setting it by hand collapses every rank onto GPU 0.
- **Log per node with `%N`, checkpoint on `SIGTERM`, requeue automatically.** A long run is an ops artifact; build the lifecycle in from the start.
- **Measure per-GPU tokens/s in steady state, after warm-up, with a synchronized clock** — and rule out the loader confound before you blame comms.

## Further reading

- [Why distributed training](/blog/machine-learning/distributed-training/why-distributed-training) — the four walls that force you onto a cluster in the first place.
- [Your first multi-GPU run](/blog/machine-learning/distributed-training/your-first-multi-gpu-run) — `torchrun`, rank/local-rank/world-size, and `init_process_group` on a single node, before you add SLURM.
- [Multinode slower than single node](/blog/machine-learning/distributed-training/multinode-slower-than-single-node) — the interconnect-fallback autopsy behind the NCCL fabric variables here.
- [The data pipeline at scale](/blog/machine-learning/distributed-training/the-data-pipeline-at-scale) — why an unset `OMP_NUM_THREADS` starves the GPUs, in full.
- [Fault tolerance and elastic training](/blog/machine-learning/distributed-training/fault-tolerance-and-elastic-training) — the `--requeue` plus checkpoint-on-signal loop, and elastic rendezvous, in depth.
- [Debugging DDP and multi-GPU jobs](/blog/machine-learning/debugging-training/debugging-ddp-and-multi-gpu) — the broader taxonomy of multi-GPU hangs and crashes.
- [The distributed training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) — the capstone checklist that ties launching, debugging, and scaling together.
- The SLURM documentation (`sbatch`, `srun`, `sacct` man pages), the PyTorch `torchrun`/Elastic docs, and the NVIDIA NCCL environment-variable reference — the three primary sources for everything above.

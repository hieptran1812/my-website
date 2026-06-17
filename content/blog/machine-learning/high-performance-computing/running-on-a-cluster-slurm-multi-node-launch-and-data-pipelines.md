---
title: "Running on a Cluster: SLURM, Multi-Node Launch, and Data Pipelines"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Go from one box to a whole cluster: submit with SLURM, launch torchrun across nodes, feed the GPU so it never starves, and checkpoint at scale without freezing the job."
tags:
  [
    "high-performance-computing",
    "gpu",
    "slurm",
    "distributed-training",
    "data-pipeline",
    "torchrun",
    "checkpointing",
    "deep-learning",
    "ml-systems",
    "webdataset",
  ]
category: "machine-learning"
subcategory: "High Performance Computing"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/running-on-a-cluster-slurm-multi-node-launch-and-data-pipelines-1.png"
---

You spent three weeks getting your training loop perfect on a single 8-GPU box. The model is fast, the kernels are fused, the GPUs sit at 92% utilization, and you are proud of it. Then your manager says the magic words: "Great. Now run it on 64 GPUs and have it done by Friday." You SSH into the cluster, and the first thing that happens is that your script does nothing. No GPUs found. No error you recognize. You are no longer on a machine — you are a guest in a shared facility with a bouncer at the door called the scheduler, and you do not yet speak its language.

This is the moment most engineers discover that "scaling up" is not one problem but three, and only one of them is the model. The first is **getting the job to launch at all** across many machines you do not own and cannot SSH into directly. The second is **feeding those GPUs** — because the dirty secret of large training runs is that the data loader, not the GPU, is the bottleneck more often than anyone admits, and a cluster makes this worse, not better, by putting your data on a filesystem that lives across the network. The third is **surviving** — checkpointing a model so big that the naive "save everything to one file" approach freezes your entire 64-GPU job for four minutes every hour while the GPUs sit idle and the meter keeps running.

This post is the unglamorous 30% of cluster training that separates a run at 45% effective utilization from one at 90%. We will go from one node to many, the same Transformer the whole way: submitting with **SLURM** (the scheduler that owns the cluster), launching **`torchrun` across nodes** so 32 or 64 processes find each other and form one training job, building a **data pipeline** that out-produces the GPU instead of starving it, and **checkpointing at scale** so a save costs nine seconds instead of two hundred. None of this is about the model architecture. All of it decides whether your run finishes Friday or the following Wednesday.

![Diagram of a GPU cluster showing a login node submitting to a SLURM scheduler that allocates many compute nodes connected by InfiniBand to a shared parallel filesystem](/imgs/blogs/running-on-a-cluster-slurm-multi-node-launch-and-data-pipelines-1.png)

The figure above is the whole world you are about to enter. You log into a **login node** — a small shared machine with no GPUs whose only job is to let you edit files and submit work. You hand a script to the **scheduler** (SLURM's controller), which finds idle **compute nodes** (the machines with the 8 GPUs each), reserves them for you, and starts your processes on them. Those nodes talk to each other over a fast **InfiniBand** fabric (more on that later) and they all read their training data from one **parallel filesystem** shared by the entire cluster. Every problem in this post lives somewhere on this diagram. Let us walk it, left to right, and make each piece concrete.

This is the eleventh stop in the series on high-performance computing for AI engineers. If you have not read [why HPC is the bottleneck for modern AI](/blog/machine-learning/high-performance-computing/why-hpc-is-the-bottleneck-for-modern-ai), it sets up the three walls — compute, memory bandwidth, and communication — that every post here reads off the roofline and the profiler. Today we add a fourth practical wall that the roofline does not show you: **the cluster itself**, where the scheduler, the network filesystem, and the checkpoint I/O quietly eat a third of your throughput if you let them.

## What a cluster actually is, and why you cannot just SSH in

On your single box, you owned the machine. You typed `python train.py`, the operating system gave your process all 8 GPUs, and that was that. A cluster is different in one fundamental way: **it is shared, and a program decides who gets what.** A modern training cluster might have 200 nodes, each with 8 H100 GPUs — 1,600 GPUs total — and at any moment dozens of engineers and automated pipelines want a slice. If everyone just SSH'd in and ran whatever they wanted, two jobs would land on the same GPU, both would slow to a crawl or crash with out-of-memory, and the \$40-million facility would deliver a fraction of its value.

So a cluster has a **scheduler**: a piece of software that owns every node and hands out GPUs the way a maître d' hands out tables. You do not pick the node. You describe your needs — "I want 4 nodes, 8 GPUs each, for up to 24 hours" — and submit that request to a queue. When enough resources are free, the scheduler **allocates** them to you, runs your program, and reclaims them when you finish or hit your time limit. The dominant scheduler in HPC and AI is **SLURM** (Simple Linux Utility for Resource Management). When you read a training paper that says "we trained on the cluster," there is a SLURM script behind it that no one ever shows you. We are going to show you.

The key mental shift is this: on one box you launch processes; on a cluster you **submit jobs**. A **job** is a unit of work the scheduler tracks — it has an ID, a state (pending, running, completed, failed), an owner, and a resource reservation. You do not run your program; you ask SLURM to run it on your behalf, somewhere, when it can. The login node you SSH into is deliberately weak — often no GPUs at all — precisely so that you cannot accidentally hog compute there. Run a heavy job on the login node and an admin will email you within the hour.

Three pieces of vocabulary, defined once and used the whole post:

- **SLURM** — the cluster's scheduler and resource manager. It runs a controller daemon (`slurmctld`) on a management node and an agent (`slurmd`) on every compute node. You talk to it through a handful of commands.
- **Partition** — a named group of nodes with a shared policy, like a queue lane. A cluster typically has partitions like `gpu` (the H100 nodes, with a long time limit), `debug` (a few nodes with a 30-minute limit for quick tests), and `cpu` (no-GPU nodes for data prep). You submit *to* a partition. Picking the wrong one is the single most common reason a job sits pending forever.
- **Node** — one physical server. On a GPU cluster, one node is typically 8 GPUs connected internally by **NVLink** (very fast, ~900 GB/s on H100), and nodes connect to each other by InfiniBand (fast, ~400 Gb/s = 50 GB/s, but ~18× slower than intra-node NVLink). That ratio — fast inside a node, slower between nodes — shapes every decision about how you split a model, which is the heart of [parallelism strategies](/blog/machine-learning/high-performance-computing/parallelism-strategies-data-tensor-pipeline-and-expert).

Here is the number that motivates everything: a single H100 SXM delivers roughly 989 bf16 TFLOP/s of dense matmul. Eight of them is ~7.9 PFLOP/s in one node; sixty-four is ~63 PFLOP/s. At a few dollars per GPU-hour, a 64-GPU run costs on the order of \$130–\$200 per hour all-in. If your effective utilization is 45% instead of 90% because the data loader is starving the GPUs, you are not just slow — you are **paying double**. That is the lens for the whole post: every section is about turning idle GPU-seconds back into useful work.

One more thing about *why* the scheduler exists, because it explains behavior that otherwise looks arbitrary. A shared cluster has to balance two things that pull in opposite directions: **utilization** (keep every GPU busy) and **fairness** (no single team monopolizes the facility). SLURM does this with a priority system, usually fair-share: each account has a target slice of the cluster, and an account that has used *less* than its share recently gets *higher* priority on its pending jobs. So if your job is stuck behind a teammate's, it is not random — it is the scheduler enforcing that your group already consumed its fair slice this week. This also explains **backfill**: when a big job is waiting for, say, 16 nodes to free up, the scheduler will slip smaller, shorter jobs into the gaps in the meantime, as long as they finish before the big job's nodes are ready. The practical consequence is concrete and worth knowing: **asking for less time gets you scheduled sooner.** A job that declares `--time=02:00:00` is far more backfillable than one that declares `--time=24:00:00`, because the scheduler can fit a 2-hour job into many more gaps. Pad your time limit for safety, but do not pad it 10×, or you will sit in the queue while shorter jobs jump ahead of you. The scheduler is not your adversary; it is an optimizer with constraints you can play to.

### The five SLURM commands you will actually type

You can have a long career on a cluster knowing five commands. Here they are, with what they do:

```bash
# Submit a batch script to the scheduler. Returns a job ID, then exits.
sbatch train.slurm

# Show the queue: your jobs, their state (PENDING/RUNNING), and why.
squeue --me

# Cancel a job by its ID.
scancel 1837245

# Show what resources exist and which partitions are idle or full.
sinfo

# After a job finishes, show its accounting: runtime, memory, exit code.
sacct -j 1837245 --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

The two stars are `sbatch` and `srun`. **`sbatch`** (batch submit) hands a *script* to the queue and returns immediately — your terminal is free, the job runs whenever resources free up, and its output goes to a file. This is how every real training run is launched: you submit and walk away. **`srun`** (run) launches a *parallel program* across the allocated nodes — it is the thing that actually starts N copies of your process, one per task, on the machines SLURM gave you. Inside a batch script, `sbatch` reserves the nodes and `srun` puts your processes on them. You will see both in the script below. The mental model: `sbatch` is "reserve the room," `srun` is "send everyone in."

![Timeline of a SLURM job moving through submit, pending in the queue, allocate four nodes, srun launches thirty-two ranks, running, periodic checkpoint, and completed or requeue](/imgs/blogs/running-on-a-cluster-slurm-multi-node-launch-and-data-pipelines-2.png)

The timeline above is the life of one job, and it is worth internalizing because most cluster frustration comes from not knowing which stage you are stuck in. You `sbatch` a script. The job enters **PENDING** and waits in the queue — this can be seconds on an idle cluster or hours on a busy one, and `squeue` tells you the reason code (`Resources` means "waiting for nodes," `Priority` means "other jobs are ahead of you," `QOSMaxJobsPerUserLimit` means "you already have too many running"). When enough nodes are free, SLURM **allocates** them and the job flips to **RUNNING**; your batch script begins executing on the first allocated node. Inside it, `srun` launches your 32 ranks across the 4 nodes. The job runs, periodically checkpointing (the next-to-last stage, and the subject of the final section), until it either **COMPLETES** or hits its time limit. If it hits the limit mid-training, a well-built script **requeues** itself — resubmits from the last checkpoint — so a 5-day run survives a 24-hour-per-job limit. That requeue loop is not optional at scale; it is the difference between a run that finishes and one that dies at hour 23.

#### Worked example: reading why your job is stuck

You submit, run `squeue --me`, and see your job in state `PD` (pending) with reason `(Resources)`. What does that tell you, concretely? It means the scheduler has accepted your request as valid — your partition exists, your resource ask is satisfiable in principle — but right now there are not 4 idle nodes in the `gpu` partition to give you. You wait. Now suppose instead the reason is `(PartitionConfig)` or `(QOSMaxGRESPerUser)`: that means your *request itself* is the problem — you asked for more GPUs than your account is allowed, or named a partition that cannot satisfy the GPU type. The fix is editing the script, not waiting. And if you see `(launch failed requeued held)`, your program crashed on startup and SLURM gave up — check the output file. Learning to read that one column in `squeue` saves more wall-clock time than any kernel optimization, because a job that is pending for the wrong reason will pend *forever*, and you will not get an error, just silence.

## The batch script: turning a resource request into a launched job

The `sbatch` script is the contract between you and the cluster. The top of the file is a block of `#SBATCH` directives — special comments SLURM reads to size your reservation — and the bottom is ordinary shell that runs once the nodes are yours. Every directive maps onto a physical thing the scheduler must reserve.

![Matrix mapping each SBATCH directive to the concrete hardware or time resource it reserves, showing nodes to servers, gpus-per-node to GPUs, ntasks-per-node to ranks, cpus-per-task to loader cores, and time to wall-clock limit](/imgs/blogs/running-on-a-cluster-slurm-multi-node-launch-and-data-pipelines-8.png)

The matrix above is the decoder ring. Read it once and the script below will be obvious rather than mysterious. `--nodes=4` reserves four physical servers from the partition. `--gpus-per-node=8` says each server must give you all 8 of its GPUs (so this is a full-node job — no sharing). `--ntasks-per-node=8` tells SLURM to start 8 tasks (processes) on each node — one per GPU, which is exactly what `torchrun` wants, one rank per GPU. `--cpus-per-task=12` reserves 12 CPU cores per task, and those cores are what your data-loader workers run on — get this number wrong and your loader has nowhere to run, which we will return to with hard numbers. `--time=24:00:00` is the wall-clock limit; at 24 hours SLURM kills the job whether it is done or not, which is why checkpointing and requeue exist.

Here is a complete, runnable multi-node training script. This is the real thing, not a sketch — copy it, change the paths, and it launches.

```bash
#!/bin/bash
#SBATCH --job-name=transformer-7b
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --signal=B:USR1@120
#SBATCH --requeue

set -euo pipefail

# --- Pick the rendezvous host: the first node in the allocation ---
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
head_node=$(echo "$nodes" | head -n 1)
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
export MASTER_ADDR="$head_node_ip"
export MASTER_PORT=29500

# --- NCCL: name the fast fabric, turn off slow fallbacks ---
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME=ib0
export NCCL_ASYNC_ERROR_HANDLING=1

echo "Head node: $head_node ($head_node_ip), nodes: $SLURM_NNODES"

# --- Launch torchrun once per node via srun; each spawns 8 ranks ---
srun --cpu-bind=none bash -c '
torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=8 \
  --node_rank=$SLURM_NODEID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  --rdzv_id=$SLURM_JOB_ID \
  train.py --config configs/7b.yaml --resume auto
'
```

Let us read the parts that are not obvious. `--mem=0` means "give me all the memory on each node" — you want it, and asking for a specific number just risks under-asking. `--output=logs/%x-%j.out` writes stdout to a file named with the job name (`%x`) and job ID (`%j`); on a cluster you do not watch a terminal, you `tail -f` a log file. `--signal=B:USR1@120` tells SLURM to send your script a `USR1` signal 120 seconds *before* the time limit, so a signal handler in your training code can save a final checkpoint and requeue before being killed — this is the mechanism behind that "or requeue" stage in the timeline. `--requeue` makes the job eligible to be resubmitted automatically.

The shell at the bottom does three things. First it figures out the **rendezvous address** — the IP of the first node in the allocation, which becomes `MASTER_ADDR`, the meeting point all 32 processes will dial to find each other. Second it sets the **NCCL** environment so the GPUs talk over the fast InfiniBand fabric instead of falling back to slow Ethernet (we will dwell on this — it is a classic 3× slowdown if you skip it). Third it runs `srun ... torchrun ...`: `srun` puts *one* `torchrun` per node (because the outer task count is per-node here, and torchrun spawns the 8 GPU processes itself), and each `torchrun` is told its `--node_rank` from SLURM's `$SLURM_NODEID` so they coordinate. The interplay — `srun` for "one launcher per node," `torchrun` for "8 ranks per launcher" — is the part everyone gets wrong the first time. It is the seam between SLURM and PyTorch.

### Job arrays: when you want a hundred runs, not one

One more SLURM feature pays for itself constantly: **job arrays.** Suppose you are not training one big model but sweeping 50 hyperparameter configs, or processing 200 shards of a dataset through a preprocessing step. You do not submit 50 scripts. You submit one with `--array=0-49`, and SLURM launches 50 copies, each with a different `$SLURM_ARRAY_TASK_ID` (0 through 49) that your script uses to pick its config or its shard. The scheduler packs them onto whatever nodes free up, in parallel, respecting your fair-share limit.

```bash
#!/bin/bash
#SBATCH --job-name=preprocess-shards
#SBATCH --partition=cpu
#SBATCH --array=0-199%32      # 200 tasks, at most 32 running at once
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/prep-%A_%a.out

set -euo pipefail
shard_id=$SLURM_ARRAY_TASK_ID
python make_webdataset.py \
  --input  /raw/images/part-${shard_id}.parquet \
  --output /lustre/shards/train-$(printf "%06d" $shard_id).tar
```

The `%32` throttle is the part to remember: `--array=0-199%32` means 200 total tasks but never more than 32 running simultaneously, so you do not flood the cluster and anger the other tenants. Job arrays are how you turn "build the dataset" from a day-long serial loop into a 30-minute parallel burst, and they are how the sharded WebDataset we use later actually gets built. The `%A` and `%a` in the output filename are the array job ID and the task index, so each task gets its own log.

## Multi-node launch: how 32 processes find each other

You have nodes. Now you need 32 Python processes — 8 on each of 4 nodes — to stop being 32 strangers and become one training job that shares gradients every step. This is the part that feels like magic the first time it works and like a curse every time it does not. The mechanism is called **rendezvous**, and once you understand it, multi-node launch stops being scary.

Define it plainly. **Rendezvous** is the process by which all the worker processes of a distributed job discover one another, agree on how many there are (the `world_size`), and assign each a unique global rank (0 to 31). They cannot do this without a meeting point, because process 17 on node 2 has no idea process 3 on node 0 even exists — they are separate OS processes on separate machines. So they all dial the same address and port — the **rendezvous endpoint**, which lives on node 0 — and a small coordination service there (the "store") collects everyone, waits until all 32 have checked in, and then broadcasts the full roster: you are rank 17, there are 32 of you, here is everyone's address. After that, NCCL (the GPU communication library, covered in depth in [collective communication and NCCL](/blog/machine-learning/high-performance-computing/collective-communication-and-nccl-all-reduce-from-scratch)) uses that roster to wire the GPUs into a ring for all-reduce.

![Diagram of multi-node torchrun rendezvous where four node ranks each dial the same c10d endpoint on node zero, form a world size of thirty-two, and arrive at an NCCL ring ready for all-reduce](/imgs/blogs/running-on-a-cluster-slurm-multi-node-launch-and-data-pipelines-3.png)

The figure shows it cleanly. The **rendezvous endpoint** sits on node 0 at a fixed port (29500 in our script). Each node's group of ranks — `node_rank 0` owns ranks 0–7, `node_rank 1` owns 8–15, and so on — **joins** the endpoint. Once all four node-groups have checked in, they pass a **barrier** (a synchronization point where everyone waits for everyone) and emerge with a single `world_size 32` process group. From there NCCL forms its ring and the job is ready to all-reduce gradients. Every one of those ranks runs the identical `train.py`; the only thing that differs is the rank number each one is handed, which it reads from the environment.

Here is the `torchrun` invocation again, isolated, because the flags are the whole ballgame:

```bash
torchrun \
  --nnodes=4 \
  --nproc_per_node=8 \
  --node_rank=$SLURM_NODEID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:29500 \
  --rdzv_id=$SLURM_JOB_ID \
  train.py --config configs/7b.yaml
```

- `--nnodes=4` — how many nodes total. With `--nproc_per_node=8` this means 32 ranks.
- `--nproc_per_node=8` — how many processes per node, one per GPU. torchrun spawns these for you and assigns each its `LOCAL_RANK` (0–7), which your code uses to pick its GPU (`torch.cuda.set_device(local_rank)`).
- `--node_rank` — which node this is (0–3), fed from SLURM's `$SLURM_NODEID` so each node launcher knows its place. Get this wrong — say, two nodes both think they are node 0 — and rendezvous hangs forever because the roster never completes.
- `--rdzv_backend=c10d` — the rendezvous backend. `c10d` is PyTorch's built-in store, the modern default; it needs no external service (older setups used `etcd`).
- `--rdzv_endpoint=$MASTER_ADDR:29500` — the meeting point. Every node uses the *same* address (node 0's IP) and the same port. This is the single most important value to get right.
- `--rdzv_id=$SLURM_JOB_ID` — a unique name for this rendezvous, so two different jobs on the same cluster do not collide on the store. Using the SLURM job ID guarantees uniqueness.

Inside `train.py`, the standard initialization reads what torchrun set in the environment and forms the process group:

```python
import os
import torch
import torch.distributed as dist

def setup_distributed():
    # torchrun populates these env vars in every process.
    rank       = int(os.environ["RANK"])         # global, 0..31
    local_rank = int(os.environ["LOCAL_RANK"])   # within node, 0..7
    world_size = int(os.environ["WORLD_SIZE"])   # 32

    torch.cuda.set_device(local_rank)            # this process owns one GPU
    dist.init_process_group(
        backend="nccl",                          # GPUs talk via NCCL
        init_method="env://",                    # read MASTER_ADDR/PORT from env
    )
    if rank == 0:
        print(f"World ready: {world_size} ranks across "
              f"{world_size // torch.cuda.device_count()} nodes")
    return rank, local_rank, world_size

rank, local_rank, world_size = setup_distributed()
device = torch.device(f"cuda:{local_rank}")
# ... build model, wrap in DDP or FSDP on `device`, train ...
dist.destroy_process_group()
```

The line `init_method="env://"` is the quiet hero: it tells PyTorch to read `MASTER_ADDR`, `MASTER_PORT`, `RANK`, and `WORLD_SIZE` from the environment that torchrun and our SLURM script already set. You never hardcode an IP. The whole chain — SLURM picks the head node, the script exports its IP as `MASTER_ADDR`, torchrun passes it as the rendezvous endpoint, PyTorch reads it via `env://` — is how an address discovered at runtime flows down to every one of the 32 processes.

### When rendezvous hangs: the three classic failures

Rendezvous either works in two seconds or hangs forever, and the difference is almost always one of three mistakes you can diagnose by glancing at the figure above. The barrier in the middle of that diagram only releases when *all* node-groups have checked in, so any rank that cannot reach the endpoint silently holds the entire job in the join state, producing no error — just a process group that never forms while your allocation burns. The first failure is a **wrong or unreachable `MASTER_ADDR`**: node 0 advertises an IP that the other nodes cannot route to (for example, you grabbed the management-network hostname instead of the InfiniBand IP). Ranks 8–31 dial an address that does not answer, and they wait. The fix in our SLURM script — `srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address` — deliberately asks node 0 for an IP that is reachable from the compute fabric, not its login alias. The second failure is a **duplicated `--node_rank`**: if two node launchers both believe they are node 0 (a copy-paste bug, or `$SLURM_NODEID` not being read inside the `srun` subshell), the roster expects 32 distinct ranks but only ever sees ranks from three distinct nodes, so the count never reaches 32 and the barrier never releases. The third is a **firewalled or already-bound port**: port 29500 is blocked between nodes, or a previous crashed job left a process holding it, so the store cannot accept connections. Changing `MASTER_PORT` or keying `--rdzv_id` to `$SLURM_JOB_ID` (so a stale rendezvous from a dead job cannot collide) clears it.

The practical diagnostic loop is short. Add `export TORCH_DISTRIBUTED_DEBUG=DETAIL` and `export NCCL_DEBUG=INFO` to the script, submit, and `tail -f` the log. If you see every rank print "joined rendezvous" but the job never reaches your first training step, the count is wrong — suspect a duplicated `node_rank` or a missing node. If you see fewer than `WORLD_SIZE` ranks ever print anything, some node cannot reach the endpoint — suspect `MASTER_ADDR` or the port. If rendezvous completes but the first all-reduce hangs, rendezvous succeeded and the problem moved to NCCL — which is the next section. Reading *where* in the timeline the hang occurs tells you which of the three it is, and that single distinction saves an afternoon of blind flag-twiddling. A useful habit: set a generous but finite rendezvous timeout (torchrun's default is ten minutes) so a misconfigured job dies and requeues instead of holding 32 GPUs hostage until the wall-clock limit. A job that fails in ten minutes and tells you why is worth far more than one that hangs for twenty-four hours and tells you nothing.

There is one more subtlety worth naming because it bites teams moving from a single node to many: on one node, `torchrun --standalone --nproc_per_node=8` works without any of this rendezvous machinery, because all 8 processes share `localhost` and the store lives in the same machine. The moment you cross to a second node, `localhost` is no longer a shared meeting point — node 1's `localhost` is a different machine than node 0's — and you *must* supply a real `--rdzv_endpoint` reachable from both. Engineers who never set `MASTER_ADDR` on a single node are often surprised that multi-node "suddenly needs networking," when in truth the single-node case was hiding the rendezvous behind `localhost` all along. The cluster simply makes the meeting point explicit.

### The NCCL environment: the 3× slowdown hiding in plain sight

Here is a failure mode that has burned every multi-node team at least once, and it produces no error — just a job that is mysteriously 3× slower than it should be. NCCL, the library that moves gradients between GPUs, has to choose which network interface to use. A GPU node has several: the fast **InfiniBand** adapters (the ones rated 400 Gb/s) and a slow management **Ethernet** interface (maybe 10 or 25 Gb/s) used for SSH and logging. If you do not tell NCCL which to use, it may pick the slow one. Your all-reduce, which should take milliseconds over InfiniBand, now takes 3–4× longer over Ethernet, and since all-reduce happens every single step, your whole run crawls — at full GPU utilization, so the profiler looks "busy" and you are baffled.

The fix is the environment block from the SLURM script. Name the fast fabric explicitly:

```bash
export NCCL_DEBUG=INFO            # print which transport NCCL chose (read the log!)
export NCCL_IB_HCA=mlx5           # use the Mellanox InfiniBand host adapters
export NCCL_SOCKET_IFNAME=ib0     # for any TCP fallback, use the IB interface, not eth0
export NCCL_ASYNC_ERROR_HANDLING=1  # fail fast on a dead rank instead of hanging
export MASTER_ADDR=$head_node_ip  # rendezvous host (set earlier from SLURM)
export MASTER_PORT=29500
```

`NCCL_DEBUG=INFO` is non-negotiable for the first run on any new cluster: it prints a line like `NCCL INFO NET/IB : Using [0]mlx5_0` if it found InfiniBand, or `NCCL INFO NET/Socket` if it fell back to slow TCP. Read that line. If it says Socket, your interconnect choice is wrong and you are about to waste a lot of money. `NCCL_ASYNC_ERROR_HANDLING=1` is the other one that matters: without it, if one rank dies, the other 31 hang at the next all-reduce *forever*, holding the whole allocation hostage until the time limit kills them. With it, the job fails fast and your `--requeue` restarts it. The interconnect choices here are exactly the [NVLink, InfiniBand, and RDMA](/blog/machine-learning/high-performance-computing/interconnects-nvlink-nvswitch-infiniband-and-rdma) tradeoffs made concrete: name the fast path or pay for the slow one.

#### Worked example: scaling efficiency from 8 to 64 GPUs

You have a 7B-parameter Transformer training at, say, 1.0 (normalized) throughput on a single 8-GPU node. You scale to 8 nodes (64 GPUs) hoping for 8× the throughput. What do you actually get, and how do you read it?

Define **scaling efficiency** as the fraction of perfect linear scaling you achieve: efficiency $= T_N / (N \cdot T_1)$, where $T_N$ is throughput at $N$ nodes and $T_1$ at one node. Perfect is 1.0 (100%). In practice, going from 1 node to 8 nodes adds inter-node all-reduce over InfiniBand, which is ~18× slower than the intra-node NVLink you had for free. With a well-tuned setup — gradient bucketing, overlap of communication with backward compute, the fast fabric named — a data-parallel 7B model typically lands around 88–92% efficiency at 64 GPUs, so 8 nodes gives roughly $8 \times 0.90 = 7.2\times$ the throughput, not 8×. That 10% is the communication tax, and it is the price of the [collective communication](/blog/machine-learning/high-performance-computing/collective-communication-and-nccl-all-reduce-from-scratch) you cannot avoid in data parallelism. Now the diagnostic: if instead you measure 60% efficiency (4.8× instead of 7.2×), something is wrong, and it is almost always one of three things — NCCL fell back to Ethernet, the data loader cannot feed 64 GPUs (the next section), or your global batch grew so large that you are no longer compute-bound per step. Knowing the ~90% number is what lets you tell "expected communication tax" from "a bug costing me \$50/hour." Treat these efficiency figures as approximate and model-dependent; always measure your own.

## Feeding the beast: the data loading bottleneck

Now the part that humbles everyone. You have 64 H100s wired into a ring, rendezvous works, gradients flow. You start training and the GPUs sit at **40% utilization.** Not because the model is small. Not because the network is slow. Because the **data loader cannot deliver samples fast enough**, and a GPU with no batch to compute on does the only thing it can: it waits. This is the single most common, most expensive, and most overlooked bottleneck in large-scale training, and on a cluster it is worse than on your laptop because your data now lives on a filesystem across the network.

Let us make the requirement quantitative, because this is where intuition fails and arithmetic saves you. There is a simple throughput law governing whether your GPU starves.

### The data-loading throughput requirement

Your GPU consumes batches at some rate. Say each training step takes 42 ms of pure compute and processes a batch of 256 images. Then the GPU *demands* $256 / 0.042 = 6{,}095$ images per second. Per GPU. Across 64 GPUs that is ~390,000 images/second the data pipeline must supply, sustained, or somewhere a GPU is idle.

The loader supplies samples using $W$ parallel **worker** processes (CPU processes that read, decode, and augment data in the background). If decoding and augmenting one image takes $t_s$ seconds of CPU time, then $W$ workers produce $W / t_s$ images per second. The starvation condition is brutally simple:

$$\frac{W}{t_s} \;<\; \frac{B}{t_{\text{step}}} \quad\Longrightarrow\quad \text{GPU starves.}$$

The loader must out-produce the step. If one image takes $t_s = 5$ ms of CPU work (a JPEG decode plus a resize plus a random crop is easily that), one worker makes 200 images/s. To feed a single GPU demanding 6,095 images/s you need $6095 / 200 \approx 31$ workers — for *one* GPU. That is impossible; you do not have 31 CPU cores per GPU. So either you make $t_s$ smaller (cheaper decode, pre-resized data), or you make $t_{\text{step}}$ larger (it already is, with a real model), or you accept you must engineer the pipeline carefully. This is why "just add `num_workers`" is not a strategy — the math tells you exactly how many you need and whether it is even feasible.

![Stack diagram of the data pipeline showing a sample traveling from a parallel filesystem shard through worker decode and batch collate into a pinned host buffer, then across the copy engine to the GPU step](/imgs/blogs/running-on-a-cluster-slurm-multi-node-launch-and-data-pipelines-4.png)

The stack above is every stop a single sample makes between disk and the GPU, and each stop is a place it can get stuck. It starts as bytes in a **shard** on the parallel filesystem (aggregate ~200 GB/s across the cluster, but only a slice of that per node). A **worker** process reads it, decodes the JPEG, and applies augmentations — this is the CPU-bound step with cost $t_s$. The workers **collate** decoded samples into a batch tensor. That batch lands in a **pinned host buffer** (we will define pinned memory next), and a GPU **copy engine** moves it host-to-device. Only then does the **GPU step** consume it. If any stage is slower than the GPU's demand rate, the GPU waits. The art of feeding the beast is making sure every stage out-produces the step, and the figure tells you exactly where to look when it does not.

### Little's law and the prefetch queue

There is one more piece of science that explains *why* prefetching works, and it is **Little's law** — the most useful queueing result in all of systems engineering. It states that for any stable queue, the average number of items in the system equals the arrival rate times the average time each item spends in the system: $L = \lambda \cdot W$. Apply it to the data pipeline. You want the GPU to *never* find an empty queue when it asks for the next batch. The queue between the loader and the GPU is the **prefetch buffer**. If the GPU pulls a batch every $t_{\text{step}}$ = 42 ms ($\lambda \approx 24$ batches/s) and a batch takes $W$ = 200 ms to travel from disk to ready (storage read + decode + collate + copy), then to keep the queue non-empty you need on average $L = \lambda W = 24 \times 0.2 \approx 5$ batches *in flight* at all times. That is the `prefetch_factor` you must set. Set it to 2 when the math says 5 and the queue drains, the GPU stalls, and you are back at 40%. Little's law is how you size the buffer instead of guessing.

This is the whole reason `DataLoader` has `num_workers` *and* `prefetch_factor`: workers control the production rate ($W / t_s$), prefetch controls the buffer depth ($L$), and you need both right. The PyTorch `DataLoader` configured to actually feed a GPU looks like this:

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=256,
    num_workers=12,            # 12 CPU worker processes per GPU process
    pin_memory=True,           # collate into page-locked host memory -> fast H2D
    prefetch_factor=4,         # each worker stages 4 batches ahead (buffer depth)
    persistent_workers=True,   # do NOT respawn workers every epoch (huge win)
    drop_last=True,            # equal batch sizes across ranks (DDP needs this)
    pin_memory_device=f"cuda:{local_rank}",
)
```

Three of these flags are doing real work and deserve a sentence each.

**`pin_memory=True`** uses **pinned memory** — page-locked host RAM that the operating system promises never to swap out. Define it: normal host memory can be paged to disk by the OS, and a GPU's copy engine cannot DMA directly from pageable memory, so a non-pinned transfer secretly copies through a staging buffer first. Pinned memory lets the copy engine read host RAM directly, and crucially it enables **asynchronous** host-to-device copies that overlap with GPU compute. On an A100/H100 over PCIe Gen4 (~25 GB/s effective), pinning a transfer is typically the difference between the copy hiding behind compute and the copy stalling the step. It costs a little host RAM; it is almost always worth it.

**`persistent_workers=True`** is a free 5–15% that almost everyone leaves on the table. By default, `DataLoader` tears down all worker processes at the end of each epoch and forks them again at the start of the next — and forking 12 processes that each re-open the dataset, re-seed RNG, and re-establish filesystem handles costs real seconds, multiplied by every epoch. Persistent workers keep them alive across epochs. On a run with many short epochs this alone can move utilization several points.

**`prefetch_factor=4`** is the Little's-law buffer depth: each worker keeps 4 batches staged ahead, so $12 \times 4 = 48$ batches are buffered, far more than the ~5 the law demanded, which gives slack for jitter (a slow storage read, a GC pause). The cost is host RAM (48 batches resident); set it as high as your RAM allows once the queue is reliably full.

There is a hard ceiling on `num_workers` that the cluster, not PyTorch, sets: the CPU cores you reserved. Recall the SLURM directive `--cpus-per-task=12`. That task is your *one* GPU process, and its 12 cores are the pool your 12 workers run on. Ask for `num_workers=24` and you have not doubled your throughput — you have oversubscribed 12 cores with 24 processes that now fight each other for CPU time, context-switch constantly, and often run *slower* than 12 workers would. The correct procedure is to size `--cpus-per-task` from the data-loading math first, then set `num_workers` to match. If the throughput inequality says you need 12 workers per GPU to feed the step, you request `--cpus-per-task=12`, and on an 8-GPU node with `--ntasks-per-node=8` that is $8 \times 12 = 96$ cores — which is why GPU nodes ship with 96 or 128 CPU cores in the first place. The cores exist *to feed the GPUs*. When you cannot get enough cores per GPU to satisfy the inequality, the answer is not more workers; it is making $t_s$ smaller (pre-resize images offline so the worker skips the expensive resize, decode to a cheaper format, or move augmentation onto the GPU itself with a library like DALI or Kornia). You cannot brute-force past a core budget with worker count alone.

A word on a subtle interaction that surprises people: more workers does not always mean higher throughput, even with cores to spare, because the workers compete for the *same* parallel-filesystem bandwidth and metadata server. Eight workers each reading a different shard can saturate the node's slice of Lustre; adding eight more does not conjure more network bandwidth, it just adds contention. This is why the worker count and the storage strategy are coupled — fixing the storage layout (sharding, local cache) often does more than adding workers, and the two must be tuned together rather than in isolation.

#### Worked example: how many workers does one GPU need?

Make the inequality concrete with a real-feeling case. You train a 7B Transformer on tokenized text, not images, so the per-sample CPU cost is small — say $t_s = 0.4$ ms per sample to read and tensorize a sequence. The step is large: $t_{\text{step}} = 50$ ms for a batch of $B = 16$ sequences (long contexts, big model). The GPU demands $16 / 0.050 = 320$ sequences/s. One worker supplies $1 / 0.0004 = 2{,}500$ sequences/s — already eight times the demand. So a *text* loader needs only one or two workers, and people who copy a 12-worker config from a vision tutorial waste cores and host RAM for nothing. Now flip to vision: 4-megapixel JPEGs with a decode-plus-resize-plus-augment cost of $t_s = 6$ ms, a step of $t_{\text{step}} = 40$ ms over $B = 256$ images. Demand is $256 / 0.040 = 6{,}400$ images/s; one worker supplies $1 / 0.006 \approx 167$ images/s; you need $6400 / 167 \approx 38$ workers to feed one GPU — which is infeasible on a 12-core budget, so you *must* shrink $t_s$ (pre-resized shards drop the resize, halving $t_s$) and accept that even then you are at the edge. The same inequality, two wildly different answers: this is why "set `num_workers` to the number of cores" is folk wisdom that is right by accident for vision and wrong for text. Compute the demand, compute the supply, and let the arithmetic — not a tutorial — pick the number.

### Sharded datasets: why millions of small files kill a parallel filesystem

Here is the cluster-specific twist that does not exist on your laptop. Your training set is, say, 10 million images. On your laptop they sit on a local SSD and random access is cheap. On a **parallel filesystem** (Lustre, GPFS — a filesystem whose data is striped across many storage servers so its *aggregate* bandwidth is enormous, ~200 GB/s, but whose *metadata* server is a single point of contention), opening 10 million tiny files is death. Every `open()` is a metadata operation, and a parallel filesystem's metadata server handles maybe tens of thousands of opens per second total, across the whole cluster. With 64 GPUs each spawning 12 workers, you have 768 processes all hammering the metadata server with `open()` calls for 200 KB files. The metadata server melts, every read stalls, and your GPUs starve — not because bandwidth is short but because you are doing millions of *tiny random reads* on a system built for *large sequential reads*.

The fix is **sharded datasets**: instead of 10 million files, you pack them into a few thousand **shards** — large `.tar` archives of ~1 GB each, each holding ~5,000 samples in sequence. Now a worker opens *one* file and streams 5,000 samples out of it sequentially, which is exactly what a parallel filesystem is fast at. You traded 10 million metadata ops for ~2,000. This is the **WebDataset** format, and it is the standard way large training jobs read data on a cluster. Reading a shard looks like this:

```python
import webdataset as wds

def make_loader(shard_urls, batch_size, num_workers, world_size, rank):
    dataset = (
        wds.WebDataset(shard_urls, shardshuffle=True, nodesplitter=wds.split_by_node)
        .shuffle(2000)                      # in-memory shuffle buffer (samples)
        .decode("pil")                      # decode jpg/png to PIL images
        .to_tuple("jpg", "cls")             # (image, label) per sample
        .map_tuple(train_transform, lambda x: x)
        .batched(batch_size, partial=False)
    )
    loader = wds.WebLoader(
        dataset, batch_size=None,           # dataset already batched
        num_workers=num_workers, pin_memory=True,
        prefetch_factor=4, persistent_workers=True,
    )
    return loader

# shards named train-000000.tar .. train-001999.tar
urls = "pipe:cat /lustre/shards/train-{000000..001999}.tar"
loader = make_loader(urls, 256, 12, world_size, rank)
```

Two details earn their keep. `nodesplitter=wds.split_by_node` makes each of the 4 nodes read a *disjoint* subset of shards, so 768 workers are not all reading the same files — they spread the load across the filesystem's storage servers, which is how you actually approach that 200 GB/s aggregate. And `.shuffle(2000)` does an *in-memory* shuffle of a 2,000-sample window: you cannot randomly seek inside a `.tar` cheaply, so WebDataset approximates a global shuffle by shuffling shards (which shard you read next) plus shuffling a buffer of samples in RAM. It is not a perfect global shuffle, but for large datasets it is statistically indistinguishable and it preserves the sequential-read pattern the filesystem needs.

### The before and after that pays for the whole post

This is the result that justifies the section. A real image-classification run on a single DGX node (8× A100 80GB) reading from Lustre, before and after fixing the loader:

![Before and after comparison showing a loader-bound GPU at forty percent utilization with two workers becoming a fed GPU at ninety-five percent utilization with twelve workers, prefetch four, and an NVMe cache](/imgs/blogs/running-on-a-cluster-slurm-multi-node-launch-and-data-pipelines-5.png)

The before-and-after figure tells the story in numbers. **Before:** 2 workers, reading 200 KB JPEGs as individual files from Lustre, no pinning, default prefetch. The loader produced ~1,100 images/s; the 8 GPUs demanded ~3,400; the GPUs sat **40% utilized**, idle 60% of the time, and each step took 95 ms of which 55 ms was the GPU waiting for data. **After:** 12 workers, WebDataset shards instead of loose files, `pin_memory=True`, `prefetch_factor=4`, and a one-time copy of the shards to each node's **local NVMe** drive (3–7 GB/s, ~100 µs latency — versus the network round-trip to Lustre) used as a cache for the second epoch onward. The loader now produces ~3,400 images/s, matching demand; the GPUs run at **95% utilization**, step time drops to 42 ms with only ~2 ms of stall. Same hardware, same model, **2.3× more throughput** — and the only thing that changed was the data pipeline. That is the unglamorous 30% made concrete. The GPUs were never the problem; we just stopped starving them. If you want to see how to *find* a stall like this in the first place, [profiling GPU workloads](/blog/machine-learning/high-performance-computing/profiling-gpu-workloads-finding-the-real-bottleneck) shows you the trace where the 55 ms gap stares back at you.

| Knob | Before | After | Why it mattered |
| --- | --- | --- | --- |
| `num_workers` | 2 | 12 | More CPU producers; loader out-produces the step |
| Data format | loose JPEGs | WebDataset shards | Sequential reads, ~2,000× fewer metadata ops |
| `pin_memory` | False | True | Async H2D copy overlaps with compute |
| `prefetch_factor` | 2 | 4 | Deeper queue, survives storage jitter (Little's law) |
| `persistent_workers` | False | True | No per-epoch worker respawn |
| Local NVMe cache | none | shards cached | Epoch 2+ reads at 5 GB/s, not over the network |
| **GPU utilization** | **40%** | **95%** | The whole point |

## Storage: where your data lives and why it matters

We have leaned on "the parallel filesystem" and "local NVMe" without laying out the full storage picture, and on a cluster the storage hierarchy is as important as the memory hierarchy is on a single GPU. There are three tiers, and using each for what it is good at is the difference between a pipeline that flows and one that stalls.

![Matrix comparing storage tiers showing the parallel filesystem with high bandwidth for active shards, object storage with high latency for archives, and local NVMe with lowest latency for the hot epoch cache](/imgs/blogs/running-on-a-cluster-slurm-multi-node-launch-and-data-pipelines-6.png)

The matrix lays out the trade. **Parallel filesystem** (Lustre, GPFS) is the cluster's shared workhorse: 100–300 GB/s aggregate bandwidth, petabytes of capacity, visible from every node — but millisecond latency and that fragile metadata server, so it loves big sequential reads and hates millions of tiny files. This is where your **active dataset shards** live. **Object storage** (S3, GCS) is the cheap, effectively infinite archive: exabytes at low cost, but 10–100 ms latency per object and accessed over HTTP, so it is your **source of truth** and cold tier — you stage data *from* S3 *to* the parallel filesystem before a run, you do not read training batches directly from it at full speed. **Local NVMe** is the secret weapon: each node has 1–15 TB of local SSD delivering 3–7 GB/s per drive at under 100 µs latency, completely private to the node and not shared across the network. If your dataset (or a working subset) fits on local NVMe, copy it there once at job start and read from it for the rest of the run — every read becomes local, the parallel filesystem stops being a bottleneck, and you get the 95%-utilization story from the last section.

The decisive pattern, stated as a rule: **archive in object storage, run from the parallel filesystem, cache the hot set on local NVMe.** A common job-start snippet does exactly this:

```bash
# Stage shards from S3 (source of truth) to fast local NVMe, in parallel,
# only on local_rank 0 of each node so 8 GPUs don't copy 8 times.
LOCAL_CACHE=/scratch/$SLURM_JOB_ID/shards   # NVMe scratch dir per node
mkdir -p "$LOCAL_CACHE"

if [ "$SLURM_LOCALID" = "0" ]; then
  echo "Node $SLURM_NODEID staging shards to local NVMe..."
  # Each node grabs its disjoint slice of shards (split_by_node logic mirrored)
  aws s3 cp --recursive --quiet \
    "s3://my-bucket/imagenet-wds/node-$SLURM_NODEID/" "$LOCAL_CACHE/"
fi
# Barrier: wait for the staging rank before any rank starts reading
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 true
```

The `if [ "$SLURM_LOCALID" = "0" ]` guard is the detail that saves you: `$SLURM_LOCALID` is the rank *within the node* (0–7), so only one of the 8 GPU processes per node does the copy — otherwise all 8 would download the same shards, wasting bandwidth and disk. The trailing `srun ... true` is a cheap cross-node barrier ensuring no rank starts reading before the cache is populated.

#### Worked example: does the dataset fit on local NVMe?

You have an 800 GB sharded dataset and nodes with 7 TB of local NVMe each. Does the caching strategy apply? With WebDataset's `split_by_node` across 4 nodes, each node only needs *its* quarter of the shards: 800 / 4 = 200 GB per node, which fits comfortably in 7 TB. Staging cost: 200 GB from S3 at, say, an aggregate 5 GB/s download lands in ~40 seconds per node, paid once at job start. After that, every epoch reads at local-NVMe speed (~5 GB/s, ~100 µs latency) instead of contending for Lustre across the network. For a 50-epoch run, you pay 40 seconds once to save the per-epoch network read 50 times — an overwhelming win. The decision flips if the dataset is 50 TB: it does not fit on NVMe even split four ways, so you stream from the parallel filesystem with WebDataset's sequential reads and accept Lustre as the data source. The rule of thumb: **if the per-node shard slice fits on local NVMe with room to spare, always cache; if it does not, stream sequentially from the parallel filesystem and never touch object storage in the hot loop.** Numbers here are illustrative of the regime, not a specific cluster — measure your own staging bandwidth.

| Tier | Bandwidth | Latency | Capacity | Use it for |
| --- | --- | --- | --- | --- |
| Object store (S3/GCS) | 10–100 GB/s (many streams) | 10–100 ms/object | EB, cheap | Archive, source of truth, cold tier |
| Parallel FS (Lustre/GPFS) | 100–300 GB/s aggregate | ms, metadata-heavy | PB, shared | Active dataset shards, run-from |
| Local NVMe (per node) | 3–7 GB/s per drive | < 100 µs | 1–15 TB, private | Hot-epoch cache, scratch |

## Checkpointing at scale: how a save freezes 64 GPUs

You are training a 70B-parameter model on 64 GPUs. Every hour you save a checkpoint so a crash does not cost you a day. And every hour, your 64 GPUs go completely idle for **three and a half minutes** while one process slowly writes a 280 GB file. Over a 5-day run with hourly checkpoints, that is roughly 7 hours of 64 idle H100s — call it \$1,000 burned doing nothing but waiting for a disk. This is the last and most overlooked piece of the cluster, and the fix is one of the most satisfying wins in distributed training.

First the arithmetic, because the size is what makes naive checkpointing untenable. A checkpoint must store the model parameters, the optimizer state, and often a bit of metadata. For a model with $\Psi$ parameters trained with Adam in mixed precision, the dominant costs are: fp32 master weights ($4\Psi$ bytes), fp32 Adam first and second moments ($4\Psi + 4\Psi$ bytes), and the bf16 weights ($2\Psi$). That is roughly $14\Psi$ bytes — for a 70B model, about **980 GB**, call it ~1 TB. (The exact factor depends on what you save; the point is it is an order of magnitude bigger than the model's bf16 weights alone, which is why "just save `model.state_dict()`" understates the problem.) Now the write: if one process gathers all that state to rank 0 and writes it through a single stream at ~3.2 GB/s, the save takes $980 / 3.2 \approx 306$ seconds — five minutes, during which the entire job is frozen because every other rank is blocked at the gather. This is the failure the next figure contrasts.

![Before and after comparison showing a monolithic checkpoint gathering seven hundred gigabytes to rank zero over a single stream taking two hundred twenty seconds versus thirty-two ranks writing sharded async at one hundred sixty gigabytes per second in nine seconds](/imgs/blogs/running-on-a-cluster-slurm-multi-node-launch-and-data-pipelines-7.png)

The before-and-after is the entire argument for **distributed (sharded) checkpointing.** On the left, the monolithic save: all ranks ship their slice of the state to rank 0, which holds the whole ~700 GB in one place and writes it through one stream at ~3.2 GB/s, taking ~220 seconds with all GPUs idle. On the right, the sharded save: each of the 32 ranks writes *its own shard* — ~22 GB each — directly and in parallel to the parallel filesystem. Now you are using the filesystem's aggregate bandwidth: 32 streams hitting Lustre can sustain ~160 GB/s combined, so the write finishes in ~9 seconds. And if you make it **asynchronous** — copy the state to a host buffer fast, then let a background thread do the slow disk write while training resumes — the GPUs barely pause at all. A 220-second freeze becomes a ~9-second flush, or effectively zero with async overlap. That is a 24× reduction in the worst case and the reclaiming of those 7 idle GPU-hours.

PyTorch's `torch.distributed.checkpoint` (often imported as `dcp`) does exactly this — every rank writes its own shard, the bandwidth is aggregate, and loading reshards automatically if you resume on a different number of GPUs. The save and load look like this:

```python
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_state_dict, set_state_dict,
)

def save_checkpoint(model, optimizer, step, ckpt_dir):
    # Pull the *sharded* state dict: each rank holds only its slice.
    model_sd, optim_sd = get_state_dict(model, optimizer)
    state = {"model": model_sd, "optim": optim_sd, "step": step}
    # Every rank writes its own shard in parallel; no gather to rank 0.
    dcp.save(state, checkpoint_id=f"{ckpt_dir}/step-{step}")

def load_checkpoint(model, optimizer, ckpt_dir, step):
    model_sd, optim_sd = get_state_dict(model, optimizer)
    state = {"model": model_sd, "optim": optim_sd}
    # dcp re-shards on load: resume on 32 GPUs from a 64-GPU checkpoint works.
    dcp.load(state, checkpoint_id=f"{ckpt_dir}/step-{step}")
    set_state_dict(model, optimizer,
                   model_state_dict=state["model"],
                   optim_state_dict=state["optim"])
```

Two properties make this the right tool. First, **no rank-0 gather**: `get_state_dict` returns each rank's local shard, and `dcp.save` writes them concurrently, so you get the parallel-write speedup automatically — this is the natural companion to FSDP, covered in [memory optimization with ZeRO and FSDP](/blog/machine-learning/high-performance-computing/memory-optimization-zero-fsdp-activation-checkpointing-and-offload), where the model is already sharded across ranks. Second, **resharding on load**: a distributed checkpoint is stored as logical tensors plus a sharding plan, not as "rank 5's bytes," so you can save on 64 GPUs and resume on 32 (or 128) — `dcp.load` figures out which bytes each new rank needs. That portability is impossible with monolithic `torch.save`, and it is what lets you recover a run on whatever node count the scheduler happens to give you next.

For the asynchronous variant — overlapping the disk write with continued training — recent PyTorch exposes `dcp.async_save`, which snapshots the state to a staging buffer quickly and returns a future while a background thread flushes to disk:

```python
ckpt_future = None

def maybe_async_save(model, optimizer, step, ckpt_dir):
    global ckpt_future
    if ckpt_future is not None:
        ckpt_future.result()        # ensure the previous save finished
    model_sd, optim_sd = get_state_dict(model, optimizer)
    state = {"model": model_sd, "optim": optim_sd, "step": step}
    ckpt_future = dcp.async_save(state, checkpoint_id=f"{ckpt_dir}/step-{step}")
    # training continues immediately; the disk write happens in the background
```

The pattern — wait for the previous save before starting the next, so you never have two in flight — keeps host memory bounded and the GPUs almost never stall. Combined with the `--signal=B:USR1@120` handler from the SLURM script, you also get a *graceful* final checkpoint when the time limit approaches: the signal fires, your handler triggers one last `dcp.save`, the job requeues, and the next allocation resumes from exactly where it stopped. That is how a 5-day training run survives a cluster that only lets any single job run for 24 hours.

#### Worked example: sizing the checkpoint budget

You are deciding how often to checkpoint a 13B model on 32 GPUs. The state is roughly $14 \times 13\text{B} \approx 182$ GB. With sharded `dcp.save`, 32 ranks each write ~5.7 GB; if the parallel filesystem sustains ~120 GB/s aggregate for your job, the write takes $182 / 120 \approx 1.5$ seconds of actual I/O, and with async overlap the GPU stall is near zero. So checkpointing every 200 steps costs essentially nothing and you should do it — frequent checkpoints mean a crash costs minutes, not hours. Contrast the monolithic path: gather 182 GB to rank 0, write at 3.2 GB/s, and each save is ~57 seconds of *frozen* GPUs; at that cost you would checkpoint rarely (every few thousand steps), and a crash would cost you all the work since the last one. The sharded approach does not just save time per checkpoint — it lets you checkpoint *more often*, which shrinks your worst-case loss from a crash. The two wins compound. (Bandwidth figures are regime-typical; your filesystem and contention will differ — measure with a few real saves and read `sacct`/the filesystem monitor.)

| Aspect | Monolithic (`torch.save` on rank 0) | Sharded (`dcp.save`, async) |
| --- | --- | --- |
| Who writes | rank 0 only, after gather | every rank, its own shard |
| Bandwidth used | one stream (~3 GB/s) | aggregate (~100+ GB/s) |
| 70B save time | ~5 minutes, GPUs frozen | ~9 s, or ~0 with async |
| Resume on different N GPUs | no (shape mismatch) | yes (reshards on load) |
| Host memory peak | full state on rank 0 | one shard per rank |
| Right for | small models, single node | large models, multi-node |

## Case studies / real numbers

Numbers from the wild make the abstract concrete. A few documented patterns, framed honestly as approximate where the exact figure depends on configuration.

**A data-loader-bound vision run on a DGX node.** This is the canonical case and it matches the before/after we built. A team training an image model on 8× A100 saw the GPUs hover at ~40% utilization with `nvidia-smi` showing the cards mostly idle between steps. The model was not the issue; profiling (a `torch.profiler` trace showing long gaps before each forward pass) revealed the GPU waiting on the loader. Switching from loose JPEG files to WebDataset shards, raising `num_workers` from 2 to ~12, enabling `pin_memory` and `persistent_workers`, and caching shards on local NVMe lifted utilization to ~95% and roughly **2.3× the throughput** on the same hardware. The lesson the whole industry keeps re-learning: at scale, the data pipeline is co-equal with the model, and it is usually the cheaper thing to fix. PyTorch's own `DataLoader` documentation and the WebDataset project both center this exact workflow.

**Large-cluster checkpoint strategy.** Training runs at the scale of LLaMA, GPT-class, and PaLM models all report some form of frequent, distributed checkpointing — saving every few hundred to few thousand steps so that a hardware failure (and at 1,000+ GPUs, *something* fails most days) costs minutes of recovery, not a restart. The mechanics are exactly the sharded/async pattern: each rank writes its slice in parallel, recovery reshards onto whatever healthy nodes remain. The PyTorch distributed-checkpoint (`torch.distributed.checkpoint`) documentation describes the parallel-write and resharding design directly, and it is now the default for FSDP-based training. The headline takeaway: a checkpoint that takes 5 minutes monolithically takes single-digit seconds sharded, and that gap is what makes frequent checkpointing affordable, which is what makes a multi-day run on thousands of GPUs survivable.

**Storage bandwidth as the ceiling.** On a large cluster, the parallel filesystem's aggregate bandwidth is a hard, shared ceiling — Lustre/GPFS deployments commonly land in the 100–300 GB/s range for a single job's slice, and that bandwidth is *shared* across every running job. When 50 jobs all read at once, your effective bandwidth is a fraction of the rated peak, which is precisely why the local-NVMe caching pattern matters: it moves your steady-state reads off the contended shared resource and onto private per-node SSD. The same physics governs checkpoint writes — your 32 parallel streams compete with everyone else's, so the "160 GB/s sharded write" is a good-day number, and on a busy cluster you should expect less and measure it. Treat any single bandwidth figure as a ceiling under no contention, not a guarantee.

**Multi-node scaling efficiency.** Published distributed-training results consistently show that well-tuned data-parallel training holds high scaling efficiency — frequently 85–95% — out to dozens or low hundreds of GPUs, with the loss to ideal linear scaling being the inter-node all-reduce tax. The drop accelerates when communication is misconfigured (NCCL on the wrong fabric), when the global batch grows so large the model becomes communication-bound per step, or when the data loader cannot feed the larger GPU count. The reason this series keeps returning to the [GPU benchmark methodology](/blog/machine-learning/mlops/llm-gpu-benchmark) is that the *only* way to know your real efficiency is to measure tokens/s at 1, 8, and 64 GPUs and compute the ratio — a vendor's marketing slide will not tell you whether your cluster is at 90% or 60%.

**How the three failures compound.** The most instructive real-world cases are the ones where two or three of these problems hide behind each other, because fixing one reveals the next. A team scales from 8 to 64 GPUs and sees only 4× speedup (50% efficiency). They suspect the network, name the InfiniBand fabric for NCCL, and recover to 6× — better, but still short of the expected ~7.2×. Now the all-reduce is fast, so the *next* bottleneck surfaces: the data loader that comfortably fed 8 GPUs cannot feed 64, because the parallel filesystem's metadata server is now serving 768 worker processes instead of 96, and reads stall. They shard the data into WebDataset archives and add a local-NVMe cache, and efficiency climbs to 90%. Only *then* does the checkpoint freeze become visible — it was always there, but it was lost in the noise of a slow job; on a fast 90%-efficient run, a 4-minute hourly freeze is suddenly a glaring 7% of wall-clock, so they switch to sharded async checkpointing. Each fix unmasked the next, and the team that does not understand this structure declares victory after the NCCL fix and leaves 40% of their compute on the table. The lesson is to measure after *every* change and keep asking "what is the bottleneck *now*," exactly as the [profiling](/blog/machine-learning/high-performance-computing/profiling-gpu-workloads-finding-the-real-bottleneck) discipline teaches — the bottleneck moves, and the cluster is a stack of bottlenecks, not a single one.

## When to reach for this (and when not to)

Going multi-node is a cost, not a free upgrade, and the honest advice is to delay it as long as you can.

**Stay on one node if the model and a healthy batch fit on 8 GPUs and DDP saturates NVLink.** A single node's 8 GPUs over NVLink communicate ~18× faster than across nodes, with none of the rendezvous, fabric-configuration, or scheduler complexity. If one node gets you 90%+ utilization and the run finishes in acceptable time, **do not go multi-node** — you will add the InfiniBand tax, the NCCL-fabric footgun, and a class of distributed bugs for no benefit. The cluster is for when you genuinely need more GPUs than one node holds, or more aggregate memory than 8 cards provide.

**Fix the data loader before you add GPUs.** This is the most important "when not to" in the post. If your single node is at 40% utilization, adding a second node does not double your throughput — it gives you *sixteen* starving GPUs instead of eight, and now you are paying double to waste compute at twice the scale. Get to 90%+ on one node first. The loader fix is almost always cheaper, faster, and higher-leverage than the parallelism fix, and a starving pipeline does not magically heal when you scale it.

**Use sharded checkpointing the moment your state exceeds a few tens of GB, or you are on more than one node.** Below that, monolithic `torch.save` is simpler and fine. Above it — any large model, any multi-node FSDP run — the monolithic gather-to-rank-0 is a self-inflicted wound that freezes the job. Switch to `dcp.save`. There is essentially no downside at scale, and the resharding-on-load property is worth it on its own.

**Cache on local NVMe only when the per-node shard slice fits.** If it does, always cache — the staging cost is paid once and the per-epoch win is large. If the dataset is too big to fit even split across nodes, do not contort yourself; stream sequentially from the parallel filesystem with WebDataset and accept Lustre as the source. And never read training batches directly from object storage in the hot loop — its per-object latency will starve you regardless of how many workers you throw at it.

**Reach for job arrays whenever you have many similar runs.** Hyperparameter sweeps, per-shard preprocessing, ablations — these are arrays, not 50 separate submissions. The scheduler packs them efficiently and the `%N` throttle keeps you a good cluster citizen. Conversely, do not use an array for one big training job; that is a single multi-node submission with `--requeue`.

## Key takeaways

- A cluster is a **shared** facility owned by a **scheduler**: you do not run programs, you `sbatch` jobs that describe a resource request, and SLURM allocates nodes when they are free.
- The seam between SLURM and PyTorch is `srun` launching one `torchrun` per node, and `torchrun` spawning one rank per GPU; `--node_rank` from `$SLURM_NODEID` is what makes 32 processes coordinate instead of colliding.
- **Rendezvous** is how ranks find each other: every process dials the same `MASTER_ADDR:PORT` on node 0, forms the `world_size`, and gets a unique rank. `init_method="env://"` flows the runtime-discovered address into every process.
- Always name the fast fabric for NCCL (`NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`) and turn on `NCCL_DEBUG=INFO` once — silently falling back to Ethernet is a 3× slowdown with no error message.
- The data loader must **out-produce the step**: $W/t_s \ge B/t_{\text{step}}$, or the GPU starves. Size workers by that inequality and the prefetch buffer by Little's law ($L = \lambda W$), not by guessing.
- On a parallel filesystem, **shard your data** (WebDataset `.tar` archives of ~5,000 samples each): trade millions of metadata-killing tiny reads for thousands of fast sequential ones.
- `pin_memory=True`, `persistent_workers=True`, and a local-NVMe cache are the unglamorous knobs that turn a **40%-utilized** GPU into a **95%-utilized** one — same hardware, ~2.3× throughput.
- Checkpoint state is ~$14\Psi$ bytes (~1 TB for 70B). Use **sharded async** `torch.distributed.checkpoint`: every rank writes its own shard in parallel, turning a 5-minute frozen save into a ~9-second flush, and it reshards on load so you can resume on any GPU count.
- Fix the loader before adding nodes; stay single-node while DDP saturates NVLink; measure scaling efficiency at 1, 8, and 64 GPUs rather than trusting a slide. The cluster gives you scale only if you stop wasting it.

## Further reading

- [Parallelism strategies: data, tensor, pipeline, and expert](/blog/machine-learning/high-performance-computing/parallelism-strategies-data-tensor-pipeline-and-expert) — how to split a model that does not fit on one GPU, the next decision after you can launch on many nodes.
- [Collective communication and NCCL: all-reduce from scratch](/blog/machine-learning/high-performance-computing/collective-communication-and-nccl-all-reduce-from-scratch) — what actually happens between GPUs every step, and why the fabric choice sets your scaling efficiency.
- [Interconnects: NVLink, NVSwitch, InfiniBand, and RDMA](/blog/machine-learning/high-performance-computing/interconnects-nvlink-nvswitch-infiniband-and-rdma) — the hardware behind the 18× intra-node vs inter-node bandwidth ratio that shapes every parallelism decision.
- [Profiling GPU workloads: finding the real bottleneck](/blog/machine-learning/high-performance-computing/profiling-gpu-workloads-finding-the-real-bottleneck) — how to see the data-loader stall in a trace instead of guessing, with `torch.profiler` and Nsight.
- [The HPC playbook for AI engineers](/blog/machine-learning/high-performance-computing/the-hpc-playbook-for-ai-engineers) — the capstone that ties the scheduler, the pipeline, the collectives, and the roofline into one decision procedure.
- [Choosing and benchmarking GPUs for LLMs](/blog/machine-learning/mlops/llm-gpu-benchmark) — how to measure tokens/s and scaling efficiency honestly so you know whether your cluster is at 90% or quietly burning money at 60%.
- The SLURM documentation (`sbatch`, `srun`, `sinfo`, job arrays), the WebDataset project, and the PyTorch `torch.distributed.checkpoint` and `DataLoader` docs are the primary references for every command and API in this post.

---
title: "Debugging vLLM distributed serving: the failures that only appear across GPUs and nodes"
date: "2026-07-06"
publishDate: "2026-07-06"
description: "A field guide to the failure modes unique to running vLLM across multiple GPUs and nodes — silent NCCL hangs, wedged engines from a single dead worker, Ray placement-group deadlocks, one-rank OOM stalls — and how to diagnose each against vLLM's real architecture."
tags:
  [
    "model-serving",
    "inference",
    "vllm",
    "distributed-serving",
    "nccl",
    "tensor-parallelism",
    "ray",
    "debugging",
    "gpu",
    "infiniband",
    "llm-serving",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/debugging-vllm-distributed-serving-1.webp"
---

The page said `0 tokens/s`. Not an error, not a crash, not a `CUDA out of memory` traceback — just a 16-GPU vLLM engine that had been happily serving a 70B model on two nodes, now sitting at exactly zero throughput with every process still alive, every GPU still allocated, and every log file silent since the last successful step forty seconds ago. `kubectl get pods` showed all pods `Running`. `nvidia-smi` showed 79 GB used on all sixteen cards. The health check was still returning `200 OK` because the HTTP server was fine — it was the engine behind it that had turned to stone. There was no obvious thing to restart and no obvious thing to read.

This is the signature of a distributed-serving failure, and it is a genuinely different animal from anything you debug on a single GPU. On one card, when something goes wrong you get a Python traceback with a line number. Across GPUs and nodes, the thing that goes wrong is usually a *collective* — a synchronization point where every rank must participate — and the symptom of a broken collective is not an exception. It is a hang. One rank dies or falls behind, and the other fifteen wait for it, forever, because that is exactly what a barrier is supposed to do. The failure is silent by construction.

![Layered stack of a distributed vLLM engine from clients down through the API server, EngineCore, MultiProcExecutor, worker ranks, NCCL collectives, and GPU interconnect](/imgs/blogs/debugging-vllm-distributed-serving-1.webp)

This post is a field guide to those failures — the ones that only appear once vLLM spans more than one GPU, and especially once it spans more than one node. We will map each failure mode onto vLLM's actual architecture: the `AsyncLLM` front end, the `EngineCore`, the `MultiProcExecutor` that spawns one `WorkerProc` per rank, the `rpc_broadcast_mq` shared-memory queue it broadcasts work on, the ZMQ sockets that stitch the API server to the engine, the NCCL collectives that move activations between GPUs, the Ray actors that place workers across nodes, and the `DPCoordinator` that keeps data-parallel replicas in lockstep. For each failure you get the same three things: the symptom you actually observe, the specific log or component that confirms the root cause, and the fix. This is post H3-adjacent troubleshooting in the Model Deployment and Serving series; it assumes you know the SLO triangle of **latency ↔ throughput ↔ cost** from [what is model serving](/blog/machine-learning/model-serving/what-is-model-serving), and it is the debugging companion to the [vLLM distributed architecture anatomy](/blog/machine-learning/model-serving/vllm-distributed-architecture-anatomy) and [running vLLM distributed in production](/blog/machine-learning/model-serving/running-vllm-distributed-in-production) siblings. By the end you will be able to look at a wedged multi-node engine and, within about five minutes, know which of nine failure modes you are looking at and which command to run next.

## 1. The mechanics: why one slow or dead rank freezes all of them

Before any individual incident, you need the one mechanical fact that explains three-quarters of them. It is this: **a distributed vLLM forward pass is a broadcast-then-collect wrapped around a chain of synchronizing collectives, and every link in that chain is a barrier.** Get this and the hangs stop being mysterious.

Here is the control flow, straight from vLLM's V1 architecture. When the `EngineCore` decides to run a step, it hands the batch to the executor. For multi-GPU, that is `MultiProcExecutor`, which at startup spawned one daemon `WorkerProc` per rank via `WorkerProc.make_worker_process` — one process per GPU, each pinned to its device, each having loaded its shard of the weights. The executor does not call the workers like functions. It **enqueues** the work item into `rpc_broadcast_mq`, a message queue implemented over shared memory, and that enqueue is non-blocking — it returns immediately. Every worker sits in a busy loop blocked on `rpc_broadcast_mq.dequeue()`. When the item lands, all workers dequeue it at once and begin executing the same step. The executor then **blocks** on the designated output rank's `worker_response_mq.dequeue()`, waiting for the single result it needs to return upstream.

That is the outer shell: one broadcast out, one blocking collect back. Now look inside the step. A tensor-parallel forward pass is not embarrassingly parallel. After every attention block and every MLP block, the partial results computed on each GPU's shard must be summed across the whole TP group — that is an `all_reduce`. For a model with $L$ layers, a single decode step performs on the order of $2L$ all-reduces, each one a hard synchronization point where **no rank proceeds to the next layer until every rank has contributed its shard**.

![Dataflow graph showing the executor broadcasting to three worker ranks where rank two has died, the all-reduce barrier never completing, and the executor blocking forever while collecting results](/imgs/blogs/debugging-vllm-distributed-serving-2.webp)

The figure above is the whole tragedy in one picture. The executor broadcasts to ranks 0, 1, 2. Ranks 0 and 1 reach the first all-reduce and wait. Rank 2 has died — segfaulted during a kernel, or been killed by the OOM reaper, or hit an assertion. It never reaches the all-reduce. NCCL's all-reduce is a *collective*: it completes on every participating rank only when all of them have called it. With rank 2 gone, ranks 0 and 1 block inside `ncclAllReduce` on the GPU, spinning on a CUDA stream that will never be signalled. Meanwhile the executor is blocked on `worker_response_mq.dequeue()` for the output rank's result, which will never be enqueued because the output rank is stuck in the collective. Every layer of the stack is now waiting on the layer below it, and the layer at the bottom is waiting on a corpse. Zero tokens per second, no exception, no log line — the engine is wedged.

### The straggler math

You can make the "wait for the slowest" claim quantitative, and it is worth doing because it also explains the *slow* (not dead) case. Let $t_i$ be the wall-clock time at which rank $i$ finishes its local share of the work before a given collective. Because the collective is a barrier, the time every rank observes for that collective to complete is

$$t_{\text{coll}} = \max_{i \in \{0,\dots,N-1\}} t_i + t_{\text{comm}}$$

where $t_{\text{comm}}$ is the actual data-movement cost of the reduce. The step latency of the whole TP group is governed by the **slowest** rank, not the average. If one rank is doing extra work — say it is thrashing on memory because it is closer to OOM, or it landed on a GPU that is thermally throttling — that rank sets the pace for all $N$. This is why a single degraded GPU in a fleet does not slow itself down by 10%; it slows the *entire tensor-parallel group* down to its own speed, and if it stalls entirely, the group stalls entirely.

The communication term itself is not free, and it is where the node boundary bites. NCCL's ring all-reduce moves $\frac{2(N-1)}{N}\cdot S$ bytes across each GPU's link for a payload of $S$ bytes, so

$$t_{\text{comm}} \approx \frac{2(N-1)}{N}\cdot\frac{S}{B} + (2N-2)\,\alpha$$

where $B$ is the effective per-link bandwidth and $\alpha$ is the per-hop latency. The whole reason tensor parallelism must stay *inside* a node is that $B$ for NVLink is roughly 900 GB/s while $B$ for a good InfiniBand fabric is roughly 400 Gb/s (~50 GB/s) and for TCP over a 25 GbE management NIC it is a few GB/s at best. Put a TP all-reduce on the slow path and $t_{\text{comm}}$ dominates the step; the GPUs sit idle waiting on the wire. Hold that formula — it reappears in three separate incidents below.

Make it concrete with a real payload. Take a 70B-class model with hidden size $h = 8192$ running a decode step at batch $b = 64$ over a tensor-parallel group of $N = 8$. Each all-reduce moves the post-projection activation, which in BF16 is $S = b \cdot h \cdot 2 = 64 \cdot 8192 \cdot 2 \approx 1.05$ MB. On NVLink at ${\sim}900$ GB/s the transfer term is $\frac{2 \cdot 7}{8} \cdot \frac{1.05\text{ MB}}{900\text{ GB/s}} \approx 2.0\,\mu s$ — utterly negligible against the tens of microseconds of compute per layer. Now put that same all-reduce on a TCP link at ${\sim}3$ GB/s: the transfer term balloons to ${\sim}610\,\mu s$, and with roughly $2L = 160$ all-reduces per step for an 80-layer model you have added on the order of $160 \cdot 610\,\mu s \approx 98$ ms of pure wire time to a step whose compute is a few milliseconds. That is not "20% slower"; that is a step dominated 20-to-1 by communication, and it is exactly why the accidental-TP-across-nodes incident produces GPUs sitting at single-digit utilization. The same arithmetic run with $N=8$ pipeline stages instead of tensor parallelism moves activations only *once per stage boundary* — a few transfers per step instead of $2L$ — which is precisely why pipeline parallelism tolerates the slow inter-node link and tensor parallelism does not.

### Why the failure is silent

The last piece is the silence, and it has a mundane cause. The workers are child processes. When a `WorkerProc` dies, its stderr may or may not be plumbed back to the parent, and even when it is, the *parent's* symptom is not "child died" — it is "my `dequeue()` is taking a long time." vLLM does install health checks and, in recent versions, a heartbeat between the executor and its workers, so a dead worker is eventually detected and the engine is torn down rather than hanging forever. But "eventually" can be tens of seconds to minutes depending on configuration, and the NCCL collective on the GPU has its own separate watchdog (default 600 s / 10 min for `TORCH_NCCL_HEARTBEAT_TIMEOUT`-style timeouts). In the window between "rank died" and "some watchdog fired," you have a live-looking engine serving nothing. Your job in that window is to find the dead or lagging rank faster than the watchdog does.

## 2. A map of the failure space

Nine distinct failure modes will bite you across a distributed vLLM deployment, and the single most useful triage move is the first fork: **did it fail at startup, before it ever served a token, or did it fail mid-serve, under live traffic?** These two branches have almost disjoint root-cause sets, and knowing which branch you are on eliminates half the candidates immediately.

![Triage tree splitting a distributed vLLM failure into startup failures and mid-serve failures, each with three concrete root causes](/imgs/blogs/debugging-vllm-distributed-serving-3.webp)

A **startup** failure is one where the process never reaches "engine ready." The hallmark is that it hangs or dies during the initialization banner — you see `Initializing an LLM engine`, maybe `init_process_group`, and then nothing. Startup failures are almost always *configuration*: the ranks cannot find each other on the network, the parallelism degree does not match the GPU count, a placement group cannot be satisfied, or a port is already bound. The good news is that startup failures are deterministic — they fail the same way every launch — so you can iterate on them quickly.

A **mid-serve** failure is one where the engine came up, served real requests, and then stalled or crashed under traffic. The hallmark is a period of healthy `tokens/s` in the metrics followed by a cliff. Mid-serve failures are almost always *dynamic*: a request pattern drove one rank to OOM, a GPU degraded, a collective stalled, or a data-parallel replica desynchronized. These are harder because they are load-dependent and sometimes intermittent, and because the useful evidence (which rank was slow, what the memory looked like at the moment of the stall) is gone by the time you go looking unless you captured it.

Everything that follows is organized along this fork: startup incidents first (Sections 3–4), then the mid-serve incidents (Sections 5–7), then the distribution-layer specifics — Ray, ZMQ, containers — that cut across both (Sections 8–10), then the master table (Section 11) and real case studies. Keep the tree in your head; it is the fastest thing you own.

## 3. Startup hang at "init distributed": the classic multi-node freeze

This is the failure that every team hits on their first multi-node launch, and it is worth dwelling on because the fix is trivial once you can *see* it and impossible until you can. The symptom: you launch the engine, the log prints something like `Started a local Ray instance` or `init_process_group`, and then it hangs. No error. Ten minutes later a NCCL or Gloo timeout finally fires with a wall of text about a rank not responding. Under Kubernetes it manifests as pods that never pass their readiness probe, `CrashLoopBackOff` after the init timeout, and a 4 AM page.

The root cause is almost always one of three things, and they are all about the ranks failing to establish the NCCL communicator across the node boundary:

1. **Wrong network interface.** NCCL auto-selects a NIC to carry its bootstrap and, if it is not using GPUDirect RDMA, its data. On a multi-homed host it frequently picks the wrong one — the Docker bridge `docker0`, a `lo` loopback, or a management NIC with no route to the other node — and then the ranks on node A and node B simply cannot reach each other. You fix this by pinning `NCCL_SOCKET_IFNAME` to the correct high-speed interface and, for vLLM specifically, setting `VLLM_HOST_IP` (older env: `HOST_IP`) so the engine advertises the address the *other* node can actually route to, not `127.0.0.1`.
2. **No InfiniBand where you thought there was.** The fabric is there physically but NCCL is not using it, so it silently falls back to TCP sockets. This does not hang — it makes everything ten times slower, which is Section 3's evil twin and its own case study below.
3. **A firewall on the rendezvous port.** The ranks need to reach the master's rendezvous/RPC port (for vLLM's data-parallel setup, `--data-parallel-rpc-port`; for the torch rendezvous, the `MASTER_PORT`). If a security group or `iptables` rule blocks it, the bootstrap connection never completes and you hang.

### Confirming it with `NCCL_DEBUG=INFO`

You do not guess at which of these it is. You turn on NCCL's own logging and read which transport it chose. This single environment variable is the most valuable debugging tool in the distributed-serving toolbox:

```bash
# Launch the engine with NCCL and vLLM debug logging on, on BOTH nodes.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET      # only the init + net subsystems, not the flood
export VLLM_LOGGING_LEVEL=DEBUG

# Pin the interface and the advertised IP explicitly instead of trusting auto-select.
export NCCL_SOCKET_IFNAME=ens1f0       # your actual high-speed NIC, from `ip -br addr`
export GLOO_SOCKET_IFNAME=ens1f0       # torch's CPU-side rendezvous uses Gloo
export VLLM_HOST_IP=10.0.3.11          # this node's routable IP on that NIC

vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend ray \
  2>&1 | tee /var/log/vllm/node0.log
```

Now grep the log for the transport NCCL actually selected. The good case and the bad case look like this:

```console
# GOOD — NCCL found and is using InfiniBand with GPUDirect RDMA:
node0:pid:tid NCCL INFO NET/IB : Using [0]mlx5_0:1/IB [1]mlx5_1:1/IB ; OOB ens1f0:10.0.3.11<0>
node0:pid:tid NCCL INFO Channel 00/0 : 0[7] -> 8[7] via NET/IB/0/GDRDMA
node0:pid:tid NCCL INFO Connected all rings

# BAD — NCCL fell back to plain TCP sockets over some interface:
node0:pid:tid NCCL INFO NET/Socket : Using [0]docker0:172.17.0.1<0>
node0:pid:tid NCCL INFO Channel 00/0 : 0[7] -> 8[7] via NET/Socket
# ...and then it hangs here forever, because 172.17.0.1 is not routable to node1.
```

The line that matters is `NET/IB` versus `NET/Socket`, and the interface name in brackets. `NET/IB ... GDRDMA` means you are on the fast path. `NET/Socket : Using [0]docker0` means NCCL is trying to talk between nodes over the Docker bridge, which is exactly the hang. The moment you see the wrong interface, you know the fix is `NCCL_SOCKET_IFNAME`. If you see `NET/IB` but the throughput is still terrible, that is a different problem (Section 3's twin). If you see nothing after `NET/Socket` and it just stops, that is the firewall — the connection to the peer is being dropped.

A useful confirmation trick before you even launch vLLM: run NCCL's own `all_reduce_perf` test (from `nccl-tests`) across the two nodes. If that hangs or reports single-digit GB/s, the problem is your fabric, not vLLM, and you have just saved yourself an hour of chasing the wrong layer.

#### Worked example: a two-node 70B launch that hung for ten minutes

A team brought up Llama-3.1-70B with `tensor_parallel_size=8, pipeline_parallel_size=2` across two 8×H100 nodes. Every launch hung after `init_process_group` and died ten minutes later with a Gloo timeout. `nvidia-smi` on both nodes showed all sixteen GPUs at 100% utilization during the hang — which looked like progress but was actually sixteen GPUs spinning inside a collective waiting for a connection that would never open.

They set `NCCL_DEBUG=INFO` and relaunched. Node 0's log showed `NET/Socket : Using [0]eth0:10.0.0.5` — and node 1 was advertising `172.18.x.x`, an address on a *different* subnet that node 0 could not route to. The cluster had two networks: a 25 GbE management network on `eth0` and a 400 Gb/s InfiniBand fabric on `ib0`/`mlx5`, and NCCL had auto-selected `eth0` on one node and a container-internal bridge on the other. The fix was three lines in the launch env on both nodes:

```bash
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export VLLM_HOST_IP=$(ip -4 -br addr show ib0 | awk '{print $3}' | cut -d/ -f1)
```

Relaunch: `NET/IB ... GDRDMA`, `Connected all rings`, engine ready in 90 seconds. The straggler math from Section 1 tells you why the *right* fix mattered beyond just "not hanging": moving the eventual inter-node pipeline traffic from ~3 GB/s TCP to ~45 GB/s IB dropped the cross-node transfer term by ~15×. Time-to-first-token went from a projected disaster to 340 ms. The whole incident was ten minutes of hang plus fifteen minutes of reading one log line.

## 4. TP/PP config mismatch: when the arithmetic doesn't close

The second startup failure is pure arithmetic, and it produces some of the most confusing error messages in the stack because the *symptom* (a shape mismatch or a NCCL "invalid rank" error) is far downstream of the *cause* (you asked for a world size that does not match your GPUs). vLLM's world size is `tensor_parallel_size × pipeline_parallel_size × data_parallel_size`. Every one of those GPUs must exist, be visible, and be evenly divisible into the parallel groups. When they are not, you get one of these:

- `tensor_parallel_size` does not divide the model's attention head count → a shard has a fractional number of heads → a reshape fails deep in the model with a cryptic size mismatch.
- `world_size` exceeds the number of visible GPUs → a worker is assigned to a device that does not exist → `RuntimeError: CUDA error: invalid device ordinal`.
- Uneven GPUs per node under Ray → the placement group cannot pack the ranks the way TP requires (all TP peers on one node) → either a placement failure (Section 8) or, worse, TP peers split across the node boundary and a catastrophic slowdown.

The confirmation is a thirty-second sanity check you should run *before* every distributed launch, not after it fails:

```python
# preflight.py — run on each node before launching vLLM.
import torch

tp = 8          # --tensor-parallel-size
pp = 2          # --pipeline-parallel-size
dp = 1          # --data-parallel-size
world = tp * pp * dp

visible = torch.cuda.device_count()
print(f"world_size = tp*pp*dp = {tp}*{pp}*{dp} = {world}")
print(f"visible GPUs on this node = {visible}")

# For Llama-3.1-70B: 64 attention heads, 8 KV heads.
num_heads, num_kv_heads = 64, 8
assert num_heads % tp == 0, f"tp={tp} must divide num_heads={num_heads}"
assert num_kv_heads % tp == 0 or tp % num_kv_heads == 0, \
    f"tp={tp} vs num_kv_heads={num_kv_heads}: GQA sharding will be uneven"

# TP peers MUST be co-located on one node. Check GPUs-per-node * nodes == world.
gpus_per_node, num_nodes = 8, 2
assert gpus_per_node % tp == 0 or tp % gpus_per_node == 0, \
    f"tp={tp} does not tile cleanly onto {gpus_per_node} GPUs/node — TP will cross nodes"
assert gpus_per_node * num_nodes == world, "world_size != total GPUs"
print("preflight OK")
```

The head-divisibility assertions are the ones people forget. A 70B model with 64 heads and 8 KV heads shards cleanly at `tp ∈ {1,2,4,8}` but *not* at `tp=6` — and vLLM will not tell you that in a friendly way; it will fail with a tensor reshape error 200 lines into model loading. The error you actually see looks like this, and it names the *symptom* (a shape that will not view) rather than the cause (`tp` does not divide the heads):

```console
RuntimeError: shape '[-1, 6, 128]' is invalid for input of size 8192
  File ".../vllm/model_executor/layers/linear.py", line ..., in weight_loader
  File ".../vllm/model_executor/models/llama.py", line ..., in load_weights
# 8192 = 64 heads * 128 head_dim; asking for 6 shards leaves 64/6 heads per rank,
# which is not an integer -> the view fails. tp=6 was never valid for this model.
```

When you see an "invalid for input of size N" reshape error during `load_weights`, do the division in your head: `N / head_dim` gives the total heads, and if `tp` does not divide it evenly, that is your bug — not a corrupt checkpoint, not a vLLM version issue. The rule to internalize: **`tensor_parallel_size` must divide the KV-head count (for GQA models) and the attention-head count, and TP peers must be co-located on a single node.** If you need more parallelism than a node has GPUs, you add it with pipeline parallelism (which crosses the node boundary cheaply because it only sends activations at stage boundaries) or data parallelism (independent replicas), never by stretching TP across nodes. The [tensor, pipeline, and expert parallelism for serving](/blog/machine-learning/model-serving/tensor-pipeline-expert-parallelism-for-serving) post covers why each axis has the cost profile it does; here the operational takeaway is simply that the arithmetic must close *and* respect the interconnect hierarchy.

### Slow multi-node throughput: accidental TP across the node boundary

The nastiest version of a parallelism-config mistake does not fail — it *runs*, slowly, and looks like a vLLM performance bug. It happens when the launch topology puts a tensor-parallel group's ranks on *different* nodes. vLLM assigns ranks to GPUs in order, and if your Ray placement or your `CUDA_VISIBLE_DEVICES` layout interleaves GPUs across nodes, ranks 0–3 of a TP=8 group can land on node A and ranks 4–7 on node B. Every per-layer all-reduce now crosses the inter-node link — the one thing the Section 1 math says must never happen. The engine comes up, serves correct tokens, and posts single-digit GPU utilization with throughput 5–20× below what the same GPUs deliver when TP stays intra-node.

The confirmation is two-pronged. First, `nvidia-smi` on every node during serving: healthy TP shows all GPUs pegged near 100% compute; accidental-cross-node TP shows all GPUs idling in the 5–20% range because they spend each step waiting on the wire. Second, `NCCL_DEBUG=INFO` shows the ring topology — look for TP-group channels routed `via NET/IB` or `via NET/Socket` (crossing the node) instead of `via P2P`/`NVLink` (staying on-node). A TP all-reduce that reports a `NET/` transport at all is the smoking gun; intra-node TP should never touch the network. The related sub-case is **PCIe instead of NVLink** even within a node — on a server without NVSwitch, or where the topology is misconfigured, TP peers talk over PCIe at ${\sim}32$ GB/s (Gen4 x16) rather than NVLink's ${\sim}900$ GB/s. `nvidia-smi topo -m` prints the GPU interconnect matrix; `NV#` between two GPUs means NVLink, `PIX`/`PXB`/`SYS` means they are talking over PCIe or across a socket. If your TP peers show `SYS` (cross-socket), you have found a ${\sim}28\times$ bandwidth cliff hiding inside a single box. The fix is topology-aware placement: keep each TP group on GPUs that share NVLink, add cross-node scale with pipeline or data parallelism, and verify with `nvidia-smi topo -m` before you trust a benchmark.

## 5. A worker dies silently and the engine wedges

Now we cross into mid-serve territory, and the first incident is the one from the intro: everything is up, traffic is flowing, and then throughput drops to zero with no error. This is the failure the Section 1 mechanics predicted — a `WorkerProc` died, and `MultiProcExecutor` is blocked forever on `worker_response_mq.dequeue()` for the output rank while the surviving workers spin inside a NCCL collective waiting for the dead one.

The symptom checklist that identifies *this* failure specifically, as opposed to a generic slowdown:

- Throughput is **exactly** zero, not merely degraded. A slow rank gives you low-but-nonzero tokens/s; a dead rank gives you a clean flatline.
- The HTTP server still answers health checks (`/health` returns 200) because the API server process is fine — it is the engine core behind it that is stuck.
- `nvidia-smi` shows one or more GPUs pinned at high utilization (the survivors spinning in the collective) and possibly one GPU with a process that has vanished or a Xid error in `dmesg`.
- No new lines in the engine log for tens of seconds, then eventually a NCCL watchdog timeout or a vLLM `EngineDeadError`.

### Confirming it: DEBUG logs plus a stack dump of the hung workers

Two tools nail this. First, if you can reproduce it, run with `VLLM_LOGGING_LEVEL=DEBUG` so the executor logs each broadcast and each collected response; the last successful step number tells you exactly where it stopped. Second — and this is the one that works even in production, on a live hung process, with no restart — `py-spy dump` on each worker PID. `py-spy` reads the stack of a running Python process without stopping it, so you can see precisely what each rank is blocked on:

```bash
# 1. Find the worker PIDs. vLLM's workers are children of the EngineCore process.
#    Each WorkerProc runs worker_main and shows up as a python process holding a GPU.
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
pgrep -af "VLLM::Worker\|worker_main\|from_engine_core" | awk '{print $1}'

# 2. Dump the Python stack of every worker WITHOUT killing it.
for pid in $(pgrep -f "worker_main"); do
  echo "===== worker pid $pid ====="
  py-spy dump --pid "$pid" --native      # --native also shows the C/CUDA frames
done
```

A healthy worker mid-step shows a stack ending in a forward pass. The wedged survivors show a stack ending inside NCCL — the tell-tale frame is a blocked `ncclAllReduce` or a wait on a CUDA stream:

```console
# A SURVIVING rank, stuck waiting for the dead peer:
Thread 0x7f... (idle): "MainThread"
    all_reduce (vllm/distributed/communication_op.py)
    forward (vllm/model_executor/models/llama.py:...)
    execute_model (vllm/worker/gpu_worker.py:...)
    worker_busy_loop (vllm/v1/executor/multiproc_executor.py:...)   # blocked on the collective
```

Now count your workers. If you launched `tp=8, pp=2` you expect sixteen worker PIDs each holding a GPU. If `nvidia-smi` shows only fifteen compute processes, or one GPU with zero used memory where the others have 79 GB, **that is your dead rank.** Cross-reference the missing rank with `dmesg -T | grep -i -E "xid|nvrm|out of memory"` and the kernel log usually tells you why it died: an Xid 79 (GPU fell off the bus), an Xid 13 (illegal instruction — often a real model bug or a bad kernel), or an `oom-killer` line naming the python PID.

The fix depends on *why* the worker died, and that is the point of finding it rather than blindly restarting:

- **OOM-killed** (see `dmesg` oom-killer): lower `--gpu-memory-utilization` or `--max-num-seqs`; the rank ran out of memory under a request pattern the others survived. This is Section 7's territory.
- **Xid 79 / GPU fell off bus**: hardware or driver — cordon the node, drain it, and get the GPU replaced. No config change will help.
- **Segfault in a custom kernel / illegal memory access (Xid 13/31)**: usually a specific input shape or a build mismatch; capture the request that triggered it and pin the vLLM/CUDA versions.

#### Worked example: counting workers to find the corpse

A `tp=8` single-node engine wedged at 0 tok/s under a burst of long-context requests. The on-call engineer's first move was the worker count. `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` listed **seven** python processes, not eight — GPU 3 had a compute process using 0 MB, an orphan. `py-spy dump` on the seven survivors all showed the same `ncclAllReduce` frame from Section 1: they were blocked in the collective. `dmesg -T | grep -i oom` on the node produced the answer in one line — `Out of memory: Killed process 48213 (VLLM::Worker)` — the OS OOM-killer had reaped the rank on GPU 3 because a spike in host-pinned memory (the KV-cache staging buffers under a long-context burst) pushed the container over its cgroup memory limit. Note the subtlety: the *GPU* had memory to spare, but the *container's host RAM* did not, and the OOM-killer does not care which. The fix was to raise the container's memory limit and cap `--max-num-batched-tokens` so the pinned-memory staging could not spike as hard. Total diagnosis time once they ran the worker count: about four minutes. The lesson is that "count your workers against `tp × pp`" is the single fastest discriminator between "a worker died" and "a worker is merely slow" — a dead rank is a *missing* process, and `nvidia-smi` shows it immediately.

vLLM does defend against this on its own. Recent versions run a heartbeat between the `EngineCore` and its `WorkerProc` children and raise an `EngineDeadError` (tearing down the engine so the orchestrator can restart it) once a worker stops responding. But the heartbeat interval plus the NCCL collective watchdog means detection can lag the actual death by tens of seconds to minutes, and during that window the engine looks healthy to a naive TCP health check while serving nothing. The durable mitigation, separate from the root cause, is to not rely on that detection latency. A liveness probe that actually exercises the engine — a tiny generation request with a short timeout, not just a TCP check on the port — turns a silent wedge into a fast pod restart. We wire one up in Section 10's checklist.

## 6. NCCL timeout mid-serve: the collective stall

Closely related but distinct: instead of a rank *dying*, a rank falls *behind*, and a collective stalls until NCCL's watchdog gives up and aborts the whole communicator. The symptom is a burst of healthy serving followed by a stall, then — after the timeout, default around 600 seconds unless you tuned it — a dramatic NCCL error naming a rank that "has not joined" the collective, followed by the engine tearing down.

The confusing part is that the rank named in the NCCL error is usually the *victim*, not the culprit. NCCL reports the ranks it is still *waiting on*, which are the ones that arrived at the collective and are blocked — the slow rank that never arrived is often not the one in the error message. So you cannot just read the error and restart the named rank; you have to find the straggler.

![Timeline of triaging a NCCL timeout incident from the watchdog abort through confirming the stall, finding the lagging rank, and rebalancing the memory budget](/imgs/blogs/debugging-vllm-distributed-serving-5.webp)

The triage timeline above is the exact sequence that works. When the watchdog fires, do not restart yet — if you can, dump the state first, because a restart destroys the only evidence. Turn on `NCCL_DEBUG=INFO` (if it was not already on) and look for the stuck collective. Then `py-spy dump` every rank as in Section 5, but this time you are looking for the *odd one out*: fifteen ranks blocked in `ncclAllReduce`, and one rank blocked somewhere else — in `cudaMalloc`, in a CPU-side data load, in a Python GC pause. That one is your straggler. Cross-reference with per-node `nvidia-smi` to see if the straggler's GPU is near its memory ceiling (about to OOM, thrashing) or thermally throttling (`nvidia-smi -q -d TEMPERATURE,CLOCK` shows a clock far below its peers).

Two NCCL environment variables change the *character* of this failure and are worth knowing:

```bash
# Make the watchdog fail FASTER (default is generous, 10+ minutes) so a stall
# turns into a clean abort-and-restart instead of a 10-minute silent wedge.
export NCCL_TIMEOUT=300                 # seconds; also TORCH_NCCL_* variants exist

# Make the abort produce a full stack trace of the stuck collective, so the
# post-mortem names the operation and the ranks involved.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576   # record recent collectives for a flight recorder
```

Tuning the timeout *down* is counterintuitive but usually correct for serving: a 10-minute wedge is far more expensive than a fast crash-and-restart, because during the wedge you are dropping every request while looking healthy. A shorter timeout converts a long silent outage into a short, loud one that your orchestrator can react to.

#### Worked example: a one-rank OOM that stalled an all-reduce

A 16-GPU engine serving a mix of short chat turns and the occasional 30k-token document-summarization request ran fine for hours, then stalled every afternoon around peak. The NCCL error named ranks 0–14 as "waiting on rank 15," which sent the team chasing rank 15 for a day before they realized rank 15 was the victim.

`py-spy dump` during the next stall told the real story: ranks 0–14 were blocked in `ncclAllReduce`, and **rank 5** — not 15 — was blocked in `cudaMalloc`, trying and failing to allocate a KV-cache block. Per-node `nvidia-smi` confirmed it: rank 5 sat at 79.4 GB of 80 GB while every other rank sat around 68 GB. When a long-context request's KV cache grew, it happened to land its incremental blocks on the ranks that were already fullest, and rank 5 hit the ceiling first. It could not allocate, so it never reached the all-reduce, so the whole group stalled, so NCCL eventually named the ranks *waiting on* the collective (the survivors), which pointed everyone at the wrong rank.

The fix was two-part. Immediate: drop `--gpu-memory-utilization` from 0.98 to 0.90, which gave every rank ~1.6 GB of headroom and stopped the OOM. Durable: cap `--max-num-seqs` and `--max-model-len` so the worst-case concurrent KV footprint could not exceed the per-rank budget derived from the Section 1 memory math. The measured result:

| Metric | Before (mem_util 0.98) | After (mem_util 0.90, capped) |
|---|---|---|
| Stalls per day | 1–2 (afternoon peak) | 0 over 3 weeks |
| Peak rank memory | 79.4 GB / 80 GB | 71 GB / 80 GB |
| Throughput (tok/s) | 4,100 (when alive) | 3,850 |
| p99 TTFT | 340 ms | 360 ms |
| Effective uptime | ~92% | 99.9% |

Note the trade: 6% lower peak throughput and 20 ms more TTFT bought a jump from 92% to 99.9% effective uptime. That is the SLO triangle doing its job — you spent a little throughput to buy reliability, and for a production endpoint that is almost always the right trade.

## 7. Uneven GPU memory across ranks and one-rank OOM

The previous incident's root cause deserves its own section because it is subtle and common: **memory is not perfectly balanced across ranks, so one rank OOMs while the others have room.** People assume tensor parallelism splits everything evenly, so if the model fits at all it fits everywhere. It mostly does — but "mostly" is what pages you at 3 AM.

![Grid of eight GPU ranks under tensor parallelism where seven ranks hold about 68 gigabytes and one rank holds 79 gigabytes near the out-of-memory threshold](/imgs/blogs/debugging-vllm-distributed-serving-6.webp)

The grid above is what a real per-rank memory snapshot looks like when this is about to bite: seven ranks near 68 GB and one rank at 79.4 GB, one bad request away from the ceiling. Several things create this imbalance:

- **GQA and uneven head sharding.** When the KV-head count does not divide evenly by `tp`, some ranks own more KV heads than others and therefore more KV cache per token.
- **Pipeline-stage imbalance.** Under PP, the first stage holds the embedding table and the last stage holds the LM head and the sampler's logits buffer, so those ranks carry weight the middle stages do not.
- **Rank 0's extra duties.** The driver/output rank often does a bit more — logits processing, detokenization staging — and carries a slightly larger footprint.
- **Allocator fragmentation.** PyTorch's caching allocator fragments differently under different request histories, so two ranks with identical logical load can have different *reserved* memory.

The diagnostic is a per-node, per-rank memory scan you should be able to run in one command during any incident. This is the script to keep in your runbook:

```bash
#!/usr/bin/env bash
# rank_mem.sh — snapshot per-GPU memory on every node, flag the outlier rank.
# Run via your cluster's fan-out (pdsh, ray exec, or a k8s DaemonSet exec).
set -euo pipefail

echo "host,gpu,mem_used_MiB,mem_total_MiB,util_pct,temp_C"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu \
           --format=csv,noheader,nounits \
  | while IFS=, read -r idx used total util temp; do
      echo "$(hostname),${idx// /},${used// /},${total// /},${util// /},${temp// /}"
    done

# Flag any GPU within 5% of its memory ceiling — that is your next OOM.
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
  | awk -F, '{ pct = $2/$3*100; if (pct > 95) printf "WARN gpu %s at %.1f%% memory (%s/%s MiB) — OOM risk\n", $1, pct, $2, $3 }'
```

Fan this out across nodes (`pdsh -w node[0-1] bash rank_mem.sh`, or `ray exec` per node, or a Kubernetes `kubectl exec` loop) and you get a fleet-wide picture in seconds. The rank that is 10 GB above its peers is the one that will OOM next, and the `WARN` line names it before it does.

The fixes, in order of preference:

1. **Lower `--gpu-memory-utilization`** (e.g., 0.90 instead of the aggressive 0.95–0.98). This is the blunt, reliable fix: it reserves less of each GPU for the KV cache, leaving headroom that absorbs the imbalance. You pay in maximum batch size (fewer concurrent sequences), which is a throughput cost, but it is a knob you can turn in one line.
2. **Cap `--max-num-seqs` and `--max-model-len`** so the worst-case concurrent KV footprint is bounded below the *tightest* rank's budget, not the average.
3. **Pick a `tensor_parallel_size` that shards heads evenly** so the imbalance never arises. If your KV-head count is 8, prefer `tp ∈ {1,2,4,8}`.
4. **Watch for the `enforce_eager` trap**: CUDA graphs (the default) capture buffers that add a few GB per rank; if you are memory-constrained, `--enforce-eager` frees them at a latency cost. Know the trade before you reach for it.

## 8. Ray-specific failures: placement groups, dead actors, version skew

If you run vLLM with `--distributed-executor-backend ray`, you inherit a second distributed system with its own failure modes, and they are worth calling out because their diagnostics live in a completely different place — the Ray dashboard and `ray status`, not the vLLM logs.

**Placement group cannot be satisfied.** vLLM asks Ray for a placement group with one bundle per GPU (one `{"GPU": 1}` bundle per rank), and it needs those bundles packed such that TP peers land on the same node. If the cluster does not have enough free GPUs, or the scheduler cannot pack them the way the strategy requires, the placement group sits `PENDING` forever and the engine hangs at startup — a startup failure, but one whose evidence is not in the NCCL log at all. The confirmation:

```bash
# Is the cluster even big enough, and are the GPUs free?
ray status
# Look at the "Resources" section:
#   Usage: 14.0/16.0 GPU        <- only 2 GPUs free, but you asked for 16
#   Demands:
#     {'GPU': 1.0} * 16 (PACK): 16+ pending   <- your placement group, stuck

# List placement groups and their state:
ray list placement-groups
#   PLACEMENT_GROUP_ID   STATE      ...    <- look for PENDING that never becomes CREATED

# Which nodes exist and what do they actually have?
ray list nodes
```

The `Demands` line with a `PENDING` placement group and a `Usage` line showing fewer free GPUs than you requested is the unambiguous signature. The root cause is almost always one of: another job is holding GPUs, a node is down (`ray list nodes` shows it `DEAD`), or you asked for a `world_size` larger than the cluster. The fix is to match the placement group's GPU demand to the actual free capacity, evict the competing job, or bring the dead node back.

#### Worked example: the placement group that packed wrong

A team ran two vLLM engines on one 4-node (32-GPU) Ray cluster: a `tp=8` production engine and a `tp=8` staging engine. Production was healthy; staging hung at startup, forever, with no NCCL log at all — because it never got as far as NCCL. `ray status` told the whole story: `Usage: 24.0/32.0 GPU`, and a `Demands: {'GPU': 1.0} * 8 (STRICT_PACK): 8 pending`. Production held 24 GPUs across three nodes, leaving 8 free — but those 8 were split 4-and-4 across the two remaining partially-used nodes, and vLLM's placement asks for a `STRICT_PACK` so that all 8 TP peers land on one node. There was no single node with 8 free GPUs, so the placement group could never be satisfied even though the *total* free count (8) matched the request. The engine was not out of GPUs; it was out of GPUs *that could be packed the way TP requires*. The fix was to drain a node so 8 contiguous GPUs freed up. The lesson: for a distributed engine, "enough free GPUs" is necessary but not sufficient — they must be packable onto the node topology the parallelism plan demands, and `ray status`'s `Demands` line with a `PACK`/`STRICT_PACK` strategy is where you read that.

**Actor died / node lost.** Ray runs each vLLM worker as an actor. If a node hosting actors dies (hardware, preemption of a spot instance, OOM at the OS level), Ray reports the actors as `DEAD` and the engine that depended on them wedges or errors. The Ray dashboard's Actors tab and `ray list actors --filter "state=DEAD"` name them. Under Kubernetes this is often a spot-instance reclaim; the fix is architectural (don't put a single indivisible engine's ranks on preemptible nodes) more than tactical.

**Version and CUDA skew across nodes.** This one produces the most baffling errors because it violates an assumption you did not know you were making: *every node in the cluster must run the exact same vLLM, PyTorch, NCCL, and CUDA versions.* A node that pulled a slightly different image — a `:latest` tag that drifted, a manually patched host — will fail to form the NCCL communicator (mismatched NCCL versions refuse to talk) or will produce subtle numerical divergence that shows up as garbage output on some ranks. The check is mechanical and belongs in your preflight:

```bash
# Run on every node; the outputs must be byte-identical across the cluster.
python -c "import vllm, torch; print(vllm.__version__, torch.__version__, torch.version.cuda)"
python -c "import torch; print('nccl', torch.cuda.nccl.version())"
nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1
```

Pipe these through `md5sum` across nodes in CI, or bake the versions into a single immutable image that every node pulls by digest, not by tag. The [multi-node LLM serving for 100B+ models](/blog/machine-learning/model-serving/multi-node-llm-serving-100b-plus) post covers the launch topology in depth; the debugging lesson here is narrow and absolute: version skew is a silent correctness bug, so eliminate the possibility rather than trying to detect its symptoms.

## 9. ZMQ sockets, data-parallel lockstep, and the DPCoordinator

The last architectural layer with its own failures is the one between the API server and the engine, and the one that keeps data-parallel replicas synchronized. In vLLM's V1 design, the API server (`AsyncLLM`) and the `EngineCore` are separate processes connected by ZMQ sockets — the front end does not call the engine in-process; it sends requests over a socket and receives outputs over another. When you scale to data parallelism across nodes, a `DPCoordinator` process and per-node `EngineCoreProc` instances join the picture, handshaking over ZMQ `DEALER` sockets.

![Graph of the multi-node ZMQ topology where the API server talks to the DPCoordinator which dispatches to two engine cores that must step in lockstep before results reach the output queue](/imgs/blogs/debugging-vllm-distributed-serving-7.webp)

Two families of failure live here.

**ZMQ / socket errors between the API server and the engine.** These are the ordinary networking failures, and they mostly bite at startup or restart:

- `Address already in use` (`EADDRINUSE`): a previous engine process did not release its socket, or two engines are trying to bind the same port. `ss -ltnp | grep <port>` finds the process holding it; kill the zombie or change the port. Under Kubernetes, a pod that restarted too fast before the old socket drained is the usual cause.
- `Cannot assign requested address` (`EADDRNOTAVAIL`): the engine is trying to *bind* to an IP that does not exist on this host — almost always because `VLLM_HOST_IP` or a data-parallel address was set to another node's IP or to a stale value. The bind address must be a local interface.
- The API server comes up but every request times out: the front end bound and is listening, but its ZMQ connection to the `EngineCore` never established (wrong address, firewall on the internal port). The tell is a healthy HTTP server with an engine that never logs a received request.

**Data-parallel lockstep and DPCoordinator desync.** This is the subtle one. Under data parallelism, vLLM runs the replicas in lockstep: *if any replica has work, all replicas execute a forward step together.* A replica with no requests does not simply idle — it runs a **dummy step** to participate in the synchronization the group requires. This is deliberate: without it, replicas would drift out of the collective schedule and deadlock. But it means two failure modes:

1. **One replica starved while others are busy** shows up as uneven load, and if the coordinator's load-balancing information is stale or a replica's input thread stops reporting, one replica can be handed nothing while another is overloaded. The `DPCoordinator` is the process that mediates this — it sends queue sizes and running/waiting counts to the front end's stats-update task and reports wave-state transitions. If it is unhealthy, load balancing degrades.
2. **A dummy-step desync**, where one replica fails to take its dummy step in time (because it is stuck on something else), stalls the whole data-parallel group at the barrier — the same broadcast-then-collect wedge from Section 1, one level up, at the replica granularity instead of the rank granularity.

The confirmation lives in the DP coordination logs (run with `VLLM_LOGGING_LEVEL=DEBUG` and grep for `DPCoordinator`, `wave`, and `dummy`), the per-replica queue-depth metrics (a replica reporting zero queued requests while others report hundreds is starved), and the ZMQ socket state (`ss -x` for the internal sockets). The fix is usually to correct the data-parallel address configuration (`--data-parallel-address`, `--data-parallel-rpc-port`) so the coordinator handshake completes across nodes, and to make sure every replica's `EngineCore` is reachable — a starved replica is frequently a replica whose handshake never fully completed, so the coordinator does not route to it.

#### Worked example: the replica that never took work

A `--data-parallel-size 4` deployment across two nodes (two replicas per node) showed a puzzling load pattern: three replicas ran hot at 200–300 queued requests each while the fourth sat at zero, and aggregate throughput was stuck at three-quarters of capacity. The obvious hypothesis — a load-balancer bug sending nothing to replica 3 — was almost right but pointed at the wrong layer. `VLLM_LOGGING_LEVEL=DEBUG` and a grep for `DPCoordinator` showed the coordinator's wave-state updates listing only three replicas as participating; replica 3 was executing **dummy steps** on every wave (keeping lockstep so the group did not deadlock) but was never handed real requests. The coordinator was not routing to it because replica 3's initial `DEALER` handshake had never completed — its `EngineCore` had bound to a `--data-parallel-address` that resolved to the wrong node's IP, so the coordinator's connection attempt silently failed and it dropped replica 3 from the routable set. The engine did not crash because the dummy-step machinery is *designed* to let a non-participating replica keep the barrier alive; the cost was just a permanently idle quarter of the fleet. Correcting the `--data-parallel-address` on that node so the handshake completed brought replica 3 into the routable set, queue depth balanced across all four within a wave, and throughput jumped to full capacity. The lesson: under data parallelism, a replica at zero load while others are saturated is not usually "the balancer forgot it" — it is "the coordinator never successfully handshook it," and the dummy-step design masks the failure by keeping the group running without it.

## 10. Container gotchas: shared memory, IPC, and NCCL environment

A disproportionate number of "it works on bare metal but hangs in the container" incidents come down to two Docker flags, and they are worth their own section because the error they produce — a shared-memory allocation failure or a `Bus error` — looks nothing like their cause.

Recall from Section 1 that `MultiProcExecutor` broadcasts work over `rpc_broadcast_mq`, a message queue **implemented over shared memory**, and that PyTorch's multiprocessing and NCCL also use `/dev/shm` for inter-process tensors and buffers. Docker's default `/dev/shm` is **64 MB**. That is nowhere near enough for a multi-worker vLLM engine, and when a worker tries to allocate a shared-memory segment larger than what is available, you get a `Bus error (core dumped)` or a hang during worker startup — a failure that points at nothing useful in the vLLM logs.

The fix is two flags, and you should treat them as mandatory for any GPU inference container:

```bash
docker run --gpus all \
  --ipc=host \                 # share the host IPC namespace so /dev/shm is large
  --shm-size=16g \             # or size it explicitly; 64MB default is the killer
  --ulimit memlock=-1 \        # unlimited locked memory — required for GPUDirect RDMA / IB
  --ulimit stack=67108864 \    # generous stack for deep model graphs
  --network=host \             # let NCCL use the host's high-speed NICs directly
  -e NCCL_DEBUG=INFO \
  -e NCCL_SOCKET_IFNAME=ib0 \
  -e VLLM_HOST_IP=10.0.3.11 \
  vllm/vllm-openai:latest \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 8 --pipeline-parallel-size 2
```

`--ipc=host` and `--shm-size` are the two that fix the shared-memory queue failures; use one or the other (host IPC namespace, or an explicitly large `/dev/shm`), and in practice teams set both. `--ulimit memlock=-1` is required for InfiniBand and GPUDirect RDMA to register pinned memory — without it, NCCL cannot use the fast path and silently falls back to sockets, which is Section 3's twin all over again. `--network=host` lets NCCL bind the host's real NICs rather than a bridge, which is often what makes `NCCL_SOCKET_IFNAME=ib0` actually work.

On Kubernetes the equivalents are a `medium: Memory` `emptyDir` mounted at `/dev/shm` (sized appropriately), `securityContext` capabilities for locked memory, and `hostNetwork: true` with the RDMA device plugin. The single most common Kubernetes-specific version of this incident is a pod that runs fine at `tp=2` on one node and hangs at `tp=8` across two nodes because the `/dev/shm` `emptyDir` was never added — the shared-memory pressure only shows up past a worker-count threshold.

Finally, the liveness probe that turns Section 5's silent wedge into a fast restart. Do not health-check the TCP port; that stays up while the engine is stone. Check that the engine actually *generates*:

```bash
#!/usr/bin/env bash
# engine_liveness.sh — used as a k8s livenessProbe exec. Fails fast if the
# engine is wedged even though the HTTP port is still open.
set -e
curl -sf -m 5 http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"served-model","prompt":"ping","max_tokens":1,"temperature":0}' \
  | grep -q '"text"' || { echo "engine not generating — wedged"; exit 1; }
```

Wire that as a `livenessProbe` with a sensible `periodSeconds` and `failureThreshold`, and a wedged engine gets killed and rescheduled in under a minute instead of bleeding requests until a human notices.

## 11. The master triage table

Put it all together and you get a single lookup table: for each incident, the one component that emits its signal, the symptom you observe, the root cause, and the fix. This is the artifact to pin above your desk — the whole point of the exercise is that the log you read next is never a guess.

![Matrix mapping six distributed incidents to the component that emits their signal, the observable symptom, the root cause, and the fix](/imgs/blogs/debugging-vllm-distributed-serving-8.webp)

Here is the same table in text, expanded to all nine failure modes with the exact command to run for each:

| Incident | Component / where to look | Signal you see | Root cause | First command → fix |
|---|---|---|---|---|
| Startup NCCL hang | NCCL init log | Hangs after `init_process_group`, timeout after 10 min | Wrong NIC / no route between nodes | `NCCL_DEBUG=INFO` shows `NET/Socket` → set `NCCL_SOCKET_IFNAME`, `VLLM_HOST_IP` |
| Silent TCP fallback | NCCL init log | Comes up but ~10× slow inter-node | IB present but unused | `NCCL_DEBUG=INFO` shows `NET/Socket` not `NET/IB` → `memlock=-1`, correct IFNAME |
| TP/PP mismatch | model-load traceback | Shape mismatch or `invalid device ordinal` | `world_size` ≠ GPUs, or `tp` ∤ heads | `preflight.py` assertions → fix `tp`/`pp` |
| Worker died, wedged | `MultiProcExecutor` / `nvidia-smi` | Exactly 0 tok/s, no error, health 200 | `WorkerProc` crashed (OOM/Xid) | `py-spy dump` all ranks + `dmesg` → fix per cause |
| NCCL timeout mid-serve | NCCL watchdog | Stall, then abort naming a rank after 600 s | One slow / OOM straggler | `py-spy` finds the odd-one-out rank → rebalance |
| One-rank OOM | KV block manager | OOM on rank *k* only | Uneven KV/head sharding | `rank_mem.sh` → lower `--gpu-memory-utilization` |
| Ray PG pending | `ray status` | Engine hangs at startup, PG `PENDING` | Too few free GPUs / bad pack | `ray status` demands → match PG to capacity |
| ZMQ bind error | engine startup log | `EADDRINUSE` / `EADDRNOTAVAIL` | Zombie socket or wrong bind IP | `ss -ltnp` → free port / fix `VLLM_HOST_IP` |
| DP replica starved | `DPCoordinator` log | One replica idle, others overloaded | Stale LB info / incomplete handshake | grep `DPCoordinator` → fix `--data-parallel-address` |
| shm too small (container) | worker startup | `Bus error` / hang at worker init | `/dev/shm` = 64 MB default | add `--ipc=host --shm-size=16g` |

Two comparison tables are worth internalizing alongside it. First, the interconnect hierarchy that decides where each parallelism axis is allowed to live, because half the incidents above are ultimately "you put fast traffic on a slow link":

| Link | Effective bandwidth | Where it lives | Which traffic belongs here |
|---|---|---|---|
| NVLink / NVSwitch | ~900 GB/s | Intra-node, GPU↔GPU | Tensor-parallel all-reduce (per layer, huge volume) |
| InfiniBand (400 Gb) | ~45–50 GB/s | Inter-node | Pipeline-stage activations, expert routing |
| 25–100 GbE (TCP) | ~3–12 GB/s | Inter-node fallback | Control plane only — never a TP all-reduce |
| `/dev/shm` (host) | memory-speed | Intra-node, proc↔proc | `rpc_broadcast_mq` broadcast + response |

Second, the startup-vs-mid-serve split as a decision aid, because it is the first fork you make:

| | Startup failure | Mid-serve failure |
|---|---|---|
| Hallmark | Hangs before "engine ready" | Healthy, then a throughput cliff |
| Determinism | Fails identically every launch | Load-dependent, sometimes intermittent |
| Usual cause | Config: NICs, ports, world size, PG | Dynamic: OOM, degraded GPU, desync |
| First tool | `NCCL_DEBUG=INFO`, `ray status`, `preflight.py` | `py-spy dump`, `rank_mem.sh`, `nvidia-smi` |
| Evidence lifetime | Reproducible on demand | Gone after restart — capture first |

## Case studies

Three incidents, framed the way they actually unfold, each ending in the resolution and the durable fix.

**Case study 1 — the InfiniBand that wasn't (silent TCP fallback).** A team migrated a 70B deployment from a single 8-GPU node to two nodes to make room for KV cache and larger batches. The engine came up cleanly — no hang, no error — but throughput on the two-node setup was *lower* than the single node had been: ~110 tok/s against an expectation of nearly a thousand, with all sixteen GPUs sitting around 20% utilization. Everyone assumed a vLLM scheduling bug and spent two days tuning batch sizes.

![Before-and-after comparison of NCCL running over a misconfigured TCP socket versus InfiniBand, showing roughly ten times higher inter-node all-reduce bandwidth on the InfiniBand path](/imgs/blogs/debugging-vllm-distributed-serving-4.webp)

The picture above is what they finally found when they set `NCCL_DEBUG=INFO`: `NET/Socket` on the management NIC, not `NET/IB`. The InfiniBand fabric was physically connected and the `mlx5` devices were visible, but the container had been launched without `--ulimit memlock=-1`, so NCCL could not register the pinned memory GPUDirect RDMA needs and silently fell back to TCP over the 25 GbE management interface. Every inter-node pipeline transfer was crawling over a link roughly ten times slower than the fabric they had paid for. The straggler math from Section 1 explains the 20% utilization exactly: the GPUs finished their compute quickly and then sat idle for the vast majority of each step waiting on the wire. Adding `--ulimit memlock=-1` and pinning `NCCL_SOCKET_IFNAME=ib0` flipped the log to `NET/IB ... GDRDMA`, all-reduce latency dropped from ~38 ms to ~3 ms, and throughput went to ~920 tok/s — an 8.4× improvement from two flags. The lesson: a distributed engine that comes up "fine" but slow is a network-layer bug until `NCCL_DEBUG=INFO` proves otherwise, and the confirmation is a single line of the init log.

**Case study 2 — the shared-memory bus error.** A different team packaged vLLM in a custom container to add their own auth middleware. It worked in every test — because every test ran at `tensor_parallel_size=1` on a single GPU. The first `tp=4` deployment crashed during worker startup with `Bus error (core dumped)` and a stack that pointed into PyTorch's shared-memory allocator, nowhere near their code. They suspected their middleware, then a corrupted checkpoint, then a driver bug.

The cause was the Dockerfile: a clean base image with the default 64 MB `/dev/shm`. At `tp=1` there is a single worker and the shared-memory pressure is trivial; at `tp=4`, the `MultiProcExecutor` sets up `rpc_broadcast_mq` across four workers plus PyTorch's inter-process tensors, and 64 MB is exhausted immediately, so the allocation faults with a bus error. Adding `--ipc=host --shm-size=16g` to the run command (and the equivalent `/dev/shm` `emptyDir` to their Kubernetes pod spec) fixed it permanently. The durable lesson they wrote into their runbook: *any* GPU inference container gets `--ipc=host --shm-size` and `--ulimit memlock=-1` as non-negotiable defaults, and multi-GPU functionality must be tested at the real `tp` degree, never at `tp=1`, because the entire class of shared-memory and NCCL failures is invisible below the multi-worker threshold.

**Case study 3 — the afternoon-peak stall (one-rank OOM).** This is the Section 6 worked example, seen as a case study end to end. The failure only appeared at peak, only in the afternoon, and named the wrong rank in its error, which is why it took a day to diagnose despite being a textbook one-rank OOM. The resolution — dropping `--gpu-memory-utilization` from 0.98 to 0.90 and capping `--max-num-seqs`/`--max-model-len` so the worst-case KV footprint provably fit the tightest rank — traded 6% of peak throughput for a jump from ~92% to 99.9% effective uptime. The meta-lesson is about *evidence lifetime*: the team could only find rank 5 as the culprit because someone ran `py-spy dump` and `rank_mem.sh` *during* a live stall instead of restarting immediately. A restart would have cleared the memory pressure and destroyed the only evidence, and the stall would have recurred the next afternoon, unexplained. Mid-serve failures reward the discipline of capturing state before you reach for the restart button.

**Case study 4 — the two-node engine that was slower than one node.** A team scaled a 34B model from a single 8-GPU node (`tp=8`) to two nodes to raise concurrency, using `tp=8, pp=2` for a 16-GPU engine. The engine came up, served correct output, and delivered *lower* aggregate throughput than the single node it replaced, with all sixteen GPUs hovering around 12% utilization. The instinct — "the second node isn't helping, the code must not parallelize" — was wrong; the second node was actively hurting. `nvidia-smi topo -m` on each node was normal (NVLink within each box), so the intra-node fabric was fine. The tell was in `NCCL_DEBUG=INFO`: the tensor-parallel channels for the group were routed `via NET/IB` — the TP all-reduce was crossing the node boundary. The Ray placement had interleaved the ranks so that a single TP=8 group straddled both nodes rather than keeping each pipeline stage's TP group on one node. Every one of the ${\sim}2L$ per-step all-reduces was paying the inter-node latency that the Section 1 arithmetic says is fatal, so the GPUs finished their compute in a millisecond and then waited tens of milliseconds on the wire. The fix was to set the parallelism so that `tp=8` filled exactly one node and `pp=2` spanned the two nodes — pipeline traffic crosses the boundary a few times per step, tensor traffic stays on NVLink. After the correction, GPU utilization jumped to the 80s and throughput roughly quadrupled the single-node baseline. The lesson generalizes every incident in this post: with distributed serving, "correct output" tells you almost nothing about "correct configuration," and the difference is written in `nvidia-smi` utilization and one line of the NCCL topology log.

## When to use this (and when not to) — a prevention checklist

The honest framing: most of these failures are preventable, and the debugging skills above are what you fall back on when prevention lapsed. If you are standing up a new distributed vLLM deployment, spend the hour on prevention rather than earning the 4 AM page. The checklist, in order:

1. **Run `nccl-tests` `all_reduce_perf` across your nodes before vLLM.** If it does not report near-line-rate bandwidth on `NET/IB`, stop — fix the fabric first. This one test prevents the two most expensive incidents (Sections 3 and its slow twin).
2. **Pin `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, and `VLLM_HOST_IP` explicitly.** Never trust auto-selection on a multi-homed host. Bake them into the launch template.
3. **Set container flags as mandatory defaults**: `--ipc=host --shm-size=16g --ulimit memlock=-1`, `--network=host` (or the Kubernetes equivalents). Test at the real `tp` degree.
4. **Run `preflight.py`** — assert `world_size` matches visible GPUs and `tp` divides the head counts — as the first line of your launch script.
5. **Leave memory headroom.** Start at `--gpu-memory-utilization 0.90`, not 0.98, and only raise it after you have measured the per-rank imbalance under real traffic. Cap `--max-num-seqs` and `--max-model-len` to bound worst-case KV.
6. **Use an immutable, digest-pinned image on every node.** Version skew is a silent correctness bug; make it structurally impossible rather than trying to detect it.
7. **Tune the NCCL timeout *down*** (e.g., 300 s) so a stall becomes a fast crash-and-restart instead of a silent 10-minute wedge, and wire a real generation liveness probe, not a TCP check.
8. **Keep `py-spy`, `rank_mem.sh`, and `NCCL_DEBUG=INFO` one command away** in your runbook, and capture state before restarting a mid-serve failure.

When is distributed serving *not* worth the failure surface it adds? If your model fits comfortably on a single GPU, or on a single node with `tp` inside NVLink, do not cross the node boundary — every failure mode in Sections 3, 8, 9, and 10 exists only because you are spanning nodes. A 13B or 34B model on one 8×H100 node with `tp=2` or `tp=4` is dramatically simpler to operate than the same model stretched across two nodes for no memory reason. Cross the boundary only when the memory math forces you to, as covered in [multi-node LLM serving for 100B+ models](/blog/machine-learning/model-serving/multi-node-llm-serving-100b-plus), and when you do, treat the checklist above as the price of admission. For the broader serving-failure surface beyond the distributed layer — timeouts, cascades, capacity — the [troubleshooting LLM serving runbook](/blog/machine-learning/model-serving/troubleshooting-llm-serving-runbook) is the companion, and [tracing and debugging LLM serving](/blog/machine-learning/model-serving/tracing-and-debugging-llm-serving) covers the request-level observability that catches many of these before they page you.

## Key takeaways

- **A broken collective is a hang, not an exception.** Across GPUs and nodes, the dominant failure symptom is silence — 0 tok/s with everything alive — because a barrier waiting for a missing rank is doing exactly what a barrier does.
- **The slowest rank sets the pace for all of them.** $t_{\text{coll}} = \max_i t_i + t_{\text{comm}}$: one degraded or OOM-close GPU does not slow itself by 10%, it stalls the entire tensor-parallel group. Find the straggler, not the victim.
- **Fork on startup vs mid-serve first.** Startup failures are deterministic config problems (NICs, ports, world size, placement groups); mid-serve failures are dynamic (OOM, degraded GPU, desync). The two have nearly disjoint causes.
- **`NCCL_DEBUG=INFO` is your most valuable tool.** The single line `NET/IB` versus `NET/Socket` distinguishes a healthy fabric from a silent TCP fallback, and it confirms or rules out the two most expensive incidents in seconds.
- **`py-spy dump` works on a live wedged process.** No restart required — dump every rank, find the one blocked somewhere other than `ncclAllReduce`, and you have your culprit.
- **Capture state before you restart a mid-serve failure.** `py-spy`, `rank_mem.sh`, `nvidia-smi`, and `dmesg` during the stall are the only evidence you get; a restart clears the memory pressure and the mystery recurs.
- **Container flags are not optional.** `--ipc=host --shm-size --ulimit memlock=-1` prevent the entire shared-memory and RDMA-fallback failure class, and their absence is invisible until you run at the real `tp` degree.
- **Tensor parallelism stays inside the node.** NVLink is ~20× faster than the inter-node fabric; put a TP all-reduce on the slow link and the GPUs sit idle on the wire. Pipeline and data parallelism cross the boundary, TP never does.
- **Leave memory headroom and tune the timeout down.** `--gpu-memory-utilization 0.90` absorbs per-rank imbalance; a 300 s NCCL timeout converts a silent 10-minute wedge into a loud, recoverable crash.

## Further reading

- **vLLM, "Anatomy of a vLLM"** (vLLM blog, 2025) — the definitive walk-through of `AsyncLLM`, `EngineCore`, `MultiProcExecutor`, `rpc_broadcast_mq`/`worker_response_mq`, the ZMQ front-end sockets, and the `DPCoordinator`. Read it to map every failure above onto the real components.
- **vLLM documentation, "Distributed Inference and Serving" and "Troubleshooting"** — the official guidance on `--tensor-parallel-size`, `--pipeline-parallel-size`, `--distributed-executor-backend ray`, `NCCL_DEBUG`, `VLLM_HOST_IP`, and the known-hang checklist.
- **NVIDIA, "NCCL Environment Variables" and `nccl-tests`** — the reference for `NCCL_SOCKET_IFNAME`, `NCCL_IB_*`, GPUDirect RDMA, and the `all_reduce_perf` benchmark you should run before vLLM.
- **Ray documentation, "Placement Groups" and "Observability"** — how `ray status`, the dashboard, and `ray list` diagnose pending placement groups and dead actors.
- **Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention"** (SOSP 2023) — the paper behind vLLM's KV-cache block manager, which is where the one-rank-OOM incidents ultimately live.
- Within this series: [vLLM distributed architecture anatomy](/blog/machine-learning/model-serving/vllm-distributed-architecture-anatomy), [running vLLM distributed in production](/blog/machine-learning/model-serving/running-vllm-distributed-in-production), [multi-node LLM serving for 100B+ models](/blog/machine-learning/model-serving/multi-node-llm-serving-100b-plus), [observability for LLM serving](/blog/machine-learning/model-serving/observability-for-llm-serving), and the [troubleshooting LLM serving runbook](/blog/machine-learning/model-serving/troubleshooting-llm-serving-runbook).

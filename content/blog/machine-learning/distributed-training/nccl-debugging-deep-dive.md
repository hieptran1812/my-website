---
title: "NCCL Debugging Deep Dive: Reading the Logs That Nobody Reads"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "NCCL is the black box under every all-reduce: it picks the algorithm, picks the transport, and is the thing that silently falls back to TCP and halves your throughput. This is how to read its logs line by line, decode which transport each connection chose, wield the env vars that matter, and turn a slow-and-mysterious multi-node run into a diagnosis you can prove."
tags:
  [
    "distributed-training",
    "multi-gpu",
    "nccl",
    "multi-node",
    "pytorch",
    "debugging",
    "infiniband",
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

The run was correct. That was the maddening part. Sixteen H100s across two nodes, loss coming down clean, checkpoints saving, not a single error in the logs — and a model-FLOPs utilization of `18%` when the same code on a single 8-GPU node hit `43%`. Adding the second node had made each GPU *slower*. Every dashboard was green. `nvidia-smi` showed high utilization. The InfiniBand link tested fine with `ib_write_bw`. And yet the cross-node all-reduce was crawling, dragging the whole run down to less than half the throughput we paid for.

The answer was sitting in a log we had never once opened. Buried in thousands of lines of startup spam, on every rank, was a single string: `NET/Socket`. NCCL — the library that actually moves the gradients between GPUs — had failed to bring up the InfiniBand path and had *silently* fallen back to sending our `100 MB` gradient buckets over TCP sockets, through the kernel network stack, bouncing off host memory on both ends. No error. No warning that rose above `INFO`. Just a quiet decision, logged once at startup, that cost us more than half our compute. One environment variable fixed it, and the run went from `18%` to `43%` MFU in the time it took to restart.

That log is the subject of this post. NCCL (the NVIDIA Collective Communications Library) is the single most important piece of software in your training stack that you have probably never read the output of. It sits under *every* collective — every `all_reduce` in DDP, every `all_gather` and `reduce_scatter` in FSDP, every `all_to_all` in an MoE — and on each of those calls it makes two decisions that determine your throughput: **which algorithm** to run (ring, tree, CollNet) and **which transport** to move the bytes over (NVLink, PCIe peer-to-peer, shared memory, InfiniBand with GPUDirect RDMA, or — the villain of our story — a TCP socket). When a distributed job is slow, hangs, or falls back, the truth is in the NCCL log, and almost nobody reads it. Figure 1 is the shape of what NCCL is actually doing on every collective, and the rest of this post is how to read the evidence it leaves behind.

![diagram showing a single all-reduce call entering NCCL which independently selects an algorithm and a transport before the two choices combine into the actual bytes moved on the wire](/imgs/blogs/nccl-debugging-deep-dive-1.webp)

By the end you will be able to: turn on `NCCL_DEBUG=INFO` and read the startup log line by line — the topology it detected, the rings and trees it built, and the transport it chose for every single connection; instantly tell a healthy `P2P/CUMEM` or `NET/IB/GDRDMA` line from a slow `NET/Socket` fallback; reach for the right environment variable out of the two dozen that matter, knowing exactly *when* each one earns its place; understand why NCCL picks ring for big messages and tree for small ones, and when to override it; read a flight-recorder dump to find the one rank stuck on the wrong collective; and recognize the half-dozen failure signatures that account for most of the NCCL tickets ever filed. This is the "how you debug it" wall of the [Distributed Training in the Trenches](/blog/machine-learning/distributed-training/why-distributed-training) series — the layer beneath the collectives we built in [collectives from scratch](/blog/machine-learning/distributed-training/collectives-from-scratch) and the interconnects we mapped in [the interconnect physics](/blog/machine-learning/distributed-training/the-interconnect-physics).

## What NCCL actually is, and why its log is ground truth

When you call `dist.all_reduce(grad)` in PyTorch, almost nothing happens in Python. The call hands a pointer and a size to NCCL and returns. NCCL is where the real work lives: it is a C++/CUDA library, shipped as `libnccl.so`, that implements the collective operations — all-reduce, all-gather, reduce-scatter, broadcast, all-to-all — as GPU kernels that coordinate across every rank in a process group. It is the thing that knows your cluster's wiring, decides how to route bytes across it, launches the kernels, and hands control back to PyTorch when the reduction is done. PyTorch's `ProcessGroupNCCL` is a thin wrapper; DeepSpeed and Megatron-LM call the same library underneath. When people say "the all-reduce is slow," what they almost always mean is "NCCL made a choice I did not see."

It helps to hold the software stack in one glance. Your training loop calls a PyTorch collective; PyTorch calls NCCL; NCCL selects an algorithm and a transport; the transport drives the hardware — NVLink and NVSwitch inside a node, InfiniBand or RoCE between nodes, and the CUDA driver underneath all of it. Every layer can be the bottleneck, but only one layer *logs its decisions*, and that is NCCL. The GPU's utilization counter tells you the SMs are busy; it does not tell you whether they are busy doing a `900 GB/s` NVLink copy or spinning on a `1 GB/s` socket transfer. Only the NCCL log distinguishes those, which is why it is the ground truth for every throughput and hang investigation.

The two decisions in Figure 1 are worth pinning down because everything downstream is about reading how they were resolved.

**The algorithm** is *how* the ranks combine their data. A ring all-reduce passes shards around a logical ring of GPUs; a tree all-reduce reduces up a tree and broadcasts back down; CollNet and NVLS offload part of the reduction into the network switch itself. Each has a different latency-versus-bandwidth profile, and NCCL picks one per collective based on message size and the topology it detected. We derive the ring-versus-tree crossover later, and we assembled the ring byte-for-byte in [collectives from scratch](/blog/machine-learning/distributed-training/collectives-from-scratch).

**The transport** is *over what medium* two ranks actually exchange bytes. Between two GPUs on the same NVSwitch fabric, NCCL uses direct peer-to-peer copies over NVLink (logged as `P2P`). Between two GPUs on the same host without a P2P path, it uses shared host memory (`SHM`). Between two nodes it uses the network plugin — InfiniBand or RoCE, ideally with GPUDirect RDMA so the NIC reads GPU memory directly (`NET/IB/GDRDMA`), and if all of that fails, a plain TCP socket (`NET/Socket`). The transport is the single biggest lever on throughput, because the difference between the best and worst choice is not `10%` — it is often `10×`. That is why our `18%`-MFU run was a transport bug and not an algorithm bug.

The reason to read the log rather than infer from throughput is that NCCL's choices are *silent and sticky*. It decides at initialization, logs the decision once at `INFO` verbosity, and then never mentions it again for the entire run. If it made a bad choice — GDR unavailable, wrong NIC, P2P disabled by a driver quirk — nothing errors and nothing warns. The run is correct and slow, forever, until you read the one line that says which transport it picked. Learning to read that line is the highest-leverage debugging skill in multi-node training.

## Turning on the logs: NCCL_DEBUG and its subsystems

By default NCCL is silent. You turn it on with a single environment variable, and the level you choose determines how much you see:

```bash
# The four verbosity levels, least to most.
export NCCL_DEBUG=VERSION   # one line: the NCCL version. Confirms which lib loaded.
export NCCL_DEBUG=WARN      # only warnings/errors. Leave this on in production.
export NCCL_DEBUG=INFO      # the startup story: topology, rings, transports. The workhorse.
export NCCL_DEBUG=TRACE     # per-call firehose. For deep hangs only; drowns everything.
```

`WARN` is what you want running all the time: it costs nothing and it surfaces the loud failures (a NIC that vanished, an unhandled CUDA error). `INFO` is the one you turn on when something is slow or hangs — it prints, once per rank at startup, the entire story of how NCCL set itself up. `TRACE` logs every collective as it happens and is only for chasing a specific hang; on a real run it produces gigabytes and slows everything down.

The problem with `INFO` is volume. Eight ranks each printing a few hundred lines, interleaved on the same stderr, is unreadable. Two variables fix that, and they are the difference between a useful log and noise:

```bash
export NCCL_DEBUG=INFO
# Restrict to the subsystems you care about instead of all of them.
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH
# Write one file per host and pid, so ranks do not interleave. %h=host, %p=pid.
export NCCL_DEBUG_FILE=/logs/nccl.%h.%p.log
```

`NCCL_DEBUG_SUBSYS` filters by subsystem, and knowing the useful ones saves you from drowning:

- **`INIT`** — the setup: version, topology, ring/tree construction, the "Init COMPLETE" line. This is where transport decisions live. Start here.
- **`NET`** — the network plugin: which NIC, IB versus socket, GDR on or off. The subsystem for our socket-fallback bug.
- **`GRAPH`** — the topology graph NCCL built: how it thinks your GPUs and NICs are wired. Useful when placement looks wrong.
- **`P2P`** — peer-to-peer setup between GPUs; whether NVLink paths came up.
- **`COLL`** — per-collective traces; pairs with the flight recorder for hang debugging. Verbose.
- **`TUNING`** — the algorithm/protocol NCCL chose for each message size. Where you confirm ring versus tree.
- **`ENV`** — echoes which `NCCL_*` variables you actually set, which catches typos (a misspelled variable is silently ignored).

`NCCL_DEBUG_FILE` is the unsung hero. Without it, all ranks write to the same stderr and you get a shredded log where line 3 of rank 0 sits between line 1 and line 2 of rank 5. With it, you get one clean file per process — `nccl.gpu-a3.47021.log` — that reads top to bottom like a story. Always set it when you turn on `INFO`.

A note that saves an afternoon: NCCL only reads its environment variables at `init_process_group` time, and it *silently ignores* any it does not recognize. If you set `NCCL_SOCKET_IFNAME` but spelled it `NCCL_SOCKET_IFACE`, nothing complains — the variable simply does nothing. Turn on `NCCL_DEBUG_SUBSYS=ENV` once and confirm NCCL echoes back the variables you think you set.

## Reading an NCCL_DEBUG=INFO log line by line

Here is the core skill. Below is a lightly trimmed, real-shaped `INFO` log from rank 0 of a healthy single-node, 8-GPU run — the kind of thing that scrolls past you every time a job starts. Most people never look at it. Read it once, slowly, and it becomes a checklist you can scan in ten seconds. Figure 2 lays out the same log as an ordered sequence of stages, so you know what *should* appear and in what order.

![timeline of the stages an NCCL initialization log prints from bootstrap through topology detection ring construction and per-connection transport selection to init complete](/imgs/blogs/nccl-debugging-deep-dive-2.webp)

```console
gpu-a3:47021:47088 [0] NCCL INFO Bootstrap : Using eth0:10.4.2.31<0>
gpu-a3:47021:47088 [0] NCCL INFO cudaDriverVersion 12040
gpu-a3:47021:47088 [0] NCCL INFO NCCL version 2.20.5+cuda12.4
gpu-a3:47021:47095 [0] NCCL INFO NET/IB : Using [0]mlx5_0:1/IB [1]mlx5_1:1/IB [RO]; OOB eth0:10.4.2.31<0>
gpu-a3:47021:47095 [0] NCCL INFO Using network IB
gpu-a3:47021:47095 [0] NCCL INFO comm 0x55f2a1 rank 0 nranks 8 cudaDev 0 nvmlDev 0 busId 1b000 - Init START
gpu-a3:47021:47095 [0] NCCL INFO NVLS multicast support is available on dev 0
gpu-a3:47021:47095 [0] NCCL INFO Channel 00/08 : 0 1 2 3 4 5 6 7
gpu-a3:47021:47095 [0] NCCL INFO Channel 01/08 : 0 1 2 3 4 5 6 7
gpu-a3:47021:47095 [0] NCCL INFO Ring 00 : 0[0] -> 1[1] via P2P/CUMEM
gpu-a3:47021:47095 [0] NCCL INFO Ring 00 : 7[7] -> 0[0] via P2P/CUMEM
gpu-a3:47021:47095 [0] NCCL INFO Connected all rings
gpu-a3:47021:47095 [0] NCCL INFO Connected all trees
gpu-a3:47021:47095 [0] NCCL INFO threadThresholds 8/8/64 | 64/8/64 | 512 | 512
gpu-a3:47021:47095 [0] NCCL INFO 8 coll channels, 8 p2p channels, comm 0x55f2a1 - Init COMPLETE
```

Walk it top to bottom, because every line answers a question you will eventually need to ask.

**The prefix** `gpu-a3:47021:47095 [0]` is `hostname:pid:tid [cudaDevice]`. It tells you which host, process, and GPU emitted the line — essential when you are diffing logs across ranks to find the odd one out. The `[0]` is the local CUDA device, not the global rank.

**`Bootstrap : Using eth0`** is the very first network NCCL brings up — the *out-of-band* channel it uses only to exchange setup information (addresses, unique IDs) before the fast transports exist. This is TCP, and it *should* be on your management interface. If bootstrap picks the wrong interface — a Docker bridge, a slow management NIC — ranks can fail to find each other and you get a hang before training even starts. This is the interface `NCCL_SOCKET_IFNAME` controls.

**`NCCL version 2.20.5+cuda12.4`** and **`cudaDriverVersion 12040`** are the first thing to check when something is inexplicable. A version mismatch between nodes — one host on NCCL 2.18, another on 2.20 — produces bizarre, hard-to-reproduce failures. Confirm every node prints the same version. This line alone resolves a surprising fraction of "it works on node A but not node B" tickets.

**`NET/IB : Using [0]mlx5_0:1/IB [1]mlx5_1:1/IB`** is the money line for cross-node runs. It says NCCL found the InfiniBand plugin and is using two HCAs, `mlx5_0` and `mlx5_1`, both on IB port 1. **`Using network IB`** confirms the network backend is InfiniBand. If instead you see `NET/Socket : Using [0]eth0` and `Using network Socket`, stop — that is the fallback that will halve your throughput, and it is exactly the bug from the intro.

**`Init START`** with `rank 0 nranks 8` confirms the process group size. If `nranks` is not what you launched with, your rendezvous is wrong and ranks are in different groups.

**`NVLS multicast support is available`** tells you NVLink SHARP (in-switch reduction) is usable on this hardware — a good sign on H100/NVSwitch systems, because NVLS is often the fastest algorithm for medium messages.

**`Channel 00/08 ... Channel 01/08`** is the channel count: NCCL built 8 channels (parallel rings/trees) to saturate the available bandwidth. More channels means more parallelism across the links. On a healthy 8-GPU NVSwitch node you expect a handful to a couple dozen; a channel count that collapses to 1 or 2 when you expected more usually means NCCL could not find the fast paths it wanted.

**The `Ring` lines are the heart of the transport story.** `Ring 00 : 0[0] -> 1[1] via P2P/CUMEM` says: on channel 0, rank 0 sends to rank 1 using peer-to-peer over CUDA memory — that is **NVLink**, the fast intra-node path. This is what you want to see inside a node. The transport tag after `via` is the whole diagnosis:

- **`via P2P/CUMEM`** or **`via P2P/direct`** — direct GPU-to-GPU over NVLink/NVSwitch. Fastest. Intra-node ideal.
- **`via SHM/direct`** — through shared host memory. Intra-node but *not* using NVLink — slower, and a sign that P2P was disabled or unavailable (an IOMMU setting, `NCCL_P2P_DISABLE`, a topology without NVLink between that pair).
- **`via NET/IB/0/GDRDMA`** — cross-node over InfiniBand with GPUDirect RDMA: the NIC reads GPU memory directly, no host bounce. Cross-node ideal.
- **`via NET/IB/0`** (no `GDRDMA`) — InfiniBand, but staging through host memory because GDR is off or unavailable. Works, but leaves bandwidth on the table.
- **`via NET/Socket/0`** — TCP through the kernel. The slow fallback. If you see this cross-node, that is your bug.

**`Connected all rings` / `Connected all trees`** means both the ring and tree topologies came up cleanly for every channel. If the log stalls here, or one rank never prints it, you have a connection failure — a link that would not come up, a NIC that hung — and that rank will block the group.

**`Init COMPLETE`** with `8 coll channels, 8 p2p channels` is the finish line: this communicator is ready. Every rank must reach this line. A rank that prints everything up to `Connected all rings` but never `Init COMPLETE` is stuck bringing up a transport — and on a hung startup, that is the rank to investigate.

That is the entire skill. Turn on `INFO`, open one rank's file, and scan for four things: the **version** (consistent?), the **network** (`IB` or `Socket`?), the **ring transports** (`P2P`/`GDRDMA` or `Socket`?), and **`Init COMPLETE`** (did every rank finish?). Ninety percent of NCCL debugging is those four checks. The single most useful command in this whole post is the grep that pulls the transport decisions straight out of the log:

```bash
# What transport did NCCL actually choose? The one-line diagnosis.
grep -E "via (P2P|SHM|NET)" /logs/nccl.*.log | sort | uniq -c
```

On the healthy run that prints a pile of `via P2P/CUMEM` (intra-node) and `via NET/IB/0/GDRDMA` (cross-node). On the broken run it prints `via NET/Socket/0`, and you have your answer in one command instead of one afternoon.

Two more lines in the log are worth knowing, because they explain performance nuances once transport is healthy. The **channel count** — `8 coll channels` in our log — is how many parallel rings/trees NCCL runs to fill the links; on an NVSwitch node NCCL will build more channels to saturate the aggregate bandwidth, and a channel count that unexpectedly collapses is a sign it could not find the paths it wanted. The **protocol**, which the `TUNING` subsystem logs, is one of `LL`, `LL128`, or `Simple`: `LL` ("low latency") and `LL128` trade a little bandwidth for much lower latency on small messages by using lightweight flags instead of full synchronization, while `Simple` is the full-bandwidth protocol for large transfers. Like the algorithm, NCCL picks the protocol per message size, and you almost never set `NCCL_PROTO` by hand — but seeing `LL128` on your small gradient-norm all-reduce and `Simple` on your big bucket all-reduce confirms NCCL is tuning correctly.

When the bug might live on a specific host, you want *every* rank's log side by side. Collect them with a fan-out and then diff the transport decisions across the cluster in one pass:

```bash
# Gather every rank's NCCL log across a SLURM allocation, then compare transports.
srun --overlap --ntasks-per-node=1 bash -c \
  'cat /logs/nccl.$(hostname).*.log' > /tmp/all_nccl.log 2>&1
# If any host disagrees with the majority, it prints here.
grep -hoE "Using network (IB|Socket)" /tmp/all_nccl.log | sort | uniq -c
grep -hoE "NCCL version [0-9.]+" /tmp/all_nccl.log | sort | uniq -c
```

If the second `uniq -c` shows two different NCCL versions, you have found a version-mismatch bug before it has a chance to produce a mysterious crash. If the first shows `15 Using network IB` and `1 Using network Socket`, you have found the one misconfigured node in a sixteen-node job. This "collect per-rank, find the odd one out" pattern is the whole workflow, and it is the same pattern the flight recorder automates for runtime hangs.

## The environment variables that matter

NCCL exposes dozens of environment variables. You will use about a dozen with any regularity, and the skill is knowing *when* each earns its place rather than cargo-culting a block of exports from someone's Slack. Figure 3 is the reference matrix I keep pinned; the fuller table below it adds the ones that come up less often but matter when they do.

![reference matrix of the most useful NCCL environment variables showing what each controls when to set it and a typical value](/imgs/blogs/nccl-debugging-deep-dive-3.webp)

| Variable | What it controls | When to set it | Typical value |
|---|---|---|---|
| `NCCL_DEBUG` | Log verbosity | Always `WARN`; `INFO` when debugging | `INFO` |
| `NCCL_DEBUG_SUBSYS` | Which subsystems log | With `INFO`, to cut noise | `INIT,NET,GRAPH` |
| `NCCL_DEBUG_FILE` | Per-process log file | Always, with `INFO` | `/logs/nccl.%h.%p.log` |
| `NCCL_SOCKET_IFNAME` | Bootstrap/socket NIC | Multi-NIC hosts; wrong iface picked | `eth0` or `=^docker0` |
| `NCCL_IB_HCA` | Which IB devices to use | Pin the right HCAs on multi-NIC hosts | `mlx5_0,mlx5_1` |
| `NCCL_IB_DISABLE` | Turn IB off entirely | Debugging: force the socket path | `0` (never `1` in prod) |
| `NCCL_NET_GDR_LEVEL` | GPUDirect RDMA reach | When GDR is off but should be on | `PHB` or `SYS` |
| `NCCL_P2P_DISABLE` | Turn NVLink P2P off | Debugging a P2P/IOMMU issue only | `0` |
| `NCCL_SHM_DISABLE` | Turn shared-mem transport off | Debugging a SHM issue only | `0` |
| `NCCL_ALGO` | Force ring/tree/CollNet | Small-message latency tuning | `Tree` or `Ring` |
| `NCCL_PROTO` | Force LL/LL128/Simple | Rarely; last-resort tuning | (leave unset) |
| `NCCL_IB_GID_INDEX` | RoCE GID selection | RoCE clusters where default GID is wrong | `3` |
| `NCCL_CROSS_NIC` | Allow rings to cross NICs | Multi-rail IB tuning | `0`/`1`/`2` |
| `TORCH_NCCL_ASYNC_ERROR_HANDLING` | Watchdog aborts on timeout | Always on | `1` |
| `TORCH_NCCL_TRACE_BUFFER_SIZE` | Flight-recorder ring buffer | Always on in prod | `2000` |
| `TORCH_NCCL_DUMP_ON_TIMEOUT` | Dump flight recorder on hang | Always on with the buffer | `1` |

A few of these deserve more than a table cell because they are the ones that actually fix things:

**`NCCL_SOCKET_IFNAME`** picks the interface for bootstrap and for the socket transport. On a host with several interfaces — a fast data NIC, a slow management NIC, a Docker bridge, a loopback — NCCL's auto-selection sometimes grabs the wrong one, and ranks either fail to rendezvous or route their out-of-band traffic over a slow link. You can name the interface (`eth0`) or, more robustly, *exclude* the bad ones with the `^` prefix: `NCCL_SOCKET_IFNAME==^docker0,lo` says "use anything except the Docker bridge and loopback." This is often the fix for a startup hang on a cloud VM with a virtual bridge.

**`NCCL_IB_HCA`** pins which InfiniBand adapters NCCL uses. On an 8-GPU node with 8 NICs, letting NCCL guess the GPU-to-NIC affinity is usually fine, but when placement is wrong — a GPU routing to a NIC three PCIe hops away — pinning the HCAs (`NCCL_IB_HCA=mlx5_0,mlx5_1,...`) restores locality. It is also how you fix "IB not found" when the plugin picked the management HCA instead of the data-fabric ones.

**`NCCL_NET_GDR_LEVEL`** controls how aggressively NCCL uses GPUDirect RDMA — the feature that lets the NIC DMA straight into GPU memory without a host-memory bounce. If your cross-node lines say `NET/IB/0` without the `GDRDMA` suffix, GDR is off, and you are paying a host-copy tax on every cross-node byte. Setting `NCCL_NET_GDR_LEVEL=PHB` (or `SYS` on some topologies) tells NCCL to use GDR even when the GPU and NIC are a little further apart in the PCIe tree. Confirm it worked by re-reading the log for the `GDRDMA` tag.

**`NCCL_P2P_DISABLE`, `NCCL_SHM_DISABLE`, `NCCL_IB_DISABLE`** are *debugging* switches, not production settings. Their value is diagnostic: if you suspect a specific transport is broken, disable it and see whether the problem moves. `NCCL_P2P_DISABLE=1` forces intra-node traffic onto shared memory instead of NVLink — if a flaky run suddenly becomes stable, you have localized the fault to the P2P/NVLink path (a bad link, an IOMMU setting). Then you turn the switch back off and fix the real cause. Leaving any of these disabled in production is throwing away the fast path on purpose.

The three `TORCH_NCCL_*` variables belong to PyTorch's wrapper, not NCCL itself, and they are the hang-debugging toolkit. `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` makes the watchdog abort the job on a collective timeout instead of hanging forever; `TORCH_NCCL_TRACE_BUFFER_SIZE=2000` and `TORCH_NCCL_DUMP_ON_TIMEOUT=1` enable the flight recorder we use below. Turn all three on and leave them on — they cost almost nothing and turn a future 3 a.m. mystery into a mechanical diagnosis. The full hang-debugging escalation ladder is in the sibling war story [the NCCL timeout that hung the job](/blog/machine-learning/distributed-training/the-nccl-timeout-that-hung-the-job).

## How NCCL picks a transport

The transport decision — the `via ...` tag in every ring line — is not random. NCCL walks a topology graph it builds at startup and picks the fastest usable path between each pair of ranks. Figure 4 is that decision as a tree, and internalizing it lets you predict what the log *should* say before you read it, so a wrong choice jumps out.

![decision tree showing how NCCL chooses a transport based on whether the peer rank is on the same node or a different node down to NVLink shared memory InfiniBand or a socket fallback](/imgs/blogs/nccl-debugging-deep-dive-4.webp)

The first fork is **where the peer rank lives**. If the two ranks are on the *same node*, NCCL prefers a direct GPU-to-GPU copy over NVLink or NVSwitch — this is the `P2P/CUMEM` (or `P2P/direct`) path, and on an H100 SXM node it moves data at up to roughly `900 GB/s` aggregate per GPU. If there is no P2P path between that pair — no NVLink between those specific GPUs, or P2P disabled — it falls back to `SHM`, copying through shared host memory, which is bounded by PCIe and host-memory bandwidth (order tens of GB/s) and is meaningfully slower.

If the peer is on a *different node*, NCCL uses the network plugin. The best case is InfiniBand (or RoCE) with GPUDirect RDMA — `NET/IB/GDRDMA` — where the NIC reads GPU memory directly and moves data at the link rate, around `25 GB/s` for a single `200 Gb/s` HDR port (multiple rails aggregate higher). If GDR is unavailable, it still uses IB but stages through host memory (`NET/IB`, no GDR tag), losing bandwidth to the extra copy. And if InfiniBand itself cannot be brought up — no IB devices found, the plugin failed to load, a misconfigured fabric — it falls all the way back to `NET/Socket`: TCP through the kernel network stack, the slowest path by an order of magnitude, and the one that silently wrecked our intro run.

The failure mode that costs the most is not the socket fallback being *chosen loudly* — it is chosen *silently*. NCCL treats the socket transport as a legitimate last resort, not an error, because on a cluster with no InfiniBand at all, sockets are the only option and the run should still work. So it drops to sockets and logs it exactly once, at `INFO`, and never complains again. On a cluster that *does* have IB, that silent fallback means something is misconfigured — the wrong NIC, a firewall, a driver mismatch, GDR blocked — and the fix is almost always one of `NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`, or `NCCL_NET_GDR_LEVEL`. The tree in Figure 4 is the map: read the transport tag, find which leaf you landed on, and the parent nodes tell you which knob moves you up to a faster path.

There is a subtlety worth stating because it trips people up. NCCL builds separate rings for intra-node and inter-node segments and stitches them together, so a single all-reduce on a multi-node job uses *both* NVLink (within each node) and IB (between nodes) in the same operation. A healthy multi-node log therefore shows a *mix* — mostly `P2P/CUMEM` lines with a few `NET/IB/GDRDMA` lines at the node boundaries. If you see `NET/Socket` anywhere in that mix, even one line, the cross-node hop is on TCP and the whole collective moves at the speed of its slowest segment. One bad line poisons the run.

### Confirming the wiring with nvidia-smi topo -m

When the log shows a transport you did not expect — `SHM` between two GPUs you thought had NVLink, or IB without GDR — the next question is *what is the hardware actually wired like*, and the answer is `nvidia-smi topo -m`. It prints the GPU-to-GPU and GPU-to-NIC connectivity matrix that NCCL itself reads to build its topology graph:

```console
$ nvidia-smi topo -m
        GPU0  GPU1  GPU2  GPU3  NIC0  NIC1  CPU Affinity
GPU0     X    NV18  NV18  NV18  PXB   SYS   0-31
GPU1    NV18   X    NV18  NV18  PXB   SYS   0-31
GPU2    NV18  NV18   X    NV18  SYS   PXB   32-63
GPU3    NV18  NV18  NV18   X    SYS   PXB   32-63

Legend:  X = self   NV# = NVLink (# links)   PXB = PCIe switch hops
         SYS = across the CPU/QPI link (slowest)   PIX = single PCIe bridge
```

Read it like a distance matrix. `NV18` between every GPU pair means full NVLink connectivity — exactly what should produce `via P2P/CUMEM` in the NCCL log; if the matrix instead showed `SYS` or `PHB` between two GPUs, `via SHM` would be expected and correct, not a bug. The GPU-to-NIC columns matter just as much: `GPU0` reaching `NIC0` over `PXB` (a PCIe switch, close) but `NIC1` over `SYS` (across the CPU link, far) tells you GPU0 should use NIC0 for GPUDirect RDMA. This is the map behind `NCCL_NET_GDR_LEVEL`: GDR only works when the GPU and NIC are close enough in the PCIe tree, and the level you set (`PIX`, `PXB`, `PHB`, `SYS`) is the maximum distance at which NCCL will still use it. If the log says `NET/IB` without `GDRDMA` and `topo -m` shows the GPU and its NIC are `SYS` apart, that is why — and either raising `NCCL_NET_GDR_LEVEL` or pinning a closer NIC with `NCCL_IB_HCA` is the fix. Reading `topo -m` alongside the NCCL log turns "NCCL made a weird choice" into "NCCL made the *correct* choice given this wiring, and the wiring is what I need to change."

## The algorithms: ring, tree, CollNet, NVLS

Transport is *how the bytes move*; algorithm is *how the ranks combine them*. NCCL ships several, and it picks one per collective based on message size and topology. Understanding the two main ones — ring and tree — is the difference between accepting NCCL's choice and knowing when to override it. Figure 5 contrasts them on the axis that decides between them: message size.

![comparison of ring and tree all-reduce contrasting bandwidth-optimal large-message behavior against latency-optimal small-message behavior](/imgs/blogs/nccl-debugging-deep-dive-5.webp)

### The mechanism: why ring wins big and tree wins small

Model a collective as two costs: a per-hop **latency** $\alpha$ (the time to get a message moving, dominated by fabric and software overhead) and a **bandwidth** term, the payload size $S$ divided by the link bandwidth $B$. The total time for a collective is roughly the number of communication steps times $\alpha$, plus the total bytes each GPU pushes divided by $B$.

**Ring all-reduce** arranges the $N$ GPUs in a logical ring and runs a reduce-scatter followed by an all-gather. As we derived in [collectives from scratch](/blog/machine-learning/distributed-training/collectives-from-scratch), each GPU sends and receives $2\frac{N-1}{N}S$ bytes total, in $2(N-1)$ steps. So its time is approximately

$$T_\text{ring} \approx 2(N-1)\,\alpha \;+\; 2\frac{N-1}{N}\cdot\frac{S}{B}.$$

The bandwidth term is *optimal* — as $N$ grows, $2\frac{N-1}{N} \to 2$, so each GPU moves essentially $2S/B$ regardless of how many GPUs there are. That is why ring is the bandwidth champion: it does not get more expensive per byte as you scale. But the latency term grows *linearly* with $N$ — $2(N-1)$ hops — so for small messages, where $\alpha$ dominates and the payload is trivial, ring's cost is proportional to the number of GPUs. On 64 GPUs a tiny message pays 126 hops of latency for almost no data.

**Tree all-reduce** reduces up a binary tree and broadcasts back down, so it completes in about $2\log_2 N$ steps instead of $2(N-1)$:

$$T_\text{tree} \approx 2\log_2 N\,\alpha \;+\; (\text{a larger bandwidth constant})\cdot\frac{S}{B}.$$

The latency term is now *logarithmic* in $N$ — 12 steps for 64 GPUs instead of 126 — which is a massive win when latency dominates. The price is a worse bandwidth constant: the tree does not spread the payload as evenly as the ring, so for large messages it moves more bytes per GPU than the ring does. The two curves cross. Below some message size (latency-bound), tree wins; above it (bandwidth-bound), ring wins. NCCL knows this and *automatically* picks tree for small messages and ring for large ones, using an internal tuning model calibrated per architecture.

This is why you rarely need to set `NCCL_ALGO` — NCCL's default is right for the common case. You override it when you know something NCCL's model does not: a workload dominated by many tiny collectives (force `Tree`), or a large-message benchmark where you want to pin `Ring` to measure peak bandwidth. Confirm which one it used with `NCCL_DEBUG_SUBSYS=TUNING`, which logs the algorithm and protocol chosen per size bucket.

```bash
# Force the tree algorithm (latency-optimal) for a small-message-heavy workload,
# and confirm the choice landed with the TUNING subsystem.
export NCCL_ALGO=Tree
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,TUNING
# ...launch; then grep the log for the algorithm NCCL selected per message size.
```

### CollNet and NVLS: pushing reduction into the network

Two more algorithms matter on modern hardware, and both work by offloading arithmetic *out of the GPUs and into the fabric*:

- **CollNet** (backed by NVIDIA SHARP on InfiniBand) performs part of the reduction inside the InfiniBand switches. Instead of every GPU's data touching every other GPU, the switch sums contributions in-network, roughly halving the data each GPU must move for an all-reduce. On a SHARP-enabled fabric it can beat both ring and tree for the mid-to-large message sizes typical of gradient all-reduce. It shows up in the log as CollNet channels and requires the SHARP daemon and a compatible fabric.
- **NVLS** (NVLink SHARP) does the same trick *inside the node* using the NVSwitch's in-network reduction. On H100/NVSwitch systems it is often the fastest intra-node algorithm for medium messages, which is why the `NVLS multicast support is available` line in our log was a good omen. NCCL selects it automatically when the hardware supports it.

You do not usually configure these — you *enable the hardware* (a SHARP-capable fabric, NVSwitch) and let NCCL discover them. The reason to know they exist is diagnostic: if a benchmark shows bandwidth well above what ring's $2S/B$ model predicts, in-network reduction is why, and if you *expected* CollNet/NVLS and the log does not show it, something (a missing SHARP daemon, a `NCCL_ALGO` override) disabled it.

| Algorithm | Latency scaling | Bandwidth | NCCL picks it for | Override |
|---|---|---|---|---|
| Ring | $O(N)$ hops | Optimal ($2S/B$) | Large messages | `NCCL_ALGO=Ring` |
| Tree | $O(\log_2 N)$ hops | Worse constant | Small messages | `NCCL_ALGO=Tree` |
| CollNet (SHARP) | In-network | Beats ring mid-large | SHARP fabric present | needs SHARP daemon |
| NVLS (NVLink SHARP) | In-network | Fastest intra-node mid | NVSwitch present | auto on H100 |

## The flight recorder: which collective each rank was on

Reading the startup log tells you how NCCL *set itself up*. But some failures happen mid-run — a hang at step 4,000 where one rank is executing a different collective than the rest. For that you need a *runtime* record, and PyTorch's **NCCL flight recorder** is exactly that: a per-rank ring buffer of the last $K$ collectives, each with its sequence number, operation type, input/output sizes, and its state — `scheduled`, `started`, or `completed`. When the watchdog fires on a timeout, it dumps every rank's buffer to disk. Figure 6 is how you turn that pile of dumps into a single named culprit.

![diagram of the flight recorder dumping every rank's collective buffer on a watchdog timeout then comparing the highest sequence number per rank to name the one rank that never posted the collective](/imgs/blogs/nccl-debugging-deep-dive-6.webp)

Enable it with the three environment variables from the reference matrix, set before you launch:

```bash
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1              # watchdog aborts on timeout
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000              # keep the last 2000 collectives/rank
export TORCH_NCCL_DUMP_ON_TIMEOUT=1                   # write the buffers when it fires
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/logs/nccl_trace   # dump path prefix, one file per rank
```

The insight that makes the dump readable is the one from the [collectives](/blog/machine-learning/distributed-training/collectives-from-scratch) mechanics: NCCL matches collectives *positionally* by a monotonically increasing sequence number, and a hang means the group is blocked on some sequence number that one rank never reached. So you do not read the dumps line by line — you compare the *highest sequence number each rank reached* and find the one that is behind. Here is a compact reader (the full version, with the escalation ladder around it, lives in [the NCCL timeout post](/blog/machine-learning/distributed-training/the-nccl-timeout-that-hung-the-job)):

```python
# find_stuck_rank.py — compare per-rank flight-recorder dumps and name the rank
# whose highest collective sequence number is behind the majority.
import glob, pickle
from collections import Counter

highest = {}   # rank -> highest seq_id that reached "started" or "completed"
for path in glob.glob("/logs/nccl_trace*"):
    with open(path, "rb") as f:
        trace = pickle.load(f)             # {'rank': r, 'entries': [ {seq_id,state,...} ]}
    rank = trace["rank"]
    highest[rank] = max((e["seq_id"] for e in trace["entries"]
                         if e["state"] in ("started", "completed")), default=-1)

target = Counter(highest.values()).most_common(1)[0][0]   # where the majority is stuck
print(f"group is blocked on collective seq_id={target}")
for rank in sorted(highest):
    behind = "   <-- STUCK: never posted this collective" if highest[rank] < target else ""
    print(f"  rank {rank:>3}: highest seq_id={highest[rank]}{behind}")
```

#### Worked example: the flight recorder pinpoints rank 3

A 64-GPU H100 run froze at step 4,000. Utilization pinned at `99%`, no crash, no traceback — the classic hang signature. The watchdog fired after ten minutes, the flight recorder dumped, and `find_stuck_rank.py` produced this:

```console
group is blocked on collective seq_id=4001
  rank   0: highest seq_id=4001
  rank   1: highest seq_id=4001
  rank   2: highest seq_id=4001
  rank   3: highest seq_id=4000   <-- STUCK: never posted this collective
  rank   4: highest seq_id=4001
  ...
  rank  63: highest seq_id=4001
```

Sixty-three ranks reached all-reduce number 4,001 and blocked waiting for the sixty-fourth. Rank 3's highest was 4,000 — it never posted collective 4,001, so the other sixty-three wait forever. That is the entire diagnosis, produced mechanically, in the seconds it took the dumps to write, at a scale where diffing stacks by eye is hopeless. The *why* rank 3 diverged (an uneven data shard, a rank-varying branch, a swallowed exception) is the next investigation, and the catalog of causes is in [the NCCL timeout that hung the job](/blog/machine-learning/distributed-training/the-nccl-timeout-that-hung-the-job). The point for *this* post is that the flight recorder is a log too — a runtime one — and reading it is the same skill as reading the startup log: find the rank whose record disagrees with the majority.

One caveat: the exact field names in the dump (`seq_id`, `state`, `profiling_name`) drift a little across PyTorch versions. Load one dump interactively with `pickle`, print an entry, and adjust the reader to match. The *idea* — the group is blocked on a sequence number one rank never reached — is stable across versions.

## Common failure signatures and their fixes

Most NCCL tickets are one of a handful of signatures, each with a tell in the log and a specific fix. This is the field guide. The first one gets a full worked example because it is the most common and the most expensive.

#### Worked example: the socket-fallback that halved MFU

Back to the intro. A 16-GPU, two-node H100 run (8× H100 SXM per node, NVLink inside a node, `200 Gb/s` InfiniBand HDR between) was hitting `18%` MFU cross-node versus `43%` single-node. No errors. Here is what turning on the log revealed:

```console
gpu-a3:47021:47095 [0] NCCL INFO NET/IB : No device found.
gpu-a3:47021:47095 [0] NCCL INFO NET/Socket : Using [0]eth0:10.4.2.31<0>
gpu-a3:47021:47095 [0] NCCL INFO Using network Socket
gpu-a3:47021:47095 [0] NCCL INFO Channel 00 : 8[0] -> 0[0] [receive] via NET/Socket/0
```

There it is: `NET/IB : No device found`, then a silent drop to `Using network Socket`, and every cross-node hop `via NET/Socket/0`. NCCL could not see the InfiniBand HCAs — in this case because the container had been launched without the IB devices mapped in and with the wrong interface visible — so it fell back to TCP. The gradient all-reduce, `100 MB` per bucket, was crossing the node boundary over kernel sockets bouncing off host memory instead of over `200 Gb/s` RDMA.

The measurement that proves it is `nccl-tests`, the canonical NCCL benchmark. Run `all_reduce_perf` before and after and read the **bus bandwidth** (`busbw`) column — the effective all-reduce bandwidth, which factors out the algorithm so you can compare fairly:

```bash
# Build and run the NCCL all-reduce benchmark across both nodes.
git clone https://github.com/NVIDIA/nccl-tests && cd nccl-tests
make MPI=1 CUDA_HOME=/usr/local/cuda
# 16 ranks, 8 per node; sweep message sizes from 8 B to 4 GB.
mpirun -np 16 -H gpu-a3:8,gpu-a4:8 \
  -x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=NET \
  ./build/all_reduce_perf -b 8 -e 4G -f 2 -g 1
```

The `busbw` at the largest message size told the whole story: on the socket path it topped out around `1.2 GB/s`; after the fix, on IB with GDR, it reached roughly `24 GB/s` — close to the `200 Gb/s` line rate. The fix was to expose the IB devices to the container and pin them:

```bash
# Make NCCL see and use the InfiniBand HCAs, and exclude the wrong interface.
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NCCL_SOCKET_IFNAME==^docker0,lo      # bootstrap over the real NIC, not the bridge
export NCCL_NET_GDR_LEVEL=PHB               # allow GPUDirect RDMA
# (and: launch the container with --device=/dev/infiniband and the IB drivers mounted)
```

After restarting, the log read `NET/IB : Using [0]mlx5_0 ...`, `Using network IB`, and the cross-node lines said `via NET/IB/0/GDRDMA`. Figure 7 is the before-and-after, and it is the entire argument for reading the log: the run was *correct* the whole time, and more than half the compute was being thrown away silently until someone read the one line that named the transport.

![before and after comparison of a cross-node run falling back to TCP sockets versus the same run pinned to InfiniBand with GPUDirect RDMA showing the jump in all-reduce bandwidth and model FLOPs utilization](/imgs/blogs/nccl-debugging-deep-dive-7.webp)

| Metric | Before (`NET/Socket`) | After (`NET/IB/GDRDMA`) |
|---|---|---|
| Transport (cross-node) | TCP sockets, host bounce | InfiniBand + GPUDirect RDMA |
| all-reduce `busbw` (large msg) | ~`1.2 GB/s` | ~`24 GB/s` |
| MFU (16 GPUs, 2 nodes) | `18%` | `43%` |
| Scaling vs single node | `0.4×` per GPU | ~parity per GPU |
| Fix | `NCCL_IB_HCA` + IB devices in container | (one restart) |

The rest of the field guide, more briefly:

**Socket fallback (`NET/Socket` cross-node).** Covered above. *Tell:* `via NET/Socket` in the log, `busbw` an order of magnitude low. *Fix:* `NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`, expose IB to the container.

**IB found but GDR off (`NET/IB` without `GDRDMA`).** InfiniBand came up, but the NIC is staging through host memory. *Tell:* `via NET/IB/0` with no `GDRDMA` suffix; `busbw` decent but below line rate. *Fix:* `NCCL_NET_GDR_LEVEL=PHB` (or `SYS`); check the GPU-NIC PCIe topology with `nvidia-smi topo -m`.

**Version mismatch across nodes.** *Tell:* different `NCCL version` lines on different hosts; intermittent, host-specific failures. *Fix:* pin one NCCL/PyTorch build across the whole cluster; confirm with the version line.

**P2P disabled intra-node (`SHM` where you expected NVLink).** *Tell:* `via SHM/direct` between GPUs that should have NVLink; intra-node all-reduce slower than expected. *Fix:* check `nvidia-smi topo -m` for `NV#` links; a stray `NCCL_P2P_DISABLE=1` in the environment; an IOMMU/ACS BIOS setting that blocks P2P.

**A hang from mismatched collectives.** *Tell:* utilization pinned at `~100%`, zero progress, one rank missing from the watchdog timeout log. *Fix:* the flight recorder finds the stuck rank; the divergence cause (uneven shard, rank-varying branch) is in [the NCCL timeout post](/blog/machine-learning/distributed-training/the-nccl-timeout-that-hung-the-job).

**"unhandled cuda error" / "remote process exited".** *Tell:* one rank prints a CUDA error or a peer-disconnect; the others print a NCCL communication error because their peer vanished. *Fix:* this is a *victim* message — the real fault is on the rank that died first (an OOM, an ECC error, a segfault). Find the earliest-timestamped error across all rank logs; that rank is the culprit, the rest are collateral. `NCCL_DEBUG=WARN` plus per-rank log files (`NCCL_DEBUG_FILE`) is what lets you order the failures in time.

The unifying discipline is the same as reading the startup log: **collect one log file per rank, then find the rank whose story disagrees with the majority.** Whether it is a socket line, a version string, a missing `Init COMPLETE`, or a stuck sequence number, the bug is almost always the one rank (or one connection) that differs.

## Measuring honestly: benchmarks and confounds

Two rules keep you from fooling yourself when you benchmark a NCCL fix.

**Use `nccl-tests`, and read `busbw`, not `algbw`.** The `all_reduce_perf` tool reports two bandwidths: `algbw` (bytes divided by time, the raw rate) and `busbw` (the "bus bandwidth," which normalizes for the algorithm's data movement so you can compare across GPU counts and operations). For all-reduce, `busbw` is what you compare against the hardware line rate, because ring all-reduce inherently moves $2\frac{N-1}{N}S$ bytes and `busbw` accounts for that. Comparing `algbw` across different world sizes will mislead you.

**Sweep message sizes; do not benchmark at one size.** The ring-versus-tree crossover means the *shape* of the bandwidth curve matters. A fix that helps large messages (the GDR path) might do nothing for small ones, and vice versa. Run the full `-b 8 -e 4G -f 2` sweep and look at where your real gradient bucket sizes land (typically `10–100 MB`), because that is the regime your training actually lives in. The output makes the curve obvious — the `busbw` column climbs with message size until it plateaus at the fabric's ceiling:

```console
#       size    count   type   time      algbw    busbw
#        (B)  (elts)          (us)      (GB/s)   (GB/s)
       32768    8192  float    31.2      1.05     1.84
     1048576  262144  float    98.4     10.66    18.65
    33554432 8388608  float   1620.5    20.71    36.24
   536870912  ...     float  25100.3    21.39    37.43
```

Read the bottom rows: at large messages the `busbw` plateaus near the hardware ceiling (here an intra-node NVSwitch path), while the tiny `32 KB` row is latency-bound and nowhere near it. That plateau is the number you compare before and after a transport fix; the small-message rows are where algorithm and protocol choice (tree, `LL128`) matter most.

And the confounds specific to NCCL benchmarking:

- **Warm up.** The first collective on a communicator pays for ring/tree setup and CUDA context creation. `nccl-tests` has warm-up iterations built in; if you roll your own timing, discard the first several iterations and `torch.cuda.synchronize()` before you start and stop the clock.
- **Pin the clocks.** GPU boost clocks and thermal throttling shift bandwidth run to run. For a clean comparison, lock clocks (`nvidia-smi -lgc`) or at least keep the node cool and idle between runs.
- **Isolate the variable.** Change one env var at a time and re-read the log to confirm the transport actually changed. A "fix" that also happened to reduce contention from a neighbor job is not a fix you can trust. The log is the proof the transport changed; the benchmark is the proof it mattered.

## Case studies and real numbers

NCCL's behavior is documented, and the numbers below are drawn from vendor specs and public benchmarks rather than invented; where a figure is approximate, it is flagged.

- **NVLink versus InfiniBand bus bandwidth.** On an 8-GPU H100 SXM node, `nccl-tests` all-reduce `busbw` commonly lands in the mid-hundreds of `GB/s` intra-node (NVSwitch/NVLink), while a single `200 Gb/s` HDR InfiniBand rail caps a cross-node all-reduce near `~24 GB/s` of `busbw` — roughly a `10×` gap between the intra-node and single-rail inter-node paths. This gap is *the* reason you saturate one node before going multi-node, and the reason a socket fallback (another order of magnitude down) is so catastrophic. The physics behind the gap is in [the interconnect physics](/blog/machine-learning/distributed-training/the-interconnect-physics).
- **In-network reduction (SHARP/NVLS).** NVIDIA's SHARP offloads reduction into the InfiniBand switches; published NVIDIA material reports meaningful all-reduce speedups on SHARP-enabled fabrics for the mid-to-large messages typical of gradient sync, because each GPU moves roughly half the data. NVLS brings the same in-switch reduction inside the NVSwitch node. Both are why a well-configured cluster can beat the naive ring bandwidth model — and why the `NVLS multicast support` log line is worth looking for.
- **The socket fallback in the wild.** The `NET/Socket` silent fallback is common enough that NVIDIA's own NCCL troubleshooting guide leads with it: check the network the log reports, pin `NCCL_IB_HCA` and `NCCL_SOCKET_IFNAME`, and verify GDR. The failure is nearly always environmental — a container missing `/dev/infiniband`, a firewall on the IB subnet, the wrong interface auto-selected — rather than a NCCL bug. The tell is always the same one line in the log.
- **The flight recorder's reason to exist.** PyTorch added the NCCL flight recorder precisely because, at thousands of ranks, stack-diffing to find a stuck rank did not scale. The design — a per-rank ring buffer of recent collectives with their state, dumped on timeout — is built around the exact diagnosis we ran above: find the rank whose sequence number is behind. It exists because reading NCCL's runtime record is the only tractable way to debug a hang at scale.

The through-line: the fast paths (NVLink, IB+GDR, in-network reduction) and the slow fallbacks (SHM, socket) differ by *orders of magnitude*, the choice between them is made silently at startup, and the only place the choice is recorded is the log almost nobody reads.

## When to reach for these knobs, and when not

Every NCCL environment variable is a way to override a default that is usually right. Reach for them deliberately, not reflexively.

**Do read the log (`NCCL_DEBUG=INFO`) whenever** a multi-node run is slower than expected, hangs at startup, or scales worse than single-node. It costs one restart and answers the transport question definitively. This should be the *first* thing you do, before profiling, before tuning batch size — the log will tell you in thirty seconds whether you have a transport bug or a real compute problem.

**Do pin `NCCL_IB_HCA` and `NCCL_SOCKET_IFNAME`** on any multi-NIC host or any containerized environment, because auto-selection is where the silent socket fallback comes from. On a known-good cluster these belong in your launch template permanently.

**Do turn on `NCCL_NET_GDR_LEVEL`** if the log shows IB without the `GDRDMA` tag — you are leaving bandwidth on the table otherwise.

**Do not set `NCCL_ALGO` or `NCCL_PROTO`** unless you have measured that NCCL's automatic choice is wrong for your message-size distribution. The tuning model is calibrated per architecture and is right for the overwhelming majority of workloads; a hand-forced `Ring` on a small-message job can be *slower* than the default. Override only with a benchmark in hand.

**Do not leave the `*_DISABLE` switches on.** `NCCL_P2P_DISABLE`, `NCCL_SHM_DISABLE`, and `NCCL_IB_DISABLE` are diagnostic tools for localizing a fault by process of elimination. Left on, they force NCCL off its fast paths. Use them to *find* the broken transport, fix the underlying cause, then turn them off.

**Do not tune NCCL before you have read its log.** The most common mistake is reaching for `NCCL_ALGO` or buffer-size variables to "speed up the all-reduce" when the actual problem is a socket fallback that no algorithm choice can fix. Read the transport first. If it says `NET/Socket` on a cluster with IB, no amount of algorithm tuning will save you — you have an environment bug, and the fix is a device mapping or a pinned NIC, not a NCCL knob. The order is always: read the log, identify the transport, fix the transport, *then* consider algorithm tuning.

## Key takeaways

- **NCCL is the black box under every collective**, and it makes two silent, sticky decisions at startup — the *algorithm* (ring/tree/CollNet/NVLS) and the *transport* (NVLink/SHM/IB-GDR/socket) — that determine your throughput. The only record of those decisions is the log.
- **`NCCL_DEBUG=INFO` plus `NCCL_DEBUG_FILE` is the core skill.** Turn it on, open one rank's file, and check four things: consistent version, `IB` versus `Socket` network, the `via ...` transport tags, and that every rank reached `Init COMPLETE`.
- **The `via` tag is the whole diagnosis.** `P2P/CUMEM` is NVLink (fast), `SHM` is host-memory (slow intra-node), `NET/IB/GDRDMA` is RDMA (fast cross-node), `NET/IB` without GDR leaves bandwidth on the table, and `NET/Socket` is the TCP fallback that silently halves your throughput.
- **The socket fallback is the most expensive silent bug in multi-node training.** It logs once at `INFO`, never errors, and can cost `10×` bandwidth. `grep "via NET" nccl.*.log` finds it in one command; `NCCL_IB_HCA` and `NCCL_SOCKET_IFNAME` usually fix it.
- **Ring wins on bandwidth, tree wins on latency.** Ring moves an optimal $2\frac{N-1}{N}S$ bytes but pays $O(N)$ latency; tree pays only $O(\log_2 N)$ latency at a worse bandwidth constant. NCCL picks automatically by message size — override with `NCCL_ALGO` only with a benchmark.
- **The flight recorder is a runtime log**, and reading it is the same skill as reading the startup log: dump every rank's buffer on a timeout and find the one whose sequence number is behind the majority. That names the stuck rank mechanically, at any scale.
- **Benchmark with `nccl-tests`, read `busbw`, sweep message sizes, and warm up.** Compare against the hardware line rate, isolate one variable at a time, and use the log to *prove* the transport changed and the benchmark to prove it mattered.
- **Read the log before you tune anything.** Most "slow all-reduce" tickets are transport bugs no algorithm knob can fix. Identify the transport first; fix the environment; tune the algorithm last, if ever.

## Further reading

- [Why distributed training](/blog/machine-learning/distributed-training/why-distributed-training) — the four walls and the map of the whole series; this post is the "how you debug the comms" layer.
- [Collectives from scratch](/blog/machine-learning/distributed-training/collectives-from-scratch) — the ring all-reduce and the $2\frac{N-1}{N}S$ byte volume that the algorithm section builds on.
- [The interconnect physics](/blog/machine-learning/distributed-training/the-interconnect-physics) — NVLink, NVSwitch, PCIe, InfiniBand, and RoCE; the bandwidth and latency numbers behind every transport choice.
- [The NCCL timeout that hung the job](/blog/machine-learning/distributed-training/the-nccl-timeout-that-hung-the-job) — the full hang-debugging escalation ladder and the catalog of causes behind a stuck collective.
- [Multi-node slower than single node](/blog/machine-learning/distributed-training/multinode-slower-than-single-node) — the throughput autopsy where a socket fallback like this one is the usual suspect.
- [Debugging distributed jobs](/blog/machine-learning/distributed-training/debugging-distributed-jobs) — the first-ten-minutes discipline for where to look when a distributed run goes wrong.
- [The distributed training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) — the capstone decision-and-debugging checklist tying the whole series together.
- [Debugging DDP and multi-GPU](/blog/machine-learning/debugging-training/debugging-ddp-and-multi-gpu) — the broader multi-GPU bug taxonomy from the debugging-training pillar.
- **NCCL docs (NVIDIA):** the environment-variable reference (`NCCL_DEBUG`, `NCCL_DEBUG_SUBSYS`, `NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`, `NCCL_NET_GDR_LEVEL`, `NCCL_ALGO`), the troubleshooting guide, and the `nccl-tests` repository for `all_reduce_perf`.
- **PyTorch docs:** `torch.distributed` process groups and timeouts, and the NCCL flight recorder / c10d debugging guide (`TORCH_NCCL_TRACE_BUFFER_SIZE`, `TORCH_NCCL_DUMP_ON_TIMEOUT`).

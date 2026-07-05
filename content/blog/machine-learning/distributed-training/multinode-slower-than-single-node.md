---
title: "When Multi-Node Was Slower Than Single-Node: An Interconnect Autopsy"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "You doubled from one 8-GPU node to two and throughput fell to 0.8x. Here is the systematic autopsy that finds the socket fallback, the bad placement, and the batch you forgot to grow, with the numbers before and after each fix."
tags:
  [
    "distributed-training",
    "multi-gpu",
    "multi-node",
    "nccl",
    "infiniband",
    "pytorch",
    "gpu",
    "ml-systems",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Distributed Training"
author: "Hiep Tran"
featured: true
readTime: 36
---

The single node was beautiful. Eight A100 80GB SXM cards, all wired through NVSwitch, chewing through a 7-billion-parameter transformer at about 29,000 tokens per second and roughly 50% MFU. That is a genuinely healthy number. The GPUs were busy, the gradient all-reduce vanished into the backward pass, and the loss curve was smooth. So we did the obvious thing: we asked for a second identical node and expected the throughput to roughly double.

It did not double. It went **down**. Sixteen GPUs delivered about 23,000 tokens per second, roughly **0.8x** of what eight GPUs did alone. We had spent twice the money to make the run slower. Two nodes were slower than one. The MFU had collapsed from ~50% to ~21%, the GPUs were idle most of the time, and `nvidia-smi` on both nodes showed the cards sitting at 30% utilization while something invisible ate the clock.

This is one of the most common and most demoralizing failures in scaling out, and it is almost always the interconnect. The all-reduce that hid perfectly on NVLink is now crossing an inter-node fabric that is either the wrong fabric, a misconfigured fabric, or no fast fabric at all. This post is the autopsy: the mechanism that explains why one slow link poisons the whole run, the *one measurement you must take before touching anything else*, a decision tree that turns a bandwidth number into a diagnosis, the full catalog of causes with detection and fix for each, and the before-and-after numbers when we finally pinned the traffic to InfiniBand. By the end you will be able to walk up to a multi-node run that is slower than it should be and localize the cause in about ten minutes.

![before and after view of the same gradient synchronization running fast on a single node then collapsing across two nodes](/imgs/blogs/multinode-slower-than-single-node-1.webp)

If you have not yet read the [interconnect physics post](/blog/machine-learning/distributed-training/the-interconnect-physics) or the [collectives-from-scratch post](/blog/machine-learning/distributed-training/collectives-from-scratch), skim them first: they establish the NVLink-vs-InfiniBand bandwidth gap and the ring all-reduce byte volume that this whole autopsy hangs on. This post is Track E of the [distributed training series](/blog/machine-learning/distributed-training/why-distributed-training), the war-story track, and it feeds directly into the [debugging playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook).

## The setup: one node that screamed, two nodes that crawled

Let me pin down the hardware and the run, because the numbers only make sense against a specific machine. Every figure below traces to this configuration.

- **Model**: a 7B-parameter decoder-only transformer (Llama-shaped), trained in bf16 with a bf16 master copy of the weights in the optimizer.
- **Node**: a DGX-style box, 8x A100 80GB SXM. Inside the node the GPUs are fully connected through NVSwitch. NVLink3 gives roughly 600 GB/s of bidirectional bandwidth per GPU into the switch, and an 8-GPU in-node ring all-reduce measures around 235 GB/s of *bus bandwidth* (the number `nccl-tests` reports; more on that word in a moment).
- **Inter-node fabric**: the cluster has InfiniBand HDR, 200 Gb/s per port, which is about 25 GB/s of line rate per NIC. The good nodes have one HDR NIC per GPU. There is also a separate 40 GbE management/storage network that every node uses for SSH, NFS, and telemetry.
- **Parallelism**: pure data parallelism with PyTorch DDP. Global batch of about 116,000 tokens on the single node, assembled as 8 GPUs x 4 gradient-accumulation micro-batches x ~3,625 tokens per micro-batch. One optimizer step took about 4.0 seconds on the single node.

The gradient all-reduce is the entire story. A 7B model in bf16 has **14 GB** of gradients. Data-parallel training synchronizes those 14 GB across every rank on every optimizer step. On the single node that synchronization is trivial; across nodes it is the bottleneck. Here is the arithmetic that decides everything, and it is worth internalizing before we touch a single flag.

A ring all-reduce moves $2(N-1)/N \cdot S$ bytes per GPU, where $S$ is the total payload (14 GB here) and $N$ is the number of participating GPUs. For 16 GPUs that is $2 \cdot 15/16 \cdot 14 = 26.25$ GB of traffic per GPU, and the time it takes is that volume divided by the achieved bus bandwidth $B$:

$$T_\text{allreduce} = \frac{2(N-1)}{N} \cdot \frac{S}{B} \approx \frac{26.25\ \text{GB}}{B}$$

Now plug in the two bandwidths and watch what happens:

- **In-node, on NVLink**, $B \approx 235$ GB/s, so $T_\text{allreduce} \approx 0.11$ s. Against a 4.0 s compute step, that is under 3% of the time. DDP overlaps it with the backward pass and it disappears entirely.
- **Cross-node, on the 40 GbE management NIC** (which is where we accidentally ended up), $B \approx 4$ GB/s, so $T_\text{allreduce} \approx 6.6$ s. That is **larger than the entire compute step**. There is no hiding a 6.6 s all-reduce behind a 4.0 s of compute. Most of it is exposed, the step balloons to ~10 s, and the throughput craters.

That single division, 26.25 GB divided by whatever bandwidth you actually achieved, is the number that tells you whether multi-node will fly or die. Everything else in this post is about measuring $B$ honestly and getting it as high as the hardware allows.

### Why one slow link poisons the entire run

Here is the intuition that trips people up. On the single node the all-reduce was 3% of the step, so you might assume that a slower link just makes it, say, 30% of the step and costs you a bit of scaling. That is not how it works, because of two multiplicative effects.

First, the bandwidth gap is not 2x, it is **60x** (235 GB/s in-node versus 4 GB/s on the management NIC). The all-reduce does not get a little slower, it gets 60x slower. Second, and this is the cruel part, the all-reduce is on the **critical path** of every single step. There is no useful work to do while 14 GB crawls across a slow wire; the GPUs finished the backward pass and now they wait. So the exposed comms time adds directly to wall-clock, step after step, forever. A 60x slower link on a 3% cost does not give you a 2x slowdown; it gives you a comms cost that dwarfs compute and turns your expensive GPUs into space heaters.

This is exactly why the series keeps hammering the [overlap of compute and communication](/blog/machine-learning/distributed-training/overlapping-compute-and-communication): overlap is the only thing that hides comms, and overlap can only hide comms that are *smaller than* the compute they run alongside. Once comms exceed compute, overlap saturates and the excess is exposed. Multi-node is where that boundary gets crossed, because the fabric is 10-60x slower than NVLink and the payload (the full gradient) is fixed by the model size, not the batch.

## The mechanism: the all-reduce that left the building

Let me make the two data paths concrete, because the whole bug is about which path your bytes take. When a gradient bucket is ready and NCCL needs to send it to a peer rank on the other node, it has a choice of transport, and that choice is the difference between 22 GB/s and 4 GB/s.

![two data paths for the cross node gradient all reduce, one direct over InfiniBand and one bouncing through host memory on TCP sockets](/imgs/blogs/multinode-slower-than-single-node-2.webp)

On the **fast path**, NCCL uses InfiniBand verbs with GPUDirect RDMA (GDR). The NIC reads the gradient directly out of GPU memory and writes it directly into the peer GPU's memory across the fabric. The CPU is not involved in the data movement, there is no extra copy, and you get close to the line rate of the NIC. This is the path you want, and it is the path NCCL takes automatically **when it can find a working IB device and the GDR plumbing is in place**.

On the **slow path**, NCCL falls back to TCP sockets. Now the sequence is: copy the gradient from GPU memory to a staging buffer in host CPU RAM, hand it to the kernel's TCP stack, serialize it into packets, push it out whatever network interface the socket is bound to, and reverse the whole thing on the other side. Every one of those steps is a tax. You lose the RDMA directness, you pay a device-to-host copy, you pay TCP/IP overhead, and worst of all NCCL often binds the socket to the *management* interface because that is the one it found first. You are now doing your 26 GB/GPU gradient sync over a 40 GbE admin network that was designed to carry SSH sessions and NFS metadata.

The reason this is so insidious is that **the run does not crash**. There is no error. The loss still goes down. The only symptom is that it is slow, and "slow" is the hardest bug to notice because there is nothing to grep for in the logs unless you know exactly where to look. The job trains, it converges, it just costs you 3x the GPU-hours it should. This is why the first rule of multi-node is: never trust that the fast path was taken. Measure it.

### The comms-to-compute ratio, the number that predicts everything

Define the ratio that governs whether scaling out helps:

$$r = \frac{T_\text{allreduce}}{T_\text{compute}} = \frac{2(N-1)/N \cdot S / B}{T_\text{compute}}$$

When $r \ll 1$, comms hides under compute and you scale near-linearly. When $r \approx 1$, you are on the knife's edge and overlap saves maybe half of it. When $r > 1$, comms dominates, the GPUs starve, and adding more of them makes it *worse* because a larger $N$ pushes the $2(N-1)/N$ factor toward 2 and there are more ranks contending for the same fabric. Here is the ratio for our four configurations:

| Configuration | Bus bandwidth $B$ | $T_\text{allreduce}$ | $T_\text{compute}$ | $r$ | Verdict |
|---|---|---|---|---|---|
| 1 node, NVLink | 235 GB/s | 0.11 s | 4.0 s | 0.03 | comms invisible |
| 2 nodes, TCP sockets | 4 GB/s | 6.6 s | 4.0 s | 1.65 | comms dominates |
| 2 nodes, 1 IB NIC + GDR | 22 GB/s | 1.2 s | 4.0 s | 0.30 | mostly hidden |
| 2 nodes, 8 IB NICs + GDR | 160 GB/s | 0.16 s | 4.0 s | 0.04 | comms invisible |

Read the second row and the fourth row. Same model, same GPUs, same code, same 26.25 GB of gradient. The only thing that changed is the achieved bus bandwidth of the fabric, and it took the comms-to-compute ratio from 1.65 (catastrophe) to 0.04 (perfect). The bug is never in your model. It is in that $B$.

## The first number that matters: run nccl-tests before you touch anything

Here is the single most important habit in multi-node training, and the one that separates people who debug this in ten minutes from people who lose a week to it: **before you launch your training job, measure the raw inter-node all-reduce bandwidth in isolation with `nccl-tests`.** Do not profile the training run first. Do not read your framework's logs first. Measure the fabric, naked, with the same NCCL your job will use. The number it gives you immediately tells you which universe you are in.

`nccl-tests` is NVIDIA's official microbenchmark for exactly this. You build it against your NCCL and MPI, then run `all_reduce_perf` across the two nodes.

```bash
# Build nccl-tests once against your CUDA + NCCL + MPI
git clone https://github.com/NVIDIA/nccl-tests
cd nccl-tests
make MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi CUDA_HOME=/usr/local/cuda

# Run an all-reduce sweep across 2 nodes, 8 GPUs each (16 ranks total).
# -b 8 -e 8G sweeps sizes from 8 bytes to 8 GB; -g 1 = one GPU per rank.
mpirun -np 16 -H node1:8,node2:8 \
  -x NCCL_DEBUG=INFO \
  -x NCCL_IB_HCA=mlx5 \
  ./build/all_reduce_perf -b 8 -e 8G -f 2 -g 1
```

The output is a table of message sizes with two bandwidth columns that matter: `algbw` (algorithm bandwidth, size divided by time) and `busbw` (bus bandwidth, which factors in the $2(N-1)/N$ ring cost so it is comparable across different $N$). Read the `busbw` at the large sizes, near 8 GB, because that is the regime your 14 GB gradient lives in. Small messages are latency-bound and lie to you.

Here is what a **healthy** two-node run looks like, one HDR NIC per GPU with GDR working:

```console
#       size         count      type     time    algbw    busbw
#        (B)    (elements)              (us)   (GB/s)   (GB/s)
   536870912     134217728     float   3421.0    156.9    294.2   <- small, don't trust
  1073741824     268435456     float   6210.0    172.9    324.2
  2147483648     536870912     float  12100.0    177.5    332.9
  4294967296    1073741824     float  24010.0    178.9    335.5
  8589934592    2147483648     float  47800.0    179.7    336.9
# Avg bus bandwidth : ~160 GB/s at the sizes that matter
```

And here is what the **broken** run looked like, the one that caused the 0.8x collapse, socket fallback over the management NIC:

```console
#       size         count      type     time    algbw    busbw
#        (B)    (elements)              (us)   (GB/s)   (GB/s)
   536870912     134217728     float  268000.0     2.00     3.76
  1073741824     268435456     float  536000.0     2.00     3.75
  2147483648     536870912     float 1072000.0     2.00     3.75
  4294967296    1073741824     float 2144000.0     2.00     3.75
  8589934592    2147483648     float 4288000.0     2.00     3.75
# Avg bus bandwidth : ~3.75 GB/s  <- you are on sockets, not IB
```

That number, ~4 GB/s, is the whole diagnosis in one line. It is impossible on InfiniBand. HDR line rate is 25 GB/s per NIC; even a single degraded IB link would show ~10-15 GB/s. A busbw of 4 GB/s means your gradient is going over Ethernet through the TCP stack. You have found your bug before you launched a single training step. Contrast the two: 160 versus 3.75 is a 40x difference, and it is exactly the difference between a run that scales and a run that regresses.

### The decision tree: what the bandwidth number tells you

The beauty of leading with `nccl-tests` is that the single busbw number partitions the whole space of causes. Once you have it, you know which branch of the diagnosis to walk down.

![decision tree that routes the measured bus bandwidth number to the transport fix it implies](/imgs/blogs/multinode-slower-than-single-node-3.webp)

- **busbw is 1-4 GB/s** → you are on TCP sockets, not IB. This is the number-one cause by a wide margin. Either NCCL never found an IB device, or it found one and could not use it (missing drivers, wrong container, GDR broken) and silently fell back. Go check the transport with `NCCL_DEBUG=INFO` and pin `NCCL_IB_HCA` / `NCCL_SOCKET_IFNAME`.
- **busbw is ~20-24 GB/s** → you are on a single IB NIC per node and it is working, but you are not aggregating multiple NICs and GDR may be off. This is *fine* for a first correct run and often good enough. To go further, enable GDR and use all the NICs.
- **busbw is ~150+ GB/s** → the fabric is healthy and close to in-node speeds. If your training is still slow, the problem is not the transport; it is placement, batch size, or the data pipeline. Stop looking at NCCL.

The reason a decision tree is the right tool here is that the busbw number is nearly a bijection with the cause. You do not guess; you read the number and it hands you the branch. That is the difference between a systematic diagnosis and flailing with random env vars off a forum post.

## Reading the transport: NCCL_DEBUG=INFO and the NET/Socket smoking gun

Once `nccl-tests` tells you the bandwidth is wrong, the next question is *why*, and `NCCL_DEBUG=INFO` answers it. This environment variable makes NCCL print, at startup, exactly which transport and which network interface every rank chose. It is verbose, but you only need one line.

```bash
NCCL_DEBUG=INFO torchrun \
  --nnodes 2 --nproc_per_node 8 \
  --node_rank $NODE_RANK \
  --rdzv_backend c10d --rdzv_endpoint node1:29500 \
  train.py 2>&1 | grep -E "NET/|NCCL INFO Bootstrap|GDR"
```

On the **broken** run, the grep returns the smoking gun:

```log
node1:12345:12400 [0] NCCL INFO NET/Socket : Using [0]eno1:10.0.0.5<0>
node1:12345:12400 [0] NCCL INFO Using network Socket
node2:54321:54390 [0] NCCL INFO NET/Socket : Using [0]eno1:10.0.1.5<0>
node2:54321:54390 [0] NCCL INFO Using network Socket
```

`NET/Socket`. `Using network Socket`. And it bound to `eno1`, the 40 GbE management interface, at 10.0.x.x. That is the diagnosis in three words: NCCL did not use InfiniBand. It never even tried, or it tried and failed silently and fell back to TCP.

On the **fixed** run, the same grep returns what you want to see:

```log
node1:12345:12400 [0] NCCL INFO NET/IB : Using [0]mlx5_0:1/IB [1]mlx5_1:1/IB ... [7]mlx5_7:1/IB
node1:12345:12400 [0] NCCL INFO Using network IB
node1:12345:12400 [0] NCCL INFO GPU Direct RDMA Enabled for GPU 0 / HCA 0
```

`NET/IB`, `Using network IB`, and `GPU Direct RDMA Enabled`. That is the healthy signature. The rule is dead simple and worth taping to your monitor: **if you see `NET/Socket` on a multi-node run, you are broken, full stop.** There is no configuration in which crossing nodes on sockets is acceptable when IB hardware exists. For a deeper tour of the NCCL log format, the [nccl-debugging-deep-dive](/blog/machine-learning/distributed-training/why-distributed-training) sibling in Track F walks every line; here I only need the transport line.

## The catalog of causes: detection and fix for each

`nccl-tests` and `NCCL_DEBUG=INFO` will catch the big one (socket fallback) most of the time, but multi-node has a whole menagerie of ways to be slow. Here is the full catalog. Each row is a cause, the signal that confirms it, the one-line fix, and its effect on bandwidth. This is the matrix I keep in my head when I walk up to a slow multi-node run.

![matrix mapping each multi node slowdown cause to its detection signal, one line fix, and bandwidth effect](/imgs/blogs/multinode-slower-than-single-node-4.webp)

Let me walk each one, because the fix for each is specific and the detection is what makes it fast.

### Cause 1: NCCL fell back to TCP sockets (the #1 cause)

**Detection**: `NCCL_DEBUG=INFO` shows `NET/Socket`, and `nccl-tests` busbw is 1-4 GB/s. **Fix**: force IB and name the device.

```bash
export NCCL_IB_HCA=mlx5          # use the Mellanox HCAs (all mlx5_* devices)
export NCCL_SOCKET_IFNAME=ib0    # bootstrap/OOB over the IB interface, not eno1
export NCCL_IB_DISABLE=0         # explicitly allow IB (some images set 1)
export NCCL_NET_GDR_LEVEL=SYS    # allow GDR across the PCIe topology
```

`NCCL_IB_HCA=mlx5` tells NCCL to use the Mellanox host channel adapters instead of scanning and giving up. `NCCL_SOCKET_IFNAME=ib0` fixes the *bootstrap* interface (NCCL still uses a TCP socket for the initial rendezvous handshake even when data goes over IB; you want that handshake on the IB IP-over-IB interface, not the slow management NIC, or the bootstrap itself can misbehave). If forcing these still yields sockets, the IB stack is not actually present in your environment: check `ibstat`, check that the container was launched with `--device` / `--privileged` or the right IB devices mapped, and check that the OFED drivers are installed.

### Cause 2: the wrong network interface was selected

**Detection**: `NCCL_DEBUG=INFO` shows `NET/Socket : Using [0]docker0` or `eno1` or some virtual interface. This happens constantly inside containers, where `docker0`, `veth*`, and `lo` all exist and NCCL picks the wrong one alphabetically. **Fix**: name the fast interface explicitly, and exclude the decoys.

```bash
export NCCL_SOCKET_IFNAME=ib0        # or =^docker0,lo  to exclude by prefix
```

The `^` prefix syntax means "everything except these." On a node with a proper IB setup you name `ib0`; on a RoCE (RDMA over Converged Ethernet) setup you name the fast Ethernet interface, e.g. `ens5f0`. The point is to never let NCCL guess.

### Cause 3: no GPUDirect RDMA, so every transfer bounces through host memory

**Detection**: `NCCL_DEBUG=INFO` shows `NET/IB` (good, you are on IB) but no `GPU Direct RDMA Enabled` line, or it says GDR is disabled, and your busbw is stuck around half of line rate (e.g. ~12 GB/s on a 25 GB/s NIC). **Fix**: the GDR plumbing.

```bash
# On the host: the peer-memory kernel module must be loaded
sudo modprobe nvidia_peermem      # newer stacks; older: nv_peer_mem
lsmod | grep -E "nvidia_peermem|nv_peer_mem"
export NCCL_NET_GDR_LEVEL=PHB      # or SYS; controls how "far" GDR is allowed
```

Without `nvidia_peermem`, the NIC cannot read GPU memory directly, so NCCL stages every transfer through a host bounce buffer: GPU to CPU RAM, then CPU RAM to NIC. That extra copy roughly halves your effective bandwidth and adds latency. Loading the module and setting `NCCL_NET_GDR_LEVEL` so the GDR path is allowed across your PCIe topology is the fix. On many clusters this is the difference between 12 and 22 GB/s per NIC.

### Cause 4: a degraded or misconfigured IB link

**Detection**: `ibstat` shows a link that is not `LinkUp`, or is running at a lower rate than expected (e.g. `4X SDR` instead of `4X HDR`), or the `dmesg` log shows IB errors. **Fix**: hardware, not software.

```console
$ ibstat mlx5_0
CA 'mlx5_0'
    Port 1:
        State:           Active
        Physical state:  LinkUp
        Rate:            200                 <- want 200 (HDR); 100 = degraded to EDR
        Base lid:        14
        Link layer:      InfiniBand
```

If `Rate` reads 100 when it should be 200, a cable is negotiating down, a port is bad, or an optic is dying. Confirm the point-to-point bandwidth between two specific nodes with `ib_write_bw`, which is the IB-native version of `nccl-tests` and isolates the fabric from NCCL entirely:

```console
# On node1:  ib_write_bw -d mlx5_0 -a
# On node2:  ib_write_bw -d mlx5_0 -a node1
 #bytes     #iterations    BW peak[MB/sec]    BW average[MB/sec]
 8388608    1000           24180.5            24102.3     <- healthy HDR
```

If `ib_write_bw` is slow, the problem is below NCCL and you escalate to the fabric team: reseat the cable, move to another switch port, or drain the node. If `ib_write_bw` is fast but `nccl-tests` is slow, the problem is above the fabric, back in your NCCL config.

### Cause 5: tensor parallelism accidentally spanning nodes

This one is a placement bug, not a transport bug, and it is vicious because the transport is *fine*, you are just sending the wrong traffic across it. Tensor parallelism does an all-reduce inside every transformer layer, on the critical path, dozens of times per step. That traffic **must** stay on NVLink. If your rank assignment puts a tensor-parallel group across two nodes, you are running the highest-frequency, most latency-sensitive collective in the whole model over the slow fabric.

**Detection**: `nvidia-smi topo -m` to see the intra-node topology, plus a check of which ranks form each TP group against which node they live on.

```console
$ nvidia-smi topo -m
        GPU0    GPU1    GPU2    GPU3    ...    NIC0    NIC1
GPU0     X      NV18    NV18    NV18           PXB     SYS
GPU1    NV18     X      NV18    NV18           PXB     SYS
...
Legend: NV# = NVLink; PXB = PCIe switch; SYS = across the CPU root complex
```

`NV18` between GPUs means NVLink; that is where TP wants to live. The placement law is simple: **the tensor-parallel degree must divide the GPUs-per-node**, so a TP group never straddles the node boundary. With 8 GPUs per node, TP of 2, 4, or 8 is safe; TP of 16 forces a cross-node TP group and is almost always wrong. This is the same placement discipline the [3D parallelism post](/blog/machine-learning/distributed-training/3d-parallelism) and the [tensor parallelism post](/blog/machine-learning/distributed-training/tensor-parallelism-megatron) derive in detail: put the chattiest parallelism dimension on the fastest link. **Fix**: reorder the rank-to-device mapping so TP groups are intra-node, and keep TP $\leq 8$ on an 8-GPU node.

### Cause 6: MTU and flow-control misconfiguration (RoCE especially)

**Detection**: on RoCE (RDMA over Ethernet, not native IB), `ip link` shows an MTU of 1500 instead of 9000, or the fabric is dropping packets under load because Priority Flow Control (PFC) and ECN are not configured. Native InfiniBand mostly avoids this; RoCE is where MTU and flow control bite. **Fix**: jumbo frames and lossless config.

```bash
sudo ip link set ib0 mtu 9000     # jumbo frames; 1500 wrecks large-message BW
```

On RoCE you also need PFC and ECN enabled on both the NICs and the switches so that RDMA sees a lossless fabric; without it, congestion causes drops, drops cause retransmits, and your busbw collapses in a way that *looks* like a bandwidth problem but is really a packet-loss problem. This is a network-engineering fix that usually involves the fabric team, but you detect it from the training side by watching busbw fall off a cliff only at large message sizes.

### Cause 7: the global batch was not grown when nodes were added

The last cause has nothing to do with the network at all, and it is the one people forget. When you go from 8 to 16 GPUs, you have two choices: keep the **per-GPU** batch fixed and double the **global** batch, or keep the global batch fixed and halve the per-GPU work. If you do the second (which is the default if you do not think about it), each GPU now does half the compute per step while the all-reduce payload stays exactly the same size (the gradient is the size of the model, independent of batch). So the comms-to-compute ratio $r$ **doubles**, and you have made overlap harder purely by adding hardware.

**Detection**: your per-GPU tokens-per-step dropped when you added nodes, and the comms fraction in the profiler rose. **Fix**: grow the global batch proportionally to the GPU count so per-GPU compute stays constant. We will work this one fully in the second worked example, because it is subtle and it is where a lot of otherwise-correct multi-node setups leave 20% on the table.

## The systematic diagnosis loop

Put the tools in order and you have a loop that localizes almost any multi-node slowdown in ten minutes. The order matters: each step is cheaper than the next and narrows the space, so you never run the expensive profiler before the cheap microbenchmark has told you where to look.

![ordered diagnosis loop from bandwidth microbenchmark through transport log, profiler, topology, fix, and re-measure](/imgs/blogs/multinode-slower-than-single-node-5.webp)

1. **`nccl-tests all_reduce_perf`** across the nodes. This is the isolation test: it measures the fabric with no training code in the way. The busbw number immediately tells you IB (tens to hundreds of GB/s) versus sockets (~1-4 GB/s). Two minutes.
2. **`NCCL_DEBUG=INFO`** on the real launch, grep for `NET/`. This tells you *why* the bandwidth is what it is: which transport and which interface NCCL chose. One line of output. One minute.
3. **`torch.profiler`** on a handful of training steps, if the fabric is confirmed fast but the run is still slow. This shows you the *exposed* all-reduce on the timeline: how much of each step the GPUs spend waiting on `nccl:all_reduce` versus computing. Five minutes.
4. **`nvidia-smi topo -m` and `ibstat`** to confirm placement and link health. Are TP groups on NVLink? Is every IB port `LinkUp` at full rate? Two minutes.
5. **Apply the fix** the number pointed to (pin IB, set the interface, enable GDR, reorder ranks, grow the batch).
6. **Re-measure** with `nccl-tests` and then the real run. Confirm busbw jumped and MFU recovered. Never declare victory without the after-number.

The discipline here is the same as any performance work, and it is worth stating as a rule: **measure the isolated component first, read the configuration second, profile the integrated system third, and always re-measure the fix.** The failure mode I see most often is engineers skipping straight to step 3, staring at a profiler timeline full of exposed all-reduce, and concluding "comms is slow" without ever asking *why* comms is slow, which the two-minute `nccl-tests` in step 1 would have answered.

Here is the profiler block for step 3, which makes the exposed comms visible on the timeline:

```python
import torch
from torch.profiler import profile, ProfilerActivity, schedule

def train_step():
    optimizer.zero_grad(set_to_none=True)
    for micro in range(accum_steps):
        with model.no_sync() if micro < accum_steps - 1 else nullcontext():
            loss = model(batch[micro]).loss / accum_steps
            loss.backward()          # all-reduce fires on the LAST micro-batch
    optimizer.step()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=2, active=3, repeat=1),
    record_shapes=False,
    with_stack=False,
) as prof:
    for _ in range(6):
        train_step()
        prof.step()

# Sort by CUDA time; a slow-fabric run shows nccl:all_reduce near the top,
# with a large gap between GPU-kernel time and wall time = exposed comms.
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))
```

On the broken run this table shows `ncclAllReduce` (or `nccl:all_reduce`) consuming 60%+ of the step's CUDA time, with the GPUs idle during it. On the fixed run that same collective is a thin sliver overlapped with the backward kernels. The profiler is the ground truth that overlap is or is not happening; but again, it tells you *that* comms is exposed, and `nccl-tests` already told you *why*.

Note the `no_sync()` detail in that snippet, because it matters for the arithmetic. With gradient accumulation, DDP suppresses the all-reduce on every micro-batch except the last (that is what `no_sync()` does). So the full 14 GB all-reduce fires **once per optimizer step**, at the end, and can only overlap with the *final* micro-batch's backward pass, roughly 0.5 s of compute. Everything beyond that 0.5 s is exposed. That is why a 6.6 s socket all-reduce leaves ~6.1 s exposed even though the total compute step is 4.0 s: only a fraction of the step is available to hide behind. This is a real subtlety of DDP plus accumulation that surprises people. (FSDP and ZeRO behave differently, streaming reduce-scatters throughout the backward; more on that in the fixes section.)

#### Worked example: the socket fallback and the one env var that fixed it

Let me put the numbers together end to end, because this is the exact sequence that took our run from 0.8x to 1.9x.

**Symptom.** Two nodes, 16 A100s, 23,000 tokens/s. One node, 8 A100s, 29,000 tokens/s. Scaling efficiency of a laughable 0.4 (we got 0.79x the single-node throughput from 2x the hardware, so per-GPU efficiency versus single-node was $23000/29000/2 = 0.40$). MFU had fallen from ~50% to ~21%.

**Step 1, nccl-tests.** `all_reduce_perf` at 8 GB reported **busbw 3.75 GB/s**. Impossible on IB (HDR line rate is 25 GB/s). We knew we were on sockets before touching the training code.

**Step 2, NCCL_DEBUG=INFO.** The log confirmed it: `NET/Socket : Using [0]eno1`. NCCL had bound to the 40 GbE management interface. The container image we were using set `NCCL_IB_DISABLE=1` in its base layer (a "safe default" some images ship to avoid crashes on IB-less machines), which forced the fallback. NCCL found no IB, shrugged, and used Ethernet.

**Step 3, the fix.** One block of environment variables:

```bash
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME=ib0
export NCCL_NET_GDR_LEVEL=SYS
```

**Step 4, re-measure.** `nccl-tests` busbw jumped from 3.75 to **160 GB/s** (all 8 NICs per node, GDR enabled). The all-reduce time for our 14 GB gradient fell from 6.6 s to 0.16 s, fully hidden under the backward pass again.

**Result.** Two-node throughput went from 23,000 to **57,300 tokens/s**, a genuine 1.98x over the single node, ~98% scaling efficiency, MFU back to ~49%. The fix was, quite literally, deleting one wrong default (`NCCL_IB_DISABLE=1`) and naming the hardware. The lesson is not "set these four variables"; it is *always run nccl-tests first*, because it would have told us on day one, before we ran a single wasteful training step, that the fabric was on sockets.

## The fixes and the numbers

Let me lay out the full menu of fixes and what each is worth, because "pin IB" is only the first move and there is real headroom beyond it. The transport stack has several layers, and each layer is a place the traffic can go wrong or a knob you can turn.

![vertical stack of the transport layers from the DDP all reduce call down to the physical fabric where the socket fallback hides](/imgs/blogs/multinode-slower-than-single-node-6.webp)

- **Pin IB and name the HCA** (`NCCL_IB_HCA=mlx5`, `NCCL_IB_DISABLE=0`): gets you off sockets and onto IB. This is the 4 → 22 GB/s jump per NIC, the single biggest win. Non-negotiable.
- **Fix the interface** (`NCCL_SOCKET_IFNAME=ib0`): ensures the bootstrap handshake and any socket traffic use the fast interface, not `eno1`/`docker0`. Prevents the fallback and the mysterious slow-rendezvous.
- **Enable GPUDirect RDMA** (`nvidia_peermem` module + `NCCL_NET_GDR_LEVEL`): removes the host bounce buffer. This is the ~12 → 22 GB/s jump per NIC, roughly a 2x, by cutting the device-to-host copy.
- **Aggregate all the NICs**: with 8 HDR NICs per node and one per GPU, NCCL rails traffic across all of them and cross-node busbw climbs toward in-node speeds (~160 GB/s in our case). This is the 22 → 160 GB/s jump, and it is what makes 2-node scaling near-perfect.
- **Keep heavy traffic in-node with HYBRID_SHARD**: if you are on FSDP rather than DDP, use `ShardingStrategy.HYBRID_SHARD`. It shards the model *within* a node (all-gather and reduce-scatter stay on NVLink) and only does a smaller all-reduce *across* nodes. The chattiest collectives never touch the fabric.

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.HYBRID_SHARD,  # shard in-node, replicate across
    device_id=torch.cuda.current_device(),
    # ... mixed precision, auto_wrap_policy, etc.
)
```

- **Grow the NCCL buckets**: larger buckets mean fewer, bigger all-reduce messages, which is more efficient on a high-latency fabric where per-message overhead hurts. `NCCL_BUFFSIZE` and DDP's `bucket_cap_mb` are the knobs. On a slow fabric, going from the default 25 MB buckets to 100+ MB can claw back a few percent.
- **Grow the global batch**: covered in the next section, this restores the comms-to-compute ratio when you add nodes.

Here is the full before-and-after on our named hardware, the 8x A100 node and the HDR fabric, every number measured with warm-up steps discarded and `torch.cuda.synchronize()` before timing:

| Config | Cross-node busbw | All-reduce time | Exposed comms | Step time | Tokens/s | Scaling vs 1 node | MFU |
|---|---|---|---|---|---|---|---|
| 1 node, 8 GPU (NVLink) | 235 GB/s | 0.11 s | 0 (hidden) | 4.0 s | 29,000 | 1.00x | ~50% |
| 2 nodes, sockets (broken) | 4 GB/s | 6.6 s | 6.0 s | 10.0 s | 23,000 | **0.79x** | 21% |
| 2 nodes, 1 IB NIC + GDR | 22 GB/s | 1.2 s | 0.6 s | 4.6 s | 50,400 | 1.74x | 45% |
| 2 nodes, 8 NICs + GDR | 160 GB/s | 0.16 s | ~0 (hidden) | 4.05 s | 57,300 | 1.98x | ~49% |

Read the last column down. The socket fallback was not a small tax; it was a 0.79x regression, worse than not adding the node at all. Pinning IB with a single NIC already recovered a 1.74x. Aggregating all 8 NICs with GDR got us to a near-perfect 1.98x. Same GPUs, same model, same code. The only variable was the achieved bandwidth of the fabric.

### How to measure this honestly

A word on measurement, because it is easy to fool yourself. Every number in that table came with the following hygiene, and if you skip it your before-and-after is noise:

- **Discard warm-up steps.** The first several steps include CUDA context creation, NCCL ring construction, cuDNN autotuning, and cold caches. Time steady-state, typically steps 10 through 50, not step 1.
- **`torch.cuda.synchronize()` before you read the clock.** CUDA kernels are asynchronous; without the sync you are timing the launch, not the execution, and your numbers will be fiction.
- **Watch the data loader.** A slow `DataLoader` starves the GPU and looks exactly like a comms problem: low utilization, long steps. Confirm the loader is not the bottleneck (`num_workers` high enough, `prefetch_factor` set, `pin_memory=True`) before you blame the fabric. The [data pipeline at scale post](/blog/machine-learning/distributed-training/why-distributed-training) covers this confound.
- **Pin the clocks.** Thermal throttling and power-cap variation move your TFLOP/s around. For an apples-to-apples before-and-after, lock the clocks with `nvidia-smi -lgc` or at least confirm the cards are not throttling.

Only with that hygiene is a 0.79x versus 1.98x a real result rather than measurement drift.

## The batch you forgot to grow

Now the subtle one, the cause that survives even after you have pinned IB and enabled GDR. Suppose your fabric is now healthy at 22 GB/s and your all-reduce takes 1.2 s. You went from 8 to 16 GPUs but you kept the **global** batch fixed at 116,000 tokens because that is what the learning-rate schedule was tuned for. What happened to per-GPU work?

Each GPU now processes half as many tokens per step (the same global batch split over twice the GPUs). Compute per step halves, from 4.0 s to 2.0 s. But the all-reduce payload is the **gradient**, which is the size of the model (14 GB) regardless of batch size. So the all-reduce still takes 1.2 s. The comms-to-compute ratio went from $1.2/4.0 = 0.30$ to $1.2/2.0 = 0.60$. You doubled the comms fraction purely by adding hardware without growing the batch, and overlap can hide less of it.

![before and after comparison of holding the global batch fixed versus growing it to keep per GPU compute constant](/imgs/blogs/multinode-slower-than-single-node-7.webp)

The fix is a mindset shift: when you add GPUs for data parallelism, hold the **per-GPU** batch constant and let the **global** batch grow with the GPU count. Sixteen GPUs should process twice the global batch that eight did, so each GPU keeps its ~3,625 tokens per micro-batch and its full 4.0 s of compute, and the 1.2 s all-reduce stays a manageable 0.30 fraction. This is the correct way to weak-scale data-parallel training, and it is why large runs use global batches of millions of tokens.

There is a caveat, and it is an important one: growing the global batch changes the optimization, not just the systems. A larger batch needs a larger learning rate (roughly the square-root or linear scaling rule, up to a critical batch size), a longer warm-up, and past some point it stops helping convergence per token. So you cannot grow the batch without bound to chase MFU; you grow it up to the critical batch size for your model and no further. Beyond that, the right lever is not a bigger batch but a faster fabric or a smarter parallelism layout. This is the tension the [compute-optimal scaling work](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) formalizes: there is a batch size that is efficient for the systems and a batch size that is efficient for the optimization, and they are not always the same number.

#### Worked example: growing the global batch to restore the ratio

**Setup.** Fabric already fixed at 22 GB/s (all-reduce = 1.2 s). We scale 8 → 16 GPUs.

**The mistake.** Keep global batch at 116,000 tokens. Per-GPU tokens per step halve from ~3,625 to ~1,813. Compute per step halves to 2.0 s. Comms-to-compute ratio rises to $1.2/2.0 = 0.60$. Exposed comms after overlap (~0.5 s of the final backward hides under it) is ~0.7 s. Step time = $2.0 + 0.7 = 2.7$ s, processing 116,000 tokens, so $116000/2.7 \approx 43{,}000$ tokens/s. That is only **1.48x** the single node from 2x the hardware, ~74% scaling efficiency. You left a quarter of the throughput on the floor.

**The fix.** Grow global batch to 232,000 tokens (double, matching the doubled GPU count). Per-GPU work returns to ~3,625 tokens per micro-batch and 4.0 s of compute. Comms-to-compute ratio falls to $1.2/4.0 = 0.30$. Exposed comms ~0.6 s, step time $4.0 + 0.6 = 4.6$ s, processing 232,000 tokens, so $232000/4.6 \approx 50{,}400$ tokens/s, a **1.74x** over the single node, ~87% scaling efficiency. Then bump the learning rate and warm-up to match the bigger batch so convergence per token holds.

The point: the socket-fallback fix (transport) and the batch fix (comms-to-compute) are *independent* levers, and a fully-optimized 2-node run needs both. Fix the transport to make the all-reduce fast, and grow the batch to make what is left of it hide.

## The correct multi-node launch, end to end

Here is a launch that puts it all together, the `torchrun` command with a real rendezvous and the NCCL configuration baked into the environment. This is the SLURM-style two-node launch we settled on; note the env block is the load-bearing part.

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1

# --- The NCCL config that forces the fast path (the whole point) ---
export NCCL_IB_DISABLE=0            # allow InfiniBand (some images default to 1)
export NCCL_IB_HCA=mlx5            # use all Mellanox HCAs
export NCCL_SOCKET_IFNAME=ib0      # bootstrap over IB, never eno1/docker0
export NCCL_NET_GDR_LEVEL=SYS      # allow GPUDirect RDMA across the topology
export NCCL_DEBUG=INFO             # print the transport at startup; grep NET/
export NCCL_DEBUG_SUBSYS=INIT,NET  # keep the log to the parts that matter

# Rendezvous endpoint = first node, a free port
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)

srun torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${HEAD_NODE}:29500" \
  --rdzv_id="${SLURM_JOB_ID}" \
  train.py --global-batch-tokens 232000
```

Three things make this correct. First, the NCCL env block forces IB and names the hardware, so there is no silent fallback. Second, `--rdzv_backend=c10d` with a single head-node endpoint gives a clean rendezvous that survives restarts (the [fault-tolerance post](/blog/machine-learning/distributed-training/why-distributed-training) covers elastic variants). Third, the `--global-batch-tokens 232000` grows the batch with the node count so the comms-to-compute ratio stays healthy. Launch it, immediately grep the log for `NET/IB`, and confirm before you let it run for hours.

## Case studies and real numbers

A few grounding numbers from the literature and from vendor benchmarks, so the figures above sit in a real context. I have tried to be conservative; where I am approximating I say so.

**NVLink versus InfiniBand bus bandwidth.** NVIDIA's own `nccl-tests` results and countless cluster reports put an 8-GPU A100 in-node all-reduce busbw in the low-to-mid 200s of GB/s (NVLink3 through NVSwitch). A single HDR InfiniBand NIC caps around 22-24 GB/s of achievable busbw against its 25 GB/s line rate. That roughly 10x gap per link is the entire reason in-node all-reduce is free and cross-node all-reduce is expensive, and it is the physics the whole [interconnect post](/blog/machine-learning/distributed-training/the-interconnect-physics) is built on. With 8 NICs per node plus GDR, well-tuned clusters recover cross-node busbw into the 150-190 GB/s range, which is why the largest published runs scale to thousands of GPUs at all.

**Megatron-LM at scale.** The Megatron-LM tensor-parallel papers report sustained MFU in the 45-52% range on large clusters, and they are explicit that this depends on keeping tensor parallelism *within* the NVLink domain and reserving the InfiniBand fabric for the less-frequent pipeline and data-parallel collectives. Their scaling curves fall off exactly when a communication-heavy dimension is forced across the slower link, which is the placement law of Cause 5 stated as a research result.

**GPT-3 and PaLM-class reports.** The headline large-language-model training reports (GPT-3, PaLM, OPT, LLaMA) cluster their achieved MFU in the ~30-55% band on named hardware (A100 and TPU pods), and every one of them attributes the gap between theoretical peak and achieved throughput substantially to communication overhead and the exposed collectives. Nobody hits 100% of peak; the game is minimizing the exposed fraction, and the fabric configuration is the first-order term. When a run reports 20% MFU on a modern cluster, the near-certain cause is that comms is exposed, and the near-certain root cause is the fabric.

**ZeRO/FSDP fitting large models.** DeepSpeed ZeRO and PyTorch FSDP make a 13B or 70B model *fit* by sharding optimizer state, gradients, and parameters, but sharding adds communication (all-gather of parameters, reduce-scatter of gradients) on top of the data-parallel all-reduce. On a slow fabric that added communication is precisely what kills you, which is why HYBRID_SHARD, keeping the shard-related traffic in-node, is the standard recipe for multi-node FSDP. The [DeepSpeed ZeRO deep-dive](/blog/machine-learning/open-source-library/deepspeed-zero-3d-parallelism-deep-dive) works the memory-versus-comms trade in full.

## When to reach for multi-node (and when not to)

Multi-node is a cost, and like every technique in this series it has a break-even. Say it plainly:

- **Do not go multi-node until you have saturated one node.** If a single 8-GPU node is at 50% MFU and the model fits, a second node buys you throughput *only if* the fabric can hide the all-reduce. If your fabric is Ethernet-only (no IB, no RoCE), scaling out data-parallel training will often lose you throughput, exactly the failure this post is about. Buy the interconnect before you buy the nodes.
- **Do go multi-node when the model does not fit on one node**, or when one node's throughput genuinely bottlenecks your schedule and you have a fast fabric. At that point the question is not *whether* but *how*: which parallelism dimension goes on NVLink, which goes on IB, and how big the global batch should be.
- **Match the parallelism to the fabric.** Tensor parallelism is the chattiest and must stay in-node on NVLink. Pipeline parallelism sends less and tolerates the fabric better. Data parallelism sends the full gradient once per step and lives or dies on the fabric's ability to hide it. Expert parallelism's all-to-all is fabric-heavy and needs the best interconnect you have. The [picking-a-parallelism-strategy post](/blog/machine-learning/distributed-training/why-distributed-training) turns this into a decision tree keyed on model size, cluster size, and interconnect.
- **If you cannot get IB or RoCE**, prefer techniques that reduce cross-node traffic: gradient compression, larger global batches (fewer syncs per token), HYBRID_SHARD to keep heavy traffic in-node, or simply staying on one bigger node. Do not brute-force data parallelism over sockets and expect it to scale; it will not.

The meta-rule: the interconnect is not a detail you tune after the fact. It is a first-class design input, on par with the model size and the GPU count. Choose the parallelism layout *around* the fabric you have, not the other way around.

## Key takeaways

- **Run `nccl-tests all_reduce_perf` before every multi-node campaign.** The busbw number at large message sizes is the single measurement that tells you IB (tens to hundreds of GB/s) versus sockets (~1-4 GB/s). It costs two minutes and saves days.
- **`NET/Socket` in the `NCCL_DEBUG=INFO` log means you are broken.** There is no acceptable multi-node run on TCP sockets when IB hardware exists. Pin IB with `NCCL_IB_HCA`, fix the interface with `NCCL_SOCKET_IFNAME`, and set `NCCL_IB_DISABLE=0`.
- **The all-reduce time is $2(N-1)/N \cdot S / B \approx 26$ GB divided by your achieved busbw.** That one division predicts whether multi-node flies or dies. On NVLink it is 0.1 s and free; on sockets it is 6+ s and fatal.
- **Comms hides only when it is smaller than compute.** The comms-to-compute ratio $r = T_\text{allreduce}/T_\text{compute}$ must stay well under 1. The fabric sets the numerator; the batch sets the denominator; you control both.
- **GPUDirect RDMA is a 2x.** Without `nvidia_peermem`, every transfer bounces through host memory and you get half the bandwidth. Confirm `GPU Direct RDMA Enabled` in the log.
- **Keep tensor parallelism in-node.** TP does an all-reduce inside every layer; forced across nodes it destroys throughput even on a perfect fabric. Keep TP $\leq$ GPUs-per-node.
- **Grow the global batch with the GPU count.** Holding it fixed halves per-GPU compute and doubles the comms fraction. Weak-scale: constant per-GPU batch, growing global batch, up to the critical batch size.
- **Diagnose in cost order:** nccl-tests, then NCCL_DEBUG, then the profiler, then topo and ibstat. Measure the isolated fabric before you profile the integrated run, and always re-measure the fix.

## Further reading

- [Why distributed training](/blog/machine-learning/distributed-training/why-distributed-training) - the four walls and the map of the whole series.
- [The interconnect physics](/blog/machine-learning/distributed-training/the-interconnect-physics) - NVLink, NVSwitch, InfiniBand, RoCE, and why placement matters.
- [Collectives from scratch](/blog/machine-learning/distributed-training/collectives-from-scratch) - the ring all-reduce and its $2(N-1)/N \cdot S$ byte volume, derived.
- [Overlapping compute and communication](/blog/machine-learning/distributed-training/overlapping-compute-and-communication) - why overlap is the whole game and when it saturates.
- [DDP from first principles](/blog/machine-learning/distributed-training/ddp-from-first-principles) - gradient all-reduce, bucketing, and the `no_sync()` accumulation path.
- [3D parallelism](/blog/machine-learning/distributed-training/3d-parallelism) - composing TP x PP x DP and the device-mesh placement law.
- [The distributed training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) - the capstone checklist that ties the whole series together.
- [Interconnects: NVLink, NVSwitch, InfiniBand, and RDMA](/blog/machine-learning/high-performance-computing/interconnects-nvlink-nvswitch-infiniband-and-rdma) - the HPC-pillar deep dive on the fabric hardware.
- [Collective communication and NCCL all-reduce from scratch](/blog/machine-learning/high-performance-computing/collective-communication-and-nccl-all-reduce-from-scratch) - a from-first-principles build of the collective this post depends on.
- The NVIDIA `nccl-tests` repository and the NCCL environment-variable documentation - the authoritative reference for every flag used above.

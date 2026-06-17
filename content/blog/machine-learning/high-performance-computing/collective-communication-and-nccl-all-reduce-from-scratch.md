---
title: "Collective Communication and NCCL: All-Reduce From Scratch"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Build every multi-GPU communication primitive from the ground up, derive why ring all-reduce moves a fixed number of bytes no matter how many GPUs you add, and learn how NCCL and DDP hide the cost under the backward pass."
tags:
  [
    "high-performance-computing",
    "gpu",
    "nccl",
    "all-reduce",
    "distributed-training",
    "collective-communication",
    "ddp",
    "deep-learning",
    "ml-systems",
    "pytorch",
  ]
category: "machine-learning"
subcategory: "High Performance Computing"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/collective-communication-and-nccl-all-reduce-from-scratch-1.png"
---

The first time you run a training job on eight GPUs instead of one, you expect to go eight times faster. You don't. You get maybe five and a half times faster, sometimes less, and if you wired the box up wrong — PCIe instead of NVLink, the wrong NCCL environment variables, a model whose layers are too small — you might get *three* times faster on eight GPUs, which means five of your expensive accelerators are sitting idle, melting electricity, waiting. The GPUs are not the bottleneck. The wires between them are.

This post is about those wires and the software that drives them. Specifically, it is about the single operation that dominates almost every multi-GPU training run: the **all-reduce**, the collective that takes the gradients computed independently on every GPU, sums them, and hands the identical sum back to all of them so that every GPU can take the same optimizer step. In standard data-parallel training, an all-reduce happens on *every* backward pass, on *every* step, for the *entire* gradient buffer — for a 7-billion-parameter model that is 14 GB of data flying across the interconnect tens of thousands of times over a training run. Get it wrong and it dwarfs your compute. Get it right and you barely notice it's there, because it hides underneath the backward pass you were going to run anyway.

We are going to build the whole thing from scratch. We will define the six collective communication primitives — **broadcast, reduce, all-reduce, all-gather, reduce-scatter, all-to-all** — and say exactly which kind of parallelism leans on each one. We will derive, with arithmetic you can check on a napkin, why the **ring all-reduce** moves exactly $2(N-1)/N \cdot S$ bytes per GPU and why that number stops growing once you have a handful of GPUs — the single most counterintuitive and important fact in distributed training. We will contrast ring against **tree** (bandwidth versus latency), look at how **NCCL** — NVIDIA's collective communication library, the thing PyTorch actually calls — picks an algorithm and a wire protocol for you, and then watch **PyTorch DDP** chop the gradient buffer into buckets and fire the all-reduce early so it overlaps the backward pass. Throughout, the running example is the spine of this whole series: take a Transformer of roughly 1 to 7 billion parameters and scale its gradient synchronization from 1 GPU to 8 to many.

![A comparison grid of the six collective communication primitives showing what each one does, how many bytes per GPU it moves, and which parallelism strategy uses it](/imgs/blogs/collective-communication-and-nccl-all-reduce-from-scratch-1.png)

If you have read [why HPC is the bottleneck for modern AI](/blog/machine-learning/high-performance-computing/why-hpc-is-the-bottleneck-for-modern-ai), you already know the frame: there are three walls — compute, memory bandwidth, and communication — and a training job is only as fast as its slowest wall. Single-GPU work (precision, kernels, fusion) chips away at the first two walls. This post is the third wall. By the end you will be able to look at a multi-GPU job, compute how many bytes the all-reduce *should* move, measure how many it *actually* moves, decide whether you are bandwidth-bound or latency-bound, and choose the algorithm, protocol, and bucketing that gets you back to near-linear scaling. Let's build it.

## 1. The six collectives: a vocabulary for moving data between GPUs

Before all-reduce, we need the vocabulary. A **collective communication operation** (a "collective") is a coordinated data movement involving a whole *group* of processes at once — here, one process per GPU — as opposed to a point-to-point send from one rank to one other rank. The word **rank** is the standard term for "the integer ID of one process in the group"; if you have 8 GPUs you have ranks 0 through 7, and the **world size** $N$ is 8. Every collective is, at heart, a function that takes a particular layout of data spread across the ranks and produces a new layout. That is all they are. Once you see them as layout transforms, the whole zoo becomes memorable.

There are six you need to know. Let $S$ be the size in bytes of the buffer involved (for gradients, $S$ is the size of the full gradient tensor), and let $N$ be the world size.

**Broadcast.** One rank — usually rank 0 — has a buffer; afterward, every rank has an identical copy. This is how you make sure all GPUs start training from the same initial weights: rank 0 initializes the model, broadcasts the parameters, done. Bytes leaving rank 0: $S$ (it sends one copy that gets fanned out). Used by: weight initialization at the start of any data-parallel job, and broadcasting a freshly loaded checkpoint.

**Reduce.** Every rank has a buffer of the same shape; afterward, *one* rank (say rank 0) holds the element-wise sum (or max, or product — the *reduction operator*) of all of them, and the others hold nothing useful. Bytes that must arrive at rank 0: $(N-1)/N \cdot S$ in the optimal case. Used by: gathering a scalar metric (total loss, total token count) onto rank 0 for logging, where only one rank needs the answer.

**All-reduce.** Like reduce, but afterward *every* rank holds the identical sum. This is the workhorse of data-parallel training: each GPU computed gradients on its own slice of the batch, and now all of them need the average (sum then divide) so they can step in lockstep. All-reduce is logically a reduce followed by a broadcast — but, as we will see, you should never implement it that way, because a clever ring does it in one fused pass for half the bytes. Optimal bytes per GPU: $2(N-1)/N \cdot S$. Used by: data-parallel gradient synchronization. This is the one we obsess over.

**All-gather.** Each rank starts with a distinct *shard* (a $1/N$ slice) of a buffer; afterward every rank has the full concatenation of all shards. No arithmetic, just collection. Bytes received per GPU: $(N-1)/N \cdot S$ (you already have your own $1/N$). Used by: **FSDP** and **ZeRO-3**, where each GPU permanently stores only a shard of the weights and must all-gather the full parameter tensor just before computing a layer, then throw it away after.

**Reduce-scatter.** The mirror image of all-gather and, as we will see, half of an all-reduce. Each rank starts with a full buffer; afterward each rank holds the *summed* value of only its own $1/N$ shard. So it reduces (sums across ranks) and scatters (each rank keeps a different piece) in one operation. Bytes per GPU: $(N-1)/N \cdot S$. Used by: FSDP/ZeRO gradient reduction — each GPU only needs the summed gradient for the shard of weights it owns, so a reduce-scatter is exactly right and moves half the bytes of a full all-reduce.

**All-to-all.** The most general and most expensive. Every rank holds $N$ slices, one destined for each rank; afterward every rank holds the $N$ slices that were destined for it — a full transpose of the data across the group. Bytes per GPU: $(N-1)/N \cdot S$. Used by: **Mixture-of-Experts (MoE)** routing, where each GPU holds some tokens that need to be dispatched to whichever GPU hosts the expert they were routed to, and the results gathered back.

The figure above lays all six out as a table — what each does, the optimal bytes per GPU, and the parallelism mode that needs it. Notice the pattern: five of the six move $(N-1)/N \cdot S$ bytes per GPU in their optimal implementation, which approaches $S$ for large $N$ (you move essentially the whole buffer once). All-reduce is the odd one out at $2(N-1)/N \cdot S$ — *twice* the others — because it is doing two jobs (sum and distribute) where the others do one. Hold onto that factor of two; deriving exactly where it comes from is the heart of this post.

A note on the reduction operator. "Reduce" in all of these means an associative, commutative binary operation applied element-wise: **sum** (the default and the one gradients use), **max**, **min**, **product**, or a logical op. Associativity is what lets us reorder the work across the ring — GPU 0 can add to a partial sum that GPU 3 started, and the answer is the same. This is not a minor detail; it is the mathematical license for everything that follows.

It helps to anchor the difference between "every rank gets the answer" and "one rank gets the answer," because it is the only thing separating four of these six. Broadcast and all-gather *distribute* (no arithmetic); reduce and reduce-scatter *combine* (arithmetic, then keep one piece); all-reduce *combines then distributes to all*; all-to-all *permutes* (each rank's data fans out to everyone, transposed). If you ever forget which is which, ask two questions: "is there a reduction operator involved?" and "does everyone end up with the full result, or just a shard, or just one rank?" Those two axes pin down all six. Data-parallel gradient sync needs a reduction (sum the gradients) and needs everyone to have the result (so they step identically) — that is precisely all-reduce, and nothing else fits.

#### Worked example: how many bytes does a logging reduce versus a gradient all-reduce move

Suppose you are training the 7B model on 8 GPUs and, every step, you do two collectives. First, you `all_reduce` the scalar loss for your progress bar: $S = 4$ bytes. The optimal per-GPU volume is $2(8-1)/8 \times 4 = 7$ bytes — utterly free on bandwidth; its entire cost is the message latency of a handful of microseconds, which is why NCCL will run it as a *tree* with the low-latency protocol. Second, you sync the 14 GB gradient buffer with `all_reduce`: $S = 14 \times 10^9$ bytes, and the per-GPU volume is $1.75 \times 14 = 24.5$ GB. The ratio between the two collectives' byte volumes is about $3.5$ *billion* to one. They are the same API call, `dist.all_reduce`, but they live in completely different cost regimes — and a good library treats them completely differently. Holding that contrast in your head is most of what you need to reason about distributed performance: find the big buffer, make *its* all-reduce cheap, and stop worrying about the small ones.

Here is the punchline you should carry into the next section. **In standard data-parallel training the only collective on the hot path is all-reduce, and it runs once per step on the full gradient buffer.** Everything else — the broadcast at startup, the occasional reduce for logging — is negligible. So if you want multi-GPU training to scale, you must make all-reduce cheap. The rest of this post is about exactly that.

## 2. The ring: a topology where every GPU only talks to its neighbor

The naive way to implement all-reduce is to send every GPU's buffer to rank 0, have rank 0 sum them, and broadcast the result back. On $N$ GPUs that funnels $(N-1) \cdot S$ bytes *into* rank 0 and $(N-1) \cdot S$ back out — and crucially, rank 0's single network link is the bottleneck for all of it. Double the GPUs and rank 0's link has twice the work. This does not scale; it is the textbook example of a hot spot. We need an algorithm where no single link is special and the total work spreads across all the links in the system.

The **ring** is that algorithm. The mental model is simple: arrange the $N$ GPUs in a logical circle, where GPU $i$ sends only to GPU $(i+1) \bmod N$ — its right-hand neighbor — and receives only from GPU $(i-1) \bmod N$. Every GPU has exactly one outgoing link it ever uses and one incoming link. There is no center, no hot spot. If each link runs at bandwidth $B$, then all $N$ links run *simultaneously*, so the aggregate bandwidth of the system is $N \cdot B$. That is the whole trick: a ring turns $N$ independent links into one big pipe.

![A grid showing four GPUs arranged in a ring where each one holds four chunks of its gradient buffer and sends one chunk to its right neighbor each step](/imgs/blogs/collective-communication-and-nccl-all-reduce-from-scratch-2.png)

The figure shows four GPUs, each holding its full gradient buffer split into four chunks (because there are four GPUs — in general you split into $N$ chunks). The key idea — and this is the part people miss — is that we do **not** send a GPU's whole buffer to its neighbor at once. We split the buffer of size $S$ into $N$ chunks of size $S/N$ each, and the ring moves *one chunk at a time*. This chunking is what makes the ring bandwidth-optimal. Imagine the buffer as a pie cut into $N$ slices; on each step every GPU passes one slice to its right and receives a different slice from its left, and over $N-1$ steps the slices circulate all the way around. By the time a slice has visited every GPU, it has been summed across all of them.

Why split into exactly $N$ chunks and not, say, 2 or $2N$? Because we want every GPU's outgoing link to be busy on every step, and we want each step to move the same amount of data. With $N$ chunks and $N$ GPUs, on every step each GPU sends exactly one chunk ($S/N$ bytes) and receives one chunk, so all links are saturated and balanced. Fewer chunks and the pipeline can't fill; more chunks and you pay more per-message latency overhead for no bandwidth benefit. $N$ chunks is the sweet spot, and it falls out of the math we are about to do.

Let me state the ring's two phases plainly before we trace them. A ring all-reduce is **reduce-scatter then all-gather**:

1. **Reduce-scatter phase** ($N-1$ steps): the chunks circulate and accumulate, so that at the end, GPU $i$ holds the *fully summed* version of exactly one chunk — chunk $i$ — and partial sums of the others. Each GPU ends up "owning" one finished slice of the answer.
2. **All-gather phase** ($N-1$ steps): now every GPU owns one finished slice; we circulate those finished slices around the ring so that, after another $N-1$ steps, every GPU has every finished slice — the full summed buffer.

That is it. Two phases, $N-1$ steps each, $2(N-1)$ steps total, and on every step every link moves exactly $S/N$ bytes. The total bytes each GPU sends is therefore $2(N-1) \cdot S/N = 2(N-1)/N \cdot S$. We will derive this carefully in the next section, but you can already see the shape of it: $2$ phases, $N-1$ steps, $S/N$ per step.

The reason this is called "bandwidth-optimal" is worth pausing on. There is a hard lower bound on how few bytes any all-reduce can move per GPU. Each GPU must end up with information from all $N-1$ other GPUs, so at minimum it must *receive* about $(N-1)/N \cdot S$ bytes (everyone else's contribution to the parts it didn't compute). And in an all-reduce, the result must also be *distributed* to it, which costs another $(N-1)/N \cdot S$. Add them and you get $2(N-1)/N \cdot S$ — exactly what the ring achieves. The ring is not merely good; it provably cannot be beaten on bytes-per-GPU. (It can be beaten on *latency*, which is where trees come in — section 5.) This bandwidth-optimality result is the one that the Baidu team popularized for deep learning in 2017, adapting an algorithm long known in the HPC/MPI community, and it is why every serious training framework uses a ring under the hood for large buffers.

## 3. Deriving the bandwidth-optimal cost from scratch

Now the science. Let's trace a 4-GPU ring all-reduce step by step with concrete chunk labels, then read off the cost formula, then prove the headline claim: the per-GPU byte volume is independent of $N$ at the bandwidth limit.

![A timeline showing ring all-reduce decomposed into a reduce-scatter phase of N minus one steps followed by an all-gather phase of N minus one steps](/imgs/blogs/collective-communication-and-nccl-all-reduce-from-scratch-3.png)

Set up the notation. We have $N = 4$ GPUs, ranks 0–3. Each GPU holds its own gradient buffer split into 4 chunks. Write chunk $c$ on GPU $g$ as $g_c$ — so GPU 0 holds $\{0_0, 0_1, 0_2, 0_3\}$, GPU 1 holds $\{1_0, 1_1, 1_2, 1_3\}$, and so on. The final answer we want, in chunk $c$, is the sum $\Sigma_c = 0_c + 1_c + 2_c + 3_c$, and at the end every GPU must hold all four $\Sigma_c$.

**Reduce-scatter phase, step by step.** The rule each step: GPU $i$ sends one chunk to GPU $i+1$, and GPU $i+1$ *adds* the received chunk into its own corresponding chunk. We stagger which chunk each GPU sends so that, like a relay, each chunk accumulates as it travels.

- **Step 1.** GPU $i$ sends chunk $i$ to GPU $i+1$. So GPU 0 sends $0_0$ to GPU 1, which sets its chunk 0 to $1_0 + 0_0$. GPU 1 sends $1_1$ to GPU 2 ($\to 2_1 + 1_1$). GPU 2 sends $2_2$ to GPU 3 ($\to 3_2 + 2_2$). GPU 3 sends $3_3$ to GPU 0 ($\to 0_3 + 3_3$). After this step, four chunks each hold a sum of two terms.
- **Step 2.** Each GPU sends the chunk it *just updated* onward. GPU 1 sends its chunk 0 ($1_0 + 0_0$) to GPU 2, which adds its $2_0$: now chunk 0 on GPU 2 is $2_0 + 1_0 + 0_0$ — a sum of three. The same happens around the ring for the other chunks.
- **Step 3** (the last reduce-scatter step, since $N-1 = 3$). The three-term partial sums travel one more hop and pick up the final term. Now chunk 0 is fully summed ($\Sigma_0 = 0_0+1_0+2_0+3_0$) and lives on GPU 3. Chunk 1's full sum $\Sigma_1$ lives on GPU 0, $\Sigma_2$ on GPU 1, $\Sigma_3$ on GPU 2.

After $N-1 = 3$ steps, every GPU owns exactly one fully-summed chunk. That is the "scatter" in reduce-scatter: the reduced result is scattered, one chunk per GPU. No GPU has the whole answer yet — each has $1/N$ of it, finished.

**All-gather phase, step by step.** Now we just need to copy each finished chunk to everyone. Same ring, same one-chunk-per-step rhythm, but now we *overwrite* rather than add (the chunk is already final).

- **Step 1.** Each GPU sends its finished chunk to its right neighbor, which stores it. GPU 3 sends $\Sigma_0$ to GPU 0; GPU 0 sends $\Sigma_1$ to GPU 1; etc. Now two GPUs hold each finished chunk.
- **Steps 2 and 3.** The finished chunks keep circulating. After $N-1 = 3$ steps, every chunk has visited every GPU, so every GPU holds all four $\Sigma_c$ — the complete summed buffer.

Total: $3 + 3 = 6 = 2(N-1)$ steps. On each step, each GPU sent exactly one chunk of $S/N$ bytes. So the total bytes each GPU sends is

$$ \text{bytes per GPU} = 2(N-1) \cdot \frac{S}{N} = \frac{2(N-1)}{N} \cdot S. $$

There it is, derived rather than asserted. The factor of 2 is the two phases (you move the data once to sum it, once to distribute it). The $(N-1)$ is the number of hops in each phase. The $1/N$ is the chunk size. This counts bytes *sent*; bytes *received* per GPU is identical by symmetry.

Now the time. If each link runs at bandwidth $B$ (bytes per second) and we charge a fixed per-message **latency** $\alpha$ (seconds) for each of the $2(N-1)$ steps to account for the cost of initiating a transfer, the wall-clock time is

$$ T_\text{ring} = \underbrace{2(N-1)\,\alpha}_{\text{latency term}} + \underbrace{\frac{2(N-1)}{N} \cdot \frac{S}{B}}_{\text{bandwidth term}}. $$

This is the standard $\alpha$-$\beta$ cost model from parallel computing ($\beta = 1/B$ is the per-byte cost). The two terms tell the whole story:

**The bandwidth term is nearly independent of $N$.** Look at the factor $2(N-1)/N = 2 - 2/N$. At $N=2$ it is $1.0$; at $N=8$ it is $1.75$; at $N=64$ it is $1.97$; as $N \to \infty$ it approaches $2$ and *stops*. So for a fixed buffer $S$, the bandwidth-limited time to all-reduce barely grows as you add GPUs — going from 8 to 64 GPUs only raises the factor from 1.75 to 1.97, about a 12% increase, even though you added 8× the GPUs. **This is the single most important fact about ring all-reduce.** It is why data-parallel training can scale to thousands of GPUs without the gradient sync exploding. Per-GPU communication cost is essentially constant; you are not paying more to talk to more peers, because the ring spreads the work across all the new links the new GPUs brought with them.

**The latency term grows linearly with $N$.** The $2(N-1)\alpha$ term scales with $N$, and for *small* buffers (where $S/B$ is tiny) this term dominates. A 64-GPU ring pays 126 message latencies in sequence. If your buffer is a single small tensor, the ring is the wrong choice — and that is precisely the regime where a tree wins (section 5).

#### Worked example: the all-reduce time for a 7B model on an 8-GPU node

Let's put numbers on it. A 7-billion-parameter model in bf16 has gradients of size $S = 7 \times 10^9 \times 2\text{ bytes} = 14\text{ GB}$. Suppose we run on an 8-GPU NVLink node where each GPU's effective all-reduce link bandwidth (the "bus bandwidth" NCCL reports, after the ring overhead is accounted for) is about $B = 200\text{ GB/s}$. Take latency $\alpha \approx 5\,\mu s$ per step, which is generous for NVLink.

Bandwidth term: $\frac{2(8-1)}{8} \cdot \frac{14\text{ GB}}{200\text{ GB/s}} = 1.75 \times 0.070\text{ s} = 0.1225\text{ s} \approx 123\text{ ms}$.

Latency term: $2(8-1) \times 5\,\mu s = 14 \times 5\,\mu s = 70\,\mu s$, utterly negligible against 123 ms.

So one full-buffer all-reduce costs about **123 ms** on this node. Per step. If a training step's forward+backward compute is, say, 400 ms, then a *naive* implementation that does the all-reduce after the backward completes would add 123 ms of pure stall — a 30% slowdown, five idle GPU-seconds for every sixteen of work. That is the cost we are going to hide with bucketing in section 6. But first, notice that the latency term is irrelevant here because $S$ is huge; for gradient all-reduce of a large model, you are firmly bandwidth-bound, which is exactly the regime the ring is optimal for.

#### Worked example: why the factor stops growing — the same model at 64 GPUs

Now scale the same 7B model from 8 to 64 GPUs (8 nodes of 8, say). The buffer $S$ is still 14 GB — every GPU still holds the full gradient. The factor goes from $1.75$ to $2(63)/64 = 1.969$. If the *bus* bandwidth per GPU stayed at 200 GB/s, the bandwidth term would rise from 123 ms only to $1.969 \times 0.070\text{ s} = 138\text{ ms}$ — a 12% increase for 8× the GPUs. The per-GPU cost is nearly flat. *That* is linear scaling: the all-reduce cost per step is roughly constant, so your throughput per GPU stays high.

The catch in the real world is the word "if." Inside a node, links are NVLink at hundreds of GB/s. *Between* nodes, you cross InfiniBand, where per-rank bandwidth is more like 12–25 GB/s. So the bus bandwidth $B$ that NCCL achieves on a 64-GPU job is gated by the slowest tier in the ring — the inter-node hops. The factor $2(N-1)/N$ behaves; it is $B$ that falls off a cliff when the ring has to cross the slow network. We will quantify that in the case studies, and the fix — topology-aware hierarchical algorithms — is what NCCL does for you automatically. The interconnect physics behind this is the subject of [interconnects: NVLink, NVSwitch, InfiniBand and RDMA](/blog/machine-learning/high-performance-computing/interconnects-nvlink-nvswitch-infiniband-and-rdma); here we just need to know that $B$ is set by the wires.

### The factor that stops growing — tabulated

It is worth staring at the convergence of $2(N-1)/N$ directly, because it is so central and so counterintuitive that most engineers don't believe it until they see the table. The naive expectation — "more GPUs means more peers to talk to, so communication must grow with $N$" — is exactly wrong for a ring. The factor approaches a hard ceiling of 2 and never exceeds it. Hold the buffer fixed at $S = 4$ GB and a per-GPU bus bandwidth of 200 GB/s, and watch the per-GPU bytes and time barely move as you scale from 2 GPUs to a hypothetical infinite cluster.

![A comparison matrix showing that the ring per-GPU byte factor rises from one toward two and then flattens so cost is nearly independent of GPU count](/imgs/blogs/collective-communication-and-nccl-all-reduce-from-scratch-5.png)

The figure makes the convergence concrete: at $N=2$ the factor is exactly 1.00 (you send your half once, receive the other half once); at $N=4$ it is 1.50; at $N=8$ it is 1.75; at $N=64$ it is 1.97; and as $N \to \infty$ it asymptotes to 2.00 and stops. The time to all-reduce a 4 GB buffer at 200 GB/s correspondingly creeps from 20 ms at $N=2$ to a hard cap of 40 ms — *doubling at most*, no matter whether you have 8 GPUs or 8,000. Compare that against the naive funnel-to-rank-0 algorithm, whose cost at rank 0's link grows as $(N-1) \cdot S$ — *linearly* — so going from 8 to 64 GPUs makes the naive version 8× slower while the ring stays essentially flat. The whole reason ring all-reduce is the default in every framework is captured in this one asymptote.

Why does the factor have this shape? Because $2(N-1)/N = 2 - 2/N$. The "2" is the two phases (reduce-scatter and all-gather) that any all-reduce must perform — you cannot do better than moving the data once to combine it and once to share the combined result. The "$-2/N$" is a *discount* that shrinks as $N$ grows: with more GPUs, each GPU's chunk $S/N$ is smaller, and on the very first step of each phase a GPU effectively contributes a chunk it already owns "for free," which is the $1/N$ saving per phase. As $N \to \infty$ that discount vanishes and you converge to the pure two-phase cost of $2S$. There is no term anywhere in the formula that scales *up* with $N$ on the bandwidth side. That is the mathematical reason data parallelism scales — and it is why, when scaling *doesn't* hold in practice, you should look at the bandwidth $B$ (the wire) and the latency term and the compute-to-comm ratio, never at the byte-count factor, which is innocent.

| World size $N$ | Factor $2(N-1)/N$ | Bytes / GPU (for $S$ = 4 GB) | Time at 200 GB/s |
|---|---|---|---|
| 2 | 1.00 | 4.0 GB | ~20 ms |
| 4 | 1.50 | 6.0 GB | ~30 ms |
| 8 | 1.75 | 7.0 GB | ~35 ms |
| 64 | 1.97 | 7.9 GB | ~39 ms |
| $\to \infty$ | 2.00 (cap) | 8.0 GB | ~40 ms (cap) |

## 4. Hand-rolling a ring all-reduce in PyTorch

Theory is cheap. Let's actually build a ring all-reduce out of point-to-point sends and receives, so the formula stops being abstract. PyTorch's `torch.distributed` gives us `dist.send`, `dist.recv` (blocking) and `dist.isend`, `dist.irecv` (non-blocking), which is all we need. First, the boilerplate every distributed PyTorch program starts with — initializing the **process group**, the object that knows the world size, this process's rank, and which backend (here NCCL) to use.

```python
import os
import torch
import torch.distributed as dist

def setup():
    # torchrun sets RANK, LOCAL_RANK, WORLD_SIZE in the environment.
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)   # pin this process to one GPU
    return dist.get_rank(), dist.get_world_size(), local_rank

def cleanup():
    dist.destroy_process_group()
```

`dist.init_process_group("nccl")` is the line that picks NCCL as the transport. It blocks until every process in the group has called it (a rendezvous), then NCCL discovers the topology — which GPUs are on NVLink, which are across PCIe, which are on other nodes over InfiniBand — and builds its internal rings and trees. The `LOCAL_RANK` dance pins each process to a distinct GPU on its node; getting this wrong (two processes on one GPU) is the single most common "my distributed job hangs" bug.

You launch this with `torchrun`, which spawns one process per GPU and sets those environment variables:

```bash
# One node, 8 GPUs:
torchrun --nproc_per_node=8 --nnodes=1 train.py

# Two nodes, 8 GPUs each, rendezvous over a host:port:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
         --rdzv_backend=c10d --rdzv_endpoint=host0:29500 train.py
```

Now the ring itself. We implement the two phases exactly as derived: reduce-scatter, then all-gather. We split the flat tensor into `world_size` chunks and circulate them.

```python
def ring_all_reduce(tensor, rank, world_size):
    """Hand-rolled bandwidth-optimal all-reduce (sum) over a ring.
    Mutates `tensor` in place so every rank ends with the global sum."""
    chunks = list(tensor.chunk(world_size))   # N views, each ~S/N bytes
    send_to = (rank + 1) % world_size         # my right neighbor
    recv_from = (rank - 1) % world_size       # my left neighbor

    # ---- Phase 1: reduce-scatter (N-1 steps) ----
    # On step k, rank r sends chunk (r - k) and receives into chunk (r-1-k),
    # adding the incoming partial sum to its own.
    for step in range(world_size - 1):
        send_idx = (rank - step) % world_size
        recv_idx = (rank - step - 1) % world_size
        recv_buf = torch.empty_like(chunks[recv_idx])
        # Post the recv first to avoid deadlock, then send.
        recv_op = dist.P2POp(dist.irecv, recv_buf, recv_from)
        send_op = dist.P2POp(dist.isend, chunks[send_idx].contiguous(), send_to)
        for req in dist.batch_isend_irecv([send_op, recv_op]):
            req.wait()
        chunks[recv_idx] += recv_buf          # accumulate

    # After phase 1, chunk (rank + 1) % N on this rank is fully summed.
    # ---- Phase 2: all-gather (N-1 steps) ----
    # Same ring, but now copy finished chunks instead of adding.
    for step in range(world_size - 1):
        send_idx = (rank + 1 - step) % world_size
        recv_idx = (rank - step) % world_size
        recv_buf = torch.empty_like(chunks[recv_idx])
        recv_op = dist.P2POp(dist.irecv, recv_buf, recv_from)
        send_op = dist.P2POp(dist.isend, chunks[send_idx].contiguous(), send_to)
        for req in dist.batch_isend_irecv([send_op, recv_op]):
            req.wait()
        chunks[recv_idx].copy_(recv_buf)      # overwrite, don't add

    return tensor
```

A few things to internalize. We **post the receive before (or concurrently with) the send** using `batch_isend_irecv`; if every rank tried a blocking `send` first, they would all wait for a matching `recv` that nobody has posted yet, and the job deadlocks. This is the classic ring-deadlock, and the fix is always "non-blocking, post both, then wait." Each step moves exactly one chunk of $S/N$ bytes in each direction, and there are $2(N-1)$ steps — so this code moves exactly the $2(N-1)/N \cdot S$ bytes we derived. You can sanity-check by summing the bytes sent.

We can verify our hand-rolled version against the library's `dist.all_reduce`, which is the function you would actually use in production:

```python
def main():
    rank, world_size, local_rank = setup()
    torch.manual_seed(rank)                 # different data per rank
    x = torch.randn(1_000_000, device="cuda")  # ~4 MB fp32

    # The real thing: one call, NCCL picks ring/tree + protocol for us.
    ref = x.clone()
    dist.all_reduce(ref, op=dist.ReduceOp.SUM)

    # Our from-scratch ring:
    ours = x.clone()
    ring_all_reduce(ours, rank, world_size)

    err = (ref - ours).abs().max().item()
    if rank == 0:
        print(f"max abs diff vs NCCL all_reduce: {err:.2e}")  # ~1e-5, fp rounding
    cleanup()

if __name__ == "__main__":
    main()
```

The `dist.all_reduce(ref, op=dist.ReduceOp.SUM)` call is what you write in real code — one line, and NCCL handles chunking, ring-vs-tree selection, protocol, and topology. Our hand-rolled version exists to demystify what that one line does, and to make the byte-count concrete. The tiny `1e-5` difference is just floating-point non-associativity: the library sums chunks in a different order than we do, and fp addition isn't perfectly associative. For gradients this is irrelevant; for some reductions it matters, which is why NCCL is deterministic-per-configuration but not bit-identical across algorithms.

One more detail you will hit in practice: to average gradients (not just sum them) in data-parallel training, you all-reduce with SUM and then divide by `world_size`, or use `op=dist.ReduceOp.AVG` if your NCCL version supports it. DDP does the averaging for you, but when you hand-roll, remember the divide.

## 5. Ring versus tree: bandwidth-optimal versus latency-optimal

The ring is bandwidth-optimal, but its Achilles' heel is the latency term: $2(N-1)$ sequential steps. For a 1,024-GPU job that is 2,046 hops in series, and each hop pays the message-initiation latency $\alpha$. When the buffer is small — a single tiny gradient tensor, a scalar metric, the first few buckets of a model — the $S/B$ bandwidth term is negligible and the $2(N-1)\alpha$ latency term is *everything*. A ring is the wrong tool for small messages at large scale.

The alternative is a **tree** all-reduce. Instead of a circle, arrange the GPUs as a binary tree. The reduction flows *up* the tree: leaves send to their parents, parents sum and send to grandparents, until the root has the global sum; then the result flows *down* the tree by broadcast. The depth of a binary tree over $N$ nodes is $\log_2 N$, and each phase (up, down) takes about $\log_2 N$ steps, so the total number of sequential hops is about $2\log_2 N$ — **logarithmic in $N$, not linear.**

![A side by side comparison of ring all-reduce which is bandwidth optimal against tree all-reduce which is latency optimal for small messages](/imgs/blogs/collective-communication-and-nccl-all-reduce-from-scratch-4.png)

The figure contrasts the two. Put numbers on the hop counts: at $N = 1{,}024$, the ring takes $2(N-1) = 2{,}046$ sequential hops, while the tree takes $2\log_2(1024) = 2 \times 10 = 20$ hops. For a latency-dominated small message, that is a **100× reduction in latency.** The tree wins decisively when the message is small.

But the tree pays for that latency win with bandwidth. In a tree, the root and the nodes near it carry disproportionate traffic — the root must receive and re-send the full reduced buffer, so internal links move roughly $2S$ per GPU near the top, and the tree does not spread the load across all links the way a ring does. So for *large* buffers, where you are bandwidth-bound, the tree's effective bandwidth is worse than the ring's, and the ring wins. The two algorithms occupy opposite corners of the design space:

| Property | Ring all-reduce | Tree all-reduce |
|---|---|---|
| Sequential hops (latency) | $2(N-1)$, grows linearly | $2\log_2 N$, grows logarithmically |
| Bytes per GPU (bandwidth) | $2(N-1)/N \cdot S \to 2S$, optimal | $\sim 2S$ near root, load not balanced |
| Best for | Large buffers (gradient sync) | Small buffers, large $N$, latency-bound |
| Bottleneck | Latency at large $N$, small $S$ | Bandwidth at large $S$ |
| Hot spots | None — symmetric | Root and near-root links |
| Crossover (rough) | Wins above ~a few MB | Wins below ~hundreds of KB |

The practical consequence: **the right algorithm depends on the message size**, and a good library switches between them. Small message? Tree, to dodge the latency. Large message? Ring, to maximize bandwidth. There is a crossover buffer size, typically somewhere in the hundreds-of-kilobytes to low-megabytes range depending on the hardware, below which tree wins and above which ring wins. You do not want to be choosing this by hand on a per-message basis — which is exactly the problem NCCL solves.

#### Worked example: choosing ring vs tree for two different all-reduces in one model

Imagine training that 7B model on 64 GPUs. Two different all-reduces happen. First, the **gradient all-reduce**: $S = 14$ GB, enormous, firmly bandwidth-bound. Ring wins; the $2(N-1)/N$ factor of 1.97 is fine and the latency of 126 hops at ~5 µs is $\sim 0.6$ ms, lost in the noise against a ~700 ms bandwidth term. Second, suppose you also all-reduce a **scalar loss for logging** every step: $S = 4$ bytes. The bandwidth term is essentially zero; the cost is pure latency. A ring would pay 126 sequential hops; a tree pays $2\log_2 64 = 12$ hops. The tree is roughly 10× faster for this tiny message. *Same job, same GPUs, two different optimal algorithms*, chosen by buffer size. This is why "which algorithm" is not a global setting — it is per-collective, per-size. And it is why you should let NCCL decide rather than pinning `NCCL_ALGO` unless you are debugging.

There is a third option worth naming because frameworks use it at scale: **hierarchical / double-binary-tree and ring-of-rings** algorithms that combine the two — a ring within each NVLink node (bandwidth) and a tree across nodes over InfiniBand (latency, and to keep the slow inter-node links off the critical path). NCCL's default for many large topologies is a double binary tree, which gets near-ring bandwidth *and* logarithmic latency by cleverly using two complementary trees so that every link is busy. You do not have to implement these; you have to know they exist so the next section makes sense.

Why does the double-binary-tree trick recover bandwidth that a single tree throws away? In a single binary tree, the leaf nodes have only one link doing work (up to the parent) while interior nodes near the root are the bottleneck — so roughly half the GPUs (the leaves) under-use their outgoing bandwidth. NCCL builds *two* trees over the same GPUs such that a node that is a leaf in the first tree is an interior node in the second, and runs both trees concurrently on different halves of the buffer. Now every GPU's links are busy in at least one of the two trees, the load balances across all links the way a ring does, and you keep the $\log N$ latency. It is a genuinely clever construction, and it is why "tree" in modern NCCL is not the bandwidth-loser the simple-binary-tree analysis suggests. The practical upshot for you: at large $N$ with mixed message sizes, NCCL's tree path is competitive on bandwidth *and* far better on latency, which is why it is often the default for multi-node jobs. Trust the tuner; verify with `nccl-tests`.

## 6. DDP gradient bucketing: hiding the all-reduce under the backward pass

We have an all-reduce that costs ~123 ms for our 7B model on 8 GPUs. If we wait for the backward pass to fully finish and *then* fire one giant all-reduce, that 123 ms is pure stall — the GPUs compute nothing while the wires work. The whole game of efficient data-parallel training is to make that stall disappear by **overlapping communication with computation**. PyTorch's `DistributedDataParallel` (DDP) does this with **gradient bucketing**, and understanding it is the difference between 60% and 95% scaling efficiency.

Here is the key observation. The backward pass computes gradients *layer by layer, from the output back to the input*. The gradient for the last layer is ready almost immediately; the gradient for the first layer isn't ready until the very end. So instead of waiting for *all* gradients, we can start all-reducing the *last* layer's gradient the instant it's computed, while the backward pass is still grinding through earlier layers. The communication for late layers overlaps the computation for early layers. By the time backward finishes, most of the all-reduce is already done, hiding underneath compute that was happening anyway.

![A timeline showing how DDP fills gradient buckets during the backward pass and fires an all-reduce on each bucket so communication overlaps later gradient computation](/imgs/blogs/collective-communication-and-nccl-all-reduce-from-scratch-6.png)

But there's a wrinkle: a 7B model has hundreds of individual parameter tensors, and firing a separate all-reduce for each tiny tensor would drown in per-message latency — remember the $2(N-1)\alpha$ term punishes many small messages. So DDP **buckets**: it groups consecutive gradients into fixed-size buckets (default ~25 MB), and fires *one* all-reduce per bucket once the bucket is full. This batches the latency cost (one message per 25 MB, not per tensor) while still allowing overlap (you don't wait for the whole model, just for a bucket). The figure shows the dance: as each bucket fills during backprop, its all-reduce launches asynchronously and runs concurrently while the next bucket's gradients are still being computed. Only the *final* bucket — the first layers' gradients, computed last — can't be overlapped, because there's no more computation to hide behind it. That last bucket is the irreducible exposed cost.

The mechanism, concretely: DDP registers an **autograd hook** on every parameter. When a gradient is produced, the hook marks it ready and copies it into its bucket. When the last gradient in a bucket becomes ready, DDP launches an asynchronous `all_reduce` on that bucket. The CUDA stream for communication runs in parallel with the compute stream. At `optimizer.step()`, DDP waits for all the in-flight all-reduces to finish — but by then almost all of them already have.

Using DDP is three lines. The important flag is `gradient_as_bucket_view`:

```python
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

def build_ddp_model(rank, local_rank):
    model = MyTransformer().to(local_rank)     # 7B params, say
    ddp_model = DDP(
        model,
        device_ids=[local_rank],
        gradient_as_bucket_view=True,  # grads point INTO the bucket: no extra copy
        bucket_cap_mb=25,              # bucket size; tune for your interconnect
        broadcast_buffers=False,       # skip buffer sync if you have no BN running stats
    )
    return ddp_model

# The training loop looks exactly like single-GPU. DDP handles sync in the
# backward() call via its autograd hooks. No explicit all_reduce in sight.
def train_step(ddp_model, batch, optimizer, loss_fn):
    optimizer.zero_grad(set_to_none=True)
    out = ddp_model(batch["x"])
    loss = loss_fn(out, batch["y"])
    loss.backward()        # <-- bucketed all-reduces fire DURING this call
    optimizer.step()       # <-- waits for any still-in-flight bucket all-reduce
    return loss.item()
```

`gradient_as_bucket_view=True` is the flag people forget. Normally DDP keeps gradients and the communication buckets as *separate* memory, which means an extra copy from `param.grad` into the bucket before each all-reduce — wasted bandwidth and memory. With `gradient_as_bucket_view=True`, the `.grad` tensors are *views into* the bucket's flat buffer, so the gradient is written directly where the all-reduce will read it. No copy, and you save a full gradient buffer's worth of memory (14 GB for our 7B model — not nothing). For any large model, turn it on.

A subtlety worth knowing: when you don't want to sync every step — for **gradient accumulation**, where you do several forward/backward passes before one optimizer step — wrap the non-syncing steps in `ddp_model.no_sync()`. Inside that context DDP skips the bucket all-reduces entirely, accumulating gradients locally, and you let the all-reduce fire only on the final accumulation step. This turns $K$ accumulation steps' worth of communication into one, which is a large win when the interconnect is the bottleneck.

```python
for micro_step in range(accum_steps):
    is_last = (micro_step == accum_steps - 1)
    ctx = ddp_model.no_sync() if not is_last else contextlib.nullcontext()
    with ctx:
        loss = ddp_model(batch[micro_step]).pow(2).mean()  # toy loss
        (loss / accum_steps).backward()  # syncs only on the last micro-step
optimizer.step()
```

How well does overlap work? On our 7B / 8-GPU example, the all-reduce was 123 ms and the backward pass was, say, ~200 ms. With good bucketing, nearly all of the 123 ms hides under the 200 ms of backward compute, and the *exposed* communication — the final bucket — might be 10–20 ms. So instead of $200 + 123 = 323$ ms per step, you see $\sim 215$ ms, recovering most of the scaling you'd otherwise lose. We'll quantify this in the case studies; the headline is that overlap routinely hides 80–95% of the all-reduce on a well-connected node.

There is a tuning tension in `bucket_cap_mb` worth understanding rather than cargo-culting the default. Bigger buckets mean fewer all-reduce launches, so you amortize the per-message latency $\alpha$ over more bytes — good on slow links (InfiniBand, PCIe) where latency is expensive and you want few, large messages. But bigger buckets also mean you wait *longer* for a bucket to fill before its all-reduce can start, which delays the overlap and leaves a larger final bucket exposed at the end — bad for hiding communication. Smaller buckets start the overlap sooner and shrink the exposed tail, but pay more launch latency. The 25 MB default is a reasonable middle for NVLink; on a slow inter-node fabric you might raise it to 50–100 MB to batch the latency, and on a very fast NVSwitch box you might lower it to overlap more aggressively. The honest answer is to measure: run a few steps at 10, 25, and 50 MB and read the per-step time. There is rarely a dramatic difference once you're in the right order of magnitude, which is itself a useful fact — the default is fine for most jobs, and you should spend your tuning energy on the interconnect and the global batch size, not on shaving the bucket size.

One more failure mode to name, because it bites people who refactor models: DDP's bucketing assumes that *every* parameter that requires a gradient participates in *every* backward pass, so it can know when a bucket is "full." If your model has conditionally-used parameters — a branch that only runs for some inputs, an unused output head — some gradients never become ready, the bucket never completes, and DDP either hangs waiting or (with `find_unused_parameters=True`) does an expensive extra graph traversal each step to detect the stragglers. The clean fix is to make the model's parameter usage static; reach for `find_unused_parameters=True` only when you genuinely can't, and know that it costs you. This is the kind of thing that turns a 96%-efficient job into a 70%-efficient one for a non-obvious reason, and `NCCL_DEBUG` won't show it — it's a DDP-level issue, visible in the DDP logging, not the collective.

## 7. How NCCL picks an algorithm and a protocol for you

You will almost never hand-roll a ring in production. You call `dist.all_reduce` and **NCCL** — the NVIDIA Collective Communications Library — does the work. NCCL is the C++ library PyTorch's NCCL backend links against; it implements every collective we've discussed (broadcast, reduce, all-reduce, all-gather, reduce-scatter, all-to-all), discovers your hardware topology, builds optimized rings and trees across that topology, and at runtime *chooses* an algorithm and a wire protocol per call based on the message size and the machine. Knowing what it chooses, and how to inspect and override it, is the difference between trusting a black box and operating it.

![A decision tree showing how NCCL selects ring or tree by message size and topology and then picks a wire protocol by the latency and bandwidth tradeoff](/imgs/blogs/collective-communication-and-nccl-all-reduce-from-scratch-7.png)

NCCL makes two choices per collective, shown in the figure:

**Choice 1 — the algorithm: Ring vs Tree (vs CollNet/NVLS).** As we derived, ring is bandwidth-optimal for large messages, tree is latency-optimal for small ones. NCCL has, at startup, profiled your topology and built both a ring and a (double-binary) tree across your GPUs. At each call it estimates the cost of each algorithm for *this* message size on *this* topology using an internal $\alpha$-$\beta$ model (tuned per GPU generation), and picks the cheaper one. Large gradient buffers get Ring; tiny tensors get Tree. On NVSwitch systems it may pick **NVLS** (NVLink SHARP), which offloads the reduction arithmetic into the switch hardware itself.

**Choice 2 — the protocol: LL, LL128, or Simple.** Independently of the algorithm, NCCL picks a *wire protocol* that trades latency against bandwidth:

- **LL ("Low Latency")** uses small 8-byte flits with inline flags so a receiver knows data has arrived without a separate synchronization — minimal latency, but it wastes ~half the bandwidth on flags. Used for the smallest messages (below ~256 KB).
- **LL128** is the same idea tuned for NVLink's 128-byte transactions: low latency with only ~5% bandwidth overhead. The sweet spot for mid-size messages on NVLink.
- **Simple** uses full-size transfers with explicit synchronization — highest latency to set up, but full bandwidth. Used for large messages where bandwidth dominates and the setup latency is amortized.

So a single `all_reduce` of a 14 GB gradient buffer typically runs **Ring + Simple** (bandwidth-optimal everywhere), while an `all_reduce` of a 4-byte scalar runs **Tree + LL** (latency-optimal everywhere). NCCL makes this decision for you, per call, and it's usually right.

You inspect what it actually chose with `NCCL_DEBUG`. This is the first thing to set when a distributed job is mysteriously slow:

```bash
# Print NCCL's topology discovery, the rings/trees it built, and per-collective choices.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL,GRAPH   # what to log: init, collectives, the topo graph
torchrun --nproc_per_node=8 train.py 2>&1 | grep -E "NCCL INFO (Ring|Trees|Channel|comm)"
```

In that output you'll see lines describing the rings and trees NCCL built across your GPUs (e.g. how many "channels" — parallel rings — it created, which is how NCCL uses more than one link per GPU), which protocol it selected, and crucially any warnings like falling back to PCIe because it couldn't find NVLink, or routing over a slow network interface. A huge fraction of "why is my 8-GPU job only 4× faster" bugs are visible right here: NCCL silently using the wrong path.

You can also *force* choices for debugging or benchmarking — though you should rarely keep these in production, because the auto-tuner usually beats a hand-pin:

```bash
export NCCL_ALGO=Ring          # force the ring algorithm (or Tree, NVLS)
export NCCL_PROTO=Simple       # force the full-bandwidth protocol
export NCCL_MIN_NCHANNELS=4    # use at least 4 parallel rings (more links per GPU)
export NCCL_IB_HCA=mlx5_0,mlx5_1   # which InfiniBand NICs to use, in order
export NCCL_SOCKET_IFNAME=eth0     # which network interface for the bootstrap/TCP path
export NCCL_P2P_DISABLE=0          # keep GPU-to-GPU direct (P2P) enabled — leave at 0!
```

A few of these are load-bearing in the real world. `NCCL_IB_HCA` tells NCCL which InfiniBand adapters to use — on a multi-NIC node, picking the wrong one (or letting it default to a NIC on the wrong PCIe root complex relative to the GPU) silently halves your inter-node bandwidth. `NCCL_SOCKET_IFNAME` keeps NCCL's bootstrap off a slow management interface. `NCCL_P2P_DISABLE=1` is a thing people set while debugging and then forget, which forces every GPU-to-GPU transfer through host memory and tanks performance — if your intra-node all-reduce is mysteriously 5× too slow, check that it's `0`.

The single most useful tool NCCL ships is **`nccl-tests`**, a standalone benchmark that measures the achieved **bus bandwidth** of each collective without any framework overhead. It is how you find out what your hardware is *actually* capable of, which you then compare against what your training job achieves:

```bash
# Measure all-reduce bus bandwidth from 8 bytes up to 8 GB, on 8 GPUs:
./build/all_reduce_perf -b 8 -e 8G -f 2 -g 8
# Output columns include "busbw" (GB/s) — the number you care about.
# On an 8x A100 NVSwitch node, busbw plateaus around 230-250 GB/s for large sizes.
```

The metric to watch is **bus bandwidth** ("busbw"), which NCCL defines specifically so that it's *independent of $N$* for an ideal all-reduce — it's the achieved bandwidth normalized by the $2(N-1)/N$ factor, so a perfect ring reports a busbw equal to the hardware link bandwidth regardless of GPU count. That normalization is not an accident; it's NCCL encoding our derivation directly into its reporting. If busbw is flat as you scale GPUs, your all-reduce is scaling perfectly. If it sags, you've hit a slow tier in the topology.

## 8. Which parallelism uses which collective: reduce-scatter + all-gather = all-reduce

We opened with a table mapping each collective to a parallelism strategy. Now that we've built the machinery, let's close the loop and make the most important identity in this space concrete: **an all-reduce is exactly a reduce-scatter followed by an all-gather.** This isn't a coincidence; it's the two phases of the ring we derived, and it's the key to understanding why FSDP/ZeRO can be *cheaper* than plain DDP per byte.

![A comparison matrix showing the gradient all-reduce volume for a seven billion parameter model and the per-GPU bytes and time on NVLink versus InfiniBand](/imgs/blogs/collective-communication-and-nccl-all-reduce-from-scratch-8.png)

Walk through which strategy needs which collective:

- **Data parallelism (DDP)** replicates the full model on every GPU and synchronizes gradients with **all-reduce** — $2(N-1)/N \cdot S$ bytes per GPU. Every GPU ends with the full averaged gradient and steps independently. Simple, robust, the default.
- **Fully Sharded Data Parallel (FSDP) / ZeRO-3** shards the parameters, gradients, *and* optimizer state across GPUs so no GPU holds the whole model — this is how you fit a model that doesn't fit on one GPU. The collective pattern is different and cheaper per op: **all-gather** the parameter shards just before each layer's forward (to assemble the full weight), then **reduce-scatter** the gradients after backward (each GPU keeps only its shard's summed gradient). Notice: all-gather is $(N-1)/N \cdot S$ and reduce-scatter is $(N-1)/N \cdot S$, and together they're $2(N-1)/N \cdot S$ — the *same total* as one all-reduce. FSDP just *separates* the two halves so it can shard the model in between, trading more memory savings for the same communication volume (plus the extra forward all-gather). This is the subject of [memory optimization: ZeRO, FSDP, activation checkpointing and offload](/blog/machine-learning/high-performance-computing/memory-optimization-zero-fsdp-activation-checkpointing-and-offload).
- **Tensor parallelism** splits each matmul across GPUs (each holds a slice of the weight matrix) and uses an **all-reduce** to sum the partial outputs back together — twice per Transformer block, on the forward *and* the backward. Because this all-reduce is on the critical path of every layer (you can't overlap it the way DDP overlaps gradient sync), tensor parallelism is only viable over the fastest interconnect, NVLink within a node.
- **Pipeline parallelism** splits the model into stages on different GPUs and uses point-to-point sends between stages — not a collective at all, just `send`/`recv` of activations between adjacent stages.
- **Expert parallelism (MoE)** routes tokens to experts on different GPUs with **all-to-all** — dispatch tokens to their expert's GPU, then all-to-all again to gather the results back. These two all-to-alls are the dominant cost of MoE training. This mapping of strategy-to-collective is exactly what [parallelism strategies: data, tensor, pipeline and expert](/blog/machine-learning/high-performance-computing/parallelism-strategies-data-tensor-pipeline-and-expert) builds on.

Let's make the reduce-scatter-plus-all-gather identity executable, because it's the kind of thing that's much clearer in code than prose:

```python
import torch
import torch.distributed as dist

def all_reduce_via_rs_ag(tensor, world_size):
    """Prove all-reduce == reduce-scatter then all-gather, by hand."""
    # Pad so the tensor splits evenly into world_size shards.
    assert tensor.numel() % world_size == 0
    shard_numel = tensor.numel() // world_size

    # ---- reduce-scatter: each rank ends with the SUM of only its shard ----
    flat = tensor.contiguous().view(-1)
    input_list = list(flat.chunk(world_size))        # N shards of the full buffer
    my_summed_shard = torch.empty(shard_numel, device=tensor.device, dtype=tensor.dtype)
    dist.reduce_scatter(my_summed_shard, input_list, op=dist.ReduceOp.SUM)
    # Now my_summed_shard = sum over ranks of my 1/N slice. Cost: (N-1)/N * S.

    # ---- all-gather: every rank collects all the summed shards ----
    gathered = [torch.empty_like(my_summed_shard) for _ in range(world_size)]
    dist.all_gather(gathered, my_summed_shard)
    # Cost: another (N-1)/N * S. Total: 2(N-1)/N * S -- same as all_reduce.

    result = torch.cat(gathered).view_as(tensor)
    return result

def check():
    rank = dist.get_rank(); world_size = dist.get_world_size()
    x = torch.arange(8 * world_size, dtype=torch.float32, device="cuda") + rank

    ref = x.clone(); dist.all_reduce(ref, op=dist.ReduceOp.SUM)   # the library
    ours = all_reduce_via_rs_ag(x, world_size)                    # rs + ag by hand

    assert torch.allclose(ref, ours), (ref, ours)
    if rank == 0:
        print("reduce_scatter + all_gather == all_reduce  ✓")
```

Run that on any multi-GPU box and `dist.reduce_scatter` + `dist.all_gather` reproduces `dist.all_reduce` exactly, moving the same $2(N-1)/N \cdot S$ total bytes. That equivalence is not a curiosity — it's *why* FSDP's communication volume equals DDP's even though FSDP shards the model: it's literally doing the two halves of the all-reduce with sharding sandwiched between them. Internally, NCCL even implements `all_reduce` as a fused reduce-scatter-then-all-gather over the ring, which is the same two phases we traced by hand in section 3. The model you should hold is: there is one fundamental data movement — sum and redistribute — and the collectives are different *slicings* of it for different parallelism strategies.

## Case studies / real numbers

Enough derivation; let's ground it in measured, named-hardware numbers. The figure for this section quantifies the 7B-model all-reduce across interconnect tiers; here we walk through three real regimes. Where a number is a representative figure rather than an exact measurement from one specific run, I mark it approximate — you should re-measure on your own hardware with `nccl-tests` before trusting any of it for capacity planning.

**Case 1 — NCCL bus bandwidth: NVLink versus InfiniBand.** On a single DGX-class node of 8× A100 80GB SXM connected by NVSwitch, `all_reduce_perf` reports a large-message bus bandwidth in the range of roughly **230–250 GB/s** (approximate; depends on the exact node and NCCL version). This is the number that makes intra-node data parallelism cheap. Now go across nodes: each A100 node typically has InfiniBand NICs delivering on the order of **100–200 Gb/s per direction per NIC**, i.e. roughly **12–25 GB/s per GPU** of effective all-reduce bandwidth across the fabric (approximate, and very topology-dependent). That is a **~10× drop** from NVLink to InfiniBand. The lesson is stark and it drives every multi-node design decision: keep as much communication as possible *inside* the NVLink domain, and minimize what crosses the slow inter-node fabric. NCCL's hierarchical algorithms do this automatically — a ring within each node over NVLink, a tree across nodes over IB — which is why a well-configured 64-GPU job doesn't simply run at the IB bandwidth for everything.

| Tier | Hardware | Effective all-reduce BW per GPU (approx.) | Time to all-reduce 14 GB (7B grads) |
|---|---|---|---|
| NVLink / NVSwitch | 8× A100 80GB SXM, one node | ~230–250 GB/s (busbw) | ~110–123 ms |
| InfiniBand HDR | A100 nodes, ~200 Gb/s NIC | ~12–25 GB/s per GPU | ~0.8–1.4 s |
| PCIe Gen4 (no NVLink) | consumer / mismatched box | ~5–12 GB/s per GPU | ~2–4 s |

The third row is the trap: a box wired with PCIe instead of NVLink (a common consumer or budget-cloud configuration) does the *same* all-reduce 20× slower than the DGX node. The arithmetic is identical; only $B$ changed. If your 8-GPU job is barely faster than 1, run `nccl-tests` first — a low busbw tells you instantly that you're on a slow interconnect path, and no amount of bucketing tuning will fix the wire.

**Case 2 — DDP overlap hiding most of the all-reduce.** Take the 7B model on the 8× A100 node, all-reduce ~123 ms, backward pass ~200 ms (approximate, model-dependent). Without overlap, step communication is fully exposed: $\sim 123$ ms of stall on top of $\sim 400$ ms forward+backward compute, so $\sim 31\%$ of the step is pure communication and your scaling efficiency from 1→8 GPUs is roughly $1/1.31 \approx 76\%$. With DDP bucketing and `gradient_as_bucket_view=True`, nearly all of the 123 ms hides under the 200 ms backward; the *exposed* communication shrinks to the final bucket, maybe 10–20 ms. Now communication is $\sim 4\%$ of the step and scaling efficiency climbs to roughly $1/1.04 \approx 96\%$. That jump from ~76% to ~96% efficiency, on the same hardware, is *entirely* the overlap — no faster wires, just firing the all-reduce early. This is the highest-leverage knob in data-parallel training and it costs you one constructor flag.

| Configuration | Exposed comm / step (approx.) | Step time | Scaling efficiency 1→8 |
|---|---|---|---|
| No overlap (all-reduce after backward) | ~123 ms | ~523 ms | ~76% |
| DDP bucketing, default 25 MB | ~15 ms | ~415 ms | ~96% |
| DDP + `no_sync` over 4 accum steps | ~4 ms amortized | ~404 ms | ~98% |

**Case 3 — the all-reduce-bound regime at scale.** The thing that breaks scaling is not the $2(N-1)/N$ factor — we proved that's nearly constant. It's two other effects that bite as $N$ grows. First, the **bandwidth tier drops**: at 64 GPUs across 8 nodes, the ring must cross InfiniBand, so the effective $B$ falls from ~230 GB/s (NVLink) toward the IB tier, and the all-reduce time for our 14 GB buffer can rise from ~123 ms toward the high-hundreds-of-ms or worse if not hierarchically routed. Second, the **exposed-comm-to-compute ratio** worsens because as you scale data-parallel width, you often *shrink the per-GPU batch*, which shrinks per-GPU compute (less backward to hide behind) while the all-reduce volume stays fixed at $S$. Eventually communication can no longer hide under compute, and you become **all-reduce-bound**: adding GPUs stops helping. This is the wall that pushes large-scale training toward FSDP (shard the model so each op moves less), toward larger global batch sizes (more compute per all-reduce), and toward 3D parallelism that keeps the expensive collectives inside the NVLink domain. The serving-side analogue — where the collective is the bottleneck on the inference path instead — is covered in [serving LLMs at scale: production systems](/blog/machine-learning/large-language-model/serving-llms-at-scale-production-systems).

The historical thread is worth a sentence, because the people who figured this out left a paper trail you should read. The ring-reduce algorithm has roots in the MPI/HPC community (Rabenseifner, Thakur, and others worked out the optimal-bandwidth collectives in the early 2000s). Baidu's 2017 "Bringing HPC Techniques to Deep Learning" post and code brought the bandwidth-optimal ring all-reduce to deep learning specifically, showing near-linear scaling. **Horovod** (Sergeev and Del Balso, 2018, from Uber) packaged ring all-reduce into an easy-to-use library on top of TensorFlow/PyTorch and popularized it across the industry; PyTorch DDP and NCCL then absorbed the same ideas into the core stack. When you call `dist.all_reduce` today, you're standing on that lineage.

## When to reach for this (and when not to)

Every technique here is a cost as well as a benefit. Be decisive about when each pays.

**Use plain DDP with bucketing when the model fits on one GPU.** This is the default and you should resist complicating it. If your model fits in GPU memory and DDP's overlapped all-reduce saturates NVLink, you are done — adding FSDP, tensor parallelism, or pipeline parallelism buys nothing but complexity and often *loses* performance to extra collectives. Turn on `gradient_as_bucket_view=True`, tune `bucket_cap_mb` to your interconnect (bigger buckets amortize latency on slow links; smaller buckets overlap better on fast ones), and measure.

**Reach for FSDP / ZeRO only when the model (plus optimizer state) doesn't fit.** Sharding trades communication and complexity for memory. If you don't *need* the memory, you're paying the forward all-gather for nothing. The crossover is roughly when params + grads + Adam optimizer state exceed your GPU's memory — for fp16/bf16 mixed-precision training with Adam, that's about $16$–$18$ bytes per parameter, so a 7B model needs ~120 GB of optimizer-related state and won't fit on an 80 GB GPU, making FSDP/ZeRO genuinely necessary.

**Don't pin `NCCL_ALGO` or `NCCL_PROTO` in production** unless you've benchmarked and the auto-tuner is demonstrably wrong. NCCL's per-message selection beats a global pin in almost every mixed workload. Pin them only to *isolate* a problem while debugging, then remove the pin.

**Use a tree (let NCCL choose it) for small, frequent, latency-bound collectives** — metric reductions, small control tensors, and the early small buckets at very large $N$. Don't force a ring on a 4-byte scalar across 1,024 GPUs; that's 2,046 pointless hops.

**Don't scale data-parallel width past the point where the per-GPU batch starves compute.** If shrinking the local batch to add GPUs makes the all-reduce un-hideable, you've crossed into the all-reduce-bound regime; the fix is a bigger global batch, gradient accumulation with `no_sync`, or switching to a parallelism strategy that keeps collectives off the critical inter-node path — not more data-parallel GPUs.

**Always check the interconnect first when scaling is bad.** Before tuning any algorithm, run `nccl-tests` and read `NCCL_DEBUG=INFO`. A huge fraction of "poor scaling" is simply the wrong wire — PCIe instead of NVLink, the wrong IB NIC, P2P disabled, a slow bootstrap interface. No software knob fixes a slow physical path; you have to see it first.

## Key takeaways

- **There are six collectives** — broadcast, reduce, all-reduce, all-gather, reduce-scatter, all-to-all — and each maps to a parallelism strategy. In standard data-parallel training, the only one on the hot path is **all-reduce**, run once per step on the full gradient buffer.
- **Ring all-reduce moves exactly $2(N-1)/N \cdot S$ bytes per GPU**, which you derive as two phases (reduce-scatter + all-gather) of $N-1$ steps each, moving $S/N$ per step. This is provably bandwidth-optimal.
- **That cost is nearly independent of $N$**: the factor $2(N-1)/N$ rises from 1.0 at $N=2$ to 1.97 at $N=64$ and caps at 2. Per-GPU communication barely grows as you add GPUs — the foundation of scalable data parallelism.
- **Ring is bandwidth-optimal; tree is latency-optimal.** Ring takes $2(N-1)$ hops (bad for small messages at large $N$); tree takes $\sim 2\log_2 N$ hops (bad for bandwidth). Choose by message size; let NCCL choose for you.
- **NCCL makes two choices per call**: algorithm (Ring / Tree / NVLS) by message size and topology, and protocol (LL / LL128 / Simple) by the latency-bandwidth tradeoff. Inspect with `NCCL_DEBUG=INFO`; benchmark with `nccl-tests` and watch **busbw**.
- **DDP bucketing overlaps the all-reduce with the backward pass**, hiding 80–95% of the communication and turning ~76% scaling efficiency into ~96%. Set `gradient_as_bucket_view=True` and use `no_sync()` for gradient accumulation.
- **An all-reduce is a reduce-scatter then an all-gather** — the same two phases as the ring, and exactly why FSDP's per-step communication equals DDP's while sharding the model in between.
- **The interconnect sets $B$**: NVLink (~230 GB/s busbw) is ~10× InfiniBand and ~20× PCIe for the same all-reduce. When scaling is bad, check the wire before the algorithm.

## Further reading

- **NCCL documentation** — the algorithms (ring, tree, CollNet/NVLS), the protocols (LL, LL128, Simple), the environment variables (`NCCL_ALGO`, `NCCL_PROTO`, `NCCL_IB_HCA`, `NCCL_DEBUG`), and the bus-bandwidth definition. The `nccl-tests` repository for measuring busbw on your own hardware.
- **Baidu, "Bringing HPC Techniques to Deep Learning" (2017)** — the bandwidth-optimal ring all-reduce explained and benchmarked for deep learning, with the $2(N-1)/N$ derivation.
- **Sergeev and Del Balso, "Horovod: fast and easy distributed deep learning in TensorFlow" (2018)** — the library that popularized ring all-reduce for the industry.
- **PyTorch DDP documentation and the DDP design note** — gradient bucketing, autograd hooks, `gradient_as_bucket_view`, `no_sync`, and the overlap mechanism. The `torch.distributed` collectives reference.
- **Rabenseifner / Thakur et al. on optimization of collective communication operations in MPI** — the HPC lineage of the bandwidth-optimal algorithms NCCL implements.
- Within this series: [why HPC is the bottleneck for modern AI](/blog/machine-learning/high-performance-computing/why-hpc-is-the-bottleneck-for-modern-ai) (the three-walls frame), [parallelism strategies: data, tensor, pipeline and expert](/blog/machine-learning/high-performance-computing/parallelism-strategies-data-tensor-pipeline-and-expert), [interconnects: NVLink, NVSwitch, InfiniBand and RDMA](/blog/machine-learning/high-performance-computing/interconnects-nvlink-nvswitch-infiniband-and-rdma), [memory optimization: ZeRO, FSDP, activation checkpointing and offload](/blog/machine-learning/high-performance-computing/memory-optimization-zero-fsdp-activation-checkpointing-and-offload), and the capstone [the HPC playbook for AI engineers](/blog/machine-learning/high-performance-computing/the-hpc-playbook-for-ai-engineers).

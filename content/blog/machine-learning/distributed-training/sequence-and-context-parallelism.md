---
title: "Sequence and Context Parallelism: Training on Sequences That Don't Fit"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "Split the sequence, not just the batch: how Megatron sequence parallelism, ring attention, and DeepSpeed-Ulysses train 128K-to-1M-token context without the activation memory blowing up a single GPU."
tags:
  [
    "distributed-training",
    "multi-gpu",
    "sequence-parallelism",
    "context-parallelism",
    "ring-attention",
    "long-context",
    "megatron",
    "pytorch",
    "flash-attention",
    "ml-systems",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Distributed Training"
author: "Hiep Tran"
featured: true
readTime: 35
---

You have a 7-billion-parameter transformer that trains fine at 8K context on eight A100 80GB cards. Data parallelism, gradient all-reduce, activation checkpointing — the whole rig hums along at a respectable MFU. Then a product requirement lands: the model needs to read entire codebases, entire legal filings, entire books. You need 256K tokens of context. You bump the sequence length, keep the batch at one sequence per GPU, launch the same job, and it dies on the first forward pass. Out of memory. You cut the batch to nothing — it is already one. You turn on full activation checkpointing — still OOM. You shard the weights harder with FSDP, dropping the resident weight footprint to a couple of gigabytes per card — and it *still* OOMs, now with 70 GB of the card sitting empty of weights and full of something else.

That something else is *activations*, and this is the moment every long-context project hits a wall the classic parallelism strategies were never built to climb. Data parallelism splits the batch. Tensor parallelism splits the hidden dimension. Pipeline parallelism splits the layers. Not one of them splits the *sequence*. So when the sequence is the thing that has grown — when a single sequence's activations no longer fit on a single GPU even at batch size one — none of the three levers you already know how to pull will help you. The memory you need to shed is spread along a dimension none of them touch.

![A before and after comparison of activation memory for a seven billion parameter model at long context, showing one GPU overflowing its eighty gigabytes while an eight way sequence split fits comfortably](/imgs/blogs/sequence-and-context-parallelism-1.webp)

This post is about the parallelism that finally splits the sequence. We will build up three tools, each solving a different slice of the problem: **Megatron sequence parallelism**, which shards the LayerNorm and dropout regions that tensor parallelism leaves replicated; **ring attention**, the headline technique that streams key/value blocks around a ring of GPUs so no single card ever holds the full attention matrix; and **DeepSpeed-Ulysses**, the all-to-all alternative that trades a different communication pattern for the same memory win. By the end you will be able to look at a long-context OOM, know *exactly* which dimension of memory is overflowing, pick the right sequence or context parallel strategy, derive its memory and communication cost before you launch, and write the device mesh that composes it with the tensor, pipeline, and data parallelism you already run. This is the fifth of the [four walls](/blog/machine-learning/distributed-training/why-distributed-training) — the model that won't fit — but hit from the one direction the [map of parallelism](/blog/machine-learning/distributed-training/the-map-of-parallelism) has so far only sketched.

## Why the sequence is the dimension nobody splits

Start with the physics of the failure, because until you can see *why* the memory grows the way it does, every fix looks like guesswork. A transformer's memory during training breaks into two piles. The first pile is the model state: the parameters, their gradients, and the optimizer's moments. For a 7B model in bf16 that is 14 GB of parameters, 14 GB of gradients, and — with mixed-precision Adam keeping fp32 master weights plus fp32 first and second moments, 12 bytes per parameter — 84 GB of optimizer state. Call it 112 GB of model state. That already overflows a single 80 GB card, which is why you shard it. FSDP or ZeRO splits those 112 GB across your eight ranks, and now each card holds 14 GB of model state. Solved, or so it seems.

The second pile is *activations*: every intermediate tensor the forward pass produces and the backward pass needs to compute gradients. And here is the thing FSDP never touches — **activation memory scales with the sequence length, and data parallelism replicates it in full on every rank.** Each data-parallel GPU processes its own sequences end to end, so each one must hold the entire activation working set for those sequences. Sharding the *weights* did nothing for the activations. The 70 GB of empty card you saw in the intro was empty of weights and drowning in activations.

![A layered memory budget stack for a seven billion parameter model showing sharded parameters gradients and optimizer states as small fixed bars while the activation bar grows with sequence length until it overflows the card](/imgs/blogs/sequence-and-context-parallelism-2.webp)

How fast do activations grow? There are two regimes, and the distinction matters enormously. Naive attention materializes the full score matrix — an $S \times S$ tensor per attention head. That is $O(S^2)$ memory. At 256K tokens, one head's score matrix is $262144^2$ entries; multiply by the head count and by two bytes and you are looking at *terabytes* per layer. This is why nobody trains long context with naive attention. FlashAttention (Dao et al., 2022) is not optional here — it computes attention block by block with an online softmax and never materializes the $S \times S$ matrix, collapsing attention's memory from $O(S^2)$ to $O(S)$. Everything in this post assumes FlashAttention is on. It is the precondition that makes long-context training conceivable at all, and — as we will see — ring attention is essentially FlashAttention's online softmax stretched across GPUs.

With FlashAttention on, the surviving activation memory is the *linear* term: the per-layer inputs and intermediate tensors you must keep for the backward pass, which scale as $O(S)$ per layer. Concretely, the activation working set of one transformer layer without the score matrix is roughly $34\, s\, b\, h$ bytes in bf16, where $s$ is sequence length, $b$ is the local batch, and $h$ is the hidden size. For our 7B model — hidden 4096, 32 layers — a single layer at 256K tokens and batch one is about 36 GB of activations. Thirty-two of them without checkpointing would be over a terabyte. Full activation checkpointing rescues most of it: you keep only each layer's *input* (about 2 GB per layer at 256K) and recompute the interior during the backward pass, so peak memory becomes the sum of all the saved inputs (about 64 GB across 32 layers) plus the working set of the single layer you are currently recomputing (about 36 GB). That is roughly 100 GB — on an 80 GB card. OOM, exactly as observed, and no amount of weight sharding closes the gap.

Here is the whole table, so you can see where the wall is for our running example — a 7B model, batch one, FlashAttention plus full activation checkpointing, on 80 GB cards. These numbers are order-of-magnitude estimates from the linear activation model above; treat them as "which side of the wall," not accounting-grade figures.

| Context length | Activation memory, 1 GPU | Fits 80 GB? | 8-way sequence split, per GPU | Fits? |
|---|---|---|---|---|
| 8K | ~3 GB | Yes | ~0.4 GB | Yes (split unnecessary) |
| 32K | ~11 GB | Yes | ~1.4 GB | Yes |
| 128K | ~50 GB | Tight | ~6 GB | Yes |
| 256K | ~100 GB | **No — OOM** | ~13 GB | Yes |
| 1M | ~390 GB | No | ~49 GB | Yes |

Read the last two rows carefully, because they contain the entire thesis of this post. At 256K the activation memory overflows one card no matter how well you shard the weights, and the *only* term that changed between the row that fits and the row that OOMs is the sequence length. If you could split the activation memory along the sequence — give each of eight GPUs one-eighth of the sequence and therefore one-eighth of the activations — the 256K case drops to 13 GB per GPU and fits with room to spare. That splitting-the-sequence operation is what sequence and context parallelism *are*. The rest of this post is how to do it correctly, in three flavors, with the communication cost each one pays.

Notice one more thing: at 8K and 32K, the sequence split is *unnecessary* — activations already fit, and adding cross-GPU sequence communication would only slow you down. This is the recurring discipline of the whole [distributed training series](/blog/machine-learning/distributed-training/why-distributed-training): every parallelism dimension is a cost, and you add it only when a wall forces you to. Long-context sequence parallelism is the sharpest example. Below roughly 32K tokens you almost never want it; past 128K you almost always need it.

## What data, tensor, and pipeline parallelism do not do

Before we split the sequence, let's be precise about why the parallelism you already run does not. It is tempting to assume that *some* combination of data, tensor, and pipeline parallelism must eventually shard the activations along the sequence — but none of them do, and understanding why is what lets you diagnose a long-context OOM in ten seconds instead of an afternoon.

![A comparison matrix of five parallelism strategies scored by what dimension each one splits whether activation memory shrinks with sequence length and what communication each one adds](/imgs/blogs/sequence-and-context-parallelism-3.webp)

**Data parallelism** splits the *batch*. Each rank gets different sequences, runs a full forward and backward on them, and the ranks all-reduce their gradients. The sequence dimension is entirely intact on every rank — in fact data parallelism *replicates* the activation cost, because every rank pays the full per-sequence activation memory for its own batch. Doubling your data-parallel degree does nothing for a single sequence that doesn't fit; it just lets you process more of the too-big sequences in parallel, each still overflowing its card.

**Tensor parallelism** ([Megatron-style](/blog/machine-learning/distributed-training/tensor-parallelism-megatron)) splits the *hidden dimension and the attention heads*. A column-parallel linear shards the output features across ranks; a row-parallel linear shards the input features and all-reduces the result. This genuinely reduces some activation memory — the attention and MLP activations are sharded across the tensor-parallel group by a factor of the TP degree $t$. But two problems remain. First, tensor parallelism keeps the full sequence length on every rank; it shards *within* each token's representation, not across tokens. Second — and this is the gap Megatron sequence parallelism was invented to close — the regions *between* the tensor-parallel blocks, the LayerNorm and dropout operations, are *not* sharded by tensor parallelism. They operate on the full, un-sharded activation, replicated on every rank. At long context those replicated LayerNorm-region activations are a real fraction of the memory, and every rank pays them in full.

**Pipeline parallelism** splits the *layers* across stages. Each stage holds a contiguous block of transformer layers and passes activations to the next stage. This reduces the *number of layers* whose activations any one GPU stores, but each stage still processes the full sequence for its layers — the per-layer activation memory, which is the $O(S)$ term that grows with context, is untouched. A pipeline stage holding four layers of a 256K-context batch still needs the full-sequence activations for those four layers.

So the scorecard is stark. Of the three classic parallelism dimensions, one replicates the per-sequence activation memory (data), one shards it partially but leaves the LayerNorm regions and the full sequence length intact (tensor), and one reduces the layer count but not the per-layer sequence cost (pipeline). The activation memory that grows with $S$ — the memory that OOMs your long-context run — is left standing by all three. You need a fourth idea: split along $S$ itself. That splits into two closely related techniques. **Sequence parallelism** in the Megatron sense shards the *LayerNorm/dropout* activations along the sequence as a companion to tensor parallelism. **Context parallelism** — ring attention and Ulysses — shards the *attention computation itself* along the sequence, which is the harder and more powerful move because attention is the one operation where every token must interact with every other token. We take them in that order.

## Megatron sequence parallelism: sharding the regions tensor parallelism forgot

The cheapest sequence-splitting win costs you *zero extra communication*, and almost every tensor-parallel job should already be using it. This is Megatron sequence parallelism (Korthikanti et al., 2022, "Reducing Activation Recomputation in Large Transformer Models"), and the idea is a small, exact accounting trick on top of tensor parallelism.

Recall the tensor-parallel transformer block. The attention and MLP sub-layers are sharded across the tensor-parallel group of degree $t$: each rank holds a shard of the heads (attention) or the hidden dimension (MLP), computes on it, and a **all-reduce** combines the partial results at the end of each sub-layer. In between the sharded sub-layers sit LayerNorm and dropout. Tensor parallelism cannot shard these along the hidden dimension — LayerNorm needs the full hidden vector of each token to compute its statistics — so it leaves them *replicated*: every rank in the tensor-parallel group holds the full, un-sharded activation for the LayerNorm and dropout regions. That replicated activation is pure waste; it is the same tensor on all $t$ ranks.

Sequence parallelism's insight: the LayerNorm and dropout operations are *independent across tokens*. LayerNorm normalizes each token's hidden vector on its own; dropout masks each element independently. So while you cannot shard these regions along the hidden dimension, you *can* shard them along the **sequence** dimension — give each of the $t$ ranks a different one-$t$-th slice of the tokens. Each rank does LayerNorm and dropout on its slice, holding only $s/t$ tokens' worth of that activation instead of the full $s$. The replicated waste is gone, cut by a factor of $t$.

![A dataflow graph showing a sequence sharded LayerNorm region feeding an all gather into a tensor parallel attention and MLP region that fans across two GPU shards before a reduce scatter returns to the sequence sharded region](/imgs/blogs/sequence-and-context-parallelism-4.webp)

The elegant part is the communication accounting. To enter a tensor-parallel region you need the full sequence (each rank's matmul needs all tokens), but you only have your $s/t$ slice — so you **all-gather** along the sequence to reconstruct the full-sequence activation, then run the sharded attention or MLP. To leave the region and return to sequence-sharded LayerNorm, you must both *sum* the tensor-parallel partial results *and* *scatter* them back into sequence slices — which is exactly a **reduce-scatter**. Now here is the trick: without sequence parallelism, the boundary operation was a single all-reduce. And an all-reduce is, internally, *precisely a reduce-scatter followed by an all-gather* — that is how ring all-reduce is implemented, as we derived in [collectives from scratch](/blog/machine-learning/distributed-training/collectives-from-scratch). So sequence parallelism replaces one all-reduce with one all-gather (entering) plus one reduce-scatter (leaving), and the total bytes moved are *identical*. You get the activation-memory reduction for free.

Let me make the byte count explicit, because "free" is a strong claim. The all-reduce in plain tensor parallelism moves $\tfrac{2(t-1)}{t} \cdot M$ bytes per rank for a message of size $M$ (the ring all-reduce volume). Sequence parallelism's all-gather moves $\tfrac{t-1}{t} \cdot M$ bytes per rank, and its reduce-scatter moves another $\tfrac{t-1}{t} \cdot M$ bytes per rank. Add them: $\tfrac{2(t-1)}{t} \cdot M$ — the same number. No extra communication volume, and each rank's LayerNorm-region activation drops from $M$ to $M/t$. This is why sequence parallelism is essentially always the right default when you already run tensor parallelism: it is a strict improvement.

In Megatron-LM you turn it on with a single flag, but note the constraint — it *requires* tensor parallelism, because it piggybacks on the tensor-parallel group's collectives.

```bash
# Megatron-LM: sequence parallelism piggybacks on tensor parallelism.
# --sequence-parallel is only valid with --tensor-model-parallel-size > 1.
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --tensor-model-parallel-size 8 \
    --sequence-parallel \
    --use-flash-attn \
    --seq-length 32768 \
    --num-layers 32 --hidden-size 4096 --num-attention-heads 32 \
    --micro-batch-size 1 \
    --recompute-activations --recompute-granularity selective
```

Sequence parallelism gets you a factor-of-$t$ reduction on the LayerNorm-region activations. That is a meaningful win — Korthikanti et al. report it removes the need for a chunk of activation recomputation — but it does *not* shard the attention computation itself, and it is bounded by the tensor-parallel degree $t$, which you rarely push past 8 (beyond a single NVLink domain, tensor parallelism's all-reduce becomes a bottleneck). For 32K context on a well-tuned tensor-parallel job, sequence parallelism plus activation checkpointing may be all you need. For 256K and beyond, you need to shard the attention itself along the sequence — and that is ring attention.

## Ring attention: streaming the sequence around a ring

Here is the hard problem sequence parallelism does not solve. Attention is the one operation in the transformer where *every token must interact with every other token*. Query token $i$ attends to key/value tokens $1$ through $S$. If you shard the queries across GPUs — GPU 0 gets tokens 1 to $S/N$, GPU 1 gets the next block, and so on — then each GPU has its own query block but needs *all* the key/value blocks, which live on the other GPUs. You cannot compute attention on a slice of the sequence in isolation; a naive sharding would require every GPU to hold the full key/value tensor, which defeats the purpose. Attention's all-to-all coupling is exactly what makes splitting the sequence hard, and exactly what ring attention (Liu, Zaharia, Abbeel, 2023, "Ring Attention with Blockwise Transformers") solves.

The idea rides on FlashAttention's online softmax. Recall how FlashAttention computes exact attention without ever materializing the full score matrix: it walks the key/value tensor in blocks, and for each block it computes the partial attention scores against the queries, then updates a *running* softmax — a running maximum $m$, a running normalizer $\ell$, and a running weighted output accumulator $o$ — using the numerically stable online-softmax recurrence. After the last key/value block, the accumulator holds the exact attention output. The full $S \times S$ matrix is never in memory; only one block of scores exists at a time. Attention becomes a *loop over key/value blocks* with a running accumulator, and the memory is $O(\text{block size})$, not $O(S)$.

Ring attention takes that loop and distributes it across GPUs. Split the sequence into $N$ blocks, one per GPU. GPU $i$ permanently holds its query block $Q_i$ and initially its own key/value block $(K_i, V_i)$. Now run the FlashAttention loop, but instead of each GPU walking through key/value blocks it holds in its own memory, the key/value blocks **rotate around a ring**. At each ring step, every GPU computes the partial attention of its fixed query block against the key/value block it currently holds, updates its running softmax accumulator, and then *sends* that key/value block to its neighbor while *receiving* the next block from its other neighbor. After $N$ steps, every query block has seen every key/value block — exact, full attention — and no GPU ever held more than its own query block plus one key/value block at a time.

![A grid showing three GPUs each holding a fixed query block while the key value block they compute against rotates one hop per ring step so that after three steps every query has attended to every key value block](/imgs/blogs/sequence-and-context-parallelism-5.webp)

Trace the memory. GPU $i$ holds: its query block $Q_i$ of size $O(S/N \cdot d)$, one key/value block of $O(S/N \cdot d)$, the output accumulator of $O(S/N \cdot d)$, and the running softmax stats of $O(S/N)$. Every term is $O(S/N)$. **The per-GPU attention memory drops from $O(S)$ to $O(S/N)$** — split the sequence across $N$ GPUs and each one holds one-$N$-th of it. That is precisely the last row of the memory table from earlier: 256K context that OOMs one card fits in 13 GB across eight. And crucially, the full $S \times S$ score matrix is never materialized *anywhere* — not on one GPU, not summed across the ring. It exists only as a sequence of $S/N \times S/N$ blocks, one per ring step.

Now the communication, and why it hides. Each ring step, every GPU sends one key/value block: $2 \cdot (S/N) \cdot d$ elements (the key and the value), which at two bytes each is a fixed message size. Over $N$ steps the total per-GPU communication is $2 \cdot S \cdot d$ elements — each GPU eventually passes the equivalent of the whole key/value tensor through it. That is $O(S)$ communication per GPU, which sounds expensive. But it *overlaps completely with compute*, and here is the derivation that shows why.

The compute per ring step is the block attention: query block ($S/N \times d$) times key block transposed produces an $S/N \times S/N$ score block, and that times the value block. That is roughly $4 (S/N)^2 d$ floating-point operations — it grows *quadratically* in the block size $S/N$. The communication per step is $2 \cdot (S/N) \cdot d \cdot 2$ bytes — it grows only *linearly* in the block size. Take the ratio of compute time to communication time:

$$\frac{T_\text{compute}}{T_\text{comms}} = \frac{4 (S/N)^2 d \,/\, F}{4 (S/N) d \,/\, L} = \frac{(S/N)\, L}{F}$$

where $F$ is the GPU's achievable FLOP/s and $L$ is the ring link bandwidth in bytes/s. The block size $S/N$ is in the *numerator*. As long as each GPU's block is big enough — as long as $S/N \geq F/L$ — the compute takes longer than the communication and the ring send is fully hidden behind the block attention. Longer context or fewer GPUs makes the block *bigger*, which makes overlap *easier*. This is the beautiful structural property of ring attention: the regime where you most need it (huge $S$) is exactly the regime where its communication is most trivially hidden. Contrast that with tensor parallelism, whose all-reduce becomes *harder* to hide as you scale. Ring attention is the rare parallelism dimension that gets *more* efficient as the problem gets bigger.

Here is the block loop, sketched as runnable PyTorch. This is a simplified single-head illustration — production implementations (`ring-flash-attention`, Megatron-LM's context parallelism, PyTorch's context-parallel API) fuse the block attention into a FlashAttention kernel and handle causal masking, multiple heads, and grouped-query attention — but the ring structure and the online-softmax accumulation are exactly this.

```python
import torch
import torch.distributed as dist

def ring_attention(q, k, v, cp_group):
    """Exact attention over a sequence sharded across the cp_group ring.
    q, k, v: this rank's local blocks, shape [block, d]. Returns local output."""
    world = dist.get_world_size(cp_group)
    rank = dist.get_rank(cp_group)
    d = q.shape[-1]
    scale = d ** -0.5

    # Running online-softmax state (FlashAttention-style), per query row.
    o = torch.zeros_like(q)                       # output accumulator
    m = torch.full((q.shape[0], 1), float("-inf"), device=q.device)   # running max
    l = torch.zeros((q.shape[0], 1), device=q.device)                 # running sum

    k_blk, v_blk = k, v                           # start with our own K/V block
    next_rank = (rank + 1) % world
    prev_rank = (rank - 1) % world

    for step in range(world):
        # --- overlap: kick off the ring send/recv of the NEXT K/V block ---
        if step < world - 1:
            k_recv, v_recv = torch.empty_like(k_blk), torch.empty_like(v_blk)
            reqs = [dist.P2POp(dist.isend, k_blk, next_rank, cp_group),
                    dist.P2POp(dist.irecv, k_recv, prev_rank, cp_group),
                    dist.P2POp(dist.isend, v_blk, next_rank, cp_group),
                    dist.P2POp(dist.irecv, v_recv, prev_rank, cp_group)]
            handles = dist.batch_isend_irecv(reqs)

        # --- compute: block attention on the CURRENT K/V block (hides comms) ---
        scores = (q @ k_blk.transpose(-1, -2)) * scale        # [block, block]
        blk_max = scores.max(dim=-1, keepdim=True).values
        new_m = torch.maximum(m, blk_max)
        p = torch.exp(scores - new_m)                          # unnormalized weights
        corr = torch.exp(m - new_m)                            # rescale old accumulator
        l = l * corr + p.sum(dim=-1, keepdim=True)
        o = o * corr + p @ v_blk
        m = new_m

        # --- wait for the ring transfer, swap in the next block ---
        if step < world - 1:
            for h in handles:
                h.wait()
            k_blk, v_blk = k_recv, v_recv

    return o / l          # finalize: normalize the accumulator
```

Two engineering details make or break a real implementation. First, **overlap must be explicit**: notice the send/recv is launched *before* the block attention runs and waited on *after*. If you compute first and communicate second, you serialize the ring and lose the whole benefit. Second, **causal masking creates load imbalance**. In a causal (decoder) model, query token $i$ attends only to key tokens $\leq i$, so query block 0 attends to nothing beyond its own block, while the last query block attends to everything. If you assign contiguous blocks, GPU 0 does almost no work and the last GPU does full work — a straggler by construction. The fix is **striped or zigzag assignment**: give each GPU a strided set of tokens (GPU $i$ gets tokens $i, i+N, i+2N, \dots$) so every GPU's block spans the whole sequence range and does equal causal work. This is the difference between 50% efficiency and 95% efficiency on a causal ring, and it is why "ring attention" in practice means "striped/zigzag ring attention."

#### Worked example: activation memory, one GPU versus eight-way context parallel

Take the running 7B model — hidden 4096, 32 layers, 32 heads — at 256K context, batch one, on H100 80GB SXM cards, FlashAttention plus full activation checkpointing. On one GPU, the activation memory is the sum of saved layer inputs plus one layer's recompute working set: roughly $32 \times 2\,\text{GB} = 64\,\text{GB}$ of saved inputs plus about 36 GB for the single layer being recomputed, totaling about 100 GB. The 80 GB card OOMs. This is not a tuning problem — there is no batch size, no checkpointing schedule, no weight-sharding trick that fits 100 GB of activations into 80 GB, because none of those levers touch the sequence dimension.

Now split the sequence eight ways with context parallelism. Each GPU holds one-eighth of the sequence: 32K tokens' worth of queries and, at any ring step, one 32K key/value block. Every activation term divides by eight. Saved inputs drop to 8 GB, the recompute working set to 4.5 GB, total about 13 GB per GPU. The 256K run that was physically impossible on one card now fits with 67 GB to spare — room you would spend on a larger micro-batch or a bigger model. The math on the card is the same math; you have simply spread the sequence, and therefore its activations, across the ring. Push to 1M context and one GPU needs about 390 GB (hopeless); eight-way context parallel needs 49 GB per GPU (comfortable), and sixteen-way needs 25 GB. The sequence length you can train scales *linearly* with the context-parallel degree.

#### Worked example: ring communication hides under block attention

Does the ring communication actually disappear behind compute, or is that a hopeful approximation? Put numbers on it. Same 7B model, 256K context, eight-way ring on an H100 node (NVLink). Hidden dimension $d = 4096$, block size $S/N = 32768$.

The compute per ring step is the block attention: about $4 \times (32768)^2 \times 4096 \approx 1.76 \times 10^{13}$ FLOPs. At an achievable 600 TFLOP/s in bf16 (roughly 60% of the H100's 989 TFLOP/s peak, a realistic FlashAttention number), that is about 29 ms of compute per step.

The communication per ring step is one key block plus one value block: $2 \times 32768 \times 4096 \times 2\,\text{bytes} \approx 537\,\text{MB}$. Over an intra-node NVLink ring at an effective 400 GB/s per hop, that transfers in about 1.3 ms.

Compute is 29 ms; communication is 1.3 ms. The ring send is under 5% of the block-attention time — *completely hidden*, with the GPU compute-bound the entire ring. The ratio $(S/N) L / F$ predicted this: with a 32K block, NVLink bandwidth, and H100 FLOP/s, the ratio comes out around 20, meaning compute dominates communication by roughly twenty to one. Now stress it: drop to PCIe instead of NVLink (say 25 GB/s effective) and the communication becomes about 21 ms — now roughly comparable to the 29 ms compute, and overlap gets tight; you are no longer comfortably hidden. Or shrink the context to 32K with the same eight-way ring, so the block is only 4K tokens: compute per step falls quadratically to about 0.45 ms while communication falls only linearly to about 0.17 ms, and the ratio collapses toward one — which is precisely why you *don't* use ring attention at short context. The overlap that makes ring attention free at 256K evaporates at 32K. The tool matches the regime.

## DeepSpeed-Ulysses: the all-to-all alternative

Ring attention is not the only way to split the sequence. DeepSpeed-Ulysses (Jacobs et al., 2023) reaches the same $O(S/N)$ memory with a completely different communication pattern — and the two make a clean trade-off you should understand before choosing.

Ulysses starts, like ring attention, with the sequence sharded across $N$ GPUs: each GPU holds $S/N$ tokens, but the *full* hidden dimension for those tokens. The problem is attention needs each query to see all keys. Ulysses's move is a **transpose via all-to-all**. Right before the attention operation, it performs an all-to-all that redistributes the data so that each GPU now holds *all* $S$ tokens but only a *subset of the attention heads* — head-sharded instead of sequence-sharded. With all tokens present locally (for its head subset), each GPU computes standard, unmodified, full attention on its heads — no ring, no online-softmax choreography, just a normal FlashAttention call over the complete sequence for those heads. Then a second all-to-all transposes back to sequence-sharded for the rest of the layer. Two all-to-alls per attention, four counting the backward pass.

![A comparison matrix of ring attention against DeepSpeed Ulysses scored on communication pattern communication volume per GPU parallel degree cap and per GPU memory](/imgs/blogs/sequence-and-context-parallelism-6.webp)

The trade-off has three axes. **Communication volume**: Ulysses's all-to-all moves the activation tensor, whose per-GPU volume is roughly $O(S \cdot d / N)$ per all-to-all — and all-to-all is bandwidth-efficient, so in aggregate Ulysses often moves *less* data than ring attention's $O(S \cdot d)$ per-GPU total. **Overlap**: ring attention's communication overlaps with compute step by step and hides naturally at long context; Ulysses's all-to-all is a hard synchronization barrier before and after attention that is harder to overlap, though it is a single efficient collective rather than $N$ sequential sends. **The degree cap**: this is the decisive constraint. Because Ulysses shards the *heads* across GPUs during attention, its parallel degree cannot exceed the number of attention heads. A model with 32 heads caps Ulysses at 32-way — and with grouped-query attention, where the key/value heads are far fewer (say 8), the effective cap is tighter still. Ring attention has *no such cap*: it shards along the sequence, which you can split into arbitrarily many blocks, so ring scales to hundreds or thousands of GPUs for extreme context. This is why the very-long-context frontier (512K, 1M, and beyond) runs on ring attention or ring/Ulysses hybrids, while Ulysses shines in the moderate regime — 64K to 256K on a node or two — where its degree fits under the head count and its all-to-all is efficient.

Here is the compact decision. Full comparison in the table.

| Dimension | Ring attention | DeepSpeed-Ulysses |
|---|---|---|
| Communication pattern | Ring point-to-point send/recv of K/V blocks | Two all-to-all transposes per attention |
| Per-GPU comms volume | $O(S \cdot d)$ total, spread over $N$ steps | $O(S \cdot d / N)$ per all-to-all |
| Overlaps with compute? | Yes — step-by-step, hides at long $S$ | Harder — collective is a sync barrier |
| Parallel degree cap | None — split sequence into any number of blocks | Capped at attention head count |
| Attention kernel | Modified: online-softmax across ring steps | Standard: unmodified FlashAttention locally |
| Best regime | Extreme context (512K–1M+), many GPUs | Moderate context (64K–256K), degree ≤ heads |

In DeepSpeed you enable Ulysses through the sequence-parallel size in the config and the runtime, and — critically — you must keep it at or below the head count.

```json
{
  "train_micro_batch_size_per_gpu": 1,
  "sequence_parallel_size": 8,
  "zero_optimization": { "stage": 3 },
  "bf16": { "enabled": true },
  "flash_attention": { "enabled": true }
}
```

```python
# DeepSpeed-Ulysses: the sequence-parallel group must divide evenly, and
# sequence_parallel_size <= num_attention_heads (32 here) or attention breaks.
import deepspeed
from deepspeed.sequence.layer import DistributedAttention

# Wrap your attention module so its inputs are all-to-all'd into head-sharded
# form, computed with a standard local attention, and all-to-all'd back.
model_attn = DistributedAttention(local_attention, sequence_process_group)
engine, _, _, _ = deepspeed.initialize(model=model, config="ds_config.json")
```

## Composing it: context parallelism as a device-mesh dimension

Sequence and context parallelism are not replacements for the parallelism you already run — they are an *additional dimension* you compose alongside data, tensor, and pipeline parallelism. The clean mental model is a **device mesh**: a multi-dimensional grid where each dimension is one parallelism strategy, and each GPU sits at one coordinate. Context parallelism adds a "cp" axis to that mesh.

Concretely, take 64 GPUs arranged as a mesh of shape (data=2, pipeline=2, tensor=4, context=4). A GPU's coordinate tells it exactly which group it belongs to for each collective: it all-reduces gradients within its data-parallel group of 2, passes pipeline activations along its pipeline group of 2, all-reduces attention/MLP within its tensor group of 4, and rings its key/value blocks around its context group of 4. Each axis runs independently, and — the important placement rule — you put the *highest-bandwidth* parallelism on the *highest-bandwidth* interconnect. Tensor and context parallelism are chatty (all-reduce per layer, ring per attention), so they go *inside* a node on NVLink. Data and pipeline parallelism are less chatty, so they span nodes over InfiniBand. Get that placement wrong — context parallelism crossing a slow inter-node link — and the ring communication that hid perfectly on NVLink suddenly dominates, exactly as the PCIe stress test above showed.

PyTorch's `DeviceMesh` makes this composition explicit, and the emerging context-parallel API plugs into it. Here is the mesh construction and the context-parallel attention wiring, using PyTorch's experimental context-parallel support.

```python
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.experimental import context_parallel

# 64 GPUs as (dp=2, pp=2, tp=4, cp=4). Order the axes so the two chatty ones
# (tp, cp) are innermost -> mapped onto intra-node NVLink; dp/pp span nodes.
mesh = init_device_mesh(
    "cuda",
    mesh_shape=(2, 2, 4, 4),
    mesh_dim_names=("dp", "pp", "tp", "cp"),
)
cp_mesh = mesh["cp"]                      # the ring group this rank belongs to

# Shard the sequence dimension of q/k/v across the cp group and run attention
# as a ring. The context manager patches the attention op to do ring/Ulysses
# communication over cp_mesh internally.
with context_parallel(cp_mesh):
    out = model(input_ids)               # attention inside runs sequence-sharded
```

In Megatron-LM the same composition is three integer flags, and this is the production path most large long-context runs actually take. The context-parallel size multiplies into the world size alongside tensor and pipeline sizes.

```bash
# Megatron-LM: 3D + context parallelism. world_size must equal
# tp * pp * cp * dp. Here 4 * 2 * 4 * (dp) = 32 * dp GPUs.
torchrun --nnodes=8 --nproc_per_node=8 pretrain_gpt.py \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 2 \
    --context-parallel-size 4 \
    --sequence-parallel \
    --use-flash-attn \
    --seq-length 262144 \
    --num-layers 32 --hidden-size 4096 --num-attention-heads 32 \
    --micro-batch-size 1
```

Notice `--sequence-parallel` and `--context-parallel-size` together: sequence parallelism shards the LayerNorm regions along the tensor-parallel group (the free win), while context parallelism shards the attention along the separate context-parallel group (the ring). They are complementary axes, not alternatives, and a serious long-context recipe uses both. This is exactly the kind of end-to-end composition the [capstone playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) ties together for a full training run.

## Measuring it honestly

If you cannot measure activation memory and throughput correctly, you cannot tell whether context parallelism helped or whether you just moved the OOM somewhere subtler. Two measurements matter: peak memory and steady-state throughput, and both have traps at long context.

For **peak memory**, do not trust `nvidia-smi` — it reports the caching allocator's reserved pool, not what your run actually needed, and it lags. Use PyTorch's allocator counters, and reset them after warm-up so you measure steady state, not the one-time cuBLAS/NCCL workspace allocations of the first step.

```python
import torch

# ... run a few warm-up steps first so allocators and NCCL buffers settle ...
torch.cuda.reset_peak_memory_stats()
run_one_training_step()
torch.cuda.synchronize()
peak_gb = torch.cuda.max_memory_allocated() / 1e9
reserved_gb = torch.cuda.max_memory_reserved() / 1e9
print(f"peak allocated={peak_gb:.1f} GB   reserved={reserved_gb:.1f} GB")
```

The gap between `max_memory_allocated` (what you used) and `max_memory_reserved` (what the allocator grabbed from the driver) is fragmentation, and at long context with the large, oddly-sized activation tensors of a ring, fragmentation can be several gigabytes — enough to OOM a run that "should" fit. If reserved sits far above allocated, set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and re-measure before you conclude you need more context-parallel degree.

For **throughput**, the north star is model FLOP utilization (MFU) — the fraction of the GPU's peak FLOP/s your run actually achieves — and the honest way to compute it accounts for the fact that attention's FLOPs grow quadratically with sequence length. At short context the attention term is a rounding error and the standard "$6 N$ FLOPs per token" (for $N$ parameters) approximation is fine; at 256K the attention FLOPs are a *large* fraction of the total, and if you leave them out of the numerator your MFU looks artificially low. Time the step with a CUDA synchronize before and after, discard warm-up steps, run at steady state, and watch three confounds that bite hardest at long context: the **data loader** (tokenizing and packing 256K-token sequences is expensive; a slow loader starves the ring and you will blame the GPUs), **thermal throttling** (a fully compute-bound ring runs the GPUs hot and sustained; clocks droop after minutes, so a 30-second benchmark overstates steady-state throughput), and **the causal load imbalance** discussed earlier (without striped assignment, half your ring GPUs idle and your MFU halves). Measure over minutes, not seconds, with `dcgmi` or `nvidia-smi dmon` logging clocks alongside your throughput, so you can see a clock droop for what it is.

## Case studies and real numbers

Three grounded data points, because "it scales" is a claim and the literature has receipts.

**Megatron-LM context parallelism.** NVIDIA's Megatron-LM ships context parallelism (ring attention with striped assignment) as a first-class `--context-parallel-size` axis precisely because their large-context training runs needed it. Their published guidance is the practical rule this whole post argues: context parallelism is the axis you add when the *activation* memory, not the *weight* memory, is the constraint — i.e., when the sequence is long — and it composes multiplicatively with tensor, pipeline, and data parallelism in the device mesh. The activation-memory reduction is the clean $1/\text{cp}$ factor we derived; the throughput cost, on NVLink intra-node, is near zero at the long-context lengths where you would actually turn it on, because the ring communication hides under the quadratic attention compute.

**The FlashAttention precondition.** None of this exists without FlashAttention. Dao et al. (2022) showed exact attention with $O(S)$ memory instead of $O(S^2)$ by tiling and online softmax; ring attention (Liu et al., 2023) is that same online-softmax accumulation distributed across devices. The lineage is direct — ring attention is "FlashAttention where the key/value blocks live on other GPUs and rotate" — and it is why the ring loop's inner kernel is literally a FlashAttention call. If your framework's attention is not FlashAttention-based, long-context sequence parallelism cannot help you; the $O(S^2)$ score matrix will OOM you before the ring ever gets a chance to shard anything.

**Open long-context models in the wild.** Open-weights models now ship native context windows that would have been unthinkable to train a couple of years ago — see the deep dive on [Seed-OSS-36B and its 512K native context](/blog/machine-learning/large-language-model/seed-oss-36b-open-long-context-thinking-budget), a scale of context that is only trainable with sequence/context parallelism sharding the activations across the cluster. The engineering pattern behind any such release is the one in this post: FlashAttention to kill the $O(S^2)$ term, activation checkpointing to trade compute for the $O(S)$ term, and context parallelism (ring, Ulysses, or a hybrid) to split that remaining $O(S)$ activation memory across GPUs so a single 512K sequence's activations fit. There is no other known way to train those context lengths; the sequence dimension has to be split.

## When to reach for this — and when not to

Context parallelism is the sharpest example in this series of a technique that is *mandatory* in its regime and *pure overhead* outside it. The decision is almost entirely a function of one number: sequence length.

**Reach for Megatron sequence parallelism** whenever you already run tensor parallelism — full stop. It costs zero extra communication, reduces the LayerNorm-region activations by the tensor-parallel factor, and there is no regime where it hurts. If you run tensor parallelism without `--sequence-parallel`, you are leaving free memory on the table. Turn it on.

**Reach for context parallelism (ring or Ulysses)** when the *activation* memory, driven by sequence length, is your OOM — practically, past roughly 32K to 64K tokens, and unambiguously past 128K. The tell is exactly the intro: you have sharded the weights with FSDP or tensor parallelism, the weight footprint is small, and the card still OOMs on the forward pass at batch one. That empty-of-weights, full-of-activations card is context parallelism's signal. Choose *ring* for extreme context (512K–1M+) and high parallel degree, because it has no head-count cap and its communication hides better as context grows; choose *Ulysses* for the moderate regime (64K–256K) where the degree fits under your head count and its all-to-all is efficient, or use a ring/Ulysses hybrid that runs Ulysses within a node and ring across nodes.

**Do not reach for context parallelism** when the sequence already fits. Below 32K tokens, the activation memory is fine on a single card, and adding a ring only buys you cross-GPU communication you did not need — pure slowdown. Do not reach for it to solve a *weight*-memory problem either: if your model doesn't fit but your sequences are short, you need FSDP/ZeRO or tensor/pipeline parallelism, not context parallelism, because context parallelism shards activations, not weights. And do not put the context-parallel ring across a slow inter-node link if you can avoid it — keep the ring on NVLink inside a node, and if you must go multi-node for extreme context, use a Ulysses-on-node, ring-across-nodes hybrid so the hard synchronization stays on the fast fabric. If a long-context run suddenly OOMs after a resume or a config change, check the [FSDP and sharding](/blog/machine-learning/distributed-training/fsdp-in-practice) interaction and the memory model before assuming you need more context-parallel degree — fragmentation and a wrong wrap policy masquerade as needing more GPUs.

## Key takeaways

- **Activation memory is the long-context wall, and it scales with sequence length.** Weight sharding (FSDP/ZeRO) shrinks the parameter/gradient/optimizer footprint but does nothing for activations — each data-parallel rank holds the full per-sequence activation working set. Past ~128K tokens that working set OOMs a single card at batch one.
- **None of data, tensor, or pipeline parallelism splits the sequence.** Data replicates activations, tensor shards the hidden dimension (leaving LayerNorm regions and the full sequence intact), pipeline shards layers. The $O(S)$ activation term survives all three; you need a dimension that splits $S$ itself.
- **FlashAttention is the precondition.** It collapses attention from $O(S^2)$ to $O(S)$ memory. Without it the score matrix OOMs you before any sequence sharding matters. Ring attention *is* FlashAttention's online softmax stretched across GPUs.
- **Megatron sequence parallelism is free.** It shards the LayerNorm/dropout activations along the sequence, converting the tensor-parallel all-reduce into an all-gather plus reduce-scatter — identical bytes, $1/t$ the region activation. Always enable it with tensor parallelism.
- **Ring attention drops per-GPU attention memory from $O(S)$ to $O(S/N)$** by keeping each GPU's query block fixed and rotating key/value blocks around a ring with online-softmax accumulation. Its communication is $O(S)$ per GPU but overlaps with compute, and the overlap gets *easier* as context grows because compute is quadratic in block size while comms is linear.
- **Use striped/zigzag token assignment for causal models**, or contiguous blocks will make the last ring GPU do all the work and the first do none — a 2x throughput hit disguised as a parallelism win.
- **Ulysses trades a ring for two all-to-alls** and standard local attention, moving less data but capping the parallel degree at the attention head count. Ring scales without a cap; Ulysses wins in the moderate regime under the head count.
- **Compose it as a device-mesh axis** alongside data/tensor/pipeline, and keep the chatty ring on NVLink inside a node — a context-parallel ring across a slow link stops hiding and starts dominating.
- **Match the tool to the regime.** Below 32K, don't; past 128K, you must. Context parallelism is mandatory where activations overflow and pure overhead where they don't.

## Further reading

- Liu, Zaharia, Abbeel (2023), *Ring Attention with Blockwise Transformers for Near-Infinite Context* — the ring attention paper; the online-softmax-across-GPUs derivation.
- Dao, Fu, Ermon, Rudra, Ré (2022), *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* — the $O(S)$-memory attention that everything here builds on.
- Korthikanti et al. (2022), *Reducing Activation Recomputation in Large Transformer Models* — Megatron sequence parallelism and the activation-memory accounting.
- Jacobs et al. (2023), *DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models* — the all-to-all alternative.
- [Why distributed training](/blog/machine-learning/distributed-training/why-distributed-training) — the four walls and the series map.
- [The map of parallelism](/blog/machine-learning/distributed-training/the-map-of-parallelism) — where sequence/context parallelism sits among data/tensor/pipeline/expert.
- [Tensor parallelism the Megatron way](/blog/machine-learning/distributed-training/tensor-parallelism-megatron) — the companion sequence parallelism piggybacks on.
- [Collectives from scratch](/blog/machine-learning/distributed-training/collectives-from-scratch) — why an all-reduce equals a reduce-scatter plus an all-gather, the fact that makes Megatron-SP free.
- [The distributed training playbook](/blog/machine-learning/distributed-training/the-distributed-training-playbook) — the capstone that composes every axis, including context parallelism, into one recipe.

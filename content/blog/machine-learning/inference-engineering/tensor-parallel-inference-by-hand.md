---
title: "Tensor-parallel inference by hand: sharding a model across GPUs, and why it hurts decode most"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "A 70B model in bf16 is 140 GB and will not fit on one 80 GB GPU. Split every matmul across GPUs with tensor parallelism, pay exactly two all-reduces a layer, and learn the inference-specific twist nobody warns you about: the same collective that hides behind prefill's giant GEMM stands fully exposed behind decode's tiny GEMV. Built in nanoserve with real NCCL."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "tensor-parallelism",
    "distributed-inference",
    "nccl",
    "pytorch",
    "gpu",
    "kv-cache",
    "latency",
    "throughput",
    "ml-systems",
    "vllm",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 46
---

Load Llama-3.1-70B in bf16 and the weights alone are 140 GB: 70 billion parameters at 2 bytes each. An H100 has 80 GB. The model does not fit. Not "fits tightly and OOMs under load" — it does not fit at all, before you have allocated a single byte of KV cache, before a single request has arrived. So you reach for the obvious answer, the one every tutorial gives: use more GPUs. `tensor_parallel_size=8`, `torchrun`, done. It works. Throughput goes up. Everybody is happy.

And then you look at your decode latency and it is worse per-GPU than you expected, and the `nsys` trace has fat orange bars between every layer that say `ncclAllReduce`, and someone asks why the eight-GPU box is not eight times faster, and you do not have a crisp answer. This post is the crisp answer. Tensor parallelism (TP) is not "run the model on more GPUs." It is a specific, surgical split of *every matmul in the model* across ranks, and that split has a cost you can derive to the microsecond — a cost that is nearly free during prefill and brutal during decode, for one clean reason.

![A comparison matrix of tensor parallelism versus pipeline parallelism across what they split, communication per layer, the link they want, and their decode cost](/imgs/blogs/tensor-parallel-inference-by-hand-1.webp)

That figure is the whole map. There are two ways to cut a transformer across GPUs. **Tensor parallelism** splits each matmul — every rank does a slice of every layer, and they synchronize twice per layer with an *all-reduce* (a collective that sums a tensor across all ranks and hands every rank the total). **Pipeline parallelism** (PP, the subject of [the next post](/blog/machine-learning/inference-engineering/pipeline-parallel-and-multi-node-inference)) splits by layer — rank 0 runs layers 0–9, rank 1 runs 10–19, and activations flow across the boundary once per stage. TP wants a fast in-node link because it talks constantly; PP tolerates a slow cross-node link because it talks rarely. By the end of this post you will have built TP into `nanoserve` — a column-parallel and row-parallel `Linear` with a real NCCL all-reduce, the two-all-reduce decoder layer, head-sharded attention wired to the paged KV cache so each rank caches only its own heads, sharded weight loading that reads just this rank's slice off disk, and a `torchrun` entry point — and you will be able to predict, before you launch, exactly how much the collectives will cost you at batch 1 versus batch 64, on NVLink versus PCIe.

Standard promise for this series, restated from [the introduction](/blog/machine-learning/inference-engineering/what-inference-engineering-is): **I have no GPU and I have run none of this.** Every number below is arithmetic I show you in full, a spec or paper I cite with a link, or a range framed as what you will see when you run the script. The results tables carry a `Source` column. The one headline benchmark I lean on — a 13.9× jump in KV blocks — is the vLLM team's, cited and dated, never mine.

---

## 1. Why one GPU is not enough, and the two ways out

Start with the memory arithmetic, because it is what forces the whole conversation. A model's weight footprint is `parameters × bytes-per-parameter`. Llama-3.1-70B has about 70.6 billion parameters (the [model card](https://huggingface.co/meta-llama/Llama-3.1-70B) lists the config: 80 layers, hidden size 8192, 64 attention heads, 8 key-value heads, `intermediate_size` 28672, `head_dim` 128). In bf16 that is `70.6e9 × 2 ≈ 141 GB`. Round it to 140. The largest single-GPU HBM you can rent at scale is 80 GB (A100 80GB or H100 80GB SXM; the B200 pushes it to 192 GB, and we will come back to that). So the model is 1.75× larger than the biggest GPU. You are not optimizing here — you are solving a packing problem. The weights have to live somewhere, and no single "somewhere" is big enough.

There are exactly two ways to cut a transformer so that no rank has to hold the whole thing.

**Split by layer (pipeline parallelism).** Give rank 0 the embedding and layers 0–39, rank 1 layers 40–79 and the output head. A token's activations flow through rank 0, cross the boundary once, flow through rank 1, and emerge. Each rank holds half the weights. The communication is one activation tensor per stage boundary — tiny, infrequent, and it survives a slow link. The cost is a *pipeline bubble*: rank 1 sits idle while rank 0 works on the first microbatch, and you need many in-flight requests to keep both stages busy. PP is the across-node default, and it is [the next post](/blog/machine-learning/inference-engineering/pipeline-parallel-and-multi-node-inference).

**Split by matmul (tensor parallelism).** Keep all 80 layers on all ranks, but give each rank only a *slice* of every weight matrix. Rank 0 computes the first 1/8 of every projection, rank 3 the fourth 1/8, and so on. Each rank holds 1/8 of the weights. The communication is an all-reduce inside every layer, twice — frequent, latency-sensitive, and it *demands* a fast link. TP is the within-node default because a modern node has NVLink, a GPU-to-GPU fabric an order of magnitude faster than PCIe.

The comparison figure above is the executive summary: TP splits each matmul and pays two all-reduces per layer on NVLink and stalls hard at the collective barrier during decode; PP splits by layer, sends once per stage, tolerates PCIe or InfiniBand across nodes, and pays a bubble instead of a barrier. This post is entirely about the first row. The whole reason TP earns its complexity is the second column — those two all-reduces — so we derive them from the matmul up, then we measure what they cost. Most posts stop at "TP has communication overhead." We are going to compute it.

One framing to keep in your pocket, because it is the thesis: **an all-reduce costs the same number of bytes whether the GPU just did a giant prefill GEMM or a tiny decode GEMV. Prefill hides the collective behind its huge matmul. Decode cannot. That asymmetry is the entire inference story of tensor parallelism**, and it is why a technique that looks like a pure throughput win on paper can quietly tax your per-token latency.

---

## 2. One matmul, two ways to split

Everything in TP reduces to one question asked of every weight matrix: *when I cut this matrix across GPUs, do I split its output dimension or its input dimension?* Those are the only two choices, they have different communication consequences, and a transformer layer is built so that they cancel out into exactly two all-reduces. Let me derive both, because once you see why one is free and the other is not, the rest of the post is bookkeeping.

Take a linear layer $y = xW$, where $x$ is a row vector of length $d_\text{in}$ (one token's hidden state) and $W$ is $d_\text{in} \times d_\text{out}$. There are two natural ways to shard $W$ across $N$ ranks.

**Column-parallel** splits $W$ along its **output** columns. Rank $r$ holds $W_r$, a $d_\text{in} \times (d_\text{out}/N)$ slice, and every rank gets the **full** input $x$. Rank $r$ computes $y_r = x W_r$, a slice of the output of width $d_\text{out}/N$. Here is the key: each rank multiplies the *whole* input by its columns and produces its own *piece* of the output, and those pieces are already correct — they never overlap, so there is nothing to add up. **Column-parallel needs no communication for the matmul itself.** The output arrives pre-sharded, one slice per rank, and — this is the trick that makes the layer cheap — you can *keep* it sharded and feed it straight into the next matmul.

**Row-parallel** splits $W$ along its **input** rows. Rank $r$ holds $W_r$, a $(d_\text{in}/N) \times d_\text{out}$ slice, and receives only its **slice** of the input, $x_r$ of length $d_\text{in}/N$. Rank $r$ computes $y_r = x_r W_r$, a $d_\text{out}$-length vector — full width, but a **partial sum**. Why partial? Because the true output is $y = \sum_{k} x_k W_{k}$ summed over the whole contracted dimension, and rank $r$ only owns the terms for its slice of $k$. Each rank has a fraction of the sum. To recover the true $y$ you must add the partials across all ranks: $y = \sum_r y_r$. That sum, computed collectively so every rank ends up holding the total, **is an all-reduce**. Row-parallel cannot avoid it — the math is a sum over a dimension that has been scattered across GPUs.

![A before and after comparison contrasting column-parallel splitting the output dimension with no communication against row-parallel splitting the input dimension and needing an all-reduce](/imgs/blogs/tensor-parallel-inference-by-hand-3.webp)

The figure makes the asymmetry concrete: column-parallel splits the output dimension, every rank keeps the full input, runs a local matmul, and stops — no communication. Row-parallel splits the input dimension, every rank ends with a partial sum, and only an all-reduce completes it. That is not a design preference you can tune away; it is a property of which dimension you cut. Split the output and the pieces are independent. Split the contracted input and the pieces must be summed.

Now the beautiful part, the one that makes a transformer layer cost *two* all-reduces and not more. If you make projection A column-parallel, its output comes out sharded along the feature dimension. If the very next projection B is row-parallel, it *wants* its input sharded along exactly that dimension. So A's sharded output flows straight into B with no reshuffle, B produces the partial sum, and you all-reduce once — at B, not at A. **Column-then-row is a communication-free handoff followed by a single all-reduce.** Megatron-LM's tensor-parallel design (Shoeybi et al., 2019, [arxiv.org/abs/1909.08053](https://arxiv.org/abs/1909.08053)) is built entirely on this pairing, and it is why the pattern is universal. You pay for the row half of each pair; the column half is free.

A transformer layer has two such pairs. In attention, the QKV projection is column-parallel (split the heads across ranks) and the output projection O is row-parallel (all-reduce #1). In the MLP, the gate and up projections are column-parallel and the down projection is row-parallel (all-reduce #2). Two column-then-row pairs, two all-reduces, per layer. That is the number to burn into memory.

#### Worked example: sizing the two projections in Llama-3.1-70B

Concretely, per layer at TP=8, here is what each rank holds and where the comm lands. The QKV projection maps hidden 8192 into `Q(64×128) + K(8×128) + V(8×128) = 8192 + 1024 + 1024 = 10240` output features; column-parallel at TP=8 gives each rank `10240/8 = 1280` output features — that is 8 query heads, 1 key head, 1 value head per rank, computed from the full hidden state with **no comm**. The O projection maps the 8192 attention output back to 8192 hidden; row-parallel at TP=8, each rank holds `8192/8 = 1024` input rows, produces a partial sum, and the ranks **all-reduce** it (#1). The MLP repeats the shape: gate and up are column-parallel `8192 → 28672/8 = 3584` per rank (no comm), down is row-parallel `28672/8 = 3584 → 8192` per rank ending in the second all-reduce. `Source: derived` (dimensions from the Llama-3.1-70B config; the column/row pairing from Megatron-LM). Notice the KV heads: 8 of them, one per rank at TP=8. Hold that thought — it is the cleanest possible case, and we will break it in the stress test.

---

## 3. Two all-reduces per layer, and where they live

Let me trace one full decoder layer with the communication marked, because the *placement* of the two all-reduces — not just their count — is what determines whether they hide or expose. This is also the section whose figure you should stare at longest.

The layer, in order: RMSNorm the input (no comm — RMSNorm is per-token and each rank has the full hidden state at the residual stream). Column-parallel QKV projection (no comm; output is sharded, each rank now holds its own heads' Q, K, V). Attention runs **entirely locally** on each rank — rank $r$ attends rank $r$'s query heads over rank $r$'s cached key/value heads, and because heads are independent in multi-head attention, no rank needs another rank's heads. Row-parallel O projection produces a partial sum and **all-reduces (#1)**, so the attention output re-materializes as the full, correct hidden state on every rank. Residual add (no comm). RMSNorm again. Column-parallel gate and up (no comm; sharded). SwiGLU activation, elementwise, local. Row-parallel down projection produces a partial sum and **all-reduces (#2)**. Residual add. Done. Two collectives, both at the *end* of a row-parallel projection, both on a full-width hidden-dimension tensor.

![A branching dataflow graph showing the hidden state fanning to two ranks for column-parallel QKV, local attention heads on each rank, a first all-reduce at the output projection, then the MLP repeating the pattern into a second all-reduce](/imgs/blogs/tensor-parallel-inference-by-hand-2.webp)

Read the graph as the shape of the communication. The full hidden state $x$ replicates to every rank at the top — that is free, it is already there on the residual stream. It fans out into per-rank column-parallel QKV with no communication on the fan-out edges. Each rank runs its own attention heads in isolation. Then the two arrows converge into `all-reduce 1`: that is the row-parallel O projection summing the partial outputs back into a whole. The MLP is the same shape one more time, converging into `all-reduce 2`. The two caution-colored nodes are the only two places in the entire layer where ranks must talk. Everything else — the projections, the attention, the norms, the activation — is embarrassingly parallel.

This is worth dwelling on because it is *why TP works at all*. Attention heads are independent: head 7's output does not depend on head 12's. Split the heads across ranks and each rank does complete, correct attention for its heads with zero cross-rank dependency — the KV cache for rank $r$'s heads lives on rank $r$ and never needs to move. The MLP's hidden dimension is likewise a sum of independent contributions. The transformer is, structurally, almost perfectly shardable. The two all-reduces are the *only* seams, and they exist purely because the residual stream must be whole again before the next norm reads it. If you could keep the hidden state sharded end to end you would pay nothing — but RMSNorm needs the full vector to compute its normalization statistic, so twice a layer you must reassemble.

There is a subtlety worth naming so you do not trip on it: the two all-reduces are what a plain implementation does, and it is what `nanoserve` will do. Production engines shave one of them in specific configurations — *sequence parallelism* replaces the all-reduce with a reduce-scatter plus all-gather split around the norm, moving the same total bytes but overlapping better, and fused kernels merge the all-reduce with the RMSNorm that follows it. We will cite those in the case studies. But the mental model — and the count you reason with — is two all-reduces per layer, on a full hidden-dimension tensor, at the end of each row-parallel projection. Get that right and the cost model in section 6 falls out mechanically.

---

## 4. Building it in nanoserve

Time to make it real. TP in `nanoserve` is four pieces: two parallel `Linear` layers (column and row), the decoder layer that wires them into the two-all-reduce pattern, head-sharded attention hooked to the paged cache, and sharded weight loading. Plus a `torchrun` entry point. This is genuine `torch.distributed` / NCCL code — copy-adaptable, real shapes, real edge cases.

First, the process group and the collective. Every rank runs the same script; `torchrun` sets `RANK`, `WORLD_SIZE`, and `LOCAL_RANK` in the environment, and each rank binds to one GPU.

```python
# nanoserve/parallel.py
import os
import torch
import torch.distributed as dist

def init_tp() -> tuple[int, int]:
    """Initialize the tensor-parallel process group. One rank == one GPU.
    Launched by: torchrun --nproc_per_node=8 -m nanoserve.serve ...
    Returns (tp_rank, tp_size)."""
    tp_rank = int(os.environ["RANK"])
    tp_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return tp_rank, tp_size

def all_reduce(x: torch.Tensor) -> torch.Tensor:
    """Sum x across all ranks in place; every rank ends with the total.
    This is the ONLY collective a TP transformer layer needs, twice per layer."""
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(x, op=dist.ReduceOp.SUM)  # NCCL ring all-reduce
    return x
```

Now the two linear layers. The column-parallel layer holds `out_features / tp_size` output rows and does no communication in the forward pass — its output is deliberately left sharded so the next (row-parallel) layer can consume it directly.

```python
# nanoserve/parallel.py (continued)
import torch.nn.functional as F

class ColumnParallelLinear(torch.nn.Module):
    """Split the OUTPUT dimension across ranks. Each rank holds out/TP rows
    of W and computes its own slice of y = x W^T from the FULL input x.
    No communication: the sharded output is fed straight to a RowParallelLinear."""
    def __init__(self, in_features: int, out_features: int,
                 tp_size: int, bias: bool = False):
        super().__init__()
        assert out_features % tp_size == 0, "out dim must divide by TP size"
        self.out_per_rank = out_features // tp_size
        # weight is [out_per_rank, in_features]; this rank's slice only
        self.weight = torch.nn.Parameter(
            torch.empty(self.out_per_rank, in_features, dtype=torch.bfloat16))
        self.bias = (torch.nn.Parameter(torch.empty(self.out_per_rank,
                     dtype=torch.bfloat16)) if bias else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [tokens, in_features] (full) -> y: [tokens, out_per_rank] (sharded)
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(torch.nn.Module):
    """Split the INPUT dimension across ranks. Each rank holds in/TP columns
    of W and receives its slice of x, producing a PARTIAL sum of full width.
    The forward ends in the all-reduce that completes the sum."""
    def __init__(self, in_features: int, out_features: int,
                 tp_size: int, bias: bool = False):
        super().__init__()
        assert in_features % tp_size == 0, "in dim must divide by TP size"
        self.in_per_rank = in_features // tp_size
        # weight is [out_features, in_per_rank]; this rank's columns only
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, self.in_per_rank, dtype=torch.bfloat16))
        # bias added once, after the all-reduce, so add it only on rank 0
        self.bias = (torch.nn.Parameter(torch.empty(out_features,
                     dtype=torch.bfloat16)) if bias else None)

    def forward(self, x_shard: torch.Tensor) -> torch.Tensor:
        # x_shard: [tokens, in_per_rank] -> partial [tokens, out_features]
        y_partial = F.linear(x_shard, self.weight)   # NO bias yet
        y = all_reduce(y_partial)                     # THE all-reduce
        if self.bias is not None:
            y = y + self.bias                         # bias once, post-reduce
        return y
```

Two details that bite people. First, the bias on a row-parallel layer is added **after** the all-reduce, once, not on every rank before it — otherwise you sum the bias $N$ times. Second, notice `RowParallelLinear.forward` takes an *already-sharded* input; it never sees the full $x$. That is the contract that makes column-then-row free: the column layer emits sharded, the row layer consumes sharded, and the only reassembly is the final all-reduce.

Now the decoder layer, wiring the two pairs into the exact two-all-reduce pattern from section 3.

```python
# nanoserve/tp_layer.py
import torch
import torch.nn.functional as F
from .parallel import ColumnParallelLinear, RowParallelLinear

class TPDecoderLayer(torch.nn.Module):
    def __init__(self, cfg, tp_size: int):
        super().__init__()
        self.n_heads = cfg.num_attention_heads // tp_size       # local Q heads
        self.n_kv    = cfg.num_key_value_heads // tp_size        # local KV heads
        self.head_dim = cfg.head_dim
        h = cfg.hidden_size
        qkv_out = (cfg.num_attention_heads + 2 * cfg.num_key_value_heads) \
                  * cfg.head_dim
        # attention: column-parallel QKV (no comm) -> row-parallel O (all-reduce #1)
        self.qkv_proj = ColumnParallelLinear(h, qkv_out, tp_size)
        self.o_proj   = RowParallelLinear(cfg.num_attention_heads * cfg.head_dim,
                                          h, tp_size)
        # mlp: column-parallel gate & up (no comm) -> row-parallel down (all-reduce #2)
        self.gate_proj = ColumnParallelLinear(h, cfg.intermediate_size, tp_size)
        self.up_proj   = ColumnParallelLinear(h, cfg.intermediate_size, tp_size)
        self.down_proj = RowParallelLinear(cfg.intermediate_size, h, tp_size)
        self.input_norm = RMSNorm(h)
        self.post_norm  = RMSNorm(h)

    def forward(self, x, positions, kv_cache, block_table):
        # --- attention block ---
        h = self.input_norm(x)                       # full hidden, local
        qkv = self.qkv_proj(h)                        # column-parallel, NO comm
        q, k, v = self._split_qkv(qkv)                # this rank's heads only
        attn = paged_attention(q, k, v, positions,    # local heads, local cache
                               kv_cache, block_table,
                               self.n_heads, self.n_kv, self.head_dim)
        x = x + self.o_proj(attn)                     # row-parallel -> ALL-REDUCE #1
        # --- mlp block ---
        h = self.post_norm(x)                         # full hidden, local
        act = F.silu(self.gate_proj(h)) * self.up_proj(h)   # column-parallel, NO comm
        x = x + self.down_proj(act)                   # row-parallel -> ALL-REDUCE #2
        return x

    def _split_qkv(self, qkv):
        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv * self.head_dim
        q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)
        return (q.view(-1, self.n_heads, self.head_dim),
                k.view(-1, self.n_kv, self.head_dim),
                v.view(-1, self.n_kv, self.head_dim))
```

The attention is head-sharded and wired to the paged cache from [the paged KV cache post](/blog/machine-learning/inference-engineering/paged-kv-cache-implementing-blocks-and-a-block-table): each rank owns `num_kv_heads / tp_size` key-value heads and stores only *those* heads' K and V in its own cache. This is the memory win in code — the KV cache is sharded exactly like the weights, so an 8-way split gives each rank 1/8 of the cache too.

```python
# nanoserve/tp_attention.py
def paged_attention(q, k, v, positions, kv_cache, block_table,
                    n_heads, n_kv, head_dim):
    """Each rank caches and attends ONLY its local KV heads.
    kv_cache holds this rank's n_kv heads -> 1/TP of the global cache."""
    # append this step's K,V for THIS rank's heads into the paged cache
    write_kv_to_cache(kv_cache, block_table, positions, k, v)   # local heads
    k_all, v_all = gather_from_cache(kv_cache, block_table)     # local heads
    # GQA: n_heads query heads share n_kv KV heads within this rank
    return grouped_query_attention(q, k_all, v_all,
                                   n_heads=n_heads, n_kv=n_kv,
                                   head_dim=head_dim)            # NO cross-rank comm
```

Finally, sharded weight loading — the piece that keeps you from OOMing on load. You must **never** materialize the full 140 GB tensor and then slice it; you slice as you read, straight out of the safetensors file, so each rank only ever touches its 1/8.

```python
# nanoserve/load_tp.py
from safetensors import safe_open

def load_column_shard(path, name, tp_rank, tp_size):
    """Load only this rank's OUTPUT rows of a column-parallel weight."""
    with safe_open(path, framework="pt", device="cpu") as f:
        w = f.get_slice(name)               # lazy view, nothing read yet
        out_dim = w.get_shape()[0]
        per = out_dim // tp_size
        lo, hi = tp_rank * per, (tp_rank + 1) * per
        return w[lo:hi, :].to("cuda", torch.bfloat16)   # reads ONLY this slice

def load_row_shard(path, name, tp_rank, tp_size):
    """Load only this rank's INPUT columns of a row-parallel weight."""
    with safe_open(path, framework="pt", device="cpu") as f:
        w = f.get_slice(name)
        in_dim = w.get_shape()[1]
        per = in_dim // tp_size
        lo, hi = tp_rank * per, (tp_rank + 1) * per
        return w[:, lo:hi].to("cuda", torch.bfloat16)   # reads ONLY this slice
```

`safe_open(...).get_slice(name)[lo:hi, :]` reads just the requested bytes off disk — it does not load the whole tensor to slice it in memory. That is the difference between eight ranks each reading 17.5 GB and eight ranks each trying to read 140 GB and dying. The `torchrun` launch ties it together:

```bash
torchrun --nproc_per_node=8 --master_port=29500 \
  -m nanoserve.serve \
  --model meta-llama/Llama-3.1-70B \
  --tensor-parallel-size 8
```

Each of the eight processes calls `init_tp()`, binds to its GPU, loads its column/row shards, builds the `TPDecoderLayer` stack, and enters the same decode loop from [the continuous-batching post](/blog/machine-learning/inference-engineering/writing-a-continuous-batching-loop). The all-reduces inside `RowParallelLinear` are the only synchronization. That is a complete, honest tensor-parallel inference engine in a few hundred lines. Now let us find out what it costs.

---

## 5. Each rank holds 1/TP, and the memory that frees

The first-order effect of TP is the one that motivated it: each rank holds `1/TP` of the weights. Split 140 GB eight ways and every rank carries 17.5 GB. That is not just "the model now fits" — it is a large amount of freed HBM on every GPU, and what you do with that freed room is where TP stops being a packing hack and becomes a throughput lever.

![A stacked memory budget for one H100 showing weights over TP taking a small slice, activations and a collective-buffer reserve, and most of the 80 GB left free for the KV cache](/imgs/blogs/tensor-parallel-inference-by-hand-4.webp)

The stack shows one rank's 80 GB after an 8-way split: 17.5 GB of weights, a few GB of activations and workspace, a reserve for NCCL's collective buffers (the all-reduce needs scratch space — typically on the order of a gigabyte, and it is real, do not forget it), and the rest — the large success-colored block — free for the KV cache. Every GPU gets that much KV room. Contrast the single-GPU fantasy where 70 GB of weights (even in fp8) leaves almost nothing.

Here is the arithmetic for the running model across TP degrees, on 80 GB H100s, bf16 weights. Let $W = 140$ GB be the total weight bytes and $C \approx 74$ GB be the usable per-GPU capacity after activations, workspace, and the collective reserve (call the overhead ~6 GB; it varies with batch and sequence length). Weights per rank are $W / \text{TP}$, and the KV room per rank is $C - W/\text{TP}$.

| Config | Weights/rank | KV room/rank | KV room total | Source |
| --- | --- | --- | --- | --- |
| TP=1 | 140 GB | does not fit | — | derived |
| TP=2 | 70 GB | 4 GB | 8 GB | derived |
| TP=4 | 35 GB | 39 GB | 156 GB | derived |
| TP=8 | 17.5 GB | 56.5 GB | 452 GB | derived |

Read down the "KV room total" column and something jumps out: it does not grow linearly with TP, it grows *faster*. Going TP=2 → TP=4 doubles the GPU count but takes total KV room from 8 GB to 156 GB — nearly 20×. That super-linearity is the counterintuitive win, and it has a clean derivation. Total KV room across all ranks is

$$\text{KV}_\text{total} = \text{TP} \cdot \left(C - \frac{W}{\text{TP}}\right) = \text{TP}\cdot C - W.$$

The freed-memory term $\text{TP}\cdot C$ grows linearly, but the weight tax $W$ is a **fixed** subtraction that does not grow with TP. When the weights nearly fill the GPU — when $W$ is close to $C$ — the single-GPU KV room $C - W$ is a tiny sliver, and adding even one more GPU pushes $\text{TP}\cdot C - W$ up by a whole $C$, which is enormous relative to that sliver. The ratio blows up precisely because you were memory-starved to begin with. We will put a cited number on this in the case studies; for now, the formula tells you *why* it is super-linear: **TP does not just add memory, it stops each GPU from paying the full weight tax.**

#### Worked example: how many concurrent sequences does the freed KV buy?

At TP=8, each rank has ~56.5 GB of KV room. Llama-3.1-70B keeps 8 KV heads × 128 dims × 80 layers × 2 (K and V) × 2 bytes = 327,680 bytes ≈ 320 KB per token *globally*; sharded 8 ways, each rank stores 40 KB per token (its one KV head's share). So one rank's 56.5 GB holds `56.5e9 / 40e3 ≈ 1.4 million` token-slots. Divide by a 4k-token context and you can hold ~340 concurrent 4k sequences before the cache is full — on a model that did not even fit a moment ago. `Source: derived` (KV formula from [the memory-math post](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache); capacity and overhead are order-of-magnitude estimates — measure yours with `torch.cuda.mem_get_info()`). The point is not the exact 340; it is that the freed room is measured in *hundreds of sequences*, which is what turns "the model fits" into "the server has throughput."

---

## 6. The all-reduce bill, derived

Now the cost side. Everything above is why TP is attractive; this section is the price, and it is where you earn the right to predict latency instead of guessing. We derive the per-token communication volume from the layer structure, then divide by interconnect bandwidth to get the floor on collective time.

Each all-reduce operates on the layer's hidden-dimension tensor. For one token in bf16, that tensor is `hidden × 2 bytes = 8192 × 2 = 16,384 bytes`, call it 16 KB. A NCCL ring all-reduce does not move exactly 16 KB per rank, though — a ring all-reduce over $N$ ranks moves ${2(N-1)/N}$ times the message size across each rank's links (a reduce-scatter phase plus an all-gather phase, each moving $(N-1)/N$ of the data). So per all-reduce, per rank, the wire traffic is

$$B_\text{ar} = \frac{2(N-1)}{N}\cdot S, \qquad S = \text{hidden} \times \text{bytes} = 16\text{ KB per token}.$$

There are two all-reduces per layer and 80 layers, so 160 all-reduces per token. The per-token communication volume per rank is

$$B_\text{token} = 160 \cdot \frac{2(N-1)}{N}\cdot 16\text{ KB}.$$

Plug in the TP degrees. At TP=2 the ring factor is $2\cdot1/2 = 1.0$; at TP=4 it is $2\cdot3/4 = 1.5$; at TP=8 it is $2\cdot7/8 = 1.75$. That gives roughly 2.6, 3.8, and 4.5 MB moved per token per rank. Then divide by bandwidth for the *bandwidth-limited floor* on collective time — the pure data-movement term, ignoring latency and launch overhead. Modern NVLink runs at 1.8 TB/s per GPU (the vLLM/SemiAnalysis [InferenceMAX Blackwell writeup](https://vllm.ai/blog/2025-10-09-blackwell-inferencemax), 2025-10-09, lists B200 NVLink at 1.8 TB/s per GPU; note Hopper's NVLink4 is 900 GB/s, so double the NVLink column on an H100). PCIe Gen4 x16 is about 32 GB/s per direction — roughly 56× slower.

![A matrix of the per-token all-reduce traffic at three tensor-parallel degrees with the bandwidth-limited time on NVLink versus PCIe](/imgs/blogs/tensor-parallel-inference-by-hand-5.webp)

| TP | Comm/token/rank | NVLink 1.8 TB/s | PCIe 32 GB/s | Source |
| --- | --- | --- | --- | --- |
| 2 | ~2.6 MB | ~1.4 µs | ~80 µs | derived |
| 4 | ~3.8 MB | ~2.1 µs | ~120 µs | derived |
| 8 | ~4.5 MB | ~2.5 µs | ~140 µs | derived |

The figure and the table say the same thing from two sides: the raw volume is only a few megabytes per token, which on NVLink is a couple of microseconds and on PCIe is a hundred-plus. That 56× gap is the entire reason the folklore rule exists — the vLLM team states it plainly in [Distributed Inference with vLLM](https://vllm.ai/blog/2025-02-17-distributed-inference) (2025-02-17): *"Use pipeline parallelism across nodes and tensor parallelism within nodes when interconnects are slow."* TP wants NVLink because TP talks 160 times per token; put those 160 collectives on PCIe and you are moving megabytes through a straw, 160 times, every token.

Two honesty caveats on these numbers, because they are floors, not measurements. **First, this is only the bandwidth term.** A real all-reduce also pays a fixed *latency* — the kernel launch, the ring setup, the barrier synchronization — that is on the order of several microseconds *per collective regardless of message size*. At batch 1 the message (16 KB) is far below the size where bandwidth dominates, so each of the 160 collectives is actually **latency-bound**, and the real per-token comm cost is closer to `160 × (a few µs of latency)` than to the tiny bandwidth term. The table's microseconds are the optimistic floor; the batch-1 reality is worse, and that is exactly the setup for the next section. **Second, on PCIe the latency penalty is much larger** (no direct GPU-to-GPU path; traffic crosses a PCIe switch or the host), so the PCIe column is doubly pessimistic at batch 1. `Source: derived` for the bandwidth terms; the latency floor is a cited property of collective libraries (NCCL), not a first-hand measurement.

#### Worked example: the batch-1 comm tax at TP=8 on NVLink

Put the two halves together for one concrete decode step. At TP=8 the per-token bandwidth term is ~2.5 µs (the table), but that is the floor — at batch 1 each of the 160 collectives is latency-bound, not bandwidth-bound, because a 16 KB message is far below the size where NVLink's 1.8 TB/s matters. Take a per-collective latency of ~7 µs (a plausible NVLink figure — measure yours with a NCCL micro-benchmark). Then the two all-reduces in one layer cost about 14 µs of exposed barrier. The useful compute in that same layer, from section 7, is ~11 µs of attention GEMV plus ~52 µs of MLP GEMV, so ~63 µs. The comm-to-compute ratio is `14 / 63 ≈ 22%`. Across 80 layers that is `80 × 14 = 1.12 ms` of pure collective stall added to a decode step whose compute floor is `80 × 63 µs ≈ 5.0 ms`. So the batch-1 TPOT climbs from ~5.0 ms to ~6.1 ms — a real, un-tunable tax you can predict before you launch. `Source: derived` (bandwidth from the section-6 table; the ~7 µs per-collective latency is an order-of-magnitude estimate from NCCL's fixed-cost regime — replace it with your measured value). Now redo the arithmetic with the per-collective latency of PCIe, which is several times higher, and the 22% becomes the majority of the step: that is the "TP over PCIe collapses" claim, quantified.

---

## 7. Why decode cannot hide the collective

Here is the inference-specific twist, the thing that separates serving from training and the reason this post exists. Training and prefill do TP all the time and barely feel it. Decode does the *same* two all-reduces per layer and they dominate. The difference is not the communication — it is what the communication is standing next to.

Recall the two phases. **Prefill** processes the whole prompt at once: thousands of tokens through each matmul, so every projection is a big, fat GEMM (matrix times matrix) that runs for a long time and saturates the tensor cores. **Decode** processes one new token per sequence: each projection is a GEMV (matrix times a single vector) — a skinny, memory-bound trickle that finishes almost instantly and leaves the tensor cores nearly idle (this is the whole subject of [the skinny-matrix GEMM post](/blog/machine-learning/inference-engineering/gemm-for-decode-the-skinny-matrix-problem)). The all-reduce between two projections is the same collective in both phases. But in prefill it hides behind the long GEMM; in decode there is nothing for it to hide behind.

![A timeline of one decode layer showing a tiny attention GEMV, then an exposed all-reduce barrier where the GPU idles, then a tiny MLP GEMV, then a second all-reduce barrier that stalls the next layer](/imgs/blogs/tensor-parallel-inference-by-hand-6.webp)

Trace the decode timeline. The attention GEMV finishes — at TP=8 on an H100, each rank's attention weights are `151M params / 8 = 18.9M`, which at 2 bytes is 37.7 MB, streamed across HBM at 3.35 TB/s in about 11 µs (`37.7e6 / 3.35e12 ≈ 11 µs`, derived). Then the layer hits all-reduce #1 — a barrier, every rank must wait for every other rank, and because the GEMV was tiny there is no queued compute to overlap it with, so the GPU **idles** on the collective. Then the MLP GEMV runs: each rank's MLP weights are `705M / 8 = 88M params`, 176 MB, about 52 µs at 3.35 TB/s (`176e6 / 3.35e12 ≈ 52 µs`, derived). Then all-reduce #2, another exposed barrier, and the next layer stalls waiting for it. Per layer you have ~63 µs of useful GEMV compute and two fully-exposed collective barriers.

Put a latency number on the barrier and the regime is obvious. If each batch-1 all-reduce costs ~7 µs of latency (a reasonable NVLink figure — cite NCCL's per-collective floor and measure yours), then two per layer is ~14 µs of pure stall on ~63 µs of compute: a **22% comm tax** at TP=8 on NVLink, at batch 1, that no amount of kernel optimization removes, because it is not a kernel — it is a barrier. Now move it to PCIe, where the collective latency is several times higher *and* the bandwidth term climbs into the tens of microseconds, and the comm can exceed the compute: decode throughput collapses, and you have discovered the hard way why nobody runs TP over PCIe.

The contrast with prefill is the payoff, and it is worth animating because the meaning is entirely in the *timing*, not in a still frame. In prefill the same all-reduce is launched, but the GEMM before it runs for hundreds of microseconds (thousands of tokens of compute), and NCCL overlaps the collective with that compute — the all-reduce tucks inside the GEMM and is never on the critical path. In decode the GEMV finishes in microseconds and the collective stands fully exposed. Same bytes, opposite outcome, because the thing it hides behind shrank to nothing.

<figure class="blog-anim">
<svg viewBox="0 0 720 360" role="img" aria-label="Two tensor-parallel lanes under one shared clock: the prefill lane runs a long GEMM that fully overlaps its all-reduce, while the decode lane runs a tiny GEMV then sits exposed on the all-reduce barrier and idles" style="width:100%;height:auto;max-width:860px">
<title>Prefill hides the all-reduce behind a long GEMM while decode stalls exposed on it</title>
<style>
.tp-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.tp-sub{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.tp-track{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.tp-gemm{fill:var(--accent,#6366f1);opacity:.22}
.tp-ar{fill:#d97706;opacity:.55}
.tp-idle{fill:var(--text-secondary,#6b7280);opacity:.16}
.tp-chip{font:600 12px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.tp-now{stroke:var(--accent,#6366f1);stroke-width:3;opacity:.85}
.tp-fin{font:600 12px ui-sans-serif,system-ui;text-anchor:middle;opacity:0}
.tp-win{fill:var(--accent,#6366f1)}
.tp-lose{fill:#d97706}
@keyframes tp-clock{0%{transform:translateX(0)}88%{transform:translateX(600px)}100%{transform:translateX(600px)}}
@keyframes tp-reveal{0%,10%{opacity:0}22%,100%{opacity:1}}
.tp-sweep{animation:tp-clock 11s linear infinite}
.tp-fin{animation:tp-reveal 11s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.tp-sweep{animation:none;transform:translateX(600px)}.tp-fin{animation:none;opacity:1}}
</style>
<text class="tp-lbl" x="24" y="30">Prefill: one long GEMM, the all-reduce tucks inside it</text>
<text class="tp-sub" x="24" y="50">thousands of tokens of compute, the collective overlaps and stays hidden</text>
<rect class="tp-track" x="24" y="64" width="600" height="56" rx="8"/>
<rect class="tp-gemm" x="28" y="68" width="592" height="48" rx="6"/>
<rect class="tp-ar" x="470" y="74" width="70" height="36" rx="5"/>
<text class="tp-chip" x="300" y="97">big GEMM runs the whole step</text>
<text class="tp-chip" x="505" y="97" style="fill:#fff">AR hidden</text>
<text class="tp-fin tp-win" x="590" y="140">done on time</text>
<text class="tp-lbl" x="24" y="200">Decode: a tiny GEMV, then the all-reduce barrier stands exposed</text>
<text class="tp-sub" x="24" y="220">one token of compute, the same collective now dominates the step</text>
<rect class="tp-track" x="24" y="234" width="600" height="56" rx="8"/>
<rect class="tp-gemm" x="28" y="238" width="150" height="48" rx="6"/>
<rect class="tp-ar" x="182" y="238" width="150" height="48" rx="6"/>
<rect class="tp-idle" x="336" y="238" width="284" height="48" rx="6"/>
<text class="tp-chip" x="103" y="267">GEMV</text>
<text class="tp-chip" x="257" y="267" style="fill:#fff">all-reduce</text>
<text class="tp-chip" x="478" y="267">GPU idle · waits on comm</text>
<text class="tp-fin tp-lose" x="300" y="312">finishes late · barrier stalls the rank</text>
<rect class="tp-now tp-sweep" x="28" y="60" width="2" height="234"/>
</svg>
<figcaption>Under one shared clock the prefill lane keeps a long GEMM running so its all-reduce overlaps and stays hidden, while the decode lane finishes its tiny GEMV early and then sits exposed on the all-reduce barrier, idling for the rest of the step.</figcaption>
</figure>

The way to fight this is not to remove the all-reduce — you cannot, the math needs it — but to make it hideable or cheaper. Larger batches help: at batch 64 the GEMV becomes a fatter GEMV-batched matmul that runs long enough to overlap the collective, and the comm-to-compute ratio improves, which is why TP throughput at high batch looks nothing like TP latency at batch 1. Fused kernels help: merging the all-reduce with the RMSNorm that immediately follows it removes a launch and overlaps the reduction with the norm — the vLLM team reports the AllReduce+RMSNorm fusion is worth up to 15% (cited below). And [CUDA graphs](/blog/machine-learning/inference-engineering/cuda-graphs-and-torch-compile-for-the-decode-loop) capture the collective into the replayed step so the launch overhead around it vanishes. But the barrier itself remains: **at batch 1, on any link, TP taxes decode, and the tax grows with TP degree because every added rank is another participant the barrier must wait for.**

---

## 8. The counterintuitive win: TP frees more KV than it costs

Now stack the cost against the benefit and something surprising falls out, and it is the number to remember from this whole post. TP *adds* a per-token comm tax to decode — that is real and we just derived it. But TP also *frees* KV-cache room super-linearly, and freed KV room means larger batches, and larger batches mean higher throughput. On a memory-starved model the freed-memory win can dwarf the comm tax by a wide margin.

![A before and after comparison of a memory-starved model on one GPU versus split across two GPUs, showing the KV room and achievable throughput jumping super-linearly](/imgs/blogs/tensor-parallel-inference-by-hand-7.webp)

The vLLM team measured exactly this. In [Distributed Inference with vLLM](https://vllm.ai/blog/2025-02-17-distributed-inference) (2025-02-17), moving from TP=1 to TP=2 gave **13.9× more KV-cache blocks** and **3.9× more token throughput** — super-linear, and explicitly attributed to the freed memory rather than the doubled compute. The figure shows the mechanism: a 70 GB model (fp8) on one 80 GB GPU leaves only ~6 GB for the cache and can hold a handful of sequences; split across two GPUs, each rank holds 35 GB of weights and frees ~41 GB, so the two ranks together offer ~82 GB of KV room. Room goes 6 → 82 GB, a 13.7× jump — which is the 13.9× the vLLM team reports, right out of the $\text{KV}_\text{total} = \text{TP}\cdot C - W$ formula from section 5.

Why 3.9× throughput and not 13.9×? Because throughput does not scale one-for-one with KV blocks once you have enough of them. More blocks let you run a larger batch, and a larger batch amortizes the weight-read and the collective across more tokens — up to the point where you become compute- or bandwidth-bound and adding sequences stops helping. So 13.9× more blocks converts to 3.9× more tokens per second: still nearly quadruple throughput from doubling the GPUs, still firmly super-linear, still a fantastic trade — but not linear in the block count. `Source: cited (vLLM Distributed Inference, 2025-02-17)` for both multipliers; `derived` for the 6 → 82 GB mechanism.

This is the resolution of the apparent paradox. Section 7 said TP hurts decode; section 8 says TP triples throughput. Both are true, and they are not in conflict, because they measure different things. **Per-token latency** at batch 1 gets *worse* under TP (the exposed all-reduce). **Throughput** at server-scale batch gets *dramatically better* under TP (the freed KV cache lets you run the big batch that hides the collective and amortizes everything). The mistake is benchmarking TP at batch 1 and concluding it is slow; that is measuring the one regime where its cost is naked and its benefit is invisible. Measure it under load, with the batch the freed memory unlocked, and the picture inverts.

---

## 9. Case studies and real numbers

Four public results, each cited, that ground everything above.

**The super-linear KV win (vLLM, 2025-02-17).** [Distributed Inference with vLLM](https://vllm.ai/blog/2025-02-17-distributed-inference) reports TP=1 → TP=2 giving 13.9× more KV-cache blocks and 3.9× more token throughput, and states the operating rule verbatim: *"Use pipeline parallelism across nodes and tensor parallelism within nodes when interconnects are slow."* That single sentence is the placement policy for the whole distributed stack — TP inside the NVLink domain, PP across the slow inter-node link — and it falls directly out of the 56× NVLink-vs-PCIe bandwidth gap in section 6. `Source: cited`.

**The fusion that hides comm (vLLM, 2025-08-20).** The [torch.compile Integration with vLLM](https://vllm.ai/blog/2025-08-20-torch-compile) writeup reports an AllReduce+RMSNorm fusion worth up to 15%, plus a SiLU+Quant FP8 fusion up to 8% (measured on Llama-3.1-405B FP8 on 8×MI300). The AllReduce+RMSNorm fusion is exactly the seam from section 3 — the all-reduce at the end of a row-parallel projection immediately followed by the next layer's norm — merged into one kernel so the reduction and the normalization overlap. This is the productionized version of the two-all-reduce layer you built in `nanoserve`, and the forward pointer is [the CUDA-graphs-and-torch.compile post](/blog/machine-learning/inference-engineering/cuda-graphs-and-torch-compile-for-the-decode-loop) where those fusions and the graph capture around them live. `Source: cited`.

**The interconnect that makes TP cheap (InferenceMAX, 2025-10-09).** The vLLM/SemiAnalysis [Blackwell InferenceMAX writeup](https://vllm.ai/blog/2025-10-09-blackwell-inferencemax) lists B200 at 192 GB HBM3e, 8 TB/s of HBM bandwidth, and 1.8 TB/s of NVLink per GPU, with fused AllReduce+RMSNorm+quant in the serving path. The NVLink number is the divisor in section 6's table; the 192 GB HBM is what lets a 140 GB model fit at TP=1 on Blackwell (changing when you *need* TP at all). The generational framing — "up to 4× higher throughput at similar latency vs Hopper" across chat/reasoning/summarization scenarios — is a Pareto claim, not a single peak; cite it as such. `Source: cited`.

**Where TP meets its ceiling (DeepSeek-V3.2 on GB300, 2026-02-13).** The [GB300 DeepSeek writeup](https://vllm.ai/blog/2026-02-13-gb300-deepseek) is a useful reality check on TP degree: TP2 gave a 1.8× prefill speedup but TP4 added only +14% on top — the comm and sync overhead of the extra ranks ate most of the compute benefit. That is the diminishing-returns tail of TP made concrete: past a point, another rank is mostly another barrier participant, not another unit of useful throughput. It is also why the frontier is drifting toward *expert parallelism* for MoE models (the subject of [the MoE inference post](/blog/machine-learning/inference-engineering/moe-inference-routing-expert-parallel-and-the-load-imbalance-problem)), which shards the experts instead of every matmul. `Source: cited`.

How to measure any of this honestly on your own box: fix the clocks (`nvidia-smi -lgc`), warm up until the NCCL buffers and the CUDA graphs are built, time with CUDA events around a steady-state window (not the first step, which pays graph-capture and buffer-allocation cost), and — critically — run an **open-loop** load generator with Poisson arrivals so your batch size is what the server actually sees, not a fixed batch you hand-picked. Report TTFT (time to first token, dominated by prefill where TP is cheap) and TPOT (time per output token, dominated by decode where TP is taxed) *separately*, because TP moves them in opposite directions. A single "tokens/sec at batch 1" number will lie to you about tensor parallelism more than about almost any other technique in this series. The setup lives in [the reproducible-benchmark post](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark).

---

## 10. Stress tests: where TP breaks

Four scenarios that expose the sharp edges. Reason through each; they are the difference between "it worked in the demo" and "it worked under load."

**TP=8 with 8 KV heads (the clean case).** Llama-3.1-70B has exactly 8 key-value heads. At TP=8 each rank gets precisely one KV head, one clean 1/8 of the KV cache, and the GQA grouping (8 query heads sharing 1 KV head per rank) is exactly the model's native group size. Nothing is wasted. This is the case the whole post assumed, and it is why TP=8 is the sweet spot for this architecture — the shard count divides both the query heads (64) and the KV heads (8) evenly.

**TP=16 with 8 KV heads (KV replication).** Push to TP=16 and you have 16 ranks but only 8 KV heads. You cannot give each rank half a KV head — a head is atomic. So the KV heads must be **replicated**: two ranks share each KV head, both storing the *same* K and V. Now the KV cache is duplicated across the pairs, so your per-rank KV memory does not drop to 1/16 — it drops to 1/8 and then gets copied, and the clean super-linear memory win from section 5 partially breaks. Worse, the query heads still shard to 64/16 = 4 per rank, so you have paid for 16 ranks of comm barriers while the KV cache only shards 8 ways. The rule: **do not let TP exceed the number of KV heads** unless you have measured that the extra ranks' compute wins beat the replicated-KV and doubled-barrier cost. For GQA/MQA models with few KV heads (some have 1), this ceiling is low and it is the reason expert parallelism, not more TP, is where the very large deployments go.

**TP over PCIe (comm-bound decode).** Take the section 6 table's PCIe column — 140 µs of bandwidth term per token at TP=8, plus a collective latency several times NVLink's — and put it against section 7's ~63 µs of per-layer decode compute. The comm now *exceeds* the compute, per layer, and decode throughput does not degrade gracefully, it collapses: the GPUs spend more time in the all-reduce barrier than computing. This is the concrete form of the vLLM rule — TP belongs inside the NVLink domain. If your only option is PCIe (a workstation with two consumer cards, no NVLink bridge), TP will disappoint you on decode, and PP or a smaller quantized model that fits on one card is the better call.

**Batch 1 versus the barrier.** At batch 1 the all-reduce is latency-bound (16 KB is far below bandwidth saturation), so the collective cost is nearly fixed while the useful compute is at its smallest — the worst possible comm-to-compute ratio, exactly the exposed-barrier picture from the animation. Every technique that helps (larger batch, fused AllReduce+RMSNorm, CUDA-graph capture of the collective) is really a way of either growing the compute the barrier hides behind or shrinking the barrier's fixed cost. There is no way to make batch-1 TP decode free; there is only making it *less* exposed.

**One slow rank stalls everyone.** An all-reduce is a barrier: it completes only when the *slowest* rank arrives. So one straggler — a GPU with a thermal throttle, a noisy neighbor on a shared PCIe switch, a rank that drew a slightly larger batch in a ragged schedule — stalls all eight ranks, twice per layer, 160 times per token. TP has no slack for heterogeneity; it assumes all ranks march in lockstep. This is why TP wants identical GPUs on a dedicated node, and why a single degraded card in a TP group tanks the whole group's latency rather than just its own. Monitor per-rank step time; a barrier makes the group only as fast as its worst member.

---

## 11. When to reach for tensor parallelism (and when not to)

A decisive recommendation, because "use more GPUs" is not a strategy.

**Reach for TP when the model does not fit on one GPU and you have a fast in-node link.** This is the canonical case: a 70B in bf16 (140 GB) or a 405B in fp8 (~405 GB) on a node of NVLink-connected H100s or B200s. TP inside the NVLink domain is the default, it fits the model, and it frees KV room super-linearly. `tensor_parallel_size` = the number of GPUs in one node, capped at the KV-head count. Use vLLM or SGLang; do not ship your own — `nanoserve`'s TP taught you the mechanism, but the production engines have the fused kernels, sequence parallelism, and CUDA-graph capture that turn the two-all-reduce layer from a 22% decode tax into single digits.

**Reach for PP (not TP) when your fast link stops at the node boundary.** TP's 160 collectives per token are fine on NVLink and fatal on InfiniBand or Ethernet. Cross nodes with pipeline parallelism, which sends once per stage and tolerates the slow link — TP within each node, PP across nodes, exactly the vLLM rule. [The next post](/blog/machine-learning/inference-engineering/pipeline-parallel-and-multi-node-inference) builds it.

**Reach for expert parallelism (not more TP) when the model is a large MoE.** For a mixture-of-experts model, sharding every dense matmul is the wrong cut; sharding the *experts* across ranks (expert parallelism) matches the model's structure and avoids TP's per-layer barriers on the dense path. That is [the MoE post](/blog/machine-learning/inference-engineering/moe-inference-routing-expert-parallel-and-the-load-imbalance-problem).

**Do not reach for TP when the model already fits on one GPU.** If your 8B in bf16 (16 GB) fits on a 24 GB card with room for the cache, TP=2 buys you nothing but two all-reduces per layer of pure decode tax and a straggler risk. Single-GPU is strictly better until you are memory-bound. **Do not reach for TP over PCIe** for latency-sensitive decode. **Do not push TP past the KV-head count** without measuring — replication eats the memory win. And **do not benchmark TP at batch 1** and conclude it is slow; that is measuring the one regime engineered to make its cost visible and its benefit invisible. The out-of-series companion, [tensor, pipeline, and expert parallelism for serving](/blog/machine-learning/model-serving/tensor-pipeline-expert-parallelism-for-serving), lays out the same trade-offs from the production-engine side.

The decision compresses to one table — pick the split that matches where your fast link ends and what the model's structure is.

| Situation | Split to use | Why | Watch out for |
| --- | --- | --- | --- |
| Model fits on one GPU | none (single-GPU) | no barrier, no straggler risk | do not add TP for "safety" |
| Too big for one GPU, one NVLink node | TP within node | fits the model, frees KV super-linearly | cap TP at the KV-head count |
| Spans multiple nodes, slow inter-node link | TP within node + PP across | PP sends once per stage, tolerates the slow link | pipeline bubble needs many in-flight requests |
| Large mixture-of-experts | expert parallelism (+ TP) | shards experts, not every dense matmul | routing load imbalance |
| Only PCIe between GPUs | avoid TP for decode; PP or a fitting quantized model | 160 collectives per token on a slow link tank decode | latency-bound batch-1 collapse |

---

## Key takeaways

- **A 70B in bf16 is 140 GB and does not fit on an 80 GB GPU.** TP splits every matmul across ranks so each holds 1/TP of the weights; PP splits by layer. TP is the within-node default because it talks constantly and needs NVLink.
- **Column-parallel splits the output dim and needs no communication; row-parallel splits the input dim and must all-reduce the partial sums.** Column-then-row is a free handoff plus one all-reduce.
- **A transformer layer costs exactly two all-reduces** — one at the row-parallel O projection, one at the row-parallel down projection — both on the full hidden-dimension tensor. That count is your entire cost model.
- **Per-token comm is `160 × 2(N-1)/N × 16 KB`** for Llama-3.1-70B — a few MB, microseconds on NVLink (1.8 TB/s), a hundred-plus microseconds on PCIe (~32 GB/s). The 56× gap is why TP wants NVLink.
- **TP hurts decode more than prefill because prefill's long GEMM hides the collective and decode's tiny GEMV cannot.** At batch 1 the all-reduce is latency-bound and fully exposed — a ~22% decode tax at TP=8 on NVLink that no kernel tuning removes.
- **TP frees KV room super-linearly** (`TP·C − W`): the vLLM team measured 13.9× more KV blocks and 3.9× more throughput going TP=1 → TP=2, because the fixed weight tax stops being paid on every GPU.
- **Latency and throughput move in opposite directions under TP.** Batch-1 latency gets worse; server-scale throughput gets much better. Benchmark under open-loop load and report TTFT and TPOT separately.
- **Do not let TP exceed the KV-head count** (replication), do not run it over PCIe for decode, and remember one slow rank stalls the whole barrier 160 times per token.

## Further reading

- [What inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is) — the series frame this post plugs into, and the standing honesty rule.
- [The inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) — the capstone that ties TP to the rest of the stack (forward link).
- [Pipeline-parallel and multi-node inference](/blog/machine-learning/inference-engineering/pipeline-parallel-and-multi-node-inference) — the by-layer split, and TP-within-node / PP-across-node.
- [MoE inference: routing, expert parallelism, and the load-imbalance problem](/blog/machine-learning/inference-engineering/moe-inference-routing-expert-parallel-and-the-load-imbalance-problem) — why large MoEs shard experts instead of every matmul.
- [The memory math of the KV cache](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache) — the per-token KV formula this post shards 1/TP.
- [CUDA graphs and torch.compile for the decode loop](/blog/machine-learning/inference-engineering/cuda-graphs-and-torch-compile-for-the-decode-loop) — capturing the collective and fusing AllReduce+RMSNorm.
- [Tensor, pipeline, and expert parallelism for serving](/blog/machine-learning/model-serving/tensor-pipeline-expert-parallelism-for-serving) — the same trade-offs from the production-engine side.
- Shoeybi et al., [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) (2019) — the origin of the column-then-row tensor-parallel design.
- vLLM, [Distributed Inference with vLLM](https://vllm.ai/blog/2025-02-17-distributed-inference) (2025-02-17) — the 13.9×/3.9× super-linear result and the TP-within / PP-across rule.
- vLLM & SemiAnalysis, [Blackwell InferenceMAX](https://vllm.ai/blog/2025-10-09-blackwell-inferencemax) (2025-10-09) — B200 HBM and NVLink specs, fused AllReduce+RMSNorm+quant.

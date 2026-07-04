---
title: "Tensor, Pipeline, and Expert Parallelism for Serving: Sharding a Model Across GPUs for Inference"
date: "2026-07-03"
publishDate: "2026-07-03"
description: "How to split a 70B-to-671B model across GPUs for inference: the memory math that forces sharding, the all-reduce tax of tensor parallelism, the pipeline bubble, MoE all-to-all, and a worked derivation of the TP x PP x DP layout that hits your SLO."
tags:
  [
    "model-serving",
    "inference",
    "tensor-parallelism",
    "pipeline-parallelism",
    "expert-parallelism",
    "llm-serving",
    "vllm",
    "megatron",
    "distributed-inference",
    "gpu",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/tensor-pipeline-expert-parallelism-for-serving-1.webp"
---

The pager goes off at 2 a.m. because someone in product tried to load Llama-3.1-405B onto a single H100 to "see if it fits." It does not fit. It was never going to fit. The weights alone are 810 GB in FP16 and the GPU has 80 GB. The CUDA out-of-memory error fires before the first layer is even allocated. This is not a bug you fix with a smaller batch or a `torch.cuda.empty_cache()`. It is a physics problem: the model is an order of magnitude bigger than the box you tried to put it in, and the only way through is to spread it across many GPUs at once.

Sharding a model across GPUs is old news in *training* — Megatron-LM and DeepSpeed have done it for years. But serving is a different animal. In training you care about aggregate throughput over a multi-week run and you can afford a 30 percent pipeline bubble because the alternative is not training the model at all. In serving you are staring at a p99 latency SLO, a time-to-first-token (TTFT) budget of a few hundred milliseconds, and a per-token cost that finance reviews every quarter. The same three knobs — tensor parallelism, pipeline parallelism, expert parallelism — behave differently when your objective is an interactive request instead of a gradient step. Get the layout wrong and you will either OOM, blow your latency budget with cross-node all-reduces, or leave half your GPUs idle in a pipeline bubble.

This post is about doing it right for inference. We start from the memory math that forces you to shard at all, then take each parallelism axis in turn — what it splits, what it communicates, and how that communication maps onto your latency-versus-throughput trade. The recurring spine of this series is the SLO triangle: latency, throughput, cost, pick your trade. Every sharding decision is a move on that triangle, and by the end you will be able to derive a concrete `tensor_parallel_size x pipeline_parallel_size x data_parallel` layout from a model size, a GPU, and an SLO — and defend it to the on-call engineer who inherits it. Figure 1 is the map: four ways to shard, each splitting a different thing and paying a different communication tax.

![Comparison matrix of tensor, pipeline, expert, and data parallelism across what each splits, its communication primitive, per-token communication volume, interconnect requirement, and its position on the latency-throughput trade](/imgs/blogs/tensor-pipeline-expert-parallelism-for-serving-1.webp)

If you have not yet read [why LLM serving is different](/blog/machine-learning/model-serving/why-llm-serving-is-different) — the KV-cache memory wall and the autoregressive decode bottleneck — read it first; the memory pressure it describes is exactly what pushes single-GPU serving off a cliff and into the multi-GPU regime this post lives in. And if the word "serving" itself is fuzzy, [what is model serving](/blog/machine-learning/model-serving/what-is-model-serving) sets up the latency/throughput/concurrency vocabulary I will lean on throughout.

## 1. Why one GPU isn't enough: the memory math

Before any parallelism strategy, you need to know *why* you are being forced off a single GPU. There are two consumers of high-bandwidth memory (HBM) on an inference GPU, and both scale with model size.

The first is the **model weights**. Parameter count times bytes-per-parameter. In FP16 or BF16 that is 2 bytes each; in FP8 (native on H100/H200) it is 1 byte; in INT4 (GPTQ/AWQ) it is roughly 0.5 bytes plus scales. So:

$$W = P \times b_{\text{param}}$$

where $P$ is parameter count and $b_{\text{param}}$ is bytes per parameter. For a 70B model in FP16, that is ${70 \times 10^9 \times 2 = 140}$ GB. That number alone exceeds any single GPU on the market — an H100 or H200 tops out at 80 GB and 141 GB respectively — so the model does not fit before you have served a single token.

The second consumer is the **KV cache**, the per-token attention state that grows as the conversation grows. For a model with $L$ layers, $H_{kv}$ key/value heads, and head dimension $d_h$, each token stores both a key and a value in every layer:

$$\text{KV}_{\text{token}} = 2 \times L \times H_{kv} \times d_h \times b_{\text{kv}}$$

The KV cache is what makes serving memory-hungry in a way training never is. It scales with *concurrency times context length*, both of which you do not control — your users do. The deep mechanics of that pressure and how to fight it live in [KV cache optimization](/blog/machine-learning/model-serving/kv-cache-optimization); here we just need its size, because the KV cache competes with the weights for the same HBM.

Figure 2 puts real numbers on a 70B model and shows the escape hatch: shard the weights across eight GPUs and the per-GPU footprint drops from an impossible 140 GB to a comfortable 17.5 GB, leaving room for a large KV cache and a healthy batch.

![Before-and-after diagram contrasting a single 80GB H100 that OOMs on a 70B model's 140GB of weights plus 40GB of KV cache against eight H100s in tensor parallel that hold 17.5GB of weights and 5GB of KV cache each with headroom to spare](/imgs/blogs/tensor-pipeline-expert-parallelism-for-serving-2.webp)

#### Worked example: sizing Llama-3-70B on H100

Llama-3-70B has $L = 80$ layers, hidden size $d = 8192$, 64 attention heads, and — thanks to grouped-query attention (GQA) — only $H_{kv} = 8$ key/value heads with head dimension $d_h = 128$. In FP16 ($b_{\text{kv}} = 2$):

- **Weights**: ${70 \times 10^9 \times 2}$ = 140 GB.
- **KV per token**: ${2 \times 80 \times 8 \times 128 \times 2}$ = 327,680 bytes ≈ 320 KB.
- **KV for one 4096-token sequence**: ${320 \text{ KB} \times 4096}$ ≈ 1.31 GB.
- **KV for 32 concurrent sequences**: ${1.31 \times 32}$ ≈ 42 GB.

So the working set for a modest batch of 32 requests at 4k context is ${140 + 42 = 182}$ GB. A single 80 GB H100 is short by 100 GB. Even an H200 at 141 GB cannot hold weights plus a useful KV cache. You *must* spread the weights across GPUs — and once you do, the KV cache spreads with them, because each GPU only holds the slice of every layer it is responsible for.

Here is the memory-fit picture across the model sizes you are likely to serve. "Min GPUs (weights only)" is the floor from weights alone on 80 GB H100s; in practice you need more headroom for KV cache and activations, so treat it as a lower bound.

| Model | Params | FP16 weights | FP8 weights | Min GPUs (FP16, 80GB) | Min GPUs (FP8, 80GB) |
|---|---|---|---|---|---|
| Llama-3-8B | 8B | 16 GB | 8 GB | 1 | 1 |
| Llama-3-70B | 70B | 140 GB | 70 GB | 2 | 1 (tight) |
| GPT-3-class | 175B | 350 GB | 175 GB | 5 | 3 |
| Llama-3.1-405B | 405B | 810 GB | 405 GB | 11 | 6 |
| DeepSeek-V3 | 671B (37B active) | 1342 GB | 671 GB | 17 | 9 |

Two things jump out. First, quantization to FP8 roughly halves the GPU count — this is why FP8 serving on H100 is table stakes for anything above 70B, and why [quantization for LLM serving](/blog/machine-learning/model-serving/quantization-for-llm-serving) is a prerequisite skill for large-scale serving, not an optimization you bolt on later. Second, once the min-GPU count crosses 8, you are off a single node and into multi-node territory, which changes the interconnect you can rely on — and that, as we will see, is the entire reason tensor and pipeline parallelism exist as *separate* knobs.

There is a subtlety hiding in that table. The "min GPUs" column counts *weights only*, but the KV cache and a third consumer — activations plus framework overhead — draw on the same HBM, so the true minimum is always higher. Before we shard, it is worth naming all three consumers precisely, because a layout that budgets only for weights is the classic way to OOM the moment real traffic arrives.

#### The third consumer: activations and framework overhead

Weights and KV cache dominate, but two smaller line items reliably surprise people who size a deployment on a spreadsheet:

- **CUDA context and library buffers.** Every process that touches a GPU pays a fixed CUDA context cost (roughly 300–600 MB) plus whatever cuBLAS, cuDNN, and NCCL allocate for workspaces and communication buffers. NCCL reserves send/receive staging buffers that scale with message size and rank count; on an 8-way TP group these can total 1–2 GB per GPU. This is why serving frameworks default `gpu_memory_utilization` to 0.90 rather than 1.0 — the last 10 percent is not waste, it is the floor the runtime needs to not crash.
- **Transient activations.** During prefill, the intermediate tensors of a forward pass (the ${b \times s \times d}$ hidden states, the ${b \times s \times d_{ff}}$ FFN intermediates, the attention scores) are live at once. For a long prompt this is not negligible: a 4k-token prefill at batch 32 on a 70B model materializes several GB of activations before they are freed. Decode activations are tiny by comparison (one token per request), which is exactly why prefill is the memory-spike phase of a request.

A safe rule is to reserve 25–35 percent of HBM for KV cache, activations, and overhead combined, and fit weights into the remainder — which is where the `weight_headroom = 0.70` in the calculator later in this post comes from.

#### Worked example: sizing GPT-3-175B and Llama-3.1-405B

Run the same arithmetic one tier up. GPT-3-175B has $L = 96$ layers, hidden $d = 12288$, 96 attention heads (no GQA, so $H_{kv} = 96$), head dimension $d_h = 128$. In FP16:

- **Weights**: ${175 \times 10^9 \times 2 = 350}$ GB — nearly 4.4 full H100s just for parameters.
- **KV per token**: ${2 \times 96 \times 96 \times 128 \times 2 = 4{,}718{,}592}$ bytes ≈ 4.5 MB. Note the jump: without GQA the KV cache is roughly 14x larger per token than Llama-3-70B's 320 KB, because all 96 heads store keys and values instead of just 8.
- **KV for one 2048-token sequence**: ${4.5 \text{ MB} \times 2048 \approx 9.2}$ GB — a single conversation's cache is larger than an 8B model's entire weight footprint.

That KV explosion is the clearest argument for grouped-query attention and for aggressive KV-cache management; it also means a 175B model without GQA is bottlenecked on KV memory long before compute. For Llama-3.1-405B ($L = 126$, $d = 16384$, GQA with $H_{kv} = 8$, $d_h = 128$) in FP16, weights are 810 GB — eleven H100s for parameters alone — but GQA holds the KV cache to ${2 \times 126 \times 8 \times 128 \times 2 = 516{,}096}$ bytes ≈ 504 KB/token, so a 4k sequence is only about 2 GB. The lesson generalizes: **weights force the GPU count, but GQA versus multi-head decides whether the KV cache lets you actually use those GPUs at high concurrency.**

Once you shard, both weights and KV divide by the same TP × PP factor, because each GPU owns a fixed slice of every layer. Here is the per-GPU weight footprint as you raise the shard count — the number you actually check against 80 GB:

| Model (FP16 weights) | 1 GPU | TP=2 | TP=4 | TP=8 | TP=8 × PP=2 (16) |
|---|---|---|---|---|---|
| Llama-3-70B (140 GB) | 140 | 70 | 35 | 17.5 | 8.75 |
| GPT-3-175B (350 GB) | 350 | 175 | 87.5 | 43.75 | 21.9 |
| Llama-3.1-405B (810 GB) | 810 | 405 | 202.5 | 101.25 | 50.6 |

Read across the 70B row: TP=8 brings weights to 17.5 GB, leaving about 55 GB per GPU for KV cache and activations — comfortable. Read the 405B row: even TP=8 leaves 101 GB of weights per GPU, which *still* overflows 80 GB, so you are forced to 16 GPUs (TP=8 × PP=2) to bring it to about 51 GB and only then have room for KV. That single cell — 405B needing more than one node — is why the rest of this post spends so long on how to cross the node boundary without wrecking your latency.

#### Worked example: why MoE memory is total, not active

DeepSeek-V3 activates only 37B of its 671B parameters per token, which tempts people to size it like a 37B model. That is a trap. The router can send any token to any expert, so **every** expert's weights must be resident in HBM at all times — you cannot page them in on demand at decode latency. So the memory bill is the full 671B (1342 GB in FP16, 671 GB in FP8), even though the per-token FLOPs are those of a 37B model. This is the defining tension of MoE serving: you pay for 671B of memory to get 37B of compute cost. In FP8 that is still ${671 / (80 \times 0.7) \approx 12}$ GPUs for weights before KV — which is why DeepSeek-V3 layouts start at two 8-GPU nodes and lean on expert parallelism to spread those resident-but-mostly-idle experts across the cluster. The active/total split changes your *throughput* math (cheap per token) but never your *memory* math (you host all of it).

## 2. Tensor parallelism: split every matmul

Tensor parallelism (TP) is the most aggressive way to shard: it splits the *math inside each layer* across GPUs, so all the GPUs work on the same token at the same time. The canonical scheme is from the Megatron-LM paper (Shoeybi et al., 2019), and it is worth understanding exactly because its structure is what dictates the communication cost.

Take the feed-forward block, two matmuls with a nonlinearity between them: $Y = \text{GeLU}(XA)B$. Megatron splits the first weight matrix $A$ **column-wise** across GPUs and the second matrix $B$ **row-wise**. The magic is in the ordering. Because $A$ is split by columns, each GPU computes an independent slice of the hidden activation with *no communication* — GPU 0 computes $\text{GeLU}(XA_1)$, GPU 1 computes $\text{GeLU}(XA_2)$, and they never need to talk. Because $B$ is split by rows to match, each GPU then computes a *partial* output $\text{GeLU}(XA_i)B_i$. To get the true output you sum the partials across GPUs — one **all-reduce**. Figure 3 traces this: the hidden state fans out to both GPUs, each does its column-then-row matmul, and a single all-reduce at the end reconciles the partial sums into an identical result on every GPU.

![Graph showing tensor parallelism inside one MLP: the hidden state broadcasts to GPU0 and GPU1, each computes a column-split then row-split matmul producing partial sums, which merge in a single all-reduce that outputs identical replicas on both GPUs](/imgs/blogs/tensor-pipeline-expert-parallelism-for-serving-3.webp)

The attention block gets the same treatment: the QKV projection is column-split so each GPU owns a subset of attention heads (with 64 heads and TP=8, each GPU owns 8 heads), attention runs independently per head, and the output projection is row-split with a closing all-reduce. So a full transformer layer under TP incurs **exactly two all-reduces** — one after attention, one after the FFN — regardless of how many GPUs you shard across. Figure 4 lays out the sharded layer top to bottom: norms are replicated and cheap, the two projection pairs are the only sharded matmuls, and each pair ends in one collective.

![Stack diagram of a transformer layer sharded by tensor parallelism from LayerNorm through QKV projection, attention, output projection with all-reduce number one, FFN up-projection, FFN down-projection with all-reduce number two, to the layer output, showing exactly two all-reduces per layer](/imgs/blogs/tensor-pipeline-expert-parallelism-for-serving-4.webp)

### The all-reduce tax

The elegance of Megatron's scheme is that it confines *all* cross-GPU traffic to those two all-reduces. But those all-reduces are on the critical path of every single forward pass, and there are two of them per layer. For an 80-layer model that is 160 all-reduces per token. This is the price of tensor parallelism, and it is why TP is picky about its interconnect.

A bandwidth-optimal **ring all-reduce** moves, per GPU, a total of:

$$V_{\text{allreduce}} = 2 \times \frac{N-1}{N} \times M$$

bytes, where $N$ is the number of GPUs and $M$ is the message size (the activation tensor being reduced). The factor of 2 is because a ring all-reduce is a reduce-scatter followed by an all-gather, each moving ${\frac{N-1}{N} M}$ bytes. The message size $M$ is the activation: batch times sequence times hidden, in bytes. For decode (one token per request), $M = b \times 1 \times d \times b_{\text{act}}$; for prefill it is $b \times s_{\text{prompt}} \times d \times b_{\text{act}}$.

#### Worked example: the per-token all-reduce budget on NVLink

Take Llama-3-70B decode, batch $b = 32$, hidden $d = 8192$, FP16 activations, TP=8:

- **Message size per all-reduce**: ${32 \times 8192 \times 2}$ = 512 KB.
- **Ring all-reduce traffic per GPU**: ${2 \times \frac{7}{8} \times 512 \text{ KB}}$ = 896 KB.
- **Per layer** (2 all-reduces): ${2 \times 896}$ = 1.75 MB.
- **Per token, all 80 layers**: ${80 \times 1.75}$ ≈ 140 MB moved per GPU.

On H100 NVLink 4.0 at 900 GB/s aggregate bidirectional bandwidth, the achievable all-reduce bandwidth is lower — call it ~450 GB/s after protocol overhead. So ${140 \text{ MB} / 450 \text{ GB/s}}$ ≈ **0.31 ms** of pure communication per decode step, on top of the compute. At a target TPOT (time-per-output-token) of, say, 20 ms, 0.31 ms is a 1.5 percent tax — acceptable. Now run the *same* all-reduce over 100 Gbps Ethernet (12.5 GB/s): ${140 \text{ MB} / 12.5 \text{ GB/s}}$ ≈ **11 ms per token**, more than half your entire token budget spent waiting on the network. This single calculation is why **tensor parallelism must stay inside a single node.**

The interconnect hierarchy is the whole story:

| Interconnect | Bandwidth (per GPU, aggregate) | TP all-reduce time (140 MB/token) | Verdict for TP |
|---|---|---|---|
| NVLink 4.0 (H100, NVSwitch) | ~900 GB/s | ~0.31 ms | Ideal — use for TP |
| NVLink 3.0 (A100, NVSwitch) | ~600 GB/s | ~0.47 ms | Good for TP |
| InfiniBand NDR (400 Gb/s) | ~50 GB/s | ~2.8 ms | Painful for TP; OK for PP |
| Ethernet 100 GbE | ~12.5 GB/s | ~11 ms | Never for TP |

An 8-GPU HGX/DGX node wires all eight GPUs to a full NVSwitch mesh at 900 GB/s. The moment TP crosses the node boundary — TP=16 spanning two nodes — the all-reduce falls off NVLink onto InfiniBand or Ethernet and the per-token comms cost jumps 10-20x. That is the origin of the field's most durable rule of thumb: **TP degree is capped at the number of GPUs in one node, which is 8 for almost everyone.** You will hear it stated as dogma; now you can derive it.

TP's payoff, in exchange for that comms tax, is **latency**. Splitting each matmul across 8 GPUs means each GPU does 1/8th of the FLOPs and 1/8th of the memory reads, so a single forward pass finishes faster. Since decode is memory-bandwidth-bound (you re-read the whole weight matrix for every token), dividing the weights across 8 GPUs multiplies your effective memory bandwidth by ~8, which directly cuts TPOT. TP is the latency lever.

#### Prefill versus decode: two comms regimes in one deployment

The all-reduce math has a wrinkle that trips up capacity planning: the message size $M$ is proportional to the number of tokens in flight, and prefill and decode differ there by three orders of magnitude. Reuse Llama-3-70B, TP=8, FP16, hidden $d = 8192$:

- **Decode** processes one token per request. At batch $b = 32$, ${M = 32 \times 1 \times 8192 \times 2 = 512}$ KB per all-reduce — the number we used above, giving about 0.31 ms/token of comms.
- **Prefill** processes the whole prompt at once. For a 2048-token prompt at batch 8, ${M = 8 \times 2048 \times 8192 \times 2 = 256}$ MB per all-reduce — 500x larger. Per layer that is two ~256 MB all-reduces; across 80 layers the prefill forward moves tens of GB of collective traffic per GPU.

The consequence is that prefill is far more sensitive to interconnect than decode. On NVLink the prefill all-reduce is amortized against heavy matmul compute (prefill is compute-bound, not memory-bound), so it stays a small fraction of the phase. But push TP across a slow link and prefill latency — your TTFT — collapses first, well before decode does. This asymmetry is one reason disaggregating prefill and decode onto different GPU pools, each with its own parallelism layout, is a live design pattern for large models.

#### Why TP specifically helps the memory-bound decode phase

There is a deeper reason TP is the *latency* lever rather than merely a *fit* lever. Autoregressive decode is memory-bandwidth-bound: to generate one token you stream every weight matrix through the compute units once, and the matmuls are skinny (one token wide), so arithmetic intensity is low and the GPU spends most of its time waiting on HBM reads rather than doing FLOPs. Sharding the weights across 8 GPUs means each GPU reads only 1/8th of the weights per token, so aggregate effective memory bandwidth scales with the TP degree — and since decode latency is set by weight-read time, TPOT drops roughly proportionally until the all-reduce overhead catches up. Prefill, by contrast, is compute-bound, so TP helps it through raw FLOP division instead. One knob, two mechanisms, both pointing at lower latency.

#### Sequence parallelism: reclaiming the replicated activations

Vanilla Megatron TP leaves the LayerNorm, dropout, and residual-add operations *replicated* on every GPU — they sit between the sharded matmuls and each GPU redundantly computes them on the full activation tensor. That is wasted memory (every rank holds the full ${b \times s \times d}$ activation) and wasted work. Sequence parallelism (Korthikanti et al., 2022) closes the gap: it shards those replicated regions along the *sequence* dimension, so each of the 8 GPUs holds only 1/8th of the activation for the norm and dropout too. The two per-layer all-reduces become an all-gather (entering the sharded matmul) plus a reduce-scatter (leaving it) — the *same total bytes* as the all-reduce, so comms cost is unchanged, but activation memory drops by the TP factor. For long-context serving where activation memory is real, TP frameworks enable sequence parallelism alongside TP by default; you rarely set it explicitly, but it is why measured activation memory is lower than a naive "full activation on every rank" estimate.

#### TP=4 versus TP=8: the efficiency crossover in numbers

Put the sublinearity on a table. Model the decode step as compute time that falls as $1/\text{TP}$ plus all-reduce time that grows as ${\frac{TP-1}{TP}}$ of a fixed volume. For Llama-3-70B decode on NVLink, illustrative numbers look like this — your measured values will differ, but the *shape* holds:

| TP | Per-GPU compute (rel.) | All-reduce/token | TPOT (approx) | Speedup vs TP=1 | Per-GPU efficiency |
|---|---|---|---|---|---|
| 1 | 1.00 | 0 | 80 ms | 1.0x | 100% |
| 2 | 0.50 | 0.10 ms | 42 ms | 1.9x | 95% |
| 4 | 0.25 | 0.22 ms | 23 ms | 3.5x | 87% |
| 8 | 0.125 | 0.31 ms | 13 ms | 6.2x | 77% |
| 16 (2 nodes) | 0.0625 | 5.6 ms | 18 ms | 4.4x | 28% |

The story is in the last two columns. Through TP=8 the speedup keeps climbing (6.2x on 8 GPUs) even as per-GPU efficiency erodes from 100 to 77 percent — you are paying more per GPU but still buying latency. At TP=16, where the all-reduce falls off NVLink onto InfiniBand, TPOT *rises* and efficiency craters to 28 percent: you added 8 GPUs and made every request slower. That inversion — not a gentle diminishing return but an actual reversal — is the hard ceiling behind "cap TP at the node." The right move at that boundary is never TP=16; it is TP=8 plus PP or DP to use the second node.

## 3. Pipeline parallelism: split the layers into stages

Pipeline parallelism (PP) shards differently: instead of splitting each layer across GPUs, it assigns *whole layers* to different GPUs. With PP=2 on an 80-layer model, GPU group A holds layers 0-39 and GPU group B holds layers 40-79. A request flows through stage A, then its activations are shipped to stage B, then out.

The communication is completely different from TP. There is no per-layer all-reduce; there is just a **point-to-point (P2P) send** of the activation tensor at each stage boundary. For a PP=2 layout, that is one send per token per boundary — a single ${b \times s \times d}$ tensor handed from stage A to stage B. That is why PP tolerates a slow interconnect: you are moving one activation across the node boundary per stage, not doing 160 collectives. InfiniBand or even good Ethernet is fine.

The catch with pipeline parallelism is the **bubble**. When the pipeline starts, stage B sits idle while stage A processes the first micro-batch; when the pipeline drains, stage A sits idle while stage B finishes the last one. Those idle periods are wasted GPU time. For $P$ stages and $M$ micro-batches flowing through, the fraction of time wasted in the bubble is:

$$\text{bubble fraction} = \frac{P - 1}{M + P - 1}$$

Figure 5 walks the schedule: fill, ramp to a full pipeline, steady state, then drain — with the bubble fraction spelled out for $P = 4$, $M = 8$.

![Timeline of a four-stage pipeline schedule showing the fill phase where microbatch one enters stage zero, the ramp where microbatches spread across stages, the full and steady state with all four stages busy, then the drain where stages go idle, ending with the bubble fraction formula giving 27 percent for four stages and eight microbatches](/imgs/blogs/tensor-pipeline-expert-parallelism-for-serving-5.webp)

#### Worked example: the bubble at four stages

With $P = 4$ stages and $M = 8$ micro-batches:

$$\frac{4 - 1}{8 + 4 - 1} = \frac{3}{11} \approx 0.27$$

27 percent of GPU time is wasted. Push $M$ to 32 micro-batches and the bubble shrinks to ${3/35 \approx 8.6}$ percent. Push it to 128 and it is 2.3 percent. **The bubble is a function of how many things are in flight**, and this is where inference PP diverges sharply from training PP.

### Why the inference bubble is different (and often better)

In training, "micro-batches" are the pieces you split one big global batch into, and the bubble is a fixed tax you pay every step. The GPipe and 1F1B (PipeDream) schedules exist to manage exactly this. In *serving*, the "micro-batches" flowing through the pipeline are your **concurrent requests** — and with continuous batching (the vLLM-style scheduler that admits and retires requests token-by-token, covered in [continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention)), there is a steady stream of them. As long as you have many concurrent requests in flight, the pipeline stays full and the bubble is small. The bubble bites you hardest at *low load* — when only a handful of requests are active, stages sit idle waiting for work.

This flips the usual intuition. Pipeline parallelism for serving is a **throughput** technique that shines under sustained high concurrency and hurts under sparse, latency-sensitive traffic. If your traffic is bursty and low-QPS, a deep pipeline will spend most of its time in bubbles. If you are saturated with hundreds of concurrent requests, the pipeline runs nearly bubble-free and you get near-linear throughput scaling across nodes for the price of one P2P send per boundary. The training-side treatment of the same schedule and its bubble math is in [pipeline parallelism and the bubble](/blog/machine-learning/distributed-training/pipeline-parallelism-and-the-bubble) — the mechanics are identical; only the source of the micro-batches (concurrent requests vs. a split global batch) differs.

One more inference-specific wrinkle: prefill and decode have very different stage timings. A long prefill occupies a stage for many milliseconds while a single decode step is sub-millisecond, so a naive pipeline can develop uneven stage occupancy. Systems that combine PP with [request scheduling and preemption](/blog/machine-learning/model-serving/request-scheduling-and-preemption) or with prefill/decode disaggregation manage this by keeping the stage granularity coarse and the in-flight count high.

#### Worked example: the P2P handoff is cheap

Quantify why PP tolerates a slow link. The only cross-stage traffic is one point-to-point send of the boundary activation. For Llama-3-70B split PP=2, the tensor handed from stage A to stage B during decode at batch $b = 32$ is ${32 \times 1 \times 8192 \times 2 = 512}$ KB — one send, once per token, per boundary. Over 100 GbE at 12.5 GB/s that is ${512 \text{ KB} / 12.5 \text{ GB/s} \approx 0.04}$ ms. Compare that to the 11 ms the *same* model would spend if you ran TP=8 across that Ethernet link: the P2P send is roughly 275x cheaper than the cross-node all-reduce, which is the entire quantitative case for "TP inside, PP across." Even a 2048-token prefill boundary tensor (${8 \times 2048 \times 8192 \times 2 = 256}$ MB at batch 8) crosses 100 GbE in about 20 ms once — a one-time TTFT cost, not a per-token tax, and it overlaps with the next micro-batch's compute in a filled pipeline.

#### Interleaved 1F1B: shrinking the bubble without more micro-batches

The bubble formula ${\frac{P-1}{M+P-1}}$ says you can shrink the bubble by raising $M$, but there is a second lever from the training world that inference inherits: interleaving. Instead of giving each device one contiguous block of layers, an *interleaved* (virtual-pipeline) schedule gives each device several smaller, non-contiguous chunks — device 0 might hold layers 0–9 *and* 40–49. With $v$ chunks per device the bubble shrinks by roughly a factor of $v$, because the fill-and-drain ramp is subdivided into finer steps. The cost is $v$ times as many P2P sends per token, so interleaving trades a little of PP's cheap-comms advantage for a smaller bubble — worth it on fast inter-node fabric (InfiniBand), rarely worth it on Ethernet. For most inference deployments the simpler win is keeping the in-flight request count high with continuous batching, which fills the pipeline the same way a large $M$ does, for free.

#### Balancing the stages: not all layers are equal

A subtlety in laying out PP: the first and last stages carry more than transformer layers. The first stage holds the token embedding table; the last holds the final norm and the language-model head (the ${d \times V}$ output projection over a vocabulary $V$ that can exceed 128k rows). For Llama-3 that head is about 1 GB in FP16 on its own. A naive even split of transformer layers therefore leaves the end stages heavier in memory and, because the lm_head runs a large matmul over the vocabulary, heavier in decode compute too. Production PP configs compensate by giving the embedding and head stages *fewer* transformer layers so wall-clock per stage stays balanced — an unbalanced stage is a slow stage, and in a pipeline the slowest stage sets the steady-state throughput of the whole thing. When you see a PP layout that assigns, say, 18/21/21/20 layers instead of a clean 20/20/20/20, this is why.

## 4. Expert parallelism: spread the MoE experts

Mixture-of-Experts (MoE) models introduce a third axis. In a dense model every token flows through every parameter. In an MoE model each transformer block has many parallel FFN "experts" and a lightweight **router** that sends each token to only the top-$k$ of them. DeepSeek-V3 has 256 routed experts per MoE layer and activates 8 per token, so only 37B of its 671B parameters do work on any given token. The model is huge in memory but cheap in FLOPs.

Expert parallelism (EP) shards the experts across GPUs: GPU 0 holds experts 0-127, GPU 1 holds experts 128-255, and so on. When a batch of tokens arrives, the router decides which experts each token needs, and then two **all-to-all** collectives move the data: a dispatch that scatters each token to the GPU(s) holding its chosen experts, and a combine that gathers the expert outputs back. Figure 7 shows the flow — router, all-to-all dispatch, per-GPU expert FFNs, all-to-all combine.

![Graph of expert parallelism showing a token batch entering a router that selects the top eight of two hundred fifty six experts, an all-to-all dispatch that routes tokens to GPU0 holding experts zero to one hundred twenty seven and GPU1 holding experts one hundred twenty eight to two hundred fifty five, then an all-to-all combine that produces the output activating thirty seven billion of six hundred seventy one billion parameters](/imgs/blogs/tensor-pipeline-expert-parallelism-for-serving-7.webp)

The all-to-all is a different beast from the all-reduce. Its cost depends on the routing distribution: if all tokens happen to pick experts on one GPU, that GPU is a hotspot and the all-to-all serializes behind it. Load imbalance is the central operational headache of EP serving, and MoE inference systems spend real effort on expert placement, replication of hot experts, and capacity factors. That depth belongs to the MoE-serving sibling post ([serving MoE models at scale](/blog/machine-learning/model-serving/serving-moe-models-at-scale)); here the point is narrower: **EP replaces the per-layer all-reduce with two per-layer all-to-alls**, and because those all-to-alls also move data proportional to activation size but with a routing-dependent, potentially skewed pattern, EP wants a fast interconnect (NVLink intra-node, InfiniBand across nodes) just like TP does. The upside is that MoE's sparsity means each GPU only runs the experts it holds, so you get enormous total capacity (671B params) at the compute cost of a much smaller dense model (37B active).

#### The all-to-all bill and the capacity factor

The all-to-all's cost is worth a number. In EP each of the ${b \times s}$ tokens is sent to its top-$k$ experts, so the dispatch collective moves, in aggregate across the EP group, about ${k \times (b \times s) \times d \times b_{\text{act}}}$ bytes, and the combine moves the same back. For a DeepSeek-V3 MoE layer with $k = 8$, hidden $d = 7168$, FP8 activations, processing a decode batch of $b = 64$ tokens: dispatch moves ${8 \times 64 \times 7168 \times 1 \approx 3.7}$ MB across the group per layer, and combine another 3.7 MB. That is comparable to a TP all-reduce in volume, but its *pattern* is the problem — where a ring all-reduce touches every peer evenly, the all-to-all's traffic is dictated by the router, so a skewed batch that sends most tokens to a handful of hot experts serializes behind the GPUs that hold them.

Systems bound the damage with a **capacity factor**. Each expert is given a fixed token budget per batch — ${\text{capacity} = f_c \times \frac{k \cdot b \cdot s}{E}}$ for $E$ experts and capacity factor $f_c$ (typically 1.0–1.5) — and tokens beyond that budget are either dropped (their contribution zeroed) or spilled to a slower path. A capacity factor of 1.0 is perfectly load-balanced and cheapest; a higher factor tolerates skew at the cost of more compute and comms. Inference systems often run drop-less variants to avoid quality loss, which makes hot-expert replication and smart placement — not token-dropping — the primary tools for taming imbalance. That machinery is the whole subject of the MoE-serving sibling post; the takeaway here is just that EP's comms volume is TP-like but its *tail* is routing-dependent, which is why it wants both fast wires and careful placement.

For the rest of this post I will treat EP as a specialized axis you reach for only when serving an MoE model, and focus the layout math on the TP/PP/DP combination that covers dense models and composes with EP when needed.

## 5. Combining them: TP inside, PP across, DP outside

No real deployment uses one axis alone. The standard large-model serving layout is a **hierarchy that matches each collective to the interconnect it can afford**:

- **Tensor parallelism inside a node**, across the NVLink-connected GPUs, because TP's per-layer all-reduce needs NVLink bandwidth.
- **Pipeline parallelism across nodes**, because PP only sends one activation per stage boundary and tolerates InfiniBand/Ethernet.
- **Data parallelism outside**, replicating the whole TP x PP unit N times behind a load balancer for throughput, because DP replicas are independent and communicate nothing during inference.

Figure 6 shows the canonical 70B layout: an 8-GPU deployment as PP=2 x TP=4. Each node's four GPUs run tensor-parallel over NVLink (holding half the layers); the two nodes are pipeline stages connected by a single P2P activation handoff over Ethernet. All-reduces never leave NVLink; only activations cross the slow link.

![Grid diagram of the classic serving layout showing eight GPUs arranged as two pipeline stages by four tensor-parallel ranks, where node A holds layers zero to thirty nine across four NVLink-connected GPUs and node B holds layers forty to seventy nine, with tensor parallel all-reduces staying inside each node and pipeline point-to-point sends crossing between nodes](/imgs/blogs/tensor-pipeline-expert-parallelism-for-serving-6.webp)

The total GPU count is the product: $\text{GPUs} = TP \times PP \times DP$. A layout of TP=4, PP=2, DP=3 uses ${4 \times 2 \times 3 = 24}$ GPUs and serves three independent replicas of a model that is itself sharded 8 ways. The comparison table below summarizes the division of labor:

| Axis | Splits | Comms primitive | Frequency | Interconnect | Buys you | Costs you |
|---|---|---|---|---|---|---|
| Tensor (TP) | Each matmul | all-reduce | 2 / layer | NVLink (intra-node) | Lower latency | Per-layer comms; capped at 8 |
| Pipeline (PP) | Layers into stages | P2P send/recv | 1 / boundary | IB / Ethernet OK | Throughput, cross-node scale | Pipeline bubble at low load |
| Expert (EP) | MoE experts | all-to-all | 2 / MoE layer | NVLink / IB | MoE capacity, sparse FLOPs | Load imbalance, routing skew |
| Data (DP) | Whole replicas | none | 0 | Any | Linear throughput | N x the memory |

The mental model to carry: **TP and EP are bandwidth-hungry and stay close; PP is bandwidth-frugal and reaches across nodes; DP is embarrassingly parallel and scales horizontally.** You compose them so the expensive collectives ride the fast wires.

It helps to see the same four axes as *scaling laws* rather than a feature list — how each one bends per-GPU memory and per-token communication as you turn its knob:

| Axis (degree $n$) | Per-GPU weight memory | Per-GPU KV memory | Per-token comms | Adds latency? |
|---|---|---|---|---|
| Tensor (TP=$n$) | $W/n$ | $\text{KV}/n$ | 2 all-reduces/layer, ${\propto \frac{n-1}{n}}$ | Lowers it |
| Pipeline (PP=$n$) | $W/n$ | $\text{KV}/n$ | 1 P2P send/boundary, size-constant | Adds a little |
| Expert (EP=$n$) | experts$/n$ | unchanged | 2 all-to-alls/MoE layer | Adds a little |
| Data (DP=$n$) | $W$ (full copy) | $\text{KV}$ (full) | none | None |

The columns encode the trade in one glance: TP and PP both cut per-GPU memory by their degree, but TP pays per-layer collectives to do it while PP pays a single boundary send; EP cuts only the expert memory (attention and shared layers still need TP or replication); DP cuts nothing and pays nothing but multiplies the whole bill. When you compose them, per-GPU weight memory divides by ${TP \times PP}$ (or by EP for the expert weights), while the memory *bill* for the fleet multiplies by DP. Keep that arithmetic in your head and no layout will surprise you at the invoice.

#### Composing all four: a DeepSeek-V3 layout in full

To see the hierarchy fully composed, take the DeepSeek-V3 layout the case studies return to: two 8-GPU nodes as one replica, then replicated for QPS. Inside each node, attention and the shared dense layers run tensor-parallel over NVLink (TP=8). The 256 routed experts are spread by expert parallelism across all 16 GPUs of the replica, so each GPU hosts 16 experts and the router's all-to-all rides NVLink within a node and InfiniBand between the two. Pipeline parallelism splits the 61 layers across the two nodes so neither has to hold every layer's experts at once. Then the whole 16-GPU unit is a data-parallel replica: stamp out three of them behind a load balancer for 48 GPUs total. Four axes, one deployment — each collective mapped to the wire it can afford: TP all-reduce and EP all-to-all on the fast fabric, PP's activation handoff on the slower inter-node link, and DP replicas sharing nothing. This is the general shape every 100B-plus serving stack converges on.

## 6. The latency vs throughput trade of each axis

Every axis moves you on the SLO triangle in a characteristic direction, and confusing them is the most common layout mistake. Figure 8 contrasts the two levers directly.

![Before-and-after diagram contrasting raising the tensor-parallel degree as a latency lever that cuts time-to-first-token but goes sublinear past eight GPUs when communication exceeds compute, against raising the pipeline-parallel degree as a throughput lever that tolerates ten gigabit Ethernet with one point-to-point send per boundary and scales near-linearly at the cost of a fixed pipeline bubble](/imgs/blogs/tensor-pipeline-expert-parallelism-for-serving-8.webp)

**Tensor parallelism is a latency lever.** Because it divides the work of a single forward pass across GPUs, it directly reduces TTFT and TPOT — up to the point where the all-reduce cost overtakes the compute savings. That crossover is why TP scales *sublinearly*: going from TP=1 to TP=2 might cut latency by 1.8x, TP=2 to TP=4 by another 1.7x, but TP=4 to TP=8 only by ~1.4x, because the all-reduce volume grows while the per-GPU compute shrinks. Past 8 GPUs (off NVLink), the curve inverts and latency gets *worse*. TP is how you meet a tight latency SLO on a model that would be slow on fewer GPUs — and its ceiling is the node.

**Pipeline parallelism is a throughput lever.** It does not make any single request faster; a request still traverses all $P$ stages, and the stage-to-stage handoff *adds* a little latency. What PP gives you is the ability to keep more GPUs busy on more concurrent requests across node boundaries, scaling aggregate tokens/second near-linearly with stage count as long as you keep the pipeline full. PP is how you scale a deployment beyond one node without paying the cross-node all-reduce tax that TP would.

**Data parallelism is pure throughput** with zero latency cost and zero inference-time communication — but it multiplies your memory bill, because every replica holds the full (sharded) model again. You reach for DP when you have latency headroom and just need to serve more QPS.

The practical decision rule falls out of this:

- **Latency-bound and the model fits in a node?** Maximize TP up to the node size (usually 8), then add DP replicas for throughput.
- **Model too big for one node?** Fill TP to 8 within each node, then add PP across nodes to hold the rest of the layers, then DP for throughput.
- **Throughput-bound with slack latency?** Prefer DP replicas of the smallest layout that fits, and keep TP modest — a smaller TP degree means less all-reduce overhead and better per-GPU efficiency.

#### A throughput-vs-latency worked trade

Make the trade concrete with a fixed budget of 16 H100s and a 70B model, and ask what three different layouts buy. Assume measured throughput per replica scales roughly as the earlier efficiency table implies:

| Layout on 16 GPUs | Replicas | Per-request TPOT | Aggregate throughput | Best for |
|---|---|---|---|---|
| TP=8 × DP=2 | 2 | ~13 ms (fast) | moderate | Tight latency SLO, moderate QPS |
| TP=4 × DP=4 | 4 | ~23 ms | high | Balanced; often the sweet spot |
| TP=2 × DP=8 | 8 | ~42 ms | highest | Throughput-bound, loose latency |

Same 16 GPUs, three points on the SLO triangle. TP=8 × DP=2 gives the lowest per-request latency but the fewest replicas, so it saturates at lower QPS; TP=2 × DP=8 gives eight independent replicas and the highest aggregate throughput but each request is roughly 3x slower. The middle row is where many production dense-model deployments land, because per-GPU efficiency is still high (87 percent) while latency stays interactive. The decision is not "which axis" in isolation — it is *where on this table your SLO sits*, and that is a business question (is a user waiting on this token, or is it a batch job?) as much as an engineering one.

## 7. Picking degrees for an SLO: a worked derivation

Now the payoff. Given a model, a GPU, and an SLO, here is how to derive the layout. The procedure:

1. **Compute the weight memory** and divide by usable per-GPU memory (reserve ~30 percent for KV cache and activations) to get the minimum GPU count $G_{\min}$.
2. **Set TP** to the smallest value that makes the per-GPU shard fit *and* meets latency, capped at the node size (8).
3. **Set PP** to cover any remaining layers when the model does not fit in one TP node: ${PP = \lceil G_{\min} / TP \rceil}$.
4. **Set DP** to hit your throughput target: ${DP = \lceil \text{target QPS} / \text{per-replica QPS} \rceil}$.

#### Worked example: DeepSeek-V3 (671B MoE) under an interactive SLO

Serve DeepSeek-V3 in FP8 (671 GB of weights, 37B active) on H100 80 GB nodes, target TTFT < 500 ms and TPOT < 40 ms, 200 QPS aggregate.

- **Weight fit**: ${671 \text{ GB} / (80 \times 0.7)}$ ≈ 12 GPUs minimum for weights, before KV cache. Round up to a clean layout.
- **Because it is MoE**, the natural axis is EP for the expert FFNs, combined with TP for the attention and shared layers. A common DeepSeek-V3 serving layout is 2 nodes of 8 H100 = 16 GPUs, running attention TP within each node and experts distributed by EP across all 16.
- **PP** across the 2 nodes covers the layer split so no single node holds all 61 layers' worth of experts.
- **Per-replica capacity**: suppose one 16-GPU replica sustains ~70 QPS at the SLO. For 200 QPS you need ${\lceil 200 / 70 \rceil = 3}$ replicas → DP=3.
- **Total**: ${16 \times 3 = 48}$ H100s.

The exact numbers depend on your measured per-replica throughput — the point is the *procedure*: memory forces the minimum, TP/EP degree is capped by the node and the interconnect, PP covers overflow layers across nodes, DP multiplies for QPS. Encode that procedure in a calculator so you are not doing it on a whiteboard at 2 a.m.:

```python
import math

def plan_layout(
    params_b: float,          # billions of parameters (total)
    bytes_per_param: float,   # 2.0 FP16, 1.0 FP8, 0.5 INT4
    gpu_mem_gb: float,        # e.g. 80 for H100
    gpus_per_node: int,       # NVLink domain size, e.g. 8
    n_layers: int,
    d_model: int,
    kv_heads: int,
    head_dim: int,
    ctx_len: int,             # target context length
    max_concurrency: int,     # concurrent sequences per replica
    kv_bytes: float = 2.0,    # KV dtype bytes
    weight_headroom: float = 0.70,  # fraction of HBM usable for weights
):
    # 1. Weight memory and the minimum GPU floor.
    weight_gb = params_b * 1e9 * bytes_per_param / 1e9
    usable_per_gpu = gpu_mem_gb * weight_headroom
    g_min = math.ceil(weight_gb / usable_per_gpu)

    # 2. KV cache budget per token, per sequence, and for the batch.
    kv_per_token = 2 * n_layers * kv_heads * head_dim * kv_bytes
    kv_per_seq_gb = kv_per_token * ctx_len / 1e9
    kv_total_gb = kv_per_seq_gb * max_concurrency

    # 3. TP: fill the node, capped at the NVLink domain.
    tp = min(gpus_per_node, g_min)

    # 4. PP: cover remaining layers across nodes.
    pp = math.ceil(g_min / tp)

    # 5. Per-GPU footprint check with weights + KV both sharded by TP*PP.
    shard = tp * pp
    per_gpu_weight = weight_gb / shard
    per_gpu_kv = kv_total_gb / shard
    per_gpu_total = per_gpu_weight + per_gpu_kv
    fits = per_gpu_total < gpu_mem_gb

    return {
        "weight_gb": round(weight_gb, 1),
        "kv_total_gb": round(kv_total_gb, 1),
        "g_min": g_min,
        "tensor_parallel_size": tp,
        "pipeline_parallel_size": pp,
        "gpus_per_replica": shard,
        "per_gpu_gb": round(per_gpu_total, 1),
        "fits": fits,
    }

# Llama-3-70B, FP16, 32 concurrent 4k sequences on H100 nodes.
print(plan_layout(
    params_b=70, bytes_per_param=2.0, gpu_mem_gb=80, gpus_per_node=8,
    n_layers=80, d_model=8192, kv_heads=8, head_dim=128,
    ctx_len=4096, max_concurrency=32,
))
# -> tensor_parallel_size=2, pipeline_parallel_size=1, per_gpu_gb ~91 -> fits=False
#    KV pressure pushes you to TP=4 or FP8; bump the degree until fits=True.
```

The calculator deliberately reports `fits=False` when weights-plus-KV overflow even though weights alone would fit at a low TP degree — that is the KV cache biting, and it is the single most common reason a layout that "should fit" OOMs in production under load. Bump TP (or quantize) until `fits=True` with real headroom.

#### Worked example: Llama-3.1-405B in FP8 under a chat SLO

Walk the four steps by hand for 405B, FP8 (405 GB weights), on H100 80 GB nodes of 8, target TTFT < 600 ms, TPOT < 50 ms, 120 QPS.

1. **Weight floor.** ${405 / (80 \times 0.7) = 405 / 56 \approx 8}$ GPUs for weights — but that leaves nothing for KV, so treat 8 as a hard floor and expect to round up.
2. **TP.** The model exceeds one GPU by a wide margin and latency is tight, so fill the node: TP=8. Per-GPU weights become ${405 / 8 \approx 51}$ GB, leaving about 29 GB for KV and activations — usable but not roomy at long context.
3. **PP.** Weights fit in one 8-GPU node at TP=8 (51 GB < 80 GB), so PP=1 is *possible*. But if you serve 16k-plus contexts at high concurrency, KV pressure pushes you to PP=2 across two nodes to halve per-GPU weights to about 25 GB and free HBM for cache. This is the judgment call: PP here is bought not for layer-fit but for KV headroom.
4. **DP.** Suppose a measured single replica (TP=8, PP=2, 16 GPUs) sustains about 40 QPS at the SLO. For 120 QPS, ${\lceil 120 / 40 \rceil = 3}$ replicas → DP=3. Total: ${16 \times 3 = 48}$ H100s.

Notice PP appeared for a memory reason, not a layer-count reason — a case the naive "PP only when layers overflow" heuristic misses. The calculator's `fits` flag catches exactly this by checking weights *plus* KV against HBM, not weights alone.

The 70B calculator call above returns `tensor_parallel_size=2` with `fits=False` for a reason worth internalizing: 70B weights are 140 GB, which at TP=2 is 70 GB/GPU, and adding about 21 GB of KV for 32 concurrent 4k sequences overflows 80 GB. The fix the comment points to — bump to TP=4 (35 GB weights plus KV fits) or quantize to FP8 — is the same move you make in production when a "should fit" layout OOMs under load. Always let the KV term, not the weight term, have the final word on the degree.

Before you commit a derived layout to a fleet, validate it against three checks the calculator does not cover:

1. **KV headroom at peak, not average.** Re-run the fit with your *p99* concurrency and *max* context, not the mean. The layout must survive the worst hour, because that is when it OOMs.
2. **Comms fraction under the TPOT budget.** Estimate the per-token all-reduce time (a helper for this appears in the next section) and confirm it is a single-digit percentage of TPOT. If it is not, the layout is comms-bound before you have served a token.
3. **Divisibility.** TP must divide the attention head count (and the KV head count under GQA); PP must divide the layer count sensibly with room for the embedding-and-head imbalance. A layout that does not divide cleanly is rejected by the framework at launch.

## 8. The practice: vLLM, TGI, and multi-node with Ray

The mechanics above map onto two flags in every modern serving framework. Here is how to actually launch these layouts.

### vLLM: tensor and pipeline parallel

vLLM exposes both axes directly. The offline `LLM` API:

```python
from vllm import LLM, SamplingParams

# 8 GPUs in one node as pure tensor parallel (the 70B latency-first layout).
llm = LLM(
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    tensor_parallel_size=8,          # all-reduce over NVLink
    pipeline_parallel_size=1,
    dtype="float16",
    gpu_memory_utilization=0.90,     # leave 10% for CUDA/NCCL buffers
    max_model_len=8192,
)

out = llm.generate(
    ["Explain tensor parallelism in one sentence."],
    SamplingParams(temperature=0.7, max_tokens=128),
)
print(out[0].outputs[0].text)
```

The online server, which is what you actually deploy behind a gateway, takes the same knobs as CLI flags:

```bash
# TP=8 within the node, PP=2 across two nodes -> a 16-GPU deployment
# for a model too big for a single 8-GPU node.
vllm serve meta-llama/Meta-Llama-3.1-405B-Instruct-FP8 \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 16384 \
    --port 8000
```

vLLM uses Megatron-style tensor parallelism internally and NCCL for the collectives; the `tensor_parallel_size` must divide the number of attention heads (and key/value heads for GQA), which is why you see 8 and not, say, 6.

### TGI: sharding with `--num-shard`

HuggingFace's text-generation-inference uses a single flag, `--num-shard`, for tensor-parallel sharding, and runs NCCL for the collectives:

```bash
# TGI shards a 70B model across 4 GPUs (tensor parallel).
text-generation-launcher \
    --model-id meta-llama/Meta-Llama-3-70B-Instruct \
    --num-shard 4 \
    --dtype float16 \
    --max-batch-prefill-tokens 4096 \
    --max-total-tokens 8192 \
    --port 8080
```

TGI's `--num-shard` is tensor parallelism only; for models that exceed a node you combine TGI shards with an external router doing data-parallel load balancing, or reach for a framework with native pipeline parallelism. The deeper TGI internals — flash attention, continuous batching, token streaming — are covered in [the TGI deep dive](/blog/machine-learning/model-serving/text-generation-inference-deep-dive).

For a model that is tight even when sharded, TGI combines `--num-shard` with a quantization flag so the per-shard weights shrink further:

```bash
# 70B sharded 4 ways AND quantized to FP8: ~17.5 GB/shard becomes ~8.75 GB,
# freeing HBM for a much larger KV cache and higher concurrency.
text-generation-launcher \
    --model-id meta-llama/Meta-Llama-3-70B-Instruct \
    --num-shard 4 \
    --quantize fp8 \
    --max-batch-prefill-tokens 8192 \
    --max-total-tokens 16384 \
    --port 8080
```

The interaction between the shard count and the quantization dtype is the whole per-GPU memory story in one command: `--num-shard` divides the weights by 4, `--quantize` halves them again, and the two together decide how much HBM is left for the KV cache that actually sets your concurrency ceiling.

### Multi-node vLLM with Ray

When a layout crosses node boundaries (any `pipeline_parallel_size > 1`, or a `tensor_parallel_size` larger than one node — which you should avoid), vLLM uses Ray to place workers across the cluster. You start a Ray cluster, then launch vLLM against it:

```bash
# --- On the head node ---
ray start --head --port=6379 --num-gpus=8

# --- On each worker node (point at the head's IP) ---
ray start --address='HEAD_NODE_IP:6379' --num-gpus=8

# --- Launch vLLM once, on the head; it schedules workers via Ray ---
# 16 GPUs total = TP=8 (within each node) x PP=2 (across the two nodes).
vllm serve meta-llama/Meta-Llama-3.1-405B-Instruct-FP8 \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --distributed-executor-backend ray \
    --gpu-memory-utilization 0.92 \
    --max-model-len 32768
```

The critical detail: vLLM (and Ray) will place each tensor-parallel *group* on GPUs connected by the fastest available link. With `pipeline_parallel_size=2`, it puts the two 8-GPU TP groups on separate nodes and pipelines between them over the inter-node network. If you instead set `tensor_parallel_size=16`, vLLM would try to run a 16-way all-reduce spanning both nodes over InfiniBand every layer — the exact anti-pattern the interconnect table warned about. **Always keep TP within the NVLink domain and use PP to cross nodes.** This is the operational heart of [multi-node LLM serving for 100B-plus models](/blog/machine-learning/model-serving/multi-node-llm-serving-100b-plus), which takes the cluster-level orchestration much further.

### mp versus ray: which distributed backend

vLLM offers two backends for the collectives. The default `mp` (multiprocessing) backend spawns one worker process per GPU on a *single* node and is the right choice for any single-node TP layout — lower overhead, no cluster to stand up. The `ray` backend is required the moment a layout spans nodes (`pipeline_parallel_size > 1` across machines, or a placement group Ray must schedule). A common operational mistake is reaching for Ray on a single 8-GPU box, paying its scheduling overhead for nothing; the rule is `mp` within a node, `ray` across nodes.

A handful of environment variables materially affect these collectives and are worth setting explicitly rather than trusting defaults:

```bash
# Pin NCCL to the fast fabric and fail loudly if it silently falls back to TCP.
export NCCL_IB_DISABLE=0            # allow InfiniBand for cross-node PP
export NCCL_P2P_LEVEL=NVL           # keep intra-node TP peer traffic on NVLink
export NCCL_DEBUG=WARN              # surface fallbacks (e.g. NVLink -> PCIe) early
export VLLM_HOST_IP=10.0.0.1        # the head node IP the workers dial back to
```

The `NCCL_DEBUG=WARN` line earns its keep: a mislabeled topology that quietly runs your "NVLink" all-reduce over PCIe (a roughly 5x bandwidth cliff) is one of the most common and hardest-to-spot causes of a TP layout missing its latency SLO. The warning is your canary.

### A comms-time estimator to sanity-check a layout

Before you rent 48 GPUs, estimate the per-token TP all-reduce time and confirm it is a small fraction of your TPOT budget. This closes the loop between the interconnect table and a concrete layout:

```python
def tp_allreduce_ms_per_token(
    n_layers: int,
    d_model: int,
    batch: int,
    tp: int,
    link_gbps_per_gpu: float,   # achievable all-reduce BW, e.g. ~450 for NVLink4
    act_bytes: float = 2.0,     # FP16 activations
):
    # One all-reduce message = the activation being reduced (decode: 1 token).
    msg_bytes = batch * 1 * d_model * act_bytes
    # Ring all-reduce moves 2*(N-1)/N * M per GPU; two collectives per layer.
    per_layer = 2 * (2 * (tp - 1) / tp) * msg_bytes
    total_bytes = per_layer * n_layers
    return total_bytes / (link_gbps_per_gpu * 1e9) * 1e3  # ms

# Llama-3-70B decode, batch 32, on NVLink (~450 GB/s) vs 100GbE (~12.5 GB/s).
for bw, name in [(450, "NVLink4"), (12.5, "100GbE")]:
    ms = tp_allreduce_ms_per_token(80, 8192, 32, 8, bw)
    print(f"{name:8s} TP=8 all-reduce: {ms:.2f} ms/token")
# NVLink4  TP=8 all-reduce: 0.31 ms/token   -> ~1.5% of a 20 ms TPOT budget
# 100GbE   TP=8 all-reduce: 11.2 ms/token   -> over half the budget: never do this
```

If the estimate is more than about 5–10 percent of your TPOT budget, the layout is comms-bound and you should either drop the TP degree, move TP onto a faster link, or convert the cross-node portion to pipeline parallelism. This is the same 0.31 ms versus 11 ms result from the mechanics section, now as a function you can run against any candidate layout.

One more practice note that interacts with TP degree: CUDA graphs. At high TP the per-layer kernel-launch overhead (many small kernels plus the NCCL call) becomes a real fraction of a sub-millisecond decode step. vLLM captures the decode path into a CUDA graph by default so those launches replay as one graph launch; passing `--enforce-eager` disables that and you will see decode latency regress, more so at TP=8 than TP=2 because there are more launches to amortize. Leave CUDA graphs on in production; disable them only to debug.

### Benchmarking across TP degrees

Never trust the theory over the measurement. Here is a minimal harness to sweep TP degree and record latency and throughput, so you can find your own crossover point:

```python
import time
import statistics
from vllm import LLM, SamplingParams

def bench_tp(model: str, tp: int, prompts: list[str], out_tokens: int = 256):
    llm = LLM(model=model, tensor_parallel_size=tp,
              gpu_memory_utilization=0.90, dtype="float16")
    sp = SamplingParams(temperature=0.0, max_tokens=out_tokens, ignore_eos=True)

    # Warmup so CUDA graphs and NCCL buffers are allocated.
    llm.generate(prompts[:2], sp)

    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp)
    dt = time.perf_counter() - t0

    total_out = sum(len(o.outputs[0].token_ids) for o in outs)
    ttfts = [o.metrics.first_token_time - o.metrics.arrival_time
             for o in outs if o.metrics.first_token_time]
    return {
        "tp": tp,
        "throughput_tok_s": round(total_out / dt, 1),
        "throughput_per_gpu": round(total_out / dt / tp, 1),
        "p50_ttft_ms": round(statistics.median(ttfts) * 1000, 1),
        "p99_ttft_ms": round(sorted(ttfts)[int(0.99 * len(ttfts))] * 1000, 1),
    }

prompts = ["Summarize the theory of tensor parallelism."] * 256
for tp in (2, 4, 8):
    print(bench_tp("meta-llama/Meta-Llama-3-70B-Instruct", tp, prompts))
# Expect: aggregate throughput rises with tp, but throughput_per_gpu FALLS
# (all-reduce overhead), while p99_ttft drops. That gap is the TP tax.
```

The line to watch is `throughput_per_gpu`. It *falls* as you raise TP, because you are spending an increasing fraction of each GPU's time in all-reduce instead of compute. Aggregate throughput and latency both improve, but efficiency drops — which is exactly why you do not crank TP to the max unless latency forces you to.

## 9. Case studies and benchmarks

### Megatron-LM: where the sharding scheme comes from

Megatron-LM (Shoeybi et al., NVIDIA, 2019) introduced the column-then-row tensor-parallel decomposition that every serving framework now uses. Its central result for our purposes: by structuring the MLP and attention so that only *two* all-reduces per layer are needed, TP scales efficiently *within* the high-bandwidth NVLink domain but degrades across slower links. NVIDIA's own scaling studies showed TP efficiency dropping sharply once the tensor-parallel group spanned nodes — the empirical basis for the "TP ≤ node size" rule. Later Megatron work formalized the 3D layout (TP x PP x DP) that we derive here; the training-side treatment with full derivations is in [tensor parallelism and Megatron](/blog/machine-learning/distributed-training/tensor-parallelism-megatron).

The paper's own scaling study is the empirical anchor for everything above: on a multi-hundred-GPU cluster NVIDIA showed near-linear scaling for TP *within* a DGX node's NVLink domain but a sharp efficiency drop once the tensor-parallel group had to reach across the slower inter-node links — which is precisely why the modern recipe pairs an intra-node TP degree with inter-node pipeline and data parallelism rather than one giant TP group. Serving inherits the structure wholesale; only the workload — latency-sensitive single requests instead of a throughput-maximizing training step — changes which point on the curve you target.

### vLLM tensor-parallel scaling

The vLLM project (Kwon et al., SOSP 2023) reports tensor-parallel scaling on multi-GPU nodes that matches the sublinear pattern the all-reduce math predicts. In published and community benchmarks, Llama-70B-class models on 4x and 8x A100/H100 show aggregate throughput rising with TP degree while per-GPU throughput declines — the all-reduce tax made visible. The framework's guidance is explicit: use tensor parallelism up to the number of GPUs in a node, and only add pipeline parallelism when the model exceeds a single node. vLLM's PagedAttention and continuous batching (the [PagedAttention deep dive](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention)) are what keep the pipeline full enough that PP's bubble stays small in practice.

Concretely, the pattern community benchmarks report for a 70B-class model on 8x H100 is aggregate decode throughput rising sublinearly from TP=1 through TP=8 while per-GPU throughput falls monotonically — the exact fingerprint of the all-reduce tax. Prefill (compute-bound) scales closer to linear with TP than decode (memory-bound) does, which is why TTFT often improves more cleanly with added TP than TPOT does. The operational guidance vLLM ships is a one-liner that now has a full derivation behind it: set `tensor_parallel_size` to your per-node GPU count, add `pipeline_parallel_size` only to cross nodes, and never set TP larger than one node.

### TGI sharding in production

HuggingFace's text-generation-inference popularized single-flag tensor sharding (`--num-shard`) with custom flash-attention and paged KV kernels. TGI's design choice to expose *only* tensor parallelism (and lean on external replication for scale-out) reflects the same principle from the other direction: within a node, TP is the right tool, and TGI optimizes that path hard rather than exposing a pipeline knob most single-node deployments would misuse.

### DeepSpeed-Inference: kernel injection meets tensor parallelism

DeepSpeed-Inference (Aminabadi et al., 2022) is the other lineage worth knowing, because it attacks the same problem from the kernel side. Where vLLM and TGI lean on off-the-shelf tensor-parallel matmuls, DeepSpeed-Inference *injects* fused, hand-tuned transformer kernels that keep the tensor-parallel all-reduce boundaries but cut the number of separate kernel launches per layer, shrinking the fixed overhead that matters most at decode's tiny batch sizes. It also introduced ZeRO-Inference, which offloads weights to CPU or NVMe and streams them onto the GPU layer by layer — a throughput-oriented option for when you would rather trade latency to fit a giant model on fewer GPUs than a full TP x PP layout would need. The general lesson: the all-reduce *structure* Megatron defined is fixed, but how efficiently you execute the compute *between* the all-reduces (kernel fusion, CUDA graphs to erase launch overhead, quantized matmuls) is a large and separate lever on the same layout.

### The 405B and 671B reality

Llama-3.1-405B in FP16 needs 810 GB of weights — that is 11 H100s for weights alone, more than one node, so a production layout is TP=8 within each node plus PP across nodes (or FP8 to shrink to ~6 GPUs and fit closer to a node). DeepSeek-V3 at 671B total / 37B active is served with expert parallelism spreading its 256 experts across a multi-node cluster combined with attention TP — the sparse-FLOP payoff is that a 671B-capacity model costs roughly what a 37B dense model costs to run per token. The full multi-node orchestration for these is the subject of the [100B-plus serving post](/blog/machine-learning/model-serving/multi-node-llm-serving-100b-plus); the MoE-specific expert-placement and load-balancing depth is in [serving MoE models at scale](/blog/machine-learning/model-serving/serving-moe-models-at-scale).

The pragmatic version of these two cases is worth stating plainly, because it is what most teams actually run. For 405B, quantize to FP8 first and prefer a single-node TP=8 layout: 405 GB of weights becomes ~51 GB/GPU, fits one HGX node, keeps every all-reduce on NVLink, and sidesteps the cross-node handoff entirely — you only add PP=2 when long-context KV pressure forces it, exactly the judgment call from the 405B worked example. For DeepSeek-V3, the sparsity is the whole economic argument: because only 37B parameters are active per token, the per-token compute and therefore the marginal serving cost track a mid-size dense model, while the 671B of resident weights buy the quality of a frontier model. That asymmetry — frontier-model quality at mid-model per-token cost — is why MoE plus expert parallelism has become the default architecture for the largest served models, and why the EP axis, niche as it looks in this post, is where a fast-growing share of the industry's serving-cost optimization now happens.

### A named-hardware before-and-after

The single most impactful decision in a multi-GPU layout is keeping TP on the fast link. Here is the same 70B model, same 8 GPUs, wired two ways:

| Layout | Interconnect for TP | TP all-reduce/token | TPOT (approx) | Verdict |
|---|---|---|---|---|
| TP=8 within one HGX node | NVLink 4.0, 900 GB/s | ~0.31 ms | ~20 ms | Meets interactive SLO |
| TP=8 spanning 2 nodes (4+4) | InfiniBand NDR, 50 GB/s | ~5.6 ms | ~26+ ms | Comms-bound, margin gone |
| TP=8 spanning 2 nodes | Ethernet 100 GbE, 12.5 GB/s | ~11 ms | ~35+ ms | SLO blown |

Same silicon, same model, same TP degree — the only variable is whether the all-reduce rides NVLink or crosses the node. The correct fix for the bottom two rows is not "buy faster Ethernet," it is "use TP=4 within each node and PP=2 across them," which moves the cross-node traffic from 160 all-reduces per token down to 2 P2P sends per token.

## 10. When to use this (and when not to)

Sharding a model across GPUs is powerful and expensive in complexity, debugging surface, and blast radius. Reach for it deliberately.

**Use tensor parallelism when:**
- The model does not fit on one GPU (weights + KV at your target concurrency), or
- It fits but latency is too high and you have spare GPUs in the same NVLink node.
- **Cap TP at the node size (8).** Never span nodes with TP.

**Use pipeline parallelism when:**
- The model is too big for one node even at TP=8, so you need to hold layers across nodes, and
- You have enough concurrent load to keep the pipeline full (high, sustained QPS).
- PP tolerates InfiniBand/Ethernet, so it is the right tool for crossing the node boundary.

**Use expert parallelism when:**
- You are serving an MoE model and the experts do not fit on the TP group. It is not a general dense-model technique.

**Use data parallelism when:**
- You need more QPS and have latency headroom — replicate the smallest layout that fits and load-balance.

**Do NOT shard when:**
- **The model fits comfortably on one GPU.** A 7B/8B model in FP16 (16 GB) or an FP8 13B runs fine on a single 80 GB H100 with a large KV cache. Sharding it adds all-reduce latency and NCCL failure modes for nothing. Serve it single-GPU and scale with DP replicas.
- **You are tempted to set TP > node size.** The cross-node all-reduce will destroy your latency. Switch that degree to pipeline parallelism instead.
- **Your traffic is low and bursty and you built a deep pipeline.** The bubble will dominate. Prefer TP + DP for low-QPS latency-sensitive workloads; save PP for saturated, throughput-bound fleets.
- **You have not measured.** The theory tells you the shape of the curve; only a benchmark on your model, your hardware, and your traffic tells you where the crossover sits. Sweep TP degree with the harness above before committing a layout.

### Five layout mistakes and their fixes

The failure modes cluster into a short list you can screen a proposed layout against:

- **TP across nodes.** Symptom: latency fine in a single-node test, then 3–5x worse in the real two-node deployment. Fix: TP=8 within each node, PP across them.
- **Sizing an MoE model by active params.** Symptom: OOM at load despite "37B" fitting easily. Fix: budget the full total-parameter memory; all experts are resident.
- **Deep pipeline on bursty low-QPS traffic.** Symptom: GPUs mostly idle, latency dominated by the bubble. Fix: shallow PP (or none), lean on TP plus DP.
- **Cranking TP for throughput.** Symptom: aggregate throughput flat or falling as TP rises past the sweet spot, per-GPU efficiency in the 30s. Fix: lower TP, add DP replicas.
- **Budgeting weights but not KV.** Symptom: fits in a smoke test at concurrency 1, OOMs at concurrency 32. Fix: size with peak concurrency times max context KV included.

Every one of these is a specific misreading of the same underlying model — memory forces the shard, the interconnect caps the collective-heavy axes, and the KV cache has the final say on the degree. Get those three constraints straight, in that order, and the layout very nearly writes itself.

## Key takeaways

1. **You shard because of memory, first and always.** Weights (params x bytes) plus KV cache (${2 \cdot L \cdot H_{kv} \cdot d_h}$ per token, times concurrency times context) must fit in HBM. Do that math before anything else; the KV cache is what turns a "should fit" into an OOM under load.
2. **Tensor parallelism splits every matmul and pays two all-reduces per layer.** Its comms volume is ${2\frac{N-1}{N}}$ times the activation size per collective — bandwidth-hungry, so it must stay on NVLink, inside one node, capped at ~8 GPUs.
3. **Pipeline parallelism splits layers into stages and pays one P2P send per boundary.** It tolerates slow interconnect and is your tool for crossing node boundaries, at the cost of a bubble of fraction ${\frac{P-1}{M+P-1}}$ — which shrinks as concurrent requests rise.
4. **Expert parallelism spreads MoE experts and pays two all-to-alls per layer.** It unlocks huge sparse-capacity models (671B total / 37B active) but adds load-imbalance risk; defer to the MoE-serving post for depth.
5. **TP is a latency lever; PP and DP are throughput levers.** TP cuts TTFT/TPOT sublinearly and inverts past the node; PP scales aggregate throughput across nodes; DP scales QPS with zero comms but N x the memory.
6. **The classic layout is TP inside a node, PP across nodes, DP outside** — matching each collective to the interconnect it can afford. Total GPUs = TP x PP x DP.
7. **Derive the layout, don't guess it:** memory sets the minimum GPU count, the node size caps TP, PP covers overflow layers, DP multiplies for QPS. Encode it in a calculator and validate with `fits=True` including KV headroom.
8. **Measure the crossover.** Watch throughput-per-GPU fall as TP rises; that gap is the all-reduce tax, and it tells you when to stop.

## Further reading

- **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism** — Shoeybi et al., 2019. The original column/row tensor-parallel decomposition and the two-all-reduces-per-layer structure.
- **GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism** — Huang et al., 2019. The pipeline bubble and micro-batch scheduling that inference PP inherits.
- **Efficient Memory Management for Large Language Model Serving with PagedAttention** — Kwon et al., SOSP 2023. The vLLM paper; PagedAttention and continuous batching that keep pipelines full.
- **DeepSpeed-Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale** — Aminabadi et al., 2022. Tensor-parallel inference with kernel injection.
- **vLLM distributed serving documentation** — the authoritative reference for `--tensor-parallel-size`, `--pipeline-parallel-size`, and the Ray multi-node backend.
- **text-generation-inference documentation** — TGI's `--num-shard` sharding and its NCCL-based tensor parallelism.
- Within this series: [tensor parallelism and Megatron](/blog/machine-learning/distributed-training/tensor-parallelism-megatron) and [pipeline parallelism and the bubble](/blog/machine-learning/distributed-training/pipeline-parallelism-and-the-bubble) for the training-side derivations; [multi-node LLM serving for 100B-plus models](/blog/machine-learning/model-serving/multi-node-llm-serving-100b-plus) and [serving MoE models at scale](/blog/machine-learning/model-serving/serving-moe-models-at-scale) for the cluster- and MoE-level continuations; and [why LLM serving is different](/blog/machine-learning/model-serving/why-llm-serving-is-different) for the memory-wall foundation.

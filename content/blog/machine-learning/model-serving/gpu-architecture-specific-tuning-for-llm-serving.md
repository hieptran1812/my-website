---
title: "GPU-architecture-specific tuning for LLM serving: why the right config on an H100 is wrong on an A100"
date: "2026-07-07"
publishDate: "2026-07-07"
description: "Tune LLM serving architecture by architecture — Hopper H100/H200, Ampere A100, Blackwell B200, AMD MI300X, and Ada L40S each demand a different dtype, KV budget, and parallelism plan, and copying one GPU's launch config onto another is how you OOM in production."
tags:
  [
    "model-serving",
    "inference",
    "gpu",
    "vllm",
    "fp8",
    "fp4",
    "hopper",
    "blackwell",
    "mi300x",
    "tensor-parallelism",
    "llm-serving",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/gpu-architecture-specific-tuning-for-llm-serving-1.webp"
---

The pager goes off at 02:14. A capacity crunch pushed half your Llama-3-70B traffic onto a spare pool of A100 80GB nodes, and the deploy that has run flawlessly on your H100 fleet for six months is now crash-looping. The logs are blunt: `ValueError: fp8 quantization is not supported on this device`, then on the nodes that do start, `torch.cuda.OutOfMemoryError` the instant real traffic arrives. Nothing about the model changed. Nothing about the request mix changed. The only thing that changed is the three letters stamped on the silicon — and that was enough to turn a healthy config into an outage.

This is the single most expensive misconception in LLM serving: that a GPU is a GPU, and a launch command that works on one will work on another if it has "enough memory." It will not. The optimal serving configuration for a large language model is not a property of the model. It is a property of the *hardware* — specifically three hardware facts that differ on every architecture: what numeric precisions the tensor cores can execute, how many gigabytes of high-bandwidth memory (HBM) the card carries and how fast that memory is, and whether the card has NVLink or is stuck talking over PCIe. Change any one of those and the correct dtype, the correct batch size, the correct KV-cache budget, and the correct parallelism plan all move with it.

![Matrix comparing six data-center GPUs across memory and bandwidth, tensor-core precision floor, interconnect, and the resulting tuning verdict, showing that each architecture implies a different serving config](/imgs/blogs/gpu-architecture-specific-tuning-for-llm-serving-1.webp)

By the end of this post you will be able to walk up to any of the six GPUs in the matrix above — NVIDIA Hopper H100 and H200, Ampere A100, Blackwell B200/GB200, AMD MI300X, and Ada L40S — and derive, from first principles, the dtype to run, the batch size and KV budget to set, the parallelism scheme to pick, and the exact vLLM flags to launch with. We will do the memory math that turns "how big is my model" into "how many GPUs do I need," the bandwidth math that explains why an H200 serves the *same* model at seven times the concurrency of an H100, and the interconnect math that tells you when tensor parallelism is a free win and when it is a self-inflicted latency wound. Every technique lands back on the serving SLO triangle — **latency ↔ throughput ↔ cost** — because every one of these knobs is a trade on that triangle, and the trade is priced differently on each architecture.

If you have never shipped a model to production, one framing before we go deep: LLM inference is memory-bound, not compute-bound, for almost the entire generation phase. The model reads its multi-gigabyte weight matrices and its growing KV cache out of HBM for every single token it emits, and the arithmetic units sit half-idle waiting for those reads. That one fact — the memory wall — is why capacity, bandwidth, and the precision you store weights and KV cache in dominate the tuning discussion far more than raw FLOPS do. Keep it in mind; it explains most of what follows. For the deeper treatment of why decode is bandwidth-bound while prefill is compute-bound, see the companion post on [roofline analysis for LLM inference](/blog/machine-learning/model-serving/roofline-analysis-for-llm-inference).

## 1. The three hardware facts that decide every serving config

Before we go architecture by architecture, we need the reasoning framework that makes the whole exercise mechanical rather than mystical. Tuning a GPU for LLM serving is not a bag of tricks. It is a fixed, top-down decision procedure, and every architecture just plugs different numbers into the same four layers.

![Four-layer stack showing the tuning decision order from precision floor at the top, down through KV and batch budget, parallelism plan, and finally the vLLM launch flags](/imgs/blogs/gpu-architecture-specific-tuning-for-llm-serving-2.webp)

The layers resolve in order, and each layer's output constrains the next:

1. **Precision floor — what can the tensor cores execute?** A GPU can only accelerate a numeric format if it has hardware datapaths for it. Hopper and newer have FP8 tensor cores; Blackwell adds FP4; Ampere has neither and tops out at BF16/FP16/INT8. This is not a software toggle — you cannot flip a flag and get FP8 throughput on an A100, because the multiply-accumulate units to do it physically are not on the die. The precision floor sets your candidate dtypes for both weights and KV cache.

2. **KV + batch budget — how many gigabytes are left after the weights land?** Once you pick a weight precision, the model occupies a fixed number of gigabytes. Whatever HBM remains is your working memory: the KV cache, activations, and CUDA graph capture. That remaining budget, divided by the per-sequence KV footprint, is your maximum concurrent batch. More memory means more concurrency, full stop.

3. **Parallelism plan — does the interconnect support tensor parallelism?** If the model does not fit on one GPU, you must split it. Tensor parallelism (TP) shards each layer across GPUs and performs an all-reduce every layer — which is only viable if the GPUs are wired with a fast interconnect (NVLink). On PCIe-only cards, that per-layer all-reduce over a 64 GB/s link becomes the bottleneck and destroys decode latency, so you fall back to pipeline parallelism (PP) or simply replicate the model across single GPUs.

4. **Launch flags — the config falls out.** Only now do the vLLM (or TGI, or SGLang) arguments become determined: `--quantization`, `--kv-cache-dtype`, `--tensor-parallel-size`, `--max-num-seqs`, `--gpu-memory-utilization`. They are outputs of the three decisions above, not independent choices.

The reason a config is portable across models but not across GPUs is that layers 1–3 read the hardware, not the checkpoint. Swap the checkpoint and the byte counts change; swap the GPU and the *rules* change. Let us make each of the three hardware facts quantitative.

### The mechanics: three equations

**Fact 1 — tensor-core precision dictates dtype.** There is no formula here, only a lookup table baked into silicon, but it is the gate everything else passes through:

| Architecture | Tensor-core precisions | FP8? | FP4? |
|---|---|---|---|
| Ampere (A100) | TF32, BF16, FP16, INT8, INT4 | No | No |
| Ada (L40S, RTX 4090) | BF16, FP16, FP8 (E4M3/E5M2), INT8 | Yes | No |
| Hopper (H100, H200) | BF16, FP16, FP8 (E4M3/E5M2), INT8 | Yes | No |
| Blackwell (B200, GB200) | BF16, FP8, FP4 (E2M1/MXFP4), INT8 | Yes | Yes |
| CDNA3 (MI300X) | BF16, FP16, FP8 (OCP E4M3/E5M2), INT8 | Yes | No |

If your target dtype is not in the row, you either fall back to a supported dtype (losing the memory and throughput benefit) or the engine errors out. This is why `--quantization fp8` on an A100 is not "slow" — it is *impossible* in hardware, and vLLM will refuse it.

**Fact 2 — HBM capacity dictates shard count.** The total memory a served model needs is:

$$M_\text{total} = P \cdot b_w + N_\text{seq} \cdot L_\text{ctx} \cdot k + M_\text{overhead}$$

where $P$ is the parameter count, $b_w$ is bytes per weight (2 for BF16, 1 for FP8, 0.5 for FP4), $N_\text{seq}$ is the number of concurrent sequences, $L_\text{ctx}$ is the context length in tokens, $k$ is KV-cache bytes per token, and $M_\text{overhead}$ is a few gigabytes of activations plus CUDA-graph and framework reserve. The number of GPUs you need is then:

$$G = \left\lceil \frac{M_\text{total}}{U \cdot C_\text{HBM}} \right\rceil$$

where $C_\text{HBM}$ is the per-GPU HBM capacity and $U$ is the usable fraction (vLLM's `--gpu-memory-utilization`, typically 0.90). Notice $b_w$ sits inside $M_\text{total}$: halving the weight precision can push $G$ from 2 down to 1. Precision and shard count are one linked decision, not two.

**Fact 3 — interconnect dictates TP feasibility.** Tensor parallelism issues two all-reduce collectives per transformer layer (one after attention, one after the MLP). For a hidden size $H$ and batch of $B$ tokens in flight, each all-reduce moves roughly $2 B H \cdot b_a$ bytes of activations (with $b_a$ the activation byte width), and the per-layer communication time is approximately:

$$t_\text{comm} \approx \frac{2 \cdot 2 B H b_a}{W_\text{link}}$$

where $W_\text{link}$ is the interconnect bandwidth. On NVLink 4 ($W_\text{link} \approx 900$ GB/s) this is tens of microseconds and hides behind compute. On PCIe Gen4 ($W_\text{link} \approx 64$ GB/s), it is roughly $14\times$ slower per hop, repeated 80 times per token for a 70B model — and that is why TP over PCIe is a mistake, not a config. The interconnect is what makes tensor parallelism cheap or ruinous.

### Putting a number on the all-reduce tax

The phrase "TP over PCIe is ruinous" only lands when you see the microseconds, so let us plug real numbers into the Fact-3 equation. Take Llama-3-70B: hidden size $H = 8192$, 80 layers, and a decode step carrying an in-flight batch of $B = 32$ sequences (one token each). Per the formula, each layer's two all-reduces move about $2 \cdot 2 \cdot 32 \cdot 8192 \cdot 2 \approx 2.1$ MB of BF16 activations. Divide by the link:

- **NVLink 4 (900 GB/s):** about 2.3 microseconds per layer, so roughly 0.19 ms of communication per token across all 80 layers — a small fraction of the decode step, and most of it overlaps with compute.
- **PCIe Gen4 (64 GB/s):** about 33 microseconds per layer, so roughly 2.6 ms per token — and none of it overlaps cleanly, because each of the 160 collectives per token also pays a fixed kernel-launch and synchronization cost that compounds. In practice TP-over-PCIe decode runs three to five times slower than even this bandwidth math predicts.

On a token you wanted to emit in 25 ms, burning 2.6 ms-plus in collectives that will not hide is a self-inflicted p99 wound. Same model, same batch, same arithmetic — the interconnect alone decides whether tensor parallelism is invisible or fatal. That is why the four-layer procedure resolves interconnect before it ever prints a flag: link bandwidth is a hard gate on which parallelism schemes are even admissible, and no amount of downstream batch or precision tuning rescues a scheme the wiring cannot carry.

Those three facts — the precision table, the shard-count equation, and the all-reduce cost — are the entire mechanical spine of this post. Everything below is an application of them to a specific piece of silicon.

## 2. The five archetypes: which family your GPU belongs to

There are dozens of GPU SKUs, but for serving purposes they collapse into three archetypes defined by exactly the two axes that dominate the equations above: interconnect and memory capacity. The vendor logo does not matter; the archetype does.

![Tree grouping serving GPUs into three archetypes — NVLink data-center, big-HBM single-GPU, and PCIe-only cards — each mapping to a distinct tuning recipe](/imgs/blogs/gpu-architecture-specific-tuning-for-llm-serving-3.webp)

**Archetype 1 — NVLink data-center GPUs.** H100, H200, A100, and B200 all sit in 8-GPU HGX/DGX baseboards wired with NVLink and an NVSwitch fabric. Here tensor parallelism is a first-class tool: splitting a big model across 2, 4, or 8 GPUs costs almost nothing in latency because the all-reduce rides a 600–1800 GB/s link. Tuning these is about picking the lowest precision the tensor cores support (to shrink the model and grow the KV budget), then scaling batch size until you hit either the memory ceiling or your p99 latency SLA. The Blackwell members of this archetype push it further: the GB200 NVL72 makes 72 GPUs one NVLink domain, which changes the parallelism economics for the very largest models.

**Archetype 2 — big-HBM single-GPU.** AMD's MI300X carries 192 GB of HBM3. That is enough that a 70B model fits comfortably on *one* GPU with room for a large KV cache, so the entire tensor-parallelism question can disappear. The tuning philosophy inverts: instead of "how do I split this model across GPUs without paying too much communication," it becomes "how do I keep this single enormous GPU busy." Fewer shards means no cross-GPU all-reduce in the hot path at all, which is a latency advantage — as long as one GPU's compute and bandwidth can feed your throughput target.

**Archetype 3 — PCIe-only cards.** The Ada-generation L40S (48 GB) and consumer cards like the RTX 4090 (24 GB) have no NVLink. They talk to each other only over PCIe, at roughly 64 GB/s. For these, tensor parallelism is off the table for latency-sensitive serving. The correct pattern is replication — run one full model per GPU and load-balance across replicas — or, for models too big for one card, pipeline parallelism, which communicates far less than TP. These cards win on one axis only: dollars. When your model is small enough to fit on one card and your QPS is modest, an L40S at a fraction of an H100's price is often the right economic call.

Every GPU in this post is one of these three archetypes. When a new SKU launches, place it on the interconnect axis and the capacity axis and you already know how to tune it before you have read a single benchmark. Now let us do the memory math that turns capacity into a shard count, then go architecture by architecture.

## 3. The memory math: how HBM capacity dictates shard count

This is the calculation that would have prevented the 02:14 outage. Let us make it concrete for Llama-3-70B, the running example for the rest of the post.

![Graph showing Llama-3-70B weights branching by precision into BF16 at 140 gigabytes, FP8 at 70 gigabytes, and FP4 at 35 gigabytes, then merging into the GPU configurations each footprint requires](/imgs/blogs/gpu-architecture-specific-tuning-for-llm-serving-4.webp)

**Step 1: weight memory.** Llama-3-70B has about 70.6 billion parameters. Weight memory is simply parameters times bytes per parameter:

- BF16 (2 bytes): $70.6 \times 10^9 \times 2 \approx 141$ GB — call it 140 GB.
- FP8 (1 byte): $\approx 70$ GB.
- FP4 (0.5 byte): $\approx 35$ GB.

Right away the precision choice determines feasibility. 140 GB does not fit on any single 80 GB GPU, so BF16 forces sharding. 70 GB fits on one 80 GB card with a sliver left over. 35 GB fits on almost anything. This is the branch in the figure above.

**Step 2: KV-cache memory.** The KV cache stores the key and value vectors for every token of every in-flight sequence, so the model does not recompute attention over the whole prefix each step. Its per-token size is:

$$k = 2 \cdot L \cdot H_\text{kv} \cdot d_\text{head} \cdot b_\text{kv}$$

The factor 2 is for K and V; $L$ is the number of layers; $H_\text{kv}$ is the number of key/value heads (with grouped-query attention this is far smaller than the number of attention heads); $d_\text{head}$ is the head dimension; and $b_\text{kv}$ is bytes per KV element. For Llama-3-70B, $L = 80$, $H_\text{kv} = 8$ (GQA), $d_\text{head} = 128$:

$$k = 2 \cdot 80 \cdot 8 \cdot 128 \cdot b_\text{kv} = 163{,}840 \cdot b_\text{kv} \ \text{bytes/token}$$

In BF16 ($b_\text{kv} = 2$) that is about 320 KB per token; in FP8 ($b_\text{kv} = 1$) about 160 KB per token. At a 4,096-token context, one sequence's KV cache is:

- BF16 KV: $4096 \times 320\,\text{KB} \approx 1.31$ GB per sequence.
- FP8 KV: $4096 \times 160\,\text{KB} \approx 0.64$ GB per sequence.

**Step 3: shard count and concurrency.** Plug into the equations from section 1. On an 80 GB H100 running FP8 weights (70 GB) at $U = 0.90$, usable memory is about 72 GB, leaving roughly 2–10 GB for KV after overhead — call it 10 GB in a lean deployment. At 0.64 GB per FP8 sequence, that is about 16 concurrent sequences. Tight. On an H200's 141 GB, the same 70 GB of weights leaves about 71 GB for KV — roughly 110 concurrent sequences, a nearly $7\times$ jump for the *identical model and dtype*. Capacity, through this arithmetic, is concurrency.

Here is a small calculator you can adapt to any model and GPU. It is the tool that answers "how many GPUs and how much batch" before you ever touch a launch command.

```python
# kv_budget.py - derive shard count and concurrency for a model on a GPU.
import math

def bytes_per_weight(dtype: str) -> float:
    return {"bf16": 2.0, "fp16": 2.0, "fp8": 1.0, "int8": 1.0, "fp4": 0.5}[dtype]

def kv_bytes_per_token(n_layers, n_kv_heads, head_dim, kv_dtype):
    b = {"bf16": 2, "fp16": 2, "fp8": 1, "int8": 1}[kv_dtype]
    return 2 * n_layers * n_kv_heads * head_dim * b  # 2 = K and V

def plan(params_b, gpu_hbm_gb, weight_dtype, kv_dtype,
         n_layers, n_kv_heads, head_dim, ctx_len,
         util=0.90, overhead_gb=3.0):
    weight_gb = params_b * 1e9 * bytes_per_weight(weight_dtype) / 1e9
    usable_gb = gpu_hbm_gb * util
    # Shard count: enough GPUs so weights + overhead fit with room for KV.
    gpus = max(1, math.ceil((weight_gb + overhead_gb) / usable_gb))
    total_usable = gpus * usable_gb
    kv_gb = total_usable - weight_gb - overhead_gb
    per_seq_gb = kv_bytes_per_token(n_layers, n_kv_heads, head_dim, kv_dtype) \
                 * ctx_len / 1e9
    max_seqs = max(0, int(kv_gb / per_seq_gb))
    return dict(weight_gb=round(weight_gb), gpus=gpus,
                kv_gb=round(kv_gb, 1), max_concurrent=max_seqs)

# Llama-3-70B, 4k context, on one H100 (FP8) vs one H200 (FP8)
for name, hbm in [("H100 80GB", 80), ("H200 141GB", 141)]:
    p = plan(70.6, hbm, "fp8", "fp8", 80, 8, 128, 4096)
    print(name, p)
```

Run it and you get the numbers from the figure: the H100 fits the model on one GPU but supports only a handful of concurrent sequences, while the H200 supports many times more. Change `weight_dtype` to `bf16` and the H100 plan jumps to two GPUs — the shard count moved because the precision moved. This little function is the whole of section 1 made executable, and it is worth wiring into your capacity-planning pipeline so nobody ever again discovers the shard count at 02:14. For the accuracy side of this trade — how much quality you give up per byte you save — see [quantization for LLM serving](/blog/machine-learning/model-serving/quantization-for-llm-serving) and the low-precision deep-dive on [FP8 and FP4 serving](/blog/machine-learning/model-serving/fp8-fp4-low-precision-serving-deep-dive).

### What the overhead term actually holds

The $M_\text{overhead}$ term is not a fudge factor; it is a real, itemizable claim on HBM that trips teams who budget only for weights and KV. On a vLLM worker it covers the activation buffers for the largest in-flight batch (a few hundred megabytes to a couple of gigabytes depending on batch and sequence length), the CUDA graph captures vLLM records to eliminate per-step launch overhead (these can claim 1–3 GB when many batch sizes are captured), the NCCL communication buffers for any tensor-parallel group, the framework's own reserved arena, and fragmentation slack. This is why `--gpu-memory-utilization` defaults near 0.90 rather than 0.98: the last ten percent is the margin that keeps a burst of long sequences from tipping the worker into an out-of-memory kill. Set it too high and you trade a few extra concurrent sequences for a crash-loop under load; set it too low and you strand KV budget you paid for. On the 192 GB cards the overhead is a rounding error and you can push utilization toward 0.95; on an 80 GB card serving a 70 GB model it is the difference between sixteen concurrent sequences and eight.

The same weight-memory arithmetic, tabulated for the model sizes you are most likely to serve, is the fastest sanity check there is — read across a row and you know immediately which precisions fit a given card:

| Model | BF16 (2 B/param) | FP8 (1 B) | FP4 (0.5 B) | INT4 (0.5 B) |
|---|---|---|---|---|
| 7B   | 14 GB  | 7 GB   | 3.5 GB | 3.5 GB |
| 13B  | 26 GB  | 13 GB  | 6.5 GB | 6.5 GB |
| 34B  | 68 GB  | 34 GB  | 17 GB  | 17 GB  |
| 70B  | 141 GB | 70 GB  | 35 GB  | 35 GB  |
| 405B | 810 GB | 405 GB | 203 GB | 203 GB |

Weights only; add KV and overhead on top. The table makes the shard boundaries jump out: a 70B model crosses the 80 GB line in BF16 (forcing sharding) but sits under it in FP8, and a 405B model still needs eight 80 GB cards even in FP8 but drops to four in FP4. Every horizontal step of two in this table is one bit of precision traded for one halving of the GPU count — the linked precision-and-parallelism decision from Fact 2, made visible.

## 4. NVIDIA Hopper H100: the FP8 workhorse

The H100 SXM is the default LLM-serving GPU of this era, and its defining tuning feature is FP8. Hopper's fourth-generation tensor cores execute two 8-bit float formats natively — E4M3 (4 exponent bits, 3 mantissa, the accuracy-favoring format used for weights and activations) and E5M2 (wider range, used for gradients in training and occasionally for KV). Combined with the Transformer Engine's per-tensor scaling, FP8 on Hopper is close to free in quality for most models while roughly doubling arithmetic throughput and, more importantly for serving, halving the bytes read from HBM per token.

The hardware profile you are tuning against:

- **Memory:** 80 GB HBM3, about 3.35 TB/s bandwidth.
- **Precision floor:** FP8 (E4M3/E5M2), BF16, FP16, INT8. No FP4.
- **Interconnect:** NVLink 4, 900 GB/s per GPU aggregate, in an 8-GPU NVSwitch fabric.
- **Extras:** TMA (Tensor Memory Accelerator) for asynchronous bulk copies, a larger 50 MB L2 cache than Ampere, and DPX instructions.

Those extras are not marketing; they are why FP8 on Hopper actually reaches its theoretical throughput instead of stalling on memory movement. TMA lets a kernel issue a single instruction that copies a whole tile of weights or KV from HBM into shared memory asynchronously, freeing the warps to keep computing instead of babysitting the load — which is exactly what FlashAttention-3 exploits to push Hopper attention close to its FP8 roofline. The fourth-generation tensor cores add `wgmma`, a warpgroup-wide asynchronous matrix-multiply instruction that overlaps with those TMA loads, and thread-block clusters let neighboring SMs share data directly through distributed shared memory rather than round-tripping to L2. The Transformer Engine ties it together by tracking a running amax history per tensor and choosing FP8 scale factors automatically, so the E4M3 weights and activations keep their dynamic range without manual calibration. For a serving engineer the practical takeaway is narrow but load-bearing: the near-2× FP8 speedup on the matmul-heavy phases materializes only because these copy-and-scale mechanisms keep the tensor cores fed, which is also why an FP8 kernel that predates these paths leaves much of that speedup on the table. Pin a recent vLLM and CUDA, and let the engine select the Hopper-specific attention kernels.

The tuning verdict follows straight from the three facts: run **FP8 weights and FP8 KV cache** to shrink the footprint and grow the KV budget, then push batch size as far as your p99 allows because NVLink makes any needed sharding cheap.

![Grid contrasting the Llama-3-70B memory budget in BF16 versus FP8 on an H100, where the BF16 path needs two GPUs and the FP8 path fits on one](/imgs/blogs/gpu-architecture-specific-tuning-for-llm-serving-5.webp)

The figure above is the whole H100 tuning story in one picture: moving weights and KV from BF16 to FP8 collapses a two-GPU tensor-parallel deployment into a single-GPU one, eliminating the per-layer all-reduce tax entirely. That is a latency win *and* a cost win — the rare knob that helps two corners of the SLO triangle at once, paid for only in a small, usually-tolerable accuracy cost.

Here is a production-grade vLLM launch for Llama-3-70B on H100. Note the FP8 KV-cache flag, which is the Hopper-specific lever most teams forget:

```bash
# H100 80GB - single-GPU FP8 serving of Llama-3-70B.
# FP8 weights (70GB) fit on one card; FP8 KV doubles the concurrency budget.
vllm serve meta-llama/Meta-Llama-3-70B-Instruct \
  --quantization fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 8192 \
  --max-num-seqs 24 \
  --gpu-memory-utilization 0.92 \
  --enable-chunked-prefill \
  --port 8000
```

If you need more concurrency or longer contexts than one H100's KV budget allows, scale to `--tensor-parallel-size 2` and the NVLink fabric absorbs the all-reduce cost. Two H100s in FP8 give you the model replicated in effect-free TP with a much larger KV pool — roughly doubling `--max-num-seqs`. Because the interconnect is fast, the decision to shard is driven purely by whether you need the extra memory, not by any latency penalty.

Two H100-specific tuning notes that matter in practice:

- **Prefer `fp8_e4m3` for the KV cache**, not `fp8_e5m2`. E4M3's extra mantissa bit preserves attention-score precision better; E5M2's wider range is rarely needed for cached keys and values, which are bounded in magnitude by layernorm.
- **Enable chunked prefill.** On Hopper, mixing prefill chunks with decode steps in the same batch keeps the tensor cores fed and smooths TTFT (time to first token) under bursty load without starving decode. It interacts well with FP8 because the smaller KV footprint leaves more room for prefill chunks.

#### Worked example: Llama-3-70B on one H100 vs the naive two-GPU BF16 config

A team ships Llama-3-70B on 2× H100 in BF16 because that is what the model card's example command used. Weights take 140 GB across two cards (70 GB each), leaving about 4 GB per card for KV in BF16 — roughly 3 sequences per card, 6 total, at 4k context. Throughput is bottlenecked not by compute but by that tiny KV pool: the scheduler cannot batch enough sequences to saturate the tensor cores, and GPU utilization hovers around 40%. Switching to FP8 weights + FP8 KV on a *single* H100 fits the 70 GB model with about 10 GB of KV — roughly 16 concurrent sequences on one card, versus 6 on two. That is a $2\times$ improvement in per-request density and a halving of the GPU bill simultaneously, because FP8 turned a memory-starved two-GPU deployment into a comfortable one-GPU deployment. The BF16 config was not wrong for the model; it was wrong for the H100.

## 5. NVIDIA Hopper H200: same compute, far more memory

The H200 is the most misunderstood tuning target in this list, because on paper it looks like a minor refresh — same Hopper GPU, same FP8 tensor cores, same NVLink 4. The datasheet difference is only two lines: 141 GB of HBM3e instead of 80 GB of HBM3, and 4.8 TB/s of bandwidth instead of 3.35 TB/s. But for LLM serving those two lines are the whole game, because inference is memory-bound. The extra 61 GB is almost entirely usable as KV cache, and the extra 1.45 TB/s of bandwidth directly accelerates the decode phase.

![Before-and-after comparison of Llama-3-70B in FP8 on an H100 versus an H200, showing KV budget rising from about 10 to 71 gigabytes and concurrency from roughly 16 to 110 sequences](/imgs/blogs/gpu-architecture-specific-tuning-for-llm-serving-8.webp)

The tuning implication is the cleanest in this post: **on an H200, raise `--max-num-seqs` and `--max-model-len` aggressively, because your bottleneck moved.** The same Llama-3-70B in FP8 that supported ~16 concurrent sequences on an H100 supports roughly 110 on an H200 — the KV budget went from ~10 GB to ~71 GB, and concurrency scales almost linearly with it. If you copy your H100 config to an H200 unchanged, you leave that $7\times$ concurrency headroom completely on the floor; the GPU runs at a fraction of its capacity while you wonder why the upgrade "didn't help."

```bash
# H200 141GB - same model, same dtype, but the memory-driven knobs open up.
vllm serve meta-llama/Meta-Llama-3-70B-Instruct \
  --quantization fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 32768 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.92 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --port 8000
```

Two changes versus the H100 line, both unlocked by memory: `--max-num-seqs` jumps from 24 to 128, and `--max-model-len` grows from 8k to 32k because the KV pool can now hold long contexts for many sequences at once. I also switched on `--enable-prefix-caching`: with 71 GB of KV headroom, retaining shared prompt prefixes across requests is cheap and pays off handsomely for chat and RAG workloads where system prompts repeat.

The bandwidth half of the upgrade matters too, and it is easy to miss because it does not change any flag. Decode reads the entire weight matrix and the growing KV cache from HBM every token; at 4.8 TB/s versus 3.35 TB/s, each decode step's memory reads complete about 1.4× faster, which shows up as lower TPOT (time per output token) at the same batch size. So the H200 gives you both more concurrency (from capacity) and faster per-token latency (from bandwidth) — it moves throughput and latency in the same direction, which almost nothing else in serving does.

To put a number on the bandwidth half: in the decode phase a single sequence's per-token latency is floored by the time to stream the model weights and its slice of the KV cache out of HBM once. For 70 GB of FP8 weights that floor is roughly ${70\,\text{GB} / 3.35\,\text{TB/s} \approx 21}$ ms on an H100 versus ${70\,\text{GB} / 4.8\,\text{TB/s} \approx 15}$ ms on an H200 — about a 30% lower memory-bound TPOT for the identical model, before you have batched anything. Batching amortizes that weight read across many sequences so the per-sequence gap narrows at high concurrency, but in the low-batch, latency-sensitive regime the H200's bandwidth is a direct per-token win that no flag exposes and no config on the H100 can match. It is the sharpest possible illustration of the memory-wall framing from the introduction: two GPUs with identical arithmetic units, and the faster one is faster purely because it reads memory faster.

#### Worked example: sizing the H200 KV budget for a long-context RAG service

A RAG service needs 16k-token contexts (large retrieved passages) and targets 80 concurrent users. Per-sequence FP8 KV at 16k tokens is $16384 \times 160\,\text{KB} \approx 2.6$ GB. Eighty sequences need about 210 GB of KV — impossible on a single H100 (10 GB budget) and even beyond a single H200 (71 GB). The plan function says: on H200, `--tensor-parallel-size 2` gives about 142 GB of KV budget across two cards, still short. Three H200s (≈213 GB KV) meet it. On H100s in FP8 you would need far more cards for the same KV pool, because each contributes only ~10 GB. The H200's capacity is not a luxury here; it is what makes the service buildable in single-digit GPU counts. This is the memory-bandwidth-wins pattern that the H200 was designed for, and it is why long-context and high-concurrency services are the workloads that justify its premium. For the broader treatment of packing concurrency against an SLA, see [high-concurrency SLO management](/blog/machine-learning/model-serving/high-concurrency-slo-management).

## 6. NVIDIA Ampere A100: no FP8, tune for BF16 and INT8

The A100 is still everywhere — it is the workhorse of the previous generation, it fills spot-instance pools, and it is what your capacity crunch will spill onto. Tuning it correctly starts with accepting what it *cannot* do. Ampere's third-generation tensor cores support TF32, BF16, FP16, INT8, and INT4 — but **not FP8**. There is no E4M3 datapath on the die. Every FP8 flag you rely on for Hopper is off the table.

The hardware profile:

- **Memory:** 40 GB or 80 GB HBM2e; the 80 GB SXM variant runs about 2.0 TB/s (the 40 GB PCIe variant is ~1.55 TB/s).
- **Precision floor:** BF16, FP16, TF32, INT8, INT4. No FP8, no FP4.
- **Interconnect:** NVLink 3, 600 GB/s per GPU — slower than Hopper's 900, but still firmly in "TP is cheap" territory.

![Before-and-after comparison showing a Hopper FP8 launch config failing on an A100 at the quantization, KV-cache, and batch-size layers, versus a corrected A100 config using BF16 and a smaller KV budget](/imgs/blogs/gpu-architecture-specific-tuning-for-llm-serving-6.webp)

The figure shows exactly how the H100 config fails when it lands on an A100, layer by layer: `--quantization fp8` hits nonexistent hardware, `--kv-cache-dtype fp8` has no FP8 KV datapath, and the batch size tuned for FP8's small footprint OOMs when weights and KV are suddenly twice as large in BF16. The corrected config re-derives each layer for Ampere.

Because you cannot use FP8 to shrink weights, the A100 tuning philosophy is: **use BF16 as the default, reach for INT8 (via GPTQ or AWQ) when you need the memory back, and budget the KV cache tightly because it is expensive in BF16.** A 70B model in BF16 is 140 GB, which needs at least two 80 GB A100s just for weights, and realistically four to leave any meaningful KV budget.

```bash
# A100 80GB - BF16 tensor-parallel serving of Llama-3-70B.
# No FP8 on Ampere; weights are 140GB, so TP across 4 cards for KV headroom.
vllm serve meta-llama/Meta-Llama-3-70B-Instruct \
  --dtype bfloat16 \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --max-num-seqs 48 \
  --gpu-memory-utilization 0.90 \
  --enable-chunked-prefill \
  --port 8000
```

If four A100s is more than your budget, the INT8 path via a pre-quantized checkpoint gets 70B weights down to ~70 GB, fitting on two cards with room for KV:

```bash
# A100 80GB - INT8 (AWQ/GPTQ) to halve weight memory without FP8 hardware.
vllm serve TheBloke/Llama-3-70B-AWQ \
  --quantization awq \
  --dtype float16 \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.90 \
  --port 8000
```

Note the difference in kind between the two quantization stories. On Hopper, FP8 is a *hardware-accelerated* format — you get both the memory savings and a compute speedup. On Ampere, INT8 via AWQ/GPTQ gives you the memory savings but the speedup is smaller and comes from weight-only quantization that dequantizes on the fly; the tensor cores still do the matmul in FP16. So INT8 on A100 is primarily a *capacity* trick (fit the model on fewer cards), not a throughput trick, whereas FP8 on H100 is both. That distinction changes when you would reach for it: on A100 you quantize to reduce GPU count; on H100 you quantize even when the model already fits, because it is faster.

**MIG is the A100's other serving lever, and it points the opposite direction from everything above.** Multi-Instance GPU partitions a single A100 into as many as seven hardware-isolated slices, each with its own dedicated SMs, L2 slice, and memory bank — a `1g.10gb` slice gets one-seventh of the compute and 10 GB of the 80 GB. For LLM serving this is not the tool for big models; it is the tool for *many small ones*. If your fleet runs a swarm of 1B–7B models (routers, classifiers, embedding models, small chat endpoints) with modest individual QPS, MIG lets one A100 present as seven independent, non-interfering endpoints, each immune to a noisy neighbor's latency spikes because the isolation is enforced in silicon rather than by the scheduler. The tuning move is to size the model to a slice: a 7B model in INT8 fits a `2g.20gb` slice with KV headroom, and you get guaranteed per-tenant QoS that shared-GPU multiplexing cannot promise. It is the one case where the Ampere philosophy flips from "shard a big model across cards" to "pack many small models into one card." That the older, cheaper A100 carries full MIG support is a real reason it stays in fleets long after the frontier moves on — the workload that most wants hardware partitioning is exactly the fleet-of-small-models workload where a discounted A100 is the right economic call.

One more Ampere-specific caution: the A100's KV budget is genuinely tight in BF16, so if you see the scheduler thrashing — preempting and recomputing sequences under load — that is the KV pool being too small, not a scheduler bug. The fix is either fewer concurrent sequences, more cards, or dropping to INT8. Do not chase it in the scheduler config; chase it in the memory budget. The mechanics of that preemption behavior are covered in [request scheduling and preemption](/blog/machine-learning/model-serving/request-scheduling-and-preemption).

#### Worked example: the 02:14 outage, solved

Return to the opening scenario. The H100 launch line — `--quantization fp8 --kv-cache-dtype fp8 --max-num-seqs 24` on a single card — was copied onto A100 80GB nodes. It fails three ways: FP8 quantization is rejected (no hardware), and on any node where someone stripped the FP8 flags, the BF16 70B model at 140 GB cannot even load on one 80 GB card, let alone serve batch 24. The correct A100 config is a different config entirely: BF16 with `--tensor-parallel-size 4` (or INT8/AWQ with TP2), `--max-num-seqs` dropped to match the smaller KV budget, and no FP8 flags at all. The lesson the postmortem should record is not "the A100s were broken." It is "our deploy hard-coded a Hopper config, and configs are per-architecture." The fix is to make the launch config a function of the detected GPU — which is exactly the config-picker we build in section 10.

## 7. NVIDIA Blackwell B200 and GB200: FP4 and the NVL72 domain

Blackwell is the current frontier, and it introduces two tuning levers that did not exist before: native FP4 tensor cores and a radically larger NVLink domain. Both change the calculus for the largest models.

The B200 hardware profile:

- **Memory:** 192 GB HBM3e, about 8.0 TB/s bandwidth — roughly 2.4× the H100's capacity and 2.4× its bandwidth.
- **Precision floor:** FP4 (E2M1 / MXFP4 with microscaling), FP8, BF16, INT8. The second-generation Transformer Engine manages FP4 scaling automatically.
- **Interconnect:** fifth-generation NVLink at 1.8 TB/s per GPU; in the GB200 NVL72 rack, 72 Blackwell GPUs form a single NVLink domain that behaves like one enormous accelerator.

**When is FP4 worth it?** FP4 halves the footprint again versus FP8 — Llama-3-70B drops to about 35 GB in weights — and roughly doubles arithmetic throughput on Blackwell's tensor cores. For very large models (hundreds of billions of parameters) and for throughput-maximizing offline or batch workloads, FP4 is a large win: it can turn a multi-GPU FP8 deployment into a single-GPU FP4 one, or a whole rack into a fraction of it. But FP4 has only four bits, and the accuracy cost is real and model-dependent; naive FP4 can noticeably degrade quality on reasoning-heavy tasks. The microscaling MXFP4 format (a shared scale per small block of values) recovers much of that, and modern recipes keep sensitive layers (embeddings, the final projection, sometimes attention) in FP8 or BF16 while running the bulk of the MLP weights in FP4. So the tuning rule is: reach for FP4 on Blackwell when throughput or capacity is the binding constraint and you have validated quality on your eval set; keep latency-critical, quality-sensitive paths in FP8. It is a throughput-corner move, not a default.

**How four bits stay usable: microscaling.** A raw FP4 value (E2M1: 1 sign, 2 exponent, 1 mantissa bit) represents only sixteen distinct levels, nowhere near enough dynamic range for a whole weight matrix. Microscaling recovers the range by attaching a shared scale factor to each small block of values rather than to the tensor as a whole. MXFP4, the open OCP format, groups 32 elements under one shared E8M0 (power-of-two) scale; NVIDIA's NVFP4 uses a finer block of 16 elements with an FP8 E4M3 scale plus a second-level per-tensor FP32 scale, which buys back noticeably more accuracy at a small metadata cost. The second-generation Transformer Engine computes and applies these scales in hardware during the GEMM, so from the serving side you pick the quantized checkpoint and the engine handles the block bookkeeping. The tuning consequence is that not all "FP4" is equal: an NVFP4 checkpoint typically holds quality better than a naive MXFP4 one on the same model, so validate the specific format you are shipping, not FP4 as a category.

```bash
# B200 192GB - FP4 serving for maximum throughput; validate quality first.
# 70B in FP4 is ~35GB, leaving enormous KV headroom on one card.
vllm serve meta-llama/Meta-Llama-3-70B-Instruct \
  --quantization modelopt_fp4 \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 32768 \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.90 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --port 8000
```

Two things to note. First, weights are FP4 but the KV cache is still FP8 — FP4 KV is generally not worth the accuracy hit, and FP8 KV already gives you plenty of headroom on 192 GB. Second, `--max-num-seqs` is high because the tiny weight footprint leaves most of 192 GB for KV. On a B200 the binding constraint for many models becomes compute or bandwidth saturation rather than memory, which is a different tuning regime than every GPU above it in this list.

**The NVL72 domain changes parallelism economics.** On an 8-GPU HGX, tensor parallelism beyond 8 is impossible and pipeline or expert parallelism must cross slower node boundaries. In a GB200 NVL72, all 72 GPUs share one NVLink fabric, so tensor and expert parallelism can span dozens of GPUs at NVLink bandwidth. For a 671B-parameter mixture-of-experts model, expert parallelism (EP) that would otherwise be throttled by inter-node networking now runs inside the NVLink domain — which is precisely the workload NVIDIA built the NVL72 for. If you are serving frontier-scale MoE models, the NVL72's large coherent domain is the single biggest architectural lever available, because it makes wide EP cheap. For the parallelism taxonomy this builds on, see [tensor, pipeline, and expert parallelism for serving](/blog/machine-learning/model-serving/tensor-pipeline-expert-parallelism-for-serving) and the multi-node treatment in [serving 100B+ models across nodes](/blog/machine-learning/model-serving/multi-node-llm-serving-100b-plus).

A note on accuracy and honesty: NVIDIA's headline "up to 4× faster inference than H100" for Blackwell folds together FP4's compute doubling, the memory and bandwidth increase, and architectural improvements, and it is measured on specific models and batch configurations. Treat it as a best-case vendor figure, not a guarantee for your workload. The reliable, first-principles claims are narrower and hold everywhere: FP4 halves weight bytes versus FP8, 192 GB is 2.4× the H100's capacity, and the NVL72 makes 72-GPU NVLink domains real. Tune against those facts, and measure the rest on your own model.

#### Worked example: 405B, one Blackwell domain versus an H100 cluster

Take Llama-3.1-405B, the size where FP4 stops being optional and starts being decisive. In BF16 the weights alone are 810 GB — twelve 80 GB H100s just to hold them, realistically sixteen once you want any KV budget, wired across two nodes with the inter-node hop that pipeline or tensor parallelism must cross. In FP8 that halves to 405 GB, still six-to-eight H100s. In FP4 on Blackwell it drops to about 203 GB, which fits inside two B200s (384 GB) with room for KV, and every GPU in a GB200 NVL72 sits in one NVLink domain — so even a wide tensor- or expert-parallel split runs at 1.8 TB/s instead of crossing a slow node boundary. The same checkpoint that needs a two-node H100 cluster with an inter-node bottleneck becomes a two-GPU, single-domain deployment on Blackwell. That is the frontier-scale payoff of FP4 plus NVL72: not merely faster, but the removal of the multi-node communication problem for a whole class of models that could not previously be served inside one NVLink fabric. The catch is the one from just above — 405B at four bits demands the most careful quality validation of anything in this post, so budget eval time before you budget the GPUs.

## 8. AMD MI300X: 192 GB means fewer shards

The MI300X is the most interesting tuning target in this list because its enormous 192 GB of HBM3 flips the central problem on its head. Where NVIDIA data-center GPUs push you toward splitting models across cards, the MI300X invites you to *stop splitting*.

The hardware profile (CDNA3 architecture):

- **Memory:** 192 GB HBM3, about 5.3 TB/s bandwidth — the same capacity as a B200 and more than 2× an H100, at bandwidth between H200 and B200.
- **Precision floor:** FP8 (OCP E4M3/E5M2), BF16, FP16, INT8. No FP4. FP8 is supported through ROCm and vLLM's ROCm backend.
- **Interconnect:** Infinity Fabric between GPUs in a node (roughly 896 GB/s aggregate per GPU in the 8-GPU OAM baseboard), and crucially **no NVLink** — the fabric is AMD's own, and cross-node scaling uses standard networking.

The 192 GB is a direct consequence of CDNA3's chiplet layout, and the layout matters for how you reason about the card. An MI300X is not a monolithic die: it is eight accelerator complex dies (XCDs, 304 compute units in total) mounted on four I/O dies over eight HBM3 stacks, all knit together by a 256 MB Infinity Cache and on-package Infinity Fabric. Packing eight HBM3 stacks around the compute is what lets a single package carry 192 GB where a reticle-limited monolithic die tops out far lower — the same advanced-packaging bet Blackwell makes with its dual-die B200. For a serving engineer the payoff is that the card presents as one enormous coherent GPU to the software: you address 192 GB and 304 CUs as a single device, which is exactly why the 70B-on-one-GPU pattern holds. The one architectural caveat is that intra-package bandwidth, while very high, is not infinite, and extremely bandwidth-hungry kernels can feel the chiplet boundaries; but for the weight-streaming, KV-reading pattern of LLM decode, the 5.3 TB/s aggregate is what you actually experience, and it is excellent.

**The big-HBM tuning philosophy.** A 70B model in FP8 is 70 GB, and in BF16 it is 140 GB — both fit on a single 192 GB MI300X with room for a substantial KV cache. That means for models up to roughly 140B in FP8, you can serve on *one* GPU with *zero* tensor parallelism in the hot path. No per-layer all-reduce, no cross-GPU synchronization on the decode critical path, and no interconnect bottleneck to reason about. The tuning question becomes keeping a single very large GPU busy: crank `--max-num-seqs` to fill the generous KV budget, and let continuous batching saturate the compute.

```bash
# MI300X 192GB (ROCm) - serve Llama-3-70B on ONE GPU, no tensor parallelism.
# 70B FP8 = ~70GB weights, leaving ~100GB+ for KV on a single card.
# Requires the ROCm build of vLLM (rocm/vllm image).
VLLM_USE_TRITON_FLASH_ATTN=1 \
vllm serve meta-llama/Meta-Llama-3-70B-Instruct \
  --quantization fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --max-num-seqs 192 \
  --gpu-memory-utilization 0.92 \
  --port 8000
```

That `--tensor-parallel-size 1` on a 70B model is the whole point — it is a configuration that is simply impossible on an 80 GB NVIDIA card, and it removes an entire class of distributed-systems failure modes (NCCL hangs, one-rank OOM, all-reduce stalls) from your production surface. When your model fits on one GPU, distributed serving bugs cannot happen because there is nothing distributed.

Practical notes on tuning the MI300X:

- **Use the ROCm build of vLLM** (`rocm/vllm` container) and, for attention, the ROCm-optimized FlashAttention or the Triton FlashAttention path (`VLLM_USE_TRITON_FLASH_ATTN=1`). The ROCm software stack has matured substantially, and FP8 GEMMs go through hipBLASLt/Composable Kernel, but the ecosystem is younger than CUDA's — pin your versions and test.
- **When you do need multiple MI300X**, remember there is no NVLink. Intra-node Infinity Fabric is fast enough for TP within the 8-GPU baseboard, but you have less headroom than NVLink 4/5, so favor keeping models on one GPU when they fit, and use TP only to reach memory or throughput you cannot get from a single card.
- **The bandwidth is excellent** (5.3 TB/s), so decode is fast; the more common bottleneck is that a single GPU's compute must serve all the concurrency you packed into its large KV budget. Watch GPU utilization: if it saturates before the KV pool fills, you are compute-bound and the fix is another replica, not more batch.

#### Worked example: Llama-3-70B, one MI300X vs two H100s

Serving Llama-3-70B in FP8: on H100s you need either one memory-starved card (~16 concurrent sequences) or two cards in TP for a healthier KV pool. On a single MI300X, the 70 GB of FP8 weights leave over 100 GB for KV — roughly 160+ concurrent sequences at 4k context on *one* GPU, with no TP and no all-reduce. The MI300X delivers the concurrency of a small H100 cluster from a single accelerator, trading NVIDIA's software maturity and NVLink scaling for raw single-GPU capacity. For teams whose models sit in the 30B–140B range and who value operational simplicity, that trade is often worth taking. This single-GPU large-model pattern is the MI300X's signature win, and it is why it shows up in AMD's published Llama-3-70B serving results as a one-GPU deployment where NVIDIA needs two.

## 9. Ada L40S and consumer cards: no NVLink, so do not force TP

The last archetype is the budget tier: NVIDIA's Ada-generation L40S (a 48 GB data-center card) and consumer cards like the RTX 4090 (24 GB). These are defined by one hardware fact that dominates every tuning decision: **no NVLink**. They communicate only over PCIe, at roughly 64 GB/s — about 14× slower than NVLink 4.

The L40S profile:

- **Memory:** 48 GB GDDR6 (not HBM), about 864 GB/s bandwidth — a fraction of the data-center cards' HBM bandwidth.
- **Precision floor:** FP8 (Ada's fourth-generation tensor cores support E4M3/E5M2), BF16, FP16, INT8, INT4. So, unlike the A100, the L40S *does* have FP8 — a genuine advantage for a card in its price class.
- **Interconnect:** PCIe Gen4 only. No NVLink on the L40S; NVLink was removed from the RTX 4090 entirely.

Before the interconnect even enters the picture, note the second hardware fact that shapes L40S serving: it carries GDDR6, not HBM. At about 864 GB/s the L40S has roughly one-quarter the memory bandwidth of an H100 and one-sixth of a B200. Because LLM decode is memory-bound, that bandwidth gap translates almost directly into per-token latency: streaming the same FP8 weights out of memory once takes about four times longer on an L40S than on an H100. The card is not slow at arithmetic — its FP8 tensor cores are genuinely capable — it is slow at *feeding* those tensor cores from memory. That is why the L40S is a throughput-per-dollar card, not a latency card: batch many requests so the expensive weight read amortizes across dozens of sequences and the per-user token cost drops to something competitive, but any single low-batch request will always feel the GDDR6 wall. Size your SLA accordingly — the L40S earns its place on latency-tolerant, high-batch workloads and embarrasses itself on tight-p99 single-stream ones.

The tuning rule writes itself from the interconnect: **never use tensor parallelism on PCIe-only cards for latency-sensitive serving.** The per-layer all-reduce over PCIe, from the equation in section 1, is roughly 14× slower per hop than over NVLink, repeated for every layer of every token. TP on an L40S turns decode into a communication-bound crawl. The two correct patterns are replication and pipeline parallelism.

**Replication (the default).** If your model fits on one card — a 7B–13B model in BF16, or a larger model in FP8/INT4 — run one full copy per GPU and put a load balancer in front. Each request is served entirely on one card; the only cross-GPU traffic is the load balancer's routing, not per-token collectives. This is the economically sweet spot for the L40S: several cheap cards, each a full replica, scaling throughput linearly with no interconnect penalty.

```bash
# L40S 48GB - replicate, do NOT use TP. One full model per GPU, load-balanced.
# 13B in BF16 (~26GB) or a 34B in FP8 (~34GB) fits comfortably on one card.
# Launch one server per GPU on distinct ports; front them with a router.
for GPU in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$GPU \
  vllm serve mistralai/Mistral-Small-24B-Instruct \
    --quantization fp8 \
    --kv-cache-dtype fp8_e4m3 \
    --max-model-len 16384 \
    --max-num-seqs 32 \
    --gpu-memory-utilization 0.90 \
    --port $((8000 + GPU)) &
done
wait
```

The Kubernetes-native version of the same idea is a Deployment with `replicas: 4`, each pod requesting one `nvidia.com/gpu`, behind a Service — the platform load-balances across replicas and there is no TP anywhere:

```yaml
# l40s-replicas.yaml - four independent single-GPU replicas, no cross-GPU comms.
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-l40s
spec:
  replicas: 4
  selector:
    matchLabels: { app: llm-l40s }
  template:
    metadata:
      labels: { app: llm-l40s }
    spec:
      containers:
        - name: vllm
          image: vllm/vllm-openai:latest
          args:
            - "--model=mistralai/Mistral-Small-24B-Instruct"
            - "--quantization=fp8"
            - "--kv-cache-dtype=fp8_e4m3"
            - "--max-num-seqs=32"
          resources:
            limits:
              nvidia.com/gpu: 1        # exactly one GPU per replica
```

**Pipeline parallelism (only when the model is too big for one card).** If you must serve a model that does not fit on a single L40S — say a 70B that needs 2–4 cards — use `--pipeline-parallel-size`, not `--tensor-parallel-size`. Pipeline parallelism splits the model by *layers* across cards and passes only the activation tensor between stages once per micro-batch, communicating orders of magnitude less than TP's per-layer all-reduce. It adds pipeline-fill latency and needs enough in-flight micro-batches to keep all stages busy, so it favors throughput over single-request latency — but over PCIe it is the only distributed option that does not collapse.

```bash
# L40S 48GB x4 - pipeline parallelism for a 70B that won't fit on one card.
# PP passes activations between stages (light traffic), unlike TP's per-layer all-reduce.
vllm serve meta-llama/Meta-Llama-3-70B-Instruct \
  --quantization fp8 \
  --pipeline-parallel-size 4 \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.90 \
  --port 8000
```

**When do these cards make sense?** When cost per token dominates and your latency SLA is lenient. An L40S is a fraction of an H100's price, has FP8, and for small-to-medium models replicated across a few cards it delivers excellent throughput per dollar. It is the wrong choice for a single very large model with a tight p99, and a great choice for a fleet serving many small models or a high-throughput, latency-tolerant workload. The economics are covered in depth in [cost optimization at LLM scale](/blog/machine-learning/model-serving/cost-optimization-at-llm-scale).

#### Worked example: a 13B chat model, an L40S replica fleet versus one H100

A product team serves a 13B instruct model for a chat feature: bursty traffic, a lenient 1.5-second p95, and cost as the hard constraint. On an H100 the model is trivial — 13 GB in FP8 leaves the whole 80 GB card for KV — but you are renting one of the most expensive accelerators made to run a model that uses a fraction of it. The L40S alternative: 13 GB in FP8 fits one 48 GB card with about 30 GB left for KV, so each L40S is a full, independent replica. Four L40S cards, each a replica behind a router, cost meaningfully less than a single H100 while delivering four independent streams of throughput with zero cross-GPU communication. The decode latency per token is higher on each L40S (the GDDR6 wall), but the 1.5-second p95 absorbs it, and the aggregate tokens-per-dollar comfortably beats the single H100. The lesson generalizes: when the model is small, the SLA is loose, and traffic parallelizes across requests, the right tuning move is not a better config on a big card — it is the cheapest card the model fits on, replicated. Force that same 13B into tensor parallelism across the four L40S to "use all the GPUs" and you would wreck the economics and the latency at once; replication is the entire point.

## 10. Same model, five GPUs, five configs

We have now derived, architecture by architecture, that Llama-3-70B demands a different config on each GPU. Here they are side by side — the payoff of the whole post.

![Matrix showing the same Llama-3-70B served on five GPUs with a different precision, GPU count, parallelism scheme, and batching note for each architecture](/imgs/blogs/gpu-architecture-specific-tuning-for-llm-serving-7.webp)

The same 70.6B parameters, one checkpoint, produce five genuinely different deployments:

| GPU | Weight dtype | GPUs needed | Parallelism | KV / batch note |
|---|---|---|---|---|
| H100 80GB | FP8 | 1 (tight) or 2 | TP2 over NVLink 4 | FP8 KV, ~16 seqs on 1 card |
| H200 141GB | FP8 | 1 | none (fits) | FP8 KV, ~110 seqs, long ctx |
| A100 80GB | BF16 (or INT8/AWQ) | 4 (BF16) or 2 (INT8) | TP over NVLink 3 | no FP8, small KV, tight |
| B200 192GB | FP4 (or FP8) | 1 (FP4) | none, or NVL72 EP for MoE | huge KV, compute-bound |
| MI300X 192GB | FP8 | 1 | none (ROCm) | ~160 seqs on 1 card, no TP |
| L40S 48GB | FP8 / INT4 | 2–4 (PP) or replicate | PP not TP (PCIe) | replicate for small models |

Read down the "parallelism" column and the whole thesis is visible: the same model needs 4-way tensor parallelism on A100, 2-way or single-GPU on H100, single-GPU on H200/MI300X, wide expert parallelism on a B200 NVL72 for MoE variants, and pipeline parallelism or replication on L40S. The parallelism scheme is a property of the hardware, derived from capacity and interconnect — never a property you carry over from the last GPU you deployed on.

The qualitative table has a quantitative twin — the same five architectures, but with the numbers the `plan()` function produces for Llama-3-70B at 4k context. This is the table to paste into a capacity review, because it turns "which GPU" into "how many, at what concurrency":

| GPU | Weight dtype | Weight GB | GPUs | Parallelism | ~Concurrent seqs (4k ctx) |
|---|---|---|---|---|---|
| H100 80GB    | FP8  | 70  | 1 | none (tight) | ~16 |
| H200 141GB   | FP8  | 70  | 1 | none         | ~110 |
| A100 80GB    | BF16 | 141 | 4 | TP4          | ~95 (4 cards, BF16 KV) |
| B200 192GB   | FP4  | 35  | 1 | none         | ~210 (FP8 KV) |
| MI300X 192GB | FP8  | 70  | 1 | none         | ~160 |
| L40S 48GB    | FP8  | 70  | 4 | PP4          | ~64 (throughput) |

The concurrency column is the whole post rendered as integers: the identical 70B checkpoint supports roughly sixteen simultaneous users on a lone H100 and over two hundred on a B200 — a thirteen-fold spread driven entirely by capacity and precision, with the model itself held fixed. Two rows deserve a second look. The A100 reaches its ~95 only by spending four cards, and a single H200 nearly matches it on one; that is the A100's whole story in one comparison — no FP8 means more silicon for less concurrency per card. And the L40S is the only row using pipeline parallelism rather than a single card or TP: its number is a throughput figure that carries the single-request latency penalty PP imposes, which is exactly the trade the PCIe archetype forces on you. Read the table as a map from "which GPU" to "how many, at what concurrency," and the config-picker below is that map turned executable.

### The config picker: make the launch a function of the GPU

The durable fix for the 02:14 outage is to stop hard-coding configs and instead *detect* the GPU and emit the right launch arguments. Here is a config-picker that encodes everything above — the precision floor, the capacity-driven shard count, and the interconnect-driven parallelism choice — into one function:

```python
# pick_config.py - emit the right vLLM launch args for the detected GPU.
# Run on the target node; it reads the GPU name and applies the per-arch rules.
import subprocess, math, sys

# (hbm_gb, mem_bw_tbs, has_fp8, has_fp4, has_nvlink)
GPU_SPECS = {
    "H100":   (80,  3.35, True,  False, True),
    "H200":   (141, 4.80, True,  False, True),
    "A100":   (80,  2.00, False, False, True),
    "B200":   (192, 8.00, True,  True,  True),
    "MI300X": (192, 5.30, True,  False, False),  # Infinity Fabric, no NVLink
    "L40S":   (48,  0.86, True,  False, False),  # PCIe only
}

def detect_gpu() -> str:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True) if _has("nvidia-smi") else _rocm_name()
    name = out.strip().splitlines()[0]
    for key in GPU_SPECS:
        if key in name.upper().replace(" ", ""):
            return key
    raise SystemExit(f"unknown GPU: {name}")

def _has(cmd):
    from shutil import which
    return which(cmd) is not None

def _rocm_name():
    return subprocess.check_output(
        ["rocm-smi", "--showproductname"], text=True)

def pick(gpu, params_b=70.6, ctx=8192, kv_per_tok_bytes=163840):
    hbm, bw, fp8, fp4, nvlink = GPU_SPECS[gpu]
    # 1. precision floor -> weight dtype
    if fp4:
        wdtype, wbytes, quant = "fp4", 0.5, "modelopt_fp4"
    elif fp8:
        wdtype, wbytes, quant = "fp8", 1.0, "fp8"
    else:
        wdtype, wbytes, quant = "bf16", 2.0, None   # Ampere: no FP8
    kv_dtype = "fp8_e4m3" if fp8 else "auto"
    # 2. capacity -> shard count
    weight_gb = params_b * wbytes
    usable = hbm * 0.90
    if nvlink:                                       # TP is cheap
        gpus = max(1, math.ceil((weight_gb + 4) / usable))
        parallel = f"--tensor-parallel-size {gpus}"
    else:                                            # PCIe/Infinity Fabric
        if weight_gb <= usable:                      # fits on one card
            gpus, parallel = 1, "--tensor-parallel-size 1"
        else:                                        # too big -> PP, never TP
            gpus = math.ceil((weight_gb + 4) / usable)
            parallel = f"--pipeline-parallel-size {gpus}"
    # 3. KV budget -> max concurrent sequences
    kv_gb = gpus * usable - weight_gb - 3
    per_seq_gb = kv_per_tok_bytes * ctx * (0.5 if kv_dtype != "auto" else 1) / 1e9
    max_seqs = max(1, int(kv_gb / per_seq_gb))
    # 4. flags fall out
    flags = [f"--dtype {'bfloat16' if not fp8 else 'auto'}"]
    if quant: flags.append(f"--quantization {quant}")
    if fp8:   flags.append(f"--kv-cache-dtype {kv_dtype}")
    flags += [parallel, f"--max-num-seqs {max_seqs}",
              f"--max-model-len {ctx}", "--gpu-memory-utilization 0.90"]
    return " \\\n  ".join(["vllm serve $MODEL"] + flags)

if __name__ == "__main__":
    g = sys.argv[1] if len(sys.argv) > 1 else detect_gpu()
    print(f"# detected/selected: {g}\n{pick(g)}")
```

This is the section-1 decision procedure turned into code. It reads the GPU, applies the precision floor, computes the shard count from capacity, picks TP-vs-PP-vs-replicate from the interconnect, and derives `--max-num-seqs` from the KV budget. Wire it into your deployment pipeline and the launch command becomes a pure function of the node it lands on — which is the only way to make a fleet that spans H100, A100, and spot-pool GPUs behave correctly without a human in the loop. It composes naturally with an inference control plane; see [LLM control planes with AIBrix and KServe](/blog/machine-learning/model-serving/llm-control-planes-aibrix-kserve) for routing heterogeneous GPU pools.

## Case studies and benchmarks

Real numbers, honestly framed. Exact figures depend heavily on vLLM version, model, context length, batch size, and measurement methodology, so treat these as representative orders of magnitude and the *directions* as reliable, not the third significant figure.

**FP8 on H100 versus BF16 (the Hopper capacity + throughput win).** vLLM's FP8 support on Hopper consistently shows roughly 1.5–2× throughput improvement over BF16 for large models, driven by both the halved HBM traffic per token and the doubled tensor-core rate, with typical quality loss under a point on standard benchmarks for well-scaled FP8. The larger practical effect is the capacity one from section 4: FP8 frequently collapses a two-GPU deployment to one, which halves cost independently of the throughput gain. This is the most reliable, broadly-applicable tuning win in the whole post.

**H200 memory-bandwidth wins (the same-model, more-concurrency case).** NVIDIA and independent benchmarks report the H200 delivering roughly 1.4–1.9× the inference throughput of an H100 on memory-bound LLM workloads such as Llama-2/3-70B, attributable to the 76% larger memory and 43% higher bandwidth with identical compute. The mechanism is exactly section 5: more HBM becomes more KV cache and thus more concurrent sequences, while more bandwidth accelerates each decode step. The win is largest for high-concurrency and long-context serving and smallest for single-stream, short-context latency tests — which is the expected signature of a memory-driven, not compute-driven, upgrade.

**MI300X single-GPU large-model serving.** AMD's published serving results and the vLLM ROCm ecosystem demonstrate Llama-3-70B (and larger) served on a *single* MI300X, using the 192 GB HBM3 to hold weights plus a large KV cache without any tensor parallelism. The headline is not a raw-throughput crown but an *operational* one: eliminating in-hot-path cross-GPU communication removes a class of distributed failures and simplifies the deployment. The trade is a younger software stack (ROCm/hipBLASLt versus CUDA/cuBLAS) — real, but far narrower than it was two years ago. For models in the 30B–140B range, the single-GPU MI300X is a legitimate, sometimes preferable, alternative to a 2× NVIDIA deployment.

**Blackwell B200 FP4 (the frontier-throughput case).** NVIDIA reports up to 4× H100 inference performance for Blackwell on select LLM workloads, combining FP4's compute doubling, the 2.4× memory and bandwidth, and second-generation Transformer Engine scaling. Frame the "4×" as a best-case, model-and-config-specific vendor figure. The dependable first-principles facts are narrower and universal: FP4 halves weight bytes versus FP8 (35 GB for a 70B), 192 GB is 2.4× the H100's capacity, and the GB200 NVL72's 72-GPU NVLink domain makes rack-scale tensor and expert parallelism run at NVLink bandwidth — the decisive lever for frontier-scale MoE serving. Validate FP4 quality on your own evaluation set before shipping it; four bits is a real accuracy trade, not a free lunch.

**Heterogeneous-fleet spillover (the config picker earning its keep).** A team runs primary traffic on H100s and, during demand spikes, spills overflow onto a cheaper A100 spot pool. Before the config picker, that spill was a hard-coded Hopper launch line that crash-looped on Ampere — the 02:14 outage that opens this post. After: the deployment ran `pick_config.py` detection at pod start, so an H100 pod came up in FP8 single-GPU and an A100 pod came up in BF16 with TP4, from the *same* container image and the *same* model reference. The result was not a throughput record; it was the disappearance of an entire class of pages. Frame the win as availability, not speed — making the launch a pure function of the detected silicon turned a spot-pool spillover from an incident into a routine autoscaling event. Any fleet spanning more than one GPU generation should treat this detection step as non-optional infrastructure, not a nicety.

**L40S replica fleet for a small-model product (throughput per dollar).** A support-assistant feature serving a 24B model in FP8 was benchmarked on a single H100 versus a fleet of L40S replicas. The H100 delivered lower single-request latency, as its HBM bandwidth predicts. But the workload's p95 target was two seconds — well inside what an L40S replica meets — and traffic parallelized cleanly across independent requests. Replicating the model across several L40S cards, each a full copy behind a router with no cross-GPU communication, delivered more total throughput per dollar than the H100 while comfortably holding the SLA. The transferable claim is the archetype rule from section 9, borne out in numbers: for small-to-medium models under a lenient latency target, the PCIe-only budget card wins on cost per token precisely because replication sidesteps its one weakness — the absent interconnect — entirely. The failure mode to avoid is the opposite instinct: do not tensor-parallel a model across L40S cards to chase the H100's single-request latency, or you lose both the latency and the economics.

## A per-GPU measurement matrix

The reference table to keep next to your capacity planner. Bandwidth and capacity are published specifications; the "recommended" columns are the tuning verdicts derived throughout this post. Compute figures are approximate peak dense tensor-core rates and vary by source and sparsity assumptions.

| GPU (arch) | HBM | Bandwidth | Peak tensor (approx) | FP8 | FP4 | Interconnect | Recommended dtype | Parallelism default |
|---|---|---|---|---|---|---|---|---|
| H100 SXM (Hopper) | 80 GB HBM3 | ~3.35 TB/s | ~2.0 PFLOPS FP8 | Yes | No | NVLink 4, 900 GB/s | FP8 weights + FP8 KV | TP within node |
| H200 SXM (Hopper) | 141 GB HBM3e | ~4.8 TB/s | ~2.0 PFLOPS FP8 | Yes | No | NVLink 4, 900 GB/s | FP8, large batch/ctx | often single-GPU |
| A100 SXM (Ampere) | 80 GB HBM2e | ~2.0 TB/s | ~0.6 PFLOPS INT8 | No | No | NVLink 3, 600 GB/s | BF16, or INT8/AWQ | TP within node |
| B200 (Blackwell) | 192 GB HBM3e | ~8.0 TB/s | ~9 PFLOPS FP4 | Yes | Yes | NVLink 5, 1.8 TB/s | FP4 (validate) + FP8 KV | single-GPU or NVL72 EP |
| MI300X (CDNA3) | 192 GB HBM3 | ~5.3 TB/s | ~2.6 PFLOPS FP8 | Yes | No | Infinity Fabric, no NVLink | FP8 weights + FP8 KV | single-GPU preferred |
| L40S (Ada) | 48 GB GDDR6 | ~864 GB/s | ~0.7 PFLOPS FP8 | Yes | No | PCIe Gen4, no NVLink | FP8 / INT4 | replicate; PP if oversized |

A few cross-cutting reads from this table. The two 192 GB cards (B200, MI300X) are the ones that most often let you *stop sharding* — capacity is the lever that removes parallelism. The two Hopper cards share compute and interconnect and differ only in memory, which is why H200 tuning is "same config, bigger batch." The A100 is the only NVLink card here without FP8, which is why it is the one that most often needs more GPUs for the same model. And the L40S is the only card that pairs FP8 with no NVLink — modern precision, budget interconnect — which is exactly why "replicate, do not TP" is its signature rule.

## When to use this (and when not to)

Per-architecture tuning is not free effort; it is a discipline you apply where it pays. Here is where it does and does not.

**Do tune per-architecture when:**

- **You run a heterogeneous fleet.** The moment your traffic can land on more than one GPU type — H100 plus A100 spot, or NVIDIA plus MI300X — a single hard-coded config is a latent outage. Build the config picker.
- **You just upgraded and did not see the gain.** If an H100→H200 or H100→B200 upgrade "didn't help," you almost certainly carried the old config forward and left the new capacity or precision on the floor. Re-derive the config for the new silicon.
- **You are choosing hardware.** The matching of model size to GPU capacity is a purchasing decision as much as a serving one. A 70B model is a single-GPU workload on MI300X/H200/B200 and a multi-GPU workload on A100 — that ratio drives your GPU count and your budget directly.

**Do not over-invest when:**

- **One model, one GPU type, stable fleet.** If everything runs on H100 and always will, derive the config once and move on. There is no per-architecture problem to solve when there is only one architecture.
- **You have not validated the low-precision quality.** Do not ship FP4 (or even aggressive FP8) because the memory math is attractive. The precision floor tells you what the hardware *can* run; your eval set tells you what you *should* run. Measure quality before you chase the byte savings.
- **You would force TP onto PCIe to avoid buying a bigger card.** Tensor parallelism on L40S/4090 to serve a model that does not fit is the classic false economy — you save on hardware and pay it back many times over in decode latency. Either pick a card the model fits on, use pipeline parallelism, or accept replication of a smaller model. Do not force TP across a 64 GB/s link.
- **Your bottleneck is elsewhere.** If your p99 is dominated by a slow retrieval step, a cold cache, or queue depth, per-GPU dtype tuning will not move it. Profile first; tune the binding constraint. [Roofline analysis for LLM inference](/blog/machine-learning/model-serving/roofline-analysis-for-llm-inference) is the tool for finding which constraint that is.

## Key takeaways

- **The serving config is a property of the GPU, not the model.** Precision floor, HBM capacity, and interconnect are the three hardware facts; the dtype, KV budget, batch size, and parallelism plan all derive from them. Copy a config across architectures and you get an outage.
- **Precision floor is silicon, not software.** FP8 needs Hopper or newer; FP4 needs Blackwell; A100 has neither. `--quantization fp8` on an A100 is impossible, not slow.
- **Capacity dictates shard count, and precision changes capacity.** A 70B model is 140 GB in BF16, 70 GB in FP8, 35 GB in FP4 — enough to move the required GPU count from 2 to 1. Precision and parallelism are one linked decision.
- **Interconnect dictates tensor parallelism.** NVLink (600–1800 GB/s) makes TP cheap; PCIe (~64 GB/s) makes it ruinous. On PCIe-only cards, replicate or use pipeline parallelism — never TP for latency-sensitive serving.
- **H200 tuning is H100 tuning with bigger numbers.** Same compute and interconnect, 76% more memory and 43% more bandwidth: raise `--max-num-seqs` and `--max-model-len`, or you waste the upgrade.
- **192 GB cards let you stop sharding.** MI300X and B200 can hold a 70B model plus a large KV cache on one GPU, removing cross-GPU communication and a whole class of distributed failures from the hot path.
- **FP4 is a throughput-corner move, not a default.** It halves bytes again and doubles Blackwell's tensor throughput, but four bits is a real accuracy trade — validate on your eval set and keep sensitive layers and the KV cache in FP8.
- **Make the launch a function of the detected GPU.** The durable fix for cross-architecture outages is a config picker that reads the hardware and emits the flags — precision floor, capacity-driven shards, interconnect-driven parallelism — automatically.

## Further reading

- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023) — the vLLM paper; the KV-cache memory management that makes the capacity math in section 3 actionable.
- NVIDIA, "NVIDIA Hopper Architecture In-Depth" and the H100/H200 datasheets — the FP8 tensor cores, Transformer Engine, TMA, and the memory/bandwidth specs used throughout.
- NVIDIA, "NVIDIA Blackwell Architecture Technical Brief" and the GB200 NVL72 documentation — FP4/MXFP4, second-generation Transformer Engine, and the 72-GPU NVLink domain.
- AMD, "AMD CDNA 3 Architecture" white paper and the ROCm/vLLM documentation — the MI300X 192 GB HBM3 profile and the ROCm serving path.
- Micikevicius et al., "FP8 Formats for Deep Learning" (2022) — the E4M3/E5M2 definitions and the rationale behind Hopper's FP8 datapaths.
- Within this series: [roofline analysis for LLM inference](/blog/machine-learning/model-serving/roofline-analysis-for-llm-inference), [quantization for LLM serving](/blog/machine-learning/model-serving/quantization-for-llm-serving), [FP8 and FP4 low-precision serving](/blog/machine-learning/model-serving/fp8-fp4-low-precision-serving-deep-dive), [tensor, pipeline, and expert parallelism for serving](/blog/machine-learning/model-serving/tensor-pipeline-expert-parallelism-for-serving), [serving 100B+ models across nodes](/blog/machine-learning/model-serving/multi-node-llm-serving-100b-plus), and [cost optimization at LLM scale](/blog/machine-learning/model-serving/cost-optimization-at-llm-scale).

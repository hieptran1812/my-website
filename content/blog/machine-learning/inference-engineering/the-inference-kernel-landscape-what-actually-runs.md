---
title: "The inference kernel landscape: what actually runs on the GPU"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Open the black box of a single decode step, enumerate the ten kernels that actually run per layer, derive which ones own your latency, and learn to read a profile so you can name the wall instead of guessing at it."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "cuda",
    "gpu",
    "kernels",
    "roofline",
    "kv-cache",
    "pytorch",
    "latency",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 46
---

For twenty-one posts this series has treated the forward pass as a box that eats token ids and produces logits. We put a KV cache in front of it, a scheduler above it, a sampler behind it, a tokenizer on either side — and never once asked what happens *inside* the box during the microseconds it is running. This post opens the box.

Here is the thing the box hides. When your engine generates one token of Llama-3.1-8B, it does not run "the model." It launches a list of small GPU programs — a normalization here, a matrix multiply there, a rotation of the query vector, an attention kernel, three more matrix multiplies for the MLP, a couple of elementwise adds — roughly **ten kernels per layer, thirty-two layers, plus a final norm and the vocabulary projection**. That is on the order of **330 kernel launches to produce a single token**. Every one of them reads something from the GPU's main memory, does a little arithmetic, and writes something back. Which of those reads and writes dominates your latency is not a matter of opinion, and it is not the kernel you would guess. On an A100 at batch 1, the arithmetic across all 330 kernels takes about **0.05 ms**; the step itself takes about **7.9 ms**. The GPU spends more than 99% of the step waiting on memory, and the tensor cores — the thing you paid for — sit idle almost the entire time.

![A single decode step fanning out into the ten kernels that run per layer, from the input norm through the attention branch and the MLP branch to the sampler](/imgs/blogs/the-inference-kernel-landscape-what-actually-runs-1.webp)

This is the map the whole of Track E navigates by. If you cannot say, for your model on your GPU, which kernel is the wall and why, then every kernel you are about to write — the fused RMSNorm, the KV-append, the paged-attention kernel, the dequant-fused GEMM — is a shot in the dark. So before we write a single line of CUDA, we spend one post learning to read the terrain: take one decode step apart, put a rough FLOP and byte cost on each kernel, place each of them on the roofline, and derive from first principles where the time actually goes. By the end you will have `nanoserve/kernel_budget.py` — an analytical model that predicts the per-kernel budget from `model.config` alone — and `nanoserve/profile_step.py`, which instruments a real step so you can check the model against a trace. And you will be able to look at an `nsys` timeline and say "that white space is launch overhead" or "that kernel is bandwidth-bound" instead of squinting.

One promise, restated from [the series introduction](/blog/machine-learning/inference-engineering/what-inference-engineering-is): **I have no GPU and I have run none of this.** Every number below is derived from arithmetic I show you, cited from a vendor spec or a paper with a link, or framed as a range you should expect when you run the script yourself. The results table carries a `Source` column for exactly that reason. A kernel-landscape post that invented its measurements would be worse than useless — it would teach you to trust the wrong wall.

---

## 1. What "a kernel" is, and why the forward pass is a list of them

A **kernel** is a single GPU program launched from the CPU (the *host*) to run on the GPU (the *device*). You call it once; it spawns thousands of threads that all run the same code on different data, and it runs to completion before the next one starts (within a stream). PyTorch does not have a "run the transformer" kernel. It has a `layer_norm` kernel, a `matmul` kernel (which dispatches into cuBLAS), an elementwise `mul` kernel, an attention kernel, and so on. When you call `model(input_ids)`, the Python code walks the module tree and each operation it hits enqueues one or more kernels onto the GPU's work queue.

So the forward pass is not one thing. It is a **sequence of kernel launches**, and the shape of that sequence is fixed by the architecture. For a Llama-style decoder layer the per-layer sequence is, in order:

1. **Input RMSNorm** — normalize the hidden state before attention.
2. **QKV projection** — one (usually fused) matrix multiply that produces the query, key, and value vectors.
3. **RoPE** — rotate the query and key vectors by their position.
4. **KV-cache write** — append the new key and value into the paged cache (`reshape_and_cache`).
5. **Attention** — the query attends over all cached keys and values (the paged-attention kernel).
6. **Output projection** — a matrix multiply mapping the attention output back to the hidden size.
7. **Residual add** — add the attention output to the pre-attention hidden state.
8. **Post-attention RMSNorm** — normalize before the MLP.
9. **MLP gate + up projections** — two matrix multiplies (SwiGLU has two input projections).
10. **SwiGLU activation** — `silu(gate) * up`, elementwise.
11. **MLP down projection** — a matrix multiply back to the hidden size.
12. **Residual add** — add the MLP output back.

Call it ten to twelve kernels, depending on how aggressively your framework fuses. After the last layer there is a **final RMSNorm**, the **lm_head** projection (hidden size to vocabulary), and then the **sampler** turns logits into a token id. Multiply the per-layer count by 32 layers and you get the ~330 launches per token from the intro. That number is not incidental; it comes back to bite us in section 6.

![The ordered sequence of kernels inside one decoder layer, from the input norm through the attention block to the MLP block and the two residual adds](/imgs/blogs/the-inference-kernel-landscape-what-actually-runs-2.webp)

Notice that most of the twelve are *tiny* — the two norms, the RoPE rotation, the KV-cache write, the SwiGLU activation, the two residual adds are each a few kilobytes of traffic and microseconds of work. Only five of them are large: the QKV projection, the output projection, and the three MLP projections. The tiny kernels do almost no useful work, but each one still costs a full launch and a full HBM round-trip of the activation vector — which is exactly why the fusion opportunities in section 9 target the small kernels, not the big ones. The big kernels are already at their bandwidth floor; the small ones are pure overhead you can make disappear.

The reason this matters is that **each kernel is an independent trip to memory**. RMSNorm reads the hidden vector, does its arithmetic, writes a normalized vector back to HBM. The QKV projection then reads that vector back, multiplies it by a weight matrix it also reads from HBM, and writes three vectors out. Nothing stays on-chip between kernels — the GPU's registers and shared memory are scratch space that evaporates when the kernel exits. So a decode step is, physically, a *chain of loads and stores against HBM* with a little arithmetic sprinkled on top. That framing is the whole post. Everything else is putting numbers on it.

> A note on terminology I will use without apology from here on. **HBM** is the GPU's high-bandwidth main memory — the 24/80/... gigabytes on the card, fast by CPU standards (hundreds to thousands of GB/s) but glacial compared to on-chip SRAM. **Arithmetic intensity** (AI) is the ratio of floating-point operations to bytes moved: FLOPs divided by bytes. **Compute-bound** means the kernel's time is set by how fast the arithmetic units run; **memory-bound** (or bandwidth-bound) means it is set by how fast HBM can feed them. The [memory-hierarchy post](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm) develops these from the metal up; I lean on them here.

---

## 2. The one split that explains everything: prefill is a GEMM, decode is a GEMV

Every performance fact in LLM inference descends from a single distinction, and we introduced it back in [the naive-decode-loop post](/blog/machine-learning/inference-engineering/the-naive-decode-loop-and-your-first-baseline) as a floor. Here we make it quantitative at the level of individual kernels.

**Prefill** processes the whole prompt at once. If the prompt is 1,000 tokens, the QKV projection is a multiply of a `[1000, 4096]` activation matrix by a `[4096, 6144]` weight matrix — a genuine **matrix-matrix multiply**, a GEMM (General Matrix Multiply). The weight matrix is read once from HBM and *reused across all 1,000 rows*. That reuse is the whole game: the arithmetic scales with the number of rows while the weight bytes stay fixed, so the arithmetic intensity climbs into the hundreds and the kernel becomes **compute-bound**. The tensor cores light up. This is the regime GPUs are designed for.

**Decode** processes one token at a time. The same QKV projection is now a multiply of a `[1, 4096]` activation vector by the `[4096, 6144]` weight matrix — a **matrix-vector multiply**, a GEMV (General Matrix-Vector multiply). The weight matrix is still read in full from HBM, but now it is used *once*: one row of output per weight element loaded. The arithmetic intensity collapses to roughly one FLOP per byte, and the kernel becomes **memory-bound**. The tensor cores are starved — there is nothing to reuse, so they wait on HBM. This is the regime GPUs are worst at, and it is the regime decode lives in.

![Prefill runs matrix-matrix multiplies that reuse each weight across many rows and saturate the tensor cores, while decode runs matrix-vector multiplies that read each weight once and stall on memory](/imgs/blogs/the-inference-kernel-landscape-what-actually-runs-6.webp)

Put a number on the arithmetic intensity of a decode GEMM. A projection `[1, K] × [K, N]` does $2 \cdot K \cdot N$ FLOPs (a multiply and an add per weight element) and reads $2 \cdot K \cdot N$ bytes of weight in bf16 (two bytes each). So:

$$\text{AI}_{\text{decode GEMM}} = \frac{2KN}{2KN + \text{small}} \approx 1 \text{ FLOP/byte}.$$

One. Now compare that to the **machine balance** of the GPU — the ratio of its peak compute to its peak bandwidth, which is where the roofline's slanted memory line meets its flat compute ceiling. NVIDIA's [A100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf) lists 312 TFLOP/s of bf16 tensor-core throughput and the 80GB SXM part carries 2,039 GB/s of HBM2e bandwidth, so:

$$\text{ridge}_{\text{A100}} = \frac{312 \times 10^{12}}{2039 \times 10^{9}} \approx 153 \text{ FLOP/byte}.$$

A decode GEMM sits at AI $\approx 1$ and the A100 does not become compute-bound until AI $\approx 153$. Decode is more than **two orders of magnitude** below the ridge. That is not a small inefficiency you can tune away with better kernels; it is a structural property of generating one token at a time, and the [roofline post](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) is the reference for why the ridge is where it is. Every technique in this series — batching, paging, quantization, speculative decoding — is, at bottom, an attempt to raise that arithmetic intensity or reduce those bytes.

The same ridge for the other GPUs in our matrix, from their datasheets: the RTX 4090 (1,008 GB/s, ~165 TFLOP/s bf16 with FP32 accumulate) lands near 164 FLOP/byte; the H100 SXM ([datasheet](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c): 3,350 GB/s HBM3, 989.5 TFLOP/s bf16) near 295. Faster cards have *higher* ridges, which means a batch-1 decode leaves an even larger fraction of an H100 idle than of a 4090. The expensive GPU is the one you waste hardest at batch 1.

---

## 3. Where the bytes actually go in one decode step

If decode is memory-bound, then the kernel that moves the most bytes owns the most time. So let us account for every byte a single decode step moves, for Llama-3.1-8B at batch 1. We need the architecture, which the [model card](https://huggingface.co/meta-llama/Llama-3.1-8B) gives and `model.config` exposes: 32 layers, hidden size 4,096, 32 query heads and 8 key/value heads (grouped-query attention), head dim 128, MLP intermediate size 14,336, vocabulary 128,256, weights in bf16.

**Weights.** Every weight is read exactly once per decode step, because decode is a GEMV and there is nothing to reuse. So the weight traffic equals the model size: 8.03 billion parameters at 2 bytes each is **16.1 GB**. Break it down per layer:

- QKV projection: $4096 \times (4096 + 1024 + 1024) = 4096 \times 6144$ = 25.2M params → 50.3 MB.
- Output projection: $4096 \times 4096$ = 16.8M → 33.6 MB.
- MLP gate: $4096 \times 14336$ = 58.7M → 117.4 MB.
- MLP up: same, 117.4 MB.
- MLP down: $14336 \times 4096$ = 58.7M → 117.4 MB.

That is **436 MB of weights per layer**, and $436 \times 32 = 13.96$ GB across the layers. Add the **lm_head** ($128256 \times 4096 \times 2$ = 1.05 GB) and the norm weights (negligible), and you are at ~15 GB, the rest being the embedding table and rounding — call it the full 16.1 GB. Notice the split: the **MLP is 70% of the weight traffic** (three big projections per layer), attention projections are ~17%, and the lm_head is ~6.5%. The kernel that owns your decode step is not the attention kernel everyone talks about — it is the **MLP down projection**, three of them, thirty-two times over.

**KV cache.** The attention kernel reads the cached keys and values for every position in the context. The bytes per token of context, derived once and reused across the whole series:

$$\text{KV bytes/token} = 2 \cdot L \cdot H_{kv} \cdot d \cdot b = 2 \cdot 32 \cdot 8 \cdot 128 \cdot 2 = 131{,}072 \text{ bytes} = 128 \text{ KB}.$$

The leading 2 is for K and V; $L$ is layers, $H_{kv}$ is KV heads (8, not 32 — that is what GQA buys you), $d$ is head dim, $b$ is bytes per element. So at a context of 2,048 tokens the attention kernels read $128 \text{ KB} \times 2048 = 256$ MB across all layers. Against 16.1 GB of weights, that is **1.7% of the step**. At batch 1 and moderate context, attention is a rounding error on the byte budget. (This flips hard at long context and at large batch — sections 7 and 8.)

**Activations.** The hidden vector is 4,096 elements — 8 KB in bf16. It gets read and rewritten a handful of times per layer. Total activation traffic per step is a few hundred KB. Negligible.

![The HBM traffic budget for one batch-1 decode step, showing weights dominating at over ninety percent while the KV cache and activations are rounding errors](/imgs/blogs/the-inference-kernel-landscape-what-actually-runs-3.webp)

So the byte budget is: **~16 GB weights, ~0.26 GB KV cache, ~0.001 GB activations**. Divide the total by bandwidth to get the floor:

$$t_{\text{step}} \gtrsim \frac{16.1 \text{ GB}}{2039 \text{ GB/s}} \approx 7.9 \text{ ms}.$$

That is the A100 batch-1 decode floor — the same 7.9 ms we derived in the baseline post, now decomposed into which kernels contribute it. And the arithmetic? The whole step does about 16.1 GFLOP (2 FLOPs per parameter used, plus a gigaflop of attention). At 312 TFLOP/s that is $16.1 \times 10^9 / 312 \times 10^{12} = 0.052$ ms. **Compute is 0.65% of the memory time.** The tensor cores are idle for 99.35% of the step. That single ratio is the reason this series exists.

#### Worked example: the decode-step kernel budget for Llama-3.1-8B on an A100

Here is the full per-kernel budget for one batch-1 decode step at 2,048 tokens of context. Because the step is memory-bound, the "% of step" column is allocated by bytes moved (time is proportional to HBM traffic). Every row is derived from the arithmetic above; nothing here was measured.

| Kernel (×count/step) | FLOPs | HBM bytes | AI (FLOP/byte) | Bound | % of step |
| --- | --- | --- | --- | --- | --- |
| RMSNorm (×64) | ~1.3 M each | ~24 KB each | ~0.5 | memory | ~0.3% |
| QKV projection GEMM (×32) | 50.3 M each | 50.3 MB each | ~1.0 | memory | ~10% |
| RoPE (×32) | ~40 K each | ~20 KB each | ~1 | memory | ~0.1% |
| KV-cache write (×32) | none | ~4 KB each | 0 | memory | <0.1% |
| Paged attention (×32) | 33.6 M each | 8.4 MB each | ~4.0 | memory | ~1.7% |
| Output projection GEMM (×32) | 33.6 M each | 33.6 MB each | ~1.0 | memory | ~6.7% |
| MLP gate + up GEMM (×64) | 117.4 M each | 117.4 MB each | ~1.0 | memory | ~47% |
| SwiGLU activation (×32) | ~86 K each | ~86 KB each | ~1 | memory | ~0.2% |
| MLP down GEMM (×32) | 117.4 M each | 117.4 MB each | ~1.0 | memory | ~23% |
| Residual add (×64) | ~4 K each | ~24 KB each | ~0.2 | memory | ~0.3% |
| Final norm + lm_head (×1) | 1.05 G | 1.05 GB | ~1.0 | memory | ~6.5% |
| Sampler (×1) | ~0.5 M | ~1 MB | ~0.5 | memory | <0.1% |
| **Total** | **~16.1 GFLOP** | **~16.1 GB** | **~1.0** | **memory** | **100%** |
| | | | | | *Source: derived* |

Two things jump out. First, **every single kernel is memory-bound** — there is not one compute-bound kernel in a batch-1 decode step, which is why "the tensor cores are idle" is not hyperbole. Second, the MLP GEMMs (gate + up + down) are **70% of the step**, the attention *projections* are ~17%, and the actual attention kernel — the one this track spends three posts writing — is **1.7%** at this context length. Attention is not where your batch-1 decode latency lives. Weights are. Keep that in your pocket; it will save you from optimizing the wrong kernel for a week.

---

## 4. Putting the kernels on the roofline

The roofline is the one picture that makes "compute-bound vs memory-bound" visual instead of a slogan. The x-axis is arithmetic intensity; the y-axis is achieved FLOP/s. A slanted line rises from the origin — that is the bandwidth limit, `AI × bandwidth`. It hits a horizontal ceiling — the peak compute. Where they meet is the ridge. A kernel is a point: if it sits under the slanted part, it is memory-bound and bandwidth is the wall; if it sits under the flat part, it is compute-bound and the arithmetic units are the wall.

Plot our kernels. Every decode kernel we just budgeted has AI $\approx 1$, and the A100 ridge is at 153. So **every decode kernel is a point crammed against the far-left of the roofline, deep under the slanted line** — pinned to the bandwidth limit, achieving roughly `1 × 2039 = 2` GFLOP/s out of a possible 312,000. Now plot the *prefill* version of the same QKV kernel: at a 1,000-token prompt its AI is in the hundreds, so it sits up near the compute ceiling, achieving something close to peak. **Same weights, same kernel, same GPU — the only thing that changed is whether one row or a thousand rows share the weight read**, and that alone moves the point from the bottom-left floor to the top-right ceiling.

![The RTX 4090, A100, and H100 compared by memory bandwidth, peak throughput, and the arithmetic intensity of their roofline ridge, each ridge landing far above the batch-one decode point](/imgs/blogs/the-inference-kernel-landscape-what-actually-runs-7.webp)

This is why "batch it" is the first and most important optimization in the whole series, and why it is *free* until memory runs out. Adding a second request to a decode step does not read the weights a second time — it rides along on the same 16.1 GB read, doing a second row of arithmetic on data already in flight. The FLOPs double, the bytes barely move, the arithmetic intensity climbs. You are walking the point up the slanted line toward the ridge, converting idle tensor-core time into throughput at almost no latency cost. The only thing that stops you is the KV cache filling VRAM — which is the entire subject of the KV-cache and paging posts. Batching is the lever; paging is what lets you pull it.

The batch at which a decode step finally reaches the ridge follows directly. The step's arithmetic intensity scales with batch (FLOPs grow with batch, weight bytes stay fixed), so `AI(batch) ≈ batch × 1`. Set that equal to the ridge:

- A100: crosses at batch ≈ 153.
- RTX 4090: batch ≈ 164.
- H100: batch ≈ 295.

Below those batch sizes you are on the slanted line (bandwidth-bound); above them you are under the ceiling (compute-bound). The full four-GPU version of this — decode floors and ceilings per card — is the table in [the baseline post](/blog/machine-learning/inference-engineering/the-naive-decode-loop-and-your-first-baseline); it and this post are two views of the same arithmetic.

---

## 5. The implementation: an analytical budget you can run, and a profiler you can check it against

Enough prose. Let us build the two tools that make all of this concrete in `nanoserve`. The first is an **analytical model** that reads `model.config` and prints the byte and FLOP budget — the table from section 3, computed rather than hand-derived. This is the honest kind of number: it is a formula you can inspect, not a measurement I am asking you to trust.

```python
# nanoserve/kernel_budget.py
from dataclasses import dataclass


@dataclass
class ModelDims:
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int          # MLP intermediate size
    vocab: int
    dtype_bytes: int = 2   # bf16

    @classmethod
    def from_hf_config(cls, cfg):
        return cls(
            n_layers=cfg.num_hidden_layers,
            d_model=cfg.hidden_size,
            n_heads=cfg.num_attention_heads,
            n_kv_heads=cfg.num_key_value_heads,
            head_dim=getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads),
            d_ff=cfg.intermediate_size,
            vocab=cfg.vocab_size,
        )


def decode_step_budget(m: ModelDims, seq_len: int, batch: int = 1):
    """Bytes and FLOPs for ONE decode step. Everything derived, nothing measured."""
    b = m.dtype_bytes
    qkv_out = (m.n_heads + 2 * m.n_kv_heads) * m.head_dim   # fused QKV width

    # weight bytes per layer (read once per step at any batch -- shared across the batch)
    w = {
        "qkv":  m.d_model * qkv_out * b,
        "o":    m.d_model * m.d_model * b,
        "gate": m.d_model * m.d_ff * b,
        "up":   m.d_model * m.d_ff * b,
        "down": m.d_ff * m.d_model * b,
    }
    weight_bytes = sum(w.values()) * m.n_layers + m.vocab * m.d_model * b  # + lm_head

    # KV cache bytes read by attention: per token of context, per request in the batch
    kv_bytes = 2 * m.n_layers * m.n_kv_heads * m.head_dim * b * seq_len * batch

    # FLOPs: 2 per weight element per row (batch rows), plus attention
    matmul_flops = 2 * (sum(w.values()) // b * m.n_layers + m.vocab * m.d_model) * batch
    attn_flops = 4 * m.n_heads * m.head_dim * seq_len * m.n_layers * batch

    return {
        "weight_bytes": weight_bytes,      # constant in batch
        "kv_bytes": kv_bytes,              # scales with batch AND context
        "total_bytes": weight_bytes + kv_bytes,
        "total_flops": matmul_flops + attn_flops,
        "arithmetic_intensity": (matmul_flops + attn_flops) / (weight_bytes + kv_bytes),
    }
```

Feed it Llama-3.1-8B's dimensions and it reproduces section 3:

```python
llama = ModelDims(n_layers=32, d_model=4096, n_heads=32, n_kv_heads=8,
                  head_dim=128, d_ff=14336, vocab=128256)

b = decode_step_budget(llama, seq_len=2048, batch=1)
print(f"weights : {b['weight_bytes']/1e9:6.2f} GB")
print(f"kv      : {b['kv_bytes']/1e9:6.3f} GB")
print(f"flops   : {b['total_flops']/1e9:6.2f} GFLOP")
print(f"AI      : {b['arithmetic_intensity']:6.2f} FLOP/byte")
```

```console
weights :  16.06 GB
kv      :  0.268 GB
flops   :  16.06 GFLOP
AI      :   0.98 FLOP/byte
```

There is the whole story in four lines of output: 16 GB of weights, a quarter-gigabyte of KV, and an arithmetic intensity just under 1. Turn that into a latency floor by dividing by a GPU's bandwidth:

```python
def decode_floor_ms(budget, hbm_gbps):
    """Lower bound on step time = bytes / bandwidth. Ignores overhead (see s.6)."""
    return budget["total_bytes"] / (hbm_gbps * 1e9) * 1e3

for name, bw in [("RTX 4090", 1008), ("L4", 300), ("A100", 2039), ("H100", 3350)]:
    print(f"{name:9s}: {decode_floor_ms(b, bw):5.1f} ms/step  ->  {1000/decode_floor_ms(b, bw):5.0f} tok/s")
```

```console
RTX 4090 :  16.2 ms/step  ->    62 tok/s
L4       :  54.4 ms/step  ->    18 tok/s
A100     :   8.0 ms/step  ->   125 tok/s
H100     :   4.9 ms/step  ->   205 tok/s
```

These are *floors* — the best you could do if memory bandwidth were the only cost and every byte moved at peak. Real numbers are worse because of overheads the model ignores, and the gap between this floor and reality is itself diagnostic (section 6). The bandwidth figures are cited from each card's datasheet; the L4's 300 GB/s is from [NVIDIA's L4 spec](https://www.nvidia.com/en-us/data-center/l4/).

Now the second tool: a profiler that runs a real step and tells you the truth. `torch.profiler` gives you per-kernel timings without leaving Python, and it is the honest way to check the analytical model against a real device.

```python
# nanoserve/profile_step.py
import torch
from torch.profiler import profile, ProfilerActivity


@torch.inference_mode()
def profile_one_decode_step(model, input_ids, kv_cache):
    """Profile a single decode step. Warm up first, sync, then capture ONE step."""
    # 1. Warmup -- the first launches trigger allocation, autotuning, cuBLAS handle
    #    creation. Timing them measures the framework, not the model. Always discard.
    for _ in range(5):
        _ = model.decode_step(input_ids, kv_cache)
    torch.cuda.synchronize()

    # 2. Capture one steady-state step with kernel-level detail.
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True) as prof:
        _ = model.decode_step(input_ids, kv_cache)
        torch.cuda.synchronize()   # make sure the GPU work finished before we stop

    # 3. Sort by total GPU time, not by call -- a kernel called 32x with a small
    #    per-call time can still be the wall.
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    return prof
```

The self-timing rules baked into that snippet are the ones that separate a real number from a fantasy, and they are worth stating plainly because everyone gets them wrong the first time: **warm up before timing** (the first few steps pay for allocation and cuBLAS autotuning and would triple your average); **synchronize before you stop the timer** (kernel launches are asynchronous — the CPU races ahead of the GPU, so without a `torch.cuda.synchronize()` you are timing the *launch*, not the *work*); and **sort by total time, not by count**. That last one catches people: the MLP down projection is called 32 times with a modest per-call cost, and it is the single biggest consumer of your step — but a naive profile sorted alphabetically or by call count buries it. The [reproducible-benchmark post](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) is the long version of this discipline.

If you run that profiler on an 8B model, here is what to expect (**reproduce it — do not take my word**): the GEMM kernels (which show up as `ampere_...` or `cutlass_...` or `sm90_...` names depending on your architecture) dominate the CUDA-time column, the MLP GEMMs above the attention projections above the lm_head, and the attention kernel sits well down the list at short context. If instead the top of your list is dominated by tiny elementwise kernels or the total GPU time is a fraction of your wall-clock step time, you are not memory-bound — you are **launch-bound**, which is the next section.

---

## 6. Kernel launch overhead: the wall that is not on the GPU at all

Everything so far assumed the GPU is the bottleneck. Often it is not. Recall the count: ~10 kernels per layer, 32 layers, plus the head — about **330 kernel launches per token**. Each launch is a small amount of CPU work: build the launch descriptor, push it onto the stream, let the driver hand it to the GPU. That costs a few microseconds of *host* time per launch. Empirically it lands around 5 µs per launch on a typical driver — NVIDIA's [CUDA Graphs documentation](https://developer.nvidia.com/blog/cuda-graphs/) motivates the entire feature by exactly this per-launch CPU cost, and it is the number to reproduce with `nsys` on your own box.

Do the arithmetic. If each launch costs $L$ microseconds and a step issues $K$ launches, the CPU needs $K \cdot L$ microseconds just to *submit* the work:

$$t_{\text{launch}} = K \cdot L = 330 \times 5\,\mu s = 1.65 \text{ ms}.$$

Now compare that to the GPU-side floor. On an A100 the step takes 7.9 ms of GPU work, and the CPU can submit all 330 launches in 1.65 ms — comfortably ahead, so the GPU stays fed and launch overhead hides completely behind the compute. But watch what happens as the GPU gets faster or the model gets smaller:

$$\text{launch-bound when } \quad t_{\text{launch}} > t_{\text{GPU}} \quad \Longleftrightarrow \quad K \cdot L > \frac{\text{weight bytes}}{\text{bandwidth}}.$$

On an H100 the same 8B step is only 4.9 ms of GPU work; 1.65 ms of launch time is now a third of it, and any hiccup in the launch loop (a Python GIL stall, a host-side sync, a slow tokenizer callback) leaves the GPU idle between kernels. On a *small* model the regime flips entirely. Take Llama-3.2-1B: 16 layers, ~2.5 GB in bf16. Its GPU floor on an A100 is ${2.5 / 2039 = 1.2}$ ms, but it still issues ~160 launches, costing $160 \times 5\,\mu s = 0.8$ ms of pure CPU submission. Now launch overhead is **67% of the GPU floor** — the CPU cannot launch kernels fast enough to keep the GPU busy, and the GPU spends much of the step idling between one kernel finishing and the next arriving. The model is **launch-bound**, and no amount of kernel optimization helps, because the kernels are not the problem — the *gaps between them* are.

<figure class="blog-anim">
<svg viewBox="0 0 760 300" role="img" aria-label="Two decode timelines under one wall clock: a launch-bound lane with idle gaps between kernels finishes late, while a CUDA-graph lane packs the same kernels contiguously and finishes early" style="width:100%;height:auto;max-width:860px">
<title>Launch-bound decode versus a CUDA-graph replay under the same wall clock</title>
<style>
.k1-track{fill:none;stroke:var(--border,#d1d5db);stroke-width:1.5}
.k1-kern{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.k1-busy{fill:var(--accent,#6366f1);opacity:.85}
.k1-gap{fill:var(--text-secondary,#6b7280);opacity:.14}
.k1-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.k1-sub{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.k1-tag{font:600 12px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.k1-head{stroke:var(--accent,#6366f1);stroke-width:2.5}
@keyframes k1-sweep{0%{transform:translateX(0)}92%,100%{transform:translateX(660px)}}
@keyframes k1-idle{0%,45%{opacity:0}55%,88%{opacity:.9}92%,100%{opacity:0}}
@keyframes k1-done{0%,58%{opacity:0}66%,88%{opacity:.9}92%,100%{opacity:0}}
.k1-play{animation:k1-sweep 9s linear infinite}
.k1-idletag{animation:k1-idle 9s ease-in-out infinite}
.k1-donetag{animation:k1-done 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.k1-play{animation:none;transform:translateX(340px)}.k1-idletag{animation:none;opacity:.9}.k1-donetag{animation:none;opacity:.9}}
</style>
<text class="k1-lbl" x="20" y="34">Launch-bound (one kernel launched at a time)</text>
<line class="k1-track" x1="20" y1="96" x2="700" y2="96"/>
<rect class="k1-busy" x="30"  y="66" width="46" height="46" rx="5"/>
<rect class="k1-busy" x="112" y="66" width="46" height="46" rx="5"/>
<rect class="k1-busy" x="194" y="66" width="46" height="46" rx="5"/>
<rect class="k1-busy" x="276" y="66" width="46" height="46" rx="5"/>
<rect class="k1-busy" x="358" y="66" width="46" height="46" rx="5"/>
<rect class="k1-gap" x="76"  y="66" width="36" height="46" rx="5"/>
<rect class="k1-gap" x="158" y="66" width="36" height="46" rx="5"/>
<rect class="k1-gap" x="240" y="66" width="36" height="46" rx="5"/>
<rect class="k1-gap" x="322" y="66" width="36" height="46" rx="5"/>
<text class="k1-sub" x="30" y="132">GPU idle in every grey gap while the CPU builds the next launch</text>
<text class="k1-lbl" x="20" y="188">CUDA-graph replay (kernels packed, launched as one)</text>
<line class="k1-track" x1="20" y1="250" x2="700" y2="250"/>
<rect class="k1-busy" x="30"  y="220" width="46" height="46" rx="5"/>
<rect class="k1-busy" x="78"  y="220" width="46" height="46" rx="5"/>
<rect class="k1-busy" x="126" y="220" width="46" height="46" rx="5"/>
<rect class="k1-busy" x="174" y="220" width="46" height="46" rx="5"/>
<rect class="k1-busy" x="222" y="220" width="46" height="46" rx="5"/>
<line class="k1-head k1-play" x1="30" y1="54" x2="30" y2="278"/>
<text class="k1-tag k1-idletag" x="404" y="58">same work, still running</text>
<text class="k1-tag k1-donetag" x="300" y="212">already done</text>
</svg>
<figcaption>Under one wall clock the launch-bound lane spends real time in the grey idle gaps and finishes late; packing the identical kernels into a single CUDA-graph replay removes the gaps and the step finishes early.</figcaption>
</figure>

The fix is not to write faster kernels — it is to stop launching them one at a time. **CUDA graphs** capture the entire sequence of ~330 launches once and replay it as a single unit, collapsing 1.65 ms of per-launch CPU work into one graph launch. `torch.compile` does the adjacent job of *fusing* small kernels so there are fewer to launch in the first place. Both are the subject of [the CUDA-graphs-and-torch.compile post](/blog/machine-learning/inference-engineering/cuda-graphs-and-torch-compile-for-the-decode-loop) later in this track; for now the point is diagnostic. The production engines already do this: the vLLM team's [Model Runner V2 write-up](https://vllm.ai/blog/2026-03-24-mrv2) (2026-03-24) describes building the step's input tensors (`input_ids`, `positions`, `query_start_loc`, `seq_lens`) *on the GPU* via Triton kernels and a "zero-sync async scheduling" design where the CPU schedules step N+1 while the GPU runs step N — precisely to keep the launch loop from ever becoming the wall. They report a 56% throughput gain (25K vs 16K output tok/s) on a tiny Qwen3-0.6B on one GB200, a deliberately host-overhead-dominated stress case. That is the launch-bound regime, measured by someone with the hardware, confirming the arithmetic above.

#### Worked example: is a tiny model on a fast GPU launch-bound?

Take Llama-3.2-1B (16 layers, ~2.5 GB) on an H100 (3,350 GB/s). GPU floor: ${2.5 / 3350 = 0.75}$ ms. Launches: $16 \times 10 + 3 \approx 163$, at 5 µs each = 0.82 ms of CPU submission. **The launch time exceeds the GPU floor.** Without CUDA graphs this model cannot saturate an H100 no matter how good its kernels are — the CPU is the bottleneck. `Source: derived` (launch count from architecture; per-launch 5 µs cited from NVIDIA's CUDA-Graphs blog; verify both with `nsys`). This is the single clearest case where reading the profile beats reasoning about FLOPs: the FLOP budget says "trivial," the wall-clock says "slow," and the reconciliation is the white space between kernels.

---

## 7. Reading a profile honestly: naming the wall

You now have three candidate walls — bandwidth, launch overhead, and (at long context or large batch) the attention kernel. The tooling to tell them apart is two NVIDIA profilers, and the honest workflow is to use them in order.

**`nsys` (Nsight Systems) — the timeline.** This is the wide-angle view: a horizontal timeline of every kernel on the GPU and every launch on the CPU, aligned on one clock. You are looking for one thing above all: **white space on the GPU row**. A dense, gapless GPU row means the GPU is fed and you are compute- or bandwidth-bound — optimize kernels or batch harder. Visible gaps between kernels, with CPU activity in them, means the GPU is *waiting for the CPU to launch the next kernel* — you are launch-bound, and the fix is CUDA graphs, not kernel tuning. The [Nsight Systems for AI services post](/blog/machine-learning/performance-engineering/nsight-systems-for-ai-services) walks a real capture; the short version is that the timeline answers "is the GPU even busy?" before you ask any harder question.

![A decision tree for naming the decode wall by reading a profile, branching from step time to bandwidth, launch overhead, and the long-context attention regime](/imgs/blogs/the-inference-kernel-landscape-what-actually-runs-4.webp)

**`ncu` (Nsight Compute) — one kernel's Speed-of-Light.** Once the timeline tells you the GPU is busy and you want to know *why a specific kernel is slow*, `ncu` profiles a single kernel and reports its "Speed-of-Light" (SOL): the percentage of peak compute and the percentage of peak memory bandwidth it achieved. For a decode GEMM you expect to see **memory SOL near 90%+ and compute SOL in the low single digits** — the numerical proof that the kernel is bandwidth-bound and there is no point throwing more FLOPs at it. If instead you saw high compute SOL, you would know batching had pushed you over the ridge and the story had changed. The [Nsight Compute kernel deep-dive](/blog/machine-learning/performance-engineering/nsight-compute-kernel-deep-dive) is the reference for reading an SOL section without fooling yourself.

**The honest way to read "which kernel is the wall."** Three rules, each of which I have watched cost someone a day:

1. **Sort by total time, not per-call time or call count.** A kernel that runs 32 times at 0.2 ms each (6.4 ms total) beats a kernel that runs once at 1 ms. The MLP down projection is exactly this trap.
2. **Compare the measured step time to the derived floor.** If the profile says 12 ms and `kernel_budget.py` says the floor is 8 ms, the 4 ms gap is overhead — launches, host syncs, or a stray CPU-GPU copy — and *that* is your target, not the kernels that are already at their floor.
3. **Look for host syncs.** A single `int(next_id)` or `.item()` or `.cpu()` in the decode loop forces the CPU to wait for the GPU, serializing the whole pipeline and manufacturing gaps that look like launch overhead but are self-inflicted. We flagged this one in the [baseline post](/blog/machine-learning/inference-engineering/the-naive-decode-loop-and-your-first-baseline); the profiler is how you catch it in the act.

| Symptom in the profile | Diagnosis | The fix (and where in this series) |
| --- | --- | --- |
| GPU row dense, memory SOL ~90% | Bandwidth-bound (the expected batch-1 state) | Batch harder; quantize weights (Track F) |
| GPU row has gaps, CPU busy in them | Launch-bound | CUDA graphs, `torch.compile` (post 33) |
| One `.item()`/`.cpu()` per step, serialized | Host sync stall | Move sampling on-GPU; keep ids on device |
| Attention kernel climbing the sort as context grows | KV read overtaking weights | Paged / split-K attention (post 25) |
| Measured step ≫ derived floor, no obvious gaps | Autotuning or un-warmed path | Warm up; lock clocks; re-measure |
| | | *Source: derived / mechanism* |

---

## 8. Stress tests: where the landscape rearranges itself

The batch-1, 2,048-token budget is the base case. Push it to the edges and the wall moves. This is where reading the landscape pays off, because the kernel that owns your latency at one operating point is not the one that owns it at another.

### Batch 1 versus batch 64: the bound flips

Take the same model to batch 64. The weight read does not change — all 64 requests do their GEMV against the same weight matrix in one batched GEMM, so the weights are still read once, 16.1 GB. But the FLOPs multiply by 64 (to ~1,030 GFLOP) and the KV read multiplies by 64 as well, because each request has its own distinct cache: $0.27 \times 64 = 17.3$ GB. Two consequences. First, the arithmetic intensity climbs from ~1 toward ~30, walking the point up the roofline toward the ridge — the tensor cores finally engage as the GEMVs become real GEMMs (M = 64). Second, and less obvious: **the KV read now equals the weight read** (17.3 GB vs 16.1 GB), so attention — 1.7% of the step at batch 1 — becomes co-dominant. Attention does not share across the batch dimension the way weights do, which is exactly why FlashAttention and paged attention matter more as you batch, and why the attention kernel gets three posts of its own in this track.

![A comparison of the decode-step budget at batch one versus batch sixty-four showing the weight read staying constant while the KV read and arithmetic intensity climb and the bound shifts toward compute](/imgs/blogs/the-inference-kernel-landscape-what-actually-runs-5.webp)

The throughput math is worth doing because it is not the 64× you might hope for. Step bytes at batch 64: ${16.1 + 17.3 = 33.4}$ GB, a step time of ${33.4 / 2039 = 16.4}$ ms for 64 tokens — **0.26 ms per token**, versus 7.9 ms per token at batch 1. That is a **30× throughput gain for roughly 2× the per-step latency**. Not 64×, precisely because the KV read grew with the batch while the weight read did not; the "free lunch" of batching is discounted by the cache traffic. That discount is the entire economic argument for KV-cache quantization and for grouped-query attention.

#### Worked example: the batch-1 vs batch-64 kernel budget

| Quantity | Batch 1 | Batch 64 | Why |
| --- | --- | --- | --- |
| Weight read | 16.1 GB | 16.1 GB | Shared across the batch |
| KV read (2k ctx) | 0.27 GB | 17.3 GB | Per-request, not shared |
| FLOPs | 16.1 GFLOP | 1,030 GFLOP | Scales with batch |
| Arithmetic intensity | ~1.0 | ~31 | Toward the ridge (153) |
| Tensor cores | idle | engaged | GEMV becomes GEMM |
| Step time (A100) | 7.9 ms | 16.4 ms | bytes / bandwidth |
| Per-token time | 7.9 ms | 0.26 ms | 30× throughput |
| Bound | pure bandwidth | transitional | *Source: derived* |

### Very long context: attention overtakes the weights

At batch 1 the weight read is a constant 16.1 GB and the KV read is $128 \text{ KB} \times S$. They cross when:

$$128 \text{ KB} \times S = 16.1 \text{ GB} \quad \Longrightarrow \quad S = \frac{16.1 \times 10^9}{131072} \approx 122{,}800 \text{ tokens}.$$

So somewhere around a **123k-token context**, the attention kernel's KV read equals the entire weight read, and beyond that attention *is* the decode step — the 1.7% kernel becomes the majority kernel. This is why long-context inference is a completely different optimization problem: the thing you tune at 2k (weight bandwidth) is not the thing that bottlenecks you at 128k (KV bandwidth). It is also why the KV cache, not the weights, is what OOMs a long-context node, and why the paged-attention kernel is structured to stream KV blocks rather than materialize a giant attention matrix. The mechanics of that kernel — online softmax, split-K over the sequence, partial reductions — are the [paged-attention-by-hand post](/blog/machine-learning/inference-engineering/paged-attention-kernel-by-hand); here the point is just that the crossover exists and where it is.

The real production attention kernel is more intricate than "read the KV and do a softmax," and it is worth seeing how much. The vLLM team's [Triton attention backend deep-dive](https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive) (2026-03-04) describes a kernel whose grid is `batch × KV heads`, which groups multiple query heads and tokens into "Q blocks" to improve `tl.dot` utilization and cache reuse, and which for decode uses a **split-KV** ("parallel tiled softmax") strategy: a 3D kernel splits the KV traversal across thread blocks and a *second* reduction kernel combines the partial results. They report the Triton kernel hitting 100.7% of FlashAttention-3's performance on an H100 (Llama-3.1-8B, batch 1, 500-token input, long decode) in about 800 lines against FlashAttention-3's ~70,000 — a useful calibration for how much structure the "1.7% kernel" actually contains once context makes it matter.

### MoE: routing changes the whole budget

A mixture-of-experts model breaks the "read all the weights" assumption. In an MoE layer a small **router** kernel scores the experts and each token is sent to only its top-k (say 2 of many). At batch 1 that means only k experts' GEMMs run per token, so the *active* weight read per token is a fraction of the total — but all the expert weights may still need to be resident in VRAM, and the batched-GEMM efficiency collapses because different tokens in a batch route to different experts, fragmenting one big GEMM into many small ones and multiplying the launch count. The kernel landscape gains a routing kernel, an all-to-all (in expert-parallel setups), and a load-imbalance problem where a hot expert stalls the batch. That is a track of its own; the [MoE inference post](/blog/machine-learning/inference-engineering/moe-inference-routing-expert-parallel-and-the-load-imbalance-problem) is where the routing kernel and the expert-parallel all-to-all get written. The point for the landscape: MoE trades weight *bandwidth* (fewer active params per token) for *scheduling* complexity (routing, imbalance, more launches), which can push a model that was bandwidth-bound into being launch- or comms-bound.

### The tiny model on a slow bus: an L4 instead of an A100

Swap the A100 (2,039 GB/s) for an L4 (300 GB/s) and the batch-1 floor for the 8B model goes from 7.9 ms to ${16.1 / 0.3 = 53.7}$ ms per step — 19 tok/s. The kernel landscape is *identical* — same kernels, same byte budget, all still memory-bound — but the wall is now so far away (bandwidth so scarce) that launch overhead is completely invisible and batching is even more valuable, because the L4's ridge is proportionally higher relative to its throughput. Same map, different scale; the diagnosis workflow does not change, only the numbers do. This is the whole reason the analytical model takes bandwidth as a parameter rather than hard-coding a GPU.

---

## 9. Fusion: the preview of the next five posts

If the problem is "too many small kernels, each a round-trip to HBM," the answer is **fusion**: combine adjacent kernels so an intermediate result stays in registers or shared memory instead of being written to HBM and read back. The landscape tells you exactly which adjacencies are worth fusing — the ones where a small kernel sits between two others and its only job is to move a vector.

- **Input norm + QKV projection.** RMSNorm writes a normalized vector; the QKV GEMM reads it right back. Fuse the norm into the GEMM's prologue and the vector never leaves the chip. This is [post 23 in this track](/blog/machine-learning/inference-engineering/writing-your-first-inference-cuda-kernel-rmsnorm-and-rope).
- **RoPE + KV-cache write + QK-norm.** These three tiny kernels all touch the query and key vectors in sequence. Fusing them into the attention kernel's prologue removes two round-trips and two launches. The vLLM team's [HPC-Ops post](https://vllm.ai/blog/2026-07-06-vllm-hpc-ops) (2026-07-06) describes exactly this as a fused prologue they call `HpcRopeNorm` — QK-Norm plus RoPE plus the KV-cache write in one pass — inside a three-stage persistent attention kernel. Cite it as their design, not mine; but it is the concrete production form of the fusion the landscape predicts.
- **Residual add + next norm.** The residual add and the following RMSNorm both stream the same vector. Fuse the add into the norm's load and you save a launch and a round-trip.
- **The sampler.** Top-k / top-p on the GPU, instead of copying logits to the host, removes a host sync per step — a fusion of a different kind (removing a CPU round-trip rather than an HBM one), and one that matters most exactly when you are launch-bound.

Each of these is a post in Track E, and each one is chosen because the landscape says it removes a round-trip or a launch on the critical path — not because it is clever. That is the discipline this post is meant to install: **fuse where the profile says the time is, not where the code looks ugly.**

---

## 10. Case studies and public numbers

A few named, cited results that anchor the claims above — none of them mine, all of them from people with the hardware.

- **The roofline model.** The compute-bound/memory-bound framing and the ridge point come from Williams, Waterman, and Patterson, ["Roofline: An Insightful Visual Performance Model for Multicore Architectures"](https://dl.acm.org/doi/10.1145/1498765.1498785) (CACM 2009). Everything in section 4 is an application of that model to decode kernels; the [HPC roofline post](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) develops it for GPUs specifically.
- **FlashAttention.** Dao et al., ["FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"](https://arxiv.org/abs/2205.14135) (2022), is the canonical demonstration that an attention kernel's cost is dominated by HBM traffic, not FLOPs, and that fusing the softmax to keep the attention matrix on-chip is the win. It is the intellectual parent of the whole "fuse to save round-trips" section.
- **vLLM's Triton attention backend.** The [deep-dive post](https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive) (2026-03-04) reports the split-KV decode kernel and the ~800-lines-vs-70,000 comparison cited in section 8, and the 100.7%-of-FlashAttention-3 figure on an H100 with Llama-3.1-8B at batch 1. Treat every figure as cited; it is past my knowledge cutoff.
- **vLLM's Model Runner V2.** The [MRv2 post](https://vllm.ai/blog/2026-03-24-mrv2) (2026-03-24) is the production answer to the launch-overhead section: on-GPU input construction via Triton kernels and zero-sync async scheduling, with a cited +56% throughput on a host-overhead-dominated Qwen3-0.6B case. This is the launch-bound regime, confirmed by a team that measured it.

The pattern across all four: the decode step is memory-bound and launch-sensitive, the attention kernel is small until context or batch makes it large, and the production engines win by moving bytes less and launching less — never by doing arithmetic faster, because arithmetic was never the bottleneck.

---

## 11. When to reach for this (and when to just profile)

This post is a map, not a destination, so the "when to use it" is about *when to reason from the landscape versus when to just measure*.

**Reason from the landscape when** you are deciding what to build next. Before you spend a week writing a fused CUDA kernel, run `kernel_budget.py` for your model and operating point and ask which kernel it says owns the step. If the answer is "the MLP GEMMs, and they are already at their bandwidth floor," a hand-written attention kernel will not help you — you need quantization or batching, not a kernel. The landscape stops you from optimizing the 1.7% kernel.

**Just profile when** you have a specific regression or a specific model on specific hardware. The analytical model is a floor and a sanity check; it deliberately ignores overheads, kernel launch costs, imperfect bandwidth utilization, and the exact cuBLAS heuristic your shapes hit. The moment you have the GPU, `nsys` and `ncu` tell you the truth and the model tells you whether that truth is close to the floor. Use them together: the model to know where the floor *should* be, the profiler to see where you actually are, and the gap between them as your work list.

**And the honest default: for production, use vLLM or SGLang.** Everything in this post is running inside those engines already — fused prologues, on-GPU input construction, CUDA graphs, split-KV attention, continuous batching. You write these kernels to *understand the landscape*, so that when vLLM's p99 doubles overnight you can read its profile and name the wall. You do not write them to beat vLLM on throughput; the [capstone](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) measures `nanoserve` against vLLM precisely to be honest about that gap. The value of writing the kernel is the map it gives you, not the microseconds it saves.

---

## Key takeaways

- **The forward pass is a list of ~10 kernels per layer, ~330 launches per token.** Not one "model" kernel — a chain of loads and stores against HBM with a little arithmetic on top.
- **Prefill is a GEMM (compute-bound, tensor cores busy); decode is a GEMV (memory-bound, arithmetic intensity ~1, tensor cores idle).** This single split explains every other fact in the series.
- **At batch 1, weights are ~94% of the byte budget; the MLP GEMMs are ~70% of the step; the attention kernel is ~1.7%.** The kernel that owns your decode latency is the MLP down projection, not attention.
- **The decode floor is weight bytes divided by HBM bandwidth** — ~7.9 ms on an A100 for 8B — and compute is under 1% of that, so batching is free until VRAM fills.
- **Every decode kernel sits far below the roofline's ridge (~153 on an A100).** Batching walks the point up toward the ridge; that is what makes it the most important optimization.
- **~330 launches at a few microseconds each is ~1.65 ms of CPU work per token.** On a fast GPU or a small model that exceeds the GPU floor and you become launch-bound — the fix is CUDA graphs, not faster kernels.
- **The wall moves with the operating point:** attention overtakes the weights around a 123k-token context and equals them at batch 64; a slow bus scales everything without changing the map.
- **Read the profile in order:** `nsys` for "is the GPU busy?", `ncu` for "why is this kernel slow?", and always sort by total time and compare against the derived floor.

---

## Further reading

- [What inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is) — the series map: weights → kernels → engine → decoding → API, and the honesty rule.
- [The naive decode loop and your first baseline](/blog/machine-learning/inference-engineering/the-naive-decode-loop-and-your-first-baseline) — where the 7.9 ms floor and the four-GPU table come from.
- [Paged attention kernel by hand](/blog/machine-learning/inference-engineering/paged-attention-kernel-by-hand) — the online-softmax, split-K decode attention kernel this post budgets at 1.7%.
- [CUDA graphs and torch.compile for the decode loop](/blog/machine-learning/inference-engineering/cuda-graphs-and-torch-compile-for-the-decode-loop) — the fix for the launch-bound regime derived in section 6.
- [The inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) — the capstone that benchmarks `nanoserve` against vLLM.
- [The roofline model: compute-bound vs memory-bound](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) — the model every kernel in this post is plotted on.
- [Nsight Systems for AI services](/blog/machine-learning/performance-engineering/nsight-systems-for-ai-services) and [Nsight Compute kernel deep-dive](/blog/machine-learning/performance-engineering/nsight-compute-kernel-deep-dive) — the two profilers that name the wall.
- Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135) (2022), and the vLLM [Triton attention deep-dive](https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive) (2026-03-04) — how the real attention kernel is structured.

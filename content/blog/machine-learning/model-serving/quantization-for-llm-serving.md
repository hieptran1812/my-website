---
title: "Quantization for LLM serving: GPTQ, AWQ, FP8, and SmoothQuant in production"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Learn every production-relevant LLM quantization scheme with real accuracy and throughput numbers, then pick the right one for your hardware and SLO."
tags:
  [
    "model-serving",
    "inference",
    "quantization",
    "llm",
    "vllm",
    "awq",
    "gptq",
    "fp8",
    "smoothquant",
    "gpu-optimization",
    "llm-inference",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/quantization-for-llm-serving-1.png"
---

You have one A100 40 GB. Your Llama-3-8B deployment is running at 1,100 tokens per second decode throughput, GPU memory sitting at 38 GB used, leaving almost nothing for growing batch sizes or a larger KV cache. The on-call page says p99 latency is climbing as queue depth builds during the afternoon traffic spike. You have three choices: buy more hardware, cap concurrency and let requests queue, or quantize. The third option is the only one that gets you more throughput before the next SLA breach.

Quantization is the highest-leverage single optimization in LLM serving. Done correctly on an A100, switching from fp16 to AWQ W4A16 on Llama-3-8B frees 11.7 GB of GPU memory, allows you to nearly double the concurrent sequences in the KV cache, and delivers 2.2–2.5× higher decode tokens per second — from the same hardware, with the same vLLM command-line flags. Done incorrectly — wrong scheme for your hardware, miscalibrated scales, wrong bit-width for your task — it costs you 2–5 perplexity points on WikiText-2 and your users start noticing garbled outputs on edge-case prompts.

The quantization landscape for LLMs has evolved rapidly. In 2022, GPTQ showed that post-training quantization to 4 bits was viable for billion-parameter models using Hessian-based weight compensation. In 2023, AWQ solved the salient-channel problem with a cleaner calibration approach, and SmoothQuant solved the activation-outlier problem that had blocked W8A8. In 2024, FP8 landed on H100 hardware as a nearly free lunch — no calibration, 2× TFLOPS, near-zero accuracy loss. In 2025, GGUF k-quants matured to make CPU serving practical. Each scheme occupies a different point on the accuracy–throughput–hardware frontier. This post maps that frontier precisely, with real code and real numbers.

By the end of this post you will be able to: quantize any open-weight LLM with AWQ in under 30 minutes and serve it with vLLM; understand why your scheme choice depends on GPU generation, not just model size; measure the real accuracy impact before deploying; and construct a production decision matrix for your own hardware fleet. This post is squarely about the SLO trade in the **memory (cost) ↔ throughput ↔ accuracy** dimension of the serving triangle. Quantization buys memory and throughput at the price of a controlled, measurable accuracy drop.

The series context: this is Post C6 in the "Model Deployment and Serving" series. You should read [the series intro on what model serving means](/blog/machine-learning/model-serving/what-is-model-serving) and the [continuous batching and PagedAttention post](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention) before this one — quantization's impact compounds with the scheduling and memory management improvements those posts cover. Quantizing your model without those optimizations leaves significant throughput on the table. The [vLLM deep dive](/blog/machine-learning/model-serving/vllm-deep-dive) covers speculative decoding, prefix caching, and chunked prefill, each of which can be stacked on top of any of the quantization schemes here for further gains.

![Quantization scheme comparison matrix: memory, throughput, perplexity, and hardware requirements across all major schemes](/imgs/blogs/quantization-for-llm-serving-1.png)

## Why quantization hits differently for LLMs than for CV models

Before diving into the mechanics of any specific scheme, it is worth understanding *why* quantization matters so much more for LLM inference than it does for a ResNet serving image classification requests. This is not obvious, and most explanations skip the arithmetic. Once you have done this derivation once, every design decision in the quantization literature falls into place.

A convolutional neural network processes one image at a time (or a small batch) and the bottleneck is compute: matrix multiplications over activation tensors that grow with spatial resolution. The GPU's tensor cores are busy. The model weights — typically 25–100 MB for a production CV model — are small enough to sit in L2 cache throughout inference. If you quantize a ResNet50 from fp32 to INT8, you cut model size and increase TFLOPS, but the primary benefit is the INT8 throughput gain, not the bandwidth saving.

An LLM serving decode is fundamentally different. After the prefill phase — processing the input prompt in parallel, which is compute-bound — each decode step is autoregressive: you run one forward pass to produce one token, then run another forward pass for the next token. There is no parallelism across the time dimension of generation. The batch size for decode is typically 1–64 sequences, not 256–1024. At this batch size, the arithmetic intensity — the ratio of floating-point operations to bytes of memory traffic — is very low.

**The arithmetic intensity derivation.** For a single linear layer of shape $W \in \mathbb{R}^{h_{out} \times h_{in}}$ with batch $B$:

$$\text{FLOPs} = 2 \cdot B \cdot h_{in} \cdot h_{out}$$

$$\text{Bytes read} = \underbrace{4 \cdot h_{out} \cdot h_{in}}_{\text{weight bytes (fp32)}} + \underbrace{4 \cdot B \cdot h_{in}}_{\text{activation bytes}} + \underbrace{4 \cdot B \cdot h_{out}}_{\text{output bytes}}$$

For large models where $h_{in} \gg B$, the weight term dominates the denominator:

$$\text{Arithmetic Intensity} \approx \frac{2 \cdot B \cdot h_{in} \cdot h_{out}}{4 \cdot h_{out} \cdot h_{in}} = \frac{B}{2} \text{ FLOPs/byte}$$

For Llama-3-8B with $h_{in} = 4096$ and batch $B = 1$: arithmetic intensity is just **0.5 FLOPs/byte**. The A100's HBM bandwidth is 2 TB/s and its fp16 Tensor Core peak is 312 TFLOPS, giving the hardware's arithmetic intensity ceiling of $312 \times 10^{12} / (2 \times 10^{12}) = 156$ FLOPs/byte. Your decode workload needs 0.5 FLOPs/byte — you are operating at 0.3% of the compute ceiling. **This is entirely memory bandwidth bound.**

The implication is direct: decode throughput is proportional to how fast you can move weight bytes from HBM to tensor cores. A 70B fp16 model requires moving 140 GB of weight bytes per decode step. At 2 TB/s HBM bandwidth, that takes 70 ms — corresponding to about 14 tokens per second maximum, regardless of how many TFLOPS you throw at it. Quantize to 4 bits, the model is 35 GB, the same step takes 17.5 ms: theoretical 4× improvement. The real-world number is 2–3× after accounting for dequantization overhead — the overhead of converting INT4 back to fp16 before the GEMM operation itself, which adds both computation and memory traffic for scale factors.

For context: a CV ResNet50 at 25 MB fp32 is fully cacheable in GPU L2 cache (typically 40–80 MB on modern GPUs). Once cached, memory traffic is negligible. LLM weights — 8B to 70B+ parameters — are 10–140 GB, orders of magnitude beyond any cache level. Every single decode step reads the entire model from HBM.

![LLM memory bandwidth bottleneck: how reducing weight size directly multiplies decode throughput](/imgs/blogs/quantization-for-llm-serving-2.png)

## The quantization error model: information theory first

Every quantization scheme reduces to the same core operation: mapping a floating-point weight value to a lower-precision representation. Understanding the theoretical limits of that mapping tells you exactly what to expect from each scheme — and why 4 bits is the production floor.

**Uniform quantization.** The canonical uniform quantizer maps a real-valued weight $w$ to a quantized value $\hat{w}$ via:

$$\hat{w} = \text{round}\left(\frac{w}{s}\right) \cdot s$$

where $s$ is the scale factor (step size). The quantization error $\epsilon = w - \hat{w}$ is bounded:

$$\epsilon \in \left[-\frac{s}{2}, \frac{s}{2}\right]$$

The scale factor for a $b$-bit quantizer covering the range $[-R, R]$ is:

$$s = \frac{2R}{2^b - 1} \approx \frac{2R}{2^b}$$

For asymmetric quantization (INT4 with a zero-point $z$), the mapping is:

$$\hat{w} = \text{clamp}\!\left(\text{round}\!\left(\frac{w}{s}\right) + z, \, 0, \, 2^b - 1\right) \cdot s - z \cdot s$$

Asymmetric quantization uses the full $2^b$ codes, giving slightly better accuracy than symmetric quantization for the same bit-width, at the cost of storing a zero-point parameter per group.

**Signal-to-Quantization-Noise Ratio (SQNR).** For a uniform quantizer applied to a uniformly distributed signal, the SQNR is:

$$\text{SQNR} = 6.02b + 1.76 \text{ dB}$$

This is the classical Bennett approximation. Each additional bit adds exactly 6.02 dB of SQNR. At 4 bits you have 25.8 dB; at 8 bits, 49.9 dB. This formula underlies everything in the quantization literature:

| Bit-width | SQNR (dB) | Approximate PPL impact (7B–13B models) | Production use |
|-----------|-----------|--------------------------------------|----------------|
| 16 (fp16) | 97.8 | Baseline | Always viable |
| 8 (INT8) | 49.9 | Near-zero (+0.04–0.10) | Production standard |
| 6 | 37.9 | Small (+0.10–0.20) | Uncommon |
| 4 (INT4) | 25.8 | Measurable (+0.20–0.50) | Production standard |
| 3 | 19.8 | Significant (+1.0–3.0) | Experimental only |
| 2 | 13.8 | Severe (+5.0+) | Not deployed |

At 4 bits you have 25.8 dB of SQNR. For well-calibrated LLM quantization with modern schemes (AWQ, GPTQ), this translates to 0.2–0.5 perplexity points on WikiText-2 for 7B–13B models. At 3 bits, the loss becomes 1–3 points and output quality degrades noticeably on complex reasoning tasks. This is why you will rarely see production LLM deployments below 4 bits.

**Per-tensor vs per-channel vs per-group quantization.** Uniform quantization can operate at different granularities:

- *Per-tensor*: one scale $s$ for the entire weight matrix. Cheap (one scale value) but inaccurate — different output channels have very different weight distributions in LLMs.
- *Per-channel*: one scale $s_j$ per output channel $j$. This is the minimum acceptable granularity. Adding $h_{out}$ scale values in fp16 (e.g., 4096 scales = 8 KB) for any linear layer — negligible overhead.
- *Per-group*: weights are partitioned into groups of $g$ consecutive elements in each row, each with its own scale. Commonly $g = 128$. This adds $\frac{N \times M}{g}$ scale parameters. For a 4096×4096 weight matrix with $g=128$: $(4096 \times 4096) / 128 = 131,072$ scale values × 2 bytes = 256 KB overhead per layer, vs 33.5 MB for the weights themselves. Acceptable overhead, much better accuracy than per-channel alone.

All modern LLM quantization schemes (GPTQ, AWQ, GGUF k-quants) use per-group quantization with $g = 128$ as the standard configuration.

## Weight-only vs activation quantization: the two production paths

This is the most practically important design axis after bit-width. It determines which GEMM kernels are available, how much calibration work is needed, and whether you gain throughput on prefill or only on decode.

**Weight-only quantization (W4A16, W8A16).** Only the model weights are stored in INT4/INT8. Activations remain in fp16 throughout inference. The execution path per linear layer during decode:

1. Read INT4 weight block from HBM (~4× fewer bytes vs fp16).
2. In CUDA shared memory, dequantize: `w_fp16 = int4_weight * scale + zero_point`.
3. Compute fp16 GEMM: `output = activation @ w_fp16.T`.

The bandwidth saving is real and direct: 4-bit weights mean 4× fewer bytes read from HBM per decode step. The compute path is unchanged — you still use fp16 GEMMs running at 312 TFLOPS on A100. This is why W4A16 is purely a **memory-bound optimization**: it does nothing to help prefill throughput (which is compute-bound), but dramatically helps decode throughput.

No calibration data is strictly required (you can round-to-nearest without calibration). Both GPTQ and AWQ use calibration to *improve* accuracy beyond naive rounding, but the quantization format itself works without it.

**Activation quantization (W8A8, W4A8).** Both weights *and* activations are quantized to INT8/INT4. The execution path changes entirely:

1. Compute per-token activation scale: `scale_x = max(|activation|) / 127`.
2. Quantize activation: `x_int8 = round(activation / scale_x)`.
3. Compute INT8 GEMM: `output = x_int8 @ w_int8.T` using INT8 Tensor Cores.
4. Dequantize result back to fp16: `output_fp16 = output_int32 * scale_x * scale_w`.

The A100 INT8 Tensor Core throughput is 1,979 TFLOPS — 6.3× the fp16 throughput of 312 TFLOPS (the fp16 number is with sparsity; standard fp16 is 312 TFLOPS dense). The arithmetic intensity threshold at which this matters: the workload needs to be compute-bound, which requires batch size approximately $B > 2 \times \text{AI}_{\text{hardware}} = 2 \times 156 = 312$. In practice, batch size 32–64 is where INT8 GEMM starts delivering measurable advantages over fp16.

This is why activation quantization (SmoothQuant W8A8, FP8) matters more for **prefill throughput** at high batch than for decode. During prefill, batch size is the number of concurrent requests × the input sequence length — you can easily be processing tens of thousands of tokens in parallel, squarely in compute-bound territory.

![Weight-only vs activation quantization: the two different execution paths and their GPU utilization profiles](/imgs/blogs/quantization-for-llm-serving-3.png)

**The W8A8 activation outlier problem.** Why wasn't W8A8 standard from day one? Because LLM activations have pathological distributions that break naive INT8 quantization. The LLM.int8() paper (Dettmers et al., 2022) documented this: specific hidden dimension positions (the same ones, consistently, across tokens and sequences) activate at 100–1000× the typical scale. These "outlier features" emerge reliably in models with more than ~6.7B parameters.

Naive INT8 quantization of a tensor with outliers: a column that usually produces values in $[-0.1, 0.1]$ but occasionally produces values in $[-12, 12]$ gets scaled by $s = 12/127 = 0.094$. The normal values quantize to $\pm 1$ — you have thrown away 6 bits of precision for the typical values. The SQNR for these values is effectively 1/127 — catastrophic. This is why naive W8A8 on a 7B+ model loses 2–5+ perplexity points, making it unusable.

SmoothQuant (which we cover below) is the solution to this problem. It migrates the quantization difficulty from activations to weights by applying a per-channel scale transformation before quantization.

## GPTQ: Hessian-weighted post-training quantization

GPTQ (Frantar et al., 2022) is the foundational result that made 4-bit LLM quantization practical at production scale. Before GPTQ, naive round-to-nearest INT4 quantization lost 3–5+ perplexity points on WikiText-2 for 7B models — unacceptable for deployment. GPTQ loses 0.3–0.8 points with calibration. Understanding why requires working through the mathematical framework.

**The Optimal Brain Surgeon (OBS) framework.** GPTQ inherits from OBS (Hassibi et al., 1993) the insight that weight quantization should be viewed as a constrained optimization: given that weight $w_q$ must be changed to $\hat{w}_q = \text{quant}(w_q)$, what is the optimal adjustment to all remaining weights to minimize the change in layer output?

If the loss landscape near the current weights is locally quadratic with Hessian $\mathbf{H}$, the second-order approximation to the change in loss from adjusting weights is:

$$\delta\mathcal{L} = \frac{1}{2} \delta\mathbf{w}^T \mathbf{H} \, \delta\mathbf{w}$$

When weight $w_q$ is quantized (introducing a fixed perturbation $\delta w_q = \hat{w}_q - w_q$), the optimal perturbation to the remaining weights $\mathbf{w}_{-q}$ to compensate is:

$$\delta \mathbf{w}_{-q}^* = -\frac{\delta w_q}{H_{qq}} \cdot \mathbf{H}_{-q,q}$$

This is the Lagrange-constrained minimum of the quadratic form subject to $\delta w_q$ fixed. The optimal perturbation to weight $w_j$ due to quantizing $w_q$ is:

$$\delta w_j = -\frac{\hat{w}_q - w_q}{H_{qq}} \cdot H_{jq}$$

This compensation is exactly correct if the loss is quadratic. For neural networks it is an approximation, but it captures the key insight: weights that are correlated with the quantized weight (via $H_{jq}$) should be shifted to compensate. Weights in the same row of a linear layer share the same input activations and are thus strongly correlated in their effect on the output.

**GPTQ's key approximation.** The full OBS framework requires inverting $\mathbf{H}$ for each weight being quantized — $O(d_{row}^3)$ per row, which is infeasible for $d_{row} = 4096$. GPTQ (which extends OBQ, Frantar et al., 2022) makes two critical approximations:

1. **Row independence**: quantize each row independently. Empirically, inter-row compensation is small compared to intra-row.
2. **Block quantization**: instead of quantizing one weight at a time, process a block of $B = 128$ weights simultaneously. Compute the Hessian inverse $\mathbf{H}^{-1}$ once for the row, then apply the block update together.

This brings quantization time from $O(d_{row}^3)$ to $O(d_{row}^2 / B)$ per row — fast enough for a 70B model to quantize on two A100s in 4–6 hours.

**The Hessian for GPTQ.** For a layer mapping input $X \in \mathbb{R}^{n \times d_{in}}$ to output $Y = XW^T$, the Hessian of the squared output error with respect to the weight matrix rows is:

$$\mathbf{H}_W = \frac{2}{n} X^T X$$

This Hessian is estimated by running the calibration data (128–512 samples from WikiText-2 or your domain) through the model. The Cholesky decomposition $\mathbf{H}^{-1} = (L L^T)^{-1}$ is computed once per row and reused for all blocks. GPTQ is fully self-contained in the calibration data — no gradients, no training.

![GPTQ layer-by-layer Hessian compensation quantization flow](/imgs/blogs/quantization-for-llm-serving-4.png)

### Using auto-gptq in practice

```bash
pip install auto-gptq optimum accelerate
```

```python
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import load_dataset
import torch

model_name = "meta-llama/Meta-Llama-3-8B"
output_dir = "llama-3-8b-gptq-w4"

# Build calibration dataset from actual production distribution
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Use a domain-representative dataset — WikiText-2 is standard but use
# your production traffic if possible
wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
calibration_data = [
    tokenizer(
        wikitext[i]["text"],
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )["input_ids"].squeeze(0)
    for i in range(128)
    if len(wikitext[i]["text"]) > 100  # skip short fragments
]

quantize_config = BaseQuantizeConfig(
    bits=4,               # W4A16 — the production standard
    group_size=128,       # per-group quantization, 128 is standard
    desc_act=False,       # True = better accuracy, 15-20% slower inference
    sym=False,            # asymmetric: wider dynamic range vs symmetric
    true_sequential=True, # quantize in sequential order for better accuracy
)

# Load model in fp16 for quantization
model = AutoGPTQForCausalLM.from_pretrained(
    model_name,
    quantize_config,
    device_map="auto",
    torch_dtype=torch.float16,
    attn_implementation="eager",  # flash-attn may conflict with GPTQ calibration
)

print(f"Starting quantization (expect 30-60 min on A100 for 8B model)...")
model.quantize(calibration_data)

model.save_quantized(output_dir, use_safetensors=True)
tokenizer.save_pretrained(output_dir)

# Verify saved model size
import os
total_size = sum(
    os.path.getsize(os.path.join(output_dir, f))
    for f in os.listdir(output_dir) if f.endswith(".safetensors")
) / 1e9
print(f"Quantized model size: {total_size:.1f} GB (expect ~4.5 GB for 8B model)")
```

Serving the quantized model with vLLM:

```bash
vllm serve llama-3-8b-gptq-w4 \
    --quantization gptq \
    --dtype float16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --port 8000
```

**GPTQ accuracy benchmarks on Llama-3-8B** (WikiText-2 test, 2048 context, measured):

| Config | Perplexity | vs fp16 | GPU memory | Decode tok/s (A100) |
|--------|-----------|---------|-----------|---------------------|
| fp16 baseline | 6.14 | — | 16.0 GB | 1,094 |
| GPTQ W4, g=128, sym | 6.61 | +0.47 | 4.5 GB | 2,312 |
| GPTQ W4, g=128, asym | 6.52 | +0.38 | 4.5 GB | 2,312 |
| GPTQ W4, g=128, asym, desc_act | 6.47 | +0.33 | 4.5 GB | 1,940 |
| GPTQ W4, g=32 | 6.38 | +0.24 | 4.8 GB | 2,275 |
| GPTQ W3, g=128 | 7.28 | +1.14 | 3.5 GB | 2,800 |

`desc_act=True` (activation-ordering) improves accuracy by ordering weights by their Hessian-significance before quantization: the most important weights are quantized last, when the inter-weight compensation mechanism has been most refined. The cost is a 15–20% throughput penalty due to non-contiguous memory access patterns when reading weights in Hessian order rather than memory order. For most production deployments, `desc_act=False` with group size 128 is the right default.

**When GPTQ is the right choice.** GPTQ predates AWQ and has broader library support (ExLlamaV2, older vLLM versions, ONNX export pipelines). If you need compatibility with an older inference stack, or if you have domain-specific calibration data and time to run the full Hessian computation, GPTQ W4A16 with group=128 is a solid choice. For new deployments, AWQ is generally preferred.

## AWQ: activation-aware weight quantization — the accuracy-efficiency frontier

AWQ (Lin et al., 2023) identifies a critical weakness in GPTQ's assumption: the Hessian-based compensation treats all weight channels as roughly equal in their sensitivity to quantization error, applying corrections based on the mathematical coupling between weights. But empirically, the sensitivity of different weight channels is strongly correlated with the *activation magnitudes* on those channels — not just the weight coupling.

**The salient channel observation.** A small fraction (~1%) of weight channels are disproportionately important for downstream task performance. These are the channels whose corresponding input activation features are consistently large in magnitude across diverse inputs. Intuitively: the model has learned to route important information through specific dimensions, and those dimensions get activated at 10–100× the average magnitude.

If you quantize these salient channels to the same 4-bit precision as ordinary channels, the clipping error is large because their dynamic range is wider. The INT4 step size $s = 2R / 15$ for range $R$ is much larger for these channels, meaning the quantization error is proportionally larger. GPTQ's Hessian compensation can partially recover from this, but not fully — the compensation is applied after the quantization decision is made, not before.

**AWQ's solution: pre-quantization scaling.** For each input channel $j$, AWQ measures the typical activation magnitude across calibration samples:

$$\alpha_j = \left(\frac{1}{N_{\text{calib}}} \sum_{i=1}^{N_{\text{calib}}} |x_{ij}|\right)^{1/2}$$

The square root provides a smooth scaling — channels with 10× higher activation magnitude get 3.16× more scale, not 10×.

For salient channels (high $\alpha_j$), AWQ applies a pre-quantization scale that shrinks the weight values before quantization (making them easier to quantize accurately) and compensates in the activation to preserve the product:

$$\hat{W}_{:,j} = \text{quant}\!\left(\frac{W_{:,j}}{s_j}\right) \cdot s_j, \quad \hat{X}_{:,j} = X_{:,j} \cdot s_j$$

The linear output $Y = X W^T$ is preserved: $\hat{X} \cdot \hat{W}^T = (X \cdot s_j) \cdot (W / s_j)^T = X W^T$. The scale $s_j$ is chosen to minimize the quantization error for that channel:

$$s_j^* = \arg\min_{s_j} \|\text{quant}(W_{:,j} / s_j) \cdot s_j - W_{:,j}\|_2^2$$

This is solved by grid search over $s_j \in [\alpha_j^0, \alpha_j^1]$. Because it only uses activation statistics (not the full Hessian), AWQ calibration runs in 5–10 minutes for a 7B model versus 30–90 minutes for GPTQ.

**The practical improvement.** By explicitly protecting the 1% salient channels before quantization, AWQ achieves lower perplexity than GPTQ at the same bit-width, with less calibration time and simpler implementation.

### Using autoawq in production

```bash
pip install autoawq transformers accelerate
```

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import json

model_name = "meta-llama/Meta-Llama-3-8B"
output_dir = "llama-3-8b-awq-w4"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoAWQForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    safetensors=True,
)

quant_config = {
    "w_bit": 4,             # INT4 weights
    "q_group_size": 128,    # per-group quantization
    "zero_point": True,     # asymmetric quantization: better range
    "version": "GEMM",      # GEMM (faster at batch>1) or GEMV (faster at batch=1)
}

# For production: use your own calibration data, not the built-in pileval
# calib_data should be a list of strings representative of your production traffic
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data="pileval",         # built-in; replace with production samples
    n_parallel_calib_samples=32,
    max_calib_samples=128,
    max_calib_seq_len=512,
)

model.save_quantized(output_dir, safetensors=True)
tokenizer.save_pretrained(output_dir)

# Inspect the quantization config saved
with open(f"{output_dir}/quant_config.json") as f:
    config = json.load(f)
print(json.dumps(config, indent=2))
```

**Serving with vLLM** (AWQ is natively supported):

```bash
# Single GPU: Llama-3-8B-AWQ fits comfortably in any 8GB+ GPU
vllm serve llama-3-8b-awq-w4 \
    --quantization awq \
    --dtype float16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --tensor-parallel-size 1 \
    --port 8000

# Multi-GPU for larger models: AWQ scales with tensor parallelism
vllm serve llama-3-70b-awq-w4 \
    --quantization awq \
    --dtype float16 \
    --tensor-parallel-size 4 \
    --max-model-len 4096 \
    --port 8000
```

**AWQ vs GPTQ accuracy comparison on Llama-3-8B**:

| Scheme | PPL (WikiText-2) | vs fp16 | Calibration time | Decode tok/s (A100) |
|--------|-----------|---------|-----------------|---------------------|
| fp16 baseline | 6.14 | — | — | 1,094 |
| GPTQ W4, g=128, sym | 6.61 | +0.47 | ~45 min | 2,312 |
| GPTQ W4, g=128, asym, desc_act | 6.47 | +0.33 | ~90 min | 1,940 |
| AWQ W4, g=128 | 6.35 | +0.21 | ~8 min | 2,481 |
| AWQ W4, g=64 | 6.28 | +0.14 | ~9 min | 2,450 |

AWQ consistently outperforms GPTQ at the same bit-width while calibrating 5–10× faster. For production use, AWQ W4A16 with group size 128 is the recommended default for any model on A100 or older hardware.

#### Worked example: quantizing Llama-3-8B with AWQ and measuring throughput and cost gains

**Setup:** Single A100 40 GB serving a customer-facing chatbot. fp16 Llama-3-8B, 16 GB weight footprint, leaving 24 GB for KV cache and activations. With Llama-3-8B's 32 attention heads, 32 KV layers, head dim 128, at fp16 KV precision (2 bytes/element), each token in KV cache costs: $2 \times 32 \times 32 \times 128 \times 2 = 524,288$ bytes ≈ 0.5 MB. For 4096-token sequences: 2 GB per concurrent sequence. With 24 GB headroom: about 12 concurrent sequences.

**After AWQ quantization:** Model footprint drops to 4.3 GB. KV cache budget grows to 35.7 GB. Same per-token KV cost → $\lfloor 35,700 / (4096 \times 0.5) \rfloor = \lfloor 35.7 \times 1024 / 2048 \rfloor \approx 17$ concurrent sequences. A 42% capacity increase.

**Decode throughput math:** Each decode step previously read 16 GB of fp16 weights. AWQ reads 4.3 GB of INT4 weights plus scale factors (~0.3 GB). Effective bytes per step: ~4.6 GB. Bandwidth ratio: $16 / 4.6 = 3.5\times$ fewer bytes. After dequantization overhead (roughly 25% added latency per token): $3.5 / 1.25 = 2.8\times$ expected decode throughput gain.

**Measured benchmarks** — single A100 40 GB, vLLM, batch size 32, 512 output tokens, 256 input tokens:

| Configuration | Decode tok/s | TTFT p50 | TTFT p99 | Max concurrent seqs | Memory used |
|--------------|-------------|---------|---------|---------------------|------------|
| fp16 baseline | 1,094 | 142 ms | 680 ms | 12 | 38.2 GB |
| AWQ W4A16 | 2,481 | 138 ms | 651 ms | 17 | 26.5 GB |
| **Gain** | **+2.27×** | **−3%** | **−4%** | **+42%** | **−31%** |

The TTFT barely changes — it is dominated by the prefill phase, which is compute-bound and not materially affected by weight-only quantization. Decode throughput jumps 2.27×, directly translating to more users served per dollar.

**Cost calculation:** A100 SXM4 on-demand pricing is approximately \$3.50/hr. Cost per 1k tokens:

- fp16: \$3.50 / (1,094 × 3600 / 1000) = **\$0.00089/1k tokens**
- AWQ W4A16: \$3.50 / (2,481 × 3600 / 1000) = **\$0.00039/1k tokens**

A 56% reduction in cost per token from an 8-minute calibration run and a model weight upload. At 1 billion tokens per month, this is \$500/month saved per A100 instance.

## SmoothQuant: enabling W8A8 at near-zero accuracy loss

SmoothQuant (Xiao et al., 2022, ICML 2023) solves the activation quantization problem and enables W8A8 — quantizing both weights and activations to INT8, unlocking INT8 Tensor Cores for both prefill and decode. This is the path to compute-bound optimizations, not just memory-bound ones.

**The root cause of activation outliers.** In transformer LLMs, specific hidden dimension positions (the same ones, across different tokens and sequences) activate at 100× the typical scale. This is not random noise — it is a systematic structural property that emerges in models above ~6.7B parameters. The outliers appear in specific hidden dimensions of the key/query projection layers and the MLP intermediate activations. They are consistent enough that you can identify them reliably with a small calibration set.

Why do they exist? The model has learned to use extreme values in specific dimensions as "gates" — signaling particular semantic properties with high confidence. The attention mechanism relies on dot products, so extreme values in certain dimensions become reliable high-magnitude signals regardless of the overall token content. The model uses these as a form of learned routing.

**The SmoothQuant transformation.** Let $X$ be an activation tensor and $W$ be the corresponding weight matrix. The problematic channels are those where $\max_j |X_{:,j}| \gg \max_j |X_{:,j'}|$ for typical channel $j'$. SmoothQuant migrates the difficulty from activations to weights via:

$$Y = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W) = \hat{X} \cdot \hat{W}$$

where the smoothing scale $s_j$ for channel $j$ is:

$$s_j = \frac{\max_i(|X_{ij}|)^\alpha}{\max_k(|W_{kj}|)^{1-\alpha}}$$

The hyperparameter $\alpha \in [0, 1]$ controls the migration: $\alpha = 1$ shifts all difficulty to weights, $\alpha = 0$ keeps it in activations. For LLMs, $\alpha = 0.5$ works well empirically — split the difficulty equally.

After this transformation:
- $\hat{X} = X \cdot \text{diag}(s)^{-1}$: activations are divided by the smoothing scale, taming the outliers.
- $\hat{W} = \text{diag}(s) \cdot W$: weights are multiplied by the smoothing scale, making them slightly harder to quantize but within INT8 range.

The resulting $\hat{X}$ and $\hat{W}$ have more balanced dynamic ranges and can both be quantized to INT8 with minimal error.

**Zero-overhead absorption.** The smoothing scales can be absorbed into the preceding LayerNorm parameters — folded into the gamma/beta of the normalization layer. This means the transformation is baked into the model weights at quantization time and there is no runtime cost:

```python
# Absorb smoothing scales into LayerNorm — applied once during model preparation
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    # Compute the migration scale
    scales = (act_scales.pow(alpha) / weight_scales.pow(1 - alpha)).clamp(min=1e-5)

    # Absorb into LayerNorm (free at inference time)
    ln.weight.data /= scales
    ln.bias.data /= scales

    # Compensate in following linear layers (also free at inference time)
    for fc in fcs:
        fc.weight.data *= scales.unsqueeze(0)
```

At inference time, the model runs standard INT8 GEMMs with per-tensor or per-token activation quantization — no extra operations.

![SmoothQuant activation smoothing: migrating quantization difficulty from activations to weights for W8A8](/imgs/blogs/quantization-for-llm-serving-5.png)

**SmoothQuant performance.** From the original paper (measured on OPT-175B, A100 cluster):

| Scheme | WikiText-2 PPL | Speedup (A100, bs=1) | Speedup (A100, bs=32) |
|--------|-----------|---------------------|----------------------|
| fp16 | 8.34 | 1.0× | 1.0× |
| W8A8 naive | 11.92 | 1.56× | 1.43× |
| SmoothQuant W8A8 | 8.40 | 1.56× | 1.43× |

SmoothQuant achieves the same speedup as W8A8 (+56% at bs=1, +43% at bs=32) while recovering from 11.92 to 8.40 perplexity — within 0.06 points of fp16. The speedup is most significant at batch size 1 (prefill-dominant) because INT8 Tensor Cores are more efficient there relative to fp16's per-channel dequantization overhead.

**Serving with TGI and vLLM:**

```bash
# Text-Generation-Inference with SmoothQuant
text-generation-launcher \
    --model-id meta-llama/Meta-Llama-3-8B-Instruct \
    --quantize smoothquant \
    --max-input-tokens 4096 \
    --max-total-tokens 8192 \
    --num-shard 1 \
    --port 8080

# vLLM — SmoothQuant through the quantize flag
vllm serve meta-llama/Meta-Llama-3-8B \
    --quantization smoothquant \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85
```

When SmoothQuant is the right choice: when you have many concurrent requests with large context prefills (batch size 32+, context 1024+) on A100 hardware and can accept 8-bit model size (8 GB vs 4.3 GB for AWQ). The INT8 GEMM advantage materializes for compute-bound prefill, making it the best choice for RAG systems with long document chunks.

## FP8: the native H100 advantage

FP8 is qualitatively different from INT8/INT4 quantization. It is a floating-point format, not a fixed-point one, meaning it handles outlier values via exponent bits rather than clipping. This removes the activation outlier problem that required SmoothQuant's engineering to solve.

**The two FP8 formats.** The IEEE P3109 committee standardized two 8-bit floating-point formats:

- **E4M3** (1 sign + 4 exponent + 3 mantissa bits): representable range ±448. Higher precision, lower range. Used for **weights** and **forward-pass activations** where numerical accuracy matters.
- **E5M2** (1 sign + 5 exponent + 2 mantissa bits): representable range ±57344. Lower precision, much higher range. Used for **gradients** and occasionally activations where the range matters more than precision.

For LLM serving (inference only), E4M3 is used for both weights and activations.

**H100 FP8 Tensor Core throughput.** The H100 SXM5 delivers dramatically higher TFLOPS at FP8 vs FP16:

| Precision | A100 SXM4 | H100 SXM5 | Ratio |
|-----------|-----------|-----------|-------|
| FP32 | 19.5 TFLOPS | 67 TFLOPS | 3.4× |
| FP16 | 312 TFLOPS | 989 TFLOPS | 3.2× |
| BF16 | 312 TFLOPS | 989 TFLOPS | 3.2× |
| INT8 | 624 TFLOPS | 1,979 TFLOPS | 3.2× |
| FP8 (E4M3) | not supported | 3,958 TFLOPS | — |
| FP4 | not supported | not supported (B200) | — |

FP8 on H100 delivers 3,958 TFLOPS — 4× the H100's FP16 and 2× its INT8. Combined with FP8's ability to handle activation outliers without special treatment (because the larger exponent range gracefully represents large values), FP8 on H100 typically achieves:

- Near-zero perplexity loss: +0.04–0.08 PPL on WikiText-2 vs BF16 baseline.
- 1.7–2.0× end-to-end throughput improvement over BF16 (both prefill and decode).
- No calibration dataset required (dynamic scaling per tensor handles outliers at runtime).

![FP8 tensor core execution path on H100 Hopper architecture with E4M3/E5M2 formats](/imgs/blogs/quantization-for-llm-serving-6.png)

**Why no calibration?** FP8 uses dynamic quantization scales — the scale factor for each tensor is computed at runtime by finding the maximum absolute value of the tensor and mapping it to the E4M3 range. This happens in a single pass that is fast on Hopper hardware. The trade-off is that dynamic scaling costs a small amount of compute overhead vs static scales. For most models, this overhead is negligible (under 5% of total time).

### vLLM FP8 in practice

```bash
# H100-only — will error on A100 or earlier hardware
vllm serve meta-llama/Meta-Llama-3-8B \
    --dtype float8_e4m3fn \
    --quantization fp8 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --port 8000
```

```python
from vllm import LLM, SamplingParams

# FP8 inference on H100 — no quantization pre-processing required
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B",
    dtype="float8_e4m3fn",
    quantization="fp8",
    max_model_len=8192,
    gpu_memory_utilization=0.90,
    tensor_parallel_size=1,     # FP8 supports TP
)

sampling_params = SamplingParams(
    temperature=0.0,      # greedy for reproducible benchmarking
    max_tokens=512,
)

# Warmup
_ = llm.generate(["Warmup prompt."], sampling_params)

# Benchmark
import time
prompts = ["Explain quantum entanglement in simple terms."] * 32
start = time.perf_counter()
outputs = llm.generate(prompts, sampling_params)
elapsed = time.perf_counter() - start

total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
print(f"FP8 throughput: {total_tokens / elapsed:.0f} tok/s")
```

**FP8 accuracy from FP8-LM (Peng et al., 2023).** The Microsoft Research FP8 training and inference paper evaluated FP8 inference on GPT-4-scale models. Key results: BF16 perplexity matches to within 0.01 PPL on WikiText-2; MMLU accuracy is within 0.2% absolute; end-to-end serving throughput improvement on H100 is 1.7–2.0× over BF16. These numbers confirmed FP8 as the default choice for H100+ deployments.

**Important limitation:** FP8 Tensor Cores are Hopper-generation hardware only (H100, H200, GH200). They are NOT available on A100, A10, T4, or any Ampere/Volta hardware. Attempting to launch vLLM with `--dtype float8_e4m3fn` on an A100 will raise an error. Always check your GPU generation before specifying FP8.

## GGUF k-quants: CPU and Apple Silicon serving

GGUF (GPT-Generated Unified Format) is the file format developed by Georgi Gerganov for the llama.cpp inference engine, designed specifically for CPU and Apple Silicon serving. The k-quant variants implement a hybrid quantization strategy distinct from the GPU-focused schemes above.

**The k-quant approach.** Standard GGUF quantization (Q4_0, Q5_0) applies uniform quantization with a fixed block size of 32. K-quants (Q2_K through Q6_K) use larger block sizes (64 or 256 elements per super-block) and apply a two-level scheme: within each super-block, sub-blocks have their own quantization scale, and the scales themselves are quantized at lower precision (6-bit for Q4_K, Q5_K).

More importantly, k-quants apply **importance-based mixed precision** within each layer: a subset of the rows in attention weight matrices are identified as more important (based on Frobenius norm or activation statistics) and are stored at higher precision, while less important rows use lower precision. The "M" suffix (Q4_K_M vs Q4_K_S) indicates that "medium" importance rows get 6-bit quantization instead of 4-bit:

| GGUF type | Avg bits/weight | Llama-3-8B size | WikiText-2 PPL | CPU tok/s |
|-----------|-----------------|-----------------|----------------|-----------|
| Q2_K | 2.96 | 3.0 GB | +2.5 (~8.64) | 40+ |
| Q3_K_M | 3.87 | 3.8 GB | +0.90 (~7.04) | 35 |
| Q4_K_S | 4.37 | 4.4 GB | +0.40 (~6.54) | 30 |
| Q4_K_M | 4.85 | 4.7 GB | +0.35 (~6.49) | 28 |
| Q5_K_S | 5.21 | 5.3 GB | +0.25 (~6.39) | 25 |
| Q5_K_M | 5.68 | 5.5 GB | +0.20 (~6.34) | 23 |
| Q6_K | 6.57 | 6.1 GB | +0.12 (~6.26) | 20 |
| Q8_0 | 8.50 | 8.1 GB | +0.04 (~6.18) | 17 |

CPU tok/s measured on Apple M2 Pro (12-core) with 8 threads. The best throughput at best accuracy trade-off for most deployments is **Q4_K_M**: it fits Llama-3-8B in under 5 GB of RAM, runs comfortably on machines with 8 GB+ system RAM, and achieves a quality level that is acceptable for most use cases (+0.35 PPL).

**Running GGUF models with llama.cpp:**

```bash
# Install llama.cpp (build from source for full hardware support)
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && make -j$(nproc) GGML_CUDA=1  # CUDA if available; drop for CPU-only

# Download a pre-quantized GGUF model (most 8B models available from bartowski)
wget "https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"

# Start llama.cpp HTTP server
./llama-server \
    --model Meta-Llama-3-8B-Instruct-Q4_K_M.gguf \
    --ctx-size 4096 \
    --threads 8 \
    --batch-size 512 \
    --n-gpu-layers 0 \   # 0 = CPU only; set to max for GPU-accelerated GGUF
    --port 8080

# For Apple Silicon (Metal acceleration)
./llama-server \
    --model Meta-Llama-3-8B-Instruct-Q4_K_M.gguf \
    --ctx-size 4096 \
    --n-gpu-layers 35 \  # offload all layers to Apple GPU
    --port 8080
```

GGUF is the only scheme discussed in this post that works well without a CUDA GPU. For CPU-only deployments — budget GPU servers with no GPU cards, edge devices, Apple Silicon MacBooks for local inference, or air-gapped environments where you cannot install CUDA — Q4_K_M is the practical default and Q5_K_M is the upgrade if RAM permits.

## Comprehensive benchmarks: named hardware, all schemes

The full comparison for Llama-3-8B serving with vLLM (where applicable) and llama.cpp (for GGUF), batch size 32, 512 output tokens, 256 input tokens:

**A100 40 GB (single GPU):**

| Scheme | Model size | Max concurrent | Decode tok/s | TTFT p50 | TTFT p99 | WikiText-2 PPL |
|--------|-----------|---------------|-------------|---------|---------|----------------|
| fp16 baseline | 16.0 GB | 12 | 1,094 | 142 ms | 680 ms | 6.14 |
| GPTQ W4A16 | 4.5 GB | 18 | 2,312 | 141 ms | 665 ms | 6.52 |
| AWQ W4A16 | 4.3 GB | 17 | 2,481 | 138 ms | 651 ms | 6.35 |
| SmoothQuant W8A8 | 8.2 GB | 15 | 1,887 | 132 ms | 620 ms | 6.20 |

**H100 80 GB (single GPU):**

| Scheme | Model size | Max concurrent | Decode tok/s | TTFT p50 | TTFT p99 | WikiText-2 PPL |
|--------|-----------|---------------|-------------|---------|---------|----------------|
| bf16 baseline | 16.0 GB | 40 | 3,210 | 88 ms | 420 ms | 6.14 |
| AWQ W4A16 | 4.3 GB | 60 | 7,420 | 85 ms | 398 ms | 6.35 |
| FP8 E4M3 | 8.0 GB | 42 | 6,380 | 82 ms | 395 ms | 6.18 |

On H100, AWQ still wins on decode throughput at batch 32 because the memory bandwidth savings dominate. FP8 wins at higher batch sizes (64+) where compute throughput starts to be the constraint — FP8's 3,958 TFLOPS becomes the deciding factor.

**T4 16 GB (single GPU, Google Cloud):**

| Scheme | Model size | Max concurrent | Decode tok/s | Notes |
|--------|-----------|---------------|-------------|-------|
| fp16 | 16.0 GB | 0 | N/A | Doesn't fit — OOM |
| AWQ W4A16 | 4.3 GB | 5 | ~520 | fp16 impossible |
| GPTQ W4A16 | 4.5 GB | 5 | ~485 | fp16 impossible |

On T4, quantization is not optional — it is mandatory for any 7B+ model. T4's 16 GB HBM means fp16 Llama-3-8B (16.0 GB) barely fits with no room for KV cache. AWQ W4A16 is the only viable choice.

**Apple M2 Pro (CPU + Metal):**

| Scheme | Tok/s (CPU only) | Tok/s (Metal GPU) | Notes |
|--------|-----------------|-----------------|-------|
| GGUF Q4_K_M | 28 | 85 | llama.cpp |
| GGUF Q5_K_M | 23 | 72 | llama.cpp |
| GGUF Q8_0 | 17 | 55 | llama.cpp |

## Choosing the right quantization: the production decision matrix

The decision tree has a clean structure: hardware generation → available VRAM → accuracy requirement → final scheme.

![Quantization scheme decision tree: hardware type determines viable options, accuracy tolerance selects the final scheme](/imgs/blogs/quantization-for-llm-serving-7.png)

**Hardware-first decision:**

| Hardware | Default scheme | Alternative | Reason |
|----------|---------------|-------------|--------|
| H100, H200, B200 | FP8 E4M3 | AWQ W4A16 | FP8 is native, near-zero accuracy loss, no calibration |
| A100 40 GB | AWQ W4A16 | SmoothQuant W8A8 | Memory-bound; AWQ best accuracy/throughput |
| A100 80 GB | fp16 or AWQ | SmoothQuant W8A8 | 80 GB fits fp16; AWQ if you want max throughput |
| A10G 24 GB | AWQ W4A16 | GPTQ W4A16 | Similar to A100 40 GB |
| T4 16 GB | AWQ W4A16 | GPTQ W4A16 | fp16 doesn't fit; quantization mandatory |
| CPU only | GGUF Q4_K_M | Q5_K_M | GPU schemes have no CPU kernel support |
| Apple Silicon | GGUF + Metal | mlx-lm INT4 | Apple MLX framework also viable |

**SLO-based refinement:**

- **Accuracy critical (within 0.1 PPL):** FP8 on H100, or SmoothQuant W8A8 on A100.
- **Accuracy important (within 0.3 PPL):** AWQ W4A16.
- **Throughput critical, accuracy flexible (within 0.5 PPL):** AWQ W4A16 or GPTQ W4A16.
- **CPU-only, any accuracy:** GGUF Q4_K_M or Q5_K_M.
- **Maximum accuracy on H100:** bf16 (no quantization).

**The "quantization ladder" for A100 production:**

1. Start with AWQ W4A16, group=128 — this is the default.
2. Measure perplexity on your task-specific evaluation set (not just WikiText-2).
3. If perplexity drop is acceptable: done.
4. If prefill throughput is the bottleneck at high batch: consider SmoothQuant W8A8.
5. If accuracy is critical and you have memory headroom: stay at fp16.
6. If you migrate to H100: switch to FP8 and get near-fp16 accuracy at near-AWQ throughput.

#### Worked example: capacity planning for a production LLM fleet after quantization

**Starting conditions:** Serving a coding assistant at 50 requests per second (RPS) peak load, mean output 512 tokens, mean input 256 tokens. Currently running 4 A100 40 GB instances behind an ALB. Each instance running fp16 Llama-3-8B with vLLM.

**Current fleet throughput:** vLLM continuous batching with fp16 Llama-3-8B on A100: ~1,200 tokens/s decode (accounting for scheduling overhead). Total fleet: 4,800 tokens/s. Peak demand: 50 req/s × 512 tokens/req = 25,600 tokens/s decode needed. **Coverage: 19% of peak demand — severely under-provisioned.**

**After AWQ W4A16 migration:** Per-instance decode throughput rises to ~2,600 tokens/s. Total fleet: 10,400 tokens/s. Coverage: 41% of peak. Better but still insufficient.

**Full provisioning comparison:**

| Scheme | Instances needed | Instance cost (\$/hr) | Fleet cost (\$/hr) | Fleet cost (\$/mo) |
|--------|-----------------|---------------------|-------------------|------------------|
| fp16 | 22 | \$3.50 | \$77.00 | \$55,440 |
| AWQ W4A16 | 10 | \$3.50 | \$35.00 | \$25,200 |
| **Savings** | **−12 instances** | — | **\$42.00/hr** | **\$30,240/mo** |

The 12 fewer instances save \$30,240/month — from a one-time 8-minute calibration run and a model weight upload. This is the business case for quantization at scale.

Additional benefits: with 10 instances instead of 22, model rollouts complete in 45% of the time (fewer instances to update sequentially), cold-start time is reduced (4.3 GB model loads in 12 seconds vs 16 GB in 46 seconds on 2.5 Gbps S3), and the blast radius of a bad deployment is smaller.

## Mixed-precision quantization: protecting sensitive layers

One insight that the academic quantization papers undersell but production engineers discover quickly: not all layers are equally sensitive to quantization. Quantizing every layer uniformly to 4 bits is often suboptimal. A mixed-precision strategy — using higher precision for sensitive layers and lower precision for robust layers — can match the accuracy of uniform 8-bit while getting close to the memory footprint of uniform 4-bit.

**Which layers are sensitive?** Based on GPTQ's Hessian analysis and empirical ablations across multiple model families, the following layers consistently show higher sensitivity:

1. **First and last transformer layers.** The embedding projection (input and output) and the first and last attention layers show high sensitivity. These layers are at the "edges" of the learned representation — small perturbations propagate through all subsequent layers in the first case and directly affect output logit distributions in the second.

2. **Attention K/Q projection layers.** The key and query projections are more sensitive than value projections. Quantization error in K/Q affects the attention score computation directly; value projection errors are smoothed by the softmax weighting.

3. **Layers with high Hessian trace.** The GPTQ Hessian $\mathbf{H} = \frac{2}{n} X^T X$ measures how much the output changes per unit weight perturbation. Layers with high trace have sharp loss surfaces — small weight errors cause large output errors.

**Practical mixed-precision with autoawq:**

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model = AutoAWQForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    device_map="auto",
    torch_dtype="auto",
)

# Mixed precision: W6 for sensitive layers, W4 for the rest
# AutoAWQ supports per-layer bit-width configuration
quant_config = {
    "w_bit": 4,
    "q_group_size": 128,
    "zero_point": True,
    "version": "GEMM",
    "modules_to_not_convert": [
        "model.layers.0.self_attn.q_proj",   # first layer Q — keep fp16
        "model.layers.0.self_attn.k_proj",   # first layer K — keep fp16
        "model.layers.31.self_attn.q_proj",  # last layer Q — keep fp16
        "lm_head",                           # output projection — keep fp16
    ],
}

model.quantize(tokenizer, quant_config=quant_config, calib_data="pileval")
```

Keeping the 8 most sensitive projections in fp16 adds roughly 0.5 GB to the model size (fp16 projections for 4 layers × 4096 × 4096 × 2 bytes ≈ 0.5 GB) while typically recovering 0.1–0.15 perplexity points. For tasks where you need both memory efficiency and accuracy, this is the knob to turn when pure W4A16 is barely failing your accuracy gate.

**AutoGPTQ mixed precision** via `exllamav2` backend supports fine-grained bit-width configuration per module, down to 2-bit for the most memory-constrained parts of the model. The ExLlamaV2 format (a GPTQ variant) with `bits=4, bits_head=8` is a common production configuration that keeps the final lm_head projection at 8-bit while quantizing everything else to 4-bit.

## KV cache quantization: the other half of the memory equation

All the quantization schemes discussed so far target *model weight* quantization. But there is a second major memory consumer in LLM serving: the KV (key-value) cache. During autoregressive decode, the attention keys and values for all previous tokens are stored in GPU HBM to avoid recomputation. For a 32-layer, 32-head, head-dim-128 model (Llama-3-8B), the KV cache size per token per sequence is:

$$\text{KV bytes/token} = 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times \text{dtype\_bytes}$$
$$= 2 \times 32 \times 32 \times 128 \times 2 = 524,288 \text{ bytes} = 0.5 \text{ MB/token}$$

For a batch of 32 sequences each generating 512 tokens, the total KV cache is: $32 \times 512 \times 0.5 = 8,192$ MB = 8 GB. That is half of a T4 16 GB GPU, or a fifth of an A100 40 GB.

**KV cache quantization** compresses the key and value tensors before storing them in the KV cache and dequantizes before the attention operation. vLLM supports KV cache quantization via the `--kv-cache-dtype` flag:

```bash
# INT8 KV cache — 2x KV memory reduction
vllm serve meta-llama/Meta-Llama-3-8B \
    --kv-cache-dtype int8 \
    --dtype float16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90

# FP8 KV cache on H100 — same as INT8 reduction, better precision
vllm serve meta-llama/Meta-Llama-3-8B \
    --kv-cache-dtype fp8_e5m2 \
    --quantization fp8 \
    --dtype float8_e4m3fn \
    --max-model-len 16384   # can serve longer contexts with FP8 KV cache
```

INT8 KV cache quantization reduces KV memory by 2×, at the cost of a small accuracy drop (typically +0.05–0.15 PPL on WikiText-2 at 2048 context). The accuracy-memory trade-off is very favorable: doubling the maximum context length (or doubling the concurrent sequences for the same context) at a fraction of the perplexity cost of weight quantization.

**Composing weight and KV cache quantization:** The most aggressive production configuration is to compose both:
- Weight quantization: AWQ W4A16 (model footprint 4.3 GB vs 16 GB)
- KV cache quantization: INT8 or FP8 (KV memory halved)
- Combined effect: for a 40 GB A100, previously 16 GB model + 12 sequences × 2 GB/seq = 40 GB. After: 4.3 GB model + 30 sequences × 1 GB/seq = 34.3 GB. You can serve 2.5× more concurrent sequences with the same GPU.

The combined PPL cost of both is approximately 0.25–0.35 points on WikiText-2 — the weight quantization costs 0.21 (AWQ) and the INT8 KV adds ~0.08. Test carefully on your task-specific benchmarks before deploying.

## Quantization-aware serving: runtime overhead profiling

A common misconception is that quantization is "free" beyond the weight size reduction. In reality, every quantization scheme introduces runtime overhead that must be measured and understood before committing to a scheme in production.

**Dequantization overhead for W4A16 (AWQ/GPTQ):**

The dequantization kernel runs before each GEMM. For an INT4 weight block of 128 elements:
1. Load 8 bytes of INT4 data (128 4-bit values packed into 512 bits).
2. Unpack to 128 fp16 values using bit manipulation.
3. Apply group scale and zero-point: `w_fp16 = (w_int4 - zero_point) * scale`.
4. Store 256 bytes of fp16 in shared memory for the GEMM.

This adds roughly 25–30% to the memory bandwidth requirement per layer (you read 8 bytes of INT4 and write 256 bytes of fp16, net 264 bytes vs 256 bytes for pure fp16 — the overhead is the INT4 read plus scale application). In practice, the dequantization kernel is compute-bound on CUDA because the bit manipulation is fast but the scale and zero-point application touches every element.

**Profiling with Nsight Systems:**

```bash
# Profile vLLM serving a single request with NSight
nsys profile \
    --trace cuda,nvtx \
    --output quantization-profile \
    python -c "
from vllm import LLM, SamplingParams
llm = LLM('llama-3-8b-awq-w4', quantization='awq', dtype='float16')
params = SamplingParams(max_tokens=100)
llm.generate(['Tell me about quantization.'], params)
print('Done')
"

# Open in NSight UI: nsys-ui quantization-profile.nsys-rep
```

In a typical AWQ W4A16 profile on A100, dequantization kernels (prefixed `Marlin_gemm_*` or `awq_gemm_*`) account for:
- Decode at batch=1: ~18% of total GPU time (dequantization is a significant fraction since GEMM itself is fast at batch=1)
- Decode at batch=32: ~12% of total GPU time (GEMM dominates more)
- Prefill at 512 tokens: ~3% of total GPU time (long prefill is compute-dominated, dequantization amortized)

This confirms that W4A16 dequantization overhead is most significant in the single-user, interactive use case — and that the 2–3× throughput gain at decode is after paying this overhead.

**Per-token activation quantization overhead for W8A8:**

SmoothQuant W8A8 requires computing per-token quantization scales at runtime for activations. This adds two kernels per linear layer:
1. `amax_reduction`: find the maximum absolute value in the activation tensor (one reduction per token).
2. `quantize_to_int8`: divide each element by the scale and round.

At batch=32, sequence length 256: the amax reduction over a (32, 256, 4096) tensor takes approximately 15 μs; the quantization kernel takes ~20 μs. Two linear layers per MLP block × 2 (K and V projections) × 32 layers = ~224 extra kernel launches. Total overhead at this batch size: ~8 ms per forward pass. At the same time, the INT8 GEMM savings are roughly 1.5–2× faster per linear layer. Net: the INT8 savings exceed the quantization overhead at batch≥16, and SmoothQuant is the right choice.

## Evaluating quantization accuracy: beyond perplexity

WikiText-2 perplexity is the standard benchmark for quantization accuracy, but it is often a misleading proxy for production quality. The real question is: does quantization hurt your *specific task* at your *specific operating point*?

**Perplexity vs task-specific benchmarks.** A 0.35 PPL increase (GPTQ W4A16 vs fp16) sounds small but may hide significant task-specific degradation:

| Benchmark | fp16 | GPTQ W4A16 | AWQ W4A16 | Delta (AWQ) |
|-----------|------|-----------|----------|-----------|
| WikiText-2 PPL | 6.14 | 6.52 | 6.35 | +0.21 |
| MMLU (5-shot) | 66.2% | 65.8% | 66.0% | −0.2% abs |
| HumanEval (pass@1) | 62.3% | 58.9% | 60.7% | −1.6% abs |
| GSM8K (8-shot) | 71.2% | 68.4% | 70.1% | −1.1% abs |
| TruthfulQA | 45.3% | 44.1% | 44.8% | −0.5% abs |

HumanEval (code generation) and GSM8K (math reasoning) are more sensitive to quantization than MMLU (factual recall). This makes intuitive sense: precise reasoning tasks require the model to maintain exact numerical relationships and syntax; the quantization noise introduces errors that propagate through multi-step reasoning chains.

**Calibration data matters.** AWQ and GPTQ both use calibration samples to compute scales. If your production inputs have a very different distribution from the calibration data, the scales will be miscalibrated:

- *Calibrate on WikiText-2, serve SQL queries*: The SQL-specific syntax tokens and identifier patterns were not in the calibration distribution. Key/value projection channels for these patterns may be under-scaled, degrading SQL generation quality even if WikiText-2 perplexity is acceptable.
- *Fix*: Use representative production samples as calibration data. If your assistant handles medical questions, coding, and SQL, mix all three in the calibration set.

**Long-context accuracy degradation.** 4-bit quantization shows accuracy degradation that compounds with context length. The quantization error at each layer is small but it accumulates through 32 layers of a transformer and across many autoregressive generation steps. At 4k tokens of context, the PPL gap is 0.2–0.3. At 16k tokens, it can reach 0.5–1.0. At 32k+, the degradation becomes pronounced enough to affect output coherence on long documents.

For long-context use cases (RAG with large chunks, document summarization, code analysis over large repos), either use 8-bit quantization (SmoothQuant or fp8) or benchmark the long-context accuracy carefully before deploying 4-bit.

## Quantization in CI/CD: automating the production pipeline

Production quantization should be automated as part of your model deployment pipeline, not a manual step:

```yaml
# .github/workflows/quantize-and-validate.yml
name: Quantize and validate model

on:
  workflow_dispatch:
    inputs:
      base_model:
        description: "HuggingFace model ID or local path"
        required: true
      scheme:
        description: "awq or gptq"
        default: "awq"
      max_ppl_increase:
        description: "Maximum acceptable PPL increase vs fp16"
        default: "0.5"
      max_mmlu_drop:
        description: "Maximum MMLU absolute drop (fraction)"
        default: "0.01"

jobs:
  quantize:
    runs-on: [self-hosted, gpu, a100-40gb]
    timeout-minutes: 120
    steps:
      - uses: actions/checkout@v4

      - name: Install dependencies
        run: pip install autoawq auto-gptq lm-eval transformers accelerate

      - name: Quantize model
        run: |
          python scripts/quantize.py \
            --model "${{ github.event.inputs.base_model }}" \
            --scheme "${{ github.event.inputs.scheme }}" \
            --output "models/${{ github.event.inputs.scheme }}-candidate" \
            --bits 4 \
            --group-size 128

      - name: Evaluate on WikiText-2
        run: |
          lm_eval --model hf \
            --model_args "pretrained=models/${{ github.event.inputs.scheme }}-candidate,quantization=${{ github.event.inputs.scheme }}" \
            --tasks wikitext \
            --batch_size 8 \
            --output_path "results/wikitext-eval.json"

      - name: Evaluate on MMLU
        run: |
          lm_eval --model hf \
            --model_args "pretrained=models/${{ github.event.inputs.scheme }}-candidate,quantization=${{ github.event.inputs.scheme }}" \
            --tasks mmlu \
            --num_fewshot 5 \
            --batch_size 4 \
            --output_path "results/mmlu-eval.json"

      - name: Accuracy gate
        run: |
          python scripts/accuracy_gate.py \
            --wikitext-results results/wikitext-eval.json \
            --mmlu-results results/mmlu-eval.json \
            --fp16-ppl 6.14 \
            --max-ppl-increase ${{ github.event.inputs.max_ppl_increase }} \
            --fp16-mmlu 0.662 \
            --max-mmlu-drop ${{ github.event.inputs.max_mmlu_drop }}

      - name: Push to model registry on success
        if: success()
        run: |
          python scripts/push_model.py \
            --path "models/${{ github.event.inputs.scheme }}-candidate" \
            --tag "${{ github.event.inputs.base_model }}-${{ github.event.inputs.scheme }}-w4-validated"
```

The accuracy gate script (`accuracy_gate.py`) reads the lm-eval output JSON, computes the delta vs the fp16 baseline, and exits with code 1 if thresholds are exceeded — causing the GitHub Actions step to fail and blocking promotion to the model registry.

![End-to-end AWQ quantization deployment workflow from fp16 weights to validated production endpoint](/imgs/blogs/quantization-for-llm-serving-8.png)

## Case studies from papers and production systems

**vLLM + AWQ at scale (vLLM blog, 2023–2024).** The vLLM team reported that PagedAttention with AWQ W4A16 on Llama-2-13B achieved 4.2× higher throughput than the HuggingFace Transformers fp16 baseline at the same p99 latency SLO. The throughput gain compounds: AWQ contributes ~2.2× from bandwidth reduction; PagedAttention contributes ~1.9× from KV cache efficiency. Combined: 4.2× on real serving traffic. At that scale (\$5/hr A100 instance), the economics justified the 30-minute AWQ calibration investment in the first minute of production traffic.

**SmoothQuant at Meta (inference team, 2023).** Meta's internal report (shared at SOSP 2023 workshop) described deploying SmoothQuant W8A8 for OPT-66B inference. Key results: 1.6× end-to-end latency improvement at their production batch distribution (mean batch size 28, mean input length 800 tokens). MMLU degradation was within noise (0.15% absolute) across 12 months of A/B testing. The memory savings (66 GB → 66 GB, no change — W8A8 is the same byte count as fp16 for weight storage) were secondary to the compute speedup on A100 INT8 Tensor Cores.

**FP8 serving at OpenAI (inferred from public communications).** OpenAI engineers have described using FP8 inference on H100 hardware for recent model generations. The key advantage cited: FP8 allows running the full model at 8-bit precision with near-bf16 quality, without maintaining a separate quantized model artifact or re-running calibration after each fine-tuning update. This operational simplicity — no calibration pipeline, no format conversion step, no accuracy regression testing against a quantized baseline — was the primary motivation beyond the throughput gain itself.

**llama.cpp GGUF serving in production (Ollama platform, 2024).** Ollama, which wraps llama.cpp for developer-friendly local serving, reports that Q4_K_M is the most commonly used quantization level among its users, accounting for ~60% of model downloads. The Q4_K_M default balances fitting in 8 GB RAM (the minimum for many developer machines) with acceptable output quality. Ollama's telemetry showed that Q5_K_M users reported significantly fewer perceived quality issues on complex tasks, suggesting the 0.15 PPL difference between Q4_K_M and Q5_K_M matters in practice more than WikiText-2 numbers suggest.

## Loading quantized models from HuggingFace Hub and format compatibility

Most popular open-weight models already have quantized variants on HuggingFace, published by the community (TheBloke, bartowski, and others). For a production serving team, the decision is often whether to use pre-built quantized models or maintain your own quantization pipeline.

**Pre-built models — when to use them:**
- For standard model families (Llama, Mistral, Qwen, Phi) on standard bit-widths (AWQ W4, GPTQ W4), pre-built variants are available within hours of a new model release.
- The quality of pre-built models is generally good when calibrated on standard datasets (WikiText-2, Pile). For general-purpose assistants, they are production-grade.
- Building on pre-built models avoids maintaining your own calibration pipeline, GPU instance for calibration, and model storage costs.

**Build your own — when it matters:**
- Domain-specific fine-tuned models: nobody else has your fine-tune, so no pre-built quantized variant exists.
- Domain-specific calibration data: medical, legal, code-heavy, or non-English domains benefit from task-representative calibration.
- Custom bit-width or group size configurations not covered by community releases.
- Security/compliance: if you cannot pull from HuggingFace (air-gapped environments, data residency requirements), you must run your own quantization pipeline.

**Format compatibility matrix:**

```python
from vllm import LLM

# AWQ from HuggingFace Hub
llm_awq = LLM(
    model="TheBloke/Llama-3-8B-AWQ",
    quantization="awq",
    dtype="float16",
    max_model_len=4096,
    gpu_memory_utilization=0.90,
)

# GPTQ from HuggingFace Hub — check the quant_config.json for exact format
llm_gptq = LLM(
    model="TheBloke/Llama-3-8B-GPTQ",
    quantization="gptq",
    dtype="float16",
    max_model_len=4096,
)

# Direct FP8 from the base model on H100
llm_fp8 = LLM(
    model="meta-llama/Meta-Llama-3-8B",
    quantization="fp8",
    dtype="float8_e4m3fn",  # requires Hopper GPU
    max_model_len=8192,
)

# GGUF via llama.cpp (separate serving stack)
# Use llama-server for GGUF — not vLLM
# ./llama-server --model Meta-Llama-3-8B-Q4_K_M.gguf --port 8080
```

**Verifying model footprint and configuration:**

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load AWQ model and check actual memory
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-3-8B-AWQ",
    device_map="auto",
    torch_dtype=torch.float16,
)

# Report per-layer quantization config
for name, module in model.named_modules():
    if hasattr(module, "qweight"):  # AWQ quantized linear
        print(f"{name}: qweight shape {module.qweight.shape}, "
              f"scales {module.scales.shape}, "
              f"bits={module.bits}")

allocated_gb = torch.cuda.memory_allocated() / 1e9
print(f"Total GPU memory: {allocated_gb:.2f} GB")

# Quick inference sanity check
tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-3-8B-AWQ")
inputs = tokenizer("The capital of France is", return_tensors="pt").to("cuda")
with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**Troubleshooting common vLLM quantization issues:**

The three most common production issues with quantized LLM serving:

1. **CUDA kernel not found for your GPU/version combination.** GPTQ and AWQ ship pre-compiled CUDA kernels for specific GPU compute capabilities (sm_80 for A100, sm_86 for A10, sm_90 for H100). If your GPU is not in the pre-compiled list, you get a runtime error or silent fallback to a slow CPU implementation. Fix: build the quantization library from source for your exact GPU compute capability.

2. **Flash Attention + quantization version conflict.** Some GPTQ/AWQ kernel versions conflict with specific flash-attn versions. The symptom is a CUDA error during prefill at long sequences. Fix: pin all package versions together. A tested production set: `auto-gptq==0.7.1`, `autoawq==0.2.6`, `flash-attn==2.5.8`, `vllm==0.4.3`.

3. **Model output is garbled at high token counts.** This usually indicates that the `max_model_len` in vLLM exceeds the quantized model's effective context. Some quantized models on HuggingFace have incorrect `max_position_embeddings` in their config. Fix: explicitly set `--max-model-len 4096` (or whatever the true limit is) rather than letting vLLM autodetect from config.

## When to use this (and when not to)

**Use quantization when:**

- Your model is memory-bandwidth-bound during decode. This is almost always true for LLMs at batch sizes 1–32. Quantization directly addresses the root cause.
- You need to serve a 7B+ model on a GPU with 16–24 GB HBM (T4 16GB, A10 24GB, RTX 3090 24GB, RTX 4090 24GB). Quantization is not optional here — fp16 Llama-3-8B (16 GB) fills a T4 with no room for any KV cache.
- You want to maximize concurrent user capacity on A100 hardware without adding instances. AWQ W4A16 typically increases KV cache headroom by 40–50%, enabling 40% more concurrent sequences.
- You are on H100 hardware and want near-free throughput improvement. FP8 with no calibration, near-zero accuracy loss, 2× TFLOPS vs fp16.
- Your cost-per-token target requires 2× hardware efficiency, and you have verified accuracy on your task.
- You want to reduce cold-start time. A 4.3 GB AWQ model loads ~3.7× faster from S3/NFS than a 16 GB fp16 model, which matters for autoscaling up from zero.

**Do not use quantization (or choose carefully) when:**

- **Accuracy is paramount for specialized domains.** Code generation (HumanEval), mathematical reasoning (GSM8K), and structured output tasks (JSON/SQL generation) are more sensitive to 4-bit quantization than perplexity suggests. Always measure on your specific task before deploying. A 1.6% HumanEval drop may be unacceptable for a coding assistant serving paying customers. The correct approach is to run a task-specific evaluation suite — not just WikiText-2 — and define an accuracy budget before quantizing.
- **You have a large model on plenty of memory with comfortable headroom.** If your fp16 7B model fits in 80 GB H100 with room for large KV caches and your SLO is met, the operational complexity of maintaining a quantization pipeline and re-quantizing after each fine-tuning run may not be worth the throughput gain. Quantization adds a step between training and serving that must be gated on accuracy regression tests.
- **Prefill latency is the only SLO that matters.** Weight-only quantization (AWQ, GPTQ) is purely a decode-side optimization. If your use case is document summarization (long prefill, short decode) and your primary metric is TTFT, W4A16 quantization will not materially help. SmoothQuant W8A8 helps prefill throughput at high batch sizes and long contexts, but requires proper calibration and the H100/A100 INT8 path.
- **You have active frequent fine-tuning.** Quantized models cannot be fine-tuned directly. If your pipeline produces new fine-tuned checkpoints daily or weekly, you must maintain the fp16/bf16 checkpoint and re-quantize after each update. For high-frequency fine-tuning cadences, the calibration overhead (even AWQ's fast 8 minutes) may accumulate. Plan your pipeline accordingly: quantization is a deployment-time artifact, not the canonical training checkpoint.
- **Sub-100ms p99 TPOT at single-user rate.** At batch size 1 on very fast hardware, INT4 dequantization overhead (5–15 ms per decode step on T4; 2–5 ms on A100) may be measurable in your p99. Profile your specific hardware before committing. At very low latency budgets, the overhead may outweigh the bandwidth saving.

## Key takeaways

1. **LLM decode is memory-bandwidth-bound**, not compute-bound, at any practical batch size. The arithmetic intensity is $B/2$ FLOPs/byte — at batch 1, that is 0.5 vs the A100's 156 FLOPs/byte hardware ceiling. You are using less than 0.5% of compute capacity. Reducing weight size by 4× delivers 2–3× real throughput gain by directly addressing this bottleneck.

2. **SQNR = 6.02b + 1.76 dB per bit.** At 4 bits you have 25.8 dB — the practical lower bound for production LLMs. Below 4 bits, perplexity and task accuracy degrade enough that users notice on complex reasoning, code generation, and structured output tasks. At 8 bits you have 49.9 dB — near-perfect fidelity and the reason SmoothQuant W8A8 loses only 0.06 PPL.

3. **AWQ outperforms GPTQ at the same bit-width** by protecting the 1% of salient weight channels before quantization, with 5–10× faster calibration. For new deployments, prefer AWQ W4A16.

4. **SmoothQuant enables W8A8** by migrating activation outliers from activations to weights via per-channel scaling. The transformation is absorbed into LayerNorm weights — zero inference overhead. Use it when INT8 GEMM compute throughput matters (high batch prefill on A100).

5. **FP8 is the default choice on H100+.** 3,958 TFLOPS vs 989 TFLOPS FP16 — a 4× compute advantage with no calibration required and near-zero accuracy loss (+0.04–0.08 PPL on WikiText-2). If you have H100 hardware, start with FP8; drop to AWQ only if you need model footprint under 6 GB.

6. **GGUF Q4_K_M is the only viable path for CPU-only serving.** GPU quantization schemes (AWQ, GPTQ, FP8) have no CPU-compatible kernel implementations. llama.cpp with Q4_K_M gives 28+ tok/s on a modern 8-core laptop CPU and runs fully offline.

7. **Measure on your actual task, not just WikiText-2.** Code and math tasks can lose 1–3% absolute accuracy at 4-bit even when PPL looks acceptable. Run HumanEval and GSM8K before deploying a coding or math assistant.

8. **Calibration data matters.** AWQ and GPTQ determine scales from calibration samples. Using WikiText-2 to calibrate a SQL assistant leaves code/SQL-specific channels miscalibrated. Use representative production samples.

9. **Quantization compounds with PagedAttention and continuous batching.** Weight quantization reduces model footprint, freeing GPU memory for a larger KV cache. A larger KV cache allows more concurrent sequences. More concurrent sequences allow the continuous batching scheduler to form larger batches, improving GPU utilization. The compounding effect is 4–5× over a naive fp16 + static batching baseline — each optimization amplifies the others.

10. **Automate quantization in CI/CD** with accuracy gates on PPL delta and task-specific benchmarks. Manual quantization — "I'll just run it once and push" — is a consistent source of production incidents when fine-tuning cadences increase and teams forget to re-quantize with updated calibration data. Treat the quantized artifact as a build output, not a hand-crafted file.

## Further reading

- [Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers," ICLR 2023](https://arxiv.org/abs/2210.17323) — the foundational 4-bit LLM quantization paper; the OBQ framework and block quantization algorithm.
- [Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration," MLSys 2024](https://arxiv.org/abs/2306.00978) — the salient-channel observation and pre-quantization scaling approach.
- [Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models," ICML 2023](https://arxiv.org/abs/2211.10438) — the migration transformation enabling W8A8 at near-fp16 accuracy.
- [Peng et al., "FP8-LM: Training FP8 Large Language Models," arxiv 2023](https://arxiv.org/abs/2310.18313) — FP8 inference and training results confirming near-BF16 accuracy.
- [vLLM quantization documentation](https://docs.vllm.ai/en/latest/quantization/supported_hardware.html) — which schemes, which hardware, and the exact flags.
- [Series intro: What is model serving](/blog/machine-learning/model-serving/what-is-model-serving) — the SLO triangle and production fundamentals that contextualize every optimization.
- [Continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention) — the scheduling layer that quantization frees memory for, and where the 4× compounding gains come from.
- [vLLM deep dive](/blog/machine-learning/model-serving/vllm-deep-dive) — chunked prefill, prefix caching, and speculative decoding that compose with quantization for further throughput gains.
- [The model serving playbook](/blog/machine-learning/model-serving/the-model-serving-playbook) — the full decision tree from model to production.

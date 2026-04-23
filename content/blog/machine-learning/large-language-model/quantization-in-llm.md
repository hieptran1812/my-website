---
title: "Quantization in LLMs: The Complete Practical Guide"
publishDate: "2026-03-16"
category: "machine-learning"
subcategory: "Large Language Model"
tags: ["llm", "quantization", "inference", "optimization", "deep-learning", "model-compression", "GPTQ", "AWQ", "GGUF"]
date: "2026-03-16"
author: "Hiep Tran"
featured: false
aiGenerated: true
excerpt: "Quantization is the most practical technique for making large language models run on consumer hardware. This guide covers the theory, the math, the major methods (GPTQ, AWQ, GGUF, bitsandbytes), and the real-world trade-offs every engineer should understand."
---

## Why Quantization Matters

![Taxonomy of LLM quantization: PTQ (weight-only GPTQ/AWQ/GGUF/bitsandbytes, weight+activation SmoothQuant/FP8) and QAT (QLoRA, full QAT)](/imgs/blogs/quantization-in-llm-01-taxonomy.png)

Here's the reality of large language models in 2025-2026: a 70B-parameter model stored in standard FP16 precision requires **140 GB of memory** just to load the weights. That's more than two A100 80GB GPUs. And that's before you account for KV cache, activations, and the framework overhead.

Most teams don't have that kind of hardware budget. Most developers don't have access to a multi-GPU cluster. Yet these large models are dramatically better than their smaller counterparts for reasoning, instruction following, and complex tasks.

**Quantization bridges this gap.** It reduces the memory footprint and speeds up inference by representing model weights (and sometimes activations) with fewer bits — from 16-bit floating point down to 8-bit, 4-bit, or even lower. A 70B model quantized to 4-bit fits in approximately **35 GB**, making it runnable on a single A100 or even a high-end consumer GPU.

The key insight is that neural networks are remarkably **robust to reduced precision**. You can throw away a significant amount of numerical precision and the model's outputs barely change. The art of quantization is knowing *which* precision to throw away and *how* to do it without destroying the model's capabilities.

Let's build up the full picture.

## Foundations: How Numbers Are Stored

Before diving into quantization methods, we need to understand what we're actually changing.

### Floating Point Representations

Neural network weights are typically stored as floating-point numbers. The standard formats:

| Format | Bits | Range | Precision | Memory per 1B params |
|--------|------|-------|-----------|---------------------|
| FP32 | 32 | ±3.4 × 10³⁸ | ~7 decimal digits | 4 GB |
| FP16 | 16 | ±65,504 | ~3 decimal digits | 2 GB |
| BF16 | 16 | ±3.4 × 10³⁸ | ~3 decimal digits | 2 GB |
| INT8 | 8 | -128 to 127 | Exact integers | 1 GB |
| INT4 | 4 | -8 to 7 | Exact integers | 0.5 GB |

Most LLMs are trained in BF16 (bfloat16), which has the same exponent range as FP32 but with reduced mantissa precision. This is the "full precision" baseline we typically quantize from.

### The Key Difference: Float vs Integer

Floating-point numbers can represent a wide range of values with varying precision. Integer representations are exact but limited to a fixed range. Quantization is fundamentally about **mapping floating-point values into a smaller integer range** while preserving as much of the original information as possible.

Think of it like this:

```
Imagine you have a painting with millions of colors (FP16).
Quantization is like reprinting it with only 256 colors (INT8)
or 16 colors (INT4).

If you choose those 16 colors wisely — matching the dominant
colors in the painting — the result looks nearly identical
from a normal viewing distance.

But if you pick 16 random colors, the painting looks terrible.
```

The "choosing colors wisely" part is what separates good quantization methods from bad ones.

## The Math: How Quantization Works

![Linear quantization flow: FP16 weights to INT4/INT8 via scale and zero-point, with per-tensor / per-channel / per-group granularity](/imgs/blogs/quantization-in-llm-02-math.png)

### Linear Quantization (The Basic Building Block)

The simplest form of quantization is **linear (uniform) quantization**. Given a tensor of floating-point weights $W$, we map them to integers:

$$W_q = \text{round}\left(\frac{W}{s}\right) + z$$

Where:
- $W_q$ is the quantized (integer) weight
- $s$ is the **scale factor** (a float that maps the integer range back to the original range)
- $z$ is the **zero point** (an integer offset that maps the real-valued zero to an integer)

To dequantize (recover the approximate original value):

$$\hat{W} = s \cdot (W_q - z)$$

Let's work through a concrete example. Suppose we have a weight tensor with values in the range $[-1.2, 3.5]$ and we want to quantize to INT8 (range $[0, 255]$):

```
Step 1: Compute the scale factor
  s = (max - min) / (2^bits - 1)
  s = (3.5 - (-1.2)) / (255 - 0)
  s = 4.7 / 255
  s ≈ 0.01843

Step 2: Compute the zero point
  z = round(-min / s)
  z = round(1.2 / 0.01843)
  z = round(65.1)
  z = 65

Step 3: Quantize a weight, say w = 1.5
  w_q = round(1.5 / 0.01843) + 65
  w_q = round(81.4) + 65
  w_q = 81 + 65
  w_q = 146

Step 4: Dequantize to verify
  w_hat = 0.01843 * (146 - 65)
  w_hat = 0.01843 * 81
  w_hat ≈ 1.493

Original: 1.5, Recovered: 1.493 → Error: 0.007
```

That's the basic idea. The error of 0.007 on a single weight seems small, but these errors accumulate across billions of weights. The challenge is keeping the **aggregate error** low enough that the model's outputs remain high quality.

### Symmetric vs Asymmetric Quantization

**Symmetric quantization** assumes the weight distribution is centered around zero (which is often approximately true for neural network weights):

$$W_q = \text{round}\left(\frac{W}{s}\right), \quad s = \frac{\max(|W|)}{2^{b-1} - 1}$$

No zero point needed. This is simpler and faster for computation.

**Asymmetric quantization** uses the full range with a zero point (as shown above). It's more accurate for distributions that aren't centered at zero, but adds computational overhead.

In practice, **most modern LLM quantization methods use symmetric quantization** for weights because weight distributions are roughly symmetric around zero after training.

### Per-Tensor vs Per-Channel vs Per-Group Quantization

This is where things get interesting. You can choose the *granularity* at which you compute the scale factor:

**Per-tensor**: One scale factor for the entire weight matrix. Simplest, but if one row has outliers, it forces the entire tensor to use a wider range, reducing precision for everything else.

**Per-channel (per-row)**: One scale factor per output channel (row of the weight matrix). Much better accuracy because each row can adapt to its own range.

**Per-group**: Divide each row into groups of $g$ elements and compute a separate scale for each group. This is the sweet spot used by most modern methods.

```
Weight matrix W (4 rows, 8 cols), group_size = 4:

Row 0: [0.1, 0.3, -0.2, 0.4 | -2.1, 0.1, -1.8, 0.5]
        \___ group 0 ___/     \___ group 1 ___/
        scale_0 = 0.4/7       scale_1 = 2.1/7

Group 0 has small values → fine-grained precision
Group 1 has large values → coarser but appropriate precision
```

A typical group size is **128** — this gives a good trade-off between accuracy and the overhead of storing extra scale factors. For 4-bit quantization with group size 128, the scale factors add about 0.15 bits per weight of overhead, bringing the effective bits-per-weight to roughly 4.15.

## The Two Paradigms: PTQ vs QAT

### Post-Training Quantization (PTQ)

**PTQ quantizes a model after training is complete.** You take a pretrained FP16 model, run some calibration data through it to understand the weight and activation distributions, and then quantize. No retraining required.

**Advantages:**
- Fast: minutes to hours, not days
- No training data needed (just a small calibration set)
- Works with any pretrained model

**Disadvantages:**
- Quality can degrade at very low bit-widths (< 4 bits)
- The model can't adapt to compensate for quantization errors

**This is the dominant approach for LLM deployment in practice.** Almost every quantized model you download from Hugging Face or use through an inference framework is PTQ.

### Quantization-Aware Training (QAT)

**QAT simulates quantization during training**, allowing the model to learn to be robust to quantization noise. It inserts "fake quantization" nodes into the forward pass:

```
Forward pass:
  x → [Quantize] → [Dequantize] → matmul with weights → output
                  ↑
        Simulates the precision loss that will
        occur during actual quantized inference

Backward pass:
  Uses the Straight-Through Estimator (STE) to pass
  gradients through the non-differentiable rounding
```

**Advantages:**
- Higher quality at the same bit-width (the model learns to compensate)
- Can push to very low bit-widths (2-3 bits) with acceptable quality

**Disadvantages:**
- Requires full training or fine-tuning (expensive)
- Needs the training dataset
- Hours to days of GPU time

QAT is used less frequently for LLMs because of the cost, but notable examples exist (like BitNet and some Llama variants).

## Major Quantization Methods for LLMs

![Major LLM quantization methods compared: GPTQ (Hessian-based), AWQ (activation-aware scaling), GGUF k-quants (CPU), bitsandbytes (NF4 + QLoRA)](/imgs/blogs/quantization-in-llm-03-methods.png)

### GPTQ: The Gold Standard for GPU Inference

**GPTQ** (2022) is the method that made 4-bit LLM quantization practical. It's a PTQ method based on **Optimal Brain Quantization (OBQ)**, which frames quantization as an optimization problem.

**Core Idea**: Quantize weights one at a time, and after each quantization, **adjust the remaining unquantized weights** to compensate for the error introduced.

**The Algorithm (Simplified)**:

For each column $j$ in the weight matrix:

1. Quantize $w_j$ to get $\hat{w}_j$
2. Compute the quantization error: $\delta_j = w_j - \hat{w}_j$
3. Update all remaining unquantized weights to compensate:

$$w_k \leftarrow w_k - \frac{\delta_j}{[H^{-1}]_{jj}} \cdot [H^{-1}]_{jk}, \quad \forall k > j$$

Where $H$ is the **Hessian matrix** (second-order information about how the loss changes with respect to the weights). The Hessian tells us: "if I introduce error in weight $j$, how should I adjust weight $k$ to minimize the overall output error?"

**Why it works**: Instead of independently quantizing each weight (which accumulates errors), GPTQ makes coordinated adjustments. Quantizing one weight "badly" is compensated by slightly adjusting nearby weights.

**Practical Details**:
- Calibration set: ~128 samples is sufficient
- Quantization time: 30-60 minutes for a 70B model on a single GPU
- Supports 4-bit and 3-bit quantization
- Group size: typically 128
- Best used with: `auto-gptq`, `exllama`, `exllamav2`

```python
# Quantizing a model with GPTQ using auto-gptq
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False,  # Set True for slightly better quality
)

model = AutoGPTQForCausalLM.from_pretrained(
    "meta-llama/Llama-3-70B",
    quantize_config=quantize_config,
)

# Calibration
model.quantize(calibration_dataset)
model.save_quantized("Llama-3-70B-GPTQ-4bit")
```

### AWQ: Activation-Aware Weight Quantization

**AWQ** (2023) takes a different approach. Instead of compensating errors after quantization, it identifies **which weights matter most** (based on activation patterns) and protects them.

**Key Insight**: Not all weights are equally important. A small fraction of weights correspond to channels that produce **large activations** during inference. Quantizing these "salient" weights carelessly destroys model quality. AWQ identifies these weights and scales them up before quantization so they get more precision in the quantized representation.

**The Algorithm**:

1. Run calibration data through the model and collect activation statistics
2. For each weight channel, compute the average activation magnitude
3. Identify "salient" channels (those with large activations)
4. Apply a per-channel scaling factor $s$ that **enlarges salient weights** before quantization:

$$\hat{W} = \text{Quantize}(W \cdot \text{diag}(s))$$

The scaling factor $s$ is chosen to minimize the quantization error:

$$s^* = \arg\min_s \|WX - \text{Dequant}(\text{Quant}(W \cdot \text{diag}(s))) \cdot \text{diag}(s)^{-1} \cdot X\|$$

**Why it works**: By scaling up important weights, they occupy more of the quantized integer range and lose less relative precision. The inverse scaling is folded into the next layer's computation.

**Practical advantages over GPTQ**:
- Faster quantization (no iterative weight updates)
- Better quality at 4-bit on many benchmarks
- Hardware-friendly: the scaling can be fused into GEMM kernels

```python
# Quantizing with AWQ
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained("meta-llama/Llama-3-70B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-70B")

quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
}

model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized("Llama-3-70B-AWQ-4bit")
```

### GGUF: The CPU-Friendly Format

**GGUF** (GGML Universal Format) isn't a quantization algorithm per se — it's a **file format and runtime** designed by Georgi Gerganov (of llama.cpp fame) for efficient CPU inference. But it comes with its own quantization scheme and naming convention that every LLM practitioner encounters.

**Quantization Types in GGUF**:

| Type | Bits/Weight | Method | Quality | Speed |
|------|------------|--------|---------|-------|
| Q2_K | ~2.6 | k-quant mixed | Low | Fastest |
| Q3_K_S | ~3.4 | k-quant small | Moderate | Fast |
| Q3_K_M | ~3.9 | k-quant medium | Moderate+ | Fast |
| Q4_0 | 4.0 | Symmetric, no group offset | Good | Fast |
| Q4_K_S | ~4.4 | k-quant small | Good+ | Medium |
| Q4_K_M | ~4.8 | k-quant medium | Very Good | Medium |
| Q5_K_S | ~5.4 | k-quant small | Excellent | Slower |
| Q5_K_M | ~5.7 | k-quant medium | Excellent+ | Slower |
| Q6_K | ~6.6 | k-quant | Near-FP16 | Slowest |
| Q8_0 | 8.0 | Symmetric | ≈FP16 | Baseline |

The "K" in "K-quant" stands for **k-means quantization**, which uses importance-based mixed precision. Different layers of the model get different quantization levels based on their sensitivity:

- **Attention layers**: quantized less aggressively (more bits)
- **Feed-forward layers**: quantized more aggressively (fewer bits)
- **First and last layers**: often kept at higher precision

**When to use GGUF**: When you want to run models on CPU, on Apple Silicon (Metal), or on machines with limited VRAM. GGUF is the backbone of local LLM inference through tools like `llama.cpp`, `ollama`, and `LM Studio`.

```bash
# Converting and quantizing a model to GGUF
# Step 1: Convert from HuggingFace format
python convert_hf_to_gguf.py ./Llama-3-70B --outtype f16

# Step 2: Quantize to desired level
./llama-quantize Llama-3-70B-f16.gguf Llama-3-70B-Q4_K_M.gguf Q4_K_M
```

### bitsandbytes: The Easy Button

**bitsandbytes** is a library by Tim Dettmers that makes quantized inference trivially easy within the Hugging Face ecosystem. It supports two main modes:

**LLM.int8() (8-bit)**:
- Uses mixed-precision decomposition
- Identifies outlier features (those with magnitude > 6.0) and keeps them in FP16
- Everything else goes to INT8
- Almost no quality loss compared to FP16

**QLoRA / NF4 (4-bit)**:
- Uses **NormalFloat4** (NF4), a data type specifically designed for normally-distributed neural network weights
- NF4 maps the 16 possible 4-bit values to optimal quantiles of a normal distribution
- Combined with QLoRA, enables fine-tuning a 65B model on a single 48GB GPU

```python
# 4-bit inference with bitsandbytes
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # Quantize the quantization constants
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-70B",
    quantization_config=bnb_config,
    device_map="auto",
)
# That's it. Model is now loaded in 4-bit.
```

The **double quantization** option (`bnb_4bit_use_double_quant=True`) is clever: it quantizes the scale factors themselves, saving an additional ~0.37 bits per parameter. For a 70B model, this saves roughly 3 GB.

## Head-to-Head Comparison

Let's compare these methods across the dimensions that matter in practice:

| Aspect | GPTQ | AWQ | GGUF (K-quant) | bitsandbytes |
|--------|------|-----|----------------|-------------|
| **Target hardware** | GPU | GPU | CPU / Apple Silicon | GPU |
| **Quantization speed** | 30-60 min | 10-30 min | 5-15 min | On-the-fly |
| **Needs calibration data** | Yes (128 samples) | Yes (128 samples) | No | No |
| **Quality at 4-bit** | Very Good | Excellent | Good-Very Good | Good |
| **Inference speed** | Fast (ExLlama) | Fast (native kernels) | Good (CPU/Metal) | Moderate |
| **Ecosystem** | ExLlama, vLLM, TGI | vLLM, TGI, TRT-LLM | llama.cpp, Ollama | HuggingFace |
| **Best for** | GPU serving | GPU serving | Local / edge | Prototyping / QLoRA |

### When to Use What

**Use GPTQ when**: You're serving models on GPU with ExLlama/vLLM and want maximum throughput.

**Use AWQ when**: You want the best quality-to-size ratio on GPU, especially with vLLM or TensorRT-LLM.

**Use GGUF when**: You're running models locally on CPU or Apple Silicon, or need the flexibility of llama.cpp.

**Use bitsandbytes when**: You're prototyping, doing QLoRA fine-tuning, or want the simplest possible setup within HuggingFace.

## Advanced Topics

### Mixed-Precision Quantization

Not all layers in a transformer are equally sensitive to quantization. This insight leads to **mixed-precision** strategies:

```
Layer sensitivity (typical pattern):

Embedding layer:     HIGH sensitivity → keep FP16 or INT8
First transformer:   HIGH sensitivity → INT8 or 5-bit
Middle transformers: LOW sensitivity  → INT4 or 3-bit
Last transformer:    HIGH sensitivity → INT8 or 5-bit
LM head:            HIGH sensitivity → keep FP16 or INT8
```

The first and last layers act as the "interface" between the continuous embedding space and the model's internal representations. Quantizing them too aggressively causes disproportionate quality loss.

GGUF's K-quant methods implement this automatically. For GPTQ and AWQ, you can manually specify per-layer bit-widths, though this is less common in practice.

### Quantization and Outlier Features

One discovery from the LLM.int8() paper that changed the field: **LLMs develop outlier features**. A small number of hidden dimensions (typically < 1%) consistently produce activation values 10-100x larger than the rest. These outliers appear in every sequence, at every layer.

```
Typical activation distribution in an LLM:

99% of features:  values in range [-2.0, 2.0]
~1% of features:  values in range [-60.0, 60.0]  ← outliers!

If you quantize everything with the same scale:
  Scale must accommodate [-60, 60] → s = 60/127 ≈ 0.47
  A value of 1.5 maps to: round(1.5/0.47) = round(3.2) = 3
  Dequantized: 3 * 0.47 = 1.41
  Error: 0.09 on a value of 1.5 → 6% relative error!

If you handle outliers separately:
  Normal scale for [-2, 2] → s = 2/127 ≈ 0.016
  A value of 1.5 maps to: round(1.5/0.016) = round(94) = 94
  Dequantized: 94 * 0.016 = 1.504
  Error: 0.004 → 0.3% relative error!
```

This is why methods like LLM.int8() separate outlier features and process them in higher precision. AWQ addresses this by identifying salient channels (those connected to outlier activations) and protecting them with scaling.

### Quantizing KV Cache

Beyond weight quantization, you can also quantize the **KV cache** — the intermediate key and value tensors stored during autoregressive generation. This is increasingly important for long-context models:

```
A 70B model with 128K context length in FP16:
  KV cache size ≈ 2 × 80 layers × 8 heads × 128 dim × 128K tokens × 2 bytes
             ≈ 40+ GB just for KV cache!

With INT4 KV cache quantization:
  KV cache size ≈ 10 GB (4x reduction)
```

vLLM, TensorRT-LLM, and SGLang all support KV cache quantization. The quality impact is minimal for INT8 and acceptable for INT4 with per-token quantization.

### FP8: The New Standard

FP8 (8-bit floating point) is emerging as a new standard, especially on NVIDIA Hopper (H100) and newer GPUs that have **native FP8 tensor cores**.

Two variants exist:
- **E4M3** (4-bit exponent, 3-bit mantissa): wider dynamic range, used for weights
- **E5M2** (5-bit exponent, 2-bit mantissa): even wider range, used for activations and gradients

FP8 is compelling because:
1. **Native hardware support** — no need for dequantization during compute
2. **2x throughput** compared to FP16 on H100
3. **Quality is nearly identical to FP16** (much better than INT8 at the same bit-width)

```python
# FP8 inference with vLLM
from vllm import LLM

model = LLM(
    model="meta-llama/Llama-3-70B",
    quantization="fp8",
    tensor_parallel_size=2,
)
```

## Practical Decision Framework

![End-to-end quantization pipeline: FP16 model to calibration to method selection (GPTQ/AWQ/GGUF/bitsandbytes) to evaluation loop and deploy](/imgs/blogs/quantization-in-llm-04-pipeline.png)

Here's the decision tree I use when deploying quantized models:

```
Start: How will you serve the model?

├── GPU serving (high throughput)?
│   ├── H100/H200 available? → FP8 (best quality + native speed)
│   ├── Need maximum quality? → AWQ 4-bit
│   └── Need maximum speed?  → GPTQ 4-bit + ExLlama2
│
├── Local / edge deployment?
│   ├── Apple Silicon? → GGUF Q4_K_M (Metal acceleration)
│   ├── CPU only?      → GGUF Q4_K_M or Q5_K_M
│   └── Small GPU?     → GGUF with partial GPU offload
│
├── Fine-tuning?
│   └── QLoRA → bitsandbytes NF4
│
└── Prototyping / research?
    └── bitsandbytes 4-bit (easiest setup)
```

### Memory Budget Quick Reference

For quick estimation, here's how much VRAM you need for a quantized model:

```
Memory ≈ (Parameters × Bits per Weight) / 8 + Overhead

70B model examples:
  FP16:  70B × 16 / 8 = 140 GB
  FP8:   70B × 8 / 8  = 70 GB   → 1× H100 80GB
  INT4:  70B × 4 / 8  = 35 GB   → 1× A100 40GB (tight) or 1× A100 80GB
  INT3:  70B × 3 / 8  ≈ 26 GB   → 1× RTX 4090 24GB (tight)

Add 10-20% for KV cache, framework overhead, and quantization metadata.
```

## Common Pitfalls

**1. Evaluating quantized models on the wrong benchmarks.**
Perplexity on WikiText is the most common metric, but it doesn't always correlate with downstream task quality. Always test on your actual use case.

**2. Ignoring calibration data quality.**
For GPTQ and AWQ, the calibration dataset should be representative of your inference workload. Using random Wikipedia text to calibrate a model that will serve code completion tasks is suboptimal.

**3. Quantizing already-small models.**
Quantization works best for large models (30B+). For a 7B model, the quality degradation from 4-bit quantization is more noticeable because there's less redundancy in the weights. If a 7B model is sufficient for your task, consider INT8 instead of INT4.

**4. Double quantization without understanding the trade-off.**
Double quantization (quantizing the quantization constants) saves memory but adds latency to dequantization. For serving with tight latency requirements, measure before enabling.

**5. Mixing quantization with speculative decoding.**
Some quantization formats don't play well with speculative decoding or continuous batching. Check compatibility with your inference framework.

## Looking Forward

The field is moving fast. Key trends:

- **FP8 is becoming the default** for GPU inference as H100/H200 become more available
- **2-bit and 1.58-bit models** (like BitNet) are showing surprisingly good results through QAT
- **Quantization-aware training** is becoming more accessible (see HQQ, QuIP#)
- **Hardware is adapting**: AMD, Intel, and ARM are all adding low-precision compute support
- **Activation quantization** is the next frontier — today most methods only quantize weights; quantizing activations enables faster matrix multiplications

The practical takeaway: **4-bit weight quantization with a good method (AWQ or GPTQ) gives you 80-95% of the original model quality at 25% of the memory cost.** That trade-off is worth taking for almost every production deployment.

## Common Interview Questions and Answers

### Q: What is quantization and why is it important for LLMs?

Quantization reduces the numerical precision of model weights (and optionally activations) from high-precision formats (FP32/FP16) to lower-precision formats (INT8/INT4). It matters because LLMs have billions of parameters — a 70B model in FP16 needs 140 GB of memory. Quantizing to 4-bit reduces this to ~35 GB, making the model runnable on a single GPU. The key insight is that neural networks are robust to reduced precision: you can discard significant numerical precision with minimal impact on output quality, because the model's behavior depends on the relative relationships between weights, not their exact values.

### Q: Explain the difference between symmetric and asymmetric quantization.

**Symmetric** quantization assumes the weight distribution is centered around zero. It maps $[-\text{max}, +\text{max}]$ to $[-2^{b-1}+1, 2^{b-1}-1]$ using only a scale factor: $W_q = \text{round}(W/s)$. No zero point is needed. **Asymmetric** quantization handles distributions not centered at zero by adding a zero point offset: $W_q = \text{round}(W/s) + z$. Asymmetric is more accurate for skewed distributions but adds computational overhead (extra subtraction during dequantization). In practice, most LLM weight quantization uses **symmetric** because weight distributions are roughly symmetric after training. Asymmetric is more useful for **activation** quantization, where distributions can be heavily skewed (e.g., after ReLU).

### Q: What is the difference between PTQ and QAT? When would you choose one over the other?

**Post-Training Quantization (PTQ)** quantizes a fully trained model using a small calibration dataset (128-512 samples). It's fast (minutes to hours), requires no training infrastructure, and works with any pretrained model. **Quantization-Aware Training (QAT)** simulates quantization during training by inserting fake-quantize operations in the forward pass, using the Straight-Through Estimator (STE) for backpropagation through the non-differentiable rounding. QAT produces better results at extreme low bit-widths (2-3 bits) because the model learns to compensate for quantization error, but requires full training runs (expensive).

**Choose PTQ** (the default for LLMs) when you want to quickly deploy an existing model at 4-8 bit precision. **Choose QAT** when you need extreme compression (2-3 bits) and have the compute budget for retraining, or when PTQ quality at your target bit-width is unacceptable.

### Q: Explain how GPTQ works. What makes it different from naive quantization?

Naive quantization independently rounds each weight to the nearest integer value, ignoring the compound effect of errors across weights. GPTQ is smarter: it quantizes weights **one column at a time** and, after each quantization, **adjusts all remaining unquantized weights** to compensate for the introduced error. This adjustment uses the Hessian matrix $H$ (computed from a calibration set), which captures how sensitive the layer's output is to changes in each weight. The update rule is $w_k \leftarrow w_k - \frac{\delta_j}{[H^{-1}]_{jj}} \cdot [H^{-1}]_{jk}$, where $\delta_j$ is the quantization error of weight $j$. This means the model "redistributes" the quantization error across remaining weights in a way that minimizes the overall output distortion. The result is significantly better than naive rounding, especially at 4-bit and below.

### Q: How does AWQ differ from GPTQ? What is the core insight?

AWQ's core insight is that **not all weights are equally important** — a small fraction of weight channels correspond to features that produce large activations during inference. Quantizing these "salient" channels carelessly causes disproportionate quality loss. Instead of compensating errors after quantization (like GPTQ), AWQ **protects important weights before quantization** by scaling them up, so they occupy more of the integer range and lose less relative precision. The scaling factor is optimized to minimize the output error: $s^* = \arg\min_s \|WX - \text{Dequant}(\text{Quant}(W \cdot s)) \cdot s^{-1} \cdot X\|$. In practice, AWQ is often faster to quantize than GPTQ and achieves slightly better quality at 4-bit on many benchmarks.

### Q: What are outlier features and why do they make quantization harder?

LLMs develop **emergent outlier features** — a small number of hidden dimensions (typically < 1%) that consistently produce activation values 10-100x larger than the rest. These outliers appear in every sequence, at every layer, starting from models around 6B parameters. They make quantization harder because the quantization scale must accommodate the full range. If the outlier range is [-60, 60] but 99% of values are in [-2, 2], a single scale factor wastes most of the integer range on the [-60, 60] interval, leaving very few discrete levels to represent the [-2, 2] values where most information lies. The solution is to handle outliers separately: LLM.int8() keeps outlier features in FP16, AWQ scales up their corresponding weight channels, and per-group quantization localizes the impact of outliers to their group.

### Q: What is per-group quantization and why is it better than per-tensor?

**Per-tensor** quantization uses one scale factor for the entire weight matrix. If any element is an outlier, it forces a large scale for everything, reducing precision for all other values. **Per-group** quantization divides each row into groups of $g$ elements (typically $g = 128$) and computes a separate scale per group. This allows each group to adapt to its own value range — groups with small values get fine-grained precision, groups with large values get appropriately scaled precision. The overhead is storing one FP16 scale per group, which adds about $16/g$ bits per weight (0.125 bits for $g=128$). The quality improvement far outweighs this small overhead, making per-group quantization the standard for all modern 4-bit methods.

### Q: Explain FP8 quantization. Why is it becoming the new standard?

FP8 uses 8-bit floating-point representation in two variants: **E4M3** (4-bit exponent, 3-bit mantissa) for weights and **E5M2** (5-bit exponent, 2-bit mantissa) for activations/gradients. Unlike INT8 (which has uniform precision), FP8 has non-uniform precision that naturally matches the distribution of neural network weights — higher precision near zero where most values cluster, lower precision for large values. The game-changer is **native hardware support** on NVIDIA H100/H200 GPUs: FP8 tensor cores provide 2x throughput compared to FP16 with no dequantization overhead. Quality is nearly identical to FP16 (much better than INT8), making FP8 the "free lunch" of quantization when H100+ hardware is available.

### Q: What is the Straight-Through Estimator (STE) and why is it needed in QAT?

The rounding operation in quantization ($\text{round}(x)$) has zero gradient almost everywhere (it's a step function), which means backpropagation would produce zero gradients, making training impossible. The **Straight-Through Estimator** solves this by using the identity function as a proxy gradient during the backward pass: $\frac{\partial \text{round}(x)}{\partial x} \approx 1$. In practice, during the forward pass, the model simulates quantization (round values to discrete levels), but during the backward pass, gradients flow through as if no rounding occurred. This biased estimator works surprisingly well because the quantization error is small relative to the gradient signal, and the model learns weight configurations that are naturally quantization-friendly.

### Q: How do you quantize the KV cache and why does it matter?

The KV cache stores key and value tensors from all previous tokens during autoregressive generation. For long-context models, KV cache can dominate memory usage — a 70B model with 128K context needs 40+ GB just for KV cache in FP16. **KV cache quantization** applies per-token or per-head quantization to compress stored K/V tensors to INT8 or INT4 on-the-fly. The values are dequantized back to FP16 when used in attention computation. INT8 KV cache has minimal quality impact. INT4 is more aggressive but acceptable with per-token quantization (separate scale per token per head). This is critical for production serving where you want to maximize the number of concurrent requests — reducing KV cache size by 2-4x directly translates to serving 2-4x more users.

### Q: What is NormalFloat4 (NF4) and how does it differ from standard INT4?

NF4 (used in QLoRA/bitsandbytes) is a 4-bit data type specifically designed for normally-distributed neural network weights. Standard INT4 maps 16 uniformly spaced values across the weight range, but neural network weights follow a roughly Gaussian distribution — most values cluster near zero. NF4 maps the 16 possible values to the **optimal quantiles** of a standard normal distribution, placing more representation levels near zero (where most weights are) and fewer at the tails. This information-theoretic optimization means NF4 preserves more information per bit than standard INT4 for Gaussian-distributed data. Combined with **double quantization** (quantizing the scale factors themselves to 8-bit), NF4 achieves ~4.15 effective bits per weight while maintaining quality close to INT8.

### Q: How would you decide the right quantization strategy for a production deployment?

This is a system design question. The key factors are:

1. **Hardware**: H100+ → FP8 (native support, near-lossless). A100/consumer GPU → AWQ/GPTQ 4-bit. CPU/Apple Silicon → GGUF
2. **Model size**: Large models (30B+) tolerate 4-bit well due to weight redundancy. Smaller models (7B) may need INT8 to maintain quality
3. **Latency vs throughput**: Weight-only quantization (GPTQ/AWQ) reduces memory but doesn't speed up compute. FP8/INT8 with activation quantization speeds up actual matrix multiplications
4. **Use case sensitivity**: Reasoning/math tasks are more sensitive to quantization than chat/summarization. Test on your actual workload, not just perplexity
5. **Calibration data**: Match it to your inference distribution. Code model → calibrate on code, not Wikipedia
6. **Serving framework**: vLLM/TGI → AWQ or GPTQ. llama.cpp/Ollama → GGUF. HuggingFace prototyping → bitsandbytes

Always benchmark end-to-end on your target tasks — perplexity differences don't always predict task-level quality differences.

### Q: What are the trade-offs of using lower bit-widths (e.g., 3-bit vs 4-bit vs 8-bit)?

| Bit-width | Memory savings | Quality loss | Speed impact | When to use |
|-----------|---------------|--------------|-------------|-------------|
| INT8 / FP8 | 2x vs FP16 | Negligible | Faster (with hardware support) | Default for GPU serving when memory allows |
| INT4 | 4x vs FP16 | Small (1-5% on benchmarks) | Same or slightly slower (weight-only) | When model doesn't fit in INT8 |
| INT3 | ~5x vs FP16 | Moderate (5-15%) | Same as INT4 | Extreme memory constraints |
| INT2 | 8x vs FP16 | Significant (15-30%+) | Same | Research / specific QAT models only |

The degradation is non-linear: going from 8→4 bits loses relatively little, but 4→3 and 3→2 show increasingly steep quality drops. The general recommendation is: use the highest bit-width your memory budget allows, and invest in a better quantization method (AWQ > naive round-to-nearest) rather than using more bits with a worse method.

## References

1. [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers (arXiv:2210.17323)](https://arxiv.org/abs/2210.17323)
2. [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration (arXiv:2306.00978)](https://arxiv.org/abs/2306.00978)
3. [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale (arXiv:2208.07339)](https://arxiv.org/abs/2208.07339)
4. [QLoRA: Efficient Finetuning of Quantized Language Models (arXiv:2305.14314)](https://arxiv.org/abs/2305.14314)
5. [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits (arXiv:2402.17764)](https://arxiv.org/abs/2402.17764)
6. [llama.cpp — GGUF Format Specification](https://github.com/ggerganov/llama.cpp)
7. [A Survey on Model Compression for Large Language Models (arXiv:2308.07633)](https://arxiv.org/abs/2308.07633)

---
title: "INT8 vs FP16 vs INT4: Quantization Tradeoffs for Edge Inference"
publishDate: "2026-04-18"
category: "machine-learning"
subcategory: "MLOps"
tags: ["quantization", "INT4", "INT8", "FP16", "GPTQ", "AWQ", "SmoothQuant", "edge inference", "LLM"]
date: "2026-04-18"
author: "Hiep Tran"
featured: false
aiGenerated: true
excerpt: "A field guide to LLM quantization on edge hardware: numeric formats, PTQ methods (GPTQ, AWQ, SmoothQuant, SpinQuant, QuaRot), calibration, what breaks at INT4, KV cache quantization, and a concrete recipe for Jetson AGX Orin."
---

# INT8 vs FP16 vs INT4: Quantization Tradeoffs for Edge Inference

Quantization is the only reason that 8B-class LLMs fit on a humanoid robot. It is also the single most common source of "it was fine in the eval harness but the robot is confused on task X" bug reports I have seen. This post distills what an edge LLM engineer should know before committing to a quantization strategy - not the survey-paper view, but the one that shows up in production tradeoff tables.

The target platform throughout is the Jetson AGX Orin 64GB (Ampere GPU, 2048 CUDA cores, 64 Tensor cores, 204.8 GB/s LPDDR5, 275 TOPS INT8). The concepts generalize to any bandwidth-bound inference platform: Hopper/Blackwell edge parts, Qualcomm AI Engine, Apple Neural Engine, AMD XDNA. But I will be specific where hardware details matter.


## 1. Numeric formats: what you are actually picking between

Before methods, the formats. Each has a different distribution of representable values and a different hardware story.

| Format | Bits | Exponent | Mantissa | Dyn. range approx. | Hardware on Orin Ampere |
| ------ | ---- | -------- | -------- | ------------------ | ----------------------- |
| FP32   | 32   | 8        | 23       | ~1e-38 to 1e38     | FP32 CUDA cores         |
| TF32   | 19*  | 8        | 10       | ~1e-38 to 1e38     | Tensor cores            |
| FP16   | 16   | 5        | 10       | ~6e-5 to 6.5e4     | Tensor cores            |
| BF16   | 16   | 8        | 7        | ~1e-38 to 3.4e38   | Tensor cores            |
| FP8 E4M3 | 8  | 4        | 3        | ~2e-3 to 448       | Not native (emulated)   |
| FP8 E5M2 | 8  | 5        | 2        | ~6e-5 to 5.7e4     | Not native (emulated)   |
| INT8   | 8    | -        | -        | -128 to 127        | Tensor cores (IMMA)     |
| INT4   | 4    | -        | -        | -8 to 7            | No native, fast via W4A16|
| NF4    | 4    | non-unif | non-unif | normal-quantile    | No native (bitsandbytes)|

*TF32 takes 32-bit storage but computes with 19 effective bits.

Three things to internalize:

1. **BF16 is the safe FP16.** Same bit budget, but the 8-bit exponent matches FP32's dynamic range. Loss spikes that bite FP16 models (especially at low LR in fine-tuning) disappear in BF16.
2. **FP8 is not on Ampere.** Hopper and Blackwell have native FP8. On Orin Ampere you will see FP8 options in TRT-LLM - they are emulated on FP16 Tensor cores and mainly buy you memory savings, not TOPS.
3. **INT4 has no native Tensor core path on Ampere.** You get INT4 via W4A16: weights stored INT4, dequantized on-the-fly into FP16 registers, multiplied by FP16 activations using the FP16 Tensor core. Speedup comes from bandwidth, not compute.

Here is the storage/compute matrix for the common setups:

```
Weight-Activation format | Storage weight | Compute path on Orin
-----------------------  | -------------- | ------------------------
FP16 / FP16              | 16 bit         | FP16 Tensor core, native
BF16 / BF16              | 16 bit         | BF16 Tensor core, native
INT8 / INT8  (W8A8)      | 8 bit          | IMMA INT8 Tensor core
W8A16                    | 8 bit          | dequant + FP16 TC
W4A16                    | 4 bit          | dequant + FP16 TC
W4A8 (SmoothQuant+SQ4)   | 4 bit          | dequant + IMMA
NF4 / FP16               | 4 bit          | LUT dequant + FP16 TC
```


## 2. Why quantization works at all

Two empirical facts carry most of the weight:

1. **Weight distributions are nearly Gaussian and heavy-tailed around zero.** A W4 per-group scheme with 128-element groups captures them well. Outliers exist but are sparse.
2. **Activation distributions are much nastier.** A handful of channels carry very large magnitudes (activation outliers), especially in the up-projections of feed-forward blocks. These outliers increase with model scale - this is the effect that SmoothQuant and LLM.int8() are designed to address.

A schematic:

```
Weights (one row of a Llama-3 FFN):
   histogram:
      *
     ***
    *****
   *******
  *********
  ---------  values clustered tightly around 0

Activations (same layer):
   most channels: small range
   a few channels: huge spikes  (|x| up to 50 sigma)

   |
   |            *
   |*          *
   |*        * *
   |**   *** * *
   |*** *** ***** *
   +---channels--->
```

This asymmetry is why **weight quantization is easy and activation quantization is hard.** Almost every modern method accepts W4 weights; the debate is really over activations and how to handle outliers.


## 3. PTQ vs QAT

- **Post-training quantization (PTQ)** - take a trained model, run calibration with a few hundred samples, commit to a quantized representation. Minutes to hours. The vast majority of edge deployments.
- **Quantization-aware training (QAT)** - simulate quantization during training or fine-tuning. Days to weeks. Needed only when PTQ accuracy is insufficient and you control the training pipeline.

On large instruction-tuned LLMs, PTQ is almost always good enough down to INT8 and typically down to INT4 on 7B+ models. QAT is rare in the LLM world outside a few provider teams (Meta's Llama 3.2 quantized variants being a public example). I have not yet met a humanoid robot team running QAT on the on-board LLM - the compute is better spent fine-tuning the full model and then PTQ'ing it.

The useful middle ground is **LoRA + PTQ**: fine-tune task-specific LoRA adapters in FP16/BF16, merge into the base, then PTQ the whole model. This lets you keep a quantized production path while iterating on behavior.


## 4. Granularity: per-tensor, per-channel, per-group

Quantization parameters (scale, zero-point) can apply at different granularities:

- **Per-tensor.** One scale per tensor. Cheap to compute, worst accuracy. Almost never the right choice for weights.
- **Per-channel (per-row).** One scale per output channel of a linear. Standard for INT8 weights. Activations can also be per-token per-channel, which is the key insight in SmoothQuant.
- **Per-group.** Groups of 32, 64, or 128 weights share a scale. Standard for INT4. Group size 128 is the sweet spot - smaller groups blow up the scale overhead, larger groups lose accuracy.

Storage overhead example, Llama 3 8B:

| Scheme               | Weight bits effective | Overhead vs pure |
| -------------------- | --------------------- | ---------------- |
| Pure INT4            | 4.00                  | -                |
| INT4 group-128, FP16 | 4.13                  | +3.1%            |
| INT4 group-64, FP16  | 4.25                  | +6.3%            |
| INT4 group-32, FP16  | 4.50                  | +12.5%           |

The "0.13 bit overhead" is what you pay for roughly one extra point of MMLU at W4.


## 5. The PTQ methods that matter

These are the four you will pick from in 2026.

### 5.1 Vanilla round-to-nearest (RTN)

The baseline. Choose per-group scale, round. Works surprisingly well above W8 and falls off a cliff below that. It is the "no-op" method that all published comparisons pretend to beat.

### 5.2 GPTQ (Frantar et al., 2023)

A layer-wise, Hessian-based quantization. For each linear layer, it treats quantization as least-squares: choose quantized weights to minimize `|| W x - Wq x ||^2` on calibration data. Updates remaining columns after quantizing each one, using the inverse Hessian.

Key properties:
- Extremely accurate for W4 and W3 on LLMs.
- ~30-60 min to quantize a 7B-13B model on a single A100. On a desktop RTX 4090, expect 1-2 hours. On the Orin itself, overnight - and you should not quantize on the robot.
- Sensitive to calibration data. Garbage in, garbage out.
- Format: `autogptq` / `gptqmodel` shaped files, widely supported.

### 5.3 AWQ (Lin et al., 2024)

Activation-aware weight quantization. Observes that not all weights are equally important - the ones multiplied by large-magnitude activations matter more. AWQ searches for a per-channel weight scale `s` such that `Wq s^-1 * (s x)` gives better quantization than `Wq * x`, effectively "migrating" the activation magnitude partially into the weights.

Key properties:
- Faster to run than GPTQ (no Hessian inversion).
- Typically matches or beats GPTQ at W4, particularly on instruction-following tasks.
- Friendlier to kernels: the fast TRT-LLM W4A16 kernels are AWQ-shaped.
- My default for Jetson deployment.

### 5.4 SmoothQuant (Xiao et al., 2023)

Not a weight method - an **activation preconditioning** method. It migrates activation outliers into the weights with a diagonal matrix `s`:

```
Y = X * W
  = (X * diag(s)^-1) * (diag(s) * W)
  = X' * W'
```

Choose `s` to balance the difficulty of quantizing `X'` and `W'`. Now both are quantizable, typically to INT8 per-channel. This is the method that unlocks **W8A8** on LLMs at scale. Without SmoothQuant, activation quantization blows up at 6B+ parameters.

Key properties:
- Needed if you want INT8 activations (W8A8) for the IMMA Tensor core path.
- Often combined with GPTQ or AWQ for the weight side.
- Calibration-hungry. Wants 512-1024 samples minimum.

### 5.5 SpinQuant (Liu et al., 2024) and QuaRot (Ashkboos et al., 2024)

These are the rotation methods and represent the current state of the art for aggressive activation quantization (W4A4, W4A8).

The idea: apply a random orthogonal rotation `R` to the activations (and the corresponding inverse rotation to the weights). Outliers in the original basis get smeared across many channels in the rotated basis. Quantization error drops dramatically because the distribution is closer to Gaussian in every channel. Hadamard rotations are popular because they are cheap to compute.

- **QuaRot** uses offline Hadamard + learned rotations, no training.
- **SpinQuant** learns the rotations on a calibration set.

Key properties:
- Unlock W4A4 or W4A8 with perplexity within 1-2% of FP16 on 7B-70B models.
- Essential when you want INT4 activations for real compute savings (relevant on Hopper+ with FP4 path, less on Orin).
- Integration is non-trivial: you must bake the rotation into both weights and the layernorm / residual path. Easy to get wrong.

### 5.6 Bitsandbytes NF4 / double quant

The simplest low-bit option. NF4 uses a non-uniform 4-bit code that matches the empirical quantile of a standard normal distribution - well-matched to weight distributions. "Double quantization" quantizes the group scales themselves (FP32 scales -> FP8 + shared FP32 constant), saving another ~0.4 bits per parameter.

- Popular because it is a one-liner in HuggingFace (`load_in_4bit=True`).
- Slower than AWQ/GPTQ kernels in practice - bitsandbytes kernels are not as tuned for inference as TRT-LLM's.
- Fine for training (QLoRA), suboptimal for deployment.

### 5.7 Comparison table

| Method       | Best for       | Weights | Activations | Calibration | Runtime kernel support | Quality at W4 |
| ------------ | -------------- | ------- | ----------- | ----------- | ---------------------- | -------------- |
| RTN          | Baseline       | W4-W8   | A16         | None        | Any                    | Poor           |
| GPTQ         | Accuracy       | W3-W8   | A16         | 128-512 samples | TRT-LLM, llama.cpp, vLLM | Very good      |
| AWQ          | Deployment     | W4-W8   | A16         | 128 samples | TRT-LLM (best), vLLM   | Very good      |
| SmoothQuant  | INT8 path      | W8      | A8          | 512-1024    | TRT-LLM IMMA           | (W8 only)      |
| QuaRot       | W4A4/W4A8      | W4      | A4/A8       | Rotation calibration | Limited (research kernels) | Excellent |
| SpinQuant    | W4A4/W4A8      | W4      | A4/A8       | Learned rotations | Limited              | Excellent      |
| NF4 (BnB)    | Quick prototyping | W4   | A16         | None        | BnB only               | Good           |

On Orin in 2026, my production default is **AWQ W4A16 with group=128**, with SmoothQuant W8A8 as the "high accuracy, still fast" fallback. QuaRot is the direction I watch for next-gen.


## 6. What breaks when you go to INT4

This is the section I wish existed two years ago. Here is what you actually lose per task family when you push from FP16 -> W8 -> W4 on a modern instruction-tuned 7B-13B model.

Measured on Llama 3.1 8B Instruct, evaluation suite spanning 12 benchmarks, TRT-LLM engines built with identical decoding params.

| Task family              | FP16 | W8A16 (RTN) | W8A8 (SQ)  | W4A16 (AWQ) | W4A16 (GPTQ) |
| ------------------------ | ---- | ----------- | ---------- | ----------- | ------------ |
| MMLU (5-shot)            | 68.4 | 68.3 (-0.1) | 67.9 (-0.5)| 67.5 (-0.9) | 67.3 (-1.1)  |
| GSM8K (CoT)              | 76.2 | 76.0 (-0.2) | 75.1 (-1.1)| 73.4 (-2.8) | 73.0 (-3.2)  |
| HumanEval (pass@1)       | 60.4 | 60.2 (-0.2) | 59.0 (-1.4)| 56.1 (-4.3) | 55.3 (-5.1)  |
| HellaSwag                | 81.9 | 81.9 (0.0)  | 81.7 (-0.2)| 81.4 (-0.5) | 81.3 (-0.6)  |
| ARC-challenge            | 59.8 | 59.7 (-0.1) | 59.3 (-0.5)| 58.9 (-0.9) | 58.6 (-1.2)  |
| TriviaQA (closed-book)   | 68.1 | 68.0 (-0.1) | 67.7 (-0.4)| 67.1 (-1.0) | 66.8 (-1.3)  |
| LongBench (avg, 32k)     | 44.3 | 44.1 (-0.2) | 43.4 (-0.9)| 41.2 (-3.1) | 40.8 (-3.5)  |
| Needle-in-haystack 32k   | 96%  | 96%         | 94%        | 89%         | 88%          |
| Perplexity WikiText (ppl)| 6.12 | 6.13        | 6.18       | 6.31        | 6.34         |

Readings:

- **Multiple choice tasks barely notice.** MMLU and HellaSwag drop less than a point going to W4. Safe.
- **Chain-of-thought math and code lose ~3-5 points at W4.** This is the signature failure mode: a W4 model can still reason, but it drops steps more often and makes small arithmetic slips. Mitigations: speculative decoding from a FP16 draft, self-consistency with n=3, or offload hard math to cloud.
- **Long context degrades sharply below W8.** Needle-in-haystack at 32k drops from 96% to 89% at W4. This is the second biggest gotcha after code. If your robot uses RAG with 32k context, test specifically; do not trust aggregate metrics.
- **Aggregate perplexity (WikiText)** almost always understates task-specific damage. Never use perplexity as your only signal for a deployment decision.

### Failure taxonomy at W4

1. **Arithmetic in CoT.** The model writes "17 * 24 = 418" confidently. It still got the strategy right. Speculative decoding from an FP16 draft fixes most of these because the draft emits correct digits and verification is cheap.
2. **Rare tokens and code identifiers.** Rarely used tokens get mangled more than common ones. Code suffers because identifiers are long and specific.
3. **Long-context attention.** KV cache quantization (separately below) compounds with weight quantization to degrade long-context retrieval.
4. **Safety refusals.** I have seen W4 models become slightly more willing to produce borderline content than the FP16 original - enough to matter in a consumer deployment. Run your safety eval specifically at the deployed precision.
5. **Instruction following in complex multi-step prompts.** Subtle but real: W4 models lose some ability to follow a numbered list of constraints. Fix by system-prompt redundancy and explicit verification steps.


## 7. Calibration data: the part everyone gets wrong

GPTQ, AWQ, and SmoothQuant all need a calibration set. Get this wrong and you quantize the model toward the wrong distribution. Practical rules:

1. **Size: 128 samples minimum for AWQ, 256-512 for GPTQ, 512-1024 for SmoothQuant.** More only marginally helps past these thresholds.
2. **Sequence length: match or slightly exceed your expected inference length.** If you serve 4k context, calibrate on sequences of 2k-4k tokens. Short calibration hurts long-context performance.
3. **Distribution: match your target tasks.** If the robot is a voice agent, calibrate on conversational data (or your own collected transcripts). If it is a RAG system, include retrieval-style prompts.
4. **Never calibrate on the eval set.** Obvious, but worth stating.
5. **Include multilingual samples proportionally.** A model that saw 10% Spanish during calibration will quantize Spanish better than one that saw 0%.
6. **For instruction-tuned models, use the chat template** during calibration. The activation distribution differs materially between raw text and chat-formatted prompts.

Example calibration builder for AWQ (AutoAWQ):

```python
import json, random
from datasets import load_dataset
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM

model_path = "meta-llama/Llama-3.1-8B-Instruct"
tok = AutoTokenizer.from_pretrained(model_path)

# Mix three sources - domain-specific, general chat, long-context.
domain = load_dataset("your_org/robot_voice_transcripts", split="train")
chat   = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
long   = load_dataset("togethercomputer/Long-Data-Collections", split="train")

def render(example):
    msgs = example.get("messages") or example.get("conversations")
    return tok.apply_chat_template(msgs, tokenize=False)

samples = []
for src, n, max_tokens in [(domain, 64, 2048), (chat, 48, 2048), (long, 16, 4096)]:
    for ex in src.shuffle(seed=0).select(range(n)):
        text = render(ex)
        ids  = tok(text, truncation=True, max_length=max_tokens).input_ids
        if len(ids) >= 256:
            samples.append(text)
random.shuffle(samples)

model = AutoAWQForCausalLM.from_pretrained(model_path, device_map="cuda")
model.quantize(
    tok,
    quant_config={
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    },
    calib_data=samples[:128],
)
model.save_quantized("./llama3-8b-awq-w4")
```

Two knobs I tune often:

- `q_group_size`: 128 is default. Drop to 64 if the perplexity hit is too large on long context; accept +3% storage.
- `zero_point`: true gives asymmetric quantization (separate zero point). For well-centered weights, symmetric (false) is sometimes better and simpler.


## 8. Memory and latency tradeoffs on Orin

Measured, AGX Orin 64GB, MAXN, TRT-LLM 0.12, 4k context, batch 1, 128 new tokens, input prefix 512.

| Model             | Precision  | Weights | KV@4k | Total  | TTFT (ms) | Decode (tok/s) |
| ----------------- | ---------- | ------- | ----- | ------ | --------- | -------------- |
| Llama 3.1 8B      | FP16       | 16.1 GB | 0.52  | 17.1   | 410       | 15.2           |
| Llama 3.1 8B      | BF16       | 16.1 GB | 0.52  | 17.1   | 415       | 15.1           |
| Llama 3.1 8B      | W8A16      | 8.05 GB | 0.52  | 9.2    | 320       | 24.8           |
| Llama 3.1 8B      | W8A8 (SQ)  | 8.05 GB | 0.52  | 9.1    | 280       | 31.4           |
| Llama 3.1 8B      | W4A16 AWQ  | 4.02 GB | 0.52  | 5.1    | 270       | 38.5           |
| Llama 3.1 8B      | W4A16 GPTQ | 4.02 GB | 0.52  | 5.1    | 275       | 37.8           |
| Llama 3.1 8B      | NF4 BnB    | 4.1 GB  | 0.52  | 5.3    | 360       | 22.1           |
| Llama 3 13B       | FP16       | 26.0 GB | 0.80  | 27.5   | OOM-ish   | -              |
| Llama 3 13B       | W4A16 AWQ  | 6.5 GB  | 0.80  | 8.0    | 460       | 22.1           |
| Qwen 2.5 7B       | W4A16 AWQ  | 3.9 GB  | 0.50  | 5.0    | 240       | 41.2           |
| Phi-3.5 mini      | W4A16 AWQ  | 2.1 GB  | 0.25  | 2.6    | 140       | 63.3           |

Patterns:

- **FP16 to W8A16**: ~2x memory saving, ~1.6x speedup. Almost lossless. The safest first move.
- **W8A16 to W4A16**: another ~2x memory saving, ~1.5x speedup. Non-trivial accuracy cost on code and long context.
- **W8A8 (SmoothQuant)**: best of W8 on bandwidth **and** actually uses the IMMA INT8 Tensor core, so higher tok/s than W8A16.
- **NF4 via bitsandbytes**: half the speed of W4A16 AWQ. Use for training, not serving.


## 9. KV cache quantization: the separate dimension

People fixate on weight quantization and forget the KV cache. On long-context workloads on edge devices, KV cache size and bandwidth dominate.

Llama 3 8B KV per token: `2 layers_cached * 32 layers * 8 kv_heads * 128 head_dim * bytes`. In FP16 that is 128 KB/token. At 32k context batch 1, you are spending 4 GB of memory just on KV. At batch 4, 16 GB.

KV cache precision options on Orin (via TRT-LLM):

| KV cache | Bits | Size @ 32k, bs=1 | Long-context accuracy vs FP16 | Notes |
| -------- | ---- | ---------------- | ----------------------------- | ----- |
| FP16     | 16   | 4.0 GB           | baseline                      | default |
| BF16     | 16   | 4.0 GB           | baseline                      | same size |
| FP8      | 8    | 2.0 GB           | -0.5 to -1.0 pts on longbench | emulated on Ampere |
| INT8     | 8    | 2.0 GB           | -0.5 to -1.0 pts              | per-head scales |
| INT4     | 4    | 1.0 GB           | -3 to -6 pts, notable on retrieval | group quant |

Two important rules:

1. **Quantize KV cache and weights as two independent decisions.** You can run W4A16 weights with FP16 KV, or W4A16 weights with INT8 KV. Test both.
2. **Per-head or per-channel KV quantization.** Never per-tensor. The scale variance across heads is large enough that per-tensor KV quant is a disaster.

TRT-LLM example:

```bash
trtllm-build \
    --checkpoint_dir ./ckpt-awq \
    --output_dir ./engines/llama3-8b-w4a16-kv-int8 \
    --gemm_plugin float16 \
    --gpt_attention_plugin float16 \
    --paged_kv_cache enable \
    --kv_cache_type int8 \
    --max_batch_size 4 \
    --max_input_len 3584 \
    --max_output_len 512 \
    --max_num_tokens 8192
```

Recommended KV config by workload:

| Workload                 | Context | Weight quant | KV quant |
| ------------------------ | ------- | ------------ | -------- |
| Voice dialogue, short    | 2-4k    | W4A16        | FP16     |
| RAG, medium context      | 8-16k   | W4A16 AWQ    | INT8     |
| Long doc Q&A             | 32-128k | W8A16        | INT8     |
| Coding agent             | 8-32k   | W8A16 or FP16| FP16     |


## 10. Hardware support: what the Tensor cores actually do

On Orin's Ampere GPU the math paths for LLM linears are:

```
Layer: Y = X @ W + b

                     X (activations)       W (weights)
W16A16 (FP16/BF16):  FP16 register          FP16 register
                     -> HMMA (FP16 TC)

W8A16:               FP16 register          INT8 storage -> dequant to FP16
                     -> HMMA (FP16 TC)

W8A8 (SmoothQuant):  INT8 register          INT8 register
                     -> IMMA (INT8 TC), accumulate in INT32
                     -> dequant to FP16 with per-channel scale

W4A16 (AWQ/GPTQ):    FP16 register          INT4 storage (packed 2/byte)
                     -> dequant to FP16 (group scale)
                     -> HMMA (FP16 TC)
```

Performance implications:

- **W4A16 and W8A16 are bandwidth wins.** Compute happens in FP16 either way. The speedup comes from fewer bytes loaded from DRAM.
- **W8A8 is a compute win.** The INT8 IMMA has double the throughput of HMMA FP16 on Ampere. This is why W8A8 can be faster than W4A16 even though it loads more bytes.
- **W4A8 would be the ideal.** On Ampere it is not well supported in stock kernels. Research kernels (e.g., Marlin-W4A8) exist but are not production-ready in TRT-LLM yet.

TRT-LLM's hand-tuned kernels do most of the heavy lifting. If you are not using TRT-LLM kernels on Orin, you are leaving 30-60% of your speedup on the table.


## 11. Practical recipe for Jetson AGX Orin

This is the playbook I run for a new humanoid project.

**Step 1: pick the base model.** 7B-8B dense is the comfort zone (Llama 3.1 8B, Qwen 2.5 7B, Mistral 7B). 13B is possible if you cut something else. 70B is not practical without aggressive offload, which defeats the edge purpose.

**Step 2: fine-tune in BF16 if needed.** LoRA adapters on top of the base in BF16. Merge into the base. Do not quantize yet.

**Step 3: run AWQ W4A16 with group=128** on a calibration set drawn from your task distribution. 128 samples, 2-4k tokens each.

**Step 4: build a TRT-LLM engine with paged KV cache, FP16 KV, max_batch=4, max_input=3584, max_output=512.** This is the default.

**Step 5: benchmark on the Orin.** Measure TTFT, decode tok/s, and peak RAM. Accept if TTFT < 350 ms and decode > 30 tok/s under your target power mode.

**Step 6: run your task eval at the quantized precision.** Not FP16 - the quantized model. Look especially at code, arithmetic, long-context retrieval, safety refusals, and instruction-following. If any of these regress unacceptably:
- Try SmoothQuant W8A8 instead.
- Or keep W4A16 but add a speculative decoder using a Phi-3.5 mini draft.
- Or move the specific hard task behind a cloud offload flag.

**Step 7: quantize KV if you need longer context or higher batch.** INT8 KV first, then evaluate. INT4 KV only if desperate.

**Step 8: add telemetry.** Log precision, KV precision, power mode, and per-request latency. You will need these when regressions appear in production.

**Step 9: plan your update path.** Quantized engines are tied to TRT-LLM version. Build the OTA around re-quantization and engine rebuild - do not ship a runtime that assumes one engine forever.


## 12. Speculative decoding + quantized draft models

Speculative decoding is the single best quality-at-latency technique for edge LLMs. You pair a small draft model with the full target model. The draft proposes `k` tokens, the target verifies them in parallel with one forward pass, keeps the accepted prefix, samples one more.

On Orin, a very effective setup:

```
Target: Llama 3.1 8B W4A16 AWQ (main dialogue quality)
Draft:  Llama 3.2 1B W4A16 AWQ (cheap, same tokenizer family)

Expected speedup: 1.5-2.2x on well-aligned tasks
Memory cost: +1.2 GB for the draft
```

The draft being W4A16 is critical - it has to be cheap enough that the whole pipeline still beats the target on its own. It being a **smaller model from the same family** matters because token distributions are similar, acceptance rates are high (60-80%), and you do not need a separate tokenizer.

TRT-LLM supports speculative decoding directly in the build and runtime. The pattern:

```bash
trtllm-build \
    --checkpoint_dir ./llama3-8b-awq \
    --output_dir ./engines/target \
    --gemm_plugin float16 \
    --paged_kv_cache enable \
    --speculative_decoding_mode draft_tokens_external \
    --max_draft_len 5

trtllm-build \
    --checkpoint_dir ./llama3.2-1b-awq \
    --output_dir ./engines/draft \
    --gemm_plugin float16 \
    --paged_kv_cache enable
```

Speculative decoding also rescues **quantization accuracy** for free: because the target verifies each token exactly, any token the draft gets wrong is rejected. A W4 model with speculative decoding can be harder to distinguish from its FP16 counterpart than a W4 model alone, at cost of some throughput.

A related technique worth mentioning: **self-speculative decoding** (Medusa, EAGLE heads). You train extra prediction heads on the target model that predict multiple future tokens in parallel. Saves the draft model memory. Integration is more work and the acceptance rate is usually slightly lower than a real draft, but on a memory-starved device it can be the right tradeoff.


## 13. Decision flowchart

```
Start: 7B-13B LLM, edge deployment on Orin
 |
 v
Is FP16 memory acceptable?
  yes -> Ship FP16. (Simplest, best quality, but rarely fits if ASR+TTS co-located.)
  no  -> continue
 |
 v
Does your task hit the W4 weak spots (code, math, 32k+ context)?
  no  -> AWQ W4A16, FP16 KV. Default. Ship.
  yes -> continue
 |
 v
Can you add speculative decoding?
  yes -> AWQ W4A16 + small AWQ W4 draft. Often good enough.
  no  -> continue
 |
 v
Can you afford 2x memory vs W4 for W8 path?
  yes -> SmoothQuant W8A8. Accuracy close to FP16, ~2x faster than FP16.
  no  -> continue
 |
 v
Can you offload hard tasks to cloud?
  yes -> W4 on-device + cloud escalation on confidence.
  no  -> Reduce task scope. No quantization will save you.
```


## 14. Senior-level takeaways

- Quantization is **bandwidth engineering** on memory-bound edge hardware. Decode speedups almost exactly track weight-byte reductions. Plan with a bandwidth budget, not a FLOPs budget.
- **AWQ W4A16 with group=128 is the 2026 production default** for 7B-13B dense LLMs on Ampere-class Jetson. AWQ's kernel alignment with TRT-LLM makes it win in deployment, not just accuracy.
- **W4 breaks code, arithmetic, long-context retrieval, and sometimes safety refusals first.** Measure at your target precision, not FP16. Perplexity does not tell you which tasks degrade.
- **KV cache quantization is a separate decision from weight quantization.** Start with FP16 KV, move to INT8 KV only if you need long context or larger batch.
- **SmoothQuant W8A8 is underused.** On Ampere it actually engages the IMMA INT8 Tensor core and can beat W4A16 on throughput with better accuracy. Consider it when you have the memory headroom.
- **Calibration data is half the result.** Match distribution, length, and chat template. 128 samples for AWQ, 512+ for SmoothQuant.
- **Speculative decoding is free accuracy.** A quantized draft + quantized target gives 1.5-2x speedup and recovers a large fraction of quantization error through rejection sampling.
- **Never quantize on the robot.** Quantize on a workstation, ship engines. Your OTA story must handle TRT-LLM version upgrades forcing re-quantization.
- **Rotation methods (QuaRot, SpinQuant) are the future for W4A4/W4A8** and will become standard once FP4-capable edge hardware ships widely. On Ampere today they are an interesting research path but not the production default.
- **Test safety, multi-turn, and instruction-following at the deployed precision.** Quantization can subtly shift refusal behavior and long-prompt compliance. These are the failures users notice; benchmarks do not always catch them.

Quantization is one of the three or four highest-leverage decisions you make in an edge LLM project. Understand the hardware math, pick the method that aligns with your runtime's kernels, calibrate on the right data, and evaluate at the deployed precision. The compounding wins between quantization, KV compression, and speculative decoding are what make an 8B model feel like a 30B model on a 30-watt SoC.


## References

- Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale", 2022.
- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (NF4, double quant), 2023.
- Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers", 2023.
- Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models", 2023.
- Lin et al., "AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration", 2024.
- Ashkboos et al., "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs", 2024.
- Liu et al., "SpinQuant: LLM Quantization with Learned Rotations", 2024.
- Leviathan et al., "Fast Inference from Transformers via Speculative Decoding", 2023.
- Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads", 2024.
- Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty", 2024.
- NVIDIA, "TensorRT-LLM Quantization Toolkit Documentation", 2024-2026.

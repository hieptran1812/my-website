---
title: "Unsloth: Fine-tune LLMs 2x Faster with 70% Less Memory"
publishDate: "2025-11-02"
category: "machine-learning"
subcategory: "Open Source Library"
tags: ["llm-training", "fine-tuning", "optimization", "lora", "qlora"]
date: "2025-11-02"
author: "Hiep Tran"
featured: false
image: ""
excerpt: "Unsloth is a lightweight library that makes LLM fine-tuning 2x faster with 70% less VRAM, fully compatible with the Hugging Face ecosystem."
---

## Introduction

**Unsloth** is a lightweight, high-performance library designed to dramatically accelerate LLM fine-tuning while reducing memory consumption. It achieves these gains through hand-optimized GPU kernels and mathematical derivations, all while maintaining full compatibility with the Hugging Face ecosystem (Hub, Transformers, PEFT, TRL).

Developed by Daniel and Michael Han, Unsloth has become one of the most widely used open-source frameworks for efficient LLM fine-tuning, particularly popular among researchers and practitioners with limited GPU resources.

## Deep-dive series: Inside Unsloth

This post is the overview. If you want to understand *how* Unsloth actually achieves these gains — kernel by kernel — the **Inside Unsloth** series takes the library apart:

1. [The anatomy of an Unsloth speedup](/blog/machine-learning/open-source-library/unsloth-speedup-anatomy) — where VRAM and time actually go, and the five levers
2. [Fused Triton kernels](/blog/machine-learning/open-source-library/unsloth-triton-kernel-fusion) — rewriting RMSNorm and SwiGLU
3. [Hand-derived backpropagation](/blog/machine-learning/open-source-library/unsloth-manual-backprop) — beating autograd
4. [RoPE and attention kernels](/blog/machine-learning/open-source-library/unsloth-rope-attention-kernels) — fused rotary embeddings
5. [4-bit NF4 quantization](/blog/machine-learning/open-source-library/unsloth-4bit-quantization-nf4) — shrinking the frozen base model
6. [The cross-entropy memory wall](/blog/machine-learning/open-source-library/unsloth-fused-cross-entropy) — never materializing the softmax
7. [Smart gradient checkpointing](/blog/machine-learning/open-source-library/unsloth-gradient-checkpointing-offload) — offloading activations to system RAM
8. [8-bit and paged optimizers](/blog/machine-learning/open-source-library/unsloth-8bit-paged-optimizers) — taming Adam's memory
9. [Long-context training on tiny VRAM](/blog/machine-learning/open-source-library/unsloth-long-context-training) — fitting 32K on a single GPU
10. [The full VRAM budget and export](/blog/machine-learning/open-source-library/unsloth-vram-budget-and-export) — assembling everything Unsloth saves

## Performance Overview

### Speed and Memory Improvements

| Metric | Improvement |
|--------|-------------|
| Training speed | Up to **2x faster** |
| VRAM usage | **70% less** memory |
| vs. Flash Attention 2 (single GPU) | Up to **10x faster** |
| vs. Flash Attention 2 (multi-GPU) | Up to **30x faster** |

### Accuracy Guarantee

**Zero accuracy degradation** compared to standard QLoRA - Unsloth makes no approximations in its optimized code.

## How It Works

Unsloth achieves its performance gains through several key optimizations:

### 1. Manual Backpropagation Derivation

Instead of relying on automatic differentiation, Unsloth manually derives all compute-heavy mathematical steps:

```
Standard Approach:
Forward Pass → AutoGrad Records Operations → Backward Pass (computed automatically)

Unsloth Approach:
Forward Pass → Manually Derived Gradients → Optimized Backward Pass
```

This eliminates overhead from the autograd graph and enables operation fusion.

### 2. Custom Triton Kernels

All PyTorch modules are rewritten as optimized Triton kernels:

```python
# Standard PyTorch (simplified)
def attention(q, k, v):
    scores = torch.matmul(q, k.transpose(-2, -1))
    scores = scores / math.sqrt(d_k)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v)

# Unsloth: Fused Triton kernel
# - Single kernel launch instead of multiple
# - Reduced memory bandwidth
# - Better GPU utilization
@triton.jit
def fused_attention_kernel(...):
    # All operations fused into single kernel
    pass
```

### 3. Memory-Efficient Operations

- Gradient checkpointing with minimal overhead
- Optimized memory allocation patterns
- Efficient handling of variable-length sequences

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Unsloth Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 User-Facing API                          │   │
│  │  FastLanguageModel.from_pretrained()                     │   │
│  │  FastLanguageModel.get_peft_model()                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                             │                                    │
│  ┌──────────────────────────▼──────────────────────────────┐   │
│  │              Optimized Model Patching                    │   │
│  │  • Replace attention with fused kernels                  │   │
│  │  • Replace MLP with optimized versions                   │   │
│  │  • Patch embedding and output layers                     │   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                             │                                    │
│  ┌──────────────────────────▼──────────────────────────────┐   │
│  │                Custom Triton Kernels                     │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │   │
│  │  │  Fused   │  │ Optimized│  │  Memory  │              │   │
│  │  │Attention │  │   MLP    │  │ Efficient│              │   │
│  │  │ Kernel   │  │  Kernel  │  │  Embeds  │              │   │
│  │  └──────────┘  └──────────┘  └──────────┘              │   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                             │                                    │
│  ┌──────────────────────────▼──────────────────────────────┐   │
│  │           Hugging Face Ecosystem Integration             │   │
│  │  Transformers │ PEFT │ TRL │ Datasets │ Hub             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Supported Models

### Text Models

| Model Family | Sizes | Status |
|--------------|-------|--------|
| Llama 3.3 | 70B | Fully supported |
| Llama 3.2 | 1B, 3B | Fully supported |
| Llama 3.1 | 8B, 70B, 405B | Fully supported |
| Qwen 2.5 | 0.5B - 72B | Fully supported |
| Qwen 3 | All sizes | Fully supported |
| DeepSeek-R1 | All sizes | Fully supported |
| Gemma 3 | All sizes | Fully supported |
| Mistral | 7B, 8x7B | Fully supported |
| Phi-3/4 | All sizes | Fully supported |

### Vision-Language Models

| Model | Size | Status |
|-------|------|--------|
| Llama 3.2 Vision | 11B | Supported |
| Qwen 2.5 VL | 7B | Supported |
| Pixtral | 12B | Supported |

### Other Models

- TTS (Text-to-Speech) models
- BERT and encoder models
- Custom architectures via extension API

## Installation

### Standard Installation

```bash
pip install unsloth
```

### With Specific CUDA Version

```bash
# For CUDA 12.1
pip install unsloth[cu121]

# For CUDA 11.8
pip install unsloth[cu118]
```

### From Source (Latest Features)

```bash
pip install git+https://github.com/unslothai/unsloth.git
```

### Conda Installation

```bash
conda install -c conda-forge unsloth
```

## Quick Start

### Basic Fine-Tuning

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# Load model with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.1-8B-bnb-4bit",
    max_seq_length=2048,
    dtype=None,  # Auto-detect
    load_in_4bit=True,
)

# Apply LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                      # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,            # Optimized for 0 dropout
    bias="none",
    use_gradient_checkpointing="unsloth",  # Use Unsloth's optimized checkpointing
    random_state=42,
)

# Load dataset
dataset = load_dataset("yahma/alpaca-cleaned", split="train")

# Format dataset
def format_prompt(example):
    return {
        "text": f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
    }

dataset = dataset.map(format_prompt)

# Configure training
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    max_steps=100,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    optim="adamw_8bit",
    save_strategy="steps",
    save_steps=50,
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=training_args,
)

# Train
trainer.train()
```

### Saving and Loading Models

```python
# Save LoRA adapters
model.save_pretrained("./lora_model")
tokenizer.save_pretrained("./lora_model")

# Save merged model (for inference without Unsloth)
model.save_pretrained_merged(
    "./merged_model",
    tokenizer,
    save_method="merged_16bit",  # or "merged_4bit", "lora"
)

# Save as GGUF for llama.cpp
model.save_pretrained_gguf(
    "./gguf_model",
    tokenizer,
    quantization_method="q4_k_m",  # Various quantization options
)
```

### Inference

```python
from unsloth import FastLanguageModel

# Load fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./lora_model",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Enable faster inference
FastLanguageModel.for_inference(model)

# Generate
inputs = tokenizer(
    "### Instruction:\nExplain machine learning.\n\n### Response:\n",
    return_tensors="pt"
).to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    do_sample=True,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 2025 Features

### Reinforcement Learning Support

Unsloth is now the most efficient library for RL fine-tuning:

```python
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length=4096,
    load_in_4bit=True,
)

# Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
)

# GRPO training with 80% less VRAM than alternatives
config = GRPOConfig(
    output_dir="./grpo_output",
    per_device_train_batch_size=2,
    num_generations=4,
    learning_rate=1e-6,
)

trainer = GRPOTrainer(
    model=model,
    args=config,
    tokenizer=tokenizer,
    reward_funcs=my_reward_function,
    train_dataset=prompts_dataset,
)

trainer.train()
```

Supported RL algorithms:
- **GRPO** (Group Relative Policy Optimization)
- **GSPO** (Group Soft Preference Optimization)
- **DrGRPO** (Direct Reward GRPO)
- **DAPO** (Direct Alignment from Preferences Optimization)

### Long-Context Reasoning

Train reasoning models with minimal VRAM:

```python
from unsloth import FastLanguageModel

# Train 32K context reasoning model with only 5GB VRAM
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.1-8B-bnb-4bit",
    max_seq_length=32768,  # 32K context
    load_in_4bit=True,
)

# Enable RoPE scaling for long context
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    use_gradient_checkpointing="unsloth",
    rope_scaling={
        "type": "dynamic",
        "factor": 4.0,  # Scale to 4x original context
    },
)
```

### Vision Model Fine-Tuning

```python
from unsloth import FastVisionModel

# Load vision-language model
model, tokenizer = FastVisionModel.from_pretrained(
    model_name="unsloth/Llama-3.2-11B-Vision-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Apply LoRA for vision fine-tuning
model = FastVisionModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    finetune_vision_layers=True,  # Also fine-tune vision encoder
)
```

## VRAM Requirements

| Model | Standard | Unsloth | Savings |
|-------|----------|---------|---------|
| Llama 3.1 8B (QLoRA) | 24GB | 8GB | 67% |
| Llama 3.1 70B (QLoRA) | 80GB | 48GB | 40% |
| Qwen 2.5 7B (QLoRA) | 18GB | 6GB | 67% |
| Reasoning (32K ctx) | 40GB | 5GB | 87% |

### Free Tier Compatibility

| Platform | GPU | VRAM | Compatible |
|----------|-----|------|------------|
| Google Colab (Free) | T4 | 15GB | Yes |
| Kaggle | P100/T4 | 16GB | Yes |
| Lightning AI | T4 | 16GB | Yes |

## Hardware Support

### NVIDIA GPUs

| GPU Generation | Support Level |
|----------------|--------------|
| H100/H200 | Full support, optimized |
| A100 | Full support, optimized |
| A10/A10G | Full support |
| RTX 4090/4080 | Full support |
| RTX 3090/3080 | Full support |
| RTX A6000/A5000 | Full support |
| T4 | Full support |
| V100 | Supported |

### Other Platforms

- **AMD GPUs**: Portable via ROCm
- **Intel GPUs**: Experimental support
- **Apple Silicon**: Via MLX export

## Comparison with Alternatives

| Feature | Unsloth | Standard HF | Axolotl | LLaMA-Factory |
|---------|---------|-------------|---------|---------------|
| Speed | 2x faster | Baseline | ~1.2x | ~1.3x |
| VRAM reduction | 70% | - | 20-30% | 30-40% |
| Multi-GPU | Yes | Yes | Yes | Yes |
| Accuracy loss | 0% | - | 0% | 0% |
| Setup complexity | Low | Low | Medium | Low |
| Free tier compatible | Yes | Limited | Limited | Limited |

**Note**: Multi-GPU training is now supported in Unsloth (with a larger upgrade in progress); earlier releases were single-GPU only. Single-GPU efficiency remains Unsloth's headline strength — the 2x faster / 70% less VRAM figures are all measured on one card.

## Best Practices

### 1. Optimal LoRA Configuration

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                      # 8-64, higher = more capacity
    lora_alpha=16,             # Usually equal to r
    lora_dropout=0,            # 0 is optimized in Unsloth
    target_modules=[           # Cover all linear layers
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
```

### 2. Memory Optimization

```python
# Use 8-bit optimizer
training_args = TrainingArguments(
    optim="adamw_8bit",        # 8-bit Adam
    gradient_accumulation_steps=4,  # Accumulate if batch doesn't fit
)

# Or use paged optimizer for extreme memory savings
training_args = TrainingArguments(
    optim="paged_adamw_8bit",
)
```

### 3. Dataset Packing

```python
# Pack multiple samples into single sequences for efficiency
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    packing=True,              # Enable packing
    max_seq_length=2048,
)
```

## Resources

- **Website**: [unsloth.ai](https://unsloth.ai/)
- **GitHub**: [unslothai/unsloth](https://github.com/unslothai/unsloth)
- **Documentation**: [docs.unsloth.ai](https://docs.unsloth.ai/)
- **Colab Notebooks**: [unsloth.ai/notebooks](https://unsloth.ai/notebooks)
- **Discord**: [discord.gg/unsloth](https://discord.gg/unsloth)
- **Hugging Face Blog**: [Make LLM Fine-tuning 2x faster with Unsloth](https://huggingface.co/blog/unsloth-trl)

## Citation

```bibtex
@software{unsloth2024,
  author = {Han, Daniel and Han, Michael},
  title = {Unsloth: Fine-tune LLMs 2x faster with 70% less memory},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/unslothai/unsloth}
}
```

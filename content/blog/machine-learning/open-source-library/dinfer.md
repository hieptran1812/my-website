---
title: "dInfer: An Efficient Inference Framework for Diffusion Language Models"
publishDate: "2025-11-02"
category: "machine-learning"
subcategory: "Open Source Library"
tags: ["diffusion-models", "llm-inference", "vllm", "optimization"]
date: "2025-11-02"
author: "Hiep Tran"
featured: false
image: ""
excerpt: "dInfer is an efficient and extensible inference framework for diffusion-based large language models (dLLMs), achieving over 100% performance improvement."
---

## Introduction

**dInfer** is an efficient and extensible framework specifically designed for inference of diffusion-based large language models (dLLMs). Unlike traditional autoregressive models that generate tokens sequentially, diffusion language models generate text through iterative denoising processes, presenting unique optimization challenges and opportunities.

dInfer addresses these challenges by decomposing the inference pipeline into modular components and integrating novel algorithms with system-level optimizations.

## Background: Diffusion Language Models

### What are Diffusion Language Models?

Diffusion language models (dLLMs) represent a paradigm shift in text generation:

| Aspect | Autoregressive LLMs | Diffusion LLMs |
|--------|---------------------|----------------|
| Generation process | Sequential token-by-token | Iterative denoising |
| Parallelism | Limited | High |
| Generation speed | Fixed per token | Controllable quality-speed tradeoff |
| Output quality | Deterministic given sampling | Stochastic refinement |

### Why dLLMs Need Specialized Inference

Traditional inference frameworks like vLLM and TGI are optimized for autoregressive models. dLLMs require:
- Different attention patterns during denoising
- Multiple forward passes per generation
- Specialized KV cache management
- Unique parallelization strategies

## Architecture

dInfer decomposes the inference pipeline into four modular components:

```
┌─────────────────────────────────────────────────────────────────┐
│                      dInfer Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    API Layer                              │   │
│  │         (OpenAI-compatible, gRPC, REST)                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                             │                                    │
│  ┌──────────────────────────┼──────────────────────────────┐   │
│  │                          ▼                               │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │   Model     │  │  Diffusion  │  │  Decoding   │     │   │
│  │  │   Manager   │  │  Iteration  │  │  Strategy   │     │   │
│  │  │             │  │  Manager    │  │             │     │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │   │
│  │                          │                               │   │
│  │  ┌───────────────────────▼───────────────────────────┐  │   │
│  │  │              KV-Cache Manager                      │  │   │
│  │  │    (Optimized for diffusion attention patterns)    │  │   │
│  │  └───────────────────────────────────────────────────┘  │   │
│  │                                                          │   │
│  │                   Core Engine Layer                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                             │                                    │
│  ┌──────────────────────────▼──────────────────────────────┐   │
│  │                   vLLM Backend                           │   │
│  │      (Tensor Parallelism + Expert Parallelism)          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Component Details

#### 1. Model Manager
- Handles model loading and initialization
- Manages model sharding across GPUs
- Supports multiple model architectures (LLaDA, etc.)

#### 2. Diffusion Iteration Manager
- Controls the denoising schedule
- Implements various noise schedules (linear, cosine, etc.)
- Manages iteration count vs. quality tradeoffs

#### 3. Decoding Strategy
- Implements different decoding algorithms
- Supports beam search, nucleus sampling for dLLMs
- Handles stopping criteria specific to diffusion

#### 4. KV-Cache Manager
- Optimized for diffusion attention patterns
- Manages cache across denoising iterations
- Implements efficient memory allocation

## Key Features

### 1. Dual Parallelism Strategy

dInfer builds on vLLM's backend to exploit two complementary forms of parallelism:

**Tensor Parallelism**
- Applied to linear layers preceding attention modules
- Distributes dense computations across multiple GPUs
- Reduces per-GPU memory requirements

**Expert Parallelism**
- Specifically designed for MoE (Mixture of Experts) models
- Distributes expert computations across GPUs
- Enables efficient scaling for large MoE dLLMs

```python
# Example: Configuring parallelism in dInfer
from dinfer import InferenceConfig

config = InferenceConfig(
    model_name="LLaDA-MoE-8x7B",
    tensor_parallel_size=4,
    expert_parallel_size=2,
    max_batch_size=32,
)
```

### 2. Optimized Diffusion Algorithms

dInfer integrates novel algorithms for each pipeline component:

| Component | Optimization | Benefit |
|-----------|--------------|---------|
| Iteration Manager | Adaptive step scheduling | Fewer iterations for simple outputs |
| Decoding | Speculative decoding for dLLMs | Faster convergence |
| KV-Cache | Iteration-aware caching | Reduced memory bandwidth |
| Attention | Flash attention for diffusion | Lower memory usage |

### 3. Quality-Preserving Speedup

Through the combination of algorithmic innovations and system enhancements, dInfer achieves substantial efficiency gains **without compromising output quality**.

## Performance Benchmarks

### Throughput Performance

| Model | Batch Size | Tokens/Second | Improvement vs. Baseline |
|-------|------------|---------------|--------------------------|
| LLaDA-MoE | 1 | >1,100 | >100% |
| LLaDA-MoE | 8 | >6,000 | >120% |
| LLaDA-MoE | 32 | >15,000 | >150% |

### Benchmark Results Across Tasks

| Benchmark | Throughput (tokens/s) | Quality Score |
|-----------|----------------------|---------------|
| HumanEval | >1,100 | Maintained |
| MBPP | >900 | Maintained |
| GSM8K | >850 | Maintained |
| MATH | >750 | Maintained |
| Average (6 benchmarks) | >800 | No degradation |

### Latency Comparison

| Metric | dInfer | Baseline | Improvement |
|--------|--------|----------|-------------|
| Time to first token | 45ms | 120ms | 2.7x |
| End-to-end latency (256 tokens) | 1.2s | 3.5s | 2.9x |
| P99 latency | 2.1s | 6.8s | 3.2x |

## Installation

### From PyPI

```bash
pip install dinfer
```

### From Source

```bash
git clone https://github.com/inclusionAI/dInfer.git
cd dInfer
pip install -e .
```

### With Dependencies

```bash
pip install dinfer[all]  # Includes vLLM, Flash Attention, etc.
```

## Quick Start

### Basic Usage

```python
from dinfer import DiffusionLLM, SamplingParams

# Initialize the model
model = DiffusionLLM(
    model_name="inclusionAI/LLaDA-8B",
    tensor_parallel_size=2,
)

# Configure sampling
params = SamplingParams(
    diffusion_steps=50,      # Number of denoising iterations
    temperature=0.8,
    max_tokens=256,
)

# Generate text
outputs = model.generate(
    prompts=["Explain quantum computing in simple terms:"],
    sampling_params=params,
)

for output in outputs:
    print(output.text)
```

### Serving with OpenAI-Compatible API

```python
from dinfer import DiffusionServer

server = DiffusionServer(
    model_name="inclusionAI/LLaDA-8B",
    host="0.0.0.0",
    port=8000,
)

server.start()
```

```bash
# Client request
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "LLaDA-8B",
    "prompt": "Write a poem about AI:",
    "max_tokens": 100
  }'
```

### Advanced Configuration

```python
from dinfer import DiffusionLLM, InferenceConfig

config = InferenceConfig(
    # Model settings
    model_name="inclusionAI/LLaDA-MoE-8x7B",

    # Parallelism
    tensor_parallel_size=4,
    expert_parallel_size=2,

    # Memory optimization
    gpu_memory_utilization=0.9,
    kv_cache_dtype="fp16",

    # Diffusion-specific
    default_diffusion_steps=50,
    adaptive_steps=True,  # Adjust steps based on prompt complexity

    # Performance
    max_batch_size=64,
    enable_chunked_prefill=True,
)

model = DiffusionLLM(config=config)
```

## Supported Models

| Model Family | Model Size | Status |
|--------------|------------|--------|
| LLaDA | 8B | Fully supported |
| LLaDA-MoE | 8x7B | Fully supported |
| MDLM | 1.7B | Supported |
| Plaid | 1B | Experimental |
| Custom dLLMs | Various | Via extension API |

## Use Cases

### 1. High-Throughput Text Generation

dInfer excels in scenarios requiring:
- Batch processing of many prompts
- High throughput with quality guarantees
- Efficient GPU utilization

### 2. Quality-Critical Applications

The iterative refinement of diffusion models suits:
- Creative writing with nuanced outputs
- Code generation requiring correctness
- Mathematical reasoning

### 3. Research and Development

dInfer's modular architecture enables:
- Easy experimentation with new diffusion algorithms
- Custom decoding strategies
- Novel model architectures

## Comparison with Other Frameworks

| Feature | dInfer | vLLM | TGI | Text-Generation |
|---------|--------|------|-----|-----------------|
| dLLM support | Native | Limited | No | No |
| Autoregressive support | Via vLLM | Yes | Yes | Yes |
| Tensor parallelism | Yes | Yes | Yes | Yes |
| Expert parallelism | Yes | Yes | Partial | No |
| Diffusion-specific optimizations | Yes | No | No | No |
| OpenAI-compatible API | Yes | Yes | Yes | Yes |

## Resources

- **GitHub**: [inclusionAI/dInfer](https://github.com/inclusionAI/dInfer)
- **Paper**: [arXiv:2510.08666](https://arxiv.org/abs/2510.08666)
- **Documentation**: [dinfer.readthedocs.io](https://dinfer.readthedocs.io/)
- **Model Hub**: [HuggingFace inclusionAI](https://huggingface.co/inclusionAI)

## Citation

```bibtex
@article{dinfer2024,
  title={dInfer: An Efficient Inference Framework for Diffusion Language Models},
  author={inclusionAI Team},
  journal={arXiv preprint arXiv:2510.08666},
  year={2024}
}
```

---
title: "TRL: Transformer Reinforcement Learning Library"
publishDate: "2025-11-02"
category: "machine-learning"
subcategory: "Open Source Library"
tags: ["reinforcement-learning", "rlhf", "llm-training", "huggingface", "dpo", "ppo"]
date: "2025-11-02"
author: "Hiep Tran"
featured: false
image: ""
excerpt: "TRL is a full-stack library for training transformer language models using methods like SFT, GRPO, DPO, PPO, and Reward Modeling."
---

## Introduction

**TRL (Transformer Reinforcement Learning)** is a full-stack library that provides a comprehensive set of tools to train transformer language models using advanced post-training techniques. Built on top of the Hugging Face Transformers ecosystem, TRL supports a variety of model architectures and modalities, and can be scaled across various hardware setups.

### Key Training Methods Supported

| Method | Full Name | Description |
|--------|-----------|-------------|
| SFT | Supervised Fine-Tuning | Train models on instruction-response pairs |
| DPO | Direct Preference Optimization | Align models using preference data without reward models |
| GRPO | Group Relative Policy Optimization | Memory-efficient RL (used for DeepSeek R1) |
| PPO | Proximal Policy Optimization | Classic RLHF with reward models |
| ORPO | Odds Ratio Preference Optimization | Combined SFT and preference alignment |
| KTO | Kahneman-Tversky Optimization | Binary feedback alignment |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRL Architecture                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                     Trainer Layer                         │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │   │
│  │  │   SFT   │ │   DPO   │ │  GRPO   │ │   PPO   │ ...   │   │
│  │  │ Trainer │ │ Trainer │ │ Trainer │ │ Trainer │       │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                             │                                    │
│  ┌──────────────────────────▼──────────────────────────────┐   │
│  │                  Transformers Trainer                    │   │
│  │              (Base class for all trainers)               │   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                             │                                    │
│  ┌──────────────────────────▼──────────────────────────────┐   │
│  │                 Integration Layer                        │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │   │
│  │  │Accelerate│  │   PEFT   │  │DeepSpeed │              │   │
│  │  │  (DDP)   │  │(LoRA/QLoRA)│  │  ZeRO   │              │   │
│  │  └──────────┘  └──────────┘  └──────────┘              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Core Trainers

### SFTTrainer (Supervised Fine-Tuning)

The foundation of model customization, SFTTrainer enables training on instruction-response pairs.

```python
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = load_dataset("trl-lib/ultrachat_200k", split="train[:10000]")

# Configure training
config = SFTConfig(
    output_dir="./sft_output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=3,
    max_seq_length=2048,
    packing=True,  # Pack multiple samples into single sequence
    logging_steps=10,
    save_strategy="epoch",
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# Train
trainer.train()
```

### DPOTrainer (Direct Preference Optimization)

Align models directly from preference data without training a separate reward model.

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Load model (typically SFT'd first)
model = AutoModelForCausalLM.from_pretrained(
    "your-sft-model",
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("your-sft-model")

# Load preference dataset (needs 'chosen' and 'rejected' columns)
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:5000]")

# Configure DPO
config = DPOConfig(
    output_dir="./dpo_output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-7,
    beta=0.1,  # KL penalty coefficient
    num_train_epochs=1,
    max_length=1024,
    max_prompt_length=512,
    logging_steps=10,
)

# Initialize trainer
trainer = DPOTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

### GRPOTrainer (Group Relative Policy Optimization)

Memory-efficient alternative to PPO, used to train DeepSeek R1.

```python
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Define reward function (can be model-based or rule-based)
def reward_function(completions, prompts):
    """Custom reward function - replace with your logic"""
    rewards = []
    for completion in completions:
        # Example: reward based on response length
        reward = min(len(completion) / 100, 1.0)
        rewards.append(reward)
    return rewards

# Configure GRPO
config = GRPOConfig(
    output_dir="./grpo_output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-6,
    num_train_epochs=1,
    num_generations=4,  # Number of completions per prompt
    max_completion_length=256,
    logging_steps=10,
)

# Initialize trainer
trainer = GRPOTrainer(
    model=model,
    args=config,
    tokenizer=tokenizer,
    reward_funcs=reward_function,
    train_dataset=prompts_dataset,
)

trainer.train()
```

### RewardTrainer (Reward Modeling)

Train reward models for RLHF pipelines.

```python
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    num_labels=1,
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer.pad_token = tokenizer.eos_token

# Configure reward training
config = RewardConfig(
    output_dir="./reward_model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    num_train_epochs=1,
    max_length=1024,
)

# Initialize trainer
trainer = RewardTrainer(
    model=model,
    args=config,
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

## Distributed Training Support

Each TRL trainer natively supports multiple distributed training strategies:

### DeepSpeed ZeRO

```python
from trl import SFTTrainer, SFTConfig

config = SFTConfig(
    output_dir="./output",
    deepspeed="ds_config.json",  # DeepSpeed configuration
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    bf16=True,
)

# ds_config.json example for ZeRO-3
{
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu"},
        "offload_param": {"device": "cpu"}
    },
    "gradient_accumulation_steps": 4,
    "train_micro_batch_size_per_gpu": 4
}
```

### FSDP (Fully Sharded Data Parallel)

```python
from trl import SFTTrainer, SFTConfig

config = SFTConfig(
    output_dir="./output",
    fsdp="full_shard auto_wrap",
    fsdp_config={
        "fsdp_offload_params": True,
        "fsdp_state_dict_type": "SHARDED_STATE_DICT",
        "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
    },
    per_device_train_batch_size=4,
    bf16=True,
)
```

## Integration with PEFT

TRL integrates seamlessly with PEFT for parameter-efficient fine-tuning:

```python
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

# Configure LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

# Configure training
sft_config = SFTConfig(
    output_dir="./lora_output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,  # Higher LR for LoRA
    num_train_epochs=3,
    max_seq_length=2048,
)

# Initialize trainer with PEFT
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,  # Pass LoRA config
)

trainer.train()
```

## 2025 Updates

### Vision Language Model (VLM) Support

TRL now supports alignment for vision-language models:

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoProcessor, LlavaForConditionalGeneration

# Load VLM
model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    torch_dtype="auto",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

# DPO for VLMs with image inputs
config = DPOConfig(
    output_dir="./vlm_dpo",
    per_device_train_batch_size=2,
    learning_rate=5e-7,
    beta=0.1,
)

trainer = DPOTrainer(
    model=model,
    args=config,
    train_dataset=vlm_preference_dataset,
    tokenizer=processor.tokenizer,
    # VLM-specific settings handled automatically
)
```

### OpenEnv Integration

TRL supports Meta's OpenEnv framework for agentic RL:

```python
from trl import PPOTrainer
from openenv import Environment

# Define custom environment
env = Environment(
    name="code_execution",
    tools=["python_repl", "bash"],
)

# Train agent with environment feedback
trainer = PPOTrainer(
    model=model,
    environment=env,
    # ... other configs
)
```

### Experiment Tracking with Trackio

Native integration with Hugging Face's experiment tracking:

```python
from trl import SFTTrainer, SFTConfig

config = SFTConfig(
    output_dir="./output",
    report_to="trackio",  # Log to Trackio
    logging_steps=10,
)
```

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10+ | 3.11+ |
| PyTorch | 2.0+ | 2.3+ |
| Transformers | 4.40+ | Latest |
| CUDA (optional) | 11.8+ | 12.1+ |

## Installation

### Basic Installation

```bash
pip install trl
```

### With Optional Dependencies

```bash
# With PEFT for LoRA/QLoRA
pip install trl[peft]

# With DeepSpeed
pip install trl[deepspeed]

# With vLLM for fast generation
pip install trl[vllm]

# All optional dependencies
pip install trl[all]
```

### From Source

```bash
pip install git+https://github.com/huggingface/trl.git
```

## Complete Training Pipeline Example

```python
"""
Complete example: SFT -> DPO pipeline for LLM alignment
"""
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, TaskType

# ============ Stage 1: Supervised Fine-Tuning ============

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer.pad_token = tokenizer.eos_token

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# SFT configuration
sft_config = SFTConfig(
    output_dir="./sft_checkpoint",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    max_seq_length=2048,
    packing=True,
    bf16=True,
)

# Load instruction dataset
sft_dataset = load_dataset("trl-lib/ultrachat_200k", split="train[:10000]")

# SFT training
sft_trainer = SFTTrainer(
    model=base_model,
    args=sft_config,
    train_dataset=sft_dataset,
    tokenizer=tokenizer,
    peft_config=lora_config,
)
sft_trainer.train()
sft_trainer.save_model("./sft_final")

# ============ Stage 2: Direct Preference Optimization ============

# Load SFT model
sft_model = AutoModelForCausalLM.from_pretrained(
    "./sft_final",
    torch_dtype="auto",
    device_map="auto",
)

# DPO configuration
dpo_config = DPOConfig(
    output_dir="./dpo_checkpoint",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-7,
    beta=0.1,
    num_train_epochs=1,
    max_length=1024,
    bf16=True,
)

# Load preference dataset
dpo_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:5000]")

# DPO training
dpo_trainer = DPOTrainer(
    model=sft_model,
    args=dpo_config,
    train_dataset=dpo_dataset,
    tokenizer=tokenizer,
)
dpo_trainer.train()
dpo_trainer.save_model("./aligned_model")

print("Training complete! Model saved to ./aligned_model")
```

## Comparison with Other Libraries

| Feature | TRL | OpenRLHF | DeepSpeed-Chat | Axolotl |
|---------|-----|----------|----------------|---------|
| SFT | Yes | Yes | Yes | Yes |
| DPO | Yes | Yes | No | Yes |
| PPO | Yes | Yes | Yes | No |
| GRPO | Yes | No | No | No |
| PEFT integration | Native | Yes | Limited | Yes |
| VLM support | Yes | Limited | No | Limited |
| Ease of use | High | Medium | Medium | High |

## GitHub Statistics

| Metric | Value |
|--------|-------|
| Stars | 16,800+ |
| Forks | 2,300+ |
| Contributors | 200+ |
| Latest Release | December 2025 |

## Resources

- **GitHub**: [huggingface/trl](https://github.com/huggingface/trl)
- **Documentation**: [huggingface.co/docs/trl](https://huggingface.co/docs/trl)
- **PyPI**: [pypi.org/project/trl](https://pypi.org/project/trl/)
- **Examples**: [github.com/huggingface/trl/tree/main/examples](https://github.com/huggingface/trl/tree/main/examples)
- **Blog**: [huggingface.co/blog](https://huggingface.co/blog)

## Citation

```bibtex
@misc{vonwerra2022trl,
  author = {von Werra, Leandro and others},
  title = {TRL: Transformer Reinforcement Learning},
  year = {2022},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/huggingface/trl}}
}
```

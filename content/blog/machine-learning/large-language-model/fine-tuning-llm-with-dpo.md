---
title: "Fine-Tuning LLMs with DPO: A Practical Guide from Theory to Production"
publishDate: "2026-03-18"
category: "machine-learning"
subcategory: "Large Language Model"
tags: ["llm", "dpo", "rlhf", "fine-tuning", "alignment", "preference-optimization", "trl", "unsloth", "axolotl", "deep-learning"]
date: "2026-03-18"
author: "Hiep Tran"
featured: false
aiGenerated: true
excerpt: "Direct Preference Optimization (DPO) is the most practical way to align LLMs with human preferences — no reward model, no RL loop, just a clever loss function. This guide covers the theory, the math, every major training framework (TRL, Unsloth, Axolotl, LLaMA-Factory), hardware planning, and hard-won lessons from production."
---

## The Alignment Problem in One Paragraph

You've pretrained (or downloaded) a large language model. It can generate fluent text. But it doesn't *behave* the way you want — it might be verbose, refuse reasonable requests, hallucinate confidently, or produce outputs that don't match your domain's style. **Alignment** is the process of teaching the model to prefer certain behaviors over others. The gold standard has been RLHF (Reinforcement Learning from Human Feedback), but it's complex, unstable, and expensive. DPO achieves the same goal with a single supervised training step.

This isn't a toy technique. Llama 3, Zephyr, Intel's Neural Chat, Tulu 2, and many production systems use DPO (or its variants) as the core alignment method. If you're building anything that involves an LLM generating text for users, you need to understand this.

## From RLHF to DPO: Why DPO Exists

![RLHF vs DPO: RLHF trains a reward model then runs PPO rollouts with policy + value + ref networks; DPO collapses it into a single supervised loss using only the reference policy and preference pairs](/imgs/blogs/dpo-01-rlhf-vs-dpo.png)

To understand DPO, you need to understand what it replaces.

### The RLHF Pipeline

The standard RLHF pipeline has three stages:

```
Stage 1: Supervised Fine-Tuning (SFT)
    → Train on high-quality prompt-response pairs
    → Gives the model basic instruction-following ability

Stage 2: Reward Model Training
    → Train a separate model to score outputs
    → Needs ranked preference data: (prompt, chosen, rejected)
    → The reward model learns to predict which response humans prefer

Stage 3: RL Optimization (PPO)
    → Use the reward model to generate rewards
    → Optimize the policy (LLM) with PPO
    → KL penalty to prevent drift from the SFT model
    → Requires policy, reference, reward, and value models simultaneously
```

This works — it's how ChatGPT was trained. But every stage introduces failure modes:

| Stage | Problem | Impact |
|-------|---------|--------|
| Reward Model | Reward hacking — the model exploits shortcuts in the reward model | Model learns to game the metric rather than be genuinely helpful |
| Reward Model | Calibration drift — reward model scores become unreliable outside training distribution | Optimization goes off-track for novel prompts |
| PPO | Hyperparameter sensitivity — learning rate, KL coefficient, clip range, batch size all interact | Training runs fail silently or produce degraded models |
| PPO | Training instability — reward collapse, mode collapse, divergence | Wasted GPU hours and unpredictable outcomes |
| Infrastructure | You need 4 models in memory simultaneously (policy, reference, reward, value) | For a 7B model, that's ~56GB just for the weights in FP16 |
| Cost | 2-4x more GPU hours than supervised training | A 70B RLHF run can cost $50K+ in compute |

### The DPO Insight

The key paper — [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) (Rafailov et al., 2023) — made a beautiful observation:

**You don't need an explicit reward model.** The optimal policy under the RLHF objective has a closed-form relationship with the reward function. By rearranging the math, you can derive a loss function that directly optimizes the policy using preference pairs — no reward model, no RL loop.

The intuition: instead of learning "this response scores 0.8 and that one scores 0.3" (reward modeling) and then optimizing against that score (RL), you directly learn "this response should be more likely than that one" (classification-like loss). This collapses three stages into one.

```
RLHF Pipeline:               DPO Pipeline:
┌──────────┐                  ┌──────────┐
│   SFT    │                  │   SFT    │
└────┬─────┘                  └────┬─────┘
     │                              │
┌────▼─────┐                  ┌────▼─────┐
│  Reward  │                  │   DPO    │  ← Single step!
│  Model   │                  │ Training │
└────┬─────┘                  └────┬─────┘
     │                              │
┌────▼─────┐                  ┌────▼─────┐
│   PPO    │                  │  Done!   │
│ Training │                  └──────────┘
└────┬─────┘
     │
┌────▼─────┐
│  Done!   │
└──────────┘
```

### Why This Matters Practically

| Metric | RLHF (PPO) | DPO |
|--------|-----------|-----|
| Models in memory | 4 (policy + ref + reward + value) | 2 (policy + ref) |
| Training stability | Fragile, many failure modes | Stable, similar to SFT |
| Hyperparameters to tune | 10+ (LR, KL coeff, clip range, epochs, GAE lambda, value coeff, ...) | 3-4 (beta, LR, epochs, batch size) |
| Implementation complexity | High (RL loop, rollouts, advantage estimation) | Low (single loss function) |
| GPU hours (7B model, 50K examples) | ~200-400 A100-hours | ~50-100 A100-hours |
| Performance | Slightly better ceiling in some tasks | Comparable in most benchmarks |

## The Math Behind DPO

![DPO loss computation: for each preference pair, compute Δ = β·(log π_θ − log π_ref) for both chosen and rejected, then L = −log σ(Δ_chosen − Δ_rejected) and backprop](/imgs/blogs/dpo-02-loss.png)

Let's build up the derivation step by step. Understanding the math helps you debug training issues and choose the right hyperparameters.

### Starting Point: The RLHF Objective

In RLHF, we want to find a policy $\pi_\theta$ that maximizes the expected reward while staying close to a reference policy $\pi_{\text{ref}}$ (the SFT model):

$$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y|x)} \left[ r(x, y) \right] - \beta \, \text{KL}\left[\pi_\theta(y|x) \| \pi_{\text{ref}}(y|x)\right]$$

Where:
- $r(x, y)$ is the reward for generating response $y$ given prompt $x$
- $\beta$ controls how far the policy can drift from the reference
- $\text{KL}[\cdot \| \cdot]$ is the Kullback-Leibler divergence

The KL term is critical — without it, the model would collapse to always generating the single highest-reward response, losing diversity and generalization. The $\beta$ parameter balances "be good" (maximize reward) with "be safe" (don't deviate too far from what you already know).

### The Closed-Form Solution

This constrained optimization problem has a known analytical solution. The optimal policy is:

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)$$

Where $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)$ is the partition function (normalization constant).

**Intuition**: The optimal policy takes the reference model's distribution and reweights it exponentially by the reward. High-reward responses get boosted, low-reward responses get suppressed. The $\beta$ controls how aggressively this reweighting happens.

Rearranging for the reward:

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

This tells us something profound: **the reward is implicitly encoded in the log-ratio of the optimal policy to the reference policy**. You don't need a separate reward model — the policy *is* the reward model.

### The DPO Loss

Now comes the clever part. Human preferences are modeled using the Bradley-Terry model:

$$p(y_w \succ y_l | x) = \sigma\left(r(x, y_w) - r(x, y_l)\right)$$

Where $y_w$ is the preferred (chosen) response, $y_l$ is the dispreferred (rejected) response, and $\sigma$ is the sigmoid function.

Substituting the reward expression into the Bradley-Terry model, the $Z(x)$ terms **cancel out** (this is the key mathematical insight — the intractable partition function vanishes!), giving us:

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

In plain English: **increase the relative probability of the chosen response and decrease the relative probability of the rejected response, both measured against the reference model.**

Let's define the **implicit reward** for convenience:

$$\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$$

Then the loss simplifies to:

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E} \left[ \log \sigma \left( \hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l) \right) \right]$$

This is just a **binary cross-entropy loss** on the implicit reward margin. The model should assign a higher implicit reward to the chosen response than the rejected response. That's it.

### What the Gradient Looks Like

The gradient of the DPO loss has an intuitive form:

$$\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \, \mathbb{E} \left[ \underbrace{\sigma(\hat{r}_\theta(y_l) - \hat{r}_\theta(y_w))}_{\text{higher weight when model is wrong}} \left[ \underbrace{\nabla_\theta \log \pi_\theta(y_w|x)}_{\text{increase chosen likelihood}} - \underbrace{\nabla_\theta \log \pi_\theta(y_l|x)}_{\text{decrease rejected likelihood}} \right] \right]$$

The weighting term $\sigma(\hat{r}_\theta(y_l) - \hat{r}_\theta(y_w))$ is crucial — it means the model focuses on examples **it currently gets wrong**. If the model already assigns higher implicit reward to the chosen response, the gradient is small. If it assigns higher implicit reward to the rejected response, the gradient is large. This is essentially an automatic curriculum.

**Why this matters for debugging**: If your training loss drops to zero very quickly, it means the weighting term is approaching zero for most examples — the model has "solved" the training set. This usually means overfitting, not good alignment.

### Computing Log-Probabilities in Practice

In implementation, $\log \pi_\theta(y|x)$ is computed as the sum of per-token log-probabilities:

$$\log \pi_\theta(y|x) = \sum_{t=1}^{T} \log \pi_\theta(y_t | x, y_{<t})$$

This is just the standard autoregressive language model log-likelihood. The DPO trainer computes this for both the chosen and rejected responses, for both the policy and reference models — giving four forward passes per training example.

```python
# Pseudocode for what DPO actually computes
def dpo_loss(model, ref_model, prompt, chosen, rejected, beta):
    # Forward pass: compute log-probs for all four combinations
    policy_chosen_logps = get_log_probs(model, prompt, chosen)      # log π_θ(y_w|x)
    policy_rejected_logps = get_log_probs(model, prompt, rejected)  # log π_θ(y_l|x)
    ref_chosen_logps = get_log_probs(ref_model, prompt, chosen)     # log π_ref(y_w|x)
    ref_rejected_logps = get_log_probs(ref_model, prompt, rejected) # log π_ref(y_l|x)

    # Compute implicit rewards (log-ratios)
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    # DPO loss = -log(sigmoid(chosen_reward - rejected_reward))
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    return loss
```

## Training Frameworks & Libraries: Choosing Your Weapon

One of the most practical decisions you'll make is **which framework to use** for DPO training. Each has distinct strengths and trade-offs. Here's an honest comparison based on real-world usage.

### Framework Comparison Overview

| Framework | Best For | DPO Support | Speed | Multi-GPU | VRAM Efficiency | Setup Complexity | Community |
|-----------|---------|-------------|-------|-----------|-----------------|------------------|-----------|
| **TRL** | Standard HF workflow | Native | Baseline | Yes (DeepSpeed/FSDP) | Good | Low | Largest |
| **Unsloth** | Single-GPU, budget hardware | Via TRL | 2x faster | No | Best (70% less) | Low | Large |
| **Axolotl** | Multi-GPU, complex configs | Native | Good | Yes | Good | Medium | Medium |
| **LLaMA-Factory** | No-code/low-code | Native | Good | Yes | Good | Lowest | Large (China) |
| **OpenRLHF** | Full RLHF pipeline | Yes | Good | Yes (Ray) | Good | High | Small |
| **Alignment Handbook** | HF recipes/reference | Via TRL | Baseline | Yes | Good | Medium | Small |

### Framework 1: TRL (Hugging Face) — The Standard

**TRL** is the reference implementation. If you're reading a DPO paper and want to reproduce results, TRL is what the authors likely used. It's built on top of the Hugging Face Transformers ecosystem and integrates natively with PEFT, Accelerate, DeepSpeed, and FSDP.

**When to use TRL:**
- You want maximum control over every parameter
- You're doing research or need to implement custom loss variants
- You need multi-GPU training with DeepSpeed or FSDP
- You want the most up-to-date implementation of new methods (KTO, ORPO, SimPO, etc.)

**Installation:**

```bash
pip install trl transformers datasets peft accelerate bitsandbytes
# Optional: for fast generation during evaluation
pip install trl[vllm]
```

**Full DPO Training with TRL:**

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig

# ============================================================
# Configuration
# ============================================================
MODEL_PATH = "./sft-llama-8b-final"  # Your SFT model
OUTPUT_DIR = "./dpo-llama-8b"
BETA = 0.1  # KL penalty coefficient — the most important hyperparameter

# ============================================================
# Load model and tokenizer
# ============================================================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# LoRA config (recommended for 8B+ models)
# ============================================================
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

# ============================================================
# Load preference dataset
# ============================================================
dataset = load_dataset(
    "HuggingFaceH4/ultrafeedback_binarized",
    split="train_prefs"
)

# Format into the expected structure
def format_dpo_example(example):
    """Convert dataset format to what DPOTrainer expects."""
    prompt = example["prompt"]
    return {
        "prompt": tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        ),
        "chosen": example["chosen"][1]["content"],
        "rejected": example["rejected"][1]["content"],
    }

dataset = dataset.map(format_dpo_example)

# ============================================================
# DPO Training Configuration
# ============================================================
training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    beta=BETA,

    # Batch size & accumulation
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    # Effective batch size = 2 * 8 * num_gpus

    # Learning rate — DPO is sensitive, keep it low
    learning_rate=5e-7,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,

    # Training duration
    num_train_epochs=1,  # 1-3 epochs is typical. More risks overfitting.
    max_steps=-1,

    # Sequence lengths
    max_length=1024,
    max_prompt_length=512,

    # Precision
    bf16=True,

    # Logging & saving
    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    eval_strategy="steps",
    eval_steps=200,

    # Memory optimization
    gradient_checkpointing=True,

    # Important: remove unused columns
    remove_unused_columns=False,

    # Seed for reproducibility
    seed=42,

    # Experiment tracking
    report_to="wandb",  # or "tensorboard"
)

# ============================================================
# Initialize trainer and train
# ============================================================
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)

trainer.train()
trainer.save_model(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
```

**Launching with multi-GPU (DeepSpeed ZeRO-3):**

```bash
accelerate launch --config_file configs/deepspeed_z3.yaml train_dpo.py
```

```yaml
# configs/deepspeed_z3.yaml
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
deepspeed_config:
  deepspeed_multinode_launcher: standard
  gradient_accumulation_steps: 8
  gradient_clipping: 1.0
  offload_optimizer_device: cpu
  offload_param_device: none
  zero3_init_flag: true
  zero_stage: 3
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4  # Number of GPUs
```

### Framework 2: Unsloth — 2x Faster, 70% Less Memory

**Unsloth** is the best choice when you're working with a single GPU or limited VRAM. It achieves its speed through hand-optimized Triton kernels and manual backpropagation derivations — no approximations, zero accuracy loss.

**When to use Unsloth:**
- You have a single GPU (even a free Colab T4)
- You want the fastest possible training on one card
- You're prototyping and want fast iteration cycles
- You're training 7B-8B models on consumer GPUs (RTX 3090/4090)

**Limitation**: Unsloth does not support multi-GPU training. For multi-node setups, use TRL with DeepSpeed or Axolotl.

**Installation:**

```bash
pip install unsloth
# Or for specific CUDA version:
pip install unsloth[cu121]
```

**Full DPO Training with Unsloth:**

```python
from unsloth import FastLanguageModel
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
import torch

# ============================================================
# 1. Load model with Unsloth's optimized loader
#    This automatically applies fused kernels and memory optimizations
# ============================================================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length=2048,
    dtype=None,        # Auto-detect best dtype
    load_in_4bit=True, # QLoRA: 4-bit quantized base model
)

# ============================================================
# 2. Apply LoRA with Unsloth's optimized implementation
#    Key: use_gradient_checkpointing="unsloth" for 30% less VRAM
# ============================================================
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0,  # Unsloth is optimized for 0 dropout (actually faster)
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
    use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
    random_state=42,
)

# ============================================================
# 3. Load and prepare preference data
# ============================================================
dataset = load_dataset(
    "argilla/dpo-mix-7k",
    split="train"
)

def format_for_dpo(example):
    """Format data for the DPO trainer."""
    system_prompt = "You are a helpful, harmless, and honest assistant."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example["prompt"]},
    ]
    return {
        "prompt": tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        ),
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }

dataset = dataset.map(format_for_dpo)
dataset = dataset.train_test_split(test_size=0.02, seed=42)

# ============================================================
# 4. Configure DPO training
# ============================================================
dpo_config = DPOConfig(
    output_dir="./dpo-unsloth-output",
    beta=0.1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-7,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    num_train_epochs=1,
    max_length=1024,
    max_prompt_length=512,
    bf16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    eval_strategy="steps",
    eval_steps=200,
    optim="adamw_8bit",  # 8-bit Adam for memory savings
    gradient_checkpointing=True,
    remove_unused_columns=False,
    seed=42,
)

# ============================================================
# 5. Train — DPOTrainer from TRL works directly with Unsloth models
# ============================================================
trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
)

trainer.train()

# ============================================================
# 6. Save in multiple formats
# ============================================================

# Save LoRA adapters (smallest, fastest)
model.save_pretrained("./dpo-model-lora")
tokenizer.save_pretrained("./dpo-model-lora")

# Save merged model in 16-bit (for vLLM / TGI deployment)
model.save_pretrained_merged(
    "./dpo-model-merged-16bit",
    tokenizer,
    save_method="merged_16bit",
)

# Save as GGUF for llama.cpp / Ollama deployment
model.save_pretrained_gguf(
    "./dpo-model-gguf",
    tokenizer,
    quantization_method="q4_k_m",  # Good quality-size balance
)

# Upload to Hugging Face Hub (optional)
# model.push_to_hub_merged("your-username/dpo-llama-8b", tokenizer, save_method="merged_16bit")
# model.push_to_hub_gguf("your-username/dpo-llama-8b-gguf", tokenizer, quantization_method="q4_k_m")
```

**Unsloth VRAM comparison for DPO:**

| Model | Standard TRL (QLoRA) | Unsloth (QLoRA) | Savings |
|-------|---------------------|-----------------|---------|
| Llama 3.1 8B | ~24 GB | ~8 GB | 67% |
| Mistral 7B | ~18 GB | ~6 GB | 67% |
| Qwen 2.5 7B | ~18 GB | ~6 GB | 67% |
| Gemma 2 9B | ~26 GB | ~9 GB | 65% |

This means DPO training on a **free Google Colab T4 (15 GB)** is possible with Unsloth for 7B-8B models. Without Unsloth, you'd need at least an A10G or RTX 4090.

### Framework 3: Axolotl — The Config-Driven Powerhouse

**Axolotl** is a config-file-driven training framework that wraps TRL, PEFT, and DeepSpeed into a YAML-configurable pipeline. It's the go-to choice for teams running many training experiments because you can define everything in a single YAML file — no Python scripting needed.

**When to use Axolotl:**
- You're running many experiments and want reproducible configs
- You need multi-GPU training with minimal code
- You want built-in support for many dataset formats
- Your team has varying levels of Python experience

**Installation:**

```bash
pip install axolotl
# Or from source for latest features:
pip install git+https://github.com/axolotl-ai-cloud/axolotl.git
```

**Full DPO Training with Axolotl (YAML config):**

```yaml
# dpo_config.yaml — This is the entire training configuration

# Model
base_model: meta-llama/Llama-3.1-8B-Instruct
model_type: LlamaForCausalLM

# Tokenizer
tokenizer_type: AutoTokenizer
special_tokens:
  pad_token: "<|end_of_text|>"

# LoRA configuration
adapter: lora
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
lora_target_linear: true

# DPO-specific configuration
rl: dpo
rl_beta: 0.1    # DPO beta parameter

# Dataset — Axolotl handles many formats automatically
datasets:
  - path: argilla/dpo-mix-7k
    type: chatml.intel  # Built-in format handler for DPO data
    split: train
  # You can add multiple datasets:
  # - path: Intel/orca_dpo_pairs
  #   type: chatml.intel
  #   split: train

# Dataset settings
dataset_prepared_path: ./prepared_data
val_set_size: 0.02

# Sequence lengths
sequence_len: 1024
max_prompt_len: 512

# Training hyperparameters
micro_batch_size: 2
gradient_accumulation_steps: 8
num_epochs: 1
learning_rate: 5e-7
lr_scheduler: cosine
warmup_ratio: 0.1

# Optimizer
optimizer: adamw_bnb_8bit  # 8-bit Adam

# Precision
bf16: auto
tf32: true

# Memory optimization
gradient_checkpointing: true
flash_attention: true
sample_packing: false  # Disable for DPO (packing doesn't work well with paired data)

# Quantization (QLoRA)
load_in_4bit: true
bnb_4bit_quant_type: nf4
bnb_4bit_compute_dtype: bfloat16
bnb_4bit_use_double_quant: true

# Saving & logging
output_dir: ./dpo-axolotl-output
save_strategy: steps
save_steps: 200
logging_steps: 10
eval_steps: 200

# Experiment tracking
wandb_project: dpo-training
wandb_run_id:  # Auto-generated if empty

# DeepSpeed (for multi-GPU)
# deepspeed: configs/deepspeed_z3.json

# Seed
seed: 42
```

**Launch training:**

```bash
# Single GPU
accelerate launch -m axolotl.cli.train dpo_config.yaml

# Multi-GPU with DeepSpeed
accelerate launch --config_file configs/accelerate_ds_z3.yaml -m axolotl.cli.train dpo_config.yaml

# Preview data processing (dry run — recommended before training)
python -m axolotl.cli.preprocess dpo_config.yaml
```

**Merge LoRA and export after training:**

```bash
# Merge LoRA adapters into base model
python -m axolotl.cli.merge_lora dpo_config.yaml --lora_model_dir="./dpo-axolotl-output"
```

### Framework 4: LLaMA-Factory — The No-Code Option

**LLaMA-Factory** is the easiest entry point. It provides both a **web UI** and a **CLI** for training, with support for 100+ model architectures. It's especially popular in the Chinese AI community and is great for quick experiments.

**When to use LLaMA-Factory:**
- You want a web UI for configuring and launching training
- You're new to DPO and want the lowest possible barrier to entry
- You need quick prototyping without writing code
- You want built-in dataset management

**Installation:**

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

**Launch the Web UI:**

```bash
llamafactory-cli webui
```

This opens a browser UI where you can:
1. Select model (Llama, Mistral, Qwen, etc.)
2. Choose training method (DPO)
3. Upload or select preference dataset
4. Configure hyperparameters with sliders
5. Launch training with one click
6. Monitor training curves in real-time

**CLI training (YAML config):**

```yaml
# llama_factory_dpo.yaml

### Model
model_name_or_path: meta-llama/Llama-3.1-8B-Instruct
trust_remote_code: true

### Method
stage: dpo           # Key: this sets DPO training mode
do_train: true
finetuning_type: lora
pref_beta: 0.1       # DPO beta
pref_loss: sigmoid   # "sigmoid" = standard DPO, "ipo" for IPO variant

### LoRA
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target: all     # Target all linear layers

### Dataset
dataset: dpo_mix_en  # Built-in dataset name, or path to custom data
dataset_dir: data
template: llama3
cutoff_len: 1024
preprocessing_num_workers: 16

### Training
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 5.0e-7
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
gradient_checkpointing: true

### Output
output_dir: ./saves/llama3-8b-dpo
logging_steps: 10
save_steps: 200
eval_steps: 200
plot_loss: true      # Generate loss plots automatically

### Quantization
quantization_bit: 4
quantization_method: bitsandbytes
```

```bash
llamafactory-cli train llama_factory_dpo.yaml
```

**Custom dataset format for LLaMA-Factory:**

Create a JSON file at `data/my_dpo_data.json`:

```json
[
  {
    "conversations": [
      {"from": "human", "value": "What are the benefits of exercise?"}
    ],
    "chosen": {
      "from": "gpt",
      "value": "Regular exercise offers numerous evidence-based benefits..."
    },
    "rejected": {
      "from": "gpt",
      "value": "Exercise is good for you. You should exercise more."
    }
  }
]
```

Register it in `data/dataset_info.json`:

```json
{
  "my_dpo_data": {
    "file_name": "my_dpo_data.json",
    "ranking": true,
    "formatting": "sharegpt"
  }
}
```

### Framework 5: OpenRLHF — For the Full RL Pipeline

**OpenRLHF** is designed for teams that need the complete RLHF pipeline (including PPO) at scale using Ray for distributed training. It also supports DPO.

**When to use OpenRLHF:**
- You need PPO + DPO in the same codebase
- You're training at scale (dozens of GPUs)
- You need Ray-based distributed training
- You're doing advanced RL research (RAFT, rejection sampling, etc.)

```bash
pip install openrlhf
```

```bash
# DPO training with OpenRLHF
deepspeed --module openrlhf.cli.train_dpo \
   --save_path ./dpo-checkpoint \
   --save_steps 200 \
   --logging_steps 10 \
   --eval_steps 200 \
   --micro_train_batch_size 2 \
   --pretrain meta-llama/Llama-3.1-8B-Instruct \
   --bf16 \
   --max_epochs 1 \
   --max_len 1024 \
   --zero_stage 3 \
   --beta 0.1 \
   --learning_rate 5e-7 \
   --dataset argilla/dpo-mix-7k \
   --flash_attn \
   --gradient_checkpointing \
   --lora_rank 16 \
   --lora_alpha 32
```

### Framework Decision Tree

```
Start here: What's your situation?
│
├── Single GPU, limited VRAM (<24GB)?
│   └── Use Unsloth
│       - 2x faster, 70% less VRAM
│       - Works on free Colab T4
│       - Limitation: no multi-GPU
│
├── Multi-GPU, production training?
│   ├── Want config files, no code?
│   │   └── Use Axolotl
│   │       - YAML-driven, reproducible
│   │       - DeepSpeed/FSDP built-in
│   │
│   ├── Want maximum flexibility (code)?
│   │   └── Use TRL directly
│   │       - Most features, most control
│   │       - Best for custom loss functions
│   │
│   └── Need PPO + DPO at scale?
│       └── Use OpenRLHF
│           - Ray-based distributed training
│           - Full RL pipeline
│
├── Want a web UI / no code?
│   └── Use LLaMA-Factory
│       - Browser-based training interface
│       - Easiest setup
│
└── Reproducing a paper?
    └── Use TRL
        - Reference implementation
        - What most papers use
```

## The Full Training Pipeline

![DPO training pipeline: SFT model cloned to frozen π_ref and initialized as π_θ, trained on preference pairs with β=0.1-0.5, evaluated on reward accuracy and pairwise judge; adjust β/LR/pairs on failure](/imgs/blogs/dpo-03-pipeline.png)

Here's the end-to-end pipeline:

```
1. Start with a base model (or already fine-tuned model)
2. Run SFT on high-quality instruction data
3. Collect or create preference data: (prompt, chosen, rejected)
4. Train with DPO loss
5. Evaluate and iterate
```

### Step 1: Choose Your Base Model

For DPO to work well, you need a model that's already reasonably capable. DPO is for *alignment*, not *capability*. Common starting points:

| Base Model | Parameters | Good For | HF Model ID |
|-----------|-----------|---------|-------------|
| Llama 3.1 | 8B / 70B | General purpose, strong baseline | `meta-llama/Llama-3.1-8B-Instruct` |
| Mistral v0.3 | 7B | Efficient, good code/reasoning | `mistralai/Mistral-7B-Instruct-v0.3` |
| Qwen 2.5 | 7B / 72B | Strong multilingual, math | `Qwen/Qwen2.5-7B-Instruct` |
| Gemma 2 | 9B / 27B | Good instruction following | `google/gemma-2-9b-it` |
| Phi-3.5 | 3.8B | Best quality-per-parameter | `microsoft/Phi-3.5-mini-instruct` |

**Important choice**: Start from an **Instruct** model (already SFT'd) if you want to skip the SFT step and go directly to DPO. Start from a **base** model if you want full control over the SFT stage.

### Step 2: Supervised Fine-Tuning (SFT)

DPO works best when applied on top of an SFT model. The SFT stage gives the model the basic format and style you want, and DPO refines the preferences.

**With TRL:**

```python
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch

model_name = "meta-llama/Llama-3.1-8B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

def format_chat(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False
        )
    }

dataset = dataset.map(format_chat)

sft_config = SFTConfig(
    output_dir="./sft-llama-8b",
    max_seq_length=2048,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=1,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    gradient_checkpointing=True,
    packing=True,  # Pack short sequences together for efficiency
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
trainer.save_model("./sft-llama-8b-final")
```

**With Unsloth (2x faster):**

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
)

# ... same SFTTrainer setup as above
```

### Step 3: Prepare Preference Data

This is the most important step. The quality of your preference data **directly determines** the quality of your DPO model. Garbage in, garbage out — more so than any other stage.

#### Data Format

DPO requires triplets of `(prompt, chosen, rejected)`:

```json
{
  "prompt": "Explain quantum entanglement to a 10-year-old.",
  "chosen": "Imagine you have two magic coins. When you flip one and it lands on heads, the other one — no matter how far away — instantly lands on tails. Nobody sent a message between them, they just always match up in opposite ways. Scientists call this 'entanglement'. The spooky part is that before you look at either coin, both are spinning — they haven't decided yet. The moment you look at one, both decide at the same time. Einstein thought this was so weird he called it 'spooky action at a distance'.",
  "rejected": "Quantum entanglement is a phenomenon in quantum mechanics where two or more particles become correlated in such a way that the quantum state of each particle cannot be described independently. When a measurement is performed on one particle, the state of the other particle is instantaneously affected, regardless of the distance separating them. This is described mathematically by the Bell states and violates Bell's inequality, demonstrating non-local correlations."
}
```

The `chosen` response is better because it actually follows the instruction (explain to a 10-year-old). The `rejected` is technically correct but ignores the audience constraint.

#### Sources of Preference Data

**Option 1: Human annotation** (best quality, most expensive)

- Have annotators rate pairs of responses
- Use platforms like Scale AI, Surge AI, Labelbox, or internal teams
- Cost: $1-5 per comparison
- Recommendation: Start with 5K-10K examples for domain-specific tasks
- **Key tip**: Write very detailed annotation guidelines. Vague instructions like "pick the better response" lead to inconsistent labels. Specify what "better" means: more accurate? More helpful? Safer? More concise?

**Option 2: AI-generated preferences / RLAIF** (good quality, much cheaper)

- Generate multiple responses from your SFT model
- Use a strong model (GPT-4o, Claude 3.5 Sonnet) to judge which is better
- This is called "AI feedback" or "RLAIF" (RL from AI Feedback)
- Cost: $0.01-0.05 per comparison

```python
import openai
from datasets import Dataset

def generate_preference_pair(prompt, model_responses):
    """Use a strong model to judge which response is better."""
    judge_prompt = f"""You are evaluating two AI responses to the same prompt.
Rate which response is better based on:
1. Accuracy and correctness
2. Helpfulness and completeness
3. Clarity and readability
4. Safety (does not include harmful content)

Prompt: {prompt}

Response A:
{model_responses[0]}

Response B:
{model_responses[1]}

Which response is better? Think step by step, then reply with ONLY "A" or "B" on the last line."""

    client = openai.OpenAI()
    judgment = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": judge_prompt}],
        max_tokens=256,
        temperature=0,
    )

    response_text = judgment.choices[0].message.content.strip()
    # Extract the last line (the verdict)
    winner = response_text.strip().split("\n")[-1].strip()

    if winner == "A":
        return model_responses[0], model_responses[1]
    else:
        return model_responses[1], model_responses[0]
```

**Option 3: Use existing preference datasets**

| Dataset | Size | Domain | Quality |
|---------|------|--------|---------|
| `HuggingFaceH4/ultrafeedback_binarized` | 64K | General instruction following | High |
| `Anthropic/hh-rlhf` | 170K | Helpfulness & harmlessness | High |
| `argilla/dpo-mix-7k` | 7K | Mixed quality signals | Medium |
| `Intel/orca_dpo_pairs` | 13K | Reasoning & knowledge | Medium |
| `openbmb/UltraInteract_pair` | 290K | Math & code | High |
| `nvidia/HelpSteer2` | 21K | Multi-aspect (helpfulness, correctness, coherence) | Very high |
| `allenai/tulu-3-pref-mixture-v3.1` | 326K | Full alignment mixture (Tulu 3) | Very high |

**Option 4: On-policy generation** (recommended for best results)

The most effective approach is generating responses from your own SFT model, then ranking them. This is critical because DPO's loss computes log-probability ratios — if the training responses are very different from what your model would generate, the gradients become noisy and less informative.

```python
from transformers import pipeline
from datasets import Dataset
import random

generator = pipeline(
    "text-generation",
    model="./sft-llama-8b-final",
    torch_dtype="bfloat16",
    device_map="auto",
)

def generate_on_policy_pairs(prompts, n_samples=4, temperature=0.8):
    """Generate multiple responses per prompt, then rank them."""
    preference_data = []

    for prompt in prompts:
        # Generate n_samples responses with diversity
        responses = []
        for _ in range(n_samples):
            output = generator(
                prompt,
                max_new_tokens=512,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
            )
            responses.append(output[0]["generated_text"][len(prompt):])

        # Score each response using a judge model
        scored = score_responses_with_judge(prompt, responses)
        scored.sort(key=lambda x: x[1], reverse=True)

        # Create pair from best and worst
        # Skip if the quality gap is too small (ambiguous preference)
        if scored[0][1] - scored[-1][1] > 0.2:  # Minimum score gap threshold
            preference_data.append({
                "prompt": prompt,
                "chosen": scored[0][0],
                "rejected": scored[-1][0],
            })

    return Dataset.from_list(preference_data)
```

#### Data Quality Checklist

Before training, verify your data:

```python
import numpy as np
from collections import Counter

def validate_preference_data(dataset, tokenizer):
    """Comprehensive sanity checks for DPO data quality."""
    issues = []
    chosen_lengths = []
    rejected_lengths = []

    for i, example in enumerate(dataset):
        prompt = example["prompt"]
        chosen = example["chosen"]
        rejected = example["rejected"]

        # 1. Check for empty responses
        if not chosen.strip() or not rejected.strip():
            issues.append(f"Row {i}: Empty response")

        # 2. Check for identical responses
        if chosen.strip() == rejected.strip():
            issues.append(f"Row {i}: Chosen and rejected are identical")

        # 3. Check for very short responses (likely truncated or degenerate)
        if len(chosen.split()) < 5:
            issues.append(f"Row {i}: Chosen response suspiciously short ({len(chosen.split())} words)")

        # 4. Check for prompt leaking into response
        if chosen.startswith(prompt[:50]):
            issues.append(f"Row {i}: Possible prompt leak in chosen")

        # 5. Check response length ratio (extreme differences are suspicious)
        len_ratio = len(chosen) / max(len(rejected), 1)
        if len_ratio > 10 or len_ratio < 0.1:
            issues.append(f"Row {i}: Extreme length ratio ({len_ratio:.1f}x)")

        # 6. Check token count isn't too long for training
        chosen_tokens = len(tokenizer.encode(prompt + chosen))
        rejected_tokens = len(tokenizer.encode(prompt + rejected))
        if max(chosen_tokens, rejected_tokens) > 2048:
            issues.append(f"Row {i}: Sequence too long ({max(chosen_tokens, rejected_tokens)} tokens)")

        chosen_lengths.append(len(chosen.split()))
        rejected_lengths.append(len(rejected.split()))

    # 7. Check for systematic length bias
    avg_chosen_len = np.mean(chosen_lengths)
    avg_rejected_len = np.mean(rejected_lengths)
    length_ratio = avg_chosen_len / max(avg_rejected_len, 1)

    print(f"\n=== Data Quality Report ===")
    print(f"Total examples: {len(dataset)}")
    print(f"Issues found: {len(issues)}")
    print(f"Avg chosen length: {avg_chosen_len:.0f} words")
    print(f"Avg rejected length: {avg_rejected_len:.0f} words")
    print(f"Length ratio (chosen/rejected): {length_ratio:.2f}x")

    if length_ratio > 1.5:
        print(f"⚠ WARNING: Chosen responses are {length_ratio:.1f}x longer than rejected.")
        print(f"  This will train a verbose model. Consider length-balancing your data.")
    elif length_ratio < 0.67:
        print(f"⚠ WARNING: Chosen responses are {1/length_ratio:.1f}x shorter than rejected.")
        print(f"  This will train a terse model. Make sure this is intentional.")

    if issues:
        print(f"\nFirst 20 issues:")
        for issue in issues[:20]:
            print(f"  - {issue}")

    return len(issues) == 0
```

### Step 4: DPO Training

See the full training scripts in the **Training Frameworks** section above. Here's a quick summary of what each framework needs:

```
TRL:            Python script with DPOTrainer + DPOConfig
Unsloth:        FastLanguageModel.from_pretrained() → DPOTrainer (same as TRL)
Axolotl:        YAML config with rl: dpo
LLaMA-Factory:  YAML config with stage: dpo (or use Web UI)
OpenRLHF:       CLI command with --module openrlhf.cli.train_dpo
```

### Step 5: Merge LoRA Weights and Export

After training with LoRA, you need to merge the adapters for deployment:

**With standard PEFT:**

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base_model = AutoModelForCausalLM.from_pretrained(
    "./sft-llama-8b-final",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, "./dpo-llama-8b/final")
model = model.merge_and_unload()

model.save_pretrained("./dpo-llama-8b-merged")
tokenizer = AutoTokenizer.from_pretrained("./dpo-llama-8b/final")
tokenizer.save_pretrained("./dpo-llama-8b-merged")
```

**With Unsloth (multiple export formats):**

```python
# 16-bit merged (for vLLM, TGI, TensorRT-LLM)
model.save_pretrained_merged("./merged-16bit", tokenizer, save_method="merged_16bit")

# 4-bit merged (for bitsandbytes inference)
model.save_pretrained_merged("./merged-4bit", tokenizer, save_method="merged_4bit")

# GGUF for llama.cpp / Ollama / LM Studio
model.save_pretrained_gguf("./gguf", tokenizer, quantization_method="q4_k_m")

# Available GGUF quantization methods:
# q4_k_m  — good balance of quality and size (recommended)
# q5_k_m  — slightly better quality, larger
# q8_0    — best quality GGUF, largest
# q2_k    — smallest, lowest quality
```

**With Axolotl:**

```bash
python -m axolotl.cli.merge_lora dpo_config.yaml --lora_model_dir="./dpo-axolotl-output"
```

## Hardware Planning: What Do You Actually Need?

One of the most common questions is "can I run this on my hardware?" Here's a detailed breakdown.

### VRAM Requirements by Model Size and Method

| Model Size | Full FT (FP16) | LoRA (FP16) | QLoRA (4-bit) | Unsloth QLoRA |
|-----------|---------------|-------------|---------------|---------------|
| 3B | 24 GB | 16 GB | 8 GB | 4 GB |
| 7-8B | 60 GB | 32 GB | 18 GB | 6-8 GB |
| 13B | 104 GB | 48 GB | 28 GB | 12 GB |
| 27B | 216 GB | 80 GB | 48 GB | 20 GB |
| 70B | 560 GB | 160 GB | 80 GB | 48 GB |

**Note**: DPO requires ~1.5x the VRAM of SFT because it holds both the policy and reference model. The reference model is frozen, so it can be loaded in lower precision. Some frameworks (Unsloth, TRL) automatically optimize this.

### Recommended Hardware Configurations

| Budget | Hardware | What You Can Train |
|--------|----------|-------------------|
| Free | Google Colab T4 (15 GB) | 7-8B with Unsloth QLoRA |
| $0 | Kaggle P100/T4 (16 GB) | 7-8B with Unsloth QLoRA |
| $200/mo | Lambda Cloud 1xA10G (24 GB) | 7-8B with QLoRA (any framework) |
| $500/mo | RunPod 1xA100 40GB | 13B with QLoRA, 7-8B with full LoRA |
| $1K/mo | RunPod 1xA100 80GB | 27B with QLoRA, 13B with full LoRA |
| $3K/mo | RunPod 4xA100 80GB | 70B with QLoRA + DeepSpeed ZeRO-3 |
| $10K/mo | 8xH100 80GB | 70B full fine-tuning |

### Training Time Estimates

For DPO on 50K preference pairs (1 epoch):

| Model | 1xA100 80GB | 4xA100 80GB | 1xA100 + Unsloth |
|-------|-------------|-------------|-------------------|
| 7-8B (QLoRA) | ~4 hours | ~1.5 hours | ~2 hours |
| 13B (QLoRA) | ~8 hours | ~3 hours | ~4 hours |
| 70B (QLoRA) | ~48 hours | ~14 hours | N/A (multi-GPU) |

## Hyperparameter Deep Dive

DPO has fewer hyperparameters than PPO, but the ones it has matter a lot.

### Beta ($\beta$) — The Most Important Hyperparameter

$\beta$ controls the strength of the KL constraint. It determines how far the trained model can deviate from the reference model.

| $\beta$ Value | Behavior | When to Use |
|-------|----------|-------------|
| 0.01 - 0.05 | Aggressive optimization, high divergence from reference | When preference signal is very strong and clean |
| **0.1** | **Standard starting point** | **Default — start here** |
| 0.2 - 0.5 | Conservative, stays close to reference | When data is noisy or you want subtle adjustments |
| 0.5 - 1.0 | Very conservative, minimal change | When you only want to fix specific failure modes |

**Deep insight on $\beta$**: Think of $\beta$ as a "temperature" for how much you trust your preference data. Low $\beta$ = "I trust my data completely, move aggressively." High $\beta$ = "My data might be noisy, be careful." If you have 5K expert-annotated pairs for a specific domain, use lower $\beta$. If you scraped 100K noisy AI-judged pairs, use higher $\beta$.

**How to tune it:** Track the *implicit reward margin* during training:

$$\text{reward\_margin} = \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}$$

- If the margin plateaus near 0: $\beta$ is too high, the model can't learn
- If the margin shoots up rapidly (>5.0 in early training): $\beta$ is too low, risk of overfitting
- A healthy training run shows a **steady increase** from ~0 to 1.0-3.0 over the full run

### Learning Rate

DPO is more sensitive to learning rate than standard fine-tuning. The loss landscape is sharper because you're optimizing ratios of probabilities, not raw probabilities.

```
Rule of thumb:
- Full fine-tuning: 1e-7 to 5e-7
- LoRA fine-tuning: 5e-7 to 5e-6
- QLoRA fine-tuning: 1e-6 to 1e-5

If you see these problems:
- Loss drops to 0 quickly → learning rate too high, model is memorizing
- Loss barely moves → learning rate too low
- Loss is unstable (oscillating) → learning rate too high or batch size too small
- Loss decreases then increases → overfitting, reduce LR or add more data
```

**Pro tip**: Start with the lower end of the range and increase if the loss barely moves. It's much easier to recover from "too low" (just increase) than "too high" (may need to restart from checkpoint).

### Batch Size and Gradient Accumulation

DPO benefits from larger effective batch sizes more than SFT because each example is a *comparison* — you need enough comparisons per update for stable gradients.

```python
# Recommended effective batch sizes:
# Small dataset (<10K):  effective_batch = 16-32
# Medium dataset (10K-100K): effective_batch = 32-64
# Large dataset (>100K): effective_batch = 64-128

# effective_batch = per_device_batch * gradient_accumulation * num_gpus
# Example: 2 * 8 * 4 = 64
```

### Sequence Length and Truncation

DPO computes log-probabilities over the full response. Truncation can distort the loss:

```python
# BAD: Truncating in the middle of a response
# This changes the effective preference signal because the model
# only sees partial responses, making the comparison unfair

# GOOD: Filter out examples that are too long BEFORE training
max_total_length = 1024

def filter_by_length(example, tokenizer, max_length=1024):
    prompt_len = len(tokenizer.encode(example["prompt"]))
    chosen_len = len(tokenizer.encode(example["chosen"]))
    rejected_len = len(tokenizer.encode(example["rejected"]))
    return (prompt_len + max(chosen_len, rejected_len)) <= max_length

dataset = dataset.filter(lambda x: filter_by_length(x, tokenizer))
print(f"Kept {len(dataset)} / {original_len} examples after length filtering")
```

### Number of Epochs

```
Golden rule: Start with 1 epoch.

DPO overfits MUCH faster than SFT because:
1. Each example is a binary comparison — easy to memorize
2. The loss has a natural minimum around -log(1) = 0
3. The model only needs to learn the relative ordering, not generate text

If 1 epoch isn't enough:
- First check if your data is the bottleneck (add more data)
- Then try 2 epochs with early stopping
- Never go beyond 3 epochs without very strong justification
```

## Monitoring Training: What to Watch

### Key Metrics

```python
# These are automatically logged by DPOTrainer in TRL
# Monitor them in Weights & Biases or TensorBoard

# 1. train/loss — should decrease steadily
#    Healthy: starts ~0.69 (= -log(0.5)), decreases to 0.3-0.5
#    Why 0.69? At initialization, the model assigns equal implicit reward
#    to chosen and rejected, so the sigmoid is 0.5, and -log(0.5) ≈ 0.693

# 2. train/rewards/chosen — implicit reward for chosen responses
#    Should increase over training

# 3. train/rewards/rejected — implicit reward for rejected responses
#    Should decrease (or increase less than chosen)

# 4. train/rewards/margins — difference between chosen and rejected rewards
#    Should increase steadily (this is the most informative metric)

# 5. train/rewards/accuracies — % of examples where chosen > rejected
#    Should increase from ~50% to 70-85%
#    Healthy range: 65-85%
#    If it hits 95%+: you're likely overfitting

# 6. eval/loss — evaluation loss on held-out data
#    Should track train/loss but stay slightly higher
#    If eval_loss increases while train_loss decreases: overfitting
```

### Visualizing a Healthy Training Run

```
                    Healthy DPO Training Curves

Loss:                       Reward Margin:
 0.69 ─┐                   3.0 ─┐
       │╲                       │         ╱──
       │ ╲                      │       ╱
       │  ╲                     │     ╱
       │   ╲                    │   ╱
       │    ╲───────            │ ╱
 0.35 ─┘         ──── eval     │╱
                       train   0.0 ─┘
       └──────────────         └──────────────
        Steps                   Steps

Accuracy:                   Chosen vs Rejected Rewards:
 1.0 ──┐                   2.0 ─┐
       │       ────────         │    chosen ──────
       │     ╱                  │  ╱
       │   ╱                    │╱
       │ ╱                   0.0│──────────────────
       │╱                       │╲
 0.5 ──┘                        │  rejected ──────
       └──────────────    -1.0 ─┘
        Steps                   └──────────────
                                 Steps
```

### Red Flags During Training

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Loss drops to 0 in < 100 steps | Learning rate too high | Reduce LR by 5-10x |
| Reward accuracy > 95% | Overfitting | More data, higher $\beta$, fewer epochs |
| Reward accuracy stuck at 50% | Model not learning | Lower $\beta$, increase LR, check data quality |
| Chosen reward decreasing | Both responses getting penalized | Data quality issue — chosen responses may be bad |
| Loss is NaN | Numerical instability | Use BF16 (not FP16), reduce LR, check for empty examples |
| Eval loss increasing, train loss decreasing | Overfitting | Stop training, use best checkpoint |
| Reward margin > 10 | Model has diverged too far | Increase $\beta$, reduce LR |
| Both rewards increasing equally | Model learning to be verbose, not to discriminate | Check for length bias in data |

### Evaluation During Training

```python
# Split your data
dataset = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# The DPOTrainer will automatically evaluate on eval_dataset
# when eval_strategy="steps" is set
```

## DPO Variants: When Standard DPO Isn't Enough

The original DPO paper spawned many variants. Here are the ones worth knowing, with clear guidance on when to use each.

### IPO (Identity Preference Optimization)

**Problem it solves:** DPO can overfit when the preference data has label noise (some "chosen" responses are actually worse). The DPO loss uses a log-sigmoid which can push probabilities to extreme values. IPO uses a squared loss which is bounded and more robust.

$$\mathcal{L}_{\text{IPO}} = \left( \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} - \frac{1}{2\beta} \right)^2$$

**When to use**: Your preference data is AI-generated (inherently noisy) or from crowdsourced annotators (inconsistent quality).

```python
# In TRL, just change the loss_type
training_args = DPOConfig(
    loss_type="ipo",
    beta=0.1,
    # ... rest of config
)
```

### cDPO (Conservative DPO)

**Problem it solves:** Real-world data has a fraction of mislabeled preferences. cDPO accounts for this by introducing a label smoothing parameter that assumes some fraction of labels are flipped.

```python
training_args = DPOConfig(
    label_smoothing=0.1,  # Assumes 10% of labels might be flipped
    beta=0.1,
    # ... rest of config
)
```

### KTO (Kahneman-Tversky Optimization)

**Problem it solves:** You don't have paired preferences. You just have individual responses labeled as "good" or "bad".

This is a game-changer in practice because it's much easier to collect thumbs-up/thumbs-down data than to compare two responses side by side. Think about your product: users naturally give binary feedback, not pairwise comparisons.

```python
from trl import KTOTrainer, KTOConfig

# KTO data format — no pairs needed!
# Each example has: prompt, completion, label (True/False)
kto_dataset = Dataset.from_list([
    {"prompt": "What is 2+2?", "completion": "4", "label": True},
    {"prompt": "What is 2+2?", "completion": "22", "label": False},
    {"prompt": "Write a haiku about rain", "completion": "Gentle drops fall down\nPuddles form on quiet streets\nEarth drinks deeply here", "label": True},
    {"prompt": "Write a haiku about rain", "completion": "Rain is water that falls from clouds. It's wet.", "label": False},
])

kto_config = KTOConfig(
    output_dir="./kto-model",
    beta=0.1,
    desirable_weight=1.0,    # Weight for positive examples
    undesirable_weight=1.0,  # Weight for negative examples
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-7,
    num_train_epochs=1,
    bf16=True,
)

trainer = KTOTrainer(
    model=model,
    args=kto_config,
    train_dataset=kto_dataset,
    processing_class=tokenizer,
)
trainer.train()
```

### ORPO (Odds Ratio Preference Optimization)

**Problem it solves:** ORPO eliminates the need for both a reference model AND a separate SFT stage. It combines SFT and preference optimization into a single training stage by adding an odds-ratio preference loss to the standard language modeling loss.

$$\mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{SFT}}(y_w) + \lambda \cdot \mathcal{L}_{\text{OR}}$$

**When to use**: You want to skip the SFT stage entirely and go from base model to aligned model in one step.

```python
from trl import ORPOTrainer, ORPOConfig

orpo_config = ORPOConfig(
    output_dir="./orpo-model",
    beta=0.1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,  # Can use higher LR than DPO since SFT is included
    num_train_epochs=1,
    bf16=True,
    gradient_checkpointing=True,
)

trainer = ORPOTrainer(
    model=model,  # Can be a BASE model, not SFT
    args=orpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)
trainer.train()
```

### SimPO (Simple Preference Optimization)

**Problem it solves:** DPO uses the SFT model as reference, which means you need to keep it in memory (doubling VRAM). SimPO removes the reference model and uses length-normalized sequence log-probabilities as the implicit reward.

$$r_{\text{SimPO}}(x, y) = \frac{\beta}{|y|} \log \pi_\theta(y|x) - \gamma$$

Where $\gamma$ is a target reward margin.

**When to use**: You're memory-constrained and can't afford to hold both policy and reference models.

### Online DPO

**Problem it solves:** Standard DPO is offline — it uses a fixed dataset. Online DPO generates new responses during training, keeping the data on-policy.

```python
from trl import OnlineDPOTrainer, OnlineDPOConfig

online_config = OnlineDPOConfig(
    output_dir="./online-dpo",
    beta=0.1,
    learning_rate=5e-7,
    num_train_epochs=1,
    # Online DPO generates responses during training
    # and scores them with a reward model or judge
)

trainer = OnlineDPOTrainer(
    model=model,
    args=online_config,
    reward_model=reward_model,  # A trained reward model or judge
    train_dataset=prompts_only_dataset,
    processing_class=tokenizer,
)
trainer.train()
```

### Quick Comparison Table

| Method | Ref Model | Paired Data | SFT Stage | Memory | Best Use Case |
|--------|-----------|-------------|-----------|--------|---------------|
| DPO | Yes | Yes | Yes | High | Standard alignment |
| IPO | Yes | Yes | Yes | High | Noisy preference data |
| KTO | Yes | No (labels) | Yes | High | Binary feedback data |
| ORPO | No | Yes | No (combined) | Lower | Skip SFT, save time |
| SimPO | No | Yes | Yes | Lower | Memory-constrained |
| Online DPO | Yes | Generated | Yes | Higher | Best quality, iterative |

## Practical Tips from Production

### 1. Data Quality > Data Quantity

I've seen 5K high-quality preference pairs outperform 100K noisy ones. Focus on:

- **Clear preference signal**: The chosen response should be *obviously* better to a domain expert
- **Diverse prompts**: Don't have 1000 variations of "write a poem"
- **Balanced difficulty**: Include easy examples (model already gets right) and hard ones (model consistently fails)
- **Domain coverage**: Cover the actual distribution your model will face in production
- **Minimal ambiguity**: Remove examples where annotators disagreed — they add noise, not signal

### 2. The SFT Model Matters More Than You Think

If your SFT model can't produce good responses at all, DPO can't fix that. DPO selects between behaviors the model is already capable of. It's a refinement tool, not a capability injection tool.

```
Think of it this way:
- Pretraining teaches the model to SPEAK the language
- SFT teaches the model WHAT to talk about and HOW to format responses
- DPO teaches the model WHICH way to respond is PREFERRED by humans

You can't skip levels. A bad SFT model → a bad DPO model.
```

### 3. On-Policy Data is King

Responses generated by your own model make better training data than responses from other models. The distribution match matters deeply for DPO because the loss computes log-probability ratios — if the responses are out-of-distribution for your model, the ratios become noisy.

```python
# Best practice workflow:
# 1. Train SFT model
# 2. Generate responses from YOUR SFT model (not GPT-4 responses)
# 3. Rank those responses using a judge (GPT-4 is fine as judge)
# 4. Train DPO on those ranked responses
# 5. Optionally: repeat steps 2-4 with your DPO model for iterative improvement
```

### 4. Length Bias is Real

Models trained with DPO often develop a length bias — either becoming verbose (if longer responses are consistently chosen) or terse (if shorter ones are).

```python
# When creating data, ensure length isn't a confound
# Include examples where:
# - Short chosen, long rejected (quality > length)
# - Long chosen, short rejected (completeness matters)
# - Similar length, different quality (pure quality signal)

# You can also explicitly check for bias in your dataset:
import numpy as np

chosen_lengths = [len(ex["chosen"].split()) for ex in dataset]
rejected_lengths = [len(ex["rejected"].split()) for ex in dataset]

print(f"Chosen mean length: {np.mean(chosen_lengths):.0f} words")
print(f"Rejected mean length: {np.mean(rejected_lengths):.0f} words")
print(f"Correlation between 'being chosen' and length: "
      f"{np.corrcoef([1]*len(chosen_lengths) + [0]*len(rejected_lengths), "
      f"chosen_lengths + rejected_lengths)[0,1]:.3f}")
```

### 5. Multi-Turn DPO

For chatbots, you often want to optimize multi-turn conversations. Format your data accordingly:

```python
# Multi-turn preference data
example = {
    "prompt": [
        {"role": "user", "content": "I'm feeling anxious about my job interview tomorrow."},
        {"role": "assistant", "content": "That's completely natural. What specific aspects are making you anxious?"},
        {"role": "user", "content": "I'm worried I'll blank on technical questions."},
    ],
    "chosen": "That's a really common fear, and there are concrete things you can do about it. First, review the most common technical questions for your role — most interviews draw from a predictable pool. Second, practice thinking out loud: interviewers care more about your reasoning process than getting the perfect answer. Third, prepare a fallback phrase for when you're stuck: 'Let me think about this step by step.' This buys you time and shows structured thinking. Would you like to do a mock technical question right now?",
    "rejected": "You shouldn't worry about that. Just be yourself and you'll be fine! Everyone gets nervous before interviews. I'm sure you'll do great!",
}
```

### 6. Iterative DPO (How the Best Models Are Actually Trained)

One round of DPO is good. Multiple rounds, done carefully, produce the best results. This is how Llama 3, Tulu 3, and most top-performing open models are actually trained:

```
Round 1: SFT model → Generate responses → Judge → DPO → Model v1
Round 2: Generate from Model v1 → Judge → DPO (from v1 as ref) → Model v2
Round 3: Generate from Model v2 → Judge → DPO (from v2 as ref) → Model v3
```

Key principles for iterative DPO:
- Each round uses **on-policy data** from the latest model
- The reference model for round N is the output of round N-1
- Each round typically uses **fewer examples** but **higher quality** (the model is already better)
- Monitor for reward hacking — if the model starts producing formulaic responses, your judge may be rewarding surface patterns

### 7. Memory-Efficient Training Strategies

```python
# Strategy 1: QLoRA (most common for single-GPU)
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# Strategy 2: Unsloth (best single-GPU efficiency)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Strategy 3: DeepSpeed ZeRO-3 (for multi-GPU)
# accelerate launch --config_file deepspeed_z3.yaml train_dpo.py

# Strategy 4: FSDP (alternative multi-GPU)
training_args = DPOConfig(
    fsdp="full_shard auto_wrap",
    fsdp_config={
        "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
    },
)
```

**DeepSpeed ZeRO-3 config for DPO:**

```json
{
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu"},
        "offload_param": {"device": "none"},
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_scatter": true
    },
    "gradient_accumulation_steps": 8,
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto"
}
```

### 8. Choosing the Right Optimizer

```python
# Standard Adam (most memory-hungry)
optim="adamw_torch"  # 8 bytes per parameter for optimizer states

# 8-bit Adam (recommended default)
optim="adamw_bnb_8bit"  # 4 bytes per parameter — 2x savings
# pip install bitsandbytes

# Paged 8-bit Adam (for extreme memory constraints)
optim="paged_adamw_8bit"  # Offloads optimizer states to CPU when needed
# Slower but prevents OOM on large models

# Adafactor (no momentum, less memory)
optim="adafactor"  # ~4 bytes per parameter, but can be less stable

# For Unsloth specifically:
optim="adamw_8bit"  # Unsloth's optimized 8-bit Adam
```

## Evaluation: How Do You Know It Worked?

### Automated Evaluation

```python
import random

def compute_win_rate(sft_model, dpo_model, eval_prompts, judge_model="gpt-4o"):
    """Compute DPO model's win rate against SFT baseline."""
    wins, ties, losses = 0, 0, 0
    total = 0

    for prompt in eval_prompts:
        sft_response = generate(sft_model, prompt, temperature=0.7)
        dpo_response = generate(dpo_model, prompt, temperature=0.7)

        # Randomize order to avoid position bias in the judge
        if random.random() > 0.5:
            a, b = sft_response, dpo_response
            dpo_is_a = True
        else:
            a, b = dpo_response, sft_response
            dpo_is_a = False

        winner = judge_with_explanation(prompt, a, b, model=judge_model)

        if winner == "tie":
            ties += 1
        elif (winner == "A" and dpo_is_a) or (winner == "B" and not dpo_is_a):
            wins += 1
        else:
            losses += 1
        total += 1

    win_rate = wins / total
    print(f"Win: {wins} | Tie: {ties} | Loss: {losses} | Total: {total}")
    print(f"Win rate: {win_rate:.1%}")
    print(f"Win rate (excluding ties): {wins/(wins+losses):.1%}")
    return win_rate

# Target: >55% win rate against SFT model
# Great: >65% win rate
# Excellent: >70% win rate
# If <50%: DPO made things worse — check data quality
```

### Benchmarks

| Benchmark | What It Measures | Tool | Notes |
|-----------|-----------------|------|-------|
| AlpacaEval 2.0 | General instruction following | `alpaca_eval` | Length-controlled version is more reliable |
| MT-Bench | Multi-turn conversation quality | `FastChat` | 80 prompts, GPT-4 judged |
| Arena-Hard | Head-to-head comparison | `arena-hard-auto` | Based on Chatbot Arena prompts |
| IFEval | Instruction following precision | `lm-evaluation-harness` | Tests format compliance |
| TruthfulQA | Factuality/hallucination | `lm-evaluation-harness` | Tests factual grounding |
| WildBench | Real-world user prompts | `allenai/WildBench` | More diverse than MT-Bench |

```bash
# Run IFEval with lm-evaluation-harness
lm_eval --model hf \
    --model_args pretrained=./dpo-llama-8b-merged \
    --tasks ifeval \
    --output_path ./eval_results \
    --batch_size 4

# Run MT-Bench with FastChat
python gen_model_answer.py \
    --model-path ./dpo-llama-8b-merged \
    --model-id my-dpo-model

python gen_judgment.py \
    --model-list my-dpo-model \
    --judge-model gpt-4o
```

### Human Evaluation

For production systems, there's no substitute for human evaluation:

```
1. Blind A/B test: Show users responses from SFT and DPO models
   - Randomize which model is A vs B
   - Show both responses side by side
   - Ask: "Which response do you prefer?" + optional text reason

2. Track metrics:
   - Preference rate (% choosing DPO model)
   - Task completion rate (did the response actually help?)
   - User satisfaction (1-5 scale)
   - Safety violations (count of harmful/inappropriate responses)

3. Sample size: 200-500 comparisons for statistical significance
   - Use bootstrapping to compute confidence intervals
   - Target p < 0.05 for significance

4. Quality control:
   - Inter-annotator agreement (Cohen's kappa > 0.6)
   - Include attention check questions
   - Pay annotators fairly (quality drops with cheap annotation)
```

## Complete End-to-End Example (Production-Ready)

Here's a complete script that works end-to-end with best practices:

```python
"""
Production-ready DPO training script with Unsloth.
Run on a single GPU (16GB+ VRAM) or Google Colab.

Requirements:
    pip install unsloth trl transformers datasets wandb

Usage:
    python train_dpo.py
"""

import torch
from unsloth import FastLanguageModel
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
import numpy as np

# ============================================================
# Configuration — change these for your use case
# ============================================================
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
DATASET_NAME = "argilla/dpo-mix-7k"
OUTPUT_DIR = "./dpo-llama-8b-production"
MAX_SEQ_LENGTH = 1024
BETA = 0.1
LEARNING_RATE = 5e-7
NUM_EPOCHS = 1
LORA_R = 16
LORA_ALPHA = 32

# ============================================================
# 1. Load model with Unsloth optimizations
# ============================================================
print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

# ============================================================
# 2. Apply LoRA
# ============================================================
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# Print trainable parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# ============================================================
# 3. Load and validate preference data
# ============================================================
print("Loading dataset...")
dataset = load_dataset(DATASET_NAME, split="train")

def format_for_dpo(example):
    messages = [{"role": "user", "content": example["prompt"]}]
    return {
        "prompt": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        ),
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }

dataset = dataset.map(format_for_dpo)

# Filter out too-long examples
def length_ok(example):
    prompt_len = len(tokenizer.encode(example["prompt"]))
    chosen_len = len(tokenizer.encode(example["chosen"]))
    rejected_len = len(tokenizer.encode(example["rejected"]))
    return (prompt_len + max(chosen_len, rejected_len)) <= MAX_SEQ_LENGTH

original_size = len(dataset)
dataset = dataset.filter(length_ok)
print(f"Filtered: {original_size} → {len(dataset)} examples (removed {original_size - len(dataset)} too-long)")

# Quick data quality check
chosen_lens = [len(ex["chosen"].split()) for ex in dataset]
rejected_lens = [len(ex["rejected"].split()) for ex in dataset]
print(f"Avg chosen length: {np.mean(chosen_lens):.0f} words")
print(f"Avg rejected length: {np.mean(rejected_lens):.0f} words")
print(f"Length ratio: {np.mean(chosen_lens)/np.mean(rejected_lens):.2f}x")

# Train/eval split
dataset = dataset.train_test_split(test_size=0.02, seed=42)

# ============================================================
# 4. Configure and run DPO training
# ============================================================
training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    beta=BETA,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    num_train_epochs=NUM_EPOCHS,
    max_length=MAX_SEQ_LENGTH,
    max_prompt_length=MAX_SEQ_LENGTH // 2,
    bf16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    eval_strategy="steps",
    eval_steps=200,
    load_best_model_at_end=True,  # Keep the best checkpoint
    metric_for_best_model="eval_loss",
    optim="adamw_8bit",
    gradient_checkpointing=True,
    remove_unused_columns=False,
    seed=42,
    report_to="wandb",  # Set to "none" to disable
)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
)

print(f"\nStarting DPO training...")
print(f"  Dataset: {len(dataset['train'])} train, {len(dataset['test'])} eval")
print(f"  Beta: {BETA}")
print(f"  LR: {LEARNING_RATE}")
print(f"  Effective batch size: {2 * 4} = 8")

trainer.train()
print("Training complete!")

# ============================================================
# 5. Save in multiple formats
# ============================================================
print("\nSaving models...")

# LoRA adapters (for further training or lightweight deployment)
model.save_pretrained(f"{OUTPUT_DIR}/lora")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora")

# Merged 16-bit (for vLLM / TGI deployment)
model.save_pretrained_merged(
    f"{OUTPUT_DIR}/merged-16bit",
    tokenizer,
    save_method="merged_16bit",
)

# GGUF for llama.cpp / Ollama
model.save_pretrained_gguf(
    f"{OUTPUT_DIR}/gguf-q4km",
    tokenizer,
    quantization_method="q4_k_m",
)

print(f"All models saved to {OUTPUT_DIR}/")

# ============================================================
# 6. Quick inference test
# ============================================================
print("\n=== Inference Test ===")
FastLanguageModel.for_inference(model)

test_prompts = [
    "Explain the difference between a compiler and an interpreter.",
    "What should I consider before adopting a dog?",
    "Write a Python function to check if a string is a palindrome.",
]

for prompt in test_prompts:
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    print(f"\nQ: {prompt}")
    print(f"A: {response[:300]}...")
    print("-" * 60)
```

## Deployment After DPO Training

### Option 1: vLLM (Recommended for Production)

```bash
pip install vllm

# Serve the merged model
python -m vllm.entrypoints.openai.api_server \
    --model ./dpo-llama-8b-production/merged-16bit \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 4096
```

```python
# Client
import openai
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
response = client.chat.completions.create(
    model="./dpo-llama-8b-production/merged-16bit",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### Option 2: Ollama (Easiest for Local Use)

```bash
# Create a Modelfile
cat > Modelfile << 'EOF'
FROM ./dpo-llama-8b-production/gguf-q4km/unsloth.Q4_K_M.gguf
TEMPLATE """{{ .System }}
{{ .Prompt }}"""
PARAMETER temperature 0.7
PARAMETER top_p 0.9
EOF

ollama create my-dpo-model -f Modelfile
ollama run my-dpo-model
```

### Option 3: Hugging Face TGI (Docker)

```bash
docker run --gpus all \
    -v ./dpo-llama-8b-production/merged-16bit:/model \
    -p 8080:80 \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id /model \
    --max-input-length 2048 \
    --max-total-tokens 4096
```

## Common Failure Modes and How to Debug

### The Model Gets Worse After DPO

This happens more often than people admit. Common causes:

1. **Bad SFT baseline**: The SFT model isn't good enough. DPO can't create capabilities, only select between existing ones. **Fix**: Improve SFT quality first. Use a better base model or more SFT data.
2. **Noisy preference data**: If your chosen/rejected labels are inconsistent, the model learns noise. **Fix**: Clean your data. Remove ambiguous pairs. Use IPO for robust training.
3. **Distribution mismatch**: Preference data comes from a different model than your SFT model. **Fix**: Generate on-policy data from your SFT model.
4. **Too many epochs**: DPO overfits quickly. **Fix**: Use 1 epoch. Enable early stopping.
5. **Wrong $\beta$**: Too low → model diverges. Too high → model doesn't learn. **Fix**: Start at 0.1, then tune based on reward margin.

### The Model Becomes Repetitive

Usually caused by:
- Training too long (reduce epochs)
- Learning rate too high (reduce by 5-10x)
- Low $\beta$ causing the model to move too far from the reference

**Diagnostic**: Generate 10 responses to the same prompt with temperature=0.8. If they're all nearly identical, the model has mode-collapsed.

### The Model Becomes Verbose or Terse

Length bias in the data. Check if your chosen responses are systematically longer or shorter than rejected ones.

**Fix**: Add length-balanced examples. Or use SimPO which explicitly normalizes by length.

### Reward Hacking (Subtle)

Even without an explicit reward model, DPO can exhibit reward hacking. The model learns surface patterns that correlate with being chosen:
- Using more hedge words ("I think", "It's important to note")
- Adding unnecessary structure (bullet points, numbered lists)
- Starting with "Great question!"
- Being excessively agreeable

**Fix**: Diversify your preference data so these surface patterns don't consistently correlate with being chosen. Include examples where terse, direct responses are preferred over verbose hedged ones.

### The Model Refuses Too Much or Too Little

DPO can shift the refusal boundary. If your "rejected" examples include many unsafe responses, the model learns to be very cautious — sometimes refusing benign queries. Conversely, if you never include safety-relevant examples, the model may become less safe.

**Fix**: Include both types in your data:
- Examples where the chosen response helpfully answers and the rejected response unnecessarily refuses
- Examples where the chosen response appropriately refuses and the rejected response complies with a harmful request

## When to Use DPO vs Alternatives

| Scenario | Recommendation | Why |
|----------|---------------|-----|
| You have paired preferences | **DPO** | Standard, well-understood, stable |
| You only have thumbs up/down | **KTO** | Doesn't need paired data |
| You want SFT + alignment in one step | **ORPO** | Saves one training stage |
| Your preference data is noisy | **IPO** | Robust to label noise |
| You need maximum control and quality | **RLHF with PPO** | Higher ceiling, but much harder |
| You're doing iterative alignment | **Online DPO** | On-policy data generation |
| Memory constrained (single GPU) | **SimPO** + Unsloth | No reference model needed |
| You want reasoning/math improvement | **GRPO** | Group-based RL, used for DeepSeek R1 |

## Conclusion

DPO is the workhorse of modern LLM alignment. It's simpler than RLHF, more stable, and in many cases achieves comparable results. The key lessons:

1. **Data quality is everything.** Invest more in preference data curation than in hyperparameter tuning. 5K clean pairs > 100K noisy ones.
2. **SFT first.** DPO works best on a solid SFT foundation. Don't skip this step.
3. **Start with the defaults:** $\beta = 0.1$, learning rate $= 5 \times 10^{-7}$, 1 epoch, LoRA with $r = 16$.
4. **Choose the right framework.** Unsloth for single-GPU speed, Axolotl for multi-GPU configs, TRL for maximum control, LLaMA-Factory for no-code.
5. **Monitor the reward margin.** It should increase steadily without saturating.
6. **On-policy data wins.** Generate from your own model and rank the results.
7. **Iterate.** The best models do multiple rounds of generation → ranking → DPO.
8. **Evaluate seriously.** Win rate against the SFT baseline with a strong judge + human evaluation for production.

## References

- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) — Rafailov et al., 2023 (the original DPO paper)
- [A General Theoretical Paradigm to Understand Learning from Human Feedback](https://arxiv.org/abs/2310.12036) — Azar et al., 2023 (IPO)
- [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306) — Ethayarajh et al., 2024
- [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691) — Hong et al., 2024
- [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734) — Meng et al., 2024
- [Self-Play Preference Optimization for Language Model Alignment](https://arxiv.org/abs/2405.00675) — Wu et al., 2024
- [Zephyr: Direct Distillation of LM Alignment](https://arxiv.org/abs/2310.16944) — Tunstall et al., 2023
- [Tulu 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.15124) — Ivison et al., 2024
- [TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) — Hugging Face
- [Unsloth](https://github.com/unslothai/unsloth) — Daniel & Michael Han
- [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) — Axolotl AI
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) — Yaowei Zheng et al.
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — OpenRLHF Team
